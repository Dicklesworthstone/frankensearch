//! Data source trait and mock implementation.
//!
//! The [`DataSource`] trait decouples the TUI from the concrete data backend.
//! Product screens read from a `DataSource`; the real implementation queries
//! `FrankenSQLite` (wired in via bd-2yu.4.3), while [`MockDataSource`] provides
//! test data for development and testing.

use std::collections::HashMap;
use std::io;
use std::path::Path;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::discovery::StaticDiscoverySource;
use crate::discovery::{DiscoveryConfig, DiscoveryEngine, DiscoverySignalKind, DiscoverySource};
use crate::state::{
    ControlPlaneMetrics, FleetSnapshot, InstanceAttribution, InstanceInfo, InstanceLifecycle,
    LifecycleSignal, LifecycleTrackerConfig, ProjectAttributionResolver, ProjectLifecycleTracker,
    ResourceMetrics, SearchMetrics,
};
use crate::storage::{ResourceTrendPoint, SearchSummarySnapshot, SummaryWindow};
use crate::{DiscoveredInstance, DiscoveryStatus, InstanceSighting, OpsStorage};
use fsqlite::Row;
use fsqlite_types::value::SqliteValue;
use tracing::warn;

// ─── Time Window ─────────────────────────────────────────────────────────────

/// Time window for metric queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeWindow {
    /// Last 1 minute.
    OneMinute,
    /// Last 15 minutes.
    FifteenMinutes,
    /// Last 1 hour.
    OneHour,
    /// Last 6 hours.
    SixHours,
    /// Last 24 hours.
    TwentyFourHours,
    /// Last 3 days.
    ThreeDays,
    /// Last 1 week.
    OneWeek,
}

impl TimeWindow {
    /// All windows in ascending order.
    pub const ALL: &'static [Self] = &[
        Self::OneMinute,
        Self::FifteenMinutes,
        Self::OneHour,
        Self::SixHours,
        Self::TwentyFourHours,
        Self::ThreeDays,
        Self::OneWeek,
    ];

    /// Human-readable label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::OneMinute => "1m",
            Self::FifteenMinutes => "15m",
            Self::OneHour => "1h",
            Self::SixHours => "6h",
            Self::TwentyFourHours => "24h",
            Self::ThreeDays => "3d",
            Self::OneWeek => "1w",
        }
    }

    /// Duration in seconds.
    #[must_use]
    pub const fn seconds(self) -> u64 {
        match self {
            Self::OneMinute => 60,
            Self::FifteenMinutes => 15 * 60,
            Self::OneHour => 3600,
            Self::SixHours => 6 * 3600,
            Self::TwentyFourHours => 24 * 3600,
            Self::ThreeDays => 3 * 24 * 3600,
            Self::OneWeek => 7 * 24 * 3600,
        }
    }
}

impl std::fmt::Display for TimeWindow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

// ─── Data Source Trait ────────────────────────────────────────────────────────

/// Trait for data backends that feed the ops TUI.
///
/// Synchronous interface for the render loop. Background async tasks
/// populate the backing store; the `DataSource` provides read access.
///
/// The real implementation queries `FrankenSQLite` (bd-2yu.4.3).
/// [`MockDataSource`] provides synthetic data for development.
pub trait DataSource {
    /// Get the current fleet snapshot.
    fn fleet_snapshot(&self) -> FleetSnapshot;

    /// Get search metrics for a given time window.
    fn search_metrics(&self, instance_id: &str, window: TimeWindow) -> Option<SearchMetrics>;

    /// Get resource metrics for a given instance.
    fn resource_metrics(&self, instance_id: &str) -> Option<ResourceMetrics>;

    /// Get control-plane self-monitoring metrics.
    fn control_plane_metrics(&self) -> ControlPlaneMetrics;

    /// Attribution metadata for an instance.
    fn attribution(&self, instance_id: &str) -> Option<InstanceAttribution>;

    /// Lifecycle snapshot for an instance.
    fn lifecycle(&self, instance_id: &str) -> Option<InstanceLifecycle>;
}

const DEFAULT_STORAGE_LIMIT_BYTES: u64 = 512 * 1024 * 1024;
const DEFAULT_RSS_LIMIT_BYTES: u64 = 1024 * 1024 * 1024;
const DISCOVERY_INSTANCE_KEY_PREFIX: &str = "iid:";
const DISCOVERY_IDENTITY_PREFIX: &str = "instance:iid:";
const TIMESTAMP_WRAP_MODULUS_MS: u64 = 1_u64 << 32;

#[derive(Debug, Clone, Default)]
struct StorageDataCache {
    snapshot: FleetSnapshot,
    control_plane: ControlPlaneMetrics,
    instance_project_keys: HashMap<String, String>,
}

#[derive(Debug, Clone, Copy, Default)]
struct ThroughputTracker {
    last_ts_ms: u64,
    last_inserted_total: u64,
}

#[derive(Debug, Clone)]
struct StorageInstanceRow {
    project_key: String,
    instance_id: String,
    host_name: Option<String>,
    pid: Option<u32>,
    version: Option<String>,
    last_heartbeat_ms: u64,
}

/// Storage-backed data source used by the production ops runtime.
pub struct StorageDataSource {
    storage: OpsStorage,
    discovery_engine: Mutex<DiscoveryEngine>,
    lifecycle_tracker: Mutex<ProjectLifecycleTracker>,
    cache: Mutex<StorageDataCache>,
    throughput_tracker: Mutex<ThroughputTracker>,
    storage_limit_bytes: u64,
    rss_limit_bytes: u64,
}

impl StorageDataSource {
    #[must_use]
    pub fn new(storage: OpsStorage) -> Self {
        Self {
            storage,
            discovery_engine: Mutex::new(DiscoveryEngine::new(DiscoveryConfig::default())),
            lifecycle_tracker: Mutex::new(ProjectLifecycleTracker::new(
                LifecycleTrackerConfig::default(),
            )),
            cache: Mutex::new(StorageDataCache::default()),
            throughput_tracker: Mutex::new(ThroughputTracker::default()),
            storage_limit_bytes: DEFAULT_STORAGE_LIMIT_BYTES,
            rss_limit_bytes: DEFAULT_RSS_LIMIT_BYTES,
        }
    }

    fn refresh(&self) -> (FleetSnapshot, ControlPlaneMetrics) {
        match self.compute_refresh() {
            Ok((snapshot, control_plane, instance_project_keys)) => {
                let mut cache = self.cache.lock().unwrap_or_else(|poisoned| {
                    warn!("storage data-source cache lock poisoned; recovering");
                    poisoned.into_inner()
                });
                cache.snapshot = snapshot.clone();
                cache.control_plane = control_plane.clone();
                cache.instance_project_keys = instance_project_keys;
                drop(cache);
                (snapshot, control_plane)
            }
            Err(error) => {
                warn!(error = %error, "storage data-source refresh failed; using cached snapshot");
                let cache = self.cache.lock().unwrap_or_else(|poisoned| {
                    warn!("storage data-source cache lock poisoned; recovering");
                    poisoned.into_inner()
                });
                (cache.snapshot.clone(), cache.control_plane.clone())
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    fn compute_refresh(
        &self,
    ) -> frankensearch_core::SearchResult<(
        FleetSnapshot,
        ControlPlaneMetrics,
        HashMap<String, String>,
    )> {
        let now_ms = unix_timestamp_ms_u64()?;
        let mut rows = self.load_instance_rows()?;
        for row in &mut rows {
            row.last_heartbeat_ms = normalize_storage_timestamp_ms(row.last_heartbeat_ms, now_ms);
        }
        let mut discovery_source = StaticDiscoverySource::new(
            rows.iter()
                .map(|row| InstanceSighting {
                    source: DiscoverySignalKind::Heartbeat,
                    observed_at_ms: row.last_heartbeat_ms,
                    project_key_hint: Some(row.project_key.clone()),
                    host_name: row.host_name.clone(),
                    pid: row.pid,
                    instance_key_hint: Some(format!(
                        "{DISCOVERY_INSTANCE_KEY_PREFIX}{}",
                        row.instance_id
                    )),
                    control_endpoint: None,
                    socket_path: None,
                    heartbeat_path: None,
                    version: row.version.clone(),
                })
                .collect(),
        );

        let discovery_started = std::time::Instant::now();
        let mut discovery_engine = self.discovery_engine.lock().unwrap_or_else(|poisoned| {
            warn!("storage data-source discovery lock poisoned; recovering");
            poisoned.into_inner()
        });
        let mut sources: [&mut dyn DiscoverySource; 1] = [&mut discovery_source];
        discovery_engine.poll(now_ms, &mut sources);
        let discovered = discovery_engine.snapshot();
        let discovery_latency_ms =
            u64::try_from(discovery_started.elapsed().as_millis()).unwrap_or(u64::MAX);
        drop(discovery_engine);

        let normalized_discovered: Vec<DiscoveredInstance> = discovered
            .into_iter()
            .filter_map(|mut instance| {
                let instance_id = instance_id_from_discovery_identity(&instance)?;
                instance.instance_id = instance_id;
                Some(instance)
            })
            .collect();
        let mut normalized_discovered = normalized_discovered;
        if normalized_discovered.is_empty() && !rows.is_empty() {
            warn!(
                row_count = rows.len(),
                "discovery snapshot had no parseable instance ids; falling back to storage rows"
            );
            normalized_discovered = discovered_from_storage_rows(&rows, now_ms);
        }

        let mut lifecycle_tracker = self.lifecycle_tracker.lock().unwrap_or_else(|poisoned| {
            warn!("storage data-source lifecycle lock poisoned; recovering");
            poisoned.into_inner()
        });
        let lifecycle_events = lifecycle_tracker.ingest_discovery(now_ms, &normalized_discovered);
        let attribution = lifecycle_tracker.attribution_snapshot();
        let lifecycle = lifecycle_tracker.lifecycle_snapshot();
        drop(lifecycle_tracker);

        let mut snapshot = FleetSnapshot {
            attribution,
            lifecycle,
            lifecycle_events,
            ..FleetSnapshot::default()
        };
        let mut instance_project_keys = HashMap::new();

        for instance in normalized_discovered {
            let instance_id = instance.instance_id.clone();
            let project_key = instance
                .project_key_hint
                .clone()
                .unwrap_or_else(|| "unknown".to_owned());
            instance_project_keys.insert(instance_id.clone(), project_key.clone());

            let search_summary = self
                .storage
                .latest_search_summary(&project_key, &instance_id, SummaryWindow::OneHour)
                .ok()
                .flatten();
            if let Some(search_summary) = search_summary.as_ref() {
                snapshot.search_metrics.insert(
                    instance_id.clone(),
                    search_metrics_from_summary(search_summary),
                );
            }

            let latest_resource = self
                .latest_resource_point(&project_key, &instance_id, now_ms)
                .ok()
                .flatten();
            if let Some(resource) = latest_resource.as_ref() {
                snapshot
                    .resources
                    .insert(instance_id.clone(), resource_metrics_from_point(resource));
            }

            let doc_count = self
                .latest_doc_count(&project_key, &instance_id)
                .ok()
                .flatten()
                .unwrap_or_default();
            let pending_jobs = latest_resource
                .as_ref()
                .and_then(|point| point.queue_depth)
                .unwrap_or_default();
            let project = snapshot.attribution.get(&instance_id).map_or_else(
                || project_key.clone(),
                |value| value.resolved_project.clone(),
            );

            snapshot.instances.push(InstanceInfo {
                id: instance_id,
                project,
                pid: instance.pid,
                healthy: instance.status == DiscoveryStatus::Active,
                doc_count,
                pending_jobs,
            });
        }

        snapshot
            .instances
            .sort_unstable_by(|left, right| left.id.cmp(&right.id));

        let control_plane = self.build_control_plane_metrics(now_ms, discovery_latency_ms);
        Ok((snapshot, control_plane, instance_project_keys))
    }

    fn build_control_plane_metrics(
        &self,
        now_ms: u64,
        discovery_latency_ms: u64,
    ) -> ControlPlaneMetrics {
        let ingestion = self.storage.ingestion_metrics();
        let storage_bytes = storage_file_size_bytes(self.storage.config().db_path.as_path());
        #[allow(clippy::cast_precision_loss)]
        let throughput_eps = {
            let mut tracker = self.throughput_tracker.lock().unwrap_or_else(|poisoned| {
                warn!("storage data-source throughput lock poisoned; recovering");
                poisoned.into_inner()
            });
            let total_inserted = ingestion.total_inserted;
            let eps = if tracker.last_ts_ms > 0 && now_ms > tracker.last_ts_ms {
                let elapsed_secs = (now_ms.saturating_sub(tracker.last_ts_ms)) as f64 / 1000.0;
                (total_inserted.saturating_sub(tracker.last_inserted_total)) as f64 / elapsed_secs
            } else {
                0.0
            };
            tracker.last_ts_ms = now_ms;
            tracker.last_inserted_total = total_inserted;
            eps
        };

        ControlPlaneMetrics {
            ingestion_lag_events: u64::try_from(ingestion.pending_events).unwrap_or(u64::MAX),
            storage_bytes,
            storage_limit_bytes: self.storage_limit_bytes,
            frame_time_ms: 16.0,
            discovery_latency_ms,
            event_throughput_eps: throughput_eps,
            rss_bytes: 0,
            rss_limit_bytes: self.rss_limit_bytes,
            dead_letter_events: ingestion
                .total_failed_records
                .saturating_add(ingestion.total_backpressured_batches),
        }
    }

    fn load_instance_rows(&self) -> frankensearch_core::SearchResult<Vec<StorageInstanceRow>> {
        let rows = self
            .storage
            .connection()
            .query(
                "SELECT project_key, instance_id, host_name, pid, version, last_heartbeat_ms \
                 FROM instances \
                 ORDER BY instance_id ASC;",
            )
            .map_err(data_source_error)?;
        rows.iter().map(parse_instance_row).collect()
    }

    fn latest_resource_point(
        &self,
        project_key: &str,
        instance_id: &str,
        now_ms: u64,
    ) -> frankensearch_core::SearchResult<Option<ResourceTrendPoint>> {
        let rows = self
            .storage
            .connection()
            .query_with_params(
                "SELECT cpu_pct, rss_bytes, io_read_bytes, io_write_bytes, queue_depth, ts_ms \
                 FROM resource_samples \
                 WHERE project_key = ?1 AND instance_id = ?2 \
                 ORDER BY rowid DESC LIMIT 1;",
                &[
                    SqliteValue::Text(project_key.to_owned().into()),
                    SqliteValue::Text(instance_id.to_owned().into()),
                ],
            )
            .map_err(data_source_error)?;
        let Some(row) = rows.first() else {
            return Ok(None);
        };
        let raw_ts = i64_to_u64_non_negative(
            row_i64(row, 5, "resource_samples.ts_ms")?,
            "resource_samples.ts_ms",
        )?;
        let normalized_ts = normalize_storage_timestamp_ms(raw_ts, now_ms);
        let ts_ms = i64::try_from(normalized_ts).unwrap_or(i64::MAX);

        Ok(Some(ResourceTrendPoint {
            ts_ms,
            cpu_pct: row_opt_f64(row, 0, "resource_samples.cpu_pct")?,
            rss_bytes: opt_i64_to_u64_non_negative(
                row_opt_i64(row, 1, "resource_samples.rss_bytes")?,
                "resource_samples.rss_bytes",
            )?,
            io_read_bytes: opt_i64_to_u64_non_negative(
                row_opt_i64(row, 2, "resource_samples.io_read_bytes")?,
                "resource_samples.io_read_bytes",
            )?,
            io_write_bytes: opt_i64_to_u64_non_negative(
                row_opt_i64(row, 3, "resource_samples.io_write_bytes")?,
                "resource_samples.io_write_bytes",
            )?,
            queue_depth: opt_i64_to_u64_non_negative(
                row_opt_i64(row, 4, "resource_samples.queue_depth")?,
                "resource_samples.queue_depth",
            )?,
        }))
    }

    fn latest_doc_count(
        &self,
        project_key: &str,
        instance_id: &str,
    ) -> frankensearch_core::SearchResult<Option<u64>> {
        let rows = self
            .storage
            .connection()
            .query_with_params(
                "SELECT record_count FROM index_inventory_snapshots \
                 WHERE project_key = ?1 AND instance_id = ?2 \
                 ORDER BY ts_ms DESC LIMIT 1;",
                &[
                    SqliteValue::Text(project_key.to_owned().into()),
                    SqliteValue::Text(instance_id.to_owned().into()),
                ],
            )
            .map_err(data_source_error)?;
        let Some(row) = rows.first() else {
            return Ok(None);
        };
        let raw = row_i64(row, 0, "index_inventory_snapshots.record_count")?;
        Ok(Some(i64_to_u64_non_negative(
            raw,
            "index_inventory_snapshots.record_count",
        )?))
    }

    fn project_key_for_instance(&self, instance_id: &str) -> Option<String> {
        let cache = self.cache.lock().unwrap_or_else(|poisoned| {
            warn!("storage data-source cache lock poisoned; recovering");
            poisoned.into_inner()
        });
        cache.instance_project_keys.get(instance_id).cloned()
    }
}

impl DataSource for StorageDataSource {
    fn fleet_snapshot(&self) -> FleetSnapshot {
        self.refresh().0
    }

    fn search_metrics(&self, instance_id: &str, window: TimeWindow) -> Option<SearchMetrics> {
        let _ = self.refresh();
        let project_key = self.project_key_for_instance(instance_id)?;
        let summary = self
            .storage
            .latest_search_summary(
                &project_key,
                instance_id,
                summary_window_from_time_window(window),
            )
            .ok()
            .flatten()?;
        Some(search_metrics_from_summary(&summary))
    }

    fn resource_metrics(&self, instance_id: &str) -> Option<ResourceMetrics> {
        let (snapshot, _) = self.refresh();
        snapshot.resources.get(instance_id).cloned()
    }

    fn control_plane_metrics(&self) -> ControlPlaneMetrics {
        self.refresh().1
    }

    fn attribution(&self, instance_id: &str) -> Option<InstanceAttribution> {
        let (snapshot, _) = self.refresh();
        snapshot.attribution.get(instance_id).cloned()
    }

    fn lifecycle(&self, instance_id: &str) -> Option<InstanceLifecycle> {
        let (snapshot, _) = self.refresh();
        snapshot.lifecycle.get(instance_id).cloned()
    }
}

fn search_metrics_from_summary(summary: &SearchSummarySnapshot) -> SearchMetrics {
    let avg_latency_us = summary
        .p50_latency_us
        .or(summary.p95_latency_us)
        .or(summary.p99_latency_us)
        .unwrap_or_default();
    SearchMetrics {
        total_searches: summary.search_count,
        avg_latency_us,
        p95_latency_us: summary.p95_latency_us.unwrap_or(avg_latency_us),
        refined_count: 0,
    }
}

fn resource_metrics_from_point(point: &ResourceTrendPoint) -> ResourceMetrics {
    ResourceMetrics {
        cpu_percent: point.cpu_pct.unwrap_or_default(),
        memory_bytes: point.rss_bytes.unwrap_or_default(),
        io_read_bytes: point.io_read_bytes.unwrap_or_default(),
        io_write_bytes: point.io_write_bytes.unwrap_or_default(),
    }
}

const fn summary_window_from_time_window(window: TimeWindow) -> SummaryWindow {
    match window {
        TimeWindow::OneMinute => SummaryWindow::OneMinute,
        TimeWindow::FifteenMinutes => SummaryWindow::FifteenMinutes,
        TimeWindow::OneHour => SummaryWindow::OneHour,
        TimeWindow::SixHours => SummaryWindow::SixHours,
        TimeWindow::TwentyFourHours => SummaryWindow::TwentyFourHours,
        TimeWindow::ThreeDays => SummaryWindow::ThreeDays,
        TimeWindow::OneWeek => SummaryWindow::OneWeek,
    }
}

fn instance_id_from_discovery_identity(instance: &DiscoveredInstance) -> Option<String> {
    instance.identity_keys.iter().find_map(|key| {
        key.strip_prefix(DISCOVERY_IDENTITY_PREFIX)
            .map(ToOwned::to_owned)
    })
}

fn normalize_storage_timestamp_ms(raw_ms: u64, now_ms: u64) -> u64 {
    if raw_ms >= 100_000_000_000 {
        return raw_ms;
    }
    let modulus = i128::from(TIMESTAMP_WRAP_MODULUS_MS);
    let raw = i128::from(raw_ms);
    let now = i128::from(now_ms);
    let wraps = (now - raw + (modulus / 2)) / modulus;
    let candidate = raw + (wraps * modulus);
    if candidate < 0 {
        raw_ms
    } else {
        u64::try_from(candidate).unwrap_or(raw_ms)
    }
}

fn discovered_from_storage_rows(
    rows: &[StorageInstanceRow],
    now_ms: u64,
) -> Vec<DiscoveredInstance> {
    let stale_after_ms = DiscoveryConfig::default().normalized().stale_after_ms;
    let mut discovered = Vec::with_capacity(rows.len());
    for row in rows {
        let status = if now_ms.saturating_sub(row.last_heartbeat_ms) >= stale_after_ms {
            DiscoveryStatus::Stale
        } else {
            DiscoveryStatus::Active
        };
        discovered.push(DiscoveredInstance {
            instance_id: row.instance_id.clone(),
            project_key_hint: Some(row.project_key.clone()),
            host_name: row.host_name.clone(),
            pid: row.pid,
            version: row.version.clone(),
            first_seen_ms: row.last_heartbeat_ms,
            last_seen_ms: row.last_heartbeat_ms,
            status,
            sources: vec![DiscoverySignalKind::Heartbeat],
            identity_keys: vec![format!(
                "instance:{DISCOVERY_INSTANCE_KEY_PREFIX}{}",
                row.instance_id
            )],
        });
    }
    discovered.sort_unstable_by(|left, right| left.instance_id.cmp(&right.instance_id));
    discovered
}

fn storage_file_size_bytes(path: &Path) -> u64 {
    if path.as_os_str() == ":memory:" {
        return 0;
    }
    let base = path.to_path_buf();
    let wal = Path::new(&format!("{}-wal", base.display())).to_path_buf();
    let shm = Path::new(&format!("{}-shm", base.display())).to_path_buf();
    [base, wal, shm]
        .into_iter()
        .filter_map(|candidate| std::fs::metadata(candidate).ok())
        .map(|metadata| metadata.len())
        .sum()
}

fn parse_instance_row(row: &Row) -> frankensearch_core::SearchResult<StorageInstanceRow> {
    let project_key = row_text(row, 0, "instances.project_key")?.to_owned();
    let instance_id = row_text(row, 1, "instances.instance_id")?.to_owned();
    let host_name = row_opt_text(row, 2, "instances.host_name")?.map(ToOwned::to_owned);
    let pid = row_opt_i64(row, 3, "instances.pid")?
        .map(|value| {
            if value < 0 {
                Err(io::Error::other("instances.pid must be >= 0"))
            } else {
                u32::try_from(value).map_err(io::Error::other)
            }
        })
        .transpose()
        .map_err(|error| frankensearch_core::SearchError::SubsystemError {
            subsystem: "ops-data-source",
            source: Box::new(error),
        })?;
    let version = row_opt_text(row, 4, "instances.version")?.map(ToOwned::to_owned);
    let last_heartbeat_ms = i64_to_u64_non_negative(
        row_i64(row, 5, "instances.last_heartbeat_ms")?,
        "instances.last_heartbeat_ms",
    )?;

    Ok(StorageInstanceRow {
        project_key,
        instance_id,
        host_name,
        pid,
        version,
        last_heartbeat_ms,
    })
}

fn data_source_error(error: impl std::fmt::Display) -> frankensearch_core::SearchError {
    frankensearch_core::SearchError::SubsystemError {
        subsystem: "ops-data-source",
        source: Box::new(io::Error::other(error.to_string())),
    }
}

fn row_i64(row: &Row, index: usize, field: &str) -> frankensearch_core::SearchResult<i64> {
    match row.get(index) {
        Some(SqliteValue::Integer(value)) => Ok(*value),
        Some(other) => Err(frankensearch_core::SearchError::SubsystemError {
            subsystem: "ops-data-source",
            source: Box::new(io::Error::other(format!(
                "unexpected type for {field}: {other:?}"
            ))),
        }),
        None => Err(frankensearch_core::SearchError::SubsystemError {
            subsystem: "ops-data-source",
            source: Box::new(io::Error::other(format!("missing column for {field}"))),
        }),
    }
}

fn row_opt_i64(
    row: &Row,
    index: usize,
    field: &str,
) -> frankensearch_core::SearchResult<Option<i64>> {
    match row.get(index) {
        Some(SqliteValue::Integer(value)) => Ok(Some(*value)),
        Some(SqliteValue::Null) | None => Ok(None),
        Some(other) => Err(frankensearch_core::SearchError::SubsystemError {
            subsystem: "ops-data-source",
            source: Box::new(io::Error::other(format!(
                "unexpected type for {field}: {other:?}"
            ))),
        }),
    }
}

fn row_opt_f64(
    row: &Row,
    index: usize,
    field: &str,
) -> frankensearch_core::SearchResult<Option<f64>> {
    match row.get(index) {
        Some(SqliteValue::Float(value)) => Ok(Some(*value)),
        Some(SqliteValue::Integer(value)) => i64_to_f64_exact(*value, field).map(Some),
        Some(SqliteValue::Null) | None => Ok(None),
        Some(other) => Err(frankensearch_core::SearchError::SubsystemError {
            subsystem: "ops-data-source",
            source: Box::new(io::Error::other(format!(
                "unexpected type for {field}: {other:?}"
            ))),
        }),
    }
}

fn row_text<'a>(
    row: &'a Row,
    index: usize,
    field: &str,
) -> frankensearch_core::SearchResult<&'a str> {
    match row.get(index) {
        Some(SqliteValue::Text(value)) => Ok(&**value),
        Some(other) => Err(frankensearch_core::SearchError::SubsystemError {
            subsystem: "ops-data-source",
            source: Box::new(io::Error::other(format!(
                "unexpected type for {field}: {other:?}"
            ))),
        }),
        None => Err(frankensearch_core::SearchError::SubsystemError {
            subsystem: "ops-data-source",
            source: Box::new(io::Error::other(format!("missing column for {field}"))),
        }),
    }
}

fn row_opt_text<'a>(
    row: &'a Row,
    index: usize,
    field: &str,
) -> frankensearch_core::SearchResult<Option<&'a str>> {
    match row.get(index) {
        Some(SqliteValue::Text(value)) => Ok(Some(&**value)),
        Some(SqliteValue::Null) | None => Ok(None),
        Some(other) => Err(frankensearch_core::SearchError::SubsystemError {
            subsystem: "ops-data-source",
            source: Box::new(io::Error::other(format!(
                "unexpected type for {field}: {other:?}"
            ))),
        }),
    }
}

fn i64_to_f64_exact(value: i64, field: &str) -> frankensearch_core::SearchResult<f64> {
    const MAX_EXACT_F64_INTEGER: i64 = 9_007_199_254_740_992;
    if !(-MAX_EXACT_F64_INTEGER..=MAX_EXACT_F64_INTEGER).contains(&value) {
        return Err(frankensearch_core::SearchError::SubsystemError {
            subsystem: "ops-data-source",
            source: Box::new(io::Error::other(format!(
                "integer value for {field} exceeds exact f64 range: {value}"
            ))),
        });
    }
    value.to_string().parse::<f64>().map_err(data_source_error)
}

fn i64_to_u64_non_negative(value: i64, field: &str) -> frankensearch_core::SearchResult<u64> {
    if value < 0 {
        return Err(frankensearch_core::SearchError::SubsystemError {
            subsystem: "ops-data-source",
            source: Box::new(io::Error::other(format!("{field} must be >= 0"))),
        });
    }
    Ok(u64::try_from(value).unwrap_or(u64::MAX))
}

fn opt_i64_to_u64_non_negative(
    value: Option<i64>,
    field: &str,
) -> frankensearch_core::SearchResult<Option<u64>> {
    value
        .map(|raw| i64_to_u64_non_negative(raw, field))
        .transpose()
}

fn unix_timestamp_ms_u64() -> frankensearch_core::SearchResult<u64> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(data_source_error)?;
    Ok(u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX))
}

// ─── Mock Data Source ────────────────────────────────────────────────────────

/// Mock data source for development and testing.
///
/// Provides synthetic fleet data so the TUI can be developed and tested
/// independently of the real `FrankenSQLite` backend.
pub struct MockDataSource {
    snapshot: FleetSnapshot,
    control_plane: ControlPlaneMetrics,
}

impl MockDataSource {
    /// Create a mock with sample data.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn sample() -> Self {
        let mut snapshot = FleetSnapshot::default();
        let resolver = ProjectAttributionResolver;

        snapshot.instances.push(InstanceInfo {
            id: "cass-001".to_string(),
            project: "cass".to_string(),
            pid: Some(12345),
            healthy: true,
            doc_count: 48_532,
            pending_jobs: 7,
        });
        snapshot.instances.push(InstanceInfo {
            id: "xf-001".to_string(),
            project: "xf".to_string(),
            pid: Some(23456),
            healthy: true,
            doc_count: 12_801,
            pending_jobs: 0,
        });
        snapshot.instances.push(InstanceInfo {
            id: "amail-001".to_string(),
            project: "agent-mail".to_string(),
            pid: Some(34567),
            healthy: false,
            doc_count: 91_204,
            pending_jobs: 1_542,
        });

        snapshot.resources.insert(
            "cass-001".to_string(),
            ResourceMetrics {
                cpu_percent: 12.5,
                memory_bytes: 256 * 1024 * 1024,
                io_read_bytes: 1024 * 1024,
                io_write_bytes: 512 * 1024,
            },
        );
        snapshot.resources.insert(
            "xf-001".to_string(),
            ResourceMetrics {
                cpu_percent: 3.2,
                memory_bytes: 128 * 1024 * 1024,
                io_read_bytes: 256 * 1024,
                io_write_bytes: 64 * 1024,
            },
        );
        snapshot.resources.insert(
            "amail-001".to_string(),
            ResourceMetrics {
                cpu_percent: 87.3,
                memory_bytes: 512 * 1024 * 1024,
                io_read_bytes: 4 * 1024 * 1024,
                io_write_bytes: 2 * 1024 * 1024,
            },
        );

        snapshot.search_metrics.insert(
            "cass-001".to_string(),
            SearchMetrics {
                total_searches: 1_247,
                avg_latency_us: 850,
                p95_latency_us: 2_100,
                refined_count: 312,
            },
        );
        snapshot.search_metrics.insert(
            "xf-001".to_string(),
            SearchMetrics {
                total_searches: 89,
                avg_latency_us: 1_200,
                p95_latency_us: 3_500,
                refined_count: 15,
            },
        );

        snapshot.attribution.insert(
            "cass-001".to_string(),
            resolver.resolve(
                Some("cass"),
                Some("cass-devbox"),
                Some("coding-agent-session-search"),
            ),
        );
        snapshot.attribution.insert(
            "xf-001".to_string(),
            resolver.resolve(Some("xf"), Some("xf-worker"), Some("xf")),
        );
        snapshot.attribution.insert(
            "amail-001".to_string(),
            resolver.resolve(
                Some("agent-mail"),
                Some("mail-runner-42"),
                Some("mcp_agent_mail_rust"),
            ),
        );

        let mut cass_lifecycle = InstanceLifecycle::new(1_000);
        cass_lifecycle.apply_signal(LifecycleSignal::Heartbeat, 1_250, None);
        snapshot
            .lifecycle
            .insert("cass-001".to_string(), cass_lifecycle);

        let mut xf_lifecycle = InstanceLifecycle::new(1_000);
        xf_lifecycle.apply_signal(LifecycleSignal::Heartbeat, 1_300, None);
        snapshot
            .lifecycle
            .insert("xf-001".to_string(), xf_lifecycle);

        let mut amail_lifecycle = InstanceLifecycle::new(1_000);
        amail_lifecycle.apply_signal(LifecycleSignal::Heartbeat, 1_100, None);
        amail_lifecycle.mark_stale_if_heartbeat_gap(9_500, 5_000);
        snapshot
            .lifecycle
            .insert("amail-001".to_string(), amail_lifecycle);

        let control_plane = ControlPlaneMetrics {
            ingestion_lag_events: 1_549,
            storage_bytes: 384 * 1024 * 1024,
            storage_limit_bytes: 512 * 1024 * 1024,
            frame_time_ms: 19.8,
            discovery_latency_ms: 320,
            event_throughput_eps: 145.2,
            rss_bytes: 620 * 1024 * 1024,
            rss_limit_bytes: 1024 * 1024 * 1024,
            dead_letter_events: 2,
        };

        Self {
            snapshot,
            control_plane,
        }
    }

    /// Create an empty mock (no instances).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            snapshot: FleetSnapshot::default(),
            control_plane: ControlPlaneMetrics::default(),
        }
    }

    /// Build a mock snapshot from reconciled discovery output.
    ///
    /// This keeps view-level tests deterministic while exercising the same
    /// instance identity model used by the discovery engine.
    #[must_use]
    pub fn from_discovery(instances: &[DiscoveredInstance]) -> Self {
        let mut snapshot = FleetSnapshot::default();
        let now_ms = instances
            .iter()
            .map(|instance| instance.last_seen_ms)
            .max()
            .unwrap_or(0);
        let mut tracker = ProjectLifecycleTracker::new(LifecycleTrackerConfig::default());
        let lifecycle_events = tracker.ingest_discovery(now_ms, instances);

        for instance in instances {
            let instance_id = instance.instance_id.clone();
            let attribution = tracker
                .attribution_for(&instance_id)
                .cloned()
                .unwrap_or_else(|| {
                    InstanceAttribution::unknown(
                        instance.project_key_hint.as_deref(),
                        instance.host_name.as_deref(),
                        "attribution.discovery.unresolved",
                    )
                });

            snapshot.instances.push(InstanceInfo {
                id: instance_id.clone(),
                project: attribution.resolved_project.clone(),
                pid: instance.pid,
                healthy: instance.status == DiscoveryStatus::Active,
                doc_count: 0,
                pending_jobs: 0,
            });
        }
        snapshot.attribution = tracker.attribution_snapshot();
        snapshot.lifecycle = tracker.lifecycle_snapshot();
        snapshot.lifecycle_events = lifecycle_events;
        snapshot
            .instances
            .sort_by(|left, right| left.id.cmp(&right.id));
        Self {
            snapshot,
            control_plane: ControlPlaneMetrics::default(),
        }
    }
}

impl DataSource for MockDataSource {
    fn fleet_snapshot(&self) -> FleetSnapshot {
        self.snapshot.clone()
    }

    fn search_metrics(&self, instance_id: &str, _window: TimeWindow) -> Option<SearchMetrics> {
        self.snapshot.search_metrics.get(instance_id).cloned()
    }

    fn resource_metrics(&self, instance_id: &str) -> Option<ResourceMetrics> {
        self.snapshot.resources.get(instance_id).cloned()
    }

    fn control_plane_metrics(&self) -> ControlPlaneMetrics {
        self.control_plane.clone()
    }

    fn attribution(&self, instance_id: &str) -> Option<InstanceAttribution> {
        self.snapshot.attribution.get(instance_id).cloned()
    }

    fn lifecycle(&self, instance_id: &str) -> Option<InstanceLifecycle> {
        self.snapshot.lifecycle.get(instance_id).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_sample_has_instances() {
        let mock = MockDataSource::sample();
        let snap = mock.fleet_snapshot();
        assert_eq!(snap.instance_count(), 3);
        assert_eq!(snap.healthy_count(), 2);
    }

    #[test]
    fn mock_empty_has_no_instances() {
        let mock = MockDataSource::empty();
        let snap = mock.fleet_snapshot();
        assert_eq!(snap.instance_count(), 0);
    }

    #[test]
    fn mock_search_metrics() {
        let mock = MockDataSource::sample();
        let metrics = mock.search_metrics("cass-001", TimeWindow::OneHour);
        assert!(metrics.is_some());
        assert!(metrics.unwrap().total_searches > 0);
    }

    #[test]
    fn mock_resource_metrics() {
        let mock = MockDataSource::sample();
        let metrics = mock.resource_metrics("xf-001");
        assert!(metrics.is_some());
    }

    #[test]
    fn mock_unknown_instance() {
        let mock = MockDataSource::sample();
        assert!(
            mock.search_metrics("unknown", TimeWindow::OneMinute)
                .is_none()
        );
        assert!(mock.resource_metrics("unknown").is_none());
        assert!(mock.attribution("unknown").is_none());
        assert!(mock.lifecycle("unknown").is_none());
    }

    #[test]
    fn mock_control_plane_metrics_present() {
        let mock = MockDataSource::sample();
        let metrics = mock.control_plane_metrics();
        assert!(metrics.ingestion_lag_events > 0);
        assert!(metrics.storage_limit_bytes > metrics.storage_bytes / 2);
    }

    #[test]
    fn mock_attribution_metadata_present() {
        let mock = MockDataSource::sample();
        let attribution = mock
            .attribution("cass-001")
            .expect("cass attribution exists");
        assert_eq!(attribution.resolved_project, "coding_agent_session_search");
        assert!(attribution.confidence_score >= 80);
    }

    #[test]
    fn mock_lifecycle_contains_stale_instance() {
        let mock = MockDataSource::sample();
        let lifecycle = mock.lifecycle("amail-001").expect("amail lifecycle exists");
        assert_eq!(lifecycle.state, frankensearch_core::LifecycleState::Stale);
    }

    #[test]
    fn mock_from_discovery_maps_instance_health() {
        let instances = vec![
            DiscoveredInstance {
                instance_id: "inst-a".to_owned(),
                project_key_hint: Some("cass".to_owned()),
                host_name: Some("host-a".to_owned()),
                pid: Some(11),
                version: Some("0.1.0".to_owned()),
                first_seen_ms: 10,
                last_seen_ms: 20,
                status: DiscoveryStatus::Active,
                sources: vec![crate::DiscoverySignalKind::Process],
                identity_keys: vec!["hostpid:host-a:11".to_owned()],
            },
            DiscoveredInstance {
                instance_id: "inst-b".to_owned(),
                project_key_hint: None,
                host_name: Some("host-b".to_owned()),
                pid: Some(22),
                version: Some("0.1.0".to_owned()),
                first_seen_ms: 10,
                last_seen_ms: 20,
                status: DiscoveryStatus::Stale,
                sources: vec![crate::DiscoverySignalKind::Heartbeat],
                identity_keys: vec!["heartbeat:/tmp/inst-b".to_owned()],
            },
        ];

        let mock = MockDataSource::from_discovery(&instances);
        let fleet = mock.fleet_snapshot();
        assert_eq!(fleet.instance_count(), 2);
        assert_eq!(fleet.healthy_count(), 1);
        assert_eq!(fleet.instances[0].project, "coding_agent_session_search");
        assert_eq!(fleet.instances[1].project, "unknown");

        let inst_a_attr = mock
            .attribution("inst-a")
            .expect("inst-a attribution should be present");
        assert_eq!(inst_a_attr.resolved_project, "coding_agent_session_search");
        assert!(!inst_a_attr.collision);
        assert_eq!(inst_a_attr.reason_code, "attribution.telemetry_project_key");

        let inst_b_lifecycle = mock
            .lifecycle("inst-b")
            .expect("inst-b lifecycle should be present");
        assert_eq!(
            inst_b_lifecycle.state,
            frankensearch_core::LifecycleState::Stale
        );
        assert_eq!(inst_b_lifecycle.reason_code, "lifecycle.heartbeat_gap");
        assert!(
            !fleet.lifecycle_events().is_empty(),
            "lifecycle transitions should be emitted for timeline/alerts"
        );
    }

    #[test]
    fn mock_from_discovery_handles_missing_identity_hints() {
        let instances = vec![DiscoveredInstance {
            instance_id: "inst-unknown".to_owned(),
            project_key_hint: None,
            host_name: None,
            pid: None,
            version: Some("0.1.0".to_owned()),
            first_seen_ms: 10,
            last_seen_ms: 20,
            status: DiscoveryStatus::Active,
            sources: vec![crate::DiscoverySignalKind::Heartbeat],
            identity_keys: vec!["fallback:heartbeat:20".to_owned()],
        }];

        let mock = MockDataSource::from_discovery(&instances);
        let attribution = mock
            .attribution("inst-unknown")
            .expect("unknown attribution should exist");
        assert_eq!(attribution.resolved_project, "unknown");
        assert_eq!(attribution.reason_code, "attribution.unknown");
        assert_eq!(attribution.confidence_score, 20);
        assert!(
            attribution
                .evidence_trace
                .iter()
                .any(|entry| entry == "resolved_project=unknown")
        );

        let lifecycle = mock
            .lifecycle("inst-unknown")
            .expect("unknown lifecycle should exist");
        assert_eq!(lifecycle.state, frankensearch_core::LifecycleState::Healthy);
    }

    #[test]
    fn mock_from_discovery_surfaces_attribution_collisions() {
        let instances = vec![DiscoveredInstance {
            instance_id: "inst-collision".to_owned(),
            project_key_hint: Some("xf".to_owned()),
            host_name: Some("cass-devbox".to_owned()),
            pid: Some(333),
            version: Some("0.3.0".to_owned()),
            first_seen_ms: 100,
            last_seen_ms: 120,
            status: DiscoveryStatus::Active,
            sources: vec![crate::DiscoverySignalKind::Heartbeat],
            identity_keys: vec!["hostpid:cass-devbox:333".to_owned()],
        }];

        let mock = MockDataSource::from_discovery(&instances);
        let attribution = mock
            .attribution("inst-collision")
            .expect("collision attribution should exist");
        assert_eq!(attribution.resolved_project, "xf");
        assert!(attribution.collision);
        assert_eq!(attribution.reason_code, "attribution.collision");

        let lifecycle = mock
            .lifecycle("inst-collision")
            .expect("collision lifecycle should exist");
        assert_eq!(lifecycle.state, frankensearch_core::LifecycleState::Healthy);

        let events = mock.fleet_snapshot().lifecycle_events;
        assert!(
            events
                .iter()
                .any(|event| event.instance_id == "inst-collision"),
            "collision instance should emit lifecycle events"
        );
        assert!(
            events.iter().all(|event| event.attribution_collision),
            "event metadata should carry attribution collision flag"
        );
    }

    #[test]
    fn storage_data_source_refreshes_snapshot_from_ops_storage() {
        let storage = OpsStorage::open_in_memory().expect("open in-memory ops storage");
        let now_ms = i64::try_from(unix_timestamp_ms_u64().expect("now ms")).expect("i64 now");
        storage
            .connection()
            .execute(
                "INSERT INTO projects(project_key, display_name, created_at_ms, updated_at_ms) \
                 VALUES ('proj-a', 'proj-a', 1, 1);",
            )
            .expect("insert project");
        let first_seen_ms = now_ms.saturating_sub(1_000);
        let summary_start_ms = now_ms.saturating_sub(3_600_000);
        storage
            .connection()
            .execute(&format!(
                "INSERT INTO instances(\
                    instance_id, project_key, host_name, pid, version, first_seen_ms, \
                    last_heartbeat_ms, state\
                 ) VALUES (\
                    'inst-a', 'proj-a', 'host-a', 123, '0.1.0', {first_seen_ms}, {now_ms}, 'healthy'\
                 );"
            ))
            .expect("insert instance");
        storage
            .connection()
            .execute(&format!(
                "INSERT INTO search_summaries(\
                    project_key, instance_id, window, window_start_ms, search_count, \
                    p50_latency_us, p95_latency_us, p99_latency_us, avg_result_count\
                 ) VALUES (\
                    'proj-a', 'inst-a', '1h', {summary_start_ms}, 42, 1200, 2400, 3600, 6.0\
                 );"
            ))
            .expect("insert summary");
        storage
            .connection()
            .execute(&format!(
                "INSERT INTO resource_samples(\
                    project_key, instance_id, cpu_pct, rss_bytes, io_read_bytes, io_write_bytes, \
                    queue_depth, ts_ms\
                 ) VALUES (\
                    'proj-a', 'inst-a', 22.5, 4096, 1024, 2048, 3, {now_ms}\
                 );"
            ))
            .expect("insert resource sample");
        storage
            .connection()
            .execute(&format!(
                "INSERT INTO index_inventory_snapshots(\
                    snapshot_id, project_key, instance_id, index_name, index_type, record_count, \
                    file_size_bytes, file_hash, is_stale, ts_ms\
                 ) VALUES (\
                    'snap-a', 'proj-a', 'inst-a', 'vector', 'fsvi', 9001, 0, NULL, 0, {now_ms}\
                 );"
            ))
            .expect("insert inventory");

        let source = StorageDataSource::new(storage);
        let fleet = source.fleet_snapshot();
        assert_eq!(fleet.instance_count(), 1);
        assert_eq!(fleet.healthy_count(), 1);
        assert_eq!(fleet.instances[0].id, "inst-a");
        assert_eq!(fleet.instances[0].doc_count, 9001);
        assert_eq!(fleet.instances[0].pending_jobs, 3);

        let search = source
            .search_metrics("inst-a", TimeWindow::OneHour)
            .expect("search metrics");
        assert_eq!(search.total_searches, 42);
        assert_eq!(search.p95_latency_us, 2400);

        let resource = source.resource_metrics("inst-a").expect("resource metrics");
        assert_eq!(resource.memory_bytes, 4096);
        assert_eq!(resource.io_write_bytes, 2048);

        let control = source.control_plane_metrics();
        assert_eq!(control.ingestion_lag_events, 0);
    }

    #[test]
    fn storage_data_source_marks_instances_stale_after_heartbeat_gap() {
        let storage = OpsStorage::open_in_memory().expect("open in-memory ops storage");
        let now_ms = i64::try_from(unix_timestamp_ms_u64().expect("now ms")).expect("i64 now");
        storage
            .connection()
            .execute(
                "INSERT INTO projects(project_key, display_name, created_at_ms, updated_at_ms) \
                 VALUES ('proj-stale', 'proj-stale', 1, 1);",
            )
            .expect("insert project");
        let first_seen_ms = now_ms.saturating_sub(120_000);
        let last_heartbeat_ms = now_ms.saturating_sub(90_000);
        storage
            .connection()
            .execute(&format!(
                "INSERT INTO instances(\
                    instance_id, project_key, host_name, pid, version, first_seen_ms, \
                    last_heartbeat_ms, state\
                 ) VALUES (\
                    'inst-stale', 'proj-stale', 'host-z', 555, '0.1.0', {first_seen_ms}, \
                    {last_heartbeat_ms}, 'healthy'\
                 );"
            ))
            .expect("insert instance");

        let source = StorageDataSource::new(storage);
        let fleet = source.fleet_snapshot();
        assert_eq!(fleet.instance_count(), 1);
        assert_eq!(fleet.healthy_count(), 0);

        let lifecycle = source
            .lifecycle("inst-stale")
            .expect("lifecycle for stale instance");
        assert_eq!(lifecycle.state, frankensearch_core::LifecycleState::Stale);
    }

    #[test]
    fn time_window_all() {
        assert_eq!(TimeWindow::ALL.len(), 7);
    }

    #[test]
    fn time_window_labels() {
        assert_eq!(TimeWindow::OneMinute.label(), "1m");
        assert_eq!(TimeWindow::OneWeek.label(), "1w");
    }

    #[test]
    fn time_window_seconds() {
        assert_eq!(TimeWindow::OneHour.seconds(), 3600);
        assert_eq!(TimeWindow::OneWeek.seconds(), 7 * 24 * 3600);
    }

    #[test]
    fn time_window_display() {
        assert_eq!(TimeWindow::FifteenMinutes.to_string(), "15m");
    }
}
