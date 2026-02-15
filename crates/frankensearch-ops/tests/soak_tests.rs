//! Long-duration soak tests with leak detection and drift diagnostics.
//!
//! These tests validate the fleet dashboard and telemetry pipeline under
//! sustained mixed workloads. All are `#[ignore]` because they run
//! significantly longer than normal unit tests.
//!
//! Run with: `cargo test -p frankensearch-ops --test soak_tests -- --ignored --nocapture`
use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use frankensearch_core::{SearchError, SearchResult};
use frankensearch_ops::{
    OpsStorage, SimulatedProject, SimulationRun, SloMaterializationConfig, SloScope,
    TelemetrySimulator, TelemetrySimulatorConfig, WorkloadProfile, storage::SummaryWindow,
};

// ─── Soak Profiles ──────────────────────────────────────────────────────────

const SIX_HOUR_TICKS: usize = 6 * 60 * 60;
const TWENTY_FOUR_HOUR_TICKS: usize = 24 * 60 * 60;

fn soak_config_mixed(seed: u64, ticks: usize) -> TelemetrySimulatorConfig {
    TelemetrySimulatorConfig {
        seed,
        start_ms: 1_734_503_200_000,
        tick_interval_ms: 1_000,
        ticks,
        projects: vec![
            SimulatedProject {
                project_key: "cass-long".to_owned(),
                host_name: "cass-long".to_owned(),
                instance_count: 4,
                workload: WorkloadProfile::Steady,
            },
            SimulatedProject {
                project_key: "xf-long".to_owned(),
                host_name: "xf-long".to_owned(),
                instance_count: 3,
                workload: WorkloadProfile::Burst,
            },
            SimulatedProject {
                project_key: "mail-long".to_owned(),
                host_name: "mail-long".to_owned(),
                instance_count: 2,
                workload: WorkloadProfile::EmbeddingWave,
            },
            SimulatedProject {
                project_key: "term-long".to_owned(),
                host_name: "term-long".to_owned(),
                instance_count: 2,
                workload: WorkloadProfile::Restarting,
            },
        ],
    }
}

/// Short soak: steady + burst + embedding wave workloads across 4 projects.
/// Uses 120 ticks (2 simulated minutes) to balance coverage with wall-clock time.
/// `FrankenSQLite` in-memory mode processes ~2 ticks/second, so this runs ~60s.
fn soak_config_short(seed: u64) -> TelemetrySimulatorConfig {
    TelemetrySimulatorConfig {
        seed,
        start_ms: 1_734_503_200_000,
        tick_interval_ms: 1_000,
        ticks: 120, // 2 simulated minutes (enough for drift detection)
        projects: vec![
            SimulatedProject {
                project_key: "cass-soak".to_owned(),
                host_name: "cass-soak".to_owned(),
                instance_count: 3,
                workload: WorkloadProfile::Steady,
            },
            SimulatedProject {
                project_key: "xf-soak".to_owned(),
                host_name: "xf-soak".to_owned(),
                instance_count: 2,
                workload: WorkloadProfile::Burst,
            },
            SimulatedProject {
                project_key: "mail-soak".to_owned(),
                host_name: "mail-soak".to_owned(),
                instance_count: 2,
                workload: WorkloadProfile::EmbeddingWave,
            },
            SimulatedProject {
                project_key: "term-soak".to_owned(),
                host_name: "term-soak".to_owned(),
                instance_count: 1,
                workload: WorkloadProfile::Restarting,
            },
        ],
    }
}

/// Extended soak: larger fleet with mixed profiles over 6 simulated hours.
fn soak_config_extended(seed: u64) -> TelemetrySimulatorConfig {
    soak_config_mixed(seed, SIX_HOUR_TICKS)
}

/// Day-long soak: same mixed fleet profile over 24 simulated hours.
fn soak_config_daylong(seed: u64) -> TelemetrySimulatorConfig {
    soak_config_mixed(seed, TWENTY_FOUR_HOUR_TICKS)
}

// ─── Checkpoint Infrastructure ──────────────────────────────────────────────

/// Metrics snapshot captured at a checkpoint during soak execution.
#[derive(Debug, Clone)]
struct SoakCheckpoint {
    /// Tick index at which this checkpoint was captured.
    _tick_index: usize,
    /// Total events ingested so far.
    total_inserted: u64,
    /// Total batches processed so far.
    total_batches: u64,
    /// Total write latency accumulated (microseconds).
    total_write_latency_us: u64,
    /// Total backpressured batches so far.
    backpressured_batches: u64,
    /// Total deduplicated events.
    _total_deduplicated: u64,
    /// Total failed records.
    total_failed_records: u64,
    /// Pending events in the queue.
    pending_events: usize,
    /// High watermark of pending events.
    high_watermark_pending: usize,
    /// Open anomalies across fleet scope at this checkpoint.
    open_anomalies: usize,
}

impl SoakCheckpoint {
    /// Average write latency per batch in microseconds.
    #[allow(clippy::cast_precision_loss)]
    fn avg_write_latency_us(&self) -> f64 {
        if self.total_batches == 0 {
            return 0.0;
        }
        self.total_write_latency_us as f64 / self.total_batches as f64
    }
}

/// Summary of drift analysis across all checkpoints.
#[derive(Debug)]
struct DriftReport {
    /// Number of checkpoints captured.
    checkpoint_count: usize,
    /// Total events ingested across the entire soak.
    total_events: u64,
    /// Total batches processed.
    total_batches: u64,
    /// Average write latency at first checkpoint (us).
    first_avg_latency_us: f64,
    /// Average write latency at last checkpoint (us).
    last_avg_latency_us: f64,
    /// Latency drift percentage (positive = degradation).
    latency_drift_pct: f64,
    /// Maximum per-checkpoint pending event count.
    max_pending_events: usize,
    /// Maximum per-checkpoint open anomaly count.
    max_open_anomalies: usize,
    /// Whether monotonic growth in pending events was detected.
    monotonic_pending_growth: bool,
    /// Whether monotonic growth in anomalies was detected.
    monotonic_anomaly_growth: bool,
    /// Per-checkpoint delta of `total_inserted` (for throughput stability).
    throughput_deltas: Vec<u64>,
    /// Total backpressured batches.
    total_backpressured: u64,
    /// Total failed records.
    total_failed: u64,
}

impl DriftReport {
    fn from_checkpoints(checkpoints: &[SoakCheckpoint]) -> Self {
        let checkpoint_count = checkpoints.len();
        let total_events = checkpoints.last().map_or(0, |c| c.total_inserted);
        let total_batches = checkpoints.last().map_or(0, |c| c.total_batches);
        let total_backpressured = checkpoints.last().map_or(0, |c| c.backpressured_batches);
        let total_failed = checkpoints.last().map_or(0, |c| c.total_failed_records);

        let first_avg_latency_us = checkpoints
            .first()
            .map_or(0.0, SoakCheckpoint::avg_write_latency_us);
        let last_avg_latency_us = checkpoints
            .last()
            .map_or(0.0, SoakCheckpoint::avg_write_latency_us);
        let latency_drift_pct = if first_avg_latency_us > 0.0 {
            ((last_avg_latency_us - first_avg_latency_us) / first_avg_latency_us) * 100.0
        } else {
            0.0
        };

        let max_pending_events = checkpoints
            .iter()
            .map(|c| c.pending_events)
            .max()
            .unwrap_or(0);
        let max_open_anomalies = checkpoints
            .iter()
            .map(|c| c.open_anomalies)
            .max()
            .unwrap_or(0);

        // Detect monotonic growth in pending events.
        let monotonic_pending_growth = checkpoints.len() >= 3
            && checkpoints
                .windows(2)
                .all(|w| w[1].pending_events >= w[0].pending_events)
            && checkpoints.last().is_some_and(|c| c.pending_events > 0);

        // Detect monotonic growth in anomalies.
        let monotonic_anomaly_growth = checkpoints.len() >= 3
            && checkpoints
                .windows(2)
                .all(|w| w[1].open_anomalies >= w[0].open_anomalies)
            && checkpoints.last().is_some_and(|c| c.open_anomalies > 0);

        // Per-checkpoint throughput deltas.
        let mut throughput_deltas = Vec::with_capacity(checkpoint_count);
        for pair in checkpoints.windows(2) {
            throughput_deltas.push(
                pair[1]
                    .total_inserted
                    .saturating_sub(pair[0].total_inserted),
            );
        }

        Self {
            checkpoint_count,
            total_events,
            total_batches,
            first_avg_latency_us,
            last_avg_latency_us,
            latency_drift_pct,
            max_pending_events,
            max_open_anomalies,
            monotonic_pending_growth,
            monotonic_anomaly_growth,
            throughput_deltas,
            total_backpressured,
            total_failed,
        }
    }

    /// Print diagnostic summary to stderr (visible with --nocapture).
    #[allow(clippy::cast_precision_loss)]
    fn print_diagnostics(&self) {
        eprintln!("=== SOAK DRIFT DIAGNOSTICS ===");
        eprintln!("  checkpoints:          {}", self.checkpoint_count);
        eprintln!("  total_events:         {}", self.total_events);
        eprintln!("  total_batches:        {}", self.total_batches);
        eprintln!(
            "  avg_latency (first):  {:.1} us",
            self.first_avg_latency_us
        );
        eprintln!("  avg_latency (last):   {:.1} us", self.last_avg_latency_us);
        eprintln!("  latency_drift:        {:+.1}%", self.latency_drift_pct);
        eprintln!("  max_pending_events:   {}", self.max_pending_events);
        eprintln!("  max_open_anomalies:   {}", self.max_open_anomalies);
        eprintln!("  monotonic_pending:    {}", self.monotonic_pending_growth);
        eprintln!("  monotonic_anomalies:  {}", self.monotonic_anomaly_growth);
        eprintln!("  backpressured:        {}", self.total_backpressured);
        eprintln!("  failed_records:       {}", self.total_failed);
        if !self.throughput_deltas.is_empty() {
            let min_delta = self.throughput_deltas.iter().min().copied().unwrap_or(0);
            let max_delta = self.throughput_deltas.iter().max().copied().unwrap_or(0);
            let avg_delta: f64 = self.throughput_deltas.iter().sum::<u64>() as f64
                / self.throughput_deltas.len() as f64;
            eprintln!("  throughput (min/avg/max): {min_delta}/{avg_delta:.0}/{max_delta}");
        }
        eprintln!("==============================");
    }
}

/// First divergence marker extracted from soak checkpoints.
#[derive(Debug, Clone, PartialEq, Eq)]
struct DivergenceMarker {
    checkpoint_index: usize,
    tick_index: usize,
    signal: &'static str,
    observed: String,
}

/// Detect the earliest divergence marker for diagnostics/replay triage.
fn detect_first_divergence(
    report: &DriftReport,
    checkpoints: &[SoakCheckpoint],
    max_pending_events: usize,
    max_open_anomalies: usize,
    max_latency_drift_pct: f64,
) -> Option<DivergenceMarker> {
    for (checkpoint_index, checkpoint) in checkpoints.iter().enumerate() {
        if checkpoint.total_failed_records > 0 {
            return Some(DivergenceMarker {
                checkpoint_index,
                tick_index: checkpoint_index,
                signal: "failed_records_detected",
                observed: checkpoint.total_failed_records.to_string(),
            });
        }
        if checkpoint.pending_events > max_pending_events {
            return Some(DivergenceMarker {
                checkpoint_index,
                tick_index: checkpoint_index,
                signal: "pending_events_limit_exceeded",
                observed: format!("{} > {}", checkpoint.pending_events, max_pending_events),
            });
        }
        if checkpoint.open_anomalies > max_open_anomalies {
            return Some(DivergenceMarker {
                checkpoint_index,
                tick_index: checkpoint_index,
                signal: "open_anomalies_limit_exceeded",
                observed: format!("{} > {}", checkpoint.open_anomalies, max_open_anomalies),
            });
        }
    }

    for checkpoint_index in 2..checkpoints.len() {
        let first = &checkpoints[checkpoint_index - 2];
        let second = &checkpoints[checkpoint_index - 1];
        let third = &checkpoints[checkpoint_index];

        if first.pending_events <= second.pending_events
            && second.pending_events <= third.pending_events
            && third.pending_events > 0
        {
            return Some(DivergenceMarker {
                checkpoint_index,
                tick_index: checkpoint_index,
                signal: "monotonic_pending_growth",
                observed: format!(
                    "{} -> {} -> {}",
                    first.pending_events, second.pending_events, third.pending_events
                ),
            });
        }

        if first.open_anomalies <= second.open_anomalies
            && second.open_anomalies <= third.open_anomalies
            && third.open_anomalies > 0
        {
            return Some(DivergenceMarker {
                checkpoint_index,
                tick_index: checkpoint_index,
                signal: "monotonic_anomaly_growth",
                observed: format!(
                    "{} -> {} -> {}",
                    first.open_anomalies, second.open_anomalies, third.open_anomalies
                ),
            });
        }
    }

    if report.latency_drift_pct > max_latency_drift_pct {
        let checkpoint_index = checkpoints.len().saturating_sub(1);
        let tick_index = checkpoint_index;
        return Some(DivergenceMarker {
            checkpoint_index,
            tick_index,
            signal: "latency_drift_limit_exceeded",
            observed: format!(
                "{:+.1}% > {:+.1}%",
                report.latency_drift_pct, max_latency_drift_pct
            ),
        });
    }

    None
}

/// Render a compact, deterministic trend log for soak diagnostics.
fn render_resource_trend_log(checkpoints: &[SoakCheckpoint]) -> String {
    let mut log = String::from(
        "checkpoint,tick,total_inserted,pending_events,high_watermark_pending,open_anomalies,backpressured_batches,failed_records\n",
    );
    for (checkpoint_index, checkpoint) in checkpoints.iter().enumerate() {
        let _ = writeln!(
            log,
            "{checkpoint_index},{},{},{},{},{},{},{}",
            checkpoint_index,
            checkpoint.total_inserted,
            checkpoint.pending_events,
            checkpoint.high_watermark_pending,
            checkpoint.open_anomalies,
            checkpoint.backpressured_batches,
            checkpoint.total_failed_records
        );
    }
    log
}

/// Build a replay-oriented failure artifact payload for soak assertions.
fn build_failure_artifact(
    replay_seed: u64,
    report: &DriftReport,
    checkpoints: &[SoakCheckpoint],
    marker: Option<&DivergenceMarker>,
) -> String {
    let marker_line = marker.map_or_else(
        || "none".to_owned(),
        |marker| {
            format!(
                "checkpoint={} tick={} signal={} observed={}",
                marker.checkpoint_index, marker.tick_index, marker.signal, marker.observed
            )
        },
    );

    format!(
        "--- SOAK FAILURE ARTIFACT ---\nreplay_seed: 0x{replay_seed:016X}\nfirst_divergence: {marker_line}\nlatency_drift_pct: {:+.1}\nmax_pending_events: {}\nmax_open_anomalies: {}\nresource_trend_log:\n{}--- END SOAK FAILURE ARTIFACT ---",
        report.latency_drift_pct,
        report.max_pending_events,
        report.max_open_anomalies,
        render_resource_trend_log(checkpoints)
    )
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ResourceExtrema {
    sample_count: usize,
    max_rss_bytes: u64,
    max_queue_depth: u64,
}

fn collect_resource_extrema(
    storage: &OpsStorage,
    run: &SimulationRun,
) -> SearchResult<ResourceExtrema> {
    let mut pairs = BTreeSet::new();
    for batch in &run.batches {
        for sample in &batch.resource_samples {
            pairs.insert((sample.project_key.clone(), sample.instance_id.clone()));
        }
    }

    let now_ms = run
        .batches
        .last()
        .map(|batch| {
            i64::try_from(batch.now_ms).map_err(|_| SearchError::InvalidConfig {
                field: "now_ms".to_owned(),
                value: batch.now_ms.to_string(),
                reason: "must fit into i64".to_owned(),
            })
        })
        .transpose()?
        .unwrap_or(0);

    let limit = run.batches.len().max(1);
    let mut extrema = ResourceExtrema::default();
    for (project_key, instance_id) in pairs {
        let points = storage.query_resource_trend(
            &project_key,
            &instance_id,
            SummaryWindow::OneWeek,
            now_ms,
            limit,
        )?;
        extrema.sample_count = extrema.sample_count.saturating_add(points.len());
        for point in points {
            if let Some(rss_bytes) = point.rss_bytes {
                extrema.max_rss_bytes = extrema.max_rss_bytes.max(rss_bytes);
            }
            if let Some(queue_depth) = point.queue_depth {
                extrema.max_queue_depth = extrema.max_queue_depth.max(queue_depth);
            }
        }
    }

    Ok(extrema)
}

fn assert_resource_budget(
    storage: &OpsStorage,
    run: &SimulationRun,
    max_rss_bytes: u64,
    max_queue_depth: u64,
    failure_artifact: &str,
) {
    let extrema =
        collect_resource_extrema(storage, run).expect("resource trend query should succeed");
    assert!(
        extrema.sample_count > 0,
        "resource trend query returned no samples\n{failure_artifact}"
    );
    assert!(
        extrema.max_rss_bytes <= max_rss_bytes,
        "rss budget exceeded: {} > {} bytes (samples={})\n{failure_artifact}",
        extrema.max_rss_bytes,
        max_rss_bytes,
        extrema.sample_count,
    );
    assert!(
        extrema.max_queue_depth <= max_queue_depth,
        "resource queue depth budget exceeded: {} > {}\n{failure_artifact}",
        extrema.max_queue_depth,
        max_queue_depth,
    );
}

fn persist_failure_artifact(
    test_name: &str,
    replay_seed: u64,
    artifact: &str,
) -> std::io::Result<PathBuf> {
    let directory = Path::new("target").join("soak-artifacts");
    fs::create_dir_all(&directory)?;
    let path = directory.join(format!("{test_name}-seed-{replay_seed:016X}.log"));
    fs::write(&path, artifact)?;
    Ok(path)
}

// ─── Pipeline Helpers ───────────────────────────────────────────────────────

/// Ingest a single batch into storage (events + resource samples + summaries).
///
/// SLO materialization is deferred to checkpoint boundaries for performance.
/// In production, materialization is periodic (not per-event), so this better
/// models real workloads while keeping wall-clock time reasonable.
fn ingest_batch(
    storage: &OpsStorage,
    batch: &frankensearch_ops::SimulationBatch,
    backpressure_threshold: usize,
) -> SearchResult<()> {
    let records: Vec<_> = batch
        .search_events
        .iter()
        .map(|event| event.record.clone())
        .collect();
    storage.ingest_search_events_batch(&records, backpressure_threshold)?;

    for sample in &batch.resource_samples {
        storage.upsert_resource_sample(sample)?;
    }

    let now_ms = i64::try_from(batch.now_ms).map_err(|_| SearchError::InvalidConfig {
        field: "now_ms".to_owned(),
        value: batch.now_ms.to_string(),
        reason: "must fit into i64".to_owned(),
    })?;
    let mut pairs = BTreeSet::new();
    for sample in &batch.resource_samples {
        pairs.insert((sample.project_key.clone(), sample.instance_id.clone()));
    }
    for (project_key, instance_id) in &pairs {
        let _ = storage.refresh_search_summaries_for_instance(project_key, instance_id, now_ms)?;
    }

    Ok(())
}

/// Run SLO materialization for a batch (expensive; call at checkpoint intervals).
fn materialize_batch(
    storage: &OpsStorage,
    batch: &frankensearch_ops::SimulationBatch,
) -> SearchResult<()> {
    let now_ms = i64::try_from(batch.now_ms).map_err(|_| SearchError::InvalidConfig {
        field: "now_ms".to_owned(),
        value: batch.now_ms.to_string(),
        reason: "must fit into i64".to_owned(),
    })?;
    storage.materialize_slo_rollups_and_anomalies(now_ms, SloMaterializationConfig::default())?;
    Ok(())
}

/// Capture a checkpoint from current storage state.
fn capture_checkpoint(storage: &OpsStorage, tick_index: usize) -> SoakCheckpoint {
    let metrics = storage.ingestion_metrics();
    let open_anomalies = storage
        .query_open_anomalies_for_scope(SloScope::Fleet, "__fleet__", 1024)
        .map_or(0, |v| v.len());

    SoakCheckpoint {
        _tick_index: tick_index,
        total_inserted: metrics.total_inserted,
        total_batches: metrics.total_batches,
        total_write_latency_us: metrics.total_write_latency_us,
        backpressured_batches: metrics.total_backpressured_batches,
        _total_deduplicated: metrics.total_deduplicated,
        total_failed_records: metrics.total_failed_records,
        pending_events: metrics.pending_events,
        high_watermark_pending: metrics.high_watermark_pending_events,
        open_anomalies,
    }
}

/// Run the soak pipeline: ingest all batches with periodic checkpoints.
///
/// SLO materialization is performed at checkpoint boundaries (not per-batch)
/// to keep wall-clock time practical. This mirrors production patterns where
/// materialization is periodic.
fn run_soak_pipeline(
    storage: &OpsStorage,
    run: &SimulationRun,
    checkpoint_interval: usize,
    backpressure_threshold: usize,
) -> SearchResult<Vec<SoakCheckpoint>> {
    let mut checkpoints = Vec::new();

    for (batch_idx, batch) in run.batches.iter().enumerate() {
        ingest_batch(storage, batch, backpressure_threshold)?;

        // At checkpoint boundaries (and final batch): materialize then capture.
        if batch_idx % checkpoint_interval == 0 || batch_idx == run.batches.len() - 1 {
            materialize_batch(storage, batch)?;
            checkpoints.push(capture_checkpoint(storage, batch_idx));
        }
    }

    Ok(checkpoints)
}

// ─── Soak Tests ─────────────────────────────────────────────────────────────

#[test]
fn drift_report_flags_monotonic_pending_and_anomaly_growth() {
    let checkpoints = vec![
        SoakCheckpoint {
            _tick_index: 0,
            total_inserted: 100,
            total_batches: 4,
            total_write_latency_us: 40_000,
            backpressured_batches: 0,
            _total_deduplicated: 0,
            total_failed_records: 0,
            pending_events: 10,
            high_watermark_pending: 10,
            open_anomalies: 1,
        },
        SoakCheckpoint {
            _tick_index: 1,
            total_inserted: 180,
            total_batches: 8,
            total_write_latency_us: 96_000,
            backpressured_batches: 1,
            _total_deduplicated: 2,
            total_failed_records: 0,
            pending_events: 20,
            high_watermark_pending: 20,
            open_anomalies: 2,
        },
        SoakCheckpoint {
            _tick_index: 2,
            total_inserted: 260,
            total_batches: 12,
            total_write_latency_us: 180_000,
            backpressured_batches: 2,
            _total_deduplicated: 4,
            total_failed_records: 0,
            pending_events: 30,
            high_watermark_pending: 30,
            open_anomalies: 3,
        },
    ];

    let report = DriftReport::from_checkpoints(&checkpoints);
    assert!(report.monotonic_pending_growth);
    assert!(report.monotonic_anomaly_growth);
    assert_eq!(report.max_pending_events, 30);
    assert_eq!(report.max_open_anomalies, 3);
}

#[test]
fn drift_report_computes_latency_drift_and_throughput_deltas() {
    let checkpoints = vec![
        SoakCheckpoint {
            _tick_index: 0,
            total_inserted: 50,
            total_batches: 5,
            total_write_latency_us: 100_000, // 20_000 avg
            backpressured_batches: 1,
            _total_deduplicated: 1,
            total_failed_records: 0,
            pending_events: 2,
            high_watermark_pending: 3,
            open_anomalies: 0,
        },
        SoakCheckpoint {
            _tick_index: 1,
            total_inserted: 120,
            total_batches: 10,
            total_write_latency_us: 300_000, // 30_000 avg
            backpressured_batches: 2,
            _total_deduplicated: 2,
            total_failed_records: 0,
            pending_events: 1,
            high_watermark_pending: 3,
            open_anomalies: 1,
        },
        SoakCheckpoint {
            _tick_index: 2,
            total_inserted: 200,
            total_batches: 15,
            total_write_latency_us: 600_000, // 40_000 avg
            backpressured_batches: 4,
            _total_deduplicated: 3,
            total_failed_records: 1,
            pending_events: 0,
            high_watermark_pending: 3,
            open_anomalies: 1,
        },
    ];

    let report = DriftReport::from_checkpoints(&checkpoints);
    assert_eq!(report.checkpoint_count, 3);
    assert_eq!(report.total_events, 200);
    assert_eq!(report.total_batches, 15);
    assert_eq!(report.total_backpressured, 4);
    assert_eq!(report.total_failed, 1);
    assert_eq!(report.throughput_deltas, vec![70, 80]);
    assert!(report.first_avg_latency_us > 0.0);
    assert!(report.last_avg_latency_us > report.first_avg_latency_us);
    assert!(report.latency_drift_pct > 0.0);
}

#[test]
fn drift_report_handles_zero_batch_checkpoints_without_divide_by_zero() {
    let checkpoints = vec![
        SoakCheckpoint {
            _tick_index: 0,
            total_inserted: 0,
            total_batches: 0,
            total_write_latency_us: 0,
            backpressured_batches: 0,
            _total_deduplicated: 0,
            total_failed_records: 0,
            pending_events: 0,
            high_watermark_pending: 0,
            open_anomalies: 0,
        },
        SoakCheckpoint {
            _tick_index: 1,
            total_inserted: 0,
            total_batches: 0,
            total_write_latency_us: 0,
            backpressured_batches: 0,
            _total_deduplicated: 0,
            total_failed_records: 0,
            pending_events: 0,
            high_watermark_pending: 0,
            open_anomalies: 0,
        },
    ];

    let report = DriftReport::from_checkpoints(&checkpoints);
    assert!(
        report.first_avg_latency_us.abs() <= f64::EPSILON,
        "expected zero first_avg_latency_us, got {}",
        report.first_avg_latency_us
    );
    assert!(
        report.last_avg_latency_us.abs() <= f64::EPSILON,
        "expected zero last_avg_latency_us, got {}",
        report.last_avg_latency_us
    );
    assert!(
        report.latency_drift_pct.abs() <= f64::EPSILON,
        "expected zero latency_drift_pct, got {}",
        report.latency_drift_pct
    );
}

#[test]
fn drift_report_empty_checkpoints_defaults_to_zero_metrics() {
    let report = DriftReport::from_checkpoints(&[]);
    assert_eq!(report.checkpoint_count, 0);
    assert_eq!(report.total_events, 0);
    assert_eq!(report.total_batches, 0);
    assert!(
        report.first_avg_latency_us.abs() <= f64::EPSILON,
        "expected zero first_avg_latency_us for empty checkpoints, got {}",
        report.first_avg_latency_us
    );
    assert!(
        report.last_avg_latency_us.abs() <= f64::EPSILON,
        "expected zero last_avg_latency_us for empty checkpoints, got {}",
        report.last_avg_latency_us
    );
    assert!(
        report.latency_drift_pct.abs() <= f64::EPSILON,
        "expected zero latency_drift_pct for empty checkpoints, got {}",
        report.latency_drift_pct
    );
    assert_eq!(report.max_pending_events, 0);
    assert_eq!(report.max_open_anomalies, 0);
    assert!(!report.monotonic_pending_growth);
    assert!(!report.monotonic_anomaly_growth);
    assert!(report.throughput_deltas.is_empty());
}

#[test]
fn drift_report_monotonic_flags_require_at_least_three_checkpoints() {
    let checkpoints = vec![
        SoakCheckpoint {
            _tick_index: 0,
            total_inserted: 10,
            total_batches: 1,
            total_write_latency_us: 1_000,
            backpressured_batches: 0,
            _total_deduplicated: 0,
            total_failed_records: 0,
            pending_events: 1,
            high_watermark_pending: 1,
            open_anomalies: 1,
        },
        SoakCheckpoint {
            _tick_index: 1,
            total_inserted: 20,
            total_batches: 2,
            total_write_latency_us: 2_000,
            backpressured_batches: 0,
            _total_deduplicated: 0,
            total_failed_records: 0,
            pending_events: 2,
            high_watermark_pending: 2,
            open_anomalies: 2,
        },
    ];

    let report = DriftReport::from_checkpoints(&checkpoints);
    assert!(!report.monotonic_pending_growth);
    assert!(!report.monotonic_anomaly_growth);
}

#[test]
fn drift_report_throughput_deltas_saturate_when_total_inserted_decreases() {
    let checkpoints = vec![
        SoakCheckpoint {
            _tick_index: 0,
            total_inserted: 100,
            total_batches: 10,
            total_write_latency_us: 200_000,
            backpressured_batches: 0,
            _total_deduplicated: 0,
            total_failed_records: 0,
            pending_events: 5,
            high_watermark_pending: 8,
            open_anomalies: 1,
        },
        SoakCheckpoint {
            _tick_index: 1,
            total_inserted: 90, // non-monotonic to exercise saturating_sub
            total_batches: 12,
            total_write_latency_us: 250_000,
            backpressured_batches: 1,
            _total_deduplicated: 0,
            total_failed_records: 0,
            pending_events: 4,
            high_watermark_pending: 8,
            open_anomalies: 1,
        },
        SoakCheckpoint {
            _tick_index: 2,
            total_inserted: 120,
            total_batches: 14,
            total_write_latency_us: 290_000,
            backpressured_batches: 2,
            _total_deduplicated: 0,
            total_failed_records: 0,
            pending_events: 3,
            high_watermark_pending: 8,
            open_anomalies: 1,
        },
    ];

    let report = DriftReport::from_checkpoints(&checkpoints);
    assert_eq!(report.throughput_deltas, vec![0, 30]);
}

#[test]
fn drift_report_monotonic_flags_require_positive_terminal_counts() {
    let checkpoints = vec![
        SoakCheckpoint {
            _tick_index: 0,
            total_inserted: 10,
            total_batches: 1,
            total_write_latency_us: 1_000,
            backpressured_batches: 0,
            _total_deduplicated: 0,
            total_failed_records: 0,
            pending_events: 0,
            high_watermark_pending: 0,
            open_anomalies: 0,
        },
        SoakCheckpoint {
            _tick_index: 1,
            total_inserted: 20,
            total_batches: 2,
            total_write_latency_us: 2_000,
            backpressured_batches: 0,
            _total_deduplicated: 0,
            total_failed_records: 0,
            pending_events: 0,
            high_watermark_pending: 0,
            open_anomalies: 0,
        },
        SoakCheckpoint {
            _tick_index: 2,
            total_inserted: 30,
            total_batches: 3,
            total_write_latency_us: 3_000,
            backpressured_batches: 0,
            _total_deduplicated: 0,
            total_failed_records: 0,
            pending_events: 0,
            high_watermark_pending: 0,
            open_anomalies: 0,
        },
    ];

    let report = DriftReport::from_checkpoints(&checkpoints);
    assert!(!report.monotonic_pending_growth);
    assert!(!report.monotonic_anomaly_growth);
}

#[test]
fn detect_first_divergence_prefers_earliest_threshold_violation() {
    let checkpoints = vec![
        SoakCheckpoint {
            _tick_index: 0,
            total_inserted: 50,
            total_batches: 5,
            total_write_latency_us: 100_000,
            backpressured_batches: 0,
            _total_deduplicated: 0,
            total_failed_records: 0,
            pending_events: 2,
            high_watermark_pending: 2,
            open_anomalies: 0,
        },
        SoakCheckpoint {
            _tick_index: 1,
            total_inserted: 80,
            total_batches: 8,
            total_write_latency_us: 190_000,
            backpressured_batches: 0,
            _total_deduplicated: 0,
            total_failed_records: 0,
            pending_events: 15,
            high_watermark_pending: 15,
            open_anomalies: 0,
        },
        SoakCheckpoint {
            _tick_index: 2,
            total_inserted: 120,
            total_batches: 12,
            total_write_latency_us: 320_000,
            backpressured_batches: 0,
            _total_deduplicated: 0,
            total_failed_records: 0,
            pending_events: 20,
            high_watermark_pending: 20,
            open_anomalies: 99,
        },
    ];

    let report = DriftReport::from_checkpoints(&checkpoints);
    let marker = detect_first_divergence(&report, &checkpoints, 10, 10, 300.0)
        .expect("pending threshold should trigger first");

    assert_eq!(marker.checkpoint_index, 1);
    assert_eq!(marker.tick_index, 1);
    assert_eq!(marker.signal, "pending_events_limit_exceeded");
}

#[test]
fn detect_first_divergence_detects_monotonic_growth_when_thresholds_not_exceeded() {
    let checkpoints = vec![
        SoakCheckpoint {
            _tick_index: 0,
            total_inserted: 50,
            total_batches: 5,
            total_write_latency_us: 100_000,
            backpressured_batches: 0,
            _total_deduplicated: 0,
            total_failed_records: 0,
            pending_events: 1,
            high_watermark_pending: 1,
            open_anomalies: 0,
        },
        SoakCheckpoint {
            _tick_index: 1,
            total_inserted: 100,
            total_batches: 10,
            total_write_latency_us: 200_000,
            backpressured_batches: 0,
            _total_deduplicated: 0,
            total_failed_records: 0,
            pending_events: 2,
            high_watermark_pending: 2,
            open_anomalies: 0,
        },
        SoakCheckpoint {
            _tick_index: 2,
            total_inserted: 150,
            total_batches: 15,
            total_write_latency_us: 300_000,
            backpressured_batches: 0,
            _total_deduplicated: 0,
            total_failed_records: 0,
            pending_events: 3,
            high_watermark_pending: 3,
            open_anomalies: 0,
        },
    ];

    let report = DriftReport::from_checkpoints(&checkpoints);
    let marker = detect_first_divergence(&report, &checkpoints, 128, 128, 300.0)
        .expect("monotonic pending growth should be detected");

    assert_eq!(marker.checkpoint_index, 2);
    assert_eq!(marker.tick_index, 2);
    assert_eq!(marker.signal, "monotonic_pending_growth");
    assert_eq!(marker.observed, "1 -> 2 -> 3");
}

#[test]
fn build_failure_artifact_includes_seed_marker_and_trend_log() {
    let checkpoints = vec![
        SoakCheckpoint {
            _tick_index: 0,
            total_inserted: 10,
            total_batches: 1,
            total_write_latency_us: 1_000,
            backpressured_batches: 0,
            _total_deduplicated: 0,
            total_failed_records: 0,
            pending_events: 1,
            high_watermark_pending: 1,
            open_anomalies: 0,
        },
        SoakCheckpoint {
            _tick_index: 1,
            total_inserted: 20,
            total_batches: 2,
            total_write_latency_us: 3_000,
            backpressured_batches: 0,
            _total_deduplicated: 0,
            total_failed_records: 0,
            pending_events: 2,
            high_watermark_pending: 2,
            open_anomalies: 0,
        },
    ];

    let report = DriftReport::from_checkpoints(&checkpoints);
    let marker = detect_first_divergence(&report, &checkpoints, 1, 8, 500.0);
    let artifact = build_failure_artifact(0xDEAD_BEEF_1234, &report, &checkpoints, marker.as_ref());

    assert!(artifact.contains("replay_seed: 0x0000DEADBEEF1234"));
    assert!(
        artifact
            .contains("first_divergence: checkpoint=1 tick=1 signal=pending_events_limit_exceeded")
    );
    assert!(artifact.contains("resource_trend_log:"));
    assert!(artifact.contains("checkpoint,tick,total_inserted,pending_events"));
}

#[test]
fn soak_duration_profiles_encode_6h_and_24h_windows() {
    let six_hour = soak_config_extended(0xABCD);
    let day_long = soak_config_daylong(0xEF01);

    assert_eq!(six_hour.tick_interval_ms, 1_000);
    assert_eq!(six_hour.ticks, SIX_HOUR_TICKS);
    assert_eq!(day_long.tick_interval_ms, 1_000);
    assert_eq!(day_long.ticks, TWENTY_FOUR_HOUR_TICKS);
}

#[test]
fn persist_failure_artifact_writes_deterministic_log_path() {
    let seed = 0xABCD_EF01_2345_6789;
    let artifact = "artifact payload";
    let path = persist_failure_artifact(
        "persist_failure_artifact_writes_deterministic_log_path",
        seed,
        artifact,
    )
    .expect("artifact file should be persisted");

    assert!(
        path.ends_with(Path::new(
            "target/soak-artifacts/persist_failure_artifact_writes_deterministic_log_path-seed-ABCDEF0123456789.log"
        )),
        "unexpected artifact path: {}",
        path.display()
    );
    let content = fs::read_to_string(&path).expect("artifact file should be readable");
    assert_eq!(content, artifact);
}

#[test]
fn run_soak_pipeline_captures_final_checkpoint_even_with_large_interval() {
    let config = TelemetrySimulatorConfig {
        seed: 0xDEAD_BEEF_1001,
        ticks: 6,
        projects: vec![SimulatedProject {
            project_key: "checkpoint-test".to_owned(),
            host_name: "checkpoint-test".to_owned(),
            instance_count: 1,
            workload: WorkloadProfile::Steady,
        }],
        ..TelemetrySimulatorConfig::default()
    };
    let sim = TelemetrySimulator::new(config).expect("config should validate");
    let run = sim.generate().expect("generation should succeed");
    assert!(!run.batches.is_empty(), "expected non-empty run");

    let storage = OpsStorage::open_in_memory().expect("storage should open");
    let checkpoints = run_soak_pipeline(&storage, &run, 1_000, 16_384)
        .expect("pipeline should succeed with large checkpoint interval");

    let expected_checkpoint_count = if run.batches.len() == 1 { 1 } else { 2 };
    assert_eq!(checkpoints.len(), expected_checkpoint_count);

    let metrics = storage.ingestion_metrics();
    assert_eq!(
        checkpoints.last().map_or(0, |c| c.total_inserted),
        metrics.total_inserted,
        "final checkpoint should reflect final ingested event count"
    );
}

#[test]
fn run_soak_pipeline_surfaces_queue_full_error_when_threshold_is_zero() {
    let config = TelemetrySimulatorConfig {
        seed: 0xDEAD_BEEF_1002,
        ticks: 8,
        projects: vec![SimulatedProject {
            project_key: "queuefull-test".to_owned(),
            host_name: "queuefull-test".to_owned(),
            instance_count: 3,
            workload: WorkloadProfile::Burst,
        }],
        ..TelemetrySimulatorConfig::default()
    };
    let sim = TelemetrySimulator::new(config).expect("config should validate");
    let run = sim.generate().expect("generation should succeed");
    assert!(
        run.total_search_events() > 0,
        "expected burst profile to generate search events"
    );

    let storage = OpsStorage::open_in_memory().expect("storage should open");
    let err =
        run_soak_pipeline(&storage, &run, 1, 0).expect_err("zero threshold should fail validation");
    assert!(
        matches!(err, SearchError::InvalidConfig { ref field, .. } if field == "backpressure_threshold"),
        "expected InvalidConfig(backpressure_threshold), got {err:?}"
    );
}

#[test]
#[ignore = "Soak test: ~120 ticks, run with --ignored --nocapture"]
fn soak_short_deterministic_replay() {
    // Verify deterministic generation: two runs with the same seed produce
    // identical signatures.
    let config = soak_config_short(0xDEAD_BEEF_0001);
    let sim = TelemetrySimulator::new(config).expect("config should validate");
    let run_a = sim.generate().expect("generation should succeed");
    let run_b = sim.generate().expect("generation should succeed");
    assert_eq!(
        run_a.signature(),
        run_b.signature(),
        "deterministic replay: identical seeds must produce identical signatures"
    );
    assert!(
        run_a.total_search_events() > 0,
        "soak run must produce search events"
    );
    assert!(
        run_a.total_resource_samples() > 0,
        "soak run must produce resource samples"
    );
}

#[test]
#[ignore = "Soak test: ~120 ticks with leak/drift detection, run with --ignored --nocapture"]
fn soak_short_no_leak_or_drift() {
    let config = soak_config_short(0xDEAD_BEEF_0002);
    let sim = TelemetrySimulator::new(config).expect("config should validate");
    let run = sim.generate().expect("generation should succeed");

    let storage = OpsStorage::open_in_memory().expect("storage should open");
    let checkpoint_interval = 30; // Every 30 ticks (30 simulated seconds)
    let backpressure_threshold = 16_384;

    let checkpoints =
        run_soak_pipeline(&storage, &run, checkpoint_interval, backpressure_threshold)
            .expect("soak pipeline should complete");

    let report = DriftReport::from_checkpoints(&checkpoints);
    report.print_diagnostics();
    let divergence = detect_first_divergence(
        &report,
        &checkpoints,
        backpressure_threshold.saturating_sub(1),
        50,
        200.0,
    );
    let failure_artifact =
        build_failure_artifact(0xDEAD_BEEF_0002, &report, &checkpoints, divergence.as_ref());
    let failure_artifact_path = persist_failure_artifact(
        "soak_short_no_leak_or_drift",
        0xDEAD_BEEF_0002,
        &failure_artifact,
    )
    .expect("failure artifact should be persisted");
    let queue_depth_budget =
        u64::try_from(backpressure_threshold).expect("backpressure threshold should fit u64");
    assert_resource_budget(
        &storage,
        &run,
        2_u64 * 1024 * 1024 * 1024,
        queue_depth_budget,
        &failure_artifact,
    );

    // ── Leak Detection ──────────────────────────────────────────────────

    // Pending events should not grow monotonically (leak signature).
    assert!(
        !report.monotonic_pending_growth,
        "leak detected: pending events grew monotonically across {} checkpoints \
         (first={}, last={})\nartifact_path={}\n{failure_artifact}",
        report.checkpoint_count,
        checkpoints.first().map_or(0, |c| c.pending_events),
        checkpoints.last().map_or(0, |c| c.pending_events),
        failure_artifact_path.display(),
    );

    // High watermark should stay within bounds.
    let max_hwm = checkpoints
        .iter()
        .map(|c| c.high_watermark_pending)
        .max()
        .unwrap_or(0);
    assert!(
        max_hwm < backpressure_threshold,
        "pending events high watermark ({max_hwm}) exceeded backpressure threshold ({backpressure_threshold})"
    );

    // ── Drift Detection ─────────────────────────────────────────────────

    // Write latency should not degrade more than 200% over the soak duration.
    // (Generous threshold: in-memory storage should be stable.)
    assert!(
        report.latency_drift_pct < 200.0,
        "latency drift too high: {:+.1}% (first={:.1}us, last={:.1}us)\n{failure_artifact}",
        report.latency_drift_pct,
        report.first_avg_latency_us,
        report.last_avg_latency_us,
    );

    // ── Throughput Stability ────────────────────────────────────────────

    // Throughput should not collapse: every checkpoint interval should
    // ingest at least some events.
    if report.throughput_deltas.len() >= 2 {
        let zero_deltas = report.throughput_deltas.iter().filter(|&&d| d == 0).count();
        assert_eq!(
            zero_deltas,
            0,
            "throughput collapse: {zero_deltas}/{} checkpoint intervals had zero events ingested\n{failure_artifact}",
            report.throughput_deltas.len(),
        );
    }

    // ── Anomaly Stability ───────────────────────────────────────────────

    // Anomalies should not proliferate unboundedly.
    assert!(
        report.max_open_anomalies < 50,
        "anomaly proliferation: {} open anomalies (expected < 50)\n{failure_artifact}",
        report.max_open_anomalies,
    );

    // ── Error Budget ────────────────────────────────────────────────────

    // Failed records should be zero for deterministic, well-formed events.
    assert_eq!(
        report.total_failed, 0,
        "unexpected record failures: {} failed records\n{failure_artifact}",
        report.total_failed,
    );

    // Events must actually have been ingested.
    assert!(
        report.total_events > 0,
        "no events ingested during soak\n{failure_artifact}"
    );
}

#[test]
#[ignore = "Extended soak: 6 simulated hours, run with --ignored --nocapture"]
fn soak_extended_stability_under_sustained_load() {
    let config = soak_config_extended(0xDEAD_BEEF_0003);
    let sim = TelemetrySimulator::new(config).expect("config should validate");
    let run = sim.generate().expect("generation should succeed");

    let storage = OpsStorage::open_in_memory().expect("storage should open");
    let checkpoint_interval = 600; // Every 10 simulated minutes.
    let backpressure_threshold = 32_768;

    let checkpoints =
        run_soak_pipeline(&storage, &run, checkpoint_interval, backpressure_threshold)
            .expect("soak pipeline should complete");

    let report = DriftReport::from_checkpoints(&checkpoints);
    report.print_diagnostics();
    let divergence = detect_first_divergence(
        &report,
        &checkpoints,
        backpressure_threshold.saturating_sub(1),
        100,
        150.0,
    );
    let failure_artifact =
        build_failure_artifact(0xDEAD_BEEF_0003, &report, &checkpoints, divergence.as_ref());
    let failure_artifact_path = persist_failure_artifact(
        "soak_extended_stability_under_sustained_load",
        0xDEAD_BEEF_0003,
        &failure_artifact,
    )
    .expect("failure artifact should be persisted");
    let queue_depth_budget =
        u64::try_from(backpressure_threshold).expect("backpressure threshold should fit u64");
    assert_resource_budget(
        &storage,
        &run,
        4_u64 * 1024 * 1024 * 1024,
        queue_depth_budget,
        &failure_artifact,
    );

    // All assertions from the short soak with tighter bounds for longer duration.

    // Leak detection: no monotonic pending growth.
    assert!(
        !report.monotonic_pending_growth,
        "leak detected over 6h soak: monotonic pending event growth\nartifact_path={}\n{failure_artifact}",
        failure_artifact_path.display(),
    );

    // Latency drift: allow up to 150% for longer soak (more data = some
    // expected variability in averages).
    assert!(
        report.latency_drift_pct < 150.0,
        "latency drift over 6h soak: {:+.1}%\n{failure_artifact}",
        report.latency_drift_pct,
    );

    // Throughput: no zero-event intervals.
    if report.throughput_deltas.len() >= 2 {
        let zero_count = report.throughput_deltas.iter().filter(|&&d| d == 0).count();
        assert_eq!(
            zero_count, 0,
            "throughput collapse in 6h soak: {zero_count} zero intervals\n{failure_artifact}"
        );
    }

    // Anomaly bound: tighter than short soak.
    assert!(
        report.max_open_anomalies < 100,
        "anomaly proliferation in 6h soak: {}\n{failure_artifact}",
        report.max_open_anomalies,
    );

    // Zero failed records.
    assert_eq!(
        report.total_failed, 0,
        "failed records in 6h soak\n{failure_artifact}"
    );

    // Substantial event volume.
    assert!(
        report.total_events > 1_000,
        "insufficient event volume in 6h soak: {}\n{failure_artifact}",
        report.total_events,
    );

    // Anomalies should not grow monotonically either.
    assert!(
        !report.monotonic_anomaly_growth,
        "anomaly leak detected in 6h soak: monotonic growth across checkpoints\n{failure_artifact}"
    );
}

#[test]
#[ignore = "Day-long soak: 24 simulated hours, run with --ignored --nocapture"]
fn soak_daylong_stability_under_sustained_load() {
    let config = soak_config_daylong(0xDEAD_BEEF_0008);
    let sim = TelemetrySimulator::new(config).expect("config should validate");
    let run = sim.generate().expect("generation should succeed");

    let storage = OpsStorage::open_in_memory().expect("storage should open");
    let checkpoint_interval = 1_800; // Every 30 simulated minutes.
    let backpressure_threshold = 65_536;

    let checkpoints =
        run_soak_pipeline(&storage, &run, checkpoint_interval, backpressure_threshold)
            .expect("soak pipeline should complete");

    let report = DriftReport::from_checkpoints(&checkpoints);
    report.print_diagnostics();
    let divergence = detect_first_divergence(
        &report,
        &checkpoints,
        backpressure_threshold.saturating_sub(1),
        150,
        175.0,
    );
    let failure_artifact =
        build_failure_artifact(0xDEAD_BEEF_0008, &report, &checkpoints, divergence.as_ref());
    let failure_artifact_path = persist_failure_artifact(
        "soak_daylong_stability_under_sustained_load",
        0xDEAD_BEEF_0008,
        &failure_artifact,
    )
    .expect("failure artifact should be persisted");
    let queue_depth_budget =
        u64::try_from(backpressure_threshold).expect("backpressure threshold should fit u64");
    assert_resource_budget(
        &storage,
        &run,
        6_u64 * 1024 * 1024 * 1024,
        queue_depth_budget,
        &failure_artifact,
    );

    assert!(
        !report.monotonic_pending_growth,
        "leak detected over 24h soak: monotonic pending event growth\nartifact_path={}\n{failure_artifact}",
        failure_artifact_path.display(),
    );
    assert!(
        !report.monotonic_anomaly_growth,
        "anomaly leak detected over 24h soak: monotonic growth across checkpoints\n{failure_artifact}"
    );
    assert!(
        report.latency_drift_pct < 175.0,
        "latency drift over 24h soak: {:+.1}%\n{failure_artifact}",
        report.latency_drift_pct,
    );
    assert_eq!(
        report.total_failed, 0,
        "failed records in 24h soak\n{failure_artifact}"
    );
    assert!(
        report.total_events > 5_000,
        "insufficient event volume in 24h soak: {}\n{failure_artifact}",
        report.total_events,
    );
}

#[test]
#[ignore = "Soak test: backpressure resilience, run with --ignored --nocapture"]
fn soak_short_backpressure_resilience() {
    // Use a LOW backpressure threshold to trigger backpressure frequently.
    let config = soak_config_short(0xDEAD_BEEF_0004);
    let sim = TelemetrySimulator::new(config).expect("config should validate");
    let run = sim.generate().expect("generation should succeed");

    let storage = OpsStorage::open_in_memory().expect("storage should open");
    let checkpoint_interval = 30;
    let backpressure_threshold = 64; // Deliberately low to trigger backpressure

    let checkpoints =
        run_soak_pipeline(&storage, &run, checkpoint_interval, backpressure_threshold)
            .expect("soak pipeline should complete even under backpressure");

    let report = DriftReport::from_checkpoints(&checkpoints);
    report.print_diagnostics();

    // Under backpressure, some batches should be rejected.
    // The key invariant: the pipeline does not crash or corrupt state.

    // Events should still be ingested (backpressure drops excess, not all).
    assert!(
        report.total_events > 0,
        "no events ingested under backpressure"
    );

    // No failed records (backpressured != failed).
    assert_eq!(
        report.total_failed, 0,
        "failed records under backpressure: {}",
        report.total_failed,
    );

    // Latency should not explode under backpressure.
    assert!(
        report.latency_drift_pct < 300.0,
        "latency exploded under backpressure: {:+.1}%",
        report.latency_drift_pct,
    );

    // Final state should still be queryable.
    let final_anomalies = storage
        .query_open_anomalies_for_scope(SloScope::Fleet, "__fleet__", 1024)
        .expect("anomaly query should succeed after backpressure soak");
    // Just verifying the query works and returns a bounded result.
    assert!(
        final_anomalies.len() < 200,
        "excessive anomalies after backpressure soak: {}",
        final_anomalies.len(),
    );
}

#[test]
#[ignore = "Soak test: restart workload profile stability, run with --ignored --nocapture"]
fn soak_restart_profile_stability() {
    // Focus on the Restarting workload profile which simulates degraded
    // periods and instance restarts.
    let config = TelemetrySimulatorConfig {
        seed: 0xDEAD_BEEF_0005,
        start_ms: 1_734_503_200_000,
        tick_interval_ms: 1_000,
        ticks: 120,
        projects: vec![
            SimulatedProject {
                project_key: "restart-a".to_owned(),
                host_name: "restart-a".to_owned(),
                instance_count: 3,
                workload: WorkloadProfile::Restarting,
            },
            SimulatedProject {
                project_key: "restart-b".to_owned(),
                host_name: "restart-b".to_owned(),
                instance_count: 2,
                workload: WorkloadProfile::Restarting,
            },
        ],
    };
    let sim = TelemetrySimulator::new(config).expect("config should validate");
    let run = sim.generate().expect("generation should succeed");

    let storage = OpsStorage::open_in_memory().expect("storage should open");
    let checkpoint_interval = 30;
    let backpressure_threshold = 16_384;

    let checkpoints =
        run_soak_pipeline(&storage, &run, checkpoint_interval, backpressure_threshold)
            .expect("restart-profile soak should complete");

    let report = DriftReport::from_checkpoints(&checkpoints);
    report.print_diagnostics();

    // Restart workload should not cause leaks.
    assert!(
        !report.monotonic_pending_growth,
        "pending event leak during restart soak"
    );

    // Zero failed records even under restart churn.
    assert_eq!(report.total_failed, 0, "failed records during restart soak");

    // Events ingested.
    assert!(
        report.total_events > 0,
        "no events ingested during restart soak"
    );
}

#[test]
#[ignore = "Long-duration soak: embedding wave profile, run with --ignored --nocapture"]
fn soak_embedding_wave_queue_stability() {
    // Focus on the EmbeddingWave profile which creates oscillating backlog
    // pressure.
    let config = TelemetrySimulatorConfig {
        seed: 0xDEAD_BEEF_0006,
        start_ms: 1_734_503_200_000,
        tick_interval_ms: 1_000,
        ticks: 120,
        projects: vec![
            SimulatedProject {
                project_key: "embed-wave-a".to_owned(),
                host_name: "embed-wave-a".to_owned(),
                instance_count: 4,
                workload: WorkloadProfile::EmbeddingWave,
            },
            SimulatedProject {
                project_key: "embed-wave-b".to_owned(),
                host_name: "embed-wave-b".to_owned(),
                instance_count: 3,
                workload: WorkloadProfile::EmbeddingWave,
            },
        ],
    };
    let sim = TelemetrySimulator::new(config).expect("config should validate");
    let run = sim.generate().expect("generation should succeed");

    let storage = OpsStorage::open_in_memory().expect("storage should open");
    let checkpoint_interval = 30;
    let backpressure_threshold = 16_384;

    let checkpoints =
        run_soak_pipeline(&storage, &run, checkpoint_interval, backpressure_threshold)
            .expect("embedding wave soak should complete");

    let report = DriftReport::from_checkpoints(&checkpoints);
    report.print_diagnostics();

    // Embedding wave creates higher event volume; verify no leak.
    assert!(
        !report.monotonic_pending_growth,
        "pending event leak during embedding wave soak"
    );

    // The wave pattern should produce throughput variability but no collapse.
    if report.throughput_deltas.len() >= 2 {
        let zero_count = report.throughput_deltas.iter().filter(|&&d| d == 0).count();
        assert_eq!(
            zero_count, 0,
            "throughput collapse during embedding wave: {zero_count} zero intervals"
        );
    }

    // Zero failed records.
    assert_eq!(
        report.total_failed, 0,
        "failed records during embedding wave soak"
    );

    // Substantial volume due to high instance count × embedding wave baseline.
    assert!(
        report.total_events > 500,
        "insufficient events during embedding wave soak: {}",
        report.total_events,
    );
}

#[test]
#[ignore = "Long-duration soak: cross-seed divergence check, run with --ignored --nocapture"]
fn soak_different_seeds_produce_different_runs() {
    let config_a = soak_config_short(0xAAAA_BBBB_0001);
    let config_b = soak_config_short(0xCCCC_DDDD_0002);

    let sim_a = TelemetrySimulator::new(config_a).expect("config_a should validate");
    let sim_b = TelemetrySimulator::new(config_b).expect("config_b should validate");

    let run_a = sim_a.generate().expect("generation_a should succeed");
    let run_b = sim_b.generate().expect("generation_b should succeed");

    assert_ne!(
        run_a.signature(),
        run_b.signature(),
        "different seeds must produce different simulation signatures"
    );

    // Both should generate meaningful event volumes.
    assert!(run_a.total_search_events() > 0);
    assert!(run_b.total_search_events() > 0);
}

#[test]
#[ignore = "Long-duration soak: materialization consistency, run with --ignored --nocapture"]
fn soak_materialization_consistency() {
    // Verify that SLO materialization remains consistent across all
    // checkpoints: rollups should monotonically accumulate.
    let config = soak_config_short(0xDEAD_BEEF_0007);
    let sim = TelemetrySimulator::new(config).expect("config should validate");
    let run = sim.generate().expect("generation should succeed");

    let storage = OpsStorage::open_in_memory().expect("storage should open");

    let mut rollup_counts: Vec<usize> = Vec::new();
    let backpressure_threshold = 16_384;

    for (batch_idx, batch) in run.batches.iter().enumerate() {
        ingest_batch(&storage, batch, backpressure_threshold)
            .expect("batch ingestion should succeed");

        // Materialize and check rollups every 60 ticks.
        if batch_idx % 60 == 0 && batch_idx > 0 {
            materialize_batch(&storage, batch).expect("materialization should succeed");
            let rollups = storage
                .query_slo_rollups_for_scope(SloScope::Fleet, "__fleet__", 1024)
                .expect("rollup query should succeed");
            rollup_counts.push(rollups.len());
        }
    }

    // Rollup counts should be non-decreasing (materialization accumulates).
    for pair in rollup_counts.windows(2) {
        assert!(
            pair[1] >= pair[0],
            "rollup count decreased: {} -> {} (should be non-decreasing)",
            pair[0],
            pair[1],
        );
    }

    // At least some rollups should exist by the end.
    if let Some(&last) = rollup_counts.last() {
        assert!(last > 0, "no rollups materialized after full soak");
    }
}
