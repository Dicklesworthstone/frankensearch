//! Core telemetry collectors and canonical event payloads.
//!
//! This module provides the runtime-safe collector layer used by control-plane
//! integrations. It is intentionally lightweight:
//!
//! - default sampling cadence is 1000ms (from the ops config contract)
//! - minimum allowed cadence is 100ms (guardrail against accidental hot polling)
//! - per-event overhead is O(1): one atomic increment plus payload assembly
//!
//! Collectors emit canonical `search`, `embedding`, `index`, and `resource`
//! payloads aligned with `schemas/telemetry-event-v1.schema.json`.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::error::{SearchError, SearchResult};
use crate::query_class::QueryClass;

/// Canonical telemetry envelope schema version.
pub const TELEMETRY_SCHEMA_VERSION: u8 = 1;

/// Default collector interval (1s) from `docs/ops-config-contract.md`.
pub const DEFAULT_COLLECTION_INTERVAL_MS: u64 = 1_000;

/// Minimum allowed interval to bound overhead.
pub const MIN_COLLECTION_INTERVAL_MS: u64 = 100;

/// Default in-memory stream buffer size for live search events.
pub const DEFAULT_SEARCH_STREAM_CAPACITY: usize = 1_024;

const EMPTY_QUERY_PLACEHOLDER: &str = "<empty>";
const MAX_QUERY_TEXT_CHARS: usize = 500;

/// Runtime collector configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollectorConfig {
    /// Telemetry collection interval in milliseconds.
    pub collection_interval_ms: u64,
}

impl Default for CollectorConfig {
    fn default() -> Self {
        Self {
            collection_interval_ms: DEFAULT_COLLECTION_INTERVAL_MS,
        }
    }
}

impl CollectorConfig {
    /// Validate collector settings.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when interval is below the
    /// minimum supported value.
    pub fn validate(&self) -> SearchResult<()> {
        if self.collection_interval_ms < MIN_COLLECTION_INTERVAL_MS {
            return Err(SearchError::InvalidConfig {
                field: "collector.collection_interval_ms".to_owned(),
                value: self.collection_interval_ms.to_string(),
                reason: format!(
                    "must be >= {MIN_COLLECTION_INTERVAL_MS}ms to bound collector overhead"
                ),
            });
        }
        Ok(())
    }

    /// Return the configured interval as [`Duration`].
    #[must_use]
    pub const fn interval(&self) -> Duration {
        Duration::from_millis(self.collection_interval_ms)
    }
}

/// Point-in-time collector counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollectorSnapshot {
    /// Configured collection interval in milliseconds.
    pub collection_interval_ms: u64,
    /// Number of search events emitted.
    pub search_events_emitted: u64,
    /// Number of embedding events emitted.
    pub embedding_events_emitted: u64,
    /// Number of index events emitted.
    pub index_events_emitted: u64,
    /// Number of resource events emitted.
    pub resource_events_emitted: u64,
    /// Total events emitted across all families.
    pub total_events_emitted: u64,
}

/// Bounded-buffer policy for live search streaming.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchStreamMode {
    /// Drop oldest queued event when buffer is full; never block producer.
    Lossy,
    /// Reject new event when buffer is full; caller must handle backpressure.
    NonLossy,
}

/// Configuration for the live search stream emitter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchStreamConfig {
    /// Maximum number of queued search events.
    pub capacity: usize,
    /// Backpressure/drop policy.
    pub mode: SearchStreamMode,
}

impl Default for SearchStreamConfig {
    fn default() -> Self {
        Self {
            capacity: DEFAULT_SEARCH_STREAM_CAPACITY,
            mode: SearchStreamMode::Lossy,
        }
    }
}

impl SearchStreamConfig {
    /// Validate stream configuration.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] if capacity is zero.
    pub fn validate(&self) -> SearchResult<()> {
        if self.capacity == 0 {
            return Err(SearchError::InvalidConfig {
                field: "search_stream.capacity".to_owned(),
                value: self.capacity.to_string(),
                reason: "must be >= 1 for bounded buffering".to_owned(),
            });
        }
        Ok(())
    }
}

/// A buffered live-search frame for timeline/live-feed consumers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LiveSearchFrame {
    /// Opaque resume cursor.
    pub cursor: String,
    /// Monotonic sequence number.
    pub sequence: u64,
    /// Number of events dropped since the prior emitted frame.
    pub dropped_since_last: u64,
    /// Canonical search telemetry event.
    pub event: TelemetryEnvelope,
}

/// Result of publishing one search event into the stream emitter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchStreamPublishOutcome {
    /// Cursor assigned to this frame.
    pub cursor: String,
    /// Sequence assigned to this frame.
    pub sequence: u64,
    /// Current queue depth after publish.
    pub buffered: usize,
    /// Number of dropped events attached to this frame.
    pub dropped_since_last: u64,
}

/// Point-in-time stream health counters for dashboards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchStreamHealth {
    /// Configured buffering mode.
    pub mode: SearchStreamMode,
    /// Maximum buffer capacity.
    pub capacity: usize,
    /// Current queued frame count.
    pub buffered: usize,
    /// Total successfully emitted frames.
    pub emitted_total: u64,
    /// Total dropped frames (lossy mode only).
    pub dropped_total: u64,
    /// Total rejected publishes due to full buffer (non-lossy mode).
    pub backpressure_rejections: u64,
}

/// Bounded live-search stream emitter with explicit drop/backpressure accounting.
///
/// This is intentionally synchronous and lightweight so pipeline code can publish
/// search events without async runtime coupling.
#[derive(Debug)]
pub struct LiveSearchStreamEmitter {
    config: SearchStreamConfig,
    queue: Mutex<VecDeque<LiveSearchFrame>>,
    next_sequence: AtomicU64,
    emitted_total: AtomicU64,
    dropped_total: AtomicU64,
    backpressure_rejections: AtomicU64,
    pending_dropped_since_last: AtomicU64,
}

impl Default for LiveSearchStreamEmitter {
    fn default() -> Self {
        Self {
            config: SearchStreamConfig::default(),
            queue: Mutex::new(VecDeque::new()),
            next_sequence: AtomicU64::new(0),
            emitted_total: AtomicU64::new(0),
            dropped_total: AtomicU64::new(0),
            backpressure_rejections: AtomicU64::new(0),
            pending_dropped_since_last: AtomicU64::new(0),
        }
    }
}

impl LiveSearchStreamEmitter {
    /// Create a stream emitter with explicit configuration.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] if `config` is invalid.
    pub fn new(config: SearchStreamConfig) -> SearchResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            ..Self::default()
        })
    }

    fn lock_queue(&self) -> MutexGuard<'_, VecDeque<LiveSearchFrame>> {
        match self.queue.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    /// Publish a canonical `search` event into the stream.
    ///
    /// In lossy mode, if full, the oldest queued frame is dropped.
    /// In non-lossy mode, if full, returns [`SearchError::QueueFull`].
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when `event` is not a `search` event.
    /// Returns [`SearchError::QueueFull`] when mode is non-lossy and queue is full.
    pub fn publish_search(
        &self,
        event: TelemetryEnvelope,
    ) -> SearchResult<SearchStreamPublishOutcome> {
        if !matches!(event.event, TelemetryEvent::Search { .. }) {
            return Err(SearchError::InvalidConfig {
                field: "search_stream.event.type".to_owned(),
                value: "non-search".to_owned(),
                reason: "live search stream only accepts search telemetry events".to_owned(),
            });
        }

        let mut queue = self.lock_queue();
        if queue.len() >= self.config.capacity {
            match self.config.mode {
                SearchStreamMode::Lossy => {
                    let _ = queue.pop_front();
                    self.dropped_total.fetch_add(1, Ordering::Relaxed);
                    self.pending_dropped_since_last
                        .fetch_add(1, Ordering::Relaxed);
                }
                SearchStreamMode::NonLossy => {
                    self.backpressure_rejections.fetch_add(1, Ordering::Relaxed);
                    return Err(SearchError::QueueFull {
                        pending: queue.len(),
                        capacity: self.config.capacity,
                    });
                }
            }
        }

        let sequence = self.next_sequence.fetch_add(1, Ordering::Relaxed);
        let cursor = format!("search-{sequence:020}");
        let dropped_since_last = self.pending_dropped_since_last.swap(0, Ordering::Relaxed);
        queue.push_back(LiveSearchFrame {
            cursor: cursor.clone(),
            sequence,
            dropped_since_last,
            event,
        });
        self.emitted_total.fetch_add(1, Ordering::Relaxed);

        Ok(SearchStreamPublishOutcome {
            cursor,
            sequence,
            buffered: queue.len(),
            dropped_since_last,
        })
    }

    /// Drain up to `max_items` frames from oldest to newest.
    #[must_use]
    pub fn drain(&self, max_items: usize) -> Vec<LiveSearchFrame> {
        if max_items == 0 {
            return Vec::new();
        }

        let mut queue = self.lock_queue();
        let take = max_items.min(queue.len());
        let mut drained = Vec::with_capacity(take);
        for _ in 0..take {
            if let Some(frame) = queue.pop_front() {
                drained.push(frame);
            }
        }
        drained
    }

    /// Snapshot stream health counters for timeline/live-feed diagnostics.
    #[must_use]
    pub fn health(&self) -> SearchStreamHealth {
        let buffered = self.lock_queue().len();
        SearchStreamHealth {
            mode: self.config.mode,
            capacity: self.config.capacity,
            buffered,
            emitted_total: self.emitted_total.load(Ordering::Relaxed),
            dropped_total: self.dropped_total.load(Ordering::Relaxed),
            backpressure_rejections: self.backpressure_rejections.load(Ordering::Relaxed),
        }
    }
}

/// O(1) runtime collector for canonical telemetry payloads.
#[derive(Debug)]
pub struct RuntimeMetricsCollector {
    config: CollectorConfig,
    search_events: AtomicU64,
    embedding_events: AtomicU64,
    index_events: AtomicU64,
    resource_events: AtomicU64,
}

impl Default for RuntimeMetricsCollector {
    fn default() -> Self {
        Self {
            config: CollectorConfig::default(),
            search_events: AtomicU64::new(0),
            embedding_events: AtomicU64::new(0),
            index_events: AtomicU64::new(0),
            resource_events: AtomicU64::new(0),
        }
    }
}

impl RuntimeMetricsCollector {
    /// Create a collector with explicit configuration.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when `config` is invalid.
    pub fn new(config: CollectorConfig) -> SearchResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            ..Self::default()
        })
    }

    /// Read-only access to collector configuration.
    #[must_use]
    pub const fn config(&self) -> CollectorConfig {
        self.config
    }

    /// Emit a canonical `search` event envelope.
    #[must_use]
    pub fn emit_search(
        &self,
        ts: impl Into<String>,
        instance: TelemetryInstance,
        correlation: TelemetryCorrelation,
        sample: SearchCollectorSample,
    ) -> TelemetryEnvelope {
        self.search_events.fetch_add(1, Ordering::Relaxed);
        let SearchCollectorSample {
            query_text,
            query_class,
            phase,
            result_count,
            lexical_count,
            semantic_count,
            latency_us,
            memory_bytes,
        } = sample;
        let query_text = sanitize_query_text(&query_text);

        TelemetryEnvelope::new(
            ts,
            TelemetryEvent::Search {
                instance,
                correlation,
                query: TelemetrySearchQuery {
                    text: query_text,
                    class: TelemetryQueryClass::from(query_class),
                    phase,
                },
                results: TelemetrySearchResults {
                    result_count,
                    lexical_count,
                    semantic_count,
                },
                metrics: TelemetrySearchMetrics {
                    latency_us,
                    memory_bytes,
                },
            },
        )
    }

    /// Emit a canonical `embedding` event envelope.
    #[must_use]
    pub fn emit_embedding(
        &self,
        ts: impl Into<String>,
        instance: TelemetryInstance,
        correlation: TelemetryCorrelation,
        sample: EmbeddingCollectorSample,
    ) -> TelemetryEnvelope {
        self.embedding_events.fetch_add(1, Ordering::Relaxed);

        TelemetryEnvelope::new(
            ts,
            TelemetryEvent::Embedding {
                instance,
                correlation,
                job: TelemetryEmbeddingJob {
                    job_id: sample.job_id,
                    queue_depth: sample.queue_depth,
                    doc_count: sample.doc_count,
                    stage: sample.stage,
                },
                embedder: TelemetryEmbedderInfo {
                    id: sample.embedder_id,
                    tier: sample.tier,
                    dimension: sample.dimension,
                },
                status: sample.status,
                duration_ms: sample.duration_ms,
            },
        )
    }

    /// Emit a canonical `index` event envelope.
    #[must_use]
    pub fn emit_index(
        &self,
        ts: impl Into<String>,
        instance: TelemetryInstance,
        correlation: TelemetryCorrelation,
        sample: IndexCollectorSample,
    ) -> TelemetryEnvelope {
        self.index_events.fetch_add(1, Ordering::Relaxed);

        TelemetryEnvelope::new(
            ts,
            TelemetryEvent::Index {
                instance,
                correlation,
                operation: sample.operation,
                inventory: sample.inventory,
                dimension: sample.dimension,
                quantization: sample.quantization,
                status: sample.status,
                duration_ms: sample.duration_ms,
            },
        )
    }

    /// Emit a canonical `resource` event envelope.
    #[must_use]
    pub fn emit_resource(
        &self,
        ts: impl Into<String>,
        instance: TelemetryInstance,
        correlation: TelemetryCorrelation,
        sample: ResourceCollectorSample,
    ) -> TelemetryEnvelope {
        self.resource_events.fetch_add(1, Ordering::Relaxed);
        let cpu_pct = sanitize_cpu_pct(sample.cpu_pct);
        let interval_ms = sample.interval_ms.max(1);
        let load_avg_1m = sample.load_avg_1m.and_then(normalize_non_negative_f64);

        TelemetryEnvelope::new(
            ts,
            TelemetryEvent::Resource {
                instance,
                correlation,
                sample: TelemetryResourceSample {
                    cpu_pct,
                    rss_bytes: sample.rss_bytes,
                    io_read_bytes: sample.io_read_bytes,
                    io_write_bytes: sample.io_write_bytes,
                    interval_ms,
                    load_avg_1m,
                    pressure_profile: sample.pressure_profile,
                },
            },
        )
    }

    /// Snapshot collector counters.
    #[must_use]
    pub fn snapshot(&self) -> CollectorSnapshot {
        let search_events = self.search_events.load(Ordering::Relaxed);
        let embedding_events = self.embedding_events.load(Ordering::Relaxed);
        let index_events = self.index_events.load(Ordering::Relaxed);
        let resource_events = self.resource_events.load(Ordering::Relaxed);

        CollectorSnapshot {
            collection_interval_ms: self.config.collection_interval_ms,
            search_events_emitted: search_events,
            embedding_events_emitted: embedding_events,
            index_events_emitted: index_events,
            resource_events_emitted: resource_events,
            total_events_emitted: search_events
                .saturating_add(embedding_events)
                .saturating_add(index_events)
                .saturating_add(resource_events),
        }
    }
}

/// Canonical telemetry instance identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TelemetryInstance {
    pub instance_id: String,
    pub project_key: String,
    pub host_name: String,
    pub pid: Option<u32>,
}

/// Canonical telemetry correlation IDs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TelemetryCorrelation {
    pub event_id: String,
    pub root_request_id: String,
    pub parent_event_id: Option<String>,
}

/// Telemetry envelope used by the control plane.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TelemetryEnvelope {
    pub v: u8,
    pub ts: String,
    pub event: TelemetryEvent,
}

impl TelemetryEnvelope {
    /// Build a v1 telemetry envelope.
    #[must_use]
    pub fn new(ts: impl Into<String>, event: TelemetryEvent) -> Self {
        Self {
            v: TELEMETRY_SCHEMA_VERSION,
            ts: ts.into(),
            event,
        }
    }
}

/// Canonical telemetry event families.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TelemetryEvent {
    Search {
        instance: TelemetryInstance,
        correlation: TelemetryCorrelation,
        query: TelemetrySearchQuery,
        results: TelemetrySearchResults,
        metrics: TelemetrySearchMetrics,
    },
    Embedding {
        instance: TelemetryInstance,
        correlation: TelemetryCorrelation,
        job: TelemetryEmbeddingJob,
        embedder: TelemetryEmbedderInfo,
        status: EmbeddingStatus,
        duration_ms: u64,
    },
    Index {
        instance: TelemetryInstance,
        correlation: TelemetryCorrelation,
        operation: IndexOperation,
        inventory: IndexInventory,
        dimension: usize,
        quantization: QuantizationMode,
        status: IndexStatus,
        duration_ms: u64,
    },
    Resource {
        instance: TelemetryInstance,
        correlation: TelemetryCorrelation,
        sample: TelemetryResourceSample,
    },
    Lifecycle {
        instance: TelemetryInstance,
        correlation: TelemetryCorrelation,
        state: LifecycleState,
        severity: LifecycleSeverity,
        reason: Option<String>,
        uptime_ms: Option<u64>,
    },
}

/// Search query sub-payload for `search` events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TelemetrySearchQuery {
    pub text: String,
    pub class: TelemetryQueryClass,
    pub phase: SearchEventPhase,
}

/// Search result-count sub-payload for `search` events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TelemetrySearchResults {
    pub result_count: usize,
    pub lexical_count: usize,
    pub semantic_count: usize,
}

/// Search metric sub-payload for `search` events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TelemetrySearchMetrics {
    pub latency_us: u64,
    pub memory_bytes: Option<u64>,
}

/// Embedding job sub-payload for `embedding` events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TelemetryEmbeddingJob {
    pub job_id: String,
    pub queue_depth: usize,
    pub doc_count: usize,
    pub stage: EmbeddingStage,
}

/// Embedder identity sub-payload for `embedding` events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TelemetryEmbedderInfo {
    pub id: String,
    pub tier: EmbedderTier,
    pub dimension: usize,
}

/// Index inventory sub-payload for `index` events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexInventory {
    pub words: u64,
    pub tokens: u64,
    pub lines: u64,
    pub bytes: u64,
    pub docs: u64,
}

/// Resource sample sub-payload for `resource` events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TelemetryResourceSample {
    pub cpu_pct: f64,
    pub rss_bytes: u64,
    pub io_read_bytes: u64,
    pub io_write_bytes: u64,
    pub interval_ms: u64,
    pub load_avg_1m: Option<f64>,
    pub pressure_profile: Option<PressureProfile>,
}

/// Query class values normalized for telemetry contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TelemetryQueryClass {
    Empty,
    Identifier,
    ShortKeyword,
    NaturalLanguage,
}

impl From<QueryClass> for TelemetryQueryClass {
    fn from(value: QueryClass) -> Self {
        match value {
            QueryClass::Empty => Self::Empty,
            QueryClass::Identifier => Self::Identifier,
            QueryClass::ShortKeyword => Self::ShortKeyword,
            QueryClass::NaturalLanguage => Self::NaturalLanguage,
        }
    }
}

/// Search phase values for telemetry contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchEventPhase {
    Initial,
    Refined,
    RefinementFailed,
}

/// Embedding job stage values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingStage {
    Fast,
    Quality,
    Background,
}

/// Embedding tier values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmbedderTier {
    Hash,
    Fast,
    Quality,
}

/// Embedding lifecycle status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Index operation kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexOperation {
    Build,
    Rebuild,
    Append,
    Compact,
    Repair,
    Snapshot,
}

/// Quantization modes used by index telemetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuantizationMode {
    F32,
    F16,
    Int8,
    Pq,
}

/// Index operation status values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexStatus {
    Started,
    Completed,
    Failed,
}

/// Resource pressure profile values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PressureProfile {
    Strict,
    Performance,
    Degraded,
}

/// Lifecycle event state values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LifecycleState {
    Started,
    Stopped,
    Healthy,
    Degraded,
    Stale,
    Recovering,
}

/// Lifecycle severity values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LifecycleSeverity {
    Info,
    Warn,
    Error,
}

/// Input sample for `search` event collection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchCollectorSample {
    pub query_text: String,
    pub query_class: QueryClass,
    pub phase: SearchEventPhase,
    pub result_count: usize,
    pub lexical_count: usize,
    pub semantic_count: usize,
    pub latency_us: u64,
    pub memory_bytes: Option<u64>,
}

/// Input sample for `embedding` event collection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingCollectorSample {
    pub job_id: String,
    pub queue_depth: usize,
    pub doc_count: usize,
    pub stage: EmbeddingStage,
    pub embedder_id: String,
    pub tier: EmbedderTier,
    pub dimension: usize,
    pub status: EmbeddingStatus,
    pub duration_ms: u64,
}

/// Input sample for `index` event collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexCollectorSample {
    pub operation: IndexOperation,
    pub inventory: IndexInventory,
    pub dimension: usize,
    pub quantization: QuantizationMode,
    pub status: IndexStatus,
    pub duration_ms: u64,
}

/// Input sample for `resource` event collection.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ResourceCollectorSample {
    pub cpu_pct: f64,
    pub rss_bytes: u64,
    pub io_read_bytes: u64,
    pub io_write_bytes: u64,
    pub interval_ms: u64,
    pub load_avg_1m: Option<f64>,
    pub pressure_profile: Option<PressureProfile>,
}

fn sanitize_query_text(query_text: &str) -> String {
    let trimmed = query_text.trim();
    if trimmed.is_empty() {
        return EMPTY_QUERY_PLACEHOLDER.to_owned();
    }
    trimmed.chars().take(MAX_QUERY_TEXT_CHARS).collect()
}

const fn sanitize_cpu_pct(cpu_pct: f64) -> f64 {
    if !cpu_pct.is_finite() {
        return 0.0;
    }
    cpu_pct.clamp(0.0, 100.0)
}

fn normalize_non_negative_f64(value: f64) -> Option<f64> {
    if value.is_finite() && value >= 0.0 {
        Some(value)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn fixture_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../schemas/fixtures")
            .join(name)
    }

    fn parse_fixture(name: &str) -> TelemetryEnvelope {
        let path = fixture_path(name);
        let body = std::fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("read fixture {}", path.display()));
        serde_json::from_str::<TelemetryEnvelope>(&body)
            .unwrap_or_else(|_| panic!("deserialize fixture {}", path.display()))
    }

    fn instance() -> TelemetryInstance {
        TelemetryInstance {
            instance_id: "01JAH9A2W8F8Q6GQ4C7M3N2P1R".to_owned(),
            project_key: "/data/projects/frankensearch".to_owned(),
            host_name: "atlas".to_owned(),
            pid: Some(4242),
        }
    }

    fn correlation() -> TelemetryCorrelation {
        TelemetryCorrelation {
            event_id: "01JAH9A2X1K2M3N4P5Q6R7S8T9".to_owned(),
            root_request_id: "01JAH9A2WZZZZZZZZZZZZZZZZZ".to_owned(),
            parent_event_id: None,
        }
    }

    fn search_event(query_text: &str, latency_us: u64) -> TelemetryEnvelope {
        RuntimeMetricsCollector::default().emit_search(
            "2026-02-14T00:45:00Z",
            instance(),
            correlation(),
            SearchCollectorSample {
                query_text: query_text.to_owned(),
                query_class: QueryClass::NaturalLanguage,
                phase: SearchEventPhase::Initial,
                result_count: 10,
                lexical_count: 18,
                semantic_count: 30,
                latency_us,
                memory_bytes: Some(1_835_008),
            },
        )
    }

    #[test]
    fn config_default_matches_ops_contract() {
        let cfg = CollectorConfig::default();
        assert_eq!(cfg.collection_interval_ms, DEFAULT_COLLECTION_INTERVAL_MS);
        assert_eq!(cfg.interval(), Duration::from_secs(1));
        cfg.validate().expect("default should be valid");
    }

    #[test]
    fn config_rejects_too_fast_interval() {
        let cfg = CollectorConfig {
            collection_interval_ms: 99,
        };
        let err = cfg.validate().expect_err("interval below min must fail");
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[test]
    fn collector_emits_search_event_with_canonical_fields() {
        let collector = RuntimeMetricsCollector::default();
        let sample = SearchCollectorSample {
            query_text: "distributed consensus".to_owned(),
            query_class: QueryClass::NaturalLanguage,
            phase: SearchEventPhase::Initial,
            result_count: 10,
            lexical_count: 18,
            semantic_count: 30,
            latency_us: 8_421,
            memory_bytes: Some(1_835_008),
        };

        let envelope =
            collector.emit_search("2026-02-14T00:45:00Z", instance(), correlation(), sample);
        let value = serde_json::to_value(&envelope).expect("serialize");

        assert_eq!(value["v"], 1);
        assert_eq!(value["event"]["type"], "search");
        assert_eq!(value["event"]["query"]["class"], "natural_language");
        assert_eq!(value["event"]["query"]["phase"], "initial");
        assert_eq!(value["event"]["metrics"]["latency_us"], 8_421);

        let snapshot = collector.snapshot();
        assert_eq!(snapshot.search_events_emitted, 1);
        assert_eq!(snapshot.total_events_emitted, 1);
    }

    #[test]
    fn collector_emits_embedding_index_resource_events() {
        let collector = RuntimeMetricsCollector::default();
        let ts = "2026-02-14T00:45:00Z";
        let base_instance = instance();
        let base_correlation = correlation();

        let embedding = collector.emit_embedding(
            ts,
            base_instance.clone(),
            base_correlation.clone(),
            EmbeddingCollectorSample {
                job_id: "01JAH9A31K2M3N4P5Q6R7S8T9W".to_owned(),
                queue_depth: 12,
                doc_count: 32,
                stage: EmbeddingStage::Background,
                embedder_id: "potion-multilingual-128M".to_owned(),
                tier: EmbedderTier::Fast,
                dimension: 256,
                status: EmbeddingStatus::Completed,
                duration_ms: 14,
            },
        );
        assert!(matches!(embedding.event, TelemetryEvent::Embedding { .. }));

        let index = collector.emit_index(
            ts,
            base_instance.clone(),
            base_correlation.clone(),
            IndexCollectorSample {
                operation: IndexOperation::Rebuild,
                inventory: IndexInventory {
                    words: 245_781,
                    tokens: 331_992,
                    lines: 58_740,
                    bytes: 19_833_421,
                    docs: 12_034,
                },
                dimension: 384,
                quantization: QuantizationMode::F16,
                status: IndexStatus::Completed,
                duration_ms: 18_423,
            },
        );
        assert!(matches!(index.event, TelemetryEvent::Index { .. }));

        let resource = collector.emit_resource(
            ts,
            base_instance,
            base_correlation,
            ResourceCollectorSample {
                cpu_pct: 142.0,
                rss_bytes: 73_400_320,
                io_read_bytes: 993_421,
                io_write_bytes: 341_204,
                interval_ms: 0,
                load_avg_1m: Some(1.43),
                pressure_profile: Some(PressureProfile::Performance),
            },
        );

        let value = serde_json::to_value(resource).expect("serialize");
        assert_eq!(value["event"]["type"], "resource");
        assert_eq!(value["event"]["sample"]["cpu_pct"], 100.0);
        assert_eq!(value["event"]["sample"]["interval_ms"], 1);

        let snapshot = collector.snapshot();
        assert_eq!(snapshot.embedding_events_emitted, 1);
        assert_eq!(snapshot.index_events_emitted, 1);
        assert_eq!(snapshot.resource_events_emitted, 1);
        assert_eq!(snapshot.total_events_emitted, 3);
    }

    #[test]
    fn empty_query_is_normalized_for_schema() {
        let collector = RuntimeMetricsCollector::default();
        let envelope = collector.emit_search(
            "2026-02-14T00:45:00Z",
            instance(),
            correlation(),
            SearchCollectorSample {
                query_text: "   ".to_owned(),
                query_class: QueryClass::Empty,
                phase: SearchEventPhase::Initial,
                result_count: 0,
                lexical_count: 0,
                semantic_count: 0,
                latency_us: 0,
                memory_bytes: None,
            },
        );
        let value = serde_json::to_value(envelope).expect("serialize");
        assert_eq!(value["event"]["query"]["text"], "<empty>");
        assert_eq!(value["event"]["query"]["class"], "empty");
    }

    #[test]
    fn search_stream_lossy_mode_drops_oldest_with_accounting() {
        let emitter = LiveSearchStreamEmitter::new(SearchStreamConfig {
            capacity: 1,
            mode: SearchStreamMode::Lossy,
        })
        .expect("valid stream config");

        let first = emitter
            .publish_search(search_event("first query", 111))
            .expect("first publish should succeed");
        assert_eq!(first.sequence, 0);
        assert_eq!(first.dropped_since_last, 0);

        let second = emitter
            .publish_search(search_event("second query", 222))
            .expect("second publish should succeed in lossy mode");
        assert_eq!(second.sequence, 1);
        assert_eq!(second.dropped_since_last, 1);

        let drained = emitter.drain(10);
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].sequence, 1);
        assert_eq!(drained[0].dropped_since_last, 1);

        let health = emitter.health();
        assert_eq!(health.mode, SearchStreamMode::Lossy);
        assert_eq!(health.capacity, 1);
        assert_eq!(health.emitted_total, 2);
        assert_eq!(health.dropped_total, 1);
        assert_eq!(health.backpressure_rejections, 0);
    }

    #[test]
    fn search_stream_non_lossy_mode_reports_backpressure() {
        let emitter = LiveSearchStreamEmitter::new(SearchStreamConfig {
            capacity: 1,
            mode: SearchStreamMode::NonLossy,
        })
        .expect("valid stream config");

        emitter
            .publish_search(search_event("first query", 111))
            .expect("first publish should succeed");

        let err = emitter
            .publish_search(search_event("second query", 222))
            .expect_err("second publish should fail in non-lossy mode");
        assert!(matches!(
            err,
            SearchError::QueueFull {
                pending: 1,
                capacity: 1
            }
        ));

        let health = emitter.health();
        assert_eq!(health.emitted_total, 1);
        assert_eq!(health.dropped_total, 0);
        assert_eq!(health.backpressure_rejections, 1);
    }

    #[test]
    fn search_stream_rejects_non_search_events() {
        let emitter = LiveSearchStreamEmitter::default();
        let non_search = RuntimeMetricsCollector::default().emit_index(
            "2026-02-14T00:45:05Z",
            instance(),
            correlation(),
            IndexCollectorSample {
                operation: IndexOperation::Rebuild,
                inventory: IndexInventory {
                    words: 1,
                    tokens: 1,
                    lines: 1,
                    bytes: 1,
                    docs: 1,
                },
                dimension: 384,
                quantization: QuantizationMode::F16,
                status: IndexStatus::Completed,
                duration_ms: 10,
            },
        );

        let err = emitter
            .publish_search(non_search)
            .expect_err("non-search event should be rejected");
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[test]
    fn search_stream_preserves_search_metrics_and_correlation() {
        let emitter = LiveSearchStreamEmitter::default();
        let event = search_event("correlation query", 8421);
        emitter
            .publish_search(event)
            .expect("publish should succeed in default mode");

        let drained = emitter.drain(1);
        assert_eq!(drained.len(), 1);
        let frame = &drained[0];
        assert_eq!(frame.sequence, 0);

        if let TelemetryEvent::Search {
            correlation,
            metrics,
            ..
        } = &frame.event.event
        {
            assert_eq!(correlation.event_id, "01JAH9A2X1K2M3N4P5Q6R7S8T9");
            assert_eq!(metrics.latency_us, 8421);
            assert_eq!(metrics.memory_bytes, Some(1_835_008));
        } else {
            panic!("expected search event");
        }
    }

    #[test]
    fn search_stream_config_rejects_zero_capacity() {
        let cfg = SearchStreamConfig {
            capacity: 0,
            mode: SearchStreamMode::Lossy,
        };
        let err = cfg.validate().expect_err("zero capacity must fail");
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[test]
    fn telemetry_event_fixtures_deserialize() {
        for fixture in [
            "telemetry-search-v1.json",
            "telemetry-embedding-v1.json",
            "telemetry-index-v1.json",
            "telemetry-resource-v1.json",
            "telemetry-lifecycle-v1.json",
        ] {
            let envelope = parse_fixture(fixture);
            assert_eq!(envelope.v, TELEMETRY_SCHEMA_VERSION);
        }
    }
}
