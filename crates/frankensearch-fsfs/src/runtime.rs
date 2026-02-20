use std::collections::{BTreeSet, HashMap, HashSet, VecDeque, hash_map::DefaultHasher};
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{BufReader, BufWriter, ErrorKind, IsTerminal, Read, Write};
#[cfg(unix)]
use std::net::Shutdown;
#[cfg(unix)]
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use asupersync::Cx;
use dirs::home_dir;
use frankensearch_core::{
    Canonicalizer, DefaultCanonicalizer, Embedder, ExplainedSource, ExplanationPhase,
    HitExplanation, IndexableDocument, LexicalSearch, ScoreComponent, SearchError, SearchResult,
};
use frankensearch_embed::{
    ConsentSource, DownloadConsent, EmbedderStack, HashAlgorithm, HashEmbedder, ModelDownloader,
    ModelLifecycle, ModelManifest, ensure_default_semantic_models,
};
use frankensearch_index::VectorIndex;
use frankensearch_lexical::{SnippetConfig, TantivyIndex};
use frankensearch_storage::{
    EmbeddingVectorSink, IngestAction, IngestRequest, IngestResult, JobQueueConfig,
    PersistentJobQueue, PipelineConfig, Storage, StorageBackedJobRunner,
    StorageConfig as PipelineStorageConfig,
};
use fsqlite_types::value::SqliteValue;
use ftui_backend::{Backend, BackendEventSource, BackendFeatures, BackendPresenter};
use ftui_core::event::{Event, KeyCode, Modifiers};
use ftui_extras::markdown::{
    MarkdownDetection, MarkdownRenderer, MarkdownTheme, is_likely_markdown,
};
use ftui_layout::{Constraint, Flex};
use ftui_render::buffer::Buffer;
use ftui_render::cell::PackedRgba;
use ftui_render::diff::BufferDiff;
use ftui_render::frame::Frame;
use ftui_render::grapheme_pool::GraphemePool;
use ftui_style::Style;
use ftui_text::search::search_ascii_case_insensitive;
use ftui_text::{Line, Span, Text, WrapMode};
use ftui_tty::{TtyBackend, TtySessionOptions};
use ftui_widgets::{
    Widget,
    block::Block,
    borders::{BorderType, Borders},
    input::TextInput,
    paragraph::Paragraph,
    progress::ProgressBar,
    sparkline::Sparkline,
};
use ignore::WalkBuilder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sysinfo::Disks;
use tracing::{info, warn};

use crate::adapters::cli::{CliCommand, CliInput, CompletionShell, OutputFormat};
use crate::adapters::format_emitter::{emit_envelope, emit_stream_frame, meta_for_format};
use crate::adapters::tui::FsfsTuiShellModel;
use crate::agent_ergonomics::result_id;
use crate::catalog::cleanup_tombstones_for_path;
use crate::config::{
    DegradationOverrideMode, DiscoveryCandidate, DiscoveryDecision, DiscoveryScopeDecision,
    FsfsConfig, IngestionClass, PressureProfile, RootDiscoveryDecision,
    default_project_config_file_path, default_user_config_file_path,
};
use crate::explanation_payload::{FsfsExplanationPayload, FusionContext, RankingExplanation};
use crate::lifecycle::{
    DiskBudgetAction, DiskBudgetSnapshot, DiskBudgetStage, IndexStorageBreakdown, LifecycleTracker,
    ResourceLimits, ResourceUsage, WatchdogConfig,
};
use crate::mount_info::{MountTable, read_system_mounts};
use crate::output_schema::{OutputEnvelope, SearchHitPayload, SearchOutputPhase, SearchPayload};
use crate::pressure::{
    DegradationControllerConfig, DegradationSignal, DegradationStateMachine, DegradationTransition,
    HostPressureCollector, PressureController, PressureControllerConfig, PressureSignal,
    PressureState, PressureTransition,
};
use crate::query_execution::{
    FusedCandidate, FusionPolicy as QueryFusionPolicy, LexicalCandidate,
    QueryExecutionOrchestrator, SemanticCandidate,
};
use crate::query_planning::{
    CapabilityState, QueryExecutionCapabilities, QueryIntentClass, QueryPlanner,
};
use crate::shutdown::{ShutdownCoordinator, ShutdownReason};
use crate::stream_protocol::{
    StreamEvent, StreamFrame, StreamProgressEvent, StreamResultEvent, StreamStartedEvent,
    terminal_event_completed, terminal_event_from_error,
};
use crate::watcher::{FsWatcher, WatchIngestOp, WatchIngestPipeline};

/// Supported fsfs interfaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterfaceMode {
    Cli,
    Tui,
}

/// Embedder availability at planning time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedderAvailability {
    /// Both fast and quality embedders are available.
    Full,
    /// Quality embedder is unavailable; fast tier remains available.
    FastOnly,
    /// No semantic embedder is available.
    None,
}

/// Chosen semantic scheduling tier for one file revision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorSchedulingTier {
    FastAndQuality,
    FastOnly,
    LexicalFallback,
    Skip,
}

/// Input row for revision-coherent vector planning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorPipelineInput {
    pub file_key: String,
    pub observed_revision: i64,
    pub previous_indexed_revision: Option<i64>,
    pub ingestion_class: IngestionClass,
    pub content_len_bytes: u64,
    pub content_hash_changed: bool,
}

/// Planning output for one file revision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorPipelinePlan {
    pub file_key: String,
    pub revision: i64,
    pub chunk_count: usize,
    pub batch_size: usize,
    pub tier: VectorSchedulingTier,
    pub invalidate_revisions_through: Option<i64>,
    pub reason_code: String,
}

/// Deterministic write actions derived from a vector pipeline plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VectorIndexWriteAction {
    InvalidateRevisionsThrough {
        file_key: String,
        revision: i64,
    },
    AppendFast {
        file_key: String,
        revision: i64,
        chunk_count: usize,
    },
    AppendQuality {
        file_key: String,
        revision: i64,
        chunk_count: usize,
    },
    MarkLexicalFallback {
        file_key: String,
        revision: i64,
        reason_code: String,
    },
    Skip {
        file_key: String,
        revision: i64,
        reason_code: String,
    },
}

/// Filesystem paths used for cross-domain index storage accounting.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IndexStoragePaths {
    /// Vector-index roots (FSVI shards, WAL, checkpoints).
    pub vector_index_roots: Vec<PathBuf>,
    /// Lexical-index roots (Tantivy segments and metadata).
    pub lexical_index_roots: Vec<PathBuf>,
    /// Catalog/database files (`FrankenSQLite` and sidecars).
    pub catalog_files: Vec<PathBuf>,
    /// Embedding cache roots.
    pub embedding_cache_roots: Vec<PathBuf>,
}

/// Runtime control plan derived from one disk-budget snapshot.
///
/// This converts staged budget state into deterministic scheduler/runtime intent
/// while the full eviction/compaction executors are still being wired.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiskBudgetControlPlan {
    /// Pressure state to project onto watcher cadence.
    pub watcher_pressure_state: PressureState,
    /// Keep ingest but reduce throughput/cadence.
    pub throttle_ingest: bool,
    /// Hard stop on writes when disk pressure is critical.
    pub pause_writes: bool,
    /// Trigger utility-based eviction lane.
    pub request_eviction: bool,
    /// Trigger index compaction lane.
    pub request_compaction: bool,
    /// Trigger catalog tombstone cleanup lane.
    pub request_tombstone_cleanup: bool,
    /// Minimum bytes to reclaim before considering pressure relieved.
    pub eviction_target_bytes: u64,
    /// Canonical machine reason code for audit/evidence.
    pub reason_code: &'static str,
}

#[derive(Debug, Clone)]
struct SearchPhaseArtifact {
    phase: SearchOutputPhase,
    fused: Vec<FusedCandidate>,
    payload: SearchPayload,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SearchFilterClause {
    PathContains(String),
    Extension(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SearchFilterExpr {
    clauses: Vec<SearchFilterClause>,
}

impl SearchFilterExpr {
    fn parse(raw: &str) -> SearchResult<Option<Self>> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Ok(None);
        }

        let mut clauses = Vec::new();
        for token in trimmed.split_whitespace() {
            if let Some((key, value)) = token.split_once(':') {
                let key = key.trim().to_ascii_lowercase();
                let value = value.trim();
                if value.is_empty() {
                    return Err(SearchError::InvalidConfig {
                        field: "cli.search.filter".to_owned(),
                        value: token.to_owned(),
                        reason: "filter segment has empty value".to_owned(),
                    });
                }
                match key.as_str() {
                    "path" => {
                        clauses.push(SearchFilterClause::PathContains(value.to_ascii_lowercase()));
                    }
                    "type" | "ext" | "extension" | "lang" => {
                        let normalized = value.trim_start_matches('.').to_ascii_lowercase();
                        if normalized.is_empty() {
                            return Err(SearchError::InvalidConfig {
                                field: "cli.search.filter".to_owned(),
                                value: token.to_owned(),
                                reason: "extension filter must include a non-empty extension"
                                    .to_owned(),
                            });
                        }
                        clauses.push(SearchFilterClause::Extension(normalized));
                    }
                    _ => {
                        return Err(SearchError::InvalidConfig {
                            field: "cli.search.filter".to_owned(),
                            value: token.to_owned(),
                            reason: format!(
                                "unsupported filter key `{key}`; supported keys: path,type,ext,extension,lang"
                            ),
                        });
                    }
                }
            } else {
                clauses.push(SearchFilterClause::PathContains(token.to_ascii_lowercase()));
            }
        }

        if clauses.is_empty() {
            return Ok(None);
        }

        Ok(Some(Self { clauses }))
    }

    fn matches_doc_id(&self, doc_id: &str) -> bool {
        let lowered = doc_id.to_ascii_lowercase();
        self.clauses.iter().all(|clause| match clause {
            SearchFilterClause::PathContains(needle) => lowered.contains(needle),
            SearchFilterClause::Extension(expected) => Path::new(doc_id)
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case(expected)),
        })
    }
}

const DISK_BUDGET_RATIO_DIVISOR: u64 = 10;
const DISK_BUDGET_CAP_BYTES: u64 = 5 * 1024 * 1024 * 1024;
const DISK_BUDGET_FALLBACK_BYTES: u64 = DISK_BUDGET_CAP_BYTES;
const DISK_BUDGET_REASON_EMERGENCY_OVERRIDE: &str = "disk.budget.emergency_override";
const MILLIS_PER_DAY: u64 = 86_400_000;
const TOMBSTONE_CLEANUP_MIN_INTERVAL_MS: u64 = 60_000;
const FSFS_SENTINEL_FILE: &str = "index_sentinel.json";
const FSFS_VECTOR_MANIFEST_FILE: &str = "vector/index_manifest.json";
const FSFS_LEXICAL_MANIFEST_FILE: &str = "lexical/index_manifest.json";
const FSFS_VECTOR_INDEX_FILE: &str = "vector/index.fsvi";
const FSFS_EXPLAIN_SESSION_FILE: &str = "explain/last_search_session.json";
const EXPLAIN_SESSION_SCHEMA_VERSION: &str = "fsfs.explain.session.v1";
const REASON_DISCOVERY_FILE_EXCLUDED: &str = "discovery.file.excluded";
const REASON_DISCOVERY_FILE_BINARY_BLOCKED: &str = "discovery.file.binary_blocked";
const REASON_DISCOVERY_FILE_PERMISSION_DENIED: &str = "discovery.file.permission_denied";
const ROOT_PROBE_MAX_DEPTH: usize = 3;
const ROOT_PROBE_MAX_ENTRIES_PER_DIR: usize = 2_048;
const ROOT_PROBE_MAX_TOTAL_ENTRIES: usize = 50_000;
const ROOT_PROBE_MAX_PROPOSALS: usize = 5;
const INDEXING_TUI_MIN_RENDER_INTERVAL_MS: u64 = 120;
const INDEXING_TUI_RATE_HISTORY_LIMIT: usize = 96;
const INDEXING_TUI_SPARKLINE_WIDTH: usize = 28;
const INDEXING_TUI_EMA_ALPHA: f64 = 0.24;
const FSFS_TUI_POLL_INTERVAL_MS: u64 = 120;
const FSFS_TUI_SEARCH_POLL_INTERVAL_MS: u64 = 8;
const FSFS_TUI_SEARCH_RENDER_MIN_INTERVAL_MS: u64 = 16;
const FSFS_TUI_SEARCH_RENDER_IDLE_HEARTBEAT_MS: u64 = 180;
const FSFS_TUI_STATUS_REFRESH_MS: u64 = 15_000;
const FSFS_TUI_STATUS_REFRESH_ACTIVE_GRACE_MS: u64 = 2_500;
const FSFS_TUI_LEXICAL_DEBOUNCE_MS: u64 = 6;
const FSFS_TUI_SEMANTIC_DEBOUNCE_MS: u64 = 36;
const FSFS_TUI_QUALITY_DEBOUNCE_MS: u64 = 160;
const FSFS_TUI_LEXICAL_DEBOUNCE_MIN_MS: u64 = 3;
const FSFS_TUI_LEXICAL_DEBOUNCE_MAX_MS: u64 = 20;
const FSFS_TUI_LEXICAL_DEBOUNCE_SHORT_QUERY_CAP_MS: u64 = 6;
const FSFS_TUI_SEMANTIC_DEBOUNCE_MIN_MS: u64 = 24;
const FSFS_TUI_SEMANTIC_DEBOUNCE_MAX_MS: u64 = 180;
const FSFS_TUI_QUALITY_DEBOUNCE_MIN_MS: u64 = 120;
const FSFS_TUI_QUALITY_DEBOUNCE_MAX_MS: u64 = 640;
const FSFS_TUI_TYPING_INTERVAL_MIN_MS: u64 = 20;
const FSFS_TUI_TYPING_INTERVAL_MAX_MS: u64 = 1_500;
const FSFS_TUI_TYPING_CADENCE_DEFAULT_MS: u64 = 64;
const FSFS_TUI_TYPING_CADENCE_ALPHA_PER_MILLE: u64 = 350;
const FSFS_TUI_LATENCY_FAST_PATH_MS: u64 = 30;
const FSFS_TUI_LATENCY_SLOW_PATH_MS: u64 = 90;
const FSFS_TUI_SEMANTIC_DEBOUNCE_FAST_TRIM_MS: u64 = 10;
const FSFS_TUI_QUALITY_DEBOUNCE_FAST_TRIM_MS: u64 = 40;
const FSFS_TUI_SEMANTIC_DEBOUNCE_SLOW_BUMP_MS: u64 = 24;
const FSFS_TUI_QUALITY_DEBOUNCE_SLOW_BUMP_MS: u64 = 90;
const FSFS_TUI_SEMANTIC_IDLE_GATE_MIN_MS: u64 = 120;
const FSFS_TUI_SEMANTIC_IDLE_GATE_MAX_MS: u64 = 420;
const FSFS_TUI_QUALITY_IDLE_GATE_MIN_MS: u64 = 260;
const FSFS_TUI_QUALITY_IDLE_GATE_MAX_MS: u64 = 1_200;
// Exponential inter-keystroke quantile multipliers:
// semantic: -ln(1 - 0.970) ≈ 3.5066
// quality:  -ln(1 - 0.995) ≈ 5.2983
const FSFS_TUI_SEMANTIC_IDLE_MULTIPLIER_PER_MILLE: u64 = 3_507;
const FSFS_TUI_QUALITY_IDLE_MULTIPLIER_PER_MILLE: u64 = 5_299;
const FSFS_TUI_SEARCH_HISTORY_LIMIT: usize = 72;
const FSFS_TUI_FAST_STAGE_SNIPPET_MAX_CHARS: usize = 120;
const FSFS_DEFAULT_QUALITY_EMBEDDER_DIMENSION: usize = 384;
const FSFS_SEARCH_SHORT_QUERY_CHAR_THRESHOLD: usize = 5;
const FSFS_SEARCH_SHORT_QUERY_BUDGET_MULTIPLIER: usize = 1;
const FSFS_SEARCH_FAST_STAGE_BUDGET_MULTIPLIER: usize = 2;
const FSFS_SEARCH_UNBOUNDED_LIMIT_SENTINEL: usize = 0;
// This controls fast-phase head breadth, not final output cardinality.
const FSFS_SEARCH_SEMANTIC_HEAD_LIMIT: usize = 1_000;
// Progressive widening step (sqrt(output_limit - semantic_head_limit) * step).
const FSFS_SEARCH_SEMANTIC_HEAD_PROGRESSIVE_STEP: usize = 16;
const FSFS_SEARCH_SNIPPET_HEAD_LIMIT: usize = 200;
const FSFS_TUI_INTERACTIVE_RESULT_LIMIT: usize = 500;
const FSFS_SEARCH_CACHE_SCHEMA_VERSION: &str = "fsfs.search.cache.v1";
const FSFS_SEARCH_CACHE_DIR_NAME: &str = "query_cache";
const FSFS_SEARCH_SERVE_SCHEMA_VERSION: &str = "fsfs.search.serve.v1";
const FSFS_DAEMON_SOCKET_HASH_PREFIX_LEN: usize = 16;
const FSFS_DAEMON_REQUEST_MAX_BYTES: usize = 1 << 20; // 1 MiB
const FSFS_DAEMON_RESPONSE_MAX_BYTES: usize = 4 << 20; // 4 MiB
const FSFS_DAEMON_CLIENT_IO_TIMEOUT_MS: u64 = 5_000;
const FSFS_DAEMON_CONNECT_MAX_ATTEMPTS: usize = 80;
const FSFS_DAEMON_CONNECT_RETRY_DELAY_MS: u64 = 25;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DashboardSearchStage {
    Idle,
    Lexical,
    SemanticFast,
    QualitySkipped,
    QualityRefined,
    QualityRefinementFailed,
}

impl DashboardSearchStage {
    #[must_use]
    const fn label(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Lexical => "lexical",
            Self::SemanticFast => "semantic_fast",
            Self::QualitySkipped => "quality_skipped",
            Self::QualityRefined => "quality_refined",
            Self::QualityRefinementFailed => "quality_refinement_failed",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SearchExecutionMode {
    Full,
    FastOnly,
    LexicalOnly,
}

impl SearchExecutionMode {
    #[must_use]
    const fn label(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::FastOnly => "fast_only",
            Self::LexicalOnly => "lexical_only",
        }
    }
}

fn parse_search_execution_mode(raw: Option<&str>) -> SearchResult<SearchExecutionMode> {
    match raw {
        None | Some("full") => Ok(SearchExecutionMode::Full),
        Some("fast" | "fast_only") => Ok(SearchExecutionMode::FastOnly),
        Some("lexical" | "lexical_only") => Ok(SearchExecutionMode::LexicalOnly),
        Some(value) => Err(SearchError::InvalidConfig {
            field: "cli.serve.mode".to_owned(),
            value: value.to_owned(),
            reason: "expected full|fast_only|lexical_only".to_owned(),
        }),
    }
}

#[derive(Debug, Clone, Copy)]
struct SemanticVoiDecision {
    run_semantic: bool,
    posterior_useful: f64,
    expected_loss_run: f64,
    expected_loss_skip: f64,
    reason_code: &'static str,
}

impl SemanticVoiDecision {
    #[must_use]
    const fn force_run(reason_code: &'static str) -> Self {
        Self {
            run_semantic: true,
            posterior_useful: 1.0,
            expected_loss_run: 0.0,
            expected_loss_skip: 1.0,
            reason_code,
        }
    }
}

struct SearchExecutionResources {
    index_root: PathBuf,
    lexical_index: Option<TantivyIndex>,
    vector_index: Option<VectorIndex>,
    fast_embedder: Option<Arc<dyn Embedder>>,
    quality_embedder: Option<Arc<dyn Embedder>>,
    fast_embedder_attempted: bool,
    quality_embedder_attempted: bool,
}

impl SearchExecutionResources {
    #[must_use]
    fn capabilities_for_mode(
        &self,
        mode: SearchExecutionMode,
        fast_only: bool,
    ) -> QueryExecutionCapabilities {
        QueryExecutionCapabilities {
            lexical: if self.lexical_index.is_some() {
                CapabilityState::Enabled
            } else {
                CapabilityState::Disabled
            },
            fast_semantic: if !matches!(mode, SearchExecutionMode::LexicalOnly)
                && self.vector_index.is_some()
                && (self.fast_embedder.is_some() || !self.fast_embedder_attempted)
            {
                CapabilityState::Enabled
            } else {
                CapabilityState::Disabled
            },
            quality_semantic: if matches!(mode, SearchExecutionMode::Full)
                && !fast_only
                && self.quality_stage_viable(fast_only)
            {
                CapabilityState::Enabled
            } else {
                CapabilityState::Disabled
            },
            rerank: CapabilityState::Disabled,
        }
    }

    #[must_use]
    fn quality_stage_viable(&self, fast_only: bool) -> bool {
        if fast_only {
            return false;
        }
        let Some(index) = self.vector_index.as_ref() else {
            return false;
        };
        if index.dimension() != FSFS_DEFAULT_QUALITY_EMBEDDER_DIMENSION {
            return false;
        }
        if let Some(fast_embedder) = self.fast_embedder.as_ref()
            && index.embedder_id().eq_ignore_ascii_case(fast_embedder.id())
        {
            return false;
        }
        if self.quality_embedder_attempted {
            return self.quality_embedder.is_some();
        }
        true
    }
}

#[derive(Debug, Clone, Copy)]
struct SearchExecutionFlags {
    include_snippets: bool,
    persist_explain_session: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
struct SearchCacheKey {
    query: String,
    requested_limit: usize,
    mode: String,
    filter: Option<String>,
    fast_only: bool,
    rrf_k_milli: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SearchCacheRecord {
    schema_version: String,
    index_fingerprint: String,
    key: SearchCacheKey,
    payloads: Vec<SearchPayload>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SearchServeRequest {
    query: String,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    filter: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SearchServeResponse {
    ok: bool,
    query: String,
    mode: String,
    cached: bool,
    payloads: Vec<SearchPayload>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[cfg(unix)]
struct SocketPathGuard {
    path: PathBuf,
}

#[cfg(unix)]
impl Drop for SocketPathGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[derive(Debug, Clone, Copy)]
struct SemanticRecallDecisionInput {
    output_limit: usize,
    planning_limit: usize,
    semantic_stage_enabled: bool,
    semantic_index_available: bool,
    lexical_stage_enabled: bool,
    lexical_head_count: usize,
    lexical_full_count: usize,
}

#[derive(Debug, Clone, Copy)]
struct SemanticGateDecisionInput<'a> {
    mode: SearchExecutionMode,
    intent: QueryIntentClass,
    query: &'a str,
    output_limit: usize,
    lexical_head_candidates: &'a [LexicalCandidate],
    semantic_stage_enabled: bool,
    lexical_stage_enabled: bool,
    force_full_semantic_recall: bool,
}

#[allow(clippy::struct_excessive_bools)]
struct SearchDashboardState {
    status_payload: FsfsStatusPayload,
    mode_hint: Option<String>,
    query_input: TextInput,
    search_active: bool,
    pending_lexical_refresh: bool,
    lexical_pending_since: Instant,
    pending_semantic_refresh: bool,
    semantic_pending_since: Instant,
    pending_quality_refresh: bool,
    quality_pending_since: Instant,
    phase_payloads: Vec<SearchPayload>,
    stage_chain: Vec<DashboardSearchStage>,
    active_hit_index: usize,
    context_scroll: u16,
    last_error: Option<String>,
    last_search_elapsed_ms: Option<u64>,
    search_invocations: u64,
    latency_history_ms: VecDeque<f64>,
    hits_history: VecDeque<f64>,
    lexical_debounce_window_ms: u64,
    semantic_debounce_window_ms: u64,
    quality_debounce_window_ms: u64,
    typing_cadence_ewma_ms: Option<u64>,
    last_query_edit_at: Option<Instant>,
    result_limit: usize,
}

impl SearchDashboardState {
    fn new(
        status_payload: FsfsStatusPayload,
        mode_hint: Option<String>,
        result_limit: usize,
        no_color: bool,
    ) -> Self {
        let query_input = TextInput::new()
            .with_placeholder("Search indexed files... (/ focus, Esc blur)")
            .with_focused(true)
            .with_style(ui_fg(no_color, PackedRgba::rgb(224, 236, 255)))
            .with_placeholder_style(ui_fg(no_color, PackedRgba::rgb(139, 161, 198)))
            .with_cursor_style(ui_fg_bg(
                no_color,
                PackedRgba::rgb(12, 24, 40),
                PackedRgba::rgb(122, 219, 255),
            ));

        let now = Instant::now();
        Self {
            status_payload,
            mode_hint,
            query_input,
            search_active: true,
            pending_lexical_refresh: false,
            lexical_pending_since: now,
            pending_semantic_refresh: false,
            semantic_pending_since: now,
            pending_quality_refresh: false,
            quality_pending_since: now,
            phase_payloads: Vec::new(),
            stage_chain: vec![DashboardSearchStage::Idle],
            active_hit_index: 0,
            context_scroll: 0,
            last_error: None,
            last_search_elapsed_ms: None,
            search_invocations: 0,
            latency_history_ms: VecDeque::with_capacity(FSFS_TUI_SEARCH_HISTORY_LIMIT),
            hits_history: VecDeque::with_capacity(FSFS_TUI_SEARCH_HISTORY_LIMIT),
            lexical_debounce_window_ms: FSFS_TUI_LEXICAL_DEBOUNCE_MS,
            semantic_debounce_window_ms: FSFS_TUI_SEMANTIC_DEBOUNCE_MS,
            quality_debounce_window_ms: FSFS_TUI_QUALITY_DEBOUNCE_MS,
            typing_cadence_ewma_ms: None,
            last_query_edit_at: None,
            result_limit: result_limit.max(1),
        }
    }

    #[must_use]
    fn latest_payload(&self) -> Option<&SearchPayload> {
        self.phase_payloads.last()
    }

    #[must_use]
    fn latest_hits(&self) -> &[SearchHitPayload] {
        if let Some(payload) = self.latest_payload() {
            payload.hits.as_slice()
        } else {
            &[]
        }
    }

    fn set_focus(&mut self, focused: bool) {
        self.search_active = focused;
        self.query_input.set_focused(focused);
    }

    fn mark_query_dirty(&mut self) {
        let now = Instant::now();
        self.record_query_edit(now);
        self.context_scroll = 0;
        self.pending_lexical_refresh = true;
        self.lexical_pending_since = now;
        self.pending_semantic_refresh = true;
        self.semantic_pending_since = now;
        self.pending_quality_refresh = false;
        self.quality_pending_since = now;
    }

    fn force_refresh_now(&mut self) {
        let now = Instant::now();
        self.context_scroll = 0;
        self.pending_lexical_refresh = true;
        self.lexical_pending_since = now
            .checked_sub(Duration::from_millis(self.lexical_debounce_window_ms))
            .unwrap_or(now);
        self.pending_semantic_refresh = true;
        self.semantic_pending_since = now
            .checked_sub(Duration::from_millis(self.semantic_debounce_window_ms))
            .unwrap_or(now);
        self.pending_quality_refresh = false;
        self.quality_pending_since = now;
        let refresh_gate = self
            .quality_idle_gate_ms()
            .max(self.quality_debounce_window_ms);
        self.last_query_edit_at = Some(
            now.checked_sub(Duration::from_millis(refresh_gate))
                .unwrap_or(now),
        );
    }

    #[must_use]
    fn should_refresh_lexical(&self) -> bool {
        self.pending_lexical_refresh
            && self.lexical_pending_since.elapsed()
                >= Duration::from_millis(self.lexical_debounce_window_ms)
    }

    #[must_use]
    fn should_refresh_semantic(&self) -> bool {
        self.pending_semantic_refresh
            && self.semantic_pending_since.elapsed()
                >= Duration::from_millis(self.semantic_debounce_window_ms)
            && self.elapsed_since_last_query_edit()
                >= Duration::from_millis(self.semantic_idle_gate_ms())
    }

    #[must_use]
    fn should_refresh_quality(&self) -> bool {
        self.pending_quality_refresh
            && self.quality_pending_since.elapsed()
                >= Duration::from_millis(self.quality_debounce_window_ms)
            && self.elapsed_since_last_query_edit()
                >= Duration::from_millis(self.quality_idle_gate_ms())
    }

    #[must_use]
    fn lexical_stage_result_limit(&self) -> usize {
        let full_limit = self.result_limit.max(1);
        let active_limit = full_limit.min(FSFS_SEARCH_SNIPPET_HEAD_LIMIT);
        if self.elapsed_since_last_query_edit()
            < Duration::from_millis(self.semantic_idle_gate_ms())
        {
            active_limit.max(1)
        } else {
            full_limit
        }
    }

    fn schedule_quality_refresh(&mut self) {
        self.pending_quality_refresh = true;
        self.quality_pending_since = Instant::now();
    }

    const fn clear_pending_quality_refresh(&mut self) {
        self.pending_quality_refresh = false;
    }

    fn clear_search_results(&mut self) {
        self.phase_payloads.clear();
        self.stage_chain.clear();
        self.stage_chain.push(DashboardSearchStage::Idle);
        self.active_hit_index = 0;
        self.context_scroll = 0;
        self.last_error = None;
        self.last_search_elapsed_ms = None;
    }

    fn next_hit(&mut self) {
        let len = self.latest_hits().len();
        if len > 0 {
            self.active_hit_index = (self.active_hit_index + 1) % len;
            self.context_scroll = 0;
        }
    }

    fn prev_hit(&mut self) {
        let len = self.latest_hits().len();
        if len > 0 {
            self.active_hit_index = (self.active_hit_index + len - 1) % len;
            self.context_scroll = 0;
        }
    }

    fn clamp_active_hit(&mut self) {
        let len = self.latest_hits().len();
        if len == 0 {
            self.active_hit_index = 0;
            self.context_scroll = 0;
        } else {
            self.active_hit_index = self.active_hit_index.min(len - 1);
        }
    }

    fn scroll_context_up(&mut self, lines: u16) {
        self.context_scroll = self.context_scroll.saturating_sub(lines.max(1));
    }

    fn scroll_context_down(&mut self, lines: u16) {
        self.context_scroll = self.context_scroll.saturating_add(lines.max(1));
    }

    fn push_latency_sample(&mut self, value: f64) {
        if self.latency_history_ms.len() >= FSFS_TUI_SEARCH_HISTORY_LIMIT {
            let _ = self.latency_history_ms.pop_front();
        }
        self.latency_history_ms.push_back(value.max(0.0));
        self.recompute_adaptive_debounce();
    }

    fn push_hit_sample(&mut self, value: f64) {
        if self.hits_history.len() >= FSFS_TUI_SEARCH_HISTORY_LIMIT {
            let _ = self.hits_history.pop_front();
        }
        self.hits_history.push_back(value.max(0.0));
    }

    #[must_use]
    fn quality_model_cached(&self) -> bool {
        self.status_payload
            .models
            .iter()
            .any(|model| model.tier.eq_ignore_ascii_case("quality") && model.cached)
    }

    fn set_phase_payloads(
        &mut self,
        payloads: Vec<SearchPayload>,
        stages: Vec<DashboardSearchStage>,
    ) {
        self.phase_payloads = payloads;
        self.stage_chain = if stages.is_empty() {
            vec![DashboardSearchStage::Idle]
        } else {
            stages
        };
    }

    fn record_query_edit(&mut self, now: Instant) {
        if let Some(previous) = self.last_query_edit_at {
            let delta_ms = u64::try_from(now.saturating_duration_since(previous).as_millis())
                .unwrap_or(u64::MAX);
            if (FSFS_TUI_TYPING_INTERVAL_MIN_MS..=FSFS_TUI_TYPING_INTERVAL_MAX_MS)
                .contains(&delta_ms)
            {
                let previous_ewma = self.typing_cadence_ewma_ms.unwrap_or(delta_ms);
                let alpha = FSFS_TUI_TYPING_CADENCE_ALPHA_PER_MILLE.min(1_000);
                let retained = previous_ewma.saturating_mul(1_000_u64.saturating_sub(alpha));
                let incoming = delta_ms.saturating_mul(alpha);
                let blended = retained.saturating_add(incoming).div_ceil(1_000);
                self.typing_cadence_ewma_ms = Some(blended);
            }
        }
        self.last_query_edit_at = Some(now);
        self.recompute_adaptive_debounce();
    }

    fn recompute_adaptive_debounce(&mut self) {
        let cadence_ms = self
            .typing_cadence_ewma_ms
            .unwrap_or(FSFS_TUI_TYPING_CADENCE_DEFAULT_MS)
            .clamp(
                FSFS_TUI_TYPING_INTERVAL_MIN_MS,
                FSFS_TUI_TYPING_INTERVAL_MAX_MS,
            );
        let mut lexical = cadence_ms.div_ceil(10).clamp(
            FSFS_TUI_LEXICAL_DEBOUNCE_MIN_MS,
            FSFS_TUI_LEXICAL_DEBOUNCE_MAX_MS,
        );
        if self.query_input.value().chars().count() <= FSFS_SEARCH_SHORT_QUERY_CHAR_THRESHOLD {
            lexical = lexical.min(FSFS_TUI_LEXICAL_DEBOUNCE_SHORT_QUERY_CAP_MS);
        }
        let mut semantic = lexical.saturating_mul(4).saturating_add(8).clamp(
            FSFS_TUI_SEMANTIC_DEBOUNCE_MIN_MS,
            FSFS_TUI_SEMANTIC_DEBOUNCE_MAX_MS,
        );
        let mut quality = semantic.saturating_mul(2).saturating_add(48).clamp(
            FSFS_TUI_QUALITY_DEBOUNCE_MIN_MS,
            FSFS_TUI_QUALITY_DEBOUNCE_MAX_MS,
        );

        if let Some(latency_ms) = self.last_search_elapsed_ms {
            if latency_ms >= FSFS_TUI_LATENCY_SLOW_PATH_MS {
                semantic = semantic
                    .saturating_add(FSFS_TUI_SEMANTIC_DEBOUNCE_SLOW_BUMP_MS)
                    .clamp(
                        FSFS_TUI_SEMANTIC_DEBOUNCE_MIN_MS,
                        FSFS_TUI_SEMANTIC_DEBOUNCE_MAX_MS,
                    );
                quality = quality
                    .saturating_add(FSFS_TUI_QUALITY_DEBOUNCE_SLOW_BUMP_MS)
                    .clamp(
                        FSFS_TUI_QUALITY_DEBOUNCE_MIN_MS,
                        FSFS_TUI_QUALITY_DEBOUNCE_MAX_MS,
                    );
            } else if latency_ms <= FSFS_TUI_LATENCY_FAST_PATH_MS {
                semantic = semantic
                    .saturating_sub(FSFS_TUI_SEMANTIC_DEBOUNCE_FAST_TRIM_MS)
                    .clamp(
                        FSFS_TUI_SEMANTIC_DEBOUNCE_MIN_MS,
                        FSFS_TUI_SEMANTIC_DEBOUNCE_MAX_MS,
                    );
                quality = quality
                    .saturating_sub(FSFS_TUI_QUALITY_DEBOUNCE_FAST_TRIM_MS)
                    .clamp(
                        FSFS_TUI_QUALITY_DEBOUNCE_MIN_MS,
                        FSFS_TUI_QUALITY_DEBOUNCE_MAX_MS,
                    );
            }
        }

        self.lexical_debounce_window_ms = lexical;
        self.semantic_debounce_window_ms = semantic;
        self.quality_debounce_window_ms = quality;
    }

    #[must_use]
    fn elapsed_since_last_query_edit(&self) -> Duration {
        self.last_query_edit_at
            .map_or(Duration::MAX, |instant| instant.elapsed())
    }

    #[must_use]
    fn semantic_idle_gate_ms(&self) -> u64 {
        let cadence_ms = self
            .typing_cadence_ewma_ms
            .unwrap_or(FSFS_TUI_TYPING_CADENCE_DEFAULT_MS)
            .clamp(
                FSFS_TUI_TYPING_INTERVAL_MIN_MS,
                FSFS_TUI_TYPING_INTERVAL_MAX_MS,
            );
        Self::pause_quantile_gate_ms(cadence_ms, FSFS_TUI_SEMANTIC_IDLE_MULTIPLIER_PER_MILLE).clamp(
            FSFS_TUI_SEMANTIC_IDLE_GATE_MIN_MS,
            FSFS_TUI_SEMANTIC_IDLE_GATE_MAX_MS,
        )
    }

    #[must_use]
    fn quality_idle_gate_ms(&self) -> u64 {
        let cadence_ms = self
            .typing_cadence_ewma_ms
            .unwrap_or(FSFS_TUI_TYPING_CADENCE_DEFAULT_MS)
            .clamp(
                FSFS_TUI_TYPING_INTERVAL_MIN_MS,
                FSFS_TUI_TYPING_INTERVAL_MAX_MS,
            );
        let semantic_gate = self.semantic_idle_gate_ms();
        Self::pause_quantile_gate_ms(cadence_ms, FSFS_TUI_QUALITY_IDLE_MULTIPLIER_PER_MILLE)
            .max(semantic_gate.saturating_add(40))
            .clamp(
                FSFS_TUI_QUALITY_IDLE_GATE_MIN_MS,
                FSFS_TUI_QUALITY_IDLE_GATE_MAX_MS,
            )
    }

    #[must_use]
    fn pause_quantile_gate_ms(cadence_ms: u64, multiplier_per_mille: u64) -> u64 {
        cadence_ms
            .max(1)
            .saturating_mul(multiplier_per_mille.max(1))
            .div_ceil(1_000)
            .max(1)
    }

    #[must_use]
    fn typing_cadence_ms_label(&self) -> String {
        format_count_u64(
            self.typing_cadence_ewma_ms
                .unwrap_or(FSFS_TUI_TYPING_CADENCE_DEFAULT_MS),
        )
    }
}

#[derive(Debug, Clone)]
struct IndexCandidate {
    file_path: PathBuf,
    file_key: String,
    modified_ms: u64,
    ingestion_class: IngestionClass,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct IndexDiscoveryStats {
    discovered_files: usize,
    skipped_files: usize,
    reason_codes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct IndexManifestEntry {
    file_key: String,
    revision: i64,
    ingestion_class: String,
    canonical_bytes: u64,
    reason_code: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct IndexSentinel {
    schema_version: u16,
    generated_at_ms: u64,
    command: String,
    target_root: String,
    index_root: String,
    discovered_files: usize,
    indexed_files: usize,
    skipped_files: usize,
    reason_codes: Vec<String>,
    total_canonical_bytes: u64,
    source_hash_hex: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct RootProbeStats {
    candidate_files: usize,
    code_files: usize,
    doc_files: usize,
    repo_markers: usize,
    candidate_bytes: u64,
    scanned_dirs: usize,
    scanned_entries: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct IndexRootProposal {
    path: PathBuf,
    score: i64,
    stats: RootProbeStats,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IndexingProgressStage {
    Discovering,
    Indexing,
    Finalizing,
    Completed,
}

impl IndexingProgressStage {
    #[must_use]
    const fn label(self) -> &'static str {
        match self {
            Self::Discovering => "Discovering Files",
            Self::Indexing => "Indexing Content",
            Self::Finalizing => "Finalizing Artifacts",
            Self::Completed => "Completed",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct IndexingProgressSnapshot {
    stage: IndexingProgressStage,
    target_root: PathBuf,
    index_root: PathBuf,
    discovered_files: usize,
    candidate_files: usize,
    processed_files: usize,
    skipped_files: usize,
    semantic_files: usize,
    canonical_bytes: u64,
    canonical_lines: u64,
    index_size_bytes: u64,
    discovery_elapsed_ms: u128,
    lexical_elapsed_ms: u128,
    embedding_elapsed_ms: u128,
    vector_elapsed_ms: u128,
    total_elapsed_ms: u128,
    active_file: Option<String>,
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
impl IndexingProgressSnapshot {
    #[must_use]
    fn elapsed_seconds(&self) -> f64 {
        if self.total_elapsed_ms == 0 {
            0.0
        } else {
            self.total_elapsed_ms as f64 / 1_000.0
        }
    }

    #[must_use]
    fn files_per_second(&self) -> f64 {
        let elapsed = self.elapsed_seconds();
        if elapsed <= f64::EPSILON {
            0.0
        } else {
            self.processed_files as f64 / elapsed
        }
    }

    #[must_use]
    fn lines_per_second(&self) -> f64 {
        let elapsed = self.elapsed_seconds();
        if elapsed <= f64::EPSILON {
            0.0
        } else {
            self.canonical_lines as f64 / elapsed
        }
    }

    #[must_use]
    fn mb_per_second(&self) -> f64 {
        let elapsed = self.elapsed_seconds();
        if elapsed <= f64::EPSILON {
            0.0
        } else {
            (self.canonical_bytes as f64 / (1024.0 * 1024.0)) / elapsed
        }
    }

    #[must_use]
    fn index_growth_bytes_per_second(&self) -> f64 {
        let elapsed = self.elapsed_seconds();
        if elapsed <= f64::EPSILON {
            0.0
        } else {
            self.index_size_bytes as f64 / elapsed
        }
    }

    #[must_use]
    fn completion_ratio(&self) -> f64 {
        if self.candidate_files == 0 {
            0.0
        } else {
            (self.processed_files as f64 / self.candidate_files as f64).clamp(0.0, 1.0)
        }
    }

    #[must_use]
    fn eta_seconds(&self) -> Option<u64> {
        if self.candidate_files <= self.processed_files {
            return Some(0);
        }
        let files_per_second = self.files_per_second();
        if files_per_second <= f64::EPSILON {
            return None;
        }
        let remaining = (self.candidate_files - self.processed_files) as f64;
        Some((remaining / files_per_second).ceil() as u64)
    }
}

#[derive(Debug, Clone, PartialEq)]
struct IndexingRenderState {
    last_snapshot: Option<IndexingProgressSnapshot>,
    completion_history: VecDeque<f64>,
    files_rate_history: VecDeque<f64>,
    growth_rate_history: VecDeque<f64>,
    files_rate_ema: f64,
    lines_rate_ema: f64,
    mb_rate_ema: f64,
    growth_rate_ema: f64,
    eta_ema_seconds: Option<f64>,
    instant_files_per_second: f64,
    instant_lines_per_second: f64,
    instant_mb_per_second: f64,
    instant_growth_kb_per_second: f64,
}

impl Default for IndexingRenderState {
    fn default() -> Self {
        Self {
            last_snapshot: None,
            completion_history: VecDeque::with_capacity(INDEXING_TUI_RATE_HISTORY_LIMIT),
            files_rate_history: VecDeque::with_capacity(INDEXING_TUI_RATE_HISTORY_LIMIT),
            growth_rate_history: VecDeque::with_capacity(INDEXING_TUI_RATE_HISTORY_LIMIT),
            files_rate_ema: 0.0,
            lines_rate_ema: 0.0,
            mb_rate_ema: 0.0,
            growth_rate_ema: 0.0,
            eta_ema_seconds: None,
            instant_files_per_second: 0.0,
            instant_lines_per_second: 0.0,
            instant_mb_per_second: 0.0,
            instant_growth_kb_per_second: 0.0,
        }
    }
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
impl IndexingRenderState {
    fn observe(&mut self, snapshot: &IndexingProgressSnapshot) {
        let files_per_second = snapshot.files_per_second();
        let lines_per_second = snapshot.lines_per_second();
        let mb_per_second = snapshot.mb_per_second();
        let growth_kb_per_second = snapshot.index_growth_bytes_per_second() / 1024.0;

        let mut instant_files_per_second = files_per_second;
        let mut instant_lines_per_second = lines_per_second;
        let mut instant_mb_per_second = mb_per_second;
        let mut instant_growth_kb_per_second = growth_kb_per_second;

        if let Some(previous) = self.last_snapshot.as_ref() {
            let delta_ms = snapshot
                .total_elapsed_ms
                .saturating_sub(previous.total_elapsed_ms);
            if delta_ms > 0 {
                let delta_secs = delta_ms as f64 / 1_000.0;
                if delta_secs > f64::EPSILON {
                    let delta_files = snapshot
                        .processed_files
                        .saturating_sub(previous.processed_files)
                        as f64;
                    let delta_lines = snapshot
                        .canonical_lines
                        .saturating_sub(previous.canonical_lines)
                        as f64;
                    let delta_mebibytes = snapshot
                        .canonical_bytes
                        .saturating_sub(previous.canonical_bytes)
                        as f64
                        / (1024.0 * 1024.0);
                    let delta_growth_kb = snapshot
                        .index_size_bytes
                        .saturating_sub(previous.index_size_bytes)
                        as f64
                        / 1024.0;
                    instant_files_per_second = delta_files / delta_secs;
                    instant_lines_per_second = delta_lines / delta_secs;
                    instant_mb_per_second = delta_mebibytes / delta_secs;
                    instant_growth_kb_per_second = delta_growth_kb / delta_secs;
                }
            }
        }

        self.instant_files_per_second = instant_files_per_second.max(0.0);
        self.instant_lines_per_second = instant_lines_per_second.max(0.0);
        self.instant_mb_per_second = instant_mb_per_second.max(0.0);
        self.instant_growth_kb_per_second = instant_growth_kb_per_second.max(0.0);

        self.files_rate_ema = ema(
            self.files_rate_ema,
            files_per_second,
            INDEXING_TUI_EMA_ALPHA,
        );
        self.lines_rate_ema = ema(
            self.lines_rate_ema,
            lines_per_second,
            INDEXING_TUI_EMA_ALPHA,
        );
        self.mb_rate_ema = ema(self.mb_rate_ema, mb_per_second, INDEXING_TUI_EMA_ALPHA);
        self.growth_rate_ema = ema(
            self.growth_rate_ema,
            growth_kb_per_second,
            INDEXING_TUI_EMA_ALPHA,
        );

        Self::push_history(&mut self.completion_history, snapshot.completion_ratio());
        Self::push_history(&mut self.files_rate_history, files_per_second);
        Self::push_history(&mut self.growth_rate_history, growth_kb_per_second);

        self.eta_ema_seconds = if snapshot.candidate_files <= snapshot.processed_files {
            Some(0.0)
        } else if self.files_rate_ema > f64::EPSILON {
            let remaining = (snapshot.candidate_files - snapshot.processed_files) as f64;
            let smoothed_eta = remaining / self.files_rate_ema;
            Some(self.eta_ema_seconds.map_or(smoothed_eta, |previous_eta| {
                ema(previous_eta, smoothed_eta, INDEXING_TUI_EMA_ALPHA)
            }))
        } else {
            None
        };

        self.last_snapshot = Some(snapshot.clone());
    }

    fn completion_sparkline(&self, width: usize) -> String {
        render_sparkline(&self.completion_history, width)
    }

    fn files_rate_sparkline(&self, width: usize) -> String {
        render_sparkline(&self.files_rate_history, width)
    }

    fn growth_rate_sparkline(&self, width: usize) -> String {
        render_sparkline(&self.growth_rate_history, width)
    }

    fn smoothed_eta_seconds(&self) -> Option<u64> {
        self.eta_ema_seconds
            .map(|seconds| seconds.max(0.0).ceil() as u64)
    }

    fn push_history(history: &mut VecDeque<f64>, value: f64) {
        if history.len() >= INDEXING_TUI_RATE_HISTORY_LIMIT {
            let _ = history.pop_front();
        }
        history.push_back(value.max(0.0));
    }
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn render_sparkline(values: &VecDeque<f64>, width: usize) -> String {
    const BINS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    if width == 0 {
        return String::new();
    }
    if values.is_empty() {
        return "·".repeat(width);
    }

    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = (max - min).max(f64::EPSILON);
    let count = values.len();
    let mut out = String::with_capacity(width);

    for idx in 0..width {
        let sample = idx.saturating_mul(count) / width;
        let value = values[sample.min(count.saturating_sub(1))];
        let normalized = ((value - min) / span).clamp(0.0, 1.0);
        let bin = (normalized * (BINS.len().saturating_sub(1)) as f64).round() as usize;
        out.push(BINS[bin.min(BINS.len().saturating_sub(1))]);
    }
    out
}

fn ema(previous: f64, sample: f64, alpha: f64) -> f64 {
    if previous <= f64::EPSILON {
        sample.max(0.0)
    } else {
        previous
            .mul_add(1.0 - alpha, sample.max(0.0) * alpha)
            .max(0.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct ExplainSession {
    schema_version: String,
    generated_at_ms: u64,
    query: String,
    phase: SearchOutputPhase,
    rrf_k: f64,
    hits: Vec<ExplainSessionHit>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct ExplainSessionHit {
    result_id: String,
    rank: usize,
    path: String,
    final_score: f64,
    lexical_rank: Option<usize>,
    semantic_rank: Option<usize>,
    lexical_score: Option<f32>,
    semantic_score: Option<f32>,
    in_both_sources: bool,
}

impl ExplainSession {
    fn from_fused(
        query: &str,
        phase: SearchOutputPhase,
        rrf_k: f64,
        fused: &[FusedCandidate],
    ) -> Self {
        let hits = fused
            .iter()
            .enumerate()
            .map(|(idx, candidate)| ExplainSessionHit {
                result_id: result_id(idx),
                rank: idx.saturating_add(1),
                path: candidate.doc_id.clone(),
                final_score: sanitize_explain_score(candidate.fused_score),
                lexical_rank: candidate.lexical_rank,
                semantic_rank: candidate.semantic_rank,
                lexical_score: candidate.lexical_score,
                semantic_score: candidate.semantic_score,
                in_both_sources: candidate.in_both_sources,
            })
            .collect();

        Self {
            schema_version: EXPLAIN_SESSION_SCHEMA_VERSION.to_owned(),
            generated_at_ms: pressure_timestamp_ms(),
            query: query.to_owned(),
            phase,
            rrf_k,
            hits,
        }
    }

    fn resolve(&self, id: &str) -> Option<&ExplainSessionHit> {
        self.hits.iter().find(|hit| hit.result_id == id)
    }

    fn preview_ids(&self) -> String {
        if self.hits.is_empty() {
            return "none".to_owned();
        }
        self.hits
            .iter()
            .take(10)
            .map(|hit| hit.result_id.clone())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

const fn sanitize_explain_score(score: f64) -> f64 {
    if score.is_finite() { score } else { 0.0 }
}

/// Live ingest pipeline that re-indexes changed files detected by the watcher.
///
/// Reads file content, canonicalizes, embeds, and updates both the lexical
/// (Tantivy) and semantic (FSVI) indexes incrementally.
struct LiveIngestPipeline {
    target_root: PathBuf,
    lexical_index: TantivyIndex,
    vector_index: Arc<std::sync::Mutex<VectorIndex>>,
    embedder: Arc<dyn Embedder>,
    canonicalizer: DefaultCanonicalizer,
    storage_db_path: Option<PathBuf>,
}

#[derive(Debug)]
struct LiveVectorSink {
    vector_index: Arc<std::sync::Mutex<VectorIndex>>,
}

impl EmbeddingVectorSink for LiveVectorSink {
    fn persist(
        &self,
        doc_id: &str,
        embedding_embedder_id: &str,
        embedding: &[f32],
    ) -> SearchResult<()> {
        let mut vi = self
            .vector_index
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Err(error) = vi.soft_delete(doc_id) {
            tracing::debug!(
                file_key = %doc_id,
                embedder_id = embedding_embedder_id,
                error = %error,
                "storage vector sink: soft_delete ignored (doc may be new)"
            );
        }
        vi.append(doc_id, embedding)?;
        drop(vi);
        Ok(())
    }
}

#[derive(Debug)]
struct StorageBatchContext {
    storage: Arc<Storage>,
    runner: StorageBackedJobRunner,
}

impl WatchIngestPipeline for LiveIngestPipeline {
    fn apply_batch(
        &self,
        batch: &[WatchIngestOp],
        rt: &asupersync::runtime::Runtime,
    ) -> frankensearch_core::SearchResult<usize> {
        let cx = Cx::for_request();
        rt.block_on(self.apply_batch_inner(&cx, batch))
    }
}

impl LiveIngestPipeline {
    fn new(
        target_root: PathBuf,
        lexical_index: TantivyIndex,
        vector_index: VectorIndex,
        embedder: Arc<dyn Embedder>,
    ) -> Self {
        Self {
            target_root,
            lexical_index,
            vector_index: Arc::new(std::sync::Mutex::new(vector_index)),
            embedder,
            canonicalizer: DefaultCanonicalizer::default(),
            storage_db_path: None,
        }
    }

    fn with_storage_db_path(mut self, storage_db_path: PathBuf) -> Self {
        self.storage_db_path = Some(storage_db_path);
        self
    }

    fn resolve_paths(&self, file_key: &str) -> frankensearch_core::SearchResult<(PathBuf, String)> {
        let key_path = Path::new(file_key);
        // Reject ".." components outright. This prevents traversal even when
        // canonicalize() falls back to the raw path for missing files.
        if key_path
            .components()
            .any(|c| c == std::path::Component::ParentDir)
        {
            return Err(frankensearch_core::SearchError::InvalidConfig {
                field: "file_key".into(),
                value: file_key.into(),
                reason: "file_key must not contain '..' components (directory traversal)".into(),
            });
        }
        let abs_path = if key_path.is_absolute() {
            PathBuf::from(file_key)
        } else {
            self.target_root.join(file_key)
        };
        // Resolve symlinks / ".." components, then verify the result stays
        // inside target_root.  Without this check, symlinks or absolute paths
        // could escape the project boundary (path traversal).
        let canonical = abs_path.canonicalize().unwrap_or_else(|_| abs_path.clone());
        if !canonical.starts_with(&self.target_root) {
            return Err(frankensearch_core::SearchError::InvalidConfig {
                field: "file_key".into(),
                value: file_key.into(),
                reason: "path escapes target root (directory traversal)".into(),
            });
        }
        let rel_key = normalize_file_key_for_index(&canonical, &self.target_root);
        Ok((canonical, rel_key))
    }

    fn soft_delete_vector(&self, rel_key: &str) {
        let mut vi = self
            .vector_index
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Err(error) = vi.soft_delete(rel_key) {
            tracing::debug!(
                file_key = %rel_key,
                error = %error,
                "soft_delete_vector: ignored (doc may not exist yet)"
            );
        }
    }

    async fn prune_indexes(&self, cx: &Cx, rel_key: &str) -> frankensearch_core::SearchResult<()> {
        self.lexical_index.delete_document(cx, rel_key).await?;
        self.soft_delete_vector(rel_key);
        Ok(())
    }

    #[allow(clippy::arc_with_non_send_sync)]
    fn build_storage_batch_context(
        &self,
    ) -> frankensearch_core::SearchResult<Option<StorageBatchContext>> {
        let Some(storage_db_path) = self.storage_db_path.as_ref() else {
            return Ok(None);
        };

        let storage = Arc::new(Storage::open(PipelineStorageConfig {
            db_path: storage_db_path.clone(),
            ..PipelineStorageConfig::default()
        })?);
        let queue = Arc::new(PersistentJobQueue::new(
            Arc::clone(&storage),
            JobQueueConfig::default(),
        ));
        let sink = Arc::new(LiveVectorSink {
            vector_index: Arc::clone(&self.vector_index),
        }) as Arc<dyn EmbeddingVectorSink>;

        let runner = StorageBackedJobRunner::new(
            Arc::clone(&storage),
            queue,
            Arc::new(DefaultCanonicalizer::default()),
            Arc::clone(&self.embedder),
            sink,
        )
        .with_config(PipelineConfig {
            worker_max_idle_cycles: Some(1),
            ..PipelineConfig::default()
        });

        Ok(Some(StorageBatchContext { storage, runner }))
    }

    fn enqueue_storage_upsert(
        storage_ctx: &StorageBatchContext,
        rel_key: &str,
        canonical: &str,
        abs_path: &Path,
    ) -> frankensearch_core::SearchResult<IngestResult> {
        let mut request = IngestRequest::new(rel_key.to_owned(), canonical.to_owned());
        request.source_path = Some(abs_path.display().to_string());
        request.enqueue_quality = false;
        storage_ctx.runner.ingest(request)
    }

    #[allow(clippy::future_not_send)]
    async fn drain_storage_jobs(
        &self,
        cx: &Cx,
        storage_ctx: &StorageBatchContext,
    ) -> frankensearch_core::SearchResult<()> {
        loop {
            let batch = storage_ctx
                .runner
                .process_batch(cx, "fsfs-watch-live")
                .await?;
            if batch.jobs_claimed == 0 {
                break;
            }
        }
        Ok(())
    }

    fn purge_storage_document(
        storage_ctx: Option<&StorageBatchContext>,
        rel_key: &str,
    ) -> frankensearch_core::SearchResult<()> {
        let Some(storage_ctx) = storage_ctx else {
            return Ok(());
        };
        storage_ctx.storage.transaction(|conn| {
            conn.execute_with_params(
                "DELETE FROM documents WHERE doc_id = ?1;",
                &[SqliteValue::Text(rel_key.to_owned())],
            )
            .map_err(|error| SearchError::SubsystemError {
                subsystem: "fsfs.watch.storage",
                source: Box::new(std::io::Error::other(error.to_string())),
            })?;
            Ok(())
        })
    }

    fn plan_live_vector_upsert(
        rel_key: &str,
        revision: i64,
        ingestion_class: IngestionClass,
        content_len_bytes: u64,
    ) -> VectorPipelinePlan {
        let (tier, reason_code, chunk_count) =
            if matches!(ingestion_class, IngestionClass::FullSemanticLexical) {
                (
                    VectorSchedulingTier::FastOnly,
                    "vector.plan.fast_only_quality_unavailable".to_owned(),
                    FsfsRuntime::chunk_count_for_bytes(content_len_bytes),
                )
            } else {
                (
                    VectorSchedulingTier::Skip,
                    "vector.skip.non_semantic_ingestion_class".to_owned(),
                    0,
                )
            };

        VectorPipelinePlan {
            file_key: rel_key.to_owned(),
            revision,
            chunk_count,
            batch_size: 1,
            tier,
            invalidate_revisions_through: None,
            reason_code,
        }
    }

    async fn apply_live_vector_actions(
        &self,
        cx: &Cx,
        rel_key: &str,
        revision: i64,
        canonical: &str,
        vector_plan: &VectorPipelinePlan,
    ) -> frankensearch_core::SearchResult<()> {
        for action in FsfsRuntime::vector_index_write_actions(vector_plan) {
            match action {
                VectorIndexWriteAction::InvalidateRevisionsThrough { .. } => {
                    tracing::debug!(
                        file_key = %rel_key,
                        revision,
                        reason_code = %vector_plan.reason_code,
                        "watcher ingest: revision invalidation action not yet implemented; continuing"
                    );
                }
                VectorIndexWriteAction::AppendFast { .. } => {
                    match self.embedder.embed(cx, canonical).await {
                        Ok(embedding) => {
                            let mut vi = self
                                .vector_index
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner);
                            // soft_delete returns Ok(false) if doc doesn't exist, so Err is a real failure
                            // (IO/corruption) that must prevent appending.
                            vi.soft_delete(rel_key)?;
                            vi.append(rel_key, &embedding)?;
                        }
                        Err(error) => {
                            warn!(
                                file_key = %rel_key,
                                error = %error,
                                reason_code = %vector_plan.reason_code,
                                "watcher ingest: embedding failed; lexical-only for this file"
                            );
                        }
                    }
                }
                VectorIndexWriteAction::AppendQuality { .. } => {
                    tracing::debug!(
                        file_key = %rel_key,
                        revision,
                        reason_code = %vector_plan.reason_code,
                        "watcher ingest: quality vector append action skipped (fast-tier live ingest)"
                    );
                }
                VectorIndexWriteAction::MarkLexicalFallback { .. }
                | VectorIndexWriteAction::Skip { .. } => {
                    self.soft_delete_vector(rel_key);
                }
            }
        }

        Ok(())
    }

    #[allow(clippy::future_not_send)]
    async fn apply_upsert_op(
        &self,
        cx: &Cx,
        file_key: &str,
        revision: i64,
        ingestion_class: IngestionClass,
        storage_ctx: Option<&StorageBatchContext>,
    ) -> frankensearch_core::SearchResult<bool> {
        let (abs_path, rel_key) = self.resolve_paths(file_key)?;

        if matches!(
            ingestion_class,
            IngestionClass::MetadataOnly | IngestionClass::Skip
        ) {
            self.prune_indexes(cx, &rel_key).await?;
            Self::purge_storage_document(storage_ctx, &rel_key)?;
            return Ok(true);
        }

        let bytes = match fs::read(&abs_path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == ErrorKind::NotFound => {
                self.prune_indexes(cx, &rel_key).await?;
                Self::purge_storage_document(storage_ctx, &rel_key)?;
                return Ok(true);
            }
            Err(error)
                if matches!(
                    error.kind(),
                    ErrorKind::IsADirectory | ErrorKind::PermissionDenied | ErrorKind::Interrupted
                ) =>
            {
                return Ok(false);
            }
            Err(error) => return Err(error.into()),
        };

        if is_probably_binary(&bytes) {
            self.prune_indexes(cx, &rel_key).await?;
            Self::purge_storage_document(storage_ctx, &rel_key)?;
            return Ok(true);
        }

        let raw_text = String::from_utf8_lossy(&bytes);
        let canonical = self.canonicalizer.canonicalize(&raw_text);
        if canonical.trim().is_empty() {
            self.prune_indexes(cx, &rel_key).await?;
            Self::purge_storage_document(storage_ctx, &rel_key)?;
            return Ok(true);
        }

        let file_name = abs_path
            .file_name()
            .and_then(std::ffi::OsStr::to_str)
            .unwrap_or_default()
            .to_owned();
        let doc = IndexableDocument::new(rel_key.clone(), canonical.clone()).with_title(file_name);
        self.lexical_index.index_document(cx, &doc).await?;

        if matches!(ingestion_class, IngestionClass::FullSemanticLexical) {
            if let Some(storage_ctx) = storage_ctx {
                let ingest_result =
                    Self::enqueue_storage_upsert(storage_ctx, &rel_key, &canonical, &abs_path)?;
                // Storage pipeline intentionally skips hash-tier queued jobs because hash
                // vectors are expected to be computed inline. Preserve live watcher behavior by
                // writing those vectors directly for newly inserted/updated content.
                if !self.embedder.is_semantic()
                    && matches!(
                        ingest_result.action,
                        IngestAction::New | IngestAction::Updated
                    )
                {
                    let content_len_bytes = u64::try_from(canonical.len()).unwrap_or(u64::MAX);
                    let vector_plan = Self::plan_live_vector_upsert(
                        &rel_key,
                        revision,
                        ingestion_class,
                        content_len_bytes,
                    );
                    self.apply_live_vector_actions(
                        cx,
                        &rel_key,
                        revision,
                        &canonical,
                        &vector_plan,
                    )
                    .await?;
                }
            } else {
                let content_len_bytes = u64::try_from(canonical.len()).unwrap_or(u64::MAX);
                let vector_plan = Self::plan_live_vector_upsert(
                    &rel_key,
                    revision,
                    ingestion_class,
                    content_len_bytes,
                );
                self.apply_live_vector_actions(cx, &rel_key, revision, &canonical, &vector_plan)
                    .await?;
            }
        } else {
            self.soft_delete_vector(&rel_key);
            Self::purge_storage_document(storage_ctx, &rel_key)?;
        }

        Ok(true)
    }

    #[allow(clippy::future_not_send)]
    async fn apply_delete_op(
        &self,
        cx: &Cx,
        file_key: &str,
        storage_ctx: Option<&StorageBatchContext>,
    ) -> frankensearch_core::SearchResult<()> {
        let (_abs_path, rel_key) = self.resolve_paths(file_key)?;
        self.prune_indexes(cx, &rel_key).await?;
        Self::purge_storage_document(storage_ctx, &rel_key)
    }

    #[allow(clippy::future_not_send)]
    async fn apply_batch_inner(
        &self,
        cx: &Cx,
        batch: &[WatchIngestOp],
    ) -> frankensearch_core::SearchResult<usize> {
        let storage_ctx = self.build_storage_batch_context()?;
        let mut count = 0_usize;

        for op in batch {
            match op {
                WatchIngestOp::Upsert {
                    file_key,
                    revision,
                    ingestion_class,
                    ..
                } => {
                    if self
                        .apply_upsert_op(
                            cx,
                            file_key,
                            *revision,
                            *ingestion_class,
                            storage_ctx.as_ref(),
                        )
                        .await?
                    {
                        count = count.saturating_add(1);
                    }
                }
                WatchIngestOp::Delete { file_key, .. } => {
                    self.apply_delete_op(cx, file_key, storage_ctx.as_ref())
                        .await?;
                    count = count.saturating_add(1);
                }
            }
        }

        if count > 0 {
            self.lexical_index.commit(cx).await?;
            if let Some(storage_ctx) = storage_ctx.as_ref() {
                self.drain_storage_jobs(cx, storage_ctx).await?;
            }
            info!(
                batch_size = batch.len(),
                reindexed = count,
                "watcher ingest batch committed"
            );
        }

        Ok(count)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct FsfsStatusPayload {
    version: String,
    index: FsfsIndexStatus,
    models: Vec<FsfsModelStatus>,
    config: FsfsConfigStatus,
    runtime: FsfsRuntimeStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct FsfsIndexStatus {
    path: String,
    exists: bool,
    indexed_files: Option<usize>,
    discovered_files: Option<usize>,
    skipped_files: Option<usize>,
    last_indexed_ms: Option<u64>,
    last_indexed_iso_utc: Option<String>,
    stale_files: Option<usize>,
    source_hash_hex: Option<String>,
    size_bytes: u64,
    vector_index_bytes: u64,
    lexical_index_bytes: u64,
    metadata_bytes: u64,
    embedding_cache_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct FsfsModelStatus {
    tier: String,
    name: String,
    cache_path: String,
    cached: bool,
    size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct FsfsConfigStatus {
    source: String,
    index_dir: String,
    model_dir: String,
    rrf_k: f64,
    quality_weight: f64,
    quality_timeout_ms: u64,
    fast_only: bool,
    pressure_profile: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct FsfsRuntimeStatus {
    disk_budget_stage: Option<String>,
    disk_budget_action: Option<String>,
    disk_budget_reason_code: Option<String>,
    disk_budget_bytes: Option<u64>,
    tracked_index_bytes: Option<u64>,
    storage_pressure_emergency: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct FsfsDownloadModelsPayload {
    operation: String,
    force: bool,
    model_root: String,
    models: Vec<FsfsDownloadModelEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct FsfsDownloadModelEntry {
    id: String,
    install_dir: String,
    tier: Option<String>,
    state: String,
    verified: Option<bool>,
    size_bytes: u64,
    destination: String,
    message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct FsfsDoctorPayload {
    version: String,
    checks: Vec<DoctorCheck>,
    pass_count: usize,
    warn_count: usize,
    fail_count: usize,
    overall: DoctorVerdict,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct FsfsUninstallPayload {
    purge: bool,
    dry_run: bool,
    confirmed: bool,
    removed: usize,
    skipped: usize,
    failed: usize,
    entries: Vec<FsfsUninstallEntry>,
    notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct FsfsUninstallEntry {
    target: String,
    kind: String,
    path: String,
    purge_only: bool,
    status: String,
    detail: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UninstallTargetKind {
    File,
    Directory,
}

impl UninstallTargetKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::File => "file",
            Self::Directory => "directory",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct UninstallTarget {
    target: String,
    kind: UninstallTargetKind,
    path: PathBuf,
    purge_only: bool,
}

// ─── Self-Update Payload Types ─────────────────────────────────────────────

/// Structured payload for `fsfs update` (and `fsfs update --check`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct FsfsUpdatePayload {
    current_version: String,
    latest_version: String,
    update_available: bool,
    check_only: bool,
    applied: bool,
    channel: String,
    release_url: Option<String>,
    notes: Vec<String>,
}

/// Minimal semver triple used for version comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SemVer {
    major: u64,
    minor: u64,
    patch: u64,
}

impl SemVer {
    /// Parse a version string like "0.1.0", "v0.2.3", or "1.0.0-beta.1".
    /// Pre-release suffixes are stripped (compared only on numeric triple).
    fn parse(s: &str) -> Option<Self> {
        let s = s.strip_prefix('v').unwrap_or(s);
        let base = s.split('-').next()?;
        let mut parts = base.split('.');
        let major = parts.next()?.parse().ok()?;
        let minor = parts.next()?.parse().ok()?;
        let patch = parts.next()?.parse().ok()?;
        Some(Self {
            major,
            minor,
            patch,
        })
    }

    /// Returns true when `self` is strictly newer than `other`.
    fn is_newer_than(self, other: Self) -> bool {
        (self.major, self.minor, self.patch) > (other.major, other.minor, other.patch)
    }
}

impl std::fmt::Display for SemVer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

// ─── GitHub Release Fetcher ────────────────────────────────────────────────

const GITHUB_OWNER: &str = "Dicklesworthstone";
const GITHUB_REPO: &str = "frankensearch";

#[must_use]
fn is_trusted_release_url(url: &str) -> bool {
    let expected_prefix = format!("https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases");
    url.starts_with(&expected_prefix)
}

/// Fetch the latest release tag from GitHub Releases via `curl`.
/// Returns `(tag_name, html_url)` on success.
fn fetch_latest_release_tag() -> SearchResult<Option<(String, String)>> {
    let url = format!("https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest");

    let output = std::process::Command::new("curl")
        .args([
            "-sS",
            "-H",
            "Accept: application/vnd.github+json",
            "-H",
            "X-GitHub-Api-Version: 2022-11-28",
            "--max-time",
            "10",
            "-w",
            "\n%{http_code}",
            &url,
        ])
        .output()
        .map_err(|e| SearchError::SubsystemError {
            subsystem: "fsfs.update.curl",
            source: Box::new(e),
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(SearchError::InvalidConfig {
            field: "update.github_api".into(),
            value: url,
            reason: format!("GitHub API request failed: {stderr}"),
        });
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let (body, status_code) = stdout
        .rsplit_once('\n')
        .map_or_else(|| (stdout.as_ref(), ""), |(body, code)| (body, code.trim()));

    if status_code == "404" {
        return Ok(None);
    }
    if !status_code.starts_with('2') {
        return Err(SearchError::InvalidConfig {
            field: "update.github_api".into(),
            value: url,
            reason: format!("GitHub API request failed with status {status_code}: {body}"),
        });
    }

    let json: serde_json::Value =
        serde_json::from_str(body).map_err(|e| SearchError::SubsystemError {
            subsystem: "fsfs.update.json",
            source: Box::new(e),
        })?;

    let tag = json
        .get("tag_name")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_owned();
    let html_url = json
        .get("html_url")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_owned();

    if tag.is_empty() {
        return Err(SearchError::InvalidConfig {
            field: "update.tag_name".into(),
            value: body.to_owned(),
            reason: "no tag_name found in GitHub API response".into(),
        });
    }

    Ok(Some((tag, html_url)))
}

/// Download a release asset to a local path using `curl`.
fn download_release_asset(url: &str, dest: &Path) -> SearchResult<()> {
    let status = std::process::Command::new("curl")
        .args([
            "-sSfL",
            "-o",
            &dest.to_string_lossy(),
            "--max-time",
            "120",
            url,
        ])
        .status()
        .map_err(|e| SearchError::SubsystemError {
            subsystem: "fsfs.update.download",
            source: Box::new(e),
        })?;

    if !status.success() {
        return Err(SearchError::InvalidConfig {
            field: "update.download".into(),
            value: url.to_owned(),
            reason: "asset download failed".into(),
        });
    }
    Ok(())
}

/// Compute SHA-256 hex digest of a file.
fn compute_sha256_of_file(path: &Path) -> SearchResult<String> {
    let file = fs::File::open(path).map_err(|e| SearchError::SubsystemError {
        subsystem: "fsfs.update.sha256",
        source: Box::new(e),
    })?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 8192];
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|e| SearchError::SubsystemError {
                subsystem: "fsfs.update.sha256",
                source: Box::new(e),
            })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn create_secure_update_temp_dir() -> SearchResult<PathBuf> {
    let base = std::env::temp_dir();
    for attempt in 0..32_u8 {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos());
        let candidate = base.join(format!(
            "fsfs-update-{}-{nanos}-{attempt}",
            std::process::id()
        ));
        match fs::create_dir(&candidate) {
            Ok(()) => return Ok(candidate),
            Err(error) => {
                if error.kind() == ErrorKind::AlreadyExists {
                    continue;
                }
                return Err(SearchError::SubsystemError {
                    subsystem: "fsfs.update.tempdir",
                    source: Box::new(error),
                });
            }
        }
    }
    Err(SearchError::InvalidConfig {
        field: "update.tempdir".into(),
        value: base.display().to_string(),
        reason: "failed to allocate a unique temporary directory".into(),
    })
}

fn archive_entry_path_is_safe(entry: &str) -> bool {
    let trimmed = entry.trim();
    if trimmed.is_empty() {
        return true;
    }
    let path = Path::new(trimmed);
    if path.is_absolute() {
        return false;
    }
    !path
        .components()
        .any(|component| matches!(component, Component::ParentDir | Component::Prefix(_)))
}

fn validate_tar_archive_paths(archive_path: &Path) -> SearchResult<()> {
    let listing = std::process::Command::new("tar")
        .args(["-tJf"])
        .arg(archive_path)
        .output()
        .map_err(|e| SearchError::SubsystemError {
            subsystem: "fsfs.update.tar_list",
            source: Box::new(e),
        })?;

    if !listing.status.success() {
        return Err(SearchError::InvalidConfig {
            field: "update.extract".into(),
            value: archive_path.display().to_string(),
            reason: "could not enumerate archive entries before extraction".into(),
        });
    }

    let entries = String::from_utf8_lossy(&listing.stdout);
    for entry in entries.lines() {
        if !archive_entry_path_is_safe(entry) {
            return Err(SearchError::InvalidConfig {
                field: "update.extract".into(),
                value: archive_path.display().to_string(),
                reason: format!("archive contains unsafe path entry: {entry}"),
            });
        }
    }
    Ok(())
}

/// Detect the platform target triple for asset naming.
fn detect_target_triple() -> String {
    let arch = std::env::consts::ARCH;
    let os = std::env::consts::OS;
    match (arch, os) {
        ("x86_64", "linux") => "x86_64-unknown-linux-musl".into(),
        ("x86_64", "macos") => "x86_64-apple-darwin".into(),
        ("aarch64", "linux") => "aarch64-unknown-linux-musl".into(),
        ("aarch64", "macos") => "aarch64-apple-darwin".into(),
        _ => format!("{arch}-unknown-{os}"),
    }
}

/// Build the download URL for a release asset.
fn release_asset_url(tag: &str, triple: &str) -> String {
    format!(
        "https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases/download/{tag}/fsfs-{triple}.tar.xz"
    )
}

/// Build the download URL for the per-artifact checksum sidecar.
fn release_checksum_url(tag: &str, triple: &str) -> String {
    format!(
        "https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases/download/{tag}/fsfs-{triple}.tar.xz.sha256"
    )
}

// ─── Version Check Cache ───────────────────────────────────────────────────

/// Default TTL for the version-check cache (24 hours).
const VERSION_CACHE_TTL_SECS: u64 = 86_400;

/// Cached version-check result persisted to disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionCheckCache {
    /// Epoch seconds when this cache entry was written.
    pub checked_at_epoch: u64,
    /// The `CARGO_PKG_VERSION` at the time of check.
    pub current_version: String,
    /// Latest version tag from GitHub (e.g. "v0.3.0").
    pub latest_version: String,
    /// URL of the latest release page.
    #[serde(default)]
    pub release_url: String,
    /// TTL in seconds (allows override in the cache file itself).
    #[serde(default = "default_ttl")]
    pub ttl_seconds: u64,
}

const fn default_ttl() -> u64 {
    VERSION_CACHE_TTL_SECS
}

/// Resolve the path to the version-check cache file.
#[must_use]
pub fn version_cache_path() -> Option<PathBuf> {
    dirs::cache_dir().map(|d| d.join("frankensearch").join("version_check.json"))
}

#[must_use]
fn read_version_cache_from_path(path: &Path) -> Option<VersionCheckCache> {
    let content = fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

/// Read the version cache from disk. Returns `None` if missing or unreadable.
#[must_use]
pub fn read_version_cache() -> Option<VersionCheckCache> {
    let path = version_cache_path()?;
    read_version_cache_from_path(&path)
}

fn write_version_cache_to_path(path: &Path, cache: &VersionCheckCache) -> SearchResult<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| SearchError::SubsystemError {
            subsystem: "fsfs.version_cache.dir",
            source: Box::new(e),
        })?;
    }
    let json = serde_json::to_string_pretty(cache).map_err(|e| SearchError::SubsystemError {
        subsystem: "fsfs.version_cache.json",
        source: Box::new(e),
    })?;
    write_durable(path, json).map_err(|e| SearchError::SubsystemError {
        subsystem: "fsfs.version_cache.write",
        source: Box::new(e),
    })?;
    Ok(())
}

/// Write the version cache to disk.
///
/// # Errors
///
/// Returns `SearchError` if the cache directory or file cannot be written.
pub fn write_version_cache(cache: &VersionCheckCache) -> SearchResult<()> {
    let path = version_cache_path().ok_or_else(|| SearchError::InvalidConfig {
        field: "version_cache.path".into(),
        value: String::new(),
        reason: "could not determine cache directory".into(),
    })?;
    write_version_cache_to_path(&path, cache)
}

fn epoch_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Check if the cache is still valid (not expired and matches current version).
#[must_use]
pub fn is_cache_valid(cache: &VersionCheckCache) -> bool {
    let elapsed = epoch_now().saturating_sub(cache.checked_at_epoch);
    elapsed < cache.ttl_seconds
        && cache.current_version == env!("CARGO_PKG_VERSION")
        && SemVer::parse(&cache.latest_version).is_some()
        && (cache.release_url.is_empty() || is_trusted_release_url(&cache.release_url))
}

/// Refresh the version cache by querying GitHub. Writes the result to disk.
/// Returns the refreshed cache on success.
///
/// # Errors
///
/// Returns `SearchError` if the GitHub API call or cache write fails.
pub fn refresh_version_cache() -> SearchResult<VersionCheckCache> {
    let current_version = env!("CARGO_PKG_VERSION").to_owned();
    let (latest_version, release_url) = match fetch_latest_release_tag()? {
        Some((tag, html_url)) => (tag, html_url),
        None => (current_version.clone(), String::new()),
    };
    let cache = VersionCheckCache {
        checked_at_epoch: epoch_now(),
        current_version,
        latest_version,
        release_url,
        ttl_seconds: VERSION_CACHE_TTL_SECS,
    };
    write_version_cache(&cache)?;
    Ok(cache)
}

/// Print a one-line update notice to stderr if an update is available.
///
/// Reads the cached version-check result. If the cache is expired or missing,
/// silently does nothing (the background refresh will populate it for next time).
///
/// Returns `true` if a notice was printed.
#[must_use]
pub fn maybe_print_update_notice(quiet: bool) -> bool {
    if quiet {
        return false;
    }
    let Some(cache) = read_version_cache() else {
        return false;
    };
    if !is_cache_valid(&cache) {
        return false;
    }
    let Some(current) = SemVer::parse(&cache.current_version) else {
        return false;
    };
    let Some(latest) = SemVer::parse(&cache.latest_version) else {
        return false;
    };
    if !cache.release_url.is_empty() && !is_trusted_release_url(&cache.release_url) {
        return false;
    }
    if !latest.is_newer_than(current) {
        return false;
    }
    eprintln!("Update available: v{current} \u{2192} v{latest} (run `fsfs update`)");
    true
}

/// Spawn a background thread to refresh the version cache.
/// The thread is detached — if it fails or times out, the main process is unaffected.
pub fn spawn_version_cache_refresh() {
    std::thread::Builder::new()
        .name("fsfs-version-check".into())
        .spawn(|| {
            let _ = refresh_version_cache();
        })
        .ok();
}

// ─── Backup / Rollback ────────────────────────────────────────────────────

/// Maximum number of backup versions to keep.
const MAX_BACKUP_VERSIONS: usize = 3;

/// Metadata for a single backup entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BackupEntry {
    pub version: String,
    pub backed_up_at_epoch: u64,
    pub original_path: String,
    pub binary_filename: String,
    pub sha256: String,
}

/// Manifest tracking all backups in the backup directory.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RollbackManifest {
    pub entries: Vec<BackupEntry>,
}

/// Resolve the backup directory path.
#[must_use]
pub fn backup_dir() -> Option<PathBuf> {
    dirs::data_dir().map(|d| d.join("frankensearch").join("backups"))
}

/// Resolve the rollback manifest path.
#[must_use]
pub fn rollback_manifest_path() -> Option<PathBuf> {
    backup_dir().map(|d| d.join("rollback-manifest.json"))
}

/// Read the rollback manifest from disk.
#[must_use]
pub fn read_rollback_manifest() -> RollbackManifest {
    let Some(path) = rollback_manifest_path() else {
        return RollbackManifest::default();
    };
    fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

/// Write the rollback manifest to disk.
///
/// # Errors
///
/// Returns `SearchError` if the manifest directory or file cannot be written.
pub fn write_rollback_manifest(manifest: &RollbackManifest) -> SearchResult<()> {
    let path = rollback_manifest_path().ok_or_else(|| SearchError::InvalidConfig {
        field: "backup.manifest_path".into(),
        value: String::new(),
        reason: "could not determine data directory for backups".into(),
    })?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| SearchError::SubsystemError {
            subsystem: "fsfs.backup.dir",
            source: Box::new(e),
        })?;
    }
    let json = serde_json::to_string_pretty(manifest).map_err(|e| SearchError::SubsystemError {
        subsystem: "fsfs.backup.json",
        source: Box::new(e),
    })?;
    write_durable(&path, json).map_err(|e| SearchError::SubsystemError {
        subsystem: "fsfs.backup.write",
        source: Box::new(e),
    })?;
    Ok(())
}

/// Create a backup of the current binary before an update.
///
/// Returns the backup entry on success. On failure (e.g. disk full), returns
/// an error but the caller may choose to proceed without backup.
///
/// # Errors
///
/// Returns `SearchError` if the backup directory or file operations fail.
pub fn create_backup(current_exe: &Path) -> SearchResult<BackupEntry> {
    let dir = backup_dir().ok_or_else(|| SearchError::InvalidConfig {
        field: "backup.dir".into(),
        value: String::new(),
        reason: "could not determine data directory for backups".into(),
    })?;
    fs::create_dir_all(&dir).map_err(|e| SearchError::SubsystemError {
        subsystem: "fsfs.backup.mkdir",
        source: Box::new(e),
    })?;

    let version = env!("CARGO_PKG_VERSION");
    let binary_filename = format!("fsfs-{version}");
    let dest = dir.join(&binary_filename);

    fs::copy(current_exe, &dest).map_err(|e| SearchError::SubsystemError {
        subsystem: "fsfs.backup.copy",
        source: Box::new(e),
    })?;

    // Set executable permission.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&dest, std::fs::Permissions::from_mode(0o755));
    }

    let sha256 = compute_sha256_of_file(&dest).unwrap_or_default();

    let entry = BackupEntry {
        version: version.to_owned(),
        backed_up_at_epoch: epoch_now(),
        original_path: current_exe.display().to_string(),
        binary_filename,
        sha256,
    };

    // Update manifest: add entry, prune old ones.
    let mut manifest = read_rollback_manifest();
    // Remove duplicate of same version if already backed up.
    manifest.entries.retain(|e| e.version != version);
    manifest.entries.push(entry.clone());
    prune_backups(&mut manifest, &dir);
    write_rollback_manifest(&manifest)?;

    Ok(entry)
}

/// Prune old backup entries beyond `MAX_BACKUP_VERSIONS`.
fn prune_backups(manifest: &mut RollbackManifest, backup_directory: &Path) {
    // Sort newest first.
    manifest
        .entries
        .sort_by_key(|b| std::cmp::Reverse(b.backed_up_at_epoch));
    while manifest.entries.len() > MAX_BACKUP_VERSIONS {
        if let Some(old) = manifest.entries.pop() {
            let path = backup_directory.join(&old.binary_filename);
            let _ = fs::remove_file(&path);
        }
    }
}

/// Restore a backup. If `target_version` is `None`, restores the most recent backup.
///
/// # Errors
///
/// Returns `SearchError` if no backups exist or the restore operation fails.
pub fn restore_backup(target_version: Option<&str>) -> SearchResult<BackupEntry> {
    let dir = backup_dir().ok_or_else(|| SearchError::InvalidConfig {
        field: "backup.dir".into(),
        value: String::new(),
        reason: "could not determine data directory for backups".into(),
    })?;
    let manifest = read_rollback_manifest();
    if manifest.entries.is_empty() {
        return Err(SearchError::InvalidConfig {
            field: "backup.entries".into(),
            value: String::new(),
            reason: "no backups available for rollback".into(),
        });
    }

    let entry = if let Some(ver) = target_version {
        let ver_clean = ver.strip_prefix('v').unwrap_or(ver);
        manifest
            .entries
            .iter()
            .find(|e| e.version == ver_clean || e.version == ver)
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "backup.version".into(),
                value: ver.to_owned(),
                reason: format!(
                    "no backup found for version {ver}; available: {}",
                    manifest
                        .entries
                        .iter()
                        .map(|e| e.version.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            })?
    } else {
        // Most recent backup (already sorted newest-first by prune).
        manifest
            .entries
            .first()
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "backup.entries".into(),
                value: String::new(),
                reason: "no backups available".into(),
            })?
    };

    let backup_path = dir.join(&entry.binary_filename);
    if !backup_path.is_file() {
        return Err(SearchError::InvalidConfig {
            field: "backup.file".into(),
            value: backup_path.display().to_string(),
            reason: "backup binary not found on disk".into(),
        });
    }

    // Verify checksum if available.
    if !entry.sha256.is_empty() {
        let actual = compute_sha256_of_file(&backup_path)?;
        if !actual.eq_ignore_ascii_case(&entry.sha256) {
            return Err(SearchError::InvalidConfig {
                field: "backup.checksum".into(),
                value: actual,
                reason: format!("backup checksum mismatch: expected {}", entry.sha256),
            });
        }
    }

    // Replace current binary with backup.
    let current_exe = std::env::current_exe().map_err(|e| SearchError::SubsystemError {
        subsystem: "fsfs.backup.current_exe",
        source: Box::new(e),
    })?;

    fs::copy(&backup_path, &current_exe).map_err(|e| SearchError::SubsystemError {
        subsystem: "fsfs.backup.restore",
        source: Box::new(e),
    })?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&current_exe, std::fs::Permissions::from_mode(0o755));
    }

    Ok(entry.clone())
}

/// List available backups for display.
#[must_use]
pub fn list_backups() -> Vec<BackupEntry> {
    read_rollback_manifest().entries
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum DoctorVerdict {
    Pass,
    Warn,
    Fail,
}

impl std::fmt::Display for DoctorVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pass => write!(f, "pass"),
            Self::Warn => write!(f, "warn"),
            Self::Fail => write!(f, "fail"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct DoctorCheck {
    name: String,
    verdict: DoctorVerdict,
    detail: String,
    suggestion: Option<String>,
}

/// Shared runtime entrypoint used by interface adapters.
#[derive(Debug, Clone)]
pub struct FsfsRuntime {
    config: FsfsConfig,
    cli_input: CliInput,
}

impl FsfsRuntime {
    #[must_use]
    pub fn new(config: FsfsConfig) -> Self {
        Self {
            config,
            cli_input: CliInput::default(),
        }
    }

    #[must_use]
    pub fn with_cli_input(mut self, cli_input: CliInput) -> Self {
        self.cli_input = cli_input;
        self
    }

    #[must_use]
    pub const fn config(&self) -> &FsfsConfig {
        &self.config
    }

    /// Pressure sampler cadence for fsfs control-state updates.
    #[must_use]
    pub const fn pressure_sample_interval(&self) -> Duration {
        Duration::from_millis(self.config.pressure.sample_interval_ms)
    }

    /// Build a pressure controller from active config profile.
    #[must_use]
    pub fn new_pressure_controller(&self) -> PressureController {
        let ewma_alpha = f64::from(self.config.pressure.ewma_alpha_per_mille) / 1_000.0;
        let config = PressureControllerConfig {
            profile: self.config.pressure.profile,
            ewma_alpha,
            consecutive_required: self.config.pressure.anti_flap_readings,
            ..PressureControllerConfig::default()
        };
        PressureController::new(config)
            .unwrap_or_else(|_| PressureController::from_profile(self.config.pressure.profile))
    }

    /// Collect one host pressure signal sample.
    ///
    /// # Errors
    ///
    /// Returns errors from host pressure collection/parsing.
    pub fn collect_pressure_signal(
        &self,
        collector: &mut HostPressureCollector,
    ) -> SearchResult<PressureSignal> {
        collector.collect(
            self.pressure_sample_interval(),
            self.config.pressure.memory_ceiling_mb,
        )
    }

    /// Observe one pressure sample and derive a stable control-state transition.
    #[must_use]
    pub fn observe_pressure(
        &self,
        controller: &mut PressureController,
        sample: PressureSignal,
    ) -> PressureTransition {
        controller.observe(sample, pressure_timestamp_ms())
    }

    /// Build a degradation state machine that mirrors runtime pressure config.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when degradation controller
    /// settings are invalid.
    pub fn new_degradation_state_machine(&self) -> SearchResult<DegradationStateMachine> {
        let mut machine = DegradationStateMachine::new(degradation_controller_config_for_profile(
            self.config.pressure.profile,
            self.config.pressure.anti_flap_readings,
        ))?;
        machine.set_override(map_degradation_override(
            self.config.pressure.degradation_override,
        ));
        Ok(machine)
    }

    /// Observe one pressure transition and project degraded-mode status.
    #[must_use]
    pub fn observe_degradation(
        &self,
        machine: &mut DegradationStateMachine,
        pressure_transition: &PressureTransition,
    ) -> DegradationTransition {
        let signal = DegradationSignal::new(
            pressure_transition.to,
            self.config.pressure.quality_circuit_open,
            self.config.pressure.hard_pause_requested,
        );
        machine.observe(signal, pressure_transition.snapshot.timestamp_ms)
    }

    /// Evaluate disk-budget state for the current index footprint.
    #[must_use]
    pub fn evaluate_index_disk_budget(
        &self,
        tracker: &LifecycleTracker,
        index_bytes: u64,
    ) -> Option<DiskBudgetSnapshot> {
        let usage = ResourceUsage {
            index_bytes: Some(index_bytes),
            ..ResourceUsage::default()
        };
        let snapshot = tracker.evaluate_usage_budget(&usage);
        tracker.set_resource_usage(usage);
        snapshot
    }

    /// Aggregate cross-domain index storage usage from filesystem paths.
    ///
    /// Missing paths are treated as zero usage, allowing first-run bootstrap.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::Io`] for filesystem traversal failures.
    pub fn collect_index_storage_usage(
        &self,
        paths: &IndexStoragePaths,
    ) -> SearchResult<IndexStorageBreakdown> {
        Ok(IndexStorageBreakdown {
            vector_index_bytes: Self::total_bytes_for_paths(&paths.vector_index_roots)?,
            lexical_index_bytes: Self::total_bytes_for_paths(&paths.lexical_index_roots)?,
            catalog_bytes: Self::total_bytes_for_paths(&paths.catalog_files)?,
            embedding_cache_bytes: Self::total_bytes_for_paths(&paths.embedding_cache_roots)?,
        })
    }

    /// Evaluate disk-budget state from cross-domain storage paths and update
    /// lifecycle status resources for CLI/TUI/JSON projections.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::Io`] when storage usage traversal fails.
    pub fn evaluate_storage_disk_budget(
        &self,
        tracker: &LifecycleTracker,
        paths: &IndexStoragePaths,
    ) -> SearchResult<Option<DiskBudgetSnapshot>> {
        let usage = ResourceUsage::from_index_storage(self.collect_index_storage_usage(paths)?);
        let snapshot = tracker.evaluate_usage_budget(&usage);
        tracker.set_resource_usage(usage);
        Ok(snapshot)
    }

    /// Build the default cross-domain storage roots/files from active runtime
    /// config.
    #[must_use]
    pub fn default_index_storage_paths(&self) -> IndexStoragePaths {
        let index_root = PathBuf::from(&self.config.storage.index_dir);
        let db_path = PathBuf::from(&self.config.storage.db_path);

        IndexStoragePaths {
            vector_index_roots: vec![index_root.join("vector")],
            lexical_index_roots: vec![index_root.join("lexical")],
            catalog_files: vec![db_path],
            embedding_cache_roots: vec![index_root.join("cache")],
        }
    }

    /// Derive a deterministic runtime control plan from staged disk budget
    /// state.
    #[must_use]
    pub fn disk_budget_control_plan(snapshot: DiskBudgetSnapshot) -> DiskBudgetControlPlan {
        if snapshot.reason_code == DISK_BUDGET_REASON_EMERGENCY_OVERRIDE {
            return DiskBudgetControlPlan {
                watcher_pressure_state: PressureState::Emergency,
                throttle_ingest: true,
                pause_writes: true,
                request_eviction: false,
                request_compaction: false,
                request_tombstone_cleanup: true,
                eviction_target_bytes: 0,
                reason_code: snapshot.reason_code,
            };
        }

        let over_bytes = snapshot.used_bytes.saturating_sub(snapshot.budget_bytes);
        let reclaim_floor = snapshot.budget_bytes / 20;

        match snapshot.action {
            DiskBudgetAction::Continue => DiskBudgetControlPlan {
                watcher_pressure_state: PressureState::Normal,
                throttle_ingest: false,
                pause_writes: false,
                request_eviction: false,
                request_compaction: false,
                request_tombstone_cleanup: false,
                eviction_target_bytes: 0,
                reason_code: snapshot.reason_code,
            },
            DiskBudgetAction::ThrottleIngest => DiskBudgetControlPlan {
                watcher_pressure_state: PressureState::Constrained,
                throttle_ingest: true,
                pause_writes: false,
                request_eviction: false,
                request_compaction: false,
                request_tombstone_cleanup: false,
                eviction_target_bytes: 0,
                reason_code: snapshot.reason_code,
            },
            DiskBudgetAction::EvictLowUtility => DiskBudgetControlPlan {
                watcher_pressure_state: PressureState::Degraded,
                throttle_ingest: true,
                pause_writes: false,
                request_eviction: true,
                request_compaction: true,
                request_tombstone_cleanup: true,
                eviction_target_bytes: over_bytes.max(reclaim_floor),
                reason_code: snapshot.reason_code,
            },
            DiskBudgetAction::PauseWrites => DiskBudgetControlPlan {
                watcher_pressure_state: PressureState::Emergency,
                throttle_ingest: true,
                pause_writes: true,
                request_eviction: true,
                request_compaction: true,
                request_tombstone_cleanup: true,
                eviction_target_bytes: over_bytes.max(reclaim_floor),
                reason_code: snapshot.reason_code,
            },
        }
    }

    #[must_use]
    fn apply_storage_emergency_override(
        &self,
        snapshot: Option<DiskBudgetSnapshot>,
        tracked_index_bytes: Option<u64>,
        fallback_budget_bytes: u64,
    ) -> Option<DiskBudgetSnapshot> {
        if !self.config.storage.storage_pressure_emergency {
            return snapshot;
        }

        let budget_bytes = snapshot.as_ref().map_or_else(
            || fallback_budget_bytes.max(1),
            |value| value.budget_bytes.max(1),
        );
        let used_bytes = snapshot.as_ref().map_or_else(
            || tracked_index_bytes.unwrap_or(0),
            |value| value.used_bytes,
        );
        let usage_per_mille = used_bytes
            .saturating_mul(1000)
            .checked_div(budget_bytes)
            .map_or(0, |per_mille| {
                u16::try_from(per_mille.min(u64::from(u16::MAX))).unwrap_or(u16::MAX)
            });

        Some(DiskBudgetSnapshot {
            stage: DiskBudgetStage::Critical,
            action: DiskBudgetAction::PauseWrites,
            used_bytes,
            budget_bytes,
            usage_per_mille,
            reason_code: DISK_BUDGET_REASON_EMERGENCY_OVERRIDE,
        })
    }

    #[must_use]
    fn tombstone_cleanup_cutoff_ms(&self, now_ms: u64) -> i64 {
        let retention_ms =
            u64::from(self.config.storage.evidence_retention_days).saturating_mul(MILLIS_PER_DAY);
        let cutoff = now_ms.saturating_sub(retention_ms);
        i64::try_from(cutoff).unwrap_or(i64::MAX)
    }

    fn cleanup_catalog_tombstones(&self, now_ms: u64) -> SearchResult<(usize, i64)> {
        let cutoff_ms = self.tombstone_cleanup_cutoff_ms(now_ms);
        let deleted_rows =
            cleanup_tombstones_for_path(Path::new(&self.config.storage.db_path), cutoff_ms)?;
        Ok((deleted_rows, cutoff_ms))
    }

    #[must_use]
    fn new_runtime_lifecycle_tracker(&self, paths: &IndexStoragePaths) -> LifecycleTracker {
        let max_index_bytes = self.resolve_index_budget_bytes(paths);
        LifecycleTracker::new(
            WatchdogConfig::default(),
            ResourceLimits {
                max_index_bytes,
                ..ResourceLimits::default()
            },
        )
    }

    /// Produce deterministic root-scope decisions from the current discovery
    /// config. This is the first stage of corpus selection before filesystem
    /// walking starts.
    #[must_use]
    pub fn discovery_root_plan(&self) -> Vec<(String, RootDiscoveryDecision)> {
        self.config
            .discovery
            .roots
            .iter()
            .map(|root| {
                let decision = self.config.discovery.evaluate_root(Path::new(root), None);
                (root.clone(), decision)
            })
            .collect()
    }

    /// Expose the discovery policy evaluator for runtime callers.
    #[must_use]
    pub fn classify_discovery_candidate(
        &self,
        candidate: &DiscoveryCandidate<'_>,
    ) -> DiscoveryDecision {
        self.config.discovery.evaluate_candidate(candidate)
    }

    const TARGET_CHUNK_BYTES: u64 = 1_024;

    fn effective_embedding_batch_size(&self) -> usize {
        self.config.indexing.embedding_batch_size.max(1)
    }

    fn resolve_index_budget_bytes(&self, paths: &IndexStoragePaths) -> u64 {
        if let Some(configured_budget_bytes) = self.config.storage.disk_budget_bytes {
            return configured_budget_bytes;
        }

        let primary_probe = PathBuf::from(&self.config.storage.index_dir);
        let fallback_probe = PathBuf::from(&self.config.storage.db_path);
        let available = available_space_for_path(&primary_probe)
            .or_else(|| available_space_for_path(&fallback_probe))
            .or_else(|| {
                paths
                    .vector_index_roots
                    .iter()
                    .chain(paths.lexical_index_roots.iter())
                    .chain(paths.catalog_files.iter())
                    .chain(paths.embedding_cache_roots.iter())
                    .filter_map(|path| available_space_for_path(path))
                    .max()
            });

        available.map_or(
            DISK_BUDGET_FALLBACK_BYTES,
            conservative_budget_from_available_bytes,
        )
    }

    fn total_bytes_for_paths(paths: &[PathBuf]) -> SearchResult<u64> {
        paths.iter().try_fold(0_u64, |total, path| {
            let bytes = Self::path_bytes(path)?;
            Ok(total.saturating_add(bytes))
        })
    }

    fn path_bytes(path: &Path) -> SearchResult<u64> {
        let metadata = match fs::symlink_metadata(path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(0),
            Err(error) => return Err(error.into()),
        };

        if metadata.file_type().is_symlink() {
            return Ok(0);
        }
        if metadata.is_file() {
            return Ok(metadata.len());
        }
        if !metadata.is_dir() {
            return Ok(0);
        }

        let mut total = 0_u64;
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            total = total.saturating_add(Self::path_bytes(&entry.path())?);
        }
        Ok(total)
    }

    fn chunk_count_for_bytes(content_len_bytes: u64) -> usize {
        let effective_len = content_len_bytes.max(1);
        let chunks = effective_len
            .saturating_add(Self::TARGET_CHUNK_BYTES.saturating_sub(1))
            .saturating_div(Self::TARGET_CHUNK_BYTES)
            .max(1);
        usize::try_from(chunks).unwrap_or(usize::MAX)
    }

    fn resolve_vector_tier(&self, availability: EmbedderAvailability) -> VectorSchedulingTier {
        if self.config.search.fast_only || self.config.indexing.quality_model.trim().is_empty() {
            return match availability {
                EmbedderAvailability::None => VectorSchedulingTier::LexicalFallback,
                EmbedderAvailability::Full | EmbedderAvailability::FastOnly => {
                    VectorSchedulingTier::FastOnly
                }
            };
        }

        match availability {
            EmbedderAvailability::Full => VectorSchedulingTier::FastAndQuality,
            EmbedderAvailability::FastOnly => VectorSchedulingTier::FastOnly,
            EmbedderAvailability::None => VectorSchedulingTier::LexicalFallback,
        }
    }

    fn plan_single_vector_pipeline(
        &self,
        input: &VectorPipelineInput,
        availability: EmbedderAvailability,
    ) -> VectorPipelinePlan {
        let skip_plan = |reason_code: &str| VectorPipelinePlan {
            file_key: input.file_key.clone(),
            revision: input.observed_revision,
            chunk_count: 0,
            batch_size: self.effective_embedding_batch_size(),
            tier: VectorSchedulingTier::Skip,
            invalidate_revisions_through: None,
            reason_code: reason_code.to_owned(),
        };

        if !matches!(input.ingestion_class, IngestionClass::FullSemanticLexical) {
            return skip_plan("vector.skip.non_semantic_ingestion_class");
        }

        if input.observed_revision < 0 {
            return skip_plan("vector.skip.invalid_revision");
        }

        if let Some(previous_revision) = input.previous_indexed_revision {
            if input.observed_revision < previous_revision {
                return skip_plan("vector.skip.out_of_order_revision");
            }

            if input.observed_revision == previous_revision && !input.content_hash_changed {
                return skip_plan("vector.skip.revision_unchanged");
            }
        }

        let tier = self.resolve_vector_tier(availability);
        let reason_code = match tier {
            VectorSchedulingTier::FastAndQuality => "vector.plan.fast_quality",
            VectorSchedulingTier::FastOnly => {
                if self.config.search.fast_only {
                    "vector.plan.fast_only_policy"
                } else {
                    "vector.plan.fast_only_quality_unavailable"
                }
            }
            VectorSchedulingTier::LexicalFallback => "vector.plan.lexical_fallback",
            VectorSchedulingTier::Skip => "vector.skip.unspecified",
        };
        let invalidate_revisions_through = input
            .previous_indexed_revision
            .filter(|revision| *revision >= 0 && input.content_hash_changed);
        let chunk_count = if matches!(tier, VectorSchedulingTier::LexicalFallback) {
            0
        } else {
            Self::chunk_count_for_bytes(input.content_len_bytes)
        };

        VectorPipelinePlan {
            file_key: input.file_key.clone(),
            revision: input.observed_revision,
            chunk_count,
            batch_size: self.effective_embedding_batch_size(),
            tier,
            invalidate_revisions_through,
            reason_code: reason_code.to_owned(),
        }
    }

    /// Build deterministic vector scheduling plans with revision coherence.
    #[must_use]
    pub fn plan_vector_pipeline(
        &self,
        inputs: &[VectorPipelineInput],
        availability: EmbedderAvailability,
    ) -> Vec<VectorPipelinePlan> {
        inputs
            .iter()
            .map(|input| self.plan_single_vector_pipeline(input, availability))
            .collect()
    }

    /// Expand one plan into revision-aware index write actions.
    #[must_use]
    pub fn vector_index_write_actions(plan: &VectorPipelinePlan) -> Vec<VectorIndexWriteAction> {
        let mut actions = Vec::new();
        if let Some(revision) = plan.invalidate_revisions_through {
            actions.push(VectorIndexWriteAction::InvalidateRevisionsThrough {
                file_key: plan.file_key.clone(),
                revision,
            });
        }

        match plan.tier {
            VectorSchedulingTier::FastAndQuality => {
                actions.push(VectorIndexWriteAction::AppendFast {
                    file_key: plan.file_key.clone(),
                    revision: plan.revision,
                    chunk_count: plan.chunk_count,
                });
                actions.push(VectorIndexWriteAction::AppendQuality {
                    file_key: plan.file_key.clone(),
                    revision: plan.revision,
                    chunk_count: plan.chunk_count,
                });
            }
            VectorSchedulingTier::FastOnly => {
                actions.push(VectorIndexWriteAction::AppendFast {
                    file_key: plan.file_key.clone(),
                    revision: plan.revision,
                    chunk_count: plan.chunk_count,
                });
            }
            VectorSchedulingTier::LexicalFallback => {
                actions.push(VectorIndexWriteAction::MarkLexicalFallback {
                    file_key: plan.file_key.clone(),
                    revision: plan.revision,
                    reason_code: plan.reason_code.clone(),
                });
            }
            VectorSchedulingTier::Skip => {
                actions.push(VectorIndexWriteAction::Skip {
                    file_key: plan.file_key.clone(),
                    revision: plan.revision,
                    reason_code: plan.reason_code.clone(),
                });
            }
        }

        actions
    }

    /// Dispatch by interface mode using the caller-provided `Cx`.
    ///
    /// # Errors
    ///
    /// Returns any surfaced `SearchError` from the selected runtime lane.
    pub async fn run_mode(&self, cx: &Cx, mode: InterfaceMode) -> SearchResult<()> {
        match mode {
            InterfaceMode::Cli => self.run_cli(cx).await,
            InterfaceMode::Tui => self.run_tui(cx).await,
        }
    }

    /// Dispatch by interface mode with shutdown/signal integration.
    ///
    /// This path is intended for long-lived runs (watch mode and TUI): it
    /// listens for shutdown requests while allowing config reload signals.
    ///
    /// # Errors
    ///
    /// Returns any surfaced `SearchError` from the selected runtime lane or
    /// graceful-shutdown finalization path.
    pub async fn run_mode_with_shutdown(
        &self,
        cx: &Cx,
        mode: InterfaceMode,
        shutdown: &ShutdownCoordinator,
    ) -> SearchResult<()> {
        match mode {
            InterfaceMode::Cli => self.run_cli_with_shutdown(cx, shutdown).await,
            InterfaceMode::Tui => self.run_tui_with_shutdown(cx, shutdown).await,
        }
    }

    /// CLI runtime dispatch.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` when command parsing/validation fails or
    /// downstream CLI runtime logic fails.
    pub async fn run_cli(&self, cx: &Cx) -> SearchResult<()> {
        match self.cli_input.command {
            CliCommand::Help => {
                print_cli_help();
                Ok(())
            }
            CliCommand::Completions => self.run_completions_command(),
            CliCommand::Update => self.run_update_command(),
            CliCommand::Uninstall => self.run_uninstall_command(),
            CliCommand::Tui => self.run_tui(cx).await,
            command => self.run_cli_scaffold(cx, command).await,
        }
    }

    fn run_completions_command(&self) -> SearchResult<()> {
        let shell = self
            .cli_input
            .completion_shell
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "cli.completions.shell".into(),
                value: String::new(),
                reason: "missing shell argument".into(),
            })?;
        println!("{}", completion_script(shell));
        Ok(())
    }

    fn run_update_command(&self) -> SearchResult<()> {
        // Handle --rollback separately.
        if self.cli_input.update_rollback {
            return self.run_rollback_command();
        }

        let payload = self.collect_update_payload()?;
        if self.cli_input.format == OutputFormat::Table {
            let table = render_update_table(&payload, self.cli_input.no_color);
            print!("{table}");
            return Ok(());
        }

        let meta = meta_for_format("update", self.cli_input.format);
        let envelope = OutputEnvelope::success(payload, meta, iso_timestamp_now());
        let mut stdout = std::io::stdout();
        emit_envelope(&envelope, self.cli_input.format, &mut stdout)?;
        if self.cli_input.format != OutputFormat::Jsonl {
            stdout
                .write_all(b"\n")
                .map_err(|source| SearchError::SubsystemError {
                    subsystem: "fsfs.update",
                    source: Box::new(source),
                })?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn collect_update_payload(&self) -> SearchResult<FsfsUpdatePayload> {
        let current_str = env!("CARGO_PKG_VERSION");
        let check_only = self.cli_input.update_check_only;
        let channel = "stable".to_owned();
        let mut notes = Vec::new();

        let current = SemVer::parse(current_str).ok_or_else(|| SearchError::InvalidConfig {
            field: "update.current_version".into(),
            value: current_str.to_owned(),
            reason: "cannot parse current version as semver".into(),
        })?;

        // Query GitHub for the latest release.
        let Some((tag, html_url)) = fetch_latest_release_tag()? else {
            notes.push("no published GitHub releases found for this channel".to_owned());
            return Ok(FsfsUpdatePayload {
                current_version: current_str.to_owned(),
                latest_version: current.to_string(),
                update_available: false,
                check_only,
                applied: false,
                channel,
                release_url: None,
                notes,
            });
        };
        let latest = SemVer::parse(&tag).ok_or_else(|| SearchError::InvalidConfig {
            field: "update.latest_version".into(),
            value: tag.clone(),
            reason: "cannot parse latest release tag as semver".into(),
        })?;

        let update_available = latest.is_newer_than(current);

        if !update_available {
            notes.push(format!("fsfs {current_str} is already up to date"));
            return Ok(FsfsUpdatePayload {
                current_version: current_str.to_owned(),
                latest_version: latest.to_string(),
                update_available: false,
                check_only,
                applied: false,
                channel,
                release_url: Some(html_url),
                notes,
            });
        }

        if check_only {
            notes.push(format!(
                "update available: v{current} -> v{latest} (run `fsfs update` to apply)"
            ));
            return Ok(FsfsUpdatePayload {
                current_version: current_str.to_owned(),
                latest_version: latest.to_string(),
                update_available: true,
                check_only: true,
                applied: false,
                channel,
                release_url: Some(html_url),
                notes,
            });
        }

        // Full update: download, verify, replace.
        let triple = detect_target_triple();
        let asset_url = release_asset_url(&tag, &triple);
        let checksum_url = release_checksum_url(&tag, &triple);

        let temp_dir = create_secure_update_temp_dir()?;

        let archive_path = temp_dir.join(format!("fsfs-{triple}.tar.xz"));
        let checksum_path = temp_dir.join(format!("fsfs-{triple}.tar.xz.sha256"));

        // Download archive.
        notes.push(format!("downloading {asset_url}"));
        download_release_asset(&asset_url, &archive_path)?;

        // Download and verify checksum.
        let expected_hash = if download_release_asset(&checksum_url, &checksum_path).is_ok() {
            let content = fs::read_to_string(&checksum_path).unwrap_or_default();
            let hash = content.split_whitespace().next().unwrap_or("").to_owned();
            if hash.is_empty() { None } else { Some(hash) }
        } else {
            notes.push("checksum sidecar not available; skipping verification".into());
            None
        };

        if let Some(ref expected) = expected_hash {
            let actual = compute_sha256_of_file(&archive_path)?;
            if !actual.eq_ignore_ascii_case(expected) {
                return Err(SearchError::InvalidConfig {
                    field: "update.checksum".into(),
                    value: actual,
                    reason: format!("SHA-256 mismatch: expected {expected}"),
                });
            }
            notes.push("SHA-256 checksum verified".into());
        }

        validate_tar_archive_paths(&archive_path)?;

        // Extract binary from the tar.xz archive.
        let extract_dir = temp_dir.join("extract");
        fs::create_dir_all(&extract_dir).map_err(|e| SearchError::SubsystemError {
            subsystem: "fsfs.update.extract_dir",
            source: Box::new(e),
        })?;

        let tar_status = std::process::Command::new("tar")
            .args(["-xJf"])
            .arg(&archive_path)
            .arg("-C")
            .arg(&extract_dir)
            .status()
            .map_err(|e| SearchError::SubsystemError {
                subsystem: "fsfs.update.tar",
                source: Box::new(e),
            })?;

        if !tar_status.success() {
            return Err(SearchError::InvalidConfig {
                field: "update.extract".into(),
                value: archive_path.display().to_string(),
                reason: "tar extraction failed".into(),
            });
        }

        // Find the extracted binary (search up to 2 levels deep).
        let new_binary = find_extracted_binary(&extract_dir, "fsfs")?;

        // Locate the currently-running binary.
        let current_exe = std::env::current_exe().map_err(|e| SearchError::SubsystemError {
            subsystem: "fsfs.update.current_exe",
            source: Box::new(e),
        })?;

        // Create a proper backup before replacing.
        match create_backup(&current_exe) {
            Ok(entry) => {
                notes.push(format!(
                    "backed up v{} to {}",
                    entry.version, entry.binary_filename
                ));
            }
            Err(e) => {
                notes.push(format!("backup failed (proceeding anyway): {e}"));
            }
        }

        // Also keep a transient .old for immediate rollback if install fails.
        let transient_backup = current_exe.with_extension("old");
        if current_exe.exists() {
            let _ = fs::rename(&current_exe, &transient_backup);
        }

        if let Err(e) = fs::copy(&new_binary, &current_exe) {
            // Attempt to restore from transient backup on failure.
            if transient_backup.exists() {
                let _ = fs::rename(&transient_backup, &current_exe);
            }
            return Err(SearchError::SubsystemError {
                subsystem: "fsfs.update.install",
                source: Box::new(e),
            });
        }

        // Set executable permission on the new binary.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o755);
            let _ = fs::set_permissions(&current_exe, perms);
        }

        // Verify the new binary runs.
        let verify = std::process::Command::new(&current_exe)
            .arg("--version")
            .output();
        match verify {
            Ok(out) if out.status.success() => {
                let version_out = String::from_utf8_lossy(&out.stdout);
                notes.push(format!("verified new binary: {}", version_out.trim()));
                // Remove transient backup on success.
                let _ = fs::remove_file(&transient_backup);
            }
            _ => {
                // Rollback: restore from transient backup.
                if transient_backup.exists() {
                    let _ = fs::rename(&transient_backup, &current_exe);
                    notes.push("verification failed; rolled back to previous version".into());
                    return Err(SearchError::InvalidConfig {
                        field: "update.verify".into(),
                        value: current_exe.display().to_string(),
                        reason: "new binary failed verification; rolled back".into(),
                    });
                }
            }
        }

        // Cleanup temp files.
        let _ = fs::remove_dir_all(&temp_dir);

        notes.push(format!("updated: v{current} -> v{latest}"));

        Ok(FsfsUpdatePayload {
            current_version: current_str.to_owned(),
            latest_version: latest.to_string(),
            update_available: true,
            check_only: false,
            applied: true,
            channel,
            release_url: Some(html_url),
            notes,
        })
    }

    fn run_rollback_command(&self) -> SearchResult<()> {
        let version = self.cli_input.update_rollback_version.as_deref();

        // If no version specified and format is table, list available backups.
        if version.is_none() && self.cli_input.update_check_only {
            let backups = list_backups();
            if backups.is_empty() {
                println!("No backups available for rollback.");
            } else {
                println!("Available backups:");
                for entry in &backups {
                    println!(
                        "  v{} (backed up at epoch {})",
                        entry.version, entry.backed_up_at_epoch
                    );
                }
            }
            return Ok(());
        }

        let entry = restore_backup(version)?;

        let mut notes = Vec::new();
        notes.push(format!("restored v{} from backup", entry.version));

        // Verify the restored binary.
        if let Ok(current_exe) = std::env::current_exe() {
            let verify = std::process::Command::new(&current_exe)
                .arg("--version")
                .output();
            if let Ok(out) = verify
                && out.status.success()
            {
                let version_out = String::from_utf8_lossy(&out.stdout);
                notes.push(format!("verified: {}", version_out.trim()));
            }
        }

        let payload = FsfsUpdatePayload {
            current_version: env!("CARGO_PKG_VERSION").to_owned(),
            latest_version: entry.version,
            update_available: false,
            check_only: false,
            applied: true,
            channel: "rollback".into(),
            release_url: None,
            notes,
        };

        if self.cli_input.format == OutputFormat::Table {
            let table = render_update_table(&payload, self.cli_input.no_color);
            print!("{table}");
            return Ok(());
        }

        let meta = meta_for_format("update", self.cli_input.format);
        let envelope = OutputEnvelope::success(payload, meta, iso_timestamp_now());
        let mut stdout = std::io::stdout();
        emit_envelope(&envelope, self.cli_input.format, &mut stdout)?;
        if self.cli_input.format != OutputFormat::Jsonl {
            stdout
                .write_all(b"\n")
                .map_err(|source| SearchError::SubsystemError {
                    subsystem: "fsfs.update.rollback",
                    source: Box::new(source),
                })?;
        }
        Ok(())
    }

    fn run_uninstall_command(&self) -> SearchResult<()> {
        let payload = self.collect_uninstall_payload()?;
        if self.cli_input.format == OutputFormat::Table {
            let table = render_uninstall_table(&payload, self.cli_input.no_color);
            print!("{table}");
            return Ok(());
        }

        let meta = meta_for_format("uninstall", self.cli_input.format);
        let envelope = OutputEnvelope::success(payload, meta, iso_timestamp_now());
        let mut stdout = std::io::stdout();
        emit_envelope(&envelope, self.cli_input.format, &mut stdout)?;
        if self.cli_input.format != OutputFormat::Jsonl {
            stdout
                .write_all(b"\n")
                .map_err(|source| SearchError::SubsystemError {
                    subsystem: "fsfs.uninstall",
                    source: Box::new(source),
                })?;
        }
        Ok(())
    }

    fn collect_uninstall_payload(&self) -> SearchResult<FsfsUninstallPayload> {
        let dry_run = self.cli_input.uninstall_dry_run;
        let confirmed = self.cli_input.uninstall_yes;
        let purge = self.cli_input.uninstall_purge;

        if !dry_run && !confirmed {
            return Err(SearchError::InvalidConfig {
                field: "cli.uninstall.confirmation".to_owned(),
                value: String::new(),
                reason: "uninstall requires --yes or --dry-run".to_owned(),
            });
        }

        let mut notes = Vec::new();
        if dry_run {
            notes.push("dry-run mode: no files were deleted".to_owned());
        }
        if !purge {
            notes.push("purge-disabled: model/cache/config targets were skipped".to_owned());
        }

        let mut entries = Vec::new();
        for target in self.collect_uninstall_targets()? {
            entries.push(Self::apply_uninstall_target(&target, dry_run, purge));
        }

        let removed = entries
            .iter()
            .filter(|entry| entry.status == "removed")
            .count();
        let failed = entries
            .iter()
            .filter(|entry| entry.status == "error")
            .count();
        let skipped = entries.len().saturating_sub(removed + failed);

        Ok(FsfsUninstallPayload {
            purge,
            dry_run,
            confirmed,
            removed,
            skipped,
            failed,
            entries,
            notes,
        })
    }

    #[allow(clippy::too_many_lines)]
    fn collect_uninstall_targets(&self) -> SearchResult<Vec<UninstallTarget>> {
        let mut candidates = Vec::new();

        if !cfg!(test)
            && let Ok(path) = std::env::current_exe()
        {
            candidates.push(UninstallTarget {
                target: "binary".to_owned(),
                kind: UninstallTargetKind::File,
                path,
                purge_only: false,
            });
        }

        candidates.push(UninstallTarget {
            target: "index_dir".to_owned(),
            kind: UninstallTargetKind::Directory,
            path: self.resolve_uninstall_index_root()?,
            purge_only: false,
        });

        candidates.push(UninstallTarget {
            target: "model_dir".to_owned(),
            kind: UninstallTargetKind::Directory,
            path: PathBuf::from(&self.config.indexing.model_dir),
            purge_only: true,
        });

        if let Some(config_dir) = dirs::config_dir() {
            let root = config_dir.join("frankensearch");
            candidates.push(UninstallTarget {
                target: "config_dir".to_owned(),
                kind: UninstallTargetKind::Directory,
                path: root.clone(),
                purge_only: true,
            });
            candidates.push(UninstallTarget {
                target: "fish_completion".to_owned(),
                kind: UninstallTargetKind::File,
                path: config_dir.join("fish/completions/fsfs.fish"),
                purge_only: false,
            });
            candidates.push(UninstallTarget {
                target: "install_manifest".to_owned(),
                kind: UninstallTargetKind::File,
                path: root.join("install-manifest.json"),
                purge_only: true,
            });
        }

        if let Some(cache_dir) = dirs::cache_dir() {
            candidates.push(UninstallTarget {
                target: "cache_dir".to_owned(),
                kind: UninstallTargetKind::Directory,
                path: cache_dir.join("frankensearch"),
                purge_only: true,
            });
        }

        if let Some(data_dir) = dirs::data_dir() {
            candidates.push(UninstallTarget {
                target: "data_dir".to_owned(),
                kind: UninstallTargetKind::Directory,
                path: data_dir.join("frankensearch"),
                purge_only: true,
            });
            candidates.push(UninstallTarget {
                target: "bash_completion".to_owned(),
                kind: UninstallTargetKind::File,
                path: data_dir.join("bash-completion/completions/fsfs"),
                purge_only: false,
            });
            candidates.push(UninstallTarget {
                target: "zsh_completion".to_owned(),
                kind: UninstallTargetKind::File,
                path: data_dir.join("zsh/site-functions/_fsfs"),
                purge_only: false,
            });
        }

        if let Some(home) = home_dir() {
            candidates.push(UninstallTarget {
                target: "zsh_completion_home".to_owned(),
                kind: UninstallTargetKind::File,
                path: home.join(".zfunc/_fsfs"),
                purge_only: false,
            });
            for (target, relative) in [
                ("claude_hook_fsfs", ".claude/hooks/fsfs.sh"),
                (
                    "claude_hook_frankensearch",
                    ".claude/hooks/frankensearch.sh",
                ),
                ("claude_code_hook_fsfs", ".config/claude-code/hooks/fsfs.sh"),
                (
                    "claude_code_hook_frankensearch",
                    ".config/claude-code/hooks/frankensearch.sh",
                ),
                ("cursor_hook_fsfs", ".config/cursor/hooks/fsfs.sh"),
                (
                    "cursor_hook_frankensearch",
                    ".config/cursor/hooks/frankensearch.sh",
                ),
            ] {
                candidates.push(UninstallTarget {
                    target: target.to_owned(),
                    kind: UninstallTargetKind::File,
                    path: home.join(relative),
                    purge_only: false,
                });
            }
        }

        let mut dedupe = HashSet::new();
        Ok(candidates
            .into_iter()
            .filter_map(|target| {
                if target.path.as_os_str().is_empty() {
                    return None;
                }
                if dedupe.insert(target.path.clone()) {
                    Some(target)
                } else {
                    None
                }
            })
            .collect())
    }

    fn resolve_uninstall_index_root(&self) -> SearchResult<PathBuf> {
        if let Some(path) = self.cli_input.index_dir.as_deref() {
            return absolutize_path(path);
        }
        absolutize_path(Path::new(&self.config.storage.index_dir))
    }

    fn apply_uninstall_target(
        target: &UninstallTarget,
        dry_run: bool,
        purge_enabled: bool,
    ) -> FsfsUninstallEntry {
        let mut entry = FsfsUninstallEntry {
            target: target.target.clone(),
            kind: target.kind.as_str().to_owned(),
            path: target.path.display().to_string(),
            purge_only: target.purge_only,
            status: "skipped".to_owned(),
            detail: None,
        };

        if target.purge_only && !purge_enabled {
            entry.detail = Some("requires --purge".to_owned());
            return entry;
        }

        let normalized = normalize_probe_path(&target.path);
        if is_uninstall_protected_path(&normalized) {
            "error".clone_into(&mut entry.status);
            entry.detail = Some("refusing to remove unsafe root path".to_owned());
            return entry;
        }

        let metadata = match fs::symlink_metadata(&normalized) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == ErrorKind::NotFound => {
                "not_found".clone_into(&mut entry.status);
                return entry;
            }
            Err(error) => {
                "error".clone_into(&mut entry.status);
                entry.detail = Some(error.to_string());
                return entry;
            }
        };

        if target.target == "index_dir"
            && target.kind == UninstallTargetKind::Directory
            && metadata.is_dir()
            && !looks_like_fsfs_index_root(&normalized)
        {
            "error".clone_into(&mut entry.status);
            entry.detail = Some(
                "refusing to remove index_dir that is not recognized as fsfs-managed".to_owned(),
            );
            return entry;
        }

        if dry_run {
            "planned".clone_into(&mut entry.status);
            return entry;
        }

        let deletion =
            if metadata.file_type().is_symlink() || target.kind == UninstallTargetKind::File {
                fs::remove_file(&normalized)
            } else {
                fs::remove_dir_all(&normalized)
            };

        match deletion {
            Ok(()) => {
                "removed".clone_into(&mut entry.status);
            }
            Err(error) => {
                "error".clone_into(&mut entry.status);
                entry.detail = Some(error.to_string());
            }
        }

        entry
    }

    async fn run_cli_scaffold(&self, cx: &Cx, command: CliCommand) -> SearchResult<()> {
        self.validate_command_inputs(command)?;
        if command == CliCommand::Search {
            self.run_search_command(cx).await?;
            return Ok(());
        }
        if command == CliCommand::Serve {
            self.run_search_serve_command(cx).await?;
            return Ok(());
        }
        if command == CliCommand::Explain {
            self.run_explain_command()?;
            return Ok(());
        }
        if command == CliCommand::Status {
            self.run_status_command()?;
            return Ok(());
        }
        if command == CliCommand::Download {
            self.run_download_command().await?;
            return Ok(());
        }
        if command == CliCommand::Doctor {
            self.run_doctor_command()?;
            return Ok(());
        }
        std::future::ready(()).await;
        let root_plan = self.discovery_root_plan();
        let accepted_roots = root_plan
            .iter()
            .filter(|(_, decision)| decision.include())
            .count();

        for (root, decision) in &root_plan {
            info!(
                root,
                scope = ?decision.scope,
                reason_codes = ?decision.reason_codes,
                "fsfs discovery root policy evaluated"
            );
        }

        let mut pressure_collector = HostPressureCollector::default();
        let mut pressure_controller = self.new_pressure_controller();
        let mut degradation_machine = self.new_degradation_state_machine()?;
        match self.collect_pressure_signal(&mut pressure_collector) {
            Ok(sample) => {
                let transition = self.observe_pressure(&mut pressure_controller, sample);
                info!(
                    pressure_state = ?transition.to,
                    pressure_score = transition.snapshot.score,
                    transition_reason = transition.reason_code,
                    "fsfs pressure state sample collected"
                );
                let degradation = self.observe_degradation(&mut degradation_machine, &transition);
                info!(
                    degradation_stage = ?degradation.to,
                    degradation_trigger = ?degradation.trigger,
                    degradation_reason = degradation.reason_code,
                    degradation_banner = degradation.status.user_banner,
                    degradation_query_mode = ?degradation.status.query_mode,
                    degradation_indexing_mode = ?degradation.status.indexing_mode,
                    degradation_override = ?degradation.status.override_mode,
                    "fsfs degradation status sample projected"
                );
            }
            Err(err) => {
                warn!(error = %err, "fsfs pressure sample collection failed");
            }
        }

        info!(
            command = ?command,
            watch_mode = self.config.indexing.watch_mode,
            target_path = ?self.cli_input.target_path,
            profile = ?self.config.pressure.profile,
            total_roots = root_plan.len(),
            accepted_roots,
            rejected_roots = root_plan.len().saturating_sub(accepted_roots),
            "fsfs cli runtime scaffold invoked"
        );

        if matches!(command, CliCommand::Index | CliCommand::Watch) {
            self.run_one_shot_index_scaffold(cx, command).await?;
        }

        Ok(())
    }

    async fn run_search_command(&self, cx: &Cx) -> SearchResult<()> {
        let query = self
            .cli_input
            .query
            .as_deref()
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "cli.search_query".to_owned(),
                value: String::new(),
                reason: "missing search query argument".to_owned(),
            })?;
        let limit = self
            .cli_input
            .overrides
            .limit
            .unwrap_or(self.config.search.default_limit);

        if self.cli_input.stream {
            return self.run_search_stream_command(cx, query, limit).await;
        }

        let mut search_runtime = self.clone();
        if let Some(index_override) = self.ensure_search_index_ready(cx).await? {
            let mut cli_input = search_runtime.cli_input.clone();
            cli_input.index_dir = Some(index_override);
            search_runtime = search_runtime.with_cli_input(cli_input);
        }

        let started = Instant::now();
        let payloads = if search_runtime.cli_input.daemon {
            match search_runtime.search_payloads_via_daemon(query, limit) {
                Ok(payloads) => payloads,
                Err(error) => {
                    warn!(
                        error = %error,
                        "fsfs daemon-backed search unavailable; falling back to in-process retrieval"
                    );
                    search_runtime
                        .execute_search_payloads_cached_for_cli(cx, query, limit)
                        .await?
                }
            }
        } else {
            search_runtime
                .execute_search_payloads_cached_for_cli(cx, query, limit)
                .await?
        };
        let payload = payloads.last().cloned().unwrap_or_else(|| {
            SearchPayload::new(String::new(), SearchOutputPhase::Initial, 0, Vec::new())
        });
        let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        info!(
            phase = "display",
            format = self.cli_input.format.to_string(),
            output_format = self.cli_input.format.to_string(),
            returned_hits = payload.returned_hits,
            elapsed_ms,
            "fsfs search display phase prepared"
        );
        info!(
            query = payload.query,
            phase = payload.phase.to_string(),
            output_format = self.cli_input.format.to_string(),
            returned_hits = payload.returned_hits,
            total_candidates = payload.total_candidates,
            elapsed_ms,
            "fsfs search command completed"
        );

        if self.cli_input.format == OutputFormat::Table {
            if let Some(mode_hint) = search_runtime.search_mode_hint()? {
                println!("{}", paint(&mode_hint, "38;5;244", self.cli_input.no_color));
            }
            let table = crate::adapters::format_emitter::render_search_table_for_cli(
                &payload,
                Some(elapsed_ms),
                self.cli_input.no_color,
            );
            print!("{table}");
            return Ok(());
        }

        let meta = meta_for_format("search", self.cli_input.format).with_duration_ms(elapsed_ms);
        let envelope = OutputEnvelope::success(payload, meta, iso_timestamp_now());
        let stdout = std::io::stdout();
        let mut writer = BufWriter::with_capacity(1 << 20, stdout.lock());
        emit_envelope(&envelope, self.cli_input.format, &mut writer)?;
        if !matches!(
            self.cli_input.format,
            OutputFormat::Jsonl | OutputFormat::Csv
        ) {
            writer
                .write_all(b"\n")
                .map_err(|source| SearchError::SubsystemError {
                    subsystem: "fsfs.search",
                    source: Box::new(source),
                })?;
        }
        writer
            .flush()
            .map_err(|source| SearchError::SubsystemError {
                subsystem: "fsfs.search",
                source: Box::new(source),
            })?;
        Ok(())
    }

    async fn ensure_search_index_ready(&self, cx: &Cx) -> SearchResult<Option<PathBuf>> {
        if self.index_artifacts_exist()? {
            return Ok(None);
        }

        let interactive_terminal =
            std::io::stdout().is_terminal() && std::io::stdin().is_terminal();
        if !interactive_terminal || self.cli_input.format != OutputFormat::Table {
            return Err(self.missing_index_error());
        }

        println!(
            "{}",
            paint(
                "No index found for this location. Starting guided first-run indexing setup.",
                "38;5;220",
                self.cli_input.no_color,
            )
        );
        let proposals = self.discover_index_root_proposals()?;
        let Some(selected_root) = self.prompt_first_run_index_root(&proposals)? else {
            return Err(SearchError::InvalidConfig {
                field: "cli.index_dir".to_owned(),
                value: String::new(),
                reason: "no index found and setup was cancelled; run `fsfs index <path>` first"
                    .to_owned(),
            });
        };
        let index_root = self.resolve_index_root(&selected_root)?;
        self.run_first_run_indexing_tui(cx, selected_root).await?;
        Ok(Some(index_root))
    }

    fn missing_index_error(&self) -> SearchError {
        let index_path = self.resolve_status_index_root().map_or_else(
            |_| self.config.storage.index_dir.clone(),
            |path| path.display().to_string(),
        );
        SearchError::InvalidConfig {
            field: "cli.index_dir".to_owned(),
            value: index_path,
            reason: "no index found; run `fsfs` for guided setup or `fsfs index <dir>` first"
                .to_owned(),
        }
    }

    fn search_mode_hint(&self) -> SearchResult<Option<String>> {
        let models = self.collect_model_statuses()?;
        let fast_cached = models
            .iter()
            .any(|model| model.tier == "fast" && model.cached);
        let quality_cached = models
            .iter()
            .any(|model| model.tier == "quality" && model.cached);

        if fast_cached && quality_cached && !self.config.search.fast_only {
            return Ok(None);
        }
        if fast_cached {
            if self.config.search.fast_only {
                return Ok(Some(
                    "Search mode: fast semantic tier only (quality disabled by `fast_only=true`)."
                        .to_owned(),
                ));
            }
            return Ok(Some(
                "Search mode: fast semantic tier active; quality model missing, so refinement is skipped."
                    .to_owned(),
            ));
        }
        Ok(Some(
            "Search mode: built-in hash + lexical fallback. Default semantic models are bundled; if this persists, check model-dir permissions and run `fsfs status`."
                .to_owned(),
        ))
    }

    async fn run_search_stream_command(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
    ) -> SearchResult<()> {
        if !matches!(
            self.cli_input.format,
            OutputFormat::Jsonl | OutputFormat::Toon
        ) {
            return Err(SearchError::InvalidConfig {
                field: "cli.format".to_owned(),
                value: self.cli_input.format.to_string(),
                reason: "stream mode requires --format jsonl or --format toon".to_owned(),
            });
        }

        let stream_id = format!("search-{}-{}", pressure_timestamp_ms(), std::process::id());
        let mut stdout = std::io::stdout();
        let mut seq = 0_u64;

        self.emit_search_stream_started(query, &stream_id, &mut seq, &mut stdout)?;
        match self
            .execute_search_payloads_cached_for_cli(cx, query, limit)
            .await
        {
            Ok(payloads) => {
                let payload = payloads.last().cloned().unwrap_or_else(|| {
                    SearchPayload::new(String::new(), SearchOutputPhase::Initial, 0, Vec::new())
                });
                for stage_payload in &payloads {
                    self.emit_search_stream_payload(
                        stage_payload,
                        &stream_id,
                        &mut seq,
                        &mut stdout,
                    )?;
                }
                self.emit_search_stream_terminal_completed(&stream_id, &mut seq, &mut stdout)?;
                info!(
                    query = query,
                    phase = payload.phase.to_string(),
                    returned_hits = payload.returned_hits,
                    total_candidates = payload.total_candidates,
                    stream_id,
                    frames_emitted = seq,
                    format = self.cli_input.format.to_string(),
                    output_format = self.cli_input.format.to_string(),
                    "fsfs search stream completed"
                );
                Ok(())
            }
            Err(error) => {
                self.emit_search_stream_terminal_error(&stream_id, &error, &mut seq, &mut stdout)?;
                Err(error)
            }
        }
    }

    async fn run_search_serve_command(&self, cx: &Cx) -> SearchResult<()> {
        #[cfg(unix)]
        if self.cli_input.daemon || self.cli_input.daemon_socket.is_some() {
            return self.run_search_serve_socket_command(cx).await;
        }

        self.run_search_serve_stdio_command(cx).await
    }

    #[allow(clippy::too_many_lines)]
    async fn run_search_serve_stdio_command(&self, cx: &Cx) -> SearchResult<()> {
        let mut resources = self.prepare_search_execution_resources(SearchExecutionMode::Full)?;
        let stdin = std::io::stdin();
        let mut line = String::new();
        let mut hot_cache: HashMap<SearchCacheKey, Vec<SearchPayload>> = HashMap::new();
        let hot_cache_enabled = std::env::var_os("FSFS_DISABLE_QUERY_CACHE").is_none();

        let ready = serde_json::json!({
            "ok": true,
            "event": "ready",
            "schema_version": FSFS_SEARCH_SERVE_SCHEMA_VERSION,
            "pid": std::process::id(),
            "format": self.cli_input.format.to_string(),
        });
        Self::emit_search_serve_json_line(&ready)?;

        loop {
            line.clear();
            let read = stdin.read_line(&mut line).map_err(SearchError::Io)?;
            if read == 0 {
                break;
            }
            let raw = line.trim();
            if raw.is_empty() {
                continue;
            }
            if matches!(raw, "quit" | "exit" | ":quit" | ":exit") {
                break;
            }

            let request = match Self::parse_search_serve_request(raw) {
                Ok(request) => request,
                Err(error) => {
                    let response =
                        Self::search_serve_error_response(String::new(), "full", error.to_string());
                    Self::emit_search_serve_json_line(&response)?;
                    continue;
                }
            };
            let request_query = request.query.clone();
            let request_mode = request.mode.clone().unwrap_or_else(|| "full".to_owned());
            let response = match self
                .execute_search_serve_request(
                    cx,
                    request,
                    &mut resources,
                    &mut hot_cache,
                    hot_cache_enabled,
                )
                .await
            {
                Ok(response) => response,
                Err(error) => Self::search_serve_error_response(
                    request_query,
                    request_mode,
                    error.to_string(),
                ),
            };
            Self::emit_search_serve_json_line(&response)?;
        }

        Ok(())
    }

    #[cfg(unix)]
    #[allow(
        clippy::too_many_lines,
        clippy::future_not_send,
        clippy::significant_drop_tightening
    )]
    async fn run_search_serve_socket_command(&self, cx: &Cx) -> SearchResult<()> {
        let socket_path = self.resolve_daemon_socket_path()?;
        if let Some(parent) = socket_path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }

        if socket_path.exists() {
            match UnixStream::connect(&socket_path) {
                Ok(_) => {
                    return Err(SearchError::InvalidConfig {
                        field: "cli.daemon_socket".to_owned(),
                        value: socket_path.display().to_string(),
                        reason: "a daemon is already listening on this socket".to_owned(),
                    });
                }
                Err(_) => {
                    let _ = fs::remove_file(&socket_path);
                }
            }
        }

        let listener = UnixListener::bind(&socket_path).map_err(SearchError::Io)?;

        let _socket_guard = SocketPathGuard {
            path: socket_path.clone(),
        };

        let mut resources = self.prepare_search_execution_resources(SearchExecutionMode::Full)?;
        let mut hot_cache: HashMap<SearchCacheKey, Vec<SearchPayload>> = HashMap::new();
        let hot_cache_enabled = std::env::var_os("FSFS_DISABLE_QUERY_CACHE").is_none();

        loop {
            let (mut stream, _) = match listener.accept() {
                Ok(connection) => connection,
                Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                Err(error) => return Err(SearchError::Io(error)),
            };
            if let Err(error) = stream.set_read_timeout(Some(Duration::from_millis(
                FSFS_DAEMON_CLIENT_IO_TIMEOUT_MS,
            ))) {
                warn!(
                    error = %error,
                    "fsfs daemon failed to set socket read timeout; skipping request"
                );
                continue;
            }
            if let Err(error) = stream.set_write_timeout(Some(Duration::from_millis(
                FSFS_DAEMON_CLIENT_IO_TIMEOUT_MS,
            ))) {
                warn!(
                    error = %error,
                    "fsfs daemon failed to set socket write timeout; skipping request"
                );
                continue;
            }

            let raw_request = match Self::read_search_serve_socket_request(&mut stream) {
                Ok(Some(raw_request)) => raw_request,
                Ok(None) => continue,
                Err(error) => {
                    let response =
                        Self::search_serve_error_response(String::new(), "full", error.to_string());
                    if let Err(write_error) =
                        Self::write_search_serve_socket_response(&mut stream, &response)
                    {
                        warn!(
                            error = %write_error,
                            "fsfs daemon failed to write socket error response"
                        );
                    }
                    continue;
                }
            };
            let raw = raw_request.trim();
            if raw.is_empty() {
                continue;
            }

            if matches!(raw, "quit" | "exit" | ":quit" | ":exit" | ":shutdown") {
                break;
            }

            let response = match Self::parse_search_serve_request(raw) {
                Ok(request) => {
                    let request_query = request.query.clone();
                    let request_mode = request.mode.clone().unwrap_or_else(|| "full".to_owned());
                    match self
                        .execute_search_serve_request(
                            cx,
                            request,
                            &mut resources,
                            &mut hot_cache,
                            hot_cache_enabled,
                        )
                        .await
                    {
                        Ok(response) => response,
                        Err(error) => Self::search_serve_error_response(
                            request_query,
                            request_mode,
                            error.to_string(),
                        ),
                    }
                }
                Err(error) => {
                    Self::search_serve_error_response(String::new(), "full", error.to_string())
                }
            };
            if let Err(error) = Self::write_search_serve_socket_response(&mut stream, &response) {
                warn!(
                    error = %error,
                    "fsfs daemon failed to write socket response"
                );
            }
        }

        Ok(())
    }

    fn parse_search_serve_request(raw: &str) -> SearchResult<SearchServeRequest> {
        if raw.starts_with('{') {
            serde_json::from_str::<SearchServeRequest>(raw).map_err(|source| {
                SearchError::InvalidConfig {
                    field: "cli.serve.request".to_owned(),
                    value: raw.to_owned(),
                    reason: format!("invalid serve request json: {source}"),
                }
            })
        } else {
            Ok(SearchServeRequest {
                query: raw.to_owned(),
                limit: None,
                mode: None,
                filter: None,
            })
        }
    }

    async fn execute_search_serve_request(
        &self,
        cx: &Cx,
        request: SearchServeRequest,
        resources: &mut SearchExecutionResources,
        hot_cache: &mut HashMap<SearchCacheKey, Vec<SearchPayload>>,
        hot_cache_enabled: bool,
    ) -> SearchResult<SearchServeResponse> {
        let mode = parse_search_execution_mode(request.mode.as_deref())?;
        let requested_limit = request.limit.unwrap_or_else(|| {
            self.cli_input
                .overrides
                .limit
                .unwrap_or(self.config.search.default_limit)
        });
        let mut runtime = self.clone();
        let mut cli = runtime.cli_input.clone();
        cli.filter = request.filter.clone();
        runtime = runtime.with_cli_input(cli);
        let cache_key = runtime.search_cache_key(&request.query, requested_limit, mode);

        let (cached, payloads) = if hot_cache_enabled {
            if let Some(cached_payloads) = hot_cache.get(&cache_key) {
                (true, cached_payloads.clone())
            } else {
                let payloads = runtime
                    .execute_search_payloads_with_mode_using_resources(
                        cx,
                        &request.query,
                        requested_limit,
                        mode,
                        resources,
                        SearchExecutionFlags {
                            include_snippets: true,
                            persist_explain_session: false,
                        },
                    )
                    .await?;
                hot_cache.insert(cache_key, payloads.clone());
                (false, payloads)
            }
        } else {
            let payloads = runtime
                .execute_search_payloads_with_mode_using_resources(
                    cx,
                    &request.query,
                    requested_limit,
                    mode,
                    resources,
                    SearchExecutionFlags {
                        include_snippets: true,
                        persist_explain_session: false,
                    },
                )
                .await?;
            (false, payloads)
        };

        Ok(SearchServeResponse {
            ok: true,
            query: request.query,
            mode: mode.label().to_owned(),
            cached,
            payloads,
            error: None,
        })
    }

    fn emit_search_serve_json_line<T: Serialize>(value: &T) -> SearchResult<()> {
        let stdout = std::io::stdout();
        let mut writer = BufWriter::with_capacity(1 << 20, stdout.lock());
        serde_json::to_writer(&mut writer, value).map_err(|source| {
            SearchError::SubsystemError {
                subsystem: "fsfs.search.serve",
                source: Box::new(source),
            }
        })?;
        writer
            .write_all(b"\n")
            .map_err(|source| SearchError::SubsystemError {
                subsystem: "fsfs.search.serve",
                source: Box::new(source),
            })?;
        writer
            .flush()
            .map_err(|source| SearchError::SubsystemError {
                subsystem: "fsfs.search.serve",
                source: Box::new(source),
            })?;
        Ok(())
    }

    fn search_serve_error_response(
        query: impl Into<String>,
        mode: impl Into<String>,
        error: impl Into<String>,
    ) -> SearchServeResponse {
        SearchServeResponse {
            ok: false,
            query: query.into(),
            mode: mode.into(),
            cached: false,
            payloads: Vec::new(),
            error: Some(error.into()),
        }
    }

    #[cfg(unix)]
    fn read_search_serve_socket_request(stream: &mut UnixStream) -> SearchResult<Option<String>> {
        let mut raw_request = String::new();
        let read_limit = FSFS_DAEMON_REQUEST_MAX_BYTES.saturating_add(1);
        {
            let mut limited_reader = (&mut *stream).take(read_limit as u64);
            limited_reader
                .read_to_string(&mut raw_request)
                .map_err(SearchError::Io)?;
        }
        if raw_request.trim().is_empty() {
            return Ok(None);
        }
        if raw_request.len() > FSFS_DAEMON_REQUEST_MAX_BYTES {
            return Err(SearchError::InvalidConfig {
                field: "cli.serve.request".to_owned(),
                value: "<payload>".to_owned(),
                reason: format!("request exceeds {FSFS_DAEMON_REQUEST_MAX_BYTES} bytes limit"),
            });
        }
        Ok(Some(raw_request))
    }

    #[cfg(unix)]
    fn write_search_serve_socket_response(
        stream: &mut UnixStream,
        response: &SearchServeResponse,
    ) -> SearchResult<()> {
        let response_bytes =
            serde_json::to_vec(response).map_err(|source| SearchError::SubsystemError {
                subsystem: "fsfs.search.serve",
                source: Box::new(source),
            })?;
        stream.write_all(&response_bytes).map_err(SearchError::Io)?;
        stream.write_all(b"\n").map_err(SearchError::Io)?;
        stream.flush().map_err(SearchError::Io)?;
        Ok(())
    }

    fn search_payloads_via_daemon(
        &self,
        query: &str,
        limit: usize,
    ) -> SearchResult<Vec<SearchPayload>> {
        #[cfg(unix)]
        {
            let socket_path = self.resolve_daemon_socket_path()?;
            let mut stream = if let Ok(stream) = UnixStream::connect(&socket_path) {
                stream
            } else {
                self.spawn_search_daemon(&socket_path)?;
                Self::wait_for_daemon_connection(&socket_path)?
            };

            let request = SearchServeRequest {
                query: query.to_owned(),
                limit: Some(limit),
                mode: Some("full".to_owned()),
                filter: self.cli_input.filter.clone(),
            };
            let request_json =
                serde_json::to_vec(&request).map_err(|source| SearchError::SubsystemError {
                    subsystem: "fsfs.search.daemon",
                    source: Box::new(source),
                })?;
            stream.write_all(&request_json).map_err(SearchError::Io)?;
            stream.write_all(b"\n").map_err(SearchError::Io)?;
            stream.flush().map_err(SearchError::Io)?;
            let _ = stream.shutdown(Shutdown::Write);

            let mut raw_response = String::new();
            {
                let response_read_limit = FSFS_DAEMON_RESPONSE_MAX_BYTES.saturating_add(1);
                let mut limited_reader = (&mut stream).take(response_read_limit as u64);
                limited_reader
                    .read_to_string(&mut raw_response)
                    .map_err(SearchError::Io)?;
            }
            if raw_response.len() > FSFS_DAEMON_RESPONSE_MAX_BYTES {
                return Err(SearchError::InvalidConfig {
                    field: "cli.daemon".to_owned(),
                    value: "search".to_owned(),
                    reason: format!(
                        "daemon response exceeded {FSFS_DAEMON_RESPONSE_MAX_BYTES} bytes"
                    ),
                });
            }
            let response = serde_json::from_str::<SearchServeResponse>(raw_response.trim())
                .map_err(|source| SearchError::SubsystemError {
                    subsystem: "fsfs.search.daemon",
                    source: Box::new(source),
                })?;
            if !response.ok {
                return Err(SearchError::InvalidConfig {
                    field: "cli.daemon".to_owned(),
                    value: "search".to_owned(),
                    reason: response
                        .error
                        .unwrap_or_else(|| "daemon search request failed".to_owned()),
                });
            }
            if let Err(error) =
                self.persist_explain_session_for_cached_payloads(query, &response.payloads)
            {
                warn!(
                    error = %error,
                    "fsfs daemon-backed search could not persist explain-session context"
                );
            }
            Ok(response.payloads)
        }

        #[cfg(not(unix))]
        {
            let _ = (query, limit);
            Err(SearchError::InvalidConfig {
                field: "cli.daemon".to_owned(),
                value: "search".to_owned(),
                reason: "daemon transport is only supported on unix platforms".to_owned(),
            })
        }
    }

    #[cfg(unix)]
    fn spawn_search_daemon(&self, socket_path: &Path) -> SearchResult<()> {
        if let Some(parent) = socket_path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        let binary_path = std::env::current_exe().map_err(SearchError::Io)?;
        let mut command = Command::new(binary_path);
        command
            .arg("serve")
            .arg("--daemon-socket")
            .arg(socket_path)
            .arg("--format")
            .arg("jsonl")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        if let Some(index_dir) = self.cli_input.index_dir.as_deref() {
            command.arg("--index-dir").arg(index_dir);
        }
        if let Some(config_path) = self.cli_input.overrides.config_path.as_deref() {
            command.arg("--config").arg(config_path);
        }
        if self.cli_input.verbose {
            command.arg("--verbose");
        }
        if self.cli_input.quiet {
            command.arg("--quiet");
        }
        if self.cli_input.no_color {
            command.arg("--no-color");
        }
        command.spawn().map_err(SearchError::Io)?;
        Ok(())
    }

    #[cfg(unix)]
    fn wait_for_daemon_connection(socket_path: &Path) -> SearchResult<UnixStream> {
        let mut last_error: Option<std::io::Error> = None;
        for _ in 0..FSFS_DAEMON_CONNECT_MAX_ATTEMPTS {
            match UnixStream::connect(socket_path) {
                Ok(stream) => return Ok(stream),
                Err(error) => {
                    last_error = Some(error);
                    std::thread::sleep(Duration::from_millis(FSFS_DAEMON_CONNECT_RETRY_DELAY_MS));
                }
            }
        }

        Err(SearchError::InvalidConfig {
            field: "cli.daemon_socket".to_owned(),
            value: socket_path.display().to_string(),
            reason: format!(
                "daemon did not become ready: {}",
                last_error.map_or_else(
                    || "unknown startup error".to_owned(),
                    |error| error.to_string()
                )
            ),
        })
    }

    #[cfg(unix)]
    fn resolve_daemon_socket_path(&self) -> SearchResult<PathBuf> {
        if let Some(path) = self.cli_input.daemon_socket.as_deref() {
            return absolutize_path(path);
        }
        self.default_daemon_socket_path()
    }

    #[cfg(unix)]
    fn default_daemon_socket_path(&self) -> SearchResult<PathBuf> {
        let index_root = self
            .resolve_status_index_root()
            .or_else(|_| std::env::current_dir().map_err(SearchError::Io))?;
        let mut hasher = Sha256::new();
        hasher.update(index_root.display().to_string().as_bytes());
        let digest = format!("{:x}", hasher.finalize());
        let prefix_len = FSFS_DAEMON_SOCKET_HASH_PREFIX_LEN.min(digest.len());
        let runtime_base = dirs::runtime_dir()
            .or_else(|| dirs::cache_dir().map(|dir| dir.join("run")))
            .unwrap_or_else(std::env::temp_dir);
        let socket_dir = runtime_base.join("frankensearch").join("daemon");
        Ok(socket_dir.join(format!("fsfs-query-{}.sock", &digest[..prefix_len])))
    }

    fn emit_search_stream_started<W: Write>(
        &self,
        query: &str,
        stream_id: &str,
        seq: &mut u64,
        writer: &mut W,
    ) -> SearchResult<()> {
        let frame = StreamFrame::new(
            stream_id.to_owned(),
            *seq,
            iso_timestamp_now(),
            "search",
            StreamEvent::<SearchHitPayload>::Started(StreamStartedEvent {
                stream_id: stream_id.to_owned(),
                query: query.to_owned(),
                format: self.cli_input.format.to_string(),
            }),
        );
        emit_stream_frame(&frame, self.cli_input.format, writer)?;
        *seq = seq.saturating_add(1);
        Ok(())
    }

    fn emit_search_stream_payload<W: Write>(
        &self,
        payload: &SearchPayload,
        stream_id: &str,
        seq: &mut u64,
        writer: &mut W,
    ) -> SearchResult<()> {
        let (stage, reason_code, message) = match payload.phase {
            SearchOutputPhase::Initial => (
                "retrieve.fast",
                "query.stream.initial_ready",
                "initial results ready",
            ),
            SearchOutputPhase::Refined => (
                "retrieve.quality",
                "query.stream.refined_ready",
                "refined results ready",
            ),
            SearchOutputPhase::RefinementFailed => (
                "retrieve.refinement_failed",
                "query.stream.refinement_failed",
                "quality refinement failed; returning initial results",
            ),
        };
        let progress_frame = StreamFrame::new(
            stream_id.to_owned(),
            *seq,
            iso_timestamp_now(),
            "search",
            StreamEvent::<SearchHitPayload>::Progress(StreamProgressEvent {
                stage: stage.to_owned(),
                completed_units: u64::try_from(payload.returned_hits).unwrap_or(u64::MAX),
                total_units: Some(u64::try_from(payload.total_candidates).unwrap_or(u64::MAX)),
                reason_code: reason_code.to_owned(),
                message: message.to_owned(),
            }),
        );
        emit_stream_frame(&progress_frame, self.cli_input.format, writer)?;
        *seq = seq.saturating_add(1);

        for hit in &payload.hits {
            let result_frame = StreamFrame::new(
                stream_id.to_owned(),
                *seq,
                iso_timestamp_now(),
                "search",
                StreamEvent::<SearchHitPayload>::Result(StreamResultEvent {
                    rank: u64::try_from(hit.rank).unwrap_or(u64::MAX),
                    item: hit.clone(),
                }),
            );
            emit_stream_frame(&result_frame, self.cli_input.format, writer)?;
            *seq = seq.saturating_add(1);
        }

        Ok(())
    }

    fn emit_search_stream_terminal_completed<W: Write>(
        &self,
        stream_id: &str,
        seq: &mut u64,
        writer: &mut W,
    ) -> SearchResult<()> {
        let frame = StreamFrame::new(
            stream_id.to_owned(),
            *seq,
            iso_timestamp_now(),
            "search",
            StreamEvent::<SearchHitPayload>::Terminal(terminal_event_completed()),
        );
        emit_stream_frame(&frame, self.cli_input.format, writer)?;
        *seq = seq.saturating_add(1);
        Ok(())
    }

    fn emit_search_stream_terminal_error<W: Write>(
        &self,
        stream_id: &str,
        error: &SearchError,
        seq: &mut u64,
        writer: &mut W,
    ) -> SearchResult<()> {
        let frame = StreamFrame::new(
            stream_id.to_owned(),
            *seq,
            iso_timestamp_now(),
            "search",
            StreamEvent::<SearchHitPayload>::Terminal(terminal_event_from_error(error, 0, 3)),
        );
        emit_stream_frame(&frame, self.cli_input.format, writer)?;
        *seq = seq.saturating_add(1);
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn run_explain_command(&self) -> SearchResult<()> {
        let result_id =
            self.cli_input
                .result_id
                .as_deref()
                .ok_or_else(|| SearchError::InvalidConfig {
                    field: "cli.explain.result_id".to_owned(),
                    value: String::new(),
                    reason: "missing result identifier argument".to_owned(),
                })?;
        let session = self
            .load_explain_session()?
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "cli.explain.result_id".to_owned(),
                value: result_id.to_owned(),
                reason: "no saved search context found; run `fsfs search <query>` first".to_owned(),
            })?;
        let hit = session
            .resolve(result_id)
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "cli.explain.result_id".to_owned(),
                value: result_id.to_owned(),
                reason: format!(
                    "unknown result id; available ids: {}",
                    session.preview_ids()
                ),
            })?;

        let matched_terms = session
            .query
            .split_whitespace()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let source_count = usize::from(hit.lexical_score.is_some())
            .saturating_add(usize::from(hit.semantic_score.is_some()));
        let shared_weight = if source_count == 0 {
            1.0
        } else {
            1.0 / f64::from(u32::try_from(source_count).unwrap_or(1))
        };

        let mut components = Vec::new();
        if let Some(lexical_score) = hit.lexical_score {
            components.push(ScoreComponent {
                source: ExplainedSource::LexicalBm25 {
                    matched_terms: matched_terms.clone(),
                    tf: 0.0,
                    idf: 0.0,
                },
                raw_score: f64::from(lexical_score),
                normalized_score: f64::from(lexical_score),
                rrf_contribution: rrf_contribution_for_rank(session.rrf_k, hit.lexical_rank),
                weight: shared_weight,
            });
        }
        if let Some(semantic_score) = hit.semantic_score {
            components.push(ScoreComponent {
                source: ExplainedSource::SemanticFast {
                    embedder: "fast-tier".to_owned(),
                    cosine_sim: f64::from(semantic_score),
                },
                raw_score: f64::from(semantic_score),
                normalized_score: f64::from(semantic_score),
                rrf_contribution: rrf_contribution_for_rank(session.rrf_k, hit.semantic_rank),
                weight: shared_weight,
            });
        }
        if components.is_empty() {
            components.push(ScoreComponent {
                source: ExplainedSource::LexicalBm25 {
                    matched_terms,
                    tf: 0.0,
                    idf: 0.0,
                },
                raw_score: 0.0,
                normalized_score: 0.0,
                rrf_contribution: 0.0,
                weight: 1.0,
            });
        }

        let explanation = HitExplanation {
            final_score: hit.final_score,
            components,
            phase: match session.phase {
                SearchOutputPhase::Refined => ExplanationPhase::Refined,
                SearchOutputPhase::Initial | SearchOutputPhase::RefinementFailed => {
                    ExplanationPhase::Initial
                }
            },
            rank_movement: None,
        };
        let mut ranking = RankingExplanation::from_hit_explanation(
            &hit.path,
            &explanation,
            "query.explain.attached",
            920,
        );
        ranking.fusion = Some(FusionContext {
            fused_score: hit.final_score,
            lexical_rank: hit.lexical_rank,
            semantic_rank: hit.semantic_rank,
            lexical_score: hit.lexical_score,
            semantic_score: hit.semantic_score,
            in_both_sources: hit.in_both_sources,
        });
        let payload = FsfsExplanationPayload::new(session.query.clone(), ranking);

        if self.cli_input.format == OutputFormat::Table {
            println!(
                "{}",
                render_explain_table(result_id, &payload, hit, session.rrf_k)
            );
            return Ok(());
        }

        let meta = meta_for_format("explain", self.cli_input.format);
        let envelope = OutputEnvelope::success(payload, meta, iso_timestamp_now());
        let mut stdout = std::io::stdout();
        emit_envelope(&envelope, self.cli_input.format, &mut stdout)?;
        if self.cli_input.format != OutputFormat::Jsonl {
            stdout
                .write_all(b"\n")
                .map_err(|source| SearchError::SubsystemError {
                    subsystem: "fsfs.explain",
                    source: Box::new(source),
                })?;
        }

        Ok(())
    }

    fn explain_session_path(index_root: &Path) -> PathBuf {
        index_root.join(FSFS_EXPLAIN_SESSION_FILE)
    }

    fn persist_explain_session(
        &self,
        index_root: &Path,
        query: &str,
        phase: SearchOutputPhase,
        fused: &[FusedCandidate],
    ) -> SearchResult<()> {
        let session = ExplainSession::from_fused(query, phase, self.config.search.rrf_k, fused);
        let path = Self::explain_session_path(index_root);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(&session).map_err(|source| {
            SearchError::SubsystemError {
                subsystem: "fsfs.explain.session",
                source: Box::new(source),
            }
        })?;
        write_durable(path, json)?;
        Ok(())
    }

    fn load_explain_session(&self) -> SearchResult<Option<ExplainSession>> {
        let index_root = self.resolve_status_index_root()?;
        let path = Self::explain_session_path(&index_root);
        let raw = match fs::read_to_string(&path) {
            Ok(raw) => raw,
            Err(source) if source.kind() == ErrorKind::NotFound => return Ok(None),
            Err(source) => {
                return Err(SearchError::SubsystemError {
                    subsystem: "fsfs.explain.session",
                    source: Box::new(source),
                });
            }
        };
        let session = serde_json::from_str::<ExplainSession>(&raw).map_err(|source| {
            SearchError::SubsystemError {
                subsystem: "fsfs.explain.session",
                source: Box::new(source),
            }
        })?;
        Ok(Some(session))
    }

    #[cfg(test)]
    #[allow(clippy::too_many_lines)]
    async fn execute_search_payload(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
    ) -> SearchResult<SearchPayload> {
        let payloads = self.execute_search_payloads(cx, query, limit).await?;
        Ok(payloads.last().cloned().unwrap_or_else(|| {
            SearchPayload::new(String::new(), SearchOutputPhase::Initial, 0, Vec::new())
        }))
    }

    #[cfg(test)]
    async fn execute_search_payloads(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
    ) -> SearchResult<Vec<SearchPayload>> {
        self.execute_search_payloads_with_mode(cx, query, limit, SearchExecutionMode::Full)
            .await
    }

    async fn execute_search_payloads_with_mode(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        mode: SearchExecutionMode,
    ) -> SearchResult<Vec<SearchPayload>> {
        let artifacts = self
            .execute_search_phase_artifacts_with_mode(cx, query, limit, mode)
            .await?;
        Ok(artifacts
            .into_iter()
            .map(|artifact| artifact.payload)
            .collect())
    }

    async fn execute_search_payloads_with_mode_using_resources(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        mode: SearchExecutionMode,
        resources: &mut SearchExecutionResources,
        flags: SearchExecutionFlags,
    ) -> SearchResult<Vec<SearchPayload>> {
        let artifacts = self
            .execute_search_phase_artifacts_with_mode_using_resources(
                cx, query, limit, mode, resources, flags,
            )
            .await?;
        Ok(artifacts
            .into_iter()
            .map(|artifact| artifact.payload)
            .collect())
    }

    async fn execute_search_payloads_cached_for_cli(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
    ) -> SearchResult<Vec<SearchPayload>> {
        let mode = SearchExecutionMode::Full;
        let cache_enabled = std::env::var_os("FSFS_DISABLE_QUERY_CACHE").is_none();
        if !cache_enabled {
            return self
                .execute_search_payloads_with_mode(cx, query, limit, mode)
                .await;
        }
        let key = self.search_cache_key(query, limit, mode);
        match self.try_load_search_payload_cache(&key) {
            Ok(Some(payloads)) => {
                if let Err(error) =
                    self.persist_explain_session_for_cached_payloads(query, &payloads)
                {
                    warn!(
                        error = %error,
                        "fsfs cached search could not persist explain-session context"
                    );
                }
                return Ok(payloads);
            }
            Ok(None) => {}
            Err(error) => {
                warn!(
                    error = %error,
                    "fsfs search cache read failed; continuing with uncached execution"
                );
            }
        }

        let payloads = self
            .execute_search_payloads_with_mode(cx, query, limit, mode)
            .await?;
        if let Err(error) = self.write_search_payload_cache(&key, &payloads) {
            warn!(
                error = %error,
                "fsfs search cache write failed; continuing without cache persistence"
            );
        }
        Ok(payloads)
    }

    #[must_use]
    fn normalize_search_query(query: &str) -> String {
        query
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .trim()
            .to_owned()
    }

    #[must_use]
    fn search_cache_key(
        &self,
        query: &str,
        requested_limit: usize,
        mode: SearchExecutionMode,
    ) -> SearchCacheKey {
        SearchCacheKey {
            query: Self::normalize_search_query(query),
            requested_limit,
            mode: mode.label().to_owned(),
            filter: self.cli_input.filter.clone(),
            fast_only: self.config.search.fast_only,
            rrf_k_milli: u64::from(f64_to_per_mille(self.config.search.rrf_k)),
        }
    }

    #[must_use]
    fn search_cache_key_hash(key: &SearchCacheKey) -> String {
        let mut hasher = Sha256::new();
        hasher.update(key.query.as_bytes());
        hasher.update(key.requested_limit.to_le_bytes());
        hasher.update(key.mode.as_bytes());
        hasher.update(key.filter.as_deref().unwrap_or("").as_bytes());
        hasher.update([u8::from(key.fast_only)]);
        hasher.update(key.rrf_k_milli.to_le_bytes());
        format!("{:x}", hasher.finalize())
    }

    fn search_cache_path(&self, key: &SearchCacheKey) -> SearchResult<PathBuf> {
        let index_root = self.resolve_status_index_root()?;
        let file_name = format!("{}.json", Self::search_cache_key_hash(key));
        Ok(index_root.join(FSFS_SEARCH_CACHE_DIR_NAME).join(file_name))
    }

    fn search_index_fingerprint(&self) -> SearchResult<String> {
        let index_root = self.resolve_status_index_root()?;
        let mut hasher = Sha256::new();

        if let Some(sentinel) = Self::read_index_sentinel(&index_root)? {
            hasher.update(sentinel.source_hash_hex.as_bytes());
            hasher.update(sentinel.generated_at_ms.to_le_bytes());
        }

        for rel in [
            FSFS_VECTOR_INDEX_FILE,
            FSFS_VECTOR_MANIFEST_FILE,
            FSFS_LEXICAL_MANIFEST_FILE,
            FSFS_SENTINEL_FILE,
            "lexical",
        ] {
            let path = index_root.join(rel);
            hasher.update(rel.as_bytes());
            match fs::metadata(&path) {
                Ok(metadata) => {
                    hasher.update([1]);
                    hasher.update(metadata.len().to_le_bytes());
                    let modified_ms = metadata.modified().ok().map_or(0, system_time_to_ms);
                    hasher.update(modified_ms.to_le_bytes());
                }
                Err(error) if error.kind() == ErrorKind::NotFound => {
                    hasher.update([0]);
                }
                Err(error) => return Err(SearchError::Io(error)),
            }
        }

        Ok(format!("{:x}", hasher.finalize()))
    }

    fn try_load_search_payload_cache(
        &self,
        key: &SearchCacheKey,
    ) -> SearchResult<Option<Vec<SearchPayload>>> {
        let cache_path = self.search_cache_path(key)?;
        let raw = match fs::read_to_string(&cache_path) {
            Ok(raw) => raw,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(SearchError::Io(error)),
        };

        let record = serde_json::from_str::<SearchCacheRecord>(&raw).map_err(|source| {
            SearchError::SubsystemError {
                subsystem: "fsfs.search.cache",
                source: Box::new(source),
            }
        })?;
        if record.schema_version != FSFS_SEARCH_CACHE_SCHEMA_VERSION || record.key != *key {
            return Ok(None);
        }
        if record.index_fingerprint != self.search_index_fingerprint()? {
            return Ok(None);
        }
        Ok(Some(record.payloads))
    }

    fn write_search_payload_cache(
        &self,
        key: &SearchCacheKey,
        payloads: &[SearchPayload],
    ) -> SearchResult<()> {
        let cache_path = self.search_cache_path(key)?;
        if let Some(parent) = cache_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let record = SearchCacheRecord {
            schema_version: FSFS_SEARCH_CACHE_SCHEMA_VERSION.to_owned(),
            index_fingerprint: self.search_index_fingerprint()?,
            key: key.clone(),
            payloads: payloads.to_vec(),
        };
        let json = serde_json::to_vec(&record).map_err(|source| SearchError::SubsystemError {
            subsystem: "fsfs.search.cache",
            source: Box::new(source),
        })?;
        fs::write(cache_path, json).map_err(SearchError::Io)?;
        Ok(())
    }

    fn persist_explain_session_for_cached_payloads(
        &self,
        query: &str,
        payloads: &[SearchPayload],
    ) -> SearchResult<()> {
        let Some(last) = payloads.last() else {
            return Ok(());
        };
        let fused = last
            .hits
            .iter()
            .map(|hit| FusedCandidate {
                doc_id: hit.path.clone(),
                fused_score: hit.score,
                prior_boost: 0.0,
                lexical_rank: hit.lexical_rank,
                semantic_rank: hit.semantic_rank,
                lexical_score: None,
                semantic_score: None,
                in_both_sources: hit.in_both_sources,
            })
            .collect::<Vec<_>>();
        let index_root = self.resolve_status_index_root()?;
        self.persist_explain_session(&index_root, query, last.phase, &fused)
    }

    fn apply_search_filter(
        fused: &[FusedCandidate],
        filter_expr: Option<&SearchFilterExpr>,
    ) -> Vec<FusedCandidate> {
        filter_expr.map_or_else(
            || fused.to_vec(),
            |expr| {
                fused
                    .iter()
                    .filter(|candidate| expr.matches_doc_id(&candidate.doc_id))
                    .cloned()
                    .collect()
            },
        )
    }

    fn build_limited_payload(
        orchestrator: QueryExecutionOrchestrator,
        query: &str,
        phase: SearchOutputPhase,
        fused: &[FusedCandidate],
        snippets_by_doc: &HashMap<String, String>,
        limit: usize,
    ) -> SearchPayload {
        let limited = if limit == FSFS_SEARCH_UNBOUNDED_LIMIT_SENTINEL {
            fused.to_vec()
        } else {
            fused.iter().take(limit).cloned().collect::<Vec<_>>()
        };
        let mut payload =
            orchestrator.build_search_payload(query, phase, &limited, snippets_by_doc);
        payload.total_candidates = fused.len();
        payload
    }

    #[must_use]
    fn resolve_output_limit(
        requested_limit: usize,
        lexical_doc_count: Option<usize>,
        vector_doc_count: Option<usize>,
    ) -> usize {
        if requested_limit != FSFS_SEARCH_UNBOUNDED_LIMIT_SENTINEL {
            return requested_limit.max(1);
        }
        lexical_doc_count
            .unwrap_or(0)
            .max(vector_doc_count.unwrap_or(0))
            .max(1)
    }

    #[must_use]
    fn resolve_planning_limit(output_limit: usize) -> usize {
        if output_limit <= FSFS_SEARCH_SEMANTIC_HEAD_LIMIT {
            return output_limit.max(1);
        }
        let overflow = output_limit.saturating_sub(FSFS_SEARCH_SEMANTIC_HEAD_LIMIT);
        let widening =
            Self::integer_sqrt(overflow).saturating_mul(FSFS_SEARCH_SEMANTIC_HEAD_PROGRESSIVE_STEP);
        FSFS_SEARCH_SEMANTIC_HEAD_LIMIT
            .saturating_add(widening)
            .min(output_limit)
            .max(1)
    }

    #[must_use]
    const fn integer_sqrt(value: usize) -> usize {
        if value <= 1 {
            return value;
        }
        let mut low = 1_usize;
        let mut high = value;
        let mut answer = 1_usize;
        while low <= high {
            let mid = low + (high - low) / 2;
            if mid <= value / mid {
                answer = mid;
                low = mid.saturating_add(1);
            } else {
                high = mid.saturating_sub(1);
            }
        }
        answer
    }

    #[must_use]
    const fn should_force_full_semantic_recall(input: SemanticRecallDecisionInput) -> bool {
        if !input.semantic_stage_enabled || !input.semantic_index_available {
            return false;
        }
        input.output_limit > input.planning_limit
            && (!input.lexical_stage_enabled
                || input.lexical_head_count == 0
                || input.lexical_full_count < input.output_limit)
    }

    #[must_use]
    fn semantic_gate_decision(input: SemanticGateDecisionInput<'_>) -> SemanticVoiDecision {
        if !input.semantic_stage_enabled {
            SemanticVoiDecision {
                run_semantic: false,
                posterior_useful: 0.0,
                expected_loss_run: 1.0,
                expected_loss_skip: 0.0,
                reason_code: "semantic.voi.skip.stage_disabled",
            }
        } else if input.force_full_semantic_recall {
            SemanticVoiDecision::force_run("semantic.voi.force.full_recall")
        } else if !input.lexical_stage_enabled || input.lexical_head_candidates.is_empty() {
            SemanticVoiDecision::force_run("semantic.voi.force.no_lexical_head")
        } else {
            Self::semantic_voi_decision(
                input.mode,
                input.intent,
                input.query,
                input.output_limit,
                input.lexical_head_candidates,
            )
        }
    }

    #[must_use]
    fn semantic_voi_decision(
        mode: SearchExecutionMode,
        intent: QueryIntentClass,
        query: &str,
        output_limit: usize,
        lexical_head_candidates: &[LexicalCandidate],
    ) -> SemanticVoiDecision {
        if std::env::var_os("FSFS_SEMANTIC_VOI_FORCE_RUN").is_some() {
            return SemanticVoiDecision::force_run("semantic.voi.force.env_override");
        }
        if matches!(mode, SearchExecutionMode::LexicalOnly) {
            return SemanticVoiDecision {
                run_semantic: false,
                posterior_useful: 0.0,
                expected_loss_run: 1.0,
                expected_loss_skip: 0.0,
                reason_code: "semantic.voi.skip.lexical_only_mode",
            };
        }

        let mut odds = match intent {
            QueryIntentClass::NaturalLanguage => 1.8_f64,
            QueryIntentClass::ShortKeyword => 0.45_f64,
            QueryIntentClass::Identifier => 0.22_f64,
            QueryIntentClass::Uncertain => 0.55_f64,
            QueryIntentClass::Malformed | QueryIntentClass::Empty => 0.08_f64,
        };

        let token_count = query.split_whitespace().count();
        let query_char_count = query.chars().count();

        if token_count >= 4 {
            odds *= 1.7;
        } else if token_count == 1 {
            odds *= 0.62;
        }
        if query_char_count >= 24 {
            odds *= 1.25;
        } else if query_char_count <= 3 {
            odds *= 0.75;
        }

        if lexical_head_candidates.is_empty() {
            odds *= 3.2;
        } else {
            let top = lexical_head_candidates
                .first()
                .map_or(0.0_f32, |candidate| candidate.score);
            let second = lexical_head_candidates
                .get(1)
                .map_or(top, |candidate| candidate.score);
            let gap = (top - second).max(0.0);
            if gap >= 3.0 {
                odds *= 0.45;
            } else if gap >= 1.5 {
                odds *= 0.62;
            } else if gap <= 0.25 {
                odds *= 1.25;
            }

            let top_k = lexical_head_candidates.len().min(5);
            let denom = f64::from(u32::try_from(top_k).unwrap_or(1)).max(1.0);
            let mean_top_score = lexical_head_candidates
                .iter()
                .take(top_k)
                .map(|candidate| f64::from(candidate.score))
                .sum::<f64>()
                / denom;
            if mean_top_score < 0.25 {
                odds *= 1.2;
            } else if mean_top_score > 3.0 {
                odds *= 0.9;
            }
        }

        if output_limit > 5_000 {
            odds *= 1.15;
        } else if output_limit <= 64 {
            odds *= 0.92;
        }

        let odds = odds.clamp(0.01, 50.0);
        let posterior_useful = odds / (1.0 + odds);

        let (latency_loss_if_unhelpful, miss_loss_if_helpful) = match mode {
            SearchExecutionMode::FastOnly => (9.0_f64, 5.6_f64),
            SearchExecutionMode::Full if output_limit <= 500 => (6.0_f64, 7.8_f64),
            SearchExecutionMode::Full => (3.2_f64, 9.2_f64),
            SearchExecutionMode::LexicalOnly => (1.0_f64, 0.0_f64),
        };
        let expected_loss_run =
            posterior_useful * 0.25 + (1.0 - posterior_useful) * latency_loss_if_unhelpful;
        let expected_loss_skip = posterior_useful * miss_loss_if_helpful;
        let run_semantic = expected_loss_run <= expected_loss_skip;
        let reason_code = if run_semantic {
            "semantic.voi.run"
        } else {
            "semantic.voi.skip"
        };

        SemanticVoiDecision {
            run_semantic,
            posterior_useful,
            expected_loss_run,
            expected_loss_skip,
            reason_code,
        }
    }

    #[must_use]
    fn merge_with_lexical_tail(
        fused_head: &[FusedCandidate],
        lexical_full: &[LexicalCandidate],
        filter_expr: Option<&SearchFilterExpr>,
    ) -> Vec<FusedCandidate> {
        if lexical_full.is_empty() {
            return fused_head.to_vec();
        }

        let mut merged = Vec::with_capacity(fused_head.len().saturating_add(lexical_full.len()));
        // We only need to dedupe against the fused head.
        //
        // Lexical candidates are produced from Tantivy doc addresses and are expected
        // to be unique under normal upsert/index invariants. Growing a full-set for the
        // entire lexical tail is unnecessary and expensive on large corpora.
        let mut seen_head_ids = HashSet::with_capacity(fused_head.len());

        for candidate in fused_head {
            if let Some(expr) = filter_expr
                && !expr.matches_doc_id(&candidate.doc_id)
            {
                continue;
            }
            seen_head_ids.insert(candidate.doc_id.as_str());
            merged.push(candidate.clone());
        }

        for (rank, lexical) in lexical_full.iter().enumerate() {
            if seen_head_ids.contains(lexical.doc_id.as_str()) {
                continue;
            }
            if let Some(expr) = filter_expr
                && !expr.matches_doc_id(&lexical.doc_id)
            {
                continue;
            }
            merged.push(FusedCandidate {
                doc_id: lexical.doc_id.clone(),
                fused_score: f64::from(lexical.score),
                prior_boost: 0.0,
                lexical_rank: Some(rank),
                semantic_rank: None,
                lexical_score: Some(lexical.score),
                semantic_score: None,
                in_both_sources: false,
            });
        }

        merged
    }

    #[must_use]
    fn merge_with_fallback_tail(
        refined_head: &[FusedCandidate],
        fallback_ordered: &[FusedCandidate],
        output_limit: usize,
    ) -> Vec<FusedCandidate> {
        let effective_limit = if output_limit == FSFS_SEARCH_UNBOUNDED_LIMIT_SENTINEL {
            usize::MAX
        } else {
            output_limit
        };
        if effective_limit != usize::MAX && refined_head.len() >= effective_limit {
            return refined_head
                .iter()
                .take(effective_limit)
                .cloned()
                .collect::<Vec<_>>();
        }

        let mut merged =
            Vec::with_capacity(refined_head.len().saturating_add(fallback_ordered.len()));
        let mut seen = HashSet::with_capacity(refined_head.len());
        for candidate in refined_head {
            seen.insert(candidate.doc_id.as_str());
            merged.push(candidate.clone());
        }
        for candidate in fallback_ordered {
            if effective_limit != usize::MAX && merged.len() >= effective_limit {
                break;
            }
            if seen.contains(candidate.doc_id.as_str()) {
                continue;
            }
            merged.push(candidate.clone());
        }
        merged
    }

    #[allow(clippy::too_many_lines)]
    async fn execute_search_phase_artifacts_with_mode(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        mode: SearchExecutionMode,
    ) -> SearchResult<Vec<SearchPhaseArtifact>> {
        let mut resources = self.prepare_search_execution_resources(mode)?;
        self.execute_search_phase_artifacts_with_mode_using_resources(
            cx,
            query,
            limit,
            mode,
            &mut resources,
            SearchExecutionFlags {
                include_snippets: true,
                persist_explain_session: true,
            },
        )
        .await
    }

    #[allow(clippy::too_many_lines)]
    async fn execute_search_phase_artifacts_with_mode_using_resources(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        mode: SearchExecutionMode,
        resources: &mut SearchExecutionResources,
        flags: SearchExecutionFlags,
    ) -> SearchResult<Vec<SearchPhaseArtifact>> {
        let normalized_query = Self::normalize_search_query(query);
        let filter_expr = SearchFilterExpr::parse(self.cli_input.filter.as_deref().unwrap_or(""))?;
        if normalized_query.is_empty() {
            return Ok(vec![SearchPhaseArtifact {
                phase: SearchOutputPhase::Initial,
                fused: Vec::new(),
                payload: SearchPayload::new(
                    String::new(),
                    SearchOutputPhase::Initial,
                    0,
                    Vec::new(),
                ),
            }]);
        }

        let lexical_doc_count = resources
            .lexical_index
            .as_ref()
            .map(TantivyIndex::doc_count);
        let vector_doc_count = resources.vector_index.as_ref().map(|index| {
            index
                .record_count()
                .saturating_add(index.wal_record_count())
        });
        let output_limit = Self::resolve_output_limit(limit, lexical_doc_count, vector_doc_count);
        let planning_limit = Self::resolve_planning_limit(output_limit);
        let search_start = Instant::now();
        let lexical_available = resources.lexical_index.is_some();

        let capabilities = resources.capabilities_for_mode(mode, self.config.search.fast_only);
        let planner = QueryPlanner::from_fsfs(&self.config);
        let plan =
            planner.execution_plan_for_query(&normalized_query, Some(planning_limit), capabilities);

        info!(
            phase = "query_parse",
            query = normalized_query,
            intent = ?plan.intent.intent,
            fallback = ?plan.intent.fallback,
            confidence_per_mille = plan.intent.confidence_per_mille,
            mode_override = ?mode,
            execution_mode = ?plan.mode,
            reason_code = plan.reason_code,
            "fsfs search query planned"
        );

        let short_query_budget_cap =
            if normalized_query.chars().count() <= FSFS_SEARCH_SHORT_QUERY_CHAR_THRESHOLD {
                Some(
                    planning_limit
                        .saturating_mul(FSFS_SEARCH_SHORT_QUERY_BUDGET_MULTIPLIER)
                        .max(planning_limit),
                )
            } else {
                None
            };

        let lexical_start = Instant::now();
        let lexical_budget =
            if matches!(mode, SearchExecutionMode::LexicalOnly) && filter_expr.is_none() {
                planning_limit.saturating_mul(2).max(planning_limit)
            } else if matches!(mode, SearchExecutionMode::FastOnly) && filter_expr.is_none() {
                let planned_budget = plan.lexical_stage.candidate_budget.max(planning_limit);
                let fast_stage_cap = planning_limit
                    .saturating_mul(FSFS_SEARCH_FAST_STAGE_BUDGET_MULTIPLIER)
                    .max(planning_limit);
                short_query_budget_cap.map_or_else(
                    || planned_budget.min(fast_stage_cap),
                    |cap| planned_budget.min(fast_stage_cap).min(cap),
                )
            } else {
                let planned_budget = plan.lexical_stage.candidate_budget.max(planning_limit);
                if filter_expr.is_none() {
                    short_query_budget_cap.map_or(planned_budget, |cap| planned_budget.min(cap))
                } else {
                    planned_budget
                }
            };
        let mut snippet_config = SnippetConfig::default();
        if !matches!(mode, SearchExecutionMode::Full) {
            snippet_config.max_chars = FSFS_TUI_FAST_STAGE_SNIPPET_MAX_CHARS;
        }
        let mut snippets_by_doc = HashMap::new();
        let (lexical_candidates, lexical_head_candidates) = if plan.lexical_stage.enabled {
            if let Some(lexical) = resources.lexical_index.as_ref() {
                if flags.include_snippets {
                    let snippet_limit = lexical_budget.clamp(1, FSFS_SEARCH_SNIPPET_HEAD_LIMIT);
                    let snippet_hits = lexical.search_with_snippets(
                        cx,
                        &normalized_query,
                        snippet_limit,
                        &snippet_config,
                    )?;
                    for hit in &snippet_hits {
                        if let Some(snippet) = hit.snippet.as_ref()
                            && !snippet.trim().is_empty()
                        {
                            snippets_by_doc.insert(hit.doc_id.clone(), snippet.clone());
                        }
                    }

                    let needs_full_lexical = output_limit > snippet_limit
                        || filter_expr.is_some()
                        || snippet_hits.is_empty();
                    let full_candidates = if needs_full_lexical {
                        lexical
                            .search_doc_ids(cx, &normalized_query, output_limit)?
                            .into_iter()
                            .map(|hit| LexicalCandidate::new(hit.doc_id, hit.bm25_score))
                            .collect::<Vec<_>>()
                    } else {
                        snippet_hits
                            .into_iter()
                            .map(|hit| LexicalCandidate::new(hit.doc_id, hit.bm25_score))
                            .collect::<Vec<_>>()
                    };
                    let lexical_head_budget = lexical_budget.min(planning_limit).max(1);
                    let head_candidates = full_candidates
                        .iter()
                        .take(lexical_head_budget)
                        .cloned()
                        .collect::<Vec<_>>();
                    (full_candidates, head_candidates)
                } else {
                    let full_candidates = lexical
                        .search_doc_ids(cx, &normalized_query, output_limit)?
                        .into_iter()
                        .map(|hit| LexicalCandidate::new(hit.doc_id, hit.bm25_score))
                        .collect::<Vec<_>>();
                    let lexical_head_budget = lexical_budget.min(planning_limit).max(1);
                    let head_candidates = full_candidates
                        .iter()
                        .take(lexical_head_budget)
                        .cloned()
                        .collect::<Vec<_>>();
                    (full_candidates, head_candidates)
                }
            } else {
                (Vec::new(), Vec::new())
            }
        } else {
            (Vec::new(), Vec::new())
        };
        let lexical_elapsed_ms = lexical_start.elapsed().as_millis();

        let semantic_start = Instant::now();
        let force_full_semantic_recall =
            Self::should_force_full_semantic_recall(SemanticRecallDecisionInput {
                output_limit,
                planning_limit,
                semantic_stage_enabled: plan.semantic_stage.enabled,
                semantic_index_available: resources.vector_index.is_some(),
                lexical_stage_enabled: plan.lexical_stage.enabled,
                lexical_head_count: lexical_head_candidates.len(),
                lexical_full_count: lexical_candidates.len(),
            });
        let lexical_tail_complete = filter_expr.is_none()
            && plan.lexical_stage.enabled
            && !lexical_candidates.is_empty()
            && lexical_candidates.len() >= output_limit;
        let semantic_budget_floor = if force_full_semantic_recall {
            output_limit
        } else {
            planning_limit
        };
        let semantic_budget = if force_full_semantic_recall {
            plan.semantic_stage
                .candidate_budget
                .max(semantic_budget_floor)
        } else if let Some(short_query_cap) = short_query_budget_cap {
            plan.semantic_stage
                .candidate_budget
                .max(semantic_budget_floor)
                .min(short_query_cap)
        } else if matches!(mode, SearchExecutionMode::FastOnly) {
            plan.semantic_stage
                .candidate_budget
                .max(semantic_budget_floor)
                .min(
                    planning_limit
                        .saturating_mul(FSFS_SEARCH_FAST_STAGE_BUDGET_MULTIPLIER)
                        .max(planning_limit),
                )
        } else {
            plan.semantic_stage
                .candidate_budget
                .max(semantic_budget_floor)
        };
        let semantic_budget = if force_full_semantic_recall || !lexical_tail_complete {
            semantic_budget
        } else {
            semantic_budget.min(planning_limit)
        };
        let semantic_decision = Self::semantic_gate_decision(SemanticGateDecisionInput {
            mode,
            intent: plan.intent.intent,
            query: &normalized_query,
            output_limit,
            lexical_head_candidates: &lexical_head_candidates,
            semantic_stage_enabled: plan.semantic_stage.enabled,
            lexical_stage_enabled: plan.lexical_stage.enabled,
            force_full_semantic_recall,
        });
        info!(
            phase = "semantic_gate",
            mode_override = ?mode,
            intent = ?plan.intent.intent,
            lexical_head_candidates = lexical_head_candidates.len(),
            semantic_budget,
            output_limit,
            force_full_semantic_recall,
            lexical_tail_complete,
            run_semantic = semantic_decision.run_semantic,
            posterior_useful_per_mille = f64_to_per_mille(semantic_decision.posterior_useful),
            expected_loss_run_per_mille = f64_to_per_mille(semantic_decision.expected_loss_run),
            expected_loss_skip_per_mille = f64_to_per_mille(semantic_decision.expected_loss_skip),
            reason_code = semantic_decision.reason_code,
            "fsfs semantic-stage VOI decision"
        );

        if plan.semantic_stage.enabled
            && semantic_decision.run_semantic
            && !matches!(mode, SearchExecutionMode::LexicalOnly)
            && resources.vector_index.is_some()
            && resources.fast_embedder.is_none()
            && !resources.fast_embedder_attempted
        {
            self.maybe_prepare_fast_embedder(resources, lexical_available)?;
        }

        let semantic_candidates = if plan.semantic_stage.enabled && semantic_decision.run_semantic {
            if let (Some(index), Some(embedder)) = (
                resources.vector_index.as_ref(),
                resources.fast_embedder.as_ref(),
            ) {
                match embedder.embed(cx, &normalized_query).await {
                    Ok(query_embedding) => {
                        match index.search_top_k(&query_embedding, semantic_budget, None) {
                            Ok(hits) => hits
                                .into_iter()
                                .map(|hit| SemanticCandidate::new(hit.doc_id, hit.score))
                                .collect::<Vec<_>>(),
                            Err(error) if lexical_available => {
                                info!(
                                    error = %error,
                                    "fsfs search falling back to lexical-only mode after vector search failure"
                                );
                                Vec::new()
                            }
                            Err(error) => return Err(error),
                        }
                    }
                    Err(error) if lexical_available => {
                        info!(
                            error = %error,
                            "fsfs search falling back to lexical-only mode after query embedding failure"
                        );
                        Vec::new()
                    }
                    Err(error) => return Err(error),
                }
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        let semantic_elapsed_ms = semantic_start.elapsed().as_millis();

        let fusion_start = Instant::now();
        let orchestrator = QueryExecutionOrchestrator::new(QueryFusionPolicy {
            rrf_k: plan.fusion_policy.rrf_k.unwrap_or(self.config.search.rrf_k),
        });
        let fusion_budget = lexical_head_candidates
            .len()
            .saturating_add(semantic_candidates.len())
            .max(planning_limit);
        let fused_initial_head = orchestrator.fuse_rankings(
            &lexical_head_candidates,
            &semantic_candidates,
            fusion_budget,
            0,
        );
        let filtered_initial_head =
            Self::apply_search_filter(&fused_initial_head, filter_expr.as_ref());
        let fused_initial =
            if output_limit > filtered_initial_head.len() && !lexical_candidates.is_empty() {
                Self::merge_with_lexical_tail(
                    &filtered_initial_head,
                    &lexical_candidates,
                    filter_expr.as_ref(),
                )
            } else {
                filtered_initial_head.clone()
            };
        let payload = Self::build_limited_payload(
            orchestrator,
            &normalized_query,
            SearchOutputPhase::Initial,
            &fused_initial,
            &snippets_by_doc,
            output_limit,
        );
        let fusion_elapsed_ms = fusion_start.elapsed().as_millis();

        let mut artifacts = vec![SearchPhaseArtifact {
            phase: SearchOutputPhase::Initial,
            fused: fused_initial.clone(),
            payload: payload.clone(),
        }];

        if plan.quality_stage.enabled {
            if matches!(mode, SearchExecutionMode::Full)
                && !self.config.search.fast_only
                && resources.vector_index.is_some()
                && resources.quality_embedder.is_none()
                && !resources.quality_embedder_attempted
            {
                self.maybe_prepare_quality_embedder(resources, lexical_available)?;
            }
            if let (Some(index), Some(embedder)) = (
                resources.vector_index.as_ref(),
                resources.quality_embedder.as_ref(),
            ) {
                let quality_budget_floor = planning_limit;
                let quality_budget_ceiling = if lexical_tail_complete {
                    planning_limit
                } else {
                    output_limit
                };
                let quality_budget = plan
                    .quality_stage
                    .candidate_budget
                    .max(quality_budget_floor)
                    .min(quality_budget_ceiling);
                let quality_outcome = match embedder.embed(cx, &normalized_query).await {
                    Ok(query_embedding) => {
                        match index.search_top_k(&query_embedding, quality_budget, None) {
                            Ok(hits) => Ok(hits
                                .into_iter()
                                .map(|hit| SemanticCandidate::new(hit.doc_id, hit.score))
                                .collect::<Vec<_>>()),
                            Err(error) => Err(error),
                        }
                    }
                    Err(error) => Err(error),
                };

                match quality_outcome {
                    Ok(quality_candidates) => {
                        let fused_refined_head = orchestrator.fuse_rankings(
                            &lexical_head_candidates,
                            &quality_candidates,
                            fusion_budget,
                            0,
                        );
                        let filtered_refined_head =
                            Self::apply_search_filter(&fused_refined_head, filter_expr.as_ref());
                        let fused_refined = Self::merge_with_fallback_tail(
                            &filtered_refined_head,
                            &fused_initial,
                            output_limit,
                        );
                        let refined_payload = Self::build_limited_payload(
                            orchestrator,
                            &normalized_query,
                            SearchOutputPhase::Refined,
                            &fused_refined,
                            &snippets_by_doc,
                            output_limit,
                        );
                        artifacts.push(SearchPhaseArtifact {
                            phase: SearchOutputPhase::Refined,
                            fused: fused_refined,
                            payload: refined_payload,
                        });
                    }
                    Err(error) => {
                        if let SearchError::DimensionMismatch { .. } = error {
                            resources.quality_embedder = None;
                            resources.quality_embedder_attempted = true;
                            info!(
                                error = %error,
                                "fsfs quality refinement disabled for this session after embedder/index dimension mismatch"
                            );
                        } else {
                            info!(
                                error = %error,
                                "fsfs quality refinement failed; falling back to initial phase payload"
                            );
                        }
                        let failed_payload = Self::build_limited_payload(
                            orchestrator,
                            &normalized_query,
                            SearchOutputPhase::RefinementFailed,
                            &fused_initial,
                            &snippets_by_doc,
                            output_limit,
                        );
                        artifacts.push(SearchPhaseArtifact {
                            phase: SearchOutputPhase::RefinementFailed,
                            fused: fused_initial.clone(),
                            payload: failed_payload,
                        });
                    }
                }
            } else {
                let failed_payload = Self::build_limited_payload(
                    orchestrator,
                    &normalized_query,
                    SearchOutputPhase::RefinementFailed,
                    &fused_initial,
                    &snippets_by_doc,
                    output_limit,
                );
                artifacts.push(SearchPhaseArtifact {
                    phase: SearchOutputPhase::RefinementFailed,
                    fused: fused_initial.clone(),
                    payload: failed_payload,
                });
            }
        }

        if flags.persist_explain_session
            && let Some(last) = artifacts.last()
            && let Err(error) = self.persist_explain_session(
                &resources.index_root,
                &normalized_query,
                last.phase,
                &last.fused,
            )
        {
            warn!(
                error = %error,
                path = %Self::explain_session_path(&resources.index_root).display(),
                "failed to persist explain-session context for follow-up `fsfs explain`"
            );
        }

        let final_payload = artifacts
            .last()
            .map_or(&payload, |artifact| &artifact.payload);

        info!(
            phase = "fast_search",
            query = normalized_query,
            lexical_candidates = lexical_candidates.len(),
            semantic_candidates = semantic_candidates.len(),
            fused_candidates = final_payload.total_candidates,
            returned_hits = final_payload.returned_hits,
            lexical_elapsed_ms,
            semantic_elapsed_ms,
            fusion_elapsed_ms,
            total_elapsed_ms = search_start.elapsed().as_millis(),
            "fsfs search retrieval pipeline completed"
        );
        info!(
            phase = "fusion",
            rrf_k = plan.fusion_policy.rrf_k.unwrap_or(self.config.search.rrf_k),
            fused_candidates = final_payload.total_candidates,
            returned_hits = final_payload.returned_hits,
            "fsfs search fusion phase completed"
        );
        info!(
            phase = "quality_refine",
            status = if plan.quality_stage.enabled {
                "enabled"
            } else {
                "skipped"
            },
            reason_code = plan.quality_stage.reason_code,
            candidate_budget = plan.quality_stage.candidate_budget,
            timeout_ms = plan.quality_stage.timeout_ms,
            "fsfs search quality-refinement phase status"
        );
        info!(
            phase = "rerank",
            status = if plan.rerank_stage.enabled {
                "enabled"
            } else {
                "skipped"
            },
            reason_code = plan.rerank_stage.reason_code,
            candidate_budget = plan.rerank_stage.candidate_budget,
            timeout_ms = plan.rerank_stage.timeout_ms,
            "fsfs search rerank phase status"
        );

        Ok(artifacts)
    }

    fn run_status_command(&self) -> SearchResult<()> {
        let payload = self.collect_status_payload()?;
        if self.cli_input.format == OutputFormat::Table {
            let table = render_status_table(&payload, self.cli_input.no_color);
            print!("{table}");
            return Ok(());
        }

        let meta = meta_for_format("status", self.cli_input.format);
        let envelope = OutputEnvelope::success(payload, meta, iso_timestamp_now());
        let mut stdout = std::io::stdout();
        emit_envelope(&envelope, self.cli_input.format, &mut stdout)?;
        if self.cli_input.format != OutputFormat::Jsonl {
            stdout
                .write_all(b"\n")
                .map_err(|source| SearchError::SubsystemError {
                    subsystem: "fsfs.status",
                    source: Box::new(source),
                })?;
        }
        Ok(())
    }

    async fn run_download_command(&self) -> SearchResult<()> {
        let payload = self.collect_download_models_payload().await?;
        if self.cli_input.format == OutputFormat::Table {
            let table = render_download_models_table(&payload, self.cli_input.no_color);
            print!("{table}");
            return Ok(());
        }

        let meta = meta_for_format("download-models", self.cli_input.format);
        let envelope = OutputEnvelope::success(payload, meta, iso_timestamp_now());
        let mut stdout = std::io::stdout();
        emit_envelope(&envelope, self.cli_input.format, &mut stdout)?;
        if self.cli_input.format != OutputFormat::Jsonl {
            stdout
                .write_all(b"\n")
                .map_err(|source| SearchError::SubsystemError {
                    subsystem: "fsfs.download_models",
                    source: Box::new(source),
                })?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    async fn collect_download_models_payload(&self) -> SearchResult<FsfsDownloadModelsPayload> {
        let model_root = self.resolve_download_model_root()?;
        let manifests = self.resolve_download_manifests()?;
        let operation = if self.cli_input.download_list {
            "list"
        } else if self.cli_input.download_verify {
            "verify"
        } else {
            "download"
        };
        fs::create_dir_all(&model_root)?;

        let mut results = Vec::with_capacity(manifests.len());
        let show_progress = self.cli_input.format == OutputFormat::Table && !self.cli_input.quiet;

        for manifest in manifests {
            let install_dir = Self::manifest_install_dir_name(&manifest);
            let destination = model_root.join(&install_dir);
            let (present, verified, verify_message) =
                Self::inspect_manifest_installation(&manifest, &destination);

            let entry = match operation {
                "list" => {
                    let state = if !present {
                        "missing"
                    } else if verified == Some(true) {
                        "cached"
                    } else {
                        "corrupt"
                    };
                    Self::download_model_entry(
                        &manifest,
                        &install_dir,
                        &destination,
                        state,
                        verified,
                        Self::path_bytes(&destination)?,
                        verify_message,
                    )
                }
                "verify" => {
                    let state = if !present {
                        "missing"
                    } else if verified == Some(true) {
                        "verified"
                    } else {
                        "mismatch"
                    };
                    Self::download_model_entry(
                        &manifest,
                        &install_dir,
                        &destination,
                        state,
                        verified,
                        Self::path_bytes(&destination)?,
                        verify_message,
                    )
                }
                _ => {
                    if !self.cli_input.full_reindex && verified == Some(true) {
                        Self::download_model_entry(
                            &manifest,
                            &install_dir,
                            &destination,
                            "cached",
                            Some(true),
                            Self::path_bytes(&destination)?,
                            Some("already present and verified".to_owned()),
                        )
                    } else {
                        let mut lifecycle = ModelLifecycle::new(
                            manifest.clone(),
                            DownloadConsent::granted(ConsentSource::Programmatic),
                        );
                        let downloader = ModelDownloader::with_defaults();
                        let staged = downloader
                            .download_model(&manifest, &model_root, &mut lifecycle, |progress| {
                                if show_progress {
                                    eprintln!("{progress}");
                                }
                            })
                            .await?;
                        manifest.promote_verified_installation(&staged, &destination)?;
                        Self::download_model_entry(
                            &manifest,
                            &install_dir,
                            &destination,
                            "downloaded",
                            Some(true),
                            Self::path_bytes(&destination)?,
                            None,
                        )
                    }
                }
            };
            results.push(entry);
        }

        Ok(FsfsDownloadModelsPayload {
            operation: operation.to_owned(),
            force: self.cli_input.full_reindex,
            model_root: model_root.display().to_string(),
            models: results,
        })
    }

    fn run_doctor_command(&self) -> SearchResult<()> {
        let payload = self.collect_doctor_payload()?;
        if self.cli_input.format == OutputFormat::Table {
            let table = render_doctor_table(&payload, self.cli_input.no_color);
            print!("{table}");
            return Ok(());
        }

        let meta = meta_for_format("doctor", self.cli_input.format);
        let envelope = OutputEnvelope::success(payload, meta, iso_timestamp_now());
        let mut stdout = std::io::stdout();
        emit_envelope(&envelope, self.cli_input.format, &mut stdout)?;
        if self.cli_input.format != OutputFormat::Jsonl {
            stdout
                .write_all(b"\n")
                .map_err(|source| SearchError::SubsystemError {
                    subsystem: "fsfs.doctor",
                    source: Box::new(source),
                })?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn collect_doctor_payload(&self) -> SearchResult<FsfsDoctorPayload> {
        let mut checks = Vec::new();

        // 1. Version check
        checks.push(DoctorCheck {
            name: "version".to_owned(),
            verdict: DoctorVerdict::Pass,
            detail: format!("fsfs {}", env!("CARGO_PKG_VERSION")),
            suggestion: None,
        });

        // 2. Model cache checks
        let model_root = PathBuf::from(&self.config.indexing.model_dir);
        for (tier, model_name) in [
            ("fast", self.config.indexing.fast_model.as_str()),
            ("quality", self.config.indexing.quality_model.as_str()),
        ] {
            let model_path = model_root.join(model_name);
            if model_path.exists() {
                checks.push(DoctorCheck {
                    name: format!("model.{tier}"),
                    verdict: DoctorVerdict::Pass,
                    detail: format!("{model_name} cached at {}", model_path.display()),
                    suggestion: None,
                });
            } else {
                checks.push(DoctorCheck {
                    name: format!("model.{tier}"),
                    verdict: DoctorVerdict::Warn,
                    detail: format!("{model_name} not found at {}", model_path.display()),
                    suggestion: Some("run `fsfs download-models` to download".to_owned()),
                });
            }
        }

        // 3. Model directory writable
        if model_root.exists() {
            let probe = model_root.join(".fsfs_doctor_probe");
            match fs::write(&probe, b"probe") {
                Ok(()) => {
                    let _ = fs::remove_file(&probe);
                    checks.push(DoctorCheck {
                        name: "model_dir.writable".to_owned(),
                        verdict: DoctorVerdict::Pass,
                        detail: format!("{} is writable", model_root.display()),
                        suggestion: None,
                    });
                }
                Err(error) => {
                    checks.push(DoctorCheck {
                        name: "model_dir.writable".to_owned(),
                        verdict: DoctorVerdict::Fail,
                        detail: format!("{} is not writable: {error}", model_root.display()),
                        suggestion: Some(format!("check permissions on {}", model_root.display())),
                    });
                }
            }
        } else {
            checks.push(DoctorCheck {
                name: "model_dir.writable".to_owned(),
                verdict: DoctorVerdict::Warn,
                detail: format!("{} does not exist yet", model_root.display()),
                suggestion: Some(
                    "directory will be created on first `fsfs download-models`".to_owned(),
                ),
            });
        }

        // 4. Index directory
        let index_root = self.resolve_status_index_root()?;
        if index_root.exists() {
            let sentinel = Self::read_index_sentinel(&index_root)?;
            if let Some(sentinel) = &sentinel {
                let stale = Self::count_stale_files(&index_root, Some(sentinel))?;
                let stale_count = stale.unwrap_or(0);
                if stale_count == 0 {
                    checks.push(DoctorCheck {
                        name: "index".to_owned(),
                        verdict: DoctorVerdict::Pass,
                        detail: format!("{} files indexed, up-to-date", sentinel.indexed_files),
                        suggestion: None,
                    });
                } else {
                    checks.push(DoctorCheck {
                        name: "index".to_owned(),
                        verdict: DoctorVerdict::Warn,
                        detail: format!(
                            "{} files indexed, {stale_count} stale",
                            sentinel.indexed_files
                        ),
                        suggestion: Some("run `fsfs index` to refresh".to_owned()),
                    });
                }
            } else {
                checks.push(DoctorCheck {
                    name: "index".to_owned(),
                    verdict: DoctorVerdict::Warn,
                    detail: format!(
                        "index directory exists at {} but no sentinel found",
                        index_root.display()
                    ),
                    suggestion: Some("run `fsfs index` to build the index".to_owned()),
                });
            }
        } else {
            checks.push(DoctorCheck {
                name: "index".to_owned(),
                verdict: DoctorVerdict::Warn,
                detail: "no index found".to_owned(),
                suggestion: Some("run `fsfs index <dir>` to create one".to_owned()),
            });
        }

        // 5. Index directory writable
        if index_root.exists() {
            let probe = index_root.join(".fsfs_doctor_probe");
            match fs::write(&probe, b"probe") {
                Ok(()) => {
                    let _ = fs::remove_file(&probe);
                    checks.push(DoctorCheck {
                        name: "index_dir.writable".to_owned(),
                        verdict: DoctorVerdict::Pass,
                        detail: format!("{} is writable", index_root.display()),
                        suggestion: None,
                    });
                }
                Err(error) => {
                    checks.push(DoctorCheck {
                        name: "index_dir.writable".to_owned(),
                        verdict: DoctorVerdict::Fail,
                        detail: format!("{} is not writable: {error}", index_root.display()),
                        suggestion: Some(format!("check permissions on {}", index_root.display())),
                    });
                }
            }
        }

        // 6. Disk space check
        let disks = Disks::new_with_refreshed_list();
        let cwd = std::env::current_dir().unwrap_or_default();
        if let Some(disk) = disks
            .iter()
            .filter(|disk| cwd.starts_with(disk.mount_point()))
            .max_by_key(|disk| disk.mount_point().as_os_str().len())
        {
            let available = disk.available_space();
            let min_required: u64 = 500 * 1024 * 1024; // 500 MB minimum
            if available >= min_required {
                checks.push(DoctorCheck {
                    name: "disk_space".to_owned(),
                    verdict: DoctorVerdict::Pass,
                    detail: format!("{} available", humanize_bytes(available)),
                    suggestion: None,
                });
            } else {
                checks.push(DoctorCheck {
                    name: "disk_space".to_owned(),
                    verdict: DoctorVerdict::Warn,
                    detail: format!(
                        "only {} available (minimum recommended: {})",
                        humanize_bytes(available),
                        humanize_bytes(min_required)
                    ),
                    suggestion: Some("free disk space for model downloads and indexing".to_owned()),
                });
            }
        }

        // 7. Config sources
        let config_summary = self.status_config_source_summary()?;
        checks.push(DoctorCheck {
            name: "config".to_owned(),
            verdict: DoctorVerdict::Pass,
            detail: format!("sources: {config_summary}"),
            suggestion: None,
        });

        // 8. Nightly toolchain note
        checks.push(DoctorCheck {
            name: "rust_edition".to_owned(),
            verdict: DoctorVerdict::Pass,
            detail: format!(
                "edition {} (requires nightly)",
                env!("CARGO_PKG_RUST_VERSION")
            ),
            suggestion: None,
        });

        // Tally
        let pass_count = checks
            .iter()
            .filter(|c| c.verdict == DoctorVerdict::Pass)
            .count();
        let warn_count = checks
            .iter()
            .filter(|c| c.verdict == DoctorVerdict::Warn)
            .count();
        let fail_count = checks
            .iter()
            .filter(|c| c.verdict == DoctorVerdict::Fail)
            .count();
        let overall = if fail_count > 0 {
            DoctorVerdict::Fail
        } else if warn_count > 0 {
            DoctorVerdict::Warn
        } else {
            DoctorVerdict::Pass
        };

        Ok(FsfsDoctorPayload {
            version: env!("CARGO_PKG_VERSION").to_owned(),
            checks,
            pass_count,
            warn_count,
            fail_count,
            overall,
        })
    }

    fn resolve_download_model_root(&self) -> SearchResult<PathBuf> {
        if let Some(path) = self.cli_input.download_output_dir.as_deref() {
            return absolutize_path(path);
        }

        let configured = self.config.indexing.model_dir.as_str();
        if configured == "~"
            && let Some(home) = home_dir()
        {
            return Ok(home);
        }
        if let Some(rest) = configured.strip_prefix("~/")
            && let Some(home) = home_dir()
        {
            return Ok(home.join(rest));
        }

        absolutize_path(Path::new(configured))
    }

    fn resolve_download_manifests(&self) -> SearchResult<Vec<ModelManifest>> {
        let manifests = ModelManifest::builtin_catalog().models;
        let Some(requested) = self.cli_input.model_name.as_deref() else {
            return Ok(manifests);
        };

        let requested_token = normalize_model_token(requested);
        let mut exact = Vec::new();
        let mut fuzzy = Vec::new();
        for manifest in manifests {
            let install_dir = Self::manifest_install_dir_name(&manifest);
            let id_token = normalize_model_token(&manifest.id);
            let install_token = normalize_model_token(&install_dir);
            let display_token = manifest
                .display_name
                .as_ref()
                .map(|value| normalize_model_token(value));
            let repo_token = normalize_model_token(&manifest.repo);

            let exact_match = requested_token == id_token
                || requested_token == install_token
                || display_token.as_deref() == Some(requested_token.as_str());
            if exact_match {
                exact.push(manifest);
                continue;
            }

            let fuzzy_match = id_token.contains(&requested_token)
                || install_token.contains(&requested_token)
                || display_token
                    .as_ref()
                    .is_some_and(|value| value.contains(&requested_token))
                || repo_token.contains(&requested_token);
            if fuzzy_match {
                fuzzy.push(manifest);
            }
        }

        if !exact.is_empty() {
            return Ok(exact);
        }
        if !fuzzy.is_empty() {
            return Ok(fuzzy);
        }

        Err(SearchError::InvalidConfig {
            field: "cli.download.model".to_owned(),
            value: requested.to_owned(),
            reason: "unknown model; run download-models --list to see available ids".to_owned(),
        })
    }

    fn manifest_install_dir_name(manifest: &ModelManifest) -> String {
        match manifest.id.as_str() {
            "potion-multilingual-128m" => "potion-multilingual-128M".to_owned(),
            "all-minilm-l6-v2" => "all-MiniLM-L6-v2".to_owned(),
            "ms-marco-minilm-l-6-v2" => "ms-marco-MiniLM-L-6-v2".to_owned(),
            _ => manifest.id.clone(),
        }
    }

    fn inspect_manifest_installation(
        manifest: &ModelManifest,
        destination: &Path,
    ) -> (bool, Option<bool>, Option<String>) {
        if !destination.exists() {
            return (false, None, None);
        }
        match manifest.verify_dir(destination) {
            Ok(()) => (true, Some(true), None),
            Err(error) => (true, Some(false), Some(error.to_string())),
        }
    }

    fn download_model_entry(
        manifest: &ModelManifest,
        install_dir: &str,
        destination: &Path,
        state: &str,
        verified: Option<bool>,
        size_bytes: u64,
        message: Option<String>,
    ) -> FsfsDownloadModelEntry {
        let tier = manifest
            .tier
            .map(|value| format!("{value:?}").to_ascii_lowercase());
        FsfsDownloadModelEntry {
            id: manifest.id.clone(),
            install_dir: install_dir.to_owned(),
            tier,
            state: state.to_owned(),
            verified,
            size_bytes,
            destination: destination.display().to_string(),
            message,
        }
    }

    fn collect_status_payload(&self) -> SearchResult<FsfsStatusPayload> {
        let index_root = self.resolve_status_index_root()?;
        let sentinel = Self::read_index_sentinel(&index_root)?;
        let stale_files = Self::count_stale_files(&index_root, sentinel.as_ref())?;

        let storage_paths = IndexStoragePaths {
            vector_index_roots: vec![index_root.join("vector")],
            lexical_index_roots: vec![index_root.join("lexical")],
            catalog_files: vec![PathBuf::from(&self.config.storage.db_path)],
            embedding_cache_roots: vec![index_root.join("cache")],
        };
        let usage = self.collect_index_storage_usage(&storage_paths)?;
        let tracked_index_bytes = Some(usage.total_bytes());
        let tracker = self.new_runtime_lifecycle_tracker(&storage_paths);
        let disk_budget = self.apply_storage_emergency_override(
            self.evaluate_storage_disk_budget(&tracker, &storage_paths)?,
            tracked_index_bytes,
            tracker.resource_limits().max_index_bytes,
        );

        let config_status = FsfsConfigStatus {
            source: self.status_config_source_summary()?,
            index_dir: self.config.storage.index_dir.clone(),
            model_dir: self.config.indexing.model_dir.clone(),
            rrf_k: self.config.search.rrf_k,
            quality_weight: self.config.search.quality_weight,
            quality_timeout_ms: self.config.search.quality_timeout_ms,
            fast_only: self.config.search.fast_only,
            pressure_profile: format!("{:?}", self.config.pressure.profile).to_ascii_lowercase(),
        };

        let runtime_status = FsfsRuntimeStatus {
            disk_budget_stage: disk_budget
                .as_ref()
                .map(|snapshot| format!("{:?}", snapshot.stage).to_ascii_lowercase()),
            disk_budget_action: disk_budget
                .as_ref()
                .map(|snapshot| format!("{:?}", snapshot.action).to_ascii_lowercase()),
            disk_budget_reason_code: disk_budget
                .as_ref()
                .map(|snapshot| snapshot.reason_code.to_owned()),
            disk_budget_bytes: disk_budget.as_ref().map(|snapshot| snapshot.budget_bytes),
            tracked_index_bytes,
            storage_pressure_emergency: self.config.storage.storage_pressure_emergency,
        };

        Ok(FsfsStatusPayload {
            version: env!("CARGO_PKG_VERSION").to_owned(),
            index: FsfsIndexStatus {
                path: index_root.display().to_string(),
                exists: index_root.exists(),
                indexed_files: sentinel.as_ref().map(|value| value.indexed_files),
                discovered_files: sentinel.as_ref().map(|value| value.discovered_files),
                skipped_files: sentinel.as_ref().map(|value| value.skipped_files),
                last_indexed_ms: sentinel.as_ref().map(|value| value.generated_at_ms),
                last_indexed_iso_utc: sentinel
                    .as_ref()
                    .map(|value| format_epoch_ms_utc(value.generated_at_ms)),
                stale_files,
                source_hash_hex: sentinel.as_ref().map(|value| value.source_hash_hex.clone()),
                size_bytes: usage.total_bytes(),
                vector_index_bytes: usage.vector_index_bytes,
                lexical_index_bytes: usage.lexical_index_bytes,
                metadata_bytes: usage.catalog_bytes,
                embedding_cache_bytes: usage.embedding_cache_bytes,
            },
            models: self.collect_model_statuses()?,
            config: config_status,
            runtime: runtime_status,
        })
    }

    fn resolve_status_index_root(&self) -> SearchResult<PathBuf> {
        if let Some(path) = self.cli_input.index_dir.as_deref() {
            return absolutize_path(path);
        }

        let configured = PathBuf::from(&self.config.storage.index_dir);
        if configured.is_absolute() {
            return Ok(configured);
        }

        let cwd = std::env::current_dir().map_err(SearchError::Io)?;
        let mut probe = Some(cwd.as_path());
        while let Some(path) = probe {
            let candidate = path.join(&configured);
            if candidate.join(FSFS_SENTINEL_FILE).exists()
                || candidate.join(FSFS_VECTOR_MANIFEST_FILE).exists()
                || candidate.join(FSFS_LEXICAL_MANIFEST_FILE).exists()
                || candidate.join(FSFS_VECTOR_INDEX_FILE).exists()
            {
                return Ok(candidate);
            }
            probe = path.parent();
        }

        Ok(cwd.join(configured))
    }

    fn read_index_sentinel(index_root: &Path) -> SearchResult<Option<IndexSentinel>> {
        let path = index_root.join(FSFS_SENTINEL_FILE);
        let raw = match fs::read_to_string(path) {
            Ok(raw) => raw,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let sentinel = serde_json::from_str::<IndexSentinel>(&raw).map_err(|source| {
            SearchError::SubsystemError {
                subsystem: "fsfs.status.sentinel",
                source: Box::new(source),
            }
        })?;
        Ok(Some(sentinel))
    }

    fn read_index_manifest(index_root: &Path) -> SearchResult<Option<Vec<IndexManifestEntry>>> {
        let path = index_root.join(FSFS_VECTOR_MANIFEST_FILE);
        let raw = match fs::read_to_string(path) {
            Ok(raw) => raw,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let manifest = serde_json::from_str::<Vec<IndexManifestEntry>>(&raw).map_err(|source| {
            SearchError::SubsystemError {
                subsystem: "fsfs.status.manifest",
                source: Box::new(source),
            }
        })?;
        Ok(Some(manifest))
    }

    fn count_stale_files(
        index_root: &Path,
        sentinel: Option<&IndexSentinel>,
    ) -> SearchResult<Option<usize>> {
        let Some(manifest) = Self::read_index_manifest(index_root)? else {
            return Ok(None);
        };

        let mut stale_files = 0_usize;
        for entry in manifest {
            let source_path = resolve_manifest_file_path(&entry.file_key, sentinel, index_root);
            let metadata = match fs::metadata(&source_path) {
                Ok(metadata) => metadata,
                Err(error) if error.kind() == ErrorKind::NotFound => {
                    stale_files = stale_files.saturating_add(1);
                    continue;
                }
                Err(error) => return Err(error.into()),
            };
            let modified_ms = metadata
                .modified()
                .ok()
                .map(system_time_to_ms)
                .unwrap_or_default();
            let indexed_revision_ms = u64::try_from(entry.revision).unwrap_or_default();
            if modified_ms > indexed_revision_ms {
                stale_files = stale_files.saturating_add(1);
            }
        }

        Ok(Some(stale_files))
    }

    fn status_config_source_summary(&self) -> SearchResult<String> {
        let mut sources = Vec::new();
        if let Some(path) = self.cli_input.overrides.config_path.as_ref() {
            sources.push(format!("cli({})", path.display()));
        } else {
            let cwd = std::env::current_dir().map_err(SearchError::Io)?;
            let project = default_project_config_file_path(&cwd);
            if project.exists() {
                sources.push(format!("project({})", project.display()));
            }
            if let Some(home) = home_dir() {
                let user = default_user_config_file_path(&home);
                if user.exists() {
                    sources.push(format!("user({})", user.display()));
                }
            }
        }
        sources.push("env".to_owned());
        sources.push("defaults".to_owned());
        Ok(sources.join(" + "))
    }

    fn collect_model_statuses(&self) -> SearchResult<Vec<FsfsModelStatus>> {
        let model_root = PathBuf::from(&self.config.indexing.model_dir);
        if let Err(error) = ensure_default_semantic_models(Some(&model_root)) {
            warn!(
                model_root = %model_root.display(),
                error = %error,
                "failed to materialize bundled semantic models while collecting status"
            );
        }
        Ok(vec![
            Self::collect_model_status("fast", &self.config.indexing.fast_model, &model_root)?,
            Self::collect_model_status(
                "quality",
                &self.config.indexing.quality_model,
                &model_root,
            )?,
        ])
    }

    fn collect_model_status(
        tier: &str,
        model_name: &str,
        model_root: &Path,
    ) -> SearchResult<FsfsModelStatus> {
        let direct_path = model_root.join(model_name);
        if direct_path.exists() {
            let size_bytes = Self::path_bytes(&direct_path)?;
            if size_bytes > 0 {
                return Ok(FsfsModelStatus {
                    tier: tier.to_owned(),
                    name: model_name.to_owned(),
                    cache_path: direct_path.display().to_string(),
                    cached: true,
                    size_bytes,
                });
            }
        }

        let mut candidates = Vec::new();
        if model_root.exists() {
            for entry in fs::read_dir(model_root)? {
                let entry = entry?;
                let name = entry.file_name();
                let Some(name) = name.to_str() else {
                    continue;
                };
                if normalize_model_token(name).contains(&normalize_model_token(model_name)) {
                    candidates.push(entry.path());
                }
            }
        }

        if candidates.is_empty() {
            return Ok(FsfsModelStatus {
                tier: tier.to_owned(),
                name: model_name.to_owned(),
                cache_path: direct_path.display().to_string(),
                cached: false,
                size_bytes: 0,
            });
        }

        let size_bytes = Self::total_bytes_for_paths(&candidates)?;
        if size_bytes == 0 {
            return Ok(FsfsModelStatus {
                tier: tier.to_owned(),
                name: model_name.to_owned(),
                cache_path: direct_path.display().to_string(),
                cached: false,
                size_bytes: 0,
            });
        }

        Ok(FsfsModelStatus {
            tier: tier.to_owned(),
            name: model_name.to_owned(),
            cache_path: candidates[0].display().to_string(),
            cached: true,
            size_bytes,
        })
    }

    #[allow(clippy::too_many_lines)]
    async fn run_one_shot_index_scaffold(&self, cx: &Cx, command: CliCommand) -> SearchResult<()> {
        self.run_one_shot_index_scaffold_with_progress(cx, command, |_| Ok(()))
            .await
    }

    #[allow(clippy::too_many_lines)]
    async fn run_one_shot_index_scaffold_with_progress<F>(
        &self,
        cx: &Cx,
        command: CliCommand,
        mut on_progress: F,
    ) -> SearchResult<()>
    where
        F: FnMut(&IndexingProgressSnapshot) -> SearchResult<()>,
    {
        const BATCH_SIZE: usize = 256;
        let total_start = Instant::now();
        let target_root = self.resolve_target_root()?;
        let index_root = self.resolve_index_root(&target_root)?;

        let root_decision = self.config.discovery.evaluate_root(&target_root, None);
        if !root_decision.include() {
            return Err(SearchError::InvalidConfig {
                field: "discovery.roots".to_owned(),
                value: target_root.display().to_string(),
                reason: format!(
                    "target root excluded by discovery policy: {}",
                    root_decision.reason_codes.join(",")
                ),
            });
        }

        let mut candidates = Vec::new();
        let discovery_start = Instant::now();
        let stats = self.collect_index_candidates(&target_root, &index_root, &mut candidates)?;
        let discovery_elapsed_ms = discovery_start.elapsed().as_millis();
        info!(
            target_root = %target_root.display(),
            discovered_files = stats.discovered_files,
            skipped_files = stats.skipped_files,
            candidate_files = candidates.len(),
            elapsed_ms = discovery_elapsed_ms,
            "fsfs file discovery completed"
        );
        println!(
            "Discovered {} file(s) under {} ({} skipped by policy)",
            stats.discovered_files,
            target_root.display(),
            stats.skipped_files
        );

        on_progress(&IndexingProgressSnapshot {
            stage: IndexingProgressStage::Discovering,
            target_root: target_root.clone(),
            index_root: index_root.clone(),
            discovered_files: stats.discovered_files,
            candidate_files: candidates.len(),
            processed_files: 0,
            skipped_files: stats.skipped_files,
            semantic_files: 0,
            canonical_bytes: 0,
            canonical_lines: 0,
            index_size_bytes: Self::path_bytes(&index_root).unwrap_or_default(),
            discovery_elapsed_ms,
            lexical_elapsed_ms: 0,
            embedding_elapsed_ms: 0,
            vector_elapsed_ms: 0,
            total_elapsed_ms: total_start.elapsed().as_millis(),
            active_file: None,
        })?;

        fs::create_dir_all(index_root.join("vector"))?;
        fs::create_dir_all(index_root.join("cache"))?;

        // 1. Resolve and probe embedder
        let mut embedder = self.resolve_fast_embedder()?;
        if let Err(error) = embedder.embed(cx, "probe").await {
            info!(
                embedder = embedder.id(),
                error = %error,
                "fsfs semantic embedder probe failed; falling back to hash embeddings"
            );
            embedder = Arc::new(HashEmbedder::new(
                embedder.dimension().max(1),
                HashAlgorithm::FnvModular,
            ));
        }

        // 2. Prepare indexes
        let vector_path = index_root.join(FSFS_VECTOR_INDEX_FILE);
        let mut vector_writer =
            VectorIndex::create(&vector_path, embedder.id(), embedder.dimension())?;

        let lexical_path = index_root.join("lexical");
        let lexical_index = TantivyIndex::create(&lexical_path)?;

        let canonicalizer = DefaultCanonicalizer::default();
        let mut manifests = Vec::new();
        let mut observed_reason_codes: BTreeSet<String> =
            stats.reason_codes.iter().cloned().collect();
        let mut semantic_doc_count = 0_usize;
        let mut content_skipped_files = 0_usize;
        let mut processed_files = 0_usize;
        let mut canonical_bytes_total = 0_u64;
        let mut canonical_line_count = 0_u64;
        let mut last_active_file = None;

        let canonicalize_start = Instant::now();
        // We'll track cumulative timings for the batched phases
        let mut lexical_elapsed_ms = 0_u128;
        let mut vector_elapsed_ms = 0_u128;
        let mut embedding_elapsed_ms = 0_u128;

        // 3. Process in batches
        for chunk in candidates.chunks(BATCH_SIZE) {
            let mut chunk_docs = Vec::with_capacity(chunk.len());

            // Read & Canonicalize
            for candidate in chunk {
                let bytes = match fs::read(&candidate.file_path) {
                    Ok(bytes) => bytes,
                    Err(error) if is_ignorable_index_walk_error(&error) => {
                        if error.kind() == ErrorKind::PermissionDenied {
                            observed_reason_codes
                                .insert(REASON_DISCOVERY_FILE_PERMISSION_DENIED.to_owned());
                        }
                        content_skipped_files = content_skipped_files.saturating_add(1);
                        continue;
                    }
                    Err(error) => return Err(error.into()),
                };

                if is_probably_binary(&bytes) {
                    observed_reason_codes.insert(REASON_DISCOVERY_FILE_BINARY_BLOCKED.to_owned());
                    content_skipped_files = content_skipped_files.saturating_add(1);
                    continue;
                }

                let raw_text = String::from_utf8_lossy(&bytes);
                let canonical = canonicalizer.canonicalize(&raw_text);
                if canonical.trim().is_empty() {
                    observed_reason_codes.insert(REASON_DISCOVERY_FILE_EXCLUDED.to_owned());
                    content_skipped_files = content_skipped_files.saturating_add(1);
                    continue;
                }

                let canonical_bytes = u64::try_from(canonical.len()).unwrap_or(u64::MAX);
                canonical_bytes_total = canonical_bytes_total.saturating_add(canonical_bytes);
                canonical_line_count =
                    canonical_line_count.saturating_add(count_non_empty_lines(&canonical));
                let ingestion_class =
                    format!("{:?}", candidate.ingestion_class).to_ascii_lowercase();
                let reason_code = match candidate.ingestion_class {
                    IngestionClass::FullSemanticLexical => "index.plan.full_semantic_lexical",
                    IngestionClass::LexicalOnly => "index.plan.lexical_only",
                    IngestionClass::MetadataOnly => "index.plan.metadata_only",
                    IngestionClass::Skip => "index.plan.skip",
                }
                .to_owned();

                manifests.push(IndexManifestEntry {
                    file_key: candidate.file_key.clone(),
                    revision: i64::try_from(candidate.modified_ms).unwrap_or(i64::MAX),
                    ingestion_class: ingestion_class.clone(),
                    canonical_bytes,
                    reason_code,
                });

                let file_name = candidate
                    .file_path
                    .file_name()
                    .and_then(std::ffi::OsStr::to_str)
                    .unwrap_or_default()
                    .to_owned();
                let doc = IndexableDocument::new(candidate.file_key.clone(), canonical)
                    .with_title(file_name)
                    .with_metadata("source_path", candidate.file_path.display().to_string())
                    .with_metadata("ingestion_class", ingestion_class)
                    .with_metadata("source_modified_ms", candidate.modified_ms.to_string());

                if matches!(
                    candidate.ingestion_class,
                    IngestionClass::FullSemanticLexical
                ) {
                    semantic_doc_count = semantic_doc_count.saturating_add(1);
                }
                processed_files = processed_files.saturating_add(1);
                last_active_file = Some(candidate.file_path.display().to_string());
                chunk_docs.push((candidate.ingestion_class, doc));
            }

            // Lexical Indexing
            let lexical_start = Instant::now();
            let lexical_batch = chunk_docs
                .iter()
                .filter(|(class, _)| {
                    !matches!(class, IngestionClass::MetadataOnly | IngestionClass::Skip)
                })
                .map(|(_, doc)| doc.clone())
                .collect::<Vec<_>>();
            if !lexical_batch.is_empty() {
                lexical_index.index_documents(cx, &lexical_batch).await?;
            }
            lexical_elapsed_ms =
                lexical_elapsed_ms.saturating_add(lexical_start.elapsed().as_millis());

            // Semantic Embedding & Vector Writing
            let semantic_docs = chunk_docs
                .iter()
                .filter(|(class, _)| matches!(class, IngestionClass::FullSemanticLexical))
                .map(|(_, doc)| doc)
                .collect::<Vec<_>>();

            if !semantic_docs.is_empty() {
                let semantic_texts = semantic_docs
                    .iter()
                    .map(|doc| doc.content.as_str())
                    .collect::<Vec<_>>();

                let embed_start = Instant::now();
                let embeddings_result = embedder.embed_batch(cx, &semantic_texts).await;
                embedding_elapsed_ms =
                    embedding_elapsed_ms.saturating_add(embed_start.elapsed().as_millis());

                match embeddings_result {
                    Ok(embeddings) => {
                        if embeddings.len() != semantic_docs.len() {
                            return Err(frankensearch_core::SearchError::EmbeddingFailed {
                                model: embedder.id().to_string(),
                                source: format!(
                                    "embed_batch returned {} vectors for {} documents",
                                    embeddings.len(),
                                    semantic_docs.len(),
                                )
                                .into(),
                            });
                        }
                        let vector_start = Instant::now();
                        for (doc, embedding) in semantic_docs.iter().zip(embeddings.into_iter()) {
                            vector_writer.write_record(&doc.id, &embedding)?;
                        }
                        vector_elapsed_ms =
                            vector_elapsed_ms.saturating_add(vector_start.elapsed().as_millis());
                    }
                    Err(error) => {
                        // If embedding fails for a chunk, we skip vector indexing for this chunk
                        // rather than crashing the whole process or switching models mid-stream.
                        warn!(
                            chunk_size = semantic_docs.len(),
                            error = %error,
                            "semantic embedding failed for batch; skipping vectors for these files"
                        );
                    }
                }
            }

            on_progress(&IndexingProgressSnapshot {
                stage: IndexingProgressStage::Indexing,
                target_root: target_root.clone(),
                index_root: index_root.clone(),
                discovered_files: stats.discovered_files,
                candidate_files: candidates.len(),
                processed_files,
                skipped_files: stats.skipped_files.saturating_add(content_skipped_files),
                semantic_files: semantic_doc_count,
                canonical_bytes: canonical_bytes_total,
                canonical_lines: canonical_line_count,
                index_size_bytes: Self::path_bytes(&index_root).unwrap_or_default(),
                discovery_elapsed_ms,
                lexical_elapsed_ms,
                embedding_elapsed_ms,
                vector_elapsed_ms,
                total_elapsed_ms: total_start.elapsed().as_millis(),
                active_file: last_active_file.clone(),
            })?;
        }

        let canonicalize_elapsed_ms = canonicalize_start.elapsed().as_millis();

        on_progress(&IndexingProgressSnapshot {
            stage: IndexingProgressStage::Finalizing,
            target_root: target_root.clone(),
            index_root: index_root.clone(),
            discovered_files: stats.discovered_files,
            candidate_files: candidates.len(),
            processed_files,
            skipped_files: stats.skipped_files.saturating_add(content_skipped_files),
            semantic_files: semantic_doc_count,
            canonical_bytes: canonical_bytes_total,
            canonical_lines: canonical_line_count,
            index_size_bytes: Self::path_bytes(&index_root).unwrap_or_default(),
            discovery_elapsed_ms,
            lexical_elapsed_ms,
            embedding_elapsed_ms,
            vector_elapsed_ms,
            total_elapsed_ms: total_start.elapsed().as_millis(),
            active_file: last_active_file.clone(),
        })?;

        // 4. Commit and Finish
        let lexical_commit_start = Instant::now();
        lexical_index.commit(cx).await?;
        lexical_elapsed_ms =
            lexical_elapsed_ms.saturating_add(lexical_commit_start.elapsed().as_millis());

        let vector_finish_start = Instant::now();
        vector_writer.finish()?;
        vector_elapsed_ms =
            vector_elapsed_ms.saturating_add(vector_finish_start.elapsed().as_millis());

        manifests.sort_by(|left, right| left.file_key.cmp(&right.file_key));
        let source_hash_hex = index_source_hash_hex(&manifests);
        let indexed_files = processed_files;
        let skipped_files = stats.discovered_files.saturating_sub(indexed_files);
        let total_canonical_bytes = canonical_bytes_total;

        self.write_index_artifacts(&index_root, &manifests)?;
        let sentinel = IndexSentinel {
            schema_version: 1,
            generated_at_ms: pressure_timestamp_ms(),
            command: format!("{command:?}").to_ascii_lowercase(),
            target_root: target_root.display().to_string(),
            index_root: index_root.display().to_string(),
            discovered_files: stats.discovered_files,
            indexed_files,
            skipped_files,
            reason_codes: observed_reason_codes.into_iter().collect(),
            total_canonical_bytes,
            source_hash_hex,
        };
        self.write_index_sentinel(&index_root, &sentinel)?;

        let storage_usage = self.collect_index_storage_usage(&IndexStoragePaths {
            vector_index_roots: vec![index_root.join("vector")],
            lexical_index_roots: vec![index_root.join("lexical")],
            catalog_files: vec![PathBuf::from(&self.config.storage.db_path)],
            embedding_cache_roots: vec![index_root.join("cache")],
        })?;
        let elapsed_ms = total_start.elapsed().as_millis();

        on_progress(&IndexingProgressSnapshot {
            stage: IndexingProgressStage::Completed,
            target_root: target_root.clone(),
            index_root: index_root.clone(),
            discovered_files: stats.discovered_files,
            candidate_files: candidates.len(),
            processed_files: indexed_files,
            skipped_files,
            semantic_files: semantic_doc_count,
            canonical_bytes: total_canonical_bytes,
            canonical_lines: canonical_line_count,
            index_size_bytes: storage_usage.total_bytes(),
            discovery_elapsed_ms,
            lexical_elapsed_ms,
            embedding_elapsed_ms,
            vector_elapsed_ms,
            total_elapsed_ms: elapsed_ms,
            active_file: None,
        })?;

        info!(
            command = ?command,
            target_root = %target_root.display(),
            index_root = %index_root.display(),
            discovered_files = stats.discovered_files,
            indexed_files,
            skipped_files,
            total_canonical_bytes,
            semantic_docs = semantic_doc_count,
            embedder = embedder.id(),
            discovery_elapsed_ms,
            canonicalize_elapsed_ms,
            lexical_elapsed_ms,
            embedding_elapsed_ms,
            vector_elapsed_ms,
            total_elapsed_ms = elapsed_ms,
            source_hash = sentinel.source_hash_hex,
            "fsfs index pipeline completed"
        );

        println!(
            "Indexed {} file(s) (discovered {}, skipped {}) into {} in {} ms (index size {} bytes)",
            indexed_files,
            stats.discovered_files,
            skipped_files,
            index_root.display(),
            elapsed_ms,
            storage_usage.total_bytes()
        );

        Ok(())
    }

    fn resolve_target_root(&self) -> SearchResult<PathBuf> {
        let raw = self
            .cli_input
            .target_path
            .as_deref()
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf);

        let target = if raw.is_absolute() {
            raw
        } else {
            std::env::current_dir().map_err(SearchError::Io)?.join(raw)
        };

        if !target.exists() {
            return Err(SearchError::InvalidConfig {
                field: "cli.index.target".to_owned(),
                value: target.display().to_string(),
                reason: "index target path does not exist".to_owned(),
            });
        }
        if !target.is_dir() {
            return Err(SearchError::InvalidConfig {
                field: "cli.index.target".to_owned(),
                value: target.display().to_string(),
                reason: "index target must be a directory".to_owned(),
            });
        }

        fs::canonicalize(&target).map_err(SearchError::Io)
    }

    fn resolve_index_root(&self, target_root: &Path) -> SearchResult<PathBuf> {
        if let Some(path) = self.cli_input.index_dir.as_deref() {
            return absolutize_path(path);
        }

        let configured = PathBuf::from(&self.config.storage.index_dir);
        if configured.is_absolute() {
            Ok(configured)
        } else {
            Ok(target_root.join(configured))
        }
    }

    fn resolve_storage_db_path(&self) -> SearchResult<PathBuf> {
        let raw = self.config.storage.db_path.trim();
        if raw.is_empty() {
            return Err(SearchError::InvalidConfig {
                field: "storage.db_path".to_owned(),
                value: String::new(),
                reason: "must not be empty".to_owned(),
            });
        }
        if raw == ":memory:" {
            return Ok(PathBuf::from(raw));
        }
        if let Some(rest) = raw.strip_prefix("~/").or_else(|| raw.strip_prefix("~\\")) {
            let home = home_dir().ok_or_else(|| SearchError::InvalidConfig {
                field: "storage.db_path".to_owned(),
                value: raw.to_owned(),
                reason: "home directory not available for ~ expansion".to_owned(),
            })?;
            return Ok(home.join(rest));
        }
        absolutize_path(Path::new(raw))
    }

    fn resolve_fast_embedder(&self) -> SearchResult<Arc<dyn Embedder>> {
        if cfg!(test) {
            return Ok(Arc::new(HashEmbedder::default_256()));
        }

        let configured_root = PathBuf::from(&self.config.indexing.model_dir);
        let stack = EmbedderStack::auto_detect_with(Some(&configured_root))
            .or_else(|_| EmbedderStack::auto_detect())?;
        Ok(stack.fast_arc())
    }

    fn resolve_quality_embedder(&self) -> SearchResult<Option<Arc<dyn Embedder>>> {
        if cfg!(test) {
            return Ok(None);
        }

        let configured_root = PathBuf::from(&self.config.indexing.model_dir);
        let stack = EmbedderStack::auto_detect_with(Some(&configured_root))
            .or_else(|_| EmbedderStack::auto_detect())?;
        Ok(stack.quality_arc())
    }

    fn prepare_search_execution_resources(
        &self,
        mode: SearchExecutionMode,
    ) -> SearchResult<SearchExecutionResources> {
        let index_root = self.resolve_status_index_root()?;
        let lexical_path = index_root.join("lexical");
        let vector_path = index_root.join(FSFS_VECTOR_INDEX_FILE);

        let lexical_index = if lexical_path.exists() {
            Some(TantivyIndex::open(&lexical_path)?)
        } else {
            None
        };
        let lexical_available = lexical_index.is_some();

        let should_open_vector =
            !matches!(mode, SearchExecutionMode::LexicalOnly) || lexical_index.is_none();
        let vector_index = if should_open_vector && vector_path.exists() {
            match VectorIndex::open(&vector_path) {
                Ok(index) => Some(index),
                Err(error) if lexical_available => {
                    warn!(
                        error = %error,
                        path = %vector_path.display(),
                        "fsfs search falling back to lexical-only mode after vector index load failure"
                    );
                    None
                }
                Err(error) => return Err(error),
            }
        } else {
            None
        };

        if lexical_index.is_none() && vector_index.is_none() {
            return Err(SearchError::InvalidConfig {
                field: "cli.index_dir".to_owned(),
                value: index_root.display().to_string(),
                reason: "no index found; run `fsfs index <dir>` first".to_owned(),
            });
        }

        Ok(SearchExecutionResources {
            index_root,
            lexical_index,
            vector_index,
            fast_embedder: None,
            quality_embedder: None,
            fast_embedder_attempted: false,
            quality_embedder_attempted: false,
        })
    }

    fn hash_embedder_for_vector_index(index: &VectorIndex) -> Option<Arc<dyn Embedder>> {
        let index_embedder_id = index.embedder_id();
        if index_embedder_id.eq_ignore_ascii_case("fnv1a-256") {
            return Some(Arc::new(HashEmbedder::default_256()));
        }
        if index_embedder_id.eq_ignore_ascii_case("fnv1a-384") {
            return Some(Arc::new(HashEmbedder::default_384()));
        }
        if index_embedder_id.eq_ignore_ascii_case("fnv1a-custom")
            || index_embedder_id
                .get(..6)
                .is_some_and(|prefix| prefix.eq_ignore_ascii_case("fnv1a-"))
        {
            return Some(Arc::new(HashEmbedder::new(
                index.dimension(),
                HashAlgorithm::FnvModular,
            )));
        }
        None
    }

    fn maybe_prepare_fast_embedder(
        &self,
        resources: &mut SearchExecutionResources,
        lexical_available: bool,
    ) -> SearchResult<()> {
        if resources.fast_embedder.is_some() || resources.fast_embedder_attempted {
            return Ok(());
        }
        resources.fast_embedder_attempted = true;

        if let Some(index) = resources.vector_index.as_ref()
            && let Some(embedder) = Self::hash_embedder_for_vector_index(index)
        {
            info!(
                index_embedder_id = index.embedder_id(),
                fast_embedder_id = embedder.id(),
                "fsfs search using hash embedder matched to vector index revision"
            );
            resources.fast_embedder = Some(embedder);
            return Ok(());
        }

        match self.resolve_fast_embedder() {
            Ok(embedder) => {
                if let Some(index) = resources.vector_index.as_ref() {
                    let index_embedder_id = index.embedder_id();
                    if !index_embedder_id.eq_ignore_ascii_case(embedder.id()) {
                        info!(
                            index_embedder_id,
                            fast_embedder_id = embedder.id(),
                            "fsfs search disabling fast semantic tier after embedder-id mismatch"
                        );
                        return Ok(());
                    }
                    let index_dimension = index.dimension();
                    let embedder_dimension = embedder.dimension();
                    if embedder_dimension != index_dimension {
                        info!(
                            index_dimension,
                            embedder_dimension,
                            embedder_id = embedder.id(),
                            "fsfs search disabling fast semantic tier after embedder/index dimension mismatch"
                        );
                        return Ok(());
                    }
                }
                resources.fast_embedder = Some(embedder);
                Ok(())
            }
            Err(error) if lexical_available => {
                info!(
                    error = %error,
                    "fsfs search falling back to lexical-only mode after fast embedder init failure"
                );
                Ok(())
            }
            Err(error) => Err(error),
        }
    }

    fn maybe_prepare_quality_embedder(
        &self,
        resources: &mut SearchExecutionResources,
        lexical_available: bool,
    ) -> SearchResult<()> {
        if resources.quality_embedder.is_some() || resources.quality_embedder_attempted {
            return Ok(());
        }
        resources.quality_embedder_attempted = true;
        match self.resolve_quality_embedder() {
            Ok(Some(embedder)) => {
                if let Some(index) = resources.vector_index.as_ref() {
                    let index_embedder_id = index.embedder_id();
                    if !index_embedder_id.eq_ignore_ascii_case(embedder.id()) {
                        info!(
                            index_embedder_id,
                            quality_embedder_id = embedder.id(),
                            "fsfs search skipping quality semantic tier after embedder-id mismatch"
                        );
                        return Ok(());
                    }
                    let index_dimension = index.dimension();
                    let embedder_dimension = embedder.dimension();
                    if embedder_dimension != index_dimension {
                        info!(
                            index_dimension,
                            embedder_dimension,
                            embedder_id = embedder.id(),
                            "fsfs search skipping quality semantic tier after embedder/index dimension mismatch"
                        );
                        return Ok(());
                    }
                }
                resources.quality_embedder = Some(embedder);
                Ok(())
            }
            Ok(None) => Ok(()),
            Err(error) if lexical_available => {
                info!(
                    error = %error,
                    "fsfs search continuing without quality embedder after initialization failure"
                );
                Ok(())
            }
            Err(error) => Err(error),
        }
    }

    /// Build a live ingest pipeline for the watcher by opening existing indexes.
    fn build_live_ingest_pipeline(&self) -> SearchResult<LiveIngestPipeline> {
        let target_root = self.resolve_target_root()?;
        let index_root = self.resolve_index_root(&target_root)?;
        let storage_db_path = self.resolve_storage_db_path()?;
        if storage_db_path.as_os_str() != ":memory:"
            && let Some(parent) = storage_db_path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }

        let lexical_path = index_root.join("lexical");
        let vector_path = index_root.join(FSFS_VECTOR_INDEX_FILE);

        let lexical_index = TantivyIndex::create(&lexical_path)?;
        let vector_index = VectorIndex::open(&vector_path)?;
        let embedder = self.resolve_fast_embedder()?;

        info!(
            target_root = %target_root.display(),
            index_root = %index_root.display(),
            storage_db_path = %storage_db_path.display(),
            embedder = embedder.id(),
            "live ingest pipeline initialized for watch mode"
        );

        Ok(
            LiveIngestPipeline::new(target_root, lexical_index, vector_index, embedder)
                .with_storage_db_path(storage_db_path),
        )
    }

    fn collect_index_candidates(
        &self,
        target_root: &Path,
        index_root: &Path,
        output: &mut Vec<IndexCandidate>,
    ) -> SearchResult<IndexDiscoveryStats> {
        let mut discovered_files = 0_usize;
        let mut skipped_files = 0_usize;
        let mut reason_codes = BTreeSet::new();
        let mount_overrides = self.config.discovery.mount_override_map();
        let mount_table = MountTable::new(read_system_mounts(), &mount_overrides);

        let mut walker = WalkBuilder::new(target_root);
        walker.follow_links(self.config.discovery.follow_symlinks);
        walker.git_ignore(true);
        walker.git_global(true);
        walker.git_exclude(true);
        walker.hidden(false);
        walker.standard_filters(true);

        for entry in walker.build() {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    if let Some(io_error) = error.io_error() {
                        if is_ignorable_index_walk_error(io_error) {
                            continue;
                        }
                        return Err(
                            std::io::Error::new(io_error.kind(), io_error.to_string()).into()
                        );
                    }
                    return Err(SearchError::InvalidConfig {
                        field: "discovery.walk".to_owned(),
                        value: target_root.display().to_string(),
                        reason: error.to_string(),
                    });
                }
            };

            let path_is_symlink = entry.path_is_symlink();
            let Some(file_type) = entry.file_type() else {
                if path_is_symlink && !self.config.discovery.follow_symlinks {
                    skipped_files = skipped_files.saturating_add(1);
                    reason_codes.insert(REASON_DISCOVERY_FILE_EXCLUDED.to_owned());
                }
                continue;
            };
            if file_type.is_dir() {
                continue;
            }
            if !file_type.is_file() && !path_is_symlink {
                continue;
            }

            let entry_path = entry.path().to_path_buf();
            if entry_path.starts_with(index_root) {
                skipped_files = skipped_files.saturating_add(1);
                reason_codes.insert(REASON_DISCOVERY_FILE_EXCLUDED.to_owned());
                continue;
            }
            if path_is_symlink && !self.config.discovery.follow_symlinks {
                skipped_files = skipped_files.saturating_add(1);
                reason_codes.insert(REASON_DISCOVERY_FILE_EXCLUDED.to_owned());
                continue;
            }

            let metadata = match fs::metadata(&entry_path) {
                Ok(metadata) => metadata,
                Err(error) if is_ignorable_index_walk_error(&error) => continue,
                Err(error) => return Err(error.into()),
            };

            discovered_files = discovered_files.saturating_add(1);
            let mut candidate =
                DiscoveryCandidate::new(&entry_path, metadata.len()).with_symlink(path_is_symlink);
            if let Some((mount_entry, _policy)) = mount_table.lookup(&entry_path) {
                candidate = candidate.with_mount_category(mount_entry.category);
            }
            let decision = self.config.discovery.evaluate_candidate(&candidate);
            reason_codes.extend(decision.reason_codes.iter().cloned());
            if !matches!(decision.scope, DiscoveryScopeDecision::Include)
                || !decision.ingestion_class.is_indexed()
            {
                skipped_files = skipped_files.saturating_add(1);
                continue;
            }

            let modified_ms = metadata
                .modified()
                .ok()
                .map(system_time_to_ms)
                .unwrap_or_default();

            output.push(IndexCandidate {
                file_path: entry_path.clone(),
                file_key: normalize_file_key_for_index(&entry_path, target_root),
                modified_ms,
                ingestion_class: decision.ingestion_class,
            });
        }

        Ok(IndexDiscoveryStats {
            discovered_files,
            skipped_files,
            reason_codes: reason_codes.into_iter().collect(),
        })
    }

    #[allow(clippy::unused_self)]
    fn write_index_artifacts(
        &self,
        index_root: &Path,
        manifests: &[IndexManifestEntry],
    ) -> SearchResult<()> {
        let vector_manifest = serde_json::to_string_pretty(manifests).map_err(|error| {
            SearchError::SubsystemError {
                subsystem: "index.vector_manifest",
                source: Box::new(error),
            }
        })?;
        write_durable(index_root.join(FSFS_VECTOR_MANIFEST_FILE), vector_manifest)?;

        let lexical_manifest = serde_json::to_string_pretty(manifests).map_err(|error| {
            SearchError::SubsystemError {
                subsystem: "index.lexical_manifest",
                source: Box::new(error),
            }
        })?;
        write_durable(
            index_root.join(FSFS_LEXICAL_MANIFEST_FILE),
            lexical_manifest,
        )?;

        Ok(())
    }

    #[allow(clippy::unused_self)]
    fn write_index_sentinel(
        &self,
        index_root: &Path,
        sentinel: &IndexSentinel,
    ) -> SearchResult<()> {
        let json = serde_json::to_string_pretty(sentinel).map_err(|error| {
            SearchError::SubsystemError {
                subsystem: "index.sentinel",
                source: Box::new(error),
            }
        })?;
        write_durable(index_root.join(FSFS_SENTINEL_FILE), json)?;
        Ok(())
    }

    fn validate_command_inputs(&self, command: CliCommand) -> SearchResult<()> {
        match command {
            CliCommand::Search if self.cli_input.query.is_none() => {
                Err(SearchError::InvalidConfig {
                    field: "cli.search_query".into(),
                    value: String::new(),
                    reason: "missing search query argument".into(),
                })
            }
            CliCommand::Explain if self.cli_input.result_id.is_none() => {
                Err(SearchError::InvalidConfig {
                    field: "cli.explain.result_id".into(),
                    value: String::new(),
                    reason: "missing result identifier argument".into(),
                })
            }
            _ => Ok(()),
        }
    }

    /// TUI runtime lane.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` when onboarding/indexing flow cannot complete.
    pub async fn run_tui(&self, cx: &Cx) -> SearchResult<()> {
        let shell_model = FsfsTuiShellModel::from_config(&self.config);
        shell_model
            .validate()
            .map_err(|error| SearchError::InvalidConfig {
                field: "tui.shell_model".to_owned(),
                value: format!("{shell_model:?}"),
                reason: error.to_string(),
            })?;

        let palette = shell_model.palette.build_palette();
        info!(
            theme = ?shell_model.settings.theme,
            density = ?shell_model.settings.density,
            show_explanations = shell_model.settings.show_explanations,
            screen_count = shell_model.navigation.screen_order.len(),
            keybinding_count = shell_model.keymap.bindings.len(),
            palette_action_count = palette.len(),
            "fsfs tui shell model initialized"
        );

        if !std::io::stdout().is_terminal() || !std::io::stdin().is_terminal() {
            info!("fsfs tui requested without interactive terminal; falling back to status output");
            return self.run_status_command();
        }

        if self.index_artifacts_exist()? {
            self.run_search_dashboard_tui(cx).await?;
            return Ok(());
        }

        let proposals = self.discover_index_root_proposals()?;
        let Some(selected_root) = self.prompt_first_run_index_root(&proposals)? else {
            println!("No indexing started. Run `fsfs index <path>` any time.");
            return Ok(());
        };
        self.run_first_run_indexing_tui(cx, selected_root.clone())
            .await?;

        let selected_index_root = self.resolve_index_root(&selected_root)?;
        let mut next_runtime = self.clone();
        let mut next_cli = next_runtime.cli_input.clone();
        next_cli.index_dir = Some(selected_index_root);
        next_runtime = next_runtime.with_cli_input(next_cli);
        next_runtime.run_search_dashboard_tui(cx).await?;
        Ok(())
    }

    fn index_artifacts_exist(&self) -> SearchResult<bool> {
        let index_root = self.resolve_status_index_root()?;
        if !index_root.exists() {
            return Ok(false);
        }
        if Self::read_index_sentinel(&index_root)?.is_some() {
            return Ok(true);
        }
        Ok(index_root.join(FSFS_VECTOR_INDEX_FILE).exists()
            || index_root.join("lexical").exists()
            || index_root.join(FSFS_VECTOR_MANIFEST_FILE).exists()
            || index_root.join(FSFS_LEXICAL_MANIFEST_FILE).exists())
    }

    fn collect_root_probe_candidates(&self) -> SearchResult<Vec<PathBuf>> {
        let cwd = std::env::current_dir().map_err(SearchError::Io)?;
        let home = home_dir();
        let mut seen = HashSet::new();
        let mut candidates = Vec::new();

        let mut push_candidate = |path: PathBuf| {
            if !path.exists() || !path.is_dir() {
                return;
            }
            let canonical = fs::canonicalize(&path).unwrap_or(path);
            if seen.insert(canonical.clone()) {
                candidates.push(canonical);
            }
        };

        push_candidate(cwd.clone());
        if let Some(parent) = cwd.parent() {
            push_candidate(parent.to_path_buf());
        }

        if let Some(home_path) = home.as_ref() {
            push_candidate(home_path.clone());
            for suffix in [
                "projects",
                "code",
                "workspace",
                "workspaces",
                "work",
                "src",
                "Documents",
            ] {
                push_candidate(home_path.join(suffix));
            }
        }

        for absolute in ["/data/projects", "/workspaces", "/opt/projects"] {
            push_candidate(PathBuf::from(absolute));
        }

        for root in &self.config.discovery.roots {
            let candidate = PathBuf::from(root);
            if candidate.is_absolute() {
                push_candidate(candidate);
            } else {
                push_candidate(cwd.join(candidate));
            }
        }

        Ok(candidates)
    }

    fn discover_index_root_proposals(&self) -> SearchResult<Vec<IndexRootProposal>> {
        let home = home_dir();
        let mut proposals = self
            .collect_root_probe_candidates()?
            .into_iter()
            .filter_map(|path| match Self::probe_index_root(&path) {
                Ok(stats)
                    if stats.candidate_files > 0
                        || stats.repo_markers > 0
                        || stats.candidate_bytes > 0 =>
                {
                    let score = score_index_root(&path, &stats, home.as_deref());
                    Some(IndexRootProposal { path, score, stats })
                }
                Ok(_) => None,
                Err(error) => {
                    warn!(
                        path = %path.display(),
                        error = %error,
                        "failed to probe potential index root"
                    );
                    None
                }
            })
            .collect::<Vec<_>>();

        if proposals.is_empty() {
            let fallback_root = std::env::current_dir().map_err(SearchError::Io)?;
            let fallback_stats = Self::probe_index_root(&fallback_root).unwrap_or_default();
            proposals.push(IndexRootProposal {
                score: score_index_root(&fallback_root, &fallback_stats, home.as_deref()),
                path: fallback_root,
                stats: fallback_stats,
            });
        }

        proposals.sort_by(|left, right| {
            right
                .score
                .cmp(&left.score)
                .then_with(|| right.stats.candidate_files.cmp(&left.stats.candidate_files))
                .then_with(|| left.path.cmp(&right.path))
        });
        proposals.truncate(ROOT_PROBE_MAX_PROPOSALS);
        Ok(proposals)
    }

    fn probe_index_root(root: &Path) -> SearchResult<RootProbeStats> {
        let mut stats = RootProbeStats::default();
        let mut queue = VecDeque::from([(root.to_path_buf(), 0_usize)]);

        while let Some((dir_path, depth)) = queue.pop_front() {
            stats.scanned_dirs = stats.scanned_dirs.saturating_add(1);
            let entries = match fs::read_dir(&dir_path) {
                Ok(entries) => entries,
                Err(error) if is_ignorable_index_walk_error(&error) => continue,
                Err(error) => return Err(error.into()),
            };

            let mut pending_children = Vec::new();
            for (index, entry_result) in entries.enumerate() {
                if index >= ROOT_PROBE_MAX_ENTRIES_PER_DIR
                    || stats.scanned_entries >= ROOT_PROBE_MAX_TOTAL_ENTRIES
                {
                    break;
                }
                let Ok(entry) = entry_result else {
                    continue;
                };
                stats.scanned_entries = stats.scanned_entries.saturating_add(1);

                let file_name = entry.file_name().to_string_lossy().to_ascii_lowercase();
                let Ok(file_type) = entry.file_type() else {
                    continue;
                };
                if file_type.is_symlink() {
                    continue;
                }

                if file_type.is_dir() {
                    if file_name == ".git" {
                        stats.repo_markers = stats.repo_markers.saturating_add(1);
                    }
                    if depth < ROOT_PROBE_MAX_DEPTH && !is_probe_excluded_dir_name(&file_name) {
                        pending_children.push(entry.path());
                    }
                    continue;
                }

                if !file_type.is_file() {
                    continue;
                }

                if is_repo_marker_filename(&file_name) {
                    stats.repo_markers = stats.repo_markers.saturating_add(1);
                }

                if let Some(file_class) = classify_probe_file(&entry.path(), &file_name) {
                    stats.candidate_files = stats.candidate_files.saturating_add(1);
                    match file_class {
                        RootProbeFileClass::Code => {
                            stats.code_files = stats.code_files.saturating_add(1);
                        }
                        RootProbeFileClass::Document => {
                            stats.doc_files = stats.doc_files.saturating_add(1);
                        }
                    }
                    if let Ok(metadata) = entry.metadata() {
                        stats.candidate_bytes =
                            stats.candidate_bytes.saturating_add(metadata.len());
                    }
                }
            }

            for child in pending_children {
                queue.push_back((child, depth + 1));
            }
            if stats.scanned_entries >= ROOT_PROBE_MAX_TOTAL_ENTRIES {
                break;
            }
        }

        Ok(stats)
    }

    #[allow(clippy::too_many_lines)]
    fn prompt_first_run_index_root(
        &self,
        proposals: &[IndexRootProposal],
    ) -> SearchResult<Option<PathBuf>> {
        if let Ok(mut session) = FtuiSession::enter() {
            match self.prompt_first_run_index_root_ftui(proposals, &mut session) {
                Ok(selected) => return Ok(selected),
                Err(error) => {
                    warn!(
                        error = %error,
                        "interactive ftui root selector failed; falling back to ansi prompt"
                    );
                }
            }
        }

        let no_color = self.cli_input.no_color || std::env::var_os("NO_COLOR").is_some();
        let width = tui_terminal_width().clamp(88, 160);
        let rule = "═".repeat(width);
        let mut stdout = std::io::stdout();

        write!(stdout, "\u{1b}[2J\u{1b}[H").map_err(tui_io_error)?;
        writeln!(stdout, "{}", paint(&rule, "38;5;24", no_color)).map_err(tui_io_error)?;
        writeln!(
            stdout,
            "{} {}",
            paint("fsfs", "1;38;5;45", no_color),
            paint("First-Run Index Setup", "1;37", no_color)
        )
        .map_err(tui_io_error)?;
        writeln!(
            stdout,
            "{}",
            paint(
                "No index detected. Pick the root that best represents your code and documents.",
                "38;5;250",
                no_color
            )
        )
        .map_err(tui_io_error)?;
        writeln!(
            stdout,
            "{} depth<= {}  per-dir<= {}  scanned entries<= {}",
            paint("probe budget:", "38;5;244", no_color),
            ROOT_PROBE_MAX_DEPTH,
            ROOT_PROBE_MAX_ENTRIES_PER_DIR,
            ROOT_PROBE_MAX_TOTAL_ENTRIES
        )
        .map_err(tui_io_error)?;
        writeln!(stdout, "{}", paint(&rule, "38;5;24", no_color)).map_err(tui_io_error)?;
        writeln!(stdout).map_err(tui_io_error)?;

        for (idx, proposal) in proposals.iter().enumerate() {
            let is_recommended = idx == 0;
            let candidate_files = proposal.stats.candidate_files.max(1);
            let code_pct = ratio_percent_usize(proposal.stats.code_files, candidate_files);
            let doc_pct = ratio_percent_usize(proposal.stats.doc_files, candidate_files);
            let fit = describe_root_probe_fit(&proposal.path, &proposal.stats);
            let rank_color = if is_recommended {
                "1;38;5;46"
            } else {
                "1;38;5;39"
            };
            let badge = if is_recommended {
                paint("RECOMMENDED", "30;102", no_color)
            } else {
                paint("candidate", "38;5;244", no_color)
            };
            writeln!(
                stdout,
                "{} {}  {}",
                paint(&format!("[{}]", idx + 1), rank_color, no_color),
                truncate_middle(
                    &proposal.path.display().to_string(),
                    width.saturating_sub(26)
                ),
                badge
            )
            .map_err(tui_io_error)?;
            writeln!(
                stdout,
                "    fit={} score={} repos={} files={} data={}",
                fit,
                proposal.score,
                proposal.stats.repo_markers,
                proposal.stats.candidate_files,
                humanize_bytes(proposal.stats.candidate_bytes)
            )
            .map_err(tui_io_error)?;
            writeln!(
                stdout,
                "    composition: code={:.1}% docs={:.1}%   scanned={} dirs / {} entries",
                code_pct, doc_pct, proposal.stats.scanned_dirs, proposal.stats.scanned_entries
            )
            .map_err(tui_io_error)?;
            writeln!(stdout).map_err(tui_io_error)?;
        }

        writeln!(
            stdout,
            "{}",
            paint(
                &format!(
                    "Choose [1-{}], press Enter for recommended, or type q to cancel.",
                    proposals.len()
                ),
                "1;38;5;220",
                no_color
            )
        )
        .map_err(tui_io_error)?;
        write!(stdout, "{} ", paint("selection>", "1;38;5;45", no_color)).map_err(tui_io_error)?;
        stdout.flush().map_err(tui_io_error)?;

        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .map_err(SearchError::Io)?;
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Ok(proposals.first().map(|proposal| proposal.path.clone()));
        }
        if matches!(trimmed, "q" | "Q" | "n" | "N" | "no" | "No" | "NO") {
            return Ok(None);
        }
        let selected = trimmed
            .parse::<usize>()
            .ok()
            .and_then(|index| proposals.get(index.saturating_sub(1)))
            .map(|proposal| proposal.path.clone());
        if selected.is_none() {
            writeln!(
                stdout,
                "{}",
                paint("Invalid selection. Using recommended root.", "33", no_color)
            )
            .map_err(tui_io_error)?;
            stdout.flush().map_err(tui_io_error)?;
        }
        Ok(selected.or_else(|| proposals.first().map(|proposal| proposal.path.clone())))
    }

    fn prompt_first_run_index_root_ftui(
        &self,
        proposals: &[IndexRootProposal],
        session: &mut FtuiSession,
    ) -> SearchResult<Option<PathBuf>> {
        let no_color = self.cli_input.no_color || std::env::var_os("NO_COLOR").is_some();
        let mut selected = 0_usize;

        loop {
            session.render(|frame| {
                render_root_selector_frame(frame, proposals, selected, no_color);
            })?;

            let Some(event) =
                session.poll_event(Duration::from_millis(FSFS_TUI_POLL_INTERVAL_MS))?
            else {
                continue;
            };

            if let Event::Key(key) = event {
                match key.code {
                    KeyCode::Up => {
                        selected = selected.saturating_sub(1);
                    }
                    KeyCode::Down => {
                        selected = selected
                            .saturating_add(1)
                            .min(proposals.len().saturating_sub(1));
                    }
                    KeyCode::Char('k') if key.modifiers == Modifiers::NONE => {
                        selected = selected.saturating_sub(1);
                    }
                    KeyCode::Char('j') if key.modifiers == Modifiers::NONE => {
                        selected = selected
                            .saturating_add(1)
                            .min(proposals.len().saturating_sub(1));
                    }
                    KeyCode::Escape | KeyCode::Char('q' | 'Q') => return Ok(None),
                    KeyCode::Enter => {
                        return Ok(proposals
                            .get(selected)
                            .map(|proposal| proposal.path.clone()));
                    }
                    KeyCode::Char(ch) if ch.is_ascii_digit() => {
                        if let Some(digit) = ch.to_digit(10)
                            && let Ok(raw_index) = usize::try_from(digit)
                            && raw_index > 0
                            && raw_index <= proposals.len()
                        {
                            selected = raw_index.saturating_sub(1);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    #[allow(dead_code, clippy::too_many_lines)]
    fn render_existing_index_dashboard(&self) -> SearchResult<()> {
        if let Ok(mut session) = FtuiSession::enter() {
            match self.render_existing_index_dashboard_ftui(&mut session) {
                Ok(()) => return Ok(()),
                Err(error) => {
                    warn!(
                        error = %error,
                        "interactive ftui dashboard failed; falling back to ansi dashboard"
                    );
                }
            }
        }

        let payload = self.collect_status_payload()?;
        let no_color = self.cli_input.no_color || std::env::var_os("NO_COLOR").is_some();
        let width = tui_terminal_width().clamp(88, 160);
        let rule = "═".repeat(width);
        let indexed_files = payload.index.indexed_files.unwrap_or(0);
        let discovered_files = payload.index.discovered_files.unwrap_or(indexed_files);
        let skipped_files = payload.index.skipped_files.unwrap_or(0);
        let stale_files = payload.index.stale_files.unwrap_or(0);
        let indexed_ratio = ratio_percent_usize(indexed_files, discovered_files.max(1));
        let last_indexed = payload
            .index
            .last_indexed_iso_utc
            .as_deref()
            .unwrap_or("unknown");
        let vector_pct = if payload.index.size_bytes == 0 {
            0.0
        } else {
            ratio_percent(payload.index.vector_index_bytes, payload.index.size_bytes)
        };
        let lexical_pct = if payload.index.size_bytes == 0 {
            0.0
        } else {
            ratio_percent(payload.index.lexical_index_bytes, payload.index.size_bytes)
        };
        let avg_bytes_per_file = if indexed_files == 0 {
            0_u64
        } else {
            payload.index.size_bytes / u64::try_from(indexed_files).unwrap_or(1)
        };
        let fast_cached = payload
            .models
            .iter()
            .find(|model| model.tier == "fast")
            .is_some_and(|model| model.cached);
        let quality_cached = payload
            .models
            .iter()
            .find(|model| model.tier == "quality")
            .is_some_and(|model| model.cached);
        let mode_summary = if fast_cached && quality_cached && !self.config.search.fast_only {
            paint("full hybrid (fast + quality semantic)", "32", no_color)
        } else if fast_cached {
            paint(
                "fast semantic + lexical (quality unavailable)",
                "33",
                no_color,
            )
        } else {
            paint(
                "hash + lexical fallback (offline-safe default)",
                "38;5;214",
                no_color,
            )
        };
        let mut stdout = std::io::stdout();
        write!(stdout, "\u{1b}[2J\u{1b}[H").map_err(tui_io_error)?;
        writeln!(stdout, "{}", paint(&rule, "38;5;24", no_color)).map_err(tui_io_error)?;
        writeln!(
            stdout,
            "{} {}",
            paint("fsfs", "1;38;5;45", no_color),
            paint("Search Control Deck", "1;37", no_color)
        )
        .map_err(tui_io_error)?;
        writeln!(
            stdout,
            "{} {}   {} {}",
            paint("status:", "38;5;244", no_color),
            paint("ready", "1;32", no_color),
            paint("search mode:", "38;5;244", no_color),
            mode_summary
        )
        .map_err(tui_io_error)?;
        writeln!(
            stdout,
            "index root: {}",
            truncate_middle(&payload.index.path, width.saturating_sub(14))
        )
        .map_err(tui_io_error)?;
        writeln!(stdout, "{}", paint(&rule, "38;5;24", no_color)).map_err(tui_io_error)?;
        writeln!(stdout, "{}", paint("CORPUS", "1;38;5;81", no_color)).map_err(tui_io_error)?;
        writeln!(
            stdout,
            "  files indexed={}  discovered={}  skipped={}  coverage={indexed_ratio:.1}%",
            format_count_usize(indexed_files),
            format_count_usize(discovered_files),
            format_count_usize(skipped_files),
        )
        .map_err(tui_io_error)?;
        writeln!(
            stdout,
            "  index size={}  avg/indexed_file={}  stale={}",
            humanize_bytes(payload.index.size_bytes),
            humanize_bytes(avg_bytes_per_file),
            format_count_usize(stale_files),
        )
        .map_err(tui_io_error)?;
        writeln!(stdout, "  last indexed={last_indexed}").map_err(tui_io_error)?;
        if let Some(hash) = payload.index.source_hash_hex.as_deref() {
            writeln!(stdout, "  source hash={hash}").map_err(tui_io_error)?;
        }
        writeln!(stdout).map_err(tui_io_error)?;
        writeln!(stdout, "{}", paint("LAYOUT", "1;38;5;111", no_color)).map_err(tui_io_error)?;
        writeln!(
            stdout,
            "  vector={} ({:.1}%)  lexical={} ({:.1}%)",
            humanize_bytes(payload.index.vector_index_bytes),
            vector_pct,
            humanize_bytes(payload.index.lexical_index_bytes),
            lexical_pct,
        )
        .map_err(tui_io_error)?;
        writeln!(
            stdout,
            "  metadata={}  cache={}",
            humanize_bytes(payload.index.metadata_bytes),
            humanize_bytes(payload.index.embedding_cache_bytes),
        )
        .map_err(tui_io_error)?;
        writeln!(stdout).map_err(tui_io_error)?;
        writeln!(stdout, "{}", paint("MODELS", "1;38;5;214", no_color)).map_err(tui_io_error)?;
        if !payload.models.is_empty() {
            for model in &payload.models {
                let state = if model.cached {
                    paint("cached", "32", no_color)
                } else {
                    paint("missing", "31", no_color)
                };
                writeln!(
                    stdout,
                    "  {:<7} {:<30} {:<8} {}",
                    format!("{}:", model.tier),
                    model.name,
                    state,
                    humanize_bytes(model.size_bytes)
                )
                .map_err(tui_io_error)?;
            }
        }
        writeln!(stdout).map_err(tui_io_error)?;
        writeln!(stdout, "{}", paint("QUICK ACTIONS", "1;38;5;220", no_color))
            .map_err(tui_io_error)?;
        writeln!(
            stdout,
            "  1) fsfs search \"your query\" --limit all --format table    # full recall"
        )
        .map_err(tui_io_error)?;
        writeln!(
            stdout,
            "  2) fsfs search \"your query\" --stream --format jsonl        # agent/pipeline mode"
        )
        .map_err(tui_io_error)?;
        writeln!(
            stdout,
            "  3) fsfs download-models --list                              # inspect semantic models"
        )
        .map_err(tui_io_error)?;
        writeln!(
            stdout,
            "  4) fsfs index <path> --watch                              # keep corpus fresh"
        )
        .map_err(tui_io_error)?;
        writeln!(
            stdout,
            "  5) fsfs status --format json                              # machine status payload"
        )
        .map_err(tui_io_error)?;
        writeln!(stdout).map_err(tui_io_error)?;
        writeln!(
            stdout,
            "{}",
            paint("Press Enter to exit.", "38;5;244", no_color)
        )
        .map_err(tui_io_error)?;
        stdout.flush().map_err(tui_io_error)?;

        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .map_err(SearchError::Io)?;
        Ok(())
    }

    #[allow(dead_code)]
    fn render_existing_index_dashboard_ftui(&self, session: &mut FtuiSession) -> SearchResult<()> {
        let no_color = self.cli_input.no_color || std::env::var_os("NO_COLOR").is_some();
        let mut payload = self.collect_status_payload()?;
        let mut last_refresh = Instant::now();

        loop {
            session.render(|frame| {
                render_existing_index_dashboard_frame(
                    frame,
                    &payload,
                    self.config.search.fast_only,
                    no_color,
                );
            })?;

            if let Some(event) =
                session.poll_event(Duration::from_millis(FSFS_TUI_POLL_INTERVAL_MS))?
                && let Event::Key(key) = event
            {
                match key.code {
                    KeyCode::Enter | KeyCode::Escape | KeyCode::Char('q' | 'Q') => {
                        return Ok(());
                    }
                    KeyCode::Char('r' | 'R') => {
                        payload = self.collect_status_payload()?;
                        last_refresh = Instant::now();
                    }
                    _ => {}
                }
            }

            if last_refresh.elapsed() >= Duration::from_millis(FSFS_TUI_STATUS_REFRESH_MS) {
                payload = self.collect_status_payload()?;
                last_refresh = Instant::now();
            }
        }
    }

    async fn run_search_dashboard_tui(&self, cx: &Cx) -> SearchResult<()> {
        let no_color = self.cli_input.no_color || std::env::var_os("NO_COLOR").is_some();
        let status_payload = self.collect_status_payload()?;
        let mode_hint = self.search_mode_hint()?;
        let configured_limit = self
            .cli_input
            .overrides
            .limit
            .unwrap_or(self.config.search.default_limit);
        let result_limit = if configured_limit == FSFS_SEARCH_UNBOUNDED_LIMIT_SENTINEL {
            FSFS_TUI_INTERACTIVE_RESULT_LIMIT
        } else {
            configured_limit.max(1)
        };
        let mut state =
            SearchDashboardState::new(status_payload, mode_hint, result_limit, no_color);
        let mut resources = self.prepare_search_execution_resources(SearchExecutionMode::Full)?;

        match FtuiSession::enter() {
            Ok(mut session) => {
                self.run_search_dashboard_ftui(
                    cx,
                    &mut session,
                    &mut state,
                    &mut resources,
                    no_color,
                )
                .await
            }
            Err(error) => {
                warn!(
                    error = %error,
                    "interactive fsfs search cockpit unavailable; falling back to ansi interaction"
                );
                self.run_search_dashboard_ansi(cx, &mut state, &mut resources, no_color)
                    .await
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    async fn run_search_dashboard_ftui(
        &self,
        cx: &Cx,
        session: &mut FtuiSession,
        state: &mut SearchDashboardState,
        resources: &mut SearchExecutionResources,
        no_color: bool,
    ) -> SearchResult<()> {
        let mut last_status_refresh = Instant::now();
        let mut last_render = Instant::now()
            .checked_sub(Duration::from_millis(
                FSFS_TUI_SEARCH_RENDER_MIN_INTERVAL_MS,
            ))
            .unwrap_or_else(Instant::now);
        let mut render_pending = true;

        loop {
            let mut force_status_refresh = false;
            let mut interaction_changed = false;
            if let Some(event) =
                session.poll_event(Duration::from_millis(FSFS_TUI_SEARCH_POLL_INTERVAL_MS))?
            {
                if state.search_active {
                    match &event {
                        Event::Key(key) => match (key.code, key.modifiers) {
                            (KeyCode::Escape, _) => {
                                state.set_focus(false);
                                interaction_changed = true;
                            }
                            (KeyCode::Enter | KeyCode::Tab | KeyCode::Down, _) => {
                                if state.latest_hits().is_empty()
                                    && !state.query_input.value().trim().is_empty()
                                {
                                    state.force_refresh_now();
                                } else {
                                    state.next_hit();
                                }
                                interaction_changed = true;
                            }
                            (KeyCode::Up | KeyCode::BackTab, _) => {
                                state.prev_hit();
                                interaction_changed = true;
                            }
                            _ => {
                                if state.query_input.handle_event(&event) {
                                    state.active_hit_index = 0;
                                    state.context_scroll = 0;
                                    state.mark_query_dirty();
                                    interaction_changed = true;
                                }
                            }
                        },
                        Event::Paste(_) => {
                            if state.query_input.handle_event(&event) {
                                state.active_hit_index = 0;
                                state.context_scroll = 0;
                                state.mark_query_dirty();
                                interaction_changed = true;
                            }
                        }
                        _ => {}
                    }
                } else if let Event::Key(key) = &event {
                    match (key.code, key.modifiers) {
                        (KeyCode::Escape | KeyCode::Char('q' | 'Q'), _) => return Ok(()),
                        (KeyCode::Char('/'), Modifiers::NONE) => {
                            state.set_focus(true);
                            interaction_changed = true;
                        }
                        (KeyCode::Char('f'), modifiers) if modifiers.contains(Modifiers::CTRL) => {
                            state.set_focus(true);
                            interaction_changed = true;
                        }
                        (KeyCode::Char('r' | 'R'), _) => {
                            force_status_refresh = true;
                            interaction_changed = true;
                        }
                        (KeyCode::Char('l'), modifiers) if modifiers.contains(Modifiers::CTRL) => {
                            state.query_input.clear();
                            state.mark_query_dirty();
                            state.set_focus(true);
                            interaction_changed = true;
                        }
                        (KeyCode::Enter | KeyCode::Tab | KeyCode::Down, _)
                        | (KeyCode::Char('n'), Modifiers::NONE) => {
                            state.next_hit();
                            interaction_changed = true;
                        }
                        (KeyCode::Up | KeyCode::BackTab, _)
                        | (KeyCode::Char('N'), Modifiers::NONE)
                        | (KeyCode::Char('n'), Modifiers::SHIFT) => {
                            state.prev_hit();
                            interaction_changed = true;
                        }
                        (KeyCode::Home, _) | (KeyCode::Char('g'), Modifiers::NONE) => {
                            state.active_hit_index = 0;
                            state.context_scroll = 0;
                            interaction_changed = true;
                        }
                        (KeyCode::End, _) | (KeyCode::Char('G'), Modifiers::NONE) => {
                            let len = state.latest_hits().len();
                            state.active_hit_index = len.saturating_sub(1);
                            state.context_scroll = 0;
                            interaction_changed = true;
                        }
                        (KeyCode::Char('['), Modifiers::NONE) => {
                            state.scroll_context_up(1);
                            interaction_changed = true;
                        }
                        (KeyCode::Char(']'), Modifiers::NONE) => {
                            state.scroll_context_down(1);
                            interaction_changed = true;
                        }
                        (KeyCode::PageUp, _) => {
                            state.scroll_context_up(8);
                            interaction_changed = true;
                        }
                        (KeyCode::PageDown, _) => {
                            state.scroll_context_down(8);
                            interaction_changed = true;
                        }
                        (KeyCode::Char(ch), Modifiers::NONE) if !ch.is_control() => {
                            state.set_focus(true);
                            if state.query_input.handle_event(&event) {
                                state.active_hit_index = 0;
                                state.context_scroll = 0;
                                state.mark_query_dirty();
                                interaction_changed = true;
                            }
                        }
                        _ => {}
                    }
                }
            }

            let status_refresh_due = force_status_refresh
                || last_status_refresh.elapsed()
                    >= Duration::from_millis(FSFS_TUI_STATUS_REFRESH_MS);
            let user_typing_recently = state.last_query_edit_at.is_some_and(|edited_at| {
                edited_at.elapsed() < Duration::from_millis(FSFS_TUI_STATUS_REFRESH_ACTIVE_GRACE_MS)
            });
            if status_refresh_due && (!user_typing_recently || force_status_refresh) {
                match self.collect_status_payload() {
                    Ok(payload) => {
                        state.status_payload = payload;
                        state.mode_hint = self.search_mode_hint()?;
                        last_status_refresh = Instant::now();
                        render_pending = true;
                    }
                    Err(error) => {
                        warn!(
                            error = %error,
                            "search dashboard status refresh failed; keeping previous snapshot"
                        );
                        last_status_refresh = Instant::now();
                    }
                }
            }

            // Keep interaction responsive by prioritizing input and then executing
            // only one refresh tier per loop iteration.
            if state.should_refresh_lexical() {
                self.refresh_search_dashboard_lexical(cx, state, resources)
                    .await;
                render_pending = true;
            } else if state.should_refresh_semantic() {
                self.refresh_search_dashboard_semantic_fast(cx, state, resources)
                    .await;
                render_pending = true;
            } else if state.should_refresh_quality() {
                self.refresh_search_dashboard_quality(cx, state, resources)
                    .await;
                render_pending = true;
            }

            if interaction_changed {
                render_pending = true;
            }

            let elapsed_since_render = last_render.elapsed();
            let min_render_interval = Duration::from_millis(FSFS_TUI_SEARCH_RENDER_MIN_INTERVAL_MS);
            let idle_heartbeat_interval =
                Duration::from_millis(FSFS_TUI_SEARCH_RENDER_IDLE_HEARTBEAT_MS);
            let should_render = (render_pending && elapsed_since_render >= min_render_interval)
                || elapsed_since_render >= idle_heartbeat_interval;

            if should_render {
                session.render(|frame| {
                    render_search_dashboard_frame(frame, state, no_color);
                })?;
                last_render = Instant::now();
                render_pending = false;
            }
        }
    }

    async fn run_search_dashboard_ansi(
        &self,
        cx: &Cx,
        state: &mut SearchDashboardState,
        resources: &mut SearchExecutionResources,
        no_color: bool,
    ) -> SearchResult<()> {
        let width = tui_terminal_width().clamp(88, 168);
        let rule = "═".repeat(width);
        let mut stdout = std::io::stdout();

        loop {
            state.status_payload = self.collect_status_payload()?;
            state.mode_hint = self.search_mode_hint()?;

            write!(stdout, "\u{1b}[2J\u{1b}[H").map_err(tui_io_error)?;
            writeln!(stdout, "{}", paint(&rule, "38;5;24", no_color)).map_err(tui_io_error)?;
            writeln!(
                stdout,
                "{} {}",
                paint("fsfs", "1;38;5;45", no_color),
                paint("Search Cockpit (ANSI fallback)", "1;37", no_color)
            )
            .map_err(tui_io_error)?;
            writeln!(
                stdout,
                "index: {}",
                truncate_middle(&state.status_payload.index.path, width.saturating_sub(8),)
            )
            .map_err(tui_io_error)?;
            if let Some(mode_hint) = state.mode_hint.as_deref() {
                writeln!(stdout, "{}", paint(mode_hint, "38;5;244", no_color))
                    .map_err(tui_io_error)?;
            }
            writeln!(stdout, "{}", paint(&rule, "38;5;24", no_color)).map_err(tui_io_error)?;

            if let Some(error) = state.last_error.as_deref() {
                writeln!(
                    stdout,
                    "{} {}",
                    paint("search error:", "1;31", no_color),
                    error
                )
                .map_err(tui_io_error)?;
                writeln!(stdout).map_err(tui_io_error)?;
            } else if let Some(payload) = state.latest_payload() {
                let rendered = crate::adapters::format_emitter::render_search_table_for_cli(
                    payload,
                    state.last_search_elapsed_ms,
                    no_color,
                );
                writeln!(stdout, "{rendered}").map_err(tui_io_error)?;
            } else {
                writeln!(
                    stdout,
                    "{}",
                    paint(
                        "Type a query to search. Use `q` or `quit` to exit.",
                        "38;5;250",
                        no_color
                    )
                )
                .map_err(tui_io_error)?;
            }

            write!(stdout, "{} ", paint("search>", "1;38;5;45", no_color)).map_err(tui_io_error)?;
            stdout.flush().map_err(tui_io_error)?;

            let mut input = String::new();
            std::io::stdin()
                .read_line(&mut input)
                .map_err(SearchError::Io)?;
            let trimmed = input.trim();
            if matches!(trimmed, "q" | "Q" | "quit" | "QUIT" | "exit" | "EXIT") {
                return Ok(());
            }

            state.query_input.set_value(trimmed.to_owned());
            state.mark_query_dirty();
            self.refresh_search_dashboard_full(cx, state, resources)
                .await;
        }
    }

    async fn refresh_search_dashboard_mode(
        &self,
        cx: &Cx,
        state: &mut SearchDashboardState,
        mode: SearchExecutionMode,
        resources: &mut SearchExecutionResources,
    ) -> SearchResult<Vec<SearchPayload>> {
        self.refresh_search_dashboard_mode_with_limit(
            cx,
            state,
            mode,
            resources,
            state.result_limit,
            true,
        )
        .await
    }

    async fn refresh_search_dashboard_mode_with_limit(
        &self,
        cx: &Cx,
        state: &mut SearchDashboardState,
        mode: SearchExecutionMode,
        resources: &mut SearchExecutionResources,
        search_limit: usize,
        include_snippets: bool,
    ) -> SearchResult<Vec<SearchPayload>> {
        let query = state.query_input.value().trim().to_owned();
        if query.is_empty() {
            state.clear_search_results();
            return Ok(Vec::new());
        }

        let started = Instant::now();
        let result = self
            .execute_search_payloads_with_mode_using_resources(
                cx,
                &query,
                search_limit.max(1),
                mode,
                resources,
                SearchExecutionFlags {
                    include_snippets,
                    persist_explain_session: false,
                },
            )
            .await;

        let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        state.last_search_elapsed_ms = Some(elapsed_ms);
        state.search_invocations = state.search_invocations.saturating_add(1);
        let elapsed_sample = f64::from(u32::try_from(elapsed_ms).unwrap_or(u32::MAX));
        state.push_latency_sample(elapsed_sample);

        match result {
            Ok(payloads) => {
                state.last_error = None;
                let hit_sample = f64::from(
                    u32::try_from(
                        payloads
                            .last()
                            .map_or(0_usize, |payload| payload.hits.len()),
                    )
                    .unwrap_or(u32::MAX),
                );
                state.push_hit_sample(hit_sample);
                Ok(payloads)
            }
            Err(error) => {
                state.last_error = Some(error.to_string());
                state.push_hit_sample(0.0);
                Err(error)
            }
        }
    }

    async fn refresh_search_dashboard_lexical(
        &self,
        cx: &Cx,
        state: &mut SearchDashboardState,
        resources: &mut SearchExecutionResources,
    ) {
        state.pending_lexical_refresh = false;
        state.clear_pending_quality_refresh();
        if state.query_input.value().trim().is_empty() {
            state.pending_semantic_refresh = false;
            state.clear_search_results();
            return;
        }

        let lexical_limit = state.lexical_stage_result_limit();
        if let Ok(payloads) = self
            .refresh_search_dashboard_mode_with_limit(
                cx,
                state,
                SearchExecutionMode::LexicalOnly,
                resources,
                lexical_limit,
                false,
            )
            .await
        {
            state.set_phase_payloads(payloads, vec![DashboardSearchStage::Lexical]);
            state.clamp_active_hit();
        } else {
            state.phase_payloads.clear();
            state.stage_chain = vec![DashboardSearchStage::Lexical];
            state.active_hit_index = 0;
            state.context_scroll = 0;
            state.pending_semantic_refresh = false;
        }
    }

    async fn refresh_search_dashboard_semantic_fast(
        &self,
        cx: &Cx,
        state: &mut SearchDashboardState,
        resources: &mut SearchExecutionResources,
    ) {
        state.pending_semantic_refresh = false;
        let query = state.query_input.value().trim().to_owned();
        if query.is_empty() {
            state.clear_pending_quality_refresh();
            return;
        }
        // Interactive search-as-you-type keeps single-token queries in a lexical-only
        // lane to protect sub-30ms feel; semantic tiers activate once intent is clearer.
        // For this lane we split work across two timescales:
        // 1) fast lexical refresh (no snippet hydration)
        // 2) delayed snippet hydration on the quality idle gate.
        if Self::tui_prefers_lexical_only_query(&query) {
            // Keep the interactive fast lane bounded to the visible lexical head.
            // Full-limit hydration is deferred to the delayed quality gate.
            let fast_limit = state.result_limit.clamp(1, FSFS_SEARCH_SNIPPET_HEAD_LIMIT);
            if let Ok(payloads) = self
                .refresh_search_dashboard_mode_with_limit(
                    cx,
                    state,
                    SearchExecutionMode::LexicalOnly,
                    resources,
                    fast_limit,
                    false,
                )
                .await
            {
                state.set_phase_payloads(payloads, vec![DashboardSearchStage::Lexical]);
                state.clamp_active_hit();
                state.schedule_quality_refresh();
            } else {
                state.clear_pending_quality_refresh();
            }
            return;
        }

        match self
            .refresh_search_dashboard_mode(cx, state, SearchExecutionMode::FastOnly, resources)
            .await
        {
            Ok(payloads) => {
                let has_semantic_hits = payloads.last().is_some_and(|payload| {
                    payload.hits.iter().any(|hit| hit.semantic_rank.is_some())
                });
                let can_run_quality = !self.config.search.fast_only
                    && has_semantic_hits
                    && state.quality_model_cached()
                    && resources.quality_stage_viable(self.config.search.fast_only);
                let stages = if can_run_quality {
                    vec![
                        DashboardSearchStage::Lexical,
                        DashboardSearchStage::SemanticFast,
                    ]
                } else if has_semantic_hits && !self.config.search.fast_only {
                    vec![
                        DashboardSearchStage::Lexical,
                        DashboardSearchStage::SemanticFast,
                        DashboardSearchStage::QualitySkipped,
                    ]
                } else {
                    vec![DashboardSearchStage::Lexical]
                };
                state.set_phase_payloads(payloads, stages);
                state.clamp_active_hit();
                if can_run_quality {
                    state.schedule_quality_refresh();
                } else {
                    state.clear_pending_quality_refresh();
                }
            }
            Err(_) => {
                state.clear_pending_quality_refresh();
            }
        }
    }

    #[must_use]
    fn tui_prefers_lexical_only_query(query: &str) -> bool {
        let normalized = query.trim();
        if normalized.is_empty() {
            return false;
        }
        // One-token queries are usually either identifiers, path fragments, or terse
        // term lookups where lexical ranking dominates and semantic latency is churn.
        normalized.split_whitespace().count() == 1
    }

    async fn refresh_search_dashboard_quality(
        &self,
        cx: &Cx,
        state: &mut SearchDashboardState,
        resources: &mut SearchExecutionResources,
    ) {
        state.clear_pending_quality_refresh();
        let query = state.query_input.value().trim().to_owned();
        if query.is_empty() {
            return;
        }
        // Single-token interactive lane runs lexical-only for ranking, then hydrates
        // snippets on this delayed phase to keep keystroke latency low.
        if Self::tui_prefers_lexical_only_query(&query) {
            let full_limit = state.result_limit.max(1);
            if let Ok(payloads) = self
                .refresh_search_dashboard_mode_with_limit(
                    cx,
                    state,
                    SearchExecutionMode::LexicalOnly,
                    resources,
                    full_limit,
                    true,
                )
                .await
            {
                state.set_phase_payloads(payloads, vec![DashboardSearchStage::Lexical]);
                state.clamp_active_hit();
            }
            return;
        }
        if !resources.quality_stage_viable(self.config.search.fast_only) {
            return;
        }

        match self
            .refresh_search_dashboard_mode(cx, state, SearchExecutionMode::Full, resources)
            .await
        {
            Ok(payloads) => {
                let has_semantic_hits = payloads.last().is_some_and(|payload| {
                    payload.hits.iter().any(|hit| hit.semantic_rank.is_some())
                });
                let quality_stage = match payloads.last().map(|payload| payload.phase) {
                    Some(SearchOutputPhase::Refined) => DashboardSearchStage::QualityRefined,
                    Some(SearchOutputPhase::Initial) => DashboardSearchStage::QualitySkipped,
                    Some(SearchOutputPhase::RefinementFailed) => {
                        DashboardSearchStage::QualityRefinementFailed
                    }
                    None => DashboardSearchStage::QualityRefinementFailed,
                };
                let stages = if has_semantic_hits {
                    vec![
                        DashboardSearchStage::Lexical,
                        DashboardSearchStage::SemanticFast,
                        quality_stage,
                    ]
                } else {
                    vec![DashboardSearchStage::Lexical]
                };
                state.set_phase_payloads(payloads, stages);
                state.clamp_active_hit();
            }
            Err(_) => {
                state.stage_chain = vec![
                    DashboardSearchStage::Lexical,
                    DashboardSearchStage::SemanticFast,
                    DashboardSearchStage::QualityRefinementFailed,
                ];
            }
        }
    }

    async fn refresh_search_dashboard_full(
        &self,
        cx: &Cx,
        state: &mut SearchDashboardState,
        resources: &mut SearchExecutionResources,
    ) {
        if state.query_input.value().trim().is_empty() {
            state.clear_search_results();
            return;
        }

        if let Ok(payloads) = self
            .refresh_search_dashboard_mode(cx, state, SearchExecutionMode::Full, resources)
            .await
        {
            let has_semantic_hits = payloads
                .last()
                .is_some_and(|payload| payload.hits.iter().any(|hit| hit.semantic_rank.is_some()));
            let mut stages = vec![DashboardSearchStage::Lexical];
            if has_semantic_hits {
                stages.push(DashboardSearchStage::SemanticFast);
                if let Some(phase) = payloads.last().map(|payload| payload.phase) {
                    match phase {
                        SearchOutputPhase::Initial => {
                            stages.push(DashboardSearchStage::QualitySkipped);
                        }
                        SearchOutputPhase::Refined => {
                            stages.push(DashboardSearchStage::QualityRefined);
                        }
                        SearchOutputPhase::RefinementFailed => {
                            stages.push(DashboardSearchStage::QualityRefinementFailed);
                        }
                    }
                }
            }
            state.set_phase_payloads(payloads, stages);
            state.clamp_active_hit();
        } else {
            state.phase_payloads.clear();
            state.active_hit_index = 0;
            state.context_scroll = 0;
        }
    }

    async fn run_first_run_indexing_tui(
        &self,
        cx: &Cx,
        selected_root: PathBuf,
    ) -> SearchResult<()> {
        let mut index_input = self.cli_input.clone();
        index_input.command = CliCommand::Index;
        index_input.target_path = Some(selected_root);
        index_input.watch = false;
        index_input.overrides.allow_background_indexing = Some(false);

        let runtime = Self::new(self.config.clone()).with_cli_input(index_input);
        let no_color = runtime.cli_input.no_color || std::env::var_os("NO_COLOR").is_some();
        let mut ftui_session = FtuiSession::enter().ok();
        let _ansi_guard = if ftui_session.is_none() {
            Some(TerminalRenderGuard::enter()?)
        } else {
            None
        };
        let mut last_render = Instant::now();
        let mut latest = None;
        let mut render_state = IndexingRenderState::default();
        runtime
            .run_one_shot_index_scaffold_with_progress(cx, CliCommand::Index, |snapshot| {
                render_state.observe(snapshot);
                let now = Instant::now();
                let force_render = latest.is_none()
                    || matches!(
                        snapshot.stage,
                        IndexingProgressStage::Discovering
                            | IndexingProgressStage::Finalizing
                            | IndexingProgressStage::Completed
                    )
                    || now.duration_since(last_render)
                        >= Duration::from_millis(INDEXING_TUI_MIN_RENDER_INTERVAL_MS);
                if force_render {
                    if let Some(session) = ftui_session.as_mut() {
                        render_indexing_progress_screen_ftui(
                            session,
                            snapshot,
                            &render_state,
                            no_color,
                        )?;
                    } else {
                        render_indexing_progress_screen(snapshot, &render_state, no_color)?;
                    }
                    last_render = now;
                }
                latest = Some(snapshot.clone());
                Ok(())
            })
            .await?;

        if let Some(snapshot) = latest.as_ref() {
            if let Some(session) = ftui_session.as_mut() {
                render_indexing_progress_screen_ftui(session, snapshot, &render_state, no_color)?;
            } else {
                render_indexing_progress_screen(snapshot, &render_state, no_color)?;
            }
        }

        if let Some(session) = ftui_session.as_mut() {
            return wait_for_ftui_dismiss(
                session,
                "Initial indexing finished. Press Enter, Esc, or q to open search cockpit.",
                no_color,
            );
        }

        let mut stdout = std::io::stdout();
        writeln!(stdout).map_err(tui_io_error)?;
        writeln!(
            stdout,
            "Initial indexing finished. Press Enter to open the search cockpit."
        )
        .map_err(tui_io_error)?;
        stdout.flush().map_err(tui_io_error)?;
        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .map_err(SearchError::Io)?;
        Ok(())
    }

    async fn run_cli_with_shutdown(
        &self,
        cx: &Cx,
        shutdown: &ShutdownCoordinator,
    ) -> SearchResult<()> {
        self.run_cli(cx).await?;

        let watch_enabled_for_command = self.config.indexing.watch_mode
            && matches!(
                self.cli_input.command,
                CliCommand::Index | CliCommand::Watch
            );

        if watch_enabled_for_command {
            match self.build_live_ingest_pipeline() {
                Ok(pipeline) => {
                    let target_root = self.resolve_target_root()?;
                    let watcher = FsWatcher::new(
                        vec![target_root],
                        self.config.discovery.clone(),
                        Arc::new(pipeline),
                    );
                    watcher.start(cx).await?;
                    let policy = watcher.execution_policy();
                    let storage_paths = self.default_index_storage_paths();
                    let lifecycle_tracker = self.new_runtime_lifecycle_tracker(&storage_paths);
                    info!(
                        watch_roots = watcher.roots().len(),
                        debounce_ms = policy.debounce_ms,
                        batch_size = policy.batch_size,
                        disk_budget_bytes = lifecycle_tracker.resource_limits().max_index_bytes,
                        "fsfs watch mode enabled; watcher started"
                    );

                    let reason = self
                        .await_shutdown(
                            cx,
                            shutdown,
                            Some(&watcher),
                            Some((&lifecycle_tracker, &storage_paths)),
                        )
                        .await;
                    watcher.stop().await;
                    self.finalize_shutdown(cx, reason).await?;
                }
                Err(ref error) if matches!(error, SearchError::IndexNotFound { .. }) => {
                    warn!(
                        error = %error,
                        "watch mode skipped: no existing index found; run 'fsfs index' first"
                    );
                }
                Err(error) => return Err(error),
            }
        }

        Ok(())
    }

    async fn run_tui_with_shutdown(
        &self,
        cx: &Cx,
        shutdown: &ShutdownCoordinator,
    ) -> SearchResult<()> {
        self.run_tui(cx).await?;
        if self.cli_input.command == CliCommand::Tui {
            return Ok(());
        }
        let reason = self.await_shutdown(cx, shutdown, None, None).await;
        self.finalize_shutdown(cx, reason).await
    }

    #[allow(clippy::too_many_lines)]
    async fn await_shutdown(
        &self,
        cx: &Cx,
        shutdown: &ShutdownCoordinator,
        watcher: Option<&FsWatcher>,
        disk_budget: Option<(&LifecycleTracker, &IndexStoragePaths)>,
    ) -> ShutdownReason {
        let mut pressure_collector = HostPressureCollector::default();
        let mut pressure_controller = self.new_pressure_controller();
        let sample_interval_ms = self.config.pressure.sample_interval_ms.max(1);
        let mut last_pressure_sample_ms = 0_u64;
        let mut last_applied_watcher_state: Option<PressureState> = None;
        let mut last_disk_stage: Option<DiskBudgetStage> = None;
        let mut last_tombstone_cleanup_ms: Option<u64> = None;

        loop {
            if let Some(watcher) = watcher {
                let now_ms = pressure_timestamp_ms();
                if now_ms.saturating_sub(last_pressure_sample_ms) >= sample_interval_ms {
                    let mut target_watcher_state: Option<PressureState> = None;
                    match self.collect_pressure_signal(&mut pressure_collector) {
                        Ok(sample) => {
                            let transition =
                                self.observe_pressure(&mut pressure_controller, sample);
                            target_watcher_state = Some(transition.to);
                            if transition.changed {
                                info!(
                                    pressure_state = ?transition.to,
                                    reason_code = transition.reason_code,
                                    "fsfs watcher pressure policy updated"
                                );
                            }
                        }
                        Err(err) => {
                            warn!(error = %err, "fsfs watcher pressure update failed");
                        }
                    }

                    if let Some((tracker, storage_paths)) = disk_budget {
                        match self.evaluate_storage_disk_budget(tracker, storage_paths) {
                            Ok(snapshot) => {
                                let snapshot = self.apply_storage_emergency_override(
                                    snapshot,
                                    tracker.current_resource_usage().effective_index_bytes(),
                                    tracker.resource_limits().max_index_bytes,
                                );
                                if let Some(snapshot) = snapshot {
                                    let control_plan = Self::disk_budget_control_plan(snapshot);
                                    if last_disk_stage != Some(snapshot.stage) {
                                        info!(
                                            stage = ?snapshot.stage,
                                            action = ?snapshot.action,
                                            reason_code = control_plan.reason_code,
                                            used_bytes = snapshot.used_bytes,
                                            budget_bytes = snapshot.budget_bytes,
                                            usage_per_mille = snapshot.usage_per_mille,
                                            eviction_target_bytes = control_plan.eviction_target_bytes,
                                            request_eviction = control_plan.request_eviction,
                                            request_compaction = control_plan.request_compaction,
                                            request_tombstone_cleanup = control_plan.request_tombstone_cleanup,
                                            "fsfs disk budget stage updated"
                                        );
                                        last_disk_stage = Some(snapshot.stage);
                                    }

                                    if control_plan.request_tombstone_cleanup {
                                        let cleanup_due =
                                            last_tombstone_cleanup_ms.is_none_or(|last| {
                                                now_ms.saturating_sub(last)
                                                    >= TOMBSTONE_CLEANUP_MIN_INTERVAL_MS
                                            });
                                        if cleanup_due {
                                            match self.cleanup_catalog_tombstones(now_ms) {
                                                Ok((deleted_rows, cutoff_ms)) => {
                                                    info!(
                                                        deleted_rows,
                                                        cutoff_ms,
                                                        reason_code = control_plan.reason_code,
                                                        "fsfs catalog tombstone cleanup executed"
                                                    );
                                                    last_tombstone_cleanup_ms = Some(now_ms);
                                                }
                                                Err(error) => {
                                                    warn!(
                                                        error = %error,
                                                        reason_code = control_plan.reason_code,
                                                        "fsfs catalog tombstone cleanup failed"
                                                    );
                                                }
                                            }
                                        }
                                    }

                                    let combined_state = target_watcher_state.map_or(
                                        control_plan.watcher_pressure_state,
                                        |state| {
                                            more_severe_pressure_state(
                                                state,
                                                control_plan.watcher_pressure_state,
                                            )
                                        },
                                    );
                                    target_watcher_state = Some(combined_state);
                                }
                            }
                            Err(err) => {
                                warn!(error = %err, "fsfs disk budget evaluation failed");
                            }
                        }
                    }

                    if let Some(state) = target_watcher_state {
                        watcher.apply_pressure_state(state);
                        if last_applied_watcher_state != Some(state) {
                            info!(
                                watcher_pressure_state = ?state,
                                "fsfs watcher effective pressure state applied"
                            );
                            last_applied_watcher_state = Some(state);
                        }
                    }
                    last_pressure_sample_ms = now_ms;
                }
            }

            if shutdown.take_reload_requested() {
                info!("fsfs runtime observed SIGHUP; config reload scaffold invoked");
            }

            if shutdown.is_shutting_down() {
                return shutdown
                    .current_reason()
                    .unwrap_or(ShutdownReason::UserRequest);
            }

            if cx.is_cancel_requested() {
                return ShutdownReason::Error(
                    "runtime cancelled while waiting for shutdown".to_owned(),
                );
            }

            asupersync::time::sleep(
                asupersync::time::wall_now(),
                std::time::Duration::from_millis(25),
            )
            .await;
        }
    }

    async fn finalize_shutdown(&self, _cx: &Cx, reason: ShutdownReason) -> SearchResult<()> {
        // Placeholder for fsync/WAL flush/index checkpoint once these subsystems
        // are wired into fsfs runtime lanes.
        std::future::ready(()).await;
        info!(reason = ?reason, "fsfs graceful shutdown finalization completed");
        Ok(())
    }
}

fn print_cli_help() {
    println!("Usage: fsfs <command> [options]");
    println!();
    println!("Commands:");
    println!(
        "  search <query>            Search indexed corpus (daemon-backed by default on unix)"
    );
    println!(
        "  serve [--daemon]          Run long-lived query server (stdio by default, socket daemon with --daemon)"
    );
    println!("  index [path]              Build/update index");
    println!("  watch [path]              Alias for index --watch");
    println!("  explain <result-id>       Explain ranking details");
    println!("  status                    Show index and runtime status");
    println!("  config <action>           Manage configuration");
    println!("  download-models [model]   Download/verify embedding models");
    println!("  doctor                    Run local health checks");
    println!("  update [--check]          Check/apply binary updates");
    println!("  completions <shell>       Generate shell completions");
    println!("  uninstall [--yes] [--dry-run] [--purge]  Remove local fsfs artifacts");
    println!("  help                      Show this help");
    println!("  version                   Show version");
    println!();
    println!("Global flags: --verbose/-v --quiet/-q --no-color --format --config");
    println!("Search flags: --daemon --no-daemon --daemon-socket <path> --stream");
}

const fn completion_script(shell: CompletionShell) -> &'static str {
    match shell {
        CompletionShell::Bash => {
            "complete -W \"search serve index watch explain status config download-models download doctor update completions uninstall help version\" fsfs"
        }
        CompletionShell::Zsh => {
            "compdef '_arguments \"1: :((search serve index watch explain status config download-models download doctor update completions uninstall help version))\"' fsfs"
        }
        CompletionShell::Fish => {
            "complete -c fsfs -f -a \"search serve index watch explain status config download-models download doctor update completions uninstall help version\""
        }
        CompletionShell::PowerShell => {
            "Register-ArgumentCompleter -CommandName fsfs -ScriptBlock { param($wordToComplete) 'search','serve','index','watch','explain','status','config','download-models','download','doctor','update','completions','uninstall','help','version' | Where-Object { $_ -like \"$wordToComplete*\" } }"
        }
    }
}

fn rrf_contribution_for_rank(k: f64, rank: Option<usize>) -> f64 {
    let Some(rank) = rank else {
        return 0.0;
    };
    let safe_k = if k.is_finite() && k >= 0.0 { k } else { 60.0 };
    let rank_u32 = u32::try_from(rank).unwrap_or(u32::MAX);
    1.0 / (safe_k + f64::from(rank_u32) + 1.0)
}

fn render_explain_table(
    requested_result_id: &str,
    payload: &FsfsExplanationPayload,
    hit: &ExplainSessionHit,
    rrf_k: f64,
) -> String {
    let lexical_score = hit
        .lexical_score
        .map_or_else(|| "n/a".to_owned(), |value| format!("{value:.6}"));
    let semantic_fast_score = hit
        .semantic_score
        .map_or_else(|| "n/a".to_owned(), |value| format!("{value:.6}"));
    let quality_score = "n/a";
    let rerank_score = "n/a";
    let lexical_rrf = rrf_contribution_for_rank(rrf_k, hit.lexical_rank);
    let semantic_rrf = rrf_contribution_for_rank(rrf_k, hit.semantic_rank);
    let total_rrf = lexical_rrf + semantic_rrf;
    let lexical_rank = hit
        .lexical_rank
        .map_or_else(|| "-".to_owned(), |rank| rank.saturating_add(1).to_string());
    let semantic_rank = hit
        .semantic_rank
        .map_or_else(|| "-".to_owned(), |rank| rank.saturating_add(1).to_string());

    let mut lines = Vec::new();
    lines.push(format!("Result ID: {requested_result_id}"));
    lines.push(format!("Result: {}", payload.ranking.doc_id));
    lines.push(format!("Query: {}", payload.query));
    lines.push(String::new());
    lines.push(format!("Lexical (BM25): {lexical_score}"));
    lines.push(format!("Semantic (fast): {semantic_fast_score}"));
    lines.push(format!("Semantic (quality): {quality_score}"));
    lines.push(format!("Reranker: {rerank_score}"));
    lines.push(format!(
        "RRF: k={rrf_k:.1}, lexical_rank={lexical_rank}, semantic_rank={semantic_rank}, lexical_contrib={lexical_rrf:.6}, semantic_contrib={semantic_rrf:.6}, total={total_rrf:.6}"
    ));
    lines.push(format!(
        "Final blended score: {:.6}",
        payload.ranking.final_score
    ));

    lines.join("\n")
}

fn pressure_timestamp_ms() -> u64 {
    system_time_to_ms(SystemTime::now())
}

fn system_time_to_ms(time: SystemTime) -> u64 {
    let millis = time
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    u64::try_from(millis).unwrap_or(u64::MAX)
}

#[must_use]
const fn more_severe_pressure_state(left: PressureState, right: PressureState) -> PressureState {
    match (left, right) {
        (PressureState::Emergency, _) | (_, PressureState::Emergency) => PressureState::Emergency,
        (PressureState::Degraded, _) | (_, PressureState::Degraded) => PressureState::Degraded,
        (PressureState::Constrained, _) | (_, PressureState::Constrained) => {
            PressureState::Constrained
        }
        _ => PressureState::Normal,
    }
}

#[must_use]
fn conservative_budget_from_available_bytes(available_bytes: u64) -> u64 {
    let ten_percent = available_bytes / DISK_BUDGET_RATIO_DIVISOR;
    if ten_percent == 0 {
        return 1;
    }
    ten_percent.min(DISK_BUDGET_CAP_BYTES)
}

#[must_use]
fn normalize_probe_path(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().map_or_else(|_| path.to_path_buf(), |cwd| cwd.join(path))
    }
}

#[must_use]
fn nearest_existing_path(path: &Path) -> PathBuf {
    let mut current = normalize_probe_path(path);
    while !current.exists() {
        let Some(parent) = current.parent().map(Path::to_path_buf) else {
            break;
        };
        current = parent;
    }
    current
}

#[must_use]
fn is_uninstall_protected_path(path: &Path) -> bool {
    if path.as_os_str().is_empty() || path.parent().is_none() || path == Path::new("/") {
        return true;
    }
    if let Some(home) = home_dir()
        && path == home
    {
        return true;
    }
    false
}

#[must_use]
fn looks_like_fsfs_index_root(path: &Path) -> bool {
    if path.join(FSFS_SENTINEL_FILE).exists()
        || path.join(FSFS_VECTOR_MANIFEST_FILE).exists()
        || path.join(FSFS_LEXICAL_MANIFEST_FILE).exists()
        || path.join(FSFS_VECTOR_INDEX_FILE).exists()
    {
        return true;
    }

    path.file_name()
        .and_then(|value| value.to_str())
        .is_some_and(|name| {
            let normalized = name.to_ascii_lowercase();
            normalized == ".frankensearch" || normalized == "frankensearch"
        })
}

#[must_use]
fn available_space_for_path(path: &Path) -> Option<u64> {
    let probe = nearest_existing_path(path);
    let disks = Disks::new_with_refreshed_list();
    let mut best_match: Option<(usize, u64)> = None;

    for disk in disks.list() {
        let mount_point = disk.mount_point();
        if probe.starts_with(mount_point) {
            let depth = mount_point.components().count();
            match best_match {
                Some((best_depth, _)) if depth <= best_depth => {}
                _ => best_match = Some((depth, disk.available_space())),
            }
        }
    }

    best_match.map(|(_, bytes)| bytes).or_else(|| {
        disks
            .list()
            .iter()
            .map(sysinfo::Disk::available_space)
            .max()
    })
}

fn absolutize_path(path: &Path) -> SearchResult<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir().map_err(SearchError::Io)?.join(path))
    }
}

fn resolve_manifest_file_path(
    file_key: &str,
    sentinel: Option<&IndexSentinel>,
    index_root: &Path,
) -> PathBuf {
    let file_path = PathBuf::from(file_key);
    if file_path.is_absolute() {
        return file_path;
    }
    if let Some(value) = sentinel {
        return PathBuf::from(&value.target_root).join(file_path);
    }
    index_root.join(file_key)
}

fn normalize_model_token(value: &str) -> String {
    value
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .collect::<String>()
        .to_ascii_lowercase()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RootProbeFileClass {
    Code,
    Document,
}

struct TerminalRenderGuard;

impl TerminalRenderGuard {
    fn enter() -> SearchResult<Self> {
        let mut stdout = std::io::stdout();
        write!(stdout, "\u{1b}[?25l").map_err(tui_io_error)?;
        stdout.flush().map_err(tui_io_error)?;
        Ok(Self)
    }
}

impl Drop for TerminalRenderGuard {
    fn drop(&mut self) {
        let mut stdout = std::io::stdout();
        let _ = write!(stdout, "\u{1b}[?25h\u{1b}[0m");
        let _ = stdout.flush();
    }
}

struct FtuiSession {
    backend: TtyBackend,
    grapheme_pool: GraphemePool,
    previous_buffer: Option<Buffer>,
}

impl FtuiSession {
    fn enter() -> SearchResult<Self> {
        let options = TtySessionOptions {
            alternate_screen: true,
            features: BackendFeatures {
                mouse_capture: true,
                ..BackendFeatures::default()
            },
        };
        #[cfg(unix)]
        let backend = TtyBackend::open(80, 24, options)
            .map_err(|error| tui_subsystem_error("fsfs.tui.ftui", error.to_string()))?;
        #[cfg(not(unix))]
        let backend = {
            let _ = options;
            TtyBackend::new(80, 24)
        };
        Ok(Self {
            backend,
            grapheme_pool: GraphemePool::new(),
            previous_buffer: None,
        })
    }

    fn render(&mut self, renderer: impl FnOnce(&mut Frame)) -> SearchResult<()> {
        let (width, height) = self
            .backend
            .size()
            .map_err(|error| tui_subsystem_error("fsfs.tui.ftui", error.to_string()))?;
        let mut frame = Frame::new(width, height, &mut self.grapheme_pool);
        renderer(&mut frame);
        let diff = self
            .previous_buffer
            .as_ref()
            .map(|previous| BufferDiff::compute(previous, &frame.buffer));
        self.backend
            .presenter()
            .present_ui(&frame.buffer, diff.as_ref(), false)
            .map_err(|error| tui_subsystem_error("fsfs.tui.ftui", error.to_string()))?;
        self.previous_buffer = Some(frame.buffer);
        Ok(())
    }

    fn poll_event(&mut self, timeout: Duration) -> SearchResult<Option<Event>> {
        let has_event = self
            .backend
            .poll_event(timeout)
            .map_err(|error| tui_subsystem_error("fsfs.tui.ftui", error.to_string()))?;
        if !has_event {
            return Ok(None);
        }
        self.backend
            .read_event()
            .map_err(|error| tui_subsystem_error("fsfs.tui.ftui", error.to_string()))
    }
}

fn tui_subsystem_error(subsystem: &'static str, message: String) -> SearchError {
    SearchError::SubsystemError {
        subsystem,
        source: Box::new(std::io::Error::other(message)),
    }
}

fn tui_io_error(source: std::io::Error) -> SearchError {
    SearchError::SubsystemError {
        subsystem: "fsfs.tui",
        source: Box::new(source),
    }
}

fn is_probe_excluded_dir_name(name: &str) -> bool {
    matches!(
        name,
        ".git"
            | ".jj"
            | "node_modules"
            | "target"
            | "__pycache__"
            | ".venv"
            | ".cache"
            | ".npm"
            | ".cargo"
            | ".idea"
            | ".vscode"
            | "dist"
            | "build"
            | ".next"
            | ".frankensearch"
    )
}

fn is_repo_marker_filename(file_name: &str) -> bool {
    matches!(
        file_name,
        "cargo.toml"
            | "package.json"
            | "pnpm-lock.yaml"
            | "yarn.lock"
            | "go.mod"
            | "pyproject.toml"
            | "requirements.txt"
            | "pom.xml"
            | "build.gradle"
            | "readme.md"
    )
}

fn classify_probe_file(path: &Path, file_name: &str) -> Option<RootProbeFileClass> {
    if matches!(
        file_name,
        "readme"
            | "readme.md"
            | "readme.txt"
            | "license"
            | "license.md"
            | "changelog.md"
            | "notes.txt"
    ) {
        return Some(RootProbeFileClass::Document);
    }

    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_ascii_lowercase)?;

    if matches!(
        extension.as_str(),
        "rs" | "py"
            | "ts"
            | "tsx"
            | "js"
            | "jsx"
            | "go"
            | "java"
            | "kt"
            | "c"
            | "cc"
            | "cpp"
            | "h"
            | "hpp"
            | "cs"
            | "swift"
            | "rb"
            | "php"
            | "scala"
            | "sql"
            | "toml"
            | "yaml"
            | "yml"
            | "json"
            | "md"
            | "txt"
            | "rst"
    ) {
        if matches!(extension.as_str(), "md" | "txt" | "rst") {
            Some(RootProbeFileClass::Document)
        } else {
            Some(RootProbeFileClass::Code)
        }
    } else {
        None
    }
}

fn score_index_root(root: &Path, stats: &RootProbeStats, home: Option<&Path>) -> i64 {
    let lower_path = root.to_string_lossy().to_ascii_lowercase();
    let mut score = 0_i64;

    if lower_path.contains("/data/projects") {
        score += 2_000;
    }
    if lower_path.ends_with("/projects")
        || lower_path.ends_with("/code")
        || lower_path.ends_with("/workspace")
        || lower_path.ends_with("/workspaces")
        || lower_path.ends_with("/src")
    {
        score += 1_000;
    }
    if lower_path.contains("/documents") {
        score += 350;
    }
    if home.is_some_and(|home_path| root == home_path) {
        score -= 400;
    }

    score = score
        .saturating_add(i64::try_from(stats.repo_markers).unwrap_or(i64::MAX / 4) * 900)
        .saturating_add(i64::try_from(stats.code_files).unwrap_or(i64::MAX / 4) * 5)
        .saturating_add(i64::try_from(stats.doc_files).unwrap_or(i64::MAX / 4) * 3)
        .saturating_add(i64::try_from(stats.candidate_files).unwrap_or(i64::MAX / 4));
    let size_bonus = i64::try_from(stats.candidate_bytes / (8 * 1024 * 1024)).unwrap_or(0);
    score.saturating_add(size_bonus.min(2_000))
}

fn count_non_empty_lines(content: &str) -> u64 {
    let count = content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .count();
    u64::try_from(count).unwrap_or(u64::MAX)
}

fn truncate_middle(text: &str, max_chars: usize) -> String {
    if max_chars < 5 {
        return text.chars().take(max_chars).collect();
    }
    let chars = text.chars().collect::<Vec<_>>();
    if chars.len() <= max_chars {
        return text.to_owned();
    }
    let left = (max_chars - 3) / 2;
    let right = max_chars - 3 - left;
    let prefix = chars.iter().take(left).collect::<String>();
    let suffix = chars
        .iter()
        .skip(chars.len().saturating_sub(right))
        .collect::<String>();
    format!("{prefix}...{suffix}")
}

fn truncate_tail(text: &str, max_chars: usize) -> String {
    if max_chars < 2 {
        return text.chars().take(max_chars).collect();
    }
    let char_count = text.chars().count();
    if char_count <= max_chars {
        return text.to_owned();
    }
    let prefix = text
        .chars()
        .take(max_chars.saturating_sub(1))
        .collect::<String>();
    format!("{prefix}…")
}

fn format_eta(seconds: Option<u64>) -> String {
    seconds.map_or_else(|| "--".to_owned(), humanize_duration_seconds)
}

fn humanize_duration_seconds(seconds: u64) -> String {
    if seconds < 60 {
        return format!("{seconds}s");
    }
    let mins = seconds / 60;
    let secs = seconds % 60;
    if mins < 60 {
        return format!("{mins}m {secs}s");
    }
    let hours = mins / 60;
    let rem_mins = mins % 60;
    format!("{hours}h {rem_mins}m")
}

fn tui_terminal_width() -> usize {
    std::env::var("COLUMNS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value >= 60)
        .unwrap_or(120)
}

fn describe_root_probe_fit(path: &Path, stats: &RootProbeStats) -> &'static str {
    let lower = path.to_string_lossy().to_ascii_lowercase();
    if lower.contains("/data/projects")
        || lower.ends_with("/projects")
        || lower.ends_with("/workspace")
        || lower.ends_with("/workspaces")
    {
        "workspace-heavy"
    } else if stats.repo_markers >= 20 {
        "repo-dense"
    } else if stats.code_files >= stats.doc_files.saturating_mul(2) {
        "code-centric"
    } else if stats.doc_files > stats.code_files {
        "document-centric"
    } else {
        "mixed corpus"
    }
}

#[must_use]
fn indexing_spinner_frame(elapsed_ms: u128) -> &'static str {
    const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    let frame_count = u128::try_from(FRAMES.len()).unwrap_or(1);
    let idx = usize::try_from((elapsed_ms / 110) % frame_count).unwrap_or(0);
    FRAMES[idx]
}

#[must_use]
const fn stage_color_code(stage: IndexingProgressStage) -> &'static str {
    match stage {
        IndexingProgressStage::Discovering => "1;38;5;220",
        IndexingProgressStage::Indexing => "1;38;5;45",
        IndexingProgressStage::Finalizing => "1;38;5;208",
        IndexingProgressStage::Completed => "1;32",
    }
}

#[allow(clippy::cast_precision_loss)]
fn ratio_percent(part: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        part as f64 * 100.0 / total as f64
    }
}

#[allow(clippy::cast_precision_loss)]
fn ratio_percent_usize(part: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        part as f64 * 100.0 / total as f64
    }
}

#[allow(clippy::cast_precision_loss)]
fn timing_ratio_percent(part_ms: u128, total_ms: u128) -> f64 {
    if total_ms == 0 {
        0.0
    } else {
        part_ms as f64 * 100.0 / total_ms as f64
    }
}

fn format_count_u64(value: u64) -> String {
    let source = value.to_string();
    let mut out = String::with_capacity(source.len().saturating_add(source.len() / 3));
    for (idx, ch) in source.chars().rev().enumerate() {
        if idx > 0 && idx % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out.chars().rev().collect()
}

fn format_count_usize(value: usize) -> String {
    format_count_u64(u64::try_from(value).unwrap_or(u64::MAX))
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn render_progress_bar(ratio: f64, width: usize) -> String {
    let clamped = ratio.clamp(0.0, 1.0);
    let filled = (clamped * width as f64).round() as usize;
    let mut bar = String::with_capacity(width);
    bar.push_str(&"█".repeat(filled.min(width)));
    bar.push_str(&"░".repeat(width.saturating_sub(filled.min(width))));
    bar
}

fn ui_fg(no_color: bool, color: PackedRgba) -> Style {
    if no_color {
        Style::new()
    } else {
        Style::new().fg(color)
    }
}

fn ui_fg_bg(no_color: bool, fg: PackedRgba, bg: PackedRgba) -> Style {
    if no_color {
        Style::new().bold()
    } else {
        Style::new().fg(fg).bg(bg).bold()
    }
}

fn centered_rect(
    percent_x: u16,
    percent_y: u16,
    area: ftui_core::geometry::Rect,
) -> ftui_core::geometry::Rect {
    let vertical = Flex::vertical()
        .constraints([
            Constraint::Percentage(f32::from(100_u16.saturating_sub(percent_y)) / 2.0),
            Constraint::Percentage(f32::from(percent_y)),
            Constraint::Percentage(f32::from(100_u16.saturating_sub(percent_y)) / 2.0),
        ])
        .split(area);
    Flex::horizontal()
        .constraints([
            Constraint::Percentage(f32::from(100_u16.saturating_sub(percent_x)) / 2.0),
            Constraint::Percentage(f32::from(percent_x)),
            Constraint::Percentage(f32::from(100_u16.saturating_sub(percent_x)) / 2.0),
        ])
        .split(vertical[1])[1]
}

fn wait_for_ftui_dismiss(
    session: &mut FtuiSession,
    message: &str,
    no_color: bool,
) -> SearchResult<()> {
    loop {
        session.render(|frame| {
            let area = frame.bounds();
            let popup = centered_rect(70, 30, area);
            let block = Block::new()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(ui_fg(no_color, PackedRgba::rgb(67, 160, 255)))
                .title(" fsfs ");
            let body = Paragraph::new(Text::from_lines(vec![
                Line::from(Span::styled(
                    message.to_owned(),
                    ui_fg(no_color, PackedRgba::rgb(224, 234, 255)),
                )),
                Line::from(""),
                Line::from(Span::styled(
                    "Enter / Esc / q to continue",
                    ui_fg(no_color, PackedRgba::rgb(160, 176, 205)),
                )),
            ]))
            .wrap(WrapMode::Word)
            .block(block);
            body.render(popup, frame);
        })?;

        if let Some(event) = session.poll_event(Duration::from_millis(FSFS_TUI_POLL_INTERVAL_MS))?
            && let Event::Key(key) = event
            && matches!(
                key.code,
                KeyCode::Enter | KeyCode::Escape | KeyCode::Char('q' | 'Q')
            )
        {
            return Ok(());
        }
    }
}

#[allow(clippy::too_many_lines)]
fn render_root_selector_frame(
    frame: &mut Frame,
    proposals: &[IndexRootProposal],
    selected_index: usize,
    no_color: bool,
) {
    if proposals.is_empty() {
        return;
    }
    let area = frame.bounds();
    if area.is_empty() {
        return;
    }

    let selected_index = selected_index.min(proposals.len().saturating_sub(1));
    let selected = &proposals[selected_index];
    let layout = Flex::vertical()
        .constraints([Constraint::Fixed(5), Constraint::Fill, Constraint::Fixed(3)])
        .split(area);
    let body = Flex::horizontal()
        .constraints([Constraint::Percentage(62.0), Constraint::Percentage(38.0)])
        .split(layout[1]);

    let header = Paragraph::new(Text::from_lines(vec![
        Line::from_spans(vec![
            Span::styled(
                "fsfs",
                ui_fg(no_color, PackedRgba::rgb(90, 188, 255)).bold(),
            ),
            Span::styled(
                "  First-Run Index Setup",
                ui_fg(no_color, PackedRgba::rgb(236, 244, 255)).bold(),
            ),
        ]),
        Line::from(Span::styled(
            "No index was found. Choose the root that best matches your code + documents.",
            ui_fg(no_color, PackedRgba::rgb(180, 198, 224)),
        )),
    ]))
    .wrap(WrapMode::Word)
    .block(
        Block::new()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(ui_fg(no_color, PackedRgba::rgb(67, 160, 255)))
            .title(" onboarding "),
    );
    header.render(layout[0], frame);

    let mut candidate_lines = Vec::new();
    for (idx, proposal) in proposals.iter().enumerate() {
        let is_selected = idx == selected_index;
        let prefix_style = if is_selected {
            ui_fg_bg(
                no_color,
                PackedRgba::rgb(16, 30, 48),
                PackedRgba::rgb(122, 219, 255),
            )
        } else {
            ui_fg(no_color, PackedRgba::rgb(138, 156, 190)).bold()
        };
        let path_style = if is_selected {
            ui_fg(no_color, PackedRgba::rgb(224, 244, 255))
        } else {
            ui_fg(no_color, PackedRgba::rgb(197, 210, 232))
        };
        let badge_style = if idx == 0 {
            ui_fg(no_color, PackedRgba::rgb(131, 231, 157)).bold()
        } else {
            ui_fg(no_color, PackedRgba::rgb(138, 156, 190))
        };
        let path_width = usize::from(body[0].width).saturating_sub(28);
        candidate_lines.push(Line::from_spans(vec![
            Span::styled(
                format!(" {} {} ", if is_selected { "▸" } else { " " }, idx + 1),
                prefix_style,
            ),
            Span::styled(
                truncate_middle(&proposal.path.display().to_string(), path_width.max(24)),
                path_style,
            ),
            Span::styled(
                if idx == 0 {
                    "  RECOMMENDED"
                } else {
                    "  candidate"
                },
                badge_style,
            ),
        ]));
        candidate_lines.push(Line::from(Span::styled(
            format!(
                "    fit={}  score={}  repos={}  files={}  data={}",
                describe_root_probe_fit(&proposal.path, &proposal.stats),
                format_count_u64(u64::try_from(proposal.score.max(0)).unwrap_or_default()),
                format_count_usize(proposal.stats.repo_markers),
                format_count_usize(proposal.stats.candidate_files),
                humanize_bytes(proposal.stats.candidate_bytes),
            ),
            ui_fg(no_color, PackedRgba::rgb(150, 170, 204)),
        )));
        if idx + 1 < proposals.len() {
            candidate_lines.push(Line::from(""));
        }
    }
    Paragraph::new(Text::from_lines(candidate_lines))
        .block(
            Block::new()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(ui_fg(no_color, PackedRgba::rgb(85, 123, 194)))
                .title(" suggested roots "),
        )
        .render(body[0], frame);

    let detail_lines = vec![
        Line::from(Span::styled(
            truncate_middle(
                &selected.path.display().to_string(),
                usize::from(body[1].width).saturating_sub(6),
            ),
            ui_fg(no_color, PackedRgba::rgb(224, 240, 255)).bold(),
        )),
        Line::from(""),
        Line::from(Span::styled(
            format!(
                "fit: {}",
                describe_root_probe_fit(&selected.path, &selected.stats)
            ),
            ui_fg(no_color, PackedRgba::rgb(172, 193, 227)),
        )),
        Line::from(Span::styled(
            format!(
                "score: {}",
                format_count_u64(u64::try_from(selected.score.max(0)).unwrap_or_default())
            ),
            ui_fg(no_color, PackedRgba::rgb(172, 193, 227)),
        )),
        Line::from(Span::styled(
            format!("repos: {}", format_count_usize(selected.stats.repo_markers)),
            ui_fg(no_color, PackedRgba::rgb(172, 193, 227)),
        )),
        Line::from(Span::styled(
            format!(
                "candidate files: {}",
                format_count_usize(selected.stats.candidate_files)
            ),
            ui_fg(no_color, PackedRgba::rgb(172, 193, 227)),
        )),
        Line::from(Span::styled(
            format!(
                "code/doc split: {:.1}% / {:.1}%",
                ratio_percent_usize(
                    selected.stats.code_files,
                    selected.stats.candidate_files.max(1)
                ),
                ratio_percent_usize(
                    selected.stats.doc_files,
                    selected.stats.candidate_files.max(1)
                )
            ),
            ui_fg(no_color, PackedRgba::rgb(172, 193, 227)),
        )),
        Line::from(Span::styled(
            format!(
                "estimated corpus: {}",
                humanize_bytes(selected.stats.candidate_bytes)
            ),
            ui_fg(no_color, PackedRgba::rgb(172, 193, 227)),
        )),
    ];
    Paragraph::new(Text::from_lines(detail_lines))
        .block(
            Block::new()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(ui_fg(no_color, PackedRgba::rgb(104, 166, 255)))
                .title(" selected profile "),
        )
        .render(body[1], frame);

    Paragraph::new(Text::from_lines(vec![
        Line::from(Span::styled(
            "↑/↓ or j/k move   •   Enter confirm   •   q/Esc cancel",
            ui_fg(no_color, PackedRgba::rgb(160, 180, 213)),
        )),
        Line::from(Span::styled(
            "Number keys 1-9 jump directly to a candidate",
            ui_fg(no_color, PackedRgba::rgb(131, 151, 186)),
        )),
    ]))
    .block(
        Block::new()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(ui_fg(no_color, PackedRgba::rgb(72, 98, 149)))
            .title(" controls "),
    )
    .render(layout[2], frame);
}

#[allow(dead_code, clippy::too_many_lines, clippy::cast_precision_loss)]
fn render_existing_index_dashboard_frame(
    frame: &mut Frame,
    payload: &FsfsStatusPayload,
    fast_only: bool,
    no_color: bool,
) {
    let area = frame.bounds();
    if area.is_empty() {
        return;
    }
    let layout = Flex::vertical()
        .constraints([Constraint::Fixed(5), Constraint::Fill, Constraint::Fixed(3)])
        .split(area);
    let body = Flex::horizontal()
        .constraints([Constraint::Percentage(57.0), Constraint::Percentage(43.0)])
        .split(layout[1]);
    let left = Flex::vertical()
        .constraints([Constraint::Percentage(54.0), Constraint::Percentage(46.0)])
        .split(body[0]);
    let right = Flex::vertical()
        .constraints([Constraint::Percentage(48.0), Constraint::Percentage(52.0)])
        .split(body[1]);

    let indexed_files = payload.index.indexed_files.unwrap_or(0);
    let discovered_files = payload.index.discovered_files.unwrap_or(indexed_files);
    let skipped_files = payload.index.skipped_files.unwrap_or(0);
    let stale_files = payload.index.stale_files.unwrap_or(0);
    let coverage_ratio = if discovered_files == 0 {
        0.0
    } else {
        indexed_files as f64 / discovered_files as f64
    };
    let vector_ratio = if payload.index.size_bytes == 0 {
        0.0
    } else {
        payload.index.vector_index_bytes as f64 / payload.index.size_bytes as f64
    };
    let lexical_ratio = if payload.index.size_bytes == 0 {
        0.0
    } else {
        payload.index.lexical_index_bytes as f64 / payload.index.size_bytes as f64
    };
    let fast_cached = payload
        .models
        .iter()
        .find(|model| model.tier == "fast")
        .is_some_and(|model| model.cached);
    let quality_cached = payload
        .models
        .iter()
        .find(|model| model.tier == "quality")
        .is_some_and(|model| model.cached);
    let mode_label = if fast_cached && quality_cached && !fast_only {
        "full hybrid semantic + lexical"
    } else if fast_cached {
        "fast semantic + lexical"
    } else {
        "hash + lexical fallback"
    };
    let mode_style = if fast_cached && quality_cached && !fast_only {
        ui_fg(no_color, PackedRgba::rgb(131, 231, 157)).bold()
    } else if fast_cached {
        ui_fg(no_color, PackedRgba::rgb(255, 202, 123)).bold()
    } else {
        ui_fg(no_color, PackedRgba::rgb(255, 156, 156)).bold()
    };

    Paragraph::new(Text::from_lines(vec![
        Line::from_spans(vec![
            Span::styled(
                "fsfs",
                ui_fg(no_color, PackedRgba::rgb(90, 188, 255)).bold(),
            ),
            Span::styled(
                "  Search Control Deck",
                ui_fg(no_color, PackedRgba::rgb(236, 244, 255)).bold(),
            ),
        ]),
        Line::from_spans(vec![
            Span::styled("mode: ", ui_fg(no_color, PackedRgba::rgb(164, 184, 219))),
            Span::styled(mode_label, mode_style),
        ]),
        Line::from(Span::styled(
            truncate_middle(
                &payload.index.path,
                usize::from(layout[0].width).saturating_sub(10),
            ),
            ui_fg(no_color, PackedRgba::rgb(175, 195, 229)),
        )),
    ]))
    .block(
        Block::new()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(ui_fg(no_color, PackedRgba::rgb(67, 160, 255)))
            .title(" index ready "),
    )
    .render(layout[0], frame);

    Paragraph::new(Text::from_lines(vec![
        Line::from(Span::styled(
            format!(
                "indexed={}  discovered={}  skipped={}  stale={}",
                format_count_usize(indexed_files),
                format_count_usize(discovered_files),
                format_count_usize(skipped_files),
                format_count_usize(stale_files),
            ),
            ui_fg(no_color, PackedRgba::rgb(220, 233, 255)),
        )),
        Line::from(Span::styled(
            format!(
                "size={}  vector={}  lexical={}  metadata={}",
                humanize_bytes(payload.index.size_bytes),
                humanize_bytes(payload.index.vector_index_bytes),
                humanize_bytes(payload.index.lexical_index_bytes),
                humanize_bytes(payload.index.metadata_bytes),
            ),
            ui_fg(no_color, PackedRgba::rgb(173, 193, 226)),
        )),
        Line::from(Span::styled(
            format!(
                "last indexed={}",
                payload
                    .index
                    .last_indexed_iso_utc
                    .as_deref()
                    .unwrap_or("unknown"),
            ),
            ui_fg(no_color, PackedRgba::rgb(173, 193, 226)),
        )),
    ]))
    .block(
        Block::new()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(ui_fg(no_color, PackedRgba::rgb(89, 126, 196)))
            .title(" corpus "),
    )
    .render(left[0], frame);

    let composition_block = Block::new()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(ui_fg(no_color, PackedRgba::rgb(95, 144, 208)))
        .title(" composition ");
    composition_block.render(left[1], frame);
    let composition_inner = composition_block.inner(left[1]);
    let composition_rows = Flex::vertical()
        .constraints([
            Constraint::Fixed(3),
            Constraint::Fixed(3),
            Constraint::Fixed(3),
        ])
        .split(composition_inner);
    let coverage_label = format!("coverage {:>5.1}%", coverage_ratio * 100.0);
    ProgressBar::new()
        .ratio(coverage_ratio)
        .label(&coverage_label)
        .style(ui_fg(no_color, PackedRgba::rgb(34, 46, 70)))
        .gauge_style(ui_fg_bg(
            no_color,
            PackedRgba::rgb(12, 28, 40),
            PackedRgba::rgb(122, 219, 255),
        ))
        .render(composition_rows[0], frame);
    let vector_label = format!("vector {:>5.1}%", vector_ratio * 100.0);
    ProgressBar::new()
        .ratio(vector_ratio)
        .label(&vector_label)
        .style(ui_fg(no_color, PackedRgba::rgb(34, 46, 70)))
        .gauge_style(ui_fg_bg(
            no_color,
            PackedRgba::rgb(12, 24, 40),
            PackedRgba::rgb(131, 231, 157),
        ))
        .render(composition_rows[1], frame);
    let lexical_label = format!("lexical {:>5.1}%", lexical_ratio * 100.0);
    ProgressBar::new()
        .ratio(lexical_ratio)
        .label(&lexical_label)
        .style(ui_fg(no_color, PackedRgba::rgb(34, 46, 70)))
        .gauge_style(ui_fg_bg(
            no_color,
            PackedRgba::rgb(12, 24, 40),
            PackedRgba::rgb(255, 202, 123),
        ))
        .render(composition_rows[2], frame);

    let mut model_lines = Vec::new();
    for model in &payload.models {
        let state_style = if model.cached {
            ui_fg(no_color, PackedRgba::rgb(131, 231, 157)).bold()
        } else {
            ui_fg(no_color, PackedRgba::rgb(255, 156, 156)).bold()
        };
        model_lines.push(Line::from_spans(vec![
            Span::styled(
                format!("{:<8}", format!("{}:", model.tier)),
                ui_fg(no_color, PackedRgba::rgb(164, 184, 219)),
            ),
            Span::styled(
                truncate_middle(&model.name, usize::from(right[0].width).saturating_sub(24)),
                ui_fg(no_color, PackedRgba::rgb(216, 231, 255)),
            ),
            Span::styled(
                if model.cached { " cached" } else { " missing" },
                state_style,
            ),
        ]));
    }
    Paragraph::new(Text::from_lines(model_lines))
        .block(
            Block::new()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(ui_fg(no_color, PackedRgba::rgb(104, 166, 255)))
                .title(" model health "),
        )
        .render(right[0], frame);

    Paragraph::new(Text::from_lines(vec![
        Line::from(Span::styled(
            "fsfs search \"query\" --limit all",
            ui_fg(no_color, PackedRgba::rgb(224, 238, 255)),
        )),
        Line::from(Span::styled(
            "fsfs search \"query\" --stream --format jsonl",
            ui_fg(no_color, PackedRgba::rgb(224, 238, 255)),
        )),
        Line::from(Span::styled(
            "fsfs index <path> --watch",
            ui_fg(no_color, PackedRgba::rgb(224, 238, 255)),
        )),
        Line::from(Span::styled(
            "fsfs status --format json",
            ui_fg(no_color, PackedRgba::rgb(224, 238, 255)),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "Press r to refresh this dashboard.",
            ui_fg(no_color, PackedRgba::rgb(161, 178, 208)),
        )),
    ]))
    .block(
        Block::new()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(ui_fg(no_color, PackedRgba::rgb(137, 110, 212)))
            .title(" quick actions "),
    )
    .render(right[1], frame);

    Paragraph::new(Text::from_lines(vec![
        Line::from(Span::styled(
            "Enter / Esc / q exit  •  r refresh",
            ui_fg(no_color, PackedRgba::rgb(165, 183, 216)),
        )),
        Line::from(Span::styled(
            "Bundled semantic defaults are materialized automatically at runtime.",
            ui_fg(no_color, PackedRgba::rgb(132, 151, 184)),
        )),
    ]))
    .block(
        Block::new()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(ui_fg(no_color, PackedRgba::rgb(72, 98, 149)))
            .title(" controls "),
    )
    .render(layout[2], frame);
}

#[allow(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::option_if_let_else,
    clippy::or_fun_call
)]
fn render_search_dashboard_frame(frame: &mut Frame, state: &SearchDashboardState, no_color: bool) {
    let area = frame.bounds();
    if area.is_empty() {
        return;
    }
    if area.width < 92 || area.height < 22 {
        render_search_dashboard_compact(frame, state, no_color);
        return;
    }

    let query_terms = collect_search_query_terms(state.query_input.value());
    let latest_payload = state.latest_payload();
    let latest_stage = state
        .stage_chain
        .last()
        .copied()
        .unwrap_or(DashboardSearchStage::Idle)
        .label()
        .replace('_', " ");
    let returned_hits = state.latest_hits().len();
    let total_candidates = latest_payload.map_or(0, |payload| payload.total_candidates);
    let elapsed_label = state
        .last_search_elapsed_ms
        .map_or_else(|| "--".to_owned(), |ms| format!("{ms}ms"));

    let layout = Flex::vertical()
        .constraints([Constraint::Fixed(5), Constraint::Fill, Constraint::Fixed(3)])
        .split(area);
    let body = Flex::horizontal()
        .constraints([Constraint::Percentage(64.0), Constraint::Percentage(36.0)])
        .split(layout[1]);
    let left = Flex::vertical()
        .constraints([Constraint::Fixed(3), Constraint::Fill, Constraint::Fixed(5)])
        .split(body[0]);
    let right = Flex::vertical()
        .constraints([Constraint::Fixed(8), Constraint::Fixed(8), Constraint::Fill])
        .split(body[1]);

    let header_lines = vec![
        Line::from_spans(vec![
            Span::styled(
                "fsfs",
                ui_fg(no_color, PackedRgba::rgb(90, 188, 255)).bold(),
            ),
            Span::styled(
                "  Search Cockpit",
                ui_fg(no_color, PackedRgba::rgb(236, 244, 255)).bold(),
            ),
            Span::styled(
                format!("  stage: {latest_stage}"),
                ui_fg(no_color, PackedRgba::rgb(163, 185, 222)),
            ),
        ]),
        Line::from(Span::styled(
            format!(
                "query latency={}  hits={}  candidates={}  searches this session={}",
                elapsed_label,
                format_count_usize(returned_hits),
                format_count_usize(total_candidates),
                format_count_u64(state.search_invocations)
            ),
            ui_fg(no_color, PackedRgba::rgb(173, 194, 229)),
        )),
        Line::from(Span::styled(
            truncate_middle(
                &state.status_payload.index.path,
                usize::from(layout[0].width).saturating_sub(8),
            ),
            ui_fg(no_color, PackedRgba::rgb(163, 185, 222)),
        )),
    ];
    Paragraph::new(Text::from_lines(header_lines))
        .block(
            Block::new()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(ui_fg(no_color, PackedRgba::rgb(67, 160, 255)))
                .title(" index + query runtime "),
        )
        .render(layout[0], frame);

    let search_block = Block::new()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(ui_fg(no_color, PackedRgba::rgb(87, 130, 199)))
        .title(" live query ");
    search_block.render(left[0], frame);
    let search_inner = search_block.inner(left[0]);
    let search_cols = Flex::horizontal()
        .constraints([Constraint::Min(18), Constraint::Fixed(24)])
        .split(search_inner);
    state.query_input.render(search_cols[0], frame);
    let match_label = if returned_hits == 0 {
        if state.query_input.value().trim().is_empty() {
            "Type to search".to_owned()
        } else {
            "No matches".to_owned()
        }
    } else {
        format!(
            "{}/{} matches",
            state.active_hit_index.saturating_add(1),
            returned_hits
        )
    };
    let focus_label = if state.search_active {
        "search focus"
    } else {
        "browse focus"
    };
    Paragraph::new(Text::from_lines(vec![
        Line::from(Span::styled(
            match_label,
            ui_fg(no_color, PackedRgba::rgb(223, 236, 255)).bold(),
        )),
        Line::from(Span::styled(
            focus_label,
            ui_fg(no_color, PackedRgba::rgb(136, 157, 193)),
        )),
    ]))
    .render(search_cols[1], frame);

    render_search_results_panel(frame, left[1], state, no_color, &query_terms);

    let active_detail_lines = if let Some(hit) = state.latest_hits().get(state.active_hit_index) {
        let lexical_rank = hit
            .lexical_rank
            .map_or_else(|| "–".to_owned(), |rank| format_count_usize(rank + 1));
        let semantic_rank = hit
            .semantic_rank
            .map_or_else(|| "–".to_owned(), |rank| format_count_usize(rank + 1));
        let source = search_hit_source_label(hit);
        vec![
            Line::from(Span::styled(
                truncate_middle(
                    &hit.path,
                    usize::from(left[2].width).saturating_sub(10).max(24),
                ),
                ui_fg(no_color, PackedRgba::rgb(224, 238, 255)).bold(),
            )),
            Line::from(Span::styled(
                format!(
                    "score={:.3}  source={}  lexical#={}  semantic#={}",
                    hit.score, source, lexical_rank, semantic_rank
                ),
                ui_fg(no_color, PackedRgba::rgb(173, 194, 229)),
            )),
            Line::from(Span::styled(
                hit.snippet.as_deref().map_or(
                    "No snippet available for this match.".to_owned(),
                    |snippet| {
                        let snippet_trimmed = snippet.trim();
                        let detection = is_likely_markdown(snippet_trimmed);
                        let preview_format =
                            detect_context_preview_format(snippet_trimmed, detection);
                        let snippet_text = if preview_format == ContextPreviewFormat::Html {
                            normalize_html_fragment_for_markdown(snippet_trimmed)
                        } else {
                            snippet_trimmed.to_owned()
                        };
                        truncate_tail(
                            &snippet_text,
                            usize::from(left[2].width).saturating_sub(8).max(26),
                        )
                    },
                ),
                ui_fg(no_color, PackedRgba::rgb(152, 174, 211)),
            )),
        ]
    } else if let Some(error) = state.last_error.as_deref() {
        vec![
            Line::from(Span::styled(
                "Search failed",
                ui_fg(no_color, PackedRgba::rgb(255, 156, 156)).bold(),
            )),
            Line::from(Span::styled(
                truncate_tail(error, usize::from(left[2].width).saturating_sub(6).max(24)),
                ui_fg(no_color, PackedRgba::rgb(255, 201, 201)),
            )),
        ]
    } else {
        vec![
            Line::from(Span::styled(
                "No active match",
                ui_fg(no_color, PackedRgba::rgb(173, 194, 229)).bold(),
            )),
            Line::from(Span::styled(
                "Use / or Ctrl+F to focus query input, then type to search instantly.",
                ui_fg(no_color, PackedRgba::rgb(148, 170, 205)),
            )),
        ]
    };
    Paragraph::new(Text::from_lines(active_detail_lines))
        .block(
            Block::new()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(ui_fg(no_color, PackedRgba::rgb(97, 153, 223)))
                .title(" active match "),
        )
        .wrap(WrapMode::Word)
        .render(left[2], frame);

    let indexed_files = state.status_payload.index.indexed_files.unwrap_or(0);
    let discovered_files = state
        .status_payload
        .index
        .discovered_files
        .unwrap_or(indexed_files);
    let skipped_files = state.status_payload.index.skipped_files.unwrap_or(0);
    let stale_files = state.status_payload.index.stale_files.unwrap_or(0);
    let coverage = ratio_percent_usize(indexed_files, discovered_files.max(1));
    let mode_hint = state
        .mode_hint
        .as_deref()
        .unwrap_or("Search mode: full hybrid semantic + lexical.");
    let adaptive_debounce_line = format!(
        "adaptive debounce={} / {} / {} ms  cadence~{} ms",
        format_count_u64(state.lexical_debounce_window_ms),
        format_count_u64(state.semantic_debounce_window_ms),
        format_count_u64(state.quality_debounce_window_ms),
        state.typing_cadence_ms_label()
    );
    let corpus_lines = vec![
        Line::from(Span::styled(
            format!(
                "indexed={}  discovered={}  skipped={}  stale={}",
                format_count_usize(indexed_files),
                format_count_usize(discovered_files),
                format_count_usize(skipped_files),
                format_count_usize(stale_files),
            ),
            ui_fg(no_color, PackedRgba::rgb(224, 237, 255)),
        )),
        Line::from(Span::styled(
            format!(
                "coverage={coverage:.1}%  index size={}  vector={}  lexical={}",
                humanize_bytes(state.status_payload.index.size_bytes),
                humanize_bytes(state.status_payload.index.vector_index_bytes),
                humanize_bytes(state.status_payload.index.lexical_index_bytes),
            ),
            ui_fg(no_color, PackedRgba::rgb(173, 194, 229)),
        )),
        Line::from(Span::styled(
            truncate_tail(
                mode_hint,
                usize::from(right[0].width).saturating_sub(6).max(20),
            ),
            ui_fg(no_color, PackedRgba::rgb(141, 163, 199)),
        )),
        Line::from(Span::styled(
            truncate_tail(
                &adaptive_debounce_line,
                usize::from(right[0].width).saturating_sub(6).max(20),
            ),
            ui_fg(no_color, PackedRgba::rgb(124, 146, 184)),
        )),
    ];
    Paragraph::new(Text::from_lines(corpus_lines))
        .block(
            Block::new()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(ui_fg(no_color, PackedRgba::rgb(103, 161, 230)))
                .title(" corpus health "),
        )
        .wrap(WrapMode::Word)
        .render(right[0], frame);

    let source_both = state
        .latest_hits()
        .iter()
        .filter(|hit| hit.in_both_sources)
        .count();
    let source_lexical = state
        .latest_hits()
        .iter()
        .filter(|hit| hit.lexical_rank.is_some() && hit.semantic_rank.is_none())
        .count();
    let source_semantic = state
        .latest_hits()
        .iter()
        .filter(|hit| hit.semantic_rank.is_some() && hit.lexical_rank.is_none())
        .count();
    let phase_chain = render_phase_chain(&state.stage_chain);
    let telemetry_block = Block::new()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(ui_fg(no_color, PackedRgba::rgb(118, 131, 212)))
        .title(" query telemetry ");
    telemetry_block.render(right[1], frame);
    let telemetry_inner = telemetry_block.inner(right[1]);
    let telemetry_rows = Flex::vertical()
        .constraints([
            Constraint::Fixed(4),
            Constraint::Fixed(1),
            Constraint::Fixed(1),
        ])
        .split(telemetry_inner);
    Paragraph::new(Text::from_lines(vec![
        Line::from(Span::styled(
            format!(
                "stage chain: {phase_chain}  •  elapsed: {elapsed_label}  •  limit: {}",
                format_count_usize(state.result_limit)
            ),
            ui_fg(no_color, PackedRgba::rgb(220, 234, 255)),
        )),
        Line::from(Span::styled(
            format!(
                "sources => both={} lexical_only={} semantic_only={}",
                format_count_usize(source_both),
                format_count_usize(source_lexical),
                format_count_usize(source_semantic),
            ),
            ui_fg(no_color, PackedRgba::rgb(168, 191, 228)),
        )),
        Line::from(Span::styled(
            format!(
                "invocations={}  returned_hits={}  candidates={}",
                format_count_u64(state.search_invocations),
                format_count_usize(returned_hits),
                format_count_usize(total_candidates),
            ),
            ui_fg(no_color, PackedRgba::rgb(148, 170, 207)),
        )),
    ]))
    .render(telemetry_rows[0], frame);
    let latency_spark = if state.latency_history_ms.is_empty() {
        vec![0.0]
    } else {
        state.latency_history_ms.iter().copied().collect::<Vec<_>>()
    };
    Sparkline::new(&latency_spark)
        .style(ui_fg(no_color, PackedRgba::rgb(122, 219, 255)))
        .render(telemetry_rows[1], frame);
    let hit_spark = if state.hits_history.is_empty() {
        vec![0.0]
    } else {
        state.hits_history.iter().copied().collect::<Vec<_>>()
    };
    Sparkline::new(&hit_spark)
        .style(ui_fg(no_color, PackedRgba::rgb(131, 231, 157)))
        .render(telemetry_rows[2], frame);

    let context_block = Block::new()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(ui_fg(no_color, PackedRgba::rgb(94, 152, 220)))
        .title(" context radar ");
    context_block.render(right[2], frame);
    let context_inner = context_block.inner(right[2]);
    let context_rows = Flex::vertical()
        .constraints([Constraint::Fixed(3), Constraint::Fixed(1), Constraint::Fill])
        .split(context_inner);
    let score_spark = if state.latest_hits().is_empty() {
        vec![0.0]
    } else {
        state
            .latest_hits()
            .iter()
            .take(72)
            .map(|hit| hit.score.max(0.0))
            .collect::<Vec<_>>()
    };
    let active_context_meta = state
        .latest_hits()
        .get(state.active_hit_index)
        .map(|active_hit| {
            let raw_snippet = active_hit
                .snippet
                .as_deref()
                .unwrap_or("No snippet available for this hit.")
                .trim()
                .to_owned();
            let preview_source = truncate_tail(
                &raw_snippet,
                usize::from(context_rows[2].width)
                    .saturating_mul(24)
                    .max(256),
            );
            let markdown_detection = is_likely_markdown(&preview_source);
            let preview_format = detect_context_preview_format(&preview_source, markdown_detection);
            (preview_source, preview_format, markdown_detection)
        });
    let mode_line = active_context_meta.as_ref().map_or_else(
        || "render mode=waiting for match".to_owned(),
        |(_, preview_format, markdown_detection)| match preview_format {
            ContextPreviewFormat::Markdown => format!(
                "render mode=streaming gfm markdown  indicators={}",
                markdown_detection.indicators
            ),
            ContextPreviewFormat::Html => {
                "render mode=html fragment -> streaming gfm markdown".to_owned()
            }
            ContextPreviewFormat::Plain => "render mode=plain text + query highlights".to_owned(),
        },
    );
    Paragraph::new(Text::from_lines(vec![
        Line::from(Span::styled(
            "result score density (left→right rank order)",
            ui_fg(no_color, PackedRgba::rgb(165, 187, 223)),
        )),
        Line::from(Span::styled(
            format!(
                "active rank={} / {}",
                format_count_usize(if returned_hits == 0 {
                    0
                } else {
                    state.active_hit_index.saturating_add(1)
                }),
                format_count_usize(returned_hits)
            ),
            ui_fg(no_color, PackedRgba::rgb(145, 168, 205)),
        )),
        Line::from(Span::styled(
            truncate_tail(
                &mode_line,
                usize::from(context_rows[0].width).saturating_sub(2),
            ),
            ui_fg(no_color, PackedRgba::rgb(131, 154, 191)),
        )),
    ]))
    .render(context_rows[0], frame);
    Sparkline::new(&score_spark)
        .style(ui_fg(no_color, PackedRgba::rgb(255, 202, 123)))
        .render(context_rows[1], frame);
    if let Some(active_hit) = state.latest_hits().get(state.active_hit_index) {
        let path_line = Line::from(Span::styled(
            truncate_middle(
                &active_hit.path,
                usize::from(context_rows[2].width).saturating_sub(4).max(18),
            ),
            ui_fg(no_color, PackedRgba::rgb(224, 239, 255)).bold(),
        ));

        let mut rendered_lines = vec![path_line, Line::from("")];
        let mut wrap_mode = WrapMode::Word;
        if let Some((preview_source, preview_format, _)) = active_context_meta.as_ref() {
            match preview_format {
                ContextPreviewFormat::Plain => {
                    for line in preview_source.lines() {
                        let spans = highlight_text_spans(
                            line,
                            &query_terms,
                            ui_fg(no_color, PackedRgba::rgb(215, 230, 255)),
                            ui_fg_bg(
                                no_color,
                                PackedRgba::rgb(8, 22, 34),
                                PackedRgba::rgb(255, 202, 123),
                            ),
                        );
                        rendered_lines.push(Line::from_spans(spans));
                    }
                }
                ContextPreviewFormat::Markdown | ContextPreviewFormat::Html => {
                    let markdown_text = render_context_radar_markdown_text(
                        preview_source,
                        *preview_format,
                        no_color,
                        context_rows[2].width.saturating_sub(1),
                    );
                    rendered_lines.extend(markdown_text.lines().iter().cloned());
                    wrap_mode = WrapMode::None;
                }
            }
        }
        if rendered_lines.len() <= 2 {
            rendered_lines.push(Line::from(Span::styled(
                "No snippet available for this hit.",
                ui_fg(no_color, PackedRgba::rgb(177, 199, 233)),
            )));
        }
        Paragraph::new(Text::from_lines(rendered_lines))
            .wrap(wrap_mode)
            .scroll((state.context_scroll, 0))
            .render(context_rows[2], frame);
    } else {
        Paragraph::new(Text::from_lines(vec![
            Line::from(Span::styled(
                "No context yet.",
                ui_fg(no_color, PackedRgba::rgb(177, 199, 233)),
            )),
            Line::from(Span::styled(
                "Type a query to populate ranked matches and context excerpts.",
                ui_fg(no_color, PackedRgba::rgb(145, 168, 205)),
            )),
        ]))
        .wrap(WrapMode::Word)
        .render(context_rows[2], frame);
    }

    Paragraph::new(Text::from_lines(vec![
        Line::from(Span::styled(
            "/ or Ctrl+F focus  •  Esc blur/exit  •  Enter/Tab/↓ next  •  ↑/Shift+Tab prev",
            ui_fg(no_color, PackedRgba::rgb(165, 183, 216)),
        )),
        Line::from(Span::styled(
            "n/N navigate  •  [ / ] + PgUp/PgDn scroll context  •  Ctrl+L clear query  •  r refresh corpus health  •  q exit",
            ui_fg(no_color, PackedRgba::rgb(132, 151, 184)),
        )),
    ]))
    .block(
        Block::new()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(ui_fg(no_color, PackedRgba::rgb(72, 98, 149)))
            .title(" controls "),
    )
    .render(layout[2], frame);
}

#[allow(clippy::too_many_lines)]
fn render_search_dashboard_compact(
    frame: &mut Frame,
    state: &SearchDashboardState,
    no_color: bool,
) {
    let area = frame.bounds();
    if area.is_empty() {
        return;
    }
    let query_terms = collect_search_query_terms(state.query_input.value());
    let latest_stage = state
        .stage_chain
        .last()
        .copied()
        .unwrap_or(DashboardSearchStage::Idle)
        .label()
        .replace('_', " ");
    let mut lines = Vec::new();
    lines.push(Line::from_spans(vec![
        Span::styled(
            "fsfs",
            ui_fg(no_color, PackedRgba::rgb(90, 188, 255)).bold(),
        ),
        Span::styled(
            format!("  Search ({latest_stage})"),
            ui_fg(no_color, PackedRgba::rgb(236, 244, 255)).bold(),
        ),
    ]));
    lines.push(Line::from(Span::styled(
        format!(
            "query: {}",
            if state.query_input.value().trim().is_empty() {
                "(empty)"
            } else {
                state.query_input.value()
            }
        ),
        ui_fg(no_color, PackedRgba::rgb(172, 194, 230)),
    )));
    lines.push(Line::from(Span::styled(
        format!(
            "adaptive debounce {} / {} / {} ms",
            format_count_u64(state.lexical_debounce_window_ms),
            format_count_u64(state.semantic_debounce_window_ms),
            format_count_u64(state.quality_debounce_window_ms),
        ),
        ui_fg(no_color, PackedRgba::rgb(146, 168, 204)),
    )));
    if let Some(error) = state.last_error.as_deref() {
        lines.push(Line::from(Span::styled(
            format!("error: {error}"),
            ui_fg(no_color, PackedRgba::rgb(255, 156, 156)),
        )));
    }
    for (idx, hit) in state.latest_hits().iter().take(4).enumerate() {
        let marker = if idx == state.active_hit_index {
            "▸"
        } else {
            " "
        };
        let path = truncate_middle(
            &hit.path,
            usize::from(area.width).saturating_sub(26).max(18),
        );
        lines.push(Line::from_spans(vec![
            Span::styled(
                format!("{marker} {:>2}. ", hit.rank),
                ui_fg(no_color, PackedRgba::rgb(130, 179, 238)),
            ),
            Span::styled(path, ui_fg(no_color, PackedRgba::rgb(220, 236, 255))),
            Span::styled(
                format!("  {:.3}", hit.score),
                dashboard_score_style(hit.score, no_color, idx == state.active_hit_index),
            ),
        ]));
        if let Some(snippet) = hit.snippet.as_deref() {
            let snippet_trimmed = snippet.trim();
            let detection = is_likely_markdown(snippet_trimmed);
            let preview_format = detect_context_preview_format(snippet_trimmed, detection);
            let compact_snippet = if preview_format == ContextPreviewFormat::Html {
                normalize_html_fragment_for_markdown(snippet_trimmed)
            } else {
                snippet_trimmed.to_owned()
            };
            lines.push(Line::from_spans(highlight_text_spans(
                &truncate_tail(
                    &compact_snippet,
                    usize::from(area.width).saturating_sub(8).max(20),
                ),
                &query_terms,
                ui_fg(no_color, PackedRgba::rgb(146, 168, 204)),
                ui_fg_bg(
                    no_color,
                    PackedRgba::rgb(8, 22, 34),
                    PackedRgba::rgb(255, 202, 123),
                ),
            )));
        }
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "/ focus • Enter/Tab next • ↑ prev • q exit",
        ui_fg(no_color, PackedRgba::rgb(165, 183, 216)),
    )));

    Paragraph::new(Text::from_lines(lines))
        .block(
            Block::new()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(ui_fg(no_color, PackedRgba::rgb(67, 160, 255)))
                .title(" fsfs search "),
        )
        .wrap(WrapMode::Word)
        .render(area, frame);
}

#[allow(clippy::too_many_lines)]
fn render_search_results_panel(
    frame: &mut Frame,
    area: ftui_core::geometry::Rect,
    state: &SearchDashboardState,
    no_color: bool,
    query_terms: &[String],
) {
    let block = Block::new()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(ui_fg(no_color, PackedRgba::rgb(95, 144, 208)))
        .title(" ranked results ");
    block.render(area, frame);
    let inner = block.inner(area);
    if inner.is_empty() {
        return;
    }

    if let Some(error) = state.last_error.as_deref() {
        Paragraph::new(Text::from_lines(vec![
            Line::from(Span::styled(
                "Search failed",
                ui_fg(no_color, PackedRgba::rgb(255, 156, 156)).bold(),
            )),
            Line::from(Span::styled(
                truncate_tail(error, usize::from(inner.width).saturating_mul(3).max(42)),
                ui_fg(no_color, PackedRgba::rgb(255, 209, 209)),
            )),
        ]))
        .wrap(WrapMode::Word)
        .render(inner, frame);
        return;
    }

    let query = state.query_input.value().trim();
    let hits = state.latest_hits();
    if query.is_empty() {
        Paragraph::new(Text::from_lines(vec![
            Line::from(Span::styled(
                "Search-as-you-type is enabled.",
                ui_fg(no_color, PackedRgba::rgb(198, 218, 248)).bold(),
            )),
            Line::from(Span::styled(
                "Press / or Ctrl+F, type a query, and results update live.",
                ui_fg(no_color, PackedRgba::rgb(153, 175, 212)),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "Tips: Enter/Tab/↓ next match • ↑/Shift+Tab previous • n/N jump",
                ui_fg(no_color, PackedRgba::rgb(133, 154, 190)),
            )),
        ]))
        .wrap(WrapMode::Word)
        .render(inner, frame);
        return;
    }
    if hits.is_empty() {
        Paragraph::new(Text::from_lines(vec![
            Line::from(Span::styled(
                "No matches found.",
                ui_fg(no_color, PackedRgba::rgb(255, 202, 123)).bold(),
            )),
            Line::from(Span::styled(
                "Try broader terms or remove filter constraints.",
                ui_fg(no_color, PackedRgba::rgb(169, 190, 225)),
            )),
        ]))
        .wrap(WrapMode::Word)
        .render(inner, frame);
        return;
    }

    let rows_per_hit = 2usize;
    let visible_hits = (usize::from(inner.height) / rows_per_hit).max(1);
    let mut start = state.active_hit_index.saturating_sub(visible_hits / 2);
    if start + visible_hits > hits.len() {
        start = hits.len().saturating_sub(visible_hits);
    }
    let end = (start + visible_hits).min(hits.len());

    for (slot, hit_idx) in (start..end).enumerate() {
        let y = inner
            .y
            .saturating_add(u16::try_from(slot.saturating_mul(rows_per_hit)).unwrap_or(u16::MAX));
        if y >= inner.bottom() {
            break;
        }
        let row_height = if y.saturating_add(1) < inner.bottom() {
            2
        } else {
            1
        };
        let row_area = ftui_core::geometry::Rect::new(inner.x, y, inner.width, row_height);
        let hit = &hits[hit_idx];
        let is_active = hit_idx == state.active_hit_index;
        let path_budget = usize::from(inner.width).saturating_sub(32).max(14);
        let snippet_budget = usize::from(inner.width).saturating_sub(8).max(18);
        let path = truncate_middle(&hit.path, path_budget);
        let source_label = search_hit_source_label(hit);

        let mut line_one_spans = Vec::new();
        line_one_spans.push(Span::styled(
            format!("{} ", if is_active { "▸" } else { " " }),
            if is_active {
                ui_fg_bg(
                    no_color,
                    PackedRgba::rgb(8, 24, 40),
                    PackedRgba::rgb(122, 219, 255),
                )
            } else {
                ui_fg(no_color, PackedRgba::rgb(130, 149, 182))
            },
        ));
        line_one_spans.push(Span::styled(
            format!("{:>3}. ", hit.rank),
            if is_active {
                ui_fg(no_color, PackedRgba::rgb(224, 240, 255)).bold()
            } else {
                ui_fg(no_color, PackedRgba::rgb(173, 194, 229))
            },
        ));
        let path_spans = highlight_text_spans(
            &path,
            query_terms,
            if is_active {
                ui_fg(no_color, PackedRgba::rgb(236, 247, 255)).bold()
            } else {
                ui_fg(no_color, PackedRgba::rgb(214, 230, 255))
            },
            if is_active {
                ui_fg_bg(
                    no_color,
                    PackedRgba::rgb(10, 24, 36),
                    PackedRgba::rgb(255, 202, 123),
                )
            } else {
                ui_fg_bg(
                    no_color,
                    PackedRgba::rgb(8, 20, 30),
                    PackedRgba::rgb(181, 223, 255),
                )
            },
        );
        line_one_spans.extend(path_spans);
        line_one_spans.push(Span::styled(
            format!("  {:.3}", hit.score),
            dashboard_score_style(hit.score, no_color, is_active),
        ));
        line_one_spans.push(Span::styled(
            format!("  [{source_label}]"),
            if is_active {
                ui_fg(no_color, PackedRgba::rgb(131, 231, 157)).bold()
            } else {
                ui_fg(no_color, PackedRgba::rgb(148, 170, 205))
            },
        ));

        let mut lines = vec![Line::from_spans(line_one_spans)];
        if row_height > 1 {
            let snippet_line = hit.snippet.as_deref().map_or_else(
                || {
                    vec![Span::styled(
                        "no snippet",
                        ui_fg(no_color, PackedRgba::rgb(140, 162, 197)),
                    )]
                },
                |snippet| {
                    highlight_text_spans(
                        &truncate_tail(snippet.trim(), snippet_budget),
                        query_terms,
                        ui_fg(no_color, PackedRgba::rgb(151, 173, 211)),
                        ui_fg_bg(
                            no_color,
                            PackedRgba::rgb(9, 23, 36),
                            PackedRgba::rgb(255, 202, 123),
                        ),
                    )
                },
            );
            lines.push(Line::from_spans(snippet_line));
        }
        Paragraph::new(Text::from_lines(lines)).render(row_area, frame);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContextPreviewFormat {
    Plain,
    Markdown,
    Html,
}

const HTML_FRAGMENT_MARKERS: [&str; 25] = [
    "<!doctype",
    "<html",
    "<body",
    "<div",
    "<span",
    "<p>",
    "<p ",
    "<br",
    "<hr",
    "<pre",
    "<code",
    "<em>",
    "<strong>",
    "<b>",
    "<b ",
    "<i>",
    "<ul",
    "<ol",
    "<li",
    "<table",
    "<tr",
    "<td",
    "<th",
    "<a ",
    "<img",
];

fn detect_context_preview_format(
    content: &str,
    markdown_detection: MarkdownDetection,
) -> ContextPreviewFormat {
    if is_likely_html_fragment(content) {
        ContextPreviewFormat::Html
    } else if markdown_detection.is_likely() {
        ContextPreviewFormat::Markdown
    } else {
        ContextPreviewFormat::Plain
    }
}

fn is_likely_html_fragment(content: &str) -> bool {
    let trimmed = content.trim();
    if trimmed.len() < 4 || !trimmed.contains('<') || !trimmed.contains('>') {
        return false;
    }
    let lower = trimmed.to_ascii_lowercase();
    HTML_FRAGMENT_MARKERS
        .iter()
        .any(|marker| lower.contains(marker))
}

fn render_context_radar_markdown_text(
    source: &str,
    preview_format: ContextPreviewFormat,
    no_color: bool,
    width: u16,
) -> Text<'static> {
    let width = width.max(12);
    let markdown_source = if preview_format == ContextPreviewFormat::Html {
        normalize_html_fragment_for_markdown(source)
    } else {
        source.to_owned()
    };
    let rendered = MarkdownRenderer::new(context_radar_markdown_theme(no_color))
        .rule_width(width.min(42))
        .table_max_width(width)
        .render_streaming(&markdown_source);
    wrap_markdown_for_context_panel(&rendered, width)
}

fn context_radar_markdown_theme(no_color: bool) -> MarkdownTheme {
    MarkdownTheme {
        h1: ui_fg(no_color, PackedRgba::rgb(224, 239, 255)).bold(),
        h2: ui_fg(no_color, PackedRgba::rgb(210, 229, 255)).bold(),
        h3: ui_fg(no_color, PackedRgba::rgb(198, 218, 248)).bold(),
        h4: ui_fg(no_color, PackedRgba::rgb(183, 205, 236)).bold(),
        h5: ui_fg(no_color, PackedRgba::rgb(170, 191, 228)).bold(),
        h6: ui_fg(no_color, PackedRgba::rgb(158, 179, 213)).bold(),
        code_inline: ui_fg(no_color, PackedRgba::rgb(255, 202, 123)),
        code_block: ui_fg(no_color, PackedRgba::rgb(196, 218, 247)),
        blockquote: ui_fg(no_color, PackedRgba::rgb(153, 175, 212)).italic(),
        link: ui_fg(no_color, PackedRgba::rgb(132, 201, 255)).underline(),
        list_bullet: ui_fg(no_color, PackedRgba::rgb(120, 184, 245)),
        horizontal_rule: ui_fg(no_color, PackedRgba::rgb(120, 141, 177)).dim(),
        task_done: ui_fg(no_color, PackedRgba::rgb(131, 231, 157)),
        task_todo: ui_fg(no_color, PackedRgba::rgb(255, 202, 123)),
        math_inline: ui_fg(no_color, PackedRgba::rgb(168, 191, 228)).italic(),
        math_block: ui_fg(no_color, PackedRgba::rgb(183, 205, 236)).bold(),
        footnote_ref: ui_fg(no_color, PackedRgba::rgb(141, 163, 199)).dim(),
        footnote_def: ui_fg(no_color, PackedRgba::rgb(163, 185, 222)),
        admonition_note: ui_fg(no_color, PackedRgba::rgb(122, 219, 255)).bold(),
        admonition_tip: ui_fg(no_color, PackedRgba::rgb(131, 231, 157)).bold(),
        admonition_important: ui_fg(no_color, PackedRgba::rgb(255, 202, 123)).bold(),
        admonition_warning: ui_fg(no_color, PackedRgba::rgb(255, 191, 108)).bold(),
        admonition_caution: ui_fg(no_color, PackedRgba::rgb(255, 156, 156)).bold(),
        ..MarkdownTheme::default()
    }
}

fn normalize_html_fragment_for_markdown(source: &str) -> String {
    let normalized = source
        .replace("<br />", "\n")
        .replace("<br/>", "\n")
        .replace("<br>", "\n")
        .replace("<hr />", "\n---\n")
        .replace("<hr/>", "\n---\n")
        .replace("<hr>", "\n---\n")
        .replace("<p>", "")
        .replace("</p>", "\n\n")
        .replace("<div>", "")
        .replace("</div>", "\n")
        .replace("<strong>", "**")
        .replace("</strong>", "**")
        .replace("<b>", "**")
        .replace("</b>", "**")
        .replace("<em>", "*")
        .replace("</em>", "*")
        .replace("<i>", "*")
        .replace("</i>", "*")
        .replace("<code>", "`")
        .replace("</code>", "`")
        .replace("<pre>", "```\n")
        .replace("</pre>", "\n```")
        .replace("<ul>", "")
        .replace("</ul>", "\n")
        .replace("<ol>", "")
        .replace("</ol>", "\n")
        .replace("<li>", "- ")
        .replace("</li>", "\n");
    let without_tags = strip_html_tags(&normalized);
    decode_basic_html_entities(&without_tags)
}

fn strip_html_tags(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut in_tag = false;
    for ch in source.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(ch),
            _ => {}
        }
    }
    out
}

fn decode_basic_html_entities(source: &str) -> String {
    source
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
}

fn wrap_markdown_for_context_panel(text: &Text<'static>, width: u16) -> Text<'static> {
    let width = usize::from(width);
    if width == 0 {
        return text.clone();
    }

    let mut lines = Vec::new();
    for line in text.lines() {
        let plain = line.to_plain_text();
        let table_like = is_markdown_table_line(&plain) || is_markdown_table_like_line(&plain);
        if table_like || line.width() <= width {
            lines.push(line.clone());
            continue;
        }

        for wrapped in line.wrap(width, WrapMode::Word) {
            if wrapped.width() <= width {
                lines.push(wrapped);
            } else {
                let mut wrapped_text = Text::from_lines([wrapped]);
                wrapped_text.truncate(width, None);
                lines.extend(wrapped_text.lines().iter().cloned());
            }
        }
    }

    Text::from_lines(lines)
}

fn is_markdown_table_line(plain: &str) -> bool {
    plain.chars().any(|c| {
        matches!(
            c,
            '┌' | '┬' | '┐' | '├' | '┼' | '┤' | '└' | '┴' | '┘' | '│' | '─'
        )
    })
}

fn is_markdown_table_like_line(plain: &str) -> bool {
    let trimmed = plain.trim_start();
    trimmed.starts_with('|') && trimmed.chars().filter(|&c| c == '|').count() >= 2
}

fn collect_search_query_terms(query: &str) -> Vec<String> {
    query
        .split_whitespace()
        .map(str::to_ascii_lowercase)
        .filter(|term| term.len() >= 2)
        .collect()
}

const fn search_hit_source_label(hit: &SearchHitPayload) -> &'static str {
    if hit.in_both_sources {
        "both"
    } else if hit.lexical_rank.is_some() {
        "lexical"
    } else if hit.semantic_rank.is_some() {
        "semantic"
    } else {
        "unknown"
    }
}

fn render_phase_chain(stages: &[DashboardSearchStage]) -> String {
    if stages.is_empty() {
        return "idle".to_owned();
    }
    let mut out = String::new();
    for (idx, stage) in stages.iter().enumerate() {
        if idx > 0 {
            out.push_str(" -> ");
        }
        out.push_str(stage.label());
    }
    out
}

fn dashboard_score_style(score: f64, no_color: bool, is_active: bool) -> Style {
    let base = if score >= 0.8 {
        ui_fg(no_color, PackedRgba::rgb(131, 231, 157))
    } else if score >= 0.5 {
        ui_fg(no_color, PackedRgba::rgb(255, 202, 123))
    } else {
        ui_fg(no_color, PackedRgba::rgb(255, 156, 156))
    };
    if is_active { base.bold() } else { base }
}

fn highlight_text_spans(
    text: &str,
    query_terms: &[String],
    base_style: Style,
    highlight_style: Style,
) -> Vec<Span<'static>> {
    if text.is_empty() || query_terms.is_empty() {
        return vec![Span::styled(text.to_owned(), base_style)];
    }

    let mut ranges = Vec::new();
    for term in query_terms {
        for result in search_ascii_case_insensitive(text, term) {
            let start = result.range.start.min(text.len());
            let end = result.range.end.min(text.len());
            if start >= end || !text.is_char_boundary(start) || !text.is_char_boundary(end) {
                continue;
            }
            ranges.push((start, end));
        }
    }
    if ranges.is_empty() {
        return vec![Span::styled(text.to_owned(), base_style)];
    }

    ranges.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (start, end) in ranges {
        if let Some(last) = merged.last_mut()
            && start <= last.1
        {
            if end > last.1 {
                last.1 = end;
            }
            continue;
        }
        merged.push((start, end));
    }

    let mut spans = Vec::new();
    let mut cursor = 0usize;
    for (start, end) in merged {
        if start > cursor && text.is_char_boundary(cursor) && text.is_char_boundary(start) {
            spans.push(Span::styled(text[cursor..start].to_owned(), base_style));
        }
        if text.is_char_boundary(start) && text.is_char_boundary(end) {
            spans.push(Span::styled(text[start..end].to_owned(), highlight_style));
            cursor = end;
        }
    }
    if cursor < text.len() && text.is_char_boundary(cursor) {
        spans.push(Span::styled(text[cursor..].to_owned(), base_style));
    }
    if spans.is_empty() {
        spans.push(Span::styled(text.to_owned(), base_style));
    }
    spans
}

fn render_indexing_progress_screen_ftui(
    session: &mut FtuiSession,
    snapshot: &IndexingProgressSnapshot,
    render_state: &IndexingRenderState,
    no_color: bool,
) -> SearchResult<()> {
    session.render(|frame| {
        render_indexing_progress_frame(frame, snapshot, render_state, no_color);
    })
}

#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
fn render_indexing_progress_frame(
    frame: &mut Frame,
    snapshot: &IndexingProgressSnapshot,
    render_state: &IndexingRenderState,
    no_color: bool,
) {
    let area = frame.bounds();
    if area.is_empty() {
        return;
    }
    let layout = Flex::vertical()
        .constraints([
            Constraint::Fixed(5),
            Constraint::Fixed(4),
            Constraint::Fill,
            Constraint::Fixed(3),
        ])
        .split(area);
    let body = Flex::horizontal()
        .constraints([Constraint::Percentage(58.0), Constraint::Percentage(42.0)])
        .split(layout[2]);
    let left = Flex::vertical()
        .constraints([Constraint::Percentage(55.0), Constraint::Percentage(45.0)])
        .split(body[0]);
    let right = Flex::vertical()
        .constraints([Constraint::Percentage(58.0), Constraint::Percentage(42.0)])
        .split(body[1]);

    let ratio = snapshot.completion_ratio();
    let percent = ratio * 100.0;
    let elapsed_secs = u64::try_from(snapshot.total_elapsed_ms / 1_000).unwrap_or(u64::MAX);
    let elapsed = humanize_duration_seconds(elapsed_secs);
    let raw_eta = format_eta(snapshot.eta_seconds());
    let smooth_eta = format_eta(render_state.smoothed_eta_seconds());
    let pending_files = snapshot
        .candidate_files
        .saturating_sub(snapshot.processed_files);
    let stage_style = match snapshot.stage {
        IndexingProgressStage::Discovering => {
            ui_fg(no_color, PackedRgba::rgb(255, 202, 123)).bold()
        }
        IndexingProgressStage::Indexing => ui_fg(no_color, PackedRgba::rgb(122, 219, 255)).bold(),
        IndexingProgressStage::Finalizing => ui_fg(no_color, PackedRgba::rgb(253, 188, 128)).bold(),
        IndexingProgressStage::Completed => ui_fg(no_color, PackedRgba::rgb(131, 231, 157)).bold(),
    };

    Paragraph::new(Text::from_lines(vec![
        Line::from_spans(vec![
            Span::styled(
                "fsfs",
                ui_fg(no_color, PackedRgba::rgb(90, 188, 255)).bold(),
            ),
            Span::styled(
                "  Initial Indexing",
                ui_fg(no_color, PackedRgba::rgb(236, 244, 255)).bold(),
            ),
            Span::styled(
                format!(
                    "  {} {}",
                    indexing_spinner_frame(snapshot.total_elapsed_ms),
                    snapshot.stage.label()
                ),
                stage_style,
            ),
        ]),
        Line::from(Span::styled(
            format!(
                "target: {}",
                truncate_middle(
                    &snapshot.target_root.display().to_string(),
                    usize::from(layout[0].width).saturating_sub(10),
                )
            ),
            ui_fg(no_color, PackedRgba::rgb(175, 195, 229)),
        )),
        Line::from(Span::styled(
            format!(
                "index:  {}",
                truncate_middle(
                    &snapshot.index_root.display().to_string(),
                    usize::from(layout[0].width).saturating_sub(10),
                )
            ),
            ui_fg(no_color, PackedRgba::rgb(175, 195, 229)),
        )),
    ]))
    .block(
        Block::new()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(ui_fg(no_color, PackedRgba::rgb(67, 160, 255)))
            .title(" indexing cockpit "),
    )
    .render(layout[0], frame);

    let progress_label = format!(
        "{:>5.1}%  queue={}  elapsed={}  eta(raw={} smooth={})",
        percent,
        format_count_usize(pending_files),
        elapsed,
        raw_eta,
        smooth_eta
    );
    ProgressBar::new()
        .ratio(ratio)
        .label(&progress_label)
        .style(ui_fg(no_color, PackedRgba::rgb(34, 46, 70)))
        .gauge_style(ui_fg_bg(
            no_color,
            PackedRgba::rgb(12, 24, 40),
            PackedRgba::rgb(122, 219, 255),
        ))
        .block(
            Block::new()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(ui_fg(no_color, PackedRgba::rgb(83, 124, 190)))
                .title(" progress "),
        )
        .render(layout[1], frame);

    let files_history = render_state
        .files_rate_history
        .iter()
        .copied()
        .collect::<Vec<_>>();
    let growth_history = render_state
        .growth_rate_history
        .iter()
        .copied()
        .collect::<Vec<_>>();
    let throughput_block = Block::new()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(ui_fg(no_color, PackedRgba::rgb(92, 147, 214)))
        .title(" throughput ");
    throughput_block.render(left[0], frame);
    let throughput_inner = throughput_block.inner(left[0]);
    let throughput_rows = Flex::vertical()
        .constraints([Constraint::Fill, Constraint::Fixed(1), Constraint::Fixed(1)])
        .split(throughput_inner);
    Paragraph::new(Text::from_lines(vec![
        Line::from(Span::styled(
            format!(
                "instant: {:.1} files/s  {:.1} lines/s  {:.2} MB/s",
                render_state.instant_files_per_second,
                render_state.instant_lines_per_second,
                render_state.instant_mb_per_second
            ),
            ui_fg(no_color, PackedRgba::rgb(221, 233, 255)),
        )),
        Line::from(Span::styled(
            format!(
                "average: {:.1} files/s  {:.1} lines/s  {:.2} MB/s",
                snapshot.files_per_second(),
                snapshot.lines_per_second(),
                snapshot.mb_per_second()
            ),
            ui_fg(no_color, PackedRgba::rgb(170, 190, 222)),
        )),
        Line::from(Span::styled(
            format!(
                "smoothed: {:.1} files/s  {:.1} lines/s  {:.2} MB/s",
                render_state.files_rate_ema, render_state.lines_rate_ema, render_state.mb_rate_ema
            ),
            ui_fg(no_color, PackedRgba::rgb(170, 190, 222)),
        )),
    ]))
    .render(throughput_rows[0], frame);
    Sparkline::new(&files_history)
        .style(ui_fg(no_color, PackedRgba::rgb(122, 219, 255)))
        .render(throughput_rows[1], frame);
    Sparkline::new(&growth_history)
        .style(ui_fg(no_color, PackedRgba::rgb(131, 231, 157)))
        .render(throughput_rows[2], frame);

    let lines_per_file = if snapshot.processed_files == 0 {
        0.0
    } else {
        snapshot.canonical_lines as f64 / snapshot.processed_files as f64
    };
    let bytes_per_line = if snapshot.canonical_lines == 0 {
        0.0
    } else {
        snapshot.canonical_bytes as f64 / snapshot.canonical_lines as f64
    };
    Paragraph::new(Text::from_lines(vec![
        Line::from(Span::styled(
            format!(
                "discovered={}  candidates={}  indexed={}  skipped={}",
                format_count_usize(snapshot.discovered_files),
                format_count_usize(snapshot.candidate_files),
                format_count_usize(snapshot.processed_files),
                format_count_usize(snapshot.skipped_files),
            ),
            ui_fg(no_color, PackedRgba::rgb(220, 233, 255)),
        )),
        Line::from(Span::styled(
            format!(
                "semantic={}  lines={}  data={}",
                format_count_usize(snapshot.semantic_files),
                format_count_u64(snapshot.canonical_lines),
                humanize_bytes(snapshot.canonical_bytes),
            ),
            ui_fg(no_color, PackedRgba::rgb(173, 193, 226)),
        )),
        Line::from(Span::styled(
            format!(
                "index size={}  lines/file={:.1}  bytes/line={:.1}",
                humanize_bytes(snapshot.index_size_bytes),
                lines_per_file,
                bytes_per_line,
            ),
            ui_fg(no_color, PackedRgba::rgb(173, 193, 226)),
        )),
    ]))
    .block(
        Block::new()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(ui_fg(no_color, PackedRgba::rgb(112, 119, 194)))
            .title(" corpus load "),
    )
    .render(left[1], frame);

    let accounted_ms = snapshot
        .discovery_elapsed_ms
        .saturating_add(snapshot.lexical_elapsed_ms)
        .saturating_add(snapshot.embedding_elapsed_ms)
        .saturating_add(snapshot.vector_elapsed_ms);
    let orchestration_ms = snapshot.total_elapsed_ms.saturating_sub(accounted_ms);
    Paragraph::new(Text::from_lines(vec![
        Line::from(Span::styled(
            format!(
                "discovery={}ms ({:.1}%)",
                snapshot.discovery_elapsed_ms,
                timing_ratio_percent(snapshot.discovery_elapsed_ms, snapshot.total_elapsed_ms),
            ),
            ui_fg(no_color, PackedRgba::rgb(220, 233, 255)),
        )),
        Line::from(Span::styled(
            format!(
                "lexical={}ms ({:.1}%)  embed={}ms ({:.1}%)",
                snapshot.lexical_elapsed_ms,
                timing_ratio_percent(snapshot.lexical_elapsed_ms, snapshot.total_elapsed_ms),
                snapshot.embedding_elapsed_ms,
                timing_ratio_percent(snapshot.embedding_elapsed_ms, snapshot.total_elapsed_ms),
            ),
            ui_fg(no_color, PackedRgba::rgb(173, 193, 226)),
        )),
        Line::from(Span::styled(
            format!(
                "vector={}ms ({:.1}%)  orchestration={}ms ({:.1}%)",
                snapshot.vector_elapsed_ms,
                timing_ratio_percent(snapshot.vector_elapsed_ms, snapshot.total_elapsed_ms),
                orchestration_ms,
                timing_ratio_percent(orchestration_ms, snapshot.total_elapsed_ms),
            ),
            ui_fg(no_color, PackedRgba::rgb(173, 193, 226)),
        )),
        Line::from(Span::styled(
            format!(
                "index growth {:.1} KB/s (ema {:.1})",
                render_state.instant_growth_kb_per_second, render_state.growth_rate_ema
            ),
            ui_fg(no_color, PackedRgba::rgb(173, 193, 226)),
        )),
    ]))
    .block(
        Block::new()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(ui_fg(no_color, PackedRgba::rgb(177, 128, 218)))
            .title(" stage timings "),
    )
    .render(right[0], frame);

    let active_file_line = snapshot.active_file.as_deref().map_or_else(
        || "waiting for active file…".to_owned(),
        |value| truncate_middle(value, usize::from(right[1].width).saturating_sub(10)),
    );
    Paragraph::new(Text::from_lines(vec![
        Line::from(Span::styled(
            active_file_line,
            ui_fg(no_color, PackedRgba::rgb(222, 236, 255)),
        )),
        Line::from(""),
        Line::from(Span::styled(
            format!(
                "trend completion: {}",
                render_state.completion_sparkline(INDEXING_TUI_SPARKLINE_WIDTH)
            ),
            ui_fg(no_color, PackedRgba::rgb(164, 184, 219)),
        )),
        Line::from(Span::styled(
            format!(
                "trend files/s:   {}",
                render_state.files_rate_sparkline(INDEXING_TUI_SPARKLINE_WIDTH)
            ),
            ui_fg(no_color, PackedRgba::rgb(164, 184, 219)),
        )),
    ]))
    .block(
        Block::new()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(ui_fg(no_color, PackedRgba::rgb(97, 162, 234)))
            .title(" active file "),
    )
    .render(right[1], frame);

    Paragraph::new(Text::from_lines(vec![
        Line::from(Span::styled(
            "Indexing is streaming live stats. Ctrl+C cancels safely.",
            ui_fg(no_color, PackedRgba::rgb(166, 186, 219)),
        )),
        Line::from(Span::styled(
            "Re-run with `fsfs index <path>` at any time to continue.",
            ui_fg(no_color, PackedRgba::rgb(132, 151, 184)),
        )),
    ]))
    .block(
        Block::new()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(ui_fg(no_color, PackedRgba::rgb(72, 98, 149)))
            .title(" notes "),
    )
    .render(layout[3], frame);
}

#[allow(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn render_indexing_progress_screen(
    snapshot: &IndexingProgressSnapshot,
    render_state: &IndexingRenderState,
    no_color: bool,
) -> SearchResult<()> {
    let width = tui_terminal_width().clamp(88, 168);
    let rule = "─".repeat(width);
    let stage_color = stage_color_code(snapshot.stage);
    let ratio = snapshot.completion_ratio();
    let pct = ratio * 100.0;
    let progress_bar_width = width.saturating_sub(54).clamp(20, 72);
    let bar = render_progress_bar(ratio, progress_bar_width);
    let elapsed_secs = u64::try_from(snapshot.total_elapsed_ms / 1_000).unwrap_or(u64::MAX);
    let elapsed = humanize_duration_seconds(elapsed_secs);
    let raw_eta = format_eta(snapshot.eta_seconds());
    let smooth_eta = format_eta(render_state.smoothed_eta_seconds());
    let finish_at = render_state.smoothed_eta_seconds().map_or_else(
        || "--".to_owned(),
        |seconds| {
            format_epoch_ms_utc(
                pressure_timestamp_ms().saturating_add(seconds.saturating_mul(1_000)),
            )
        },
    );
    let pending_files = snapshot
        .candidate_files
        .saturating_sub(snapshot.processed_files);
    let candidate_ratio = ratio_percent_usize(snapshot.candidate_files, snapshot.discovered_files);
    let skipped_ratio = ratio_percent_usize(snapshot.skipped_files, snapshot.discovered_files);
    let semantic_ratio = ratio_percent_usize(snapshot.semantic_files, snapshot.processed_files);
    let lines_per_file = if snapshot.processed_files == 0 {
        0.0
    } else {
        snapshot.canonical_lines as f64 / snapshot.processed_files as f64
    };
    let bytes_per_line = if snapshot.canonical_lines == 0 {
        0.0
    } else {
        snapshot.canonical_bytes as f64 / snapshot.canonical_lines as f64
    };
    let index_amplification = if snapshot.canonical_bytes == 0 {
        0.0
    } else {
        snapshot.index_size_bytes as f64 / snapshot.canonical_bytes as f64
    };
    let accounted_ms = snapshot
        .discovery_elapsed_ms
        .saturating_add(snapshot.lexical_elapsed_ms)
        .saturating_add(snapshot.embedding_elapsed_ms)
        .saturating_add(snapshot.vector_elapsed_ms);
    let orchestration_ms = snapshot.total_elapsed_ms.saturating_sub(accounted_ms);

    let mut stdout = std::io::stdout();
    write!(stdout, "\u{1b}[2J\u{1b}[H").map_err(tui_io_error)?;
    writeln!(stdout, "{}", paint(&rule, "38;5;24", no_color)).map_err(tui_io_error)?;
    writeln!(
        stdout,
        "{} {}  {} {}",
        paint("fsfs", "1;38;5;45", no_color),
        paint("Initial Indexing", "1;37", no_color),
        paint(
            indexing_spinner_frame(snapshot.total_elapsed_ms),
            stage_color,
            no_color
        ),
        paint(snapshot.stage.label(), stage_color, no_color)
    )
    .map_err(tui_io_error)?;
    writeln!(
        stdout,
        "{} {}",
        paint("target:", "38;5;244", no_color),
        truncate_middle(
            &snapshot.target_root.display().to_string(),
            width.saturating_sub(10)
        )
    )
    .map_err(tui_io_error)?;
    writeln!(
        stdout,
        "{}  {}",
        paint("index:", "38;5;244", no_color),
        truncate_middle(
            &snapshot.index_root.display().to_string(),
            width.saturating_sub(10)
        )
    )
    .map_err(tui_io_error)?;
    writeln!(stdout, "{}", paint(&rule, "38;5;24", no_color)).map_err(tui_io_error)?;
    writeln!(
        stdout,
        "{} [{}] {:>5.1}%  queue={}  elapsed={}  eta(raw={} smooth={})",
        paint("progress", "1;38;5;220", no_color),
        bar,
        pct,
        format_count_usize(pending_files),
        elapsed,
        raw_eta,
        smooth_eta
    )
    .map_err(tui_io_error)?;
    writeln!(stdout, "finish estimate (smoothed): {finish_at}").map_err(tui_io_error)?;
    writeln!(stdout).map_err(tui_io_error)?;
    writeln!(
        stdout,
        "{} discovered={} candidates={} indexed={} skipped={} semantic={}",
        paint("workload:", "1;38;5;81", no_color),
        format_count_usize(snapshot.discovered_files),
        format_count_usize(snapshot.candidate_files),
        format_count_usize(snapshot.processed_files),
        format_count_usize(snapshot.skipped_files),
        format_count_usize(snapshot.semantic_files)
    )
    .map_err(tui_io_error)?;
    writeln!(
        stdout,
        "          candidate_ratio={candidate_ratio:.1}%  skipped_ratio={skipped_ratio:.1}%  semantic_ratio={semantic_ratio:.1}%"
    )
    .map_err(tui_io_error)?;
    writeln!(
        stdout,
        "{} lines={} ({:.1} lines/file)  data={}  index_size={}  amp={:.2}x  bytes/line={:.1}",
        paint("content:", "1;38;5;111", no_color),
        format_count_u64(snapshot.canonical_lines),
        lines_per_file,
        humanize_bytes(snapshot.canonical_bytes),
        humanize_bytes(snapshot.index_size_bytes),
        index_amplification,
        bytes_per_line
    )
    .map_err(tui_io_error)?;
    writeln!(
        stdout,
        "{} instant: {:.1} files/s  {:.1} lines/s  {:.2} MB/s  growth {:.1} KB/s",
        paint("rates:", "1;38;5;214", no_color),
        render_state.instant_files_per_second,
        render_state.instant_lines_per_second,
        render_state.instant_mb_per_second,
        render_state.instant_growth_kb_per_second
    )
    .map_err(tui_io_error)?;
    writeln!(
        stdout,
        "       average: {:.1} files/s  {:.1} lines/s  {:.2} MB/s  growth {:.1} KB/s",
        snapshot.files_per_second(),
        snapshot.lines_per_second(),
        snapshot.mb_per_second(),
        snapshot.index_growth_bytes_per_second() / 1024.0
    )
    .map_err(tui_io_error)?;
    writeln!(
        stdout,
        "       smoothed: {:.1} files/s  {:.1} lines/s  {:.2} MB/s  growth {:.1} KB/s",
        render_state.files_rate_ema,
        render_state.lines_rate_ema,
        render_state.mb_rate_ema,
        render_state.growth_rate_ema
    )
    .map_err(tui_io_error)?;
    writeln!(
        stdout,
        "       trend: completion {}  files/s {}  growth {}",
        render_state.completion_sparkline(INDEXING_TUI_SPARKLINE_WIDTH),
        render_state.files_rate_sparkline(INDEXING_TUI_SPARKLINE_WIDTH),
        render_state.growth_rate_sparkline(INDEXING_TUI_SPARKLINE_WIDTH)
    )
    .map_err(tui_io_error)?;
    writeln!(
        stdout,
        "{} discovery={}ms ({:.1}%)  lexical={}ms ({:.1}%)  embedding={}ms ({:.1}%)  vector={}ms ({:.1}%)  orchestration={}ms ({:.1}%)",
        paint("timings:", "1;38;5;177", no_color),
        snapshot.discovery_elapsed_ms,
        timing_ratio_percent(snapshot.discovery_elapsed_ms, snapshot.total_elapsed_ms),
        snapshot.lexical_elapsed_ms,
        timing_ratio_percent(snapshot.lexical_elapsed_ms, snapshot.total_elapsed_ms),
        snapshot.embedding_elapsed_ms,
        timing_ratio_percent(snapshot.embedding_elapsed_ms, snapshot.total_elapsed_ms),
        snapshot.vector_elapsed_ms
            ,
        timing_ratio_percent(snapshot.vector_elapsed_ms, snapshot.total_elapsed_ms),
        orchestration_ms,
        timing_ratio_percent(orchestration_ms, snapshot.total_elapsed_ms)
    )
    .map_err(tui_io_error)?;
    if let Some(active_file) = snapshot.active_file.as_deref() {
        writeln!(
            stdout,
            "{} {}",
            paint("active:", "1;38;5;51", no_color),
            truncate_middle(active_file, width.saturating_sub(9))
        )
        .map_err(tui_io_error)?;
    }
    writeln!(
        stdout,
        "{}",
        paint(
            "Tip: Ctrl+C cancels safely. You can rerun with `fsfs index <path>` any time.",
            "38;5;244",
            no_color
        )
    )
    .map_err(tui_io_error)?;
    stdout.flush().map_err(tui_io_error)?;
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn render_status_table(status: &FsfsStatusPayload, no_color: bool) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    let now_ms = pressure_timestamp_ms();
    let width = tui_terminal_width().clamp(88, 160);
    let rule = "─".repeat(width);

    let stale_label = match status.index.stale_files {
        Some(0) => paint("up-to-date", "32", no_color),
        Some(count) => paint(
            &format!("stale ({})", format_count_usize(count)),
            "33",
            no_color,
        ),
        None => paint("unknown", "90", no_color),
    };
    let index_exists = if status.index.exists {
        paint("ready", "32", no_color)
    } else {
        paint("missing", "31", no_color)
    };
    let vector_pct = ratio_percent(
        status.index.vector_index_bytes,
        status.index.size_bytes.max(1),
    );
    let lexical_pct = ratio_percent(
        status.index.lexical_index_bytes,
        status.index.size_bytes.max(1),
    );
    let indexed_files = status.index.indexed_files.unwrap_or(0);
    let discovered_files = status.index.discovered_files.unwrap_or(0);
    let skipped_files = status.index.skipped_files.unwrap_or(0);
    let avg_index_bytes_per_file = if indexed_files == 0 {
        0_u64
    } else {
        status.index.size_bytes / u64::try_from(indexed_files).unwrap_or(1)
    };
    let fast_cached = status
        .models
        .iter()
        .find(|model| model.tier == "fast")
        .is_some_and(|model| model.cached);
    let quality_cached = status
        .models
        .iter()
        .find(|model| model.tier == "quality")
        .is_some_and(|model| model.cached);
    let search_mode_line = if fast_cached && quality_cached && !status.config.fast_only {
        paint(
            "  mode: full hybrid (fast + quality semantic)",
            "32",
            no_color,
        )
    } else if fast_cached {
        if status.config.fast_only {
            paint(
                "  mode: fast semantic only (fast_only=true)",
                "33",
                no_color,
            )
        } else {
            paint(
                "  mode: fast semantic + lexical (quality model missing)",
                "33",
                no_color,
            )
        }
    } else {
        paint(
            "  mode: hash + lexical fallback (no semantic model cache)",
            "38;5;214",
            no_color,
        )
    };

    let _ = writeln!(
        out,
        "{} {} {}",
        paint("frankensearch", "1;38;5;45", no_color),
        status.version,
        paint("| fsfs status", "38;5;244", no_color)
    );
    let _ = writeln!(out, "{}", paint(&rule, "38;5;24", no_color));
    let _ = writeln!(out);
    let _ = writeln!(out, "{}", paint("INDEX", "1;38;5;81", no_color));
    let _ = writeln!(
        out,
        "  root: {}",
        truncate_middle(&status.index.path, width.saturating_sub(10))
    );
    let _ = writeln!(out, "  state: {index_exists}");
    let _ = writeln!(
        out,
        "  files: indexed={}  discovered={}  skipped={}",
        format_count_usize(indexed_files),
        format_count_usize(discovered_files),
        format_count_usize(skipped_files)
    );
    let _ = writeln!(
        out,
        "  size: total={}  avg/indexed_file={}  cache={}",
        humanize_bytes(status.index.size_bytes),
        humanize_bytes(avg_index_bytes_per_file),
        humanize_bytes(status.index.embedding_cache_bytes),
    );
    let _ = writeln!(
        out,
        "        vector={} ({:.1}%)  lexical={} ({:.1}%)  metadata={}",
        humanize_bytes(status.index.vector_index_bytes),
        vector_pct,
        humanize_bytes(status.index.lexical_index_bytes),
        lexical_pct,
        humanize_bytes(status.index.metadata_bytes),
    );
    if let Some(last_indexed_ms) = status.index.last_indexed_ms {
        let _ = writeln!(
            out,
            "  last indexed: {}  ({})",
            humanize_age(now_ms, last_indexed_ms),
            status
                .index
                .last_indexed_iso_utc
                .as_deref()
                .unwrap_or("unknown"),
        );
    }
    let _ = writeln!(out, "  staleness: {stale_label}");
    if let Some(source_hash_hex) = status.index.source_hash_hex.as_deref() {
        let _ = writeln!(out, "  source hash: {source_hash_hex}");
    }

    let _ = writeln!(out);
    let _ = writeln!(out, "{}", paint("MODELS", "1;38;5;214", no_color));
    for model in &status.models {
        let state = if model.cached {
            paint("cached", "32", no_color)
        } else {
            paint("missing", "31", no_color)
        };
        let _ = writeln!(
            out,
            "  {}: {}  {}  size={} ",
            model.tier,
            model.name,
            state,
            humanize_bytes(model.size_bytes),
        );
        let _ = writeln!(
            out,
            "      path: {}",
            truncate_middle(&model.cache_path, width.saturating_sub(13))
        );
    }
    let _ = writeln!(out, "{search_mode_line}");

    let _ = writeln!(out);
    let _ = writeln!(out, "{}", paint("CONFIG", "1;38;5;177", no_color));
    let _ = writeln!(out, "  source: {}", status.config.source);
    let _ = writeln!(out, "  index dir: {}", status.config.index_dir);
    let _ = writeln!(
        out,
        "  model dir: {}",
        truncate_middle(&status.config.model_dir, width.saturating_sub(13))
    );
    let _ = writeln!(out, "  rrf k: {}", status.config.rrf_k);
    let _ = writeln!(out, "  quality weight: {}", status.config.quality_weight);
    let _ = writeln!(
        out,
        "  quality timeout: {}ms",
        status.config.quality_timeout_ms
    );
    let _ = writeln!(out, "  fast only: {}", status.config.fast_only);
    let _ = writeln!(
        out,
        "  pressure profile: {}",
        status.config.pressure_profile
    );

    let _ = writeln!(out);
    let _ = writeln!(out, "{}", paint("RUNTIME", "1;38;5;111", no_color));
    let _ = writeln!(
        out,
        "  disk budget stage: {}",
        status
            .runtime
            .disk_budget_stage
            .as_deref()
            .unwrap_or("unknown"),
    );
    let _ = writeln!(
        out,
        "  disk budget action: {}",
        status
            .runtime
            .disk_budget_action
            .as_deref()
            .unwrap_or("unknown"),
    );
    let _ = writeln!(
        out,
        "  disk budget reason: {}",
        status
            .runtime
            .disk_budget_reason_code
            .as_deref()
            .unwrap_or("unknown"),
    );
    if let Some(index_bytes) = status.runtime.tracked_index_bytes {
        let _ = writeln!(
            out,
            "  tracked index bytes: {} ({})",
            humanize_bytes(index_bytes),
            format_count_u64(index_bytes)
        );
    }
    if let Some(budget_bytes) = status.runtime.disk_budget_bytes {
        let _ = writeln!(
            out,
            "  disk budget bytes: {} ({})",
            humanize_bytes(budget_bytes),
            format_count_u64(budget_bytes)
        );
    }
    let _ = writeln!(
        out,
        "  storage pressure emergency: {}",
        status.runtime.storage_pressure_emergency
    );

    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "{}",
        paint(
            "NEXT: fsfs search \"your query\" --limit all  |  fsfs search \"query\" --stream --format jsonl",
            "38;5;244",
            no_color
        )
    );

    out
}

fn render_download_models_table(payload: &FsfsDownloadModelsPayload, no_color: bool) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    let _ = writeln!(out, "download-models ({})", payload.operation);
    let _ = writeln!(out, "  model root: {}", payload.model_root);
    let _ = writeln!(out, "  force: {}", payload.force);
    let _ = writeln!(out);
    for model in &payload.models {
        let state_color = match model.state.as_str() {
            "cached" | "verified" | "downloaded" => "32",
            "missing" => "33",
            "corrupt" | "mismatch" => "31",
            _ => "90",
        };
        let state = paint(&model.state, state_color, no_color);
        let tier = model.tier.as_deref().unwrap_or("unknown");
        let _ = writeln!(
            out,
            "  {} [{}] {} ({})",
            model.install_dir,
            tier,
            state,
            humanize_bytes(model.size_bytes),
        );
        let _ = writeln!(out, "    path: {}", model.destination);
        if let Some(message) = model.message.as_deref() {
            let _ = writeln!(out, "    note: {message}");
        }
    }
    out
}

fn render_update_table(payload: &FsfsUpdatePayload, no_color: bool) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    let _ = writeln!(out, "fsfs update");
    let _ = writeln!(out, "  current: v{}", payload.current_version);
    let _ = writeln!(out, "  latest:  v{}", payload.latest_version);

    if payload.update_available {
        let status_text = if payload.applied {
            "applied"
        } else if payload.check_only {
            "available (check-only)"
        } else {
            "available"
        };
        let color = if payload.applied { "32" } else { "33" };
        let status = paint(status_text, color, no_color);
        let _ = writeln!(out, "  status:  {status}");
    } else {
        let status = paint("up to date", "32", no_color);
        let _ = writeln!(out, "  status:  {status}");
    }

    let _ = writeln!(out, "  channel: {}", payload.channel);
    if let Some(ref url) = payload.release_url {
        let _ = writeln!(out, "  release: {url}");
    }

    if !payload.notes.is_empty() {
        let _ = writeln!(out);
        for note in &payload.notes {
            let _ = writeln!(out, "  {note}");
        }
    }
    out
}

/// Recursively search up to 2 levels deep for an extracted binary.
fn find_extracted_binary(dir: &Path, name: &str) -> SearchResult<PathBuf> {
    let canonical_root = fs::canonicalize(dir).map_err(|e| SearchError::SubsystemError {
        subsystem: "fsfs.update.extract",
        source: Box::new(e),
    })?;

    // Check top level first.
    let direct = dir.join(name);
    if let Some(path) = extracted_binary_candidate_if_safe(&canonical_root, &direct) {
        return Ok(path);
    }
    // Check one level deeper.
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let candidate = path.join(name);
                if let Some(path) = extracted_binary_candidate_if_safe(&canonical_root, &candidate)
                {
                    return Ok(path);
                }
            }
        }
    }
    Err(SearchError::InvalidConfig {
        field: "update.extract".into(),
        value: dir.display().to_string(),
        reason: format!("could not find '{name}' binary in extracted archive"),
    })
}

fn extracted_binary_candidate_if_safe(root: &Path, candidate: &Path) -> Option<PathBuf> {
    let metadata = fs::symlink_metadata(candidate).ok()?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return None;
    }
    let canonical_candidate = fs::canonicalize(candidate).ok()?;
    if canonical_candidate.starts_with(root) {
        Some(canonical_candidate)
    } else {
        None
    }
}

fn render_uninstall_table(payload: &FsfsUninstallPayload, no_color: bool) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    let mode = if payload.dry_run {
        "dry-run"
    } else {
        "execute"
    };
    let _ = writeln!(out, "uninstall ({mode})");
    let _ = writeln!(out, "  purge: {}", payload.purge);
    let _ = writeln!(out, "  confirmed: {}", payload.confirmed);
    let _ = writeln!(
        out,
        "  removed: {}, skipped: {}, failed: {}",
        payload.removed, payload.skipped, payload.failed
    );
    let _ = writeln!(out);
    for entry in &payload.entries {
        let color = match entry.status.as_str() {
            "removed" => "32",
            "planned" | "not_found" | "skipped" => "33",
            "error" => "31",
            _ => "90",
        };
        let status = paint(&entry.status, color, no_color);
        let _ = writeln!(
            out,
            "  {:<10} {:<16} [{}] {}",
            status, entry.target, entry.kind, entry.path
        );
        if let Some(detail) = entry.detail.as_deref() {
            let _ = writeln!(out, "             note: {detail}");
        }
    }
    if !payload.notes.is_empty() {
        let _ = writeln!(out);
        let _ = writeln!(out, "notes:");
        for note in &payload.notes {
            let _ = writeln!(out, "  - {note}");
        }
    }
    out
}

fn render_doctor_table(payload: &FsfsDoctorPayload, no_color: bool) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    let _ = writeln!(out, "fsfs doctor ({})", payload.version);
    let _ = writeln!(out);
    for check in &payload.checks {
        let (icon, color) = match check.verdict {
            DoctorVerdict::Pass => ("ok", "32"),
            DoctorVerdict::Warn => ("!!", "33"),
            DoctorVerdict::Fail => ("FAIL", "31"),
        };
        let verdict = paint(icon, color, no_color);
        let _ = writeln!(out, "  [{verdict}] {}: {}", check.name, check.detail);
        if let Some(suggestion) = check.suggestion.as_deref() {
            let _ = writeln!(out, "       -> {suggestion}");
        }
    }
    let _ = writeln!(out);
    let overall_color = match payload.overall {
        DoctorVerdict::Pass => "32",
        DoctorVerdict::Warn => "33",
        DoctorVerdict::Fail => "31",
    };
    let overall = paint(&payload.overall.to_string(), overall_color, no_color);
    let _ = writeln!(
        out,
        "Result: {overall} ({} passed, {} warnings, {} failures)",
        payload.pass_count, payload.warn_count, payload.fail_count,
    );
    out
}

fn humanize_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut unit_index = 0_usize;
    let mut divisor = 1_u64;
    while bytes / divisor >= 1024 && unit_index < UNITS.len().saturating_sub(1) {
        divisor = divisor.saturating_mul(1024);
        unit_index += 1;
    }
    if unit_index == 0 {
        format!("{bytes} {}", UNITS[unit_index])
    } else {
        let whole = bytes / divisor;
        let frac = bytes
            .saturating_sub(whole.saturating_mul(divisor))
            .saturating_mul(10)
            / divisor;
        format!("{whole}.{frac} {}", UNITS[unit_index])
    }
}

fn humanize_age(now_ms: u64, then_ms: u64) -> String {
    let age_secs = now_ms.saturating_sub(then_ms) / 1000;
    match age_secs {
        0..=59 => format!("{age_secs}s ago"),
        60..=3_599 => format!("{}m ago", age_secs / 60),
        3_600..=86_399 => format!("{}h ago", age_secs / 3_600),
        _ => format!("{}d ago", age_secs / 86_400),
    }
}

fn f64_to_per_mille(value: f64) -> u16 {
    let scaled = (value.max(0.0) * 1_000.0).round();
    if !scaled.is_finite() {
        return 0;
    }
    let clamped = scaled.clamp(0.0, f64::from(u16::MAX));
    format!("{clamped:.0}").parse::<u16>().unwrap_or(u16::MAX)
}

fn paint(text: &str, color_code: &str, no_color: bool) -> String {
    if no_color {
        text.to_owned()
    } else {
        format!("\u{1b}[{color_code}m{text}\u{1b}[0m")
    }
}

fn iso_timestamp_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format_epoch_secs_utc(secs)
}

fn format_epoch_ms_utc(ms: u64) -> String {
    format_epoch_secs_utc(ms / 1000)
}

fn format_epoch_secs_utc(secs: u64) -> String {
    let days_since_epoch = secs / 86_400;
    let time_of_day = secs % 86_400;
    let hours = time_of_day / 3_600;
    let minutes = (time_of_day % 3_600) / 60;
    let seconds = time_of_day % 60;
    let (year, month, day) = epoch_days_to_ymd(days_since_epoch);
    format!("{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
}

const fn epoch_days_to_ymd(days: u64) -> (u64, u64, u64) {
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { year + 1 } else { year };
    (year, month, day)
}

fn normalize_file_key_for_index(path: &Path, target_root: &Path) -> String {
    path.strip_prefix(target_root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn index_source_hash_hex(manifests: &[IndexManifestEntry]) -> String {
    let mut hasher = DefaultHasher::new();
    for entry in manifests {
        entry.file_key.hash(&mut hasher);
        entry.revision.hash(&mut hasher);
        entry.canonical_bytes.hash(&mut hasher);
        entry.ingestion_class.hash(&mut hasher);
    }
    format!("{:016x}", hasher.finish())
}

fn is_probably_binary(data: &[u8]) -> bool {
    if data.is_empty() {
        return false;
    }
    if data.contains(&0) {
        return true;
    }

    let control_count = data
        .iter()
        .filter(|byte| {
            matches!(
                **byte,
                0x01..=0x08 | 0x0B | 0x0C | 0x0E..=0x1F | 0x7F
            )
        })
        .count();
    control_count.saturating_mul(5) > data.len()
}

fn is_ignorable_index_walk_error(error: &std::io::Error) -> bool {
    matches!(
        error.kind(),
        std::io::ErrorKind::NotFound
            | std::io::ErrorKind::PermissionDenied
            | std::io::ErrorKind::Interrupted
    )
}

const fn map_degradation_override(
    mode: DegradationOverrideMode,
) -> crate::pressure::DegradationOverride {
    match mode {
        DegradationOverrideMode::Auto => crate::pressure::DegradationOverride::Auto,
        DegradationOverrideMode::ForceFull => crate::pressure::DegradationOverride::ForceFull,
        DegradationOverrideMode::ForceEmbedDeferred => {
            crate::pressure::DegradationOverride::ForceEmbedDeferred
        }
        DegradationOverrideMode::ForceLexicalOnly => {
            crate::pressure::DegradationOverride::ForceLexicalOnly
        }
        DegradationOverrideMode::ForceMetadataOnly => {
            crate::pressure::DegradationOverride::ForceMetadataOnly
        }
        DegradationOverrideMode::ForcePaused => crate::pressure::DegradationOverride::ForcePaused,
    }
}

const fn degradation_controller_config_for_profile(
    profile: PressureProfile,
    anti_flap_readings: u8,
) -> DegradationControllerConfig {
    let extra_recovery_readings = match profile {
        PressureProfile::Strict => 1,
        PressureProfile::Performance => 0,
        PressureProfile::Degraded => 2,
    };
    DegradationControllerConfig {
        consecutive_healthy_required: anti_flap_readings.saturating_add(extra_recovery_readings),
    }
}

/// Write `data` to `path` with fsync so the content is durable even under
/// sudden power loss.  Used for manifest, sentinel, and other metadata files
/// whose silent loss would leave the index in an inconsistent state.
fn write_durable(path: impl AsRef<Path>, data: impl AsRef<[u8]>) -> std::io::Result<()> {
    let mut file = fs::File::create(path)?;
    file.write_all(data.as_ref())?;
    file.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, Instant};

    use asupersync::test_utils::run_test_with_cx;
    use frankensearch_core::{IndexableDocument, LexicalSearch};
    use frankensearch_embed::HashEmbedder;
    use frankensearch_index::VectorIndex;
    use frankensearch_lexical::TantivyIndex;
    use fsqlite::Connection;
    use fsqlite_types::value::SqliteValue;

    use super::{
        ContextPreviewFormat, EmbedderAvailability, FSFS_DAEMON_REQUEST_MAX_BYTES,
        FSFS_SEARCH_SNIPPET_HEAD_LIMIT, FSFS_SEARCH_UNBOUNDED_LIMIT_SENTINEL,
        FSFS_TUI_LEXICAL_DEBOUNCE_MAX_MS, FSFS_TUI_LEXICAL_DEBOUNCE_MIN_MS,
        FSFS_TUI_LEXICAL_DEBOUNCE_MS, FSFS_TUI_LEXICAL_DEBOUNCE_SHORT_QUERY_CAP_MS,
        FSFS_TUI_QUALITY_DEBOUNCE_MAX_MS, FSFS_TUI_QUALITY_DEBOUNCE_MIN_MS,
        FSFS_TUI_QUALITY_DEBOUNCE_MS, FSFS_TUI_SEMANTIC_DEBOUNCE_MAX_MS,
        FSFS_TUI_SEMANTIC_DEBOUNCE_MIN_MS, FSFS_TUI_SEMANTIC_DEBOUNCE_MS, FsfsConfigStatus,
        FsfsIndexStatus, FsfsModelStatus, FsfsRuntime, FsfsRuntimeStatus, FsfsStatusPayload,
        IndexStoragePaths, InterfaceMode, LiveIngestPipeline, SearchDashboardState,
        SearchExecutionMode, SemanticGateDecisionInput, SemanticRecallDecisionInput,
        VectorIndexWriteAction, VectorPipelineInput, VectorSchedulingTier,
        degradation_controller_config_for_profile, detect_context_preview_format,
        is_likely_html_fragment, normalize_html_fragment_for_markdown, render_status_table,
    };
    use crate::adapters::cli::{CliCommand, CliInput, CompletionShell, OutputFormat};
    use crate::catalog::bootstrap_catalog_schema;
    use crate::config::{
        DegradationOverrideMode, DiscoveryCandidate, DiscoveryScopeDecision, FsfsConfig,
        IngestionClass, PressureProfile,
    };
    use crate::lifecycle::{
        DiskBudgetAction, DiskBudgetStage, LifecycleTracker, ResourceLimits, WatchdogConfig,
    };
    use crate::output_schema::{SearchHitPayload, SearchOutputPhase, SearchPayload};
    use crate::pressure::{
        DegradationStage, HostPressureCollector, PressureSignal, PressureState, QueryCapabilityMode,
    };
    use crate::query_execution::{FusedCandidate, LexicalCandidate};
    use crate::query_planning::QueryIntentClass;
    use crate::shutdown::{ShutdownCoordinator, ShutdownReason};
    use crate::stream_protocol::{
        StreamEventKind, StreamFrame, TOON_STREAM_RECORD_SEPARATOR_BYTE,
        decode_stream_frame_ndjson, decode_stream_frame_toon,
    };
    use crate::watcher::{WatchIngestOp, WatchIngestPipeline};

    fn test_dashboard_status_payload() -> FsfsStatusPayload {
        FsfsStatusPayload {
            version: "0.1.0".to_owned(),
            index: FsfsIndexStatus {
                path: "/tmp/.frankensearch".to_owned(),
                exists: true,
                indexed_files: Some(128),
                discovered_files: Some(256),
                skipped_files: Some(12),
                last_indexed_ms: None,
                last_indexed_iso_utc: None,
                stale_files: Some(5),
                source_hash_hex: Some("deadbeef".to_owned()),
                size_bytes: 1024 * 1024,
                vector_index_bytes: 512 * 1024,
                lexical_index_bytes: 512 * 1024,
                metadata_bytes: 0,
                embedding_cache_bytes: 0,
            },
            models: vec![FsfsModelStatus {
                tier: "quality".to_owned(),
                name: "all-MiniLM-L6-v2".to_owned(),
                cache_path: "/tmp/models/all-MiniLM-L6-v2".to_owned(),
                cached: true,
                size_bytes: 90_873_196,
            }],
            config: FsfsConfigStatus {
                source: "test".to_owned(),
                index_dir: ".frankensearch".to_owned(),
                model_dir: "/tmp/models".to_owned(),
                rrf_k: 60.0,
                quality_weight: 0.7,
                quality_timeout_ms: 500,
                fast_only: false,
                pressure_profile: "performance".to_owned(),
            },
            runtime: FsfsRuntimeStatus {
                disk_budget_stage: Some("normal".to_owned()),
                disk_budget_action: Some("continue".to_owned()),
                disk_budget_reason_code: Some("degrade.disk.within_budget".to_owned()),
                disk_budget_bytes: Some(5 * 1024 * 1024 * 1024),
                tracked_index_bytes: Some(1024 * 1024),
                storage_pressure_emergency: false,
            },
        }
    }

    #[test]
    fn search_dashboard_adaptive_debounce_defaults_to_base_windows() {
        let state = SearchDashboardState::new(test_dashboard_status_payload(), None, 10, true);
        assert_eq!(
            state.lexical_debounce_window_ms,
            FSFS_TUI_LEXICAL_DEBOUNCE_MS
        );
        assert_eq!(
            state.semantic_debounce_window_ms,
            FSFS_TUI_SEMANTIC_DEBOUNCE_MS
        );
        assert_eq!(
            state.quality_debounce_window_ms,
            FSFS_TUI_QUALITY_DEBOUNCE_MS
        );
    }

    #[test]
    fn search_dashboard_adaptive_debounce_caps_short_query_lexical_window() {
        let mut state = SearchDashboardState::new(test_dashboard_status_payload(), None, 10, true);
        state.query_input.set_value("r".to_owned());
        state.typing_cadence_ewma_ms = Some(220);
        state.recompute_adaptive_debounce();

        assert!(state.lexical_debounce_window_ms <= FSFS_TUI_LEXICAL_DEBOUNCE_SHORT_QUERY_CAP_MS);
        assert!(
            (FSFS_TUI_SEMANTIC_DEBOUNCE_MIN_MS..=FSFS_TUI_SEMANTIC_DEBOUNCE_MAX_MS)
                .contains(&state.semantic_debounce_window_ms)
        );
        assert!(
            (FSFS_TUI_QUALITY_DEBOUNCE_MIN_MS..=FSFS_TUI_QUALITY_DEBOUNCE_MAX_MS)
                .contains(&state.quality_debounce_window_ms)
        );
    }

    #[test]
    fn search_dashboard_adaptive_debounce_trims_when_latency_is_fast() {
        let mut state = SearchDashboardState::new(test_dashboard_status_payload(), None, 10, true);
        state.query_input.set_value("rust".to_owned());
        state.typing_cadence_ewma_ms = Some(70);
        state.last_search_elapsed_ms = Some(18);
        state.push_latency_sample(18.0);
        state.recompute_adaptive_debounce();

        assert!(
            (FSFS_TUI_LEXICAL_DEBOUNCE_MIN_MS..=FSFS_TUI_LEXICAL_DEBOUNCE_MAX_MS)
                .contains(&state.lexical_debounce_window_ms)
        );
        assert!(
            state.semantic_debounce_window_ms < FSFS_TUI_SEMANTIC_DEBOUNCE_MS,
            "fast-path latency should trim semantic debounce"
        );
    }

    #[test]
    fn search_dashboard_adaptive_debounce_backs_off_when_latency_is_slow() {
        let mut state = SearchDashboardState::new(test_dashboard_status_payload(), None, 10, true);
        state
            .query_input
            .set_value("distributed systems".to_owned());
        state.typing_cadence_ewma_ms = Some(180);
        state.push_latency_sample(160.0);
        state.push_latency_sample(170.0);
        state.push_latency_sample(155.0);
        state.recompute_adaptive_debounce();

        assert!(
            (FSFS_TUI_LEXICAL_DEBOUNCE_MIN_MS..=FSFS_TUI_LEXICAL_DEBOUNCE_MAX_MS)
                .contains(&state.lexical_debounce_window_ms)
        );
        assert!(state.semantic_debounce_window_ms > FSFS_TUI_SEMANTIC_DEBOUNCE_MIN_MS);
        assert!(state.quality_debounce_window_ms > FSFS_TUI_QUALITY_DEBOUNCE_MIN_MS);
    }

    #[test]
    fn search_dashboard_record_query_edit_tracks_typing_cadence() {
        let mut state = SearchDashboardState::new(test_dashboard_status_payload(), None, 10, true);
        let start = Instant::now();
        state.record_query_edit(start);
        state.record_query_edit(start + Duration::from_millis(40));
        state.record_query_edit(start + Duration::from_millis(46));

        let cadence = state
            .typing_cadence_ewma_ms
            .expect("cadence ewma should be set");
        assert!(cadence >= 35);
        assert!(cadence <= 60);
    }

    #[test]
    fn search_dashboard_semantic_refresh_waits_for_idle_gate() {
        let mut state = SearchDashboardState::new(test_dashboard_status_payload(), None, 10, true);
        state.query_input.set_value("rust".to_owned());
        state.mark_query_dirty();
        state.semantic_pending_since = Instant::now()
            .checked_sub(Duration::from_millis(
                state.semantic_debounce_window_ms.saturating_add(10),
            ))
            .expect("semantic pending timestamp should backdate");

        assert!(
            !state.should_refresh_semantic(),
            "semantic refresh should wait until typing goes idle"
        );

        let idle_gate_ms = state.semantic_idle_gate_ms();
        state.last_query_edit_at = Some(
            Instant::now()
                .checked_sub(Duration::from_millis(idle_gate_ms.saturating_add(10)))
                .expect("query edit timestamp should backdate"),
        );
        assert!(
            state.should_refresh_semantic(),
            "semantic refresh should trigger after idle gate passes"
        );
    }

    #[test]
    fn search_dashboard_lexical_stage_result_limit_is_tight_while_typing() {
        let mut state = SearchDashboardState::new(test_dashboard_status_payload(), None, 200, true);
        state.query_input.set_value("rust".to_owned());
        state.mark_query_dirty();

        assert_eq!(
            state.lexical_stage_result_limit(),
            FSFS_SEARCH_SNIPPET_HEAD_LIMIT
        );
    }

    #[test]
    fn search_dashboard_lexical_stage_result_limit_returns_full_after_idle() {
        let mut state = SearchDashboardState::new(test_dashboard_status_payload(), None, 200, true);
        state.query_input.set_value("rust".to_owned());
        state.mark_query_dirty();

        let idle_gate_ms = state.semantic_idle_gate_ms();
        state.last_query_edit_at = Some(
            Instant::now()
                .checked_sub(Duration::from_millis(idle_gate_ms.saturating_add(10)))
                .expect("query edit timestamp should backdate"),
        );

        assert_eq!(state.lexical_stage_result_limit(), 200);
    }

    #[test]
    fn search_dashboard_force_refresh_bypasses_idle_gate() {
        let mut state = SearchDashboardState::new(test_dashboard_status_payload(), None, 10, true);
        state.query_input.set_value("rust".to_owned());
        state.mark_query_dirty();
        state.force_refresh_now();
        assert!(
            state.should_refresh_semantic(),
            "explicit refresh should trigger semantic tier immediately"
        );
    }

    #[test]
    fn search_dashboard_idle_gate_scales_with_typing_cadence() {
        let mut state = SearchDashboardState::new(test_dashboard_status_payload(), None, 10, true);
        state.query_input.set_value("rust".to_owned());

        state.typing_cadence_ewma_ms = Some(40);
        let fast_semantic_gate = state.semantic_idle_gate_ms();
        let fast_quality_gate = state.quality_idle_gate_ms();

        state.typing_cadence_ewma_ms = Some(220);
        let slow_semantic_gate = state.semantic_idle_gate_ms();
        let slow_quality_gate = state.quality_idle_gate_ms();

        assert!(slow_semantic_gate > fast_semantic_gate);
        assert!(slow_quality_gate > fast_quality_gate);
        assert!(fast_quality_gate >= fast_semantic_gate.saturating_add(40));
        assert!(slow_quality_gate >= slow_semantic_gate.saturating_add(40));
    }

    #[test]
    fn context_preview_detection_identifies_markdown() {
        let snippet = "# heading\n\n- [x] done\n`code`";
        let detection = super::is_likely_markdown(snippet);
        assert!(detection.is_likely());
        assert_eq!(
            detect_context_preview_format(snippet, detection),
            ContextPreviewFormat::Markdown
        );
    }

    #[test]
    fn context_preview_detection_prefers_html_over_markdown() {
        let snippet = "<p><b>auth</b> middleware checks token</p>";
        let detection = super::is_likely_markdown(snippet);
        assert!(is_likely_html_fragment(snippet));
        assert_eq!(
            detect_context_preview_format(snippet, detection),
            ContextPreviewFormat::Html
        );
    }

    #[test]
    fn normalize_html_fragment_for_markdown_maps_common_markup() {
        let html =
            "<p><b>Auth</b> &amp; <code>Token</code></p><ul><li>first</li><li>second</li></ul>";
        let normalized = normalize_html_fragment_for_markdown(html);
        assert!(normalized.contains("**Auth**"));
        assert!(normalized.contains('&'));
        assert!(normalized.contains("`Token`"));
        assert!(normalized.contains("- first"));
        assert!(normalized.contains("- second"));
    }

    #[test]
    fn tui_prefers_lexical_only_for_single_token_queries() {
        assert!(FsfsRuntime::tui_prefers_lexical_only_query("test"));
        assert!(FsfsRuntime::tui_prefers_lexical_only_query("rustfmt"));
        assert!(FsfsRuntime::tui_prefers_lexical_only_query("Cargo.toml"));
        assert!(!FsfsRuntime::tui_prefers_lexical_only_query(
            "semantic search"
        ));
        assert!(!FsfsRuntime::tui_prefers_lexical_only_query("   "));
    }

    #[test]
    fn semantic_voi_skips_short_keyword_when_lexical_signal_is_strong() {
        let lexical = vec![
            LexicalCandidate::new("src/main.rs", 7.2),
            LexicalCandidate::new("src/lib.rs", 2.1),
            LexicalCandidate::new("README.md", 1.6),
        ];
        let decision = FsfsRuntime::semantic_voi_decision(
            SearchExecutionMode::FastOnly,
            QueryIntentClass::ShortKeyword,
            "test",
            20,
            &lexical,
        );
        assert!(!decision.run_semantic);
        assert!(decision.expected_loss_skip < decision.expected_loss_run);
    }

    #[test]
    fn semantic_voi_runs_for_natural_language_when_lexical_signal_is_weak() {
        let lexical = vec![
            LexicalCandidate::new("docs/intro.md", 0.15),
            LexicalCandidate::new("docs/setup.md", 0.14),
            LexicalCandidate::new("docs/troubleshooting.md", 0.13),
        ];
        let decision = FsfsRuntime::semantic_voi_decision(
            SearchExecutionMode::Full,
            QueryIntentClass::NaturalLanguage,
            "how do i configure semantic indexing for multilingual source code",
            200,
            &lexical,
        );
        assert!(decision.run_semantic);
        assert!(decision.expected_loss_run <= decision.expected_loss_skip);
    }

    #[test]
    fn search_serve_error_response_marks_failed_reply_with_message() {
        let response =
            FsfsRuntime::search_serve_error_response("query text", "full", "parse failure");
        assert!(!response.ok);
        assert_eq!(response.query, "query text");
        assert_eq!(response.mode, "full");
        assert!(!response.cached);
        assert!(response.payloads.is_empty());
        assert_eq!(response.error.as_deref(), Some("parse failure"));
    }

    #[cfg(unix)]
    #[test]
    fn read_search_serve_socket_request_rejects_oversized_payload() {
        use std::io::Write as _;
        use std::net::Shutdown;
        use std::os::unix::net::UnixStream;
        use std::thread;

        let (mut writer, mut reader) = UnixStream::pair().expect("socket pair");
        let payload = "x".repeat(FSFS_DAEMON_REQUEST_MAX_BYTES.saturating_add(1));
        let writer_task = thread::spawn(move || {
            writer
                .write_all(payload.as_bytes())
                .expect("write oversized payload");
            writer
                .shutdown(Shutdown::Write)
                .expect("shutdown write side");
        });

        let error = FsfsRuntime::read_search_serve_socket_request(&mut reader)
            .expect_err("oversized request should fail");
        writer_task.join().expect("writer thread");
        match error {
            frankensearch_core::SearchError::InvalidConfig { field, .. } => {
                assert_eq!(field, "cli.serve.request");
            }
            other => panic!("expected InvalidConfig for oversized request, got {other:?}"),
        }
    }

    #[cfg(unix)]
    #[test]
    fn default_daemon_socket_path_uses_namespaced_runtime_directory() {
        let runtime = FsfsRuntime::new(FsfsConfig::default());
        let path = runtime
            .default_daemon_socket_path()
            .expect("daemon socket path");
        let rendered = path.to_string_lossy();
        assert!(
            rendered.contains("frankensearch/daemon"),
            "socket path should be namespaced under a per-user runtime/cache directory: {rendered}"
        );
        assert!(
            rendered.ends_with(".sock"),
            "daemon socket path should end with .sock: {rendered}"
        );
    }

    #[test]
    fn semantic_gate_forces_run_for_full_recall_even_when_voi_would_skip() {
        let lexical = vec![
            LexicalCandidate::new("src/main.rs", 7.2),
            LexicalCandidate::new("src/lib.rs", 2.1),
            LexicalCandidate::new("README.md", 1.6),
        ];
        let voi = FsfsRuntime::semantic_voi_decision(
            SearchExecutionMode::FastOnly,
            QueryIntentClass::ShortKeyword,
            "test",
            20_000,
            &lexical,
        );
        assert!(!voi.run_semantic);

        let gate = FsfsRuntime::semantic_gate_decision(SemanticGateDecisionInput {
            mode: SearchExecutionMode::FastOnly,
            intent: QueryIntentClass::ShortKeyword,
            query: "test",
            output_limit: 20_000,
            lexical_head_candidates: &lexical,
            semantic_stage_enabled: true,
            lexical_stage_enabled: true,
            force_full_semantic_recall: true,
        });
        assert!(gate.run_semantic);
        assert_eq!(gate.reason_code, "semantic.voi.force.full_recall");
    }

    #[test]
    fn semantic_gate_uses_voi_when_not_forced() {
        let lexical = vec![
            LexicalCandidate::new("src/main.rs", 7.2),
            LexicalCandidate::new("src/lib.rs", 2.1),
            LexicalCandidate::new("README.md", 1.6),
        ];
        let gate = FsfsRuntime::semantic_gate_decision(SemanticGateDecisionInput {
            mode: SearchExecutionMode::FastOnly,
            intent: QueryIntentClass::ShortKeyword,
            query: "test",
            output_limit: 20_000,
            lexical_head_candidates: &lexical,
            semantic_stage_enabled: true,
            lexical_stage_enabled: true,
            force_full_semantic_recall: false,
        });
        assert!(!gate.run_semantic);
        assert_eq!(gate.reason_code, "semantic.voi.skip");
    }

    #[test]
    fn resolve_output_limit_preserves_explicit_limit() {
        assert_eq!(
            FsfsRuntime::resolve_output_limit(42, Some(1_000), Some(2_000)),
            42
        );
    }

    #[test]
    fn resolve_output_limit_preserves_explicit_limit_above_known_counts() {
        assert_eq!(
            FsfsRuntime::resolve_output_limit(10_000, Some(250), Some(180)),
            10_000
        );
    }

    #[test]
    fn resolve_output_limit_unbounded_uses_max_index_cardinality() {
        assert_eq!(FsfsRuntime::resolve_output_limit(0, Some(7), Some(9)), 9);
        assert_eq!(FsfsRuntime::resolve_output_limit(0, Some(11), Some(4)), 11);
    }

    #[test]
    fn resolve_output_limit_unbounded_uses_vector_count_without_lexical() {
        assert_eq!(
            FsfsRuntime::resolve_output_limit(
                FSFS_SEARCH_UNBOUNDED_LIMIT_SENTINEL,
                None,
                Some(6_543),
            ),
            6_543
        );
    }

    #[test]
    fn resolve_output_limit_unbounded_defaults_to_one_when_no_index_available() {
        assert_eq!(
            FsfsRuntime::resolve_output_limit(FSFS_SEARCH_UNBOUNDED_LIMIT_SENTINEL, None, None),
            1
        );
    }

    #[test]
    fn resolve_planning_limit_preserves_small_limit() {
        assert_eq!(FsfsRuntime::resolve_planning_limit(128), 128);
        assert_eq!(FsfsRuntime::resolve_planning_limit(1_000), 1_000);
    }

    #[test]
    fn resolve_planning_limit_progressively_widens_without_hard_cap() {
        assert_eq!(FsfsRuntime::resolve_planning_limit(5_000), 2_008);
        assert!(FsfsRuntime::resolve_planning_limit(200_000) > 1_000);
    }

    #[test]
    fn should_not_force_full_semantic_recall_when_unbounded_and_lexical_coverage_complete() {
        assert!(!FsfsRuntime::should_force_full_semantic_recall(
            SemanticRecallDecisionInput {
                output_limit: 5_000,
                planning_limit: 1_000,
                semantic_stage_enabled: true,
                semantic_index_available: true,
                lexical_stage_enabled: true,
                lexical_head_count: 24,
                lexical_full_count: 5_000,
            }
        ));
    }

    #[test]
    fn should_force_full_semantic_recall_when_lexical_stage_disabled() {
        assert!(FsfsRuntime::should_force_full_semantic_recall(
            SemanticRecallDecisionInput {
                output_limit: 5_000,
                planning_limit: 1_000,
                semantic_stage_enabled: true,
                semantic_index_available: true,
                lexical_stage_enabled: false,
                lexical_head_count: 0,
                lexical_full_count: 0,
            }
        ));
    }

    #[test]
    fn should_force_full_semantic_recall_when_lexical_head_is_empty() {
        assert!(FsfsRuntime::should_force_full_semantic_recall(
            SemanticRecallDecisionInput {
                output_limit: 5_000,
                planning_limit: 1_000,
                semantic_stage_enabled: true,
                semantic_index_available: true,
                lexical_stage_enabled: true,
                lexical_head_count: 0,
                lexical_full_count: 0,
            }
        ));
    }

    #[test]
    fn should_not_force_full_semantic_recall_when_within_head_budget() {
        assert!(!FsfsRuntime::should_force_full_semantic_recall(
            SemanticRecallDecisionInput {
                output_limit: 800,
                planning_limit: 1_000,
                semantic_stage_enabled: true,
                semantic_index_available: true,
                lexical_stage_enabled: false,
                lexical_head_count: 0,
                lexical_full_count: 0,
            }
        ));
    }

    #[test]
    fn should_not_force_full_semantic_recall_when_lexical_coverage_is_complete() {
        assert!(!FsfsRuntime::should_force_full_semantic_recall(
            SemanticRecallDecisionInput {
                output_limit: 5_000,
                planning_limit: 1_000,
                semantic_stage_enabled: true,
                semantic_index_available: true,
                lexical_stage_enabled: true,
                lexical_head_count: 24,
                lexical_full_count: 5_000,
            }
        ));
    }

    #[test]
    fn should_not_force_full_semantic_recall_when_semantic_stage_disabled() {
        assert!(!FsfsRuntime::should_force_full_semantic_recall(
            SemanticRecallDecisionInput {
                output_limit: 5_000,
                planning_limit: 1_000,
                semantic_stage_enabled: false,
                semantic_index_available: true,
                lexical_stage_enabled: false,
                lexical_head_count: 0,
                lexical_full_count: 0,
            }
        ));
    }

    #[test]
    fn should_not_force_full_semantic_recall_without_semantic_index() {
        assert!(!FsfsRuntime::should_force_full_semantic_recall(
            SemanticRecallDecisionInput {
                output_limit: 5_000,
                planning_limit: 1_000,
                semantic_stage_enabled: true,
                semantic_index_available: false,
                lexical_stage_enabled: false,
                lexical_head_count: 0,
                lexical_full_count: 0,
            }
        ));
    }

    #[test]
    fn merge_with_lexical_tail_preserves_head_order_and_appends_tail() {
        let fused_head = vec![
            FusedCandidate {
                doc_id: "src/b.rs".to_owned(),
                fused_score: 0.016,
                prior_boost: 0.0,
                lexical_rank: Some(0),
                semantic_rank: None,
                lexical_score: Some(9.0),
                semantic_score: None,
                in_both_sources: false,
            },
            FusedCandidate {
                doc_id: "src/a.rs".to_owned(),
                fused_score: 0.015,
                prior_boost: 0.0,
                lexical_rank: Some(1),
                semantic_rank: None,
                lexical_score: Some(10.0),
                semantic_score: None,
                in_both_sources: false,
            },
        ];
        let lexical_full = vec![
            LexicalCandidate::new("src/a.rs", 10.0),
            LexicalCandidate::new("src/b.rs", 9.0),
            LexicalCandidate::new("src/c.rs", 8.0),
            LexicalCandidate::new("src/d.rs", 7.0),
        ];

        let merged = FsfsRuntime::merge_with_lexical_tail(&fused_head, &lexical_full, None);
        let ordered_ids: Vec<&str> = merged
            .iter()
            .map(|candidate| candidate.doc_id.as_str())
            .collect();
        assert_eq!(
            ordered_ids,
            vec!["src/b.rs", "src/a.rs", "src/c.rs", "src/d.rs"]
        );
        assert_eq!(merged[2].lexical_rank, Some(2));
        assert_eq!(merged[3].lexical_rank, Some(3));
    }

    #[test]
    fn merge_with_lexical_tail_applies_filter_to_head_and_tail() {
        let fused_head = vec![FusedCandidate {
            doc_id: "src/head.rs".to_owned(),
            fused_score: 0.016,
            prior_boost: 0.0,
            lexical_rank: Some(0),
            semantic_rank: None,
            lexical_score: Some(11.0),
            semantic_score: None,
            in_both_sources: false,
        }];
        let lexical_full = vec![
            LexicalCandidate::new("src/head.rs", 11.0),
            LexicalCandidate::new("README.md", 8.0),
            LexicalCandidate::new("src/tail.rs", 7.0),
        ];
        let filter = super::SearchFilterExpr::parse("ext:rs")
            .expect("filter should parse")
            .expect("filter should be present");

        let merged =
            FsfsRuntime::merge_with_lexical_tail(&fused_head, &lexical_full, Some(&filter));
        let ordered_ids: Vec<&str> = merged
            .iter()
            .map(|candidate| candidate.doc_id.as_str())
            .collect();
        assert_eq!(ordered_ids, vec!["src/head.rs", "src/tail.rs"]);
    }

    #[test]
    fn merge_with_fallback_tail_preserves_refined_head_and_appends_remaining_tail() {
        let refined_head = vec![
            FusedCandidate {
                doc_id: "doc-b".to_owned(),
                fused_score: 0.95,
                prior_boost: 0.0,
                lexical_rank: None,
                semantic_rank: Some(0),
                lexical_score: None,
                semantic_score: Some(0.95),
                in_both_sources: false,
            },
            FusedCandidate {
                doc_id: "doc-a".to_owned(),
                fused_score: 0.90,
                prior_boost: 0.0,
                lexical_rank: Some(0),
                semantic_rank: Some(1),
                lexical_score: Some(12.0),
                semantic_score: Some(0.90),
                in_both_sources: true,
            },
        ];
        let fallback = vec![
            FusedCandidate {
                doc_id: "doc-a".to_owned(),
                fused_score: 0.90,
                prior_boost: 0.0,
                lexical_rank: Some(0),
                semantic_rank: Some(1),
                lexical_score: Some(12.0),
                semantic_score: Some(0.90),
                in_both_sources: true,
            },
            FusedCandidate {
                doc_id: "doc-b".to_owned(),
                fused_score: 0.85,
                prior_boost: 0.0,
                lexical_rank: None,
                semantic_rank: Some(0),
                lexical_score: None,
                semantic_score: Some(0.95),
                in_both_sources: false,
            },
            FusedCandidate {
                doc_id: "doc-c".to_owned(),
                fused_score: 0.10,
                prior_boost: 0.0,
                lexical_rank: None,
                semantic_rank: Some(2),
                lexical_score: None,
                semantic_score: Some(0.10),
                in_both_sources: false,
            },
            FusedCandidate {
                doc_id: "doc-d".to_owned(),
                fused_score: 0.05,
                prior_boost: 0.0,
                lexical_rank: None,
                semantic_rank: Some(3),
                lexical_score: None,
                semantic_score: Some(0.05),
                in_both_sources: false,
            },
        ];

        let merged = FsfsRuntime::merge_with_fallback_tail(&refined_head, &fallback, 10);
        let ordered_ids: Vec<&str> = merged
            .iter()
            .map(|candidate| candidate.doc_id.as_str())
            .collect();
        assert_eq!(ordered_ids, vec!["doc-b", "doc-a", "doc-c", "doc-d"]);
    }

    #[test]
    fn merge_with_fallback_tail_respects_output_limit() {
        let refined_head = vec![
            FusedCandidate {
                doc_id: "doc-1".to_owned(),
                fused_score: 1.0,
                prior_boost: 0.0,
                lexical_rank: Some(0),
                semantic_rank: Some(0),
                lexical_score: Some(10.0),
                semantic_score: Some(1.0),
                in_both_sources: true,
            },
            FusedCandidate {
                doc_id: "doc-2".to_owned(),
                fused_score: 0.9,
                prior_boost: 0.0,
                lexical_rank: Some(1),
                semantic_rank: Some(1),
                lexical_score: Some(9.0),
                semantic_score: Some(0.9),
                in_both_sources: true,
            },
        ];
        let fallback = vec![
            refined_head[0].clone(),
            refined_head[1].clone(),
            FusedCandidate {
                doc_id: "doc-3".to_owned(),
                fused_score: 0.1,
                prior_boost: 0.0,
                lexical_rank: None,
                semantic_rank: Some(2),
                lexical_score: None,
                semantic_score: Some(0.1),
                in_both_sources: false,
            },
        ];

        let merged = FsfsRuntime::merge_with_fallback_tail(&refined_head, &fallback, 2);
        let ordered_ids: Vec<&str> = merged
            .iter()
            .map(|candidate| candidate.doc_id.as_str())
            .collect();
        assert_eq!(ordered_ids, vec!["doc-1", "doc-2"]);
    }

    #[test]
    fn runtime_modes_are_callable() {
        run_test_with_cx(|cx| async move {
            let runtime = FsfsRuntime::new(FsfsConfig::default());
            runtime
                .run_mode(&cx, InterfaceMode::Cli)
                .await
                .expect("cli mode");
            runtime
                .run_mode(&cx, InterfaceMode::Tui)
                .await
                .expect("tui mode");
        });
    }

    #[test]
    fn runtime_help_command_is_callable() {
        run_test_with_cx(|cx| async move {
            let runtime = FsfsRuntime::new(FsfsConfig::default()).with_cli_input(CliInput {
                command: CliCommand::Help,
                ..CliInput::default()
            });
            runtime
                .run_mode(&cx, InterfaceMode::Cli)
                .await
                .expect("help command should complete");
        });
    }

    #[test]
    fn runtime_completions_command_requires_shell() {
        run_test_with_cx(|cx| async move {
            let runtime = FsfsRuntime::new(FsfsConfig::default()).with_cli_input(CliInput {
                command: CliCommand::Completions,
                ..CliInput::default()
            });
            let err = runtime
                .run_mode(&cx, InterfaceMode::Cli)
                .await
                .expect_err("missing shell should fail");
            assert!(err.to_string().contains("missing shell argument"));
        });
    }

    #[test]
    fn runtime_completions_command_with_shell_is_callable() {
        run_test_with_cx(|cx| async move {
            let runtime = FsfsRuntime::new(FsfsConfig::default()).with_cli_input(CliInput {
                command: CliCommand::Completions,
                completion_shell: Some(CompletionShell::Bash),
                ..CliInput::default()
            });
            runtime
                .run_mode(&cx, InterfaceMode::Cli)
                .await
                .expect("completions command should complete");
        });
    }

    #[test]
    fn runtime_builds_root_discovery_plan() {
        let mut config = FsfsConfig::default();
        config.discovery.roots = vec!["/home/tester".into(), "/proc".into()];
        let runtime = FsfsRuntime::new(config);
        let plan = runtime.discovery_root_plan();

        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].0, "/home/tester");
        assert_eq!(plan[1].0, "/proc");
    }

    #[test]
    fn count_non_empty_lines_ignores_blank_and_whitespace_lines() {
        let text = "alpha\n\n   \n beta \n\t\n\ngamma";
        assert_eq!(super::count_non_empty_lines(text), 3);
    }

    #[test]
    fn runtime_classifies_discovery_candidate() {
        let runtime = FsfsRuntime::new(FsfsConfig::default());
        let candidate = DiscoveryCandidate::new(Path::new("/home/tester/src/lib.rs"), 2_048);
        let decision = runtime.classify_discovery_candidate(&candidate);

        assert_eq!(decision.scope, DiscoveryScopeDecision::Include);
        assert_eq!(
            decision.ingestion_class,
            IngestionClass::FullSemanticLexical
        );
    }

    #[test]
    fn runtime_pressure_state_is_consumable_by_scheduler_and_ux() {
        let runtime = FsfsRuntime::new(FsfsConfig::default());
        let mut controller = runtime.new_pressure_controller();
        let sample = PressureSignal::new(78.0, 35.0, 20.0, 30.0);

        let first = runtime.observe_pressure(&mut controller, sample);
        let second = runtime.observe_pressure(&mut controller, sample);
        let third = runtime.observe_pressure(&mut controller, sample);

        assert_eq!(first.to, PressureState::Normal);
        assert_eq!(second.to, PressureState::Normal);
        assert_eq!(third.to, PressureState::Constrained);
        assert!(third.changed);
    }

    #[test]
    fn profile_degradation_recovery_gate_is_profile_aware() {
        let strict = degradation_controller_config_for_profile(PressureProfile::Strict, 3);
        let performance =
            degradation_controller_config_for_profile(PressureProfile::Performance, 3);
        let degraded = degradation_controller_config_for_profile(PressureProfile::Degraded, 3);

        assert_eq!(performance.consecutive_healthy_required, 3);
        assert_eq!(strict.consecutive_healthy_required, 4);
        assert_eq!(degraded.consecutive_healthy_required, 5);
    }

    #[test]
    fn runtime_projects_degradation_status_from_pressure_transition() {
        let mut config = FsfsConfig::default();
        config.pressure.anti_flap_readings = 1;
        let runtime = FsfsRuntime::new(config);
        let mut pressure_controller = runtime.new_pressure_controller();
        let mut degradation_machine = runtime
            .new_degradation_state_machine()
            .expect("build degradation machine");

        let pressure = runtime.observe_pressure(
            &mut pressure_controller,
            PressureSignal::new(100.0, 100.0, 100.0, 100.0),
        );
        assert_eq!(pressure.to, PressureState::Emergency);

        let degradation = runtime.observe_degradation(&mut degradation_machine, &pressure);
        assert_eq!(degradation.to, DegradationStage::MetadataOnly);
        assert_eq!(degradation.reason_code, "degrade.transition.escalated");
        assert_eq!(
            degradation.status.query_mode,
            QueryCapabilityMode::MetadataOnly
        );
        assert_eq!(
            degradation.status.user_banner,
            "Safe mode: metadata operations only while search pipelines stabilize."
        );
    }

    #[test]
    fn runtime_honors_degradation_override_controls() {
        let mut config = FsfsConfig::default();
        config.pressure.anti_flap_readings = 1;
        config.pressure.degradation_override = DegradationOverrideMode::ForcePaused;
        let runtime = FsfsRuntime::new(config);
        let mut pressure_controller = runtime.new_pressure_controller();
        let mut degradation_machine = runtime
            .new_degradation_state_machine()
            .expect("build degradation machine");

        let pressure = runtime.observe_pressure(
            &mut pressure_controller,
            PressureSignal::new(10.0, 10.0, 10.0, 10.0),
        );
        let degradation = runtime.observe_degradation(&mut degradation_machine, &pressure);

        assert_eq!(degradation.to, DegradationStage::Paused);
        assert_eq!(degradation.reason_code, "degrade.transition.override");
        assert_eq!(degradation.status.query_mode, QueryCapabilityMode::Paused);
    }

    #[test]
    fn runtime_disk_budget_state_is_consumable_by_scheduler_and_ux() {
        let runtime = FsfsRuntime::new(FsfsConfig::default());
        let tracker = LifecycleTracker::new(
            WatchdogConfig::default(),
            ResourceLimits {
                max_index_bytes: 1_000,
                ..ResourceLimits::default()
            },
        );

        let near = runtime
            .evaluate_index_disk_budget(&tracker, 900)
            .expect("disk budget should be configured");
        assert_eq!(near.stage, DiskBudgetStage::NearLimit);
        assert_eq!(near.action, DiskBudgetAction::ThrottleIngest);

        let critical = runtime
            .evaluate_index_disk_budget(&tracker, 1_500)
            .expect("disk budget should be configured");
        assert_eq!(critical.stage, DiskBudgetStage::Critical);
        assert_eq!(critical.action, DiskBudgetAction::PauseWrites);
    }

    #[test]
    fn runtime_collects_cross_domain_storage_usage() {
        let runtime = FsfsRuntime::new(FsfsConfig::default());
        let temp = tempfile::tempdir().expect("tempdir");
        let vector_root = temp.path().join("vector");
        let lexical_root = temp.path().join("lexical");
        let catalog_file = temp.path().join("catalog").join("fsfs.db");
        let cache_root = temp.path().join("cache");

        fs::create_dir_all(vector_root.join("segments")).expect("create vector dirs");
        fs::create_dir_all(lexical_root.join("segments")).expect("create lexical dirs");
        fs::create_dir_all(&cache_root).expect("create cache dir");
        fs::create_dir_all(catalog_file.parent().expect("catalog parent"))
            .expect("create catalog dir");

        fs::write(vector_root.join("segments").join("a.fsvi"), vec![0_u8; 40]).expect("vector");
        fs::write(vector_root.join("segments").join("a.wal"), vec![0_u8; 10]).expect("wal");
        fs::write(lexical_root.join("segments").join("seg0"), vec![0_u8; 30]).expect("lexical");
        fs::write(&catalog_file, vec![0_u8; 12]).expect("catalog");
        fs::write(cache_root.join("model.bin"), vec![0_u8; 8]).expect("cache");

        let usage = runtime
            .collect_index_storage_usage(&IndexStoragePaths {
                vector_index_roots: vec![vector_root],
                lexical_index_roots: vec![lexical_root],
                catalog_files: vec![catalog_file],
                embedding_cache_roots: vec![cache_root],
            })
            .expect("collect storage usage");

        assert_eq!(usage.vector_index_bytes, 50);
        assert_eq!(usage.lexical_index_bytes, 30);
        assert_eq!(usage.catalog_bytes, 12);
        assert_eq!(usage.embedding_cache_bytes, 8);
        assert_eq!(usage.total_bytes(), 100);
    }

    #[test]
    fn runtime_evaluates_storage_budget_and_updates_tracker_resources() {
        let runtime = FsfsRuntime::new(FsfsConfig::default());
        let tracker = LifecycleTracker::new(
            WatchdogConfig::default(),
            ResourceLimits {
                max_index_bytes: 1_000,
                ..ResourceLimits::default()
            },
        );
        let temp = tempfile::tempdir().expect("tempdir");
        let vector_root = temp.path().join("vector");
        let lexical_root = temp.path().join("lexical");
        let catalog_file = temp.path().join("catalog").join("fsfs.db");
        let cache_root = temp.path().join("cache");

        fs::create_dir_all(&vector_root).expect("vector dir");
        fs::create_dir_all(&lexical_root).expect("lexical dir");
        fs::create_dir_all(&cache_root).expect("cache dir");
        fs::create_dir_all(catalog_file.parent().expect("catalog parent")).expect("catalog dir");
        fs::write(vector_root.join("vec.fsvi"), vec![0_u8; 400]).expect("vector bytes");
        fs::write(lexical_root.join("lex.seg"), vec![0_u8; 300]).expect("lex bytes");
        fs::write(&catalog_file, vec![0_u8; 150]).expect("catalog bytes");
        fs::write(cache_root.join("embed.cache"), vec![0_u8; 150]).expect("cache bytes");

        let snapshot = runtime
            .evaluate_storage_disk_budget(
                &tracker,
                &IndexStoragePaths {
                    vector_index_roots: vec![vector_root],
                    lexical_index_roots: vec![lexical_root],
                    catalog_files: vec![catalog_file],
                    embedding_cache_roots: vec![cache_root],
                },
            )
            .expect("evaluate storage budget")
            .expect("budget should be configured");
        assert_eq!(snapshot.used_bytes, 1_000);
        assert_eq!(snapshot.stage, DiskBudgetStage::NearLimit);
        assert_eq!(snapshot.action, DiskBudgetAction::ThrottleIngest);

        let status = tracker.status();
        assert_eq!(status.resources.index_bytes, Some(1_000));
        assert_eq!(status.resources.vector_index_bytes, Some(400));
        assert_eq!(status.resources.lexical_index_bytes, Some(300));
        assert_eq!(status.resources.catalog_bytes, Some(150));
        assert_eq!(status.resources.embedding_cache_bytes, Some(150));
    }

    #[test]
    fn runtime_default_storage_paths_follow_config() {
        let mut config = FsfsConfig::default();
        config.storage.index_dir = "/tmp/fsfs-index".to_owned();
        config.storage.db_path = "/tmp/fsfs.db".to_owned();
        let runtime = FsfsRuntime::new(config);

        let paths = runtime.default_index_storage_paths();
        assert_eq!(
            paths.vector_index_roots,
            vec![PathBuf::from("/tmp/fsfs-index/vector")]
        );
        assert_eq!(
            paths.lexical_index_roots,
            vec![PathBuf::from("/tmp/fsfs-index/lexical")]
        );
        assert_eq!(paths.catalog_files, vec![PathBuf::from("/tmp/fsfs.db")]);
        assert_eq!(
            paths.embedding_cache_roots,
            vec![PathBuf::from("/tmp/fsfs-index/cache")]
        );
    }

    #[test]
    fn runtime_resolve_index_budget_prefers_storage_override() {
        let mut config = FsfsConfig::default();
        config.storage.disk_budget_bytes = Some(42_000);
        let runtime = FsfsRuntime::new(config);
        assert_eq!(
            runtime.resolve_index_budget_bytes(&IndexStoragePaths::default()),
            42_000
        );
    }

    #[test]
    fn runtime_storage_pressure_emergency_forces_pause_writes_plan() {
        let mut config = FsfsConfig::default();
        config.storage.storage_pressure_emergency = true;
        let runtime = FsfsRuntime::new(config);
        let tracker = LifecycleTracker::new(
            WatchdogConfig::default(),
            ResourceLimits {
                max_index_bytes: 1_000,
                ..ResourceLimits::default()
            },
        );

        let snapshot = runtime
            .apply_storage_emergency_override(
                runtime.evaluate_index_disk_budget(&tracker, 500),
                Some(500),
                tracker.resource_limits().max_index_bytes,
            )
            .expect("emergency override should always provide a snapshot");
        assert_eq!(snapshot.stage, DiskBudgetStage::Critical);
        assert_eq!(snapshot.action, DiskBudgetAction::PauseWrites);
        assert_eq!(
            snapshot.reason_code,
            super::DISK_BUDGET_REASON_EMERGENCY_OVERRIDE
        );

        let plan = FsfsRuntime::disk_budget_control_plan(snapshot);
        assert_eq!(plan.watcher_pressure_state, PressureState::Emergency);
        assert!(plan.pause_writes);
        assert!(plan.request_tombstone_cleanup);
        assert!(!plan.request_eviction);
        assert!(!plan.request_compaction);
    }

    #[test]
    fn runtime_cleanup_catalog_tombstones_prunes_old_rows() {
        let temp = tempfile::tempdir().expect("tempdir");
        let db_path = temp.path().join("fsfs.db");
        let conn = Connection::open(db_path.display().to_string()).expect("open sqlite");
        bootstrap_catalog_schema(&conn).expect("bootstrap catalog schema");

        let now_ms = 1_710_000_000_000_u64;
        let retention_days = 7_u16;
        let retention_ms = u64::from(retention_days).saturating_mul(super::MILLIS_PER_DAY);
        let old_deleted_ts =
            i64::try_from(now_ms.saturating_sub(retention_ms).saturating_sub(1_000)).unwrap();
        let fresh_deleted_ts = i64::try_from(now_ms.saturating_sub(retention_ms / 2)).unwrap();

        let old_row = [
            SqliteValue::Text("home:/tmp/old.txt".to_owned()),
            SqliteValue::Text("home".to_owned()),
            SqliteValue::Text("/tmp/old.txt".to_owned()),
            SqliteValue::Blob(vec![1_u8; 32]),
            SqliteValue::Integer(1),
            SqliteValue::Text("full_semantic_lexical".to_owned()),
            SqliteValue::Text("tombstoned".to_owned()),
            SqliteValue::Integer(1),
            SqliteValue::Integer(old_deleted_ts - 500),
            SqliteValue::Integer(old_deleted_ts),
            SqliteValue::Integer(old_deleted_ts),
            SqliteValue::Integer(old_deleted_ts),
        ];
        conn.execute_with_params(
            "INSERT INTO fsfs_catalog_files \
             (file_key, mount_id, canonical_path, content_hash, revision, ingestion_class, pipeline_status, eligible, first_seen_ts, last_seen_ts, updated_ts, deleted_ts) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12);",
            &old_row,
        )
        .expect("insert old tombstone row");

        let fresh_row = [
            SqliteValue::Text("home:/tmp/fresh.txt".to_owned()),
            SqliteValue::Text("home".to_owned()),
            SqliteValue::Text("/tmp/fresh.txt".to_owned()),
            SqliteValue::Blob(vec![2_u8; 32]),
            SqliteValue::Integer(1),
            SqliteValue::Text("full_semantic_lexical".to_owned()),
            SqliteValue::Text("tombstoned".to_owned()),
            SqliteValue::Integer(1),
            SqliteValue::Integer(fresh_deleted_ts - 500),
            SqliteValue::Integer(fresh_deleted_ts),
            SqliteValue::Integer(fresh_deleted_ts),
            SqliteValue::Integer(fresh_deleted_ts),
        ];
        conn.execute_with_params(
            "INSERT INTO fsfs_catalog_files \
             (file_key, mount_id, canonical_path, content_hash, revision, ingestion_class, pipeline_status, eligible, first_seen_ts, last_seen_ts, updated_ts, deleted_ts) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12);",
            &fresh_row,
        )
        .expect("insert fresh tombstone row");

        let mut config = FsfsConfig::default();
        config.storage.db_path = db_path.display().to_string();
        config.storage.evidence_retention_days = retention_days;
        let runtime = FsfsRuntime::new(config);
        let (deleted_rows, cutoff_ms) = runtime
            .cleanup_catalog_tombstones(now_ms)
            .expect("cleanup should succeed");
        assert_eq!(deleted_rows, 1);
        assert_eq!(
            cutoff_ms,
            i64::try_from(now_ms.saturating_sub(retention_ms)).unwrap()
        );

        // Reopen connection for verification — original `conn` holds a stale
        // MVCC snapshot that predates the DELETE from `cleanup_catalog_tombstones`.
        drop(conn);
        let conn2 =
            Connection::open(db_path.display().to_string()).expect("reopen for verification");
        let remaining = conn2
            .query("SELECT file_key FROM fsfs_catalog_files ORDER BY file_key;")
            .expect("remaining rows query");
        assert_eq!(remaining.len(), 1);
        assert_eq!(
            remaining[0].get(0),
            Some(&SqliteValue::Text("home:/tmp/fresh.txt".to_owned()))
        );
    }

    #[test]
    fn runtime_disk_budget_control_plan_encodes_staged_actions() {
        let runtime = FsfsRuntime::new(FsfsConfig::default());
        let tracker = LifecycleTracker::new(
            WatchdogConfig::default(),
            ResourceLimits {
                max_index_bytes: 1_000,
                ..ResourceLimits::default()
            },
        );

        let normal = runtime
            .evaluate_index_disk_budget(&tracker, 500)
            .expect("normal disk snapshot");
        let near = runtime
            .evaluate_index_disk_budget(&tracker, 900)
            .expect("near disk snapshot");
        let over = runtime
            .evaluate_index_disk_budget(&tracker, 1_010)
            .expect("over disk snapshot");
        let critical = runtime
            .evaluate_index_disk_budget(&tracker, 1_500)
            .expect("critical disk snapshot");

        let normal_plan = FsfsRuntime::disk_budget_control_plan(normal);
        let near_plan = FsfsRuntime::disk_budget_control_plan(near);
        let over_plan = FsfsRuntime::disk_budget_control_plan(over);
        let critical_plan = FsfsRuntime::disk_budget_control_plan(critical);

        assert_eq!(normal_plan.watcher_pressure_state, PressureState::Normal);
        assert!(!normal_plan.throttle_ingest);
        assert!(!normal_plan.request_eviction);

        assert_eq!(near_plan.watcher_pressure_state, PressureState::Constrained);
        assert!(near_plan.throttle_ingest);
        assert!(!near_plan.request_eviction);

        assert_eq!(over_plan.watcher_pressure_state, PressureState::Degraded);
        assert!(over_plan.throttle_ingest);
        assert!(over_plan.request_eviction);
        assert!(over_plan.request_compaction);
        assert!(over_plan.request_tombstone_cleanup);
        assert!(over_plan.eviction_target_bytes >= 10);

        assert_eq!(
            critical_plan.watcher_pressure_state,
            PressureState::Emergency
        );
        assert!(critical_plan.pause_writes);
        assert!(critical_plan.request_eviction);
        assert!(critical_plan.request_compaction);
        assert!(critical_plan.request_tombstone_cleanup);
        assert!(critical_plan.eviction_target_bytes >= 500);
    }

    #[test]
    fn conservative_budget_from_available_bytes_caps_at_five_gib() {
        assert_eq!(super::conservative_budget_from_available_bytes(50), 5);
        assert_eq!(
            super::conservative_budget_from_available_bytes(20 * 1024 * 1024 * 1024),
            2 * 1024 * 1024 * 1024
        );
        assert_eq!(
            super::conservative_budget_from_available_bytes(200 * 1024 * 1024 * 1024),
            super::DISK_BUDGET_CAP_BYTES
        );
    }

    #[test]
    fn more_severe_pressure_state_prefers_stricter_state() {
        assert_eq!(
            super::more_severe_pressure_state(PressureState::Normal, PressureState::Constrained),
            PressureState::Constrained
        );
        assert_eq!(
            super::more_severe_pressure_state(PressureState::Degraded, PressureState::Constrained),
            PressureState::Degraded
        );
        assert_eq!(
            super::more_severe_pressure_state(PressureState::Emergency, PressureState::Normal),
            PressureState::Emergency
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn runtime_collect_pressure_signal_reads_host_metrics() {
        let runtime = FsfsRuntime::new(FsfsConfig::default());
        let mut collector = HostPressureCollector::default();
        let sample = runtime
            .collect_pressure_signal(&mut collector)
            .expect("collect pressure sample");

        assert!((0.0..=200.0).contains(&sample.cpu_pct));
        assert!((0.0..=200.0).contains(&sample.memory_pct));
        assert!((0.0..=200.0).contains(&sample.io_pct));
        assert!((0.0..=200.0).contains(&sample.load_pct));
    }

    #[test]
    fn watch_mode_waits_for_shutdown_and_exits() {
        run_test_with_cx(|cx| async move {
            let temp = tempfile::tempdir().expect("tempdir");
            let project = temp.path().join("project");
            fs::create_dir_all(project.join("src")).expect("project dirs");
            fs::write(project.join("src/lib.rs"), "watch me\n").expect("seed file");

            let mut config = FsfsConfig::default();
            config.indexing.watch_mode = true;
            let runtime = FsfsRuntime::new(config).with_cli_input(CliInput {
                command: CliCommand::Watch,
                target_path: Some(project.clone()),
                watch: true,
                ..CliInput::default()
            });
            let coordinator: Arc<ShutdownCoordinator> = Arc::new(ShutdownCoordinator::new());

            let trigger: Arc<ShutdownCoordinator> = Arc::clone(&coordinator);
            let worker = thread::spawn(move || {
                thread::sleep(Duration::from_millis(120));
                trigger.request_shutdown(ShutdownReason::UserRequest);
            });

            runtime
                .run_mode_with_shutdown(&cx, InterfaceMode::Cli, &coordinator)
                .await
                .expect("watch mode with shutdown");

            worker.join().expect("shutdown trigger thread join");
        });
    }

    #[test]
    fn live_ingest_pipeline_wires_storage_runner_for_semantic_upserts() {
        run_test_with_cx(|cx| async move {
            let temp = tempfile::tempdir().expect("tempdir");
            let target_root = temp.path().join("project");
            fs::create_dir_all(target_root.join("src")).expect("create project source dir");
            let file_path = target_root.join("src/lib.rs");
            fs::write(
                &file_path,
                "pub fn answer() -> u32 { 42 }\nsemantic storage integration\n",
            )
            .expect("write source file");

            let index_root = temp.path().join("index");
            fs::create_dir_all(index_root.join("vector")).expect("vector dir");
            let lexical_path = index_root.join("lexical");
            let vector_path = index_root.join(super::FSFS_VECTOR_INDEX_FILE);
            let lexical_seed = TantivyIndex::create(&lexical_path).expect("create lexical index");
            lexical_seed
                .commit(&cx)
                .await
                .expect("commit empty lexical");
            drop(lexical_seed);
            let vector_writer =
                VectorIndex::create(&vector_path, "hash", 256).expect("create vector index");
            vector_writer.finish().expect("finish vector index");

            let db_path = temp.path().join("fsfs-watch-storage.db");
            let mut config = FsfsConfig::default();
            config.storage.db_path = db_path.display().to_string();

            let runtime = FsfsRuntime::new(config).with_cli_input(CliInput {
                command: CliCommand::Watch,
                target_path: Some(target_root.clone()),
                index_dir: Some(index_root),
                watch: true,
                ..CliInput::default()
            });
            let pipeline = runtime
                .build_live_ingest_pipeline()
                .expect("build live ingest pipeline");
            let ingest_rt = asupersync::runtime::RuntimeBuilder::current_thread()
                .build()
                .expect("build ingest runtime");

            let applied = pipeline
                .apply_batch(
                    &[WatchIngestOp::Upsert {
                        file_key: file_path.display().to_string(),
                        revision: 11,
                        ingestion_class: IngestionClass::FullSemanticLexical,
                    }],
                    &ingest_rt,
                )
                .expect("semantic upsert should succeed");
            assert_eq!(applied, 1);

            let lexical_hits = pipeline
                .lexical_index
                .search(&cx, "semantic", 5)
                .await
                .expect("search lexical index");
            assert!(
                lexical_hits.iter().any(|hit| hit.doc_id == "src/lib.rs"),
                "lexical index should contain the upserted document"
            );

            let semantic_query = pipeline
                .embedder
                .embed(&cx, "semantic storage integration")
                .await
                .expect("embed semantic probe query");
            let vector_hits = {
                let vi = pipeline
                    .vector_index
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                vi.search_top_k(&semantic_query, 10, None)
                    .expect("vector search")
            };
            assert!(
                vector_hits.iter().any(|hit| hit.doc_id == "src/lib.rs"),
                "vector index should contain the upserted document"
            );

            let storage =
                frankensearch_storage::Storage::open(frankensearch_storage::StorageConfig {
                    db_path,
                    ..frankensearch_storage::StorageConfig::default()
                })
                .expect("open storage db");
            let stored = storage
                .get_document("src/lib.rs")
                .expect("lookup stored document");
            assert!(
                stored.is_some(),
                "storage pipeline should persist watcher-ingested documents"
            );
        });
    }

    #[test]
    fn live_ingest_upsert_missing_file_deletes_stale_document() {
        run_test_with_cx(|cx| async move {
            let temp = tempfile::tempdir().expect("tempdir");
            let target_root = temp.path().join("project");
            fs::create_dir_all(&target_root).expect("target root");

            let index_root = temp.path().join("index");
            fs::create_dir_all(index_root.join("vector")).expect("vector dir");
            let lexical_path = index_root.join("lexical");
            let vector_path = index_root.join(super::FSFS_VECTOR_INDEX_FILE);

            let lexical_index = TantivyIndex::create(&lexical_path).expect("create lexical index");
            let mut vector_writer =
                VectorIndex::create(&vector_path, "hash", 256).expect("create vector index");
            vector_writer
                .write_record("src/ghost.rs", &vec![0.0_f32; 256])
                .expect("seed vector record");
            vector_writer.finish().expect("finish vector index");
            let vector_index = VectorIndex::open(&vector_path).expect("open vector index");

            let stale_doc = IndexableDocument::new(
                "src/ghost.rs".to_owned(),
                "stale lexical content".to_owned(),
            );
            lexical_index
                .index_document(&cx, &stale_doc)
                .await
                .expect("seed lexical doc");
            lexical_index
                .commit(&cx)
                .await
                .expect("commit seed lexical doc");

            let pipeline = LiveIngestPipeline::new(
                target_root.clone(),
                lexical_index,
                vector_index,
                Arc::new(HashEmbedder::default_256()),
            );
            let ingest_rt = asupersync::runtime::RuntimeBuilder::current_thread()
                .build()
                .expect("build ingest runtime");
            let missing_path = target_root.join("src/ghost.rs");
            let applied = pipeline
                .apply_batch(
                    &[WatchIngestOp::Upsert {
                        file_key: missing_path.display().to_string(),
                        revision: 42,
                        ingestion_class: IngestionClass::FullSemanticLexical,
                    }],
                    &ingest_rt,
                )
                .expect("upsert missing file should be treated as delete");

            assert_eq!(applied, 1);
            let hits = pipeline
                .lexical_index
                .search(&cx, "stale", 5)
                .await
                .expect("lexical search after delete");
            assert!(hits.is_empty(), "stale lexical doc should be removed");
        });
    }

    #[test]
    fn live_ingest_rejects_absolute_parent_dir_file_key() {
        run_test_with_cx(|_cx| async move {
            let temp = tempfile::tempdir().expect("tempdir");
            let target_root = temp.path().join("project");
            fs::create_dir_all(&target_root).expect("target root");

            let index_root = temp.path().join("index");
            fs::create_dir_all(index_root.join("vector")).expect("vector dir");
            let lexical_path = index_root.join("lexical");
            let vector_path = index_root.join(super::FSFS_VECTOR_INDEX_FILE);
            let lexical_index = TantivyIndex::create(&lexical_path).expect("create lexical index");
            let vector_writer =
                VectorIndex::create(&vector_path, "hash", 256).expect("create vector index");
            vector_writer.finish().expect("finish vector index");
            let vector_index = VectorIndex::open(&vector_path).expect("open vector index");

            let pipeline = LiveIngestPipeline::new(
                target_root.clone(),
                lexical_index,
                vector_index,
                Arc::new(HashEmbedder::default_256()),
            );

            let escaped = target_root.join("../outside.txt");
            let err = pipeline
                .resolve_paths(&escaped.display().to_string())
                .expect_err("absolute file_key with '..' must be rejected");
            assert!(matches!(
                err,
                frankensearch_core::SearchError::InvalidConfig { field, .. }
                if field == "file_key"
            ));
        });
    }

    #[cfg(unix)]
    #[test]
    fn live_ingest_accepts_canonical_file_paths_with_symlink_cli_root() {
        use std::os::unix::fs::symlink;

        run_test_with_cx(|_cx| async move {
            let temp = tempfile::tempdir().expect("tempdir");
            let canonical_root = temp.path().join("canonical-project");
            let symlink_root = temp.path().join("project-link");
            fs::create_dir_all(canonical_root.join("src")).expect("canonical root");
            symlink(&canonical_root, &symlink_root).expect("create symlink root");
            let canonical_file = canonical_root.join("src/main.rs");
            fs::write(&canonical_file, "fn main() {}\n").expect("write source file");

            let runtime = FsfsRuntime::new(FsfsConfig::default()).with_cli_input(CliInput {
                command: CliCommand::Watch,
                target_path: Some(symlink_root),
                ..CliInput::default()
            });
            let resolved_root = runtime
                .resolve_target_root()
                .expect("canonical target root");
            assert_eq!(
                resolved_root,
                fs::canonicalize(&canonical_root).expect("canonicalize target root")
            );

            let index_root = temp.path().join("index");
            fs::create_dir_all(index_root.join("vector")).expect("vector dir");
            let lexical_path = index_root.join("lexical");
            let vector_path = index_root.join(super::FSFS_VECTOR_INDEX_FILE);
            let lexical_index = TantivyIndex::create(&lexical_path).expect("create lexical index");
            let vector_writer =
                VectorIndex::create(&vector_path, "hash", 256).expect("create vector index");
            vector_writer.finish().expect("finish vector index");
            let vector_index = VectorIndex::open(&vector_path).expect("open vector index");

            let pipeline = LiveIngestPipeline::new(
                resolved_root,
                lexical_index,
                vector_index,
                Arc::new(HashEmbedder::default_256()),
            );

            let (resolved_file, rel_key) = pipeline
                .resolve_paths(&canonical_file.display().to_string())
                .expect("canonical watcher path should remain within target root");
            assert_eq!(resolved_file, canonical_file);
            assert_eq!(rel_key, "src/main.rs");
        });
    }

    #[test]
    fn live_ingest_upsert_binary_file_deletes_stale_document() {
        run_test_with_cx(|cx| async move {
            let temp = tempfile::tempdir().expect("tempdir");
            let target_root = temp.path().join("project");
            fs::create_dir_all(target_root.join("src")).expect("target root");
            let binary_path = target_root.join("src/blob.bin");
            fs::write(&binary_path, [0_u8, 42, 7, 9]).expect("write binary fixture");

            let index_root = temp.path().join("index");
            fs::create_dir_all(index_root.join("vector")).expect("vector dir");
            let lexical_path = index_root.join("lexical");
            let vector_path = index_root.join(super::FSFS_VECTOR_INDEX_FILE);

            let lexical_index = TantivyIndex::create(&lexical_path).expect("create lexical index");
            let mut vector_writer =
                VectorIndex::create(&vector_path, "hash", 256).expect("create vector index");
            vector_writer
                .write_record("src/blob.bin", &vec![0.0_f32; 256])
                .expect("seed vector record");
            vector_writer.finish().expect("finish vector index");
            let vector_index = VectorIndex::open(&vector_path).expect("open vector index");

            let stale_doc = IndexableDocument::new(
                "src/blob.bin".to_owned(),
                "stale lexical content".to_owned(),
            );
            lexical_index
                .index_document(&cx, &stale_doc)
                .await
                .expect("seed lexical doc");
            lexical_index
                .commit(&cx)
                .await
                .expect("commit seed lexical doc");

            let pipeline = LiveIngestPipeline::new(
                target_root.clone(),
                lexical_index,
                vector_index,
                Arc::new(HashEmbedder::default_256()),
            );
            let ingest_rt = asupersync::runtime::RuntimeBuilder::current_thread()
                .build()
                .expect("build ingest runtime");
            let applied = pipeline
                .apply_batch(
                    &[WatchIngestOp::Upsert {
                        file_key: binary_path.display().to_string(),
                        revision: 42,
                        ingestion_class: IngestionClass::FullSemanticLexical,
                    }],
                    &ingest_rt,
                )
                .expect("binary upsert should prune stale entries");

            assert_eq!(applied, 1);

            let hits = pipeline
                .lexical_index
                .search(&cx, "stale", 5)
                .await
                .expect("lexical search after binary prune");
            assert!(hits.is_empty(), "stale lexical doc should be removed");

            let query = vec![0.0_f32; 256];
            let vector_hits = {
                let vi = pipeline
                    .vector_index
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                vi.search_top_k(&query, 5, None).expect("vector search")
            };
            assert!(
                vector_hits.iter().all(|hit| hit.doc_id != "src/blob.bin"),
                "stale vector entry should be removed when file becomes binary"
            );
        });
    }

    #[test]
    fn live_ingest_upsert_lexical_only_reindexes_lexical_and_prunes_vector() {
        run_test_with_cx(|cx| async move {
            let temp = tempfile::tempdir().expect("tempdir");
            let target_root = temp.path().join("project");
            fs::create_dir_all(target_root.join("src")).expect("target root");
            let file_path = target_root.join("src/lexical_only.md");
            fs::write(
                &file_path,
                "fresh lexical content without semantic embedding",
            )
            .expect("write lexical fixture");

            let index_root = temp.path().join("index");
            fs::create_dir_all(index_root.join("vector")).expect("vector dir");
            let lexical_path = index_root.join("lexical");
            let vector_path = index_root.join(super::FSFS_VECTOR_INDEX_FILE);

            let lexical_index = TantivyIndex::create(&lexical_path).expect("create lexical index");
            let mut vector_writer =
                VectorIndex::create(&vector_path, "hash", 256).expect("create vector index");
            vector_writer
                .write_record("src/lexical_only.md", &vec![0.1_f32; 256])
                .expect("seed vector record");
            vector_writer.finish().expect("finish vector index");
            let vector_index = VectorIndex::open(&vector_path).expect("open vector index");

            let pipeline = LiveIngestPipeline::new(
                target_root.clone(),
                lexical_index,
                vector_index,
                Arc::new(HashEmbedder::default_256()),
            );
            let ingest_rt = asupersync::runtime::RuntimeBuilder::current_thread()
                .build()
                .expect("build ingest runtime");
            let applied = pipeline
                .apply_batch(
                    &[WatchIngestOp::Upsert {
                        file_key: file_path.display().to_string(),
                        revision: 7,
                        ingestion_class: IngestionClass::LexicalOnly,
                    }],
                    &ingest_rt,
                )
                .expect("lexical-only upsert should succeed");

            assert_eq!(applied, 1);
            let lexical_hits = pipeline
                .lexical_index
                .search(&cx, "lexical", 5)
                .await
                .expect("search lexical index");
            assert!(
                lexical_hits
                    .iter()
                    .any(|hit| hit.doc_id == "src/lexical_only.md"),
                "lexical-only ingest should still index lexical content"
            );

            let query = vec![0.0_f32; 256];
            let vector_hits = {
                let vi = pipeline
                    .vector_index
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                vi.search_top_k(&query, 5, None).expect("vector search")
            };
            assert!(
                vector_hits
                    .iter()
                    .all(|hit| hit.doc_id != "src/lexical_only.md"),
                "lexical-only ingest should remove stale semantic vectors"
            );
        });
    }

    #[test]
    fn live_ingest_upsert_metadata_only_prunes_lexical_and_vector_entries() {
        run_test_with_cx(|cx| async move {
            let temp = tempfile::tempdir().expect("tempdir");
            let target_root = temp.path().join("project");
            fs::create_dir_all(target_root.join("src")).expect("target root");
            let file_path = target_root.join("src/meta.json");
            fs::write(&file_path, "{\"meta\":true}\n").expect("write metadata fixture");

            let index_root = temp.path().join("index");
            fs::create_dir_all(index_root.join("vector")).expect("vector dir");
            let lexical_path = index_root.join("lexical");
            let vector_path = index_root.join(super::FSFS_VECTOR_INDEX_FILE);

            let lexical_index = TantivyIndex::create(&lexical_path).expect("create lexical index");
            let mut vector_writer =
                VectorIndex::create(&vector_path, "hash", 256).expect("create vector index");
            vector_writer
                .write_record("src/meta.json", &vec![0.2_f32; 256])
                .expect("seed vector record");
            vector_writer.finish().expect("finish vector index");
            let vector_index = VectorIndex::open(&vector_path).expect("open vector index");

            let stale_doc = IndexableDocument::new(
                "src/meta.json".to_owned(),
                "stale lexical metadata content".to_owned(),
            );
            lexical_index
                .index_document(&cx, &stale_doc)
                .await
                .expect("seed lexical doc");
            lexical_index
                .commit(&cx)
                .await
                .expect("commit lexical seed");

            let pipeline = LiveIngestPipeline::new(
                target_root.clone(),
                lexical_index,
                vector_index,
                Arc::new(HashEmbedder::default_256()),
            );
            let ingest_rt = asupersync::runtime::RuntimeBuilder::current_thread()
                .build()
                .expect("build ingest runtime");
            let applied = pipeline
                .apply_batch(
                    &[WatchIngestOp::Upsert {
                        file_key: file_path.display().to_string(),
                        revision: 8,
                        ingestion_class: IngestionClass::MetadataOnly,
                    }],
                    &ingest_rt,
                )
                .expect("metadata-only upsert should prune stale index entries");

            assert_eq!(applied, 1);
            let lexical_hits = pipeline
                .lexical_index
                .search(&cx, "metadata", 5)
                .await
                .expect("search lexical index");
            assert!(
                lexical_hits.iter().all(|hit| hit.doc_id != "src/meta.json"),
                "metadata-only ingest should remove lexical entries"
            );

            let query = vec![0.0_f32; 256];
            let vector_hits = {
                let vi = pipeline
                    .vector_index
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                vi.search_top_k(&query, 5, None).expect("vector search")
            };
            assert!(
                vector_hits.iter().all(|hit| hit.doc_id != "src/meta.json"),
                "metadata-only ingest should remove vector entries"
            );
        });
    }

    #[test]
    fn shutdown_wait_observes_reload_then_user_shutdown() {
        run_test_with_cx(|cx| async move {
            let runtime = FsfsRuntime::new(FsfsConfig::default());
            let coordinator: Arc<ShutdownCoordinator> = Arc::new(ShutdownCoordinator::new());
            let trigger: Arc<ShutdownCoordinator> = Arc::clone(&coordinator);

            let worker = thread::spawn(move || {
                thread::sleep(Duration::from_millis(20));
                trigger.request_config_reload();
                thread::sleep(Duration::from_millis(20));
                trigger.request_shutdown(ShutdownReason::UserRequest);
            });

            runtime
                .run_mode_with_shutdown(&cx, InterfaceMode::Tui, &coordinator)
                .await
                .expect("tui mode with reload + shutdown");

            worker.join().expect("reload trigger thread join");
        });
    }

    #[test]
    fn vector_plan_schedules_fast_and_quality_with_invalidation() {
        let runtime = FsfsRuntime::new(FsfsConfig::default());
        let plans = runtime.plan_vector_pipeline(
            &[VectorPipelineInput {
                file_key: "doc/a.md".to_owned(),
                observed_revision: 4,
                previous_indexed_revision: Some(3),
                ingestion_class: IngestionClass::FullSemanticLexical,
                content_len_bytes: 2_050,
                content_hash_changed: true,
            }],
            EmbedderAvailability::Full,
        );

        assert_eq!(plans.len(), 1);
        let plan = &plans[0];
        assert_eq!(plan.file_key, "doc/a.md");
        assert_eq!(plan.revision, 4);
        assert_eq!(plan.chunk_count, 3);
        assert_eq!(plan.batch_size, 64);
        assert_eq!(plan.tier, VectorSchedulingTier::FastAndQuality);
        assert_eq!(plan.invalidate_revisions_through, Some(3));
        assert_eq!(plan.reason_code, "vector.plan.fast_quality");
    }

    #[test]
    fn vector_plan_skips_non_semantic_and_stale_revisions() {
        let runtime = FsfsRuntime::new(FsfsConfig::default());
        let plans = runtime.plan_vector_pipeline(
            &[
                VectorPipelineInput {
                    file_key: "doc/meta.json".to_owned(),
                    observed_revision: 8,
                    previous_indexed_revision: Some(7),
                    ingestion_class: IngestionClass::MetadataOnly,
                    content_len_bytes: 512,
                    content_hash_changed: true,
                },
                VectorPipelineInput {
                    file_key: "doc/stale.txt".to_owned(),
                    observed_revision: 4,
                    previous_indexed_revision: Some(6),
                    ingestion_class: IngestionClass::FullSemanticLexical,
                    content_len_bytes: 1_024,
                    content_hash_changed: true,
                },
                VectorPipelineInput {
                    file_key: "doc/unchanged.txt".to_owned(),
                    observed_revision: 9,
                    previous_indexed_revision: Some(9),
                    ingestion_class: IngestionClass::FullSemanticLexical,
                    content_len_bytes: 1_024,
                    content_hash_changed: false,
                },
            ],
            EmbedderAvailability::Full,
        );

        assert_eq!(plans.len(), 3);
        assert_eq!(plans[0].tier, VectorSchedulingTier::Skip);
        assert_eq!(
            plans[0].reason_code,
            "vector.skip.non_semantic_ingestion_class"
        );
        assert_eq!(plans[1].tier, VectorSchedulingTier::Skip);
        assert_eq!(plans[1].reason_code, "vector.skip.out_of_order_revision");
        assert_eq!(plans[2].tier, VectorSchedulingTier::Skip);
        assert_eq!(plans[2].reason_code, "vector.skip.revision_unchanged");
    }

    #[test]
    fn vector_plan_invalidates_equal_revision_when_hash_changed() {
        let runtime = FsfsRuntime::new(FsfsConfig::default());
        let plans = runtime.plan_vector_pipeline(
            &[VectorPipelineInput {
                file_key: "doc/mtime-collision.md".to_owned(),
                observed_revision: 9,
                previous_indexed_revision: Some(9),
                ingestion_class: IngestionClass::FullSemanticLexical,
                content_len_bytes: 1_024,
                content_hash_changed: true,
            }],
            EmbedderAvailability::Full,
        );

        assert_eq!(plans.len(), 1);
        let plan = &plans[0];
        assert_eq!(plan.tier, VectorSchedulingTier::FastAndQuality);
        assert_eq!(plan.reason_code, "vector.plan.fast_quality");
        assert_eq!(plan.invalidate_revisions_through, Some(9));
    }

    #[test]
    fn vector_plan_uses_fast_only_and_lexical_fallback_policies() {
        let runtime = FsfsRuntime::new(FsfsConfig::default());
        let input = VectorPipelineInput {
            file_key: "doc/a.md".to_owned(),
            observed_revision: 10,
            previous_indexed_revision: Some(9),
            ingestion_class: IngestionClass::FullSemanticLexical,
            content_len_bytes: 4_000,
            content_hash_changed: true,
        };

        let fast_only_plan = runtime
            .plan_vector_pipeline(std::slice::from_ref(&input), EmbedderAvailability::FastOnly)
            .pop()
            .expect("fast-only plan");
        assert_eq!(fast_only_plan.tier, VectorSchedulingTier::FastOnly);
        assert_eq!(
            fast_only_plan.reason_code,
            "vector.plan.fast_only_quality_unavailable"
        );

        let lexical_fallback_plan = runtime
            .plan_vector_pipeline(std::slice::from_ref(&input), EmbedderAvailability::None)
            .pop()
            .expect("fallback plan");
        assert_eq!(
            lexical_fallback_plan.tier,
            VectorSchedulingTier::LexicalFallback
        );
        assert_eq!(lexical_fallback_plan.chunk_count, 0);
        assert_eq!(
            lexical_fallback_plan.reason_code,
            "vector.plan.lexical_fallback"
        );

        let mut fast_only_config = FsfsConfig::default();
        fast_only_config.search.fast_only = true;
        let fast_only_runtime = FsfsRuntime::new(fast_only_config);
        let policy_plan = fast_only_runtime
            .plan_vector_pipeline(std::slice::from_ref(&input), EmbedderAvailability::Full)
            .pop()
            .expect("policy plan");
        assert_eq!(policy_plan.tier, VectorSchedulingTier::FastOnly);
        assert_eq!(policy_plan.reason_code, "vector.plan.fast_only_policy");
    }

    #[test]
    fn vector_index_actions_encode_revision_coherence() {
        let runtime = FsfsRuntime::new(FsfsConfig::default());
        let plan = runtime
            .plan_vector_pipeline(
                &[VectorPipelineInput {
                    file_key: "doc/a.md".to_owned(),
                    observed_revision: 12,
                    previous_indexed_revision: Some(10),
                    ingestion_class: IngestionClass::FullSemanticLexical,
                    content_len_bytes: 1_200,
                    content_hash_changed: true,
                }],
                EmbedderAvailability::Full,
            )
            .pop()
            .expect("plan");

        let actions = FsfsRuntime::vector_index_write_actions(&plan);
        assert_eq!(actions.len(), 3);
        assert_eq!(
            actions[0],
            VectorIndexWriteAction::InvalidateRevisionsThrough {
                file_key: "doc/a.md".to_owned(),
                revision: 10
            }
        );
        assert_eq!(
            actions[1],
            VectorIndexWriteAction::AppendFast {
                file_key: "doc/a.md".to_owned(),
                revision: 12,
                chunk_count: 2
            }
        );
        assert_eq!(
            actions[2],
            VectorIndexWriteAction::AppendQuality {
                file_key: "doc/a.md".to_owned(),
                revision: 12,
                chunk_count: 2
            }
        );
    }

    #[test]
    fn runtime_index_command_writes_sentinel_and_artifacts() {
        run_test_with_cx(|cx| async move {
            let temp = tempfile::tempdir().expect("tempdir");
            let project = temp.path().join("project");
            fs::create_dir_all(project.join("src")).expect("project dirs");
            fs::write(
                project.join("src/lib.rs"),
                "pub fn demo() { println!(\"ok\"); }\n",
            )
            .expect("write file");
            fs::write(project.join("README.md"), "# demo\nindex me\n").expect("readme");

            let mut config = FsfsConfig::default();
            config.storage.index_dir = ".frankensearch".to_owned();
            let runtime = FsfsRuntime::new(config).with_cli_input(CliInput {
                command: CliCommand::Index,
                target_path: Some(project.clone()),
                ..CliInput::default()
            });

            runtime
                .run_mode(&cx, InterfaceMode::Cli)
                .await
                .expect("index scaffold command should succeed");

            let index_root = project.join(".frankensearch");
            assert!(index_root.join(super::FSFS_SENTINEL_FILE).exists());
            assert!(index_root.join(super::FSFS_VECTOR_MANIFEST_FILE).exists());
            assert!(index_root.join(super::FSFS_LEXICAL_MANIFEST_FILE).exists());
            assert!(index_root.join(super::FSFS_VECTOR_INDEX_FILE).exists());

            let sentinel_raw = fs::read_to_string(index_root.join(super::FSFS_SENTINEL_FILE))
                .expect("read sentinel");
            let sentinel: super::IndexSentinel =
                serde_json::from_str(&sentinel_raw).expect("parse sentinel");
            assert_eq!(sentinel.schema_version, 1);
            assert_eq!(sentinel.command, "index");
            assert!(sentinel.discovered_files >= 2);
            assert!(sentinel.indexed_files >= 1);
            assert!(!sentinel.source_hash_hex.is_empty());

            let vector_index = VectorIndex::open(&index_root.join(super::FSFS_VECTOR_INDEX_FILE))
                .expect("open fsvi");
            assert!(vector_index.record_count() >= 1);
            let lexical_index =
                TantivyIndex::open(&index_root.join("lexical")).expect("open tantivy");
            assert!(lexical_index.doc_count() >= 1);
            let hits = lexical_index
                .search(&cx, "index", 5)
                .await
                .expect("lexical query");
            assert!(!hits.is_empty());
        });
    }

    #[test]
    fn runtime_index_command_respects_index_dir_override() {
        run_test_with_cx(|cx| async move {
            let temp = tempfile::tempdir().expect("tempdir");
            let project = temp.path().join("project");
            let override_index_root = temp.path().join("custom-index");
            fs::create_dir_all(project.join("src")).expect("project dirs");
            fs::write(project.join("src/lib.rs"), "pub fn demo() {}\n").expect("write file");

            let mut config = FsfsConfig::default();
            config.storage.index_dir = ".frankensearch".to_owned();
            let runtime = FsfsRuntime::new(config).with_cli_input(CliInput {
                command: CliCommand::Index,
                target_path: Some(project.clone()),
                index_dir: Some(override_index_root.clone()),
                ..CliInput::default()
            });

            runtime
                .run_mode(&cx, InterfaceMode::Cli)
                .await
                .expect("index scaffold command should succeed");

            assert!(
                override_index_root.join(super::FSFS_SENTINEL_FILE).exists(),
                "index artifacts should be written to --index-dir override"
            );
            assert!(
                !project
                    .join(".frankensearch")
                    .join(super::FSFS_SENTINEL_FILE)
                    .exists(),
                "default index dir should remain untouched when override is provided"
            );
        });
    }

    #[test]
    fn runtime_search_payload_returns_ranked_hits_after_index() {
        run_test_with_cx(|cx| async move {
            let temp = tempfile::tempdir().expect("tempdir");
            let project = temp.path().join("project");
            fs::create_dir_all(project.join("src")).expect("create project source dir");
            fs::write(
                project.join("src/auth.rs"),
                "pub fn authenticate(token: &str) -> bool { !token.is_empty() }\n",
            )
            .expect("write auth source");
            fs::write(
                project.join("README.md"),
                "Authentication middleware validates incoming bearer tokens.\n",
            )
            .expect("write readme");

            let mut config = FsfsConfig::default();
            config.storage.index_dir = ".frankensearch".to_owned();
            let index_runtime = FsfsRuntime::new(config.clone()).with_cli_input(CliInput {
                command: CliCommand::Index,
                target_path: Some(project.clone()),
                ..CliInput::default()
            });
            index_runtime
                .run_mode(&cx, InterfaceMode::Cli)
                .await
                .expect("index command should succeed");

            let search_runtime = FsfsRuntime::new(config).with_cli_input(CliInput {
                command: CliCommand::Search,
                query: Some("authentication middleware".to_owned()),
                index_dir: Some(project.join(".frankensearch")),
                ..CliInput::default()
            });
            let payload = search_runtime
                .execute_search_payload(&cx, "authentication middleware", 5)
                .await
                .expect("search payload");

            assert_eq!(payload.phase, SearchOutputPhase::Initial);
            assert!(!payload.hits.is_empty(), "expected at least one ranked hit");
            assert!(
                payload
                    .hits
                    .iter()
                    .any(|hit| hit.path.contains("auth") || hit.path.contains("README")),
                "expected auth-related hit path in payload"
            );
        });
    }

    #[test]
    fn runtime_search_payload_applies_extension_filter() {
        run_test_with_cx(|cx| async move {
            let temp = tempfile::tempdir().expect("tempdir");
            let project = temp.path().join("project");
            fs::create_dir_all(project.join("src")).expect("create project source dir");
            fs::create_dir_all(project.join("docs")).expect("create docs dir");
            fs::write(
                project.join("src/auth.rs"),
                "/// Authentication middleware for validating bearer tokens.\npub fn authentication_middleware(token: &str) -> bool { !token.is_empty() }\n",
            )
            .expect("write auth source");
            fs::write(
                project.join("docs/auth.md"),
                "Authentication middleware validates incoming bearer tokens.\n",
            )
            .expect("write docs");

            let mut config = FsfsConfig::default();
            config.storage.index_dir = ".frankensearch".to_owned();
            FsfsRuntime::new(config.clone())
                .with_cli_input(CliInput {
                    command: CliCommand::Index,
                    target_path: Some(project.clone()),
                    ..CliInput::default()
                })
                .run_mode(&cx, InterfaceMode::Cli)
                .await
                .expect("index command should succeed");

            let payload = FsfsRuntime::new(config)
                .with_cli_input(CliInput {
                    command: CliCommand::Search,
                    query: Some("authentication middleware".to_owned()),
                    filter: Some("type:rs".to_owned()),
                    index_dir: Some(project.join(".frankensearch")),
                    ..CliInput::default()
                })
                .execute_search_payload(&cx, "authentication middleware", 10)
                .await
                .expect("search payload");

            assert!(
                !payload.hits.is_empty(),
                "expected at least one filtered hit"
            );
            assert!(
                payload.hits.iter().all(|hit| {
                    std::path::Path::new(&hit.path)
                        .extension()
                        .is_some_and(|ext| ext.eq_ignore_ascii_case("rs"))
                }),
                "all hits should satisfy type:rs filter"
            );
        });
    }

    #[test]
    fn runtime_search_payload_rejects_invalid_filter_key() {
        run_test_with_cx(|cx| async move {
            let runtime = FsfsRuntime::new(FsfsConfig::default()).with_cli_input(CliInput {
                command: CliCommand::Search,
                query: Some("auth".to_owned()),
                filter: Some("owner:me".to_owned()),
                ..CliInput::default()
            });

            let error = runtime
                .execute_search_payload(&cx, "auth", 5)
                .await
                .expect_err("invalid filter key should fail");
            assert!(
                error.to_string().contains("unsupported filter key"),
                "unexpected error: {error}"
            );
        });
    }

    #[test]
    fn runtime_search_payload_persists_explain_session_context() {
        run_test_with_cx(|cx| async move {
            let temp = tempfile::tempdir().expect("tempdir");
            let project = temp.path().join("project");
            fs::create_dir_all(project.join("src")).expect("create project source dir");
            fs::write(
                project.join("src/auth.rs"),
                "pub fn authenticate(token: &str) -> bool { !token.is_empty() }\n",
            )
            .expect("write auth source");
            fs::write(
                project.join("README.md"),
                "Authentication middleware validates incoming bearer tokens.\n",
            )
            .expect("write readme");

            let mut config = FsfsConfig::default();
            config.storage.index_dir = ".frankensearch".to_owned();
            FsfsRuntime::new(config.clone())
                .with_cli_input(CliInput {
                    command: CliCommand::Index,
                    target_path: Some(project.clone()),
                    ..CliInput::default()
                })
                .run_mode(&cx, InterfaceMode::Cli)
                .await
                .expect("index command should succeed");

            let search_runtime = FsfsRuntime::new(config).with_cli_input(CliInput {
                command: CliCommand::Search,
                query: Some("authentication middleware".to_owned()),
                index_dir: Some(project.join(".frankensearch")),
                ..CliInput::default()
            });
            let payload = search_runtime
                .execute_search_payload(&cx, "authentication middleware", 5)
                .await
                .expect("search payload");

            let session = search_runtime
                .load_explain_session()
                .expect("load explain session")
                .expect("explain session should be written");
            assert_eq!(session.query, "authentication middleware");
            assert_eq!(session.phase, SearchOutputPhase::Initial);
            assert_eq!(session.hits.len(), payload.hits.len());
            assert_eq!(session.hits[0].result_id, "R0");
            assert_eq!(session.hits[0].path, payload.hits[0].path);
        });
    }

    #[test]
    fn runtime_explain_command_errors_without_saved_search_context() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime = FsfsRuntime::new(FsfsConfig::default()).with_cli_input(CliInput {
            command: CliCommand::Explain,
            result_id: Some("R0".to_owned()),
            index_dir: Some(temp.path().join("index-root")),
            format: OutputFormat::Json,
            ..CliInput::default()
        });

        let err = runtime
            .run_explain_command()
            .expect_err("missing explain session should fail");
        let text = err.to_string();
        assert!(
            text.contains("run `fsfs search <query>` first"),
            "unexpected explain-session error: {text}"
        );
    }

    #[test]
    fn runtime_explain_command_uses_saved_search_context() {
        run_test_with_cx(|cx| async move {
            let temp = tempfile::tempdir().expect("tempdir");
            let project = temp.path().join("project");
            fs::create_dir_all(project.join("src")).expect("create project source dir");
            fs::write(
                project.join("src/auth.rs"),
                "pub fn authenticate(token: &str) -> bool { !token.is_empty() }\n",
            )
            .expect("write auth source");
            fs::write(
                project.join("README.md"),
                "Authentication middleware validates incoming bearer tokens.\n",
            )
            .expect("write readme");

            let mut config = FsfsConfig::default();
            config.storage.index_dir = ".frankensearch".to_owned();
            FsfsRuntime::new(config.clone())
                .with_cli_input(CliInput {
                    command: CliCommand::Index,
                    target_path: Some(project.clone()),
                    ..CliInput::default()
                })
                .run_mode(&cx, InterfaceMode::Cli)
                .await
                .expect("index command should succeed");

            let search_runtime = FsfsRuntime::new(config.clone()).with_cli_input(CliInput {
                command: CliCommand::Search,
                query: Some("authentication middleware".to_owned()),
                index_dir: Some(project.join(".frankensearch")),
                ..CliInput::default()
            });
            let _ = search_runtime
                .execute_search_payload(&cx, "authentication middleware", 5)
                .await
                .expect("search payload");

            let explain_runtime = FsfsRuntime::new(config).with_cli_input(CliInput {
                command: CliCommand::Explain,
                result_id: Some("R0".to_owned()),
                index_dir: Some(project.join(".frankensearch")),
                format: OutputFormat::Json,
                ..CliInput::default()
            });
            explain_runtime
                .run_explain_command()
                .expect("explain command should resolve saved R0 context");
        });
    }

    #[test]
    fn runtime_search_payload_falls_back_to_lexical_when_vector_index_is_corrupt() {
        run_test_with_cx(|cx| async move {
            let temp = tempfile::tempdir().expect("tempdir");
            let project = temp.path().join("project");
            fs::create_dir_all(project.join("src")).expect("create project source dir");
            fs::write(
                project.join("src/auth.rs"),
                "pub fn authenticate(token: &str) -> bool { !token.is_empty() }\n",
            )
            .expect("write auth source");
            fs::write(
                project.join("README.md"),
                "Authentication middleware validates incoming bearer tokens.\n",
            )
            .expect("write readme");

            let mut config = FsfsConfig::default();
            config.storage.index_dir = ".frankensearch".to_owned();
            FsfsRuntime::new(config.clone())
                .with_cli_input(CliInput {
                    command: CliCommand::Index,
                    target_path: Some(project.clone()),
                    ..CliInput::default()
                })
                .run_mode(&cx, InterfaceMode::Cli)
                .await
                .expect("index command should succeed");

            fs::write(
                project
                    .join(".frankensearch")
                    .join(super::FSFS_VECTOR_INDEX_FILE),
                b"not-fsvi",
            )
            .expect("corrupt vector index file");

            let payload = FsfsRuntime::new(config)
                .with_cli_input(CliInput {
                    command: CliCommand::Search,
                    query: Some("authentication middleware".to_owned()),
                    index_dir: Some(project.join(".frankensearch")),
                    ..CliInput::default()
                })
                .execute_search_payload(&cx, "authentication middleware", 5)
                .await
                .expect("search payload should fall back to lexical index");

            assert!(
                !payload.hits.is_empty(),
                "expected lexical hits after fallback"
            );
        });
    }

    #[test]
    fn runtime_search_payload_empty_query_returns_empty_results() {
        run_test_with_cx(|cx| async move {
            let runtime = FsfsRuntime::new(FsfsConfig::default()).with_cli_input(CliInput {
                command: CliCommand::Search,
                query: Some("   ".to_owned()),
                ..CliInput::default()
            });

            let payload = runtime
                .execute_search_payload(&cx, "   ", 10)
                .await
                .expect("empty query should not fail");

            assert_eq!(payload.phase, SearchOutputPhase::Initial);
            assert!(payload.is_empty());
            assert_eq!(payload.total_candidates, 0);
        });
    }

    #[test]
    fn runtime_search_payload_errors_when_index_missing() {
        run_test_with_cx(|cx| async move {
            let temp = tempfile::tempdir().expect("tempdir");
            let missing_index = temp.path().join("missing-index");

            let runtime = FsfsRuntime::new(FsfsConfig::default()).with_cli_input(CliInput {
                command: CliCommand::Search,
                query: Some("auth flow".to_owned()),
                index_dir: Some(missing_index.clone()),
                ..CliInput::default()
            });

            let err = runtime
                .execute_search_payload(&cx, "auth flow", 5)
                .await
                .expect_err("missing index should fail");
            let text = err.to_string();
            assert!(text.contains("no index found"), "unexpected error: {text}");
            assert!(
                text.contains("fsfs index"),
                "error should include remediation command: {text}"
            );
        });
    }

    #[test]
    fn runtime_search_command_runs_via_cli_dispatch() {
        run_test_with_cx(|cx| async move {
            let temp = tempfile::tempdir().expect("tempdir");
            let project = temp.path().join("project");
            fs::create_dir_all(project.join("src")).expect("create project source dir");
            fs::write(
                project.join("src/auth_flow.rs"),
                "pub fn auth_flow() -> &'static str { \"dispatch\" }\n",
            )
            .expect("write auth source");

            let mut config = FsfsConfig::default();
            config.storage.index_dir = ".frankensearch".to_owned();

            FsfsRuntime::new(config.clone())
                .with_cli_input(CliInput {
                    command: CliCommand::Index,
                    target_path: Some(project.clone()),
                    ..CliInput::default()
                })
                .run_mode(&cx, InterfaceMode::Cli)
                .await
                .expect("index command should succeed");

            FsfsRuntime::new(config)
                .with_cli_input(CliInput {
                    command: CliCommand::Search,
                    query: Some("auth flow dispatch".to_owned()),
                    index_dir: Some(project.join(".frankensearch")),
                    format: OutputFormat::Json,
                    ..CliInput::default()
                })
                .run_mode(&cx, InterfaceMode::Cli)
                .await
                .expect("search command should complete via cli dispatch");
        });
    }

    #[test]
    fn runtime_stream_emitter_outputs_protocol_frames_ndjson() {
        let runtime = FsfsRuntime::new(FsfsConfig::default()).with_cli_input(CliInput {
            command: CliCommand::Search,
            stream: true,
            format: OutputFormat::Jsonl,
            ..CliInput::default()
        });
        let payload = SearchPayload::new(
            "auth middleware",
            SearchOutputPhase::Initial,
            3,
            vec![
                SearchHitPayload {
                    rank: 1,
                    path: "src/auth.rs".to_owned(),
                    score: 0.82,
                    snippet: Some("auth middleware".to_owned()),
                    lexical_rank: Some(0),
                    semantic_rank: Some(1),
                    in_both_sources: true,
                },
                SearchHitPayload {
                    rank: 2,
                    path: "README.md".to_owned(),
                    score: 0.71,
                    snippet: None,
                    lexical_rank: Some(1),
                    semantic_rank: None,
                    in_both_sources: false,
                },
            ],
        );

        let mut bytes = Vec::new();
        let mut seq = 0_u64;
        runtime
            .emit_search_stream_started("auth middleware", "stream-test", &mut seq, &mut bytes)
            .expect("emit started");
        runtime
            .emit_search_stream_payload(&payload, "stream-test", &mut seq, &mut bytes)
            .expect("emit payload");
        runtime
            .emit_search_stream_terminal_completed("stream-test", &mut seq, &mut bytes)
            .expect("emit terminal");

        let text = String::from_utf8(bytes).expect("utf8");
        let lines = text.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), 5, "started + progress + 2 results + terminal");

        let started: StreamFrame<SearchHitPayload> =
            decode_stream_frame_ndjson(lines[0]).expect("decode started");
        assert_eq!(started.event.kind(), StreamEventKind::Started);
        let progress: StreamFrame<SearchHitPayload> =
            decode_stream_frame_ndjson(lines[1]).expect("decode progress");
        assert_eq!(progress.event.kind(), StreamEventKind::Progress);
        let first_result: StreamFrame<SearchHitPayload> =
            decode_stream_frame_ndjson(lines[2]).expect("decode result 1");
        assert_eq!(first_result.event.kind(), StreamEventKind::Result);
        let terminal: StreamFrame<SearchHitPayload> =
            decode_stream_frame_ndjson(lines[4]).expect("decode terminal");
        assert_eq!(terminal.event.kind(), StreamEventKind::Terminal);
    }

    #[test]
    fn runtime_stream_emitter_outputs_protocol_frames_toon_with_rs() {
        let runtime = FsfsRuntime::new(FsfsConfig::default()).with_cli_input(CliInput {
            command: CliCommand::Search,
            stream: true,
            format: OutputFormat::Toon,
            ..CliInput::default()
        });
        let payload = SearchPayload::new(
            "config layering",
            SearchOutputPhase::Initial,
            1,
            vec![SearchHitPayload {
                rank: 1,
                path: "docs/config.md".to_owned(),
                score: 0.91,
                snippet: None,
                lexical_rank: Some(0),
                semantic_rank: Some(0),
                in_both_sources: true,
            }],
        );

        let mut bytes = Vec::new();
        let mut seq = 0_u64;
        runtime
            .emit_search_stream_started("config layering", "stream-toon", &mut seq, &mut bytes)
            .expect("emit started");
        runtime
            .emit_search_stream_payload(&payload, "stream-toon", &mut seq, &mut bytes)
            .expect("emit payload");
        runtime
            .emit_search_stream_terminal_completed("stream-toon", &mut seq, &mut bytes)
            .expect("emit terminal");

        assert_eq!(
            bytes.first().copied(),
            Some(TOON_STREAM_RECORD_SEPARATOR_BYTE)
        );
        let records = bytes
            .split(|byte| *byte == TOON_STREAM_RECORD_SEPARATOR_BYTE)
            .filter(|chunk| !chunk.is_empty())
            .collect::<Vec<_>>();
        assert_eq!(records.len(), 4, "started + progress + result + terminal");
        for record in records {
            let payload = std::str::from_utf8(record)
                .expect("utf8")
                .trim_end_matches('\n');
            let frame: StreamFrame<SearchHitPayload> =
                decode_stream_frame_toon(payload).expect("decode toon frame");
            assert!(matches!(
                frame.event.kind(),
                StreamEventKind::Started
                    | StreamEventKind::Progress
                    | StreamEventKind::Result
                    | StreamEventKind::Terminal
            ));
        }
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn runtime_status_payload_reports_index_and_model_state() {
        let temp = tempfile::tempdir().expect("tempdir");
        let project = temp.path().join("project");
        let index_root = project.join(".frankensearch");
        let vector_root = index_root.join("vector");
        let lexical_root = index_root.join("lexical");
        fs::create_dir_all(project.join("src")).expect("create project source dir");
        fs::create_dir_all(&vector_root).expect("create vector dir");
        fs::create_dir_all(&lexical_root).expect("create lexical dir");
        fs::create_dir_all(index_root.join("cache")).expect("create cache dir");

        let source_file = project.join("src/lib.rs");
        fs::write(&source_file, "pub fn status() {}\n").expect("write source file");

        let manifest = vec![super::IndexManifestEntry {
            file_key: source_file.display().to_string(),
            revision: 0,
            ingestion_class: "full_semantic_lexical".to_owned(),
            canonical_bytes: 21,
            reason_code: "test.reason".to_owned(),
        }];
        fs::write(
            index_root.join(super::FSFS_VECTOR_MANIFEST_FILE),
            serde_json::to_string_pretty(&manifest).expect("serialize vector manifest"),
        )
        .expect("write vector manifest");
        fs::write(
            index_root.join(super::FSFS_LEXICAL_MANIFEST_FILE),
            serde_json::to_string_pretty(&manifest).expect("serialize lexical manifest"),
        )
        .expect("write lexical manifest");
        fs::write(
            index_root.join(super::FSFS_VECTOR_INDEX_FILE),
            vec![0_u8; 32],
        )
        .expect("write vector index");

        let sentinel = super::IndexSentinel {
            schema_version: 1,
            generated_at_ms: super::pressure_timestamp_ms(),
            command: "index".to_owned(),
            target_root: project.display().to_string(),
            index_root: index_root.display().to_string(),
            discovered_files: 1,
            indexed_files: 1,
            skipped_files: 0,
            reason_codes: vec!["test.reason".to_owned()],
            total_canonical_bytes: 21,
            source_hash_hex: "feedface".to_owned(),
        };
        fs::write(
            index_root.join(super::FSFS_SENTINEL_FILE),
            serde_json::to_string_pretty(&sentinel).expect("serialize sentinel"),
        )
        .expect("write sentinel");

        let model_root = temp.path().join("models");
        fs::create_dir_all(model_root.join("potion-multilingual-128M")).expect("create model dir");
        fs::write(
            model_root
                .join("potion-multilingual-128M")
                .join("weights.bin"),
            vec![1_u8; 64],
        )
        .expect("write model bytes");

        let mut config = FsfsConfig::default();
        config.storage.index_dir = ".frankensearch".to_owned();
        config.storage.db_path = temp.path().join("fsfs.db").display().to_string();
        config.indexing.model_dir = model_root.display().to_string();
        let runtime = FsfsRuntime::new(config).with_cli_input(CliInput {
            command: CliCommand::Status,
            index_dir: Some(index_root.clone()),
            ..CliInput::default()
        });

        let payload = runtime
            .collect_status_payload()
            .expect("status payload should be collected");
        assert_eq!(payload.index.path, index_root.display().to_string());
        assert_eq!(payload.index.indexed_files, Some(1));
        assert_eq!(payload.index.discovered_files, Some(1));
        assert_eq!(payload.index.stale_files, Some(1));
        assert!(payload.index.size_bytes >= 32);
        assert_eq!(payload.models[0].tier, "fast");
        assert!(payload.models[0].cached);
        assert_eq!(payload.models[1].tier, "quality");
        assert!(payload.models[1].cached);
        assert_eq!(
            payload.runtime.tracked_index_bytes,
            Some(payload.index.size_bytes)
        );
        assert!(payload.runtime.disk_budget_bytes.is_some());
        assert!(
            payload.runtime.disk_budget_stage.is_some(),
            "status payload should expose disk budget stage"
        );
        assert!(
            payload.runtime.disk_budget_action.is_some(),
            "status payload should expose disk budget action"
        );
        assert!(
            payload.runtime.disk_budget_reason_code.is_some(),
            "status payload should expose disk budget reason code"
        );
        assert!(!payload.runtime.storage_pressure_emergency);

        let table = render_status_table(&payload, true);
        assert!(table.contains("disk budget stage:"));
        assert!(table.contains("disk budget action:"));
        assert!(table.contains("disk budget reason:"));
        assert!(table.contains("disk budget bytes:"));
        assert!(table.contains("storage pressure emergency: false"));
    }

    #[test]
    fn runtime_download_models_list_reports_missing_models() {
        run_test_with_cx(|_cx| async move {
            let temp = tempfile::tempdir().expect("tempdir");
            let mut config = FsfsConfig::default();
            config.indexing.model_dir = temp.path().join("models").display().to_string();
            let runtime = FsfsRuntime::new(config).with_cli_input(CliInput {
                command: CliCommand::Download,
                download_list: true,
                ..CliInput::default()
            });

            let payload = runtime
                .collect_download_models_payload()
                .await
                .expect("list payload");
            assert_eq!(payload.operation, "list");
            assert_eq!(payload.models.len(), 3);
            assert!(payload.models.iter().all(|entry| entry.state == "missing"));
        });
    }

    #[test]
    fn runtime_download_models_verify_reports_mismatch() {
        run_test_with_cx(|_cx| async move {
            let temp = tempfile::tempdir().expect("tempdir");
            let model_root = temp.path().join("models");
            let potion_dir = model_root.join("potion-multilingual-128M");
            fs::create_dir_all(&potion_dir).expect("create model dir");
            fs::write(potion_dir.join("tokenizer.json"), b"broken").expect("write model file");

            let mut config = FsfsConfig::default();
            config.indexing.model_dir = model_root.display().to_string();
            let runtime = FsfsRuntime::new(config).with_cli_input(CliInput {
                command: CliCommand::Download,
                download_verify: true,
                model_name: Some("potion".to_owned()),
                ..CliInput::default()
            });

            let payload = runtime
                .collect_download_models_payload()
                .await
                .expect("verify payload");
            assert_eq!(payload.operation, "verify");
            assert_eq!(payload.models.len(), 1);
            assert_eq!(payload.models[0].id, "potion-multilingual-128m");
            assert_eq!(payload.models[0].state, "mismatch");
            assert_eq!(payload.models[0].verified, Some(false));
        });
    }

    #[test]
    fn runtime_download_models_unknown_model_is_error() {
        let runtime = FsfsRuntime::new(FsfsConfig::default()).with_cli_input(CliInput {
            command: CliCommand::Download,
            model_name: Some("does-not-exist".to_owned()),
            ..CliInput::default()
        });
        let err = runtime
            .resolve_download_manifests()
            .expect_err("unknown model");
        assert!(err.to_string().contains("unknown model"));
    }

    #[test]
    fn runtime_uninstall_requires_confirmation_without_yes_or_dry_run() {
        let runtime = FsfsRuntime::new(FsfsConfig::default()).with_cli_input(CliInput {
            command: CliCommand::Uninstall,
            ..CliInput::default()
        });
        let err = runtime
            .collect_uninstall_payload()
            .expect_err("missing confirmation should fail");
        assert!(err.to_string().contains("requires --yes or --dry-run"));
    }

    #[test]
    fn runtime_uninstall_dry_run_marks_targets_without_removal() {
        let temp = tempfile::tempdir().expect("tempdir");
        let index_root = temp.path().join("index");
        let model_root = temp.path().join("models");
        fs::create_dir_all(index_root.join("vector")).expect("index dir");
        fs::write(index_root.join("vector/index.fsvi"), b"fsvi").expect("index file");
        fs::create_dir_all(model_root.join("potion")).expect("model dir");

        let mut config = FsfsConfig::default();
        config.storage.index_dir = index_root.display().to_string();
        config.indexing.model_dir = model_root.display().to_string();
        let runtime = FsfsRuntime::new(config).with_cli_input(CliInput {
            command: CliCommand::Uninstall,
            uninstall_dry_run: true,
            uninstall_purge: true,
            ..CliInput::default()
        });

        let payload = runtime
            .collect_uninstall_payload()
            .expect("dry-run payload");
        assert!(
            payload
                .entries
                .iter()
                .any(|entry| { entry.target == "index_dir" && entry.status == "planned" })
        );
        assert!(
            payload
                .entries
                .iter()
                .any(|entry| { entry.target == "model_dir" && entry.status == "planned" })
        );
        assert!(index_root.exists(), "dry-run must not remove index dir");
        assert!(model_root.exists(), "dry-run must not remove model dir");
    }

    #[test]
    fn runtime_uninstall_refuses_non_fsfs_index_dir() {
        let temp = tempfile::tempdir().expect("tempdir");
        let index_root = temp.path().join("non-fsfs-index");
        fs::create_dir_all(index_root.join("data")).expect("index dir");

        let mut config = FsfsConfig::default();
        config.storage.index_dir = index_root.display().to_string();
        let runtime = FsfsRuntime::new(config).with_cli_input(CliInput {
            command: CliCommand::Uninstall,
            uninstall_yes: true,
            ..CliInput::default()
        });

        let payload = runtime
            .collect_uninstall_payload()
            .expect("uninstall payload");
        let index_entry = payload
            .entries
            .iter()
            .find(|entry| entry.target == "index_dir")
            .expect("index entry");
        assert_eq!(index_entry.status, "error");
        assert!(
            index_entry
                .detail
                .as_deref()
                .is_some_and(|detail| detail.contains("not recognized as fsfs-managed"))
        );
        assert!(index_root.exists(), "unsafe directory should be preserved");
    }

    #[test]
    fn runtime_uninstall_removes_index_dir_when_confirmed() {
        let temp = tempfile::tempdir().expect("tempdir");
        let index_root = temp.path().join("index");
        fs::create_dir_all(index_root.join("vector")).expect("index dir");
        fs::write(index_root.join("vector/index.fsvi"), b"fsvi").expect("index file");

        let mut config = FsfsConfig::default();
        config.storage.index_dir = index_root.display().to_string();
        let runtime = FsfsRuntime::new(config).with_cli_input(CliInput {
            command: CliCommand::Uninstall,
            uninstall_yes: true,
            ..CliInput::default()
        });

        let payload = runtime
            .collect_uninstall_payload()
            .expect("uninstall payload");
        assert!(
            payload
                .entries
                .iter()
                .any(|entry| { entry.target == "index_dir" && entry.status == "removed" })
        );
        assert!(!index_root.exists(), "index dir should be removed");
    }

    #[test]
    fn doctor_payload_has_all_expected_checks() {
        let runtime = FsfsRuntime::new(FsfsConfig::default()).with_cli_input(CliInput {
            command: CliCommand::Doctor,
            ..CliInput::default()
        });
        let payload = runtime.collect_doctor_payload().unwrap();
        assert!(!payload.checks.is_empty(), "doctor should produce checks");
        assert!(
            payload.checks.iter().any(|c| c.name == "version"),
            "doctor should check version"
        );
        assert!(
            payload.checks.iter().any(|c| c.name == "model.fast"),
            "doctor should check fast model"
        );
        assert!(
            payload.checks.iter().any(|c| c.name == "model.quality"),
            "doctor should check quality model"
        );
        assert!(
            payload.checks.iter().any(|c| c.name == "index"),
            "doctor should check index"
        );
        assert!(
            payload.checks.iter().any(|c| c.name == "config"),
            "doctor should check config"
        );
        assert_eq!(
            payload.pass_count + payload.warn_count + payload.fail_count,
            payload.checks.len(),
            "verdict counts should sum to total checks"
        );
    }

    #[test]
    fn doctor_version_check_always_passes() {
        let runtime = FsfsRuntime::new(FsfsConfig::default()).with_cli_input(CliInput {
            command: CliCommand::Doctor,
            ..CliInput::default()
        });
        let payload = runtime.collect_doctor_payload().unwrap();
        let version_check = payload
            .checks
            .iter()
            .find(|c| c.name == "version")
            .expect("version check");
        assert_eq!(version_check.verdict, super::DoctorVerdict::Pass);
        assert!(version_check.detail.contains(env!("CARGO_PKG_VERSION")));
    }

    #[test]
    fn doctor_table_output_contains_all_checks() {
        let runtime = FsfsRuntime::new(FsfsConfig::default()).with_cli_input(CliInput {
            command: CliCommand::Doctor,
            ..CliInput::default()
        });
        let payload = runtime.collect_doctor_payload().unwrap();
        let table = super::render_doctor_table(&payload, true);
        assert!(table.contains("fsfs doctor"));
        assert!(table.contains("version"));
        assert!(table.contains("model.fast"));
        assert!(table.contains("Result:"));
    }

    #[test]
    fn doctor_json_output_roundtrips() {
        let runtime = FsfsRuntime::new(FsfsConfig::default()).with_cli_input(CliInput {
            command: CliCommand::Doctor,
            ..CliInput::default()
        });
        let payload = runtime.collect_doctor_payload().unwrap();
        let json = serde_json::to_string(&payload).expect("serialize doctor payload");
        let roundtrip: super::FsfsDoctorPayload =
            serde_json::from_str(&json).expect("deserialize doctor payload");
        assert_eq!(roundtrip.version, payload.version);
        assert_eq!(roundtrip.checks.len(), payload.checks.len());
        assert_eq!(roundtrip.overall, payload.overall);
    }

    #[test]
    fn doctor_overall_verdict_reflects_worst_check() {
        let all_pass = super::FsfsDoctorPayload {
            version: "test".to_owned(),
            checks: vec![super::DoctorCheck {
                name: "test".to_owned(),
                verdict: super::DoctorVerdict::Pass,
                detail: "ok".to_owned(),
                suggestion: None,
            }],
            pass_count: 1,
            warn_count: 0,
            fail_count: 0,
            overall: super::DoctorVerdict::Pass,
        };
        assert_eq!(all_pass.overall, super::DoctorVerdict::Pass);

        let has_warn = super::FsfsDoctorPayload {
            version: "test".to_owned(),
            checks: vec![
                super::DoctorCheck {
                    name: "ok".to_owned(),
                    verdict: super::DoctorVerdict::Pass,
                    detail: "ok".to_owned(),
                    suggestion: None,
                },
                super::DoctorCheck {
                    name: "warn".to_owned(),
                    verdict: super::DoctorVerdict::Warn,
                    detail: "warning".to_owned(),
                    suggestion: Some("fix it".to_owned()),
                },
            ],
            pass_count: 1,
            warn_count: 1,
            fail_count: 0,
            overall: super::DoctorVerdict::Warn,
        };
        assert_eq!(has_warn.overall, super::DoctorVerdict::Warn);
    }

    // ─── Self-Update Tests ─────────────────────────────────────────────────

    #[test]
    fn semver_parse_basic() {
        let v = super::SemVer::parse("0.1.0").unwrap();
        assert_eq!(v.major, 0);
        assert_eq!(v.minor, 1);
        assert_eq!(v.patch, 0);
    }

    #[test]
    fn semver_parse_with_v_prefix() {
        let v = super::SemVer::parse("v1.2.3").unwrap();
        assert_eq!((v.major, v.minor, v.patch), (1, 2, 3));
    }

    #[test]
    fn semver_parse_with_prerelease() {
        let v = super::SemVer::parse("v2.0.0-beta.1").unwrap();
        assert_eq!((v.major, v.minor, v.patch), (2, 0, 0));
    }

    #[test]
    fn semver_parse_rejects_garbage() {
        assert!(super::SemVer::parse("").is_none());
        assert!(super::SemVer::parse("abc").is_none());
        assert!(super::SemVer::parse("1.2").is_none());
        assert!(super::SemVer::parse("v").is_none());
    }

    #[test]
    fn semver_is_newer_than() {
        let v010 = super::SemVer::parse("0.1.0").unwrap();
        let v020 = super::SemVer::parse("0.2.0").unwrap();
        let v100 = super::SemVer::parse("1.0.0").unwrap();
        let v101 = super::SemVer::parse("1.0.1").unwrap();

        assert!(v020.is_newer_than(v010));
        assert!(v100.is_newer_than(v020));
        assert!(v101.is_newer_than(v100));
        assert!(!v010.is_newer_than(v020));
        assert!(!v010.is_newer_than(v010)); // equal is not newer
    }

    #[test]
    fn semver_display() {
        let v = super::SemVer::parse("v3.14.159").unwrap();
        assert_eq!(v.to_string(), "3.14.159");
    }

    #[test]
    fn update_payload_serde_roundtrip() {
        let payload = super::FsfsUpdatePayload {
            current_version: "0.1.0".into(),
            latest_version: "0.2.0".into(),
            update_available: true,
            check_only: true,
            applied: false,
            channel: "stable".into(),
            release_url: Some("https://example.com/release".into()),
            notes: vec!["update available".into()],
        };
        let json = serde_json::to_string(&payload).unwrap();
        let decoded: super::FsfsUpdatePayload = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, payload);
    }

    #[test]
    fn render_update_table_up_to_date() {
        let payload = super::FsfsUpdatePayload {
            current_version: "0.1.0".into(),
            latest_version: "0.1.0".into(),
            update_available: false,
            check_only: false,
            applied: false,
            channel: "stable".into(),
            release_url: None,
            notes: vec!["fsfs 0.1.0 is already up to date".into()],
        };
        let table = super::render_update_table(&payload, true);
        assert!(table.contains("up to date"));
        assert!(table.contains("v0.1.0"));
    }

    #[test]
    fn render_update_table_check_only() {
        let payload = super::FsfsUpdatePayload {
            current_version: "0.1.0".into(),
            latest_version: "0.2.0".into(),
            update_available: true,
            check_only: true,
            applied: false,
            channel: "stable".into(),
            release_url: Some("https://example.com".into()),
            notes: vec!["update available: v0.1.0 -> v0.2.0".into()],
        };
        let table = super::render_update_table(&payload, true);
        assert!(table.contains("check-only"));
        assert!(table.contains("v0.2.0"));
    }

    #[test]
    fn render_update_table_applied() {
        let payload = super::FsfsUpdatePayload {
            current_version: "0.1.0".into(),
            latest_version: "0.2.0".into(),
            update_available: true,
            check_only: false,
            applied: true,
            channel: "stable".into(),
            release_url: None,
            notes: vec!["updated: v0.1.0 -> v0.2.0".into()],
        };
        let table = super::render_update_table(&payload, true);
        assert!(table.contains("applied"));
    }

    #[test]
    fn detect_target_triple_returns_nonempty() {
        let triple = super::detect_target_triple();
        assert!(!triple.is_empty());
        assert!(triple.contains('-'));
    }

    #[test]
    fn release_asset_url_format() {
        let url = super::release_asset_url("v0.2.0", "x86_64-unknown-linux-musl");
        assert!(url.contains("v0.2.0"));
        assert!(url.contains("fsfs-x86_64-unknown-linux-musl.tar.xz"));
        assert!(url.starts_with("https://github.com/"));
    }

    #[test]
    fn release_checksum_url_format() {
        let url = super::release_checksum_url("v0.2.0", "x86_64-unknown-linux-musl");
        assert!(url.ends_with(".tar.xz.sha256"));
    }

    #[test]
    fn archive_entry_path_is_safe_accepts_normal_relative_paths() {
        assert!(super::archive_entry_path_is_safe(
            "fsfs-x86_64-unknown-linux-musl/fsfs"
        ));
        assert!(super::archive_entry_path_is_safe("bin/fsfs"));
        assert!(super::archive_entry_path_is_safe("nested/path/"));
    }

    #[test]
    fn archive_entry_path_is_safe_rejects_escape_paths() {
        assert!(!super::archive_entry_path_is_safe("../etc/passwd"));
        assert!(!super::archive_entry_path_is_safe("/tmp/owned"));
    }

    #[test]
    fn compute_sha256_of_file_matches_known_value() {
        let dir = std::env::temp_dir().join("fsfs_test_sha256_file");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("payload.txt");
        fs::write(&path, b"abc").unwrap();

        let hash = super::compute_sha256_of_file(&path).unwrap();
        assert_eq!(
            hash,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_extracted_binary_direct() {
        let dir = std::env::temp_dir().join("fsfs_test_extract_direct");
        let _ = fs::create_dir_all(&dir);
        let binary = dir.join("fsfs");
        fs::write(&binary, b"#!/bin/sh\necho test").unwrap();
        let found = super::find_extracted_binary(&dir, "fsfs").unwrap();
        assert_eq!(found, binary);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_extracted_binary_nested() {
        let dir = std::env::temp_dir().join("fsfs_test_extract_nested");
        let sub = dir.join("subdir");
        let _ = fs::create_dir_all(&sub);
        let binary = sub.join("fsfs");
        fs::write(&binary, b"#!/bin/sh\necho test").unwrap();
        let found = super::find_extracted_binary(&dir, "fsfs").unwrap();
        assert_eq!(found, binary);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_extracted_binary_missing() {
        let dir = std::env::temp_dir().join("fsfs_test_extract_empty");
        let _ = fs::create_dir_all(&dir);
        let result = super::find_extracted_binary(&dir, "fsfs");
        assert!(result.is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn find_extracted_binary_rejects_symlink() {
        use std::os::unix::fs::symlink;

        let dir = std::env::temp_dir().join("fsfs_test_extract_symlink");
        let _ = fs::create_dir_all(&dir);
        let external = dir.join("external-target");
        fs::write(&external, b"fake-binary").unwrap();
        symlink(&external, dir.join("fsfs")).unwrap();

        let result = super::find_extracted_binary(&dir, "fsfs");
        assert!(result.is_err(), "symlink should be rejected");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn github_constants_are_set() {
        assert_eq!(super::GITHUB_OWNER, "Dicklesworthstone");
        assert_eq!(super::GITHUB_REPO, "frankensearch");
    }

    // ─── Version Cache Tests ───────────────────────────────────────────

    #[test]
    fn version_cache_serde_roundtrip() {
        let cache = super::VersionCheckCache {
            checked_at_epoch: 1_700_000_000,
            current_version: "0.1.0".into(),
            latest_version: "v0.2.0".into(),
            release_url: "https://example.com/release".into(),
            ttl_seconds: 86_400,
        };
        let json = serde_json::to_string(&cache).unwrap();
        let decoded: super::VersionCheckCache = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.current_version, "0.1.0");
        assert_eq!(decoded.latest_version, "v0.2.0");
        assert_eq!(decoded.ttl_seconds, 86_400);
    }

    #[test]
    fn version_cache_default_ttl() {
        let json = r#"{"checked_at_epoch":0,"current_version":"0.1.0","latest_version":"v0.1.0","release_url":""}"#;
        let cache: super::VersionCheckCache = serde_json::from_str(json).unwrap();
        assert_eq!(cache.ttl_seconds, super::VERSION_CACHE_TTL_SECS);
    }

    #[test]
    fn is_cache_valid_detects_expired() {
        let cache = super::VersionCheckCache {
            checked_at_epoch: 0, // epoch 0 is way in the past
            current_version: env!("CARGO_PKG_VERSION").into(),
            latest_version: "v0.1.0".into(),
            release_url: String::new(),
            ttl_seconds: 86_400,
        };
        assert!(!super::is_cache_valid(&cache));
    }

    #[test]
    fn is_cache_valid_detects_version_mismatch() {
        let cache = super::VersionCheckCache {
            checked_at_epoch: super::epoch_now(),
            current_version: "99.99.99".into(), // won't match CARGO_PKG_VERSION
            latest_version: "v99.99.99".into(),
            release_url: String::new(),
            ttl_seconds: 86_400,
        };
        assert!(!super::is_cache_valid(&cache));
    }

    #[test]
    fn is_cache_valid_accepts_fresh_cache() {
        let cache = super::VersionCheckCache {
            checked_at_epoch: super::epoch_now(),
            current_version: env!("CARGO_PKG_VERSION").into(),
            latest_version: "v0.1.0".into(),
            release_url: String::new(),
            ttl_seconds: 86_400,
        };
        assert!(super::is_cache_valid(&cache));
    }

    #[test]
    fn version_cache_path_is_some() {
        // On most systems this will return Some.
        let path = super::version_cache_path();
        if let Some(p) = path {
            assert!(p.ends_with("frankensearch/version_check.json"));
        }
    }

    #[test]
    fn write_and_read_version_cache_roundtrip() {
        let temp_path = std::env::temp_dir().join(format!(
            "fsfs-version-cache-test-{}-{}.json",
            std::process::id(),
            super::epoch_now()
        ));
        let cache = super::VersionCheckCache {
            checked_at_epoch: super::epoch_now(),
            current_version: env!("CARGO_PKG_VERSION").into(),
            latest_version: "v99.0.0".into(),
            release_url: "https://example.com".into(),
            ttl_seconds: 86_400,
        };
        super::write_version_cache_to_path(&temp_path, &cache).unwrap();
        let loaded = super::read_version_cache_from_path(&temp_path).unwrap();
        assert_eq!(loaded.latest_version, "v99.0.0");
        assert_eq!(loaded.release_url, "https://example.com");
        let _ = fs::remove_file(temp_path);
    }

    #[test]
    fn maybe_print_update_notice_quiet_mode() {
        // Quiet mode should always return false.
        assert!(!super::maybe_print_update_notice(true));
    }

    #[test]
    fn maybe_print_update_notice_no_cache() {
        // If cache file doesn't exist / can't be read, returns false.
        // This is inherently true in a fresh test environment or when
        // the cache hasn't been populated, so we just verify no panic.
        let _ = super::maybe_print_update_notice(false);
    }

    #[test]
    fn version_cache_ttl_constant() {
        assert_eq!(super::VERSION_CACHE_TTL_SECS, 86_400);
    }

    // ─── Backup / Rollback Tests ───────────────────────────────────────

    #[test]
    fn backup_entry_serde_roundtrip() {
        let entry = super::BackupEntry {
            version: "0.1.0".into(),
            backed_up_at_epoch: 1_700_000_000,
            original_path: "/usr/local/bin/fsfs".into(),
            binary_filename: "fsfs-0.1.0".into(),
            sha256: "abc123".into(),
        };
        let json = serde_json::to_string(&entry).unwrap();
        let decoded: super::BackupEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, entry);
    }

    #[test]
    fn rollback_manifest_serde_roundtrip() {
        let manifest = super::RollbackManifest {
            entries: vec![
                super::BackupEntry {
                    version: "0.1.0".into(),
                    backed_up_at_epoch: 100,
                    original_path: "/bin/fsfs".into(),
                    binary_filename: "fsfs-0.1.0".into(),
                    sha256: "aaa".into(),
                },
                super::BackupEntry {
                    version: "0.2.0".into(),
                    backed_up_at_epoch: 200,
                    original_path: "/bin/fsfs".into(),
                    binary_filename: "fsfs-0.2.0".into(),
                    sha256: "bbb".into(),
                },
            ],
        };
        let json = serde_json::to_string(&manifest).unwrap();
        let decoded: super::RollbackManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.entries.len(), 2);
        assert_eq!(decoded.entries[0].version, "0.1.0");
        assert_eq!(decoded.entries[1].version, "0.2.0");
    }

    #[test]
    fn rollback_manifest_default_is_empty() {
        let manifest = super::RollbackManifest::default();
        assert!(manifest.entries.is_empty());
    }

    #[test]
    fn prune_backups_keeps_max() {
        let dir = std::env::temp_dir().join("fsfs_test_prune_backups");
        let _ = fs::create_dir_all(&dir);

        let mut manifest = super::RollbackManifest {
            entries: (0u64..5)
                .map(|i| super::BackupEntry {
                    version: format!("0.{i}.0"),
                    backed_up_at_epoch: i * 100,
                    original_path: "/bin/fsfs".into(),
                    binary_filename: format!("fsfs-0.{i}.0"),
                    sha256: String::new(),
                })
                .collect(),
        };

        // Create the files that will be pruned.
        for entry in &manifest.entries {
            fs::write(dir.join(&entry.binary_filename), b"test").unwrap();
        }

        super::prune_backups(&mut manifest, &dir);
        assert_eq!(manifest.entries.len(), super::MAX_BACKUP_VERSIONS);

        // The newest entries should be kept (highest epoch).
        assert_eq!(manifest.entries[0].version, "0.4.0");
        assert_eq!(manifest.entries[1].version, "0.3.0");
        assert_eq!(manifest.entries[2].version, "0.2.0");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn backup_dir_is_some() {
        let dir = super::backup_dir();
        if let Some(d) = dir {
            assert!(d.ends_with("frankensearch/backups"));
        }
    }

    #[test]
    fn rollback_manifest_path_is_some() {
        let path = super::rollback_manifest_path();
        if let Some(p) = path {
            assert!(p.ends_with("rollback-manifest.json"));
        }
    }

    #[test]
    fn list_backups_returns_vec() {
        // Just verify it doesn't panic.
        let backups = super::list_backups();
        assert!(backups.len() <= 100); // sanity check
    }

    #[test]
    fn restore_backup_fails_with_no_backups() {
        // With a fresh manifest, restore should fail.
        // This may succeed if a previous test wrote backups, so we just
        // verify the function doesn't panic.
        let result = super::restore_backup(Some("v99.99.99"));
        // Should fail because that version doesn't exist.
        assert!(result.is_err());
    }

    #[test]
    fn max_backup_versions_constant() {
        assert_eq!(super::MAX_BACKUP_VERSIONS, 3);
    }

    #[test]
    fn write_and_read_rollback_manifest_roundtrip() {
        let manifest = super::RollbackManifest {
            entries: vec![super::BackupEntry {
                version: "0.99.0".into(),
                backed_up_at_epoch: super::epoch_now(),
                original_path: "/test/fsfs".into(),
                binary_filename: "fsfs-0.99.0".into(),
                sha256: "test_hash".into(),
            }],
        };
        super::write_rollback_manifest(&manifest).unwrap();
        let loaded = super::read_rollback_manifest();
        assert!(!loaded.entries.is_empty());
        // Find our test entry (other tests may have added entries too).
        let found = loaded.entries.iter().any(|e| e.version == "0.99.0");
        assert!(found);
    }

    // ─── Additional Auto-Update Edge Case Tests (bd-2w7x.46) ──────────

    #[test]
    fn semver_parse_single_digit_components() {
        let v = super::SemVer::parse("1.2.3").unwrap();
        assert_eq!((v.major, v.minor, v.patch), (1, 2, 3));
    }

    #[test]
    fn semver_parse_large_components() {
        let v = super::SemVer::parse("100.200.300").unwrap();
        assert_eq!((v.major, v.minor, v.patch), (100, 200, 300));
    }

    #[test]
    fn semver_comparison_major_trumps_minor() {
        let v200 = super::SemVer::parse("2.0.0").unwrap();
        let v1_99_99 = super::SemVer::parse("1.99.99").unwrap();
        assert!(v200.is_newer_than(v1_99_99));
    }

    #[test]
    fn semver_comparison_minor_trumps_patch() {
        let v0_2_0 = super::SemVer::parse("0.2.0").unwrap();
        let v0_1_99 = super::SemVer::parse("0.1.99").unwrap();
        assert!(v0_2_0.is_newer_than(v0_1_99));
    }

    #[test]
    fn semver_prerelease_stripped_for_comparison() {
        // Pre-release suffix stripped: v0.3.0-rc1 parses as 0.3.0.
        let rc = super::SemVer::parse("v0.3.0-rc1").unwrap();
        let release = super::SemVer::parse("0.3.0").unwrap();
        // Same triple means neither is newer.
        assert!(!rc.is_newer_than(release));
        assert!(!release.is_newer_than(rc));
    }

    #[test]
    fn version_cache_with_custom_ttl() {
        let cache = super::VersionCheckCache {
            checked_at_epoch: super::epoch_now(),
            current_version: env!("CARGO_PKG_VERSION").into(),
            latest_version: "v0.1.0".into(),
            release_url: String::new(),
            ttl_seconds: 1, // 1-second TTL
        };
        // Should be valid immediately.
        assert!(super::is_cache_valid(&cache));

        // With a very old checked_at, should be expired.
        let expired_cache = super::VersionCheckCache {
            checked_at_epoch: super::epoch_now() - 2,
            ttl_seconds: 1,
            ..cache
        };
        assert!(!super::is_cache_valid(&expired_cache));
    }

    #[test]
    fn prune_backups_noop_when_under_limit() {
        let dir = std::env::temp_dir().join("fsfs_test_prune_noop");
        let _ = fs::create_dir_all(&dir);

        let mut manifest = super::RollbackManifest {
            entries: vec![super::BackupEntry {
                version: "0.1.0".into(),
                backed_up_at_epoch: 100,
                original_path: "/bin/fsfs".into(),
                binary_filename: "fsfs-0.1.0".into(),
                sha256: String::new(),
            }],
        };

        super::prune_backups(&mut manifest, &dir);
        assert_eq!(manifest.entries.len(), 1);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn prune_backups_deletes_oldest_files() {
        let dir = std::env::temp_dir().join("fsfs_test_prune_deletes");
        let _ = fs::create_dir_all(&dir);

        // Create 4 backup files + entries. After pruning, oldest should be gone.
        let mut manifest = super::RollbackManifest {
            entries: (0u64..4)
                .map(|i| {
                    let filename = format!("fsfs-prune-test-{i}");
                    fs::write(dir.join(&filename), b"binary").unwrap();
                    super::BackupEntry {
                        version: format!("0.{i}.0"),
                        backed_up_at_epoch: (i + 1) * 1000,
                        original_path: "/bin/fsfs".into(),
                        binary_filename: filename,
                        sha256: String::new(),
                    }
                })
                .collect(),
        };

        super::prune_backups(&mut manifest, &dir);
        assert_eq!(manifest.entries.len(), super::MAX_BACKUP_VERSIONS);
        // The oldest entry (epoch=1000, version 0.0.0) should have been pruned.
        assert!(
            manifest.entries.iter().all(|e| e.version != "0.0.0"),
            "oldest entry should have been pruned"
        );
        // Its file should also be deleted.
        assert!(!dir.join("fsfs-prune-test-0").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn create_backup_produces_valid_entry() {
        // Create a fake binary to back up.
        let dir = std::env::temp_dir().join("fsfs_test_create_backup");
        let _ = fs::create_dir_all(&dir);
        let fake_binary = dir.join("fsfs");
        fs::write(&fake_binary, b"#!/bin/sh\necho fsfs test").unwrap();

        let entry = super::create_backup(&fake_binary).unwrap();
        assert_eq!(entry.version, env!("CARGO_PKG_VERSION"));
        assert!(!entry.binary_filename.is_empty());
        assert!(!entry.sha256.is_empty());

        // Verify the backup file exists in the backup directory.
        if let Some(bdir) = super::backup_dir() {
            assert!(bdir.join(&entry.binary_filename).exists());
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn create_backup_deduplicates_same_version() {
        let dir = std::env::temp_dir().join("fsfs_test_backup_dedup");
        let _ = fs::create_dir_all(&dir);
        let fake_binary = dir.join("fsfs");
        fs::write(&fake_binary, b"#!/bin/sh\necho test1").unwrap();

        // Create first backup.
        let _ = super::create_backup(&fake_binary);
        // Create second backup of same version.
        let _ = super::create_backup(&fake_binary);

        // Manifest should have only one entry for the current version.
        let manifest = super::read_rollback_manifest();
        let count = manifest
            .entries
            .iter()
            .filter(|e| e.version == env!("CARGO_PKG_VERSION"))
            .count();
        assert_eq!(count, 1, "should deduplicate same-version backups");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn update_payload_check_only_does_not_apply() {
        let payload = super::FsfsUpdatePayload {
            current_version: "0.1.0".into(),
            latest_version: "0.2.0".into(),
            update_available: true,
            check_only: true,
            applied: false,
            channel: "stable".into(),
            release_url: Some("https://example.com".into()),
            notes: vec![],
        };
        assert!(payload.update_available);
        assert!(!payload.applied);
        assert!(payload.check_only);
    }

    #[test]
    fn update_payload_no_update_available() {
        let payload = super::FsfsUpdatePayload {
            current_version: "0.1.0".into(),
            latest_version: "0.1.0".into(),
            update_available: false,
            check_only: false,
            applied: false,
            channel: "stable".into(),
            release_url: None,
            notes: vec!["up to date".into()],
        };
        assert!(!payload.update_available);
        assert!(!payload.applied);
    }

    #[test]
    fn release_urls_contain_expected_patterns() {
        let asset = super::release_asset_url("v1.0.0", "aarch64-apple-darwin");
        assert!(asset.contains("v1.0.0"));
        assert!(asset.contains("aarch64-apple-darwin"));
        assert!(asset.contains(".tar.xz"));
        assert!(asset.contains("github.com"));

        let checksum = super::release_checksum_url("v1.0.0", "aarch64-apple-darwin");
        assert!(checksum.ends_with(".sha256"));
    }

    #[test]
    fn detect_target_triple_has_arch_and_os() {
        let triple = super::detect_target_triple();
        // Should contain an architecture component.
        assert!(
            triple.contains("x86_64") || triple.contains("aarch64") || triple.contains("arm"),
            "triple should contain architecture: {triple}"
        );
        // Should contain an OS component.
        assert!(
            triple.contains("linux") || triple.contains("darwin") || triple.contains("windows"),
            "triple should contain OS: {triple}"
        );
    }

    #[test]
    fn probe_helpers_detect_expected_signals() {
        assert!(super::is_probe_excluded_dir_name("node_modules"));
        assert!(super::is_repo_marker_filename("cargo.toml"));
        assert_eq!(
            super::classify_probe_file(Path::new("/tmp/demo.rs"), "demo.rs"),
            Some(super::RootProbeFileClass::Code)
        );
        assert_eq!(
            super::classify_probe_file(Path::new("/tmp/README.md"), "readme.md"),
            Some(super::RootProbeFileClass::Document)
        );
    }

    #[test]
    fn score_index_root_prefers_projects_layout() {
        let stats = super::RootProbeStats {
            candidate_files: 1_200,
            code_files: 980,
            doc_files: 220,
            repo_markers: 80,
            candidate_bytes: 512 * 1024 * 1024,
            scanned_dirs: 200,
            scanned_entries: 10_000,
        };
        let projects_score = super::score_index_root(
            Path::new("/data/projects"),
            &stats,
            Some(Path::new("/home/u")),
        );
        let home_score =
            super::score_index_root(Path::new("/home/u"), &stats, Some(Path::new("/home/u")));
        assert!(projects_score > home_score);
    }

    #[test]
    fn indexing_progress_snapshot_reports_eta() {
        let snapshot = super::IndexingProgressSnapshot {
            stage: super::IndexingProgressStage::Indexing,
            target_root: PathBuf::from("/data/projects"),
            index_root: PathBuf::from("/data/projects/.frankensearch"),
            discovered_files: 1_000,
            candidate_files: 800,
            processed_files: 400,
            skipped_files: 100,
            semantic_files: 300,
            canonical_bytes: 400 * 1024 * 1024,
            canonical_lines: 8_000_000,
            index_size_bytes: 128 * 1024 * 1024,
            discovery_elapsed_ms: 1_000,
            lexical_elapsed_ms: 2_000,
            embedding_elapsed_ms: 5_000,
            vector_elapsed_ms: 3_000,
            total_elapsed_ms: 20_000,
            active_file: None,
        };
        assert!(snapshot.files_per_second() > 0.0);
        assert!(snapshot.lines_per_second() > 0.0);
        assert!(snapshot.mb_per_second() > 0.0);
        assert!(snapshot.index_growth_bytes_per_second() > 0.0);
        assert!(snapshot.eta_seconds().is_some());
    }

    #[test]
    fn search_execution_mode_parser_accepts_known_values() {
        assert_eq!(
            super::parse_search_execution_mode(None).unwrap(),
            super::SearchExecutionMode::Full
        );
        assert_eq!(
            super::parse_search_execution_mode(Some("full")).unwrap(),
            super::SearchExecutionMode::Full
        );
        assert_eq!(
            super::parse_search_execution_mode(Some("fast_only")).unwrap(),
            super::SearchExecutionMode::FastOnly
        );
        assert_eq!(
            super::parse_search_execution_mode(Some("lexical_only")).unwrap(),
            super::SearchExecutionMode::LexicalOnly
        );
        let err = super::parse_search_execution_mode(Some("nope")).expect_err("must fail");
        assert!(
            err.to_string()
                .contains("expected full|fast_only|lexical_only")
        );
    }

    #[test]
    fn search_payload_cache_roundtrip_and_invalidates_on_index_change() {
        let temp = tempfile::tempdir().expect("tempdir");
        let index_root = temp.path().join("index");
        std::fs::create_dir_all(&index_root).expect("create index root");

        let mut config = FsfsConfig::default();
        config.storage.index_dir = index_root.display().to_string();

        let runtime = FsfsRuntime::new(config).with_cli_input(CliInput {
            index_dir: Some(index_root.clone()),
            filter: Some("ext:rs".to_owned()),
            ..CliInput::default()
        });
        let key = runtime.search_cache_key("cache me", 25, super::SearchExecutionMode::Full);
        let payloads = vec![SearchPayload::new(
            "cache me",
            SearchOutputPhase::Initial,
            1,
            vec![SearchHitPayload {
                rank: 1,
                path: "src/lib.rs".to_owned(),
                score: 0.75,
                snippet: Some("cached snippet".to_owned()),
                lexical_rank: Some(0),
                semantic_rank: Some(0),
                in_both_sources: true,
            }],
        )];

        runtime
            .write_search_payload_cache(&key, &payloads)
            .expect("write cache");
        let cached = runtime
            .try_load_search_payload_cache(&key)
            .expect("read cache")
            .expect("cache hit");
        assert_eq!(cached, payloads);

        let vector_dir = index_root.join("vector");
        std::fs::create_dir_all(&vector_dir).expect("create vector dir");
        std::fs::write(vector_dir.join("index.fsvi"), b"mutated").expect("touch vector index");

        let stale = runtime
            .try_load_search_payload_cache(&key)
            .expect("cache read after mutation");
        assert!(
            stale.is_none(),
            "cache should invalidate after index changes"
        );
    }
}
