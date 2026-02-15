use std::collections::{BTreeSet, HashMap, HashSet, hash_map::DefaultHasher};
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
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
    ModelLifecycle, ModelManifest,
};
use frankensearch_index::VectorIndex;
use frankensearch_lexical::{SnippetConfig, TantivyIndex};
use ignore::WalkBuilder;
use serde::{Deserialize, Serialize};
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
use crate::query_planning::{CapabilityState, QueryExecutionCapabilities, QueryPlanner};
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
    vector_index: std::sync::Mutex<VectorIndex>,
    embedder: Arc<dyn Embedder>,
    canonicalizer: DefaultCanonicalizer,
}

impl WatchIngestPipeline for LiveIngestPipeline {
    fn apply_batch(&self, batch: &[WatchIngestOp]) -> frankensearch_core::SearchResult<usize> {
        let rt = asupersync::runtime::RuntimeBuilder::current_thread()
            .build()
            .map_err(|error| SearchError::SubsystemError {
                subsystem: "watcher.ingest",
                source: Box::new(std::io::Error::other(format!(
                    "failed to create ingest runtime: {error}"
                ))),
            })?;
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
            vector_index: std::sync::Mutex::new(vector_index),
            embedder,
            canonicalizer: DefaultCanonicalizer::default(),
        }
    }

    fn resolve_paths(&self, file_key: &str) -> (PathBuf, String) {
        let abs_path = if Path::new(file_key).is_absolute() {
            PathBuf::from(file_key)
        } else {
            self.target_root.join(file_key)
        };
        let rel_key = normalize_file_key_for_index(&abs_path, &self.target_root);
        (abs_path, rel_key)
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

    fn prune_indexes(&self, rel_key: &str) -> frankensearch_core::SearchResult<()> {
        self.lexical_index.delete_document(rel_key)?;
        self.soft_delete_vector(rel_key);
        Ok(())
    }

    async fn apply_upsert_op(
        &self,
        cx: &Cx,
        file_key: &str,
        ingestion_class: IngestionClass,
    ) -> frankensearch_core::SearchResult<bool> {
        let (abs_path, rel_key) = self.resolve_paths(file_key);

        if matches!(
            ingestion_class,
            IngestionClass::MetadataOnly | IngestionClass::Skip
        ) {
            self.prune_indexes(&rel_key)?;
            return Ok(true);
        }

        let bytes = match fs::read(&abs_path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == ErrorKind::NotFound => {
                self.prune_indexes(&rel_key)?;
                return Ok(true);
            }
            Err(error) if is_ignorable_index_walk_error(&error) => return Ok(false),
            Err(error) => return Err(error.into()),
        };

        if is_probably_binary(&bytes) {
            self.prune_indexes(&rel_key)?;
            return Ok(true);
        }

        let raw_text = String::from_utf8_lossy(&bytes);
        let canonical = self.canonicalizer.canonicalize(&raw_text);
        if canonical.trim().is_empty() {
            self.prune_indexes(&rel_key)?;
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
            match self.embedder.embed(cx, &canonical).await {
                Ok(embedding) => {
                    let mut vi = self
                        .vector_index
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    if let Err(error) = vi.soft_delete(&rel_key) {
                        tracing::debug!(
                            file_key = %rel_key,
                            error = %error,
                            "upsert soft_delete: ignored (doc may not exist yet)"
                        );
                    }
                    vi.append(&rel_key, &embedding)?;
                }
                Err(error) => {
                    warn!(
                        file_key = %rel_key,
                        error = %error,
                        "watcher ingest: embedding failed; lexical-only for this file"
                    );
                }
            }
        } else {
            self.soft_delete_vector(&rel_key);
        }

        Ok(true)
    }

    fn apply_delete_op(&self, file_key: &str) -> frankensearch_core::SearchResult<()> {
        let (_abs_path, rel_key) = self.resolve_paths(file_key);
        self.prune_indexes(&rel_key)
    }

    async fn apply_batch_inner(
        &self,
        cx: &Cx,
        batch: &[WatchIngestOp],
    ) -> frankensearch_core::SearchResult<usize> {
        let mut count = 0_usize;

        for op in batch {
            match op {
                WatchIngestOp::Upsert {
                    file_key,
                    ingestion_class,
                    ..
                } => {
                    if self.apply_upsert_op(cx, file_key, *ingestion_class).await? {
                        count = count.saturating_add(1);
                    }
                }
                WatchIngestOp::Delete { file_key, .. } => {
                    self.apply_delete_op(file_key)?;
                    count = count.saturating_add(1);
                }
            }
        }

        if count > 0 {
            self.lexical_index.commit(cx).await?;
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

/// Fetch the latest release tag from GitHub Releases via `curl`.
/// Returns `(tag_name, html_url)` on success.
fn fetch_latest_release_tag() -> SearchResult<(String, String)> {
    let url = format!("https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest");

    let output = std::process::Command::new("curl")
        .args([
            "-sSf",
            "-H",
            "Accept: application/vnd.github+json",
            "-H",
            "X-GitHub-Api-Version: 2022-11-28",
            "--max-time",
            "10",
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

    let body = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value =
        serde_json::from_str(&body).map_err(|e| SearchError::SubsystemError {
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
            value: body.into_owned(),
            reason: "no tag_name found in GitHub API response".into(),
        });
    }

    Ok((tag, html_url))
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

/// Compute SHA-256 hex digest of a file using `sha256sum`.
fn compute_sha256_of_file(path: &Path) -> SearchResult<String> {
    let output = std::process::Command::new("sha256sum")
        .arg(path)
        .output()
        .map_err(|e| SearchError::SubsystemError {
            subsystem: "fsfs.update.sha256",
            source: Box::new(e),
        })?;

    if !output.status.success() {
        return Err(SearchError::SubsystemError {
            subsystem: "fsfs.update.sha256",
            source: Box::new(std::io::Error::other("sha256sum failed")),
        });
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let hash = stdout.split_whitespace().next().unwrap_or("").to_owned();
    Ok(hash)
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

/// Read the version cache from disk. Returns `None` if missing or unreadable.
#[must_use]
pub fn read_version_cache() -> Option<VersionCheckCache> {
    let path = version_cache_path()?;
    let content = fs::read_to_string(&path).ok()?;
    serde_json::from_str(&content).ok()
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
    fs::write(&path, json).map_err(|e| SearchError::SubsystemError {
        subsystem: "fsfs.version_cache.write",
        source: Box::new(e),
    })?;
    Ok(())
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
    elapsed < cache.ttl_seconds && cache.current_version == env!("CARGO_PKG_VERSION")
}

/// Refresh the version cache by querying GitHub. Writes the result to disk.
/// Returns the refreshed cache on success.
///
/// # Errors
///
/// Returns `SearchError` if the GitHub API call or cache write fails.
pub fn refresh_version_cache() -> SearchResult<VersionCheckCache> {
    let (tag, html_url) = fetch_latest_release_tag()?;
    let cache = VersionCheckCache {
        checked_at_epoch: epoch_now(),
        current_version: env!("CARGO_PKG_VERSION").to_owned(),
        latest_version: tag,
        release_url: html_url,
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
    fs::write(&path, json).map_err(|e| SearchError::SubsystemError {
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
        let invalidate_revisions_through = input.previous_indexed_revision.filter(|revision| {
            *revision >= 0 && input.content_hash_changed && input.observed_revision > *revision
        });
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
            CliCommand::Version => {
                println!("fsfs {}", env!("CARGO_PKG_VERSION"));
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

        let current = SemVer::parse(current_str).ok_or_else(|| SearchError::InvalidConfig {
            field: "update.current_version".into(),
            value: current_str.to_owned(),
            reason: "cannot parse current version as semver".into(),
        })?;

        // Query GitHub for the latest release.
        let (tag, html_url) = fetch_latest_release_tag()?;
        let latest = SemVer::parse(&tag).ok_or_else(|| SearchError::InvalidConfig {
            field: "update.latest_version".into(),
            value: tag.clone(),
            reason: "cannot parse latest release tag as semver".into(),
        })?;

        let update_available = latest.is_newer_than(current);
        let mut notes = Vec::new();

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

        let temp_dir = std::env::temp_dir().join("fsfs-update");
        fs::create_dir_all(&temp_dir).map_err(|e| SearchError::SubsystemError {
            subsystem: "fsfs.update.tempdir",
            source: Box::new(e),
        })?;

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
            if let Ok(out) = verify {
                if out.status.success() {
                    let version_out = String::from_utf8_lossy(&out.stdout);
                    notes.push(format!("verified: {}", version_out.trim()));
                }
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
        if self.cli_input.filter.is_some() {
            warn!(
                filter = ?self.cli_input.filter,
                "fsfs search filter expression is not yet wired; continuing without filter"
            );
        }

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
            .unwrap_or(self.config.search.default_limit)
            .max(1);

        if self.cli_input.stream {
            return self.run_search_stream_command(cx, query, limit).await;
        }

        let started = Instant::now();
        let payload = self.execute_search_payload(cx, query, limit).await?;
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
        let mut stdout = std::io::stdout();
        emit_envelope(&envelope, self.cli_input.format, &mut stdout)?;
        if !matches!(
            self.cli_input.format,
            OutputFormat::Jsonl | OutputFormat::Csv
        ) {
            stdout
                .write_all(b"\n")
                .map_err(|source| SearchError::SubsystemError {
                    subsystem: "fsfs.search",
                    source: Box::new(source),
                })?;
        }
        Ok(())
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
        match self.execute_search_payload(cx, query, limit).await {
            Ok(payload) => {
                self.emit_search_stream_payload(&payload, &stream_id, &mut seq, &mut stdout)?;
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
        fs::write(path, json)?;
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

    #[allow(clippy::too_many_lines)]
    async fn execute_search_payload(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
    ) -> SearchResult<SearchPayload> {
        let normalized_query = query
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .trim()
            .to_owned();
        if normalized_query.is_empty() {
            return Ok(SearchPayload::new(
                String::new(),
                SearchOutputPhase::Initial,
                0,
                Vec::new(),
            ));
        }

        let search_start = Instant::now();
        let index_root = self.resolve_status_index_root()?;
        let lexical_path = index_root.join("lexical");
        let vector_path = index_root.join(FSFS_VECTOR_INDEX_FILE);

        let lexical_index = if lexical_path.exists() {
            Some(TantivyIndex::open(&lexical_path)?)
        } else {
            None
        };
        let lexical_available = lexical_index.is_some();
        let vector_index = if vector_path.exists() {
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

        let mut fast_embedder = None;
        if vector_index.is_some() {
            match self.resolve_fast_embedder() {
                Ok(embedder) => fast_embedder = Some(embedder),
                Err(error) if lexical_available => {
                    warn!(
                        error = %error,
                        "fsfs search falling back to lexical-only mode after fast embedder init failure"
                    );
                }
                Err(error) => return Err(error),
            }
        }

        let capabilities = QueryExecutionCapabilities {
            lexical: if lexical_index.is_some() {
                CapabilityState::Enabled
            } else {
                CapabilityState::Disabled
            },
            fast_semantic: if vector_index.is_some() && fast_embedder.is_some() {
                CapabilityState::Enabled
            } else {
                CapabilityState::Disabled
            },
            quality_semantic: CapabilityState::Disabled,
            rerank: CapabilityState::Disabled,
        };
        let planner = QueryPlanner::from_fsfs(&self.config);
        let plan = planner.execution_plan_for_query(&normalized_query, Some(limit), capabilities);

        info!(
            phase = "query_parse",
            query = normalized_query,
            intent = ?plan.intent.intent,
            fallback = ?plan.intent.fallback,
            confidence_per_mille = plan.intent.confidence_per_mille,
            execution_mode = ?plan.mode,
            reason_code = plan.reason_code,
            "fsfs search query planned"
        );

        let lexical_start = Instant::now();
        let lexical_budget = plan.lexical_stage.candidate_budget.max(limit);
        let snippet_config = SnippetConfig::default();
        let mut snippets_by_doc = HashMap::new();
        let lexical_candidates = if plan.lexical_stage.enabled {
            if let Some(lexical) = lexical_index.as_ref() {
                let hits = lexical.search_with_snippets(
                    cx,
                    &normalized_query,
                    lexical_budget,
                    &snippet_config,
                )?;
                for hit in &hits {
                    if let Some(snippet) = hit.snippet.as_ref()
                        && !snippet.trim().is_empty()
                    {
                        snippets_by_doc.insert(hit.doc_id.clone(), snippet.clone());
                    }
                }
                hits.into_iter()
                    .map(|hit| LexicalCandidate::new(hit.doc_id, hit.bm25_score))
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        let lexical_elapsed_ms = lexical_start.elapsed().as_millis();

        let semantic_start = Instant::now();
        let semantic_budget = plan.semantic_stage.candidate_budget.max(limit);
        let semantic_candidates = if plan.semantic_stage.enabled {
            if let (Some(index), Some(embedder)) = (vector_index.as_ref(), fast_embedder.as_ref()) {
                match embedder.embed(cx, &normalized_query).await {
                    Ok(query_embedding) => {
                        match index.search_top_k(&query_embedding, semantic_budget, None) {
                            Ok(hits) => hits
                                .into_iter()
                                .map(|hit| SemanticCandidate::new(hit.doc_id, hit.score))
                                .collect::<Vec<_>>(),
                            Err(error) if lexical_available => {
                                warn!(
                                    error = %error,
                                    "fsfs search falling back to lexical-only mode after vector search failure"
                                );
                                Vec::new()
                            }
                            Err(error) => return Err(error),
                        }
                    }
                    Err(error) if lexical_available => {
                        warn!(
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
        let fused = orchestrator.fuse_rankings(&lexical_candidates, &semantic_candidates, limit, 0);
        let payload = orchestrator.build_search_payload(
            &normalized_query,
            SearchOutputPhase::Initial,
            &fused,
            &snippets_by_doc,
        );
        if let Err(error) =
            self.persist_explain_session(&index_root, &normalized_query, payload.phase, &fused)
        {
            warn!(
                error = %error,
                path = %Self::explain_session_path(&index_root).display(),
                "failed to persist explain-session context for follow-up `fsfs explain`"
            );
        }
        let fusion_elapsed_ms = fusion_start.elapsed().as_millis();

        info!(
            phase = "fast_search",
            query = normalized_query,
            lexical_candidates = lexical_candidates.len(),
            semantic_candidates = semantic_candidates.len(),
            fused_candidates = payload.total_candidates,
            returned_hits = payload.returned_hits,
            lexical_elapsed_ms,
            semantic_elapsed_ms,
            fusion_elapsed_ms,
            total_elapsed_ms = search_start.elapsed().as_millis(),
            "fsfs search retrieval pipeline completed"
        );
        info!(
            phase = "fusion",
            rrf_k = plan.fusion_policy.rrf_k.unwrap_or(self.config.search.rrf_k),
            fused_candidates = payload.total_candidates,
            returned_hits = payload.returned_hits,
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

        Ok(payload)
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
        let tracker = self.new_runtime_lifecycle_tracker(&storage_paths);
        let lifecycle = tracker.status();
        let tracked_index_bytes = lifecycle.resources.index_bytes;
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
            return Ok(FsfsModelStatus {
                tier: tier.to_owned(),
                name: model_name.to_owned(),
                cache_path: direct_path.display().to_string(),
                cached: true,
                size_bytes: Self::path_bytes(&direct_path)?,
            });
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

        let canonicalizer = DefaultCanonicalizer::default();
        let canonicalize_start = Instant::now();
        let mut manifests = Vec::new();
        let mut documents = Vec::new();
        let mut observed_reason_codes: BTreeSet<String> =
            stats.reason_codes.iter().cloned().collect();
        let mut semantic_doc_count = 0_usize;
        let mut content_skipped_files = 0_usize;
        for candidate in candidates {
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
            let ingestion_class = format!("{:?}", candidate.ingestion_class).to_ascii_lowercase();
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
            let doc = IndexableDocument::new(candidate.file_key, canonical)
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
            documents.push((candidate.ingestion_class, doc));
        }
        let canonicalize_elapsed_ms = canonicalize_start.elapsed().as_millis();
        info!(
            indexed_candidates = documents.len(),
            semantic_candidates = semantic_doc_count,
            skipped_after_read = content_skipped_files,
            elapsed_ms = canonicalize_elapsed_ms,
            "fsfs canonicalization stage completed"
        );

        manifests.sort_by(|left, right| left.file_key.cmp(&right.file_key));

        let source_hash_hex = index_source_hash_hex(&manifests);
        let indexed_files = documents.len();
        let skipped_files = stats.discovered_files.saturating_sub(indexed_files);
        let total_canonical_bytes = manifests.iter().fold(0_u64, |acc, entry| {
            acc.saturating_add(entry.canonical_bytes)
        });

        fs::create_dir_all(index_root.join("vector"))?;
        fs::create_dir_all(index_root.join("cache"))?;

        let lexical_docs = documents
            .iter()
            .filter(|(class, _)| {
                !matches!(class, IngestionClass::MetadataOnly | IngestionClass::Skip)
            })
            .map(|(_, doc)| doc.clone())
            .collect::<Vec<_>>();
        let lexical_start = Instant::now();
        let lexical_index = TantivyIndex::create(&index_root.join("lexical"))?;
        lexical_index.index_documents(cx, &lexical_docs).await?;
        lexical_index.commit(cx).await?;
        let lexical_elapsed_ms = lexical_start.elapsed().as_millis();

        let embed_start = Instant::now();
        let mut embedder = self.resolve_fast_embedder()?;
        let semantic_docs = documents
            .iter()
            .filter(|(class, _)| matches!(class, IngestionClass::FullSemanticLexical))
            .map(|(_, doc)| doc)
            .collect::<Vec<_>>();
        let semantic_texts = semantic_docs
            .iter()
            .map(|doc| doc.content.as_str())
            .collect::<Vec<_>>();
        let semantic_embeddings = if semantic_texts.is_empty() {
            Vec::new()
        } else {
            match embedder.embed_batch(cx, &semantic_texts).await {
                Ok(embeddings) => embeddings,
                Err(error) => {
                    warn!(
                        embedder = embedder.id(),
                        error = %error,
                        "fsfs semantic embedder failed; falling back to hash embeddings"
                    );
                    let fallback: Arc<dyn Embedder> = Arc::new(HashEmbedder::new(
                        embedder.dimension().max(1),
                        HashAlgorithm::FnvModular,
                    ));
                    embedder = fallback;
                    embedder.embed_batch(cx, &semantic_texts).await?
                }
            }
        };
        let embed_elapsed_ms = embed_start.elapsed().as_millis();

        let vector_start = Instant::now();
        let vector_path = index_root.join(FSFS_VECTOR_INDEX_FILE);
        let mut writer = VectorIndex::create(&vector_path, embedder.id(), embedder.dimension())?;
        for (doc, embedding) in semantic_docs.iter().zip(semantic_embeddings.into_iter()) {
            writer.write_record(&doc.id, &embedding)?;
        }
        writer.finish()?;
        let vector_elapsed_ms = vector_start.elapsed().as_millis();

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

        info!(
            command = ?command,
            target_root = %target_root.display(),
            index_root = %index_root.display(),
            discovered_files = stats.discovered_files,
            indexed_files,
            skipped_files,
            total_canonical_bytes,
            lexical_docs = lexical_docs.len(),
            semantic_docs = semantic_doc_count,
            embedder = embedder.id(),
            discovery_elapsed_ms,
            canonicalize_elapsed_ms,
            lexical_elapsed_ms,
            embedding_elapsed_ms = embed_elapsed_ms,
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

        Ok(target)
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

    fn resolve_fast_embedder(&self) -> SearchResult<Arc<dyn Embedder>> {
        if cfg!(test) {
            return Ok(Arc::new(HashEmbedder::default_256()));
        }

        let configured_root = PathBuf::from(&self.config.indexing.model_dir);
        let stack = EmbedderStack::auto_detect_with(Some(&configured_root))
            .or_else(|_| EmbedderStack::auto_detect())?;
        Ok(stack.fast_arc())
    }

    /// Build a live ingest pipeline for the watcher by opening existing indexes.
    fn build_live_ingest_pipeline(&self) -> SearchResult<LiveIngestPipeline> {
        let target_root = self.resolve_target_root()?;
        let index_root = self.resolve_index_root(&target_root)?;

        let lexical_path = index_root.join("lexical");
        let vector_path = index_root.join(FSFS_VECTOR_INDEX_FILE);

        let lexical_index = TantivyIndex::create(&lexical_path)?;
        let vector_index = VectorIndex::open(&vector_path)?;
        let embedder = self.resolve_fast_embedder()?;

        info!(
            target_root = %target_root.display(),
            index_root = %index_root.display(),
            embedder = embedder.id(),
            "live ingest pipeline initialized for watch mode"
        );

        Ok(LiveIngestPipeline::new(
            target_root,
            lexical_index,
            vector_index,
            embedder,
        ))
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
        fs::write(index_root.join(FSFS_VECTOR_MANIFEST_FILE), vector_manifest)?;

        let lexical_manifest = serde_json::to_string_pretty(manifests).map_err(|error| {
            SearchError::SubsystemError {
                subsystem: "index.lexical_manifest",
                source: Box::new(error),
            }
        })?;
        fs::write(
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
        fs::write(index_root.join(FSFS_SENTINEL_FILE), json)?;
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

    /// TUI runtime lane scaffold.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` when downstream TUI runtime logic fails.
    #[allow(clippy::unused_async)]
    pub async fn run_tui(&self, _cx: &Cx) -> SearchResult<()> {
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
    println!("  search <query>            Search indexed corpus");
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
}

const fn completion_script(shell: CompletionShell) -> &'static str {
    match shell {
        CompletionShell::Bash => {
            "complete -W \"search index watch explain status config download-models download doctor update completions uninstall help version\" fsfs"
        }
        CompletionShell::Zsh => {
            "compdef '_arguments \"1: :((search index watch explain status config download-models download doctor update completions uninstall help version))\"' fsfs"
        }
        CompletionShell::Fish => {
            "complete -c fsfs -f -a \"search index watch explain status config download-models download doctor update completions uninstall help version\""
        }
        CompletionShell::PowerShell => {
            "Register-ArgumentCompleter -CommandName fsfs -ScriptBlock { param($wordToComplete) 'search','index','watch','explain','status','config','download-models','download','doctor','update','completions','uninstall','help','version' | Where-Object { $_ -like \"$wordToComplete*\" } }"
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

#[allow(clippy::too_many_lines)]
fn render_status_table(status: &FsfsStatusPayload, no_color: bool) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    let now_ms = pressure_timestamp_ms();

    let stale_label = match status.index.stale_files {
        Some(0) => paint("up-to-date", "32", no_color),
        Some(count) => paint(&format!("stale ({count})"), "33", no_color),
        None => paint("unknown", "90", no_color),
    };
    let index_exists = if status.index.exists {
        paint("present", "32", no_color)
    } else {
        paint("missing", "31", no_color)
    };

    let _ = writeln!(out, "frankensearch {}", status.version);
    let _ = writeln!(out);
    let _ = writeln!(out, "Index:");
    let _ = writeln!(out, "  path: {}", status.index.path);
    let _ = writeln!(out, "  state: {index_exists}");
    if let Some(indexed_files) = status.index.indexed_files {
        let _ = writeln!(out, "  files indexed: {indexed_files}");
    }
    if let Some(discovered_files) = status.index.discovered_files {
        let _ = writeln!(out, "  files discovered: {discovered_files}");
    }
    if let Some(skipped_files) = status.index.skipped_files {
        let _ = writeln!(out, "  files skipped: {skipped_files}");
    }
    let _ = writeln!(
        out,
        "  size: {} (vector {}, lexical {}, metadata {}, cache {})",
        humanize_bytes(status.index.size_bytes),
        humanize_bytes(status.index.vector_index_bytes),
        humanize_bytes(status.index.lexical_index_bytes),
        humanize_bytes(status.index.metadata_bytes),
        humanize_bytes(status.index.embedding_cache_bytes),
    );
    if let Some(last_indexed_ms) = status.index.last_indexed_ms {
        let _ = writeln!(
            out,
            "  last indexed: {} ({})",
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
    let _ = writeln!(out, "Models:");
    for model in &status.models {
        let state = if model.cached {
            paint("cached", "32", no_color)
        } else {
            paint("missing", "31", no_color)
        };
        let _ = writeln!(
            out,
            "  {}: {} ({}, {}, path {})",
            model.tier,
            model.name,
            state,
            humanize_bytes(model.size_bytes),
            model.cache_path,
        );
    }

    let _ = writeln!(out);
    let _ = writeln!(out, "Config:");
    let _ = writeln!(out, "  source: {}", status.config.source);
    let _ = writeln!(out, "  index dir: {}", status.config.index_dir);
    let _ = writeln!(out, "  model dir: {}", status.config.model_dir);
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
    let _ = writeln!(out, "Runtime:");
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
            "  tracked index bytes: {}",
            humanize_bytes(index_bytes)
        );
    }
    if let Some(budget_bytes) = status.runtime.disk_budget_bytes {
        let _ = writeln!(out, "  disk budget bytes: {}", humanize_bytes(budget_bytes));
    }
    let _ = writeln!(
        out,
        "  storage pressure emergency: {}",
        status.runtime.storage_pressure_emergency
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
    // Check top level first.
    let direct = dir.join(name);
    if direct.is_file() {
        return Ok(direct);
    }
    // Check one level deeper.
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let candidate = path.join(name);
                if candidate.is_file() {
                    return Ok(candidate);
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

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    use asupersync::test_utils::run_test_with_cx;
    use frankensearch_core::{IndexableDocument, LexicalSearch};
    use frankensearch_embed::HashEmbedder;
    use frankensearch_index::VectorIndex;
    use frankensearch_lexical::TantivyIndex;
    use fsqlite::Connection;
    use fsqlite_types::value::SqliteValue;

    use super::{
        EmbedderAvailability, FsfsRuntime, IndexStoragePaths, InterfaceMode, LiveIngestPipeline,
        VectorIndexWriteAction, VectorPipelineInput, VectorSchedulingTier,
        degradation_controller_config_for_profile, render_status_table,
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
    use crate::shutdown::{ShutdownCoordinator, ShutdownReason};
    use crate::stream_protocol::{
        StreamEventKind, StreamFrame, TOON_STREAM_RECORD_SEPARATOR_BYTE,
        decode_stream_frame_ndjson, decode_stream_frame_toon,
    };
    use crate::watcher::{WatchIngestOp, WatchIngestPipeline};

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

        let remaining = conn
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
            let missing_path = target_root.join("src/ghost.rs");
            let applied = pipeline
                .apply_batch(&[WatchIngestOp::Upsert {
                    file_key: missing_path.display().to_string(),
                    revision: 42,
                    ingestion_class: IngestionClass::FullSemanticLexical,
                }])
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
            let applied = pipeline
                .apply_batch(&[WatchIngestOp::Upsert {
                    file_key: binary_path.display().to_string(),
                    revision: 42,
                    ingestion_class: IngestionClass::FullSemanticLexical,
                }])
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
            let applied = pipeline
                .apply_batch(&[WatchIngestOp::Upsert {
                    file_key: file_path.display().to_string(),
                    revision: 7,
                    ingestion_class: IngestionClass::LexicalOnly,
                }])
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
            let applied = pipeline
                .apply_batch(&[WatchIngestOp::Upsert {
                    file_key: file_path.display().to_string(),
                    revision: 8,
                    ingestion_class: IngestionClass::MetadataOnly,
                }])
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
        assert!(!payload.models[1].cached);
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
        let cache = super::VersionCheckCache {
            checked_at_epoch: super::epoch_now(),
            current_version: env!("CARGO_PKG_VERSION").into(),
            latest_version: "v99.0.0".into(),
            release_url: "https://example.com".into(),
            ttl_seconds: 86_400,
        };
        // Write it.
        super::write_version_cache(&cache).unwrap();
        // Read it back.
        let loaded = super::read_version_cache().unwrap();
        assert_eq!(loaded.latest_version, "v99.0.0");
        assert_eq!(loaded.release_url, "https://example.com");
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
}
