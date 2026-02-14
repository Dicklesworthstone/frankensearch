use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use asupersync::Cx;
use dirs::home_dir;
use frankensearch_core::{
    Canonicalizer, DefaultCanonicalizer, Embedder, IndexableDocument, LexicalSearch, SearchError,
    SearchResult,
};
use frankensearch_embed::EmbedderStack;
use frankensearch_index::VectorIndex;
use frankensearch_lexical::TantivyIndex;
use ignore::WalkBuilder;
use serde::{Deserialize, Serialize};
use sysinfo::Disks;
use tracing::{info, warn};

use crate::adapters::cli::{CliCommand, CliInput, CompletionShell, OutputFormat};
use crate::adapters::format_emitter::{emit_envelope, meta_for_format};
use crate::adapters::tui::FsfsTuiShellModel;
use crate::config::{
    DegradationOverrideMode, DiscoveryCandidate, DiscoveryDecision, DiscoveryScopeDecision,
    FsfsConfig, IngestionClass, PressureProfile, RootDiscoveryDecision,
    default_project_config_file_path, default_user_config_file_path,
};
use crate::lifecycle::{
    DiskBudgetAction, DiskBudgetSnapshot, DiskBudgetStage, IndexStorageBreakdown, LifecycleTracker,
    ResourceLimits, ResourceUsage, WatchdogConfig,
};
use crate::output_schema::OutputEnvelope;
use crate::pressure::{
    DegradationControllerConfig, DegradationSignal, DegradationStateMachine, DegradationTransition,
    HostPressureCollector, PressureController, PressureControllerConfig, PressureSignal,
    PressureState, PressureTransition,
};
use crate::shutdown::{ShutdownCoordinator, ShutdownReason};
use crate::watcher::{FsWatcher, NoopWatchIngestPipeline};

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
const FSFS_SENTINEL_FILE: &str = "index_sentinel.json";
const FSFS_VECTOR_MANIFEST_FILE: &str = "vector/index_manifest.json";
const FSFS_LEXICAL_MANIFEST_FILE: &str = "lexical/index_manifest.json";
const FSFS_VECTOR_INDEX_FILE: &str = "vector/index.fsvi";

#[derive(Debug, Clone)]
struct IndexCandidate {
    file_path: PathBuf,
    file_key: String,
    modified_ms: u64,
    ingestion_class: IngestionClass,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct IndexDiscoveryStats {
    discovered_files: usize,
    skipped_files: usize,
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
    total_canonical_bytes: u64,
    source_hash_hex: String,
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
    tracked_index_bytes: Option<u64>,
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
            CliCommand::Update => {
                self.run_update_command();
                Ok(())
            }
            CliCommand::Uninstall => {
                Self::run_uninstall_command();
                Ok(())
            }
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

    fn run_update_command(&self) {
        if self.cli_input.update_check_only {
            println!(
                "fsfs {}: update check complete (updater backend not configured)",
                env!("CARGO_PKG_VERSION")
            );
        } else {
            println!(
                "fsfs {}: updater backend not configured; no changes applied",
                env!("CARGO_PKG_VERSION")
            );
        }
    }

    fn run_uninstall_command() {
        println!(
            "Uninstall is manual in this build. Remove the fsfs binary and any index/cache directories explicitly."
        );
    }

    async fn run_cli_scaffold(&self, cx: &Cx, command: CliCommand) -> SearchResult<()> {
        self.validate_command_inputs(command)?;
        if command == CliCommand::Status {
            self.run_status_command()?;
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

    fn run_status_command(&self) -> SearchResult<()> {
        if self.cli_input.format == OutputFormat::Csv {
            return Err(SearchError::InvalidConfig {
                field: "cli.format".to_owned(),
                value: self.cli_input.format.to_string(),
                reason: "status command does not support csv output".to_owned(),
            });
        }

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
        let disk_budget = self.evaluate_storage_disk_budget(&tracker, &storage_paths)?;
        let lifecycle = tracker.status();

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
            tracked_index_bytes: lifecycle.resources.index_bytes,
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

    async fn run_one_shot_index_scaffold(&self, cx: &Cx, command: CliCommand) -> SearchResult<()> {
        let total_start = Instant::now();
        let target_root = self.resolve_target_root()?;
        let index_root = self.resolve_index_root(&target_root);

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
        let mut semantic_doc_count = 0_usize;
        let mut content_skipped_files = 0_usize;
        for candidate in candidates {
            let bytes = match fs::read(&candidate.file_path) {
                Ok(bytes) => bytes,
                Err(error) if is_ignorable_index_walk_error(&error) => {
                    content_skipped_files = content_skipped_files.saturating_add(1);
                    continue;
                }
                Err(error) => return Err(error.into()),
            };
            if is_probably_binary(&bytes) {
                content_skipped_files = content_skipped_files.saturating_add(1);
                continue;
            }
            let raw_text = String::from_utf8_lossy(&bytes);
            let canonical = canonicalizer.canonicalize(&raw_text);
            if canonical.trim().is_empty() {
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
        let embedder = self.resolve_fast_embedder()?;
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
            embedder.embed_batch(cx, &semantic_texts).await?
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

    fn resolve_index_root(&self, target_root: &Path) -> PathBuf {
        let configured = PathBuf::from(&self.config.storage.index_dir);
        if configured.is_absolute() {
            configured
        } else {
            target_root.join(configured)
        }
    }

    fn resolve_fast_embedder(&self) -> SearchResult<Arc<dyn Embedder>> {
        let configured_root = PathBuf::from(&self.config.indexing.model_dir);
        let stack = EmbedderStack::auto_detect_with(Some(&configured_root))
            .or_else(|_| EmbedderStack::auto_detect())?;
        Ok(stack.fast_arc())
    }

    fn collect_index_candidates(
        &self,
        target_root: &Path,
        index_root: &Path,
        output: &mut Vec<IndexCandidate>,
    ) -> SearchResult<IndexDiscoveryStats> {
        let mut discovered_files = 0_usize;
        let mut skipped_files = 0_usize;

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

            let Some(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir() {
                continue;
            }
            if !file_type.is_file() && !entry.path_is_symlink() {
                continue;
            }

            let entry_path = entry.path().to_path_buf();
            if entry_path.starts_with(index_root) {
                skipped_files = skipped_files.saturating_add(1);
                continue;
            }
            if entry.path_is_symlink() && !self.config.discovery.follow_symlinks {
                skipped_files = skipped_files.saturating_add(1);
                continue;
            }

            let metadata = match fs::metadata(&entry_path) {
                Ok(metadata) => metadata,
                Err(error) if is_ignorable_index_walk_error(&error) => continue,
                Err(error) => return Err(error.into()),
            };

            discovered_files = discovered_files.saturating_add(1);
            let candidate = DiscoveryCandidate::new(&entry_path, metadata.len())
                .with_symlink(entry.path_is_symlink());
            let decision = self.config.discovery.evaluate_candidate(&candidate);
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

        if self.config.indexing.watch_mode {
            let watcher = FsWatcher::from_config(&self.config, Arc::new(NoopWatchIngestPipeline));
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
                            Ok(Some(snapshot)) => {
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
                            Ok(None) => {}
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
    println!("  download [model]          Download embedding models");
    println!("  doctor                    Run local health checks");
    println!("  update [--check]          Check/apply binary updates");
    println!("  completions <shell>       Generate shell completions");
    println!("  uninstall                 Print uninstall instructions");
    println!("  help                      Show this help");
    println!("  version                   Show version");
    println!();
    println!("Global flags: --verbose/-v --quiet/-q --no-color --format --config");
}

const fn completion_script(shell: CompletionShell) -> &'static str {
    match shell {
        CompletionShell::Bash => {
            "complete -W \"search index watch explain status config download doctor update completions uninstall help version\" fsfs"
        }
        CompletionShell::Zsh => {
            "compdef '_arguments \"1: :((search index watch explain status config download doctor update completions uninstall help version))\"' fsfs"
        }
        CompletionShell::Fish => {
            "complete -c fsfs -f -a \"search index watch explain status config download doctor update completions uninstall help version\""
        }
        CompletionShell::PowerShell => {
            "Register-ArgumentCompleter -CommandName fsfs -ScriptBlock { param($wordToComplete) 'search','index','watch','explain','status','config','download','doctor','update','completions','uninstall','help','version' | Where-Object { $_ -like \"$wordToComplete*\" } }"
        }
    }
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

    use super::{
        EmbedderAvailability, FsfsRuntime, IndexStoragePaths, InterfaceMode,
        VectorIndexWriteAction, VectorPipelineInput, VectorSchedulingTier,
        degradation_controller_config_for_profile,
    };
    use crate::adapters::cli::{CliCommand, CliInput, CompletionShell};
    use crate::config::{
        DegradationOverrideMode, DiscoveryCandidate, DiscoveryScopeDecision, FsfsConfig,
        IngestionClass, PressureProfile,
    };
    use crate::lifecycle::{
        DiskBudgetAction, DiskBudgetStage, LifecycleTracker, ResourceLimits, WatchdogConfig,
    };
    use crate::pressure::{
        DegradationStage, HostPressureCollector, PressureSignal, PressureState, QueryCapabilityMode,
    };
    use crate::shutdown::{ShutdownCoordinator, ShutdownReason};

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
            let mut config = FsfsConfig::default();
            config.indexing.watch_mode = true;
            let runtime = FsfsRuntime::new(config);
            let coordinator: Arc<ShutdownCoordinator> = Arc::new(ShutdownCoordinator::new());

            let trigger: Arc<ShutdownCoordinator> = Arc::clone(&coordinator);
            let worker = thread::spawn(move || {
                thread::sleep(Duration::from_millis(30));
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
        });
    }

    #[test]
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
    }
}
