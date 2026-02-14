use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use asupersync::Cx;
use frankensearch_core::SearchResult;
use tracing::{info, warn};

use crate::config::{
    DegradationOverrideMode, DiscoveryCandidate, DiscoveryDecision, FsfsConfig, IngestionClass,
    PressureProfile, RootDiscoveryDecision,
};
use crate::lifecycle::{
    DiskBudgetSnapshot, IndexStorageBreakdown, LifecycleTracker, ResourceUsage,
};
use crate::pressure::{
    DegradationControllerConfig, DegradationSignal, DegradationStateMachine, DegradationTransition,
    HostPressureCollector, PressureController, PressureControllerConfig, PressureSignal,
    PressureTransition,
};
use crate::shutdown::{ShutdownCoordinator, ShutdownReason};

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

/// Shared runtime entrypoint used by interface adapters.
#[derive(Debug, Clone)]
pub struct FsfsRuntime {
    config: FsfsConfig,
}

impl FsfsRuntime {
    #[must_use]
    pub const fn new(config: FsfsConfig) -> Self {
        Self { config }
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

    /// CLI runtime lane scaffold.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` when downstream CLI runtime logic fails.
    pub async fn run_cli(&self, _cx: &Cx) -> SearchResult<()> {
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
            profile = ?self.config.pressure.profile,
            total_roots = root_plan.len(),
            accepted_roots,
            rejected_roots = root_plan.len().saturating_sub(accepted_roots),
            "fsfs cli runtime scaffold invoked"
        );
        Ok(())
    }

    /// TUI runtime lane scaffold.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` when downstream TUI runtime logic fails.
    pub async fn run_tui(&self, _cx: &Cx) -> SearchResult<()> {
        std::future::ready(()).await;
        info!(theme = ?self.config.tui.theme, "fsfs tui runtime scaffold invoked");
        Ok(())
    }

    async fn run_cli_with_shutdown(
        &self,
        cx: &Cx,
        shutdown: &ShutdownCoordinator,
    ) -> SearchResult<()> {
        self.run_cli(cx).await?;

        if self.config.indexing.watch_mode {
            let reason = self.await_shutdown(cx, shutdown).await;
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
        let reason = self.await_shutdown(cx, shutdown).await;
        self.finalize_shutdown(cx, reason).await
    }

    async fn await_shutdown(&self, cx: &Cx, shutdown: &ShutdownCoordinator) -> ShutdownReason {
        loop {
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

fn pressure_timestamp_ms() -> u64 {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    u64::try_from(millis).unwrap_or(u64::MAX)
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
    use std::path::Path;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    use asupersync::test_utils::run_test_with_cx;

    use super::{
        EmbedderAvailability, FsfsRuntime, IndexStoragePaths, InterfaceMode,
        VectorIndexWriteAction, VectorPipelineInput, VectorSchedulingTier,
        degradation_controller_config_for_profile,
    };
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
}
