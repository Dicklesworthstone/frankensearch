//! Daemon lifecycle, health checks, PID file management, and self-supervision.
//!
//! Provides types and logic for running fsfs as a long-lived background process.
//! The lifecycle model covers:
//!
//! 1. **PID file management** — Prevents duplicate daemon instances using
//!    file-based locking with stale PID detection.
//! 2. **Subsystem health tracking** — Each pipeline stage reports its health
//!    via [`SubsystemHealth`]; the daemon aggregates into [`DaemonStatus`].
//! 3. **Watchdog supervision** — Detects crashed subsystems and manages
//!    restart attempts with exponential backoff.
//! 4. **Resource limits** — Configurable caps on threads, memory, and open files.
//! 5. **Status reporting** — Machine-readable status for CLI `fsfs status` command.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::config::PressureConfig;
use crate::evidence::FsfsReasonCode;

// ─── Daemon Phase ───────────────────────────────────────────────────────────

/// Lifecycle phase of the fsfs daemon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DaemonPhase {
    /// Process starting, acquiring lock file, validating config.
    Initializing,
    /// Performing index integrity checks and loading existing indices.
    StartingUp,
    /// All subsystems running normally.
    Running,
    /// One or more subsystems degraded but service still available.
    Degraded,
    /// Graceful shutdown in progress.
    ShuttingDown,
    /// Process has exited.
    Stopped,
}

impl DaemonPhase {
    /// Whether the daemon is accepting work in this phase.
    #[must_use]
    pub const fn is_accepting_work(&self) -> bool {
        matches!(self, Self::Running | Self::Degraded)
    }

    /// Whether the daemon is in a terminal state.
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        matches!(self, Self::Stopped)
    }
}

impl std::fmt::Display for DaemonPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Initializing => "initializing",
            Self::StartingUp => "starting_up",
            Self::Running => "running",
            Self::Degraded => "degraded",
            Self::ShuttingDown => "shutting_down",
            Self::Stopped => "stopped",
        };
        f.write_str(s)
    }
}

// ─── Subsystem Health ───────────────────────────────────────────────────────

/// Known subsystem identifiers within the fsfs daemon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubsystemId {
    /// File system crawler/scanner.
    Crawler,
    /// Fast-tier embedding worker.
    EmbedFast,
    /// Quality-tier embedding worker.
    EmbedQuality,
    /// Lexical indexer.
    LexicalIndexer,
    /// Query server (search requests).
    QueryServer,
    /// Index refresh/cache update worker.
    RefreshWorker,
}

impl SubsystemId {
    /// Human-readable name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Crawler => "crawler",
            Self::EmbedFast => "embed_fast",
            Self::EmbedQuality => "embed_quality",
            Self::LexicalIndexer => "lexical_indexer",
            Self::QueryServer => "query_server",
            Self::RefreshWorker => "refresh_worker",
        }
    }
}

impl std::fmt::Display for SubsystemId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Health status of a single subsystem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HealthStatus {
    /// Not yet started.
    Pending,
    /// Running normally.
    Healthy,
    /// Running but with issues (e.g., high error rate).
    Degraded,
    /// Crashed or stopped unexpectedly.
    Failed,
    /// Stopped intentionally (during shutdown).
    Stopped,
}

impl HealthStatus {
    /// Whether this status counts as "alive" (not crashed/stopped).
    #[must_use]
    pub const fn is_alive(&self) -> bool {
        matches!(self, Self::Pending | Self::Healthy | Self::Degraded)
    }
}

/// Detailed health report for a subsystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubsystemHealth {
    /// Which subsystem.
    pub id: SubsystemId,
    /// Current health status.
    pub status: HealthStatus,
    /// Number of times this subsystem has been restarted.
    pub restart_count: u32,
    /// Last error message, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
    /// When the subsystem last reported healthy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_healthy_at: Option<String>,
    /// When the subsystem entered its current status.
    pub status_since: String,
}

// ─── Daemon Status ──────────────────────────────────────────────────────────

/// Complete daemon status snapshot for reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonStatus {
    /// Current lifecycle phase.
    pub phase: DaemonPhase,
    /// Process ID.
    pub pid: u32,
    /// When the daemon started.
    pub started_at: String,
    /// How long the daemon has been running.
    pub uptime_secs: u64,
    /// Per-subsystem health reports.
    pub subsystems: Vec<SubsystemHealth>,
    /// Total errors across all subsystems.
    pub total_errors: u64,
    /// Total panics caught and recovered.
    pub total_panics_recovered: u64,
    /// Resource usage snapshot.
    pub resources: ResourceUsage,
    /// Disk-budget state derived from current resource usage and configured limits.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disk_budget: Option<DiskBudgetSnapshot>,
    /// Stable reason code for the current disk-budget state for JSON/TOON consumers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disk_budget_reason_code: Option<String>,
}

/// Resource usage snapshot.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Resident set size in bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rss_bytes: Option<u64>,
    /// Number of active threads.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thread_count: Option<u32>,
    /// Number of open file descriptors.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub open_fds: Option<u32>,
    /// Current index/storage footprint in bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index_bytes: Option<u64>,
    /// Vector index footprint (FSVI/WAL) in bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_index_bytes: Option<u64>,
    /// Lexical index footprint in bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lexical_index_bytes: Option<u64>,
    /// Catalog/database footprint in bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub catalog_bytes: Option<u64>,
    /// Embedding cache footprint in bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_cache_bytes: Option<u64>,
}

impl ResourceUsage {
    /// Build a resource usage snapshot from storage-domain byte totals.
    #[must_use]
    pub const fn from_index_storage(storage: IndexStorageBreakdown) -> Self {
        Self {
            rss_bytes: None,
            thread_count: None,
            open_fds: None,
            index_bytes: Some(storage.total_bytes()),
            vector_index_bytes: Some(storage.vector_index_bytes),
            lexical_index_bytes: Some(storage.lexical_index_bytes),
            catalog_bytes: Some(storage.catalog_bytes),
            embedding_cache_bytes: Some(storage.embedding_cache_bytes),
        }
    }

    /// Effective index footprint used for budget and limit checks.
    ///
    /// Prefers explicit `index_bytes`, then falls back to summing domain fields.
    #[must_use]
    pub fn effective_index_bytes(&self) -> Option<u64> {
        self.index_bytes.or_else(|| {
            let mut had_component = false;
            let mut total = 0_u64;
            for bytes in [
                self.vector_index_bytes,
                self.lexical_index_bytes,
                self.catalog_bytes,
                self.embedding_cache_bytes,
            ]
            .into_iter()
            .flatten()
            {
                had_component = true;
                total = total.saturating_add(bytes);
            }
            had_component.then_some(total)
        })
    }
}

/// Cross-domain index/storage footprint used for budget accounting.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexStorageBreakdown {
    /// Vector index footprint (FSVI/WAL) in bytes.
    pub vector_index_bytes: u64,
    /// Lexical index footprint in bytes.
    pub lexical_index_bytes: u64,
    /// Catalog/database footprint in bytes.
    pub catalog_bytes: u64,
    /// Embedding cache footprint in bytes.
    pub embedding_cache_bytes: u64,
}

impl IndexStorageBreakdown {
    /// Total storage footprint across all index domains.
    #[must_use]
    pub const fn total_bytes(&self) -> u64 {
        self.vector_index_bytes
            .saturating_add(self.lexical_index_bytes)
            .saturating_add(self.catalog_bytes)
            .saturating_add(self.embedding_cache_bytes)
    }
}

/// Disk-budget stage derived from index footprint vs configured budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiskBudgetStage {
    Normal,
    NearLimit,
    OverLimit,
    Critical,
}

/// Recommended controller action for the current disk-budget stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiskBudgetAction {
    Continue,
    ThrottleIngest,
    EvictLowUtility,
    PauseWrites,
}

/// Snapshot of disk-budget state for scheduler/UX/evidence consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiskBudgetSnapshot {
    pub budget_bytes: u64,
    pub used_bytes: u64,
    /// Budget utilization in per-mille (1000 == 100% of budget).
    pub usage_per_mille: u16,
    pub stage: DiskBudgetStage,
    pub action: DiskBudgetAction,
    #[serde(skip)]
    pub reason_code: &'static str,
}

/// Policy thresholds for staged disk-budget responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiskBudgetPolicy {
    /// Enter near-limit mode at this per-mille of budget.
    pub near_limit_per_mille: u16,
    /// Enter critical mode at this per-mille of budget.
    pub critical_per_mille: u16,
}

impl Default for DiskBudgetPolicy {
    fn default() -> Self {
        Self {
            near_limit_per_mille: 850,
            critical_per_mille: 1100,
        }
    }
}

impl DiskBudgetPolicy {
    /// Derive a staged disk-budget response for one footprint reading.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn evaluate(&self, used_bytes: u64, budget_bytes: u64) -> DiskBudgetSnapshot {
        if budget_bytes == 0 {
            // Treat zero budget as maximally over-limit.
            return DiskBudgetSnapshot {
                used_bytes,
                budget_bytes,
                usage_per_mille: u16::MAX,
                stage: DiskBudgetStage::Critical,
                action: DiskBudgetAction::PauseWrites,
                reason_code: FsfsReasonCode::DEGRADE_DISK_CRITICAL,
            };
        }
        let usage_per_mille_u64 = used_bytes
            .saturating_mul(1000)
            .saturating_add(budget_bytes.saturating_sub(1))
            / budget_bytes;
        let usage_per_mille = usage_per_mille_u64.min(u64::from(u16::MAX)) as u16;

        let (stage, action, reason_code) = if usage_per_mille >= self.critical_per_mille {
            (
                DiskBudgetStage::Critical,
                DiskBudgetAction::PauseWrites,
                FsfsReasonCode::DEGRADE_DISK_CRITICAL,
            )
        } else if used_bytes > budget_bytes {
            (
                DiskBudgetStage::OverLimit,
                DiskBudgetAction::EvictLowUtility,
                FsfsReasonCode::DEGRADE_DISK_OVER_BUDGET,
            )
        } else if usage_per_mille >= self.near_limit_per_mille {
            (
                DiskBudgetStage::NearLimit,
                DiskBudgetAction::ThrottleIngest,
                FsfsReasonCode::DEGRADE_DISK_NEAR_BUDGET,
            )
        } else {
            (
                DiskBudgetStage::Normal,
                DiskBudgetAction::Continue,
                FsfsReasonCode::DEGRADE_DISK_WITHIN_BUDGET,
            )
        };

        DiskBudgetSnapshot {
            budget_bytes,
            used_bytes,
            usage_per_mille,
            stage,
            action,
            reason_code,
        }
    }
}

// ─── Index Footprint Advisor ────────────────────────────────────────────────

pub const INDEX_FOOTPRINT_ADVISOR_SCHEMA_VERSION: u32 = 1;
pub const INDEX_FOOTPRINT_ADVISOR_CONTRACT_KIND: &str = "fsfs_index_footprint_advisor_contract";
pub const INDEX_FOOTPRINT_ADVISOR_REPORT_KIND: &str = "fsfs_index_footprint_advisor_report";
pub const INDEX_FOOTPRINT_ADVISOR_POLICY_VERSION: &str = "fsfs-index-footprint-advisor-policy-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexFootprintDomain {
    VectorIndex,
    LexicalIndex,
    Metadata,
    ModelCache,
    Artifact,
}

impl IndexFootprintDomain {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::VectorIndex => "vector_index",
            Self::LexicalIndex => "lexical_index",
            Self::Metadata => "metadata",
            Self::ModelCache => "model_cache",
            Self::Artifact => "artifact",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexFootprintScenario {
    Small,
    Fragmented,
    Oversized,
}

impl IndexFootprintScenario {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Small => "small",
            Self::Fragmented => "fragmented",
            Self::Oversized => "oversized",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexFootprintAdvisorAction {
    Compaction,
    Rebuild,
    Retention,
    FeatureAdjustment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexFootprintAdvisorRisk {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexFootprintAdvisorPolicy {
    pub small_index_max_bytes: u64,
    pub fragmentation_threshold_per_mille: u16,
    pub oversize_threshold_per_mille: u16,
    pub dominant_domain_threshold_per_mille: u16,
    pub minimum_projected_savings_bytes: u64,
}

impl Default for IndexFootprintAdvisorPolicy {
    fn default() -> Self {
        Self {
            small_index_max_bytes: 128 * 1024 * 1024,
            fragmentation_threshold_per_mille: 200,
            oversize_threshold_per_mille: 950,
            dominant_domain_threshold_per_mille: 500,
            minimum_projected_savings_bytes: 1024 * 1024,
        }
    }
}

impl IndexFootprintAdvisorPolicy {
    #[must_use]
    pub fn classify(
        &self,
        total_bytes: u64,
        budget_bytes: Option<u64>,
        measurements: &[IndexFootprintDomainFootprint],
    ) -> IndexFootprintScenario {
        if self.is_oversized(total_bytes, budget_bytes, measurements) {
            return IndexFootprintScenario::Oversized;
        }
        if self.is_fragmented(measurements) {
            return IndexFootprintScenario::Fragmented;
        }
        IndexFootprintScenario::Small
    }

    #[must_use]
    pub fn action_for(
        &self,
        scenario: IndexFootprintScenario,
        footprint: &IndexFootprintDomainFootprint,
        total_bytes: u64,
    ) -> IndexFootprintAdvisorAction {
        match scenario {
            IndexFootprintScenario::Small => IndexFootprintAdvisorAction::Retention,
            IndexFootprintScenario::Fragmented => match footprint.domain {
                IndexFootprintDomain::VectorIndex | IndexFootprintDomain::Artifact
                    if self.is_reclaimable(footprint) =>
                {
                    IndexFootprintAdvisorAction::Compaction
                }
                IndexFootprintDomain::LexicalIndex | IndexFootprintDomain::Metadata
                    if self.is_reclaimable(footprint) =>
                {
                    IndexFootprintAdvisorAction::Rebuild
                }
                _ => IndexFootprintAdvisorAction::Retention,
            },
            IndexFootprintScenario::Oversized => match footprint.domain {
                IndexFootprintDomain::ModelCache if self.is_dominant(footprint, total_bytes) => {
                    IndexFootprintAdvisorAction::FeatureAdjustment
                }
                IndexFootprintDomain::Artifact if footprint.reclaimable_bytes > 0 => {
                    IndexFootprintAdvisorAction::Retention
                }
                IndexFootprintDomain::VectorIndex if self.is_reclaimable(footprint) => {
                    IndexFootprintAdvisorAction::Compaction
                }
                IndexFootprintDomain::LexicalIndex | IndexFootprintDomain::Metadata
                    if self.is_reclaimable(footprint) =>
                {
                    IndexFootprintAdvisorAction::Rebuild
                }
                _ => IndexFootprintAdvisorAction::Retention,
            },
        }
    }

    #[must_use]
    pub fn build_recommendation(
        &self,
        scenario: IndexFootprintScenario,
        footprint: &IndexFootprintDomainFootprint,
        total_bytes: u64,
    ) -> IndexFootprintRecommendation {
        let action = self.action_for(scenario, footprint, total_bytes);
        let projected_savings_bytes = match action {
            IndexFootprintAdvisorAction::Retention if scenario == IndexFootprintScenario::Small => {
                0
            }
            IndexFootprintAdvisorAction::Retention => footprint.reclaimable_bytes,
            IndexFootprintAdvisorAction::Compaction
            | IndexFootprintAdvisorAction::Rebuild
            | IndexFootprintAdvisorAction::FeatureAdjustment => footprint.reclaimable_bytes,
        };
        IndexFootprintRecommendation {
            domain: footprint.domain,
            action,
            reason_code: recommendation_reason_code(scenario, action),
            risk: recommendation_risk(scenario, action),
            measured_bytes: footprint.bytes,
            projected_savings_bytes,
            replay_command: replay_command(scenario, footprint.domain),
            operator_command: operator_command(action, footprint.domain),
            rationale: recommendation_rationale(scenario, action, footprint.domain),
        }
    }

    #[must_use]
    fn is_fragmented(&self, measurements: &[IndexFootprintDomainFootprint]) -> bool {
        measurements.iter().any(|measurement| {
            measurement.fragmentation_per_mille >= self.fragmentation_threshold_per_mille
                || measurement.reclaimable_bytes >= self.minimum_projected_savings_bytes
        })
    }

    #[must_use]
    fn is_oversized(
        &self,
        total_bytes: u64,
        budget_bytes: Option<u64>,
        measurements: &[IndexFootprintDomainFootprint],
    ) -> bool {
        if let Some(budget_bytes) = budget_bytes {
            if budget_bytes > 0
                && total_bytes.saturating_mul(1000)
                    >= budget_bytes.saturating_mul(u64::from(self.oversize_threshold_per_mille))
            {
                return true;
            }
        }
        total_bytes > self.small_index_max_bytes
            && measurements
                .iter()
                .any(|measurement| self.is_dominant(measurement, total_bytes))
    }

    #[must_use]
    fn is_reclaimable(&self, footprint: &IndexFootprintDomainFootprint) -> bool {
        footprint.fragmentation_per_mille >= self.fragmentation_threshold_per_mille
            || footprint.reclaimable_bytes >= self.minimum_projected_savings_bytes
    }

    #[must_use]
    fn is_dominant(&self, footprint: &IndexFootprintDomainFootprint, total_bytes: u64) -> bool {
        total_bytes > 0
            && footprint.bytes.saturating_mul(1000)
                >= total_bytes.saturating_mul(u64::from(self.dominant_domain_threshold_per_mille))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexFootprintAdvisorContractDefinition {
    pub kind: String,
    pub v: u32,
    pub dry_run_only: bool,
    pub automatic_deletion_allowed: bool,
    pub policy_version: String,
    pub required_domains: Vec<IndexFootprintDomain>,
    pub required_actions: Vec<IndexFootprintAdvisorAction>,
    pub policy: IndexFootprintAdvisorPolicy,
    pub replay_command_template: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexFootprintDomainFootprint {
    pub domain: IndexFootprintDomain,
    pub bytes: u64,
    pub reclaimable_bytes: u64,
    pub fragmentation_per_mille: u16,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexFootprintRecommendation {
    pub domain: IndexFootprintDomain,
    pub action: IndexFootprintAdvisorAction,
    pub reason_code: String,
    pub risk: IndexFootprintAdvisorRisk,
    pub measured_bytes: u64,
    pub projected_savings_bytes: u64,
    pub replay_command: String,
    pub operator_command: String,
    pub rationale: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexFootprintAdvisorSummary {
    pub recommendation_count: usize,
    pub projected_savings_bytes: u64,
    pub highest_risk: IndexFootprintAdvisorRisk,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexFootprintAdvisorReport {
    pub kind: String,
    pub v: u32,
    pub generated_at: String,
    pub surface: String,
    pub policy_version: String,
    pub scenario: IndexFootprintScenario,
    pub dry_run: bool,
    pub automatic_deletion_allowed: bool,
    pub budget_bytes: Option<u64>,
    pub total_bytes: u64,
    pub measurements: Vec<IndexFootprintDomainFootprint>,
    pub recommendations: Vec<IndexFootprintRecommendation>,
    pub summary: IndexFootprintAdvisorSummary,
}

#[must_use]
pub fn index_footprint_advisor_contract_definition() -> IndexFootprintAdvisorContractDefinition {
    IndexFootprintAdvisorContractDefinition {
        kind: INDEX_FOOTPRINT_ADVISOR_CONTRACT_KIND.to_owned(),
        v: INDEX_FOOTPRINT_ADVISOR_SCHEMA_VERSION,
        dry_run_only: true,
        automatic_deletion_allowed: false,
        policy_version: INDEX_FOOTPRINT_ADVISOR_POLICY_VERSION.to_owned(),
        required_domains: required_index_footprint_domains().to_vec(),
        required_actions: vec![
            IndexFootprintAdvisorAction::Compaction,
            IndexFootprintAdvisorAction::Rebuild,
            IndexFootprintAdvisorAction::Retention,
            IndexFootprintAdvisorAction::FeatureAdjustment,
        ],
        policy: IndexFootprintAdvisorPolicy::default(),
        replay_command_template: "FSFS_INDEX_FOOTPRINT_FIXTURE={scenario} FSFS_INDEX_FOOTPRINT_DOMAIN={domain} cargo test -p frankensearch-fsfs index_footprint_advisor_policy_suite -- --nocapture".to_owned(),
    }
}

#[must_use]
pub fn index_footprint_advisor_small_fixture() -> IndexFootprintAdvisorReport {
    index_footprint_advisor_report_fixture(
        "2026-05-08T00:00:00Z",
        "fsfs-status-small-fixture",
        Some(512 * 1024 * 1024),
        vec![
            footprint(
                IndexFootprintDomain::VectorIndex,
                8 * 1024 * 1024,
                0,
                8,
                "fsvi",
            ),
            footprint(
                IndexFootprintDomain::LexicalIndex,
                6 * 1024 * 1024,
                0,
                7,
                "lexical",
            ),
            footprint(
                IndexFootprintDomain::Metadata,
                2 * 1024 * 1024,
                0,
                4,
                "sqlite",
            ),
            footprint(
                IndexFootprintDomain::ModelCache,
                18 * 1024 * 1024,
                0,
                0,
                "model-cache",
            ),
            footprint(
                IndexFootprintDomain::Artifact,
                1024 * 1024,
                0,
                0,
                "artifact-manifest",
            ),
        ],
    )
}

#[must_use]
pub fn index_footprint_advisor_fragmented_fixture() -> IndexFootprintAdvisorReport {
    index_footprint_advisor_report_fixture(
        "2026-05-08T00:00:00Z",
        "fsfs-status-fragmented-fixture",
        Some(2 * 1024 * 1024 * 1024),
        vec![
            footprint(
                IndexFootprintDomain::VectorIndex,
                420 * 1024 * 1024,
                96 * 1024 * 1024,
                260,
                "fsvi+wal",
            ),
            footprint(
                IndexFootprintDomain::LexicalIndex,
                280 * 1024 * 1024,
                72 * 1024 * 1024,
                240,
                "lexical",
            ),
            footprint(
                IndexFootprintDomain::Metadata,
                80 * 1024 * 1024,
                16 * 1024 * 1024,
                205,
                "sqlite",
            ),
            footprint(
                IndexFootprintDomain::ModelCache,
                220 * 1024 * 1024,
                0,
                0,
                "model-cache",
            ),
            footprint(
                IndexFootprintDomain::Artifact,
                140 * 1024 * 1024,
                48 * 1024 * 1024,
                360,
                "artifact-manifest",
            ),
        ],
    )
}

#[must_use]
pub fn index_footprint_advisor_oversized_fixture() -> IndexFootprintAdvisorReport {
    index_footprint_advisor_report_fixture(
        "2026-05-08T00:00:00Z",
        "fsfs-status-oversized-fixture",
        Some(1024 * 1024 * 1024),
        vec![
            footprint(
                IndexFootprintDomain::VectorIndex,
                380 * 1024 * 1024,
                64 * 1024 * 1024,
                190,
                "fsvi+wal",
            ),
            footprint(
                IndexFootprintDomain::LexicalIndex,
                240 * 1024 * 1024,
                24 * 1024 * 1024,
                90,
                "lexical",
            ),
            footprint(
                IndexFootprintDomain::Metadata,
                90 * 1024 * 1024,
                8 * 1024 * 1024,
                80,
                "sqlite",
            ),
            footprint(
                IndexFootprintDomain::ModelCache,
                1080 * 1024 * 1024,
                640 * 1024 * 1024,
                0,
                "model-cache",
            ),
            footprint(
                IndexFootprintDomain::Artifact,
                260 * 1024 * 1024,
                120 * 1024 * 1024,
                120,
                "artifact-manifest",
            ),
        ],
    )
}

fn index_footprint_advisor_report_fixture(
    generated_at: &str,
    surface: &str,
    budget_bytes: Option<u64>,
    measurements: Vec<IndexFootprintDomainFootprint>,
) -> IndexFootprintAdvisorReport {
    let policy = IndexFootprintAdvisorPolicy::default();
    let total_bytes = measurements
        .iter()
        .map(|measurement| measurement.bytes)
        .sum::<u64>();
    let scenario = policy.classify(total_bytes, budget_bytes, &measurements);
    let recommendations = measurements
        .iter()
        .map(|measurement| policy.build_recommendation(scenario, measurement, total_bytes))
        .collect::<Vec<_>>();
    let projected_savings_bytes = recommendations
        .iter()
        .map(|recommendation| recommendation.projected_savings_bytes)
        .sum();
    let recommendation_count = recommendations.len();
    let highest_risk = recommendations
        .iter()
        .map(|recommendation| recommendation.risk)
        .max_by_key(|risk| risk_rank(*risk))
        .unwrap_or(IndexFootprintAdvisorRisk::Low);

    IndexFootprintAdvisorReport {
        kind: INDEX_FOOTPRINT_ADVISOR_REPORT_KIND.to_owned(),
        v: INDEX_FOOTPRINT_ADVISOR_SCHEMA_VERSION,
        generated_at: generated_at.to_owned(),
        surface: surface.to_owned(),
        policy_version: INDEX_FOOTPRINT_ADVISOR_POLICY_VERSION.to_owned(),
        scenario,
        dry_run: true,
        automatic_deletion_allowed: false,
        budget_bytes,
        total_bytes,
        measurements,
        recommendations,
        summary: IndexFootprintAdvisorSummary {
            recommendation_count,
            projected_savings_bytes,
            highest_risk,
        },
    }
}

impl IndexFootprintAdvisorContractDefinition {
    /// Validates the index-footprint advisor contract.
    ///
    /// # Errors
    ///
    /// Returns an error when the contract allows mutations or omits required
    /// domains, actions, thresholds, or replay commands.
    pub fn validate(&self) -> Result<(), String> {
        if self.kind != INDEX_FOOTPRINT_ADVISOR_CONTRACT_KIND {
            return Err("unexpected index-footprint advisor contract kind".to_owned());
        }
        if self.v != INDEX_FOOTPRINT_ADVISOR_SCHEMA_VERSION {
            return Err("unsupported index-footprint advisor contract version".to_owned());
        }
        if !self.dry_run_only || self.automatic_deletion_allowed {
            return Err("index-footprint advisor contract must be dry-run only".to_owned());
        }
        if self.policy_version != INDEX_FOOTPRINT_ADVISOR_POLICY_VERSION {
            return Err("unexpected index-footprint advisor policy version".to_owned());
        }
        if required_index_footprint_domains()
            .iter()
            .any(|required_domain| !self.required_domains.contains(required_domain))
        {
            return Err("index-footprint advisor missing required domain".to_owned());
        }
        if required_index_footprint_actions()
            .iter()
            .any(|required_action| !self.required_actions.contains(required_action))
        {
            return Err("index-footprint advisor missing required action".to_owned());
        }
        if self.policy.fragmentation_threshold_per_mille == 0
            || self.policy.oversize_threshold_per_mille == 0
            || self.policy.dominant_domain_threshold_per_mille == 0
            || self.policy.minimum_projected_savings_bytes == 0
        {
            return Err("index-footprint advisor thresholds must be nonzero".to_owned());
        }
        if !self
            .replay_command_template
            .contains("cargo test -p frankensearch-fsfs index_footprint_advisor_policy_suite")
        {
            return Err("index-footprint advisor missing exact replay template".to_owned());
        }
        Ok(())
    }
}

impl IndexFootprintAdvisorReport {
    /// Validates an index-footprint advisor report.
    ///
    /// # Errors
    ///
    /// Returns an error when the report can mutate data, omits a domain, or
    /// publishes a recommendation without projected savings/risk/replay data.
    pub fn validate(&self) -> Result<(), String> {
        if self.kind != INDEX_FOOTPRINT_ADVISOR_REPORT_KIND {
            return Err("unexpected index-footprint advisor report kind".to_owned());
        }
        if self.v != INDEX_FOOTPRINT_ADVISOR_SCHEMA_VERSION {
            return Err("unsupported index-footprint advisor report version".to_owned());
        }
        if self.generated_at.trim().is_empty() || self.surface.trim().is_empty() {
            return Err("index-footprint advisor report missing identity fields".to_owned());
        }
        if !self.dry_run || self.automatic_deletion_allowed {
            return Err("index-footprint advisor report must be dry-run only".to_owned());
        }
        if self.policy_version != INDEX_FOOTPRINT_ADVISOR_POLICY_VERSION {
            return Err("unexpected index-footprint advisor policy version".to_owned());
        }
        if required_index_footprint_domains()
            .iter()
            .any(|required_domain| {
                !self
                    .measurements
                    .iter()
                    .any(|measurement| measurement.domain == *required_domain)
            })
        {
            return Err("index-footprint advisor report missing domain footprint".to_owned());
        }
        for measurement in &self.measurements {
            measurement.validate()?;
        }
        let total_bytes = self
            .measurements
            .iter()
            .map(|measurement| measurement.bytes)
            .sum::<u64>();
        if self.total_bytes != total_bytes {
            return Err("index-footprint advisor total bytes mismatch".to_owned());
        }
        if self.recommendations.is_empty() {
            return Err("index-footprint advisor requires recommendations".to_owned());
        }
        for recommendation in &self.recommendations {
            recommendation.validate()?;
        }
        let projected_savings_bytes = self
            .recommendations
            .iter()
            .map(|recommendation| recommendation.projected_savings_bytes)
            .sum::<u64>();
        if self.summary.projected_savings_bytes != projected_savings_bytes {
            return Err("index-footprint advisor projected savings mismatch".to_owned());
        }
        if self.summary.recommendation_count != self.recommendations.len() {
            return Err("index-footprint advisor recommendation count mismatch".to_owned());
        }
        Ok(())
    }
}

impl IndexFootprintDomainFootprint {
    fn validate(&self) -> Result<(), String> {
        if self.source.trim().is_empty() {
            return Err("index-footprint domain source is required".to_owned());
        }
        if self.reclaimable_bytes > self.bytes {
            return Err("index-footprint reclaimable bytes exceed measured bytes".to_owned());
        }
        if self.fragmentation_per_mille > 1000 {
            return Err("index-footprint fragmentation must be at most 1000 per-mille".to_owned());
        }
        Ok(())
    }
}

impl IndexFootprintRecommendation {
    fn validate(&self) -> Result<(), String> {
        if !self.reason_code.starts_with("index_footprint.") {
            return Err("index-footprint recommendation reason code namespace mismatch".to_owned());
        }
        if self.rationale.trim().is_empty() {
            return Err("index-footprint recommendation rationale is required".to_owned());
        }
        if self.replay_command.trim().is_empty()
            || !self
                .replay_command
                .contains("cargo test -p frankensearch-fsfs index_footprint_advisor_policy_suite")
        {
            return Err("index-footprint recommendation missing exact replay command".to_owned());
        }
        if self.operator_command.trim().is_empty() {
            return Err("index-footprint recommendation missing operator command".to_owned());
        }
        if !self.operator_command.contains("--dry-run") {
            return Err("index-footprint recommendation must be dry-run".to_owned());
        }
        if self.projected_savings_bytes > self.measured_bytes {
            return Err("index-footprint projected savings exceed measured bytes".to_owned());
        }
        Ok(())
    }
}

#[must_use]
fn footprint(
    domain: IndexFootprintDomain,
    bytes: u64,
    reclaimable_bytes: u64,
    fragmentation_per_mille: u16,
    source: &str,
) -> IndexFootprintDomainFootprint {
    IndexFootprintDomainFootprint {
        domain,
        bytes,
        reclaimable_bytes,
        fragmentation_per_mille,
        source: source.to_owned(),
    }
}

#[must_use]
fn required_index_footprint_domains() -> &'static [IndexFootprintDomain] {
    &[
        IndexFootprintDomain::VectorIndex,
        IndexFootprintDomain::LexicalIndex,
        IndexFootprintDomain::Metadata,
        IndexFootprintDomain::ModelCache,
        IndexFootprintDomain::Artifact,
    ]
}

#[must_use]
fn required_index_footprint_actions() -> &'static [IndexFootprintAdvisorAction] {
    &[
        IndexFootprintAdvisorAction::Compaction,
        IndexFootprintAdvisorAction::Rebuild,
        IndexFootprintAdvisorAction::Retention,
        IndexFootprintAdvisorAction::FeatureAdjustment,
    ]
}

#[must_use]
fn recommendation_reason_code(
    scenario: IndexFootprintScenario,
    action: IndexFootprintAdvisorAction,
) -> String {
    let action = match action {
        IndexFootprintAdvisorAction::Compaction => "compaction",
        IndexFootprintAdvisorAction::Rebuild => "rebuild",
        IndexFootprintAdvisorAction::Retention => "retention",
        IndexFootprintAdvisorAction::FeatureAdjustment => "feature_adjustment",
    };
    format!("index_footprint.{}.{}", scenario.as_str(), action)
}

#[must_use]
fn recommendation_risk(
    scenario: IndexFootprintScenario,
    action: IndexFootprintAdvisorAction,
) -> IndexFootprintAdvisorRisk {
    match (scenario, action) {
        (IndexFootprintScenario::Small, _) | (_, IndexFootprintAdvisorAction::Compaction) => {
            IndexFootprintAdvisorRisk::Low
        }
        (_, IndexFootprintAdvisorAction::Rebuild | IndexFootprintAdvisorAction::Retention) => {
            IndexFootprintAdvisorRisk::Medium
        }
        (_, IndexFootprintAdvisorAction::FeatureAdjustment) => IndexFootprintAdvisorRisk::High,
    }
}

#[must_use]
fn risk_rank(risk: IndexFootprintAdvisorRisk) -> u8 {
    match risk {
        IndexFootprintAdvisorRisk::Low => 0,
        IndexFootprintAdvisorRisk::Medium => 1,
        IndexFootprintAdvisorRisk::High => 2,
    }
}

#[must_use]
fn replay_command(scenario: IndexFootprintScenario, domain: IndexFootprintDomain) -> String {
    format!(
        "FSFS_INDEX_FOOTPRINT_FIXTURE={} FSFS_INDEX_FOOTPRINT_DOMAIN={} cargo test -p frankensearch-fsfs index_footprint_advisor_policy_suite -- --nocapture",
        scenario.as_str(),
        domain.as_str()
    )
}

#[must_use]
fn operator_command(action: IndexFootprintAdvisorAction, domain: IndexFootprintDomain) -> String {
    match action {
        IndexFootprintAdvisorAction::Compaction => {
            "fsfs compact --dry-run --format json".to_owned()
        }
        IndexFootprintAdvisorAction::Rebuild => {
            format!(
                "fsfs index --rebuild {} --dry-run --format json",
                domain.as_str()
            )
        }
        IndexFootprintAdvisorAction::Retention => {
            "fsfs doctor --retention-audit --dry-run --format json".to_owned()
        }
        IndexFootprintAdvisorAction::FeatureAdjustment => {
            "fsfs config set search.fast_only true --dry-run --format json".to_owned()
        }
    }
}

#[must_use]
fn recommendation_rationale(
    scenario: IndexFootprintScenario,
    action: IndexFootprintAdvisorAction,
    domain: IndexFootprintDomain,
) -> String {
    let action = match action {
        IndexFootprintAdvisorAction::Compaction => "compact reclaimable fragments",
        IndexFootprintAdvisorAction::Rebuild => "rebuild the domain to remove stale segments",
        IndexFootprintAdvisorAction::Retention => "retain current files and only audit policy",
        IndexFootprintAdvisorAction::FeatureAdjustment => {
            "adjust feature settings before rebuilding or retaining large caches"
        }
    };
    format!("{}: {} for {}", scenario.as_str(), action, domain.as_str())
}

// ─── Pressure Sensing & Control-State Model ────────────────────────────────

/// Stable host-pressure control states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PressureState {
    Normal,
    Constrained,
    Degraded,
    Emergency,
}

impl PressureState {
    #[must_use]
    pub const fn severity(self) -> u8 {
        match self {
            Self::Normal => 0,
            Self::Constrained => 1,
            Self::Degraded => 2,
            Self::Emergency => 3,
        }
    }
}

/// One pressure telemetry sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PressureSignals {
    pub cpu_pct: f64,
    pub memory_pct: f64,
    pub io_bytes_per_sec: f64,
    pub load_avg_1m: f64,
}

impl PressureSignals {
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            cpu_pct: 0.0,
            memory_pct: 0.0,
            io_bytes_per_sec: 0.0,
            load_avg_1m: 0.0,
        }
    }
}

/// Threshold bundle used to map smoothed signals into control states.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PressureThresholds {
    pub constrained_cpu_pct: f64,
    pub degraded_cpu_pct: f64,
    pub emergency_cpu_pct: f64,
    pub constrained_memory_pct: f64,
    pub degraded_memory_pct: f64,
    pub emergency_memory_pct: f64,
    pub constrained_io_bytes_per_sec: f64,
    pub degraded_io_bytes_per_sec: f64,
    pub emergency_io_bytes_per_sec: f64,
    pub constrained_load_avg_1m: f64,
    pub degraded_load_avg_1m: f64,
    pub emergency_load_avg_1m: f64,
}

impl PressureThresholds {
    /// Build profile-aware thresholds from fsfs pressure config.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn from_config(config: &PressureConfig) -> Self {
        let cpu_base = f64::from(config.cpu_ceiling_pct);
        let io_base = config.io_ceiling_bytes_per_sec as f64;
        let load_base = f64::from(config.load_ceiling_per_mille) / 1000.0;
        let (memory_degraded_pct, constrained_pad, emergency_pad): (f64, f64, f64) =
            match config.profile {
                crate::config::PressureProfile::Strict => (75.0, 15.0, 8.0),
                crate::config::PressureProfile::Performance => (85.0, 10.0, 10.0),
                crate::config::PressureProfile::Degraded => (90.0, 8.0, 7.0),
            };

        Self {
            constrained_cpu_pct: (cpu_base - 10.0).max(1.0),
            degraded_cpu_pct: cpu_base,
            emergency_cpu_pct: (cpu_base + 10.0).min(100.0),
            constrained_memory_pct: (memory_degraded_pct - constrained_pad).max(1.0),
            degraded_memory_pct: memory_degraded_pct,
            emergency_memory_pct: (memory_degraded_pct + emergency_pad).min(99.0),
            constrained_io_bytes_per_sec: io_base * 0.8,
            degraded_io_bytes_per_sec: io_base,
            emergency_io_bytes_per_sec: io_base * 1.5,
            constrained_load_avg_1m: (load_base * 0.75).max(0.1),
            degraded_load_avg_1m: load_base.max(0.1),
            emergency_load_avg_1m: (load_base * 1.35).max(0.2),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PressureStateUpdate {
    pub state: PressureState,
    pub target_state: PressureState,
    pub transition_reason_code: Option<&'static str>,
    pub smoothed: PressureSignals,
    pub consecutive_observations: usize,
}

/// Anti-flap pressure-state machine with EWMA smoothing.
#[derive(Debug, Clone)]
pub struct PressureStateMachine {
    state: PressureState,
    alpha: f64,
    anti_flap_readings: usize,
    thresholds: PressureThresholds,
    smoothed: Option<PressureSignals>,
    pending_state: Option<PressureState>,
    pending_count: usize,
}

impl PressureStateMachine {
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn from_config(config: &PressureConfig) -> Self {
        let alpha = f64::from(config.ewma_alpha_per_mille) / 1000.0;
        Self {
            state: PressureState::Normal,
            alpha,
            anti_flap_readings: usize::from(config.anti_flap_readings.max(1)),
            thresholds: PressureThresholds::from_config(config),
            smoothed: None,
            pending_state: None,
            pending_count: 0,
        }
    }

    #[must_use]
    pub const fn state(&self) -> PressureState {
        self.state
    }

    #[must_use]
    pub const fn thresholds(&self) -> &PressureThresholds {
        &self.thresholds
    }

    #[must_use]
    pub const fn smoothed(&self) -> Option<PressureSignals> {
        self.smoothed
    }

    /// Ingest one sample and derive the stable control state.
    pub fn observe(&mut self, sample: PressureSignals) -> PressureStateUpdate {
        let smoothed = if let Some(previous) = self.smoothed {
            PressureSignals {
                cpu_pct: ewma(previous.cpu_pct, sample.cpu_pct, self.alpha),
                memory_pct: ewma(previous.memory_pct, sample.memory_pct, self.alpha),
                io_bytes_per_sec: ewma(
                    previous.io_bytes_per_sec,
                    sample.io_bytes_per_sec,
                    self.alpha,
                ),
                load_avg_1m: ewma(previous.load_avg_1m, sample.load_avg_1m, self.alpha),
            }
        } else {
            sample
        };
        self.smoothed = Some(smoothed);

        let (target_state, target_reason_code) = self.evaluate_target(smoothed);
        if target_state == self.state {
            self.pending_state = None;
            self.pending_count = 0;
            return PressureStateUpdate {
                state: self.state,
                target_state,
                transition_reason_code: None,
                smoothed,
                consecutive_observations: 0,
            };
        }

        if self.pending_state == Some(target_state) {
            self.pending_count = self.pending_count.saturating_add(1);
        } else {
            self.pending_state = Some(target_state);
            self.pending_count = 1;
        }

        if self.pending_count < self.anti_flap_readings {
            return PressureStateUpdate {
                state: self.state,
                target_state,
                transition_reason_code: Some(FsfsReasonCode::DEGRADE_TRANSITION_HELD),
                smoothed,
                consecutive_observations: self.pending_count,
            };
        }

        let previous = self.state;
        self.state = target_state;
        self.pending_state = None;
        self.pending_count = 0;

        let transition_reason_code = if target_state.severity() < previous.severity() {
            Some(FsfsReasonCode::DEGRADE_TRANSITION_RECOVERED)
        } else {
            Some(target_reason_code)
        };

        PressureStateUpdate {
            state: self.state,
            target_state,
            transition_reason_code,
            smoothed,
            consecutive_observations: self.anti_flap_readings,
        }
    }

    #[must_use]
    fn evaluate_target(&self, smoothed: PressureSignals) -> (PressureState, &'static str) {
        if smoothed.cpu_pct >= self.thresholds.emergency_cpu_pct
            || smoothed.memory_pct >= self.thresholds.emergency_memory_pct
            || smoothed.io_bytes_per_sec >= self.thresholds.emergency_io_bytes_per_sec
            || smoothed.load_avg_1m >= self.thresholds.emergency_load_avg_1m
        {
            let reason = if smoothed.cpu_pct >= self.thresholds.emergency_cpu_pct {
                FsfsReasonCode::DEGRADE_CPU_CEILING_HIT
            } else if smoothed.memory_pct >= self.thresholds.emergency_memory_pct {
                FsfsReasonCode::DEGRADE_MEMORY_CEILING_HIT
            } else if smoothed.io_bytes_per_sec >= self.thresholds.emergency_io_bytes_per_sec {
                FsfsReasonCode::DEGRADE_IO_CEILING_HIT
            } else {
                FsfsReasonCode::DEGRADE_LOAD_CEILING_HIT
            };
            return (PressureState::Emergency, reason);
        }

        if smoothed.cpu_pct >= self.thresholds.degraded_cpu_pct
            || smoothed.memory_pct >= self.thresholds.degraded_memory_pct
            || smoothed.io_bytes_per_sec >= self.thresholds.degraded_io_bytes_per_sec
            || smoothed.load_avg_1m >= self.thresholds.degraded_load_avg_1m
        {
            let reason = if smoothed.cpu_pct >= self.thresholds.degraded_cpu_pct {
                FsfsReasonCode::DEGRADE_CPU_CEILING_HIT
            } else if smoothed.memory_pct >= self.thresholds.degraded_memory_pct {
                FsfsReasonCode::DEGRADE_MEMORY_CEILING_HIT
            } else if smoothed.io_bytes_per_sec >= self.thresholds.degraded_io_bytes_per_sec {
                FsfsReasonCode::DEGRADE_IO_CEILING_HIT
            } else {
                FsfsReasonCode::DEGRADE_LOAD_CEILING_HIT
            };
            return (PressureState::Degraded, reason);
        }

        if smoothed.cpu_pct >= self.thresholds.constrained_cpu_pct
            || smoothed.memory_pct >= self.thresholds.constrained_memory_pct
            || smoothed.io_bytes_per_sec >= self.thresholds.constrained_io_bytes_per_sec
            || smoothed.load_avg_1m >= self.thresholds.constrained_load_avg_1m
        {
            let reason = if smoothed.cpu_pct >= self.thresholds.constrained_cpu_pct {
                FsfsReasonCode::DEGRADE_CPU_CEILING_HIT
            } else if smoothed.memory_pct >= self.thresholds.constrained_memory_pct {
                FsfsReasonCode::DEGRADE_MEMORY_CEILING_HIT
            } else if smoothed.io_bytes_per_sec >= self.thresholds.constrained_io_bytes_per_sec {
                FsfsReasonCode::DEGRADE_IO_CEILING_HIT
            } else {
                FsfsReasonCode::DEGRADE_LOAD_CEILING_HIT
            };
            return (PressureState::Constrained, reason);
        }

        (
            PressureState::Normal,
            FsfsReasonCode::DEGRADE_TRANSITION_RECOVERED,
        )
    }
}

#[must_use]
fn ewma(previous: f64, current: f64, alpha: f64) -> f64 {
    alpha.mul_add(current, (1.0 - alpha) * previous)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CpuCounters {
    total: u64,
    idle: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct IoCounters {
    read_bytes: u64,
    write_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct RawPressureCounters {
    cpu: CpuCounters,
    io: IoCounters,
    memory_total_bytes: u64,
    memory_available_bytes: u64,
    load_avg_1m: f64,
    sampled_at_ms: u64,
}

/// Linux `/proc` pressure sampler with delta-based CPU/IO rates.
#[derive(Debug, Clone, Default)]
pub struct HostPressureCollector {
    previous: Option<RawPressureCounters>,
}

impl HostPressureCollector {
    #[must_use]
    pub const fn new() -> Self {
        Self { previous: None }
    }

    /// Sample host pressure metrics from `/proc`.
    ///
    /// Returns `None` on unsupported platforms or when proc parsing fails.
    pub fn sample_now(&mut self, sampled_at_ms: u64) -> Option<PressureSignals> {
        let current = read_procfs_counters(sampled_at_ms)?;
        let sample = self.previous.map_or_else(
            || PressureSignals {
                cpu_pct: 0.0,
                memory_pct: memory_pct(current.memory_total_bytes, current.memory_available_bytes),
                io_bytes_per_sec: 0.0,
                load_avg_1m: current.load_avg_1m,
            },
            |previous| build_pressure_signals(previous, current),
        );
        self.previous = Some(current);
        Some(sample)
    }
}

#[must_use]
fn memory_pct(total_bytes: u64, available_bytes: u64) -> f64 {
    if total_bytes == 0 {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let used = total_bytes.saturating_sub(available_bytes) as f64;
    #[allow(clippy::cast_precision_loss)]
    let total = total_bytes as f64;
    (used / total) * 100.0
}

#[must_use]
fn build_pressure_signals(
    previous: RawPressureCounters,
    current: RawPressureCounters,
) -> PressureSignals {
    let cpu_delta_total = current.cpu.total.saturating_sub(previous.cpu.total);
    let cpu_delta_idle = current.cpu.idle.saturating_sub(previous.cpu.idle);
    let cpu_pct = if cpu_delta_total == 0 {
        0.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        let busy = cpu_delta_total.saturating_sub(cpu_delta_idle) as f64;
        #[allow(clippy::cast_precision_loss)]
        let total = cpu_delta_total as f64;
        (busy / total) * 100.0
    };

    let elapsed_ms = current.sampled_at_ms.saturating_sub(previous.sampled_at_ms);
    let io_delta = current
        .io
        .read_bytes
        .saturating_add(current.io.write_bytes)
        .saturating_sub(
            previous
                .io
                .read_bytes
                .saturating_add(previous.io.write_bytes),
        );
    let io_bytes_per_sec = if elapsed_ms == 0 {
        0.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        let delta = io_delta as f64;
        #[allow(clippy::cast_precision_loss)]
        let elapsed_secs = elapsed_ms as f64 / 1000.0;
        delta / elapsed_secs
    };

    PressureSignals {
        cpu_pct,
        memory_pct: memory_pct(current.memory_total_bytes, current.memory_available_bytes),
        io_bytes_per_sec,
        load_avg_1m: current.load_avg_1m,
    }
}

#[must_use]
fn read_procfs_counters(sampled_at_ms: u64) -> Option<RawPressureCounters> {
    #[cfg(target_os = "linux")]
    {
        let cpu_text = std::fs::read_to_string("/proc/stat").ok()?;
        let mem_text = std::fs::read_to_string("/proc/meminfo").ok()?;
        let io_text = std::fs::read_to_string("/proc/self/io").ok()?;
        let load_text = std::fs::read_to_string("/proc/loadavg").ok()?;

        let cpu = parse_proc_stat_cpu_line(&cpu_text)?;
        let (memory_total_bytes, memory_available_bytes) = parse_proc_meminfo(&mem_text)?;
        let io = parse_proc_self_io(&io_text)?;
        let load_avg_1m = parse_proc_loadavg(&load_text)?;

        Some(RawPressureCounters {
            cpu,
            io,
            memory_total_bytes,
            memory_available_bytes,
            load_avg_1m,
            sampled_at_ms,
        })
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = sampled_at_ms;
        None
    }
}

#[cfg(any(target_os = "linux", test))]
#[must_use]
fn parse_proc_stat_cpu_line(contents: &str) -> Option<CpuCounters> {
    let line = contents.lines().find(|line| line.starts_with("cpu "))?;
    let fields: Vec<u64> = line
        .split_whitespace()
        .skip(1)
        .filter_map(|value| value.parse::<u64>().ok())
        .collect();
    if fields.len() < 4 {
        return None;
    }
    let total = fields.iter().copied().sum();
    let idle = fields
        .get(3)
        .copied()
        .unwrap_or(0)
        .saturating_add(fields.get(4).copied().unwrap_or(0));
    Some(CpuCounters { total, idle })
}

#[cfg(any(target_os = "linux", test))]
#[must_use]
fn parse_proc_meminfo(contents: &str) -> Option<(u64, u64)> {
    let mut total_kb = None;
    let mut available_kb = None;

    for line in contents.lines() {
        if line.starts_with("MemTotal:") {
            total_kb = line
                .split_whitespace()
                .nth(1)
                .and_then(|value| value.parse::<u64>().ok());
        } else if line.starts_with("MemAvailable:") {
            available_kb = line
                .split_whitespace()
                .nth(1)
                .and_then(|value| value.parse::<u64>().ok());
        }
    }

    let total = total_kb?;
    let available = available_kb?;
    Some((total.saturating_mul(1024), available.saturating_mul(1024)))
}

#[cfg(any(target_os = "linux", test))]
#[must_use]
fn parse_proc_self_io(contents: &str) -> Option<IoCounters> {
    let mut read_bytes = None;
    let mut write_bytes = None;

    for line in contents.lines() {
        if line.starts_with("read_bytes:") {
            read_bytes = line
                .split_whitespace()
                .nth(1)
                .and_then(|value| value.parse::<u64>().ok());
        } else if line.starts_with("write_bytes:") {
            write_bytes = line
                .split_whitespace()
                .nth(1)
                .and_then(|value| value.parse::<u64>().ok());
        }
    }

    Some(IoCounters {
        read_bytes: read_bytes?,
        write_bytes: write_bytes?,
    })
}

#[cfg(any(target_os = "linux", test))]
#[must_use]
fn parse_proc_loadavg(contents: &str) -> Option<f64> {
    contents
        .split_whitespace()
        .next()
        .and_then(|value| value.parse::<f64>().ok())
}

// ─── PID File Management ────────────────────────────────────────────────────

/// Contents of a PID file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PidFileContents {
    /// Process ID.
    pub pid: u32,
    /// When the daemon started.
    pub started_at_ms: u64,
    /// Hostname.
    pub hostname: String,
    /// Version string.
    pub version: String,
}

impl PidFileContents {
    /// Create contents for the current process.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn current(version: impl Into<String>) -> Self {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            pid: std::process::id(),
            started_at_ms: now,
            hostname: hostname(),
            version: version.into(),
        }
    }

    /// Whether the PID in this file is still alive on the local host.
    #[must_use]
    pub fn is_alive(&self) -> bool {
        if self.hostname != hostname() {
            return true; // Can't verify cross-host.
        }
        pid_is_alive(self.pid)
    }
}

/// Check whether a PID is alive on the local system.
fn pid_is_alive(pid: u32) -> bool {
    crate::concurrency::is_pid_alive(pid)
}

/// Manage a PID file at the given path.
pub struct PidFile {
    path: PathBuf,
    acquired: bool,
}

impl PidFile {
    /// Create a PID file manager for the given path.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            acquired: false,
        }
    }

    /// Default PID file path.
    #[must_use]
    pub fn default_path() -> PathBuf {
        let runtime_dir = dirs::runtime_dir()
            .or_else(|| dirs::cache_dir().map(|d| d.join("run")))
            .unwrap_or_else(|| PathBuf::from("/tmp"));
        runtime_dir.join("fsfs.pid")
    }

    /// Maximum retries when racing against stale PID file cleanup.
    const ACQUIRE_MAX_RETRIES: usize = 2;

    /// Attempt to acquire the PID file. Returns `Ok(())` if acquired,
    /// `Err` if another live process holds it.
    ///
    /// Uses `O_CREAT|O_EXCL` (via [`OpenOptions::create_new`]) to atomically
    /// create the PID file, eliminating the TOCTOU race between checking for
    /// an existing daemon and writing our own PID.
    ///
    /// # Errors
    ///
    /// Returns error if PID file is held by a live process or I/O fails.
    pub fn acquire(&mut self, version: &str) -> std::io::Result<()> {
        // Ensure parent directory exists.
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let contents = PidFileContents::current(version);
        let json = serde_json::to_string_pretty(&contents)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        for _ in 0..=Self::ACQUIRE_MAX_RETRIES {
            // Attempt atomic exclusive creation — only one process can succeed.
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&self.path)
            {
                Ok(mut file) => {
                    file.write_all(json.as_bytes())?;
                    self.acquired = true;
                    info!(
                        target: "frankensearch.fsfs.lifecycle",
                        pid = contents.pid,
                        path = %self.path.display(),
                        "PID file acquired"
                    );
                    return Ok(());
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    // PID file exists — check if the holder is still alive.
                    match read_pid_file(&self.path) {
                        Ok(existing) if existing.is_alive() => {
                            return Err(std::io::Error::new(
                                std::io::ErrorKind::AddrInUse,
                                format!(
                                    "fsfs daemon already running (PID {}, started at {})",
                                    existing.pid, existing.started_at_ms
                                ),
                            ));
                        }
                        Ok(existing) => {
                            // Stale PID file — remove and retry.
                            warn!(
                                target: "frankensearch.fsfs.lifecycle",
                                pid = existing.pid,
                                "Removing stale PID file (process is dead)"
                            );
                            let _ = std::fs::remove_file(&self.path);
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                            // Removed between our create_new and read. Retry.
                        }
                        Err(e) => return Err(e),
                    }
                }
                Err(e) => return Err(e),
            }
        }

        Err(std::io::Error::new(
            std::io::ErrorKind::AddrInUse,
            format!(
                "PID file acquisition failed after {} retries (contention)",
                Self::ACQUIRE_MAX_RETRIES + 1
            ),
        ))
    }

    /// Release the PID file (remove it). No-op if not acquired.
    pub fn release(&mut self) {
        if !self.acquired {
            return;
        }
        self.acquired = false;
        match std::fs::remove_file(&self.path) {
            Ok(()) => {
                debug!(
                    target: "frankensearch.fsfs.lifecycle",
                    path = %self.path.display(),
                    "PID file released"
                );
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Already gone — fine.
            }
            Err(e) => {
                warn!(
                    target: "frankensearch.fsfs.lifecycle",
                    error = %e,
                    path = %self.path.display(),
                    "Failed to remove PID file"
                );
            }
        }
    }

    /// Read the current PID file contents.
    ///
    /// # Errors
    ///
    /// Returns error if the file doesn't exist or can't be parsed.
    pub fn read(&self) -> std::io::Result<PidFileContents> {
        read_pid_file(&self.path)
    }

    /// Path to the PID file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for PidFile {
    fn drop(&mut self) {
        self.release();
    }
}

fn read_pid_file(path: &Path) -> std::io::Result<PidFileContents> {
    let contents = std::fs::read_to_string(path)?;
    serde_json::from_str(&contents)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

// ─── Watchdog / Supervision ─────────────────────────────────────────────────

/// Configuration for subsystem watchdog/restart behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchdogConfig {
    /// Initial backoff before restarting a failed subsystem.
    #[serde(default = "default_initial_backoff_ms")]
    pub initial_backoff_ms: u64,
    /// Maximum backoff between restart attempts.
    #[serde(default = "default_max_backoff_ms")]
    pub max_backoff_ms: u64,
    /// Backoff multiplier.
    #[serde(default = "default_backoff_multiplier")]
    pub backoff_multiplier: f64,
    /// Maximum restart attempts before giving up on a subsystem.
    #[serde(default = "default_max_restarts")]
    pub max_restarts: u32,
    /// Health check interval.
    #[serde(default = "default_health_check_interval_ms")]
    pub health_check_interval_ms: u64,
}

const fn default_initial_backoff_ms() -> u64 {
    5_000
}
const fn default_max_backoff_ms() -> u64 {
    60_000
}
const fn default_backoff_multiplier() -> f64 {
    2.0
}
const fn default_max_restarts() -> u32 {
    5
}
const fn default_health_check_interval_ms() -> u64 {
    10_000
}

impl Default for WatchdogConfig {
    fn default() -> Self {
        Self {
            initial_backoff_ms: default_initial_backoff_ms(),
            max_backoff_ms: default_max_backoff_ms(),
            backoff_multiplier: default_backoff_multiplier(),
            max_restarts: default_max_restarts(),
            health_check_interval_ms: default_health_check_interval_ms(),
        }
    }
}

impl WatchdogConfig {
    /// Compute the backoff delay for the given restart attempt (0-indexed).
    #[must_use]
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn backoff_delay(&self, attempt: u32) -> Duration {
        let multiplier = self.backoff_multiplier.powi(attempt as i32);
        let delay_ms = (self.initial_backoff_ms as f64) * multiplier;
        let capped_ms = delay_ms.min(self.max_backoff_ms as f64);
        // NaN/negative/infinite multiplier can produce NaN or negative
        // delay_ms. NaN propagates through .min(), and `NaN as u64` yields 0
        // in Rust — causing a zero-backoff tight retry loop.
        if !capped_ms.is_finite() || capped_ms < 0.0 {
            return Duration::from_millis(self.initial_backoff_ms);
        }
        Duration::from_millis(capped_ms as u64)
    }

    /// Whether the restart limit has been reached.
    #[must_use]
    pub const fn is_exhausted(&self, attempt: u32) -> bool {
        attempt >= self.max_restarts
    }
}

// ─── Resource Limits ────────────────────────────────────────────────────────

/// Configurable resource limits for the daemon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum RSS in bytes (0 = unlimited).
    #[serde(default)]
    pub max_rss_bytes: u64,
    /// Maximum number of threads (0 = unlimited).
    #[serde(default)]
    pub max_threads: u32,
    /// Maximum open file descriptors (0 = unlimited).
    #[serde(default)]
    pub max_open_fds: u32,
    /// Maximum disk usage for index data in bytes (0 = unlimited).
    #[serde(default)]
    pub max_index_bytes: u64,
}

impl Default for ResourceLimits {
    #[allow(clippy::derivable_impls)]
    fn default() -> Self {
        Self {
            max_rss_bytes: 0,
            max_threads: 0,
            max_open_fds: 0,
            max_index_bytes: 0,
        }
    }
}

impl ResourceLimits {
    /// Check whether the given usage exceeds any configured limit.
    #[must_use]
    pub fn check(&self, usage: &ResourceUsage) -> Vec<LimitViolation> {
        let mut violations = Vec::new();

        if self.max_rss_bytes > 0
            && let Some(rss) = usage.rss_bytes
            && rss > self.max_rss_bytes
        {
            violations.push(LimitViolation {
                resource: "rss_bytes",
                limit: self.max_rss_bytes,
                actual: rss,
            });
        }

        if self.max_threads > 0
            && let Some(threads) = usage.thread_count
            && threads > self.max_threads
        {
            violations.push(LimitViolation {
                resource: "threads",
                limit: u64::from(self.max_threads),
                actual: u64::from(threads),
            });
        }

        if self.max_open_fds > 0
            && let Some(fds) = usage.open_fds
            && fds > self.max_open_fds
        {
            violations.push(LimitViolation {
                resource: "open_fds",
                limit: u64::from(self.max_open_fds),
                actual: u64::from(fds),
            });
        }

        if self.max_index_bytes > 0
            && let Some(index_bytes) = usage.effective_index_bytes()
            && index_bytes > self.max_index_bytes
        {
            violations.push(LimitViolation {
                resource: "index_bytes",
                limit: self.max_index_bytes,
                actual: index_bytes,
            });
        }

        violations
    }
}

/// A resource limit violation.
#[derive(Debug, Clone)]
pub struct LimitViolation {
    /// Which resource is over limit.
    pub resource: &'static str,
    /// The configured limit.
    pub limit: u64,
    /// The actual value.
    pub actual: u64,
}

impl std::fmt::Display for LimitViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: {} exceeds limit {}",
            self.resource, self.actual, self.limit
        )
    }
}

// ─── Lifecycle Tracker ──────────────────────────────────────────────────────

/// Tracks the lifecycle state of the daemon and all subsystems.
pub struct LifecycleTracker {
    phase: std::sync::Mutex<DaemonPhase>,
    started_at: Instant,
    subsystems: std::sync::Mutex<HashMap<SubsystemId, SubsystemHealth>>,
    total_errors: AtomicU64,
    total_panics: AtomicU64,
    resource_usage: std::sync::Mutex<ResourceUsage>,
    watchdog_config: WatchdogConfig,
    resource_limits: ResourceLimits,
}

impl LifecycleTracker {
    /// Create a new tracker in the Initializing phase.
    #[must_use]
    pub fn new(watchdog_config: WatchdogConfig, resource_limits: ResourceLimits) -> Self {
        Self {
            phase: std::sync::Mutex::new(DaemonPhase::Initializing),
            started_at: Instant::now(),
            subsystems: std::sync::Mutex::new(HashMap::new()),
            total_errors: AtomicU64::new(0),
            total_panics: AtomicU64::new(0),
            resource_usage: std::sync::Mutex::new(ResourceUsage::default()),
            watchdog_config,
            resource_limits,
        }
    }

    /// Transition to a new lifecycle phase.
    pub fn transition_to(&self, phase: DaemonPhase) {
        let mut current = lock_or_recover(&self.phase);
        info!(
            target: "frankensearch.fsfs.lifecycle",
            from = %*current,
            to = %phase,
            "Daemon phase transition"
        );
        *current = phase;
    }

    /// Current lifecycle phase.
    #[must_use]
    pub fn current_phase(&self) -> DaemonPhase {
        *lock_or_recover(&self.phase)
    }

    /// Register a subsystem with initial Pending status.
    pub fn register_subsystem(&self, id: SubsystemId) {
        let now = iso_now();
        let mut subs = lock_or_recover(&self.subsystems);
        subs.insert(
            id,
            SubsystemHealth {
                id,
                status: HealthStatus::Pending,
                restart_count: 0,
                last_error: None,
                last_healthy_at: None,
                status_since: now,
            },
        );
    }

    /// Update a subsystem's health status.
    pub fn update_subsystem(&self, id: SubsystemId, status: HealthStatus, error: Option<String>) {
        let now = iso_now();
        let mut subs = lock_or_recover(&self.subsystems);
        if let Some(health) = subs.get_mut(&id) {
            if status == HealthStatus::Healthy {
                health.last_healthy_at = Some(now.clone());
            }
            if status == HealthStatus::Failed {
                self.total_errors.fetch_add(1, Ordering::Relaxed);
            }
            health.status = status;
            health.last_error = error;
            health.status_since = now;
        }
    }

    /// Record a recovered panic for a subsystem.
    pub fn record_panic_recovery(&self, id: SubsystemId, error_msg: &str) {
        self.total_panics.fetch_add(1, Ordering::Relaxed);
        let now = iso_now();
        let mut subs = lock_or_recover(&self.subsystems);
        if let Some(health) = subs.get_mut(&id) {
            health.restart_count += 1;
            health.last_error = Some(error_msg.to_owned());
            health.status = HealthStatus::Degraded;
            health.status_since = now;
        }
        drop(subs);

        warn!(
            target: "frankensearch.fsfs.lifecycle",
            subsystem = id.name(),
            error = error_msg,
            "Panic recovered in subsystem"
        );
    }

    /// Whether the watchdog should restart a given subsystem.
    #[must_use]
    pub fn should_restart(&self, id: SubsystemId) -> Option<Duration> {
        let subs = lock_or_recover(&self.subsystems);
        let health = subs.get(&id)?;

        if health.status != HealthStatus::Failed {
            return None;
        }

        let restart_count = health.restart_count;
        drop(subs);

        if self.watchdog_config.is_exhausted(restart_count) {
            warn!(
                target: "frankensearch.fsfs.lifecycle",
                subsystem = id.name(),
                restarts = restart_count,
                "Subsystem restart limit exhausted"
            );
            return None;
        }

        Some(self.watchdog_config.backoff_delay(restart_count))
    }

    /// Check resource limits against current usage.
    #[must_use]
    pub fn check_resource_limits(&self, usage: &ResourceUsage) -> Vec<LimitViolation> {
        self.resource_limits.check(usage)
    }

    /// Update the latest resource snapshot used by status reporting.
    pub fn set_resource_usage(&self, usage: ResourceUsage) {
        let mut current = lock_or_recover(&self.resource_usage);
        *current = usage;
    }

    /// Read the latest resource snapshot.
    #[must_use]
    pub fn current_resource_usage(&self) -> ResourceUsage {
        lock_or_recover(&self.resource_usage).clone()
    }

    /// Evaluate staged disk-budget control state for the current index footprint.
    ///
    /// Returns `None` when no index budget is configured.
    #[must_use]
    pub fn evaluate_index_budget(&self, index_bytes: u64) -> Option<DiskBudgetSnapshot> {
        let budget_bytes = self.resource_limits.max_index_bytes;
        if budget_bytes == 0 {
            return None;
        }
        Some(DiskBudgetPolicy::default().evaluate(index_bytes, budget_bytes))
    }

    /// Evaluate staged disk-budget control state from an aggregated usage snapshot.
    ///
    /// Returns `None` when no index budget is configured or no index usage is known.
    #[must_use]
    pub fn evaluate_usage_budget(&self, usage: &ResourceUsage) -> Option<DiskBudgetSnapshot> {
        usage
            .effective_index_bytes()
            .and_then(|index_bytes| self.evaluate_index_budget(index_bytes))
    }

    /// Build a complete status snapshot.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn status(&self) -> DaemonStatus {
        let subs = lock_or_recover(&self.subsystems);
        let subsystem_vec: Vec<SubsystemHealth> = subs.values().cloned().collect();
        drop(subs);
        let resources = self.current_resource_usage();
        let disk_budget = self.evaluate_usage_budget(&resources);
        let disk_budget_reason_code = disk_budget.map(|snapshot| snapshot.reason_code.to_owned());

        let started_at_ms = (SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64)
            .saturating_sub(self.started_at.elapsed().as_millis() as u64);

        // Compute aggregate phase.
        let phase = *lock_or_recover(&self.phase);

        DaemonStatus {
            phase,
            pid: std::process::id(),
            started_at: format_epoch_ms(started_at_ms),
            uptime_secs: self.started_at.elapsed().as_secs(),
            subsystems: subsystem_vec,
            total_errors: self.total_errors.load(Ordering::Relaxed),
            total_panics_recovered: self.total_panics.load(Ordering::Relaxed),
            resources,
            disk_budget,
            disk_budget_reason_code,
        }
    }

    /// Watchdog config reference.
    #[must_use]
    pub const fn watchdog_config(&self) -> &WatchdogConfig {
        &self.watchdog_config
    }

    /// Resource limits reference.
    #[must_use]
    pub const fn resource_limits(&self) -> &ResourceLimits {
        &self.resource_limits
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn hostname() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("HOST"))
        .unwrap_or_else(|_| "unknown".into())
}

fn lock_or_recover<T>(mutex: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            warn!(
                target: "frankensearch.fsfs.lifecycle",
                "poisoned mutex encountered; recovering inner state"
            );
            poisoned.into_inner()
        }
    }
}

fn iso_now() -> String {
    let secs = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format_epoch_secs(secs)
}

fn format_epoch_ms(ms: u64) -> String {
    format_epoch_secs(ms / 1000)
}

fn format_epoch_secs(secs: u64) -> String {
    // Simple ISO 8601 formatting without chrono dependency.
    // Good enough for status reporting.
    let days_since_epoch = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Approximate date calculation (not accounting for leap seconds).
    let (year, month, day) = epoch_days_to_ymd(days_since_epoch);
    format!("{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
}

const fn epoch_days_to_ymd(days: u64) -> (u64, u64, u64) {
    // Civil calendar calculation from days since 1970-01-01.
    // Based on Howard Hinnant's algorithm.
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    // ── DaemonPhase ──

    #[test]
    fn daemon_phase_accepting_work() {
        assert!(!DaemonPhase::Initializing.is_accepting_work());
        assert!(!DaemonPhase::StartingUp.is_accepting_work());
        assert!(DaemonPhase::Running.is_accepting_work());
        assert!(DaemonPhase::Degraded.is_accepting_work());
        assert!(!DaemonPhase::ShuttingDown.is_accepting_work());
        assert!(!DaemonPhase::Stopped.is_accepting_work());
    }

    #[test]
    fn daemon_phase_terminal() {
        assert!(!DaemonPhase::Running.is_terminal());
        assert!(DaemonPhase::Stopped.is_terminal());
    }

    #[test]
    fn daemon_phase_display() {
        assert_eq!(format!("{}", DaemonPhase::Running), "running");
        assert_eq!(format!("{}", DaemonPhase::Degraded), "degraded");
    }

    #[test]
    fn daemon_phase_serde_roundtrip() {
        let phase = DaemonPhase::ShuttingDown;
        let json = serde_json::to_string(&phase).unwrap();
        assert_eq!(json, "\"shutting_down\"");
        let parsed: DaemonPhase = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, phase);
    }

    // ── SubsystemId ──

    #[test]
    fn subsystem_id_names() {
        assert_eq!(SubsystemId::Crawler.name(), "crawler");
        assert_eq!(SubsystemId::QueryServer.name(), "query_server");
    }

    // ── HealthStatus ──

    #[test]
    fn health_status_alive() {
        assert!(HealthStatus::Pending.is_alive());
        assert!(HealthStatus::Healthy.is_alive());
        assert!(HealthStatus::Degraded.is_alive());
        assert!(!HealthStatus::Failed.is_alive());
        assert!(!HealthStatus::Stopped.is_alive());
    }

    // ── Pressure Control ──

    #[test]
    fn pressure_machine_requires_hysteresis_before_escalation() {
        let config = PressureConfig {
            anti_flap_readings: 3,
            cpu_ceiling_pct: 80,
            ..PressureConfig::default()
        };
        let mut machine = PressureStateMachine::from_config(&config);

        let hot = PressureSignals {
            cpu_pct: 95.0,
            memory_pct: 50.0,
            io_bytes_per_sec: 10_000.0,
            load_avg_1m: 0.5,
        };

        let u1 = machine.observe(hot);
        assert_eq!(u1.state, PressureState::Normal);
        assert_eq!(
            u1.transition_reason_code,
            Some(FsfsReasonCode::DEGRADE_TRANSITION_HELD)
        );
        let u2 = machine.observe(hot);
        assert_eq!(u2.state, PressureState::Normal);
        let u3 = machine.observe(hot);
        assert_eq!(u3.state, PressureState::Emergency);
        assert_eq!(
            u3.transition_reason_code,
            Some(FsfsReasonCode::DEGRADE_CPU_CEILING_HIT)
        );
    }

    #[test]
    fn pressure_machine_recovers_with_hysteresis() {
        let config = PressureConfig {
            anti_flap_readings: 2,
            cpu_ceiling_pct: 75,
            ewma_alpha_per_mille: 1000,
            ..PressureConfig::default()
        };
        let mut machine = PressureStateMachine::from_config(&config);

        let hot = PressureSignals {
            cpu_pct: 95.0,
            memory_pct: 92.0,
            io_bytes_per_sec: 160_000_000.0,
            load_avg_1m: 10.0,
        };
        let cool = PressureSignals {
            cpu_pct: 5.0,
            memory_pct: 20.0,
            io_bytes_per_sec: 1_000.0,
            load_avg_1m: 0.2,
        };

        let _ = machine.observe(hot);
        let up = machine.observe(hot);
        assert_eq!(up.state, PressureState::Emergency);

        let held = machine.observe(cool);
        assert_eq!(held.state, PressureState::Emergency);
        assert_eq!(
            held.transition_reason_code,
            Some(FsfsReasonCode::DEGRADE_TRANSITION_HELD)
        );

        let recovered = machine.observe(cool);
        assert_eq!(recovered.state, PressureState::Normal);
        assert_eq!(
            recovered.transition_reason_code,
            Some(FsfsReasonCode::DEGRADE_TRANSITION_RECOVERED)
        );
    }

    #[test]
    fn pressure_machine_applies_ewma_smoothing() {
        let config = PressureConfig {
            anti_flap_readings: 1,
            ewma_alpha_per_mille: 300,
            ..PressureConfig::default()
        };
        let mut machine = PressureStateMachine::from_config(&config);

        let _ = machine.observe(PressureSignals {
            cpu_pct: 100.0,
            memory_pct: 0.0,
            io_bytes_per_sec: 0.0,
            load_avg_1m: 0.0,
        });
        let update = machine.observe(PressureSignals {
            cpu_pct: 0.0,
            memory_pct: 0.0,
            io_bytes_per_sec: 0.0,
            load_avg_1m: 0.0,
        });
        assert!(
            (update.smoothed.cpu_pct - 70.0).abs() < 0.0001,
            "expected EWMA cpu to be 70.0, got {}",
            update.smoothed.cpu_pct
        );
    }

    #[test]
    fn procfs_parsers_extract_expected_values() {
        let cpu = parse_proc_stat_cpu_line("cpu  10 20 30 40 50 0 0 0 0 0\n").expect("cpu parse");
        assert_eq!(cpu.total, 150);
        assert_eq!(cpu.idle, 90);

        let (mem_total, mem_available) =
            parse_proc_meminfo("MemTotal:       16000 kB\nMemAvailable:    4000 kB\n")
                .expect("mem parse");
        assert_eq!(mem_total, 16_384_000);
        assert_eq!(mem_available, 4_096_000);

        let io = parse_proc_self_io("read_bytes: 123\nwrite_bytes: 456\n").expect("io parse");
        assert_eq!(io.read_bytes, 123);
        assert_eq!(io.write_bytes, 456);

        let load = parse_proc_loadavg("1.23 0.55 0.33 1/200 12345\n").expect("load parse");
        assert!((load - 1.23).abs() < 0.0001);
    }

    #[test]
    fn build_pressure_signals_uses_cpu_and_io_deltas() {
        let previous = RawPressureCounters {
            cpu: CpuCounters {
                total: 1_000,
                idle: 200,
            },
            io: IoCounters {
                read_bytes: 1_000,
                write_bytes: 500,
            },
            memory_total_bytes: 16_000,
            memory_available_bytes: 8_000,
            load_avg_1m: 1.0,
            sampled_at_ms: 1_000,
        };
        let current = RawPressureCounters {
            cpu: CpuCounters {
                total: 1_500,
                idle: 300,
            },
            io: IoCounters {
                read_bytes: 3_000,
                write_bytes: 1_500,
            },
            memory_total_bytes: 16_000,
            memory_available_bytes: 4_000,
            load_avg_1m: 2.0,
            sampled_at_ms: 2_000,
        };

        let sample = build_pressure_signals(previous, current);
        assert!((sample.cpu_pct - 80.0).abs() < 0.0001);
        assert!((sample.io_bytes_per_sec - 3_000.0).abs() < 0.0001);
        assert!((sample.memory_pct - 75.0).abs() < 0.0001);
        assert!((sample.load_avg_1m - 2.0).abs() < 0.0001);
    }

    // ── PID File ──

    #[test]
    fn pid_file_acquire_and_release() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("test.pid");
        let mut pid_file = PidFile::new(&path);

        pid_file.acquire("0.1.0").expect("acquire");
        assert!(path.exists());

        // Read back.
        let contents = pid_file.read().expect("read");
        assert_eq!(contents.pid, std::process::id());
        assert_eq!(contents.version, "0.1.0");

        // Second acquire should fail (same PID is alive).
        let mut pid_file2 = PidFile::new(&path);
        assert!(pid_file2.acquire("0.1.0").is_err());

        // Release.
        pid_file.release();
        assert!(!path.exists());
    }

    #[test]
    fn pid_file_stale_recovery() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("test.pid");

        // Write a PID file with a dead PID.
        let mut contents = PidFileContents::current("0.1.0");
        contents.pid = 999_999_999;
        let json = serde_json::to_string_pretty(&contents).unwrap();
        std::fs::write(&path, json.as_bytes()).unwrap();

        // Should recover and acquire.
        let mut pid_file = PidFile::new(&path);
        pid_file.acquire("0.2.0").expect("recover stale");

        let new_contents = pid_file.read().expect("read");
        assert_eq!(new_contents.pid, std::process::id());
        assert_eq!(new_contents.version, "0.2.0");
    }

    #[test]
    fn pid_file_drop_releases() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("test.pid");

        {
            let mut pid_file = PidFile::new(&path);
            pid_file.acquire("0.1.0").expect("acquire");
            assert!(path.exists());
        } // Drop releases.

        assert!(!path.exists());
    }

    #[test]
    fn pid_file_default_path() {
        let path = PidFile::default_path();
        assert!(path.to_string_lossy().contains("fsfs.pid"));
    }

    // ── WatchdogConfig ──

    #[test]
    fn watchdog_backoff_exponential() {
        let config = WatchdogConfig::default();
        let d0 = config.backoff_delay(0);
        let d1 = config.backoff_delay(1);
        let d2 = config.backoff_delay(2);

        assert_eq!(d0, Duration::from_secs(5));
        assert_eq!(d1, Duration::from_secs(10));
        assert_eq!(d2, Duration::from_secs(20));
    }

    #[test]
    fn watchdog_backoff_caps_at_max() {
        let config = WatchdogConfig {
            initial_backoff_ms: 30_000,
            max_backoff_ms: 60_000,
            backoff_multiplier: 2.0,
            ..WatchdogConfig::default()
        };
        let d2 = config.backoff_delay(2);
        assert_eq!(d2, Duration::from_mins(1));
    }

    #[test]
    fn watchdog_exhaustion() {
        let config = WatchdogConfig {
            max_restarts: 3,
            ..WatchdogConfig::default()
        };
        assert!(!config.is_exhausted(2));
        assert!(config.is_exhausted(3));
    }

    // ── Resource Limits ──

    #[test]
    fn resource_limits_no_violations_when_unlimited() {
        let limits = ResourceLimits::default();
        let usage = ResourceUsage {
            rss_bytes: Some(1_000_000_000),
            thread_count: Some(100),
            open_fds: Some(5000),
            index_bytes: Some(10_000_000_000),
            ..ResourceUsage::default()
        };
        assert!(limits.check(&usage).is_empty());
    }

    #[test]
    fn resource_limits_detects_violations() {
        let limits = ResourceLimits {
            max_rss_bytes: 100_000,
            max_threads: 10,
            max_open_fds: 100,
            max_index_bytes: 1_000,
        };
        let usage = ResourceUsage {
            rss_bytes: Some(200_000),
            thread_count: Some(20),
            open_fds: Some(50), // Under limit.
            index_bytes: Some(2_000),
            ..ResourceUsage::default()
        };
        let violations = limits.check(&usage);
        assert_eq!(violations.len(), 3);
        assert_eq!(violations[0].resource, "rss_bytes");
        assert_eq!(violations[1].resource, "threads");
        assert_eq!(violations[2].resource, "index_bytes");
    }

    #[test]
    fn resource_limits_use_index_breakdown_when_total_missing() {
        let limits = ResourceLimits {
            max_index_bytes: 1_000,
            ..ResourceLimits::default()
        };
        let usage = ResourceUsage {
            vector_index_bytes: Some(400),
            lexical_index_bytes: Some(350),
            catalog_bytes: Some(200),
            embedding_cache_bytes: Some(150),
            ..ResourceUsage::default()
        };
        let violations = limits.check(&usage);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].resource, "index_bytes");
        assert_eq!(violations[0].actual, 1_100);
    }

    #[test]
    fn resource_limits_none_usage_no_violation() {
        let limits = ResourceLimits {
            max_rss_bytes: 100_000,
            ..ResourceLimits::default()
        };
        let usage = ResourceUsage::default();
        assert!(limits.check(&usage).is_empty());
    }

    #[test]
    fn disk_budget_policy_stages_actions() {
        let policy = DiskBudgetPolicy::default();

        let normal = policy.evaluate(700, 1_000);
        assert_eq!(normal.stage, DiskBudgetStage::Normal);
        assert_eq!(normal.action, DiskBudgetAction::Continue);
        assert_eq!(
            normal.reason_code,
            FsfsReasonCode::DEGRADE_DISK_WITHIN_BUDGET
        );

        let near = policy.evaluate(900, 1_000);
        assert_eq!(near.stage, DiskBudgetStage::NearLimit);
        assert_eq!(near.action, DiskBudgetAction::ThrottleIngest);
        assert_eq!(near.reason_code, FsfsReasonCode::DEGRADE_DISK_NEAR_BUDGET);

        let over = policy.evaluate(1_020, 1_000);
        assert_eq!(over.stage, DiskBudgetStage::OverLimit);
        assert_eq!(over.action, DiskBudgetAction::EvictLowUtility);
        assert_eq!(over.reason_code, FsfsReasonCode::DEGRADE_DISK_OVER_BUDGET);

        let critical = policy.evaluate(1_250, 1_000);
        assert_eq!(critical.stage, DiskBudgetStage::Critical);
        assert_eq!(critical.action, DiskBudgetAction::PauseWrites);
        assert_eq!(critical.reason_code, FsfsReasonCode::DEGRADE_DISK_CRITICAL);
    }

    #[test]
    fn lifecycle_tracker_evaluates_index_budget_from_limits() {
        let limits = ResourceLimits {
            max_index_bytes: 1_000,
            ..ResourceLimits::default()
        };
        let tracker = LifecycleTracker::new(WatchdogConfig::default(), limits);

        let near = tracker
            .evaluate_index_budget(900)
            .expect("budget should be configured");
        assert_eq!(near.stage, DiskBudgetStage::NearLimit);

        let unbounded = LifecycleTracker::new(WatchdogConfig::default(), ResourceLimits::default());
        assert!(unbounded.evaluate_index_budget(900).is_none());
    }

    #[test]
    fn lifecycle_tracker_evaluates_usage_budget_from_breakdown() {
        let tracker = LifecycleTracker::new(
            WatchdogConfig::default(),
            ResourceLimits {
                max_index_bytes: 1_000,
                ..ResourceLimits::default()
            },
        );
        let usage = ResourceUsage {
            vector_index_bytes: Some(400),
            lexical_index_bytes: Some(300),
            catalog_bytes: Some(150),
            embedding_cache_bytes: Some(150),
            ..ResourceUsage::default()
        };

        let snapshot = tracker
            .evaluate_usage_budget(&usage)
            .expect("usage budget should be available");
        assert_eq!(snapshot.used_bytes, 1_000);
        assert_eq!(snapshot.stage, DiskBudgetStage::NearLimit);
    }

    #[test]
    fn index_footprint_advisor_policy_suite_classifies_thresholds() {
        let policy = IndexFootprintAdvisorPolicy::default();

        let small = index_footprint_advisor_small_fixture();
        assert_eq!(
            policy.classify(small.total_bytes, small.budget_bytes, &small.measurements),
            IndexFootprintScenario::Small
        );

        let fragmented = index_footprint_advisor_fragmented_fixture();
        assert_eq!(
            policy.classify(
                fragmented.total_bytes,
                fragmented.budget_bytes,
                &fragmented.measurements,
            ),
            IndexFootprintScenario::Fragmented
        );

        let oversized = index_footprint_advisor_oversized_fixture();
        assert_eq!(
            policy.classify(
                oversized.total_bytes,
                oversized.budget_bytes,
                &oversized.measurements,
            ),
            IndexFootprintScenario::Oversized
        );
    }

    #[test]
    fn index_footprint_advisor_policy_suite_selects_deterministic_actions() {
        let fragmented = index_footprint_advisor_fragmented_fixture();
        let vector = fragmented
            .recommendations
            .iter()
            .find(|recommendation| recommendation.domain == IndexFootprintDomain::VectorIndex)
            .expect("vector recommendation");
        assert_eq!(vector.action, IndexFootprintAdvisorAction::Compaction);
        assert_eq!(vector.risk, IndexFootprintAdvisorRisk::Low);
        assert!(vector.operator_command.contains("--dry-run"));

        let lexical = fragmented
            .recommendations
            .iter()
            .find(|recommendation| recommendation.domain == IndexFootprintDomain::LexicalIndex)
            .expect("lexical recommendation");
        assert_eq!(lexical.action, IndexFootprintAdvisorAction::Rebuild);
        assert_eq!(lexical.risk, IndexFootprintAdvisorRisk::Medium);
        assert!(lexical.operator_command.contains("--dry-run"));

        let oversized = index_footprint_advisor_oversized_fixture();
        let model_cache = oversized
            .recommendations
            .iter()
            .find(|recommendation| recommendation.domain == IndexFootprintDomain::ModelCache)
            .expect("model-cache recommendation");
        assert_eq!(
            model_cache.action,
            IndexFootprintAdvisorAction::FeatureAdjustment
        );
        assert_eq!(model_cache.risk, IndexFootprintAdvisorRisk::High);
        assert!(model_cache.operator_command.contains("--dry-run"));
    }

    #[test]
    fn index_footprint_advisor_policy_suite_fixtures_validate() {
        index_footprint_advisor_contract_definition()
            .validate()
            .expect("contract should validate");

        for report in [
            index_footprint_advisor_small_fixture(),
            index_footprint_advisor_fragmented_fixture(),
            index_footprint_advisor_oversized_fixture(),
        ] {
            report.validate().expect("advisor report should validate");
            assert!(report.dry_run);
            assert!(!report.automatic_deletion_allowed);
            assert_eq!(
                report.summary.projected_savings_bytes,
                report
                    .recommendations
                    .iter()
                    .map(|recommendation| recommendation.projected_savings_bytes)
                    .sum::<u64>()
            );
            assert!(report.recommendations.iter().all(|recommendation| {
                recommendation
                    .replay_command
                    .contains("index_footprint_advisor_policy_suite")
            }));
        }
    }

    // ── LifecycleTracker ──

    #[test]
    fn lifecycle_tracker_phase_transitions() {
        let tracker = LifecycleTracker::new(WatchdogConfig::default(), ResourceLimits::default());

        assert_eq!(tracker.current_phase(), DaemonPhase::Initializing);

        tracker.transition_to(DaemonPhase::StartingUp);
        assert_eq!(tracker.current_phase(), DaemonPhase::StartingUp);

        tracker.transition_to(DaemonPhase::Running);
        assert_eq!(tracker.current_phase(), DaemonPhase::Running);
    }

    #[test]
    fn lifecycle_tracker_subsystem_health() {
        let tracker = LifecycleTracker::new(WatchdogConfig::default(), ResourceLimits::default());

        tracker.register_subsystem(SubsystemId::Crawler);
        tracker.register_subsystem(SubsystemId::QueryServer);

        tracker.update_subsystem(SubsystemId::Crawler, HealthStatus::Healthy, None);
        tracker.update_subsystem(
            SubsystemId::QueryServer,
            HealthStatus::Failed,
            Some("connection refused".into()),
        );

        let status = tracker.status();
        assert_eq!(status.subsystems.len(), 2);
        assert_eq!(status.total_errors, 1);
    }

    #[test]
    fn lifecycle_tracker_panic_recovery() {
        let tracker = LifecycleTracker::new(WatchdogConfig::default(), ResourceLimits::default());

        tracker.register_subsystem(SubsystemId::EmbedFast);
        tracker.record_panic_recovery(SubsystemId::EmbedFast, "stack overflow");

        let status = tracker.status();
        assert_eq!(status.total_panics_recovered, 1);

        let sub = status
            .subsystems
            .iter()
            .find(|s| s.id == SubsystemId::EmbedFast)
            .unwrap();
        assert_eq!(sub.restart_count, 1);
        assert_eq!(sub.status, HealthStatus::Degraded);
        assert_eq!(sub.last_error.as_deref(), Some("stack overflow"));
    }

    #[test]
    fn lifecycle_tracker_should_restart() {
        let config = WatchdogConfig {
            max_restarts: 2,
            initial_backoff_ms: 1000,
            ..WatchdogConfig::default()
        };
        let tracker = LifecycleTracker::new(config, ResourceLimits::default());

        tracker.register_subsystem(SubsystemId::Crawler);
        tracker.update_subsystem(SubsystemId::Crawler, HealthStatus::Failed, None);

        // First failure: should restart with backoff.
        let delay = tracker.should_restart(SubsystemId::Crawler);
        assert!(delay.is_some());
        assert_eq!(delay.unwrap(), Duration::from_secs(1));

        // Simulate restart + second failure.
        tracker.record_panic_recovery(SubsystemId::Crawler, "crash");
        tracker.update_subsystem(SubsystemId::Crawler, HealthStatus::Failed, None);
        let delay = tracker.should_restart(SubsystemId::Crawler);
        assert!(delay.is_some());

        // Simulate another restart + third failure (exhausted).
        tracker.record_panic_recovery(SubsystemId::Crawler, "crash again");
        tracker.update_subsystem(SubsystemId::Crawler, HealthStatus::Failed, None);
        let delay = tracker.should_restart(SubsystemId::Crawler);
        assert!(delay.is_none(), "should be exhausted");
    }

    #[test]
    fn lifecycle_tracker_healthy_subsystem_not_restarted() {
        let tracker = LifecycleTracker::new(WatchdogConfig::default(), ResourceLimits::default());
        tracker.register_subsystem(SubsystemId::Crawler);
        tracker.update_subsystem(SubsystemId::Crawler, HealthStatus::Healthy, None);
        assert!(tracker.should_restart(SubsystemId::Crawler).is_none());
    }

    #[test]
    fn lifecycle_tracker_status_snapshot() {
        let tracker = LifecycleTracker::new(WatchdogConfig::default(), ResourceLimits::default());
        tracker.transition_to(DaemonPhase::Running);
        tracker.register_subsystem(SubsystemId::Crawler);

        let status = tracker.status();
        assert_eq!(status.phase, DaemonPhase::Running);
        assert_eq!(status.pid, std::process::id());
        assert!(status.uptime_secs < 5);
        assert_eq!(status.subsystems.len(), 1);
    }

    #[test]
    fn lifecycle_tracker_status_includes_last_resource_usage() {
        let tracker = LifecycleTracker::new(
            WatchdogConfig::default(),
            ResourceLimits {
                max_index_bytes: 2_000,
                ..ResourceLimits::default()
            },
        );
        let breakdown = IndexStorageBreakdown {
            vector_index_bytes: 500,
            lexical_index_bytes: 600,
            catalog_bytes: 200,
            embedding_cache_bytes: 100,
        };
        tracker.set_resource_usage(ResourceUsage::from_index_storage(breakdown));

        let status = tracker.status();
        assert_eq!(status.resources.index_bytes, Some(1_400));
        assert_eq!(status.resources.vector_index_bytes, Some(500));
        assert_eq!(status.resources.lexical_index_bytes, Some(600));
        assert_eq!(status.resources.catalog_bytes, Some(200));
        assert_eq!(status.resources.embedding_cache_bytes, Some(100));
    }

    #[test]
    fn lifecycle_tracker_status_includes_disk_budget_snapshot() {
        let tracker = LifecycleTracker::new(
            WatchdogConfig::default(),
            ResourceLimits {
                max_index_bytes: 1_000,
                ..ResourceLimits::default()
            },
        );
        tracker.set_resource_usage(ResourceUsage {
            index_bytes: Some(1_100),
            ..ResourceUsage::default()
        });

        let status = tracker.status();
        let snapshot = status
            .disk_budget
            .expect("disk budget snapshot should be present");
        assert_eq!(snapshot.budget_bytes, 1_000);
        assert_eq!(snapshot.used_bytes, 1_100);
        assert_eq!(snapshot.stage, DiskBudgetStage::Critical);
        assert_eq!(snapshot.action, DiskBudgetAction::PauseWrites);
        assert_eq!(snapshot.reason_code, FsfsReasonCode::DEGRADE_DISK_CRITICAL);
        assert_eq!(
            status.disk_budget_reason_code.as_deref(),
            Some(FsfsReasonCode::DEGRADE_DISK_CRITICAL)
        );
    }

    #[test]
    fn daemon_status_serializes_to_json() {
        let status = DaemonStatus {
            phase: DaemonPhase::Running,
            pid: 12345,
            started_at: "2026-02-14T06:00:00Z".into(),
            uptime_secs: 3600,
            subsystems: vec![SubsystemHealth {
                id: SubsystemId::Crawler,
                status: HealthStatus::Healthy,
                restart_count: 0,
                last_error: None,
                last_healthy_at: Some("2026-02-14T06:59:00Z".into()),
                status_since: "2026-02-14T06:00:00Z".into(),
            }],
            total_errors: 0,
            total_panics_recovered: 0,
            resources: ResourceUsage::default(),
            disk_budget: None,
            disk_budget_reason_code: None,
        };

        let json = serde_json::to_string_pretty(&status).unwrap();
        assert!(json.contains("\"running\""));
        assert!(json.contains("12345"));
        assert!(json.contains("\"crawler\""));
    }

    #[test]
    fn daemon_status_serializes_disk_budget_reason_code() {
        let status = DaemonStatus {
            phase: DaemonPhase::Running,
            pid: 12345,
            started_at: "2026-02-14T06:00:00Z".into(),
            uptime_secs: 3600,
            subsystems: vec![],
            total_errors: 0,
            total_panics_recovered: 0,
            resources: ResourceUsage {
                index_bytes: Some(1_100),
                ..ResourceUsage::default()
            },
            disk_budget: Some(DiskBudgetSnapshot {
                budget_bytes: 1_000,
                used_bytes: 1_100,
                usage_per_mille: 1_100,
                stage: DiskBudgetStage::Critical,
                action: DiskBudgetAction::PauseWrites,
                reason_code: FsfsReasonCode::DEGRADE_DISK_CRITICAL,
            }),
            disk_budget_reason_code: Some(FsfsReasonCode::DEGRADE_DISK_CRITICAL.to_owned()),
        };

        let json = serde_json::to_string(&status).expect("status should serialize");
        assert!(json.contains("\"disk_budget_reason_code\":\"degrade.disk.critical\""));
        assert!(!json.contains("\"reason_code\":"));
    }

    // ── Date Formatting ──

    #[test]
    fn epoch_days_to_ymd_epoch() {
        let (y, m, d) = epoch_days_to_ymd(0);
        assert_eq!((y, m, d), (1970, 1, 1));
    }

    #[test]
    fn epoch_days_to_ymd_2026() {
        // 2026-02-14 is day 20498 since epoch.
        let (y, m, d) = epoch_days_to_ymd(20498);
        assert_eq!((y, m, d), (2026, 2, 14));
    }

    #[test]
    fn format_epoch_secs_basic() {
        let s = format_epoch_secs(0);
        assert_eq!(s, "1970-01-01T00:00:00Z");
    }

    // ── LimitViolation Display ──

    #[test]
    fn limit_violation_display() {
        let v = LimitViolation {
            resource: "rss_bytes",
            limit: 100,
            actual: 200,
        };
        assert_eq!(format!("{v}"), "rss_bytes: 200 exceeds limit 100");
    }
}
