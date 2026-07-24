//! Incremental change detection contract types for fsfs.
//!
//! This module defines the data structures for the fsfs incremental change
//! detection contract v1, which specifies:
//! - mtime/size/hash tradeoff policy
//! - rename/move detection semantics
//! - crash/restart recovery behavior
//! - stale-state reconciliation guarantees

use serde::{Deserialize, Deserializer, Serialize, de};
use std::collections::{BTreeMap, BTreeSet};

// ─── Kind Constants ──────────────────────────────────────────────────────────

pub const KIND_CONTRACT_DEFINITION: &str = "fsfs_incremental_change_detection_contract_definition";
pub const KIND_CHANGE_DECISION: &str = "fsfs_incremental_change_decision";
pub const KIND_RECOVERY_CHECKPOINT: &str = "fsfs_incremental_recovery_checkpoint";
pub const KIND_INDEX_FRESHNESS_AUDIT_REPORT: &str = "fsfs_index_freshness_audit_report";
pub const KIND_INDEX_FRESHNESS_REPAIR_PLAN: &str = "fsfs_index_freshness_repair_plan";
pub const CONTRACT_VERSION: u32 = 1;

// ─── Event Types ─────────────────────────────────────────────────────────────

pub const EVENT_TYPE_CREATE: &str = "create";
pub const EVENT_TYPE_MODIFY: &str = "modify";
pub const EVENT_TYPE_DELETE: &str = "delete";
pub const EVENT_TYPE_RENAME: &str = "rename";
pub const EVENT_TYPE_MOVE: &str = "move";
pub const EVENT_TYPE_RECONCILE: &str = "reconcile";

// ─── Detection Modes ─────────────────────────────────────────────────────────

pub const DETECTION_MODE_FASTPATH: &str = "mtime_size_fastpath";
pub const DETECTION_MODE_HASH_CONFIRM: &str = "hash_confirm";
pub const DETECTION_MODE_FULL_RECONCILE: &str = "full_reconcile";
pub const DETECTION_MODE_JOURNAL_REPLAY: &str = "journal_replay";

// ─── Queue Actions ───────────────────────────────────────────────────────────

pub const QUEUE_ACTION_ENQUEUE_EMBED: &str = "enqueue_embed";
pub const QUEUE_ACTION_SKIP_NO_CHANGE: &str = "skip_no_change";
pub const QUEUE_ACTION_MARK_STALE: &str = "mark_stale";
pub const QUEUE_ACTION_RECONCILE_FULL: &str = "reconcile_full";
pub const QUEUE_ACTION_DROP_MISSING: &str = "drop_missing";

pub const ACTION_ON_RESTART_REPLAY_PENDING: &str = "replay_pending";
pub const ACTION_ON_RESTART_FORCE_RECONCILE: &str = "force_reconcile";
pub const ACTION_ON_RESTART_RESUME_TAIL: &str = "resume_tail";

// ─── Evaluation Input ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncrementalEventType {
    Create,
    Modify,
    Delete,
    Rename,
    Move,
    Reconcile,
}

impl IncrementalEventType {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Create => EVENT_TYPE_CREATE,
            Self::Modify => EVENT_TYPE_MODIFY,
            Self::Delete => EVENT_TYPE_DELETE,
            Self::Rename => EVENT_TYPE_RENAME,
            Self::Move => EVENT_TYPE_MOVE,
            Self::Reconcile => EVENT_TYPE_RECONCILE,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChangeEvaluationInput {
    pub path: String,
    pub event_type: IncrementalEventType,
    pub previous_state: Option<FileState>,
    pub current_state: Option<FileState>,
    pub rename_from: Option<String>,
    pub rename_to: Option<String>,
    pub fastpath_skips: u32,
}

impl ChangeEvaluationInput {
    #[must_use]
    pub fn new(path: impl Into<String>, event_type: IncrementalEventType) -> Self {
        Self {
            path: path.into(),
            event_type,
            previous_state: None,
            current_state: None,
            rename_from: None,
            rename_to: None,
            fastpath_skips: 0,
        }
    }

    #[must_use]
    pub fn with_previous_state(mut self, state: FileState) -> Self {
        self.previous_state = Some(state);
        self
    }

    #[must_use]
    pub fn with_current_state(mut self, state: FileState) -> Self {
        self.current_state = Some(state);
        self
    }

    #[must_use]
    pub fn with_rename_paths(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.rename_from = Some(from.into());
        self.rename_to = Some(to.into());
        self
    }

    #[must_use]
    pub const fn with_fastpath_skips(mut self, skips: u32) -> Self {
        self.fastpath_skips = skips;
        self
    }
}

// ─── Policy Structs ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FastpathPolicy {
    pub mtime_granularity_ns: u64,
    pub require_size_change: bool,
    pub hash_on_mtime_only: bool,
    pub max_fastpath_skips: u32,
}

impl Default for FastpathPolicy {
    fn default() -> Self {
        Self {
            mtime_granularity_ns: 1_000_000, // 1ms
            require_size_change: false,
            hash_on_mtime_only: true,
            max_fastpath_skips: 10,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HashPolicy {
    pub algorithm: String,
    pub sample_prefix_bytes: u32,
    pub full_hash_threshold_bytes: u32,
}

impl Default for HashPolicy {
    fn default() -> Self {
        Self {
            algorithm: "sha256".to_owned(),
            sample_prefix_bytes: 4096,
            full_hash_threshold_bytes: 1_048_576, // 1MB
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RenameMovePolicy {
    pub identity_keys: Vec<String>,
    pub same_device_rename_preserves_identity: bool,
    pub cross_device_move: String,
}

impl Default for RenameMovePolicy {
    fn default() -> Self {
        Self {
            identity_keys: vec!["inode".to_owned(), "content_hash".to_owned()],
            same_device_rename_preserves_identity: true,
            cross_device_move: "hash_confirm".to_owned(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RecoveryPolicy {
    pub journal_required: bool,
    pub replay_order: String,
    pub pending_ttl_seconds: u32,
}

impl Default for RecoveryPolicy {
    fn default() -> Self {
        Self {
            journal_required: true,
            replay_order: "sequence_asc".to_owned(),
            pending_ttl_seconds: 3600,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReconciliationPolicy {
    pub full_scan_interval_seconds: u32,
    pub stale_after_seconds: u32,
    pub orphan_entry_action: String,
}

impl Default for ReconciliationPolicy {
    fn default() -> Self {
        Self {
            full_scan_interval_seconds: 86400, // 24 hours
            stale_after_seconds: 3600,
            orphan_entry_action: "mark_stale".to_owned(),
        }
    }
}

// ─── Contract Definition ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct IncrementalChangeDetectionContractDefinition {
    pub kind: String,
    pub v: u32,
    pub fastpath_policy: FastpathPolicy,
    pub hash_policy: HashPolicy,
    pub rename_move_policy: RenameMovePolicy,
    pub recovery_policy: RecoveryPolicy,
    pub reconciliation_policy: ReconciliationPolicy,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawIncrementalChangeDetectionContractDefinition {
    kind: String,
    v: u32,
    fastpath_policy: FastpathPolicy,
    hash_policy: HashPolicy,
    rename_move_policy: RenameMovePolicy,
    recovery_policy: RecoveryPolicy,
    reconciliation_policy: ReconciliationPolicy,
}

impl Default for IncrementalChangeDetectionContractDefinition {
    fn default() -> Self {
        Self {
            kind: KIND_CONTRACT_DEFINITION.to_owned(),
            v: CONTRACT_VERSION,
            fastpath_policy: FastpathPolicy::default(),
            hash_policy: HashPolicy::default(),
            rename_move_policy: RenameMovePolicy::default(),
            recovery_policy: RecoveryPolicy::default(),
            reconciliation_policy: ReconciliationPolicy::default(),
        }
    }
}

impl<'de> Deserialize<'de> for IncrementalChangeDetectionContractDefinition {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawIncrementalChangeDetectionContractDefinition::deserialize(deserializer)?;
        let contract = Self {
            kind: raw.kind,
            v: raw.v,
            fastpath_policy: raw.fastpath_policy,
            hash_policy: raw.hash_policy,
            rename_move_policy: raw.rename_move_policy,
            recovery_policy: raw.recovery_policy,
            reconciliation_policy: raw.reconciliation_policy,
        };
        contract.validate().map_err(de::Error::custom)?;
        Ok(contract)
    }
}

// ─── File State ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FileState {
    pub file_id: String,
    pub size_bytes: u64,
    pub mtime_ns: u64,
    pub content_hash: String,
}

impl FileState {
    /// Check if mtime changed beyond the granularity threshold.
    #[must_use]
    pub fn mtime_changed(&self, other: &Self, granularity_ns: u64) -> bool {
        let diff = self.mtime_ns.abs_diff(other.mtime_ns);
        diff >= granularity_ns
    }

    /// Check if size changed.
    #[must_use]
    pub fn size_changed(&self, other: &Self) -> bool {
        self.size_bytes != other.size_bytes
    }

    /// Check if content hash differs.
    #[must_use]
    pub fn content_changed(&self, other: &Self) -> bool {
        self.content_hash != other.content_hash
    }
}

// ─── Change Decision ─────────────────────────────────────────────────────────

/// Decision artifact emitted after evaluating a file change event.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct IncrementalChangeDecision {
    pub kind: String,
    pub v: u32,
    pub path: String,
    pub event_type: String,
    pub detection_mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_state: Option<FileState>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_state: Option<FileState>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rename_from: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rename_to: Option<String>,
    pub queue_action: String,
    pub reason_code: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawIncrementalChangeDecision {
    kind: String,
    v: u32,
    path: String,
    event_type: String,
    detection_mode: String,
    previous_state: Option<FileState>,
    current_state: Option<FileState>,
    rename_from: Option<String>,
    rename_to: Option<String>,
    queue_action: String,
    reason_code: String,
    confidence: f64,
}

impl IncrementalChangeDecision {
    /// Create a new decision with default kind and version.
    #[must_use]
    pub fn new(
        path: String,
        event_type: String,
        detection_mode: String,
        previous_state: FileState,
        current_state: FileState,
        queue_action: String,
        reason_code: String,
        confidence: f64,
    ) -> Self {
        Self {
            kind: KIND_CHANGE_DECISION.to_owned(),
            v: CONTRACT_VERSION,
            path,
            event_type,
            detection_mode,
            previous_state: Some(previous_state),
            current_state: Some(current_state),
            rename_from: None,
            rename_to: None,
            queue_action,
            reason_code,
            confidence,
        }
    }

    /// Returns true if this decision requires embedding work.
    #[must_use]
    pub fn requires_embedding(&self) -> bool {
        self.queue_action == QUEUE_ACTION_ENQUEUE_EMBED
    }

    #[must_use]
    pub fn requires_reconcile(&self) -> bool {
        self.queue_action == QUEUE_ACTION_RECONCILE_FULL
    }
}

impl<'de> Deserialize<'de> for IncrementalChangeDecision {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawIncrementalChangeDecision::deserialize(deserializer)?;
        let decision = Self {
            kind: raw.kind,
            v: raw.v,
            path: raw.path,
            event_type: raw.event_type,
            detection_mode: raw.detection_mode,
            previous_state: raw.previous_state,
            current_state: raw.current_state,
            rename_from: raw.rename_from,
            rename_to: raw.rename_to,
            queue_action: raw.queue_action,
            reason_code: raw.reason_code,
            confidence: raw.confidence,
        };
        decision.validate().map_err(de::Error::custom)?;
        Ok(decision)
    }
}

// ─── Recovery Checkpoint ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct IncrementalRecoveryCheckpoint {
    pub kind: String,
    pub v: u32,
    pub checkpoint_id: String,
    pub last_applied_seq: u64,
    pub pending_changes: u32,
    pub journal_clean: bool,
    pub stale_entries: u32,
    pub action_on_restart: String,
    pub reason_code: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawIncrementalRecoveryCheckpoint {
    kind: String,
    v: u32,
    checkpoint_id: String,
    last_applied_seq: u64,
    pending_changes: u32,
    journal_clean: bool,
    stale_entries: u32,
    action_on_restart: String,
    reason_code: String,
}

impl IncrementalRecoveryCheckpoint {
    /// Create a new checkpoint with default kind and version.
    #[must_use]
    pub fn new(
        checkpoint_id: String,
        last_applied_seq: u64,
        pending_changes: u32,
        journal_clean: bool,
        stale_entries: u32,
        action_on_restart: String,
        reason_code: String,
    ) -> Self {
        Self {
            kind: KIND_RECOVERY_CHECKPOINT.to_owned(),
            v: CONTRACT_VERSION,
            checkpoint_id,
            last_applied_seq,
            pending_changes,
            journal_clean,
            stale_entries,
            action_on_restart,
            reason_code,
        }
    }

    /// Returns true if journal replay is needed on restart.
    #[must_use]
    pub fn needs_replay(&self) -> bool {
        !self.journal_clean || self.pending_changes > 0
    }
}

impl<'de> Deserialize<'de> for IncrementalRecoveryCheckpoint {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawIncrementalRecoveryCheckpoint::deserialize(deserializer)?;
        let checkpoint = Self {
            kind: raw.kind,
            v: raw.v,
            checkpoint_id: raw.checkpoint_id,
            last_applied_seq: raw.last_applied_seq,
            pending_changes: raw.pending_changes,
            journal_clean: raw.journal_clean,
            stale_entries: raw.stale_entries,
            action_on_restart: raw.action_on_restart,
            reason_code: raw.reason_code,
        };
        checkpoint.validate().map_err(de::Error::custom)?;
        Ok(checkpoint)
    }
}

// ─── Index Freshness Audit ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexFreshnessAuditVerdict {
    Clean,
    FailClosed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexFreshnessFindingKind {
    MissingCatalog,
    StaleCatalog,
    OrphanCatalog,
    MissingVector,
    MissingLexical,
    DoubleIndexed,
    WatcherCheckpointStale,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexFreshnessRepairActionKind {
    EnqueueReindex,
    MarkStale,
    ReconcileCatalog,
    RebuildVectorMembership,
    RebuildLexicalMembership,
    QuarantineDuplicate,
    ForceWatcherReconcile,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FilesystemSnapshotEntry {
    pub file_key: String,
    pub path: String,
    pub content_hash: String,
    pub observed_at_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogSnapshotEntry {
    pub file_key: String,
    pub path: String,
    pub content_hash: String,
    pub revision: u64,
    pub last_seen_at_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deleted_at_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndexMembershipEntry {
    pub doc_id: String,
    pub file_key: String,
    pub revision: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WatcherCheckpointSnapshot {
    pub checkpoint_id: String,
    pub last_applied_seq: u64,
    pub pending_changes: u32,
    pub watermark_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndexFreshnessAuditInput {
    pub run_id: String,
    pub filesystem: Vec<FilesystemSnapshotEntry>,
    pub catalog: Vec<CatalogSnapshotEntry>,
    pub vector_index: Vec<IndexMembershipEntry>,
    pub lexical_index: Vec<IndexMembershipEntry>,
    pub watcher_checkpoint: WatcherCheckpointSnapshot,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndexFreshnessAuditFinding {
    pub kind: IndexFreshnessFindingKind,
    pub file_key: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    pub reason_code: String,
    pub expected_count: u32,
    pub observed_count: u32,
    pub repair_action: IndexFreshnessRepairActionKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndexFreshnessRepairAction {
    pub action: IndexFreshnessRepairActionKind,
    pub file_key: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    pub reason_code: String,
    pub destructive: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndexFreshnessRepairPlan {
    pub kind: String,
    pub v: u32,
    pub dry_run: bool,
    pub fail_closed: bool,
    pub actions: Vec<IndexFreshnessRepairAction>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndexFreshnessAuditSummary {
    pub verdict: IndexFreshnessAuditVerdict,
    pub filesystem_entries: u32,
    pub catalog_entries: u32,
    pub vector_memberships: u32,
    pub lexical_memberships: u32,
    pub finding_count: u32,
    pub repair_action_count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndexFreshnessAuditReport {
    pub kind: String,
    pub v: u32,
    pub input: IndexFreshnessAuditInput,
    pub summary: IndexFreshnessAuditSummary,
    pub findings: Vec<IndexFreshnessAuditFinding>,
    pub repair_plan: IndexFreshnessRepairPlan,
    pub audit_jsonl_path: String,
    pub summary_json_path: String,
    pub replay_command: String,
}

impl IndexFreshnessAuditReport {
    #[must_use]
    pub fn from_input(input: IndexFreshnessAuditInput) -> Self {
        let findings = classify_index_freshness(&input);
        let repair_plan = IndexFreshnessRepairPlan::from_findings(&findings);
        let verdict = if findings.is_empty() {
            IndexFreshnessAuditVerdict::Clean
        } else {
            IndexFreshnessAuditVerdict::FailClosed
        };
        let summary = IndexFreshnessAuditSummary {
            verdict,
            filesystem_entries: usize_to_u32(input.filesystem.len()),
            catalog_entries: usize_to_u32(input.catalog.len()),
            vector_memberships: usize_to_u32(input.vector_index.len()),
            lexical_memberships: usize_to_u32(input.lexical_index.len()),
            finding_count: usize_to_u32(findings.len()),
            repair_action_count: usize_to_u32(repair_plan.actions.len()),
        };
        let artifact_root = format!("runs/{}/index_freshness", input.run_id);
        let replay_command = format!(
            "scripts/check_fsfs_index_freshness_audit.sh --mode e2e --run-id {}",
            input.run_id
        );

        Self {
            kind: KIND_INDEX_FRESHNESS_AUDIT_REPORT.to_owned(),
            v: CONTRACT_VERSION,
            input,
            summary,
            findings,
            repair_plan,
            audit_jsonl_path: format!("{artifact_root}/audit-events.jsonl"),
            summary_json_path: format!("{artifact_root}/summary.json"),
            replay_command,
        }
    }

    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.summary.verdict == IndexFreshnessAuditVerdict::Clean
    }
}

impl IndexFreshnessRepairPlan {
    #[must_use]
    pub fn from_findings(findings: &[IndexFreshnessAuditFinding]) -> Self {
        let actions = findings
            .iter()
            .map(|finding| IndexFreshnessRepairAction {
                action: finding.repair_action,
                file_key: finding.file_key.clone(),
                path: finding.path.clone(),
                reason_code: finding.reason_code.clone(),
                destructive: false,
            })
            .collect();

        Self {
            kind: KIND_INDEX_FRESHNESS_REPAIR_PLAN.to_owned(),
            v: CONTRACT_VERSION,
            dry_run: true,
            fail_closed: !findings.is_empty(),
            actions,
        }
    }
}

fn classify_index_freshness(input: &IndexFreshnessAuditInput) -> Vec<IndexFreshnessAuditFinding> {
    let filesystem_by_key = filesystem_by_key(&input.filesystem);
    let catalog_by_key = catalog_by_key(&input.catalog);
    let vector_by_file = membership_by_file(&input.vector_index);
    let lexical_by_file = membership_by_file(&input.lexical_index);
    let mut findings = Vec::new();

    for (&file_key, fs_entry) in &filesystem_by_key {
        let Some(catalog_entry) = catalog_by_key.get(file_key) else {
            findings.push(audit_finding(
                IndexFreshnessFindingKind::MissingCatalog,
                file_key,
                Some(&fs_entry.path),
                "FSFS_AUDIT_MISSING_CATALOG",
                1,
                0,
            ));
            continue;
        };

        if catalog_entry.content_hash != fs_entry.content_hash {
            findings.push(audit_finding(
                IndexFreshnessFindingKind::StaleCatalog,
                file_key,
                Some(&fs_entry.path),
                "FSFS_AUDIT_STALE_CATALOG_HASH",
                1,
                1,
            ));
        }

        push_membership_findings(
            &mut findings,
            file_key,
            &fs_entry.path,
            vector_by_file.get(file_key).map_or(0, Vec::len),
            IndexFreshnessFindingKind::MissingVector,
            IndexFreshnessFindingKind::DoubleIndexed,
            "FSFS_AUDIT_MISSING_VECTOR_MEMBERSHIP",
            "FSFS_AUDIT_DOUBLE_VECTOR_MEMBERSHIP",
        );
        push_membership_findings(
            &mut findings,
            file_key,
            &fs_entry.path,
            lexical_by_file.get(file_key).map_or(0, Vec::len),
            IndexFreshnessFindingKind::MissingLexical,
            IndexFreshnessFindingKind::DoubleIndexed,
            "FSFS_AUDIT_MISSING_LEXICAL_MEMBERSHIP",
            "FSFS_AUDIT_DOUBLE_LEXICAL_MEMBERSHIP",
        );
    }

    for (&file_key, catalog_entry) in &catalog_by_key {
        if !filesystem_by_key.contains_key(file_key) && catalog_entry.deleted_at_ms.is_none() {
            findings.push(audit_finding(
                IndexFreshnessFindingKind::OrphanCatalog,
                file_key,
                Some(&catalog_entry.path),
                "FSFS_AUDIT_ORPHAN_CATALOG_ENTRY",
                0,
                1,
            ));
        }
    }

    let newest_filesystem_observation = input
        .filesystem
        .iter()
        .map(|entry| entry.observed_at_ms)
        .max()
        .unwrap_or(0);
    if input.watcher_checkpoint.pending_changes > 0
        || input.watcher_checkpoint.watermark_ms < newest_filesystem_observation
    {
        findings.push(audit_finding(
            IndexFreshnessFindingKind::WatcherCheckpointStale,
            &input.watcher_checkpoint.checkpoint_id,
            None,
            "FSFS_AUDIT_WATCHER_CHECKPOINT_STALE",
            0,
            input.watcher_checkpoint.pending_changes,
        ));
    }

    findings.sort_by(|left, right| {
        left.kind
            .cmp(&right.kind)
            .then_with(|| left.file_key.cmp(&right.file_key))
            .then_with(|| left.reason_code.cmp(&right.reason_code))
    });
    findings
}

fn filesystem_by_key(
    entries: &[FilesystemSnapshotEntry],
) -> BTreeMap<&str, &FilesystemSnapshotEntry> {
    entries
        .iter()
        .map(|entry| (entry.file_key.as_str(), entry))
        .collect()
}

fn catalog_by_key(entries: &[CatalogSnapshotEntry]) -> BTreeMap<&str, &CatalogSnapshotEntry> {
    entries
        .iter()
        .map(|entry| (entry.file_key.as_str(), entry))
        .collect()
}

fn membership_by_file(
    entries: &[IndexMembershipEntry],
) -> BTreeMap<&str, Vec<&IndexMembershipEntry>> {
    let mut by_file: BTreeMap<&str, Vec<&IndexMembershipEntry>> = BTreeMap::new();
    for entry in entries {
        by_file
            .entry(entry.file_key.as_str())
            .or_default()
            .push(entry);
    }
    by_file
}

fn push_membership_findings(
    findings: &mut Vec<IndexFreshnessAuditFinding>,
    file_key: &str,
    path: &str,
    observed_count: usize,
    missing_kind: IndexFreshnessFindingKind,
    duplicate_kind: IndexFreshnessFindingKind,
    missing_reason: &str,
    duplicate_reason: &str,
) {
    if observed_count == 0 {
        findings.push(audit_finding(
            missing_kind,
            file_key,
            Some(path),
            missing_reason,
            1,
            0,
        ));
    } else if observed_count > 1 {
        findings.push(audit_finding(
            duplicate_kind,
            file_key,
            Some(path),
            duplicate_reason,
            1,
            usize_to_u32(observed_count),
        ));
    }
}

fn audit_finding(
    kind: IndexFreshnessFindingKind,
    file_key: &str,
    path: Option<&str>,
    reason_code: &str,
    expected_count: u32,
    observed_count: u32,
) -> IndexFreshnessAuditFinding {
    IndexFreshnessAuditFinding {
        kind,
        file_key: file_key.to_owned(),
        path: path.map(str::to_owned),
        reason_code: reason_code.to_owned(),
        expected_count,
        observed_count,
        repair_action: repair_action_for(kind),
    }
}

fn repair_action_for(kind: IndexFreshnessFindingKind) -> IndexFreshnessRepairActionKind {
    match kind {
        IndexFreshnessFindingKind::MissingCatalog => {
            IndexFreshnessRepairActionKind::ReconcileCatalog
        }
        IndexFreshnessFindingKind::StaleCatalog => IndexFreshnessRepairActionKind::EnqueueReindex,
        IndexFreshnessFindingKind::OrphanCatalog => IndexFreshnessRepairActionKind::MarkStale,
        IndexFreshnessFindingKind::MissingVector => {
            IndexFreshnessRepairActionKind::RebuildVectorMembership
        }
        IndexFreshnessFindingKind::MissingLexical => IndexFreshnessRepairActionKind::EnqueueReindex,
        IndexFreshnessFindingKind::DoubleIndexed => {
            IndexFreshnessRepairActionKind::QuarantineDuplicate
        }
        IndexFreshnessFindingKind::WatcherCheckpointStale => {
            IndexFreshnessRepairActionKind::ForceWatcherReconcile
        }
    }
}

fn usize_to_u32(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

impl IncrementalChangeDetectionContractDefinition {
    /// Evaluate one filesystem change candidate using the contract's
    /// mtime/size/hash, rename/move, delete, and reconcile rules.
    ///
    /// The returned decision is always shaped as a contract v1 decision and is
    /// valid for serialization. Ambiguous or incomplete inputs favor
    /// `reconcile_full` over speculative fast-path skipping.
    #[must_use]
    pub fn evaluate_change(&self, input: &ChangeEvaluationInput) -> IncrementalChangeDecision {
        match input.event_type {
            IncrementalEventType::Create => Self::evaluate_create(input),
            IncrementalEventType::Modify => self.evaluate_modify(input),
            IncrementalEventType::Delete => Self::evaluate_delete(input),
            IncrementalEventType::Rename | IncrementalEventType::Move => {
                self.evaluate_rename_or_move(input)
            }
            IncrementalEventType::Reconcile => Self::reconcile_decision(
                input,
                "FSFS_RECONCILE_REQUESTED",
                1.0,
                IncrementalEventType::Reconcile,
            ),
        }
    }

    fn evaluate_create(input: &ChangeEvaluationInput) -> IncrementalChangeDecision {
        if input.current_state.is_some() {
            return Self::decision(
                input,
                IncrementalEventType::Create,
                DETECTION_MODE_FASTPATH,
                QUEUE_ACTION_ENQUEUE_EMBED,
                "FSFS_NEW_FILE_ENQUEUE_EMBED",
                0.95,
            );
        }
        Self::reconcile_decision(
            input,
            "FSFS_CREATE_STATE_MISSING_RECONCILE",
            0.5,
            IncrementalEventType::Create,
        )
    }

    fn evaluate_modify(&self, input: &ChangeEvaluationInput) -> IncrementalChangeDecision {
        let (Some(previous), Some(current)) = (&input.previous_state, &input.current_state) else {
            return Self::reconcile_decision(
                input,
                "FSFS_MODIFY_STATE_MISSING_RECONCILE",
                0.5,
                IncrementalEventType::Modify,
            );
        };

        let mtime_changed =
            previous.mtime_changed(current, self.fastpath_policy.mtime_granularity_ns);
        let size_changed = previous.size_changed(current);
        let content_changed = previous.content_changed(current);
        let force_hash = input.fastpath_skips >= self.fastpath_policy.max_fastpath_skips;

        if !mtime_changed && !size_changed && !force_hash {
            return Self::decision(
                input,
                IncrementalEventType::Modify,
                DETECTION_MODE_FASTPATH,
                QUEUE_ACTION_SKIP_NO_CHANGE,
                "FSFS_MTIME_SIZE_UNCHANGED",
                0.99,
            );
        }

        if content_changed {
            return Self::decision(
                input,
                IncrementalEventType::Modify,
                DETECTION_MODE_HASH_CONFIRM,
                QUEUE_ACTION_ENQUEUE_EMBED,
                "FSFS_HASH_CONFIRMED_CONTENT_CHANGE",
                0.97,
            );
        }

        let reason = if force_hash {
            "FSFS_FORCED_HASH_CONFIRMED_NO_CHANGE"
        } else {
            "FSFS_HASH_CONFIRMED_NO_CHANGE"
        };
        Self::decision(
            input,
            IncrementalEventType::Modify,
            DETECTION_MODE_HASH_CONFIRM,
            QUEUE_ACTION_SKIP_NO_CHANGE,
            reason,
            0.96,
        )
    }

    fn evaluate_delete(input: &ChangeEvaluationInput) -> IncrementalChangeDecision {
        let action = if input.previous_state.is_some() {
            QUEUE_ACTION_DROP_MISSING
        } else {
            QUEUE_ACTION_MARK_STALE
        };
        let reason = if input.previous_state.is_some() {
            "FSFS_DELETE_DROP_MISSING"
        } else {
            "FSFS_DELETE_PRIOR_STATE_MISSING_MARK_STALE"
        };
        Self::decision(
            input,
            IncrementalEventType::Delete,
            DETECTION_MODE_JOURNAL_REPLAY,
            action,
            reason,
            0.99,
        )
    }

    fn evaluate_rename_or_move(&self, input: &ChangeEvaluationInput) -> IncrementalChangeDecision {
        if input.rename_from.as_deref().is_none_or(str::is_empty)
            || input.rename_to.as_deref().is_none_or(str::is_empty)
        {
            return Self::reconcile_decision(
                input,
                "FSFS_RENAME_PATHS_MISSING_RECONCILE",
                0.5,
                IncrementalEventType::Reconcile,
            );
        }

        let (Some(previous), Some(current)) = (&input.previous_state, &input.current_state) else {
            return Self::reconcile_decision(
                input,
                "FSFS_RENAME_STATE_MISSING_RECONCILE",
                0.5,
                IncrementalEventType::Reconcile,
            );
        };

        if previous.content_changed(current) {
            return Self::decision(
                input,
                input.event_type,
                DETECTION_MODE_HASH_CONFIRM,
                QUEUE_ACTION_ENQUEUE_EMBED,
                "FSFS_HASH_CONFIRMED_CONTENT_CHANGE",
                0.97,
            );
        }

        if previous.file_id == current.file_id
            && self
                .rename_move_policy
                .same_device_rename_preserves_identity
        {
            return Self::decision(
                input,
                input.event_type,
                DETECTION_MODE_FASTPATH,
                QUEUE_ACTION_SKIP_NO_CHANGE,
                "FSFS_RENAME_IDENTITY_PRESERVED",
                0.94,
            );
        }

        if input.event_type == IncrementalEventType::Move
            && self.rename_move_policy.cross_device_move == "treat_as_delete_create"
        {
            return Self::decision(
                input,
                input.event_type,
                DETECTION_MODE_HASH_CONFIRM,
                QUEUE_ACTION_ENQUEUE_EMBED,
                "FSFS_CROSS_DEVICE_MOVE_DELETE_CREATE",
                0.85,
            );
        }

        Self::decision(
            input,
            input.event_type,
            DETECTION_MODE_HASH_CONFIRM,
            QUEUE_ACTION_SKIP_NO_CHANGE,
            "FSFS_MOVE_HASH_CONFIRMED_TRANSFER",
            0.90,
        )
    }

    fn reconcile_decision(
        input: &ChangeEvaluationInput,
        reason_code: &str,
        confidence: f64,
        event_type: IncrementalEventType,
    ) -> IncrementalChangeDecision {
        Self::decision(
            input,
            event_type,
            DETECTION_MODE_FULL_RECONCILE,
            QUEUE_ACTION_RECONCILE_FULL,
            reason_code,
            confidence,
        )
    }

    fn decision(
        input: &ChangeEvaluationInput,
        event_type: IncrementalEventType,
        detection_mode: &str,
        queue_action: &str,
        reason_code: &str,
        confidence: f64,
    ) -> IncrementalChangeDecision {
        IncrementalChangeDecision {
            kind: KIND_CHANGE_DECISION.to_owned(),
            v: CONTRACT_VERSION,
            path: input.path.clone(),
            event_type: event_type.as_str().to_owned(),
            detection_mode: detection_mode.to_owned(),
            previous_state: input.previous_state.clone(),
            current_state: input.current_state.clone(),
            rename_from: input.rename_from.clone(),
            rename_to: input.rename_to.clone(),
            queue_action: queue_action.to_owned(),
            reason_code: reason_code.to_owned(),
            confidence,
        }
    }

    fn validate(&self) -> Result<(), &'static str> {
        validate_kind(
            &self.kind,
            KIND_CONTRACT_DEFINITION,
            "kind must be fsfs_incremental_change_detection_contract_definition",
        )?;
        validate_schema_version(self.v)?;
        self.fastpath_policy.validate()?;
        self.hash_policy.validate()?;
        self.rename_move_policy.validate()?;
        self.recovery_policy.validate()?;
        self.reconciliation_policy.validate()?;
        Ok(())
    }
}

impl FastpathPolicy {
    fn validate(&self) -> Result<(), &'static str> {
        if self.mtime_granularity_ns == 0 {
            return Err("fastpath_policy.mtime_granularity_ns must be >= 1");
        }
        if !self.hash_on_mtime_only {
            return Err("fastpath_policy.hash_on_mtime_only must be true");
        }
        Ok(())
    }
}

impl HashPolicy {
    fn validate(&self) -> Result<(), &'static str> {
        if !matches!(self.algorithm.as_str(), "sha256" | "blake3") {
            return Err("hash_policy.algorithm must be sha256 or blake3");
        }
        if self.full_hash_threshold_bytes == 0 {
            return Err("hash_policy.full_hash_threshold_bytes must be >= 1");
        }
        Ok(())
    }
}

impl RenameMovePolicy {
    fn validate(&self) -> Result<(), &'static str> {
        if self.identity_keys.is_empty() {
            return Err("rename_move_policy.identity_keys must not be empty");
        }
        if !self.same_device_rename_preserves_identity {
            return Err("rename_move_policy.same_device_rename_preserves_identity must be true");
        }

        let mut unique_keys = BTreeSet::new();
        for key in &self.identity_keys {
            if !matches!(key.as_str(), "device" | "inode" | "content_hash") {
                return Err("rename_move_policy.identity_keys contains an unsupported key");
            }
            if !unique_keys.insert(key.as_str()) {
                return Err("rename_move_policy.identity_keys must be unique");
            }
        }

        if !matches!(
            self.cross_device_move.as_str(),
            "treat_as_delete_create" | "hash_confirm"
        ) {
            return Err("rename_move_policy.cross_device_move must match the schema contract");
        }

        Ok(())
    }
}

impl RecoveryPolicy {
    fn validate(&self) -> Result<(), &'static str> {
        if !self.journal_required {
            return Err("recovery_policy.journal_required must be true");
        }
        if self.replay_order != "sequence_asc" {
            return Err("recovery_policy.replay_order must be sequence_asc");
        }
        if self.pending_ttl_seconds == 0 {
            return Err("recovery_policy.pending_ttl_seconds must be >= 1");
        }
        Ok(())
    }
}

impl ReconciliationPolicy {
    fn validate(&self) -> Result<(), &'static str> {
        if self.full_scan_interval_seconds == 0 {
            return Err("reconciliation_policy.full_scan_interval_seconds must be >= 1");
        }
        if self.stale_after_seconds == 0 {
            return Err("reconciliation_policy.stale_after_seconds must be >= 1");
        }
        if !matches!(
            self.orphan_entry_action.as_str(),
            "delete" | "quarantine" | "mark_stale"
        ) {
            return Err("reconciliation_policy.orphan_entry_action must match the schema contract");
        }
        Ok(())
    }
}

impl FileState {
    fn validate(&self) -> Result<(), &'static str> {
        if self.file_id.is_empty() {
            return Err("file_id must not be empty");
        }
        if !is_lower_hex_hash(&self.content_hash) {
            return Err("content_hash must be a 64-character lowercase hex digest");
        }
        Ok(())
    }
}

impl IncrementalChangeDecision {
    fn validate(&self) -> Result<(), &'static str> {
        validate_kind(
            &self.kind,
            KIND_CHANGE_DECISION,
            "kind must be fsfs_incremental_change_decision",
        )?;
        validate_schema_version(self.v)?;
        if self.path.is_empty() {
            return Err("path must not be empty");
        }
        if !matches!(
            self.event_type.as_str(),
            EVENT_TYPE_CREATE
                | EVENT_TYPE_MODIFY
                | EVENT_TYPE_DELETE
                | EVENT_TYPE_RENAME
                | EVENT_TYPE_MOVE
                | EVENT_TYPE_RECONCILE
        ) {
            return Err("event_type must match the schema contract");
        }
        if !matches!(
            self.detection_mode.as_str(),
            DETECTION_MODE_FASTPATH
                | DETECTION_MODE_HASH_CONFIRM
                | DETECTION_MODE_FULL_RECONCILE
                | DETECTION_MODE_JOURNAL_REPLAY
        ) {
            return Err("detection_mode must match the schema contract");
        }
        if !matches!(
            self.queue_action.as_str(),
            QUEUE_ACTION_ENQUEUE_EMBED
                | QUEUE_ACTION_SKIP_NO_CHANGE
                | QUEUE_ACTION_DROP_MISSING
                | QUEUE_ACTION_MARK_STALE
                | QUEUE_ACTION_RECONCILE_FULL
        ) {
            return Err("queue_action must match the schema contract");
        }
        if !is_reason_code(&self.reason_code) {
            return Err("reason_code must match ^FSFS_[A-Z0-9_]+$");
        }
        validate_unit_interval(self.confidence, "confidence must be between 0 and 1")?;
        if let Some(previous_state) = &self.previous_state {
            previous_state.validate()?;
        }
        if let Some(current_state) = &self.current_state {
            current_state.validate()?;
        }

        let rename_requires_paths = matches!(
            self.event_type.as_str(),
            EVENT_TYPE_RENAME | EVENT_TYPE_MOVE
        );
        if rename_requires_paths {
            if self.rename_from.as_deref().is_none_or(str::is_empty) {
                return Err("rename events require rename_from");
            }
            if self.rename_to.as_deref().is_none_or(str::is_empty) {
                return Err("rename events require rename_to");
            }
        }

        if self.event_type == EVENT_TYPE_DELETE
            && !matches!(
                self.queue_action.as_str(),
                QUEUE_ACTION_DROP_MISSING | QUEUE_ACTION_MARK_STALE | QUEUE_ACTION_RECONCILE_FULL
            )
        {
            return Err("delete events must not enqueue embeddings");
        }

        Ok(())
    }
}

impl IncrementalRecoveryCheckpoint {
    fn validate(&self) -> Result<(), &'static str> {
        validate_kind(
            &self.kind,
            KIND_RECOVERY_CHECKPOINT,
            "kind must be fsfs_incremental_recovery_checkpoint",
        )?;
        validate_schema_version(self.v)?;
        if !is_checkpoint_id(&self.checkpoint_id) {
            return Err("checkpoint_id must match ^[a-zA-Z0-9._:-]+$");
        }
        if !matches!(
            self.action_on_restart.as_str(),
            ACTION_ON_RESTART_REPLAY_PENDING
                | ACTION_ON_RESTART_FORCE_RECONCILE
                | ACTION_ON_RESTART_RESUME_TAIL
        ) {
            return Err("action_on_restart must match the schema contract");
        }
        if !is_reason_code(&self.reason_code) {
            return Err("reason_code must match ^FSFS_[A-Z0-9_]+$");
        }
        if self.journal_clean && self.pending_changes != 0 {
            return Err("pending_changes must be zero when journal_clean is true");
        }
        Ok(())
    }
}

fn validate_schema_version(value: u32) -> Result<(), &'static str> {
    match value {
        CONTRACT_VERSION => Ok(()),
        _ => Err("schema version 1"),
    }
}

fn validate_kind(actual: &str, expected: &str, message: &'static str) -> Result<(), &'static str> {
    if actual == expected {
        Ok(())
    } else {
        Err(message)
    }
}

fn validate_unit_interval(value: f64, message: &'static str) -> Result<(), &'static str> {
    if (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(message)
    }
}

fn is_reason_code(value: &str) -> bool {
    value.len() > 5
        && value.starts_with("FSFS_")
        && value
            .chars()
            .all(|ch| ch.is_ascii_uppercase() || ch.is_ascii_digit() || ch == '_')
}

fn is_lower_hex_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_checkpoint_id(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'-'))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use serde_json::json;

    use super::*;

    fn valid_state(file_id: &str, size_bytes: u64, mtime_ns: u64, hash_nibble: &str) -> FileState {
        FileState {
            file_id: file_id.to_owned(),
            size_bytes,
            mtime_ns,
            content_hash: hash_nibble.repeat(64),
        }
    }

    fn fs_entry(
        file_key: &str,
        path: &str,
        hash_nibble: &str,
        observed_at_ms: u64,
    ) -> FilesystemSnapshotEntry {
        FilesystemSnapshotEntry {
            file_key: file_key.to_owned(),
            path: path.to_owned(),
            content_hash: hash_nibble.repeat(64),
            observed_at_ms,
        }
    }

    fn catalog_entry(
        file_key: &str,
        path: &str,
        hash_nibble: &str,
        revision: u64,
        deleted_at_ms: Option<u64>,
    ) -> CatalogSnapshotEntry {
        CatalogSnapshotEntry {
            file_key: file_key.to_owned(),
            path: path.to_owned(),
            content_hash: hash_nibble.repeat(64),
            revision,
            last_seen_at_ms: 1_000,
            deleted_at_ms,
        }
    }

    fn membership(file_key: &str, source: &str, revision: u64) -> IndexMembershipEntry {
        IndexMembershipEntry {
            doc_id: format!("{source}:{file_key}:{revision}"),
            file_key: file_key.to_owned(),
            revision,
        }
    }

    fn clean_audit_input() -> IndexFreshnessAuditInput {
        IndexFreshnessAuditInput {
            run_id: "clean-run".to_owned(),
            filesystem: vec![fs_entry("src/lib.rs", "/workspace/src/lib.rs", "a", 1_000)],
            catalog: vec![catalog_entry(
                "src/lib.rs",
                "/workspace/src/lib.rs",
                "a",
                7,
                None,
            )],
            vector_index: vec![membership("src/lib.rs", "vector", 7)],
            lexical_index: vec![membership("src/lib.rs", "lexical", 7)],
            watcher_checkpoint: WatcherCheckpointSnapshot {
                checkpoint_id: "watcher-clean".to_owned(),
                last_applied_seq: 42,
                pending_changes: 0,
                watermark_ms: 1_000,
            },
        }
    }

    #[test]
    fn default_contract_has_correct_kind_and_version() {
        let contract = IncrementalChangeDetectionContractDefinition::default();
        assert_eq!(contract.kind, KIND_CONTRACT_DEFINITION);
        assert_eq!(contract.v, CONTRACT_VERSION);
    }

    #[test]
    fn default_fastpath_policy_values() {
        let policy = FastpathPolicy::default();
        assert_eq!(policy.mtime_granularity_ns, 1_000_000);
        assert!(policy.hash_on_mtime_only);
        assert_eq!(policy.max_fastpath_skips, 10);
    }

    #[test]
    fn default_hash_policy_uses_sha256() {
        let policy = HashPolicy::default();
        assert_eq!(policy.algorithm, "sha256");
    }

    #[test]
    fn evaluator_skips_unchanged_fastpath_candidate() {
        let contract = IncrementalChangeDetectionContractDefinition::default();
        let state = valid_state("dev1:ino1", 100, 1_000_000_000, "a");
        let decision = contract.evaluate_change(
            &ChangeEvaluationInput::new("/workspace/src/lib.rs", IncrementalEventType::Modify)
                .with_previous_state(state.clone())
                .with_current_state(state),
        );

        assert_eq!(decision.event_type, EVENT_TYPE_MODIFY);
        assert_eq!(decision.detection_mode, DETECTION_MODE_FASTPATH);
        assert_eq!(decision.queue_action, QUEUE_ACTION_SKIP_NO_CHANGE);
        assert_eq!(decision.reason_code, "FSFS_MTIME_SIZE_UNCHANGED");
        assert!(!decision.requires_embedding());

        let json = serde_json::to_string(&decision).expect("serialize decision");
        let parsed: IncrementalChangeDecision =
            serde_json::from_str(&json).expect("decision remains schema-valid");
        assert_eq!(parsed, decision);
    }

    #[test]
    fn evaluator_forces_hash_after_fastpath_budget() {
        let contract = IncrementalChangeDetectionContractDefinition::default();
        let state = valid_state("dev1:ino1", 100, 1_000_000_000, "a");
        let decision = contract.evaluate_change(
            &ChangeEvaluationInput::new("/workspace/src/lib.rs", IncrementalEventType::Modify)
                .with_previous_state(state.clone())
                .with_current_state(state)
                .with_fastpath_skips(contract.fastpath_policy.max_fastpath_skips),
        );

        assert_eq!(decision.detection_mode, DETECTION_MODE_HASH_CONFIRM);
        assert_eq!(decision.queue_action, QUEUE_ACTION_SKIP_NO_CHANGE);
        assert_eq!(decision.reason_code, "FSFS_FORCED_HASH_CONFIRMED_NO_CHANGE");
    }

    #[test]
    fn evaluator_enqueues_hash_confirmed_modify() {
        let contract = IncrementalChangeDetectionContractDefinition::default();
        let previous = valid_state("dev1:ino1", 100, 1_000_000_000, "a");
        let current = valid_state("dev1:ino1", 100, 1_100_000_000, "b");
        let decision = contract.evaluate_change(
            &ChangeEvaluationInput::new("/workspace/src/lib.rs", IncrementalEventType::Modify)
                .with_previous_state(previous)
                .with_current_state(current),
        );

        assert_eq!(decision.detection_mode, DETECTION_MODE_HASH_CONFIRM);
        assert_eq!(decision.queue_action, QUEUE_ACTION_ENQUEUE_EMBED);
        assert_eq!(decision.reason_code, "FSFS_HASH_CONFIRMED_CONTENT_CHANGE");
        assert!(decision.requires_embedding());
    }

    #[test]
    fn evaluator_delete_never_enqueues_embedding() {
        let contract = IncrementalChangeDetectionContractDefinition::default();
        let previous = valid_state("dev1:ino1", 100, 1_000_000_000, "a");
        let decision = contract.evaluate_change(
            &ChangeEvaluationInput::new("/workspace/src/lib.rs", IncrementalEventType::Delete)
                .with_previous_state(previous),
        );

        assert_eq!(decision.event_type, EVENT_TYPE_DELETE);
        assert_eq!(decision.queue_action, QUEUE_ACTION_DROP_MISSING);
        assert!(!decision.requires_embedding());
    }

    #[test]
    fn evaluator_preserves_same_device_rename_identity() {
        let contract = IncrementalChangeDetectionContractDefinition::default();
        let previous = valid_state("dev1:ino1", 100, 1_000_000_000, "a");
        let current = valid_state("dev1:ino1", 100, 1_100_000_000, "a");
        let decision = contract.evaluate_change(
            &ChangeEvaluationInput::new("/workspace/src/new.rs", IncrementalEventType::Rename)
                .with_previous_state(previous)
                .with_current_state(current)
                .with_rename_paths("/workspace/src/old.rs", "/workspace/src/new.rs"),
        );

        assert_eq!(decision.event_type, EVENT_TYPE_RENAME);
        assert_eq!(decision.detection_mode, DETECTION_MODE_FASTPATH);
        assert_eq!(decision.queue_action, QUEUE_ACTION_SKIP_NO_CHANGE);
        assert_eq!(decision.reason_code, "FSFS_RENAME_IDENTITY_PRESERVED");
        assert_eq!(
            decision.rename_from.as_deref(),
            Some("/workspace/src/old.rs")
        );
        assert_eq!(decision.rename_to.as_deref(), Some("/workspace/src/new.rs"));
    }

    #[test]
    fn evaluator_reconciles_rename_without_paths() {
        let contract = IncrementalChangeDetectionContractDefinition::default();
        let previous = valid_state("dev1:ino1", 100, 1_000_000_000, "a");
        let current = valid_state("dev1:ino1", 100, 1_100_000_000, "a");
        let decision = contract.evaluate_change(
            &ChangeEvaluationInput::new("/workspace/src/new.rs", IncrementalEventType::Rename)
                .with_previous_state(previous)
                .with_current_state(current),
        );

        assert_eq!(decision.event_type, EVENT_TYPE_RECONCILE);
        assert_eq!(decision.queue_action, QUEUE_ACTION_RECONCILE_FULL);
        assert!(decision.requires_reconcile());

        let json = serde_json::to_string(&decision).expect("serialize decision");
        serde_json::from_str::<IncrementalChangeDecision>(&json)
            .expect("missing rename paths produce a valid reconcile decision");
    }

    #[test]
    fn index_freshness_audit_clean_fixture_has_no_findings() {
        let report = IndexFreshnessAuditReport::from_input(clean_audit_input());

        assert!(report.is_clean());
        assert_eq!(report.kind, KIND_INDEX_FRESHNESS_AUDIT_REPORT);
        assert!(report.findings.is_empty());
        assert!(report.repair_plan.dry_run);
        assert!(!report.repair_plan.fail_closed);
        assert!(report.repair_plan.actions.is_empty());
        assert_eq!(
            report.audit_jsonl_path,
            "runs/clean-run/index_freshness/audit-events.jsonl"
        );
        assert_eq!(
            report.summary_json_path,
            "runs/clean-run/index_freshness/summary.json"
        );
        assert!(
            report
                .replay_command
                .contains("scripts/check_fsfs_index_freshness_audit.sh --mode e2e")
        );

        let json = serde_json::to_string(&report).expect("serialize audit report");
        let parsed: IndexFreshnessAuditReport =
            serde_json::from_str(&json).expect("audit report roundtrip");
        assert_eq!(parsed, report);
    }

    #[test]
    fn index_freshness_audit_classifies_drift_cases_deterministically() {
        let report = IndexFreshnessAuditReport::from_input(IndexFreshnessAuditInput {
            run_id: "drift-run".to_owned(),
            filesystem: vec![
                fs_entry("src/clean.rs", "/workspace/src/clean.rs", "a", 1_000),
                fs_entry("src/stale.rs", "/workspace/src/stale.rs", "b", 1_100),
                fs_entry(
                    "src/missing_catalog.rs",
                    "/workspace/src/missing_catalog.rs",
                    "c",
                    1_200,
                ),
                fs_entry(
                    "src/missing_membership.rs",
                    "/workspace/src/missing_membership.rs",
                    "d",
                    1_300,
                ),
                fs_entry(
                    "src/duplicate.rs",
                    "/workspace/src/duplicate.rs",
                    "e",
                    1_400,
                ),
            ],
            catalog: vec![
                catalog_entry("src/clean.rs", "/workspace/src/clean.rs", "a", 1, None),
                catalog_entry("src/stale.rs", "/workspace/src/stale.rs", "f", 2, None),
                catalog_entry(
                    "src/missing_membership.rs",
                    "/workspace/src/missing_membership.rs",
                    "d",
                    3,
                    None,
                ),
                catalog_entry(
                    "src/duplicate.rs",
                    "/workspace/src/duplicate.rs",
                    "e",
                    4,
                    None,
                ),
                catalog_entry("src/orphan.rs", "/workspace/src/orphan.rs", "9", 5, None),
            ],
            vector_index: vec![
                membership("src/clean.rs", "vector", 1),
                membership("src/stale.rs", "vector", 2),
                membership("src/duplicate.rs", "vector", 4),
                membership("src/duplicate.rs", "vector-extra", 4),
            ],
            lexical_index: vec![
                membership("src/clean.rs", "lexical", 1),
                membership("src/stale.rs", "lexical", 2),
                membership("src/duplicate.rs", "lexical", 4),
                membership("src/duplicate.rs", "lexical-extra", 4),
            ],
            watcher_checkpoint: WatcherCheckpointSnapshot {
                checkpoint_id: "watcher-drift".to_owned(),
                last_applied_seq: 40,
                pending_changes: 3,
                watermark_ms: 900,
            },
        });

        let kinds = report
            .findings
            .iter()
            .map(|finding| finding.kind)
            .collect::<BTreeSet<_>>();
        assert!(kinds.contains(&IndexFreshnessFindingKind::MissingCatalog));
        assert!(kinds.contains(&IndexFreshnessFindingKind::StaleCatalog));
        assert!(kinds.contains(&IndexFreshnessFindingKind::OrphanCatalog));
        assert!(kinds.contains(&IndexFreshnessFindingKind::MissingVector));
        assert!(kinds.contains(&IndexFreshnessFindingKind::MissingLexical));
        assert!(kinds.contains(&IndexFreshnessFindingKind::DoubleIndexed));
        assert!(kinds.contains(&IndexFreshnessFindingKind::WatcherCheckpointStale));
        assert!(report.findings.iter().any(|finding| {
            finding.kind == IndexFreshnessFindingKind::MissingLexical
                && finding.repair_action == IndexFreshnessRepairActionKind::EnqueueReindex
        }));
        assert_eq!(
            report.summary.verdict,
            IndexFreshnessAuditVerdict::FailClosed
        );
        assert!(report.repair_plan.dry_run);
        assert!(report.repair_plan.fail_closed);
        assert!(
            report
                .repair_plan
                .actions
                .iter()
                .all(|action| !action.destructive)
        );

        let reasons = report
            .findings
            .iter()
            .map(|finding| finding.reason_code.as_str())
            .collect::<BTreeSet<_>>();
        assert!(reasons.contains("FSFS_AUDIT_MISSING_CATALOG"));
        assert!(reasons.contains("FSFS_AUDIT_STALE_CATALOG_HASH"));
        assert!(reasons.contains("FSFS_AUDIT_ORPHAN_CATALOG_ENTRY"));
        assert!(reasons.contains("FSFS_AUDIT_MISSING_VECTOR_MEMBERSHIP"));
        assert!(reasons.contains("FSFS_AUDIT_MISSING_LEXICAL_MEMBERSHIP"));
        assert!(reasons.contains("FSFS_AUDIT_DOUBLE_VECTOR_MEMBERSHIP"));
        assert!(reasons.contains("FSFS_AUDIT_DOUBLE_LEXICAL_MEMBERSHIP"));
        assert!(reasons.contains("FSFS_AUDIT_WATCHER_CHECKPOINT_STALE"));
    }

    #[test]
    fn file_state_mtime_changed_respects_granularity() {
        let state1 = FileState {
            file_id: "f1".to_owned(),
            size_bytes: 100,
            mtime_ns: 1_000_000_000,
            content_hash: "abc".to_owned(),
        };
        let state2 = FileState {
            file_id: "f1".to_owned(),
            size_bytes: 100,
            mtime_ns: 1_000_500_000, // 500μs difference
            content_hash: "abc".to_owned(),
        };

        // 1ms granularity: 500μs difference is below threshold
        assert!(!state1.mtime_changed(&state2, 1_000_000));

        // 100μs granularity: 500μs difference exceeds threshold
        assert!(state1.mtime_changed(&state2, 100_000));
    }

    #[test]
    fn file_state_size_changed() {
        let state1 = FileState {
            file_id: "f1".to_owned(),
            size_bytes: 100,
            mtime_ns: 1_000_000_000,
            content_hash: "abc".to_owned(),
        };
        let state2 = FileState {
            file_id: "f1".to_owned(),
            size_bytes: 200,
            mtime_ns: 1_000_000_000,
            content_hash: "abc".to_owned(),
        };

        assert!(state1.size_changed(&state2));
        assert!(!state1.size_changed(&state1));
    }

    #[test]
    fn file_state_content_changed() {
        let state1 = FileState {
            file_id: "f1".to_owned(),
            size_bytes: 100,
            mtime_ns: 1_000_000_000,
            content_hash: "abc".to_owned(),
        };
        let state2 = FileState {
            file_id: "f1".to_owned(),
            size_bytes: 100,
            mtime_ns: 1_000_000_000,
            content_hash: "def".to_owned(),
        };

        assert!(state1.content_changed(&state2));
        assert!(!state1.content_changed(&state1));
    }

    #[test]
    fn change_decision_requires_embedding() {
        let state = FileState {
            file_id: "f1".to_owned(),
            size_bytes: 100,
            mtime_ns: 1_000_000_000,
            content_hash: "abc".to_owned(),
        };

        let decision = IncrementalChangeDecision::new(
            "/path/to/file".to_owned(),
            EVENT_TYPE_MODIFY.to_owned(),
            DETECTION_MODE_HASH_CONFIRM.to_owned(),
            state.clone(),
            state,
            QUEUE_ACTION_ENQUEUE_EMBED.to_owned(),
            "FSFS_CONTENT_CHANGED".to_owned(),
            0.95,
        );

        assert!(decision.requires_embedding());
        assert_eq!(decision.kind, KIND_CHANGE_DECISION);
        assert_eq!(decision.v, CONTRACT_VERSION);
    }

    #[test]
    fn recovery_checkpoint_needs_replay() {
        let dirty = IncrementalRecoveryCheckpoint::new(
            "cp-1".to_owned(),
            100,
            5,
            false,
            0,
            "replay_pending".to_owned(),
            "FSFS_PENDING_CHANGES".to_owned(),
        );
        assert!(dirty.needs_replay());

        let clean = IncrementalRecoveryCheckpoint::new(
            "cp-2".to_owned(),
            100,
            0,
            true,
            0,
            ACTION_ON_RESTART_RESUME_TAIL.to_owned(),
            "FSFS_CLEAN".to_owned(),
        );
        assert!(!clean.needs_replay());
    }

    #[test]
    fn contract_roundtrip_serialization() {
        let contract = IncrementalChangeDetectionContractDefinition::default();
        let json = serde_json::to_string(&contract).unwrap();
        let parsed: IncrementalChangeDetectionContractDefinition =
            serde_json::from_str(&json).unwrap();
        assert_eq!(contract, parsed);
    }

    #[test]
    fn decision_roundtrip_serialization() {
        let state = FileState {
            file_id: "f1".to_owned(),
            size_bytes: 100,
            mtime_ns: 1_000_000_000,
            content_hash: "abc123def456abc123def456abc123def456abc123def456abc123def456abcd"
                .to_owned(),
        };
        let decision = IncrementalChangeDecision::new(
            "/path".to_owned(),
            EVENT_TYPE_MODIFY.to_owned(),
            DETECTION_MODE_FASTPATH.to_owned(),
            state.clone(),
            state,
            QUEUE_ACTION_SKIP_NO_CHANGE.to_owned(),
            "FSFS_NO_CHANGE".to_owned(),
            1.0,
        );
        let json = serde_json::to_string(&decision).unwrap();
        let parsed: IncrementalChangeDecision = serde_json::from_str(&json).unwrap();
        assert_eq!(decision, parsed);
    }

    #[test]
    fn checkpoint_roundtrip_serialization() {
        let checkpoint = IncrementalRecoveryCheckpoint::new(
            "cp-test".to_owned(),
            42,
            0,
            true,
            0,
            ACTION_ON_RESTART_RESUME_TAIL.to_owned(),
            "FSFS_CLEAN".to_owned(),
        );
        let json = serde_json::to_string(&checkpoint).unwrap();
        let parsed: IncrementalRecoveryCheckpoint = serde_json::from_str(&json).unwrap();
        assert_eq!(checkpoint, parsed);
    }

    #[test]
    fn contract_rejects_wrong_version() {
        let mut value = json!(IncrementalChangeDetectionContractDefinition::default());
        value["v"] = json!(2);

        let error = serde_json::from_value::<IncrementalChangeDetectionContractDefinition>(value)
            .expect_err("reject bad version");

        assert!(error.to_string().contains("schema version 1"));
    }

    #[test]
    fn decision_rejects_delete_enqueue_embed() {
        let value = json!({
            "kind": KIND_CHANGE_DECISION,
            "v": CONTRACT_VERSION,
            "path": "/workspace/src/old.rs",
            "event_type": EVENT_TYPE_DELETE,
            "detection_mode": DETECTION_MODE_JOURNAL_REPLAY,
            "queue_action": QUEUE_ACTION_ENQUEUE_EMBED,
            "reason_code": "FSFS_DELETE_OBSERVED",
            "confidence": 0.99
        });

        let error = serde_json::from_value::<IncrementalChangeDecision>(value)
            .expect_err("reject delete enqueue");

        assert!(
            error
                .to_string()
                .contains("delete events must not enqueue embeddings")
        );
    }

    #[test]
    fn decision_rejects_rename_without_paths() {
        let value = json!({
            "kind": KIND_CHANGE_DECISION,
            "v": CONTRACT_VERSION,
            "path": "/workspace/src/new_name.rs",
            "event_type": EVENT_TYPE_RENAME,
            "detection_mode": DETECTION_MODE_FASTPATH,
            "queue_action": QUEUE_ACTION_SKIP_NO_CHANGE,
            "reason_code": "FSFS_RENAME_IDENTITY_PRESERVED",
            "confidence": 0.9
        });

        let error = serde_json::from_value::<IncrementalChangeDecision>(value)
            .expect_err("reject missing rename paths");

        assert!(
            error
                .to_string()
                .contains("rename events require rename_from")
        );
    }

    #[test]
    fn recovery_checkpoint_rejects_clean_journal_with_pending_changes() {
        let value = json!({
            "kind": KIND_RECOVERY_CHECKPOINT,
            "v": CONTRACT_VERSION,
            "checkpoint_id": "fsfs-ckpt-invalid",
            "last_applied_seq": 500,
            "pending_changes": 9,
            "journal_clean": true,
            "stale_entries": 0,
            "action_on_restart": ACTION_ON_RESTART_RESUME_TAIL,
            "reason_code": "FSFS_RECOVERY_STATE_INCONSISTENT"
        });

        let error = serde_json::from_value::<IncrementalRecoveryCheckpoint>(value)
            .expect_err("reject inconsistent clean journal");

        assert!(
            error
                .to_string()
                .contains("pending_changes must be zero when journal_clean is true")
        );
    }

    #[test]
    fn decision_roundtrip_omits_absent_optional_fields() {
        let state = FileState {
            file_id: "f1".to_owned(),
            size_bytes: 100,
            mtime_ns: 1_000_000_000,
            content_hash: "abc123def456abc123def456abc123def456abc123def456abc123def456abcd"
                .to_owned(),
        };
        let decision = IncrementalChangeDecision::new(
            "/path".to_owned(),
            EVENT_TYPE_MODIFY.to_owned(),
            DETECTION_MODE_FASTPATH.to_owned(),
            state.clone(),
            state,
            QUEUE_ACTION_SKIP_NO_CHANGE.to_owned(),
            "FSFS_NO_CHANGE".to_owned(),
            1.0,
        );

        let value = serde_json::to_value(&decision).expect("serialize decision");

        assert!(value.get("rename_from").is_none());
        assert!(value.get("rename_to").is_none());
    }
}
