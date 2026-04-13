use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FastpathPolicy {
    pub mtime_granularity_ns: u64,
    pub require_size_change: bool,
    pub hash_on_mtime_only: bool,
    pub max_fastpath_skips: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HashPolicy {
    pub algorithm: String, // "sha256", etc.
    pub sample_prefix_bytes: u32,
    pub full_hash_threshold_bytes: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RenameMovePolicy {
    pub identity_keys: Vec<String>,
    pub same_device_rename_preserves_identity: bool,
    pub cross_device_move: String, // "hash_confirm"
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RecoveryPolicy {
    pub journal_required: bool,
    pub replay_order: String, // "sequence_asc"
    pub pending_ttl_seconds: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReconciliationPolicy {
    pub full_scan_interval_seconds: u32,
    pub stale_after_seconds: u32,
    pub orphan_entry_action: String, // "mark_stale"
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IncrementalChangeDetectionContractDefinition {
    pub kind: String, // "fsfs_incremental_change_detection_contract_definition"
    pub v: u32,       // 1
    pub fastpath_policy: FastpathPolicy,
    pub hash_policy: HashPolicy,
    pub rename_move_policy: RenameMovePolicy,
    pub recovery_policy: RecoveryPolicy,
    pub reconciliation_policy: ReconciliationPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FileState {
    pub file_id: String,
    pub size_bytes: u64,
    pub mtime_ns: u64,
    pub content_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IncrementalChangeDecision {
    pub kind: String, // "fsfs_incremental_change_decision"
    pub v: u32,       // 1
    pub path: String,
    pub event_type: String,     // "modify"
    pub detection_mode: String, // "hash_confirm"
    pub previous_state: FileState,
    pub current_state: FileState,
    pub queue_action: String, // "enqueue_embed"
    pub reason_code: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IncrementalRecoveryCheckpoint {
    pub kind: String, // "fsfs_incremental_recovery_checkpoint"
    pub v: u32,       // 1
    pub checkpoint_id: String,
    pub last_applied_seq: u64,
    pub pending_changes: u32,
    pub journal_clean: bool,
    pub stale_entries: u32,
    pub action_on_restart: String, // "replay_pending"
    pub reason_code: String,
}
