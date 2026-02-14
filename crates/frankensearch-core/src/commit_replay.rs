//! Commit-stream replay for Native Mode state reconstruction.
//!
//! Provides document-operation replay from a commit stream to deterministically
//! reconstruct local search state. This is the Phase 4 ("Option A") path
//! described in the Native Mode architecture: each commit contains
//! document-level operations (add/update/delete) that replicas can replay
//! to rebuild their search artifacts from logical history.
//!
//! The [`CommitReplayEngine`] tracks replay progress via a
//! [`ReplayWatermark`] and feeds operations to a [`ReplayConsumer`] that
//! handles the actual index mutations.

use std::collections::BTreeMap;
use std::sync::RwLock;

use serde::{Deserialize, Serialize};

use crate::SearchError;

// ---------------------------------------------------------------------------
// Document operations
// ---------------------------------------------------------------------------

/// A document-level operation within a commit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DocumentOp {
    /// Add a new document to the corpus.
    Add {
        /// Unique document identifier.
        doc_id: String,
        /// Document text content.
        content: String,
        /// Arbitrary key-value metadata.
        metadata: BTreeMap<String, String>,
    },
    /// Update an existing document (full replacement).
    Update {
        /// Unique document identifier.
        doc_id: String,
        /// New document text content.
        content: String,
        /// New metadata (replaces previous).
        metadata: BTreeMap<String, String>,
    },
    /// Delete a document from the corpus.
    Delete {
        /// Unique document identifier.
        doc_id: String,
    },
}

impl DocumentOp {
    /// The document identifier affected by this operation.
    #[must_use]
    pub fn doc_id(&self) -> &str {
        match self {
            Self::Add { doc_id, .. } | Self::Update { doc_id, .. } | Self::Delete { doc_id } => {
                doc_id
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Commit entry
// ---------------------------------------------------------------------------

/// A single entry in the commit stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitEntry {
    /// Monotonically increasing commit sequence number.
    pub commit_seq: u64,
    /// Content-derived commit identifier (e.g. BLAKE3 hash of payload).
    pub commit_id: String,
    /// Document operations contained in this commit.
    pub operations: Vec<DocumentOp>,
    /// Unix timestamp (millis) when the commit was created.
    pub timestamp: u64,
}

// ---------------------------------------------------------------------------
// Replay progress
// ---------------------------------------------------------------------------

/// Tracks replay progress through the commit stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayWatermark {
    /// Last successfully applied commit sequence number.
    /// `0` means no commits have been applied yet.
    pub last_applied_seq: u64,
    /// Unix timestamp (millis) of the last successful apply.
    pub last_applied_at: u64,
    /// Total number of commits applied since engine creation.
    pub total_commits_applied: u64,
    /// Total number of document operations applied.
    pub total_ops_applied: u64,
}

impl ReplayWatermark {
    /// A fresh watermark with no replay history.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            last_applied_seq: 0,
            last_applied_at: 0,
            total_commits_applied: 0,
            total_ops_applied: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Commit outcome
// ---------------------------------------------------------------------------

/// Outcome of attempting to apply a single commit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommitOutcome {
    /// Commit was successfully applied.
    Applied {
        /// The commit sequence that was applied.
        commit_seq: u64,
        /// Number of document operations applied.
        ops_applied: usize,
    },
    /// Commit was skipped without error.
    Skipped {
        /// The commit sequence that was skipped.
        commit_seq: u64,
        /// Why it was skipped.
        reason: SkipReason,
    },
    /// Commit application failed.
    Failed {
        /// The commit sequence that failed.
        commit_seq: u64,
        /// Error description.
        error: String,
    },
}

/// Reason a commit was skipped during replay.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SkipReason {
    /// Commit sequence was already applied (at or below watermark).
    AlreadyApplied,
    /// Commit arrived out of order when strict ordering is enabled.
    OutOfOrder {
        /// Expected next sequence.
        expected: u64,
        /// Actual sequence received.
        got: u64,
    },
    /// Commit contained no operations.
    EmptyCommit,
}

// ---------------------------------------------------------------------------
// Replay consumer trait
// ---------------------------------------------------------------------------

/// Consumer that applies document operations to the local search state.
///
/// Implementors handle the actual index mutations: adding documents to
/// embedding queues, updating lexical indices, removing tombstoned entries.
pub trait ReplayConsumer: Send + Sync {
    /// Apply a document addition.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if the document cannot be added.
    fn apply_add(
        &self,
        doc_id: &str,
        content: &str,
        metadata: &BTreeMap<String, String>,
    ) -> Result<(), SearchError>;

    /// Apply a document update (full content replacement).
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if the document cannot be updated.
    fn apply_update(
        &self,
        doc_id: &str,
        content: &str,
        metadata: &BTreeMap<String, String>,
    ) -> Result<(), SearchError>;

    /// Apply a document deletion.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if the document cannot be deleted.
    fn apply_delete(&self, doc_id: &str) -> Result<(), SearchError>;
}

// ---------------------------------------------------------------------------
// Replay policy
// ---------------------------------------------------------------------------

/// Configuration for the replay engine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct ReplayPolicy {
    /// Require commits to arrive in strict sequential order.
    /// When true, a commit with `seq != watermark + 1` is rejected.
    pub strict_ordering: bool,
    /// Whether to silently skip commits that have already been applied.
    pub skip_duplicates: bool,
    /// Whether to skip commits that contain no operations.
    pub skip_empty: bool,
    /// Stop replay on first failure (true) or continue with remaining
    /// commits (false).
    pub stop_on_failure: bool,
}

impl Default for ReplayPolicy {
    fn default() -> Self {
        Self {
            strict_ordering: true,
            skip_duplicates: true,
            skip_empty: true,
            stop_on_failure: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Replay engine
// ---------------------------------------------------------------------------

/// Engine that replays a commit stream to reconstruct local search state.
///
/// Thread-safe: progress is tracked via `std::sync::RwLock` for synchronous
/// access (no `&Cx` required on the replay hot path).
pub struct CommitReplayEngine {
    /// Current replay progress.
    watermark: RwLock<ReplayWatermark>,
    /// Replay configuration.
    policy: ReplayPolicy,
}

impl CommitReplayEngine {
    /// Create a new engine with the given policy, starting from empty state.
    #[must_use]
    pub const fn new(policy: ReplayPolicy) -> Self {
        Self {
            watermark: RwLock::new(ReplayWatermark::empty()),
            policy,
        }
    }

    /// Create an engine that resumes from a saved watermark.
    #[must_use]
    pub const fn resume_from(watermark: ReplayWatermark, policy: ReplayPolicy) -> Self {
        Self {
            watermark: RwLock::new(watermark),
            policy,
        }
    }

    /// Apply a single commit entry through the consumer.
    ///
    /// Returns the outcome of the replay attempt.
    pub fn apply(
        &self,
        entry: &CommitEntry,
        consumer: &dyn ReplayConsumer,
        now_millis: u64,
    ) -> CommitOutcome {
        // Check skip conditions.
        if let Some(skip) = self.check_skip(entry) {
            return skip;
        }

        // Apply each operation.
        let mut ops_applied = 0usize;
        for op in &entry.operations {
            let result = match op {
                DocumentOp::Add {
                    doc_id,
                    content,
                    metadata,
                } => consumer.apply_add(doc_id, content, metadata),
                DocumentOp::Update {
                    doc_id,
                    content,
                    metadata,
                } => consumer.apply_update(doc_id, content, metadata),
                DocumentOp::Delete { doc_id } => consumer.apply_delete(doc_id),
            };

            match result {
                Ok(()) => ops_applied += 1,
                Err(e) => {
                    return CommitOutcome::Failed {
                        commit_seq: entry.commit_seq,
                        error: e.to_string(),
                    };
                }
            }
        }

        // Advance watermark.
        self.advance_watermark(entry.commit_seq, now_millis, ops_applied);

        CommitOutcome::Applied {
            commit_seq: entry.commit_seq,
            ops_applied,
        }
    }

    /// Replay a batch of commits in order.
    ///
    /// Returns outcomes for each commit. If `stop_on_failure` is set and
    /// a commit fails, remaining commits are not processed.
    pub fn replay_batch(
        &self,
        entries: &[CommitEntry],
        consumer: &dyn ReplayConsumer,
        now_millis: u64,
    ) -> Vec<CommitOutcome> {
        let mut outcomes = Vec::with_capacity(entries.len());
        for entry in entries {
            let outcome = self.apply(entry, consumer, now_millis);
            let failed = matches!(outcome, CommitOutcome::Failed { .. });
            outcomes.push(outcome);
            if failed && self.policy.stop_on_failure {
                break;
            }
        }
        outcomes
    }

    /// Current replay watermark snapshot.
    #[must_use]
    pub fn watermark(&self) -> ReplayWatermark {
        self.watermark
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Reset the engine to empty state (for re-bootstrapping).
    pub fn reset(&self) {
        let mut wm = self
            .watermark
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *wm = ReplayWatermark::empty();
    }

    /// Check whether a commit should be skipped.
    fn check_skip(&self, entry: &CommitEntry) -> Option<CommitOutcome> {
        let last_applied_seq = {
            let wm = self
                .watermark
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            wm.last_applied_seq
        };

        // Already applied.
        if entry.commit_seq <= last_applied_seq && self.policy.skip_duplicates {
            return Some(CommitOutcome::Skipped {
                commit_seq: entry.commit_seq,
                reason: SkipReason::AlreadyApplied,
            });
        }

        // Strict ordering check.
        if self.policy.strict_ordering && last_applied_seq > 0 {
            let expected = last_applied_seq + 1;
            if entry.commit_seq != expected {
                return Some(CommitOutcome::Skipped {
                    commit_seq: entry.commit_seq,
                    reason: SkipReason::OutOfOrder {
                        expected,
                        got: entry.commit_seq,
                    },
                });
            }
        }

        // Empty commit.
        if entry.operations.is_empty() && self.policy.skip_empty {
            return Some(CommitOutcome::Skipped {
                commit_seq: entry.commit_seq,
                reason: SkipReason::EmptyCommit,
            });
        }

        None
    }

    /// Advance watermark after successful apply.
    fn advance_watermark(&self, commit_seq: u64, now_millis: u64, ops: usize) {
        let mut wm = self
            .watermark
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        if commit_seq > wm.last_applied_seq {
            wm.last_applied_seq = commit_seq;
        }
        wm.last_applied_at = now_millis;
        wm.total_commits_applied += 1;
        wm.total_ops_applied += ops as u64;
        drop(wm);
    }
}

impl Default for CommitReplayEngine {
    fn default() -> Self {
        Self::new(ReplayPolicy::default())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Test consumer that records all operations.
    struct RecordingConsumer {
        log: Mutex<Vec<String>>,
    }

    impl RecordingConsumer {
        fn new() -> Self {
            Self {
                log: Mutex::new(Vec::new()),
            }
        }

        fn entries(&self) -> Vec<String> {
            self.log.lock().unwrap().clone()
        }
    }

    impl ReplayConsumer for RecordingConsumer {
        fn apply_add(
            &self,
            doc_id: &str,
            content: &str,
            _metadata: &BTreeMap<String, String>,
        ) -> Result<(), SearchError> {
            self.log
                .lock()
                .unwrap()
                .push(format!("ADD:{doc_id}:{content}"));
            Ok(())
        }

        fn apply_update(
            &self,
            doc_id: &str,
            content: &str,
            _metadata: &BTreeMap<String, String>,
        ) -> Result<(), SearchError> {
            self.log
                .lock()
                .unwrap()
                .push(format!("UPDATE:{doc_id}:{content}"));
            Ok(())
        }

        fn apply_delete(&self, doc_id: &str) -> Result<(), SearchError> {
            self.log.lock().unwrap().push(format!("DELETE:{doc_id}"));
            Ok(())
        }
    }

    /// Consumer that always fails.
    struct FailingConsumer;

    impl ReplayConsumer for FailingConsumer {
        fn apply_add(
            &self,
            _doc_id: &str,
            _content: &str,
            _metadata: &BTreeMap<String, String>,
        ) -> Result<(), SearchError> {
            Err(SearchError::SubsystemError {
                subsystem: "replay",
                source: "test failure".into(),
            })
        }

        fn apply_update(
            &self,
            _doc_id: &str,
            _content: &str,
            _metadata: &BTreeMap<String, String>,
        ) -> Result<(), SearchError> {
            Err(SearchError::SubsystemError {
                subsystem: "replay",
                source: "test failure".into(),
            })
        }

        fn apply_delete(&self, _doc_id: &str) -> Result<(), SearchError> {
            Err(SearchError::SubsystemError {
                subsystem: "replay",
                source: "test failure".into(),
            })
        }
    }

    fn make_commit(seq: u64, ops: Vec<DocumentOp>) -> CommitEntry {
        CommitEntry {
            commit_seq: seq,
            commit_id: format!("commit-{seq}"),
            operations: ops,
            timestamp: 1_700_000_000_000 + seq * 1000,
        }
    }

    fn add_op(doc_id: &str, content: &str) -> DocumentOp {
        DocumentOp::Add {
            doc_id: doc_id.into(),
            content: content.into(),
            metadata: BTreeMap::new(),
        }
    }

    fn update_op(doc_id: &str, content: &str) -> DocumentOp {
        DocumentOp::Update {
            doc_id: doc_id.into(),
            content: content.into(),
            metadata: BTreeMap::new(),
        }
    }

    fn delete_op(doc_id: &str) -> DocumentOp {
        DocumentOp::Delete {
            doc_id: doc_id.into(),
        }
    }

    #[test]
    fn empty_engine_has_zero_watermark() {
        let engine = CommitReplayEngine::default();
        let wm = engine.watermark();
        assert_eq!(wm.last_applied_seq, 0);
        assert_eq!(wm.total_commits_applied, 0);
        assert_eq!(wm.total_ops_applied, 0);
    }

    #[test]
    fn single_commit_apply() {
        let engine = CommitReplayEngine::default();
        let consumer = RecordingConsumer::new();

        let commit = make_commit(1, vec![add_op("doc-1", "hello world")]);
        let outcome = engine.apply(&commit, &consumer, 5000);

        assert!(matches!(
            outcome,
            CommitOutcome::Applied {
                commit_seq: 1,
                ops_applied: 1
            }
        ));
        assert_eq!(consumer.entries(), vec!["ADD:doc-1:hello world"]);
        assert_eq!(engine.watermark().last_applied_seq, 1);
        assert_eq!(engine.watermark().total_commits_applied, 1);
    }

    #[test]
    fn multi_op_commit() {
        let engine = CommitReplayEngine::default();
        let consumer = RecordingConsumer::new();

        let commit = make_commit(
            1,
            vec![
                add_op("doc-1", "first"),
                add_op("doc-2", "second"),
                update_op("doc-1", "first-updated"),
                delete_op("doc-3"),
            ],
        );
        let outcome = engine.apply(&commit, &consumer, 5000);

        assert!(matches!(
            outcome,
            CommitOutcome::Applied { ops_applied: 4, .. }
        ));
        assert_eq!(engine.watermark().total_ops_applied, 4);
        assert_eq!(consumer.entries().len(), 4);
    }

    #[test]
    fn sequential_commits() {
        let engine = CommitReplayEngine::default();
        let consumer = RecordingConsumer::new();

        let c1 = make_commit(1, vec![add_op("doc-1", "v1")]);
        let c2 = make_commit(2, vec![update_op("doc-1", "v2")]);
        let c3 = make_commit(3, vec![delete_op("doc-1")]);

        engine.apply(&c1, &consumer, 1000);
        engine.apply(&c2, &consumer, 2000);
        engine.apply(&c3, &consumer, 3000);

        assert_eq!(engine.watermark().last_applied_seq, 3);
        assert_eq!(engine.watermark().total_commits_applied, 3);
        assert_eq!(engine.watermark().total_ops_applied, 3);
        assert_eq!(
            consumer.entries(),
            vec!["ADD:doc-1:v1", "UPDATE:doc-1:v2", "DELETE:doc-1"]
        );
    }

    #[test]
    fn skip_duplicate_commit() {
        let engine = CommitReplayEngine::default();
        let consumer = RecordingConsumer::new();

        let c1 = make_commit(1, vec![add_op("doc-1", "v1")]);
        engine.apply(&c1, &consumer, 1000);

        // Replay same commit.
        let outcome = engine.apply(&c1, &consumer, 2000);
        assert!(matches!(
            outcome,
            CommitOutcome::Skipped {
                reason: SkipReason::AlreadyApplied,
                ..
            }
        ));

        // Only one operation was applied.
        assert_eq!(consumer.entries().len(), 1);
        assert_eq!(engine.watermark().total_commits_applied, 1);
    }

    #[test]
    fn strict_ordering_rejects_gap() {
        let engine = CommitReplayEngine::default();
        let consumer = RecordingConsumer::new();

        let c1 = make_commit(1, vec![add_op("doc-1", "v1")]);
        engine.apply(&c1, &consumer, 1000);

        // Skip seq 2, try seq 3.
        let c3 = make_commit(3, vec![add_op("doc-3", "v1")]);
        let outcome = engine.apply(&c3, &consumer, 2000);

        assert!(matches!(
            outcome,
            CommitOutcome::Skipped {
                reason: SkipReason::OutOfOrder {
                    expected: 2,
                    got: 3,
                },
                ..
            }
        ));
    }

    #[test]
    fn relaxed_ordering_allows_gap() {
        let policy = ReplayPolicy {
            strict_ordering: false,
            ..ReplayPolicy::default()
        };
        let engine = CommitReplayEngine::new(policy);
        let consumer = RecordingConsumer::new();

        let c1 = make_commit(1, vec![add_op("doc-1", "v1")]);
        engine.apply(&c1, &consumer, 1000);

        // Skip seq 2, apply seq 3.
        let c3 = make_commit(3, vec![add_op("doc-3", "v1")]);
        let outcome = engine.apply(&c3, &consumer, 2000);

        assert!(matches!(outcome, CommitOutcome::Applied { .. }));
        assert_eq!(engine.watermark().last_applied_seq, 3);
    }

    #[test]
    fn skip_empty_commit() {
        let engine = CommitReplayEngine::default();
        let consumer = RecordingConsumer::new();

        let c1 = make_commit(1, vec![]);
        let outcome = engine.apply(&c1, &consumer, 1000);

        assert!(matches!(
            outcome,
            CommitOutcome::Skipped {
                reason: SkipReason::EmptyCommit,
                ..
            }
        ));
        // Watermark should not advance.
        assert_eq!(engine.watermark().last_applied_seq, 0);
    }

    #[test]
    fn allow_empty_commit_when_policy_permits() {
        let policy = ReplayPolicy {
            skip_empty: false,
            ..ReplayPolicy::default()
        };
        let engine = CommitReplayEngine::new(policy);
        let consumer = RecordingConsumer::new();

        let c1 = make_commit(1, vec![]);
        let outcome = engine.apply(&c1, &consumer, 1000);

        assert!(matches!(
            outcome,
            CommitOutcome::Applied { ops_applied: 0, .. }
        ));
        assert_eq!(engine.watermark().last_applied_seq, 1);
    }

    #[test]
    fn consumer_failure_returns_failed_outcome() {
        let engine = CommitReplayEngine::default();
        let consumer = FailingConsumer;

        let c1 = make_commit(1, vec![add_op("doc-1", "v1")]);
        let outcome = engine.apply(&c1, &consumer, 1000);

        assert!(matches!(outcome, CommitOutcome::Failed { .. }));
        // Watermark should NOT advance on failure.
        assert_eq!(engine.watermark().last_applied_seq, 0);
    }

    #[test]
    fn partial_failure_in_multi_op_commit() {
        // Consumer that fails on the second operation.
        struct FailOnSecond {
            count: Mutex<usize>,
        }

        impl ReplayConsumer for FailOnSecond {
            fn apply_add(
                &self,
                _doc_id: &str,
                _content: &str,
                _metadata: &BTreeMap<String, String>,
            ) -> Result<(), SearchError> {
                let mut c = self.count.lock().unwrap();
                *c += 1;
                if *c >= 2 {
                    Err(SearchError::SubsystemError {
                        subsystem: "replay",
                        source: "second op fails".into(),
                    })
                } else {
                    Ok(())
                }
            }

            fn apply_update(
                &self,
                _: &str,
                _: &str,
                _: &BTreeMap<String, String>,
            ) -> Result<(), SearchError> {
                Ok(())
            }

            fn apply_delete(&self, _: &str) -> Result<(), SearchError> {
                Ok(())
            }
        }

        let engine = CommitReplayEngine::default();
        let consumer = FailOnSecond {
            count: Mutex::new(0),
        };

        let commit = make_commit(1, vec![add_op("doc-1", "ok"), add_op("doc-2", "will-fail")]);
        let outcome = engine.apply(&commit, &consumer, 1000);

        assert!(matches!(
            outcome,
            CommitOutcome::Failed { commit_seq: 1, .. }
        ));
        // Watermark should NOT advance.
        assert_eq!(engine.watermark().last_applied_seq, 0);
    }

    #[test]
    fn batch_replay_sequential() {
        let engine = CommitReplayEngine::default();
        let consumer = RecordingConsumer::new();

        let entries = vec![
            make_commit(1, vec![add_op("a", "1")]),
            make_commit(2, vec![add_op("b", "2")]),
            make_commit(3, vec![add_op("c", "3")]),
        ];

        let outcomes = engine.replay_batch(&entries, &consumer, 5000);
        assert_eq!(outcomes.len(), 3);
        assert!(
            outcomes
                .iter()
                .all(|o| matches!(o, CommitOutcome::Applied { .. }))
        );
        assert_eq!(engine.watermark().last_applied_seq, 3);
        assert_eq!(engine.watermark().total_commits_applied, 3);
    }

    #[test]
    fn batch_replay_stops_on_failure() {
        let engine = CommitReplayEngine::default();
        let consumer = FailingConsumer;

        let entries = vec![
            make_commit(1, vec![add_op("a", "1")]),
            make_commit(2, vec![add_op("b", "2")]),
        ];

        let outcomes = engine.replay_batch(&entries, &consumer, 5000);
        // Should stop after first failure.
        assert_eq!(outcomes.len(), 1);
        assert!(matches!(outcomes[0], CommitOutcome::Failed { .. }));
    }

    #[test]
    fn batch_replay_continues_on_failure_when_policy_permits() {
        let policy = ReplayPolicy {
            stop_on_failure: false,
            ..ReplayPolicy::default()
        };
        let engine = CommitReplayEngine::new(policy);
        let consumer = FailingConsumer;

        let entries = vec![
            make_commit(1, vec![add_op("a", "1")]),
            make_commit(2, vec![add_op("b", "2")]),
        ];

        let outcomes = engine.replay_batch(&entries, &consumer, 5000);
        // Both should be attempted.
        assert_eq!(outcomes.len(), 2);
    }

    #[test]
    fn resume_from_watermark() {
        let saved = ReplayWatermark {
            last_applied_seq: 42,
            last_applied_at: 1000,
            total_commits_applied: 42,
            total_ops_applied: 100,
        };
        let engine = CommitReplayEngine::resume_from(saved, ReplayPolicy::default());
        let consumer = RecordingConsumer::new();

        // Commit <= watermark should be skipped.
        let old = make_commit(40, vec![add_op("old", "stale")]);
        let outcome = engine.apply(&old, &consumer, 2000);
        assert!(matches!(
            outcome,
            CommitOutcome::Skipped {
                reason: SkipReason::AlreadyApplied,
                ..
            }
        ));

        // Next expected commit should apply.
        let next = make_commit(43, vec![add_op("new", "fresh")]);
        let outcome = engine.apply(&next, &consumer, 3000);
        assert!(matches!(outcome, CommitOutcome::Applied { .. }));
        assert_eq!(engine.watermark().last_applied_seq, 43);
        assert_eq!(engine.watermark().total_commits_applied, 43);
    }

    #[test]
    fn reset_clears_watermark() {
        let engine = CommitReplayEngine::default();
        let consumer = RecordingConsumer::new();

        let c1 = make_commit(1, vec![add_op("doc-1", "v1")]);
        engine.apply(&c1, &consumer, 1000);
        assert_eq!(engine.watermark().last_applied_seq, 1);

        engine.reset();
        assert_eq!(engine.watermark().last_applied_seq, 0);
        assert_eq!(engine.watermark().total_commits_applied, 0);
    }

    #[test]
    fn document_op_doc_id_accessor() {
        let add = add_op("id-add", "content");
        assert_eq!(add.doc_id(), "id-add");

        let upd = update_op("id-upd", "content");
        assert_eq!(upd.doc_id(), "id-upd");

        let del = delete_op("id-del");
        assert_eq!(del.doc_id(), "id-del");
    }

    #[test]
    fn watermark_serde_roundtrip() {
        let wm = ReplayWatermark {
            last_applied_seq: 99,
            last_applied_at: 1_700_000_000_000,
            total_commits_applied: 99,
            total_ops_applied: 500,
        };
        let json = serde_json::to_string(&wm).expect("serialize");
        let back: ReplayWatermark = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(wm, back);
    }

    #[test]
    fn commit_entry_serde_roundtrip() {
        let entry = make_commit(
            1,
            vec![
                add_op("doc-1", "hello"),
                update_op("doc-2", "world"),
                delete_op("doc-3"),
            ],
        );
        let json = serde_json::to_string(&entry).expect("serialize");
        let back: CommitEntry = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(entry, back);
    }

    #[test]
    fn replay_policy_serde_roundtrip() {
        let policy = ReplayPolicy {
            strict_ordering: false,
            skip_duplicates: true,
            skip_empty: false,
            stop_on_failure: true,
        };
        let json = serde_json::to_string(&policy).expect("serialize");
        let back: ReplayPolicy = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(policy, back);
    }

    #[test]
    fn first_commit_with_strict_ordering_accepts_any_seq() {
        let engine = CommitReplayEngine::default();
        let consumer = RecordingConsumer::new();

        // First commit can be any seq (watermark is 0, strict ordering only
        // checks when last_applied_seq > 0).
        let c = make_commit(42, vec![add_op("doc-1", "v1")]);
        let outcome = engine.apply(&c, &consumer, 1000);
        assert!(matches!(outcome, CommitOutcome::Applied { .. }));
        assert_eq!(engine.watermark().last_applied_seq, 42);
    }
}
