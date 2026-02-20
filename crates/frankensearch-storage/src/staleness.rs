//! Storage-backed staleness detector for frankensearch indices.
//!
//! Bridges the `FrankenSQLite` storage layer (bd-3w1.11 index metadata, bd-3w1.2
//! document metadata) with the `IndexCache` staleness detection system (bd-3un.41).
//! Instead of relying solely on file timestamps, this queries the document store
//! for precise change detection.
//!
//! The detector provides two tiers:
//! - **Quick check**: O(1) COUNT of pending `embedding_status` rows
//! - **Full check**: comprehensive analysis of doc changes, embedder revision,
//!   index age, and embedding drift

use std::sync::Arc;
use std::time::Duration;

use frankensearch_core::{SearchError, SearchResult};
use serde::{Deserialize, Serialize};

use crate::connection::Storage;
use crate::document::count_documents;
use crate::index_metadata::StalenessReason;
use crate::schema::SCHEMA_VERSION;

/// Configuration for the storage-backed staleness detector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StalenessConfig {
    /// Minimum number of new/changed documents before triggering rebuild.
    pub min_change_threshold: usize,
    /// Maximum age of index before forced rebuild (even if no changes).
    pub max_index_age_secs: Option<u64>,
    /// Check model revision changes (embedder update).
    pub check_model_revision: bool,
    /// Check schema version changes.
    pub check_schema_version: bool,
    /// Fraction of documents changed that triggers full rebuild vs incremental.
    /// `IncrementalUpdate` when changed < this fraction; `FullRebuild` otherwise.
    pub full_rebuild_fraction: f64,
}

impl Default for StalenessConfig {
    fn default() -> Self {
        Self {
            min_change_threshold: 10,
            max_index_age_secs: None,
            full_rebuild_fraction: 0.30,
            check_model_revision: true,
            check_schema_version: true,
        }
    }
}

/// How stale an index is, from no change to mandatory rebuild.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum StalenessLevel {
    /// Index is perfectly fresh; no changes detected.
    None,
    /// A few documents changed (below `min_change_threshold`).
    Minor,
    /// Significant content change (above threshold or >10% of corpus).
    Significant,
    /// Model revision, schema, or index missing — rebuild is mandatory.
    Critical,
}

/// What the caller should do in response to staleness.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendedAction {
    /// Index is fresh, nothing to do.
    NoAction,
    /// Only embed new/changed documents and update index in-place.
    IncrementalUpdate {
        /// Number of documents to process.
        doc_count: usize,
    },
    /// Schema change, model update, or massive content shift — rebuild all.
    FullRebuild {
        /// Human-readable reason for the rebuild.
        reason: String,
    },
}

/// Statistics gathered during a staleness check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StalenessStats {
    /// Total documents in the store.
    pub total_documents: usize,
    /// Number of documents included in the current index build.
    pub indexed_documents: usize,
    /// Number of documents pending embedding for this embedder.
    pub pending_documents: usize,
    /// Number of documents that failed embedding.
    pub failed_documents: usize,
    /// Documents changed since last build (modified + new).
    pub docs_changed_since_build: usize,
    /// Age of the current index since its last build.
    pub index_age: Duration,
    /// Duration of the last build, if known.
    pub last_build_duration: Option<Duration>,
}

/// Full staleness report combining all signals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StalenessReport {
    pub index_name: String,
    pub is_stale: bool,
    pub level: StalenessLevel,
    pub reasons: Vec<StalenessReason>,
    pub recommended_action: RecommendedAction,
    pub stats: StalenessStats,
}

/// Quick staleness check result (just pending count).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct QuickStalenessCheck {
    pub pending_count: u64,
    pub is_stale: bool,
}

/// Storage-backed staleness detector. Wraps a `Storage` reference and
/// queries document metadata + index metadata to determine whether an
/// index needs rebuilding.
#[derive(Debug)]
pub struct StorageBackedStaleness {
    storage: Arc<Storage>,
    config: StalenessConfig,
}

impl StorageBackedStaleness {
    /// Create a new staleness detector with the given storage and config.
    pub fn new(storage: Arc<Storage>, config: StalenessConfig) -> Self {
        Self { storage, config }
    }

    /// Create with default configuration.
    pub fn with_defaults(storage: Arc<Storage>) -> Self {
        Self::new(storage, StalenessConfig::default())
    }

    /// Quick O(1) check: count pending embeddings for an embedder.
    /// Use this on every refresh-worker iteration. Only falls through to
    /// the full `check()` when this returns `is_stale = true`.
    pub fn quick_check(&self, embedder_id: &str) -> SearchResult<QuickStalenessCheck> {
        let counts = self.storage.count_by_status(embedder_id)?;
        let pending = counts.pending;
        let is_stale = pending > 0;

        tracing::trace!(
            target: "frankensearch.storage",
            op = "quick_staleness_check",
            embedder_id,
            pending,
            is_stale,
            "quick staleness check completed"
        );

        Ok(QuickStalenessCheck {
            pending_count: pending,
            is_stale,
        })
    }

    /// Comprehensive staleness check combining all signals.
    ///
    /// Queries index metadata, document counts, embedding status, and
    /// optionally compares embedder revision and index age.
    pub fn check(
        &self,
        index_name: &str,
        current_embedder_revision: Option<&str>,
    ) -> SearchResult<StalenessReport> {
        // Use the index_metadata staleness check as the foundation.
        let staleness = self.storage.check_index_staleness(
            index_name,
            if self.config.check_model_revision {
                current_embedder_revision
            } else {
                None
            },
        )?;

        // Gather document stats.
        let total_docs = count_documents(self.storage.connection())?;
        let total_documents = usize::try_from(total_docs).unwrap_or(0);

        // Get index metadata for build info.
        let meta = self.storage.get_index_metadata(index_name)?;

        let (indexed_documents, index_age, last_build_duration) = match &meta {
            Some(m) => {
                let indexed = usize::try_from(m.source_doc_count).unwrap_or(0);
                let age = if let Some(built_at) = m.built_at {
                    let now = now_ms()?;
                    let age_ms = now.saturating_sub(built_at);
                    Duration::from_millis(u64::try_from(age_ms).unwrap_or(0))
                } else {
                    Duration::ZERO
                };
                let build_dur = m
                    .build_duration_ms
                    .map(|ms| Duration::from_millis(u64::try_from(ms).unwrap_or(0)));
                (indexed, age, build_dur)
            }
            None => (0, Duration::ZERO, None),
        };

        // Get embedding status counts if we have embedder info.
        let (pending_documents, failed_documents) = if let Some(m) = &meta {
            let counts = self.storage.count_by_status(&m.embedder_id)?;
            (
                usize::try_from(counts.pending).unwrap_or(0),
                usize::try_from(counts.failed).unwrap_or(0),
            )
        } else {
            (0, 0)
        };

        let docs_changed =
            usize::try_from(staleness.docs_modified.saturating_add(staleness.docs_added))
                .unwrap_or(0);

        // Collect reasons, starting from the base staleness check.
        let mut reasons = staleness.reasons;

        if self.config.check_schema_version
            && meta
                .as_ref()
                .is_some_and(|m| m.schema_version != Some(SCHEMA_VERSION))
            && !reasons.contains(&StalenessReason::SchemaChanged)
        {
            reasons.push(StalenessReason::SchemaChanged);
        }

        // Check index age if configured.
        if let Some(max_age_secs) = self.config.max_index_age_secs
            && index_age > Duration::from_secs(max_age_secs)
            && !reasons.contains(&StalenessReason::AgeExceeded)
        {
            reasons.push(StalenessReason::AgeExceeded);
        }

        // Determine severity level.
        let level = compute_level(&reasons, docs_changed, total_documents, &self.config);

        // Determine recommended action.
        let recommended_action = compute_action(
            &reasons,
            docs_changed,
            indexed_documents,
            total_documents,
            &self.config,
        );

        let is_stale = level > StalenessLevel::None;

        let stats = StalenessStats {
            total_documents,
            indexed_documents,
            pending_documents,
            failed_documents,
            docs_changed_since_build: docs_changed,
            index_age,
            last_build_duration,
        };

        tracing::debug!(
            target: "frankensearch.storage",
            op = "full_staleness_check",
            index_name,
            is_stale,
            level = ?level,
            docs_changed,
            total_documents,
            pending_documents,
            "staleness check completed"
        );

        Ok(StalenessReport {
            index_name: index_name.to_owned(),
            is_stale,
            level,
            reasons,
            recommended_action,
            stats,
        })
    }

    /// Access the underlying config.
    #[must_use]
    pub fn config(&self) -> &StalenessConfig {
        &self.config
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn compute_level(
    reasons: &[StalenessReason],
    docs_changed: usize,
    total_docs: usize,
    config: &StalenessConfig,
) -> StalenessLevel {
    if reasons.is_empty() {
        return StalenessLevel::None;
    }

    // Critical: never built, embedder changed, schema changed.
    if reasons.contains(&StalenessReason::NeverBuilt)
        || reasons.contains(&StalenessReason::EmbedderChanged)
        || reasons.contains(&StalenessReason::AgeExceeded)
        || reasons.contains(&StalenessReason::SchemaChanged)
    {
        return StalenessLevel::Critical;
    }

    // Significant: above threshold or >10% of corpus changed.
    if docs_changed >= config.min_change_threshold {
        return StalenessLevel::Significant;
    }
    if total_docs > 0 && docs_changed * 10 > total_docs {
        return StalenessLevel::Significant;
    }

    // Minor: some changes, but below threshold.
    StalenessLevel::Minor
}

fn compute_action(
    reasons: &[StalenessReason],
    docs_changed: usize,
    indexed_docs: usize,
    total_docs: usize,
    config: &StalenessConfig,
) -> RecommendedAction {
    if reasons.is_empty() {
        return RecommendedAction::NoAction;
    }

    // Critical reasons always trigger full rebuild.
    if reasons.contains(&StalenessReason::NeverBuilt) {
        return RecommendedAction::FullRebuild {
            reason: "index has never been built".to_owned(),
        };
    }
    if reasons.contains(&StalenessReason::EmbedderChanged) {
        return RecommendedAction::FullRebuild {
            reason: "embedder model revision changed".to_owned(),
        };
    }
    if reasons.contains(&StalenessReason::AgeExceeded) {
        return RecommendedAction::FullRebuild {
            reason: "index age exceeded configured maximum".to_owned(),
        };
    }
    if reasons.contains(&StalenessReason::SchemaChanged) {
        return RecommendedAction::FullRebuild {
            reason: "storage schema version changed".to_owned(),
        };
    }

    // Incremental vs full rebuild based on change fraction against the
    // corpus the current index was built from. If that baseline is missing,
    // infer it from unchanged documents.
    let baseline_docs = if indexed_docs > 0 {
        indexed_docs
    } else {
        total_docs.saturating_sub(docs_changed)
    };
    if baseline_docs > 0 {
        #[allow(clippy::cast_precision_loss)]
        let fraction = docs_changed as f64 / baseline_docs as f64;
        // NaN full_rebuild_fraction makes >= always false, silently
        // disabling full-rebuild detection. Fall back to the default.
        let threshold = if config.full_rebuild_fraction.is_finite() {
            config.full_rebuild_fraction
        } else {
            0.30
        };
        if fraction >= threshold {
            return RecommendedAction::FullRebuild {
                reason: format!(
                    "{docs_changed}/{baseline_docs} baseline documents changed ({:.0}%)",
                    fraction * 100.0
                ),
            };
        }
    }

    RecommendedAction::IncrementalUpdate {
        doc_count: docs_changed,
    }
}

fn now_ms() -> SearchResult<i64> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration =
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| SearchError::SubsystemError {
                subsystem: "storage",
                source: Box::new(e),
            })?;
    i64::try_from(duration.as_millis()).map_err(|e| SearchError::SubsystemError {
        subsystem: "storage",
        source: Box::new(std::io::Error::other(format!("timestamp overflow: {e}"))),
    })
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::arc_with_non_send_sync)]

    use super::*;
    use crate::document::DocumentRecord;
    use crate::index_metadata::{BuildTrigger, RecordBuildParams};
    use crate::schema::SCHEMA_VERSION;

    fn test_storage() -> Arc<Storage> {
        Arc::new(Storage::open_in_memory().expect("in-memory storage"))
    }

    fn sample_build_params(name: &str) -> RecordBuildParams {
        RecordBuildParams {
            index_name: name.to_owned(),
            index_type: "fsvi".to_owned(),
            embedder_id: "potion-128m".to_owned(),
            embedder_revision: Some("v1.0".to_owned()),
            dimension: 256,
            record_count: 100,
            source_doc_count: 0,
            build_duration_ms: 500,
            trigger: BuildTrigger::Initial,
            file_path: None,
            file_size_bytes: None,
            file_hash: None,
            schema_version: Some(SCHEMA_VERSION),
            config_json: None,
            fec_path: None,
            fec_size_bytes: None,
            notes: None,
            mean_norm: None,
            variance: None,
        }
    }

    fn insert_doc(storage: &Storage, id: &str, created_at: i64, updated_at: i64) {
        let doc = DocumentRecord::new(id, "preview", [0x42; 32], 64, created_at, updated_at);
        storage.upsert_document(&doc).expect("doc insert");
    }

    #[test]
    fn never_built_is_critical() {
        let storage = test_storage();
        let detector = StorageBackedStaleness::with_defaults(storage);

        let report = detector.check("missing-index", None).expect("check");

        assert!(report.is_stale);
        assert_eq!(report.level, StalenessLevel::Critical);
        assert!(report.reasons.contains(&StalenessReason::NeverBuilt));
        assert!(matches!(
            report.recommended_action,
            RecommendedAction::FullRebuild { .. }
        ));
    }

    #[test]
    fn fresh_index_is_not_stale() {
        let storage = test_storage();
        let params = sample_build_params("idx");
        storage.record_index_build(&params).expect("build");

        let detector = StorageBackedStaleness::with_defaults(storage);
        let report = detector.check("idx", Some("v1.0")).expect("check");

        assert!(!report.is_stale);
        assert_eq!(report.level, StalenessLevel::None);
        assert_eq!(report.recommended_action, RecommendedAction::NoAction);
    }

    #[test]
    fn new_documents_trigger_staleness() {
        let storage = test_storage();
        let params = sample_build_params("idx");
        storage.record_index_build(&params).expect("build");

        let meta = storage
            .get_index_metadata("idx")
            .expect("get")
            .expect("exists");
        let built_at = meta.built_at.unwrap();

        // Add 15 documents after the build (above default threshold of 10).
        for i in 0..15 {
            insert_doc(
                &storage,
                &format!("doc-{i}"),
                built_at + 1000,
                built_at + 1000,
            );
        }

        let detector = StorageBackedStaleness::with_defaults(storage);
        let report = detector.check("idx", Some("v1.0")).expect("check");

        assert!(report.is_stale);
        assert_eq!(report.level, StalenessLevel::Significant);
        assert!(matches!(
            report.recommended_action,
            RecommendedAction::IncrementalUpdate { doc_count: 15 }
        ));
    }

    #[test]
    fn minor_changes_below_threshold() {
        let storage = test_storage();
        let params = sample_build_params("idx");
        storage.record_index_build(&params).expect("build");

        let meta = storage
            .get_index_metadata("idx")
            .expect("get")
            .expect("exists");
        let built_at = meta.built_at.unwrap();

        // Add documents below the 10% threshold and min_change_threshold.
        // With 100+ total docs and only 3 new, this is Minor.
        for i in 0..100 {
            insert_doc(
                &storage,
                &format!("existing-{i}"),
                built_at - 10000,
                built_at - 10000,
            );
        }
        for i in 0..3 {
            insert_doc(
                &storage,
                &format!("new-{i}"),
                built_at + 1000,
                built_at + 1000,
            );
        }

        let detector = StorageBackedStaleness::with_defaults(storage);
        let report = detector.check("idx", Some("v1.0")).expect("check");

        assert!(report.is_stale);
        assert_eq!(report.level, StalenessLevel::Minor);
    }

    #[test]
    fn embedder_change_is_critical() {
        let storage = test_storage();
        let params = sample_build_params("idx");
        storage.record_index_build(&params).expect("build");

        let detector = StorageBackedStaleness::with_defaults(storage);
        let report = detector.check("idx", Some("v2.0")).expect("check");

        assert!(report.is_stale);
        assert_eq!(report.level, StalenessLevel::Critical);
        assert!(matches!(
            report.recommended_action,
            RecommendedAction::FullRebuild { .. }
        ));
    }

    #[test]
    fn quick_check_detects_pending_embeddings() {
        let storage = test_storage();
        insert_doc(&storage, "doc-1", 1_000_000, 1_000_000);
        // No embedding status means it's "pending" (implicit).

        let detector = StorageBackedStaleness::with_defaults(storage);
        let quick = detector.quick_check("potion-128m").expect("quick check");

        assert!(quick.is_stale);
        assert!(quick.pending_count > 0);
    }

    #[test]
    fn quick_check_all_embedded_is_fresh() {
        let storage = test_storage();
        insert_doc(&storage, "doc-1", 1_000_000, 1_000_000);
        storage.mark_embedded("doc-1", "potion-128m").expect("mark");

        let detector = StorageBackedStaleness::with_defaults(storage);
        let quick = detector.quick_check("potion-128m").expect("quick check");

        assert!(!quick.is_stale);
        assert_eq!(quick.pending_count, 0);
    }

    #[test]
    fn large_change_triggers_full_rebuild() {
        let storage = test_storage();
        let params = sample_build_params("idx");
        storage.record_index_build(&params).expect("build");

        let meta = storage
            .get_index_metadata("idx")
            .expect("get")
            .expect("exists");
        let built_at = meta.built_at.unwrap();

        // Insert 10 "existing" docs and 5 new ones => 50% change rate > 30% threshold.
        for i in 0..10 {
            insert_doc(
                &storage,
                &format!("old-{i}"),
                built_at - 10000,
                built_at - 10000,
            );
        }
        for i in 0..5 {
            insert_doc(
                &storage,
                &format!("new-{i}"),
                built_at + 1000,
                built_at + 1000,
            );
        }

        let detector = StorageBackedStaleness::with_defaults(storage);
        let report = detector.check("idx", Some("v1.0")).expect("check");

        assert!(report.is_stale);
        // 5 new out of 15 total = 33% > 30% threshold → FullRebuild.
        assert!(
            matches!(
                report.recommended_action,
                RecommendedAction::FullRebuild { .. }
            ),
            "expected FullRebuild for 33% change rate, got {:?}",
            report.recommended_action
        );
    }

    #[test]
    fn disabled_model_revision_check() {
        let storage = test_storage();
        let params = sample_build_params("idx");
        storage.record_index_build(&params).expect("build");

        let config = StalenessConfig {
            check_model_revision: false,
            ..StalenessConfig::default()
        };
        let detector = StorageBackedStaleness::new(storage, config);

        // Even though revision changed from v1.0 to v2.0, with check disabled
        // it should not report embedder change.
        let report = detector.check("idx", Some("v2.0")).expect("check");

        assert!(!report.is_stale);
        assert!(!report.reasons.contains(&StalenessReason::EmbedderChanged));
    }

    #[test]
    fn schema_mismatch_is_critical_when_enabled() {
        let storage = test_storage();
        let mut params = sample_build_params("idx");
        params.schema_version = Some(SCHEMA_VERSION.saturating_sub(1));
        storage.record_index_build(&params).expect("build");

        let detector = StorageBackedStaleness::with_defaults(storage);
        let report = detector.check("idx", Some("v1.0")).expect("check");

        assert!(report.is_stale);
        assert_eq!(report.level, StalenessLevel::Critical);
        assert!(report.reasons.contains(&StalenessReason::SchemaChanged));
        assert!(matches!(
            report.recommended_action,
            RecommendedAction::FullRebuild { .. }
        ));
    }

    #[test]
    fn schema_mismatch_is_ignored_when_disabled() {
        let storage = test_storage();
        let mut params = sample_build_params("idx");
        params.schema_version = Some(SCHEMA_VERSION.saturating_sub(1));
        storage.record_index_build(&params).expect("build");

        let detector = StorageBackedStaleness::new(
            storage,
            StalenessConfig {
                check_schema_version: false,
                ..StalenessConfig::default()
            },
        );
        let report = detector.check("idx", Some("v1.0")).expect("check");

        assert!(!report.is_stale);
        assert!(!report.reasons.contains(&StalenessReason::SchemaChanged));
    }

    #[test]
    fn index_age_limit_forces_full_rebuild() {
        let storage = test_storage();
        let params = sample_build_params("idx");
        storage.record_index_build(&params).expect("build");

        std::thread::sleep(Duration::from_millis(1_100));

        let detector = StorageBackedStaleness::new(
            storage,
            StalenessConfig {
                max_index_age_secs: Some(0),
                ..StalenessConfig::default()
            },
        );
        let report = detector.check("idx", Some("v1.0")).expect("check");

        assert!(report.is_stale);
        assert_eq!(report.level, StalenessLevel::Critical);
        assert!(report.reasons.contains(&StalenessReason::AgeExceeded));
        assert!(matches!(
            report.recommended_action,
            RecommendedAction::FullRebuild { .. }
        ));
    }

    #[test]
    fn index_age_limit_uses_subsecond_precision() {
        let storage = test_storage();
        let params = sample_build_params("idx");
        storage.record_index_build(&params).expect("build");

        // Sleep slightly above one second so truncating to whole seconds would miss it.
        std::thread::sleep(Duration::from_millis(1_100));

        let detector = StorageBackedStaleness::new(
            storage,
            StalenessConfig {
                max_index_age_secs: Some(1),
                ..StalenessConfig::default()
            },
        );
        let report = detector.check("idx", Some("v1.0")).expect("check");

        assert!(
            report.reasons.contains(&StalenessReason::AgeExceeded),
            "age threshold should be evaluated with subsecond precision"
        );
    }

    #[test]
    fn report_includes_stats() {
        let storage = test_storage();

        for i in 0..5 {
            insert_doc(&storage, &format!("doc-{i}"), 1_000_000, 1_000_000);
        }

        let params = sample_build_params("idx");
        storage.record_index_build(&params).expect("build");

        let detector = StorageBackedStaleness::with_defaults(storage);
        let report = detector.check("idx", Some("v1.0")).expect("check");

        assert_eq!(report.stats.total_documents, 5);
        assert!(report.stats.last_build_duration.is_some());
        assert_eq!(
            report.stats.last_build_duration,
            Some(Duration::from_millis(500))
        );
    }

    #[test]
    fn compute_level_empty_reasons() {
        let config = StalenessConfig::default();
        let level = compute_level(&[], 0, 100, &config);
        assert_eq!(level, StalenessLevel::None);
    }

    #[test]
    fn staleness_level_ordering() {
        assert!(StalenessLevel::None < StalenessLevel::Minor);
        assert!(StalenessLevel::Minor < StalenessLevel::Significant);
        assert!(StalenessLevel::Significant < StalenessLevel::Critical);
    }
}
