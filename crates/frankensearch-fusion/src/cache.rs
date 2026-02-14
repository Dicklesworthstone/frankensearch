//! Index staleness detection and cache management.
//!
//! [`IndexCache`] wraps a [`TwoTierIndex`] and adds:
//! - Staleness detection via [`StalenessDetector`]
//! - Atomic index replacement for background rebuilds
//!
//! The [`StalenessDetector`] trait abstracts how staleness is determined.
//! The default implementation ([`SentinelFileDetector`]) compares a JSON
//! sentinel file against the current index state.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use frankensearch_core::config::TwoTierConfig;
use frankensearch_core::{SearchError, SearchResult};
use serde::{Deserialize, Serialize};
use tracing::debug;

use frankensearch_index::TwoTierIndex;

/// Sentinel file name written alongside indices after a successful build.
pub const SENTINEL_FILENAME: &str = ".frankensearch_index_meta";

/// Current sentinel file format version.
pub const SENTINEL_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Staleness report
// ---------------------------------------------------------------------------

/// Report describing the staleness state of an index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStaleness {
    /// Whether the index is considered stale.
    pub is_stale: bool,
    /// Number of records in the current index.
    pub index_record_count: usize,
    /// Expected number of source documents, if known.
    pub estimated_source_count: Option<usize>,
    /// Human-readable reason for staleness.
    pub reason: Option<String>,
}

impl IndexStaleness {
    /// Create a report indicating the index is fresh.
    #[must_use]
    pub const fn fresh(record_count: usize) -> Self {
        Self {
            is_stale: false,
            index_record_count: record_count,
            estimated_source_count: None,
            reason: None,
        }
    }

    /// Create a report indicating the index is stale.
    #[must_use]
    pub fn stale(record_count: usize, reason: impl Into<String>) -> Self {
        Self {
            is_stale: true,
            index_record_count: record_count,
            estimated_source_count: None,
            reason: Some(reason.into()),
        }
    }
}

// ---------------------------------------------------------------------------
// Sentinel file
// ---------------------------------------------------------------------------

/// JSON sentinel written after a successful index build.
///
/// This file is the source of truth for staleness detection: if it is missing
/// or its contents don't match the current state, the index is considered stale.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexSentinel {
    /// Format version for forward compatibility.
    pub version: u32,
    /// ISO-8601 timestamp when the index was built.
    pub built_at: String,
    /// Number of documents indexed.
    pub source_count: usize,
    /// Optional hash of the source document list for change detection.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_hash: Option<String>,
    /// Embedder used for fast tier.
    pub fast_embedder: String,
    /// Embedder used for quality tier, if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality_embedder: Option<String>,
    /// Fast-tier vector dimensionality.
    pub fast_dimension: usize,
    /// Quality-tier vector dimensionality, if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality_dimension: Option<usize>,
}

impl IndexSentinel {
    /// Write this sentinel to a JSON file.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::Io` on write failure.
    pub fn write_to(&self, dir: &Path) -> SearchResult<()> {
        let path = dir.join(SENTINEL_FILENAME);
        let json = serde_json::to_string_pretty(self).map_err(|e| SearchError::InvalidConfig {
            field: "sentinel".to_owned(),
            value: "<serialization>".to_owned(),
            reason: e.to_string(),
        })?;
        std::fs::write(&path, json)?;
        debug!(
            target: "frankensearch.cache",
            path = %path.display(),
            source_count = self.source_count,
            "wrote index sentinel"
        );
        Ok(())
    }

    /// Read a sentinel from a directory, if present.
    ///
    /// Returns `None` if the file does not exist.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::Io` for I/O failures other than missing file,
    /// or `SearchError::InvalidConfig` for malformed JSON.
    pub fn read_from(dir: &Path) -> SearchResult<Option<Self>> {
        let path = dir.join(SENTINEL_FILENAME);
        if !path.exists() {
            return Ok(None);
        }
        let data = std::fs::read_to_string(&path)?;
        let sentinel: Self =
            serde_json::from_str(&data).map_err(|e| SearchError::InvalidConfig {
                field: "sentinel".to_owned(),
                value: path.display().to_string(),
                reason: format!("malformed sentinel JSON: {e}"),
            })?;
        Ok(Some(sentinel))
    }
}

// ---------------------------------------------------------------------------
// Staleness detector trait
// ---------------------------------------------------------------------------

/// Determines whether the index at a given path is stale.
///
/// Default implementation: [`SentinelFileDetector`] uses the JSON sentinel file.
/// When the `storage` feature is enabled, a storage-backed implementation can
/// be substituted (queries the document database for more accurate detection).
pub trait StalenessDetector: Send + Sync {
    /// Full staleness check returning a detailed report.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if the check itself fails (e.g., I/O error reading sentinel).
    fn check(&self, index_dir: &Path, index: &TwoTierIndex) -> SearchResult<IndexStaleness>;

    /// Quick boolean check: is the index stale?
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if the underlying check fails.
    fn is_stale(&self, index_dir: &Path, index: &TwoTierIndex) -> SearchResult<bool> {
        Ok(self.check(index_dir, index)?.is_stale)
    }
}

impl fmt::Debug for dyn StalenessDetector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("dyn StalenessDetector").finish()
    }
}

/// File-based staleness detector using the JSON sentinel file.
///
/// Staleness is detected when:
/// 1. The sentinel file is missing (first run or deleted)
/// 2. The document count in the sentinel differs from the current index
/// 3. A source hash is provided and differs from the sentinel
#[derive(Debug, Clone)]
pub struct SentinelFileDetector {
    /// Expected source document count, if known by the caller.
    expected_source_count: Option<usize>,
    /// Expected source hash, if known by the caller.
    expected_source_hash: Option<String>,
}

impl SentinelFileDetector {
    /// Create a detector with no expected counts (sentinel-only checks).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            expected_source_count: None,
            expected_source_hash: None,
        }
    }

    /// Set the expected source document count for count-mismatch detection.
    #[must_use]
    pub const fn with_expected_count(mut self, count: usize) -> Self {
        self.expected_source_count = Some(count);
        self
    }

    /// Set the expected source hash for change detection.
    #[must_use]
    pub fn with_expected_hash(mut self, hash: impl Into<String>) -> Self {
        self.expected_source_hash = Some(hash.into());
        self
    }
}

impl Default for SentinelFileDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl StalenessDetector for SentinelFileDetector {
    fn check(&self, index_dir: &Path, index: &TwoTierIndex) -> SearchResult<IndexStaleness> {
        let Some(sentinel) = IndexSentinel::read_from(index_dir)? else {
            debug!(
                target: "frankensearch.cache",
                dir = %index_dir.display(),
                "sentinel file missing, index is stale"
            );
            return Ok(IndexStaleness::stale(
                index.doc_count(),
                "sentinel file missing (first run or deleted)",
            ));
        };

        // Check document count mismatch between sentinel and index
        if sentinel.source_count != index.doc_count() {
            return Ok(IndexStaleness::stale(
                index.doc_count(),
                format!(
                    "document count mismatch: sentinel has {}, index has {}",
                    sentinel.source_count,
                    index.doc_count()
                ),
            ));
        }

        // Check expected source count from caller
        if let Some(expected) = self.expected_source_count
            && expected != index.doc_count()
        {
            return Ok(IndexStaleness {
                is_stale: true,
                index_record_count: index.doc_count(),
                estimated_source_count: Some(expected),
                reason: Some(format!(
                    "source count mismatch: expected {expected}, index has {}",
                    index.doc_count()
                )),
            });
        }

        // Check source hash from caller
        if let Some(ref expected_hash) = self.expected_source_hash
            && let Some(ref sentinel_hash) = sentinel.source_hash
            && expected_hash != sentinel_hash
        {
            return Ok(IndexStaleness::stale(
                index.doc_count(),
                format!(
                    "source hash mismatch: expected '{expected_hash}', sentinel has '{sentinel_hash}'"
                ),
            ));
        }

        Ok(IndexStaleness::fresh(index.doc_count()))
    }
}

// ---------------------------------------------------------------------------
// Index cache
// ---------------------------------------------------------------------------

/// Cached, atomically-replaceable wrapper around [`TwoTierIndex`].
///
/// Uses [`Arc`] + [`RwLock`] for lock-free reads and atomic replacement.
/// Readers hold an `Arc` clone and are never blocked by a concurrent refresh.
///
/// # Usage pattern
///
/// ```rust,ignore
/// let cache = IndexCache::open(dir, config, detector)?;
///
/// // Read path (cheap, non-blocking):
/// let index = cache.current();
/// let hits = index.search_fast(&query, 10)?;
///
/// // Check staleness:
/// if cache.is_stale()? {
///     // Rebuild and replace atomically:
///     let new_index = TwoTierIndex::open(&dir, config)?;
///     cache.replace(new_index);
/// }
/// ```
#[derive(Debug)]
pub struct IndexCache {
    /// Current index behind an atomic swap.
    inner: RwLock<Arc<TwoTierIndex>>,
    /// Staleness detection strategy.
    detector: Box<dyn StalenessDetector>,
    /// Directory containing the index files.
    dir: PathBuf,
    /// Configuration for opening replacement indices.
    config: TwoTierConfig,
}

impl IndexCache {
    /// Open an index cache from a directory.
    ///
    /// Eagerly loads the `TwoTierIndex` and associates it with the given
    /// staleness detector.
    ///
    /// # Errors
    ///
    /// Returns errors from `TwoTierIndex::open`.
    pub fn open(
        dir: &Path,
        config: TwoTierConfig,
        detector: Box<dyn StalenessDetector>,
    ) -> SearchResult<Self> {
        let index = TwoTierIndex::open(dir, config.clone())?;
        debug!(
            target: "frankensearch.cache",
            dir = %dir.display(),
            doc_count = index.doc_count(),
            "index cache opened"
        );
        Ok(Self {
            inner: RwLock::new(Arc::new(index)),
            detector,
            dir: dir.to_path_buf(),
            config,
        })
    }

    /// Get a snapshot of the current index.
    ///
    /// Returns an `Arc<TwoTierIndex>` that remains valid even if the cache
    /// is refreshed concurrently. Readers holding this reference will continue
    /// to use the old index until they drop it.
    ///
    /// # Panics
    ///
    /// Panics if the internal `RwLock` is poisoned.
    #[must_use]
    pub fn current(&self) -> Arc<TwoTierIndex> {
        self.inner
            .read()
            .expect("index cache rwlock poisoned")
            .clone()
    }

    /// Atomically replace the cached index with a new one.
    ///
    /// Existing readers holding `Arc<TwoTierIndex>` from [`current()`](Self::current)
    /// are unaffected. The old index is dropped when its last `Arc` reference
    /// goes out of scope.
    ///
    /// # Panics
    ///
    /// Panics if the internal `RwLock` is poisoned.
    pub fn replace(&self, new_index: TwoTierIndex) {
        let mut guard = self.inner.write().expect("index cache rwlock poisoned");
        debug!(
            target: "frankensearch.cache",
            dir = %self.dir.display(),
            old_count = guard.doc_count(),
            new_count = new_index.doc_count(),
            "replacing cached index"
        );
        *guard = Arc::new(new_index);
    }

    /// Reload the index from disk and atomically replace the cached version.
    ///
    /// # Errors
    ///
    /// Returns errors from `TwoTierIndex::open`.
    pub fn reload(&self) -> SearchResult<()> {
        let new_index = TwoTierIndex::open(&self.dir, self.config.clone())?;
        self.replace(new_index);
        Ok(())
    }

    /// Check whether the current index is stale.
    ///
    /// # Errors
    ///
    /// Returns errors from the staleness detector.
    pub fn check_staleness(&self) -> SearchResult<IndexStaleness> {
        let index = self.current();
        self.detector.check(&self.dir, &index)
    }

    /// Quick boolean staleness check.
    ///
    /// # Errors
    ///
    /// Returns errors from the staleness detector.
    pub fn is_stale(&self) -> SearchResult<bool> {
        let index = self.current();
        self.detector.is_stale(&self.dir, &index)
    }

    /// Directory containing the index files.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Reference to the current configuration.
    #[must_use]
    pub const fn config(&self) -> &TwoTierConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use frankensearch_index::{
        Quantization, TwoTierIndex, VECTOR_INDEX_FAST_FILENAME, VectorIndex,
    };

    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "frankensearch-cache-{name}-{}-{now}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn write_fast_index(dir: &Path, records: &[(&str, Vec<f32>)]) {
        let dim = records.first().map_or(4, |(_, v)| v.len());
        let path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        let mut writer =
            VectorIndex::create_with_revision(&path, "potion-128M", "v1", dim, Quantization::F16)
                .expect("writer");
        for (doc_id, vec) in records {
            writer.write_record(doc_id, vec).expect("write");
        }
        writer.finish().expect("finish");
    }

    fn sample_records() -> Vec<(&'static str, Vec<f32>)> {
        vec![
            ("doc-a", vec![1.0, 0.0, 0.0, 0.0]),
            ("doc-b", vec![0.8, 0.6, 0.0, 0.0]),
            ("doc-c", vec![0.0, 0.0, 1.0, 0.0]),
        ]
    }

    fn sample_sentinel(count: usize) -> IndexSentinel {
        IndexSentinel {
            version: SENTINEL_VERSION,
            built_at: "2026-01-15T10:30:00Z".to_owned(),
            source_count: count,
            source_hash: None,
            fast_embedder: "potion-128M".to_owned(),
            quality_embedder: Some("MiniLM-L6-v2".to_owned()),
            fast_dimension: 4,
            quality_dimension: Some(8),
        }
    }

    // ── Sentinel tests ──────────────────────────────────────────────

    #[test]
    fn sentinel_write_read_roundtrip() {
        let dir = temp_dir("sentinel-roundtrip");
        let sentinel = sample_sentinel(100);
        sentinel.write_to(&dir).expect("write");
        let read = IndexSentinel::read_from(&dir)
            .expect("read")
            .expect("sentinel should exist");
        assert_eq!(read.source_count, 100);
        assert_eq!(read.fast_embedder, "potion-128M");
        assert_eq!(read.quality_embedder.as_deref(), Some("MiniLM-L6-v2"));
        assert_eq!(read.version, SENTINEL_VERSION);
    }

    #[test]
    fn sentinel_missing_returns_none() {
        let dir = temp_dir("sentinel-missing");
        let result = IndexSentinel::read_from(&dir).expect("no io error");
        assert!(result.is_none());
    }

    #[test]
    fn sentinel_malformed_returns_error() {
        let dir = temp_dir("sentinel-malformed");
        std::fs::write(dir.join(SENTINEL_FILENAME), "not json at all").expect("write");
        let err = IndexSentinel::read_from(&dir).expect_err("should fail");
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[test]
    fn sentinel_with_source_hash() {
        let dir = temp_dir("sentinel-hash");
        let sentinel = IndexSentinel {
            source_hash: Some("sha256:abc123".to_owned()),
            ..sample_sentinel(50)
        };
        sentinel.write_to(&dir).expect("write");
        let read = IndexSentinel::read_from(&dir)
            .expect("read")
            .expect("exists");
        assert_eq!(read.source_hash.as_deref(), Some("sha256:abc123"));
    }

    // ── StalenessDetector tests ─────────────────────────────────────

    #[test]
    fn sentinel_detector_missing_sentinel_is_stale() {
        let dir = temp_dir("stale-missing");
        write_fast_index(&dir, &sample_records());

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        let detector = SentinelFileDetector::new();
        let report = detector.check(&dir, &index).expect("check");
        assert!(report.is_stale);
        assert!(
            report
                .reason
                .as_deref()
                .unwrap()
                .contains("sentinel file missing")
        );
    }

    #[test]
    fn sentinel_detector_matching_sentinel_is_fresh() {
        let dir = temp_dir("stale-fresh");
        write_fast_index(&dir, &sample_records());
        sample_sentinel(3).write_to(&dir).expect("write sentinel");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        let detector = SentinelFileDetector::new();
        let report = detector.check(&dir, &index).expect("check");
        assert!(!report.is_stale);
        assert!(report.reason.is_none());
    }

    #[test]
    fn sentinel_detector_count_mismatch_is_stale() {
        let dir = temp_dir("stale-count");
        write_fast_index(&dir, &sample_records());
        // Sentinel says 10 docs, but index has 3
        sample_sentinel(10).write_to(&dir).expect("write sentinel");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        let detector = SentinelFileDetector::new();
        let report = detector.check(&dir, &index).expect("check");
        assert!(report.is_stale);
        assert!(
            report
                .reason
                .as_deref()
                .unwrap()
                .contains("document count mismatch")
        );
    }

    #[test]
    fn sentinel_detector_expected_count_mismatch() {
        let dir = temp_dir("stale-expected-count");
        write_fast_index(&dir, &sample_records());
        sample_sentinel(3).write_to(&dir).expect("write sentinel");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        let detector = SentinelFileDetector::new().with_expected_count(100);
        let report = detector.check(&dir, &index).expect("check");
        assert!(report.is_stale);
        assert_eq!(report.estimated_source_count, Some(100));
    }

    #[test]
    fn sentinel_detector_expected_hash_mismatch() {
        let dir = temp_dir("stale-hash-mismatch");
        write_fast_index(&dir, &sample_records());
        let sentinel = IndexSentinel {
            source_hash: Some("sha256:old_hash".to_owned()),
            ..sample_sentinel(3)
        };
        sentinel.write_to(&dir).expect("write sentinel");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        let detector = SentinelFileDetector::new().with_expected_hash("sha256:new_hash");
        let report = detector.check(&dir, &index).expect("check");
        assert!(report.is_stale);
        assert!(
            report
                .reason
                .as_deref()
                .unwrap()
                .contains("source hash mismatch")
        );
    }

    #[test]
    fn sentinel_detector_expected_hash_matches() {
        let dir = temp_dir("stale-hash-ok");
        write_fast_index(&dir, &sample_records());
        let sentinel = IndexSentinel {
            source_hash: Some("sha256:same_hash".to_owned()),
            ..sample_sentinel(3)
        };
        sentinel.write_to(&dir).expect("write sentinel");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        let detector = SentinelFileDetector::new().with_expected_hash("sha256:same_hash");
        let report = detector.check(&dir, &index).expect("check");
        assert!(!report.is_stale);
    }

    #[test]
    fn is_stale_shorthand() {
        let dir = temp_dir("is-stale");
        write_fast_index(&dir, &sample_records());

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        let detector = SentinelFileDetector::new();
        // No sentinel = stale
        assert!(detector.is_stale(&dir, &index).expect("check"));
    }

    // ── IndexStaleness constructors ─────────────────────────────────

    #[test]
    fn staleness_fresh_constructor() {
        let report = IndexStaleness::fresh(42);
        assert!(!report.is_stale);
        assert_eq!(report.index_record_count, 42);
        assert!(report.reason.is_none());
    }

    #[test]
    fn staleness_stale_constructor() {
        let report = IndexStaleness::stale(7, "test reason");
        assert!(report.is_stale);
        assert_eq!(report.index_record_count, 7);
        assert_eq!(report.reason.as_deref(), Some("test reason"));
    }

    // ── IndexCache tests ────────────────────────────────────────────

    #[test]
    fn cache_open_and_current() {
        let dir = temp_dir("cache-open");
        write_fast_index(&dir, &sample_records());

        let cache = IndexCache::open(
            &dir,
            TwoTierConfig::default(),
            Box::new(SentinelFileDetector::new()),
        )
        .expect("open cache");

        let index = cache.current();
        assert_eq!(index.doc_count(), 3);
    }

    #[test]
    fn cache_replace_atomic() {
        let dir = temp_dir("cache-replace");
        write_fast_index(&dir, &sample_records());

        let cache = IndexCache::open(
            &dir,
            TwoTierConfig::default(),
            Box::new(SentinelFileDetector::new()),
        )
        .expect("open cache");

        // Grab reference to old index
        let old = cache.current();
        assert_eq!(old.doc_count(), 3);

        // Write a new index with more records
        write_fast_index(
            &dir,
            &[
                ("doc-1", vec![1.0, 0.0, 0.0, 0.0]),
                ("doc-2", vec![0.0, 1.0, 0.0, 0.0]),
                ("doc-3", vec![0.0, 0.0, 1.0, 0.0]),
                ("doc-4", vec![0.0, 0.0, 0.0, 1.0]),
                ("doc-5", vec![0.5, 0.5, 0.0, 0.0]),
            ],
        );

        // Replace with reloaded index
        let new_index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("reopen");
        cache.replace(new_index);

        // Old reference still works with old count
        assert_eq!(old.doc_count(), 3);
        // New reference has updated count
        let fresh = cache.current();
        assert_eq!(fresh.doc_count(), 5);
    }

    #[test]
    fn cache_reload() {
        let dir = temp_dir("cache-reload");
        write_fast_index(&dir, &sample_records());

        let cache = IndexCache::open(
            &dir,
            TwoTierConfig::default(),
            Box::new(SentinelFileDetector::new()),
        )
        .expect("open cache");

        assert_eq!(cache.current().doc_count(), 3);

        // Write a new fast index with 2 records
        write_fast_index(
            &dir,
            &[
                ("doc-x", vec![1.0, 0.0, 0.0, 0.0]),
                ("doc-y", vec![0.0, 1.0, 0.0, 0.0]),
            ],
        );

        cache.reload().expect("reload");
        assert_eq!(cache.current().doc_count(), 2);
    }

    #[test]
    fn cache_check_staleness_no_sentinel() {
        let dir = temp_dir("cache-stale");
        write_fast_index(&dir, &sample_records());

        let cache = IndexCache::open(
            &dir,
            TwoTierConfig::default(),
            Box::new(SentinelFileDetector::new()),
        )
        .expect("open cache");

        let report = cache.check_staleness().expect("check");
        assert!(report.is_stale);
        assert!(cache.is_stale().expect("is_stale"));
    }

    #[test]
    fn cache_check_staleness_with_sentinel() {
        let dir = temp_dir("cache-fresh");
        write_fast_index(&dir, &sample_records());
        sample_sentinel(3).write_to(&dir).expect("sentinel");

        let cache = IndexCache::open(
            &dir,
            TwoTierConfig::default(),
            Box::new(SentinelFileDetector::new()),
        )
        .expect("open cache");

        let report = cache.check_staleness().expect("check");
        assert!(!report.is_stale);
        assert!(!cache.is_stale().expect("is_stale"));
    }

    #[test]
    fn cache_dir_and_config() {
        let dir = temp_dir("cache-accessors");
        write_fast_index(&dir, &sample_records());

        let config = TwoTierConfig {
            rrf_k: 99.0,
            ..Default::default()
        };
        let cache = IndexCache::open(&dir, config, Box::new(SentinelFileDetector::new()))
            .expect("open cache");

        assert_eq!(cache.dir(), dir);
        assert!((cache.config().rrf_k - 99.0).abs() < f64::EPSILON);
    }

    #[test]
    fn cache_missing_dir_returns_error() {
        let dir = std::env::temp_dir().join("frankensearch-cache-nonexistent-xyz");
        let err = IndexCache::open(
            &dir,
            TwoTierConfig::default(),
            Box::new(SentinelFileDetector::new()),
        )
        .expect_err("should fail");
        assert!(matches!(err, SearchError::IndexNotFound { .. }));
    }

    #[test]
    fn sentinel_serde_roundtrip() {
        let sentinel = IndexSentinel {
            version: SENTINEL_VERSION,
            built_at: "2026-02-13T12:00:00Z".to_owned(),
            source_count: 42,
            source_hash: Some("sha256:deadbeef".to_owned()),
            fast_embedder: "potion-128M".to_owned(),
            quality_embedder: None,
            fast_dimension: 256,
            quality_dimension: None,
        };
        let json = serde_json::to_string(&sentinel).expect("serialize");
        let rt: IndexSentinel = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(rt.source_count, 42);
        assert_eq!(rt.source_hash.as_deref(), Some("sha256:deadbeef"));
        assert!(rt.quality_embedder.is_none());
        assert!(rt.quality_dimension.is_none());
    }

    #[test]
    fn sentinel_detector_expected_hash_but_sentinel_has_no_hash() {
        // Expected hash set, but sentinel lacks source_hash -> should be fresh
        // because the hash comparison only triggers when BOTH are present.
        let dir = temp_dir("stale-hash-missing-sentinel");
        write_fast_index(&dir, &sample_records());
        // Sentinel with no source_hash (default from sample_sentinel)
        sample_sentinel(3).write_to(&dir).expect("write sentinel");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        let detector = SentinelFileDetector::new().with_expected_hash("sha256:anything");
        let report = detector.check(&dir, &index).expect("check");
        // No sentinel hash to compare against, so not stale from hash mismatch.
        assert!(!report.is_stale);
    }

    #[test]
    fn sentinel_detector_count_mismatch_takes_priority_over_hash() {
        // When sentinel count differs from index, it triggers before hash check.
        let dir = temp_dir("stale-count-before-hash");
        write_fast_index(&dir, &sample_records());
        let sentinel = IndexSentinel {
            source_hash: Some("sha256:old".to_owned()),
            ..sample_sentinel(999) // Mismatched count
        };
        sentinel.write_to(&dir).expect("write sentinel");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        let detector = SentinelFileDetector::new().with_expected_hash("sha256:new");
        let report = detector.check(&dir, &index).expect("check");
        assert!(report.is_stale);
        assert!(
            report
                .reason
                .as_deref()
                .unwrap()
                .contains("document count mismatch")
        );
    }

    #[test]
    fn staleness_fresh_has_no_estimated_source_count() {
        let report = IndexStaleness::fresh(10);
        assert!(report.estimated_source_count.is_none());
    }

    #[test]
    fn staleness_serde_roundtrip() {
        let report = IndexStaleness::stale(42, "content changed");
        let json = serde_json::to_string(&report).expect("serialize");
        let rt: IndexStaleness = serde_json::from_str(&json).expect("deserialize");
        assert!(rt.is_stale);
        assert_eq!(rt.index_record_count, 42);
        assert_eq!(rt.reason.as_deref(), Some("content changed"));
    }

    #[test]
    fn sentinel_zero_source_count_roundtrips() {
        let dir = temp_dir("sentinel-zero-count");
        let sentinel = sample_sentinel(0);
        sentinel.write_to(&dir).expect("write");
        let read = IndexSentinel::read_from(&dir)
            .expect("read")
            .expect("exists");
        assert_eq!(read.source_count, 0);
    }
}
