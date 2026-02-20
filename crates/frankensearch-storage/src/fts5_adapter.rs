//! FTS5 alternative lexical search adapter.
//!
//! Uses `FrankenSQLite`'s built-in FTS5 implementation as an alternative to
//! Tantivy for BM25 full-text search. Both implement the [`LexicalSearch`]
//! trait from `frankensearch-core`.
//!
//! # Advantages over Tantivy
//!
//! - Zero additional binary size (FTS5 is part of `FrankenSQLite`)
//! - MVCC concurrent reads and writes
//! - Single deployment artifact (one `.db` file)
//!
//! # Content mode
//!
//! Only `Stored` and `Contentless` modes are supported. External content mode
//! is NOT available in `FrankenSQLite` V1.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use asupersync::Cx;
use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::traits::{LexicalSearch, SearchFuture};
use frankensearch_core::types::{IndexableDocument, ScoreSource, ScoredResult};
use fsqlite_ext_fts5::{Fts5Table, snippet as fts5_snippet};
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument, warn};

// ─── Constants ──────────────────────────────────────────────────────────────

/// BM25 boost applied to title field matches (mirrors Tantivy adapter).
const TITLE_BOOST: f64 = 2.0;

/// Maximum query length in characters before truncation.
const MAX_QUERY_LENGTH: usize = 10_000;

/// Default snippet window size in tokens.
const DEFAULT_SNIPPET_TOKENS: usize = 20;

/// Column index: content (primary search field).
const COL_CONTENT: usize = 2;
/// Column index: `metadata_json` (stored, not searched).
const COL_METADATA: usize = 3;

// ─── Configuration ──────────────────────────────────────────────────────────

/// Content storage mode for the FTS5 index.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Fts5ContentMode {
    /// FTS5 stores its own copy of the content (supports snippets).
    #[default]
    Stored,
    /// Index-only mode — no content retrieval or snippet support.
    Contentless,
}

/// Tokenizer selection for the FTS5 index.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Fts5TokenizerChoice {
    /// Unicode-aware tokenizer with optional diacritic removal.
    #[default]
    Unicode61,
    /// English Porter stemming (wraps unicode61).
    Porter,
    /// Trigram tokenizer for substring matching (slower but more flexible).
    Trigram,
}

/// Configuration for the FTS5 lexical search adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fts5AdapterConfig {
    /// Content storage mode.
    #[serde(default)]
    pub content_mode: Fts5ContentMode,
    /// Tokenizer to use.
    #[serde(default)]
    pub tokenizer: Fts5TokenizerChoice,
    /// BM25 boost for title field matches.
    #[serde(default = "default_title_boost")]
    pub title_boost: f64,
}

fn default_title_boost() -> f64 {
    TITLE_BOOST
}

impl Default for Fts5AdapterConfig {
    fn default() -> Self {
        Self {
            content_mode: Fts5ContentMode::default(),
            tokenizer: Fts5TokenizerChoice::default(),
            title_boost: TITLE_BOOST,
        }
    }
}

// ─── Row ID mapping ─────────────────────────────────────────────────────────

/// Maps between string `doc_ids` and i64 rowids required by `Fts5Table`.
#[derive(Debug, Default)]
struct RowIdMap {
    doc_to_row: HashMap<String, i64>,
    row_to_doc: HashMap<i64, String>,
    next_rowid: i64,
}

impl RowIdMap {
    fn new() -> Self {
        Self {
            doc_to_row: HashMap::new(),
            row_to_doc: HashMap::new(),
            next_rowid: 1,
        }
    }

    fn get_or_assign(&mut self, doc_id: &str) -> i64 {
        if let Some(&rowid) = self.doc_to_row.get(doc_id) {
            return rowid;
        }
        let rowid = self.next_rowid;
        self.next_rowid += 1;
        self.doc_to_row.insert(doc_id.to_owned(), rowid);
        self.row_to_doc.insert(rowid, doc_id.to_owned());
        rowid
    }

    fn get_rowid(&self, doc_id: &str) -> Option<i64> {
        self.doc_to_row.get(doc_id).copied()
    }

    fn get_doc_id(&self, rowid: i64) -> Option<&str> {
        self.row_to_doc.get(&rowid).map(String::as_str)
    }

    fn remove(&mut self, doc_id: &str) -> Option<i64> {
        if let Some(rowid) = self.doc_to_row.remove(doc_id) {
            self.row_to_doc.remove(&rowid);
            Some(rowid)
        } else {
            None
        }
    }
}

// ─── FTS5 Lexical Search ────────────────────────────────────────────────────

/// FTS5-backed implementation of [`LexicalSearch`].
///
/// Uses `FrankenSQLite`'s `Fts5Table` directly for full-text indexing
/// and BM25-ranked search. Thread-safe via internal `Mutex`.
pub struct Fts5LexicalSearch {
    table: Mutex<Fts5Table>,
    rowid_map: Mutex<RowIdMap>,
    config: Fts5AdapterConfig,
    doc_count: AtomicUsize,
}

#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for Fts5LexicalSearch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fts5LexicalSearch")
            .field("config", &self.config)
            .field("doc_count", &self.doc_count.load(Ordering::Relaxed))
            .finish()
    }
}

impl Fts5LexicalSearch {
    /// Create a new FTS5 lexical search instance.
    #[must_use]
    pub fn new(config: Fts5AdapterConfig) -> Self {
        let columns = vec![
            "doc_id".to_owned(),
            "title".to_owned(),
            "content".to_owned(),
            "metadata_json".to_owned(),
        ];

        let table = Fts5Table::with_columns(columns);

        Self {
            table: Mutex::new(table),
            rowid_map: Mutex::new(RowIdMap::new()),
            config,
            doc_count: AtomicUsize::new(0),
        }
    }

    /// Create a new FTS5 lexical search instance with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(Fts5AdapterConfig::default())
    }

    /// Get the adapter configuration.
    #[must_use]
    pub fn config(&self) -> &Fts5AdapterConfig {
        &self.config
    }

    /// Truncate overly long queries to prevent pathological parsing.
    fn truncate_query(query: &str) -> &str {
        if query.len() > MAX_QUERY_LENGTH {
            warn!(
                original_len = query.len(),
                max = MAX_QUERY_LENGTH,
                "fts5: query truncated"
            );
            // Truncate at a char boundary.
            let mut end = MAX_QUERY_LENGTH;
            while end > 0 && !query.is_char_boundary(end) {
                end -= 1;
            }
            &query[..end]
        } else {
            query
        }
    }

    /// Build column values from an `IndexableDocument`.
    fn doc_to_columns(doc: &IndexableDocument) -> Vec<String> {
        let metadata_json = if doc.metadata.is_empty() {
            String::new()
        } else {
            serde_json::to_string(&doc.metadata).unwrap_or_default()
        };

        vec![
            doc.id.clone(),
            doc.title.clone().unwrap_or_default(),
            doc.content.clone(),
            metadata_json,
        ]
    }

    /// Search with snippet generation (richer result type).
    #[allow(clippy::significant_drop_tightening)]
    pub fn search_with_snippets(&self, query: &str, limit: usize) -> SearchResult<Vec<Fts5Hit>> {
        let query = Self::truncate_query(query);
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }

        let table = self.table.lock().map_err(lock_error)?;
        let rowid_map = self.rowid_map.lock().map_err(lock_error)?;

        let search_results = table
            .search(query)
            .map_err(|e| SearchError::QueryParseError {
                query: query.to_owned(),
                detail: e.to_string(),
            })?;

        let query_terms: Vec<String> = query
            .split_whitespace()
            .map(|t| t.trim_matches('"').to_lowercase())
            .collect();

        let mut hits = Vec::with_capacity(search_results.len().min(limit));
        for (rank, (rowid, score)) in search_results.into_iter().take(limit).enumerate() {
            let doc_id = rowid_map.get_doc_id(rowid).unwrap_or("").to_owned();

            // FTS5 scores are negative (lower = better). Negate for positive.
            #[allow(clippy::cast_possible_truncation)]
            let bm25_score = (-score) as f32;

            // Generate snippet from content column if available.
            let snippet = table
                .get_document(rowid)
                .and_then(|cols| cols.get(COL_CONTENT))
                .map(|content| {
                    fts5_snippet(
                        content,
                        &query_terms,
                        "<b>",
                        "</b>",
                        "...",
                        DEFAULT_SNIPPET_TOKENS,
                    )
                });

            let metadata = table
                .get_document(rowid)
                .and_then(|cols| cols.get(COL_METADATA))
                .filter(|s| !s.is_empty())
                .and_then(|s| serde_json::from_str(s).ok());

            hits.push(Fts5Hit {
                doc_id,
                bm25_score,
                rank,
                snippet,
                metadata,
            });
        }

        debug!(hits = hits.len(), query, "fts5 search completed");
        Ok(hits)
    }

    /// Delete a single document by ID.
    ///
    /// Returns `true` if the document existed and was removed.
    pub fn delete_document(&self, doc_id: &str) -> SearchResult<bool> {
        let mut table = self.table.lock().map_err(lock_error)?;
        let mut rowid_map = self.rowid_map.lock().map_err(lock_error)?;

        let Some(rowid) = rowid_map.remove(doc_id) else {
            return Ok(false);
        };
        table.delete_document(rowid);
        drop(rowid_map);
        drop(table);
        self.doc_count.fetch_sub(1, Ordering::Relaxed);
        Ok(true)
    }

    /// Delete all indexed documents.
    pub fn clear(&self) -> SearchResult<()> {
        let mut table = self.table.lock().map_err(lock_error)?;
        let mut rowid_map = self.rowid_map.lock().map_err(lock_error)?;

        // Collect all rowids to delete.
        let rowids: Vec<i64> = rowid_map.row_to_doc.keys().copied().collect();
        for rowid in rowids {
            table.delete_document(rowid);
        }
        rowid_map.doc_to_row.clear();
        rowid_map.row_to_doc.clear();
        drop(rowid_map);
        drop(table);
        self.doc_count.store(0, Ordering::Relaxed);

        debug!("fts5: cleared all documents");
        Ok(())
    }
}

// ─── LexicalSearch trait implementation ─────────────────────────────────────

#[allow(clippy::significant_drop_tightening)]
impl LexicalSearch for Fts5LexicalSearch {
    #[instrument(skip_all, fields(query = %query, limit = limit))]
    fn search<'a>(
        &'a self,
        _cx: &'a Cx,
        query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, Vec<ScoredResult>> {
        Box::pin(async move {
            let query = Self::truncate_query(query);

            if query.trim().is_empty() {
                return Ok(Vec::new());
            }

            let table = self.table.lock().map_err(lock_error)?;
            let rowid_map = self.rowid_map.lock().map_err(lock_error)?;

            let search_results = table
                .search(query)
                .map_err(|e| SearchError::QueryParseError {
                    query: query.to_owned(),
                    detail: e.to_string(),
                })?;

            debug!(hits = search_results.len(), "fts5 BM25 search completed");

            let mut results = Vec::with_capacity(search_results.len().min(limit));
            for (rowid, score) in search_results.into_iter().take(limit) {
                let doc_id = rowid_map.get_doc_id(rowid).unwrap_or("").to_owned();

                // FTS5 BM25 scores are negative (lower = better).
                // Negate to produce positive scores (higher = better).
                #[allow(clippy::cast_possible_truncation)]
                let bm25_score = (-score) as f32;

                let metadata = table
                    .get_document(rowid)
                    .and_then(|cols| cols.get(COL_METADATA))
                    .filter(|s| !s.is_empty())
                    .and_then(|s| serde_json::from_str(s).ok());

                results.push(ScoredResult {
                    doc_id,
                    score: bm25_score,
                    source: ScoreSource::Lexical,
                    index: None,
                    fast_score: None,
                    quality_score: None,
                    lexical_score: Some(bm25_score),
                    rerank_score: None,
                    explanation: None,
                    metadata,
                });
            }

            Ok(results)
        })
    }

    fn index_document<'a>(
        &'a self,
        _cx: &'a Cx,
        doc: &'a IndexableDocument,
    ) -> SearchFuture<'a, ()> {
        Box::pin(async move {
            let mut table = self.table.lock().map_err(lock_error)?;
            let mut rowid_map = self.rowid_map.lock().map_err(lock_error)?;

            // Upsert: delete existing document with same ID first.
            if let Some(old_rowid) = rowid_map.get_rowid(&doc.id) {
                table.delete_document(old_rowid);
                // Don't decrement doc_count here — it nets out with the add below.
            } else {
                // Only increment if this is truly new.
                self.doc_count.fetch_add(1, Ordering::Relaxed);
            }

            let rowid = rowid_map.get_or_assign(&doc.id);
            let columns = Self::doc_to_columns(doc);
            table.insert_document(rowid, &columns);

            Ok(())
        })
    }

    fn index_documents<'a>(
        &'a self,
        _cx: &'a Cx,
        docs: &'a [IndexableDocument],
    ) -> SearchFuture<'a, ()> {
        Box::pin(async move {
            let mut table = self.table.lock().map_err(lock_error)?;
            let mut rowid_map = self.rowid_map.lock().map_err(lock_error)?;

            for doc in docs {
                if let Some(old_rowid) = rowid_map.get_rowid(&doc.id) {
                    table.delete_document(old_rowid);
                } else {
                    self.doc_count.fetch_add(1, Ordering::Relaxed);
                }

                let rowid = rowid_map.get_or_assign(&doc.id);
                let columns = Self::doc_to_columns(doc);
                table.insert_document(rowid, &columns);
            }

            debug!(count = docs.len(), "fts5: batch indexed documents");
            Ok(())
        })
    }

    fn commit<'a>(&'a self, _cx: &'a Cx) -> SearchFuture<'a, ()> {
        // FTS5 in-memory table has no separate commit phase.
        Box::pin(async { Ok(()) })
    }

    fn doc_count(&self) -> usize {
        self.doc_count.load(Ordering::Relaxed)
    }
}

// ─── Hit type for snippet-aware search ──────────────────────────────────────

/// A hit from FTS5 search with optional snippet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fts5Hit {
    /// Document identifier.
    pub doc_id: String,
    /// BM25 relevance score (higher = better).
    pub bm25_score: f32,
    /// Position in results (0-indexed).
    pub rank: usize,
    /// Highlighted content snippet around matching terms.
    pub snippet: Option<String>,
    /// Document metadata.
    pub metadata: Option<serde_json::Value>,
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn lock_error<T>(_: T) -> SearchError {
    SearchError::SubsystemError {
        subsystem: "fts5",
        source: Box::new(std::io::Error::other("fts5 mutex poisoned")),
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::future::Future;

    use super::*;

    /// Helper: run async test code with a `Cx` (asupersync, NO tokio).
    fn run_with_cx<F, Fut>(f: F)
    where
        F: FnOnce(Cx) -> Fut,
        Fut: Future<Output = ()>,
    {
        asupersync::test_utils::run_test_with_cx(f);
    }

    fn make_doc(id: &str, content: &str) -> IndexableDocument {
        IndexableDocument::new(id, content)
    }

    fn make_doc_with_title(id: &str, title: &str, content: &str) -> IndexableDocument {
        IndexableDocument::new(id, content).with_title(title)
    }

    fn make_doc_with_metadata(
        id: &str,
        content: &str,
        key: &str,
        value: &str,
    ) -> IndexableDocument {
        IndexableDocument::new(id, content).with_metadata(key, value)
    }

    // -- Construction --

    #[test]
    fn new_instance_is_empty() {
        let search = Fts5LexicalSearch::with_defaults();
        assert_eq!(search.doc_count(), 0);
    }

    #[test]
    fn config_defaults_are_sane() {
        let config = Fts5AdapterConfig::default();
        assert_eq!(config.content_mode, Fts5ContentMode::Stored);
        assert_eq!(config.tokenizer, Fts5TokenizerChoice::Unicode61);
        assert!((config.title_boost - TITLE_BOOST).abs() < f64::EPSILON);
    }

    // -- Indexing --

    #[test]
    fn index_single_document() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            let doc = make_doc("doc1", "hello world of search");
            search.index_document(&cx, &doc).await.unwrap();
            assert_eq!(search.doc_count(), 1);
        });
    }

    #[test]
    fn index_batch_documents() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            let docs = vec![
                make_doc("a", "first document"),
                make_doc("b", "second document"),
                make_doc("c", "third document"),
            ];
            search.index_documents(&cx, &docs).await.unwrap();
            assert_eq!(search.doc_count(), 3);
        });
    }

    #[test]
    fn upsert_replaces_existing_document() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            let doc_v1 = make_doc("doc1", "original content");
            search.index_document(&cx, &doc_v1).await.unwrap();
            assert_eq!(search.doc_count(), 1);

            let doc_v2 = make_doc("doc1", "updated content completely different");
            search.index_document(&cx, &doc_v2).await.unwrap();
            assert_eq!(search.doc_count(), 1);

            // Search should find updated content.
            let results = search.search(&cx, "updated", 10).await.unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].doc_id, "doc1");

            // Old content should not match.
            let results = search.search(&cx, "original", 10).await.unwrap();
            assert!(results.is_empty());
        });
    }

    // -- Search --

    #[test]
    fn search_finds_matching_document() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            search
                .index_document(&cx, &make_doc("doc1", "rust programming language"))
                .await
                .unwrap();
            search
                .index_document(&cx, &make_doc("doc2", "python programming language"))
                .await
                .unwrap();

            let results = search.search(&cx, "rust", 10).await.unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].doc_id, "doc1");
            assert_eq!(results[0].source, ScoreSource::Lexical);
            assert!(results[0].lexical_score.is_some());
            assert!(results[0].score > 0.0);
        });
    }

    #[test]
    fn search_returns_results_sorted_by_relevance() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            // doc1 mentions "search" more times -> higher BM25.
            search
                .index_document(
                    &cx,
                    &make_doc("doc1", "search search search algorithms for search"),
                )
                .await
                .unwrap();
            search
                .index_document(&cx, &make_doc("doc2", "search algorithms"))
                .await
                .unwrap();

            let results = search.search(&cx, "search", 10).await.unwrap();
            assert_eq!(results.len(), 2);
            // Higher TF should produce higher BM25 score.
            assert!(results[0].score >= results[1].score);
        });
    }

    #[test]
    fn search_empty_query_returns_empty() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            search
                .index_document(&cx, &make_doc("doc1", "hello world"))
                .await
                .unwrap();

            let results = search.search(&cx, "", 10).await.unwrap();
            assert!(results.is_empty());

            let results = search.search(&cx, "   ", 10).await.unwrap();
            assert!(results.is_empty());
        });
    }

    #[test]
    fn search_no_match_returns_empty() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            search
                .index_document(&cx, &make_doc("doc1", "hello world"))
                .await
                .unwrap();

            let results = search.search(&cx, "zzzznonexistent", 10).await.unwrap();
            assert!(results.is_empty());
        });
    }

    #[test]
    fn search_respects_limit() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            for i in 0..10 {
                search
                    .index_document(
                        &cx,
                        &make_doc(&format!("doc{i}"), "common term in all docs"),
                    )
                    .await
                    .unwrap();
            }

            let results = search.search(&cx, "common", 3).await.unwrap();
            assert_eq!(results.len(), 3);
        });
    }

    #[test]
    fn search_preserves_metadata() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            let doc = make_doc_with_metadata("doc1", "searchable content", "category", "test");
            search.index_document(&cx, &doc).await.unwrap();

            let results = search.search(&cx, "searchable", 10).await.unwrap();
            assert_eq!(results.len(), 1);
            let meta = results[0].metadata.as_ref().unwrap();
            assert_eq!(meta["category"], "test");
        });
    }

    #[test]
    fn search_with_title_and_content() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            let doc = make_doc_with_title("doc1", "Important Title", "body text here");
            search.index_document(&cx, &doc).await.unwrap();

            // Should match on title.
            let results = search.search(&cx, "important", 10).await.unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].doc_id, "doc1");

            // Should match on content.
            let results = search.search(&cx, "body", 10).await.unwrap();
            assert_eq!(results.len(), 1);
        });
    }

    // -- Snippets --

    #[test]
    fn search_with_snippets_returns_highlighted_text() {
        let search = Fts5LexicalSearch::with_defaults();

        {
            let mut table = search.table.lock().unwrap();
            let mut rowid_map = search.rowid_map.lock().unwrap();

            let doc = make_doc("doc1", "The quick brown fox jumps over the lazy dog");
            let rowid = rowid_map.get_or_assign(&doc.id);
            let columns = Fts5LexicalSearch::doc_to_columns(&doc);
            table.insert_document(rowid, &columns);
            search.doc_count.fetch_add(1, Ordering::Relaxed);
        }

        let hits = search.search_with_snippets("fox", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "doc1");
        assert!(hits[0].snippet.is_some());
        let snippet = hits[0].snippet.as_ref().unwrap();
        assert!(
            snippet.contains("<b>fox</b>"),
            "snippet should highlight match: {snippet}"
        );
    }

    // -- Delete --

    #[test]
    fn delete_document_removes_it() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            search
                .index_document(&cx, &make_doc("doc1", "findable content"))
                .await
                .unwrap();
            assert_eq!(search.doc_count(), 1);

            let removed = search.delete_document("doc1").unwrap();
            assert!(removed);
            assert_eq!(search.doc_count(), 0);

            let results = search.search(&cx, "findable", 10).await.unwrap();
            assert!(results.is_empty());
        });
    }

    #[test]
    fn delete_nonexistent_returns_false() {
        let search = Fts5LexicalSearch::with_defaults();
        let removed = search.delete_document("nonexistent").unwrap();
        assert!(!removed);
    }

    // -- Clear --

    #[test]
    fn clear_removes_all_documents() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            search
                .index_document(&cx, &make_doc("doc1", "hello"))
                .await
                .unwrap();
            search
                .index_document(&cx, &make_doc("doc2", "world"))
                .await
                .unwrap();
            assert_eq!(search.doc_count(), 2);

            search.clear().unwrap();
            assert_eq!(search.doc_count(), 0);

            let results = search.search(&cx, "hello", 10).await.unwrap();
            assert!(results.is_empty());
        });
    }

    // -- Commit is no-op --

    #[test]
    fn commit_succeeds_without_error() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            search.commit(&cx).await.unwrap();
        });
    }

    // -- Edge cases --

    #[test]
    fn document_with_empty_content() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            let doc = make_doc("doc1", "");
            search.index_document(&cx, &doc).await.unwrap();
            assert_eq!(search.doc_count(), 1);

            let results = search.search(&cx, "anything", 10).await.unwrap();
            assert!(results.is_empty());
        });
    }

    #[test]
    fn document_with_special_characters() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            let doc = make_doc(
                "doc1",
                "error: fn<T>(x: &mut Vec<u8>) -> Result<(), Box<dyn Error>>",
            );
            search.index_document(&cx, &doc).await.unwrap();

            let results = search.search(&cx, "error", 10).await.unwrap();
            assert_eq!(results.len(), 1);
        });
    }

    #[test]
    fn batch_upsert_mixed_new_and_existing() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            search
                .index_document(&cx, &make_doc("doc1", "original"))
                .await
                .unwrap();
            assert_eq!(search.doc_count(), 1);

            let batch = vec![
                make_doc("doc1", "updated"),   // existing
                make_doc("doc2", "brand new"), // new
            ];
            search.index_documents(&cx, &batch).await.unwrap();
            assert_eq!(search.doc_count(), 2);
        });
    }

    // -- Trait object safety --

    #[test]
    fn fts5_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Fts5LexicalSearch>();
    }

    // -- Config serialization --

    #[test]
    fn config_serde_roundtrip() {
        let config = Fts5AdapterConfig {
            content_mode: Fts5ContentMode::Contentless,
            tokenizer: Fts5TokenizerChoice::Porter,
            title_boost: 3.0,
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: Fts5AdapterConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.content_mode, Fts5ContentMode::Contentless);
        assert_eq!(deserialized.tokenizer, Fts5TokenizerChoice::Porter);
        assert!((deserialized.title_boost - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn content_mode_default_is_stored() {
        assert_eq!(Fts5ContentMode::default(), Fts5ContentMode::Stored);
    }

    #[test]
    fn tokenizer_default_is_unicode61() {
        assert_eq!(
            Fts5TokenizerChoice::default(),
            Fts5TokenizerChoice::Unicode61
        );
    }

    // -- Query truncation --

    #[test]
    fn long_query_is_truncated() {
        let long_query = "a".repeat(MAX_QUERY_LENGTH + 100);
        let truncated = Fts5LexicalSearch::truncate_query(&long_query);
        assert!(truncated.len() <= MAX_QUERY_LENGTH);
    }

    #[test]
    fn normal_query_is_not_truncated() {
        let query = "normal search query";
        let result = Fts5LexicalSearch::truncate_query(query);
        assert_eq!(result, query);
    }

    // -- Debug impl --

    #[test]
    fn debug_format_includes_doc_count() {
        let search = Fts5LexicalSearch::with_defaults();
        let debug = format!("{search:?}");
        assert!(debug.contains("Fts5LexicalSearch"));
        assert!(debug.contains("doc_count"));
    }

    // -- Fts5Hit serde --

    #[test]
    fn fts5_hit_serde_roundtrip() {
        let hit = Fts5Hit {
            doc_id: "doc1".to_owned(),
            bm25_score: 1.5,
            rank: 0,
            snippet: Some("hello <b>world</b>".to_owned()),
            metadata: None,
        };
        let json = serde_json::to_string(&hit).unwrap();
        let deserialized: Fts5Hit = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.doc_id, "doc1");
        assert!((deserialized.bm25_score - 1.5).abs() < f32::EPSILON);
    }
}
