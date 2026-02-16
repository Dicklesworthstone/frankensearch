//! Tantivy BM25 full-text search integration for frankensearch.
//!
//! Provides the [`TantivyIndex`] implementation of the [`LexicalSearch`] trait,
//! including schema creation, document indexing, BM25 query parsing,
//! and search result ranking.
//!
//! # Schema
//!
//! | Field | Tantivy Options | Source |
//! |-------|-----------------|--------|
//! | `id` | `STRING \| STORED` | `IndexableDocument::id` |
//! | `content` | `TEXT \| STORED` | `IndexableDocument::content` |
//! | `title` | `TEXT \| STORED` | `IndexableDocument::title` (empty if `None`) |
//! | `metadata_json` | `STORED` | Serialized `IndexableDocument::metadata` |
//!
//! The `content` and `title` fields are searched with BM25 scoring.
//! Title matches receive a 2× boost via `QueryParser::set_field_boost`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use asupersync::Cx;
use asupersync::sync::Mutex;
use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::traits::{LexicalSearch, SearchFuture};
use frankensearch_core::types::{IndexableDocument, ScoreSource, ScoredResult};
use serde::{Deserialize, Serialize};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, STORED, STRING, Schema, TextFieldIndexing, TextOptions, Value};
use tantivy::tokenizer::{LowerCaser, SimpleTokenizer, TextAnalyzer};
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument, Term};
use tracing::{debug, instrument, warn};

// ─── Constants ──────────────────────────────────────────────────────────────

/// Name for the custom tokenizer registered with the Tantivy index.
const TOKENIZER_NAME: &str = "frankensearch_default";

/// Default heap size for the Tantivy `IndexWriter` (50 MB).
const WRITER_HEAP_BYTES: usize = 50_000_000;

/// BM25 boost applied to title field matches.
const TITLE_BOOST: f32 = 2.0;

/// Maximum query length in characters. Queries exceeding this are truncated
/// with a warning log. Prevents pathological parsing of enormous inputs.
const MAX_QUERY_LENGTH: usize = 10_000;

/// Default maximum snippet length in characters.
const DEFAULT_SNIPPET_MAX_CHARS: usize = 200;

// ─── Query Explanation ──────────────────────────────────────────────────────

/// Classification of a parsed query for debugging and diagnostics.
///
/// Returned by [`TantivyIndex::search_with_snippets`] to help callers
/// understand how a query was interpreted by Tantivy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryExplanation {
    /// Query was empty or whitespace-only; no search performed.
    Empty,
    /// Single-term query (e.g., `"authentication"`).
    Simple,
    /// Quoted phrase query (e.g., `"error handling"`).
    Phrase,
    /// Multi-term query interpreted as boolean OR (default Tantivy behavior).
    Boolean,
}

impl std::fmt::Display for QueryExplanation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "empty"),
            Self::Simple => write!(f, "simple"),
            Self::Phrase => write!(f, "phrase"),
            Self::Boolean => write!(f, "boolean"),
        }
    }
}

/// Classify a raw query string into a [`QueryExplanation`].
fn classify_query(query: &str) -> QueryExplanation {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return QueryExplanation::Empty;
    }
    // Check for quoted phrase: starts and ends with matching quotes.
    if (trimmed.starts_with('"') && trimmed.ends_with('"'))
        || (trimmed.starts_with('\'') && trimmed.ends_with('\''))
    {
        return QueryExplanation::Phrase;
    }
    // Count whitespace-separated tokens.
    let token_count = trimmed.split_whitespace().count();
    if token_count <= 1 {
        QueryExplanation::Simple
    } else {
        QueryExplanation::Boolean
    }
}

// ─── Snippet Configuration ──────────────────────────────────────────────────

/// Configuration for snippet generation in [`TantivyIndex::search_with_snippets`].
#[derive(Debug, Clone)]
pub struct SnippetConfig {
    /// Maximum number of characters for the snippet fragment.
    pub max_chars: usize,
    /// HTML tag prefix for highlighted terms (e.g., `"<b>"`).
    pub highlight_prefix: String,
    /// HTML tag postfix for highlighted terms (e.g., `"</b>"`).
    pub highlight_postfix: String,
}

impl Default for SnippetConfig {
    fn default() -> Self {
        Self {
            max_chars: DEFAULT_SNIPPET_MAX_CHARS,
            highlight_prefix: "<b>".to_owned(),
            highlight_postfix: "</b>".to_owned(),
        }
    }
}

// ─── LexicalHit ─────────────────────────────────────────────────────────────

/// An enriched search result from [`TantivyIndex::search_with_snippets`].
///
/// Contains everything in [`ScoredResult`] plus a snippet and query explanation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexicalHit {
    /// Unique document identifier.
    pub doc_id: String,
    /// BM25 relevance score.
    pub bm25_score: f32,
    /// 0-based rank in the result set.
    pub rank: usize,
    /// Highlighted snippet from the content field, if available.
    pub snippet: Option<String>,
    /// How the query was classified.
    pub query_type: QueryExplanation,
    /// Arbitrary document metadata.
    pub metadata: Option<serde_json::Value>,
}

// ─── Schema fields ──────────────────────────────────────────────────────────

/// Named fields from the Tantivy schema for type-safe access.
#[derive(Debug, Clone, Copy)]
struct SchemaFields {
    id: Field,
    content: Field,
    title: Field,
    metadata_json: Field,
}

/// Build the frankensearch Tantivy schema.
fn build_schema() -> (Schema, SchemaFields) {
    let mut builder = Schema::builder();

    // ID: exact-match only, stored for retrieval.
    let id = builder.add_text_field("id", STRING | STORED);

    // Content: full-text indexed with our custom tokenizer, stored for snippet use.
    let content_options = TextOptions::default()
        .set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer(TOKENIZER_NAME)
                .set_index_option(tantivy::schema::IndexRecordOption::WithFreqsAndPositions),
        )
        .set_stored();
    let content = builder.add_text_field("content", content_options);

    // Title: full-text indexed with our custom tokenizer, stored.
    let title_options = TextOptions::default()
        .set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer(TOKENIZER_NAME)
                .set_index_option(tantivy::schema::IndexRecordOption::WithFreqsAndPositions),
        )
        .set_stored();
    let title = builder.add_text_field("title", title_options);

    // Metadata: stored as JSON string, not indexed.
    let metadata_json = builder.add_text_field("metadata_json", STORED);

    let schema = builder.build();
    let fields = SchemaFields {
        id,
        content,
        title,
        metadata_json,
    };
    (schema, fields)
}

/// Build and register the custom tokenizer.
///
/// Pipeline: `SimpleTokenizer` (Unicode word boundary splitting) → `LowerCaser`.
/// This handles hyphenated identifiers like `POL-358` as a single token.
fn build_tokenizer() -> TextAnalyzer {
    TextAnalyzer::builder(SimpleTokenizer::default())
        .filter(LowerCaser)
        .build()
}

// ─── TantivyIndex ───────────────────────────────────────────────────────────

/// A Tantivy-backed full-text search index implementing [`LexicalSearch`].
///
/// Thread-safe for concurrent reads. Writes are serialized internally via
/// the Tantivy `IndexWriter` (which requires `&mut self` for `add_document`
/// but is wrapped here for the trait interface).
pub struct TantivyIndex {
    index: Index,
    fields: SchemaFields,
    reader: IndexReader,
    writer: Mutex<IndexWriter>,
    doc_count: AtomicUsize,
    path: Option<PathBuf>,
}

impl std::fmt::Debug for TantivyIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TantivyIndex")
            .field("doc_count", &self.doc_count.load(Ordering::Relaxed))
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl TantivyIndex {
    /// Create a new Tantivy index at the given directory path.
    ///
    /// If the directory does not exist, it will be created.
    /// If an index already exists at this path, it will be opened.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::SubsystemError` if the Tantivy index cannot be
    /// created or opened.
    pub fn create(path: &Path) -> SearchResult<Self> {
        let (schema, fields) = build_schema();

        std::fs::create_dir_all(path).map_err(|e| SearchError::SubsystemError {
            subsystem: "tantivy",
            source: Box::new(e),
        })?;

        let index = Index::create_in_dir(path, schema.clone())
            .or_else(|_| {
                // If creation fails (already exists), try opening instead.
                Index::open_in_dir(path)
            })
            .map_err(|e| SearchError::SubsystemError {
                subsystem: "tantivy",
                source: Box::new(e),
            })?;

        Self::from_index(index, schema, fields, Some(path.to_path_buf()))
    }

    /// Open an existing Tantivy index at the given directory path.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::IndexNotFound` if the path does not exist.
    /// Returns `SearchError::SubsystemError` if the index cannot be opened.
    pub fn open(path: &Path) -> SearchResult<Self> {
        if !path.exists() {
            return Err(SearchError::IndexNotFound {
                path: path.to_path_buf(),
            });
        }

        let (schema, fields) = build_schema();
        let index = Index::open_in_dir(path).map_err(|e| SearchError::SubsystemError {
            subsystem: "tantivy",
            source: Box::new(e),
        })?;

        Self::from_index(index, schema, fields, Some(path.to_path_buf()))
    }

    /// Create an in-memory Tantivy index (useful for testing).
    ///
    /// # Errors
    ///
    /// Returns `SearchError::SubsystemError` if the index cannot be created.
    pub fn in_memory() -> SearchResult<Self> {
        let (schema, fields) = build_schema();
        let index = Index::create_in_ram(schema.clone());
        Self::from_index(index, schema, fields, None)
    }

    /// Internal constructor shared by `create`, `open`, and `in_memory`.
    fn from_index(
        index: Index,
        _schema: Schema,
        fields: SchemaFields,
        path: Option<PathBuf>,
    ) -> SearchResult<Self> {
        // Register our custom tokenizer.
        let tokenizer_manager = index.tokenizers().clone();
        tokenizer_manager.register(TOKENIZER_NAME, build_tokenizer());

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .map_err(|e| SearchError::SubsystemError {
                subsystem: "tantivy",
                source: Box::new(e),
            })?;

        let writer = index
            .writer(WRITER_HEAP_BYTES)
            .map_err(|e| SearchError::SubsystemError {
                subsystem: "tantivy",
                source: Box::new(e),
            })?;

        // Count existing documents.
        let searcher = reader.searcher();
        let doc_count = usize::try_from(searcher.num_docs()).unwrap_or(usize::MAX);

        Ok(Self {
            index,
            fields,
            reader,
            writer: Mutex::new(writer),
            doc_count: AtomicUsize::new(doc_count),
            path,
        })
    }

    /// Convert an `IndexableDocument` to a Tantivy document.
    fn to_tantivy_doc(&self, doc: &IndexableDocument) -> TantivyDocument {
        let mut tantivy_doc = TantivyDocument::new();
        tantivy_doc.add_text(self.fields.id, &doc.id);
        tantivy_doc.add_text(self.fields.content, &doc.content);
        tantivy_doc.add_text(self.fields.title, doc.title.as_deref().unwrap_or(""));

        // Serialize metadata as JSON string.
        if !doc.metadata.is_empty()
            && let Ok(json) = serde_json::to_string(&doc.metadata)
        {
            tantivy_doc.add_text(self.fields.metadata_json, &json);
        }

        tantivy_doc
    }

    /// Build a `QueryParser` for BM25 search with title boost.
    fn query_parser(&self) -> QueryParser {
        let mut parser =
            QueryParser::for_index(&self.index, vec![self.fields.content, self.fields.title]);
        parser.set_field_boost(self.fields.title, TITLE_BOOST);
        parser
    }

    /// Parse a query using lenient mode (never fails, returns best-effort query).
    ///
    /// Unknown field prefixes, unbalanced quotes, and other syntax issues are
    /// silently ignored rather than producing errors. This makes user-facing
    /// search robust against arbitrary input.
    fn parse_query_lenient(&self, query: &str) -> Box<dyn tantivy::query::Query> {
        let parser = self.query_parser();
        let (parsed, errors) = parser.parse_query_lenient(query);
        if let Some(first_error) = errors.first() {
            debug!(
                error_count = errors.len(),
                first_error = %first_error,
                "lenient query parse produced warnings"
            );
        }
        parsed
    }

    /// Delete a document by its ID.
    ///
    /// The deletion is staged; call [`commit`](LexicalSearch::commit) to persist.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::SubsystemError` if the writer lock is poisoned
    /// or cancelled.
    pub async fn delete_document(&self, cx: &Cx, doc_id: &str) -> SearchResult<()> {
        let term = Term::from_field_text(self.fields.id, doc_id);
        self.writer
            .lock(cx)
            .await
            .map_err(|e| match e {
                asupersync::sync::LockError::Poisoned => SearchError::SubsystemError {
                    subsystem: "tantivy",
                    source: Box::new(std::io::Error::other("writer mutex poisoned")),
                },
                asupersync::sync::LockError::Cancelled => SearchError::Cancelled {
                    phase: "tantivy.delete".into(),
                    reason: "writer lock cancelled".into(),
                },
            })?
            .delete_term(term);
        Ok(())
    }

    /// Returns the directory path for this index, if on-disk.
    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Truncate an overlong query and log a warning.
    fn truncate_query(query: &str) -> &str {
        if query.len() <= MAX_QUERY_LENGTH {
            return query;
        }
        warn!(
            original_len = query.len(),
            max = MAX_QUERY_LENGTH,
            "query truncated to MAX_QUERY_LENGTH"
        );
        // Truncate at a char boundary.
        let mut end = MAX_QUERY_LENGTH;
        while end > 0 && !query.is_char_boundary(end) {
            end -= 1;
        }
        &query[..end]
    }

    /// Search with snippet generation and query explanation.
    ///
    /// Returns [`LexicalHit`] results enriched with highlighted snippets
    /// from the content field and a [`QueryExplanation`] indicating how
    /// the query was parsed.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if the query cannot be parsed or search fails.
    #[instrument(skip_all, fields(query = %query, limit = limit))]
    pub fn search_with_snippets(
        &self,
        _cx: &Cx,
        query: &str,
        limit: usize,
        snippet_config: &SnippetConfig,
    ) -> SearchResult<Vec<LexicalHit>> {
        let query = Self::truncate_query(query);
        let explanation = classify_query(query);

        if explanation == QueryExplanation::Empty {
            return Ok(Vec::new());
        }

        let parsed = self.parse_query_lenient(query);

        let searcher = self.reader.searcher();
        let top_docs = searcher
            .search(&*parsed, &TopDocs::with_limit(limit))
            .map_err(|e| SearchError::SubsystemError {
                subsystem: "tantivy",
                source: Box::new(e),
            })?;

        // Build snippet generator for the content field.
        let snippet_gen = match tantivy::snippet::SnippetGenerator::create(
            &searcher,
            &*parsed,
            self.fields.content,
        ) {
            Ok(mut sg) => {
                sg.set_max_num_chars(snippet_config.max_chars);
                Some(sg)
            }
            Err(e) => {
                debug!(error = %e, "failed to create snippet generator, snippets will be absent");
                None
            }
        };

        debug!(
            hits = top_docs.len(),
            query_type = %explanation,
            "tantivy search_with_snippets completed"
        );

        let mut results = Vec::with_capacity(top_docs.len());
        for (rank, (bm25_score, doc_address)) in top_docs.into_iter().enumerate() {
            let doc: TantivyDocument =
                searcher
                    .doc(doc_address)
                    .map_err(|e| SearchError::SubsystemError {
                        subsystem: "tantivy",
                        source: Box::new(e),
                    })?;

            let doc_id = doc
                .get_first(self.fields.id)
                .and_then(|v| v.as_str())
                .unwrap_or_else(|| {
                    debug!("tantivy document missing id field, using empty doc_id");
                    ""
                })
                .to_owned();

            let metadata = doc
                .get_first(self.fields.metadata_json)
                .and_then(|v| v.as_str())
                .and_then(|s| match serde_json::from_str(s) {
                    Ok(val) => Some(val),
                    Err(e) => {
                        debug!(doc_id = %doc_id, error = %e, "failed to deserialize metadata JSON");
                        None
                    }
                });

            // Generate snippet from the document.
            let snippet = snippet_gen.as_ref().and_then(|sg| {
                let mut s = sg.snippet_from_doc(&doc);
                s.set_snippet_prefix_postfix(
                    &snippet_config.highlight_prefix,
                    &snippet_config.highlight_postfix,
                );
                let html = s.to_html();
                if html.is_empty() { None } else { Some(html) }
            });

            results.push(LexicalHit {
                doc_id,
                bm25_score,
                rank,
                snippet,
                query_type: explanation,
                metadata,
            });
        }

        Ok(results)
    }
}

// ─── LexicalSearch implementation ───────────────────────────────────────────

impl LexicalSearch for TantivyIndex {
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

            let parsed = self.parse_query_lenient(query);

            let searcher = self.reader.searcher();
            let top_docs = searcher
                .search(&*parsed, &TopDocs::with_limit(limit))
                .map_err(|e| SearchError::SubsystemError {
                    subsystem: "tantivy",
                    source: Box::new(e),
                })?;

            debug!(hits = top_docs.len(), "tantivy BM25 search completed");

            let mut results = Vec::with_capacity(top_docs.len());
            for (bm25_score, doc_address) in top_docs {
                let doc: TantivyDocument =
                    searcher
                        .doc(doc_address)
                        .map_err(|e| SearchError::SubsystemError {
                            subsystem: "tantivy",
                            source: Box::new(e),
                        })?;

                let doc_id = doc
                    .get_first(self.fields.id)
                    .and_then(|v| v.as_str())
                    .unwrap_or_else(|| {
                        debug!("tantivy document missing id field, using empty doc_id");
                        ""
                    })
                    .to_owned();

                let metadata = doc
                    .get_first(self.fields.metadata_json)
                    .and_then(|v| v.as_str())
                    .and_then(|s| match serde_json::from_str(s) {
                        Ok(val) => Some(val),
                        Err(e) => {
                            debug!(doc_id = %doc_id, error = %e, "failed to deserialize metadata JSON");
                            None
                        }
                    });

                results.push(ScoredResult {
                    doc_id,
                    score: bm25_score,
                    source: ScoreSource::Lexical,
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
        cx: &'a Cx,
        doc: &'a IndexableDocument,
    ) -> SearchFuture<'a, ()> {
        Box::pin(async move {
            let tantivy_doc = self.to_tantivy_doc(doc);

            {
                let writer = self.writer.lock(cx).await.map_err(|e| match e {
                    asupersync::sync::LockError::Poisoned => SearchError::SubsystemError {
                        subsystem: "tantivy",
                        source: Box::new(std::io::Error::other("writer mutex poisoned")),
                    },
                    asupersync::sync::LockError::Cancelled => SearchError::Cancelled {
                        phase: "tantivy.index".into(),
                        reason: "writer lock cancelled".into(),
                    },
                })?;

                // Delete any existing document with same ID (upsert semantics).
                let term = Term::from_field_text(self.fields.id, &doc.id);
                writer.delete_term(term);
                writer
                    .add_document(tantivy_doc)
                    .map_err(|e| SearchError::SubsystemError {
                        subsystem: "tantivy",
                        source: Box::new(e),
                    })?;
            }

            self.doc_count.fetch_add(1, Ordering::Relaxed);
            Ok(())
        })
    }

    fn index_documents<'a>(
        &'a self,
        cx: &'a Cx,
        docs: &'a [IndexableDocument],
    ) -> SearchFuture<'a, ()> {
        Box::pin(async move {
            {
                let writer = self.writer.lock(cx).await.map_err(|e| match e {
                    asupersync::sync::LockError::Poisoned => SearchError::SubsystemError {
                        subsystem: "tantivy",
                        source: Box::new(std::io::Error::other("writer mutex poisoned")),
                    },
                    asupersync::sync::LockError::Cancelled => SearchError::Cancelled {
                        phase: "tantivy.batch_index".into(),
                        reason: "writer lock cancelled".into(),
                    },
                })?;

                for doc in docs {
                    let tantivy_doc = self.to_tantivy_doc(doc);
                    // Upsert: delete existing then add.
                    let term = Term::from_field_text(self.fields.id, &doc.id);
                    writer.delete_term(term);
                    writer
                        .add_document(tantivy_doc)
                        .map_err(|e| SearchError::SubsystemError {
                            subsystem: "tantivy",
                            source: Box::new(e),
                        })?;
                }
            }

            self.doc_count.fetch_add(docs.len(), Ordering::Relaxed);

            debug!(count = docs.len(), "batch indexed documents");
            Ok(())
        })
    }

    fn commit<'a>(&'a self, cx: &'a Cx) -> SearchFuture<'a, ()> {
        Box::pin(async move {
            {
                let mut writer = self.writer.lock(cx).await.map_err(|e| match e {
                    asupersync::sync::LockError::Poisoned => SearchError::SubsystemError {
                        subsystem: "tantivy",
                        source: Box::new(std::io::Error::other("writer mutex poisoned")),
                    },
                    asupersync::sync::LockError::Cancelled => SearchError::Cancelled {
                        phase: "tantivy.commit".into(),
                        reason: "writer lock cancelled".into(),
                    },
                })?;

                writer.commit().map_err(|e| SearchError::SubsystemError {
                    subsystem: "tantivy",
                    source: Box::new(e),
                })?;
            }

            // Reload the reader to pick up committed changes.
            self.reader
                .reload()
                .map_err(|e| SearchError::SubsystemError {
                    subsystem: "tantivy",
                    source: Box::new(e),
                })?;

            // Re-count after commit for accuracy.
            let searcher = self.reader.searcher();
            let actual = usize::try_from(searcher.num_docs()).unwrap_or(usize::MAX);
            self.doc_count.store(actual, Ordering::Relaxed);

            debug!(doc_count = actual, "tantivy commit completed");
            Ok(())
        })
    }

    fn doc_count(&self) -> usize {
        self.doc_count.load(Ordering::Relaxed)
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use frankensearch_core::types::IndexableDocument;

    /// Helper: run async test code with a `Cx`.
    fn run_with_cx<F, Fut>(f: F)
    where
        F: FnOnce(Cx) -> Fut,
        Fut: Future<Output = ()>,
    {
        asupersync::test_utils::run_test_with_cx(f);
    }

    fn sample_docs() -> Vec<IndexableDocument> {
        vec![
            IndexableDocument::new("doc-1", "Rust is a systems programming language")
                .with_title("Rust Overview")
                .with_metadata("lang", "en"),
            IndexableDocument::new(
                "doc-2",
                "Python is great for data science and machine learning",
            )
            .with_title("Python for ML"),
            IndexableDocument::new("doc-3", "The Rust borrow checker prevents data races")
                .with_title("Rust Safety"),
            IndexableDocument::new(
                "doc-4",
                "Distributed consensus algorithms like Raft and Paxos",
            )
            .with_title("Consensus Algorithms"),
            IndexableDocument::new(
                "doc-5",
                "Machine learning models for natural language processing",
            )
            .with_title("NLP Models"),
        ]
    }

    // ─── Schema tests ───────────────────────────────────────────────────

    #[test]
    fn schema_has_required_fields() {
        let (schema, fields) = build_schema();
        assert!(schema.get_field_entry(fields.id).is_stored());
        assert!(schema.get_field_entry(fields.content).is_stored());
        assert!(schema.get_field_entry(fields.title).is_stored());
        assert!(schema.get_field_entry(fields.metadata_json).is_stored());
    }

    // ─── Index lifecycle tests ──────────────────────────────────────────

    #[test]
    fn create_in_memory() {
        let idx = TantivyIndex::in_memory().expect("create");
        assert_eq!(idx.doc_count(), 0);
    }

    #[test]
    fn create_on_disk() {
        let dir = tempfile::tempdir().expect("tempdir");
        let idx = TantivyIndex::create(dir.path()).expect("create");
        assert_eq!(idx.doc_count(), 0);
        assert_eq!(idx.path(), Some(dir.path()));
    }

    #[test]
    fn open_nonexistent_returns_error() {
        let result = TantivyIndex::open(Path::new("/nonexistent/tantivy_index_xyz"));
        assert!(result.is_err());
    }

    // ─── Indexing tests ─────────────────────────────────────────────────

    #[test]
    fn index_single_document() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            let doc = IndexableDocument::new("doc-1", "Hello world");
            idx.index_document(&cx, &doc).await.expect("index");
            idx.commit(&cx).await.expect("commit");
            assert_eq!(idx.doc_count(), 1);
        });
    }

    #[test]
    fn index_batch_documents() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            let docs = sample_docs();
            idx.index_documents(&cx, &docs).await.expect("batch index");
            idx.commit(&cx).await.expect("commit");
            assert_eq!(idx.doc_count(), 5);
        });
    }

    #[test]
    fn upsert_replaces_existing_document() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            let doc_v1 = IndexableDocument::new("doc-1", "Version one content");
            idx.index_document(&cx, &doc_v1).await.expect("index v1");
            idx.commit(&cx).await.expect("commit v1");

            let doc_v2 = IndexableDocument::new("doc-1", "Version two content updated");
            idx.index_document(&cx, &doc_v2).await.expect("index v2");
            idx.commit(&cx).await.expect("commit v2");

            let results = idx.search(&cx, "updated", 10).await.expect("search");
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].doc_id, "doc-1");
        });
    }

    // ─── Search tests ───────────────────────────────────────────────────

    #[test]
    fn search_empty_query_returns_empty() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            let docs = sample_docs();
            idx.index_documents(&cx, &docs).await.expect("index");
            idx.commit(&cx).await.expect("commit");

            let results = idx.search(&cx, "", 10).await.expect("search");
            assert!(results.is_empty());

            let results = idx.search(&cx, "   ", 10).await.expect("search whitespace");
            assert!(results.is_empty());
        });
    }

    #[test]
    fn search_returns_relevant_results() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            let docs = sample_docs();
            idx.index_documents(&cx, &docs).await.expect("index");
            idx.commit(&cx).await.expect("commit");

            let results = idx.search(&cx, "Rust", 10).await.expect("search");
            assert!(!results.is_empty(), "should find documents mentioning Rust");
            let ids: Vec<&str> = results.iter().map(|r| r.doc_id.as_str()).collect();
            assert!(ids.contains(&"doc-1"), "should find doc-1");
            assert!(ids.contains(&"doc-3"), "should find doc-3");
        });
    }

    #[test]
    fn search_respects_limit() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            let docs = sample_docs();
            idx.index_documents(&cx, &docs).await.expect("index");
            idx.commit(&cx).await.expect("commit");

            let results = idx
                .search(&cx, "machine learning", 1)
                .await
                .expect("search");
            assert_eq!(results.len(), 1);
        });
    }

    #[test]
    fn search_results_have_lexical_source() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            let docs = sample_docs();
            idx.index_documents(&cx, &docs).await.expect("index");
            idx.commit(&cx).await.expect("commit");

            let results = idx.search(&cx, "Rust", 5).await.expect("search");
            for r in &results {
                assert_eq!(r.source, ScoreSource::Lexical);
                assert!(r.lexical_score.is_some());
                assert!(r.lexical_score.unwrap() > 0.0);
                assert!(r.fast_score.is_none());
                assert!(r.quality_score.is_none());
                assert!(r.rerank_score.is_none());
                assert!(r.explanation.is_none());
            }
        });
    }

    #[test]
    fn search_scores_are_descending() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            let docs = sample_docs();
            idx.index_documents(&cx, &docs).await.expect("index");
            idx.commit(&cx).await.expect("commit");

            let results = idx.search(&cx, "language", 10).await.expect("search");
            if results.len() > 1 {
                for pair in results.windows(2) {
                    assert!(
                        pair[0].score >= pair[1].score,
                        "scores should be descending: {} >= {}",
                        pair[0].score,
                        pair[1].score
                    );
                }
            }
        });
    }

    #[test]
    fn title_boost_affects_ranking() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            let doc_a =
                IndexableDocument::new("doc-a", "consensus algorithm for distributed systems");
            let doc_b = IndexableDocument::new("doc-b", "some distributed system design")
                .with_title("Consensus Protocol");

            idx.index_document(&cx, &doc_a).await.expect("index a");
            idx.index_document(&cx, &doc_b).await.expect("index b");
            idx.commit(&cx).await.expect("commit");

            let results = idx.search(&cx, "consensus", 2).await.expect("search");
            assert_eq!(results.len(), 2);
            assert_eq!(
                results[0].doc_id, "doc-b",
                "title-boosted document should rank first"
            );
        });
    }

    #[test]
    fn metadata_preserved_in_results() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            let doc = IndexableDocument::new("doc-1", "test content")
                .with_metadata("source", "unit_test")
                .with_metadata("lang", "en");

            idx.index_document(&cx, &doc).await.expect("index");
            idx.commit(&cx).await.expect("commit");

            let results = idx.search(&cx, "test", 1).await.expect("search");
            assert_eq!(results.len(), 1);

            let meta = results[0].metadata.as_ref().expect("metadata present");
            assert_eq!(meta["source"], "unit_test");
            assert_eq!(meta["lang"], "en");
        });
    }

    #[test]
    fn no_results_for_unmatched_query() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            let docs = sample_docs();
            idx.index_documents(&cx, &docs).await.expect("index");
            idx.commit(&cx).await.expect("commit");

            let results = idx.search(&cx, "xylophone", 10).await.expect("search");
            assert!(results.is_empty(), "no documents mention xylophone");
        });
    }

    // ─── Delete tests ───────────────────────────────────────────────────

    #[test]
    fn delete_document_removes_from_index() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            let docs = sample_docs();
            idx.index_documents(&cx, &docs).await.expect("index");
            idx.commit(&cx).await.expect("commit");
            assert_eq!(idx.doc_count(), 5);

            idx.delete_document(&cx, "doc-1").await.expect("delete");
            idx.commit(&cx).await.expect("commit after delete");

            let results = idx.search(&cx, "Rust systems", 10).await.expect("search");
            assert!(
                !results.iter().any(|r| r.doc_id == "doc-1"),
                "deleted document should not appear"
            );
        });
    }

    // ─── Edge case tests ────────────────────────────────────────────────

    #[test]
    fn search_with_special_characters() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            let doc = IndexableDocument::new("doc-1", "Error code ERR-404: page not found");
            idx.index_document(&cx, &doc).await.expect("index");
            idx.commit(&cx).await.expect("commit");

            let results = idx.search(&cx, "ERR-404", 10).await.expect("search");
            assert!(!results.is_empty(), "should find hyphenated term");
        });
    }

    #[test]
    fn case_insensitive_search() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            let doc = IndexableDocument::new("doc-1", "Rust Programming Language");
            idx.index_document(&cx, &doc).await.expect("index");
            idx.commit(&cx).await.expect("commit");

            let results = idx.search(&cx, "rust", 10).await.expect("search lowercase");
            assert!(!results.is_empty());

            let results = idx.search(&cx, "RUST", 10).await.expect("search uppercase");
            assert!(!results.is_empty());
        });
    }

    #[test]
    fn empty_metadata_not_stored() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            let doc = IndexableDocument::new("doc-1", "no metadata here");
            idx.index_document(&cx, &doc).await.expect("index");
            idx.commit(&cx).await.expect("commit");

            let results = idx.search(&cx, "metadata", 1).await.expect("search");
            assert_eq!(results.len(), 1);
            assert!(results[0].metadata.is_none());
        });
    }

    #[test]
    fn doc_count_accurate_after_operations() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            assert_eq!(idx.doc_count(), 0);

            let doc = IndexableDocument::new("doc-1", "first");
            idx.index_document(&cx, &doc).await.expect("index");
            idx.commit(&cx).await.expect("commit");
            assert_eq!(idx.doc_count(), 1);

            let doc = IndexableDocument::new("doc-2", "second");
            idx.index_document(&cx, &doc).await.expect("index");
            idx.commit(&cx).await.expect("commit");
            assert_eq!(idx.doc_count(), 2);

            idx.delete_document(&cx, "doc-1").await.expect("delete");
            idx.commit(&cx).await.expect("commit delete");
            assert_eq!(idx.doc_count(), 1);
        });
    }

    // ─── Trait object safety ────────────────────────────────────────────

    #[test]
    fn tantivy_index_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TantivyIndex>();
    }

    // ─── On-disk persistence tests ──────────────────────────────────────

    #[test]
    fn reopen_preserves_documents() {
        let dir = tempfile::tempdir().expect("tempdir");

        // Phase 1: Create and populate.
        {
            let idx = TantivyIndex::create(dir.path()).expect("create");
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let doc = IndexableDocument::new("doc-1", "persistent content");
                idx.index_document(&cx, &doc).await.expect("index");
                idx.commit(&cx).await.expect("commit");
            });
        }

        // Phase 2: Reopen and verify.
        {
            let idx = TantivyIndex::open(dir.path()).expect("open");
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let results = idx.search(&cx, "persistent", 10).await.expect("search");
                assert_eq!(results.len(), 1);
                assert_eq!(results[0].doc_id, "doc-1");
            });
        }
    }

    // ─── Query explanation tests (bd-3un.18) ─────────────────────────────

    #[test]
    fn classify_empty_query() {
        assert_eq!(classify_query(""), QueryExplanation::Empty);
        assert_eq!(classify_query("   "), QueryExplanation::Empty);
    }

    #[test]
    fn classify_simple_query() {
        assert_eq!(classify_query("rust"), QueryExplanation::Simple);
        assert_eq!(
            classify_query("  authentication  "),
            QueryExplanation::Simple
        );
    }

    #[test]
    fn classify_phrase_query() {
        assert_eq!(
            classify_query("\"error handling\""),
            QueryExplanation::Phrase
        );
        assert_eq!(classify_query("'single quotes'"), QueryExplanation::Phrase);
    }

    #[test]
    fn classify_boolean_query() {
        assert_eq!(classify_query("rust async"), QueryExplanation::Boolean);
        assert_eq!(
            classify_query("distributed consensus algorithm"),
            QueryExplanation::Boolean
        );
    }

    #[test]
    fn query_explanation_display() {
        assert_eq!(QueryExplanation::Empty.to_string(), "empty");
        assert_eq!(QueryExplanation::Simple.to_string(), "simple");
        assert_eq!(QueryExplanation::Phrase.to_string(), "phrase");
        assert_eq!(QueryExplanation::Boolean.to_string(), "boolean");
    }

    #[test]
    fn query_explanation_serde_roundtrip() {
        let json = serde_json::to_string(&QueryExplanation::Phrase).unwrap();
        let decoded: QueryExplanation = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, QueryExplanation::Phrase);
    }

    // ─── Snippet config tests ────────────────────────────────────────────

    #[test]
    fn snippet_config_default() {
        let config = SnippetConfig::default();
        assert_eq!(config.max_chars, DEFAULT_SNIPPET_MAX_CHARS);
        assert_eq!(config.highlight_prefix, "<b>");
        assert_eq!(config.highlight_postfix, "</b>");
    }

    // ─── search_with_snippets tests (bd-3un.18) ─────────────────────────

    #[test]
    fn search_with_snippets_returns_results() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            let docs = sample_docs();
            idx.index_documents(&cx, &docs).await.expect("index");
            idx.commit(&cx).await.expect("commit");

            let config = SnippetConfig::default();
            let results = idx
                .search_with_snippets(&cx, "Rust", 10, &config)
                .expect("search");

            assert!(!results.is_empty());
            assert_eq!(results[0].rank, 0);
            assert_eq!(results[0].query_type, QueryExplanation::Simple);
            assert!(results[0].bm25_score > 0.0);
        });
    }

    #[test]
    fn search_with_snippets_empty_query() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            let docs = sample_docs();
            idx.index_documents(&cx, &docs).await.expect("index");
            idx.commit(&cx).await.expect("commit");

            let config = SnippetConfig::default();
            let results = idx
                .search_with_snippets(&cx, "", 10, &config)
                .expect("search");
            assert!(results.is_empty());
        });
    }

    #[test]
    fn search_with_snippets_has_highlighted_content() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            let doc = IndexableDocument::new(
                "doc-1",
                "The Rust programming language is fast and memory-safe",
            );
            idx.index_document(&cx, &doc).await.expect("index");
            idx.commit(&cx).await.expect("commit");

            let config = SnippetConfig::default();
            let results = idx
                .search_with_snippets(&cx, "Rust", 1, &config)
                .expect("search");

            assert_eq!(results.len(), 1);
            if let Some(snippet) = &results[0].snippet {
                assert!(
                    snippet.contains("<b>"),
                    "snippet should have highlight tags: {snippet}"
                );
            }
        });
    }

    #[test]
    fn search_with_snippets_custom_highlight_tags() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            let doc = IndexableDocument::new("doc-1", "Rust is awesome for systems programming");
            idx.index_document(&cx, &doc).await.expect("index");
            idx.commit(&cx).await.expect("commit");

            let config = SnippetConfig {
                max_chars: 200,
                highlight_prefix: "<em>".to_owned(),
                highlight_postfix: "</em>".to_owned(),
            };
            let results = idx
                .search_with_snippets(&cx, "Rust", 1, &config)
                .expect("search");

            assert_eq!(results.len(), 1);
            if let Some(snippet) = &results[0].snippet {
                assert!(
                    snippet.contains("<em>"),
                    "snippet should use custom highlight: {snippet}"
                );
                assert!(
                    !snippet.contains("<b>"),
                    "should NOT use default highlight: {snippet}"
                );
            }
        });
    }

    #[test]
    fn search_with_snippets_ranks_are_sequential() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            let docs = sample_docs();
            idx.index_documents(&cx, &docs).await.expect("index");
            idx.commit(&cx).await.expect("commit");

            let config = SnippetConfig::default();
            let results = idx
                .search_with_snippets(&cx, "language", 10, &config)
                .expect("search");

            for (i, hit) in results.iter().enumerate() {
                assert_eq!(hit.rank, i, "rank should be sequential");
            }
        });
    }

    #[test]
    fn search_with_snippets_metadata_preserved() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            let doc = IndexableDocument::new("doc-1", "metadata test content")
                .with_metadata("key", "value");
            idx.index_document(&cx, &doc).await.expect("index");
            idx.commit(&cx).await.expect("commit");

            let config = SnippetConfig::default();
            let results = idx
                .search_with_snippets(&cx, "metadata", 1, &config)
                .expect("search");

            assert_eq!(results.len(), 1);
            let meta = results[0].metadata.as_ref().expect("metadata");
            assert_eq!(meta["key"], "value");
        });
    }

    #[test]
    fn search_with_snippets_phrase_query() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            let doc = IndexableDocument::new("doc-1", "error handling in Rust is explicit");
            idx.index_document(&cx, &doc).await.expect("index");
            idx.commit(&cx).await.expect("commit");

            let config = SnippetConfig::default();
            let results = idx
                .search_with_snippets(&cx, "\"error handling\"", 10, &config)
                .expect("search");

            assert_eq!(results.len(), 1);
            assert_eq!(results[0].query_type, QueryExplanation::Phrase);
        });
    }

    #[test]
    fn search_with_snippets_boolean_query() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            let docs = sample_docs();
            idx.index_documents(&cx, &docs).await.expect("index");
            idx.commit(&cx).await.expect("commit");

            let config = SnippetConfig::default();
            let results = idx
                .search_with_snippets(&cx, "machine learning", 10, &config)
                .expect("search");

            assert!(!results.is_empty());
            assert_eq!(results[0].query_type, QueryExplanation::Boolean);
        });
    }

    // ─── Query truncation tests ──────────────────────────────────────────

    #[test]
    fn truncate_query_short_passthrough() {
        let q = "hello world";
        assert_eq!(TantivyIndex::truncate_query(q), q);
    }

    #[test]
    fn truncate_query_at_limit() {
        let q = "a".repeat(MAX_QUERY_LENGTH);
        assert_eq!(TantivyIndex::truncate_query(&q), q.as_str());
    }

    #[test]
    fn truncate_query_over_limit() {
        let q = "a".repeat(MAX_QUERY_LENGTH + 100);
        let truncated = TantivyIndex::truncate_query(&q);
        assert_eq!(truncated.len(), MAX_QUERY_LENGTH);
    }

    #[test]
    fn truncate_query_multibyte_boundary() {
        // Create a string with multibyte chars that would split mid-char at MAX_QUERY_LENGTH.
        let base = "x".repeat(MAX_QUERY_LENGTH - 1);
        let over = format!("{base}\u{00E9}\u{00E9}\u{00E9}"); // é is 2 bytes in UTF-8
        let truncated = TantivyIndex::truncate_query(&over);
        assert!(truncated.is_char_boundary(truncated.len()));
        assert!(truncated.len() <= MAX_QUERY_LENGTH);
    }

    #[test]
    fn overlong_query_still_searches() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            let doc = IndexableDocument::new("doc-1", "findable content");
            idx.index_document(&cx, &doc).await.expect("index");
            idx.commit(&cx).await.expect("commit");

            // Build a query with the search term followed by lots of padding.
            let mut long_query = "findable ".to_owned();
            long_query.push_str(&"padding ".repeat(2000));
            assert!(long_query.len() > MAX_QUERY_LENGTH);

            let results = idx
                .search(&cx, &long_query, 10)
                .await
                .expect("should not error");
            assert!(
                !results.is_empty(),
                "truncated query should still find docs"
            );
        });
    }

    // ─── LexicalHit serde test ───────────────────────────────────────────

    #[test]
    fn lexical_hit_serde_roundtrip() {
        let hit = LexicalHit {
            doc_id: "doc-42".into(),
            bm25_score: 2.75,
            rank: 0,
            snippet: Some("<b>Rust</b> is great".into()),
            query_type: QueryExplanation::Simple,
            metadata: Some(serde_json::json!({"lang": "en"})),
        };
        let json = serde_json::to_string(&hit).expect("serialize");
        let roundtripped: LexicalHit = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(roundtripped.doc_id, "doc-42");
        assert!((roundtripped.bm25_score - 2.75).abs() < f32::EPSILON);
        assert_eq!(roundtripped.rank, 0);
        assert_eq!(
            roundtripped.snippet.as_deref(),
            Some("<b>Rust</b> is great")
        );
        assert_eq!(roundtripped.query_type, QueryExplanation::Simple);
    }

    // ─── Special character robustness tests ──────────────────────────────

    #[test]
    fn search_with_special_chars_no_crash() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            let doc = IndexableDocument::new("doc-1", "some content");
            idx.index_document(&cx, &doc).await.expect("index");
            idx.commit(&cx).await.expect("commit");

            // These queries contain special characters that could trip up a parser.
            // In lenient mode, Tantivy should handle them gracefully.
            for query in &["@user", "#hashtag", "foo:bar", "a+b", "hello!"] {
                let result = idx.search(&cx, query, 10).await;
                assert!(result.is_ok(), "query '{query}' should not error");
            }
        });
    }

    #[test]
    fn search_no_results_returns_empty_not_error() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            let doc = IndexableDocument::new("doc-1", "hello world");
            idx.index_document(&cx, &doc).await.expect("index");
            idx.commit(&cx).await.expect("commit");

            let results = idx
                .search(&cx, "nonexistentterm", 10)
                .await
                .expect("no error");
            assert!(results.is_empty());
        });
    }
}
