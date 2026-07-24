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

pub mod cass_compat;
pub mod quill_contract;

pub use cass_compat::{
    CASS_SCHEMA_HASH, CASS_SCHEMA_VERSION, CassDocument, CassDocumentRef, CassFields,
    CassMergeStatus, CassQueryFilters, CassQueryToken, CassSourceFilter, CassTantivyIndex,
    CassWildcardPattern, cass_build_preview, cass_build_schema, cass_build_tantivy_query,
    cass_ensure_tokenizer, cass_fields_from_schema, cass_generate_edge_ngrams,
    cass_has_boolean_operators, cass_index_dir, cass_open_search_reader, cass_parse_boolean_query,
    cass_regex_query_cached, cass_regex_query_uncached, cass_sanitize_query,
    cass_schema_hash_matches,
};

// Re-export tantivy types that appear in frankensearch-lexical's public API.
// Consumers can import these from `frankensearch::lexical::` instead of adding
// a direct tantivy dependency.
pub use tantivy::collector::{Count, TopDocs};
pub use tantivy::query::{BooleanQuery, Occur, Query, TermQuery};
pub use tantivy::schema::{Field, IndexRecordOption, Schema, Value};
pub use tantivy::{
    self as tantivy_crate, DocAddress, Index, IndexReader, IndexWriter, ReloadPolicy, Searcher,
    TantivyDocument, Term,
};

use std::path::{Path, PathBuf};
use std::sync::RwLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use asupersync::Cx;
use asupersync::sync::Mutex;
use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::traits::{LexicalSearch, SearchFuture};
use frankensearch_core::types::{DocId, IndexableDocument, ScoreSource, ScoredResult};
use serde::{Deserialize, Serialize};
#[cfg(feature = "bench-internals")]
use tantivy::HasLen;
use tantivy::collector::DocSetCollector;
#[cfg(feature = "bench-internals")]
use tantivy::directory::Directory;
use tantivy::query::QueryParser;
use tantivy::schema::{FAST, STORED, STRING, TextFieldIndexing, TextOptions};
use tantivy::tokenizer::{TextAnalyzer, Token, TokenStream, Tokenizer};
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

/// Default Tantivy snippet window value. Tantivy names this a character limit,
/// but observably compares UTF-8 byte offsets and cuts at token boundaries.
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
    /// Tantivy snippet window value. Despite the upstream name, the pinned
    /// implementation enforces this against UTF-8 byte offsets at token
    /// boundaries.
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

/// Raw lexical hit containing BM25 score and Tantivy doc address.
///
/// This is useful for callers that need custom field extraction from stored
/// documents while still reusing frankensearch's query execution helpers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LexicalDocHit {
    /// BM25 relevance score returned by Tantivy.
    pub bm25_score: f32,
    /// 0-based rank in the returned page.
    pub rank: usize,
    /// Tantivy document address inside a segment.
    pub doc_address: DocAddress,
}

/// Paginated lexical search result containing both the matched hits and the
/// total number of documents matching the query.
///
/// The `total_count` reflects **all** matching documents in the index, not just
/// the page returned in `hits`. Clients can use this for pagination UI
/// (e.g., `page 2 of ceil(total_count / page_size)`).
#[derive(Debug, Clone, PartialEq)]
pub struct LexicalSearchResult {
    /// The paginated slice of matching documents.
    pub hits: Vec<LexicalDocHit>,
    /// Total number of documents matching the query across the entire index.
    pub total_count: usize,
}

/// Lightweight lexical hit used by hot paths that only need `doc_id` + score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LexicalIdHit {
    /// Unique document identifier. `DocId` (`CompactString`, SSO) so the
    /// per-hit id-materialization clone (`ord_table` lookup / docstore) is an
    /// inline memcpy for short ids and needs no `String→CompactString`
    /// re-conversion when it flows into `ScoredResult` at the fusion boundary.
    pub doc_id: DocId,
    /// BM25 relevance score.
    pub bm25_score: f32,
    /// 0-based rank in the returned page.
    pub rank: usize,
}

/// Dev-only ranked oracle evidence retaining Tantivy's full native tie key.
#[cfg(feature = "tantivy-oracle")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OracleRankedHit {
    /// External document identifier used for cross-engine alignment.
    pub doc_id: String,
    /// Exact BM25 score representation.
    pub score_bits: u32,
    /// Native rank in the fetched oracle result list.
    pub rank: usize,
    /// Tantivy segment ordinal (the first half of `DocAddress`).
    pub segment_ord: u32,
    /// Segment-local document ID (the second half of `DocAddress`).
    pub segment_doc_id: u32,
    /// Rendered snippet; `None` remains distinct from an empty snippet.
    pub snippet: Option<String>,
}

/// Dev-only complete observation used by the Quill conformance gauntlet.
#[cfg(feature = "tantivy-oracle")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OracleQueryObservation {
    /// Requested top-k rows in native Tantivy order.
    pub hits: Vec<OracleRankedHit>,
    /// Expanded exact-score group at the top-k cutoff.
    pub cutoff_tie_group: Vec<OracleRankedHit>,
    /// False when the configured expansion budget ended inside the tie group.
    pub cutoff_tie_complete: bool,
    /// Exact total number of matches, independent of top-k.
    pub total_count: usize,
    /// Exact live-document count.
    pub doc_count: usize,
}

/// Execute a pre-built Tantivy query with offset pagination.
///
/// This helper centralizes error mapping and result-shape normalization so
/// downstream callers can keep custom query construction while reusing the
/// lexical execution core from this crate.
///
/// # Errors
///
/// Returns [`SearchError::SubsystemError`] when Tantivy search fails.
#[instrument(skip(searcher, query), fields(limit = limit, offset = offset))]
pub fn execute_query_with_offset(
    searcher: &Searcher,
    query: &dyn tantivy::query::Query,
    limit: usize,
    offset: usize,
) -> SearchResult<LexicalSearchResult> {
    if limit == 0 {
        let total_count =
            searcher
                .search(query, &Count)
                .map_err(|e| SearchError::SubsystemError {
                    subsystem: "tantivy",
                    source: Box::new(e),
                })?;
        return Ok(LexicalSearchResult {
            hits: Vec::new(),
            total_count,
        });
    }

    let (top_docs, total_count) = searcher
        .search(
            query,
            &(
                TopDocs::with_limit(limit)
                    .and_offset(offset)
                    .order_by_score(),
                Count,
            ),
        )
        .map_err(|e| SearchError::SubsystemError {
            subsystem: "tantivy",
            source: Box::new(e),
        })?;

    let hits = top_docs
        .into_iter()
        .enumerate()
        .map(|(rank, (bm25_score, doc_address))| LexicalDocHit {
            bm25_score,
            rank,
            doc_address,
        })
        .collect();

    Ok(LexicalSearchResult { hits, total_count })
}

/// Execute a query for the top-`limit` hits by BM25 score without computing the
/// total match count.
///
/// Unlike [`execute_query_with_offset`], this omits the [`Count`] collector.
/// `Count` must visit every matching document, while this ID-only path only
/// needs the ranked page.
///
/// # Errors
///
/// Returns [`SearchError::SubsystemError`] when Tantivy search fails.
pub fn execute_top_k(
    searcher: &Searcher,
    query: &dyn tantivy::query::Query,
    limit: usize,
    offset: usize,
) -> SearchResult<Vec<LexicalDocHit>> {
    if limit == 0 {
        return Ok(Vec::new());
    }

    let top_docs = searcher
        .search(
            query,
            &TopDocs::with_limit(limit)
                .and_offset(offset)
                .order_by_score(),
        )
        .map_err(|e| SearchError::SubsystemError {
            subsystem: "tantivy",
            source: Box::new(e),
        })?;

    Ok(top_docs
        .into_iter()
        .enumerate()
        .map(|(rank, (bm25_score, doc_address))| LexicalDocHit {
            bm25_score,
            rank,
            doc_address,
        })
        .collect())
}

/// Load a stored Tantivy document by address.
///
/// # Errors
///
/// Returns [`SearchError::SubsystemError`] when document loading fails.
pub fn load_doc(searcher: &Searcher, doc_address: DocAddress) -> SearchResult<TantivyDocument> {
    searcher
        .doc(doc_address)
        .map_err(|e| SearchError::SubsystemError {
            subsystem: "tantivy",
            source: Box::new(e),
        })
}

fn is_counted_query_syntax(byte: u8) -> bool {
    matches!(
        byte,
        b'"' | b'\''
            | b'('
            | b')'
            | b':'
            | b'+'
            | b'-'
            | b'!'
            | b'^'
            | b'['
            | b']'
            | b'{'
            | b'}'
            | b'*'
            | b'?'
            | b'~'
            | b'\\'
            | b'/'
    )
}

fn use_count_free_top_k(query: &str) -> bool {
    let mut term_count = 0usize;
    let mut in_term = false;

    for byte in query.bytes() {
        if byte.is_ascii_whitespace() {
            in_term = false;
            continue;
        }
        if is_counted_query_syntax(byte) {
            return false;
        }
        if !in_term {
            term_count += 1;
            if term_count > 2 {
                return false;
            }
            in_term = true;
        }
    }

    term_count > 0
}

/// Try to build a snippet generator for a query/content field pair.
///
/// Returns `None` if snippet generation cannot be initialized. This mirrors the
/// tolerant behavior used by `search_with_snippets`.
#[must_use]
pub fn try_build_snippet_generator(
    searcher: &Searcher,
    query: &dyn tantivy::query::Query,
    content_field: Field,
    snippet_config: &SnippetConfig,
) -> Option<tantivy::snippet::SnippetGenerator> {
    match tantivy::snippet::SnippetGenerator::create(searcher, query, content_field) {
        Ok(mut generator) => {
            generator.set_max_num_chars(snippet_config.max_chars);
            Some(generator)
        }
        Err(e) => {
            debug!(error = %e, "failed to create snippet generator, snippets will be absent");
            None
        }
    }
}

/// Render snippet HTML for a document with caller-specified highlight tags.
#[must_use]
pub fn render_snippet_html(
    snippet_generator: &tantivy::snippet::SnippetGenerator,
    doc: &TantivyDocument,
    highlight_prefix: &str,
    highlight_postfix: &str,
) -> Option<String> {
    let mut snippet = snippet_generator.snippet_from_doc(doc);
    snippet.set_snippet_prefix_postfix(highlight_prefix, highlight_postfix);
    let html = snippet.to_html();
    if html.is_empty() { None } else { Some(html) }
}

// ─── Schema fields ──────────────────────────────────────────────────────────

/// Named fields from the Tantivy schema for type-safe access.
#[derive(Debug, Clone, Copy)]
struct SchemaFields {
    id: Field,
    content: Field,
    title: Field,
    metadata_json: Field,
    /// Dense per-document insertion ordinal, stored as a `u64` FAST (columnar)
    /// field. `None` for indexes created before this field existed (resolved by
    /// name in `from_index`); such indexes materialize ids via the docstore.
    ord: Option<Field>,
}

/// Build the frankensearch Tantivy schema.
fn build_schema() -> (Schema, SchemaFields) {
    build_schema_with_positions(true)
}

fn build_schema_with_positions(positions: bool) -> (Schema, SchemaFields) {
    let mut builder = Schema::builder();

    // ID: exact-match only, stored for retrieval.
    let id = builder.add_text_field("id", STRING | STORED);

    // Content: full-text indexed with our custom tokenizer, stored for snippet use.
    let index_record_option = if positions {
        tantivy::schema::IndexRecordOption::WithFreqsAndPositions
    } else {
        tantivy::schema::IndexRecordOption::WithFreqs
    };
    let content_options = TextOptions::default()
        .set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer(TOKENIZER_NAME)
                .set_index_option(index_record_option),
        )
        .set_stored();
    let content = builder.add_text_field("content", content_options);

    // Title: full-text indexed with our custom tokenizer, stored.
    let title_options = TextOptions::default()
        .set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer(TOKENIZER_NAME)
                .set_index_option(index_record_option),
        )
        .set_stored();
    let title = builder.add_text_field("title", title_options);

    // Metadata: stored as JSON string, not indexed.
    let metadata_json = builder.add_text_field("metadata_json", STORED);

    // Dense insertion ordinal: a flat bit-packed `u64` FAST column (no
    // dictionary) plus an external `ordinal -> doc_id` table lets id
    // materialization skip the stored-document decompress entirely. STORED too
    // so the table can be rebuilt from a reopened index if needed.
    let ord = builder.add_u64_field("ord", FAST | STORED);

    let schema = builder.build();
    let fields = SchemaFields {
        id,
        content,
        title,
        metadata_json,
        ord: Some(ord),
    };
    (schema, fields)
}

#[cfg(feature = "bench-internals")]
fn validate_benchmark_writer_threads(writer_threads: usize) -> SearchResult<()> {
    if writer_threads == 0 {
        return Err(SearchError::InvalidConfig {
            field: "tantivy.writer_threads".to_owned(),
            value: writer_threads.to_string(),
            reason: "benchmark writer thread count must be positive".to_owned(),
        });
    }
    Ok(())
}

/// Fused equivalent of Tantivy's `SimpleTokenizer` followed by `LowerCaser`.
///
/// ASCII characters are classified directly from their byte and each token is
/// lowercased without the generic lowercaser's extra `is_ascii` scan. Non-ASCII
/// characters retain the exact `char::is_alphanumeric` and `char::to_lowercase`
/// behavior of the former two-stage analyzer.
#[derive(Clone, Default)]
struct FrankensearchTokenizer {
    token: Token,
}

struct FrankensearchTokenStream<'a> {
    text: &'a str,
    cursor: usize,
    token: &'a mut Token,
}

#[inline]
fn tokenizer_next_char(text: &str, offset: usize) -> Option<(char, usize)> {
    let remaining = text.get(offset..)?;
    let first = *remaining.as_bytes().first()?;
    if first.is_ascii() {
        Some((char::from(first), offset + 1))
    } else {
        let ch = remaining.chars().next()?;
        Some((ch, offset + ch.len_utf8()))
    }
}

#[inline]
fn tokenizer_is_alphanumeric(ch: char) -> bool {
    if ch.is_ascii() {
        ch.is_ascii_alphanumeric()
    } else {
        ch.is_alphanumeric()
    }
}

impl Tokenizer for FrankensearchTokenizer {
    type TokenStream<'a> = FrankensearchTokenStream<'a>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        self.token.reset();
        FrankensearchTokenStream {
            text,
            cursor: 0,
            token: &mut self.token,
        }
    }
}

impl TokenStream for FrankensearchTokenStream<'_> {
    fn advance(&mut self) -> bool {
        self.token.text.clear();
        self.token.position = self.token.position.wrapping_add(1);

        while let Some((ch, next_cursor)) = tokenizer_next_char(self.text, self.cursor) {
            if !tokenizer_is_alphanumeric(ch) {
                self.cursor = next_cursor;
                continue;
            }

            let offset_from = self.cursor;
            let mut offset_to = next_cursor;
            let mut resume_at = next_cursor;
            let mut all_ascii = ch.is_ascii();
            while let Some((next_ch, after_next)) = tokenizer_next_char(self.text, resume_at) {
                if !tokenizer_is_alphanumeric(next_ch) {
                    resume_at = after_next;
                    break;
                }
                all_ascii &= next_ch.is_ascii();
                offset_to = after_next;
                resume_at = after_next;
            }

            self.token.offset_from = offset_from;
            self.token.offset_to = offset_to;
            let source = &self.text[offset_from..offset_to];
            if all_ascii {
                self.token.text.push_str(source);
                self.token.text.make_ascii_lowercase();
            } else {
                for source_char in source.chars() {
                    self.token.text.extend(source_char.to_lowercase());
                }
            }
            self.cursor = resume_at;
            return true;
        }
        false
    }

    fn token(&self) -> &Token {
        self.token
    }

    fn token_mut(&mut self) -> &mut Token {
        self.token
    }
}

/// Build and register the custom tokenizer.
///
/// Semantics match `SimpleTokenizer` (split on non-alphanumeric characters)
/// followed by `LowerCaser`. `POL-358` therefore becomes `pol`, `358`. For
/// hyphen-preserving tokenization see `cass_ensure_tokenizer`.
fn build_tokenizer() -> TextAnalyzer {
    TextAnalyzer::from(FrankensearchTokenizer::default())
}

/// Return the production analyzer for a same-binary benchmark comparator.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
#[must_use]
pub fn default_tokenizer_for_bench() -> TextAnalyzer {
    build_tokenizer()
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
    /// Append-only `ordinal -> doc_id` table backing the fast id-materialization
    /// path. Index `i` holds the `doc_id` of the document assigned ordinal `i`.
    /// Grows by one per indexed document (including re-upserts); ordinals are
    /// monotonic and never reused, so it stays correct across deletes/merges
    /// (a deleted doc's ordinal simply never appears in search results).
    ord_table: RwLock<Vec<DocId>>,
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
    fn map_writer_lock_error(phase: &str, error: asupersync::sync::LockError) -> SearchError {
        match error {
            asupersync::sync::LockError::Poisoned => SearchError::SubsystemError {
                subsystem: "tantivy",
                source: Box::new(std::io::Error::other("writer mutex poisoned")),
            },
            asupersync::sync::LockError::Cancelled => SearchError::Cancelled {
                phase: phase.into(),
                reason: "writer lock cancelled".into(),
            },
            asupersync::sync::LockError::PolledAfterCompletion => SearchError::SubsystemError {
                subsystem: "tantivy",
                source: Box::new(std::io::Error::other(format!(
                    "writer mutex future reused after completion during {phase}"
                ))),
            },
            asupersync::sync::LockError::TimedOut(deadline) => SearchError::Cancelled {
                phase: phase.into(),
                reason: format!("writer lock timed out at {deadline:?}"),
            },
        }
    }

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

        Self::from_index(
            index,
            schema,
            fields,
            Some(path.to_path_buf()),
            WRITER_HEAP_BYTES,
        )
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

        Self::from_index(
            index,
            schema,
            fields,
            Some(path.to_path_buf()),
            WRITER_HEAP_BYTES,
        )
    }

    /// Create an in-memory Tantivy index (useful for testing).
    ///
    /// # Errors
    ///
    /// Returns `SearchError::SubsystemError` if the index cannot be created.
    pub fn in_memory() -> SearchResult<Self> {
        let (schema, fields) = build_schema();
        let index = Index::create_in_ram(schema.clone());
        Self::from_index(index, schema, fields, None, WRITER_HEAP_BYTES)
    }

    /// Create an in-memory oracle with one deterministic indexing worker.
    ///
    /// This is reserved for differential campaigns that compare Tantivy's
    /// native `DocAddress` tie order across repeated runs. Shipping callers
    /// should use [`Self::in_memory`], which retains Tantivy's default writer
    /// parallelism.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::SubsystemError` if the index writer cannot be
    /// created.
    #[cfg(feature = "tantivy-oracle")]
    #[doc(hidden)]
    pub fn in_memory_single_threaded_oracle() -> SearchResult<Self> {
        let (schema, fields) = build_schema();
        let index = Index::create_in_ram(schema.clone());
        Self::from_index_with_writer_threads(
            index,
            schema,
            fields,
            None,
            WRITER_HEAP_BYTES,
            Some(1),
        )
    }

    /// Create an in-memory index with an explicit writer heap budget.
    ///
    /// This exists only for same-binary benchmark comparisons of Tantivy's
    /// writer parallelism. Shipping callers always use [`Self::in_memory`].
    #[cfg(feature = "bench-internals")]
    #[doc(hidden)]
    pub fn in_memory_with_writer_heap_bytes(writer_heap_bytes: usize) -> SearchResult<Self> {
        let (schema, fields) = build_schema();
        let index = Index::create_in_ram(schema.clone());
        Self::from_index(index, schema, fields, None, writer_heap_bytes)
    }

    /// Create an in-memory oracle with an explicitly pinned benchmark schema,
    /// writer count, and heap budget.
    ///
    /// This is the same shipping analyzer, document conversion, writer, and
    /// commit path used by [`LexicalSearch`]. Only the knobs that the Quill QG
    /// matrix must hold equal across engines are exposed. Shipping callers use
    /// [`Self::in_memory`] instead.
    ///
    /// # Errors
    ///
    /// Returns an invalid-config error for zero writer threads, or the ordinary
    /// Tantivy writer-construction error for an unsupported heap budget.
    #[cfg(feature = "bench-internals")]
    #[doc(hidden)]
    pub fn in_memory_with_benchmark_config(
        writer_heap_bytes: usize,
        writer_threads: usize,
        positions: bool,
    ) -> SearchResult<Self> {
        validate_benchmark_writer_threads(writer_threads)?;
        let (schema, fields) = build_schema_with_positions(positions);
        let index = Index::create_in_ram(schema.clone());
        Self::from_index_with_writer_threads(
            index,
            schema,
            fields,
            None,
            writer_heap_bytes,
            Some(writer_threads),
        )
    }

    /// Create an on-disk oracle with the same explicit benchmark pins as
    /// [`Self::in_memory_with_benchmark_config`].
    ///
    /// # Errors
    ///
    /// Returns a typed filesystem, invalid-config, or Tantivy construction
    /// error.
    #[cfg(feature = "bench-internals")]
    #[doc(hidden)]
    pub fn create_with_benchmark_config(
        path: &Path,
        writer_heap_bytes: usize,
        writer_threads: usize,
        positions: bool,
    ) -> SearchResult<Self> {
        validate_benchmark_writer_threads(writer_threads)?;
        let (schema, fields) = build_schema_with_positions(positions);
        std::fs::create_dir_all(path).map_err(|error| SearchError::SubsystemError {
            subsystem: "tantivy",
            source: Box::new(error),
        })?;
        let index = Index::create_in_dir(path, schema.clone()).map_err(|error| {
            SearchError::SubsystemError {
                subsystem: "tantivy",
                source: Box::new(error),
            }
        })?;
        Self::from_index_with_writer_threads(
            index,
            schema,
            fields,
            Some(path.to_path_buf()),
            writer_heap_bytes,
            Some(writer_threads),
        )
    }

    /// Reopen an on-disk oracle with the benchmark's pinned writer budget.
    ///
    /// # Errors
    ///
    /// Returns a typed missing-index, invalid-config, or Tantivy construction
    /// error.
    #[cfg(feature = "bench-internals")]
    #[doc(hidden)]
    pub fn open_with_benchmark_config(
        path: &Path,
        writer_heap_bytes: usize,
        writer_threads: usize,
        positions: bool,
    ) -> SearchResult<Self> {
        validate_benchmark_writer_threads(writer_threads)?;
        if !path.exists() {
            return Err(SearchError::IndexNotFound {
                path: path.to_path_buf(),
            });
        }
        let (schema, fields) = build_schema_with_positions(positions);
        let index = Index::open_in_dir(path).map_err(|error| SearchError::SubsystemError {
            subsystem: "tantivy",
            source: Box::new(error),
        })?;
        Self::from_index_with_writer_threads(
            index,
            schema,
            fields,
            Some(path.to_path_buf()),
            writer_heap_bytes,
            Some(writer_threads),
        )
    }

    /// Disable automatic segment merging for a force-merge benchmark setup.
    ///
    /// # Errors
    ///
    /// Returns a typed cancellation or writer-lock error.
    #[cfg(feature = "bench-internals")]
    #[doc(hidden)]
    pub async fn benchmark_disable_auto_merge(&self, cx: &Cx) -> SearchResult<()> {
        let writer =
            self.writer.lock(cx).await.map_err(|error| {
                Self::map_writer_lock_error("tantivy.benchmark_no_merge", error)
            })?;
        writer.set_merge_policy(Box::new(tantivy::merge_policy::NoMergePolicy));
        Ok(())
    }

    /// Force-merge every currently searchable segment and reload the reader.
    ///
    /// This exists only for QG-5's same-binary oracle arm. It invokes
    /// Tantivy's real merge machinery and waits for the merge future; no
    /// benchmark-only codec or document path is involved.
    ///
    /// # Errors
    ///
    /// Returns a typed cancellation, writer-lock, merge, or reader-reload
    /// error.
    #[cfg(feature = "bench-internals")]
    #[doc(hidden)]
    pub async fn benchmark_force_merge(&self, cx: &Cx) -> SearchResult<()> {
        let segment_ids =
            self.index
                .searchable_segment_ids()
                .map_err(|error| SearchError::SubsystemError {
                    subsystem: "tantivy",
                    source: Box::new(error),
                })?;
        if segment_ids.len() < 2 {
            return Ok(());
        }
        {
            let mut writer = self
                .writer
                .lock(cx)
                .await
                .map_err(|error| Self::map_writer_lock_error("tantivy.force_merge", error))?;
            writer
                .merge(&segment_ids)
                .wait()
                .map_err(|error| SearchError::SubsystemError {
                    subsystem: "tantivy",
                    source: Box::new(error),
                })?;
        }
        self.reader
            .reload()
            .map_err(|error| SearchError::SubsystemError {
                subsystem: "tantivy",
                source: Box::new(error),
            })?;
        Ok(())
    }

    /// Return `(searchable_segments, managed_index_bytes)` for index-build benchmarks.
    #[cfg(feature = "bench-internals")]
    #[doc(hidden)]
    pub fn benchmark_index_layout(&self) -> SearchResult<(usize, u64)> {
        let segment_metas =
            self.index
                .searchable_segment_metas()
                .map_err(|e| SearchError::SubsystemError {
                    subsystem: "tantivy",
                    source: Box::new(e),
                })?;
        let segment_count = segment_metas.len();
        let managed_files = self.index.directory().list_managed_files();
        let mut index_bytes = 0u64;
        for path in segment_metas
            .iter()
            .flat_map(|segment_meta| segment_meta.list_files())
            .filter(|path| managed_files.contains(path))
        {
            let file = self.index.directory().open_read(&path).map_err(|e| {
                SearchError::SubsystemError {
                    subsystem: "tantivy",
                    source: Box::new(e),
                }
            })?;
            index_bytes = index_bytes.saturating_add(u64::try_from(file.len()).unwrap_or(u64::MAX));
        }
        Ok((segment_count, index_bytes))
    }

    /// Internal constructor shared by `create`, `open`, and `in_memory`.
    fn from_index(
        index: Index,
        schema: Schema,
        fields: SchemaFields,
        path: Option<PathBuf>,
        writer_heap_bytes: usize,
    ) -> SearchResult<Self> {
        Self::from_index_with_writer_threads(index, schema, fields, path, writer_heap_bytes, None)
    }

    fn from_index_with_writer_threads(
        index: Index,
        _schema: Schema,
        mut fields: SchemaFields,
        path: Option<PathBuf>,
        writer_heap_bytes: usize,
        writer_threads: Option<usize>,
    ) -> SearchResult<Self> {
        // Resolve the `ord` fast field against the *actual* index schema so
        // indexes created before this field existed (no `ord`) cleanly fall back
        // to docstore materialization instead of using a phantom field handle.
        fields.ord = index.schema().get_field("ord").ok();

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

        let writer = writer_threads
            .map_or_else(
                || index.writer(writer_heap_bytes),
                |thread_count| index.writer_with_num_threads(thread_count, writer_heap_bytes),
            )
            .map_err(|e| SearchError::SubsystemError {
                subsystem: "tantivy",
                source: Box::new(e),
            })?;

        // Count existing documents.
        let searcher = reader.searcher();
        let doc_count = usize::try_from(searcher.num_docs()).unwrap_or(usize::MAX);

        // Restore the ordinal→doc_id table for a reopened on-disk index from its
        // sidecar so id materialization keeps the fast path instead of falling
        // back to the docstore. The sidecar is best-effort: if it is absent,
        // corrupt, or stale the table is left short and those ordinals resolve
        // via the docstore (correct, just unaccelerated).
        let mut ord_table = match (path.as_deref(), fields.ord) {
            (Some(dir), Some(_)) => Self::load_ord_table_sidecar(dir).unwrap_or_default(),
            _ => Vec::new(),
        };
        if fields.ord.is_some() {
            // Guard against a stale-short sidecar (e.g. a persist that failed on
            // the last commit): `Column::max_value` is O(1) columnar metadata, so
            // padding the table to cover the highest existing ordinal keeps the
            // next assigned ordinal (`ord_table.len()`) collision-free, and the
            // padded empty slots fall back to the docstore on read.
            let mut max_ord = 0u64;
            let mut saw_ord = false;
            for sr in searcher.segment_readers() {
                if let Ok(col) = sr.fast_fields().u64("ord")
                    && col.num_docs() > 0
                {
                    max_ord = max_ord.max(col.max_value());
                    saw_ord = true;
                }
            }
            if saw_ord {
                let needed = usize::try_from(max_ord)
                    .unwrap_or(usize::MAX)
                    .saturating_add(1);
                if ord_table.len() < needed {
                    ord_table.resize(needed, DocId::default());
                }
            }
        }

        Ok(Self {
            index,
            fields,
            reader,
            writer: Mutex::new(writer),
            doc_count: AtomicUsize::new(doc_count),
            ord_table: RwLock::new(ord_table),
            path,
        })
    }

    /// Load the persisted `ordinal → doc_id` table sidecar (`ord_table.json`)
    /// from an on-disk index directory. Returns `None` on any error or absence
    /// so the caller can fall back to docstore materialization.
    fn load_ord_table_sidecar(dir: &Path) -> Option<Vec<DocId>> {
        let file = std::fs::File::open(dir.join("ord_table.json")).ok()?;
        serde_json::from_reader(std::io::BufReader::new(file)).ok()
    }

    /// Persist the `ordinal → doc_id` table to the index directory sidecar so a
    /// later `open` can restore the fast id-materialization path. Best-effort:
    /// errors are logged and swallowed (the in-memory fast path is unaffected;
    /// only a future reopen would fall back to the docstore). Written atomically
    /// via a temp file + rename. No-op for in-memory indexes / pre-`ord` schemas.
    fn persist_ord_table(&self) {
        let Some(dir) = self.path.as_ref() else {
            return;
        };
        if self.fields.ord.is_none() {
            return;
        }
        let Ok(table) = self.ord_table.read() else {
            return;
        };
        let tmp = dir.join("ord_table.json.tmp");
        let final_path = dir.join("ord_table.json");
        let write = std::fs::File::create(&tmp).and_then(|file| {
            let mut writer = std::io::BufWriter::new(file);
            serde_json::to_writer(&mut writer, &*table).map_err(std::io::Error::other)?;
            std::io::Write::flush(&mut writer)
        });
        match write {
            Ok(()) => {
                if let Err(e) = std::fs::rename(&tmp, &final_path) {
                    debug!(error = %e, "ord_table sidecar rename failed; reopen will use docstore");
                }
            }
            Err(e) => {
                debug!(error = %e, "ord_table sidecar write failed; reopen will use docstore");
                let _ = std::fs::remove_file(&tmp);
            }
        }
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

    /// Assign the next dense ordinal to `tantivy_doc` and append `doc_id` to the
    /// `ord_table` so id materialization can skip the docstore decompress.
    ///
    /// Must be called while holding the writer lock: that serializes ordinal
    /// assignment with `add_document`, so ordinal `i` always corresponds to
    /// `ord_table[i]`. No-op when the schema has no `ord` field (pre-`ord`
    /// indexes) or when the table lock is poisoned (the doc is left without an
    /// ordinal and that hit falls back to docstore materialization).
    fn assign_ord(&self, tantivy_doc: &mut TantivyDocument, doc_id: &str) {
        if let Some(ord_field) = self.fields.ord
            && let Ok(mut table) = self.ord_table.write()
        {
            let ord = u64::try_from(table.len()).unwrap_or(u64::MAX);
            table.push(DocId::from(doc_id));
            tantivy_doc.add_u64(ord_field, ord);
        }
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
            .map_err(|e| Self::map_writer_lock_error("tantivy.delete", e))?
            .delete_term(term);
        Ok(())
    }

    /// Returns the directory path for this index, if on-disk.
    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Cloneable handle to the underlying Tantivy index.
    ///
    /// This is primarily used by durability wrappers that protect/verify
    /// segment artifacts outside the lexical crate.
    #[must_use]
    pub fn index_handle(&self) -> Index {
        self.index.clone()
    }

    /// Truncate an overlong query and log a warning.
    fn truncate_query(query: &str) -> &str {
        // UTF-8 uses at least one byte per character, so this is a cheap
        // common-case proof that the query is within the character limit.
        if query.len() <= MAX_QUERY_LENGTH {
            return query;
        }

        // `char_indices().nth(MAX_QUERY_LENGTH)` points at the first character
        // outside the allowed prefix. If it is absent, the string exceeds the
        // byte count but still contains at most MAX_QUERY_LENGTH characters.
        let Some((end, _)) = query.char_indices().nth(MAX_QUERY_LENGTH) else {
            return query;
        };
        warn!(
            original_len_bytes = query.len(),
            max_chars = MAX_QUERY_LENGTH,
            "query truncated to MAX_QUERY_LENGTH"
        );
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
        let search_result = execute_query_with_offset(&searcher, &*parsed, limit, 0)?;
        let snippet_gen =
            try_build_snippet_generator(&searcher, &*parsed, self.fields.content, snippet_config);

        debug!(
            hits = search_result.hits.len(),
            total_count = search_result.total_count,
            query_type = %explanation,
            "tantivy search_with_snippets completed"
        );

        let mut results = Vec::with_capacity(search_result.hits.len());
        for hit in search_result.hits {
            let doc = load_doc(&searcher, hit.doc_address)?;

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
            let snippet = snippet_gen.as_ref().and_then(|generator| {
                render_snippet_html(
                    generator,
                    &doc,
                    &snippet_config.highlight_prefix,
                    &snippet_config.highlight_postfix,
                )
            });

            results.push(LexicalHit {
                doc_id,
                bm25_score: hit.bm25_score,
                rank: hit.rank,
                snippet,
                query_type: explanation,
                metadata,
            });
        }

        Ok(results)
    }

    /// Observe the shipping parser/search/snippet path for Quill conformance.
    ///
    /// This surface is deliberately feature-gated: it retains the full Tantivy
    /// `DocAddress`, exact count, score bits, and an expanded cutoff tie group.
    /// Shipping consumers should use the ordinary lexical APIs instead.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if query execution or stored-field loading fails.
    #[cfg(feature = "tantivy-oracle")]
    #[instrument(skip_all, fields(query = %query, limit = limit, tie_expansion_limit = tie_expansion_limit))]
    pub fn oracle_observe_query(
        &self,
        _cx: &Cx,
        query: &str,
        limit: usize,
        tie_expansion_limit: usize,
        snippet_config: &SnippetConfig,
    ) -> SearchResult<OracleQueryObservation> {
        let query = Self::truncate_query(query);
        if query.trim().is_empty() {
            return Ok(OracleQueryObservation {
                hits: Vec::new(),
                cutoff_tie_group: Vec::new(),
                cutoff_tie_complete: true,
                total_count: 0,
                doc_count: self.doc_count.load(Ordering::Relaxed),
            });
        }

        let parsed = self.parse_query_lenient(query);
        let searcher = self.reader.searcher();
        let fetch_limit = if limit == 0 {
            0
        } else {
            limit.saturating_add(tie_expansion_limit)
        };
        let search_result = execute_query_with_offset(&searcher, &*parsed, fetch_limit, 0)?;
        let snippet_gen =
            try_build_snippet_generator(&searcher, &*parsed, self.fields.content, snippet_config);
        let mut materialized = Vec::with_capacity(search_result.hits.len());
        for hit in search_result.hits {
            let doc = load_doc(&searcher, hit.doc_address)?;
            let doc_id = doc
                .get_first(self.fields.id)
                .and_then(|value| value.as_str())
                .unwrap_or_default()
                .to_owned();
            let snippet = snippet_gen.as_ref().and_then(|generator| {
                render_snippet_html(
                    generator,
                    &doc,
                    &snippet_config.highlight_prefix,
                    &snippet_config.highlight_postfix,
                )
            });
            materialized.push(OracleRankedHit {
                doc_id,
                score_bits: hit.bm25_score.to_bits(),
                rank: hit.rank,
                segment_ord: hit.doc_address.segment_ord,
                segment_doc_id: hit.doc_address.doc_id,
                snippet,
            });
        }

        let top_len = limit.min(materialized.len());
        let cutoff_bits = top_len
            .checked_sub(1)
            .and_then(|index| materialized.get(index))
            .map(|hit| hit.score_bits);
        let cutoff_tie_group = cutoff_bits.map_or_else(Vec::new, |cutoff| {
            materialized
                .iter()
                .filter(|hit| {
                    f32::from_bits(hit.score_bits)
                        .total_cmp(&f32::from_bits(cutoff))
                        .is_eq()
                })
                .cloned()
                .collect()
        });
        let cutoff_tie_complete = cutoff_bits.is_none_or(|cutoff| {
            search_result.total_count <= fetch_limit
                || materialized.last().is_none_or(|last| {
                    !f32::from_bits(last.score_bits)
                        .total_cmp(&f32::from_bits(cutoff))
                        .is_eq()
                })
        });
        materialized.truncate(top_len);

        Ok(OracleQueryObservation {
            hits: materialized,
            cutoff_tie_group,
            cutoff_tie_complete,
            total_count: search_result.total_count,
            doc_count: self.doc_count.load(Ordering::Relaxed),
        })
    }

    /// Search and return only `(doc_id, score, rank)` rows.
    ///
    /// This avoids metadata JSON decoding and is intended for latency-critical
    /// callers that only require identifiers plus BM25 scores.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if the query cannot be parsed or search fails.
    #[instrument(skip_all, fields(query = %query, limit = limit))]
    pub fn search_doc_ids(
        &self,
        _cx: &Cx,
        query: &str,
        limit: usize,
    ) -> SearchResult<Vec<LexicalIdHit>> {
        let query = Self::truncate_query(query);
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }

        let parsed = self.parse_query_lenient(query);
        let searcher = self.reader.searcher();
        let hits = if use_count_free_top_k(query) {
            execute_top_k(&searcher, &*parsed, limit, 0)?
        } else {
            execute_query_with_offset(&searcher, &*parsed, limit, 0)?.hits
        };
        self.collect_id_hits(&searcher, hits)
    }

    /// Enumerate every live document identifier in the committed index.
    ///
    /// This is intended for generation reconciliation and repair paths that
    /// cannot trust an external manifest. Results are sorted and deduplicated
    /// so callers can stage deterministic deletes even when an older index
    /// contains duplicate IDs.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if a live stored document cannot be loaded or the
    /// segment ordinal cannot be represented by Tantivy's address type.
    pub fn all_doc_ids(&self) -> SearchResult<Vec<DocId>> {
        let searcher = self.reader.searcher();
        let mut ids = Vec::with_capacity(self.doc_count.load(Ordering::Relaxed));
        for (segment_ord, segment) in searcher.segment_readers().iter().enumerate() {
            let segment_ord =
                u32::try_from(segment_ord).map_err(|_| SearchError::InvalidConfig {
                    field: "tantivy.segment_ord".to_owned(),
                    value: segment_ord.to_string(),
                    reason: "segment ordinal must fit in u32".to_owned(),
                })?;
            for doc_id in 0..segment.max_doc() {
                if segment.is_deleted(doc_id) {
                    continue;
                }
                let doc = load_doc(&searcher, DocAddress::new(segment_ord, doc_id))?;
                if let Some(doc_id) = doc
                    .get_first(self.fields.id)
                    .and_then(|value| value.as_str())
                {
                    ids.push(DocId::from(doc_id));
                }
            }
        }
        ids.sort_unstable();
        ids.dedup();
        Ok(ids)
    }

    /// Export every live committed document for exact-generation shadow replay.
    ///
    /// Results are ordered by document id. Duplicate ids from legacy indexes
    /// collapse deterministically to the last live segment/document address.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if a stored live document cannot be loaded or its
    /// metadata payload is malformed.
    pub fn all_documents(&self) -> SearchResult<Vec<IndexableDocument>> {
        let searcher = self.reader.searcher();
        let mut documents = std::collections::BTreeMap::new();
        for (segment_ord, segment) in searcher.segment_readers().iter().enumerate() {
            let segment_ord =
                u32::try_from(segment_ord).map_err(|_| SearchError::InvalidConfig {
                    field: "tantivy.segment_ord".to_owned(),
                    value: segment_ord.to_string(),
                    reason: "segment ordinal must fit in u32".to_owned(),
                })?;
            for doc_id in 0..segment.max_doc() {
                if segment.is_deleted(doc_id) {
                    continue;
                }
                let doc = load_doc(&searcher, DocAddress::new(segment_ord, doc_id))?;
                let Some(id) = doc
                    .get_first(self.fields.id)
                    .and_then(|value| value.as_str())
                else {
                    continue;
                };
                let content = doc
                    .get_first(self.fields.content)
                    .and_then(|value| value.as_str())
                    .unwrap_or_default()
                    .to_owned();
                let title = doc
                    .get_first(self.fields.title)
                    .and_then(|value| value.as_str())
                    .filter(|value| !value.is_empty())
                    .map(str::to_owned);
                let metadata = doc
                    .get_first(self.fields.metadata_json)
                    .and_then(|value| value.as_str())
                    .map_or_else(
                        || Ok(std::collections::HashMap::new()),
                        |json| {
                            serde_json::from_str(json).map_err(|error| {
                                SearchError::SubsystemError {
                                    subsystem: "tantivy.shadow_export",
                                    source: Box::new(error),
                                }
                            })
                        },
                    )?;
                documents.insert(
                    id.to_owned(),
                    IndexableDocument {
                        id: id.to_owned(),
                        content,
                        title,
                        metadata,
                    },
                );
            }
        }
        Ok(documents.into_values().collect())
    }

    /// Pre-optimization baseline for [`Self::search_doc_ids`] that retains the
    /// discarded total-count collector. This is used only by the `doc_ids_topk`
    /// benchmark so the optimized and counted paths can be compared in one
    /// binary.
    #[doc(hidden)]
    #[instrument(skip_all, fields(query = %query, limit = limit))]
    pub fn search_doc_ids_counted(
        &self,
        _cx: &Cx,
        query: &str,
        limit: usize,
    ) -> SearchResult<Vec<LexicalIdHit>> {
        let query = Self::truncate_query(query);
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }

        let parsed = self.parse_query_lenient(query);
        let searcher = self.reader.searcher();
        let search_result = execute_query_with_offset(&searcher, &*parsed, limit, 0)?;
        self.collect_id_hits(&searcher, search_result.hits)
    }

    /// Pre-optimization baseline for [`Self::search_doc_ids`] that forces the
    /// docstore id-materialization path (ignoring the `ord` fast field + table).
    /// Used only by the `search_doc_ids_materialize` benchmark to A/B the fast
    /// materialization wiring in one binary; not for production use.
    #[doc(hidden)]
    #[instrument(skip_all, fields(query = %query, limit = limit))]
    pub fn search_doc_ids_via_docstore(
        &self,
        _cx: &Cx,
        query: &str,
        limit: usize,
    ) -> SearchResult<Vec<LexicalIdHit>> {
        let query = Self::truncate_query(query);
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }

        let parsed = self.parse_query_lenient(query);
        let searcher = self.reader.searcher();
        let hits = if use_count_free_top_k(query) {
            execute_top_k(&searcher, &*parsed, limit, 0)?
        } else {
            execute_query_with_offset(&searcher, &*parsed, limit, 0)?.hits
        };
        let mut results = Vec::with_capacity(hits.len());
        for hit in hits {
            results.push(LexicalIdHit {
                doc_id: self.docstore_id(&searcher, hit.doc_address)?,
                bm25_score: hit.bm25_score,
                rank: hit.rank,
            });
        }
        Ok(results)
    }

    /// Read a single hit's `doc_id` from the stored document (the docstore
    /// fallback path: decompresses the stored block to read the `id` field).
    fn docstore_id(&self, searcher: &Searcher, addr: DocAddress) -> SearchResult<DocId> {
        let doc = load_doc(searcher, addr)?;
        Ok(DocId::from(
            doc.get_first(self.fields.id)
                .and_then(|v| v.as_str())
                .unwrap_or_else(|| {
                    debug!("tantivy document missing id field, using empty doc_id");
                    ""
                }),
        ))
    }

    /// Materialize ranked Tantivy hits into [`LexicalIdHit`] rows.
    ///
    /// Fast path: read each hit's dense ordinal from the `ord` `u64` FAST column
    /// (a flat bit-packed read — no stored-document decompress, no dictionary
    /// seek) and resolve it through the in-memory `ord_table`. Falls back
    /// per-hit to [`Self::docstore_id`] for any ordinal the table cannot resolve
    /// (documents written before `ord` existed, a reopened-but-not-rebuilt
    /// table, or a poisoned lock), so results are identical either way.
    // The `ord_table` read guard is intentionally held across the whole hit
    // loop — it is the once-per-call snapshot every hit resolves against.
    #[allow(clippy::significant_drop_tightening)]
    fn collect_id_hits(
        &self,
        searcher: &Searcher,
        hits: Vec<LexicalDocHit>,
    ) -> SearchResult<Vec<LexicalIdHit>> {
        let mut results = Vec::with_capacity(hits.len());

        // Take the table snapshot once; `None` when the field is absent or the
        // table is empty → pure docstore path. `ord` columns are opened lazily
        // per *touched* segment (top-k hits cluster in a few segments, so we
        // avoid paying for segments no hit lands in).
        let table = self.fields.ord.and_then(|_| {
            let table = self.ord_table.read().ok()?;
            if table.is_empty() { None } else { Some(table) }
        });
        // Cache the opened `ord` column per segment in a flat Vec indexed by
        // `segment_ord` — O(1), no per-hit hash — while still opening each
        // column lazily on first touch (outer `None` = not yet opened) so
        // segments no hit lands in are never opened. `segment_ord` is a dense
        // in-range index for hits from this `searcher`. Only sized on the
        // fast-field path (`table` present); the docstore path never indexes it.
        let mut columns: Vec<Option<Option<tantivy::columnar::Column<u64>>>> = Vec::new();
        if table.is_some() {
            columns.resize_with(searcher.segment_readers().len(), || None);
        }

        for hit in hits {
            let addr = hit.doc_address;
            let doc_id = match table.as_ref().and_then(|table| {
                columns[addr.segment_ord as usize]
                    .get_or_insert_with(|| {
                        searcher
                            .segment_reader(addr.segment_ord)
                            .fast_fields()
                            .u64("ord")
                            .ok()
                    })
                    .as_ref()
                    .and_then(|c| c.first(addr.doc_id))
                    .and_then(|ord| usize::try_from(ord).ok())
                    .and_then(|ord| table.get(ord))
                    // Empty = a padded/stale slot (sidecar didn't cover this
                    // ordinal) → fall back to the docstore for the real id.
                    .filter(|id| !id.is_empty())
                    .cloned()
            }) {
                Some(id) => id,
                None => self.docstore_id(searcher, addr)?,
            };

            results.push(LexicalIdHit {
                doc_id,
                bm25_score: hit.bm25_score,
                rank: hit.rank,
            });
        }

        Ok(results)
    }

    fn hydrate_fusion_metadata_with_collector(
        &self,
        results: &mut [ScoredResult],
        unscored: bool,
    ) -> SearchResult<()> {
        let clauses: Vec<(Occur, Box<dyn Query>)> = results
            .iter()
            .filter(|result| result.lexical_score.is_some() && result.metadata.is_none())
            .map(|result| {
                let term = Term::from_field_text(self.fields.id, result.doc_id.as_str());
                (
                    Occur::Should,
                    Box::new(TermQuery::new(term, IndexRecordOption::Basic)) as Box<dyn Query>,
                )
            })
            .collect();
        if clauses.is_empty() {
            return Ok(());
        }

        let limit = clauses.len();
        let query = BooleanQuery::new(clauses);
        let searcher = self.reader.searcher();
        let doc_addresses: Vec<DocAddress> = if unscored {
            // Every clause is an exact lookup on the unique `id` field. Hydration
            // consumes neither BM25 scores nor hit order, so asking Tantivy to
            // score and heap-sort these documents is pure overhead.
            searcher
                .search(&query, &DocSetCollector)
                .map_err(|error| SearchError::SubsystemError {
                    subsystem: "tantivy",
                    source: Box::new(error),
                })?
                .into_iter()
                .collect()
        } else {
            searcher
                .search(&query, &TopDocs::with_limit(limit).order_by_score())
                .map_err(|error| SearchError::SubsystemError {
                    subsystem: "tantivy",
                    source: Box::new(error),
                })?
                .into_iter()
                .map(|(_, doc_address)| doc_address)
                .collect()
        };

        for doc_address in doc_addresses {
            let doc = load_doc(&searcher, doc_address)?;
            let doc_id = doc
                .get_first(self.fields.id)
                .and_then(|value| value.as_str())
                .unwrap_or_else(|| {
                    debug!("tantivy document missing id field while hydrating metadata");
                    ""
                });
            let metadata = doc
                .get_first(self.fields.metadata_json)
                .and_then(|value| value.as_str())
                .and_then(|raw| match serde_json::from_str(raw) {
                    Ok(value) => Some(value),
                    Err(error) => {
                        debug!(%doc_id, %error, "failed to deserialize metadata JSON");
                        None
                    }
                });
            if let Some(result) = results
                .iter_mut()
                .find(|result| result.doc_id.as_str() == doc_id)
            {
                result.metadata = metadata;
            }
        }

        Ok(())
    }

    /// Scored-collector baseline retained for the exact hydration A/B benchmark.
    #[cfg(feature = "bench-internals")]
    #[doc(hidden)]
    pub fn hydrate_fusion_metadata_scored_for_bench<'a>(
        &'a self,
        _cx: &'a Cx,
        results: &'a mut [ScoredResult],
    ) -> SearchFuture<'a, ()> {
        Box::pin(async move { self.hydrate_fusion_metadata_with_collector(results, false) })
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

            if query.trim().is_empty() || limit == 0 {
                return Ok(Vec::new());
            }

            let parsed = self.parse_query_lenient(query);

            let searcher = self.reader.searcher();
            let top_docs = searcher
                .search(&*parsed, &TopDocs::with_limit(limit).order_by_score())
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
                    doc_id: doc_id.into(),
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

    #[instrument(skip_all, fields(query = %query, limit = limit))]
    fn search_fusion_candidates<'a>(
        &'a self,
        cx: &'a Cx,
        query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, Vec<ScoredResult>> {
        Box::pin(async move {
            self.search_doc_ids(cx, query, limit).map(|hits| {
                hits.into_iter()
                    .map(|hit| ScoredResult {
                        doc_id: hit.doc_id,
                        score: hit.bm25_score,
                        source: ScoreSource::Lexical,
                        index: None,
                        fast_score: None,
                        quality_score: None,
                        lexical_score: Some(hit.bm25_score),
                        rerank_score: None,
                        explanation: None,
                        metadata: None,
                    })
                    .collect()
            })
        })
    }

    fn fusion_metadata_is_deferred(&self) -> bool {
        true
    }

    #[instrument(skip_all, fields(results = results.len()))]
    fn hydrate_fusion_metadata<'a>(
        &'a self,
        _cx: &'a Cx,
        results: &'a mut [ScoredResult],
    ) -> SearchFuture<'a, ()> {
        Box::pin(async move { self.hydrate_fusion_metadata_with_collector(results, true) })
    }

    fn index_document<'a>(
        &'a self,
        cx: &'a Cx,
        doc: &'a IndexableDocument,
    ) -> SearchFuture<'a, ()> {
        Box::pin(async move {
            let mut tantivy_doc = self.to_tantivy_doc(doc);

            {
                let writer = self
                    .writer
                    .lock(cx)
                    .await
                    .map_err(|e| Self::map_writer_lock_error("tantivy.index", e))?;

                // Delete any existing document with same ID (upsert semantics).
                let term = Term::from_field_text(self.fields.id, &doc.id);
                writer.delete_term(term);
                self.assign_ord(&mut tantivy_doc, &doc.id);
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
                let writer = self
                    .writer
                    .lock(cx)
                    .await
                    .map_err(|e| Self::map_writer_lock_error("tantivy.batch_index", e))?;

                for doc in docs {
                    let mut tantivy_doc = self.to_tantivy_doc(doc);
                    // Upsert: delete existing then add.
                    let term = Term::from_field_text(self.fields.id, &doc.id);
                    writer.delete_term(term);
                    self.assign_ord(&mut tantivy_doc, &doc.id);
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
                let mut writer = self
                    .writer
                    .lock(cx)
                    .await
                    .map_err(|e| Self::map_writer_lock_error("tantivy.commit", e))?;

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

            // Persist the ordinal→doc_id table so a reopen keeps the fast path.
            self.persist_ord_table();

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

    #[test]
    fn reopen_on_disk_restores_fast_id_materialization() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().to_path_buf();
        run_with_cx(|cx| async move {
            let docs: Vec<IndexableDocument> = (0..20)
                .map(|i| {
                    IndexableDocument::new(
                        format!("doc-{i:03}"),
                        format!("alpha beta document number {i} searchable content"),
                    )
                })
                .collect();

            let ids_before = {
                let idx = TantivyIndex::create(&path).expect("create");
                idx.index_documents(&cx, &docs).await.expect("index");
                idx.commit(&cx).await.expect("commit");
                idx.search_doc_ids(&cx, "document", 20)
                    .expect("search")
                    .into_iter()
                    .map(|h| h.doc_id)
                    .collect::<Vec<_>>()
            }; // original index (and its writer lock) dropped here

            // Commit persisted the ordinal→doc_id sidecar.
            assert!(
                path.join("ord_table.json").exists(),
                "ord_table sidecar should be written on commit"
            );

            // Reopen: the sidecar restores the fast materialization path, and
            // results must be byte-identical to the pre-close ranking.
            let reopened = TantivyIndex::open(&path).expect("open");
            let ids_after = reopened
                .search_doc_ids(&cx, "document", 20)
                .expect("search")
                .into_iter()
                .map(|h| h.doc_id)
                .collect::<Vec<_>>();

            assert!(!ids_after.is_empty(), "reopened index should return hits");
            assert_eq!(
                ids_before, ids_after,
                "reopened index must return identical ranked doc_ids"
            );
        });
    }

    // ─── Indexing tests ─────────────────────────────────────────────────

    #[test]
    fn map_writer_lock_error_polled_after_completion_maps_to_subsystem_error() {
        let err = TantivyIndex::map_writer_lock_error(
            "tantivy.index",
            asupersync::sync::LockError::PolledAfterCompletion,
        );
        assert!(
            matches!(err, SearchError::SubsystemError { .. }),
            "expected subsystem error, got {err:?}"
        );
        if let SearchError::SubsystemError { source, .. } = err {
            assert!(
                source
                    .to_string()
                    .contains("writer mutex future reused after completion during tantivy.index")
            );
        }
    }

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
    fn deferred_fusion_candidates_restore_exact_metadata() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            let docs = sample_docs();
            idx.index_documents(&cx, &docs).await.expect("index");
            idx.commit(&cx).await.expect("commit");

            let full = idx.search(&cx, "Rust", 10).await.expect("full search");
            let mut candidates = idx
                .search_fusion_candidates(&cx, "Rust", 10)
                .await
                .expect("fusion candidates");

            assert!(idx.fusion_metadata_is_deferred());
            assert_eq!(candidates.len(), full.len());
            assert!(candidates.iter().all(|result| result.metadata.is_none()));
            for (candidate, expected) in candidates.iter().zip(&full) {
                assert_eq!(candidate.doc_id, expected.doc_id);
                assert_eq!(candidate.score.to_bits(), expected.score.to_bits());
                assert_eq!(
                    candidate.lexical_score.map(f32::to_bits),
                    expected.lexical_score.map(f32::to_bits)
                );
            }

            idx.hydrate_fusion_metadata(&cx, &mut candidates)
                .await
                .expect("hydrate winners");
            for (candidate, expected) in candidates.iter().zip(&full) {
                assert_eq!(candidate.metadata, expected.metadata);
            }
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

    #[test]
    fn zero_limit_returns_empty_without_collector_panic() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            idx.index_document(&cx, &IndexableDocument::new("doc-1", "searchable text"))
                .await
                .expect("index");
            idx.commit(&cx).await.expect("commit");

            let parsed = idx.parse_query_lenient("searchable");
            let searcher = idx.reader.searcher();
            let counted = execute_query_with_offset(&searcher, &*parsed, 0, 0)
                .expect("counted zero-limit search");
            assert!(counted.hits.is_empty());
            assert_eq!(counted.total_count, 1);
            assert!(
                execute_top_k(&searcher, &*parsed, 0, 0)
                    .expect("count-free zero-limit search")
                    .is_empty()
            );

            assert!(
                idx.search(&cx, "searchable", 0)
                    .await
                    .expect("search")
                    .is_empty()
            );
            assert!(
                idx.search_doc_ids(&cx, "searchable", 0)
                    .expect("id search")
                    .is_empty()
            );
            assert!(
                idx.search_with_snippets(&cx, "searchable", 0, &SnippetConfig::default())
                    .expect("snippet search")
                    .is_empty()
            );
        });
    }

    #[test]
    fn tantivy_indexing_enforces_max_token_len_boundary() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            let max = tantivy::tokenizer::MAX_TOKEN_LEN;
            let kept = "k".repeat(max);
            let dropped = "d".repeat(max + 1);
            idx.index_document(&cx, &IndexableDocument::new("kept", &kept))
                .await
                .expect("index boundary token");
            idx.index_document(&cx, &IndexableDocument::new("dropped", &dropped))
                .await
                .expect("index oversized token document");
            idx.commit(&cx).await.expect("commit");

            let searcher = idx.reader.searcher();
            let kept_query = TermQuery::new(
                Term::from_field_text(idx.fields.content, &kept),
                IndexRecordOption::WithFreqsAndPositions,
            );
            let dropped_query = TermQuery::new(
                Term::from_field_text(idx.fields.content, &dropped),
                IndexRecordOption::WithFreqsAndPositions,
            );
            assert_eq!(searcher.search(&kept_query, &Count).expect("search"), 1);
            assert_eq!(searcher.search(&dropped_query, &Count).expect("search"), 0);
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

    #[test]
    fn snippet_unicode_window_uses_tantivy_byte_offsets() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            idx.index_document(&cx, &IndexableDocument::new("unicode", "éé alpha"))
                .await
                .expect("index");
            idx.commit(&cx).await.expect("commit");

            let config = SnippetConfig {
                max_chars: 6,
                ..SnippetConfig::default()
            };
            let results = idx
                .search_with_snippets(&cx, "éé", 1, &config)
                .expect("snippet search");
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].snippet.as_deref(), Some("<b>éé</b>"));
        });
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
    fn search_doc_ids_returns_ranked_identifiers() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            let docs = sample_docs();
            idx.index_documents(&cx, &docs).await.expect("index");
            idx.commit(&cx).await.expect("commit");

            let results = idx.search_doc_ids(&cx, "Rust", 10).expect("search");
            assert!(!results.is_empty());
            for (expected_rank, hit) in results.iter().enumerate() {
                assert_eq!(hit.rank, expected_rank, "rank should be sequential");
                assert!(!hit.doc_id.is_empty());
                assert!(hit.bm25_score.is_finite());
            }
        });
    }

    #[test]
    fn all_doc_ids_enumerates_only_live_committed_documents() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            let docs = [
                IndexableDocument::new("doc-c", "gamma"),
                IndexableDocument::new("doc-a", "alpha"),
                IndexableDocument::new("doc-b", "beta"),
            ];
            idx.index_documents(&cx, &docs).await.expect("index");
            idx.commit(&cx).await.expect("commit");
            idx.delete_document(&cx, "doc-b").await.expect("delete");
            idx.commit(&cx).await.expect("commit delete");

            assert_eq!(
                idx.all_doc_ids().expect("enumerate ids"),
                vec![DocId::from("doc-a"), DocId::from("doc-c")]
            );
            let exported = idx.all_documents().expect("export documents");
            assert_eq!(
                exported
                    .iter()
                    .map(|document| document.id.as_str())
                    .collect::<Vec<_>>(),
                vec!["doc-a", "doc-c"]
            );
            assert_eq!(exported[0].content, "alpha");
            assert_eq!(exported[1].content, "gamma");
        });
    }

    #[test]
    fn search_doc_ids_matches_counted_baseline() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            let docs = sample_docs();
            idx.index_documents(&cx, &docs).await.expect("index");
            idx.commit(&cx).await.expect("commit");

            for query in ["Rust", "Rust search", "\"Rust programming\""] {
                let optimized = idx
                    .search_doc_ids(&cx, query, 10)
                    .expect("optimized search");
                let counted = idx
                    .search_doc_ids_counted(&cx, query, 10)
                    .expect("counted search");
                assert_eq!(optimized, counted, "query {query:?}");
            }
        });
    }

    #[test]
    fn count_free_doc_ids_only_accepts_one_or_two_plain_terms() {
        assert!(use_count_free_top_k("Rust"));
        assert!(use_count_free_top_k("rust_2026"));
        assert!(use_count_free_top_k("Rust search"));
        assert!(use_count_free_top_k("  Rust\tsearch\n"));

        for query in [
            "",
            "Rust search relevance",
            "\"Rust programming\"",
            "title:Rust",
            "Rust*",
            "+Rust",
            "doc-000001",
            "path/to/doc",
        ] {
            assert!(!use_count_free_top_k(query), "query {query:?}");
        }
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
        assert_eq!(truncated.chars().count(), MAX_QUERY_LENGTH);
    }

    #[test]
    fn truncate_query_counts_multibyte_characters() {
        let over = "\u{00E9}".repeat(MAX_QUERY_LENGTH + 3);
        let truncated = TantivyIndex::truncate_query(&over);
        assert!(truncated.is_char_boundary(truncated.len()));
        assert_eq!(truncated.chars().count(), MAX_QUERY_LENGTH);
        assert_eq!(truncated.len(), MAX_QUERY_LENGTH * '\u{00E9}'.len_utf8());
    }

    #[test]
    fn truncate_query_preserves_multibyte_query_within_character_limit() {
        let query = "\u{00E9}".repeat(MAX_QUERY_LENGTH / 2 + 1);
        assert!(query.len() > MAX_QUERY_LENGTH);
        assert_eq!(TantivyIndex::truncate_query(&query), query);
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
