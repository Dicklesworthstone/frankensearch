use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{OnceLock, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use frankensearch_core::error::{SearchError, SearchResult};
use tantivy::query::{
    AllQuery, BooleanQuery, Occur, PhraseQuery, Query, RangeQuery, RegexQuery, TermQuery,
};
use tantivy::schema::IndexRecordOption;
use tantivy::schema::{
    FAST, Field, INDEXED, STORED, STRING, Schema, TEXT, TextFieldIndexing, TextOptions,
};
use tantivy::tokenizer::{LowerCaser, RemoveLongFilter, SimpleTokenizer, TextAnalyzer};
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy, Term, doc};
use tracing::{debug, info, warn};

/// Schema version namespace used for cass-compatible Tantivy indexes.
pub const CASS_SCHEMA_VERSION: &str = "v6";
/// Content hash used to detect schema/tokenizer changes that require rebuild.
pub const CASS_SCHEMA_HASH: &str = "tantivy-schema-v6-long-tokens";

/// Minimum time (ms) between merge operations.
const MERGE_COOLDOWN_MS: i64 = 300_000;
/// Segment count threshold above which merge is triggered.
const MERGE_SEGMENT_THRESHOLD: usize = 4;

/// Global last merge timestamp (ms since epoch).
static LAST_MERGE_TS: AtomicI64 = AtomicI64::new(0);

const CASS_REGEX_QUERY_CACHE_CAP: usize = 128;
static CASS_REGEX_QUERY_CACHE: OnceLock<RwLock<HashMap<Field, HashMap<String, RegexQuery>>>> =
    OnceLock::new();

fn tantivy_err<E>(err: E) -> SearchError
where
    E: std::error::Error + Send + Sync + 'static,
{
    SearchError::SubsystemError {
        subsystem: "tantivy",
        source: Box::new(err),
    }
}

/// Build a Tantivy [`RegexQuery`] using a small global cache (cass compatibility).
///
/// Motivation: cass can generate many regex queries when handling `*foo` / `*foo*`
/// patterns and boolean combinations. Compiling Tantivy regex queries repeatedly
/// is expensive; caching improves interactive search latency significantly.
///
/// Cache behavior:
/// - Bounded in-memory cache (clears on overflow).
/// - Keyed by `(field, pattern)`.
/// - Thread-safe via `RwLock`.
///
/// # Errors
///
/// Returns [`SearchError::SubsystemError`] when Tantivy fails to compile the regex.
pub fn cass_regex_query_cached(field: Field, pattern: &str) -> SearchResult<RegexQuery> {
    let cache = CASS_REGEX_QUERY_CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    {
        let guard = cache
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(field_cache) = guard.get(&field)
            && let Some(q) = field_cache.get(pattern)
        {
            return Ok(q.clone());
        }
    }

    let query = RegexQuery::from_pattern(pattern, field).map_err(tantivy_err)?;
    let mut guard = cache
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let total_entries: usize = guard.values().map(HashMap::len).sum();
    if total_entries >= CASS_REGEX_QUERY_CACHE_CAP {
        guard.clear();
    }
    guard
        .entry(field)
        .or_default()
        .insert(pattern.to_string(), query.clone());
    drop(guard);
    Ok(query)
}

/// Build a Tantivy [`RegexQuery`] without caching (baseline for benchmarks/tests).
///
/// # Errors
///
/// Returns [`SearchError::SubsystemError`] when Tantivy fails to compile the regex.
pub fn cass_regex_query_uncached(field: Field, pattern: &str) -> SearchResult<RegexQuery> {
    RegexQuery::from_pattern(pattern, field).map_err(tantivy_err)
}

/// Returns true if the given stored hash matches the current schema hash.
#[must_use]
pub fn cass_schema_hash_matches(stored: &str) -> bool {
    stored == CASS_SCHEMA_HASH
}

/// Named fields used by cass-compatible query and indexing code.
#[derive(Clone, Copy, Debug)]
pub struct CassFields {
    pub agent: Field,
    pub workspace: Field,
    pub workspace_original: Field,
    pub source_path: Field,
    pub msg_idx: Field,
    pub created_at: Field,
    pub title: Field,
    pub content: Field,
    pub title_prefix: Field,
    pub content_prefix: Field,
    pub preview: Field,
    pub source_id: Field,
    pub origin_kind: Field,
    pub origin_host: Field,
}

/// Merge status for cass-compatible Tantivy segment optimization.
#[derive(Debug, Clone)]
pub struct CassMergeStatus {
    pub segment_count: usize,
    pub last_merge_ts: i64,
    pub ms_since_last_merge: i64,
    pub merge_threshold: usize,
    pub cooldown_ms: i64,
}

impl CassMergeStatus {
    #[must_use]
    pub const fn should_merge(&self) -> bool {
        self.segment_count >= self.merge_threshold
            && (self.ms_since_last_merge < 0 || self.ms_since_last_merge >= self.cooldown_ms)
    }
}

/// Cass-specific lexical document shape for index ingestion.
#[derive(Debug, Clone)]
pub struct CassDocument {
    pub agent: String,
    pub workspace: Option<String>,
    pub workspace_original: Option<String>,
    pub source_path: String,
    pub msg_idx: u64,
    pub created_at: Option<i64>,
    pub title: Option<String>,
    pub content: String,
    pub source_id: String,
    pub origin_kind: String,
    pub origin_host: Option<String>,
}

/// Tantivy index compatible with cass lexical schema and lifecycle.
pub struct CassTantivyIndex {
    index: Index,
    writer: IndexWriter,
    fields: CassFields,
}

impl CassTantivyIndex {
    /// Open existing index or create/rebuild as needed.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError`] if filesystem I/O, schema extraction, or Tantivy
    /// index creation/opening fails.
    pub fn open_or_create(path: &Path) -> SearchResult<Self> {
        std::fs::create_dir_all(path).map_err(tantivy_err)?;

        let meta_path = path.join("schema_hash.json");
        let needs_rebuild = if meta_path.exists()
            && let Ok(meta) = std::fs::read_to_string(&meta_path)
            && let Ok(json) = serde_json::from_str::<serde_json::Value>(&meta)
            && json.get("schema_hash").and_then(|v| v.as_str()) == Some(CASS_SCHEMA_HASH)
        {
            false
        } else {
            true
        };

        if needs_rebuild {
            let _ = std::fs::remove_dir_all(path);
            std::fs::create_dir_all(path).map_err(tantivy_err)?;
        }

        let mut index = if path.join("meta.json").exists() && !needs_rebuild {
            match Index::open_in_dir(path) {
                Ok(idx) => idx,
                Err(e) => {
                    warn!(
                        error = %e,
                        "failed to open existing cass-compatible index; rebuilding"
                    );
                    let _ = std::fs::remove_dir_all(path);
                    std::fs::create_dir_all(path).map_err(tantivy_err)?;
                    Index::create_in_dir(path, cass_build_schema()).map_err(tantivy_err)?
                }
            }
        } else {
            Index::create_in_dir(path, cass_build_schema()).map_err(tantivy_err)?
        };

        cass_ensure_tokenizer(&mut index);
        std::fs::write(
            &meta_path,
            format!("{{\"schema_hash\":\"{CASS_SCHEMA_HASH}\"}}"),
        )
        .map_err(tantivy_err)?;

        let actual_schema = index.schema();
        let writer = index.writer(50_000_000).map_err(tantivy_err)?;
        let fields = cass_fields_from_schema(&actual_schema)?;
        Ok(Self {
            index,
            writer,
            fields,
        })
    }

    #[must_use]
    pub const fn fields(&self) -> CassFields {
        self.fields
    }

    /// Open an [`IndexReader`] for this index.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::SubsystemError`] when Tantivy reader construction fails.
    pub fn reader(&self) -> SearchResult<IndexReader> {
        self.index.reader().map_err(tantivy_err)
    }

    /// Delete all indexed documents.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::SubsystemError`] if Tantivy delete fails.
    pub fn delete_all(&mut self) -> SearchResult<()> {
        self.writer.delete_all_documents().map_err(tantivy_err)?;
        Ok(())
    }

    /// Commit all pending writer operations.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::SubsystemError`] if Tantivy commit fails.
    pub fn commit(&mut self) -> SearchResult<()> {
        self.writer.commit().map_err(tantivy_err)?;
        Ok(())
    }

    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.index
            .searchable_segment_ids()
            .map_or(0, |ids| ids.len())
    }

    #[must_use]
    pub fn merge_status(&self) -> CassMergeStatus {
        let last_merge_ts = LAST_MERGE_TS.load(Ordering::Relaxed);
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX));
        let ms_since_last = if last_merge_ts > 0 {
            now_ms - last_merge_ts
        } else {
            -1
        };
        CassMergeStatus {
            segment_count: self.segment_count(),
            last_merge_ts,
            ms_since_last_merge: ms_since_last,
            merge_threshold: MERGE_SEGMENT_THRESHOLD,
            cooldown_ms: MERGE_COOLDOWN_MS,
        }
    }

    /// Trigger async merge when threshold/cooldown permit.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::SubsystemError`] if segment enumeration fails.
    pub fn optimize_if_idle(&mut self) -> SearchResult<bool> {
        let segment_ids = self.index.searchable_segment_ids().map_err(tantivy_err)?;
        let segment_count = segment_ids.len();
        if segment_count < MERGE_SEGMENT_THRESHOLD {
            debug!(
                segments = segment_count,
                threshold = MERGE_SEGMENT_THRESHOLD,
                "skipping merge: below threshold"
            );
            return Ok(false);
        }

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX));
        let last_merge = LAST_MERGE_TS.load(Ordering::Relaxed);
        if last_merge > 0 && (now_ms - last_merge) < MERGE_COOLDOWN_MS {
            debug!(
                ms_since_last = now_ms - last_merge,
                cooldown = MERGE_COOLDOWN_MS,
                "skipping merge: cooldown active"
            );
            return Ok(false);
        }

        info!(
            segments = segment_count,
            "starting cass-compatible segment merge"
        );
        let _merge_future = self.writer.merge(&segment_ids);
        LAST_MERGE_TS.store(now_ms, Ordering::Relaxed);
        Ok(true)
    }

    /// Force immediate merge and block until completion.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::SubsystemError`] if segment enumeration or merge fails.
    pub fn force_merge(&mut self) -> SearchResult<()> {
        let segment_ids = self.index.searchable_segment_ids().map_err(tantivy_err)?;
        if segment_ids.is_empty() {
            return Ok(());
        }
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX));

        let merge_future = self.writer.merge(&segment_ids);
        match merge_future.wait() {
            Ok(_) => {
                LAST_MERGE_TS.store(now_ms, Ordering::Relaxed);
                Ok(())
            }
            Err(err) => Err(tantivy_err(err)),
        }
    }

    /// Add a batch of cass-compatible documents.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::SubsystemError`] if adding a document to Tantivy fails.
    pub fn add_cass_documents(&mut self, docs: &[CassDocument]) -> SearchResult<()> {
        for cass_doc in docs {
            let mut d = doc! {
                self.fields.agent => cass_doc.agent.clone(),
                self.fields.source_path => cass_doc.source_path.clone(),
                self.fields.msg_idx => cass_doc.msg_idx,
                self.fields.content => cass_doc.content.clone(),
                self.fields.source_id => cass_doc.source_id.clone(),
                self.fields.origin_kind => cass_doc.origin_kind.clone(),
            };

            if let Some(host) = &cass_doc.origin_host
                && !host.is_empty()
            {
                d.add_text(self.fields.origin_host, host);
            }
            if let Some(workspace) = &cass_doc.workspace {
                d.add_text(self.fields.workspace, workspace);
            }
            if let Some(workspace_original) = &cass_doc.workspace_original {
                d.add_text(self.fields.workspace_original, workspace_original);
            }
            if let Some(ts) = cass_doc.created_at {
                d.add_i64(self.fields.created_at, ts);
            }
            if let Some(title) = &cass_doc.title {
                d.add_text(self.fields.title, title);
                d.add_text(self.fields.title_prefix, cass_generate_edge_ngrams(title));
            }
            d.add_text(
                self.fields.content_prefix,
                cass_generate_edge_ngrams(&cass_doc.content),
            );
            d.add_text(
                self.fields.preview,
                cass_build_preview(&cass_doc.content, 400),
            );
            self.writer.add_document(d).map_err(tantivy_err)?;
        }
        Ok(())
    }
}

/// Build cass-compatible Tantivy schema.
#[must_use]
pub fn cass_build_schema() -> Schema {
    let mut schema_builder = Schema::builder();
    let text = TextOptions::default()
        .set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer("hyphen_normalize")
                .set_index_option(tantivy::schema::IndexRecordOption::WithFreqsAndPositions),
        )
        .set_stored();

    let text_not_stored = TextOptions::default().set_indexing_options(
        TextFieldIndexing::default()
            .set_tokenizer("hyphen_normalize")
            .set_index_option(tantivy::schema::IndexRecordOption::WithFreqsAndPositions),
    );

    schema_builder.add_text_field("agent", STRING | STORED);
    schema_builder.add_text_field("workspace", STRING | STORED);
    schema_builder.add_text_field("workspace_original", STORED);
    schema_builder.add_text_field("source_path", STORED);
    schema_builder.add_u64_field("msg_idx", INDEXED | STORED);
    schema_builder.add_i64_field("created_at", INDEXED | STORED | FAST);
    schema_builder.add_text_field("title", text.clone());
    schema_builder.add_text_field("content", text);
    schema_builder.add_text_field("title_prefix", text_not_stored.clone());
    schema_builder.add_text_field("content_prefix", text_not_stored);
    schema_builder.add_text_field("preview", TEXT | STORED);
    schema_builder.add_text_field("source_id", STRING | STORED);
    schema_builder.add_text_field("origin_kind", STRING | STORED);
    schema_builder.add_text_field("origin_host", STRING | STORED);
    schema_builder.build()
}

/// Extract cass-compatible schema fields from a Tantivy schema handle.
///
/// # Errors
///
/// Returns [`SearchError::InvalidConfig`] when required fields are missing.
pub fn cass_fields_from_schema(schema: &Schema) -> SearchResult<CassFields> {
    let get = |name: &str| {
        schema
            .get_field(name)
            .map_err(|_| SearchError::InvalidConfig {
                field: "schema".to_string(),
                value: name.to_string(),
                reason: format!("schema missing required field `{name}`"),
            })
    };

    Ok(CassFields {
        agent: get("agent")?,
        workspace: get("workspace")?,
        workspace_original: get("workspace_original")?,
        source_path: get("source_path")?,
        msg_idx: get("msg_idx")?,
        created_at: get("created_at")?,
        title: get("title")?,
        content: get("content")?,
        title_prefix: get("title_prefix")?,
        content_prefix: get("content_prefix")?,
        preview: get("preview")?,
        source_id: get("source_id")?,
        origin_kind: get("origin_kind")?,
        origin_host: get("origin_host")?,
    })
}

/// Open a cass-compatible search reader (no writer) with caller-specified reload policy.
///
/// This centralizes tokenizer registration + field extraction, mirroring the expectations of
/// cass query execution code while keeping Tantivy lifecycle ownership inside frankensearch.
///
/// # Errors
///
/// Returns [`SearchError::SubsystemError`] if Tantivy open/build operations fail, or
/// [`SearchError::InvalidConfig`] if schema fields are incomplete.
pub fn cass_open_search_reader(
    index_path: &Path,
    reload_policy: ReloadPolicy,
) -> SearchResult<(IndexReader, CassFields)> {
    let mut index = Index::open_in_dir(index_path).map_err(tantivy_err)?;
    cass_ensure_tokenizer(&mut index);
    let schema = index.schema();
    let fields = cass_fields_from_schema(&schema)?;
    let reader = index
        .reader_builder()
        .reload_policy(reload_policy)
        .try_into()
        .map_err(tantivy_err)?;
    let _ = reader.reload();
    Ok((reader, fields))
}

/// Resolve cass-compatible index directory under a data root.
///
/// # Errors
///
/// Returns [`SearchError::SubsystemError`] if the directory cannot be created.
pub fn cass_index_dir(base: &Path) -> SearchResult<PathBuf> {
    let dir = base.join("index").join(CASS_SCHEMA_VERSION);
    std::fs::create_dir_all(&dir).map_err(tantivy_err)?;
    Ok(dir)
}

/// Register the tokenizer used by cass-compatible lexical fields.
pub fn cass_ensure_tokenizer(index: &mut Index) {
    let analyzer = TextAnalyzer::builder(SimpleTokenizer::default())
        .filter(LowerCaser)
        .filter(RemoveLongFilter::limit(256))
        .build();
    index.tokenizers().register("hyphen_normalize", analyzer);
}

/// Generate edge n-grams from text for prefix search acceleration.
#[must_use]
pub fn cass_generate_edge_ngrams(text: &str) -> String {
    const MAX_NGRAM_INDICES: usize = 21;
    let mut ngrams = String::with_capacity(text.len() * 2);
    for word in text.split(|c: char| !c.is_alphanumeric()) {
        let indices: Vec<usize> = word
            .char_indices()
            .map(|(i, _)| i)
            .chain(std::iter::once(word.len()))
            .take(MAX_NGRAM_INDICES)
            .collect();

        if indices.len() < 3 {
            continue;
        }
        for &end_idx in &indices[2..] {
            if !ngrams.is_empty() {
                ngrams.push(' ');
            }
            ngrams.push_str(&word[..end_idx]);
        }
    }
    ngrams
}

/// Build a bounded-length preview from message content.
#[must_use]
pub fn cass_build_preview(content: &str, max_chars: usize) -> String {
    let mut out = String::new();
    let mut chars = content.chars();
    for _ in 0..max_chars {
        if let Some(ch) = chars.next() {
            out.push(ch);
        } else {
            return out;
        }
    }
    if chars.next().is_some() {
        out.push('…');
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Cass Lexical Query Builder
// ─────────────────────────────────────────────────────────────────────────────

/// Source filter options used by cass-compatible lexical queries.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum CassSourceFilter {
    /// No source filtering.
    #[default]
    All,
    /// Local-only (`origin_kind` == "local").
    Local,
    /// Remote-only (`origin_kind` == "ssh").
    Remote,
    /// Specific source id (`source_id` == <id>).
    SourceId(String),
}

/// Cass-compatible lexical filters applied directly in Tantivy.
#[derive(Debug, Clone, Default)]
pub struct CassQueryFilters {
    pub agents: Vec<String>,
    pub workspaces: Vec<String>,
    pub created_from: Option<i64>,
    pub created_to: Option<i64>,
    pub source_filter: CassSourceFilter,
}

/// Token types for cass-style boolean query parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CassQueryToken {
    /// A search term (may include wildcards).
    Term(String),
    /// A quoted phrase for exact-order matching.
    Phrase(String),
    /// Explicit AND operator.
    And,
    /// OR operator.
    Or,
    /// NOT operator (negates the next term/phrase).
    Not,
}

/// Sanitize query string to match Tantivy's `SimpleTokenizer` behavior for cass indexes.
///
/// `SimpleTokenizer` splits text on any character that is not alphanumeric.
/// To ensure query terms match indexed tokens, we replace non-alphanumeric characters
/// with spaces here (preserving `*` for wildcards and `\"` for phrases).
#[must_use]
pub fn cass_sanitize_query(raw: &str) -> String {
    raw.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '*' || c == '"' {
                c
            } else {
                ' '
            }
        })
        .collect()
}

#[must_use]
fn cass_escape_regex(s: &str) -> String {
    let mut escaped = String::with_capacity(s.len() * 2);
    for c in s.chars() {
        match c {
            '\\' | '.' | '+' | '*' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '^' | '$' => {
                escaped.push('\\');
                escaped.push(c);
            }
            _ => escaped.push(c),
        }
    }
    escaped
}

/// Represents different wildcard patterns for a cass lexical search term.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CassWildcardPattern {
    Exact(String),
    Prefix(String),
    Suffix(String),
    Substring(String),
    Complex(String),
}

impl CassWildcardPattern {
    #[must_use]
    pub fn parse(term: &str) -> Self {
        let starts_with_star = term.starts_with('*');
        let ends_with_star = term.ends_with('*');

        let core = term.trim_matches('*').to_lowercase();
        if core.is_empty() {
            return Self::Exact(String::new());
        }

        // Internal wildcards (e.g. f*o) -> complex pattern.
        if core.contains('*') {
            return Self::Complex(term.to_lowercase());
        }

        match (starts_with_star, ends_with_star) {
            (true, true) => Self::Substring(core),
            (true, false) => Self::Suffix(core),
            (false, true) => Self::Prefix(core),
            (false, false) => Self::Exact(core),
        }
    }

    #[must_use]
    pub fn to_regex(&self) -> Option<String> {
        match self {
            Self::Suffix(core) => Some(format!(".*{}$", cass_escape_regex(core))),
            Self::Substring(core) => Some(format!(".*{}.*", cass_escape_regex(core))),
            Self::Complex(full_term) => {
                let mut regex = String::with_capacity(full_term.len() * 2 + 2);

                if full_term.starts_with('*') {
                    regex.push_str(".*");
                } else {
                    regex.push('^');
                }

                let trimmed_start = full_term.trim_start_matches('*');
                let trimmed = trimmed_start.trim_end_matches('*');
                for c in trimmed.chars() {
                    if c == '*' {
                        regex.push_str(".*");
                    } else {
                        match c {
                            '\\' | '.' | '+' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|'
                            | '^' | '$' => {
                                regex.push('\\');
                                regex.push(c);
                            }
                            _ => regex.push(c),
                        }
                    }
                }

                if full_term.ends_with('*') {
                    regex.push_str(".*");
                } else {
                    regex.push('$');
                }

                Some(regex)
            }
            _ => None,
        }
    }
}

/// Parse a query string into boolean tokens.
///
/// Supports:
/// - AND / && (explicit AND; implicit AND between terms is handled by query construction)
/// - OR / || (OR)
/// - NOT / -prefix (negation)
/// - \"quoted phrases\" (phrase match)
#[must_use]
pub fn cass_parse_boolean_query(query: &str) -> Vec<CassQueryToken> {
    let mut tokens = Vec::new();
    let mut chars = query.chars().peekable();
    let mut current_word = String::new();

    while let Some(c) = chars.next() {
        match c {
            '"' => {
                if !current_word.is_empty() {
                    tokens.push(CassQueryToken::Term(std::mem::take(&mut current_word)));
                }
                let mut phrase = String::new();
                while let Some(&next) = chars.peek() {
                    if next == '"' {
                        chars.next();
                        break;
                    }
                    if let Some(c) = chars.next() {
                        phrase.push(c);
                    }
                }
                if !phrase.is_empty() {
                    tokens.push(CassQueryToken::Phrase(phrase));
                }
            }
            '&' if chars.peek() == Some(&'&') => {
                chars.next();
                if !current_word.is_empty() {
                    tokens.push(CassQueryToken::Term(std::mem::take(&mut current_word)));
                }
                tokens.push(CassQueryToken::And);
            }
            '|' if chars.peek() == Some(&'|') => {
                chars.next();
                if !current_word.is_empty() {
                    tokens.push(CassQueryToken::Term(std::mem::take(&mut current_word)));
                }
                tokens.push(CassQueryToken::Or);
            }
            '-' if current_word.is_empty() => {
                tokens.push(CassQueryToken::Not);
            }
            ' ' | '\t' | '\n' => {
                if !current_word.is_empty() {
                    let word = std::mem::take(&mut current_word);
                    let upper = word.to_ascii_uppercase();
                    match upper.as_str() {
                        "AND" => tokens.push(CassQueryToken::And),
                        "OR" => tokens.push(CassQueryToken::Or),
                        "NOT" => tokens.push(CassQueryToken::Not),
                        _ => tokens.push(CassQueryToken::Term(word)),
                    }
                }
            }
            _ => current_word.push(c),
        }
    }

    if !current_word.is_empty() {
        let upper = current_word.to_ascii_uppercase();
        match upper.as_str() {
            "AND" => tokens.push(CassQueryToken::And),
            "OR" => tokens.push(CassQueryToken::Or),
            "NOT" => tokens.push(CassQueryToken::Not),
            _ => tokens.push(CassQueryToken::Term(current_word)),
        }
    }

    tokens
}

#[must_use]
pub fn cass_has_boolean_operators(query: &str) -> bool {
    let tokens = cass_parse_boolean_query(query);
    tokens.iter().any(|t| {
        matches!(
            t,
            CassQueryToken::And
                | CassQueryToken::Or
                | CassQueryToken::Not
                | CassQueryToken::Phrase(_)
        )
    })
}

/// Normalize a term into tokenizer-aligned parts (preserving `*` for wildcards).
#[must_use]
fn cass_normalize_term_parts(raw: &str) -> Vec<String> {
    cass_sanitize_query(raw)
        .split_whitespace()
        .map(str::to_owned)
        .collect()
}

/// Normalize phrase text into tokenizer-aligned terms (lowercased, no wildcards).
#[must_use]
fn cass_normalize_phrase_terms(raw: &str) -> Vec<String> {
    cass_sanitize_query(raw)
        .split_whitespace()
        .map(|s| s.trim_matches('*').to_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}

fn cass_flush_pending_or_group(
    pending_or_group: &mut Vec<Box<dyn Query>>,
    clauses: &mut Vec<(Occur, Box<dyn Query>)>,
) {
    if pending_or_group.is_empty() {
        return;
    }
    let or_clauses: Vec<_> = std::mem::take(pending_or_group)
        .into_iter()
        .map(|query| (Occur::Should, query))
        .collect();
    clauses.push((Occur::Must, Box::new(BooleanQuery::new(or_clauses))));
}

fn cass_lift_must_clause_into_or_group(
    clauses: &mut Vec<(Occur, Box<dyn Query>)>,
    pending_or_group: &mut Vec<Box<dyn Query>>,
) {
    let can_pull = clauses
        .last()
        .is_some_and(|(occ, _)| *occ == Occur::Must || *occ == Occur::MustNot);
    if !can_pull {
        return;
    }

    if let Some((occur, last_query)) = clauses.pop() {
        let lifted_query = if occur == Occur::MustNot {
            Box::new(BooleanQuery::new(vec![
                (Occur::Must, Box::new(AllQuery)),
                (Occur::MustNot, last_query),
            ]))
        } else {
            last_query
        };
        pending_or_group.push(lifted_query);
    }
}

fn cass_wrap_negated_clause(query: Box<dyn Query>) -> Box<dyn Query> {
    Box::new(BooleanQuery::new(vec![
        (Occur::Must, Box::new(AllQuery)),
        (Occur::MustNot, query),
    ]))
}

fn cass_apply_query_token(
    query: Box<dyn Query>,
    next_occur: Occur,
    in_or_sequence: &mut bool,
    just_saw_or: &mut bool,
    pending_or_group: &mut Vec<Box<dyn Query>>,
    clauses: &mut Vec<(Occur, Box<dyn Query>)>,
) {
    if *in_or_sequence && *just_saw_or {
        if pending_or_group.is_empty() {
            cass_lift_must_clause_into_or_group(clauses, pending_or_group);
        }
        let pushed_query = if next_occur == Occur::MustNot {
            cass_wrap_negated_clause(query)
        } else {
            query
        };
        pending_or_group.push(pushed_query);
    } else {
        cass_flush_pending_or_group(pending_or_group, clauses);
        *in_or_sequence = false;
        clauses.push((next_occur, query));
    }

    *just_saw_or = false;
}

/// Build query clauses for a single term based on its wildcard pattern.
fn cass_build_term_query_clauses(
    pattern: &CassWildcardPattern,
    fields: &CassFields,
) -> Vec<(Occur, Box<dyn Query>)> {
    let mut shoulds: Vec<(Occur, Box<dyn Query>)> = Vec::new();

    match pattern {
        CassWildcardPattern::Exact(term) | CassWildcardPattern::Prefix(term) => {
            if term.is_empty() {
                return shoulds;
            }
            for field in [
                fields.title,
                fields.content,
                fields.title_prefix,
                fields.content_prefix,
            ] {
                shoulds.push((
                    Occur::Should,
                    Box::new(TermQuery::new(
                        Term::from_field_text(field, term),
                        IndexRecordOption::WithFreqsAndPositions,
                    )),
                ));
            }
        }
        CassWildcardPattern::Suffix(_)
        | CassWildcardPattern::Substring(_)
        | CassWildcardPattern::Complex(_) => {
            if let Some(regex_pattern) = pattern.to_regex() {
                if let Ok(rq) = cass_regex_query_cached(fields.content, &regex_pattern) {
                    shoulds.push((Occur::Should, Box::new(rq)));
                }
                if let Ok(rq) = cass_regex_query_cached(fields.title, &regex_pattern) {
                    shoulds.push((Occur::Should, Box::new(rq)));
                }
            }
        }
    }

    shoulds
}

/// Build a compound query that requires all term parts to match (implicit AND).
fn cass_build_compound_term_query(parts: &[String], fields: &CassFields) -> Option<Box<dyn Query>> {
    let mut subqueries: Vec<Box<dyn Query>> = Vec::new();
    for part in parts {
        let pattern = CassWildcardPattern::parse(part);
        let term_shoulds = cass_build_term_query_clauses(&pattern, fields);
        if !term_shoulds.is_empty() {
            subqueries.push(Box::new(BooleanQuery::new(term_shoulds)));
        }
    }

    match subqueries.len() {
        0 => None,
        1 => subqueries.pop(),
        _ => {
            let musts = subqueries.into_iter().map(|q| (Occur::Must, q)).collect();
            Some(Box::new(BooleanQuery::new(musts)))
        }
    }
}

/// Build a phrase query (exact order) across title/content fields.
fn cass_build_phrase_query(terms: &[String], fields: &CassFields) -> Option<Box<dyn Query>> {
    if terms.is_empty() {
        return None;
    }
    if terms.len() == 1 {
        return cass_build_compound_term_query(terms, fields);
    }

    let mut shoulds: Vec<(Occur, Box<dyn Query>)> = Vec::new();
    for field in [fields.title, fields.content] {
        let phrase_terms = terms
            .iter()
            .map(|t| Term::from_field_text(field, t))
            .collect::<Vec<_>>();
        shoulds.push((Occur::Should, Box::new(PhraseQuery::new(phrase_terms))));
    }
    Some(Box::new(BooleanQuery::new(shoulds)))
}

/// Build Tantivy query clauses from boolean tokens.
///
/// Operator precedence is intentionally non-standard: `OR` binds tighter than `AND`.
fn cass_build_boolean_query_clauses(
    tokens: &[CassQueryToken],
    fields: &CassFields,
) -> Vec<(Occur, Box<dyn Query>)> {
    let mut clauses: Vec<(Occur, Box<dyn Query>)> = Vec::new();
    let mut pending_or_group: Vec<Box<dyn Query>> = Vec::new();
    let mut next_occur = Occur::Must;
    let mut in_or_sequence = false;
    let mut just_saw_or = false;

    for token in tokens {
        match token {
            CassQueryToken::And => {
                cass_flush_pending_or_group(&mut pending_or_group, &mut clauses);
                in_or_sequence = false;
                just_saw_or = false;
                next_occur = Occur::Must;
            }
            CassQueryToken::Or => {
                in_or_sequence = true;
                just_saw_or = true;
            }
            CassQueryToken::Not => {
                if just_saw_or {
                    just_saw_or = true;
                } else {
                    cass_flush_pending_or_group(&mut pending_or_group, &mut clauses);
                    in_or_sequence = false;
                    just_saw_or = false;
                }
                next_occur = Occur::MustNot;
            }
            CassQueryToken::Term(term) => {
                let parts = cass_normalize_term_parts(term);
                let term_query = cass_build_compound_term_query(&parts, fields);
                let Some(term_query) = term_query else {
                    continue;
                };
                cass_apply_query_token(
                    term_query,
                    next_occur,
                    &mut in_or_sequence,
                    &mut just_saw_or,
                    &mut pending_or_group,
                    &mut clauses,
                );
                next_occur = Occur::Must;
            }
            CassQueryToken::Phrase(phrase) => {
                let terms = cass_normalize_phrase_terms(phrase);
                let phrase_query = cass_build_phrase_query(&terms, fields);
                let Some(phrase_query) = phrase_query else {
                    continue;
                };
                cass_apply_query_token(
                    phrase_query,
                    next_occur,
                    &mut in_or_sequence,
                    &mut just_saw_or,
                    &mut pending_or_group,
                    &mut clauses,
                );
                next_occur = Occur::Must;
            }
        }
    }

    cass_flush_pending_or_group(&mut pending_or_group, &mut clauses);

    clauses
}

/// Build a cass-compatible Tantivy query for `raw_query`, applying filters.
///
/// Returns an `AllQuery` when the query is empty.
#[must_use]
pub fn cass_build_tantivy_query(
    raw_query: &str,
    filters: &CassQueryFilters,
    fields: &CassFields,
) -> Box<dyn Query> {
    let mut clauses: Vec<(Occur, Box<dyn Query>)> = Vec::new();

    let tokens = cass_parse_boolean_query(raw_query);
    if tokens.is_empty() {
        clauses.push((Occur::Must, Box::new(AllQuery)));
    } else if cass_has_boolean_operators(raw_query) {
        clauses.extend(cass_build_boolean_query_clauses(&tokens, fields));
    } else {
        for token in tokens {
            if let CassQueryToken::Term(term_str) = token {
                let parts = cass_normalize_term_parts(&term_str);
                if let Some(term_query) = cass_build_compound_term_query(&parts, fields) {
                    clauses.push((Occur::Must, term_query));
                }
            }
        }
    }

    if !filters.agents.is_empty() {
        let terms = filters
            .agents
            .iter()
            .map(|agent| {
                (
                    Occur::Should,
                    Box::new(TermQuery::new(
                        Term::from_field_text(fields.agent, agent),
                        IndexRecordOption::Basic,
                    )) as Box<dyn Query>,
                )
            })
            .collect();
        clauses.push((Occur::Must, Box::new(BooleanQuery::new(terms))));
    }

    if !filters.workspaces.is_empty() {
        let terms = filters
            .workspaces
            .iter()
            .map(|ws| {
                (
                    Occur::Should,
                    Box::new(TermQuery::new(
                        Term::from_field_text(fields.workspace, ws),
                        IndexRecordOption::Basic,
                    )) as Box<dyn Query>,
                )
            })
            .collect();
        clauses.push((Occur::Must, Box::new(BooleanQuery::new(terms))));
    }

    if filters.created_from.is_some() || filters.created_to.is_some() {
        use std::ops::Bound::{Included, Unbounded};
        let lower = filters.created_from.map_or(Unbounded, |v| {
            Included(Term::from_field_i64(fields.created_at, v))
        });
        let upper = filters.created_to.map_or(Unbounded, |v| {
            Included(Term::from_field_i64(fields.created_at, v))
        });
        let range = RangeQuery::new(lower, upper);
        clauses.push((Occur::Must, Box::new(range)));
    }

    match &filters.source_filter {
        CassSourceFilter::All => {}
        CassSourceFilter::Local => {
            let term = Term::from_field_text(fields.origin_kind, "local");
            clauses.push((
                Occur::Must,
                Box::new(TermQuery::new(term, IndexRecordOption::Basic)),
            ));
        }
        CassSourceFilter::Remote => {
            let term = Term::from_field_text(fields.origin_kind, "ssh");
            clauses.push((
                Occur::Must,
                Box::new(TermQuery::new(term, IndexRecordOption::Basic)),
            ));
        }
        CassSourceFilter::SourceId(source_id) => {
            let term = Term::from_field_text(fields.source_id, source_id);
            clauses.push((
                Occur::Must,
                Box::new(TermQuery::new(term, IndexRecordOption::Basic)),
            ));
        }
    }

    match clauses.len() {
        0 => Box::new(AllQuery),
        1 => {
            if let Some((occur, query_box)) = clauses.pop() {
                match occur {
                    Occur::Must => query_box,
                    _ => Box::new(BooleanQuery::new(vec![(occur, query_box)])),
                }
            } else {
                Box::new(AllQuery)
            }
        }
        _ => Box::new(BooleanQuery::new(clauses)),
    }
}

#[cfg(test)]
mod cass_query_tests {
    use super::*;

    fn fields() -> CassFields {
        let schema = cass_build_schema();
        cass_fields_from_schema(&schema).expect("cass fields")
    }

    #[test]
    fn cass_sanitize_query_preserves_wildcards_and_quotes() {
        let out = cass_sanitize_query("c++ \"hello-world\" *config*");
        assert!(out.contains('"'));
        assert!(out.contains('*'));
        assert!(out.contains("hello world"));
    }

    #[test]
    fn cass_build_query_empty_returns_allquery() {
        let f = fields();
        let q = cass_build_tantivy_query("", &CassQueryFilters::default(), &f);
        assert!(format!("{q:?}").to_ascii_lowercase().contains("allquery"));
    }

    #[test]
    fn cass_build_query_applies_agent_filter() {
        let f = fields();
        let filters = CassQueryFilters {
            agents: vec!["claude".to_string(), "codex".to_string()],
            ..CassQueryFilters::default()
        };
        let q = cass_build_tantivy_query("auth", &filters, &f);
        let dbg = format!("{q:?}");
        assert!(
            dbg.contains("BooleanQuery"),
            "expected boolean query: {dbg}"
        );
    }
}
