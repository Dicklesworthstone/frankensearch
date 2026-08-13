//! Tantivy BM25 full-text search integration for frankensearch.
//!
//! Provides the [`TantivyIndex`] implementation of the [`LexicalSearch`] trait,
//! including schema creation, document indexing, BM25 query parsing,
//! and search result ranking.
//!
//! The crate version constant is part of gauntlet dependency provenance: it
//! identifies the concrete lexical wrapper compiled around Tantivy.
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

/// Exact `frankensearch-lexical` crate version compiled into this adapter.
pub const FRANKENSEARCH_LEXICAL_CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

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
// Consumers can import these from `frankensearch::lexical_tantivy::` instead
// of adding a direct Tantivy dependency. The explicit namespace keeps
// foreign-format and oracle callers stable when the facade's generic
// `lexical` feature selects Quill.
pub use tantivy::collector::{Count, TopDocs};
pub use tantivy::query::{BooleanQuery, Occur, Query, TermQuery};
pub use tantivy::schema::{Field, IndexRecordOption, Schema, Value};
pub use tantivy::{
    self as tantivy_crate, DocAddress, Index, IndexReader, IndexWriter, ReloadPolicy, Searcher,
    TantivyDocument, Term,
};

use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::sync::RwLock;
#[cfg(feature = "bench-internals")]
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicUsize, Ordering};

use asupersync::Cx;
use asupersync::sync::Mutex;
use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::traits::{LexicalCandidateBatch, LexicalHydrationContext, SearchFuture};
use frankensearch_core::types::{DocId, IndexableDocument, ScoreSource, ScoredResult};
use serde::{Deserialize, Serialize};
use tantivy::collector::DocSetCollector;
use tantivy::query::QueryParser;
use tantivy::schema::{FAST, STORED, STRING, TextFieldIndexing, TextOptions};
use tantivy::tokenizer::{TextAnalyzer, Token, TokenStream, Tokenizer};
use tracing::{debug, instrument, warn};

// ─── Constants ──────────────────────────────────────────────────────────────

/// Name for the custom tokenizer registered with the Tantivy index.
const TOKENIZER_NAME: &str = "frankensearch_default";

/// Default heap size for the Tantivy `IndexWriter` (50 MB).
const WRITER_HEAP_BYTES: usize = 50_000_000;

/// Process-local issuer for opaque benchmark writer attestations.
///
/// A construction ID proves only that a distinct writer was constructed during
/// this process. It is deliberately not a persisted or cross-process identity.
#[cfg(feature = "bench-internals")]
static NEXT_BENCHMARK_WRITER_CONSTRUCTION_ID: AtomicU64 = AtomicU64::new(1);

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
/// Strip the boost from any parenthesized group that contains a negation
/// (bd-f20ye), for the SHIPPING search path only.
///
/// # Why a boost has to be dropped rather than moved
///
/// Tantivy 0.26.1 lowers `(A NOT B)` as a `MustNot` clause OF the enclosing
/// boolean, which excludes, but lowers `(A NOT B)^2` by nesting the negation in
/// its own `BooleanQuery { [(MustNot, …)], msm: 0 }` and attaching THAT as a
/// clause of the outer boolean. A matcher meaning "every document except B"
/// then becomes a positive alternative, so the group stops excluding:
/// `(A NOT B)^2` returns the documents B was supposed to remove, and
/// `(A AND NOT B)^2` fails the other way and returns nothing.
///
/// The repair cannot be done on the parsed tree: `BoostQuery`'s fields are
/// private and it exposes no accessors, so the boolean inside a boost is
/// unreachable. It therefore happens on the query text, before parsing.
///
/// Dropping the boost is exact for MEMBERSHIP, which is the property at stake,
/// and lossy for SCORING: a `MustNot` clause contributes no score, so the boost
/// only ever scaled the positive operands, and those keep their relative order
/// without it. Redistributing the factor onto each positive operand would
/// preserve scores too, but requires splitting arbitrary group syntax by hand —
/// more ways to be wrong, for an ordering nicety, on a shape Tantivy cannot
/// currently express correctly at all.
///
/// Quoted phrases are respected, so a literal `"a NOT b"` never triggers it.
/// Rewrite `A AND NOT B` to `A NOT B` for the SHIPPING path only (bd-eeq0q).
///
/// THE DEFECT, measured on tantivy 0.26.1 with p1="alpha beta", p2="alpha gamma":
///
/// ```text
///   alpha NOT beta            [p2]   correct
///   alpha -beta               [p2]   correct
///   alpha AND NOT beta        []     WRONG — p2 has alpha and no beta
///   (alpha AND NOT beta)      []     WRONG
///   (alpha AND NOT beta)^2    []     WRONG, and untouched by the bd-f20ye repair
/// ```
///
/// `A AND NOT B` lowers to
/// `Bool{[(Must, A), (Must, Bool{[(MustNot, B)], msm: 0})], msm: 0}`. The second
/// `Must` operand is a boolean with only a `MustNot` clause and no positive
/// clause, so it matches nothing and the whole conjunction is empty. The same
/// engine returns the right answer for `A NOT B` and `A -B`, so this is an
/// inconsistency inside one query language rather than a defensible reading.
///
/// The repair deletes the redundant `AND` in front of a `NOT`, which is exactly
/// the spelling the engine already handles. It is textual for the same reason
/// [`repair_boosted_group_negation`] is: the lowering happens inside tantivy's
/// parser, and the resulting `BooleanQuery` cannot be rewritten from outside.
///
/// SCOPE, deliberately narrow. Only the exact token sequence `AND` `NOT` is
/// touched, both uppercase, both standalone. Quoted text is skipped, so a
/// literal `"a AND NOT b"` phrase query is never rewritten. Lowercase `and not`
/// is a pair of ordinary terms in this grammar and is left alone.
///
/// NOT A CONFORMANCE CHANGE. This runs only from
/// [`TantivyIndex::parse_query_shipping`]; every `oracle_observe_*` caller
/// keeps using [`TantivyIndex::parse_query_lenient`] and still reproduces the
/// defect, so the pinned comparator does not move. Quill was MEASURED to share
/// this behaviour (it returns nothing for `A AND NOT B` too, because
/// `wrap_not_for_and` in its parser mirrors the same lowering), which is why
/// repairing this side alone changes no Quill-versus-oracle comparison.
fn repair_and_not(query: &str) -> Cow<'_, str> {
    // Cheap reject: the token pair cannot be present.
    if !query.contains("AND") {
        return Cow::Borrowed(query);
    }

    let bytes = query.as_bytes();
    let mut drop_ranges: Vec<(usize, usize)> = Vec::new();
    let mut in_quotes = false;
    let mut index = 0_usize;

    while index < bytes.len() {
        let byte = bytes[index];
        if byte == b'"' {
            in_quotes = !in_quotes;
            index += 1;
            continue;
        }
        if in_quotes {
            index += 1;
            continue;
        }
        if byte == b'A' && query[index..].starts_with("AND") {
            let starts_token = index.checked_sub(1).is_none_or(|previous| {
                bytes[previous].is_ascii_whitespace() || bytes[previous] == b'('
            });
            // `AND` must be followed by whitespace, then `NOT` as its own token.
            let mut cursor = index + 3;
            let space_start = cursor;
            while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
                cursor += 1;
            }
            let had_space = cursor > space_start;
            let followed_by_not = had_space
                && query[cursor..].starts_with("NOT")
                && bytes
                    .get(cursor + 3)
                    .is_some_and(|next| next.is_ascii_whitespace() || *next == b'(');
            if starts_token && followed_by_not {
                // Drop `AND` and the whitespace that separated it from `NOT`,
                // leaving the preceding separator intact: `a AND NOT b`
                // becomes `a NOT b`, never `aNOT b`.
                drop_ranges.push((index, cursor));
                index = cursor;
                continue;
            }
        }
        index += 1;
    }

    if drop_ranges.is_empty() {
        return Cow::Borrowed(query);
    }
    let mut repaired = String::with_capacity(query.len());
    let mut copied = 0_usize;
    for (start, end) in drop_ranges {
        repaired.push_str(&query[copied..start]);
        copied = end;
    }
    repaired.push_str(&query[copied..]);
    Cow::Owned(repaired)
}

fn repair_boosted_group_negation(query: &str) -> Cow<'_, str> {
    // Cheap reject: no group boost, nothing to repair.
    if !query.contains(")^") {
        return Cow::Borrowed(query);
    }

    let bytes = query.as_bytes();
    // One flag per open group: did THIS group contain a negation directly?
    let mut group_negates: Vec<bool> = Vec::new();
    // Byte ranges of `^<number>` suffixes to remove.
    let mut drop_ranges: Vec<(usize, usize)> = Vec::new();
    let mut in_quotes = false;
    let mut index = 0_usize;

    while index < bytes.len() {
        let byte = bytes[index];
        if byte == b'"' {
            in_quotes = !in_quotes;
            index += 1;
            continue;
        }
        if in_quotes {
            index += 1;
            continue;
        }
        match byte {
            b'(' => group_negates.push(false),
            b')' => {
                let negates = group_negates.pop().unwrap_or(false);
                // A boost suffix is `^` followed by a number, possibly decimal.
                let mut cursor = index + 1;
                if negates && cursor < bytes.len() && bytes[cursor] == b'^' {
                    let start = cursor;
                    cursor += 1;
                    let digits_start = cursor;
                    while cursor < bytes.len()
                        && (bytes[cursor].is_ascii_digit() || bytes[cursor] == b'.')
                    {
                        cursor += 1;
                    }
                    if cursor > digits_start {
                        drop_ranges.push((start, cursor));
                    }
                }
            }
            b'N' => {
                // `NOT` as a standalone token, at this group's own level.
                let is_token = query[index..].starts_with("NOT")
                    && index.checked_sub(1).is_none_or(|previous| {
                        bytes[previous].is_ascii_whitespace() || bytes[previous] == b'('
                    })
                    && bytes
                        .get(index + 3)
                        .is_none_or(|next| next.is_ascii_whitespace() || *next == b'(');
                if is_token && let Some(current) = group_negates.last_mut() {
                    *current = true;
                }
            }
            b'-' => {
                // Prefix exclusion: `-term`, not a hyphen inside a word.
                let is_prefix = index.checked_sub(1).is_none_or(|previous| {
                    bytes[previous].is_ascii_whitespace() || bytes[previous] == b'('
                }) && bytes
                    .get(index + 1)
                    .is_some_and(|next| !next.is_ascii_whitespace());
                if is_prefix && let Some(current) = group_negates.last_mut() {
                    *current = true;
                }
            }
            _ => {}
        }
        index += 1;
    }

    if drop_ranges.is_empty() {
        return Cow::Borrowed(query);
    }
    let mut repaired = String::with_capacity(query.len());
    let mut cursor = 0_usize;
    for (start, end) in drop_ranges {
        repaired.push_str(&query[cursor..start]);
        cursor = end;
    }
    repaired.push_str(&query[cursor..]);
    Cow::Owned(repaired)
}

/// One operand of a boolean level, with its occurrence prefix decoded.
struct LevelOperand {
    negated: bool,
    required: bool,
    text: String,
}

/// A parsed element of one boolean level.
enum LevelItem {
    Operand(LevelOperand),
    And,
    Or,
}

/// Repair a CONJUNCTION that contains a negated operand (bd-8a2a8, **DIV-010**).
///
/// THE DEFECT. The pinned grammar declares `default join := OR; explicit AND has
/// precedence over OR` (`docs/contracts/quill-language-contract.md`), so
/// `a NOT b AND c` reads as `a OR (c AND NOT b)`. Tantivy 0.26.1 lowers the
/// `AND` conjunct to `Bool{[(Must, Bool{[(MustNot, b)], msm: 0}), (Must, c)]}`,
/// whose first operand has no positive clause and therefore matches nothing, so
/// the whole conjunct drops and only `a` survives. MEASURED on the shared
/// Core100 fixture: `release NOT bounds AND small` returns 21 documents
/// (`|release|`) where the declared reading is 41 (`|release ∪ small|`), which
/// is what Quill returns and what this engine ITSELF returns when the same
/// grouping is written out as `release ((small) NOT bounds)`. Two independent
/// implementations agreeing on the explicit form is what attributes the defect
/// to the implicit lowering rather than to either engine's semantics.
///
/// THE REPAIR normalises a whole `AND` chain that contains a negation into the
/// occurrence spelling this engine already lowers correctly: every positive
/// operand becomes `+operand` and every negative one `-operand`, in the operand
/// order the user wrote, so `a NOT b AND c` becomes `a (-b +c)`. That is exact
/// rather than approximate —
/// `+`/`-` ARE the `Must`/`MustNot` the chain means — which matters because the
/// obvious narrow alternative, deleting the `AND`, was measured to break queries
/// that were already correct (`explains NOT bounds AND refactors` answers 60 and
/// dropping the `AND` answers 59). It is textual for the same reason
/// [`repair_and_not`] and [`repair_boosted_group_negation`] are: the lowering
/// happens inside tantivy's parser and the resulting `BooleanQuery` cannot be
/// rewritten from outside.
///
/// SCOPE, deliberately narrow. A chain is rewritten only when it holds more than
/// one operand and at least one of them is negated; a pure-positive `a AND b`
/// keeps its spelling. Anything the scanner cannot decode — an unbalanced quote,
/// a dangling `AND`, an empty operand — returns the query untouched rather than
/// guessing. Quoted text is skipped, so a literal `"a AND NOT b"` phrase is
/// never rewritten, and a `field:(a b)` operand is carried through opaquely.
///
/// A CHAIN THAT SPANS ITS WHOLE LEVEL IS EMITTED WITHOUT PARENTHESES, which is
/// load-bearing rather than cosmetic: [`repair_boosted_group_negation`] records
/// negation against the INNERMOST open group, so wrapping `(a AND NOT b)^2` into
/// `((+a -b))^2` would hide the negation from it and silently regress DIV-009.
///
/// NOT A CONFORMANCE CHANGE. This runs only from
/// [`TantivyIndex::parse_query_shipping`]; every `oracle_observe_*` caller keeps
/// using [`TantivyIndex::parse_query_lenient`] and still reproduces the defect,
/// so the pinned comparator does not move.
fn repair_negated_conjunction(query: &str) -> Cow<'_, str> {
    // Cheap reject: without an explicit `AND` there is no chain to normalise.
    if !query.contains("AND") {
        return Cow::Borrowed(query);
    }
    rewrite_conjunction_level(query).map_or(Cow::Borrowed(query), Cow::Owned)
}

/// Is `token` present at `index` as a standalone token at this level?
fn is_level_token(level: &str, index: usize, token: &str) -> bool {
    let bytes = level.as_bytes();
    level[index..].starts_with(token)
        && index
            .checked_sub(1)
            .is_none_or(|previous| bytes[previous].is_ascii_whitespace() || bytes[previous] == b'(')
        && bytes
            .get(index + token.len())
            .is_none_or(|next| next.is_ascii_whitespace() || *next == b'(')
}

/// Find the `)` matching the `(` at `open`, ignoring parentheses inside quotes.
fn matching_close_paren(level: &str, open: usize) -> Option<usize> {
    let bytes = level.as_bytes();
    let mut depth = 0_usize;
    let mut in_quotes = false;
    for (offset, byte) in bytes[open..].iter().enumerate() {
        let cursor = open + offset;
        match byte {
            b'"' => in_quotes = !in_quotes,
            b'(' if !in_quotes => depth += 1,
            b')' if !in_quotes => {
                depth -= 1;
                if depth == 0 {
                    return Some(cursor);
                }
            }
            _ => {}
        }
    }
    None
}

/// Read one operand body starting at `start`, returning its text, the index just
/// past it, and whether a nested level inside it was rewritten.
fn parse_operand_body(level: &str, start: usize) -> Option<(String, usize, bool)> {
    let bytes = level.as_bytes();
    if start >= bytes.len() {
        return None;
    }
    if bytes[start] == b'(' {
        let close = matching_close_paren(level, start)?;
        let inner = &level[start + 1..close];
        let rewritten = rewrite_conjunction_level(inner);
        let inner_changed = rewritten.is_some();
        let inner_text = rewritten.unwrap_or_else(|| inner.to_string());
        // Trailing modifiers stay attached to the group: `^boost`, `~slop`.
        let mut cursor = close + 1;
        while cursor < bytes.len()
            && !bytes[cursor].is_ascii_whitespace()
            && bytes[cursor] != b')'
            && bytes[cursor] != b'('
        {
            cursor += 1;
        }
        let suffix = &level[close + 1..cursor];
        return Some((format!("({inner_text}){suffix}"), cursor, inner_changed));
    }

    let mut in_quotes = false;
    let mut cursor = start;
    while cursor < bytes.len() {
        let byte = bytes[cursor];
        if byte == b'"' {
            in_quotes = !in_quotes;
            cursor += 1;
            continue;
        }
        if in_quotes {
            cursor += 1;
            continue;
        }
        // A `(` inside a bare operand is a fielded group like `field:(a b)`:
        // carry it through opaquely rather than splitting the operand in two.
        if byte == b'(' {
            cursor = matching_close_paren(level, cursor)? + 1;
            continue;
        }
        if byte.is_ascii_whitespace() || byte == b')' {
            break;
        }
        cursor += 1;
    }
    if in_quotes || cursor == start {
        return None;
    }
    Some((level[start..cursor].to_string(), cursor, false))
}

/// Rewrite every negated `AND` chain at one boolean level, recursing into
/// parenthesised operands. `None` means "nothing to change, or not decodable".
fn rewrite_conjunction_level(level: &str) -> Option<String> {
    let bytes = level.as_bytes();
    let mut items: Vec<LevelItem> = Vec::new();
    let mut nested_changed = false;
    let mut index = 0_usize;

    while index < bytes.len() {
        if bytes[index].is_ascii_whitespace() {
            index += 1;
            continue;
        }
        if is_level_token(level, index, "AND") {
            items.push(LevelItem::And);
            index += 3;
            continue;
        }
        if is_level_token(level, index, "OR") {
            items.push(LevelItem::Or);
            index += 2;
            continue;
        }
        // Occurrence prefixes: any run of negators, and an explicit `+`.
        let mut negated = false;
        let mut required = false;
        loop {
            let byte = *bytes.get(index)?;
            if byte == b'-' || byte == b'+' {
                // A bare `-` or `+` with nothing attached is not an occurrence
                // prefix, and this scanner declines rather than guessing.
                if bytes
                    .get(index + 1)
                    .is_none_or(|next| next.is_ascii_whitespace())
                {
                    return None;
                }
                if byte == b'-' {
                    negated = true;
                } else {
                    required = true;
                }
                index += 1;
                continue;
            }
            if is_level_token(level, index, "NOT") {
                negated = true;
                index += 3;
                while index < bytes.len() && bytes[index].is_ascii_whitespace() {
                    index += 1;
                }
                continue;
            }
            break;
        }
        let (text, next, inner_changed) = parse_operand_body(level, index)?;
        nested_changed |= inner_changed;
        items.push(LevelItem::Operand(LevelOperand {
            negated,
            required,
            text,
        }));
        index = next;
    }

    // Group the level into `AND` chains. `AND` binds tighter than the default
    // join, so a maximal chain is exactly the unit whose lowering is defective.
    let mut chains: Vec<Vec<LevelOperand>> = Vec::new();
    let mut joins: Vec<&'static str> = Vec::new();
    let mut expect_operand = false;
    let mut next_join = " ";
    for item in items {
        match item {
            LevelItem::And => {
                if chains.is_empty() || expect_operand {
                    return None;
                }
                expect_operand = true;
            }
            LevelItem::Or => {
                if chains.is_empty() || expect_operand {
                    return None;
                }
                next_join = " OR ";
            }
            LevelItem::Operand(operand) => {
                if expect_operand {
                    chains.last_mut()?.push(operand);
                    expect_operand = false;
                } else {
                    if !chains.is_empty() {
                        joins.push(next_join);
                    }
                    next_join = " ";
                    chains.push(vec![operand]);
                }
            }
        }
    }
    if expect_operand || chains.is_empty() {
        return None;
    }

    let single_chain = chains.len() == 1;
    let mut rewrote = false;
    let mut rendered: Vec<String> = Vec::with_capacity(chains.len());
    for chain in &chains {
        if chain.len() > 1 && chain.iter().any(|operand| operand.negated) {
            rewrote = true;
            let mut body = String::new();
            for operand in chain {
                if !body.is_empty() {
                    body.push(' ');
                }
                body.push(if operand.negated { '-' } else { '+' });
                body.push_str(&operand.text);
            }
            // Bare when it IS the level, so DIV-009 still sees the negation.
            rendered.push(if single_chain {
                body
            } else {
                format!("({body})")
            });
        } else if chain.len() > 1 {
            rendered.push(
                chain
                    .iter()
                    .map(render_operand)
                    .collect::<Vec<_>>()
                    .join(" AND "),
            );
        } else {
            rendered.push(render_operand(&chain[0]));
        }
    }

    if !rewrote && !nested_changed {
        return None;
    }
    let mut output = rendered.first()?.clone();
    for (position, piece) in rendered.iter().enumerate().skip(1) {
        output.push_str(joins.get(position - 1)?);
        output.push_str(piece);
    }
    Some(output)
}

/// Re-emit an operand outside a rewritten chain, preserving its occurrence.
fn render_operand(operand: &LevelOperand) -> String {
    if operand.negated {
        format!("-{}", operand.text)
    } else if operand.required {
        format!("+{}", operand.text)
    } else {
        operand.text.clone()
    }
}

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

/// Dev-only exact counted page used by the Quill replacement witness.
///
/// This is deliberately separate from [`OracleQueryObservation`]: the latter
/// owns cutoff-tie expansion for rank-parity campaigns, while this DTO proves
/// the incumbent's real offset-pagination contract. Shipping callers should
/// not depend on this evidence-only surface.
#[cfg(feature = "tantivy-oracle")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OraclePageObservation {
    /// Ordered page returned by Tantivy's real offset collector.
    pub hits: Vec<OraclePageHit>,
    /// Exact number of matches, independent of `limit` and `offset`.
    pub total_count: usize,
    /// Exact live-document count.
    pub doc_count: usize,
}

/// One ordered hit from [`OraclePageObservation`].
#[cfg(feature = "tantivy-oracle")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OraclePageHit {
    /// Stable external document identifier.
    pub doc_id: String,
    /// Exact BM25 score representation.
    pub score_bits: u32,
    /// Zero-based rank inside this returned page.
    pub page_rank: usize,
    /// Zero-based rank in the complete result stream.
    pub absolute_rank: usize,
}

#[cfg(test)]
std::thread_local! {
    static COUNTED_COLLECTOR_INVOCATIONS: std::cell::Cell<u64> =
        const { std::cell::Cell::new(0) };
    static TOP_K_COLLECTOR_INVOCATIONS: std::cell::Cell<u64> =
        const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn reset_collector_invocations() {
    COUNTED_COLLECTOR_INVOCATIONS.set(0);
    TOP_K_COLLECTOR_INVOCATIONS.set(0);
}

#[cfg(test)]
fn collector_invocations() -> (u64, u64) {
    (
        COUNTED_COLLECTOR_INVOCATIONS.get(),
        TOP_K_COLLECTOR_INVOCATIONS.get(),
    )
}

/// Execute a pre-built Tantivy query with offset pagination.
/// Execute a Tantivy collector search behind a panic guard.
///
/// The pinned Tantivy 0.26.1 scorer stack contains at least one panic
/// reachable from ordinary user input: a negated phrase whose exact sequence
/// is absent seeks a terminated docset in `PhraseScorer` (bd-nqeb4, found by
/// the bd-bsjw structure-aware campaign). A search engine must degrade, not
/// abort the host process, so the execution boundary converts an engine
/// panic into the same typed error surface as an engine `Err`. The searcher
/// is an immutable snapshot, so the unwind leaves no poisoned state behind.
fn search_guarded<C: tantivy::collector::Collector>(
    searcher: &Searcher,
    query: &dyn tantivy::query::Query,
    collector: &C,
) -> SearchResult<C::Fruit> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        searcher.search(query, collector)
    })) {
        Ok(result) => result.map_err(|error| SearchError::SubsystemError {
            subsystem: "tantivy",
            source: Box::new(error),
        }),
        Err(panic) => {
            let detail = panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or("non-string panic payload");
            warn!(
                panic = detail,
                "tantivy panicked during query execution; degrading to a typed error (bd-nqeb4)"
            );
            Err(SearchError::SubsystemError {
                subsystem: "tantivy",
                source: format!("tantivy panicked during query execution: {detail}").into(),
            })
        }
    }
}

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
    #[cfg(test)]
    COUNTED_COLLECTOR_INVOCATIONS.set(COUNTED_COLLECTOR_INVOCATIONS.get().saturating_add(1));

    if limit == 0 {
        let total_count = search_guarded(searcher, query, &Count)?;
        return Ok(LexicalSearchResult {
            hits: Vec::new(),
            total_count,
        });
    }

    let (top_docs, total_count) = search_guarded(
        searcher,
        query,
        &(
            TopDocs::with_limit(limit)
                .and_offset(offset)
                .order_by_score(),
            Count,
        ),
    )?;

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

    #[cfg(test)]
    TOP_K_COLLECTOR_INVOCATIONS.set(TOP_K_COLLECTOR_INVOCATIONS.get().saturating_add(1));

    let top_docs = search_guarded(
        searcher,
        query,
        &TopDocs::with_limit(limit)
            .and_offset(offset)
            .order_by_score(),
    )?;

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
        tantivy::schema::IndexRecordOption::Basic
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

/// Which Tantivy writer constructor a benchmark index literally called.
///
/// This is a constructor identity, not a request record: `ShippingAuto` means
/// [`Index::writer`] was called and Tantivy chose the pool width itself;
/// `Fixed` means [`Index::writer_with_num_threads`] was called with exactly
/// the recorded width.
#[cfg(feature = "bench-internals")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkWriterMode {
    /// The shipping default path: `Index::writer(heap)`.
    ShippingAuto,
    /// The pinned path: `Index::writer_with_num_threads(threads, heap)`.
    Fixed {
        /// Exact width handed to Tantivy.
        threads: usize,
    },
}

/// Writer-pool width, and whether it is authenticated or merely unknown.
///
/// Tantivy 0.26.1 computes the `ShippingAuto` width inside `Index::writer`
/// from `available_parallelism()` clamped by the heap budget, and exposes no
/// accessor for it — `pub fn num_threads` does not exist anywhere in that
/// crate. Recomputing it here would mean copying `MAX_NUM_THREAD` and
/// `MEMORY_BUDGET_NUM_BYTES_MIN` into this crate and presenting an inference
/// as an observation, which would drift silently on any version bump. The
/// unknown is therefore typed, not filled in.
#[cfg(feature = "bench-internals")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkMaterializedWidth {
    /// The exact width Tantivy was given and used.
    Authenticated(usize),
    /// No trustworthy width is observable; evidence needing one must fail closed.
    Unobservable {
        /// Why the width cannot be authenticated.
        reason: BenchmarkWidthUnobservableReason,
    },
}

#[cfg(feature = "bench-internals")]
impl BenchmarkMaterializedWidth {
    /// The authenticated width, or `None` when no width is observable.
    #[must_use]
    pub const fn authenticated(self) -> Option<usize> {
        match self {
            Self::Authenticated(threads) => Some(threads),
            Self::Unobservable { .. } => None,
        }
    }
}

/// Why a width is not authenticated.
///
/// Typed and version-neutral on purpose: the receipt states *that* the
/// incumbent selects the width internally and publishes no accessor, which
/// stays true across upgrades, instead of pinning a version string that would
/// silently become a lie on the next bump.
#[cfg(feature = "bench-internals")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkWidthUnobservableReason {
    /// The engine chose the pool width inside its own constructor and exposes
    /// no accessor for the value it chose.
    EngineSelectedWidthNotExposed,
}

/// Which writer construction the caller explicitly asked for.
///
/// The plan, not the thread count, decides whether a benchmark receipt exists.
/// Inferring "benchmark" from `writer_threads: None` made every ordinary
/// `create`/`open`/`in_memory` index carry a shipping-auto receipt it never
/// asked for, which would let ordinary constructions masquerade as screened
/// benchmark candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WriterPlan {
    /// Ordinary shipping construction through Tantivy's own width selection.
    /// Never produces a benchmark receipt.
    Shipping,
    /// Ordinary construction at a pinned width. Never produces a benchmark
    /// receipt either.
    ///
    /// Width and receipt are independent axes. Collapsing this into
    /// `Shipping` silently moved the single-threaded oracle off
    /// `writer_with_num_threads(1, heap)` and onto Tantivy's auto selection —
    /// a real behaviour change bought only to express "no receipt".
    ///
    /// Gated with its only constructor, so the default build carries no
    /// variant that nothing can build.
    #[cfg(feature = "tantivy-oracle")]
    PinnedWidth(usize),
    /// Explicit benchmark construction through Tantivy's own width selection.
    #[cfg(feature = "bench-internals")]
    BenchmarkShippingAuto,
    /// Explicit benchmark construction at a pinned width.
    #[cfg(feature = "bench-internals")]
    BenchmarkFixed(usize),
}

#[cfg(feature = "bench-internals")]
impl WriterPlan {
    /// Width recorded on the benchmark accessor.
    ///
    /// Only a benchmark plan reports one: `PinnedWidth` pins a width without
    /// claiming any screening identity. Gated with the field it feeds, so the
    /// default build has no unreachable helper to warn about.
    const fn benchmark_threads(self) -> Option<usize> {
        match self {
            Self::BenchmarkFixed(threads) => Some(threads),
            Self::Shipping | Self::BenchmarkShippingAuto => None,
            #[cfg(feature = "tantivy-oracle")]
            Self::PinnedWidth(_) => None,
        }
    }
}

/// Read the positions setting back out of a live index schema.
///
/// The benchmark reopen path receives a caller's `positions` claim but attaches
/// to a schema that already exists, so the claim is an assertion about someone
/// else's bytes. This reads the indexing option actually recorded on the
/// `content` field, which is the only authority.
#[cfg(feature = "bench-internals")]
fn positions_from_live_schema(schema: &Schema) -> Option<bool> {
    let field = schema.get_field("content").ok()?;
    match schema.get_field_entry(field).field_type() {
        tantivy::schema::FieldType::Str(options) => {
            options.get_indexing_options().map(|indexing| {
                indexing.index_option() == tantivy::schema::IndexRecordOption::WithFreqsAndPositions
            })
        }
        _ => None,
    }
}

/// Typed receipt for one benchmark writer construction.
///
/// Oracle identity is exactly what this crate can authenticate in-process: the
/// tokenizer name actually registered, the field names actually present in the
/// live index schema, and the positions flag that built that schema. Tantivy's
/// crate version and source identity are deliberately absent — they are not
/// observable here, and asserting them would be a restatement that can drift.
#[cfg(feature = "bench-internals")]
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct BenchmarkWriterReceipt {
    /// Which constructor was called.
    pub mode: BenchmarkWriterMode,
    /// Heap budget handed to that constructor.
    pub writer_heap_bytes: usize,
    /// Authenticated width, or a typed unknown.
    pub materialized_width: BenchmarkMaterializedWidth,
    /// Positions option read back from the live index schema.
    pub positions: bool,
    /// Field names present in the live index schema, in schema order.
    pub schema_fields: Vec<String>,
    /// Tokenizer registered on this index. Owned so the whole receipt can be
    /// bound into an artifact without borrowing this crate's statics.
    pub tokenizer_name: String,
    /// Whether this writer replaced an earlier joined writer.
    pub writer_rearmed: bool,
}

/// One-shot live capability for a benchmark writer that was actually created.
///
/// This is deliberately neither `Clone` nor serializable/deserializable: it is
/// an in-process hand-off from the successful constructor branch to a live
/// benchmark consumer, not a persisted authentication claim. The descriptive
/// [`BenchmarkWriterReceipt`] remains available separately for diagnostics.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
#[derive(Debug)]
pub struct BenchmarkWriterAttestation {
    receipt: BenchmarkWriterReceipt,
    construction_id: u64,
}

#[cfg(feature = "bench-internals")]
impl BenchmarkWriterAttestation {
    /// Mint an attestation only after its writer constructor has succeeded.
    fn mint(receipt: BenchmarkWriterReceipt) -> Self {
        let construction_id = NEXT_BENCHMARK_WRITER_CONSTRUCTION_ID
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .expect("benchmark writer construction IDs exhausted");
        Self {
            receipt,
            construction_id,
        }
    }

    /// Return the authenticated receipt carried by this live capability.
    #[must_use]
    pub const fn receipt(&self) -> &BenchmarkWriterReceipt {
        &self.receipt
    }

    /// Return this process-local writer-construction identity.
    #[must_use]
    pub const fn construction_id(&self) -> u64 {
        self.construction_id
    }
}

#[cfg(feature = "bench-internals")]
impl BenchmarkWriterReceipt {
    /// Seed a receipt from the constructor branch that actually ran.
    ///
    /// `mode` is produced by the branch itself rather than passed in beside it,
    /// so the recorded mode and heap cannot drift from the `Index::writer` or
    /// `writer_with_num_threads` call that really happened. `positions` is read
    /// back from the live schema, never from a caller's claim.
    fn seed(
        mode: BenchmarkWriterMode,
        writer_heap_bytes: usize,
        index: &Index,
    ) -> SearchResult<Self> {
        let materialized_width = match mode {
            BenchmarkWriterMode::ShippingAuto => BenchmarkMaterializedWidth::Unobservable {
                reason: BenchmarkWidthUnobservableReason::EngineSelectedWidthNotExposed,
            },
            BenchmarkWriterMode::Fixed { threads } => {
                BenchmarkMaterializedWidth::Authenticated(threads)
            }
        };
        let schema = index.schema();
        // Absent positions evidence is a missing fact, not a false one. A
        // defaulted `false` would put an unearned claim into every artifact
        // this receipt is later bound into.
        let positions =
            positions_from_live_schema(&schema).ok_or_else(|| SearchError::InvalidConfig {
                field: "tantivy.positions".to_owned(),
                value: "unavailable".to_owned(),
                reason: "live index schema exposes no indexed content field, so the positions \
                         option cannot be authenticated"
                    .to_owned(),
            })?;
        Ok(Self {
            mode,
            writer_heap_bytes,
            materialized_width,
            positions,
            schema_fields: schema
                .fields()
                .map(|(_, entry)| entry.name().to_owned())
                .collect(),
            tokenizer_name: TOKENIZER_NAME.to_owned(),
            writer_rearmed: false,
        })
    }
}

/// Reject a benchmark caller whose `positions` claim disagrees with the schema
/// it is actually attaching to.
///
/// Creating an index builds the schema from this flag, but reopening one only
/// asserts about bytes that already exist. A silent disagreement would put a
/// false positions value into every downstream receipt.
#[cfg(feature = "bench-internals")]
fn reject_positions_disagreement(index: &Index, claimed: bool) -> SearchResult<()> {
    let observed =
        positions_from_live_schema(&index.schema()).ok_or_else(|| SearchError::InvalidConfig {
            field: "tantivy.positions".to_owned(),
            value: claimed.to_string(),
            reason: "live index schema exposes no indexed content field to authenticate positions"
                .to_owned(),
        })?;
    if observed == claimed {
        return Ok(());
    }
    Err(SearchError::InvalidConfig {
        field: "tantivy.positions".to_owned(),
        value: claimed.to_string(),
        reason: format!("live index schema records positions = {observed}"),
    })
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

/// Counted receipt for a benchmark-only Tantivy writer lifecycle fence.
///
/// `wait_merging_threads` consumes Tantivy's writer after joining its indexing
/// and segment-updater workers. Warm/update fixtures may rearm the same index
/// with the pinned writer configuration, while terminal bulk measurements
/// deliberately stop after the join so they do not construct and immediately
/// discard an unused replacement writer.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BenchmarkWriterJoinReceipt {
    /// Searchable segment count immediately before joining the writer.
    pub searchable_segments_before: usize,
    /// Searchable segment count after every background worker has joined.
    pub searchable_segments_after: usize,
    /// Time spent inside Tantivy's worker/merge join, excluding writer rearm.
    pub join_elapsed_ns: u64,
    /// Whether a replacement writer was successfully constructed after joining.
    pub writer_rearmed: bool,
}

/// Read-only Tantivy search handle retained across a terminal writer join.
///
/// This deliberately owns no writer.  A terminal benchmark obtains it only by
/// consuming [`TantivyIndex`] through
/// [`TantivyIndex::benchmark_join_workers_retaining_reader`], which joins the
/// real Tantivy workers first and leaves this handle alive for the required
/// post-join searchable-tail query.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub struct BenchmarkRetainedTantivyReader {
    fields: SchemaFields,
    reader: IndexReader,
}

#[cfg(feature = "bench-internals")]
impl BenchmarkRetainedTantivyReader {
    /// Run an exact stored-ID query through the reader retained after a writer
    /// join.
    ///
    /// This is intentionally a narrow benchmark seam rather than a general
    /// second lexical API: it proves that the terminal prepared-tail document
    /// is searchable *after* workers have quiesced, without creating or
    /// rearming a writer.
    ///
    /// # Errors
    ///
    /// Returns a typed Tantivy search failure.
    pub fn benchmark_search_exact_id(&self, document_id: &str) -> SearchResult<Vec<DocId>> {
        let query = TermQuery::new(
            Term::from_field_text(self.fields.id, document_id),
            IndexRecordOption::Basic,
        );
        let searcher = self.reader.searcher();
        let top_docs = search_guarded(&searcher, &query, &TopDocs::with_limit(2).order_by_score())?;

        // The exact `id` term itself authenticates membership.  Do not add a
        // stored-document hydration that Quill's matching IDHASH probe does
        // not pay; terminal proof cost is part of the timed lifecycle.
        Ok(top_docs
            .into_iter()
            .map(|_| document_id.to_owned().into())
            .collect())
    }
}

/// Ordered searchable-segment geometry from the pinned Tantivy oracle.
///
/// This conformance-only receipt preserves Tantivy's native `segment_ord`
/// assignment as well as physical and live document counts. It is absent from
/// normal shipping builds.
#[cfg(feature = "tantivy-oracle")]
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OracleSegmentLayout {
    /// Native Tantivy segment ordinal used in `DocAddress` tie-breaking.
    pub segment_ord: u32,
    /// Physical document cardinality, including deleted rows.
    pub max_doc: u32,
    /// Live document cardinality.
    pub num_docs: u32,
}

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
    /// Exact writer-pool width accepted by Tantivy's benchmark constructor.
    #[cfg(feature = "bench-internals")]
    benchmark_writer_threads: Option<usize>,
    /// Typed receipt for the benchmark writer construction, when this index
    /// was built through a benchmark seam.
    #[cfg(feature = "bench-internals")]
    benchmark_writer_receipt: Option<BenchmarkWriterReceipt>,
    /// One-shot live attestation for the benchmark writer construction.
    #[cfg(feature = "bench-internals")]
    benchmark_writer_attestation: Option<BenchmarkWriterAttestation>,
    /// Which Tantivy writer constructor this index actually invoked.
    ///
    /// Per-instance and test-only: a plan or receipt records what the caller
    /// *asked for*, so a test reading either proves nothing about the call that
    /// ran. This is written by the helper that performs the call itself. It is
    /// deliberately not a global counter — parallel tests would race one.
    #[cfg(test)]
    observed_writer_call: WriterCall,
}

/// The Tantivy writer constructor a `TantivyIndex` actually invoked.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WriterCall {
    /// `Index::writer(heap)` — Tantivy selected the width.
    Auto,
    /// `Index::writer_with_num_threads(threads, heap)`.
    ///
    /// Gated exactly as `call_fixed_writer` is: the only thing that can build
    /// this variant is that helper, so without those features it would be a
    /// variant nothing constructs.
    #[cfg(any(feature = "tantivy-oracle", feature = "bench-internals"))]
    Fixed(usize),
}

/// Perform the shipping-auto Tantivy call, recording that it was the one made.
fn call_auto_writer(
    index: &Index,
    writer_heap_bytes: usize,
    #[cfg(test)] observed: &mut WriterCall,
) -> tantivy::Result<IndexWriter> {
    #[cfg(test)]
    {
        *observed = WriterCall::Auto;
    }
    index.writer(writer_heap_bytes)
}

/// Perform the pinned-width Tantivy call, recording that it was the one made.
///
/// Gated to the features that actually reach it — the oracle's pinned plan and
/// the benchmark seams — so the default build has no helper without a caller.
#[cfg(any(feature = "tantivy-oracle", feature = "bench-internals"))]
fn call_fixed_writer(
    index: &Index,
    threads: usize,
    writer_heap_bytes: usize,
    #[cfg(test)] observed: &mut WriterCall,
) -> tantivy::Result<IndexWriter> {
    #[cfg(test)]
    {
        *observed = WriterCall::Fixed(threads);
    }
    index.writer_with_num_threads(threads, writer_heap_bytes)
}

impl std::fmt::Debug for TantivyIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TantivyIndex")
            .field(
                "staged_doc_count_hint",
                &self.doc_count.load(Ordering::Relaxed),
            )
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl TantivyIndex {
    fn checked_searcher_doc_count(searcher: &Searcher) -> SearchResult<usize> {
        usize::try_from(searcher.num_docs()).map_err(|_| SearchError::SubsystemError {
            subsystem: "tantivy",
            source: "current Tantivy reader document count does not fit usize".into(),
        })
    }

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

    /// Return the writer-pool width successfully materialized by the
    /// benchmark-only Tantivy constructor.
    ///
    /// `None` means no width is authenticated, which is the honest answer for
    /// the shipping-auto path; it is never a stand-in for an inferred width.
    #[cfg(feature = "bench-internals")]
    #[doc(hidden)]
    #[must_use]
    pub const fn benchmark_materialized_writer_threads(&self) -> Option<usize> {
        self.benchmark_writer_threads
    }

    /// Typed receipt for the benchmark writer this index was constructed with.
    #[cfg(feature = "bench-internals")]
    #[doc(hidden)]
    #[must_use]
    pub const fn benchmark_writer_receipt(&self) -> Option<&BenchmarkWriterReceipt> {
        self.benchmark_writer_receipt.as_ref()
    }

    /// Take the one-shot live attestation for this benchmark writer.
    ///
    /// The attestation is minted only after the writer constructor succeeds.
    /// Taking it does not remove the descriptive receipt, so a later rearm can
    /// mint a fresh attestation for the replacement writer.
    #[cfg(feature = "bench-internals")]
    #[doc(hidden)]
    pub fn take_benchmark_writer_attestation(&mut self) -> Option<BenchmarkWriterAttestation> {
        self.benchmark_writer_attestation.take()
    }

    /// Construct an in-memory oracle through Tantivy's **shipping** writer
    /// selection: `Index::writer(heap)` is called literally, and the resulting
    /// pool width is reported as unobservable rather than guessed.
    ///
    /// # Errors
    ///
    /// Returns the ordinary Tantivy writer-construction error, including the
    /// heap-too-small case, which stays fail-closed here.
    #[cfg(feature = "bench-internals")]
    #[doc(hidden)]
    pub fn in_memory_with_shipping_auto_writer(
        writer_heap_bytes: usize,
        positions: bool,
    ) -> SearchResult<Self> {
        let (schema, fields) = build_schema_with_positions(positions);
        let index = Index::create_in_ram(schema.clone());
        reject_positions_disagreement(&index, positions)?;
        Self::from_index_with_writer_threads(
            index,
            schema,
            fields,
            None,
            writer_heap_bytes,
            WriterPlan::BenchmarkShippingAuto,
        )
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
            // An ordinary single-threaded oracle, not a benchmark candidate:
            // it really does pin width 1, and claims no screening receipt.
            WriterPlan::PinnedWidth(1),
        )
    }

    /// Disable automatic merging for an exact oracle segment-topology proof.
    ///
    /// This method is intentionally separate from the benchmark-only helper:
    /// conformance fixtures use Tantivy's shipping indexing path but must keep
    /// explicit commit boundaries observable for native `DocAddress` ties.
    ///
    /// # Errors
    ///
    /// Returns a typed cancellation or writer-lock error.
    #[cfg(feature = "tantivy-oracle")]
    #[doc(hidden)]
    pub async fn oracle_disable_auto_merge(&self, cx: &Cx) -> SearchResult<()> {
        let writer = self
            .writer
            .lock(cx)
            .await
            .map_err(|error| Self::map_writer_lock_error("tantivy.oracle_no_merge", error))?;
        writer.set_merge_policy(Box::new(tantivy::merge_policy::NoMergePolicy));
        Ok(())
    }

    /// Return searchable oracle segments in native `segment_ord` order.
    ///
    /// # Errors
    ///
    /// Returns an invalid-config error if a segment ordinal cannot fit
    /// Tantivy's public address type.
    #[cfg(feature = "tantivy-oracle")]
    #[doc(hidden)]
    pub fn oracle_segment_layout(&self) -> SearchResult<Vec<OracleSegmentLayout>> {
        self.reader
            .searcher()
            .segment_readers()
            .iter()
            .enumerate()
            .map(|(segment_ord, segment)| {
                let segment_ord =
                    u32::try_from(segment_ord).map_err(|_| SearchError::InvalidConfig {
                        field: "tantivy.segment_ord".to_owned(),
                        value: segment_ord.to_string(),
                        reason: "segment ordinal must fit in u32".to_owned(),
                    })?;
                Ok(OracleSegmentLayout {
                    segment_ord,
                    max_doc: segment.max_doc(),
                    num_docs: segment.num_docs(),
                })
            })
            .collect()
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
        reject_positions_disagreement(&index, positions)?;
        Self::from_index_with_writer_threads(
            index,
            schema,
            fields,
            None,
            writer_heap_bytes,
            WriterPlan::BenchmarkFixed(writer_threads),
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
        reject_positions_disagreement(&index, positions)?;
        Self::from_index_with_writer_threads(
            index,
            schema,
            fields,
            Some(path.to_path_buf()),
            writer_heap_bytes,
            WriterPlan::BenchmarkFixed(writer_threads),
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
        reject_positions_disagreement(&index, positions)?;
        Self::from_index_with_writer_threads(
            index,
            schema,
            fields,
            Some(path.to_path_buf()),
            writer_heap_bytes,
            WriterPlan::BenchmarkFixed(writer_threads),
        )
    }

    /// Join every indexing and merge worker, then rearm the same benchmark index.
    ///
    /// The returned duration measures only Tantivy's
    /// [`IndexWriter::wait_merging_threads`] call. Constructing the replacement
    /// writer is excluded because it is harness bookkeeping, not incumbent
    /// maintenance. Searchable segment counts make the lifecycle boundary
    /// auditable without relying on timing variance.
    ///
    /// # Errors
    ///
    /// Returns an invalid writer-thread configuration, poisoned writer mutex,
    /// Tantivy worker/merge failure, segment metadata failure, or writer
    /// reconstruction failure.
    #[cfg(feature = "bench-internals")]
    #[doc(hidden)]
    pub fn benchmark_join_workers_and_rearm(
        self,
        writer_heap_bytes: usize,
        writer_threads: usize,
    ) -> SearchResult<(Self, BenchmarkWriterJoinReceipt)> {
        validate_benchmark_writer_threads(writer_threads)?;
        let Self {
            index,
            fields,
            reader,
            writer,
            doc_count,
            ord_table,
            path,
            benchmark_writer_threads: _,
            benchmark_writer_receipt,
            benchmark_writer_attestation: _,
            #[cfg(test)]
                observed_writer_call: _,
        } = self;
        let mut receipt = Self::benchmark_join_writer(&index, writer)?;
        // The rearm reconstructs through the same helper the fixed plan uses,
        // so its observation is the call it actually made rather than the one
        // the old writer had made.
        #[cfg(test)]
        let mut observed_writer_call = WriterCall::Auto;
        let writer = call_fixed_writer(
            &index,
            writer_threads,
            writer_heap_bytes,
            #[cfg(test)]
            &mut observed_writer_call,
        )
        .map_err(|error| SearchError::SubsystemError {
            subsystem: "tantivy",
            source: Box::new(error),
        })?;
        receipt.writer_rearmed = true;
        let benchmark_writer_receipt =
            benchmark_writer_receipt.map(|previous| BenchmarkWriterReceipt {
                mode: BenchmarkWriterMode::Fixed {
                    threads: writer_threads,
                },
                writer_heap_bytes,
                materialized_width: BenchmarkMaterializedWidth::Authenticated(writer_threads),
                writer_rearmed: true,
                ..previous
            });
        let benchmark_writer_attestation = benchmark_writer_receipt
            .as_ref()
            .map(|receipt| BenchmarkWriterAttestation::mint(receipt.clone()));
        Ok((
            Self {
                index,
                fields,
                reader,
                writer: Mutex::new(writer),
                doc_count,
                ord_table,
                path,
                benchmark_writer_threads: Some(writer_threads),
                benchmark_writer_receipt,
                benchmark_writer_attestation,
                #[cfg(test)]
                observed_writer_call,
            },
            receipt,
        ))
    }

    /// Join every indexing and merge worker while retaining a read-only search
    /// handle and without constructing another writer.
    ///
    /// The returned handle is the only terminal benchmark surface capable of
    /// proving that a query returned after the actual worker join.  Segment
    /// metadata is retained in the receipt for diagnostics but is not a
    /// searchability proof.
    ///
    /// # Errors
    ///
    /// Returns a poisoned writer-mutex, Tantivy worker/merge, or segment
    /// metadata error.
    #[cfg(feature = "bench-internals")]
    #[doc(hidden)]
    pub fn benchmark_join_workers_retaining_reader(
        self,
    ) -> SearchResult<(BenchmarkRetainedTantivyReader, BenchmarkWriterJoinReceipt)> {
        let Self {
            index,
            fields,
            reader,
            writer,
            doc_count,
            ord_table,
            path,
            benchmark_writer_threads,
            benchmark_writer_receipt,
            benchmark_writer_attestation: _,
            #[cfg(test)]
                observed_writer_call: _,
        } = self;
        let receipt = Self::benchmark_join_writer(&index, writer)?;
        drop((doc_count, ord_table, path, benchmark_writer_threads));
        drop(benchmark_writer_receipt);
        Ok((BenchmarkRetainedTantivyReader { fields, reader }, receipt))
    }

    /// Join every indexing and merge worker without constructing another writer.
    ///
    /// This is the terminal lifecycle fence for one-shot bulk measurements. It
    /// consumes the benchmark index, waits for every incumbent background
    /// worker, and returns only the counted receipt. Callers that will perform
    /// subsequent writes must use [`Self::benchmark_join_workers_and_rearm`].
    ///
    /// # Errors
    ///
    /// Returns a poisoned writer-mutex, Tantivy worker/merge, or segment
    /// metadata error.
    #[cfg(feature = "bench-internals")]
    #[doc(hidden)]
    pub fn benchmark_join_workers(self) -> SearchResult<BenchmarkWriterJoinReceipt> {
        let (_reader, receipt) = self.benchmark_join_workers_retaining_reader()?;
        Ok(receipt)
    }

    #[cfg(feature = "bench-internals")]
    fn benchmark_join_writer(
        index: &Index,
        writer: Mutex<IndexWriter>,
    ) -> SearchResult<BenchmarkWriterJoinReceipt> {
        let searchable_segments_before = index
            .searchable_segment_ids()
            .map_err(|error| SearchError::SubsystemError {
                subsystem: "tantivy",
                source: Box::new(error),
            })?
            .len();
        let writer = writer
            .into_inner()
            .map_err(|error| Self::map_writer_lock_error("tantivy.benchmark_join", error))?;
        let timer = std::time::Instant::now();
        writer
            .wait_merging_threads()
            .map_err(|error| SearchError::SubsystemError {
                subsystem: "tantivy",
                source: Box::new(error),
            })?;
        let join_elapsed_ns = u64::try_from(timer.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let searchable_segments_after = index
            .searchable_segment_ids()
            .map_err(|error| SearchError::SubsystemError {
                subsystem: "tantivy",
                source: Box::new(error),
            })?
            .len();
        Ok(BenchmarkWriterJoinReceipt {
            searchable_segments_before,
            searchable_segments_after,
            join_elapsed_ns,
            writer_rearmed: false,
        })
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
        let usage =
            self.reader
                .searcher()
                .space_usage()
                .map_err(|e| SearchError::SubsystemError {
                    subsystem: "tantivy",
                    source: Box::new(e),
                })?;
        let segment_count = usage.segments().len();
        let index_bytes = usage.total().get_bytes();
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
        Self::from_index_with_writer_threads(
            index,
            schema,
            fields,
            path,
            writer_heap_bytes,
            WriterPlan::Shipping,
        )
    }

    fn from_index_with_writer_threads(
        index: Index,
        _schema: Schema,
        mut fields: SchemaFields,
        path: Option<PathBuf>,
        writer_heap_bytes: usize,
        plan: WriterPlan,
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

        // The selection branch is the single source of both the writer and its
        // receipt seed. Nothing downstream can relabel a `writer_with_num_threads`
        // construction as shipping-auto, or vice versa, because the mode is
        // produced here by the arm that actually ran.
        #[cfg(feature = "bench-internals")]
        let mut benchmark_writer_receipt = None;
        #[cfg(feature = "bench-internals")]
        let mut benchmark_writer_attestation = None;
        // Default is overwritten by whichever helper actually runs below; the
        // helper is the only writer of this value.
        #[cfg(test)]
        let mut observed_writer_call = WriterCall::Auto;
        let writer = match plan {
            // Ordinary shipping construction reaches the same Tantivy call as a
            // benchmark shipping-auto construction, and deliberately produces no
            // receipt: only an explicit benchmark plan may claim one.
            WriterPlan::Shipping => call_auto_writer(
                &index,
                writer_heap_bytes,
                #[cfg(test)]
                &mut observed_writer_call,
            ),
            // Pinned but unscreened: the same Tantivy call a benchmark fixed
            // plan makes, deliberately without a receipt.
            #[cfg(feature = "tantivy-oracle")]
            WriterPlan::PinnedWidth(thread_count) => call_fixed_writer(
                &index,
                thread_count,
                writer_heap_bytes,
                #[cfg(test)]
                &mut observed_writer_call,
            ),
            #[cfg(feature = "bench-internals")]
            WriterPlan::BenchmarkShippingAuto => {
                let writer = call_auto_writer(
                    &index,
                    writer_heap_bytes,
                    #[cfg(test)]
                    &mut observed_writer_call,
                );
                if writer.is_ok() {
                    let receipt = BenchmarkWriterReceipt::seed(
                        BenchmarkWriterMode::ShippingAuto,
                        writer_heap_bytes,
                        &index,
                    )?;
                    benchmark_writer_attestation =
                        Some(BenchmarkWriterAttestation::mint(receipt.clone()));
                    benchmark_writer_receipt = Some(receipt);
                }
                writer
            }
            #[cfg(feature = "bench-internals")]
            WriterPlan::BenchmarkFixed(thread_count) => {
                let writer = call_fixed_writer(
                    &index,
                    thread_count,
                    writer_heap_bytes,
                    #[cfg(test)]
                    &mut observed_writer_call,
                );
                if writer.is_ok() {
                    let receipt = BenchmarkWriterReceipt::seed(
                        BenchmarkWriterMode::Fixed {
                            threads: thread_count,
                        },
                        writer_heap_bytes,
                        &index,
                    )?;
                    benchmark_writer_attestation =
                        Some(BenchmarkWriterAttestation::mint(receipt.clone()));
                    benchmark_writer_receipt = Some(receipt);
                }
                writer
            }
        }
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
            #[cfg(feature = "bench-internals")]
            benchmark_writer_threads: plan.benchmark_threads(),
            // Seeded above by the selection branch itself, so the record and
            // the call it describes cannot disagree.
            //
            // This constructor is shared with the default build, so the field
            // needs the same gate as its neighbour above — without it the
            // no-feature build names a field that does not exist.
            #[cfg(feature = "bench-internals")]
            benchmark_writer_receipt,
            #[cfg(feature = "bench-internals")]
            benchmark_writer_attestation,
            #[cfg(test)]
            observed_writer_call,
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

    /// Parse for the SHIPPING search path: lenient, plus the bd-f20ye repair.
    ///
    /// THE TWO ROLES OF THIS CRATE DIVERGE HERE, AND ONLY HERE.
    /// `frankensearch-lexical` is both the shipping Tantivy backend and the
    /// Quill gauntlet's pinned conformance oracle. Every `oracle_observe_*`
    /// method keeps calling [`Self::parse_query_lenient`] and stays bit-faithful
    /// to Tantivy 0.26.1, because the oracle is a pinned COMPARATOR and moving
    /// it would silently move the target Quill is measured against. Every
    /// user-facing `search*` method calls THIS instead, because a boost that
    /// changes boolean MEMBERSHIP is a defect by any reading and shipping it to
    /// users is not defensible.
    ///
    /// The resulting divergence between the two roles is registered as
    /// **DIV-009** in `docs/contracts/quill-divergence-register.md`, disposition
    /// accepted-with-rationale. Do not "unify" these two paths without
    /// superseding that entry: the shipping path is deliberately NOT
    /// bit-faithful to the oracle for this one query shape.
    fn parse_query_shipping(&self, query: &str) -> Box<dyn tantivy::query::Query> {
        // ORDER IS LOAD-BEARING. bd-8a2a8 runs first because it decodes whole
        // `AND` chains, and it needs to see the `AND` that bd-eeq0q deletes:
        // after that deletion `a AND NOT b AND c` reads as the unrelated chain
        // `NOT b AND c` and would be normalised to the wrong grouping. Both
        // repairs leave a `-` or `NOT` inside the group they touch, so the
        // bd-f20ye repair still recognises `(a AND NOT b)^2` as negating — it
        // records negation against the innermost open group, which is why
        // bd-8a2a8 emits a whole-level chain without adding parentheses.
        let conjunction = repair_negated_conjunction(query);
        let repaired = repair_and_not(&conjunction);
        self.parse_query_lenient(&repair_boosted_group_negation(&repaired))
    }

    /// Parse a query using lenient mode (never fails, returns best-effort query).
    ///
    /// Unknown field prefixes, unbalanced quotes, and other syntax issues are
    /// silently ignored rather than producing errors. This makes user-facing
    /// search robust against arbitrary input.
    ///
    /// BIT-FAITHFUL TO THE PINNED ORACLE. Callers in the `oracle_observe_*`
    /// family depend on this reproducing Tantivy 0.26.1 exactly, defects
    /// included — see [`Self::parse_query_shipping`] and DIV-009.
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
    /// A successful return commits the deletion and refreshes the readable
    /// generation before releasing the writer lock.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::SubsystemError` if the writer lock is poisoned
    /// or cancelled.
    pub async fn delete_document(&self, cx: &Cx, doc_id: &str) -> SearchResult<()> {
        let term = Term::from_field_text(self.fields.id, doc_id);
        let mut writer = self
            .writer
            .lock(cx)
            .await
            .map_err(|error| Self::map_writer_lock_error("tantivy.delete", error))?;
        writer.delete_term(term);
        writer
            .commit()
            .map_err(|error| SearchError::SubsystemError {
                subsystem: "tantivy",
                source: Box::new(error),
            })?;

        self.reader
            .reload()
            .map_err(|error| SearchError::SubsystemError {
                subsystem: "tantivy",
                source: Box::new(error),
            })?;

        let actual = Self::checked_searcher_doc_count(&self.reader.searcher())?;
        self.doc_count.store(actual, Ordering::Relaxed);

        debug!(doc_count = actual, %doc_id, "tantivy delete committed and reloaded");
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

        let parsed = self.parse_query_shipping(query);

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
        let searcher = self.reader.searcher();
        let doc_count = Self::checked_searcher_doc_count(&searcher)?;
        if query.trim().is_empty() {
            return Ok(OracleQueryObservation {
                hits: Vec::new(),
                cutoff_tie_group: Vec::new(),
                cutoff_tie_complete: true,
                total_count: 0,
                doc_count,
            });
        }

        let parsed = self.parse_query_lenient(query);
        let fetch_limit = if limit == 0 {
            0
        } else {
            limit.saturating_add(tie_expansion_limit)
        };
        // Preserve the exact shipping TopDocs-only path for both `hits` and
        // the expanded tie evidence. Tantivy may choose a different score
        // accumulation or cutoff strategy when `TopDocs` is paired with
        // `Count`, and a larger collector limit may select a different member
        // of an exact-score cutoff tie. Count therefore runs independently,
        // and the expanded query may describe the tie envelope but must never
        // redefine the incumbent result whose latency and rank we compare.
        let native_hits = execute_top_k(&searcher, &*parsed, limit, 0)?;
        let expanded_hits = execute_top_k(&searcher, &*parsed, fetch_limit, 0)?;
        let total_count = execute_query_with_offset(&searcher, &*parsed, 0, 0)?.total_count;
        let snippet_gen =
            try_build_snippet_generator(&searcher, &*parsed, self.fields.content, snippet_config);
        let materialize = |hit: LexicalDocHit| -> SearchResult<OracleRankedHit> {
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
            Ok(OracleRankedHit {
                doc_id,
                score_bits: hit.bm25_score.to_bits(),
                rank: hit.rank,
                segment_ord: hit.doc_address.segment_ord,
                segment_doc_id: hit.doc_address.doc_id,
                snippet,
            })
        };
        let hits = native_hits
            .into_iter()
            .map(&materialize)
            .collect::<SearchResult<Vec<_>>>()?;
        let expanded_hits = expanded_hits
            .into_iter()
            .map(materialize)
            .collect::<SearchResult<Vec<_>>>()?;

        let cutoff_bits = hits.last().map(|hit| hit.score_bits);
        let cutoff_tie_group = cutoff_bits.map_or_else(Vec::new, |cutoff| {
            expanded_hits
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
            total_count <= fetch_limit
                || expanded_hits.last().is_none_or(|last| {
                    !f32::from_bits(last.score_bits)
                        .total_cmp(&f32::from_bits(cutoff))
                        .is_eq()
                })
        });

        Ok(OracleQueryObservation {
            hits,
            cutoff_tie_group,
            cutoff_tie_complete,
            total_count,
            doc_count,
        })
    }

    /// Observe Tantivy's real exact-count offset-pagination path.
    ///
    /// This method exists only for the replacement conformance harness. It
    /// intentionally uses the same private lenient parser, reader snapshot,
    /// stored-ID materialization, and [`execute_query_with_offset`] collector
    /// as the incumbent engine. The exact count is returned by Tantivy's
    /// [`Count`] collector and is never inferred from the page length.
    ///
    /// # Errors
    ///
    /// Returns a typed search failure when query execution, rank arithmetic,
    /// or stored-document materialization fails.
    #[cfg(feature = "tantivy-oracle")]
    #[instrument(skip_all, fields(query = %query, limit = limit, offset = offset))]
    pub fn oracle_observe_page(
        &self,
        _cx: &Cx,
        query: &str,
        limit: usize,
        offset: usize,
    ) -> SearchResult<OraclePageObservation> {
        let query = Self::truncate_query(query);
        let searcher = self.reader.searcher();
        let doc_count = Self::checked_searcher_doc_count(&searcher)?;
        if query.trim().is_empty() {
            return Ok(OraclePageObservation {
                hits: Vec::new(),
                total_count: 0,
                doc_count,
            });
        }

        let parsed = self.parse_query_lenient(query);
        let search_result = execute_query_with_offset(&searcher, &*parsed, limit, offset)?;
        let mut hits = Vec::with_capacity(search_result.hits.len());
        for hit in search_result.hits {
            let doc = load_doc(&searcher, hit.doc_address)?;
            let doc_id = doc
                .get_first(self.fields.id)
                .and_then(|value| value.as_str())
                .unwrap_or_default()
                .to_owned();
            let absolute_rank =
                offset
                    .checked_add(hit.rank)
                    .ok_or_else(|| SearchError::InvalidConfig {
                        field: "tantivy.oracle_page_rank".to_owned(),
                        value: format!("{offset}+{}", hit.rank),
                        reason: "absolute result rank must fit usize".to_owned(),
                    })?;
            hits.push(OraclePageHit {
                doc_id,
                score_bits: hit.bm25_score.to_bits(),
                page_rank: hit.rank,
                absolute_rank,
            });
        }

        Ok(OraclePageObservation {
            hits,
            total_count: search_result.total_count,
            doc_count,
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

        let parsed = self.parse_query_shipping(query);
        let searcher = self.reader.searcher();
        let hits = execute_top_k(&searcher, &*parsed, limit, 0)?;
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

        let parsed = self.parse_query_shipping(query);
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

        let parsed = self.parse_query_shipping(query);
        let searcher = self.reader.searcher();
        let hits = execute_top_k(&searcher, &*parsed, limit, 0)?;
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

    /// Hydrate winners from an explicitly supplied searcher.
    ///
    /// A Tantivy `Searcher` holds a fixed segment set, so passing the *same*
    /// searcher that produced the scores is what makes hydration read the
    /// scoring generation rather than the newest one (`bd-8nqz.1`).
    fn hydrate_with_searcher(
        &self,
        searcher: &Searcher,
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
        let doc_addresses: Vec<DocAddress> = if unscored {
            // Every clause is an exact lookup on the unique `id` field. Hydration
            // consumes neither BM25 scores nor hit order, so asking Tantivy to
            // score and heap-sort these documents is pure overhead.
            search_guarded(searcher, &query, &DocSetCollector)?
                .into_iter()
                .collect()
        } else {
            search_guarded(
                searcher,
                &query,
                &TopDocs::with_limit(limit).order_by_score(),
            )?
            .into_iter()
            .map(|(_, doc_address)| doc_address)
            .collect()
        };

        for doc_address in doc_addresses {
            let doc = load_doc(searcher, doc_address)?;
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
        // The scored-collector baseline deliberately reads the CURRENT
        // generation, matching what the retired combined-trait path did, so the
        // A/B keeps comparing collectors rather than pinning strategies.
        Box::pin(async move { self.hydrate_with_searcher(&self.reader.searcher(), results, false) })
    }

    /// Unscored-collector arm of the same A/B.
    ///
    /// The shipping path reaches this through `hydrate_candidates` with a
    /// pinned searcher; the benchmark needs it without a batch in hand, and on
    /// the same current-generation footing as the scored baseline above, so
    /// the measured difference stays the collector and nothing else.
    #[cfg(feature = "bench-internals")]
    #[doc(hidden)]
    pub fn hydrate_fusion_metadata_unscored_for_bench<'a>(
        &'a self,
        _cx: &'a Cx,
        results: &'a mut [ScoredResult],
    ) -> SearchFuture<'a, ()> {
        Box::pin(async move { self.hydrate_with_searcher(&self.reader.searcher(), results, true) })
    }
}

// ─── LexicalSearch implementation ───────────────────────────────────────────

impl frankensearch_core::traits::LexicalRead for TantivyIndex {
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

            let parsed = self.parse_query_shipping(query);

            let searcher = self.reader.searcher();
            let top_docs = search_guarded(
                &searcher,
                &*parsed,
                &TopDocs::with_limit(limit).order_by_score(),
            )?;

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

    /// Score candidates and pin the searcher that produced them.
    ///
    /// Tantivy's deferred-metadata path skips loading stored fields for the
    /// whole candidate pool and restores them only for the winners. The batch
    /// carries the exact [`Searcher`] used for scoring: a Tantivy searcher
    /// holds a fixed segment set, so it *is* the immutable generation handle
    /// this bead requires, and hydration through it cannot observe a commit
    /// that landed after scoring.
    fn search_candidates<'a>(
        &'a self,
        cx: &'a Cx,
        query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, LexicalCandidateBatch> {
        Box::pin(async move {
            let truncated = Self::truncate_query(query);
            // One searcher for scoring AND hydration. Taking a second one at
            // hydration time is exactly the generation race this pins shut.
            let searcher = self.reader.searcher();
            // An empty query or a zero limit still returns a *pinned* batch
            // rather than an eager one. Keeping the shape uniform means a
            // `None` context always signals caller misuse, so hydration can
            // reject it without having to guess whether the batch was legitimately
            // eager.
            let hits = if truncated.trim().is_empty() || limit == 0 {
                Vec::new()
            } else {
                let parsed = self.parse_query_shipping(truncated);
                execute_top_k(&searcher, &*parsed, limit, 0)?
            };
            let candidates = self
                .collect_id_hits(&searcher, hits)?
                .into_iter()
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
                .collect();
            let _ = cx;
            Ok(LexicalCandidateBatch::deferred(
                candidates,
                LexicalHydrationContext::new("tantivy", Box::new(searcher)),
            ))
        })
    }

    /// Restore winner metadata from the pinned scoring searcher.
    ///
    /// Rejects a missing or foreign context with a typed error rather than
    /// silently reading the current generation.
    fn hydrate_candidates<'a>(
        &'a self,
        _cx: &'a Cx,
        context: Option<&'a LexicalHydrationContext>,
        results: &'a mut [ScoredResult],
    ) -> SearchFuture<'a, ()> {
        Box::pin(async move {
            let Some(context) = context else {
                return Err(SearchError::SubsystemError {
                    subsystem: "tantivy.hydration",
                    source: "deferred Tantivy candidates require their batch context; \
                             none was provided"
                        .into(),
                });
            };
            let Some(searcher) = context.downcast_ref::<Searcher>() else {
                return Err(SearchError::SubsystemError {
                    subsystem: "tantivy.hydration",
                    source: format!(
                        "hydration context from backend {:?} is not a Tantivy searcher pin; \
                         refusing cross-engine hydration",
                        context.backend()
                    )
                    .into(),
                });
            };
            self.hydrate_with_searcher(searcher, results, true)
        })
    }

    fn doc_count(&self) -> SearchResult<usize> {
        Self::checked_searcher_doc_count(&self.reader.searcher())
    }
}

impl frankensearch_core::traits::LexicalWrite for TantivyIndex {
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
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    // The split capabilities, so `idx.search(..)` / `idx.doc_count()` resolve
    // now that the combined trait is gone. B1 deliberately kept these out of
    // module scope to avoid method ambiguity while both surfaces coexisted.
    use frankensearch_core::traits::{LexicalRead, LexicalWrite};
    use frankensearch_core::types::IndexableDocument;

    /// Helper: run async test code with a `Cx`.
    fn run_with_cx<F, Fut>(f: F)
    where
        F: FnOnce(Cx) -> Fut,
        Fut: Future<Output = ()>,
    {
        asupersync::test_utils::run_test_with_cx(f);
    }

    /// bd-f20ye: BOOSTING A GROUP CHANGES ITS BOOLEAN MEANING in tantivy
    /// 0.26.1 whenever the group contains a negation. Pinned, not fixed.
    ///
    /// The bead was filed — by me — claiming the lenient-parse fallback "drops"
    /// the negation, which is what the Divergence Register had said about this
    /// shape since DIV-007. Probing the parser falsified all of it: nothing is
    /// dropped, the `MustNot` clause is present in the parsed query, strict and
    /// lenient parses agree, and `parse_query_lenient` reports NO errors. The
    /// mechanism is structural.
    ///
    /// An unboosted group lowers its negation as a `MustNot` clause OF the
    /// enclosing boolean, which excludes. A BOOSTED group instead nests the
    /// negation in its own `BooleanQuery { [(MustNot, ...)], msm: 0 }` and
    /// attaches THAT as a clause of the outer boolean — so a matcher meaning
    /// "every document except B" becomes a positive alternative, and the group
    /// stops meaning what it read as.
    ///
    /// Both directions are wrong, which is why this is pinned as behaviour
    /// rather than reasoned about:
    ///
    ///   (alpha NOT beta)        [p2]      correct exclusion
    ///   (alpha NOT beta)^2      [p1, p2]  the excluded document comes back
    ///   (alpha AND NOT beta)^2  []        and the AND form loses everything
    ///
    /// NOT FIXED HERE, deliberately. `frankensearch-lexical` is simultaneously
    /// the shipping tantivy backend and the gauntlet's PINNED CONFORMANCE
    /// ORACLE. Rewriting queries here to restore the intuitive meaning would
    /// silently move the conformance target that Quill is measured against —
    /// the exact shape of "weaken the oracle to make the comparison pass". Quill
    /// executes these shapes correctly, so the divergence is registered
    /// (DIV-009, blocking on bd-f20ye) and this test exists so that a tantivy
    /// upgrade which changes the behaviour becomes VISIBLE instead of quietly
    /// shifting that target. Same posture as bd-nqeb4's `#[should_panic]` pin.
    /// bd-eeq0q: `A AND NOT B` returns A-minus-B on the SHIPPING path, while
    /// the ORACLE path still reproduces tantivy 0.26.1's empty result.
    ///
    /// PLANTED NEGATIVES RUN BOTH DIRECTIONS, because this repair can fail two
    /// opposite ways and only one of them is obvious. If the repair stops
    /// working, the shipping assertions go red. If it LEAKS into the oracle
    /// surface, the oracle assertions go red — that direction matters more,
    /// since a repaired oracle would silently move the conformance target Quill
    /// is measured against, which is the "weaken the oracle to make the
    /// comparison pass" failure this crate's two-role split exists to prevent.
    ///
    /// The quoted control is the third direction: a literal `"a AND NOT b"`
    /// phrase must never be rewritten, or the repair would corrupt user text
    /// rather than user syntax.
    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn and_not_returns_a_minus_b_on_shipping_while_the_oracle_stays_bit_faithful() {
        run_with_cx(|cx| async move {
            let index = TantivyIndex::in_memory().expect("bd-eeq0q index");
            index
                .index_documents(
                    &cx,
                    &[
                        IndexableDocument::new("p1", "alpha beta"),
                        IndexableDocument::new("p2", "alpha gamma"),
                    ],
                )
                .await
                .expect("index bd-eeq0q documents");
            index.commit(&cx).await.expect("commit bd-eeq0q documents");

            let ship = |query: &'static str| {
                let index = &index;
                let cx = &cx;
                async move {
                    let mut ids = index
                        .search(cx, query, 10)
                        .await
                        .expect("bd-eeq0q shipping search")
                        .into_iter()
                        .map(|hit| hit.doc_id)
                        .collect::<Vec<_>>();
                    ids.sort();
                    ids
                }
            };

            // Every spelling of "alpha but not beta" agrees on the shipping
            // path. The first two already worked; the rest are the repair.
            for query in [
                "alpha NOT beta",
                "alpha -beta",
                "alpha AND NOT beta",
                "(alpha AND NOT beta)",
                "(alpha AND NOT beta)^2",
            ] {
                assert_eq!(
                    ship(query).await,
                    vec!["p2".to_owned()],
                    "shipping {query:?} must return A-minus-B"
                );
            }

            // Repeated exclusion still excludes, and does not resurrect p1.
            assert_eq!(
                ship("alpha AND NOT beta AND NOT delta").await,
                vec!["p2".to_owned()],
                "a second AND NOT must keep excluding"
            );

            // A literal phrase is user TEXT, not user SYNTAX: never rewritten.
            assert!(
                ship("\"alpha AND NOT beta\"").await.is_empty(),
                "a quoted phrase must not be rewritten into an operator form"
            );

            // THE ORACLE HAS NOT MOVED. Same index, same queries, through the
            // conformance surface: tantivy 0.26.1's behaviour, defect included.
            let oracle_ids = |query: &'static str| {
                let mut ids = index
                    .oracle_observe_query(&cx, query, 10, 64, &SnippetConfig::default())
                    .expect("bd-eeq0q oracle observation")
                    .hits
                    .into_iter()
                    .map(|hit| hit.doc_id)
                    .collect::<Vec<_>>();
                ids.sort();
                ids
            };
            let oracle_empty = oracle_ids("alpha AND NOT beta");
            assert!(
                oracle_empty.is_empty(),
                "the oracle must still reproduce the tantivy 0.26.1 defect, got {oracle_empty:?}"
            );
            assert!(
                oracle_ids("(alpha AND NOT beta)^2").is_empty(),
                "the boosted form must stay defective on the oracle surface too"
            );
            assert_eq!(
                oracle_ids("alpha NOT beta"),
                vec!["p2".to_owned()],
                "the oracle's own correct spelling must be unaffected"
            );
        });
    }

    /// bd-8a2a8: the normal form the repair emits, asserted as TEXT.
    ///
    /// The end-to-end test below proves the answers; this one pins the shape,
    /// because two of the properties that make the repair safe are invisible in
    /// a result set. `(a AND NOT b)^2` must keep its `-` DIRECTLY inside the
    /// boosted group — `repair_boosted_group_negation` records negation against
    /// the innermost open group, so an extra layer of parentheses would silently
    /// un-repair DIV-009 while every membership assertion stayed green. And a
    /// query with nothing to repair must come back BORROWED, which is both the
    /// allocation contract and a check that the scanner declines rather than
    /// guesses on input it cannot decode.
    #[test]
    fn a_negated_conjunction_normalises_to_explicit_occurrences() {
        for (query, expected) in [
            // The reported defect: `AND` binds tighter than the default join.
            // Operand ORDER is preserved; only the occurrences become explicit.
            ("alpha NOT beta AND gamma", "alpha (-beta +gamma)"),
            // A chain that IS the level keeps its operand order and adds no
            // parentheses.
            ("alpha AND NOT beta AND gamma", "+alpha -beta +gamma"),
            ("gamma AND NOT beta", "+gamma -beta"),
            ("NOT beta AND gamma", "-beta +gamma"),
            // DIV-009 interlock: the `-` stays directly inside the boost.
            ("(alpha AND NOT beta)^2", "(+alpha -beta)^2"),
            // Operands are carried through whole: phrases, fields, boosts, and
            // a fielded group that must not be split at its parenthesis.
            ("alpha AND NOT \"beta gamma\"", "+alpha -\"beta gamma\""),
            ("content:alpha^2 AND NOT beta", "+content:alpha^2 -beta"),
            (
                "field:(alpha beta) AND NOT gamma",
                "+field:(alpha beta) -gamma",
            ),
            // Occurrence outside the chain survives: a `+` that is dropped
            // turns a Must clause into a Should one.
            ("+alpha delta AND NOT beta", "+alpha (+delta -beta)"),
            // Recursion into a group, with the outer level left alone.
            (
                "(alpha NOT beta AND gamma) OR delta",
                "(alpha (-beta +gamma)) OR delta",
            ),
            // Stacked negators lower to one exclusion.
            ("alpha NOT -beta AND gamma", "alpha (-beta +gamma)"),
            // bd-iiidv: an ALL-NEGATIVE chain normalises too, in either operand
            // order, and the normal form carries no `AND` -- which is why the
            // repair already covered this shape while the register recorded it
            // as agreement. There is no `+` to add: every operand is excluded,
            // and the complement anchoring is the parser's job, not this
            // rewrite's.
            ("NOT beta AND NOT gamma", "-beta -gamma"),
            ("NOT gamma AND NOT beta", "-gamma -beta"),
        ] {
            assert_eq!(
                repair_negated_conjunction(query).as_ref(),
                expected,
                "normal form for {query:?}"
            );
        }

        // PLANTED NEGATIVES: everything here must be returned untouched, and
        // `Cow::Borrowed` proves the scanner did not merely round-trip it.
        for untouched in [
            // No explicit `AND`: nothing to normalise.
            "alpha NOT beta",
            "alpha -beta",
            "alpha beta",
            // A pure-positive conjunction is not defective; leave the spelling.
            "alpha AND gamma",
            "alpha AND beta AND gamma",
            // User TEXT, not user syntax.
            "\"alpha AND NOT beta\"",
            // Lowercase `and` is a pair of ordinary terms in this grammar.
            "alpha and not beta",
            // Undecodable input is declined, never guessed at.
            "alpha AND",
            "AND NOT beta",
            "alpha AND OR beta",
            "alpha AND \"unbalanced",
        ] {
            assert!(
                matches!(repair_negated_conjunction(untouched), Cow::Borrowed(_)),
                "{untouched:?} must be left untouched, got {:?}",
                repair_negated_conjunction(untouched)
            );
        }
    }

    /// bd-8a2a8: `A NOT B AND C` returns the declared reading on the SHIPPING
    /// path, while the ORACLE path still reproduces tantivy 0.26.1's answer.
    ///
    /// THE DECLARED READING is `default join := OR; explicit AND has precedence
    /// over OR` (`docs/contracts/quill-language-contract.md`), so
    /// `alpha NOT beta AND gamma` is `alpha OR (gamma AND NOT beta)`. Tantivy
    /// 0.26.1 lowers the conjunct's negation into a positive-less boolean, so
    /// the conjunct matches nothing and the whole `AND` clause disappears —
    /// `p3` is the document that only the correct reading returns.
    ///
    /// THE ATTRIBUTION CONTROL is the same query with its grouping written out.
    /// `alpha ((gamma) NOT beta)` carries no `AND`, is therefore untouched by
    /// every repair, and this engine answers it CORRECTLY on both roles. So the
    /// engine already agrees with the declared reading when the grouping is
    /// explicit, which is what places the defect in the implicit lowering
    /// rather than in either side's semantics.
    ///
    /// PLANTED NEGATIVES RUN THREE DIRECTIONS. If the repair stops working the
    /// shipping assertions go red; if it LEAKS into the oracle surface the
    /// oracle assertions go red, which is the direction that matters more,
    /// because a repaired oracle silently moves the conformance target Quill is
    /// measured against. The third is the repair ORDER: `repair_and_not` deletes
    /// the `AND` in front of a `NOT`, so if it ran first, `alpha AND NOT beta
    /// AND gamma` would arrive here as the unrelated chain `NOT beta AND gamma`
    /// and normalise to `alpha (+gamma -beta)` — a disjunction returning three
    /// documents where the conjunction returns one.
    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn a_negated_conjunction_reads_as_a_disjunct_on_shipping_while_the_oracle_stays_bit_faithful() {
        run_with_cx(|cx| async move {
            let index = TantivyIndex::in_memory().expect("bd-8a2a8 index");
            index
                .index_documents(
                    &cx,
                    &[
                        IndexableDocument::new("p1", "alpha beta"),
                        IndexableDocument::new("p2", "alpha gamma"),
                        IndexableDocument::new("p3", "gamma"),
                        IndexableDocument::new("p4", "beta gamma"),
                        IndexableDocument::new("p5", "delta"),
                    ],
                )
                .await
                .expect("index bd-8a2a8 documents");
            index.commit(&cx).await.expect("commit bd-8a2a8 documents");

            let ship = |query: &'static str| {
                let index = &index;
                let cx = &cx;
                async move {
                    let mut ids = index
                        .search(cx, query, 10)
                        .await
                        .expect("bd-8a2a8 shipping search")
                        .into_iter()
                        .map(|hit| hit.doc_id)
                        .collect::<Vec<_>>();
                    ids.sort();
                    ids
                }
            };
            let ids =
                |wanted: &[&str]| wanted.iter().map(|id| (*id).to_owned()).collect::<Vec<_>>();

            // THE DEFECT, and the same reading spelled three other ways. `p3`
            // matches `gamma` without `beta`, so it appears only when the `AND`
            // conjunct survives.
            for query in [
                "alpha NOT beta AND gamma",
                "alpha NOT -beta AND gamma",
                "alpha ((gamma) NOT beta)",
                "alpha (gamma AND NOT beta)",
            ] {
                assert_eq!(
                    ship(query).await,
                    ids(&["p1", "p2", "p3"]),
                    "shipping {query:?} must return alpha OR (gamma minus beta)"
                );
            }

            // THE CHAIN STILL BINDS AS A CONJUNCTION. This is the repair-order
            // negative: a disjunctive reading here would return three documents.
            assert_eq!(
                ship("alpha AND NOT beta AND gamma").await,
                ids(&["p2"]),
                "a single AND chain must stay a conjunction"
            );
            for query in ["gamma AND NOT beta", "NOT beta AND gamma"] {
                assert_eq!(
                    ship(query).await,
                    ids(&["p2", "p3"]),
                    "shipping {query:?} must return gamma minus beta"
                );
            }

            // The contract's all-documents literal is an operand like any
            // other, and must survive being given an explicit occurrence.
            assert_eq!(
                ship("* AND NOT beta").await,
                ids(&["p2", "p3", "p5"]),
                "every document except the excluded ones"
            );

            // Controls: a pure-positive conjunction is unchanged, an explicit
            // `+` still means Must, and a quoted phrase is text.
            assert_eq!(ship("alpha AND gamma").await, ids(&["p2"]));
            assert_eq!(
                ship("+alpha delta AND NOT beta").await,
                ids(&["p1", "p2"]),
                "a required clause must not be demoted to optional"
            );
            assert_eq!(
                ship("alpha delta AND NOT beta").await,
                ids(&["p1", "p2", "p5"]),
                "without the `+` the same query is a disjunction"
            );
            assert!(
                ship("\"alpha NOT beta AND gamma\"").await.is_empty(),
                "a quoted phrase must not be rewritten into an operator form"
            );

            // THE ORACLE HAS NOT MOVED. Same index, same queries, through the
            // conformance surface: tantivy 0.26.1's behaviour, defect included.
            let oracle_ids = |query: &'static str| {
                let mut ids = index
                    .oracle_observe_query(&cx, query, 10, 64, &SnippetConfig::default())
                    .expect("bd-8a2a8 oracle observation")
                    .hits
                    .into_iter()
                    .map(|hit| hit.doc_id)
                    .collect::<Vec<_>>();
                ids.sort();
                ids
            };
            assert_eq!(
                oracle_ids("alpha NOT beta AND gamma"),
                ids(&["p1", "p2"]),
                "the oracle must still drop the AND conjunct entirely"
            );
            assert!(
                oracle_ids("gamma AND NOT beta").is_empty(),
                "the oracle must still return nothing for a bare negated conjunction"
            );
            assert_eq!(
                oracle_ids("alpha ((gamma) NOT beta)"),
                ids(&["p1", "p2", "p3"]),
                "the explicit grouping is answered correctly by BOTH roles, which \
                 is what attributes the defect to the implicit lowering"
            );
        });
    }

    /// bd-iiidv: an ALL-NEGATIVE conjunction is DIV-010's defect too, and the
    /// register said it was agreement.
    ///
    /// The Divergence Register's DIV-010 entry claimed `NOT alpha AND NOT beta`
    /// "still returns nothing on both, because an exclusion-only conjunction
    /// has no positive term — that is agreement, not a defect, and is pinned so
    /// a later change cannot silently turn it into match-all". Measured, it is
    /// not agreement and there was no such pin. This is the pin.
    ///
    /// The claim is wrong twice over. An all-negative ROOT does not match
    /// nothing — both engines give it complement semantics for every spelling
    /// that carries no explicit `AND` — and the `AND` spelling does not agree:
    /// the pinned oracle empties it while the declared reading, Quill and the
    /// repaired shipping path all return the complement.
    ///
    /// It is the SAME mechanism as the rest of DIV-010, not a new one: an
    /// explicit `AND` whose operand is a negation lowers to a positive-less
    /// boolean that matches nothing and empties the conjunction. That is why
    /// `repair_negated_conjunction` already fixes it — the normal form carries
    /// no `AND` — and why the fix needed no new repair, only this proof that it
    /// covers the shape.
    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn an_all_negative_conjunction_is_complement_on_shipping_and_empty_on_the_oracle() {
        run_with_cx(|cx| async move {
            let index = TantivyIndex::in_memory().expect("bd-iiidv index");
            index
                .index_documents(
                    &cx,
                    &[
                        IndexableDocument::new("p1", "alpha beta"),
                        IndexableDocument::new("p2", "alpha gamma"),
                        IndexableDocument::new("p3", "delta gamma"),
                        IndexableDocument::new("p4", "epsilon"),
                    ],
                )
                .await
                .expect("index bd-iiidv documents");
            index.commit(&cx).await.expect("commit bd-iiidv documents");

            let shipping = |query: &'static str| {
                let index = &index;
                let cx = &cx;
                async move {
                    let mut ids = index
                        .search(cx, query, 10)
                        .await
                        .expect("bd-iiidv search")
                        .into_iter()
                        .map(|hit| hit.doc_id)
                        .collect::<Vec<_>>();
                    ids.sort();
                    ids
                }
            };
            let oracle = |query: &'static str| {
                let index = &index;
                let cx = &cx;
                async move {
                    let mut ids = index
                        .oracle_observe_query(cx, query, 10, 64, &SnippetConfig::default())
                        .expect("bd-iiidv oracle observation")
                        .hits
                        .into_iter()
                        .map(|hit| hit.doc_id)
                        .collect::<Vec<_>>();
                    ids.sort();
                    ids
                }
            };

            // THE DEFECT, in both operand orders. `p4` is the only document
            // matching neither term, so the complement is exactly `[p4]` and an
            // empty answer is not "no positive term" -- it is the conjunct
            // being emptied.
            for query in ["NOT beta AND NOT gamma", "NOT gamma AND NOT beta"] {
                assert_eq!(
                    oracle(query).await,
                    Vec::<String>::new(),
                    "the pinned oracle must still empty an all-negative conjunction: {query}"
                );
                assert_eq!(
                    shipping(query).await,
                    vec!["p4".to_owned()],
                    "the shipping path must answer the complement: {query}"
                );
            }

            // THE ATTRIBUTION, and the refutation of the "agreement" claim in
            // one measurement: the SAME oracle answers every spelling of the
            // same intent that carries no explicit `AND` with the complement.
            // An engine that gives an all-negative root complement semantics
            // whenever the conjunction is spelled another way is mislowering
            // the `AND` form, not asserting that exclusion-only matches
            // nothing.
            for query in ["NOT beta NOT gamma", "-beta -gamma", "NOT beta -gamma"] {
                assert_eq!(
                    oracle(query).await,
                    vec!["p4".to_owned()],
                    "the oracle gives an all-negative root complement semantics here: {query}"
                );
                assert_eq!(
                    shipping(query).await,
                    vec!["p4".to_owned()],
                    "and the shipping path agrees, so these spellings are not divergent: {query}"
                );
            }

            // NOT match-all, which is the failure the register's own sentence
            // was worried about: a single negation still excludes, and the
            // complement of a term that matches nothing is still bounded by the
            // corpus.
            assert_eq!(
                shipping("NOT beta").await,
                vec!["p2".to_owned(), "p3".to_owned(), "p4".to_owned()]
            );
            assert_eq!(
                oracle("NOT beta").await,
                vec!["p2".to_owned(), "p3".to_owned(), "p4".to_owned()]
            );
        });
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn boosting_a_group_that_negates_changes_its_meaning_in_the_pinned_oracle() {
        run_with_cx(|cx| async move {
            let index = TantivyIndex::in_memory().expect("bd-f20ye index");
            index
                .index_documents(
                    &cx,
                    &[
                        IndexableDocument::new("p1", "alpha beta"),
                        IndexableDocument::new("p2", "alpha gamma"),
                    ],
                )
                .await
                .expect("index bd-f20ye documents");
            index.commit(&cx).await.expect("commit bd-f20ye documents");

            // The SHIPPING role, through the public search surface.
            let hits = |query: &'static str| {
                let index = &index;
                let cx = &cx;
                async move {
                    let mut ids = index
                        .search(cx, query, 10)
                        .await
                        .expect("bd-f20ye search")
                        .into_iter()
                        .map(|hit| hit.doc_id)
                        .collect::<Vec<_>>();
                    ids.sort();
                    ids
                }
            };
            // The ORACLE role, through the conformance observation surface.
            let oracle_hits = |query: &'static str| {
                let index = &index;
                let cx = &cx;
                async move {
                    let mut ids = index
                        .oracle_observe_query(cx, query, 10, 64, &SnippetConfig::default())
                        .expect("bd-f20ye oracle observation")
                        .hits
                        .into_iter()
                        .map(|hit| hit.doc_id)
                        .collect::<Vec<_>>();
                    ids.sort();
                    ids
                }
            };

            // The negation is NOT dropped and recovery is NOT involved: the
            // lenient parser reports no errors at all for the broken shape.
            let (_parsed, errors) = index
                .query_parser()
                .parse_query_lenient("(alpha NOT beta)^2");
            assert!(
                errors.is_empty(),
                "bd-f20ye is not a lenient-recovery defect; the parser reported {errors:?}"
            );

            // CORRECT, unboosted — both roles agree here.
            assert_eq!(hits("alpha NOT beta").await, vec!["p2".to_owned()]);
            assert_eq!(hits("(alpha NOT beta)").await, vec!["p2".to_owned()]);
            assert!(hits("(alpha NOT alpha)").await.is_empty());

            // The page-observation surface must ALSO stay bit-faithful: a
            // repair leaking into any oracle_observe_* method is what would
            // move the conformance target without a comparison going red.
            let page = index
                .oracle_observe_page(&cx, "(alpha NOT beta)^2", 10, 0)
                .expect("bd-f20ye oracle page observation");
            assert_eq!(
                page.hits.len(),
                2,
                "the oracle page surface must reproduce the defect, not the repair"
            );

            // STILL WRONG IN THE ORACLE ROLE, and that is the point: the
            // comparator must reproduce Tantivy 0.26.1 byte for byte, defects
            // included, or the conformance target moves under Quill.
            assert_eq!(
                oracle_hits("(alpha NOT beta)^2").await,
                vec!["p1".to_owned(), "p2".to_owned()],
                "if this is now [p2], tantivy fixed the boosted-group negation and DIV-009 \
                 must be re-measured before it is retired"
            );
            assert_eq!(
                oracle_hits("(alpha NOT alpha)^2").await,
                vec!["p1".to_owned(), "p2".to_owned()],
                "if this is now empty, tantivy fixed the boosted-group negation"
            );
            assert!(
                oracle_hits("(alpha AND NOT beta)^2").await.is_empty(),
                "if this is now [p2], tantivy fixed the boosted-group negation"
            );

            // REPAIRED IN THE SHIPPING ROLE (bd-f20ye owner ruling): a boost
            // must never change which documents match.
            assert_eq!(
                hits("(alpha NOT beta)^2").await,
                vec!["p2".to_owned()],
                "the shipping path must not return a document the query excluded"
            );
            assert!(
                hits("(alpha NOT alpha)^2").await.is_empty(),
                "a self-contradictory group must match nothing however it is boosted"
            );
            // THE INVARIANT THE RULING ASKS FOR, stated as an invariant rather
            // than as hardcoded expectations: a boost must never change WHICH
            // documents match. Asserting `boosted == unboosted` says exactly
            // that, and stays true whatever the unboosted form happens to
            // return.
            for group in [
                "(alpha NOT beta)",
                "(alpha NOT alpha)",
                "(alpha AND NOT beta)",
                "(alpha -beta)",
                "(alpha OR beta)",
            ] {
                let boosted = format!("{group}^2");
                assert_eq!(
                    hits(Box::leak(boosted.into_boxed_str())).await,
                    hits(Box::leak(group.to_owned().into_boxed_str())).await,
                    "boosting {group} changed which documents match in the shipping path"
                );
            }
            // SUPERSEDED BY bd-eeq0q, deliberately flipped rather than deleted.
            // When this test landed, `A AND NOT B` returned nothing on the
            // shipping path too — a SEPARATE defect from the boosted-group one,
            // filed as bd-eeq0q and repaired there by `repair_and_not`. This
            // assertion recorded the defect as it stood; it now records the
            // repair. The invariant ABOVE is what belongs to this bead and is
            // unchanged: a boost may not alter membership. Keeping the line
            // (rather than dropping it) preserves the fact that the two defects
            // were separable and were separated.
            assert_eq!(
                hits("alpha AND NOT beta").await,
                vec!["p2".to_owned()],
                "bd-eeq0q repairs the ungrouped form on the shipping path"
            );
            // NESTED, so the repair is not merely a whole-query special case.
            assert_eq!(
                hits("beta OR (alpha NOT beta)^2").await,
                vec!["p1".to_owned(), "p2".to_owned()]
            );
            assert_eq!(
                hits("(alpha -beta)^3").await,
                vec!["p2".to_owned()],
                "the `-term` exclusion form must be repaired too"
            );

            // CONTROL: a boosted group WITHOUT a negation keeps its meaning in
            // BOTH roles, so the repair is specific to negation and does not
            // quietly disarm boosts in general.
            for query in ["(alpha OR beta)^2", "(beta)^2"] {
                assert_eq!(
                    hits(query).await,
                    oracle_hits(query).await,
                    "{query} must be identical in both roles"
                );
            }
            assert_eq!(
                hits("(alpha OR beta)^2").await,
                vec!["p1".to_owned(), "p2".to_owned()]
            );
            assert_eq!(hits("(beta)^2").await, vec!["p1".to_owned()]);
        });
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

    fn shared_full120_docs() -> Vec<IndexableDocument> {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../../tests/fixtures/corpus.json"))
                .expect("parse committed Full120 corpus");
        let documents = fixture["documents"]
            .as_array()
            .expect("fixture documents array");
        assert_eq!(documents.len(), 120, "Full120 fixture count drifted");
        documents
            .iter()
            .enumerate()
            .map(|(ordinal, document)| {
                let mut metadata = document["metadata"]
                    .as_object()
                    .expect("fixture metadata")
                    .iter()
                    .map(|(key, value)| {
                        (
                            key.clone(),
                            value
                                .as_str()
                                .map_or_else(|| value.to_string(), str::to_owned),
                        )
                    })
                    .collect::<std::collections::HashMap<_, _>>();
                metadata.insert(
                    "created_at".to_owned(),
                    document["created_at"]
                        .as_str()
                        .expect("fixture created_at")
                        .to_owned(),
                );
                metadata.insert(
                    "doc_type".to_owned(),
                    document["doc_type"]
                        .as_str()
                        .expect("fixture doc_type")
                        .to_owned(),
                );
                metadata.insert("created_at_ms".to_owned(), ordinal.to_string());
                IndexableDocument {
                    id: document["doc_id"]
                        .as_str()
                        .expect("fixture doc_id")
                        .to_owned(),
                    content: document["content"]
                        .as_str()
                        .expect("fixture content")
                        .to_owned(),
                    title: Some(
                        document["title"]
                            .as_str()
                            .expect("fixture title")
                            .to_owned(),
                    ),
                    metadata,
                }
            })
            .collect()
    }

    fn optional_score_bits(score: Option<f32>) -> Option<u32> {
        score.map(f32::to_bits)
    }

    fn assert_scored_result_contract(
        actual: &ScoredResult,
        expected: &ScoredResult,
        context: &str,
    ) {
        assert_eq!(actual.doc_id, expected.doc_id, "doc_id: {context}");
        assert_eq!(
            actual.score.to_bits(),
            expected.score.to_bits(),
            "score: {context}"
        );
        assert_eq!(actual.source, expected.source, "source: {context}");
        assert_eq!(actual.index, expected.index, "index: {context}");
        assert_eq!(
            optional_score_bits(actual.fast_score),
            optional_score_bits(expected.fast_score),
            "fast score: {context}"
        );
        assert_eq!(
            optional_score_bits(actual.quality_score),
            optional_score_bits(expected.quality_score),
            "quality score: {context}"
        );
        assert_eq!(
            optional_score_bits(actual.lexical_score),
            optional_score_bits(expected.lexical_score),
            "lexical score: {context}"
        );
        assert_eq!(
            optional_score_bits(actual.rerank_score),
            optional_score_bits(expected.rerank_score),
            "rerank score: {context}"
        );
        assert_eq!(
            serde_json::to_value(&actual.explanation).expect("serialize actual explanation"),
            serde_json::to_value(&expected.explanation).expect("serialize expected explanation"),
            "explanation: {context}"
        );
        assert_eq!(
            actual.metadata.as_deref(),
            expected.metadata.as_deref(),
            "metadata: {context}"
        );
    }

    async fn assert_live_lexical_contract(
        idx: &TantivyIndex,
        cx: &Cx,
        query: &str,
        limit: usize,
        expected_total_count: usize,
    ) {
        let context = format!("query={query:?}, limit={limit}");
        let full = idx.search(cx, query, limit).await.expect("full search");
        let id_hits = idx
            .search_doc_ids(cx, query, limit)
            .expect("ord-table ID search");
        let docstore_hits = idx
            .search_doc_ids_via_docstore(cx, query, limit)
            .expect("docstore ID search");
        let candidate_batch = idx
            .search_candidates(cx, query, limit)
            .await
            .expect("fusion candidate search");
        let (mut candidates, candidate_pin) = candidate_batch.into_parts();

        assert_eq!(
            id_hits, docstore_hits,
            "ordinal table must independently resolve the same IDs as stored documents: {context}"
        );
        assert_eq!(full.len(), id_hits.len(), "full/id length: {context}");
        assert_eq!(
            full.len(),
            candidates.len(),
            "full/candidate length: {context}"
        );
        assert_eq!(
            full.len(),
            limit.min(expected_total_count),
            "result count: {context}"
        );

        for (rank, ((expected, id_hit), candidate)) in
            full.iter().zip(&id_hits).zip(&candidates).enumerate()
        {
            let row_context = format!("{context}, rank={rank}");
            assert_eq!(id_hit.rank, rank, "ID rank: {row_context}");
            assert_eq!(id_hit.doc_id, expected.doc_id, "ID doc_id: {row_context}");
            assert_eq!(
                id_hit.bm25_score.to_bits(),
                expected.score.to_bits(),
                "ID score: {row_context}"
            );
            assert_eq!(
                candidate.doc_id, expected.doc_id,
                "candidate doc_id: {row_context}"
            );
            assert_eq!(
                candidate.score.to_bits(),
                expected.score.to_bits(),
                "candidate score: {row_context}"
            );
            assert_eq!(
                optional_score_bits(candidate.lexical_score),
                optional_score_bits(expected.lexical_score),
                "candidate lexical score: {row_context}"
            );
            assert_eq!(candidate.source, expected.source, "source: {row_context}");
            assert_eq!(candidate.index, expected.index, "index: {row_context}");
            assert_eq!(
                optional_score_bits(candidate.fast_score),
                optional_score_bits(expected.fast_score),
                "fast score: {row_context}"
            );
            assert_eq!(
                optional_score_bits(candidate.quality_score),
                optional_score_bits(expected.quality_score),
                "quality score: {row_context}"
            );
            assert_eq!(
                optional_score_bits(candidate.rerank_score),
                optional_score_bits(expected.rerank_score),
                "rerank score: {row_context}"
            );
            assert_eq!(
                serde_json::to_value(&candidate.explanation)
                    .expect("serialize candidate explanation"),
                serde_json::to_value(&expected.explanation)
                    .expect("serialize expected explanation"),
                "explanation: {row_context}"
            );
            assert!(
                candidate.metadata.is_none(),
                "candidate metadata must be deferred: {row_context}"
            );
        }

        idx.hydrate_candidates(cx, candidate_pin.as_ref(), &mut candidates)
            .await
            .expect("hydrate fusion metadata");
        for (rank, (candidate, expected)) in candidates.iter().zip(&full).enumerate() {
            assert_scored_result_contract(
                candidate,
                expected,
                &format!("{context}, hydrated rank={rank}"),
            );
        }

        let parsed = idx.parse_query_lenient(query);
        let counted = execute_query_with_offset(&idx.reader.searcher(), &*parsed, limit, 0)
            .expect("exact counted search");
        assert_eq!(
            counted.total_count, expected_total_count,
            "Count must remain exact and independent of k: {context}"
        );
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
        assert_eq!(idx.doc_count().expect("document count"), 0);
    }

    #[test]
    fn create_on_disk() {
        let dir = tempfile::tempdir().expect("tempdir");
        let idx = TantivyIndex::create(dir.path()).expect("create");
        assert_eq!(idx.doc_count().expect("document count"), 0);
        assert_eq!(idx.path(), Some(dir.path()));
    }

    #[test]
    fn open_nonexistent_returns_error() {
        let result = TantivyIndex::open(Path::new("/nonexistent/tantivy_index_xyz"));
        assert!(result.is_err());
    }

    /// Pins the ORACLE's behavior for stacked unary prefixes (`NOT -x`,
    /// `NOT NOT x`), which sit OUTSIDE the contract grammar's single-prefix
    /// fragment rule (`docs/contracts/quill-language-contract.md` §"fragment").
    /// The pinned Tantivy 0.26.1 parser is the normative behavior for such
    /// sequences; Quill's lowering must reproduce whatever this test observes
    /// (bd-251nt — found by the bd-bsjw structure-aware campaign).
    #[test]
    fn stacked_negation_prefixes_pin_oracle_semantics() {
        run_with_cx(|cx| async move {
            let docs = vec![
                IndexableDocument::new("doc-a", "alpha need"),
                IndexableDocument::new("doc-b", "alpha other"),
                IndexableDocument::new("doc-c", "need only"),
                IndexableDocument::new("doc-d", "plain filler"),
            ];
            let idx = TantivyIndex::in_memory().expect("create");
            idx.index_documents(&cx, &docs).await.expect("index");
            idx.commit(&cx).await.expect("commit");

            let observe = |query: &str| {
                let mut ids = idx
                    .search_doc_ids(&cx, query, 10)
                    .expect("search")
                    .into_iter()
                    .map(|hit| hit.doc_id)
                    .collect::<Vec<_>>();
                ids.sort_unstable();
                ids
            };

            assert_eq!(
                observe("NOT -need"),
                vec!["doc-b".to_owned(), "doc-d".to_owned()],
                "oracle collapses NOT + '-' to a single exclusion of `need`"
            );
            assert_eq!(
                observe("NOT NOT need"),
                vec![
                    "doc-a".to_owned(),
                    "doc-b".to_owned(),
                    "doc-c".to_owned(),
                    "doc-d".to_owned(),
                ],
                "oracle treats NOT NOT as a lenient-parse match-all, NOT as \
                 logical double negation"
            );
            assert_eq!(
                observe("alpha NOT -need"),
                vec!["doc-b".to_owned()],
                "oracle: alpha-docs minus need-docs"
            );

            // In-GROUP stacked prefixes (bd-bsjw finding 4). `observe` here is
            // the SHIPPING surface (`search_doc_ids`), which bd-f20ye repaired,
            // so this now pins the repaired behaviour: a boost on the group no
            // longer changes membership.
            //
            // The original assertion recorded the opposite and explained it as
            // "the lenient fallback DROPS the negations". Measurement under
            // bd-f20ye falsified that account — nothing is dropped, the
            // `MustNot` survives parsing, and the lenient parser reports no
            // errors. The defect was that a boosted group re-nests its negation
            // as a positive alternative. The ORACLE-side pin of the unrepaired
            // behaviour, which is what this test's name is about, now lives in
            // `boosting_a_group_that_negates_changes_its_meaning_in_the_pinned_oracle`,
            // where it is observed through `oracle_observe_*` rather than
            // through a shipping method.
            assert_eq!(
                observe("(alpha NOT -need)"),
                vec!["doc-b".to_owned()],
                "unboosted group keeps the stacked-prefix collapse"
            );
            assert_eq!(
                observe("(alpha NOT -need)^2"),
                observe("(alpha NOT -need)"),
                "bd-f20ye: in the shipping path a boost must not change which \
                 documents match"
            );
            assert_eq!(
                observe("(alpha NOT need)^2"),
                observe("(alpha NOT need)"),
                "bd-f20ye: plain NOT under a group boost is repaired in the \
                 shipping path too, not just the stacked-prefix form"
            );
        });
    }

    /// Tantivy 0.26.1's `PhraseScorer` panics on an illegal post-termination
    /// seek when a NEGATED PHRASE rides beside a positive term — found by the
    /// bd-bsjw structure-aware campaign (`generic NOT "indexes Parser or
    /// minimal"`). Quill executes the same shape without incident. The
    /// shipping execution boundary converts the upstream panic into a typed
    /// degradation instead of aborting the host process, and the index stays
    /// fully usable afterwards (bd-nqeb4). When the pinned oracle is upgraded
    /// past the upstream defect, the typed-error assertion here fails —
    /// making the fix visible instead of silently shifting the target.
    #[test]
    fn oracle_negated_phrase_beside_term_degrades_typed_instead_of_panicking() {
        run_with_cx(|cx| async move {
            let docs = vec![
                IndexableDocument::new("doc-a", "alpha need only"),
                IndexableDocument::new("doc-b", "alpha other"),
                IndexableDocument::new("doc-c", "need only"),
            ];
            let idx = TantivyIndex::in_memory().expect("create");
            idx.index_documents(&cx, &docs).await.expect("index");
            idx.commit(&cx).await.expect("commit");
            // The phrase's terms exist but the SEQUENCE does not, so the
            // phrase docset terminates immediately — the precondition for the
            // upstream illegal seek.
            let error = idx
                .search_doc_ids(&cx, "alpha NOT \"only need\"", 10)
                .expect_err("upstream PhraseScorer defect must surface as a typed error");
            assert!(
                error
                    .to_string()
                    .contains("panicked during query execution"),
                "degradation must name the panic boundary: {error}"
            );

            // The availability property: the same index keeps serving ordinary
            // queries after the guarded failure.
            let survivors = idx
                .search_doc_ids(&cx, "alpha", 10)
                .expect("index must remain fully usable after a guarded panic");
            assert_eq!(survivors.len(), 2);
        });
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
            assert_eq!(idx.doc_count().expect("document count"), 1);
        });
    }

    #[test]
    fn index_batch_documents() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            let docs = sample_docs();
            idx.index_documents(&cx, &docs).await.expect("batch index");
            idx.commit(&cx).await.expect("commit");
            assert_eq!(idx.doc_count().expect("document count"), 5);
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

    #[cfg(feature = "bench-internals")]
    #[test]
    fn benchmark_schema_without_positions_uses_presence_only_postings() {
        let idx = TantivyIndex::in_memory_with_benchmark_config(50_000_000, 1, false)
            .expect("create position-free benchmark oracle");
        run_with_cx(|cx| async move {
            idx.index_documents(
                &cx,
                &[
                    IndexableDocument::new(
                        "repeated",
                        "term00001 term00001 term00001 term00001 term00001 qgpreflight",
                    ),
                    IndexableDocument::new("single", "term00001 qgpreflight"),
                    IndexableDocument::new("decoy", "term00002 qgpreflight"),
                ],
            )
            .await
            .expect("index position-free oracle fixture");
            idx.commit(&cx)
                .await
                .expect("commit position-free oracle fixture");

            let hits = idx
                .search(&cx, "term00001", 2)
                .await
                .expect("search position-free oracle fixture");
            let ids: Vec<_> = hits.iter().map(|hit| hit.doc_id.as_str()).collect();
            assert_eq!(ids, ["single", "repeated"]);
        });
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn benchmark_writer_join_rearms_without_changing_searchable_state() {
        let idx = TantivyIndex::in_memory_with_benchmark_config(50_000_000, 1, true)
            .expect("create benchmark oracle");
        run_with_cx(|cx| async move {
            idx.index_document(
                &cx,
                &IndexableDocument::new("before-join", "lifecycle fence"),
            )
            .await
            .expect("index pre-fence document");
            idx.commit(&cx).await.expect("commit pre-fence document");

            let (idx, receipt) = idx
                .benchmark_join_workers_and_rearm(50_000_000, 1)
                .expect("join workers and rearm writer");
            assert!(receipt.searchable_segments_before > 0);
            assert!(receipt.searchable_segments_after > 0);
            assert!(receipt.writer_rearmed);
            assert_eq!(
                idx.search(&cx, "lifecycle", 10)
                    .await
                    .expect("search pre-fence document")
                    .len(),
                1
            );

            idx.index_document(
                &cx,
                &IndexableDocument::new("after-join", "writer remains usable"),
            )
            .await
            .expect("index post-fence document");
            idx.commit(&cx).await.expect("commit post-fence document");
            assert_eq!(
                idx.search(&cx, "writer", 10)
                    .await
                    .expect("search post-fence document")
                    .len(),
                1
            );
        });
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn benchmark_writer_join_terminal_retains_searchable_reader_without_rearm() {
        let idx = TantivyIndex::in_memory_with_benchmark_config(50_000_000, 1, true)
            .expect("create benchmark oracle");
        run_with_cx(|cx| async move {
            idx.index_document(
                &cx,
                &IndexableDocument::new("terminal-join", "one shot bulk fixture"),
            )
            .await
            .expect("index terminal-fence document");
            idx.commit(&cx)
                .await
                .expect("commit terminal-fence document");

            let (reader, receipt) = idx
                .benchmark_join_workers_retaining_reader()
                .expect("join workers while retaining the terminal reader");
            assert!(receipt.searchable_segments_before > 0);
            assert!(receipt.searchable_segments_after > 0);
            assert!(!receipt.writer_rearmed);
            let tail_ids = reader
                .benchmark_search_exact_id("terminal-join")
                .expect("search the prepared terminal tail after worker join");
            assert_eq!(
                tail_ids.iter().map(DocId::as_str).collect::<Vec<_>>(),
                ["terminal-join"],
                "a counted segment is not accepted in place of a post-join tail search"
            );
            assert!(
                reader
                    .benchmark_search_exact_id("missing-terminal-tail")
                    .expect("search missing terminal tail after worker join")
                    .is_empty(),
                "planted negative: the retained reader must not fabricate tail visibility"
            );
        });
    }

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
            let batch = idx
                .search_candidates(&cx, "Rust", 10)
                .await
                .expect("fusion candidates");

            // `is_deferred` is the split-trait replacement for the retired
            // `fusion_metadata_is_deferred` capability flag.
            assert!(batch.is_deferred());
            let (mut candidates, pin) = batch.into_parts();
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

            idx.hydrate_candidates(&cx, pin.as_ref(), &mut candidates)
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

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn oracle_tie_expansion_preserves_the_shipping_top_k() {
        let idx = TantivyIndex::in_memory_single_threaded_oracle().expect("create oracle");
        run_with_cx(|cx| async move {
            idx.writer
                .lock(&cx)
                .await
                .expect("lock writer")
                .set_merge_policy(Box::new(tantivy::merge_policy::NoMergePolicy));

            const DOC_COUNT: usize = 256;
            const SEGMENT_SIZE: usize = 32;
            const LIMIT: usize = 100;
            const QUERY: &str = "term00001 term00007 generated record";
            for segment_start in (0..DOC_COUNT).step_by(SEGMENT_SIZE) {
                let docs = (segment_start..segment_start + SEGMENT_SIZE)
                    .map(|ordinal| {
                        IndexableDocument::new(
                            format!("tie-{ordinal:03}"),
                            "term00001 term00007 generated record",
                        )
                    })
                    .collect::<Vec<_>>();
                idx.index_documents(&cx, &docs)
                    .await
                    .expect("index tie segment");
                idx.commit(&cx).await.expect("commit tie segment");
            }

            let native = idx
                .search_doc_ids(&cx, QUERY, LIMIT)
                .expect("shipping top-k");
            let observed = idx
                .oracle_observe_query(
                    &cx,
                    QUERY,
                    LIMIT,
                    DOC_COUNT,
                    &SnippetConfig {
                        max_chars: 0,
                        ..SnippetConfig::default()
                    },
                )
                .expect("expanded oracle observation");
            let native_rows = native
                .iter()
                .map(|hit| (hit.doc_id.as_str(), hit.bm25_score.to_bits()))
                .collect::<Vec<_>>();
            let observed_rows = observed
                .hits
                .iter()
                .map(|hit| (hit.doc_id.as_str(), hit.score_bits))
                .collect::<Vec<_>>();

            assert_eq!(
                observed_rows, native_rows,
                "expanded tie evidence must not redefine the shipping top-k"
            );
            assert_eq!(observed.total_count, DOC_COUNT);
            assert!(observed.cutoff_tie_complete);
            assert_eq!(observed.cutoff_tie_group.len(), DOC_COUNT);
        });
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn oracle_counted_pages_keep_exact_totals_and_absolute_ranks() {
        let idx = TantivyIndex::in_memory_single_threaded_oracle().expect("create oracle");
        run_with_cx(|cx| async move {
            let docs = (0..9)
                .map(|ordinal| {
                    IndexableDocument::new(
                        format!("doc-{ordinal:02}"),
                        format!("shared counted-page term document {ordinal}"),
                    )
                })
                .collect::<Vec<_>>();
            idx.index_documents(&cx, &docs).await.expect("index");
            idx.commit(&cx).await.expect("commit");

            idx.index_document(
                &cx,
                &IndexableDocument::new("staged-doc", "shared counted-page term document staged"),
            )
            .await
            .expect("stage uncommitted document");

            let count_only = idx
                .oracle_observe_page(&cx, "counted-page", 0, 0)
                .expect("count-only page");
            let first = idx
                .oracle_observe_page(&cx, "counted-page", 3, 0)
                .expect("first page");
            let second = idx
                .oracle_observe_page(&cx, "counted-page", 3, 3)
                .expect("second page");
            let prefix = idx
                .oracle_observe_page(&cx, "counted-page", 6, 0)
                .expect("combined prefix");
            let past_end = idx
                .oracle_observe_page(&cx, "counted-page", 3, 100)
                .expect("past-end page");
            let ranked = idx
                .oracle_observe_query(&cx, "counted-page", 3, 16, &SnippetConfig::default())
                .expect("ranked observation");

            for page in [&count_only, &first, &second, &prefix, &past_end] {
                assert_eq!(page.total_count, 9, "exact Count must ignore k and offset");
                assert_eq!(
                    page.doc_count, 9,
                    "oracle document count must share the page searcher's generation"
                );
            }
            assert_eq!(ranked.total_count, 9);
            assert_eq!(
                ranked.doc_count, 9,
                "ranked oracle document count must ignore staged writer state"
            );
            assert!(
                count_only.hits.is_empty(),
                "limit zero must retain Count only"
            );
            assert!(past_end.hits.is_empty(), "past-end offset must be empty");
            assert_eq!(first.hits.len(), 3);
            assert_eq!(second.hits.len(), 3);
            assert_eq!(prefix.hits.len(), 6);

            for (page_rank, hit) in first.hits.iter().enumerate() {
                assert_eq!(hit.page_rank, page_rank);
                assert_eq!(hit.absolute_rank, page_rank);
            }
            for (page_rank, hit) in second.hits.iter().enumerate() {
                assert_eq!(hit.page_rank, page_rank);
                assert_eq!(hit.absolute_rank, page_rank + 3);
            }
            let combined = first
                .hits
                .iter()
                .chain(&second.hits)
                .map(|hit| (&hit.doc_id, hit.score_bits))
                .collect::<Vec<_>>();
            let prefix = prefix
                .hits
                .iter()
                .map(|hit| (&hit.doc_id, hit.score_bits))
                .collect::<Vec<_>>();
            assert_eq!(combined, prefix, "offset pages must compose exactly");
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
            assert_eq!(idx.doc_count().expect("document count"), 5);

            idx.delete_document(&cx, "doc-1").await.expect("delete");
            idx.commit(&cx).await.expect("commit after delete");

            let results = idx.search(&cx, "Rust systems", 10).await.expect("search");
            assert!(
                !results.iter().any(|r| r.doc_id == "doc-1"),
                "deleted document should not appear"
            );
        });
    }

    #[test]
    fn delete_document_publishes_to_public_lexical_readers_before_return() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            let retained = IndexableDocument::new("retained", "delete visibility sentinel");
            let deleted = IndexableDocument::new("deleted", "delete visibility sentinel");
            LexicalWrite::index_documents(&idx, &cx, &[retained, deleted])
                .await
                .expect("index documents through the public write trait");
            LexicalWrite::commit(&idx, &cx)
                .await
                .expect("publish documents through the public write trait");

            assert_eq!(
                LexicalRead::doc_count(&idx).expect("count published documents"),
                2
            );

            idx.delete_document(&cx, "deleted")
                .await
                .expect("delete must publish before returning");

            let hits = LexicalRead::search(&idx, &cx, "delete visibility sentinel", 10)
                .await
                .expect("public lexical reader must see the committed deletion");
            assert_eq!(
                hits.iter()
                    .map(|hit| hit.doc_id.as_str())
                    .collect::<Vec<_>>(),
                vec!["retained"]
            );
            assert_eq!(
                LexicalRead::doc_count(&idx).expect("count refreshed readable generation"),
                1
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
            assert_eq!(idx.doc_count().expect("document count"), 0);

            let doc = IndexableDocument::new("doc-1", "first");
            idx.index_document(&cx, &doc).await.expect("index");
            assert_eq!(
                idx.doc_count().expect("pre-commit document count"),
                0,
                "the count must describe the current searcher, not staged writer state"
            );
            idx.commit(&cx).await.expect("commit");
            assert_eq!(idx.doc_count().expect("document count"), 1);

            let doc = IndexableDocument::new("doc-2", "second");
            idx.index_document(&cx, &doc).await.expect("index");
            idx.commit(&cx).await.expect("commit");
            assert_eq!(idx.doc_count().expect("document count"), 2);

            idx.delete_document(&cx, "doc-1").await.expect("delete");
            idx.commit(&cx).await.expect("commit delete");
            assert_eq!(idx.doc_count().expect("document count"), 1);
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
    fn every_live_tantivy_search_surface_has_one_total_order_under_score_ties() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            idx.writer
                .lock(&cx)
                .await
                .expect("lock writer")
                .set_merge_policy(Box::new(tantivy::merge_policy::NoMergePolicy));

            const DOC_COUNT: usize = 120;
            const SEGMENT_SIZE: usize = 30;
            for segment_start in (0..DOC_COUNT).step_by(SEGMENT_SIZE) {
                let docs = (segment_start..segment_start + SEGMENT_SIZE)
                    .map(|ordinal| {
                        IndexableDocument::new(
                            format!("tie-{ordinal:03}"),
                            "alpha beta gamma delta shared exact score",
                        )
                        .with_title("alpha beta heading")
                        .with_metadata("ordinal", ordinal.to_string())
                    })
                    .collect::<Vec<_>>();
                idx.index_documents(&cx, &docs)
                    .await
                    .expect("index tie segment");
                idx.commit(&cx).await.expect("commit tie segment");
            }

            let query_shapes = [
                "alpha",
                "alpha beta",
                "alpha beta gamma",
                "\"alpha beta\"",
                "alpha AND beta",
                "title:alpha",
            ];
            for query in query_shapes {
                for limit in [0, 1, DOC_COUNT - 1, DOC_COUNT, DOC_COUNT + 1] {
                    assert_live_lexical_contract(&idx, &cx, query, limit, DOC_COUNT).await;
                }
            }
            assert_live_lexical_contract(&idx, &cx, "", 20, 0).await;
            assert_live_lexical_contract(&idx, &cx, "term-not-present", 20, 0).await;
        });
    }

    #[test]
    fn harvested_natural_language_query_keeps_count_off_the_public_candidate_path() {
        let idx = TantivyIndex::in_memory().expect("create");
        run_with_cx(|cx| async move {
            let documents = shared_full120_docs();
            for batch in documents.chunks(5) {
                idx.index_documents(&cx, batch)
                    .await
                    .expect("index Full120 campaign batch");
            }
            idx.commit(&cx).await.expect("commit Full120");

            const QUERY: &str = "how to sear a steak properly";
            const LIMIT: usize = 20;
            let full = idx.search(&cx, QUERY, LIMIT).await.expect("full search");
            reset_collector_invocations();
            let ids = idx
                .search_doc_ids(&cx, QUERY, LIMIT)
                .expect("count-free ID search");
            let candidates = idx
                .search_candidates(&cx, QUERY, LIMIT)
                .await
                .expect("fusion candidates")
                .into_parts()
                .0;
            assert_eq!(
                collector_invocations(),
                (0, 2),
                "both public candidate surfaces must construct TopDocs-only and never Count"
            );
            let counted = idx
                .search_doc_ids_counted(&cx, QUERY, LIMIT)
                .expect("counted benchmark control");
            assert_eq!(
                collector_invocations(),
                (1, 2),
                "only the explicit benchmark control may construct (TopDocs, Count)"
            );

            assert_eq!(full.len(), LIMIT);
            assert_eq!(ids.len(), full.len());
            assert_eq!(candidates.len(), full.len());
            assert_eq!(counted.len(), LIMIT);
            for (rank, ((expected, id_hit), candidate)) in
                full.iter().zip(&ids).zip(&candidates).enumerate()
            {
                let context = format!("harvested-14 rank={rank}");
                assert_eq!(id_hit.rank, rank, "{context}");
                assert_eq!(id_hit.doc_id, expected.doc_id, "{context}");
                assert_eq!(
                    id_hit.bm25_score.to_bits(),
                    expected.score.to_bits(),
                    "{context}"
                );
                assert_eq!(candidate.doc_id, expected.doc_id, "{context}");
                assert_eq!(
                    candidate.score.to_bits(),
                    expected.score.to_bits(),
                    "{context}"
                );
            }

            let parsed = idx.parse_query_lenient(QUERY);
            let counted_page =
                execute_query_with_offset(&idx.reader.searcher(), &*parsed, LIMIT, 0)
                    .expect("exact counted API");
            assert_eq!(
                counted_page.total_count, 110,
                "the pinned Full120 harvested-14 exact count drifted"
            );
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

    /// Tantivy candidates hydrate from the generation that scored them.
    ///
    /// The same contract Quill already proves, now that `TantivyIndex`
    /// overrides `search_candidates`/`hydrate_candidates` instead of falling
    /// back to the eager default (`bd-8nqz.1`). Scores come from generation
    /// N; a commit publishes N+1; the retained batch must still hydrate N's
    /// metadata while a fresh search observes N+1.
    #[test]
    fn tantivy_candidates_hydrate_from_the_scoring_generation() {
        use frankensearch_core::traits::LexicalRead;

        run_with_cx(|cx| async move {
            let dir = tempfile::tempdir().expect("tempdir");
            let idx = TantivyIndex::create(dir.path()).expect("create");

            let v1 =
                IndexableDocument::new("doc-a", "alpha pinned content").with_metadata("rev", "v1");
            LexicalWrite::index_document(&idx, &cx, &v1)
                .await
                .expect("index N");
            LexicalWrite::commit(&idx, &cx).await.expect("publish N");

            // Score on N. Candidates defer metadata and carry the searcher pin.
            let batch = LexicalRead::search_candidates(&idx, &cx, "pinned", 10)
                .await
                .expect("candidates on N");
            assert!(
                batch.is_deferred(),
                "Tantivy candidates must carry a searcher pin, not fall back to eager"
            );
            assert_eq!(batch.results().len(), 1);
            assert!(
                batch.results()[0].metadata.is_none(),
                "deferred candidates skip metadata until hydration"
            );

            // Publish N+1 with different metadata for the same document.
            let v2 =
                IndexableDocument::new("doc-a", "alpha pinned content").with_metadata("rev", "v2");
            LexicalWrite::index_document(&idx, &cx, &v2)
                .await
                .expect("upsert N+1");
            LexicalWrite::commit(&idx, &cx).await.expect("publish N+1");

            let rev_of = |result: &ScoredResult| -> Option<String> {
                result.metadata.as_ref().and_then(|metadata| {
                    metadata
                        .get("rev")
                        .and_then(|value| value.as_str())
                        .map(str::to_owned)
                })
            };

            // A fresh search observes N+1.
            let fresh = LexicalRead::search(&idx, &cx, "pinned", 10)
                .await
                .expect("fresh search");
            assert_eq!(rev_of(&fresh[0]).as_deref(), Some("v2"));

            // The retained batch hydrates from its PINNED generation N.
            let (mut winners, context) = batch.into_parts();
            LexicalRead::hydrate_candidates(&idx, &cx, context.as_ref(), &mut winners)
                .await
                .expect("pinned hydration");
            assert_eq!(
                rev_of(&winners[0]).as_deref(),
                Some("v1"),
                "hydration must read the scoring searcher, not the newest generation"
            );
        });
    }

    /// A foreign or missing context is refused rather than silently reading
    /// the current generation.
    #[test]
    fn tantivy_hydration_rejects_a_foreign_or_absent_context() {
        use frankensearch_core::traits::{LexicalHydrationContext, LexicalRead};

        run_with_cx(|cx| async move {
            let dir = tempfile::tempdir().expect("tempdir");
            let idx = TantivyIndex::create(dir.path()).expect("create");
            let doc = IndexableDocument::new("doc-a", "alpha").with_metadata("rev", "v1");
            LexicalWrite::index_document(&idx, &cx, &doc)
                .await
                .expect("index");
            LexicalWrite::commit(&idx, &cx).await.expect("commit");

            let mut winners = vec![ScoredResult {
                doc_id: DocId::from("doc-a"),
                score: 1.0,
                source: ScoreSource::Lexical,
                index: None,
                fast_score: None,
                quality_score: None,
                lexical_score: Some(1.0),
                rerank_score: None,
                explanation: None,
                metadata: None,
            }];

            let absent = LexicalRead::hydrate_candidates(&idx, &cx, None, &mut winners)
                .await
                .expect_err("deferred winners without their pin must not hydrate silently");
            assert!(matches!(
                absent,
                SearchError::SubsystemError {
                    subsystem: "tantivy.hydration",
                    ..
                }
            ));

            let foreign = LexicalHydrationContext::new("quill", Box::new(7_u64));
            let error = LexicalRead::hydrate_candidates(&idx, &cx, Some(&foreign), &mut winners)
                .await
                .expect_err("a foreign pin must be refused");
            match error {
                SearchError::SubsystemError { subsystem, source } => {
                    assert_eq!(subsystem, "tantivy.hydration");
                    assert!(
                        source.to_string().contains("quill"),
                        "the rejection must name the foreign backend"
                    );
                }
                other => panic!("expected a typed subsystem error, got {other:?}"),
            }
        });
    }
}

#[cfg(all(test, feature = "bench-internals"))]
mod benchmark_writer_mode_tests {
    use super::{
        BenchmarkMaterializedWidth, BenchmarkWidthUnobservableReason, BenchmarkWriterMode,
        TOKENIZER_NAME, TantivyIndex, positions_from_live_schema,
    };
    use frankensearch_core::SearchError;

    /// Tantivy requires a per-writer floor, so an eight-thread pool needs at
    /// least `8 * 15_000_000` bytes. Ask for that plus headroom, or the widest
    /// case silently exercises a narrower pool than it claims.
    const HEAP: usize = 8 * 15_000_000 + 64 * 1024 * 1024;

    #[test]
    fn shipping_auto_uses_the_pinned_selection_path_and_reports_no_width() {
        let mut index = TantivyIndex::in_memory_with_shipping_auto_writer(HEAP, true)
            .expect("shipping-auto writer");
        let attestation = index
            .take_benchmark_writer_attestation()
            .expect("successful benchmark writer mints one live attestation");
        let receipt = index
            .benchmark_writer_receipt()
            .cloned()
            .expect("shipping-auto stamps a receipt");

        assert_eq!(attestation.receipt(), &receipt);
        assert!(
            index.take_benchmark_writer_attestation().is_none(),
            "the live attestation is a one-shot capability"
        );
        assert_eq!(receipt.mode, BenchmarkWriterMode::ShippingAuto);
        assert_eq!(
            receipt.materialized_width,
            BenchmarkMaterializedWidth::Unobservable {
                reason: BenchmarkWidthUnobservableReason::EngineSelectedWidthNotExposed,
            }
        );
        // The unknown must stay unknown everywhere it is read.
        assert_eq!(receipt.materialized_width.authenticated(), None);
        assert_eq!(index.benchmark_materialized_writer_threads(), None);
        assert_eq!(receipt.writer_heap_bytes, HEAP);
        assert!(!receipt.writer_rearmed);
    }

    #[test]
    fn identical_fixed_constructors_mint_distinct_live_attestations() {
        let mut first = TantivyIndex::in_memory_with_benchmark_config(HEAP, 4, true)
            .expect("first fixed writer");
        let mut second = TantivyIndex::in_memory_with_benchmark_config(HEAP, 4, true)
            .expect("second fixed writer");
        let first_attestation = first
            .take_benchmark_writer_attestation()
            .expect("first live attestation");
        let second_attestation = second
            .take_benchmark_writer_attestation()
            .expect("second live attestation");

        assert_ne!(
            first_attestation.construction_id(),
            second_attestation.construction_id(),
            "identical benchmark requests still construct separate live writers"
        );
        assert_eq!(
            first_attestation.receipt(),
            first.benchmark_writer_receipt().expect("first receipt")
        );
        assert_eq!(
            second_attestation.receipt(),
            second.benchmark_writer_receipt().expect("second receipt")
        );
        assert_eq!(
            first_attestation.receipt(),
            second_attestation.receipt(),
            "the diagnostic receipt may agree for identical real constructors"
        );
        assert_eq!(
            first_attestation.receipt().mode,
            BenchmarkWriterMode::Fixed { threads: 4 }
        );
        assert_eq!(
            first_attestation.receipt().materialized_width,
            BenchmarkMaterializedWidth::Authenticated(4)
        );
    }

    #[test]
    fn fixed_width_is_authenticated_exactly() {
        for threads in [1usize, 3, 8] {
            let index = TantivyIndex::in_memory_with_benchmark_config(HEAP, threads, true)
                .expect("fixed writer");
            let receipt = index.benchmark_writer_receipt().expect("fixed receipt");

            assert_eq!(receipt.mode, BenchmarkWriterMode::Fixed { threads });
            assert_eq!(
                receipt.materialized_width,
                BenchmarkMaterializedWidth::Authenticated(threads)
            );
            assert_eq!(receipt.materialized_width.authenticated(), Some(threads));
            assert_eq!(index.benchmark_materialized_writer_threads(), Some(threads));
        }
    }

    #[test]
    fn invalid_width_and_insufficient_heap_fail_closed() {
        assert!(matches!(
            TantivyIndex::in_memory_with_benchmark_config(HEAP, 0, true),
            Err(SearchError::InvalidConfig { .. })
        ));
        // A heap below Tantivy's per-writer floor must surface its typed
        // construction error, never a silently downgraded pool.
        assert!(TantivyIndex::in_memory_with_benchmark_config(1, 4, true).is_err());
        assert!(TantivyIndex::in_memory_with_shipping_auto_writer(1, true).is_err());
    }

    #[test]
    fn shipping_and_fixed_receipts_are_distinguishable() {
        let auto = TantivyIndex::in_memory_with_shipping_auto_writer(HEAP, true)
            .expect("shipping-auto writer");
        let fixed =
            TantivyIndex::in_memory_with_benchmark_config(HEAP, 4, true).expect("fixed writer");
        let auto = auto
            .benchmark_writer_receipt()
            .expect("auto receipt")
            .clone();
        let fixed = fixed.benchmark_writer_receipt().expect("fixed receipt");

        assert_ne!(
            &auto, fixed,
            "the two modes must not produce equal receipts"
        );
        assert_ne!(auto.mode, fixed.mode);
        assert_ne!(auto.materialized_width, fixed.materialized_width);
        // Identity that is genuinely shared must still match, or the receipt
        // would be distinguishing on incidental noise instead of on mode.
        assert_eq!(auto.schema_fields, fixed.schema_fields);
        assert_eq!(auto.tokenizer_name, fixed.tokenizer_name);
    }

    #[test]
    fn schema_and_oracle_identity_track_the_index_that_was_built() {
        let with_positions_index = TantivyIndex::in_memory_with_shipping_auto_writer(HEAP, true)
            .expect("positions-on writer");
        let without_positions_index =
            TantivyIndex::in_memory_with_shipping_auto_writer(HEAP, false)
                .expect("positions-off writer");
        let with_positions = with_positions_index
            .benchmark_writer_receipt()
            .expect("receipt")
            .clone();
        let without_positions = without_positions_index
            .benchmark_writer_receipt()
            .expect("receipt");

        assert!(with_positions.positions);
        assert!(!without_positions.positions);
        assert_ne!(
            &with_positions, without_positions,
            "a positions mutation must change the receipt"
        );
        assert_eq!(with_positions.tokenizer_name, TOKENIZER_NAME);
        // The receipt's positions value must be the live indexing option, not a
        // remembered argument, so read the schema back independently.
        assert_eq!(
            positions_from_live_schema(&with_positions_index.index.schema()),
            Some(true)
        );
        assert_eq!(
            positions_from_live_schema(&without_positions_index.index.schema()),
            Some(false)
        );
        // Pin the exact field set this crate's schema builds, so the receipt is
        // read from the live schema rather than reporting a plausible shape.
        // A field added, renamed, or dropped must fail here.
        assert_eq!(
            with_positions.schema_fields,
            vec![
                "id".to_owned(),
                "content".to_owned(),
                "title".to_owned(),
                "metadata_json".to_owned(),
                "ord".to_owned(),
            ],
            "schema identity must match the fields build_schema_with_positions creates"
        );
        assert_eq!(
            without_positions.schema_fields, with_positions.schema_fields,
            "positions is an index option, not a field, so the field set must not move"
        );
    }

    #[test]
    fn a_fresh_writer_never_claims_a_rearm() {
        for index in [
            TantivyIndex::in_memory_with_shipping_auto_writer(HEAP, true).expect("auto"),
            TantivyIndex::in_memory_with_benchmark_config(HEAP, 2, true).expect("fixed"),
        ] {
            assert!(
                !index
                    .benchmark_writer_receipt()
                    .expect("receipt")
                    .writer_rearmed,
                "a freshly constructed writer must not report a rearm"
            );
        }
    }

    #[test]
    fn the_construction_receipt_flips_after_a_rearm() {
        let mut index =
            TantivyIndex::in_memory_with_shipping_auto_writer(HEAP, true).expect("auto writer");
        let original_attestation = index
            .take_benchmark_writer_attestation()
            .expect("original live attestation");
        assert_eq!(
            index.benchmark_writer_receipt().expect("receipt").mode,
            BenchmarkWriterMode::ShippingAuto
        );

        let (mut rearmed, _join) = index
            .benchmark_join_workers_and_rearm(HEAP, 2)
            .expect("join and rearm");
        let rearmed_attestation = rearmed
            .take_benchmark_writer_attestation()
            .expect("replacement writer mints a fresh live attestation");
        let receipt = rearmed.benchmark_writer_receipt().expect("rearm receipt");

        // A rearm replaces the writer, so the construction receipt must stop
        // describing the original one: the mode moves to the fixed constructor
        // the rearm actually called, and the rearm flag is set.
        assert!(receipt.writer_rearmed);
        assert_eq!(receipt.mode, BenchmarkWriterMode::Fixed { threads: 2 });
        assert_eq!(
            receipt.materialized_width,
            BenchmarkMaterializedWidth::Authenticated(2)
        );
        assert_eq!(rearmed_attestation.receipt(), receipt);
        assert_ne!(
            original_attestation.construction_id(),
            rearmed_attestation.construction_id(),
            "rearming replaces the writer and must mint a new live identity"
        );
        // The receipt states the rearm's intent; this states the call it made.
        // Carrying the old writer's observation forward instead would leave a
        // rearmed index reporting Auto, which this rejects.
        assert_eq!(
            rearmed.observed_writer_call,
            super::WriterCall::Fixed(2),
            "the rearm must observe its own writer_with_num_threads call"
        );
    }

    #[test]
    fn the_receipt_seed_cannot_drift_from_the_constructor_branch() {
        // Discriminating observer: the two factories differ only in which
        // Tantivy constructor they reach, and the receipt is seeded inside that
        // branch. If the seed were passed in beside the call instead, these two
        // could report the same mode while calling different constructors.
        let factories: [(&str, fn() -> TantivyIndex, BenchmarkWriterMode); 2] = [
            (
                "Index::writer",
                || TantivyIndex::in_memory_with_shipping_auto_writer(HEAP, true).expect("auto"),
                BenchmarkWriterMode::ShippingAuto,
            ),
            (
                "Index::writer_with_num_threads",
                || TantivyIndex::in_memory_with_benchmark_config(HEAP, 8, true).expect("fixed"),
                BenchmarkWriterMode::Fixed { threads: 8 },
            ),
        ];
        for (constructor, factory, expected_mode) in factories {
            let index = factory();
            let receipt = index.benchmark_writer_receipt().expect("receipt");
            assert_eq!(
                receipt.mode, expected_mode,
                "receipt mode must name the constructor that ran: {constructor}"
            );
            assert_eq!(
                receipt.writer_heap_bytes, HEAP,
                "heap must come from the same branch as the mode: {constructor}"
            );
            assert_eq!(
                receipt.materialized_width.authenticated(),
                match expected_mode {
                    BenchmarkWriterMode::ShippingAuto => None,
                    BenchmarkWriterMode::Fixed { threads } => Some(threads),
                },
                "width authentication must follow the branch: {constructor}"
            );
        }
    }

    /// Heap that is exactly one thread's floor.
    ///
    /// This is the label-independent discriminator. `writer_with_num_threads`
    /// rejects a width whose per-thread share falls under Tantivy's floor,
    /// while `Index::writer` *clamps* the width instead of failing. So at this
    /// budget the two constructors disagree observably: a fixed eight-wide
    /// construction must fail and a shipping-auto construction must succeed.
    const ONE_THREAD_FLOOR_HEAP: usize = 15_000_000;

    #[test]
    fn the_constructor_is_discriminated_by_behaviour_not_by_its_label() {
        // If ShippingAuto secretly called writer_with_num_threads(8, ..) this
        // would fail; if Fixed{8} secretly called Index::writer it would
        // succeed. Neither can be rescued by a matching label.
        let auto = TantivyIndex::in_memory_with_shipping_auto_writer(ONE_THREAD_FLOOR_HEAP, true)
            .expect("Index::writer clamps the width instead of rejecting this budget");
        assert_eq!(
            auto.benchmark_writer_receipt().expect("receipt").mode,
            BenchmarkWriterMode::ShippingAuto
        );

        assert!(
            TantivyIndex::in_memory_with_benchmark_config(ONE_THREAD_FLOOR_HEAP, 8, true).is_err(),
            "writer_with_num_threads must reject eight writers sharing one writer's floor"
        );
    }

    #[test]
    fn the_rearm_constructor_is_discriminated_the_same_way() {
        let index = TantivyIndex::in_memory_with_shipping_auto_writer(HEAP, true)
            .expect("auto writer at a comfortable budget");
        // The rearm must reconstruct through writer_with_num_threads, so it
        // inherits that constructor's floor rejection. Succeeding here would
        // prove it had fallen back to Index::writer.
        assert!(
            index
                .benchmark_join_workers_and_rearm(ONE_THREAD_FLOOR_HEAP, 8)
                .is_err(),
            "rearm must reconstruct at a pinned width, not through the auto path"
        );
    }

    #[test]
    fn ordinary_constructors_claim_no_benchmark_receipt() {
        // Only an explicit benchmark plan may produce a receipt. An ordinary
        // index that happens to reach the same Tantivy call must not be able to
        // present itself as a screened candidate.
        let mut ordinary = TantivyIndex::in_memory().expect("ordinary in-memory index");
        assert!(ordinary.benchmark_writer_receipt().is_none());
        assert!(ordinary.benchmark_materialized_writer_threads().is_none());
        assert!(
            ordinary.take_benchmark_writer_attestation().is_none(),
            "ordinary construction must not mint a live benchmark capability"
        );

        // The pinned oracle only exists behind `tantivy-oracle`; this module is
        // gated on `bench-internals` alone, so its negative coverage has to be
        // conditional or `bench-internals` by itself stops compiling.
        #[cfg(feature = "tantivy-oracle")]
        {
            let mut oracle =
                TantivyIndex::in_memory_single_threaded_oracle().expect("single-threaded oracle");
            assert!(
                oracle.benchmark_writer_receipt().is_none(),
                "pinning a width is not the same as claiming a screening receipt"
            );
            assert!(
                oracle.benchmark_materialized_writer_threads().is_none(),
                "an unscreened oracle authenticates no width to a screening consumer"
            );
            assert!(
                oracle.take_benchmark_writer_attestation().is_none(),
                "an ordinary pinned oracle must not mint a live benchmark capability"
            );
        }
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn the_ordinary_pinned_oracle_invokes_the_fixed_constructor() {
        use super::WriterCall;

        // Asserting "no receipt" here proves nothing: reverting this oracle to
        // WriterPlan::Shipping keeps the receipt absent while silently moving
        // the writer onto Tantivy's auto selection, which is exactly the
        // regression this test exists to catch. So observe the call itself,
        // recorded by the helper that performed it, on this instance — not a
        // global counter that parallel tests would race.
        let oracle =
            TantivyIndex::in_memory_single_threaded_oracle().expect("single-threaded oracle");

        assert_eq!(
            oracle.observed_writer_call,
            WriterCall::Fixed(1),
            "the oracle must invoke writer_with_num_threads(1, ..), not the auto path"
        );
        assert!(
            oracle.benchmark_writer_receipt().is_none(),
            "invoking the fixed constructor still claims no screening identity"
        );
    }

    #[test]
    fn the_observed_call_tracks_the_constructor_each_plan_reaches() {
        use super::WriterCall;

        // The same observer over the benchmark plans, so a plan that reached
        // the wrong Tantivy entry point cannot hide behind a matching label.
        let auto =
            TantivyIndex::in_memory_with_shipping_auto_writer(HEAP, true).expect("auto writer");
        assert_eq!(auto.observed_writer_call, WriterCall::Auto);

        let fixed =
            TantivyIndex::in_memory_with_benchmark_config(HEAP, 4, true).expect("fixed writer");
        assert_eq!(fixed.observed_writer_call, WriterCall::Fixed(4));

        let ordinary = TantivyIndex::in_memory().expect("ordinary in-memory index");
        assert_eq!(ordinary.observed_writer_call, WriterCall::Auto);
    }

    #[test]
    fn a_reopened_index_rejects_a_disagreeing_positions_claim() {
        let directory = tempfile::tempdir().expect("temporary index directory");
        let path = directory.path().join("oracle");
        TantivyIndex::create_with_benchmark_config(&path, HEAP, 2, true)
            .expect("create with positions");

        // Reopening while claiming the opposite positions setting is an
        // assertion about bytes that already exist, and must fail closed rather
        // than writing a false value into every downstream receipt.
        assert!(matches!(
            TantivyIndex::open_with_benchmark_config(&path, HEAP, 2, false),
            Err(SearchError::InvalidConfig { .. })
        ));

        let reopened = TantivyIndex::open_with_benchmark_config(&path, HEAP, 2, true)
            .expect("reopen with the true positions setting");
        assert!(
            reopened
                .benchmark_writer_receipt()
                .expect("receipt")
                .positions,
            "the reopened receipt must carry the live schema's positions option"
        );
    }

    #[test]
    fn a_reopened_positions_off_index_rejects_the_mirrored_claim() {
        // Mirror of the case above: the rejection must be symmetric, or a
        // positions-off index could be reopened while claiming positions-on and
        // carry that false claim into every downstream receipt.
        let directory = tempfile::tempdir().expect("temporary index directory");
        let path = directory.path().join("oracle");
        TantivyIndex::create_with_benchmark_config(&path, HEAP, 2, false)
            .expect("create without positions");

        assert!(matches!(
            TantivyIndex::open_with_benchmark_config(&path, HEAP, 2, true),
            Err(SearchError::InvalidConfig { .. })
        ));

        let reopened = TantivyIndex::open_with_benchmark_config(&path, HEAP, 2, false)
            .expect("reopen with the true positions setting");
        assert!(
            !reopened
                .benchmark_writer_receipt()
                .expect("receipt")
                .positions,
            "the reopened receipt must carry the live schema's positions option"
        );
    }
}

/// Default-feature coverage for the writer-call observation.
///
/// Without this the observation is written on every build but only read behind
/// `bench-internals`, so a default `--all-targets` run would carry a field that
/// is never read. The fix is a real reader, not an allow: the ordinary
/// in-memory constructor genuinely must reach `Index::writer`, and that is
/// worth asserting on its own.
#[cfg(test)]
mod writer_call_observation_tests {
    use super::{TantivyIndex, WriterCall};

    #[test]
    fn the_ordinary_in_memory_constructor_invokes_the_auto_writer() {
        let index = TantivyIndex::in_memory().expect("ordinary in-memory index");
        assert_eq!(
            index.observed_writer_call,
            WriterCall::Auto,
            "the ordinary constructor must reach Index::writer, not a pinned width"
        );
    }
}
