//! External CASS Tantivy-format interoperability.
//!
//! NOTE: this module is intentionally not a fallback for native Quill. It
//! reads and writes the external CASS tool's schema-v8 Tantivy indexes at
//! `<base>/index/v8/`, so removing it before CASS migrates to FSLX (or drops
//! the integration) would break a foreign-format boundary. The facade exposes
//! it only through `cass-compat`; default builds do not include this crate.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{OnceLock, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use frankensearch_core::error::{SearchError, SearchResult};
use tantivy::SegmentMeta;
use tantivy::indexer::{LogMergePolicy, MergeCandidate, MergePolicy, NoMergePolicy, UserOperation};
use tantivy::query::{
    AllQuery, BooleanQuery, EnableScoring, Explanation, Occur, PhraseQuery, Query, RangeQuery,
    RegexQuery, Scorer, TermQuery, Weight,
};
use tantivy::schema::IndexRecordOption;
use tantivy::schema::{
    FAST, Field, INDEXED, STORED, STRING, Schema, TextFieldIndexing, TextOptions,
};
#[cfg(test)]
use tantivy::tokenizer::RegexTokenizer;
use tantivy::tokenizer::{TextAnalyzer, Token, TokenFilter, TokenStream, Tokenizer};
use tantivy::{
    DocId, DocSet, Index, IndexReader, IndexWriter, ReloadPolicy, Score, SegmentReader, TERMINATED,
    TantivyError, Term, doc,
};
use tracing::{debug, info, warn};

/// Adapter that re-wraps an `Arc<dyn MergePolicy>` as a `Box<dyn MergePolicy>`,
/// so a policy snapshot taken via [`IndexWriter::get_merge_policy`] (which
/// returns an `Arc`) can be reinstalled via
/// [`IndexWriter::set_merge_policy`] (which takes a `Box`).
///
/// This is the missing glue that lets [`CassTantivyIndex::force_merge_bounded`]
/// preserve and restore the *exact* previously-configured merge policy
/// (including any caller customization) instead of resetting to
/// `LogMergePolicy::default()`.
#[derive(Debug)]
struct ArcMergePolicy(Arc<dyn MergePolicy>);

impl MergePolicy for ArcMergePolicy {
    fn compute_merge_candidates(&self, segments: &[SegmentMeta]) -> Vec<MergeCandidate> {
        self.0.compute_merge_candidates(segments)
    }
}

/// Schema version namespace used for cass-compatible Tantivy indexes.
pub const CASS_SCHEMA_VERSION: &str = "v8";
/// Content hash used to detect schema/tokenizer changes that require rebuild.
pub const CASS_SCHEMA_HASH: &str =
    "tantivy-schema-v8-hyphen-cjk-bigrams-bounded-content-prefix-preview-stored-content-external";

/// Specialized tokenizer for cass lexical fields.
///
/// This preserves the current regex tokenizer semantics for:
/// - ASCII alphanumeric runs
/// - hyphen-joined ASCII alphanumeric runs
/// - contiguous runs in the exact CJK ranges covered by the old regex
///
/// A dedicated scanner avoids the regex DFA hot path that dominates rebuild
/// time on large corpora.
#[derive(Clone, Default)]
struct CassTokenizer {
    token: Token,
}

struct CassTokenStream<'a> {
    text: &'a str,
    cursor: usize,
    token: &'a mut Token,
}

#[inline]
fn is_cass_tokenizer_cjk(c: char) -> bool {
    matches!(
        c,
        '\u{4E00}'..='\u{9FFF}'
            | '\u{3400}'..='\u{4DBF}'
            | '\u{3040}'..='\u{309F}'
            | '\u{30A0}'..='\u{30FF}'
            | '\u{AC00}'..='\u{D7AF}'
            | '\u{3100}'..='\u{312F}'
            | '\u{3300}'..='\u{33FF}'
            | '\u{F900}'..='\u{FAFF}'
            | '\u{20000}'..='\u{2A6DF}'
    )
}

#[inline]
fn next_char_from(text: &str, offset: usize) -> Option<(char, usize)> {
    // ASCII fast-path: the tokenizer scans char-by-char, and its inputs (English, code, IDs) are
    // overwhelmingly ASCII. `offset` is always a char boundary (the tokenizer only advances by whole
    // chars), so a leading byte `< 128` is a complete single-byte UTF-8 char — return it directly,
    // skipping the `chars().next()` UTF-8 decode + `len_utf8()` recompute. Bit-identical to the slow
    // path for every input (asserted in `next_char_from_ascii_matches_decode`).
    let b = *text.as_bytes().get(offset)?;
    if b < 128 {
        return Some((b as char, offset + 1));
    }
    let ch = text[offset..].chars().next()?;
    Some((ch, offset + ch.len_utf8()))
}

/// Pre-fast-path [`next_char_from`], retained doc-hidden for the same-binary A/B and the parity
/// test. Not a shipping path.
#[doc(hidden)]
#[must_use]
pub fn next_char_from_slow(text: &str, offset: usize) -> Option<(char, usize)> {
    let ch = text[offset..].chars().next()?;
    Some((ch, offset + ch.len_utf8()))
}

/// Doc-hidden bench harness: walk `text` char-by-char via the shipped (ASCII-fast) `next_char_from`,
/// folding char values so the loop cannot be optimized away. Mirrors the tokenizer's scan cost.
#[doc(hidden)]
#[must_use]
pub fn cass_char_walk_fast(text: &str) -> u64 {
    let mut acc = 0u64;
    let mut off = 0usize;
    while let Some((ch, next)) = next_char_from(text, off) {
        acc = acc.wrapping_add(ch as u64);
        off = next;
    }
    acc
}

/// Doc-hidden bench harness: the same walk via [`next_char_from_slow`] (the pre-fast-path decode).
#[doc(hidden)]
#[must_use]
pub fn cass_char_walk_slow(text: &str) -> u64 {
    let mut acc = 0u64;
    let mut off = 0usize;
    while let Some((ch, next)) = next_char_from_slow(text, off) {
        acc = acc.wrapping_add(ch as u64);
        off = next;
    }
    acc
}

impl Tokenizer for CassTokenizer {
    type TokenStream<'a> = CassTokenStream<'a>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        self.token.reset();
        CassTokenStream {
            text,
            cursor: 0,
            token: &mut self.token,
        }
    }
}

impl CassTokenStream<'_> {
    fn scan_ascii_token(&self, mut cursor: usize) -> usize {
        let mut end = cursor;
        let mut last_was_ascii_alnum = false;

        while let Some((ch, next_cursor)) = next_char_from(self.text, cursor) {
            if ch.is_ascii_alphanumeric() {
                end = next_cursor;
                cursor = next_cursor;
                last_was_ascii_alnum = true;
                continue;
            }

            if ch == '-'
                && last_was_ascii_alnum
                && let Some((next_ch, _)) = next_char_from(self.text, next_cursor)
                && next_ch.is_ascii_alphanumeric()
            {
                end = next_cursor;
                cursor = next_cursor;
                last_was_ascii_alnum = false;
                continue;
            }

            break;
        }

        end
    }

    fn scan_cjk_token(&self, mut cursor: usize) -> usize {
        let mut end = cursor;
        while let Some((ch, next_cursor)) = next_char_from(self.text, cursor) {
            if !is_cass_tokenizer_cjk(ch) {
                break;
            }
            end = next_cursor;
            cursor = next_cursor;
        }
        end
    }
}

impl TokenStream for CassTokenStream<'_> {
    fn advance(&mut self) -> bool {
        self.token.text.clear();
        self.token.position = self.token.position.wrapping_add(1);

        while let Some((ch, next_cursor)) = next_char_from(self.text, self.cursor) {
            if ch.is_ascii_alphanumeric() {
                let offset_from = self.cursor;
                let offset_to = self.scan_ascii_token(self.cursor);
                self.token.offset_from = offset_from;
                self.token.offset_to = offset_to;
                self.token.text.push_str(&self.text[offset_from..offset_to]);
                self.cursor = offset_to;
                return true;
            }

            if is_cass_tokenizer_cjk(ch) {
                let offset_from = self.cursor;
                let offset_to = self.scan_cjk_token(next_cursor);
                self.token.offset_from = offset_from;
                self.token.offset_to = offset_to;
                self.token.text.push_str(&self.text[offset_from..offset_to]);
                self.cursor = offset_to;
                return true;
            }

            self.cursor = next_cursor;
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

// ─── HyphenDecompose token filter ────────────────────────────────────────────
//
// When a token contains an interior hyphen (e.g. `bd-q3fy`), the filter emits
// the compound form **and** the individual sub-parts so that both exact and
// partial matches work.  Tokens without hyphens pass through unchanged.

/// A [`TokenFilter`] that decomposes hyphenated tokens into their compound form
/// plus each hyphen-delimited part.
///
/// Given the token `bd-q3fy`, the stream will yield:
///   1. `bd-q3fy`   (compound, same position)
///   2. `bd`         (part, same position)
///   3. `q3fy`       (part, same position)
#[derive(Clone)]
pub struct HyphenDecompose;

impl TokenFilter for HyphenDecompose {
    type Tokenizer<T: Tokenizer> = HyphenDecomposeFilter<T>;

    fn transform<T: Tokenizer>(self, tokenizer: T) -> HyphenDecomposeFilter<T> {
        HyphenDecomposeFilter {
            inner: tokenizer,
            pending: Vec::new(),
        }
    }
}

#[derive(Clone)]
pub struct HyphenDecomposeFilter<T> {
    inner: T,
    pending: Vec<Token>,
}

impl<T: Tokenizer> Tokenizer for HyphenDecomposeFilter<T> {
    type TokenStream<'a> = HyphenDecomposeStream<'a, T::TokenStream<'a>>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        self.pending.clear();
        HyphenDecomposeStream {
            tail: self.inner.token_stream(text),
            pending: &mut self.pending,
        }
    }
}

pub struct HyphenDecomposeStream<'a, T> {
    tail: T,
    pending: &'a mut Vec<Token>,
}

impl<T: TokenStream> HyphenDecomposeStream<'_, T> {
    /// If the current upstream token contains hyphens, push the compound form
    /// and sub-parts (in reverse order so `.pop()` yields them in order).
    fn decompose(&mut self) {
        let token = self.tail.token();
        if !token.text.contains('-') {
            return;
        }
        let parts: Vec<&str> = token.text.split('-').filter(|s| !s.is_empty()).collect();
        if parts.len() < 2 {
            return;
        }
        // Push sub-parts in reverse so pop() yields them left-to-right.
        for &part in parts.iter().rev() {
            self.pending.push(Token {
                text: part.to_owned(),
                position: token.position,
                offset_from: token.offset_from,
                offset_to: token.offset_to,
                position_length: token.position_length,
            });
        }
        // Push the compound form last so it is popped first.
        self.pending.push(token.clone());
    }
}

impl<T: TokenStream> TokenStream for HyphenDecomposeStream<'_, T> {
    fn advance(&mut self) -> bool {
        // Drain any buffered tokens from a previous decomposition first.
        self.pending.pop();
        if !self.pending.is_empty() {
            return true;
        }

        if !self.tail.advance() {
            return false;
        }

        self.decompose();
        // If decompose produced tokens, the first will come from pending.
        // Otherwise the plain upstream token is returned via `token()`.
        true
    }

    fn token(&self) -> &Token {
        self.pending.last().unwrap_or_else(|| self.tail.token())
    }

    fn token_mut(&mut self) -> &mut Token {
        self.pending
            .last_mut()
            .unwrap_or_else(|| self.tail.token_mut())
    }
}

// ─── CJK Bigram token filter ────────────────────────────────────────────────
//
// CJK scripts (Chinese, Japanese kanji, Korean hangul) are written without
// whitespace between words.  A single CJK token such as "搜索引擎" must be
// decomposed into overlapping character bigrams ("搜索", "索引", "引擎") so
// that sub-string queries can match.  Non-CJK tokens pass through unchanged.

/// Returns `true` when `c` falls in a CJK-relevant Unicode range.
#[inline]
fn is_cjk(c: char) -> bool {
    matches!(c,
        '\u{4E00}'..='\u{9FFF}'   // CJK Unified Ideographs
        | '\u{3400}'..='\u{4DBF}' // CJK Extension A
        | '\u{3040}'..='\u{309F}' // Hiragana
        | '\u{30A0}'..='\u{30FF}' // Katakana
        | '\u{AC00}'..='\u{D7AF}' // Hangul Syllables
        | '\u{3100}'..='\u{312F}' // Bopomofo
        | '\u{3300}'..='\u{33FF}' // CJK Compatibility
        | '\u{F900}'..='\u{FAFF}' // CJK Compatibility Ideographs
        | '\u{20000}'..='\u{2A6DF}' // CJK Extension B
    )
}

/// One-pass CJK-token collection for the bigram decomposer: decode `text`
/// **once**, validating that every char is CJK while collecting them. Returns
/// `Some(chars)` iff the token is entirely CJK with ≥2 chars (the only case
/// that decomposes into bigrams); `None` otherwise. A non-CJK token bails on
/// its first char *before* allocating, so the common ASCII/Latin token pays
/// nothing.
///
/// This fuses the two full UTF-8 decode passes the naive form makes — the
/// guard scan `chars().all(is_cjk)` followed by `chars().collect()` — into a
/// single pass. Byte-identical to [`cass_cjk_collect_slow`]: `decompose_cjk`
/// only emits bigrams when all chars are CJK and the count is ≥2, exactly the
/// `Some` case here (empty, non-CJK, and single-CJK all map to `None`).
#[doc(hidden)]
#[must_use]
pub fn cass_cjk_collect_fast(text: &str) -> Option<Vec<char>> {
    let mut chars = text.chars();
    // First char decides the common case: a non-CJK lead (ASCII/Latin/space)
    // rejects immediately with no allocation.
    let first = chars.next()?;
    if !is_cjk(first) {
        return None;
    }
    let mut buf: Vec<char> = Vec::with_capacity(text.len() / 3 + 1);
    buf.push(first);
    for c in chars {
        if !is_cjk(c) {
            return None;
        }
        buf.push(c);
    }
    if buf.len() < 2 {
        // Single CJK character: unchanged (emitted as its own unigram).
        return None;
    }
    Some(buf)
}

/// The pre-fusion two-pass form, retained as the bench oracle for
/// [`cass_cjk_collect_fast`]: decode `text` once to validate (`all`) and again
/// to `collect`.
#[doc(hidden)]
#[must_use]
pub fn cass_cjk_collect_slow(text: &str) -> Option<Vec<char>> {
    if text.is_empty() || !text.chars().all(is_cjk) {
        return None;
    }
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < 2 {
        return None;
    }
    Some(chars)
}

/// Pre-optimization CJK bigram materialization retained for same-binary benchmarks.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub fn cass_cjk_bigrams_staged(text: &str, token: &Token, pending: &mut Vec<Token>) {
    let Some(chars) = cass_cjk_collect_fast(text) else {
        return;
    };
    let mut bigrams: Vec<Token> = Vec::with_capacity(chars.len());
    for i in (0..chars.len() - 1).rev() {
        let mut bigram = String::with_capacity(8);
        bigram.push(chars[i]);
        bigram.push(chars[i + 1]);
        bigrams.push(Token {
            text: bigram,
            position: token.position,
            offset_from: token.offset_from,
            offset_to: token.offset_to,
            position_length: token.position_length,
        });
    }
    pending.extend(bigrams);
}

/// Direct-to-pending CJK bigram materialization candidate for same-binary benchmarks.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub fn cass_cjk_bigrams_direct(text: &str, token: &Token, pending: &mut Vec<Token>) {
    let Some(chars) = cass_cjk_collect_fast(text) else {
        return;
    };
    pending.reserve(chars.len() - 1);
    for i in (0..chars.len() - 1).rev() {
        let mut bigram = String::with_capacity(8);
        bigram.push(chars[i]);
        bigram.push(chars[i + 1]);
        pending.push(Token {
            text: bigram,
            position: token.position,
            offset_from: token.offset_from,
            offset_to: token.offset_to,
            position_length: token.position_length,
        });
    }
}

/// A [`TokenFilter`] that decomposes tokens consisting entirely of CJK
/// characters into overlapping character bigrams.
///
/// Given the token `搜索引擎`, the stream will yield:
///   `搜索`, `索引`, `引擎`
///
/// A single CJK character emits itself as a unigram.
/// Tokens that contain no CJK characters pass through unchanged.
#[derive(Clone)]
pub struct CjkBigramDecompose;

impl TokenFilter for CjkBigramDecompose {
    type Tokenizer<T: Tokenizer> = CjkBigramDecomposeFilter<T>;

    fn transform<T: Tokenizer>(self, tokenizer: T) -> CjkBigramDecomposeFilter<T> {
        CjkBigramDecomposeFilter {
            inner: tokenizer,
            pending: Vec::new(),
        }
    }
}

#[derive(Clone)]
pub struct CjkBigramDecomposeFilter<T> {
    inner: T,
    pending: Vec<Token>,
}

impl<T: Tokenizer> Tokenizer for CjkBigramDecomposeFilter<T> {
    type TokenStream<'a> = CjkBigramDecomposeStream<'a, T::TokenStream<'a>>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        self.pending.clear();
        CjkBigramDecomposeStream {
            tail: self.inner.token_stream(text),
            pending: &mut self.pending,
        }
    }
}

pub struct CjkBigramDecomposeStream<'a, T> {
    tail: T,
    pending: &'a mut Vec<Token>,
}

impl<T: TokenStream> CjkBigramDecomposeStream<'_, T> {
    fn decompose_cjk(&mut self) {
        let token = self.tail.token();

        // Only decompose tokens that are entirely CJK with ≥2 chars. One fused
        // pass validates + collects (most cass tokens are ASCII and reject on
        // their first char with no allocation); empty / non-CJK / single-CJK
        // tokens all yield `None` and are left unchanged.
        let Some(chars) = cass_cjk_collect_fast(&token.text) else {
            return;
        };

        // Build bigrams in reverse so `.pop()` yields them left-to-right.
        // We replace the original token entirely with bigrams.
        self.pending.reserve(chars.len() - 1);
        for i in (0..chars.len() - 1).rev() {
            let mut bigram = String::with_capacity(8);
            bigram.push(chars[i]);
            bigram.push(chars[i + 1]);
            self.pending.push(Token {
                text: bigram,
                position: token.position,
                offset_from: token.offset_from,
                offset_to: token.offset_to,
                position_length: token.position_length,
            });
        }

        // The first bigram is popped first (it's at the end of the vec).
    }
}

impl<T: TokenStream> TokenStream for CjkBigramDecomposeStream<'_, T> {
    fn advance(&mut self) -> bool {
        self.pending.pop();
        if !self.pending.is_empty() {
            return true;
        }

        if !self.tail.advance() {
            return false;
        }

        self.decompose_cjk();
        true
    }

    fn token(&self) -> &Token {
        self.pending.last().unwrap_or_else(|| self.tail.token())
    }

    fn token_mut(&mut self) -> &mut Token {
        self.pending
            .last_mut()
            .unwrap_or_else(|| self.tail.token_mut())
    }
}

/// A cass-specific normalization filter that preserves the shipping CASS
/// analyzer's inclusive 256-byte admission contract.
///
/// This intentionally differs at exactly 256 bytes from Tantivy's
/// `RemoveLongFilter::limit(256)`, whose predicate retains only tokens shorter
/// than the configured limit. CASS has historically retained 256-byte tokens
/// and dropped tokens starting at 257 bytes, so that behavior is the oracle.
///
/// `CassTokenizer` only emits ASCII alphanumeric runs (optionally hyphenated)
/// plus CJK runs, so ASCII-only in-place lowercasing is behaviorally identical
/// to Tantivy's generic lowercaser here while avoiding its broader Unicode path.
#[derive(Clone, Copy, Default)]
pub struct CassNormalizeAndLimit;

impl TokenFilter for CassNormalizeAndLimit {
    type Tokenizer<T: Tokenizer> = CassNormalizeAndLimitFilter<T>;

    fn transform<T: Tokenizer>(self, tokenizer: T) -> CassNormalizeAndLimitFilter<T> {
        CassNormalizeAndLimitFilter { inner: tokenizer }
    }
}

#[derive(Clone)]
pub struct CassNormalizeAndLimitFilter<T> {
    inner: T,
}

impl<T: Tokenizer> Tokenizer for CassNormalizeAndLimitFilter<T> {
    type TokenStream<'a> = CassNormalizeAndLimitStream<T::TokenStream<'a>>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        CassNormalizeAndLimitStream {
            tail: self.inner.token_stream(text),
        }
    }
}

pub struct CassNormalizeAndLimitStream<T> {
    tail: T,
}

impl<T: TokenStream> TokenStream for CassNormalizeAndLimitStream<T> {
    fn advance(&mut self) -> bool {
        while self.tail.advance() {
            let token = self.tail.token_mut();
            if token.text.len() > 256 {
                continue;
            }
            token.text.make_ascii_lowercase();
            return true;
        }

        false
    }

    fn token(&self) -> &Token {
        self.tail.token()
    }

    fn token_mut(&mut self) -> &mut Token {
        self.tail.token_mut()
    }
}

/// Minimum time (ms) between merge operations.
const MERGE_COOLDOWN_MS: i64 = 300_000;
/// Segment count threshold above which merge is triggered.
const MERGE_SEGMENT_THRESHOLD: usize = 4;
/// Cap cass rebuilds to a more aggressive Tantivy worker pool size. The
/// rebuild path is bulk-load oriented and can profit from many more indexing
/// workers on 32+ core servers than the original workstation-oriented cap.
///
/// Operators can still lower this via `CASS_TANTIVY_MAX_WRITER_THREADS`.
const CASS_MAX_WRITER_THREADS: usize = 32;
/// Reserve a healthy minimum heap budget so large repairs do not churn tiny
/// segments and constant flush/merge cycles on multi-million-message corpora.
const CASS_MIN_WRITER_HEAP_BYTES: usize = 256 * 1024 * 1024;
/// Scale the heap budget with the worker count so each indexing thread has
/// enough in-memory runway before Tantivy has to flush segments.
const CASS_WRITER_HEAP_PER_THREAD_BYTES: usize = 128 * 1024 * 1024;
/// During cass bulk rebuilds, delay background merges until there is a
/// meaningful backlog of segments instead of burning CPU on near-continuous
/// compaction while the canonical database is still streaming documents in.
const CASS_BULK_LOAD_MIN_SEGMENTS_PER_MERGE: usize = 256;
/// Large cass rebuild batches should be split into multiple independent writer
/// submissions so Tantivy's worker pool can consume them concurrently instead
/// of idling behind a single giant `IndexWriter::run(...)` call.
const CASS_PARALLEL_ADD_MIN_DOCS: usize = 2_048;
const CASS_PARALLEL_ADD_TARGET_BATCH_DOCS: usize = 512;
const CASS_PARALLEL_ADD_MAX_BATCHES: usize = 64;

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
    pub conversation_id: Option<Field>,
}

/// Canonical CASS document identity: `"{source_id}#{msg_idx}"`.
///
/// The CASS schema deliberately has no dedicated id field — a document is a
/// message inside a source, so identity is the pair. The synthetic conformance
/// corpus repeats `source_id` on purpose (several messages share one session),
/// which is exactly why `source_id` alone cannot serve as the key and `msg_idx`
/// supplies per-message uniqueness.
///
/// This exact rendering is part of the CASS profile contract
/// (`document_identity=source_id#msg_idx` in the schema contract preimage) and
/// must match on both the subject and the oracle side. It is not an adapter
/// convenience: the gauntlet comparator keys every divergence signature on this
/// string, so changing the format invalidates the register.
///
/// Missing stored fields degrade to an empty `source_id` and `msg_idx` 0 rather
/// than erroring, so a malformed document surfaces as a *comparable* identity
/// mismatch in the differential report instead of aborting the campaign.
#[cfg(feature = "tantivy-oracle")]
#[must_use]
pub fn cass_document_identity(document: &tantivy::TantivyDocument, fields: &CassFields) -> String {
    use tantivy::schema::Value as _;

    let source_id = document
        .get_first(fields.source_id)
        .and_then(|value| value.as_str())
        .unwrap_or_default();
    let msg_idx = document
        .get_first(fields.msg_idx)
        .and_then(|value| value.as_u64())
        .unwrap_or_default();
    cass_document_identity_parts(source_id, msg_idx)
}

/// Render the canonical CASS document identity from its two components.
///
/// This is the single source of truth for the `"{source_id}#{msg_idx}"`
/// rendering named by the CASS profile contract. It lives in this crate rather
/// than in the gauntlet because the dependency runs gauntlet → lexical: the
/// corpus generator and both engine adapters must agree byte-for-byte, and the
/// only way to guarantee that is for all of them to call one function.
///
/// Prefer [`cass_document_identity`] when you already hold a stored document;
/// use this form when you are constructing the corpus and hold the parts.
#[cfg(feature = "tantivy-oracle")]
#[must_use]
pub fn cass_document_identity_parts(source_id: &str, msg_idx: u64) -> String {
    format!("{source_id}#{msg_idx}")
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
    pub conversation_id: Option<i64>,
}

#[derive(Debug, Clone, Copy)]
pub struct CassDocumentRef<'a> {
    pub agent: &'a str,
    pub workspace: Option<&'a str>,
    pub workspace_original: Option<&'a str>,
    pub source_path: &'a str,
    pub msg_idx: u64,
    pub created_at: Option<i64>,
    pub title: Option<&'a str>,
    pub content: &'a str,
    pub source_id: &'a str,
    pub origin_kind: &'a str,
    pub origin_host: Option<&'a str>,
    pub conversation_id: Option<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CassWriterConfig {
    num_threads: usize,
    heap_size_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CassAddPlan {
    parallel_batches: bool,
    batch_docs: usize,
    batch_count: usize,
}

fn cass_writer_config() -> CassWriterConfig {
    let available_parallelism = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    cass_writer_config_for_parallelism(available_parallelism)
}

fn cass_writer_config_for_parallelism(available_parallelism: usize) -> CassWriterConfig {
    let max_threads = std::env::var("CASS_TANTIVY_MAX_WRITER_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(CASS_MAX_WRITER_THREADS);
    let num_threads = available_parallelism.clamp(1, max_threads);
    let heap_size_bytes = num_threads
        .saturating_mul(CASS_WRITER_HEAP_PER_THREAD_BYTES)
        .max(CASS_MIN_WRITER_HEAP_BYTES);
    CassWriterConfig {
        num_threads,
        heap_size_bytes,
    }
}

fn cass_parallel_add_min_docs() -> usize {
    std::env::var("CASS_TANTIVY_PARALLEL_ADD_MIN_DOCS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(CASS_PARALLEL_ADD_MIN_DOCS)
}

fn cass_parallel_add_target_batch_docs() -> usize {
    std::env::var("CASS_TANTIVY_PARALLEL_ADD_BATCH_DOCS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(CASS_PARALLEL_ADD_TARGET_BATCH_DOCS)
}

fn cass_parallel_add_max_batches() -> usize {
    std::env::var("CASS_TANTIVY_PARALLEL_ADD_MAX_BATCHES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(CASS_PARALLEL_ADD_MAX_BATCHES)
}

fn cass_parallel_add_plan(
    doc_count: usize,
    available_parallelism: usize,
    target_batch_docs: usize,
    min_parallel_docs: usize,
    max_batches: usize,
) -> CassAddPlan {
    if doc_count == 0 {
        return CassAddPlan {
            parallel_batches: false,
            batch_docs: 1,
            batch_count: 0,
        };
    }

    let target_batch_docs = target_batch_docs.max(1);
    let max_batches = max_batches.max(1);
    let parallelism_batches = available_parallelism.max(1).saturating_mul(2).max(1);
    let batch_count = doc_count
        .div_ceil(target_batch_docs)
        .min(max_batches)
        .min(parallelism_batches)
        .max(1);
    let batch_docs = doc_count.div_ceil(batch_count).max(1);

    CassAddPlan {
        parallel_batches: doc_count >= min_parallel_docs && batch_count > 1,
        batch_docs,
        batch_count,
    }
}

/// Tantivy index compatible with cass lexical schema and lifecycle.
pub struct CassTantivyIndex {
    index: Index,
    writer: IndexWriter,
    fields: CassFields,
}

impl CassTantivyIndex {
    /// Create a fresh in-memory CASS oracle with one deterministic writer.
    ///
    /// This constructor is restricted to differential campaigns whose native
    /// `DocAddress` evidence must be reproducible across same-seed replays.
    /// Shipping callers retain the host-scaled writer configuration exposed by
    /// [`Self::open_or_create`].
    ///
    /// # Errors
    ///
    /// Returns [`SearchError`] if schema extraction or writer construction
    /// fails.
    #[cfg(feature = "tantivy-oracle")]
    #[doc(hidden)]
    pub fn in_memory_single_threaded_oracle() -> SearchResult<Self> {
        let mut index = Index::create_in_ram(cass_build_schema());
        cass_ensure_tokenizer(&mut index);
        let fields = cass_fields_from_schema(&index.schema())?;
        let writer_config = cass_writer_config_for_parallelism(1);
        let writer = index
            .writer_with_num_threads(writer_config.num_threads, writer_config.heap_size_bytes)
            .map_err(tantivy_err)?;
        Ok(Self {
            index,
            writer,
            fields,
        })
    }

    /// Open existing index or create/rebuild as needed.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError`] if filesystem I/O, schema extraction, or Tantivy
    /// index creation/opening fails.
    pub fn open_or_create(path: &Path) -> SearchResult<Self> {
        let available_parallelism = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1);
        Self::open_or_create_with_writer_parallelism(path, available_parallelism)
    }

    /// Open existing index or create/rebuild as needed, using an explicit
    /// Tantivy writer parallelism budget instead of the host-wide default.
    ///
    /// This is used by cass staged shard builds so multiple isolated shard
    /// writers can run concurrently without every shard trying to claim the
    /// machine-wide maximum writer thread count.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError`] if filesystem I/O, schema extraction, or Tantivy
    /// index creation/opening fails.
    pub fn open_or_create_with_writer_parallelism(
        path: &Path,
        available_parallelism: usize,
    ) -> SearchResult<Self> {
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
            if let Err(e) = std::fs::remove_dir_all(path) {
                if e.kind() != std::io::ErrorKind::NotFound {
                    return Err(SearchError::from(e));
                }
            }
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
                    if let Err(e) = std::fs::remove_dir_all(path) {
                        if e.kind() != std::io::ErrorKind::NotFound {
                            return Err(SearchError::from(e));
                        }
                    }
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
        let writer_config = cass_writer_config_for_parallelism(available_parallelism.max(1));
        debug!(
            tantivy_writer_threads = writer_config.num_threads,
            tantivy_writer_heap_mb = writer_config.heap_size_bytes / (1024 * 1024),
            "opening cass-compatible tantivy writer"
        );
        let writer = index
            .writer_with_num_threads(writer_config.num_threads, writer_config.heap_size_bytes)
            .map_err(tantivy_err)?;
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

    /// Observe the CASS-compatible parser/search path for Quill conformance.
    ///
    /// The CASS-profile analogue of [`crate::TantivyIndex::oracle_observe_query`].
    /// Like that producer it retains the full Tantivy `DocAddress`, the exact
    /// total count, raw score **bits**, and an expanded cutoff tie group, because
    /// the gauntlet comparator classifies `TieOrder` from native ordering and
    /// cannot do so once that evidence is canonicalized away.
    ///
    /// Two deliberate differences from the default-profile producer, both
    /// required by the CASS activation contract:
    ///
    /// * **No empty-query short circuit.** The default profile returns an empty
    ///   observation for a blank query. Under CASS an empty query carrying
    ///   structured filters is a legitimate filter-only browse that matches
    ///   documents, so short-circuiting would report zero hits where the engine
    ///   returns a full filtered page.
    /// * **Snippets are always `None`.** The activation profile is snippet-free
    ///   (`snippets=disabled` in the schema contract preimage); preview and
    ///   highlight parity belong to a separate lane, and `SnippetWindow` is not
    ///   an active allowance for this profile.
    ///
    /// Document identity is the profile contract's `"{source_id}#{msg_idx}"`
    /// rendering (see [`cass_document_identity`]). It is pinned by the contract
    /// preimage rather than chosen here, because changing the rendering silently
    /// changes every mismatch signature already recorded in the divergence
    /// register.
    ///
    /// Cancellation is the caller's responsibility: this is a synchronous
    /// Tantivy path, so the gauntlet adapter checks its `Cx` before calling
    /// rather than threading a context through the oracle.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError`] if reader/searcher construction, query execution,
    /// or stored-document loading fails.
    #[cfg(feature = "tantivy-oracle")]
    pub fn cass_oracle_observe_query(
        &self,
        raw_query: &str,
        filters: &CassQueryFilters,
        limit: usize,
        tie_expansion_limit: usize,
    ) -> SearchResult<crate::OracleQueryObservation> {
        cass_oracle_preflight(raw_query, filters, limit, tie_expansion_limit)?;
        let fields = self.fields();
        let tokens = cass_parse_boolean_query_bounded(raw_query, MAX_CASS_ORACLE_TOKENS).map_err(
            |actual| cass_oracle_invalid_bound("token_count", actual, MAX_CASS_ORACLE_TOKENS),
        )?;
        let has_boolean_operators = cass_tokens_have_boolean_operators(&tokens);
        let query = cass_fail_closed_tantivy_query(
            raw_query,
            cass_try_build_tantivy_query_from_tokens(
                raw_query,
                tokens,
                has_boolean_operators,
                filters,
                &fields,
                cass_regex_query_cached,
            ),
        );
        self.cass_oracle_observe_built_query(&*query, limit, tie_expansion_limit)
    }

    /// Observe the real CASS query build and retrieval boundaries without the
    /// compatibility API's silent match-none fallback.
    ///
    /// The returned in-memory witness carries plaintext parser values only so
    /// the gauntlet can immediately redact them. It is not an artifact DTO.
    /// Query construction is total here: a regex, lowering, reader, search, or
    /// document-loading failure is retained as [`CassOracleProfileOutcome::Error`]
    /// instead of becoming an indistinguishable successful empty result.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] before token retention, filter
    /// cloning, or search when any request bound is exceeded.
    #[cfg(feature = "tantivy-oracle")]
    pub fn cass_oracle_observe_query_profile(
        &self,
        raw_query: &str,
        filters: &CassQueryFilters,
        limit: usize,
        tie_expansion_limit: usize,
    ) -> SearchResult<CassOracleProfileObservation> {
        self.cass_oracle_observe_query_profile_with_regex_factory(
            raw_query,
            filters,
            limit,
            tie_expansion_limit,
            cass_regex_query_cached,
        )
    }

    #[cfg(feature = "tantivy-oracle")]
    fn cass_oracle_observe_query_profile_with_regex_factory(
        &self,
        raw_query: &str,
        filters: &CassQueryFilters,
        limit: usize,
        tie_expansion_limit: usize,
        regex_query_factory: CassRegexQueryFactory,
    ) -> SearchResult<CassOracleProfileObservation> {
        cass_oracle_preflight(raw_query, filters, limit, tie_expansion_limit)?;
        let tokens = cass_parse_boolean_query_bounded(raw_query, MAX_CASS_ORACLE_TOKENS).map_err(
            |actual| cass_oracle_invalid_bound("token_count", actual, MAX_CASS_ORACLE_TOKENS),
        )?;
        let has_boolean_operators = cass_tokens_have_boolean_operators(&tokens);
        let fields = self.fields();
        let outcome = cass_try_build_tantivy_query_from_tokens(
            raw_query,
            tokens.clone(),
            has_boolean_operators,
            filters,
            &fields,
            regex_query_factory,
        )
        .and_then(|query| self.cass_oracle_observe_built_query(&*query, limit, tie_expansion_limit))
        .map_or_else(
            CassOracleProfileOutcome::Error,
            CassOracleProfileOutcome::Success,
        );
        Ok(CassOracleProfileObservation {
            sanitized_query: cass_sanitize_query(raw_query),
            tokens,
            has_boolean_operators,
            filters: filters.clone(),
            outcome,
        })
    }

    #[cfg(feature = "tantivy-oracle")]
    fn cass_oracle_observe_built_query(
        &self,
        query: &dyn Query,
        limit: usize,
        tie_expansion_limit: usize,
    ) -> SearchResult<crate::OracleQueryObservation> {
        let searcher = self.reader()?.searcher();
        let fields = self.fields();

        // Fetch past `limit` so the cutoff tie group can be observed whole; a
        // truncated group is reported through `cutoff_tie_complete` instead of
        // being silently accepted as complete.
        let fetch_limit = if limit == 0 {
            0
        } else {
            limit.checked_add(tie_expansion_limit).ok_or_else(|| {
                cass_oracle_invalid_bound(
                    "expanded_fetch_hits",
                    usize::MAX,
                    MAX_CASS_ORACLE_FETCH_HITS,
                )
            })?
        };
        cass_oracle_bound(
            "expanded_fetch_hits",
            fetch_limit,
            MAX_CASS_ORACLE_FETCH_HITS,
        )?;
        let search_result = crate::execute_query_with_offset(&searcher, query, fetch_limit, 0)?;

        let mut materialized = Vec::with_capacity(search_result.hits.len());
        let mut retained_identity_bytes = 0_usize;
        for hit in search_result.hits {
            let document = crate::load_doc(&searcher, hit.doc_address)?;
            let doc_id = cass_document_identity(&document, &fields);
            cass_oracle_bound(
                "document_identity_bytes",
                doc_id.len(),
                MAX_CASS_ORACLE_DOCUMENT_ID_BYTES,
            )?;
            retained_identity_bytes = retained_identity_bytes.saturating_add(doc_id.len());
            cass_oracle_bound(
                "aggregate_document_identity_bytes",
                retained_identity_bytes,
                MAX_CASS_ORACLE_DOCUMENT_ID_AGGREGATE_BYTES,
            )?;
            materialized.push(crate::OracleRankedHit {
                doc_id,
                score_bits: hit.bm25_score.to_bits(),
                rank: hit.rank,
                segment_ord: hit.doc_address.segment_ord,
                segment_doc_id: hit.doc_address.doc_id,
                snippet: None,
            });
        }

        let top_len = limit.min(materialized.len());
        let cutoff_bits = top_len
            .checked_sub(1)
            .and_then(|index| materialized.get(index))
            .map(|hit| hit.score_bits);
        let mut cutoff_tie_group = Vec::new();
        if let Some(cutoff) = cutoff_bits {
            for hit in &materialized {
                if f32::from_bits(hit.score_bits)
                    .total_cmp(&f32::from_bits(cutoff))
                    .is_eq()
                {
                    retained_identity_bytes =
                        retained_identity_bytes.saturating_add(hit.doc_id.len());
                    cass_oracle_bound(
                        "aggregate_document_identity_bytes",
                        retained_identity_bytes,
                        MAX_CASS_ORACLE_DOCUMENT_ID_AGGREGATE_BYTES,
                    )?;
                    cutoff_tie_group.push(hit.clone());
                }
            }
        }
        // The group is complete when either every match was fetched or the last
        // fetched hit scored strictly below the cutoff — otherwise the expansion
        // budget ended inside the tie and the comparator must not assume more.
        let cutoff_tie_complete = cutoff_bits.is_none_or(|cutoff| {
            search_result.total_count <= fetch_limit
                || materialized.last().is_none_or(|last| {
                    !f32::from_bits(last.score_bits)
                        .total_cmp(&f32::from_bits(cutoff))
                        .is_eq()
                })
        });
        let expanded_hits = materialized.clone();
        materialized.truncate(top_len);

        let doc_count =
            usize::try_from(searcher.num_docs()).map_err(|_| SearchError::SubsystemError {
                subsystem: "tantivy",
                source: "current Tantivy reader document count does not fit usize".into(),
            })?;

        Ok(crate::OracleQueryObservation {
            hits: materialized,
            cutoff_tie_group,
            cutoff_tie_complete,
            total_count: search_result.total_count,
            doc_count,
            expanded_hits,
            searcher_sha256: crate::oracle_searcher_sha256(&searcher),
        })
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

    pub fn configure_bulk_load_merge_policy(&mut self) {
        let mut merge_policy = LogMergePolicy::default();
        merge_policy.set_min_num_segments(CASS_BULK_LOAD_MIN_SEGMENTS_PER_MERGE);
        self.writer.set_merge_policy(Box::new(merge_policy));
        debug!(
            min_num_segments = CASS_BULK_LOAD_MIN_SEGMENTS_PER_MERGE,
            "configured cass bulk-load merge policy"
        );
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

    /// Force a full merge while never opening more than `batch_size` segments
    /// at once, returning the number of merge operations performed.
    ///
    /// [`force_merge`](Self::force_merge) merges *all* searchable segments in a
    /// single `merge()` call, which memory-maps every input segment
    /// simultaneously. On an index that has accumulated many segments (e.g.
    /// after a long ingest run with transient auto-merge failures), that single
    /// call can map more regions than the OS allows (`vm.max_map_count`,
    /// 65530 by default on Linux) and fail with an out-of-memory / mmap error,
    /// leaving the caller with no in-wrapper recovery path.
    ///
    /// This consolidates the index in bounded passes instead: each iteration
    /// merges at most `batch_size` segments, so peak mmap pressure is capped at
    /// `batch_size * files-per-segment` regardless of how many segments exist.
    /// Repeated passes drive the segment count down to a single top-level
    /// segment.
    ///
    /// For the duration of the operation a [`NoMergePolicy`] is installed: a
    /// completed `merge()` calls `consider_merge_options()` internally, and the
    /// previously installed [`LogMergePolicy`] (or any caller-configured
    /// policy) could otherwise schedule its own unbounded merge of the
    /// remaining same-tier segments — reintroducing exactly the mmap pressure
    /// we are avoiding. The *previously configured* merge policy is preserved
    /// (via [`IndexWriter::get_merge_policy`]) and restored on return — on both
    /// the success and the error paths — so any caller-side customization
    /// (e.g. [`Self::configure_bulk_load_merge_policy`]) is not silently
    /// dropped. Restoration round-trips the `Arc`-shared policy back through
    /// tantivy's `Box`-typed setter via a private `ArcMergePolicy` adapter.
    ///
    /// `batch_size` is clamped to a minimum of 2 (a batch of 1 can never make
    /// progress).
    ///
    /// # Caller contract
    ///
    /// The writer lock is held exclusively for the entire call, which can take
    /// many minutes on a large index. Any other process attempting to acquire
    /// the writer lock on the same directory fails fast with `LockBusy`.
    /// Callers must coordinate externally — e.g. stop the ingester before
    /// invoking this method.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::SubsystemError`] if segment enumeration or any
    /// individual batch merge fails. The previous merge policy is restored
    /// even on the error path.
    pub fn force_merge_bounded(&mut self, batch_size: usize) -> SearchResult<usize> {
        let batch_size = batch_size.max(2);

        // Snapshot the prior merge policy via the cheap Arc clone exposed by
        // tantivy 0.26, then install NoMergePolicy for the duration of the
        // bounded loop. The Arc snapshot is the key bit: restoring
        // `LogMergePolicy::default()` (as a naive implementation would) would
        // silently discard any caller-side customization such as
        // [`Self::configure_bulk_load_merge_policy`]'s tuned
        // `min_num_segments`.
        let prior = self.writer.get_merge_policy();
        self.writer.set_merge_policy(Box::new(NoMergePolicy));

        let mut merges = 0usize;
        let outcome: SearchResult<usize> = (|| {
            loop {
                let segment_ids = self.index.searchable_segment_ids().map_err(tantivy_err)?;
                if segment_ids.len() <= 1 {
                    return Ok(merges);
                }
                let take = segment_ids.len().min(batch_size);
                self.writer
                    .merge(&segment_ids[..take])
                    .wait()
                    .map_err(tantivy_err)?;
                merges += 1;
            }
        })();

        // Restore the *original* merge policy (not a fresh default) on every
        // exit path that is not a panic. Wrapping `prior` in
        // [`ArcMergePolicy`] lets us hand the previously-shared `Arc` back to
        // Tantivy through its `Box<dyn MergePolicy>` setter; the inner Arc is
        // simply re-shared, no rebuilding of policy state.
        self.writer
            .set_merge_policy(Box::new(ArcMergePolicy(prior)));

        if matches!(outcome, Ok(n) if n > 0) {
            let now_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX));
            LAST_MERGE_TS.store(now_ms, Ordering::Relaxed);
        }

        outcome
    }

    /// Add a batch of cass-compatible documents.
    ///
    /// The per-doc prep work (edge-ngram generation and preview extraction) is
    /// CPU-heavy; for non-trivial batches we run it in parallel on rayon and
    /// then feed the pre-built `TantivyDocument`s serially into the writer
    /// (which needs `&mut self`). On a 4-core indexing run this keeps the
    /// tantivy writer threads fed instead of starving on a single-threaded
    /// producer.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::SubsystemError`] if adding a document to Tantivy fails.
    pub fn add_cass_documents(&mut self, docs: &[CassDocument]) -> SearchResult<()> {
        use rayon::prelude::*;

        // Below this size the rayon split/join overhead isn't worth it.
        const PARALLEL_PREP_THRESHOLD: usize = 8;

        let fields = self.fields;
        let writer_parallelism = cass_writer_config().num_threads.max(1);
        let add_plan = cass_parallel_add_plan(
            docs.len(),
            writer_parallelism,
            cass_parallel_add_target_batch_docs(),
            cass_parallel_add_min_docs(),
            cass_parallel_add_max_batches(),
        );

        debug!(
            docs = docs.len(),
            batch_docs = add_plan.batch_docs,
            batch_count = add_plan.batch_count,
            parallel_batches = add_plan.parallel_batches,
            writer_parallelism,
            "submitting cass-compatible documents to tantivy writer"
        );

        if add_plan.parallel_batches {
            let writer = &self.writer;
            docs.par_chunks(add_plan.batch_docs)
                .try_for_each(|chunk| -> SearchResult<()> {
                    writer
                        .run(
                            chunk
                                .iter()
                                .map(|cass_doc| build_cass_tantivy_document(fields, cass_doc))
                                .map(UserOperation::Add),
                        )
                        .map_err(tantivy_err)?;
                    Ok(())
                })?;
        } else {
            let prepared: Vec<tantivy::TantivyDocument> = if docs.len() < PARALLEL_PREP_THRESHOLD {
                docs.iter()
                    .map(|cass_doc| build_cass_tantivy_document(fields, cass_doc))
                    .collect()
            } else {
                docs.par_iter()
                    .map(|cass_doc| build_cass_tantivy_document(fields, cass_doc))
                    .collect()
            };
            self.writer
                .run(prepared.into_iter().map(UserOperation::Add))
                .map_err(tantivy_err)?;
        }
        Ok(())
    }

    /// Add a batch of borrowed cass-compatible documents.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::SubsystemError`] if adding a document to Tantivy fails.
    pub fn add_cass_document_refs(&mut self, docs: &[CassDocumentRef<'_>]) -> SearchResult<()> {
        use rayon::prelude::*;

        const PARALLEL_PREP_THRESHOLD: usize = 8;

        let fields = self.fields;
        let writer_parallelism = cass_writer_config().num_threads.max(1);
        let add_plan = cass_parallel_add_plan(
            docs.len(),
            writer_parallelism,
            cass_parallel_add_target_batch_docs(),
            cass_parallel_add_min_docs(),
            cass_parallel_add_max_batches(),
        );

        debug!(
            docs = docs.len(),
            batch_docs = add_plan.batch_docs,
            batch_count = add_plan.batch_count,
            parallel_batches = add_plan.parallel_batches,
            writer_parallelism,
            "submitting borrowed cass-compatible documents to tantivy writer"
        );

        if add_plan.parallel_batches {
            let writer = &self.writer;
            docs.par_chunks(add_plan.batch_docs)
                .try_for_each(|chunk| -> SearchResult<()> {
                    writer
                        .run(
                            chunk
                                .iter()
                                .copied()
                                .map(|cass_doc| build_cass_tantivy_document_ref(fields, cass_doc))
                                .map(UserOperation::Add),
                        )
                        .map_err(tantivy_err)?;
                    Ok(())
                })?;
        } else {
            let prepared: Vec<tantivy::TantivyDocument> = if docs.len() < PARALLEL_PREP_THRESHOLD {
                docs.iter()
                    .copied()
                    .map(|cass_doc| build_cass_tantivy_document_ref(fields, cass_doc))
                    .collect()
            } else {
                docs.par_iter()
                    .copied()
                    .map(|cass_doc| build_cass_tantivy_document_ref(fields, cass_doc))
                    .collect()
            };
            self.writer
                .run(prepared.into_iter().map(UserOperation::Add))
                .map_err(tantivy_err)?;
        }
        Ok(())
    }
}

/// Build a single `TantivyDocument` for a cass document. Pure function so it
/// can run on any rayon worker.
fn build_cass_tantivy_document(
    fields: CassFields,
    cass_doc: &CassDocument,
) -> tantivy::TantivyDocument {
    let mut d = doc! {
        fields.agent => cass_doc.agent.clone(),
        fields.source_path => cass_doc.source_path.clone(),
        fields.msg_idx => cass_doc.msg_idx,
        fields.content => cass_doc.content.clone(),
        fields.source_id => cass_doc.source_id.clone(),
        fields.origin_kind => cass_doc.origin_kind.clone(),
    };

    if let Some(host) = &cass_doc.origin_host
        && !host.is_empty()
    {
        d.add_text(fields.origin_host, host);
    }
    if let Some(field) = fields.conversation_id
        && let Some(conversation_id) = cass_doc.conversation_id
    {
        d.add_i64(field, conversation_id);
    }
    if let Some(workspace) = &cass_doc.workspace {
        d.add_text(fields.workspace, workspace);
    }
    if let Some(workspace_original) = &cass_doc.workspace_original {
        d.add_text(fields.workspace_original, workspace_original);
    }
    if let Some(ts) = cass_doc.created_at {
        d.add_i64(fields.created_at, ts);
    }
    if let Some(title) = &cass_doc.title {
        d.add_text(fields.title, title);
        d.add_text(fields.title_prefix, cass_generate_edge_ngrams(title));
    }
    let (content_prefix, preview) = cass_build_content_prefix_and_preview(&cass_doc.content);
    d.add_text(fields.content_prefix, content_prefix);
    d.add_text(fields.preview, preview);
    d
}

fn build_cass_tantivy_document_ref(
    fields: CassFields,
    cass_doc: CassDocumentRef<'_>,
) -> tantivy::TantivyDocument {
    let mut d = doc! {
        fields.agent => cass_doc.agent,
        fields.source_path => cass_doc.source_path,
        fields.msg_idx => cass_doc.msg_idx,
        fields.content => cass_doc.content,
        fields.source_id => cass_doc.source_id,
        fields.origin_kind => cass_doc.origin_kind,
    };

    if let Some(host) = cass_doc.origin_host
        && !host.is_empty()
    {
        d.add_text(fields.origin_host, host);
    }
    if let Some(field) = fields.conversation_id
        && let Some(conversation_id) = cass_doc.conversation_id
    {
        d.add_i64(field, conversation_id);
    }
    if let Some(workspace) = cass_doc.workspace {
        d.add_text(fields.workspace, workspace);
    }
    if let Some(workspace_original) = cass_doc.workspace_original {
        d.add_text(fields.workspace_original, workspace_original);
    }
    if let Some(ts) = cass_doc.created_at {
        d.add_i64(fields.created_at, ts);
    }
    if let Some(title) = cass_doc.title {
        d.add_text(fields.title, title);
        d.add_text(fields.title_prefix, cass_generate_edge_ngrams(title));
    }
    let (content_prefix, preview) = cass_build_content_prefix_and_preview(cass_doc.content);
    d.add_text(fields.content_prefix, content_prefix);
    d.add_text(fields.preview, preview);
    d
}

/// Build cass-compatible Tantivy schema.
#[must_use]
pub fn cass_build_schema() -> Schema {
    let mut schema_builder = Schema::builder();
    let indexed_text = TextOptions::default().set_indexing_options(
        TextFieldIndexing::default()
            .set_tokenizer("hyphen_normalize")
            .set_index_option(tantivy::schema::IndexRecordOption::WithFreqsAndPositions),
    );
    let stored_indexed_text = indexed_text.clone().set_stored();
    let prefix_text = TextOptions::default().set_indexing_options(
        TextFieldIndexing::default()
            .set_tokenizer("prefix_normalize")
            .set_index_option(tantivy::schema::IndexRecordOption::Basic),
    );

    schema_builder.add_text_field("agent", STRING | STORED);
    schema_builder.add_text_field("workspace", STRING | STORED);
    schema_builder.add_text_field("workspace_original", STORED);
    schema_builder.add_text_field("source_path", STORED);
    schema_builder.add_u64_field("msg_idx", INDEXED | STORED);
    schema_builder.add_i64_field("created_at", INDEXED | STORED | FAST);
    schema_builder.add_text_field("title", stored_indexed_text);
    schema_builder.add_text_field("content", indexed_text);
    schema_builder.add_text_field("title_prefix", prefix_text.clone());
    schema_builder.add_text_field("content_prefix", prefix_text);
    schema_builder.add_text_field("preview", STORED);
    schema_builder.add_text_field("source_id", STRING | STORED);
    schema_builder.add_text_field("origin_kind", STRING | STORED);
    schema_builder.add_text_field("origin_host", STRING | STORED);
    schema_builder.add_i64_field("conversation_id", STORED);
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
        conversation_id: schema.get_field("conversation_id").ok(),
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
    if let Err(e) = reader.reload() {
        warn!(error = %e, "index reader reload failed — searches may serve stale results");
    }
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

/// Register the tokenizers used by cass-compatible lexical fields.
///
/// `hyphen_normalize` pipeline:
///   1. `CassTokenizer` — matches ASCII alphanumeric runs (with hyphens)
///      **and** CJK character runs as separate tokens.
///   2. `HyphenDecompose` — for each hyphenated token, emits the compound
///      form *and* the individual sub-parts (all at the same position) so
///      both exact ID searches and partial-word searches match.
///   3. `CjkBigramDecompose` — decomposes CJK-only tokens into overlapping
///      character bigrams so that sub-string queries work for Chinese,
///      Japanese, and Korean text.
///   4. `CassNormalizeAndLimit` — applies ASCII lowercase plus the shipping
///      CASS inclusive 256-byte limit (not Tantivy's strict `< 256` predicate).
///
/// `prefix_normalize` is used only for generated edge-ngram fields. Those
/// values are whitespace-separated prefix terms produced by
/// `cass_generate_edge_ngrams`, so they cannot contain hyphens. Keeping the CJK
/// bigram and normalization filters preserves prefix-field query behavior while
/// avoiding the per-token `HyphenDecompose` layer on the highest-volume basic
/// postings fields.
pub fn cass_ensure_tokenizer(index: &mut Index) {
    let analyzer = TextAnalyzer::builder(CassTokenizer::default())
        .filter(HyphenDecompose)
        .filter(CjkBigramDecompose)
        .filter(CassNormalizeAndLimit)
        .build();
    index.tokenizers().register("hyphen_normalize", analyzer);
    index
        .tokenizers()
        .register("prefix_normalize", cass_prefix_analyzer());
}

fn cass_prefix_analyzer() -> TextAnalyzer {
    TextAnalyzer::builder(CassTokenizer::default())
        .filter(CjkBigramDecompose)
        .filter(CassNormalizeAndLimit)
        .build()
}

fn cass_push_prefix_term(out: &mut String, term: &str) {
    if !out.is_empty() {
        out.push(' ');
    }
    out.push_str(term);
}

/// Generate edge n-grams from text for prefix search acceleration.
///
/// For each word, emits its length-2..N prefixes (capped at 20 chars) space-separated. The original
/// re-decoded every word with `char_indices` to find prefix boundaries — a **second** UTF-8 decode
/// on top of the `split` that already scanned the text. For an **ASCII word** (the common case for
/// English/code/IDs), char boundaries are byte positions, so the prefixes are `word[..2..=min(len,
/// 20)]` directly — no `char_indices` decode. Non-ASCII words take the original boundary-collecting
/// path. Byte-for-byte identical output (`cass_generate_edge_ngrams_matches_slow`).
#[must_use]
pub fn cass_generate_edge_ngrams(text: &str) -> String {
    const MAX_NGRAM_INDICES: usize = 21;
    let mut ngrams = String::with_capacity(text.len() * 2);
    for word in text.split(|c: char| !c.is_alphanumeric()) {
        // ASCII fast-path: byte positions ARE char boundaries, so skip the `char_indices` re-decode
        // and slice prefixes directly. `2..=upper` is empty (no prefixes) when `upper < 2`.
        if word.is_ascii() {
            let upper = word.len().min(MAX_NGRAM_INDICES - 1);
            for end in 2..=upper {
                cass_push_prefix_term(&mut ngrams, &word[..end]);
            }
            continue;
        }

        let mut indices = [0usize; MAX_NGRAM_INDICES];
        let mut index_count = 0usize;

        for (i, _) in word.char_indices() {
            if index_count == MAX_NGRAM_INDICES {
                break;
            }
            indices[index_count] = i;
            index_count += 1;
        }

        if index_count < MAX_NGRAM_INDICES {
            indices[index_count] = word.len();
            index_count += 1;
        }

        if index_count < 3 {
            continue;
        }
        for &end_idx in &indices[2..index_count] {
            cass_push_prefix_term(&mut ngrams, &word[..end_idx]);
        }
    }
    ngrams
}

/// Pre-ASCII-fast-path [`cass_generate_edge_ngrams`], retained doc-hidden for the same-binary A/B +
/// parity test.
#[doc(hidden)]
#[must_use]
pub fn cass_generate_edge_ngrams_slow(text: &str) -> String {
    const MAX_NGRAM_INDICES: usize = 21;
    let mut ngrams = String::with_capacity(text.len() * 2);
    for word in text.split(|c: char| !c.is_alphanumeric()) {
        let mut indices = [0usize; MAX_NGRAM_INDICES];
        let mut index_count = 0usize;
        for (i, _) in word.char_indices() {
            if index_count == MAX_NGRAM_INDICES {
                break;
            }
            indices[index_count] = i;
            index_count += 1;
        }
        if index_count < MAX_NGRAM_INDICES {
            indices[index_count] = word.len();
            index_count += 1;
        }
        if index_count < 3 {
            continue;
        }
        for &end_idx in &indices[2..index_count] {
            cass_push_prefix_term(&mut ngrams, &word[..end_idx]);
        }
    }
    ngrams
}

/// Build a bounded-length preview from message content.
///
/// Takes the first `max_chars` characters and appends `…` if the content is longer. One
/// `char_indices` scan finds the cut byte-offset, then a single `push_str` bulk-copies the prefix
/// into a pre-sized buffer — versus pushing `max_chars` chars individually (each re-encoding the
/// char and growing an unallocated `String` through repeated reallocations). Byte-for-byte identical
/// output (`cass_build_preview_matches_slow`).
#[must_use]
pub fn cass_build_preview(content: &str, max_chars: usize) -> String {
    // Byte offset just past the `max_chars`-th char, or `content.len()` if there are fewer.
    let mut cut = content.len();
    let mut count = 0usize;
    // Pinned-oracle scan kept in its shipped shape (single `char_indices` walk).
    #[allow(clippy::explicit_counter_loop)]
    for (byte_idx, _) in content.char_indices() {
        if count == max_chars {
            cut = byte_idx;
            break;
        }
        count += 1;
    }
    let truncated = cut < content.len();
    let mut out = String::with_capacity(cut + if truncated { '…'.len_utf8() } else { 0 });
    out.push_str(&content[..cut]);
    if truncated {
        out.push('…');
    }
    out
}

/// Pre-bulk-copy [`cass_build_preview`], retained doc-hidden for the same-binary A/B + parity test.
#[doc(hidden)]
#[must_use]
pub fn cass_build_preview_slow(content: &str, max_chars: usize) -> String {
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

#[must_use]
fn cass_build_content_prefix_and_preview(content: &str) -> (String, String) {
    const PREVIEW_MAX_CHARS: usize = 400;
    const CONTENT_PREFIX_MAX_BYTES: usize = 4 * 1024;

    let prefix_source = cass_prefix_source(content, CONTENT_PREFIX_MAX_BYTES);
    (
        cass_generate_edge_ngrams(prefix_source),
        cass_build_preview(content, PREVIEW_MAX_CHARS),
    )
}

/// The largest char-boundary prefix of `content` that is `≤ max_bytes` bytes.
///
/// The original walked `char_indices` forward from byte 0 — `O(max_bytes)` char decodes just to locate
/// the boundary near `max_bytes`. Since UTF-8 chars are ≤ 4 bytes, the largest boundary `≤ max_bytes`
/// is at most 3 bytes below `max_bytes`, so a backward `is_char_boundary` walk finds it in **≤ 4
/// steps** — O(1). Byte-for-byte identical result (`cass_prefix_source_matches_slow`).
fn cass_prefix_source(content: &str, max_bytes: usize) -> &str {
    if content.len() <= max_bytes {
        return content;
    }
    // `max_bytes < content.len()` here, so it is a valid index; `is_char_boundary(0)` is always true,
    // so the walk terminates within one UTF-8 char width.
    let mut end = max_bytes;
    while !content.is_char_boundary(end) {
        end -= 1;
    }
    &content[..end]
}

/// Pre-O(1) [`cass_prefix_source`], retained doc-hidden for the same-binary A/B + parity test.
#[doc(hidden)]
#[must_use]
pub fn cass_prefix_source_slow(content: &str, max_bytes: usize) -> &str {
    if content.len() <= max_bytes {
        return content;
    }
    let mut end = 0usize;
    for (byte_idx, _) in content.char_indices() {
        if byte_idx > max_bytes {
            break;
        }
        end = byte_idx;
    }
    &content[..end]
}

/// Doc-hidden bench wrapper (the shipped `cass_prefix_source` is private).
#[doc(hidden)]
#[must_use]
pub fn cass_prefix_source_fast_bench(content: &str, max_bytes: usize) -> usize {
    cass_prefix_source(content, max_bytes).len()
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
#[derive(Debug, Clone, PartialEq, Eq, Default)]
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
    /// NOT operator (negates the next term, phrase, or group).
    Not,
    /// A group-opening parenthesis.
    LParen,
    /// A group-closing parenthesis.
    RParen,
}

/// In-memory evidence from the real fallible Tantivy CASS query boundary.
///
/// This type deliberately retains plaintext only between the adapter and the
/// gauntlet's redaction step. It must never be serialized directly. The
/// profile-specific `ArtifactStore` v4 DTO owns any durable representation.
#[cfg(feature = "tantivy-oracle")]
#[derive(Debug)]
pub struct CassOracleProfileObservation {
    /// Shipping CASS sanitization of the raw query.
    pub sanitized_query: String,
    /// Ordered tokens passed to the actual Tantivy query builder.
    pub tokens: Vec<CassQueryToken>,
    /// Whether the builder selected its explicit Boolean grammar.
    pub has_boolean_operators: bool,
    /// Structured filters consumed by the real query builder.
    pub filters: CassQueryFilters,
    /// Native rank/tie/count result or the original typed failure.
    pub outcome: CassOracleProfileOutcome,
}

/// Total result of one fallible Tantivy CASS profile attempt.
#[cfg(feature = "tantivy-oracle")]
#[derive(Debug)]
pub enum CassOracleProfileOutcome {
    /// Query construction and retrieval succeeded.
    Success(crate::OracleQueryObservation),
    /// Query construction or retrieval failed with its original typed error.
    Error(SearchError),
}

/// Maximum raw-query bytes admitted by the dev-only CASS oracle boundary.
#[cfg(feature = "tantivy-oracle")]
pub const MAX_CASS_ORACLE_RAW_QUERY_BYTES: usize = 1_048_576;

/// Maximum token count retained by one dev-only CASS oracle observation.
#[cfg(feature = "tantivy-oracle")]
pub const MAX_CASS_ORACLE_TOKENS: usize = 20_000;

/// Maximum number of exact filter values admitted by one oracle request.
#[cfg(feature = "tantivy-oracle")]
pub const MAX_CASS_ORACLE_FILTER_VALUES: usize = 4_096;

/// Maximum bytes admitted for any individual exact filter value.
#[cfg(feature = "tantivy-oracle")]
pub const MAX_CASS_ORACLE_FILTER_VALUE_BYTES: usize = 4_096;

/// Maximum aggregate bytes admitted across all exact filter values.
#[cfg(feature = "tantivy-oracle")]
pub const MAX_CASS_ORACLE_FILTER_BYTES: usize = 1_048_576;

/// Maximum requested or expanded fetch cardinality for one oracle call.
#[cfg(feature = "tantivy-oracle")]
pub const MAX_CASS_ORACLE_FETCH_HITS: usize = 100_000;

/// Maximum UTF-8 bytes retained for one canonical CASS document identity.
#[cfg(feature = "tantivy-oracle")]
pub const MAX_CASS_ORACLE_DOCUMENT_ID_BYTES: usize = 1_024;

/// Maximum aggregate identity bytes retained across fetched and cutoff rows.
#[cfg(feature = "tantivy-oracle")]
pub const MAX_CASS_ORACLE_DOCUMENT_ID_AGGREGATE_BYTES: usize = 16 * 1_024 * 1_024;

#[cfg(feature = "tantivy-oracle")]
fn cass_oracle_preflight(
    raw_query: &str,
    filters: &CassQueryFilters,
    limit: usize,
    tie_expansion_limit: usize,
) -> SearchResult<()> {
    cass_oracle_bound(
        "raw_query_bytes",
        raw_query.len(),
        MAX_CASS_ORACLE_RAW_QUERY_BYTES,
    )?;
    cass_oracle_bound("limit", limit, MAX_CASS_ORACLE_FETCH_HITS)?;
    cass_oracle_bound(
        "tie_expansion_limit",
        tie_expansion_limit,
        MAX_CASS_ORACLE_FETCH_HITS,
    )?;
    let expanded_fetch = limit.saturating_add(tie_expansion_limit);
    cass_oracle_bound(
        "expanded_fetch_hits",
        expanded_fetch,
        MAX_CASS_ORACLE_FETCH_HITS,
    )?;

    let source_value = match &filters.source_filter {
        CassSourceFilter::SourceId(value) => Some(value.as_str()),
        CassSourceFilter::All | CassSourceFilter::Local | CassSourceFilter::Remote => None,
    };
    let filter_count = filters
        .agents
        .len()
        .checked_add(filters.workspaces.len())
        .and_then(|count| count.checked_add(usize::from(source_value.is_some())))
        .unwrap_or(usize::MAX);
    cass_oracle_bound(
        "filter_value_count",
        filter_count,
        MAX_CASS_ORACLE_FILTER_VALUES,
    )?;

    let mut aggregate_bytes = 0_usize;
    for value in filters
        .agents
        .iter()
        .chain(&filters.workspaces)
        .map(String::as_str)
        .chain(source_value)
    {
        cass_oracle_bound(
            "filter_value_bytes",
            value.len(),
            MAX_CASS_ORACLE_FILTER_VALUE_BYTES,
        )?;
        aggregate_bytes = aggregate_bytes.saturating_add(value.len());
        cass_oracle_bound(
            "aggregate_filter_bytes",
            aggregate_bytes,
            MAX_CASS_ORACLE_FILTER_BYTES,
        )?;
    }
    Ok(())
}

#[cfg(feature = "tantivy-oracle")]
fn cass_oracle_bound(field: &'static str, actual: usize, limit: usize) -> SearchResult<()> {
    if actual > limit {
        return Err(cass_oracle_invalid_bound(field, actual, limit));
    }
    Ok(())
}

#[cfg(feature = "tantivy-oracle")]
fn cass_oracle_invalid_bound(field: &'static str, actual: usize, limit: usize) -> SearchError {
    SearchError::InvalidConfig {
        field: format!("cass_oracle.{field}"),
        value: actual.to_string(),
        reason: format!("must be no greater than {limit}"),
    }
}

/// Sanitize query string to match the `hyphen_normalize` tokenizer for cass indexes.
///
/// The tokenizer preserves hyphens inside words (e.g. `bd-q3fy`, `POL-358`).
/// We therefore keep hyphens alongside `*` (wildcards) and `"` (phrases),
/// replacing all other non-alphanumeric characters with spaces so that query
/// terms align with indexed tokens.
#[must_use]
pub fn cass_sanitize_query(raw: &str) -> String {
    raw.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '*' || c == '"' || c == '-' {
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
            // tantivy-fst regexes already match the complete term and reject
            // zero-width assertions, so explicit `^` / `$` anchors are both
            // unnecessary and invalid here.
            Self::Suffix(core) => Some(format!(".*{}", cass_escape_regex(core))),
            Self::Substring(core) => Some(format!(".*{}.*", cass_escape_regex(core))),
            Self::Complex(full_term) => {
                let mut regex = String::with_capacity(full_term.len() * 2 + 2);

                if full_term.starts_with('*') {
                    regex.push_str(".*");
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
/// - ( ) grouping: a `(` opens a group only at the start of a word and a `)`
///   closes one only while a group is open, so `foo(bar)` stays one term
#[must_use]
pub fn cass_parse_boolean_query(query: &str) -> Vec<CassQueryToken> {
    match cass_parse_boolean_query_with_sink(query, |tokens, token| {
        tokens.push(token);
        Ok::<(), std::convert::Infallible>(())
    }) {
        Ok(tokens) => tokens,
        Err(never) => match never {},
    }
}

#[cfg(feature = "tantivy-oracle")]
fn cass_parse_boolean_query_bounded(
    query: &str,
    max_tokens: usize,
) -> Result<Vec<CassQueryToken>, usize> {
    cass_parse_boolean_query_with_sink(query, |tokens, token| {
        cass_push_query_token(tokens, token, max_tokens)
    })
}

fn cass_parse_boolean_query_with_sink<Error, PushToken>(
    query: &str,
    mut push_token: PushToken,
) -> Result<Vec<CassQueryToken>, Error>
where
    PushToken: FnMut(&mut Vec<CassQueryToken>, CassQueryToken) -> Result<(), Error>,
{
    let mut tokens = Vec::new();
    let mut chars = query.chars().peekable();
    let mut current_word = String::new();
    // Groups currently open. A `(` inside a word and a `)` with no group
    // open stay word characters, so code terms like `foo(bar)` survive.
    let mut open_groups = 0_usize;

    while let Some(c) = chars.next() {
        match c {
            '(' if current_word.is_empty() => {
                push_token(&mut tokens, CassQueryToken::LParen)?;
                open_groups += 1;
            }
            ')' if open_groups > 0 => {
                if !current_word.is_empty() {
                    let token = cass_classify_query_word(std::mem::take(&mut current_word));
                    push_token(&mut tokens, token)?;
                }
                push_token(&mut tokens, CassQueryToken::RParen)?;
                open_groups -= 1;
            }
            '"' => {
                if !current_word.is_empty() {
                    push_token(
                        &mut tokens,
                        CassQueryToken::Term(std::mem::take(&mut current_word)),
                    )?;
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
                // A present quoted operand must survive even when empty.
                // Otherwise required AND "" silently loses a requirement.
                push_token(&mut tokens, CassQueryToken::Phrase(phrase))?;
            }
            '&' if chars.peek() == Some(&'&') => {
                chars.next();
                if !current_word.is_empty() {
                    push_token(
                        &mut tokens,
                        CassQueryToken::Term(std::mem::take(&mut current_word)),
                    )?;
                }
                push_token(&mut tokens, CassQueryToken::And)?;
            }
            '|' if chars.peek() == Some(&'|') => {
                chars.next();
                if !current_word.is_empty() {
                    push_token(
                        &mut tokens,
                        CassQueryToken::Term(std::mem::take(&mut current_word)),
                    )?;
                }
                push_token(&mut tokens, CassQueryToken::Or)?;
            }
            '-' if current_word.is_empty() => {
                push_token(&mut tokens, CassQueryToken::Not)?;
            }
            ' ' | '\t' | '\n' => {
                if !current_word.is_empty() {
                    let word = std::mem::take(&mut current_word);
                    let token = cass_classify_query_word(word);
                    push_token(&mut tokens, token)?;
                }
            }
            _ => current_word.push(c),
        }
    }

    if !current_word.is_empty() {
        let token = cass_classify_query_word(current_word);
        push_token(&mut tokens, token)?;
    }

    Ok(tokens)
}

fn cass_classify_query_word(word: String) -> CassQueryToken {
    if word.eq_ignore_ascii_case("AND") {
        CassQueryToken::And
    } else if word.eq_ignore_ascii_case("OR") {
        CassQueryToken::Or
    } else if word.eq_ignore_ascii_case("NOT") {
        CassQueryToken::Not
    } else {
        CassQueryToken::Term(word)
    }
}

#[cfg(feature = "tantivy-oracle")]
fn cass_push_query_token(
    tokens: &mut Vec<CassQueryToken>,
    token: CassQueryToken,
    max_tokens: usize,
) -> Result<(), usize> {
    if tokens.len() >= max_tokens {
        return Err(tokens.len().saturating_add(1));
    }
    tokens.push(token);
    Ok(())
}

#[must_use]
pub fn cass_has_boolean_operators(query: &str) -> bool {
    let tokens = cass_parse_boolean_query(query);
    cass_tokens_have_boolean_operators(&tokens)
}

fn cass_tokens_have_boolean_operators(tokens: &[CassQueryToken]) -> bool {
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
    if !cass_needs_indexed_fragments(raw) {
        return cass_sanitize_query(raw)
            .split_whitespace()
            .map(str::to_owned)
            .collect();
    }
    // Preserve original case and combining marks inside an affected word.
    // Repeated hyphens and ASCII punctuation still separate independent
    // unquoted operands; a single compound hyphen retains its indexed shape.
    raw.split("--")
        .flat_map(|part| {
            part.split(|ch: char| {
                ch.is_whitespace()
                    || (ch.is_ascii() && !ch.is_ascii_alphanumeric() && !matches!(ch, '*' | '-'))
            })
        })
        .filter(|part| !part.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Normalize phrase text without losing required words or their positions.
fn cass_normalize_phrase_terms(raw: &str) -> SearchResult<(CassIndexedFragments, bool)> {
    let parts = cass_normalize_term_parts(raw)
        .into_iter()
        .map(|part| part.trim_matches('*').to_owned())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    if parts.is_empty() {
        return Err(cass_normalization_error("phrase has no searchable operands"));
    }
    for part in &parts {
        // Edge stars retain the existing quoted behavior. An interior star
        // cannot be turned into a separator or silently dropped under NOT.
        if part.contains('*')
            || (!cass_ascii_operand_admitted(part)
                && cass_analyzed_fragments(part)?.terms.is_empty())
        {
            return Err(cass_normalization_error(
                "phrase contains an unsupported or unsearchable operand",
            ));
        }
    }
    if cass_needs_indexed_fragments(raw) {
        return Ok((cass_analyzed_fragments(&parts.join(" "))?, true));
    }
    if parts.len() > CASS_MAX_QUERY_FRAGMENTS {
        return Err(cass_normalization_error(
            "phrase has too many indexed fragments",
        ));
    }
    let terms = parts.iter().map(|part| part.to_lowercase()).collect::<Vec<_>>();
    let positions = (0..terms.len()).collect();
    Ok((
        CassIndexedFragments {
            terms,
            positions,
            ends_with_token: true,
        },
        false,
    ))
}

/// The complement of `query`: Tantivy reads a query of `MustNot` clauses
/// alone as match-none, so exclusions are anchored on `AllQuery`.
fn cass_wrap_negated_clause(query: Box<dyn Query>) -> Box<dyn Query> {
    Box::new(BooleanQuery::new(vec![
        (Occur::Must, Box::new(AllQuery)),
        (Occur::MustNot, query),
    ]))
}

/// A lowered CASS operand and whether it stands negated.
struct CassOracleNode {
    expr: CassOracleExpr,
    negative: bool,
}

/// Conjunctions keep their clause list, so a conjunction at the root is
/// emitted as flat top-level clauses: the query tree, and so the score, of
/// every query the pre-grouping builder read the same way is unchanged.
enum CassOracleExpr {
    Leaf(Box<dyn Query>),
    /// `Must`/`MustNot` clauses, anchored on `AllQuery` when all are negative.
    And(Vec<(Occur, Box<dyn Query>)>),
    /// Disjunction operands, negative ones already complemented.
    Or(Vec<Box<dyn Query>>),
}

impl CassOracleExpr {
    fn into_query(self) -> Box<dyn Query> {
        match self {
            Self::Leaf(query) => query,
            Self::And(clauses) => Box::new(BooleanQuery::new(clauses)),
            Self::Or(operands) => Box::new(BooleanQuery::new(
                operands
                    .into_iter()
                    .map(|query| (Occur::Should, query))
                    .collect(),
            )),
        }
    }
}

/// Disjunction of `operands`. A negative operand denotes its complement, so
/// it is wrapped in `All + MustNot` before joining the disjunction.
fn cass_combine_or(mut operands: Vec<CassOracleNode>) -> Option<CassOracleNode> {
    if operands.len() <= 1 {
        return operands.pop();
    }
    let operands = operands
        .into_iter()
        .map(|node| {
            let query = node.expr.into_query();
            if node.negative {
                cass_wrap_negated_clause(query)
            } else {
                query
            }
        })
        .collect();
    Some(CassOracleNode {
        expr: CassOracleExpr::Or(operands),
        negative: false,
    })
}

/// Conjunction of `operands`: positives are `Must`, negatives `MustNot`. A
/// conjunction of negatives alone is anchored on `AllQuery`, since
/// exclusions need a universe to exclude from.
fn cass_combine_and(mut operands: Vec<CassOracleNode>) -> Option<CassOracleNode> {
    if operands.len() <= 1 {
        return operands.pop();
    }
    let mut clauses = operands
        .into_iter()
        .map(|node| {
            let occur = if node.negative {
                Occur::MustNot
            } else {
                Occur::Must
            };
            (occur, node.expr.into_query())
        })
        .collect::<Vec<_>>();
    if clauses.iter().all(|(occur, _)| *occur == Occur::MustNot) {
        clauses.insert(0, (Occur::Must, Box::new(AllQuery)));
    }
    Some(CassOracleNode {
        expr: CassOracleExpr::And(clauses),
        negative: false,
    })
}

/// Recursive-descent CASS grammar, the one Quill's `CassQueryParser` parses:
///
/// ```text
/// or      := and (OR and)*
/// and     := unary ([AND] unary)*      adjacent operands are an implied AND
/// unary   := NOT* primary              an odd count of NOTs negates
/// primary := TERM | PHRASE | '(' or ')'
/// ```
///
/// Malformed input is recovered the way Quill recovers it: an unclosed `(`
/// closes at the end, and an empty group, a dangling operator, a NOT with no
/// operand and an operand that lowers to nothing are skipped. This adapter
/// has no diagnostic channel, so the recovery is silent.
struct CassOracleGrammar<'a> {
    tokens: &'a [CassQueryToken],
    position: usize,
    /// Parentheses entered and not yet closed.
    open_groups: usize,
    fields: &'a CassFields,
    regex_query_factory: CassRegexQueryFactory,
}

impl CassOracleGrammar<'_> {
    fn peek(&self) -> Option<&CassQueryToken> {
        self.tokens.get(self.position)
    }

    fn parse_or(&mut self) -> SearchResult<Option<CassOracleNode>> {
        let mut operands = Vec::new();
        loop {
            match self.peek() {
                None => break,
                Some(CassQueryToken::RParen) if self.open_groups > 0 => break,
                // The lexer emits `)` only inside a group; any other is skipped.
                Some(CassQueryToken::Or | CassQueryToken::RParen) => {
                    self.position += 1;
                    continue;
                }
                Some(_) => {}
            }
            if let Some(node) = self.parse_and()? {
                operands.push(node);
            }
        }
        Ok(cass_combine_or(operands))
    }

    fn parse_and(&mut self) -> SearchResult<Option<CassOracleNode>> {
        let mut operands = Vec::new();
        loop {
            match self.peek() {
                None | Some(CassQueryToken::Or | CassQueryToken::RParen) => break,
                Some(CassQueryToken::And) => {
                    self.position += 1;
                    continue;
                }
                Some(_) => {}
            }
            if let Some(node) = self.parse_unary()? {
                operands.push(node);
            }
        }
        Ok(cass_combine_and(operands))
    }

    fn parse_unary(&mut self) -> SearchResult<Option<CassOracleNode>> {
        let mut negated = false;
        while matches!(self.peek(), Some(CassQueryToken::Not)) {
            negated = !negated;
            self.position += 1;
        }
        if !matches!(
            self.peek(),
            Some(CassQueryToken::Term(_) | CassQueryToken::Phrase(_) | CassQueryToken::LParen)
        ) {
            return Ok(None);
        }
        Ok(self.parse_primary()?.map(|mut node| {
            node.negative ^= negated;
            node
        }))
    }

    fn parse_primary(&mut self) -> SearchResult<Option<CassOracleNode>> {
        let Some(token) = self.tokens.get(self.position) else {
            return Ok(None);
        };
        self.position += 1;
        let query = match token {
            CassQueryToken::Term(term) if term.trim().is_empty() => None,
            CassQueryToken::Term(term) => Some(
                cass_build_compound_term_query(
                    &cass_normalize_term_parts(term),
                    self.fields,
                    self.regex_query_factory,
                )?
                .ok_or_else(|| cass_normalization_error("operand has no searchable terms"))?,
            ),
            CassQueryToken::Phrase(phrase) => {
                let (fragments, normalized) = cass_normalize_phrase_terms(phrase)?;
                if normalized {
                    Some(cass_build_indexed_fragments_query(
                        &fragments,
                        false,
                        self.fields,
                    ))
                } else {
                    Some(
                        cass_build_phrase_query(
                            phrase,
                            &fragments.terms,
                            self.fields,
                            self.regex_query_factory,
                        )?
                        .ok_or_else(|| {
                            cass_normalization_error("phrase has no searchable terms")
                        })?,
                    )
                }
            }
            CassQueryToken::LParen => {
                self.open_groups += 1;
                let inner = self.parse_or()?;
                self.open_groups -= 1;
                if matches!(self.peek(), Some(CassQueryToken::RParen)) {
                    self.position += 1;
                }
                return Ok(inner);
            }
            // Operators never reach here: parse_unary admits only operands.
            CassQueryToken::And
            | CassQueryToken::Or
            | CassQueryToken::Not
            | CassQueryToken::RParen => None,
        };
        Ok(query.map(|query| CassOracleNode {
            expr: CassOracleExpr::Leaf(query),
            negative: false,
        }))
    }
}

/// Returns `true` when the string contains at least one CJK character.
#[inline]
fn contains_cjk(s: &str) -> bool {
    s.chars().any(is_cjk)
}

/// Decompose a CJK string into character bigrams (matching `CjkBigramDecompose`).
/// For a single CJK character, returns the character itself as a unigram.
fn cjk_bigrams(s: &str) -> Vec<String> {
    let chars: Vec<char> = s.chars().filter(|c| is_cjk(*c)).collect();
    if chars.len() <= 1 {
        return chars.iter().map(|c| c.to_string()).collect();
    }
    (0..chars.len() - 1)
        .map(|i| {
            let mut b = String::with_capacity(8);
            b.push(chars[i]);
            b.push(chars[i + 1]);
            b
        })
        .collect()
}

/// Build a query that requires ALL bigrams to match in at least one field.
/// This mirrors how the tokenizer indexes CJK text.
#[inline]
fn cass_term_query_fields(fields: &CassFields) -> [(Field, IndexRecordOption); 4] {
    [
        (fields.title, IndexRecordOption::WithFreqsAndPositions),
        (fields.content, IndexRecordOption::WithFreqsAndPositions),
        (fields.title_prefix, IndexRecordOption::Basic),
        (fields.content_prefix, IndexRecordOption::Basic),
    ]
}

fn cass_build_cjk_term_query(bigrams: &[String], fields: &CassFields) -> Option<Box<dyn Query>> {
    if bigrams.is_empty() {
        return None;
    }

    // For each bigram, require it to appear in at least one searchable field.
    let mut bigram_musts: Vec<(Occur, Box<dyn Query>)> = Vec::new();
    for bigram in bigrams {
        let mut field_shoulds: Vec<(Occur, Box<dyn Query>)> = Vec::new();
        for (field, index_record_option) in cass_term_query_fields(fields) {
            field_shoulds.push((
                Occur::Should,
                Box::new(TermQuery::new(
                    Term::from_field_text(field, bigram),
                    index_record_option,
                )),
            ));
        }
        bigram_musts.push((Occur::Must, Box::new(BooleanQuery::new(field_shoulds))));
    }

    match bigram_musts.len() {
        0 => None,
        1 => bigram_musts.pop().map(|(_, q)| q),
        _ => Some(Box::new(BooleanQuery::new(bigram_musts))),
    }
}

type CassRegexQueryFactory = fn(Field, &str) -> SearchResult<RegexQuery>;

/// Build query clauses for a single term based on its wildcard pattern.
fn cass_build_term_query_clauses(
    pattern: &CassWildcardPattern,
    fields: &CassFields,
    regex_query_factory: CassRegexQueryFactory,
) -> SearchResult<Vec<(Occur, Box<dyn Query>)>> {
    let mut shoulds: Vec<(Occur, Box<dyn Query>)> = Vec::new();

    match pattern {
        CassWildcardPattern::Exact(term) | CassWildcardPattern::Prefix(term) => {
            if term.is_empty() {
                return Ok(shoulds);
            }
            // CJK terms must be decomposed into bigrams to match the indexed tokens.
            if contains_cjk(term) {
                let bigrams = cjk_bigrams(term);
                if let Some(q) = cass_build_cjk_term_query(&bigrams, fields) {
                    shoulds.push((Occur::Should, q));
                }
                return Ok(shoulds);
            }
            for (field, index_record_option) in cass_term_query_fields(fields) {
                shoulds.push((
                    Occur::Should,
                    Box::new(TermQuery::new(
                        Term::from_field_text(field, term),
                        index_record_option,
                    )),
                ));
            }
        }
        CassWildcardPattern::Suffix(_)
        | CassWildcardPattern::Substring(_)
        | CassWildcardPattern::Complex(_) => {
            if let Some(regex_pattern) = pattern.to_regex() {
                let content_query = regex_query_factory(fields.content, &regex_pattern)?;
                let title_query = regex_query_factory(fields.title, &regex_pattern)?;
                shoulds.push((Occur::Should, Box::new(content_query)));
                shoulds.push((Occur::Should, Box::new(title_query)));
            }
        }
    }

    Ok(shoulds)
}

/// Build a compound query that requires all term parts to match (implicit AND).
fn cass_build_compound_term_query(
    parts: &[String],
    fields: &CassFields,
    regex_query_factory: CassRegexQueryFactory,
) -> SearchResult<Option<Box<dyn Query>>> {
    let mut subqueries: Vec<Box<dyn Query>> = Vec::new();
    for part in parts {
        let pattern = CassWildcardPattern::parse(part);
        let core = part.trim_matches('*');
        if cass_ascii_operand_admitted(core) {
            let term_shoulds =
                cass_build_term_query_clauses(&pattern, fields, regex_query_factory)?;
            subqueries.push(Box::new(BooleanQuery::new(term_shoulds)));
            continue;
        }
        let fragments = cass_analyzed_fragments(core)?;
        if fragments.terms.is_empty() {
            return Err(cass_normalization_error("operand has no searchable terms"));
        }
        if cass_needs_indexed_fragments(part) {
            let prefix = match pattern {
                CassWildcardPattern::Exact(_) => false,
                CassWildcardPattern::Prefix(_) => {
                    fragments.ends_with_token
                        && fragments.terms.last().is_some_and(|term| term.is_ascii())
                }
                CassWildcardPattern::Suffix(_)
                | CassWildcardPattern::Substring(_)
                | CassWildcardPattern::Complex(_) => {
                    return Err(cass_normalization_error(
                        "affected operands support only exact or trailing-prefix queries",
                    ));
                }
            };
            subqueries.push(cass_build_indexed_fragments_query(
                &fragments, prefix, fields,
            ));
            continue;
        }
        let term_shoulds = cass_build_term_query_clauses(&pattern, fields, regex_query_factory)?;
        if term_shoulds.is_empty() {
            return Err(cass_normalization_error("operand has no searchable terms"));
        }
        subqueries.push(Box::new(BooleanQuery::new(term_shoulds)));
    }

    Ok(match subqueries.len() {
        0 => None,
        1 => subqueries.pop(),
        _ => {
            let musts = subqueries.into_iter().map(|q| (Occur::Must, q)).collect();
            Some(Box::new(BooleanQuery::new(musts)))
        }
    })
}

/// Keep an affected word adjacent, ordered and inside one logical field.
/// CJK bigrams share positions but are all required, so they retain same-field
/// conjunction. Explicit prefixes retain the existing edge-prefix columns;
/// those columns have no positions and their historic length bound remains.
fn cass_build_indexed_fragments_query(
    fragments: &CassIndexedFragments,
    prefix: bool,
    fields: &CassFields,
) -> Box<dyn Query> {
    if fragments.terms.is_empty() {
        return cass_match_none_query();
    }
    let positional = !prefix && !fragments.terms.iter().any(|term| contains_cjk(term));
    let mut field_queries: Vec<(Occur, Box<dyn Query>)> = Vec::with_capacity(2);
    for (field, prefix_field) in [
        (fields.title, fields.title_prefix),
        (fields.content, fields.content_prefix),
    ] {
        if positional && fragments.terms.len() > 1 {
            let terms = fragments
                .terms
                .iter()
                .zip(&fragments.positions)
                .map(|(term, position)| (*position, Term::from_field_text(field, term)))
                .collect();
            field_queries.push((
                Occur::Should,
                Box::new(CassPhraseQuery(PhraseQuery::new_with_offset(terms))),
            ));
            continue;
        }
        let mut musts: Vec<(Occur, Box<dyn Query>)> = Vec::with_capacity(fragments.terms.len());
        for (index, fragment) in fragments.terms.iter().enumerate() {
            let exact: Box<dyn Query> = Box::new(TermQuery::new(
                Term::from_field_text(field, fragment),
                IndexRecordOption::WithFreqsAndPositions,
            ));
            let query = if prefix && index + 1 == fragments.terms.len() {
                Box::new(BooleanQuery::new(vec![
                    (Occur::Should, exact),
                    (
                        Occur::Should,
                        Box::new(TermQuery::new(
                            Term::from_field_text(prefix_field, fragment),
                            IndexRecordOption::Basic,
                        )),
                    ),
                ])) as Box<dyn Query>
            } else {
                exact
            };
            musts.push((Occur::Must, query));
        }
        field_queries.push((Occur::Should, Box::new(BooleanQuery::new(musts))));
    }
    Box::new(BooleanQuery::new(field_queries))
}

/// Keep CASS phrase scorers within the ordinary `DocSet` seek contract.
///
/// Tantivy 0.26.2's `Exclude` calls `seek_danger` with increasing candidates that
/// can precede the phrase's current document. That is valid for `seek_danger`,
/// but `PhraseScorer` forwards them to a forward-only intersection and asserts
/// that the candidate is at least its current document (bd-nqeb4). Use the
/// exact, valid-state default `seek_danger` instead; no panic is caught and no
/// phrase is dropped. Weight, positions, scores and term visitors stay native.
#[derive(Clone, Debug)]
struct CassPhraseQuery(PhraseQuery);

impl Query for CassPhraseQuery {
    fn weight(&self, scoring: EnableScoring<'_>) -> tantivy::Result<Box<dyn Weight>> {
        Ok(Box::new(CassPhraseWeight(self.0.weight(scoring)?)))
    }

    fn query_terms<'a>(&'a self, visitor: &mut dyn FnMut(&'a Term, bool)) {
        self.0.query_terms(visitor);
    }
}

struct CassPhraseWeight(Box<dyn Weight>);

impl Weight for CassPhraseWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> tantivy::Result<Box<dyn Scorer>> {
        // Keep the wrapper for empty scorers too. Otherwise BooleanWeight
        // can collapse the title/content union to EmptyScorer, whose default
        // seek rejects an earlier document while explaining an OR operand.
        Ok(Box::new(CassPhraseScorer(self.0.scorer(reader, boost)?)))
    }

    fn explain(&self, reader: &SegmentReader, doc: DocId) -> tantivy::Result<Explanation> {
        let mut scorer = self.scorer(reader, 1.0)?;
        if doc >= reader.max_doc() || scorer.doc() > doc || scorer.seek(doc) != doc {
            return Err(TantivyError::InvalidArgument(format!(
                "Document #({doc}) does not match"
            )));
        }
        // The matching document cannot precede the original scorer's first
        // match, so its native explanation can now seek forward safely.
        self.0.explain(reader, doc)
    }
}

struct CassPhraseScorer(Box<dyn Scorer>);

impl DocSet for CassPhraseScorer {
    fn advance(&mut self) -> DocId {
        if self.0.doc() == TERMINATED {
            TERMINATED
        } else {
            self.0.advance()
        }
    }

    fn seek(&mut self, target: DocId) -> DocId {
        let current = self.0.doc();
        if current >= target {
            current
        } else {
            self.0.seek(target)
        }
    }

    // Deliberately inherit DocSet::seek_danger: it uses this exact seek and
    // never leaves the inner phrase scorer in an invalid approximate state.
    fn doc(&self) -> DocId {
        self.0.doc()
    }

    fn size_hint(&self) -> u32 {
        self.0.size_hint()
    }

    fn cost(&self) -> u64 {
        self.0.cost()
    }
}

impl Scorer for CassPhraseScorer {
    fn score(&mut self) -> Score {
        self.0.score()
    }
}

/// Build a phrase query (exact order) across title/content fields.
fn cass_build_phrase_query(
    raw: &str,
    terms: &[String],
    fields: &CassFields,
    regex_query_factory: CassRegexQueryFactory,
) -> SearchResult<Option<Box<dyn Query>>> {
    if terms.is_empty() {
        return Ok(None);
    }
    if terms.len() == 1 {
        return cass_build_compound_term_query(terms, fields, regex_query_factory);
    }

    // CJK phrases retain the legacy non-positional AND of bigrams. Lower the
    // original operands so an affected word stays one exact, same-field group:
    // passing flattened fragments would restore prefix/cross-field matches.
    if terms.iter().any(|t| contains_cjk(t)) {
        let parts = cass_normalize_term_parts(raw)
            .into_iter()
            .map(|part| part.trim_matches('*').to_owned())
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>();
        return cass_build_compound_term_query(&parts, fields, regex_query_factory);
    }

    let mut shoulds: Vec<(Occur, Box<dyn Query>)> = Vec::new();
    for field in [fields.title, fields.content] {
        let phrase_terms = terms
            .iter()
            .map(|t| Term::from_field_text(field, t))
            .collect::<Vec<_>>();
        shoulds.push((
            Occur::Should,
            Box::new(CassPhraseQuery(PhraseQuery::new(phrase_terms))),
        ));
    }
    Ok(Some(Box::new(BooleanQuery::new(shoulds))))
}

struct CassIndexedFragments {
    terms: Vec<String>,
    positions: Vec<usize>,
    ends_with_token: bool,
}

/// Analyze original source with the complete registered prefix pipeline.
/// It retains compound tokens and the positional fields' CJK bigrams without
/// adding hyphen alternatives that would broaden an exact compound. No durable
/// index bytes change. Retain positions and refuse filtered required runs
/// instead of compacting a phrase or weakening an expression under OR/NOT.
fn cass_analyzed_fragments(raw: &str) -> SearchResult<CassIndexedFragments> {
    let mut analyzer = cass_prefix_analyzer();
    let mut stream = analyzer.token_stream(raw);
    let mut terms = Vec::new();
    let mut positions = Vec::new();
    let mut covered_to = 0;
    let mut filtered = false;
    while stream.advance() {
        let token = stream.token();
        if token.offset_from > covered_to {
            filtered |= raw[covered_to..token.offset_from]
                .bytes()
                .any(|byte| byte.is_ascii_alphanumeric());
        }
        covered_to = covered_to.max(token.offset_to);
        if terms.len() == CASS_MAX_QUERY_FRAGMENTS {
            return Err(cass_normalization_error(
                "operand has too many indexed fragments",
            ));
        }
        terms.push(token.text.clone());
        positions.push(token.position);
    }
    filtered |= raw[covered_to..]
        .bytes()
        .any(|byte| byte.is_ascii_alphanumeric());
    if filtered {
        return Err(cass_normalization_error(
            "an indexed fragment exceeds 256 bytes",
        ));
    }
    Ok(CassIndexedFragments {
        terms,
        positions,
        ends_with_token: covered_to == raw.len(),
    })
}

const CASS_MAX_QUERY_FRAGMENTS: usize = 256;
const CASS_MAX_QUERY_LENGTH: usize = 10_000;
const CASS_MAX_QUERY_NESTING: usize = 128;

fn cass_needs_indexed_fragments(raw: &str) -> bool {
    raw.chars()
        .any(|ch| !ch.is_ascii() && !is_cass_tokenizer_cjk(ch))
}

/// A short ASCII operand containing a letter or digit necessarily contributes
/// an admitted token. Avoid analyzer allocations on the existing ASCII path.
fn cass_ascii_operand_admitted(raw: &str) -> bool {
    raw.len() <= 256
        && raw.is_ascii()
        && raw.bytes().any(|byte| byte.is_ascii_alphanumeric())
}

fn cass_normalization_error(reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: "cass_query.normalization".to_owned(),
        value: "query rejected".to_owned(),
        reason: reason.to_owned(),
    }
}

fn cass_query_normalization_preflight(raw: &str) -> SearchResult<()> {
    if !raw.trim().is_empty()
        && cass_needs_indexed_fragments(raw)
        && raw.chars().take(CASS_MAX_QUERY_LENGTH + 1).count() > CASS_MAX_QUERY_LENGTH
    {
        return Err(cass_normalization_error(
            "query exceeds 10000 Unicode scalar values",
        ));
    }
    Ok(())
}

/// Count lexer groups, excluding parentheses inside phrases and code terms,
/// before the recursive grammar can consume adversarially deep input.
fn cass_query_nesting_preflight(tokens: &[CassQueryToken]) -> SearchResult<()> {
    let mut depth = 0_usize;
    for token in tokens {
        match token {
            CassQueryToken::LParen => {
                depth += 1;
                if depth > CASS_MAX_QUERY_NESTING {
                    return Err(cass_normalization_error("query exceeds 128 nested groups"));
                }
            }
            CassQueryToken::RParen => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    Ok(())
}

/// Build Tantivy query clauses from boolean tokens.
///
/// Standard precedence, as cass documents it and Quill parses it: NOT binds
/// tightest, then AND (explicit, `&&`, or implied between adjacent operands),
/// then OR (`OR`, `||`); parentheses group; negation is parity-based.
fn cass_build_boolean_query_clauses(
    tokens: &[CassQueryToken],
    fields: &CassFields,
    regex_query_factory: CassRegexQueryFactory,
) -> SearchResult<Vec<(Occur, Box<dyn Query>)>> {
    let mut grammar = CassOracleGrammar {
        tokens,
        position: 0,
        open_groups: 0,
        fields,
        regex_query_factory,
    };
    let Some(root) = grammar.parse_or()? else {
        return Ok(Vec::new());
    };
    Ok(match root {
        // A negative root denotes the complement of its operand.
        CassOracleNode {
            expr,
            negative: true,
        } => vec![
            (Occur::Must, Box::new(AllQuery)),
            (Occur::MustNot, expr.into_query()),
        ],
        // Flat, so a positive conjunction such as `auth AND NOT deprecated`
        // stays unanchored and gains no AllQuery score.
        CassOracleNode {
            expr: CassOracleExpr::And(clauses),
            negative: false,
        } => clauses,
        CassOracleNode {
            expr,
            negative: false,
        } => vec![(Occur::Must, expr.into_query())],
    })
}

fn cass_match_none_query() -> Box<dyn Query> {
    Box::new(BooleanQuery::new(vec![
        (Occur::Must, Box::new(AllQuery)),
        (Occur::MustNot, Box::new(AllQuery)),
    ]))
}

fn cass_fail_closed_tantivy_query(
    raw_query: &str,
    result: SearchResult<Box<dyn Query>>,
) -> Box<dyn Query> {
    result.unwrap_or_else(|error| {
        warn!(
            query_bytes = raw_query.len(),
            error = %error,
            "CASS query construction failed; returning an explicit match-none query"
        );
        cass_match_none_query()
    })
}

/// Build a cass-compatible Tantivy query for `raw_query`, applying filters.
///
/// Returns an `AllQuery` when the query is empty.
/// Regex construction errors are logged and converted to an explicit
/// match-none query because this compatibility API predates fallible returns.
#[must_use]
pub fn cass_build_tantivy_query(
    raw_query: &str,
    filters: &CassQueryFilters,
    fields: &CassFields,
) -> Box<dyn Query> {
    if let Err(error) = cass_query_normalization_preflight(raw_query) {
        return cass_fail_closed_tantivy_query(raw_query, Err(error));
    }
    let tokens = cass_parse_boolean_query(raw_query);
    let has_boolean_operators = !tokens.is_empty() && cass_has_boolean_operators(raw_query);
    cass_fail_closed_tantivy_query(
        raw_query,
        cass_try_build_tantivy_query_from_tokens(
            raw_query,
            tokens,
            has_boolean_operators,
            filters,
            fields,
            cass_regex_query_cached,
        ),
    )
}

/// Single-parse candidate retained for same-binary performance and query-tree
/// parity checks. Production remains on [`cass_build_tantivy_query`] because
/// the candidate did not clear the paired A/A null floor.
#[cfg(any(test, feature = "bench-internals"))]
#[doc(hidden)]
#[must_use]
pub fn cass_build_tantivy_query_single_parse(
    raw_query: &str,
    filters: &CassQueryFilters,
    fields: &CassFields,
) -> Box<dyn Query> {
    if let Err(error) = cass_query_normalization_preflight(raw_query) {
        return cass_fail_closed_tantivy_query(raw_query, Err(error));
    }
    let tokens = cass_parse_boolean_query(raw_query);
    let has_boolean_operators = cass_tokens_have_boolean_operators(&tokens);
    cass_fail_closed_tantivy_query(
        raw_query,
        cass_try_build_tantivy_query_from_tokens(
            raw_query,
            tokens,
            has_boolean_operators,
            filters,
            fields,
            cass_regex_query_cached,
        ),
    )
}

fn cass_try_build_tantivy_query_from_tokens(
    raw_query: &str,
    tokens: Vec<CassQueryToken>,
    has_boolean_operators: bool,
    filters: &CassQueryFilters,
    fields: &CassFields,
    regex_query_factory: CassRegexQueryFactory,
) -> SearchResult<Box<dyn Query>> {
    cass_query_normalization_preflight(raw_query)?;
    cass_query_nesting_preflight(&tokens)?;
    let mut clauses: Vec<(Occur, Box<dyn Query>)> = Vec::new();

    if raw_query.trim().is_empty() {
        clauses.push((Occur::Must, Box::new(AllQuery)));
    } else if has_boolean_operators {
        clauses.extend(cass_build_boolean_query_clauses(
            &tokens,
            fields,
            regex_query_factory,
        )?);
    } else {
        for token in tokens {
            if let CassQueryToken::Term(term_str) = token {
                if term_str.trim().is_empty() {
                    continue;
                }
                let parts = cass_normalize_term_parts(&term_str);
                let term_query =
                    cass_build_compound_term_query(&parts, fields, regex_query_factory)?
                        .ok_or_else(|| {
                            cass_normalization_error("operand has no searchable terms")
                        })?;
                clauses.push((Occur::Must, term_query));
            }
        }
    }

    // Only deliberate blank input may browse. Syntax recovery, punctuation,
    // unsupported text and all-star operands must not turn a nonempty query
    // into All, including when structured filters would supply its only clauses.
    if clauses.is_empty() {
        return Err(cass_normalization_error("query has no searchable operands"));
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

    Ok(match clauses.len() {
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
    })
}

#[cfg(test)]
mod cass_query_tests {
    use super::*;
    use tantivy::schema::Value as _;

    fn fields() -> CassFields {
        let schema = cass_build_schema();
        cass_fields_from_schema(&schema).expect("cass fields")
    }

    fn scored_cass_results(
        searcher: &tantivy::Searcher,
        fields: &CassFields,
        raw_query: &str,
        filters: &CassQueryFilters,
    ) -> std::collections::BTreeMap<u64, f32> {
        let query = cass_build_tantivy_query(raw_query, filters, fields);
        searcher
            .search(
                &*query,
                &tantivy::collector::TopDocs::with_limit(64).order_by_score(),
            )
            .expect("search CASS Boolean fixture")
            .into_iter()
            .map(|(score, address)| {
                let document: tantivy::TantivyDocument = searcher
                    .doc(address)
                    .expect("load CASS Boolean fixture document");
                let msg_idx = document
                    .get_first(fields.msg_idx)
                    .and_then(|value| value.as_u64())
                    .expect("stored CASS fixture msg_idx");
                (msg_idx, score)
            })
            .collect()
    }

    fn cass_result_ids(
        searcher: &tantivy::Searcher,
        fields: &CassFields,
        raw_query: &str,
        filters: &CassQueryFilters,
    ) -> Vec<u64> {
        scored_cass_results(searcher, fields, raw_query, filters)
            .into_keys()
            .collect()
    }

    fn cass_result_fixture() -> (tempfile::TempDir, CassTantivyIndex) {
        let directory = tempfile::tempdir().expect("temporary CASS result index directory");
        let mut index =
            CassTantivyIndex::open_or_create_with_writer_parallelism(directory.path(), 1)
                .expect("create CASS result index");
        let documents = [
            CassDocumentRef {
                agent: "claude",
                workspace: Some("/alpha"),
                workspace_original: Some("/alpha"),
                source_path: "/fixture/0.jsonl",
                msg_idx: 0,
                created_at: Some(100),
                title: Some("Current Alpha"),
                content: "alpha active",
                source_id: "local-a",
                origin_kind: "local",
                origin_host: None,
                conversation_id: None,
            },
            CassDocumentRef {
                agent: "claude",
                workspace: Some("/alpha"),
                workspace_original: Some("/alpha"),
                source_path: "/fixture/1.jsonl",
                msg_idx: 1,
                created_at: Some(200),
                title: Some("Legacy Alpha"),
                content: "alpha deprecated",
                source_id: "local-b",
                origin_kind: "local",
                origin_host: None,
                conversation_id: None,
            },
            CassDocumentRef {
                agent: "codex",
                workspace: Some("/beta"),
                workspace_original: Some("/beta"),
                source_path: "/fixture/2.jsonl",
                msg_idx: 2,
                created_at: Some(300),
                title: Some("Current Beta"),
                content: "beta active",
                source_id: "remote-a",
                origin_kind: "ssh",
                origin_host: Some("worker-a"),
                conversation_id: None,
            },
            CassDocumentRef {
                agent: "claude",
                workspace: Some("/gamma"),
                workspace_original: Some("/gamma"),
                source_path: "/fixture/3.jsonl",
                msg_idx: 3,
                created_at: Some(400),
                title: Some("Current Gamma"),
                content: "gamma active",
                source_id: "remote-b",
                origin_kind: "ssh",
                origin_host: Some("worker-b"),
                conversation_id: None,
            },
            CassDocumentRef {
                agent: "codex",
                workspace: Some("/delta"),
                workspace_original: Some("/delta"),
                source_path: "/fixture/4.jsonl",
                msg_idx: 4,
                created_at: Some(500),
                title: Some("Currentish Omega"),
                content: "xalpha archived",
                source_id: "local-c",
                origin_kind: "local",
                origin_host: None,
                conversation_id: None,
            },
            CassDocumentRef {
                agent: "gemini",
                workspace: Some("/delta"),
                workspace_original: Some("/delta"),
                source_path: "/fixture/5.jsonl",
                msg_idx: 5,
                created_at: Some(600),
                title: Some("Middle Marker"),
                content: "alphabeta retained",
                source_id: "local-d",
                origin_kind: "local",
                origin_host: None,
                conversation_id: None,
            },
        ];
        index
            .add_cass_document_refs(&documents)
            .expect("index CASS result documents");
        index.commit().expect("commit CASS result documents");
        (directory, index)
    }

    /// The CASS oracle observation producer must expose exactly the evidence the
    /// gauntlet comparator consumes, under the CASS profile contract
    /// (`document_identity=source_id#msg_idx`, `snippets=disabled`).
    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn cass_oracle_observation_carries_contract_identity_and_full_tie_evidence() {
        let (_directory, index) = cass_result_fixture();
        let filters = CassQueryFilters::default();

        // CASS bare terms are PREFIX-matched (the schema carries dedicated
        // `*_prefix` fields), so `alpha` retrieves `alpha active`,
        // `alpha deprecated` AND `alphabeta retained` — but not `xalpha
        // archived`, which only contains the term as an infix. That asymmetry
        // is the load-bearing part of this expectation: a subject adapter that
        // treats CASS terms as exact would pass a naive 2-hit assertion and
        // still be wrong.
        let observation = index
            .cass_oracle_observe_query("alpha", &filters, 10, 8)
            .expect("observe CASS term query");
        assert_eq!(
            observation
                .hits
                .iter()
                .map(|hit| hit.doc_id.as_str())
                .collect::<std::collections::BTreeSet<_>>(),
            ["local-a#0", "local-b#1", "local-d#5"]
                .into_iter()
                .collect(),
            "identity must be the contract's source_id#msg_idx rendering, \
             over the prefix-expanded match set",
        );
        assert_eq!(observation.total_count, 3);
        assert_eq!(observation.doc_count, 6, "live document count");
        assert!(
            observation.hits.iter().all(|hit| hit.snippet.is_none()),
            "the CASS activation profile is snippet-free",
        );
        assert!(
            observation
                .hits
                .iter()
                .enumerate()
                .all(|(index, hit)| hit.rank == index),
            "ranks must be the native fetched order",
        );

        let profiled_filters = CassQueryFilters {
            agents: vec!["claude".to_owned()],
            source_filter: CassSourceFilter::Local,
            ..CassQueryFilters::default()
        };
        let profile = index
            .cass_oracle_observe_query_profile(
                "c++ \"alpha active\" OR -deprecated",
                &profiled_filters,
                10,
                8,
            )
            .expect("bounded profile observation");
        assert_eq!(
            profile.sanitized_query,
            "c   \"alpha active\" OR -deprecated"
        );
        assert_eq!(
            profile.tokens,
            vec![
                CassQueryToken::Term("c++".to_owned()),
                CassQueryToken::Phrase("alpha active".to_owned()),
                CassQueryToken::Or,
                CassQueryToken::Not,
                CassQueryToken::Term("deprecated".to_owned()),
            ]
        );
        assert!(profile.has_boolean_operators);
        assert_eq!(profile.filters, profiled_filters);
        let profile_retrieval = match profile.outcome {
            CassOracleProfileOutcome::Success(retrieval) => retrieval,
            CassOracleProfileOutcome::Error(error) => {
                panic!("fallible CASS profile unexpectedly failed: {error:?}")
            }
        };
        assert_eq!(
            profile_retrieval,
            index
                .cass_oracle_observe_query(
                    "c++ \"alpha active\" OR -deprecated",
                    &profiled_filters,
                    10,
                    8,
                )
                .expect("legacy CASS observation for parity")
        );

        // An empty CASS query is a filter-only browse, NOT the default
        // profile's empty short-circuit: it must still match every document.
        let browse = index
            .cass_oracle_observe_query("", &filters, 10, 8)
            .expect("observe CASS filter-only browse");
        assert_eq!(
            browse.hits.len(),
            6,
            "empty CASS query must browse, not short-circuit: {:?}",
            browse
                .hits
                .iter()
                .map(|hit| &hit.doc_id)
                .collect::<Vec<_>>(),
        );
        assert_eq!(browse.total_count, 6);

        // Structured filters narrow the browse through the same producer.
        let filtered = index
            .cass_oracle_observe_query(
                "",
                &CassQueryFilters {
                    source_filter: CassSourceFilter::SourceId("local-a".to_owned()),
                    ..CassQueryFilters::default()
                },
                10,
                8,
            )
            .expect("observe CASS structured filter");
        assert_eq!(
            filtered
                .hits
                .iter()
                .map(|hit| hit.doc_id.as_str())
                .collect::<Vec<_>>(),
            vec!["local-a#0"],
        );

        // Truncating to k must not discard the cutoff tie evidence: the group is
        // computed over the expanded fetch, then `hits` is cut to k.
        let truncated = index
            .cass_oracle_observe_query("alpha", &filters, 1, 8)
            .expect("observe CASS truncated query");
        assert_eq!(truncated.hits.len(), 1, "hits truncate to the requested k");
        assert_eq!(
            truncated.total_count, 3,
            "total_count stays independent of k",
        );
        assert!(
            truncated.cutoff_tie_complete,
            "the whole match set was fetched, so the tie group is complete",
        );
        assert!(
            truncated
                .cutoff_tie_group
                .iter()
                .all(|hit| hit.score_bits == truncated.hits[0].score_bits),
            "the tie group holds exactly the cutoff-score rows",
        );
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn cass_oracle_profile_rejects_hostile_bounds_before_clone_or_search() {
        let (_directory, index) = cass_result_fixture();
        let oversized_raw = "x".repeat(MAX_CASS_ORACLE_RAW_QUERY_BYTES + 1);
        assert!(matches!(
            index.cass_oracle_observe_query_profile(
                &oversized_raw,
                &CassQueryFilters::default(),
                10,
                8,
            ),
            Err(SearchError::InvalidConfig { field, .. })
                if field == "cass_oracle.raw_query_bytes"
        ));

        let too_many_tokens = "x ".repeat(MAX_CASS_ORACLE_TOKENS + 1);
        assert!(matches!(
            index.cass_oracle_observe_query_profile(
                &too_many_tokens,
                &CassQueryFilters::default(),
                10,
                8,
            ),
            Err(SearchError::InvalidConfig { field, .. })
                if field == "cass_oracle.token_count"
        ));

        let aggregate_filters = CassQueryFilters {
            agents: vec![
                "x".repeat(MAX_CASS_ORACLE_FILTER_VALUE_BYTES);
                MAX_CASS_ORACLE_FILTER_BYTES / MAX_CASS_ORACLE_FILTER_VALUE_BYTES + 1
            ],
            ..CassQueryFilters::default()
        };
        assert!(matches!(
            index.cass_oracle_observe_query_profile("x", &aggregate_filters, 10, 8),
            Err(SearchError::InvalidConfig { field, .. })
                if field == "cass_oracle.aggregate_filter_bytes"
        ));

        assert!(matches!(
            index.cass_oracle_observe_query_profile(
                "x",
                &CassQueryFilters::default(),
                MAX_CASS_ORACLE_FETCH_HITS,
                1,
            ),
            Err(SearchError::InvalidConfig { field, .. })
                if field == "cass_oracle.expanded_fetch_hits"
        ));
    }

    #[test]
    fn cass_sanitize_query_preserves_wildcards_quotes_and_hyphens() {
        let out = cass_sanitize_query("c++ \"hello-world\" *config*");
        assert!(out.contains('"'));
        assert!(out.contains('*'));
        // Hyphens are now preserved so hyphenated identifiers stay intact.
        assert!(out.contains("hello-world"));
    }

    #[test]
    fn cass_boolean_operator_detection_matches_parsed_tokens() {
        for (query, expected) in [
            ("", false),
            ("auth token cache", false),
            ("auth AND token", true),
            ("auth && token", true),
            ("auth OR token", true),
            ("auth || token", true),
            ("NOT legacy", true),
            ("-legacy", true),
            ("\"exact phrase\"", true),
            ("search 搜索", false),
        ] {
            let tokens = cass_parse_boolean_query(query);
            assert_eq!(
                cass_tokens_have_boolean_operators(&tokens),
                expected,
                "token-slice detection mismatch for {query:?}"
            );
            assert_eq!(
                cass_has_boolean_operators(query),
                expected,
                "public detection mismatch for {query:?}"
            );
        }
    }

    #[test]
    fn cass_query_builder_single_parse_matches_operator_reparse() {
        let fields = fields();
        let filter_sets = [
            CassQueryFilters::default(),
            CassQueryFilters {
                agents: vec!["claude".to_string(), "codex".to_string()],
                workspaces: vec!["/data/projects/frankensearch".to_string()],
                created_from: Some(1_700_000_000_000),
                created_to: Some(1_800_000_000_000),
                source_filter: CassSourceFilter::Remote,
            },
        ];
        let queries = [
            "",
            "auth token cache",
            "auth AND token",
            "auth && token OR cache",
            "\"exact phrase\" OR cache",
            "NOT legacy",
            "-legacy identifier",
            "AND",
            "auth OR",
            "\"unterminated phrase",
            "search 搜索 NOT stale",
        ];

        for filters in &filter_sets {
            for query in queries {
                let legacy = cass_build_tantivy_query(query, filters, &fields);
                let single_parse = cass_build_tantivy_query_single_parse(query, filters, &fields);
                assert_eq!(
                    format!("{legacy:?}"),
                    format!("{single_parse:?}"),
                    "query tree changed for {query:?} with filters {filters:?}"
                );
            }
        }
    }

    #[test]
    fn cass_writer_config_scales_with_parallelism() {
        // Single-core: 1 thread, heap clamped up to the 256 MB floor.
        assert_eq!(
            cass_writer_config_for_parallelism(1),
            CassWriterConfig {
                num_threads: 1,
                heap_size_bytes: 256 * 1024 * 1024,
            }
        );
        // Dual-core: 2 threads × 128 MB = 256 MB hits the floor exactly.
        assert_eq!(
            cass_writer_config_for_parallelism(2),
            CassWriterConfig {
                num_threads: 2,
                heap_size_bytes: 256 * 1024 * 1024,
            }
        );
        // Quad-core: 4 threads × 128 MB = 512 MB (above the floor).
        assert_eq!(
            cass_writer_config_for_parallelism(4),
            CassWriterConfig {
                num_threads: 4,
                heap_size_bytes: 512 * 1024 * 1024,
            }
        );
        // Eight cores stay below the `CASS_MAX_WRITER_THREADS` cap.
        assert_eq!(
            cass_writer_config_for_parallelism(8),
            CassWriterConfig {
                num_threads: 8,
                heap_size_bytes: 1024 * 1024 * 1024,
            }
        );
        // At the new cap: 32 threads × 128 MB = 4 GB.
        assert_eq!(
            cass_writer_config_for_parallelism(32),
            CassWriterConfig {
                num_threads: 32,
                heap_size_bytes: 4 * 1024 * 1024 * 1024,
            }
        );
        // Above the cap (e.g. 64-core workstation): still clamped to 32 threads.
        assert_eq!(
            cass_writer_config_for_parallelism(64),
            CassWriterConfig {
                num_threads: 32,
                heap_size_bytes: 4 * 1024 * 1024 * 1024,
            }
        );
    }

    #[test]
    fn cass_parallel_add_plan_enables_parallel_submission_for_large_batches() {
        assert_eq!(
            cass_parallel_add_plan(4_096, 8, 512, 2_048, 64),
            CassAddPlan {
                parallel_batches: true,
                batch_docs: 512,
                batch_count: 8,
            }
        );
        assert_eq!(
            cass_parallel_add_plan(200_000, 32, 512, 2_048, 64),
            CassAddPlan {
                parallel_batches: true,
                batch_docs: 3_125,
                batch_count: 64,
            }
        );
        assert_eq!(
            cass_parallel_add_plan(1_024, 8, 512, 2_048, 64),
            CassAddPlan {
                parallel_batches: false,
                batch_docs: 512,
                batch_count: 2,
            }
        );
    }

    #[test]
    fn next_char_from_ascii_matches_decode() {
        // The ASCII fast-path must produce the byte-for-byte same (char, next_offset) sequence as
        // the UTF-8 decode on every input — ASCII, multi-byte, and mixed — since it feeds the
        // tokenizer whose token boundaries must not shift.
        for text in [
            "",
            "hello world",
            "bd-q3fy ID_42",
            "éclair café",
            "日本語 mixed 한국어 text",
            "a\u{300}b\u{1F600}c", // combining mark + emoji interleaved with ascii
            "\u{4E00}\u{4E01}xyz",
        ] {
            let mut off = 0usize;
            loop {
                let fast = next_char_from(text, off);
                let slow = next_char_from_slow(text, off);
                assert_eq!(fast, slow, "mismatch in {text:?} at offset {off}");
                match fast {
                    Some((_, next)) => off = next,
                    None => break,
                }
            }
        }
        // And the full-walk harnesses must agree.
        for text in ["hello world code_id-42", "日本語 mixed", ""] {
            assert_eq!(
                cass_char_walk_fast(text),
                cass_char_walk_slow(text),
                "{text:?}"
            );
        }
    }

    #[test]
    fn cass_generate_edge_ngrams_emits_expected_prefixes() {
        assert_eq!(
            cass_generate_edge_ngrams("hello world"),
            "he hel hell hello wo wor worl world"
        );
        assert_eq!(
            cass_generate_edge_ngrams("éclair"),
            "éc écl écla éclai éclair"
        );
        assert_eq!(cass_generate_edge_ngrams("x"), "");
    }

    #[test]
    fn cass_generate_edge_ngrams_caps_prefixes_at_twenty_chars() {
        assert_eq!(
            cass_generate_edge_ngrams("abcdefghijklmnopqrstuvwxy"),
            "ab abc abcd abcde abcdef abcdefg abcdefgh abcdefghi abcdefghij abcdefghijk abcdefghijkl abcdefghijklm abcdefghijklmn abcdefghijklmno abcdefghijklmnop abcdefghijklmnopq abcdefghijklmnopqr abcdefghijklmnopqrs abcdefghijklmnopqrst"
        );
    }

    #[test]
    fn cass_build_preview_preserves_existing_behavior() {
        assert_eq!(cass_build_preview("", 0), "");
        assert_eq!(cass_build_preview("hello", 0), "…");
        assert_eq!(cass_build_preview("hello", 10), "hello");
        assert_eq!(cass_build_preview("hello world", 5), "hello…");
        assert_eq!(cass_build_preview("éclair", 3), "écl…");
    }

    /// PARITY GATE: the ASCII-fast `cass_generate_edge_ngrams` must equal the `char_indices` original
    /// for ASCII words (fast path), non-ASCII words (fallback), and mixed text — including the
    /// 20-char cap, short words (<3 chars → nothing), and words straddling the ASCII/Unicode split.
    #[test]
    fn cass_generate_edge_ngrams_matches_slow() {
        for text in [
            "",
            "x",
            "hi",
            "hello world",
            "bd-q3fy ID_42 snake_case",
            "abcdefghijklmnopqrstuvwxyz longwordovercap",
            "éclair café naïve",
            "日本語 hello 한국어 world",
            "aB3 café x9 déjà vu 12345678901234567890123",
        ] {
            assert_eq!(
                cass_generate_edge_ngrams(text),
                cass_generate_edge_ngrams_slow(text),
                "mismatch for {text:?}"
            );
        }
    }

    /// PARITY GATE: the O(1) `cass_prefix_source` must return the byte-identical prefix as the
    /// forward-scan original for every `content/max_bytes` combination — including cuts that land
    /// mid-multibyte-char (must round down to the boundary), at a boundary, at 0, and past the end.
    #[test]
    fn cass_prefix_source_matches_slow() {
        let ascii = "abcdefghij".repeat(50); // 500 bytes
        let uni = "aéb日cé".repeat(50); // multibyte, varied widths
        for content in ["", "x", "abc", ascii.as_str(), uni.as_str()] {
            for max_bytes in [0usize, 1, 2, 3, 4, 5, 7, 100, 499, 500, 501, 10_000] {
                let fast = cass_prefix_source(content, max_bytes);
                let slow = cass_prefix_source_slow(content, max_bytes);
                assert_eq!(
                    fast,
                    slow,
                    "content.len()={} max_bytes={max_bytes}",
                    content.len()
                );
            }
        }
    }

    /// PARITY GATE: the bulk-copy `cass_build_preview` must equal the char-by-char original for
    /// every content length / `max_chars` / ASCII-vs-Unicode combination (boundary at, before, and
    /// after the cut; empty; all-multibyte).
    #[test]
    fn cass_build_preview_matches_slow() {
        let ascii = "the quick brown fox jumps over the lazy dog ".repeat(20);
        let uni = "café 日本語 éclair 한국어 naïve ".repeat(20);
        for content in ["", "x", "hello world", ascii.as_str(), uni.as_str()] {
            for max_chars in [0usize, 1, 3, 4, 5, 10, 50, 400, 100_000] {
                assert_eq!(
                    cass_build_preview(content, max_chars),
                    cass_build_preview_slow(content, max_chars),
                    "content.len()={} max_chars={max_chars}",
                    content.len()
                );
            }
        }
    }

    #[test]
    fn cass_prefix_fields_store_basic_without_freqs_or_positions() {
        let schema = cass_build_schema();

        for field_name in ["title_prefix", "content_prefix"] {
            let field = schema.get_field(field_name).unwrap();
            let field_entry = schema.get_field_entry(field);
            assert_eq!(
                field_entry.field_type().get_index_record_option(),
                Some(IndexRecordOption::Basic),
                "unexpected index record option for {field_name}"
            );
            let tantivy::schema::FieldType::Str(text_options) = field_entry.field_type() else {
                panic!("{field_name} should be a text field");
            };
            assert_eq!(
                text_options
                    .get_indexing_options()
                    .expect("prefix field indexing options")
                    .tokenizer(),
                "prefix_normalize",
                "prefix field should use the cheaper generated-prefix analyzer"
            );
        }
    }

    #[test]
    fn cass_prefix_analyzer_matches_full_analyzer_for_generated_prefix_terms() {
        for text in [
            "",
            "Hello, happy tax payer!",
            "bd-q3fy foo_bar baz-qux",
            "abc-123 -- def",
            "Hello搜索World",
            "foo搜索-barあいう123",
            "caf\u{00E9} 𠀀 token",
            "multi---dash and trailing- hyphen",
        ] {
            let prefix_terms = cass_generate_edge_ngrams(text);
            let mut full = TextAnalyzer::builder(CassTokenizer::default())
                .filter(HyphenDecompose)
                .filter(CjkBigramDecompose)
                .filter(CassNormalizeAndLimit)
                .build();
            let mut prefix = TextAnalyzer::builder(CassTokenizer::default())
                .filter(CjkBigramDecompose)
                .filter(CassNormalizeAndLimit)
                .build();

            let mut full_stream = full.token_stream(&prefix_terms);
            let mut full_tokens = Vec::new();
            while full_stream.advance() {
                let token = full_stream.token();
                full_tokens.push((
                    token.text.clone(),
                    token.offset_from,
                    token.offset_to,
                    token.position,
                    token.position_length,
                ));
            }

            let mut prefix_stream = prefix.token_stream(&prefix_terms);
            let mut prefix_tokens = Vec::new();
            while prefix_stream.advance() {
                let token = prefix_stream.token();
                prefix_tokens.push((
                    token.text.clone(),
                    token.offset_from,
                    token.offset_to,
                    token.position,
                    token.position_length,
                ));
            }

            assert_eq!(
                prefix_tokens, full_tokens,
                "prefix analyzer changed generated edge-ngram tokens for {text:?}"
            );
        }
    }

    #[test]
    fn cass_preview_field_is_stored_only() {
        let schema = cass_build_schema();
        let field = schema.get_field("preview").unwrap();
        let field_entry = schema.get_field_entry(field);

        assert!(field_entry.is_stored(), "preview should stay stored");
        assert_eq!(
            field_entry.field_type().get_index_record_option(),
            None,
            "preview should not be indexed"
        );
    }

    #[test]
    fn cass_content_field_is_indexed_not_stored() {
        let schema = cass_build_schema();
        let field = schema.get_field("content").unwrap();
        let field_entry = schema.get_field_entry(field);

        assert!(field_entry.is_indexed(), "content must stay indexed");
        assert!(
            !field_entry.is_stored(),
            "content should hydrate from canonical storage instead of Tantivy stored fields"
        );
        assert_eq!(
            field_entry.field_type().get_index_record_option(),
            Some(IndexRecordOption::WithFreqsAndPositions),
            "content should keep full positional indexing"
        );
    }

    #[test]
    fn cass_build_content_prefix_and_preview_matches_existing_helpers() {
        let samples = [
            "",
            "hello world",
            "éclair crème brûlée",
            "foo_bar baz-qux 12345",
            "你好 世界 from cass",
            &"alpha beta gamma ".repeat(64),
        ];

        for sample in samples {
            let (prefix, preview) = cass_build_content_prefix_and_preview(sample);
            assert_eq!(prefix, cass_generate_edge_ngrams(sample));
            assert_eq!(preview, cass_build_preview(sample, 400));
        }
    }

    #[test]
    fn cass_build_content_prefix_caps_long_message_bodies() {
        let in_prefix_window = "alpha ".repeat(900);
        let after_prefix_window = "omega ".repeat(900);
        let content = format!("{in_prefix_window}{after_prefix_window}");

        let (prefix, preview) = cass_build_content_prefix_and_preview(&content);

        assert_eq!(preview, cass_build_preview(&content, 400));
        assert!(prefix.contains("alpha"));
        assert!(
            !prefix.contains("omega"),
            "content_prefix should not edge-ngram the whole message body"
        );
        assert_eq!(
            prefix,
            cass_generate_edge_ngrams(cass_prefix_source(&content, 4 * 1024))
        );
    }

    #[test]
    fn cass_sanitize_query_splits_on_underscores() {
        // Underscores are NOT alphanumeric, so the sanitizer replaces them
        // with spaces — matching the tokenizer which uses [a-zA-Z0-9] (not \w).
        let out = cass_sanitize_query("hello_world");
        assert_eq!(out, "hello world");
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

    /// cass#52 grammar (GH #56): NOT > AND (explicit or implied) > OR,
    /// parentheses group, and malformed groups recover. Each mixed case
    /// separates the standard reading from the legacy OR-binds-tighter one.
    #[test]
    fn cass_boolean_grammar_uses_standard_precedence_and_parentheses() {
        use CassQueryToken::{LParen, Not, Or, RParen, Term};
        let term = |text: &str| Term(text.to_owned());
        // `(` inside a word and `)` with no open group are term characters.
        assert_eq!(
            cass_parse_boolean_query("foo(bar) x)"),
            [term("foo(bar)"), term("x)")]
        );
        assert_eq!(
            cass_parse_boolean_query("-(a b)"),
            [Not, LParen, term("a"), term("b"), RParen]
        );
        assert_eq!(
            cass_parse_boolean_query("(a OR)"),
            [LParen, term("a"), Or, RParen]
        );

        let (_directory, index) = cass_result_fixture();
        let reader = index.reader().expect("open CASS Boolean reader");
        reader.reload().expect("reload CASS Boolean reader");
        let searcher = reader.searcher();
        let fields = index.fields();
        let matched = |raw_query: &str| {
            scored_cass_results(&searcher, &fields, raw_query, &CassQueryFilters::default())
                .into_keys()
                .collect::<Vec<_>>()
        };
        for (raw_query, expected) in [
            // deprecated ∪ (gamma ∩ active); legacy (deprecated ∪ gamma) ∩ active = {3}
            ("deprecated OR gamma AND active", vec![1, 3]),
            // (active ∩ beta) ∪ deprecated; legacy active ∩ (beta ∪ deprecated) = {2}
            ("active AND beta OR deprecated", vec![1, 2]),
            ("active beta OR deprecated", vec![1, 2]),
            ("active && beta || deprecated", vec![1, 2]),
            ("(deprecated OR gamma) AND active", vec![3]),
            ("active AND (beta OR deprecated)", vec![2]),
            ("-(beta OR gamma)", vec![0, 1, 4, 5]),
            ("alpha -(active OR deprecated)", vec![5]),
            ("deprecated OR -(alpha OR active)", vec![1, 4]),
            ("((gamma))", vec![3]),
            // Recovery: an unclosed group closes at the end; an empty group
            // and a dangling operator are skipped.
            ("(deprecated OR gamma", vec![1, 3]),
            ("gamma ()", vec![3]),
            ("gamma AND", vec![3]),
            ("OR gamma", vec![3]),
        ] {
            assert_eq!(matched(raw_query), expected, "{raw_query:?}");
        }

        // A group around a query both grammars read alike changes no score.
        let flat = scored_cass_results(
            &searcher,
            &fields,
            "alpha AND NOT deprecated",
            &CassQueryFilters::default(),
        );
        for grouped in ["(alpha AND NOT deprecated)", "((alpha) AND -(deprecated))"] {
            let actual =
                scored_cass_results(&searcher, &fields, grouped, &CassQueryFilters::default());
            assert_eq!(
                actual
                    .iter()
                    .map(|(msg_idx, score)| (*msg_idx, score.to_bits()))
                    .collect::<Vec<_>>(),
                flat.iter()
                    .map(|(msg_idx, score)| (*msg_idx, score.to_bits()))
                    .collect::<Vec<_>>(),
                "{grouped:?}"
            );
        }
    }

    #[test]
    fn cass_standalone_negation_matches_complement_and_is_score_neutral() {
        let (_directory, index) = cass_result_fixture();
        let reader = index.reader().expect("open CASS Boolean reader");
        reader.reload().expect("reload CASS Boolean reader");
        let searcher = reader.searcher();
        let fields = index.fields();

        let all_scores = scored_cass_results(&searcher, &fields, "", &CassQueryFilters::default());
        // Negation is parity-based (cass#52 grammar): two NOTs cancel.
        assert_eq!(
            scored_cass_results(
                &searcher,
                &fields,
                "NOT NOT deprecated",
                &CassQueryFilters::default()
            ),
            scored_cass_results(
                &searcher,
                &fields,
                "deprecated",
                &CassQueryFilters::default()
            ),
        );
        for raw_query in ["NOT deprecated", "-deprecated", "NOT NOT NOT deprecated"] {
            let actual =
                scored_cass_results(&searcher, &fields, raw_query, &CassQueryFilters::default());
            assert_eq!(
                actual.keys().copied().collect::<Vec<_>>(),
                vec![0, 2, 3, 4, 5],
                "standalone complement result set for {raw_query:?}"
            );
            for (msg_idx, score) in actual {
                assert_eq!(
                    score.to_bits(),
                    all_scores[&msg_idx].to_bits(),
                    "MustNot changed the AllQuery score for {raw_query:?} document {msg_idx}"
                );
            }
        }

        let claude_filter = CassQueryFilters {
            agents: vec!["claude".to_owned()],
            ..CassQueryFilters::default()
        };
        let filtered_all_scores = scored_cass_results(&searcher, &fields, "", &claude_filter);
        for raw_query in ["NOT deprecated", "-deprecated"] {
            let actual = scored_cass_results(&searcher, &fields, raw_query, &claude_filter);
            assert_eq!(
                actual.keys().copied().collect::<Vec<_>>(),
                vec![0, 3],
                "filtered standalone complement result set for {raw_query:?}"
            );
            for (msg_idx, score) in actual {
                assert_eq!(
                    score.to_bits(),
                    filtered_all_scores[&msg_idx].to_bits(),
                    "MustNot changed the filtered score for {raw_query:?} document {msg_idx}"
                );
            }
        }

        let positive_scores =
            scored_cass_results(&searcher, &fields, "alpha", &CassQueryFilters::default());
        let mixed = scored_cass_results(
            &searcher,
            &fields,
            "alpha AND NOT deprecated",
            &CassQueryFilters::default(),
        );
        assert_eq!(mixed.keys().copied().collect::<Vec<_>>(), vec![0, 5]);
        for (msg_idx, score) in mixed {
            assert_eq!(
                score.to_bits(),
                positive_scores[&msg_idx].to_bits(),
                "positive AND NOT gained a score-producing AllQuery anchor"
            );
        }

        let all_negative = scored_cass_results(
            &searcher,
            &fields,
            "-deprecated -beta",
            &CassQueryFilters::default(),
        );
        assert_eq!(
            all_negative.keys().copied().collect::<Vec<_>>(),
            vec![0, 3, 4, 5]
        );
        for (msg_idx, score) in all_negative {
            assert_eq!(score.to_bits(), all_scores[&msg_idx].to_bits());
        }
    }

    #[test]
    fn cass_regex_globs_match_whole_terms_and_fail_closed() {
        assert_eq!(
            CassWildcardPattern::Suffix("legacy".to_owned())
                .to_regex()
                .as_deref(),
            Some(".*legacy")
        );
        assert_eq!(
            CassWildcardPattern::Complex("a*ha".to_owned())
                .to_regex()
                .as_deref(),
            Some("a.*ha")
        );
        assert_eq!(
            CassWildcardPattern::Complex("*urr*nt".to_owned())
                .to_regex()
                .as_deref(),
            Some(".*urr.*nt")
        );

        let (_directory, index) = cass_result_fixture();
        let reader = index.reader().expect("open CASS regex reader");
        reader.reload().expect("reload CASS regex reader");
        let searcher = reader.searcher();
        let fields = index.fields();
        let no_filters = CassQueryFilters::default();

        for (raw_query, expected) in [
            ("*legacy", vec![1]),
            ("*deprecated", vec![1]),
            ("*alpha", vec![0, 1, 4]),
            ("*alpha*", vec![0, 1, 4, 5]),
            ("a*ha", vec![0, 1]),
            ("*urr*nt", vec![0, 2, 3]),
            ("*pha*b*", vec![5]),
            ("*legacy *deprecated", vec![1]),
        ] {
            assert_eq!(
                cass_result_ids(&searcher, &fields, raw_query, &no_filters),
                expected,
                "regex-backed CASS result set for {raw_query:?}"
            );
        }

        let claude_filter = CassQueryFilters {
            agents: vec!["claude".to_owned()],
            ..CassQueryFilters::default()
        };
        assert_eq!(
            cass_result_ids(&searcher, &fields, "*active", &claude_filter),
            vec![0, 3]
        );

        fn force_regex_construction_failure(
            field: Field,
            _pattern: &str,
        ) -> SearchResult<RegexQuery> {
            cass_regex_query_cached(field, "(")
        }

        let raw_query = "NOT *legacy";
        let failed_build = cass_try_build_tantivy_query_from_tokens(
            raw_query,
            cass_parse_boolean_query(raw_query),
            true,
            &CassQueryFilters::default(),
            &fields,
            force_regex_construction_failure,
        );
        assert!(
            failed_build.is_err(),
            "the forced invalid regex must reach the root builder"
        );
        let fail_closed = cass_fail_closed_tantivy_query(raw_query, failed_build);
        let hits = searcher
            .search(&*fail_closed, &tantivy::collector::DocSetCollector)
            .expect("search explicit match-none fallback");
        assert!(
            hits.is_empty(),
            "a negated regex construction failure must not widen to AllQuery"
        );

        #[cfg(feature = "tantivy-oracle")]
        {
            let profile = index
                .cass_oracle_observe_query_profile_with_regex_factory(
                    raw_query,
                    &CassQueryFilters::default(),
                    10,
                    8,
                    force_regex_construction_failure,
                )
                .expect("bounded forced-failure profile");
            assert_eq!(
                profile.tokens,
                vec![
                    CassQueryToken::Not,
                    CassQueryToken::Term("*legacy".to_owned())
                ]
            );
            assert!(matches!(
                profile.outcome,
                CassOracleProfileOutcome::Error(_)
            ));
        }
    }

    #[test]
    fn is_cjk_detects_chinese_characters() {
        assert!(is_cjk('\u{4E00}')); // CJK Unified start
        assert!(is_cjk('\u{641C}')); // 搜
        assert!(is_cjk('\u{7D22}')); // 索
        assert!(!is_cjk('a'));
        assert!(!is_cjk('1'));
    }

    #[test]
    fn is_cjk_detects_japanese_hiragana_katakana() {
        assert!(is_cjk('\u{3042}')); // あ (hiragana)
        assert!(is_cjk('\u{30A2}')); // ア (katakana)
    }

    #[test]
    fn is_cjk_detects_korean_hangul() {
        assert!(is_cjk('\u{AC00}')); // 가 (hangul start)
        assert!(is_cjk('\u{D558}')); // 하
    }

    #[test]
    fn cass_cjk_collect_fast_matches_slow() {
        // Byte-identity proof for the one-pass fusion: fast == slow on every
        // reject path and every decompose path.
        let cases = [
            "",               // empty -> None
            "a",              // single ASCII -> None
            "hello world",    // ASCII with space -> None (space non-CJK)
            "搜",             // single CJK -> None (unigram, no bigrams)
            "搜索",           // 2 CJK -> Some
            "搜索引擎",       // 4 CJK -> Some
            "検索とうきょう", // mixed CJK scripts -> Some
            "한국어",         // hangul -> Some
            "搜a",            // CJK then ASCII -> None (not all-CJK)
            "a搜",            // ASCII lead -> None
            "搜 索",          // CJK, space, CJK -> None (space breaks all-CJK)
            "𠀀𠀁",           // Extension B (4-byte) CJK -> Some
        ];
        for c in cases {
            assert_eq!(
                cass_cjk_collect_fast(c),
                cass_cjk_collect_slow(c),
                "fast/slow parity for {c:?}"
            );
        }
    }

    #[test]
    fn cjk_bigram_decompose_produces_bigrams() {
        // Build a minimal tokenizer pipeline with just CJK bigrams.
        let regex_tok = RegexTokenizer::new(
            r"[\u4E00-\u9FFF\u3400-\u4DBF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF\u3100-\u312F\u3300-\u33FF\uF900-\uFAFF]+"
        ).unwrap();
        let mut analyzer = TextAnalyzer::builder(regex_tok)
            .filter(CjkBigramDecompose)
            .build();

        // Chinese: 搜索引擎 -> bigrams: 搜索, 索引, 引擎
        let mut stream = analyzer.token_stream("搜索引擎");
        let mut tokens = Vec::new();
        while stream.advance() {
            tokens.push(stream.token().text.clone());
        }
        assert_eq!(tokens, vec!["搜索", "索引", "引擎"]);
    }

    #[test]
    fn cjk_bigram_single_char_emits_unigram() {
        let regex_tok = RegexTokenizer::new(r"[\u4E00-\u9FFF]+").unwrap();
        let mut analyzer = TextAnalyzer::builder(regex_tok)
            .filter(CjkBigramDecompose)
            .build();

        let mut stream = analyzer.token_stream("搜");
        let mut tokens = Vec::new();
        while stream.advance() {
            tokens.push(stream.token().text.clone());
        }
        assert_eq!(tokens, vec!["搜"]);
    }

    #[test]
    fn cjk_bigram_passes_through_ascii() {
        // ASCII tokens should pass through unchanged.
        let regex_tok =
            RegexTokenizer::new(r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*|[\u4E00-\u9FFF]+").unwrap();
        let mut analyzer = TextAnalyzer::builder(regex_tok)
            .filter(CjkBigramDecompose)
            .build();

        let mut stream = analyzer.token_stream("hello");
        let mut tokens = Vec::new();
        while stream.advance() {
            tokens.push(stream.token().text.clone());
        }
        assert_eq!(tokens, vec!["hello"]);
    }

    #[test]
    fn cjk_mixed_text_tokenizes_both() {
        // Mixed CJK and ASCII text should produce tokens for both.
        let regex_tok = RegexTokenizer::new(
            r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*|[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF]+"
        ).unwrap();
        let mut analyzer = TextAnalyzer::builder(regex_tok)
            .filter(CjkBigramDecompose)
            .filter(CassNormalizeAndLimit)
            .build();

        // "Hello搜索World" -> hello, 搜索, world
        let mut stream = analyzer.token_stream("Hello搜索World");
        let mut tokens = Vec::new();
        while stream.advance() {
            tokens.push(stream.token().text.clone());
        }
        assert_eq!(tokens, vec!["hello", "搜索", "world"]);
    }

    #[test]
    fn cass_tokenizer_matches_legacy_regex_boundaries() {
        let regex_pattern = r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*|[\u4E00-\u9FFF\u3400-\u4DBF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF\u3100-\u312F\u3300-\u33FF\uF900-\uFAFF\U00020000-\U0002A6DF]+";
        let mut regex = RegexTokenizer::new(regex_pattern).unwrap();
        let mut custom = CassTokenizer::default();

        for text in [
            "",
            "Hello, happy tax payer!",
            "bd-q3fy foo_bar baz-qux",
            "abc-123 -- def",
            "Hello搜索World",
            "foo搜索-barあいう123",
            "caf\u{00E9} 𠀀 token",
            "multi---dash and trailing- hyphen",
        ] {
            let mut regex_stream = regex.token_stream(text);
            let mut regex_tokens = Vec::new();
            while regex_stream.advance() {
                let token = regex_stream.token();
                regex_tokens.push((
                    token.text.clone(),
                    token.offset_from,
                    token.offset_to,
                    token.position,
                ));
            }

            let mut custom_stream = custom.token_stream(text);
            let mut custom_tokens = Vec::new();
            while custom_stream.advance() {
                let token = custom_stream.token();
                custom_tokens.push((
                    token.text.clone(),
                    token.offset_from,
                    token.offset_to,
                    token.position,
                ));
            }

            assert_eq!(custom_tokens, regex_tokens, "token mismatch for {text:?}");
        }
    }

    #[test]
    fn cass_tokenizer_keeps_extension_b_cjk() {
        let mut tokenizer = CassTokenizer::default();
        let mut stream = tokenizer.token_stream("𠀀搜索 test");
        let mut tokens = Vec::new();
        while stream.advance() {
            let token = stream.token();
            tokens.push(token.text.clone());
        }

        assert_eq!(tokens, vec!["𠀀搜索".to_string(), "test".to_string()]);
    }

    #[test]
    fn cass_normalize_and_limit_matches_legacy_pipeline() {
        for text in [
            "",
            "Hello, happy tax payer!",
            "BD-Q3FY foo_bar BAZ-QUX",
            "abc-123 -- def",
            "Hello搜索World",
            "foo搜索-barあいう123",
            "caf\u{00E9} 𠀀 token",
            "multi---dash and trailing- hyphen",
            &format!("{} keep", "X".repeat(255)),
            &format!("{} keep", "X".repeat(257)),
            &format!("{} keep", "X".repeat(300)),
        ] {
            let mut legacy = TextAnalyzer::builder(CassTokenizer::default())
                .filter(HyphenDecompose)
                .filter(CjkBigramDecompose)
                .filter(tantivy::tokenizer::LowerCaser)
                .filter(tantivy::tokenizer::RemoveLongFilter::limit(256))
                .build();
            let mut optimized = TextAnalyzer::builder(CassTokenizer::default())
                .filter(HyphenDecompose)
                .filter(CjkBigramDecompose)
                .filter(CassNormalizeAndLimit)
                .build();

            let mut legacy_stream = legacy.token_stream(text);
            let mut legacy_tokens = Vec::new();
            while legacy_stream.advance() {
                let token = legacy_stream.token();
                legacy_tokens.push((
                    token.text.clone(),
                    token.offset_from,
                    token.offset_to,
                    token.position,
                    token.position_length,
                ));
            }

            let mut optimized_stream = optimized.token_stream(text);
            let mut optimized_tokens = Vec::new();
            while optimized_stream.advance() {
                let token = optimized_stream.token();
                optimized_tokens.push((
                    token.text.clone(),
                    token.offset_from,
                    token.offset_to,
                    token.position,
                    token.position_length,
                ));
            }

            assert_eq!(
                optimized_tokens, legacy_tokens,
                "normalized analyzer mismatch for {text:?}"
            );
        }
    }

    #[test]
    fn cass_normalize_and_limit_pins_inclusive_shipping_boundary() {
        let shipping_tokens = |text: &str| {
            let mut analyzer = TextAnalyzer::builder(CassTokenizer::default())
                .filter(HyphenDecompose)
                .filter(CjkBigramDecompose)
                .filter(CassNormalizeAndLimit)
                .build();
            let mut stream = analyzer.token_stream(text);
            let mut tokens = Vec::new();
            while stream.advance() {
                let token = stream.token();
                tokens.push((
                    token.text.clone(),
                    token.offset_from,
                    token.offset_to,
                    token.position,
                    token.position_length,
                ));
            }
            tokens
        };
        let tantivy_tokens = |text: &str| {
            let mut analyzer = TextAnalyzer::builder(CassTokenizer::default())
                .filter(HyphenDecompose)
                .filter(CjkBigramDecompose)
                .filter(tantivy::tokenizer::LowerCaser)
                .filter(tantivy::tokenizer::RemoveLongFilter::limit(256))
                .build();
            let mut stream = analyzer.token_stream(text);
            let mut tokens = Vec::new();
            while stream.advance() {
                let token = stream.token();
                tokens.push((
                    token.text.clone(),
                    token.offset_from,
                    token.offset_to,
                    token.position,
                    token.position_length,
                ));
            }
            tokens
        };

        for length in [255, 256, 257] {
            let input = format!("{} keep", "X".repeat(length));
            let kept = vec![
                ("x".repeat(length), 0, length, 0, 1),
                ("keep".to_owned(), length + 1, length + 5, 1, 1),
            ];
            let dropped = vec![("keep".to_owned(), length + 1, length + 5, 1, 1)];
            let shipping_actual = shipping_tokens(&input);
            let tantivy_actual = tantivy_tokens(&input);

            assert_eq!(
                shipping_actual.as_slice(),
                if length <= 256 {
                    kept.as_slice()
                } else {
                    dropped.as_slice()
                },
                "shipping CASS boundary at {length} bytes"
            );
            assert_eq!(
                tantivy_actual.as_slice(),
                if length < 256 {
                    kept.as_slice()
                } else {
                    dropped.as_slice()
                },
                "Tantivy RemoveLongFilter boundary at {length} bytes"
            );
        }
    }

    #[test]
    fn cjk_sanitize_query_preserves_chinese() {
        let out = cass_sanitize_query("搜索引擎 test");
        assert!(out.contains("搜索引擎"));
        assert!(out.contains("test"));
    }

    #[test]
    fn cjk_index_and_search_roundtrip() {
        // End-to-end: index a Chinese document, search for a substring.
        let dir = tempfile::TempDir::new().expect("temp dir");
        let mut idx = CassTantivyIndex::open_or_create(dir.path()).expect("create");

        let doc = CassDocument {
            agent: "claude".to_string(),
            workspace: None,
            workspace_original: None,
            source_path: "/tmp/test".to_string(),
            msg_idx: 0,
            created_at: Some(1_700_000_000_000),
            title: Some("搜索引擎测试".to_string()),
            content: "这是一个搜索引擎的测试".to_string(),
            conversation_id: None,
            source_id: "local".to_string(),
            origin_kind: "local".to_string(),
            origin_host: None,
        };
        idx.add_cass_documents(&[doc]).expect("add");
        idx.commit().expect("commit");

        let reader = idx.reader().expect("reader");
        reader.reload().expect("reload");
        let searcher = reader.searcher();

        let fields = idx.fields();
        let query = cass_build_tantivy_query("搜索", &CassQueryFilters::default(), &fields);

        let results = searcher
            .search(
                &query,
                &tantivy::collector::TopDocs::with_limit(10).order_by_score(),
            )
            .expect("search");
        assert!(
            !results.is_empty(),
            "CJK search for '搜索' should find the indexed Chinese document"
        );
    }

    #[test]
    fn japanese_index_and_search_roundtrip() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let mut idx = CassTantivyIndex::open_or_create(dir.path()).expect("create");

        let doc = CassDocument {
            agent: "claude".to_string(),
            workspace: None,
            workspace_original: None,
            source_path: "/tmp/test".to_string(),
            msg_idx: 0,
            created_at: Some(1_700_000_000_000),
            title: Some("テスト".to_string()),
            content: "これはテストです".to_string(),
            conversation_id: None,
            source_id: "local".to_string(),
            origin_kind: "local".to_string(),
            origin_host: None,
        };
        idx.add_cass_documents(&[doc]).expect("add");
        idx.commit().expect("commit");

        let reader = idx.reader().expect("reader");
        reader.reload().expect("reload");
        let searcher = reader.searcher();

        let fields = idx.fields();
        let query = cass_build_tantivy_query("テスト", &CassQueryFilters::default(), &fields);

        let results = searcher
            .search(
                &query,
                &tantivy::collector::TopDocs::with_limit(10).order_by_score(),
            )
            .expect("search");
        assert!(
            !results.is_empty(),
            "CJK search for 'テスト' should find the indexed Japanese document"
        );
    }

    /// `CassTokenizer` indexes `kérlek` as `k` + `rlek`; a query for the whole
    /// word must reach those terms, split like the tokenizer splits, before
    /// lowercasing (`İ`) and at hyphen runs. A word with no indexable fragment
    /// (`привет`) must still match nothing.
    #[test]
    fn accented_latin_index_and_search_roundtrip() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let mut idx = CassTantivyIndex::open_or_create(dir.path()).expect("create");
        let doc = |msg_idx: u64, title: &str, content: &str| CassDocument {
            agent: "claude".to_string(),
            workspace: None,
            workspace_original: None,
            source_path: format!("/tmp/test-{msg_idx}"),
            msg_idx,
            created_at: Some(1_700_000_000_000),
            title: Some(title.to_string()),
            content: content.to_string(),
            conversation_id: None,
            source_id: "local".to_string(),
            origin_kind: "local".to_string(),
            origin_host: None,
        };
        idx.add_cass_documents(&[
            doc(0, "Mentés", "Kérlek ellenőrizd a mentést İstanbulban"),
            doc(1, "Backup", "please check the backup"),
        ])
        .expect("add");
        idx.commit().expect("commit");
        let reader = idx.reader().expect("reader");
        reader.reload().expect("reload");
        let searcher = reader.searcher();
        let fields = idx.fields();

        for (raw, expected) in [
            ("kérlek", 1),
            ("Kérlek", 1),
            ("ellenőrizd", 1),
            ("mentést", 1),
            ("İstanbulban", 1),
            ("kérlek--mentést", 1),
            ("\"kérlek ellenőrizd\"", 1),
            ("\"ellenőrizd kérlek\"", 0),
            ("kér*", 1),
            ("kérlek AND backup", 0),
            ("kérlek OR backup", 2),
            ("please", 1),
            ("привет", 0),
        ] {
            let query = cass_build_tantivy_query(raw, &CassQueryFilters::default(), &fields);
            let hits = searcher
                .search(&query, &tantivy::collector::Count)
                .expect("search");
            assert_eq!(hits, expected, "query {raw:?}");
        }
    }

    #[test]
    fn accented_latin_queries_preserve_fields_operators_and_phrase_order() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let mut index = CassTantivyIndex::open_or_create_with_writer_parallelism(dir.path(), 1)
            .expect("create accented fixture");
        let documents = [
            ("Hungarian", "Kérlek nézd ellenőrizd a mentést İstanbulban"),
            ("k", "rlek"),
            ("Prefix traps", "king rlekin menthol seconds"),
            ("French", "café déjà vérifiez"),
            ("German", "Straße"),
            ("Polish", "zażółć gęślą jaźń"),
            ("Czech", "příliš žluťoučký"),
            ("Case maps", "İstanbul K"),
            ("Decomposed", "ke\u{301}rlek"),
            ("Separated", "k intervening rlek"),
            ("Reverse", "rlek k"),
            ("Terminal prefix", "k rlekin"),
            ("ASCII control", "backup"),
            ("Another prefix trap", "cafeteria"),
        ]
        .into_iter()
        .enumerate()
        .map(|(position, (title, content))| CassDocument {
            agent: if position == 0 { "claude" } else { "codex" }.to_owned(),
            workspace: Some(if position == 0 { "/café" } else { "/other" }.to_owned()),
            workspace_original: None,
            source_path: format!("/fixture/accented-{position}.jsonl"),
            msg_idx: u64::try_from(position).expect("fixture position fits"),
            created_at: Some(100 + i64::try_from(position).expect("fixture position fits")),
            title: Some(title.to_owned()),
            content: content.to_owned(),
            conversation_id: None,
            source_id: if position == 0 { "source-é" } else { "local" }.to_owned(),
            origin_kind: if position == 0 { "ssh" } else { "local" }.to_owned(),
            origin_host: None,
        })
        .collect::<Vec<_>>();
        index.add_cass_documents(&documents).expect("index fixture");
        index.commit().expect("commit fixture");
        let reader = index.reader().expect("fixture reader");
        reader.reload().expect("reload fixture");
        let searcher = reader.searcher();
        let fields = index.fields();

        for (query, expected) in [
            ("kérlek", vec![0]),
            ("KÉRLEK", vec![0]),
            ("kérl*", vec![0, 9, 10, 11]),
            ("kér*", vec![0]),
            ("kérlek--mentést", vec![0]),
            ("\"kérlek\"", vec![0]),
            ("\"kérlek nézd\"", vec![0]),
            ("\"nézd kérlek\"", vec![]),
            ("\"kérlek kérlek\"", vec![]),
            ("\"ké*rlek\"", vec![]),
            ("\"ké*rlek nézd\"", vec![]),
            ("\"*kérlek*\"", vec![0]),
            ("\"kérl*\"", vec![]),
            ("mentést", vec![0]),
            ("café", vec![3]),
            ("café*", vec![3]),
            ("\"café\"", vec![3]),
            ("\"café*\"", vec![3]),
            ("déjà", vec![3]),
            ("Straße", vec![4]),
            ("STRAẞE", vec![4]),
            ("STRASSE", vec![]),
            ("ZAŻÓŁĆ", vec![5]),
            ("příliš", vec![6]),
            ("İstanbul", vec![7]),
            ("\"İstanbul\"", vec![7]),
            ("İstanbul*", vec![0, 7]),
            ("istanbul", vec![]),
            ("K", vec![]),
            ("İ", vec![]),
            ("ke\u{301}rlek", vec![8]),
            ("\"ke\u{301}rlek\"", vec![8]),
            ("kérlek AND backup", vec![]),
            ("kérlek OR backup", vec![0, 12]),
            ("kérlek -mentést", vec![]),
            ("NOT NOT kérlek", vec![0]),
            ("(kérlek OR backup) AND -mentést", vec![12]),
            ("backup NOT kérlek", vec![12]),
            ("backup AND 🚀", vec![]),
            ("🚀 OR backup", vec![]),
            ("привет", vec![]),
            ("éß", vec![]),
            ("\"kérlek привет\"", vec![]),
            ("🚀", vec![]),
            ("🚀*", vec![]),
            ("!***", vec![]),
            ("...", vec![]),
            ("\"🚀\"", vec![]),
            ("\"🚀*\"", vec![]),
            ("\"*++*\"", vec![]),
            ("\"\"", vec![]),
            ("\" \u{2003}\"", vec![]),
            ("backup \u{2003}", vec![12]),
            ("*érlek", vec![]),
            ("*kérlek*", vec![]),
            ("kér*lek", vec![]),
            ("backup", vec![12]),
        ] {
            assert_eq!(
                cass_result_ids(&searcher, &fields, query, &CassQueryFilters::default()),
                expected,
                "query {query:?}"
            );
        }

        for filters in [
            CassQueryFilters {
                agents: vec!["claude".to_owned()],
                ..CassQueryFilters::default()
            },
            CassQueryFilters {
                workspaces: vec!["/café".to_owned()],
                ..CassQueryFilters::default()
            },
            CassQueryFilters {
                created_from: Some(100),
                created_to: Some(100),
                ..CassQueryFilters::default()
            },
            CassQueryFilters {
                source_filter: CassSourceFilter::SourceId("source-é".to_owned()),
                ..CassQueryFilters::default()
            },
        ] {
            assert_eq!(
                cass_result_ids(&searcher, &fields, "kérlek", &filters),
                vec![0]
            );
            assert!(cass_result_ids(&searcher, &fields, "🚀", &filters).is_empty());
        }

        let oversized = format!("{}ébackup", "x".repeat(257));
        let terminal_oversized = format!("backupé{}", "x".repeat(257));
        let too_many = "ké".repeat(CASS_MAX_QUERY_FRAGMENTS + 1);
        let too_long = format!("kérlek{}", " ".repeat(CASS_MAX_QUERY_LENGTH));
        for invalid in [oversized, terminal_oversized, too_many, too_long] {
            for query in [
                invalid.clone(),
                format!("backup OR {invalid}"),
                format!("NOT {invalid}"),
                format!("NOT \"{invalid}\""),
                format!("NOT \"привет {invalid}\""),
            ] {
                assert!(
                    cass_result_ids(&searcher, &fields, &query, &CassQueryFilters::default())
                        .is_empty(),
                    "normalization rejection must fail the whole query: {query:?}"
                );
            }
        }
        for phrase in [
            format!("{} café", "x".repeat(257)),
            format!("café {}", "x".repeat(257)),
        ] {
            for raw in [
                format!("\"{phrase}\""),
                format!("backup OR \"{phrase}\""),
                format!("NOT \"{phrase}\""),
            ] {
                assert!(
                    cass_result_ids(&searcher, &fields, &raw, &CassQueryFilters::default())
                        .is_empty(),
                    "a rejected mixed phrase must fail the whole query: {raw:?}"
                );
            }
        }
        for (term, expected) in [("kérlek", vec![0]), ("backup", vec![12])] {
            let allowed = format!(
                "{}{term}{} AND {term}",
                "(".repeat(CASS_MAX_QUERY_NESTING),
                ")".repeat(CASS_MAX_QUERY_NESTING)
            );
            assert_eq!(
                cass_result_ids(&searcher, &fields, &allowed, &CassQueryFilters::default()),
                expected,
                "the depth boundary remains searchable"
            );
            let excessive = format!(
                "{}{term}{}",
                "(".repeat(CASS_MAX_QUERY_NESTING + 1),
                ")".repeat(CASS_MAX_QUERY_NESTING + 1)
            );
            for raw in [
                excessive.clone(),
                format!("backup OR {excessive}"),
                format!("NOT {excessive}"),
            ] {
                assert!(
                    cass_result_ids(&searcher, &fields, &raw, &CassQueryFilters::default())
                        .is_empty(),
                    "excessive grouping must fail the whole query: {raw:?}"
                );
            }
        }
        for (raw, expected) in [
            (format!("kérlek{}", "(".repeat(200)), vec![0]),
            (format!("\"kérlek {} nézd\"", "(".repeat(200)), vec![0]),
        ] {
            assert_eq!(
                cass_result_ids(&searcher, &fields, &raw, &CassQueryFilters::default()),
                expected,
                "opaque parentheses are not grammar groups: {raw:?}"
            );
        }
        for all_query in ["", " \t\n", "\u{a0}"] {
            assert_eq!(
                cass_result_ids(&searcher, &fields, all_query, &CassQueryFilters::default()),
                (0..14).collect::<Vec<_>>(),
                "explicit all-documents query {all_query:?}"
            );
        }
    }

    #[test]
    fn accented_words_stay_grouped_in_legacy_cjk_phrase_fallback() {
        let directory = tempfile::tempdir().expect("mixed script directory");
        let mut index =
            CassTantivyIndex::open_or_create_with_writer_parallelism(directory.path(), 1)
                .expect("create mixed script index");
        let documents = [
            ("", "kérlek 日本語 café"),
            ("k", "rlektor 日本語 cafeteria"),
            ("k", "rlek 日本語"),
            ("", "k rlektor 日本語 cafeteria"),
            ("", "日本語 rlek k caf"),
        ]
        .into_iter()
        .enumerate()
        .map(|(position, (title, content))| CassDocument {
            agent: "claude".to_owned(),
            workspace: None,
            workspace_original: None,
            source_path: format!("/fixture/mixed-script-{position}.jsonl"),
            msg_idx: u64::try_from(position).expect("fixture ordinal"),
            created_at: None,
            title: Some(title.to_owned()),
            content: content.to_owned(),
            conversation_id: None,
            source_id: "local".to_owned(),
            origin_kind: "local".to_owned(),
            origin_host: None,
        })
        .collect::<Vec<_>>();
        index
            .add_cass_documents(&documents)
            .expect("index mixed scripts");
        index.commit().expect("commit mixed scripts");
        let reader = index.reader().expect("mixed script reader");
        reader.reload().expect("reload mixed scripts");
        let searcher = reader.searcher();
        let fields = index.fields();
        for (raw, expected) in [
            ("\"kérlek 日本語\"", vec![0, 4]),
            ("\"日本語 kérlek\"", vec![0, 4]),
            ("\"kérlek* 日本語\"", vec![0, 4]),
            ("kérlek* 日本語", vec![0, 3, 4]),
            ("\"café 日本語\"", vec![0, 4]),
            ("\"café* 日本語\"", vec![0, 4]),
            ("\"ké*rlek 日本語\"", vec![]),
            ("kérlek日本語", vec![0, 4]),
        ] {
            assert_eq!(
                cass_result_ids(&searcher, &fields, raw, &CassQueryFilters::default()),
                expected,
                "mixed-script query {raw:?}"
            );
        }
    }

    #[test]
    fn empty_quoted_operands_remain_required_in_boolean_queries() {
        let directory = tempfile::tempdir().expect("empty phrase directory");
        let mut index =
            CassTantivyIndex::open_or_create_with_writer_parallelism(directory.path(), 1)
                .expect("create empty phrase index");
        let documents = ["backup kérlek", "other kérlek", "other"]
            .into_iter()
            .enumerate()
            .map(|(position, content)| CassDocument {
                agent: "claude".to_owned(),
                workspace: None,
                workspace_original: None,
                source_path: format!("/fixture/empty-phrase-{position}.jsonl"),
                msg_idx: u64::try_from(position).expect("fixture ordinal"),
                created_at: None,
                title: None,
                content: content.to_owned(),
                conversation_id: None,
                source_id: "local".to_owned(),
                origin_kind: "local".to_owned(),
                origin_host: None,
            })
            .collect::<Vec<_>>();
        index
            .add_cass_documents(&documents)
            .expect("index operands");
        index.commit().expect("commit operands");
        let reader = index.reader().expect("empty phrase reader");
        reader.reload().expect("reload operands");
        let searcher = reader.searcher();
        let fields = index.fields();
        let filters = CassQueryFilters::default();

        for empty_operand in [
            "\"\"",
            "\" \u{2003}\"",
            "\"🚀\"",
            "🚀",
            "привет",
            "*érlek",
            "kér*lek",
            "*",
            "***",
            "\"*** \u{2003}\"",
        ] {
            for raw in [
                empty_operand.to_owned(),
                format!("backup AND {empty_operand}"),
                format!("kérlek AND {empty_operand}"),
                format!("backup {empty_operand}"),
                format!("{empty_operand} OR backup"),
                format!("backup OR {empty_operand}"),
                format!("kérlek OR {empty_operand}"),
                format!("NOT {empty_operand}"),
                format!("NOT NOT {empty_operand}"),
                format!("backup NOT {empty_operand}"),
                format!("NOT ({empty_operand} OR backup)"),
            ] {
                assert!(
                    cass_result_ids(&searcher, &fields, &raw, &filters).is_empty(),
                    "invalid operands fail the whole expression: {raw:?}"
                );
            }
        }
        for rejected in ["AND", "OR NOT", "()", "---", "!!!"] {
            assert!(cass_result_ids(&searcher, &fields, rejected, &filters).is_empty());
        }
        for omitted in ["", " \u{2003}"] {
            assert_eq!(
                cass_result_ids(&searcher, &fields, omitted, &filters),
                vec![0, 1, 2],
                "{omitted:?}"
            );
        }
    }

    #[test]
    fn accented_raw_operands_preserve_nfd_positions_and_separate_cjk_runs() {
        let directory = tempfile::tempdir().expect("raw operand directory");
        let mut index =
            CassTantivyIndex::open_or_create_with_writer_parallelism(directory.path(), 1)
                .expect("create raw operand index");
        let documents = [
            ("", "ke\u{301}rlek"),
            ("", "ke unrelated rlek"),
            ("ke", "rlek"),
            ("", "rlek ke"),
            // Legacy postings cannot distinguish this source spelling from
            // the adjacent fragments of the decomposed accented word.
            ("", "ke rlek"),
            ("", "東京é大阪"),
            ("東京", "大阪"),
            ("", "東京 only"),
            ("", "大阪 only"),
            ("", "東京大阪"),
            ("", "éOR éNOT"),
            ("", "foo-barébaz"),
            ("", "foo baz"),
            ("", "bar baz"),
            ("", "foo-bar baz"),
        ]
        .into_iter()
        .enumerate()
        .map(|(position, (title, content))| CassDocument {
            agent: "claude".to_owned(),
            workspace: None,
            workspace_original: None,
            source_path: format!("/fixture/raw-operand-{position}.jsonl"),
            msg_idx: u64::try_from(position).expect("fixture ordinal"),
            created_at: None,
            title: Some(title.to_owned()),
            content: content.to_owned(),
            conversation_id: None,
            source_id: "local".to_owned(),
            origin_kind: "local".to_owned(),
            origin_host: None,
        })
        .collect::<Vec<_>>();
        index.add_cass_documents(&documents).expect("index raw operands");
        index.commit().expect("commit raw operands");
        drop(index);
        let index = CassTantivyIndex::open_or_create_with_writer_parallelism(directory.path(), 1)
            .expect("reopen existing index without reindexing");
        let reader = index.reader().expect("raw operand reader");
        let searcher = reader.searcher();
        let fields = index.fields();
        let filters = CassQueryFilters::default();
        for (raw, expected) in [
            ("ke\u{301}rlek", vec![0, 4]),
            ("KE\u{301}RLEK", vec![0, 4]),
            ("\"ke\u{301}rlek\"", vec![0, 4]),
            ("東京é大阪", vec![5, 9]),
            ("\"東京é大阪\"", vec![5, 9]),
            ("東京", vec![5, 6, 7, 9]),
            ("éOR", vec![10]),
            ("éNOT", vec![10]),
            ("foo-barébaz", vec![11, 14]),
            ("\"foo-barébaz\"", vec![11, 14]),
        ] {
            assert_eq!(
                cass_result_ids(&searcher, &fields, raw, &filters),
                expected,
                "raw operand {raw:?}"
            );
            let query = cass_build_tantivy_query(raw, &filters, &fields);
            assert_eq!(
                searcher
                    .search(&*query, &tantivy::collector::Count)
                    .expect("count raw operands"),
                expected.len(),
                "count {raw:?}"
            );
        }
        let expected = (0..15).filter(|id| ![0, 4].contains(id)).collect::<Vec<_>>();
        assert_eq!(
            cass_result_ids(&searcher, &fields, "NOT ke\u{301}rlek", &filters),
            expected,
            "negated positional operands retain unrelated fragment occurrences"
        );
    }

    #[test]
    fn accented_latin_negated_phrases_preserve_exclusions_and_scores() {
        let directory = tempfile::tempdir().expect("negated phrase directory");
        let mut index =
            CassTantivyIndex::open_or_create_with_writer_parallelism(directory.path(), 1)
                .expect("create negated phrase index");
        let documents = [
            "backup kérlek filler nézd",
            "backup kérlek nézd",
            "backup nézd kérlek",
            "other kérlek nézd",
            "backup complete",
        ]
        .into_iter()
        .enumerate()
        .map(|(position, content)| CassDocument {
            agent: "claude".to_owned(),
            workspace: None,
            workspace_original: None,
            source_path: format!("/fixture/negative-phrase-{position}.jsonl"),
            msg_idx: u64::try_from(position).expect("fixture ordinal"),
            created_at: None,
            title: None,
            content: content.to_owned(),
            conversation_id: None,
            source_id: "local".to_owned(),
            origin_kind: "local".to_owned(),
            origin_host: None,
        })
        .collect::<Vec<_>>();
        index.add_cass_documents(&documents).expect("index phrases");
        index.commit().expect("commit phrases");
        let reader = index.reader().expect("phrase reader");
        reader.reload().expect("reload phrases");
        let searcher = reader.searcher();
        let fields = index.fields();
        let filters = CassQueryFilters::default();
        let positive = scored_cass_results(&searcher, &fields, "backup", &filters);

        // The excluded phrase starts after the first positive document, and
        // later becomes exhausted while positive candidates still remain.
        // A fully absent phrase whose individual terms exist is a distinct
        // case from a missing-term query that returns an EmptyScorer directly.
        for (raw, expected) in [
            ("backup NOT \"kérlek nézd\"", vec![0, 2, 4]),
            ("backup NOT \"kérlek kék\"", vec![0, 1, 2, 4]),
            ("backup -\"nézd kérlek\"", vec![0, 1, 4]),
            ("backup OR \"kérlek kék\"", vec![0, 1, 2, 4]),
            (
                "backup NOT (\"kérlek nézd\" OR \"nézd kérlek\")",
                vec![0, 4],
            ),
        ] {
            let actual = scored_cass_results(&searcher, &fields, raw, &filters);
            assert_eq!(
                actual.keys().copied().collect::<Vec<_>>(),
                expected,
                "{raw:?}"
            );
            for (id, score) in &actual {
                assert_eq!(score.to_bits(), positive[id].to_bits(), "score for {raw:?}");
            }
            let query = cass_build_tantivy_query(raw, &filters, &fields);
            assert_eq!(
                searcher
                    .search(&*query, &tantivy::collector::Count)
                    .expect("count negatives"),
                actual.len(),
                "count for {raw:?}"
            );
            for (score, address) in searcher
                .search(&*query, &tantivy::collector::TopDocs::with_limit(5))
                .expect("search explained exclusions")
            {
                let explanation = query
                    .explain(&searcher, address)
                    .expect("explain retained document");
                assert_eq!(explanation.value().to_bits(), score.to_bits());
            }
        }
        assert_eq!(
            cass_result_ids(&searcher, &fields, "\"kérlek nézd\"", &filters),
            vec![1, 3]
        );
        assert_eq!(
            cass_result_ids(&searcher, &fields, "backup", &filters),
            vec![0, 1, 2, 4]
        );

        let phrase = cass_build_tantivy_query("\"kérlek nézd\"", &filters, &fields);
        let absent_phrase = cass_build_tantivy_query("\"kérlek kék\"", &filters, &fields);
        let excluded = cass_build_tantivy_query("backup NOT \"kérlek nézd\"", &filters, &fields);
        // Nonmatches lie before, between and after the phrase's matches.
        // Native PhraseWeight::explain otherwise seeks backwards for the
        // first of these; the adapter must return an ordinary nonmatch error.
        for (_, address) in searcher
            .search(&*excluded, &tantivy::collector::TopDocs::with_limit(5))
            .expect("find phrase nonmatches")
        {
            assert!(phrase.explain(&searcher, address).is_err());
            assert!(absent_phrase.explain(&searcher, address).is_err());
        }
        for (score, address) in searcher
            .search(&*phrase, &tantivy::collector::TopDocs::with_limit(5))
            .expect("find phrase matches")
        {
            let explanation = phrase.explain(&searcher, address).expect("explain phrase");
            assert_eq!(explanation.value().to_bits(), score.to_bits());
        }
        let protected_phrase = CassPhraseQuery(PhraseQuery::new(
            ["k", "rlek", "n", "zd"]
                .map(|term| Term::from_field_text(fields.content, term))
                .to_vec(),
        ));
        assert!(
            protected_phrase
                .explain(
                    &searcher,
                    tantivy::DocAddress {
                        segment_ord: 0,
                        doc_id: TERMINATED,
                    },
                )
                .is_err()
        );
    }

    #[test]
    fn accented_latin_fragment_limits_and_unlowercased_token_boundaries() {
        for (raw, expected) in [
            ("KÉRLEK", vec!["k", "rlek"]),
            ("İstanbul", vec!["stanbul"]),
            ("kérlek--mentést", vec!["k", "rlek", "ment", "st"]),
            ("Straße", vec!["stra", "e"]),
            ("K", vec![]),
            ("привет", vec![]),
            ("ASCII", vec!["ascii"]),
            ("foo-bar", vec!["foo-bar"]),
            ("搜索", vec!["搜索"]),
            ("kérlek搜索", vec!["k", "rlek", "搜索"]),
            ("ke\u{301}rlek", vec!["ke", "rlek"]),
            ("東京é大阪", vec!["東京", "大阪"]),
        ] {
            assert_eq!(
                cass_analyzed_fragments(raw).expect("bounded fragments").terms,
                expected.into_iter().map(str::to_owned).collect::<Vec<_>>(),
                "raw operand {raw:?}"
            );
        }
        let mixed = cass_analyzed_fragments("日本語é東京").expect("CJK run positions");
        assert_eq!(mixed.terms, vec!["日本", "本語", "東京"]);
        assert_eq!(mixed.positions, vec![0, 0, 1]);
        assert!(mixed.ends_with_token);
        assert!(
            !cass_analyzed_fragments("café")
                .expect("terminal accent")
                .ends_with_token
        );
        let at_limit = format!("{}ébackup", "A".repeat(256));
        assert_eq!(
            cass_analyzed_fragments(&at_limit)
                .expect("256-byte token remains valid")
                .terms,
            vec!["a".repeat(256), "backup".to_owned()]
        );
        for over_limit in [
            format!("{}ébackup", "A".repeat(257)),
            format!("backupé{}", "A".repeat(257)),
            "ké".repeat(CASS_MAX_QUERY_FRAGMENTS + 1),
        ] {
            assert!(cass_analyzed_fragments(&over_limit).is_err());
        }
        let fragment_limit = "ké".repeat(CASS_MAX_QUERY_FRAGMENTS);
        assert_eq!(
            cass_analyzed_fragments(&fragment_limit)
                .expect("exact fragment limit")
                .terms
                .len(),
            CASS_MAX_QUERY_FRAGMENTS
        );
        assert!(cass_normalize_phrase_terms(&format!("{fragment_limit} café")).is_err());
        for mixed in [
            format!("{} café", "a".repeat(257)),
            format!("café {}", "a".repeat(257)),
        ] {
            assert!(cass_normalize_phrase_terms(&mixed).is_err());
        }
        assert!(cass_normalize_phrase_terms(&format!("{} café", "日本語".repeat(40))).is_ok());
        assert!(cass_normalize_phrase_terms("ké*rlek").is_err());
        assert!(cass_normalize_phrase_terms("foo*bar").is_err());
    }

    #[test]
    fn korean_index_and_search_roundtrip() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let mut idx = CassTantivyIndex::open_or_create(dir.path()).expect("create");

        let doc = CassDocument {
            agent: "claude".to_string(),
            workspace: None,
            workspace_original: None,
            source_path: "/tmp/test".to_string(),
            msg_idx: 0,
            created_at: Some(1_700_000_000_000),
            title: Some("검색엔진".to_string()),
            content: "한국어 검색엔진 테스트".to_string(),
            conversation_id: None,
            source_id: "local".to_string(),
            origin_kind: "local".to_string(),
            origin_host: None,
        };
        idx.add_cass_documents(&[doc]).expect("add");
        idx.commit().expect("commit");

        let reader = idx.reader().expect("reader");
        reader.reload().expect("reload");
        let searcher = reader.searcher();

        let fields = idx.fields();
        let query = cass_build_tantivy_query("검색", &CassQueryFilters::default(), &fields);

        let results = searcher
            .search(
                &query,
                &tantivy::collector::TopDocs::with_limit(10).order_by_score(),
            )
            .expect("search");
        assert!(
            !results.is_empty(),
            "CJK search for '검색' should find the indexed Korean document"
        );
    }

    #[test]
    fn force_merge_bounded_consolidates_segments() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let mut idx = CassTantivyIndex::open_or_create(dir.path()).expect("create");

        // Suppress auto-merge during setup so each commit leaves its own
        // segment; otherwise the default LogMergePolicy could collapse them and
        // we'd never exercise the bounded multi-pass path.
        idx.writer.set_merge_policy(Box::new(NoMergePolicy));

        let make_doc = |i: u64| CassDocument {
            agent: "claude".to_string(),
            workspace: None,
            workspace_original: None,
            source_path: format!("/tmp/test/{i}"),
            msg_idx: i,
            created_at: Some(1_700_000_000_000_i64.saturating_add(i.cast_signed())),
            title: Some(format!("doc {i}")),
            content: format!("bounded merge segment number {i}"),
            source_id: "local".to_string(),
            origin_kind: "local".to_string(),
            origin_host: None,
            conversation_id: None,
        };

        // Five separate commits => five segments (auto-merge suppressed).
        for i in 0..5u64 {
            idx.add_cass_documents(&[make_doc(i)]).expect("add");
            idx.commit().expect("commit");
        }
        let before = idx.index.searchable_segment_ids().expect("segments").len();
        assert!(before > 1, "expected multiple segments, got {before}");

        // Bounded merge in batches of 2 must consolidate to a single segment
        // and restore the default merge policy on the way out.
        let merges = idx.force_merge_bounded(2).expect("bounded merge");
        assert!(merges >= 1, "expected at least one bounded merge pass");

        let after = idx.index.searchable_segment_ids().expect("segments").len();
        assert!(
            after <= 1,
            "bounded merge should consolidate to <=1 segment, got {after}"
        );

        // All five documents survive the consolidation.
        let reader = idx.reader().expect("reader");
        reader.reload().expect("reload");
        let searcher = reader.searcher();
        let fields = idx.fields();
        let query = cass_build_tantivy_query("bounded", &CassQueryFilters::default(), &fields);
        let results = searcher
            .search(
                &query,
                &tantivy::collector::TopDocs::with_limit(10).order_by_score(),
            )
            .expect("search");
        assert_eq!(
            results.len(),
            5,
            "all documents should remain searchable after bounded merge"
        );
    }

    /// 100-segment regression: exercises the bounded-merge progression at a
    /// scale closer to the failure-mode described in #26 (many small segments,
    /// auto-merge suppressed). With `batch_size = 8` the inner loop must do
    /// at least `ceil(100 / 8) = 13` initial passes; in practice each pass
    /// collapses the leading 8 segments to 1, so the total iteration count is
    /// bounded by `O(N / (B - 1))`. We assert the operation completes and
    /// leaves at most one segment regardless of exact iteration count.
    #[test]
    fn force_merge_bounded_progressively_collapses_many_segments() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let mut idx = CassTantivyIndex::open_or_create(dir.path()).expect("create");
        idx.writer.set_merge_policy(Box::new(NoMergePolicy));

        // 100 single-doc commits => 100 segments.
        for i in 0..100u64 {
            idx.add_cass_documents(&[CassDocument {
                agent: "claude".to_string(),
                workspace: None,
                workspace_original: None,
                source_path: format!("/tmp/test100/{i}"),
                msg_idx: i,
                created_at: Some(1_700_000_000_000_i64.saturating_add(i.cast_signed())),
                title: Some(format!("doc {i}")),
                content: format!("scale test segment {i}"),
                source_id: "local".to_string(),
                origin_kind: "local".to_string(),
                origin_host: None,
                conversation_id: None,
            }])
            .expect("add");
            idx.commit().expect("commit");
        }
        let before = idx.index.searchable_segment_ids().expect("segments").len();
        assert!(
            before >= 50,
            "expected many segments before bounded merge, got {before}"
        );

        let merges = idx.force_merge_bounded(8).expect("bounded merge");
        // ceil(100/8) = 13 is the first-pass lower bound; subsequent passes
        // collapse the residual. We just assert progress was made and the
        // final segment count is the target.
        assert!(
            merges >= 13,
            "expected at least 13 bounded passes for 100 segments / batch 8, got {merges}"
        );

        let after = idx.index.searchable_segment_ids().expect("segments").len();
        assert!(
            after <= 1,
            "bounded merge should collapse 100 segments to <=1, got {after}"
        );
    }

    /// Regression: the previously-configured merge policy must be *preserved*
    /// across `force_merge_bounded`, not silently reset to
    /// `LogMergePolicy::default()`. This is the central correctness gain over
    /// the obvious "install `NoMergePolicy`, then install `LogMergePolicy` on
    /// exit" pattern: a caller that has tuned the policy (e.g. via
    /// [`CassTantivyIndex::configure_bulk_load_merge_policy`]) keeps that
    /// tuning after recovery instead of having it silently dropped.
    #[test]
    fn force_merge_bounded_restores_prior_merge_policy() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let mut idx = CassTantivyIndex::open_or_create(dir.path()).expect("create");

        // Add a couple of segments so the bounded loop actually runs.
        idx.writer.set_merge_policy(Box::new(NoMergePolicy));
        for i in 0..3u64 {
            idx.add_cass_documents(&[CassDocument {
                agent: "claude".to_string(),
                workspace: None,
                workspace_original: None,
                source_path: format!("/tmp/restore/{i}"),
                msg_idx: i,
                created_at: Some(1_700_000_000_000_i64.saturating_add(i.cast_signed())),
                title: Some(format!("restore {i}")),
                content: format!("merge policy restore probe {i}"),
                source_id: "local".to_string(),
                origin_kind: "local".to_string(),
                origin_host: None,
                conversation_id: None,
            }])
            .expect("add");
            idx.commit().expect("commit");
        }
        // Reinstall the bulk-load policy as the "prior" snapshot that
        // force_merge_bounded should preserve. The bulk-load policy is a
        // LogMergePolicy with `min_num_segments` set to
        // `CASS_BULK_LOAD_MIN_SEGMENTS_PER_MERGE` (= 256), which is the
        // distinctive fingerprint we look for after the bounded merge.
        idx.configure_bulk_load_merge_policy();
        let prior_dbg = format!("{:?}", idx.writer.get_merge_policy());
        assert!(
            prior_dbg.contains("min_num_segments: 256"),
            "sanity: prior policy should be the bulk-load LogMergePolicy, got {prior_dbg}"
        );

        let _ = idx.force_merge_bounded(2).expect("bounded merge");

        let after_dbg = format!("{:?}", idx.writer.get_merge_policy());
        // The restored policy is wrapped in `ArcMergePolicy(...)` because
        // `set_merge_policy` requires a `Box` while `get_merge_policy` only
        // exposes an `Arc`; the wrapper delegates `compute_merge_candidates`
        // to the inner policy, so steady-state auto-merge behavior is
        // preserved. The wrapped fingerprint must still carry the
        // distinctive bulk-load `min_num_segments: 256` — i.e. the caller's
        // tuning was NOT silently replaced with `LogMergePolicy::default()`.
        assert!(
            after_dbg.contains("min_num_segments: 256"),
            "force_merge_bounded dropped the bulk-load tuning: after = {after_dbg}"
        );
        assert!(
            !after_dbg.contains("NoMergePolicy"),
            "force_merge_bounded left NoMergePolicy in place: after = {after_dbg}"
        );
    }
}
