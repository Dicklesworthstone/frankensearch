//! Deterministic lexical indexing pipeline model for fsfs.
//!
//! This module provides a backend-agnostic lexical pipeline contract covering:
//! - text chunking and tokenization for mixed prose/code corpora
//! - initial and incremental index mutation planning
//! - explicit update/delete behavior on change and reclassification
//! - measurable throughput/latency target contracts

use std::collections::{BTreeMap, HashMap};

use asupersync::Cx;
use compact_str::CompactString;
use frankensearch_core::{IndexableDocument, LexicalSearch, SearchError, SearchResult};
use frankensearch_quill::{QuillIndex, indexable_document_content_hash};
use tracing::debug;

use crate::config::IngestionClass;

/// Default expected throughput for initial lexical indexing (docs/sec).
pub const TARGET_INITIAL_DOCS_PER_SECOND: u32 = 20_000;
/// Default expected throughput for incremental lexical updates (updates/sec).
pub const TARGET_INCREMENTAL_UPDATES_PER_SECOND: u32 = 5_000;
/// Default expected p95 latency budget for incremental updates.
pub const TARGET_INCREMENTAL_P95_LATENCY_MS: u32 = 25;

/// Performance contract for lexical indexing workloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LexicalPerformanceTargets {
    pub initial_docs_per_second: u32,
    pub incremental_updates_per_second: u32,
    pub incremental_p95_latency_ms: u32,
}

impl Default for LexicalPerformanceTargets {
    fn default() -> Self {
        Self {
            initial_docs_per_second: TARGET_INITIAL_DOCS_PER_SECOND,
            incremental_updates_per_second: TARGET_INCREMENTAL_UPDATES_PER_SECOND,
            incremental_p95_latency_ms: TARGET_INCREMENTAL_P95_LATENCY_MS,
        }
    }
}

impl LexicalPerformanceTargets {
    #[must_use]
    pub const fn meets_contract(
        self,
        observed_initial_docs_per_second: u32,
        observed_incremental_updates_per_second: u32,
        observed_incremental_p95_latency_ms: u32,
    ) -> bool {
        observed_initial_docs_per_second >= self.initial_docs_per_second
            && observed_incremental_updates_per_second >= self.incremental_updates_per_second
            && observed_incremental_p95_latency_ms <= self.incremental_p95_latency_ms
    }
}

/// Chunking policy for lexical indexing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LexicalChunkPolicy {
    /// Maximum UTF-8 character span per chunk.
    pub max_chars: usize,
    /// Character overlap between adjacent chunks.
    pub overlap_chars: usize,
}

impl Default for LexicalChunkPolicy {
    fn default() -> Self {
        Self {
            max_chars: 768,
            overlap_chars: 96,
        }
    }
}

/// A chunk emitted by the lexical chunker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LexicalChunk {
    pub ordinal: u32,
    pub byte_start: usize,
    pub byte_end: usize,
    pub text: String,
    pub token_count: usize,
}

/// Token span produced by deterministic lexical tokenization.
///
/// `text` is a [`CompactString`]: lexical tokens (code identifiers, prose words,
/// short paths) are almost always <=24 bytes, so they live inline with zero heap
/// allocation. The per-token lowercase heap alloc was the dominant emission cost
/// (see the `tokenize_ascii_ab` bench); SSO removes it for the common case while
/// `CompactString` remains a drop-in `&str` (`Deref`/`as_str`/`PartialEq<&str>`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LexicalToken {
    pub text: CompactString,
    pub line: u32,
    pub byte_start: usize,
    pub byte_end: usize,
}

/// Build a lowercased token text directly into a [`CompactString`], preserving the
/// exact ASCII-only lowercasing of `str::to_ascii_lowercase` (`make_ascii_lowercase`
/// touches only bytes `0x41..=0x5A`, so UTF-8 multi-byte sequences are untouched and
/// the result is byte-identical for any input). Short tokens never touch the heap.
#[inline]
fn lower_token_text(slice: &str) -> CompactString {
    let mut text = CompactString::new(slice);
    text.as_mut_str().make_ascii_lowercase();
    text
}

impl LexicalChunkPolicy {
    /// Chunk and tokenize a UTF-8 document using deterministic overlap rules.
    #[must_use]
    pub fn chunk_text(self, text: &str) -> Vec<LexicalChunk> {
        if text.is_empty() {
            return Vec::new();
        }

        let max_chars = self.max_chars.max(1);
        let overlap_chars = self.overlap_chars.min(max_chars.saturating_sub(1));

        let mut chunks = Vec::new();
        let mut start = 0usize;
        let mut ordinal = 0_u32;

        while start < text.len() {
            let raw_end = start.saturating_add(max_chars);
            let mut end = if raw_end >= text.len() {
                text.len()
            } else {
                floor_char_boundary(text, raw_end)
            };
            if end <= start {
                end = ceil_char_boundary(text, raw_end.min(text.len()));
            }
            if end <= start {
                break;
            }

            let chunk_text = text[start..end].to_owned();
            let token_count = count_lexical_tokens(&chunk_text);
            chunks.push(LexicalChunk {
                ordinal,
                byte_start: start,
                byte_end: end,
                text: chunk_text,
                token_count,
            });
            ordinal = ordinal.saturating_add(1);

            if end == text.len() {
                break;
            }
            let mut next_start = floor_char_boundary(text, end.saturating_sub(overlap_chars));
            if next_start <= start {
                next_start = end;
            }
            start = next_start;
        }

        chunks
    }
}

/// Count tokens in text without allocating strings.
///
/// ASCII text (the common case for code/docs) takes a byte-loop fast path that
/// skips `chars()` UTF-8 decoding. Bit-identical to the `chars()` path: for an
/// ASCII byte `b`, `is_token_byte(b) == is_token_char(b as char)`
/// (`char::is_alphanumeric` matches `u8::is_ascii_alphanumeric` on ASCII, and the
/// punctuation set is the same). Non-ASCII text falls back to the Unicode-aware
/// `chars()` path unchanged.
#[must_use]
pub fn count_lexical_tokens(text: &str) -> usize {
    if text.is_ascii() {
        // 256-byte class table (one load per byte) + branchless transition counting:
        // a token ends at every token→non-token transition (`prev & !cur`), removing
        // both the multi-op `is_token_byte` and the data-dependent `in_token` branch
        // that mispredicts on every token boundary. `TOKEN_BYTE[b] == is_token_byte(b)`
        // by construction, so this is bit-identical to the scalar byte path.
        let mut count = 0usize;
        let mut prev = 0u8;
        for &b in text.as_bytes() {
            let cur = TOKEN_BYTE[b as usize];
            count += (prev & !cur) as usize;
            prev = cur;
        }
        return count + prev as usize;
    }

    let mut count = 0;
    let mut in_token = false;
    for ch in text.chars() {
        if is_token_char(ch) {
            if !in_token {
                in_token = true;
            }
        } else if in_token {
            in_token = false;
            count += 1;
        }
    }
    if in_token {
        count += 1;
    }
    count
}

/// Split text into lexical tokens while preserving code/path-like identifiers.
#[must_use]
pub fn tokenize_lexical(text: &str) -> Vec<LexicalToken> {
    // ASCII byte fast path (the common case: code, English prose). For all-ASCII
    // text the byte index equals the char index and `TOKEN_BYTE` equals
    // `is_token_char` for every byte, so the emitted tokens (text/line/offsets)
    // are bit-identical — it just skips the UTF-8 decode + Unicode `is_alphanumeric`
    // per char (~1.1-1.17×, `tokenize_ascii_ab` bench). Mirrors the sibling
    // `count_lexical_tokens` ASCII fast path.
    if text.is_ascii() {
        return tokenize_lexical_ascii(text);
    }

    let mut tokens = Vec::new();
    let mut token_start: Option<usize> = None;
    let mut line = 1_u32;
    let mut token_line = 1_u32;

    for (idx, ch) in text.char_indices() {
        if is_token_char(ch) {
            if token_start.is_none() {
                token_start = Some(idx);
                token_line = line;
            }
        } else if let Some(start) = token_start.take() {
            tokens.push(LexicalToken {
                text: lower_token_text(&text[start..idx]),
                line: token_line,
                byte_start: start,
                byte_end: idx,
            });
        }

        if ch == '\n' {
            line = line.saturating_add(1);
        }
    }

    if let Some(start) = token_start {
        tokens.push(LexicalToken {
            text: lower_token_text(&text[start..]),
            line: token_line,
            byte_start: start,
            byte_end: text.len(),
        });
    }

    tokens
}

/// ASCII fast path for [`tokenize_lexical`]. Only called when `text.is_ascii()`,
/// where byte index == char index and `TOKEN_BYTE[b]` == `is_token_char(b as char)`
/// for every byte, so the output is bit-identical to the char-based path — it just
/// avoids per-char UTF-8 decoding and the Unicode `is_alphanumeric` predicate.
fn tokenize_lexical_ascii(text: &str) -> Vec<LexicalToken> {
    let bytes = text.as_bytes();
    let mut tokens = Vec::new();
    let mut token_start: Option<usize> = None;
    let mut line = 1_u32;
    let mut token_line = 1_u32;

    for (idx, &b) in bytes.iter().enumerate() {
        if TOKEN_BYTE[b as usize] == 1 {
            if token_start.is_none() {
                token_start = Some(idx);
                token_line = line;
            }
        } else if let Some(start) = token_start.take() {
            tokens.push(LexicalToken {
                text: lower_token_text(&text[start..idx]),
                line: token_line,
                byte_start: start,
                byte_end: idx,
            });
        }

        if b == b'\n' {
            line = line.saturating_add(1);
        }
    }

    if let Some(start) = token_start {
        tokens.push(LexicalToken {
            text: lower_token_text(&text[start..]),
            line: token_line,
            byte_start: start,
            byte_end: text.len(),
        });
    }

    tokens
}

#[must_use]
fn is_token_char(ch: char) -> bool {
    ch.is_alphanumeric() || matches!(ch, '_' | '-' | '.' | '/' | ':')
}

/// ASCII-only `is_token_char` for the `count_lexical_tokens` byte fast path.
/// Equals `is_token_char(b as char)` for every ASCII byte `b`.
#[inline]
const fn is_token_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.' | b'/' | b':')
}

/// Byte→token-class lookup table (`1` = token byte) for the `count_lexical_tokens`
/// ASCII fast path. Built from [`is_token_byte`] at compile time, so it is exactly
/// equal to the scalar predicate for every byte (bit-identical token counts).
#[allow(
    clippy::cast_possible_truncation,
    reason = "the loop bounds prove every index fits in u8"
)]
const TOKEN_BYTE: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0usize;
    while i < 256 {
        t[i] = is_token_byte(i as u8) as u8;
        i += 1;
    }
    t
};

fn floor_char_boundary(text: &str, index: usize) -> usize {
    let mut idx = index.min(text.len());
    while idx > 0 && !text.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

fn ceil_char_boundary(text: &str, index: usize) -> usize {
    let mut idx = index.min(text.len());
    while idx < text.len() && !text.is_char_boundary(idx) {
        idx += 1;
    }
    idx
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LexicalMutationKind {
    Upsert,
    Delete,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LexicalMutation {
    pub doc_id: String,
    pub revision: u64,
    pub ingestion_class: IngestionClass,
    pub change: LexicalMutationKind,
    pub text: Option<String>,
    pub title: Option<String>,
    pub metadata: HashMap<String, String>,
    pub reason: String,
}

impl LexicalMutation {
    #[must_use]
    pub fn upsert(
        doc_id: impl Into<String>,
        revision: u64,
        ingestion_class: IngestionClass,
        text: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            doc_id: doc_id.into(),
            revision,
            ingestion_class,
            change: LexicalMutationKind::Upsert,
            text: Some(text.into()),
            title: None,
            metadata: std::collections::HashMap::new(),
            reason: reason.into(),
        }
    }

    #[must_use]
    pub fn delete(
        doc_id: impl Into<String>,
        revision: u64,
        ingestion_class: IngestionClass,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            doc_id: doc_id.into(),
            revision,
            ingestion_class,
            change: LexicalMutationKind::Delete,
            text: None,
            title: None,
            metadata: HashMap::new(),
            reason: reason.into(),
        }
    }

    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LexicalAction {
    Upsert {
        doc_id: String,
        revision: u64,
        title: Option<String>,
        metadata: HashMap<String, String>,
        chunks: Vec<LexicalChunk>,
    },
    Delete {
        doc_id: String,
        revision: u64,
        reason: String,
    },
    Skip {
        doc_id: String,
        revision: u64,
        reason: String,
    },
}

/// Index backend contract used by `LexicalPipeline`.
pub trait LexicalIndexBackend {
    /// Apply a planned lexical action.
    ///
    /// # Errors
    ///
    /// Returns backend-specific errors while persisting the lexical action.
    fn apply(&mut self, action: LexicalAction) -> SearchResult<()>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InMemoryLexicalEntry {
    pub revision: u64,
    pub chunks: Vec<LexicalChunk>,
}

/// In-memory lexical backend used for deterministic tests and dry-runs.
#[derive(Debug, Clone, Default)]
pub struct InMemoryLexicalBackend {
    entries: BTreeMap<String, InMemoryLexicalEntry>,
}

impl InMemoryLexicalBackend {
    #[must_use]
    pub fn get(&self, doc_id: &str) -> Option<&InMemoryLexicalEntry> {
        self.entries.get(doc_id)
    }

    #[must_use]
    pub fn contains(&self, doc_id: &str) -> bool {
        self.entries.contains_key(doc_id)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl LexicalIndexBackend for InMemoryLexicalBackend {
    fn apply(&mut self, action: LexicalAction) -> SearchResult<()> {
        match action {
            LexicalAction::Upsert {
                doc_id,
                revision,
                chunks,
                ..
            } => {
                self.entries
                    .insert(doc_id, InMemoryLexicalEntry { revision, chunks });
            }
            LexicalAction::Delete { doc_id, .. } => {
                self.entries.remove(&doc_id);
            }
            LexicalAction::Skip { .. } => {}
        }
        Ok(())
    }
}

/// Quill adapter for the deterministic lexical mutation planner.
///
/// Planning remains synchronous and backend-neutral. Actions are staged so a
/// caller can flush one bounded batch through Quill's async API while
/// preserving planner order across upsert and delete barriers.
pub struct QuillLexicalBackend<'a> {
    index: &'a QuillIndex,
    pending: Vec<LexicalAction>,
}

/// Classification totals from one crash-resumable Quill flush.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QuillResumeStats {
    /// Durable IDHASH miss; a new row was staged.
    pub absent: u64,
    /// Durable IDMAP hash matched; the existing docid was preserved.
    pub unchanged: u64,
    /// Durable IDMAP hash differed; an upsert was staged.
    pub changed: u64,
    /// A stale durable identifier was tombstoned.
    pub deleted: u64,
}

impl<'a> QuillLexicalBackend<'a> {
    #[must_use]
    pub const fn new(index: &'a QuillIndex) -> Self {
        Self {
            index,
            pending: Vec::new(),
        }
    }

    #[must_use]
    pub const fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Flush every planned action in order through Quill.
    ///
    /// Contiguous upserts share one `index_documents` call. Deletes form
    /// ordering barriers so repeated document ids retain planner order.
    ///
    /// # Errors
    ///
    /// Returns the typed Quill failure converted to the workspace search error.
    pub async fn flush(&mut self, cx: &Cx) -> SearchResult<()> {
        self.flush_inner(cx, false).await.map(|_| ())
    }

    /// Flush planned actions while preserving exact durable rows on restart.
    ///
    /// Each upsert probes Quill's published IDHASH. An equal IDMAP content
    /// witness is skipped, a mismatch is upserted, and a miss is inserted.
    /// Unchanged documents therefore retain their original Q1 docids.
    ///
    /// # Errors
    ///
    /// Returns typed Quill identity, hashing, indexing, or deletion failures.
    pub async fn flush_resumable(&mut self, cx: &Cx) -> SearchResult<QuillResumeStats> {
        self.flush_inner(cx, true).await
    }

    async fn flush_inner(&mut self, cx: &Cx, resumable: bool) -> SearchResult<QuillResumeStats> {
        let actions = std::mem::take(&mut self.pending);
        let mut documents = Vec::new();
        let mut stats = QuillResumeStats::default();

        for action in actions {
            match action {
                LexicalAction::Upsert {
                    doc_id,
                    title,
                    metadata,
                    chunks,
                    ..
                } => {
                    let content = chunks_into_index_content(chunks);
                    let mut document = IndexableDocument::new(doc_id, content);
                    document.title = title;
                    document.metadata = metadata;
                    if resumable {
                        let candidate_hash = indexable_document_content_hash(&document)?;
                        match self.index.document_witness(&document.id)? {
                            Some(witness) if witness.content_hash == candidate_hash => {
                                stats.unchanged = stats.unchanged.saturating_add(1);
                                continue;
                            }
                            Some(_) => {
                                stats.changed = stats.changed.saturating_add(1);
                            }
                            None => {
                                stats.absent = stats.absent.saturating_add(1);
                            }
                        }
                    }
                    documents.push(document);
                }
                LexicalAction::Delete { doc_id, .. } => {
                    if !documents.is_empty() {
                        LexicalSearch::index_documents(self.index, cx, &documents).await?;
                        documents.clear();
                    }
                    if self.index.has_uncommitted_changes() {
                        LexicalSearch::commit(self.index, cx).await?;
                    }
                    if self.index.delete_document(cx, &doc_id).await? {
                        stats.deleted = stats.deleted.saturating_add(1);
                    }
                }
                LexicalAction::Skip { .. } => {}
            }
        }
        if !documents.is_empty() {
            LexicalSearch::index_documents(self.index, cx, &documents).await?;
        }
        Ok(stats)
    }
}

fn chunks_into_index_content(chunks: Vec<LexicalChunk>) -> String {
    let mut chunks = chunks.into_iter();
    let Some(first) = chunks.next() else {
        return String::new();
    };
    let mut content = first.text;
    for chunk in chunks {
        content.push('\n');
        content.push_str(&chunk.text);
    }
    content
}

impl LexicalIndexBackend for QuillLexicalBackend<'_> {
    fn apply(&mut self, action: LexicalAction) -> SearchResult<()> {
        self.pending.push(action);
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LexicalBatchStats {
    pub planned: u64,
    pub upserted: u64,
    pub deleted: u64,
    pub skipped: u64,
    pub emitted_chunks: u64,
    pub emitted_tokens: u64,
}

impl LexicalBatchStats {
    fn record_action(&mut self, action: &LexicalAction) {
        self.planned = self.planned.saturating_add(1);
        match action {
            LexicalAction::Upsert { chunks, .. } => {
                self.upserted = self.upserted.saturating_add(1);
                self.emitted_chunks = self.emitted_chunks.saturating_add(chunks.len() as u64);
                self.emitted_tokens = self
                    .emitted_tokens
                    .saturating_add(chunks.iter().map(|chunk| chunk.token_count as u64).sum());
            }
            LexicalAction::Delete { .. } => {
                self.deleted = self.deleted.saturating_add(1);
            }
            LexicalAction::Skip { .. } => {
                self.skipped = self.skipped.saturating_add(1);
            }
        }
    }
}

/// Backend-agnostic lexical indexing orchestrator.
#[derive(Debug, Clone)]
pub struct LexicalPipeline<B: LexicalIndexBackend> {
    chunk_policy: LexicalChunkPolicy,
    performance_targets: LexicalPerformanceTargets,
    backend: B,
}

impl<B: LexicalIndexBackend> LexicalPipeline<B> {
    #[must_use]
    pub fn new(backend: B) -> Self {
        Self {
            chunk_policy: LexicalChunkPolicy::default(),
            performance_targets: LexicalPerformanceTargets::default(),
            backend,
        }
    }

    #[must_use]
    pub const fn with_chunk_policy(mut self, chunk_policy: LexicalChunkPolicy) -> Self {
        self.chunk_policy = chunk_policy;
        self
    }

    #[must_use]
    pub const fn with_performance_targets(
        mut self,
        performance_targets: LexicalPerformanceTargets,
    ) -> Self {
        self.performance_targets = performance_targets;
        self
    }

    #[must_use]
    pub const fn chunk_policy(&self) -> LexicalChunkPolicy {
        self.chunk_policy
    }

    #[must_use]
    pub const fn performance_targets(&self) -> LexicalPerformanceTargets {
        self.performance_targets
    }

    #[must_use]
    pub const fn backend(&self) -> &B {
        &self.backend
    }

    #[must_use]
    pub const fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    #[must_use]
    pub fn into_backend(self) -> B {
        self.backend
    }

    /// Index initial corpus rows.
    ///
    /// # Errors
    ///
    /// Returns an error if a mutation is invalid or backend application fails.
    pub fn apply_initial(&mut self, docs: &[LexicalMutation]) -> SearchResult<LexicalBatchStats> {
        self.apply_mutations(docs)
    }

    /// Apply incremental lexical updates/deletes/reclassifications.
    ///
    /// # Errors
    ///
    /// Returns an error if a mutation is invalid or backend application fails.
    pub fn apply_incremental(
        &mut self,
        updates: &[LexicalMutation],
    ) -> SearchResult<LexicalBatchStats> {
        self.apply_mutations(updates)
    }

    /// Map a mutation to a deterministic lexical action.
    ///
    /// # Errors
    ///
    /// Returns an error when mutation fields fail validation.
    pub fn plan_action(&self, mutation: &LexicalMutation) -> SearchResult<LexicalAction> {
        ensure_doc_id(&mutation.doc_id)?;

        if mutation.change == LexicalMutationKind::Delete {
            return Ok(LexicalAction::Delete {
                doc_id: mutation.doc_id.clone(),
                revision: mutation.revision,
                reason: mutation.reason.clone(),
            });
        }

        if matches!(
            mutation.ingestion_class,
            IngestionClass::MetadataOnly | IngestionClass::Skip
        ) {
            return Ok(LexicalAction::Delete {
                doc_id: mutation.doc_id.clone(),
                revision: mutation.revision,
                reason: "reclassified_non_lexical".to_owned(),
            });
        }

        let body = mutation.text.as_deref().unwrap_or_default();
        if body.trim().is_empty() {
            return Ok(LexicalAction::Delete {
                doc_id: mutation.doc_id.clone(),
                revision: mutation.revision,
                reason: "empty_text".to_owned(),
            });
        }

        let chunks = self.chunk_policy.chunk_text(body);
        if chunks.is_empty() {
            return Ok(LexicalAction::Skip {
                doc_id: mutation.doc_id.clone(),
                revision: mutation.revision,
                reason: "no_chunks_emitted".to_owned(),
            });
        }

        Ok(LexicalAction::Upsert {
            doc_id: mutation.doc_id.clone(),
            revision: mutation.revision,
            title: mutation.title.clone(),
            metadata: mutation.metadata.clone(),
            chunks,
        })
    }

    fn apply_mutations(&mut self, updates: &[LexicalMutation]) -> SearchResult<LexicalBatchStats> {
        let mut stats = LexicalBatchStats::default();

        for mutation in updates {
            let action = self.plan_action(mutation)?;
            stats.record_action(&action);

            match &action {
                LexicalAction::Upsert {
                    doc_id,
                    revision,
                    chunks,
                    ..
                } => {
                    debug!(
                        target: "frankensearch.fsfs.lexical",
                        doc_id,
                        revision,
                        chunks = chunks.len(),
                        reason = mutation.reason,
                        "lexical upsert applied"
                    );
                }
                LexicalAction::Delete {
                    doc_id,
                    revision,
                    reason,
                } => {
                    debug!(
                        target: "frankensearch.fsfs.lexical",
                        doc_id,
                        revision,
                        reason,
                        "lexical delete applied"
                    );
                }
                LexicalAction::Skip {
                    doc_id,
                    revision,
                    reason,
                } => {
                    debug!(
                        target: "frankensearch.fsfs.lexical",
                        doc_id,
                        revision,
                        reason,
                        "lexical mutation skipped"
                    );
                }
            }

            self.backend.apply(action)?;
        }

        Ok(stats)
    }
}

fn ensure_doc_id(doc_id: &str) -> SearchResult<()> {
    if doc_id.trim().is_empty() {
        return Err(SearchError::InvalidConfig {
            field: "lexical.doc_id".to_owned(),
            value: doc_id.to_owned(),
            reason: "must not be empty".to_owned(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use asupersync::test_utils::run_test_with_cx;
    use frankensearch_quill::{QuillConfig, QuillIndex, SegmentStatsProvider};

    use super::{
        InMemoryLexicalBackend, LexicalAction, LexicalChunkPolicy, LexicalMutation,
        LexicalPerformanceTargets, LexicalPipeline, QuillLexicalBackend,
        TARGET_INCREMENTAL_P95_LATENCY_MS, TARGET_INCREMENTAL_UPDATES_PER_SECOND,
        TARGET_INITIAL_DOCS_PER_SECOND, chunks_into_index_content, tokenize_lexical,
    };
    use crate::config::IngestionClass;

    #[test]
    fn tokenize_lexical_preserves_code_and_path_tokens() {
        let tokens = tokenize_lexical("src/main.rs -> fn run_fast(x: i32) { return x; }");
        let token_texts: Vec<&str> = tokens.iter().map(|token| token.text.as_str()).collect();
        assert!(token_texts.contains(&"src/main.rs"));
        assert!(token_texts.contains(&"run_fast"));
        assert!(token_texts.contains(&"i32"));
    }

    #[test]
    fn chunk_policy_is_deterministic_with_overlap() {
        let policy = LexicalChunkPolicy {
            max_chars: 10,
            overlap_chars: 3,
        };
        let text = "abcdefghijklmnopqrstuvwxyz";
        let chunks = policy.chunk_text(text);
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0].byte_start, 0);
        assert_eq!(chunks[0].byte_end, 10);
        assert_eq!(chunks[1].byte_start, 7);
        assert_eq!(chunks[1].byte_end, 17);
    }

    #[test]
    fn quill_content_is_projected_from_planned_chunks() {
        let chunks = LexicalChunkPolicy {
            max_chars: 10,
            overlap_chars: 3,
        }
        .chunk_text("abcdefghijklmnopqrstuvwxyz");

        assert_eq!(
            chunks_into_index_content(chunks),
            "abcdefghij\nhijklmnopq\nopqrstuvwx\nvwxyz"
        );
    }

    #[test]
    fn reclassification_to_non_lexical_emits_delete_action() {
        let pipeline = LexicalPipeline::new(InMemoryLexicalBackend::default());
        let mutation = LexicalMutation::upsert(
            "doc-a",
            2,
            IngestionClass::MetadataOnly,
            "still has text",
            "policy downgrade",
        );
        let action = pipeline.plan_action(&mutation).expect("plan action");
        assert!(matches!(action, LexicalAction::Delete { .. }));
    }

    #[test]
    fn pipeline_supports_initial_and_incremental_updates() {
        let mut pipeline = LexicalPipeline::new(InMemoryLexicalBackend::default());

        let initial = [LexicalMutation::upsert(
            "doc-a",
            1,
            IngestionClass::FullSemanticLexical,
            "rust async structured concurrency",
            "initial ingest",
        )];
        let initial_stats = pipeline.apply_initial(&initial).expect("initial apply");
        assert_eq!(initial_stats.upserted, 1);
        assert!(pipeline.backend().contains("doc-a"));
        assert_eq!(
            pipeline.backend().get("doc-a").expect("entry").revision,
            1,
            "initial revision should be tracked"
        );

        let update = [LexicalMutation::upsert(
            "doc-a",
            2,
            IngestionClass::LexicalOnly,
            "updated lexical payload",
            "file changed",
        )];
        let update_stats = pipeline
            .apply_incremental(&update)
            .expect("incremental upsert");
        assert_eq!(update_stats.upserted, 1);
        assert_eq!(
            pipeline.backend().get("doc-a").expect("entry").revision,
            2,
            "incremental update should advance revision"
        );

        let reclassify = [LexicalMutation::upsert(
            "doc-a",
            3,
            IngestionClass::Skip,
            "text ignored",
            "policy reclassification",
        )];
        let delete_stats = pipeline
            .apply_incremental(&reclassify)
            .expect("incremental delete");
        assert_eq!(delete_stats.deleted, 1);
        assert!(!pipeline.backend().contains("doc-a"));
    }

    #[test]
    fn quill_backend_flushes_planned_upserts_and_deletes() {
        run_test_with_cx(|cx| async move {
            let index = QuillIndex::in_memory(QuillConfig {
                max_ingest_shards: 1,
                deterministic_ingest: true,
                ..QuillConfig::default()
            })
            .expect("create in-memory Quill index");
            let backend = QuillLexicalBackend::new(&index);
            let mut pipeline = LexicalPipeline::new(backend);
            let mutations = [
                LexicalMutation::upsert(
                    "doc-a",
                    1,
                    IngestionClass::FullSemanticLexical,
                    "alpha lexical body",
                    "initial",
                )
                .with_title("Alpha")
                .with_metadata("source", "test"),
                LexicalMutation::upsert(
                    "doc-b",
                    1,
                    IngestionClass::LexicalOnly,
                    "beta lexical body",
                    "initial",
                ),
            ];
            let stats = pipeline
                .apply_initial(&mutations)
                .expect("plan Quill batch");
            assert_eq!(stats.upserted, 2);
            assert_eq!(pipeline.backend().pending_len(), 2);
            pipeline
                .backend_mut()
                .flush(&cx)
                .await
                .expect("flush Quill batch");
            index.commit(&cx).await.expect("commit Quill batch");
            assert_eq!(index.segment_stats().live_docs, 2);
            assert_eq!(
                index
                    .search_doc_ids(&cx, "alpha", 10)
                    .expect("query Quill")
                    .into_iter()
                    .map(|hit| hit.document_id)
                    .collect::<Vec<_>>(),
                vec!["doc-a"]
            );

            let update = [LexicalMutation::upsert(
                "doc-a",
                2,
                IngestionClass::FullSemanticLexical,
                "gamma replacement body",
                "changed",
            )];
            pipeline
                .apply_incremental(&update)
                .expect("plan Quill update");
            pipeline
                .backend_mut()
                .flush(&cx)
                .await
                .expect("flush Quill update");
            index.commit(&cx).await.expect("commit Quill update");
            assert!(
                index
                    .search_doc_ids(&cx, "alpha", 10)
                    .expect("query replaced Quill text")
                    .is_empty()
            );
            assert_eq!(
                index
                    .search_doc_ids(&cx, "gamma", 10)
                    .expect("query updated Quill text")
                    .into_iter()
                    .map(|hit| hit.document_id)
                    .collect::<Vec<_>>(),
                vec!["doc-a"]
            );

            let delete = [LexicalMutation::delete(
                "doc-a",
                3,
                IngestionClass::FullSemanticLexical,
                "removed",
            )];
            let stats = pipeline
                .apply_incremental(&delete)
                .expect("plan Quill delete");
            assert_eq!(stats.deleted, 1);
            pipeline
                .backend_mut()
                .flush(&cx)
                .await
                .expect("flush Quill delete");
            index.commit(&cx).await.expect("commit Quill delete");
            assert!(
                index
                    .search_doc_ids(&cx, "alpha", 10)
                    .expect("query deleted Quill document")
                    .is_empty()
            );

            let mixed = [
                LexicalMutation::upsert(
                    "doc-c",
                    4,
                    IngestionClass::FullSemanticLexical,
                    "delta replacement body",
                    "mixed barrier",
                ),
                LexicalMutation::delete("doc-b", 4, IngestionClass::LexicalOnly, "mixed barrier"),
            ];
            pipeline
                .apply_incremental(&mixed)
                .expect("plan mixed Quill barrier");
            pipeline
                .backend_mut()
                .flush_resumable(&cx)
                .await
                .expect("flush upsert before delete barrier");
            assert_eq!(
                index
                    .search_doc_ids(&cx, "delta", 10)
                    .expect("query mixed upsert")
                    .into_iter()
                    .map(|hit| hit.document_id)
                    .collect::<Vec<_>>(),
                vec!["doc-c"]
            );
            assert!(
                index
                    .search_doc_ids(&cx, "beta", 10)
                    .expect("query mixed delete")
                    .is_empty()
            );

            let staged = [LexicalMutation::upsert(
                "doc-d",
                5,
                IngestionClass::FullSemanticLexical,
                "epsilon pending body",
                "cross-flush barrier",
            )];
            pipeline
                .apply_incremental(&staged)
                .expect("plan cross-flush upsert");
            pipeline
                .backend_mut()
                .flush_resumable(&cx)
                .await
                .expect("stage cross-flush upsert");
            assert!(index.has_uncommitted_changes());

            let cross_flush_delete = [LexicalMutation::delete(
                "doc-c",
                6,
                IngestionClass::FullSemanticLexical,
                "cross-flush barrier",
            )];
            pipeline
                .apply_incremental(&cross_flush_delete)
                .expect("plan cross-flush delete");
            pipeline
                .backend_mut()
                .flush_resumable(&cx)
                .await
                .expect("commit prior upsert before cross-flush delete");
            assert!(
                index
                    .search_doc_ids(&cx, "delta", 10)
                    .expect("query cross-flush delete")
                    .is_empty()
            );
            assert_eq!(
                index
                    .search_doc_ids(&cx, "epsilon", 10)
                    .expect("query cross-flush upsert")
                    .into_iter()
                    .map(|hit| hit.document_id)
                    .collect::<Vec<_>>(),
                vec!["doc-d"]
            );
        });
    }

    #[test]
    fn quill_resumable_flush_preserves_equal_docid_and_replaces_hash_mismatch() {
        run_test_with_cx(|cx| async move {
            let index = QuillIndex::in_memory(QuillConfig {
                max_ingest_shards: 1,
                deterministic_ingest: true,
                ..QuillConfig::default()
            })
            .expect("create resumable Quill fixture");
            let mut pipeline = LexicalPipeline::new(QuillLexicalBackend::new(&index));
            let initial = [
                LexicalMutation::upsert(
                    "stable",
                    1,
                    IngestionClass::FullSemanticLexical,
                    "common stable body",
                    "initial",
                ),
                LexicalMutation::upsert(
                    "changed",
                    1,
                    IngestionClass::FullSemanticLexical,
                    "common old body",
                    "initial",
                ),
            ];
            pipeline.apply_initial(&initial).expect("plan initial rows");
            pipeline
                .backend_mut()
                .flush(&cx)
                .await
                .expect("flush initial rows");
            index.commit(&cx).await.expect("commit initial rows");
            let stable_before = index
                .document_witness("stable")
                .expect("probe stable row")
                .expect("stable row");
            let changed_before = index
                .document_witness("changed")
                .expect("probe changed row")
                .expect("changed row");

            let resumed = [
                initial[0].clone(),
                LexicalMutation::upsert(
                    "changed",
                    2,
                    IngestionClass::FullSemanticLexical,
                    "common replacement body",
                    "resume_changed",
                ),
                LexicalMutation::upsert(
                    "absent",
                    1,
                    IngestionClass::FullSemanticLexical,
                    "common new body",
                    "resume_absent",
                ),
            ];
            pipeline.apply_initial(&resumed).expect("plan resumed rows");
            let resume_stats = pipeline
                .backend_mut()
                .flush_resumable(&cx)
                .await
                .expect("classify resumed rows");
            assert_eq!(resume_stats.unchanged, 1);
            assert_eq!(resume_stats.changed, 1);
            assert_eq!(resume_stats.absent, 1);
            assert_eq!(resume_stats.deleted, 0);
            index.commit(&cx).await.expect("commit resumed rows");

            let stable_after = index
                .document_witness("stable")
                .expect("probe stable row after resume")
                .expect("stable row after resume");
            let changed_after = index
                .document_witness("changed")
                .expect("probe changed row after resume")
                .expect("changed row after resume");
            assert_eq!(
                stable_after, stable_before,
                "equal content must preserve both docid and IDMAP hash"
            );
            assert_ne!(
                changed_after.global_docid, changed_before.global_docid,
                "hash mismatch must allocate a fresh Q1 docid"
            );
            assert_ne!(changed_after.content_hash, changed_before.content_hash);
            assert!(
                index
                    .search_doc_ids(&cx, "old", 10)
                    .expect("query tombstoned content")
                    .is_empty()
            );
            assert_eq!(
                index
                    .search_doc_ids(&cx, "replacement", 10)
                    .expect("query replacement content")
                    .into_iter()
                    .map(|hit| hit.document_id)
                    .collect::<Vec<_>>(),
                vec!["changed"]
            );
        });
    }

    #[test]
    fn quill_resumable_bulk_reopens_each_cadence_with_result_equivalence() {
        run_test_with_cx(|cx| async move {
            let documents = (0..7)
                .map(|ordinal| {
                    LexicalMutation::upsert(
                        format!("doc-{ordinal}"),
                        1,
                        IngestionClass::FullSemanticLexical,
                        format!("common crash resumable document {ordinal}"),
                        "bulk_fixture",
                    )
                })
                .collect::<Vec<_>>();
            let config = || QuillConfig {
                max_ingest_shards: 1,
                deterministic_ingest: true,
                scribe_shard_budget_bytes: 1,
                bulk_load_mode: true,
                bulk_publish_segment_cadence: 2,
                tier_fanout: 8,
                ..QuillConfig::default()
            };

            for published_prefix in [2_usize, 4, 6] {
                let root = tempfile::tempdir().expect("bulk cadence root");
                let resumed_path = root.path().join("resumed");
                let control_path = root.path().join("control");
                let partial = QuillIndex::create(&cx, &resumed_path, config())
                    .await
                    .expect("create partial bulk index");
                let mut partial_pipeline = LexicalPipeline::new(QuillLexicalBackend::new(&partial));
                partial_pipeline
                    .apply_initial(&documents[..published_prefix])
                    .expect("plan partial prefix");
                partial_pipeline
                    .backend_mut()
                    .flush(&cx)
                    .await
                    .expect("flush partial prefix");
                assert_eq!(
                    partial.doc_count(),
                    u64::try_from(published_prefix).expect("prefix fits u64"),
                    "prefix={published_prefix}: cadence publication must be durable"
                );
                let prefix_witnesses = documents[..published_prefix]
                    .iter()
                    .map(|document| {
                        partial
                            .document_witness(&document.doc_id)
                            .expect("probe prefix")
                            .expect("published prefix row")
                    })
                    .collect::<Vec<_>>();
                drop(partial_pipeline);
                drop(partial);

                let resumed = QuillIndex::open(&cx, &resumed_path, config())
                    .await
                    .expect("reopen partial bulk index");
                let mut resumed_pipeline = LexicalPipeline::new(QuillLexicalBackend::new(&resumed));
                resumed_pipeline
                    .apply_initial(&documents)
                    .expect("plan resumed corpus");
                let stats = resumed_pipeline
                    .backend_mut()
                    .flush_resumable(&cx)
                    .await
                    .expect("resume corpus through IDHASH");
                assert_eq!(
                    stats.unchanged,
                    u64::try_from(published_prefix).expect("prefix fits u64")
                );
                assert_eq!(
                    stats.absent,
                    u64::try_from(documents.len() - published_prefix).expect("suffix fits u64")
                );
                assert_eq!(stats.changed, 0);
                drop(resumed_pipeline);
                resumed
                    .finish_bulk_load(&cx)
                    .await
                    .expect("finish resumed bulk index");
                for (document, expected) in
                    documents[..published_prefix].iter().zip(prefix_witnesses)
                {
                    assert_eq!(
                        resumed
                            .document_witness(&document.doc_id)
                            .expect("probe resumed prefix")
                            .expect("resumed prefix row"),
                        expected,
                        "prefix={published_prefix}: unchanged row must retain its docid"
                    );
                }

                let control = QuillIndex::create(&cx, &control_path, config())
                    .await
                    .expect("create uninterrupted control");
                let mut control_pipeline = LexicalPipeline::new(QuillLexicalBackend::new(&control));
                control_pipeline
                    .apply_initial(&documents)
                    .expect("plan control corpus");
                control_pipeline
                    .backend_mut()
                    .flush(&cx)
                    .await
                    .expect("flush control corpus");
                drop(control_pipeline);
                control
                    .finish_bulk_load(&cx)
                    .await
                    .expect("finish control corpus");

                let result_artifact = |index: &QuillIndex| {
                    index
                        .search_doc_ids(&cx, "common", documents.len())
                        .expect("query result artifact")
                        .into_iter()
                        .map(|hit| (hit.document_id, hit.score.to_bits()))
                        .collect::<Vec<_>>()
                };
                assert_eq!(
                    result_artifact(&resumed),
                    result_artifact(&control),
                    "prefix={published_prefix}: resumed and uninterrupted results must match"
                );
            }
        });
    }

    #[test]
    #[ignore = "release-perf contract gate; run explicitly on an isolated RCH worker"]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    fn quill_backend_meets_fsfs_throughput_contract() {
        // The ranked visibility probe fans out through Rayon. A thread-local
        // subscriber override does not reach those workers, so install the
        // no-op subscriber before `run_test_with_cx` can initialize tracing.
        // Otherwise per-segment INFO output becomes part of the timed workload.
        tracing::subscriber::set_global_default(tracing::subscriber::NoSubscriber::default())
            .expect("install quiet subscriber for isolated performance contract");
        tracing::subscriber::with_default(tracing::subscriber::NoSubscriber::default(), || {
            run_test_with_cx(|cx| async move {
                const INITIAL_DOCS: usize = 20_000;
                const INITIAL_BATCH: usize = 256;
                const UPDATE_DOCS: usize = 5_000;
                const UPDATE_BATCH: usize = 50;

                let initial_mutations = (0..INITIAL_DOCS)
                    .map(|ordinal| {
                        LexicalMutation::upsert(
                            format!("bulk-{ordinal:05}"),
                            1,
                            IngestionClass::FullSemanticLexical,
                            format!("bulk contract document {ordinal:05} sharedterm"),
                            "contract_initial",
                        )
                    })
                    .collect::<Vec<_>>();
                let contract_root = tempfile::tempdir().expect("create contract index root");
                let bulk_index = QuillIndex::create(
                    &cx,
                    contract_root.path().join("bulk"),
                    QuillConfig {
                        bulk_load_mode: true,
                        ..QuillConfig::default()
                    },
                )
                .await
                .expect("create bulk contract index");
                let mut bulk_pipeline = LexicalPipeline::new(QuillLexicalBackend::new(&bulk_index));
                let initial_start = Instant::now();
                for batch in initial_mutations.chunks(INITIAL_BATCH) {
                    bulk_pipeline
                        .apply_initial(batch)
                        .expect("plan initial contract batch");
                    bulk_pipeline
                        .backend_mut()
                        .flush(&cx)
                        .await
                        .expect("flush initial contract batch");
                }
                bulk_index
                    .finish_bulk_load(&cx)
                    .await
                    .expect("finish bulk contract index");
                let initial_elapsed = initial_start.elapsed();
                assert_eq!(bulk_index.segment_stats().live_docs, INITIAL_DOCS);

                let watch_index = QuillIndex::create(
                    &cx,
                    contract_root.path().join("watch"),
                    QuillConfig {
                        max_ingest_shards: 1,
                        deterministic_ingest: true,
                        ..QuillConfig::default()
                    },
                )
                .await
                .expect("create watch contract index");
                let mut watch_pipeline =
                    LexicalPipeline::new(QuillLexicalBackend::new(&watch_index));
                let seed = (0..UPDATE_DOCS)
                    .map(|ordinal| {
                        LexicalMutation::upsert(
                            format!("watch-{ordinal:05}"),
                            1,
                            IngestionClass::FullSemanticLexical,
                            format!("watch seed document {ordinal:05}"),
                            "contract_seed",
                        )
                    })
                    .collect::<Vec<_>>();
                watch_pipeline
                    .apply_initial(&seed)
                    .expect("plan watch seed");
                watch_pipeline
                    .backend_mut()
                    .flush(&cx)
                    .await
                    .expect("flush watch seed");
                watch_index.commit(&cx).await.expect("commit watch seed");

                let updates = (0..UPDATE_DOCS)
                    .map(|ordinal| {
                        LexicalMutation::upsert(
                            format!("watch-{ordinal:05}"),
                            2,
                            IngestionClass::FullSemanticLexical,
                            format!("watch updated document {ordinal:05} searchabletoken"),
                            "contract_update",
                        )
                    })
                    .collect::<Vec<_>>();
                let mut batch_latencies = Vec::with_capacity(UPDATE_DOCS / UPDATE_BATCH);
                let update_start = Instant::now();
                for batch in updates.chunks(UPDATE_BATCH) {
                    let batch_start = Instant::now();
                    watch_pipeline
                        .apply_incremental(batch)
                        .expect("plan watch contract batch");
                    watch_pipeline
                        .backend_mut()
                        .flush(&cx)
                        .await
                        .expect("flush watch contract batch");
                    watch_index
                        .commit(&cx)
                        .await
                        .expect("commit watch contract batch");
                    assert!(
                        !watch_index
                            .search_doc_ids(&cx, "searchabletoken", 1)
                            .expect("probe update visibility")
                            .is_empty()
                    );
                    batch_latencies.push(batch_start.elapsed());
                }
                let update_elapsed = update_start.elapsed();

                batch_latencies.sort_unstable();
                let p95_index = batch_latencies
                    .len()
                    .saturating_mul(95)
                    .div_ceil(100)
                    .saturating_sub(1);
                let p95_latency = batch_latencies
                    .get(p95_index)
                    .copied()
                    .expect("contract run records update batches");
                // Every document in a watch batch becomes searchable only after the
                // batch publication barrier, so the full batch latency is the honest
                // update-to-searchable latency for each member of that batch.
                let p95_update_to_searchable_ms = p95_latency.as_secs_f64() * 1_000.0;
                let initial_rate = INITIAL_DOCS as f64 / initial_elapsed.as_secs_f64();
                let update_rate = UPDATE_DOCS as f64 / update_elapsed.as_secs_f64();
                let observed_initial = initial_rate.floor().clamp(0.0, f64::from(u32::MAX)) as u32;
                let observed_updates = update_rate.floor().clamp(0.0, f64::from(u32::MAX)) as u32;
                let observed_p95 = p95_update_to_searchable_ms
                    .ceil()
                    .clamp(0.0, f64::from(u32::MAX)) as u32;
                eprintln!(
                    "Quill fsfs contract: initial={observed_initial} docs/s updates={observed_updates} updates/s update-to-searchable-p95={p95_update_to_searchable_ms:.3} ms"
                );
                assert!(
                    LexicalPerformanceTargets::default().meets_contract(
                        observed_initial,
                        observed_updates,
                        observed_p95,
                    ),
                    "Quill adapter missed the 20k/5k/25ms fsfs contract"
                );
            });
        });
    }

    #[test]
    fn empty_text_upsert_is_planned_as_delete() {
        let pipeline = LexicalPipeline::new(InMemoryLexicalBackend::default());
        let mutation = LexicalMutation::upsert(
            "doc-empty",
            7,
            IngestionClass::FullSemanticLexical,
            "   \n\t",
            "empty payload",
        );
        let action = pipeline.plan_action(&mutation).expect("plan action");
        assert!(matches!(action, LexicalAction::Delete { .. }));
    }

    // ── Tokenizer edge cases ─────────────────────────────────────────

    #[test]
    fn tokenize_empty_string_returns_no_tokens() {
        assert!(tokenize_lexical("").is_empty());
    }

    #[test]
    fn tokenize_only_delimiters_returns_no_tokens() {
        assert!(tokenize_lexical("   !@# $%^ &*() ").is_empty());
    }

    #[test]
    fn tokenize_tracks_line_numbers() {
        let tokens = tokenize_lexical("first\nsecond\nthird");
        assert_eq!(tokens[0].line, 1);
        assert_eq!(tokens[0].text, "first");
        assert_eq!(tokens[1].line, 2);
        assert_eq!(tokens[1].text, "second");
        assert_eq!(tokens[2].line, 3);
        assert_eq!(tokens[2].text, "third");
    }

    #[test]
    fn tokenize_lowercases_output() {
        let tokens = tokenize_lexical("FooBar BAZZZ");
        assert_eq!(tokens[0].text, "foobar");
        assert_eq!(tokens[1].text, "bazzz");
    }

    #[test]
    fn tokenize_preserves_path_separators_and_colons() {
        let tokens = tokenize_lexical("http://example.com:8080/path");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].text, "http://example.com:8080/path");
    }

    #[test]
    fn count_lexical_tokens_matches_tokenize_lexical() {
        let inputs = [
            "",
            "hello world",
            "src/main.rs -> fn run_fast(x: i32) { return x; }",
            "   !@# $%^ &*() ",
            "multiline\ntext\n\nwith\tgaps",
        ];
        for input in inputs {
            let count = super::count_lexical_tokens(input);
            let tokens = tokenize_lexical(input);
            assert_eq!(count, tokens.len(), "Mismatch for input: {input:?}");
        }
    }

    /// The ASCII byte fast path must match the Unicode `chars()` reference for
    /// every input — including non-ASCII (which must take the fallback).
    #[test]
    fn count_lexical_tokens_ascii_fastpath_matches_chars() {
        // Reference: the pure chars() state machine (no fast path).
        fn reference(text: &str) -> usize {
            let mut count = 0;
            let mut in_token = false;
            for ch in text.chars() {
                if super::is_token_char(ch) {
                    in_token = true;
                } else if in_token {
                    in_token = false;
                    count += 1;
                }
            }
            count + usize::from(in_token)
        }
        let inputs = [
            "",
            "a",
            "hello world",
            "src/main.rs -> fn run_fast(x: i32) { return x; }",
            "   !@# $%^ &*() ",
            "trailing token",
            "leading.token/sep:colon-dash_underscore",
            "café résumé naïve",       // non-ASCII letters → fallback
            "日本語 トークン test123", // mixed CJK + ASCII → fallback
            "emoji 🚀 then word",
        ];
        for input in inputs {
            assert_eq!(
                super::count_lexical_tokens(input),
                reference(input),
                "fast path diverged from chars() reference for {input:?}"
            );
        }
    }

    // ── Chunker edge cases ──────────────────────────────────────────

    #[test]
    fn chunk_empty_text_returns_no_chunks() {
        let policy = LexicalChunkPolicy::default();
        assert!(policy.chunk_text("").is_empty());
    }

    #[test]
    fn chunk_text_shorter_than_max_returns_single_chunk() {
        let policy = LexicalChunkPolicy {
            max_chars: 100,
            overlap_chars: 10,
        };
        let chunks = policy.chunk_text("hello world");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "hello world");
        assert_eq!(chunks[0].ordinal, 0);
    }

    #[test]
    fn chunk_single_character() {
        let policy = LexicalChunkPolicy {
            max_chars: 10,
            overlap_chars: 3,
        };
        let chunks = policy.chunk_text("x");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "x");
    }

    #[test]
    fn chunk_multibyte_utf8_does_not_split_codepoints() {
        let policy = LexicalChunkPolicy {
            max_chars: 5,
            overlap_chars: 1,
        };
        // Each emoji is 4 bytes
        let text = "\u{1F600}\u{1F601}\u{1F602}\u{1F603}";
        let chunks = policy.chunk_text(text);
        // Should produce chunks without panicking on char boundaries
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            // Verify each chunk is valid UTF-8 (implicit from String type)
            assert!(!chunk.text.is_empty());
        }
    }

    #[test]
    fn chunk_ordinals_are_sequential() {
        let policy = LexicalChunkPolicy {
            max_chars: 5,
            overlap_chars: 1,
        };
        let chunks = policy.chunk_text("abcdefghijklmnopqrstuvwxyz");
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.ordinal, u32::try_from(i).unwrap());
        }
    }

    // ── plan_action edge cases ──────────────────────────────────────

    #[test]
    fn plan_action_rejects_empty_doc_id() {
        let pipeline = LexicalPipeline::new(InMemoryLexicalBackend::default());
        let mutation = LexicalMutation::upsert(
            "  ",
            1,
            IngestionClass::FullSemanticLexical,
            "some text",
            "test",
        );
        let error = pipeline.plan_action(&mutation).unwrap_err();
        assert!(
            matches!(error, frankensearch_core::SearchError::InvalidConfig { ref field, .. } if field == "lexical.doc_id"),
            "expected InvalidConfig for doc_id, got {error:?}"
        );
    }

    #[test]
    fn plan_action_delete_mutation_produces_delete_action() {
        let pipeline = LexicalPipeline::new(InMemoryLexicalBackend::default());
        let mutation =
            LexicalMutation::delete("doc-x", 5, IngestionClass::FullSemanticLexical, "removed");
        let action = pipeline.plan_action(&mutation).expect("plan");
        assert!(
            matches!(action, LexicalAction::Delete { ref doc_id, revision: 5, .. } if doc_id == "doc-x")
        );
    }

    #[test]
    fn plan_action_none_text_treated_as_empty() {
        let pipeline = LexicalPipeline::new(InMemoryLexicalBackend::default());
        let mutation = LexicalMutation {
            doc_id: "doc-none".into(),
            revision: 1,
            ingestion_class: IngestionClass::FullSemanticLexical,
            change: super::LexicalMutationKind::Upsert,
            text: None,
            title: None,
            metadata: std::collections::HashMap::new(),
            reason: "test".to_owned(),
        };
        let action = pipeline.plan_action(&mutation).expect("plan");
        assert!(
            matches!(action, LexicalAction::Delete { ref reason, .. } if reason == "empty_text")
        );
    }

    // ── Batch stats tracking ────────────────────────────────────────

    #[test]
    fn batch_stats_track_mixed_mutations() {
        let mut pipeline = LexicalPipeline::new(InMemoryLexicalBackend::default());
        let mutations = [
            LexicalMutation::upsert(
                "doc-a",
                1,
                IngestionClass::FullSemanticLexical,
                "hello world",
                "add",
            ),
            LexicalMutation::upsert(
                "doc-b",
                1,
                IngestionClass::FullSemanticLexical,
                "foo bar",
                "add",
            ),
            LexicalMutation::delete("doc-a", 2, IngestionClass::FullSemanticLexical, "remove"),
            LexicalMutation::upsert("doc-c", 1, IngestionClass::Skip, "ignored", "skip class"),
        ];
        let stats = pipeline.apply_initial(&mutations).expect("apply");
        assert_eq!(stats.planned, 4);
        assert_eq!(stats.upserted, 2);
        assert_eq!(stats.deleted, 2); // explicit delete + skip-class reclassification
        assert!(stats.emitted_chunks > 0);
        assert!(stats.emitted_tokens > 0);
    }

    // ── meets_contract failure cases ────────────────────────────────

    #[test]
    fn meets_contract_fails_when_below_throughput() {
        let targets = LexicalPerformanceTargets::default();
        assert!(!targets.meets_contract(100, 100, 5)); // initial too low
        assert!(!targets.meets_contract(100_000, 100, 5)); // incremental too low
        assert!(!targets.meets_contract(100_000, 100_000, 1000)); // latency too high
    }

    // ── InMemoryLexicalBackend ──────────────────────────────────────

    #[test]
    fn in_memory_backend_len_and_is_empty() {
        let mut pipeline = LexicalPipeline::new(InMemoryLexicalBackend::default());
        assert!(pipeline.backend().is_empty());
        assert_eq!(pipeline.backend().len(), 0);

        let mutations = [LexicalMutation::upsert(
            "doc-a",
            1,
            IngestionClass::FullSemanticLexical,
            "text",
            "add",
        )];
        pipeline.apply_initial(&mutations).expect("apply");
        assert!(!pipeline.backend().is_empty());
        assert_eq!(pipeline.backend().len(), 1);
    }

    // ── Builder pattern ─────────────────────────────────────────────

    #[test]
    fn pipeline_builder_applies_custom_policy_and_targets() {
        let policy = LexicalChunkPolicy {
            max_chars: 256,
            overlap_chars: 32,
        };
        let targets = LexicalPerformanceTargets {
            initial_docs_per_second: 50_000,
            incremental_updates_per_second: 10_000,
            incremental_p95_latency_ms: 10,
        };
        let pipeline = LexicalPipeline::new(InMemoryLexicalBackend::default())
            .with_chunk_policy(policy)
            .with_performance_targets(targets);
        assert_eq!(pipeline.chunk_policy().max_chars, 256);
        assert_eq!(
            pipeline.performance_targets().initial_docs_per_second,
            50_000
        );
    }

    // ── Original tests continue ─────────────────────────────────────

    #[test]
    fn performance_targets_have_expected_defaults() {
        let targets = LexicalPerformanceTargets::default();
        assert_eq!(
            targets.initial_docs_per_second,
            TARGET_INITIAL_DOCS_PER_SECOND
        );
        assert_eq!(
            targets.incremental_updates_per_second,
            TARGET_INCREMENTAL_UPDATES_PER_SECOND
        );
        assert_eq!(
            targets.incremental_p95_latency_ms,
            TARGET_INCREMENTAL_P95_LATENCY_MS
        );
        assert!(targets.meets_contract(
            TARGET_INITIAL_DOCS_PER_SECOND + 1_000,
            TARGET_INCREMENTAL_UPDATES_PER_SECOND + 500,
            TARGET_INCREMENTAL_P95_LATENCY_MS.saturating_sub(5),
        ));
    }
}
