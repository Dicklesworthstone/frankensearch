//! Deterministic lexical indexing pipeline model for fsfs.
//!
//! This module provides a backend-agnostic lexical pipeline contract covering:
//! - text chunking and tokenization for mixed prose/code corpora
//! - initial and incremental index mutation planning
//! - explicit update/delete behavior on change and reclassification
//! - measurable throughput/latency target contracts

use std::collections::BTreeMap;

use frankensearch_core::{SearchError, SearchResult};
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LexicalToken {
    pub text: String,
    pub line: u32,
    pub byte_start: usize,
    pub byte_end: usize,
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
            let token_count = tokenize_lexical(&chunk_text).len();
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

/// Split text into lexical tokens while preserving code/path-like identifiers.
#[must_use]
pub fn tokenize_lexical(text: &str) -> Vec<LexicalToken> {
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
                text: text[start..idx].to_ascii_lowercase(),
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
            text: text[start..].to_ascii_lowercase(),
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
            reason: reason.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LexicalAction {
    Upsert {
        doc_id: String,
        revision: u64,
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
    fn apply(&mut self, action: &LexicalAction) -> SearchResult<()>;
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
    fn apply(&mut self, action: &LexicalAction) -> SearchResult<()> {
        match action {
            LexicalAction::Upsert {
                doc_id,
                revision,
                chunks,
            } => {
                self.entries.insert(
                    doc_id.clone(),
                    InMemoryLexicalEntry {
                        revision: *revision,
                        chunks: chunks.clone(),
                    },
                );
            }
            LexicalAction::Delete { doc_id, .. } => {
                self.entries.remove(doc_id);
            }
            LexicalAction::Skip { .. } => {}
        }
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

        let body = mutation
            .text
            .as_deref()
            .map(str::trim)
            .unwrap_or_default()
            .to_owned();
        if body.is_empty() {
            return Ok(LexicalAction::Delete {
                doc_id: mutation.doc_id.clone(),
                revision: mutation.revision,
                reason: "empty_text".to_owned(),
            });
        }

        let chunks = self.chunk_policy.chunk_text(&body);
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
            chunks,
        })
    }

    fn apply_mutations(&mut self, updates: &[LexicalMutation]) -> SearchResult<LexicalBatchStats> {
        let mut stats = LexicalBatchStats::default();

        for mutation in updates {
            let action = self.plan_action(mutation)?;
            self.backend.apply(&action)?;
            stats.record_action(&action);

            match &action {
                LexicalAction::Upsert {
                    doc_id,
                    revision,
                    chunks,
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
    use super::{
        InMemoryLexicalBackend, LexicalAction, LexicalChunkPolicy, LexicalMutation,
        LexicalPerformanceTargets, LexicalPipeline, TARGET_INCREMENTAL_P95_LATENCY_MS,
        TARGET_INCREMENTAL_UPDATES_PER_SECOND, TARGET_INITIAL_DOCS_PER_SECOND, tokenize_lexical,
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
