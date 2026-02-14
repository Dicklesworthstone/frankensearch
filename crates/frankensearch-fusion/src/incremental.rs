//! Prefix-optimized incremental search mode for as-you-type experiences.
//!
//! [`IncrementalSearcher`] is a synchronous state machine that decides:
//! - Whether a search should run at all (enforces [`IncrementalConfig::min_prefix_len`])
//! - Which strategy to use based on query length
//! - Whether the new query is a prefix extension of the last query
//! - When to reuse the previous result set as a candidate pool
//!
//! # Architecture
//!
//! The searcher does NOT execute searches directly. Instead, the consumer:
//!
//! 1. Calls [`IncrementalSearcher::plan`] with the current query text
//! 2. Examines the returned [`SearchPlan`] for strategy and candidate pool
//! 3. Executes the appropriate search using their own infrastructure
//! 4. Calls [`IncrementalSearcher::update`] with the query and results
//!
//! This keeps the library synchronous, framework-agnostic, and testable
//! without requiring any async runtime or search infrastructure.
//!
//! # Strategy ladder
//!
//! | Query length | Strategy | Typical latency |
//! |---|---|---|
//! | `< min_prefix_len` | Skip (no search) | 0 ms |
//! | 1-2 chars | Lexical prefix only | <5 ms |
//! | 3-4 chars | Lexical + hash embedding | <10 ms |
//! | 5+ chars | Full hybrid (fast embedder) | <15 ms |
//! | After pause | Full two-tier quality | ~150 ms |

use std::time::Duration;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for incremental as-you-type search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalConfig {
    /// Minimum query length before any search runs. Default: 2.
    pub min_prefix_len: usize,

    /// Whether to prefer the hash embedder for very short queries (3-4 chars).
    /// Default: true.
    pub use_hash_embedder_for_short: bool,

    /// Hint: how long (ms) after the last keystroke before suggesting a
    /// full two-tier refinement. The consumer is responsible for tracking
    /// elapsed time. Default: 300.
    pub refine_after_pause_ms: u64,

    /// Maximum number of results to keep in the candidate pool for reuse.
    /// Default: 100.
    pub candidate_pool_size: usize,
}

impl Default for IncrementalConfig {
    fn default() -> Self {
        Self {
            min_prefix_len: 2,
            use_hash_embedder_for_short: true,
            refine_after_pause_ms: 300,
            candidate_pool_size: 100,
        }
    }
}

// ---------------------------------------------------------------------------
// Search strategy
// ---------------------------------------------------------------------------

/// Which search backend(s) to use for a given query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SearchStrategy {
    /// Query is below `min_prefix_len` — do not search.
    Skip,
    /// 1-2 characters: Tantivy prefix query only (fastest).
    LexicalPrefixOnly,
    /// 3-4 characters: Tantivy prefix + FNV-1a hash embedding.
    LexicalPlusHash,
    /// 5+ characters: Full hybrid search with fast embedder.
    HybridFast,
    /// After a typing pause: full two-tier search with quality refinement.
    FullTwoTier,
}

// ---------------------------------------------------------------------------
// Search plan
// ---------------------------------------------------------------------------

/// The recommended search plan for a given keystroke.
#[derive(Debug, Clone)]
pub struct SearchPlan {
    /// Recommended search strategy based on query length.
    pub strategy: SearchStrategy,

    /// Whether the candidate pool from the last search can be reused.
    ///
    /// When `true`, the consumer may re-rank [`candidate_doc_ids`] instead
    /// of scanning the full index. This gives `O(k)` per keystroke instead
    /// of `O(n)`.
    pub reuse_candidates: bool,

    /// Document IDs from the previous search that can be used as a
    /// candidate pool. Only populated when `reuse_candidates` is `true`.
    pub candidate_doc_ids: Vec<String>,
}

// ---------------------------------------------------------------------------
// Incremental searcher
// ---------------------------------------------------------------------------

/// Synchronous state machine for prefix-optimized incremental search.
///
/// See [module-level docs](self) for usage.
pub struct IncrementalSearcher {
    config: IncrementalConfig,
    last_query: Option<String>,
    last_doc_ids: Vec<String>,
}

impl IncrementalSearcher {
    /// Create a new incremental searcher with the given config.
    #[must_use]
    pub const fn new(config: IncrementalConfig) -> Self {
        Self {
            config,
            last_query: None,
            last_doc_ids: Vec::new(),
        }
    }

    /// Plan the search for the current query text.
    ///
    /// Returns a [`SearchPlan`] describing which strategy to use and
    /// whether the previous result set can be reused.
    #[must_use]
    pub fn plan(&self, query: &str) -> SearchPlan {
        let char_count = query.chars().count();

        if char_count < self.config.min_prefix_len {
            return SearchPlan {
                strategy: SearchStrategy::Skip,
                reuse_candidates: false,
                candidate_doc_ids: Vec::new(),
            };
        }

        let strategy = self.strategy_for_char_count(char_count);
        let reuse = self.is_prefix_extension(query) && !self.last_doc_ids.is_empty();

        SearchPlan {
            strategy,
            reuse_candidates: reuse,
            candidate_doc_ids: if reuse {
                self.last_doc_ids.clone()
            } else {
                Vec::new()
            },
        }
    }

    /// Update internal state after a search completes.
    ///
    /// Call this after executing the search plan so the next `plan()` call
    /// can detect prefix extensions and reuse the candidate pool.
    pub fn update(&mut self, query: &str, result_doc_ids: Vec<String>) {
        self.last_query = Some(query.to_owned());
        self.last_doc_ids = if result_doc_ids.len() > self.config.candidate_pool_size {
            result_doc_ids[..self.config.candidate_pool_size].to_vec()
        } else {
            result_doc_ids
        };
    }

    /// Whether the consumer should trigger a full two-tier refinement.
    ///
    /// The consumer passes the time elapsed since the last keystroke.
    /// Returns `true` if the pause exceeds `refine_after_pause_ms`.
    #[must_use]
    pub const fn should_refine(&self, elapsed_since_last_keystroke: Duration) -> bool {
        #[allow(clippy::cast_possible_truncation)] // Duration > u64::MAX millis is unreachable
        let millis = elapsed_since_last_keystroke.as_millis() as u64;
        millis >= self.config.refine_after_pause_ms
    }

    /// Whether the given query is a prefix extension of the last query.
    ///
    /// "sea" → "sear" → "search" are prefix extensions.
    /// "sea" → "mountain" is NOT a prefix extension.
    /// "search" → "searc" (backspace) is NOT a prefix extension.
    #[must_use]
    pub fn is_prefix_extension(&self, query: &str) -> bool {
        let Some(last) = &self.last_query else {
            return false;
        };
        query.len() > last.len() && query.starts_with(last.as_str())
    }

    /// Determine the search strategy based on character count.
    #[must_use]
    pub const fn strategy_for_char_count(&self, char_count: usize) -> SearchStrategy {
        if char_count < self.config.min_prefix_len {
            return SearchStrategy::Skip;
        }
        match char_count {
            0 => SearchStrategy::Skip,
            1..=2 => SearchStrategy::LexicalPrefixOnly,
            3..=4 => {
                if self.config.use_hash_embedder_for_short {
                    SearchStrategy::LexicalPlusHash
                } else {
                    SearchStrategy::HybridFast
                }
            }
            _ => SearchStrategy::HybridFast,
        }
    }

    /// Clear all cached state (query and results).
    ///
    /// Call this when the search context changes (e.g., switching document
    /// collections, user navigates away).
    pub fn reset(&mut self) {
        self.last_query = None;
        self.last_doc_ids.clear();
    }

    /// Borrow the current config.
    #[must_use]
    pub const fn config(&self) -> &IncrementalConfig {
        &self.config
    }

    /// Whether any previous search state exists.
    #[must_use]
    pub const fn has_previous_results(&self) -> bool {
        self.last_query.is_some() && !self.last_doc_ids.is_empty()
    }

    /// The last query string, if any.
    #[must_use]
    pub fn last_query(&self) -> Option<&str> {
        self.last_query.as_deref()
    }
}

impl std::fmt::Debug for IncrementalSearcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IncrementalSearcher")
            .field("config", &self.config)
            .field("last_query", &self.last_query)
            .field("cached_results", &self.last_doc_ids.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_searcher() -> IncrementalSearcher {
        IncrementalSearcher::new(IncrementalConfig::default())
    }

    // ── min_prefix_len enforcement ──────────────────────────────────

    #[test]
    fn skip_below_min_prefix_len() {
        let s = default_searcher();
        let plan = s.plan("a");
        assert_eq!(plan.strategy, SearchStrategy::Skip);
        assert!(!plan.reuse_candidates);
    }

    #[test]
    fn skip_empty_query() {
        let s = default_searcher();
        let plan = s.plan("");
        assert_eq!(plan.strategy, SearchStrategy::Skip);
    }

    #[test]
    fn search_at_min_prefix_len() {
        let s = default_searcher();
        // min_prefix_len=2, so 2 chars should search.
        let plan = s.plan("ab");
        assert_ne!(plan.strategy, SearchStrategy::Skip);
    }

    #[test]
    fn min_prefix_len_zero_allows_single_char() {
        let s = IncrementalSearcher::new(IncrementalConfig {
            min_prefix_len: 0,
            ..IncrementalConfig::default()
        });
        let plan = s.plan("x");
        assert_ne!(plan.strategy, SearchStrategy::Skip);
    }

    // ── Strategy ladder ─────────────────────────────────────────────

    #[test]
    fn strategy_1_2_chars_is_lexical_prefix() {
        let s = default_searcher();
        assert_eq!(
            s.strategy_for_char_count(2),
            SearchStrategy::LexicalPrefixOnly
        );
    }

    #[test]
    fn strategy_3_4_chars_is_lexical_plus_hash() {
        let s = default_searcher();
        assert_eq!(
            s.strategy_for_char_count(3),
            SearchStrategy::LexicalPlusHash
        );
        assert_eq!(
            s.strategy_for_char_count(4),
            SearchStrategy::LexicalPlusHash
        );
    }

    #[test]
    fn strategy_3_4_chars_without_hash_is_hybrid() {
        let s = IncrementalSearcher::new(IncrementalConfig {
            use_hash_embedder_for_short: false,
            ..IncrementalConfig::default()
        });
        assert_eq!(s.strategy_for_char_count(3), SearchStrategy::HybridFast);
    }

    #[test]
    fn strategy_5_plus_chars_is_hybrid_fast() {
        let s = default_searcher();
        assert_eq!(s.strategy_for_char_count(5), SearchStrategy::HybridFast);
        assert_eq!(s.strategy_for_char_count(10), SearchStrategy::HybridFast);
        assert_eq!(s.strategy_for_char_count(100), SearchStrategy::HybridFast);
    }

    // ── Prefix extension detection ──────────────────────────────────

    #[test]
    fn prefix_extension_detected() {
        let mut s = default_searcher();
        s.update("sea", vec!["doc-1".into(), "doc-2".into()]);
        assert!(s.is_prefix_extension("sear"));
        assert!(s.is_prefix_extension("search"));
    }

    #[test]
    fn non_prefix_query_not_detected() {
        let mut s = default_searcher();
        s.update("sea", vec!["doc-1".into()]);
        assert!(!s.is_prefix_extension("mountain"));
    }

    #[test]
    fn backspace_not_prefix_extension() {
        let mut s = default_searcher();
        s.update("search", vec!["doc-1".into()]);
        assert!(!s.is_prefix_extension("searc"));
    }

    #[test]
    fn same_query_not_prefix_extension() {
        let mut s = default_searcher();
        s.update("sea", vec!["doc-1".into()]);
        assert!(!s.is_prefix_extension("sea"));
    }

    #[test]
    fn no_previous_query_not_prefix_extension() {
        let s = default_searcher();
        assert!(!s.is_prefix_extension("anything"));
    }

    #[test]
    fn unicode_prefix_extension() {
        let mut s = default_searcher();
        s.update("caf\u{00e9}", vec!["doc-1".into()]);
        assert!(s.is_prefix_extension("caf\u{00e9}s"));
    }

    // ── Candidate pool reuse ────────────────────────────────────────

    #[test]
    fn prefix_extension_reuses_candidate_pool() {
        let mut s = default_searcher();
        s.update("sea", vec!["doc-1".into(), "doc-2".into(), "doc-3".into()]);

        let plan = s.plan("sear");
        assert!(plan.reuse_candidates);
        assert_eq!(plan.candidate_doc_ids.len(), 3);
    }

    #[test]
    fn non_prefix_does_not_reuse_pool() {
        let mut s = default_searcher();
        s.update("sea", vec!["doc-1".into()]);

        let plan = s.plan("mountain");
        assert!(!plan.reuse_candidates);
        assert!(plan.candidate_doc_ids.is_empty());
    }

    #[test]
    fn empty_results_no_reuse() {
        let mut s = default_searcher();
        s.update("sea", Vec::new());

        let plan = s.plan("sear");
        assert!(!plan.reuse_candidates);
    }

    // ── Candidate pool size limit ───────────────────────────────────

    #[test]
    fn candidate_pool_size_limit() {
        let mut s = IncrementalSearcher::new(IncrementalConfig {
            candidate_pool_size: 3,
            ..IncrementalConfig::default()
        });

        let results: Vec<String> = (0..10).map(|i| format!("doc-{i}")).collect();
        s.update("test", results);

        // Only 3 should be kept.
        let plan = s.plan("testi");
        assert!(plan.reuse_candidates);
        assert_eq!(plan.candidate_doc_ids.len(), 3);
    }

    // ── should_refine ───────────────────────────────────────────────

    #[test]
    fn should_refine_below_threshold() {
        let s = default_searcher();
        assert!(!s.should_refine(Duration::from_millis(200)));
    }

    #[test]
    fn should_refine_at_threshold() {
        let s = default_searcher();
        assert!(s.should_refine(Duration::from_millis(300)));
    }

    #[test]
    fn should_refine_above_threshold() {
        let s = default_searcher();
        assert!(s.should_refine(Duration::from_millis(1000)));
    }

    // ── Reset clears state ──────────────────────────────────────────

    #[test]
    fn reset_clears_all_state() {
        let mut s = default_searcher();
        s.update("test", vec!["doc-1".into()]);
        assert!(s.has_previous_results());

        s.reset();
        assert!(!s.has_previous_results());
        assert!(s.last_query().is_none());

        // After reset, no prefix extension detection.
        let plan = s.plan("testing");
        assert!(!plan.reuse_candidates);
    }

    // ── Empty query after non-empty resets ───────────────────────────

    #[test]
    fn empty_query_after_previous_is_skip() {
        let mut s = default_searcher();
        s.update("search", vec!["doc-1".into()]);

        let plan = s.plan("");
        assert_eq!(plan.strategy, SearchStrategy::Skip);
        assert!(!plan.reuse_candidates);
    }

    // ── Query shrinks (backspace) ───────────────────────────────────

    #[test]
    fn query_shrink_does_not_reuse_pool() {
        let mut s = default_searcher();
        s.update("search", vec!["doc-1".into(), "doc-2".into()]);

        // User hits backspace: "search" -> "searc"
        let plan = s.plan("searc");
        assert!(!plan.reuse_candidates, "backspace should not reuse pool");
        assert!(plan.candidate_doc_ids.is_empty());
        // But should still search (5 chars >= min_prefix_len).
        assert_eq!(plan.strategy, SearchStrategy::HybridFast);
    }

    // ── Complete query change ───────────────────────────────────────

    #[test]
    fn completely_different_query_does_full_search() {
        let mut s = default_searcher();
        s.update("ocean", vec!["doc-1".into()]);

        let plan = s.plan("mountain");
        assert!(!plan.reuse_candidates);
        assert!(plan.candidate_doc_ids.is_empty());
        assert_eq!(plan.strategy, SearchStrategy::HybridFast);
    }

    // ── Config serde roundtrip ──────────────────────────────────────

    #[test]
    fn config_serde_roundtrip() {
        let config = IncrementalConfig {
            min_prefix_len: 3,
            use_hash_embedder_for_short: false,
            refine_after_pause_ms: 500,
            candidate_pool_size: 50,
        };
        let json = serde_json::to_string(&config).unwrap();
        let decoded: IncrementalConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.min_prefix_len, 3);
        assert!(!decoded.use_hash_embedder_for_short);
        assert_eq!(decoded.refine_after_pause_ms, 500);
        assert_eq!(decoded.candidate_pool_size, 50);
    }

    // ── Strategy serde roundtrip ────────────────────────────────────

    #[test]
    fn strategy_serde_roundtrip() {
        let strategies = [
            SearchStrategy::Skip,
            SearchStrategy::LexicalPrefixOnly,
            SearchStrategy::LexicalPlusHash,
            SearchStrategy::HybridFast,
            SearchStrategy::FullTwoTier,
        ];
        for strategy in &strategies {
            let json = serde_json::to_string(strategy).unwrap();
            let decoded: SearchStrategy = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, strategy);
        }
    }

    // ── Debug format ────────────────────────────────────────────────

    #[test]
    fn debug_format() {
        let s = default_searcher();
        let debug = format!("{s:?}");
        assert!(debug.contains("IncrementalSearcher"));
        assert!(debug.contains("cached_results"));
    }

    // ── Simulate typing sequence ────────────────────────────────────

    #[test]
    fn typing_sequence_simulation() {
        let mut s = default_searcher();

        // Type "s" - too short
        let plan = s.plan("s");
        assert_eq!(plan.strategy, SearchStrategy::Skip);

        // Type "se" - reaches min_prefix_len
        let plan = s.plan("se");
        assert_eq!(plan.strategy, SearchStrategy::LexicalPrefixOnly);
        assert!(!plan.reuse_candidates);
        s.update("se", vec!["sea".into(), "search".into(), "send".into()]);

        // Type "sea" - prefix extension
        let plan = s.plan("sea");
        assert_eq!(plan.strategy, SearchStrategy::LexicalPlusHash);
        assert!(plan.reuse_candidates);
        assert_eq!(plan.candidate_doc_ids.len(), 3);
        s.update("sea", vec!["sea".into(), "search".into()]);

        // Type "sear" - still prefix extension
        let plan = s.plan("sear");
        assert!(plan.reuse_candidates);
        assert_eq!(plan.candidate_doc_ids.len(), 2);
        s.update("sear", vec!["search".into()]);

        // Type "search" - 6 chars, hybrid fast
        let plan = s.plan("search");
        assert_eq!(plan.strategy, SearchStrategy::HybridFast);
        assert!(plan.reuse_candidates);
        assert_eq!(plan.candidate_doc_ids.len(), 1);
    }

    // ── Rapid sequential queries (no state corruption) ──────────────

    #[test]
    fn rapid_sequential_queries_no_corruption() {
        let mut s = default_searcher();

        // Simulate rapid typing of "hello world".
        let sequence = ["he", "hel", "hell", "hello", "hello ", "hello w", "hello wo",
            "hello wor", "hello worl", "hello world"];

        for (i, query) in sequence.iter().enumerate() {
            let plan = s.plan(query);
            assert_ne!(
                plan.strategy,
                SearchStrategy::Skip,
                "query '{query}' should not be skipped"
            );

            // After first query, prefix extensions should be detected.
            if i > 0 {
                assert!(
                    plan.reuse_candidates,
                    "query '{query}' should reuse candidates from '{}'",
                    s.last_query().unwrap_or("none")
                );
            }

            let results: Vec<String> = (0..5).map(|j| format!("doc-{j}")).collect();
            s.update(query, results);
        }

        assert_eq!(s.last_query(), Some("hello world"));
    }
}
