//! Graph-ranking phase-1 hook (feature-gated stub).
//!
//! The concrete franken_networkx-powered implementation lands in a follow-up.

use asupersync::Cx;
use frankensearch_core::types::ScoredResult;

/// Graph-ranking engine placeholder for phase-1 wiring.
#[derive(Debug, Clone, Copy, Default)]
pub struct GraphRanker;

impl GraphRanker {
    /// Construct a no-op graph ranker.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Compute graph-ranked candidates for the current query.
    ///
    /// This stub intentionally returns `None` until graph algorithms are wired.
    #[must_use]
    pub fn rank_phase1(&self, _cx: &Cx, _query: &str, _limit: usize) -> Option<Vec<ScoredResult>> {
        None
    }
}
