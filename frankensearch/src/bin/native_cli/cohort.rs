//! Executable-only dispatch over admitted single and partitioned hybrid owners.
//!
//! Retrieval, source scopes, progressive state and reranking stay in the native
//! library. This adapter only selects those APIs and serializes their complete
//! result envelopes for the existing buffered/streaming command protocol.

use frankensearch::native_ann::builder::scope::sharded::NativeScopedShardedHybridIndex;
use frankensearch::native_ann::builder::sharded::NativeBuiltShardedHybridIndex;
use frankensearch::native_ann::{
    NativePhaseCandidates, NativeProgressiveSearch, NativeSearchPhase,
    NativeShardedProgressiveSearch, NativeShardedResult, NativeShardedSearchPhase,
};
use frankensearch::{Reranker, ScoredResult, SearchError, SearchResult};

use super::{
    ArtifactGenerationIdentityV1, Cx, Mode, NativeBuiltHybridIndex, Result, SCHEMA, Serialize,
    filter, validate_query,
};

/// Borrow one completed cohort. A live snapshot must outlive this reference and
/// every phase; layout dispatch never resolves a path or refreshes a component.
#[derive(Clone, Copy)]
pub enum Index<'a> {
    Single(&'a NativeBuiltHybridIndex),
    Sharded(&'a NativeBuiltShardedHybridIndex),
}

impl<'a> From<&'a NativeBuiltHybridIndex> for Index<'a> {
    fn from(index: &'a NativeBuiltHybridIndex) -> Self {
        Self::Single(index)
    }
}

impl<'a> From<&'a NativeBuiltShardedHybridIndex> for Index<'a> {
    fn from(index: &'a NativeBuiltShardedHybridIndex) -> Self {
        Self::Sharded(index)
    }
}

impl<'a> Index<'a> {
    fn first(self) -> &'a frankensearch::native_ann::builder::NativeBuiltIndex {
        match self {
            Self::Single(index) => index.vectors(),
            // Native shard admission requires an identity-bearing first
            // partition, including an explicitly empty complete cohort.
            Self::Sharded(index) => &index.vectors().partitions()[0],
        }
    }

    pub fn document_count(self) -> usize {
        match self {
            Self::Single(index) => index.vectors().documents().len(),
            Self::Sharded(index) => index.vectors().document_count(),
        }
    }

    pub fn has_quality(self) -> bool {
        self.first().quality().is_some()
    }

    pub fn producers(self) -> SearchResult<(String, Option<String>)> {
        let first = self.first();
        Ok((
            first.fast().embedder().identity()?.fingerprint(),
            first
                .quality()
                .map(|tier| {
                    tier.embedder()
                        .identity()
                        .map(|identity| identity.fingerprint())
                })
                .transpose()?,
        ))
    }

    pub fn all_native_hnsw(self, quality: bool) -> bool {
        let present = |partition: &frankensearch::native_ann::builder::NativeBuiltIndex| {
            if quality {
                partition
                    .quality()
                    .is_some_and(|tier| tier.graph_path().is_some())
            } else {
                partition.fast().graph_path().is_some()
            }
        };
        match self {
            Self::Single(index) => present(index.vectors()),
            Self::Sharded(index) => index.vectors().partitions().iter().all(present),
        }
    }

    pub fn generation(self) -> ArtifactGenerationIdentityV1 {
        match self {
            Self::Single(index) => index.vectors().fast().index().owner_witness().generation,
            Self::Sharded(index) => index.vectors().generation(),
        }
    }

    pub fn annotate(self, payload: &mut serde_json::Value) {
        if let Self::Sharded(index) = self {
            payload["layout"] = serde_json::json!("sharded");
            payload["partitions"] = serde_json::json!(index.vectors().partitions().len());
        }
    }
}

pub enum Prepared<'a> {
    Single(filter::Query<'a>),
    Sharded {
        index: &'a NativeBuiltShardedHybridIndex,
        scope: Option<NativeScopedShardedHybridIndex<'a>>,
    },
}

impl<'a> Prepared<'a> {
    pub fn prepare(index: Index<'a>, cx: &Cx, filters: filter::Filters<'_>) -> Result<Self> {
        let index = match index {
            Index::Single(index) => {
                return Ok(Self::Single(filter::Query::prepare(index, cx, filters)?));
            }
            Index::Sharded(index) => index,
        };
        cx.checkpoint().map_err(|error| SearchError::Cancelled {
            phase: "native_cli.sharded_scope".to_owned(),
            reason: error.to_string(),
        })?;
        for filter in filters.iter().flatten() {
            filter.validate()?;
        }
        let scope = if filters.iter().any(Option::is_some) {
            Some(index.scope(cx, |document| {
                Ok(filters
                    .iter()
                    .flatten()
                    .all(|filter| filter.matches(document)))
            })?)
        } else {
            // Do not scan source membership for an unfiltered query.
            None
        };
        Ok(Self::Sharded { index, scope })
    }

    pub fn annotate(&self, payload: &mut serde_json::Value) {
        match self {
            Self::Single(query) => query.annotate(payload),
            Self::Sharded { index, scope } => {
                Index::Sharded(index).annotate(payload);
                if let Some(scope) = scope {
                    payload["scope"] = serde_json::json!({
                        "filtered": true, "eligible_documents": scope.len(),
                    });
                }
            }
        }
    }

    pub async fn search(
        &self,
        cx: &Cx,
        text: &str,
        mode: Mode,
        limit: usize,
    ) -> Result<serde_json::Value> {
        let (index, scope) = match self {
            Self::Single(query) => return query.search(cx, text, mode, limit).await,
            Self::Sharded { index, scope } => (index, scope),
        };
        validate_query(text)?;
        let full = mode == Mode::Full && index.vectors().quality().is_some();
        let results = match (scope, mode, full) {
            (Some(scope), Mode::Quality, _) => scope.search_quality(cx, text, limit).await?,
            (Some(scope), _, true) => scope.search_refined(cx, text, limit).await?,
            (Some(scope), _, false) => scope.search(cx, text, limit).await?,
            (None, Mode::Quality, _) => index.search_quality(cx, text, limit).await?,
            (None, _, true) => index.search_refined(cx, text, limit).await?,
            (None, _, false) => index.search(cx, text, limit).await?,
        };
        let phase = if mode == Mode::Quality {
            "quality"
        } else if full {
            "refined"
        } else {
            "initial"
        };
        let mut payload = serde_json::json!({
            "schema": SCHEMA, "ok": true, "event": "results", "phase": phase,
            "generation": index.vectors().generation(), "results": Rows::from(results),
        });
        self.annotate(&mut payload);
        Ok(payload)
    }

    pub fn progressive<'q>(
        &'q self,
        cx: &'q Cx,
        text: &'q str,
        limit: usize,
    ) -> SearchResult<Progressive<'q>> {
        match self {
            Self::Single(query) => Ok(Progressive::Single(Box::new(
                query.progressive(cx, text, limit)?,
            ))),
            Self::Sharded { index, scope } => {
                let stream = scope.as_ref().map_or_else(
                    || index.progressive(cx, text, limit),
                    |scope| scope.progressive(cx, text, limit),
                )?;
                Ok(Progressive::Sharded(Box::new(stream)))
            }
        }
    }

    pub fn progressive_with_reranker<'q>(
        &'q self,
        cx: &'q Cx,
        text: &'q str,
        limit: usize,
        reranker: &'q dyn Reranker,
        window: usize,
    ) -> SearchResult<Progressive<'q>> {
        match self {
            Self::Single(query) => Ok(Progressive::Single(Box::new(
                query.progressive_with_reranker(cx, text, limit, reranker, window)?,
            ))),
            Self::Sharded { index, scope } => {
                let stream = scope.as_ref().map_or_else(
                    || index.progressive_with_reranker(cx, text, limit, reranker, window),
                    |scope| scope.progressive_with_reranker(cx, text, limit, reranker, window),
                )?;
                Ok(Progressive::Sharded(Box::new(stream)))
            }
        }
    }
}

/// Keep the two actual native state machines, not reconstructed default queries.
pub enum Progressive<'a> {
    Single(Box<NativeProgressiveSearch<'a>>),
    Sharded(Box<NativeShardedProgressiveSearch<'a>>),
}

impl Progressive<'_> {
    pub fn is_finished(&self) -> bool {
        match self {
            Self::Single(stream) => stream.is_finished(),
            Self::Sharded(stream) => stream.is_finished(),
        }
    }

    pub async fn next_phase(&mut self) -> SearchResult<Option<Phase>> {
        match self {
            Self::Single(stream) => Ok(stream.next_phase().await?.map(Phase::from)),
            Self::Sharded(stream) => Ok(stream.next_phase().await?.map(Phase::from)),
        }
    }
}

/// Only output representation differs. Never interpret a shard-local row as a
/// bare result index, or reorder the score/metadata independently of its rows.
#[derive(Serialize)]
#[serde(untagged)]
pub enum Rows {
    Single(Vec<ScoredResult>),
    Sharded(Vec<ShardedOutput>),
}

#[derive(Serialize)]
pub struct ShardedOutput {
    #[serde(flatten)]
    result: ScoredResult,
    fast_row: Option<OutputRow>,
    quality_row: Option<OutputRow>,
}

#[derive(Serialize)]
struct OutputRow {
    shard: usize,
    physical_row: u32,
}

impl From<Vec<NativeShardedResult>> for Rows {
    fn from(hits: Vec<NativeShardedResult>) -> Self {
        let row = |row: frankensearch::native_ann::NativeShardRow| OutputRow {
            shard: row.shard,
            physical_row: row.physical_row,
        };
        Self::Sharded(
            hits.into_iter()
                .map(|hit| ShardedOutput {
                    result: hit.result,
                    fast_row: hit.fast_row.map(row),
                    quality_row: hit.quality_row.map(row),
                })
                .collect(),
        )
    }
}

pub enum Phase {
    Initial {
        results: Rows,
        candidates: NativePhaseCandidates,
    },
    Refined {
        results: Rows,
        candidates: NativePhaseCandidates,
    },
    Reranked {
        results: Rows,
        candidates: NativePhaseCandidates,
        evaluated: usize,
    },
    RefinementFailed {
        initial_results: Rows,
        error: SearchError,
    },
    RerankFailed {
        previous_results: Rows,
        error: SearchError,
    },
}

impl From<NativeSearchPhase> for Phase {
    fn from(phase: NativeSearchPhase) -> Self {
        match phase {
            NativeSearchPhase::Initial {
                results,
                candidates,
            } => Self::Initial {
                results: Rows::Single(results),
                candidates,
            },
            NativeSearchPhase::Refined {
                results,
                candidates,
            } => Self::Refined {
                results: Rows::Single(results),
                candidates,
            },
            NativeSearchPhase::Reranked {
                results,
                candidates,
                evaluated,
            } => Self::Reranked {
                results: Rows::Single(results),
                candidates,
                evaluated,
            },
            NativeSearchPhase::RefinementFailed {
                initial_results,
                error,
            } => Self::RefinementFailed {
                initial_results: Rows::Single(initial_results),
                error,
            },
            NativeSearchPhase::RerankFailed {
                previous_results,
                error,
            } => Self::RerankFailed {
                previous_results: Rows::Single(previous_results),
                error,
            },
        }
    }
}

impl From<NativeShardedSearchPhase> for Phase {
    fn from(phase: NativeShardedSearchPhase) -> Self {
        match phase {
            NativeShardedSearchPhase::Initial {
                results,
                candidates,
            } => Self::Initial {
                results: results.into(),
                candidates,
            },
            NativeShardedSearchPhase::Refined {
                results,
                candidates,
            } => Self::Refined {
                results: results.into(),
                candidates,
            },
            NativeShardedSearchPhase::Reranked {
                results,
                candidates,
                evaluated,
            } => Self::Reranked {
                results: results.into(),
                candidates,
                evaluated,
            },
            NativeShardedSearchPhase::RefinementFailed {
                initial_results,
                error,
            } => Self::RefinementFailed {
                initial_results: initial_results.into(),
                error,
            },
            NativeShardedSearchPhase::RerankFailed {
                previous_results,
                error,
            } => Self::RerankFailed {
                previous_results: previous_results.into(),
                error,
            },
        }
    }
}
