//! Explicit exact retrieval when an optional native graph cannot be admitted.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::io::ErrorKind;
use std::path::Path;
use std::sync::Arc;

use frankensearch_core::BoundQueryEmbedding;
use frankensearch_index::ValidatedFsviBytes;
use frankensearch_index::native_hnsw::ValidatedNativeHnsw;

use super::{NativeAnnIndex, checkpoint, invalid};
use crate::{Cx, SearchError, SearchResult, VectorHit};

/// Why a retained owner is searched exactly rather than through a native graph.
///
/// These bounded codes contain no paths, queries, document IDs or raw errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeExactReason {
    /// The caller selected exact search without trying to open a graph.
    Requested,
    /// The optional graph or its receipt was not found.
    SidecarMissing,
    /// I/O prevented the optional graph or receipt from being read.
    SidecarUnreadable,
    /// Graph or receipt admission rejected corrupt or stale sidecar data.
    SidecarRejected,
}

/// The actual retrieval backend, independent of the embedding's semantic kind.
///
/// Exact versus approximate is not fast versus quality, or hash versus semantic.
/// The same embedding identity is enforced in both execution modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeRetrievalMode {
    /// An admitted native graph performs candidate retrieval.
    Ann,
    /// The original admitted FSVI owner is scanned without a graph.
    Exact {
        /// Explicit selection or the optional-sidecar failure that selected it.
        reason: NativeExactReason,
    },
}

#[derive(Debug)]
pub(super) enum NativeBackend {
    Ann(Box<ValidatedNativeHnsw>),
    Exact(NativeExactReason),
}

impl NativeAnnIndex {
    /// Search an already-admitted FSVI owner exactly, without building a graph.
    ///
    /// Retains the same immutable owner used by native ANN. No filesystem path
    /// is opened, no sidecar is created, and no vector is relabelled or copied
    /// into another index. All text, hybrid, quality and progressive APIs work
    /// with this handle through the same query identity admission boundary.
    ///
    /// # Errors
    ///
    /// Returns cancellation before constructing the retrieval handle.
    pub fn exact(cx: &Cx, owner: Arc<ValidatedFsviBytes>) -> SearchResult<Self> {
        checkpoint(cx, "native_ann.exact_open")?;
        Ok(Self::exact_with_reason(owner, NativeExactReason::Requested))
    }

    /// Load an optional native graph, or explicitly recover to exact retrieval.
    ///
    /// Unlike [`Self::load`], this method permits missing, unreadable, corrupt,
    /// or stale optional sidecars. It NEVER reacquires the FSVI from a pathname:
    /// exact search retains the supplied already-admitted owner. No files are
    /// repaired, replaced, deleted, or rebuilt. Inspect [`Self::retrieval_mode`]
    /// to report the realized backend and its bounded fallback reason.
    ///
    /// Cancellation, invalid configuration, and all other non-sidecar error
    /// classes still fail. In particular this is not an inference fallback and
    /// cannot turn a foreign query identity into an admissible one. Subsequent
    /// search errors propagate without retrying another backend.
    ///
    /// # Errors
    ///
    /// Propagates cancellation and errors outside the explicit sidecar recovery
    /// classes. The strict [`Self::load`] behavior remains unchanged.
    pub fn load_or_exact(
        cx: &Cx,
        owner: Arc<ValidatedFsviBytes>,
        graph_path: &Path,
    ) -> SearchResult<Self> {
        let loaded = Self::load(cx, Arc::clone(&owner), graph_path);
        // An error in load skips its final checkpoint. Cancellation must still
        // dominate a coincident sidecar failure rather than selecting fallback.
        checkpoint(cx, "native_ann.optional_load_complete")?;
        match loaded {
            Ok(index) => Ok(index),
            Err(error) => recovery_reason(&error).map_or_else(
                || Err(error),
                |reason| Ok(Self::exact_with_reason(owner, reason)),
            ),
        }
    }

    /// Realized retrieval backend. Does not claim semantic or corpus coverage.
    #[must_use]
    pub const fn retrieval_mode(&self) -> NativeRetrievalMode {
        match &self.graph {
            NativeBackend::Ann(_) => NativeRetrievalMode::Ann,
            NativeBackend::Exact(reason) => NativeRetrievalMode::Exact { reason: *reason },
        }
    }

    fn exact_with_reason(owner: Arc<ValidatedFsviBytes>, reason: NativeExactReason) -> Self {
        Self {
            owner,
            graph: NativeBackend::Exact(reason),
            // Only the ANN arm reads this value; exact scans ignore ef entirely.
            default_ef_search: 1,
        }
    }

    /// Called only after the shared query identity/dimension admission.
    pub(super) fn search_exact_filtered<F>(
        &self,
        cx: &Cx,
        query: &BoundQueryEmbedding,
        target: usize,
        accept: &F,
    ) -> SearchResult<Vec<VectorHit>>
    where
        F: Fn(&str) -> bool,
    {
        checkpoint(cx, "native_ann.exact_search")?;
        if target == 0 {
            return Ok(Vec::new());
        }
        // Keep only k winners, not an all-corpus hit vector. Decode at most one
        // row at a time, and clone a document ID only when it enters the heap.
        let mut heap: BinaryHeap<RankedHit> = BinaryHeap::new();
        for physical in 0..self.owner.record_count() {
            checkpoint(cx, "native_ann.exact_row")?;
            let row = self.owner.row(physical)?;
            if !row.flags().is_live() || !accept(row.doc_id()) {
                continue;
            }
            checkpoint(cx, "native_ann.exact_score")?;
            let values = self.owner.vector_at_f32(physical)?;
            // Use exactly the ANN facade's retained-owner rescoring kernel,
            // including its f16-to-f32 promotion, rather than rounded distance.
            let score = frankensearch_index::dot_product_f32_f32(&values, query.vector())?;
            if !score.is_finite() {
                return Err(invalid(
                    "score",
                    "non-finite",
                    "retained-owner dot product must be finite",
                ));
            }
            if heap.len() == target
                && heap.peek().is_some_and(|worst| {
                    worst
                        .0
                        .score
                        .total_cmp(&score)
                        .then_with(|| row.doc_id().cmp(worst.0.doc_id.as_str()))
                        != Ordering::Less
                })
            {
                continue;
            }
            let hit = RankedHit(VectorHit {
                index: u32::try_from(physical).map_err(|_| {
                    invalid(
                        "physical_row",
                        "out-of-range",
                        "source row does not fit u32",
                    )
                })?,
                score,
                doc_id: row.doc_id().into(),
            });
            if heap.len() == target {
                let _ = heap.pop();
            }
            heap.push(hit);
        }
        let mut hits: Vec<_> = heap.into_iter().map(|hit| hit.0).collect();
        hits.sort_unstable_by(VectorHit::cmp_rank);
        checkpoint(cx, "native_ann.exact_complete")?;
        Ok(hits)
    }
}

fn recovery_reason(error: &SearchError) -> Option<NativeExactReason> {
    match error {
        SearchError::IndexNotFound { .. } => Some(NativeExactReason::SidecarMissing),
        SearchError::Io(error) if error.kind() == ErrorKind::NotFound => {
            Some(NativeExactReason::SidecarMissing)
        }
        SearchError::Io(_) => Some(NativeExactReason::SidecarUnreadable),
        SearchError::IndexCorrupted { .. } => Some(NativeExactReason::SidecarRejected),
        _ => None,
    }
}

/// `cmp_rank` orders better hits first, so the max-heap exposes its WORST hit.
struct RankedHit(VectorHit);

impl PartialEq for RankedHit {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for RankedHit {}
impl PartialOrd for RankedHit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for RankedHit {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp_rank(&other.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    use frankensearch_core::generation::{
        ArtifactGenerationIdentityV1, EmbeddingIdentityBundleV1, QuantizationFormat,
    };
    use frankensearch_core::traits::{IdentityBoundEmbedding, ModelCategory, SearchFuture};
    use frankensearch_index::native_hnsw::HnswParams;
    use frankensearch_index::{FsviV2IdentityBinding, VectorIndex};

    use crate::native_ann::NativeSearchPhase;
    use crate::{Embedder, LexicalRead, ScoredResult};

    fn identity() -> EmbeddingIdentityBundleV1 {
        EmbeddingIdentityBundleV1::explicit_test_model("native-exact-fallback", 2)
    }

    fn query(values: [f32; 2]) -> BoundQueryEmbedding {
        BoundQueryEmbedding::new(values.to_vec(), identity()).unwrap()
    }

    fn owner(
        format: QuantizationFormat,
        generation: u64,
        rows: &[(&str, [f32; 2], bool)],
    ) -> Arc<ValidatedFsviBytes> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("owner.fsvi");
        let mut identity = identity();
        identity.storage.format = "fsvi-v2".to_owned();
        identity.storage.quantization = format;
        identity.storage.endianness = "little-endian".to_owned();
        let binding = FsviV2IdentityBinding::new(
            ArtifactGenerationIdentityV1::new(generation, [0x51; 16]).unwrap(),
            identity.freeze().unwrap(),
        )
        .unwrap();
        let mut writer = VectorIndex::create_v2(&path, binding.clone()).unwrap();
        for &(id, vector, live) in rows {
            if live {
                writer.write_record(id, &vector).unwrap();
            } else {
                writer.write_tombstone_record(id, &vector).unwrap();
            }
        }
        writer.finish().unwrap();
        let bytes: Arc<[u8]> = std::fs::read(path).unwrap().into();
        Arc::new(ValidatedFsviBytes::from_arc(bytes, &binding).unwrap())
    }

    fn rows() -> Vec<(&'static str, [f32; 2], bool)> {
        vec![
            ("dead", [1.0, 0.0], false),
            ("z-tie", [0.8, 0.2], true),
            ("a-tie", [0.8, 0.2], true),
            ("vertical", [0.0, 1.0], true),
            ("negative", [-1.0, 0.0], true),
        ]
    }

    #[test]
    fn exact_heap_matches_all_row_oracle_and_full_beam_for_f16_and_f32() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for format in [QuantizationFormat::F16, QuantizationFormat::F32] {
                let owner = owner(format, 1, &rows());
                let exact = NativeAnnIndex::exact(&cx, Arc::clone(&owner)).unwrap();
                let ann = NativeAnnIndex::build(&cx, Arc::clone(&owner), HnswParams::default(), 7)
                    .unwrap();
                for values in [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]] {
                    let query = query(values);
                    let mut oracle = owner
                        .search_top_k(query.vector(), owner.live_count(), None)
                        .unwrap();
                    oracle.sort_unstable_by(VectorHit::cmp_rank);
                    assert_eq!(
                        ann.search(&cx, &query, usize::MAX, Some(owner.record_count()))
                            .unwrap(),
                        oracle
                    );
                    for k in [0, 1, 2, 3, 4, usize::MAX] {
                        let expected = &oracle[..k.min(oracle.len())];
                        for ef in [None, Some(0), Some(1), Some(usize::MAX)] {
                            assert_eq!(exact.search(&cx, &query, k, ef).unwrap(), expected);
                        }
                    }
                }
                assert_eq!(exact.owner_witness(), owner.witness());
                assert_eq!(
                    exact.retrieval_mode(),
                    NativeRetrievalMode::Exact {
                        reason: NativeExactReason::Requested,
                    }
                );
            }
        });
    }

    #[test]
    fn exact_filters_are_applied_before_top_k_without_a_sync_bound() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let owner = owner(QuantizationFormat::F32, 1, &rows());
            let exact = NativeAnnIndex::exact(&cx, owner).unwrap();
            let calls = Cell::new(0);
            let hits = exact
                .search_filtered(&cx, &query([1.0, 0.0]), 2, Some(1), |id| {
                    calls.set(calls.get() + 1);
                    matches!(id, "vertical" | "negative")
                })
                .unwrap();
            assert_eq!(
                hits.iter()
                    .map(|hit| hit.doc_id.as_str())
                    .collect::<Vec<_>>(),
                ["vertical", "negative"]
            );
            assert_eq!(calls.get(), 4); // Tombstones never reach the predicate.
            assert!(
                exact
                    .search_filtered(&cx, &query([1.0, 0.0]), 2, None, |_| false)
                    .unwrap()
                    .is_empty()
            );
            let tied = exact.search(&cx, &query([1.0, 0.0]), 1, None).unwrap();
            assert_eq!(tied[0].doc_id, "a-tie");
            assert_eq!(tied[0].index, 2);
        });
    }

    #[test]
    fn missing_optional_sidecar_recovers_but_strict_load_still_fails() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().canonicalize().unwrap().join("missing.fshnsw");
            let owner = owner(QuantizationFormat::F32, 1, &rows());
            assert!(NativeAnnIndex::load(&cx, Arc::clone(&owner), &path).is_err());
            let weak = Arc::downgrade(&owner);
            let recovered = NativeAnnIndex::load_or_exact(&cx, Arc::clone(&owner), &path).unwrap();
            assert_eq!(
                recovered.retrieval_mode(),
                NativeRetrievalMode::Exact {
                    reason: NativeExactReason::SidecarMissing,
                }
            );
            assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
            drop(owner);
            drop(dir);
            assert!(weak.upgrade().is_some());
            assert_eq!(
                recovered.search(&cx, &query([1.0, 0.0]), 1, None).unwrap()[0].doc_id,
                "a-tie"
            );
            drop(recovered);
            assert!(weak.upgrade().is_none());
        });
    }

    #[test]
    fn optional_load_preserves_a_valid_native_graph() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().canonicalize().unwrap().join("valid.fshnsw");
            let owner = owner(QuantizationFormat::F32, 1, &rows());
            let original =
                NativeAnnIndex::build(&cx, Arc::clone(&owner), HnswParams::default(), 7).unwrap();
            original.save(&cx, &path).unwrap();
            let graph_before = std::fs::read(&path).unwrap();
            let receipt_path = dir.path().join("valid.fshnsw.receipt");
            let receipt_before = std::fs::read(&receipt_path).unwrap();
            let loaded = NativeAnnIndex::load_or_exact(&cx, owner, &path).unwrap();
            assert_eq!(loaded.retrieval_mode(), NativeRetrievalMode::Ann);
            let query = query([1.0, 0.0]);
            assert_eq!(
                loaded.search(&cx, &query, 3, None).unwrap(),
                original.search(&cx, &query, 3, None).unwrap()
            );
            assert_eq!(std::fs::read(&path).unwrap(), graph_before);
            assert_eq!(std::fs::read(receipt_path).unwrap(), receipt_before);
        });
    }

    #[test]
    fn stale_generation_recovers_to_the_supplied_owner_not_the_sidecars_owner() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().canonicalize().unwrap().join("stale.fshnsw");
            let old = owner(QuantizationFormat::F32, 1, &rows());
            NativeAnnIndex::build(&cx, old, HnswParams::default(), 7)
                .unwrap()
                .save(&cx, &path)
                .unwrap();
            let receipt_path = dir.path().join("stale.fshnsw.receipt");
            let before = (
                std::fs::read(&path).unwrap(),
                std::fs::read(&receipt_path).unwrap(),
            );
            // Same IDs and dimensions, but another generation and changed vectors.
            let mut successor_rows = rows();
            successor_rows[1].1 = [0.0, 1.0];
            successor_rows[2].1 = [0.0, 1.0];
            successor_rows[3].1 = [1.0, 0.0];
            let successor = owner(QuantizationFormat::F32, 2, &successor_rows);
            assert!(NativeAnnIndex::load(&cx, Arc::clone(&successor), &path).is_err());
            let recovered =
                NativeAnnIndex::load_or_exact(&cx, Arc::clone(&successor), &path).unwrap();
            assert_eq!(
                recovered.retrieval_mode(),
                NativeRetrievalMode::Exact {
                    reason: NativeExactReason::SidecarRejected,
                }
            );
            assert_eq!(recovered.owner_witness(), successor.witness());
            assert_eq!(
                recovered.search(&cx, &query([1.0, 0.0]), 1, None).unwrap()[0].doc_id,
                "vertical"
            );
            assert_eq!(
                (
                    std::fs::read(&path).unwrap(),
                    std::fs::read(receipt_path).unwrap()
                ),
                before
            );
        });
    }

    #[test]
    fn corrupt_graph_is_not_repaired_or_replaced_by_optional_loading() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().canonicalize().unwrap().join("broken.fshnsw");
            let owner = owner(QuantizationFormat::F32, 1, &rows());
            NativeAnnIndex::build(&cx, Arc::clone(&owner), HnswParams::default(), 7)
                .unwrap()
                .save(&cx, &path)
                .unwrap();
            std::fs::write(&path, b"broken").unwrap();
            let receipt_path = dir.path().join("broken.fshnsw.receipt");
            let receipt = std::fs::read(&receipt_path).unwrap();
            let recovered = NativeAnnIndex::load_or_exact(&cx, owner, &path).unwrap();
            assert_eq!(
                recovered.retrieval_mode(),
                NativeRetrievalMode::Exact {
                    reason: NativeExactReason::SidecarRejected,
                }
            );
            assert_eq!(
                recovered.search(&cx, &query([1.0, 0.0]), 1, None).unwrap()[0].doc_id,
                "a-tie"
            );
            assert_eq!(std::fs::read(path).unwrap(), b"broken");
            assert_eq!(std::fs::read(receipt_path).unwrap(), receipt);
        });
    }

    #[test]
    fn exact_identity_admission_precedes_zero_k_and_all_predicate_calls() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let exact =
                NativeAnnIndex::exact(&cx, owner(QuantizationFormat::F32, 1, &rows())).unwrap();
            let foreign = BoundQueryEmbedding::new(
                vec![1.0, 0.0],
                EmbeddingIdentityBundleV1::explicit_test_model("foreign", 2),
            )
            .unwrap();
            let calls = Cell::new(0);
            for k in [0, 1] {
                let error = exact
                    .search_filtered(&cx, &foreign, k, None, |_| {
                        calls.set(calls.get() + 1);
                        true
                    })
                    .unwrap_err();
                assert!(matches!(error, SearchError::InvalidConfig { ref field, .. }
                    if field == "query_embedding.native_ann.space_identity"));
            }
            assert_eq!(calls.get(), 0);
        });
    }

    #[test]
    fn exact_handles_never_implicitly_build_or_save_a_graph() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let path = dir
                .path()
                .canonicalize()
                .unwrap()
                .join("not-written.fshnsw");
            let exact =
                NativeAnnIndex::exact(&cx, owner(QuantizationFormat::F32, 1, &rows())).unwrap();
            assert!(
                matches!(exact.save(&cx, &path), Err(SearchError::InvalidConfig { ref field, .. })
                if field == "native_ann.save")
            );
            assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
            assert_eq!(exact.live_count(), 4);
        });
    }

    #[test]
    fn recovery_policy_does_not_swallow_cancellation_or_invalid_configuration() {
        for error in [
            SearchError::Cancelled {
                phase: "test".to_owned(),
                reason: "stop".to_owned(),
            },
            invalid("path", "invalid", "caller error"),
            SearchError::DimensionMismatch {
                expected: 2,
                found: 3,
            },
        ] {
            assert_eq!(recovery_reason(&error), None);
        }
        assert_eq!(
            recovery_reason(&SearchError::IndexNotFound {
                path: "missing.fshnsw".into()
            }),
            Some(NativeExactReason::SidecarMissing)
        );
        assert_eq!(
            recovery_reason(&SearchError::Io(std::io::Error::from(ErrorKind::NotFound))),
            Some(NativeExactReason::SidecarMissing)
        );
        assert_eq!(
            recovery_reason(&SearchError::Io(std::io::Error::from(
                ErrorKind::PermissionDenied
            ))),
            Some(NativeExactReason::SidecarUnreadable)
        );
    }

    struct Provider {
        identity: EmbeddingIdentityBundleV1,
        calls: AtomicUsize,
        cancelled: bool,
    }

    impl Provider {
        fn new(cancelled: bool) -> Self {
            Self {
                identity: identity(),
                calls: AtomicUsize::new(0),
                cancelled,
            }
        }
    }

    impl Embedder for Provider {
        fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            Box::pin(async move {
                self.calls.fetch_add(1, AtomicOrdering::SeqCst);
                if self.cancelled {
                    return Err(SearchError::Cancelled {
                        phase: "test.provider".to_owned(),
                        reason: "stop".to_owned(),
                    });
                }
                Ok(vec![1.0, 0.0])
            })
        }
        fn embed_bound<'a>(
            &'a self,
            cx: &'a Cx,
            text: &'a str,
        ) -> SearchFuture<'a, IdentityBoundEmbedding> {
            Box::pin(async move {
                Ok(IdentityBoundEmbedding {
                    values: self.embed(cx, text).await?,
                    identity: self.identity.clone(),
                })
            })
        }
        fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
            Ok(&self.identity)
        }
        fn id(&self) -> &'static str {
            "native-exact-fallback"
        }
        fn model_name(&self) -> &str {
            self.id()
        }
        fn dimension(&self) -> usize {
            2
        }
        fn is_ready(&self) -> bool {
            true
        }
        fn is_semantic(&self) -> bool {
            false
        }
        fn category(&self) -> ModelCategory {
            ModelCategory::HashEmbedder
        }
    }

    struct EmptyLexical;
    impl LexicalRead for EmptyLexical {
        fn search<'a>(
            &'a self,
            _cx: &'a Cx,
            _text: &'a str,
            _limit: usize,
        ) -> SearchFuture<'a, Vec<ScoredResult>> {
            Box::pin(async { Ok(Vec::new()) })
        }
        fn doc_count(&self) -> SearchResult<usize> {
            Ok(0)
        }
    }

    #[test]
    fn exact_and_ann_arms_compose_through_both_native_progressive_phases() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let owner = owner(QuantizationFormat::F32, 1, &rows());
            let exact = NativeAnnIndex::exact(&cx, Arc::clone(&owner)).unwrap();
            let ann = NativeAnnIndex::build(&cx, owner, HnswParams::default(), 7).unwrap();
            let fast_provider = Provider::new(false);
            let quality_provider = Provider::new(false);
            let lexical = EmptyLexical;
            for (fast, quality) in [(&exact, &ann), (&ann, &exact)] {
                let mut phases = fast
                    .search_hybrid_progressive(
                        &cx,
                        &fast_provider,
                        Some((quality, &quality_provider)),
                        &lexical,
                        "query",
                        1,
                    )
                    .unwrap();
                let NativeSearchPhase::Initial { results, .. } =
                    phases.next_phase().await.unwrap().unwrap()
                else {
                    panic!("initial")
                };
                assert_eq!(results[0].doc_id, "a-tie");
                let NativeSearchPhase::Refined { results, .. } =
                    phases.next_phase().await.unwrap().unwrap()
                else {
                    panic!("refined")
                };
                assert_eq!(results[0].doc_id, "a-tie");
                assert!(
                    results
                        .iter()
                        .all(|hit| hit.source == crate::ScoreSource::HashControl)
                );
                assert!(phases.next_phase().await.unwrap().is_none());
            }
            assert_eq!(fast_provider.calls.load(AtomicOrdering::SeqCst), 2);
            assert_eq!(quality_provider.calls.load(AtomicOrdering::SeqCst), 2);
        });
    }

    #[test]
    fn exact_text_cancellation_is_not_retried_and_empty_owners_skip_inference() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let exact =
                NativeAnnIndex::exact(&cx, owner(QuantizationFormat::F32, 1, &rows())).unwrap();
            let provider = Provider::new(true);
            assert!(
                matches!(exact.search_text(&cx, &provider, "query", 1, None).await,
                Err(SearchError::Cancelled { ref phase, .. }) if phase == "test.provider")
            );
            assert_eq!(provider.calls.load(AtomicOrdering::SeqCst), 1);
            for rows in [Vec::new(), vec![("dead", [1.0, 0.0], false)]] {
                let empty =
                    NativeAnnIndex::exact(&cx, owner(QuantizationFormat::F32, 1, &rows)).unwrap();
                assert!(
                    empty
                        .search_text(&cx, &provider, "query", 10, None)
                        .await
                        .unwrap()
                        .is_empty()
                );
                assert_eq!(empty.live_count(), 0);
            }
            assert_eq!(provider.calls.load(AtomicOrdering::SeqCst), 1);
        });
    }
}
