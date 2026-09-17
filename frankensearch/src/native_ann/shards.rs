//! Retrieval over an exact caller-selected inventory of disjoint native shards.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::sync::Arc;

use frankensearch_core::{BoundQueryEmbedding, DocId};
use frankensearch_index::FsviV2Witness;

use super::{NativeAnnIndex, NativeRetrievalMode, checkpoint, invalid};
use crate::{Cx, Embedder, SearchResult};

/// A physical row is meaningful only together with its retained shard ordinal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NativeShardRow {
    /// Position in the exact ordered shard inventory admitted by this set.
    pub shard: usize,
    /// Physical FSVI row within that shard, including preceding tombstones.
    pub physical_row: u32,
}

/// One globally ranked result with unambiguous physical provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct NativeShardHit {
    /// Canonical document identifier shared by the shards' input contract.
    pub doc_id: DocId,
    /// Raw retained-owner dot product, not a graph distance or normalized score.
    pub score: f32,
    /// Location in this set's immutable ordered shard inventory.
    pub row: NativeShardRow,
}

impl NativeShardHit {
    /// Score descending, then document ID and physical location ascending.
    #[must_use]
    pub fn cmp_rank(&self, other: &Self) -> Ordering {
        other.score.total_cmp(&self.score)
            .then_with(|| self.doc_id.cmp(&other.doc_id))
            .then_with(|| self.row.shard.cmp(&other.row.shard))
            .then_with(|| self.row.physical_row.cmp(&other.row.physical_row))
    }
}

/// An immutable, identity-admitted native retrieval tier spanning many shards.
///
/// Every owner must match its position in an independently selected expected
/// inventory. Missing, additional, reordered, repeated, and substituted shards
/// are errors. All shards must share the complete artifact generation, space,
/// producer, input contract and dimension. F16 and F32 storage may coexist:
/// scores retain each admitted owner's precision, not a re-quantized copy.
///
/// Physical document IDs must be disjoint across shards, including tombstones.
/// These are partitions, not overlapping versions of a mutable index. Rejecting
/// overlap prevents a live row in one partition from resurrecting another's
/// deletion, and makes top-k merging correct without deduplication underfill.
/// Admission scans row identities once; it does not evaluate vector scores.
///
/// The expected inventory belongs to the consumer's generation selection. This
/// type verifies that exact selection; it does NOT prove that a caller's list
/// includes every document in an external corpus, nor bind a lexical reader.
/// It never discovers paths, changes files, publishes a generation or spawns
/// work. The caller owns the CPU/blocking lane for per-shard graph traversal.
#[derive(Debug)]
pub struct NativeShardSet {
    shards: Vec<Arc<NativeAnnIndex>>,
    live_count: usize,
    physical_count: usize,
}

impl NativeShardSet {
    /// Admit all shards before making a searchable set available.
    ///
    /// Obtain `expected` from the caller's selected immutable inventory, not
    /// from untrusted sidecars being admitted. Empty corpora use one or more
    /// admitted empty shards; an empty inventory has no identity and is refused.
    /// No vector slabs are copied; this set retains the exact input handles.
    ///
    /// # Errors
    ///
    /// Returns cancellation, an inventory or cross-shard identity mismatch,
    /// overlapping partition membership, row-resolution failure or count overflow.
    pub fn admit(
        cx: &Cx,
        expected: &[FsviV2Witness],
        shards: Vec<Arc<NativeAnnIndex>>,
    ) -> SearchResult<Self> {
        checkpoint(cx, "native_ann.shards.admission")?;
        if shards.is_empty() || shards.len() != expected.len() {
            return Err(invalid(
                "shards.inventory", "cardinality",
                "a nonempty exact ordered shard inventory is required",
            ));
        }
        let reference = shards[0].owner_witness();
        let mut live_count = 0_usize;
        let mut physical_count = 0_usize;
        let mut seen_images = BTreeSet::new();
        // Finish all inventory/identity joins before touching any row table.
        for (shard, expected) in shards.iter().zip(expected) {
            checkpoint(cx, "native_ann.shards.identity")?;
            let actual = shard.owner_witness();
            if actual != expected {
                return Err(invalid(
                    "shards.inventory", "witness-mismatch",
                    "each shard must match the expected whole-image witness at its exact position",
                ));
            }
            if !seen_images.insert(actual.whole_image_sha256) {
                return Err(invalid(
                    "shards.inventory", "duplicate-image",
                    "the selected inventory must not repeat an identical physical shard image",
                ));
            }
            if actual.generation != reference.generation {
                return Err(invalid(
                    "shards.generation", "mismatch",
                    "all partitions must belong to the same complete artifact generation",
                ));
            }
            if actual.space_fingerprint != reference.space_fingerprint
                || actual.producer_fingerprint != reference.producer_fingerprint
                || actual.input_fingerprint != reference.input_fingerprint
                || actual.dimension != reference.dimension
            {
                return Err(invalid(
                    "shards.identity", "mismatch",
                    "partitions must agree on space, producer, input contract and dimension",
                ));
            }
            live_count = live_count.checked_add(shard.live_count()).ok_or_else(|| {
                invalid("shards.live_count", "overflow", "live row count must fit usize")
            })?;
            physical_count = physical_count.checked_add(shard.len()).ok_or_else(|| {
                invalid("shards.physical_count", "overflow", "physical row count must fit usize")
            })?;
        }
        let mut partition_by_document: BTreeMap<String, usize> = BTreeMap::new();
        for (ordinal, shard) in shards.iter().enumerate() {
            for physical in 0..shard.len() {
                checkpoint(cx, "native_ann.shards.membership")?;
                let row = shard.owner.row(physical)?;
                if partition_by_document
                    .insert(row.doc_id().to_owned(), ordinal)
                    .is_some_and(|previous| previous != ordinal)
                {
                    return Err(invalid(
                        "shards.membership", "overlap",
                        "physical document identities must not overlap across shard partitions",
                    ));
                }
            }
        }
        checkpoint(cx, "native_ann.shards.admitted")?;
        Ok(Self { shards, live_count, physical_count })
    }

    /// Number of retained partitions, including empty partitions.
    #[must_use]
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Actual live rows summed from the admitted owners, not corpus coverage.
    #[must_use]
    pub const fn live_count(&self) -> usize {
        self.live_count
    }

    /// Physical rows, including tombstones, across the admitted inventory.
    #[must_use]
    pub const fn physical_count(&self) -> usize {
        self.physical_count
    }

    /// Inspect one retained partition without rebinding or reopening it.
    #[must_use]
    pub fn shard(&self, ordinal: usize) -> Option<&NativeAnnIndex> {
        self.shards.get(ordinal).map(AsRef::as_ref)
    }

    /// Per-partition realized backends, preserving explicit exact-fallback reasons.
    pub fn retrieval_modes(&self) -> impl ExactSizeIterator<Item = NativeRetrievalMode> + '_ {
        self.shards.iter().map(|shard| shard.retrieval_mode())
    }

    /// Execute one admitted bound query across every partition.
    ///
    /// A local top-k from each disjoint shard contains every possible global
    /// top-k winner for exact retrieval. ANN shards retain their approximate
    /// recall contract. Merging uses O(k) winners plus one local result window;
    /// it does not concatenate every shard's candidates or vector slab.
    ///
    /// `ef` is forwarded to each ANN shard; exact partitions ignore it. Any
    /// partition failure aborts the entire query, never a partial/quorum result.
    ///
    /// # Errors
    ///
    /// Propagates identity, dimension, scoring, traversal and cancellation errors.
    pub fn search(
        &self,
        cx: &Cx,
        query: &BoundQueryEmbedding,
        k: usize,
        ef: Option<usize>,
    ) -> SearchResult<Vec<NativeShardHit>> {
        self.search_filtered(cx, query, k, ef, |_| true)
    }

    /// Search every partition under one stable document-ID predicate.
    ///
    /// Filtering happens inside each partition's expanding ANN or exact scan,
    /// before global top-k merging. The predicate may be called again during
    /// ANN widening and must remain stable for the whole request.
    ///
    /// # Errors
    ///
    /// Has the same error surface as [`Self::search`]. No partial set escapes.
    pub fn search_filtered<F>(
        &self,
        cx: &Cx,
        query: &BoundQueryEmbedding,
        k: usize,
        ef: Option<usize>,
        accept: F,
    ) -> SearchResult<Vec<NativeShardHit>>
    where
        F: Fn(&str) -> bool,
    {
        checkpoint(cx, "native_ann.shards.search")?;
        // Construction proved every immutable partition has these exact three
        // fingerprints. This is also required on zero-k/all-empty requests.
        self.shards[0].admit_identity(query.identity())?;
        let target = k.min(self.live_count);
        if target == 0 {
            return Ok(Vec::new());
        }
        let mut winners: BinaryHeap<RankedHit> = BinaryHeap::new();
        for (ordinal, shard) in self.shards.iter().enumerate() {
            checkpoint(cx, "native_ann.shards.partition")?;
            let candidates = shard.search_filtered(cx, query, target, ef, &accept)?;
            for candidate in candidates {
                checkpoint(cx, "native_ann.shards.merge")?;
                let hit = NativeShardHit {
                    doc_id: candidate.doc_id,
                    score: candidate.score,
                    row: NativeShardRow { shard: ordinal, physical_row: candidate.index },
                };
                if winners.len() == target {
                    if winners.peek().is_some_and(|worst| hit.cmp_rank(&worst.0) != Ordering::Less) {
                        continue;
                    }
                    let _ = winners.pop();
                }
                winners.push(RankedHit(hit));
            }
        }
        let mut hits: Vec<_> = winners.into_iter().map(|hit| hit.0).collect();
        hits.sort_unstable_by(NativeShardHit::cmp_rank);
        checkpoint(cx, "native_ann.shards.complete")?;
        Ok(hits)
    }

    /// Embed text ONCE for the common admitted identity and query all shards.
    ///
    /// Both advertised and actual provider identities are validated. No work
    /// starts for a foreign identity, even for zero-k or all-empty corpora.
    /// There is no model retry or fallback on any provider failure.
    ///
    /// # Errors
    ///
    /// Propagates provider, binding, per-shard search and cancellation errors.
    pub async fn search_text(
        &self,
        cx: &Cx,
        embedder: &dyn Embedder,
        text: &str,
        k: usize,
        ef: Option<usize>,
    ) -> SearchResult<Vec<NativeShardHit>> {
        self.search_text_filtered(cx, embedder, text, k, ef, |_| true).await
    }

    /// Text retrieval with the same stable predicate as [`Self::search_filtered`].
    ///
    /// # Errors
    ///
    /// Combines provider and filtered-search errors without partial success.
    pub async fn search_text_filtered<F>(
        &self,
        cx: &Cx,
        embedder: &dyn Embedder,
        text: &str,
        k: usize,
        ef: Option<usize>,
        accept: F,
    ) -> SearchResult<Vec<NativeShardHit>>
    where
        F: Fn(&str) -> bool + Send,
    {
        checkpoint(cx, "native_ann.shards.text")?;
        let reference = &self.shards[0];
        reference.admit_identity(embedder.identity()?)?;
        if k == 0 || self.live_count == 0 {
            return Ok(Vec::new());
        }
        let query = reference.embed_query(cx, embedder, text).await?;
        self.search_filtered(cx, &query, k, ef, accept)
    }
}

// cmp_rank puts better hits first, so the heap's maximum is the worst winner.
struct RankedHit(NativeShardHit);
impl PartialEq for RankedHit {
    fn eq(&self, other: &Self) -> bool { self.cmp(other) == Ordering::Equal }
}
impl Eq for RankedHit {}
impl PartialOrd for RankedHit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for RankedHit {
    fn cmp(&self, other: &Self) -> Ordering { self.0.cmp_rank(&other.0) }
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
    use frankensearch_index::{FsviV2IdentityBinding, ValidatedFsviBytes, VectorIndex};
    use frankensearch_index::native_hnsw::HnswParams;
    use crate::SearchError;

    fn identity() -> EmbeddingIdentityBundleV1 {
        EmbeddingIdentityBundleV1::explicit_test_model("native-shards", 2)
    }

    fn partition(
        cx: &Cx,
        identity: &EmbeddingIdentityBundleV1,
        rows: &[(&str, [f32; 2], bool)],
        generation: u64,
        format: QuantizationFormat,
        ann: bool,
    ) -> Arc<NativeAnnIndex> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("partition.fsvi");
        let mut bundle = identity.clone();
        bundle.storage.format = "fsvi-v2".to_owned();
        bundle.storage.quantization = format;
        bundle.storage.endianness = "little-endian".to_owned();
        let binding = FsviV2IdentityBinding::new(
            ArtifactGenerationIdentityV1::new(generation, [0x55; 16]).unwrap(),
            bundle.freeze().unwrap(),
        ).unwrap();
        let mut writer = VectorIndex::create_v2(&path, binding.clone()).unwrap();
        for &(id, vector, live) in rows {
            if live { writer.write_record(id, &vector).unwrap(); }
            else { writer.write_tombstone_record(id, &vector).unwrap(); }
        }
        writer.finish().unwrap();
        let bytes: Arc<[u8]> = std::fs::read(path).unwrap().into();
        let owner = Arc::new(ValidatedFsviBytes::from_arc(bytes, &binding).unwrap());
        Arc::new(if ann {
            NativeAnnIndex::build(cx, owner, HnswParams::default(), 7).unwrap()
        } else { NativeAnnIndex::exact(cx, owner).unwrap() })
    }

    fn admit(cx: &Cx, shards: Vec<Arc<NativeAnnIndex>>) -> NativeShardSet {
        let expected: Vec<_> = shards.iter().map(|shard| shard.owner_witness().clone()).collect();
        NativeShardSet::admit(cx, &expected, shards).unwrap()
    }

    fn query() -> BoundQueryEmbedding {
        BoundQueryEmbedding::new(vec![1.0, 0.0], identity()).unwrap()
    }

    struct Provider { identity: EmbeddingIdentityBundleV1, calls: AtomicUsize, cancel: bool }
    impl Provider {
        fn new() -> Self {
            Self { identity: identity(), calls: AtomicUsize::new(0), cancel: false }
        }
    }
    impl Embedder for Provider {
        fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            Box::pin(async { Ok(vec![1.0, 0.0]) })
        }
        fn embed_bound<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, IdentityBoundEmbedding> {
            Box::pin(async move {
                self.calls.fetch_add(1, AtomicOrdering::SeqCst);
                if self.cancel {
                    return Err(SearchError::Cancelled { phase: "test.provider".to_owned(), reason: "cancelled".to_owned() });
                }
                Ok(IdentityBoundEmbedding { values: vec![1.0, 0.0], identity: self.identity.clone() })
            })
        }
        fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> { Ok(&self.identity) }
        fn id(&self) -> &'static str { "native-shards-test" }
        fn model_name(&self) -> &str { self.id() }
        fn dimension(&self) -> usize { 2 }
        fn is_ready(&self) -> bool { true }
        fn is_semantic(&self) -> bool { false }
        fn category(&self) -> ModelCategory { ModelCategory::HashEmbedder }
    }

    #[test]
    fn global_top_k_matches_the_full_union_for_mixed_storage_and_backends() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for ann in [false, true] {
                let a = partition(&cx, &identity(), &[("dead", [1.0, 0.0], false), ("a", [0.7, 0.3], true), ("c", [0.2, 0.8], true)], 1, QuantizationFormat::F16, ann);
                let b = partition(&cx, &identity(), &[("b", [0.9, 0.1], true), ("d", [-0.1, 0.9], true)], 1, QuantizationFormat::F32, false);
                let set = admit(&cx, vec![a, b]);
                let mut exact = Vec::new();
                for (ordinal, shard) in set.shards.iter().enumerate() {
                    for hit in shard.search(&cx, &query(), usize::MAX, Some(100)).unwrap() {
                        exact.push(NativeShardHit { doc_id: hit.doc_id, score: hit.score, row: NativeShardRow { shard: ordinal, physical_row: hit.index } });
                    }
                }
                exact.sort_unstable_by(NativeShardHit::cmp_rank);
                for k in [0, 1, 2, 3, 4, usize::MAX] {
                    assert_eq!(set.search(&cx, &query(), k, Some(100)).unwrap(), exact[..k.min(exact.len())]);
                }
                let hits = set.search(&cx, &query(), 2, None).unwrap();
                assert_eq!(hits[0].row, NativeShardRow { shard: 1, physical_row: 0 });
                assert_eq!(hits[1].row, NativeShardRow { shard: 0, physical_row: 1 });
                assert_eq!((set.shard_count(), set.live_count(), set.physical_count()), (2, 4, 5));
                assert!(set.shard(2).is_none());
                assert_eq!(set.retrieval_modes().len(), 2);
            }
        });
    }

    #[test]
    fn filters_expand_each_partition_before_global_selection() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let a = partition(&cx, &identity(), &[("reject-a", [1.0, 0.0], true), ("keep-a", [0.2, 0.8], true)], 1, QuantizationFormat::F32, true);
            let b = partition(&cx, &identity(), &[("reject-b", [0.9, 0.1], true), ("keep-b", [0.1, 0.9], true)], 1, QuantizationFormat::F32, true);
            let set = admit(&cx, vec![a, b]);
            let calls = Cell::new(0); // This synchronous predicate is deliberately not Sync.
            let hits = set.search_filtered(&cx, &query(), 2, Some(1), |id| { calls.set(calls.get() + 1); id.starts_with("keep-") }).unwrap();
            assert_eq!(hits.iter().map(|hit| hit.doc_id.as_str()).collect::<Vec<_>>(), ["keep-a", "keep-b"]);
            assert!(calls.get() > 0);
            assert!(set.search_filtered(&cx, &query(), 2, None, |_| false).unwrap().is_empty());
        });
    }

    #[test]
    fn equal_scores_use_document_order_not_partition_order() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let z = partition(&cx, &identity(), &[("z", [1.0, 0.0], true)], 1, QuantizationFormat::F32, false);
            let a = partition(&cx, &identity(), &[("a", [1.0, 0.0], true)], 1, QuantizationFormat::F32, false);
            let set = admit(&cx, vec![z, a]);
            assert_eq!(set.search(&cx, &query(), 1, None).unwrap()[0].doc_id, "a");
        });
    }

    #[test]
    fn exact_inventory_rejects_missing_reordered_and_substituted_shards() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let a = partition(&cx, &identity(), &[("a", [1.0, 0.0], true)], 1, QuantizationFormat::F32, false);
            let b = partition(&cx, &identity(), &[("b", [0.0, 1.0], true)], 1, QuantizationFormat::F32, false);
            let replacement = partition(&cx, &identity(), &[("a", [0.0, 1.0], true)], 1, QuantizationFormat::F32, false);
            let expected = [a.owner_witness().clone(), b.owner_witness().clone()];
            for shards in [vec![Arc::clone(&a)], vec![Arc::clone(&b), Arc::clone(&a)], vec![replacement, Arc::clone(&b)], vec![Arc::clone(&a), Arc::clone(&a)]] {
                assert!(matches!(NativeShardSet::admit(&cx, &expected, shards), Err(SearchError::InvalidConfig { ref field, .. }) if field == "native_ann.shards.inventory"));
            }
            assert!(NativeShardSet::admit(&cx, &[], Vec::new()).is_err());
            assert_eq!(a.search(&cx, &query(), 1, None).unwrap()[0].doc_id, "a");
        });
    }

    #[test]
    fn independently_valid_shards_must_share_generation_and_producer() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let a = partition(&cx, &identity(), &[("a", [1.0, 0.0], true)], 1, QuantizationFormat::F32, false);
            let mut foreign = identity();
            foreign.producer.backend = "different-producer".to_owned();
            for (identity, generation, field) in [(identity(), 2, "native_ann.shards.generation"), (foreign, 1, "native_ann.shards.identity"), (EmbeddingIdentityBundleV1::explicit_test_model("other-space", 2), 1, "native_ann.shards.identity")] {
                let b = partition(&cx, &identity, &[("b", [0.0, 1.0], true)], generation, QuantizationFormat::F32, false);
                let expected = [a.owner_witness().clone(), b.owner_witness().clone()];
                assert!(matches!(NativeShardSet::admit(&cx, &expected, vec![Arc::clone(&a), b]), Err(SearchError::InvalidConfig { field: actual, .. }) if actual == field));
            }
        });
    }

    #[test]
    fn overlapping_live_or_tombstoned_partitions_are_not_deduplicated_or_resurrected() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let a = partition(&cx, &identity(), &[("same", [1.0, 0.0], true)], 1, QuantizationFormat::F32, false);
            for live in [true, false] {
                let b = partition(&cx, &identity(), &[("same", [0.0, 1.0], live)], 1, QuantizationFormat::F32, false);
                let expected = [a.owner_witness().clone(), b.owner_witness().clone()];
                assert!(matches!(NativeShardSet::admit(&cx, &expected, vec![Arc::clone(&a), b]), Err(SearchError::InvalidConfig { ref field, .. }) if field == "native_ann.shards.membership"));
            }
        });
    }

    #[test]
    fn text_embeds_once_even_when_the_first_partition_is_empty() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let a = partition(&cx, &identity(), &[], 1, QuantizationFormat::F32, false);
            let b = partition(&cx, &identity(), &[("b", [1.0, 0.0], true)], 1, QuantizationFormat::F32, true);
            let set = admit(&cx, vec![a, b]);
            let provider = Provider::new();
            let hits = set.search_text(&cx, &provider, "query", 1, None).await.unwrap();
            assert_eq!(hits[0].doc_id, "b");
            assert_eq!(hits[0].row.shard, 1);
            assert_eq!(provider.calls.load(AtomicOrdering::SeqCst), 1);
        });
    }

    #[test]
    fn zero_work_queries_still_refuse_foreign_identities_without_filter_calls() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for rows in [vec![], vec![("a", [1.0, 0.0], true)]] {
                let set = admit(&cx, vec![partition(&cx, &identity(), &rows, 1, QuantizationFormat::F32, false)]);
                let mut foreign = identity();
                foreign.producer.backend = "foreign".to_owned();
                let query = BoundQueryEmbedding::new(vec![1.0, 0.0], foreign).unwrap();
                let calls = Cell::new(0);
                for k in [0, 1] {
                    assert!(set.search_filtered(&cx, &query, k, None, |_| { calls.set(calls.get() + 1); true }).is_err());
                }
                assert_eq!(calls.get(), 0);
                let provider = Provider::new();
                assert!(set.search_text(&cx, &provider, "query", 0, None).await.unwrap().is_empty());
                assert_eq!(provider.calls.load(AtomicOrdering::SeqCst), 0);
            }
        });
    }

    #[test]
    fn selected_owners_survive_caller_drop_and_no_fallback_masks_provider_cancellation() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let a = partition(&cx, &identity(), &[("a", [1.0, 0.0], true)], 1, QuantizationFormat::F32, false);
            let weak = Arc::downgrade(&a);
            let set = admit(&cx, vec![a]);
            assert!(weak.upgrade().is_some());
            let mut provider = Provider::new();
            provider.cancel = true;
            assert!(matches!(set.search_text(&cx, &provider, "query", 1, None).await, Err(SearchError::Cancelled { ref phase, .. }) if phase == "test.provider"));
            assert_eq!(provider.calls.load(AtomicOrdering::SeqCst), 1);
            assert_eq!(set.search(&cx, &query(), 1, None).unwrap()[0].doc_id, "a");
            drop(set);
            assert!(weak.upgrade().is_none());
        });
    }
}

#[cfg(test)]
mod merge_tests {
    use super::*;

    #[test]
    fn bounded_heap_matches_sorted_union_across_orders_limits_and_signed_zero() {
        let values = [-2.0_f32, -0.0, 0.0, 1.0, 1.0, 3.0];
        for rotation in 0..values.len() {
            let input: Vec<_> = (0..values.len()).map(|offset| {
                let i = (rotation + offset) % values.len();
                NativeShardHit {
                    doc_id: format!("doc-{i}").into(),
                    score: values[i],
                    row: NativeShardRow { shard: i % 3, physical_row: u32::try_from(i / 3).unwrap() },
                }
            }).collect();
            let mut sorted = input.clone();
            sorted.sort_unstable_by(NativeShardHit::cmp_rank);
            for target in 1..=values.len() {
                let mut heap: BinaryHeap<RankedHit> = BinaryHeap::new();
                for hit in input.iter().cloned() {
                    if heap.len() == target {
                        if heap.peek().is_some_and(|worst| hit.cmp_rank(&worst.0) != Ordering::Less) {
                            continue;
                        }
                        let _ = heap.pop();
                    }
                    heap.push(RankedHit(hit));
                }
                let mut actual: Vec<_> = heap.into_iter().map(|hit| hit.0).collect();
                actual.sort_unstable_by(NativeShardHit::cmp_rank);
                assert_eq!(actual, sorted[..target]);
            }
        }
    }
}
