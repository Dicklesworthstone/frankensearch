//! Typed, owner-retaining access to the in-tree native HNSW engine.
//!
//! This is an opt-in single-tier retrieval arm, not a replacement for the
//! default two-tier searcher or a composite-generation publisher. Both build
//! and load require already-admitted FSVI v2 bytes. Every query joins the
//! retained artifact's space, producer and input fingerprints before graph
//! traversal, including zero-result requests.
//!
//! Operations are synchronous and belong on the caller's blocking/CPU lane.
//! Cancellation checkpoints bracket graph construction/loading and search;
//! they do not preempt the native engine inside a synchronous call. No runtime,
//! task, fallback embedder or detached worker is created here.

use std::path::Path;
use std::sync::Arc;

use frankensearch_core::BoundQueryEmbedding;
use frankensearch_core::generation::EmbeddingIdentityBundleV1;
use frankensearch_index::native_hnsw::{
    HnswParams, NativeHnswGenerationReceiptV2, ValidatedNativeHnsw,
};
use frankensearch_index::{FsviV2Witness, ValidatedFsviBytes, dot_product_f32_f32};

use crate::{Cx, SearchError, SearchResult, VectorHit};

/// A native ANN retrieval arm bound to one immutable, admitted vector owner.
///
/// Fields are private: callers cannot pair another owner's document lookup or
/// score vector with this graph. Owned output hits retain the exact physical
/// row identifiers of the admitted source. Query scores are recomputed from
/// that source at its declared precision, rather than recovered by subtracting
/// the rounded graph distance from one.
#[derive(Debug)]
pub struct NativeAnnIndex {
    owner: Arc<ValidatedFsviBytes>,
    graph: ValidatedNativeHnsw,
    default_ef_search: usize,
}

impl NativeAnnIndex {
    /// Build the native graph over the supplied admitted owner.
    ///
    /// Tombstones remain routing nodes, but cannot become returned hits.
    /// The owner is retained without copying its vector slab.
    ///
    /// # Errors
    ///
    /// Propagates cancellation, invalid construction parameters and graph
    /// construction/structural-admission failures. No alternate graph or vector
    /// source is substituted after a failure.
    pub fn build(
        cx: &Cx,
        owner: Arc<ValidatedFsviBytes>,
        params: HnswParams,
        seed: u64,
    ) -> SearchResult<Self> {
        checkpoint(cx, "native_ann.build")?;
        let graph = ValidatedNativeHnsw::build(Arc::clone(&owner), params, seed)?;
        checkpoint(cx, "native_ann.build_complete")?;
        Ok(Self {
            owner,
            graph,
            default_ef_search: params.ef_search,
        })
    }

    /// Reopen a graph only when its receipt matches this exact admitted owner.
    ///
    /// A missing, corrupt or stale receipt is an error, never an implicit
    /// rebuild. This keeps generation drift visible to the caller's recovery
    /// policy rather than silently switching retrieval state.
    ///
    /// # Errors
    ///
    /// Propagates cancellation and native graph/receipt admission or I/O errors.
    pub fn load(
        cx: &Cx,
        owner: Arc<ValidatedFsviBytes>,
        graph_path: &Path,
    ) -> SearchResult<Self> {
        checkpoint(cx, "native_ann.load")?;
        let (graph, receipt) = ValidatedNativeHnsw::load(Arc::clone(&owner), graph_path)?;
        let default_ef_search = usize::try_from(receipt.params.ef_search).map_err(|_| {
            invalid(
                "ef_search",
                "out-of-range",
                "persisted search width does not fit usize",
            )
        })?;
        checkpoint(cx, "native_ann.load_complete")?;
        Ok(Self {
            owner,
            graph,
            default_ef_search,
        })
    }

    /// Save this graph and its exact-owner receipt to a generation-specific path.
    ///
    /// The native writer's two atomic renames are NOT a composite transaction.
    /// The caller must serialize writes and select the generation only after
    /// both files are sealed. Cancellation is checked before starting, not
    /// after committing files, so successful publication is not misreported as
    /// an unperformed cancelled operation.
    ///
    /// # Errors
    ///
    /// Propagates cancellation before writing, path validation, and graph or
    /// receipt persistence errors. A partial pair is rejected on subsequent load.
    pub fn save(
        &self,
        cx: &Cx,
        graph_path: &Path,
    ) -> SearchResult<NativeHnswGenerationReceiptV2> {
        checkpoint(cx, "native_ann.save")?;
        self.graph.save(graph_path)
    }

    /// Retrieve up to `k` live hits from a query bound to the artifact's producer.
    ///
    /// `ef` controls the candidate pool as well as the graph beam. Candidates
    /// are rescored against the retained owner and sorted by score descending,
    /// then document id. A full-physical-row beam examines every live row;
    /// smaller beams remain approximate and make no certified recall claim.
    ///
    /// # Errors
    ///
    /// Returns cancellation, query identity/dimension errors, non-finite score
    /// errors, or the native engine's traversal and row-resolution errors.
    pub fn search(
        &self,
        cx: &Cx,
        query: &BoundQueryEmbedding,
        k: usize,
        ef: Option<usize>,
    ) -> SearchResult<Vec<VectorHit>> {
        self.search_filtered(cx, query, k, ef, |_| true)
    }

    /// Retrieve live hits accepted by a stable document predicate.
    ///
    /// Rejected documents and tombstones can still route graph traversal, but
    /// never cross the result boundary. When filtering underfills the result,
    /// candidate and beam widths grow to the physical-row bound. Thus filtering
    /// cannot silently truncate the first ANN window when more eligible hits
    /// exist. With fewer than `k` eligible rows, the final result is shorter.
    /// The predicate must return the same answer for a document throughout this
    /// call; widening may evaluate it again.
    ///
    /// # Errors
    ///
    /// Has the same error surface as [`Self::search`]. Predicate panics are not
    /// intercepted. No filtered, partial result is returned on cancellation.
    pub fn search_filtered<F>(
        &self,
        cx: &Cx,
        query: &BoundQueryEmbedding,
        k: usize,
        ef: Option<usize>,
        accept: F,
    ) -> SearchResult<Vec<VectorHit>>
    where
        F: Fn(&str) -> bool,
    {
        checkpoint(cx, "native_ann.search")?;
        self.admit_identity(query.identity())?;
        if query.vector().len() != self.owner.dimension() {
            return Err(SearchError::DimensionMismatch {
                expected: self.owner.dimension(),
                found: query.vector().len(),
            });
        }
        let target = k.min(self.owner.live_count());
        if target == 0 {
            return Ok(Vec::new());
        }
        let physical_rows = self.owner.record_count();
        let mut width = ef
            .unwrap_or(self.default_ef_search)
            .max(target)
            .min(physical_rows);
        loop {
            checkpoint(cx, "native_ann.search_window")?;
            let candidates = self.graph.search(query.vector(), width, Some(width))?;
            checkpoint(cx, "native_ann.search_candidates")?;
            let mut hits = Vec::with_capacity(candidates.len());
            for candidate in candidates {
                checkpoint(cx, "native_ann.rescore")?;
                if !accept(candidate.doc_id()) {
                    continue;
                }
                let index = candidate.physical_row();
                let row = usize::try_from(index).map_err(|_| {
                    invalid("physical_row", "out-of-range", "source row does not fit usize")
                })?;
                let vector = self.owner.vector_at_f32(row)?;
                let score = dot_product_f32_f32(&vector, query.vector())?;
                if !score.is_finite() {
                    return Err(invalid(
                        "score",
                        "non-finite",
                        "retained-owner dot product must be finite",
                    ));
                }
                hits.push(VectorHit {
                    index,
                    score,
                    doc_id: candidate.doc_id().into(),
                });
            }
            hits.sort_unstable_by(VectorHit::cmp_rank);
            if hits.len() >= target || width == physical_rows {
                hits.truncate(target);
                checkpoint(cx, "native_ann.search_complete")?;
                return Ok(hits);
            }
            // width < physical_rows, so the progress floor cannot overflow.
            width = width.saturating_mul(2).max(width + 1).min(physical_rows);
        }
    }

    /// Physical rows, including routing tombstones.
    #[must_use]
    pub fn len(&self) -> usize {
        self.owner.record_count()
    }

    /// Whether there are no physical rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.owner.record_count() == 0
    }

    /// Searchable live rows, excluding tombstones.
    #[must_use]
    pub fn live_count(&self) -> usize {
        self.owner.live_count()
    }

    /// Witness for the exact immutable owner retained by this retrieval arm.
    #[must_use]
    pub fn owner_witness(&self) -> &FsviV2Witness {
        self.graph.owner_witness()
    }

    fn admit_identity(&self, identity: &EmbeddingIdentityBundleV1) -> SearchResult<()> {
        identity.validate()?;
        let expected = self.owner.identity_v2();
        for (field, actual, fingerprint) in [
            (
                "space_identity",
                identity.space.fingerprint(),
                expected.space_fingerprint,
            ),
            (
                "producer_conformance",
                identity.producer.fingerprint(),
                expected.producer_fingerprint,
            ),
            (
                "input_identity",
                identity.input.fingerprint(),
                expected.input_fingerprint,
            ),
        ] {
            if actual != lower_hex(&fingerprint) {
                return Err(SearchError::InvalidConfig {
                    field: format!("query_embedding.native_ann.{field}"),
                    value: actual,
                    reason: "query identity does not match the retained admitted FSVI owner; \
                             same dimensions or copied producer certificates do not authorize \
                             cross-generation vector-space substitution"
                        .to_owned(),
                });
            }
        }
        Ok(())
    }
}

fn checkpoint(cx: &Cx, phase: &str) -> SearchResult<()> {
    if cx.checkpoint().is_err() || cx.is_cancel_requested() {
        return Err(SearchError::Cancelled {
            phase: phase.to_owned(),
            reason: cx.cancel_reason().map_or_else(
                || "native ANN checkpoint interrupted".to_owned(),
                |reason| reason.to_string(),
            ),
        });
    }
    Ok(())
}

fn invalid(field: &str, value: &str, reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: format!("native_ann.{field}"),
        value: value.to_owned(),
        reason: reason.to_owned(),
    }
}

fn lower_hex(bytes: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for &byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 15)]));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use frankensearch_core::generation::{ArtifactGenerationIdentityV1, QuantizationFormat};
    use frankensearch_index::{FsviV2IdentityBinding, VectorIndex};

    fn identity() -> EmbeddingIdentityBundleV1 {
        EmbeddingIdentityBundleV1::explicit_test_model("native-ann-facade", 4)
    }

    fn query(values: [f32; 4]) -> BoundQueryEmbedding {
        BoundQueryEmbedding::new(values.to_vec(), identity()).expect("bound query")
    }

    fn owner(
        rows: &[(&str, [f32; 4], bool)],
        generation: u64,
        format: QuantizationFormat,
    ) -> Arc<ValidatedFsviBytes> {
        let dir = tempfile::tempdir().expect("owner directory");
        let path = dir.path().join("owner.fsvi");
        let mut bundle = identity();
        bundle.storage.format = "fsvi-v2".to_owned();
        bundle.storage.quantization = format;
        bundle.storage.endianness = "little-endian".to_owned();
        let binding = FsviV2IdentityBinding::new(
            ArtifactGenerationIdentityV1::new(generation, [0x41; 16]).unwrap(),
            bundle.freeze().unwrap(),
        )
        .unwrap();
        let mut writer = VectorIndex::create_v2(&path, binding.clone()).unwrap();
        for &(doc_id, vector, live) in rows {
            if live {
                writer.write_record(doc_id, &vector).unwrap();
            } else {
                writer.write_tombstone_record(doc_id, &vector).unwrap();
            }
        }
        writer.finish().unwrap();
        let bytes: Arc<[u8]> = std::fs::read(path).unwrap().into();
        Arc::new(ValidatedFsviBytes::from_arc(bytes, &binding).unwrap())
    }

    fn rows() -> Vec<(&'static str, [f32; 4], bool)> {
        vec![
            ("dead", [1.0, 0.0, 0.0, 0.0], false),
            ("alpha", [0.9, 0.1, 0.0, 0.0], true),
            ("beta", [0.7, 0.3, 0.0, 0.0], true),
            ("gamma", [0.4, 0.6, 0.0, 0.0], true),
            ("delta", [0.0, 1.0, 0.0, 0.0], true),
        ]
    }

    #[test]
    fn facade_full_beam_matches_exact_owner_for_both_storage_formats() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for format in [QuantizationFormat::F16, QuantizationFormat::F32] {
                let owner = owner(&rows(), 1, format);
                let index = NativeAnnIndex::build(&cx, Arc::clone(&owner), HnswParams::default(), 7)
                    .expect("native graph");
                let query = query([1.0, 0.0, 0.0, 0.0]);
                let mut exact = owner.search_top_k(query.vector(), owner.live_count(), None).unwrap();
                exact.sort_unstable_by(VectorHit::cmp_rank);
                let hits = index.search(&cx, &query, usize::MAX, Some(owner.record_count())).unwrap();
                assert_eq!(hits, exact);
                assert_eq!(index.len(), 5);
                assert_eq!(index.live_count(), 4);
                assert!(!index.is_empty());
                assert!(hits.iter().all(|hit| hit.doc_id != "dead"));
                assert_eq!(index.owner_witness(), owner.witness());
            }
        });
    }

    #[test]
    fn filtered_search_expands_past_rejected_nearest_neighbors() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let owner = owner(&rows(), 1, QuantizationFormat::F32);
            let index = NativeAnnIndex::build(&cx, owner, HnswParams::default(), 7).unwrap();
            let query = query([1.0, 0.0, 0.0, 0.0]);
            let hits = index.search_filtered(&cx, &query, 2, Some(1), |id| {
                matches!(id, "gamma" | "delta")
            }).unwrap();
            assert_eq!(hits.iter().map(|hit| hit.doc_id.as_str()).collect::<Vec<_>>(), ["gamma", "delta"]);
            let one = index.search_filtered(&cx, &query, usize::MAX, Some(1), |id| id == "delta").unwrap();
            assert_eq!(one.len(), 1);
            assert_eq!(one[0].doc_id, "delta");
            assert!(index.search_filtered(&cx, &query, 2, Some(1), |_| false).unwrap().is_empty());
        });
    }

    #[test]
    fn query_identity_is_checked_even_before_empty_and_zero_k_paths() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for rows in [rows(), Vec::new()] {
                let owner = owner(&rows, 1, QuantizationFormat::F32);
                let index = NativeAnnIndex::build(&cx, owner, HnswParams::default(), 7).unwrap();
                let foreign = BoundQueryEmbedding::new(
                    vec![1.0, 0.0, 0.0, 0.0],
                    EmbeddingIdentityBundleV1::explicit_test_model("wrong-model", 4),
                ).unwrap();
                for k in [0, 1] {
                    let error = index.search(&cx, &foreign, k, None).unwrap_err();
                    assert!(matches!(error, SearchError::InvalidConfig { ref field, .. }
                        if field == "query_embedding.native_ann.space_identity"));
                }
                assert!(index.search(&cx, &query([1.0, 0.0, 0.0, 0.0]), 0, None).unwrap().is_empty());
            }
        });
    }

    #[test]
    fn copied_certificate_does_not_admit_a_foreign_producer() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let owner = owner(&rows(), 1, QuantizationFormat::F32);
            let index = NativeAnnIndex::build(&cx, owner, HnswParams::default(), 7).unwrap();
            let mut foreign = identity();
            foreign.producer.backend = "foreign-backend".to_owned();
            foreign.validate().unwrap();
            assert_eq!(foreign.producer.golden_vectors, identity().producer.golden_vectors);
            let query = BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0, 0.0], foreign).unwrap();
            let error = index.search(&cx, &query, 0, None).unwrap_err();
            assert!(matches!(error, SearchError::InvalidConfig { ref field, .. }
                if field == "query_embedding.native_ann.producer_conformance"));
        });
    }

    #[test]
    fn scores_are_not_recovered_from_rounded_distance() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let rows = [
                ("tiny-one", [1e-9, 1.0, 0.0, 0.0], true),
                ("tiny-two", [2e-9, 1.0, 0.0, 0.0], true),
            ];
            assert_eq!((1.0_f32 - 1e-9).to_bits(), (1.0_f32 - 2e-9).to_bits());
            let owner = owner(&rows, 1, QuantizationFormat::F32);
            let index = NativeAnnIndex::build(&cx, owner, HnswParams::default(), 7).unwrap();
            let hits = index.search(&cx, &query([1.0, 0.0, 0.0, 0.0]), 1, Some(2)).unwrap();
            assert_eq!(hits[0].doc_id, "tiny-two");
            assert_eq!(hits[0].score.to_bits(), 2e-9_f32.to_bits());
        });
    }

    #[test]
    fn receipt_round_trip_retains_source_and_rejects_an_identical_successor() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().canonicalize().unwrap().join("native.fshnsw");
            let owner = owner(&rows(), 1, QuantizationFormat::F32);
            let weak = Arc::downgrade(&owner);
            let index = NativeAnnIndex::build(&cx, Arc::clone(&owner), HnswParams::default(), 7).unwrap();
            let receipt = index.save(&cx, &path).unwrap();
            assert_eq!(receipt.fsvi_physical_row_count, 5);
            let loaded = NativeAnnIndex::load(&cx, Arc::clone(&owner), &path).unwrap();
            let query = query([1.0, 0.0, 0.0, 0.0]);
            assert_eq!(index.search(&cx, &query, 4, None).unwrap(), loaded.search(&cx, &query, 4, None).unwrap());
            let successor = self::owner(&rows(), 2, QuantizationFormat::F32);
            assert!(matches!(NativeAnnIndex::load(&cx, successor, &path), Err(SearchError::IndexCorrupted { .. })));
            drop(index);
            drop(owner);
            drop(dir);
            assert!(weak.upgrade().is_some());
            assert_eq!(loaded.search(&cx, &query, 1, None).unwrap()[0].doc_id, "alpha");
            drop(loaded);
            assert!(weak.upgrade().is_none());
        });
    }

    #[test]
    fn all_tombstoned_owner_has_no_searchable_results() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let owner = owner(&[("dead", [1.0, 0.0, 0.0, 0.0], false)], 1, QuantizationFormat::F32);
            let index = NativeAnnIndex::build(&cx, owner, HnswParams::default(), 7).unwrap();
            assert_eq!(index.len(), 1);
            assert_eq!(index.live_count(), 0);
            assert!(index.search(&cx, &query([1.0, 0.0, 0.0, 0.0]), 4, None).unwrap().is_empty());
        });
    }

    #[test]
    fn invalid_parameters_do_not_produce_a_retrieval_handle() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let owner = owner(&rows(), 1, QuantizationFormat::F32);
            let params = HnswParams { ef_search: 0, ..HnswParams::default() };
            assert!(matches!(NativeAnnIndex::build(&cx, owner, params, 7), Err(SearchError::InvalidConfig { .. })));
        });
    }
}
