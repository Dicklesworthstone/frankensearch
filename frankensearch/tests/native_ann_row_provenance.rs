//! Public-API regression for #53: a row is an address in a retained FSVI owner,
//! not an insertion ordinal, live-row ordinal, or rank in a candidate window.
//!
//! The physical orders below are independent fixture expectations, not values
//! learned from search results. First check the writer's document/vector table,
//! then check retrieval against that table. All vector components are exactly
//! representable in both F16 and F32, so storage precision cannot mask a bad join.

use std::path::Path;
use std::sync::Arc;

use frankensearch::native_ann::{
    NativeAnnIndex, NativeSearchPhase, NativeShardRow, NativeShardSet, NativeShardedSearchPhase,
};
use frankensearch::{Cx, Embedder, LexicalRead, ScoreSource, ScoredResult, SearchResult};
use frankensearch_core::BoundQueryEmbedding;
use frankensearch_core::generation::{
    ArtifactGenerationIdentityV1, EmbeddingArtifactIdentityV1, EmbeddingIdentityBundleV1,
    EmbeddingSpaceKindV1, QuantizationFormat,
};
use frankensearch_core::traits::{IdentityBoundEmbedding, ModelCategory, SearchFuture};
use frankensearch_index::native_hnsw::HnswParams;
use frankensearch_index::{FsviV2IdentityBinding, ValidatedFsviBytes, VectorIndex};

#[derive(Clone, Copy)]
struct Row {
    id: &'static str,
    vector: &'static [f32],
    live: bool,
}

const FAST: [Row; 5] = [
    Row {
        id: "z-fast",
        vector: &[1.0, 0.0],
        live: true,
    },
    Row {
        id: "middle",
        vector: &[0.75, 0.25],
        live: true,
    },
    Row {
        id: "near",
        vector: &[0.5, 0.5],
        live: true,
    },
    Row {
        id: "a-quality",
        vector: &[0.0, 1.0],
        live: true,
    },
    // Sorts before every live row and would win the query if it leaked out.
    Row {
        id: "deleted",
        vector: &[2.0, 0.0],
        live: false,
    },
];
// FNV-1a order: deleted, middle, z-fast, a-quality, near.
const FAST_PHYSICAL: [usize; 5] = [4, 1, 0, 3, 2];

const QUALITY: [Row; 4] = [
    Row {
        id: "a-quality",
        vector: &[1.0, 0.0, 0.0],
        live: true,
    },
    Row {
        id: "near",
        vector: &[0.0, 1.0, 0.0],
        live: true,
    },
    Row {
        id: "quality-padding",
        vector: &[0.0, 0.0, 1.0],
        live: true,
    },
    Row {
        id: "z-fast",
        vector: &[-1.0, 0.0, 0.0],
        live: true,
    },
];
// Changing insertion order alone cannot give shared IDs different row spaces.
// This different cohort gives a-quality fast row 3 and quality row 1.
const QUALITY_PHYSICAL: [usize; 4] = [3, 0, 2, 1];

struct Provider {
    identity: EmbeddingIdentityBundleV1,
    query: Vec<f32>,
}

impl Provider {
    fn new(name: &str, dimension: u32) -> Self {
        let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(name, dimension);
        // Synthetic semantic-shaped identity, not a real-model quality claim.
        identity.space.kind = EmbeddingSpaceKindV1::Semantic;
        identity.space.hash_control = None;
        identity.space.artifact_manifest_fingerprint = "a".repeat(64);
        identity.space.artifacts = vec![EmbeddingArtifactIdentityV1 {
            role: "weights".to_owned(),
            sha256: "b".repeat(64),
            size: 1,
        }];
        identity.producer.space_fingerprint = identity.space.fingerprint();
        identity.validate().unwrap();
        let mut query = vec![0.0; usize::try_from(dimension).unwrap()];
        query[0] = 1.0;
        Self { identity, query }
    }
}

impl Embedder for Provider {
    fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move { Ok(self.query.clone()) })
    }

    fn embed_bound<'a>(
        &'a self,
        _cx: &'a Cx,
        _text: &'a str,
    ) -> SearchFuture<'a, IdentityBoundEmbedding> {
        Box::pin(async move {
            Ok(IdentityBoundEmbedding {
                values: self.query.clone(),
                identity: self.identity.clone(),
            })
        })
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
    }

    fn id(&self) -> &str {
        &self.identity.space.logical_model_id
    }

    fn model_name(&self) -> &str {
        self.id()
    }

    fn dimension(&self) -> usize {
        self.query.len()
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn is_semantic(&self) -> bool {
        true
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::TransformerEmbedder
    }
}

struct EmptyLexical;

impl LexicalRead for EmptyLexical {
    fn search<'a>(
        &'a self,
        _cx: &'a Cx,
        _query: &'a str,
        _limit: usize,
    ) -> SearchFuture<'a, Vec<ScoredResult>> {
        Box::pin(async { Ok(Vec::new()) })
    }

    fn doc_count(&self) -> SearchResult<usize> {
        Ok(0)
    }
}

fn write_owner(
    path: &Path,
    provider: &Provider,
    format: QuantizationFormat,
    rows: &[Row],
    insertion_order: &[usize],
) -> Arc<ValidatedFsviBytes> {
    assert_eq!(rows.len(), insertion_order.len());
    let mut identity = provider.identity.clone();
    "fsvi-v2".clone_into(&mut identity.storage.format);
    identity.storage.quantization = format;
    "little-endian".clone_into(&mut identity.storage.endianness);
    let binding = FsviV2IdentityBinding::new(
        ArtifactGenerationIdentityV1::new(1, [0x53; 16]).unwrap(),
        identity.freeze().unwrap(),
    )
    .unwrap();
    let mut writer = VectorIndex::create_v2(path, binding.clone()).unwrap();
    for &position in insertion_order {
        let row = rows[position];
        if row.live {
            writer.write_record(row.id, row.vector).unwrap();
        } else {
            writer.write_tombstone_record(row.id, row.vector).unwrap();
        }
    }
    writer.finish().unwrap();
    let bytes: Arc<[u8]> = std::fs::read(path).unwrap().into();
    Arc::new(ValidatedFsviBytes::from_arc(bytes, &binding).unwrap())
}

fn assert_owner(owner: &ValidatedFsviBytes, rows: &[Row], physical_order: &[usize]) {
    assert_eq!(owner.record_count(), rows.len());
    assert_eq!(
        owner.live_count(),
        rows.iter().filter(|row| row.live).count()
    );
    for (physical_row, &source_row) in physical_order.iter().enumerate() {
        let expected = rows[source_row];
        assert_eq!(owner.doc_id_at(physical_row).unwrap(), expected.id);
        assert_eq!(owner.vector_at_f32(physical_row).unwrap(), expected.vector);
    }
}

fn assert_result_row(
    result: &ScoredResult,
    owner: &ValidatedFsviBytes,
    physical_row: u32,
    vector: &[f32],
) {
    assert_eq!(result.index, Some(physical_row));
    let physical_row = usize::try_from(physical_row).unwrap();
    assert_eq!(
        owner.doc_id_at(physical_row).unwrap(),
        result.doc_id.as_str()
    );
    assert_eq!(owner.vector_at_f32(physical_row).unwrap(), vector);
}

#[test]
fn writer_permutations_preserve_rows_through_graph_reload_exact_search_and_filtering() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let provider = Provider::new("row-provenance-fast", 2);
        let query =
            BoundQueryEmbedding::new(provider.query.clone(), provider.identity.clone()).unwrap();
        for format in [QuantizationFormat::F16, QuantizationFormat::F32] {
            let directory = tempfile::tempdir().unwrap();
            let mut image_digest = None;
            for order in [
                [0, 1, 2, 3, 4],
                [4, 3, 2, 1, 0],
                [2, 4, 0, 3, 1],
                [1, 3, 4, 0, 2],
            ] {
                let case = directory.path().join(format!("case-{}", order[0]));
                std::fs::create_dir(&case).unwrap();
                let owner = write_owner(&case.join("rows.fsvi"), &provider, format, &FAST, &order);
                // Fail here, before any graph or retrieval, on a writer-side mismatch.
                assert_owner(&owner, &FAST, &FAST_PHYSICAL);
                let digest = owner.witness().whole_image_sha256;
                if let Some(expected) = image_digest {
                    assert_eq!(digest, expected, "insertion order changed persisted bytes");
                }
                image_digest = Some(digest);
                let graph = case.join("rows.fshnsw");
                let built =
                    NativeAnnIndex::build(&cx, Arc::clone(&owner), HnswParams::default(), 7)
                        .unwrap();
                built.save(&cx, &graph).unwrap();
                let loaded = NativeAnnIndex::load(&cx, Arc::clone(&owner), &graph).unwrap();
                let exact = NativeAnnIndex::exact(&cx, Arc::clone(&owner)).unwrap();
                for index in [built, loaded, exact] {
                    let hits = index.search(&cx, &query, usize::MAX, Some(5)).unwrap();
                    assert_eq!(hits.len(), 4);
                    // Rank order differs from insertion, physical, and live-row order.
                    for (hit, (id, row, score)) in hits.iter().zip([
                        ("z-fast", 2, 1.0_f32),
                        ("middle", 1, 0.75),
                        ("near", 4, 0.5),
                        ("a-quality", 3, 0.0),
                    ]) {
                        assert_eq!(hit.doc_id.as_str(), id);
                        assert_eq!(hit.index, row);
                        assert_eq!(hit.score.to_bits(), score.to_bits());
                        assert_eq!(
                            owner
                                .doc_id_at(usize::try_from(hit.index).unwrap())
                                .unwrap(),
                            id
                        );
                    }
                    let filtered = index
                        .search_filtered(&cx, &query, 1, Some(1), |id| id == "a-quality")
                        .unwrap();
                    assert_eq!(filtered.len(), 1);
                    assert_eq!(filtered[0].doc_id, "a-quality");
                    assert_eq!(
                        filtered[0].index, 3,
                        "not the filtered candidate's rank zero"
                    );
                }
            }
        }
    });
}

struct Tiers {
    directory: tempfile::TempDir,
    fast: Provider,
    quality: Provider,
    fast_owner: Arc<ValidatedFsviBytes>,
    quality_owner: Arc<ValidatedFsviBytes>,
    fast_index: Arc<NativeAnnIndex>,
    quality_index: Arc<NativeAnnIndex>,
}

fn reopened_or_exact(
    cx: &Cx,
    owner: &Arc<ValidatedFsviBytes>,
    graph: &Path,
    ann: bool,
) -> Arc<NativeAnnIndex> {
    Arc::new(if ann {
        NativeAnnIndex::build(cx, Arc::clone(owner), HnswParams::default(), 7)
            .unwrap()
            .save(cx, graph)
            .unwrap();
        NativeAnnIndex::load(cx, Arc::clone(owner), graph).unwrap()
    } else {
        NativeAnnIndex::exact(cx, Arc::clone(owner)).unwrap()
    })
}

fn tiers(cx: &Cx, fast_ann: bool) -> Tiers {
    let directory = tempfile::tempdir().unwrap();
    let fast = Provider::new("row-provenance-fast", 2);
    let quality = Provider::new("row-provenance-quality", 3);
    let fast_owner = write_owner(
        &directory.path().join("fast.fsvi"),
        &fast,
        QuantizationFormat::F16,
        &FAST,
        &[0, 1, 2, 3, 4],
    );
    let quality_owner = write_owner(
        &directory.path().join("quality.fsvi"),
        &quality,
        QuantizationFormat::F32,
        &QUALITY,
        &[0, 1, 2, 3],
    );
    assert_owner(&fast_owner, &FAST, &FAST_PHYSICAL);
    assert_owner(&quality_owner, &QUALITY, &QUALITY_PHYSICAL);
    let fast_index = reopened_or_exact(
        cx,
        &fast_owner,
        &directory.path().join("fast.fshnsw"),
        fast_ann,
    );
    let quality_index = reopened_or_exact(
        cx,
        &quality_owner,
        &directory.path().join("quality.fshnsw"),
        !fast_ann,
    );
    Tiers {
        directory,
        fast,
        quality,
        fast_owner,
        quality_owner,
        fast_index,
        quality_index,
    }
}

#[test]
fn quality_only_and_shared_winners_use_the_contributing_owners_physical_row_space() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for fast_ann in [false, true] {
            let fixture = tiers(&cx, fast_ann);
            let lexical = EmptyLexical;
            let fast_pool = fixture
                .fast_index
                .search_text(&cx, &fixture.fast, "query", 3, None)
                .await
                .unwrap();
            assert!(fast_pool.iter().all(|hit| hit.doc_id != "a-quality"));
            let mut stream = fixture
                .fast_index
                .search_hybrid_progressive(
                    &cx,
                    &fixture.fast,
                    Some((fixture.quality_index.as_ref(), &fixture.quality)),
                    &lexical,
                    "query",
                    1,
                )
                .unwrap();
            let NativeSearchPhase::Initial { results, .. } =
                stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("expected initial phase");
            };
            assert_eq!(results[0].doc_id, "z-fast");
            assert_result_row(&results[0], &fixture.fast_owner, 2, FAST[0].vector);
            let NativeSearchPhase::Refined { results, .. } =
                stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("expected quality refinement");
            };
            assert_eq!(results.len(), 1);
            let winner = &results[0];
            assert_eq!(winner.doc_id, "a-quality");
            assert_eq!(winner.source, ScoreSource::SemanticQuality);
            assert_eq!((winner.fast_score, winner.quality_score), (None, Some(1.0)));
            assert_result_row(winner, &fixture.quality_owner, 1, QUALITY[0].vector);
            assert!(stream.next_phase().await.unwrap().is_none());

            let shared = fixture
                .fast_index
                .search_hybrid_refined_text(
                    &cx,
                    &fixture.fast,
                    (fixture.quality_index.as_ref(), &fixture.quality),
                    &lexical,
                    "query",
                    5,
                )
                .await
                .unwrap();
            let winner = shared.iter().find(|hit| hit.doc_id == "a-quality").unwrap();
            // Even a zero-valued fast score is present evidence. Source alone
            // cannot select the owner: both cases are SemanticQuality.
            assert_eq!(winner.source, ScoreSource::SemanticQuality);
            assert_eq!(
                (winner.fast_score, winner.quality_score),
                (Some(0.0), Some(1.0))
            );
            assert_result_row(winner, &fixture.fast_owner, 3, FAST[3].vector);
        }
    });
}

fn shard_set(cx: &Cx, shards: Vec<Arc<NativeAnnIndex>>) -> NativeShardSet {
    let expected = shards
        .iter()
        .map(|index| index.owner_witness().clone())
        .collect::<Vec<_>>();
    NativeShardSet::admit(cx, &expected, shards).unwrap()
}

#[test]
fn sharded_quality_only_and_shared_winners_keep_both_owner_coordinates() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for fast_ann in [false, true] {
            let fixture = tiers(&cx, fast_ann);
            let empty_owner = write_owner(
                &fixture.directory.path().join("empty.fsvi"),
                &fixture.quality,
                QuantizationFormat::F32,
                &[],
                &[],
            );
            let empty_index = Arc::new(NativeAnnIndex::exact(&cx, empty_owner).unwrap());
            let fast = shard_set(&cx, vec![Arc::clone(&fixture.fast_index)]);
            let quality = shard_set(&cx, vec![empty_index, Arc::clone(&fixture.quality_index)]);
            let lexical = EmptyLexical;
            let mut stream = fast
                .search_hybrid_progressive(
                    &cx,
                    &fixture.fast,
                    Some((&quality, &fixture.quality)),
                    &lexical,
                    "query",
                    1,
                )
                .unwrap();
            assert!(matches!(
                stream.next_phase().await.unwrap(),
                Some(NativeShardedSearchPhase::Initial { .. })
            ));
            let NativeShardedSearchPhase::Refined { results, .. } =
                stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("expected sharded quality refinement");
            };
            assert_eq!(results.len(), 1);
            let winner = &results[0];
            assert_eq!(winner.result.doc_id, "a-quality");
            assert_eq!(winner.result.source, ScoreSource::SemanticQuality);
            assert_eq!(
                (winner.result.fast_score, winner.result.quality_score),
                (None, Some(1.0))
            );
            assert!(winner.result.index.is_none());
            assert!(winner.fast_row.is_none());
            assert_eq!(
                winner.quality_row,
                Some(NativeShardRow {
                    shard: 1,
                    physical_row: 1,
                })
            );
            assert_eq!(
                fixture.quality_owner.doc_id_at(1).unwrap(),
                winner.result.doc_id.as_str()
            );
            assert!(stream.next_phase().await.unwrap().is_none());

            let results = fast
                .search_hybrid_refined_text(
                    &cx,
                    &fixture.fast,
                    (&quality, &fixture.quality),
                    &lexical,
                    "query",
                    5,
                )
                .await
                .unwrap();
            let shared = results
                .iter()
                .find(|hit| hit.result.doc_id == "a-quality")
                .unwrap();
            assert!(shared.result.index.is_none());
            assert_eq!(
                shared.fast_row,
                Some(NativeShardRow {
                    shard: 0,
                    physical_row: 3,
                })
            );
            assert_eq!(
                shared.quality_row,
                Some(NativeShardRow {
                    shard: 1,
                    physical_row: 1,
                })
            );
            assert_eq!(
                (shared.result.fast_score, shared.result.quality_score),
                (Some(0.0), Some(1.0))
            );
            // Resolve every reported location, not just the top-ranked winner.
            for hit in results {
                assert!(hit.result.index.is_none());
                for (owner, location, expected_shard) in [
                    (&fixture.fast_owner, hit.fast_row, 0),
                    (&fixture.quality_owner, hit.quality_row, 1),
                ] {
                    if let Some(row) = location {
                        assert_eq!(row.shard, expected_shard);
                        assert_eq!(
                            owner
                                .doc_id_at(usize::try_from(row.physical_row).unwrap())
                                .unwrap(),
                            hit.result.doc_id.as_str(),
                        );
                    }
                }
            }
        }
    });
}
