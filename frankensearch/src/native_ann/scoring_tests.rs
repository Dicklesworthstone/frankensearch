//! Differential tests against the former decode-then-f32-dot serving path.
//! No score tolerance, changed model identity, or performance claim is used.

use super::*;
use frankensearch_core::generation::{ArtifactGenerationIdentityV1, QuantizationFormat};
use frankensearch_index::{FsviV2IdentityBinding, VectorIndex};
use std::cell::Cell;

struct SourceRow {
    id: String,
    values: Vec<f32>,
    live: bool,
}

fn identity(dimension: usize) -> EmbeddingIdentityBundleV1 {
    EmbeddingIdentityBundleV1::explicit_test_model(
        "native-retained-score",
        u32::try_from(dimension).unwrap(),
    )
}

fn owner(
    dimension: usize,
    format: QuantizationFormat,
    rows: &[SourceRow],
) -> Arc<ValidatedFsviBytes> {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("source.fsvi");
    let mut stored = identity(dimension);
    stored.storage.format = "fsvi-v2".to_owned();
    stored.storage.quantization = format;
    stored.storage.endianness = "little-endian".to_owned();
    let binding = FsviV2IdentityBinding::new(
        ArtifactGenerationIdentityV1::new(1, [0x63; 16]).unwrap(),
        stored.freeze().unwrap(),
    )
    .unwrap();
    let mut writer = VectorIndex::create_v2(&path, binding.clone()).unwrap();
    for row in rows {
        if row.live {
            writer.write_record(&row.id, &row.values).unwrap();
        } else {
            writer.write_tombstone_record(&row.id, &row.values).unwrap();
        }
    }
    writer.finish().unwrap();
    Arc::new(ValidatedFsviBytes::from_arc(std::fs::read(path).unwrap().into(), &binding).unwrap())
}

fn vector(state: &mut u64, dimension: usize) -> Vec<f32> {
    let mut values: Vec<_> = (0..dimension)
        .map(|_| {
            *state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let bits = u32::try_from(*state >> 32).unwrap();
            f32::from_bits(0x3f00_0000 | (bits & 0x007f_ffff)) - 0.75
        })
        .collect();
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    for value in &mut values {
        *value /= norm;
    }
    values
}

fn rows(dimension: usize, prefix: &str, count: usize) -> Vec<SourceRow> {
    let mut state = 19;
    let mut rows: Vec<_> = (0..count)
        .map(|ordinal| SourceRow {
            // Longer than CompactString's inline capacity, and not row order.
            id: format!("{prefix}/retained-document-{:04}-source", count - ordinal),
            values: vector(&mut state, dimension),
            live: ordinal % 4 != 0,
        })
        .collect();
    if rows.len() > 3 {
        rows[3].values = rows[2].values.clone();
    }
    rows
}

fn baseline(owner: &ValidatedFsviBytes, physical: usize, query: &[f32]) -> f32 {
    let decoded = owner.vector_at_f32(physical).unwrap();
    dot_product_f32_f32(&decoded, query).unwrap()
}

fn oracle(
    owner: &ValidatedFsviBytes,
    query: &[f32],
    accept: impl Fn(&str) -> bool,
) -> Vec<VectorHit> {
    let mut hits = Vec::new();
    for physical in 0..owner.record_count() {
        let row = owner.row(physical).unwrap();
        if row.flags().is_live() && accept(row.doc_id()) {
            hits.push(VectorHit {
                index: u32::try_from(physical).unwrap(),
                score: baseline(owner, physical, query),
                doc_id: row.doc_id().into(),
            });
        }
    }
    hits.sort_unstable_by(VectorHit::cmp_rank);
    hits
}

fn assert_hits(actual: &[VectorHit], expected: &[VectorHit]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.doc_id, expected.doc_id);
        assert_eq!(actual.index, expected.index);
        assert_eq!(actual.score.to_bits(), expected.score.to_bits());
    }
}

#[test]
fn retained_scores_match_previous_kernel_at_every_chunk_and_tail_boundary() {
    let cx = Cx::for_testing();
    for dimension in (1..=97).chain([128, 256, 384, 768, 1024]) {
        let sources = rows(dimension, "differential", 5);
        for format in [QuantizationFormat::F16, QuantizationFormat::F32] {
            let owner = owner(dimension, format, &sources);
            let index = NativeAnnIndex::exact(&cx, Arc::clone(&owner)).unwrap();
            let mut state = 71;
            let queries = [
                vector(&mut state, dimension),
                vector(&mut state, dimension),
                vec![0.0; dimension],
                vec![-0.0; dimension],
                vec![f32::from_bits(1); dimension],
            ];
            for query in &queries {
                for physical in 0..owner.record_count() {
                    assert_eq!(
                        index.score_retained_row(physical, query).unwrap().to_bits(),
                        baseline(&owner, physical, query).to_bits(),
                        "dimension={dimension} format={format:?} row={physical}",
                    );
                }
            }
        }
    }
}

#[test]
fn f32_tail_keeps_separate_multiplication_instead_of_the_byte_kernels_fma() {
    let mut stored = vec![0.0; 9];
    let mut query = vec![0.0; 9];
    stored[0] = -1.0;
    query[0] = 1.0;
    stored[8] = f32::from_bits(0x3f80_0001); // 1 + 2^-23
    query[8] = f32::from_bits(0x3f7f_fffe); // 1 - 2^-23
    let owner = owner(
        9,
        QuantizationFormat::F32,
        &[SourceRow {
            id: "tail".to_owned(),
            values: stored,
            live: true,
        }],
    );
    let expected = baseline(&owner, 0, &query);
    let row = owner.row(0).unwrap();
    let fused_tail = dot_product_f32_bytes_f32(row.vector_bytes(), &query).unwrap();
    assert_eq!(expected.to_bits(), 0.0_f32.to_bits());
    assert_ne!(
        fused_tail.to_bits(),
        expected.to_bits(),
        "the naive substitution must fail"
    );
    let index = NativeAnnIndex::exact(&Cx::for_testing(), owner).unwrap();
    assert_eq!(
        index.score_retained_row(0, &query).unwrap().to_bits(),
        expected.to_bits()
    );
}

#[test]
fn f16_leftover_chunks_keep_the_original_accumulation_order() {
    let mut stored = vec![0.0; 40];
    stored[0] = 65_504.0;
    stored[8] = -65_504.0;
    stored[32] = 1.0 / 16_384.0;
    let query = vec![1.0; 40];
    let owner = owner(
        40,
        QuantizationFormat::F16,
        &[SourceRow {
            id: "leftover".to_owned(),
            values: stored,
            live: true,
        }],
    );
    let expected = baseline(&owner, 0, &query);
    let row = owner.row(0).unwrap();
    let regrouped = dot_product_f16_bytes_f32(row.vector_bytes(), &query).unwrap();
    assert_eq!(expected.to_bits(), (1.0_f32 / 16_384.0).to_bits());
    assert_ne!(
        regrouped.to_bits(),
        expected.to_bits(),
        "regrouping must be observable"
    );
    let index = NativeAnnIndex::exact(&Cx::for_testing(), owner).unwrap();
    assert_eq!(
        index.score_retained_row(0, &query).unwrap().to_bits(),
        expected.to_bits()
    );
}

#[test]
fn exact_and_reopened_ann_preserve_filtered_top_k_and_raw_score_bits() {
    let cx = Cx::for_testing();
    for dimension in [9, 32, 40, 384] {
        let sources = rows(dimension, "retrieval", 19);
        for format in [QuantizationFormat::F16, QuantizationFormat::F32] {
            let owner = owner(dimension, format, &sources);
            let exact = NativeAnnIndex::exact(&cx, Arc::clone(&owner)).unwrap();
            let ann =
                NativeAnnIndex::build(&cx, Arc::clone(&owner), HnswParams::default(), 19).unwrap();
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("retained.fshnsw");
            ann.save(&cx, &path).unwrap();
            let reopened = NativeAnnIndex::load(&cx, Arc::clone(&owner), &path).unwrap();
            let mut state = 313;
            let query =
                BoundQueryEmbedding::new(vector(&mut state, dimension), identity(dimension))
                    .unwrap();
            for selection in 0..3 {
                let accept = |id: &str| match selection {
                    0 => true,
                    1 => id.contains("0001") || id.contains("0002"),
                    _ => false,
                };
                let expected = oracle(&owner, query.vector(), accept);
                for k in [0, 1, 3, usize::MAX] {
                    let expected = &expected[..k.min(expected.len())];
                    for index in [&exact, &ann, &reopened] {
                        let hits = index
                            .search_filtered(&cx, &query, k, Some(owner.record_count()), accept)
                            .unwrap();
                        assert_hits(&hits, expected);
                    }
                }
            }
        }
    }
}

#[test]
fn mixed_precision_shards_keep_global_order_and_physical_coordinates() {
    let cx = Cx::for_testing();
    for dimension in [9, 32, 384] {
        let first = owner(
            dimension,
            QuantizationFormat::F16,
            &rows(dimension, "first", 9),
        );
        let second = owner(
            dimension,
            QuantizationFormat::F32,
            &rows(dimension, "second", 11),
        );
        let expected = [first.witness().clone(), second.witness().clone()];
        let shards = NativeShardSet::admit(
            &cx,
            &expected,
            vec![
                Arc::new(
                    NativeAnnIndex::build(&cx, Arc::clone(&first), HnswParams::default(), 19)
                        .unwrap(),
                ),
                Arc::new(NativeAnnIndex::exact(&cx, Arc::clone(&second)).unwrap()),
            ],
        )
        .unwrap();
        let mut state = 123;
        let query =
            BoundQueryEmbedding::new(vector(&mut state, dimension), identity(dimension)).unwrap();
        let mut expected: Vec<_> = [&first, &second]
            .iter()
            .enumerate()
            .flat_map(|(shard, owner)| {
                oracle(owner, query.vector(), |_| true)
                    .into_iter()
                    .map(move |hit| NativeShardHit {
                        doc_id: hit.doc_id,
                        score: hit.score,
                        row: NativeShardRow {
                            shard,
                            physical_row: hit.index,
                        },
                    })
            })
            .collect();
        expected.sort_unstable_by(NativeShardHit::cmp_rank);
        for k in [0, 1, 5, usize::MAX] {
            let actual = shards.search(&cx, &query, k, Some(usize::MAX)).unwrap();
            let expected = &expected[..k.min(expected.len())];
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected) {
                assert_eq!(actual.doc_id, expected.doc_id);
                assert_eq!(actual.row, expected.row);
                assert_eq!(actual.score.to_bits(), expected.score.to_bits());
            }
        }
    }
}

#[test]
fn scoring_keeps_dimension_row_identity_and_cancellation_refusals() {
    let cx = Cx::for_testing();
    let owner = owner(32, QuantizationFormat::F32, &rows(32, "admission", 9));
    let exact = NativeAnnIndex::exact(&cx, Arc::clone(&owner)).unwrap();
    let ann = NativeAnnIndex::build(&cx, owner, HnswParams::default(), 19).unwrap();
    for index in [&exact, &ann] {
        assert!(matches!(
            index.score_retained_row(0, &[1.0]),
            Err(SearchError::DimensionMismatch {
                expected: 32,
                found: 1
            })
        ));
        assert!(index.score_retained_row(usize::MAX, &[1.0; 32]).is_err());
        let foreign = BoundQueryEmbedding::new(
            vec![1.0; 32],
            EmbeddingIdentityBundleV1::explicit_test_model("foreign-score", 32),
        )
        .unwrap();
        let calls = Cell::new(0);
        for k in [0, 1] {
            assert!(
                index
                    .search_filtered(&cx, &foreign, k, None, |_| {
                        calls.set(calls.get() + 1);
                        true
                    })
                    .is_err()
            );
        }
        assert_eq!(calls.get(), 0);
        let query = BoundQueryEmbedding::new(vec![1.0; 32], identity(32)).unwrap();
        let error = index
            .search_filtered(&cx, &query, 1, Some(usize::MAX), |_| {
                cx.set_cancel_requested(true);
                false
            })
            .unwrap_err();
        cx.set_cancel_requested(false);
        assert!(matches!(error, SearchError::Cancelled { .. }));
        assert!(!index.search(&cx, &query, 1, None).unwrap().is_empty());
    }
}

#[test]
fn overflowing_finite_scores_cannot_escape_as_a_successful_prefix() {
    let cx = Cx::for_testing();
    let sources = [
        SourceRow {
            id: "good".to_owned(),
            values: vec![0.01; 32],
            live: true,
        },
        SourceRow {
            id: "overflow".to_owned(),
            values: vec![f32::MAX; 32],
            live: true,
        },
    ];
    let index = NativeAnnIndex::exact(&cx, owner(32, QuantizationFormat::F32, &sources)).unwrap();
    let query = BoundQueryEmbedding::new(vec![2.0; 32], identity(32)).unwrap();
    let error = index.search(&cx, &query, 1, None).unwrap_err();
    assert!(
        matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "native_ann.score")
    );
}
