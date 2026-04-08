//! Integration tests for in-memory vector and two-tier index APIs.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use frankensearch_core::filter::SearchFilter;
use frankensearch_core::{SearchError, VectorHit};
use frankensearch_index::{
    InMemoryTwoTierIndex, InMemoryVectorIndex, Quantization, SearchParams, VectorIndex,
};

fn temp_index_path(name: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nonce = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join("frankensearch_in_memory_integration_tests");
    std::fs::create_dir_all(&dir).expect("create test dir");
    dir.join(format!("{name}-{nonce}.fsvi"))
}

fn cleanup(path: &Path) {
    let _ = std::fs::remove_file(path);
    let _ = std::fs::remove_file(path.with_extension("fsvi.wal"));
}

fn normalize(values: Vec<f32>) -> Vec<f32> {
    let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm <= f32::EPSILON {
        return values;
    }
    values.into_iter().map(|v| v / norm).collect()
}

#[allow(clippy::cast_precision_loss)]
fn make_vector(dim: usize, seed: f32) -> Vec<f32> {
    normalize(
        (0..dim)
            .map(|i| (i as f32).mul_add(0.17, seed).sin())
            .collect::<Vec<_>>(),
    )
}

fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum::<f32>()
}

#[test]
fn from_fsvi_matches_file_backed_top_k() {
    let path = temp_index_path("from_fsvi_parity");
    cleanup(&path);

    let dim = 64;
    let doc_count = 96usize;
    let mut writer = VectorIndex::create_with_revision(
        &path,
        "test-embedder",
        "in-memory-tests",
        dim,
        Quantization::F16,
    )
    .expect("create fsvi writer");

    for idx in 0..doc_count {
        #[allow(clippy::cast_precision_loss)]
        let vec = make_vector(dim, idx as f32 * 0.37);
        writer
            .write_record(&format!("doc-{idx}"), &vec)
            .expect("write record");
    }
    writer.finish().expect("finish fsvi writer");

    let file_index = VectorIndex::open(&path).expect("open file-backed index");
    let in_memory = InMemoryVectorIndex::from_fsvi(&path).expect("load in-memory index");

    let query = make_vector(dim, 13.7);
    let file_hits = file_index
        .search_top_k(&query, 12, None)
        .expect("file-backed search");
    let mem_hits = in_memory
        .search_top_k(&query, 12, None)
        .expect("in-memory search");

    assert_eq!(file_hits.len(), mem_hits.len());
    for (file, memory) in file_hits.iter().zip(mem_hits.iter()) {
        assert_eq!(file.doc_id, memory.doc_id);
        assert!(
            (file.score - memory.score).abs() < 0.001,
            "score drift too high for {}: file={} memory={}",
            file.doc_id,
            file.score,
            memory.score
        );
    }

    cleanup(&path);
}

#[test]
fn from_vectors_top_k_filter_and_dimension_errors() {
    struct ExcludeBeta;
    impl SearchFilter for ExcludeBeta {
        fn matches(&self, doc_id: &str, _metadata: Option<&serde_json::Value>) -> bool {
            doc_id != "beta"
        }
        fn name(&self) -> &'static str {
            "exclude-beta"
        }
    }

    let index = InMemoryVectorIndex::from_vectors(
        vec!["alpha".to_owned(), "beta".to_owned(), "gamma".to_owned()],
        vec![
            normalize(vec![1.0, 0.0, 0.0, 0.0]),
            normalize(vec![0.9, 0.1, 0.0, 0.0]),
            normalize(vec![0.0, 0.0, 1.0, 0.0]),
        ],
        4,
    )
    .expect("build in-memory index");

    let query = normalize(vec![1.0, 0.0, 0.0, 0.0]);
    let hits = index
        .search_top_k(&query, 3, Some(&ExcludeBeta))
        .expect("filtered search");
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].doc_id, "alpha");
    assert!(hits.iter().all(|hit| hit.doc_id != "beta"));

    let err = index
        .search_top_k(&[1.0, 0.0], 3, None)
        .expect_err("dimension mismatch must error");
    assert!(matches!(
        err,
        SearchError::DimensionMismatch {
            expected: 4,
            found: 2
        }
    ));
}

#[test]
fn f16_scores_track_f32_reference_within_tolerance() {
    let dim = 256;
    let doc = make_vector(dim, 4.2);
    let query = make_vector(dim, 7.7);
    let expected = dot(&doc, &query);

    let index =
        InMemoryVectorIndex::from_vectors(vec!["doc-1".to_owned()], vec![doc], dim).expect("index");
    let hits = index
        .search_top_k(&query, 1, None)
        .expect("single-doc search");

    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id, "doc-1");
    assert!(
        (hits[0].score - expected).abs() < 0.005,
        "f16 score drift too large: expected={expected} actual={}",
        hits[0].score
    );
}

#[test]
fn empty_and_single_vector_behave_safely() {
    let empty = InMemoryVectorIndex::from_vectors(Vec::new(), Vec::new(), 8).expect("empty index");
    let empty_hits = empty
        .search_top_k(&[0.0; 8], 5, None)
        .expect("empty search");
    assert!(empty_hits.is_empty());

    let single = InMemoryVectorIndex::from_vectors(
        vec!["solo".to_owned()],
        vec![normalize(vec![1.0, 0.0, 0.0, 0.0])],
        4,
    )
    .expect("single index");
    let single_hits = single
        .search_top_k(&normalize(vec![1.0, 0.0, 0.0, 0.0]), 5, None)
        .expect("single search");
    assert_eq!(single_hits.len(), 1);
    assert_eq!(single_hits[0].doc_id, "solo");
}

#[test]
fn parallel_scan_matches_sequential_scan() {
    let dim = 32;
    let count = 6000usize;
    let doc_ids = (0..count).map(|i| format!("doc-{i}")).collect::<Vec<_>>();
    #[allow(clippy::cast_precision_loss)]
    let vectors: Vec<_> = (0..count)
        .map(|i| make_vector(dim, i as f32 * 0.13))
        .collect();
    let query = make_vector(dim, 31.4);
    let index = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).expect("index");

    let sequential = index
        .search_top_k_with_params(
            &query,
            20,
            None,
            SearchParams {
                parallel_enabled: false,
                parallel_threshold: 1,
                parallel_chunk_size: 128,
            },
        )
        .expect("sequential search");

    let parallel = index
        .search_top_k_with_params(
            &query,
            20,
            None,
            SearchParams {
                parallel_enabled: true,
                parallel_threshold: 1,
                parallel_chunk_size: 128,
            },
        )
        .expect("parallel search");

    assert_eq!(sequential.len(), parallel.len());
    for (lhs, rhs) in sequential.iter().zip(parallel.iter()) {
        assert_eq!(lhs.doc_id, rhs.doc_id);
        assert!(
            (lhs.score - rhs.score).abs() < 1e-6,
            "parallel mismatch for {}: {} vs {}",
            lhs.doc_id,
            lhs.score,
            rhs.score
        );
    }
}

#[test]
fn in_memory_two_tier_supports_quality_and_no_quality_modes() {
    let fast = InMemoryVectorIndex::from_vectors(
        vec!["a".to_owned(), "b".to_owned(), "c".to_owned()],
        vec![
            normalize(vec![1.0, 0.0]),
            normalize(vec![0.8, 0.2]),
            normalize(vec![0.0, 1.0]),
        ],
        2,
    )
    .expect("fast index");
    let quality = InMemoryVectorIndex::from_vectors(
        vec!["a".to_owned(), "b".to_owned(), "c".to_owned()],
        vec![
            normalize(vec![0.0, 1.0]),
            normalize(vec![1.0, 0.0]),
            normalize(vec![0.1, 0.9]),
        ],
        2,
    )
    .expect("quality index");

    let with_quality = InMemoryTwoTierIndex::new(fast.clone(), Some(quality));
    let hits = with_quality
        .search_fast(&normalize(vec![1.0, 0.0]), 3)
        .expect("fast search");
    let quality_scores = with_quality
        .quality_scores_for_hits(&normalize(vec![1.0, 0.0]), &hits)
        .expect("quality scores");
    assert_eq!(quality_scores.len(), hits.len());
    assert!(
        quality_scores
            .iter()
            .all(|score| score.is_some_and(|s| s.is_finite()))
    );

    let without_quality = InMemoryTwoTierIndex::new(fast, None);
    let none_scores = without_quality
        .quality_scores_for_hits(&normalize(vec![1.0, 0.0]), &hits)
        .expect("quality fallback scores");
    assert_eq!(none_scores, vec![None; hits.len()]);
}

#[test]
fn quality_scores_for_unknown_doc_id_returns_zero() {
    let index = InMemoryVectorIndex::from_vectors(
        vec!["known".to_owned()],
        vec![normalize(vec![1.0, 0.0])],
        2,
    )
    .expect("index");
    let hits = vec![VectorHit {
        index: 0,
        score: 1.0,
        doc_id: "missing".to_owned(),
    }];
    let scores = index
        .scores_for_hits(&normalize(vec![1.0, 0.0]), &hits)
        .expect("scores");
    assert_eq!(scores, vec![0.0]);
}
