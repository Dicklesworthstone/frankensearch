//! Cross-component unit tests for frankensearch (bd-3un.31).
//!
//! These tests verify interactions between crates — not individual components
//! in isolation (those have inline `#[cfg(test)]` modules). The focus is on:
//!
//! 1. FSVI round-trip → SIMD dot product → search correctness
//! 2. Normalize → Blend pipeline composition
//! 3. RRF + Blend end-to-end ranking consistency
//! 4. Queue → canonicalization → content hash determinism
//! 5. Cache + staleness + index reload lifecycle
//! 6. Error propagation across crate boundaries
//! 7. Config validation interactions

use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use frankensearch::TwoTierIndexPaths;
use frankensearch_core::canonicalize::DefaultCanonicalizer;
use frankensearch_core::config::{TwoTierConfig, TwoTierMetrics};
use frankensearch_core::error::SearchError;
use frankensearch_core::types::{RankChanges, ScoreSource, ScoredResult, VectorHit};
use frankensearch_embed::HashEmbedder;
use frankensearch_embed::hash_embedder::HashAlgorithm;
use frankensearch_fusion::cache::{
    IndexCache, IndexSentinel, SENTINEL_VERSION, SentinelFileDetector,
};
use frankensearch_fusion::calibration::{
    Identity, IsotonicRegression, PlattScaling, calibrate_scores_with_labels, compute_ece,
};
use frankensearch_fusion::conformal::{
    AdaptiveConformalState, ConformalSearchCalibration, MondrianConformalCalibration,
};
use frankensearch_fusion::normalize::{min_max_normalize, z_score_normalize};
use frankensearch_fusion::queue::{
    EmbeddingQueue, EmbeddingQueueConfig, EmbeddingRequest, JobOutcome,
};
use frankensearch_fusion::rrf::{RrfConfig, candidate_count, rrf_fuse};
use frankensearch_fusion::{blend_two_tier, compute_rank_changes, kendall_tau};
use frankensearch_index::{
    Quantization, TwoTierIndex, VECTOR_INDEX_FAST_FILENAME, VECTOR_INDEX_QUALITY_FILENAME,
    VectorIndex,
};

// ═══════════════════════════════════════════════════════════════════════════
// Test helpers
// ═══════════════════════════════════════════════════════════════════════════

fn temp_dir(name: &str) -> PathBuf {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "frankensearch-xcomp-{name}-{}-{now}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn write_fast_index(dir: &std::path::Path, records: &[(&str, Vec<f32>)]) {
    let path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    write_index_at(&path, "potion-128M", "v1", records);
}

fn write_quality_index(dir: &std::path::Path, records: &[(&str, Vec<f32>)]) {
    let path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);
    write_index_at(&path, "MiniLM-L6-v2", "v1", records);
}

fn write_index_at(
    path: &std::path::Path,
    embedder_id: &str,
    embedder_revision: &str,
    records: &[(&str, Vec<f32>)],
) {
    let dim = records.first().map_or(4, |(_, vector)| vector.len());
    let mut writer = VectorIndex::create_with_revision(
        path,
        embedder_id,
        embedder_revision,
        dim,
        Quantization::F16,
    )
    .expect("create writer");
    for (doc_id, vec) in records {
        writer.write_record(doc_id, vec).expect("write record");
    }
    writer.finish().expect("finish index");
}

fn hit(doc_id: &str, score: f32, index: u32) -> VectorHit {
    VectorHit {
        index,
        score,
        doc_id: doc_id.into(),
    }
}

fn scored(doc_id: &str, score: f32) -> ScoredResult {
    ScoredResult {
        doc_id: doc_id.into(),
        score,
        source: ScoreSource::Hybrid,
        index: None,
        fast_score: None,
        quality_score: None,
        lexical_score: None,
        rerank_score: None,
        explanation: None,
        metadata: None,
    }
}

fn normalize_vec(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < f32::EPSILON {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. FSVI round-trip → SIMD dot product → search correctness
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fsvi_roundtrip_preserves_search_ranking() {
    // Write vectors, read back, search, and verify ranking is consistent
    // with the known dot-product ordering.
    let dir = temp_dir("fsvi-search-ranking");

    let v_high = normalize_vec(&[0.9, 0.1, 0.0, 0.0]);
    let v_mid = normalize_vec(&[0.5, 0.5, 0.5, 0.0]);
    let v_low = normalize_vec(&[0.0, 0.0, 0.1, 0.9]);

    write_fast_index(
        &dir,
        &[("high", v_high.clone()), ("mid", v_mid), ("low", v_low)],
    );

    let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
    let query = normalize_vec(&[1.0, 0.0, 0.0, 0.0]);
    let hits = index.search_fast(&query, 3).expect("search");

    assert_eq!(hits.len(), 3);
    assert_eq!(hits[0].doc_id, "high");
    assert_eq!(hits[2].doc_id, "low");
    // f16 quantization: scores should be within 1% of f32 dot product
    let expected_high = v_high.iter().zip(&query).map(|(a, b)| a * b).sum::<f32>();
    assert!(
        (hits[0].score - expected_high).abs() < 0.01,
        "f16 roundtrip error too large: {} vs {expected_high}",
        hits[0].score
    );
}

#[test]
fn fsvi_f16_quantization_error_bounded_at_384d() {
    // Verify f16 quantization accuracy for realistic 384-dim vectors
    let dir = temp_dir("fsvi-f16-384d");

    let mut v1 = Vec::with_capacity(384);
    let mut v2 = Vec::with_capacity(384);
    for i in 0..384 {
        #[allow(clippy::cast_precision_loss)]
        let angle = (i as f32) * 0.017; // ~1 degree increments
        v1.push(angle.sin());
        v2.push(angle.cos());
    }
    let v1 = normalize_vec(&v1);
    let v2 = normalize_vec(&v2);

    write_fast_index(&dir, &[("doc-a", v1.clone()), ("doc-b", v2)]);

    let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
    let hits = index.search_fast(&v1, 2).expect("search");

    // Self-similarity should be close to 1.0
    let self_sim = hits.iter().find(|h| h.doc_id == "doc-a").unwrap().score;
    assert!(
        (self_sim - 1.0).abs() < 0.005,
        "self-similarity too far from 1.0: {self_sim}"
    );
}

#[test]
fn two_tier_index_fast_and_quality_alignment() {
    // Verify fast and quality indices share document ID namespace
    let dir = temp_dir("two-tier-alignment");

    let fast_records = vec![
        ("shared-1", normalize_vec(&[1.0, 0.0, 0.0, 0.0])),
        ("shared-2", normalize_vec(&[0.0, 1.0, 0.0, 0.0])),
        ("fast-only", normalize_vec(&[0.0, 0.0, 1.0, 0.0])),
    ];
    let quality_records = vec![
        ("shared-1", normalize_vec(&[0.9, 0.1, 0.0, 0.0, 0.0, 0.0])),
        ("shared-2", normalize_vec(&[0.1, 0.9, 0.0, 0.0, 0.0, 0.0])),
        (
            "quality-only",
            normalize_vec(&[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        ),
    ];

    write_fast_index(&dir, &fast_records);
    write_quality_index(&dir, &quality_records);

    let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
    assert!(index.has_quality_index());

    // Fast search
    let query_fast = normalize_vec(&[1.0, 0.0, 0.0, 0.0]);
    let fast_hits = index.search_fast(&query_fast, 3).expect("fast search");
    assert_eq!(fast_hits[0].doc_id, "shared-1");

    // Quality scores for fast-tier hits
    let query_quality = normalize_vec(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let quality_scores = index
        .quality_scores_for_hits(&query_quality, &fast_hits)
        .expect("quality scores");
    // shared-1 should have highest quality score (its quality embedding is close to query)
    assert!(quality_scores[0].unwrap() > quality_scores[1].unwrap());
}

#[test]
fn embedder_named_explicit_paths_search_and_reopen_without_copying() {
    let dir = temp_dir("embedder-explicit-layout");
    let fast_path = dir.join("index-fnv1a-384.fsvi");
    let quality_path = dir.join("index-minilm-384.fsvi");
    let mut fast_a = vec![0.0; 384];
    let mut fast_b = vec![0.0; 384];
    fast_a[0] = 1.0;
    fast_b[1] = 1.0;
    let mut quality_a = vec![0.0; 384];
    let mut quality_b = vec![0.0; 384];
    quality_a[2] = 1.0;
    quality_b[3] = 1.0;
    write_index_at(
        &fast_path,
        "fnv1a-384",
        "hash-fnv1a-modular-v1",
        &[("doc-a", fast_a), ("doc-b", fast_b)],
    );
    write_index_at(
        &quality_path,
        "minilm-384",
        "native-minilm-v1:test",
        &[("doc-a", quality_a), ("doc-b", quality_b)],
    );
    let paths = TwoTierIndexPaths::new(&fast_path).with_quality_index(&quality_path);

    let index =
        TwoTierIndex::open_with_paths(&paths, TwoTierConfig::default()).expect("first open");
    let mut query = vec![0.0; 384];
    query[0] = 1.0;
    let hits = index
        .search_fast(&query, 2)
        .expect("search custom fast tier");
    assert_eq!(hits[0].doc_id, "doc-a");
    assert_eq!(index.fast_embedder_id(), "fnv1a-384");
    assert_eq!(index.quality_embedder_id(), Some("minilm-384"));
    drop(index);

    let reopened = TwoTierIndex::open_with_paths(&paths, TwoTierConfig::default())
        .expect("reopen exact paths");
    assert_eq!(reopened.fast_index_path(), fast_path);
    assert_eq!(reopened.quality_index_path(), Some(quality_path.as_path()));
    assert_eq!(
        reopened.search_fast(&query, 2).expect("search")[0].doc_id,
        "doc-a"
    );
    assert!(!dir.join(VECTOR_INDEX_FAST_FILENAME).exists());
    assert!(!dir.join(VECTOR_INDEX_QUALITY_FILENAME).exists());
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Normalize → Blend pipeline composition
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn blend_applies_normalization_before_combining() {
    // Fast scores on BM25 scale (0-30), quality on cosine scale (0-1).
    // Blend should normalize independently before weighting.
    let fast = vec![hit("a", 30.0, 0), hit("b", 15.0, 1), hit("c", 0.0, 2)];
    let quality = vec![hit("a", 0.3, 0), hit("b", 0.9, 1), hit("c", 0.1, 2)];

    let blended = blend_two_tier(&fast, &quality, 0.7);

    // "b" has low fast-norm (0.5) but high quality-norm (1.0).
    // At alpha=0.7: b = 0.7*1.0 + 0.3*0.5 = 0.85
    // "a" has high fast-norm (1.0) but low quality-norm (~0.25).
    // At alpha=0.7: a = 0.7*0.25 + 0.3*1.0 = 0.475
    let b_score = blended.iter().find(|h| h.doc_id == "b").unwrap().score;
    let a_score = blended.iter().find(|h| h.doc_id == "a").unwrap().score;
    assert!(
        b_score > a_score,
        "quality-heavy doc 'b' should rank above fast-heavy doc 'a' with alpha=0.7"
    );
}

#[test]
fn normalize_then_blend_empty_sets() {
    // One empty set should still produce results from the other
    let fast = vec![hit("a", 1.0, 0), hit("b", 0.5, 1)];
    let quality: Vec<VectorHit> = vec![];

    let blended = blend_two_tier(&fast, &quality, 0.7);
    assert_eq!(blended.len(), 2);
    // With alpha=0.7 and no quality, scores are penalized
    assert!(blended.iter().all(|h| h.score >= 0.0));
}

#[test]
fn normalize_edge_cases_propagate_through_blend() {
    // All identical scores use the robust fallback path and are clamped to [0,1].
    // Here both inputs clamp to 1.0, so blend also yields 1.0.
    let fast = vec![hit("a", 5.0, 0), hit("b", 5.0, 1)];
    let quality = vec![hit("a", 5.0, 0), hit("b", 5.0, 1)];

    let blended = blend_two_tier(&fast, &quality, 0.5);
    // All equal high scores: clamp -> 1.0 each. Blend: 0.5*1.0 + 0.5*1.0 = 1.0.
    for h in &blended {
        assert!(
            (h.score - 1.0).abs() < 1e-5,
            "expected ~1.0, got {}",
            h.score
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. RRF + Blend end-to-end ranking
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rrf_output_feeds_blend_correctly() {
    // Simulate: RRF fuses lexical+semantic → fast hits.
    // Then quality hits arrive → blend produces final ranking.
    let lexical = vec![
        scored("doc-1", 12.5),
        scored("doc-2", 8.0),
        scored("doc-3", 3.0),
    ];
    let semantic = vec![
        hit("doc-2", 0.95, 0),
        hit("doc-1", 0.80, 1),
        hit("doc-4", 0.70, 2),
    ];

    let rrf_config = RrfConfig {
        k: 60.0,
        ..RrfConfig::default()
    };
    let fused = rrf_fuse(&lexical, &semantic, 10, 0, &rrf_config);

    // doc-1 and doc-2 appear in both → higher RRF scores
    let doc1 = fused.iter().find(|h| h.doc_id == "doc-1").unwrap();
    let doc4 = fused.iter().find(|h| h.doc_id == "doc-4").unwrap();
    assert!(doc1.in_both_sources);
    assert!(!doc4.in_both_sources);
    assert!(doc1.rrf_score > doc4.rrf_score);

    // Convert fused hits to VectorHits for blend input (simulating fast-tier)
    #[allow(clippy::cast_possible_truncation)]
    let fast_hits: Vec<VectorHit> = fused
        .iter()
        .enumerate()
        .map(|(i, f)| VectorHit {
            index: i as u32,
            score: f.rrf_score as f32,
            doc_id: f.doc_id.clone(),
        })
        .collect();

    // Quality hits with different ranking
    let quality_hits = vec![
        hit("doc-4", 0.99, 0), // Quality loves doc-4
        hit("doc-2", 0.85, 1),
        hit("doc-1", 0.40, 2),
    ];

    let blended = blend_two_tier(&fast_hits, &quality_hits, 0.7);
    // All docs should be present
    assert!(blended.len() >= 3);
    // Blend should be deterministic
    let blended2 = blend_two_tier(&fast_hits, &quality_hits, 0.7);
    for (a, b) in blended.iter().zip(blended2.iter()) {
        assert_eq!(a.doc_id, b.doc_id);
        assert!((a.score - b.score).abs() < 1e-6);
    }
}

#[test]
fn rrf_candidate_count_interacts_with_config() {
    // Verify candidate_count respects multiplier and saturates
    let count = candidate_count(10, 0, 3);
    assert_eq!(count, 30);

    let count_with_offset = candidate_count(10, 5, 3);
    assert_eq!(count_with_offset, 45);

    // Saturation at usize boundary
    let count_large = candidate_count(usize::MAX / 2, 0, 3);
    assert!(count_large > 0); // Should not panic on overflow
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Queue → canonicalization → content hash determinism
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn queue_canonicalization_produces_consistent_hashes() {
    // Same text with different Unicode representations → same hash after NFC canonicalization
    let queue = EmbeddingQueue::new(
        EmbeddingQueueConfig {
            capacity: 100,
            batch_size: 32,
            max_retries: 3,
        },
        Box::new(DefaultCanonicalizer::default()),
    );

    // NFC-decomposed: e + combining acute accent
    queue
        .submit(EmbeddingRequest {
            doc_id: "doc-1".into(),
            text: "caf\u{0065}\u{0301}".to_owned(),
            metadata: None,
            submitted_at: Instant::now(),
        })
        .unwrap();

    // NFC-precomposed: é
    queue
        .submit(EmbeddingRequest {
            doc_id: "doc-2".into(),
            text: "caf\u{00e9}".to_owned(),
            metadata: None,
            submitted_at: Instant::now(),
        })
        .unwrap();

    let batch = queue.drain_batch();
    assert_eq!(batch.len(), 2);
    // After NFC normalization, both forms of é should produce identical hashes
    assert_eq!(
        batch[0].content_hash, batch[1].content_hash,
        "NFC-equivalent texts should hash identically"
    );

    // Different text should produce a different hash
    let queue2 = EmbeddingQueue::new(
        EmbeddingQueueConfig {
            capacity: 100,
            batch_size: 32,
            max_retries: 3,
        },
        Box::new(DefaultCanonicalizer::default()),
    );

    queue2
        .submit(EmbeddingRequest {
            doc_id: "doc-3".into(),
            text: "completely different".to_owned(),
            metadata: None,
            submitted_at: Instant::now(),
        })
        .unwrap();

    let batch2 = queue2.drain_batch();
    assert_ne!(
        batch[0].content_hash, batch2[0].content_hash,
        "different texts should produce different hashes"
    );
}

#[test]
fn queue_dedup_survives_drain_rebuild_cycle() {
    // Submit doc → drain → record_embedded → resubmit same → should skip
    let queue = EmbeddingQueue::new(
        EmbeddingQueueConfig {
            capacity: 100,
            batch_size: 32,
            max_retries: 3,
        },
        Box::new(DefaultCanonicalizer::default()),
    );

    // First submission
    queue
        .submit(EmbeddingRequest {
            doc_id: "doc-1".into(),
            text: "Important document content".to_owned(),
            metadata: None,
            submitted_at: Instant::now(),
        })
        .unwrap();

    let batch = queue.drain_batch();
    assert_eq!(batch.len(), 1);

    // Record as embedded
    queue.record_embedded(&batch[0].doc_id, &batch[0].content_hash);

    // Re-submit identical content → should skip
    let outcome = queue
        .submit(EmbeddingRequest {
            doc_id: "doc-1".into(),
            text: "Important document content".to_owned(),
            metadata: None,
            submitted_at: Instant::now(),
        })
        .unwrap();
    assert_eq!(outcome, JobOutcome::SkippedUnchanged);

    // Re-submit modified content → should enqueue
    let outcome = queue
        .submit(EmbeddingRequest {
            doc_id: "doc-1".into(),
            text: "Modified document content".to_owned(),
            metadata: None,
            submitted_at: Instant::now(),
        })
        .unwrap();
    assert_eq!(outcome, JobOutcome::Succeeded);
    assert_eq!(queue.pending_count(), 1);
}

#[test]
fn queue_backpressure_does_not_corrupt_dedup_state() {
    let queue = EmbeddingQueue::new(
        EmbeddingQueueConfig {
            capacity: 2,
            batch_size: 32,
            max_retries: 3,
        },
        Box::new(DefaultCanonicalizer::default()),
    );

    // Fill queue
    queue
        .submit(EmbeddingRequest {
            doc_id: "doc-1".into(),
            text: "First".to_owned(),
            metadata: None,
            submitted_at: Instant::now(),
        })
        .unwrap();
    queue
        .submit(EmbeddingRequest {
            doc_id: "doc-2".into(),
            text: "Second".to_owned(),
            metadata: None,
            submitted_at: Instant::now(),
        })
        .unwrap();

    // Queue full → backpressure
    let err = queue
        .submit(EmbeddingRequest {
            doc_id: "doc-3".into(),
            text: "Third".to_owned(),
            metadata: None,
            submitted_at: Instant::now(),
        })
        .unwrap_err();
    assert!(matches!(err, SearchError::QueueFull { .. }));

    // Drain and record
    let batch = queue.drain_batch();
    for job in &batch {
        queue.record_embedded(&job.doc_id, &job.content_hash);
    }

    // Now doc-3 should work, and doc-1/doc-2 should be skipped
    let outcome = queue
        .submit(EmbeddingRequest {
            doc_id: "doc-1".into(),
            text: "First".to_owned(),
            metadata: None,
            submitted_at: Instant::now(),
        })
        .unwrap();
    assert_eq!(outcome, JobOutcome::SkippedUnchanged);

    let outcome = queue
        .submit(EmbeddingRequest {
            doc_id: "doc-3".into(),
            text: "Third".to_owned(),
            metadata: None,
            submitted_at: Instant::now(),
        })
        .unwrap();
    assert_eq!(outcome, JobOutcome::Succeeded);
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Cache + staleness + index reload lifecycle
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cache_detects_staleness_after_index_growth() {
    let dir = temp_dir("cache-staleness-growth");
    let records = vec![
        ("doc-a", normalize_vec(&[1.0, 0.0, 0.0, 0.0])),
        ("doc-b", normalize_vec(&[0.0, 1.0, 0.0, 0.0])),
    ];
    write_fast_index(&dir, &records);

    // Write sentinel matching current index
    IndexSentinel {
        version: SENTINEL_VERSION,
        built_at: "2026-01-15T10:00:00Z".to_owned(),
        source_count: 2,
        source_hash: None,
        fast_embedder: "potion-128M".to_owned(),
        quality_embedder: None,
        fast_dimension: 4,
        quality_dimension: None,
    }
    .write_to(&dir)
    .unwrap();

    let cache = IndexCache::open(
        &dir,
        TwoTierConfig::default(),
        Box::new(SentinelFileDetector::new().with_expected_count(5)),
    )
    .expect("open");

    // Index has 2 docs but caller expects 5 → stale
    assert!(cache.is_stale().expect("check"));

    let report = cache.check_staleness().expect("report");
    assert!(report.is_stale);
    assert_eq!(report.estimated_source_count, Some(5));
}

#[test]
fn cache_reload_updates_search_results() {
    let dir = temp_dir("cache-reload-search");

    // Initial index: doc-a scores highest for [1,0,0,0]
    write_fast_index(
        &dir,
        &[
            ("doc-a", normalize_vec(&[1.0, 0.0, 0.0, 0.0])),
            ("doc-b", normalize_vec(&[0.0, 1.0, 0.0, 0.0])),
        ],
    );

    let cache = IndexCache::open(
        &dir,
        TwoTierConfig::default(),
        Box::new(SentinelFileDetector::new()),
    )
    .expect("open");

    let old = cache.current();
    let query = normalize_vec(&[0.0, 1.0, 0.0, 0.0]);
    let old_hits = old.search_fast(&query, 2).expect("search");
    assert_eq!(old_hits[0].doc_id, "doc-b");

    // Rebuild index with doc-c as the best match
    write_fast_index(
        &dir,
        &[
            ("doc-a", normalize_vec(&[1.0, 0.0, 0.0, 0.0])),
            ("doc-c", normalize_vec(&[0.0, 1.0, 0.0, 0.0])), // replaces doc-b
        ],
    );

    cache.reload().expect("reload");
    let fresh = cache.current();
    let new_hits = fresh.search_fast(&query, 2).expect("search");
    assert_eq!(new_hits[0].doc_id, "doc-c");

    // Old reference still returns old results
    let old_hits_again = old.search_fast(&query, 2).expect("old still works");
    assert_eq!(old_hits_again[0].doc_id, "doc-b");
}

#[test]
fn cache_sentinel_hash_change_detects_staleness() {
    let dir = temp_dir("cache-hash-change");
    write_fast_index(&dir, &[("doc-a", normalize_vec(&[1.0, 0.0, 0.0, 0.0]))]);

    IndexSentinel {
        version: SENTINEL_VERSION,
        built_at: "2026-01-15T10:00:00Z".to_owned(),
        source_count: 1,
        source_hash: Some("sha256:aaa".to_owned()),
        fast_embedder: "potion-128M".to_owned(),
        quality_embedder: None,
        fast_dimension: 4,
        quality_dimension: None,
    }
    .write_to(&dir)
    .unwrap();

    let cache = IndexCache::open(
        &dir,
        TwoTierConfig::default(),
        Box::new(SentinelFileDetector::new().with_expected_hash("sha256:bbb")),
    )
    .expect("open");

    // Hash mismatch → stale
    assert!(cache.is_stale().expect("check"));
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Error propagation across crate boundaries
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dimension_mismatch_from_search_through_index() {
    let dir = temp_dir("dim-mismatch");
    write_fast_index(&dir, &[("doc-a", normalize_vec(&[1.0, 0.0, 0.0, 0.0]))]);

    let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");

    // Query with wrong dimension (8 instead of 4)
    let wrong_query = vec![1.0; 8];
    let err = index
        .search_fast(&wrong_query, 10)
        .expect_err("should fail");
    assert!(
        matches!(
            err,
            SearchError::DimensionMismatch {
                expected: 4,
                found: 8
            }
        ),
        "expected DimensionMismatch, got: {err:?}"
    );
}

#[test]
fn index_not_found_propagates_through_cache() {
    let dir = std::env::temp_dir().join("frankensearch-xcomp-nonexistent-dir");
    let err = IndexCache::open(
        &dir,
        TwoTierConfig::default(),
        Box::new(SentinelFileDetector::new()),
    )
    .expect_err("should fail");
    let error_debug = format!("{err:?}");
    let paths = if let SearchError::IndexCandidatesNotFound { paths } = err {
        paths
    } else {
        Vec::new()
    };
    assert_eq!(paths.len(), 2, "unexpected error: {error_debug}");
    assert!(paths.iter().any(|path| path.ends_with("vector.fast.idx")));
    assert!(paths.iter().any(|path| path.ends_with("vector.idx")));
}

#[test]
fn corrupted_sentinel_returns_config_error() {
    let dir = temp_dir("corrupt-sentinel");
    write_fast_index(&dir, &[("doc-a", normalize_vec(&[1.0, 0.0, 0.0, 0.0]))]);

    // Write invalid JSON as sentinel
    std::fs::write(
        dir.join(".frankensearch_index_meta"),
        "this is not valid json",
    )
    .expect("write corrupt sentinel");

    let cache = IndexCache::open(
        &dir,
        TwoTierConfig::default(),
        Box::new(SentinelFileDetector::new()),
    )
    .expect("cache should open despite corrupt sentinel");

    // Staleness check should fail with config error (malformed JSON)
    let err = cache.check_staleness().expect_err("should fail");
    assert!(
        matches!(err, SearchError::InvalidConfig { .. }),
        "expected InvalidConfig for corrupt sentinel, got: {err:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. Config validation interactions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn config_serde_roundtrip_preserves_all_fields() {
    let config = TwoTierConfig {
        quality_weight: 0.8,
        rrf_k: 30.0,
        candidate_multiplier: 5,
        quality_timeout_ms: 1000,
        fast_only: true,
        explain: true,
        hnsw_ef_search: 200,
        hnsw_ef_construction: 400,
        hnsw_m: 32,
        mrl_search_dims: 128,
        mrl_rescore_top_k: 50,
        ..Default::default()
    };

    let json = serde_json::to_string(&config).expect("serialize");
    let decoded: TwoTierConfig = serde_json::from_str(&json).expect("deserialize");

    assert!((decoded.quality_weight - 0.8).abs() < 1e-10);
    assert!((decoded.rrf_k - 30.0).abs() < 1e-10);
    assert_eq!(decoded.candidate_multiplier, 5);
    assert_eq!(decoded.quality_timeout_ms, 1000);
    assert!(decoded.fast_only);
    assert!(decoded.explain);
    assert_eq!(decoded.hnsw_ef_search, 200);
    assert_eq!(decoded.hnsw_m, 32);
    assert_eq!(decoded.mrl_search_dims, 128);
    assert_eq!(decoded.mrl_rescore_top_k, 50);
    // metrics_exporter is #[serde(skip)] so should be None
    assert!(decoded.metrics_exporter.is_none());
}

#[test]
#[allow(clippy::field_reassign_with_default)]
fn metrics_tracks_all_phases() {
    let mut metrics = TwoTierMetrics::default();

    // Simulate phase 1
    metrics.fast_embed_ms = 0.57;
    metrics.vector_search_ms = 3.2;
    metrics.lexical_search_ms = 1.1;
    metrics.rrf_fusion_ms = 0.3;
    metrics.phase1_total_ms = 5.17;
    metrics.fast_embedder_id = Some("potion-128M".into());
    metrics.semantic_candidates = 30;
    metrics.lexical_candidates = 50;

    // Simulate phase 2
    metrics.quality_embed_ms = 128.0;
    metrics.quality_search_ms = 3.5;
    metrics.blend_ms = 0.2;
    metrics.rerank_ms = 15.0;
    metrics.phase2_total_ms = 146.7;
    metrics.quality_embedder_id = Some("MiniLM-L6-v2".into());

    // Rank changes
    metrics.rank_changes = RankChanges {
        promoted: 3,
        demoted: 2,
        stable: 5,
    };
    metrics.kendall_tau = Some(0.75);

    // Verify serde roundtrip preserves all fields
    let json = serde_json::to_string(&metrics).expect("serialize");
    let decoded: TwoTierMetrics = serde_json::from_str(&json).expect("deserialize");

    assert!((decoded.fast_embed_ms - 0.57).abs() < 1e-10);
    assert!((decoded.phase2_total_ms - 146.7).abs() < 1e-10);
    assert_eq!(decoded.rank_changes.promoted, 3);
    assert_eq!(decoded.rank_changes.total(), 10);
    assert_eq!(decoded.kendall_tau, Some(0.75));
    assert_eq!(decoded.semantic_candidates, 30);
    assert_eq!(decoded.lexical_candidates, 50);
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. Score edge cases: NaN, all-zero, all-identical
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rrf_with_all_zero_scores_still_ranks_by_position() {
    let lexical = vec![scored("a", 0.0), scored("b", 0.0), scored("c", 0.0)];
    let semantic = vec![hit("b", 0.0, 0), hit("c", 0.0, 1), hit("a", 0.0, 2)];

    let config = RrfConfig {
        k: 60.0,
        ..RrfConfig::default()
    };
    let fused = rrf_fuse(&lexical, &semantic, 10, 0, &config);

    // All docs should be present with valid (non-NaN) scores
    assert_eq!(fused.len(), 3);
    assert!(fused.iter().all(|h| h.rrf_score.is_finite()));

    // Docs in both sources should still score higher
    let a_score = fused.iter().find(|h| h.doc_id == "a").unwrap().rrf_score;
    let b_score = fused.iter().find(|h| h.doc_id == "b").unwrap().rrf_score;
    assert!(a_score > 0.0);
    assert!(b_score > 0.0);
}

#[test]
fn blend_with_nan_scores_sanitized() {
    let fast = vec![hit("a", f32::NAN, 0), hit("b", 1.0, 1)];
    let quality = vec![hit("a", 0.5, 0), hit("b", f32::NAN, 1)];

    let blended = blend_two_tier(&fast, &quality, 0.5);
    // All output scores should be finite
    assert!(
        blended.iter().all(|h| h.score.is_finite()),
        "NaN should be sanitized in blend output"
    );
}

#[test]
fn normalize_single_element() {
    let mut scores = vec![42.0];
    min_max_normalize(&mut scores);
    // Single element → degenerate case → 0.5
    assert!((scores[0] - 0.5).abs() < 1e-6);

    let mut z_scores = vec![42.0];
    z_score_normalize(&mut z_scores);
    assert!((z_scores[0] - 0.5).abs() < 1e-6);
}

#[test]
fn normalize_negative_scores() {
    let mut scores = vec![-10.0, -5.0, 0.0, 5.0, 10.0];
    min_max_normalize(&mut scores);
    assert!((scores[0] - 0.0).abs() < 1e-6); // min → 0
    assert!((scores[4] - 1.0).abs() < 1e-6); // max → 1
    assert!((scores[2] - 0.5).abs() < 1e-6); // midpoint → 0.5
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. Rank change tracking across blend phases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rank_changes_reflect_blend_reordering() {
    // Phase 1: fast-only ranking
    let initial = vec![hit("a", 0.9, 0), hit("b", 0.7, 1), hit("c", 0.5, 2)];

    // Phase 2: after quality blend, c moves to top
    let refined = vec![hit("c", 0.95, 2), hit("a", 0.85, 0), hit("b", 0.3, 1)];

    let changes = compute_rank_changes(&initial, &refined);
    assert_eq!(changes.promoted, 1); // c moved up
    assert_eq!(changes.demoted, 2); // a, b moved down
    assert_eq!(changes.stable, 0);
    assert_eq!(changes.total(), 3);
}

#[test]
fn kendall_tau_detects_correlation_after_blend() {
    // Nearly identical rankings → tau close to 1.0
    let initial = vec![hit("a", 0.9, 0), hit("b", 0.7, 1), hit("c", 0.5, 2)];
    let similar = vec![hit("a", 0.95, 0), hit("b", 0.72, 1), hit("c", 0.48, 2)];
    let tau = kendall_tau(&initial, &similar).expect("tau");
    assert!((tau - 1.0).abs() < f64::EPSILON);

    // Completely reversed rankings → tau = -1.0
    let reversed = vec![hit("c", 0.99, 2), hit("b", 0.72, 1), hit("a", 0.1, 0)];
    let tau_rev = kendall_tau(&initial, &reversed).expect("tau");
    assert!((tau_rev + 1.0).abs() < f64::EPSILON);

    // Fewer than 2 common docs → None
    let disjoint = vec![hit("x", 0.9, 3), hit("y", 0.7, 4)];
    assert!(kendall_tau(&initial, &disjoint).is_none());
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. Hash embedder → index → search end-to-end
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn hash_embedder_vectors_survive_fsvi_roundtrip() {
    let embedder = HashEmbedder::new(256, HashAlgorithm::FnvModular);

    // Embed two documents
    let v1 = embedder.embed_sync("distributed consensus algorithms");
    let v2 = embedder.embed_sync("machine learning optimization");

    assert_eq!(v1.len(), 256);
    assert_eq!(v2.len(), 256);

    // Write to FSVI index
    let dir = temp_dir("hash-embed-roundtrip");
    write_fast_index(&dir, &[("doc-1", v1.clone()), ("doc-2", v2)]);

    // Search should find doc-1 closer to its own embedding
    let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
    let hits = index.search_fast(&v1, 2).expect("search");
    assert_eq!(hits[0].doc_id, "doc-1");
    assert!(hits[0].score > hits[1].score);
}

#[test]
fn hash_embedder_deterministic_across_invocations() {
    let embedder = HashEmbedder::new(384, HashAlgorithm::FnvModular);
    let text = "Frankensearch hybrid search with RRF fusion";

    let v1 = embedder.embed_sync(text);
    let v2 = embedder.embed_sync(text);
    assert_eq!(v1, v2, "hash embedder must be deterministic");

    // Different text → different embedding
    let v3 = embedder.embed_sync("something completely different");
    assert_ne!(v1, v3);
}

// ═══════════════════════════════════════════════════════════════════════════
// 11. RRF tie-breaking determinism
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rrf_deterministic_ordering_with_ties() {
    let lexical = vec![scored("a", 5.0), scored("b", 5.0), scored("c", 5.0)];
    let semantic = vec![hit("a", 0.9, 0), hit("b", 0.9, 1), hit("c", 0.9, 2)];

    let config = RrfConfig {
        k: 60.0,
        ..RrfConfig::default()
    };

    // Run twice → same output
    let fused1 = rrf_fuse(&lexical, &semantic, 10, 0, &config);
    let fused2 = rrf_fuse(&lexical, &semantic, 10, 0, &config);

    assert_eq!(fused1.len(), fused2.len());
    for (a, b) in fused1.iter().zip(fused2.iter()) {
        assert_eq!(a.doc_id, b.doc_id);
        assert!((a.rrf_score - b.rrf_score).abs() < 1e-10);
    }
}

#[test]
fn rrf_lexical_only_and_semantic_only() {
    let config = RrfConfig {
        k: 60.0,
        ..RrfConfig::default()
    };

    // Lexical-only
    let lexical = vec![scored("a", 10.0), scored("b", 5.0)];
    let semantic: Vec<VectorHit> = vec![];
    let fused = rrf_fuse(&lexical, &semantic, 10, 0, &config);
    assert_eq!(fused.len(), 2);
    assert!(!fused[0].in_both_sources);

    // Semantic-only
    let lexical: Vec<ScoredResult> = vec![];
    let semantic = vec![hit("x", 0.9, 0), hit("y", 0.8, 1)];
    let fused = rrf_fuse(&lexical, &semantic, 10, 0, &config);
    assert_eq!(fused.len(), 2);
    assert!(!fused[0].in_both_sources);
}

#[test]
fn rrf_offset_and_limit_pagination() {
    let config = RrfConfig {
        k: 60.0,
        ..RrfConfig::default()
    };
    let lexical: Vec<ScoredResult> = (0..10)
        .map(|i| {
            scored(
                &format!("doc-{i}"),
                10.0 - f32::from(u8::try_from(i).unwrap()),
            )
        })
        .collect();
    let semantic: Vec<VectorHit> = vec![];

    let page1 = rrf_fuse(&lexical, &semantic, 3, 0, &config);
    let page2 = rrf_fuse(&lexical, &semantic, 3, 3, &config);
    let all = rrf_fuse(&lexical, &semantic, 6, 0, &config);

    assert_eq!(page1.len(), 3);
    assert_eq!(page2.len(), 3);
    // page1 + page2 should equal first 6 of all
    for (i, item) in page1.iter().chain(page2.iter()).enumerate() {
        assert_eq!(item.doc_id, all[i].doc_id);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 12. Calibration integration coverage
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn score_calibration_maps_rrf_scores_to_probabilities() {
    let lexical = vec![
        scored("a", 8.0),
        scored("b", 7.0),
        scored("c", 6.0),
        scored("d", 5.0),
    ];
    let semantic = vec![
        hit("a", 0.95, 0),
        hit("c", 0.90, 2),
        hit("b", 0.75, 1),
        hit("d", 0.40, 3),
    ];

    let fused = rrf_fuse(
        &lexical,
        &semantic,
        10,
        0,
        &RrfConfig {
            k: 60.0,
            ..RrfConfig::default()
        },
    );
    let raw_scores: Vec<f64> = fused.iter().map(|h| h.rrf_score).collect();
    let labels = vec![1.0, 1.0, 0.0, 0.0];

    let (calibrated, summary) =
        calibrate_scores_with_labels(&PlattScaling::new(14.0, -0.15), &raw_scores, &labels, 8);

    assert_eq!(calibrated.len(), fused.len());
    assert_eq!(summary.count, fused.len());
    assert!(calibrated.iter().all(|s| (0.0..=1.0).contains(s)));
}

#[test]
fn isotonic_calibration_improves_ece_on_search_outputs() {
    let dir = temp_dir("calibration-search-output");
    write_fast_index(
        &dir,
        &[
            ("doc-a", normalize_vec(&[1.0, 0.0, 0.0, 0.0])),
            ("doc-b", normalize_vec(&[0.9, 0.1, 0.0, 0.0])),
            ("doc-c", normalize_vec(&[0.7, 0.3, 0.0, 0.0])),
            ("doc-d", normalize_vec(&[0.2, 0.8, 0.0, 0.0])),
        ],
    );
    let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
    let query = normalize_vec(&[1.0, 0.0, 0.0, 0.0]);
    let hits = index.search_fast(&query, 4).expect("search");

    // Deliberately invert the raw signal to simulate a badly miscalibrated scorer.
    // This gives us a stable, realistic integration fixture where isotonic fitting
    // should improve calibration error.
    let raw_scores: Vec<f64> = hits.iter().map(|h| 1.0 - f64::from(h.score)).collect();
    let labels = vec![1.0, 1.0, 0.0, 0.0];
    let bounded_raw: Vec<f64> = raw_scores.iter().map(|s| s.clamp(0.0, 1.0)).collect();
    let ece_before = compute_ece(&bounded_raw, &labels, 4);

    let isotonic = IsotonicRegression::fit(&raw_scores, &labels);
    let (calibrated, summary) = calibrate_scores_with_labels(&isotonic, &raw_scores, &labels, 4);
    let ece_after = compute_ece(&calibrated, &labels, 4);

    assert!(ece_after <= ece_before + 1e-12);
    assert!(summary.ece_after <= summary.ece_before + 1e-12);
}

#[test]
fn identity_calibration_is_passthrough_for_valid_probabilities() {
    let raw_scores = vec![0.05, 0.25, 0.5, 0.9];
    let labels = vec![0.0, 0.0, 1.0, 1.0];
    let bounded = raw_scores.clone();

    let (calibrated, summary) = calibrate_scores_with_labels(&Identity, &raw_scores, &labels, 4);
    assert_eq!(calibrated, bounded);
    assert_eq!(summary.count, 4);
}

// ═══════════════════════════════════════════════════════════════════════════
// 13. Conformal prediction integration coverage
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn conformal_required_k_tracks_requested_coverage() {
    let calibration =
        ConformalSearchCalibration::calibrate(&[1, 2, 2, 3, 5, 8]).expect("calibrate");

    let strict = calibration.required_k(0.01);
    let relaxed = calibration.required_k(0.25);
    assert!(strict >= relaxed);
    assert!(strict >= 1);
}

#[test]
fn conformal_p_value_penalizes_worse_ranks() {
    let calibration =
        ConformalSearchCalibration::calibrate(&[1, 2, 3, 3, 5, 8]).expect("calibrate");
    let top_rank = calibration.p_value(1);
    let poor_rank = calibration.p_value(8);

    assert!((0.0..=1.0).contains(&top_rank));
    assert!((0.0..=1.0).contains(&poor_rank));
    assert!(poor_rank <= top_rank);
}

#[test]
fn adaptive_conformal_state_updates_alpha_with_observed_error() {
    let calibration =
        ConformalSearchCalibration::calibrate(&[1, 2, 2, 4, 6, 9]).expect("calibrate");
    let mut state = AdaptiveConformalState::new(0.10, 0.20).expect("state");
    let update = state.update(0.30, &calibration).expect("update");

    assert!(update.alpha_after > update.alpha_before);
    assert!(update.required_k >= 1);
}

#[test]
fn conformal_heldout_coverage_is_near_target() {
    let mut calibration = Vec::with_capacity(200);
    for _ in 0..10 {
        calibration.extend(1..=20);
    }
    let calibrator = ConformalSearchCalibration::calibrate(&calibration).expect("calibrate");

    let alpha = 0.10;
    let required_k = calibrator.required_k(alpha);
    let heldout: Vec<usize> = (0..120).map(|i| (i % 20) + 1).collect();
    let covered = heldout.iter().filter(|&&rank| rank <= required_k).count();
    #[allow(clippy::cast_precision_loss)]
    let empirical_coverage = covered as f32 / heldout.len() as f32;

    assert!(
        empirical_coverage >= (1.0 - alpha - 0.03),
        "empirical coverage {empirical_coverage:.3} below tolerance"
    );
}

#[test]
fn mondrian_conformal_uses_global_fallback_for_sparse_class() {
    let examples = vec![
        ("src/main.rs".to_owned(), 1),
        ("bd-123".to_owned(), 2),
        ("vector search".to_owned(), 4),
        ("error handling".to_owned(), 5),
        ("hybrid ranking".to_owned(), 6),
        ("fusion behavior".to_owned(), 7),
    ];
    let mondrian = MondrianConformalCalibration::calibrate(&examples, 3).expect("calibrate");

    let global_k = mondrian.global().required_k(0.20);
    let identifier_k = mondrian.required_k("src/lib.rs", 0.20);
    assert_eq!(identifier_k, global_k);
}

// ═══════════════════════════════════════════════════════════════════════════
// 14. SearchError variant coverage
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn search_error_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SearchError>();
}

#[test]
fn search_error_display_messages_are_actionable() {
    let errors = vec![
        SearchError::DimensionMismatch {
            expected: 256,
            found: 384,
        },
        SearchError::IndexNotFound {
            path: PathBuf::from("/tmp/missing.fsvi"),
        },
        SearchError::QueueFull {
            pending: 100,
            capacity: 100,
        },
        SearchError::EmbedderUnavailable {
            model: "MiniLM".into(),
            reason: "model files missing".into(),
        },
    ];

    for err in &errors {
        let msg = err.to_string();
        assert!(
            !msg.is_empty(),
            "error display should not be empty: {err:?}"
        );
        // All messages should provide actionable guidance
        assert!(
            msg.len() > 20,
            "error message too short to be actionable: {msg}"
        );
    }
}

#[test]
fn io_error_converts_to_search_error() {
    let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
    let search_err: SearchError = io_err.into();
    assert!(matches!(search_err, SearchError::Io(_)));
    assert!(search_err.to_string().contains("access denied"));
}

// ==== Four-engine composite generation receipts (bd-7hvtf) ====
//
// bd-z4zr3's acceptance says the join must be proven "from the REAL producers
// -- FSVI witness, Quill descriptor, native HNSW receipt, and
// GenerationManifest, over one document set", and notes that every join test in
// the tree instead pairs real receipts with hand-built stand-ins for the other
// roles, "which is exactly why the split went unnoticed". This module removes
// the stand-ins one role at a time.
//
// SLICE 1 (this commit): the two roles that share one physical artifact. The
// ANN graph is built from the SAME Arc<ValidatedFsviBytes> the vector receipt
// witnesses, so agreement between them is a real property of one image rather
// than a coincidence between two fixtures.
//
// This slice does NOT call ExactGenerationComponentsV1::admit -- that takes all
// four roles and lexical is not Optional, so the join and its per-role drift
// controls arrive with the Quill and metadata roles in slice 2.
#[cfg(feature = "ann")]
mod four_engine_generation_receipts {
    use std::sync::Arc;

    use frankensearch::index::exact_component_adapters::{
        ann_component_receipt, vector_component_receipt,
    };
    use frankensearch::index::native_hnsw::{HnswParams, ValidatedNativeHnsw};
    use frankensearch::index::{FsviV2IdentityBinding, ValidatedFsviBytes, VectorIndex};
    use frankensearch_core::generation::{
        ArtifactGenerationIdentityV1, CommitRange, EmbeddingIdentityBundleV1, GenerationManifest,
        MANIFEST_SCHEMA_VERSION, QuantizationFormat, SourceCheckpointV1, compute_manifest_hash,
    };

    /// One document corpus, shared by every role.
    ///
    /// This is the WRITER's insertion order only. The canonical ordered docset
    /// every role attests is [`generation_order`] — the sealed image's stored
    /// order — and the docset digest is order-sensitive, so an engine that
    /// reordered live documents must not agree with the anchor by accident
    /// (pinned by the reversed-order rejection below).
    const DOCUMENTS: [&str; 3] = ["doc-alpha", "doc-beta", "doc-gamma"];

    const VECTORS: [[f32; 4]; 3] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ];

    fn binding(sequence: u64, nonce: u8) -> FsviV2IdentityBinding {
        let mut identity = EmbeddingIdentityBundleV1::explicit_test_model("7hvtf-model", 4);
        identity.storage.format.clear();
        identity.storage.format.push_str("fsvi-v2");
        identity.storage.quantization = QuantizationFormat::F16;
        identity.storage.endianness.clear();
        identity.storage.endianness.push_str("little-endian");
        let generation = ArtifactGenerationIdentityV1::new(sequence, [nonce; 16])
            .expect("valid 7hvtf generation identity");
        FsviV2IdentityBinding::new(generation, identity.freeze().expect("frozen identity"))
            .expect("valid 7hvtf FSVI v2 binding")
    }

    /// Write and admit a real FSVI v2 image over the shared document set.
    fn fsvi_owner(directory: &std::path::Path) -> Arc<ValidatedFsviBytes> {
        let path = directory.join("current.fsvi");
        let bound = binding(7, 0x7b);
        let mut writer = VectorIndex::create_v2(&path, bound.clone()).expect("FSVI v2 writer");
        for (id, vector) in DOCUMENTS.iter().zip(VECTORS.iter()) {
            writer.write_record(id, vector).expect("write FSVI record");
        }
        writer.finish().expect("seal the FSVI image");
        Arc::new(ValidatedFsviBytes::open_published(&path, &bound).expect("admit the sealed image"))
    }

    /// The owner's live document ids in exact generation order.
    ///
    /// A sealed FSVI v2 image stores its records sorted by
    /// `(doc_id_hash, doc_id)` — generation order is NOT insertion order —
    /// and the witness digest is taken over that stored order. Every role is
    /// fed this sequence: the vector/ANN adapters authenticate against it,
    /// and Quill's snapshot receipt materializes its docset in indexing
    /// order, so indexing in this same order is what makes all four roles
    /// agree on one canonical docset.
    fn generation_order(owner: &ValidatedFsviBytes) -> Vec<String> {
        let rows = usize::try_from(owner.witness().record_count).expect("record count fits");
        (0..rows)
            .map(|row| owner.doc_id_at(row).expect("validated row id").to_owned())
            .collect()
    }

    /// A real `GenerationManifest` over the shared document set.
    ///
    /// `for_metadata_manifest` requires `total_documents` to equal the docset
    /// length and DERIVES the component checkpoint from `commit_range`, so this
    /// is the only role whose drift is expressed by moving the manifest rather
    /// than by passing a different checkpoint.
    fn manifest_over(range: CommitRange) -> GenerationManifest {
        let mut manifest = GenerationManifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            generation_id: "7hvtf-gen-001".to_owned(),
            manifest_hash: String::new(),
            commit_range: range,
            build_started_at: 1_700_000_000_000,
            build_completed_at: 1_700_000_060_000,
            embedders: std::collections::BTreeMap::new(),
            vector_artifacts: Vec::new(),
            lexical_artifacts: Vec::new(),
            repair_descriptors: Vec::new(),
            activation_invariants: Vec::new(),
            total_documents: DOCUMENTS.len() as u64,
            metadata: std::collections::BTreeMap::new(),
        };
        manifest.manifest_hash =
            compute_manifest_hash(&manifest).expect("hash the 7hvtf manifest image");
        manifest
    }

    #[test]
    fn the_vector_and_ann_roles_agree_because_they_witness_one_image() {
        let directory = tempfile::tempdir().expect("7hvtf publication directory");
        let owner = fsvi_owner(directory.path());
        let order = generation_order(&owner);
        let checkpoint = SourceCheckpointV1::derive(&CommitRange { low: 1, high: 9 });

        let vector = vector_component_receipt(owner.witness(), order.clone(), checkpoint)
            .expect("vector receipt from the real FSVI witness");

        let graph =
            ValidatedNativeHnsw::build(Arc::clone(&owner), HnswParams::default(), 0x7b_5eed)
                .expect("native HNSW graph over the admitted owner");
        let graph_receipt = graph
            .save(&directory.path().join("current.fshnsw"))
            .expect("save the graph and mint its receipt");
        let ann = ann_component_receipt(&graph_receipt, order.clone(), checkpoint)
            .expect("ANN receipt from the real graph receipt");

        assert_eq!(
            vector.docset_digest, ann.docset_digest,
            "two roles over one image and one ordered docset must agree on the digest"
        );
        assert_eq!(vector.source_checkpoint, ann.source_checkpoint);
        assert_ne!(
            vector.bytes.sha256, ann.bytes.sha256,
            "the ANN component's identity is its own graph file, not the FSVI image -- binding it \
             to the image would make a rebuilt graph indistinguishable from an unchanged one"
        );

        // The caller cannot use the adapter as an oracle for inventing a new
        // canonical preimage. Reversing the IDs must fail against the
        // engine-local digest authenticated by the persisted graph receipt,
        // before a contradictory component reaches the composite join.
        let mut reordered = order.clone();
        reordered.reverse();
        assert!(
            ann_component_receipt(&graph_receipt, reordered, checkpoint).is_err(),
            "an ANN adapter must reject identifiers not witnessed by the graph's FSVI generation"
        );
    }

    /// Slice 2: all four roles, every one of them from a real producer, joined
    /// by the contract that will admit them in production.
    ///
    /// The vector and ANN roles come from one written FSVI image and a graph
    /// built over that same admitted owner (slice 1). The lexical role comes
    /// from a Quill index that actually indexed and committed the same
    /// documents. The metadata role comes from a `GenerationManifest` and DERIVES
    /// its checkpoint from the manifest's own commit range rather than
    /// accepting one, which is the fail-closed property bd-uu0ly chose and
    /// bd-z4zr3 preserved.
    ///
    /// This is the join bd-z4zr3's acceptance asked for and could not build
    /// from inside one crate: "a four-role join built from the REAL producers
    /// -- FSVI witness, Quill descriptor, native HNSW receipt, and
    /// `GenerationManifest`, over one document set -- admits."
    #[cfg(feature = "quill")]
    #[test]
    fn four_real_producers_admit_as_one_generation() {
        use frankensearch::quill::{QuillConfig, QuillIndex};
        use frankensearch_core::generation::{
            CanonicalDocsetV1, ExactComponentReceiptV1, ExactGenerationComponentsV1,
        };
        use frankensearch_core::types::IndexableDocument;

        let directory = tempfile::tempdir().expect("7hvtf publication directory");
        let owner = fsvi_owner(directory.path());
        let order = generation_order(&owner);
        let range = CommitRange { low: 1, high: 9 };
        let checkpoint = SourceCheckpointV1::derive(&range);

        let vector = vector_component_receipt(owner.witness(), order.clone(), checkpoint)
            .expect("vector receipt from the real FSVI witness");
        let graph =
            ValidatedNativeHnsw::build(Arc::clone(&owner), HnswParams::default(), 0x7b_5eed)
                .expect("native HNSW graph over the admitted owner");
        let graph_receipt = graph
            .save(&directory.path().join("current.fshnsw"))
            .expect("save the graph and mint its receipt");
        let ann = ann_component_receipt(&graph_receipt, order.clone(), checkpoint)
            .expect("ANN receipt from the real graph receipt");

        let docset = CanonicalDocsetV1::from_ordered_live_documents(order.clone())
            .expect("canonical docset over the shared ordered ids");
        let manifest = manifest_over(range);
        let manifest_bytes = serde_json::to_vec(&manifest).expect("serialize the manifest image");
        let metadata =
            ExactComponentReceiptV1::for_metadata_manifest(&manifest, &manifest_bytes, &docset)
                .expect("metadata receipt from the real manifest");

        let quill_directory = directory.path().join("quill");
        asupersync::test_utils::run_test_with_cx(move |cx| async move {
            let index = QuillIndex::create(&cx, &quill_directory, QuillConfig::default())
                .await
                .expect("create the Quill index");
            let documents = order
                .iter()
                .map(|id| IndexableDocument {
                    id: (*id).to_owned(),
                    content: format!("{id} shares the generation document set"),
                    title: None,
                    metadata: std::collections::HashMap::new(),
                })
                .collect::<Vec<_>>();
            index
                .index_documents(&cx, &documents)
                .await
                .expect("index the shared document set into Quill");
            index.commit(&cx).await.expect("commit the Quill segment");

            let lexical = index
                .snapshot()
                .expect("committed Quill snapshot is authoritative")
                .exact_lexical_component_receipt(checkpoint)
                .expect("lexical receipt from the committed keeper snapshot");

            // Every role independently agrees with the anchor on the document
            // set, without any of them having been told what the others saw.
            assert_eq!(lexical.docset_digest, vector.docset_digest);
            assert_eq!(metadata.docset_digest, vector.docset_digest);
            assert_eq!(lexical.source_checkpoint, vector.source_checkpoint);
            assert_eq!(
                metadata.source_checkpoint, vector.source_checkpoint,
                "the deriving role must reach the same checkpoint as the accepting ones"
            );

            ExactGenerationComponentsV1::admit(
                vector.clone(),
                lexical.clone(),
                Some(ann.clone()),
                metadata.clone(),
            )
            .expect("four real producers admit as one generation");
        });
    }

    /// Every non-anchor role is blamed for ITS OWN drift, over receipts that
    /// four real producers actually minted.
    ///
    /// bd-z4zr3 proved this law with the index-local adapters and hand-built
    /// stand-ins for lexical and metadata. The law is only worth as much as the
    /// producers it was proven over, so it is re-proven here where the lexical
    /// receipt came from a committed Quill snapshot and the metadata receipt
    /// from a real manifest.
    ///
    /// Each drift is applied ALONE against the same all-agreeing control, so a
    /// rejection is attributable to the role that moved rather than to the
    /// harness. The metadata drift moves the MANIFEST rather than passing a
    /// different checkpoint, because that role derives its checkpoint and
    /// refuses a caller-supplied one — the asymmetry is deliberate, not an
    /// inconsistency to clean up.
    #[cfg(feature = "quill")]
    #[test]
    fn every_real_role_is_blamed_for_its_own_drift() {
        use frankensearch::quill::{QuillConfig, QuillIndex};
        use frankensearch_core::generation::{
            CanonicalDocsetV1, ComponentJoinErrorV1, ExactComponentReceiptV1,
            ExactGenerationComponentsV1,
        };
        use frankensearch_core::types::IndexableDocument;

        let directory = tempfile::tempdir().expect("7hvtf drift directory");
        let owner = fsvi_owner(directory.path());
        let order = generation_order(&owner);
        let range = CommitRange { low: 1, high: 9 };
        let other_range = CommitRange { low: 2, high: 11 };
        let checkpoint = SourceCheckpointV1::derive(&range);
        let drifted = SourceCheckpointV1::derive(&other_range);
        assert_ne!(
            checkpoint.to_bytes(),
            drifted.to_bytes(),
            "the two commit ranges must derive different checkpoints or nothing below is a drift"
        );

        let vector = vector_component_receipt(owner.witness(), order.clone(), checkpoint)
            .expect("anchor receipt");
        let graph =
            ValidatedNativeHnsw::build(Arc::clone(&owner), HnswParams::default(), 0x7b_5eed)
                .expect("native HNSW graph over the admitted owner");
        let graph_receipt = graph
            .save(&directory.path().join("current.fshnsw"))
            .expect("save the graph and mint its receipt");
        let ann =
            ann_component_receipt(&graph_receipt, order.clone(), checkpoint).expect("ANN receipt");
        let drifted_ann = ann_component_receipt(&graph_receipt, order.clone(), drifted)
            .expect("ANN receipt on another checkpoint");

        let docset = CanonicalDocsetV1::from_ordered_live_documents(order.clone())
            .expect("canonical docset over the shared ordered ids");
        let manifest = manifest_over(range);
        let manifest_bytes = serde_json::to_vec(&manifest).expect("serialize the manifest image");
        let metadata =
            ExactComponentReceiptV1::for_metadata_manifest(&manifest, &manifest_bytes, &docset)
                .expect("metadata receipt");
        let drifted_manifest = manifest_over(other_range);
        let drifted_manifest_bytes =
            serde_json::to_vec(&drifted_manifest).expect("serialize the drifted manifest");
        let drifted_metadata = ExactComponentReceiptV1::for_metadata_manifest(
            &drifted_manifest,
            &drifted_manifest_bytes,
            &docset,
        )
        .expect("metadata receipt over another commit range");

        let quill_directory = directory.path().join("quill");
        asupersync::test_utils::run_test_with_cx(move |cx| async move {
            let index = QuillIndex::create(&cx, &quill_directory, QuillConfig::default())
                .await
                .expect("create the Quill index");
            let documents = order
                .iter()
                .map(|id| IndexableDocument {
                    id: (*id).to_owned(),
                    content: format!("{id} shares the generation document set"),
                    title: None,
                    metadata: std::collections::HashMap::new(),
                })
                .collect::<Vec<_>>();
            index
                .index_documents(&cx, &documents)
                .await
                .expect("index the shared document set into Quill");
            index.commit(&cx).await.expect("commit the Quill segment");

            let snapshot = index
                .snapshot()
                .expect("committed Quill snapshot is authoritative");
            let lexical = snapshot
                .exact_lexical_component_receipt(checkpoint)
                .expect("lexical receipt");
            let drifted_lexical = snapshot
                .exact_lexical_component_receipt(drifted)
                .expect("lexical receipt on another checkpoint");

            // CONTROL: all four agreeing admit, so every rejection below is
            // attributable to the single role that moved.
            ExactGenerationComponentsV1::admit(
                vector.clone(),
                lexical.clone(),
                Some(ann.clone()),
                metadata.clone(),
            )
            .expect("the all-agreeing control must admit");

            assert_eq!(
                ExactGenerationComponentsV1::admit(
                    vector.clone(),
                    drifted_lexical,
                    Some(ann.clone()),
                    metadata.clone(),
                ),
                Err(ComponentJoinErrorV1::CheckpointDrift { role: "lexical" }),
                "a drifted Quill receipt must be blamed on lexical, not on metadata"
            );
            assert_eq!(
                ExactGenerationComponentsV1::admit(
                    vector.clone(),
                    lexical.clone(),
                    Some(drifted_ann),
                    metadata.clone(),
                ),
                Err(ComponentJoinErrorV1::CheckpointDrift { role: "ann" })
            );
            assert_eq!(
                ExactGenerationComponentsV1::admit(
                    vector.clone(),
                    lexical.clone(),
                    Some(ann.clone()),
                    drifted_metadata,
                ),
                Err(ComponentJoinErrorV1::CheckpointDrift { role: "metadata" })
            );

            // A receipt filed in another role's slot is refused BY ROLE, before
            // any content comparison — otherwise a misfiled receipt would be
            // reported as a drift and the operator would go looking for the
            // wrong fault.
            assert_eq!(
                ExactGenerationComponentsV1::admit(
                    vector.clone(),
                    ann.clone(),
                    Some(ann.clone()),
                    metadata.clone(),
                ),
                Err(ComponentJoinErrorV1::RoleMismatch {
                    expected: "lexical",
                    found: "ann",
                })
            );
        });
    }

    /// A drifted ANCHOR is a different failure and is kept in its own test.
    ///
    /// `admit` compares every role against the vector anchor rather than
    /// pairwise, so when the anchor itself moves, the roles that did NOT move
    /// are the ones reported. Folding this into the per-role test above would
    /// let "the anchor moved" be reported as "a component drifted", which is
    /// precisely the misattribution bd-z4zr3 was filed to fix.
    ///
    /// All four roles are properly filed here on purpose: an earlier draft of
    /// this test put a metadata receipt in the lexical slot, which tripped
    /// `RoleMismatch` and never reached the anchor comparison at all — a test
    /// that would have passed without the behaviour it claims to pin.
    #[cfg(feature = "quill")]
    #[test]
    fn a_drifted_anchor_is_reported_against_the_roles_that_disagree_with_it() {
        use frankensearch::quill::{QuillConfig, QuillIndex};
        use frankensearch_core::generation::{
            CanonicalDocsetV1, ComponentJoinErrorV1, ExactComponentReceiptV1,
            ExactGenerationComponentsV1,
        };
        use frankensearch_core::types::IndexableDocument;

        let directory = tempfile::tempdir().expect("7hvtf anchor directory");
        let owner = fsvi_owner(directory.path());
        let order = generation_order(&owner);
        let range = CommitRange { low: 1, high: 9 };
        let checkpoint = SourceCheckpointV1::derive(&range);
        let moved_anchor = SourceCheckpointV1::derive(&CommitRange { low: 5, high: 6 });

        let vector =
            vector_component_receipt(owner.witness(), order.clone(), checkpoint).expect("anchor");
        let drifted_vector = vector_component_receipt(owner.witness(), order.clone(), moved_anchor)
            .expect("anchor receipt on a different checkpoint");
        let graph =
            ValidatedNativeHnsw::build(Arc::clone(&owner), HnswParams::default(), 0x7b_5eed)
                .expect("native HNSW graph over the admitted owner");
        let graph_receipt = graph
            .save(&directory.path().join("current.fshnsw"))
            .expect("save the graph and mint its receipt");
        let ann =
            ann_component_receipt(&graph_receipt, order.clone(), checkpoint).expect("ANN receipt");
        let docset = CanonicalDocsetV1::from_ordered_live_documents(order.clone())
            .expect("canonical docset over the shared ordered ids");
        let manifest = manifest_over(range);
        let manifest_bytes = serde_json::to_vec(&manifest).expect("serialize the manifest image");
        let metadata =
            ExactComponentReceiptV1::for_metadata_manifest(&manifest, &manifest_bytes, &docset)
                .expect("metadata receipt");

        let quill_directory = directory.path().join("quill");
        asupersync::test_utils::run_test_with_cx(move |cx| async move {
            let index = QuillIndex::create(&cx, &quill_directory, QuillConfig::default())
                .await
                .expect("create the Quill index");
            let documents = order
                .iter()
                .map(|id| IndexableDocument {
                    id: (*id).to_owned(),
                    content: format!("{id} shares the generation document set"),
                    title: None,
                    metadata: std::collections::HashMap::new(),
                })
                .collect::<Vec<_>>();
            index
                .index_documents(&cx, &documents)
                .await
                .expect("index the shared document set into Quill");
            index.commit(&cx).await.expect("commit the Quill segment");
            let lexical = index
                .snapshot()
                .expect("committed Quill snapshot is authoritative")
                .exact_lexical_component_receipt(checkpoint)
                .expect("lexical receipt");

            // CONTROL: the same four roles admit when the anchor has not moved,
            // so the refusal below is caused by the anchor and nothing else.
            ExactGenerationComponentsV1::admit(
                vector.clone(),
                lexical.clone(),
                Some(ann.clone()),
                metadata.clone(),
            )
            .expect("the unmoved control must admit");

            let observed =
                ExactGenerationComponentsV1::admit(drifted_vector, lexical, Some(ann), metadata);
            assert!(
                matches!(
                    &observed,
                    Err(ComponentJoinErrorV1::CheckpointDrift {
                        role: "lexical" | "ann" | "metadata"
                    })
                ),
                "a moved anchor must be reported against a non-vector role that disagrees with \
                 it, got {observed:?}"
            );
        });
    }
}
