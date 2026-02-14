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
    let dim = records.first().map_or(4, |(_, v)| v.len());
    let path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    let mut writer =
        VectorIndex::create_with_revision(&path, "potion-128M", "v1", dim, Quantization::F16)
            .expect("create writer");
    for (doc_id, vec) in records {
        writer.write_record(doc_id, vec).expect("write record");
    }
    writer.finish().expect("finish index");
}

fn write_quality_index(dir: &std::path::Path, records: &[(&str, Vec<f32>)]) {
    let dim = records.first().map_or(4, |(_, v)| v.len());
    let path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);
    let mut writer =
        VectorIndex::create_with_revision(&path, "MiniLM-L6-v2", "v1", dim, Quantization::F16)
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
        doc_id: doc_id.to_owned(),
    }
}

fn scored(doc_id: &str, score: f32) -> ScoredResult {
    ScoredResult {
        doc_id: doc_id.to_owned(),
        score,
        source: ScoreSource::Hybrid,
        fast_score: None,
        quality_score: None,
        lexical_score: None,
        rerank_score: None,
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
    let indices: Vec<usize> = fast_hits.iter().map(|h| h.index as usize).collect();
    let query_quality = normalize_vec(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let quality_scores = index
        .quality_scores_for_indices(&query_quality, &indices)
        .expect("quality scores");
    // shared-1 should have highest quality score (its quality embedding is close to query)
    assert!(quality_scores[0] > quality_scores[1]);
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
    // All identical scores → normalization produces 0.5 → blend is 0.5
    let fast = vec![hit("a", 5.0, 0), hit("b", 5.0, 1)];
    let quality = vec![hit("a", 5.0, 0), hit("b", 5.0, 1)];

    let blended = blend_two_tier(&fast, &quality, 0.5);
    // All equal scores: normalize → 0.5 each. Blend: 0.5*0.5 + 0.5*0.5 = 0.5
    for h in &blended {
        assert!(
            (h.score - 0.5).abs() < 1e-5,
            "expected ~0.5, got {}",
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

    let rrf_config = RrfConfig { k: 60.0 };
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
            doc_id: "doc-1".to_owned(),
            text: "caf\u{0065}\u{0301}".to_owned(),
            metadata: None,
            submitted_at: Instant::now(),
        })
        .unwrap();

    // NFC-precomposed: é
    queue
        .submit(EmbeddingRequest {
            doc_id: "doc-2".to_owned(),
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
            doc_id: "doc-3".to_owned(),
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
            doc_id: "doc-1".to_owned(),
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
            doc_id: "doc-1".to_owned(),
            text: "Important document content".to_owned(),
            metadata: None,
            submitted_at: Instant::now(),
        })
        .unwrap();
    assert_eq!(outcome, JobOutcome::SkippedUnchanged);

    // Re-submit modified content → should enqueue
    let outcome = queue
        .submit(EmbeddingRequest {
            doc_id: "doc-1".to_owned(),
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
            doc_id: "doc-1".to_owned(),
            text: "First".to_owned(),
            metadata: None,
            submitted_at: Instant::now(),
        })
        .unwrap();
    queue
        .submit(EmbeddingRequest {
            doc_id: "doc-2".to_owned(),
            text: "Second".to_owned(),
            metadata: None,
            submitted_at: Instant::now(),
        })
        .unwrap();

    // Queue full → backpressure
    let err = queue
        .submit(EmbeddingRequest {
            doc_id: "doc-3".to_owned(),
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
            doc_id: "doc-1".to_owned(),
            text: "First".to_owned(),
            metadata: None,
            submitted_at: Instant::now(),
        })
        .unwrap();
    assert_eq!(outcome, JobOutcome::SkippedUnchanged);

    let outcome = queue
        .submit(EmbeddingRequest {
            doc_id: "doc-3".to_owned(),
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
    assert!(
        matches!(err, SearchError::IndexNotFound { .. }),
        "expected IndexNotFound, got: {err:?}"
    );
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

    let config = RrfConfig { k: 60.0 };
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

    let config = RrfConfig { k: 60.0 };

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
    let config = RrfConfig { k: 60.0 };

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
    let config = RrfConfig { k: 60.0 };
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

    let fused = rrf_fuse(&lexical, &semantic, 10, 0, &RrfConfig { k: 60.0 });
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
