//! E2E validation: RRF fusion and score blending correctness (bd-3un.40).
//!
//! Tests RRF fusion with known rankings, score blending with various
//! blend factors, normalization, and edge cases.
//!
//! Run with: `cargo run --example validate_fusion`

use std::time::Instant;

use frankensearch_core::types::{ScoreSource, ScoredResult, VectorHit};
use frankensearch_fusion::normalize::{
    NormalizationMethod, min_max_normalize, normalize_in_place, z_score_normalize,
};
use frankensearch_fusion::rrf::{RrfConfig, candidate_count, rrf_fuse};

#[allow(clippy::too_many_lines)]
fn main() {
    let start = Instant::now();
    let mut pass = 0u32;
    let mut fail = 0u32;

    println!("\n\x1b[1;36m=== frankensearch E2E: Fusion & Normalization Validation ===\x1b[0m\n");

    // ── Step 1: Basic RRF fusion ──────────────────────────────────────────
    log_info("RRF", "Testing basic RRF fusion with known rankings...");

    let lexical = make_lexical(&["a", "b", "c", "d", "e"], &[10.0, 8.0, 6.0, 4.0, 2.0]);
    let semantic = make_semantic(&["c", "a", "f", "b", "g"], &[0.9, 0.8, 0.7, 0.6, 0.5]);
    let config = RrfConfig::default();

    let fused = rrf_fuse(&lexical, &semantic, 10, 0, &config);
    log_info(
        "RRF",
        &format!(
            "Fused {} lexical + {} semantic -> {} results",
            lexical.len(),
            semantic.len(),
            fused.len()
        ),
    );

    // "a" and "c" appear in both sources — should have highest RRF scores
    let fused_ids: Vec<&str> = fused.iter().map(|h| h.doc_id.as_str()).collect();
    log_info("RRF", &format!("Fused order: {fused_ids:?}"));

    // Documents in both sources should be marked
    let a_hit = fused.iter().find(|h| h.doc_id == "a").expect("a in fused");
    let f_hit = fused.iter().find(|h| h.doc_id == "f").expect("f in fused");
    check(
        &mut pass,
        &mut fail,
        "RRF: 'a' in both sources",
        a_hit.in_both_sources,
    );
    check(
        &mut pass,
        &mut fail,
        "RRF: 'f' in semantic only",
        !f_hit.in_both_sources,
    );

    // Documents in both sources should have higher RRF scores
    check(
        &mut pass,
        &mut fail,
        "RRF: dual-source > single-source score",
        a_hit.rrf_score > f_hit.rrf_score,
    );

    // All 7 unique docs should appear
    check(
        &mut pass,
        &mut fail,
        "RRF: all unique docs present",
        fused.len() == 7,
    );

    // Scores should be strictly descending (with deterministic tie-breaking)
    let scores_desc = fused.windows(2).all(|w| w[0].rrf_score >= w[1].rrf_score);
    check(
        &mut pass,
        &mut fail,
        "RRF: scores are descending",
        scores_desc,
    );

    // ── Step 2: RRF with limit/offset ─────────────────────────────────────
    log_info("RRF", "Testing limit and offset...");

    let fused_limited = rrf_fuse(&lexical, &semantic, 3, 0, &config);
    check(
        &mut pass,
        &mut fail,
        "RRF limit=3: returns 3",
        fused_limited.len() == 3,
    );

    let fused_offset = rrf_fuse(&lexical, &semantic, 3, 2, &config);
    check(
        &mut pass,
        &mut fail,
        "RRF offset=2 limit=3: returns 3",
        fused_offset.len() == 3,
    );

    // Offset results should continue from where limit left off
    let full = rrf_fuse(&lexical, &semantic, 10, 0, &config);
    if full.len() >= 5 {
        check(
            &mut pass,
            &mut fail,
            "RRF offset consistency",
            fused_offset[0].doc_id == full[2].doc_id,
        );
    }

    // ── Step 3: RRF edge cases ────────────────────────────────────────────
    log_info("RRF", "Testing edge cases...");

    // Empty lexical
    let empty_lex = rrf_fuse(&[], &semantic, 10, 0, &config);
    check(
        &mut pass,
        &mut fail,
        "RRF: empty lexical -> semantic only",
        empty_lex.len() == semantic.len(),
    );

    // Empty semantic
    let empty_sem = rrf_fuse(&lexical, &[], 10, 0, &config);
    check(
        &mut pass,
        &mut fail,
        "RRF: empty semantic -> lexical only",
        empty_sem.len() == lexical.len(),
    );

    // Both empty
    let both_empty = rrf_fuse(&[], &[], 10, 0, &config);
    check(
        &mut pass,
        &mut fail,
        "RRF: both empty -> empty",
        both_empty.is_empty(),
    );

    // limit=0
    let limit_zero = rrf_fuse(&lexical, &semantic, 0, 0, &config);
    check(
        &mut pass,
        &mut fail,
        "RRF: limit=0 -> empty",
        limit_zero.is_empty(),
    );

    // Complete overlap
    let overlap_lex = make_lexical(&["x", "y", "z"], &[3.0, 2.0, 1.0]);
    let overlap_sem = make_semantic(&["x", "y", "z"], &[0.9, 0.8, 0.7]);
    let fused_overlap = rrf_fuse(&overlap_lex, &overlap_sem, 10, 0, &config);
    check(
        &mut pass,
        &mut fail,
        "RRF: complete overlap -> all in_both",
        fused_overlap.iter().all(|h| h.in_both_sources),
    );
    check(
        &mut pass,
        &mut fail,
        "RRF: complete overlap -> 3 results",
        fused_overlap.len() == 3,
    );

    // Zero overlap
    let no_overlap_lex = make_lexical(&["p", "q"], &[2.0, 1.0]);
    let no_overlap_sem = make_semantic(&["r", "s"], &[0.8, 0.6]);
    let fused_no_overlap = rrf_fuse(&no_overlap_lex, &no_overlap_sem, 10, 0, &config);
    check(
        &mut pass,
        &mut fail,
        "RRF: zero overlap -> none in_both",
        fused_no_overlap.iter().all(|h| !h.in_both_sources),
    );
    check(
        &mut pass,
        &mut fail,
        "RRF: zero overlap -> 4 results",
        fused_no_overlap.len() == 4,
    );

    // ── Step 4: RRF K parameter sensitivity ───────────────────────────────
    log_info("RRF", "Testing K parameter sensitivity...");

    let low_k = RrfConfig { k: 1.0 };
    let high_k = RrfConfig { k: 1000.0 };

    let fused_low = rrf_fuse(&lexical, &semantic, 10, 0, &low_k);
    let fused_high = rrf_fuse(&lexical, &semantic, 10, 0, &high_k);

    // With low K, rank differences matter more (sharper distribution)
    let score_spread_low = fused_low.first().map_or(0.0, |h| h.rrf_score)
        - fused_low.last().map_or(0.0, |h| h.rrf_score);
    let score_spread_high = fused_high.first().map_or(0.0, |h| h.rrf_score)
        - fused_high.last().map_or(0.0, |h| h.rrf_score);

    check(
        &mut pass,
        &mut fail,
        "RRF: low K -> wider score spread",
        score_spread_low > score_spread_high,
    );
    log_info(
        "RRF",
        &format!("Score spread: K=1 -> {score_spread_low:.6}, K=1000 -> {score_spread_high:.6}"),
    );

    // ── Step 5: candidate_count helper ────────────────────────────────────
    log_info("BUDGET", "Testing candidate budget calculation...");
    check(
        &mut pass,
        &mut fail,
        "candidate_count(10, 0, 3) = 30",
        candidate_count(10, 0, 3) == 30,
    );
    check(
        &mut pass,
        &mut fail,
        "candidate_count(10, 5, 3) = 45",
        candidate_count(10, 5, 3) == 45,
    );
    // Saturation
    check(
        &mut pass,
        &mut fail,
        "candidate_count saturates on overflow",
        candidate_count(usize::MAX, 1, 2) == usize::MAX,
    );

    // ── Step 6: Score normalization ───────────────────────────────────────
    log_info("NORM", "Testing score normalization...");

    // Min-max basic
    let mut scores = vec![1.0, 3.0, 5.0, 7.0, 9.0];
    min_max_normalize(&mut scores);
    check(
        &mut pass,
        &mut fail,
        "min_max: min maps to 0.0",
        (scores[0] - 0.0).abs() < 1e-6,
    );
    check(
        &mut pass,
        &mut fail,
        "min_max: max maps to 1.0",
        (scores[4] - 1.0).abs() < 1e-6,
    );
    check(
        &mut pass,
        &mut fail,
        "min_max: mid maps to 0.5",
        (scores[2] - 0.5).abs() < 1e-6,
    );

    // Z-score basic
    let mut z_scores = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    z_score_normalize(&mut z_scores);
    check(
        &mut pass,
        &mut fail,
        "z_score: all in [0, 1]",
        z_scores.iter().all(|&s| (0.0..=1.0).contains(&s)),
    );
    // Mean of normalized scores should be near 0.5
    #[allow(clippy::cast_precision_loss)]
    let z_mean: f32 = z_scores.iter().sum::<f32>() / z_scores.len() as f32;
    check(
        &mut pass,
        &mut fail,
        "z_score: mean near 0.5",
        (z_mean - 0.5).abs() < 0.1,
    );

    // Degenerate: all identical
    let mut identical = vec![5.0, 5.0, 5.0];
    min_max_normalize(&mut identical);
    check(
        &mut pass,
        &mut fail,
        "min_max: identical -> 0.5",
        identical.iter().all(|&s| (s - 0.5).abs() < 1e-6),
    );

    // NaN handling
    let mut with_nan = vec![1.0, f32::NAN, 3.0];
    min_max_normalize(&mut with_nan);
    check(
        &mut pass,
        &mut fail,
        "min_max: NaN -> 0.0",
        with_nan[1] == 0.0,
    );
    check(
        &mut pass,
        &mut fail,
        "min_max: finite values normalized",
        (with_nan[0] - 0.0).abs() < 1e-6 && (with_nan[2] - 1.0).abs() < 1e-6,
    );

    // Infinity handling
    let mut with_inf = vec![1.0, f32::INFINITY, 3.0];
    min_max_normalize(&mut with_inf);
    check(
        &mut pass,
        &mut fail,
        "min_max: Inf -> 0.0",
        with_inf[1] == 0.0,
    );

    // Empty
    let mut empty: Vec<f32> = vec![];
    min_max_normalize(&mut empty);
    check(
        &mut pass,
        &mut fail,
        "min_max: empty is no-op",
        empty.is_empty(),
    );

    // Single element
    let mut single = vec![42.0];
    min_max_normalize(&mut single);
    check(
        &mut pass,
        &mut fail,
        "min_max: single element -> 0.5",
        (single[0] - 0.5).abs() < 1e-6,
    );

    // NormalizationMethod dispatch
    let mut dispatch_test = vec![1.0, 5.0, 9.0];
    normalize_in_place(&mut dispatch_test, NormalizationMethod::MinMax);
    check(
        &mut pass,
        &mut fail,
        "normalize_in_place MinMax: correct",
        (dispatch_test[0] - 0.0).abs() < 1e-6 && (dispatch_test[2] - 1.0).abs() < 1e-6,
    );

    let mut none_test = vec![1.0, 5.0, 9.0];
    normalize_in_place(&mut none_test, NormalizationMethod::None);
    check(
        &mut pass,
        &mut fail,
        "normalize_in_place None: unchanged",
        none_test == vec![1.0, 5.0, 9.0],
    );

    // ── Cleanup and summary ───────────────────────────────────────────────
    println!();
    println!("\x1b[1;36m=== Summary ===\x1b[0m");
    println!("  \x1b[32mPassed: {pass}\x1b[0m  \x1b[31mFailed: {fail}\x1b[0m");
    println!(
        "  Total time: {:.1}ms",
        start.elapsed().as_secs_f64() * 1000.0
    );
    println!();

    if fail > 0 {
        std::process::exit(1);
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn make_lexical(ids: &[&str], scores: &[f32]) -> Vec<ScoredResult> {
    ids.iter()
        .zip(scores.iter())
        .map(|(id, &score)| ScoredResult {
            doc_id: id.to_string(),
            score,
            source: ScoreSource::Lexical,
            fast_score: None,
            quality_score: None,
            lexical_score: Some(score),
            rerank_score: None,
            metadata: None,
        })
        .collect()
}

#[allow(clippy::cast_possible_truncation)]
fn make_semantic(ids: &[&str], scores: &[f32]) -> Vec<VectorHit> {
    ids.iter()
        .zip(scores.iter())
        .enumerate()
        .map(|(i, (id, &score))| VectorHit {
            index: i as u32,
            score,
            doc_id: id.to_string(),
        })
        .collect()
}

fn log_info(step: &str, msg: &str) {
    println!("\x1b[36m[INFO] [{step}]\x1b[0m {msg}");
}

fn log_fail(step: &str, msg: &str) {
    println!("\x1b[31m[FAIL] [{step}]\x1b[0m {msg}");
}

fn log_pass(step: &str, msg: &str) {
    println!("\x1b[32m[PASS] [{step}]\x1b[0m {msg}");
}

fn check(pass: &mut u32, fail: &mut u32, name: &str, ok: bool) {
    if ok {
        log_pass("CHECK", name);
        *pass += 1;
    } else {
        log_fail("CHECK", name);
        *fail += 1;
    }
}
