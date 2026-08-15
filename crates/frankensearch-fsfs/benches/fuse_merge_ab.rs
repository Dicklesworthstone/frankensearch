//! Hybrid-fuse merge A/B: the per-query RRF fuse (`fuse_rankings_with_priors`) merges the
//! score-sorted lexical + semantic candidate lists into one map. The former shape did a separate
//! `merged.get(doc_id)` (to skip already-ranked dups) AND a `merged.entry(doc_id.to_string())`
//! per candidate — hashing the `doc_id` twice and cloning it into an owned key on EVERY candidate,
//! including the lexical∩semantic overlap where the clone was immediately discarded by
//! `and_modify`. The lever (`merge_ranked`) does one `get_mut`-or-`insert`: overlap docs are
//! modified in place (single hash, no key clone); only genuinely new docs pay the insert.
//!
//! Byte-identical (`merge_ranked` == `merge_ranked_orig`, proven by `merge_ranked_matches_orig`
//! and re-asserted here before timing). Single-threaded, so the paired ratio is contention-robust.
//! Gate on the MEDIAN of CAND/ORIG against the A/A null floor.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR \
//!   rch exec -- cargo bench -p frankensearch-fsfs --features bench-internals --bench fuse_merge_ab
//! ```

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use frankensearch_core::bench_support::paired_median_ratio;
use frankensearch_fsfs::{LexicalCandidate, QueryExecutionOrchestrator, SemanticCandidate};

const K: f64 = 60.0;

/// Deterministic hybrid candidate set: `n` lexical + `n` semantic, with `overlap` shared `doc_ids`
/// (the regime the lever targets — shared docs hit the modify path). `doc_ids` are realistic-length.
fn make_candidates(n: usize, overlap: usize) -> (Vec<LexicalCandidate>, Vec<SemanticCandidate>) {
    let mut lexical = Vec::with_capacity(n);
    let mut semantic = Vec::with_capacity(n);
    let mut x = 0x9e37_79b9_7f4a_7c15_u64;
    let mut score = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x >> 40) as f32 / (1u64 << 24) as f32
    };
    // Shared doc_ids [0, overlap): appear in both tiers.
    for i in 0..overlap {
        lexical.push(LexicalCandidate::new(format!("doc-{i:08}"), score()));
        semantic.push(SemanticCandidate::new(format!("doc-{i:08}"), score()));
    }
    // Lexical-only [overlap, n) and semantic-only [n, 2n-overlap).
    for i in overlap..n {
        lexical.push(LexicalCandidate::new(format!("doc-{i:08}"), score()));
    }
    for i in n..(2 * n - overlap) {
        semantic.push(SemanticCandidate::new(format!("doc-{i:08}"), score()));
    }
    (lexical, semantic)
}

fn bench(_c: &mut Criterion) {
    let inner: u32 = std::env::var("FUSE_AB_INNER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    // (n, overlap): high overlap is the lever's home; a low-overlap case bounds the worst case.
    let shapes = [(200_usize, 140_usize), (200, 40), (600, 400)];

    for &(n, overlap) in &shapes {
        let (lexical, semantic) = make_candidates(n, overlap);

        // Byte-identity assert before timing.
        let mut new_v: Vec<_> =
            QueryExecutionOrchestrator::merge_ranked(K, &lexical, &semantic, false)
                .into_values()
                .collect();
        let mut orig_v: Vec<_> =
            QueryExecutionOrchestrator::merge_ranked_orig(K, &lexical, &semantic, false)
                .into_values()
                .collect();
        new_v.sort_by(|a, b| a.doc_id.cmp(&b.doc_id));
        orig_v.sort_by(|a, b| a.doc_id.cmp(&b.doc_id));
        assert_eq!(
            new_v, orig_v,
            "merge_ranked != orig for n={n} overlap={overlap}"
        );

        let run_orig = || {
            black_box(QueryExecutionOrchestrator::merge_ranked_orig(
                K,
                black_box(&lexical),
                black_box(&semantic),
                false,
            ));
        };
        let run_cand = || {
            black_box(QueryExecutionOrchestrator::merge_ranked(
                K,
                black_box(&lexical),
                black_box(&semantic),
                false,
            ));
        };

        let null = paired_median_ratio(41, inner, run_orig, run_orig);
        let lever = paired_median_ratio(41, inner, run_orig, run_cand);
        eprintln!(
            "[null]  fuse_merge/n{n}_ov{overlap}: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
            null.median, null.p5, null.p95, null.rounds
        );
        eprintln!(
            "[lever] fuse_merge/n{n}_ov{overlap}: cand/ORIG median {:.4} p5 {:.4} p95 {:.4} -> {}",
            lever.median,
            lever.p5,
            lever.p95,
            if lever.decidable_against(&null) {
                if lever.median < 1.0 {
                    "DECIDABLE WIN"
                } else {
                    "DECIDABLE REGRESSION"
                }
            } else {
                "INSIDE NULL FLOOR (not decidable)"
            }
        );
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
