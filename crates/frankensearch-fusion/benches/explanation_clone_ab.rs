//! Same-worker A/B for the `ScoredResult.explanation` clone (the residual after
//! the metadata→Arc landing).
//!
//! When `explain=true`, each winner carries a `HitExplanation` (nested
//! `Vec<ScoreComponent>` with embedder `String`s, +
//! `RankMovement.reason: String`). The async progressive-phase emission clones
//! `Vec<ScoredResult>` per phase (`searcher.rs:543/637/683/806`), deep-cloning
//! every explanation — N per phase, 2 phases per query. This measures whether an
//! `Arc<HitExplanation>` (refcount bump) is materially cheaper than the deep clone,
//! to decide if the metadata-Arc pattern is worth repeating for the `explain` path.
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-fusion --bench explanation_clone_ab
//! ```

use std::hint::black_box;
use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_core::explanation::{
    ExplainedSource, ExplanationPhase, HitExplanation, RankMovement, ScoreComponent,
};
use frankensearch_fusion::bench_support::paired_median_ratio;

/// A representative populated `HitExplanation` (a hybrid hit: lexical + semantic
/// components, plus a rank movement) — the shape the async searcher builds when
/// `explain=true`.
fn realistic_explanation(i: usize) -> HitExplanation {
    HitExplanation {
        final_score: 0.87,
        components: vec![
            ScoreComponent {
                source: ExplainedSource::LexicalBm25 { terms: Vec::new() },
                raw_score: 12.5,
                normalized_score: 0.8,
                rrf_contribution: 1.0 / 61.0,
                weight: 0.3,
            },
            ScoreComponent {
                source: ExplainedSource::SemanticFast {
                    embedder: "potion-base-8M".to_owned(),
                    cosine_sim: 0.72,
                },
                raw_score: 0.72,
                normalized_score: 0.72,
                rrf_contribution: 1.0 / 62.0,
                weight: 0.7,
            },
        ],
        phase: ExplanationPhase::Refined,
        rank_movement: Some(RankMovement {
            initial_rank: i % 50,
            refined_rank: (i % 50).saturating_sub(3),
            delta: -3,
            reason: "promoted by quality-tier cosine rescore".to_owned(),
        }),
    }
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("explanation_clone_ab");
    for n in [10_000usize, 100_000] {
        let values: Vec<HitExplanation> = (0..n).map(realistic_explanation).collect();
        let arcs: Vec<Arc<HitExplanation>> = values.iter().cloned().map(Arc::new).collect();

        // Current: N deep clones of Option<HitExplanation> (per-phase emission).
        g.bench_with_input(BenchmarkId::new("value_deep_clone", n), &values, |b, vs| {
            b.iter(|| {
                let out: Vec<Option<HitExplanation>> =
                    black_box(vs).iter().map(|v| Some(v.clone())).collect();
                black_box(out)
            });
        });
        // Candidate: N Arc refcount bumps of Option<Arc<HitExplanation>>.
        g.bench_with_input(BenchmarkId::new("arc_clone", n), &arcs, |b, as_| {
            b.iter(|| {
                let out: Vec<Option<Arc<HitExplanation>>> =
                    black_box(as_).iter().map(|a| Some(Arc::clone(a))).collect();
                black_box(out)
            });
        });

        // ── DECIDABILITY: alternating-round paired sampler + A/A null control ──
        //
        // The criterion arms above CANNOT decide this lever: criterion runs them as
        // separate benchmarks minutes apart, so worker drift between them is not
        // cancelled. The paired sampler runs both arms in ONE routine in alternating
        // rounds and takes the median per-round ratio; gate on the median against the
        // A/A null's observed spread, not on cv.
        let value_deep_clone = || {
            let out: Vec<Option<HitExplanation>> =
                black_box(&values).iter().map(|v| Some(v.clone())).collect();
            black_box(out);
        };
        let arc_clone = || {
            let out: Vec<Option<Arc<HitExplanation>>> = black_box(&arcs)
                .iter()
                .map(|a| Some(Arc::clone(a)))
                .collect();
            black_box(out);
        };
        let null = paired_median_ratio(41, 8, value_deep_clone, value_deep_clone);
        let lever = paired_median_ratio(41, 8, value_deep_clone, arc_clone);
        eprintln!(
            "[null]  explanation_clone n {n}: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
            null.median, null.p5, null.p95, null.rounds
        );
        eprintln!(
            "[lever] explanation_clone n {n}: arc_clone median {:.4} p5 {:.4} p95 {:.4} -> {}",
            lever.median,
            lever.p5,
            lever.p95,
            if lever.decidable_against(&null) {
                "DECIDABLE"
            } else {
                "INSIDE NULL FLOOR (not decidable)"
            }
        );
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
