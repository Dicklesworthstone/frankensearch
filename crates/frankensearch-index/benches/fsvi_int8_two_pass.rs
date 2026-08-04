//! FSVI (file-backed `VectorIndex`) int8 two-pass same-binary A/B — lever
//! validation for standalone large-N vector search on clustered data.
//!
//! The bench-only four-row candidate shares each query decode across adjacent
//! int8 rows; `VectorIndex::search_top_k_int8_two_pass` remains the ORIGINAL arm.
//! Both do a fast parallel int8 pass-1 plus exact f16 top-`k·mult` rescore. This is
//! NOT wired into the BOLD hybrid (that gap is not vector-bound — see
//! docs/NEGATIVE_EVIDENCE.md); this measures pure vector-search latency + recall.
//!
//! Run with:
//! ```bash
//! RAYON_NUM_THREADS=4 RUSTFLAGS='-C force-frame-pointers=yes' \
//!   RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- cargo bench \
//!   -p frankensearch-index --profile release-perf --bench fsvi_int8_two_pass -- \
//!   fsvi_int8_two_pass_ab --sample-size 50 --warm-up-time 2 --measurement-time 10 --noplot
//! ```

#![allow(
    clippy::cast_possible_truncation,
    clippy::doc_markdown,
    clippy::significant_drop_tightening,
    clippy::similar_names
)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use criterion::{Criterion, SamplingMode, criterion_group, criterion_main};
use frankensearch_index::VectorIndex;

const N: usize = 100_000;
const DIM: usize = 384;
const K: usize = 10;
const QUERIES: usize = 32;
const CLUSTERS: usize = 64;
const NOISE: f32 = 0.30;
const INTERLEAVE_BATCH: u64 = 32;
const PAIRED_ROUND_PAIRS: usize = 41;
// Eight identical full-corpus sweeps make each arm long enough to calibrate the
// per-workload scheduler floor without changing query weighting.
const PAIRED_INNER: u32 = 256;

#[derive(Debug, Clone, Copy)]
struct PairedRatio {
    median: f64,
    p5: f64,
    p95: f64,
    a_mean_us: f64,
    b_mean_us: f64,
    a_cv_pct: f64,
    b_cv_pct: f64,
    round_pairs: usize,
}

impl PairedRatio {
    fn calibrated_ratio(self, null: Self) -> f64 {
        self.median / null.median
    }

    fn calibrated_floor(null: Self) -> (f64, f64) {
        (null.p5 / null.median, null.p95 / null.median)
    }

    /// Largest `|median - 1|` an A/A control may show and still be admissible.
    /// Mirrors `frankensearch_core::bench_support::NULL_MEDIAN_TOLERANCE`.
    const NULL_MEDIAN_TOLERANCE: f64 = 0.02;

    /// A/A control admissibility: gate the null's accuracy, never its precision.
    ///
    /// Requiring `p5 <= 1.0 <= p95` vetoed a null for being *tight*: the
    /// narrower its spread, the likelier it excluded 1.0 — however small the
    /// bias and however large the effect (`bd-pjh09`). The spread still sets
    /// the effect floor below; only the median says whether the sampler itself
    /// is trustworthy.
    fn null_is_unbiased(self) -> bool {
        (self.median - 1.0).abs() <= Self::NULL_MEDIAN_TOLERANCE
    }

    fn verdict_against(self, null: Self) -> &'static str {
        if !null.null_is_unbiased() {
            "BIASED_NULL_UNDECIDABLE"
        } else if self.median < null.p5 {
            "CANDIDATE_FASTER"
        } else if self.median > null.p95 {
            "CANDIDATE_SLOWER"
        } else {
            "INSIDE_NULL_FLOOR"
        }
    }
}

fn time_batch<F: FnMut()>(inner: u32, f: &mut F) -> Duration {
    let started = Instant::now();
    for _ in 0..inner {
        f();
    }
    started.elapsed()
}

fn mean_and_cv(samples: &[f64]) -> (f64, f64) {
    let n = u32::try_from(samples.len()).expect("paired sample count fits u32");
    let n_f64 = f64::from(n);
    let mean = samples.iter().sum::<f64>() / n_f64;
    if n < 2 || mean == 0.0 {
        return (mean, 0.0);
    }
    let variance = samples
        .iter()
        .map(|sample| {
            let delta = sample - mean;
            delta * delta
        })
        .sum::<f64>()
        / f64::from(n - 1);
    (mean, 100.0 * variance.sqrt() / mean)
}

fn percentile(sorted: &[f64], pct: usize) -> f64 {
    let index = ((sorted.len() - 1) * pct + 50) / 100;
    sorted[index]
}

fn time_ordered_pair<A: FnMut(), B: FnMut()>(
    inner: u32,
    a: &mut A,
    b: &mut B,
    a_first: bool,
) -> (f64, f64) {
    let (a_elapsed, b_elapsed) = if a_first {
        (time_batch(inner, a), time_batch(inner, b))
    } else {
        let b_elapsed = time_batch(inner, b);
        let a_elapsed = time_batch(inner, a);
        (a_elapsed, b_elapsed)
    };
    (a_elapsed.as_secs_f64(), b_elapsed.as_secs_f64())
}

fn paired_median_ratio<A: FnMut(), B: FnMut()>(
    round_pairs: usize,
    inner: u32,
    mut a: A,
    mut b: B,
) -> PairedRatio {
    assert!(round_pairs > 0 && inner > 0);

    // Warm both execution orders with complete, unrecorded query sweeps.
    for _ in 0..2 {
        black_box(time_batch(inner, &mut a));
        black_box(time_batch(inner, &mut b));
        black_box(time_batch(inner, &mut b));
        black_box(time_batch(inner, &mut a));
    }

    // Each observation is one AB sweep and one BA sweep. Combining their
    // ratios geometrically cancels first/second-position effects instead of
    // letting an odd raw-round median choose the majority execution order.
    let mut a_seconds = Vec::with_capacity(round_pairs * 2);
    let mut b_seconds = Vec::with_capacity(round_pairs * 2);
    let mut ratios = Vec::with_capacity(round_pairs);
    for pair in 0..round_pairs {
        let ((a_ab, b_ab), (a_ba, b_ba)) = if pair.is_multiple_of(2) {
            (
                time_ordered_pair(inner, &mut a, &mut b, true),
                time_ordered_pair(inner, &mut a, &mut b, false),
            )
        } else {
            (
                time_ordered_pair(inner, &mut a, &mut b, false),
                time_ordered_pair(inner, &mut a, &mut b, true),
            )
        };
        if a_ab > 0.0 && a_ba > 0.0 {
            a_seconds.extend([a_ab, a_ba]);
            b_seconds.extend([b_ab, b_ba]);
            ratios.push(((b_ab / a_ab) * (b_ba / a_ba)).sqrt());
        }
    }

    assert!(
        !ratios.is_empty(),
        "paired sampler produced no positive base timing"
    );
    let (a_mean, a_cv_pct) = mean_and_cv(&a_seconds);
    let (b_mean, b_cv_pct) = mean_and_cv(&b_seconds);
    ratios.sort_unstable_by(f64::total_cmp);
    PairedRatio {
        median: percentile(&ratios, 50),
        p5: percentile(&ratios, 5),
        p95: percentile(&ratios, 95),
        a_mean_us: a_mean * 1_000_000.0 / f64::from(inner),
        b_mean_us: b_mean * 1_000_000.0 / f64::from(inner),
        a_cv_pct,
        b_cv_pct,
        round_pairs: ratios.len(),
    }
}

fn raw_vector(seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    let mut v = Vec::with_capacity(DIM);
    for _ in 0..DIM {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        v.push((state >> 40) as f32 / (1u64 << 23) as f32 - 1.0);
    }
    v
}

fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

fn make_vector(centroids: &[Vec<f32>], c: usize, noise_seed: u64) -> Vec<f32> {
    let centroid = &centroids[c % centroids.len()];
    let noise = raw_vector(noise_seed);
    normalize(
        centroid
            .iter()
            .zip(&noise)
            .map(|(a, n)| a + NOISE * n)
            .collect(),
    )
}

fn recall_at_k(exact: &[String], approx: &[String]) -> f64 {
    let hits = approx.iter().filter(|id| exact.contains(id)).count();
    hits as f64 / exact.len().max(1) as f64
}

fn ndcg_at_k(exact: &[String], approx: &[String]) -> f64 {
    let k = exact.len().min(approx.len());
    if k == 0 {
        return 1.0;
    }
    let gain_for = |doc: &str| {
        exact
            .iter()
            .position(|id| id == doc)
            .map_or(0.0, |rank| (k - rank) as f64)
    };
    let dcg: f64 = approx
        .iter()
        .take(k)
        .enumerate()
        .map(|(rank, doc)| gain_for(doc) / ((rank + 2) as f64).log2())
        .sum();
    let ideal: f64 = (0..k)
        .map(|rank| (k - rank) as f64 / ((rank + 2) as f64).log2())
        .sum();
    dcg / ideal
}

fn bench_fsvi_int8_two_pass(c: &mut Criterion) {
    eprintln!(
        "[fsvi_int8_two_pass] binary_path={}",
        std::env::current_exe()
            .expect("resolve exact measured bench binary")
            .display()
    );
    let dir = std::env::temp_dir().join(format!("fsvi_int8_two_pass_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create bench dir");
    let path = dir.join("index.idx");

    let centroids: Vec<Vec<f32>> = (0..CLUSTERS)
        .map(|i| normalize(raw_vector(0xc000_0000 + i as u64)))
        .collect();
    let mut writer = VectorIndex::create(&path, "bench-384", DIM).expect("create fsvi index");
    for i in 0..N {
        let vector = make_vector(&centroids, i % CLUSTERS, i as u64 + 1);
        writer
            .write_record(&format!("doc-{i:06}"), &vector)
            .expect("write record");
    }
    writer.finish().expect("finish fsvi index");
    let index = VectorIndex::open(&path).expect("open fsvi index");

    let queries: Vec<Vec<f32>> = (0..QUERIES)
        .map(|q| make_vector(&centroids, q % CLUSTERS, 0xdead_0000 + q as u64))
        .collect();

    // Exact flat top-K (recall reference).
    let exact: Vec<Vec<String>> = queries
        .iter()
        .map(|q| {
            index
                .search_top_k(q, K, None)
                .expect("flat")
                .into_iter()
                .map(|h| h.doc_id.to_string())
                .collect()
        })
        .collect();

    // ── Recall/nDCG + exact-output parity over candidate_multiplier. ──
    for mult in [2usize, 3, 5, 10] {
        let mut orig_recall_total = 0.0;
        let mut orig_ndcg_total = 0.0;
        let mut candidate_recall_total = 0.0;
        let mut candidate_ndcg_total = 0.0;
        for (qi, query) in queries.iter().enumerate() {
            let orig_hits = index
                .bench_search_top_k_int8_two_pass_orig(query, K, mult)
                .expect("original int8");
            let candidate_hits = index
                .bench_search_top_k_int8_two_pass_row_block_candidate(query, K, mult)
                .expect("candidate int8");
            assert_eq!(candidate_hits.len(), orig_hits.len());
            for (candidate, orig) in candidate_hits.iter().zip(&orig_hits) {
                assert_eq!(candidate.index, orig.index, "query={qi} mult={mult}");
                assert_eq!(candidate.doc_id, orig.doc_id, "query={qi} mult={mult}");
                assert_eq!(
                    candidate.score.to_bits(),
                    orig.score.to_bits(),
                    "query={qi} mult={mult}"
                );
            }
            let orig: Vec<String> = orig_hits
                .into_iter()
                .map(|h| h.doc_id.to_string())
                .collect();
            let candidate: Vec<String> = candidate_hits
                .into_iter()
                .map(|h| h.doc_id.to_string())
                .collect();
            orig_recall_total += recall_at_k(&exact[qi], &orig);
            orig_ndcg_total += ndcg_at_k(&exact[qi], &orig);
            candidate_recall_total += recall_at_k(&exact[qi], &candidate);
            candidate_ndcg_total += ndcg_at_k(&exact[qi], &candidate);
        }
        let orig_recall = orig_recall_total / QUERIES as f64;
        let orig_ndcg = orig_ndcg_total / QUERIES as f64;
        let candidate_recall = candidate_recall_total / QUERIES as f64;
        let candidate_ndcg = candidate_ndcg_total / QUERIES as f64;
        assert!(
            orig_recall >= 0.98,
            "ORIGINAL recall gate failed at mult={mult}"
        );
        assert!(
            orig_ndcg >= 0.98,
            "ORIGINAL nDCG gate failed at mult={mult}"
        );
        assert_eq!(candidate_recall.to_bits(), orig_recall.to_bits());
        assert_eq!(candidate_ndcg.to_bits(), orig_ndcg.to_bits());
        eprintln!(
            "[fsvi_int8_two_pass] N={N} dim={DIM} k={K} mult={mult} \
             orig_recall@{K}={:.4} orig_ndcg@{K}={:.4} \
             candidate_recall@{K}={:.4} candidate_ndcg@{K}={:.4}",
            orig_recall, orig_ndcg, candidate_recall, candidate_ndcg
        );
    }

    let paired_inner = usize::try_from(PAIRED_INNER).expect("paired inner fits usize");
    assert_eq!(
        paired_inner % QUERIES,
        0,
        "each paired arm must cover one or more complete query corpora"
    );
    let paired_corpus_repeats = paired_inner / QUERIES;
    let mut paired_results = Vec::with_capacity(2);
    for mult in [3_usize, 5] {
        // Per-function/workload A/A null comes first. Each arm has its own query
        // cursor, so every timed batch visits the same complete 32-query corpus.
        let mut null_a_qi = 0_usize;
        let mut null_b_qi = 0_usize;
        let null = paired_median_ratio(
            PAIRED_ROUND_PAIRS,
            PAIRED_INNER,
            || {
                let query = &queries[null_a_qi];
                null_a_qi = (null_a_qi + 1) % QUERIES;
                black_box(
                    index
                        .bench_search_top_k_int8_two_pass_orig(
                            black_box(query),
                            black_box(K),
                            black_box(mult),
                        )
                        .expect("null base a"),
                );
            },
            || {
                let query = &queries[null_b_qi];
                null_b_qi = (null_b_qi + 1) % QUERIES;
                black_box(
                    index
                        .bench_search_top_k_int8_two_pass_orig(
                            black_box(query),
                            black_box(K),
                            black_box(mult),
                        )
                        .expect("null base b"),
                );
            },
        );
        let (null_floor_low, null_floor_high) = PairedRatio::calibrated_floor(null);
        eprintln!(
            "[paired-null] mult={mult} median={:.6} p5={:.6} p95={:.6} \
             floor_low={null_floor_low:.6} floor_high={null_floor_high:.6} \
             contains_one={} base_a_us={:.3} base_b_us={:.3} \
             cv_a_pct={:.3} cv_b_pct={:.3} round_pairs={} corpus_repeats={paired_corpus_repeats}",
            null.median,
            null.p5,
            null.p95,
            null.null_is_unbiased(),
            null.a_mean_us,
            null.b_mean_us,
            null.a_cv_pct,
            null.b_cv_pct,
            null.round_pairs
        );

        let mut base_qi = 0_usize;
        let mut candidate_qi = 0_usize;
        let lever = paired_median_ratio(
            PAIRED_ROUND_PAIRS,
            PAIRED_INNER,
            || {
                let query = &queries[base_qi];
                base_qi = (base_qi + 1) % QUERIES;
                black_box(
                    index
                        .bench_search_top_k_int8_two_pass_orig(
                            black_box(query),
                            black_box(K),
                            black_box(mult),
                        )
                        .expect("paired base"),
                );
            },
            || {
                let query = &queries[candidate_qi];
                candidate_qi = (candidate_qi + 1) % QUERIES;
                black_box(
                    index
                        .bench_search_top_k_int8_two_pass_row_block_candidate(
                            black_box(query),
                            black_box(K),
                            black_box(mult),
                        )
                        .expect("paired candidate"),
                );
            },
        );
        let calibrated_ratio = lever.calibrated_ratio(null);
        eprintln!(
            "[paired-lever] mult={mult} median={:.6} p5={:.6} p95={:.6} \
             calibrated_ratio={calibrated_ratio:.6} floor_low={null_floor_low:.6} \
             floor_high={null_floor_high:.6} \
             base_us={:.3} candidate_us={:.3} cv_base_pct={:.3} cv_candidate_pct={:.3} \
             round_pairs={} corpus_repeats={paired_corpus_repeats} verdict={}",
            lever.median,
            lever.p5,
            lever.p95,
            lever.a_mean_us,
            lever.b_mean_us,
            lever.a_cv_pct,
            lever.b_cv_pct,
            lever.round_pairs,
            lever.verdict_against(null)
        );
        paired_results.push((mult, null, lever));
    }

    let (_, mult3_null, mult3_lever) = paired_results[0];
    let (_, mult5_null, mult5_lever) = paired_results[1];
    let mult3_ratio = mult3_lever.calibrated_ratio(mult3_null);
    let mult5_ratio = mult5_lever.calibrated_ratio(mult5_null);
    let (mult3_floor_low, mult3_floor_high) = PairedRatio::calibrated_floor(mult3_null);
    let (_, mult5_floor_high) = PairedRatio::calibrated_floor(mult5_null);
    let gate = if !mult3_null.null_is_unbiased() || !mult5_null.null_is_unbiased() {
        "UNDECIDABLE_NULL_BIAS"
    } else if mult3_ratio > mult3_floor_high || mult5_ratio > mult5_floor_high {
        "REJECT_SIGNAL_PROFILE_REQUIRED"
    } else if mult3_ratio < mult3_floor_low {
        if mult3_ratio < 0.97 {
            "KEEP_SIGNAL"
        } else {
            "REJECT_SIGNAL_MISSES_RATCHET_PROFILE_REQUIRED"
        }
    } else {
        "UNDECIDABLE_PRIMARY_INSIDE_NULL_FLOOR"
    };
    eprintln!(
        "[paired-gate] mult3_ratio={mult3_ratio:.6} \
         mult3_floor_low={mult3_floor_low:.6} mult3_floor_high={mult3_floor_high:.6} \
         mult5_ratio={mult5_ratio:.6} mult5_floor_high={mult5_floor_high:.6} \
         ratchet=0.970000 exact_output_parity=true recall_ndcg_parity=true signal={gate}"
    );

    // ── Diagnostic/profile arms; never use their sequential estimates as verdict. ──
    // Both measured routines execute the same alternating AB/BA sequence and add
    // only their named implementation's elapsed time. Thus each implementation is
    // first and second equally often for the same query sequence. One Criterion
    // iteration is a fixed batch of `INTERLEAVE_BATCH` timed searches plus their
    // companion arm. Criterion therefore reports batch latency; divide its JSON
    // mean and standard deviation by `INTERLEAVE_BATCH` for per-search values. The
    // batch averages remote-worker scheduling jitter without splitting the arms.
    let mut g = c.benchmark_group("fsvi_int8_two_pass_ab");
    g.sampling_mode(SamplingMode::Flat);
    for mult in [3usize, 5] {
        g.bench_function(
            format!("paired_orig_mult{mult}_batch{INTERLEAVE_BATCH}"),
            |b| {
                b.iter_custom(|iters| {
                    let mut measured = Duration::ZERO;
                    for batch in 0..iters {
                        for lane in 0..INTERLEAVE_BATCH {
                            let ordinal = batch * INTERLEAVE_BATCH + lane;
                            let q = &queries[(ordinal as usize) % QUERIES];
                            if ordinal.is_multiple_of(2) {
                                let started = Instant::now();
                                let orig = index
                                    .bench_search_top_k_int8_two_pass_orig(black_box(q), K, mult)
                                    .expect("original int8");
                                measured += started.elapsed();
                                black_box(orig);
                                black_box(
                                    index
                                        .bench_search_top_k_int8_two_pass_row_block_candidate(
                                            black_box(q),
                                            K,
                                            mult,
                                        )
                                        .expect("candidate int8"),
                                );
                            } else {
                                black_box(
                                    index
                                        .bench_search_top_k_int8_two_pass_row_block_candidate(
                                            black_box(q),
                                            K,
                                            mult,
                                        )
                                        .expect("candidate int8"),
                                );
                                let started = Instant::now();
                                let orig = index
                                    .bench_search_top_k_int8_two_pass_orig(black_box(q), K, mult)
                                    .expect("original int8");
                                measured += started.elapsed();
                                black_box(orig);
                            }
                        }
                    }
                    measured
                });
            },
        );

        g.bench_function(
            format!("paired_candidate_mult{mult}_batch{INTERLEAVE_BATCH}"),
            |b| {
                b.iter_custom(|iters| {
                    let mut measured = Duration::ZERO;
                    for batch in 0..iters {
                        for lane in 0..INTERLEAVE_BATCH {
                            let ordinal = batch * INTERLEAVE_BATCH + lane;
                            let q = &queries[(ordinal as usize) % QUERIES];
                            if ordinal.is_multiple_of(2) {
                                black_box(
                                    index
                                        .bench_search_top_k_int8_two_pass_orig(
                                            black_box(q),
                                            K,
                                            mult,
                                        )
                                        .expect("original int8"),
                                );
                                let started = Instant::now();
                                let candidate = index
                                    .bench_search_top_k_int8_two_pass_row_block_candidate(
                                        black_box(q),
                                        K,
                                        mult,
                                    )
                                    .expect("candidate int8");
                                measured += started.elapsed();
                                black_box(candidate);
                            } else {
                                let started = Instant::now();
                                let candidate = index
                                    .bench_search_top_k_int8_two_pass_row_block_candidate(
                                        black_box(q),
                                        K,
                                        mult,
                                    )
                                    .expect("candidate int8");
                                measured += started.elapsed();
                                black_box(candidate);
                                black_box(
                                    index
                                        .bench_search_top_k_int8_two_pass_orig(
                                            black_box(q),
                                            K,
                                            mult,
                                        )
                                        .expect("original int8"),
                                );
                            }
                        }
                    }
                    measured
                });
            },
        );
    }
    g.finish();

    eprintln!("[fsvi_int8_two_pass] retained fixture at {}", dir.display());
}

criterion_group!(benches, bench_fsvi_int8_two_pass);
criterion_main!(benches);
