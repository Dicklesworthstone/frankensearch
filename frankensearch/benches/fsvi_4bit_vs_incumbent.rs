//! **Claim conversion**: `FSVI 4-bit two-pass` — "the fastest lossless
//! vector-search primitive" (`CHANGELOG.md:66`, `docs/PERF_LEDGER.md:825/827`)
//! measured against a REAL third-party incumbent running LIVE in this same
//! process invocation, over the same fixture bytes.
//!
//! ## Why this bench exists
//!
//! The published superlative rests entirely on self-vs-self A/Bs: the ledger
//! rows compare the 4-bit two-pass against *our own* int8 two-pass and *our own*
//! flat f16 scan. Neither is an incumbent. "Fastest lossless vector-search
//! primitive" is an unbounded claim about the world, so it needs an arm that is
//! not our code.
//!
//! ## The incumbent
//!
//! `faiss::IndexFlatIP` — the reference for exact ("lossless") vector search —
//! computes a dense score block `queries[nq x d] . corpus[d x N]` with a
//! cache-blocked SIMD GEMM, then reduces each row to top-k. That is exactly
//! what the incumbent arm here does, using `ndarray` 0.17 over `matrixmultiply`
//! 0.3 (runtime-dispatched AVX2 f32 kernels) — a third-party crate, not ours.
//! The GEMM is the dominant cost and is entirely third-party; the top-k
//! reduction is a bounded heap written for this bench, the same reduction shape
//! faiss uses.
//!
//! Both a batched (`nq=32`, GEMM) and an unbatched (`nq=1`, GEMV) incumbent arm
//! are measured, and the FASTER of the two is used as the official incumbent
//! number, so the incumbent is reported at its best.
//!
//! ## Thread asymmetry is the trap this bench is built to avoid
//!
//! Our 4-bit pass-1 is Rayon-parallel; the incumbent GEMM is single-threaded.
//! Comparing them as-shipped measures thread count, not the primitive. So the
//! candidate is measured TWICE — inside a pinned 1-thread Rayon pool
//! (like-for-like, the headline ratio) and in the default pool (as-shipped) —
//! and CPU/wall is published per arm.
//!
//! ## Run
//!
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/frankensearch/target \
//!   cargo bench -p frankensearch --bench fsvi_4bit_vs_incumbent
//! ```

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::hint::black_box;
use std::time::{Duration, Instant};

use frankensearch_index::VectorIndex;
use ndarray::{Array2, Axis};
use sha2::{Digest, Sha256};

const N: usize = 100_000;
const DIM: usize = 384;
const K: usize = 10;
const QUERIES: usize = 32;
const CLUSTERS: usize = 64;
const NOISE: f32 = 0.30;
/// The multiplier the ledger row calls "the sweet spot" and the one the
/// published claim is stated at.
const MULT: usize = 5;

// ─────────────────────────── fixture (byte-identical to the incumbent arm) ──

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

// ───────────────────────────────────── incumbent: faiss IndexFlatIP-shaped ──

/// Bounded top-k over one contiguous score row. Same reduction faiss performs
/// after its GEMM block. Returns positional ids, best score first.
fn top_k_row(scores: &[f32], k: usize) -> Vec<u32> {
    // Bounded insertion over a k-sized ascending-by-score buffer. k=10, so the
    // linear insert is cheaper than a BinaryHeap and matches faiss's small-k
    // reduction.
    let mut best: Vec<(f32, u32)> = Vec::with_capacity(k + 1);
    let mut cutoff = f32::NEG_INFINITY;
    for (i, &s) in scores.iter().enumerate() {
        if best.len() == k && s <= cutoff {
            continue;
        }
        let pos = best
            .iter()
            .position(|&(bs, _)| s > bs)
            .unwrap_or(best.len());
        best.insert(pos, (s, i as u32));
        if best.len() > k {
            best.pop();
        }
        if best.len() == k {
            cutoff = best[k - 1].0;
        }
    }
    best.into_iter().map(|(_, i)| i).collect()
}

/// Batched incumbent: one `matrixmultiply` sgemm producing a `[nq x N]` score
/// block, then a bounded top-k per row. This is the faiss `IndexFlatIP::search`
/// structure.
fn incumbent_batch(corpus: &Array2<f32>, queries: &Array2<f32>, k: usize) -> Vec<Vec<u32>> {
    let scores = queries.dot(&corpus.t());
    scores
        .axis_iter(Axis(0))
        .map(|row| {
            let contiguous;
            let slice = match row.as_slice() {
                Some(s) => s,
                None => {
                    contiguous = row.to_owned();
                    contiguous.as_slice().expect("owned row is contiguous")
                }
            };
            top_k_row(slice, k)
        })
        .collect()
}

/// Unbatched incumbent: one GEMV per query. Measured so the incumbent can be
/// reported at whichever of the two shapes is faster.
fn incumbent_single(corpus: &Array2<f32>, queries: &Array2<f32>, k: usize) -> Vec<Vec<u32>> {
    queries
        .axis_iter(Axis(0))
        .map(|q| {
            let scores = corpus.dot(&q);
            top_k_row(scores.as_slice().expect("gemv output is contiguous"), k)
        })
        .collect()
}

// ─────────────────────────────────────────────── process self-identification ──

fn elf_sha256() -> String {
    match std::fs::read("/proc/self/exe") {
        Ok(bytes) => {
            let mut h = Sha256::new();
            h.update(&bytes);
            // sha2 0.11 returns `hybrid_array::Array`, which (unlike 0.10's
            // `GenericArray`) has no `LowerHex` impl — format the bytes.
            h.finalize().iter().fold(String::new(), |mut s, b| {
                use std::fmt::Write as _;
                let _ = write!(s, "{b:02x}");
                s
            })
        }
        Err(e) => format!("UNAVAILABLE({e})"),
    }
}

fn hostname() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "UNKNOWN".into())
}

fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|v| v.trim().to_string())
        })
        .unwrap_or_else(|| "UNKNOWN".into())
}

/// Threads that actually exist in this process right now — observed, not
/// configured.
fn observed_threads() -> usize {
    std::fs::read_dir("/proc/self/task")
        .map(|d| d.count())
        .unwrap_or(0)
}

/// Process CPU time (utime + stime) in nanoseconds, from `/proc/self/stat`.
/// Clock-tick granularity (10 ms), so it is accumulated across all rounds per
/// arm rather than read per iteration.
fn cpu_time_ns() -> u128 {
    let Ok(stat) = std::fs::read_to_string("/proc/self/stat") else {
        return 0;
    };
    // comm can contain spaces/parens; fields are counted after the final ')'.
    let Some(rest) = stat.rsplit_once(american_paren()).map(|(_, r)| r) else {
        return 0;
    };
    let f: Vec<&str> = rest.split_whitespace().collect();
    // After ')' the next field is state (index 0); utime is field 14 overall
    // (1-based), i.e. index 11 here; stime index 12.
    let utime: u128 = f.get(11).and_then(|v| v.parse().ok()).unwrap_or(0);
    let stime: u128 = f.get(12).and_then(|v| v.parse().ok()).unwrap_or(0);
    let hz: u128 = 100; // USER_HZ is 100 on every Linux target we run on.
    (utime + stime) * 1_000_000_000 / hz
}

const fn american_paren() -> char {
    ')'
}

// ───────────────────────────────────────────────────────────── statistics ──

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx]
}

#[derive(Default)]
struct Arm {
    name: &'static str,
    wall_ns: Vec<f64>,
    cpu_ns_total: u128,
    threads_seen: usize,
}

impl Arm {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            ..Default::default()
        }
    }
    fn median_wall(&self) -> f64 {
        let mut v = self.wall_ns.clone();
        median(&mut v)
    }
    fn cpu_over_wall(&self) -> f64 {
        let wall: f64 = self.wall_ns.iter().sum();
        if wall <= 0.0 {
            return f64::NAN;
        }
        self.cpu_ns_total as f64 / wall
    }
}

/// Time one arm for one round: `f` runs the full 32-query workload.
fn time_round<F: FnOnce() -> Vec<Vec<u32>>>(arm: &mut Arm, f: F) -> Vec<Vec<u32>> {
    let cpu0 = cpu_time_ns();
    let t0 = Instant::now();
    let out = black_box(f());
    let wall = t0.elapsed().as_nanos() as f64;
    arm.cpu_ns_total += cpu_time_ns().saturating_sub(cpu0);
    arm.wall_ns.push(wall);
    arm.threads_seen = arm.threads_seen.max(observed_threads());
    out
}

// ──────────────────────────────────────────────────────────────────── main ──

fn main() {
    // 15 rounds left the A/A null median at 1.0321 — a hair over the 1.030 bar,
    // almost certainly small-sample noise in a median-of-15. The timed section
    // costs well under a second per round, so the cost of a much tighter null is
    // negligible next to the build.
    let rounds: usize = std::env::var("CONV_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(61);

    println!("=== CLAIM CONVERSION: FSVI 4-bit two-pass vs third-party incumbent ===");
    println!("claim_under_test      = \"the fastest lossless vector-search primitive\"");
    println!("claim_sources         = CHANGELOG.md:66 ; docs/PERF_LEDGER.md:825,827");
    println!("incumbent             = ndarray 0.17 / matrixmultiply 0.3 sgemm+sgemv (faiss IndexFlatIP shape)");
    println!("host                  = {}", hostname());
    println!("cpu_model             = {}", cpu_model());
    println!("elf_sha256            = {}", elf_sha256());
    println!(
        "nproc                 = {}",
        std::thread::available_parallelism().map_or(0, std::num::NonZeroUsize::get)
    );
    println!("rayon_global_threads  = {}", rayon::current_num_threads());
    println!("threads_at_start      = {}", observed_threads());
    println!("N={N} DIM={DIM} K={K} QUERIES={QUERIES} MULT={MULT} rounds={rounds}");

    // ── Build the fixture ONCE; both arms read the same vectors. ────────────
    let dir = std::env::temp_dir().join(format!("fsvi_4bit_vs_incumbent_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create bench dir");
    let path = dir.join("index.idx");

    let centroids: Vec<Vec<f32>> = (0..CLUSTERS)
        .map(|i| normalize(raw_vector(0xc000_0000 + i as u64)))
        .collect();

    let mut flat: Vec<f32> = Vec::with_capacity(N * DIM);
    let mut writer = VectorIndex::create(&path, "conv-384", DIM).expect("create fsvi index");
    for i in 0..N {
        let vector = make_vector(&centroids, i % CLUSTERS, i as u64 + 1);
        writer
            .write_record(&format!("doc-{i:06}"), &vector)
            .expect("write record");
        flat.extend_from_slice(&vector);
    }
    writer.finish().expect("finish fsvi index");
    let index = VectorIndex::open(&path).expect("open fsvi index");

    // Incumbent corpus: the SAME f32 vectors, in the SAME order, so positional
    // id i corresponds to doc-{i:06} in the FSVI index.
    let corpus = Array2::from_shape_vec((N, DIM), flat).expect("corpus shape");

    let query_vecs: Vec<Vec<f32>> = (0..QUERIES)
        .map(|q| make_vector(&centroids, q % CLUSTERS, 0xdead_0000 + q as u64))
        .collect();
    let queries = Array2::from_shape_vec(
        (QUERIES, DIM),
        query_vecs.iter().flat_map(|q| q.iter().copied()).collect(),
    )
    .expect("query shape");

    println!(
        "corpus_f32_bytes      = {} ({:.1} MiB)  [incumbent working set]",
        N * DIM * 4,
        (N * DIM * 4) as f64 / (1024.0 * 1024.0)
    );
    println!(
        "packed_4bit_bytes     = {} ({:.1} MiB)  [candidate pass-1 working set]",
        N * DIM / 2,
        (N * DIM / 2) as f64 / (1024.0 * 1024.0)
    );

    // ── Correctness: is the claim's "lossless" true against the WORLD, or ────
    //    only against our own f16 exact scan? Both are reported.
    let to_ids = |hits: Vec<frankensearch_core::VectorHit>| -> BTreeSet<String> {
        hits.into_iter().map(|h| h.doc_id.to_string()).collect()
    };
    let inc_ids: Vec<BTreeSet<String>> = incumbent_batch(&corpus, &queries, K)
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|i| format!("doc-{i:06}"))
                .collect::<BTreeSet<String>>()
        })
        .collect();

    let mut agree_cand_vs_incumbent = 0usize;
    let mut agree_cand_vs_ours_exact = 0usize;
    let mut agree_ours_exact_vs_incumbent = 0usize;
    for (qi, q) in query_vecs.iter().enumerate() {
        let ours_exact = to_ids(index.search_top_k(q, K, None).expect("flat"));
        let cand = to_ids(
            index
                .search_top_k_4bit_two_pass(q, K, MULT)
                .expect("4bit two-pass"),
        );
        if cand == inc_ids[qi] {
            agree_cand_vs_incumbent += 1;
        }
        if cand == ours_exact {
            agree_cand_vs_ours_exact += 1;
        }
        if ours_exact == inc_ids[qi] {
            agree_ours_exact_vs_incumbent += 1;
        }
    }
    println!("\n--- LOSSLESSNESS (top-{K} doc-id set equality, {QUERIES} queries) ---");
    println!("candidate == ours_exact_f16     : {agree_cand_vs_ours_exact}/{QUERIES}   <- the claim's own definition");
    println!("candidate == incumbent_f32      : {agree_cand_vs_incumbent}/{QUERIES}   <- the world's definition");
    println!("ours_exact_f16 == incumbent_f32 : {agree_ours_exact_vs_incumbent}/{QUERIES}   <- our f16 storage vs f32 truth");

    // ── Timed, interleaved rounds. ──────────────────────────────────────────
    // inc_a / inc_b are the SAME arm run twice: their per-round ratio is the
    // A/A null. Order is reversed on odd rounds so drift cancels (AB/BA).
    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("1-thread pool");

    let mut inc_a = Arm::new("incumbent_batch32_A");
    let mut inc_b = Arm::new("incumbent_batch32_B");
    let mut inc_gemv = Arm::new("incumbent_gemv_nq1");
    let mut cand_t1 = Arm::new("cand_4bit_mult5_threads1");
    let mut cand_def = Arm::new("cand_4bit_mult5_default");
    let mut ours_flat = Arm::new("ours_exact_flat_default");

    let run_cand = |pool: Option<&rayon::ThreadPool>| -> Vec<Vec<u32>> {
        let work = || {
            query_vecs
                .iter()
                .map(|q| {
                    index
                        .search_top_k_4bit_two_pass(black_box(q), K, MULT)
                        .expect("4bit")
                        .into_iter()
                        .map(|h| h.index)
                        .collect::<Vec<u32>>()
                })
                .collect::<Vec<_>>()
        };
        match pool {
            Some(p) => p.install(work),
            None => work(),
        }
    };
    let run_ours_flat = || -> Vec<Vec<u32>> {
        query_vecs
            .iter()
            .map(|q| {
                index
                    .search_top_k(black_box(q), K, None)
                    .expect("flat")
                    .into_iter()
                    .map(|h| h.index)
                    .collect::<Vec<u32>>()
            })
            .collect::<Vec<_>>()
    };

    // Warm-up: page in the mmap, build the lazy 4-bit slab, prime caches.
    let _ = incumbent_batch(&corpus, &queries, K);
    let _ = run_cand(None);
    let _ = run_ours_flat();

    for round in 0..rounds {
        if round % 2 == 0 {
            time_round(&mut inc_a, || incumbent_batch(&corpus, &queries, K));
            time_round(&mut cand_t1, || run_cand(Some(&pool1)));
            time_round(&mut cand_def, || run_cand(None));
            time_round(&mut inc_gemv, || incumbent_single(&corpus, &queries, K));
            time_round(&mut ours_flat, run_ours_flat);
            time_round(&mut inc_b, || incumbent_batch(&corpus, &queries, K));
        } else {
            time_round(&mut inc_b, || incumbent_batch(&corpus, &queries, K));
            time_round(&mut ours_flat, run_ours_flat);
            time_round(&mut inc_gemv, || incumbent_single(&corpus, &queries, K));
            time_round(&mut cand_def, || run_cand(None));
            time_round(&mut cand_t1, || run_cand(Some(&pool1)));
            time_round(&mut inc_a, || incumbent_batch(&corpus, &queries, K));
        }
    }

    // ── A/A null from the two incumbent replicates. ─────────────────────────
    let mut null_ratios: Vec<f64> = inc_a
        .wall_ns
        .iter()
        .zip(&inc_b.wall_ns)
        .map(|(a, b)| b / a)
        .collect();
    let null_median = median(&mut null_ratios);
    let null_sorted = null_ratios.clone();
    let null_p5 = percentile(&null_sorted, 0.05);
    let null_p95 = percentile(&null_sorted, 0.95);

    // Pooled incumbent = both replicates of the same arm.
    let mut pooled: Vec<f64> = inc_a
        .wall_ns
        .iter()
        .chain(&inc_b.wall_ns)
        .copied()
        .collect();
    let inc_batch_med = median(&mut pooled);
    let inc_gemv_med = inc_gemv.median_wall();
    // Report the incumbent at its BEST of the two shapes.
    let (inc_best_name, inc_best) = if inc_gemv_med < inc_batch_med {
        ("incumbent_gemv_nq1", inc_gemv_med)
    } else {
        ("incumbent_batch32", inc_batch_med)
    };

    let per_q = |ns: f64| ns / QUERIES as f64 / 1000.0; // us/query

    println!("\n--- PER-ARM (median wall over {rounds} rounds; each round = {QUERIES} queries) ---");
    println!(
        "{:<28} {:>12} {:>14} {:>10} {:>9}",
        "arm", "median_us/q", "cpu/wall", "max_thr", "n"
    );
    for a in [
        &inc_a, &inc_b, &inc_gemv, &cand_t1, &cand_def, &ours_flat,
    ] {
        println!(
            "{:<28} {:>12.2} {:>14.2} {:>10} {:>9}",
            a.name,
            per_q(a.median_wall()),
            a.cpu_over_wall(),
            a.threads_seen,
            a.wall_ns.len()
        );
    }

    println!("\n--- A/A NULL (incumbent measured twice, interleaved) ---");
    println!("null_median = {null_median:.4}   null_p5 = {null_p5:.4}   null_p95 = {null_p95:.4}");
    let null_clean = (null_median - 1.0).abs() <= 0.03;
    println!(
        "null_gate(median within 1.000+/-0.030) = {}",
        if null_clean { "CLEAN" } else { "DIRTY -- ratios below are NOT decidable" }
    );

    println!("\n--- RATIOS vs INCUMBENT ({inc_best_name} @ {:.2} us/query) ---", per_q(inc_best));
    let ratio_t1 = cand_t1.median_wall() / inc_best;
    let ratio_def = cand_def.median_wall() / inc_best;
    let ratio_flat = ours_flat.median_wall() / inc_best;
    let decide = |r: f64| -> &'static str {
        if !null_clean {
            "UNDECIDABLE (dirty null)"
        } else if r < null_p5 {
            "CANDIDATE FASTER (outside null)"
        } else if r > null_p95 {
            "CANDIDATE SLOWER (outside null)"
        } else {
            "WASH (inside null)"
        }
    };
    println!(
        "ratio_like_for_like(1 thread both) = {ratio_t1:.4}  ({:.2}x)  {}",
        1.0 / ratio_t1,
        decide(ratio_t1)
    );
    println!(
        "ratio_as_shipped(cand {} thr vs inc 1 thr) = {ratio_def:.4}  ({:.2}x)  {}",
        cand_def.threads_seen,
        1.0 / ratio_def,
        decide(ratio_def)
    );
    println!(
        "ratio_ours_exact_flat              = {ratio_flat:.4}  ({:.2}x)  {}",
        1.0 / ratio_flat,
        decide(ratio_flat)
    );
    println!(
        "\nself_vs_self_context: cand_default / ours_exact_flat = {:.4} ({:.2}x) <- the number the published claim was built on",
        cand_def.median_wall() / ours_flat.median_wall(),
        ours_flat.median_wall() / cand_def.median_wall()
    );
    println!("threads_at_end        = {}", observed_threads());
    println!("elf_sha256            = {}", elf_sha256());

    std::fs::remove_dir_all(&dir).ok();
    // Keep the process alive long enough that a `perf`/`time` wrapper attributes
    // teardown to us, not to the harness.
    std::thread::sleep(Duration::from_millis(1));
}
