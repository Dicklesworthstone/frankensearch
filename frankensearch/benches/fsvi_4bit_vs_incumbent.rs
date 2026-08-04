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
            // `option_if_let_else` is wrong for this idiom: the fallback buffer
            // must outlive the borrow, which is exactly why `contiguous` is
            // declared before the match and initialized inside it. A
            // `map_or_else` closure owns its temporary, so the slice it yields
            // would not live long enough.
            #[allow(clippy::option_if_let_else)]
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

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        f64::midpoint(v[n / 2 - 1], v[n / 2])
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let scaled = (sorted.len() - 1) as f64 * p.clamp(0.0, 1.0);
    // `cast_sign_loss` cannot see that this is non-negative, but it is: both
    // factors are clamped to >= 0 above and `max(0.0)` pins the rounded value.
    // The arithmetic is deliberately left exactly as it was — this index picks
    // the A/A null band that decides FASTER/WASH/SLOWER, so switching to
    // integer rounding to satisfy the lint would shift a verdict at the margin.
    #[allow(clippy::cast_sign_loss)]
    let idx = scaled.round().max(0.0) as usize;
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

/// Time one arm for one slot: `f` runs the full 32-query workload. Returns the
/// wall time in ns so the caller can form bracketed ratios.
fn time_one<F: FnOnce() -> Vec<Vec<u32>>>(arm: &mut Arm, f: F) -> f64 {
    let cpu0 = cpu_time_ns();
    let t0 = Instant::now();
    let out = black_box(f());
    let wall = t0.elapsed().as_nanos() as f64;
    arm.cpu_ns_total += cpu_time_ns().saturating_sub(cpu0);
    arm.wall_ns.push(wall);
    arm.threads_seen = arm.threads_seen.max(observed_threads());
    drop(out);
    wall
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
    println!(
        "incumbent             = ndarray 0.17 / matrixmultiply 0.3 sgemm+sgemv (faiss IndexFlatIP shape)"
    );
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
    println!(
        "candidate == ours_exact_f16     : {agree_cand_vs_ours_exact}/{QUERIES}   <- the claim's own definition"
    );
    println!(
        "candidate == incumbent_f32      : {agree_cand_vs_incumbent}/{QUERIES}   <- the world's definition"
    );
    println!(
        "ours_exact_f16 == incumbent_f32 : {agree_ours_exact_vs_incumbent}/{QUERIES}   <- our f16 storage vs f32 truth"
    );

    // ── Timed rounds, BRACKETED. ────────────────────────────────────────────
    // Runs 1 and 2 used a whole-round A/A (inc_a first, inc_b last) and the null
    // came back dirty both times — 1.0321 at 15 rounds, 1.0598 at 61. Raising n
    // made it WORSE, because the dispersion is host drift over the ~600 ms that
    // separated the two replicates, not sampling error.
    //
    // So: bracket instead. Every subject X is measured BETWEEN two incumbent
    // runs and scored as X / mean(I_before, I_after), which cancels linear drift
    // across the subject's own measurement window. The A/A null is an incumbent
    // run scored the SAME way, so null and candidate ratios share one structure
    // and one drift-cancellation — the null now measures exactly the noise the
    // candidate ratio is exposed to. This is the bracketing control this ledger
    // already validated at PERF_LEDGER.md:822-824, where it produced B/A 0.9999
    // on this same fleet.
    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("1-thread pool");

    const S_CAND_T1: usize = 0;
    const S_CAND_DEF: usize = 1;
    const S_OURS_FLAT: usize = 2;
    const S_INC_NULL: usize = 3;
    const S_INC_GEMV: usize = 4;
    const SUBJECTS: usize = 5;
    let subject_names: [&'static str; SUBJECTS] = [
        "cand_4bit_mult5_threads1",
        "cand_4bit_mult5_default",
        "ours_exact_flat_default",
        "incumbent_null_AA",
        "incumbent_gemv_nq1",
    ];

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
        // `option_if_let_else` is wrong here: `work` is a FnOnce consumed by
        // whichever branch runs, so a `map_or_else` form would have to move it
        // into two closures at once.
        #[allow(clippy::option_if_let_else)]
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

    let mut arms: Vec<Arm> = subject_names.iter().map(|n| Arm::new(n)).collect();
    let mut inc_arm = Arm::new("incumbent_batch32_bracket");
    let mut ratios: Vec<Vec<f64>> = vec![Vec::new(); SUBJECTS];

    for round in 0..rounds {
        let mut prev = time_one(&mut inc_arm, || incumbent_batch(&corpus, &queries, K));
        for slot in 0..SUBJECTS {
            // Rotate so every subject visits every position within the round;
            // no subject is permanently adjacent to the warm-up edge.
            let s = (slot + round) % SUBJECTS;
            let t = match s {
                S_CAND_T1 => time_one(&mut arms[s], || run_cand(Some(&pool1))),
                S_CAND_DEF => time_one(&mut arms[s], || run_cand(None)),
                S_OURS_FLAT => time_one(&mut arms[s], run_ours_flat),
                S_INC_NULL => time_one(&mut arms[s], || incumbent_batch(&corpus, &queries, K)),
                _ => time_one(&mut arms[s], || incumbent_single(&corpus, &queries, K)),
            };
            let next = time_one(&mut inc_arm, || incumbent_batch(&corpus, &queries, K));
            ratios[s].push(t / f64::midpoint(prev, next));
            prev = next;
        }
    }

    // ── A/A null: an incumbent run scored by the SAME bracketing rule. ──────
    let mut null_ratios = ratios[S_INC_NULL].clone();
    let null_median = median(&mut null_ratios);
    let null_sorted = null_ratios.clone();
    let null_lower_bound = percentile(&null_sorted, 0.05);
    let null_upper_bound = percentile(&null_sorted, 0.95);
    let null_clean = (null_median - 1.0).abs() <= 0.03;

    let per_q = |ns: f64| ns / QUERIES as f64 / 1000.0; // us/query

    println!(
        "\n--- PER-ARM (median wall; {rounds} rounds x {SUBJECTS} slots, each = {QUERIES} queries) ---"
    );
    println!(
        "{:<30} {:>12} {:>10} {:>8}",
        "arm", "median_us/q", "cpu/wall", "n"
    );
    println!(
        "{:<30} {:>12.2} {:>10.2} {:>8}",
        inc_arm.name,
        per_q(inc_arm.median_wall()),
        inc_arm.cpu_over_wall(),
        inc_arm.wall_ns.len()
    );
    for a in &arms {
        println!(
            "{:<30} {:>12.2} {:>10.2} {:>8}",
            a.name,
            per_q(a.median_wall()),
            a.cpu_over_wall(),
            a.wall_ns.len()
        );
    }
    println!(
        "(cpu/wall is the per-arm concurrency evidence; /proc/self/task is \
process-wide and cannot distinguish arms)"
    );

    println!("\n--- A/A NULL (incumbent bracketed by incumbents, same rule as every ratio) ---");
    println!(
        "null_median = {null_median:.4}   null_p5 = {null_lower_bound:.4}   null_p95 = {null_upper_bound:.4}"
    );
    println!(
        "null_gate(median within 1.000+/-0.030) = {}",
        if null_clean {
            "CLEAN"
        } else {
            "DIRTY -- ratios below are NOT decidable"
        }
    );

    println!(
        "\n--- BRACKETED RATIOS vs incumbent_batch32 ({:.2} us/query) ---",
        per_q(inc_arm.median_wall())
    );
    let decide = |r: f64| -> &'static str {
        if !null_clean {
            "UNDECIDABLE (dirty null)"
        } else if r < null_lower_bound {
            "SUBJECT FASTER than incumbent (outside null)"
        } else if r > null_upper_bound {
            "SUBJECT SLOWER than incumbent (outside null)"
        } else {
            "WASH (inside null)"
        }
    };
    for s in [S_CAND_T1, S_CAND_DEF, S_OURS_FLAT, S_INC_GEMV] {
        let mut r = ratios[s].clone();
        let m = median(&mut r);
        let sorted = r.clone();
        println!(
            "{:<26} = {m:.4}  ({:.2}x)  [p5 {:.4}, p95 {:.4}]  {}",
            subject_names[s],
            1.0 / m,
            percentile(&sorted, 0.05),
            percentile(&sorted, 0.95),
            decide(m)
        );
    }
    let mut cdf = ratios[S_CAND_DEF].clone();
    let mut ofl = ratios[S_OURS_FLAT].clone();
    let (ratio_def, ratio_flat) = (median(&mut cdf), median(&mut ofl));
    println!(
        "\nself_vs_self_context: cand_default / ours_exact_flat = {:.4} ({:.2}x) \
<- the number the published claim was built on",
        ratio_def / ratio_flat,
        ratio_flat / ratio_def
    );
    println!("threads_at_end        = {}", observed_threads());
    println!("elf_sha256            = {}", elf_sha256());

    std::fs::remove_dir_all(&dir).ok();
    // Keep the process alive long enough that a `perf`/`time` wrapper attributes
    // teardown to us, not to the harness.
    std::thread::sleep(Duration::from_millis(1));
}
