//! Graph-rank (query-biased `PageRank`) power-iteration benchmark.
//!
//! `GraphRanker::rank_phase1` ran the power iteration over a `HashMap<String, f64>` rebuilt every
//! iteration, cloning a `String` `doc_id` key on every teleport/edge relaxation (all keys already
//! present, so the clones were dead) and re-checking each edge's weight finiteness every pass. The
//! new path dense-indexes the graph once and iterates over reused `Vec<f64>` buffers (CSR edges,
//! hoisted weight filter). `rank_old`/`rank_new` below reproduce those two shapes self-contained.
//!
//! **Read this before citing a number from here (bd-i40y).** Until 2026-07-10 this bench imported
//! only `std` and `criterion`: it never linked `frankensearch_fusion`, so `GraphRanker::rank_phase1`
//! — the function it is named for — had **0.000% self-time** and was never called. A REJECT row
//! (flat CSR, `NEGATIVE_EVIDENCE` 2026-07-09) was decided on `rank_new` alone. The `paired_prod` /
//! `paired_copy` arms now bench the **shipped symbol** against that copy, with a reachability gate
//! (`rank_phase1` must return `Some`) and a printed fidelity diagnostic. Measured: the copy ranks
//! identically (top-50 agreement 50/50) but runs **1.11×/1.13× cheaper** than production, because it
//! is handed a pre-normalized personalization and skips the `ranks_map` rebuild and `ScoredResult`
//! construction. It is a faithful *ranker* proxy and a biased *cost* proxy.
//!
//! `hash_siphash` vs `hash_ahash` measures the shipped dense-index hasher change: both arms call
//! the same generic production implementation, and the lookup table is never iterated. The bench
//! asserts result count, doc id, score bits, ties, and ordering before an alternating-round median
//! A/B with an A/A null control. `paired_shipped_csr` vs `paired_flat_csr` retains the earlier
//! layout lever: `rank_phase1` vs its one-variable
//! twin `rank_phase1_flat` (feature `bench-internals`), asserted **bit-identical** so the arms differ
//! by edge layout alone. `[csr-upper]` (bd-5hlw) adds the single-pass twin `rank_phase1_flat_upper`
//! (offsets from the `edges.len()` upper bound, no counting pass — the "strongest form") vs shipped:
//! measured a WASH inside the A/A null floor on both fixtures (0.957×/0.969×). `paired_null_a` /
//! `paired_null_b` are an **A/A null control** — the same arm
//! twice — measuring the harness noise floor. Read the null ratio before believing any lever ratio:
//! an effect smaller than the floor is noise. Measured floor 1.0015/1.0032 (hz2, 120 samples); flat
//! CSR loses by 1.206×/1.306×, confirming the 2026-07-09 REJECT at production fidelity.
//!
//! Run with (remote, fail-closed; a local build is a disk-pressure incident). One binary, one
//! invocation — an A/B split across two `rch` invocations lands on different workers and is invalid:
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR \
//!   rch exec -- cargo bench -p frankensearch-fusion \
//!   --features graph,bench-internals --bench graph_rank
//! ```

use std::collections::HashMap;
use std::hint::black_box;
use std::time::{Duration, Instant};

use asupersync::Cx;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_core::{DocumentGraph, EdgeType, VectorHit};
use frankensearch_fusion::GraphRanker;
use frankensearch_fusion::bench_support::paired_median_ratio;

const RESTART: f64 = 0.15;
const WALK: f64 = 1.0 - RESTART;
const MAX_ITER: usize = 20;
const TOL: f64 = 1e-6;

/// Calls per timed region in the interleaved paired sampler.
const PAIR_BATCH: u32 = 4;

type Adj = HashMap<String, Vec<(String, f32)>>;

fn finalize(mut ranks: Vec<(String, f64)>, limit: usize) -> Vec<String> {
    let total: f64 = ranks.iter().map(|(_, s)| *s).sum();
    if total > 0.0 {
        for (_, s) in &mut ranks {
            *s /= total;
        }
    }
    let mut out: Vec<(String, f64)> = ranks
        .into_iter()
        .filter(|(_, s)| s.is_finite() && *s > 0.0)
        .collect();
    out.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out.truncate(limit);
    out.into_iter().map(|(d, _)| d).collect()
}

// ── OLD: HashMap rebuilt per iteration, clone-keyed entry() ───────────────────
fn rank_old(adj: &Adj, pers: &[(String, f64)], limit: usize) -> Vec<String> {
    let mut ranks: HashMap<String, f64> = adj.keys().cloned().map(|d| (d, 0.0)).collect();
    for (d, s) in pers {
        ranks.insert(d.clone(), *s);
    }
    let out_sum: HashMap<String, f64> = adj
        .iter()
        .map(|(d, edges)| {
            let s: f64 = edges
                .iter()
                .map(|(_, w)| f64::from(*w))
                .filter(|w| w.is_finite() && *w > 0.0)
                .sum();
            (d.clone(), s)
        })
        .collect();
    for _ in 0..MAX_ITER {
        let mut next: HashMap<String, f64> = adj.keys().cloned().map(|d| (d, 0.0)).collect();
        for (d, w) in pers {
            *next.entry(d.clone()).or_insert(0.0) += RESTART * w;
        }
        let dangling: f64 = ranks
            .iter()
            .filter_map(|(d, rk)| {
                (out_sum.get(d).copied().unwrap_or(0.0) <= f64::EPSILON).then_some(*rk)
            })
            .sum();
        if dangling > 0.0 {
            for (d, w) in pers {
                *next.entry(d.clone()).or_insert(0.0) += WALK * dangling * w;
            }
        }
        for (d, edges) in adj {
            let rk = ranks.get(d).copied().unwrap_or(0.0);
            if rk <= 0.0 {
                continue;
            }
            let ot = out_sum.get(d).copied().unwrap_or(0.0);
            if ot <= f64::EPSILON {
                continue;
            }
            let base = WALK * rk / ot;
            for (nb, w) in edges {
                let w = f64::from(*w);
                if !w.is_finite() || w <= 0.0 {
                    continue;
                }
                *next.entry(nb.clone()).or_insert(0.0) += base * w;
            }
        }
        let l1: f64 = ranks
            .iter()
            .map(|(d, old)| (old - next.get(d).unwrap_or(&0.0)).abs())
            .sum();
        ranks = next;
        if l1 < TOL {
            break;
        }
    }
    finalize(ranks.into_iter().collect(), limit)
}

// ── NEW: dense index, reused Vec<f64> buffers, hoisted weight filter ──────────
fn rank_new(adj: &Adj, pers: &[(String, f64)], limit: usize) -> Vec<String> {
    let n = adj.len();
    let mut nodes: Vec<&str> = Vec::with_capacity(n);
    let mut idx: HashMap<&str, usize> = HashMap::with_capacity(n);
    for d in adj.keys() {
        idx.insert(d.as_str(), nodes.len());
        nodes.push(d.as_str());
    }
    let mut out_sum = vec![0.0_f64; n];
    let mut csr: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for (d, edges) in adj {
        let src = idx[d.as_str()];
        let mut sum = 0.0_f64;
        let mut row = Vec::with_capacity(edges.len());
        for (nb, w) in edges {
            let w = f64::from(*w);
            if !w.is_finite() || w <= 0.0 {
                continue;
            }
            sum += w;
            if let Some(&dst) = idx.get(nb.as_str()) {
                row.push((dst, w));
            }
        }
        out_sum[src] = sum;
        csr[src] = row;
    }
    let seeds: Vec<(usize, f64)> = pers
        .iter()
        .filter_map(|(d, w)| idx.get(d.as_str()).map(|&i| (i, *w)))
        .collect();
    let mut ranks = vec![0.0_f64; n];
    for &(i, w) in &seeds {
        ranks[i] = w;
    }
    let mut next = vec![0.0_f64; n];
    for _ in 0..MAX_ITER {
        next.fill(0.0);
        for &(i, w) in &seeds {
            next[i] += RESTART * w;
        }
        let dangling: f64 = (0..n)
            .filter(|&i| out_sum[i] <= f64::EPSILON)
            .map(|i| ranks[i])
            .sum();
        if dangling > 0.0 {
            for &(i, w) in &seeds {
                next[i] += WALK * dangling * w;
            }
        }
        for src in 0..n {
            let rk = ranks[src];
            if rk <= 0.0 {
                continue;
            }
            let ot = out_sum[src];
            if ot <= f64::EPSILON {
                continue;
            }
            let base = WALK * rk / ot;
            for &(dst, w) in &csr[src] {
                next[dst] += base * w;
            }
        }
        let l1: f64 = (0..n).map(|i| (ranks[i] - next[i]).abs()).sum();
        std::mem::swap(&mut ranks, &mut next);
        if l1 < TOL {
            break;
        }
    }
    let ranks_v: Vec<(String, f64)> = nodes
        .iter()
        .zip(ranks.iter())
        .map(|(&d, &r)| (d.to_owned(), r))
        .collect();
    finalize(ranks_v, limit)
}

fn make_graph(n: usize, deg: usize) -> (Adj, Vec<(String, f64)>) {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ (n as u64);
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let n_u64 = u64::try_from(n).expect("benchmark node count fits in u64");
    let mut adj: Adj = HashMap::new();
    for i in 0..n {
        adj.entry(format!("d{i:05}")).or_default();
    }
    for i in 0..n {
        for _ in 0..deg {
            let j = usize::try_from(next() % n_u64).expect("reduced node index fits in usize");
            if j != i {
                let w = 0.25 + (next() % 1000) as f32 / 1000.0;
                adj.get_mut(&format!("d{i:05}"))
                    .unwrap()
                    .push((format!("d{j:05}"), w));
                adj.entry(format!("d{j:05}")).or_default();
            }
        }
    }
    // 10 normalized seeds.
    let raw: Vec<(String, f64)> = (0..10)
        .map(|s| (format!("d{:05}", (s * 37) % n), 0.5 + s as f64 * 0.05))
        .collect();
    let tot: f64 = raw.iter().map(|(_, w)| w).sum();
    let pers = raw.into_iter().map(|(d, w)| (d, w / tot)).collect();
    (adj, pers)
}

// ── PRODUCTION FIXTURE (bd-i40y) ─────────────────────────────────────────────
//
// The rows that used this bench (`flat CSR REJECTED`, NEGATIVE_EVIDENCE 2026-07-09) measured only the
// bench-local `rank_old`/`rank_new` copies above: the bench imported nothing but `std` and `criterion`,
// so `GraphRanker::rank_phase1` — the function those rows are *about* — had 0.000% self-time and was
// never called. The arms below fix that by benching the shipped symbol.
//
// The `DocumentGraph` is the single source of truth: `adj` is derived FROM it, so the copy and
// production see byte-identical edge sets. That matters because `DocumentGraph::add_edge` *upserts*
// duplicate `(neighbor, edge_type)` pairs while a raw `Vec` push does not — building the two
// independently would silently give production fewer edges and flatter the comparison.

/// Build the production graph plus its seed hits, using the same LCG as `make_graph`.
fn make_prod(n: usize, deg: usize) -> (DocumentGraph, Vec<VectorHit>) {
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let n_u64 = u64::try_from(n).expect("benchmark node count fits in u64");
    let mut graph = DocumentGraph::new();
    for i in 0..n {
        for _ in 0..deg {
            let j = usize::try_from(next() % n_u64).expect("reduced node index fits in usize");
            if j != i {
                let w = 0.25 + (next() % 1000) as f32 / 1000.0;
                graph.add_edge(format!("d{i:05}"), format!("d{j:05}"), EdgeType::Similar, w);
            }
        }
    }
    // 10 seeds, mirroring `make_graph`'s personalization. `rank_phase1` normalizes internally.
    let seeds: Vec<VectorHit> = (0..10u32)
        .map(|s| VectorHit {
            index: s,
            score: 0.5 + s as f32 * 0.05,
            doc_id: format!("d{:05}", (s as usize * 37) % n).into(),
        })
        .collect();
    (graph, seeds)
}

/// Derive the copy's `Adj` from the production graph so both arms see the same edges.
fn adj_from_graph(graph: &DocumentGraph) -> Adj {
    graph
        .adjacency()
        .iter()
        .map(|(doc, edges)| {
            (
                doc.clone(),
                edges
                    .iter()
                    .map(|e| (e.neighbor_doc_id.clone(), e.weight))
                    .collect(),
            )
        })
        .collect()
}

/// Replicate `GraphRanker::personalization_from_seed_hits` (private): max score per in-graph doc,
/// then normalize to sum 1 — exactly what `make_graph` hands `rank_new`.
fn pers_from_seeds(graph: &DocumentGraph, seeds: &[VectorHit]) -> Vec<(String, f64)> {
    let mut w: HashMap<String, f64> = HashMap::new();
    for hit in seeds {
        if !graph.contains_node(&hit.doc_id) {
            continue;
        }
        let score = f64::from(hit.score);
        if !score.is_finite() || score <= 0.0 {
            continue;
        }
        let e = w.entry(hit.doc_id.to_string()).or_insert(0.0);
        if score > *e {
            *e = score;
        }
    }
    let total: f64 = w.values().sum();
    if total > 0.0 {
        for v in w.values_mut() {
            *v /= total;
        }
    }
    w.into_iter().collect()
}

fn bench_graph_rank(c: &mut Criterion) {
    let cases = [(500usize, 6usize), (2000, 8)];
    let mut g = c.benchmark_group("graph_rank");
    for (n, deg) in cases {
        let (adj, pers) = make_graph(n, deg);
        let limit = 50;
        // Was `debug_assert_eq!`, which is compiled OUT of the release bench — the parity gate
        // between the two copies never actually ran. It runs once per case; the cost is noise.
        assert_eq!(rank_old(&adj, &pers, limit), rank_new(&adj, &pers, limit));
        let id = format!("n{n}_deg{deg}");
        g.bench_with_input(BenchmarkId::new("old", &id), &(), |b, ()| {
            b.iter(|| black_box(rank_old(black_box(&adj), black_box(&pers), limit)));
        });
        g.bench_with_input(BenchmarkId::new("new", &id), &(), |b, ()| {
            b.iter(|| black_box(rank_new(black_box(&adj), black_box(&pers), limit)));
        });

        // ── The shipped symbol, vs the copy the REJECT rows actually measured ──
        let (graph, seeds) = make_prod(n, deg);
        let prod_adj = adj_from_graph(&graph);
        let prod_pers = pers_from_seeds(&graph, &seeds);
        let cx = Cx::for_testing();
        let ranker = GraphRanker::new();

        // REACHABILITY GATE. `rank_phase1` returns `None` on an empty graph or empty
        // personalization; a `None` here would mean the paired arm below times a no-op.
        let prod_out = ranker
            .rank_phase1(&cx, "graph rank query", &graph, &seeds, limit)
            .expect("rank_phase1")
            .expect("rank_phase1 must return Some -- else this bench measures nothing");
        assert!(
            !prod_out.is_empty(),
            "{id}: rank_phase1 returned no results"
        );
        eprintln!(
            "[reachability] {id}: rank_phase1 -> {} results over {} nodes",
            prod_out.len(),
            graph.adjacency().len()
        );

        // ── CURRENT LEVER (bd-vg3i): lookup-only dense index, SipHash -> aHash ─────────────
        //
        // Both arms call the same generic production implementation. `idx` is populated from
        // `adjacency.keys()` and then only probed, never iterated, so its hasher cannot affect node
        // numbering, edge visit order, floating-point accumulation, tie-breaking, or output order.
        let siphash_out = ranker
            .rank_phase1_siphash(&cx, "graph rank query", &graph, &seeds, limit)
            .expect("rank_phase1_siphash")
            .expect("rank_phase1_siphash must return Some -- else the legacy arm measures nothing");
        assert_eq!(
            siphash_out.len(),
            prod_out.len(),
            "{id}: SipHash/aHash result counts diverged"
        );
        for (sip, fast) in siphash_out.iter().zip(&prod_out) {
            assert_eq!(sip.doc_id, fast.doc_id, "{id}: aHash reordered the ranking");
            assert_eq!(
                sip.score.to_bits(),
                fast.score.to_bits(),
                "{id}: aHash changed doc {} score bits",
                sip.doc_id
            );
        }

        let run_siphash = || {
            black_box(ranker.rank_phase1_siphash(
                black_box(&cx),
                "graph rank query",
                black_box(&graph),
                black_box(&seeds),
                limit,
            ));
        };
        let run_ahash = || {
            black_box(ranker.rank_phase1(
                black_box(&cx),
                "graph rank query",
                black_box(&graph),
                black_box(&seeds),
                limit,
            ));
        };
        let hash_null = paired_median_ratio(41, PAIR_BATCH, run_siphash, run_siphash);
        let hash_lever = paired_median_ratio(41, PAIR_BATCH, run_siphash, run_ahash);
        eprintln!(
            "[null]  {id}: SipHash/SipHash median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
            hash_null.median, hash_null.p5, hash_null.p95, hash_null.rounds
        );
        eprintln!(
            "[lever] {id}: aHash/SipHash median {:.4} p5 {:.4} p95 {:.4} -> {}",
            hash_lever.median,
            hash_lever.p5,
            hash_lever.p95,
            if hash_lever.decidable_against(&hash_null) {
                "DECIDABLE"
            } else {
                "INSIDE NULL FLOOR"
            }
        );

        g.bench_with_input(BenchmarkId::new("hash_siphash", &id), &(), |b, ()| {
            b.iter(|| {
                black_box(ranker.rank_phase1_siphash(
                    black_box(&cx),
                    "graph rank query",
                    black_box(&graph),
                    black_box(&seeds),
                    limit,
                ))
            });
        });
        g.bench_with_input(BenchmarkId::new("hash_ahash", &id), &(), |b, ()| {
            b.iter(|| {
                black_box(ranker.rank_phase1(
                    black_box(&cx),
                    "graph rank query",
                    black_box(&graph),
                    black_box(&seeds),
                    limit,
                ))
            });
        });

        // FIDELITY DIAGNOSTIC (printed, not asserted): does the bench-local copy that the REJECT
        // rows used actually rank like production? Float accumulation order and HashMap node
        // indexing differ, so exact equality is not required -- but a low match count would mean
        // the copy was never a valid stand-in and those rows measured a different algorithm.
        let copy_out = rank_new(&prod_adj, &prod_pers, limit);
        let common = prod_out.len().min(copy_out.len());
        let matches = prod_out
            .iter()
            .zip(&copy_out)
            .filter(|(p, c)| p.doc_id.as_str() == c.as_str())
            .count();
        eprintln!(
            "[fidelity] {id}: top-{limit} positional agreement prod-vs-copy = {matches}/{common}"
        );

        // ── THE ACTUAL LEVER (bd-i40y): shipped Vec<Vec> CSR vs flat CSR, ONE variable ──
        //
        // `rank_phase1_flat` is a twin of `rank_phase1` inside the crate (feature `bench-internals`):
        // same private personalization, same power iteration, same `finalize_scores`, same edge-visit
        // order. Only the edge layout differs. The 2026-07-09 REJECT row compared two bench-local
        // copies instead, which cannot settle a layout question about the shipped path.
        //
        // PARITY GATE: identical edge-visit order ⇒ identical `next[dst]` accumulation order ⇒ the two
        // must agree bit-for-bit. A divergence means the arms stopped measuring one variable, and any
        // ratio below would be meaningless.
        let flat_out = ranker
            .rank_phase1_flat(&cx, "graph rank query", &graph, &seeds, limit)
            .expect("rank_phase1_flat must return Some -- else the flat arm measures nothing");
        assert_eq!(
            flat_out.len(),
            prod_out.len(),
            "{id}: flat/shipped result counts diverged"
        );
        for (p, f) in prod_out.iter().zip(&flat_out) {
            assert_eq!(p.doc_id, f.doc_id, "{id}: flat CSR reordered the ranking");
            assert_eq!(
                p.score.to_bits(),
                f.score.to_bits(),
                "{id}: flat CSR changed doc {} score bits -- arms differ by more than layout",
                p.doc_id
            );
        }
        eprintln!(
            "[reachability] {id}: rank_phase1_flat -> {} results, bit-identical to shipped",
            flat_out.len()
        );

        // ── bd-5hlw: single-pass UPPER-BOUND flat CSR vs shipped Vec<Vec> CSR, ONE variable ──
        //
        // `rank_phase1_flat_upper` is the "strongest form" the ledger left untested: it sizes CSR
        // `offsets` from the raw `edges.len()` upper bound (no counting pass ⇒ one `idx` probe/edge,
        // same as shipped, unlike `rank_phase1_flat`'s two), fills once, and reads only each row's
        // live prefix. Same edge-visit order ⇒ bit-identical to shipped (asserted). The A/B isolates
        // "n small row allocs + scattered blocks" vs "3 arena allocs + one over-allocated flat block".
        let upper_out = ranker
            .rank_phase1_flat_upper(&cx, "graph rank query", &graph, &seeds, limit)
            .expect("rank_phase1_flat_upper must return Some -- else the arm measures nothing");
        assert_eq!(
            upper_out.len(),
            prod_out.len(),
            "{id}: flat_upper/shipped result counts diverged"
        );
        for (p, f) in prod_out.iter().zip(&upper_out) {
            assert_eq!(
                p.doc_id, f.doc_id,
                "{id}: flat_upper CSR reordered the ranking"
            );
            assert_eq!(
                p.score.to_bits(),
                f.score.to_bits(),
                "{id}: flat_upper CSR changed doc {} score bits -- arms differ by more than layout",
                p.doc_id
            );
        }
        let run_shipped_csr = || {
            black_box(ranker.rank_phase1(
                black_box(&cx),
                "graph rank query",
                black_box(&graph),
                black_box(&seeds),
                limit,
            ));
        };
        let run_upper_csr = || {
            black_box(ranker.rank_phase1_flat_upper(
                black_box(&cx),
                "graph rank query",
                black_box(&graph),
                black_box(&seeds),
                limit,
            ));
        };
        let upper_null = paired_median_ratio(41, PAIR_BATCH, run_shipped_csr, run_shipped_csr);
        let upper_lever = paired_median_ratio(41, PAIR_BATCH, run_shipped_csr, run_upper_csr);
        eprintln!(
            "[csr-null]  {id}: shipped/shipped median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
            upper_null.median, upper_null.p5, upper_null.p95, upper_null.rounds
        );
        eprintln!(
            "[csr-upper] {id}: flat_upper/shipped median {:.4} p5 {:.4} p95 {:.4} -> {}",
            upper_lever.median,
            upper_lever.p5,
            upper_lever.p95,
            if upper_lever.decidable_against(&upper_null) {
                "DECIDABLE"
            } else {
                "INSIDE NULL FLOOR"
            }
        );

        // ── NULL CONTROL (A/A): the SAME arm registered twice, identical interleaved structure ──
        //
        // This measures the harness's noise floor. Any ratio closer to 1.000 than the null control's
        // deviation is indistinguishable from noise, and any REJECT of a lever whose effect is below
        // the floor is meaningless. If the null control is not tight (cv > ~5%, or ratio further than
        // a few percent from 1.000), the harness is not fit to decide the lever and must be fixed
        // (more iterations, pinned worker, quiesced tree) BEFORE any conclusion is drawn.
        g.bench_with_input(BenchmarkId::new("paired_null_a", &id), &(), |b, ()| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    for _ in 0..PAIR_BATCH {
                        black_box(ranker.rank_phase1(
                            black_box(&cx),
                            "graph rank query",
                            black_box(&graph),
                            black_box(&seeds),
                            limit,
                        ));
                    }
                    let t = Instant::now();
                    for _ in 0..PAIR_BATCH {
                        black_box(ranker.rank_phase1(
                            black_box(&cx),
                            "graph rank query",
                            black_box(&graph),
                            black_box(&seeds),
                            limit,
                        ));
                    }
                    total += t.elapsed();
                }
                total
            });
        });
        g.bench_with_input(BenchmarkId::new("paired_null_b", &id), &(), |b, ()| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    for _ in 0..PAIR_BATCH {
                        black_box(ranker.rank_phase1(
                            black_box(&cx),
                            "graph rank query",
                            black_box(&graph),
                            black_box(&seeds),
                            limit,
                        ));
                    }
                    let t = Instant::now();
                    for _ in 0..PAIR_BATCH {
                        black_box(ranker.rank_phase1(
                            black_box(&cx),
                            "graph rank query",
                            black_box(&graph),
                            black_box(&seeds),
                            limit,
                        ));
                    }
                    total += t.elapsed();
                }
                total
            });
        });

        g.bench_with_input(BenchmarkId::new("paired_shipped_csr", &id), &(), |b, ()| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    for _ in 0..PAIR_BATCH {
                        black_box(ranker.rank_phase1_flat(
                            black_box(&cx),
                            "graph rank query",
                            black_box(&graph),
                            black_box(&seeds),
                            limit,
                        ));
                    }
                    let t = Instant::now();
                    for _ in 0..PAIR_BATCH {
                        black_box(ranker.rank_phase1(
                            black_box(&cx),
                            "graph rank query",
                            black_box(&graph),
                            black_box(&seeds),
                            limit,
                        ));
                    }
                    total += t.elapsed();
                }
                total
            });
        });
        g.bench_with_input(BenchmarkId::new("paired_flat_csr", &id), &(), |b, ()| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    for _ in 0..PAIR_BATCH {
                        black_box(ranker.rank_phase1(
                            black_box(&cx),
                            "graph rank query",
                            black_box(&graph),
                            black_box(&seeds),
                            limit,
                        ));
                    }
                    let t = Instant::now();
                    for _ in 0..PAIR_BATCH {
                        black_box(ranker.rank_phase1_flat(
                            black_box(&cx),
                            "graph rank query",
                            black_box(&graph),
                            black_box(&seeds),
                            limit,
                        ));
                    }
                    total += t.elapsed();
                }
                total
            });
        });

        // INTERLEAVED PAIRED SAMPLER: each arm runs BOTH implementations per iteration and times
        // only its own, so worker/thermal drift cancels in the ratio (criterion group members run
        // sequentially, so a plain side-by-side registration would not).
        g.bench_with_input(BenchmarkId::new("paired_prod", &id), &(), |b, ()| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    for _ in 0..PAIR_BATCH {
                        black_box(rank_new(black_box(&prod_adj), black_box(&prod_pers), limit));
                    }
                    let t = Instant::now();
                    for _ in 0..PAIR_BATCH {
                        black_box(ranker.rank_phase1(
                            black_box(&cx),
                            "graph rank query",
                            black_box(&graph),
                            black_box(&seeds),
                            limit,
                        ));
                    }
                    total += t.elapsed();
                }
                total
            });
        });
        g.bench_with_input(BenchmarkId::new("paired_copy", &id), &(), |b, ()| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    for _ in 0..PAIR_BATCH {
                        black_box(ranker.rank_phase1(
                            black_box(&cx),
                            "graph rank query",
                            black_box(&graph),
                            black_box(&seeds),
                            limit,
                        ));
                    }
                    let t = Instant::now();
                    for _ in 0..PAIR_BATCH {
                        black_box(rank_new(black_box(&prod_adj), black_box(&prod_pers), limit));
                    }
                    total += t.elapsed();
                }
                total
            });
        });
    }
    g.finish();
}

criterion_group!(benches, bench_graph_rank);
criterion_main!(benches);
