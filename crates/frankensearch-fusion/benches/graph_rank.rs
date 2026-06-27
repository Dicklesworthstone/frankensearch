//! Graph-rank (query-biased PageRank) power-iteration benchmark.
//!
//! `GraphRanker::rank_phase1` ran the power iteration over a
//! `HashMap<String, f64>` rebuilt every iteration, cloning a `String` doc_id key
//! on every teleport/edge relaxation (all keys already present, so the clones
//! were dead) and re-checking each edge's weight finiteness every pass. The new
//! path dense-indexes the graph once and iterates over reused `Vec<f64>` buffers
//! (CSR edges, hoisted weight filter). This bench replicates old vs new
//! self-contained (the engine internals are private) over a realistic graph.
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
//!   rch exec -- cargo bench -p frankensearch-fusion --bench graph_rank
//! ```

use std::collections::HashMap;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

const RESTART: f64 = 0.15;
const WALK: f64 = 1.0 - RESTART;
const MAX_ITER: usize = 20;
const TOL: f64 = 1e-6;

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
        next.iter_mut().for_each(|v| *v = 0.0);
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
    let mut adj: Adj = HashMap::new();
    for i in 0..n {
        adj.entry(format!("d{i:05}")).or_default();
    }
    for i in 0..n {
        for _ in 0..deg {
            let j = (next() as usize) % n;
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

fn bench_graph_rank(c: &mut Criterion) {
    let cases = [(500usize, 6usize), (2000, 8)];
    let mut g = c.benchmark_group("graph_rank");
    for (n, deg) in cases {
        let (adj, pers) = make_graph(n, deg);
        let limit = 50;
        debug_assert_eq!(rank_old(&adj, &pers, limit), rank_new(&adj, &pers, limit));
        let id = format!("n{n}_deg{deg}");
        g.bench_with_input(BenchmarkId::new("old", &id), &(), |b, ()| {
            b.iter(|| black_box(rank_old(black_box(&adj), black_box(&pers), limit)));
        });
        g.bench_with_input(BenchmarkId::new("new", &id), &(), |b, ()| {
            b.iter(|| black_box(rank_new(black_box(&adj), black_box(&pers), limit)));
        });
    }
    g.finish();
}

criterion_group!(benches, bench_graph_rank);
criterion_main!(benches);
