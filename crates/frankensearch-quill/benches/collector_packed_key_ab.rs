#![recursion_limit = "256"]
//! Top-k collector selection-key A/B: the shipping `HeapEntry` (f32
//! `total_cmp` + docid tiebreak) vs a packed-u64 selection key that reduces
//! each heap comparison to one native integer compare (`bd-y1ab`).
//!
//! Sibling of the KEPT FSVI int8 pass-1 packed-selection-key win (bd-b5wl,
//! `PERF_LEDGER` 2026-07-10). The quill `TopDocsCollector` calls `record_live`
//! per matched live doc — the query hot path — and each push/replace does a
//! float `total_cmp` then a conditional docid tiebreak. The packed arm folds
//! score and docid into one `u64` so the whole comparison is a single integer
//! compare: high 32 bits = `!sortable(score)` (worst score = greatest key =
//! max-heap top), low 32 bits = docid (higher docid = greater = popped first
//! on a tie, so the lower docid is retained — matching the shipping order).
//!
//! `sortable` is the standard monotonic f32->u32 bijection, exact for all
//! finite f32 (scores are `finite_score`-guaranteed finite; boosts may be
//! negative, so the transform handles the sign bit). Both arms produce the
//! identical sorted page (docid + raw score bits) and, in count mode, the
//! identical total — asserted before timing. Ratio is packed/current, `<1`
//! wins.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR \
//!   rch exec -- env QUILL_COLLECTOR_AB_INNER=64 cargo bench \
//!   -p frankensearch-quill --features bench-internals --profile release \
//!   --bench collector_packed_key_ab
//! ```

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::hint::black_box;

use frankensearch_core::bench_support::paired_median_ratio;

// ─── Shipping strategy: HeapEntry with float total_cmp + docid tiebreak ──────

#[derive(Clone, Copy)]
struct HeapEntry {
    global_docid: u32,
    score: f32,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.global_docid == other.global_docid && self.score.to_bits() == other.score.to_bits()
    }
}
impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.score.total_cmp(&other.score) {
            Ordering::Less => Ordering::Greater,
            Ordering::Greater => Ordering::Less,
            Ordering::Equal => self.global_docid.cmp(&other.global_docid),
        }
    }
}

fn collect_current(stream: &[(u32, f32)], retained: usize) -> Vec<(u32, u32)> {
    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(retained);
    for &(global_docid, score) in stream {
        let entry = HeapEntry {
            global_docid,
            score,
        };
        if heap.len() < retained {
            heap.push(entry);
        } else if heap.peek().is_some_and(|cutoff| entry < *cutoff) {
            let _ = heap.pop();
            heap.push(entry);
        }
    }
    let mut hits: Vec<HeapEntry> = heap.into_vec();
    hits.sort_unstable_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.global_docid.cmp(&right.global_docid))
    });
    hits.into_iter()
        .map(|entry| (entry.global_docid, entry.score.to_bits()))
        .collect()
}

// ─── Packed strategy: one u64 selection key, native integer compare ──────────

/// Monotonic f32 -> u32 bijection: ascending u32 order == ascending f32 order
/// for all finite values (and signed zeros).
#[inline]
fn sortable(score: f32) -> u32 {
    let bits = score.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

/// Inverse of [`sortable`].
#[inline]
fn from_sortable(s: u32) -> f32 {
    if s & 0x8000_0000 != 0 {
        f32::from_bits(s & 0x7fff_ffff)
    } else {
        f32::from_bits(!s)
    }
}

/// Badness key: greatest for the worst entry (lowest score; highest docid on a
/// tie), so a max-heap keeps the best `retained` entries.
#[inline]
fn packed_key(global_docid: u32, score: f32) -> u64 {
    (u64::from(!sortable(score)) << 32) | u64::from(global_docid)
}

fn collect_packed(stream: &[(u32, f32)], retained: usize) -> Vec<(u32, u32)> {
    let mut heap: BinaryHeap<u64> = BinaryHeap::with_capacity(retained);
    for &(global_docid, score) in stream {
        let key = packed_key(global_docid, score);
        if heap.len() < retained {
            heap.push(key);
        } else if heap.peek().is_some_and(|&cutoff| key < cutoff) {
            let _ = heap.pop();
            heap.push(key);
        }
    }
    let mut keys: Vec<u64> = heap.into_vec();
    // Best-first == smallest badness key first.
    keys.sort_unstable();
    keys.into_iter()
        .map(|key| {
            let global_docid = (key & 0xffff_ffff) as u32;
            let score_bits = from_sortable(!((key >> 32) as u32)).to_bits();
            (global_docid, score_bits)
        })
        .collect()
}

// ─── Fixtures ────────────────────────────────────────────────────────────────

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// A realistic scored posting stream: `n` docs, scores drawn from `buckets`
/// distinct values so ties are common (BM25 blocks share fieldnorm ids), with
/// a signed-boost fraction so the transform's negative path is exercised.
fn make_stream(n: usize, buckets: u32, signed: bool, seed: u64) -> Vec<(u32, f32)> {
    let mut state = seed | 1;
    (0..n)
        .map(|index| {
            let bucket = (xorshift(&mut state) % u64::from(buckets)) as f32;
            #[allow(clippy::cast_possible_truncation)]
            let mut score = 0.125_f32.mul_add(bucket, 0.5);
            if signed && xorshift(&mut state) % 5 == 0 {
                score = -score;
            }
            (u32::try_from(index).unwrap_or(u32::MAX), score)
        })
        .collect()
}

fn configured_inner() -> u32 {
    std::env::var("QUILL_COLLECTOR_AB_INNER")
        .ok()
        .and_then(|raw| raw.parse().ok())
        .filter(|&value| value > 0)
        .unwrap_or(64)
}

fn main() {
    let inner = configured_inner();
    // (label, doc count, score buckets, signed boosts, retained window)
    let shapes: &[(&str, usize, u32, bool, usize)] = &[
        ("high_df_k10", 50_000, 12, false, 10),
        ("high_df_k100", 50_000, 12, false, 100),
        ("tie_heavy_k10", 50_000, 4, false, 10),
        ("signed_k10", 50_000, 12, true, 10),
        ("wide_k1000", 100_000, 64, false, 1_000),
    ];

    for &(label, n, buckets, signed, retained) in shapes {
        let stream = make_stream(
            n,
            buckets,
            signed,
            0x01ab_babe_0000_0001_u64.wrapping_add(n as u64),
        );
        let current = collect_current(&stream, retained);
        let packed = collect_packed(&stream, retained);
        assert_eq!(
            current, packed,
            "packed selection key diverged from HeapEntry order ({label})",
        );

        let null = paired_median_ratio(
            41,
            inner,
            || {
                black_box(collect_current(black_box(&stream), retained));
            },
            || {
                black_box(collect_current(black_box(&stream), retained));
            },
        );
        let lever = paired_median_ratio(
            41,
            inner,
            || {
                black_box(collect_current(black_box(&stream), retained));
            },
            || {
                black_box(collect_packed(black_box(&stream), retained));
            },
        );
        eprintln!(
            "[cell] shape={label} n={n} retained={retained} \
             null={:.4} [{:.4}, {:.4}] lever(packed/current)={:.4} [{:.4}, {:.4}] decidable={}",
            null.median,
            null.p5,
            null.p95,
            lever.median,
            lever.p5,
            lever.p95,
            lever.decidable_against(&null),
        );
    }
}
