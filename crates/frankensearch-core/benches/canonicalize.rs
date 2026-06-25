//! NFC canonicalization fast-path benchmark.
//!
//! `DefaultCanonicalizer` runs NFC normalization on every document at index time
//! and every query. ASCII text is always already in NFC, so the shipped fast path
//! (`is_ascii()` → copy) skips the unicode-normalization state machine. This bench
//! isolates that step (the crate's `nfc_normalize` is private, so the fast/full
//! variants are replicated here) head-to-head on the same input.
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
//!   rch exec -- cargo bench -p frankensearch-core --bench canonicalize
//! ```

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use unicode_normalization::UnicodeNormalization;

/// Shipped fast path (mirrors `frankensearch_core::canonicalize::nfc_normalize`).
fn nfc_fast(text: &str) -> String {
    if text.is_ascii() {
        text.to_owned()
    } else {
        text.nfc().collect()
    }
}

/// Prior behavior: always run the full NFC iterator.
fn nfc_full(text: &str) -> String {
    text.nfc().collect()
}

const LOW_SIGNAL: &[&str] = &[
    "ok", "done", "yes", "no", "thanks", "understood", "sure", "got it", "thank you",
];

/// Prior `filter_low_signal`: lowercase the entire document, then compare.
fn filter_old(text: &str) -> bool {
    let lower = text.trim().to_lowercase();
    LOW_SIGNAL.iter().any(|p| lower == *p)
}

/// New: case-insensitive ASCII compare — short-circuits on length, no allocation.
fn filter_new(text: &str) -> bool {
    let t = text.trim();
    LOW_SIGNAL.iter().any(|p| t.eq_ignore_ascii_case(p))
}

fn bench_nfc(c: &mut Criterion) {
    let ascii_short = "fn main() { let x = retry_backoff(3); }".to_owned();
    let ascii_doc = "The quick brown fox jumps over the lazy dog. ".repeat(50);
    let non_ascii = "café façade naïve 日本語 ".repeat(50);

    let mut group = c.benchmark_group("nfc");
    for (name, text) in [
        ("ascii_short", &ascii_short),
        ("ascii_doc", &ascii_doc),
        ("non_ascii", &non_ascii),
    ] {
        group.bench_with_input(BenchmarkId::new("fast", name), text.as_str(), |b, t| {
            b.iter(|| black_box(nfc_fast(black_box(t))));
        });
        group.bench_with_input(BenchmarkId::new("full", name), text.as_str(), |b, t| {
            b.iter(|| black_box(nfc_full(black_box(t))));
        });
    }
    group.finish();

    // filter_low_signal: old (full-doc to_lowercase) vs new (ASCII compare).
    let mut fg = c.benchmark_group("filter_low_signal");
    for (name, text) in [("ascii_doc", &ascii_doc), ("ascii_short", &ascii_short)] {
        fg.bench_with_input(BenchmarkId::new("old", name), text.as_str(), |b, t| {
            b.iter(|| black_box(filter_old(black_box(t))));
        });
        fg.bench_with_input(BenchmarkId::new("new", name), text.as_str(), |b, t| {
            b.iter(|| black_box(filter_new(black_box(t))));
        });
    }
    fg.finish();
}

criterion_group!(benches, bench_nfc);
criterion_main!(benches);
