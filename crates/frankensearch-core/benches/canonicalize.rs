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
}

criterion_group!(benches, bench_nfc);
criterion_main!(benches);
