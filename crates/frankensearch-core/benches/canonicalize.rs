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
    "ok",
    "done",
    "yes",
    "no",
    "thanks",
    "understood",
    "sure",
    "got it",
    "thank you",
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

/// Old `strip_markdown_line` inline chain: 4 allocating no-op passes per line.
fn md_old(line: &str) -> String {
    line.replace("**", "")
        .replace("__", "")
        .replace('*', "")
        .replace('`', "")
}

/// New: skip the chain entirely when the line has no inline-markdown chars.
fn md_new(line: &str) -> String {
    if line.bytes().any(|b| matches!(b, b'*' | b'_' | b'`' | b'[')) {
        line.replace("**", "")
            .replace("__", "")
            .replace('*', "")
            .replace('`', "")
    } else {
        line.to_owned()
    }
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

    // strip_markdown_line inline chain: old (always replace) vs new (skip if clean).
    let plain_doc = "fn retry(n: usize) -> bool { backoff(n) }\nThe quick brown fox jumps.\n\
                     let x = compute_value(a, b) + 1;\nplain prose with no markup here.\n"
        .repeat(20);
    let mut mg = c.benchmark_group("strip_markdown_inline");
    mg.bench_function("old", |b| {
        b.iter(|| {
            let mut acc = 0usize;
            for line in plain_doc.lines() {
                acc += md_old(black_box(line)).len();
            }
            black_box(acc)
        });
    });
    mg.bench_function("new", |b| {
        b.iter(|| {
            let mut acc = 0usize;
            for line in plain_doc.lines() {
                acc += md_new(black_box(line)).len();
            }
            black_box(acc)
        });
    });
    mg.finish();

    // Header/blockquote prefix trim: old (two to_string allocs) vs new (one).
    fn trim_old(line: &str) -> String {
        let r = line.trim_start_matches('#').trim_start().to_string();
        r.trim_start_matches('>').trim_start().to_string()
    }
    fn trim_new(line: &str) -> String {
        line.trim_start_matches('#')
            .trim_start()
            .trim_start_matches('>')
            .trim_start()
            .to_string()
    }
    let mut tg = c.benchmark_group("prefix_trim");
    tg.bench_function("old", |b| {
        b.iter(|| {
            let mut acc = 0usize;
            for line in plain_doc.lines() {
                acc += trim_old(black_box(line)).len();
            }
            black_box(acc)
        });
    });
    tg.bench_function("new", |b| {
        b.iter(|| {
            let mut acc = 0usize;
            for line in plain_doc.lines() {
                acc += trim_new(black_box(line)).len();
            }
            black_box(acc)
        });
    });
    tg.finish();

    // Pipeline tail: old `filter_low_signal` copied the whole doc, then truncate
    // copied again (2 copies); the bool predicate + pass-through truncates once.
    fn truncate(s: &str) -> String {
        if s.len() <= 2000 {
            s.to_owned()
        } else {
            s[..2000].to_owned()
        }
    }
    fn tail_old(ws: &str) -> String {
        let filtered = ws.to_string(); // old filter_low_signal whole-doc copy
        truncate(&filtered)
    }
    fn tail_new(ws: &str) -> String {
        truncate(ws) // pass the owned buffer straight through
    }
    let mut pg = c.benchmark_group("pipeline_tail");
    pg.bench_with_input("old", ascii_doc.as_str(), |b, t| {
        b.iter(|| black_box(tail_old(black_box(t))));
    });
    pg.bench_with_input("new", ascii_doc.as_str(), |b, t| {
        b.iter(|| black_box(tail_new(black_box(t))));
    });
    pg.finish();
}

criterion_group!(benches, bench_nfc);
criterion_main!(benches);
