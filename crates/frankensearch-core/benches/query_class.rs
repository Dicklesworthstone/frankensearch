//! Query classification benchmark.
//!
//! `QueryClass::classify` runs on every search query (the adaptive lexical/semantic
//! budget + the lexical short-circuit gate both call it). The committed
//! `looks_like_identifier` rescanned the query for whitespace up to four times and
//! allocated a `Vec` via `rsplitn(2,'-').collect()`. The new version computes
//! `has_whitespace` once, groups the no-whitespace checks under it, and uses
//! allocation-free `rsplit_once`. This bench is the head-to-head (the real
//! `classify` is `new`; the prior `looks_like_identifier` is replicated as `old`).
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
//!   rch exec -- cargo bench -p frankensearch-core --bench query_class
//! ```

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use frankensearch_core::query_class::QueryClass;

/// Prior `looks_like_identifier`: rescans whitespace 4× and allocates via rsplitn.
fn looks_like_identifier_old(s: &str) -> bool {
    if !s.chars().any(char::is_whitespace) && (s.contains('/') || s.contains('\\')) {
        return true;
    }
    if !s.chars().any(char::is_whitespace) && (s.contains('.') || s.contains("::")) {
        return true;
    }
    if !s.chars().any(char::is_whitespace) {
        if s.contains('_') {
            return true;
        }
        let has_lower = s.chars().any(char::is_lowercase);
        let has_upper = s.chars().any(char::is_uppercase);
        let first_upper = s.chars().next().is_some_and(char::is_uppercase);
        let rest_lower = s.chars().skip(1).all(char::is_lowercase);
        if has_lower && has_upper && !(first_upper && rest_lower) {
            return true;
        }
    }
    if !s.chars().any(char::is_whitespace) && s.contains('-') {
        let parts: Vec<&str> = s.rsplitn(2, '-').collect();
        if parts.len() == 2
            && parts[0].chars().all(|c| c.is_ascii_digit())
            && parts[1]
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
            && !parts[0].is_empty()
            && !parts[1].is_empty()
        {
            return true;
        }
    }
    s.starts_with("fn ") || s.starts_with("struct ") || s.starts_with("impl ")
}

/// Prior `classify`, using the old identifier check.
fn classify_old(query: &str) -> u8 {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return 0;
    }
    if looks_like_identifier_old(trimmed) {
        return 1;
    }
    if trimmed.split_whitespace().count() <= 3 {
        2
    } else {
        3
    }
}

/// The committed grouped `looks_like_identifier` before the ASCII-byte fast path:
/// one whitespace check, one Unicode case-flag pass, allocation-free issue-id
/// split, and capped word count.
fn looks_like_identifier_current(s: &str) -> bool {
    if !s.chars().any(char::is_whitespace) {
        if s.contains('/') || s.contains('\\') || s.contains('.') || s.contains("::") {
            return true;
        }
        if s.contains('_') {
            return true;
        }
        let mut has_lower = false;
        let mut has_upper = false;
        let mut first_upper = false;
        let mut rest_lower = true;
        for (i, c) in s.chars().enumerate() {
            let is_lower = c.is_lowercase();
            let is_upper = c.is_uppercase();
            has_lower |= is_lower;
            has_upper |= is_upper;
            if i == 0 {
                first_upper = is_upper;
            } else if !is_lower {
                rest_lower = false;
            }
        }
        if has_lower && has_upper && !(first_upper && rest_lower) {
            return true;
        }
        if let Some((prefix, suffix)) = s.rsplit_once('-')
            && !prefix.is_empty()
            && !suffix.is_empty()
            && suffix.chars().all(|c| c.is_ascii_digit())
            && prefix
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            return true;
        }
    }
    s.starts_with("fn ") || s.starts_with("struct ") || s.starts_with("impl ")
}

fn classify_current(query: &str) -> u8 {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return 0;
    }
    if looks_like_identifier_current(trimmed) {
        return 1;
    }
    if trimmed.split_whitespace().take(4).count() <= 3 {
        2
    } else {
        3
    }
}

fn bench_query_class(c: &mut Criterion) {
    // Representative query mix: identifiers (no whitespace — the worst case for the
    // prior redundant scans), short keywords, and natural language.
    let queries = [
        "src/main.rs",
        "bd-12345",
        "myFunctionName",
        "snake_case_variable_name",
        "SomeStructName",
        "crate::module::function",
        "error handling",
        "retry backoff",
        "how does the hybrid search ranking actually work in practice",
        "fn main",
        "what is the difference between lexical and semantic retrieval here",
    ];

    let mut g = c.benchmark_group("query_class");
    g.bench_function("old", |b| {
        b.iter(|| {
            let mut acc = 0u32;
            for q in &queries {
                acc += u32::from(classify_old(black_box(q)));
            }
            black_box(acc)
        });
    });
    // A/B baseline: the committed grouped impl (pre single-pass / take(4) change).
    g.bench_function("current", |b| {
        b.iter(|| {
            let mut acc = 0u32;
            for q in &queries {
                acc += u32::from(classify_current(black_box(q)));
            }
            black_box(acc)
        });
    });
    g.bench_function("new", |b| {
        b.iter(|| {
            let mut acc = 0u32;
            for q in &queries {
                acc += QueryClass::classify(black_box(q)) as u32;
            }
            black_box(acc)
        });
    });
    g.finish();
}

criterion_group!(benches, bench_query_class);
criterion_main!(benches);
