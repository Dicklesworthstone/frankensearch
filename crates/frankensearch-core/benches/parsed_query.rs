//! Negation-query parse benchmark.
//!
//! `ParsedQuery::parse` runs per search query (the searcher parses for `-term` /
//! `NOT "phrase"` negations). The committed parser always materialized a
//! `Vec<char>` and re-collected each word with `chars[a..b].iter().collect()`.
//! Most queries contain no negation-syntax chars (`-`, `"`, `\`), so the new fast
//! path returns the whitespace-normalized input directly (split + push_str),
//! skipping the char machinery. This bench is the head-to-head on a plain query
//! (`old` = char-based parse, `new` = fast path).
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
//!   rch exec -- cargo bench -p frankensearch-core --bench parsed_query
//! ```

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

/// Prior char-based positive extraction (the plain-query path of the full parser).
fn parse_old(raw: &str) -> String {
    let chars: Vec<char> = raw.chars().collect();
    let len = chars.len();
    let mut i = 0;
    let mut parts: Vec<String> = Vec::new();
    while i < len {
        if chars[i].is_whitespace() {
            i += 1;
            continue;
        }
        let start = i;
        while i < len && !chars[i].is_whitespace() {
            i += 1;
        }
        parts.push(chars[start..i].iter().collect());
    }
    parts.join(" ")
}

/// New fast path: whitespace-normalize the input directly, no `Vec<char>`.
fn parse_new(raw: &str) -> String {
    let mut positive = String::with_capacity(raw.len());
    for word in raw.split_whitespace() {
        if !positive.is_empty() {
            positive.push(' ');
        }
        positive.push_str(word);
    }
    positive
}

fn bench_parsed_query(c: &mut Criterion) {
    // A plain multi-word query (the common case — no negation syntax).
    let query = "how does the hybrid search ranking actually work in practice here";

    debug_assert_eq!(parse_old(query), parse_new(query));

    let mut g = c.benchmark_group("parsed_query");
    g.bench_with_input("old", query, |b, q| {
        b.iter(|| black_box(parse_old(black_box(q))));
    });
    g.bench_with_input("new", query, |b, q| {
        b.iter(|| black_box(parse_new(black_box(q))));
    });
    g.finish();
}

criterion_group!(benches, bench_parsed_query);
criterion_main!(benches);
