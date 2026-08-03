//! Sealed-segment query fan-out A/B: serial per-segment scoring vs rayon
//! fan-out with per-segment partial collectors (`bd-quill-e4-argus-3ycz.9`).
//!
//! Both arms run the identical shipping pipeline — parse, lower, score,
//! collect, merge — over the same published snapshot; the only variable is
//! whether sealed segments are scored on one thread into one collector or
//! fanned across rayon into per-segment partials folded under the collector's
//! total order (score `total_cmp` desc, docid asc). That order makes the
//! retained set unique, so every cell asserts bit-exact page parity
//! (docid + raw score bits) between the arms before timing.
//!
//! The A/A null is serial-vs-serial; the lever ratio is `fanned / serial`, so
//! `< 1.0` outside the null band wins. The `below_gate` cell forces fan-out
//! on a corpus the production gate keeps serial (2 segments, 1k docs): it
//! quantifies the per-task overhead the `SEGMENT_FANOUT_THRESHOLD` gate
//! exists to avoid and is informational, not a shipping configuration.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR \
//!   rch exec -- env QUILL_E49_SCALE=full QUILL_E49_ROUNDS=41 \
//!     cargo bench -p frankensearch-quill --features bench-internals \
//!       --profile release --bench segment_fanout_ab
//!
//! # Fast parity/harness check (not performance evidence):
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR \
//!   rch exec -- env QUILL_E49_SCALE=smoke QUILL_E49_ROUNDS=5 \
//!     cargo bench -p frankensearch-quill --features bench-internals \
//!       --profile release --bench segment_fanout_ab
//! ```

use std::hint::black_box;
use std::time::Instant;

use asupersync::Cx;
use frankensearch_core::IndexableDocument;
use frankensearch_core::bench_support::paired_median_ratio;
use frankensearch_quill::{QuillConfig, QuillIndex};

/// One benchmark corpus shape: sealed segments times documents per segment.
struct Shape {
    name: &'static str,
    segments: usize,
    docs_per_segment: usize,
}

static FULL_SHAPES: [Shape; 3] = [
    Shape {
        name: "gate8x2500",
        segments: 8,
        docs_per_segment: 2_500,
    },
    Shape {
        name: "gate4x12500",
        segments: 4,
        docs_per_segment: 12_500,
    },
    Shape {
        name: "below_gate2x500",
        segments: 2,
        docs_per_segment: 500,
    },
];

static SMOKE_SHAPES: [Shape; 2] = [
    Shape {
        name: "smoke4x250",
        segments: 4,
        docs_per_segment: 250,
    },
    Shape {
        name: "below_gate2x100",
        segments: 2,
        docs_per_segment: 100,
    },
];

/// Query classes over the synthetic vocabulary: a saturating high-df term, a
/// mid-df term, a prunable multi-term union (MaxScore/BMW territory), and an
/// exact phrase.
const QUERIES: [(&str, &str); 4] = [
    ("high_df_term", "shared"),
    ("mid_df_term", "gamma"),
    ("union3", "alpha OR rare OR argus"),
    ("phrase", "\"shared shared\""),
];

const VOCABULARY: [&str; 24] = [
    "shared", "shared", "shared", "shared", "alpha", "alpha", "beta", "beta", "gamma", "gamma",
    "delta", "epsilon", "zeta", "quill", "argus", "keeper", "scribe", "grimoire", "quiver",
    "segment", "posting", "cursor", "rare", "singular",
];

// The production bulk planner starts its fixed-width route at 64 documents
// per leaf. Keep each fixture-builder submission below that threshold, then
// make the declared `Shape` commit boundary explicit. This is setup only: the
// measured serial and fanned reader paths consume the same sealed snapshot.
const SCALAR_FIXTURE_INGEST_CHUNK_DOCS: usize = 63;
/// Larger than the largest declared shape so an explicit fixture commit stays
/// one observable leaf rather than becoming an unrequested tier merge.
const FIXTURE_TIER_FANOUT: usize = 9;

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

async fn build_index(cx: &Cx, shape: &Shape, seed: u64) -> QuillIndex {
    let config = QuillConfig {
        deterministic_ingest: true,
        // The benchmark's shape names are an exact sealed-segment contract.
        // Building a shape must therefore not let the production visibility
        // timer insert an extra publication partway through one declared
        // batch: that would silently change the fan-out denominator before
        // the serial/fanned parity check. Explicit `commit` calls below are
        // the only fixture publication boundaries.
        max_visibility_lag_ms: u64::MAX,
        tier_fanout: FIXTURE_TIER_FANOUT,
        ..QuillConfig::default()
    };
    let index = QuillIndex::in_memory(config).expect("in-memory bench index");
    let mut state = seed | 1;
    for segment in 0..shape.segments {
        let mut batch = Vec::with_capacity(shape.docs_per_segment);
        for ordinal in 0..shape.docs_per_segment {
            let word_count = 6 + usize::try_from(xorshift(&mut state) % 30).expect("word count");
            let mut text = String::with_capacity(word_count * 8);
            for position in 0..word_count {
                if position != 0 {
                    text.push(' ');
                }
                let pick = usize::try_from(xorshift(&mut state) % 24).expect("vocabulary index");
                text.push_str(VOCABULARY[pick]);
            }
            batch.push(IndexableDocument::new(
                format!("e49-s{segment:03}-d{ordinal:05}"),
                text,
            ));
        }
        for scalar_chunk in batch.chunks(SCALAR_FIXTURE_INGEST_CHUNK_DOCS) {
            index
                .index_documents(cx, scalar_chunk)
                .await
                .expect("accumulate bench scalar fixture chunk");
        }
        index.commit(cx).await.expect("seal bench segment");
    }
    let segment_doc_counts: Vec<u32> = index
        .snapshot()
        .segments()
        .iter()
        .map(|segment| segment.doc_count())
        .collect();
    eprintln!(
        "[setup-topology] shape={} expected_segments={} actual_segments={} docs_per_segment={} actual_doc_counts={segment_doc_counts:?}",
        shape.name,
        shape.segments,
        segment_doc_counts.len(),
        shape.docs_per_segment,
    );
    assert_eq!(
        segment_doc_counts.len(),
        shape.segments,
        "each commit must seal exactly one segment"
    );
    index
}

/// Median microseconds per call over `calls` sequential invocations.
fn absolute_us(calls: u32, mut run: impl FnMut()) -> f64 {
    let started = Instant::now();
    for _ in 0..calls {
        run();
    }
    started.elapsed().as_secs_f64() * 1_000_000.0 / f64::from(calls)
}

fn run_shape(cx: &Cx, shape: &Shape, rounds: usize, limit: usize, index: &QuillIndex) {
    for (query_name, query) in QUERIES {
        let serial_page = index
            .bench_search_sealed_forced(cx, query, limit, false, None)
            .expect("serial sealed page");
        let fanned_page = index
            .bench_search_sealed_forced(cx, query, limit, true, None)
            .expect("fanned sealed page");
        assert_eq!(
            serial_page, fanned_page,
            "{}: fan-out page diverged from serial before timing ({query_name})",
            shape.name,
        );

        let null = paired_median_ratio(
            rounds,
            4,
            || {
                black_box(
                    index
                        .bench_search_sealed_forced(cx, black_box(query), limit, false, None)
                        .expect("null arm a"),
                );
            },
            || {
                black_box(
                    index
                        .bench_search_sealed_forced(cx, black_box(query), limit, false, None)
                        .expect("null arm b"),
                );
            },
        );
        let lever = paired_median_ratio(
            rounds,
            4,
            || {
                black_box(
                    index
                        .bench_search_sealed_forced(cx, black_box(query), limit, false, None)
                        .expect("serial arm"),
                );
            },
            || {
                black_box(
                    index
                        .bench_search_sealed_forced(cx, black_box(query), limit, true, None)
                        .expect("fanned arm"),
                );
            },
        );
        let serial_us = absolute_us(16, || {
            black_box(
                index
                    .bench_search_sealed_forced(cx, black_box(query), limit, false, None)
                    .expect("serial absolute"),
            );
        });
        let fanned_us = absolute_us(16, || {
            black_box(
                index
                    .bench_search_sealed_forced(cx, black_box(query), limit, true, None)
                    .expect("fanned absolute"),
            );
        });
        eprintln!(
            "[cell] shape={} query={query_name} limit={limit} \
             null={:.4} [{:.4}, {:.4}] lever(fanned/serial)={:.4} [{:.4}, {:.4}] \
             decidable={} serial_us={serial_us:.1} fanned_us={fanned_us:.1}",
            shape.name,
            null.median,
            null.p5,
            null.p95,
            lever.median,
            lever.p5,
            lever.p95,
            lever.decidable_against(&null),
        );
    }

    // Unscored id-set lane (bd-quill-e4-argus-3ycz.9 sibling): the exhaustive
    // docid path behind `collect_docids` — the gauntlet differential
    // campaigns' per-case parity workhorse and the CASS structured-filter
    // path. Same fan-out gate, trivially deterministic (finish sorts+dedups).
    for (query_name, query) in QUERIES {
        let serial_ids = index
            .bench_collect_docids_forced(cx, query, false)
            .expect("serial docid set");
        let fanned_ids = index
            .bench_collect_docids_forced(cx, query, true)
            .expect("fanned docid set");
        assert_eq!(
            serial_ids, fanned_ids,
            "{}: docid fan-out diverged from serial before timing ({query_name})",
            shape.name,
        );

        let null = paired_median_ratio(
            rounds,
            4,
            || {
                black_box(
                    index
                        .bench_collect_docids_forced(cx, black_box(query), false)
                        .expect("docid null arm a"),
                );
            },
            || {
                black_box(
                    index
                        .bench_collect_docids_forced(cx, black_box(query), false)
                        .expect("docid null arm b"),
                );
            },
        );
        let lever = paired_median_ratio(
            rounds,
            4,
            || {
                black_box(
                    index
                        .bench_collect_docids_forced(cx, black_box(query), false)
                        .expect("docid serial arm"),
                );
            },
            || {
                black_box(
                    index
                        .bench_collect_docids_forced(cx, black_box(query), true)
                        .expect("docid fanned arm"),
                );
            },
        );
        let serial_us = absolute_us(16, || {
            black_box(
                index
                    .bench_collect_docids_forced(cx, black_box(query), false)
                    .expect("docid serial absolute"),
            );
        });
        let fanned_us = absolute_us(16, || {
            black_box(
                index
                    .bench_collect_docids_forced(cx, black_box(query), true)
                    .expect("docid fanned absolute"),
            );
        });
        eprintln!(
            "[docid-cell] shape={} query={query_name} \
             null={:.4} [{:.4}, {:.4}] lever(fanned/serial)={:.4} [{:.4}, {:.4}] \
             decidable={} serial_us={serial_us:.1} fanned_us={fanned_us:.1}",
            shape.name,
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

fn main() {
    let scale = std::env::var("QUILL_E49_SCALE").unwrap_or_else(|_| "full".to_owned());
    let rounds: usize = std::env::var("QUILL_E49_ROUNDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(41);
    let limit: usize = std::env::var("QUILL_E49_LIMIT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(10);
    let shapes: &'static [Shape] = if scale == "smoke" {
        &SMOKE_SHAPES
    } else {
        &FULL_SHAPES
    };
    eprintln!(
        "[harness] scale={scale} rounds={rounds} limit={limit} rayon_threads={}",
        rayon::current_num_threads()
    );
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for shape in shapes {
            let seed = 0x0e49_0000_0000_0001_u64
                ^ (u64::try_from(shape.segments).expect("segment count") << 32);
            let built_at = Instant::now();
            let index = build_index(&cx, shape, seed).await;
            eprintln!(
                "[setup] shape={} docs={} build_ms={:.1}",
                shape.name,
                shape.segments * shape.docs_per_segment,
                built_at.elapsed().as_secs_f64() * 1_000.0,
            );
            run_shape(&cx, shape, rounds, limit, &index);
        }
    });
}
