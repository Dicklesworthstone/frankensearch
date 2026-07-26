//! Deferred grouped `MaxScore` A/B for nested pure-term unions
//! (`bd-quill-e8-perf-doctrine-x4e4.5.1`).
//!
//! # What is under measurement
//!
//! A default multi-field query (`alpha OR beta`) lowers to a root `Should`
//! union whose children are themselves per-term unions over `content` + `title`
//! — `Union { direct_terms: false }`. Before this lever that shape failed the
//! rank-pruning gate outright, so the root scored **every** 4096-doc window with
//! no cutoff. Grouped `MaxScore` orders the child *groups* by a conservative
//! whole-group ceiling and only opens a non-essential group's window when the
//! running k-th cutoff makes it competitive.
//!
//! Arm A is the pre-lever exhaustive path (`rank_pruning = false`); arm B is the
//! grouped-pruned path (`rank_pruning = true`). Both run the identical shipping
//! pipeline over the identical published snapshot in the **same binary**; the
//! only variable is whether pruning metadata is opened and consumed.
//!
//! # Why the ratio is allowed to be believed
//!
//! Per the fleet bench-harness contract:
//!
//! 1. **ELF SHA-256 self-report.** Line 1 of stderr is the hash of
//!    `env::current_exe()` — computed by the binary that actually ran, not by a
//!    shell step standing next to it. `rch` compiles into an opaque per-worker
//!    pool target dir, and concurrent agents edit this crate continuously, so a
//!    hash computed anywhere else proves nothing.
//! 2. **A/A null control in the same invocation**, interleaved, order
//!    alternating per round, statistic = median of per-round ratios.
//! 3. **Gate on the median-CI, never on `cv`.** The decision is delegated to
//!    `bench_support::PairedRatio::decidable_against`, the fleet's canonical
//!    gate: the null must itself be admissible, and the candidate median must
//!    clear parity by the required multiple of the null's median-CI half-width.
//!    Raw `p5`/`p95` and `cv` are printed as provenance only and never decide
//!    anything — `cv` is unreachable below ~12% on this hardware and does not
//!    track decidability.
//!
//! Bit-parity is proven **before** any timing: every cell asserts the pruned and
//! exhaustive arms return identical `(global_docid, raw score bits)` pages. That
//! is the rank-safety contract — grouped `MaxScore` may reorder *work*, never
//! results.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR \
//!   rch exec -- env QUILL_E851_SCALE=full QUILL_E851_ROUNDS=41 \
//!     cargo bench -p frankensearch-quill --features bench-internals \
//!       --profile release --bench grouped_maxscore_ab
//!
//! # Fast parity/harness check (NOT performance evidence):
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR \
//!   rch exec -- env QUILL_E851_SCALE=smoke QUILL_E851_ROUNDS=5 \
//!     cargo bench -p frankensearch-quill --features bench-internals \
//!       --profile release --bench grouped_maxscore_ab
//! ```

use std::hint::black_box;
use std::time::Instant;

use asupersync::Cx;
use frankensearch_core::IndexableDocument;
use frankensearch_core::bench_support::paired_median_ratio;
use frankensearch_quill::{QuillConfig, QuillIndex};
use sha2::{Digest, Sha256};

/// One benchmark corpus shape: sealed segments times documents per segment.
struct Shape {
    name: &'static str,
    segments: usize,
    docs_per_segment: usize,
}

static FULL_SHAPES: [Shape; 2] = [
    Shape {
        name: "seg8x12500",
        segments: 8,
        docs_per_segment: 12_500,
    },
    Shape {
        name: "seg4x25000",
        segments: 4,
        docs_per_segment: 25_000,
    },
];

static SMOKE_SHAPES: [Shape; 1] = [Shape {
    name: "smoke4x500",
    segments: 4,
    docs_per_segment: 500,
}];

/// Query classes. Every entry must lower to a root union of 2..=8 *groups*
/// (multi-field terms) — that is the shape grouped `MaxScore` consumes. The
/// `direct2` control is deliberately field-scoped so it lowers to direct terms
/// and takes the pre-existing term-granular `MaxScore` path instead; it should
/// show no grouped effect and acts as an in-binary negative control.
const QUERIES: [(&str, &str); 5] = [
    ("grouped2", "alpha OR beta"),
    ("grouped3_skewed", "shared OR rare OR singular"),
    ("grouped4", "alpha OR gamma OR quill OR rare"),
    (
        "grouped8",
        "alpha OR beta OR gamma OR delta OR quill OR argus OR rare OR singular",
    ),
    ("direct2_control", "content:alpha OR content:beta"),
];

const VOCABULARY: [&str; 24] = [
    "shared", "shared", "shared", "shared", "alpha", "alpha", "beta", "beta", "gamma", "gamma",
    "delta", "epsilon", "zeta", "quill", "argus", "keeper", "scribe", "grimoire", "quiver",
    "segment", "posting", "cursor", "rare", "singular",
];

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn lower_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    let mut output = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

/// SHA-256 of the running benchmark binary, per contract part 1.
fn self_elf_sha256() -> String {
    let Ok(path) = std::env::current_exe() else {
        return "unavailable".to_owned();
    };
    let Ok(bytes) = std::fs::read(&path) else {
        return "unreadable".to_owned();
    };
    lower_hex(&Sha256::digest(&bytes))
}

/// Both default text fields are populated so each query term genuinely lowers
/// to a two-cursor group (`content` + `title`). A title-less corpus would leave
/// every group with a single active child and understate the lever.
async fn build_index(cx: &Cx, shape: &Shape, seed: u64) -> QuillIndex {
    let config = QuillConfig {
        deterministic_ingest: true,
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
            let title_words = 2 + usize::try_from(xorshift(&mut state) % 4).expect("title words");
            let mut title = String::with_capacity(title_words * 8);
            for position in 0..title_words {
                if position != 0 {
                    title.push(' ');
                }
                let pick = usize::try_from(xorshift(&mut state) % 24).expect("vocabulary index");
                title.push_str(VOCABULARY[pick]);
            }
            batch.push(
                IndexableDocument::new(format!("e851-s{segment:03}-d{ordinal:05}"), text)
                    .with_title(title),
            );
        }
        index
            .index_documents(cx, &batch)
            .await
            .expect("accumulate bench batch");
        index.commit(cx).await.expect("seal bench segment");
    }
    assert_eq!(
        index.snapshot().segments().len(),
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

fn run_shape(cx: &Cx, shape: &Shape, rounds: usize, limits: &[usize], index: &QuillIndex) {
    for (query_name, query) in QUERIES {
        for &limit in limits {
            // --- Bit-parity BEFORE timing. Rank safety is the contract. ------
            let exhaustive_page = index
                .bench_search_sealed_forced(cx, query, limit, false, Some(false))
                .expect("exhaustive page");
            let pruned_page = index
                .bench_search_sealed_forced(cx, query, limit, false, Some(true))
                .expect("grouped-pruned page");
            assert_eq!(
                exhaustive_page, pruned_page,
                "{}: grouped MaxScore perturbed the top-k before timing \
                 (query={query_name} k={limit}) — docid+score-bit parity is the \
                 rank-safety contract",
                shape.name,
            );

            let null = paired_median_ratio(
                rounds,
                4,
                || {
                    black_box(
                        index
                            .bench_search_sealed_forced(
                                cx,
                                black_box(query),
                                limit,
                                false,
                                Some(false),
                            )
                            .expect("null arm a"),
                    );
                },
                || {
                    black_box(
                        index
                            .bench_search_sealed_forced(
                                cx,
                                black_box(query),
                                limit,
                                false,
                                Some(false),
                            )
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
                            .bench_search_sealed_forced(
                                cx,
                                black_box(query),
                                limit,
                                false,
                                Some(false),
                            )
                            .expect("exhaustive arm"),
                    );
                },
                || {
                    black_box(
                        index
                            .bench_search_sealed_forced(
                                cx,
                                black_box(query),
                                limit,
                                false,
                                Some(true),
                            )
                            .expect("grouped-pruned arm"),
                    );
                },
            );
            let exhaustive_us = absolute_us(16, || {
                black_box(
                    index
                        .bench_search_sealed_forced(cx, black_box(query), limit, false, Some(false))
                        .expect("exhaustive absolute"),
                );
            });
            let pruned_us = absolute_us(16, || {
                black_box(
                    index
                        .bench_search_sealed_forced(cx, black_box(query), limit, false, Some(true))
                        .expect("pruned absolute"),
                );
            });
            // `decidable` is the only verdict. The p5/p95 pair is printed for
            // provenance so a reader can see the spread behind the decision.
            eprintln!(
                "[cell] shape={} query={query_name} k={limit} \
                 null={:.4} [{:.4}, {:.4}] lever(pruned/exhaustive)={:.4} [{:.4}, {:.4}] \
                 decidable={} exhaustive_us={exhaustive_us:.1} pruned_us={pruned_us:.1}",
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
}

fn main() {
    // Contract part 1: the binary identifies itself, first line, before anything
    // else can fail.
    eprintln!("[binary] sha256={}", self_elf_sha256());

    let scale = std::env::var("QUILL_E851_SCALE").unwrap_or_else(|_| "full".to_owned());
    let rounds: usize = std::env::var("QUILL_E851_ROUNDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(41);
    let limits: Vec<usize> = std::env::var("QUILL_E851_LIMITS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|part| part.trim().parse().ok())
                .collect()
        })
        .unwrap_or_else(|| vec![1, 10, 100, 1_000]);
    let shapes: &'static [Shape] = if scale == "smoke" {
        &SMOKE_SHAPES
    } else {
        &FULL_SHAPES
    };
    eprintln!(
        "[harness] scale={scale} rounds={rounds} limits={limits:?} rayon_threads={}",
        rayon::current_num_threads()
    );
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for shape in shapes {
            let seed = 0x0e85_1000_0000_0001_u64
                ^ (u64::try_from(shape.segments).expect("segment count") << 32);
            let built_at = Instant::now();
            let index = build_index(&cx, shape, seed).await;
            eprintln!(
                "[setup] shape={} docs={} build_ms={:.1}",
                shape.name,
                shape.segments * shape.docs_per_segment,
                built_at.elapsed().as_secs_f64() * 1_000.0,
            );
            run_shape(&cx, shape, rounds, &limits, &index);
        }
    });
}
