//! Default-analyzer tokenizer admission probe: the `bd-l5x3` boundary-mask
//! candidate vs Quill's shipping two-pass SWAR classifier.
//!
//! Tokenization is a *full-scan classify*: every input byte is visited to find
//! token boundaries, which is precisely the shape where SWAR/SIMD pays — unlike
//! the `memchr`/`contains` early-exit scans that regress when fused (bd-5hz0).
//! The A/A null and A/B run in one process through the shared median-CI harness.
//! The scalar implementation is retained as a parity oracle, not a timed arm.
//!
//! Two corpora pin the intended short-token gain and prevent a long-token
//! regression. Production stays on the shipping implementation until a later
//! measurement window admits the candidate.
//!
//! Both implementations and the scalar oracle are asserted byte-identical
//! (offsets + text) before timing.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR \
//!   rch exec -- cargo run --profile release-perf -p frankensearch-quill \
//!     --features bench-internals --bin tokenizer_simd_ab
//! ```

use std::hint::black_box;

use frankensearch_core::bench_support::{paired_median_ratio, print_bench_elf_sha256};
use frankensearch_quill::Analyzer;
use frankensearch_quill::scribe::{
    BoundaryMaskTokenizer, FrankensearchTokenizer, TokenAnalyzer, analyze_default_scalar_reference,
};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Fold one token's identity (source offsets + normalized bytes) into a running
/// digest. Shared across all arms so cross-implementation parity is one compare.
#[inline]
fn fold_token(mut digest: u64, offset_from: usize, offset_to: usize, text: &str) -> u64 {
    digest ^= u64::try_from(offset_from).unwrap_or(u64::MAX);
    digest = digest.wrapping_mul(FNV_PRIME);
    digest ^= u64::try_from(offset_to).unwrap_or(u64::MAX);
    digest = digest.wrapping_mul(FNV_PRIME);
    for byte in text.bytes() {
        digest ^= u64::from(byte);
        digest = digest.wrapping_mul(FNV_PRIME);
    }
    digest
}

fn quill_digest<A: TokenAnalyzer>(analyzer: &mut A, text: &str) -> u64 {
    let mut digest = FNV_OFFSET;
    analyzer.analyze(Analyzer::FrankensearchDefault, text, &mut |token| {
        digest = fold_token(digest, token.offset_from, token.offset_to, &token.text);
    });
    digest
}

fn quill_scalar_digest(text: &str) -> u64 {
    let mut digest = FNV_OFFSET;
    analyze_default_scalar_reference(text, &mut |token| {
        digest = fold_token(digest, token.offset_from, token.offset_to, &token.text);
    });
    digest
}

/// Realistic mostly-ASCII corpus (English prose + code identifiers + IDs) with a
/// small non-ASCII fraction — what the default analyzer actually sees at index
/// and query time — sized so the classifier, not allocation, dominates.
fn corpus() -> String {
    let words = [
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "dog",
        "search",
        "index",
        "tokenizer",
        "bd-q3fy",
        "ID_42",
        "camelCase",
        "snake_case_name",
        "http",
        "vector",
        "embedding",
        "rerank",
        "POL-358",
        "Rust2024",
        "café", // one accented word so the corpus is not 100% ASCII
    ];
    let mut text = String::with_capacity(64 * 1024);
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    while text.len() < 48 * 1024 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let idx = usize::try_from(state % words.len() as u64).unwrap_or(0);
        text.push_str(words[idx]);
        text.push(' ');
    }
    text
}

/// Long-run corpus: 24–48 byte alphanumeric tokens (hashes, base64, UUIDs,
/// long identifiers) separated by long whitespace runs — the shape where an
/// 8-lanes-per-op SWAR classifier amortizes its per-window mask setup, unlike
/// the short space-separated tokens of [`corpus`].
fn long_token_corpus() -> String {
    let words = [
        "9f8c2a1b7e4d6035af19cd82b73e05461fa9c7d20e8b34a6",
        "aGVsbG8gd29ybGQgdGhpcyBpcyBhIGxvbmcgYmFzZTY0IHN0cmluZw",
        "550e8400e29b41d4a716446655440000deadbeefcafef00d",
        "extremely_long_snake_case_identifier_for_the_tokenizer_benchmark",
        "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789AbCdEfGhIjKl",
    ];
    let mut text = String::with_capacity(64 * 1024);
    let mut state = 0x106c_9b1f_2a37_d45e_u64;
    while text.len() < 48 * 1024 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let idx = usize::try_from(state % words.len() as u64).unwrap_or(0);
        text.push_str(words[idx]);
        text.push_str("   \n\t  "); // long separator run
    }
    text
}

fn measure_default_tokenizer() {
    let text = corpus();

    // Parity before timing: every arm must emit the identical token stream.
    let mut candidate = BoundaryMaskTokenizer::default();
    let mut shipping = FrankensearchTokenizer::default();
    let scalar_ref = quill_scalar_digest(&text);
    assert_eq!(
        quill_digest(&mut candidate, &text),
        scalar_ref,
        "Quill boundary-mask tokenizer diverged from the scalar char-walk reference"
    );
    assert_eq!(
        quill_digest(&mut shipping, &text),
        scalar_ref,
        "retained shipping SWAR tokenizer diverged from the scalar char-walk reference"
    );
    // NULL (shipping vs shipping), then the lever (shipping vs boundary mask).
    // Ratio = candidate/shipping, < 1.0 = boundary batching wins. `inner`
    // batches 64 full-corpus passes so per-batch scheduler jitter does not
    // dominate a sub-250µs single pass.
    let mut shipping_a = FrankensearchTokenizer::default();
    let mut shipping_b = FrankensearchTokenizer::default();
    let null = paired_median_ratio(
        41,
        64,
        || {
            black_box(quill_digest(&mut shipping_a, black_box(&text)));
        },
        || {
            black_box(quill_digest(&mut shipping_b, black_box(&text)));
        },
    );
    let mut shipping_c = FrankensearchTokenizer::default();
    let mut candidate_a = BoundaryMaskTokenizer::default();
    let lever = paired_median_ratio(
        41,
        64,
        || {
            black_box(quill_digest(&mut shipping_c, black_box(&text)));
        },
        || {
            black_box(quill_digest(&mut candidate_a, black_box(&text)));
        },
    );
    eprintln!(
        "[null] tokenizer_boundary_mask/{}KiB shipping/shipping median {:.4} \
         median_ci95 [{:.4}, {:.4}] p5 {:.4} p95 {:.4} admissible={} ({} rounds)",
        text.len() / 1024,
        null.median,
        null.median_ci95_low,
        null.median_ci95_high,
        null.p5,
        null.p95,
        null.is_admissible_null(),
        null.rounds
    );
    eprintln!(
        "[lever] tokenizer_boundary_mask/{}KiB candidate/shipping median {:.4} \
         median_ci95 [{:.4}, {:.4}] p5 {:.4} p95 {:.4} -> {}",
        text.len() / 1024,
        lever.median,
        lever.median_ci95_low,
        lever.median_ci95_high,
        lever.p5,
        lever.p95,
        if lever.decidable_against(&null) {
            if lever.median < 1.0 {
                "DECIDABLE WIN (boundary batching faster)"
            } else {
                "DECIDABLE REGRESSION"
            }
        } else {
            "INSIDE NULL FLOOR (not decidable)"
        }
    );
}

fn measure_long_token_tokenizer() {
    let text = long_token_corpus();

    let scalar_ref = quill_scalar_digest(&text);
    let mut candidate = BoundaryMaskTokenizer::default();
    let mut shipping = FrankensearchTokenizer::default();
    assert_eq!(
        quill_digest(&mut candidate, &text),
        scalar_ref,
        "Quill boundary-mask tokenizer diverged from the scalar reference on the long-token corpus"
    );
    assert_eq!(
        quill_digest(&mut shipping, &text),
        scalar_ref,
        "retained shipping SWAR tokenizer diverged from the scalar reference on the long-token corpus"
    );

    let mut shipping_a = FrankensearchTokenizer::default();
    let mut shipping_b = FrankensearchTokenizer::default();
    let null = paired_median_ratio(
        41,
        64,
        || {
            black_box(quill_digest(&mut shipping_a, black_box(&text)));
        },
        || {
            black_box(quill_digest(&mut shipping_b, black_box(&text)));
        },
    );
    let mut shipping_c = FrankensearchTokenizer::default();
    let mut candidate_a = BoundaryMaskTokenizer::default();
    let lever = paired_median_ratio(
        41,
        64,
        || {
            black_box(quill_digest(&mut shipping_c, black_box(&text)));
        },
        || {
            black_box(quill_digest(&mut candidate_a, black_box(&text)));
        },
    );
    eprintln!(
        "[null] tokenizer_boundary_mask_long/{}KiB shipping/shipping median {:.4} \
         median_ci95 [{:.4}, {:.4}] p5 {:.4} p95 {:.4} admissible={} ({} rounds)",
        text.len() / 1024,
        null.median,
        null.median_ci95_low,
        null.median_ci95_high,
        null.p5,
        null.p95,
        null.is_admissible_null(),
        null.rounds
    );
    eprintln!(
        "[lever] tokenizer_boundary_mask_long/{}KiB candidate/shipping median {:.4} \
         median_ci95 [{:.4}, {:.4}] p5 {:.4} p95 {:.4} -> {}",
        text.len() / 1024,
        lever.median,
        lever.median_ci95_low,
        lever.median_ci95_high,
        lever.p5,
        lever.p95,
        if lever.decidable_against(&null) {
            if lever.median < 1.0 {
                "DECIDABLE WIN (boundary batching faster)"
            } else {
                "DECIDABLE REGRESSION"
            }
        } else {
            "INSIDE NULL FLOOR (not decidable)"
        }
    );
}

fn main() {
    let _identity = print_bench_elf_sha256().expect("hash executing tokenizer benchmark");
    measure_default_tokenizer();
    measure_long_token_tokenizer();
}
