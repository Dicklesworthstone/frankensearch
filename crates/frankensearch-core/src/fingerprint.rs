use serde::{Deserialize, Serialize};

use crate::filter::fnv1a_hash;

/// Default semantic-change threshold used by [`DocumentFingerprint::needs_reembedding_default`].
///
/// This corresponds to a Hamming distance of 8 bits out of 64 (`8 / 64 = 0.125`).
pub const DEFAULT_SEMANTIC_CHANGE_THRESHOLD: f64 = 8.0 / 64.0;

/// Character-count change ratio that always triggers re-embedding.
///
/// If `abs(a - b) / max(a, b) > 0.20`, the change is treated as significant
/// regardless of semantic hash distance.
pub const SIGNIFICANT_CHAR_COUNT_CHANGE_THRESHOLD: f64 = 0.20;

const SHINGLE_SIZE: usize = 3;
const SIMHASH_BITS: f64 = 64.0;
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Content-aware fingerprint for deciding whether document embeddings should be refreshed.
///
/// The fingerprint combines:
/// - exact-change detection (`content_hash`)
/// - approximate semantic-change detection (`semantic_hash`)
/// - cheap structural change signals (`char_count`, `token_estimate`)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DocumentFingerprint {
    /// FNV-1a hash of the full raw document text (exact-change detector).
    pub content_hash: u64,
    /// 64-bit `SimHash` over 3-token shingles (approximate semantic detector).
    pub semantic_hash: u64,
    /// Character count (`text.chars().count()`), saturated to `u32`.
    pub char_count: u32,
    /// Approximate token count (`split_whitespace`), saturated to `u32`.
    pub token_estimate: u32,
}

impl DocumentFingerprint {
    /// Compute a fingerprint for document text.
    #[must_use]
    pub fn compute(text: &str) -> Self {
        let tokens: Vec<&str> = text.split_whitespace().collect();

        Self {
            content_hash: fnv1a_hash(text.as_bytes()),
            semantic_hash: semantic_simhash(&tokens),
            char_count: usize_to_u32_saturating(text.chars().count()),
            token_estimate: usize_to_u32_saturating(tokens.len()),
        }
    }

    /// Return the Hamming distance between semantic hashes (0-64).
    #[must_use]
    pub const fn semantic_hamming_distance(&self, other: &Self) -> u32 {
        (self.semantic_hash ^ other.semantic_hash).count_ones()
    }

    /// Return semantic distance as a normalized ratio in `[0.0, 1.0]`.
    #[must_use]
    pub fn semantic_distance_ratio(&self, other: &Self) -> f64 {
        f64::from(self.semantic_hamming_distance(other)) / SIMHASH_BITS
    }

    /// Return normalized character-count difference in `[0.0, 1.0]`.
    #[must_use]
    pub fn char_count_delta_ratio(&self, other: &Self) -> f64 {
        let max_count = self.char_count.max(other.char_count);
        if max_count == 0 {
            return 0.0;
        }

        f64::from(self.char_count.abs_diff(other.char_count)) / f64::from(max_count)
    }

    /// Decide whether re-embedding should run.
    ///
    /// Decision rules:
    /// 1. If exact content hash is unchanged: `false`
    /// 2. If character-count delta ratio is > 20%: `true`
    /// 3. Else if semantic distance ratio is greater than `threshold`: `true`
    /// 4. Otherwise: `false`
    ///
    /// `threshold` is interpreted as a ratio in `[0.0, 1.0]` and is clamped to that range.
    #[must_use]
    pub fn needs_reembedding(&self, other: &Self, threshold: f64) -> bool {
        if self.content_hash == other.content_hash {
            return false;
        }

        if self.char_count_delta_ratio(other) > SIGNIFICANT_CHAR_COUNT_CHANGE_THRESHOLD {
            return true;
        }

        // NaN.clamp() returns NaN, making the comparison always false and
        // silently disabling semantic change detection. Fall back to default.
        let safe_threshold = if threshold.is_finite() {
            threshold.clamp(0.0, 1.0)
        } else {
            DEFAULT_SEMANTIC_CHANGE_THRESHOLD
        };
        self.semantic_distance_ratio(other) > safe_threshold
    }

    /// Convenience wrapper using [`DEFAULT_SEMANTIC_CHANGE_THRESHOLD`].
    #[must_use]
    pub fn needs_reembedding_default(&self, other: &Self) -> bool {
        self.needs_reembedding(other, DEFAULT_SEMANTIC_CHANGE_THRESHOLD)
    }
}

#[must_use]
fn usize_to_u32_saturating(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

#[must_use]
fn semantic_simhash(tokens: &[&str]) -> u64 {
    if tokens.is_empty() {
        return 0;
    }

    let mut bit_weights = [0_i32; 64];

    if tokens.len() < SHINGLE_SIZE {
        for token in tokens {
            apply_hash_votes(fnv1a_hash(token.as_bytes()), &mut bit_weights);
        }
    } else {
        for window in tokens.windows(SHINGLE_SIZE) {
            apply_hash_votes(hash_token_window(window), &mut bit_weights);
        }
    }

    let mut semantic_hash = 0_u64;
    for (bit, weight) in bit_weights.iter().enumerate() {
        if *weight > 0 {
            semantic_hash |= 1_u64 << bit;
        }
    }

    semantic_hash
}

fn apply_hash_votes(hash: u64, bit_weights: &mut [i32; 64]) {
    for (bit, weight) in bit_weights.iter_mut().enumerate() {
        if hash & (1_u64 << bit) == 0 {
            *weight -= 1;
        } else {
            *weight += 1;
        }
    }
}

#[must_use]
fn hash_token_window(window: &[&str]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;

    for token in window {
        for byte in token.bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash ^= u64::from(b' ');
        hash = hash.wrapping_mul(FNV_PRIME);
    }

    hash
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_SEMANTIC_CHANGE_THRESHOLD, DocumentFingerprint,
        SIGNIFICANT_CHAR_COUNT_CHANGE_THRESHOLD,
    };

    #[test]
    fn identical_text_produces_identical_fingerprint() {
        let a = DocumentFingerprint::compute("rust async search");
        let b = DocumentFingerprint::compute("rust async search");
        assert_eq!(a, b);
        assert!(!a.needs_reembedding_default(&b));
    }

    #[test]
    fn formatting_change_keeps_semantic_hash() {
        let a = DocumentFingerprint::compute("rust async search pipeline");
        let b = DocumentFingerprint::compute("rust   async   search pipeline");

        assert_ne!(a.content_hash, b.content_hash);
        assert_eq!(a.semantic_hash, b.semantic_hash);
        assert!(!a.needs_reembedding_default(&b));
    }

    #[test]
    fn significant_change_triggers_reembedding() {
        let a = DocumentFingerprint::compute(
            "distributed systems consensus quorum replication failover",
        );
        let b = DocumentFingerprint::compute(
            "banana smoothie recipe tropical fruit yogurt dessert breakfast",
        );

        assert_ne!(a.content_hash, b.content_hash);
        assert!(a.semantic_hamming_distance(&b) >= 8);
        assert!(a.needs_reembedding_default(&b));
    }

    #[test]
    fn threshold_controls_semantic_sensitivity() {
        let a = DocumentFingerprint::compute("alpha beta gamma delta epsilon zeta eta theta");
        let b = DocumentFingerprint::compute("alpha beta gamma delta epsilon zeta eta iota");

        assert_ne!(a.content_hash, b.content_hash);
        assert!(a.semantic_hamming_distance(&b) > 0);
        assert!(a.char_count_delta_ratio(&b) <= SIGNIFICANT_CHAR_COUNT_CHANGE_THRESHOLD);

        assert!(a.needs_reembedding(&b, 0.0));
        assert!(!a.needs_reembedding(&b, 1.0));
    }

    #[test]
    fn char_count_delta_above_threshold_forces_reembedding() {
        let short = DocumentFingerprint::compute("short text");
        let long = DocumentFingerprint::compute(&"short text ".repeat(40));

        assert!(short.char_count_delta_ratio(&long) > SIGNIFICANT_CHAR_COUNT_CHANGE_THRESHOLD);
        assert!(short.needs_reembedding_default(&long));
        assert!(long.needs_reembedding_default(&short));
    }

    #[test]
    fn token_and_char_counts_are_reported() {
        let text = "hello rust 世界\nnew line";
        let fp = DocumentFingerprint::compute(text);

        assert_eq!(fp.token_estimate, 5);
        assert_eq!(
            fp.char_count,
            u32::try_from(text.chars().count()).unwrap_or(u32::MAX)
        );
    }

    #[test]
    fn empty_and_whitespace_inputs_are_handled() {
        let empty = DocumentFingerprint::compute("");
        assert_eq!(empty.semantic_hash, 0);
        assert_eq!(empty.token_estimate, 0);
        assert_eq!(empty.char_count, 0);

        let whitespace = DocumentFingerprint::compute("   \n\t ");
        assert_eq!(whitespace.semantic_hash, 0);
        assert_eq!(whitespace.token_estimate, 0);
        assert_eq!(whitespace.char_count, 6);
    }

    #[test]
    fn unicode_input_is_supported() {
        let text = "東京 rust 🚀 مرحبا بالعالم";
        let fp = DocumentFingerprint::compute(text);

        assert!(fp.content_hash != 0);
        assert!(fp.token_estimate >= 4);
        assert_eq!(
            fp.char_count,
            u32::try_from(text.chars().count()).unwrap_or(u32::MAX)
        );
    }

    #[test]
    fn reembedding_decision_is_symmetric() {
        let a = DocumentFingerprint::compute("graph pagerank shortest path");
        let b = DocumentFingerprint::compute("graph pagerank shortest route");

        assert_eq!(
            a.needs_reembedding(&b, DEFAULT_SEMANTIC_CHANGE_THRESHOLD),
            b.needs_reembedding(&a, DEFAULT_SEMANTIC_CHANGE_THRESHOLD),
        );
    }

    #[test]
    fn large_document_computes_without_overflow() {
        let text = "token ".repeat(200_000);
        let fp = DocumentFingerprint::compute(&text);

        assert_eq!(fp.token_estimate, 200_000);
        assert!(fp.char_count >= 1_200_000);
    }

    // ─── bd-1b49 tests begin ───

    #[test]
    fn fingerprint_serde_roundtrip() {
        let fp = DocumentFingerprint::compute("hello world rust search");
        let json = serde_json::to_string(&fp).unwrap();
        let decoded: DocumentFingerprint = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, fp);
    }

    #[test]
    fn fingerprint_debug_format() {
        let fp = DocumentFingerprint::compute("test");
        let debug = format!("{fp:?}");
        assert!(debug.contains("content_hash"));
        assert!(debug.contains("semantic_hash"));
        assert!(debug.contains("char_count"));
        assert!(debug.contains("token_estimate"));
    }

    #[test]
    fn semantic_hamming_distance_with_self_is_zero() {
        let fp = DocumentFingerprint::compute("alpha beta gamma delta epsilon zeta");
        assert_eq!(fp.semantic_hamming_distance(&fp), 0);
    }

    #[test]
    fn semantic_distance_ratio_exact_value() {
        let a = DocumentFingerprint::compute("test");
        let mut b = a;
        // Force exactly 8 bits different in semantic_hash
        b.semantic_hash = a.semantic_hash ^ 0xFF; // 8 bits flipped
        b.content_hash = !a.content_hash; // ensure different content hash
        let ratio = a.semantic_distance_ratio(&b);
        assert!(
            (ratio - 8.0 / 64.0).abs() < f64::EPSILON,
            "expected 0.125, got {ratio}"
        );
    }

    #[test]
    fn char_count_delta_ratio_both_zero() {
        let a = DocumentFingerprint::compute("");
        let b = DocumentFingerprint::compute("");
        assert!(a.char_count_delta_ratio(&b).abs() < f64::EPSILON);
    }

    #[test]
    fn char_count_delta_ratio_exact_calculation() {
        let mut a = DocumentFingerprint::compute("test");
        let mut b = a;
        a.char_count = 100;
        b.char_count = 80;
        // delta = 20, max = 100, ratio = 0.2
        assert!((a.char_count_delta_ratio(&b) - 0.2).abs() < f64::EPSILON);
        assert!((b.char_count_delta_ratio(&a) - 0.2).abs() < f64::EPSILON);
    }

    #[test]
    fn needs_reembedding_same_hash_always_false() {
        let a = DocumentFingerprint::compute("hello world");
        let mut b = a;
        // Same content_hash, but fake different semantic_hash
        b.semantic_hash = !a.semantic_hash;
        b.char_count = a.char_count * 5;
        // content_hash still matches → should NOT trigger reembedding
        assert!(!a.needs_reembedding(&b, 0.0));
        assert!(!a.needs_reembedding_default(&b));
    }

    #[test]
    fn needs_reembedding_negative_threshold_clamped_to_zero() {
        let a = DocumentFingerprint::compute("alpha beta gamma delta");
        let b = DocumentFingerprint::compute("alpha beta gamma epsilon");
        // Any hamming distance > 0 should trigger with threshold clamped to 0.0
        if a.content_hash != b.content_hash && a.semantic_hamming_distance(&b) > 0 {
            assert!(a.needs_reembedding(&b, -1.0));
        }
    }

    #[test]
    fn two_token_input_uses_individual_hashing() {
        // With < SHINGLE_SIZE (3) tokens, individual tokens are hashed
        let fp = DocumentFingerprint::compute("hello world");
        assert_eq!(fp.token_estimate, 2);
        assert!(fp.semantic_hash != 0); // should still produce a non-zero hash
    }

    #[test]
    fn constants_have_expected_values() {
        assert!(
            (DEFAULT_SEMANTIC_CHANGE_THRESHOLD - 0.125).abs() < f64::EPSILON,
            "DEFAULT_SEMANTIC_CHANGE_THRESHOLD should be 8/64 = 0.125"
        );
        assert!(
            (SIGNIFICANT_CHAR_COUNT_CHANGE_THRESHOLD - 0.20).abs() < f64::EPSILON,
            "SIGNIFICANT_CHAR_COUNT_CHANGE_THRESHOLD should be 0.20"
        );
    }

    // ─── bd-1b49 tests end ───
}
