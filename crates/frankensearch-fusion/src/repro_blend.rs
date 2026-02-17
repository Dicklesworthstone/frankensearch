#[cfg(test)]
mod tests {
    use crate::blend::blend_two_tier;
    use frankensearch_core::VectorHit;

    fn hit(doc_id: &str, score: f32) -> VectorHit {
        VectorHit {
            index: 0,
            score,
            doc_id: doc_id.to_owned(),
        }
    }

    #[test]
    fn single_result_normalization_destroys_magnitude() {
        // High quality match
        let fast_high = vec![hit("doc-high", 0.99)];
        let blended_high = blend_two_tier(&fast_high, &[], 0.5);
        
        // Low quality match
        let fast_low = vec![hit("doc-low", 0.01)];
        let blended_low = blend_two_tier(&fast_low, &[], 0.5);

        // Fixed behavior: magnitudes preserved.
        // 0.99 * 0.5 = 0.495
        // 0.01 * 0.5 = 0.005
        assert!(blended_high[0].score > 0.4, "High score should be preserved");
        assert!(blended_low[0].score < 0.1, "Low score should be preserved");
        assert!(blended_high[0].score > blended_low[0].score);
    }

    #[test]
    fn small_set_normalization_distorts_distances() {
        // Two matches, very close in reality
        let fast = vec![
            hit("doc-a", 0.90),
            hit("doc-b", 0.89),
        ];
        
        // Range 0.01 is borderline. robust_normalize uses > 0.01.
        // 0.01 is NOT > 0.01. So it should clamp.
        // doc-a -> 0.90
        // doc-b -> 0.89
        
        let blended = blend_two_tier(&fast, &[], 0.0); // fast-only (alpha=0 -> use fast score directly)
        
        // Scores should be preserved (clamped), not amplified to 1.0/0.0.
        assert_eq!(blended[0].doc_id, "doc-a");
        assert!((blended[0].score - 0.90).abs() < 1e-5);
        
        assert_eq!(blended[1].doc_id, "doc-b");
        assert!((blended[1].score - 0.89).abs() < 1e-5);
    }
}
