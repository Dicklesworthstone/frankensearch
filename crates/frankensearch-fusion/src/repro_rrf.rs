
#[cfg(test)]
mod tests {
    use crate::rrf::{rrf_fuse, RrfConfig, candidate_count}; // candidate_count is pub
    use frankensearch_core::{ScoredResult, VectorHit, ScoreSource};

    fn lexical_hit(doc_id: &str, score: f32) -> ScoredResult {
        ScoredResult {
            doc_id: doc_id.into(),
            score,
            source: ScoreSource::Lexical,
            fast_score: None,
            quality_score: None,
            lexical_score: Some(score),
            rerank_score: None,
            explanation: None,
            metadata: None,
        }
    }

    #[test]
    fn repro_rrf_duplicate_handling() {
        let config = RrfConfig::default();
        // Same doc_id appearing twice in lexical results
        let lexical = vec![
            lexical_hit("dup", 10.0), // rank 0
            lexical_hit("other", 8.0),
            lexical_hit("dup", 5.0), // rank 2 (duplicate)
        ];

        let results = rrf_fuse(&lexical, &[], 10, 0, &config);

        let dup_hit = results.iter().find(|r| r.doc_id == "dup").unwrap();
        
        // Current buggy behavior: sums both scores
        // Correct behavior: should take the first (best) rank only
        
        // RRF score for rank 0 with k=60 is 1/(60+1) = 1/61 ~= 0.016393
        let expected_single = 1.0 / (60.0 + 1.0);
        
        assert!(
            (dup_hit.rrf_score - expected_single).abs() < 1e-12,
            "RRF score inflated by duplicate! Expected {}, got {}",
            expected_single,
            dup_hit.rrf_score
        );
    }
}
