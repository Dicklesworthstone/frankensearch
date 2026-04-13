use std::fs;
use std::path::PathBuf;

use frankensearch_core::{HitExplanation, RankMovement, ScoreComponent, ExplainedSource, ExplanationPhase};

fn assert_golden_json<T: serde::Serialize>(name: &str, value: &T) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden");
    let golden_path = dir.join(format!("{name}.golden.json"));
    let actual = serde_json::to_string_pretty(value).expect("serialize golden json");

    if std::env::var("UPDATE_GOLDENS").is_ok() {
        fs::create_dir_all(&dir).expect("create golden directory");
        fs::write(&golden_path, actual.as_bytes()).expect("write golden file");
        return;
    }

    let expected = fs::read_to_string(&golden_path).unwrap_or_else(|_| {
        panic!(
            "Golden file not found: {}\nSet UPDATE_GOLDENS=1 to create it.",
            golden_path.display()
        );
    });

    let actual_trimmed = actual.trim_end_matches(|c| c == '\n' || c == '\r');
    let expected_trimmed = expected.trim_end_matches(|c| c == '\n' || c == '\r');

    if actual_trimmed != expected_trimmed {
        let actual_path = golden_path.with_extension("actual.json");
        fs::write(&actual_path, actual_trimmed.as_bytes()).expect("write actual file");
        panic!(
            "GOLDEN MISMATCH: {name}\nexpected: {}\nactual: {}",
            golden_path.display(),
            actual_path.display()
        );
    }
}

#[test]
fn test_hit_explanation_conformance() {
    let explanation = HitExplanation {
        final_score: 0.812,
        phase: ExplanationPhase::Refined,
        rank_movement: Some(RankMovement {
            initial_rank: 5,
            refined_rank: 2,
            delta: -3,
            reason: "quality model promoted result".to_owned(),
        }),
        components: vec![ScoreComponent {
            source: ExplainedSource::SemanticQuality {
                embedder: "all-MiniLM-L6-v2".to_owned(),
                cosine_sim: 0.92,
            },
            raw_score: 0.92,
            normalized_score: 0.87,
            rrf_contribution: 0.014,
            weight: 0.7,
        }],
    };
    
    assert_golden_json("hit_explanation_v1", &explanation);
}
