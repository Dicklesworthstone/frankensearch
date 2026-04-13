use frankensearch_core::{
    ExplainedSource, ExplanationPhase, HitExplanation, RankMovement, ScoreComponent,
};
use frankensearch_fsfs::query_execution::{
    DegradationOverride, DegradationStatus, DegradationTransition,
};
use frankensearch_fsfs::{
    CompatibilityMode, DegradedRetrievalMode, FsfsExplanationPayload, OutputEnvelope, OutputMeta,
    PolicyDecisionExplanation, PolicyDomain, RankingExplanation, TraceLink, validate_envelope,
};

fn assert_golden_json<T: serde::Serialize>(name: &str, value: &T) {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden");
    let golden_path = dir.join(format!("{name}.golden.json"));
    let actual = serde_json::to_string_pretty(value).expect("serialize golden json");

    if std::env::var("UPDATE_GOLDENS").is_ok() {
        std::fs::create_dir_all(&dir).expect("create golden directory");
        std::fs::write(&golden_path, actual.as_bytes()).expect("write golden file");
        return;
    }

    let expected = std::fs::read_to_string(&golden_path).unwrap_or_else(|_| {
        panic!(
            "Golden file not found: {}\nSet UPDATE_GOLDENS=1 to create it.",
            golden_path.display()
        );
    });

    let actual_trimmed = actual.trim_end_matches(|c| c == '\n' || c == '\r');
    let expected_trimmed = expected.trim_end_matches(|c| c == '\n' || c == '\r');

    if actual_trimmed != expected_trimmed {
        let actual_path = golden_path.with_extension("actual.json");
        std::fs::write(&actual_path, actual_trimmed.as_bytes()).expect("write actual file");
        panic!(
            "GOLDEN MISMATCH: {name}\nexpected: {}\nactual: {}",
            golden_path.display(),
            actual_path.display()
        );
    }
}

fn sample_ranking() -> RankingExplanation {
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

    RankingExplanation::from_hit_explanation("doc-42", &explanation, "query.explain.attached", 950)
}

#[test]
fn explanation_payload_validates_inside_output_envelope() {
    let transition = DegradationTransition {
        from: DegradedRetrievalMode::EmbedDeferred,
        to: DegradedRetrievalMode::LexicalOnly,
        changed: true,
        reason_code: "degrade.transition.lexical_only",
        status: DegradationStatus {
            banner: "Degraded mode",
            controls_hint: "override:auto|normal",
        },
        override_mode: DegradationOverride::Auto,
    };
    let payload = FsfsExplanationPayload::new("rust async search", sample_ranking())
        .with_trace(TraceLink::root(
            "01JABCDEF00000000000000000",
            "01JABCDEF00000000000000001",
        ))
        .with_policy_decision(PolicyDecisionExplanation::from(&transition));

    let envelope = OutputEnvelope::success(
        payload,
        OutputMeta::new("explain", "json").with_request_id("01JABCDEF00000000000000002"),
        "2026-02-14T00:00:00Z",
    );
    let validation = validate_envelope(&envelope, CompatibilityMode::Strict);
    assert!(
        validation.valid,
        "envelope must satisfy strict output contract"
    );

    assert_golden_json("explanation_payload_envelope_v1", &envelope);

    let encoded = serde_json::to_string(&envelope).expect("serialize output envelope");
    let decoded: serde_json::Value =
        serde_json::from_str(&encoded).expect("deserialize output envelope");
    assert_eq!(decoded["meta"]["command"].as_str(), Some("explain"));
    assert_eq!(
        decoded["data"]["schema_version"].as_str(),
        Some("fsfs.explanation.payload.v1")
    );
    assert_eq!(
        decoded["data"]["policy_decisions"][0]["reason_code"].as_str(),
        Some("degrade.transition.lexical_only")
    );
    assert_eq!(
        decoded["data"]["policy_decisions"][0]["metadata"]["manual_intervention"].as_str(),
        Some("false")
    );
    assert_eq!(
        decoded["data"]["policy_decisions"][0]["metadata"]["transition_context"].as_str(),
        Some("pressure_escalation")
    );
    assert!(
        decoded["data"]["policy_decisions"][0]["metadata"]["override_guardrail"]
            .as_str()
            .is_some_and(|text| text.contains("metadata-only and pause"))
    );
}

#[test]
fn explanation_payload_toon_and_tui_use_stable_labels() {
    let payload = FsfsExplanationPayload::new("hybrid ranking", sample_ranking())
        .with_policy_decision(PolicyDecisionExplanation {
            domain: PolicyDomain::QueryExecution,
            decision: "normal".to_owned(),
            reason_code: "query.execution.normal".to_owned(),
            confidence_per_mille: 910,
            summary: "normal execution plan".to_owned(),
            metadata: std::collections::BTreeMap::new(),
        });

    assert_golden_json("explanation_payload_tui_and_toon_v1", &payload);

    let toon = payload.to_toon();
    assert!(toon.contains("phase=refined"));
    assert!(toon.contains("source=semantic_quality"));
    assert!(toon.contains("domain=query_execution"));

    let panel = payload.to_tui_panel();
    assert_eq!(panel.title, "Explainability");
    assert!(
        panel
            .lines
            .iter()
            .any(|line| line.contains("query_execution"))
    );
}
