use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use jsonschema::{Draft, JSONSchema};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("schemas/fixtures")
}

fn invalid_fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("schemas/fixtures-invalid")
}

fn schema_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("schemas")
}

fn schema_for_fixture(name: &str) -> &'static str {
    if name.starts_with("control-plane-error-") {
        return "control-plane-error-v1.schema.json";
    }
    if name.starts_with("control-plane-") {
        return "control-plane-interface-v1.schema.json";
    }
    if name.starts_with("telemetry-transport-") {
        return "telemetry-transport-v1.schema.json";
    }
    if name.starts_with("telemetry-") {
        return "telemetry-event-v1.schema.json";
    }
    if name.starts_with("ops-config-") {
        return "ops-config-v1.schema.json";
    }
    if name.starts_with("ops-telemetry-storage-") {
        return "ops-telemetry-storage-v1.schema.json";
    }
    if name.starts_with("e2e-") {
        return "e2e-artifact-v1.schema.json";
    }
    if name.starts_with("evidence-") {
        return "evidence-jsonl-v1.schema.json";
    }
    if name.starts_with("asupersync-cx-") {
        return "asupersync-cx-contract-v1.schema.json";
    }
    if name.starts_with("crate-placement-registry-") {
        return "crate-placement-registry-v1.schema.json";
    }
    if name.starts_with("slo-anomaly-") {
        return "slo-anomaly-v1.schema.json";
    }
    if name.starts_with("fsfs-alien-recommendation-") {
        return "fsfs-alien-recommendations-v1.schema.json";
    }
    if name.starts_with("fsfs-config-") {
        return "fsfs-config-v1.schema.json";
    }
    if name.starts_with("fsfs-determinism-") {
        return "fsfs-determinism-v1.schema.json";
    }
    if name.starts_with("fsfs-expected-loss-") {
        return "fsfs-expected-loss-v1.schema.json";
    }
    if name.starts_with("fsfs-explanation-payload-") {
        return "fsfs-explanation-payload-v1.schema.json";
    }
    if name.starts_with("fsfs-file-classification-") {
        return "fsfs-file-classification-v1.schema.json";
    }
    if name.starts_with("fsfs-high-cost-artifact-detectors-") {
        return "fsfs-high-cost-artifact-detectors-v1.schema.json";
    }
    if name.starts_with("fsfs-incremental-change-detection-") {
        return "fsfs-incremental-change-detection-v1.schema.json";
    }
    if name.starts_with("fsfs-packaging-release-install-") {
        return "fsfs-packaging-release-install-v1.schema.json";
    }
    if name.starts_with("fsfs-pressure-profiles-") {
        return "fsfs-pressure-profiles-v1.schema.json";
    }
    if name.starts_with("fsfs-provenance-attestation-") {
        return "fsfs-provenance-attestation-v1.schema.json";
    }
    if name.starts_with("fsfs-root-discovery-") {
        return "fsfs-root-discovery-v1.schema.json";
    }
    if name.starts_with("fsfs-scope-privacy-") {
        return "fsfs-scope-privacy-v1.schema.json";
    }
    if name.starts_with("fsfs-snippet-highlight-provenance-") {
        return "fsfs-snippet-highlight-provenance-v1.schema.json";
    }
    panic!("No schema mapping for fixture {name}");
}

fn is_semantic_invalid_fixture(name: &str) -> bool {
    matches!(
        name,
        "crate-placement-registry-invalid-duplicate-v1.json"
    )
}

fn load_schema<'a>(
    cache: &'a mut BTreeMap<String, JSONSchema>,
    schemas: &Path,
    schema_file: &str,
) -> &'a JSONSchema {
    cache.entry(schema_file.to_owned()).or_insert_with(|| {
        let schema_path = schemas.join(schema_file);
        let raw = fs::read_to_string(&schema_path).unwrap_or_else(|_| {
            panic!("schema file missing: {}", schema_path.display())
        });
        let schema_json: serde_json::Value =
            serde_json::from_str(&raw).expect("schema should parse as json");
        JSONSchema::options()
            .with_draft(Draft::Draft202012)
            .compile(&schema_json)
            .unwrap_or_else(|error| {
                panic!("failed to compile schema {}: {error}", schema_path.display())
            })
    })
}

fn assert_schema_validation(
    cache: &mut BTreeMap<String, JSONSchema>,
    schema_root: &Path,
    fixture_path: &Path,
    should_pass: bool,
) {
    let file_name = fixture_path
        .file_name()
        .and_then(|name| name.to_str())
        .expect("fixture filename");
    let schema_file = schema_for_fixture(file_name);
    let schema = load_schema(cache, schema_root, schema_file);
    let raw = fs::read_to_string(fixture_path)
        .unwrap_or_else(|_| panic!("read fixture {}", fixture_path.display()));
    let value: serde_json::Value =
        serde_json::from_str(&raw).unwrap_or_else(|error| {
            panic!("fixture {} is invalid json: {error}", fixture_path.display())
        });
    let validation = schema.validate(&value);
    if should_pass {
        if let Err(errors) = validation {
            let messages: Vec<String> = errors.map(|err| err.to_string()).collect();
            panic!(
                "fixture {} failed schema {}: {}",
                fixture_path.display(),
                schema_file,
                messages.join("; ")
            );
        }
    } else if validation.is_ok() {
        panic!(
            "fixture {} unexpectedly passed schema {}",
            fixture_path.display(),
            schema_file
        );
    }
}

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

    let actual_value = serde_json::from_str::<serde_json::Value>(actual_trimmed);
    let expected_value = serde_json::from_str::<serde_json::Value>(expected_trimmed);

    let matches = match (actual_value, expected_value) {
        (Ok(actual), Ok(expected)) => actual == expected,
        _ => actual_trimmed == expected_trimmed,
    };

    if !matches {
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
fn test_schema_fixtures_validate_against_jsonschema() {
    let mut cache: BTreeMap<String, JSONSchema> = BTreeMap::new();
    let schemas = schema_dir();

    for entry in fs::read_dir(fixture_dir()).expect("read fixtures dir") {
        let entry = entry.expect("fixture dir entry");
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        assert_schema_validation(&mut cache, &schemas, &path, true);
    }

    for entry in fs::read_dir(invalid_fixture_dir()).expect("read invalid fixtures dir") {
        let entry = entry.expect("invalid fixture dir entry");
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .expect("invalid fixture filename");
        if is_semantic_invalid_fixture(file_name) {
            continue;
        }
        assert_schema_validation(&mut cache, &schemas, &path, false);
    }
}

#[test]
fn test_fsfs_config_roundtrip_conformance() {
    let path = fixture_dir().join("fsfs-config-contract-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: serde_json::Value = serde_json::from_str(&raw).expect("parse config contract");
    assert_golden_json("fsfs_config_roundtrip_v1", &parsed);
}

#[test]
fn test_fsfs_config_effective_conformance() {
    let path = fixture_dir().join("fsfs-config-effective-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: serde_json::Value =
        serde_json::from_str(&raw).expect("parse config effective");
    assert_golden_json("fsfs_config_effective_roundtrip_v1", &parsed);
}

#[test]
fn test_fsfs_config_load_event_conformance() {
    let path = fixture_dir().join("fsfs-config-load-event-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: serde_json::Value =
        serde_json::from_str(&raw).expect("parse config load event");
    assert_golden_json("fsfs_config_loaded_event_roundtrip_v1", &parsed);
}

#[test]
fn test_file_classification_contract_conformance() {
    let path = fixture_dir().join("fsfs-file-classification-contract-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::file_classification::FileClassificationContractDefinition = serde_json::from_str(&raw).expect("parse file classification contract");
    assert_golden_json("fsfs_file_classification_contract_roundtrip_v1", &parsed);
}

#[test]
fn test_file_classification_decision_conformance() {
    let path = fixture_dir().join("fsfs-file-classification-decision-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::file_classification::FileClassificationDecision = serde_json::from_str(&raw).expect("parse file classification decision");
    assert_golden_json("fsfs_file_classification_decision_roundtrip_v1", &parsed);
}

#[test]
fn test_file_classification_corrupt_event_conformance() {
    let path = fixture_dir().join("fsfs-file-classification-corrupt-event-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::file_classification::FileClassificationCorruptEvent = serde_json::from_str(&raw).expect("parse file classification corrupt event");
    assert_golden_json("fsfs_file_classification_corrupt_event_roundtrip_v1", &parsed);
}

#[test]
fn test_expected_loss_contract_conformance() {
    let path = fixture_dir().join("fsfs-expected-loss-contract-v1.json");
    let raw = std::fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::expected_loss::ExpectedLossContractDefinition = serde_json::from_str(&raw).expect("parse expected loss contract");
    assert_golden_json("fsfs_expected_loss_contract_roundtrip_v1", &parsed);
}

#[test]
fn test_expected_loss_decision_conformance() {
    let path = fixture_dir().join("fsfs-expected-loss-decision-event-v1.json");
    let raw = std::fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::expected_loss::ExpectedLossDecisionEvent = serde_json::from_str(&raw).expect("parse expected loss decision");
    assert_golden_json("fsfs_expected_loss_decision_roundtrip_v1", &parsed);
}

#[test]
fn test_expected_loss_matrix_conformance() {
    let path = fixture_dir().join("fsfs-expected-loss-matrix-v1.json");
    let raw = std::fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::expected_loss::ExpectedLossMatrix = serde_json::from_str(&raw).expect("parse expected loss matrix");
    assert_golden_json("fsfs_expected_loss_matrix_roundtrip_v1", &parsed);
}

#[test]
fn test_high_cost_artifact_contract_conformance() {
    let path = fixture_dir().join("fsfs-high-cost-artifact-detectors-contract-v1.json");
    let raw = std::fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::high_cost_artifact::HighCostArtifactContractDefinition = serde_json::from_str(&raw).expect("parse high cost artifact contract");
    assert_golden_json("fsfs_high_cost_artifact_contract_roundtrip_v1", &parsed);
}

#[test]
fn test_high_cost_artifact_decision_conformance() {
    let path = fixture_dir().join("fsfs-high-cost-artifact-detectors-decision-v1.json");
    let raw = std::fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::high_cost_artifact::HighCostArtifactDecision = serde_json::from_str(&raw).expect("parse high cost artifact decision");
    assert_golden_json("fsfs_high_cost_artifact_decision_roundtrip_v1", &parsed);
}

#[test]
fn test_high_cost_artifact_override_conformance() {
    let path = fixture_dir().join("fsfs-high-cost-artifact-detectors-override-v1.json");
    let raw = std::fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::high_cost_artifact::HighCostOverrideEvent = serde_json::from_str(&raw).expect("parse high cost artifact override");
    assert_golden_json("fsfs_high_cost_artifact_override_roundtrip_v1", &parsed);
}

#[test]
fn test_root_discovery_contract_conformance() {
    let path = fixture_dir().join("fsfs-root-discovery-contract-v1.json");
    let raw = std::fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::root_discovery::RootDiscoveryContractDefinition = serde_json::from_str(&raw).expect("parse root discovery contract");
    assert_golden_json("fsfs_root_discovery_contract_roundtrip_v1", &parsed);
}

#[test]
fn test_root_discovery_decision_conformance() {
    let path = fixture_dir().join("fsfs-root-discovery-decision-v1.json");
    let raw = std::fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::root_discovery::RootDiscoveryDecision = serde_json::from_str(&raw).expect("parse root discovery decision");
    assert_golden_json("fsfs_root_discovery_decision_roundtrip_v1", &parsed);
}

#[test]
fn test_root_discovery_guard_event_conformance() {
    let path = fixture_dir().join("fsfs-root-discovery-guard-event-v1.json");
    let raw = std::fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::root_discovery::RootTraversalGuardEvent = serde_json::from_str(&raw).expect("parse root discovery guard event");
    assert_golden_json("fsfs_root_discovery_guard_event_roundtrip_v1", &parsed);
}

#[test]
fn test_scope_privacy_contract_conformance() {
    let path = fixture_dir().join("fsfs-scope-privacy-contract-v1.json");
    let raw = std::fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::privacy::ScopePrivacyContractDefinition = serde_json::from_str(&raw).expect("parse scope privacy contract");
    assert_golden_json("fsfs_scope_privacy_contract_roundtrip_v1", &parsed);
}

#[test]
fn test_scope_redacted_artifact_conformance() {
    let path = fixture_dir().join("fsfs-scope-privacy-redacted-artifact-v1.json");
    let raw = std::fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::privacy::RedactedArtifact = serde_json::from_str(&raw).expect("parse scope redacted artifact");
    assert_golden_json("fsfs_scope_redacted_artifact_roundtrip_v1", &parsed);
}

#[test]
fn test_scope_scan_decision_conformance() {
    let path = fixture_dir().join("fsfs-scope-privacy-scan-decision-v1.json");
    let raw = std::fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::privacy::ScopeScanDecision = serde_json::from_str(&raw).expect("parse scope scan decision");
    assert_golden_json("fsfs_scope_scan_decision_roundtrip_v1", &parsed);
}

#[test]
fn test_snippet_highlight_provenance_contract_conformance() {
    let path = fixture_dir().join("fsfs-snippet-highlight-provenance-contract-v1.json");
    let raw = std::fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::snippet_provenance::SnippetHighlightProvenanceContractDefinition = serde_json::from_str(&raw).expect("parse snippet highlight provenance contract");
    assert_golden_json("fsfs_snippet_highlight_provenance_contract_roundtrip_v1", &parsed);
}

#[test]
fn test_snippet_highlight_provenance_decision_conformance() {
    let path = fixture_dir().join("fsfs-snippet-highlight-provenance-decision-v1.json");
    let raw = std::fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::snippet_provenance::SnippetHighlightProvenanceDecision = serde_json::from_str(&raw).expect("parse snippet highlight provenance decision");
    assert_golden_json("fsfs_snippet_highlight_provenance_decision_roundtrip_v1", &parsed);
}

#[test]
fn test_provenance_contract_conformance() {
    let path = fixture_dir().join("fsfs-provenance-attestation-contract-v1.json");
    let raw = std::fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::provenance::ProvenanceContractDefinition = serde_json::from_str(&raw).expect("parse provenance contract");
    assert_golden_json("fsfs_provenance_contract_roundtrip_v1", &parsed);
}

#[test]
fn test_provenance_attestation_manifest_conformance() {
    let path = fixture_dir().join("fsfs-provenance-attestation-manifest-v1.json");
    let raw = std::fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::provenance::ProvenanceAttestationManifest = serde_json::from_str(&raw).expect("parse provenance attestation manifest");
    assert_golden_json("fsfs_provenance_attestation_manifest_roundtrip_v1", &parsed);
}

#[test]
fn test_provenance_startup_check_conformance() {
    let path = fixture_dir().join("fsfs-provenance-attestation-startup-check-v1.json");
    let raw = std::fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::provenance::ProvenanceStartupCheck = serde_json::from_str(&raw).expect("parse provenance startup check");
    assert_golden_json("fsfs_provenance_startup_check_roundtrip_v1", &parsed);
}

#[test]
fn test_packaging_contract_conformance() {
    let path = fixture_dir().join("fsfs-packaging-release-install-contract-v1.json");
    let raw = std::fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::packaging::PackagingContractDefinition = serde_json::from_str(&raw).expect("parse packaging contract");
    assert_golden_json("fsfs_packaging_contract_roundtrip_v1", &parsed);
}

#[test]
fn test_release_manifest_conformance() {
    let path = fixture_dir().join("fsfs-packaging-release-install-release-manifest-v1.json");
    let raw = std::fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::packaging::ReleaseManifest = serde_json::from_str(&raw).expect("parse release manifest");
    assert_golden_json("fsfs_release_manifest_roundtrip_v1", &parsed);
}

#[test]
fn test_upgrade_plan_conformance() {
    let path = fixture_dir().join("fsfs-packaging-release-install-upgrade-plan-v1.json");
    let raw = std::fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::packaging::UpgradePlan = serde_json::from_str(&raw).expect("parse upgrade plan");
    assert_golden_json("fsfs_upgrade_plan_roundtrip_v1", &parsed);
}

#[test]
fn test_incremental_change_contract_conformance() {
    let path = fixture_dir().join("fsfs-incremental-change-detection-contract-v1.json");
    let raw = std::fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::incremental_change::IncrementalChangeDetectionContractDefinition = serde_json::from_str(&raw).expect("parse incremental change contract");
    assert_golden_json("fsfs_incremental_change_contract_roundtrip_v1", &parsed);
}

#[test]
fn test_incremental_change_decision_conformance() {
    let path = fixture_dir().join("fsfs-incremental-change-detection-decision-v1.json");
    let raw = std::fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::incremental_change::IncrementalChangeDecision = serde_json::from_str(&raw).expect("parse incremental change decision");
    assert_golden_json("fsfs_incremental_change_decision_roundtrip_v1", &parsed);
}

#[test]
fn test_incremental_recovery_checkpoint_conformance() {
    let path = fixture_dir().join("fsfs-incremental-change-detection-recovery-v1.json");
    let raw = std::fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::incremental_change::IncrementalRecoveryCheckpoint = serde_json::from_str(&raw).expect("parse incremental change recovery");
    assert_golden_json("fsfs_incremental_recovery_checkpoint_roundtrip_v1", &parsed);
}

#[test]
fn test_pressure_profiles_roundtrip_conformance() {
    let path = fixture_dir().join("fsfs-pressure-profiles-contract-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    
    let parsed: serde_json::Value = serde_json::from_str(&raw).expect("parse pressure-profiles fixture");
    assert_golden_json("fsfs_pressure_profiles_roundtrip_v1", &parsed);
}

#[test]
fn test_pressure_profiles_decision_conformance() {
    let path = fixture_dir().join("fsfs-pressure-profiles-decision-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: serde_json::Value =
        serde_json::from_str(&raw).expect("parse pressure profiles decision fixture");
    assert_golden_json("fsfs_pressure_profiles_decision_roundtrip_v1", &parsed);
}

#[test]
fn test_alien_recommendation_bundle_conformance() {
    let path = fixture_dir().join("fsfs-alien-recommendation-bundle-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::alien_recommendations::RecommendationBundle =
        serde_json::from_str(&raw).expect("parse alien recommendation bundle");
    assert_golden_json("fsfs_alien_recommendation_bundle_roundtrip_v1", &parsed);
}

#[test]
fn test_alien_recommendation_card_ingestion_conformance() {
    let path = fixture_dir().join("fsfs-alien-recommendation-card-ingestion-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::alien_recommendations::RecommendationCard =
        serde_json::from_str(&raw).expect("parse alien recommendation card ingestion");
    assert_golden_json("fsfs_alien_recommendation_card_ingestion_roundtrip_v1", &parsed);
}

#[test]
fn test_determinism_contract_conformance() {
    let path = fixture_dir().join("fsfs-determinism-contract-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::determinism::DeterminismContractDefinition = serde_json::from_str(&raw).expect("parse determinism contract");
    assert_golden_json("fsfs_determinism_contract_roundtrip_v1", &parsed);
}

#[test]
fn test_determinism_manifest_conformance() {
    let path = fixture_dir().join("fsfs-determinism-manifest-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::determinism::ReproManifest = serde_json::from_str(&raw).expect("parse determinism manifest");
    assert_golden_json("fsfs_determinism_manifest_roundtrip_v1", &parsed);
}

#[test]
fn test_determinism_check_result_conformance() {
    let path = fixture_dir().join("fsfs-determinism-check-result-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::determinism::DeterminismCheckResult =
        serde_json::from_str(&raw).expect("parse determinism check result");
    assert_golden_json("fsfs_determinism_check_result_roundtrip_v1", &parsed);
}

#[test]
fn test_interaction_gate_policy_conformance() {
    let policy = frankensearch_fsfs::interaction_matrix::InteractionGatePolicy {
        schema: "interaction-matrix-gate-policy-v1".to_owned(),
        generated_at: "2026-02-14T00:00:00Z".to_owned(),
        bead: "bd-3un.52.6".to_owned(),
        pass_threshold: "all_required_tests_pass".to_owned(),
        required_tests: vec!["cargo test".to_owned()],
        required_failure_artifacts: vec!["failure.json".to_owned()],
    };
    assert_golden_json("interaction_gate_policy_roundtrip_v1", &policy);
}

#[test]
fn test_interaction_lane_ownership_conformance() {
    let ownership = frankensearch_fsfs::interaction_matrix::InteractionLaneOwnership {
        schema: "interaction-lane-ownership-v1".to_owned(),
        generated_at: "2026-02-14T00:00:00Z".to_owned(),
        bead: "bd-3un.52.6".to_owned(),
        lanes: vec![frankensearch_fsfs::interaction_matrix::LaneOwnership {
            lane_id: "baseline".to_owned(),
            owner_lane: "fusion-composition-owner".to_owned(),
            bead_refs: vec!["bd-3un.52".to_owned()],
            escalation: "composition-owner".to_owned(),
        }],
    };
    assert_golden_json("interaction_lane_ownership_roundtrip_v1", &ownership);
}

#[test]
fn test_composition_matrix_gate_summary_conformance() {
    let summary = frankensearch_fsfs::interaction_matrix::CompositionMatrixGateSummary {
        schema: "composition-matrix-gate-summary-v1".to_owned(),
        generated_at: "2026-02-14T00:00:00Z".to_owned(),
        bead: "bd-1pkl".to_owned(),
        matrix_anchor: "bd-3un.52".to_owned(),
        required_fields: vec!["MATRIX_LINK".to_owned()],
        fallback_contract: "ON_EXHAUSTION".to_owned(),
        required_interaction_tests: vec!["interaction_unit".to_owned()],
        ownership_artifact: "interaction_lane_ownership.json".to_owned(),
    };
    assert_golden_json("composition_matrix_gate_summary_roundtrip_v1", &summary);
}

#[test]
fn test_bead_self_doc_inventory_conformance() {
    let path = fixture_dir().join("../../docs/bead-self-documentation-debt-inventory-2026-02-14.json");
    let raw = std::fs::read_to_string(&path).expect("read inventory fixture");
    let parsed: frankensearch_fsfs::bead_self_doc::SelfDocInventory = serde_json::from_str(&raw).expect("parse self doc inventory");
    assert_golden_json("bead_self_doc_inventory_roundtrip_v1", &parsed);
}

#[test]
fn test_interaction_failure_summary_conformance() {
    let summary = frankensearch_fsfs::interaction_matrix::InteractionFailureSummary {
        schema: "interaction-failure-summary-v1".to_owned(),
        generated_at: "2026-02-14T00:00:00Z".to_owned(),
        bead: "bd-3un.52.6".to_owned(),
        workflow: "interaction-matrix-gate".to_owned(),
        run_url: "https://github.com/run/1".to_owned(),
        replay_command: "cargo test".to_owned(),
        required_artifacts: vec!["log.txt".to_owned()],
        escalation_playbook: "https://docs/playbook".to_owned(),
        escalation_metadata: frankensearch_fsfs::interaction_matrix::EscalationMetadata {
            thread_id: "bd-3un.52.6".to_owned(),
            ownership_artifact: "interaction_lane_ownership.json".to_owned(),
            summary_contract: "lane_id".to_owned(),
        },
    };
    assert_golden_json("interaction_failure_summary_roundtrip_v1", &summary);
}

#[test]
fn test_control_plane_snapshot_conformance() {
    let path = fixture_dir().join("control-plane-snapshot-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::control_plane::SnapshotResponse =
        serde_json::from_str(&raw).expect("parse control plane snapshot");
    assert_golden_json("control_plane_snapshot_roundtrip_v1", &parsed);
}

#[test]
fn test_control_plane_stream_subscribe_conformance() {
    let path = fixture_dir().join("control-plane-stream-subscribe-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::control_plane::StreamSubscribe =
        serde_json::from_str(&raw).expect("parse control plane stream subscribe");
    assert_golden_json("control_plane_stream_subscribe_roundtrip_v1", &parsed);
}

#[test]
fn test_control_plane_stream_frame_event_conformance() {
    let path = fixture_dir().join("control-plane-stream-event-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::control_plane::StreamFrame =
        serde_json::from_str(&raw).expect("parse control plane stream frame event");
    assert_golden_json("control_plane_stream_frame_event_roundtrip_v1", &parsed);
}

#[test]
fn test_control_plane_stream_frame_control_backpressure_conformance() {
    let path = fixture_dir().join("control-plane-stream-control-backpressure-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::control_plane::StreamFrame =
        serde_json::from_str(&raw).expect("parse control plane stream frame backpressure");
    assert_golden_json("control_plane_stream_frame_backpressure_roundtrip_v1", &parsed);
}

#[test]
fn test_control_plane_stream_frame_control_reconnect_conformance() {
    let path = fixture_dir().join("control-plane-stream-control-reconnect-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::control_plane::StreamFrame =
        serde_json::from_str(&raw).expect("parse control plane stream frame reconnect");
    assert_golden_json("control_plane_stream_frame_reconnect_roundtrip_v1", &parsed);
}

#[test]
fn test_control_plane_stream_frame_control_sampling_conformance() {
    let path = fixture_dir().join("control-plane-stream-control-sampling-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::control_plane::StreamFrame =
        serde_json::from_str(&raw).expect("parse control plane stream frame sampling");
    assert_golden_json("control_plane_stream_frame_sampling_roundtrip_v1", &parsed);
}

#[test]
fn test_control_plane_stream_frame_control_topology_change_conformance() {
    let path = fixture_dir().join("control-plane-stream-control-topology-change-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::control_plane::StreamFrame =
        serde_json::from_str(&raw).expect("parse control plane stream frame topology change");
    assert_golden_json(
        "control_plane_stream_frame_topology_change_roundtrip_v1",
        &parsed,
    );
}

#[test]
fn test_control_plane_stream_frame_heartbeat_conformance() {
    let path = fixture_dir().join("control-plane-stream-heartbeat-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::control_plane::StreamFrame =
        serde_json::from_str(&raw).expect("parse control plane stream frame heartbeat");
    assert_golden_json("control_plane_stream_frame_heartbeat_roundtrip_v1", &parsed);
}

#[test]
fn test_control_plane_stream_frame_error_conformance() {
    let path = fixture_dir().join("control-plane-stream-error-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::control_plane::StreamFrame =
        serde_json::from_str(&raw).expect("parse control plane stream frame error");
    assert_golden_json("control_plane_stream_frame_error_roundtrip_v1", &parsed);
}

#[test]
fn test_control_plane_error_catalog_conformance() {
    let path = fixture_dir().join("control-plane-error-catalog-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::control_plane_error::ErrorCatalog =
        serde_json::from_str(&raw).expect("parse control plane error catalog");
    assert_golden_json("control_plane_error_catalog_roundtrip_v1", &parsed);
}

#[test]
fn test_control_plane_error_event_conformance() {
    let path = fixture_dir().join("control-plane-error-event-stream-disconnected-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::control_plane_error::ControlPlaneErrorEvent =
        serde_json::from_str(&raw).expect("parse control plane error event");
    assert_golden_json("control_plane_error_event_roundtrip_v1", &parsed);
}

#[test]
fn test_control_plane_error_aggregation_conformance() {
    let path = fixture_dir().join("control-plane-error-aggregation-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::control_plane_error::ErrorAggregation =
        serde_json::from_str(&raw).expect("parse control plane error aggregation");
    assert_golden_json("control_plane_error_aggregation_roundtrip_v1", &parsed);
}

#[test]
fn test_crate_placement_registry_conformance() {
    let path = fixture_dir().join("crate-placement-registry-v1.json");
    let raw = std::fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::crate_registry::CratePlacementRegistry = serde_json::from_str(&raw).expect("parse crate placement registry");
    assert_golden_json("crate_placement_registry_roundtrip_v1", &parsed);
}

#[test]
fn test_slo_contract_definition_conformance() {
    let path = fixture_dir().join("slo-anomaly-contract-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::slo_anomaly::SloContractDefinition =
        serde_json::from_str(&raw).expect("parse slo contract definition");
    assert_golden_json("slo_contract_definition_roundtrip_v1", &parsed);
}

#[test]
fn test_slo_anomaly_event_conformance() {
    let path = fixture_dir().join("slo-anomaly-event-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::slo_anomaly::SloAnomalyEvent =
        serde_json::from_str(&raw).expect("parse slo anomaly event");
    assert_golden_json("slo_anomaly_event_roundtrip_v1", &parsed);
}

#[test]
fn test_telemetry_search_conformance() {
    let path = fixture_dir().join("telemetry-search-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::telemetry::TelemetryEnvelope =
        serde_json::from_str(&raw).expect("parse telemetry search");
    assert_golden_json("telemetry_search_roundtrip_v1", &parsed);
}

#[test]
fn test_telemetry_embedding_conformance() {
    let path = fixture_dir().join("telemetry-embedding-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::telemetry::TelemetryEnvelope =
        serde_json::from_str(&raw).expect("parse telemetry embedding");
    assert_golden_json("telemetry_embedding_roundtrip_v1", &parsed);
}

#[test]
fn test_telemetry_index_conformance() {
    let path = fixture_dir().join("telemetry-index-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::telemetry::TelemetryEnvelope =
        serde_json::from_str(&raw).expect("parse telemetry index");
    assert_golden_json("telemetry_index_roundtrip_v1", &parsed);
}

#[test]
fn test_telemetry_resource_conformance() {
    let path = fixture_dir().join("telemetry-resource-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::telemetry::TelemetryEnvelope =
        serde_json::from_str(&raw).expect("parse telemetry resource");
    assert_golden_json("telemetry_resource_roundtrip_v1", &parsed);
}

#[test]
fn test_telemetry_lifecycle_conformance() {
    let path = fixture_dir().join("telemetry-lifecycle-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::telemetry::TelemetryEnvelope =
        serde_json::from_str(&raw).expect("parse telemetry lifecycle");
    assert_golden_json("telemetry_lifecycle_roundtrip_v1", &parsed);
}

#[test]
fn test_telemetry_transport_contract_conformance() {
    let path = fixture_dir().join("telemetry-transport-contract-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::telemetry_transport::TelemetryTransportContractDefinition =
        serde_json::from_str(&raw).expect("parse telemetry transport contract");
    assert_golden_json("telemetry_transport_contract_roundtrip_v1", &parsed);
}

#[test]
fn test_telemetry_transport_endpoint_conformance() {
    let path = fixture_dir().join("telemetry-transport-endpoint-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::telemetry_transport::TransportEndpoint =
        serde_json::from_str(&raw).expect("parse telemetry transport endpoint");
    assert_golden_json("telemetry_transport_endpoint_roundtrip_v1", &parsed);
}

#[test]
fn test_telemetry_transport_subscribe_conformance() {
    let path = fixture_dir().join("telemetry-transport-subscribe-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::telemetry_transport::SubscribeFrame =
        serde_json::from_str(&raw).expect("parse telemetry transport subscribe");
    assert_golden_json("telemetry_transport_subscribe_roundtrip_v1", &parsed);
}

#[test]
fn test_telemetry_transport_frame_event_conformance() {
    let path = fixture_dir().join("telemetry-transport-frame-event-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::telemetry_transport::TransportStreamFrame =
        serde_json::from_str(&raw).expect("parse telemetry transport frame event");
    assert_golden_json("telemetry_transport_frame_event_roundtrip_v1", &parsed);
}

#[test]
fn test_telemetry_transport_frame_control_backpressure_conformance() {
    let path = fixture_dir().join("telemetry-transport-frame-control-backpressure-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::telemetry_transport::TransportStreamFrame =
        serde_json::from_str(&raw).expect("parse telemetry transport frame backpressure");
    assert_golden_json("telemetry_transport_frame_backpressure_roundtrip_v1", &parsed);
}

#[test]
fn test_telemetry_transport_frame_control_reconnect_conformance() {
    let path = fixture_dir().join("telemetry-transport-frame-control-reconnect-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::telemetry_transport::TransportStreamFrame =
        serde_json::from_str(&raw).expect("parse telemetry transport frame reconnect");
    assert_golden_json("telemetry_transport_frame_reconnect_roundtrip_v1", &parsed);
}

#[test]
fn test_telemetry_transport_frame_error_conformance() {
    let path = fixture_dir().join("telemetry-transport-frame-error-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::telemetry_transport::TransportStreamFrame =
        serde_json::from_str(&raw).expect("parse telemetry transport frame error");
    assert_golden_json("telemetry_transport_frame_error_roundtrip_v1", &parsed);
}

#[test]
fn test_telemetry_transport_frame_heartbeat_conformance() {
    let path = fixture_dir().join("telemetry-transport-frame-heartbeat-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::telemetry_transport::TransportStreamFrame =
        serde_json::from_str(&raw).expect("parse telemetry transport frame heartbeat");
    assert_golden_json("telemetry_transport_frame_heartbeat_roundtrip_v1", &parsed);
}

#[test]
fn test_evidence_alert_conformance() {
    let path = fixture_dir().join("evidence-alert-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_tui::evidence::EvidenceEnvelope =
        serde_json::from_str(&raw).expect("parse evidence alert");
    assert_golden_json("evidence_alert_roundtrip_v1", &parsed);
}

#[test]
fn test_evidence_decision_conformance() {
    let path = fixture_dir().join("evidence-decision-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_tui::evidence::EvidenceEnvelope =
        serde_json::from_str(&raw).expect("parse evidence decision");
    assert_golden_json("evidence_decision_roundtrip_v1", &parsed);
}

#[test]
fn test_evidence_degradation_conformance() {
    let path = fixture_dir().join("evidence-degradation-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_tui::evidence::EvidenceEnvelope =
        serde_json::from_str(&raw).expect("parse evidence degradation");
    assert_golden_json("evidence_degradation_roundtrip_v1", &parsed);
}

#[test]
fn test_asupersync_cx_contract_conformance() {
    let path = fixture_dir().join("asupersync-cx-contract-definition-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::asupersync_cx::AsupersyncCxContractDefinition =
        serde_json::from_str(&raw).expect("parse asupersync cx contract");
    assert_golden_json("asupersync_cx_contract_roundtrip_v1", &parsed);
}

#[test]
fn test_asupersync_cx_api_signature_async_conformance() {
    let path = fixture_dir().join("asupersync-cx-api-async-valid-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::asupersync_cx::ApiSignatureCase =
        serde_json::from_str(&raw).expect("parse asupersync cx api async");
    assert_golden_json("asupersync_cx_api_async_roundtrip_v1", &parsed);
}

#[test]
fn test_asupersync_cx_api_signature_sync_conformance() {
    let path = fixture_dir().join("asupersync-cx-api-sync-valid-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::asupersync_cx::ApiSignatureCase =
        serde_json::from_str(&raw).expect("parse asupersync cx api sync");
    assert_golden_json("asupersync_cx_api_sync_roundtrip_v1", &parsed);
}

#[test]
fn test_asupersync_cx_labruntime_result_conformance() {
    let path = fixture_dir().join("asupersync-cx-labruntime-result-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_fsfs::asupersync_cx::LabRuntimeResult =
        serde_json::from_str(&raw).expect("parse asupersync cx labruntime result");
    assert_golden_json("asupersync_cx_labruntime_result_roundtrip_v1", &parsed);
}

#[test]
fn test_fsfs_explanation_payload_contract_conformance() {
    let path = fixture_dir().join("fsfs-explanation-payload-contract-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: serde_json::Value =
        serde_json::from_str(&raw).expect("parse fsfs explanation payload contract");
    assert_golden_json("fsfs_explanation_payload_contract_roundtrip_v1", &parsed);
}

#[test]
fn test_fsfs_explanation_payload_decision_conformance() {
    let path = fixture_dir().join("fsfs-explanation-payload-decision-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: serde_json::Value =
        serde_json::from_str(&raw).expect("parse fsfs explanation payload decision");
    assert_golden_json("fsfs_explanation_payload_decision_roundtrip_v1", &parsed);
}

#[test]
fn test_e2e_manifest_conformance() {
    let path = fixture_dir().join("e2e-manifest-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_core::E2eEnvelope<frankensearch_core::ManifestBody> =
        serde_json::from_str(&raw).expect("parse e2e manifest");
    assert_golden_json("e2e_manifest_roundtrip_v1", &parsed);
}

#[test]
fn test_e2e_event_lane_start_conformance() {
    let path = fixture_dir().join("e2e-event-lane-start-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_core::E2eEnvelope<frankensearch_core::EventBody> =
        serde_json::from_str(&raw).expect("parse e2e event lane start");
    assert_golden_json("e2e_event_lane_start_roundtrip_v1", &parsed);
}

#[test]
fn test_e2e_event_oracle_check_conformance() {
    let path = fixture_dir().join("e2e-event-oracle-check-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_core::E2eEnvelope<frankensearch_core::EventBody> =
        serde_json::from_str(&raw).expect("parse e2e event oracle check");
    assert_golden_json("e2e_event_oracle_check_roundtrip_v1", &parsed);
}

#[test]
fn test_e2e_oracle_report_conformance() {
    let path = fixture_dir().join("e2e-oracle-report-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_core::E2eEnvelope<frankensearch_core::OracleReportBody> =
        serde_json::from_str(&raw).expect("parse e2e oracle report");
    assert_golden_json("e2e_oracle_report_roundtrip_v1", &parsed);
}

#[test]
fn test_e2e_replay_query_conformance() {
    let path = fixture_dir().join("e2e-replay-query-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_core::E2eEnvelope<frankensearch_core::ReplayBody> =
        serde_json::from_str(&raw).expect("parse e2e replay query");
    assert_golden_json("e2e_replay_query_roundtrip_v1", &parsed);
}

#[test]
fn test_e2e_snapshot_diff_conformance() {
    let path = fixture_dir().join("e2e-snapshot-diff-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: frankensearch_core::E2eEnvelope<frankensearch_core::SnapshotDiffBody> =
        serde_json::from_str(&raw).expect("parse e2e snapshot diff");
    assert_golden_json("e2e_snapshot_diff_roundtrip_v1", &parsed);
}

#[test]
fn test_ops_config_definition_conformance() {
    let path = fixture_dir().join("ops-config-definition-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: serde_json::Value =
        serde_json::from_str(&raw).expect("parse ops config definition");
    assert_golden_json("ops_config_definition_roundtrip_v1", &parsed);
}

#[test]
fn test_ops_config_effective_conformance() {
    let path = fixture_dir().join("ops-config-effective-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: serde_json::Value =
        serde_json::from_str(&raw).expect("parse ops config effective");
    assert_golden_json("ops_config_effective_roundtrip_v1", &parsed);
}

#[test]
fn test_ops_telemetry_storage_conformance() {
    let path = fixture_dir().join("ops-telemetry-storage-v1.json");
    let raw = fs::read_to_string(&path).expect("read fixture");
    let parsed: serde_json::Value =
        serde_json::from_str(&raw).expect("parse ops telemetry storage");
    assert_golden_json("ops_telemetry_storage_roundtrip_v1", &parsed);
}
