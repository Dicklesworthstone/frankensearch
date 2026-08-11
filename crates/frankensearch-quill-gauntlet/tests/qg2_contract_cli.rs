#![forbid(unsafe_code)]

use std::fs;
use std::process::{Command, Output};

use frankensearch_quill_gauntlet::{
    PerfGate, PerfGateArtifact, QG2_CANONICAL_CONTRACT, QG2_CONTRACT_REPORT_SCHEMA_VERSION,
    QG2_MANIFEST_BLOCK_POST_REGION, QG2_MANIFEST_BLOCK_PRE_REGION, QG2_NO_CLAIM,
    QG2_PREFLIGHT_REPORT_SCHEMA_VERSION, QG6_QUERY_GROUPS, perf_manifest_contract_sha256,
};

/// A manifest carrying every normative gate, with the given protected block
/// verbatim in the QG-2 position, so the fresh-process fixture satisfies the
/// same topology the live consumer requires.
fn manifest_with_qg2_block(block: &str) -> String {
    use std::fmt::Write as _;

    let mut manifest = String::new();
    for gate in PerfGate::ALL {
        if gate == PerfGate::Qg2 {
            manifest.push_str(block);
            continue;
        }
        let label = gate.label();
        let _ = write!(
            &mut manifest,
            "[gate.{label}]\nname = \"{label} gate\"\nfixture = \"{label} fixture\"\n\
             target = \"{label} target\"\n"
        );
        if gate == PerfGate::Qg6 {
            let _ = writeln!(&mut manifest, "queries_per_class = {QG6_QUERY_GROUPS}");
        }
        manifest.push_str("activated = false\n\n");
    }
    manifest
}
use serde_json::{Value, json};
use tempfile::TempDir;

const SUCCESS_GOLDEN: &[u8] = include_bytes!("../fixtures/qg2-contract-success-v1.json");
const INVOCATION_ERROR_GOLDEN: &str = concat!(
    "{\n",
    "  \"schema_version\": \"frankensearch.quill-qg2-contract-report.v1\",\n",
    "  \"status\": \"invocation_error\",\n",
    "  \"code\": \"qg2.cli.invalid_invocation\",\n",
    "  \"expected\": \"usage: quill-qg2-contract --repo-root <path> [--mode applied|bootstrap-preflight]\",\n",
    "  \"observed\": \"missing --repo-root\",\n",
    "  \"retry\": \"Invoke the validator with exactly one --repo-root <path> pair.\"\n",
    "}\n",
);

fn invoke(arguments: &[&std::ffi::OsStr]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_quill-qg2-contract"))
        .args(arguments)
        .output()
        .expect("fresh quill-qg2-contract process must start")
}

/// Plan document carrying both declared locators of logical surface 2, in
/// declared document order: the QG-2 gate row and the Method law-1 clause.
fn plan_document() -> String {
    format!(
        "| **QG-2 Bulk indexing, single-thread** | {QG2_CANONICAL_CONTRACT} |\n\
         | **QG-3 Watch-mode incremental** | next |\n\
         \n\
         Method: the five standing laws \u{2014} (1) no benchmark-only semantics. \
         {QG2_CANONICAL_CONTRACT}\n\
         \n\
         ## 15. The Conformance Gauntlet (Bet Q5)\n"
    )
}

/// Hyperopt document carrying campaign law 7 and the W2 commit-path fsync row.
///
/// Law 7's bounded region is byte-identical to the single-locator fixture it
/// replaces, so adding the W2 row cannot perturb law 7's receipt hash.
fn hyperopt_document() -> String {
    format!(
        "7. **QG-2 comparator scope and platform durability.** {QG2_CANONICAL_CONTRACT}\n\
         ## 2. Hardware/profile matrix\n\
         \n\
         ### W2 \u{2014} Bulk-index single-thread cost (QG-1 and QG-2)\n\
         \n\
         | Commit-path fsync count | batch directory syncs | {QG2_CANONICAL_CONTRACT} |\n\
         \n\
         ### W3 \u{2014} Parallel scale-out\n"
    )
}

fn complete_contract_fixture() -> TempDir {
    let fixture = tempfile::tempdir().expect("temporary repository");
    let root = fixture.path();
    fs::create_dir_all(root.join("docs/contracts")).expect("contract directory");
    fs::create_dir_all(root.join(".beads")).expect("tracker directory");
    fs::create_dir_all(root.join(".bench-history")).expect("history directory");

    fs::write(
        root.join("docs/contracts/quill-perf-gates.md"),
        format!(
            "# Laws\n1. **No benchmark-only semantics; comparator scope is explicit.** {QG2_CANONICAL_CONTRACT}\n2. **Distributions, not averages.** next\n"
        ),
    )
    .expect("performance gate fixture");
    fs::write(
        root.join("COMPREHENSIVE_PLAN_FOR_THE_QUILL_LEXICAL_ENGINE.md"),
        plan_document(),
    )
    .expect("plan fixture");
    fs::write(
        root.join("docs/contracts/quill-hyperopt-campaign.md"),
        hyperopt_document(),
    )
    .expect("hyperopt fixture");

    // The protected projected block verbatim, so the fresh-process fixture
    // binds the same bytes the live tree must reach.
    let manifest = manifest_with_qg2_block(QG2_MANIFEST_BLOCK_POST_REGION);
    fs::write(root.join("docs/contracts/quill-perf-gates.toml"), &manifest)
        .expect("manifest fixture");

    let mut tracker = String::new();
    for issue_id in [
        "bd-quill-e8-hyperopt-nyps",
        "bd-quill-e8-perf-doctrine-x4e4.5.5",
        "bd-h6eh",
    ] {
        tracker.push_str(
            &serde_json::to_string(&json!({
                "id": issue_id,
                "notes": QG2_CANONICAL_CONTRACT
            }))
            .expect("tracker issue"),
        );
        tracker.push('\n');
    }
    tracker.push_str(
        &serde_json::to_string(&json!({
            "id": "bd-quill-e8-hyperopt-nyps.1",
            "description": concat!(
                "BINDING SUPERSESSION 2026-07-30: the phrase below saying the QG-2 baseline ",
                "was already admissible is false and retained only as historical plan text. ",
                "The 0.349775 candidate and 0.345546 rerun remain immutable diagnostics.\n\n",
                "Historical body. Integrate the already-admissible QG-2 baseline as ",
                "first-class campaign input."
            )
        }))
        .expect("stale tracker issue"),
    );
    tracker.push('\n');
    fs::write(root.join(".beads/issues.jsonl"), tracker).expect("tracker fixture");

    let manifest_sha256 = perf_manifest_contract_sha256(&manifest);
    let template = serde_json::from_slice::<PerfGateArtifact>(include_bytes!(
        "../../../.bench-history/QG-1.v7.unmeasured.latest.json"
    ))
    .expect("sentinel template");
    for gate in PerfGate::ALL {
        let mut artifact = template.clone();
        artifact.gate = gate;
        artifact.manifest_sha256.clone_from(&manifest_sha256);
        fs::write(
            root.join(format!(
                ".bench-history/{}.unmeasured.latest.json",
                gate.label()
            )),
            serde_json::to_vec_pretty(&artifact).expect("sentinel JSON"),
        )
        .expect("sentinel fixture");
    }
    fixture
}

#[test]
fn fresh_process_success_stdout_matches_golden_and_exits_zero() {
    let fixture = complete_contract_fixture();
    let output = invoke(&[
        std::ffi::OsStr::new("--repo-root"),
        fixture.path().as_os_str(),
    ]);

    assert_eq!(output.status.code(), Some(0));
    assert!(output.stderr.is_empty());
    assert_eq!(
        String::from_utf8(output.stdout).expect("success stdout must be UTF-8"),
        std::str::from_utf8(SUCCESS_GOLDEN).expect("success golden must be UTF-8")
    );
}

#[test]
fn fresh_process_binds_every_protected_locator_from_disk() {
    let fixture = complete_contract_fixture();
    let output = invoke(&[
        std::ffi::OsStr::new("--repo-root"),
        fixture.path().as_os_str(),
    ]);

    assert_eq!(output.status.code(), Some(0));
    let report: Value =
        serde_json::from_slice(&output.stdout).expect("success stdout must be valid JSON");
    let surfaces = report["surfaces"]
        .as_array()
        .expect("report must carry surface receipts");
    assert_eq!(
        surfaces
            .iter()
            .map(|receipt| receipt["locator"].as_str().expect("locator identity"))
            .collect::<Vec<_>>(),
        vec![
            "perf_gate_law_1",
            "comprehensive_plan_qg2_row",
            "comprehensive_plan_method_law_1",
            "perf_manifest_qg2_contract",
            "hyperopt_law_7",
            "hyperopt_w2_fsync_row",
            "hyperopt_epic_active_contract",
            "qg2_r1_active_contract",
            "gate_activation_active_contract",
        ]
    );
    assert!(
        surfaces
            .iter()
            .all(|receipt| receipt["discovered"] == true && receipt["valid"] == true)
    );
    assert_eq!(report["topology"]["expected_physical_locators"], 9);
    assert_eq!(report["topology"]["validated_physical_locators"], 9);
    assert_eq!(report["sentinels"]["validated"], 10);
}

#[test]
fn fresh_process_rejects_an_omitted_w2_fsync_row_locator_with_a_typed_reason() {
    let fixture = complete_contract_fixture();
    let document = hyperopt_document();
    let (head, tail) = document
        .rsplit_once(QG2_CANONICAL_CONTRACT)
        .expect("hyperopt fixture carries the W2 row clause");
    fs::write(
        fixture
            .path()
            .join("docs/contracts/quill-hyperopt-campaign.md"),
        format!("{head}Law 7 on macOS.{tail}"),
    )
    .expect("planted W2 fsync row omission");

    let output = invoke(&[
        std::ffi::OsStr::new("--repo-root"),
        fixture.path().as_os_str(),
    ]);

    assert_eq!(output.status.code(), Some(1));
    assert!(output.stderr.is_empty());
    let report: Value =
        serde_json::from_slice(&output.stdout).expect("divergence stdout must be valid JSON");
    assert_eq!(report["status"], "divergence");
    let divergences = report["divergences"]
        .as_array()
        .expect("report must carry divergences");
    assert!(
        divergences.iter().any(|divergence| {
            divergence["code"] == "qg2.surface.marker_scope"
                && divergence["path"]
                    == "docs/contracts/quill-hyperopt-campaign.md#hyperopt_w2_fsync_row"
                && divergence["retry"]
                    .as_str()
                    .is_some_and(|retry| !retry.is_empty())
        }),
        "{divergences:#?}"
    );
    assert_eq!(report["topology"]["validated_physical_locators"], 7);
}

#[test]
fn fresh_process_contract_divergence_is_structured_and_exits_one() {
    let empty_root = tempfile::tempdir().expect("empty repository fixture");
    let output = invoke(&[
        std::ffi::OsStr::new("--repo-root"),
        empty_root.path().as_os_str(),
    ]);

    assert_eq!(output.status.code(), Some(1));
    assert!(output.stderr.is_empty());
    assert_eq!(output.stdout.last(), Some(&b'\n'));
    let report: Value =
        serde_json::from_slice(&output.stdout).expect("divergence stdout must be valid JSON");
    assert_eq!(report["schema_version"], QG2_CONTRACT_REPORT_SCHEMA_VERSION);
    assert_eq!(report["status"], "divergence");
    assert!(
        report["divergences"]
            .as_array()
            .is_some_and(|divergences| !divergences.is_empty())
    );
}

/// Rewrite the applied contract fixture back to its protected bootstrap form:
/// both renamed law headings revert, every canonical clause disappears, the
/// nested TOML table is removed, and the three tracker notes go absent.
fn revert_to_bootstrap(root: &std::path::Path) -> String {
    fs::write(
        root.join("docs/contracts/quill-perf-gates.md"),
        concat!(
            "# Laws\n",
            "1. **No benchmark-only semantics.** Durability settings and commits match shipped \
             defaults.\n",
            "2. **Distributions, not averages.** next\n",
        ),
    )
    .expect("bootstrap laws");
    fs::write(
        root.join("COMPREHENSIVE_PLAN_FOR_THE_QUILL_LEXICAL_ENGINE.md"),
        concat!(
            "| **QG-2 Bulk indexing, single-thread** | >= 1.5x tantivy |\n",
            "| **QG-3 Watch-mode incremental** | next |\n",
            "\n",
            "Method: the five standing laws \u{2014} (1) no benchmark-only semantics.\n",
            "\n",
            "## 15. The Conformance Gauntlet (Bet Q5)\n",
        ),
    )
    .expect("bootstrap plan");
    fs::write(
        root.join("docs/contracts/quill-hyperopt-campaign.md"),
        concat!(
            "7. **Platform-symmetric durability.** On macOS a commit number needs F_FULLFSYNC.\n",
            "## 2. Hardware/profile matrix\n",
            "\n",
            "### W2 \u{2014} Bulk-index single-thread cost (QG-1 and QG-2)\n",
            "\n",
            "| Commit-path fsync count | batch directory syncs | census first. |\n",
            "\n",
            "### W3 \u{2014} Parallel scale-out\n",
        ),
    )
    .expect("bootstrap hyperopt");

    let manifest = manifest_with_qg2_block(QG2_MANIFEST_BLOCK_PRE_REGION);
    fs::write(root.join("docs/contracts/quill-perf-gates.toml"), &manifest)
        .expect("bootstrap manifest");

    let mut tracker = String::new();
    for issue_id in [
        "bd-quill-e8-hyperopt-nyps",
        "bd-quill-e8-perf-doctrine-x4e4.5.5",
        "bd-h6eh",
    ] {
        tracker
            .push_str(&serde_json::to_string(&json!({ "id": issue_id })).expect("tracker issue"));
        tracker.push('\n');
    }
    fs::write(root.join(".beads/issues.jsonl"), tracker).expect("bootstrap tracker");

    let manifest_sha256 = perf_manifest_contract_sha256(&manifest);
    let template = serde_json::from_slice::<PerfGateArtifact>(include_bytes!(
        "../../../.bench-history/QG-1.v7.unmeasured.latest.json"
    ))
    .expect("sentinel template");
    for gate in PerfGate::ALL {
        let mut artifact = template.clone();
        artifact.gate = gate;
        artifact.manifest_sha256.clone_from(&manifest_sha256);
        fs::write(
            root.join(format!(
                ".bench-history/{}.unmeasured.latest.json",
                gate.label()
            )),
            serde_json::to_vec_pretty(&artifact).expect("sentinel JSON"),
        )
        .expect("bootstrap sentinel");
    }
    manifest_sha256
}

#[test]
fn fresh_process_preflight_admits_the_protected_bootstrap_base() {
    let fixture = complete_contract_fixture();
    let bootstrap_manifest_sha256 = revert_to_bootstrap(fixture.path());

    let output = invoke(&[
        std::ffi::OsStr::new("--repo-root"),
        fixture.path().as_os_str(),
        std::ffi::OsStr::new("--mode"),
        std::ffi::OsStr::new("bootstrap-preflight"),
    ]);

    assert_eq!(output.status.code(), Some(0));
    assert!(output.stderr.is_empty());
    let report: Value =
        serde_json::from_slice(&output.stdout).expect("preflight stdout must be valid JSON");
    assert_eq!(
        report["schema_version"],
        QG2_PREFLIGHT_REPORT_SCHEMA_VERSION
    );
    assert_eq!(report["state"], "bootstrap_ready");
    assert_eq!(report["no_claim"], QG2_NO_CLAIM);
    assert!(
        report["divergences"]
            .as_array()
            .is_some_and(std::vec::Vec::is_empty)
    );
    let selectors = report["selectors"]
        .as_array()
        .expect("preflight must carry selector receipts");
    assert_eq!(selectors.len(), 9);
    assert!(
        selectors
            .iter()
            .all(|receipt| receipt["state"] == "bootstrap")
    );
    assert_eq!(report["manifest_sha256_pre"], bootstrap_manifest_sha256);
    assert_ne!(
        report["manifest_sha256_post"], bootstrap_manifest_sha256,
        "inserting the typed contract must move the normalized manifest digest"
    );
    let rebinds = report["sentinel_rebinds"]
        .as_array()
        .expect("preflight must carry sentinel rebinds");
    assert_eq!(rebinds.len(), 10);
    assert!(
        rebinds
            .iter()
            .all(|rebind| rebind["rebind_required"] == true)
    );
}

#[test]
fn fresh_process_preflight_rejects_an_unexpected_tracker_note_with_a_typed_reason() {
    let fixture = complete_contract_fixture();
    revert_to_bootstrap(fixture.path());
    let mut tracker = String::new();
    for issue_id in [
        "bd-quill-e8-hyperopt-nyps",
        "bd-quill-e8-perf-doctrine-x4e4.5.5",
    ] {
        tracker
            .push_str(&serde_json::to_string(&json!({ "id": issue_id })).expect("tracker issue"));
        tracker.push('\n');
    }
    tracker.push_str(
        &serde_json::to_string(&json!({
            "id": "bd-h6eh",
            "notes": "an unrelated active note that is not the canonical contract"
        }))
        .expect("planted tracker note"),
    );
    tracker.push('\n');
    fs::write(fixture.path().join(".beads/issues.jsonl"), tracker)
        .expect("planted tracker mutation");

    let output = invoke(&[
        std::ffi::OsStr::new("--repo-root"),
        fixture.path().as_os_str(),
        std::ffi::OsStr::new("--mode"),
        std::ffi::OsStr::new("bootstrap-preflight"),
    ]);

    assert_eq!(output.status.code(), Some(1));
    assert!(output.stderr.is_empty());
    let report: Value =
        serde_json::from_slice(&output.stdout).expect("preflight stdout must be valid JSON");
    assert_eq!(report["state"], "drift");
    let divergences = report["divergences"]
        .as_array()
        .expect("report must carry divergences");
    assert!(
        divergences.iter().any(|divergence| {
            divergence["code"] == "qg2.preflight.selector_drift"
                && divergence["path"] == ".beads/issues.jsonl#bd-h6eh.notes"
                && divergence["retry"]
                    .as_str()
                    .is_some_and(|retry| !retry.is_empty())
        }),
        "{divergences:#?}"
    );
}

#[test]
fn fresh_process_preflight_refuses_an_already_applied_tree_as_a_mutation_base() {
    // The applied fixture is a truthful tree, but it is not a base a mutation
    // may consume, so the preflight reports it and exits nonzero.
    let fixture = complete_contract_fixture();
    let output = invoke(&[
        std::ffi::OsStr::new("--repo-root"),
        fixture.path().as_os_str(),
        std::ffi::OsStr::new("--mode"),
        std::ffi::OsStr::new("bootstrap-preflight"),
    ]);

    assert_eq!(output.status.code(), Some(1));
    let report: Value =
        serde_json::from_slice(&output.stdout).expect("preflight stdout must be valid JSON");
    assert_eq!(report["state"], "already_applied");
    assert!(
        report["divergences"]
            .as_array()
            .is_some_and(std::vec::Vec::is_empty),
        "an already-applied tree is refused as a base, not reported as broken"
    );
    assert_eq!(
        report["manifest_sha256_pre"], report["manifest_sha256_post"],
        "an applied tree renders to itself"
    );
}

#[test]
fn fresh_process_invalid_invocation_matches_golden_and_exits_usage() {
    let output = invoke(&[]);

    assert_eq!(output.status.code(), Some(64));
    assert!(output.stderr.is_empty());
    assert_eq!(output.stdout, INVOCATION_ERROR_GOLDEN.as_bytes());
}
