#![forbid(unsafe_code)]

use std::fs;
use std::process::{Command, Output};

use frankensearch_quill_gauntlet::{
    PerfGate, PerfGateArtifact, QG2_CANONICAL_CONTRACT, QG2_CONTRACT_REPORT_SCHEMA_VERSION,
    perf_manifest_contract_sha256,
};
use serde_json::{Value, json};
use tempfile::TempDir;

const SUCCESS_GOLDEN: &[u8] = include_bytes!("../fixtures/qg2-contract-success-v1.json");
const INVOCATION_ERROR_GOLDEN: &str = concat!(
    "{\n",
    "  \"schema_version\": \"frankensearch.quill-qg2-contract-report.v1\",\n",
    "  \"status\": \"invocation_error\",\n",
    "  \"code\": \"qg2.cli.invalid_invocation\",\n",
    "  \"expected\": \"usage: quill-qg2-contract --repo-root <path>\",\n",
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
        format!(
            "| **QG-2 Bulk indexing, single-thread** | {QG2_CANONICAL_CONTRACT} |\n| **QG-3 Watch-mode incremental** | next |\n"
        ),
    )
    .expect("plan fixture");
    fs::write(
        root.join("docs/contracts/quill-hyperopt-campaign.md"),
        format!(
            "7. **QG-2 comparator scope and platform durability.** {QG2_CANONICAL_CONTRACT}\n## 2. Hardware/profile matrix\n"
        ),
    )
    .expect("hyperopt fixture");

    let manifest = format!(
        "[gate.QG-2]\nname = \"bulk indexing, single-thread\"\nactivated = false\n\n[gate.QG-2.qg2_contract]\ncontract = {contract:?}\nstorage_topology = \"symmetric_in_memory\"\ndurability_scope = \"non_durable\"\ntiming_start = \"first_document_feed\"\ntiming_end = \"terminal_searchable_visibility_and_complete_worker_merge_queue_quiescence\"\ncommit_boundary = \"searchable_visibility_not_durable_publication\"\nexcluded_operations = [\"fsync\", \"F_FULLFSYNC\", \"crash_recovery\", \"durable_publication\", \"on_disk_bytes\"]\nsource_nonregression = \"durable_gates_and_production_source_durability_remain_mandatory\"\n",
        contract = QG2_CANONICAL_CONTRACT
    );
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
        "../../../.bench-history/QG-1.unmeasured.latest.json"
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

#[test]
fn fresh_process_invalid_invocation_matches_golden_and_exits_usage() {
    let output = invoke(&[]);

    assert_eq!(output.status.code(), Some(64));
    assert!(output.stderr.is_empty());
    assert_eq!(output.stdout, INVOCATION_ERROR_GOLDEN.as_bytes());
}
