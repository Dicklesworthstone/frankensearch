use super::*;
use frankensearch::native_ann::NativeExactReason;
use frankensearch_core::generation::{
    EmbeddingArtifactIdentityV1, EmbeddingIdentityBundleV1, EmbeddingSpaceKindV1,
    QuantizationFormat,
};
use frankensearch_core::{ModelCategory, SearchFuture, SearchResult};
use frankensearch_index::generation_root::authority_publisher::AttemptStageV1;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Provider {
    identity: EmbeddingIdentityBundleV1,
    calls: AtomicUsize,
}

impl Embedder for Provider {
    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
    }
    fn embed<'a>(&'a self, _: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Box::pin(async move {
            Ok(if text.contains("retry") {
                vec![1.0, 0.0]
            } else {
                vec![0.0, 1.0]
            })
        })
    }
    fn dimension(&self) -> usize {
        2
    }
    fn id(&self) -> &str {
        "authority-cli-test"
    }
    fn model_name(&self) -> &str {
        self.id()
    }
    fn is_semantic(&self) -> bool {
        true
    }
    fn category(&self) -> ModelCategory {
        ModelCategory::TransformerEmbedder
    }
}

fn parse(args: &[&str]) -> Result<Option<Options>> {
    Options::parse(args.iter().map(|argument| (*argument).to_owned()))
}

fn index_arguments() -> Vec<&'static str> {
    vec![
        "authority-index",
        "--root",
        "new-root",
        "--work-dir",
        "new-work",
        "--receipt",
        "new.json",
        "--tier",
        "fast",
        "--source-low",
        "1",
        "--source-high",
        "5",
        "--cooperative-local",
    ]
}

fn rebuild_arguments() -> Vec<&'static str> {
    vec![
        "authority-rebuild",
        "--from-receipt",
        "selected.json",
        "--work-dir",
        "new-work",
        "--receipt",
        "new.json",
        "--source-low",
        "1",
        "--source-high",
        "9",
    ]
}

#[test]
fn authority_serve_options_select_one_warm_receipt_and_explicit_activation_grant() {
    let arguments = ["authority-serve", "--receipt", "selected.json"];
    let options = parse(&arguments).unwrap().unwrap();
    assert_eq!(options.command, Command::Serve);
    assert!(!options.allow_activation);
    assert_eq!(options.limit, 10);
    assert!(is_command("authority-serve"));
    assert!(parse(&["authority-serve", "--help"]).unwrap().is_none());
    let mut configured = arguments.to_vec();
    configured.extend([
        "--allow-activation",
        "--limit",
        "2",
        "--max-image-bytes",
        "8192",
    ]);
    let options = parse(&configured).unwrap().unwrap();
    assert!(options.allow_activation);
    assert_eq!(options.limit, 2);
    assert_eq!(options.max_image_bytes, 8192);
    for extra in [
        vec!["--query", "retry"],
        vec!["--retrieval", "hnsw"],
        vec!["--root", "other-root"],
        vec!["--tier", "quality"],
        vec!["--migrate-model"],
        vec!["--cooperative-local"],
        vec!["--limit", "0"],
        vec!["--limit", "1001"],
    ] {
        let mut rejected = arguments.to_vec();
        rejected.extend(extra);
        assert!(parse(&rejected).is_err(), "{rejected:?}");
    }
    for command in ["authority-search", "authority-recover"] {
        assert!(parse(&[command, "--receipt", "r.json", "--allow-activation"]).is_err());
    }
}

#[test]
fn authority_retrieval_is_explicit_for_new_indexes_and_cannot_override_search() {
    let options = parse(&index_arguments()).unwrap().unwrap();
    assert!(options.retrieval.is_none());
    assert!(matches!(
        options.build_retrieval(None),
        NativeBuildRetrieval::Exact
    ));
    for mut arguments in [index_arguments(), rebuild_arguments()] {
        arguments.extend(["--retrieval", "hnsw"]);
        let options = parse(&arguments).unwrap().unwrap();
        let NativeBuildRetrieval::Hnsw { params, seed } = options.build_retrieval(None) else {
            panic!("explicit HNSW must select a graph");
        };
        assert_eq!(params, HnswParams::default());
        assert_eq!(seed, 42);
        arguments.extend(["--retrieval", "exact"]);
        assert!(parse(&arguments).is_err(), "duplicate retrieval selection");
    }
    for retrieval in ["auto", "ann", "", "EXACT"] {
        let mut arguments = index_arguments();
        arguments.extend(["--retrieval", retrieval]);
        assert!(parse(&arguments).is_err(), "accepted {retrieval:?}");
    }
    for arguments in [
        vec![
            "authority-search",
            "--receipt",
            "selected.json",
            "--query",
            "retry",
            "--retrieval",
            "exact",
        ],
        vec![
            "authority-recover",
            "--receipt",
            "selected.json",
            "--retrieval",
            "hnsw",
        ],
    ] {
        assert!(parse(&arguments).is_err());
    }
}

#[test]
fn rebuild_inherits_full_authenticated_retrieval_policy_unless_explicitly_changed() {
    let inherited_params = HnswParams {
        m: 4,
        m0: 8,
        ef_construction: 24,
        ef_search: 12,
    };
    let inherited = NativeBuildRetrieval::Hnsw {
        params: inherited_params,
        seed: 9876,
    };
    let options = parse(&rebuild_arguments()).unwrap().unwrap();
    let NativeBuildRetrieval::Hnsw { params, seed } = options.build_retrieval(Some(inherited))
    else {
        panic!("a rebuild must preserve its admitted graph policy");
    };
    assert_eq!(params, inherited_params);
    assert_eq!(seed, 9876);
    assert!(matches!(
        options.build_retrieval(Some(NativeBuildRetrieval::Exact)),
        NativeBuildRetrieval::Exact
    ));

    let mut arguments = rebuild_arguments();
    arguments.extend(["--retrieval", "exact"]);
    let options = parse(&arguments).unwrap().unwrap();
    assert!(matches!(
        options.build_retrieval(Some(inherited)),
        NativeBuildRetrieval::Exact
    ));
    *arguments.last_mut().unwrap() = "hnsw";
    let options = parse(&arguments).unwrap().unwrap();
    let NativeBuildRetrieval::Hnsw { params, seed } = options.build_retrieval(Some(inherited))
    else {
        panic!("explicit HNSW must select the default graph policy");
    };
    assert_eq!(params, HnswParams::default());
    assert_eq!(seed, 42);
}

#[test]
fn rebuild_parser_selects_an_existing_receipt_without_overriding_its_root_or_tier() {
    let options = parse(&rebuild_arguments()).unwrap().unwrap();
    assert_eq!(options.command, Command::Rebuild);
    assert_eq!(
        options.from_receipt.as_deref(),
        Some(Path::new("selected.json"))
    );
    assert!(options.root.is_none());
    assert!(options.tier.is_none());
    assert!(!options.migrate_model);
    assert!(is_command("authority-rebuild"));
    assert!(parse(&["authority-rebuild", "--help"]).unwrap().is_none());
    for extra in [
        vec!["--root", "replacement-root"],
        vec!["--tier", "quality"],
        vec!["--cooperative-local"],
        vec!["--query", "retry"],
        vec!["--limit", "1"],
        vec!["--from-receipt", "different.json"],
    ] {
        let mut arguments = rebuild_arguments();
        arguments.extend(extra);
        assert!(parse(&arguments).is_err(), "{arguments:?}");
    }
    for required in [
        "--from-receipt",
        "--work-dir",
        "--receipt",
        "--source-low",
        "--source-high",
    ] {
        let mut arguments = rebuild_arguments();
        let position = arguments
            .iter()
            .position(|argument| *argument == required)
            .unwrap();
        arguments.drain(position..position + 2);
        assert!(parse(&arguments).is_err(), "{required}");
    }
    let mut migration = rebuild_arguments();
    migration.extend([
        "--migrate-model",
        "--quality-backend",
        "native-int8",
        "--quality-model-dir",
        "model",
    ]);
    assert!(parse(&migration).unwrap().unwrap().migrate_model);
    for command in ["authority-search", "authority-recover"] {
        assert!(parse(&[command, "--receipt", "r", "--from-receipt", "old"]).is_err());
        assert!(parse(&[command, "--receipt", "r", "--migrate-model"]).is_err());
    }
}

#[test]
fn rebuild_source_boundaries_are_monotone_and_allow_an_unchanged_cut() {
    let previous = CommitRange { low: 2, high: 7 };
    for source in [
        CommitRange { low: 2, high: 7 },
        CommitRange { low: 2, high: 8 },
        CommitRange { low: 3, high: 9 },
    ] {
        validate_successor_source(&previous, &source).unwrap();
    }
    for source in [
        CommitRange { low: 1, high: 9 },
        CommitRange { low: 2, high: 6 },
        CommitRange { low: 1, high: 6 },
    ] {
        assert!(validate_successor_source(&previous, &source).is_err());
    }
}

#[test]
fn rebuild_requires_explicit_producer_migration_before_any_inference() {
    let selected = receipt();
    let mut provider = Provider {
        identity: identity().identity,
        calls: AtomicUsize::new(0),
    };
    assert_eq!(
        rebuild_identity(&selected, &provider, false).unwrap(),
        selected.identity
    );
    provider.identity.producer.implementation_revision =
        "changed-same-dimension-producer".to_owned();
    provider.identity.validate().unwrap();
    assert!(rebuild_identity(&selected, &provider, false).is_err());
    let migrated = rebuild_identity(&selected, &provider, true).unwrap();
    assert_ne!(migrated, selected.identity);
    assert_eq!(migrated.identity.producer, provider.identity.producer);
    assert_eq!(migrated.identity.space, provider.identity.space);
    assert_eq!(provider.calls.load(Ordering::Relaxed), 0);
    provider.identity.space.dimension = 3;
    assert!(rebuild_identity(&selected, &provider, true).is_err());
    assert_eq!(provider.calls.load(Ordering::Relaxed), 0);
}

#[test]
fn parser_requires_explicit_topology_source_and_security_profile() {
    let options = parse(&index_arguments()).unwrap().unwrap();
    assert_eq!(options.tier, Some(EmbedderTierTag::Fast));
    assert_eq!(
        options.source_cut().unwrap(),
        CommitRange { low: 1, high: 5 }
    );
    let mut absent_profile = index_arguments();
    absent_profile.pop();
    assert!(parse(&absent_profile).is_err());
    for extra in [
        vec!["--quality-backend", "onnx"],
        vec!["--root", "different-root"],
        vec!["--mode", "full"],
        vec!["--shard-size", "5"],
        vec!["--batch-size", "0"],
        vec!["--max-image-bytes", "0"],
    ] {
        let mut arguments = index_arguments();
        arguments.extend(extra);
        assert!(parse(&arguments).is_err(), "{arguments:?}");
    }
    let mut quality = index_arguments();
    let position = quality.iter().position(|value| *value == "fast").unwrap();
    quality[position] = "quality";
    quality.extend([
        "--quality-backend",
        "native-int8",
        "--quality-model-dir",
        "verified-model",
    ]);
    assert_eq!(
        parse(&quality).unwrap().unwrap().tier,
        Some(EmbedderTierTag::Quality)
    );
    let mut invalid_range = index_arguments();
    let low = invalid_range
        .iter()
        .position(|value| *value == "1")
        .unwrap();
    let high = invalid_range
        .iter()
        .position(|value| *value == "5")
        .unwrap();
    invalid_range[low] = "6";
    assert!(parse(&invalid_range).is_err());
    invalid_range[low] = "0";
    invalid_range[high] = "18446744073709551615";
    assert!(
        parse(&invalid_range).is_err(),
        "inclusive-range overflow must fail before indexing"
    );
}

#[test]
fn search_options_do_not_grant_publication_or_change_the_selected_tier() {
    let base = [
        "authority-search",
        "--receipt",
        "trusted.json",
        "--query",
        "retry network",
    ];
    assert_eq!(parse(&base).unwrap().unwrap().command, Command::Search);
    for extra in [
        vec!["--root", "different-root"],
        vec!["--cooperative-local"],
        vec!["--tier", "quality"],
        vec!["--input", "documents.jsonl"],
        vec!["--limit", "1001"],
        vec!["--query", "second query"],
    ] {
        let mut arguments = base.to_vec();
        arguments.extend(extra);
        assert!(parse(&arguments).is_err());
    }
    assert!(parse(&["authority-search", "--help"]).unwrap().is_none());
    assert!(parse(&["authority-search", "--receipt", "r", "--query", "   "]).is_err());
}

#[test]
fn recovery_requires_only_the_saved_attempt_and_rejects_model_or_publish_options() {
    let base = ["authority-recover", "--receipt", "trusted.json"];
    assert_eq!(parse(&base).unwrap().unwrap().command, Command::Recover);
    assert!(is_command("authority-recover"));
    assert!(parse(&["authority-recover"]).is_err());
    assert!(parse(&["authority-recover", "--help"]).unwrap().is_none());
    for extra in [
        vec!["--root", "different-root"],
        vec!["--cooperative-local"],
        vec!["--tier", "quality"],
        vec!["--input", "documents.jsonl"],
        vec!["--model-dir", "unavailable"],
        vec!["--quality-backend", "onnx"],
        vec!["--quality-model-dir", "unavailable"],
        vec!["--query", "retry"],
        vec!["--limit", "10"],
        vec!["--receipt", "different.json"],
        vec!["--max-image-bytes", "0"],
    ] {
        let mut arguments = base.to_vec();
        arguments.extend(extra);
        assert!(parse(&arguments).is_err(), "{arguments:?}");
    }
    let mut arguments = base.to_vec();
    arguments.extend(["--max-image-bytes", "4096"]);
    assert_eq!(parse(&arguments).unwrap().unwrap().max_image_bytes, 4096);
}

fn identity() -> FrozenEmbeddingIdentityBundleV1 {
    let mut identity = EmbeddingIdentityBundleV1::explicit_test_model("authority-cli-test", 2);
    // Synthetic semantic-shaped test provider; no production inference or
    // model-availability claim is made by these filesystem/identity tests.
    identity.space.kind = EmbeddingSpaceKindV1::Semantic;
    identity.space.hash_control = None;
    identity.space.artifact_manifest_fingerprint = "a".repeat(64);
    identity.space.artifacts = vec![EmbeddingArtifactIdentityV1 {
        role: "weights".to_owned(),
        sha256: "b".repeat(64),
        size: 1,
    }];
    identity.producer.space_fingerprint = identity.space.fingerprint();
    identity.storage.format = "fsvi-v2".to_owned();
    identity.storage.quantization = QuantizationFormat::F32;
    identity.storage.endianness = "little-endian".to_owned();
    identity.freeze().unwrap()
}

fn receipt() -> Receipt {
    let authority = AuthorityRefV1::new(1, [0x52; 16], 256, [0x53; 32], None).unwrap();
    let permit = AttemptPermitV1::new(
        [0x51; 16],
        [0x61; 16],
        [0x62; 16],
        authority,
        ExpectedAuthorityPairV1::default(),
        [0x63; 16],
    )
    .unwrap();
    Receipt {
        schema: RECEIPT_SCHEMA.to_owned(),
        root: PathBuf::from("/private/generation"),
        root_id: [0x51; 16],
        profile: PROFILE.as_str().to_owned(),
        authority: authority.canonical_bytes().to_vec(),
        generation: ArtifactGenerationIdentityV1::new(1, [0x54; 16]).unwrap(),
        identity: identity(),
        tier: EmbedderTierTag::Fast,
        documents: 2,
        image_bytes: 4096,
        attempt: Some(SavedAttempt::from_permit(&permit, false)),
    }
}

#[test]
fn receipt_preserves_independent_binding_and_rejects_downgrades_and_tampering() {
    let original = receipt();
    original.validate().unwrap();
    let encoded = serde_json::to_vec(&original).unwrap();
    let decoded: Receipt = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded.binding().unwrap(), original.binding().unwrap());
    assert_eq!(decoded.authority().unwrap(), original.authority().unwrap());
    let mut changed = original.clone();
    changed.profile = "required_external".to_owned();
    assert!(changed.validate().is_err());
    changed = original.clone();
    changed.root_id = [0; 16];
    assert!(changed.validate().is_err());
    changed = original.clone();
    changed.authority.push(0);
    assert!(changed.validate().is_err());
    changed = original.clone();
    changed.identity.identity.storage.dimension += 1;
    assert!(changed.validate().is_err());
    changed = original.clone();
    changed.image_bytes = 1;
    assert!(changed.validate().is_err());
    changed = original;
    changed.root = PathBuf::from("relative/root");
    assert!(changed.validate().is_err());
}

#[test]
fn saved_attempt_round_trip_preserves_both_slots_and_the_original_identities() {
    let mut selected = receipt();
    let first = selected.authority().unwrap();
    let second =
        AuthorityRefV1::new(2, [0x71; 16], 256, [0x72; 32], Some(first.fingerprint())).unwrap();
    let third =
        AuthorityRefV1::new(3, [0x73; 16], 256, [0x74; 32], Some(second.fingerprint())).unwrap();
    let permit = AttemptPermitV1::new(
        selected.root_id,
        [0x81; 16],
        [0x82; 16],
        third,
        ExpectedAuthorityPairV1 {
            first: Some(AuthoritySlotV1::new(0, selected.root_id, second).unwrap()),
            second: Some(AuthoritySlotV1::new(1, selected.root_id, first).unwrap()),
        },
        [0x83; 16],
    )
    .unwrap();
    selected.authority = third.canonical_bytes().to_vec();
    selected.attempt = Some(SavedAttempt::from_permit(&permit, false));
    let encoded = serde_json::to_vec(&selected).unwrap();
    let restored: Receipt = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(restored.recovery_attempt().unwrap(), permit);
    assert_eq!(
        restored.recovery_attempt().unwrap().stage,
        AttemptStageV1::AttemptFrame
    );
    let mut changed = restored.clone();
    changed.attempt.as_mut().unwrap().requires_floor = true;
    assert!(changed.validate().is_err());
    changed = restored.clone();
    changed.attempt.as_mut().unwrap().writer_id = [0; 16];
    assert!(changed.validate().is_err());
    changed = restored.clone();
    changed.attempt.as_mut().unwrap().attempt_id = [0; 16];
    assert!(changed.validate().is_err());
    changed = restored.clone();
    changed.attempt.as_mut().unwrap().idempotency_key = [0; 16];
    assert!(changed.validate().is_err());
    changed = restored.clone();
    changed.attempt.as_mut().unwrap().expected_first = None;
    assert!(
        changed.validate().is_err(),
        "the complete predecessor pair is required"
    );
    changed = restored;
    changed.attempt.as_mut().unwrap().expected_second = Some(vec![0; 128]);
    assert!(changed.validate().is_err());
}

#[test]
fn selection_only_receipts_remain_readable_but_cannot_authorize_recovery() {
    let mut selected = receipt();
    selected.schema = SELECTION_ONLY_RECEIPT_SCHEMA.to_owned();
    assert!(
        selected.validate().is_err(),
        "a v2 attempt cannot be relabelled v1"
    );
    selected.attempt = None;
    selected.validate().unwrap();
    let encoded = serde_json::to_vec(&selected).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
    assert!(value.get("attempt").is_none());
    let restored: Receipt = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(restored.binding().unwrap(), selected.binding().unwrap());
    assert_eq!(restored.authority().unwrap(), selected.authority().unwrap());
    assert!(restored.recovery_attempt().is_err());
    selected.schema = RECEIPT_SCHEMA.to_owned();
    assert!(
        selected.validate().is_err(),
        "v2 requires its prepared attempt"
    );
}

#[test]
fn unknown_commit_reports_the_exact_attempt_instead_of_ordinary_failure() {
    let receipt = receipt();
    let permit = AttemptPermitV1 {
        root_id: receipt.root_id,
        writer_id: [0x61; 16],
        attempt_id: [0x62; 16],
        target_slot: 1,
        candidate: receipt.authority().unwrap(),
        expected: ExpectedAuthorityPairV1::default(),
        stage: AttemptStageV1::SlotReread,
        idempotency_key: [0x63; 16],
    };
    let error = publication_error(
        &receipt,
        "unknown",
        "not_activated",
        "reconciliation_required",
        Some(&permit),
    );
    let payload = publication_failure(error.as_ref()).unwrap();
    assert_eq!(payload["publication"], "unknown");
    assert_eq!(payload["serving"], "not_activated");
    let candidate: Vec<u8> =
        serde_json::from_value(payload["attempt"]["candidate"].clone()).unwrap();
    assert_eq!(
        AuthorityRefV1::from_canonical_bytes(&candidate).unwrap(),
        permit.candidate
    );
    assert_eq!(payload["attempt"]["stage"], "SlotReread");
    let committed = publication_error(
        &receipt,
        "committed",
        "not_activated",
        "engine_admission_refused",
        None,
    );
    assert_eq!(
        publication_failure(committed.as_ref()).unwrap()["publication"],
        "committed"
    );
    assert!(publication_failure(bad("ordinary input failure").as_ref()).is_none());
}

#[test]
fn root_receipt_and_work_destinations_are_disjoint() {
    separate_destinations(&[
        Path::new("/a/root"),
        Path::new("/a/work"),
        Path::new("/a/receipt"),
    ])
    .unwrap();
    for paths in [
        [Path::new("/a/root"), Path::new("/a/root/work")],
        [Path::new("/a/root/receipt"), Path::new("/a/root")],
        [Path::new("/a/root"), Path::new("/a/root")],
    ] {
        assert!(separate_destinations(&paths).is_err());
    }
}

#[test]
fn recovery_reports_every_verdict_without_turning_uncertainty_into_an_abort() {
    use frankensearch_core::generation::GenerationAuthorityErrorV1;

    let selected = receipt();
    let permit = selected.recovery_attempt().unwrap();
    let slot = AuthoritySlotV1::new(1, selected.root_id, permit.candidate).unwrap();
    let other = AuthorityRefV1::new(1, [0x91; 16], 256, [0x92; 32], None).unwrap();
    let fork = AuthoritySlotV1::new(1, selected.root_id, other).unwrap();
    let successor = AuthorityRefV1::new(
        2,
        [0x93; 16],
        256,
        [0x94; 32],
        Some(permit.candidate.fingerprint()),
    )
    .unwrap();
    let head = AuthoritySlotV1::new(0, selected.root_id, successor).unwrap();
    for (outcome, verdict, publication, serving, resolved) in [
        (
            ReconcileOutcomeV1::Committed { slot, floor: None },
            "committed",
            "committed",
            "not_activated",
            true,
        ),
        (
            ReconcileOutcomeV1::NotCommitted,
            "not_committed",
            "not_committed",
            "not_activated",
            true,
        ),
        (
            ReconcileOutcomeV1::StillUnknown(permit),
            "still_unknown",
            "unknown",
            "not_activated",
            false,
        ),
        (
            ReconcileOutcomeV1::Fork { observed: fork },
            "fork",
            "unknown",
            "not_activated",
            false,
        ),
        (
            ReconcileOutcomeV1::RepairRequired(GenerationAuthorityErrorV1::UnresolvedAttempt),
            "repair_required",
            "unknown",
            "not_activated",
            false,
        ),
        (
            ReconcileOutcomeV1::CommittedThenSuperseded { head },
            "committed_then_superseded",
            "committed",
            "superseded",
            true,
        ),
    ] {
        let report = reconciliation_report(&selected, &permit, &outcome);
        assert_eq!(report["reconciliation"], verdict);
        assert_eq!(report["publication"], publication);
        assert_eq!(report["serving"], serving);
        assert_eq!(report["ok"], resolved);
        assert_eq!(report["attempt"], attempt_report(&permit));
        let mut output = Vec::new();
        let result = emit_recovery(&mut output, report.clone());
        if resolved {
            result.unwrap();
            let emitted: serde_json::Value = serde_json::from_slice(&output).unwrap();
            assert_eq!(emitted, report);
        } else {
            let error = result.unwrap_err();
            assert!(output.is_empty());
            assert_eq!(publication_failure(error.as_ref()).unwrap(), report);
        }
    }
}

#[test]
fn recovery_output_failure_preserves_the_exact_reconciliation_and_commit_knowledge() {
    struct Closed;
    impl Write for Closed {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(io::ErrorKind::BrokenPipe, "closed output"))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let selected = receipt();
    let permit = selected.recovery_attempt().unwrap();
    let slot = AuthoritySlotV1::new(1, selected.root_id, permit.candidate).unwrap();
    let mut report = reconciliation_report(
        &selected,
        &permit,
        &ReconcileOutcomeV1::Committed { slot, floor: None },
    );
    report["serving"] = serde_json::json!("activated");
    let error = emit_recovery(&mut Closed, report).unwrap_err();
    let failure = publication_failure(error.as_ref()).unwrap();
    assert_eq!(failure["publication"], "committed");
    assert_eq!(failure["reconciliation"], "committed");
    assert_eq!(failure["serving"], "activated");
    assert_eq!(failure["reason"], "result_delivery_failed");
    assert_eq!(failure["attempt"], attempt_report(&permit));
}

#[test]
fn new_destinations_cannot_add_files_inside_an_existing_authority_root() {
    let parent = tempfile::tempdir().unwrap().keep();
    let root = parent.join("existing-root");
    fs::create_dir(&root).unwrap();
    fs::write(root.join("AUTHORITY"), [0; 8192]).unwrap();
    for destination in ["new-root", "new-work", "receipt.json"] {
        assert!(new_path(&root.join(destination)).is_err());
    }
    #[cfg(unix)]
    {
        let alias = parent.join("alias");
        std::os::unix::fs::symlink(&root, &alias).unwrap();
        assert!(new_path(&alias.join("receipt.json")).is_err());
    }
    assert_eq!(
        fs::read_dir(&root).unwrap().count(),
        1,
        "refusal must precede any destination mutation"
    );
    assert_eq!(fs::read(root.join("AUTHORITY")).unwrap(), vec![0; 8192]);
}

#[test]
fn failed_result_delivery_preserves_known_commit_and_activation() {
    struct BrokenOutput;
    impl Write for BrokenOutput {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(io::ErrorKind::BrokenPipe, "closed output"))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let error = emit_indexed(
        &mut BrokenOutput,
        &receipt(),
        Path::new("/private/receipt.json"),
        NativeRetrievalMode::Ann,
    )
    .unwrap_err();
    let payload = publication_failure(error.as_ref()).unwrap();
    assert_eq!(payload["publication"], "committed");
    assert_eq!(payload["serving"], "activated");
    assert_eq!(payload["reason"], "result_delivery_failed");
}

#[test]
fn indexed_report_exposes_the_realized_retrieval_without_changing_the_receipt() {
    let selected = receipt();
    for (mode, label) in [
        (NativeRetrievalMode::Ann, "hnsw"),
        (
            NativeRetrievalMode::Exact {
                reason: NativeExactReason::Requested,
            },
            "exact",
        ),
    ] {
        let mut output = Vec::new();
        emit_indexed(&mut output, &selected, Path::new("/receipt.json"), mode).unwrap();
        let report: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(report["retrieval"], label);
        assert_eq!(
            report["selection"],
            serde_json::to_value(&selected).unwrap()
        );
        assert_eq!(report["publication"], "committed");
        assert_eq!(report["serving"], "activated");
    }
}

#[cfg(target_os = "linux")]
#[test]
fn qualified_authority_cli_publishes_and_reopens_fast_and_quality_generations() {
    use frankensearch::IndexableDocument;
    use std::os::unix::fs::DirBuilderExt as _;
    use std::time::{SystemTime, UNIX_EPOCH};

    // Same real-host discipline as generation_root's tests. Qualification
    // failures remain failures: no overlay/tmpfs bypass or automatic skip.
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let parent = PathBuf::from(std::env::var_os("HOME").unwrap()).join(format!(
        ".frankensearch-authority-cli-tests-{}-{stamp}",
        std::process::id(),
    ));
    fs::DirBuilder::new().mode(0o700).create(&parent).unwrap();
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        // A durable receipt can precede every attempt write. Recovery must prove
        // non-commit without requiring model files or staged component images.
        let unissued_path = parent.join("unissued-root");
        drop(create_root(&unissued_path).expect("qualified Linux root is required"));
        let mut unissued = receipt();
        unissued.root = unissued_path.clone();
        let unissued_receipt = parent.join("unissued.json");
        save_receipt(&unissued, &unissued_receipt).unwrap();
        let unissued_options = Options::parse([
            "authority-recover".to_owned(),
            "--receipt".to_owned(),
            unissued_receipt.to_str().unwrap().to_owned(),
        ])
        .unwrap()
        .unwrap();
        let mut output = Vec::new();
        recover(&cx, &unissued_options, &mut output).unwrap();
        let report: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(report["publication"], "not_committed");
        assert_eq!(report["reconciliation"], "not_committed");
        assert_eq!(report["serving"], "not_activated");
        assert_eq!(
            fs::read(unissued_path.join("AUTHORITY")).unwrap(),
            vec![0; 8192]
        );
        assert_eq!(fs::read(unissued_path.join("LOCK")).unwrap(), vec![0; 8192]);

        for (name, tier, ann) in [
            ("fast-exact", EmbedderTierTag::Fast, false),
            ("quality-exact", EmbedderTierTag::Quality, false),
            ("fast-hnsw", EmbedderTierTag::Fast, true),
            ("quality-hnsw", EmbedderTierTag::Quality, true),
        ] {
            let retrieval = if ann {
                NativeBuildRetrieval::Hnsw {
                    params: HnswParams::default(),
                    seed: 42,
                }
            } else {
                NativeBuildRetrieval::Exact
            };
            let mode = if ann {
                NativeRetrievalMode::Ann
            } else {
                NativeRetrievalMode::Exact {
                    reason: NativeExactReason::Requested,
                }
            };
            let root_path = parent.join(format!("{name}-root"));
            let root = create_root(&root_path).expect("qualified Linux root is required");
            let work = parent.join(format!("{name}-work"));
            let receipt_path = parent.join(format!("{name}.json"));
            let provider = Arc::new(Provider {
                identity: identity().identity,
                calls: AtomicUsize::new(0),
            });
            let model: Arc<dyn Embedder> = provider.clone();
            let built = NativeIndexBuilder::new(&work, new_generation(1).unwrap(), model)
                .unwrap()
                .with_fast_storage(NativeBuildPrecision::F32, retrieval)
                .add_documents([
                    IndexableDocument::new("zeta", "retry network requests")
                        .with_metadata("version", "original"),
                    IndexableDocument::new("alpha", "garden flowers"),
                ])
                .build_exact_generation(
                    &cx,
                    tier,
                    CommitRange { low: 2, high: 7 },
                    DEFAULT_IMAGE_BYTES,
                )
                .await
                .unwrap();
            assert_eq!(built.retrieval_mode(), mode);
            let published = stage_and_publish(
                &cx,
                &root,
                &root_path,
                &receipt_path,
                &built,
                None,
                DEFAULT_IMAGE_BYTES,
            )
            .unwrap();
            let successor = built
                .candidate(
                    new_generation(1).unwrap().nonce,
                    Some(published.authority().unwrap()),
                    GenerationAuthorityActionV1::Rollback,
                    [0x71; 32],
                )
                .unwrap();
            drop(published);
            drop(built);
            drop(root);
            let selected = Receipt::read(&receipt_path).unwrap();
            let recorded = selected.recovery_attempt().unwrap();
            // Reproduce the on-disk cut after the slot barrier but before the
            // owner frame. Only the pre-publication receipt is used to recover.
            let mut lock = OpenOptions::new()
                .write(true)
                .open(root_path.join("LOCK"))
                .unwrap();
            lock.write_all(&[0; 4096]).unwrap();
            lock.sync_all().unwrap();
            drop(lock);
            assert!(selected.open(&cx, DEFAULT_IMAGE_BYTES).is_err());
            let authority_before_recovery = fs::read(root_path.join("AUTHORITY")).unwrap();
            let lock_before_recovery = fs::read(root_path.join("LOCK")).unwrap();
            let mut altered = selected.clone();
            altered.attempt.as_mut().unwrap().writer_id[0] ^= 1;
            let altered_path = parent.join(format!("{name}-altered.json"));
            save_receipt(&altered, &altered_path).unwrap();
            let mut recovery = Options::parse([
                "authority-recover".to_owned(),
                "--receipt".to_owned(),
                altered_path.to_str().unwrap().to_owned(),
            ])
            .unwrap()
            .unwrap();
            let error = recover(&cx, &recovery, &mut Vec::new()).unwrap_err();
            let refused = publication_failure(error.as_ref()).unwrap();
            assert_eq!(refused["publication"], "unknown");
            assert_eq!(refused["reconciliation"], "repair_required");
            assert_eq!(
                fs::read(root_path.join("AUTHORITY")).unwrap(),
                authority_before_recovery
            );
            assert_eq!(
                fs::read(root_path.join("LOCK")).unwrap(),
                lock_before_recovery
            );

            recovery.receipt = receipt_path.clone();
            let calls_before_recovery = provider.calls.load(Ordering::Relaxed);
            let mut recovered_output = Vec::new();
            recover(&cx, &recovery, &mut recovered_output).unwrap();
            let recovered: serde_json::Value = serde_json::from_slice(&recovered_output).unwrap();
            assert_eq!(recovered["publication"], "committed");
            assert_eq!(recovered["reconciliation"], "committed");
            assert_eq!(recovered["serving"], "activated");
            assert_eq!(recovered["retrieval"], retrieval_name(mode));
            assert_eq!(recovered["attempt"], attempt_report(&recorded));
            assert_eq!(
                provider.calls.load(Ordering::Relaxed),
                calls_before_recovery
            );
            assert_eq!(
                fs::read(root_path.join("AUTHORITY")).unwrap(),
                authority_before_recovery
            );
            let owner = admit_root(&root_path)
                .unwrap()
                .authority_publisher(selected.root_id, [0xa1; 16], PROFILE)
                .unwrap()
                .owner_frame()
                .unwrap()
                .unwrap();
            assert_eq!(owner.writer_id, recorded.writer_id);
            assert_eq!(owner.attempt_id, recorded.attempt_id);
            recovery.max_image_bytes = 1;
            let error = recover(&cx, &recovery, &mut Vec::new()).unwrap_err();
            let bounded = publication_failure(error.as_ref()).unwrap();
            assert_eq!(bounded["publication"], "committed");
            assert_eq!(bounded["reconciliation"], "committed");
            assert_eq!(bounded["serving"], "not_activated");
            assert_eq!(bounded["reason"], "snapshot_activation_deferred_or_refused");
            assert_eq!(
                provider.calls.load(Ordering::Relaxed),
                calls_before_recovery
            );
            recovery.max_image_bytes = DEFAULT_IMAGE_BYTES;
            assert!(selected.open(&cx, 1).is_err());
            let before = provider.calls.load(Ordering::Relaxed);
            let snapshot = selected.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
            assert_eq!(snapshot.retrieval_mode(), mode);
            assert_eq!(
                provider.calls.load(Ordering::Relaxed),
                before,
                "reopening must not re-embed documents"
            );
            let results = snapshot
                .search_text(&cx, provider.as_ref(), "retry network", 2)
                .await
                .unwrap();
            assert_eq!(results[0].doc_id, "zeta");
            assert_eq!(snapshot.tier(), tier);
            assert_eq!(snapshot.metadata().total_documents, 2);
            assert_eq!(provider.calls.load(Ordering::Relaxed), before + 1);
            let mut wrong = selected.clone();
            wrong.tier = if tier == EmbedderTierTag::Fast {
                EmbedderTierTag::Quality
            } else {
                EmbedderTierTag::Fast
            };
            assert!(wrong.open(&cx, DEFAULT_IMAGE_BYTES).is_err());
            wrong = selected.clone();
            wrong.generation.nonce[0] ^= 1;
            assert!(wrong.open(&cx, DEFAULT_IMAGE_BYTES).is_err());
            // A valid new authority selecting the same image bytes still must
            // not be adopted through an old caller-owned selection receipt.
            let root = admit_root(&root_path).unwrap();
            write_immutable(
                &cx,
                &root_path.join(successor.manifest_name()),
                successor.manifest_bytes(),
            )
            .unwrap();
            let mut publisher =
                GenerationPublisherV1::new(&root, selected.root_id, [0x72; 16], PROFILE).unwrap();
            assert!(matches!(
                publisher
                    .publish(
                        &cx,
                        &successor,
                        snapshot.retained_snapshot().authority_pair(),
                        None,
                        [0x73; 16],
                        DEFAULT_IMAGE_BYTES as u64,
                    )
                    .unwrap()
                    .publication,
                PublicationOutcomeV1::Committed { .. }
            ));
            assert!(selected.open(&cx, DEFAULT_IMAGE_BYTES).is_err());
            let before_superseded = provider.calls.load(Ordering::Relaxed);
            let mut superseded_output = Vec::new();
            recover(&cx, &recovery, &mut superseded_output).unwrap();
            let superseded: serde_json::Value = serde_json::from_slice(&superseded_output).unwrap();
            assert_eq!(superseded["publication"], "committed");
            assert_eq!(superseded["reconciliation"], "committed_then_superseded");
            assert_eq!(superseded["serving"], "superseded");
            assert_eq!(provider.calls.load(Ordering::Relaxed), before_superseded);
            assert!(
                !snapshot
                    .search_text(&cx, provider.as_ref(), "retry", 1)
                    .await
                    .unwrap()
                    .is_empty()
            );
        }
    });
}

#[cfg(target_os = "linux")]
#[test]
fn qualified_rebuild_reuses_images_replaces_the_cohort_and_fences_a_losing_publisher() {
    use frankensearch::IndexableDocument;
    use frankensearch_core::generation::GenerationComponentRole;
    use std::os::unix::fs::{DirBuilderExt as _, MetadataExt as _};
    use std::time::{SystemTime, UNIX_EPOCH};

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let parent = PathBuf::from(std::env::var_os("HOME").unwrap()).join(format!(
        ".frankensearch-authority-rebuild-tests-{}-{stamp}",
        std::process::id(),
    ));
    fs::DirBuilder::new().mode(0o700).create(&parent).unwrap();
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for (name, tier) in [
            ("fast", EmbedderTierTag::Fast),
            ("quality", EmbedderTierTag::Quality),
        ] {
            let root_path = parent.join(format!("{name}-root"));
            let root = create_root(&root_path).expect("qualified Linux root is required");
            let provider = Arc::new(Provider {
                identity: identity().identity,
                calls: AtomicUsize::new(0),
            });
            let model: Arc<dyn Embedder> = provider.clone();
            let original_documents = || {
                [
                    IndexableDocument::new("zeta", "retry network requests")
                        .with_metadata("version", "old"),
                    IndexableDocument::new("alpha", "garden flowers"),
                ]
            };
            let first = NativeIndexBuilder::new(
                parent.join(format!("{name}-first-work")),
                new_generation(1).unwrap(),
                model.clone(),
            )
            .unwrap()
            .add_documents(original_documents())
            .build_exact_generation(
                &cx,
                tier,
                CommitRange { low: 2, high: 7 },
                DEFAULT_IMAGE_BYTES,
            )
            .await
            .unwrap();
            let first_path = parent.join(format!("{name}-first.json"));
            let selected = stage_and_publish(
                &cx,
                &root,
                &root_path,
                &first_path,
                &first,
                None,
                DEFAULT_IMAGE_BYTES,
            )
            .unwrap();
            let retained = selected
                .open_on_root(&cx, &root, DEFAULT_IMAGE_BYTES)
                .unwrap();
            let lexical_receipt = retained.retained_snapshot().manifest().components().lexical;
            let lexical_path = root_path.join(
                frankensearch_index::generation_root::generation_reader::component_object_name_v1(
                    GenerationComponentRole::Lexical,
                    lexical_receipt.sha256,
                ),
            );
            let original_inode = fs::metadata(&lexical_path).unwrap().ino();
            let original_lexical = fs::read(&lexical_path).unwrap();
            let first_receipt_bytes = fs::read(&first_path).unwrap();

            // An unchanged full cohort shares its exact lexical object. A
            // create_new-only pathname writer would fail this valid successor.
            let unchanged = NativeIndexBuilder::new(
                parent.join(format!("{name}-unchanged-work")),
                new_generation(2).unwrap(),
                model.clone(),
            )
            .unwrap()
            .add_documents(original_documents())
            .build_exact_generation(
                &cx,
                tier,
                CommitRange { low: 2, high: 7 },
                DEFAULT_IMAGE_BYTES,
            )
            .await
            .unwrap();
            assert_eq!(
                first.component_bytes(GenerationComponentRole::Lexical),
                unchanged.component_bytes(GenerationComponentRole::Lexical)
            );
            let second_path = parent.join(format!("{name}-second.json"));
            let second = stage_and_publish(
                &cx,
                &root,
                &root_path,
                &second_path,
                &unchanged,
                Some(&retained),
                DEFAULT_IMAGE_BYTES,
            )
            .unwrap();
            assert_eq!(second.root_id, selected.root_id);
            assert_eq!(second.authority().unwrap().sequence, 2);
            assert_eq!(fs::metadata(&lexical_path).unwrap().ino(), original_inode);
            assert_eq!(fs::read(&lexical_path).unwrap(), original_lexical);
            let second_snapshot = second
                .open_on_root(&cx, &root, DEFAULT_IMAGE_BYTES)
                .unwrap();
            assert_eq!(
                second.recovery_attempt().unwrap().expected,
                retained.retained_snapshot().authority_pair()
            );

            let winner = NativeIndexBuilder::new(
                parent.join(format!("{name}-winner-work")),
                new_generation(3).unwrap(),
                model.clone(),
            )
            .unwrap()
            .add_documents([
                IndexableDocument::new("zeta", "retry requests safely")
                    .with_metadata("version", "new"),
                IndexableDocument::new("beta", "fresh garden vegetables"),
            ])
            .build_exact_generation(
                &cx,
                tier,
                CommitRange { low: 2, high: 8 },
                DEFAULT_IMAGE_BYTES,
            )
            .await
            .unwrap();
            let loser = NativeIndexBuilder::new(
                parent.join(format!("{name}-loser-work")),
                new_generation(3).unwrap(),
                model,
            )
            .unwrap()
            .add_documents([IndexableDocument::new("loser", "retry obsolete requests")])
            .build_exact_generation(
                &cx,
                tier,
                CommitRange { low: 2, high: 9 },
                DEFAULT_IMAGE_BYTES,
            )
            .await
            .unwrap();
            let winner_path = parent.join(format!("{name}-winner.json"));
            let committed = stage_and_publish(
                &cx,
                &root,
                &root_path,
                &winner_path,
                &winner,
                Some(&second_snapshot),
                DEFAULT_IMAGE_BYTES,
            )
            .unwrap();
            let authority_after_winner = fs::read(root_path.join("AUTHORITY")).unwrap();
            let lock_after_winner = fs::read(root_path.join("LOCK")).unwrap();
            let loser_path = parent.join(format!("{name}-loser.json"));
            let error = stage_and_publish(
                &cx,
                &root,
                &root_path,
                &loser_path,
                &loser,
                Some(&second_snapshot),
                DEFAULT_IMAGE_BYTES,
            )
            .unwrap_err();
            assert_eq!(
                publication_failure(error.as_ref()).unwrap()["publication"],
                "not_committed"
            );
            assert_eq!(
                fs::read(root_path.join("AUTHORITY")).unwrap(),
                authority_after_winner
            );
            assert_eq!(fs::read(root_path.join("LOCK")).unwrap(), lock_after_winner);
            assert_eq!(fs::read(&first_path).unwrap(), first_receipt_bytes);
            assert_eq!(fs::read(&lexical_path).unwrap(), original_lexical);
            let lost = Receipt::read(&loser_path).unwrap();
            assert_eq!(
                lost.recovery_attempt().unwrap().expected,
                second_snapshot.retained_snapshot().authority_pair()
            );
            assert_ne!(lost.authority().unwrap(), committed.authority().unwrap());
            assert!(selected.open(&cx, DEFAULT_IMAGE_BYTES).is_err());
            assert!(second.open(&cx, DEFAULT_IMAGE_BYTES).is_err());

            let active = committed.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
            let current_results = active
                .search_text(&cx, provider.as_ref(), "retry garden", 10)
                .await
                .unwrap();
            let current_ids: BTreeSet<_> = current_results
                .iter()
                .map(|hit| hit.doc_id.as_str())
                .collect();
            assert_eq!(current_ids, BTreeSet::from(["beta", "zeta"]));
            let old_results = retained
                .search_text(&cx, provider.as_ref(), "retry garden", 10)
                .await
                .unwrap();
            let old_ids: BTreeSet<_> = old_results.iter().map(|hit| hit.doc_id.as_str()).collect();
            assert_eq!(old_ids, BTreeSet::from(["alpha", "zeta"]));
            assert_eq!(
                active.metadata().commit_range,
                CommitRange { low: 2, high: 8 }
            );
            assert_eq!(
                retained.metadata().commit_range,
                CommitRange { low: 2, high: 7 }
            );
            let calls = provider.calls.load(Ordering::Relaxed);
            let recovery = Options::parse([
                "authority-recover".to_owned(),
                "--receipt".to_owned(),
                winner_path.to_str().unwrap().to_owned(),
            ])
            .unwrap()
            .unwrap();
            let mut output = Vec::new();
            recover(&cx, &recovery, &mut output).unwrap();
            let report: serde_json::Value = serde_json::from_slice(&output).unwrap();
            assert_eq!(report["publication"], "committed");
            assert_eq!(report["serving"], "activated");
            assert_eq!(provider.calls.load(Ordering::Relaxed), calls);

            // A stale receipt is refused before even trying the deliberately
            // missing input or models, and cannot create new work/receipt files.
            let mut options = parse(&rebuild_arguments()).unwrap().unwrap();
            options.from_receipt = Some(first_path);
            let refused_work = parent.join(format!("{name}-stale-work"));
            let refused_receipt = parent.join(format!("{name}-stale.json"));
            options.work = Some(refused_work.clone());
            options.receipt = refused_receipt.clone();
            options.input = Some(parent.join("missing-input.jsonl"));
            options.models = Some(parent.join("missing-models"));
            let error = rebuild(&cx, &options, &mut Vec::new(), None)
                .await
                .unwrap_err();
            assert!(error.to_string().contains("head differs"));
            assert!(!refused_work.exists());
            assert!(!refused_receipt.exists());
            assert_eq!(
                fs::read(root_path.join("AUTHORITY")).unwrap(),
                authority_after_winner
            );
        }
    });
}

#[cfg(target_os = "linux")]
mod warm_authority {
    use std::io::{BufRead, Cursor, Read};
    use std::os::unix::fs::DirBuilderExt as _;
    use std::time::{SystemTime, UNIX_EPOCH};

    use frankensearch::IndexableDocument;

    use super::*;

    fn private_parent(label: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let parent = PathBuf::from(std::env::var_os("HOME").unwrap()).join(format!(
            ".frankensearch-authority-serve-{label}-{}-{stamp}",
            std::process::id()
        ));
        // Qualification uses the actual host filesystem. Unsupported roots
        // remain test failures; no fixture bypass admits overlay/tmpfs.
        fs::DirBuilder::new().mode(0o700).create(&parent).unwrap();
        parent
    }

    async fn publish(
        cx: &Cx,
        parent: &Path,
        previous: Option<&NativeGenerationSnapshot>,
        provider: &Arc<Provider>,
        tier: EmbedderTierTag,
        label: &str,
        ann: bool,
    ) -> Receipt {
        let root_path = parent.join("root");
        let root = if previous.is_some() {
            admit_root(&root_path).unwrap()
        } else {
            create_root(&root_path).expect("qualified Linux root is required")
        };
        let sequence = previous.map_or(1, |snapshot| snapshot.authority().sequence + 1);
        let model: Arc<dyn Embedder> = provider.clone();
        let retrieval = if ann {
            NativeBuildRetrieval::Hnsw {
                params: HnswParams::default(),
                seed: 17,
            }
        } else {
            NativeBuildRetrieval::Exact
        };
        let built = NativeIndexBuilder::new(
            parent.join(format!("{label}-work")),
            new_generation(sequence).unwrap(),
            model,
        )
        .unwrap()
        .with_fast_storage(NativeBuildPrecision::F32, retrieval)
        .add_documents([
            IndexableDocument::new(format!("retry-{label}"), "retry network requests")
                .with_metadata("version", label),
            IndexableDocument::new(format!("garden-{label}"), "garden flowers")
                .with_metadata("version", label),
        ])
        .build_exact_generation(
            cx,
            tier,
            CommitRange {
                low: 1,
                high: sequence + 4,
            },
            DEFAULT_IMAGE_BYTES,
        )
        .await
        .unwrap();
        stage_and_publish(
            cx,
            &root,
            &root_path,
            &parent.join(format!("{label}.json")),
            &built,
            previous,
            DEFAULT_IMAGE_BYTES,
        )
        .unwrap()
    }

    fn open_session(
        cx: &Cx,
        selected: Receipt,
        provider: &Arc<Provider>,
        allow_activation: bool,
    ) -> AuthoritySession {
        let root = admit_root(&selected.root).unwrap();
        let snapshot = selected
            .open_on_root(cx, &root, DEFAULT_IMAGE_BYTES)
            .unwrap();
        let model: Arc<dyn Embedder> = provider.clone();
        AuthoritySession::new(
            cx,
            root,
            selected,
            snapshot,
            model,
            (1, DEFAULT_IMAGE_BYTES),
            allow_activation,
        )
        .unwrap()
    }

    fn frames(bytes: &[u8]) -> Vec<serde_json::Value> {
        bytes
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice(line).unwrap())
            .collect()
    }

    async fn exchange(
        cx: &Cx,
        session: &mut AuthoritySession,
        messages: &[serde_json::Value],
    ) -> Vec<serde_json::Value> {
        let mut input = Vec::new();
        for message in messages {
            serde_json::to_writer(&mut input, message).unwrap();
            input.push(b'\n');
        }
        let mut output = Vec::new();
        serve_session(cx, session, &mut Cursor::new(input), &mut output)
            .await
            .unwrap();
        let frames = frames(&output);
        assert_eq!(frames.len(), messages.len() + 1);
        assert_eq!(frames[0]["event"], "ready");
        frames
    }

    fn provider() -> Arc<Provider> {
        Arc::new(Provider {
            identity: identity().identity,
            calls: AtomicUsize::new(0),
        })
    }

    #[test]
    fn warm_queries_reuse_the_admitted_model_and_images_for_both_tiers() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for (label, tier, ann) in [
                ("fast-exact", EmbedderTierTag::Fast, false),
                ("quality-exact", EmbedderTierTag::Quality, false),
                ("fast-ann", EmbedderTierTag::Fast, true),
                ("quality-ann", EmbedderTierTag::Quality, true),
            ] {
                let parent = private_parent(label);
                let provider = provider();
                let selected = publish(&cx, &parent, None, &provider, tier, "old", ann).await;
                let mut session = open_session(&cx, selected.clone(), &provider, false);
                let before = provider.calls.load(Ordering::Relaxed);
                let output = exchange(
                    &cx,
                    &mut session,
                    &[
                        serde_json::json!({"id": "first", "query": "retry"}),
                        serde_json::json!({"id": "second", "query": "garden", "limit": 2}),
                    ],
                )
                .await;
                assert_eq!(output[0]["retrieval"], if ann { "hnsw" } else { "exact" });
                assert_eq!(output[1]["event"], "results");
                assert_eq!(output[1]["id"], "first");
                assert_eq!(output[1]["results"].as_array().unwrap().len(), 1);
                assert_eq!(output[1]["results"][0]["doc_id"], "retry-old");
                assert_eq!(output[2]["id"], "second");
                assert_eq!(output[2]["results"].as_array().unwrap().len(), 2);
                assert_eq!(output[2]["results"][0]["doc_id"], "garden-old");
                assert_eq!(provider.calls.load(Ordering::Relaxed), before + 2);

                // The session keeps the complete admitted images. Neither a
                // model reload nor a component pathname reopen is necessary.
                fs::rename(&selected.root, parent.join("relocated-root")).unwrap();
                let output = exchange(
                    &cx,
                    &mut session,
                    &[serde_json::json!({"query": "retry", "limit": 2})],
                )
                .await;
                assert_eq!(output[1]["results"][0]["doc_id"], "retry-old");
                assert_eq!(
                    output[1]["authority"],
                    serde_json::json!(selected.authority)
                );
                assert_eq!(provider.calls.load(Ordering::Relaxed), before + 3);
            }
        });
    }

    #[test]
    fn explicit_activation_is_fenced_and_failures_keep_the_warm_predecessor() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for (label, tier) in [
                ("fast", EmbedderTierTag::Fast),
                ("quality", EmbedderTierTag::Quality),
            ] {
                let parent = private_parent(label);
                let provider = provider();
                let old = publish(&cx, &parent, None, &provider, tier, "old", true).await;
                let mut session = open_session(&cx, old.clone(), &provider, true);
                let mut disabled = open_session(&cx, old.clone(), &provider, false);
                let next = publish(
                    &cx,
                    &parent,
                    Some(&session.snapshot),
                    &provider,
                    tier,
                    "new",
                    false,
                )
                .await;
                let before = provider.calls.load(Ordering::Relaxed);
                let activate = serde_json::json!({
                    "op": "activate", "id": "switch", "receipt": parent.join("new.json"),
                    "expected_authority": old.authority,
                });
                let output = exchange(
                    &cx,
                    &mut disabled,
                    &[activate.clone(), serde_json::json!({"query": "retry"})],
                )
                .await;
                assert_eq!(output[1]["event"], "failed");
                assert_eq!(output[2]["results"][0]["doc_id"], "retry-old");
                assert_eq!(disabled.receipt.authority, old.authority);
                assert_eq!(provider.calls.load(Ordering::Relaxed), before + 1);

                let mut wrong = next.clone();
                wrong.tier = if tier == EmbedderTierTag::Fast {
                    EmbedderTierTag::Quality
                } else {
                    EmbedderTierTag::Fast
                };
                save_receipt(&wrong, &parent.join("wrong-tier.json")).unwrap();
                let mut wrong_producer = next.clone();
                let mut identity = wrong_producer.identity.identity.clone();
                identity.producer.implementation_revision = "different-producer".to_owned();
                wrong_producer.identity = identity.freeze().unwrap();
                save_receipt(&wrong_producer, &parent.join("wrong-producer.json")).unwrap();
                let mut stale = activate.clone();
                stale["expected_authority"] = serde_json::json!(next.authority);
                let mut wrong_tier = activate.clone();
                wrong_tier["receipt"] = serde_json::json!(parent.join("wrong-tier.json"));
                let mut wrong_identity = activate.clone();
                wrong_identity["receipt"] = serde_json::json!(parent.join("wrong-producer.json"));
                let mut missing = activate.clone();
                missing["receipt"] = serde_json::json!(parent.join("absent.json"));
                let output = exchange(
                    &cx,
                    &mut session,
                    &[
                        stale,
                        wrong_tier,
                        wrong_identity,
                        missing,
                        serde_json::json!({"query": "retry"}),
                        activate.clone(),
                        serde_json::json!({"id": "new-query", "query": "retry"}),
                        activate,
                        serde_json::json!({"query": "garden"}),
                    ],
                )
                .await;
                for frame in &output[1..5] {
                    assert_eq!(frame["event"], "failed");
                    assert_eq!(frame["authority"], serde_json::json!(old.authority));
                }
                assert_eq!(output[5]["results"][0]["doc_id"], "retry-old");
                assert_eq!(output[6]["event"], "activated");
                assert_eq!(output[6]["authority"], serde_json::json!(next.authority));
                assert_eq!(output[6]["retrieval"], "exact");
                assert_eq!(output[7]["results"][0]["doc_id"], "retry-new");
                assert_eq!(output[8]["event"], "failed");
                assert_eq!(output[9]["results"][0]["doc_id"], "garden-new");
                assert_eq!(
                    provider.calls.load(Ordering::Relaxed),
                    before + 4,
                    "only queries perform inference; rejected and accepted activations never do"
                );
                assert_eq!(session.receipt.authority, next.authority);
                assert_eq!(disabled.receipt.authority, old.authority);
            }
        });
    }

    #[test]
    fn request_failures_are_bounded_while_oversize_cancellation_and_output_end_the_session() {
        struct FailAfterReady {
            writes: usize,
        }
        impl Write for FailAfterReady {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                self.writes += 1;
                if self.writes > 1 {
                    return Err(io::Error::new(io::ErrorKind::BrokenPipe, "closed"));
                }
                Ok(bytes.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        struct CancelOnRead<'a> {
            cx: &'a Cx,
            inner: Cursor<Vec<u8>>,
        }
        impl Read for CancelOnRead<'_> {
            fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
                self.inner.read(buffer)
            }
        }
        impl BufRead for CancelOnRead<'_> {
            fn fill_buf(&mut self) -> io::Result<&[u8]> {
                self.cx.set_cancel_requested(true);
                self.inner.fill_buf()
            }
            fn consume(&mut self, amount: usize) {
                self.inner.consume(amount);
            }
        }
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let parent = private_parent("protocol");
            let provider = provider();
            let selected = publish(
                &cx,
                &parent,
                None,
                &provider,
                EmbedderTierTag::Fast,
                "old",
                true,
            )
            .await;
            let before = provider.calls.load(Ordering::Relaxed);
            let wrong = Arc::new(Provider {
                identity: {
                    let mut identity = identity().identity;
                    identity.producer.implementation_revision = "wrong-warm-producer".to_owned();
                    identity
                },
                calls: AtomicUsize::new(0),
            });
            let root = admit_root(&selected.root).unwrap();
            let snapshot = selected
                .open_on_root(&cx, &root, DEFAULT_IMAGE_BYTES)
                .unwrap();
            let wrong_model: Arc<dyn Embedder> = wrong.clone();
            assert!(
                AuthoritySession::new(
                    &cx,
                    root,
                    selected.clone(),
                    snapshot,
                    wrong_model,
                    (1, DEFAULT_IMAGE_BYTES),
                    false
                )
                .is_err()
            );
            assert_eq!(wrong.calls.load(Ordering::Relaxed), 0);

            let mut session = open_session(&cx, selected.clone(), &provider, false);
            let input = concat!(
                "{invalid}\n",
                "{\"query\":\"retry\",\"limit\":0}\n",
                "{\"query\":\"retry\",\"limit\":1001}\n",
                "{\"query\":\"retry\",\"retrieval\":\"exact\"}\n",
                "{\"op\":\"unknown\",\"query\":\"retry\"}\n",
                "{\"id\":\"valid\",\"query\":\"retry\",\"limit\":1}\n",
            );
            let mut output = Vec::new();
            serve_session(&cx, &mut session, &mut Cursor::new(input), &mut output)
                .await
                .unwrap();
            let output = frames(&output);
            assert_eq!(output.len(), 7);
            for frame in &output[1..6] {
                assert_eq!(frame["event"], "failed");
            }
            assert_eq!(output[6]["event"], "results");
            assert_eq!(output[6]["id"], "valid");
            assert_eq!(provider.calls.load(Ordering::Relaxed), before + 1);

            let mut oversized = vec![b'x'; crate::serve::MAX_REQUEST_BYTES + 2];
            oversized.extend_from_slice(b"\n{\"query\":\"retry\"}\n");
            let length = oversized.len();
            let mut oversized = Cursor::new(oversized);
            let mut output = Vec::new();
            assert!(
                serve_session(&cx, &mut session, &mut oversized, &mut output)
                    .await
                    .is_err()
            );
            assert!(usize::try_from(oversized.position()).unwrap() < length);
            assert_eq!(
                frames(&output).len(),
                1,
                "only readiness precedes a terminal frame refusal"
            );
            assert_eq!(provider.calls.load(Ordering::Relaxed), before + 1);

            let mut cancelled = CancelOnRead {
                cx: &cx,
                inner: Cursor::new(b"{\"query\":\"retry\"}\n".to_vec()),
            };
            assert!(
                serve_session(&cx, &mut session, &mut cancelled, &mut Vec::new())
                    .await
                    .is_err()
            );
            assert_eq!(provider.calls.load(Ordering::Relaxed), before + 1);
            assert_eq!(session.receipt.authority, selected.authority);
            cx.set_cancel_requested(false);

            let mut output = FailAfterReady { writes: 0 };
            let mut input = Cursor::new(b"{\"query\":\"retry\"}\n{\"query\":\"garden\"}\n");
            assert!(
                serve_session(&cx, &mut session, &mut input, &mut output)
                    .await
                    .is_err()
            );
            assert_eq!(
                provider.calls.load(Ordering::Relaxed),
                before + 2,
                "delivery failure terminates before the next query"
            );
            assert_eq!(output.writes, 2);
        });
    }
}
