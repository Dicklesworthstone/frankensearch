use super::*;
use frankensearch_core::generation::{
    EmbeddingArtifactIdentityV1, EmbeddingIdentityBundleV1, EmbeddingSpaceKindV1,
    QuantizationFormat,
};
use frankensearch_index::generation_root::authority_publisher::AttemptStageV1;

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
    )
    .unwrap_err();
    let payload = publication_failure(error.as_ref()).unwrap();
    assert_eq!(payload["publication"], "committed");
    assert_eq!(payload["serving"], "activated");
    assert_eq!(payload["reason"], "result_delivery_failed");
}

#[cfg(target_os = "linux")]
#[test]
fn qualified_authority_cli_publishes_and_reopens_fast_and_quality_generations() {
    use frankensearch::{Embedder, IndexableDocument};
    use frankensearch_core::{ModelCategory, SearchFuture, SearchResult};
    use std::os::unix::fs::DirBuilderExt as _;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

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
        for (name, tier) in [
            ("fast", EmbedderTierTag::Fast),
            ("quality", EmbedderTierTag::Quality),
        ] {
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
                .with_fast_storage(NativeBuildPrecision::F32, NativeBuildRetrieval::Exact)
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
            let published = stage_and_publish(
                &cx,
                &root,
                &root_path,
                &receipt_path,
                &built,
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
            drop(built);
            drop(root);
            let selected = Receipt::read(&receipt_path).unwrap();
            assert!(selected.open(&cx, 1).is_err());
            let before = provider.calls.load(Ordering::Relaxed);
            let snapshot = selected.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
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
