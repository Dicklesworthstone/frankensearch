use super::*;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::{IndexableDocument, ScoreSource, SearchError};
use frankensearch_core::generation::{
    ActivationInvariant, ActivationManifestV2, ArtifactGenerationIdentityV1, CommitRange,
    EmbeddingArtifactIdentityV1, EmbeddingIdentityBundleV1, EmbeddingSpaceKindV1,
    GenerationAuthorityActionV1, GenerationComponentReceiptV1, GenerationComponentReceiptsV2,
    LexicalArtifact, MANIFEST_SCHEMA_VERSION, QuantizationFormat, VectorArtifact,
    compute_manifest_hash,
};
use frankensearch_core::traits::{ModelCategory, SearchFuture};
use frankensearch_index::VectorIndex;
use frankensearch_quill::{DEFAULT_SCHEMA, QuillIndex};

const BUDGET: usize = 16 * 1024 * 1024;

struct Provider {
    identity: EmbeddingIdentityBundleV1,
    calls: AtomicUsize,
}

impl Provider {
    fn new() -> Self {
        // Synthetic semantic-shaped identity; no model-availability claim.
        let mut identity = EmbeddingIdentityBundleV1::explicit_test_model("composite-test", 2);
        identity.space.kind = EmbeddingSpaceKindV1::Semantic;
        identity.space.hash_control = None;
        identity.space.artifact_manifest_fingerprint = "a".repeat(64);
        identity.space.artifacts = vec![EmbeddingArtifactIdentityV1 {
            role: "weights".to_owned(),
            sha256: "b".repeat(64),
            size: 1,
        }];
        identity.producer.space_fingerprint = identity.space.fingerprint();
        identity.validate().unwrap();
        Self {
            identity,
            calls: AtomicUsize::new(0),
        }
    }
}

impl Embedder for Provider {
    fn embed<'a>(&'a self, _: &'a Cx, _: &'a str) -> SearchFuture<'a, Vec<f32>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Ok(vec![1.0, 0.0]) })
    }
    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
    }
    fn dimension(&self) -> usize {
        2
    }
    fn id(&self) -> &str {
        "composite-test"
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

fn config() -> QuillConfig {
    QuillConfig {
        deterministic_ingest: true,
        max_ingest_shards: 1,
        ..QuillConfig::default()
    }
}

fn source_cut() -> SourceCheckpointV1 {
    SourceCheckpointV1::derive(&CommitRange { low: 1, high: 4 })
}

fn raw_receipt(bytes: &[u8]) -> GenerationComponentReceiptV1 {
    GenerationComponentReceiptV1 {
        byte_len: u64::try_from(bytes.len()).unwrap(),
        sha256: sha2::Sha256::digest(bytes).into(),
    }
}

fn activation(
    binding: &FsviV2IdentityBinding,
    images: &[Arc<[u8]>; 3],
    documents: [u8; 32],
    ann: Option<GenerationComponentReceiptV1>,
) -> ActivationManifest {
    ActivationManifest::V2(
        ActivationManifestV2::new(
            1,
            None,
            GenerationAuthorityActionV1::Activate,
            binding.generation(),
            [0x45; 32],
            source_cut().to_bytes(),
            documents,
            GenerationComponentReceiptsV2 {
                vector: raw_receipt(&images[0]),
                lexical: raw_receipt(&images[1]),
                metadata: raw_receipt(&images[2]),
                ann,
            },
        )
        .unwrap(),
    )
}

struct Fixture {
    directory: tempfile::TempDir,
    binding: FsviV2IdentityBinding,
    metadata: GenerationManifest,
    activation: ActivationManifest,
    images: [Arc<[u8]>; 3],
}

impl Fixture {
    fn admit(&self, cx: &Cx) -> SearchResult<CompositeReaders> {
        admit_components(
            cx,
            &self.activation,
            &self.binding,
            DEFAULT_SCHEMA,
            config(),
            self.images.clone(),
            BUDGET,
        )
    }

    fn admit_resealed(&self, cx: &Cx, images: [Arc<[u8]>; 3]) -> SearchResult<CompositeReaders> {
        let selected = activation(
            &self.binding,
            &images,
            self.activation.document_set_sha256(),
            None,
        );
        admit_components(
            cx,
            &selected,
            &self.binding,
            DEFAULT_SCHEMA,
            config(),
            images,
            BUDGET,
        )
    }
}

async fn fixture(cx: &Cx, provider: &Provider, tier: EmbedderTierTag) -> Fixture {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source");
    std::fs::create_dir(&source).unwrap();
    let path = source.join("vectors.fsvi");
    let mut identity = provider.identity.clone();
    identity.storage.format = "fsvi-v2".to_owned();
    identity.storage.quantization = QuantizationFormat::F32;
    identity.storage.endianness = "little-endian".to_owned();
    let binding = FsviV2IdentityBinding::new(
        ArtifactGenerationIdentityV1::new(7, [0x37; 16]).unwrap(),
        identity.freeze().unwrap(),
    )
    .unwrap();
    let mut vector_writer = VectorIndex::create_v2(&path, binding.clone()).unwrap();
    vector_writer.write_record("alpha", &[1.0, 0.0]).unwrap();
    vector_writer.write_record("beta", &[0.8, 0.6]).unwrap();
    vector_writer
        .write_tombstone_record("dead", &[1.0, 0.0])
        .unwrap();
    vector_writer.finish().unwrap();
    let vector_bytes: Arc<[u8]> = std::fs::read(&path).unwrap().into();
    let owner = ValidatedFsviBytes::from_arc(Arc::clone(&vector_bytes), &binding).unwrap();
    let mut live_ids = Vec::new();
    let mut lexical_documents = Vec::new();
    // Match the real FSVI physical order, which is sorted by document hash.
    for row in 0..owner.record_count() {
        let row = owner.row(row).unwrap();
        if row.flags().is_live() {
            live_ids.push(row.doc_id().to_owned());
        }
        lexical_documents.push(
            IndexableDocument::new(row.doc_id(), "common retained document")
                .with_metadata("version", "old"),
        );
    }
    let lexical = QuillIndex::create(cx, source.join("lexical"), config())
        .await
        .unwrap();
    lexical
        .index_documents(cx, &lexical_documents)
        .await
        .unwrap();
    lexical.commit(cx).await.unwrap();
    assert!(lexical.delete_document(cx, "dead").await.unwrap());
    let lexical_image = lexical
        .snapshot()
        .unwrap()
        .export_exact_image(cx, source_cut(), BUDGET)
        .unwrap();
    let vector = vector_component_receipt(
        owner.witness(),
        live_ids.iter().map(String::as_str),
        source_cut(),
    )
    .unwrap();
    let documents = CanonicalDocsetV1::from_ordered_live_documents(live_ids).unwrap();
    let tier_name = match tier {
        EmbedderTierTag::Fast => "fast",
        EmbedderTierTag::Quality => "quality",
    };
    let mut metadata = GenerationManifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        generation_id: "build-seven".to_owned(),
        manifest_hash: String::new(),
        commit_range: CommitRange { low: 1, high: 4 },
        build_started_at: 1,
        build_completed_at: 2,
        embedders: BTreeMap::from([(tier_name.to_owned(), binding.frozen_identity().clone())]),
        vector_artifacts: vec![VectorArtifact {
            path: "vectors.fsvi".to_owned(),
            size_bytes: vector.bytes.byte_len,
            checksum: lower_hex(&vector.bytes.sha256),
            vector_count: 3,
            dimension: 2,
            embedder_tier: tier,
        }],
        lexical_artifacts: vec![LexicalArtifact {
            path: "lexical.image".to_owned(),
            size_bytes: lexical_image.receipt().bytes.byte_len,
            checksum: lower_hex(&lexical_image.receipt().bytes.sha256),
            document_count: 2,
        }],
        repair_descriptors: Vec::new(),
        activation_invariants: vec![ActivationInvariant {
            id: "stored-vectors".to_owned(),
            description: "physical rows including tombstones".to_owned(),
            kind: InvariantKind::VectorCountConsistency { expected_total: 3 },
        }],
        total_documents: 2,
        metadata: BTreeMap::new(),
    };
    metadata.manifest_hash = compute_manifest_hash(&metadata).unwrap();
    let metadata_bytes: Arc<[u8]> = serde_json::to_vec(&metadata).unwrap().into();
    let metadata_receipt =
        ExactComponentReceiptV1::for_metadata_manifest(&metadata, &metadata_bytes, &documents)
            .unwrap();
    ExactGenerationComponentsV1::admit(
        vector,
        lexical_image.receipt().clone(),
        None,
        metadata_receipt,
    )
    .unwrap();
    let images = [vector_bytes, lexical_image.bytes(), metadata_bytes];
    let activation = activation(&binding, &images, documents.digest(), None);
    Fixture {
        directory,
        binding,
        metadata,
        activation,
        images,
    }
}

#[test]
fn actual_components_search_and_hydrate_fast_and_quality_with_tombstones() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for tier in [EmbedderTierTag::Fast, EmbedderTierTag::Quality] {
            let provider = Provider::new();
            let fixture = fixture(&cx, &provider, tier).await;
            let readers = fixture.admit(&cx).unwrap();
            assert_eq!(readers.metadata.vector_artifacts[0].vector_count, 3);
            assert_eq!(readers.metadata.total_documents, 2);
            let hits = readers
                .search_text(&cx, &provider, "common", usize::MAX)
                .await
                .unwrap();
            assert_eq!(
                hits.len(),
                2,
                "large k is bounded by the admitted live count"
            );
            for hit in &hits {
                assert_ne!(hit.doc_id, "dead");
                assert_eq!(hit.source, ScoreSource::Hybrid);
                assert!(hit.index.is_some() && hit.lexical_score.is_some());
                assert_eq!(hit.metadata.as_deref().unwrap()["version"], "old");
                assert_eq!(hit.fast_score.is_some(), tier == EmbedderTierTag::Fast);
                assert_eq!(
                    hit.quality_score.is_some(),
                    tier == EmbedderTierTag::Quality
                );
            }
            assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        }
    });
}

#[test]
fn admitted_components_survive_source_successor_and_relocation() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let provider = Provider::new();
        let fixture = fixture(&cx, &provider, EmbedderTierTag::Fast).await;
        let retained = fixture.admit(&cx).unwrap();
        let source = fixture.directory.path().join("source");
        let writer = QuillIndex::open(&cx, source.join("lexical"), config())
            .await
            .unwrap();
        writer
            .index_document(
                &cx,
                &IndexableDocument::new("successor", "common new publication")
                    .with_metadata("version", "new"),
            )
            .await
            .unwrap();
        writer.commit(&cx).await.unwrap();
        assert_eq!(
            writer
                .search_paginated(&cx, "common", 10, 0, true)
                .unwrap()
                .total_count,
            Some(3)
        );
        drop(writer);
        std::fs::rename(&source, fixture.directory.path().join("relocated")).unwrap();
        let readmitted = fixture.admit(&cx).unwrap();
        drop(fixture);
        for readers in [retained, readmitted] {
            let hits = readers
                .search_text(&cx, &provider, "common", 10)
                .await
                .unwrap();
            assert_eq!(hits.len(), 2);
            assert!(hits.iter().all(|hit| hit.doc_id != "successor"
                && hit.metadata.as_deref().unwrap()["version"] == "old"));
        }
    });
}

#[test]
fn component_admission_rejects_raw_and_resealed_metadata_drift() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let provider = Provider::new();
        let fixture = fixture(&cx, &provider, EmbedderTierTag::Fast).await;
        assert!(fixture.admit(&cx).is_ok());
        for role in 0..3 {
            let mut images = fixture.images.clone();
            let mut bytes = images[role].to_vec();
            *bytes.last_mut().unwrap() ^= 1;
            images[role] = bytes.into();
            assert!(
                admit_components(
                    &cx,
                    &fixture.activation,
                    &fixture.binding,
                    DEFAULT_SCHEMA,
                    config(),
                    images,
                    BUDGET
                )
                .is_err(),
                "changed raw component {role}"
            );
        }
        for case in 0..4 {
            let mut metadata = fixture.metadata.clone();
            match case {
                0 => metadata.vector_artifacts[0].vector_count = 2,
                1 => metadata.lexical_artifacts[0].checksum = "0".repeat(64),
                2 => metadata.vector_artifacts[0].dimension = 3,
                _ => metadata.commit_range.high += 1,
            }
            metadata.manifest_hash = compute_manifest_hash(&metadata).unwrap();
            let mut images = fixture.images.clone();
            images[2] = serde_json::to_vec(&metadata).unwrap().into();
            assert!(
                fixture.admit_resealed(&cx, images).is_err(),
                "resealed metadata case {case}"
            );
        }
        let selected = activation(
            &fixture.binding,
            &fixture.images,
            fixture.activation.document_set_sha256(),
            Some(raw_receipt(&fixture.images[0])),
        );
        assert!(
            admit_components(
                &cx,
                &selected,
                &fixture.binding,
                DEFAULT_SCHEMA,
                config(),
                fixture.images.clone(),
                BUDGET
            )
            .is_err(),
            "declared ANN cannot be ignored"
        );
        assert!(
            admit_components(
                &cx,
                &fixture.activation,
                &fixture.binding,
                DEFAULT_SCHEMA,
                config(),
                fixture.images.clone(),
                0
            )
            .is_err()
        );
    });
}

#[test]
fn independently_valid_lexical_image_cannot_substitute_another_live_docset() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let provider = Provider::new();
        let fixture = fixture(&cx, &provider, EmbedderTierTag::Fast).await;
        let lexical = QuillIndex::in_memory(config()).unwrap();
        lexical
            .index_documents(
                &cx,
                &[
                    IndexableDocument::new("other-one", "common other first"),
                    IndexableDocument::new("other-two", "common other second"),
                ],
            )
            .await
            .unwrap();
        lexical.commit(&cx).await.unwrap();
        let image = lexical
            .snapshot()
            .unwrap()
            .export_exact_image(&cx, source_cut(), BUDGET)
            .unwrap();
        let mut images = fixture.images.clone();
        images[1] = image.bytes();
        let mut metadata = fixture.metadata.clone();
        metadata.lexical_artifacts[0].size_bytes = image.receipt().bytes.byte_len;
        metadata.lexical_artifacts[0].checksum = lower_hex(&image.receipt().bytes.sha256);
        metadata.manifest_hash = compute_manifest_hash(&metadata).unwrap();
        images[2] = serde_json::to_vec(&metadata).unwrap().into();
        assert!(fixture.admit_resealed(&cx, images).is_err());
        assert!(
            fixture.admit(&cx).is_ok(),
            "paired original complete generation"
        );
    });
}

#[test]
fn zero_k_admits_producer_without_inference_and_cancellation_is_terminal() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let provider = Provider::new();
        let fixture = fixture(&cx, &provider, EmbedderTierTag::Fast).await;
        let readers = fixture.admit(&cx).unwrap();
        let mut foreign = Provider::new();
        foreign.identity.producer.backend = "different-producer".to_owned();
        foreign.identity.validate().unwrap();
        assert!(
            readers
                .search_text(&cx, &foreign, "common", 0)
                .await
                .is_err()
        );
        assert_eq!(foreign.calls.load(Ordering::SeqCst), 0);
        assert!(
            readers
                .search_text(&cx, &provider, "common", 0)
                .await
                .unwrap()
                .is_empty()
        );
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
        cx.set_cancel_requested(true);
        assert!(matches!(
            fixture.admit(&cx),
            Err(SearchError::Cancelled { .. })
        ));
        assert!(matches!(
            readers.search_text(&cx, &provider, "common", 2).await,
            Err(SearchError::Cancelled { .. })
        ));
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    });
}

#[cfg(target_os = "linux")]
mod qualified_rollback {
    use super::*;
    use std::fs::{self, File, OpenOptions};
    use std::io::Write as _;
    use std::os::unix::fs::{DirBuilderExt as _, OpenOptionsExt as _};
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::native_ann::builder::{NativeBuiltExactGeneration, NativeIndexBuilder};
    use crate::{GenerationActivationV1, GenerationPublisherV1};
    use frankensearch_core::generation::GenerationRootSecurityProfileV1;
    use frankensearch_index::generation_root::authority_publisher::{
        AUTHORITY_PUBLISHER_LOCK_BYTES_V1, ExpectedAuthorityPairV1, PublicationOutcomeV1,
    };
    use frankensearch_index::generation_root::generation_reader::{
        SnapshotOpenOutcomeV1, activation_manifest_path_v1, component_object_path_v1,
    };
    use frankensearch_index::generation_root::{
        GENERATION_ROOT_AUTHORITY_FILE_BYTES, GenerationRootAnchorLayout, QualifiedGenerationRoot,
    };

    const ROOT_ID: [u8; 16] = [0x81; 16];
    const PROFILE: GenerationRootSecurityProfileV1 =
        GenerationRootSecurityProfileV1::CooperativeLocal;

    fn create_root(path: &Path) -> QualifiedGenerationRoot {
        fs::DirBuilder::new().mode(0o700).create(path).unwrap();
        for (name, size) in [
            ("LOCK", AUTHORITY_PUBLISHER_LOCK_BYTES_V1),
            ("AUTHORITY", GENERATION_ROOT_AUTHORITY_FILE_BYTES),
        ] {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(path.join(name))
                .unwrap();
            file.write_all(&vec![0; usize::try_from(size).unwrap()])
                .unwrap();
            file.sync_all().unwrap();
        }
        File::open(path).unwrap().sync_all().unwrap();
        File::open(path.parent().unwrap())
            .unwrap()
            .sync_all()
            .unwrap();
        QualifiedGenerationRoot::admit(
            path,
            GenerationRootAnchorLayout::new(AUTHORITY_PUBLISHER_LOCK_BYTES_V1).unwrap(),
        )
        .expect("qualified Linux ext4/btrfs root is required")
    }

    fn stage(
        root: &QualifiedGenerationRoot,
        built: &NativeBuiltExactGeneration,
        candidate: &GenerationCandidateV1,
    ) {
        for (role, receipt) in candidate.manifest().components().iter() {
            root.stage_immutable_file(
                &component_object_path_v1(role, receipt.sha256).unwrap(),
                &built.component_bytes(role).unwrap(),
            )
            .unwrap();
        }
        stage_manifest(root, candidate);
    }

    fn stage_manifest(root: &QualifiedGenerationRoot, candidate: &GenerationCandidateV1) {
        root.stage_immutable_file(
            &activation_manifest_path_v1(candidate.authority().object_id).unwrap(),
            candidate.manifest_bytes(),
        )
        .unwrap();
    }

    fn publish(
        cx: &Cx,
        root: &QualifiedGenerationRoot,
        candidate: &GenerationCandidateV1,
        binding: &FsviV2IdentityBinding,
        pair: ExpectedAuthorityPairV1,
        writer_id: [u8; 16],
    ) -> NativeGenerationSnapshot {
        let mut publisher = GenerationPublisherV1::new(root, ROOT_ID, writer_id, PROFILE).unwrap();
        let attempt = publisher
            .prepare_attempt(candidate, pair, writer_id, writer_id)
            .unwrap();
        let result = publisher
            .publish_prepared(cx, candidate, &attempt, None, BUDGET as u64)
            .unwrap();
        assert!(matches!(
            result.publication,
            PublicationOutcomeV1::Committed { .. }
        ));
        let Some(GenerationActivationV1::Installed(opened)) = result.activation else {
            panic!("known publication must activate its complete generation");
        };
        NativeGenerationSnapshot::admit_exact(cx, opened, binding, DEFAULT_SCHEMA, config(), BUDGET)
            .unwrap()
    }

    #[test]
    fn retained_rollback_advances_authority_preserves_producer_and_fences_stale_publishers() {
        // Do not weaken qualification or skip on overlay/tmpfs: this lifecycle
        // test is deliberately run on the same real-host profile as the root.
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let parent = PathBuf::from(std::env::var_os("HOME").unwrap()).join(format!(
            ".frankensearch-native-rollback-{}-{stamp}",
            std::process::id(),
        ));
        fs::DirBuilder::new().mode(0o700).create(&parent).unwrap();
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for (name, tier) in [
                ("fast", EmbedderTierTag::Fast),
                ("quality", EmbedderTierTag::Quality),
            ] {
                let root_path = parent.join(format!("{name}-root"));
                let root = create_root(&root_path);
                let original_provider = Arc::new(Provider::new());
                let provider: Arc<dyn Embedder> = original_provider.clone();
                let original = NativeIndexBuilder::new(
                    parent.join(format!("{name}-original")),
                    ArtifactGenerationIdentityV1::new(7, [0x82; 16]).unwrap(),
                    provider,
                )
                .unwrap()
                .add_documents(
                    [IndexableDocument::new("original", "common retained source")
                        .with_metadata("version", "original")],
                )
                .build_exact_generation(&cx, tier, CommitRange { low: 1, high: 4 }, BUDGET)
                .await
                .unwrap();
                let first = original
                    .candidate(
                        [0x83; 16],
                        None,
                        GenerationAuthorityActionV1::Activate,
                        [0x84; 32],
                    )
                    .unwrap();
                stage(&root, &original, &first);
                let retained = publish(
                    &cx,
                    &root,
                    &first,
                    original.binding(),
                    ExpectedAuthorityPairV1::default(),
                    [0x85; 16],
                );
                assert!(
                    retained
                        .rollback_candidate(&cx, &retained, [0x86; 16], [0x87; 32])
                        .is_err()
                );

                let mut changed_provider = Provider::new();
                changed_provider.identity.producer.backend = "successor-producer".to_owned();
                changed_provider.identity.validate().unwrap();
                let changed_provider = Arc::new(changed_provider);
                let provider: Arc<dyn Embedder> = changed_provider.clone();
                let rebuilt = NativeIndexBuilder::new(
                    parent.join(format!("{name}-rebuilt")),
                    ArtifactGenerationIdentityV1::new(8, [0x88; 16]).unwrap(),
                    provider,
                )
                .unwrap()
                .add_documents([
                    IndexableDocument::new("successor", "common replacement source")
                        .with_metadata("version", "successor"),
                ])
                .build_exact_generation(&cx, tier, CommitRange { low: 1, high: 9 }, BUDGET)
                .await
                .unwrap();
                let second = rebuilt
                    .candidate(
                        [0x89; 16],
                        Some(first.authority()),
                        GenerationAuthorityActionV1::Activate,
                        [0x8a; 32],
                    )
                    .unwrap();
                stage(&root, &rebuilt, &second);
                let current = publish(
                    &cx,
                    &root,
                    &second,
                    rebuilt.binding(),
                    retained.retained_snapshot().authority_pair(),
                    [0x8b; 16],
                );
                assert!(
                    current
                        .rollback_candidate(&cx, &retained, [0x8c; 16], [0x8d; 32])
                        .is_err()
                );
                assert!(
                    retained
                        .rollback_candidate(&cx, &current, [0; 16], [0x8d; 32])
                        .is_err()
                );

                // Equal logical root ids and equal old authority bytes do not
                // authorize selecting components from a copied physical root.
                let foreign_root = create_root(&parent.join(format!("{name}-foreign-root")));
                stage(&foreign_root, &original, &first);
                let foreign = publish(
                    &cx,
                    &foreign_root,
                    &first,
                    original.binding(),
                    ExpectedAuthorityPairV1::default(),
                    [0x8e; 16],
                );
                assert!(
                    foreign
                        .rollback_candidate(&cx, &current, [0x8f; 16], [0x90; 32])
                        .is_err()
                );
                let SnapshotOpenOutcomeV1::Opened(read_only) = root
                    .open_generation_snapshot_with_budget(
                        ROOT_ID,
                        GenerationRootSecurityProfileV1::ReadOnlyUnanchored,
                        None,
                        BUDGET as u64,
                    )
                    .unwrap()
                else {
                    panic!("read-only complete snapshot");
                };
                let read_only = NativeGenerationSnapshot::admit_exact(
                    &cx,
                    read_only,
                    rebuilt.binding(),
                    DEFAULT_SCHEMA,
                    config(),
                    BUDGET,
                )
                .unwrap();
                assert!(
                    retained
                        .rollback_candidate(&cx, &read_only, [0x91; 16], [0x92; 32])
                        .is_err()
                );

                let before_authority = fs::read(root_path.join("AUTHORITY")).unwrap();
                let before_lock = fs::read(root_path.join("LOCK")).unwrap();
                let before_inventory = fs::read_dir(&root_path).unwrap().count();
                let original_calls = original_provider.calls.load(Ordering::SeqCst);
                let changed_calls = changed_provider.calls.load(Ordering::SeqCst);
                let rollback = retained
                    .rollback_candidate(&cx, &current, [0x93; 16], [0x94; 32])
                    .unwrap();
                let loser = retained
                    .rollback_candidate(&cx, &current, [0x95; 16], [0x96; 32])
                    .unwrap();
                assert_eq!(rollback.authority().sequence, 3);
                assert_eq!(
                    rollback.manifest().generation(),
                    original.binding().generation()
                );
                assert_eq!(rollback.manifest().predecessor(), Some(current.authority()));
                assert_eq!(
                    rollback.manifest().components(),
                    first.manifest().components()
                );
                assert_eq!(
                    rollback.manifest().source_checkpoint_sha256(),
                    first.manifest().source_checkpoint_sha256()
                );
                assert_eq!(
                    rollback.manifest().document_set_sha256(),
                    first.manifest().document_set_sha256()
                );
                let ActivationManifest::V2(manifest) = rollback.manifest() else {
                    panic!("v2 rollback");
                };
                assert_eq!(manifest.action, GenerationAuthorityActionV1::Rollback);
                assert_eq!(
                    original_provider.calls.load(Ordering::SeqCst),
                    original_calls
                );
                assert_eq!(changed_provider.calls.load(Ordering::SeqCst), changed_calls);
                assert_eq!(
                    fs::read(root_path.join("AUTHORITY")).unwrap(),
                    before_authority
                );
                assert_eq!(fs::read(root_path.join("LOCK")).unwrap(), before_lock);
                assert_eq!(fs::read_dir(&root_path).unwrap().count(), before_inventory);

                stage_manifest(&root, &rollback);
                stage_manifest(&root, &loser);
                let restored = publish(
                    &cx,
                    &root,
                    &rollback,
                    original.binding(),
                    current.retained_snapshot().authority_pair(),
                    [0x97; 16],
                );
                assert_eq!(restored.authority().sequence, 3);
                assert_eq!(
                    restored.metadata().commit_range,
                    CommitRange { low: 1, high: 4 }
                );
                let after_authority = fs::read(root_path.join("AUTHORITY")).unwrap();
                let after_lock = fs::read(root_path.join("LOCK")).unwrap();
                let mut publisher =
                    GenerationPublisherV1::new(&root, ROOT_ID, [0x98; 16], PROFILE).unwrap();
                let stale = publisher
                    .prepare_attempt(
                        &loser,
                        current.retained_snapshot().authority_pair(),
                        [0x99; 16],
                        [0x9a; 16],
                    )
                    .unwrap();
                let rejected = publisher.publish_prepared(&cx, &loser, &stale, None, BUDGET as u64);
                assert!(matches!(
                    rejected,
                    Err(crate::GenerationPublicationErrorV1::StaleExpectedAuthority)
                        | Ok(crate::GenerationPublicationResultV1 {
                            publication: PublicationOutcomeV1::NotCommitted(_),
                            ..
                        })
                ));
                assert_eq!(
                    fs::read(root_path.join("AUTHORITY")).unwrap(),
                    after_authority
                );
                assert_eq!(fs::read(root_path.join("LOCK")).unwrap(), after_lock);
                for snapshot in [&retained, &restored] {
                    let hits = snapshot
                        .search_text(&cx, original_provider.as_ref(), "common", 5)
                        .await
                        .unwrap();
                    assert_eq!(hits[0].doc_id, "original");
                    assert_eq!(hits[0].metadata.as_deref().unwrap()["version"], "original");
                    assert!(
                        snapshot
                            .search_text(&cx, changed_provider.as_ref(), "common", 0)
                            .await
                            .is_err()
                    );
                }
                let hits = current
                    .search_text(&cx, changed_provider.as_ref(), "common", 5)
                    .await
                    .unwrap();
                assert_eq!(hits[0].doc_id, "successor");
                cx.set_cancel_requested(true);
                assert!(matches!(
                    retained.rollback_candidate(&cx, &restored, [0x9b; 16], [0x9c; 32]),
                    Err(SearchError::Cancelled { .. })
                ));
                cx.set_cancel_requested(false);
                assert_eq!(
                    fs::read(root_path.join("AUTHORITY")).unwrap(),
                    after_authority
                );
                assert_eq!(fs::read(root_path.join("LOCK")).unwrap(), after_lock);
            }
        });
    }
}
