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
use frankensearch_quill::{CASS_SEMANTIC_SCHEMA, DEFAULT_SCHEMA, QuillIndex, SchemaDocument};

const BUDGET: usize = 16 * 1024 * 1024;

struct ProgressiveProvider {
    identity: EmbeddingIdentityBundleV1,
    quality: bool,
    calls: AtomicUsize,
}

impl ProgressiveProvider {
    fn new(quality: bool) -> Self {
        let name = if quality {
            "progressive-quality"
        } else {
            "progressive-fast"
        };
        let mut identity =
            EmbeddingIdentityBundleV1::explicit_test_model(name, if quality { 3 } else { 2 });
        identity.space.kind = EmbeddingSpaceKindV1::Semantic;
        identity.space.hash_control = None;
        identity.space.artifact_manifest_fingerprint = if quality { "c" } else { "a" }.repeat(64);
        identity.space.artifacts = vec![EmbeddingArtifactIdentityV1 {
            role: "weights".to_owned(),
            sha256: if quality { "d" } else { "b" }.repeat(64),
            size: 1,
        }];
        identity.producer.space_fingerprint = identity.space.fingerprint();
        identity.validate().unwrap();
        Self {
            identity,
            quality,
            calls: AtomicUsize::new(0),
        }
    }
}

impl Embedder for ProgressiveProvider {
    fn embed<'a>(&'a self, _: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            let special = text == "quality-special";
            Ok(if self.quality {
                if special || text == "unmatched" {
                    vec![1.0, 0.0, 0.0]
                } else {
                    vec![0.0, 1.0, 0.0]
                }
            } else if special {
                vec![0.0, 1.0]
            } else {
                vec![1.0, 0.0]
            })
        })
    }
    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
    }
    fn dimension(&self) -> usize {
        if self.quality { 3 } else { 2 }
    }
    fn id(&self) -> &str {
        &self.identity.space.logical_model_id
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

struct ProgressiveFixture {
    directory: tempfile::TempDir,
    fast: Arc<ProgressiveProvider>,
    quality: Arc<ProgressiveProvider>,
    fast_binding: FsviV2IdentityBinding,
    quality_binding: FsviV2IdentityBinding,
    metadata: GenerationManifest,
    activation: ActivationManifest,
    images: [Arc<[u8]>; 3],
}

impl ProgressiveFixture {
    fn admit(&self, cx: &Cx) -> SearchResult<CompositeReaders> {
        admit_progressive_components(
            cx,
            &self.activation,
            &self.fast_binding,
            &self.quality_binding,
            DEFAULT_SCHEMA,
            config(),
            self.images.clone(),
            BUDGET,
        )
    }

    fn admit_resealed(&self, cx: &Cx, images: [Arc<[u8]>; 3]) -> SearchResult<CompositeReaders> {
        let activation = activation(
            &self.fast_binding,
            &images,
            self.activation.document_set_sha256(),
            None,
        );
        admit_progressive_components(
            cx,
            &activation,
            &self.fast_binding,
            &self.quality_binding,
            DEFAULT_SCHEMA,
            config(),
            images,
            BUDGET,
        )
    }
}

async fn progressive_fixture(cx: &Cx) -> ProgressiveFixture {
    let directory = tempfile::tempdir().unwrap();
    let fast = Arc::new(ProgressiveProvider::new(false));
    let quality = Arc::new(ProgressiveProvider::new(true));
    let built = crate::native_ann::builder::NativeIndexBuilder::new(
        directory.path().join("source"),
        ArtifactGenerationIdentityV1::new(7, [0x37; 16]).unwrap(),
        fast.clone(),
    )
    .unwrap()
    .with_quality_embedder(quality.clone())
    .unwrap()
    .add_documents(["a", "b", "c", "d", "e"].into_iter().map(|id| {
        IndexableDocument::new(id, "ordinary cohort document").with_metadata("version", "old")
    }))
    .add_document(
        IndexableDocument::new("quality-only", "quality-special").with_metadata("version", "old"),
    )
    .build_progressive_generation(cx, CommitRange { low: 1, high: 4 }, BUDGET)
    .await
    .unwrap();
    let candidate = built
        .candidate(
            [0x73; 16],
            None,
            GenerationAuthorityActionV1::Activate,
            [0x74; 32],
        )
        .unwrap();
    built.validate_candidate(cx, &candidate).unwrap();
    fast.calls.store(0, Ordering::SeqCst);
    quality.calls.store(0, Ordering::SeqCst);
    ProgressiveFixture {
        directory,
        fast,
        quality,
        fast_binding: built.binding().clone(),
        quality_binding: built.quality_binding().unwrap().clone(),
        metadata: built.metadata().clone(),
        activation: candidate.manifest().clone(),
        images: [
            built
                .component_bytes(GenerationComponentRole::Vector)
                .unwrap(),
            built
                .component_bytes(GenerationComponentRole::Lexical)
                .unwrap(),
            built
                .component_bytes(GenerationComponentRole::Metadata)
                .unwrap(),
        ],
    }
}

#[test]
fn progressive_authority_retrieves_quality_beyond_fast_window_and_defers_quality_work() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fixture = progressive_fixture(&cx).await;
        let readers = fixture.admit(&cx).unwrap();
        assert_eq!(
            readers.topology(),
            NativeGenerationTopology::FullProgressive
        );
        let fast = readers
            .search_text(&cx, fixture.fast.as_ref(), "unmatched", 3)
            .await
            .unwrap();
        assert!(fast.iter().all(|hit| hit.doc_id != "quality-only"));
        let refined = readers
            .search_refined_text(
                &cx,
                fixture.fast.as_ref(),
                fixture.quality.as_ref(),
                "unmatched",
                1,
            )
            .await
            .unwrap();
        assert_eq!(refined[0].doc_id, "quality-only");
        assert_eq!(refined[0].metadata.as_ref().unwrap()["version"], "old");
        let fast_before = fixture.fast.calls.load(Ordering::SeqCst);
        let quality_before = fixture.quality.calls.load(Ordering::SeqCst);
        assert_eq!(
            readers
                .search_quality_text(&cx, fixture.quality.as_ref(), "unmatched", 1)
                .await
                .unwrap()[0]
                .doc_id,
            "quality-only"
        );
        assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), fast_before);
        assert_eq!(
            fixture.quality.calls.load(Ordering::SeqCst),
            quality_before + 1
        );

        let quality_before = fixture.quality.calls.load(Ordering::SeqCst);
        let mut stream = readers
            .search_progressive_with_quality(
                &cx,
                fixture.fast.as_ref(),
                fixture.quality.as_ref(),
                "unmatched",
                1,
            )
            .unwrap();
        assert_eq!(fixture.quality.calls.load(Ordering::SeqCst), quality_before);
        let Some(crate::native_ann::NativeSearchPhase::Initial { results, .. }) =
            stream.next_phase().await.unwrap()
        else {
            panic!("initial phase required")
        };
        assert_ne!(results[0].doc_id, "quality-only");
        assert_eq!(fixture.quality.calls.load(Ordering::SeqCst), quality_before);
        let Some(crate::native_ann::NativeSearchPhase::Refined { results, .. }) =
            stream.next_phase().await.unwrap()
        else {
            panic!("refined phase required")
        };
        assert_eq!(results[0].doc_id, "quality-only");
        assert_eq!(
            fixture.quality.calls.load(Ordering::SeqCst),
            quality_before + 1
        );
        assert!(stream.next_phase().await.unwrap().is_none());
        drop(stream);

        for (name, bytes) in ["vector", "lexical", "metadata"]
            .into_iter()
            .zip(&fixture.images)
        {
            std::fs::write(fixture.directory.path().join(name), bytes).unwrap();
        }
        let reread: [Arc<[u8]>; 3] = ["vector", "lexical", "metadata"].map(|name| {
            std::fs::read(fixture.directory.path().join(name))
                .unwrap()
                .into()
        });
        let reopened = fixture.admit_resealed(&cx, reread).unwrap();
        std::fs::rename(
            fixture.directory.path().join("source"),
            fixture.directory.path().join("relocated"),
        )
        .unwrap();
        std::fs::create_dir(fixture.directory.path().join("source")).unwrap();
        std::fs::write(
            fixture.directory.path().join("source/fast.fsvi"),
            b"unrelated bytes",
        )
        .unwrap();
        assert_eq!(
            reopened
                .search_refined_text(
                    &cx,
                    fixture.fast.as_ref(),
                    fixture.quality.as_ref(),
                    "unmatched",
                    1
                )
                .await
                .unwrap()[0]
                .doc_id,
            "quality-only"
        );
        assert_eq!(readers.source_documents(&cx, BUDGET).unwrap().len(), 6);
    });
}

#[test]
fn progressive_selected_source_uses_common_membership_and_exact_payload_budget() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fixture = progressive_fixture(&cx).await;
        let readers = fixture.admit(&cx).unwrap();
        let documents = readers.source_documents(&cx, BUDGET).unwrap();
        let expected = documents
            .iter()
            .find(|document| document.id == "quality-only")
            .unwrap();
        let metadata: BTreeMap<_, _> = expected.metadata.iter().collect();
        let payload_bytes = expected.id.len()
            + expected.content.len()
            + expected.title.as_deref().unwrap_or("").len()
            + serde_json::to_vec(&metadata).unwrap().len();
        let selected = readers
            .source_documents_by_ids(
                &cx,
                &["quality-only", "absent", "quality-only"],
                payload_bytes * 2,
            )
            .unwrap();
        assert!(selected[1].is_none());
        for document in [selected[0].as_ref().unwrap(), selected[2].as_ref().unwrap()] {
            assert_eq!(document.id, expected.id);
            assert_eq!(document.content, expected.content);
            assert_eq!(document.metadata, expected.metadata);
        }
        assert_source_refused(
            readers.source_documents_by_ids(&cx, &["quality-only"], payload_bytes - 1),
            "stored-payload-limit",
        );
        assert_source_refused(
            readers.source_documents(&cx, payload_bytes),
            "stored-payload-limit",
        );
        assert!(
            readers
                .source_documents_by_ids(&cx, &["absent"], 0)
                .unwrap()[0]
                .is_none()
        );
        assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), 0);
        assert_eq!(fixture.quality.calls.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn progressive_authority_admits_both_producers_before_any_inference_even_zero_k() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fixture = progressive_fixture(&cx).await;
        let readers = fixture.admit(&cx).unwrap();
        let mut foreign = ProgressiveProvider::new(true);
        foreign
            .identity
            .producer
            .implementation_revision
            .push_str("-foreign");
        foreign.identity.validate().unwrap();
        for k in [0, 1] {
            assert!(
                readers
                    .search_refined_text(&cx, fixture.fast.as_ref(), &foreign, "unmatched", k)
                    .await
                    .is_err()
            );
            assert!(
                readers
                    .search_progressive_with_quality(
                        &cx,
                        fixture.fast.as_ref(),
                        &foreign,
                        "unmatched",
                        k
                    )
                    .is_err()
            );
            assert!(
                readers
                    .search_quality_text(&cx, &foreign, "unmatched", k)
                    .await
                    .is_err()
            );
        }
        assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), 0);
        assert_eq!(fixture.quality.calls.load(Ordering::SeqCst), 0);
        assert_eq!(foreign.calls.load(Ordering::SeqCst), 0);
        assert!(
            readers
                .search_refined_text(
                    &cx,
                    fixture.fast.as_ref(),
                    fixture.quality.as_ref(),
                    "unmatched",
                    0
                )
                .await
                .unwrap()
                .is_empty()
        );
        assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), 0);
        cx.set_cancel_requested(true);
        assert!(matches!(
            fixture.admit(&cx),
            Err(SearchError::Cancelled { .. })
        ));
        assert!(matches!(
            readers
                .search_refined_text(
                    &cx,
                    fixture.fast.as_ref(),
                    fixture.quality.as_ref(),
                    "unmatched",
                    1
                )
                .await,
            Err(SearchError::Cancelled { .. })
        ));
        assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn progressive_authority_refuses_swapped_owners_resealed_metadata_and_component_drift() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fixture = progressive_fixture(&cx).await;
        assert!(
            admit_progressive_components(
                &cx,
                &fixture.activation,
                &fixture.quality_binding,
                &fixture.fast_binding,
                DEFAULT_SCHEMA,
                config(),
                fixture.images.clone(),
                BUDGET
            )
            .is_err()
        );
        assert!(
            admit_progressive_components(
                &cx,
                &fixture.activation,
                &fixture.fast_binding,
                &fixture.quality_binding,
                DEFAULT_SCHEMA,
                config(),
                fixture.images.clone(),
                1
            )
            .is_err()
        );
        for component in 0..3 {
            let mut images = fixture.images.clone();
            let mut bytes = images[component].to_vec();
            let last = bytes.last_mut().unwrap();
            *last ^= 1;
            images[component] = bytes.into();
            assert!(fixture.admit_resealed(&cx, images).is_err());
        }
        for change in 0..6 {
            let mut metadata = fixture.metadata.clone();
            match change {
                0 => metadata.vector_artifacts[1].checksum = "a".repeat(64),
                1 => metadata.vector_artifacts[1].vector_count += 1,
                2 => metadata.vector_artifacts[1].dimension += 1,
                3 => metadata.vector_artifacts.swap(0, 1),
                4 => {
                    metadata.embedders.insert(
                        "quality".to_owned(),
                        fixture.fast_binding.frozen_identity().clone(),
                    );
                }
                _ => metadata.total_documents += 1,
            }
            metadata.manifest_hash = compute_manifest_hash(&metadata).unwrap();
            let mut images = fixture.images.clone();
            images[2] = serde_json::to_vec(&metadata).unwrap().into();
            assert!(
                fixture.admit_resealed(&cx, images).is_err(),
                "mutation {change}"
            );
        }
        assert!(fixture.admit(&cx).is_ok());
    });
}

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
    ann: Option<Arc<[u8]>>,
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
            self.ann.clone(),
            BUDGET,
        )
    }

    fn admit_resealed(&self, cx: &Cx, images: [Arc<[u8]>; 3]) -> SearchResult<CompositeReaders> {
        let selected = activation(
            &self.binding,
            &images,
            self.activation.document_set_sha256(),
            self.ann.as_deref().map(raw_receipt),
        );
        admit_components(
            cx,
            &selected,
            &self.binding,
            DEFAULT_SCHEMA,
            config(),
            images,
            self.ann.clone(),
            BUDGET,
        )
    }

    fn reseal_metadata(&mut self) {
        self.metadata.manifest_hash = compute_manifest_hash(&self.metadata).unwrap();
        self.images[2] = serde_json::to_vec(&self.metadata).unwrap().into();
        self.activation = activation(
            &self.binding,
            &self.images,
            self.activation.document_set_sha256(),
            self.ann.as_deref().map(raw_receipt),
        );
    }

    fn with_ann(mut self, cx: &Cx, params: HnswParams, seed: u64) -> Self {
        let owner = Arc::new(
            ValidatedFsviBytes::from_arc(Arc::clone(&self.images[0]), &self.binding).unwrap(),
        );
        let graph = ValidatedNativeHnsw::build(cx, owner, params, seed).unwrap();
        let path = self.directory.path().join("source/native.fshnsw");
        let receipt = graph.save(&path).unwrap();
        self.ann = Some(std::fs::read(path).unwrap().into());
        self.metadata.metadata.insert(
            NATIVE_ANN_RECEIPT_KEY.to_owned(),
            serde_json::to_string(&receipt).unwrap(),
        );
        self.reseal_metadata();
        self
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
        ann: None,
    }
}

fn source_ids(fixture: &Fixture) -> Vec<String> {
    let owner =
        ValidatedFsviBytes::from_arc(Arc::clone(&fixture.images[0]), &fixture.binding).unwrap();
    (0..owner.record_count())
        .filter_map(|index| {
            let row = owner.row(index).unwrap();
            row.flags().is_live().then(|| row.doc_id().to_owned())
        })
        .collect()
}

fn replace_source_image(cx: &Cx, fixture: &mut Fixture, writer: &QuillIndex) {
    let image = writer
        .snapshot()
        .unwrap()
        .export_exact_image(cx, source_cut(), BUDGET)
        .unwrap();
    fixture.images[1] = image.bytes();
    fixture.metadata.lexical_artifacts[0].size_bytes = image.receipt().bytes.byte_len;
    fixture.metadata.lexical_artifacts[0].checksum = lower_hex(&image.receipt().bytes.sha256);
    fixture.reseal_metadata();
}

fn retained_lexical(cx: &Cx, snapshot: &KeeperSnapshot) -> QuillSnapshotIndex {
    let image = snapshot
        .export_exact_image(cx, source_cut(), BUDGET)
        .unwrap();
    let retained = KeeperSnapshot::from_exact_image(
        cx,
        image.bytes(),
        snapshot.schema(),
        source_cut(),
        image.receipt(),
        BUDGET,
    )
    .unwrap();
    QuillSnapshotIndex::new(cx, retained, config()).unwrap()
}

fn assert_source_refused<T>(result: SearchResult<T>, expected_code: &str) {
    let Err(SearchError::InvalidConfig { field, value, .. }) = result else {
        panic!("expected bounded retained source refusal {expected_code}");
    };
    assert_eq!(field, "native_ann.generation.source");
    assert_eq!(value, expected_code);
}

#[test]
fn selected_source_preserves_requested_order_duplicates_and_only_charges_selected_payload() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for tier in [EmbedderTierTag::Fast, EmbedderTierTag::Quality] {
            let provider = Provider::new();
            let mut fixture = fixture(&cx, &provider, tier).await;
            let ids = source_ids(&fixture);
            assert_eq!(ids, ["beta", "alpha"]);
            let expected = vec![
                IndexableDocument::new(&ids[0], "unrelated large source ".repeat(4096))
                    .with_title("Unrelated title")
                    .with_metadata("kind", "large"),
                IndexableDocument::new(&ids[1], "  café🦀\n東京\u{0}\r\n")
                    .with_title("Retained \"title\"")
                    .with_metadata("escaped", "\"\n\\\t")
                    .with_metadata("empty", ""),
            ];
            let bytes = |document: &IndexableDocument| {
                let metadata: BTreeMap<_, _> = document.metadata.iter().collect();
                document.id.len()
                    + document.content.len()
                    + document.title.as_deref().unwrap_or("").len()
                    + serde_json::to_vec(&metadata).unwrap().len()
            };
            let large_bytes = bytes(&expected[0]);
            let small_bytes = bytes(&expected[1]);
            assert!(large_bytes > small_bytes * 100);
            let writer = QuillIndex::in_memory(config()).unwrap();
            writer.index_documents(&cx, &expected).await.unwrap();
            writer.commit(&cx).await.unwrap();
            replace_source_image(&cx, &mut fixture, &writer);
            let exact = fixture.admit(&cx).unwrap();
            let fixture = fixture.with_ann(&cx, HnswParams::default(), 47);
            let ann = fixture.admit(&cx).unwrap();
            drop(writer);
            std::fs::rename(
                fixture.directory.path().join("source"),
                fixture.directory.path().join("relocated-source"),
            )
            .unwrap();
            drop(fixture);
            for readers in [exact, ann] {
                // The full cohort cannot fit, but an exact selected document can.
                assert_source_refused(
                    readers.source_documents(&cx, small_bytes),
                    "stored-payload-limit",
                );
                let selected = readers
                    .source_documents_by_ids(&cx, &[&ids[1]], small_bytes)
                    .unwrap();
                let small = selected[0].as_ref().unwrap();
                assert_eq!(small.id, expected[1].id);
                assert_eq!(small.content, expected[1].content);
                assert_eq!(small.title, expected[1].title);
                assert_eq!(small.metadata, expected[1].metadata);
                assert_source_refused(
                    readers.source_documents_by_ids(&cx, &[&ids[1]], small_bytes - 1),
                    "stored-payload-limit",
                );
                let selected = readers
                    .source_documents_by_ids(
                        &cx,
                        &[&ids[1], "absent", "dead", &ids[0], &ids[1]],
                        large_bytes + 2 * small_bytes,
                    )
                    .unwrap();
                assert_eq!(selected.len(), 5);
                assert!(selected[1].is_none() && selected[2].is_none());
                for (position, expected_position) in [(0, 1), (3, 0), (4, 1)] {
                    let actual = selected[position].as_ref().unwrap();
                    let original = &expected[expected_position];
                    assert_eq!(actual.id, original.id);
                    assert_eq!(actual.content, original.content);
                    assert_eq!(actual.title, original.title);
                    assert_eq!(actual.metadata, original.metadata);
                }
                assert_source_refused(
                    readers.source_documents_by_ids(&cx, &[&ids[1], &ids[1]], small_bytes),
                    "stored-payload-limit",
                );
                assert_source_refused(
                    readers.source_documents_by_ids(
                        &cx,
                        &[&ids[1], &ids[0]],
                        small_bytes + large_bytes - 1,
                    ),
                    "stored-payload-limit",
                );
                assert!(
                    readers
                        .source_documents_by_ids(&cx, &[], 0)
                        .unwrap()
                        .is_empty()
                );
                assert!(
                    readers
                        .source_documents_by_ids(&cx, &["absent", "dead", "absent"], 0)
                        .unwrap()
                        .iter()
                        .all(Option::is_none)
                );
            }
            assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
        }
    });
}

#[test]
fn retained_source_preserves_full_payload_live_order_and_exact_byte_budget() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for tier in [EmbedderTierTag::Fast, EmbedderTierTag::Quality] {
            let provider = Provider::new();
            let mut fixture = fixture(&cx, &provider, tier).await;
            let ids = source_ids(&fixture);
            let expected = vec![
                IndexableDocument::new(&ids[0], "common 全文\nsecond line\t\"quoted\"")
                    .with_title("完整 title\n")
                    .with_metadata("z-last", "escaped \"value\"\n")
                    .with_metadata("a-first", "日本語"),
                IndexableDocument::new(&ids[1], "")
                    .with_title("")
                    .with_metadata("empty", ""),
            ];
            let writer = QuillIndex::in_memory(config()).unwrap();
            writer.index_documents(&cx, &expected).await.unwrap();
            writer.commit(&cx).await.unwrap();
            replace_source_image(&cx, &mut fixture, &writer);
            let exact = fixture.admit(&cx).unwrap();
            let fixture = fixture.with_ann(&cx, HnswParams::default(), 21);
            let ann = fixture.admit(&cx).unwrap();
            let payload_bytes: usize = expected
                .iter()
                .map(|document| {
                    let metadata: BTreeMap<_, _> = document.metadata.iter().collect();
                    document.id.len()
                        + document.content.len()
                        + document.title.as_deref().unwrap_or("").len()
                        + serde_json::to_vec(&metadata).unwrap().len()
                })
                .sum();
            drop(writer);
            std::fs::rename(
                fixture.directory.path().join("source"),
                fixture.directory.path().join("relocated"),
            )
            .unwrap();
            drop(fixture);
            for readers in [exact, ann] {
                assert_source_refused(readers.source_documents(&cx, 0), "stored-payload-limit");
                assert_source_refused(
                    readers.source_documents(&cx, payload_bytes - 1),
                    "stored-payload-limit",
                );
                let actual = readers.source_documents(&cx, payload_bytes).unwrap();
                assert_eq!(actual.len(), expected.len());
                for (actual, expected) in actual.iter().zip(&expected) {
                    assert_eq!(actual.id, expected.id);
                    assert_eq!(actual.content, expected.content);
                    assert_eq!(actual.metadata, expected.metadata);
                    assert_eq!(
                        actual.title.as_deref(),
                        expected.title.as_deref().filter(|title| !title.is_empty())
                    );
                }
            }
            assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
        }
    });
}

#[test]
fn retained_source_refuses_missing_fields_wrong_ids_and_lossy_metadata() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let provider = Provider::new();
        let mut fixture = fixture(&cx, &provider, EmbedderTierTag::Fast).await;
        let ids = source_ids(&fixture);
        for (fault, code) in [
            ("missing-content", "missing-stored-field"),
            ("missing-title", "missing-stored-field"),
            ("missing-metadata", "missing-stored-field"),
            ("wrong-id", "stored-id-mismatch"),
            ("invalid-content", "invalid-stored-utf8"),
            ("non-string-metadata", "invalid-metadata"),
            ("invalid-metadata-utf8", "invalid-metadata"),
            ("duplicate-metadata-keys", "noncanonical-metadata"),
        ] {
            let mut documents: Vec<_> = ids
                .iter()
                .map(|id| SchemaDocument {
                    id: id.clone(),
                    indexed: vec![
                        (0, id.clone()),
                        (1, "complete source".to_owned()),
                        (2, "source title".to_owned()),
                    ],
                    stored: vec![(3, b"{}".to_vec())],
                    ..SchemaDocument::default()
                })
                .collect();
            let changed = &mut documents[0];
            match fault {
                "missing-content" => changed.indexed.retain(|(field, _)| *field != 1),
                "missing-title" => changed.indexed.retain(|(field, _)| *field != 2),
                "missing-metadata" => changed.stored.clear(),
                "wrong-id" => changed.indexed[0].1 = "not-the-idmap-identity".to_owned(),
                "invalid-content" => {
                    changed.indexed.retain(|(field, _)| *field != 1);
                    changed.stored.push((1, vec![0xff]));
                }
                "non-string-metadata" => changed.stored[0].1 = br#"{"count":42}"#.to_vec(),
                "invalid-metadata-utf8" => changed.stored[0].1 = vec![0xff],
                "duplicate-metadata-keys" => {
                    changed.stored[0].1 = br#"{"key":"first","key":"second"}"#.to_vec();
                }
                _ => unreachable!(),
            }
            let writer = QuillIndex::in_memory(config()).unwrap();
            writer
                .index_schema_documents(&cx, &documents)
                .await
                .unwrap();
            writer.commit(&cx).await.unwrap();
            replace_source_image(&cx, &mut fixture, &writer);
            let readers = fixture.admit(&cx).unwrap();
            assert_source_refused(readers.source_documents(&cx, BUDGET), code);
            // Corrupt source fields of an unrelated ID are never materialized.
            let healthy = readers
                .source_documents_by_ids(&cx, &[&ids[1]], BUDGET)
                .unwrap();
            assert_eq!(healthy[0].as_ref().unwrap().content, "complete source");
            assert_source_refused(
                readers.source_documents_by_ids(&cx, &[&ids[1], &ids[0]], BUDGET),
                code,
            );
        }
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn retained_source_checks_schema_and_complete_cohort_without_text_queries() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let provider = Provider::new();
        let fixture = fixture(&cx, &provider, EmbedderTierTag::Fast).await;
        let ids = source_ids(&fixture);
        // Keep the admitted vector/metadata fixed while exercising the export
        // census independently of the earlier component admission gate.
        let mut readers = fixture.admit(&cx).unwrap();
        readers.lexical = retained_lexical(
            &cx,
            &KeeperSnapshot::in_memory(CASS_SEMANTIC_SCHEMA).unwrap(),
        );
        assert_source_refused(readers.source_documents(&cx, BUDGET), "unsupported-schema");
        assert_source_refused(
            readers.source_documents_by_ids(&cx, &[], 0),
            "unsupported-schema",
        );
        for foreign in [false, true] {
            let writer = QuillIndex::in_memory(config()).unwrap();
            let mut documents = vec![IndexableDocument::new(&ids[0], "")];
            if foreign {
                documents.push(IndexableDocument::new("foreign", "!!!"));
            }
            writer.index_documents(&cx, &documents).await.unwrap();
            writer.commit(&cx).await.unwrap();
            readers.lexical = retained_lexical(&cx, &writer.snapshot().unwrap());
            if foreign {
                assert!(
                    readers
                        .source_documents_by_ids(&cx, &["foreign", "dead"], 0)
                        .unwrap()
                        .iter()
                        .all(Option::is_none)
                );
                assert_source_refused(
                    readers.source_documents_by_ids(&cx, &[&ids[1]], BUDGET),
                    "missing-live-document",
                );
            } else {
                assert_source_refused(
                    readers.source_documents_by_ids(&cx, &[], 0),
                    "live-count-mismatch",
                );
            }
            assert_source_refused(
                readers.source_documents(&cx, BUDGET),
                if foreign {
                    "missing-live-document"
                } else {
                    "live-count-mismatch"
                },
            );
        }
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    });
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

fn assert_admission_refused(
    result: SearchResult<CompositeReaders>,
    expected_field: &str,
    expected_code: &str,
) {
    let Err(SearchError::InvalidConfig { field, value, .. }) = result else {
        panic!("expected bounded admission refusal {expected_field}/{expected_code}");
    };
    assert_eq!(field, format!("native_ann.generation.{expected_field}"));
    assert_eq!(value, expected_code);
}

#[test]
fn authenticated_ann_searches_both_tiers_and_retains_images_after_relocation() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let params = HnswParams {
            m: 4,
            m0: 8,
            ef_construction: 24,
            ef_search: 12,
        };
        for tier in [EmbedderTierTag::Fast, EmbedderTierTag::Quality] {
            let provider = Provider::new();
            let fixture = fixture(&cx, &provider, tier).await;
            let exact = fixture.admit(&cx).unwrap();
            let expected = exact
                .search_text(&cx, &provider, "common", usize::MAX)
                .await
                .unwrap();
            let fixture = fixture.with_ann(&cx, params, 9876);
            let readers = fixture.admit(&cx).unwrap();
            assert_eq!(readers.vector.retrieval_mode(), NativeRetrievalMode::Ann);
            let NativeBuildRetrieval::Hnsw {
                params: admitted,
                seed,
            } = readers.retrieval
            else {
                panic!("declared HNSW cannot become exact retrieval");
            };
            assert_eq!(admitted, params);
            assert_eq!(seed, 9876);
            assert!(readers.components.ann().is_some());
            let mut foreign = Provider::new();
            foreign.identity.producer.backend = "foreign-ann-producer".to_owned();
            assert!(
                readers
                    .search_text(&cx, &foreign, "common", 0)
                    .await
                    .is_err()
            );
            assert_eq!(foreign.calls.load(Ordering::SeqCst), 0);

            std::fs::rename(
                fixture.directory.path().join("source"),
                fixture.directory.path().join("relocated"),
            )
            .unwrap();
            let readmitted = fixture.admit(&cx).unwrap();
            drop(fixture);
            for readers in [readers, readmitted] {
                let hits = readers
                    .search_text(&cx, &provider, "common", usize::MAX)
                    .await
                    .unwrap();
                assert_eq!(hits.len(), 2);
                for (hit, oracle) in hits.iter().zip(&expected) {
                    assert_eq!(hit.doc_id, oracle.doc_id);
                    assert_eq!(hit.score.to_bits(), oracle.score.to_bits());
                    assert_eq!(hit.source, ScoreSource::Hybrid);
                    assert_ne!(hit.doc_id, "dead");
                    assert_eq!(hit.metadata.as_deref().unwrap()["version"], "old");
                    assert_eq!(hit.fast_score.is_some(), tier == EmbedderTierTag::Fast);
                    assert_eq!(
                        hit.quality_score.is_some(),
                        tier == EmbedderTierTag::Quality
                    );
                }
            }
        }
    });
}

#[test]
fn declared_ann_requires_the_authenticated_image_and_receipt_without_fallback() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let provider = Provider::new();
        let mut fixture = fixture(&cx, &provider, EmbedderTierTag::Fast)
            .await
            .with_ann(&cx, HnswParams::default(), 42);
        let ann = fixture.ann.as_ref().unwrap().clone();
        assert!(fixture.admit(&cx).is_ok());
        assert_admission_refused(
            admit_components(
                &cx,
                &fixture.activation,
                &fixture.binding,
                DEFAULT_SCHEMA,
                config(),
                fixture.images.clone(),
                None,
                BUDGET,
            ),
            "ann",
            "component-presence-mismatch",
        );
        let mut corrupt = ann.to_vec();
        *corrupt.last_mut().unwrap() ^= 1;
        assert_admission_refused(
            admit_components(
                &cx,
                &fixture.activation,
                &fixture.binding,
                DEFAULT_SCHEMA,
                config(),
                fixture.images.clone(),
                Some(corrupt.into()),
                BUDGET,
            ),
            "component",
            "image-mismatch",
        );
        let exact_budget = fixture.activation.canonical_bytes().len()
            + fixture
                .images
                .iter()
                .map(|image| image.len())
                .sum::<usize>()
            + ann.len();
        for (budget, accepted) in [(exact_budget, true), (exact_budget - 1, false)] {
            let result = admit_components(
                &cx,
                &fixture.activation,
                &fixture.binding,
                DEFAULT_SCHEMA,
                config(),
                fixture.images.clone(),
                Some(Arc::clone(&ann)),
                budget,
            );
            if accepted {
                assert!(result.is_ok());
            } else {
                assert_admission_refused(result, "budget", "encoded-image-limit");
            }
        }
        let original_receipt = fixture.metadata.metadata[NATIVE_ANN_RECEIPT_KEY].clone();
        fixture.metadata.metadata.remove(NATIVE_ANN_RECEIPT_KEY);
        fixture.reseal_metadata();
        assert_admission_refused(
            fixture.admit(&cx),
            "ann",
            "native-receipt-presence-mismatch",
        );
        fixture
            .metadata
            .metadata
            .insert(NATIVE_ANN_RECEIPT_KEY.to_owned(), "{".to_owned());
        fixture.reseal_metadata();
        assert_admission_refused(fixture.admit(&cx), "ann", "invalid-native-receipt");
        fixture
            .metadata
            .metadata
            .insert(NATIVE_ANN_RECEIPT_KEY.to_owned(), original_receipt);
        fixture.ann = None;
        fixture.reseal_metadata();
        assert_admission_refused(
            fixture.admit(&cx),
            "ann",
            "native-receipt-presence-mismatch",
        );
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn resealed_foreign_ann_receipt_cannot_substitute_same_identity_and_document_ids() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let provider = Provider::new();
        let mut fixture = fixture(&cx, &provider, EmbedderTierTag::Fast)
            .await
            .with_ann(&cx, HnswParams::default(), 42);
        let path = fixture.directory.path().join("foreign.fsvi");
        let mut writer = VectorIndex::create_v2(&path, fixture.binding.clone()).unwrap();
        writer.write_record("alpha", &[0.0, 1.0]).unwrap();
        writer.write_record("beta", &[1.0, 0.0]).unwrap();
        writer.write_tombstone_record("dead", &[1.0, 0.0]).unwrap();
        writer.finish().unwrap();
        let foreign_bytes: Arc<[u8]> = std::fs::read(path).unwrap().into();
        assert_ne!(raw_receipt(&foreign_bytes), raw_receipt(&fixture.images[0]));
        let owner =
            Arc::new(ValidatedFsviBytes::from_arc(foreign_bytes, &fixture.binding).unwrap());
        let graph =
            ValidatedNativeHnsw::build(&cx, Arc::clone(&owner), HnswParams::default(), 42).unwrap();
        let path = fixture.directory.path().join("foreign.fshnsw");
        let receipt = graph.save(&path).unwrap();
        let bytes: Arc<[u8]> = std::fs::read(path).unwrap().into();
        assert!(ValidatedNativeHnsw::from_bytes(owner, Arc::clone(&bytes), &receipt).is_ok());
        fixture.ann = Some(bytes);
        fixture.metadata.metadata.insert(
            NATIVE_ANN_RECEIPT_KEY.to_owned(),
            serde_json::to_string(&receipt).unwrap(),
        );
        // Recompute every outer metadata and authority byte receipt. Only the
        // receipt's independently validated owner binding distinguishes this
        // otherwise valid graph with the same generation, model, IDs and counts.
        fixture.reseal_metadata();
        assert_admission_refused(fixture.admit(&cx), "ann", "native-admission-failed");
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
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
            let documents = readers.source_documents(&cx, BUDGET).unwrap();
            assert_eq!(documents.len(), 2);
            assert!(documents.iter().all(|document| document.id != "dead"
                && document.id != "successor"
                && document.content == "common retained document"
                && document.title.is_none()
                && document.metadata["version"] == "old"));
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
                    None,
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
                None,
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
                None,
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
        assert!(matches!(
            readers.source_documents(&cx, BUDGET),
            Err(SearchError::Cancelled { .. })
        ));
        assert!(matches!(
            readers.source_documents(&cx, 0),
            Err(SearchError::Cancelled { .. })
        ));
        assert!(matches!(
            readers.source_documents_by_ids(&cx, &["alpha"], BUDGET),
            Err(SearchError::Cancelled { .. })
        ));
        assert!(matches!(
            readers.source_documents_by_ids(&cx, &[], 0),
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

    use crate::native_ann::builder::{
        NativeBuildPrecision, NativeBuiltExactGeneration, NativeIndexBuilder,
    };
    use crate::{GenerationActivationV1, GenerationPublisherV1, LexicalRead};
    use frankensearch_core::generation::GenerationRootSecurityProfileV1;
    use frankensearch_index::generation_root::authority_publisher::{
        AUTHORITY_PUBLISHER_LOCK_BYTES_V1, ExpectedAuthorityPairV1, PublicationOutcomeV1,
    };
    use frankensearch_index::generation_root::generation_reader::{
        SnapshotOpenOutcomeV1, activation_manifest_path_v1, component_object_name_v1,
        component_object_path_v1,
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

    fn update_parent(label: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let parent = PathBuf::from(std::env::var_os("HOME").unwrap()).join(format!(
            ".frankensearch-native-update-{label}-{}-{stamp}",
            std::process::id(),
        ));
        fs::DirBuilder::new().mode(0o700).create(&parent).unwrap();
        parent
    }

    fn publish_fixture(
        cx: &Cx,
        root: &QualifiedGenerationRoot,
        fixture: &Fixture,
    ) -> NativeGenerationSnapshot {
        let readers = fixture.admit(cx).unwrap();
        let candidate = GenerationCandidateV1::new(
            [0xa1; 16],
            None,
            GenerationAuthorityActionV1::Activate,
            fixture.binding.generation(),
            [0xa2; 32],
            &readers.components,
        )
        .unwrap();
        for (role, receipt) in candidate.manifest().components().iter() {
            let bytes = match role {
                GenerationComponentRole::Vector => &fixture.images[0],
                GenerationComponentRole::Lexical => &fixture.images[1],
                GenerationComponentRole::Metadata => &fixture.images[2],
                GenerationComponentRole::Ann => fixture.ann.as_ref().unwrap(),
            };
            root.stage_immutable_file(
                &component_object_path_v1(role, receipt.sha256).unwrap(),
                bytes,
            )
            .unwrap();
        }
        stage_manifest(root, &candidate);
        publish(
            cx,
            root,
            &candidate,
            &fixture.binding,
            ExpectedAuthorityPairV1::default(),
            [0xa3; 16],
        )
    }

    fn stored_vector(readers: &CompositeReaders, id: &str) -> Vec<f32> {
        let owner = &readers.vector.owner;
        let physical = (0..owner.record_count())
            .find(|physical| owner.row(*physical).unwrap().doc_id() == id)
            .unwrap();
        owner.vector_at_f32(physical).unwrap()
    }

    fn publish_progressive(
        cx: &Cx,
        root: &QualifiedGenerationRoot,
        candidate: &GenerationCandidateV1,
        bindings: (&FsviV2IdentityBinding, &FsviV2IdentityBinding),
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
            panic!("complete progressive publication must activate")
        };
        NativeGenerationSnapshot::admit_progressive(
            cx,
            opened,
            bindings.0,
            bindings.1,
            DEFAULT_SCHEMA,
            config(),
            BUDGET,
        )
        .unwrap()
    }

    #[test]
    fn progressive_authority_updates_reuse_both_owners_and_rollback_the_complete_cohort() {
        let parent = update_parent("progressive");
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fixture = progressive_fixture(&cx).await;
            let root_path = parent.join("root");
            let root = create_root(&root_path);
            let components = fixture.admit(&cx).unwrap();
            let original = GenerationCandidateV1::new(
                [0xc1; 16],
                None,
                GenerationAuthorityActionV1::Activate,
                fixture.fast_binding.generation(),
                [0xc2; 32],
                &components.components,
            )
            .unwrap();
            for (role, receipt) in original.manifest().components().iter() {
                let image = match role {
                    GenerationComponentRole::Vector => &fixture.images[0],
                    GenerationComponentRole::Lexical => &fixture.images[1],
                    GenerationComponentRole::Metadata => &fixture.images[2],
                    GenerationComponentRole::Ann => panic!("unexpected graph"),
                };
                root.stage_immutable_file(
                    &component_object_path_v1(role, receipt.sha256).unwrap(),
                    image,
                )
                .unwrap();
            }
            stage_manifest(&root, &original);
            let retained = publish_progressive(
                &cx,
                &root,
                &original,
                (&fixture.fast_binding, &fixture.quality_binding),
                ExpectedAuthorityPairV1::default(),
                [0xc3; 16],
            );
            assert_eq!(
                retained.topology(),
                NativeGenerationTopology::FullProgressive
            );
            let successor_generation = ArtifactGenerationIdentityV1::new(8, [0xc4; 16]).unwrap();
            let refused = parent.join("single-refused");
            assert!(
                retained
                    .begin_update(
                        &cx,
                        &refused,
                        successor_generation,
                        fixture.fast.clone(),
                        BUDGET
                    )
                    .is_err()
            );
            assert!(!refused.exists());
            let mut foreign = ProgressiveProvider::new(true);
            foreign
                .identity
                .producer
                .implementation_revision
                .push_str("-foreign");
            let refused = parent.join("foreign-refused");
            let error = retained
                .begin_progressive_update(
                    &cx,
                    &refused,
                    successor_generation,
                    fixture.fast.clone(),
                    Arc::new(foreign),
                    0,
                )
                .err()
                .unwrap();
            assert!(error.to_string().contains("producer"));
            assert!(!refused.exists());
            assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), 0);
            assert_eq!(fixture.quality.calls.load(Ordering::SeqCst), 0);

            let authority_before = fs::read(root_path.join("AUTHORITY")).unwrap();
            let lock_before = fs::read(root_path.join("LOCK")).unwrap();
            let update = retained
                .begin_progressive_update(
                    &cx,
                    parent.join("successor"),
                    successor_generation,
                    fixture.fast.clone(),
                    fixture.quality.clone(),
                    BUDGET,
                )
                .unwrap()
                .delete_document("a")
                .upsert_document(
                    IndexableDocument::new("b", "ordinary cohort document")
                        .with_title("New title")
                        .with_metadata("version", "new"),
                )
                .upsert_document(IndexableDocument::new("inserted", "quality-special"));
            fs::rename(
                fixture.directory.path().join("source"),
                fixture.directory.path().join("relocated"),
            )
            .unwrap();
            let built = update
                .build_progressive_generation(&cx, CommitRange { low: 1, high: 8 }, BUDGET)
                .await
                .unwrap();
            assert_eq!(
                fixture.fast.calls.load(Ordering::SeqCst),
                1,
                "unchanged and metadata-only edits reuse fast vectors"
            );
            assert_eq!(
                fixture.quality.calls.load(Ordering::SeqCst),
                1,
                "quality independently reuses its own unchanged vectors"
            );
            assert_eq!(
                built.binding().frozen_identity(),
                fixture.fast_binding.frozen_identity()
            );
            assert_eq!(
                built.quality_binding().unwrap().frozen_identity(),
                fixture.quality_binding.frozen_identity()
            );
            assert_eq!(built.metadata().total_documents, 6);
            assert_eq!(
                fs::read(root_path.join("AUTHORITY")).unwrap(),
                authority_before
            );
            assert_eq!(fs::read(root_path.join("LOCK")).unwrap(), lock_before);
            let candidate = built
                .candidate(
                    [0xc5; 16],
                    Some(retained.authority()),
                    GenerationAuthorityActionV1::Activate,
                    [0xc6; 32],
                )
                .unwrap();
            built.validate_candidate(&cx, &candidate).unwrap();
            stage(&root, &built, &candidate);
            let current = publish_progressive(
                &cx,
                &root,
                &candidate,
                (built.binding(), built.quality_binding().unwrap()),
                retained.retained_snapshot().authority_pair(),
                [0xc7; 16],
            );
            let documents = current.source_documents(&cx, BUDGET).unwrap();
            assert!(!documents.iter().any(|document| document.id == "a"));
            assert!(documents.iter().any(|document| document.id == "inserted"));
            let changed = documents
                .iter()
                .find(|document| document.id == "b")
                .unwrap();
            assert_eq!(changed.title.as_deref(), Some("New title"));
            assert_eq!(changed.metadata["version"], "new");
            assert!(
                retained
                    .source_documents(&cx, BUDGET)
                    .unwrap()
                    .iter()
                    .any(|document| document.id == "a")
            );
            for (old, new) in [
                (&retained.readers.vector, &current.readers.vector),
                (
                    retained.readers.quality.as_ref().unwrap(),
                    current.readers.quality.as_ref().unwrap(),
                ),
            ] {
                let row = |index: &NativeAnnIndex| {
                    (0..index.owner.record_count())
                        .find(|physical| index.owner.row(*physical).unwrap().doc_id() == "b")
                        .unwrap()
                };
                assert_eq!(
                    old.owner.vector_at_f32(row(old)).unwrap(),
                    new.owner.vector_at_f32(row(new)).unwrap()
                );
            }
            let reopened_root = QualifiedGenerationRoot::admit(
                &root_path,
                GenerationRootAnchorLayout::new(AUTHORITY_PUBLISHER_LOCK_BYTES_V1).unwrap(),
            )
            .unwrap();
            let SnapshotOpenOutcomeV1::Opened(opened) = reopened_root
                .open_generation_snapshot_with_budget(ROOT_ID, PROFILE, None, BUDGET as u64)
                .unwrap()
            else {
                panic!("current complete cohort must reopen")
            };
            let reopened = NativeGenerationSnapshot::admit_progressive(
                &cx,
                opened,
                built.binding(),
                built.quality_binding().unwrap(),
                DEFAULT_SCHEMA,
                config(),
                BUDGET,
            )
            .unwrap();
            assert_eq!(
                reopened.topology(),
                NativeGenerationTopology::FullProgressive
            );
            let rollback = retained
                .rollback_candidate(&cx, &reopened, [0xc8; 16], [0xc9; 32])
                .unwrap();
            assert_eq!(
                rollback.manifest().components().vector,
                original.manifest().components().vector
            );
            stage_manifest(&root, &rollback);
            let restored = publish_progressive(
                &cx,
                &root,
                &rollback,
                (&fixture.fast_binding, &fixture.quality_binding),
                reopened.retained_snapshot().authority_pair(),
                [0xca; 16],
            );
            assert_eq!(
                restored.authority().sequence,
                reopened.authority().sequence + 1
            );
            assert_eq!(
                restored
                    .search_refined_text(
                        &cx,
                        fixture.fast.as_ref(),
                        fixture.quality.as_ref(),
                        "unmatched",
                        1
                    )
                    .await
                    .unwrap()[0]
                    .doc_id,
                "quality-only"
            );
            assert!(
                current
                    .source_documents(&cx, BUDGET)
                    .unwrap()
                    .iter()
                    .any(|document| document.id == "inserted")
            );
            assert!(
                !restored
                    .source_documents(&cx, BUDGET)
                    .unwrap()
                    .iter()
                    .any(|document| document.id == "inserted")
            );

            // Scopes stay on the originally retained packed cohort after both a
            // successor activation and rollback. The independently searched
            // quality owner must still promote a candidate outside the scoped
            // fast window, and neither phase may return excluded IDs.
            let retained = Arc::new(retained);
            let scoped = retained
                .scope(&cx, BUDGET, |document| Ok(document.id != "a"))
                .unwrap();
            let quality_calls = fixture.quality.calls.load(Ordering::SeqCst);
            let mut stream = scoped
                .search_progressive_with_quality(
                    &cx,
                    fixture.fast.as_ref(),
                    fixture.quality.as_ref(),
                    "unmatched",
                    1,
                )
                .unwrap();
            let Some(crate::native_ann::NativeSearchPhase::Initial {
                results,
                candidates,
            }) = stream.next_phase().await.unwrap()
            else {
                panic!("scoped Initial required")
            };
            assert_eq!(candidates.fast, 3);
            assert!(
                results
                    .iter()
                    .all(|hit| hit.doc_id != "a" && hit.doc_id != "quality-only")
            );
            assert_eq!(fixture.quality.calls.load(Ordering::SeqCst), quality_calls);
            let Some(crate::native_ann::NativeSearchPhase::Refined {
                results,
                candidates,
            }) = stream.next_phase().await.unwrap()
            else {
                panic!("scoped Refined required")
            };
            assert_eq!(candidates.quality, 3);
            assert_eq!(results[0].doc_id, "quality-only");
            assert_eq!(results[0].metadata.as_ref().unwrap()["version"], "old");
            assert!(stream.next_phase().await.unwrap().is_none());
            drop(stream);
            let fast_calls = fixture.fast.calls.load(Ordering::SeqCst);
            let quality_hits = scoped
                .search_quality_text(&cx, fixture.quality.as_ref(), "unmatched", 1)
                .await
                .unwrap();
            assert_eq!(quality_hits[0].doc_id, "quality-only");
            assert!(quality_hits[0].fast_score.is_none());
            assert_eq!(quality_hits[0].quality_score, Some(1.0));
            assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), fast_calls);

            // The lexical corpus contains several excluded matches. Filtering
            // must happen before normalization/fusion and retained hydration.
            let narrowed = scoped
                .narrow(&cx, |document| Ok(document.id == "e"))
                .unwrap();
            let results = narrowed
                .search_refined_text(
                    &cx,
                    fixture.fast.as_ref(),
                    fixture.quality.as_ref(),
                    "cohort",
                    usize::MAX,
                )
                .await
                .unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].doc_id, "e");
            assert!(results[0].lexical_score.is_some());
            assert_eq!(results[0].metadata.as_ref().unwrap()["version"], "old");
            assert!(narrowed.document("quality-only").is_none());
            assert!(scoped.document("quality-only").is_some());

            let empty = scoped.narrow(&cx, |_| Ok(false)).unwrap();
            let mut foreign = ProgressiveProvider::new(true);
            foreign
                .identity
                .producer
                .implementation_revision
                .push_str("-foreign");
            assert!(
                empty
                    .search_progressive_with_quality(
                        &cx,
                        fixture.fast.as_ref(),
                        &foreign,
                        "cohort",
                        0,
                    )
                    .is_err()
            );
            assert_eq!(foreign.calls.load(Ordering::SeqCst), 0);
        });
    }

    struct ScopeProvider(Provider);

    impl Embedder for ScopeProvider {
        fn embed<'a>(&'a self, _: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            self.0.calls.fetch_add(1, Ordering::SeqCst);
            // A deterministic test provider makes excluded documents outrank
            // eligible documents in the real retained vector owner.
            let vector = if text.starts_with("kept ") {
                vec![0.6, 0.8]
            } else {
                vec![1.0, 0.0]
            };
            Box::pin(async move { Ok(vector) })
        }

        fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
            self.0.identity()
        }

        fn dimension(&self) -> usize {
            self.0.dimension()
        }

        fn id(&self) -> &str {
            self.0.id()
        }

        fn model_name(&self) -> &str {
            self.0.model_name()
        }

        fn is_semantic(&self) -> bool {
            self.0.is_semantic()
        }

        fn category(&self) -> ModelCategory {
            self.0.category()
        }
    }

    fn assert_same_scored_results(actual: &[ScoredResult], expected: &[ScoredResult]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(actual.doc_id, expected.doc_id);
            assert_eq!(actual.score.to_bits(), expected.score.to_bits());
            assert_eq!(actual.source, expected.source);
            assert_eq!(actual.index, expected.index);
            assert_eq!(
                actual.fast_score.map(f32::to_bits),
                expected.fast_score.map(f32::to_bits)
            );
            assert_eq!(
                actual.quality_score.map(f32::to_bits),
                expected.quality_score.map(f32::to_bits)
            );
            assert_eq!(
                actual.lexical_score.map(f32::to_bits),
                expected.lexical_score.map(f32::to_bits)
            );
            assert_eq!(
                actual.rerank_score.map(f32::to_bits),
                expected.rerank_score.map(f32::to_bits)
            );
            assert_eq!(actual.metadata, expected.metadata);
            assert!(actual.explanation.is_none() && expected.explanation.is_none());
        }
    }

    #[test]
    fn authority_scopes_refill_both_candidate_lanes_and_preserve_unfiltered_results() {
        let parent = update_parent("scoped-refill");
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for (name, tier, ann) in [
                ("fast-exact", EmbedderTierTag::Fast, false),
                ("quality-exact", EmbedderTierTag::Quality, false),
                ("fast-ann", EmbedderTierTag::Fast, true),
                ("quality-ann", EmbedderTierTag::Quality, true),
            ] {
                let provider = Arc::new(ScopeProvider(Provider::new()));
                let retrieval = if ann {
                    NativeBuildRetrieval::Hnsw {
                        params: HnswParams {
                            m: 8,
                            m0: 16,
                            ef_construction: 24,
                            ef_search: 3,
                        },
                        seed: 37,
                    }
                } else {
                    NativeBuildRetrieval::Exact
                };
                let mut documents: Vec<_> = (0..8)
                    .map(|index| {
                        IndexableDocument::new(
                            format!("excluded-{index}"),
                            "needle needle needle needle",
                        )
                        .with_title("needle needle")
                        .with_metadata("tenant", "red")
                    })
                    .collect();
                for id in ["included-alpha", "included-beta"] {
                    documents.push(
                        IndexableDocument::new(
                            id,
                            format!("kept needle {}", "padding ".repeat(96)),
                        )
                        .with_title("Eligible retained title")
                        .with_metadata("tenant", "blue")
                        .with_metadata("version", "selected")
                        .with_metadata("escaped", "\"\n\\\t"),
                    );
                }
                let built = NativeIndexBuilder::new(
                    parent.join(format!("{name}-build")),
                    ArtifactGenerationIdentityV1::new(1, [0xb1; 16]).unwrap(),
                    provider.clone(),
                )
                .unwrap()
                .with_fast_storage(NativeBuildPrecision::F32, retrieval)
                .add_documents(documents.clone())
                .build_exact_generation(&cx, tier, CommitRange { low: 1, high: 10 }, BUDGET)
                .await
                .unwrap();
                let root = create_root(&parent.join(format!("{name}-root")));
                let candidate = built
                    .candidate(
                        [0xb2; 16],
                        None,
                        GenerationAuthorityActionV1::Activate,
                        [0xb3; 32],
                    )
                    .unwrap();
                // The prepared scope must consume the actual retained images,
                // including escaped metadata bytes, before anything is staged.
                fs::rename(
                    parent.join(format!("{name}-build")),
                    parent.join(format!("{name}-build-relocated")),
                )
                .unwrap();
                let payload_bytes: usize = documents
                    .iter()
                    .map(|document| {
                        let metadata: BTreeMap<_, _> = document.metadata.iter().collect();
                        document.id.len()
                            + document.content.len()
                            + document.title.as_deref().unwrap_or("").len()
                            + serde_json::to_vec(&metadata).unwrap().len()
                    })
                    .sum();
                let root_path = parent.join(format!("{name}-root"));
                let before_authority = fs::read(root_path.join("AUTHORITY")).unwrap();
                let before_lock = fs::read(root_path.join("LOCK")).unwrap();
                let before_scope_calls = provider.0.calls.load(Ordering::SeqCst);
                assert!(matches!(
                    built.prepare_scope(&cx, &candidate, payload_bytes - 1, |_| {
                        panic!("the complete source budget must be admitted first")
                    }),
                    Err(SearchError::InvalidConfig { field, value, .. })
                        if field == "native_ann.generation.source"
                            && value == "stored-payload-limit"
                ));
                let mut failed_visits = 0;
                assert!(matches!(
                    built.prepare_scope(&cx, &candidate, payload_bytes, |_| {
                        failed_visits += 1;
                        if failed_visits == documents.len() {
                            Err(refused("scope-test", "last-predicate-failed"))
                        } else {
                            Ok(true)
                        }
                    }),
                    Err(SearchError::InvalidConfig { value, .. })
                        if value == "last-predicate-failed"
                ));
                assert_eq!(failed_visits, documents.len());
                let mut cancelled_visits = 0;
                assert!(matches!(
                    built.prepare_scope(&cx, &candidate, payload_bytes, |_| {
                        cancelled_visits += 1;
                        if cancelled_visits == documents.len() {
                            cx.set_cancel_requested(true);
                        }
                        Ok(true)
                    }),
                    Err(SearchError::Cancelled { .. })
                ));
                assert_eq!(cancelled_visits, documents.len());
                assert!(matches!(
                    built.prepare_scope(&cx, &candidate, 0, |_| {
                        panic!("cancelled preparation cannot evaluate a predicate")
                    }),
                    Err(SearchError::Cancelled { .. })
                ));
                cx.set_cancel_requested(false);
                let mut prepared_visits = Vec::new();
                let prepared = built
                    .prepare_scope(&cx, &candidate, payload_bytes, |document| {
                        prepared_visits.push(document.id.clone());
                        Ok(document.metadata.get("tenant").map(String::as_str) == Some("blue"))
                    })
                    .unwrap();
                assert_eq!(prepared_visits.len(), documents.len());
                // Same source, generation, tier, producer and component receipts
                // cannot authorize attaching another candidate's membership.
                let other_candidate = built
                    .candidate(
                        [0xb5; 16],
                        None,
                        GenerationAuthorityActionV1::Activate,
                        [0xb6; 32],
                    )
                    .unwrap();
                let wrong_prepared = built
                    .prepare_scope(&cx, &other_candidate, payload_bytes, |_| Ok(true))
                    .unwrap();
                assert_eq!(provider.0.calls.load(Ordering::SeqCst), before_scope_calls);
                assert_eq!(
                    fs::read(root_path.join("AUTHORITY")).unwrap(),
                    before_authority
                );
                assert_eq!(fs::read(root_path.join("LOCK")).unwrap(), before_lock);
                stage(&root, &built, &candidate);
                let retained = Arc::new(publish(
                    &cx,
                    &root,
                    &candidate,
                    built.binding(),
                    ExpectedAuthorityPairV1::default(),
                    [0xb4; 16],
                ));
                assert_eq!(retained.retrieval_mode(), built.retrieval_mode());
                assert!(matches!(
                    wrong_prepared.bind(Arc::clone(&retained)),
                    Err(SearchError::InvalidConfig { field, value, .. })
                        if field == "native_ann.scope.prepared" && value == "candidate-mismatch"
                ));
                // A cancellation arriving after known publication cannot abort
                // binding the already-prepared exact selection.
                cx.set_cancel_requested(true);
                let prepared_scope = prepared.bind(Arc::clone(&retained)).unwrap();
                cx.set_cancel_requested(false);
                assert_eq!(prepared_visits.len(), documents.len());
                assert_eq!(prepared_scope.len(), 2);
                assert!(prepared_scope.document("excluded-0").is_none());
                assert_eq!(provider.0.calls.load(Ordering::SeqCst), before_scope_calls);
                let calls = provider.0.calls.load(Ordering::SeqCst);
                let mut visited = Vec::new();
                let scope = retained
                    .scope(&cx, BUDGET, |document| {
                        visited.push(document.id.clone());
                        Ok(
                            document.metadata.get("tenant").map(String::as_str) == Some("blue")
                                && document.title.as_deref() == Some("Eligible retained title"),
                        )
                    })
                    .unwrap();
                assert_eq!(visited.len(), documents.len());
                assert_eq!(scope.len(), 2);
                assert!(!scope.is_empty());
                assert_eq!(provider.0.calls.load(Ordering::SeqCst), calls);
                assert!(scope.document("excluded-0").is_none());
                for expected in documents
                    .iter()
                    .filter(|doc| doc.id.starts_with("included-"))
                {
                    let actual = scope.document(&expected.id).unwrap();
                    assert_eq!(actual.content, expected.content);
                    assert_eq!(actual.title, expected.title);
                    assert_eq!(actual.metadata, expected.metadata);
                    let prepared_document = prepared_scope.document(&expected.id).unwrap();
                    assert_eq!(prepared_document.content, expected.content);
                    assert_eq!(prepared_document.title, expected.title);
                    assert_eq!(prepared_document.metadata, expected.metadata);
                }

                // Neither ordinary source window contains an eligible row.
                // Truncating then filtering either lane therefore loses data.
                let vectors = retained
                    .readers
                    .vector
                    .search_text(&cx, provider.as_ref(), "needle", 6, None)
                    .await
                    .unwrap();
                assert_eq!(vectors.len(), 6);
                assert!(
                    vectors
                        .iter()
                        .all(|hit| hit.doc_id.starts_with("excluded-"))
                );
                let lexical = retained
                    .readers
                    .lexical
                    .search_candidates(&cx, "needle", 6)
                    .await
                    .unwrap();
                assert_eq!(lexical.results().len(), 6);
                assert!(
                    lexical
                        .results()
                        .iter()
                        .all(|hit| hit.doc_id.starts_with("excluded-"))
                );
                let filtered = scope
                    .search_text(&cx, provider.as_ref(), "needle", 2)
                    .await
                    .unwrap();
                assert_eq!(filtered.len(), 2);
                assert_same_scored_results(
                    &prepared_scope
                        .search_text(&cx, provider.as_ref(), "needle", 2)
                        .await
                        .unwrap(),
                    &filtered,
                );
                for hit in &filtered {
                    assert!(hit.doc_id.starts_with("included-"));
                    assert_eq!(hit.source, ScoreSource::Hybrid);
                    assert!(hit.index.is_some() && hit.lexical_score.is_some());
                    assert_eq!(hit.fast_score.is_some(), tier == EmbedderTierTag::Fast);
                    assert_eq!(
                        hit.quality_score.is_some(),
                        tier == EmbedderTierTag::Quality
                    );
                    assert_eq!(hit.metadata.as_deref().unwrap()["tenant"], "blue");
                    let row = retained
                        .readers
                        .vector
                        .owner
                        .row(usize::try_from(hit.index.unwrap()).unwrap())
                        .unwrap();
                    assert_eq!(row.doc_id(), hit.doc_id.as_str());
                }
                assert_same_scored_results(
                    &scope
                        .search_text(&cx, provider.as_ref(), "needle", usize::MAX)
                        .await
                        .unwrap(),
                    &filtered,
                );
                let mut narrowed_ids = Vec::new();
                let narrowed = scope
                    .narrow(&cx, |document| {
                        narrowed_ids.push(document.id.clone());
                        Ok(document.id == "included-beta")
                    })
                    .unwrap();
                assert_eq!(narrowed_ids.len(), 2);
                assert!(narrowed_ids.iter().all(|id| id.starts_with("included-")));
                assert_eq!(scope.len(), 2);
                assert_eq!(narrowed.len(), 1);
                assert!(narrowed.document("included-alpha").is_none());
                assert_eq!(
                    narrowed
                        .narrow(&cx, |_| Ok(true))
                        .unwrap()
                        .search_text(&cx, provider.as_ref(), "needle", usize::MAX)
                        .await
                        .unwrap()[0]
                        .doc_id,
                    "included-beta"
                );
                let empty = narrowed
                    .narrow(&cx, |document| Ok(document.id == "included-alpha"))
                    .unwrap();
                assert!(empty.is_empty());
                assert!(
                    empty
                        .narrow(&cx, |_| panic!("empty membership cannot widen"))
                        .unwrap()
                        .is_empty()
                );
                let before_empty = provider.0.calls.load(Ordering::SeqCst);
                assert!(
                    empty
                        .search_text(&cx, provider.as_ref(), "needle", 1)
                        .await
                        .unwrap()
                        .is_empty()
                );
                assert_eq!(provider.0.calls.load(Ordering::SeqCst), before_empty);

                let all = retained.scope(&cx, BUDGET, |_| Ok(true)).unwrap();
                assert_eq!(all.len(), documents.len());
                for text in ["needle", "absent-term"] {
                    for k in [0, 1, 2, usize::MAX] {
                        let expected = retained
                            .search_text(&cx, provider.as_ref(), text, k)
                            .await
                            .unwrap();
                        let actual = all
                            .search_text(&cx, provider.as_ref(), text, k)
                            .await
                            .unwrap();
                        assert_same_scored_results(&actual, &expected);
                    }
                }
                let pin = Arc::downgrade(&retained);
                drop(retained);
                assert!(
                    pin.upgrade().is_some(),
                    "scoped readers retain the admitted generation"
                );
                drop(all);
                drop(empty);
                drop(narrowed);
                drop(scope);
                drop(prepared_scope);
                assert!(pin.upgrade().is_none());
            }
        });
    }

    #[test]
    fn authority_scopes_refuse_partial_admission_and_validate_empty_queries() {
        let parent = update_parent("scoped-refusals");
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let provider = Provider::new();
            let fixture = fixture(&cx, &provider, EmbedderTierTag::Fast).await;
            let root = create_root(&parent.join("root"));
            let retained = Arc::new(publish_fixture(&cx, &root, &fixture));
            let documents = retained.source_documents(&cx, BUDGET).unwrap();
            let payload_bytes: usize = documents
                .iter()
                .map(|document| {
                    let metadata: BTreeMap<_, _> = document.metadata.iter().collect();
                    document.id.len()
                        + document.content.len()
                        + document.title.as_deref().unwrap_or("").len()
                        + serde_json::to_vec(&metadata).unwrap().len()
                })
                .sum();
            let mut predicate_calls = 0;
            let result = retained.scope(&cx, payload_bytes - 1, |_| {
                predicate_calls += 1;
                Ok(false)
            });
            assert!(matches!(
                result,
                Err(SearchError::InvalidConfig { value, .. }) if value == "stored-payload-limit"
            ));
            assert_eq!(
                predicate_calls, 0,
                "selection cannot bypass the whole-source budget"
            );
            let all = retained.scope(&cx, payload_bytes, |_| Ok(true)).unwrap();
            assert_eq!(all.len(), 2);
            assert!(all.document("dead").is_none());

            let mut admitted = 0;
            let result = retained.scope(&cx, payload_bytes, |_| {
                admitted += 1;
                if admitted == 2 {
                    Err(refused("scope-test", "predicate-failed"))
                } else {
                    Ok(true)
                }
            });
            assert!(matches!(
                result,
                Err(SearchError::InvalidConfig { value, .. }) if value == "predicate-failed"
            ));
            assert_eq!(admitted, 2);
            let mut narrowed = 0;
            let result = all.narrow(&cx, |_| {
                narrowed += 1;
                if narrowed == 2 {
                    Err(refused("scope-test", "narrowing-failed"))
                } else {
                    Ok(true)
                }
            });
            assert!(matches!(
                result,
                Err(SearchError::InvalidConfig { value, .. }) if value == "narrowing-failed"
            ));
            assert_eq!(narrowed, 2);
            assert_eq!(
                all.len(),
                2,
                "failed narrowing leaves the original view intact"
            );

            let empty = all.narrow(&cx, |_| Ok(false)).unwrap();
            assert!(empty.is_empty());
            let mut foreign = Provider::new();
            foreign.identity.producer.backend = "foreign-scoped-producer".to_owned();
            foreign.identity.validate().unwrap();
            assert!(empty.search_text(&cx, &foreign, "common", 1).await.is_err());
            assert!(all.search_text(&cx, &foreign, "common", 0).await.is_err());
            assert!(
                empty
                    .search_text(&cx, &provider, "common", 1)
                    .await
                    .unwrap()
                    .is_empty()
            );
            assert!(
                all.search_text(&cx, &provider, "common", 0)
                    .await
                    .unwrap()
                    .is_empty()
            );
            assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
            assert_eq!(foreign.calls.load(Ordering::SeqCst), 0);

            cx.set_cancel_requested(true);
            assert!(matches!(
                retained.scope(&cx, 0, |_| panic!(
                    "cancelled scope must not call predicates"
                )),
                Err(SearchError::Cancelled { .. })
            ));
            assert!(matches!(
                all.narrow(&cx, |_| panic!(
                    "cancelled narrowing must not call predicates"
                )),
                Err(SearchError::Cancelled { .. })
            ));
            assert!(matches!(
                empty.search_text(&cx, &provider, "common", 0).await,
                Err(SearchError::Cancelled { .. })
            ));
            cx.set_cancel_requested(false);
            let mut visited = 0;
            assert!(matches!(
                retained.scope(&cx, payload_bytes, |_| {
                    visited += 1;
                    if visited == documents.len() {
                        cx.set_cancel_requested(true);
                    }
                    Ok(true)
                }),
                Err(SearchError::Cancelled { .. })
            ));
            assert_eq!(visited, documents.len());
            cx.set_cancel_requested(false);
            let mut visited = 0;
            assert!(matches!(
                all.narrow(&cx, |_| {
                    visited += 1;
                    if visited == all.len() {
                        cx.set_cancel_requested(true);
                    }
                    Ok(true)
                }),
                Err(SearchError::Cancelled { .. })
            ));
            assert_eq!(visited, all.len());
            cx.set_cancel_requested(false);
            assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
            assert_eq!(all.len(), 2);
            let hits = all
                .search_text(&cx, &provider, "common", usize::MAX)
                .await
                .unwrap();
            assert_eq!(hits.len(), 2);
            assert!(hits.iter().all(|hit| hit.doc_id != "dead"));
            assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        });
    }

    #[test]
    fn authority_updates_reuse_live_rows_and_preserve_searchable_predecessors() {
        let parent = update_parent("live-rows");
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for (name, tier, ann) in [
                ("fast-exact", EmbedderTierTag::Fast, false),
                ("quality-exact", EmbedderTierTag::Quality, false),
                ("fast-ann", EmbedderTierTag::Fast, true),
                ("quality-ann", EmbedderTierTag::Quality, true),
            ] {
                let provider = Arc::new(Provider::new());
                let mut fixture = fixture(&cx, provider.as_ref(), tier).await;
                // Physical FSVI order differs from the source-ID order used by
                // incremental merging, and includes an additional dead row.
                assert_eq!(source_ids(&fixture), ["beta", "alpha"]);
                let params = HnswParams {
                    m: 4,
                    m0: 8,
                    ef_construction: 24,
                    ef_search: 17,
                };
                if ann {
                    fixture = fixture.with_ann(&cx, params, 123);
                }
                let root_path = parent.join(format!("{name}-root"));
                let root = create_root(&root_path);
                let retained = publish_fixture(&cx, &root, &fixture);
                let original_vector = stored_vector(&retained.readers, "beta");
                assert_eq!(original_vector, [0.8, 0.6]);
                let old_pair = retained.retained_snapshot().authority_pair();
                let old_authority = retained.authority();
                let authority_bytes = fs::read(root_path.join("AUTHORITY")).unwrap();
                let lock_bytes = fs::read(root_path.join("LOCK")).unwrap();
                let update = retained
                    .begin_update(
                        &cx,
                        parent.join(format!("{name}-successor")),
                        ArtifactGenerationIdentityV1::new(8, [0xa4; 16]).unwrap(),
                        provider.clone(),
                        BUDGET,
                    )
                    .unwrap()
                    .delete_document("alpha")
                    .upsert_document(
                        IndexableDocument::new("beta", "common retained document")
                            .with_title("Updated title")
                            .with_metadata("version", "new"),
                    )
                    .upsert_document(IndexableDocument::new("gamma", "common inserted source"));
                // The predecessor's paths no longer exist; every reused byte
                // and full source payload must come from its retained handles.
                fs::rename(
                    fixture.directory.path().join("source"),
                    fixture.directory.path().join("relocated"),
                )
                .unwrap();
                let built = update
                    .build_exact_generation(&cx, tier, CommitRange { low: 1, high: 8 }, BUDGET)
                    .await
                    .unwrap();
                assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
                assert_eq!(built.metadata().total_documents, 2);
                assert_eq!(built.metadata().vector_artifacts[0].vector_count, 2);
                assert_eq!(
                    built.binding().frozen_identity(),
                    fixture.binding.frozen_identity()
                );
                assert_eq!(
                    fs::read(root_path.join("AUTHORITY")).unwrap(),
                    authority_bytes
                );
                assert_eq!(fs::read(root_path.join("LOCK")).unwrap(), lock_bytes);
                let candidate = built
                    .candidate(
                        [0xa5; 16],
                        Some(old_authority),
                        GenerationAuthorityActionV1::Activate,
                        [0xa6; 32],
                    )
                    .unwrap();
                built.validate_candidate(&cx, &candidate).unwrap();
                assert_eq!(candidate.manifest().components().ann.is_some(), ann);
                stage(&root, &built, &candidate);
                let current = publish(
                    &cx,
                    &root,
                    &candidate,
                    built.binding(),
                    old_pair,
                    [0xa7; 16],
                );
                assert_eq!(stored_vector(&current.readers, "beta"), original_vector);
                assert_eq!(stored_vector(&current.readers, "gamma"), [1.0, 0.0]);
                match current.retrieval_configuration() {
                    NativeBuildRetrieval::Exact => assert!(!ann),
                    NativeBuildRetrieval::Hnsw {
                        params: actual,
                        seed,
                    } => {
                        assert!(ann);
                        assert_eq!(actual, params);
                        assert_eq!(seed, 123);
                    }
                }
                let documents = current.source_documents(&cx, BUDGET).unwrap();
                assert_eq!(documents.len(), 2);
                let beta = documents.iter().find(|doc| doc.id == "beta").unwrap();
                assert_eq!(beta.content, "common retained document");
                assert_eq!(beta.title.as_deref(), Some("Updated title"));
                assert_eq!(beta.metadata["version"], "new");
                assert!(
                    !documents
                        .iter()
                        .any(|doc| doc.id == "alpha" || doc.id == "dead")
                );

                let calls = provider.calls.load(Ordering::SeqCst);
                let changed = current
                    .begin_update(
                        &cx,
                        parent.join(format!("{name}-changed")),
                        ArtifactGenerationIdentityV1::new(9, [0xa8; 16]).unwrap(),
                        provider.clone(),
                        BUDGET,
                    )
                    .unwrap()
                    .upsert_document(IndexableDocument::new("gamma", "changed complete source"))
                    .build_exact_generation(&cx, tier, CommitRange { low: 2, high: 9 }, BUDGET)
                    .await
                    .unwrap();
                assert_eq!(provider.calls.load(Ordering::SeqCst), calls + 1);
                let changed_candidate = changed
                    .candidate(
                        [0xa9; 16],
                        Some(current.authority()),
                        GenerationAuthorityActionV1::Activate,
                        [0xaa; 32],
                    )
                    .unwrap();
                changed.validate_candidate(&cx, &changed_candidate).unwrap();
                let changed_readers = admit_components(
                    &cx,
                    changed_candidate.manifest(),
                    changed.binding(),
                    DEFAULT_SCHEMA,
                    config(),
                    [
                        GenerationComponentRole::Vector,
                        GenerationComponentRole::Lexical,
                        GenerationComponentRole::Metadata,
                    ]
                    .map(|role| changed.component_bytes(role).unwrap()),
                    changed.component_bytes(GenerationComponentRole::Ann),
                    BUDGET,
                )
                .unwrap();
                assert_eq!(stored_vector(&changed_readers, "beta"), original_vector);
                assert!(
                    changed_readers
                        .source_documents(&cx, BUDGET)
                        .unwrap()
                        .iter()
                        .any(|doc| {
                            doc.id == "gamma" && doc.content == "changed complete source"
                        })
                );
                let old_hits = retained
                    .search_text(&cx, provider.as_ref(), "common", 5)
                    .await
                    .unwrap();
                let new_hits = current
                    .search_text(&cx, provider.as_ref(), "common", 5)
                    .await
                    .unwrap();
                assert!(old_hits.iter().any(|hit| hit.doc_id == "alpha"));
                assert!(
                    !old_hits
                        .iter()
                        .any(|hit| hit.doc_id == "gamma" || hit.doc_id == "dead")
                );
                assert!(new_hits.iter().any(|hit| hit.doc_id == "gamma"));
                assert!(
                    !new_hits
                        .iter()
                        .any(|hit| hit.doc_id == "alpha" || hit.doc_id == "dead")
                );
                assert_eq!(
                    retained.metadata().commit_range,
                    CommitRange { low: 1, high: 4 }
                );
                assert_eq!(
                    current.metadata().commit_range,
                    CommitRange { low: 1, high: 8 }
                );

                let pin = Arc::downgrade(retained.retained_snapshot());
                let pending = retained
                    .begin_update(
                        &cx,
                        parent.join(format!("{name}-held")),
                        ArtifactGenerationIdentityV1::new(10, [0xab; 16]).unwrap(),
                        provider.clone(),
                        BUDGET,
                    )
                    .unwrap();
                drop(retained);
                assert!(
                    pin.upgrade().is_some(),
                    "pending updates retain the admitted authority"
                );
                drop(pending);
                assert!(pin.upgrade().is_none());
            }
        });
    }

    #[test]
    fn authority_updates_refuse_invalid_contracts_before_inference_or_work() {
        let parent = update_parent("refusals");
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let provider = Arc::new(Provider::new());
            let fixture = fixture(&cx, provider.as_ref(), EmbedderTierTag::Fast).await;
            let root_path = parent.join("root");
            let root = create_root(&root_path);
            let retained = publish_fixture(&cx, &root, &fixture);
            let work = parent.join("must-not-exist");
            let newer = ArtifactGenerationIdentityV1::new(8, [0xac; 16]).unwrap();
            let authority = fs::read(root_path.join("AUTHORITY")).unwrap();
            let lock = fs::read(root_path.join("LOCK")).unwrap();
            for sequence in [1, 7] {
                assert!(
                    retained
                        .begin_update(
                            &cx,
                            &work,
                            ArtifactGenerationIdentityV1::new(sequence, [0xad; 16]).unwrap(),
                            provider.clone(),
                            BUDGET,
                        )
                        .is_err()
                );
            }
            for changed in ["producer", "input", "storage"] {
                let mut foreign = Provider::new();
                match changed {
                    "producer" => foreign.identity.producer.backend = "different".to_owned(),
                    "input" => foreign.identity.input.content_selection = "different".to_owned(),
                    _ => foreign.identity.storage.vector_normalization = "different".to_owned(),
                }
                // Each foreign bundle must be internally valid so the test
                // reaches predecessor admission rather than identity parsing.
                foreign.identity.space.input_contract_fingerprint =
                    foreign.identity.input.fingerprint();
                foreign.identity.space.output_normalization =
                    foreign.identity.storage.vector_normalization.clone();
                foreign.identity.producer.space_fingerprint = foreign.identity.space.fingerprint();
                foreign.identity.validate().unwrap();
                let foreign = Arc::new(foreign);
                assert!(matches!(
                    retained.begin_update(&cx, &work, newer, foreign.clone(), BUDGET),
                    Err(SearchError::InvalidConfig { field, .. }) if field == "native_ann.update.producer"
                ));
                assert_eq!(foreign.calls.load(Ordering::SeqCst), 0);
            }
            assert!(
                retained
                    .begin_update(&cx, &work, newer, provider.clone(), 0)
                    .is_err()
            );
            for (tier, source, budget, empty) in [
                (
                    EmbedderTierTag::Quality,
                    CommitRange { low: 1, high: 8 },
                    BUDGET,
                    false,
                ),
                (
                    EmbedderTierTag::Fast,
                    CommitRange { low: 0, high: 8 },
                    BUDGET,
                    false,
                ),
                (
                    EmbedderTierTag::Fast,
                    CommitRange { low: 1, high: 3 },
                    BUDGET,
                    false,
                ),
                (
                    EmbedderTierTag::Fast,
                    CommitRange { low: 9, high: 8 },
                    BUDGET,
                    false,
                ),
                (
                    EmbedderTierTag::Fast,
                    CommitRange { low: 1, high: 8 },
                    BUDGET,
                    true,
                ),
                (
                    EmbedderTierTag::Fast,
                    CommitRange { low: 1, high: 8 },
                    0,
                    false,
                ),
            ] {
                let mut update = retained
                    .begin_update(&cx, &work, newer, provider.clone(), BUDGET)
                    .unwrap();
                if empty {
                    update = update.delete_document("alpha").delete_document("beta");
                } else {
                    update = update.upsert_document(IndexableDocument::new("gamma", "new source"));
                }
                assert!(
                    update
                        .build_exact_generation(&cx, tier, source, budget)
                        .await
                        .is_err()
                );
                assert!(!work.exists());
                assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
            }
            let pending = retained
                .begin_update(&cx, &work, newer, provider.clone(), BUDGET)
                .unwrap()
                .upsert_document(IndexableDocument::new("gamma", "new source"));
            cx.set_cancel_requested(true);
            assert!(matches!(
                retained.begin_update(&cx, &work, newer, provider.clone(), BUDGET),
                Err(SearchError::Cancelled { .. })
            ));
            assert!(matches!(
                pending
                    .build_exact_generation(
                        &cx,
                        EmbedderTierTag::Fast,
                        CommitRange { low: 1, high: 8 },
                        BUDGET
                    )
                    .await,
                Err(SearchError::Cancelled { .. })
            ));
            cx.set_cancel_requested(false);
            let stopped = asupersync::runtime::blocking_pool::BlockingPool::new(0, 1);
            let handle = stopped.handle();
            assert!(stopped.shutdown_and_wait(std::time::Duration::from_secs(10)));
            let stopped_cx = cx.clone().with_blocking_pool_handle(Some(handle));
            assert!(matches!(
                retained.begin_update(&stopped_cx, &work, newer, provider.clone(), BUDGET),
                Err(SearchError::SubsystemError {
                    subsystem: "native_ann.build_worker",
                    ..
                })
            ));
            assert!(!work.exists());
            assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
            assert_eq!(fs::read(root_path.join("AUTHORITY")).unwrap(), authority);
            assert_eq!(fs::read(root_path.join("LOCK")).unwrap(), lock);
        });
    }

    fn assert_selected_rollback_refused(
        result: SearchResult<GenerationCandidateV1>,
        expected_code: &str,
    ) {
        let Err(SearchError::InvalidConfig { field, value, .. }) = result else {
            panic!("expected selected predecessor rollback refusal {expected_code}");
        };
        assert_eq!(field, "native_ann.generation.selected_rollback");
        assert_eq!(value, expected_code);
    }

    #[test]
    fn selected_predecessor_recovers_corrupt_current_components_without_serving_them() {
        let parent = update_parent("damaged-current");
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for (name, tier, ann) in [
                ("fast-exact", EmbedderTierTag::Fast, false),
                ("quality-exact", EmbedderTierTag::Quality, false),
                ("fast-ann", EmbedderTierTag::Fast, true),
                ("quality-ann", EmbedderTierTag::Quality, true),
            ] {
                let retrieval = if ann {
                    NativeBuildRetrieval::Hnsw {
                        params: HnswParams {
                            m: 4,
                            m0: 8,
                            ef_construction: 24,
                            ef_search: 17,
                        },
                        seed: 59,
                    }
                } else {
                    NativeBuildRetrieval::Exact
                };
                let original_provider = Arc::new(Provider::new());
                let original = NativeIndexBuilder::new(
                    parent.join(format!("{name}-original")),
                    ArtifactGenerationIdentityV1::new(7, [0xb5; 16]).unwrap(),
                    original_provider.clone(),
                )
                .unwrap()
                .with_fast_storage(NativeBuildPrecision::F32, retrieval)
                .add_documents([
                    IndexableDocument::new("original", "common intact predecessor")
                        .with_title("Original healthy title")
                        .with_metadata("version", "original"),
                ])
                .build_exact_generation(&cx, tier, CommitRange { low: 1, high: 4 }, BUDGET)
                .await
                .unwrap();
                let first = original
                    .candidate(
                        [0xb6; 16],
                        None,
                        GenerationAuthorityActionV1::Activate,
                        [0xb7; 32],
                    )
                    .unwrap();
                let mut changed_provider = Provider::new();
                changed_provider.identity.producer.backend = "damaged-current-producer".to_owned();
                changed_provider.identity.validate().unwrap();
                let changed_provider = Arc::new(changed_provider);
                let rebuilt = NativeIndexBuilder::new(
                    parent.join(format!("{name}-successor")),
                    ArtifactGenerationIdentityV1::new(8, [0xb8; 16]).unwrap(),
                    changed_provider.clone(),
                )
                .unwrap()
                .with_fast_storage(NativeBuildPrecision::F32, retrieval)
                .add_documents([
                    IndexableDocument::new("successor", "common later generation")
                        .with_metadata("version", "successor"),
                ])
                .build_exact_generation(&cx, tier, CommitRange { low: 1, high: 9 }, BUDGET)
                .await
                .unwrap();
                let second = rebuilt
                    .candidate(
                        [0xb9; 16],
                        Some(first.authority()),
                        GenerationAuthorityActionV1::Activate,
                        [0xba; 32],
                    )
                    .unwrap();

                for damaged_role in [
                    GenerationComponentRole::Vector,
                    GenerationComponentRole::Lexical,
                    GenerationComponentRole::Metadata,
                ]
                .into_iter()
                .chain(ann.then_some(GenerationComponentRole::Ann))
                {
                    let root_path = parent.join(format!("{name}-{damaged_role:?}-root"));
                    let root = create_root(&root_path);
                    stage(&root, &original, &first);
                    let retained = publish(
                        &cx,
                        &root,
                        &first,
                        original.binding(),
                        ExpectedAuthorityPairV1::default(),
                        [0xbb; 16],
                    );
                    stage(&root, &rebuilt, &second);
                    let current = publish(
                        &cx,
                        &root,
                        &second,
                        rebuilt.binding(),
                        retained.retained_snapshot().authority_pair(),
                        [0xbc; 16],
                    );
                    let expected_current_hits = current
                        .search_text(&cx, changed_provider.as_ref(), "common", 1)
                        .await
                        .unwrap();
                    for snapshot in [&retained, &current] {
                        assert_selected_rollback_refused(
                            snapshot.rollback_selected_predecessor_candidate(
                                &cx, [0xbd; 16], [0xbe; 32],
                            ),
                            "missing-selection-provenance",
                        );
                    }
                    let receipt = second
                        .manifest()
                        .components()
                        .iter()
                        .find(|(role, _)| *role == damaged_role)
                        .unwrap()
                        .1;
                    let damaged_name = component_object_name_v1(damaged_role, receipt.sha256);
                    let damaged_path = root_path.join(&damaged_name);
                    let preserved_path = parent.join(format!("{name}-{damaged_role:?}-preserved"));
                    // Fault injection affects only this test's generated file.
                    // Preserve its healthy inode outside the qualified root;
                    // create a corrupt replacement without overwriting it.
                    fs::rename(&damaged_path, &preserved_path).unwrap();
                    let mut damaged_bytes = rebuilt.component_bytes(damaged_role).unwrap().to_vec();
                    damaged_bytes[0] ^= 0x80;
                    let mut damaged_file = OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .mode(0o400)
                        .open(&damaged_path)
                        .unwrap();
                    damaged_file.write_all(&damaged_bytes).unwrap();
                    damaged_file.sync_all().unwrap();
                    drop(damaged_file);
                    File::open(&root_path).unwrap().sync_all().unwrap();
                    drop(root);
                    let root = QualifiedGenerationRoot::admit(
                        &root_path,
                        GenerationRootAnchorLayout::new(AUTHORITY_PUBLISHER_LOCK_BYTES_V1).unwrap(),
                    )
                    .unwrap();
                    assert!(matches!(
                        root.open_generation_snapshot_with_budget(
                            ROOT_ID,
                            PROFILE,
                            None,
                            BUDGET as u64
                        )
                        .unwrap(),
                        SnapshotOpenOutcomeV1::Refused(_)
                    ));
                    let before_authority = fs::read(root_path.join("AUTHORITY")).unwrap();
                    let before_lock = fs::read(root_path.join("LOCK")).unwrap();
                    for (expected, generation, target) in [
                        (
                            first.authority(),
                            original.binding().generation(),
                            first.authority(),
                        ),
                        (
                            second.authority(),
                            rebuilt.binding().generation(),
                            second.authority(),
                        ),
                        (
                            first.authority(),
                            original.binding().generation(),
                            second.authority(),
                        ),
                        (
                            second.authority(),
                            original.binding().generation(),
                            first.authority(),
                        ),
                    ] {
                        assert!(matches!(
                            root.open_predecessor_generation_for_recovery_with_budget(
                                ROOT_ID,
                                expected,
                                generation,
                                target,
                                BUDGET as u64,
                            )
                            .unwrap(),
                            SnapshotOpenOutcomeV1::Refused(_)
                        ));
                    }
                    let SnapshotOpenOutcomeV1::Opened(opened) = root
                        .open_predecessor_generation_for_recovery_with_budget(
                            ROOT_ID,
                            second.authority(),
                            rebuilt.binding().generation(),
                            first.authority(),
                            BUDGET as u64,
                        )
                        .unwrap()
                    else {
                        panic!(
                            "healthy predecessor must recover independently of {damaged_role:?}"
                        );
                    };
                    assert_eq!(opened.head(), retained.retained_snapshot().head());
                    assert_eq!(opened.selected_predecessor_of(), Some(second.authority()));
                    assert_eq!(
                        opened.authority_pair(),
                        current.retained_snapshot().authority_pair()
                    );
                    assert!(!Arc::ptr_eq(&opened, retained.retained_snapshot()));
                    assert!(
                        NativeGenerationSnapshot::admit_exact(
                            &cx,
                            Arc::clone(&opened),
                            rebuilt.binding(),
                            DEFAULT_SCHEMA,
                            config(),
                            BUDGET,
                        )
                        .is_err()
                    );
                    let recovered = NativeGenerationSnapshot::admit_exact(
                        &cx,
                        opened,
                        original.binding(),
                        DEFAULT_SCHEMA,
                        config(),
                        BUDGET,
                    )
                    .unwrap();
                    let calls = original_provider.calls.load(Ordering::SeqCst);
                    assert_selected_rollback_refused(
                        recovered.rollback_selected_predecessor_candidate(&cx, [0; 16], [0xbf; 32]),
                        "invalid-candidate",
                    );
                    cx.set_cancel_requested(true);
                    assert!(matches!(
                        recovered
                            .rollback_selected_predecessor_candidate(&cx, [0xc0; 16], [0xc1; 32]),
                        Err(SearchError::Cancelled { .. })
                    ));
                    cx.set_cancel_requested(false);
                    let rollback = recovered
                        .rollback_selected_predecessor_candidate(&cx, [0xc2; 16], [0xc3; 32])
                        .unwrap();
                    let loser = recovered
                        .rollback_selected_predecessor_candidate(&cx, [0xc4; 16], [0xc5; 32])
                        .unwrap();
                    assert_eq!(original_provider.calls.load(Ordering::SeqCst), calls);
                    assert_eq!(rollback.authority().sequence, 3);
                    assert_eq!(rollback.manifest().predecessor(), Some(second.authority()));
                    assert_eq!(
                        rollback.manifest().generation(),
                        original.binding().generation()
                    );
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
                        panic!("selected rollback retains the v2 manifest contract");
                    };
                    assert_eq!(manifest.action, GenerationAuthorityActionV1::Rollback);
                    let preview = recovered
                        .search_text(&cx, original_provider.as_ref(), "common", 1)
                        .await
                        .unwrap();
                    assert_eq!(preview[0].doc_id, "original");
                    assert_eq!(
                        fs::read(root_path.join("AUTHORITY")).unwrap(),
                        before_authority
                    );
                    assert_eq!(fs::read(root_path.join("LOCK")).unwrap(), before_lock);

                    stage_manifest(&root, &rollback);
                    stage_manifest(&root, &loser);
                    let restored = publish(
                        &cx,
                        &root,
                        &rollback,
                        original.binding(),
                        recovered.retained_snapshot().authority_pair(),
                        [0xc6; 16],
                    );
                    assert_eq!(restored.authority().sequence, 3);
                    assert_eq!(restored.retrieval_mode(), retained.retrieval_mode());
                    assert_selected_rollback_refused(
                        restored
                            .rollback_selected_predecessor_candidate(&cx, [0xc7; 16], [0xc8; 32]),
                        "missing-selection-provenance",
                    );
                    let mut publisher =
                        GenerationPublisherV1::new(&root, ROOT_ID, [0xc9; 16], PROFILE).unwrap();
                    let stale = publisher
                        .prepare_attempt(
                            &loser,
                            recovered.retained_snapshot().authority_pair(),
                            [0xca; 16],
                            [0xcb; 16],
                        )
                        .unwrap();
                    assert!(matches!(
                        publisher.publish_prepared(&cx, &loser, &stale, None, BUDGET as u64),
                        Err(crate::GenerationPublicationErrorV1::StaleExpectedAuthority)
                            | Ok(crate::GenerationPublicationResultV1 {
                                publication: PublicationOutcomeV1::NotCommitted(_),
                                ..
                            })
                    ));
                    for snapshot in [&retained, &recovered, &restored] {
                        let hits = snapshot
                            .search_text(&cx, original_provider.as_ref(), "common", 1)
                            .await
                            .unwrap();
                        assert_same_scored_results(&hits, &preview);
                        assert_eq!(hits[0].doc_id, "original");
                        assert_eq!(hits[0].metadata.as_deref().unwrap()["version"], "original");
                        let documents = snapshot.source_documents(&cx, BUDGET).unwrap();
                        assert_eq!(documents[0].content, "common intact predecessor");
                        assert_eq!(
                            documents[0].title.as_deref(),
                            Some("Original healthy title")
                        );
                    }
                    let old_current_hits = current
                        .search_text(&cx, changed_provider.as_ref(), "common", 1)
                        .await
                        .unwrap();
                    assert_same_scored_results(&old_current_hits, &expected_current_hits);
                    assert_eq!(old_current_hits[0].doc_id, "successor");
                    assert!(matches!(
                        root.open_generation_snapshot_with_budget(
                            ROOT_ID,
                            PROFILE,
                            None,
                            BUDGET as u64
                        )
                        .unwrap(),
                        SnapshotOpenOutcomeV1::Opened(_)
                    ));
                    assert!(
                        matches!(
                            root.open_predecessor_generation_for_recovery_with_budget(
                                ROOT_ID,
                                restored.authority(),
                                original.binding().generation(),
                                second.authority(),
                                BUDGET as u64,
                            )
                            .unwrap(),
                            SnapshotOpenOutcomeV1::Refused(_)
                        ),
                        "the damaged generation cannot become a rollback target"
                    );
                    assert_eq!(fs::read(&damaged_path).unwrap(), damaged_bytes);
                    assert_eq!(
                        fs::read(&preserved_path).unwrap().as_slice(),
                        rebuilt.component_bytes(damaged_role).unwrap().as_ref()
                    );
                }
            }
        });
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
            for (name, tier, ann) in [
                ("fast-exact", EmbedderTierTag::Fast, false),
                ("quality-exact", EmbedderTierTag::Quality, false),
                ("fast-ann", EmbedderTierTag::Fast, true),
                ("quality-ann", EmbedderTierTag::Quality, true),
            ] {
                let hnsw = NativeBuildRetrieval::Hnsw {
                    params: HnswParams {
                        m: 4,
                        m0: 8,
                        ef_construction: 24,
                        ef_search: 17,
                    },
                    seed: 123,
                };
                let original_retrieval = if ann {
                    hnsw
                } else {
                    NativeBuildRetrieval::Exact
                };
                let successor_retrieval = if ann {
                    NativeBuildRetrieval::Exact
                } else {
                    hnsw
                };
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
                .with_fast_storage(NativeBuildPrecision::F32, original_retrieval)
                .add_documents(
                    [IndexableDocument::new("original", "common retained source")
                        .with_title("Retained original title")
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
                let retained = Arc::new(publish(
                    &cx,
                    &root,
                    &first,
                    original.binding(),
                    ExpectedAuthorityPairV1::default(),
                    [0x85; 16],
                ));
                assert_eq!(retained.retrieval_mode(), original.retrieval_mode());
                assert_eq!(first.manifest().components().ann.is_some(), ann);
                assert!(
                    retained
                        .rollback_candidate(&cx, &retained, [0x86; 16], [0x87; 32])
                        .is_err()
                );
                let original_scope =
                    retained
                        .scope(&cx, BUDGET, |document| {
                            Ok(document.metadata.get("version").map(String::as_str)
                                == Some("original"))
                        })
                        .unwrap();
                let absent_scope = retained
                    .scope(&cx, BUDGET, |document| Ok(document.id == "successor"))
                    .unwrap();
                assert_eq!(original_scope.len(), 1);
                assert!(absent_scope.is_empty());
                fs::rename(
                    parent.join(format!("{name}-original")),
                    parent.join(format!("{name}-original-relocated")),
                )
                .unwrap();

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
                .with_fast_storage(NativeBuildPrecision::F32, successor_retrieval)
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
                let current = Arc::new(publish(
                    &cx,
                    &root,
                    &second,
                    rebuilt.binding(),
                    retained.retained_snapshot().authority_pair(),
                    [0x8b; 16],
                ));
                assert_eq!(current.retrieval_mode(), rebuilt.retrieval_mode());
                assert_ne!(current.retrieval_mode(), retained.retrieval_mode());
                assert_eq!(current.retained_snapshot().selected_predecessor_of(), None);
                let successor_scope = current
                    .scope(&cx, BUDGET, |document| {
                        Ok(document.metadata.get("version").map(String::as_str)
                            == Some("successor"))
                    })
                    .unwrap();
                assert_eq!(successor_scope.len(), 1);
                assert!(original_scope.document("successor").is_none());
                assert!(successor_scope.document("original").is_none());
                // Reopen through a new qualified root and fresh current-head
                // admission. The target is recovered from its exact authority,
                // without reusing the old reader's retained component bytes.
                let reopened_root = QualifiedGenerationRoot::admit(
                    &root_path,
                    GenerationRootAnchorLayout::new(AUTHORITY_PUBLISHER_LOCK_BYTES_V1).unwrap(),
                )
                .unwrap();
                let SnapshotOpenOutcomeV1::Opened(reopened_current) = reopened_root
                    .open_generation_snapshot_with_budget(ROOT_ID, PROFILE, None, BUDGET as u64)
                    .unwrap()
                else {
                    panic!("fresh current generation must admit before predecessor selection");
                };
                assert!(!Arc::ptr_eq(&reopened_current, current.retained_snapshot()));
                let reopened_current = NativeGenerationSnapshot::admit_exact(
                    &cx,
                    reopened_current,
                    rebuilt.binding(),
                    DEFAULT_SCHEMA,
                    config(),
                    BUDGET,
                )
                .unwrap();
                let SnapshotOpenOutcomeV1::Opened(reopened_target) = reopened_root
                    .open_predecessor_generation_snapshot_with_budget(
                        reopened_current.retained_snapshot(),
                        first.authority(),
                        BUDGET as u64,
                    )
                    .unwrap()
                else {
                    panic!("the authenticated previous slot must reopen its complete generation");
                };
                assert!(!Arc::ptr_eq(&reopened_target, retained.retained_snapshot()));
                assert_eq!(
                    reopened_target.selected_predecessor_of(),
                    Some(current.authority())
                );
                assert_eq!(reopened_target.head(), retained.retained_snapshot().head());
                assert_eq!(
                    reopened_target.authority_pair(),
                    current.retained_snapshot().authority_pair()
                );
                assert!(
                    NativeGenerationSnapshot::admit_exact(
                        &cx,
                        Arc::clone(&reopened_target),
                        rebuilt.binding(),
                        DEFAULT_SCHEMA,
                        config(),
                        BUDGET,
                    )
                    .is_err()
                );
                let reopened_target = NativeGenerationSnapshot::admit_exact(
                    &cx,
                    reopened_target,
                    original.binding(),
                    DEFAULT_SCHEMA,
                    config(),
                    BUDGET,
                )
                .unwrap();
                assert_eq!(reopened_target.retrieval_mode(), retained.retrieval_mode());
                assert_eq!(reopened_target.tier(), tier);

                // Historical reopening authenticates the target under the
                // CURRENT physical pair. It permits rollback, but cannot lend
                // that newer pair to an incremental edit of the old source.
                let refused_work = parent.join(format!("{name}-reopened-update"));
                let update_generation = ArtifactGenerationIdentityV1::new(9, [0xae; 16]).unwrap();
                let control_authority = fs::read(root_path.join("AUTHORITY")).unwrap();
                let control_lock = fs::read(root_path.join("LOCK")).unwrap();
                let before_calls = original_provider.calls.load(Ordering::SeqCst);
                let mut forbidden = Provider::new();
                forbidden.identity.producer.backend.clear();
                assert!(forbidden.identity.validate().is_err());
                let forbidden = Arc::new(forbidden);
                for provider in [original_provider.clone(), forbidden.clone()] {
                    assert!(matches!(
                        reopened_target.begin_update(
                            &cx,
                            &refused_work,
                            update_generation,
                            provider,
                            0,
                        ),
                        Err(SearchError::InvalidConfig { field, value, .. })
                            if field == "native_ann.update.authority"
                                && value == "reopened-predecessor"
                    ));
                }
                assert!(!refused_work.exists());
                assert_eq!(original_provider.calls.load(Ordering::SeqCst), before_calls);
                assert_eq!(forbidden.calls.load(Ordering::SeqCst), 0);
                assert_eq!(
                    fs::read(root_path.join("AUTHORITY")).unwrap(),
                    control_authority
                );
                assert_eq!(fs::read(root_path.join("LOCK")).unwrap(), control_lock);

                // A handle retained while it really WAS current still owns its
                // original pair. Its complete successor can be built, but that
                // stale pair must lose publication after a competing writer won.
                let old_update = retained
                    .begin_update(
                        &cx,
                        parent.join(format!("{name}-old-retained-update")),
                        update_generation,
                        original_provider.clone(),
                        BUDGET,
                    )
                    .unwrap()
                    .build_exact_generation(&cx, tier, CommitRange { low: 1, high: 10 }, BUDGET)
                    .await
                    .unwrap();
                assert_eq!(original_provider.calls.load(Ordering::SeqCst), before_calls);
                let stale_candidate = old_update
                    .candidate(
                        [0xaf; 16],
                        Some(retained.authority()),
                        GenerationAuthorityActionV1::Activate,
                        [0xb0; 32],
                    )
                    .unwrap();
                stage(&root, &old_update, &stale_candidate);
                let rejected = {
                    let mut publisher =
                        GenerationPublisherV1::new(&root, ROOT_ID, [0xb1; 16], PROFILE).unwrap();
                    let attempt = publisher
                        .prepare_attempt(
                            &stale_candidate,
                            retained.retained_snapshot().authority_pair(),
                            [0xb2; 16],
                            [0xb3; 16],
                        )
                        .unwrap();
                    publisher.publish_prepared(&cx, &stale_candidate, &attempt, None, BUDGET as u64)
                };
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
                    control_authority
                );
                assert_eq!(fs::read(root_path.join("LOCK")).unwrap(), control_lock);
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
                let rollback = reopened_target
                    .rollback_candidate(&cx, &reopened_current, [0x93; 16], [0x94; 32])
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
                let restored = Arc::new(publish(
                    &cx,
                    &root,
                    &rollback,
                    original.binding(),
                    current.retained_snapshot().authority_pair(),
                    [0x97; 16],
                ));
                assert_eq!(restored.authority().sequence, 3);
                assert_eq!(restored.retrieval_mode(), retained.retrieval_mode());
                match (
                    retained.retrieval_configuration(),
                    restored.retrieval_configuration(),
                ) {
                    (
                        NativeBuildRetrieval::Hnsw { params, seed },
                        NativeBuildRetrieval::Hnsw {
                            params: restored_params,
                            seed: restored_seed,
                        },
                    ) => {
                        assert!(ann);
                        assert_eq!(params.ef_search, 17);
                        assert_eq!(seed, 123);
                        assert_eq!(restored_params, params);
                        assert_eq!(restored_seed, seed);
                    }
                    (NativeBuildRetrieval::Exact, NativeBuildRetrieval::Exact) => {
                        assert!(!ann);
                    }
                    _ => panic!("rollback must restore the target's full retrieval policy"),
                }
                assert_eq!(rollback.manifest().components().ann.is_some(), ann);
                assert_eq!(
                    restored.metadata().commit_range,
                    CommitRange { low: 1, high: 4 }
                );
                let after_authority = fs::read(root_path.join("AUTHORITY")).unwrap();
                let after_lock = fs::read(root_path.join("LOCK")).unwrap();
                let SnapshotOpenOutcomeV1::Opened(previous_current) = root
                    .open_predecessor_generation_snapshot_with_budget(
                        restored.retained_snapshot(),
                        current.authority(),
                        BUDGET as u64,
                    )
                    .unwrap()
                else {
                    panic!("the displaced successor remains a selectable previous slot");
                };
                let previous_current = NativeGenerationSnapshot::admit_exact(
                    &cx,
                    previous_current,
                    rebuilt.binding(),
                    DEFAULT_SCHEMA,
                    config(),
                    BUDGET,
                )
                .unwrap();
                assert_eq!(
                    previous_current
                        .retained_snapshot()
                        .selected_predecessor_of(),
                    Some(restored.authority())
                );
                assert!(retained.authority().sequence < previous_current.authority().sequence);
                // The target really is older, so an age check alone would
                // allow this proposal. Explicit predecessor provenance cannot
                // authorize acting as the current head.
                let Err(SearchError::InvalidConfig { field, value, .. }) =
                    retained.rollback_candidate(&cx, &previous_current, [0xa1; 16], [0xa2; 32])
                else {
                    panic!("a selected predecessor cannot be used as current authority");
                };
                assert_eq!(field, "native_ann.generation.rollback");
                assert_eq!(value, "current-is-selected-predecessor");
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
                for snapshot in [retained.as_ref(), &reopened_target, restored.as_ref()] {
                    let documents = snapshot.source_documents(&cx, BUDGET).unwrap();
                    assert_eq!(documents.len(), 1);
                    assert_eq!(documents[0].id, "original");
                    assert_eq!(documents[0].content, "common retained source");
                    assert_eq!(
                        documents[0].title.as_deref(),
                        Some("Retained original title")
                    );
                    assert_eq!(documents[0].metadata["version"], "original");
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
                for snapshot in [current.as_ref(), &reopened_current, &previous_current] {
                    let hits = snapshot
                        .search_text(&cx, changed_provider.as_ref(), "common", 5)
                        .await
                        .unwrap();
                    assert_eq!(hits[0].doc_id, "successor");
                    assert_eq!(
                        snapshot.source_documents(&cx, BUDGET).unwrap()[0].id,
                        "successor"
                    );
                }
                let restored_scope = restored.scope(&cx, BUDGET, |_| Ok(true)).unwrap();
                let original_hits = original_scope
                    .search_text(&cx, original_provider.as_ref(), "common", usize::MAX)
                    .await
                    .unwrap();
                let restored_hits = restored_scope
                    .search_text(&cx, original_provider.as_ref(), "common", usize::MAX)
                    .await
                    .unwrap();
                assert_same_scored_results(&original_hits, &restored_hits);
                assert_eq!(original_hits[0].doc_id, "original");
                let original_document = original_scope.document("original").unwrap();
                assert_eq!(original_document.content, "common retained source");
                assert_eq!(
                    original_document.title.as_deref(),
                    Some("Retained original title")
                );
                assert_eq!(original_document.metadata["version"], "original");
                assert!(original_scope.document("successor").is_none());
                assert!(
                    original_scope
                        .narrow(&cx, |document| Ok(document.id == "successor"))
                        .unwrap()
                        .is_empty()
                );
                assert!(
                    absent_scope
                        .narrow(&cx, |_| panic!("publication cannot add scoped members"))
                        .unwrap()
                        .is_empty()
                );
                let successor_hits = successor_scope
                    .search_text(&cx, changed_provider.as_ref(), "common", usize::MAX)
                    .await
                    .unwrap();
                assert_eq!(successor_hits[0].doc_id, "successor");
                let successor_document = successor_scope.document("successor").unwrap();
                assert_eq!(successor_document.content, "common replacement source");
                assert_eq!(successor_document.metadata["version"], "successor");
                assert!(successor_scope.document("original").is_none());
                let calls = original_provider.calls.load(Ordering::SeqCst);
                assert!(
                    absent_scope
                        .search_text(&cx, original_provider.as_ref(), "common", 1)
                        .await
                        .unwrap()
                        .is_empty()
                );
                assert_eq!(original_provider.calls.load(Ordering::SeqCst), calls);
                cx.set_cancel_requested(true);
                assert!(matches!(
                    retained.rollback_candidate(&cx, &restored, [0x9b; 16], [0x9c; 32]),
                    Err(SearchError::Cancelled { .. })
                ));
                assert!(matches!(
                    retained.source_documents(&cx, BUDGET),
                    Err(SearchError::Cancelled { .. })
                ));
                cx.set_cancel_requested(false);
                assert_eq!(
                    fs::read(root_path.join("AUTHORITY")).unwrap(),
                    after_authority
                );
                assert_eq!(fs::read(root_path.join("LOCK")).unwrap(), after_lock);
                // Source reads retain their original component bytes even when
                // the entire qualified root pathname moves after publication.
                fs::rename(&root_path, parent.join(format!("{name}-root-relocated"))).unwrap();
                let calls = original_provider.calls.load(Ordering::SeqCst);
                for snapshot in [retained.as_ref(), &reopened_target, restored.as_ref()] {
                    let source = snapshot
                        .source_document(&cx, "original", BUDGET)
                        .unwrap()
                        .unwrap();
                    assert_eq!(source.content, "common retained source");
                    assert_eq!(source.title.as_deref(), Some("Retained original title"));
                    assert_eq!(source.metadata["version"], "original");
                    assert!(
                        snapshot
                            .source_document(&cx, "successor", 0)
                            .unwrap()
                            .is_none()
                    );
                    let selected = snapshot
                        .source_documents_by_ids(
                            &cx,
                            &["original", "successor", "original"],
                            BUDGET,
                        )
                        .unwrap();
                    assert!(selected[1].is_none());
                    assert_eq!(selected[0].as_ref().unwrap().content, source.content);
                    assert_eq!(selected[2].as_ref().unwrap().content, source.content);
                }
                assert_eq!(original_provider.calls.load(Ordering::SeqCst), calls);
            }
        });
    }
}
