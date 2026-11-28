//! Exercise the fsfs bridge through real fixed-authority publications. The
//! deterministic providers control ranking and producer drift; they do not
//! replace root, receipt, vector, lexical, or source admission.

#![cfg(target_os = "linux")]

use std::fs::{self, File, OpenOptions};
use std::future::{Future as _, ready};
use std::io::Write as _;
use std::os::unix::fs::{DirBuilderExt as _, OpenOptionsExt as _};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll, Waker};

use asupersync::test_utils::run_test_with_cx;
use asupersync::time::{TimerDriverHandle, VirtualClock};
use asupersync::{Budget, Cx};
use frankensearch::native_ann::builder::{
    NativeBuildPrecision, NativeBuildRetrieval, NativeBuiltExactGeneration, NativeIndexBuilder,
};
use frankensearch::native_ann::{
    NativeGenerationReceipt, NativeGenerationSavedAttempt, NativeGenerationSnapshot,
};
use frankensearch::{GenerationActivationV1, GenerationPublisherV1, IndexableDocument};
use frankensearch_core::generation::{
    ArtifactGenerationIdentityV1, CommitRange, EmbedderTierTag, EmbeddingArtifactIdentityV1,
    EmbeddingIdentityBundleV1, EmbeddingSpaceKindV1, GenerationAuthorityActionV1,
    GenerationRootSecurityProfileV1,
};
use frankensearch_core::{Embedder, ModelCategory, SearchError, SearchFuture, SearchResult};
use frankensearch_index::generation_root::authority_publisher::{
    AUTHORITY_PUBLISHER_LOCK_BYTES_V1, ExpectedAuthorityPairV1, PublicationOutcomeV1,
};
use frankensearch_index::generation_root::generation_reader::{
    activation_manifest_path_v1, component_object_path_v1,
};
use frankensearch_index::generation_root::{
    GENERATION_ROOT_AUTHORITY_FILE_BYTES, GenerationRootAnchorLayout, QualifiedGenerationRoot,
};

use crate::runtime::FsfsRuntime;
use crate::{CliCommand, CliInput, FsfsConfig, OutputFormat};

const IMAGE_BUDGET: usize = 16 * 1024 * 1024;
const ROOT_ID: [u8; 16] = [0x51; 16];
const PROFILE: GenerationRootSecurityProfileV1 = GenerationRootSecurityProfileV1::CooperativeLocal;

#[derive(Clone, Copy)]
enum Topology {
    Fast,
    Quality,
    Full,
    Partial,
}

struct Producer {
    identity: EmbeddingIdentityBundleV1,
    tier: EmbedderTierTag,
    calls: AtomicUsize,
    pending_query: bool,
}

impl Producer {
    fn new(tier: EmbedderTierTag) -> Self {
        let quality = tier == EmbedderTierTag::Quality;
        let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(
            if quality {
                "fsfs-authority-quality"
            } else {
                "fsfs-authority-fast"
            },
            if quality { 3 } else { 2 },
        );
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
            tier,
            calls: AtomicUsize::new(0),
            pending_query: false,
        }
    }

    fn reset(&self) {
        self.calls.store(0, Ordering::SeqCst);
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl Embedder for Producer {
    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
    }

    fn embed<'a>(&'a self, _: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            let query = text.contains("semantic-query");
            if query && self.pending_query {
                return std::future::pending().await;
            }
            Ok(if self.tier == EmbedderTierTag::Quality {
                if query || text.contains("quality-anchor") {
                    vec![1.0, 0.0, 0.0]
                } else {
                    vec![0.0, 1.0, 0.0]
                }
            } else if query || text.contains("fast-anchor") {
                vec![1.0, 0.0]
            } else if text.contains("ordinary-anchor") {
                vec![0.8, 0.6]
            } else if text.contains("garden") {
                vec![0.6, 0.8]
            } else {
                vec![0.0, 1.0]
            })
        })
    }

    fn dimension(&self) -> usize {
        if self.tier == EmbedderTierTag::Quality {
            3
        } else {
            2
        }
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

fn documents(version: &str) -> Vec<IndexableDocument> {
    [
        ("fast-leader", "fast-anchor retry guidance"),
        ("ordinary", "ordinary-anchor retry handling"),
        ("quality-leader", "quality-anchor retry resilience"),
        ("garden", "garden flower cultivation"),
    ]
    .into_iter()
    .map(|(id, content)| {
        IndexableDocument::new(id, content)
            .with_title(format!("retained {id}"))
            .with_metadata("version", version)
    })
    .collect()
}

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
    // Match the existing native publication tests: an unsupported filesystem
    // is a failure, not a skipped or synthetic root-admission success.
    QualifiedGenerationRoot::admit(
        path,
        GenerationRootAnchorLayout::new(AUTHORITY_PUBLISHER_LOCK_BYTES_V1).unwrap(),
    )
    .expect("qualified Linux ext4/btrfs root is required")
}

async fn build(
    cx: &Cx,
    work: &Path,
    topology: Topology,
    sequence: u8,
    producers: (&Arc<Producer>, &Arc<Producer>),
    source: Vec<IndexableDocument>,
) -> NativeBuiltExactGeneration {
    let (fast, quality) = producers;
    let primary: Arc<dyn Embedder> = if matches!(topology, Topology::Quality) {
        quality.clone()
    } else {
        fast.clone()
    };
    let mut builder = NativeIndexBuilder::new(
        work,
        ArtifactGenerationIdentityV1::new(u64::from(sequence), [sequence; 16]).unwrap(),
        primary,
    )
    .unwrap()
    .with_fast_storage(NativeBuildPrecision::F32, NativeBuildRetrieval::Exact)
    .add_documents(source);
    if matches!(topology, Topology::Full | Topology::Partial) {
        builder = builder
            .with_quality_embedder(quality.clone())
            .unwrap()
            .with_quality_storage(NativeBuildPrecision::F32, NativeBuildRetrieval::Exact)
            .unwrap();
    }
    let checkpoint = CommitRange {
        low: 1,
        high: u64::from(sequence) + 4,
    };
    match topology {
        Topology::Fast | Topology::Quality => {
            builder
                .build_exact_generation(
                    cx,
                    if matches!(topology, Topology::Quality) {
                        EmbedderTierTag::Quality
                    } else {
                        EmbedderTierTag::Fast
                    },
                    checkpoint,
                    IMAGE_BUDGET,
                )
                .await
        }
        Topology::Full => {
            builder
                .build_progressive_generation(cx, checkpoint, IMAGE_BUDGET)
                .await
        }
        Topology::Partial => {
            builder
                .build_partial_quality_generation(
                    cx,
                    checkpoint,
                    ["quality-leader".to_owned()],
                    IMAGE_BUDGET,
                )
                .await
        }
    }
    .unwrap()
}

#[allow(clippy::too_many_arguments)]
fn publish(
    cx: &Cx,
    root: &QualifiedGenerationRoot,
    root_path: &Path,
    receipt_path: &Path,
    built: &NativeBuiltExactGeneration,
    topology: Topology,
    previous: Option<&NativeGenerationSnapshot>,
    tag: u8,
) -> NativeGenerationReceipt {
    let candidate = built
        .candidate(
            [tag; 16],
            previous.map(NativeGenerationSnapshot::authority),
            GenerationAuthorityActionV1::Activate,
            [tag; 32],
        )
        .unwrap();
    built.validate_candidate(cx, &candidate).unwrap();
    for (role, receipt) in candidate.manifest().components().iter() {
        root.stage_immutable_file(
            &component_object_path_v1(role, receipt.sha256).unwrap(),
            &built.component_bytes(role).unwrap(),
        )
        .unwrap();
    }
    root.stage_immutable_file(
        &activation_manifest_path_v1(candidate.authority().object_id).unwrap(),
        candidate.manifest_bytes(),
    )
    .unwrap();
    let mut publisher = GenerationPublisherV1::new(root, ROOT_ID, [tag; 16], PROFILE).unwrap();
    let attempt = publisher
        .prepare_attempt(
            &candidate,
            previous.map_or_else(ExpectedAuthorityPairV1::default, |snapshot| {
                snapshot.retained_snapshot().authority_pair()
            }),
            [tag; 16],
            [tag; 16],
        )
        .unwrap();
    let receipt = NativeGenerationReceipt {
        schema: match topology {
            Topology::Fast | Topology::Quality => "frankensearch.native.authority-selection.v2",
            Topology::Full => "frankensearch.native.authority-selection.v3",
            Topology::Partial => "frankensearch.native.authority-selection.v4",
        }
        .to_owned(),
        root: root_path.to_path_buf(),
        root_id: ROOT_ID,
        profile: PROFILE.as_str().to_owned(),
        authority: candidate.authority().canonical_bytes().to_vec(),
        generation: built.binding().generation(),
        identity: built.binding().frozen_identity().clone(),
        tier: built.metadata().vector_artifacts[0].embedder_tier,
        quality_identity: built
            .quality_binding()
            .map(|binding| binding.frozen_identity().clone()),
        quality_documents: matches!(topology, Topology::Partial).then_some(1),
        documents: built.metadata().total_documents,
        image_bytes: u64::try_from(IMAGE_BUDGET).unwrap(),
        attempt: Some(NativeGenerationSavedAttempt {
            writer_id: attempt.writer_id,
            attempt_id: attempt.attempt_id,
            expected_first: attempt
                .expected
                .first
                .map(|slot| slot.authority.canonical_bytes().to_vec()),
            expected_second: attempt
                .expected
                .second
                .map(|slot| slot.authority.canonical_bytes().to_vec()),
            idempotency_key: attempt.idempotency_key,
            requires_floor: false,
        }),
    };
    receipt.validate().unwrap();
    fs::write(receipt_path, serde_json::to_vec(&receipt).unwrap()).unwrap();
    let result = publisher
        .publish_prepared(
            cx,
            &candidate,
            &attempt,
            None,
            u64::try_from(IMAGE_BUDGET).unwrap(),
        )
        .unwrap();
    assert!(matches!(
        result.publication,
        PublicationOutcomeV1::Committed { .. }
    ));
    assert!(matches!(
        result.activation,
        Some(GenerationActivationV1::Installed(_))
    ));
    receipt.open(cx, IMAGE_BUDGET).unwrap();
    receipt
}

struct Fixture {
    parent: tempfile::TempDir,
    root: QualifiedGenerationRoot,
    root_path: PathBuf,
    receipt_path: PathBuf,
    receipt: NativeGenerationReceipt,
    fast: Arc<Producer>,
    quality: Arc<Producer>,
}

impl Fixture {
    async fn new(cx: &Cx, topology: Topology) -> Self {
        let base = std::env::var_os("FRANKENSEARCH_TEST_TMPDIR")
            .or_else(|| std::env::var_os("HOME"))
            .expect("a qualified test directory or HOME is required");
        let parent = tempfile::Builder::new()
            .prefix("frankensearch-fsfs-authority-")
            .tempdir_in(base)
            .unwrap();
        let root_path = parent.path().join("root");
        let root = create_root(&root_path);
        let fast = Arc::new(Producer::new(EmbedderTierTag::Fast));
        let quality = Arc::new(Producer::new(EmbedderTierTag::Quality));
        let built = build(
            cx,
            &parent.path().join("work"),
            topology,
            1,
            (&fast, &quality),
            documents("original"),
        )
        .await;
        let receipt_path = parent.path().join("selected.json");
        let receipt = publish(
            cx,
            &root,
            &root_path,
            &receipt_path,
            &built,
            topology,
            None,
            0x61,
        );
        fast.reset();
        quality.reset();
        Self {
            parent,
            root,
            root_path,
            receipt_path,
            receipt,
            fast,
            quality,
        }
    }

    fn runtime(&self, mode: Option<&str>, query: &str, limit: usize) -> FsfsRuntime {
        let mut config = FsfsConfig::default();
        config.search.default_limit = limit;
        config.search.quality_timeout_ms = 60_000;
        config.pressure.profile = crate::config::PressureProfile::Performance;
        FsfsRuntime::new(config).with_cli_input(CliInput {
            command: CliCommand::Search,
            query: Some(query.to_owned()),
            authority_receipt: Some(self.receipt_path.clone()),
            authority_mode: mode.map(str::to_owned),
            format: OutputFormat::Json,
            ..CliInput::default()
        })
    }
}

fn payload(bytes: &[u8]) -> serde_json::Value {
    let envelope: serde_json::Value = serde_json::from_slice(bytes).unwrap();
    assert!(envelope["data"].is_object(), "{envelope}");
    envelope["data"].clone()
}

fn timed_context(parent: &Cx) -> (Cx, Arc<VirtualClock>, TimerDriverHandle) {
    let clock = Arc::new(VirtualClock::new());
    let timer = TimerDriverHandle::with_virtual_clock(Arc::clone(&clock));
    let cx = Cx::new_with_drivers(
        parent.region_id(),
        parent.task_id(),
        Budget::INFINITE,
        None,
        None,
        None,
        Some(timer.clone()),
        None,
    );
    (cx, clock, timer)
}

#[test]
fn authority_search_lexical_reads_retained_source_without_loading_models() {
    run_test_with_cx(|cx| async move {
        let fixture = Fixture::new(&cx, Topology::Full).await;
        let work = fixture.parent.path().join("work");
        fs::rename(&work, fixture.parent.path().join("retired-work")).unwrap();
        fs::create_dir(&work).unwrap();
        fs::write(work.join("fast-leader"), "untrusted replacement source").unwrap();
        let authority_before = fs::read(fixture.root_path.join("AUTHORITY")).unwrap();
        let mut output = Vec::new();
        let mut loads = 0;
        fixture
            .runtime(Some("lexical"), "guidance", 1)
            .run_authority_search_with_writer_and_loader(&cx, &mut output, |_| {
                loads += 1;
                ready(Ok(fixture.fast.clone() as Arc<dyn Embedder>))
            })
            .await
            .unwrap();
        let data = payload(&output);
        assert_eq!(data["mode"], "lexical");
        assert_eq!(data["topology"], "full_progressive");
        assert_eq!(
            data["authority"],
            serde_json::json!(fixture.receipt.authority)
        );
        assert_eq!(data["results"][0]["doc_id"], "fast-leader");
        assert_eq!(data["results"][0]["title"], "retained fast-leader");
        assert_eq!(data["results"][0]["metadata"]["version"], "original");
        assert!(
            data["results"][0]["snippet"]
                .as_str()
                .unwrap()
                .contains("fast-anchor retry guidance")
        );
        assert_eq!(loads, 0, "lexical search must never load either model");
        assert_eq!((fixture.fast.calls(), fixture.quality.calls()), (0, 0));
        assert_eq!(
            fs::read(fixture.root_path.join("AUTHORITY")).unwrap(),
            authority_before
        );
    });
}

#[test]
fn authority_search_wrong_producer_fails_before_query_inference_or_output() {
    run_test_with_cx(|cx| async move {
        let fixture = Fixture::new(&cx, Topology::Fast).await;
        let mut foreign = Producer::new(EmbedderTierTag::Fast);
        foreign
            .identity
            .producer
            .implementation_revision
            .push_str("-foreign");
        foreign.identity.validate().unwrap();
        let foreign = Arc::new(foreign);
        let mut loaded = Vec::new();
        let mut output = Vec::new();
        let result = fixture
            .runtime(Some("fast"), "semantic-query", 1)
            .run_authority_search_with_writer_and_loader(&cx, &mut output, |tier| {
                loaded.push(tier);
                ready(Ok(foreign.clone() as Arc<dyn Embedder>))
            })
            .await;
        assert!(result.is_err());
        assert_eq!(loaded, vec![EmbedderTierTag::Fast]);
        assert_eq!(foreign.calls(), 0);
        assert!(
            output.is_empty(),
            "a rejected producer cannot publish a search phase"
        );
        fixture.receipt.open(&cx, IMAGE_BUDGET).unwrap();
    });
}

#[test]
fn authority_search_quality_only_loads_the_independent_quality_producer() {
    run_test_with_cx(|cx| async move {
        for topology in [Topology::Quality, Topology::Full] {
            let fixture = Fixture::new(&cx, topology).await;
            let mut loaded = Vec::new();
            let mut output = Vec::new();
            let mut runtime = fixture.runtime(Some("quality"), "semantic-query", 1);
            runtime.config.indexing.fast_model = "invalid-fast-model-must-not-be-consulted".into();
            runtime
                .run_authority_search_with_writer_and_loader(&cx, &mut output, |tier| {
                    assert_eq!(tier, EmbedderTierTag::Quality);
                    loaded.push(tier);
                    ready(Ok(fixture.quality.clone() as Arc<dyn Embedder>))
                })
                .await
                .unwrap();
            let data = payload(&output);
            assert_eq!(data["mode"], "quality");
            assert_eq!(data["results"][0]["doc_id"], "quality-leader");
            assert_eq!(loaded, vec![EmbedderTierTag::Quality]);
            assert_eq!((fixture.fast.calls(), fixture.quality.calls()), (0, 1));
        }
    });
}

#[test]
fn authority_search_full_refinement_finds_a_winner_outside_the_fast_page() {
    run_test_with_cx(|cx| async move {
        let fixture = Fixture::new(&cx, Topology::Full).await;
        let mut fast_output = Vec::new();
        fixture
            .runtime(Some("fast"), "semantic-query", 3)
            .run_authority_search_with_writer_and_loader(&cx, &mut fast_output, |tier| {
                assert_eq!(tier, EmbedderTierTag::Fast);
                ready(Ok(fixture.fast.clone() as Arc<dyn Embedder>))
            })
            .await
            .unwrap();
        let fast_data = payload(&fast_output);
        assert_eq!(fast_data["results"][0]["doc_id"], "fast-leader");
        assert_eq!(fast_data["results"].as_array().unwrap().len(), 3);
        assert!(
            fast_data["results"]
                .as_array()
                .unwrap()
                .iter()
                .all(|hit| hit["doc_id"] != "quality-leader")
        );
        fixture.fast.reset();
        let mut loaded = Vec::new();
        let mut output = Vec::new();
        fixture
            .runtime(None, "semantic-query", 1)
            .run_authority_search_with_writer_and_loader(&cx, &mut output, |tier| {
                loaded.push(tier);
                let producer: Arc<dyn Embedder> = match tier {
                    EmbedderTierTag::Fast => fixture.fast.clone(),
                    EmbedderTierTag::Quality => fixture.quality.clone(),
                };
                ready(Ok(producer))
            })
            .await
            .unwrap();
        let data = payload(&output);
        assert_eq!(data["mode"], "full");
        assert_eq!(data["phase"], "refined");
        assert_eq!(data["results"][0]["doc_id"], "quality-leader");
        assert_eq!(
            loaded,
            vec![EmbedderTierTag::Fast, EmbedderTierTag::Quality]
        );
        assert_eq!((fixture.fast.calls(), fixture.quality.calls()), (1, 1));
        assert_eq!(data["results"][0]["metadata"]["version"], "original");
    });
}

#[test]
fn authority_search_partial_coverage_refuses_complete_modes_before_loading() {
    run_test_with_cx(|cx| async move {
        let fixture = Fixture::new(&cx, Topology::Partial).await;
        for mode in ["full", "quality"] {
            let mut output = Vec::new();
            let mut loads = 0;
            let result = fixture
                .runtime(Some(mode), "semantic-query", 1)
                .run_authority_search_with_writer_and_loader(&cx, &mut output, |_| {
                    loads += 1;
                    ready(Ok(fixture.fast.clone() as Arc<dyn Embedder>))
                })
                .await;
            assert!(result.is_err(), "{mode}");
            assert_eq!(loads, 0, "topology refusal must precede model work");
            assert!(output.is_empty());
        }
        let mut output = Vec::new();
        fixture
            .runtime(None, "semantic-query", 1)
            .run_authority_search_with_writer_and_loader(&cx, &mut output, |tier| {
                assert_eq!(tier, EmbedderTierTag::Fast);
                ready(Ok(fixture.fast.clone() as Arc<dyn Embedder>))
            })
            .await
            .unwrap();
        let data = payload(&output);
        assert_eq!(data["mode"], "fast");
        assert_eq!(data["topology"], "partial_quality");
        assert_eq!(data["quality_coverage"]["quality_documents"], 1);
        assert_eq!(data["quality_coverage"]["total_documents"], 4);
        assert_eq!(data["quality_coverage"]["parts_per_million"], 250_000);
        assert_eq!(data["results"][0]["doc_id"], "fast-leader");
        assert_eq!(fixture.quality.calls(), 0);

        fixture.fast.reset();
        let mut loaded = Vec::new();
        let mut refined_output = Vec::new();
        fixture
            .runtime(Some("partial"), "semantic-query", 3)
            .run_authority_search_with_writer_and_loader(&cx, &mut refined_output, |tier| {
                loaded.push(tier);
                let producer: Arc<dyn Embedder> = match tier {
                    EmbedderTierTag::Fast => fixture.fast.clone(),
                    EmbedderTierTag::Quality => fixture.quality.clone(),
                };
                ready(Ok(producer))
            })
            .await
            .unwrap();
        let refined = payload(&refined_output);
        assert_eq!(refined["mode"], "partial");
        assert_eq!(refined["phase"], "refined");
        assert_eq!(refined["topology"], "partial_quality");
        assert_eq!(refined["quality_coverage"], data["quality_coverage"]);
        assert_eq!(refined["candidates"]["quality"], 1);
        let hits = refined["results"].as_array().unwrap();
        assert_eq!(hits.len(), 3);
        let quality_hit = hits
            .iter()
            .find(|hit| hit["doc_id"] == "quality-leader")
            .expect("explicit partial refinement must return its quality candidate");
        assert_eq!(quality_hit["quality_score"], 1.0);
        assert_eq!(quality_hit["metadata"]["version"], "original");
        assert_eq!(quality_hit["title"], "retained quality-leader");
        assert_eq!(
            loaded,
            vec![EmbedderTierTag::Fast, EmbedderTierTag::Quality]
        );
        assert_eq!((fixture.fast.calls(), fixture.quality.calls()), (1, 1));
    });
}

#[test]
fn authority_search_retains_its_admitted_generation_across_model_loading() {
    run_test_with_cx(|cx| async move {
        let fixture = Fixture::new(&cx, Topology::Fast).await;
        let selected = fixture.receipt.open(&cx, IMAGE_BUDGET).unwrap();
        let successor = build(
            &cx,
            &fixture.parent.path().join("successor-work"),
            Topology::Fast,
            2,
            (&fixture.fast, &fixture.quality),
            vec![
                IndexableDocument::new("successor-only", "fast-anchor replacement body")
                    .with_metadata("version", "successor"),
            ],
        )
        .await;
        fixture.fast.reset();
        let successor_path = fixture.parent.path().join("successor.json");
        let mut successor_receipt = None;
        let mut output = Vec::new();
        fixture
            .runtime(Some("fast"), "semantic-query", 1)
            .run_authority_search_with_writer_and_loader(&cx, &mut output, |tier| {
                assert_eq!(tier, EmbedderTierTag::Fast);
                assert!(successor_receipt.is_none());
                successor_receipt = Some(publish(
                    &cx,
                    &fixture.root,
                    &fixture.root_path,
                    &successor_path,
                    &successor,
                    Topology::Fast,
                    Some(&selected),
                    0x62,
                ));
                ready(Ok(fixture.fast.clone() as Arc<dyn Embedder>))
            })
            .await
            .unwrap();
        let data = payload(&output);
        assert_eq!(
            data["authority"],
            serde_json::json!(fixture.receipt.authority)
        );
        assert_eq!(data["results"][0]["doc_id"], "fast-leader");
        assert_eq!(data["results"][0]["metadata"]["version"], "original");
        assert!(
            data["results"][0]["snippet"]
                .as_str()
                .unwrap()
                .contains("retry guidance")
        );

        // A new request must not discover the successor through the stale
        // receipt, even though the earlier admitted reader remains usable.
        let mut refused = Vec::new();
        let mut loads = 0;
        assert!(
            fixture
                .runtime(Some("fast"), "semantic-query", 1)
                .run_authority_search_with_writer_and_loader(&cx, &mut refused, |_| {
                    loads += 1;
                    ready(Ok(fixture.fast.clone() as Arc<dyn Embedder>))
                })
                .await
                .is_err()
        );
        assert_eq!(loads, 0);
        assert!(refused.is_empty());
        successor_receipt.unwrap().open(&cx, IMAGE_BUDGET).unwrap();
        let mut current = fixture.runtime(Some("fast"), "semantic-query", 1);
        current.cli_input.authority_receipt = Some(successor_path);
        let mut current_output = Vec::new();
        current
            .run_authority_search_with_writer_and_loader(&cx, &mut current_output, |_| {
                ready(Ok(fixture.fast.clone() as Arc<dyn Embedder>))
            })
            .await
            .unwrap();
        assert_eq!(
            payload(&current_output)["results"][0]["doc_id"],
            "successor-only"
        );
    });
}

#[test]
fn authority_search_reads_authenticated_history_with_one_total_budget() {
    run_test_with_cx(|cx| async move {
        let fixture = Fixture::new(&cx, Topology::Fast).await;
        let mut head = fixture.receipt.open(&cx, IMAGE_BUDGET).unwrap();
        let mut previous_path = fixture.receipt_path.clone();
        for (sequence, kind) in [(2, "retained_predecessor"), (3, "retained_ancestor")] {
            let built = build(
                &cx,
                &fixture.parent.path().join(format!("work-{sequence}")),
                Topology::Fast,
                sequence,
                (&fixture.fast, &fixture.quality),
                vec![IndexableDocument::new("new-only", "different source")],
            )
            .await;
            let head_path = fixture.parent.path().join(format!("head-{sequence}.json"));
            let receipt = publish(
                &cx,
                &fixture.root,
                &fixture.root_path,
                &head_path,
                &built,
                Topology::Fast,
                Some(&head),
                0x60 + sequence,
            );
            head = receipt.open(&cx, IMAGE_BUDGET).unwrap();
            let mut search = fixture.runtime(Some("fast"), "semantic-query", 1);
            search.cli_input.authority_current_receipt = Some(head_path.clone());
            let mut output = Vec::new();
            search
                .run_authority_search_with_writer_and_loader(&cx, &mut output, |tier| {
                    assert_eq!(tier, EmbedderTierTag::Fast);
                    ready(Ok(fixture.fast.clone() as Arc<dyn Embedder>))
                })
                .await
                .unwrap();
            let page = payload(&output);
            assert_eq!(page["selection_kind"], kind);
            assert_eq!(
                page["selected_history_of"],
                serde_json::json!(receipt.authority)
            );
            assert_eq!(
                page["authority"],
                serde_json::json!(fixture.receipt.authority)
            );
            assert_eq!(page["results"][0]["doc_id"], "fast-leader");
            assert_eq!(page["results"][0]["metadata"]["version"], "original");
            assert!(
                page["results"][0]["snippet"]
                    .as_str()
                    .unwrap()
                    .contains("retry guidance")
            );

            let head_bytes =
                usize::try_from(head.retained_snapshot().retained_image_bytes().unwrap()).unwrap();
            for (current_path, budget) in [(&head_path, head_bytes), (&previous_path, IMAGE_BUDGET)]
            {
                search.cli_input.authority_current_receipt = Some(current_path.clone());
                search.cli_input.authority_max_image_bytes = Some(budget);
                let mut refused = Vec::new();
                let mut loads = 0;
                assert!(
                    search
                        .run_authority_search_with_writer_and_loader(&cx, &mut refused, |_| {
                            loads += 1;
                            ready(Ok(fixture.fast.clone() as Arc<dyn Embedder>))
                        })
                        .await
                        .is_err()
                );
                assert_eq!(loads, 0);
                assert!(refused.is_empty());
            }
            previous_path = head_path;
        }
    });
}

#[test]
fn authority_search_refuses_receipt_and_image_budget_before_loading_models() {
    run_test_with_cx(|cx| async move {
        let fixture = Fixture::new(&cx, Topology::Fast).await;
        let snapshot = fixture.receipt.open(&cx, IMAGE_BUDGET).unwrap();
        for forbidden_floor in [false, true] {
            let mut invalid = fixture.receipt.clone();
            if forbidden_floor {
                invalid.attempt.as_mut().unwrap().requires_floor = true;
            } else {
                invalid.profile = "unsupported-profile".to_owned();
            }
            assert!(matches!(
                invalid.admit_opened(&cx, Arc::clone(snapshot.retained_snapshot()), IMAGE_BUDGET,),
                Err(SearchError::InvalidConfig { .. })
            ));
        }
        let mut bounded = fixture.receipt.clone();
        bounded.image_bytes = bounded.authority().unwrap().manifest_len;
        bounded.validate().unwrap();
        assert!(
            bounded
                .admit_opened(&cx, Arc::clone(snapshot.retained_snapshot()), IMAGE_BUDGET,)
                .is_err()
        );
        let mut changed = fixture.receipt.clone();
        changed.documents += 1;
        changed.validate().unwrap();
        let changed_path = fixture.parent.path().join("wrong-count.json");
        fs::write(&changed_path, serde_json::to_vec(&changed).unwrap()).unwrap();
        for wrong_count in [false, true] {
            let mut runtime = fixture.runtime(Some("fast"), "semantic-query", 1);
            if wrong_count {
                runtime.cli_input.authority_receipt = Some(changed_path.clone());
            } else {
                runtime.cli_input.authority_max_image_bytes = Some(1);
            }
            let mut loads = 0;
            let mut output = Vec::new();
            assert!(
                runtime
                    .run_authority_search_with_writer_and_loader(&cx, &mut output, |_| {
                        loads += 1;
                        ready(Ok(fixture.fast.clone() as Arc<dyn Embedder>))
                    })
                    .await
                    .is_err()
            );
            assert_eq!(loads, 0);
            assert!(output.is_empty());
        }
        fixture.receipt.open(&cx, IMAGE_BUDGET).unwrap();
    });
}

#[test]
fn authority_search_source_budget_and_precancellation_never_publish_success() {
    run_test_with_cx(|cx| async move {
        let fixture = Fixture::new(&cx, Topology::Fast).await;
        let mut runtime = fixture.runtime(Some("lexical"), "guidance", 1);
        runtime.cli_input.authority_max_source_bytes = Some(1);
        let mut output = Vec::new();
        let mut loads = 0;
        assert!(
            runtime
                .run_authority_search_with_writer_and_loader(&cx, &mut output, |_| {
                    loads += 1;
                    ready(Ok(fixture.fast.clone() as Arc<dyn Embedder>))
                })
                .await
                .is_err()
        );
        assert_eq!(loads, 0);
        assert!(output.is_empty());
        cx.set_cancel_requested(true);
        let result = fixture
            .runtime(Some("fast"), "semantic-query", 1)
            .run_authority_search_with_writer_and_loader(&cx, &mut output, |_| {
                loads += 1;
                ready(Ok(fixture.fast.clone() as Arc<dyn Embedder>))
            })
            .await;
        cx.set_cancel_requested(false);
        assert!(matches!(result, Err(SearchError::Cancelled { .. })));
        assert_eq!(loads, 0);
        assert!(output.is_empty());
        fixture.receipt.open(&cx, IMAGE_BUDGET).unwrap();
    });
}

#[test]
fn authority_search_rejects_unsupported_runtime_policy_and_honors_quality_override() {
    run_test_with_cx(|cx| async move {
        let fixture = Fixture::new(&cx, Topology::Full).await;
        for policy in [
            "stream",
            "rerank",
            "priors",
            "explain",
            "shadow",
            "fast_only",
            "deferred_quality",
            "hard_pause",
            "quality_circuit",
            "strict_profile",
            "degraded_profile",
            "profile_override",
        ] {
            let mut runtime = fixture.runtime(Some("full"), "semantic-query", 1);
            match policy {
                "stream" => runtime.cli_input.stream = true,
                "rerank" => runtime.config.search.rerank = true,
                "priors" => runtime.config.search.ranking_priors = true,
                "explain" => runtime.config.search.explain = true,
                "shadow" => runtime.config.search.shadow_mode = true,
                "fast_only" => runtime.config.search.fast_only = true,
                "deferred_quality" => {
                    runtime.config.pressure.degradation_override =
                        crate::config::DegradationOverrideMode::ForceEmbedDeferred;
                }
                "hard_pause" | "quality_circuit" => {
                    runtime.config.search.fast_only = true;
                    runtime.cli_input.overrides.fast_only = Some(false);
                    runtime.config.pressure.hard_pause_requested = policy == "hard_pause";
                    runtime.config.pressure.quality_circuit_open = policy == "quality_circuit";
                }
                "strict_profile" | "degraded_profile" => {
                    runtime.config.search.fast_only = true;
                    runtime.cli_input.overrides.fast_only = Some(false);
                    runtime.config.pressure.profile = if policy == "strict_profile" {
                        crate::config::PressureProfile::Strict
                    } else {
                        crate::config::PressureProfile::Degraded
                    };
                }
                "profile_override" => {
                    runtime.cli_input.overrides.profile =
                        Some(crate::config::PressureProfile::Strict);
                }
                _ => unreachable!("the policy matrix is exhaustive"),
            }
            let mut output = Vec::new();
            let mut loads = 0;
            let result = runtime
                .run_authority_search_with_writer_and_loader(&cx, &mut output, |_| {
                    loads += 1;
                    ready(Ok(fixture.fast.clone() as Arc<dyn Embedder>))
                })
                .await;
            assert!(
                matches!(result, Err(SearchError::InvalidConfig { .. })),
                "{policy}"
            );
            assert_eq!(loads, 0, "{policy} must be refused before model work");
            assert!(output.is_empty(), "{policy}");
        }
        let mut runtime = fixture.runtime(Some("full"), "semantic-query", 1);
        runtime.config.search.fast_only = true;
        runtime.cli_input.overrides.fast_only = Some(false);
        let mut output = Vec::new();
        let mut loaded = Vec::new();
        runtime
            .run_authority_search_with_writer_and_loader(&cx, &mut output, |tier| {
                loaded.push(tier);
                let producer: Arc<dyn Embedder> = match tier {
                    EmbedderTierTag::Fast => fixture.fast.clone(),
                    EmbedderTierTag::Quality => fixture.quality.clone(),
                };
                ready(Ok(producer))
            })
            .await
            .unwrap();
        assert_eq!(
            loaded,
            vec![EmbedderTierTag::Fast, EmbedderTierTag::Quality]
        );
        assert_eq!(payload(&output)["results"][0]["doc_id"], "quality-leader");
        assert_eq!((fixture.fast.calls(), fixture.quality.calls()), (1, 1));
    });
}

#[test]
fn authority_search_quality_loading_obeys_the_virtual_deadline() {
    run_test_with_cx(|cx| async move {
        let fixture = Fixture::new(&cx, Topology::Quality).await;
        let mut runtime = fixture.runtime(Some("quality"), "semantic-query", 1);
        runtime.config.search.quality_timeout_ms = 10;
        let (timed, clock, timer) = timed_context(&cx);
        let _ambient = Cx::set_current(Some(timed.clone()));
        let mut output = Vec::new();
        let mut loads = 0;
        let mut query = Box::pin(runtime.run_authority_search_with_writer_and_loader(
            &timed,
            &mut output,
            |tier| {
                assert_eq!(tier, EmbedderTierTag::Quality);
                loads += 1;
                std::future::pending::<SearchResult<Arc<dyn Embedder>>>()
            },
        ));
        assert!(
            query
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
                .is_pending()
        );
        assert_eq!(timer.pending_count(), 1);
        clock.advance(10_000_000);
        let _ = timer.process_timers();
        assert!(matches!(
            query.as_mut().poll(&mut Context::from_waker(Waker::noop())),
            Poll::Ready(Err(SearchError::SearchTimeout { budget_ms: 10, .. }))
        ));
        drop(query);
        assert_eq!(loads, 1);
        assert!(output.is_empty());
        assert!(timer.is_empty());
        assert!(!timed.is_cancel_requested());
        assert_eq!((fixture.fast.calls(), fixture.quality.calls()), (0, 0));
        fixture.receipt.open(&cx, IMAGE_BUDGET).unwrap();
    });
}

#[test]
fn authority_search_model_loading_initial_and_refinement_share_one_budget() {
    run_test_with_cx(|cx| async move {
        let fixture = Fixture::new(&cx, Topology::Full).await;
        for pending_initial in [false, true] {
            let mut fast = Producer::new(EmbedderTierTag::Fast);
            fast.pending_query = pending_initial;
            let fast = Arc::new(fast);
            let mut quality = Producer::new(EmbedderTierTag::Quality);
            quality.pending_query = !pending_initial;
            let quality = Arc::new(quality);
            let mut runtime = fixture.runtime(Some("full"), "semantic-query", 1);
            runtime.config.search.quality_timeout_ms = 10;
            let (timed, clock, timer) = timed_context(&cx);
            let _ambient = Cx::set_current(Some(timed.clone()));
            let mut output = Vec::new();
            let mut loaded = Vec::new();
            let mut query = Box::pin(runtime.run_authority_search_with_writer_and_loader(
                &timed,
                &mut output,
                |tier| {
                    loaded.push(tier);
                    let producer: Arc<dyn Embedder> = match tier {
                        EmbedderTierTag::Fast => fast.clone(),
                        EmbedderTierTag::Quality => {
                            // Cold startup spends nine of the ten milliseconds.
                            clock.advance(9_000_000);
                            quality.clone()
                        }
                    };
                    ready(Ok(producer))
                },
            ));
            assert!(
                query
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop()))
                    .is_pending()
            );
            assert_eq!(fast.calls(), 1);
            assert_eq!(quality.calls(), usize::from(!pending_initial));
            assert_eq!(timer.pending_count(), 1);
            clock.advance(999_999);
            let _ = timer.process_timers();
            assert!(
                query
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop()))
                    .is_pending()
            );
            clock.advance(1);
            let _ = timer.process_timers();
            assert!(matches!(
                query.as_mut().poll(&mut Context::from_waker(Waker::noop())),
                Poll::Ready(Err(SearchError::SearchTimeout { budget_ms: 10, .. }))
            ));
            drop(query);
            assert_eq!(
                loaded,
                vec![EmbedderTierTag::Fast, EmbedderTierTag::Quality]
            );
            assert!(output.is_empty());
            assert!(timer.is_empty());
            assert!(!timed.is_cancel_requested());
        }
    });
}
