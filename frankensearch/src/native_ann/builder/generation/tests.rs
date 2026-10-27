use super::*;
use std::future::Future;
use std::task::{Context, Poll, Waker};

use frankensearch_core::generation::{ActivationManifest, ArtifactGenerationIdentityV1};
use frankensearch_index::native_hnsw::HnswParams;
use frankensearch_quill::{KeeperSnapshot, QuillSnapshotIndex};
use sha2::{Digest as _, Sha256};

use super::super::tests::{Provider, Reply, documents, generation};
use crate::native_ann::NativeAnnIndex;
use crate::native_ann::builder::NativeBuildPrecision;
use crate::{IndexableDocument, SearchError};

const BUDGET: usize = 16 * 1024 * 1024;
const SOURCE: CommitRange = CommitRange { low: 4, high: 9 };

#[test]
fn progressive_generation_retains_independent_tiers_and_one_complete_source_cohort() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        // The second case deliberately has identical explicit providers and
        // identical vector images: role-qualified metadata paths remain unique.
        for (quality_name, quality_dimension) in [("quality", 3), ("fast", 2)] {
            let directory = tempfile::tempdir().unwrap();
            let work = directory.path().join("work");
            let fast = Arc::new(Provider::new("fast", 2, Reply::Correct));
            let quality = Arc::new(Provider::new(
                quality_name,
                quality_dimension,
                Reply::Correct,
            ));
            let built = builder(&work, &fast)
                .with_quality_embedder(quality)
                .unwrap()
                .with_fast_storage(NativeBuildPrecision::F32, NativeBuildRetrieval::Exact)
                .with_quality_storage(NativeBuildPrecision::F32, NativeBuildRetrieval::Exact)
                .unwrap()
                .with_batch_size(2)
                .unwrap()
                .build_progressive_generation(&cx, SOURCE, BUDGET)
                .await
                .unwrap();
            let candidate = candidate(&built);
            built.validate_candidate(&cx, &candidate).unwrap();
            let mut scoped_ids = Vec::new();
            built
                .prepare_scope(&cx, &candidate, BUDGET, |document| {
                    scoped_ids.push(document.id.clone());
                    Ok(document.id != "a")
                })
                .unwrap();
            assert_eq!(scoped_ids.len(), 5);
            assert_eq!(
                scoped_ids
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len(),
                5
            );
            assert!(
                built
                    .prepare_scope(&cx, &candidate, 0, |_| {
                        panic!("the complete two-tier source export must fit before predicates run")
                    })
                    .is_err()
            );
            let quality_binding = built.quality_binding().unwrap();
            assert_eq!(
                quality_binding.dimension(),
                usize::try_from(quality_dimension).unwrap()
            );
            assert_eq!(quality_binding.generation(), built.binding().generation());
            assert_eq!(built.metadata().total_documents, 5);
            assert_eq!(built.metadata().embedders.len(), 2);
            assert_eq!(built.metadata().vector_artifacts.len(), 2);
            assert_eq!(
                built.metadata().vector_artifacts[0].embedder_tier,
                EmbedderTierTag::Fast
            );
            assert_eq!(
                built.metadata().vector_artifacts[1].embedder_tier,
                EmbedderTierTag::Quality
            );
            assert_ne!(
                built.metadata().vector_artifacts[0].path,
                built.metadata().vector_artifacts[1].path
            );
            let cohort = ExactVectorCohortV1::admit(
                &cx,
                built
                    .component_bytes(GenerationComponentRole::Vector)
                    .unwrap(),
                built.binding(),
                quality_binding,
                SourceCheckpointV1::derive(&SOURCE),
                BUDGET,
            )
            .unwrap();
            assert_eq!(cohort.receipt(), built.components.vector());
            assert_eq!(cohort.fast_receipt().live_document_count, 5);
            assert_eq!(cohort.quality_receipt().live_document_count, 5);
            assert!(
                built
                    .metadata()
                    .activation_invariants
                    .iter()
                    .any(|invariant| matches!(
                        invariant.kind,
                        InvariantKind::VectorCountConsistency { expected_total: 10 }
                    ))
            );
            assert!(
                crate::native_ann::generation::admit_components(
                    &cx,
                    candidate.manifest(),
                    built.binding(),
                    DEFAULT_SCHEMA,
                    QuillConfig::default(),
                    [
                        built
                            .component_bytes(GenerationComponentRole::Vector)
                            .unwrap(),
                        built
                            .component_bytes(GenerationComponentRole::Lexical)
                            .unwrap(),
                        built
                            .component_bytes(GenerationComponentRole::Metadata)
                            .unwrap()
                    ],
                    None,
                    BUDGET,
                )
                .is_err(),
                "single-tier admission cannot discard the quality owner"
            );
            std::fs::rename(&work, directory.path().join("relocated")).unwrap();
            std::fs::create_dir(&work).unwrap();
            std::fs::write(work.join("fast.fsvi"), b"unrelated future generation").unwrap();
            built.validate_candidate(&cx, &candidate).unwrap();
        }
    });
}

#[test]
fn progressive_generation_requires_two_exact_plans_before_inference_or_io() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for plan in 0..3 {
            let directory = tempfile::tempdir().unwrap();
            let work = directory.path().join("absent");
            let fast = Arc::new(Provider::new("fast", 2, Reply::Pending));
            let mut configured = builder(&work, &fast);
            if plan != 0 {
                configured = configured
                    .with_quality_embedder(Arc::new(Provider::new("quality", 3, Reply::Pending)))
                    .unwrap();
                let ann = NativeBuildRetrieval::Hnsw {
                    params: HnswParams::default(),
                    seed: 5,
                };
                configured = if plan == 1 {
                    configured.with_fast_storage(NativeBuildPrecision::F32, ann)
                } else {
                    configured
                        .with_quality_storage(NativeBuildPrecision::F32, ann)
                        .unwrap()
                };
            }
            let mut future = Box::pin(configured.build_progressive_generation(&cx, SOURCE, BUDGET));
            assert!(matches!(
                future
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop())),
                Poll::Ready(Err(SearchError::InvalidConfig { .. }))
            ));
            assert!(!work.exists());
        }
    });
}

fn builder(path: &std::path::Path, provider: &Arc<Provider>) -> NativeIndexBuilder {
    NativeIndexBuilder::new(path, generation(), provider.clone())
        .unwrap()
        .add_documents(documents().into_iter().map(|document| {
            document
                .with_title("retained source title")
                .with_metadata("source_cut", "original")
        }))
}

fn candidate(built: &NativeBuiltExactGeneration) -> GenerationCandidateV1 {
    built
        .candidate(
            [0x49; 16],
            None,
            GenerationAuthorityActionV1::Activate,
            [0x57; 32],
        )
        .unwrap()
}

#[test]
fn real_fast_and_quality_images_join_and_query_after_source_relocation() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for (tier, precision, name) in [
            (EmbedderTierTag::Fast, NativeBuildPrecision::F32, "fast"),
            (
                EmbedderTierTag::Quality,
                NativeBuildPrecision::F16,
                "quality",
            ),
        ] {
            let directory = tempfile::tempdir().unwrap();
            let work = directory.path().join("work");
            let provider = Arc::new(Provider::new(name, 2, Reply::Correct));
            let built = builder(&work, &provider)
                .with_fast_storage(precision, NativeBuildRetrieval::Exact)
                .with_batch_size(2)
                .unwrap()
                .build_exact_generation(&cx, tier, SOURCE, BUDGET)
                .await
                .unwrap();
            let candidate = candidate(&built);
            assert!(matches!(candidate.manifest(), ActivationManifest::V2(_)));
            assert!(!built.components.has_ann());
            assert!(
                built
                    .component_bytes(GenerationComponentRole::Ann)
                    .is_none()
            );
            assert_eq!(built.metadata().total_documents, 5);
            assert_eq!(built.metadata().vector_artifacts[0].vector_count, 5);
            assert_eq!(built.metadata().vector_artifacts[0].embedder_tier, tier);
            assert_eq!(built.metadata().embedders.len(), 1);
            assert_eq!(
                built.metadata().embedders[name],
                *built.binding().frozen_identity()
            );
            assert_eq!(built.metadata().commit_range, SOURCE);
            assert!(validate_manifest(built.metadata()).is_valid());
            assert!(candidate.owned_image_bytes() <= u64::try_from(BUDGET).unwrap());
            for receipt in [
                built.components.vector(),
                built.components.lexical(),
                built.components.metadata(),
            ] {
                let bytes = built.component_bytes(receipt.role).unwrap();
                assert_eq!(receipt.bytes.byte_len, u64::try_from(bytes.len()).unwrap());
                let hash: [u8; 32] = Sha256::digest(&bytes).into();
                assert_eq!(receipt.bytes.sha256, hash);
                assert_eq!(
                    receipt.source_checkpoint,
                    SourceCheckpointV1::derive(&SOURCE).to_bytes()
                );
            }
            let owner = Arc::new(
                ValidatedFsviBytes::from_arc(
                    built
                        .component_bytes(GenerationComponentRole::Vector)
                        .unwrap(),
                    built.binding(),
                )
                .unwrap(),
            );
            let physical: Vec<_> = (0..owner.record_count())
                .map(|row| owner.row(row).unwrap().doc_id().to_owned())
                .collect();
            assert_ne!(physical, ["a", "b", "c", "x", "z"]);
            assert_eq!(
                CanonicalDocsetV1::from_ordered_live_documents(physical)
                    .unwrap()
                    .digest(),
                built.components.docset_digest(),
            );
            std::fs::rename(&work, directory.path().join("moved")).unwrap();
            std::fs::create_dir(&work).unwrap();
            std::fs::write(work.join("fast.fsvi"), b"unrelated successor").unwrap();
            built.validate_candidate(&cx, &candidate).unwrap();
            let lexical = KeeperSnapshot::from_exact_image(
                &cx,
                built
                    .component_bytes(GenerationComponentRole::Lexical)
                    .unwrap(),
                DEFAULT_SCHEMA,
                SourceCheckpointV1::derive(&SOURCE),
                built.components.lexical(),
                BUDGET,
            )
            .unwrap();
            let lexical = QuillSnapshotIndex::new(&cx, lexical, QuillConfig::default()).unwrap();
            assert_eq!(lexical.doc_count().unwrap(), 5);
            let vector = NativeAnnIndex::exact(&cx, owner).unwrap();
            let hits = match tier {
                EmbedderTierTag::Fast => {
                    vector
                        .search_hybrid_text(&cx, provider.as_ref(), &lexical, "vertical", 5)
                        .await
                }
                EmbedderTierTag::Quality => {
                    vector
                        .search_hybrid_quality_text(&cx, provider.as_ref(), &lexical, "vertical", 5)
                        .await
                }
            }
            .unwrap();
            assert_eq!(hits[0].doc_id, "b");
            assert_eq!(hits[0].metadata.as_ref().unwrap()["source_cut"], "original");
        }
    });
}

#[test]
fn candidate_validation_refuses_rebound_bytes_checkpoint_and_generation() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let provider = Arc::new(Provider::new("fast", 2, Reply::Correct));
        let built = builder(&directory.path().join("work"), &provider)
            .build_exact_generation(&cx, EmbedderTierTag::Fast, SOURCE, BUDGET)
            .await
            .unwrap();
        for mutation in 0..3 {
            let mut vector = built.components.vector().clone();
            let mut lexical = built.components.lexical().clone();
            let mut metadata = built.components.metadata().clone();
            let mut generation = built.binding().generation();
            match mutation {
                0 => vector.bytes.sha256[0] ^= 1,
                1 => {
                    let source =
                        SourceCheckpointV1::derive(&CommitRange { low: 5, high: 9 }).to_bytes();
                    vector.source_checkpoint = source;
                    lexical.source_checkpoint = source;
                    metadata.source_checkpoint = source;
                }
                _ => generation = ArtifactGenerationIdentityV1::new(10, [0x50; 16]).unwrap(),
            }
            let components =
                ExactGenerationComponentsV1::admit(vector, lexical, None, metadata).unwrap();
            let foreign = GenerationCandidateV1::new(
                [0x50; 16],
                None,
                GenerationAuthorityActionV1::Activate,
                generation,
                [0x58; 32],
                &components,
            )
            .unwrap();
            assert!(built.validate_candidate(&cx, &foreign).is_err());
        }
        built.validate_candidate(&cx, &candidate(&built)).unwrap();
    });
}

#[test]
fn unsupported_builds_fail_before_files_or_a_pending_provider_is_polled() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let provider = Arc::new(Provider::new("never-polled", 2, Reply::Pending));
        let plans = [
            (
                builder(&directory.path().join("ann"), &provider).with_fast_storage(
                    NativeBuildPrecision::F32,
                    NativeBuildRetrieval::Hnsw {
                        params: HnswParams {
                            m: 0,
                            ..HnswParams::default()
                        },
                        seed: 9,
                    },
                ),
                SOURCE,
                BUDGET,
            ),
            (
                builder(&directory.path().join("two"), &provider)
                    .with_quality_embedder(provider.clone())
                    .unwrap(),
                SOURCE,
                BUDGET,
            ),
            (
                NativeIndexBuilder::new(
                    directory.path().join("empty"),
                    generation(),
                    provider.clone(),
                )
                .unwrap(),
                SOURCE,
                BUDGET,
            ),
            (
                builder(&directory.path().join("range"), &provider),
                CommitRange { low: 9, high: 4 },
                BUDGET,
            ),
            (
                builder(&directory.path().join("overflow-range"), &provider),
                CommitRange {
                    low: 0,
                    high: u64::MAX,
                },
                BUDGET,
            ),
            (
                builder(&directory.path().join("budget"), &provider),
                SOURCE,
                0,
            ),
            (
                builder(&directory.path().join("duplicate"), &provider)
                    .add_document(IndexableDocument::new("a", "duplicate")),
                SOURCE,
                BUDGET,
            ),
        ];
        for (builder, source, budget) in plans {
            let path = builder.directory.clone();
            let mut future = Box::pin(builder.build_exact_generation(
                &cx,
                EmbedderTierTag::Fast,
                source,
                budget,
            ));
            let poll = future
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()));
            assert!(matches!(
                poll,
                Poll::Ready(Err(SearchError::InvalidConfig { .. }))
            ));
            assert!(!path.exists());
        }
    });
}

#[test]
fn capture_requires_the_original_exact_vector_witness_and_budget() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let provider = Arc::new(Provider::new("fast", 2, Reply::Correct));
        let vectors = builder(&directory.path().join("work"), &provider)
            .build(&cx)
            .await
            .unwrap();
        let (bytes, _) = capture_vector(&cx, vectors.fast(), BUDGET).unwrap();
        assert!(capture_vector(&cx, vectors.fast(), bytes.len() - 1).is_err());
        let mut altered = bytes.to_vec();
        *altered.last_mut().unwrap() ^= 1;
        std::fs::write(vectors.fast().vector_path(), altered).unwrap();
        assert!(capture_vector(&cx, vectors.fast(), BUDGET).is_err());
        let hits = vectors.fast().search(&cx, "vertical", 1).await.unwrap();
        assert_eq!(hits[0].doc_id, "b");
    });
}

#[test]
fn native_generation_retains_all_four_images_policy_and_budget_after_relocation() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for (tier, precision, name) in [
            (EmbedderTierTag::Fast, NativeBuildPrecision::F16, "fast"),
            (
                EmbedderTierTag::Quality,
                NativeBuildPrecision::F32,
                "quality",
            ),
        ] {
            let directory = tempfile::tempdir().unwrap();
            let work = directory.path().join("work");
            let provider = Arc::new(Provider::new(name, 2, Reply::Correct));
            let params = HnswParams {
                m: 4,
                m0: 8,
                ef_construction: 24,
                ef_search: 17,
            };
            let seed = 0x453;
            let mut built = builder(&work, &provider)
                .with_fast_storage(precision, NativeBuildRetrieval::Hnsw { params, seed })
                .build_exact_generation(&cx, tier, SOURCE, BUDGET)
                .await
                .unwrap();
            let candidate = candidate(&built);
            assert!(built.components.has_ann());
            assert_eq!(built.retrieval_mode(), NativeRetrievalMode::Ann);
            match built.retrieval_configuration() {
                NativeBuildRetrieval::Hnsw {
                    params: actual,
                    seed: actual_seed,
                } => {
                    assert_eq!(actual, params);
                    assert_eq!(actual_seed, seed);
                }
                NativeBuildRetrieval::Exact => panic!("the selected graph must be retained"),
            }
            let receipt: NativeHnswGenerationReceiptV2 =
                serde_json::from_str(&built.metadata().metadata[NATIVE_ANN_RECEIPT_KEY]).unwrap();
            let ann = built.component_bytes(GenerationComponentRole::Ann).unwrap();
            assert_eq!(receipt.seed, seed);
            assert_eq!(receipt.graph_byte_len, u64::try_from(ann.len()).unwrap());
            let raw_hash: [u8; 32] = Sha256::digest(&ann).into();
            assert_eq!(receipt.graph_sha256, lower_hex(&raw_hash));
            assert_eq!(
                candidate.manifest().components().ann.unwrap().sha256,
                raw_hash
            );
            assert_eq!(built.metadata().vector_artifacts[0].embedder_tier, tier);
            let owned = [
                GenerationComponentRole::Vector,
                GenerationComponentRole::Lexical,
                GenerationComponentRole::Ann,
                GenerationComponentRole::Metadata,
            ]
            .into_iter()
            .map(|role| built.component_bytes(role).unwrap().len())
            .sum::<usize>()
                + candidate.manifest().canonical_bytes().len();
            assert_eq!(candidate.owned_image_bytes(), u64::try_from(owned).unwrap());
            std::fs::rename(&work, directory.path().join("moved")).unwrap();
            std::fs::create_dir(&work).unwrap();
            std::fs::write(work.join("fast.fshnsw"), b"unrelated successor").unwrap();
            built.validate_candidate(&cx, &candidate).unwrap();

            built.max_image_bytes = owned - ann.len();
            assert!(built.validate_candidate(&cx, &candidate).is_err());
            assert!(
                built
                    .candidate(
                        [0x69; 16],
                        None,
                        GenerationAuthorityActionV1::Activate,
                        [0x79; 32]
                    )
                    .is_err()
            );
            built.max_image_bytes = BUDGET;
            let mut corrupt = ann.to_vec();
            *corrupt.last_mut().unwrap() ^= 1;
            built.ann = Some(corrupt.into());
            assert!(built.validate_candidate(&cx, &candidate).is_err());
            built.ann = Some(ann);
            built.validate_candidate(&cx, &candidate).unwrap();
        }
    });
}

#[test]
fn native_capture_rejects_changed_graph_bytes_and_includes_the_graph_budget() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let provider = Arc::new(Provider::new("fast", 2, Reply::Correct));
        let vectors = builder(&directory.path().join("work"), &provider)
            .with_fast_storage(
                NativeBuildPrecision::F32,
                NativeBuildRetrieval::Hnsw {
                    params: HnswParams::default(),
                    seed: 73,
                },
            )
            .build(&cx)
            .await
            .unwrap();
        let (_, owner) = capture_vector(&cx, vectors.fast(), BUDGET).unwrap();
        let (bytes, receipt) = capture_ann(&cx, vectors.fast(), Arc::clone(&owner), BUDGET)
            .unwrap()
            .unwrap();
        assert_eq!(receipt, *vectors.fast().graph_receipt.as_ref().unwrap());
        assert!(capture_ann(&cx, vectors.fast(), Arc::clone(&owner), bytes.len() - 1).is_err());
        let mut altered = bytes.to_vec();
        *altered.last_mut().unwrap() ^= 1;
        std::fs::write(vectors.fast().graph_path().unwrap(), altered).unwrap();
        assert!(capture_ann(&cx, vectors.fast(), owner, BUDGET).is_err());
        let hits = vectors.fast().search(&cx, "vertical", 1).await.unwrap();
        assert_eq!(hits[0].doc_id, "b");
    });
}

#[test]
fn cancellation_never_returns_a_partial_component_payload() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let work = directory.path().join("work");
        let provider = Arc::new(Provider::new("cancel", 2, Reply::Cancel));
        let result = builder(&work, &provider)
            .build_exact_generation(&cx, EmbedderTierTag::Fast, SOURCE, BUDGET)
            .await;
        assert!(matches!(result, Err(SearchError::Cancelled { .. })));
        assert!(!work.join("lexical").exists());
    });
}
