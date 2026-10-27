use super::*;
use std::future::Future;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::task::{Context, Poll, Waker};

use frankensearch_core::generation::{ActivationManifest, ArtifactGenerationIdentityV1};
use frankensearch_index::native_hnsw::HnswParams;
use frankensearch_quill::{KeeperSnapshot, QuillSnapshotIndex};
use sha2::{Digest as _, Sha256};

use super::super::tests::{Provider, Reply, documents, generation};
use crate::native_ann::NativeAnnIndex;
use crate::native_ann::builder::NativeBuildPrecision;
use crate::{Embedder, IndexableDocument, SearchError, SearchFuture};

const BUDGET: usize = 16 * 1024 * 1024;
const SOURCE: CommitRange = CommitRange { low: 4, high: 9 };

#[test]
fn empty_generations_retain_complete_single_and_progressive_images_without_inference() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for (label, tier, ann, quality_ann, full) in [
            ("fast", EmbedderTierTag::Fast, false, false, false),
            ("quality", EmbedderTierTag::Quality, false, false, false),
            ("fast-ann", EmbedderTierTag::Fast, true, false, false),
            ("quality-ann", EmbedderTierTag::Quality, true, false, false),
            ("full", EmbedderTierTag::Fast, false, false, true),
            ("full-fast-ann", EmbedderTierTag::Fast, true, false, true),
            ("full-quality-ann", EmbedderTierTag::Fast, false, true, true),
            ("full-both-ann", EmbedderTierTag::Fast, true, true, true),
        ] {
            for precision in [NativeBuildPrecision::F16, NativeBuildPrecision::F32] {
                let directory = tempfile::tempdir().unwrap();
                let work = directory.path().join(label);
                // Any document inference would cancel the build, including an
                // accidental empty batch sent to either provider.
                let provider = Arc::new(Provider::new(label, 2, Reply::Cancel));
                let mut configured = NativeIndexBuilder::new(&work, generation(), provider)
                    .unwrap()
                    .with_fast_storage(
                        precision,
                        if ann {
                            NativeBuildRetrieval::Hnsw {
                                params: HnswParams::default(),
                                seed: 79,
                            }
                        } else {
                            NativeBuildRetrieval::Exact
                        },
                    );
                if full {
                    configured = configured
                        .with_quality_embedder(Arc::new(Provider::new(
                            "independent-quality",
                            3,
                            Reply::Cancel,
                        )))
                        .unwrap()
                        .with_quality_storage(
                            precision,
                            if quality_ann {
                                NativeBuildRetrieval::Hnsw {
                                    params: HnswParams::default(),
                                    seed: 83,
                                }
                            } else {
                                NativeBuildRetrieval::Exact
                            },
                        )
                        .unwrap();
                }
                let mut built = if full {
                    configured
                        .build_progressive_generation(&cx, SOURCE, BUDGET)
                        .await
                } else {
                    configured
                        .build_exact_generation(&cx, tier, SOURCE, BUDGET)
                        .await
                }
                .unwrap();
                let candidate = candidate(&built);
                built.validate_candidate(&cx, &candidate).unwrap();
                let preview = built.preview_candidate(&cx, &candidate).unwrap();
                assert!(
                    preview
                        .search_text(&cx, &Provider::new(label, 2, Reply::Correct), "query", 1)
                        .await
                        .is_err(),
                    "an empty generation cannot prove successful semantic execution"
                );
                assert_eq!(built.metadata().total_documents, 0);
                assert_eq!(built.metadata().commit_range, SOURCE);
                assert_eq!(built.metadata().embedders.len(), if full { 2 } else { 1 });
                assert!(
                    built
                        .metadata()
                        .vector_artifacts
                        .iter()
                        .all(|item| item.vector_count == 0)
                );
                assert!(
                    built
                        .metadata()
                        .lexical_artifacts
                        .iter()
                        .all(|item| item.document_count == 0)
                );
                assert_eq!(built.components.has_ann(), ann || quality_ann);
                assert_eq!(
                    matches!(
                        built.retrieval_configuration(),
                        NativeBuildRetrieval::Hnsw { .. }
                    ),
                    ann
                );
                assert_eq!(built.quality_retrieval_configuration().is_some(), full);
                assert_eq!(
                    matches!(
                        built.quality_retrieval_configuration(),
                        Some(NativeBuildRetrieval::Hnsw { .. })
                    ),
                    quality_ann
                );
                assert_eq!(
                    built.components.docset_digest(),
                    CanonicalDocsetV1::from_ordered_live_documents(std::iter::empty::<&str>())
                        .unwrap()
                        .digest()
                );
                for (role, receipt) in candidate.manifest().components().iter() {
                    assert!(receipt.byte_len > 0);
                    assert_eq!(
                        built.component_bytes(role).unwrap().len() as u64,
                        receipt.byte_len
                    );
                }
                if full {
                    let cohort = ExactVectorCohortV1::admit(
                        &cx,
                        built.vector.clone(),
                        built.binding(),
                        built.quality_binding().unwrap(),
                        SourceCheckpointV1::derive(&SOURCE),
                        BUDGET,
                    )
                    .unwrap();
                    assert_eq!(cohort.fast().record_count(), 0);
                    assert_eq!(cohort.quality().record_count(), 0);
                    assert_ne!(
                        built.binding().dimension(),
                        built.quality_binding().unwrap().dimension()
                    );
                } else {
                    let owner = ValidatedFsviBytes::from_arc(built.vector.clone(), built.binding())
                        .unwrap();
                    assert_eq!(owner.record_count(), 0);
                }
                let readers = built.admit_candidate(&cx, &candidate).unwrap();
                assert!(readers.source_documents(&cx, 0).unwrap().is_empty());
                built
                    .prepare_scope(&cx, &candidate, 0, |_| panic!("empty corpus has no source"))
                    .unwrap();
                std::fs::rename(&work, directory.path().join("relocated")).unwrap();
                built.validate_candidate(&cx, &candidate).unwrap();

                // A complete empty generation still requires every declared image.
                let lexical = built.lexical.clone();
                built.lexical = Arc::from([]);
                assert!(built.validate_candidate(&cx, &candidate).is_err());
                built.lexical = lexical;
                if ann || quality_ann {
                    let graph = built.ann.take();
                    assert!(built.validate_candidate(&cx, &candidate).is_err());
                    built.ann = graph;
                }
                cx.set_cancel_requested(true);
                assert!(matches!(
                    built.validate_candidate(&cx, &candidate),
                    Err(SearchError::Cancelled { .. })
                ));
                cx.set_cancel_requested(false);
                built.validate_candidate(&cx, &candidate).unwrap();
            }
        }
    });
}

#[test]
fn empty_support_does_not_turn_missing_embedding_outputs_into_an_empty_generation() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for full in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let work = directory.path().join("failed");
            let fast = Arc::new(Provider::new(
                "fast",
                2,
                if full { Reply::Correct } else { Reply::Short },
            ));
            let mut configured = NativeIndexBuilder::new(&work, generation(), fast)
                .unwrap()
                .add_document(IndexableDocument::new("one", "source must have a vector"));
            if full {
                configured = configured
                    .with_quality_embedder(Arc::new(Provider::new("quality", 3, Reply::Short)))
                    .unwrap();
            }
            let result = if full {
                configured
                    .build_progressive_generation(&cx, SOURCE, BUDGET)
                    .await
            } else {
                configured
                    .build_exact_generation(&cx, EmbedderTierTag::Fast, SOURCE, BUDGET)
                    .await
            };
            assert!(result.is_err());
            assert!(!work.join("lexical").exists());
        }
    });
}

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
fn progressive_generation_requires_both_providers_before_inference_or_io() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for retrieval in [
            NativeBuildRetrieval::Exact,
            NativeBuildRetrieval::Hnsw {
                params: HnswParams::default(),
                seed: 5,
            },
        ] {
            let directory = tempfile::tempdir().unwrap();
            let work = directory.path().join("absent");
            let fast = Arc::new(Provider::new("fast", 2, Reply::Pending));
            let configured =
                builder(&work, &fast).with_fast_storage(NativeBuildPrecision::F32, retrieval);
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

#[test]
fn progressive_generation_packs_independent_graphs_and_preserves_mixed_policies() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for (fast_ann, quality_ann) in [(true, false), (false, true), (true, true)] {
            let directory = tempfile::tempdir().unwrap();
            let work = directory.path().join("work");
            let fast = Arc::new(Provider::new("fast", 2, Reply::Correct));
            let quality = Arc::new(Provider::new("quality", 3, Reply::Correct));
            let policy = |ann, seed| {
                if ann {
                    NativeBuildRetrieval::Hnsw {
                        params: HnswParams::default(),
                        seed,
                    }
                } else {
                    NativeBuildRetrieval::Exact
                }
            };
            let mut built = builder(&work, &fast)
                .with_quality_embedder(quality)
                .unwrap()
                .with_fast_storage(NativeBuildPrecision::F16, policy(fast_ann, 17))
                .with_quality_storage(NativeBuildPrecision::F32, policy(quality_ann, 29))
                .unwrap()
                .build_progressive_generation(&cx, SOURCE, BUDGET)
                .await
                .unwrap();
            let candidate = candidate(&built);
            built.validate_candidate(&cx, &candidate).unwrap();
            built
                .prepare_scope(&cx, &candidate, BUDGET, |document| Ok(document.id == "b"))
                .unwrap();
            assert_eq!(
                matches!(
                    built.retrieval_configuration(),
                    NativeBuildRetrieval::Hnsw { .. }
                ),
                fast_ann
            );
            assert_eq!(
                matches!(
                    built.quality_retrieval_configuration(),
                    Some(NativeBuildRetrieval::Hnsw { .. })
                ),
                quality_ann
            );
            let decode = |key: &str| {
                built.metadata().metadata.get(key).map(|encoded| {
                    serde_json::from_str::<NativeHnswGenerationReceiptV2>(encoded).unwrap()
                })
            };
            let fast_receipt = decode(NATIVE_ANN_RECEIPT_KEY);
            let quality_receipt = decode(NATIVE_QUALITY_ANN_RECEIPT_KEY);
            assert_eq!(fast_receipt.is_some(), fast_ann);
            assert_eq!(quality_receipt.is_some(), quality_ann);
            if let Some(receipt) = &fast_receipt {
                assert_eq!(receipt.seed, 17);
            }
            if let Some(receipt) = &quality_receipt {
                assert_eq!(receipt.seed, 29);
            }
            let vectors = ExactVectorCohortV1::admit(
                &cx,
                built
                    .component_bytes(GenerationComponentRole::Vector)
                    .unwrap(),
                built.binding(),
                built.quality_binding().unwrap(),
                SourceCheckpointV1::derive(&SOURCE),
                BUDGET,
            )
            .unwrap();
            let ann = built.component_bytes(GenerationComponentRole::Ann).unwrap();
            let graphs = NativeAnnCohortV1::admit(
                &cx,
                Arc::clone(&ann),
                &vectors,
                fast_receipt.as_ref(),
                quality_receipt.as_ref(),
                BUDGET,
            )
            .unwrap();
            assert_eq!(graphs.fast().is_some(), fast_ann);
            assert_eq!(graphs.quality().is_some(), quality_ann);
            assert_eq!(
                Some(graphs.receipt().bytes),
                candidate.manifest().components().ann
            );
            let owned = usize::try_from(candidate.owned_image_bytes()).unwrap();
            built.max_image_bytes = owned - 1;
            assert!(built.validate_candidate(&cx, &candidate).is_err());
            built.max_image_bytes = BUDGET;
            let mut corrupt = ann.to_vec();
            *corrupt.last_mut().unwrap() ^= 1;
            built.ann = Some(corrupt.into());
            assert!(built.validate_candidate(&cx, &candidate).is_err());
            built.ann = Some(ann);
            std::fs::rename(&work, directory.path().join("relocated")).unwrap();
            built.validate_candidate(&cx, &candidate).unwrap();
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
            assert!(built.preview_candidate(&cx, &foreign).is_err());
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
                CommitRange { low: 9, high: 4 },
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

#[test]
fn unpublished_candidate_previews_execute_every_tier_and_retain_original_images() {
    use crate::native_ann::NativeGenerationTopology;

    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for (tier, full, fast_ann, quality_ann) in [
            (EmbedderTierTag::Fast, false, false, false),
            (EmbedderTierTag::Quality, false, false, false),
            (EmbedderTierTag::Fast, false, true, false),
            (EmbedderTierTag::Quality, false, true, false),
            (EmbedderTierTag::Fast, true, false, false),
            (EmbedderTierTag::Fast, true, true, false),
            (EmbedderTierTag::Fast, true, false, true),
            (EmbedderTierTag::Fast, true, true, true),
        ] {
            let directory = tempfile::tempdir().unwrap();
            let work = directory.path().join("work");
            let primary = Arc::new(Provider::new("primary", 2, Reply::Correct));
            let quality = Arc::new(Provider::new("independent-quality", 3, Reply::Correct));
            let retrieval = |ann| {
                if ann {
                    NativeBuildRetrieval::Hnsw {
                        params: HnswParams::default(),
                        seed: 91,
                    }
                } else {
                    NativeBuildRetrieval::Exact
                }
            };
            let mut configured = builder(&work, &primary)
                .with_fast_storage(NativeBuildPrecision::F32, retrieval(fast_ann));
            if full {
                configured = configured
                    .with_quality_embedder(quality.clone())
                    .unwrap()
                    .with_quality_storage(NativeBuildPrecision::F32, retrieval(quality_ann))
                    .unwrap();
            }
            let built = if full {
                configured
                    .build_progressive_generation(&cx, SOURCE, BUDGET)
                    .await
            } else {
                configured
                    .build_exact_generation(&cx, tier, SOURCE, BUDGET)
                    .await
            }
            .unwrap();
            let candidate = candidate(&built);
            let preview = Arc::new(built.preview_candidate(&cx, &candidate).unwrap());
            assert_eq!(preview.candidate_authority(), candidate.authority());
            assert_eq!(preview.metadata().total_documents, 5);
            assert_eq!(
                preview.topology(),
                if full {
                    NativeGenerationTopology::FullProgressive
                } else if tier == EmbedderTierTag::Quality {
                    NativeGenerationTopology::QualityOnly
                } else {
                    NativeGenerationTopology::FastOnly
                }
            );
            assert_eq!(primary.queries.load(Ordering::SeqCst), 0);
            assert_eq!(quality.queries.load(Ordering::SeqCst), 0);
            let predicates = AtomicUsize::new(0);
            let scope = preview
                .scope(&cx, BUDGET, |document| {
                    predicates.fetch_add(1, Ordering::SeqCst);
                    Ok(document.id == "b")
                })
                .unwrap();
            assert_eq!(predicates.load(Ordering::SeqCst), 5);
            assert_eq!(scope.len(), 1);
            assert_eq!(scope.candidate_authority(), candidate.authority());
            assert!(scope.document("a").is_none());
            assert_eq!(scope.document("b").unwrap().content, "vertical");
            drop(built);
            std::fs::rename(&work, directory.path().join("moved")).unwrap();
            std::fs::create_dir(&work).unwrap();
            std::fs::write(work.join("fast.fsvi"), b"unrelated candidate").unwrap();
            let hits = preview
                .search_text(&cx, primary.as_ref(), "vertical", 2)
                .await
                .unwrap();
            assert_eq!(hits[0].doc_id, "b");
            assert_eq!(primary.queries.load(Ordering::SeqCst), 1);
            assert_eq!(hits[0].metadata.as_ref().unwrap()["source_cut"], "original");
            let scoped = scope
                .search_text(&cx, primary.as_ref(), "vertical", 10)
                .await
                .unwrap();
            assert_eq!(scoped.len(), 1);
            assert_eq!(scoped[0].doc_id, "b");
            if tier == EmbedderTierTag::Quality {
                assert!(scoped[0].fast_score.is_none());
                assert!(scoped[0].quality_score.is_some());
                let quality_only = scope
                    .search_quality_text(&cx, primary.as_ref(), "vertical", 10)
                    .await
                    .unwrap();
                assert_eq!(
                    serde_json::to_value(&scoped).unwrap(),
                    serde_json::to_value(&quality_only).unwrap()
                );
            }
            if full {
                let before = primary.queries.load(Ordering::SeqCst);
                let quality_hits = preview
                    .search_quality_text(&cx, quality.as_ref(), "vertical", 1)
                    .await
                    .unwrap();
                assert_eq!(primary.queries.load(Ordering::SeqCst), before);
                assert_eq!(quality_hits[0].doc_id, "b");
                assert!(quality_hits[0].fast_score.is_none());
                assert!(quality_hits[0].quality_score.is_some());
                let refined = preview
                    .search_refined_text(&cx, primary.as_ref(), quality.as_ref(), "vertical", 1)
                    .await
                    .unwrap();
                assert_eq!(refined[0].doc_id, "b");
                assert!(refined[0].fast_score.is_some());
                assert!(refined[0].quality_score.is_some());
                let scoped = scope
                    .search_refined_text(&cx, primary.as_ref(), quality.as_ref(), "vertical", 1)
                    .await
                    .unwrap();
                assert_eq!(scoped[0].doc_id, "b");
                assert_eq!(quality.queries.load(Ordering::SeqCst), 3);
            } else if tier == EmbedderTierTag::Fast {
                assert!(
                    preview
                        .search_quality_text(&cx, quality.as_ref(), "vertical", 1)
                        .await
                        .is_err()
                );
                assert_eq!(quality.queries.load(Ordering::SeqCst), 0);
            }
            assert_eq!(predicates.load(Ordering::SeqCst), 5);
            assert!(!directory.path().join("AUTHORITY").exists());
            assert!(!work.join("AUTHORITY").exists());
        }
    });
}

#[test]
fn candidate_preview_scopes_widen_before_fusion_and_never_accept_noop_validation() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for ann in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let provider = Arc::new(Provider::new("scope", 2, Reply::Correct));
            let mut source: Vec<_> = (0..24)
                .map(|index| {
                    IndexableDocument::new(format!("private-{index:02}"), "vertical")
                        .with_metadata("visible", "false")
                })
                .collect();
            for id in ["public-a", "public-b"] {
                source.push(
                    IndexableDocument::new(id, "vertical with enough padding to rank later")
                        .with_title("retained public source")
                        .with_metadata("visible", "true"),
                );
            }
            let built = NativeIndexBuilder::new(
                directory.path().join("work"),
                generation(),
                provider.clone(),
            )
            .unwrap()
            .add_documents(source)
            .with_fast_storage(
                NativeBuildPrecision::F32,
                if ann {
                    NativeBuildRetrieval::Hnsw {
                        params: HnswParams {
                            ef_search: 1,
                            ..HnswParams::default()
                        },
                        seed: 93,
                    }
                } else {
                    NativeBuildRetrieval::Exact
                },
            )
            .build_exact_generation(&cx, EmbedderTierTag::Fast, SOURCE, BUDGET)
            .await
            .unwrap();
            let preview = Arc::new(built.preview_candidate(&cx, &candidate(&built)).unwrap());
            let predicates = AtomicUsize::new(0);
            assert!(
                preview
                    .scope(&cx, 1, |_| {
                        predicates.fetch_add(1, Ordering::SeqCst);
                        Ok(true)
                    })
                    .is_err()
            );
            assert_eq!(predicates.load(Ordering::SeqCst), 0);
            let scope = preview
                .scope(&cx, BUDGET, |document| {
                    predicates.fetch_add(1, Ordering::SeqCst);
                    Ok(document.metadata.get("visible").map(String::as_str) == Some("true"))
                })
                .unwrap();
            let hits = scope
                .search_text(&cx, provider.as_ref(), "vertical", 2)
                .await
                .unwrap();
            assert_eq!(hits.len(), 2);
            assert!(hits.iter().all(|hit| hit.doc_id.starts_with("public-")));
            assert!(
                hits.iter()
                    .all(|hit| hit.metadata.as_ref().unwrap()["visible"] == "true")
            );
            assert_eq!(predicates.load(Ordering::SeqCst), 26);
            let before = provider.queries.load(Ordering::SeqCst);
            for (query, limit) in [("", 1), (" \n\t", 1), ("vertical", 0)] {
                assert!(
                    preview
                        .search_text(&cx, provider.as_ref(), query, limit)
                        .await
                        .is_err()
                );
                assert!(
                    scope
                        .search_text(&cx, provider.as_ref(), query, limit)
                        .await
                        .is_err()
                );
            }
            let empty = preview.scope(&cx, BUDGET, |_| Ok(false)).unwrap();
            assert!(empty.is_empty());
            assert!(
                empty
                    .search_text(&cx, provider.as_ref(), "vertical", 1)
                    .await
                    .is_err()
            );
            assert_eq!(provider.queries.load(Ordering::SeqCst), before);
            let foreign = Provider::new("other-producer", 2, Reply::Correct);
            assert!(
                scope
                    .search_text(&cx, &foreign, "vertical", 1)
                    .await
                    .is_err()
            );
            assert_eq!(foreign.queries.load(Ordering::SeqCst), 0);
            cx.set_cancel_requested(true);
            assert!(matches!(
                scope
                    .search_text(&cx, provider.as_ref(), "vertical", 1)
                    .await,
                Err(SearchError::Cancelled { .. })
            ));
            cx.set_cancel_requested(false);
            assert_eq!(provider.queries.load(Ordering::SeqCst), before);
            assert_eq!(
                scope
                    .search_text(&cx, provider.as_ref(), "vertical", 2)
                    .await
                    .unwrap()
                    .len(),
                2
            );
        }
    });
}

struct PreviewQueryProvider {
    inner: Provider,
    fault: AtomicU8,
}

impl PreviewQueryProvider {
    fn new(name: &str, dimension: u32) -> Self {
        Self {
            inner: Provider::new(name, dimension, Reply::Correct),
            fault: AtomicU8::new(0),
        }
    }
}

impl Embedder for PreviewQueryProvider {
    fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        self.inner.embed(cx, text)
    }

    fn embed_batch_bound<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<frankensearch_core::IdentityBoundEmbedding>> {
        self.inner.embed_batch_bound(cx, texts)
    }

    fn embed_bound<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
    ) -> SearchFuture<'a, frankensearch_core::IdentityBoundEmbedding> {
        Box::pin(async move {
            let values = self.inner.embed(cx, text).await?;
            let mut identity = self.inner.identity()?.clone();
            match self.fault.load(Ordering::SeqCst) {
                1 => return Err(invalid("preview.test", "failed", "preview producer failed")),
                2 => identity.producer.backend = "changed-after-admission".to_owned(),
                3 => {
                    cx.set_cancel_requested(true);
                }
                4 => return std::future::pending().await,
                _ => {}
            }
            Ok(frankensearch_core::IdentityBoundEmbedding { values, identity })
        })
    }

    fn identity(&self) -> SearchResult<&frankensearch_core::generation::EmbeddingIdentityBundleV1> {
        self.inner.identity()
    }

    fn bound_batch_is_native(&self) -> bool {
        true
    }
    fn id(&self) -> &str {
        self.inner.id()
    }
    fn model_name(&self) -> &str {
        self.inner.model_name()
    }
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }
    fn is_semantic(&self) -> bool {
        self.inner.is_semantic()
    }
    fn category(&self) -> crate::ModelCategory {
        self.inner.category()
    }
}

#[test]
fn full_candidate_preview_rejects_failed_drifted_cancelled_and_abandoned_refinement() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let fast = Arc::new(PreviewQueryProvider::new("fast", 2));
        let quality = Arc::new(PreviewQueryProvider::new("quality", 3));
        let built =
            NativeIndexBuilder::new(directory.path().join("work"), generation(), fast.clone())
                .unwrap()
                .with_quality_embedder(quality.clone())
                .unwrap()
                .add_documents(documents())
                .build_progressive_generation(&cx, SOURCE, BUDGET)
                .await
                .unwrap();
        let candidate = candidate(&built);
        let preview = Arc::new(built.preview_candidate(&cx, &candidate).unwrap());
        let scope = preview.scope(&cx, BUDGET, |_| Ok(true)).unwrap();
        for fault in [1, 2, 3] {
            quality.fault.store(fault, Ordering::SeqCst);
            assert!(
                preview
                    .search_refined_text(&cx, fast.as_ref(), quality.as_ref(), "vertical", 1)
                    .await
                    .is_err()
            );
            cx.set_cancel_requested(false);
            assert!(
                scope
                    .search_refined_text(&cx, fast.as_ref(), quality.as_ref(), "vertical", 1)
                    .await
                    .is_err()
            );
            cx.set_cancel_requested(false);
        }
        quality.fault.store(4, Ordering::SeqCst);
        let mut pending = Box::pin(scope.search_refined_text(
            &cx,
            fast.as_ref(),
            quality.as_ref(),
            "vertical",
            1,
        ));
        assert!(
            pending
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
                .is_pending()
        );
        let queries = quality.inner.queries.load(Ordering::SeqCst);
        drop(pending);
        assert_eq!(quality.inner.queries.load(Ordering::SeqCst), queries);
        quality.fault.store(0, Ordering::SeqCst);
        let hits = scope
            .search_refined_text(&cx, fast.as_ref(), quality.as_ref(), "vertical", 1)
            .await
            .unwrap();
        assert_eq!(hits[0].doc_id, "b");
        assert!(hits[0].quality_score.is_some());
        assert_eq!(preview.candidate_authority(), candidate.authority());
        assert!(!directory.path().join("AUTHORITY").exists());
    });
}
