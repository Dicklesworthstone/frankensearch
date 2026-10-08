use super::*;
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Waker};

use asupersync::test_utils::run_test_with_cx;
use frankensearch_core::generation::{EmbeddingArtifactIdentityV1, EmbeddingSpaceKindV1};
use frankensearch_core::traits::{IdentityBoundEmbedding, ModelCategory, SearchFuture};
use frankensearch_index::native_hnsw::HnswParams;

use crate::native_ann::NativeSearchPhase;
use crate::native_ann::builder::live::NativeLiveHybridIndex;
use crate::{IndexableDocument, SearchError};

const FAIL: usize = 1;
const FOREIGN: usize = 2;
const PENDING: usize = 3;
const CANCEL: usize = 4;

// Test doubles exercise real v2 writers, admitted owners, Quill and live swaps.
// Semantic-shaped identities are synthetic, not model-quality evidence.
struct Provider {
    identity: EmbeddingIdentityBundleV1,
    documents: AtomicUsize,
    queries: AtomicUsize,
    raw_calls: AtomicUsize,
    drops: AtomicUsize,
    fault: AtomicUsize,
}

impl Provider {
    fn new(name: &str, dimension: u32, semantic: bool) -> Arc<Self> {
        let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(name, dimension);
        if semantic {
            identity.space.kind = EmbeddingSpaceKindV1::Semantic;
            identity.space.hash_control = None;
            identity.space.artifact_manifest_fingerprint = "a".repeat(64);
            identity.space.artifacts = vec![EmbeddingArtifactIdentityV1 {
                role: "weights".to_owned(),
                sha256: "b".repeat(64),
                size: 1,
            }];
            identity.producer.space_fingerprint = identity.space.fingerprint();
        }
        identity.validate().unwrap();
        Arc::new(Self {
            identity,
            documents: AtomicUsize::new(0),
            queries: AtomicUsize::new(0),
            raw_calls: AtomicUsize::new(0),
            drops: AtomicUsize::new(0),
            fault: AtomicUsize::new(0),
        })
    }

    fn output(&self, text: &str) -> IdentityBoundEmbedding {
        let mut values = vec![0.0; self.dimension()];
        values[usize::from(text.contains("vertical"))] = 1.0;
        IdentityBoundEmbedding {
            identity: self.identity.clone(),
            values,
        }
    }
}

struct DropCount<'a>(&'a AtomicUsize);

impl Drop for DropCount<'_> {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

impl Embedder for Provider {
    fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            self.raw_calls.fetch_add(1, Ordering::SeqCst);
            Err(invalid(
                "migration.test.raw",
                "forbidden",
                "raw inference is not bound",
            ))
        })
    }

    fn embed_bound<'a>(
        &'a self,
        _cx: &'a Cx,
        text: &'a str,
    ) -> SearchFuture<'a, IdentityBoundEmbedding> {
        Box::pin(async move {
            self.queries.fetch_add(1, Ordering::SeqCst);
            Ok(self.output(text))
        })
    }

    fn embed_batch_bound<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<IdentityBoundEmbedding>> {
        Box::pin(async move {
            self.documents.fetch_add(texts.len(), Ordering::SeqCst);
            let _guard = DropCount(&self.drops);
            match self.fault.load(Ordering::SeqCst) {
                FAIL => {
                    return Err(SearchError::EmbeddingFailed {
                        model: "migration-required-model".to_owned(),
                        source: "deliberate inference failure".into(),
                    });
                }
                PENDING => std::future::pending::<()>().await,
                CANCEL => cx.set_cancel_requested(true),
                _ => {}
            }
            let mut output: Vec<_> = texts.iter().map(|text| self.output(text)).collect();
            if self.fault.load(Ordering::SeqCst) == FOREIGN {
                output.last_mut().unwrap().identity.producer.backend =
                    "private-foreign-model".to_owned();
            }
            Ok(output)
        })
    }

    fn bound_batch_is_native(&self) -> bool {
        true
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
    }

    fn dimension(&self) -> usize {
        usize::try_from(self.identity.space.dimension).unwrap()
    }

    fn id(&self) -> &str {
        &self.identity.space.logical_model_id
    }

    fn model_name(&self) -> &str {
        self.id()
    }

    fn is_semantic(&self) -> bool {
        self.identity.space.kind == EmbeddingSpaceKindV1::Semantic
    }

    fn category(&self) -> ModelCategory {
        if self.is_semantic() {
            ModelCategory::TransformerEmbedder
        } else {
            ModelCategory::HashEmbedder
        }
    }
}

fn generation(sequence: u64) -> ArtifactGenerationIdentityV1 {
    ArtifactGenerationIdentityV1::new(sequence, [0x59; 16]).unwrap()
}

fn documents() -> Vec<IndexableDocument> {
    vec![
        IndexableDocument::new("a", "migrationmarker horizontal café")
            .with_title("retained title")
            .with_metadata("project", "private-project"),
        IndexableDocument::new("b", "vertical"),
        IndexableDocument::new("c", "horizontal"),
    ]
}

async fn build(
    cx: &Cx,
    path: &Path,
    sequence: u64,
    fast: &Arc<Provider>,
    quality: Option<&Arc<Provider>>,
    documents: Vec<IndexableDocument>,
) -> NativeBuiltHybridIndex {
    let mut builder = NativeIndexBuilder::new(path, generation(sequence), fast.clone())
        .unwrap()
        .add_documents(documents);
    if let Some(quality) = quality {
        builder = builder.with_quality_embedder(quality.clone()).unwrap();
    }
    builder.build_hybrid(cx).await.unwrap()
}

fn assert_sources(left: &NativeBuiltHybridIndex, right: &NativeBuiltHybridIndex) {
    assert_eq!(
        left.vectors().documents().len(),
        right.vectors().documents().len()
    );
    for (left, right) in left
        .vectors()
        .documents()
        .iter()
        .zip(right.vectors().documents())
    {
        assert_eq!(left.id, right.id);
        assert_eq!(left.content, right.content);
        assert_eq!(left.title, right.title);
        assert_eq!(left.metadata, right.metadata);
    }
}

fn assert_field(error: SearchError, expected: &str) {
    assert!(matches!(error, SearchError::InvalidConfig { field, .. } if field == expected));
}

#[test]
fn migration_rebuilds_models_while_old_progressive_phases_keep_their_owners() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let fast = Provider::new("old-fast", 2, true);
        let quality = Provider::new("old-quality", 3, true);
        let initial = build(
            &cx,
            &root.path().join("old"),
            1,
            &fast,
            Some(&quality),
            documents(),
        )
        .await;
        let live = NativeLiveHybridIndex::new(&cx, initial).unwrap();
        let old = live.snapshot(&cx).await.unwrap();
        let old_bytes = std::fs::read(old.index().vectors().fast().vector_path()).unwrap();
        let mut query = old.index().progressive(&cx, "horizontal", 3).unwrap();
        assert!(matches!(
            query.next_phase().await.unwrap(),
            Some(NativeSearchPhase::Initial { .. })
        ));
        let new_fast = Provider::new("new-fast", 4, true);
        let new_quality = Provider::new("new-quality", 5, true);
        let candidate = old
            .begin_model_migration(
                &cx,
                root.path().join("new"),
                generation(2),
                new_fast.clone(),
                Some(new_quality.clone()),
            )
            .unwrap()
            .with_batch_size(2)
            .unwrap()
            .with_fast_storage(
                NativeBuildPrecision::F16,
                NativeBuildRetrieval::Hnsw {
                    params: HnswParams::default(),
                    seed: 31,
                },
            )
            .with_quality_storage(NativeBuildPrecision::F32, NativeBuildRetrieval::Exact)
            .unwrap()
            .build(&cx)
            .await
            .unwrap();
        assert_eq!(new_fast.documents.load(Ordering::SeqCst), 3);
        assert_eq!(new_quality.documents.load(Ordering::SeqCst), 3);
        assert_eq!(fast.documents.load(Ordering::SeqCst), 3);
        assert_sources(old.index(), candidate.index());
        let installed = live.install(&cx, &candidate).await.unwrap();
        assert_eq!(installed.generation(), generation(2));
        assert_eq!(installed.index().vectors().fast().embedder().dimension(), 4);
        assert_eq!(
            installed
                .index()
                .vectors()
                .quality()
                .unwrap()
                .embedder()
                .dimension(),
            5
        );
        assert!(installed.index().vectors().fast().graph_path().is_some());
        assert!(matches!(
            query.next_phase().await.unwrap(),
            Some(NativeSearchPhase::Refined { .. })
        ));
        assert_eq!(quality.queries.load(Ordering::SeqCst), 1);
        assert_eq!(new_quality.queries.load(Ordering::SeqCst), 0);
        assert_eq!(
            live.search_refined(&cx, "horizontal", 3)
                .await
                .unwrap()
                .snapshot
                .generation(),
            generation(2)
        );
        assert_eq!(new_quality.queries.load(Ordering::SeqCst), 1);
        assert_eq!(
            std::fs::read(old.index().vectors().fast().vector_path()).unwrap(),
            old_bytes
        );
        assert_eq!(old.generation(), generation(1));
        for provider in [&fast, &quality, &new_fast, &new_quality] {
            assert_eq!(provider.raw_calls.load(Ordering::SeqCst), 0);
        }
    });
}

#[test]
fn explicit_migration_can_add_and_remove_quality_and_change_control_to_semantic() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let control = Provider::new("control", 2, false);
        let initial = build(
            &cx,
            &root.path().join("control"),
            1,
            &control,
            None,
            documents(),
        )
        .await;
        let live = NativeLiveHybridIndex::new(&cx, initial).unwrap();
        let old = live.snapshot(&cx).await.unwrap();
        let fast = Provider::new("semantic-fast", 3, true);
        let quality = Provider::new("semantic-quality", 4, true);
        let candidate = old
            .begin_model_migration(
                &cx,
                root.path().join("semantic"),
                generation(2),
                fast.clone(),
                Some(quality.clone()),
            )
            .unwrap()
            .build(&cx)
            .await
            .unwrap();
        let full = live.install(&cx, &candidate).await.unwrap();
        assert!(full.index().vectors().fast().embedder().is_semantic());
        assert!(full.index().vectors().quality().is_some());
        assert!(old.index().vectors().quality().is_none());
        assert!(!old.index().vectors().fast().embedder().is_semantic());
        let candidate = full
            .begin_model_migration(
                &cx,
                root.path().join("fast-only"),
                generation(3),
                fast.clone(),
                None,
            )
            .unwrap()
            .build(&cx)
            .await
            .unwrap();
        let fast_only = live.install(&cx, &candidate).await.unwrap();
        assert!(fast_only.index().vectors().quality().is_none());
        assert!(full.index().vectors().quality().is_some());
        assert_sources(old.index(), fast_only.index());
        assert_eq!(fast.documents.load(Ordering::SeqCst), 6);
        assert_eq!(quality.documents.load(Ordering::SeqCst), 3);
    });
}

#[test]
fn migration_reembeds_unchanged_models_and_following_updates_use_the_new_contract() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let fast = Provider::new("same-model", 2, true);
        let initial = build(&cx, &root.path().join("old"), 1, &fast, None, documents()).await;
        let live = NativeLiveHybridIndex::new(&cx, initial).unwrap();
        let old = live.snapshot(&cx).await.unwrap();
        let candidate = old
            .begin_model_migration(
                &cx,
                root.path().join("rebuilt"),
                generation(2),
                fast.clone(),
                None,
            )
            .unwrap()
            .build(&cx)
            .await
            .unwrap();
        assert_eq!(
            fast.documents.load(Ordering::SeqCst),
            6,
            "migration must never borrow old vectors"
        );
        let installed = live.install(&cx, &candidate).await.unwrap();
        let update = installed
            .begin_update(&cx, root.path().join("edited"), generation(3))
            .unwrap()
            .upsert_document(IndexableDocument::new("a", "vertical changed"))
            .delete_document("b")
            .build(&cx)
            .await
            .unwrap();
        let edited = live.install(&cx, &update).await.unwrap();
        assert_eq!(
            fast.documents.load(Ordering::SeqCst),
            7,
            "ordinary updates still reuse unchanged rows"
        );
        assert_eq!(edited.index().vectors().documents().len(), 2);
        assert_eq!(
            edited.index().vectors().document("a").unwrap().content,
            "vertical changed"
        );
        assert!(installed.index().vectors().document("b").is_some());
        assert_eq!(
            installed.index().vectors().document("a").unwrap().content,
            documents()[0].content
        );
    });
}

#[test]
fn migration_and_source_updates_have_one_expected_predecessor_winner() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let fast = Provider::new("old", 2, true);
        let initial = build(&cx, &root.path().join("old"), 1, &fast, None, documents()).await;
        let live = NativeLiveHybridIndex::new(&cx, initial).unwrap();
        let old = live.snapshot(&cx).await.unwrap();
        let model = Provider::new("new", 4, true);
        let migration = old
            .begin_model_migration(
                &cx,
                root.path().join("migration"),
                generation(9),
                model.clone(),
                None,
            )
            .unwrap()
            .build(&cx)
            .await
            .unwrap();
        let update = old
            .begin_update(&cx, root.path().join("update"), generation(2))
            .unwrap()
            .upsert_document(IndexableDocument::new("a", "accepted source update"))
            .build(&cx)
            .await
            .unwrap();
        let current = live.install(&cx, &update).await.unwrap();
        assert_field(
            live.install(&cx, &migration).await.unwrap_err(),
            "native_ann.live.expected_current",
        );
        assert_eq!(
            live.snapshot(&cx)
                .await
                .unwrap()
                .index()
                .vectors()
                .document("a")
                .unwrap()
                .content,
            "accepted source update"
        );
        let stale_update = current
            .begin_update(&cx, root.path().join("stale-update"), generation(3))
            .unwrap()
            .delete_document("a")
            .build(&cx)
            .await
            .unwrap();
        let migration = current
            .begin_model_migration(
                &cx,
                root.path().join("winner"),
                generation(4),
                model,
                None,
            )
            .unwrap()
            .build(&cx)
            .await
            .unwrap();
        let selected = live.install(&cx, &migration).await.unwrap();
        assert_sources(current.index(), selected.index());
        assert_field(
            live.install(&cx, &stale_update).await.unwrap_err(),
            "native_ann.live.expected_current",
        );
        assert!(
            live.snapshot(&cx)
                .await
                .unwrap()
                .index()
                .vectors()
                .document("a")
                .is_some()
        );
    });
}

#[test]
fn ordinary_candidate_admission_does_not_acquire_model_migration_authority() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let fast = Provider::new("old", 2, true);
        let initial = build(&cx, &root.path().join("old"), 1, &fast, None, documents()).await;
        let live = NativeLiveHybridIndex::new(&cx, initial).unwrap();
        let old = live.snapshot(&cx).await.unwrap();
        let replacement = Provider::new("new", 2, true);
        let next = build(
            &cx,
            &root.path().join("unrelated"),
            2,
            &replacement,
            None,
            documents(),
        )
        .await;
        assert_field(
            NativeHybridCandidate::new(&cx, old.clone(), next).unwrap_err(),
            "native_ann.live.producer",
        );
        assert!(std::ptr::eq(
            live.snapshot(&cx).await.unwrap().index(),
            old.index()
        ));
    });
}

#[test]
fn changed_prepared_input_and_non_newer_sequences_fail_before_writes_or_inference() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let fast = Provider::new("old", 2, true);
        let initial = build(&cx, &root.path().join("old"), 1, &fast, None, documents()).await;
        let live = NativeLiveHybridIndex::new(&cx, initial).unwrap();
        let old = live.snapshot(&cx).await.unwrap();
        let replacement = Provider::new("new", 3, true);
        let stale = root.path().join("stale");
        let result =
            old.begin_model_migration(&cx, &stale, generation(1), replacement.clone(), None);
        assert_field(result.err().unwrap(), "native_ann.live.migration.generation");
        assert!(!stale.exists());
        let mut incompatible = Provider::new("other-input", 3, true);
        Arc::get_mut(&mut incompatible)
            .unwrap()
            .identity
            .input
            .doc_id_semantics = "other-document-id-policy".to_owned();
        let path = root.path().join("incompatible");
        let result =
            old.begin_model_migration(&cx, &path, generation(2), incompatible.clone(), None);
        assert_field(result.err().unwrap(), "native_ann.live.migration.input");
        assert!(!path.exists());
        assert_eq!(replacement.documents.load(Ordering::SeqCst), 0);
        assert_eq!(incompatible.documents.load(Ordering::SeqCst), 0);
        let path = root.path().join("oversized");
        let result = old
            .begin_model_migration(&cx, &path, generation(2), replacement.clone(), None)
            .unwrap()
            .with_max_batch_input_bytes(1)
            .unwrap()
            .build(&cx)
            .await;
        assert!(result.is_err());
        assert!(!path.exists());
        assert_eq!(replacement.documents.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn failed_cancelled_and_dropped_required_quality_never_return_installable_prefixes() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let fast = Provider::new("old", 2, true);
        let initial = build(&cx, &root.path().join("old"), 1, &fast, None, documents()).await;
        let live = NativeLiveHybridIndex::new(&cx, initial).unwrap();
        let old = live.snapshot(&cx).await.unwrap();
        let original = std::fs::read(old.index().vectors().fast().vector_path()).unwrap();
        for fault in [FAIL, FOREIGN, CANCEL, PENDING] {
            let new_fast = Provider::new("fast", 3, true);
            let quality = Provider::new("quality", 4, true);
            quality.fault.store(fault, Ordering::SeqCst);
            let migration = old
                .begin_model_migration(
                    &cx,
                    root.path().join(format!("failure-{fault}")),
                    generation(2),
                    new_fast.clone(),
                    Some(quality.clone()),
                )
                .unwrap()
                .with_batch_size(3)
                .unwrap();
            if fault == PENDING {
                let mut future = Box::pin(migration.build(&cx));
                let mut task = Context::from_waker(Waker::noop());
                assert!(future.as_mut().poll(&mut task).is_pending());
                assert_eq!(new_fast.documents.load(Ordering::SeqCst), 3);
                assert_eq!(quality.drops.load(Ordering::SeqCst), 0);
                drop(future);
                assert_eq!(quality.drops.load(Ordering::SeqCst), 1);
            } else {
                let error = migration.build(&cx).await.err().unwrap();
                if fault == CANCEL {
                    assert!(matches!(error, SearchError::Cancelled { .. }));
                    cx.set_cancel_requested(false);
                } else if fault == FAIL {
                    assert!(matches!(error, SearchError::EmbeddingFailed { .. }));
                } else {
                    assert!(!error.to_string().contains("private-foreign-model"));
                }
            }
            assert!(std::ptr::eq(
                live.snapshot(&cx).await.unwrap().index(),
                old.index()
            ));
            assert_eq!(
                std::fs::read(old.index().vectors().fast().vector_path()).unwrap(),
                original
            );
            assert_eq!(
                old.index().search(&cx, "horizontal", 3).await.unwrap().len(),
                3
            );
        }
    });
}

#[test]
fn empty_migration_preserves_new_identity_and_topology_without_inference() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let fast = Provider::new("old", 2, false);
        let initial = build(&cx, &root.path().join("old"), 1, &fast, None, Vec::new()).await;
        let live = NativeLiveHybridIndex::new(&cx, initial).unwrap();
        let old = live.snapshot(&cx).await.unwrap();
        let fast = Provider::new("new-fast", 3, true);
        let quality = Provider::new("new-quality", 4, true);
        let candidate = old
            .begin_model_migration(
                &cx,
                root.path().join("new"),
                generation(2),
                fast.clone(),
                Some(quality.clone()),
            )
            .unwrap()
            .build(&cx)
            .await
            .unwrap();
        let selected = live.install(&cx, &candidate).await.unwrap();
        assert_eq!(selected.generation(), generation(2));
        assert!(selected.index().vectors().documents().is_empty());
        assert_eq!(selected.index().vectors().fast().embedder().dimension(), 3);
        assert_eq!(
            selected
                .index()
                .vectors()
                .quality()
                .unwrap()
                .embedder()
                .dimension(),
            4
        );
        assert_eq!(fast.documents.load(Ordering::SeqCst), 0);
        assert_eq!(quality.documents.load(Ordering::SeqCst), 0);
    });
}