use super::*;
use crate::native_ann::builder::{NativeBuildPrecision, NativeBuildRetrieval, NativeIndexBuilder};
use crate::{Embedder, ModelCategory, SearchError, SearchFuture};
use asupersync::test_utils::run_test_with_cx;
use frankensearch_core::generation::EmbeddingIdentityBundleV1;
use frankensearch_index::native_hnsw::HnswParams;
use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Provider {
    identity: EmbeddingIdentityBundleV1,
    calls: AtomicUsize,
}

impl Provider {
    fn new(name: &str, dimension: u32) -> Arc<Self> {
        Arc::new(Self {
            identity: EmbeddingIdentityBundleV1::explicit_test_model(name, dimension),
            calls: AtomicUsize::new(0),
        })
    }
}

impl Embedder for Provider {
    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
    }
    fn id(&self) -> &str {
        &self.identity.space.logical_model_id
    }
    fn model_name(&self) -> &str {
        self.id()
    }
    fn dimension(&self) -> usize {
        usize::try_from(self.identity.space.dimension).unwrap()
    }
    fn is_semantic(&self) -> bool {
        false
    }
    fn category(&self) -> ModelCategory {
        ModelCategory::HashEmbedder
    }
    fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let mut values = vec![0.0; self.dimension()];
            values[0] = 1.0;
            Ok(values)
        })
    }
}

fn documents() -> Vec<IndexableDocument> {
    vec![
        IndexableDocument::new("a", "original source café 🦀")
            .with_title("東京")
            .with_metadata("tenant", "blue"),
        IndexableDocument::new("b", ""),
        IndexableDocument::new("c", "!!!"),
        IndexableDocument::new("d", "last partition body"),
    ]
}

fn builder(path: &Path, graphs: bool, docs: Vec<IndexableDocument>) -> NativeIndexBuilder {
    let retrieval = if graphs {
        NativeBuildRetrieval::Hnsw {
            params: HnswParams::default(),
            seed: 19,
        }
    } else {
        NativeBuildRetrieval::Exact
    };
    NativeIndexBuilder::new(
        path,
        ArtifactGenerationIdentityV1::new(7, [7; 16]).unwrap(),
        Provider::new("source-recovery-fast", 2),
    )
    .unwrap()
    .with_fast_storage(NativeBuildPrecision::F16, retrieval)
    .with_quality_embedder(Provider::new("source-recovery-quality", 3))
    .unwrap()
    .with_quality_storage(NativeBuildPrecision::F32, retrieval)
    .unwrap()
    .add_documents(docs)
}

fn manifest(path: &Path) -> Manifest {
    serde_json::from_slice(&fs::read(path.join(SNAPSHOT_FILE)).unwrap()).unwrap()
}

// Only negative fixtures mint a new authority over deliberately inconsistent
// descriptors. Production recovery always requires the caller's original receipt.
fn write_manifest(path: &Path, saved: &Manifest) -> GenerationComponentReceiptV1 {
    let bytes = serde_json::to_vec(saved).unwrap();
    fs::write(path.join(SNAPSHOT_FILE), &bytes).unwrap();
    Artifact::from_bytes(&bytes).receipt()
}

fn recover(
    cx: &Cx,
    path: &Path,
    receipt: &GenerationComponentReceiptV1,
) -> SearchResult<(ArtifactGenerationIdentityV1, Vec<IndexableDocument>)> {
    NativeBuiltShardedIndex::recover_selected_source(
        cx,
        path,
        receipt,
        NativeShardedReopenLimits::default(),
    )
}

fn hide_derived(index: &NativeBuiltShardedIndex, saved: &Path) {
    fs::create_dir(saved).unwrap();
    for (ordinal, part) in index.partitions().iter().enumerate() {
        for (name, tier) in [("fast", Some(part.fast())), ("quality", part.quality())] {
            if let Some(tier) = tier {
                fs::rename(
                    tier.vector_path(),
                    saved.join(format!("{ordinal}-{name}.fsvi")),
                )
                .unwrap();
                if let Some(graph) = tier.graph_path() {
                    fs::rename(graph, saved.join(format!("{ordinal}-{name}.hnsw"))).unwrap();
                    let mut receipt = graph.as_os_str().to_os_string();
                    receipt.push(".receipt");
                    fs::rename(receipt, saved.join(format!("{ordinal}-{name}.receipt"))).unwrap();
                }
            }
        }
    }
}

#[test]
fn recovery_owns_every_source_without_old_models_vectors_or_graphs() {
    run_test_with_cx(|cx| async move {
        for graphs in [false, true] {
            let temp = tempfile::tempdir().unwrap();
            let path = temp.path().join("index");
            let built = builder(&path, graphs, documents())
                .build_sharded(&cx, 2)
                .await
                .unwrap();
            let expected = serde_json::to_value(built.documents().collect::<Vec<_>>()).unwrap();
            let generation = built.generation();
            let receipt = built.seal_for_reopen(&cx).unwrap();
            hide_derived(&built, &temp.path().join("saved"));
            assert!(
                NativeBuiltShardedIndex::open_selected(
                    &cx,
                    &path,
                    &receipt,
                    Provider::new("source-recovery-fast", 2),
                    Some(Provider::new("source-recovery-quality", 3)),
                    NativeShardedReopenLimits::default(),
                )
                .is_err()
            );
            drop(built); // Recovery cannot depend on an admitted in-memory owner.
            let (recovered_generation, recovered) = recover(&cx, &path, &receipt).unwrap();
            assert_eq!(recovered_generation, generation);
            assert_eq!(serde_json::to_value(&recovered).unwrap(), expected);
            fs::rename(&path, temp.path().join("moved-after-recovery")).unwrap();
            assert_eq!(recovered[0].title.as_deref(), Some("東京"));
            assert_eq!(recovered[0].metadata["tenant"], "blue");
            assert!(recovered[1].content.is_empty());
            assert_eq!(recovered[3].content, "last partition body");
        }
    });
}

#[test]
fn every_link_and_the_last_source_must_authenticate_without_prefix_success() {
    run_test_with_cx(|cx| async move {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("index");
        let built = builder(&path, true, documents())
            .build_sharded(&cx, 2)
            .await
            .unwrap();
        let receipt = built.seal_for_reopen(&cx).unwrap();
        for relative in [
            SNAPSHOT_FILE,
            "shard-000000/native.snapshot.json",
            "shard-000001/native.snapshot.json",
            "shard-000001/native.source.jsonl",
        ] {
            let damaged = path.join(relative);
            let original = fs::read(&damaged).unwrap();
            let mut bytes = original.clone();
            bytes[0] ^= 1; // Same byte count, so an existence/length check is insufficient.
            fs::write(&damaged, &bytes).unwrap();
            assert!(recover(&cx, &path, &receipt).is_err(), "{relative}");
            assert_eq!(
                fs::read(&damaged).unwrap(),
                bytes,
                "recovery never repairs input"
            );
            fs::write(&damaged, original).unwrap();
        }
        let missing = path.join("shard-000001/native.source.jsonl");
        fs::rename(&missing, temp.path().join("saved-source")).unwrap();
        assert!(recover(&cx, &path, &receipt).is_err());
        fs::rename(temp.path().join("saved-source"), missing).unwrap();
        assert_eq!(recover(&cx, &path, &receipt).unwrap().1.len(), 4);
        let mut false_receipt = receipt;
        false_receipt.sha256[0] ^= 1;
        assert!(recover(&cx, &path, &false_receipt).is_err());
    });
}

#[test]
fn complete_budget_generation_and_topology_precede_any_source_read() {
    run_test_with_cx(|cx| async move {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("index");
        let built = builder(&path, false, documents())
            .build_sharded(&cx, 2)
            .await
            .unwrap();
        let receipt = built.seal_for_reopen(&cx).unwrap();
        let selected =
            select_sources(&cx, &path, &receipt, NativeShardedReopenLimits::default()).unwrap();
        let saved = manifest(&path);
        let total = receipt.byte_len
            + saved
                .partitions
                .iter()
                .zip(&selected.partitions)
                .map(|(descriptor, source)| descriptor.byte_len + source.byte_len())
                .sum::<u64>();
        let limits = NativeShardedReopenLimits {
            max_artifact_bytes: total,
            ..NativeShardedReopenLimits::default()
        };
        assert!(
            NativeBuiltShardedIndex::recover_selected_source(&cx, &path, &receipt, limits).is_ok()
        );
        let first = path.join("shard-000000/native.source.jsonl");
        fs::rename(&first, temp.path().join("held-source")).unwrap();
        let error = NativeBuiltShardedIndex::recover_selected_source(
            &cx,
            &path,
            &receipt,
            NativeShardedReopenLimits {
                max_artifact_bytes: total - 1,
                ..limits
            },
        )
        .unwrap_err();
        assert!(matches!(error, SearchError::InvalidConfig { field, .. }
            if field == "native_ann.sharded_builder.source_budget"));
        // The unbounded attempt gets as far as the deliberately missing source.
        assert!(matches!(
            recover(&cx, &path, &receipt),
            Err(SearchError::Io(_))
        ));
        for fault in ["generation", "quality", "count", "empty"] {
            let mut changed = manifest(&path);
            match fault {
                "generation" => changed.generation.sequence += 1,
                "quality" => changed.quality = !changed.quality,
                "count" => changed.documents += 1,
                "empty" => changed.partitions.clear(),
                _ => unreachable!(), // ubs:ignore — closed test-fixture cases.
            }
            let altered = write_manifest(&path, &changed);
            assert!(matches!(
                recover(&cx, &path, &altered),
                Err(SearchError::InvalidConfig { .. })
            ));
            write_manifest(&path, &saved);
        }
    });
}

#[test]
fn authenticated_partitions_cannot_overlap_or_arrive_out_of_order() {
    run_test_with_cx(|cx| async move {
        for overlap in [false, true] {
            let temp = tempfile::tempdir().unwrap();
            let path = temp.path().join("index");
            let built = builder(&path, false, documents())
                .build_sharded(&cx, 2)
                .await
                .unwrap();
            built.seal_for_reopen(&cx).unwrap();
            let mut saved = manifest(&path);
            let a = path.join("shard-000000");
            let b = path.join("shard-000001");
            if overlap {
                for name in ["native.snapshot.json", "native.source.jsonl"] {
                    fs::copy(a.join(name), b.join(name)).unwrap();
                }
                saved.partitions[1] = saved.partitions[0];
            } else {
                let held = temp.path().join("held-partition");
                fs::rename(&a, &held).unwrap();
                fs::rename(&b, &a).unwrap();
                fs::rename(&held, &b).unwrap();
                saved.partitions.swap(0, 1);
            }
            let receipt = write_manifest(&path, &saved);
            let error = recover(&cx, &path, &receipt).unwrap_err();
            assert!(matches!(error, SearchError::InvalidConfig { field, .. }
                if field == "native_ann.sharded_builder.source_order"));
        }
    });
}

#[test]
fn preflight_retains_original_source_digests_and_cancellation_remains_typed() {
    run_test_with_cx(|cx| async move {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("index");
        let built = builder(&path, false, documents())
            .build_sharded(&cx, 2)
            .await
            .unwrap();
        let receipt = built.seal_for_reopen(&cx).unwrap();
        let selected =
            select_sources(&cx, &path, &receipt, NativeShardedReopenLimits::default()).unwrap();
        let source = path.join("shard-000001/native.source.jsonl");
        let original = fs::read(&source).unwrap();
        let mut bytes = original.clone();
        bytes[0] ^= 1;
        fs::write(&source, bytes).unwrap();
        assert!(selected.read(&cx).is_err());
        fs::write(&source, original).unwrap();
        let selected =
            select_sources(&cx, &path, &receipt, NativeShardedReopenLimits::default()).unwrap();
        cx.set_cancel_requested(true);
        assert!(matches!(
            selected.read(&cx),
            Err(SearchError::Cancelled { .. })
        ));
        assert!(matches!(
            recover(&cx, &path, &receipt),
            Err(SearchError::Cancelled { .. })
        ));
        cx.set_cancel_requested(false);
        assert_eq!(recover(&cx, &path, &receipt).unwrap().1.len(), 4);
    });
}

#[test]
fn empty_sources_and_source_object_refusals_keep_the_existing_format_contract() {
    run_test_with_cx(|cx| async move {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("empty");
        let built = builder(&path, false, Vec::new())
            .build_sharded(&cx, 1)
            .await
            .unwrap();
        let receipt = built.seal_for_reopen(&cx).unwrap();
        let (_, docs) = NativeBuiltShardedIndex::recover_selected_source(
            &cx,
            &path,
            &receipt,
            NativeShardedReopenLimits {
                max_documents: 0,
                ..NativeShardedReopenLimits::default()
            },
        )
        .unwrap();
        assert!(docs.is_empty());
        let source = path.join("shard-000000/native.source.jsonl");
        let saved = temp.path().join("saved-source");
        fs::rename(&source, &saved).unwrap();
        std::os::unix::fs::symlink(&saved, &source).unwrap();
        assert!(recover(&cx, &path, &receipt).is_err());
        fs::rename(&saved, &source).unwrap();
        let alias = temp.path().join("alias");
        std::os::unix::fs::symlink(&path, &alias).unwrap();
        assert!(recover(&cx, &alias, &receipt).is_err());
        assert!(recover(&cx, &path, &receipt).unwrap().1.is_empty());
    });
}

#[cfg(feature = "quill")]
#[test]
fn hybrid_recovery_authenticates_the_outer_selection_without_any_lexical_files() {
    use crate::native_ann::builder::sharded::{
        NativeBuiltShardedHybridIndex, NativeShardedHybridReopenLimits,
    };
    run_test_with_cx(|cx| async move {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("hybrid");
        let built = builder(&path, true, documents())
            .build_sharded_hybrid(&cx, 2)
            .await
            .unwrap();
        let receipt = built.seal_for_reopen(&cx).unwrap();
        hide_derived(built.vectors(), &temp.path().join("saved-derived"));
        drop(built);
        fs::rename(path.join("lexical"), temp.path().join("saved-lexical")).unwrap();
        let recover = |receipt: &GenerationComponentReceiptV1| {
            NativeBuiltShardedHybridIndex::recover_selected_source(
                &cx,
                &path,
                receipt,
                NativeShardedHybridReopenLimits::default(),
            )
        };
        let (generation, docs) = recover(&receipt).unwrap();
        assert_eq!(generation.sequence, 7);
        assert_eq!(
            serde_json::to_value(docs).unwrap(),
            serde_json::to_value(documents()).unwrap()
        );
        assert!(!path.join("lexical").exists());
        let outer = path.join("native.sharded-hybrid.json");
        let original = fs::read(&outer).unwrap();
        fs::write(&outer, b"damaged outer descriptor").unwrap();
        assert!(recover(&receipt).is_err());
        fs::write(&outer, &original).unwrap();
        // A valid inner receipt cannot be used as complete hybrid authority.
        let value: serde_json::Value = serde_json::from_slice(&original).unwrap();
        let inner: Artifact = serde_json::from_value(value["vectors"].clone()).unwrap();
        assert!(recover(&inner.receipt()).is_err());
        let first = path.join("shard-000000/native.source.jsonl");
        fs::rename(&first, temp.path().join("held-source")).unwrap();
        let mut changed = value;
        changed["generation"]["sequence"] = serde_json::json!(8);
        let bytes = serde_json::to_vec(&changed).unwrap();
        fs::write(&outer, &bytes).unwrap();
        let error = recover(&Artifact::from_bytes(&bytes).receipt()).unwrap_err();
        assert!(matches!(error, SearchError::InvalidConfig { field, .. }
            if field == "native_ann.sharded_hybrid.source"));
    });
}
