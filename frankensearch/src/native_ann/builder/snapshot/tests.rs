use super::*;
use std::sync::atomic::Ordering;

use crate::native_ann::builder::{NativeBuildRetrieval, NativeIndexBuilder};
use crate::native_ann::builder::tests::{Provider, Reply, documents, generation};
use frankensearch_index::native_hnsw::HnswParams;

fn provider(name: &str, dimension: u32) -> Arc<Provider> {
    Arc::new(Provider::new(name, dimension, Reply::Correct))
}

async fn build(cx: &Cx, path: &Path, ann: bool, quality: bool) -> NativeBuiltIndex {
    let mut builder = NativeIndexBuilder::new(path, generation(), provider("fast", 2)).unwrap()
        .add_documents(documents());
    if ann {
        builder = builder.with_fast_storage(NativeBuildPrecision::F16,
            NativeBuildRetrieval::Hnsw { params: HnswParams::default(), seed: 7 });
    }
    if quality {
        builder = builder.with_quality_embedder(provider("quality", 3)).unwrap();
    }
    builder.build(cx).await.unwrap()
}

fn open(cx: &Cx, path: &Path, receipt: &GenerationComponentReceiptV1, quality: bool) -> SearchResult<NativeBuiltIndex> {
    let quality = quality.then(|| -> Arc<dyn Embedder> { provider("quality", 3) });
    NativeBuiltIndex::open_selected(cx, path, receipt, provider("fast", 2), quality)
}

fn change_manifest(path: &Path, change: impl FnOnce(&mut serde_json::Value)) -> GenerationComponentReceiptV1 {
    let path = path.join(SNAPSHOT_FILE);
    let mut value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    change(&mut value);
    let bytes = serde_json::to_vec(&value).unwrap();
    std::fs::write(path, &bytes).unwrap();
    Artifact::from_bytes(&bytes).receipt()
}

#[test]
fn sealed_exact_and_ann_builds_reopen_all_tiers_without_reembedding() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for ann in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("built");
            let built = build(&cx, &path, ann, true).await;
            let fast_hits = built.fast().search(&cx, "vertical", 5).await.unwrap();
            let quality_hits = built.quality().unwrap().search(&cx, "vertical", 5).await.unwrap();
            let original_documents = serde_json::to_value(built.documents()).unwrap();
            let fast_witness = built.fast().index().owner_witness().clone();
            let quality_witness = built.quality().unwrap().index().owner_witness().clone();
            let receipt = built.seal_for_reopen(&cx).unwrap();
            drop(built);
            let fast = provider("fast", 2);
            let quality = provider("quality", 3);
            let reopened = NativeBuiltIndex::open_selected(&cx, &path, &receipt, fast.clone(), Some(quality.clone())).unwrap();
            assert_eq!(fast.queries.load(Ordering::SeqCst), 0);
            assert_eq!(quality.queries.load(Ordering::SeqCst), 0);
            assert_eq!(reopened.fast().index().owner_witness(), &fast_witness);
            assert_eq!(reopened.quality().unwrap().index().owner_witness(), &quality_witness);
            assert_eq!(serde_json::to_value(reopened.documents()).unwrap(), original_documents);
            assert_eq!(reopened.fast().search(&cx, "vertical", 5).await.unwrap(), fast_hits);
            assert_eq!(reopened.quality().unwrap().search(&cx, "vertical", 5).await.unwrap(), quality_hits);
            assert_eq!(reopened.fast().graph_path().is_some(), ann);
        }
    });
}

#[test]
fn reopened_sources_retain_titles_metadata_unicode_and_empty_text() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("built");
        let mut doc = IndexableDocument::new("文書", "line one\nline two\t\"quoted\" 🚀").with_title("title\n第二行");
        // Round-trip every real source field, including optional metadata.
        let mut encoded = serde_json::to_value(&doc).unwrap();
        encoded["metadata"] = serde_json::json!({"generation": 9, "nested": ["one", {"two": true}]});
        doc = serde_json::from_value(encoded).unwrap();
        let built = NativeIndexBuilder::new(&path, generation(), provider("fast", 2)).unwrap()
            .add_document(doc.clone()).add_document(IndexableDocument::new("empty", ""))
            .build(&cx).await.unwrap();
        let receipt = built.seal_for_reopen(&cx).unwrap();
        drop(built);
        let reopened = open(&cx, &path, &receipt, false).unwrap();
        assert_eq!(serde_json::to_value(reopened.document("文書").unwrap()).unwrap(), serde_json::to_value(&doc).unwrap());
        assert_eq!(reopened.document("empty").unwrap().content, "");
    });
}

#[test]
fn selected_reopen_preserves_hash_ordered_rows_and_resolves_original_source_ids() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("built");
        let built = build(&cx, &path, false, true).await;
        let receipt = built.seal_for_reopen(&cx).unwrap();
        drop(built);
        let reopened = open(&cx, &path, &receipt, true).unwrap();
        let source_ids: Vec<_> = reopened.documents().iter().map(|doc| doc.id.as_str()).collect();
        for tier in [reopened.fast(), reopened.quality().unwrap()] {
            let owner = &tier.index.owner;
            let physical_ids: Vec<_> = (0..owner.record_count())
                .map(|position| owner.row(position).unwrap().doc_id().to_owned()).collect();
            assert_ne!(physical_ids, source_ids, "fixture must expose physical hash ordering");
            for (physical, id) in physical_ids.iter().enumerate() {
                let document = reopened.document(id).unwrap();
                let mut expected = vec![0.0; tier.embedder().dimension()];
                expected[usize::from(document.content == "vertical")] = 1.0;
                assert_eq!(owner.vector_at_f32(physical).unwrap(), expected);
            }
            assert_eq!(tier.search(&cx, "vertical", 1).await.unwrap()[0].doc_id, "b");
        }
    });
}

#[test]
fn receipt_reopens_after_directory_relocation_and_reader_owners_are_independent() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("built");
        let built = build(&cx, &path, true, true).await;
        let receipt = built.seal_for_reopen(&cx).unwrap();
        drop(built);
        let moved = dir.path().join("relocated");
        std::fs::rename(&path, &moved).unwrap();
        let one = open(&cx, &moved, &receipt, true).unwrap();
        let two = open(&cx, &moved, &receipt, true).unwrap();
        assert_eq!(one.fast().index().owner_witness(), two.fast().index().owner_witness());
        // Mutating paths AFTER admission cannot change either retained cohort.
        std::fs::write(moved.join("fast.fsvi"), b"changed after admission").unwrap();
        std::fs::write(moved.join(SOURCE_FILE), b"changed source").unwrap();
        drop(one);
        assert_eq!(two.fast().search(&cx, "vertical", 1).await.unwrap()[0].doc_id, "b");
        assert_eq!(two.document("b").unwrap().content, "vertical");
        assert!(open(&cx, &moved, &receipt, true).is_err());
    });
}

#[test]
fn empty_selected_cohort_reopens_with_a_real_nonempty_source_receipt() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty");
        let built = NativeIndexBuilder::new(&path, generation(), provider("fast", 2)).unwrap()
            .build(&cx).await.unwrap();
        let receipt = built.seal_for_reopen(&cx).unwrap();
        assert_eq!(std::fs::read(path.join(SOURCE_FILE)).unwrap(), SOURCE_HEADER);
        let reopened = NativeBuiltIndex::open_selected_with_limits(&cx, &path, &receipt,
            provider("fast", 2), None, NativeReopenLimits { max_documents: 0, ..NativeReopenLimits::default() }).unwrap();
        assert!(reopened.documents().is_empty());
        assert!(reopened.fast().search(&cx, "query", 1).await.unwrap().is_empty());
    });
}

#[test]
fn no_descriptor_means_no_implicit_rebuild_or_directory_discovery() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("built");
        let _built = build(&cx, &path, false, false).await;
        let receipt = GenerationComponentReceiptV1 { byte_len: 100, sha256: [1; 32] };
        assert!(open(&cx, &path, &receipt, false).is_err());
        assert!(!path.join(SOURCE_FILE).exists());
        assert!(!path.join(SNAPSHOT_FILE).exists());
    });
}

#[test]
fn required_quality_cannot_be_omitted_and_extra_quality_cannot_be_invented() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for quality in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("built");
            let built = build(&cx, &path, false, quality).await;
            let receipt = built.seal_for_reopen(&cx).unwrap();
            assert!(open(&cx, &path, &receipt, !quality).is_err());
            if quality {
                std::fs::rename(path.join("quality.fsvi"), path.join("quality.saved")).unwrap();
                assert!(open(&cx, &path, &receipt, true).is_err());
            }
        }
    });
}

#[test]
fn substituted_producer_fails_before_reading_source_or_running_inference() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("built");
        let built = build(&cx, &path, false, false).await;
        let receipt = built.seal_for_reopen(&cx).unwrap();
        std::fs::rename(path.join(SOURCE_FILE), path.join("source.saved")).unwrap();
        let foreign = provider("not-the-producing-model", 2);
        let error = NativeBuiltIndex::open_selected(&cx, &path, &receipt, foreign.clone(), None).err().unwrap();
        assert!(matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "native_ann.builder.producer"));
        assert_eq!(foreign.queries.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn equal_dimension_foreign_quality_does_not_drop_to_fast_only() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("built");
        let built = build(&cx, &path, false, true).await;
        let receipt = built.seal_for_reopen(&cx).unwrap();
        assert!(NativeBuiltIndex::open_selected(&cx, &path, &receipt,
            provider("fast", 2), Some(provider("foreign-quality", 3))).is_err());
    });
}

#[test]
fn original_external_receipt_rejects_an_otherwise_valid_descriptor_substitution() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("built");
        let built = build(&cx, &path, false, false).await;
        let receipt = built.seal_for_reopen(&cx).unwrap();
        let _forged_pin = change_manifest(&path, |value| value["generation"]["sequence"] = 10.into());
        assert!(open(&cx, &path, &receipt, false).is_err());
    });
}

#[test]
fn valid_vector_from_same_model_and_generation_cannot_replace_selected_bytes() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("built");
        let first = build(&cx, &path, false, false).await;
        let receipt = first.seal_for_reopen(&cx).unwrap();
        let changed: Vec<_> = documents().into_iter().map(|mut doc| { doc.content = "vertical".to_owned(); doc }).collect();
        let second = NativeIndexBuilder::new(dir.path().join("other"), generation(), provider("fast", 2)).unwrap()
            .add_documents(changed).build(&cx).await.unwrap();
        assert_eq!(first.fast().index().owner_witness().generation, second.fast().index().owner_witness().generation);
        std::fs::copy(second.fast().vector_path(), first.fast().vector_path()).unwrap();
        assert!(open(&cx, &path, &receipt, false).is_err());
        assert_eq!(first.document("a").unwrap().content, "horizontal");
    });
}

#[test]
fn same_length_source_content_mutation_cannot_pass_a_membership_only_check() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("built");
        let built = build(&cx, &path, false, false).await;
        let receipt = built.seal_for_reopen(&cx).unwrap();
        let source = std::fs::read_to_string(path.join(SOURCE_FILE)).unwrap();
        let changed = source.replace("horizontal", "horizontam");
        assert_ne!(source, changed);
        assert_eq!(source.len(), changed.len());
        std::fs::write(path.join(SOURCE_FILE), changed).unwrap();
        assert!(open(&cx, &path, &receipt, false).is_err());
    });
}

#[test]
fn another_valid_native_graph_over_the_same_owner_is_not_the_selected_graph() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("built");
        let built = build(&cx, &path, true, false).await;
        let receipt = built.seal_for_reopen(&cx).unwrap();
        let params = HnswParams { ef_search: 123, ..HnswParams::default() };
        let replacement = NativeAnnIndex::build(&cx, Arc::clone(&built.fast.index.owner), params, 7).unwrap();
        replacement.save(&cx, built.fast().graph_path().unwrap()).unwrap();
        assert_eq!(std::fs::metadata(built.fast().graph_path().unwrap()).unwrap().len(),
            built.fast.graph_receipt.as_ref().unwrap().graph_byte_len);
        // Ordinary owner-only load accepts it. Selected reopen must not.
        assert!(NativeAnnIndex::load(&cx, Arc::clone(&built.fast.index.owner), built.fast().graph_path().unwrap()).is_ok());
        assert!(open(&cx, &path, &receipt, false).is_err());
    });
}

#[test]
fn missing_or_corrupt_native_sidecars_are_errors_not_exact_fallbacks() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for artifact in ["fast.fshnsw", "fast.fshnsw.receipt"] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("built");
            let built = build(&cx, &path, true, false).await;
            let receipt = built.seal_for_reopen(&cx).unwrap();
            let artifact = path.join(artifact);
            let backup = artifact.with_extension("saved");
            std::fs::rename(&artifact, backup).unwrap();
            assert!(open(&cx, &path, &receipt, false).is_err());
            std::fs::write(&artifact, b"corrupt").unwrap();
            assert!(open(&cx, &path, &receipt, false).is_err());
            assert_eq!(std::fs::read(&artifact).unwrap(), b"corrupt");
        }
    });
}

#[test]
fn selected_exact_mode_ignores_unselected_sidecars_instead_of_autodetecting_them() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("built");
        let built = build(&cx, &path, false, false).await;
        let receipt = built.seal_for_reopen(&cx).unwrap();
        std::fs::write(path.join("fast.fshnsw"), b"unselected sidecar").unwrap();
        let reopened = open(&cx, &path, &receipt, false).unwrap();
        assert!(reopened.fast().graph_path().is_none());
        assert!(matches!(reopened.fast().index().retrieval_mode(), crate::native_ann::NativeRetrievalMode::Exact { .. }));
    });
}

#[test]
fn resource_limits_refuse_before_returning_any_selected_handle() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("built");
        let built = build(&cx, &path, true, false).await;
        let receipt = built.seal_for_reopen(&cx).unwrap();
        for limits in [
            NativeReopenLimits { max_source_bytes: SOURCE_HEADER.len() as u64, ..NativeReopenLimits::default() },
            NativeReopenLimits { max_document_bytes: 1, ..NativeReopenLimits::default() },
            NativeReopenLimits { max_documents: 4, ..NativeReopenLimits::default() },
            NativeReopenLimits { max_vector_bytes: 1, ..NativeReopenLimits::default() },
            NativeReopenLimits { max_graph_bytes: 1, ..NativeReopenLimits::default() },
            NativeReopenLimits { max_document_bytes: u64::MAX, ..NativeReopenLimits::default() },
        ] {
            assert!(NativeBuiltIndex::open_selected_with_limits(&cx, &path, &receipt, provider("fast", 2), None, limits).is_err());
        }
        assert!(open(&cx, &path, &receipt, false).is_ok());
    });
}

#[test]
fn malformed_source_order_and_false_membership_fail_even_under_a_new_trusted_pin() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for mutation in ["reversed", "duplicate", "other-id", "truncated-line", "extra-doc"] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("built");
            let built = build(&cx, &path, false, false).await;
            built.seal_for_reopen(&cx).unwrap();
            let original = std::fs::read_to_string(path.join(SOURCE_FILE)).unwrap();
            let mut lines: Vec<String> = original.lines().map(str::to_owned).collect();
            match mutation {
                "reversed" => lines[1..].reverse(),
                "duplicate" => lines[2] = lines[1].clone(),
                "other-id" => {
                    let mut doc: IndexableDocument = serde_json::from_str(&lines[1]).unwrap();
                    doc.id = "0".to_owned();
                    lines[1] = serde_json::to_string(&doc).unwrap();
                }
                "extra-doc" => { let extra = lines.last().unwrap().clone(); lines.push(extra); }
                _ => {}
            }
            let mut source = lines.join("\n");
            if mutation != "truncated-line" { source.push('\n'); }
            let artifact = Artifact::from_bytes(source.as_bytes());
            std::fs::write(path.join(SOURCE_FILE), source).unwrap();
            let receipt = change_manifest(&path, |value| value["source"] = serde_json::to_value(artifact).unwrap());
            assert!(open(&cx, &path, &receipt, false).is_err(), "{mutation}");
        }
    });
}

#[test]
fn unknown_descriptor_schema_and_fields_are_not_forward_guessed() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for unknown_field in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("built");
            let built = build(&cx, &path, false, false).await;
            built.seal_for_reopen(&cx).unwrap();
            let receipt = change_manifest(&path, |value| {
                if unknown_field { value["fallback_to_latest"] = true.into(); }
                else { value["schema"] = "future-version".into(); }
            });
            assert!(open(&cx, &path, &receipt, false).is_err());
        }
    });
}

#[test]
fn existing_complete_or_partial_seals_are_never_overwritten() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("built");
        let built = build(&cx, &path, false, false).await;
        let receipt = built.seal_for_reopen(&cx).unwrap();
        let source = std::fs::read(path.join(SOURCE_FILE)).unwrap();
        let descriptor = std::fs::read(path.join(SNAPSHOT_FILE)).unwrap();
        assert!(built.seal_for_reopen(&cx).is_err());
        assert_eq!(std::fs::read(path.join(SOURCE_FILE)).unwrap(), source);
        assert_eq!(std::fs::read(path.join(SNAPSHOT_FILE)).unwrap(), descriptor);
        assert!(open(&cx, &path, &receipt, false).is_ok());
        let partial = build(&cx, &dir.path().join("partial"), false, false).await;
        std::fs::write(partial.directory().join(SOURCE_FILE), b"partial source").unwrap();
        assert!(partial.seal_for_reopen(&cx).is_err());
        assert_eq!(std::fs::read(partial.directory().join(SOURCE_FILE)).unwrap(), b"partial source");
        assert!(!partial.directory().join(SNAPSHOT_FILE).exists());
    });
}

#[test]
fn failed_selected_open_does_not_modify_any_persistent_artifact() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("built");
        let built = build(&cx, &path, true, true).await;
        let mut receipt = built.seal_for_reopen(&cx).unwrap();
        receipt.sha256[0] ^= 1;
        let files: Vec<_> = std::fs::read_dir(&path).unwrap().map(|entry| entry.unwrap().path()).collect();
        let before: Vec<_> = files.iter().map(|path| std::fs::read(path).unwrap()).collect();
        assert!(open(&cx, &path, &receipt, true).is_err());
        let after: Vec<_> = files.iter().map(|path| std::fs::read(path).unwrap()).collect();
        assert_eq!(before, after);
    });
}

#[test]
fn cancelled_seal_and_open_neither_write_nor_return_a_cohort() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("built");
        let built = build(&cx, &path, false, false).await;
        cx.cancel_with(asupersync::CancelKind::User, Some("cancel snapshot"));
        assert!(matches!(built.seal_for_reopen(&cx), Err(SearchError::Cancelled { .. })));
        assert!(!path.join(SOURCE_FILE).exists());
        assert!(!path.join(SNAPSHOT_FILE).exists());
        let receipt = GenerationComponentReceiptV1 { byte_len: 1, sha256: [1; 32] };
        assert!(matches!(open(&cx, &path, &receipt, false), Err(SearchError::Cancelled { .. })));
    });
}

#[test]
fn artifact_verification_observes_cancellation_between_chunks() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bytes");
        let bytes = vec![0x42; 128 * 1024];
        std::fs::write(&path, &bytes).unwrap();
        let expected = Artifact::from_bytes(&bytes);
        let mut chunks = 0;
        let result = read_and_verify(&cx, &path, expected, |_| {
            chunks += 1;
            cx.cancel_with(asupersync::CancelKind::User, Some("cancel verification"));
            Ok(())
        });
        assert_eq!(chunks, 1);
        assert!(matches!(result, Err(SearchError::Cancelled { .. })));
    });
}

#[cfg(unix)]
#[test]
fn source_and_snapshot_files_are_private_and_symlink_artifacts_are_refused() {
    use std::os::unix::fs::{PermissionsExt, symlink};
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("built");
        let built = build(&cx, &path, false, false).await;
        let receipt = built.seal_for_reopen(&cx).unwrap();
        for name in [SOURCE_FILE, SNAPSHOT_FILE] {
            assert_eq!(std::fs::metadata(path.join(name)).unwrap().permissions().mode() & 0o077, 0);
        }
        std::fs::rename(path.join(SOURCE_FILE), path.join("actual-source")).unwrap();
        symlink("actual-source", path.join(SOURCE_FILE)).unwrap();
        assert!(open(&cx, &path, &receipt, false).is_err());
        let alias = dir.path().join("alias");
        symlink(&path, &alias).unwrap();
        assert!(open(&cx, &alias, &receipt, false).is_err());
    });
}

#[test]
fn corrupt_files_are_refused_before_a_seal_receipt_is_issued() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("built");
        let built = build(&cx, &path, false, false).await;
        std::fs::write(built.fast().vector_path(), b"corrupted vector").unwrap();
        assert!(built.seal_for_reopen(&cx).is_err());
        assert!(!path.join(SOURCE_FILE).exists());
        assert!(!path.join(SNAPSHOT_FILE).exists());
    });
}
