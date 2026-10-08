use asupersync::test_utils::run_test_with_cx;
use serde_json::Value;

use super::*;
use crate::generation_store::{CompleteGenerationStore, GenerationPublication};
use crate::runtime::retained_batch::RetainedMutation;
use crate::{CliCommand, CliInput, FsfsConfig};

fn manifest(id: &str) -> IndexManifestEntry {
    IndexManifestEntry {
        file_key: id.to_owned(),
        revision: 7,
        ingestion_class: "full_semantic_lexical".to_owned(),
        canonical_bytes: 12,
        reason_code: "index.plan.full_semantic_lexical".to_owned(),
        fast_windows: None,
    }
}

fn entry(manifest: &IndexManifestEntry) -> CheckpointFileEntry {
    CheckpointFileEntry {
        revision: manifest.revision,
        ingestion_class: manifest.ingestion_class.clone(),
        canonical_bytes: manifest.canonical_bytes,
        reason_code: manifest.reason_code.clone(),
        lexical_indexed: true,
        semantic_indexed: true,
        content_hash_hex: "ab".repeat(32),
        fast_windows: manifest.fast_windows.clone(),
        canonical_lines: None,
    }
}

#[test]
fn projection_preserves_survivors_but_invalidates_even_identical_replacements() {
    run_test_with_cx(|cx| async move {
        let stable = manifest("stable");
        let replaced = manifest("replaced");
        let deleted = manifest("deleted");
        let old = BTreeMap::from([
            ("stable".to_owned(), entry(&stable)),
            ("replaced".to_owned(), entry(&replaced)),
            ("deleted".to_owned(), entry(&deleted)),
        ]);
        let current = BTreeMap::from([
            ("stable".to_owned(), stable),
            ("replaced".to_owned(), replaced),
            ("added".to_owned(), manifest("added")),
        ]);
        let changed = HashSet::from([
            "replaced".to_owned(), "deleted".to_owned(), "added".to_owned(),
        ]);
        let projected = project_files(&cx, &old, &current, &changed).unwrap();
        assert_eq!(projected.len(), 3);
        assert_eq!(serde_json::to_value(&projected["stable"]).unwrap(),
            serde_json::to_value(&old["stable"]).unwrap());
        for id in ["replaced", "added"] {
            let row = &projected[id];
            assert!(!row.lexical_indexed);
            assert!(!row.semantic_indexed);
            assert!(row.content_hash_hex.is_empty());
            assert!(row.canonical_lines.is_none());
            assert!(matches_manifest(row, &current[id]));
        }
        assert!(!projected.contains_key("deleted"));
        // A subsequent unrelated batch cannot promote an already invalidated row.
        let again = project_files(&cx, &projected, &current, &HashSet::new()).unwrap();
        assert_eq!(serde_json::to_value(&again).unwrap(), serde_json::to_value(&projected).unwrap());
    });
}

#[test]
fn changed_survivor_contract_or_unexplained_addition_cannot_borrow_an_old_hash() {
    run_test_with_cx(|cx| async move {
        let original = manifest("stable");
        let old = BTreeMap::from([("stable".to_owned(), entry(&original))]);
        let mut variants = Vec::new();
        let mut changed = original.clone();
        changed.revision += 1;
        variants.push(changed);
        let mut changed = original.clone();
        changed.ingestion_class = "lexical_only".to_owned();
        variants.push(changed);
        let mut changed = original.clone();
        changed.canonical_bytes += 1;
        variants.push(changed);
        let mut changed = original;
        changed.reason_code.push_str("-other");
        variants.push(changed);
        for changed in variants {
            assert!(project_files(&cx, &old,
                &BTreeMap::from([("stable".to_owned(), changed)]), &HashSet::new()).is_err());
        }
        assert!(project_files(&cx, &old,
            &BTreeMap::from([("new".to_owned(), manifest("new"))]), &HashSet::new()).is_err());
    });
}

#[test]
fn cancelled_projection_does_not_return_partial_evidence() {
    run_test_with_cx(|cx| async move {
        let row = manifest("stable");
        let old = BTreeMap::from([("stable".to_owned(), entry(&row))]);
        cx.set_cancel_requested(true);
        let result = project_files(&cx, &old,
            &BTreeMap::from([("stable".to_owned(), row)]), &HashSet::new());
        cx.set_cancel_requested(false);
        assert!(matches!(
            result,
            Err(frankensearch_core::SearchError::Cancelled { .. })
        ));
    });
}

fn durable(outcome: GenerationPublication) -> PublishedGeneration {
    match outcome {
        GenerationPublication::Durable(generation) => generation,
        GenerationPublication::VisibleButDurabilityUncertain { .. } => {
            panic!("fixture publication must reach durability"); // ubs:ignore — test assertion.
        }
    }
}

#[test]
fn unsupported_or_absent_receipts_stay_cold_and_changed_bytes_fail_authentication() {
    run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let store = CompleteGenerationStore::create(&cx, directory.path()).unwrap();
        let first = store.begin(&cx).unwrap();
        fs::write(first.path().join("fixture"), b"payload").unwrap();
        let first = durable(first.publish(&cx, |_, _| Ok(())).unwrap());
        assert!(read_receipt(&cx, &first).unwrap().is_none());
        let next = store.begin(&cx).unwrap();
        fs::write(next.path().join(RECEIPT_FILE), b"{\"version\":73}").unwrap();
        let next = durable(next.publish(&cx, |_, _| Ok(())).unwrap());
        assert!(read_receipt(&cx, &next).unwrap().is_none());
        // Read a pinned receipt after same-length damage. No cached admission
        // may turn these different bytes into inherited indexing evidence.
        fs::write(next.path().join(RECEIPT_FILE), b"{\"version\":74}").unwrap();
        assert!(read_receipt(&cx, &next).is_err());
    });
}

async fn fixture(cx: &Cx, parent: &Path) -> (FsfsRuntime, std::path::PathBuf, PublishedGeneration) {
    let source = parent.join("source");
    let root = parent.join("store");
    fs::create_dir(&source).unwrap();
    fs::write(source.join("stable.md"), "stableword source").unwrap();
    fs::write(source.join("alpha.md"), "originalword source").unwrap();
    fs::write(source.join("new.md"), b"\0binary\0content").unwrap();
    let mut config = FsfsConfig::default();
    config.indexing.offline = true;
    config.indexing.quality_model.clear();
    config.search.fast_only = true;
    config.search.rerank = false;
    "{index_dir}/catalog.sqlite".clone_into(&mut config.storage.db_path);
    let mut runtime = FsfsRuntime::new(config).with_cli_input(CliInput {
        command: CliCommand::Index,
        target_path: Some(source),
        index_dir: Some(root.clone()),
        quiet: true,
        ..CliInput::default()
    });
    runtime.lexical_only_indexing = true;
    let first = durable(runtime.rebuild_retained_generation(cx, &root).await.unwrap());
    (runtime, root, first)
}

fn receipt_json(generation: &PublishedGeneration) -> Value {
    serde_json::from_slice(&fs::read(generation.path().join(RECEIPT_FILE)).unwrap()).unwrap()
}

// The existing resume validator, not the new writer, decides whether a partial
// checkpoint is representable. Compute its input independently from the emitted
// receipt so neither an invalid receipt nor an unnecessary cold miss can pass.
fn projected_checkpoint_is_admitted(
    cx: &Cx,
    original: &Value,
    generation: &PublishedGeneration,
    touched: &[&str],
) -> bool {
    let mut checkpoint: IndexingCheckpoint =
        serde_json::from_value(original["checkpoint"].clone()).unwrap();
    let manifests = FsfsRuntime::read_matching_manifest_generation(generation.path())
        .unwrap()
        .unwrap();
    checkpoint.files.retain(|id, _| manifests.contains_key(id));
    for id in touched {
        if let Some(manifest) = manifests.get(*id) {
            let mut row = entry(manifest);
            row.lexical_indexed = false;
            row.semantic_indexed = false;
            row.content_hash_hex.clear();
            checkpoint.files.insert((*id).to_owned(), row);
        }
        checkpoint.content_skipped.remove(*id);
    }
    let sentinel = FsfsRuntime::read_index_sentinel(generation.path()).unwrap().unwrap();
    checkpoint.index_root = generation.path().display().to_string();
    checkpoint.source_hash_hex = sentinel.source_hash_hex;
    checkpoint.discovered_files = sentinel.discovered_files;
    checkpoint.skipped_files = sentinel.skipped_files;
    checkpoint.updated_at_ms = sentinel.generated_at_ms;
    let admitted = FsfsRuntime::read_checkpoint_manifest_generation(generation.path(), &checkpoint)
        .unwrap()
        .is_some();
    assert_eq!(generation.path().join(RECEIPT_FILE).exists(), admitted);
    if admitted {
        let receipt = read_receipt(cx, generation).unwrap().unwrap();
        assert_eq!(
            serde_json::to_value(receipt.checkpoint).unwrap(),
            serde_json::to_value(checkpoint).unwrap()
        );
    }
    admitted
}

#[test]
fn real_mixed_batches_preserve_only_admitted_witnesses_without_blocking_mutation() {
    run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let (runtime, root, first) = fixture(&cx, directory.path()).await;
        let original = receipt_json(&first);
        assert!(original["checkpoint"]["content_skipped"].get("new.md").is_some());
        let original_bytes = fs::read(first.path().join(RECEIPT_FILE)).unwrap();
        let outcome = runtime.apply_retained_batch(&cx, &root, &first, &[
            RetainedMutation::Upsert { id: "alpha.md".to_owned(), text: "replacementword body".to_owned() },
            RetainedMutation::Upsert { id: "new.md".to_owned(), text: "addedword body".to_owned() },
        ]).await.unwrap();
        assert_eq!((outcome.upserted, outcome.deleted), (2, 0));
        let second = durable(outcome.publication.unwrap());
        let admitted = projected_checkpoint_is_admitted(
            &cx, &original, &second, &["alpha.md", "new.md"],
        );
        if admitted {
            let inherited = receipt_json(&second);
            assert!(inherited["checkpoint"]["content_skipped"].get("new.md").is_none());
            for key in ["version", "session", "executable_sha256", "configuration_sha256"] {
                assert_eq!(inherited[key], original[key]);
            }
            assert_eq!(inherited["checkpoint"]["files"]["stable.md"],
                original["checkpoint"]["files"]["stable.md"]);
            for id in ["alpha.md", "new.md"] {
                assert_eq!(inherited["checkpoint"]["files"][id]["lexical_indexed"], false);
                assert_eq!(inherited["checkpoint"]["files"][id]["semantic_indexed"], false);
                assert_eq!(inherited["checkpoint"]["files"][id]["content_hash_hex"], "");
            }
        }
        let outcome = runtime.apply_retained_batch(&cx, &root, &second, &[
            RetainedMutation::Delete { id: "new.md".to_owned() },
        ]).await.unwrap();
        assert_eq!((outcome.upserted, outcome.deleted), (0, 1));
        let third = durable(outcome.publication.unwrap());
        if admitted {
            projected_checkpoint_is_admitted(&cx, &receipt_json(&second), &third, &["new.md"]);
        } else {
            assert!(read_receipt(&cx, &third).unwrap().is_none());
        }
        assert_eq!(fs::read(first.path().join(RECEIPT_FILE)).unwrap(), original_bytes);
        assert_eq!(
            fs::read_to_string(directory.path().join("source/alpha.md")).unwrap(),
            "originalword source"
        );
        assert_eq!(CompleteGenerationStore::open(&cx, &root).unwrap().active(&cx).unwrap(), Some(third));
    });
}

#[test]
fn inherited_serialization_rejects_overflow_before_growing_the_output() {
    let mut output = BoundedBytes(Vec::new());
    let full = vec![b'x'; MAX_RECEIPT_BYTES];
    assert_eq!(output.write(&full).unwrap(), MAX_RECEIPT_BYTES);
    assert!(output.write(b"x").is_err());
    assert_eq!(output.0.len(), MAX_RECEIPT_BYTES);
}

#[test]
fn removal_only_batches_keep_full_survivor_evidence_and_failed_precommit_cannot_select_it() {
    run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let (runtime, root, first) = fixture(&cx, directory.path()).await;
        let original = receipt_json(&first);
        let operations = [RetainedMutation::Delete { id: "alpha.md".to_owned() }];
        let result = runtime.apply_retained_batch_with_precommit(&cx, &root, &first, &operations, |_| {
            Err(io::Error::other("injected source precommit failure").into())
        }).await;
        assert!(result.is_err());
        let store = CompleteGenerationStore::open(&cx, &root).unwrap();
        assert_eq!(store.active(&cx).unwrap(), Some(first.clone()));
        let outcome = runtime.apply_retained_batch(&cx, &root, &first, &operations).await.unwrap();
        let next = durable(outcome.publication.unwrap());
        let inherited = receipt_json(&next);
        assert_eq!(inherited["checkpoint"]["files"].as_object().unwrap().len(), 1);
        assert_eq!(inherited["checkpoint"]["files"]["stable.md"],
            original["checkpoint"]["files"]["stable.md"]);
        assert!(FsfsRuntime::read_checkpoint_manifest_generation(next.path(),
            &read_receipt(&cx, &next).unwrap().unwrap().checkpoint).unwrap().is_some());
        assert_eq!(fs::read_to_string(directory.path().join("source/alpha.md")).unwrap(), "originalword source");
    });
}

#[test]
fn replacing_every_source_does_not_mint_a_new_reuse_receipt() {
    run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let (runtime, root, first) = fixture(&cx, directory.path()).await;
        let outcome = runtime.apply_retained_batch(&cx, &root, &first, &[
            RetainedMutation::Upsert { id: "alpha.md".to_owned(), text: "changed alpha".to_owned() },
            RetainedMutation::Upsert { id: "stable.md".to_owned(), text: "changed stable".to_owned() },
        ]).await.unwrap();
        let next = durable(outcome.publication.unwrap());
        assert!(!next.path().join(RECEIPT_FILE).exists());
        assert!(first.path().join(RECEIPT_FILE).is_file());
        assert!(read_receipt(&cx, &next).unwrap().is_none());
    });
}
