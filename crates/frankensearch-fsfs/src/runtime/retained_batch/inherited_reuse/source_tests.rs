use std::collections::HashMap;

use asupersync::test_utils::run_test_with_cx;

use super::*;
use crate::generation_store::{CompleteGenerationStore, GenerationPublication};
use crate::runtime::retained_batch::{SourceAttributes, prepare, prepare_source_batch};
use crate::{CliCommand, CliInput, FsfsConfig};

fn attributes(revision: u64) -> SourceAttributes {
    SourceAttributes {
        modified_ms: revision,
        ingestion_class: IngestionClass::LexicalOnly,
        title: None,
        metadata: HashMap::new(),
    }
}

fn upsert(id: &str, text: &str) -> RetainedMutation {
    RetainedMutation::Upsert { id: id.to_owned(), text: text.to_owned() }
}

fn deleted(id: &str) -> RetainedMutation {
    RetainedMutation::Delete { id: id.to_owned() }
}

#[test]
fn source_input_witness_uses_final_raw_bytes_not_canonical_text() {
    run_test_with_cx(|cx| async move {
        let raw = "\r\n# changed source\r\n\r\nsecond paragraph α\r\n";
        let operations = [
            upsert("alpha", "superseded private body"),
            upsert("gone", "superseded then deleted"),
            deleted("alpha"),
            upsert("alpha", raw),
            deleted("gone"),
        ];
        let batch = prepare_source_batch(
            &cx, &operations,
            &BTreeMap::from([("alpha".to_owned(), attributes(7))]),
        ).unwrap().unwrap();
        assert_eq!(batch.source_inputs.len(), 1);
        let witness = &batch.source_inputs["alpha"];
        let body = batch.documents["alpha"].as_ref().unwrap();
        assert_ne!(body.embedding, raw, "fixture must distinguish raw and prepared input");
        assert_eq!(witness.content_hash_hex, crate::runtime::content_sha256_hex(raw.as_bytes()));
        assert_ne!(witness.content_hash_hex,
            crate::runtime::content_sha256_hex(body.embedding.as_bytes()));
        assert_eq!(witness.canonical_lines, crate::runtime::count_non_empty_lines(&body.embedding));
        assert_eq!(witness.revision, 7);
        assert_eq!(witness.ingestion_class, "lexical_only");
        assert!(batch.documents["gone"].is_none());
    });
}

#[test]
fn raw_document_input_without_source_authority_cannot_mint_a_witness() {
    run_test_with_cx(|cx| async move {
        let operations = [upsert("alpha", "ordinary opaque document")];
        let documents = prepare(&cx, &operations).unwrap();
        assert!(record_source_inputs(&cx, &operations, &documents).is_err());
        cx.set_cancel_requested(true);
        let cancelled = record_source_inputs(&cx, &[], &BTreeMap::new());
        cx.set_cancel_requested(false);
        assert!(matches!(cancelled, Err(frankensearch_core::SearchError::Cancelled { .. })));
    });
}

fn projected(input: &SourceInput) -> CheckpointFileEntry {
    CheckpointFileEntry {
        revision: input.revision,
        ingestion_class: input.ingestion_class.clone(),
        canonical_bytes: input.canonical_bytes,
        reason_code: input.reason_code.clone(),
        lexical_indexed: false,
        semantic_indexed: false,
        content_hash_hex: String::new(),
        fast_windows: None,
        canonical_lines: None,
    }
}

#[test]
fn source_witness_requires_the_completed_changed_member_contract() {
    run_test_with_cx(|cx| async move {
        let batch = prepare_source_batch(&cx, &[upsert("alpha", "actual source body")],
            &BTreeMap::from([("alpha".to_owned(), attributes(7))])).unwrap().unwrap();
        let row = projected(&batch.source_inputs["alpha"]);
        let changed = HashSet::from(["alpha".to_owned()]);
        for variant in 0..5 {
            let mut row = row.clone();
            match variant {
                0 => row.revision += 1,
                1 => row.canonical_bytes += 1,
                2 => row.ingestion_class = "full_semantic_lexical".to_owned(),
                3 => row.reason_code.push_str("-different"),
                _ => {}
            }
            let allowed = if variant == 4 { HashSet::new() } else { changed.clone() };
            assert!(extend_source_inputs(&cx,
                &mut BTreeMap::from([("alpha".to_owned(), row)]),
                &allowed, &batch.source_inputs).is_err());
        }
        let mut files = BTreeMap::from([("alpha".to_owned(), row)]);
        extend_source_inputs(&cx, &mut files, &changed, &batch.source_inputs).unwrap();
        assert!(files["alpha"].lexical_indexed);
        assert!(!files["alpha"].semantic_indexed, "lexical-only is not semantic coverage");
        assert_eq!(files["alpha"].content_hash_hex,
            crate::runtime::content_sha256_hex(b"actual source body"));
    });
}

fn durable(publication: GenerationPublication) -> PublishedGeneration {
    let GenerationPublication::Durable(generation) = publication else {
        panic!("fixture publication must be durable"); // ubs:ignore — test assertion.
    };
    generation
}

async fn fixture(cx: &Cx, parent: &Path) -> (FsfsRuntime, std::path::PathBuf, PublishedGeneration) {
    let source = parent.join("source");
    let store = parent.join("store");
    fs::create_dir(&source).unwrap();
    let source = fs::canonicalize(source).unwrap();
    fs::write(source.join("alpha.md"), "old alpha body").unwrap();
    fs::write(source.join("stable.md"), "stable body").unwrap();
    fs::write(source.join("skipped.md"), b"\0binary\0").unwrap();
    let mut config = FsfsConfig::default();
    config.indexing.offline = true;
    config.indexing.quality_model.clear();
    config.search.fast_only = true;
    config.search.rerank = false;
    "{index_dir}/catalog.sqlite".clone_into(&mut config.storage.db_path);
    let mut runtime = FsfsRuntime::new(config).with_cli_input(CliInput {
        command: CliCommand::Index,
        target_path: Some(source),
        index_dir: Some(store.clone()),
        quiet: true,
        ..CliInput::default()
    });
    runtime.lexical_only_indexing = true;
    let first = durable(runtime.rebuild_retained_generation(cx, &store).await.unwrap());
    (runtime, store, first)
}

fn admitted(cx: &Cx, generation: &PublishedGeneration) -> Receipt {
    let receipt = read_receipt(cx, generation).unwrap().expect("complete source evidence required");
    assert!(FsfsRuntime::read_checkpoint_manifest_generation(generation.path(),
        &receipt.checkpoint).unwrap().is_some());
    receipt
}

fn matches_scope(cx: &Cx, runtime: &FsfsRuntime, receipt: &Receipt) -> bool {
    runtime.source_reuse_scope_matches(cx, &receipt.session,
        receipt.executable_sha256.as_deref(), &receipt.configuration_sha256,
        &receipt.checkpoint.target_root).unwrap()
}

#[test]
fn source_reuse_admission_preserves_execution_configuration_and_target_scope() {
    run_test_with_cx(|cx| async move {
        let parent = tempfile::tempdir().unwrap();
        let (runtime, _, first) = fixture(&cx, parent.path()).await;
        let mut receipt = admitted(&cx, &first);
        assert!(matches_scope(&cx, &runtime, &receipt));
        let mut changed = runtime.clone();
        changed.config.indexing.quality_model.push_str("-different");
        assert!(!matches_scope(&cx, &changed, &receipt));
        let mut watch = runtime.clone();
        watch.config.indexing.watch_mode = !watch.config.indexing.watch_mode;
        assert!(matches_scope(&cx, &watch, &receipt), "watch lifetime is not an input policy");
        receipt.checkpoint.target_root.push_str("-different");
        assert!(!matches_scope(&cx, &runtime, &receipt));
        receipt = admitted(&cx, &first);
        receipt.session.push_str("-different");
        assert_eq!(matches_scope(&cx, &runtime, &receipt), receipt.executable_sha256.is_some());
        receipt.executable_sha256 = Some("invalid-executable-proof".to_owned());
        assert!(!matches_scope(&cx, &runtime, &receipt));
        cx.set_cancel_requested(true);
        let result = runtime.source_reuse_scope_matches(&cx, "", None, "", "");
        cx.set_cancel_requested(false);
        assert!(matches!(result, Err(frankensearch_core::SearchError::Cancelled { .. })));
    });
}

#[test]
fn source_batch_publishes_complete_fresh_and_inherited_evidence_atomically() {
    run_test_with_cx(|cx| async move {
        let parent = tempfile::tempdir().unwrap();
        let (runtime, root, first) = fixture(&cx, parent.path()).await;
        let before = admitted(&cx, &first);
        assert!(before.checkpoint.content_skipped.contains_key("skipped.md"));
        let original_bytes = fs::read(first.path().join(RECEIPT_FILE)).unwrap();
        let raw = "\r\n# replacementword\r\n\r\nsource α line\r\n";
        fs::write(parent.path().join("source/alpha.md"), raw).unwrap();
        let modified = crate::runtime::system_time_to_ms(
            fs::metadata(parent.path().join("source/alpha.md")).unwrap().modified().unwrap(),
        );
        fs::rename(parent.path().join("source/skipped.md"),
            parent.path().join("preserved-skipped.md")).unwrap();
        let make_batch = || prepare_source_batch(&cx,
            &[upsert("alpha.md", raw), deleted("skipped.md")],
            &BTreeMap::from([("alpha.md".to_owned(), attributes(modified))])).unwrap().unwrap();
        let failure = runtime.apply_retained_source_batch_with_precommit(
            &cx, &root, &first, make_batch(), |_| {
                Err(io::Error::other("injected source authority failure").into())
            }).await;
        assert!(failure.is_err());
        let store = CompleteGenerationStore::open(&cx, &root).unwrap();
        assert_eq!(store.active(&cx).unwrap(), Some(first.clone()));
        let outcome = runtime.apply_retained_source_batch_with_precommit(
            &cx, &root, &first, make_batch(), |_| Ok(())).await.unwrap();
        assert_eq!((outcome.upserted, outcome.deleted), (1, 0));
        let second = durable(outcome.publication.unwrap());
        let after = admitted(&cx, &second);
        let row = &after.checkpoint.files["alpha.md"];
        assert!(row.lexical_indexed);
        assert!(!row.semantic_indexed);
        assert_eq!(row.content_hash_hex, crate::runtime::content_sha256_hex(raw.as_bytes()));
        assert!(row.canonical_lines.is_some());
        assert_eq!(row.revision, i64::try_from(modified).unwrap());
        assert_eq!(serde_json::to_value(&after.checkpoint.files["stable.md"]).unwrap(),
            serde_json::to_value(&before.checkpoint.files["stable.md"]).unwrap());
        assert!(!after.checkpoint.content_skipped.contains_key("skipped.md"));
        assert_eq!(after.executable_sha256, before.executable_sha256);
        assert_eq!(after.configuration_sha256, before.configuration_sha256);
        assert_eq!(after.session, before.session);
        assert_eq!(fs::read(first.path().join(RECEIPT_FILE)).unwrap(), original_bytes);
        assert_eq!(store.active(&cx).unwrap(), Some(second));
    });
}
