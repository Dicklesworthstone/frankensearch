use std::sync::{Arc, Mutex};
use std::time::Instant;

use asupersync::test_utils::run_test_with_cx;
use notify::Watcher;

use super::super::{
    Changes, CompleteWatchSession, DEBOUNCE, RECONCILE_INTERVAL, is_source_changed, lock_changes,
    require_durable_publication,
};
use super::*;
use crate::config::{DegradationOverrideMode, DiscoveryConfig};
use crate::generation_store::{CompleteGenerationStore, PublishedGeneration};
use crate::{CliCommand, CliInput, FsfsConfig, OutputFormat};

fn observation(paths: &[(&str, u64)]) -> SourceObservation {
    SourceObservation {
        stamps: paths
            .iter()
            .map(|(path, inode)| {
                (
                    PathBuf::from(path),
                    SourceStamp {
                        device: 1,
                        inode: *inode,
                        bytes: 5,
                        modified: (1, 0),
                        changed: (1, 0),
                    },
                )
            })
            .collect(),
        classes: paths
            .iter()
            .map(|(path, _)| (PathBuf::from(path), IngestionClass::FullSemanticLexical))
            .collect(),
        directories: BTreeMap::new(),
    }
}

#[test]
fn observation_delta_tracks_membership_stamps_policy_and_same_stamp_hints() {
    run_test_with_cx(|cx| async move {
        let before = observation(&[
            ("/src/stable.md", 1),
            ("/src/deleted.md", 2),
            ("/src/edited.md", 3),
            ("/src/hinted.md", 4),
            ("/src/policy.md", 5),
        ]);
        let mut after = observation(&[
            ("/src/stable.md", 1),
            ("/src/renamed.md", 2),
            ("/src/edited.md", 6),
            ("/src/hinted.md", 4),
            ("/src/policy.md", 5),
        ]);
        after
            .classes
            .insert(PathBuf::from("/src/policy.md"), IngestionClass::LexicalOnly);
        let hint = DirtyWindow {
            first: Instant::now(),
            last: Instant::now(),
            force_rebuild: false,
            paths: BTreeSet::from([PathBuf::from("/src/hinted.md")]),
        };
        assert_eq!(
            changed_paths(&cx, &before, &after, Some(&hint))
                .unwrap()
                .unwrap(),
            [
                "/src/deleted.md",
                "/src/renamed.md",
                "/src/edited.md",
                "/src/hinted.md",
                "/src/policy.md"
            ]
            .into_iter()
            .map(PathBuf::from)
            .collect()
        );
        assert!(
            changed_paths(&cx, &before, &before, None)
                .unwrap()
                .unwrap()
                .is_empty()
        );
    });
}

#[test]
fn deletions_on_either_side_of_the_current_paths_are_changes() {
    run_test_with_cx(|cx| async move {
        let before = observation(&[("/src/a.md", 1), ("/src/m.md", 2), ("/src/z.md", 3)]);
        let after = observation(&[("/src/m.md", 2)]);
        assert_eq!(
            changed_paths(&cx, &before, &after, None).unwrap().unwrap(),
            BTreeSet::from([PathBuf::from("/src/a.md"), PathBuf::from("/src/z.md")])
        );
        assert_eq!(
            changed_paths(&cx, &before, &observation(&[]), None)
                .unwrap()
                .unwrap()
                .len(),
            3
        );
    });
}

#[test]
fn directory_hints_use_component_boundaries_and_large_deltas_fall_back() {
    run_test_with_cx(|cx| async move {
        let before = observation(&[("/src/dir/a.md", 1), ("/src/dir-sibling/a.md", 2)]);
        let mut hint = DirtyWindow {
            first: Instant::now(),
            last: Instant::now(),
            force_rebuild: false,
            paths: BTreeSet::from([PathBuf::from("/src/dir")]),
        };
        assert_eq!(
            changed_paths(&cx, &before, &before, Some(&hint))
                .unwrap()
                .unwrap(),
            BTreeSet::from([PathBuf::from("/src/dir/a.md")])
        );
        hint.force_rebuild = true;
        assert!(
            changed_paths(&cx, &before, &before, Some(&hint))
                .unwrap()
                .is_none()
        );
        let mut many = observation(&[]);
        for count in 0..=MAX_CHANGED_PATHS {
            let path = PathBuf::from(format!("/src/{count}.md"));
            many.stamps.insert(
                path.clone(),
                SourceStamp {
                    device: 1,
                    inode: count as u64,
                    bytes: 5,
                    modified: (1, 0),
                    changed: (1, 0),
                },
            );
            many.classes
                .insert(path, IngestionClass::FullSemanticLexical);
            assert_eq!(
                changed_paths(&cx, &observation(&[]), &many, None)
                    .unwrap()
                    .is_some(),
                count < MAX_CHANGED_PATHS
            );
        }
    });
}

#[test]
fn exact_source_ids_do_not_admit_lossy_aliases_or_escape_paths() {
    use std::os::unix::ffi::OsStringExt;
    let root = Path::new("/src");
    assert_eq!(
        source_id(root, Path::new("/src/dir/α.md")),
        Some("dir/α.md".to_owned())
    );
    for path in [
        "/src",
        "/elsewhere/a",
        "/src/../a",
        "/src/a\\b",
        "/src-sibling/a",
    ] {
        assert!(source_id(root, Path::new(path)).is_none(), "{path}");
    }
    let path = PathBuf::from(std::ffi::OsString::from_vec(b"/src/a\xff.md".to_vec()));
    assert!(source_id(root, &path).is_none());
}

#[test]
fn bounded_read_binds_opened_bytes_and_refuses_a_replaced_or_cancelled_source() {
    run_test_with_cx(|cx| async move {
        let parent = tempfile::tempdir().unwrap();
        let path = parent.path().join("alpha.md");
        fs::write(&path, "alpha content").unwrap();
        let stamp = SourceStamp::from_metadata(&fs::metadata(&path).unwrap());
        assert_eq!(read_source(&cx, &path, &stamp).unwrap().0, b"alpha content");
        fs::rename(&path, parent.path().join("retained-original.md")).unwrap();
        fs::write(&path, "other content").unwrap();
        assert!(is_source_changed(
            &read_source(&cx, &path, &stamp).unwrap_err()
        ));
        cx.set_cancel_requested(true);
        let result = read_source(&cx, &path, &stamp);
        cx.set_cancel_requested(false);
        assert!(matches!(result, Err(SearchError::Cancelled { .. })));
    });
}

fn fixture(parent: &Path) -> (FsfsRuntime, PathBuf, PathBuf) {
    let source = parent.join("source");
    let root = parent.join("store");
    fs::create_dir(&source).unwrap();
    fs::write(source.join("alpha.md"), "obsoleteword original").unwrap();
    fs::write(source.join("stable.md"), "stableword original").unwrap();
    let mut config = FsfsConfig::default();
    config.indexing.offline = true;
    config.indexing.quality_model.clear();
    config.search.fast_only = true;
    config.search.rerank = false;
    "{index_dir}/catalog.sqlite".clone_into(&mut config.storage.db_path);
    let mut runtime = FsfsRuntime::new(config).with_cli_input(CliInput {
        command: CliCommand::Watch,
        target_path: Some(source.clone()),
        index_dir: Some(root.clone()),
        quiet: true,
        format: OutputFormat::Jsonl,
        ..CliInput::default()
    });
    // A genuine model-free lane, not fabricated semantic success.
    runtime.lexical_only_indexing = true;
    (runtime, source, root)
}

/// Readers for the model-free fixture: a semantic-capable build serves a
/// lexical-only generation only under an explicit lexical-only override
/// (`complete_lexical_only_generation_serves_search_append_and_compact`).
fn lexical_reader(runtime: &FsfsRuntime) -> FsfsRuntime {
    let mut reader = runtime.clone();
    reader.config.pressure.degradation_override = DegradationOverrideMode::ForceLexicalOnly;
    reader
}

// `_watcher` only keeps the production watcher alive; tests detach it here.
#[allow(clippy::used_underscore_binding)]
fn controlled_session(runtime: &FsfsRuntime, cx: &Cx, root: &Path) -> CompleteWatchSession {
    let mut session = CompleteWatchSession::open(runtime, cx, root).unwrap();
    session._watcher.unwatch(&session.source.path).unwrap();
    session.changes = Arc::new(Mutex::new(Changes::default()));
    session
}

async fn initial(session: &mut CompleteWatchSession, cx: &Cx) -> PublishedGeneration {
    require_durable_publication(session.advance(cx, Instant::now()).await.unwrap().unwrap())
        .unwrap()
}

async fn change(session: &mut CompleteWatchSession, cx: &Cx, path: &Path) -> PublishedGeneration {
    let now = Instant::now();
    {
        let mut changes = lock_changes(&session.changes).unwrap();
        changes.record(now, false);
        changes.record_path(path);
    }
    require_durable_publication(session.advance(cx, now + DEBOUNCE).await.unwrap().unwrap())
        .unwrap()
}

#[test]
fn watcher_batches_edit_add_delete_and_rename_without_reindexing_unchanged_rows() {
    run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let (runtime, source, root) = fixture(directory.path());
        let mut session = controlled_session(&runtime, &cx, &root);
        let old = initial(&mut session, &cx).await;
        let mut pinned = lexical_reader(&runtime)
            .open_retained_search(&cx, &root)
            .await
            .unwrap();
        let before = FsfsRuntime::read_matching_manifest_generation(old.path())
            .unwrap()
            .unwrap();
        fs::write(source.join("alpha.md"), "replacementword new body").unwrap();
        fs::write(source.join("added.md"), "addedword new body").unwrap();
        let second = change(&mut session, &cx, &source.join("alpha.md")).await;
        assert_eq!(
            FsfsRuntime::read_index_sentinel(second.path())
                .unwrap()
                .unwrap()
                .command,
            "retained-batch",
            "the real watcher must use its mixed-mutation route"
        );
        assert_eq!(fs::read_dir(root.join("generations")).unwrap().count(), 2);
        let manifests = FsfsRuntime::read_matching_manifest_generation(second.path())
            .unwrap()
            .unwrap();
        assert_eq!(
            serde_json::to_value(&manifests["stable.md"]).unwrap(),
            serde_json::to_value(&before["stable.md"]).unwrap()
        );
        assert_eq!(
            manifests["alpha.md"].revision,
            i64::try_from(system_time_to_ms(
                fs::metadata(source.join("alpha.md"))
                    .unwrap()
                    .modified()
                    .unwrap()
            ))
            .unwrap()
        );
        let mut reader = lexical_reader(&runtime)
            .open_retained_search(&cx, &root)
            .await
            .unwrap();
        assert_eq!(
            reader
                .search(&cx, "replacementword", 10)
                .await
                .unwrap()
                .last()
                .unwrap()
                .hits
                .len(),
            1
        );
        assert!(
            reader
                .search(&cx, "obsoleteword", 10)
                .await
                .unwrap()
                .last()
                .unwrap()
                .hits
                .is_empty()
        );
        assert_eq!(
            pinned
                .search(&cx, "obsoleteword", 10)
                .await
                .unwrap()
                .last()
                .unwrap()
                .hits
                .len(),
            1
        );
        fs::rename(source.join("added.md"), source.join("renamed.md")).unwrap();
        fs::rename(
            source.join("alpha.md"),
            directory.path().join("preserved-alpha.md"),
        )
        .unwrap();
        let third = change(&mut session, &cx, &source.join("alpha.md")).await;
        assert_eq!(
            FsfsRuntime::read_index_sentinel(third.path())
                .unwrap()
                .unwrap()
                .command,
            "retained-batch"
        );
        assert_eq!(fs::read_dir(root.join("generations")).unwrap().count(), 3);
        let manifests = FsfsRuntime::read_matching_manifest_generation(third.path())
            .unwrap()
            .unwrap();
        assert_eq!(
            manifests.keys().map(String::as_str).collect::<Vec<_>>(),
            ["renamed.md", "stable.md"]
        );
        let mut current = lexical_reader(&runtime)
            .open_retained_search(&cx, &root)
            .await
            .unwrap();
        let results = current.search(&cx, "addedword", 10).await.unwrap();
        assert!(results.last().unwrap().hits[0].path.ends_with("renamed.md"));
        assert_eq!(pinned.generation(), &old);
        assert!(old.path().is_dir());
    });
}

#[test]
fn periodic_scan_finds_missed_edits_while_forced_and_unsupported_changes_rebuild() {
    run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let (runtime, source, root) = fixture(directory.path());
        let mut session = controlled_session(&runtime, &cx, &root);
        initial(&mut session, &cx).await;
        fs::write(
            source.join("alpha.md"),
            "periodicword changed without notification",
        )
        .unwrap();
        let due = session.last_reconcile + RECONCILE_INTERVAL;
        let selected =
            require_durable_publication(session.advance(&cx, due).await.unwrap().unwrap()).unwrap();
        assert_eq!(
            FsfsRuntime::read_index_sentinel(selected.path())
                .unwrap()
                .unwrap()
                .command,
            "retained-batch"
        );
        let now = Instant::now();
        lock_changes(&session.changes).unwrap().record(now, true);
        let forced = require_durable_publication(
            session.advance(&cx, now + DEBOUNCE).await.unwrap().unwrap(),
        )
        .unwrap();
        assert_ne!(
            FsfsRuntime::read_index_sentinel(forced.path())
                .unwrap()
                .unwrap()
                .command,
            "retained-batch"
        );
        fs::write(source.join("alpha.md"), b"\0binary\0content").unwrap();
        let rebuilt = change(&mut session, &cx, &source.join("alpha.md")).await;
        assert_ne!(
            FsfsRuntime::read_index_sentinel(rebuilt.path())
                .unwrap()
                .unwrap()
                .command,
            "retained-batch"
        );
        let mut reader = lexical_reader(&runtime)
            .open_retained_search(&cx, &root)
            .await
            .unwrap();
        assert!(
            reader
                .search(&cx, "periodicword", 10)
                .await
                .unwrap()
                .last()
                .unwrap()
                .hits
                .is_empty()
        );
    });
}

#[test]
fn delta_generations_seal_only_the_segments_their_manifest_names() {
    run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let (runtime, source, root) = fixture(directory.path());
        let mut session = controlled_session(&runtime, &cx, &root);
        initial(&mut session, &cx).await;
        // Each batch adds one small segment; the tier policy folds eight.
        let mut merged = false;
        for round in 0..10 {
            fs::write(source.join("alpha.md"), format!("roundword{round} body")).unwrap();
            let generation = change(&mut session, &cx, &source.join("alpha.md")).await;
            assert_eq!(
                FsfsRuntime::read_index_sentinel(generation.path())
                    .unwrap()
                    .unwrap()
                    .command,
                "retained-batch"
            );
            let engine = FsfsRuntime::resolve_lexical_engine(generation.path())
                .unwrap()
                .engine_dir()
                .unwrap();
            let named = frankensearch_quill::keeper::load_manifest_pair(&engine)
                .unwrap()
                .manifest
                .segments
                .len();
            let sealed = fs::read_dir(&engine)
                .unwrap()
                .filter(|entry| {
                    entry
                        .as_ref()
                        .unwrap()
                        .file_name()
                        .to_string_lossy()
                        .ends_with(".fslx")
                })
                .count();
            assert_eq!(
                sealed, named,
                "round {round}: a merge's inputs must not be sealed with its output"
            );
            merged |= named < round + 2;
        }
        assert!(merged, "the fixture must reach a tier merge");
    });
}

#[test]
fn prepared_delta_still_requires_post_seal_source_authority() {
    run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let (runtime, source, root) = fixture(directory.path());
        let mut session = controlled_session(&runtime, &cx, &root);
        let old = initial(&mut session, &cx).await;
        fs::write(source.join("alpha.md"), "preparedword candidate").unwrap();
        let observed = session
            .source
            .observe(&cx, &runtime.config.discovery)
            .unwrap();
        let batch = prepare(
            &cx,
            &runtime,
            &session.source,
            session.baseline.as_ref().unwrap(),
            &observed,
            None,
        )
        .await
        .unwrap()
        .unwrap();
        let result = runtime
            .apply_retained_source_batch_with_precommit(&cx, &root, &old, batch, |cx| {
                fs::write(source.join("alpha.md"), "racedword after sealing")?;
                if session.source.observe(cx, &runtime.config.discovery)? != observed {
                    return Err(source_changed());
                }
                Ok(())
            })
            .await;
        assert!(is_source_changed(&result.unwrap_err()));
        let store = CompleteGenerationStore::open(&cx, &root).unwrap();
        assert_eq!(store.active(&cx).unwrap(), Some(old));
        drop(store.begin(&cx).unwrap());
    });
}

#[test]
fn unreadable_ignore_policy_is_not_an_authoritative_membership_observation() {
    run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("alpha.md"), "ordinary source").unwrap();
        fs::write(directory.path().join(".ignore"), "[z-a]\n").unwrap();
        let source = SourceRoot::open(fs::canonicalize(directory.path()).unwrap()).unwrap();
        assert!(source.observe(&cx, &DiscoveryConfig::default()).is_err());
    });
}
