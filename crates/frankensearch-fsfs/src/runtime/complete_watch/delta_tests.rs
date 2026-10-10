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

#[test]
fn removal_absence_requires_not_found_and_preserves_cancellation() {
    use std::os::unix::fs::symlink;

    run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let missing = directory.path().join("missing");
        let file = directory.path().join("file");
        let folder = directory.path().join("folder");
        let link = directory.path().join("dangling");
        fs::write(&file, "present source").unwrap();
        fs::create_dir(&folder).unwrap();
        symlink(&missing, &link).unwrap();
        assert!(physically_absent(&cx, &missing).unwrap());
        for path in [&file, &folder, &link] {
            assert!(!physically_absent(&cx, path).unwrap());
            assert!(is_source_changed(
                &recheck_removals(&cx, std::slice::from_ref(path)).unwrap_err()
            ));
        }
        assert!(matches!(
            physically_absent(&cx, &file.join("child")),
            Err(SearchError::Io(error)) if error.kind() != ErrorKind::NotFound
        ));
        cx.set_cancel_requested(true);
        let cancelled = physically_absent(&cx, &missing);
        cx.set_cancel_requested(false);
        assert!(matches!(cancelled, Err(SearchError::Cancelled { .. })));
    });
}

#[test]
fn excluded_or_replaced_sources_fall_back_before_candidate_allocation() {
    use std::os::unix::fs::symlink;

    run_test_with_cx(|cx| async move {
        for replacement in ["ignored", "directory", "dangling"] {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source_path, root) = fixture(directory.path());
            let source = SourceRoot::open(fs::canonicalize(&source_path).unwrap()).unwrap();
            let before = source.observe(&cx, &runtime.config.discovery).unwrap();
            let path = source.path.join("alpha.md");
            if replacement == "ignored" {
                fs::write(source.path.join(".ignore"), "alpha.md\n").unwrap();
            } else {
                fs::rename(&path, directory.path().join("preserved-alpha.md")).unwrap();
                if replacement == "directory" {
                    fs::create_dir(&path).unwrap();
                } else {
                    symlink("missing-target", &path).unwrap();
                }
            }
            let after = source.observe(&cx, &runtime.config.discovery).unwrap();
            assert!(before.stamps.contains_key(&path));
            assert!(!after.stamps.contains_key(&path));
            assert!(
                prepare(&cx, &runtime, &source, &before, &after, None)
                    .await
                    .unwrap()
                    .is_none(),
                "{replacement} must use normal reconciliation"
            );
            assert!(fs::symlink_metadata(&path).is_ok());
            assert!(!root.exists());
        }
    });
}

#[test]
fn removal_delta_keeps_forced_indexing_and_pressure_policies_on_the_normal_route() {
    run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let (runtime, source_path, root) = fixture(directory.path());
        let source = SourceRoot::open(fs::canonicalize(&source_path).unwrap()).unwrap();
        let before = source.observe(&cx, &runtime.config.discovery).unwrap();
        fs::rename(
            source.path.join("alpha.md"),
            directory.path().join("preserved-alpha.md"),
        )
        .unwrap();
        let after = source.observe(&cx, &runtime.config.discovery).unwrap();
        for mode in [
            DegradationOverrideMode::ForceLexicalOnly,
            DegradationOverrideMode::ForceMetadataOnly,
            DegradationOverrideMode::ForceEmbedDeferred,
        ] {
            let mut guarded = runtime.clone();
            guarded.config.pressure.degradation_override = mode;
            assert!(
                prepare(&cx, &guarded, &source, &before, &after, None)
                    .await
                    .unwrap()
                    .is_none()
            );
        }
        let mut forced = runtime.clone();
        forced.cli_input.full_reindex = true;
        assert!(
            prepare(&cx, &forced, &source, &before, &after, None)
                .await
                .unwrap()
                .is_none()
        );
        assert!(
            prepare(&cx, &runtime, &source, &before, &after, None)
                .await
                .unwrap()
                .is_some()
        );
        assert!(!root.exists(), "preparation must not initialize a store");
    });
}

#[test]
fn recreated_ignored_source_vetoes_a_sealed_removal_and_preserves_old_readers() {
    use std::os::unix::fs::symlink;

    use crate::generation_store::{COMPLETE_GENERATION_MANIFEST, COMPLETE_GENERATION_POINTER};

    run_test_with_cx(|cx| async move {
        for replacement in ["file", "directory", "dangling"] {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source, root) = fixture(directory.path());
            let mut session = controlled_session(&runtime, &cx, &root);
            let old = initial(&mut session, &cx).await;
            let mut pinned = lexical_reader(&runtime)
                .open_retained_search(&cx, &root)
                .await
                .unwrap();
            let pointer = fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap();
            let path = source.join("alpha.md");
            fs::rename(&path, directory.path().join("preserved-alpha.md")).unwrap();
            fs::write(source.join(".ignore"), "alpha.md\n").unwrap();
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
                    match replacement {
                        "file" => fs::write(&path, "recreated source must survive")?,
                        "directory" => fs::create_dir(&path)?,
                        _ => symlink("missing-target", &path)?,
                    }
                    // The usual observation agrees because .ignore hides the
                    // recreated path. The attached check must run AFTER this.
                    assert_eq!(
                        session.source.observe(cx, &runtime.config.discovery)?,
                        observed
                    );
                    Ok(())
                })
                .await;
            assert!(is_source_changed(&result.unwrap_err()), "{replacement}");
            let store = CompleteGenerationStore::open(&cx, &root).unwrap();
            assert_eq!(store.active(&cx).unwrap().as_ref(), Some(&old));
            assert_eq!(
                fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap(),
                pointer
            );
            assert!(fs::symlink_metadata(&path).is_ok());
            assert!(directory.path().join("preserved-alpha.md").is_file());
            let candidates = fs::read_dir(root.join("generations"))
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .filter(|path| path != old.path())
                .collect::<Vec<_>>();
            assert_eq!(candidates.len(), 1);
            assert!(candidates[0].join(COMPLETE_GENERATION_MANIFEST).is_file());
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
            assert_eq!(pinned.generation(), &old);
            drop(store.begin(&cx).unwrap());
        }
    });
}

#[test]
fn common_precommit_errors_and_cancellation_never_run_extra_source_checks() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    run_test_with_cx(|cx| async move {
        for cancel in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source, root) = fixture(directory.path());
            let mut session = controlled_session(&runtime, &cx, &root);
            let old = initial(&mut session, &cx).await;
            fs::rename(
                source.join("alpha.md"),
                directory.path().join("preserved-alpha.md"),
            )
            .unwrap();
            let observed = session
                .source
                .observe(&cx, &runtime.config.discovery)
                .unwrap();
            let calls = Arc::new(AtomicUsize::new(0));
            let witness = Arc::clone(&calls);
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
            .unwrap()
            .with_source_check(move |_| {
                witness.fetch_add(1, Ordering::SeqCst);
                Ok(())
            });
            let result = runtime
                .apply_retained_source_batch_with_precommit(&cx, &root, &old, batch, |cx| {
                    if cancel {
                        cx.set_cancel_requested(true);
                        Ok(())
                    } else {
                        Err(std::io::Error::new(
                            ErrorKind::PermissionDenied,
                            "injected backend refusal",
                        )
                        .into())
                    }
                })
                .await;
            cx.set_cancel_requested(false);
            let error = result.unwrap_err();
            if cancel {
                assert!(matches!(error, SearchError::Cancelled { .. }));
            } else {
                assert!(matches!(error, SearchError::Io(error)
                    if error.kind() == ErrorKind::PermissionDenied
                        && error.to_string() == "injected backend refusal"));
            }
            assert_eq!(calls.load(Ordering::SeqCst), 0);
            let store = CompleteGenerationStore::open(&cx, &root).unwrap();
            assert_eq!(store.active(&cx).unwrap(), Some(old));
            drop(store.begin(&cx).unwrap());
        }
    });
}

#[test]
fn guarded_removal_publishes_once_after_all_source_checks_in_order() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let (runtime, source, root) = fixture(directory.path());
        let mut session = controlled_session(&runtime, &cx, &root);
        let old = initial(&mut session, &cx).await;
        let before = FsfsRuntime::read_matching_manifest_generation(old.path())
            .unwrap()
            .unwrap();
        fs::rename(
            source.join("alpha.md"),
            directory.path().join("preserved-alpha.md"),
        )
        .unwrap();
        let observed = session
            .source
            .observe(&cx, &runtime.config.discovery)
            .unwrap();
        let order = Arc::new(AtomicUsize::new(0));
        let witness = Arc::clone(&order);
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
        .unwrap()
        .with_source_check(move |_| {
            assert_eq!(witness.swap(2, Ordering::SeqCst), 1);
            Ok(())
        });
        let outcome = runtime
            .apply_retained_source_batch_with_precommit(&cx, &root, &old, batch, |_| {
                assert_eq!(order.swap(1, Ordering::SeqCst), 0);
                Ok(())
            })
            .await
            .unwrap();
        assert_eq!((outcome.upserted, outcome.deleted), (0, 1));
        let next = require_durable_publication(outcome.publication.unwrap()).unwrap();
        assert_eq!(order.load(Ordering::SeqCst), 2);
        assert_eq!(fs::read_dir(root.join("generations")).unwrap().count(), 2);
        let manifests = FsfsRuntime::read_matching_manifest_generation(next.path())
            .unwrap()
            .unwrap();
        assert_eq!(
            manifests.keys().map(String::as_str).collect::<Vec<_>>(),
            ["stable.md"]
        );
        assert_eq!(
            serde_json::to_value(&manifests["stable.md"]).unwrap(),
            serde_json::to_value(&before["stable.md"]).unwrap()
        );
        assert!(old.path().is_dir());
        assert!(directory.path().join("preserved-alpha.md").is_file());
        let store = CompleteGenerationStore::open(&cx, &root).unwrap();
        assert_eq!(store.active(&cx).unwrap(), Some(next));
    });
}

#[test]
fn a_build_that_loses_a_race_retries_as_a_delta_with_every_hint() {
    run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let (runtime, source, root) = fixture(directory.path());
        let mut session = controlled_session(&runtime, &cx, &root);
        initial(&mut session, &cx).await;
        fs::write(source.join("alpha.md"), "racedword alpha body").unwrap();
        fs::write(source.join("stable.md"), "laterword stable body").unwrap();
        // An attempt consumed alpha's hint; stable's arrived before its
        // precommit, which refused the build as a source change.
        let now = Instant::now();
        let consumed = {
            let mut changes = lock_changes(&session.changes).unwrap();
            changes.record(now, false);
            changes.record_path(&source.join("alpha.md"));
            changes.take_due(now + DEBOUNCE).unwrap()
        };
        let lost = now + DEBOUNCE;
        {
            let mut changes = lock_changes(&session.changes).unwrap();
            changes.record(lost, false);
            changes.record_path(&source.join("stable.md"));
        }
        session
            .record_unstable_source(lost, Some(&consumed))
            .unwrap();
        let (forced, hinted) = lock_changes(&session.changes)
            .unwrap()
            .dirty
            .as_ref()
            .map(|window| (window.force_rebuild, window.paths.clone()))
            .unwrap();
        assert!(!forced, "a lost race must not force a whole-corpus rebuild");
        assert_eq!(
            hinted,
            BTreeSet::from([source.join("alpha.md"), source.join("stable.md")])
        );
        let retry = require_durable_publication(
            session
                .advance(&cx, lost + DEBOUNCE)
                .await
                .unwrap()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            FsfsRuntime::read_index_sentinel(retry.path())
                .unwrap()
                .unwrap()
                .command,
            "retained-batch"
        );
        let mut reader = lexical_reader(&runtime)
            .open_retained_search(&cx, &root)
            .await
            .unwrap();
        for word in ["racedword", "laterword"] {
            let hits = reader.search(&cx, word, 10).await.unwrap();
            assert_eq!(hits.last().unwrap().hits.len(), 1, "{word}");
        }
    });
}
