use super::*;
use asupersync::test_utils::run_test_with_cx;
use std::io::Read;

fn published(store: &CompleteGenerationStore, cx: &Cx, body: &[u8]) -> PublishedGeneration {
    let build = store.begin(cx).unwrap();
    fs::create_dir(build.path().join("lexical")).unwrap();
    for name in ["lexical/MANIFEST", "vector.idx", "content.txt"] {
        fs::write(build.path().join(name), body).unwrap();
    }
    durable(build.publish(cx, |_, _| Ok(())).unwrap())
}

fn durable(publication: GenerationPublication) -> PublishedGeneration {
    match publication {
        GenerationPublication::Durable(generation) => generation,
        other @ GenerationPublication::VisibleButDurabilityUncertain { .. } => {
            panic!("expected a durable publication: {other:?}") // ubs:ignore — test assertion.
        }
    }
}

fn pointer(store: &CompleteGenerationStore) -> Vec<u8> {
    fs::read(store.root().join(COMPLETE_GENERATION_POINTER)).unwrap()
}

#[test]
fn retained_open_requires_exact_receipt_and_never_changes_selection() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
        let old = published(&store, &cx, b"old");
        let current = published(&store, &cx, b"current");
        let before = pointer(&store);
        assert_eq!(
            store
                .open_retained(&cx, old.id(), old.manifest_sha256())
                .unwrap(),
            old
        );
        assert!(store.open_retained(&cx, old.id(), &"0".repeat(64)).is_err());
        for id in ["../outside", "/absolute", "g", "", "g-0000-0000-0000"] {
            assert!(store.open_retained(&cx, id, old.manifest_sha256()).is_err());
        }
        let mut invalid_id = old.id().as_bytes().to_vec();
        invalid_id[2] = b'/';
        assert!(
            store
                .open_retained(
                    &cx,
                    std::str::from_utf8(&invalid_id).unwrap(),
                    old.manifest_sha256()
                )
                .is_err()
        );
        assert_eq!(pointer(&store), before);
        assert_eq!(store.active(&cx).unwrap(), Some(current));
    });
}

#[test]
fn restore_recovers_corrupt_selected_bundle_and_preserves_both_directories() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
        let old = published(&store, &cx, b"known good");
        let current = published(&store, &cx, b"new");
        let mut held = File::open(old.path().join("content.txt")).unwrap();
        fs::write(current.path().join("vector.idx"), b"damaged").unwrap();
        assert!(store.active(&cx).is_err());
        assert!(store.begin(&cx).is_err());
        let plan = store.prepare_restore(&cx, &old).unwrap();
        assert_eq!(plan.target(), &old);
        let before = pointer(&store);
        let outcome = plan
            .restore(&cx, |_, path| {
                assert_eq!(pointer(&store), before);
                assert_eq!(fs::read(path.join("content.txt"))?, b"known good");
                Ok(())
            })
            .unwrap();
        assert_eq!(durable(outcome), old);
        assert_eq!(store.active(&cx).unwrap(), Some(old.clone()));
        assert_eq!(
            fs::read(current.path().join("vector.idx")).unwrap(),
            b"damaged"
        );
        let mut still_pinned = Vec::new();
        held.read_to_end(&mut still_pinned).unwrap();
        assert_eq!(still_pinned, b"known good");
        assert_eq!(
            fs::read_dir(store.root().join(GENERATIONS))
                .unwrap()
                .count(),
            2
        );
        drop(store.begin(&cx).unwrap());
    });
}

#[test]
fn malformed_and_missing_selections_need_explicit_target_not_fallback() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
        let old = published(&store, &cx, b"old");
        let _new = published(&store, &cx, b"new");
        let selected = store.root().join(COMPLETE_GENERATION_POINTER);
        fs::write(&selected, b"private-malformed-selection").unwrap();
        assert!(store.active(&cx).is_err());
        let plan = store.prepare_restore(&cx, &old).unwrap();
        assert!(!format!("{plan:?}").contains("private-malformed-selection"));
        assert_eq!(fs::read(&selected).unwrap(), b"private-malformed-selection");
        assert_eq!(durable(plan.restore(&cx, |_, _| Ok(())).unwrap()), old);
        fs::rename(&selected, store.root().join("retained-pointer-evidence")).unwrap();
        assert!(store.active(&cx).unwrap().is_none());
        let plan = store.prepare_restore(&cx, &old).unwrap();
        assert!(!selected.exists());
        assert_eq!(durable(plan.restore(&cx, |_, _| Ok(())).unwrap()), old);
        assert!(store.root().join("retained-pointer-evidence").is_file());
    });
}

#[test]
fn preparation_releases_lease_and_concurrent_publication_invalidates_it() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
        let old = published(&store, &cx, b"old");
        let _second = published(&store, &cx, b"second");
        let plan = store.prepare_restore(&cx, &old).unwrap();
        // Publishing while the plan is held proves preparation retains no lease.
        let third = published(&store, &cx, b"third");
        let before = pointer(&store);
        let mut validator_called = false;
        assert!(
            plan.restore(&cx, |_, _| {
                validator_called = true;
                Ok(())
            })
            .is_err()
        );
        assert!(!validator_called);
        assert_eq!(pointer(&store), before);
        assert_eq!(store.active(&cx).unwrap(), Some(third));
    });
}

#[test]
fn damaged_target_is_rechecked_after_preparation() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
        let old = published(&store, &cx, b"old");
        let current = published(&store, &cx, b"current");
        let plan = store.prepare_restore(&cx, &old).unwrap();
        let before = pointer(&store);
        fs::write(old.path().join("vector.idx"), b"changed").unwrap();
        let mut validator_called = false;
        assert!(
            plan.restore(&cx, |_, _| {
                validator_called = true;
                Ok(())
            })
            .is_err()
        );
        assert!(!validator_called);
        assert_eq!(pointer(&store), before);
        assert_eq!(store.active(&cx).unwrap(), Some(current));
    });
}

#[test]
fn consumer_rejection_or_cancellation_does_not_switch_selection() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
        let old = published(&store, &cx, b"old");
        let current = published(&store, &cx, b"current");
        let before = pointer(&store);
        let plan = store.prepare_restore(&cx, &old).unwrap();
        assert!(
            plan.restore(&cx, |_, path| {
                Err(invalid(path, "consumer rejected the producer identity"))
            })
            .is_err()
        );
        assert_eq!(pointer(&store), before);
        let plan = store.prepare_restore(&cx, &old).unwrap();
        let error = plan
            .restore(&cx, |cx, _| {
                cx.set_cancel_requested(true);
                Ok(())
            })
            .unwrap_err();
        assert!(matches!(error, SearchError::Cancelled { .. }));
        cx.set_cancel_requested(false);
        assert_eq!(store.active(&cx).unwrap(), Some(current));
        assert_eq!(pointer(&store), before);
        drop(store.begin(&cx).unwrap());
    });
}

#[test]
fn selection_is_rechecked_after_consumer_admission() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
        let old = published(&store, &cx, b"old");
        let _current = published(&store, &cx, b"current");
        let plan = store.prepare_restore(&cx, &old).unwrap();
        assert!(
            plan.restore(&cx, |_, _| {
                // Fault injection: an out-of-protocol descriptor replacement.
                fs::write(
                    store.root().join(COMPLETE_GENERATION_POINTER),
                    b"different evidence",
                )?;
                Ok(())
            })
            .is_err()
        );
        assert_eq!(pointer(&store), b"different evidence");
        assert!(store.active(&cx).is_err());
        assert!(old.path().is_dir());
    });
}

#[test]
fn target_mutation_in_validator_cannot_be_sealed_under_old_digest() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
        let old = published(&store, &cx, b"old");
        let current = published(&store, &cx, b"current");
        let before = pointer(&store);
        let plan = store.prepare_restore(&cx, &old).unwrap();
        assert!(
            plan.restore(&cx, |_, path| {
                fs::write(path.join("vector.idx"), b"invalid validator mutation")?;
                Ok(())
            })
            .is_err()
        );
        assert_eq!(pointer(&store), before);
        assert_eq!(store.active(&cx).unwrap(), Some(current));
    });
}

#[test]
fn foreign_store_and_unsafe_descriptors_are_never_restored() {
    run_test_with_cx(|cx| async move {
        let first_root = tempfile::tempdir().unwrap();
        let second_root = tempfile::tempdir().unwrap();
        let first = CompleteGenerationStore::create(&cx, first_root.path()).unwrap();
        let second = CompleteGenerationStore::create(&cx, second_root.path()).unwrap();
        let target = published(&first, &cx, b"target");
        assert!(second.prepare_restore(&cx, &target).is_err());
        assert!(!second.root().join(COMPLETE_GENERATION_POINTER).exists());
        let selected = first.root().join(COMPLETE_GENERATION_POINTER);
        let saved = first.root().join("saved-pointer");
        fs::rename(&selected, &saved).unwrap();
        std::os::unix::fs::symlink(&saved, &selected).unwrap();
        assert!(first.prepare_restore(&cx, &target).is_err());
        fs::rename(&selected, first.root().join("retained-symlink")).unwrap();
        fs::hard_link(&saved, &selected).unwrap();
        assert!(first.prepare_restore(&cx, &target).is_err());
        fs::rename(&selected, first.root().join("retained-hardlink")).unwrap();
        fs::write(
            &selected,
            vec![b'x'; usize::try_from(MAX_POINTER_BYTES).unwrap() + 1],
        )
        .unwrap();
        assert!(first.prepare_restore(&cx, &target).is_err());
        assert!(
            first
                .root()
                .join("retained-symlink")
                .symlink_metadata()
                .is_ok()
        );
        assert!(first.root().join("retained-hardlink").is_file());
    });
}

#[test]
fn publisher_lease_excludes_flush_preparation_and_restore() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
        let target = published(&store, &cx, b"target");
        let plan = store.prepare_restore(&cx, &target).unwrap();
        let writer = store.begin(&cx).unwrap();
        let before = pointer(&store);
        assert!(store.flush_selected(&cx).is_err());
        assert!(store.prepare_restore(&cx, &target).is_err());
        assert!(plan.restore(&cx, |_, _| Ok(())).is_err());
        assert_eq!(pointer(&store), before);
        drop(writer);
        assert_eq!(durable(store.flush_selected(&cx).unwrap()), target);
    });
}

#[test]
fn post_rename_sync_failure_remains_visible_and_flush_can_confirm_it() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
        let target = published(&store, &cx, b"old");
        let current = published(&store, &cx, b"current");
        let plan = store.prepare_restore(&cx, &target).unwrap();
        let outcome = plan
            .restore_with_sync(
                &cx,
                |_, _| Ok(()),
                |_| {
                    assert_eq!(store.active(&cx).unwrap(), Some(target.clone()));
                    Err(io::Error::other("injected final directory sync failure"))
                },
            )
            .unwrap();
        match outcome {
            GenerationPublication::VisibleButDurabilityUncertain { generation, source } => {
                assert_eq!(generation, target);
                assert_eq!(source.kind(), ErrorKind::Other);
            }
            other @ GenerationPublication::Durable(_) => {
                panic!("expected uncertain durability: {other:?}") // ubs:ignore — test assertion.
            }
        }
        let before = pointer(&store);
        assert_eq!(durable(store.flush_selected(&cx).unwrap()), target);
        assert_eq!(pointer(&store), before);
        assert_eq!(
            fs::read(current.path().join("content.txt")).unwrap(),
            b"current"
        );
        assert_eq!(
            fs::read_dir(store.root().join(GENERATIONS))
                .unwrap()
                .count(),
            2
        );
    });
}

#[test]
fn cancellation_after_rename_does_not_turn_publication_into_abort() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
        let old = published(&store, &cx, b"old");
        let _current = published(&store, &cx, b"current");
        let plan = store.prepare_restore(&cx, &old).unwrap();
        let outcome = plan
            .restore_with_sync(
                &cx,
                |_, _| Ok(()),
                |path| {
                    sync_directory(path)?;
                    cx.set_cancel_requested(true);
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(durable(outcome), old);
        cx.set_cancel_requested(false);
        assert_eq!(store.active(&cx).unwrap(), Some(old));
    });
}

#[test]
fn flush_preserves_generation_bytes_and_never_scans_sources() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
        let target = published(&store, &cx, b"complete");
        let before_pointer = pointer(&store);
        let before_inventory = inventory(&cx, target.path(), false).unwrap();
        let staging = store.root().join(GENERATIONS).join("abandoned-partial");
        fs::create_dir(&staging).unwrap();
        fs::write(staging.join("partial"), b"do not publish").unwrap();
        assert_eq!(durable(store.flush_selected(&cx).unwrap()), target);
        assert_eq!(pointer(&store), before_pointer);
        assert_eq!(
            inventory(&cx, target.path(), false).unwrap(),
            before_inventory
        );
        assert_eq!(
            fs::read(staging.join("partial")).unwrap(),
            b"do not publish"
        );
        assert_eq!(
            fs::read_dir(store.root().join(GENERATIONS))
                .unwrap()
                .count(),
            2
        );
    });
}

#[test]
fn empty_corrupt_and_cancelled_flushes_never_report_success() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
        assert!(store.flush_selected(&cx).is_err());
        assert!(!store.root().join(COMPLETE_GENERATION_POINTER).exists());
        let target = published(&store, &cx, b"complete");
        let before = pointer(&store);
        cx.set_cancel_requested(true);
        assert!(matches!(
            store.flush_selected(&cx),
            Err(SearchError::Cancelled { .. })
        ));
        assert!(matches!(
            store.prepare_restore(&cx, &target),
            Err(SearchError::Cancelled { .. })
        ));
        cx.set_cancel_requested(false);
        fs::write(target.path().join("vector.idx"), b"corrupt").unwrap();
        assert!(store.flush_selected(&cx).is_err());
        assert_eq!(pointer(&store), before);
    });
}

#[test]
fn flush_final_sync_failure_preserves_selection_and_can_be_retried() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
        let target = published(&store, &cx, b"complete");
        let before = pointer(&store);
        let outcome = store
            .flush_with_sync(&cx, |_| {
                Err(io::Error::other("injected flush sync failure"))
            })
            .unwrap();
        assert!(matches!(outcome,
            GenerationPublication::VisibleButDurabilityUncertain { generation, .. }
            if generation == target));
        assert_eq!(pointer(&store), before);
        assert_eq!(durable(store.flush_selected(&cx).unwrap()), target);
    });
}

#[test]
fn repeated_explicit_restore_uses_fresh_descriptor_but_same_immutable_target() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
        let target = published(&store, &cx, b"complete");
        let before = inventory(&cx, target.path(), false).unwrap();
        for _ in 0..3 {
            let plan = store.prepare_restore(&cx, &target).unwrap();
            assert_eq!(durable(plan.restore(&cx, |_, _| Ok(())).unwrap()), target);
        }
        assert_eq!(inventory(&cx, target.path(), false).unwrap(), before);
        assert_eq!(
            fs::read_dir(store.root().join(GENERATIONS))
                .unwrap()
                .count(),
            1
        );
    });
}
