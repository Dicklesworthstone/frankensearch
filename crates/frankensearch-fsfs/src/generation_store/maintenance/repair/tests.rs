use super::*;
use asupersync::test_utils::run_test_with_cx;
use std::cell::Cell;
use std::os::unix::fs::{MetadataExt, symlink};

use super::super::super::{COMPLETE_GENERATION_POINTER, GenerationPublication, PublishedGeneration};

const COPY_BUDGET: u64 = 8 * 1024 * 1024;

fn vector_bytes(seed: u8) -> Vec<u8> {
    (0_u8..=255).cycle().take(8192).map(|value| value.wrapping_add(seed)).collect()
}

// These are real-codec/container tests, not semantic engine admission. The
// publisher's consumer callback is intentionally only an inventory fixture.
fn publish(store: &CompleteGenerationStore, cx: &Cx, seed: u8) -> PublishedGeneration {
    let build = store.begin(cx).unwrap();
    fs::create_dir(build.path().join("vector")).unwrap();
    fs::create_dir(build.path().join("lexical")).unwrap();
    fs::write(build.path().join("catalog.sqlite"), b"retained catalog fixture").unwrap();
    fs::write(build.path().join("lexical/segment"), b"retained lexical fixture").unwrap();
    let protector = FileProtector::new(Arc::new(DefaultSymbolCodec), DurabilityConfig::default()).unwrap();
    for (number, relative) in VECTOR_PATHS.iter().enumerate() {
        let path = build.path().join(relative);
        fs::write(&path, vector_bytes(seed.wrapping_add(u8::try_from(number).unwrap()))).unwrap();
        protector.protect_file(&path).unwrap();
    }
    match build.publish(cx, |_, _| Ok(())).unwrap() {
        GenerationPublication::Durable(generation) => generation,
        other => panic!("fixture was not durable: {other:?}"), // ubs:ignore — test assertion.
    }
}

fn corrupt(path: &Path) -> Vec<u8> {
    let mut bytes = fs::read(path).unwrap();
    bytes[31] ^= 0x5a;
    fs::write(path, &bytes).unwrap();
    bytes
}

fn count(store: &CompleteGenerationStore) -> usize {
    fs::read_dir(store.root.join(GENERATIONS)).unwrap().count()
}

#[test]
fn real_codec_repairs_both_tiers_in_new_inodes_without_changing_selection_or_sources() {
    run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let store = CompleteGenerationStore::create(&cx, directory.path()).unwrap();
        let original = publish(&store, &cx, 3);
        let original_files = fresh_inventory(&cx, original.path(), COPY_BUDGET, true).unwrap();
        let pointer = fs::read(store.root.join(COMPLETE_GENERATION_POINTER)).unwrap();
        let fast = original.path().join(VECTOR_PATHS[0]);
        let quality = original.path().join(VECTOR_PATHS[1]);
        let damaged_fast = corrupt(&fast);
        OpenOptions::new().write(true).open(&quality).unwrap().set_len(100).unwrap();
        let damaged_quality = fs::read(&quality).unwrap();
        let plan = store.prepare_vector_repair(&cx, original.id(), original.manifest_sha256(), COPY_BUDGET).unwrap();
        assert_ne!(plan.target().id(), original.id());
        assert_eq!(fresh_inventory(&cx, plan.target().path(), COPY_BUDGET, true).unwrap(), original_files);
        assert_eq!(fs::read(&fast).unwrap(), damaged_fast);
        assert_eq!(fs::read(&quality).unwrap(), damaged_quality);
        assert_eq!(fs::read(store.root.join(COMPLETE_GENERATION_POINTER)).unwrap(), pointer);
        for relative in original_files.keys() {
            assert_ne!(
                fs::metadata(original.path().join(relative)).unwrap().ino(),
                fs::metadata(plan.target().path().join(relative)).unwrap().ino(),
                "the repaired cohort cannot share a mutable inode: {relative}"
            );
        }
        assert_eq!(count(&store), 2);
        let target = plan.target().clone();
        assert!(matches!(plan.restore(&cx, |_, _| Ok(())).unwrap(),
            GenerationPublication::Durable(ref selected) if selected == &target));
        assert_eq!(store.active(&cx).unwrap(), Some(target));
        assert_eq!(fs::read(fast).unwrap(), damaged_fast);
        assert_eq!(fs::read(quality).unwrap(), damaged_quality);
    });
}

#[test]
fn a_missing_vector_is_reconstructed_only_from_its_receipted_sidecar() {
    run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let store = CompleteGenerationStore::create(&cx, directory.path()).unwrap();
        let original = publish(&store, &cx, 8);
        let path = original.path().join(VECTOR_PATHS[1]);
        let preserved = directory.path().join("preserved-quality.fsvi");
        fs::rename(&path, &preserved).unwrap();
        let expected = fs::read(&preserved).unwrap();
        let before = fs::read(store.root.join(COMPLETE_GENERATION_POINTER)).unwrap();
        let plan = store.prepare_vector_repair(&cx, original.id(), original.manifest_sha256(), COPY_BUDGET).unwrap();
        assert_eq!(fs::read(plan.target().path().join(VECTOR_PATHS[1])).unwrap(), expected);
        assert!(!path.exists(), "the original source pathname is never repaired in place");
        assert_eq!(fs::read(preserved).unwrap(), expected);
        assert_eq!(fs::read(store.root.join(COMPLETE_GENERATION_POINTER)).unwrap(), before);
    });
}

#[test]
fn catalog_lexical_sidecar_and_unreceipted_damage_refuse_before_allocating_a_candidate() {
    run_test_with_cx(|cx| async move {
        for relative in ["catalog.sqlite", "lexical/segment", "vector/index.fsvi.fec", "unexpected.fsvi"] {
            let directory = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, directory.path()).unwrap();
            let original = publish(&store, &cx, 4);
            let damaged = corrupt(&original.path().join(VECTOR_PATHS[0]));
            let path = original.path().join(relative);
            fs::write(&path, b"not the receipted artifact").unwrap();
            let before = fs::read(store.root.join(COMPLETE_GENERATION_POINTER)).unwrap();
            assert!(store.prepare_vector_repair(&cx, original.id(), original.manifest_sha256(), COPY_BUDGET).is_err(), "{relative}");
            assert_eq!(count(&store), 1, "{relative}");
            assert_eq!(fs::read(&path).unwrap(), b"not the receipted artifact");
            assert_eq!(fs::read(original.path().join(VECTOR_PATHS[0])).unwrap(), damaged);
            assert_eq!(fs::read(store.root.join(COMPLETE_GENERATION_POINTER)).unwrap(), before);
        }
    });
}

#[test]
fn receipt_limits_links_cancellation_and_contention_do_not_allocate_or_authorize_repair() {
    run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let store = CompleteGenerationStore::create(&cx, directory.path()).unwrap();
        let original = publish(&store, &cx, 5);
        assert!(store.prepare_vector_repair(&cx, original.id(), original.manifest_sha256(), COPY_BUDGET).is_err(), "healthy targets do not need copying");
        corrupt(&original.path().join(VECTOR_PATHS[0]));
        for budget in [0, 1] {
            assert!(store.prepare_vector_repair(&cx, original.id(), original.manifest_sha256(), budget).is_err());
        }
        assert!(store.prepare_vector_repair(&cx, "../escape", original.manifest_sha256(), COPY_BUDGET).is_err());
        assert!(store.prepare_vector_repair(&cx, original.id(), &"0".repeat(64), COPY_BUDGET).is_err());
        cx.set_cancel_requested(true);
        let cancelled = store.prepare_vector_repair(&cx, original.id(), original.manifest_sha256(), COPY_BUDGET);
        cx.set_cancel_requested(false);
        assert!(matches!(cancelled, Err(SearchError::Cancelled { .. })));
        let lease = PublicationLease::acquire(&store.root).unwrap();
        assert!(store.prepare_vector_repair(&cx, original.id(), original.manifest_sha256(), COPY_BUDGET).is_err());
        drop(lease);
        let sidecar = original.path().join("vector/index.fsvi.fec");
        let retained = directory.path().join("preserved.fec");
        fs::rename(&sidecar, &retained).unwrap();
        symlink(&retained, &sidecar).unwrap();
        assert!(store.prepare_vector_repair(&cx, original.id(), original.manifest_sha256(), COPY_BUDGET).is_err());
        assert_eq!(count(&store), 1);
        assert!(fs::symlink_metadata(sidecar).unwrap().is_symlink());
    });
}

#[test]
fn stale_repair_cannot_overwrite_a_publication_that_won_during_model_admission() {
    run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let store = CompleteGenerationStore::create(&cx, directory.path()).unwrap();
        let original = publish(&store, &cx, 1);
        let _selected = publish(&store, &cx, 2);
        corrupt(&original.path().join(VECTOR_PATHS[0]));
        let plan = store.prepare_vector_repair(&cx, original.id(), original.manifest_sha256(), COPY_BUDGET).unwrap();
        let repaired = plan.target().clone();
        let concurrent = publish(&store, &cx, 6);
        let called = Cell::new(false);
        assert!(plan.restore(&cx, |_, _| { called.set(true); Ok(()) }).is_err());
        assert!(!called.get(), "selection comparison precedes consumer validation");
        assert_eq!(store.active(&cx).unwrap(), Some(concurrent));
        assert_eq!(fs::read(repaired.path().join(VECTOR_PATHS[0])).unwrap(), vector_bytes(1));
    });
}

#[test]
fn repair_handles_malformed_selection_without_overwriting_it_before_consumer_admission() {
    run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let store = CompleteGenerationStore::create(&cx, directory.path()).unwrap();
        let original = publish(&store, &cx, 10);
        let damaged = corrupt(&original.path().join(VECTOR_PATHS[0]));
        let pointer = store.root.join(COMPLETE_GENERATION_POINTER);
        fs::write(&pointer, b"damaged selection evidence").unwrap();
        for cancel in [false, true] {
            let plan = store.prepare_vector_repair(&cx, original.id(), original.manifest_sha256(), COPY_BUDGET).unwrap();
            let candidate = plan.target().clone();
            let result = plan.restore(&cx, |cx, _| {
                if cancel {
                    cx.set_cancel_requested(true);
                    Ok(())
                } else {
                    Err(invalid(&pointer, "consumer identity refused"))
                }
            });
            cx.set_cancel_requested(false);
            assert!(result.is_err());
            assert_eq!(fs::read(&pointer).unwrap(), b"damaged selection evidence");
            assert_eq!(fs::read(original.path().join(VECTOR_PATHS[0])).unwrap(), damaged);
            assert!(candidate.path().join(COMPLETE_GENERATION_MANIFEST).exists());
        }
        let plan = store.prepare_vector_repair(&cx, original.id(), original.manifest_sha256(), COPY_BUDGET).unwrap();
        let target = plan.target().clone();
        let result = plan.restore_with_sync(&cx, |_, _| Ok(()), |_| {
            cx.set_cancel_requested(true);
            Err(std::io::Error::other("injected final synchronization failure"))
        }).unwrap();
        cx.set_cancel_requested(false);
        assert!(matches!(result, GenerationPublication::VisibleButDurabilityUncertain { ref generation, .. } if generation == &target));
        assert_eq!(store.active(&cx).unwrap(), Some(target));
        assert_eq!(fs::read(original.path().join(VECTOR_PATHS[0])).unwrap(), damaged);
    });
}

#[test]
fn repair_reads_actual_media_bytes_even_when_the_normal_digest_memo_would_accept_them() {
    run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let store = CompleteGenerationStore::create(&cx, directory.path()).unwrap();
        let original = publish(&store, &cx, 7);
        let path = original.path().join(VECTOR_PATHS[0]);
        let healthy = fs::read(&path).unwrap();
        let damaged = corrupt(&path);
        crate::generation_store::remember_digest_for_test(&path, &healthy);
        assert!(store.active(&cx).is_ok(), "control must actually exercise memo reuse");
        let plan = store.prepare_vector_repair(&cx, original.id(), original.manifest_sha256(), COPY_BUDGET).unwrap();
        assert_eq!(fs::read(plan.target().path().join(VECTOR_PATHS[0])).unwrap(), healthy);
        assert_eq!(fs::read(path).unwrap(), damaged);
    });
}

#[test]
fn a_receipted_but_mismatched_sidecar_cannot_define_recovery_success() {
    run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let store = CompleteGenerationStore::create(&cx, directory.path()).unwrap();
        let build = store.begin(&cx).unwrap();
        fs::create_dir(build.path().join("vector")).unwrap();
        let path = build.path().join(VECTOR_PATHS[0]);
        fs::write(&path, vector_bytes(1)).unwrap();
        let protector = FileProtector::new(Arc::new(DefaultSymbolCodec), DurabilityConfig::default()).unwrap();
        protector.protect_file(&path).unwrap();
        // The generic fixture deliberately seals a sidecar for another payload.
        // Only the manifest's SHA-256, not decoding or CRC success, is authority.
        fs::write(&path, vector_bytes(2)).unwrap();
        let original = match build.publish(&cx, |_, _| Ok(())).unwrap() {
            GenerationPublication::Durable(value) => value,
            other => panic!("fixture: {other:?}"), // ubs:ignore — test assertion.
        };
        corrupt(&path);
        let before = fs::read(store.root.join(COMPLETE_GENERATION_POINTER)).unwrap();
        assert!(store.prepare_vector_repair(&cx, original.id(), original.manifest_sha256(), COPY_BUDGET).is_err());
        assert_eq!(fs::read(store.root.join(COMPLETE_GENERATION_POINTER)).unwrap(), before);
        assert_eq!(count(&store), 2, "an unselected partial copy is retained on decode refusal");
        let sealed = fs::read_dir(store.root.join(GENERATIONS)).unwrap()
            .filter(|entry| entry.as_ref().unwrap().path().join(COMPLETE_GENERATION_MANIFEST).exists())
            .count();
        assert_eq!(sealed, 1, "failed reconstruction must not receive a completion seal");
    });
}

#[test]
fn repair_manifest_paths_and_size_arithmetic_fail_closed() {
    let root = Path::new("/not-read");
    for path in ["", "../escape", "/absolute", "a//b", "a/./b", "a\\b", COMPLETE_GENERATION_MANIFEST] {
        let manifest = BundleManifest {
            schema_version: 1,
            generation: "selected".to_owned(),
            files: vec![Artifact { path: path.to_owned(), bytes: 1, sha256: "0".repeat(64) }],
        };
        assert!(validate_manifest(root, &manifest, "selected", u64::MAX).is_err(), "{path}");
    }
    let mut manifest = BundleManifest {
        schema_version: 1,
        generation: "selected".to_owned(),
        files: vec![
            Artifact { path: "a".to_owned(), bytes: u64::MAX, sha256: "0".repeat(64) },
            Artifact { path: "b".to_owned(), bytes: 1, sha256: "0".repeat(64) },
        ],
    };
    assert!(validate_manifest(root, &manifest, "selected", u64::MAX).is_err());
    manifest.files[0].bytes = 1;
    assert!(validate_manifest(root, &manifest, "selected", 2).is_ok());
    manifest.files[1].path = "a/child".to_owned();
    assert!(validate_manifest(root, &manifest, "selected", 2).is_err());
}