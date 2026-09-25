use super::*;
use asupersync::test_utils::run_test_with_cx;
use frankensearch_core::generation::{
    ArtifactGenerationIdentityV1, EmbeddingIdentityBundleV1, QuantizationFormat,
};
use frankensearch_index::{Quantization, VectorIndex};
use std::fs;

use crate::cache::SentinelFileDetector;

fn binding(name: &str, dimension: u32, sequence: u64, nonce: u8) -> FsviV2IdentityBinding {
    let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(name, dimension);
    "fsvi-v2".clone_into(&mut identity.storage.format);
    "little-endian".clone_into(&mut identity.storage.endianness);
    identity.storage.quantization = QuantizationFormat::F32;
    FsviV2IdentityBinding::new(
        ArtifactGenerationIdentityV1::new(sequence, [nonce; 16]).unwrap(),
        identity.freeze().unwrap(),
    )
    .unwrap()
}

fn write(path: &Path, binding: &FsviV2IdentityBinding, ids: &[&str], coordinate: usize) {
    // Replace only this test's own pathname. All previously admitted owners
    // continue serving their independent immutable images.
    let staging = path.with_extension("staging");
    let mut writer = VectorIndex::create_v2(&staging, binding.clone()).unwrap();
    for id in ids {
        let mut values = vec![0.0; binding.dimension()];
        values[coordinate] = 1.0;
        writer.write_record(id, &values).unwrap();
    }
    writer.finish().unwrap();
    fs::rename(staging, path).unwrap();
}

fn cache(
    cx: &Cx,
    directory: &Path,
    fast: &FsviV2IdentityBinding,
    quality: Option<&FsviV2IdentityBinding>,
) -> IndexCache {
    let mut paths = TwoTierIndexPaths::new(directory.join("fast.fsvi"));
    if quality.is_some() {
        paths = paths.with_quality_index(directory.join("quality.fsvi"));
    }
    IndexCache::open_admitted_v2_with_paths(
        cx,
        paths,
        directory,
        TwoTierConfig::default(),
        fast,
        quality,
        Box::new(SentinelFileDetector::new()),
    )
    .unwrap()
}

fn open(
    cache: &IndexCache,
    fast: &FsviV2IdentityBinding,
    quality: Option<&FsviV2IdentityBinding>,
) -> TwoTierIndex {
    open_exact(
        cache.index_paths().unwrap(),
        cache.config.clone(),
        fast,
        quality,
    )
    .unwrap()
}

#[test]
fn exact_cache_opens_both_tiers_and_reloads_the_retained_bindings() {
    run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let fast = binding("fast", 2, 5, 7);
        let quality = binding("quality", 3, 5, 7);
        write(&dir.path().join("fast.fsvi"), &fast, &["a", "b"], 0);
        write(&dir.path().join("quality.fsvi"), &quality, &["a", "b"], 1);
        let before_fast = fs::read(dir.path().join("fast.fsvi")).unwrap();
        let before_quality = fs::read(dir.path().join("quality.fsvi")).unwrap();
        let cache = cache(&cx, dir.path(), &fast, Some(&quality));
        let retained = cache.current();
        assert_eq!(retained.fast_admitted_binding(), Some(&fast));
        assert_eq!(retained.quality_admitted_binding(), Some(&quality));
        assert_eq!(retained.doc_count(), 2);
        cache.reload().unwrap();
        let reloaded = cache.current();
        assert!(!Arc::ptr_eq(&retained, &reloaded));
        assert_eq!(
            retained.fast_admitted_owner().unwrap().witness(),
            reloaded.fast_admitted_owner().unwrap().witness()
        );
        assert_eq!(
            retained.quality_admitted_owner().unwrap().witness(),
            reloaded.quality_admitted_owner().unwrap().witness()
        );
        assert_eq!(fs::read(dir.path().join("fast.fsvi")).unwrap(), before_fast);
        assert_eq!(
            fs::read(dir.path().join("quality.fsvi")).unwrap(),
            before_quality
        );
        assert!(!dir.path().join(super::super::SENTINEL_FILENAME).exists());
        assert!(!dir.path().join("vector.fast.idx").exists());
    });
}

#[test]
fn selected_successor_updates_reload_contract_without_retargeting_old_readers() {
    run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let first = binding("fast", 2, 5, 7);
        let next = binding("fast", 2, 6, 8);
        let path = dir.path().join("fast.fsvi");
        write(&path, &first, &["old"], 0);
        let cache = cache(&cx, dir.path(), &first, None);
        let old = cache.current();
        write(&path, &next, &["new", "second"], 1);
        assert!(
            cache.reload().is_err(),
            "ordinary reload must not discover new bindings"
        );
        assert!(Arc::ptr_eq(&old, &cache.current()));
        assert!(
            cache
                .reload_admitted_v2_if_current(&cx, &old, &next, None)
                .unwrap()
        );
        assert_eq!(cache.current().fast_admitted_binding(), Some(&next));
        cache.reload().unwrap();
        assert_eq!(cache.current().fast_admitted_binding(), Some(&next));
        assert_eq!(cache.current().doc_count(), 2);
        let old_owner = old.fast_admitted_owner().unwrap();
        assert_eq!(old_owner.doc_id_at(0).unwrap(), "old");
        assert_eq!(old_owner.vector_at_f32(0).unwrap(), [1.0, 0.0]);
        assert_eq!(
            cache
                .current()
                .fast_admitted_owner()
                .unwrap()
                .vector_at_f32(0)
                .unwrap(),
            [0.0, 1.0]
        );
    });
}

#[test]
fn every_replace_entry_rejects_same_generation_content_or_coverage_changes() {
    run_test_with_cx(|cx| async move {
        for quality_changed in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let fast = binding("fast", 2, 5, 7);
            let quality = binding("quality", 3, 5, 7);
            write(&dir.path().join("fast.fsvi"), &fast, &["a", "b"], 0);
            // Partial quality coverage is supported, not invented as full.
            write(&dir.path().join("quality.fsvi"), &quality, &["a"], 1);
            let cache = cache(&cx, dir.path(), &fast, Some(&quality));
            let retained = cache.current();
            if quality_changed {
                write(&dir.path().join("quality.fsvi"), &quality, &["a", "b"], 1);
            } else {
                write(&dir.path().join("fast.fsvi"), &fast, &["a", "b"], 1);
            }
            let bytes = fs::read(dir.path().join("fast.fsvi")).unwrap();
            assert!(cache.replace(open(&cache, &fast, Some(&quality))).is_err());
            assert!(
                cache
                    .replace_if_current(&retained, open(&cache, &fast, Some(&quality)))
                    .is_err()
            );
            assert!(
                cache
                    .replace_admitted_v2_if_current(
                        &cx,
                        &retained,
                        open(&cache, &fast, Some(&quality)),
                    )
                    .is_err()
            );
            assert!(cache.reload().is_err());
            assert!(
                cache
                    .reload_admitted_v2_if_current(&cx, &retained, &fast, Some(&quality))
                    .is_err()
            );
            assert!(Arc::ptr_eq(&retained, &cache.current()));
            assert_eq!(fs::read(dir.path().join("fast.fsvi")).unwrap(), bytes);
        }
    });
}

#[test]
fn full_contract_drift_and_generation_regression_are_refused() {
    run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let original = binding("fast", 2, 5, 7);
        let path = dir.path().join("fast.fsvi");
        write(&path, &original, &["a"], 0);
        let cache = cache(&cx, dir.path(), &original, None);
        let expected = cache.current();
        for field in ["space", "producer", "input", "storage", "older", "same-sequence"] {
            let mut identity = original.frozen_identity().identity.clone();
            match field {
                "space" => identity.space.logical_model_id.push_str("-other"),
                "producer" => identity.producer.implementation_revision.push_str("-other"),
                "input" => identity.input.doc_id_semantics.push_str("-other"),
                "storage" => identity.storage.quantization = QuantizationFormat::F16,
                _ => {}
            }
            let sequence = match field {
                "older" => 4,
                "same-sequence" => 5,
                _ => 6,
            };
            let candidate_binding = FsviV2IdentityBinding::new(
                ArtifactGenerationIdentityV1::new(sequence, [8; 16]).unwrap(),
                identity.freeze().unwrap(),
            )
            .unwrap();
            write(&path, &candidate_binding, &["a"], 1);
            let before = fs::read(&path).unwrap();
            assert!(
                cache.replace(open(&cache, &candidate_binding, None)).is_err(),
                "{field}"
            );
            assert!(
                cache
                    .reload_admitted_v2_if_current(&cx, &expected, &candidate_binding, None)
                    .is_err(),
                "{field}"
            );
            assert!(Arc::ptr_eq(&expected, &cache.current()));
            assert_eq!(fs::read(&path).unwrap(), before);
        }
    });
}

#[test]
fn required_quality_and_full_width_generation_are_checked_before_open() {
    run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let paths = TwoTierIndexPaths::new(dir.path().join("not-created-fast.fsvi"))
            .with_quality_index(dir.path().join("not-created-quality.fsvi"));
        let fast = binding("fast", 2, 5, 7);
        let wrong_sequence = binding("quality", 3, 6, 7);
        let wrong_nonce = binding("quality", 3, 5, 8);
        for quality in [None, Some(&wrong_sequence), Some(&wrong_nonce)] {
            let result = IndexCache::open_admitted_v2_with_paths(
                &cx,
                paths.clone(),
                dir.path(),
                TwoTierConfig::default(),
                &fast,
                quality,
                Box::new(SentinelFileDetector::new()),
            );
            assert!(matches!(result, Err(SearchError::InvalidConfig { .. })));
            assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 0);
        }
        write(paths.fast_index(), &fast, &["a"], 0);
        let quality = binding("quality", 3, 5, 7);
        assert!(
            IndexCache::open_admitted_v2_with_paths(
                &cx,
                paths,
                dir.path(),
                TwoTierConfig::default(),
                &fast,
                Some(&quality),
                Box::new(SentinelFileDetector::new()),
            )
            .is_err(),
            "missing required quality must not degrade"
        );
    });
}

#[test]
fn a_v2_cache_cannot_be_downgraded_to_legacy_or_foreign_paths() {
    run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let original = binding("fast", 2, 5, 7);
        let path = dir.path().join("fast.fsvi");
        write(&path, &original, &["a"], 0);
        let cache = cache(&cx, dir.path(), &original, None);
        let retained = cache.current();
        let staged = dir.path().join("legacy.fsvi");
        let mut writer =
            VectorIndex::create_with_revision(&staged, "fast", "v1", 2, Quantization::F32)
                .unwrap();
        writer.write_record("a", &[1.0, 0.0]).unwrap();
        writer.finish().unwrap();
        fs::rename(staged, &path).unwrap();
        let legacy =
            TwoTierIndex::open_with_paths(cache.index_paths().unwrap(), TwoTierConfig::default())
                .unwrap();
        assert!(cache.replace(legacy).is_err());
        assert!(cache.reload().is_err());
        assert!(Arc::ptr_eq(&retained, &cache.current()));
        let other = dir.path().join("foreign.fsvi");
        write(&other, &original, &["a"], 0);
        let foreign = TwoTierIndex::open_admitted_v2_with_paths(
            &TwoTierIndexPaths::new(other),
            TwoTierConfig::default(),
            &original,
            None,
        )
        .unwrap();
        assert!(
            cache
                .replace_admitted_v2_if_current(&cx, &retained, foreign)
                .is_err()
        );
        assert!(Arc::ptr_eq(&retained, &cache.current()));
    });
}
