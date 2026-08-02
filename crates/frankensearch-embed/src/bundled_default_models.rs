//! Bundled default semantic model materialization.
//!
//! When the `bundled-default-models` feature is enabled, build-time generated
//! assets are embedded directly into the binary. This module materializes those
//! assets into the configured model cache so normal auto-detection can load
//! semantic embedders without runtime downloads.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use frankensearch_core::error::{SearchError, SearchResult};

use crate::model_manifest::{ModelManifest, is_verification_cached, verify_dir_and_record};
use crate::model_registry::ensure_model_storage_layout_checked;

include!(concat!(
    env!("OUT_DIR"),
    "/bundled_default_models_generated.rs"
));

const MATERIALIZATION_LOCK_FILE: &str = ".bundled-default-models.lock";
static NEXT_TEMP_FILE_ID: AtomicU64 = AtomicU64::new(0);

/// Summary of bundled-model installation activity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddedModelInstallSummary {
    /// Effective model root where bundled assets were installed.
    pub model_root: PathBuf,
    /// Number of model bundles written or repaired.
    pub models_written: usize,
    /// Total bytes written during this call.
    pub bytes_written: u64,
}

/// Materialize bundled default semantic models into the model cache.
///
/// # Errors
///
/// Returns `SearchError` when writing/verification fails.
pub fn ensure_default_semantic_models(
    model_root: Option<&Path>,
) -> SearchResult<EmbeddedModelInstallSummary> {
    let root = resolve_install_root(model_root)?;
    fs::create_dir_all(&root)?;

    with_materialization_lock(&root, || materialize_default_semantic_models(&root))
}

fn with_materialization_lock<T>(
    root: &Path,
    operation: impl FnOnce() -> SearchResult<T>,
) -> SearchResult<T> {
    let lock_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(root.join(MATERIALIZATION_LOCK_FILE))?;
    lock_file.lock()?;

    let operation_result = operation();
    let unlock_result = lock_file.unlock();
    match unlock_result {
        Ok(()) => operation_result,
        Err(error) => Err(error.into()),
    }
}

fn materialize_default_semantic_models(root: &Path) -> SearchResult<EmbeddedModelInstallSummary> {
    let manifests = [ModelManifest::potion_128m(), ModelManifest::minilm_v2()];
    let mut models_written = 0_usize;
    let mut bytes_written = 0_u64;

    for manifest in manifests {
        let install_dir =
            install_dir_for_manifest(&manifest.id).ok_or_else(|| SearchError::InvalidConfig {
                field: "bundled_default_models.manifest_id".to_owned(),
                value: manifest.id.clone(),
                reason: "unsupported bundled manifest id".to_owned(),
            })?;
        let model_dir = root.join(install_dir);

        if bundled_receipt_allows_materialization_skip(&manifest, &model_dir) {
            continue;
        }
        if verify_and_record_materialized_model(&manifest, &model_dir).is_ok() {
            continue;
        }

        let mut wrote_any_file = false;
        for file in &manifest.files {
            let entry = embedded_file_entry(&manifest.id, &file.name).ok_or_else(|| {
                SearchError::InvalidConfig {
                    field: "bundled_default_models.file".to_owned(),
                    value: format!("{}:{}", manifest.id, file.name),
                    reason: "embedded file missing for manifest".to_owned(),
                }
            })?;

            if entry.size != file.size || !entry.sha256.eq_ignore_ascii_case(&file.sha256) {
                return Err(SearchError::InvalidConfig {
                    field: "bundled_default_models.generated".to_owned(),
                    value: format!("{}:{}", manifest.id, file.name),
                    reason: "embedded metadata mismatch against manifest".to_owned(),
                });
            }

            let destination = model_dir.join(&file.name);
            if let Some(parent) = destination.parent() {
                fs::create_dir_all(parent)?;
            }

            let destination_len = fs::metadata(&destination).ok().map(|meta| meta.len());
            if destination_len == Some(entry.size)
                && crate::model_manifest::verify_file_sha256(&destination, &file.sha256, file.size)
                    .is_ok()
            {
                continue;
            }

            write_atomic_file(&destination, entry.bytes)?;
            bytes_written = bytes_written.saturating_add(entry.size);
            wrote_any_file = true;
        }

        verify_and_record_materialized_model(&manifest, &model_dir)?;
        if wrote_any_file {
            models_written = models_written.saturating_add(1);
        }
    }

    Ok(EmbeddedModelInstallSummary {
        model_root: root.to_path_buf(),
        models_written,
        bytes_written,
    })
}

fn bundled_receipt_allows_materialization_skip(manifest: &ModelManifest, model_dir: &Path) -> bool {
    is_verification_cached(manifest, model_dir)
}

/// Admit a materialized bundle only after full size-and-SHA verification.
///
/// Both the pre-existing-cache promotion path and the post-write promotion path route
/// through this boundary. A lightweight receipt is therefore evidence of a successful
/// full verification, never a substitute for the verification that creates it.
fn verify_and_record_materialized_model(
    manifest: &ModelManifest,
    model_dir: &Path,
) -> SearchResult<()> {
    verify_dir_and_record(manifest, model_dir)
}

fn resolve_install_root(model_root: Option<&Path>) -> SearchResult<PathBuf> {
    if let Some(path) = model_root {
        if let Some(name) = path.file_name().and_then(|name| name.to_str())
            && (name.eq_ignore_ascii_case("potion-multilingual-128M")
                || name.eq_ignore_ascii_case("all-MiniLM-L6-v2"))
            && let Some(parent) = path.parent()
        {
            return Ok(parent.to_path_buf());
        }
        return Ok(path.to_path_buf());
    }
    ensure_model_storage_layout_checked()
}

fn install_dir_for_manifest(manifest_id: &str) -> Option<&'static str> {
    match manifest_id {
        "potion-multilingual-128m" => Some("potion-multilingual-128M"),
        "all-minilm-l6-v2" => Some("all-MiniLM-L6-v2"),
        _ => None,
    }
}

fn embedded_file_entry(
    manifest_id: &str,
    relative_path: &str,
) -> Option<&'static EmbeddedModelFile> {
    EMBEDDED_MODEL_FILES
        .iter()
        .find(|entry| entry.manifest_id == manifest_id && entry.relative_path == relative_path)
}

fn write_atomic_file(path: &Path, bytes: &[u8]) -> SearchResult<()> {
    let (tmp_path, mut file) = reserve_temp_file(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);

    // On POSIX, rename() atomically replaces the destination — no need
    // to remove first. The uniquely reserved temporary file is retained
    // when an earlier step fails so the failure remains inspectable.
    fs::rename(&tmp_path, path)?;
    Ok(())
}

fn reserve_temp_file(path: &Path) -> SearchResult<(PathBuf, File)> {
    loop {
        let id = NEXT_TEMP_FILE_ID.fetch_add(1, Ordering::Relaxed);
        let tmp_path = path.with_extension(format!("tmp.{}.{id}", std::process::id()));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp_path)
        {
            Ok(file) => return Ok((tmp_path, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::Duration;

    use super::*;
    use crate::model_manifest::{ModelFile, verify_dir_and_record};

    fn reserve_materialization_test_root() -> std::io::Result<PathBuf> {
        loop {
            let id = NEXT_TEMP_FILE_ID.fetch_add(1, Ordering::Relaxed);
            let candidate = std::env::temp_dir().join(format!(
                "frankensearch_bundled_materialization_lock_{}_{id}",
                std::process::id()
            ));
            match fs::create_dir(&candidate) {
                Ok(()) => return Ok(candidate),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error),
            }
        }
    }

    #[test]
    fn materialization_lock_serializes_same_root_transactions() {
        let root = reserve_materialization_test_root()
            .expect("materialization test root should be atomically reservable");
        let entrants = 4;
        let start = Arc::new(Barrier::new(entrants + 1));
        let active = Arc::new(AtomicUsize::new(0));
        let maximum_active = Arc::new(AtomicUsize::new(0));

        let threads = (0..entrants)
            .map(|_| {
                let root = root.clone();
                let start = Arc::clone(&start);
                let active = Arc::clone(&active);
                let maximum_active = Arc::clone(&maximum_active);
                thread::spawn(move || {
                    start.wait();
                    with_materialization_lock(&root, || {
                        let now_active = active.fetch_add(1, Ordering::SeqCst) + 1;
                        maximum_active.fetch_max(now_active, Ordering::SeqCst);
                        thread::sleep(Duration::from_millis(10));
                        active.fetch_sub(1, Ordering::SeqCst);
                        Ok(())
                    })
                    .unwrap();
                })
            })
            .collect::<Vec<_>>();

        start.wait();
        for thread in threads {
            thread.join().unwrap();
        }

        assert_eq!(maximum_active.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn bundled_entries_cover_default_manifest_files() {
        let manifests = [ModelManifest::potion_128m(), ModelManifest::minilm_v2()];
        for manifest in manifests {
            for file in manifest.files {
                let entry = embedded_file_entry(&manifest.id, &file.name)
                    .expect("embedded entry should exist for every bundled manifest file");
                assert_eq!(entry.size, file.size);
                assert_eq!(entry.sha256, file.sha256);
            }
        }
    }

    #[test]
    fn bundled_manifest_ids_map_to_install_dirs() {
        assert_eq!(
            install_dir_for_manifest("potion-multilingual-128m"),
            Some("potion-multilingual-128M")
        );
        assert_eq!(
            install_dir_for_manifest("all-minilm-l6-v2"),
            Some("all-MiniLM-L6-v2")
        );
        assert_eq!(install_dir_for_manifest("unknown"), None);
    }

    #[test]
    fn bundled_materialization_rejects_same_id_stale_manifest_receipt() {
        let tmp = tempfile::tempdir().unwrap();
        let model_dir = tmp.path().join("model");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("model.bin"), b"abc").unwrap();
        let manifest = ModelManifest {
            id: "bundled-receipt-test".to_owned(),
            version: "v1".to_owned(),
            display_name: None,
            description: None,
            repo: "test/bundled".to_owned(),
            revision: "a".repeat(40),
            files: vec![ModelFile {
                name: "model.bin".to_owned(),
                sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
                    .to_owned(),
                size: 3,
                url: None,
            }],
            license: "MIT".to_owned(),
            dimension: Some(1),
            tier: None,
            download_size_bytes: 3,
        };
        verify_dir_and_record(&manifest, &model_dir).unwrap();
        assert!(bundled_receipt_allows_materialization_skip(
            &manifest, &model_dir
        ));

        let mut evolved = manifest;
        evolved.revision = "b".repeat(40);
        assert!(
            !bundled_receipt_allows_materialization_skip(&evolved, &model_dir),
            "bundled materialization must not skip on a same-ID stale receipt"
        );
    }

    #[test]
    fn bundled_promotion_full_sha_verifies_before_writing_receipt() {
        let tmp = tempfile::tempdir().unwrap();
        let model_dir = tmp.path().join("model");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("model.bin"), b"abd").unwrap();
        let manifest = ModelManifest {
            id: "bundled-promotion-test".to_owned(),
            version: "v1".to_owned(),
            display_name: None,
            description: None,
            repo: "test/bundled".to_owned(),
            revision: "a".repeat(40),
            files: vec![ModelFile {
                name: "model.bin".to_owned(),
                sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
                    .to_owned(),
                size: 3,
                url: None,
            }],
            license: "MIT".to_owned(),
            dimension: Some(1),
            tier: None,
            download_size_bytes: 3,
        };

        let error = verify_and_record_materialized_model(&manifest, &model_dir).unwrap_err();
        assert!(
            matches!(error, SearchError::HashMismatch { .. }),
            "same-size corrupt bytes must fail full-SHA bundled promotion: {error}"
        );
        assert!(
            !model_dir.join(".verified").exists(),
            "failed full-SHA promotion must not create a receipt"
        );

        fs::write(model_dir.join("model.bin"), b"abc").unwrap();
        verify_and_record_materialized_model(&manifest, &model_dir).unwrap();
        assert!(
            bundled_receipt_allows_materialization_skip(&manifest, &model_dir),
            "successful full-SHA promotion must create a current receipt"
        );
    }
}
