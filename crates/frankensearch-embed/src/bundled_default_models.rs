//! Bundled default semantic model materialization.
//!
//! When the `bundled-default-models` feature is enabled, build-time generated
//! assets are embedded directly into the binary. This module materializes those
//! assets into the configured model cache so normal auto-detection can load
//! semantic embedders without runtime downloads.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use frankensearch_core::error::{SearchError, SearchResult};

use crate::model_manifest::{ModelManifest, write_verification_marker};
use crate::model_registry::ensure_model_storage_layout_checked;

include!(concat!(
    env!("OUT_DIR"),
    "/bundled_default_models_generated.rs"
));

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

        if crate::model_manifest::is_verification_cached(&manifest, &model_dir) {
            continue;
        }
        if manifest.verify_dir(&model_dir).is_ok() {
            write_verification_marker(&manifest, &model_dir);
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

        manifest.verify_dir(&model_dir)?;
        write_verification_marker(&manifest, &model_dir);
        if wrote_any_file {
            models_written = models_written.saturating_add(1);
        }
    }

    Ok(EmbeddedModelInstallSummary {
        model_root: root,
        models_written,
        bytes_written,
    })
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
    let pid = std::process::id();
    let tmp_path = path.with_extension(format!("tmp.{pid}"));
    let mut file = File::create(&tmp_path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);

    if path.exists() {
        fs::remove_file(path)?;
    }
    fs::rename(tmp_path, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
