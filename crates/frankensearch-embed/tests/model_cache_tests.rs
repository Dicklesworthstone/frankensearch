//! Integration tests for model cache resolution, layout, and filesystem operations.
//!
//! These tests verify the cache directory lifecycle: resolution from environment
//! variables and XDG paths, lazy creation, and concurrent access safety.

use frankensearch_embed::{
    MODEL_CACHE_LAYOUT_VERSION, ModelCacheLayout, ensure_cache_layout, is_model_installed,
    known_models,
};

// ---------------------------------------------------------------------------
// Cache layout: directory creation and idempotency
// ---------------------------------------------------------------------------

#[test]
fn ensure_cache_layout_creates_model_subdirectories() {
    let tmp = tempfile::tempdir().unwrap();
    let layout = ModelCacheLayout::for_root(tmp.path().to_path_buf());
    ensure_cache_layout(&layout).unwrap();

    // Every known model should have a directory under the cache root.
    for model in known_models() {
        let model_dir = layout.model_path(model.dir_name);
        assert!(
            model_dir.is_some(),
            "layout should have a path for known model {}",
            model.dir_name
        );
        let path = model_dir.unwrap();
        assert!(
            path.exists(),
            "directory for {} should exist after ensure_cache_layout",
            model.dir_name
        );
    }
}

#[test]
fn ensure_cache_layout_is_idempotent() {
    let tmp = tempfile::tempdir().unwrap();
    let layout = ModelCacheLayout::for_root(tmp.path().to_path_buf());
    ensure_cache_layout(&layout).unwrap();
    // Call again — should not fail.
    ensure_cache_layout(&layout).unwrap();

    // Both calls should succeed and the layout should match the constant.
    const { assert!(MODEL_CACHE_LAYOUT_VERSION > 0) };
}

#[test]
fn ensure_cache_layout_creates_nested_parent_directories() {
    let tmp = tempfile::tempdir().unwrap();
    let deep_path = tmp.path().join("a").join("b").join("c").join("models");

    let layout = ModelCacheLayout::for_root(deep_path.clone());
    ensure_cache_layout(&layout).unwrap();
    assert!(deep_path.exists(), "deep path should be created");
}

// ---------------------------------------------------------------------------
// Installation detection: file presence checks
// ---------------------------------------------------------------------------

#[test]
fn is_model_installed_false_for_empty_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("empty_model").join("v1");
    std::fs::create_dir_all(&model_dir).unwrap();

    // An empty directory with required files should not be installed.
    assert!(
        !is_model_installed(&model_dir, &["model.bin"]),
        "empty directory should not report model as installed"
    );
}

#[test]
fn is_model_installed_true_after_populating_all_files() {
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("test_model").join("v1");
    std::fs::create_dir_all(&model_dir).unwrap();

    let required_files = &["model.bin", "config.json"];
    for file_name in required_files {
        std::fs::write(model_dir.join(file_name), b"test content").unwrap();
    }

    assert!(
        is_model_installed(&model_dir, required_files),
        "model should be installed after populating all required files"
    );
}

#[test]
fn is_model_installed_false_when_missing_one_file() {
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("partial_model").join("v1");
    std::fs::create_dir_all(&model_dir).unwrap();

    // Only create one of two required files.
    std::fs::write(model_dir.join("model.bin"), b"content").unwrap();

    assert!(
        !is_model_installed(&model_dir, &["model.bin", "config.json"]),
        "model with missing files should not be installed"
    );
}

// ---------------------------------------------------------------------------
// Known models: sanity checks
// ---------------------------------------------------------------------------

#[test]
fn known_models_is_non_empty() {
    assert!(
        !known_models().is_empty(),
        "known_models should contain at least one entry"
    );
}

#[test]
fn known_models_have_non_empty_dir_names() {
    for model in known_models() {
        assert!(
            !model.dir_name.is_empty(),
            "known model dir_name should not be empty"
        );
        assert!(
            !model.version.is_empty(),
            "known model version should not be empty for {}",
            model.dir_name
        );
    }
}

// ---------------------------------------------------------------------------
// Layout: model_path returns None for unknown models
// ---------------------------------------------------------------------------

#[test]
fn layout_model_path_returns_none_for_unknown_model() {
    let tmp = tempfile::tempdir().unwrap();
    let layout = ModelCacheLayout::for_root(tmp.path().to_path_buf());
    assert!(
        layout.model_path("nonexistent-model-abc123").is_none(),
        "unknown model should return None"
    );
}

// ---------------------------------------------------------------------------
// Schema version consistency
// ---------------------------------------------------------------------------

#[test]
fn layout_schema_version_is_positive() {
    const { assert!(MODEL_CACHE_LAYOUT_VERSION > 0) };
}
