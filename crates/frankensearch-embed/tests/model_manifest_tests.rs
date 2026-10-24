//! Integration tests for model manifest parsing, validation, and catalog operations.
//!
//! These tests verify the manifest schema, forward compatibility, embedded catalogs,
//! and validation edge cases that span multiple manifest components.

use frankensearch_embed::{
    MANIFEST_SCHEMA_VERSION, ModelFile, ModelManifest, ModelManifestCatalog, ModelTier,
    PLACEHOLDER_VERIFY_AFTER_DOWNLOAD,
};

// ---------------------------------------------------------------------------
// Catalog: empty catalog is valid
// ---------------------------------------------------------------------------

#[test]
fn empty_catalog_is_valid() {
    let catalog = ModelManifestCatalog::default();
    assert!(
        catalog.validate().is_ok(),
        "empty catalog should pass validation"
    );
}

#[test]
fn default_acquisition_catalog_contains_only_production_ready_manifests() {
    let catalog = ModelManifest::builtin_catalog();
    catalog.validate().expect("default catalog is valid");
    assert!(!catalog.models.is_empty(), "fresh setup must select models");

    for manifest in &catalog.models {
        assert!(
            manifest.is_production_ready(),
            "default setup must be able to acquire {}",
            manifest.id
        );
        manifest
            .validate_checksum_policy_for(true)
            .expect("default downloads obey release checksum policy");
        assert!(
            manifest.files.iter().all(|file| file.size > 0),
            "{} must report known, nonzero artifact sizes before consent",
            manifest.id
        );
        assert_eq!(
            manifest.download_size_bytes,
            manifest.files.iter().map(|file| file.size).sum::<u64>(),
            "{} must report the complete download size",
            manifest.id
        );
        for file in &manifest.files {
            assert!(
                manifest
                    .download_url(file)
                    .contains(&format!("/resolve/{}/", manifest.revision)),
                "{}:{} must download from its immutable revision",
                manifest.id,
                file.name
            );
        }
    }

    for tier in [ModelTier::Fast, ModelTier::Quality, ModelTier::Reranker] {
        assert!(
            catalog.models.iter().any(|model| model.tier == Some(tier)),
            "fresh setup must retain the {tier:?} tier"
        );
    }
    let native_reranker = catalog
        .models
        .iter()
        .find(|model| model.id == "ms-marco-minilm-l-6-v2")
        .expect("default setup retains the supported native reranker");
    assert!(
        native_reranker
            .files
            .iter()
            .any(|file| file.name == "model.safetensors")
    );
    assert!(
        catalog
            .models
            .iter()
            .all(|model| model.id != "flashrank-nano")
    );
}

#[test]
fn flashrank_upstream_metadata_does_not_qualify_it_for_default_setup() {
    let catalog = ModelManifest::opt_in_catalog();
    let flashrank = catalog
        .models
        .iter()
        .find(|model| model.id == "flashrank-nano")
        .expect("explicit FlashRank selection retains an unavailable descriptor");
    flashrank.validate().expect("descriptor metadata is valid");
    assert_eq!(flashrank.repo, "prithivida/flashrank");
    assert_eq!(flashrank.license, "CC-BY-SA-4.0");
    assert_eq!(flashrank.files.len(), 1);
    assert_eq!(flashrank.files[0].name, "ms-marco-TinyBERT-L-2-v2.zip");
    assert_eq!(flashrank.files[0].size, 3_421_983);
    assert_eq!(flashrank.total_size_bytes(), 3_421_983);
    assert!(!flashrank.has_verified_checksums());
    assert!(!flashrank.is_production_ready());
    assert!(flashrank.validate_checksum_policy_for(true).is_err());
}

#[test]
fn catalog_json_roundtrip_preserves_all_fields() {
    let catalog = ModelManifestCatalog {
        schema_version: MANIFEST_SCHEMA_VERSION,
        models: vec![ModelManifest {
            id: "test-roundtrip".to_owned(),
            repo: "test/repo".to_owned(),
            revision: "abc123".to_owned(),
            files: vec![ModelFile {
                name: "model.bin".to_owned(),
                sha256: "a".repeat(64),
                size: 1024,
                url: None,
            }],
            license: "MIT".to_owned(),
            tier: Some(ModelTier::Fast),
            dimension: Some(256),
            display_name: Some("Test Model".to_owned()),
            version: "1.0.0".to_owned(),
            description: Some("A test model".to_owned()),
            download_size_bytes: 1024,
        }],
    };

    let json = serde_json::to_string_pretty(&catalog).unwrap();
    let restored: ModelManifestCatalog = serde_json::from_str(&json).unwrap();

    assert_eq!(
        catalog.models.len(),
        restored.models.len(),
        "roundtrip should preserve model count"
    );
    for (orig, rest) in catalog.models.iter().zip(restored.models.iter()) {
        assert_eq!(orig.id, rest.id, "model ID should survive roundtrip");
        assert_eq!(
            orig.files.len(),
            rest.files.len(),
            "file count should survive roundtrip for {}",
            orig.id
        );
    }
}

// ---------------------------------------------------------------------------
// Schema version: forward compatibility
// ---------------------------------------------------------------------------

#[test]
fn manifest_with_unknown_fields_parses_without_error() {
    let json = r#"{
        "id": "test-forward-compat",
        "repo": "test/repo",
        "revision": "abc123",
        "files": [],
        "license": "MIT",
        "version": "1.0",
        "download_size_bytes": 0,
        "future_field_that_does_not_exist_yet": true,
        "another_new_field": { "nested": 42 }
    }"#;

    let result: Result<ModelManifest, _> = serde_json::from_str(json);
    assert!(
        result.is_ok(),
        "unknown fields should be silently ignored: {:?}",
        result.err()
    );
}

#[test]
fn schema_version_is_consistent() {
    const { assert!(MANIFEST_SCHEMA_VERSION >= 2) };
}

// ---------------------------------------------------------------------------
// Missing required fields: specific error messages
// ---------------------------------------------------------------------------

#[test]
fn missing_id_field_gives_clear_serde_error() {
    let json = r#"{
        "repo": "test/repo",
        "revision": "abc",
        "files": [],
        "license": "MIT",
        "version": "1.0",
        "download_size_bytes": 0
    }"#;

    let err = serde_json::from_str::<ModelManifest>(json).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("id"),
        "error should mention the missing 'id' field: {msg}"
    );
}

#[test]
fn missing_repo_field_gives_clear_serde_error() {
    let json = r#"{
        "id": "test",
        "revision": "abc",
        "files": [],
        "license": "MIT",
        "version": "1.0",
        "download_size_bytes": 0
    }"#;

    let err = serde_json::from_str::<ModelManifest>(json).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("repo"),
        "error should mention the missing 'repo' field: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Manifest validation: content constraints
// ---------------------------------------------------------------------------

#[test]
fn manifest_validates_non_empty_file_list() {
    let manifest = ModelManifest {
        id: "test-manifest".to_owned(),
        repo: "test/repo".to_owned(),
        revision: "abc123def456".to_owned(),
        files: vec![ModelFile {
            name: "model.bin".to_owned(),
            sha256: "a".repeat(64),
            size: 1024,
            url: None,
        }],
        license: "MIT".to_owned(),
        tier: Some(ModelTier::Fast),
        dimension: Some(256),
        display_name: Some("Test Model".to_owned()),
        version: "1.0.0".to_owned(),
        description: Some("A test model".to_owned()),
        download_size_bytes: 1024,
    };

    assert!(
        manifest.validate().is_ok(),
        "valid manifest should pass validation"
    );
}

#[test]
fn manifest_rejects_empty_file_name() {
    let manifest = ModelManifest {
        id: "test".to_owned(),
        repo: "test/repo".to_owned(),
        revision: "abc".to_owned(),
        files: vec![ModelFile {
            name: String::new(),
            sha256: "a".repeat(64),
            size: 100,
            url: None,
        }],
        license: "MIT".to_owned(),
        tier: None,
        dimension: None,
        display_name: None,
        version: "1.0".to_owned(),
        description: None,
        download_size_bytes: 100,
    };

    assert!(
        manifest.validate().is_err(),
        "manifest with empty file name should fail validation"
    );
}

#[test]
fn manifest_total_size_bytes_sums_all_files() {
    let manifest = ModelManifest {
        id: "test".to_owned(),
        repo: "r".to_owned(),
        revision: "v".to_owned(),
        files: vec![
            ModelFile {
                name: "a.bin".to_owned(),
                sha256: "a".repeat(64),
                size: 100,
                url: None,
            },
            ModelFile {
                name: "b.bin".to_owned(),
                sha256: "b".repeat(64),
                size: 200,
                url: None,
            },
        ],
        license: "MIT".to_owned(),
        tier: None,
        dimension: None,
        display_name: None,
        version: String::new(),
        description: None,
        download_size_bytes: 300,
    };

    assert_eq!(manifest.total_size_bytes(), 300);
}

// ---------------------------------------------------------------------------
// Placeholder checksums: correctly identified
// ---------------------------------------------------------------------------

#[test]
fn placeholder_checksum_is_not_valid_sha256() {
    assert_ne!(
        PLACEHOLDER_VERIFY_AFTER_DOWNLOAD.len(),
        64,
        "placeholder should not look like a valid SHA-256"
    );
}

#[test]
fn model_file_identifies_placeholder_vs_verified_checksum() {
    let placeholder = ModelFile {
        name: "model.bin".to_owned(),
        sha256: PLACEHOLDER_VERIFY_AFTER_DOWNLOAD.to_owned(),
        size: 0,
        url: None,
    };

    let verified = ModelFile {
        name: "model.bin".to_owned(),
        sha256: "a".repeat(64),
        size: 1024,
        url: None,
    };

    assert!(
        !placeholder.has_verified_checksum(),
        "placeholder should not count as verified"
    );
    assert!(
        verified.has_verified_checksum(),
        "64-char hex should count as verified"
    );
}
