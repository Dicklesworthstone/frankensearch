//! Integration tests for model integrity verification (SHA-256, corruption detection,
//! verification markers, and cached verification).
//!
//! These tests exercise the full integrity pipeline:
//! - `verify_file_sha256()` for individual file checks
//! - `VerificationMarker` for caching verification results
//! - `verify_dir_cached()` for the combined cached-verification workflow

use sha2::{Digest, Sha256};

use frankensearch_embed::{
    MANIFEST_SCHEMA_VERSION, ModelFile, ModelManifest, PLACEHOLDER_VERIFY_AFTER_DOWNLOAD,
    VerificationMarker, is_verification_cached, verify_dir_cached, verify_file_sha256,
    write_verification_marker,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

fn make_manifest(file_name: &str, content: &[u8]) -> ModelManifest {
    ModelManifest {
        id: "integrity-test-model".to_owned(),
        repo: "test/integrity".to_owned(),
        revision: "abc123".to_owned(),
        files: vec![ModelFile {
            name: file_name.to_owned(),
            sha256: sha256_hex(content),
            size: u64::try_from(content.len()).unwrap(),
            url: None,
        }],
        license: "MIT".to_owned(),
        tier: None,
        dimension: None,
        display_name: None,
        version: String::new(),
        description: None,
        download_size_bytes: u64::try_from(content.len()).unwrap(),
    }
}

// ---------------------------------------------------------------------------
// verify_file_sha256: valid files pass
// ---------------------------------------------------------------------------

#[test]
fn verify_file_sha256_succeeds_for_matching_content() {
    let tmp = tempfile::tempdir().unwrap();
    let content = b"hello model data for integrity check";
    let path = tmp.path().join("model.bin");
    std::fs::write(&path, content).unwrap();

    let hash = sha256_hex(content);
    let size = u64::try_from(content.len()).unwrap();

    let result = verify_file_sha256(&path, &hash, size);
    assert!(result.is_ok(), "valid file should pass: {:?}", result.err());
}

// ---------------------------------------------------------------------------
// verify_file_sha256: corrupt files fail
// ---------------------------------------------------------------------------

#[test]
fn verify_file_sha256_fails_for_flipped_bit() {
    let tmp = tempfile::tempdir().unwrap();
    let original = b"hello model data";
    let path = tmp.path().join("model.bin");
    std::fs::write(&path, original).unwrap();

    // Compute hash of original, then corrupt the file.
    let hash = sha256_hex(original);
    let size = u64::try_from(original.len()).unwrap();

    // Flip one bit in the first byte.
    let mut corrupted = original.to_vec();
    corrupted[0] ^= 0x01;
    std::fs::write(&path, &corrupted).unwrap();

    let result = verify_file_sha256(&path, &hash, size);
    assert!(result.is_err(), "corrupted file should fail verification");

    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("sha256=") || err_msg.contains("HashMismatch"),
        "error should mention hash mismatch: {err_msg}"
    );
}

#[test]
fn verify_file_sha256_fails_for_truncated_file() {
    let tmp = tempfile::tempdir().unwrap();
    let original = b"some model data with more bytes";
    let path = tmp.path().join("model.bin");

    let hash = sha256_hex(original);
    let size = u64::try_from(original.len()).unwrap();

    // Write only a partial file.
    std::fs::write(&path, &original[..10]).unwrap();

    let result = verify_file_sha256(&path, &hash, size);
    assert!(result.is_err(), "truncated file should fail verification");
}

#[test]
fn verify_file_sha256_fails_for_appended_data() {
    let tmp = tempfile::tempdir().unwrap();
    let original = b"model data";
    let path = tmp.path().join("model.bin");

    let hash = sha256_hex(original);
    let size = u64::try_from(original.len()).unwrap();

    // Write original + extra bytes.
    let mut extended = original.to_vec();
    extended.extend_from_slice(b"extra garbage");
    std::fs::write(&path, &extended).unwrap();

    let result = verify_file_sha256(&path, &hash, size);
    assert!(
        result.is_err(),
        "file with appended data should fail verification"
    );
}

// ---------------------------------------------------------------------------
// verify_file_sha256: missing files
// ---------------------------------------------------------------------------

#[test]
fn verify_file_sha256_fails_for_missing_file() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("nonexistent.bin");

    let result = verify_file_sha256(&path, &"a".repeat(64), 100);
    assert!(result.is_err(), "missing file should fail verification");

    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("ModelNotFound") || err_msg.contains("missing"),
        "error should indicate model not found: {err_msg}"
    );
}

// ---------------------------------------------------------------------------
// verify_file_sha256: zero-length expected size rejected
// ---------------------------------------------------------------------------

#[test]
fn verify_file_sha256_rejects_zero_expected_size() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("model.bin");
    std::fs::write(&path, b"data").unwrap();

    let result = verify_file_sha256(&path, &"a".repeat(64), 0);
    assert!(result.is_err(), "zero expected size should be rejected");
}

// ---------------------------------------------------------------------------
// verify_file_sha256: placeholder checksum rejected
// ---------------------------------------------------------------------------

#[test]
fn verify_file_sha256_rejects_placeholder_checksum() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("model.bin");
    std::fs::write(&path, b"data").unwrap();

    let result = verify_file_sha256(&path, PLACEHOLDER_VERIFY_AFTER_DOWNLOAD, 4);
    assert!(result.is_err(), "placeholder checksum should be rejected");
}

// ---------------------------------------------------------------------------
// verify_file_sha256: directory instead of file
// ---------------------------------------------------------------------------

#[test]
fn verify_file_sha256_rejects_directory_path() {
    let tmp = tempfile::tempdir().unwrap();
    let dir_path = tmp.path().join("subdir");
    std::fs::create_dir_all(&dir_path).unwrap();

    let result = verify_file_sha256(&dir_path, &"a".repeat(64), 100);
    assert!(result.is_err(), "directory path should fail verification");
}

// ---------------------------------------------------------------------------
// VerificationMarker: roundtrip and validity
// ---------------------------------------------------------------------------

#[test]
fn verification_marker_roundtrip_preserves_fields() {
    let tmp = tempfile::tempdir().unwrap();
    let content = b"marker roundtrip test";
    let manifest = make_manifest("model.bin", content);
    std::fs::write(tmp.path().join("model.bin"), content).unwrap();

    write_verification_marker(&manifest, tmp.path());

    let marker_path = tmp.path().join(".verified");
    assert!(marker_path.exists(), ".verified marker should be created");

    let raw = std::fs::read_to_string(&marker_path).unwrap();
    let marker: VerificationMarker = serde_json::from_str(&raw).unwrap();

    assert_eq!(marker.manifest_id, "integrity-test-model");
    assert_eq!(marker.schema_version, MANIFEST_SCHEMA_VERSION);
    assert!(
        marker.file_states.contains_key("model.bin"),
        "marker should record file mtime"
    );
}

// ---------------------------------------------------------------------------
// Verification cache: hit and miss scenarios
// ---------------------------------------------------------------------------

#[test]
fn verification_cache_hit_after_writing_marker() {
    let tmp = tempfile::tempdir().unwrap();
    let content = b"cache hit test";
    let manifest = make_manifest("model.bin", content);
    std::fs::write(tmp.path().join("model.bin"), content).unwrap();

    assert!(
        !is_verification_cached(&manifest, tmp.path()),
        "should not be cached before marker written"
    );

    write_verification_marker(&manifest, tmp.path());

    assert!(
        is_verification_cached(&manifest, tmp.path()),
        "should be cached after marker written"
    );
}

#[test]
fn verification_cache_miss_with_different_manifest_id() {
    let tmp = tempfile::tempdir().unwrap();
    let content = b"different manifest id test";
    let manifest = make_manifest("model.bin", content);
    std::fs::write(tmp.path().join("model.bin"), content).unwrap();

    write_verification_marker(&manifest, tmp.path());

    // Create a different manifest with a different ID.
    let mut different = manifest;
    different.id = "completely-different-model".to_owned();

    assert!(
        !is_verification_cached(&different, tmp.path()),
        "changed manifest ID should invalidate cache"
    );
}

#[test]
fn verification_cache_miss_when_file_mtime_tampered() {
    let tmp = tempfile::tempdir().unwrap();
    let content = b"mtime tamper test";
    let manifest = make_manifest("model.bin", content);
    std::fs::write(tmp.path().join("model.bin"), content).unwrap();

    write_verification_marker(&manifest, tmp.path());
    assert!(is_verification_cached(&manifest, tmp.path()));

    // Tamper with the recorded mtime in the marker.
    let marker_path = tmp.path().join(".verified");
    let raw = std::fs::read_to_string(&marker_path).unwrap();
    let mut marker: VerificationMarker = serde_json::from_str(&raw).unwrap();
    // Remove the real entry so fingerprint won't match.
    marker.file_states.remove("model.bin");
    std::fs::write(&marker_path, serde_json::to_string(&marker).unwrap()).unwrap();

    assert!(
        !is_verification_cached(&manifest, tmp.path()),
        "tampered mtime should invalidate cache"
    );
}

#[test]
fn verification_cache_miss_when_marker_is_corrupt_json() {
    let tmp = tempfile::tempdir().unwrap();
    let content = b"corrupt json test";
    let manifest = make_manifest("model.bin", content);
    std::fs::write(tmp.path().join("model.bin"), content).unwrap();

    write_verification_marker(&manifest, tmp.path());

    // Overwrite the marker with invalid JSON.
    std::fs::write(tmp.path().join(".verified"), "NOT VALID JSON {{{{").unwrap();

    assert!(
        !is_verification_cached(&manifest, tmp.path()),
        "corrupt marker JSON should not be treated as cached"
    );
}

#[test]
fn verification_cache_miss_when_marker_file_missing() {
    let tmp = tempfile::tempdir().unwrap();
    let content = b"no marker test";
    let manifest = make_manifest("model.bin", content);
    std::fs::write(tmp.path().join("model.bin"), content).unwrap();

    assert!(
        !is_verification_cached(&manifest, tmp.path()),
        "missing marker should not be treated as cached"
    );
}

// ---------------------------------------------------------------------------
// verify_dir_cached: end-to-end cached verification
// ---------------------------------------------------------------------------

#[test]
fn verify_dir_cached_creates_marker_on_first_call() {
    let tmp = tempfile::tempdir().unwrap();
    let content = b"verify dir cached e2e";
    let manifest = make_manifest("model.bin", content);
    std::fs::write(tmp.path().join("model.bin"), content).unwrap();

    let marker_path = tmp.path().join(".verified");
    assert!(!marker_path.exists());

    verify_dir_cached(&manifest, tmp.path()).unwrap();

    assert!(
        marker_path.exists(),
        "first call should create .verified marker"
    );
}

#[test]
fn verify_dir_cached_succeeds_from_cache_on_second_call() {
    let tmp = tempfile::tempdir().unwrap();
    let content = b"verify dir cached second call";
    let manifest = make_manifest("model.bin", content);
    std::fs::write(tmp.path().join("model.bin"), content).unwrap();

    // First call: full verification.
    verify_dir_cached(&manifest, tmp.path()).unwrap();

    // Second call: should succeed from cache (no re-hashing).
    verify_dir_cached(&manifest, tmp.path()).unwrap();

    // The marker should still exist.
    assert!(is_verification_cached(&manifest, tmp.path()));
}

#[test]
fn verify_dir_cached_skips_for_placeholder_checksums() {
    let tmp = tempfile::tempdir().unwrap();
    let manifest = ModelManifest {
        id: "placeholder-test".to_owned(),
        repo: "test/repo".to_owned(),
        revision: "v1".to_owned(),
        files: vec![ModelFile {
            name: "model.bin".to_owned(),
            sha256: PLACEHOLDER_VERIFY_AFTER_DOWNLOAD.to_owned(),
            size: 0,
            url: None,
        }],
        license: "MIT".to_owned(),
        tier: None,
        dimension: None,
        display_name: None,
        version: String::new(),
        description: None,
        download_size_bytes: 0,
    };

    // Should succeed (skips verification) even though the file doesn't exist.
    let result = verify_dir_cached(&manifest, tmp.path());
    assert!(
        result.is_ok(),
        "placeholder checksums should cause verification to be skipped: {:?}",
        result.err()
    );
}

#[test]
fn verify_dir_cached_fails_for_corrupt_file() {
    let tmp = tempfile::tempdir().unwrap();
    let original = b"original model data";
    let manifest = make_manifest("model.bin", original);

    // Write corrupted content.
    std::fs::write(tmp.path().join("model.bin"), b"CORRUPTED DATA").unwrap();

    let result = verify_dir_cached(&manifest, tmp.path());
    assert!(
        result.is_err(),
        "corrupt file should fail verify_dir_cached"
    );
}

// ---------------------------------------------------------------------------
// Multi-file manifest verification
// ---------------------------------------------------------------------------

#[test]
fn verify_dir_cached_checks_all_files_in_manifest() {
    let tmp = tempfile::tempdir().unwrap();
    let content_a = b"file a content";
    let content_b = b"file b content";

    let manifest = ModelManifest {
        id: "multi-file-test".to_owned(),
        repo: "test/multi".to_owned(),
        revision: "v1".to_owned(),
        files: vec![
            ModelFile {
                name: "a.bin".to_owned(),
                sha256: sha256_hex(content_a),
                size: u64::try_from(content_a.len()).unwrap(),
                url: None,
            },
            ModelFile {
                name: "b.bin".to_owned(),
                sha256: sha256_hex(content_b),
                size: u64::try_from(content_b.len()).unwrap(),
                url: None,
            },
        ],
        license: "MIT".to_owned(),
        tier: None,
        dimension: None,
        display_name: None,
        version: String::new(),
        description: None,
        download_size_bytes: u64::try_from(content_a.len() + content_b.len()).unwrap(),
    };

    std::fs::write(tmp.path().join("a.bin"), content_a).unwrap();
    std::fs::write(tmp.path().join("b.bin"), content_b).unwrap();

    let result = verify_dir_cached(&manifest, tmp.path());
    assert!(
        result.is_ok(),
        "multi-file verification should pass: {:?}",
        result.err()
    );
}

#[test]
fn verify_dir_cached_fails_when_one_of_multiple_files_corrupt() {
    let tmp = tempfile::tempdir().unwrap();
    let content_a = b"good file content";
    let content_b = b"will be corrupted";

    let manifest = ModelManifest {
        id: "multi-file-corrupt".to_owned(),
        repo: "test/multi".to_owned(),
        revision: "v1".to_owned(),
        files: vec![
            ModelFile {
                name: "a.bin".to_owned(),
                sha256: sha256_hex(content_a),
                size: u64::try_from(content_a.len()).unwrap(),
                url: None,
            },
            ModelFile {
                name: "b.bin".to_owned(),
                sha256: sha256_hex(content_b),
                size: u64::try_from(content_b.len()).unwrap(),
                url: None,
            },
        ],
        license: "MIT".to_owned(),
        tier: None,
        dimension: None,
        display_name: None,
        version: String::new(),
        description: None,
        download_size_bytes: u64::try_from(content_a.len() + content_b.len()).unwrap(),
    };

    std::fs::write(tmp.path().join("a.bin"), content_a).unwrap();
    // Write corrupted content for b.bin.
    std::fs::write(tmp.path().join("b.bin"), b"WRONG CONTENT").unwrap();

    let result = verify_dir_cached(&manifest, tmp.path());
    assert!(
        result.is_err(),
        "one corrupt file in multi-file manifest should fail verification"
    );
}
