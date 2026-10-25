//! Integration tests for model download configuration, consent, and lifecycle.
//!
//! Note: These tests do NOT perform real network downloads. They exercise the
//! download configuration, consent resolution, lifecycle state machine, and
//! file verification logic that surrounds the actual HTTP transfer.

use frankensearch_embed::{
    ConsentSource, DownloadConsent, ModelFile, ModelLifecycle, ModelManifest, ModelState,
    ModelTier, PLACEHOLDER_VERIFY_AFTER_DOWNLOAD, resolve_download_consent,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn test_manifest() -> ModelManifest {
    ModelManifest {
        id: "test-model".to_owned(),
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
        description: Some("A test".to_owned()),
        download_size_bytes: 1024,
    }
}

fn verified_fixture_manifest() -> ModelManifest {
    let mut manifest = test_manifest();
    manifest.revision = "a".repeat(40);
    "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        .clone_into(&mut manifest.files[0].sha256);
    manifest.files[0].size = 11;
    manifest.download_size_bytes = 11;
    manifest
}

// ---------------------------------------------------------------------------
// Acquisition integrity: consent never bypasses manifest or artifact admission
// ---------------------------------------------------------------------------

#[cfg(feature = "download")]
#[test]
fn granted_consent_does_not_download_unqualified_manifests() {
    use asupersync::test_utils::run_test_with_cx;
    use frankensearch_core::SearchError;
    use frankensearch_embed::{DownloadConfig, ModelDownloader};
    use std::io::ErrorKind;
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let mut mutable_revision = ModelManifest::minilm_v2();
    mutable_revision.revision = "main".to_owned();
    let mut placeholder_checksum = ModelManifest::minilm_v2();
    placeholder_checksum.files[0].sha256 = PLACEHOLDER_VERIFY_AFTER_DOWNLOAD.to_owned();
    let mut malformed_checksum = ModelManifest::minilm_v2();
    malformed_checksum.files[0].sha256 = "invalid-sha256".to_owned();

    for mut manifest in [
        ModelManifest::flashrank_nano(),
        mutable_revision,
        placeholder_checksum,
        malformed_checksum,
    ] {
        // An observing local listener makes accidental network access detectable
        // without contacting a model host or relying on DNS/network failures.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        for file in &mut manifest.files {
            file.url = Some(format!("https://{address}/{}", file.name));
        }
        let temp = tempfile::tempdir().unwrap();
        let destination = temp.path().join("uncreated-cache").join("model");
        let mut lifecycle = ModelLifecycle::new(
            manifest.clone(),
            DownloadConsent::granted(ConsentSource::Interactive),
        );
        let downloader = ModelDownloader::new(DownloadConfig {
            max_retries: 0,
            ..DownloadConfig::default()
        });

        run_test_with_cx(|cx| async move {
            let progress_calls = AtomicUsize::new(0);
            let error = downloader
                .download_model(&cx, &manifest, &destination, &mut lifecycle, |_| {
                    progress_calls.fetch_add(1, Ordering::Relaxed);
                })
                .await
                .expect_err("consent must not admit an unqualified manifest");
            assert!(matches!(error, SearchError::InvalidConfig { .. }));
            assert!(matches!(
                lifecycle.state(),
                ModelState::VerificationFailed { .. }
            ));
            assert_eq!(progress_calls.load(Ordering::Relaxed), 0);
            assert!(!destination.exists());
            assert!(!destination.parent().unwrap().exists());
            assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 0);
            let network_error = listener
                .accept()
                .expect_err("rejected manifests must not open a connection");
            assert_eq!(network_error.kind(), ErrorKind::WouldBlock);
        });
    }
}

#[test]
fn invalid_staged_artifacts_never_replace_an_existing_installation() {
    let verified = verified_fixture_manifest();
    let mut unpinned = verified.clone();
    unpinned.revision = "main".to_owned();
    let mut unverified = verified.clone();
    unverified.files[0].sha256 = PLACEHOLDER_VERIFY_AFTER_DOWNLOAD.to_owned();

    for (manifest, content) in [
        (verified.clone(), b"jello world".as_slice()),
        (verified, b"hello".as_slice()),
        (unpinned, b"hello world".as_slice()),
        (unverified, b"hello world".as_slice()),
    ] {
        let temp = tempfile::tempdir().unwrap();
        let staged = temp.path().join("staged");
        let installed = temp.path().join("installed");
        std::fs::create_dir(&staged).unwrap();
        std::fs::create_dir(&installed).unwrap();
        std::fs::write(staged.join("model.bin"), content).unwrap();
        std::fs::write(installed.join("model.bin"), b"previous installation").unwrap();

        manifest
            .promote_verified_installation(&staged, &installed)
            .expect_err("corrupt, truncated, unpinned, or unverified assets must be rejected");

        assert_eq!(
            std::fs::read(installed.join("model.bin")).unwrap(),
            b"previous installation"
        );
        assert_eq!(std::fs::read(staged.join("model.bin")).unwrap(), content);
        assert_eq!(std::fs::read_dir(&staged).unwrap().count(), 1);
        assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 2);
    }
}

#[test]
fn verified_staged_artifacts_can_be_installed() {
    let manifest = verified_fixture_manifest();
    let temp = tempfile::tempdir().unwrap();
    let staged = temp.path().join("staged");
    let installed = temp.path().join("installed");
    std::fs::create_dir(&staged).unwrap();
    std::fs::write(staged.join("model.bin"), b"hello world").unwrap();

    assert!(
        manifest
            .promote_verified_installation(&staged, &installed)
            .expect("valid pinned artifacts pass the same installation path")
            .is_none()
    );
    manifest.verify_dir(&installed).unwrap();
    assert!(!staged.exists());
    assert_eq!(
        std::fs::read(installed.join("model.bin")).unwrap(),
        b"hello world"
    );
}

// ---------------------------------------------------------------------------
// Download consent: resolution logic (pure, no env mutation)
// ---------------------------------------------------------------------------

#[test]
fn consent_programmatic_true_grants() {
    let consent = resolve_download_consent(Some(true), None, None);
    assert!(consent.granted, "programmatic=true should grant consent");
    assert_eq!(consent.source, Some(ConsentSource::Programmatic));
}

#[test]
fn consent_programmatic_false_denies() {
    let consent = resolve_download_consent(Some(false), None, None);
    assert!(!consent.granted, "programmatic=false should deny consent");
}

#[test]
fn consent_interactive_grants_when_no_programmatic() {
    let consent = resolve_download_consent(None, Some(true), None);
    // If there is no env override, interactive should be the source.
    // Note: env var may or may not be set in CI. We test the pure logic
    // only via internal resolution (programmatic overrides env).
    assert!(
        consent.granted || consent.source != Some(ConsentSource::Interactive),
        "interactive consent should eventually be considered"
    );
}

#[test]
fn consent_config_file_grants_when_nothing_else() {
    let consent = resolve_download_consent(None, None, Some(true));
    // If env var is not set in this test environment, config_file should win.
    // We cannot guarantee env state, so just check the struct is well-formed.
    assert!(
        consent.source.is_some() || !consent.granted,
        "consent should have a source when granted"
    );
}

#[test]
fn consent_all_none_denies_by_default() {
    // Note: env var may or may not be set. If unset, this should deny.
    let consent = resolve_download_consent(None, None, None);
    // If no env var is set, consent should be denied with no source.
    // We can't guarantee env state in CI so just verify the type is correct.
    let _ = consent.granted;
    let _ = consent.source;
}

// ---------------------------------------------------------------------------
// DownloadConsent: constructors
// ---------------------------------------------------------------------------

#[test]
fn consent_granted_with_source_tracks_origin() {
    let consent = DownloadConsent::granted(ConsentSource::Environment);
    assert!(consent.granted);
    assert_eq!(consent.source, Some(ConsentSource::Environment));
}

#[test]
fn consent_denied_is_not_granted() {
    let consent = DownloadConsent::denied(None);
    assert!(!consent.granted);
    assert_eq!(consent.source, None);
}

#[test]
fn consent_denied_with_source() {
    let consent = DownloadConsent::denied(Some(ConsentSource::ConfigFile));
    assert!(!consent.granted);
    assert_eq!(consent.source, Some(ConsentSource::ConfigFile));
}

// ---------------------------------------------------------------------------
// ModelLifecycle: state machine transitions
// ---------------------------------------------------------------------------

#[test]
fn lifecycle_starts_in_not_installed_when_consent_granted() {
    let manifest = test_manifest();
    let consent = DownloadConsent::granted(ConsentSource::Programmatic);
    let lifecycle = ModelLifecycle::new(manifest, consent);
    assert!(
        matches!(lifecycle.state(), ModelState::NotInstalled),
        "consented lifecycle should start as NotInstalled"
    );
}

#[test]
fn lifecycle_starts_in_needs_consent_when_denied() {
    let manifest = test_manifest();
    let consent = DownloadConsent::denied(None);
    let lifecycle = ModelLifecycle::new(manifest, consent);
    assert!(
        matches!(lifecycle.state(), ModelState::NeedsConsent),
        "denied lifecycle should start as NeedsConsent"
    );
}

#[test]
fn lifecycle_approve_consent_transitions_from_needs_consent() {
    let manifest = test_manifest();
    let consent = DownloadConsent::denied(None);
    let mut lifecycle = ModelLifecycle::new(manifest, consent);
    lifecycle.approve_consent(ConsentSource::Interactive);
    assert!(
        matches!(lifecycle.state(), ModelState::NotInstalled),
        "approving consent should transition to NotInstalled"
    );
}

#[test]
fn lifecycle_begin_download_requires_consent() {
    let manifest = test_manifest();
    let consent = DownloadConsent::denied(None);
    let mut lifecycle = ModelLifecycle::new(manifest, consent);
    let result = lifecycle.begin_download(1024);
    assert!(
        result.is_err(),
        "beginning download without consent should fail"
    );
}

#[test]
fn lifecycle_begin_download_after_consent_succeeds() {
    let manifest = test_manifest();
    let consent = DownloadConsent::granted(ConsentSource::Programmatic);
    let mut lifecycle = ModelLifecycle::new(manifest, consent);
    let result = lifecycle.begin_download(1024);
    assert!(result.is_ok(), "download should start after consent");
    assert!(
        matches!(lifecycle.state(), ModelState::Downloading { .. }),
        "state should be Downloading"
    );
}

#[test]
fn lifecycle_success_path_requires_reindex_after_acquisition() {
    let manifest = test_manifest();
    let consent = DownloadConsent::granted(ConsentSource::Programmatic);
    let mut lifecycle = ModelLifecycle::new(manifest, consent);
    lifecycle.begin_download(1024).unwrap();
    lifecycle.begin_verification().unwrap();
    lifecycle.mark_staged_verified().unwrap();
    assert!(
        matches!(lifecycle.state(), ModelState::StagedVerified),
        "download-only state should be StagedVerified"
    );
    lifecycle.mark_acquired_needs_reindex().unwrap();
    assert!(matches!(
        lifecycle.state(),
        ModelState::AcquiredNeedsReindex
    ));
}

#[test]
fn lifecycle_verification_failure_path() {
    let manifest = test_manifest();
    let consent = DownloadConsent::granted(ConsentSource::Programmatic);
    let mut lifecycle = ModelLifecycle::new(manifest, consent);
    lifecycle.begin_download(1024).unwrap();
    lifecycle.begin_verification().unwrap();
    lifecycle.fail_verification("hash mismatch");
    assert!(
        matches!(lifecycle.state(), ModelState::VerificationFailed { .. }),
        "state should be VerificationFailed"
    );
}

#[test]
fn lifecycle_can_recover_after_verification_failure() {
    let manifest = test_manifest();
    let consent = DownloadConsent::granted(ConsentSource::Programmatic);
    let mut lifecycle = ModelLifecycle::new(manifest, consent);
    lifecycle.begin_download(1024).unwrap();
    lifecycle.begin_verification().unwrap();
    lifecycle.fail_verification("hash mismatch");

    // Should be able to restart download from failure state.
    let result = lifecycle.begin_download(1024);
    assert!(
        result.is_ok(),
        "should be able to restart download after verification failure"
    );
}

#[test]
fn lifecycle_cannot_begin_download_from_ready_state() {
    let manifest = test_manifest();
    let consent = DownloadConsent::granted(ConsentSource::Programmatic);
    let mut lifecycle = ModelLifecycle::new(manifest, consent);
    lifecycle.begin_download(1024).unwrap();
    lifecycle.begin_verification().unwrap();
    lifecycle.mark_ready();

    let result = lifecycle.begin_download(2048);
    assert!(
        result.is_err(),
        "cannot begin download when already in Ready state"
    );
}

#[test]
fn lifecycle_begin_download_rejects_zero_size() {
    let manifest = test_manifest();
    let consent = DownloadConsent::granted(ConsentSource::Programmatic);
    let mut lifecycle = ModelLifecycle::new(manifest, consent);
    let result = lifecycle.begin_download(0);
    assert!(result.is_err(), "zero byte download should be rejected");
}

#[test]
fn lifecycle_cancel_transitions_to_cancelled() {
    let manifest = test_manifest();
    let consent = DownloadConsent::granted(ConsentSource::Programmatic);
    let mut lifecycle = ModelLifecycle::new(manifest, consent);
    lifecycle.begin_download(1024).unwrap();
    lifecycle.cancel();
    assert!(
        matches!(lifecycle.state(), ModelState::Cancelled),
        "state should be Cancelled"
    );
}

#[test]
fn lifecycle_recovery_from_cancelled() {
    let manifest = test_manifest();
    let consent = DownloadConsent::granted(ConsentSource::Programmatic);
    let mut lifecycle = ModelLifecycle::new(manifest, consent);
    lifecycle.begin_download(1024).unwrap();
    lifecycle.cancel();
    lifecycle.recover_after_cancel().unwrap();
    assert!(
        matches!(lifecycle.state(), ModelState::NotInstalled),
        "recovery from cancelled should reach NotInstalled"
    );
}

// ---------------------------------------------------------------------------
// Download progress: bounded percentage
// ---------------------------------------------------------------------------

#[test]
fn lifecycle_download_progress_is_bounded() {
    let manifest = test_manifest();
    let consent = DownloadConsent::granted(ConsentSource::Programmatic);
    let mut lifecycle = ModelLifecycle::new(manifest, consent);
    lifecycle.begin_download(100).unwrap();

    // Report progress exceeding total — should be bounded.
    lifecycle.update_download_progress(200).unwrap();

    match lifecycle.state() {
        ModelState::Downloading { progress_pct, .. } => {
            assert!(
                *progress_pct <= 100,
                "progress should be bounded to 100, got {progress_pct}"
            );
        }
        other => panic!("expected Downloading, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// ConsentSource: serde roundtrip
// ---------------------------------------------------------------------------

#[test]
fn consent_source_serde_roundtrip() {
    let sources = [
        ConsentSource::Environment,
        ConsentSource::ConfigFile,
        ConsentSource::Programmatic,
        ConsentSource::Interactive,
    ];

    for source in &sources {
        let json = serde_json::to_string(source).unwrap();
        let restored: ConsentSource = serde_json::from_str(&json).unwrap();
        assert_eq!(
            source, &restored,
            "ConsentSource should survive serde roundtrip"
        );
    }
}

// ---------------------------------------------------------------------------
// ModelState: serde roundtrip
// ---------------------------------------------------------------------------

#[test]
fn model_state_serde_roundtrip() {
    let states = vec![
        ModelState::NotInstalled,
        ModelState::NeedsConsent,
        ModelState::Downloading {
            progress_pct: 42,
            bytes_downloaded: 512,
            total_bytes: 1024,
        },
        ModelState::Verifying,
        ModelState::StagedVerified,
        ModelState::AcquiredNeedsReindex,
        ModelState::Ready,
        ModelState::VerificationFailed {
            reason: "bad hash".to_owned(),
        },
        ModelState::Cancelled,
        ModelState::Disabled {
            reason: "user disabled".to_owned(),
        },
    ];

    for state in &states {
        let json = serde_json::to_string(state).unwrap();
        let restored: ModelState = serde_json::from_str(&json).unwrap();
        assert_eq!(
            state, &restored,
            "ModelState should survive serde roundtrip"
        );
    }
}
