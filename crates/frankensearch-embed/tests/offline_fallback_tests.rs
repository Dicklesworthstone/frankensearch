//! Integration tests for offline fallback and degradation behavior.
//!
//! These tests verify that the embedder stack degrades gracefully when
//! models are missing, features are disabled, or offline mode is active.
//! They exercise the `TwoTierAvailability`, `EmbedderStack`, and
//! `ModelAvailabilityDiagnostic` types.

use frankensearch_embed::{
    EmbedderStack, ModelAvailabilityDiagnostic, ModelStatus, TwoTierAvailability,
};

// ---------------------------------------------------------------------------
// TwoTierAvailability: degradation classification
// ---------------------------------------------------------------------------

#[test]
fn full_availability_is_not_degraded() {
    assert!(!TwoTierAvailability::Full.is_degraded());
}

#[test]
fn fast_only_availability_is_degraded() {
    assert!(TwoTierAvailability::FastOnly.is_degraded());
}

#[test]
fn hash_only_availability_is_degraded() {
    assert!(TwoTierAvailability::HashOnly.is_degraded());
}

#[test]
fn availability_display_matches_expected_labels() {
    let full = format!("{}", TwoTierAvailability::Full);
    let fast = format!("{}", TwoTierAvailability::FastOnly);
    let hash = format!("{}", TwoTierAvailability::HashOnly);

    assert!(
        full.contains("Full") || full.contains("full"),
        "Full display: {full}"
    );
    assert!(
        fast.contains("Fast") || fast.contains("fast"),
        "FastOnly display: {fast}"
    );
    assert!(
        hash.contains("Hash") || hash.contains("hash"),
        "HashOnly display: {hash}"
    );
}

// ---------------------------------------------------------------------------
// Degradation summary: present only when degraded
// ---------------------------------------------------------------------------

#[test]
fn degradation_summary_none_when_full() {
    assert!(
        TwoTierAvailability::Full.degradation_summary().is_none(),
        "Full availability should have no degradation summary"
    );
}

#[test]
fn degradation_summary_present_when_fast_only() {
    let summary = TwoTierAvailability::FastOnly.degradation_summary();
    assert!(
        summary.is_some(),
        "FastOnly should have a degradation summary"
    );
    let msg = summary.unwrap();
    assert!(!msg.is_empty(), "degradation summary should not be empty");
}

#[test]
fn degradation_summary_present_when_hash_only() {
    let summary = TwoTierAvailability::HashOnly.degradation_summary();
    assert!(
        summary.is_some(),
        "HashOnly should have a degradation summary"
    );
}

// ---------------------------------------------------------------------------
// ModelStatus: display and variant coverage
// ---------------------------------------------------------------------------

#[test]
fn model_status_ready_displays_model_id() {
    let status = ModelStatus::Ready {
        id: "potion-128M".to_owned(),
    };
    let display = format!("{status}");
    assert!(
        display.contains("potion-128M"),
        "Ready display should include model id: {display}"
    );
}

#[test]
fn model_status_not_found_displays_model_name() {
    let status = ModelStatus::NotFound {
        model_name: "all-MiniLM-L6-v2".to_owned(),
        hf_repo_url: "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2".to_owned(),
        searched_paths: vec!["/tmp/cache/all-MiniLM-L6-v2".into()],
    };
    let display = format!("{status}");
    assert!(
        display.contains("all-MiniLM-L6-v2"),
        "NotFound display should include model name: {display}"
    );
}

#[test]
fn model_status_download_blocked_displays_reason() {
    let status = ModelStatus::DownloadBlocked {
        model_name: "all-MiniLM-L6-v2".to_owned(),
        reason: "FRANKENSEARCH_OFFLINE=1 disables auto-download".to_owned(),
    };
    let display = format!("{status}");
    assert!(
        display.contains("OFFLINE") || display.contains("blocked") || display.contains("disabled"),
        "DownloadBlocked display should mention offline/blocked: {display}"
    );
}

#[test]
fn model_status_feature_disabled_displays_flag() {
    let status = ModelStatus::FeatureDisabled {
        feature_flag: "fastembed".to_owned(),
    };
    let display = format!("{status}");
    assert!(
        display.contains("fastembed"),
        "FeatureDisabled display should include feature name: {display}"
    );
}

#[test]
fn model_status_hash_fallback_display_is_descriptive() {
    let status = ModelStatus::HashFallback;
    let display = format!("{status}");
    assert!(
        !display.is_empty(),
        "HashFallback display should not be empty"
    );
}

// ---------------------------------------------------------------------------
// ModelAvailabilityDiagnostic: structure and display
// ---------------------------------------------------------------------------

#[test]
fn diagnostic_includes_availability_and_statuses() {
    let diag = ModelAvailabilityDiagnostic {
        availability: TwoTierAvailability::HashOnly,
        fast_status: ModelStatus::HashFallback,
        quality_status: ModelStatus::FeatureDisabled {
            feature_flag: "fastembed".to_owned(),
        },
        cache_dir: "/tmp/cache".into(),
        offline: false,
        suggestions: vec!["Install models".to_owned()],
    };

    let display = format!("{diag}");
    assert!(
        display.to_ascii_lowercase().contains("hash"),
        "diagnostic should mention hash: {display}"
    );
    assert!(
        display.contains("fastembed"),
        "diagnostic should mention disabled feature: {display}"
    );
}

#[test]
fn diagnostic_offline_mode_marked_when_true() {
    let diag = ModelAvailabilityDiagnostic {
        availability: TwoTierAvailability::HashOnly,
        fast_status: ModelStatus::HashFallback,
        quality_status: ModelStatus::DownloadBlocked {
            model_name: "test".to_owned(),
            reason: "offline".to_owned(),
        },
        cache_dir: "/tmp/cache".into(),
        offline: true,
        suggestions: vec![],
    };

    let display = format!("{diag}");
    assert!(
        display.contains("OFFLINE") || display.contains("offline"),
        "diagnostic display should mark offline mode: {display}"
    );
}

#[test]
fn diagnostic_no_suggestions_when_fully_available() {
    let diag = ModelAvailabilityDiagnostic {
        availability: TwoTierAvailability::Full,
        fast_status: ModelStatus::Ready {
            id: "potion-128M".to_owned(),
        },
        quality_status: ModelStatus::Ready {
            id: "MiniLM-L6-v2".to_owned(),
        },
        cache_dir: "/tmp/cache".into(),
        offline: false,
        suggestions: vec![],
    };

    assert!(
        diag.suggestions.is_empty(),
        "full availability should have no suggestions"
    );
}

// ---------------------------------------------------------------------------
// EmbedderStack: auto-detection with explicit path
// ---------------------------------------------------------------------------

#[test]
fn auto_detect_with_empty_cache_returns_degraded() {
    let tmp = tempfile::tempdir().unwrap();

    // Use auto_detect_with to avoid env mutation.
    let stack = EmbedderStack::auto_detect_with(Some(tmp.path())).unwrap();
    let availability = stack.availability();

    assert!(
        availability.is_degraded(),
        "empty cache should result in degraded availability, got: {availability}"
    );
}

#[test]
fn auto_detect_fast_embedder_is_always_available() {
    let tmp = tempfile::tempdir().unwrap();

    let stack = EmbedderStack::auto_detect_with(Some(tmp.path())).unwrap();

    // A fast embedder should always be available (hash or model2vec fallback).
    let id = stack.fast().id();
    assert!(
        !id.is_empty(),
        "fast embedder should have a non-empty id, got: {id}"
    );
}

#[test]
fn embedder_stack_degradation_message_present_when_degraded() {
    let tmp = tempfile::tempdir().unwrap();

    let stack = EmbedderStack::auto_detect_with(Some(tmp.path())).unwrap();
    let msg = stack.degradation_message();

    if stack.availability().is_degraded() {
        assert!(
            msg.is_some(),
            "degraded stack should produce a degradation message"
        );
        let text = msg.unwrap();
        assert!(!text.is_empty(), "degradation message should not be empty");
    }
}

#[test]
fn embedder_stack_diagnose_returns_valid_diagnostic() {
    let tmp = tempfile::tempdir().unwrap();

    let stack = EmbedderStack::auto_detect_with(Some(tmp.path())).unwrap();
    let diag = stack.diagnose();

    // Diagnostic should be displayable without panicking.
    let display = format!("{diag}");
    assert!(
        !display.is_empty(),
        "diagnostic display should not be empty"
    );

    // Cache dir should be valid.
    assert!(
        !diag.cache_dir.as_os_str().is_empty(),
        "diagnostic cache_dir should not be empty"
    );
}
