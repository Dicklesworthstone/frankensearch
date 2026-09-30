//! Production entry-point checks that do not require a real terminal or models.

use std::process::{Command, Stdio};

#[test]
fn redirected_terminal_view_is_refused_before_models_or_source_publication() {
    let root = tempfile::tempdir().unwrap();
    let store = root.path().join("not-created");
    let source = root.path().join("no-source");
    let config = root.path().join("no-config.toml");
    let output = Command::new(env!("CARGO_BIN_EXE_fsfs"))
        .current_dir(root.path())
        .args(["live-search", "--tui", "--hybrid", "--once", "--query", "alpha"])
        .arg("--watch-source").arg(&source)
        .arg("--index-dir").arg(&store)
        .arg("--config").arg(&config)
        .stdin(Stdio::null())
        .output().unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let error: serde_json::Value = serde_json::from_slice(&output.stderr).unwrap();
    let message = error["message"].as_str().unwrap();
    if cfg!(unix) {
        assert!(message.contains("terminal stdin and stdout"), "{message}");
    } else {
        assert!(message.contains("Unix terminal backend"), "{message}");
    }
    assert!(!store.exists());
    assert!(!source.exists());
    assert!(!config.exists());
}

#[test]
fn explicit_machine_format_cannot_silently_turn_into_terminal_output() {
    let output = Command::new(env!("CARGO_BIN_EXE_fsfs"))
        .args(["live-search", "--index-dir", "/unused", "--query", "alpha",
            "--tui", "--format", "jsonl"])
        .stdin(Stdio::null()).output().unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    let error: serde_json::Value = serde_json::from_slice(&output.stderr).unwrap();
    assert_eq!(error["phase"], "arguments");
}
