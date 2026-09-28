//! Run the production fsfs binary through the read-only live-search dispatcher.
//! No embedded models, network, normal configuration, or source indexing needed.

use std::ffi::OsString;
use std::io;
use std::path::Path;
use std::process::{Child, Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::Value;

struct ReapedChild(Option<Child>);

impl Drop for ReapedChild {
    fn drop(&mut self) {
        if let Some(child) = self.0.as_mut() {
            // Only the test's own child is terminated. Never leave a broken
            // subscription running after an assertion or watchdog failure.
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn run(root: &Path, args: &[OsString]) -> io::Result<Output> {
    let child = Command::new(env!("CARGO_BIN_EXE_fsfs"))
        .arg("live-search")
        .args(args)
        .current_dir(root)
        .env("HOME", root)
        .env("XDG_CONFIG_HOME", root.join("config"))
        .env("XDG_DATA_HOME", root.join("data"))
        .env("XDG_CACHE_HOME", root.join("cache"))
        .env("FRANKENSEARCH_MODEL_DIR", root.join("no-models"))
        // Invalid normal options must never reach their ordinary loader.
        .env("FSFS_COMPLETE_GENERATIONS", "deliberately-invalid")
        .env("FRANKENSEARCH_COMPLETE_GENERATIONS", "deliberately-invalid")
        .env("FRANKENSEARCH_CHECK_UPDATES", "0")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let mut child = ReapedChild(Some(child));
    let started = Instant::now();
    loop {
        let running = child.0.as_mut().expect("test child retained");
        if running.try_wait()?.is_some() {
            return child
                .0
                .take()
                .expect("completed child retained")
                .wait_with_output();
        }
        if started.elapsed() > Duration::from_secs(20) {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "live-search child exceeded test watchdog",
            ));
        }
        thread::sleep(Duration::from_millis(10));
    }
}

fn strings(values: &[&str]) -> Vec<OsString> {
    values.iter().map(OsString::from).collect()
}

#[test]
fn help_works_without_models_store_or_valid_normal_configuration() {
    let root = tempfile::tempdir().unwrap();
    let output = run(root.path(), &strings(&["--help"])).unwrap();
    assert!(output.status.success(), "{:?}", output);
    let help = String::from_utf8(output.stdout).unwrap();
    assert!(help.contains("fsfs live-search --index-dir STORE --query TEXT"));
    assert!(help.contains("lexical-only"));
    assert!(output.stderr.is_empty());
    assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
}

#[test]
fn argument_errors_never_pollute_stdout_or_initialize_a_store() {
    let root = tempfile::tempdir().unwrap();
    for extra in [
        vec!["--limit", "0"],
        vec!["--format", "table"],
        vec!["--rerank"],
        vec!["--once", "--max-updates", "2"],
    ] {
        let mut args = strings(&["--index-dir", "not-created", "--query", "alpha"]);
        args.extend(strings(&extra));
        let output = run(root.path(), &args).unwrap();
        assert_eq!(output.status.code(), Some(2), "{extra:?}");
        assert!(output.stdout.is_empty());
        let error: Value = serde_json::from_slice(&output.stderr).unwrap();
        assert_eq!(error["schema_version"], "fsfs.live_search.error.v1");
        assert_eq!(error["phase"], "arguments");
        assert!(!root.path().join("not-created").exists());
    }
}

#[cfg(unix)]
mod native {
    use asupersync::test_utils::run_test_with_cx;
    use frankensearch_core::{IndexableDocument, LexicalWrite};
    use frankensearch_fsfs::generation_store::{
        COMPLETE_GENERATION_POINTER, CompleteGenerationStore, GenerationPublication,
    };
    use frankensearch_quill::{QuillConfig, QuillIndex};

    use super::*;

    fn live_args(store: &Path, extra: &[&str]) -> Vec<OsString> {
        let mut args = vec![OsString::from("--index-dir"), store.as_os_str().to_owned()];
        args.extend(strings(&["--query", "alpha"]));
        args.extend(strings(extra));
        args
    }

    #[test]
    fn production_binary_emits_a_native_snapshot_without_normal_startup() {
        let root = tempfile::tempdir().unwrap();
        let store_root = root.path().join("store");
        let fixture_root = store_root.clone();
        run_test_with_cx(|cx| async move {
            let store = CompleteGenerationStore::create(&cx, &fixture_root).unwrap();
            let build = store.begin(&cx).unwrap();
            let config = QuillConfig {
                deterministic_ingest: true,
                ..QuillConfig::default()
            };
            let index = QuillIndex::create(&cx, &build.path().join("lexical"), config)
                .await
                .unwrap();
            let documents = [IndexableDocument::new("doc-a", "alpha searchable body")
                .with_metadata("path", "/fixture/doc-a.rs")];
            LexicalWrite::index_documents(&index, &cx, &documents)
                .await
                .unwrap();
            LexicalWrite::commit(&index, &cx).await.unwrap();
            drop(index);
            assert!(matches!(
                build.publish(&cx, |_, _| Ok(())).unwrap(),
                GenerationPublication::Durable(_)
            ));
        });
        let pointer = store_root.join(COMPLETE_GENERATION_POINTER);
        let before = std::fs::read(&pointer).unwrap();
        let output = run(
            root.path(),
            &live_args(&store_root, &["--once", "--timeout-ms", "5000"]),
        )
        .unwrap();
        assert!(output.status.success(), "{:?}", output);
        let frame: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(frame["schema_version"], "fsfs.stream.live_search.v1");
        assert_eq!(frame["event"], "snapshot");
        assert_eq!(frame["sequence"], 1);
        assert_eq!(frame["result_count"], 1);
        assert_eq!(frame["results"][0]["doc_id"], "doc-a");
        assert_eq!(frame["results"][0]["item"]["path"], "/fixture/doc-a.rs");
        assert_eq!(std::fs::read(&pointer).unwrap(), before);
        for directory in ["no-models", "data", "cache", "config"] {
            assert!(
                !root.path().join(directory).exists(),
                "unexpected startup side effect: {directory}"
            );
        }
    }

    #[test]
    fn once_without_selection_is_a_failure_not_an_empty_snapshot() {
        let root = tempfile::tempdir().unwrap();
        let store = root.path().join("store");
        std::fs::create_dir(&store).unwrap();
        let output = run(root.path(), &live_args(&store, &["--once"])).unwrap();
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        let error: Value = serde_json::from_slice(&output.stderr).unwrap();
        assert_eq!(error["phase"], "subscription");
        assert_eq!(std::fs::read_dir(&store).unwrap().count(), 0);
    }

    #[test]
    fn waiting_for_an_initial_generation_honors_the_process_budget() {
        let root = tempfile::tempdir().unwrap();
        let store = root.path().join("store");
        std::fs::create_dir(&store).unwrap();
        let output = run(
            root.path(),
            &live_args(
                &store,
                &[
                    "--max-updates",
                    "1",
                    "--timeout-ms",
                    "50",
                    "--poll-ms",
                    "60000",
                ],
            ),
        )
        .unwrap();
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        let error: Value = serde_json::from_slice(&output.stderr).unwrap();
        assert_eq!(error["phase"], "subscription");
        assert!(error["message"].as_str().unwrap().contains("timed out"));
        assert_eq!(std::fs::read_dir(&store).unwrap().count(), 0);
    }

    #[test]
    fn hybrid_waits_for_publication_without_loading_missing_models_or_writing() {
        let root = tempfile::tempdir().unwrap();
        let store = root.path().join("store");
        std::fs::create_dir(&store).unwrap();
        let config = root.path().join("explicit.toml");
        std::fs::write(&config, "[indexing]\nwatch_mode = true\noffline = false\n").unwrap();
        let mut args = live_args(
            &store,
            &[
                "--hybrid",
                "--max-updates",
                "1",
                "--timeout-ms",
                "50",
                "--poll-ms",
                "60000",
            ],
        );
        args.push("--config".into());
        args.push(config.as_os_str().to_owned());
        let output = run(root.path(), &args).unwrap();
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        let error: Value = serde_json::from_slice(&output.stderr).unwrap();
        assert_eq!(error["phase"], "subscription");
        assert!(error["message"].as_str().unwrap().contains("timed out"));
        assert_eq!(std::fs::read_dir(&store).unwrap().count(), 0);
        for directory in ["no-models", "data", "cache", "config"] {
            assert!(!root.path().join(directory).exists());
        }
    }

    #[test]
    fn hybrid_configuration_requires_explicit_hybrid_mode() {
        let root = tempfile::tempdir().unwrap();
        let mut args = live_args(&root.path().join("not-created"), &["--once"]);
        args.extend(strings(&["--config", "not-opened.toml"]));
        let output = run(root.path(), &args).unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(output.stdout.is_empty());
        let error: Value = serde_json::from_slice(&output.stderr).unwrap();
        assert_eq!(error["phase"], "arguments");
        assert!(
            error["message"]
                .as_str()
                .unwrap()
                .contains("--config requires --hybrid")
        );
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
    }
}
