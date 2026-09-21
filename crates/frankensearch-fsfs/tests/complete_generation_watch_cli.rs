//! Actual-executable watch/read lifecycle, never the library's test embedder.
#![cfg(unix)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};

use frankensearch_fsfs::FsfsConfig;
use serde_json::Value;

struct Fixture {
    directory: tempfile::TempDir,
    source: PathBuf,
    store: PathBuf,
    config: PathBuf,
}

impl Fixture {
    fn new(models: Option<&Path>) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("source");
        let store = directory.path().join("store");
        let config = directory.path().join("fsfs.toml");
        fs::create_dir(&source).unwrap();
        fs::write(source.join("alpha.md"), "sharedtoken alpha document").unwrap();
        let mut settings = FsfsConfig::default();
        settings.storage.db_path = "{index_dir}/catalog.sqlite".to_owned();
        settings.indexing.offline = true;
        settings.indexing.watch_mode = false;
        settings.indexing.quality_model.clear();
        settings.search.fast_only = true;
        settings.search.rerank = false;
        if let Some(models) = models {
            settings.indexing.model_dir = models.display().to_string();
        }
        fs::write(&config, settings.to_toml().unwrap()).unwrap();
        Self { directory, source, store, config }
    }

    fn command(&self, command: &str, format: &str) -> Command {
        let mut child = Command::new(env!("CARGO_BIN_EXE_fsfs"));
        for (name, _) in std::env::vars_os() {
            let text = name.to_string_lossy();
            if text.starts_with("FSFS_") || text.starts_with("FRANKENSEARCH_") {
                child.env_remove(name);
            }
        }
        child.current_dir(self.directory.path())
            .env("FRANKENSEARCH_CHECK_UPDATES", "0")
            .arg(command).arg("--config").arg(&self.config)
            .arg("--index-dir").arg(&self.store)
            .args(["--quiet", "--format", format]);
        child
    }
}

/// Capture to files so an idle read or a full pipe cannot hang test teardown.
struct Process {
    child: Child,
    capture: tempfile::TempDir,
}

impl Process {
    fn start(command: &mut Command) -> Self {
        let capture = tempfile::tempdir().unwrap();
        let child = command.stdin(Stdio::null())
            .stdout(fs::File::create(capture.path().join("stdout")).unwrap())
            .stderr(fs::File::create(capture.path().join("stderr")).unwrap())
            .spawn().expect("start production fsfs");
        Self { child, capture }
    }

    fn stdout(&self) -> Vec<u8> {
        fs::read(self.capture.path().join("stdout")).unwrap()
    }

    #[cfg(feature = "semantic-support")]
    fn assert_alive(&mut self) {
        assert!(self.child.try_wait().unwrap().is_none(), "watch exited\n{}\n{}",
            String::from_utf8_lossy(&self.stdout()),
            fs::read_to_string(self.capture.path().join("stderr")).unwrap());
    }

    fn finish(&mut self, timeout: Duration) -> Output {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                return Output {
                    status,
                    stdout: self.stdout(),
                    stderr: fs::read(self.capture.path().join("stderr")).unwrap(),
                };
            }
            assert!(Instant::now() < deadline, "fsfs subprocess exceeded its deadline");
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
    }
}

#[test]
fn complete_watch_binary_refuses_unframed_output_before_creating_store() {
    for command in ["watch", "index"] {
        let fixture = Fixture::new(None);
        let mut invocation = fixture.command(command, "json");
        invocation.env("FSFS_COMPLETE_GENERATIONS", "1").arg(&fixture.source);
        if command == "index" {
            invocation.arg("--watch");
        }
        let output = Process::start(&mut invocation).finish(Duration::from_secs(10));
        assert!(!output.status.success());
        let result: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(result["ok"], false);
        assert!(result["error"].to_string().contains("watch_format"), "{result}");
        assert!(!fixture.store.exists());
    }
}

#[cfg(feature = "semantic-support")]
fn wait_for_visible_paths(fixture: &Fixture, watch: &mut Process, expected: &[&str]) -> PathBuf {
    let deadline = Instant::now() + Duration::from_secs(180);
    let mut last_generation = String::new();
    loop {
        watch.assert_alive();
        assert!(Instant::now() < deadline, "watch never made {expected:?} visible");
        // Ignore a partially written last line; only complete receipts admit a search.
        let bytes = watch.stdout();
        let receipt = bytes.split_inclusive(|byte| *byte == b'\n')
            .filter(|line| line.ends_with(b"\n"))
            .filter_map(|line| serde_json::from_slice::<Value>(line).ok())
            .filter(|value| value["ok"] == true && value["data"]["publication"] == "durable")
            .last();
        if let Some(receipt) = receipt {
            let generation = receipt["data"]["generation_id"].as_str().unwrap();
            if generation != last_generation {
                last_generation = generation.to_owned();
                let output = Process::start(
                    fixture.command("search", "json")
                        .args(["sharedtoken", "--no-daemon", "--limit", "10"]),
                ).finish(Duration::from_secs(60));
                watch.assert_alive();
                assert!(output.status.success(), "search while watcher alive failed: {}\n{}",
                    String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
                let result: Value = serde_json::from_slice(&output.stdout).unwrap();
                let hits = result["data"]["hits"].as_array().unwrap();
                if hits.len() == expected.len() && expected.iter().all(|name| {
                    hits.iter().any(|hit| Path::new(hit["path"].as_str().unwrap()).file_name()
                        .is_some_and(|filename| filename == std::ffi::OsStr::new(name)))
                }) {
                    return PathBuf::from(receipt["data"]["generation_path"].as_str().unwrap());
                }
            }
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[cfg(feature = "semantic-support")]
#[test]
#[ignore = "requires FSFS_COMPLETE_GENERATION_TEST_MODEL_DIR with a verified Potion cache"]
fn complete_watch_binary_keeps_search_available() {
    let models = fs::canonicalize(std::env::var_os("FSFS_COMPLETE_GENERATION_TEST_MODEL_DIR")
        .expect("provide a verified semantic model-cache root")).unwrap();
    let fixture = Fixture::new(Some(&models));
    let mut watch = Process::start(fixture.command("watch", "jsonl")
        .env("FSFS_COMPLETE_GENERATIONS", "1").arg(&fixture.source));
    let first = wait_for_visible_paths(&fixture, &mut watch, &["alpha.md"]);
    let original_manifest = fs::read(first.join("FSFS-BUNDLE.json")).unwrap();
    fs::write(fixture.source.join("beta.md"), "sharedtoken beta document").unwrap();
    let second = wait_for_visible_paths(&fixture, &mut watch, &["alpha.md", "beta.md"]);
    assert_ne!(first, second);
    fs::rename(fixture.source.join("alpha.md"), fixture.source.join("renamed.md")).unwrap();
    wait_for_visible_paths(&fixture, &mut watch, &["renamed.md", "beta.md"]);
    fs::remove_file(fixture.source.join("beta.md")).unwrap();
    wait_for_visible_paths(&fixture, &mut watch, &["renamed.md"]);
    watch.assert_alive();
    assert_eq!(fs::read(first.join("FSFS-BUNDLE.json")).unwrap(), original_manifest);
    assert!(second.is_dir());
    // Drop bounds cleanup; it is not a graceful-SIGINT assertion.
}
