//! Actual command dispatch and the opt-in watch -> publish -> query workflow.
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
    fn new(models: Option<&Path>, shadow: bool) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("source");
        let store = directory.path().join("store");
        let config = directory.path().join("fsfs.toml");
        fs::create_dir(&source).unwrap();
        fs::write(source.join("alpha.md"), "sharedtoken alpha document").unwrap();
        let mut settings = FsfsConfig::default();
        "{index_dir}/catalog.sqlite".clone_into(&mut settings.storage.db_path);
        settings.indexing.offline = true;
        settings.indexing.quality_model.clear();
        settings.search.fast_only = true;
        settings.search.rerank = false;
        settings.search.shadow_mode = shadow;
        if let Some(models) = models {
            settings.indexing.model_dir = models.display().to_string();
        }
        fs::write(&config, settings.to_toml().unwrap()).unwrap();
        Self {
            directory,
            source,
            store,
            config,
        }
    }

    fn command(&self, source: &Path, store: &Path) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_fsfs"));
        for (name, _) in std::env::vars_os() {
            let text = name.to_string_lossy();
            if text.starts_with("FSFS_") || text.starts_with("FRANKENSEARCH_") {
                command.env_remove(name);
            }
        }
        command
            .current_dir(self.directory.path())
            .env("HOME", self.directory.path())
            .env("XDG_CONFIG_HOME", self.directory.path().join("config"))
            .env("XDG_DATA_HOME", self.directory.path().join("data"))
            .env("XDG_CACHE_HOME", self.directory.path().join("cache"))
            .env("FRANKENSEARCH_CHECK_UPDATES", "0")
            .args([
                "live-search", "--hybrid", "--query", "sharedtoken", "--config",
            ])
            .arg(&self.config)
            .arg("--watch-source")
            .arg(source)
            .arg("--index-dir")
            .arg(store);
        command
    }
}

/// Files avoid pipe-buffer deadlock and let the test inspect only complete frames.
struct Process {
    child: Child,
    capture: tempfile::TempDir,
}

impl Process {
    fn start(command: &mut Command) -> Self {
        let capture = tempfile::tempdir().unwrap();
        let child = command
            .stdin(Stdio::null())
            .stdout(fs::File::create(capture.path().join("stdout")).unwrap())
            .stderr(fs::File::create(capture.path().join("stderr")).unwrap())
            .spawn()
            .expect("start production fsfs");
        Self { child, capture }
    }

    fn finish(&mut self, timeout: Duration) -> Output {
        let started = Instant::now();
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                return Output {
                    status,
                    stdout: fs::read(self.capture.path().join("stdout")).unwrap(),
                    stderr: fs::read(self.capture.path().join("stderr")).unwrap(),
                };
            }
            assert!(
                started.elapsed() < timeout,
                "watch-search child exceeded watchdog"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    #[cfg(feature = "semantic-support")]
    fn wait_for_first_frame(&mut self) -> Value {
        let started = Instant::now();
        loop {
            let bytes = fs::read(self.capture.path().join("stdout")).unwrap();
            if let Some(line) = bytes
                .split_inclusive(|byte| *byte == b'\n')
                .find(|line| line.ends_with(b"\n"))
            {
                return serde_json::from_slice(line).expect("a complete live NDJSON frame");
            }
            assert!(
                self.child.try_wait().unwrap().is_none(),
                "{}",
                fs::read_to_string(self.capture.path().join("stderr")).unwrap()
            );
            assert!(
                started.elapsed() < Duration::from_secs(180),
                "initial publication timed out"
            );
            std::thread::sleep(Duration::from_millis(25));
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

fn assert_preflight_failure(output: &Output, expected: &str) {
    assert!(!output.status.success());
    assert!(
        output.stdout.is_empty(),
        "preflight polluted the result stream"
    );
    let error: Value = serde_json::from_slice(&output.stderr).unwrap();
    assert_eq!(error["phase"], "watch_subscription");
    assert!(
        error["message"].as_str().unwrap().contains(expected),
        "{error}"
    );
}

#[test]
fn overlap_refusal_is_dispatched_before_indexing_and_preserves_source() {
    let fixture = Fixture::new(None, false);
    let overlap = fixture.source.join("must-not-be-created");
    let output = Process::start(fixture.command(&fixture.source, &overlap).arg("--once"))
        .finish(Duration::from_secs(15));
    assert_preflight_failure(&output, "overlap");
    assert!(!overlap.exists());
    assert_eq!(
        fs::read_to_string(fixture.source.join("alpha.md")).unwrap(),
        "sharedtoken alpha document"
    );
}

#[test]
fn a_missing_source_is_not_replaced_with_default_discovery() {
    let fixture = Fixture::new(None, false);
    let missing = fixture.directory.path().join("missing-source");
    let output = Process::start(fixture.command(&missing, &fixture.store).arg("--once"))
        .finish(Duration::from_secs(15));
    assert_preflight_failure(&output, "I/O");
    assert!(!missing.exists());
    assert!(!fixture.store.exists());
}

#[test]
fn shadow_policy_is_rejected_before_a_store_is_created() {
    let fixture = Fixture::new(None, true);
    let output = Process::start(fixture.command(&fixture.source, &fixture.store).arg("--once"))
        .finish(Duration::from_secs(15));
    assert_preflight_failure(&output, "shadow");
    assert!(!fixture.store.exists());
}

#[test]
fn watch_cannot_be_enabled_by_a_lexical_only_invocation() {
    let fixture = Fixture::new(None, false);
    let mut command = Command::new(env!("CARGO_BIN_EXE_fsfs"));
    command
        .args(["live-search", "--query", "sharedtoken", "--watch-source"])
        .arg(&fixture.source)
        .arg("--index-dir")
        .arg(&fixture.store);
    let output = Process::start(&mut command).finish(Duration::from_secs(15));
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    let error: Value = serde_json::from_slice(&output.stderr).unwrap();
    assert_eq!(error["phase"], "arguments");
    assert!(
        error["message"]
            .as_str()
            .unwrap()
            .contains("requires --hybrid")
    );
    assert!(!fixture.store.exists());
}

#[cfg(feature = "semantic-support")]
#[test]
#[ignore = "requires FSFS_COMPLETE_GENERATION_TEST_MODEL_DIR with a verified Potion cache"]
fn native_watcher_publishes_and_streams_a_renamed_file_then_stops_at_the_limit() {
    let models = fs::canonicalize(
        std::env::var_os("FSFS_COMPLETE_GENERATION_TEST_MODEL_DIR")
            .expect("provide a verified local semantic model cache"),
    )
    .unwrap();
    let fixture = Fixture::new(Some(&models), false);
    let mut process = Process::start(
        fixture
            .command(&fixture.source, &fixture.store)
            .args(["--max-updates", "2", "--timeout-ms", "300000"]),
    );
    let first = process.wait_for_first_frame();
    assert_eq!(first["schema_version"], "fsfs.stream.live_search.hybrid.v1");
    assert_eq!(first["phase"], "initial");
    assert_eq!(first["update"]["event"], "snapshot");
    let results = first["update"]["results"].as_array().unwrap();
    assert_eq!(results.len(), 1);
    let old_id = results[0]["doc_id"].as_str().unwrap();
    assert_eq!(Path::new(old_id).file_name().unwrap(), "alpha.md");
    let predecessor = fixture
        .store
        .join("generations")
        .join(first["generation_id"].as_str().unwrap());
    let manifest = fs::read(predecessor.join("FSFS-BUNDLE.json")).unwrap();

    // An actual filesystem rename drives native notifications and a real rebuild.
    // Do not fabricate a second receipt or invoke the query API from the test.
    fs::rename(
        fixture.source.join("alpha.md"),
        fixture.source.join("renamed.md"),
    )
    .unwrap();
    let output = process.finish(Duration::from_secs(180));
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let frames: Vec<Value> = output
        .stdout
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice(line).unwrap())
        .collect();
    // fast_only and no quality model: each completed generation has one Initial.
    assert_eq!(frames.len(), 2);
    let second = &frames[1];
    assert_ne!(second["generation_id"], first["generation_id"]);
    assert_eq!(second["phase"], "initial");
    assert_eq!(
        second["update"]["previous_generation"],
        first["update"]["generation"]
    );
    let changes = second["update"]["changes"].as_array().unwrap();
    assert!(changes.iter().any(|change| {
        change["change"] == "removed" && change["doc_id"] == old_id
    }));
    assert!(changes.iter().any(|change| {
        change["change"] == "added"
            && Path::new(change["result"]["doc_id"].as_str().unwrap())
                .file_name()
                .unwrap()
                == "renamed.md"
    }));
    assert_eq!(
        fs::read(predecessor.join("FSFS-BUNDLE.json")).unwrap(),
        manifest
    );
    let selected = fs::read(fixture.store.join("FSFS-CURRENT")).unwrap();
    fs::write(fixture.source.join("after-exit.md"), "sharedtoken after exit").unwrap();
    std::thread::sleep(Duration::from_millis(750));
    assert_eq!(
        fs::read(fixture.store.join("FSFS-CURRENT")).unwrap(),
        selected,
        "no detached publisher may survive the completed command"
    );
}
