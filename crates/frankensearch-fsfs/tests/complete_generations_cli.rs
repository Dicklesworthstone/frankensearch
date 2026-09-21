//! Production-executable coverage for complete-generation routing.
//!
//! Refusal cases do not need models. The positive case is explicitly ignored
//! unless a verified semantic cache is provisioned; it never enables the
//! library's cfg(test) hash embedder inside the shipping binary.
#![cfg(unix)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

use frankensearch_fsfs::FsfsConfig;
use frankensearch_fsfs::generation_store::COMPLETE_GENERATION_POINTER;
use serde_json::Value;

struct Fixture {
    directory: tempfile::TempDir,
    source: PathBuf,
    store: PathBuf,
    config: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().expect("fixture");
        let source = directory.path().join("source");
        let store = directory.path().join("store");
        let config = directory.path().join("fsfs.toml");
        fs::create_dir(&source).expect("source");
        fs::write(source.join("alpha.md"), "sharedtoken alpha document").expect("source file");
        let fixture = Self {
            directory,
            source,
            store,
            config,
        };
        fixture.write_config(None, "{index_dir}/catalog.sqlite");
        fixture
    }

    fn write_config(&self, model_dir: Option<&Path>, catalog: &str) {
        let mut config = FsfsConfig::default();
        config.indexing.offline = true;
        config.indexing.watch_mode = false;
        config.indexing.quality_model.clear();
        config.search.fast_only = true;
        config.search.rerank = false;
        config.storage.db_path = catalog.to_owned();
        if let Some(path) = model_dir {
            config.indexing.model_dir = path.display().to_string();
        }
        fs::write(&self.config, toml::to_string_pretty(&config).expect("TOML"))
            .expect("fixture config");
    }

    fn command(&self, command: &str, format: &str) -> Command {
        let mut child = Command::new(env!("CARGO_BIN_EXE_fsfs"));
        // Remove only application overrides from this child's environment.
        // Preserve PATH, shared-library configuration, and the parent process.
        for (key, _) in std::env::vars_os() {
            let name = key.to_string_lossy();
            if name.starts_with("FSFS_") || name.starts_with("FRANKENSEARCH_") {
                child.env_remove(key);
            }
        }
        child
            .current_dir(self.directory.path())
            .env("FRANKENSEARCH_CHECK_UPDATES", "0")
            .arg(command)
            .arg("--config")
            .arg(&self.config)
            .arg("--index-dir")
            .arg(&self.store)
            .args(["--format", format, "--quiet"]);
        child
    }
}

fn execute(command: &mut Command) -> Output {
    let capture = tempfile::tempdir().expect("capture");
    let stdout_path = capture.path().join("stdout");
    let stderr_path = capture.path().join("stderr");
    let mut child = command
        .stdin(Stdio::null())
        .stdout(fs::File::create(&stdout_path).expect("stdout"))
        .stderr(fs::File::create(&stderr_path).expect("stderr"))
        .spawn()
        .expect("spawn production fsfs");
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().expect("poll fsfs") {
            break status;
        }
        if started.elapsed() >= Duration::from_secs(180) {
            let _ = child.kill();
            let _ = child.wait();
            panic!(
                "fsfs timed out\nstdout: {}\nstderr: {}",
                fs::read_to_string(&stdout_path).unwrap_or_default(),
                fs::read_to_string(&stderr_path).unwrap_or_default(),
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    Output {
        status,
        stdout: fs::read(stdout_path).expect("captured stdout"),
        stderr: fs::read(stderr_path).expect("captured stderr"),
    }
}

fn json_output(output: &Output, success: bool) -> Value {
    assert_eq!(
        output.status.success(),
        success,
        "status: {}\nstdout: {}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    let value: Value = serde_json::from_slice(&output.stdout).expect("one JSON envelope");
    assert_eq!(value["ok"], success);
    value
}

#[test]
fn complete_cli_binary_rejects_corrupt_selection_without_legacy_fallback() {
    let fixture = Fixture::new();
    fs::create_dir(&fixture.store).unwrap();
    let pointer = fixture.store.join(COMPLETE_GENERATION_POINTER);
    fs::write(&pointer, "corrupt complete selection").unwrap();
    // No opt-in variable: the executable must still dispatch to the store.
    let output = execute(
        fixture
            .command("search", "json")
            .args(["sharedtoken", "--no-daemon"]),
    );
    let value = json_output(&output, false);
    assert!(value["error"].to_string().contains("complete-generation pointer"));
    assert_eq!(fs::read_to_string(pointer).unwrap(), "corrupt complete selection");
    assert!(!fixture.store.join("vector").exists());
}

#[test]
fn complete_cli_binary_retains_interrupted_store_without_selection() {
    let fixture = Fixture::new();
    fs::create_dir_all(fixture.store.join("generations/unfinished")).unwrap();
    let artifact = fixture.store.join("generations/unfinished/partial.idx");
    fs::write(&artifact, "abandoned build evidence").unwrap();
    let output = execute(
        fixture
            .command("search", "json")
            .env("FSFS_COMPLETE_GENERATIONS", "false")
            .args(["sharedtoken", "--no-daemon"]),
    );
    let value = json_output(&output, false);
    assert!(value["error"].to_string().contains("no complete generation"));
    assert_eq!(fs::read_to_string(artifact).unwrap(), "abandoned build evidence");
    assert!(!fixture.store.join(COMPLETE_GENERATION_POINTER).exists());
}

#[test]
fn complete_cli_binary_opt_in_refuses_catalog_escape_before_creating_store() {
    let fixture = Fixture::new();
    fixture.write_config(None, "{index_dir}/../outside.sqlite");
    let outside = fixture.directory.path().join("outside.sqlite");
    fs::write(&outside, "unrelated database").unwrap();
    let output = execute(
        fixture
            .command("index", "json")
            .env("FRANKENSEARCH_COMPLETE_GENERATIONS", "true")
            .arg(&fixture.source),
    );
    let value = json_output(&output, false);
    assert!(value["error"].to_string().contains("without parent traversal"));
    assert!(!fixture.store.exists());
    assert_eq!(fs::read_to_string(outside).unwrap(), "unrelated database");
}

#[test]
fn complete_cli_binary_rejects_malformed_opt_in_before_writing() {
    let fixture = Fixture::new();
    let output = execute(
        fixture
            .command("index", "json")
            .env("FSFS_COMPLETE_GENERATIONS", "not-a-boolean")
            .arg(&fixture.source),
    );
    let value = json_output(&output, false);
    assert!(value["error"].to_string().contains("FSFS_COMPLETE_GENERATIONS"));
    assert!(!fixture.store.exists());
}

#[cfg(feature = "semantic-support")]
#[test]
#[ignore = "requires FSFS_COMPLETE_GENERATION_TEST_MODEL_DIR with a verified Potion cache"]
fn complete_cli_binary_semantic_rebuild_search_and_stream() {
    let models = PathBuf::from(
        std::env::var_os("FSFS_COMPLETE_GENERATION_TEST_MODEL_DIR")
            .expect("provide the verified semantic model-cache root"),
    );
    let models = fs::canonicalize(models).expect("model-cache root");
    let fixture = Fixture::new();
    fixture.write_config(Some(&models), "{index_dir}/catalog.sqlite");
    let first = json_output(
        &execute(
            fixture
                .command("index", "json")
                .env("FSFS_COMPLETE_GENERATIONS", "1")
                .arg(&fixture.source),
        ),
        true,
    );
    assert_eq!(first["data"]["publication"], "durable");
    let first_path = PathBuf::from(first["data"]["generation_path"].as_str().unwrap());
    let manifest = first_path.join("FSFS-BUNDLE.json");
    let first_manifest = fs::read(&manifest).unwrap();
    fs::write(fixture.source.join("beta.md"), "sharedtoken beta document").unwrap();
    // A successor needs no opt-in: the existing layout chooses the route.
    let second = json_output(
        &execute(fixture.command("index", "json").arg(&fixture.source)),
        true,
    );
    assert_ne!(first["data"]["generation_id"], second["data"]["generation_id"]);
    assert!(first_path.is_dir());
    assert_eq!(fs::read(manifest).unwrap(), first_manifest);
    let result = json_output(
        &execute(
            fixture
                .command("search", "json")
                .args(["sharedtoken", "--no-daemon"]),
        ),
        true,
    );
    let hits = result["data"]["hits"].as_array().expect("search hits");
    assert_eq!(hits.len(), 2);
    for filename in ["alpha.md", "beta.md"] {
        assert!(hits.iter().any(|hit| hit["path"].as_str().unwrap().ends_with(filename)));
    }
    let stream = execute(
        fixture
            .command("search", "jsonl")
            .args(["sharedtoken", "--no-daemon", "--stream"]),
    );
    assert!(stream.status.success(), "{}", String::from_utf8_lossy(&stream.stderr));
    let frames = String::from_utf8(stream.stdout).unwrap();
    let frames = frames
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("stream frame"))
        .collect::<Vec<_>>();
    assert!(frames.len() > 2, "started, results and terminal are required");
    assert!(frames.iter().all(|frame| frame.get("data").is_none()));
}
