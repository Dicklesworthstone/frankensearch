//! Fresh-process recovery command tests. Container durability tests require no
//! model; the ignored semantic workflow requires explicitly supplied real models.
#![cfg(unix)]

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};

use asupersync::Cx;
use asupersync::test_utils::run_test_with_cx;
use frankensearch_fsfs::generation_store::{
    COMPLETE_GENERATION_POINTER, CompleteGenerationStore, GenerationPublication,
    PublishedGeneration,
};
use frankensearch_fsfs::default_project_config_file_path;

struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn run(command: &mut Command, directory: &Path, timeout: Duration) -> Output {
    // File-backed capture avoids a full stdout/stderr pipe blocking the child
    // while the parent enforces its deadline. The guard joins on every exit.
    let stdout = tempfile::NamedTempFile::new_in(directory).unwrap();
    let stderr = tempfile::NamedTempFile::new_in(directory).unwrap();
    command.stdout(Stdio::from(stdout.reopen().unwrap()));
    command.stderr(Stdio::from(stderr.reopen().unwrap()));
    command.stdin(Stdio::null());
    let mut child = ChildGuard(command.spawn().unwrap());
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.0.try_wait().unwrap() {
            break status;
        }
        assert!(start.elapsed() < timeout, "recovery child exceeded its deadline");
        std::thread::sleep(Duration::from_millis(10));
    };
    Output {
        status,
        stdout: fs::read(stdout.path()).unwrap(),
        stderr: fs::read(stderr.path()).unwrap(),
    }
}

fn recover(directory: &Path, store: &Path, id: &str, digest: &str) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_fsfs-recover"));
    command.current_dir(directory)
        .arg("--index-dir").arg(store)
        .args(["--generation", id, "--manifest-sha256", digest])
        .env("HOME", directory)
        .env("XDG_CONFIG_HOME", directory.join("empty-config"))
        .env("FRANKENSEARCH_MODEL_DIR", directory.join("missing-models"));
    command
}

fn body(output: &Output) -> serde_json::Value {
    serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!("invalid CLI JSON: {error}; stderr={}", String::from_utf8_lossy(&output.stderr))
    })
}

fn published(store: &CompleteGenerationStore, cx: &Cx, bytes: &[u8]) -> PublishedGeneration {
    let build = store.begin(cx).unwrap();
    fs::write(build.path().join("payload"), bytes).unwrap();
    let GenerationPublication::Durable(generation) = build.publish(cx, |_, _| Ok(())).unwrap()
    else {
        panic!("container fixture must publish durably");
    };
    generation
}

fn files(root: &Path) -> BTreeMap<std::path::PathBuf, Vec<u8>> {
    let mut files = BTreeMap::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(directory).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_dir() {
                pending.push(entry.path());
            } else {
                files.insert(entry.path().strip_prefix(root).unwrap().to_path_buf(), fs::read(entry.path()).unwrap());
            }
        }
    }
    files
}

#[test]
fn command_help_and_usage_errors_do_not_create_a_store() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("absent-store");
    let mut help = Command::new(env!("CARGO_BIN_EXE_fsfs-recover"));
    help.arg("--help").current_dir(directory.path());
    let output = run(&mut help, directory.path(), Duration::from_secs(10));
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("--confirm-durability"));
    for suffix in [vec!["--apply", "--apply"], vec!["--apply", "--confirm-durability"], vec!["--unknown"]] {
        let id = format!("g-{}-{}-{}", "a".repeat(32), "b".repeat(8), "c".repeat(16));
        let mut command = recover(directory.path(), &root, &id, &"a".repeat(64));
        command.args(suffix);
        let output = run(&mut command, directory.path(), Duration::from_secs(10));
        assert!(!output.status.success());
        assert_eq!(body(&output)["ok"], false);
        assert!(!root.exists());
    }
}

#[test]
fn confirmation_survives_invalid_config_and_missing_models_without_republication() {
    run_test_with_cx(|cx| async move {
        use std::os::unix::fs::MetadataExt;
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("store");
        let store = CompleteGenerationStore::create(&cx, &root).unwrap();
        let old = published(&store, &cx, b"old");
        let current = published(&store, &cx, b"current");
        let config = default_project_config_file_path(directory.path());
        fs::create_dir_all(config.parent().unwrap()).unwrap();
        fs::write(&config, "intentionally invalid = [").unwrap();
        let pointer = root.join(COMPLETE_GENERATION_POINTER);
        let before = fs::read(&pointer).unwrap();
        let inode = fs::metadata(&pointer).unwrap().ino();
        let old_files = files(old.path());
        let current_files = files(current.path());
        let mut command = recover(directory.path(), &root, current.id(), current.manifest_sha256());
        command.arg("--confirm-durability");
        let output = run(&mut command, directory.path(), Duration::from_secs(20));
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        let result = body(&output);
        assert_eq!(result["data"]["state"], "durability_confirmed");
        assert_eq!(result["data"]["generation_id"], current.id());
        assert_eq!(result["data"]["selection_write_performed"], false);
        assert_eq!(result["data"]["producer_admission_checked"], false);
        let mut stale = recover(directory.path(), &root, old.id(), old.manifest_sha256());
        stale.arg("--confirm-durability");
        let output = run(&mut stale, directory.path(), Duration::from_secs(20));
        assert!(!output.status.success());
        assert_eq!(body(&output)["ok"], false);
        assert_eq!(fs::read(&pointer).unwrap(), before);
        assert_eq!(fs::metadata(&pointer).unwrap().ino(), inode);
        assert_eq!(files(old.path()), old_files);
        assert_eq!(files(current.path()), current_files);
        assert!(!directory.path().join("missing-models").exists());
    });
}

#[test]
fn confirmation_rejects_corrupt_selection_and_forged_digest_without_repair() {
    run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("store");
        let store = CompleteGenerationStore::create(&cx, &root).unwrap();
        let target = published(&store, &cx, b"target");
        let pointer = root.join(COMPLETE_GENERATION_POINTER);
        fs::write(&pointer, b"damaged selection").unwrap();
        for digest in [target.manifest_sha256().to_owned(), "f".repeat(64)] {
            let mut command = recover(directory.path(), &root, target.id(), &digest);
            command.arg("--confirm-durability");
            let output = run(&mut command, directory.path(), Duration::from_secs(20));
            assert!(!output.status.success());
            assert_eq!(body(&output)["ok"], false);
            assert_eq!(fs::read(&pointer).unwrap(), b"damaged selection");
            assert_eq!(fs::read(target.path().join("payload")).unwrap(), b"target");
        }
    });
}

#[cfg(feature = "semantic-loaders")]
#[test]
#[ignore = "requires FRANKENSEARCH_RECOVERY_E2E_MODEL_DIR with verified Potion and MiniLM models"]
fn genuine_models_preview_restore_and_fresh_search_without_source_files() {
    use frankensearch_fsfs::FsfsConfig;
    use frankensearch_fsfs::output_schema::SearchOutputPhase;

    let models = std::env::var_os("FRANKENSEARCH_RECOVERY_E2E_MODEL_DIR")
        .map(std::path::PathBuf::from)
        .expect("explicit verified real-model directory is required; no mock/skip fallback");
    let models = fs::canonicalize(models).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source");
    let root = directory.path().join("store");
    fs::create_dir(&source).unwrap();
    fs::write(source.join("alpha.md"), "recoverytoken preserves durable searchable generations and the original documents").unwrap();
    let mut config = FsfsConfig::default();
    config.indexing.model_dir = models.display().to_string();
    config.indexing.offline = true;
    config.search.fast_only = false;
    config.search.rerank = false;
    config.search.quality_timeout_ms = 10_000;
    "{index_dir}/catalog.sqlite".clone_into(&mut config.storage.db_path);
    let config_path = directory.path().join("recovery.toml");
    fs::write(&config_path, config.to_toml().unwrap()).unwrap();
    let index = || {
        let mut command = Command::new(env!("CARGO_BIN_EXE_fsfs"));
        command.current_dir(directory.path()).arg("index").arg(&source)
            .arg("--index-dir").arg(&root).arg("--config").arg(&config_path)
            .args(["--format", "json", "--quiet"])
            .env("FRANKENSEARCH_COMPLETE_GENERATIONS", "1")
            .env("FRANKENSEARCH_CHECK_UPDATES", "0")
            .env("FRANKENSEARCH_MODEL_DIR", &models);
        let output = run(&mut command, directory.path(), Duration::from_secs(180));
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        body(&output)["data"].clone()
    };
    let first = index();
    fs::write(source.join("beta.md"), "recoverytoken searchable successor document").unwrap();
    let second = index();
    let old_root = std::path::PathBuf::from(first["generation_path"].as_str().unwrap());
    let new_root = std::path::PathBuf::from(second["generation_path"].as_str().unwrap());
    let old_bytes = files(&old_root);
    let new_bytes = files(&new_root);
    fs::rename(&source, directory.path().join("source-unavailable")).unwrap();
    let pointer = root.join(COMPLETE_GENERATION_POINTER);
    fs::write(&pointer, b"corrupt selection").unwrap();
    let id = first["generation_id"].as_str().unwrap();
    let digest = first["manifest_sha256"].as_str().unwrap();
    let query = "how does recoverytoken preserve durable search generations";
    for apply in [false, true] {
        let mut command = recover(directory.path(), &root, id, digest);
        command.arg("--config").arg(&config_path).args(["--query", query])
            .env("FRANKENSEARCH_MODEL_DIR", &models);
        if apply { command.arg("--apply"); }
        let output = run(&mut command, directory.path(), Duration::from_secs(180));
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        let result = body(&output);
        assert_eq!(result["data"]["generation_id"], id);
        assert_eq!(result["data"]["selection_write_performed"], apply);
        assert_eq!(result["data"]["producer_admission_checked"], true);
        let phases = result["data"]["preview"].as_array().unwrap();
        let refined = serde_json::to_value(SearchOutputPhase::Refined).unwrap();
        assert!(phases.iter().any(|phase| phase["phase"] == refined));
        assert_eq!(phases.last().unwrap()["hits"].as_array().unwrap().len(), 1);
        if !apply { assert_eq!(fs::read(&pointer).unwrap(), b"corrupt selection"); }
        assert_eq!(files(&old_root), old_bytes);
        assert_eq!(files(&new_root), new_bytes);
    }
    let mut search = Command::new(env!("CARGO_BIN_EXE_fsfs"));
    search.current_dir(directory.path()).args(["search", query])
        .arg("--index-dir").arg(&root).arg("--config").arg(&config_path)
        .args(["--format", "json", "--quiet"])
        .env("FRANKENSEARCH_CHECK_UPDATES", "0")
        .env("FRANKENSEARCH_MODEL_DIR", &models);
    let output = run(&mut search, directory.path(), Duration::from_secs(180));
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(body(&output)["data"]["hits"].as_array().unwrap().len(), 1);
    let selected = fs::read(&pointer).unwrap();
    let mut confirm = recover(directory.path(), &root, id, digest);
    confirm.arg("--confirm-durability");
    let output = run(&mut confirm, directory.path(), Duration::from_secs(20));
    assert!(output.status.success());
    assert_eq!(body(&output)["data"]["durability_confirmed"], true);
    assert_eq!(fs::read(&pointer).unwrap(), selected);
    assert_eq!(files(&old_root), old_bytes);
    assert_eq!(files(&new_root), new_bytes);
}
