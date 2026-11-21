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
use frankensearch_fsfs::default_project_config_file_path;
use frankensearch_fsfs::generation_store::{
    COMPLETE_GENERATION_POINTER, CompleteGenerationStore, GenerationPublication,
    PublishedGeneration,
};

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
        assert!(
            start.elapsed() < timeout,
            "recovery child exceeded its deadline"
        );
        std::thread::sleep(Duration::from_millis(10));
    };
    Output {
        status,
        stdout: fs::read(stdout.path()).unwrap(),
        stderr: fs::read(stderr.path()).unwrap(),
    }
}

#[derive(Clone, Copy, Debug)]
enum RecoveryEntry {
    Standalone,
    Installed,
}

impl RecoveryEntry {
    fn command(self) -> Command {
        match self {
            Self::Standalone => Command::new(env!("CARGO_BIN_EXE_fsfs-recover")),
            Self::Installed => {
                let mut command = Command::new(env!("CARGO_BIN_EXE_fsfs"));
                command.arg("recover");
                command
            }
        }
    }
}

fn recover(directory: &Path, store: &Path, id: &str, digest: &str) -> Command {
    recover_with_entry(RecoveryEntry::Standalone, directory, store, id, digest)
}

fn recover_with_entry(
    entry: RecoveryEntry,
    directory: &Path,
    store: &Path,
    id: &str,
    digest: &str,
) -> Command {
    let mut command = entry.command();
    command
        .current_dir(directory)
        .arg("--index-dir")
        .arg(store)
        .args(["--generation", id, "--manifest-sha256", digest])
        .env("HOME", directory)
        .env("XDG_CONFIG_HOME", directory.join("empty-config"))
        .env("FRANKENSEARCH_MODEL_DIR", directory.join("missing-models"));
    command
}

fn body(output: &Output) -> serde_json::Value {
    serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "invalid CLI JSON: {error}; stderr={}",
            String::from_utf8_lossy(&output.stderr)
        )
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
                files.insert(
                    entry.path().strip_prefix(root).unwrap().to_path_buf(),
                    fs::read(entry.path()).unwrap(),
                );
            }
        }
    }
    files
}

#[test]
fn command_help_and_usage_errors_do_not_create_a_store() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("absent-store");
    for entry in [RecoveryEntry::Standalone, RecoveryEntry::Installed] {
        let mut help = entry.command();
        help.arg("--help").current_dir(directory.path());
        let output = run(&mut help, directory.path(), Duration::from_secs(10));
        assert!(output.status.success(), "{entry:?}");
        let text = String::from_utf8_lossy(&output.stdout);
        assert!(text.contains("fsfs recover --index-dir STORE"));
        assert!(text.contains("--confirm-durability"));
        for suffix in [
            vec!["--apply", "--apply"],
            vec!["--apply", "--confirm-durability"],
            vec!["--unknown"],
        ] {
            let id = format!("g-{}-{}-{}", "a".repeat(32), "b".repeat(8), "c".repeat(16));
            let mut command =
                recover_with_entry(entry, directory.path(), &root, &id, &"a".repeat(64));
            command.args(suffix);
            let output = run(&mut command, directory.path(), Duration::from_secs(10));
            assert!(!output.status.success(), "{entry:?}");
            // Parsing the entire stdout also rejects an appended ordinary
            // fsfs error envelope after recovery has emitted its own result.
            assert_eq!(body(&output)["ok"], false);
            assert!(!root.exists());
        }
    }
}

#[test]
fn installed_help_advertises_recovery_and_non_utf8_arguments_return_typed_errors() {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;

    let directory = tempfile::tempdir().unwrap();
    let mut help = Command::new(env!("CARGO_BIN_EXE_fsfs"));
    help.arg("help")
        .current_dir(directory.path())
        .env("HOME", directory.path())
        .env("XDG_CONFIG_HOME", directory.path().join("empty-config"))
        .env("FRANKENSEARCH_CHECK_UPDATES", "0");
    let output = run(&mut help, directory.path(), Duration::from_secs(10));
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("recover [options]"));

    let mut invalid = RecoveryEntry::Installed.command();
    invalid
        .current_dir(directory.path())
        .arg(OsString::from_vec(vec![0xff]));
    let output = run(&mut invalid, directory.path(), Duration::from_secs(10));
    assert!(!output.status.success());
    assert_eq!(body(&output)["ok"], false);
    assert!(!String::from_utf8_lossy(&output.stderr).contains("panicked"));
}

#[test]
fn artifact_repair_prepares_and_applies_real_keyword_data_without_sources() {
    exercise_keyword_repair(true);
}

#[test]
fn artifact_repair_refuses_unprotected_stock_keyword_data_without_changing_selection() {
    exercise_keyword_repair(false);
}

fn exercise_keyword_repair(protect: bool) {
    use frankensearch_durability::{DefaultSymbolCodec, DurabilityConfig, FileProtector};
    use frankensearch_fsfs::config::DegradationOverrideMode;
    use frankensearch_fsfs::generation_store::COMPLETE_GENERATION_MANIFEST;
    use frankensearch_fsfs::{FsCategory, FsfsConfig, MountTable, read_system_mounts};

    run_test_with_cx(|cx| async move {
        // Production source discovery intentionally refuses virtual mounts.
        // Choose an admitted fixture mount without changing that policy.
        let mounts = MountTable::new(read_system_mounts(), &std::collections::HashMap::new());
        let mut parent = fs::canonicalize(std::env::temp_dir()).unwrap();
        if mounts
            .lookup(&parent)
            .is_some_and(|(entry, _)| entry.category.is_virtual())
        {
            parent = fs::canonicalize("/dev/shm").unwrap();
            assert!(mounts.lookup(&parent).is_some_and(|(entry, policy)| {
                entry.category == FsCategory::Memory && policy.enabled && policy.available
            }));
        }
        let directory = tempfile::Builder::new()
            .prefix("fsfs-repair-")
            .tempdir_in(parent)
            .unwrap();
        let root = directory.path().join("store");
        let source = directory.path().join("source");
        let config_path = directory.path().join("repair.toml");
        let preparation = directory.path().join("prepared.json");
        fs::create_dir(&source).unwrap();
        fs::write(
            source.join("alpha.md"),
            "repairquerytoken alpha durable document",
        )
        .unwrap();
        fs::write(
            source.join("beta.md"),
            "repairquerytoken beta searchable document",
        )
        .unwrap();
        let mut config = FsfsConfig::default();
        config.pressure.degradation_override = DegradationOverrideMode::ForceLexicalOnly;
        config.indexing.offline = true;
        config.indexing.watch_mode = false;
        config.indexing.quality_model.clear();
        config.search.fast_only = true;
        config.search.rerank = false;
        "{index_dir}/catalog.sqlite".clone_into(&mut config.storage.db_path);
        fs::write(&config_path, config.to_toml().unwrap()).unwrap();
        let configure = |command: &mut Command, index_dir: &Path| {
            for (key, _) in std::env::vars_os() {
                let name = key.to_string_lossy();
                if name.starts_with("FSFS_") || name.starts_with("FRANKENSEARCH_") {
                    command.env_remove(key);
                }
            }
            command
                .current_dir(directory.path())
                .arg("--index-dir")
                .arg(index_dir)
                .arg("--config")
                .arg(&config_path)
                .env("HOME", directory.path())
                .env("XDG_CONFIG_HOME", directory.path().join("empty-config"))
                .env(
                    "FRANKENSEARCH_MODEL_DIR",
                    directory.path().join("missing-models"),
                )
                .env("FRANKENSEARCH_CHECK_UPDATES", "0")
                .env("FSFS_GENERATION_RETENTION", "off");
        };
        let mut index = Command::new(env!("CARGO_BIN_EXE_fsfs"));
        index.arg("index").arg(&source);
        // Keep the healthy template outside the protected recovery store, so
        // repair has no intact predecessor to select or copy.
        let stock_root = if protect {
            directory.path().join("stock-template")
        } else {
            root.clone()
        };
        configure(&mut index, &stock_root);
        index
            .args(["--no-watch-mode", "--format", "json", "--quiet"])
            .env("FSFS_COMPLETE_GENERATIONS", "1");
        let indexed = run(&mut index, directory.path(), Duration::from_secs(120));
        assert!(
            indexed.status.success(),
            "{}",
            String::from_utf8_lossy(&indexed.stderr)
        );
        assert_eq!(body(&indexed)["data"]["indexed_files"], 2);
        let stock = CompleteGenerationStore::open(&cx, &stock_root)
            .unwrap()
            .active(&cx)
            .unwrap()
            .unwrap();
        let stock_files = files(stock.path());
        let segment = stock_files
            .keys()
            .find(|path| {
                path.extension()
                    .is_some_and(|extension| extension == "fslx")
            })
            .unwrap()
            .clone();
        let manifest = stock_files
            .keys()
            .find(|path| path.file_name().is_some_and(|name| name == "MANIFEST"))
            .unwrap()
            .clone();
        for relative in [&segment, &manifest] {
            assert!(!stock_files.contains_key(&FileProtector::sidecar_path(relative)));
        }
        let search_command = |index_dir: &Path| {
            let mut search = Command::new(env!("CARGO_BIN_EXE_fsfs"));
            search.args(["search", "repairquerytoken"]);
            configure(&mut search, index_dir);
            search.args(["--no-daemon", "--format", "json", "--quiet"]);
            search
        };
        let original = if protect {
            // Opt in as a publisher. The store must seal the real sidecars in
            // its original inventory before any data is damaged.
            let store = CompleteGenerationStore::create(&cx, &root).unwrap();
            let build = store.begin(&cx).unwrap();
            for (relative, bytes) in &stock_files {
                if relative != Path::new(COMPLETE_GENERATION_MANIFEST) {
                    let destination = build.path().join(relative);
                    fs::create_dir_all(destination.parent().unwrap()).unwrap();
                    fs::write(destination, bytes).unwrap();
                }
            }
            let protector = FileProtector::new(
                std::sync::Arc::new(DefaultSymbolCodec),
                DurabilityConfig::default(),
            )
            .unwrap();
            for relative in [&segment, &manifest] {
                protector
                    .protect_file(&build.path().join(relative))
                    .unwrap();
            }
            let publication = build
                .publish(&cx, |_, candidate| {
                    let output = run(
                        &mut search_command(candidate),
                        directory.path(),
                        Duration::from_secs(30),
                    );
                    let result = body(&output);
                    assert!(output.status.success(), "{result}");
                    assert_keyword_hits(&result["data"]);
                    Ok(())
                })
                .unwrap();
            let GenerationPublication::Durable(generation) = publication else {
                panic!("protected keyword fixture must publish durably");
            };
            generation
        } else {
            stock
        };
        let healthy = files(original.path());
        assert_eq!(fs::read_dir(root.join("generations")).unwrap().count(), 1);
        let mut damaged_manifest = healthy[&manifest].clone();
        damaged_manifest[10] ^= 0x5a;
        fs::write(original.path().join(&manifest), damaged_manifest).unwrap();
        fs::rename(
            original.path().join(&segment),
            directory.path().join("saved-segment"),
        )
        .unwrap();
        fs::rename(&source, directory.path().join("source-unavailable")).unwrap();
        let damaged = files(original.path());
        let selection = fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap();
        let mut search = search_command(&root);
        let failed = run(&mut search, directory.path(), Duration::from_secs(30));
        assert!(!failed.status.success());
        assert_eq!(body(&failed)["ok"], false);
        let mut prepare = Command::new(env!("CARGO_BIN_EXE_fsfs-recover"));
        configure(&mut prepare, &root);
        prepare
            .args([
                "--generation",
                original.id(),
                "--manifest-sha256",
                original.manifest_sha256(),
                "--prepare",
                "--repair-artifacts",
                "--query",
                "repairquerytoken",
                "--search-mode",
                "lexical",
            ])
            .arg("--save-preparation")
            .arg(&preparation);
        let output = run(&mut prepare, directory.path(), Duration::from_secs(120));
        if !protect {
            let refused = body(&output);
            assert!(!output.status.success());
            assert_eq!(refused["ok"], false);
            assert!(
                refused["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains("damaged artifact has no receipted repair sidecar"),
                "{refused}"
            );
            assert!(!preparation.exists());
            assert_eq!(
                fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap(),
                selection
            );
            assert_eq!(files(original.path()), damaged);
            assert_eq!(fs::read_dir(root.join("generations")).unwrap().count(), 1);
            assert!(!source.exists());
            assert!(!directory.path().join("missing-models").exists());
            return;
        }
        assert!(
            output.status.success(),
            "stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let prepared = body(&output)["data"].clone();
        assert_eq!(prepared["state"], "prepared");
        assert_eq!(prepared["selection_write_performed"], false);
        assert_eq!(prepared["source_rebuild_performed"], false);
        assert_eq!(prepared["repaired_from"]["generation_id"], original.id());
        assert_eq!(
            prepared["repaired_from"]["manifest_sha256"],
            original.manifest_sha256()
        );
        assert_keyword_hits(&prepared["preview"][0]);
        assert_eq!(
            fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap(),
            selection
        );
        assert_eq!(files(original.path()), damaged);
        let target = root
            .join("generations")
            .join(prepared["generation_id"].as_str().unwrap());
        for (path, bytes) in &healthy {
            if path != Path::new(COMPLETE_GENERATION_MANIFEST) {
                assert_eq!(
                    fs::read(target.join(path)).unwrap(),
                    *bytes,
                    "{}",
                    path.display()
                );
            }
        }
        let sealed = files(&target);
        let mut apply = Command::new(env!("CARGO_BIN_EXE_fsfs-recover"));
        configure(&mut apply, &root);
        apply.arg("--resume-file").arg(&preparation).args([
            "--apply",
            "--query",
            "repairquerytoken",
            "--search-mode",
            "lexical",
        ]);
        let output = run(&mut apply, directory.path(), Duration::from_secs(120));
        assert!(
            output.status.success(),
            "stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let applied = body(&output)["data"].clone();
        assert_eq!(applied["state"], "restored_durable");
        assert_eq!(applied["generation_id"], prepared["generation_id"]);
        assert_eq!(applied["repaired_from"], prepared["repaired_from"]);
        assert_keyword_hits(&applied["preview"][0]);
        let output = run(&mut search, directory.path(), Duration::from_secs(120));
        let result = body(&output);
        assert!(output.status.success(), "{result}");
        assert_keyword_hits(&result["data"]);
        assert_eq!(files(original.path()), damaged);
        assert_eq!(files(&target), sealed);
        assert_eq!(fs::read_dir(root.join("generations")).unwrap().count(), 2);
        assert!(!source.exists());
        assert!(!directory.path().join("missing-models").exists());
    });
}

fn assert_keyword_hits(payload: &serde_json::Value) {
    let mut hits = payload["hits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|hit| hit["path"].as_str().unwrap())
        .collect::<Vec<_>>();
    hits.sort_unstable();
    assert_eq!(hits, ["alpha.md", "beta.md"]);
}

#[test]
fn source_rebuild_prepares_then_resumes_in_a_fresh_process_without_rescanning() {
    for (prepare_entry, apply_entry) in [
        (RecoveryEntry::Installed, RecoveryEntry::Standalone),
        (RecoveryEntry::Standalone, RecoveryEntry::Installed),
    ] {
        exercise_source_rebuild(prepare_entry, apply_entry);
    }
}

fn exercise_source_rebuild(prepare_entry: RecoveryEntry, apply_entry: RecoveryEntry) {
    use frankensearch_fsfs::FsfsConfig;
    use frankensearch_fsfs::config::DegradationOverrideMode;

    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source");
    let root = directory.path().join("store");
    let receipt = directory.path().join("prepared.json");
    let config_path = directory.path().join("recovery.toml");
    fs::create_dir(&source).unwrap();
    fs::write(
        source.join("alpha.md"),
        "recoverystagingsentinel initial document",
    )
    .unwrap();
    fs::write(
        source.join("beta.md"),
        "recoverystagingsentinel excluded document",
    )
    .unwrap();
    let mut config = FsfsConfig::default();
    config.pressure.degradation_override = DegradationOverrideMode::ForceLexicalOnly;
    config.indexing.offline = true;
    config.search.rerank = false;
    "{index_dir}/catalog.sqlite".clone_into(&mut config.storage.db_path);
    fs::write(&config_path, config.to_toml().unwrap()).unwrap();
    let configure = |command: &mut Command| {
        command
            .current_dir(directory.path())
            .arg("--index-dir")
            .arg(&root)
            .arg("--config")
            .arg(&config_path)
            .env("HOME", directory.path())
            .env("XDG_CONFIG_HOME", directory.path().join("empty-config"))
            .env(
                "FRANKENSEARCH_MODEL_DIR",
                directory.path().join("missing-models"),
            )
            .env("FRANKENSEARCH_CHECK_UPDATES", "0");
    };
    let mut prepare = prepare_entry.command();
    configure(&mut prepare);
    prepare
        .arg("--rebuild-source")
        .arg(&source)
        .args([
            "--prepare",
            "--query",
            "recoverystagingsentinel",
            "--search-mode",
            "lexical",
            "--filter",
            "path:alpha.md",
            "--ranking-priors",
            "--no-rerank",
        ])
        .arg("--save-preparation")
        .arg(&receipt);
    let output = run(&mut prepare, directory.path(), Duration::from_secs(120));
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let prepared = body(&output)["data"].clone();
    assert_eq!(prepared["state"], "prepared");
    assert_eq!(prepared["source_rebuild_performed"], true);
    assert_eq!(prepared["selection_write_performed"], false);
    assert_eq!(prepared["preview"][0]["hits"][0]["path"], "alpha.md");
    assert_eq!(prepared["preview"][0]["hits"].as_array().unwrap().len(), 1);
    assert_eq!(prepared["preview_requirement"]["searchable"], true);
    let saved = fs::read_to_string(&receipt).unwrap();
    assert!(!saved.contains("recoverystagingsentinel"));
    assert!(!saved.contains("path:alpha.md"));
    let saved: serde_json::Value = serde_json::from_str(&saved).unwrap();
    assert_eq!(saved["schema_version"], 2);
    assert_eq!(
        saved["preview_requirement"],
        prepared["preview_requirement"]
    );
    assert!(!root.join(COMPLETE_GENERATION_POINTER).exists());
    let target = root
        .join("generations")
        .join(prepared["generation_id"].as_str().unwrap());
    let sealed = files(&target);
    fs::rename(&source, directory.path().join("source-unavailable")).unwrap();

    let store_before = files(&root);
    let original_policy = [
        "--query",
        "recoverystagingsentinel",
        "--search-mode",
        "lexical",
        "--filter",
        "path:alpha.md",
        "--ranking-priors",
        "--no-rerank",
    ];
    // Each new process independently reloads its config and the saved binding.
    // No changed request may create files, select the candidate, or rescan the
    // now-unavailable source merely because its target/selection guard matches.
    for (index, replacement) in [
        (1, "different query"),
        (3, "full"),
        (3, "fast"),
        (3, "quality"),
        (5, "path:beta.md"),
        (6, "--no-ranking-priors"),
        (7, "--rerank"),
    ] {
        let mut policy = original_policy;
        policy[index] = replacement;
        let mut changed = apply_entry.command();
        configure(&mut changed);
        changed
            .arg("--resume-file")
            .arg(&receipt)
            .arg("--apply")
            .args(policy);
        let output = run(&mut changed, directory.path(), Duration::from_secs(30));
        assert!(
            !output.status.success(),
            "changed preview policy was accepted"
        );
        assert_eq!(body(&output)["ok"], false);
        assert_eq!(files(&root), store_before);
    }
    for extra in [vec!["--limit", "9"], vec!["--format", "jsonl"]] {
        let mut changed = apply_entry.command();
        configure(&mut changed);
        changed
            .arg("--resume-file")
            .arg(&receipt)
            .arg("--apply")
            .args(original_policy)
            .args(extra);
        let output = run(&mut changed, directory.path(), Duration::from_secs(30));
        assert!(!output.status.success());
        assert_eq!(files(&root), store_before);
    }
    for policy in [
        Vec::new(),
        vec![
            "--query",
            "recoverystagingsentinel",
            "--search-mode",
            "lexical",
            "--ranking-priors",
            "--no-rerank",
        ],
    ] {
        let mut changed = apply_entry.command();
        configure(&mut changed);
        changed
            .arg("--resume-file")
            .arg(&receipt)
            .arg("--apply")
            .args(policy);
        let output = run(&mut changed, directory.path(), Duration::from_secs(30));
        assert!(!output.status.success());
        assert_eq!(files(&root), store_before);
    }
    config.search.quality_timeout_ms += 1;
    fs::write(&config_path, config.to_toml().unwrap()).unwrap();
    let mut changed = apply_entry.command();
    configure(&mut changed);
    changed
        .arg("--resume-file")
        .arg(&receipt)
        .arg("--apply")
        .args(original_policy);
    let output = run(&mut changed, directory.path(), Duration::from_secs(30));
    assert!(!output.status.success());
    assert_eq!(files(&root), store_before);
    config.search.quality_timeout_ms -= 1;
    fs::write(&config_path, config.to_toml().unwrap()).unwrap();

    let mut apply = apply_entry.command();
    configure(&mut apply);
    apply
        .arg("--resume-file")
        .arg(&receipt)
        .arg("--apply")
        .args(original_policy);
    let output = run(&mut apply, directory.path(), Duration::from_secs(120));
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let restored = body(&output)["data"].clone();
    assert_eq!(restored["state"], "restored_durable");
    assert_eq!(restored["generation_id"], prepared["generation_id"]);
    assert_eq!(restored["source_rebuild_performed"], false);
    assert_eq!(restored["preview"][0]["hits"][0]["path"], "alpha.md");
    assert_eq!(restored["preview"][0]["hits"].as_array().unwrap().len(), 1);
    assert_eq!(
        restored["preview_requirement"],
        prepared["preview_requirement"]
    );
    assert_eq!(files(&target), sealed);

    let mut search = Command::new(env!("CARGO_BIN_EXE_fsfs"));
    search.args(["search", "recoverystagingsentinel"]);
    configure(&mut search);
    search.args([
        "--no-daemon",
        "--format",
        "json",
        "--quiet",
        "--filter",
        "path:alpha.md",
    ]);
    let output = run(&mut search, directory.path(), Duration::from_secs(120));
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert_eq!(body(&output)["data"]["hits"][0]["path"], "alpha.md");
    assert_eq!(files(&target), sealed);
    assert!(!directory.path().join("missing-models").exists());
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
        for entry in [RecoveryEntry::Standalone, RecoveryEntry::Installed] {
            let mut command = recover_with_entry(
                entry,
                directory.path(),
                &root,
                current.id(),
                current.manifest_sha256(),
            );
            command.arg("--confirm-durability");
            let output = run(&mut command, directory.path(), Duration::from_secs(20));
            assert!(
                output.status.success(),
                "{entry:?}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let result = body(&output);
            assert_eq!(result["data"]["state"], "durability_confirmed");
            assert_eq!(result["data"]["generation_id"], current.id());
            assert_eq!(result["data"]["selection_write_performed"], false);
            assert_eq!(result["data"]["producer_admission_checked"], false);
            let mut stale = recover_with_entry(
                entry,
                directory.path(),
                &root,
                old.id(),
                old.manifest_sha256(),
            );
            stale.arg("--confirm-durability");
            let output = run(&mut stale, directory.path(), Duration::from_secs(20));
            assert!(!output.status.success(), "{entry:?}");
            assert_eq!(body(&output)["ok"], false);
            assert_eq!(fs::read(&pointer).unwrap(), before);
            assert_eq!(fs::metadata(&pointer).unwrap().ino(), inode);
            assert_eq!(files(old.path()), old_files);
            assert_eq!(files(current.path()), current_files);
            assert!(!directory.path().join("missing-models").exists());
        }
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

#[cfg(all(feature = "semantic-loaders", not(feature = "embedded-models")))]
#[test]
#[ignore = "requires FRANKENSEARCH_RECOVERY_E2E_MODEL_DIR with a verified MiniLM cache; no fast model required"]
fn genuine_quality_only_source_rebuild_resumes_and_searches_without_fast_or_sources() {
    use frankensearch_fsfs::FsfsConfig;

    let models = std::env::var_os("FRANKENSEARCH_RECOVERY_E2E_MODEL_DIR")
        .map(std::path::PathBuf::from)
        .expect("provide a verified MiniLM cache; no mock or skip fallback");
    let models = fs::canonicalize(models).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source");
    let root = directory.path().join("store");
    let receipt = directory.path().join("prepared-quality.json");
    let config_path = directory.path().join("quality-recovery.toml");
    fs::create_dir(&source).unwrap();
    fs::write(
        source.join("alpha.md"),
        "recoverytoken preserves durable searchable generations and the original documents",
    )
    .unwrap();
    fs::write(
        source.join("beta.md"),
        "recoverytoken excluded secondary document",
    )
    .unwrap();
    let mut config = FsfsConfig::default();
    config.indexing.offline = true;
    config.indexing.model_dir = models.display().to_string();
    "unused-invalid-fast-producer".clone_into(&mut config.indexing.fast_model);
    "all-minilm-l6-v2".clone_into(&mut config.indexing.quality_model);
    config.indexing.fast_window_max_per_file = 4;
    config.search.fast_only = false;
    config.search.rerank = false;
    config.search.quality_timeout_ms = 10_000;
    "{index_dir}/catalog.sqlite".clone_into(&mut config.storage.db_path);
    fs::write(&config_path, config.to_toml().unwrap()).unwrap();
    let configure = |command: &mut Command| {
        for (key, _) in std::env::vars_os() {
            let name = key.to_string_lossy();
            if name.starts_with("FSFS_") || name.starts_with("FRANKENSEARCH_") {
                command.env_remove(key);
            }
        }
        command
            .current_dir(directory.path())
            .arg("--index-dir")
            .arg(&root)
            .arg("--config")
            .arg(&config_path)
            .env("HOME", directory.path())
            .env("XDG_CONFIG_HOME", directory.path().join("empty-config"))
            .env("FRANKENSEARCH_MODEL_DIR", &models)
            .env("FRANKENSEARCH_CHECK_UPDATES", "0");
    };
    let query = "how does recoverytoken preserve durable search generations";
    let mut prepare = Command::new(env!("CARGO_BIN_EXE_fsfs-recover"));
    configure(&mut prepare);
    prepare
        .arg("--rebuild-source")
        .arg(&source)
        .args([
            "--prepare",
            "--query",
            query,
            "--search-mode",
            "quality",
            "--filter",
            "path:alpha.md",
            "--no-rerank",
        ])
        .arg("--save-preparation")
        .arg(&receipt);
    let output = run(&mut prepare, directory.path(), Duration::from_secs(180));
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let prepared = body(&output)["data"].clone();
    assert_eq!(prepared["state"], "prepared");
    assert_eq!(prepared["source_rebuild_performed"], true);
    assert_eq!(prepared["preview"].as_array().unwrap().len(), 1);
    assert_eq!(prepared["preview"][0]["hits"][0]["path"], "alpha.md");
    let target = root
        .join("generations")
        .join(prepared["generation_id"].as_str().unwrap());
    assert!(target.join("vector/quality.fsvi").is_file());
    assert!(!target.join("vector/index.fsvi").exists());
    assert!(!root.join(COMPLETE_GENERATION_POINTER).exists());
    let sealed = files(&target);
    fs::rename(&source, directory.path().join("source-unavailable")).unwrap();

    let mut apply = Command::new(env!("CARGO_BIN_EXE_fsfs-recover"));
    configure(&mut apply);
    apply.arg("--resume-file").arg(&receipt).args([
        "--apply",
        "--query",
        query,
        "--search-mode",
        "quality",
        "--filter",
        "path:alpha.md",
        "--no-rerank",
    ]);
    let output = run(&mut apply, directory.path(), Duration::from_secs(180));
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let restored = body(&output)["data"].clone();
    assert_eq!(restored["state"], "restored_durable");
    assert_eq!(restored["generation_id"], prepared["generation_id"]);
    assert_eq!(restored["source_rebuild_performed"], false);
    assert_eq!(restored["selection_write_performed"], true);

    for quality in [true, false] {
        let mut search = Command::new(env!("CARGO_BIN_EXE_fsfs"));
        search.args(["search", if quality { query } else { "recoverytoken" }]);
        configure(&mut search);
        search.args([
            "--format",
            "json",
            "--quiet",
            "--no-daemon",
            "--no-rerank",
            "--filter",
            "path:alpha.md",
        ]);
        if quality {
            search.arg("--quality-only");
        } else {
            search.env(
                "FRANKENSEARCH_PRESSURE_DEGRADATION_OVERRIDE",
                "force_lexical_only",
            );
        }
        let output = run(&mut search, directory.path(), Duration::from_secs(180));
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stdout)
        );
        let payload = body(&output)["data"].clone();
        assert_eq!(payload["hits"].as_array().unwrap().len(), 1);
        assert_eq!(payload["hits"][0]["path"], "alpha.md");
        assert_eq!(payload["hits"][0]["semantic_rank"].is_number(), quality);
        assert_eq!(files(&target), sealed);
    }
    assert_eq!(fs::read_dir(root.join("generations")).unwrap().count(), 1);
    assert!(!source.exists());
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
    fs::write(
        source.join("alpha.md"),
        "recoverytoken preserves durable searchable generations and the original documents",
    )
    .unwrap();
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
        command
            .current_dir(directory.path())
            .arg("index")
            .arg(&source)
            .arg("--index-dir")
            .arg(&root)
            .arg("--config")
            .arg(&config_path)
            .args(["--format", "json", "--quiet"])
            .env("FRANKENSEARCH_COMPLETE_GENERATIONS", "1")
            .env("FRANKENSEARCH_CHECK_UPDATES", "0")
            .env("FRANKENSEARCH_MODEL_DIR", &models);
        let output = run(&mut command, directory.path(), Duration::from_secs(180));
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        body(&output)["data"].clone()
    };
    let first = index();
    fs::write(
        source.join("beta.md"),
        "recoverytoken searchable successor document",
    )
    .unwrap();
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
        command
            .arg("--config")
            .arg(&config_path)
            .args(["--query", query])
            .env("FRANKENSEARCH_MODEL_DIR", &models);
        if apply {
            command.arg("--apply");
        }
        let output = run(&mut command, directory.path(), Duration::from_secs(180));
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let result = body(&output);
        assert_eq!(result["data"]["generation_id"], id);
        assert_eq!(result["data"]["selection_write_performed"], apply);
        assert_eq!(result["data"]["producer_admission_checked"], true);
        let phases = result["data"]["preview"].as_array().unwrap();
        let refined = serde_json::to_value(SearchOutputPhase::Refined).unwrap();
        assert!(phases.iter().any(|phase| phase["phase"] == refined));
        assert_eq!(phases.last().unwrap()["hits"].as_array().unwrap().len(), 1);
        if !apply {
            assert_eq!(fs::read(&pointer).unwrap(), b"corrupt selection");
        }
        assert_eq!(files(&old_root), old_bytes);
        assert_eq!(files(&new_root), new_bytes);
    }
    let mut search = Command::new(env!("CARGO_BIN_EXE_fsfs"));
    search
        .current_dir(directory.path())
        .args(["search", query])
        .arg("--index-dir")
        .arg(&root)
        .arg("--config")
        .arg(&config_path)
        .args(["--format", "json", "--quiet"])
        .env("FRANKENSEARCH_CHECK_UPDATES", "0")
        .env("FRANKENSEARCH_MODEL_DIR", &models);
    let output = run(&mut search, directory.path(), Duration::from_secs(180));
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
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
