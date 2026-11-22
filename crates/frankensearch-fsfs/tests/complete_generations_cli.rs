//! Production-executable coverage for complete-generation routing and discovery.
//!
//! Refusal and lexical-only discovery cases do not need models. The semantic
//! case is explicitly ignored unless a verified cache is provisioned; it never
//! enables the library's cfg(test) hash embedder inside the shipping binary.
#![cfg(unix)]

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{Duration, Instant};

use frankensearch_fsfs::generation_store::COMPLETE_GENERATION_POINTER;
use frankensearch_fsfs::{FsCategory, FsfsConfig, MountTable, read_system_mounts};
use serde_json::Value;

struct Fixture {
    directory: tempfile::TempDir,
    source: PathBuf,
    store: PathBuf,
    config: PathBuf,
    executable: PathBuf,
    executable_directory: Option<tempfile::TempDir>,
    child_identity: Option<(u32, u32)>,
}

impl Fixture {
    fn new() -> Self {
        Self::with_source(tempfile::tempdir().expect("fixture"), "source")
    }

    fn with_source(directory: tempfile::TempDir, root_name: &str) -> Self {
        let source = directory.path().join(root_name);
        let store = directory.path().join("store");
        let config = directory.path().join("fsfs.toml");
        fs::create_dir(&source).expect("source");
        fs::write(source.join("alpha.md"), "sharedtoken alpha document").expect("source file");
        let fixture = Self {
            directory,
            source,
            store,
            config,
            executable: PathBuf::from(env!("CARGO_BIN_EXE_fsfs")),
            executable_directory: None,
            child_identity: None,
        };
        fixture.write_config(None, "{index_dir}/catalog.sqlite");
        fixture
    }

    fn for_source_permissions() -> Self {
        // A root-owned checkout or TMPDIR may have non-traversable ancestors.
        // Use a real admitted mount for source/store fixtures. Some containers
        // put /tmp on overlay, which production discovery correctly refuses.
        let mounts = MountTable::new(read_system_mounts(), &HashMap::new());
        let mut parent = fs::canonicalize("/tmp").expect("Unix temporary directory");
        if mounts
            .lookup(&parent)
            .is_some_and(|(entry, _)| entry.category.is_virtual())
        {
            let memory = fs::canonicalize("/dev/shm").expect("admitted temporary mount");
            assert!(
                mounts.lookup(&memory).is_some_and(|(entry, policy)| {
                    entry.category == FsCategory::Memory && policy.enabled && policy.available
                }),
                "permission fixtures need a real local or memory filesystem"
            );
            parent = memory;
        }
        let directory = tempfile::Builder::new()
            .prefix("fsfs-source-access-")
            .tempdir_in(parent)
            .expect("accessible permission fixture");
        let mut fixture = Self::with_source(directory, "source");
        // UID 0 may have no DAC-bypass capability in a user namespace. Probe
        // actual access before choosing an identity; keep this file outside
        // the source so it cannot alter any indexed-document counts.
        let needs_unprivileged_child = {
            let probe = fixture.directory.path().join("permission-probe");
            fs::write(&probe, "permission probe").expect("permission probe file");
            let blocked = UnreadableSource::new(probe.clone());
            let bypasses_permissions = match fs::read(&probe) {
                Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => false,
                Ok(_) => true,
                Err(error) => panic!("cannot determine fixture read permissions: {error}"),
            };
            blocked.restore();
            bypasses_permissions
        };
        if needs_unprivileged_child {
            assert_eq!(
                fs::metadata(fixture.directory.path()).unwrap().uid(),
                0,
                "a permission-bypassing test identity must support dropping root privileges"
            );
            // Stage the executable separately: an admitted tmpfs source mount
            // may be noexec. The actual fsfs child must run without root's
            // permission bypass, even when Cargo's checkout is root-private.
            let executable_directory = tempfile::Builder::new()
                .prefix("fsfs-access-bin-")
                .tempdir_in("/tmp")
                .expect("accessible executable directory");
            fs::set_permissions(
                executable_directory.path(),
                fs::Permissions::from_mode(0o755),
            )
            .expect("make staged executable directory traversable");
            let executable = executable_directory.path().join("fsfs");
            let mode = fs::metadata(&fixture.executable).unwrap().mode();
            if mode & 0o005 != 0o005 || fs::hard_link(&fixture.executable, &executable).is_err() {
                fs::copy(&fixture.executable, &executable).expect("copy executable for child");
                fs::set_permissions(&executable, fs::Permissions::from_mode(0o755))
                    .expect("make copied executable accessible");
            }
            // Never chmod/chown a hard-linked executable: its inode also
            // belongs to Cargo's original artifact.
            fixture.executable = executable;
            fixture.executable_directory = Some(executable_directory);
            fixture.child_identity = Some((65_534, 65_534));
            for path in [
                fixture.directory.path(),
                fixture.source.as_path(),
                fixture.config.as_path(),
                &fixture.source.join("alpha.md"),
            ] {
                fixture.assign_child_ownership(path);
            }
        }
        fixture
    }

    fn assign_child_ownership(&self, path: &Path) {
        if let Some((uid, gid)) = self.child_identity {
            std::os::unix::fs::chown(path, Some(uid), Some(gid)).expect(
                "a permission-bypassing test host must support fixture ownership by UID/GID 65534",
            );
        }
    }

    fn write_config(&self, model_dir: Option<&Path>, catalog: &str) {
        let mut config = FsfsConfig::default();
        config.indexing.offline = true;
        config.indexing.watch_mode = false;
        config.indexing.quality_model.clear();
        config.search.fast_only = true;
        config.search.rerank = false;
        catalog.clone_into(&mut config.storage.db_path);
        if let Some(path) = model_dir {
            config.indexing.model_dir = path.display().to_string();
        }
        fs::write(&self.config, config.to_toml().expect("TOML")).expect("fixture config");
    }

    fn write_quality_only_config(&self, model_dir: &Path) {
        let mut config = FsfsConfig::default();
        config.indexing.offline = true;
        config.indexing.watch_mode = false;
        // A successful build or query must never resolve this fast producer.
        "unused-invalid-fast-producer".clone_into(&mut config.indexing.fast_model);
        "all-minilm-l6-v2-native".clone_into(&mut config.indexing.quality_model);
        config.indexing.model_dir = model_dir.display().to_string();
        config.search.fast_only = false;
        config.search.rerank = false;
        config.search.quality_timeout_ms = 30_000;
        "{index_dir}/catalog.sqlite".clone_into(&mut config.storage.db_path);
        fs::write(&self.config, config.to_toml().expect("TOML")).expect("quality config");
    }

    fn command(&self, command: &str, format: &str) -> Command {
        let mut child = Command::new(&self.executable);
        if let Some((uid, gid)) = self.child_identity {
            child.gid(gid).uid(uid);
        }
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
            .arg(command);
        // The CLI takes its command's positional argument before options.
        match command {
            "index" => {
                child.arg(&self.source);
            }
            "search" => {
                child.arg("sharedtoken");
            }
            _ => {}
        }
        child
            .arg("--config")
            .arg(&self.config)
            .arg("--index-dir")
            .arg(&self.store)
            .args(["--format", format, "--quiet"]);
        child
    }

    fn lexical_discovery_command(&self, command: &str) -> Command {
        let mut child = self.command(command, "json");
        let home = self.directory.path().join("home");
        child
            .env("HOME", &home)
            .env_remove("GIT_CONFIG_GLOBAL")
            .env("GIT_CONFIG_SYSTEM", "/dev/null")
            .env("XDG_CONFIG_HOME", home.join("config"))
            .env("XDG_CACHE_HOME", home.join("cache"))
            .env("XDG_DATA_HOME", home.join("data"))
            .env("TMPDIR", self.directory.path())
            .env("FRANKENSEARCH_MODEL_DIR", home.join("models"))
            .env("FRANKENSEARCH_ALLOW_DOWNLOAD", "0")
            .env(
                "FRANKENSEARCH_PRESSURE_DEGRADATION_OVERRIDE",
                "force_lexical_only",
            );
        child
    }
}

fn execute(command: &mut Command) -> Output {
    execute_with_input(command, b"")
}

fn execute_with_input(command: &mut Command, input: &[u8]) -> Output {
    let capture = tempfile::tempdir().expect("capture");
    let stdin_path = capture.path().join("stdin");
    fs::write(&stdin_path, input).expect("request input");
    let stdout_path = capture.path().join("stdout");
    let stderr_path = capture.path().join("stderr");
    let mut child = command
        .stdin(fs::File::open(stdin_path).expect("stdin"))
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
                // ubs:ignore — integration-test assertion: a timed-out child must fail the test.
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

fn assert_only_selected_document_is_searchable(fixture: &Fixture, excludes: Option<&str>) {
    let mut index = fixture.lexical_discovery_command("index");
    index.arg("--no-watch-mode");
    if let Some(patterns) = excludes {
        index.args(["--exclude", patterns]);
    }
    let indexed = json_output(&execute(&mut index), true);
    assert_eq!(indexed["data"]["indexed_files"], 1, "{indexed}");

    let result = json_output(
        &execute(fixture.lexical_discovery_command("search").args([
            "--no-daemon",
            "--limit",
            "100",
        ])),
        true,
    );
    let hits = result["data"]["hits"].as_array().expect("search hits");
    assert_eq!(
        hits.len(),
        1,
        "excluded or out-of-root content leaked: {result}"
    );
    assert_eq!(hits[0]["path"], "alpha.md", "{result}");
}

#[test]
fn discovery_cli_accepts_roots_and_temporary_stores_beneath_excluded_ancestors() {
    for ancestor in ["build", "target"] {
        let outer = tempfile::tempdir().expect("outer fixture");
        let temporary_parent = outer.path().join(ancestor);
        fs::create_dir(&temporary_parent).unwrap();
        let fixture = Fixture::with_source(
            tempfile::tempdir_in(&temporary_parent).expect("temporary fixture below exclusion"),
            "notes",
        );
        // Both the selected root and the index store are below the excluded
        // ancestor. Child-only TMPDIR also exercises this layout without
        // changing the test runner's process-wide environment.
        for excluded in [
            "build",
            "dist",
            "vendor",
            "target",
            "node_modules",
            ".venv",
            ".next",
            "__pycache__",
            ".git",
        ] {
            let nested = fixture.source.join(excluded).join("nested");
            fs::create_dir_all(&nested).unwrap();
            fs::write(
                nested.join("excluded.md"),
                "sharedtoken excluded descendant",
            )
            .unwrap();
        }
        let outside = fixture.directory.path().join("outside.md");
        let outside_body = "sharedtoken outside the selected root";
        fs::write(&outside, outside_body).unwrap();
        std::os::unix::fs::symlink(&outside, fixture.source.join("linked-outside.md")).unwrap();

        assert_only_selected_document_is_searchable(&fixture, None);
        assert_eq!(fs::read_to_string(outside).unwrap(), outside_body);
    }
}

#[test]
fn discovery_cli_accepts_relative_roots_beneath_excluded_ancestors() {
    for ancestor in ["build", "target"] {
        let outer = tempfile::tempdir().expect("outer fixture");
        let temporary_parent = outer.path().join(ancestor);
        fs::create_dir(&temporary_parent).unwrap();
        let mut fixture = Fixture::with_source(
            tempfile::tempdir_in(&temporary_parent).expect("temporary fixture below exclusion"),
            "notes",
        );
        let excluded = fixture.source.join("vendor/nested");
        fs::create_dir_all(&excluded).unwrap();
        fs::write(
            excluded.join("excluded.md"),
            "sharedtoken excluded descendant",
        )
        .unwrap();

        // The child starts in the fixture directory. The CLI-selected root is
        // relative, and differs from the discovery roots in its config file.
        fixture.source = PathBuf::from("./notes");
        assert_only_selected_document_is_searchable(&fixture, None);
    }
}

#[test]
fn discovery_cli_scopes_custom_components_and_keeps_path_and_wildcard_exclusions() {
    let outer = tempfile::tempdir().expect("outer fixture");
    let temporary_parent = outer.path().join("private");
    fs::create_dir(&temporary_parent).unwrap();
    let fixture = Fixture::with_source(
        tempfile::tempdir_in(&temporary_parent).expect("temporary fixture below custom exclusion"),
        "notes",
    );
    for relative in [
        "private/secret.md",
        "generated/nested/cache.md",
        "docs/draft-secrets.md",
    ] {
        let path = fixture.source.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, "sharedtoken custom excluded descendant").unwrap();
    }

    assert_only_selected_document_is_searchable(&fixture, Some("private,generated/**,draft-*.md"));
}

#[test]
fn discovery_cli_preserves_absolute_wildcard_exclusions_beneath_build() {
    let outer = tempfile::tempdir().expect("outer fixture");
    let temporary_parent = outer.path().join("build");
    fs::create_dir(&temporary_parent).unwrap();
    let fixture = Fixture::with_source(
        tempfile::tempdir_in(&temporary_parent).expect("temporary fixture below build"),
        "notes",
    );
    let private = fixture.source.join("private/nested");
    fs::create_dir_all(&private).unwrap();
    fs::write(
        private.join("secret.md"),
        "sharedtoken absolute excluded descendant",
    )
    .unwrap();
    // The CLI canonicalizes its root, so anchor the custom rule to the same
    // path even on platforms whose temporary directory has a symlink alias.
    let canonical_root = fs::canonicalize(&fixture.source).unwrap();
    let descendant_pattern = format!("build,{}/private/**", canonical_root.display());
    assert_only_selected_document_is_searchable(&fixture, Some(&descendant_pattern));

    let root_pattern = format!("build,{}*", canonical_root.display());
    let value = json_output(
        &execute(fixture.lexical_discovery_command("index").args([
            "--no-watch-mode",
            "--exclude",
            &root_pattern,
        ])),
        false,
    );
    assert_eq!(value["error"]["field"], "discovery.roots", "{value}");
    assert!(
        value["error"]["message"]
            .as_str()
            .unwrap()
            .contains("target root excluded by discovery policy"),
        "{value}"
    );
}

#[test]
fn discovery_cli_still_rejects_explicitly_selected_excluded_root_names() {
    for (root_name, excludes) in [
        ("build", None),
        ("target", None),
        ("private", Some("private")),
    ] {
        let fixture = Fixture::with_source(tempfile::tempdir().expect("fixture"), root_name);
        let mut index = fixture.lexical_discovery_command("index");
        index.arg("--no-watch-mode");
        if let Some(patterns) = excludes {
            index.args(["--exclude", patterns]);
        }
        let value = json_output(&execute(&mut index), false);
        assert_eq!(value["error"]["field"], "discovery.roots", "{value}");
        assert!(
            value["error"]["message"]
                .as_str()
                .unwrap()
                .contains("target root excluded by discovery policy"),
            "{root_name}: {value}"
        );
        assert_eq!(
            fs::read_to_string(fixture.source.join("alpha.md")).unwrap(),
            "sharedtoken alpha document"
        );
    }
}

struct UnreadableSource {
    path: PathBuf,
    original: fs::Permissions,
}

impl UnreadableSource {
    fn new(path: PathBuf) -> Self {
        let original = fs::metadata(&path).unwrap().permissions();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o000))
            .expect("deny source access to the production process");
        Self { path, original }
    }

    fn restore(&self) {
        fs::set_permissions(&self.path, self.original.clone()).expect("restore fixture access");
    }
}

impl Drop for UnreadableSource {
    fn drop(&mut self) {
        // Restore the test's directory access even if an assertion unwinds.
        let _ = fs::set_permissions(&self.path, self.original.clone());
    }
}

fn add_permission_source(fixture: &Fixture, subtree: bool) -> (PathBuf, &'static str) {
    let relative = if subtree {
        "restricted/beta.md"
    } else {
        "beta.md"
    };
    let source = fixture.source.join(relative);
    if subtree {
        let directory = source.parent().unwrap();
        fs::create_dir(directory).unwrap();
        fixture.assign_child_ownership(directory);
    }
    fs::write(&source, "sharedtoken beta permission fixture").unwrap();
    fixture.assign_child_ownership(&source);
    let denied = if subtree {
        source.parent().unwrap().to_path_buf()
    } else {
        source
    };
    (denied, relative)
}

fn generation_files(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
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

fn assert_search_membership(fixture: &Fixture, expected: &[&str]) {
    let result = json_output(
        &execute(fixture.lexical_discovery_command("search").args([
            "--no-daemon",
            "--limit",
            "100",
        ])),
        true,
    );
    let hits = result["data"]["hits"].as_array().expect("search hits");
    assert_eq!(hits.len(), expected.len(), "{result}");
    let paths = hits
        .iter()
        .map(|hit| hit["path"].as_str().expect("hit path"))
        .collect::<BTreeSet<_>>();
    assert_eq!(
        paths,
        expected.iter().copied().collect::<BTreeSet<_>>(),
        "{result}"
    );
}

fn assert_permission_refusal(output: &Output) {
    let value = json_output(output, false);
    assert_eq!(value["error"]["code"], "io_error", "{value}");
    assert!(
        value["error"]["message"]
            .as_str()
            .unwrap()
            .contains("PermissionDenied"),
        "the actual child must encounter a source permission failure: {value}"
    );
}

fn assert_unreadable_successor_preserves_selection(subtree: bool, force: bool) {
    let fixture = Fixture::for_source_permissions();
    let (denied, relative) = add_permission_source(&fixture, subtree);
    let first = json_output(
        &execute(
            fixture
                .lexical_discovery_command("index")
                .env("FSFS_COMPLETE_GENERATIONS", "1")
                .arg("--no-watch-mode"),
        ),
        true,
    );
    assert_eq!(first["data"]["indexed_files"], 2, "{first}");
    let predecessor = PathBuf::from(first["data"]["generation_path"].as_str().unwrap());
    let predecessor_files = generation_files(&predecessor);
    let pointer = fixture.store.join(COMPLETE_GENERATION_POINTER);
    let selection = fs::read(&pointer).unwrap();
    let blocked = UnreadableSource::new(denied);
    let mut rebuild = fixture.lexical_discovery_command("index");
    rebuild.arg("--no-watch-mode");
    if force {
        rebuild.arg("--force");
    }
    assert_permission_refusal(&execute(&mut rebuild));
    assert_eq!(fs::read(&pointer).unwrap(), selection);
    assert_eq!(generation_files(&predecessor), predecessor_files);
    assert_search_membership(&fixture, &["alpha.md", relative]);
    assert_eq!(fs::read(&pointer).unwrap(), selection);
    assert_eq!(generation_files(&predecessor), predecessor_files);

    blocked.restore();
    let retry = json_output(&execute(&mut rebuild), true);
    assert_eq!(retry["data"]["indexed_files"], 2, "{retry}");
    assert_ne!(fs::read(&pointer).unwrap(), selection);
    assert_eq!(generation_files(&predecessor), predecessor_files);
    assert_search_membership(&fixture, &["alpha.md", relative]);

    // A genuine removal from the source scope remains authoritative. Keep
    // the fixture bytes outside that scope so the control needs no deletion.
    let moved = fixture.directory.path().join("moved-out-beta.md");
    fs::rename(fixture.source.join(relative), &moved).unwrap();
    let removed = json_output(&execute(&mut rebuild), true);
    assert_eq!(removed["data"]["indexed_files"], 1, "{removed}");
    assert_search_membership(&fixture, &["alpha.md"]);
    assert_eq!(
        fs::read_to_string(moved).unwrap(),
        "sharedtoken beta permission fixture"
    );
}

#[test]
fn complete_cli_binary_unreadable_successors_preserve_selected_documents() {
    for subtree in [false, true] {
        assert_unreadable_successor_preserves_selection(subtree, false);
    }
}

#[test]
fn complete_cli_binary_force_rebuild_refuses_unreadable_sources() {
    for subtree in [false, true] {
        assert_unreadable_successor_preserves_selection(subtree, true);
    }
}

#[test]
fn complete_cli_binary_unreadable_excluded_subtree_does_not_block_successor() {
    let fixture = Fixture::for_source_permissions();
    let excluded = fixture.source.join("node_modules");
    fs::create_dir(&excluded).unwrap();
    fixture.assign_child_ownership(&excluded);
    let excluded_file = excluded.join("private.md");
    fs::write(&excluded_file, "sharedtoken excluded source").unwrap();
    fixture.assign_child_ownership(&excluded_file);
    let mut index = fixture.lexical_discovery_command("index");
    index
        .env("FSFS_COMPLETE_GENERATIONS", "1")
        .arg("--no-watch-mode");
    let first = json_output(&execute(&mut index), true);
    assert_eq!(first["data"]["indexed_files"], 1, "{first}");
    let predecessor = PathBuf::from(first["data"]["generation_path"].as_str().unwrap());
    let predecessor_files = generation_files(&predecessor);
    let pointer = fixture.store.join(COMPLETE_GENERATION_POINTER);
    let selection = fs::read(&pointer).unwrap();
    let blocked = UnreadableSource::new(excluded);

    let successor = json_output(&execute(&mut index), true);
    assert_eq!(successor["data"]["indexed_files"], 1, "{successor}");
    assert_ne!(fs::read(&pointer).unwrap(), selection);
    assert_eq!(generation_files(&predecessor), predecessor_files);
    assert_search_membership(&fixture, &["alpha.md"]);
    blocked.restore();
    assert_eq!(
        fs::read_to_string(excluded_file).unwrap(),
        "sharedtoken excluded source"
    );
}

#[test]
fn complete_cli_binary_recovery_refuses_partial_sources_and_preserves_selection() {
    for subtree in [false, true] {
        let fixture = Fixture::for_source_permissions();
        let (denied, relative) = add_permission_source(&fixture, subtree);
        let first = json_output(
            &execute(
                fixture
                    .lexical_discovery_command("index")
                    .env("FSFS_COMPLETE_GENERATIONS", "1")
                    .arg("--no-watch-mode"),
            ),
            true,
        );
        assert_eq!(first["data"]["indexed_files"], 2, "{first}");
        let predecessor = PathBuf::from(first["data"]["generation_path"].as_str().unwrap());
        let predecessor_files = generation_files(&predecessor);
        let pointer = fixture.store.join(COMPLETE_GENERATION_POINTER);
        let selection = fs::read(&pointer).unwrap();
        let blocked = UnreadableSource::new(denied);
        let mut recovery = fixture.lexical_discovery_command("search");
        recovery
            .args(["--no-daemon", "--recover-source"])
            .arg(&fixture.source);

        // The readable alpha row is a matching query hit. Previewing just
        // that surviving row must not authorize a partial replacement.
        assert_permission_refusal(&execute(&mut recovery));
        assert_eq!(fs::read(&pointer).unwrap(), selection);
        assert_eq!(generation_files(&predecessor), predecessor_files);
        assert_search_membership(&fixture, &["alpha.md", relative]);

        blocked.restore();
        let recovered = json_output(&execute(&mut recovery), true);
        assert_eq!(recovered["data"]["returned_hits"], 2, "{recovered}");
        assert_ne!(fs::read(&pointer).unwrap(), selection);
        assert_eq!(generation_files(&predecessor), predecessor_files);
        assert_search_membership(&fixture, &["alpha.md", relative]);
    }
}

#[test]
fn complete_cli_binary_fresh_recovery_requires_complete_source_access() {
    let fixture = Fixture::for_source_permissions();
    let (denied, relative) = add_permission_source(&fixture, false);
    let blocked = UnreadableSource::new(denied);
    let mut recovery = fixture.lexical_discovery_command("search");
    recovery
        .args(["--no-daemon", "--recover-source"])
        .arg(&fixture.source);
    assert_permission_refusal(&execute(&mut recovery));
    assert!(!fixture.store.join(COMPLETE_GENERATION_POINTER).exists());

    blocked.restore();
    let recovered = json_output(&execute(&mut recovery), true);
    assert_eq!(recovered["data"]["returned_hits"], 2, "{recovered}");
    assert_search_membership(&fixture, &["alpha.md", relative]);
}

#[test]
fn complete_cli_binary_fresh_index_preserves_best_effort_permission_skips() {
    let fixture = Fixture::for_source_permissions();
    let (denied, relative) = add_permission_source(&fixture, false);
    let blocked = UnreadableSource::new(denied);
    let mut index = fixture.lexical_discovery_command("index");
    index
        .env("FSFS_COMPLETE_GENERATIONS", "1")
        .arg("--no-watch-mode");
    let first = json_output(&execute(&mut index), true);
    assert_eq!(first["data"]["indexed_files"], 1, "{first}");
    assert_eq!(first["data"]["skipped_files"], 1, "{first}");
    assert!(
        first["data"]["reason_codes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|reason| reason == "discovery.file.permission_denied"),
        "a privileged test host must still exercise the real permission failure: {first}"
    );
    assert_search_membership(&fixture, &["alpha.md"]);

    blocked.restore();
    let retry = json_output(&execute(&mut index), true);
    assert_eq!(retry["data"]["indexed_files"], 2, "{retry}");
    assert_search_membership(&fixture, &["alpha.md", relative]);
}

#[test]
fn complete_cli_binary_fresh_index_skips_unreadable_subtree() {
    let fixture = Fixture::for_source_permissions();
    let (denied, relative) = add_permission_source(&fixture, true);
    let blocked = UnreadableSource::new(denied);
    let mut index = fixture.lexical_discovery_command("index");
    index
        .env("FSFS_COMPLETE_GENERATIONS", "1")
        .arg("--no-watch-mode");
    let first = json_output(&execute(&mut index), true);
    assert_eq!(first["data"]["indexed_files"], 1, "{first}");
    assert_search_membership(&fixture, &["alpha.md"]);

    blocked.restore();
    let retry = json_output(&execute(&mut index), true);
    assert_eq!(retry["data"]["indexed_files"], 2, "{retry}");
    assert_search_membership(&fixture, &["alpha.md", relative]);
}

#[test]
fn complete_cli_binary_global_ignore_errors_apply_only_inside_repositories() {
    for unreadable in [false, true] {
        let mut fixture = Fixture::for_source_permissions();
        let repository_source = fixture.source.clone();
        let repository_store = fixture.store.clone();
        let git = fixture.source.join(".git");
        fs::create_dir(&git).unwrap();
        fixture.assign_child_ownership(&git);
        let private = fixture.source.join("private.md");
        fs::write(&private, "sharedtoken private repository document").unwrap();
        fixture.assign_child_ownership(&private);

        // HOME is scoped to each production child. Exercise the real Git
        // configuration loader without changing the test process environment.
        let home = fixture.directory.path().join("home");
        fs::create_dir(&home).unwrap();
        fixture.assign_child_ownership(&home);
        let global_ignore = home.join("global-ignore");
        fs::write(&global_ignore, "private.md\n").unwrap();
        fixture.assign_child_ownership(&global_ignore);
        let git_config = home.join(".gitconfig");
        fs::write(
            &git_config,
            format!("[core]\n    excludesFile = {}\n", global_ignore.display()),
        )
        .unwrap();
        fixture.assign_child_ownership(&git_config);
        let mut index = fixture.lexical_discovery_command("index");
        index
            .env("FSFS_COMPLETE_GENERATIONS", "1")
            .arg("--no-watch-mode");
        let first = json_output(&execute(&mut index), true);
        assert_eq!(first["data"]["indexed_files"], 1, "{first}");
        assert_search_membership(&fixture, &["alpha.md"]);
        let predecessor = PathBuf::from(first["data"]["generation_path"].as_str().unwrap());
        let predecessor_files = generation_files(&predecessor);
        let pointer = fixture.store.join(COMPLETE_GENERATION_POINTER);
        let selection = fs::read(&pointer).unwrap();
        let blocked = if unreadable {
            Some(UnreadableSource::new(global_ignore.clone()))
        } else {
            fs::write(&global_ignore, "private[9-0].md\n").unwrap();
            None
        };

        let mut recovery = fixture.lexical_discovery_command("search");
        recovery
            .args(["--no-daemon", "--recover-source"])
            .arg(&fixture.source);
        for command in [&mut index, &mut recovery] {
            let refused = json_output(&execute(command), false);
            assert_eq!(refused["error"]["code"], "invalid_config", "{refused}");
            assert_eq!(
                refused["error"]["field"], "discovery.ignore_rules",
                "{refused}"
            );
            assert_eq!(fs::read(&pointer).unwrap(), selection);
            assert_eq!(generation_files(&predecessor), predecessor_files);
            assert_search_membership(&fixture, &["alpha.md"]);
        }

        // The same broken global policy is inactive for a source outside a
        // repository. Both documents remain indexable there, including the
        // basename the valid repository policy excluded above.
        fixture.source = fixture.directory.path().join("plain-source");
        fixture.store = fixture.directory.path().join("plain-store");
        fs::create_dir(&fixture.source).unwrap();
        fixture.assign_child_ownership(&fixture.source);
        for name in ["alpha.md", "private.md"] {
            let path = fixture.source.join(name);
            fs::write(&path, "sharedtoken non-repository document").unwrap();
            fixture.assign_child_ownership(&path);
        }
        let plain = json_output(
            &execute(
                fixture
                    .lexical_discovery_command("index")
                    .env("FSFS_COMPLETE_GENERATIONS", "1")
                    .arg("--no-watch-mode"),
            ),
            true,
        );
        assert_eq!(plain["data"]["indexed_files"], 2, "{plain}");
        let recovered_plain = json_output(
            &execute(
                fixture
                    .lexical_discovery_command("search")
                    .args(["--no-daemon", "--recover-source"])
                    .arg(&fixture.source),
            ),
            true,
        );
        assert_eq!(
            recovered_plain["data"]["returned_hits"], 2,
            "{recovered_plain}"
        );
        assert_search_membership(&fixture, &["alpha.md", "private.md"]);
        assert_eq!(fs::read(&pointer).unwrap(), selection);
        assert_eq!(generation_files(&predecessor), predecessor_files);

        if let Some(blocked) = blocked.as_ref() {
            blocked.restore();
        }
        fs::write(&global_ignore, "private.md\n").unwrap();
        fixture.source = repository_source;
        fixture.store = repository_store;
        let retry = json_output(&execute(&mut index), true);
        assert_eq!(retry["data"]["indexed_files"], 1, "{retry}");
        assert_ne!(fs::read(&pointer).unwrap(), selection);
        assert_eq!(generation_files(&predecessor), predecessor_files);
        assert_search_membership(&fixture, &["alpha.md"]);
    }
}

#[test]
fn complete_cli_binary_rejects_corrupt_selection_without_legacy_fallback() {
    let fixture = Fixture::new();
    fs::create_dir(&fixture.store).unwrap();
    let pointer = fixture.store.join(COMPLETE_GENERATION_POINTER);
    fs::write(&pointer, "corrupt complete selection").unwrap();
    // No opt-in variable: the executable must still dispatch to the store.
    let output = execute(fixture.command("search", "json").arg("--no-daemon"));
    let value = json_output(&output, false);
    assert!(
        value["error"]
            .to_string()
            .contains("complete-generation pointer")
    );
    assert_eq!(
        fs::read_to_string(pointer).unwrap(),
        "corrupt complete selection"
    );
    assert!(!fixture.store.join("vector").exists());
}

#[test]
fn complete_search_recovers_a_corrupt_cache_without_logging_private_values() {
    const PRIVATE_VALUE: &str = "private-cache-content-canary";
    let fixture = Fixture::new();
    json_output(
        &execute(
            fixture
                .lexical_discovery_command("index")
                .arg("--no-watch-mode"),
        ),
        true,
    );
    let mut search = fixture.lexical_discovery_command("search");
    search
        .arg("--no-daemon")
        .env("FRANKENSEARCH_LOG", "frankensearch_fsfs=warn");
    let cold = json_output(&execute(&mut search), true);
    assert_eq!(cold["data"]["hits"][0]["path"], "alpha.md");
    let mut records = fs::read_dir(fixture.store.join("query_cache")).unwrap();
    let path = records.next().unwrap().unwrap().path();
    assert!(records.next().is_none());
    let mut record: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    let mut bad_counts = record.clone();
    bad_counts["payloads"][0]["returned_hits"] = Value::from(0);
    fs::write(&path, serde_json::to_vec(&bad_counts).unwrap()).unwrap();
    let repaired_counts = json_output(&execute(&mut search), true);
    assert_eq!(repaired_counts["data"]["returned_hits"], 1);
    assert_eq!(repaired_counts["data"]["hits"], cold["data"]["hits"]);

    let mut bad_rank = record.clone();
    bad_rank["payloads"][0]["hits"][0]["rank"] = Value::from(3);
    fs::write(&path, serde_json::to_vec(&bad_rank).unwrap()).unwrap();
    let streamed = execute(fixture.lexical_discovery_command("search").args([
        "--no-daemon",
        "--format",
        "jsonl",
        "--stream",
    ]));
    assert!(streamed.status.success(), "{streamed:?}");
    let frames: Vec<Value> = std::str::from_utf8(&streamed.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let mut results = frames.iter().filter(|frame| frame["event"] == "result");
    let hit = &results.next().unwrap()["payload"];
    assert_eq!(hit["rank"], 1);
    assert_eq!(hit["item"], cold["data"]["hits"][0]);
    assert!(results.next().is_none());
    assert_eq!(frames.last().unwrap()["payload"]["status"], "completed");

    record["payloads"][0]["phase"] = Value::from(PRIVATE_VALUE);
    fs::write(&path, serde_json::to_vec(&record).unwrap()).unwrap();

    let repaired = execute(&mut search);
    assert_eq!(
        json_output(&repaired, true)["data"]["hits"],
        cold["data"]["hits"]
    );
    let warning = std::str::from_utf8(&repaired.stderr).unwrap();
    assert!(
        warning.contains("complete search cache read failed"),
        "{warning}"
    );
    assert!(warning.contains("subsystem_error"), "{warning}");
    assert!(!warning.contains(PRIVATE_VALUE), "{warning}");
    let recovered: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    assert_eq!(recovered["payloads"][0]["phase"], "initial");
}

#[test]
fn complete_cli_binary_default_search_without_a_daemon_runs_in_process() {
    let fixture = Fixture::new();
    fs::create_dir(&fixture.store).unwrap();
    let pointer = fixture.store.join(COMPLETE_GENERATION_POINTER);
    fs::write(&pointer, "corrupt complete selection").unwrap();
    // No --no-daemon and no daemon running: the default transport used to turn
    // the absent socket into "I/O error: No such file or directory". The
    // store's own admission must answer instead, without repairing anything.
    let output = execute(&mut fixture.command("search", "json"));
    let value = json_output(&output, false);
    assert!(
        value["error"]
            .to_string()
            .contains("complete-generation pointer"),
        "{value}"
    );
    assert_eq!(
        fs::read_to_string(pointer).unwrap(),
        "corrupt complete selection"
    );
    assert_eq!(fs::read_dir(&fixture.store).unwrap().count(), 1);
}

fn exclusion_only_fixture() -> Fixture {
    let fixture = Fixture::new();
    fs::write(fixture.source.join("alpha.md"), "sharedtoken rank fusion").unwrap();
    fs::write(
        fixture.source.join("beta.md"),
        "sharedtoken ordinary document",
    )
    .unwrap();
    fs::write(
        fixture.source.join("gamma.md"),
        "sharedtoken rank separated from fusion",
    )
    .unwrap();
    json_output(
        &execute(
            fixture
                .lexical_discovery_command("index")
                .arg("--no-watch-mode"),
        ),
        true,
    );

    // The ordinary search path must recognize a lexical complement before
    // resolving either producer. No model fixture or forced degradation may
    // hide an attempted semantic admission, even in loader-capable builds.
    let mut config = FsfsConfig::default();
    config.indexing.offline = true;
    config.indexing.model_dir = fixture
        .directory
        .path()
        .join("missing-models")
        .display()
        .to_string();
    "unused-exclusion-fast-producer".clone_into(&mut config.indexing.fast_model);
    "unused-exclusion-quality-producer".clone_into(&mut config.indexing.quality_model);
    config.search.fast_only = false;
    config.search.rerank = false;
    "{index_dir}/catalog.sqlite".clone_into(&mut config.storage.db_path);
    fs::write(&fixture.config, config.to_toml().unwrap()).unwrap();
    fixture
}

#[test]
fn complete_cli_binary_exclusion_only_queries_browse_without_model_admission() {
    let fixture = exclusion_only_fixture();
    let selection = fs::read(fixture.store.join(COMPLETE_GENERATION_POINTER)).unwrap();
    for (query, expected) in [
        ("-fusion", vec!["beta.md"]),
        ("NOT \"rank fusion\"", vec!["beta.md", "gamma.md"]),
        ("-absenttoken", vec!["alpha.md", "beta.md", "gamma.md"]),
    ] {
        for fast_only in [false, true] {
            // Repeat the exact command to cover persisted answer reuse as
            // well as the initial request through the actual executable.
            for _ in 0..2 {
                let result = json_output(
                    &execute(fixture.command("s", "json").args([
                        "--no-daemon",
                        "--limit",
                        "all",
                        if fast_only {
                            "--fast-only"
                        } else {
                            "--no-fast-only"
                        },
                        "--",
                        query,
                    ])),
                    true,
                );
                let payload = &result["data"];
                assert_eq!(payload["query"], query, "{result}");
                assert_eq!(payload["skip_reason"], "lexical_complement", "{result}");
                assert!(payload["vector_generation_id"].is_null(), "{result}");
                assert!(payload["semantic_blend"].is_null(), "{result}");
                let hits = payload["hits"].as_array().expect("complement hits");
                let actual = hits
                    .iter()
                    .map(|hit| {
                        assert!(hit["semantic_rank"].is_null(), "{hit}");
                        assert!(hit["hash_rank"].is_null(), "{hit}");
                        assert!(hit["lexical_rank"].is_number(), "{hit}");
                        hit["path"].as_str().expect("hit path")
                    })
                    .collect::<BTreeSet<_>>();
                assert_eq!(actual, expected.iter().copied().collect(), "{result}");
                assert_eq!(hits.len(), expected.len(), "{result}");
            }
        }
    }
    assert_eq!(
        fs::read(fixture.store.join(COMPLETE_GENERATION_POINTER)).unwrap(),
        selection,
    );
    assert!(!fixture.directory.path().join("missing-models").exists());
}

#[test]
fn complete_cli_binary_exclusion_only_stream_never_admits_semantic_results() {
    use frankensearch_fsfs::stream_protocol::{
        StreamEvent, StreamFrame, StreamTerminalStatus, decode_stream_frame_ndjson,
    };

    let fixture = exclusion_only_fixture();
    let output = execute(fixture.command("s", "jsonl").args([
        "--no-daemon",
        "--stream",
        "--limit",
        "all",
        "--",
        "NOT \"rank fusion\"",
    ]));
    assert!(output.status.success(), "{output:?}");
    let frames: Vec<StreamFrame<Value>> = std::str::from_utf8(&output.stdout)
        .unwrap()
        .lines()
        .map(|line| decode_stream_frame_ndjson(line).expect("stream frame"))
        .collect();
    let Some(StreamEvent::Started(started)) = frames.first().map(|frame| &frame.event) else {
        panic!("missing started frame: {output:?}");
    };
    assert!(!started.semantic_admitted);
    assert!(started.vector_generation_id.is_none());
    assert_eq!(started.skip_reason.as_deref(), Some("lexical_complement"));
    let hits = frames
        .iter()
        .filter_map(|frame| match &frame.event {
            StreamEvent::Result(result) => {
                assert!(result.item["semantic_rank"].is_null());
                Some(result.item["path"].as_str().expect("hit path"))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(hits.len(), 2, "{output:?}");
    assert_eq!(
        hits.into_iter().collect::<BTreeSet<_>>(),
        BTreeSet::from(["beta.md", "gamma.md"])
    );
    let Some(StreamEvent::Terminal(terminal)) = frames.last().map(|frame| &frame.event) else {
        panic!("missing terminal frame: {output:?}");
    };
    assert_eq!(terminal.status, StreamTerminalStatus::Completed);
    assert_eq!(terminal.exit_code, 0);
}

#[test]
fn complete_cli_binary_exclusion_only_named_daemon_needs_no_models() {
    use frankensearch_fsfs::stream_protocol::{
        StreamEvent, StreamFrame, StreamTerminalStatus, decode_stream_frame_ndjson,
    };
    use std::process::{Child, Stdio};

    struct Daemon(Child);
    impl Drop for Daemon {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    let fixture = exclusion_only_fixture();
    let selection = fs::read(fixture.store.join(COMPLETE_GENERATION_POINTER)).unwrap();
    let socket_root = tempfile::tempdir_in("/tmp").expect("short Unix socket directory");
    let socket = socket_root.path().join("query.sock");
    let daemon_stdout = fixture.directory.path().join("daemon.stdout");
    let daemon_stderr = fixture.directory.path().join("daemon.stderr");
    let mut daemon = Daemon(
        fixture
            .command("daemon", "json")
            .arg("--daemon-socket")
            .arg(&socket)
            .args(["--idle-timeout-ms", "30000"])
            .stdin(Stdio::null())
            .stdout(fs::File::create(&daemon_stdout).unwrap())
            .stderr(fs::File::create(&daemon_stderr).unwrap())
            .spawn()
            .expect("start model-free complement daemon"),
    );
    let started = Instant::now();
    while !socket.exists() {
        let status = daemon.0.try_wait().unwrap();
        assert!(
            status.is_none() && started.elapsed() < Duration::from_secs(20),
            "daemon did not become ready: {status:?}\nstdout: {}\nstderr: {}",
            fs::read_to_string(&daemon_stdout).unwrap(),
            fs::read_to_string(&daemon_stderr).unwrap()
        );
        std::thread::sleep(Duration::from_millis(10));
    }

    for streamed in [false, false, true, true] {
        let mut command = fixture.command("s", if streamed { "jsonl" } else { "json" });
        command.args(["--daemon", "--daemon-socket"]).arg(&socket);
        if streamed {
            command.arg("--stream");
        }
        command.args(["--limit", "all", "--", "NOT \"rank fusion\""]);
        let output = execute(&mut command);
        assert!(output.status.success(), "{output:?}");
        if streamed {
            let frames: Vec<StreamFrame<Value>> = std::str::from_utf8(&output.stdout)
                .unwrap()
                .lines()
                .map(|line| decode_stream_frame_ndjson(line).expect("stream frame"))
                .collect();
            let Some(StreamEvent::Started(started)) = frames.first().map(|frame| &frame.event)
            else {
                panic!("missing started frame: {output:?}");
            };
            assert!(!started.semantic_admitted);
            assert!(started.vector_generation_id.is_none());
            assert_eq!(started.skip_reason.as_deref(), Some("lexical_complement"));
            let hits = frames
                .iter()
                .filter_map(|frame| match &frame.event {
                    StreamEvent::Result(result) => {
                        assert!(result.item["semantic_rank"].is_null());
                        Some(result.item["path"].as_str().expect("hit path"))
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(hits.len(), 2, "{output:?}");
            assert_eq!(
                hits.into_iter().collect::<BTreeSet<_>>(),
                BTreeSet::from(["beta.md", "gamma.md"])
            );
            let Some(StreamEvent::Terminal(terminal)) = frames.last().map(|frame| &frame.event)
            else {
                panic!("missing terminal frame: {output:?}");
            };
            assert_eq!(terminal.status, StreamTerminalStatus::Completed);
            assert_eq!(terminal.exit_code, 0);
        } else {
            let result = json_output(&output, true);
            let payload = &result["data"];
            assert_eq!(payload["skip_reason"], "lexical_complement", "{result}");
            assert!(payload["vector_generation_id"].is_null(), "{result}");
            let hits = payload["hits"].as_array().unwrap();
            assert_eq!(hits.len(), 2, "{result}");
            assert_eq!(
                hits.iter()
                    .map(|hit| hit["path"].as_str().unwrap())
                    .collect::<BTreeSet<_>>(),
                BTreeSet::from(["beta.md", "gamma.md"])
            );
        }
    }
    assert_eq!(
        fs::read(fixture.store.join(COMPLETE_GENERATION_POINTER)).unwrap(),
        selection
    );
    assert!(!fixture.directory.path().join("missing-models").exists());
}

#[test]
fn complete_cli_binary_quality_only_requires_positive_text_before_model_admission() {
    let fixture = exclusion_only_fixture();
    let selection = fs::read(fixture.store.join(COMPLETE_GENERATION_POINTER)).unwrap();
    let output = execute(fixture.command("s", "json").args([
        "--no-daemon",
        "--quality-only",
        "--no-fast-only",
        "--",
        "-fusion",
    ]));
    let result = json_output(&output, false);
    assert_eq!(output.status.code(), Some(2), "{result}");
    assert_eq!(result["error"]["code"], "invalid_config", "{result}");
    assert_eq!(result["error"]["field"], "cli.search_query", "{result}");
    assert!(
        result["error"]["message"]
            .as_str()
            .unwrap()
            .contains("positive query text"),
        "{result}"
    );
    assert_eq!(
        fs::read(fixture.store.join(COMPLETE_GENERATION_POINTER)).unwrap(),
        selection
    );
    assert!(!fixture.directory.path().join("missing-models").exists());
}

#[test]
fn stream_encoding_refusal_is_reported_before_output_ownership() {
    use frankensearch_fsfs::stream_protocol::{
        StreamEvent, StreamFrame, StreamTerminalStatus, decode_stream_frame_ndjson,
    };

    let fixture = Fixture::new();
    json_output(
        &execute(
            fixture
                .lexical_discovery_command("index")
                .arg("--no-watch-mode"),
        ),
        true,
    );
    let query = "sharedtoken\u{001e}";
    for format in ["jsonl", "toon"] {
        // The alias avoids the helper's fixed search positional argument.
        let output = execute(fixture.lexical_discovery_command("s").arg(query).args([
            "--format",
            format,
            "--stream",
            "--no-daemon",
        ]));
        if format == "toon" {
            assert_eq!(output.status.code(), Some(2), "{output:?}");
            assert!(output.stdout.is_empty(), "{output:?}");
            let diagnostic = String::from_utf8(output.stderr).expect("UTF-8 public error");
            assert!(diagnostic.contains("[invalid_config]"), "{diagnostic}");
            assert!(diagnostic.contains("--format jsonl"), "{diagnostic}");
            assert!(!diagnostic.contains('\u{001e}'));
        } else {
            assert!(output.status.success(), "{output:?}");
            let text = std::str::from_utf8(&output.stdout).expect("UTF-8 JSONL");
            assert!(!text.contains('\u{001e}'));
            let frames: Vec<StreamFrame<Value>> = text
                .lines()
                .map(|line| decode_stream_frame_ndjson(line).expect("JSONL stream frame"))
                .collect();
            let Some(StreamEvent::Started(started)) = frames.first().map(|frame| &frame.event)
            else {
                panic!("started frame: {output:?}");
            };
            assert_eq!(started.query, query);
            let Some(StreamEvent::Terminal(terminal)) = frames.last().map(|frame| &frame.event)
            else {
                panic!("terminal frame: {output:?}");
            };
            assert_eq!(terminal.status, StreamTerminalStatus::Completed);
            assert_eq!(terminal.exit_code, 0);
        }
    }
}

#[test]
fn named_daemon_refusal_preserves_public_error_in_stream_terminal() {
    use frankensearch_fsfs::stream_protocol::{
        StreamEvent, StreamFailureCategory, StreamFrame, StreamRetryDirective, StreamTerminalEvent,
        StreamTerminalStatus, TOON_STREAM_RECORD_SEPARATOR, decode_stream_frame_ndjson,
        decode_stream_frame_toon, validate_stream_frame,
    };
    use std::process::{Child, Stdio};

    struct Daemon(Child);
    impl Drop for Daemon {
        fn drop(&mut self) {
            // This guard owns exactly the foreground process spawned below.
            // A failed assertion must not leave a query daemon behind.
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    let frames = |output: &Output, format: &str, expected_query: &str| -> Vec<StreamFrame<Value>> {
        let text = std::str::from_utf8(&output.stdout).expect("UTF-8 stream");
        let frames = if format == "jsonl" {
            text.lines()
                .filter(|line| !line.trim().is_empty())
                .map(|line| decode_stream_frame_ndjson(line).expect("JSONL stream frame"))
                .collect::<Vec<_>>()
        } else {
            assert!(text.starts_with(TOON_STREAM_RECORD_SEPARATOR), "{text}");
            text.split(TOON_STREAM_RECORD_SEPARATOR)
                .filter(|record| !record.trim().is_empty())
                .map(|record| decode_stream_frame_toon(record).expect("TOON stream frame"))
                .collect::<Vec<_>>()
        };
        assert!(!frames.is_empty(), "{output:?}");
        for (sequence, frame) in frames.iter().enumerate() {
            assert!(validate_stream_frame(frame).valid, "{frame:?}");
            assert_eq!(frame.seq, u64::try_from(sequence).unwrap());
            assert_eq!(frame.stream_id, frames[0].stream_id);
        }
        let StreamEvent::Started(started) = &frames[0].event else {
            panic!("started frame");
        };
        assert_eq!(started.query, expected_query, "{format}: {output:?}");
        assert_eq!(started.stream_id, frames[0].stream_id);
        assert_eq!(started.format, format);
        assert_eq!(
            frames
                .iter()
                .filter(|frame| matches!(frame.event, StreamEvent::Terminal(_)))
                .count(),
            1
        );
        assert!(matches!(
            frames.last().unwrap().event,
            StreamEvent::Terminal(_)
        ));
        frames
    };

    let fixture = Fixture::new();
    json_output(
        &execute(
            fixture
                .lexical_discovery_command("index")
                .arg("--no-watch-mode"),
        ),
        true,
    );
    assert!(fixture.store.join(COMPLETE_GENERATION_POINTER).is_file());
    // Remote Cargo places TMPDIR beneath its long source namespace. The
    // socket needs its own short path; the indexed store stays unchanged.
    let socket_root = tempfile::tempdir_in("/tmp").expect("short Unix socket directory");
    let socket = socket_root.path().join("query.sock");
    assert!(socket.as_os_str().len() < 100, "short socket test fixture");
    let daemon_stdout = fixture.directory.path().join("daemon.stdout");
    let daemon_stderr = fixture.directory.path().join("daemon.stderr");
    let mut daemon = Daemon(
        fixture
            .lexical_discovery_command("serve")
            .args(["--daemon", "--daemon-socket"])
            .arg(&socket)
            .args(["--idle-timeout-ms", "30000"])
            .stdin(Stdio::null())
            .stdout(fs::File::create(&daemon_stdout).unwrap())
            .stderr(fs::File::create(&daemon_stderr).unwrap())
            .spawn()
            .expect("start production query daemon"),
    );
    let started = Instant::now();
    while !socket.exists() {
        let status = daemon.0.try_wait().unwrap();
        assert!(
            status.is_none() && started.elapsed() < Duration::from_secs(20),
            "daemon did not become ready: {status:?}\nstdout: {}\nstderr: {}",
            fs::read_to_string(&daemon_stdout).unwrap(),
            fs::read_to_string(&daemon_stderr).unwrap()
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    let query = |format: &str, streamed: bool| {
        let mut command = fixture.lexical_discovery_command("search");
        command.args(["--daemon", "--daemon-socket"]).arg(&socket);
        command.args(["--format", format]);
        if streamed {
            command.arg("--stream");
        }
        execute(&mut command)
    };
    assert_eq!(
        json_output(&query("json", false), true)["data"]["hits"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    for format in ["jsonl", "toon"] {
        let output = query(format, true);
        assert!(output.status.success(), "{output:?}");
        let frames = frames(&output, format, "sharedtoken");
        let StreamEvent::Terminal(terminal) = &frames.last().unwrap().event else {
            panic!("terminal frame");
        };
        assert_eq!(terminal.status, StreamTerminalStatus::Completed);
        assert_eq!(terminal.exit_code, 0);
    }

    let separator_query = "sharedtoken\u{001e}";
    for format in ["jsonl", "toon"] {
        let output = execute(
            fixture
                .lexical_discovery_command("s")
                .arg(separator_query)
                .args(["--daemon", "--daemon-socket"])
                .arg(&socket)
                .args(["--format", format, "--stream"]),
        );
        if format == "toon" {
            assert_eq!(output.status.code(), Some(2), "{output:?}");
            assert!(output.stdout.is_empty(), "refuse before any TOON frame");
            let diagnostic = String::from_utf8(output.stderr).expect("UTF-8 public error");
            assert!(diagnostic.contains("[invalid_config]"), "{diagnostic}");
            assert!(diagnostic.contains("--format jsonl"), "{diagnostic}");
            assert!(!diagnostic.contains('\u{001e}'));
        } else {
            assert!(output.status.success(), "{output:?}");
            let frames = frames(&output, format, separator_query);
            assert!(matches!(&frames.last().unwrap().event,
                StreamEvent::Terminal(terminal) if terminal.status == StreamTerminalStatus::Completed
                    && terminal.exit_code == 0));
        }
    }

    let mut config: FsfsConfig =
        toml::from_str(&fs::read_to_string(&fixture.config).unwrap()).unwrap();
    // Shadowing stays disabled. This is a harmless warmed-configuration
    // mismatch, not a model/policy change or an invalid local configuration.
    config.search.shadow_sample_rate_basis_points += 1;
    fs::write(&fixture.config, config.to_toml().unwrap()).unwrap();
    let buffered = query("json", false);
    let expected = json_output(&buffered, false)["error"].clone();
    // The established usage-error contract is 2 (output_schema::exit_code_for),
    // including the buffered daemon refusal; stream metadata must agree.
    assert_eq!(buffered.status.code(), Some(2), "{buffered:?}");
    assert_eq!(expected["code"], "invalid_config");
    assert_eq!(expected["field"], "complete_generation.daemon_config");
    assert_eq!(expected["exit_code"], 2);
    let refused = ["jsonl", "toon"].map(|format| (format, query(format, true)));
    let stopped = execute(
        fixture
            .lexical_discovery_command("daemon")
            .args(["--stop", "--daemon-socket"])
            .arg(&socket),
    );
    assert!(stopped.status.success(), "{stopped:?}");
    let stopping = Instant::now();
    while daemon.0.try_wait().unwrap().is_none() {
        assert!(stopping.elapsed() < Duration::from_secs(20));
        std::thread::sleep(Duration::from_millis(10));
    }
    let actual = refused.map(|(format, output)| {
        assert_eq!(output.status.code(), Some(2), "{format}: {output:?}");
        let frames = frames(&output, format, "sharedtoken");
        assert_eq!(frames.len(), 2, "a refused query must emit no result rows");
        let StreamEvent::Terminal(terminal) = &frames[1].event else {
            panic!("terminal frame");
        };
        (format, terminal.clone())
    });
    let expected = StreamTerminalEvent {
        status: StreamTerminalStatus::Failed,
        exit_code: 2,
        failure_category: Some(StreamFailureCategory::Config),
        error: Some(serde_json::from_value(expected).unwrap()),
        retry: StreamRetryDirective::None,
    };
    assert_eq!(actual, [("jsonl", expected.clone()), ("toon", expected)]);
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
            .arg("--no-daemon"),
    );
    let value = json_output(&output, false);
    assert!(
        value["error"]
            .to_string()
            .contains("no complete generation")
    );
    assert_eq!(
        fs::read_to_string(artifact).unwrap(),
        "abandoned build evidence"
    );
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
            .env("FRANKENSEARCH_COMPLETE_GENERATIONS", "true"),
    );
    let value = json_output(&output, false);
    assert!(
        value["error"]
            .to_string()
            .contains("without parent traversal")
    );
    assert!(!fixture.store.exists());
    assert_eq!(fs::read_to_string(outside).unwrap(), "unrelated database");
}

#[test]
fn complete_cli_binary_rejects_malformed_opt_in_before_writing() {
    let fixture = Fixture::new();
    let output = execute(
        fixture
            .command("index", "json")
            .env("FSFS_COMPLETE_GENERATIONS", "not-a-boolean"),
    );
    let value = json_output(&output, false);
    assert!(
        value["error"]
            .to_string()
            .contains("FSFS_COMPLETE_GENERATIONS")
    );
    assert!(!fixture.store.exists());
}

#[test]
fn complete_cli_binary_quality_only_conflicts_do_not_create_a_store() {
    for flag in ["--fast-only", "--complete-quality"] {
        let fixture = Fixture::new();
        fixture.write_quality_only_config(&fixture.directory.path().join("missing-models"));
        let output = execute(
            fixture
                .command("index", "json")
                .args(["--quality-only", flag]),
        );
        let refused = json_output(&output, false);
        assert_eq!(
            refused["error"]["code"], "invalid_config",
            "{flag}: {refused}"
        );
        assert!(!fixture.store.exists(), "{flag} created a store");
        assert_eq!(
            fs::read_to_string(fixture.source.join("alpha.md")).unwrap(),
            "sharedtoken alpha document"
        );
    }
}

#[cfg(all(feature = "semantic-support", feature = "rerank"))]
#[test]
#[ignore = "requires FSFS_COMPLETE_GENERATION_TEST_MODEL_DIR with verified native MiniLM artifacts"]
fn complete_cli_binary_quality_only_build_reopen_and_mutate_without_fast_model() {
    fn search_paths(fixture: &Fixture) -> BTreeSet<String> {
        let result = json_output(
            &execute(fixture.command("search", "json").args([
                "--quality-only",
                "--no-daemon",
                "--limit",
                "all",
            ])),
            true,
        );
        result["data"]["hits"]
            .as_array()
            .expect("quality hits")
            .iter()
            .map(|hit| {
                assert!(hit["semantic_rank"].is_number(), "{result}");
                assert!(hit["lexical_rank"].is_null(), "{result}");
                assert!(hit["score"].as_f64().is_some_and(f64::is_finite));
                hit["path"].as_str().expect("quality hit path").to_owned()
            })
            .collect()
    }

    fn assert_standalone_quality(receipt: &Value) -> PathBuf {
        assert_eq!(receipt["data"]["publication"], "durable", "{receipt}");
        let path = PathBuf::from(receipt["data"]["generation_path"].as_str().unwrap());
        assert!(path.join("vector/quality.fsvi").is_file());
        assert!(!path.join("vector/index.fsvi").exists());
        path
    }

    let models = fs::canonicalize(
        std::env::var_os("FSFS_COMPLETE_GENERATION_TEST_MODEL_DIR")
            .expect("provide a verified native MiniLM model-cache root"),
    )
    .expect("model-cache root");
    let fixture = Fixture::new();
    fixture.write_quality_only_config(&models);
    let first = json_output(
        &execute(fixture.command("index", "json").arg("--quality-only")),
        true,
    );
    let first_path = assert_standalone_quality(&first);
    let first_files = generation_files(&first_path);
    assert_eq!(
        search_paths(&fixture),
        BTreeSet::from(["alpha.md".to_owned()])
    );

    fs::write(fixture.source.join("beta.md"), "sharedtoken beta document").unwrap();
    let second = json_output(
        &execute(fixture.command("index", "json").arg("--quality-only")),
        true,
    );
    let second_path = assert_standalone_quality(&second);
    assert_ne!(first_path, second_path);
    assert_eq!(generation_files(&first_path), first_files);
    assert_eq!(
        search_paths(&fixture),
        BTreeSet::from(["alpha.md".to_owned(), "beta.md".to_owned()])
    );

    // Both a topology switch requiring a missing fast producer and a missing
    // required quality producer must leave the selected quality generation intact.
    let selected = fs::read(fixture.store.join(COMPLETE_GENERATION_POINTER)).unwrap();
    let second_files = generation_files(&second_path);
    json_output(&execute(&mut fixture.command("index", "json")), false);
    assert_eq!(
        fs::read(fixture.store.join(COMPLETE_GENERATION_POINTER)).unwrap(),
        selected
    );
    fixture.write_quality_only_config(&fixture.directory.path().join("missing-models"));
    json_output(
        &execute(fixture.command("index", "json").arg("--quality-only")),
        false,
    );
    assert_eq!(
        fs::read(fixture.store.join(COMPLETE_GENERATION_POINTER)).unwrap(),
        selected
    );
    assert_eq!(generation_files(&second_path), second_files);
    fixture.write_quality_only_config(&models);
    assert_eq!(search_paths(&fixture).len(), 2);

    // Maintenance derives topology from the retained generation. It needs
    // neither a quality-only flag nor an available fast model.
    let appended = json_output(
        &execute_with_input(
            &mut fixture.command("append-batch", "json"),
            b"{\"id\":\"beta.md\",\"text\":\"sharedtoken revised beta document\"}\n{\"id\":\"gamma.md\",\"text\":\"sharedtoken gamma document\"}\n",
        ),
        true,
    );
    assert_standalone_quality(&appended);
    assert_eq!(
        search_paths(&fixture),
        BTreeSet::from([
            "alpha.md".to_owned(),
            "beta.md".to_owned(),
            "gamma.md".to_owned(),
        ])
    );
    assert_eq!(generation_files(&second_path), second_files);
}

#[cfg(feature = "semantic-support")]
#[test]
#[ignore = "requires verified models and a source-bound prior executable in FSFS_UPGRADE_REUSE_PRIOR_EXECUTABLE"]
fn complete_cli_binary_reuses_compatible_prior_executable_with_real_models() {
    fn executable_sha256(path: &Path) -> String {
        use std::io::Read;

        use sha2::{Digest, Sha256};

        let mut file = fs::File::open(path).unwrap();
        let mut digest = Sha256::new();
        let mut buffer = vec![0_u8; 64 * 1024];
        loop {
            let count = file.read(&mut buffer).unwrap();
            if count == 0 {
                break;
            }
            digest.update(&buffer[..count]);
        }
        frankensearch_storage::ContentHasher::to_hex(&digest.finalize().into())
    }

    fn document_embedding_batches(output: &Output) -> usize {
        std::str::from_utf8(&output.stderr)
            .unwrap()
            .lines()
            .filter(|line| line.contains("fsfs admitted indexing embedding batch"))
            .inspect(|line| eprintln!("document embedding witness: {line}"))
            .count()
    }

    fn startup_probes(output: &Output) -> usize {
        std::str::from_utf8(&output.stderr)
            .unwrap()
            .lines()
            .filter(|line| line.contains("event=\"fsfs.semantic_probe_succeeded\""))
            .count()
    }

    fn search_ranks(fixture: &Fixture) -> Vec<(Value, Value, Value)> {
        let output = execute(fixture.command("s", "json").args([
            "find a relevant document",
            "--fast-only",
            "--no-daemon",
            "--limit",
            "10",
        ]));
        let result = json_output(&output, true);
        let hits = result["data"]["hits"].as_array().unwrap();
        assert_eq!(hits.len(), 2, "{result}");
        assert_eq!(
            hits.iter()
                .map(|hit| hit["path"].as_str().unwrap())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["alpha.md", "beta.md"])
        );
        hits.iter()
            .map(|hit| {
                assert!(hit["semantic_rank"].is_number(), "{result}");
                assert!(hit["score"].is_number(), "{result}");
                (
                    hit["path"].clone(),
                    hit["semantic_rank"].clone(),
                    hit["score"].clone(),
                )
            })
            .collect()
    }

    let models = fs::canonicalize(
        std::env::var_os("FSFS_COMPLETE_GENERATION_TEST_MODEL_DIR")
            .expect("provide verified real models"),
    )
    .unwrap();
    let prior = fs::canonicalize(
        std::env::var_os("FSFS_UPGRADE_REUSE_PRIOR_EXECUTABLE")
            .expect("provide a genuine source-bound prior executable"),
    )
    .unwrap();
    let current = PathBuf::from(env!("CARGO_BIN_EXE_fsfs"));
    let prior_sha = executable_sha256(&prior);
    assert_eq!(
        prior_sha,
        std::env::var("FSFS_UPGRADE_REUSE_PRIOR_SHA256")
            .expect("provide the independently verified prior executable SHA-256")
    );
    let current_sha = executable_sha256(&current);
    assert_ne!(prior_sha, current_sha, "the executables must differ");

    let runtime = asupersync::runtime::RuntimeBuilder::current_thread()
        .build()
        .unwrap();
    let cx = runtime.request_cx_with_budget(asupersync::types::Budget::INFINITE);
    for (seed_complete, two_tier, policy, drift, carried, embeds) in [
        (true, false, "same-executable", "none", 0, true),
        (true, false, "compatible-contract", "none", 2, false),
        (true, false, "compatible-contract", "source", 1, true),
        (true, false, "compatible-contract", "window", 0, true),
        (false, false, "same-executable", "none", 0, true),
        (false, false, "compatible-contract", "none", 2, false),
        (false, true, "same-executable", "none", 0, true),
        (false, true, "compatible-contract", "none", 2, false),
        (false, true, "compatible-contract", "source", 1, true),
    ] {
        eprintln!(
            "cross-executable case prior={prior_sha} current={current_sha} seed_complete={seed_complete} two_tier={two_tier} policy={policy} drift={drift}"
        );
        let mut fixture = Fixture::new();
        fixture.write_config(Some(&models), "{index_dir}/catalog.sqlite");
        if two_tier {
            let mut config: FsfsConfig =
                toml::from_str(&fs::read_to_string(&fixture.config).unwrap()).unwrap();
            config.indexing.quality_model = "all-MiniLM-L6-v2".to_owned();
            config.search.fast_only = false;
            fs::write(&fixture.config, config.to_toml().unwrap()).unwrap();
        }
        fs::write(fixture.source.join("beta.md"), "sharedtoken beta document").unwrap();
        fixture.executable.clone_from(&prior);
        let indexing = |fixture: &Fixture, policy: &str, complete: bool| {
            let mut command = fixture.command("index", "json");
            if !complete {
                command.env("FSFS_COMPLETE_GENERATIONS", "false");
            }
            execute(
                command
                    .env(
                        "FRANKENSEARCH_LOG",
                        "frankensearch_fsfs=info,frankensearch_fsfs::indexing=debug",
                    )
                    .args(["--no-color", "--upgrade-reuse", policy]),
            )
        };
        let first_output = indexing(&fixture, "same-executable", seed_complete);
        let first = json_output(&first_output, true);
        assert_eq!(first["data"]["generation_complete"], true);
        assert_eq!(first["data"]["semantic_indexed_files"], 2);
        let seed_batches = document_embedding_batches(&first_output);
        if two_tier {
            assert_eq!(startup_probes(&first_output), 2, "cold two-tier control");
        }
        assert!(
            seed_batches > 0,
            "the log control must observe document inference"
        );
        let before_ranks = search_ranks(&fixture);
        let pinned = if seed_complete {
            let store = frankensearch_fsfs::generation_store::CompleteGenerationStore::open(
                &cx,
                &fixture.store,
            )
            .unwrap();
            let pinned = store.active(&cx).unwrap().unwrap();
            assert_eq!(
                pinned.id(),
                first["data"]["generation_id"].as_str().unwrap()
            );
            Some(pinned)
        } else {
            assert!(!fixture.store.join(COMPLETE_GENERATION_POINTER).exists());
            assert!(fixture.store.join("FSFS-LEGACY-REUSE.json").is_file());
            None
        };
        let predecessor = pinned
            .as_ref()
            .map_or(fixture.store.as_path(), |generation| generation.path());
        let before = generation_files(predecessor);

        if drift == "source" {
            let path = fixture.source.join("alpha.md");
            if two_tier {
                // Equal size and mtime still require content revalidation.
                let previous = fs::metadata(&path).unwrap();
                let before = fs::read_to_string(&path).unwrap();
                let changed = before.replace("alpha", "omega");
                assert_ne!(changed, before);
                fs::write(&path, changed).unwrap();
                fs::File::options()
                    .write(true)
                    .open(&path)
                    .unwrap()
                    .set_times(fs::FileTimes::new().set_modified(previous.modified().unwrap()))
                    .unwrap();
                let after = fs::metadata(&path).unwrap();
                assert_eq!(after.len(), previous.len());
                assert_eq!(after.modified().unwrap(), previous.modified().unwrap());
            } else {
                fs::write(path, "sharedtoken alpha revised document").unwrap();
            }
        } else if drift == "window" {
            let mut config: FsfsConfig =
                toml::from_str(&fs::read_to_string(&fixture.config).unwrap()).unwrap();
            let old_cap = config.indexing.fast_window_max_per_file;
            config.indexing.fast_window_max_per_file = if old_cap == 1 { 2 } else { 1 };
            assert_ne!(config.indexing.fast_window_max_per_file, old_cap);
            let serialized = config.to_toml().unwrap();
            let validated = frankensearch_fsfs::config::load_from_str(
                Some(&serialized),
                Some(&fixture.config),
                &HashMap::new(),
                &frankensearch_fsfs::config::CliOverrides::default(),
                fixture.directory.path(),
            )
            .expect("the changed window contract must remain a valid config");
            assert_eq!(
                validated.config.indexing.fast_window_max_per_file,
                config.indexing.fast_window_max_per_file
            );
            fs::write(&fixture.config, serialized).unwrap();
        }

        fixture.executable.clone_from(&current);
        let second_output = indexing(&fixture, policy, true);
        let second = json_output(&second_output, true);
        assert_eq!(second["data"]["generation_complete"], true);
        assert_eq!(second["data"]["semantic_indexed_files"], 2);
        assert_eq!(
            second["data"]["unchanged_files"], carried,
            "{policy}/{drift}: {second}"
        );
        assert_ne!(
            first["data"]["generation_id"],
            second["data"]["generation_id"]
        );
        let batches = document_embedding_batches(&second_output);
        assert_eq!(batches > 0, embeds, "{policy}/{drift}: {second}");
        let probes = startup_probes(&second_output);
        if two_tier {
            assert_eq!(probes, if embeds { 2 } else { 0 }, "{policy}/{drift}");
        }
        if seed_complete {
            assert_eq!(generation_files(predecessor), before);
        } else {
            for (path, bytes) in &before {
                if path == Path::new("FSFS-LEGACY-REUSE.json")
                    || matches!(
                        path.extension().and_then(|ext| ext.to_str()),
                        Some("fsvi" | "fslx")
                    )
                {
                    assert_eq!(fs::read(predecessor.join(path)).unwrap(), *bytes);
                }
            }
        }
        let after_ranks = search_ranks(&fixture);
        if policy == "compatible-contract" && drift == "none" {
            assert_eq!(after_ranks, before_ranks);
            let after = generation_files(Path::new(
                second["data"]["generation_path"].as_str().unwrap(),
            ));
            let index_artifacts = |files: BTreeMap<PathBuf, Vec<u8>>| {
                files
                    .into_iter()
                    .filter(|(path, _)| {
                        matches!(
                            path.extension().and_then(|ext| ext.to_str()),
                            Some("fsvi" | "fslx")
                        )
                    })
                    .collect::<BTreeMap<_, _>>()
            };
            let expected = index_artifacts(before);
            assert!(
                expected
                    .keys()
                    .any(|path| path.extension().unwrap() == "fsvi")
            );
            assert!(
                expected
                    .keys()
                    .any(|path| path.extension().unwrap() == "fslx")
            );
            assert_eq!(index_artifacts(after), expected);
        }
        eprintln!(
            "cross-executable reuse prior={prior_sha} current={current_sha} seed_complete={seed_complete} two_tier={two_tier} policy={policy} drift={drift} carried={carried} seed_document_batches={seed_batches} current_document_batches={batches} current_startup_probes={probes}"
        );
    }
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
                .env("FSFS_COMPLETE_GENERATIONS", "1"),
        ),
        true,
    );
    assert_eq!(first["data"]["publication"], "durable");
    let first_path = PathBuf::from(first["data"]["generation_path"].as_str().unwrap());
    let manifest = first_path.join("FSFS-BUNDLE.json");
    let first_manifest = fs::read(&manifest).unwrap();
    fs::write(fixture.source.join("beta.md"), "sharedtoken beta document").unwrap();
    // A successor needs no opt-in: the existing layout chooses the route.
    let second = json_output(&execute(&mut fixture.command("index", "json")), true);
    assert_ne!(
        first["data"]["generation_id"],
        second["data"]["generation_id"]
    );
    assert!(first_path.is_dir());
    assert_eq!(fs::read(manifest).unwrap(), first_manifest);
    // Plain search defaults to daemon transport; with none serving the store
    // it must answer in process exactly like --no-daemon.
    for plain in [false, true] {
        let mut command = fixture.command("search", "json");
        if !plain {
            command.arg("--no-daemon");
        }
        let result = json_output(&execute(&mut command), true);
        let hits = result["data"]["hits"].as_array().expect("search hits");
        assert_eq!(hits.len(), 2);
        for filename in ["alpha.md", "beta.md"] {
            assert!(
                hits.iter()
                    .any(|hit| hit["path"].as_str().unwrap().ends_with(filename))
            );
        }
    }
    let stream = execute(
        fixture
            .command("search", "jsonl")
            .args(["--no-daemon", "--stream"]),
    );
    assert!(
        stream.status.success(),
        "{}",
        String::from_utf8_lossy(&stream.stderr)
    );
    let frames = String::from_utf8(stream.stdout).unwrap();
    let frames = frames
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("stream frame"))
        .collect::<Vec<_>>();
    assert!(
        frames.len() > 2,
        "started, results and terminal are required"
    );
    assert!(frames.iter().all(|frame| frame.get("data").is_none()));

    // Exercise the actual scheduler and stdin adapter, not only the in-memory
    // serve helper: malformed input must not prevent the next cached query.
    let served = execute_with_input(
        &mut fixture.command("serve", "jsonl"),
        b"sharedtoken\n{broken\nsharedtoken\nquit\n",
    );
    assert!(
        served.status.success(),
        "{}",
        String::from_utf8_lossy(&served.stderr)
    );
    let served = String::from_utf8(served.stdout).unwrap();
    let lines = served
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("serve frame"))
        .collect::<Vec<_>>();
    assert_eq!(lines.len(), 4);
    assert_eq!(lines[0]["event"], "ready");
    assert_eq!(lines[2]["ok"], false);
    for (row, cached) in [(1, false), (3, true)] {
        assert_eq!(lines[row]["ok"], true);
        assert_eq!(lines[row]["cached"], cached);
        let phases = lines[row]["payloads"].as_array().expect("query phases");
        let hits = phases.last().unwrap()["hits"].as_array().expect("hits");
        assert_eq!(hits.len(), 2);
        for filename in ["alpha.md", "beta.md"] {
            assert!(
                hits.iter()
                    .any(|hit| hit["path"].as_str().unwrap().ends_with(filename))
            );
        }
    }
}
