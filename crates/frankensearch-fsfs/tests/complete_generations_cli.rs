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
