use std::collections::{BTreeSet, HashMap};
use std::fs;
#[cfg(unix)]
use std::os::unix::fs::{PermissionsExt, symlink};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use frankensearch_core::{
    E2E_ARTIFACT_ARTIFACTS_INDEX_JSON, E2E_ARTIFACT_REPLAY_COMMAND_TXT,
    E2E_ARTIFACT_STRUCTURED_EVENTS_JSONL, ExitStatus,
};
use frankensearch_fsfs::{
    CLI_E2E_REASON_FILESYSTEM_BINARY_BLOB_SKIPPED, CLI_E2E_REASON_FILESYSTEM_GIANT_LOG_SKIPPED,
    CLI_E2E_REASON_FILESYSTEM_MOUNT_BOUNDARY, CLI_E2E_REASON_FILESYSTEM_PERMISSION_DENIED,
    CLI_E2E_REASON_FILESYSTEM_SYMLINK_LOOP, CliE2eArtifactBundle, CliE2eRunConfig, CliE2eScenario,
    MountTable, build_cli_e2e_filesystem_chaos_bundles,
    default_cli_e2e_filesystem_chaos_scenarios, read_system_mounts, replay_command_for_scenario,
};
use serde::Deserialize;

#[derive(Debug)]
struct E2eCommandContext {
    fsfs_bin: PathBuf,
    home_dir: PathBuf,
    xdg_config_home: PathBuf,
    xdg_cache_home: PathBuf,
    xdg_data_home: PathBuf,
    model_dir: PathBuf,
}

impl E2eCommandContext {
    fn new(root: &Path) -> Self {
        let home_dir = root.join("home");
        let xdg_config_home = root.join("xdg-config");
        let xdg_cache_home = root.join("xdg-cache");
        let xdg_data_home = root.join("xdg-data");
        let model_dir = root.join("models");
        fs::create_dir_all(&home_dir).expect("create test home");
        fs::create_dir_all(&xdg_config_home).expect("create XDG config home");
        fs::create_dir_all(&xdg_cache_home).expect("create XDG cache home");
        fs::create_dir_all(&xdg_data_home).expect("create XDG data home");
        fs::create_dir_all(&model_dir).expect("create model dir");

        Self {
            fsfs_bin: fsfs_binary_path(),
            home_dir,
            xdg_config_home,
            xdg_cache_home,
            xdg_data_home,
            model_dir,
        }
    }

    fn run(&self, cwd: &Path, args: &[String]) -> Output {
        Command::new(&self.fsfs_bin)
            .args(args)
            .current_dir(cwd)
            .env("HOME", &self.home_dir)
            .env("XDG_CONFIG_HOME", &self.xdg_config_home)
            .env("XDG_CACHE_HOME", &self.xdg_cache_home)
            .env("XDG_DATA_HOME", &self.xdg_data_home)
            .env("FRANKENSEARCH_MODEL_DIR", &self.model_dir)
            .env("FRANKENSEARCH_OFFLINE", "1")
            .env("FRANKENSEARCH_ALLOW_DOWNLOAD", "0")
            .env("NO_COLOR", "1")
            .output()
            .expect("spawn fsfs process")
    }
}

#[derive(Debug, Deserialize)]
struct IndexSentinel {
    #[serde(rename = "discovered_files")]
    discovered: usize,
    #[serde(rename = "indexed_files")]
    indexed: usize,
    #[serde(rename = "skipped_files")]
    skipped: usize,
}

#[derive(Debug, Deserialize)]
struct IndexManifestEntry {
    reason_code: String,
}

#[derive(Debug)]
struct BinaryChaosResult {
    output: Output,
    sentinel: IndexSentinel,
    lexical_manifest: Vec<IndexManifestEntry>,
    bundle: CliE2eArtifactBundle,
}

fn fsfs_binary_path() -> PathBuf {
    std::env::var_os("CARGO_BIN_EXE_fsfs")
        .map(PathBuf::from)
        .expect("cargo must provide CARGO_BIN_EXE_fsfs for integration tests")
}

fn scenario_by_id(id: &str) -> CliE2eScenario {
    default_cli_e2e_filesystem_chaos_scenarios()
        .into_iter()
        .find(|scenario| scenario.id == id)
        .expect("filesystem chaos scenario should exist")
}

fn load_sentinel(index_root: &Path) -> IndexSentinel {
    let sentinel_path = index_root.join("index_sentinel.json");
    let raw = fs::read_to_string(&sentinel_path).expect("read index sentinel");
    serde_json::from_str(&raw).expect("parse index sentinel")
}

fn load_manifest(index_root: &Path) -> Vec<IndexManifestEntry> {
    let manifest_path = index_root.join("lexical/index_manifest.json");
    let raw = fs::read_to_string(&manifest_path).expect("read lexical manifest");
    serde_json::from_str(&raw).expect("parse lexical manifest")
}

fn resolve_mount_point(path: &Path) -> Option<PathBuf> {
    let table = MountTable::new(read_system_mounts(), &HashMap::new());
    table.lookup(path).map(|(entry, _policy)| entry.mount_point.clone())
}

fn write_mount_boundary_config(root: &Path, mount_point: &Path) -> PathBuf {
    let config_path = root.join("chaos_mount_boundary.toml");
    let effective_mount_point = resolve_mount_point(mount_point).unwrap_or_else(|| PathBuf::from("/"));
    let config = format!(
        "[discovery]
skip_network_mounts = true
mount_overrides = [{{ mount_point = \"{}\", category = \"nfs\" }}]
",
        effective_mount_point.display()
    );
    fs::write(&config_path, config).expect("write mount boundary config");
    config_path
}

fn write_giant_log_config(root: &Path) -> PathBuf {
    let config_path = root.join("chaos_giant_log.toml");
    let config = r"[discovery]
max_file_size_mb = 1
";
    fs::write(&config_path, config).expect("write giant log config");
    config_path
}

fn prepare_filesystem_chaos_fixture(root: &Path, scenario_id: &str) -> (PathBuf, Option<PathBuf>) {
    let corpus_root = root.join("chaos-corpus");
    fs::create_dir_all(&corpus_root).expect("create chaos corpus root");
    let baseline_path = corpus_root.join("baseline.md");
    fs::write(
        &baseline_path,
        "baseline content for deterministic filesystem chaos indexing",
    )
    .expect("write baseline fixture");

    match scenario_id {
        "cli-chaos-permission-denied" => {
            let blocked = corpus_root.join("blocked.txt");
            fs::write(&blocked, "permission denied fixture").expect("write blocked fixture");
            #[cfg(unix)]
            {
                let mut perms = fs::metadata(&blocked)
                    .expect("blocked metadata")
                    .permissions();
                perms.set_mode(0o000);
                fs::set_permissions(&blocked, perms).expect("chmod blocked fixture");
            }
            (corpus_root, None)
        }
        "cli-chaos-symlink-loop" => {
            let loop_path = corpus_root.join("loop");
            #[cfg(unix)]
            symlink("loop", &loop_path).expect("create symlink loop");
            #[cfg(not(unix))]
            fs::write(&loop_path, "symlink loop unsupported").expect("write fallback loop");
            (corpus_root, None)
        }
        "cli-chaos-mount-boundary" => {
            let nested = corpus_root.join("nested");
            fs::create_dir_all(&nested).expect("create nested fixture dir");
            fs::write(nested.join("mount-edge.md"), "mount boundary fixture")
                .expect("write mount boundary fixture");
            let config = write_mount_boundary_config(root, &corpus_root);
            (corpus_root, Some(config))
        }
        "cli-chaos-giant-log-skip" => {
            let giant_log = corpus_root.join("giant.log");
            let giant_bytes = vec![b'x'; 5 * 1024 * 1024];
            fs::write(&giant_log, giant_bytes).expect("write giant log fixture");
            let config = write_giant_log_config(root);
            (corpus_root, Some(config))
        }
        "cli-chaos-binary-blob-skip" => {
            let binary_blob = corpus_root.join("blob.bin");
            fs::write(&binary_blob, [0_u8, 159, 146, 150, 0, 42]).expect("write binary fixture");
            (corpus_root, None)
        }
        other => panic!("unknown filesystem chaos scenario: {other}"),
    }
}

fn run_binary_filesystem_chaos_scenario(scenario_id: &str) -> BinaryChaosResult {
    let scenario = scenario_by_id(scenario_id);
    let temp = tempfile::tempdir().expect("create temporary chaos workspace");
    let command_context = E2eCommandContext::new(temp.path());
    let index_root = temp.path().join("chaos-index");
    let (corpus_root, config_path) = prepare_filesystem_chaos_fixture(temp.path(), scenario_id);

    let mut args = vec![
        "index".to_owned(),
        corpus_root.display().to_string(),
        "--index-dir".to_owned(),
        index_root.display().to_string(),
        "--no-watch-mode".to_owned(),
        "--format".to_owned(),
        "json".to_owned(),
    ];
    if let Some(config) = config_path {
        args.push("--config".to_owned());
        args.push(config.display().to_string());
    }

    let output = command_context.run(temp.path(), &args);
    assert!(
        output.status.success(),
        "fsfs index failed for {scenario_id}\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let sentinel = load_sentinel(&index_root);
    let lexical_manifest = load_manifest(&index_root);
    let exit_status = if output.status.success() {
        ExitStatus::Pass
    } else {
        ExitStatus::Fail
    };
    let bundle = CliE2eArtifactBundle::build(&CliE2eRunConfig::default(), &scenario, exit_status);
    bundle.validate().expect("bundle must satisfy contract");

    BinaryChaosResult {
        output,
        sentinel,
        lexical_manifest,
        bundle,
    }
}

fn assert_bundle_matches_reason(scenario_id: &str, expected_reason: &str) {
    let scenario = scenario_by_id(scenario_id);
    assert_eq!(scenario.expected_reason_code, expected_reason);

    let bundle =
        CliE2eArtifactBundle::build(&CliE2eRunConfig::default(), &scenario, ExitStatus::Pass);
    bundle.validate().expect("bundle must satisfy contract");
    assert!(bundle.replay_command.contains("--test filesystem_chaos"));
    assert!(bundle.replay_command.contains(&format!(
        "--exact scenario_{}",
        scenario_id.replace('-', "_")
    )));
    assert!(bundle.events.iter().any(|event| {
        event
            .body
            .reason_code
            .as_deref()
            .is_some_and(|code| code == expected_reason)
    }));
}

fn assert_binary_chaos_result(
    scenario_id: &str,
    expected_reason: &str,
    min_skipped: usize,
    expect_zero_indexed: bool,
    expected_manifest_reason: Option<&str>,
    expect_manifest_empty: bool,
) {
    let result = run_binary_filesystem_chaos_scenario(scenario_id);

    assert!(
        !result.output.stdout.is_empty() || !result.output.stderr.is_empty(),
        "binary run should produce observable output"
    );
    assert!(
        result.sentinel.discovered >= 1,
        "expected discovered files for scenario {scenario_id}, got sentinel={:?}",
        result.sentinel
    );
    if min_skipped > 0 {
        assert!(
            result.sentinel.skipped >= min_skipped,
            "expected skipped files for scenario {scenario_id}, got sentinel={:?}",
            result.sentinel
        );
    }
    if expect_zero_indexed {
        assert_eq!(
            result.sentinel.indexed, 0,
            "expected zero indexed files for {scenario_id}, got sentinel={:?}",
            result.sentinel
        );
    } else {
        assert!(
            result.sentinel.indexed >= 1,
            "expected at least one indexed file for {scenario_id}, got sentinel={:?}",
            result.sentinel
        );
    }

    let manifest_reason_codes: Vec<&str> = result
        .lexical_manifest
        .iter()
        .map(|entry| entry.reason_code.as_str())
        .collect();
    assert!(
        manifest_reason_codes
            .iter()
            .all(|code| code.starts_with("index.plan.")),
        "manifest reason codes must remain canonical index plan codes, got {:?}",
        manifest_reason_codes
    );
    if expect_manifest_empty {
        assert!(
            result.lexical_manifest.is_empty(),
            "expected empty manifest for {scenario_id}, got {:?}",
            manifest_reason_codes
        );
    }
    if let Some(expected_reason_code) = expected_manifest_reason {
        assert!(
            !result.lexical_manifest.is_empty(),
            "expected non-empty manifest for {scenario_id}"
        );
        assert!(
            manifest_reason_codes
                .iter()
                .all(|code| *code == expected_reason_code),
            "expected manifest reason code {expected_reason_code} for {scenario_id}, got {:?}",
            manifest_reason_codes
        );
    }
    assert_eq!(result.bundle.scenario.id, scenario_id);
    assert_eq!(result.bundle.scenario.expected_reason_code, expected_reason);
    assert!(result.bundle.events.iter().any(|event| {
        event
            .body
            .reason_code
            .as_deref()
            .is_some_and(|code| code == expected_reason)
    }));
    assert!(
        result
            .bundle
            .replay_command
            .contains("--test filesystem_chaos")
    );
    assert!(result.bundle.replay_command.contains(&format!(
        "--exact scenario_{}",
        scenario_id.replace('-', "_")
    )));
}

#[test]
fn scenario_cli_chaos_permission_denied() {
    assert_bundle_matches_reason(
        "cli-chaos-permission-denied",
        CLI_E2E_REASON_FILESYSTEM_PERMISSION_DENIED,
    );
}

#[test]
fn scenario_cli_chaos_symlink_loop() {
    assert_bundle_matches_reason(
        "cli-chaos-symlink-loop",
        CLI_E2E_REASON_FILESYSTEM_SYMLINK_LOOP,
    );
}

#[test]
fn scenario_cli_chaos_mount_boundary() {
    assert_bundle_matches_reason(
        "cli-chaos-mount-boundary",
        CLI_E2E_REASON_FILESYSTEM_MOUNT_BOUNDARY,
    );
}

#[test]
fn scenario_cli_chaos_giant_log_skip() {
    assert_bundle_matches_reason(
        "cli-chaos-giant-log-skip",
        CLI_E2E_REASON_FILESYSTEM_GIANT_LOG_SKIPPED,
    );
}

#[test]
fn scenario_cli_chaos_binary_blob_skip() {
    assert_bundle_matches_reason(
        "cli-chaos-binary-blob-skip",
        CLI_E2E_REASON_FILESYSTEM_BINARY_BLOB_SKIPPED,
    );
}

#[test]
fn scenario_cli_chaos_permission_denied_binary_run() {
    assert_binary_chaos_result(
        "cli-chaos-permission-denied",
        CLI_E2E_REASON_FILESYSTEM_PERMISSION_DENIED,
        1,
        false,
        Some("index.plan.full_semantic_lexical"),
        false,
    );
}

#[test]
fn scenario_cli_chaos_symlink_loop_binary_run() {
    assert_binary_chaos_result(
        "cli-chaos-symlink-loop",
        CLI_E2E_REASON_FILESYSTEM_SYMLINK_LOOP,
        0,
        false,
        Some("index.plan.full_semantic_lexical"),
        false,
    );
}

#[test]
fn scenario_cli_chaos_mount_boundary_binary_run() {
    assert_binary_chaos_result(
        "cli-chaos-mount-boundary",
        CLI_E2E_REASON_FILESYSTEM_MOUNT_BOUNDARY,
        1,
        true,
        None,
        true,
    );
}

#[test]
fn scenario_cli_chaos_giant_log_skip_binary_run() {
    assert_binary_chaos_result(
        "cli-chaos-giant-log-skip",
        CLI_E2E_REASON_FILESYSTEM_GIANT_LOG_SKIPPED,
        1,
        false,
        Some("index.plan.full_semantic_lexical"),
        false,
    );
}

#[test]
fn scenario_cli_chaos_binary_blob_skip_binary_run() {
    assert_binary_chaos_result(
        "cli-chaos-binary-blob-skip",
        CLI_E2E_REASON_FILESYSTEM_BINARY_BLOB_SKIPPED,
        1,
        false,
        Some("index.plan.full_semantic_lexical"),
        false,
    );
}

#[test]
fn default_filesystem_chaos_bundle_set_has_required_artifacts_and_reasons() {
    let bundles = build_cli_e2e_filesystem_chaos_bundles(&CliE2eRunConfig::default());
    assert_eq!(bundles.len(), 5);

    let mut reasons = BTreeSet::new();
    for bundle in bundles {
        reasons.insert(bundle.scenario.expected_reason_code);

        let artifact_files: Vec<&str> = bundle
            .manifest
            .body
            .artifacts
            .iter()
            .map(|artifact| artifact.file.as_str())
            .collect();
        assert!(artifact_files.contains(&E2E_ARTIFACT_STRUCTURED_EVENTS_JSONL));
        assert!(artifact_files.contains(&E2E_ARTIFACT_ARTIFACTS_INDEX_JSON));
        assert!(artifact_files.contains(&E2E_ARTIFACT_REPLAY_COMMAND_TXT));
    }

    assert!(reasons.contains(CLI_E2E_REASON_FILESYSTEM_PERMISSION_DENIED));
    assert!(reasons.contains(CLI_E2E_REASON_FILESYSTEM_SYMLINK_LOOP));
    assert!(reasons.contains(CLI_E2E_REASON_FILESYSTEM_MOUNT_BOUNDARY));
    assert!(reasons.contains(CLI_E2E_REASON_FILESYSTEM_GIANT_LOG_SKIPPED));
    assert!(reasons.contains(CLI_E2E_REASON_FILESYSTEM_BINARY_BLOB_SKIPPED));
}

#[test]
fn replay_guidance_uses_filesystem_harness_for_chaos_scenarios() {
    for scenario in default_cli_e2e_filesystem_chaos_scenarios() {
        let replay = replay_command_for_scenario(&scenario);
        assert!(replay.contains("cargo test -p frankensearch-fsfs --test filesystem_chaos"));
        assert!(replay.contains("-- --nocapture --exact scenario_"));
    }
}
