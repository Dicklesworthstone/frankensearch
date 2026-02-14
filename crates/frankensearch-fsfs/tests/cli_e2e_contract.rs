use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

use frankensearch_core::{E2eOutcome, ExitStatus};
use frankensearch_fsfs::{
    CLI_E2E_SCHEMA_VERSION, CliE2eArtifactBundle, CliE2eRunConfig, CliE2eScenarioKind,
    build_default_cli_e2e_bundles, default_cli_e2e_scenarios, replay_command_for_scenario,
};
use serde::Deserialize;
use serde_json::Value;

fn scenario_by_kind(kind: CliE2eScenarioKind) -> frankensearch_fsfs::CliE2eScenario {
    default_cli_e2e_scenarios()
        .into_iter()
        .find(|scenario| scenario.kind == kind)
        .expect("scenario should exist")
}

#[test]
fn scenario_cli_index_baseline() {
    let scenario = scenario_by_kind(CliE2eScenarioKind::Index);
    let bundle =
        CliE2eArtifactBundle::build(&CliE2eRunConfig::default(), &scenario, ExitStatus::Pass);
    bundle.validate().expect("bundle must validate");
    assert_eq!(bundle.schema_version, CLI_E2E_SCHEMA_VERSION);
    assert_eq!(bundle.scenario.kind, CliE2eScenarioKind::Index);
    assert_eq!(
        bundle.scenario.args.first().map(String::as_str),
        Some("index")
    );
}

#[test]
fn scenario_cli_search_stream() {
    let scenario = scenario_by_kind(CliE2eScenarioKind::Search);
    let bundle =
        CliE2eArtifactBundle::build(&CliE2eRunConfig::default(), &scenario, ExitStatus::Pass);
    bundle.validate().expect("bundle must validate");
    assert_eq!(bundle.scenario.kind, CliE2eScenarioKind::Search);
    assert!(bundle.scenario.args.contains(&"--stream".to_owned()));
    assert!(bundle.events.iter().any(|event| {
        event
            .body
            .reason_code
            .as_deref()
            .is_some_and(|code| code.starts_with("e2e.cli."))
    }));
}

#[test]
fn scenario_cli_explain_hit() {
    let scenario = scenario_by_kind(CliE2eScenarioKind::Explain);
    let bundle =
        CliE2eArtifactBundle::build(&CliE2eRunConfig::default(), &scenario, ExitStatus::Pass);
    bundle.validate().expect("bundle must validate");
    assert_eq!(bundle.scenario.kind, CliE2eScenarioKind::Explain);
    assert!(bundle.scenario.args.contains(&"toon".to_owned()));
}

#[test]
fn scenario_cli_degrade_path() {
    let scenario = scenario_by_kind(CliE2eScenarioKind::Degrade);
    let bundle =
        CliE2eArtifactBundle::build(&CliE2eRunConfig::default(), &scenario, ExitStatus::Fail);
    bundle.validate().expect("bundle must validate");
    assert_eq!(bundle.scenario.kind, CliE2eScenarioKind::Degrade);
    assert!(
        bundle
            .events
            .iter()
            .any(|event| event.body.outcome == Some(E2eOutcome::Fail))
    );
    assert!(
        bundle
            .manifest
            .body
            .artifacts
            .iter()
            .any(|artifact| artifact.file == "replay_command.txt")
    );
    assert!(
        bundle
            .replay_command
            .contains("--exact scenario_cli_degrade_path")
    );
}

#[test]
fn default_bundle_set_covers_all_cli_flows() {
    let bundles = build_default_cli_e2e_bundles(&CliE2eRunConfig::default());
    assert_eq!(bundles.len(), 4);
    for bundle in bundles {
        bundle.validate().expect("bundle must validate");
        assert_eq!(bundle.schema_version, CLI_E2E_SCHEMA_VERSION);
    }
}

#[test]
fn replay_guidance_points_at_exact_scenario_test() {
    let scenario = scenario_by_kind(CliE2eScenarioKind::Search);
    let replay = replay_command_for_scenario(&scenario);
    assert!(replay.contains("cargo test -p frankensearch-fsfs --test cli_e2e_contract"));
    assert!(replay.contains("--exact scenario_cli_search_stream"));
}

const E2E_CORPUS_FIXTURE_PATH: &str = "tests/fixtures/e2e_corpus/corpus_manifest.json";
const QUERY_TIMEOUT_BUDGET: Duration = Duration::from_secs(2);

#[derive(Debug, Deserialize)]
struct E2eCorpusFixture {
    files: Vec<E2eCorpusFile>,
    queries: Vec<E2eGroundTruthQuery>,
}

#[derive(Debug, Deserialize)]
struct E2eCorpusFile {
    path: String,
    contents: String,
}

#[derive(Debug, Deserialize)]
struct E2eGroundTruthQuery {
    query: String,
    expected_docs: Vec<String>,
    min_recall_at_5: f64,
}

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

    fn run(&self, cwd: &Path, args: &[&str]) -> Output {
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

fn fsfs_binary_path() -> PathBuf {
    std::env::var_os("CARGO_BIN_EXE_fsfs")
        .map(PathBuf::from)
        .expect("cargo must provide CARGO_BIN_EXE_fsfs for integration tests")
}

fn fixture_manifest_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(E2E_CORPUS_FIXTURE_PATH)
}

fn load_e2e_corpus_fixture() -> E2eCorpusFixture {
    let path = fixture_manifest_path();
    let raw = fs::read_to_string(&path).expect("read e2e corpus fixture");
    serde_json::from_str(&raw).expect("parse e2e corpus fixture json")
}

fn materialize_fixture_corpus(fixture: &E2eCorpusFixture, root: &Path) {
    for file in &fixture.files {
        let target_path = root.join(&file.path);
        if let Some(parent) = target_path.parent() {
            fs::create_dir_all(parent).expect("create fixture parent dir");
        }
        fs::write(&target_path, &file.contents).expect("write fixture file");
    }
}

fn assert_command_success(label: &str, output: &Output) {
    assert!(
        output.status.success(),
        "{label} failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn parse_json_stdout(label: &str, output: &Output) -> Value {
    serde_json::from_slice::<Value>(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "{label} did not produce parseable JSON: {error}\nstdout:\n{}",
            String::from_utf8_lossy(&output.stdout)
        )
    })
}

fn command_exists(program: &str) -> bool {
    Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {program} >/dev/null 2>&1"))
        .status()
        .is_ok_and(|status| status.success())
}

fn pipe_stdout_into(program: &str, args: &[&str], stdout: &[u8]) -> Output {
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|error| panic!("spawn {program}: {error}"));

    if let Some(stdin) = child.stdin.as_mut() {
        stdin
            .write_all(stdout)
            .unwrap_or_else(|error| panic!("write stdin for {program}: {error}"));
    } else {
        panic!("missing stdin pipe for {program}");
    }

    child
        .wait_with_output()
        .unwrap_or_else(|error| panic!("wait for {program}: {error}"))
}

fn extract_hit_paths(search_envelope: &Value) -> Vec<String> {
    search_envelope
        .pointer("/data/hits")
        .and_then(Value::as_array)
        .map(|hits| {
            hits.iter()
                .filter_map(|hit| hit.get("path").and_then(Value::as_str))
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn recall_at_5(hit_paths: &[String], expected_docs: &[String]) -> f64 {
    if expected_docs.is_empty() {
        return 1.0;
    }

    let matched = expected_docs
        .iter()
        .filter(|expected| {
            hit_paths
                .iter()
                .any(|path| path.ends_with(expected.as_str()))
        })
        .count();
    let matched_f64 = f64::from(u32::try_from(matched).unwrap_or(u32::MAX));
    let expected_f64 = f64::from(u32::try_from(expected_docs.len()).unwrap_or(u32::MAX));
    matched_f64 / expected_f64
}

fn create_small_index_fixture() -> (tempfile::TempDir, E2eCommandContext, String) {
    let temp = tempfile::tempdir().expect("create temporary test workspace");
    let corpus_root = temp.path().join("tiny-corpus");
    let index_root = temp.path().join("tiny-index");
    fs::create_dir_all(&corpus_root).expect("create tiny corpus root");
    fs::write(
        corpus_root.join("retry.md"),
        "retry backoff strategy with jitter and exponential delay",
    )
    .expect("write retry fixture");
    fs::write(
        corpus_root.join("cache.md"),
        "cache invalidation guide and cache key stability checks",
    )
    .expect("write cache fixture");

    let command_context = E2eCommandContext::new(temp.path());
    let corpus_root_arg = corpus_root.display().to_string();
    let index_root_arg = index_root.display().to_string();
    let index_output = command_context.run(
        temp.path(),
        &[
            "index",
            &corpus_root_arg,
            "--index-dir",
            &index_root_arg,
            "--no-watch-mode",
            "--format",
            "json",
        ],
    );
    assert_command_success("tiny fixture index", &index_output);

    (temp, command_context, index_root_arg)
}

#[test]
fn search_json_output_can_be_consumed_by_jq() {
    if !command_exists("jq") {
        eprintln!("jq not available; skipping jq integration assertion");
        return;
    }

    let (temp, command_context, index_root_arg) = create_small_index_fixture();
    let search_output = command_context.run(
        temp.path(),
        &[
            "search",
            "retry backoff strategy",
            "--index-dir",
            &index_root_arg,
            "--no-watch-mode",
            "--limit",
            "5",
            "--format",
            "json",
        ],
    );
    assert_command_success("search json", &search_output);

    let jq_output = pipe_stdout_into("jq", &["-e", ".ok == true"], &search_output.stdout);
    assert_command_success("jq parse", &jq_output);
}

#[test]
fn search_csv_output_can_be_consumed_by_csv_parser() {
    let (temp, command_context, index_root_arg) = create_small_index_fixture();
    let search_output = command_context.run(
        temp.path(),
        &[
            "search",
            "cache invalidation",
            "--index-dir",
            &index_root_arg,
            "--no-watch-mode",
            "--limit",
            "5",
            "--format",
            "csv",
        ],
    );
    assert_command_success("search csv", &search_output);
    assert!(
        !search_output.stdout.is_empty(),
        "search csv output must not be empty"
    );

    if command_exists("csvtool") {
        let csvtool_output = pipe_stdout_into("csvtool", &["col", "1", "-"], &search_output.stdout);
        assert_command_success("csvtool parse", &csvtool_output);
        return;
    }

    let python_output = pipe_stdout_into(
        "python3",
        &[
            "-c",
            "import csv,sys; rows=list(csv.reader(sys.stdin)); assert rows and rows[0], 'empty csv'",
        ],
        &search_output.stdout,
    );
    assert_command_success("python csv parse fallback", &python_output);
}

#[test]
#[ignore = "Runs the fsfs binary end-to-end; enable in scheduled CI with --ignored"]
#[allow(clippy::too_many_lines)]
fn scenario_cli_index_search_recall_e2e() {
    let fixture = load_e2e_corpus_fixture();
    assert_eq!(
        fixture.files.len(),
        50,
        "fixture must include 50 corpus files"
    );
    assert_eq!(
        fixture.queries.len(),
        10,
        "fixture must include 10 query probes"
    );

    let temp = tempfile::tempdir().expect("create temp dir");
    let corpus_root = temp.path().join("corpus");
    let index_root = temp.path().join("index");
    fs::create_dir_all(&corpus_root).expect("create corpus root");
    materialize_fixture_corpus(&fixture, &corpus_root);

    let command_context = E2eCommandContext::new(temp.path());
    let corpus_root_arg = corpus_root.display().to_string();
    let index_root_arg = index_root.display().to_string();

    let index_output = command_context.run(
        temp.path(),
        &[
            "index",
            &corpus_root_arg,
            "--index-dir",
            &index_root_arg,
            "--no-watch-mode",
            "--format",
            "json",
        ],
    );
    assert_command_success("index", &index_output);
    assert!(
        index_root.join("index_sentinel.json").exists(),
        "index command must emit sentinel file"
    );
    assert!(
        index_root.join("vector/index.fsvi").exists(),
        "index command must emit vector index file"
    );
    assert!(
        index_root.join("lexical/index_manifest.json").exists(),
        "index command must emit lexical manifest"
    );

    let mut explain_target: Option<String> = None;
    for query_case in &fixture.queries {
        let started = Instant::now();
        let search_output = command_context.run(
            temp.path(),
            &[
                "search",
                &query_case.query,
                "--index-dir",
                &index_root_arg,
                "--no-watch-mode",
                "--limit",
                "5",
                "--format",
                "json",
            ],
        );
        let elapsed = started.elapsed();

        assert_command_success("search", &search_output);
        assert!(
            elapsed <= QUERY_TIMEOUT_BUDGET,
            "query exceeded latency budget: {:?} > {:?} for query '{}'",
            elapsed,
            QUERY_TIMEOUT_BUDGET,
            query_case.query
        );

        let envelope = parse_json_stdout("search", &search_output);
        assert_eq!(
            envelope.get("ok").and_then(Value::as_bool),
            Some(true),
            "search envelope must indicate success"
        );
        let hit_paths = extract_hit_paths(&envelope);
        assert!(
            !hit_paths.is_empty(),
            "search must return at least one hit for query '{}'",
            query_case.query
        );
        let recall = recall_at_5(&hit_paths, &query_case.expected_docs);
        assert!(
            recall >= query_case.min_recall_at_5,
            "recall@5 {} below threshold {} for query '{}'; hits={hit_paths:?}",
            recall,
            query_case.min_recall_at_5,
            query_case.query
        );

        if explain_target.is_none() {
            explain_target = hit_paths.first().cloned();
        }
    }

    let stream_probe = fixture.queries.first().expect("query probes");
    let stream_output = command_context.run(
        temp.path(),
        &[
            "search",
            &stream_probe.query,
            "--index-dir",
            &index_root_arg,
            "--no-watch-mode",
            "--limit",
            "5",
            "--stream",
            "--format",
            "jsonl",
        ],
    );
    assert_command_success("search stream", &stream_output);
    let stream_text = String::from_utf8(stream_output.stdout).expect("stream utf8 output");
    let stream_lines: Vec<&str> = stream_text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    assert!(
        !stream_lines.is_empty(),
        "stream mode must emit at least one NDJSON frame"
    );
    for (frame_idx, line) in stream_lines.iter().enumerate() {
        let frame: Value = serde_json::from_str(line).unwrap_or_else(|error| {
            panic!("failed to parse NDJSON frame {frame_idx}: {error}; line={line}");
        });
        assert!(
            frame.get("ok").is_some(),
            "stream frame {frame_idx} missing ok field"
        );
        assert_eq!(
            frame.pointer("/meta/format").and_then(Value::as_str),
            Some("jsonl"),
            "stream frame {frame_idx} must identify jsonl format"
        );
    }

    let explain_id = explain_target.expect("search should produce explain target");
    let explain_output = command_context.run(
        temp.path(),
        &[
            "explain",
            &explain_id,
            "--index-dir",
            &index_root_arg,
            "--no-watch-mode",
            "--format",
            "json",
        ],
    );
    assert_command_success("explain", &explain_output);
    assert!(
        !explain_output.stdout.is_empty(),
        "explain command returned empty output; explain payload wiring must be completed"
    );
    let explain_envelope = parse_json_stdout("explain", &explain_output);
    assert!(
        explain_envelope.get("ok").is_some(),
        "explain output must be a structured output envelope"
    );
}
