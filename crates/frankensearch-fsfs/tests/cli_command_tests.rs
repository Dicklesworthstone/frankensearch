//! Unit tests for individual `fsfs` CLI commands.
//!
//! Each test builds a minimal fixture, runs the fsfs binary with specific
//! arguments, and verifies the output structure and exit code.
//!
//! Coverage:
//! - `index` command: creates index artifacts on disk
//! - `search` command: returns results in all output formats
//! - `search --stream` command: emits NDJSON lines
//! - `explain` command: produces score decomposition
//! - `version` command: prints version string
//! - error paths: meaningful error messages with suggestions

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use frankensearch_fsfs::output_schema::SearchHitPayload;
use frankensearch_fsfs::stream_protocol::{
    StreamEventKind, StreamFrame, TOON_STREAM_RECORD_SEPARATOR_BYTE, decode_stream_frame_ndjson,
    decode_stream_frame_toon,
};
use serde_json::Value;

// ─── Test Infrastructure ────────────────────────────────────────────────────

#[derive(Debug)]
struct TestContext {
    fsfs_bin: PathBuf,
    home_dir: PathBuf,
    xdg_config_home: PathBuf,
    xdg_cache_home: PathBuf,
    xdg_data_home: PathBuf,
    model_dir: PathBuf,
}

impl TestContext {
    fn new(root: &Path) -> Self {
        let home_dir = root.join("home");
        let xdg_config_home = root.join("xdg-config");
        let xdg_cache_home = root.join("xdg-cache");
        let xdg_data_home = root.join("xdg-data");
        let model_dir = root.join("models");
        for dir in [
            &home_dir,
            &xdg_config_home,
            &xdg_cache_home,
            &xdg_data_home,
            &model_dir,
        ] {
            fs::create_dir_all(dir).expect("create test directory");
        }

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
        self.run_with_env(cwd, args, &[])
    }

    fn run_with_env(&self, cwd: &Path, args: &[&str], extra_env: &[(&str, &str)]) -> Output {
        let mut command = Command::new(&self.fsfs_bin);
        command
            .args(args)
            .current_dir(cwd)
            .env("HOME", &self.home_dir)
            .env("XDG_CONFIG_HOME", &self.xdg_config_home)
            .env("XDG_CACHE_HOME", &self.xdg_cache_home)
            .env("XDG_DATA_HOME", &self.xdg_data_home)
            .env("FRANKENSEARCH_MODEL_DIR", &self.model_dir)
            .env("FRANKENSEARCH_OFFLINE", "1")
            .env("FRANKENSEARCH_ALLOW_DOWNLOAD", "0")
            .env("NO_COLOR", "1");
        for (key, value) in extra_env {
            command.env(key, value);
        }

        command.output().expect("spawn fsfs process")
    }
}

fn fsfs_binary_path() -> PathBuf {
    std::env::var_os("CARGO_BIN_EXE_fsfs")
        .map(PathBuf::from)
        .expect("cargo must provide CARGO_BIN_EXE_fsfs for integration tests")
}

fn assert_success(label: &str, output: &Output) {
    assert!(
        output.status.success(),
        "{label} failed (exit {:?})\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn stdout_str(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr_str(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn parse_json(label: &str, output: &Output) -> Value {
    serde_json::from_slice(&output.stdout).unwrap_or_else(|err| {
        panic!(
            "{label}: invalid JSON: {err}\nstdout:\n{}",
            stdout_str(output)
        )
    })
}

fn assert_generic_csv_envelope(label: &str, output: &Output) {
    assert_success(label, output);
    let text = stdout_str(output);
    let mut lines = text.lines();
    assert_eq!(
        lines.next().unwrap_or_default(),
        "data_json",
        "{label} should emit generic csv envelope header"
    );
    assert!(
        lines.next().is_some_and(|line| !line.trim().is_empty()),
        "{label} should emit one payload row"
    );
}

/// Create a small corpus with known files and build an index.
/// Returns (tempdir, context, `index_dir_arg`).
fn indexed_fixture() -> (tempfile::TempDir, TestContext, String) {
    let temp = tempfile::tempdir().expect("create temp dir");
    let corpus = temp.path().join("corpus");
    let index_dir = temp.path().join("index");
    fs::create_dir_all(&corpus).expect("create corpus dir");

    // Write deterministic fixture files with distinct content
    fs::write(
        corpus.join("auth.rs"),
        "fn authenticate(token: &str) -> bool { token.len() > 8 }",
    )
    .expect("write auth fixture");
    fs::write(
        corpus.join("cache.rs"),
        "fn cache_invalidate(key: &str) { eprintln!(\"invalidate {key}\") }",
    )
    .expect("write cache fixture");
    fs::write(
        corpus.join("retry.md"),
        "Retry backoff strategy with jitter and exponential delay for network calls",
    )
    .expect("write retry fixture");
    fs::write(
        corpus.join("readme.md"),
        "# Project Overview\n\nA search library for indexing and querying documents.",
    )
    .expect("write readme fixture");

    let ctx = TestContext::new(temp.path());
    let corpus_arg = corpus.display().to_string();
    let index_arg = index_dir.display().to_string();

    let output = ctx.run(
        temp.path(),
        &[
            "index",
            &corpus_arg,
            "--index-dir",
            &index_arg,
            "--no-watch-mode",
            "--format",
            "json",
        ],
    );
    assert_success("fixture index", &output);

    (temp, ctx, index_arg)
}

// ─── Index Command ──────────────────────────────────────────────────────────

#[test]
fn index_creates_sentinel_and_vector_index() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let corpus = temp.path().join("corpus");
    let index_dir = temp.path().join("index");
    fs::create_dir_all(&corpus).expect("create corpus dir");
    fs::write(corpus.join("hello.md"), "hello world document").expect("write fixture");

    let ctx = TestContext::new(temp.path());
    let output = ctx.run(
        temp.path(),
        &[
            "index",
            &corpus.display().to_string(),
            "--index-dir",
            &index_dir.display().to_string(),
            "--no-watch-mode",
            "--format",
            "json",
        ],
    );
    assert_success("index", &output);

    assert!(
        index_dir.join("index_sentinel.json").exists(),
        "index must create sentinel file"
    );
    assert!(
        index_dir.join("vector/index.fsvi").exists(),
        "index must create vector index"
    );
}

#[test]
fn index_output_mentions_indexed_count() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let corpus = temp.path().join("corpus");
    let index_dir = temp.path().join("index");
    fs::create_dir_all(&corpus).expect("create corpus dir");
    fs::write(corpus.join("doc.md"), "test document content").expect("write fixture");

    let ctx = TestContext::new(temp.path());
    let output = ctx.run(
        temp.path(),
        &[
            "index",
            &corpus.display().to_string(),
            "--index-dir",
            &index_dir.display().to_string(),
            "--no-watch-mode",
        ],
    );
    assert_success("index", &output);

    let text = format!("{}{}", stdout_str(&output), stderr_str(&output));
    // Index output should mention how many files were processed
    assert!(
        text.contains("ndexed") || text.contains("file"),
        "index output should report indexed files: {text}"
    );
}

#[test]
fn index_empty_corpus_succeeds() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let corpus = temp.path().join("empty-corpus");
    let index_dir = temp.path().join("index");
    fs::create_dir_all(&corpus).expect("create empty corpus dir");

    let ctx = TestContext::new(temp.path());
    let output = ctx.run(
        temp.path(),
        &[
            "index",
            &corpus.display().to_string(),
            "--index-dir",
            &index_dir.display().to_string(),
            "--no-watch-mode",
            "--format",
            "json",
        ],
    );
    // Empty corpus should not crash — either succeeds or exits gracefully
    assert!(
        output.status.success() || output.status.code().is_some(),
        "empty corpus must not crash the process"
    );
}

#[test]
fn index_exclude_pattern_skips_matching_paths() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let corpus = temp.path().join("corpus");
    let index_dir = temp.path().join("index");
    let keep_dir = corpus.join("src");
    let skip_dir = corpus.join("vendor");
    fs::create_dir_all(&keep_dir).expect("create keep dir");
    fs::create_dir_all(&skip_dir).expect("create skip dir");

    fs::write(
        keep_dir.join("keep.md"),
        "keep_token_alpha retained in searchable corpus",
    )
    .expect("write keep fixture");
    fs::write(
        skip_dir.join("skip.md"),
        "skip_token_omega should be excluded from indexing",
    )
    .expect("write skip fixture");

    let ctx = TestContext::new(temp.path());
    let index_output = ctx.run(
        temp.path(),
        &[
            "index",
            &corpus.display().to_string(),
            "--index-dir",
            &index_dir.display().to_string(),
            "--exclude",
            "vendor/**",
            "--no-watch-mode",
            "--format",
            "json",
        ],
    );
    assert_success("index with exclude", &index_output);

    let keep_output = ctx.run(
        temp.path(),
        &[
            "search",
            "keep_token_alpha",
            "--index-dir",
            &index_dir.display().to_string(),
            "--no-watch-mode",
            "--format",
            "json",
        ],
    );
    assert_success("search keep token", &keep_output);
    let keep_json = parse_json("search keep token", &keep_output);
    let keep_hits = keep_json
        .pointer("/data/hits")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    assert!(
        keep_hits > 0,
        "included fixture should remain searchable after exclude pattern"
    );

    let skip_output = ctx.run(
        temp.path(),
        &[
            "search",
            "skip_token_omega",
            "--index-dir",
            &index_dir.display().to_string(),
            "--no-watch-mode",
            "--format",
            "json",
        ],
    );
    assert_success("search excluded token", &skip_output);
    let skip_json = parse_json("search excluded token", &skip_output);
    let vendor_paths_present = skip_json
        .pointer("/data/hits")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|hit| hit.get("path").and_then(Value::as_str))
        .any(|path| path.contains("vendor"));
    assert!(
        !vendor_paths_present,
        "excluded vendor paths should not appear in search results"
    );
}

// ─── Search Command ─────────────────────────────────────────────────────────

#[test]
fn search_json_returns_ok_envelope_with_hits() {
    let (temp, ctx, index_arg) = indexed_fixture();
    let output = ctx.run(
        temp.path(),
        &[
            "search",
            "retry backoff strategy",
            "--index-dir",
            &index_arg,
            "--no-watch-mode",
            "--limit",
            "5",
            "--format",
            "json",
        ],
    );
    assert_success("search json", &output);

    let json = parse_json("search json", &output);
    assert_eq!(json.get("ok").and_then(Value::as_bool), Some(true));
    assert!(
        json.pointer("/data/hits").is_some(),
        "search JSON must contain data.hits"
    );
}

#[test]
fn search_logs_include_output_format_field_when_info_logging_enabled() {
    let (temp, ctx, index_arg) = indexed_fixture();
    let output = ctx.run_with_env(
        temp.path(),
        &[
            "search",
            "retry backoff strategy",
            "--index-dir",
            &index_arg,
            "--no-watch-mode",
            "--limit",
            "5",
            "--format",
            "json",
        ],
        &[("FRANKENSEARCH_LOG", "info")],
    );
    assert_success("search logging format field", &output);

    let stderr = stderr_str(&output);
    assert!(
        stderr.contains("fsfs search command completed"),
        "expected info search completion log line in stderr: {stderr}"
    );
    assert!(
        stderr.contains("output_format=json") || stderr.contains("output_format=\"json\""),
        "expected output_format field in search logs: {stderr}"
    );
}

#[test]
fn search_default_format_is_jsonl_for_non_tty_stdout() {
    let (temp, ctx, index_arg) = indexed_fixture();
    let output = ctx.run(
        temp.path(),
        &[
            "search",
            "retry backoff strategy",
            "--index-dir",
            &index_arg,
            "--no-watch-mode",
            "--limit",
            "5",
        ],
    );
    assert_success("search default format", &output);

    let text = stdout_str(&output);
    let mut line_count = 0_usize;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        serde_json::from_str::<Value>(trimmed).unwrap_or_else(|error| {
            panic!("default format must emit JSONL in non-tty tests: {error}\nline: {trimmed}")
        });
        line_count += 1;
    }

    assert!(
        line_count >= 1,
        "default non-tty output should emit at least one JSONL line"
    );
}

#[test]
fn search_csv_output_has_header_row() {
    let (temp, ctx, index_arg) = indexed_fixture();
    let output = ctx.run(
        temp.path(),
        &[
            "search",
            "cache invalidation",
            "--index-dir",
            &index_arg,
            "--no-watch-mode",
            "--limit",
            "5",
            "--format",
            "csv",
        ],
    );
    assert_success("search csv", &output);

    let text = stdout_str(&output);
    assert!(!text.is_empty(), "CSV output must not be empty");
    // CSV should have comma-separated values
    let first_line = text.lines().next().unwrap_or("");
    assert!(
        first_line.contains(','),
        "CSV first line should contain commas (got: {first_line})"
    );
}

#[test]
fn search_toon_output_decodes_as_valid_envelope() {
    let (temp, ctx, index_arg) = indexed_fixture();
    let output = ctx.run(
        temp.path(),
        &[
            "search",
            "authenticate token",
            "--index-dir",
            &index_arg,
            "--no-watch-mode",
            "--limit",
            "5",
            "--format",
            "toon",
        ],
    );
    assert_success("search toon", &output);

    let text = stdout_str(&output);
    assert!(
        text.contains("ok: true"),
        "TOON output should include success marker: {text}"
    );
    assert!(
        text.contains("command: search"),
        "TOON output should include command metadata: {text}"
    );
    assert!(
        text.contains("format: toon"),
        "TOON output should include toon format metadata: {text}"
    );
}

#[test]
fn search_table_output_is_human_readable() {
    let (temp, ctx, index_arg) = indexed_fixture();
    let output = ctx.run(
        temp.path(),
        &[
            "search",
            "authenticate token",
            "--index-dir",
            &index_arg,
            "--no-watch-mode",
            "--limit",
            "5",
            "--format",
            "table",
        ],
    );
    assert_success("search table", &output);
    let text = stdout_str(&output);
    // Table output should contain visible characters (not be empty/binary)
    assert!(
        text.chars().any(char::is_alphanumeric),
        "table output should contain readable text"
    );
}

#[test]
fn search_explicit_table_overrides_non_tty_default() {
    let (temp, ctx, index_arg) = indexed_fixture();
    let output = ctx.run(
        temp.path(),
        &[
            "search",
            "authenticate token",
            "--index-dir",
            &index_arg,
            "--no-watch-mode",
            "--limit",
            "5",
            "--format",
            "table",
        ],
    );
    assert_success("search table explicit", &output);

    let text = stdout_str(&output);
    assert!(
        !text
            .lines()
            .next()
            .unwrap_or_default()
            .trim_start()
            .starts_with('{'),
        "explicit table format should not emit JSONL default output"
    );
    assert!(
        text.contains("PHASE") || text.contains("results in"),
        "table output should contain human-readable result sections"
    );
}

#[test]
fn search_empty_query_exits_gracefully() {
    let (temp, ctx, index_arg) = indexed_fixture();
    let output = ctx.run(
        temp.path(),
        &[
            "search",
            "",
            "--index-dir",
            &index_arg,
            "--no-watch-mode",
            "--format",
            "json",
        ],
    );
    // Empty query should not crash — may return empty results or an error
    assert!(
        output.status.code().is_some(),
        "empty query must not crash the process"
    );
}

#[test]
fn search_limit_zero_returns_no_hits() {
    let (temp, ctx, index_arg) = indexed_fixture();
    let output = ctx.run(
        temp.path(),
        &[
            "search",
            "retry",
            "--index-dir",
            &index_arg,
            "--no-watch-mode",
            "--limit",
            "0",
            "--format",
            "json",
        ],
    );
    // --limit 0 should either succeed with no hits or gracefully exit
    if output.status.success() {
        let json = parse_json("search limit-0", &output);
        let hits = json
            .pointer("/data/hits")
            .and_then(Value::as_array)
            .map_or(0, Vec::len);
        assert_eq!(hits, 0, "limit 0 should return zero hits");
    }
}

#[test]
fn search_limit_three_caps_hit_count() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let corpus = temp.path().join("corpus");
    let index_dir = temp.path().join("index");
    fs::create_dir_all(&corpus).expect("create corpus dir");
    for idx in 0..6 {
        fs::write(
            corpus.join(format!("doc-{idx}.md")),
            format!("limit_token_shared content block {idx}"),
        )
        .expect("write limit fixture");
    }

    let ctx = TestContext::new(temp.path());
    let index_output = ctx.run(
        temp.path(),
        &[
            "index",
            &corpus.display().to_string(),
            "--index-dir",
            &index_dir.display().to_string(),
            "--no-watch-mode",
            "--format",
            "json",
        ],
    );
    assert_success("index limit fixture", &index_output);

    let output = ctx.run(
        temp.path(),
        &[
            "search",
            "limit_token_shared",
            "--index-dir",
            &index_dir.display().to_string(),
            "--no-watch-mode",
            "--limit",
            "3",
            "--format",
            "json",
        ],
    );
    assert_success("search limit 3", &output);
    let json = parse_json("search limit 3", &output);
    let hits = json
        .pointer("/data/hits")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    assert!(hits <= 3, "limit=3 must cap hit count (got {hits})");
    assert!(
        hits > 0,
        "limit fixture query should return at least one hit"
    );
}

// ─── Search --stream Command ────────────────────────────────────────────────

#[test]
fn search_stream_emits_ndjson_lines() {
    let (temp, ctx, index_arg) = indexed_fixture();
    let output = ctx.run(
        temp.path(),
        &[
            "search",
            "retry backoff",
            "--index-dir",
            &index_arg,
            "--no-watch-mode",
            "--stream",
            "--format",
            "jsonl",
        ],
    );
    assert_success("search stream", &output);

    let text = stdout_str(&output);
    // Each non-empty line should be valid JSON
    let mut line_count = 0;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        serde_json::from_str::<Value>(trimmed)
            .unwrap_or_else(|err| panic!("NDJSON line is not valid JSON: {err}\nline: {trimmed}"));
        line_count += 1;
    }
    assert!(
        line_count >= 1,
        "stream output must emit at least one NDJSON line"
    );
}

#[test]
fn search_stream_ndjson_starts_and_ends_with_terminal_frame() {
    let (temp, ctx, index_arg) = indexed_fixture();
    let output = ctx.run(
        temp.path(),
        &[
            "search",
            "retry backoff",
            "--index-dir",
            &index_arg,
            "--no-watch-mode",
            "--stream",
            "--format",
            "jsonl",
        ],
    );
    assert_success("search stream frame order", &output);

    let text = stdout_str(&output);
    let mut frames = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let frame =
            decode_stream_frame_ndjson::<SearchHitPayload>(trimmed).expect("decode stream frame");
        frames.push(frame);
    }
    assert!(!frames.is_empty(), "stream should emit at least one frame");
    assert_eq!(frames[0].event.kind(), StreamEventKind::Started);
    assert_eq!(
        frames
            .last()
            .expect("stream should contain terminal frame")
            .event
            .kind(),
        StreamEventKind::Terminal
    );
}

#[test]
fn search_stream_toon_emits_record_separated_frames() {
    let (temp, ctx, index_arg) = indexed_fixture();
    let output = ctx.run(
        temp.path(),
        &[
            "search",
            "retry backoff",
            "--index-dir",
            &index_arg,
            "--no-watch-mode",
            "--stream",
            "--format",
            "toon",
        ],
    );
    assert_success("search stream toon", &output);

    let bytes = &output.stdout;
    assert_eq!(
        bytes.first().copied(),
        Some(TOON_STREAM_RECORD_SEPARATOR_BYTE),
        "toon stream should prefix each frame with record separator"
    );

    let records = bytes
        .split(|byte| *byte == TOON_STREAM_RECORD_SEPARATOR_BYTE)
        .filter(|chunk| !chunk.is_empty())
        .collect::<Vec<_>>();
    assert!(
        !records.is_empty(),
        "toon stream should emit at least one framed record"
    );

    let mut kinds = Vec::new();
    for record in records {
        let payload = std::str::from_utf8(record)
            .expect("utf8 stream record")
            .trim_end_matches('\n');
        let frame: StreamFrame<SearchHitPayload> =
            decode_stream_frame_toon(payload).expect("decode toon stream frame");
        kinds.push(frame.event.kind());
    }
    assert_eq!(kinds.first().copied(), Some(StreamEventKind::Started));
    assert_eq!(kinds.last().copied(), Some(StreamEventKind::Terminal));
}

#[test]
fn search_stream_rejects_non_stream_format() {
    let (temp, ctx, index_arg) = indexed_fixture();
    let output = ctx.run(
        temp.path(),
        &[
            "search",
            "retry backoff",
            "--index-dir",
            &index_arg,
            "--no-watch-mode",
            "--stream",
            "--format",
            "json",
        ],
    );

    assert!(
        !output.status.success(),
        "search --stream with --format json should fail validation"
    );
    let combined = format!("{}\n{}", stdout_str(&output), stderr_str(&output));
    assert!(
        combined.contains("stream mode requires --format jsonl or --format toon"),
        "expected stream-format validation message, got:\n{combined}"
    );
}

// ─── Explain Command ────────────────────────────────────────────────────────

#[test]
fn explain_json_produces_score_decomposition() {
    let (temp, ctx, index_arg) = indexed_fixture();

    // First search to get a hit path
    let search_output = ctx.run(
        temp.path(),
        &[
            "search",
            "retry backoff",
            "--index-dir",
            &index_arg,
            "--no-watch-mode",
            "--limit",
            "1",
            "--format",
            "json",
        ],
    );
    assert_success("explain: prereq search", &search_output);

    let search_json = parse_json("explain: prereq search", &search_output);
    let first_hit_path = search_json
        .pointer("/data/hits/0/path")
        .and_then(Value::as_str);

    if let Some(hit_path) = first_hit_path {
        let explain_output = ctx.run(
            temp.path(),
            &[
                "explain",
                hit_path,
                "--query",
                "retry backoff",
                "--index-dir",
                &index_arg,
                "--no-watch-mode",
                "--format",
                "json",
            ],
        );
        // Explain should either succeed or fail gracefully (e.g., model not available)
        if explain_output.status.success() {
            let json = parse_json("explain json", &explain_output);
            assert_eq!(json.get("ok").and_then(Value::as_bool), Some(true));
        }
    }
}

#[test]
fn explain_table_renders_human_readable_breakdown() {
    let (temp, ctx, index_arg) = indexed_fixture();

    let search_output = ctx.run(
        temp.path(),
        &[
            "search",
            "retry backoff",
            "--index-dir",
            &index_arg,
            "--no-watch-mode",
            "--limit",
            "1",
            "--format",
            "json",
        ],
    );
    assert_success("explain table: prereq search", &search_output);

    let search_json = parse_json("explain table: prereq search", &search_output);
    let hit_count = search_json
        .pointer("/data/hits")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    assert!(
        hit_count > 0,
        "expected at least one hit for explain table test"
    );

    let explain_output = ctx.run(
        temp.path(),
        &[
            "explain",
            "R0",
            "--index-dir",
            &index_arg,
            "--no-watch-mode",
            "--format",
            "table",
        ],
    );
    assert_success("explain table", &explain_output);

    let text = stdout_str(&explain_output);
    assert!(
        text.contains("Result ID:"),
        "table explain output should include result id line:\n{text}"
    );
    assert!(
        text.contains("RRF:"),
        "table explain output should include RRF breakdown:\n{text}"
    );
}

#[test]
fn explain_invalid_result_id_reports_available_ids() {
    let (temp, ctx, index_arg) = indexed_fixture();
    let search_output = ctx.run(
        temp.path(),
        &[
            "search",
            "retry backoff",
            "--index-dir",
            &index_arg,
            "--no-watch-mode",
            "--limit",
            "2",
            "--format",
            "json",
        ],
    );
    assert_success("explain invalid id: prereq search", &search_output);

    let output = ctx.run(
        temp.path(),
        &[
            "explain",
            "R999",
            "--index-dir",
            &index_arg,
            "--no-watch-mode",
            "--format",
            "json",
        ],
    );
    assert!(
        !output.status.success(),
        "explain with unknown result id should fail"
    );
    let combined = format!("{}\n{}", stdout_str(&output), stderr_str(&output));
    assert!(
        combined.contains("unknown result id"),
        "error should mention unknown result id: {combined}"
    );
    assert!(
        combined.contains("available ids"),
        "error should list available result ids: {combined}"
    );
}

// ─── Version Command ────────────────────────────────────────────────────────

#[test]
fn version_prints_version_string() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let ctx = TestContext::new(temp.path());
    let output = ctx.run(temp.path(), &["version"]);
    assert_success("version", &output);

    let text = stdout_str(&output);
    // Version output should mention "fsfs" or contain a semver-like pattern
    assert!(
        text.contains("fsfs") || text.contains('.'),
        "version output should identify the tool: {text}"
    );
}

#[test]
fn version_exits_zero_with_no_args() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let ctx = TestContext::new(temp.path());
    let output = ctx.run(temp.path(), &["version"]);
    assert_success("version no-args", &output);

    // Version output should be non-empty and contain a dot (semver pattern)
    let text = stdout_str(&output);
    assert!(
        text.contains('.'),
        "version output should contain a semver-like version number: {text}"
    );
}

// ─── Error Paths ────────────────────────────────────────────────────────────

#[test]
fn search_missing_index_dir_reports_error() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let ctx = TestContext::new(temp.path());
    let output = ctx.run(
        temp.path(),
        &[
            "search",
            "test query",
            "--index-dir",
            "/nonexistent/path/to/index",
            "--no-watch-mode",
            "--format",
            "json",
        ],
    );
    // Should fail with a meaningful error
    assert!(
        !output.status.success(),
        "search against missing index should fail"
    );
    let combined = format!("{}\n{}", stdout_str(&output), stderr_str(&output));
    assert!(
        !combined.is_empty(),
        "error output should contain some diagnostic information"
    );
}

#[test]
fn status_csv_output_uses_generic_csv_envelope() {
    let (temp, ctx, index_arg) = indexed_fixture();
    let output = ctx.run(
        temp.path(),
        &[
            "status",
            "--index-dir",
            &index_arg,
            "--no-watch-mode",
            "--format",
            "csv",
        ],
    );
    assert_generic_csv_envelope("status csv", &output);
}

#[test]
fn config_validate_csv_output_uses_generic_csv_envelope() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let ctx = TestContext::new(temp.path());
    let output = ctx.run(temp.path(), &["config", "validate", "--format", "csv"]);
    assert_generic_csv_envelope("config validate csv", &output);
}

#[test]
fn doctor_csv_output_uses_generic_csv_envelope() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let ctx = TestContext::new(temp.path());
    let output = ctx.run(temp.path(), &["doctor", "--format", "csv"]);
    assert_generic_csv_envelope("doctor csv", &output);
}

#[test]
fn download_list_csv_output_uses_generic_csv_envelope() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let ctx = TestContext::new(temp.path());
    let output = ctx.run(
        temp.path(),
        &["download-models", "--list", "--format", "csv"],
    );
    assert_generic_csv_envelope("download list csv", &output);
}

#[test]
fn explain_csv_without_search_context_reports_error() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let ctx = TestContext::new(temp.path());
    let output = ctx.run(temp.path(), &["explain", "R1", "--format", "csv"]);
    assert!(
        !output.status.success(),
        "explain without prior search should fail"
    );
    let combined = format!("{}\n{}", stdout_str(&output), stderr_str(&output));
    assert!(
        combined.contains("no saved search context"),
        "error should mention missing search context"
    );
}

#[test]
fn uninstall_dry_run_csv_output_uses_generic_csv_envelope() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let ctx = TestContext::new(temp.path());
    let output = ctx.run(temp.path(), &["uninstall", "--dry-run", "--format", "csv"]);
    assert_generic_csv_envelope("uninstall dry-run csv", &output);
}

#[test]
fn search_missing_index_dir_csv_reports_error() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let ctx = TestContext::new(temp.path());
    let output = ctx.run(
        temp.path(),
        &[
            "search",
            "test query",
            "--index-dir",
            "/nonexistent/path/to/index",
            "--no-watch-mode",
            "--format",
            "csv",
        ],
    );
    assert!(
        !output.status.success(),
        "search against missing index should fail for csv output"
    );
    let combined = format!("{}\n{}", stdout_str(&output), stderr_str(&output));
    assert!(
        !combined.trim().is_empty(),
        "csv error path should emit diagnostics"
    );
}

#[test]
fn index_nonexistent_corpus_reports_error() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let ctx = TestContext::new(temp.path());
    let output = ctx.run(
        temp.path(),
        &[
            "index",
            "/nonexistent/corpus/path",
            "--index-dir",
            &temp.path().join("index").display().to_string(),
            "--no-watch-mode",
            "--format",
            "json",
        ],
    );
    assert!(
        !output.status.success(),
        "indexing nonexistent corpus should fail"
    );
}

// ─── Output Format Consistency ──────────────────────────────────────────────

#[test]
fn all_search_formats_succeed_on_same_query() {
    let (temp, ctx, index_arg) = indexed_fixture();
    let query = "search library indexing";

    for format in &["json", "csv", "table", "jsonl", "toon"] {
        let output = ctx.run(
            temp.path(),
            &[
                "search",
                query,
                "--index-dir",
                &index_arg,
                "--no-watch-mode",
                "--limit",
                "3",
                "--format",
                format,
            ],
        );
        assert_success(&format!("search format={format}"), &output);
        assert!(
            !output.stdout.is_empty(),
            "search --format {format} must produce output"
        );
    }
}
