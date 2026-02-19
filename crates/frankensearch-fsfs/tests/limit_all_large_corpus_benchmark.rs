//! Dedicated `--limit all` large-corpus benchmark for fsfs search.
//!
//! Coverage:
//! - hybrid retrieval path (lexical + semantic),
//! - vector-only retrieval path (semantic-only execution by provisioning only
//!   the vector index artifact),
//! - uncapped result cardinality checks under large corpus load.
//!
//! Run manually:
//! `cargo test -p frankensearch-fsfs --test limit_all_large_corpus_benchmark -- --ignored --nocapture`

use std::fs;
use std::io::{BufRead as _, BufReader, Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

const REPORT_SCHEMA_VERSION: &str = "fsfs-limit-all-benchmark-v2";
const DOC_COUNT: usize = 3_000;
const WARMUP_RUNS: usize = 2;
const MEASURED_RUNS: usize = 7;
const INDEX_TIMEOUT: Duration = Duration::from_secs(75);
const SEARCH_TIMEOUT: Duration = Duration::from_secs(30);
const QUERY: &str = "limitallbenchkeyword retrieval throughput";
const TIMEOUT_EXIT_CODE: i32 = 124;

#[derive(Debug, Deserialize)]
struct SearchEnvelope {
    ok: bool,
    data: Option<SearchPayload>,
    error: Option<serde_json::Value>,
    meta: EnvelopeMeta,
}

#[derive(Debug, Deserialize)]
struct ServeReadyEnvelope {
    ok: bool,
    event: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ServeResponseEnvelope {
    ok: bool,
    payloads: Vec<SearchPayload>,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct EnvelopeMeta {
    duration_ms: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct SearchPayload {
    returned_hits: usize,
    total_candidates: usize,
}

#[derive(Debug)]
struct CommandResult {
    stdout: String,
    stderr: String,
    exit_code: i32,
    elapsed_ms: u64,
    timed_out: bool,
}

#[derive(Debug, Clone)]
struct ModeRunResult {
    elapsed_ms: u64,
    returned_hits: usize,
    total_candidates: usize,
}

#[derive(Debug, Serialize)]
struct ModeSummary {
    mode: String,
    samples: usize,
    latency_ms_min: u64,
    latency_ms_p50: u64,
    latency_ms_p95: u64,
    latency_ms_max: u64,
    latency_ms_mean: f64,
    returned_hits_min: usize,
    returned_hits_max: usize,
    total_candidates_min: usize,
    total_candidates_max: usize,
}

#[derive(Debug, Serialize)]
struct ModeBenchmarkSuite {
    cold: ModeSummary,
    warm: ModeSummary,
}

#[derive(Debug, Serialize)]
struct LimitAllBenchmarkReport {
    schema_version: String,
    corpus_docs: usize,
    query: String,
    warmup_runs: usize,
    measured_runs: usize,
    hybrid: ModeBenchmarkSuite,
    vector_only: ModeBenchmarkSuite,
}

fn fsfs_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_fsfs"))
}

fn write_large_corpus(corpus_root: &Path, doc_count: usize) {
    const TOPIC_BUCKETS: usize = 24;
    for idx in 0..doc_count {
        let topic = idx % TOPIC_BUCKETS;
        let path = corpus_root.join(format!("topic_{topic:02}/doc_{idx:05}.md"));
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap_or_else(|error| {
                panic!(
                    "failed to create corpus parent {}: {error}",
                    parent.display()
                )
            });
        }

        let content = format!(
            "# Topic {topic}\n\
             shared token: limitallbenchkeyword\n\
             document: {idx}\n\
             retrieval throughput benchmark corpus entry {idx}\n\
             semantic context topic_{topic} vector lexical fusion quality refinement\n\
             additional deterministic payload alpha beta gamma delta epsilon\n"
        );
        fs::write(&path, content).unwrap_or_else(|error| {
            panic!("failed to write corpus file {}: {error}", path.display())
        });
    }
}

fn write_config(path: &Path, index_dir: &Path, model_dir: &Path, db_path: &Path) {
    let content = format!(
        "[indexing]\n\
         model_dir = \"{}\"\n\
         \n\
         [storage]\n\
         index_dir = \"{}\"\n\
         db_path = \"{}\"\n\
         \n\
         [search]\n\
         fast_only = false\n\
         \n\
         [discovery]\n\
         follow_symlinks = false\n\
         max_file_size_mb = 2\n",
        model_dir.display(),
        index_dir.display(),
        db_path.display(),
    );
    fs::write(path, content)
        .unwrap_or_else(|error| panic!("failed to write config {}: {error}", path.display()));
}

fn run_fsfs(
    args: &[&str],
    config_path: &Path,
    timeout: Duration,
    disable_query_cache: bool,
) -> CommandResult {
    let mut cmd = Command::new(fsfs_binary());
    cmd.args(args);
    cmd.arg("--config").arg(config_path);
    cmd.arg("--no-color");
    cmd.env("FRANKENSEARCH_OFFLINE", "1");
    cmd.env("FRANKENSEARCH_CHECK_UPDATES", "0");
    if disable_query_cache {
        cmd.env("FSFS_DISABLE_QUERY_CACHE", "1");
    }
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let start = Instant::now();
    let mut child = cmd
        .spawn()
        .unwrap_or_else(|error| panic!("failed to spawn fsfs process: {error}"));

    let mut child_stdout = child.stdout.take().expect("piped stdout");
    let mut child_stderr = child.stderr.take().expect("piped stderr");
    let stdout_reader = std::thread::spawn(move || {
        let mut stdout = String::new();
        child_stdout.read_to_string(&mut stdout).ok();
        stdout
    });
    let stderr_reader = std::thread::spawn(move || {
        let mut stderr = String::new();
        child_stderr.read_to_string(&mut stderr).ok();
        stderr
    });

    let deadline = Instant::now() + timeout;
    let mut timed_out = false;
    let exit_code = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status.code().unwrap_or(-1),
            Ok(None) if Instant::now() >= deadline => {
                timed_out = true;
                child.kill().ok();
                child.wait().ok();
                break TIMEOUT_EXIT_CODE;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(50)),
            Err(error) => panic!("error while waiting for fsfs process: {error}"),
        }
    };

    let elapsed_ms = duration_to_ms(start.elapsed());
    let stdout = stdout_reader.join().unwrap_or_default();
    let stderr = stderr_reader.join().unwrap_or_default();
    CommandResult {
        stdout,
        stderr,
        exit_code,
        elapsed_ms,
        timed_out,
    }
}

fn duration_to_ms(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn parse_search_envelope(stdout: &str) -> Option<SearchEnvelope> {
    let json_start = stdout.find('{')?;
    serde_json::from_str::<SearchEnvelope>(&stdout[json_start..]).ok()
}

fn copy_vector_only_index_assets(source_index_root: &Path, target_index_root: &Path) {
    let source_vector = source_index_root.join("vector/index.fsvi");
    assert!(
        source_vector.is_file(),
        "source vector index does not exist at {}",
        source_vector.display()
    );

    let target_vector = target_index_root.join("vector/index.fsvi");
    if let Some(parent) = target_vector.parent() {
        fs::create_dir_all(parent).unwrap_or_else(|error| {
            panic!(
                "failed to create vector-only index directory {}: {error}",
                parent.display()
            )
        });
    }

    fs::copy(&source_vector, &target_vector).unwrap_or_else(|error| {
        panic!(
            "failed to copy vector index {} -> {}: {error}",
            source_vector.display(),
            target_vector.display()
        )
    });
    assert!(
        !target_index_root.join("lexical").exists(),
        "vector-only benchmark fixture unexpectedly contains lexical index"
    );
}

fn run_mode_benchmark_cold(
    mode: &str,
    corpus_root: &Path,
    config_path: &Path,
    expected_docs: usize,
) -> Vec<ModeRunResult> {
    let corpus_arg = corpus_root.display().to_string();
    let args = [
        "search",
        QUERY,
        "--roots",
        corpus_arg.as_str(),
        "--limit",
        "all",
        "--format",
        "json",
    ];

    for _ in 0..WARMUP_RUNS {
        let warmup = run_fsfs(&args, config_path, SEARCH_TIMEOUT, true);
        assert!(
            warmup.exit_code == 0 || warmup.timed_out,
            "warmup search failed in mode={mode}\nstdout:\n{}\nstderr:\n{}",
            warmup.stdout,
            warmup.stderr
        );
        let envelope = parse_search_envelope(&warmup.stdout).unwrap_or_else(|| {
            panic!(
                "warmup search did not emit parseable JSON envelope in mode={mode}\nstdout:\n{}\nstderr:\n{}",
                warmup.stdout,
                warmup.stderr
            )
        });
        assert!(
            envelope.ok,
            "warmup search envelope failed in mode={mode}: {:?}\nstdout:\n{}\nstderr:\n{}",
            envelope.error, warmup.stdout, warmup.stderr
        );
    }

    let mut results = Vec::with_capacity(MEASURED_RUNS);
    for _ in 0..MEASURED_RUNS {
        let run = run_fsfs(&args, config_path, SEARCH_TIMEOUT, true);
        assert!(
            run.exit_code == 0 || run.timed_out,
            "measured search failed in mode={mode}\nstdout:\n{}\nstderr:\n{}",
            run.stdout,
            run.stderr
        );
        let envelope = parse_search_envelope(&run.stdout).unwrap_or_else(|| {
            panic!(
                "measured search did not emit parseable JSON envelope in mode={mode}\nstdout:\n{}\nstderr:\n{}",
                run.stdout,
                run.stderr
            )
        });
        assert!(
            envelope.ok,
            "measured search envelope failed in mode={mode}: {:?}\nstdout:\n{}\nstderr:\n{}",
            envelope.error, run.stdout, run.stderr
        );
        let payload = envelope
            .data
            .as_ref()
            .unwrap_or_else(|| panic!("measured search missing payload in mode={mode}"));
        assert_eq!(
            payload.returned_hits, expected_docs,
            "mode={mode} returned hits did not match corpus size under --limit all"
        );
        assert_eq!(
            payload.total_candidates, expected_docs,
            "mode={mode} candidate count did not match corpus size under --limit all"
        );
        let elapsed_ms = envelope.meta.duration_ms.unwrap_or(run.elapsed_ms);
        results.push(ModeRunResult {
            elapsed_ms,
            returned_hits: payload.returned_hits,
            total_candidates: payload.total_candidates,
        });
    }

    results
}

fn run_mode_benchmark_warm(config_path: &Path, expected_docs: usize) -> Vec<ModeRunResult> {
    let mut cmd = Command::new(fsfs_binary());
    cmd.arg("serve");
    cmd.arg("--config").arg(config_path);
    cmd.arg("--no-color");
    cmd.arg("--format").arg("jsonl");
    cmd.env("FRANKENSEARCH_OFFLINE", "1");
    cmd.env("FRANKENSEARCH_CHECK_UPDATES", "0");
    cmd.env("FSFS_DISABLE_QUERY_CACHE", "1");
    cmd.stdin(Stdio::piped());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::null());

    let mut child = cmd
        .spawn()
        .unwrap_or_else(|error| panic!("failed to spawn fsfs serve process: {error}"));
    let mut child_stdin = child.stdin.take().expect("piped stdin");
    let child_stdout = child.stdout.take().expect("piped stdout");
    let mut reader = BufReader::new(child_stdout);

    let mut ready_line = String::new();
    reader
        .read_line(&mut ready_line)
        .expect("read serve ready line");
    let ready: ServeReadyEnvelope =
        serde_json::from_str(ready_line.trim()).expect("parse serve ready envelope");
    assert!(ready.ok, "serve did not report ready");
    assert_eq!(ready.event.as_deref(), Some("ready"));

    let mut results = Vec::with_capacity(MEASURED_RUNS);
    for run_idx in 0..(WARMUP_RUNS + MEASURED_RUNS) {
        let request = serde_json::json!({
            "query": QUERY,
            "limit": 0,
            "mode": "full",
        });
        let request_line = format!("{request}\n");

        let started = Instant::now();
        child_stdin
            .write_all(request_line.as_bytes())
            .expect("write serve request");
        child_stdin.flush().expect("flush serve request");

        let mut response_line = String::new();
        reader
            .read_line(&mut response_line)
            .expect("read serve response");
        let elapsed_ms = duration_to_ms(started.elapsed());
        let response: ServeResponseEnvelope =
            serde_json::from_str(response_line.trim()).expect("parse serve response");
        assert!(
            response.ok,
            "serve response failed: {:?}",
            response.error.as_deref()
        );
        let payload = response
            .payloads
            .last()
            .unwrap_or_else(|| panic!("serve response missing payload at run {run_idx}"));
        assert_eq!(
            payload.returned_hits, expected_docs,
            "warm serve returned hits did not match corpus size under --limit all"
        );
        assert_eq!(
            payload.total_candidates, expected_docs,
            "warm serve candidate count did not match corpus size under --limit all"
        );
        if run_idx >= WARMUP_RUNS {
            results.push(ModeRunResult {
                elapsed_ms,
                returned_hits: payload.returned_hits,
                total_candidates: payload.total_candidates,
            });
        }
    }

    child_stdin
        .write_all(b"quit\n")
        .expect("write serve quit signal");
    child_stdin.flush().expect("flush serve quit signal");
    let _ = child.kill();
    let _ = child.wait();

    results
}

fn summarize_mode(mode: &str, runs: &[ModeRunResult]) -> ModeSummary {
    assert!(!runs.is_empty(), "mode summary requires at least one run");

    let mut latencies = runs.iter().map(|run| run.elapsed_ms).collect::<Vec<_>>();
    latencies.sort_unstable();
    let latency_sum = latencies.iter().copied().sum::<u64>();
    let latency_mean = u64_to_f64(latency_sum) / usize_to_f64(latencies.len());

    let returned_hits = runs.iter().map(|run| run.returned_hits).collect::<Vec<_>>();
    let total_candidates = runs
        .iter()
        .map(|run| run.total_candidates)
        .collect::<Vec<_>>();

    ModeSummary {
        mode: mode.to_owned(),
        samples: runs.len(),
        latency_ms_min: latencies[0],
        latency_ms_p50: percentile_u64(&latencies, 50),
        latency_ms_p95: percentile_u64(&latencies, 95),
        latency_ms_max: *latencies.last().unwrap_or(&latencies[0]),
        latency_ms_mean: latency_mean,
        returned_hits_min: *returned_hits.iter().min().unwrap_or(&0),
        returned_hits_max: *returned_hits.iter().max().unwrap_or(&0),
        total_candidates_min: *total_candidates.iter().min().unwrap_or(&0),
        total_candidates_max: *total_candidates.iter().max().unwrap_or(&0),
    }
}

fn percentile_u64(sorted_values: &[u64], percentile: usize) -> u64 {
    assert!(
        !sorted_values.is_empty(),
        "percentile requires non-empty input"
    );
    let last_index = sorted_values.len() - 1;
    let scaled_index = last_index.saturating_mul(percentile).saturating_add(50) / 100;
    sorted_values[scaled_index.min(last_index)]
}

#[inline]
#[allow(clippy::cast_precision_loss)]
const fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

#[inline]
fn u64_to_f64(value: u64) -> f64 {
    f64::from(u32::try_from(value).unwrap_or(u32::MAX))
}

#[test]
#[ignore = "Performance benchmark; run explicitly with --ignored --nocapture"]
fn limit_all_large_corpus_vector_only_and_hybrid_benchmark() {
    let temp = tempfile::tempdir().expect("create benchmark tempdir");
    let corpus_root = temp.path().join("corpus");
    let model_root = temp.path().join("models");
    let hybrid_index_root = temp.path().join("index_hybrid");
    let vector_only_index_root = temp.path().join("index_vector_only");
    let hybrid_db = temp.path().join("hybrid.sqlite");
    let vector_only_db = temp.path().join("vector_only.sqlite");
    let hybrid_config = temp.path().join("hybrid.toml");
    let vector_only_config = temp.path().join("vector_only.toml");

    fs::create_dir_all(&corpus_root).expect("create corpus root");
    fs::create_dir_all(&model_root).expect("create model root");
    fs::create_dir_all(&hybrid_index_root).expect("create hybrid index root");
    fs::create_dir_all(&vector_only_index_root).expect("create vector-only index root");

    write_large_corpus(&corpus_root, DOC_COUNT);
    write_config(&hybrid_config, &hybrid_index_root, &model_root, &hybrid_db);
    write_config(
        &vector_only_config,
        &vector_only_index_root,
        &model_root,
        &vector_only_db,
    );

    let corpus_arg = corpus_root.display().to_string();
    let index_run = run_fsfs(
        &["index", corpus_arg.as_str()],
        &hybrid_config,
        INDEX_TIMEOUT,
        false,
    );
    assert!(
        index_run.exit_code == 0 || (index_run.timed_out && index_run.stdout.contains("Indexed ")),
        "hybrid indexing failed\nstdout:\n{}\nstderr:\n{}",
        index_run.stdout,
        index_run.stderr
    );

    copy_vector_only_index_assets(&hybrid_index_root, &vector_only_index_root);

    let vector_only_cold_runs =
        run_mode_benchmark_cold("vector_only", &corpus_root, &vector_only_config, DOC_COUNT);
    let hybrid_cold_runs =
        run_mode_benchmark_cold("hybrid", &corpus_root, &hybrid_config, DOC_COUNT);
    let vector_only_warm_runs = run_mode_benchmark_warm(&vector_only_config, DOC_COUNT);
    let hybrid_warm_runs = run_mode_benchmark_warm(&hybrid_config, DOC_COUNT);

    let vector_only_cold_summary = summarize_mode("vector_only_cold", &vector_only_cold_runs);
    let vector_only_warm_summary = summarize_mode("vector_only_warm", &vector_only_warm_runs);
    let hybrid_cold_summary = summarize_mode("hybrid_cold", &hybrid_cold_runs);
    let hybrid_warm_summary = summarize_mode("hybrid_warm", &hybrid_warm_runs);

    assert!(
        hybrid_cold_summary.latency_ms_p95
            <= vector_only_cold_summary
                .latency_ms_p95
                .saturating_mul(6)
                .saturating_add(500),
        "hybrid p95 latency regression is unexpectedly high: hybrid={}ms vector_only={}ms",
        hybrid_cold_summary.latency_ms_p95,
        vector_only_cold_summary.latency_ms_p95
    );

    assert!(
        hybrid_warm_summary.latency_ms_p95
            <= hybrid_cold_summary
                .latency_ms_p95
                .saturating_mul(3)
                .saturating_add(250),
        "warm hybrid p95 should be materially better than cold path: warm={}ms cold={}ms",
        hybrid_warm_summary.latency_ms_p95,
        hybrid_cold_summary.latency_ms_p95
    );

    let report = LimitAllBenchmarkReport {
        schema_version: REPORT_SCHEMA_VERSION.to_owned(),
        corpus_docs: DOC_COUNT,
        query: QUERY.to_owned(),
        warmup_runs: WARMUP_RUNS,
        measured_runs: MEASURED_RUNS,
        hybrid: ModeBenchmarkSuite {
            cold: hybrid_cold_summary,
            warm: hybrid_warm_summary,
        },
        vector_only: ModeBenchmarkSuite {
            cold: vector_only_cold_summary,
            warm: vector_only_warm_summary,
        },
    };

    let json = serde_json::to_string_pretty(&report).expect("serialize benchmark report");
    eprintln!("FSFS_LIMIT_ALL_BENCHMARK_REPORT={json}");
}
