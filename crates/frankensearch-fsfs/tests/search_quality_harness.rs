//! Search-quality evaluation harness for fsfs.
//!
//! This integration test runs a deterministic end-to-end relevance harness:
//! - indexes the shared fixture corpus,
//! - executes fixture relevance queries,
//! - computes nDCG@10, MRR, and Recall@10,
//! - computes 95% bootstrap confidence intervals on each metric,
//! - aggregates metrics by query slice,
//! - persists a machine-readable artifact report for replay/regression workflows.
//!
//! Beads: bd-2hz.9.7, bd-2hz.9.7.2, bd-21c8

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::fs;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use frankensearch_core::QueryClass;
use frankensearch_core::metrics_eval::{
    QualityMetric, QualityMetricSamples, bootstrap_ci, mrr, ndcg_at_k, quality_comparison,
    recall_at_k,
};
use serde::{Deserialize, Serialize};

const REPORT_SCHEMA_VERSION: &str = "fsfs-quality-harness-v2";
const REPORT_MATRIX_VERSION: &str = "fsfs-quality-matrix-v1";
const TOP_K: usize = 10;
const BOOTSTRAP_RESAMPLES: usize = 2000;
const BOOTSTRAP_CONFIDENCE: f64 = 0.95;
const BOOTSTRAP_SEED: u64 = 0xF5A5_EA5C_CAFE;
const COMPARE_K: usize = 5;
const INDEX_PROCESS_TIMEOUT: Duration = Duration::from_secs(90);
const SEARCH_PROCESS_TIMEOUT: Duration = Duration::from_secs(20);
const DEFAULT_PROCESS_TIMEOUT: Duration = Duration::from_secs(30);
const TIMEOUT_EXIT_CODE: i32 = 124;
const ALL_SLICES: [&str; 5] = [
    "identifier",
    "short_keyword",
    "natural_language",
    "path_heavy",
    "code_symbol",
];

#[derive(Debug, Deserialize)]
struct CorpusFixture {
    version: String,
    documents: Vec<FixtureDocument>,
}

#[derive(Debug, Deserialize)]
struct FixtureDocument {
    doc_id: String,
    title: String,
    content: String,
    doc_type: String,
}

#[derive(Debug, Deserialize)]
struct RelevanceFixture {
    version: String,
    queries: Vec<RelevanceQueryFixture>,
}

#[derive(Debug, Deserialize)]
struct RelevanceQueryFixture {
    query: String,
    expected_top_10: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct QuerySliceFixture {
    query: String,
    query_class: String,
}

#[derive(Debug, Deserialize)]
struct SearchEnvelope {
    v: u32,
    ok: bool,
    data: Option<SearchPayload>,
    error: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct SearchPayload {
    query: String,
    hits: Vec<SearchHit>,
}

#[derive(Debug, Deserialize)]
struct SearchHit {
    path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ConfidenceInterval {
    mean: f64,
    lower: f64,
    upper: f64,
    std_error: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MetricSummary {
    query_count: usize,
    ndcg_at_10: f64,
    mrr: f64,
    recall_at_10: f64,
    latency_ms_avg: f64,
    latency_ms_p50: u64,
    latency_ms_p95: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    ndcg_ci: Option<ConfidenceInterval>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mrr_ci: Option<ConfidenceInterval>,
    #[serde(skip_serializing_if = "Option::is_none")]
    recall_ci: Option<ConfidenceInterval>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QueryMetricRecord {
    query: String,
    query_slice: String,
    ndcg_at_10: f64,
    mrr: f64,
    recall_at_10: f64,
    latency_ms: u64,
    retrieved_doc_ids: Vec<String>,
    expected_doc_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProfileMetricComparison {
    metric: String,
    mean_a: f64,
    mean_b: f64,
    mean_diff: f64,
    ci_lower: f64,
    ci_upper: f64,
    p_value: f64,
    significant: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProfileComparisonSummary {
    profile_a: String,
    profile_b: String,
    query_count: usize,
    confidence: f64,
    n_resamples: usize,
    metrics: Vec<ProfileMetricComparison>,
    tsv_report: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QualityHarnessReport {
    schema_version: String,
    matrix_version: String,
    top_k: usize,
    corpus_fixture_version: String,
    relevance_fixture_version: String,
    total_queries: usize,
    replay_command: String,
    overall: MetricSummary,
    per_slice: BTreeMap<String, MetricSummary>,
    per_query: Vec<QueryMetricRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    profile_comparison: Option<ProfileComparisonSummary>,
}

#[derive(Debug)]
struct CommandResult {
    stdout: String,
    stderr: String,
    exit_code: i32,
    elapsed_ms: u64,
}

#[derive(Debug, Default, Clone)]
struct MetricAccumulator {
    query_count: usize,
    ndcg_sum: f64,
    mrr_sum: f64,
    recall_sum: f64,
    ndcg_scores: Vec<f64>,
    mrr_scores: Vec<f64>,
    recall_scores: Vec<f64>,
    latencies_ms: Vec<u64>,
}

impl MetricAccumulator {
    fn push(&mut self, ndcg_at_10: f64, reciprocal_rank: f64, recall_at_10: f64, latency_ms: u64) {
        self.query_count += 1;
        self.ndcg_sum += ndcg_at_10;
        self.mrr_sum += reciprocal_rank;
        self.recall_sum += recall_at_10;
        self.ndcg_scores.push(ndcg_at_10);
        self.mrr_scores.push(reciprocal_rank);
        self.recall_scores.push(recall_at_10);
        self.latencies_ms.push(latency_ms);
    }

    fn finalize(mut self) -> MetricSummary {
        self.latencies_ms.sort_unstable();

        if self.query_count == 0 {
            return MetricSummary {
                query_count: 0,
                ndcg_at_10: 0.0,
                mrr: 0.0,
                recall_at_10: 0.0,
                latency_ms_avg: 0.0,
                latency_ms_p50: 0,
                latency_ms_p95: 0,
                ndcg_ci: None,
                mrr_ci: None,
                recall_ci: None,
            };
        }

        let count = usize_to_f64(self.query_count);
        let avg_latency = self
            .latencies_ms
            .iter()
            .map(|value| u64_to_f64(*value))
            .sum::<f64>()
            / count;

        let ndcg_ci = compute_ci(&self.ndcg_scores);
        let mrr_ci = compute_ci(&self.mrr_scores);
        let recall_ci = compute_ci(&self.recall_scores);

        MetricSummary {
            query_count: self.query_count,
            ndcg_at_10: self.ndcg_sum / count,
            mrr: self.mrr_sum / count,
            recall_at_10: self.recall_sum / count,
            latency_ms_avg: avg_latency,
            latency_ms_p50: percentile_u64(&self.latencies_ms, 50),
            latency_ms_p95: percentile_u64(&self.latencies_ms, 95),
            ndcg_ci,
            mrr_ci,
            recall_ci,
        }
    }
}

fn compute_ci(scores: &[f64]) -> Option<ConfidenceInterval> {
    bootstrap_ci(
        scores,
        BOOTSTRAP_CONFIDENCE,
        BOOTSTRAP_RESAMPLES,
        BOOTSTRAP_SEED,
    )
    .map(|ci| ConfidenceInterval {
        mean: ci.mean,
        lower: ci.lower,
        upper: ci.upper,
        std_error: ci.std_error,
    })
}

#[inline]
#[allow(clippy::cast_precision_loss)]
const fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

#[inline]
#[allow(clippy::cast_precision_loss)]
const fn u64_to_f64(value: u64) -> f64 {
    value as f64
}

fn percentile_u64(sorted_values: &[u64], percentile: usize) -> u64 {
    if sorted_values.is_empty() {
        return 0;
    }

    let last_index = sorted_values.len() - 1;
    let scaled_index = last_index.saturating_mul(percentile).saturating_add(50) / 100;
    sorted_values[scaled_index.min(last_index)]
}

fn fsfs_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_fsfs"))
}

fn fixtures_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}

fn load_fixture<T>(file_name: &str) -> T
where
    T: for<'de> Deserialize<'de>,
{
    let path = fixtures_root().join(file_name);
    let content = fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("failed to read fixture {}: {err}", path.display()));
    serde_json::from_str::<T>(&content)
        .unwrap_or_else(|err| panic!("failed to parse fixture {}: {err}", path.display()))
}

fn write_corpus_fixture(corpus: &CorpusFixture, corpus_dir: &Path) {
    for doc in &corpus.documents {
        let relative = format!("{}/{}.txt", doc.doc_type, doc.doc_id);
        let path = corpus_dir.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap_or_else(|err| {
                panic!(
                    "failed to create fixture directory {}: {err}",
                    parent.display()
                )
            });
        }

        let body = format!(
            "# {}\n\n{}\n\nid: {}\ncluster: {}\n",
            doc.title, doc.content, doc.doc_id, doc.doc_type
        );
        fs::write(&path, body)
            .unwrap_or_else(|err| panic!("failed to write fixture file {}: {err}", path.display()));
    }
}

fn write_config(path: &Path, index_dir: &Path, model_dir: &Path, db_path: &Path) {
    let config = format!(
        "[indexing]\n\
         model_dir = \"{}\"\n\
         watch_mode = false\n\
         \n\
         [storage]\n\
         index_dir = \"{}\"\n\
         db_path = \"{}\"\n\
         \n\
         [search]\n\
         fast_only = true\n\
         \n\
         [discovery]\n\
         follow_symlinks = false\n\
         max_file_size_mb = 10\n",
        model_dir.display(),
        index_dir.display(),
        db_path.display(),
    );
    fs::write(path, config)
        .unwrap_or_else(|err| panic!("failed to write test config {}: {err}", path.display()));
}

fn run_fsfs(args: &[String], config_path: &Path) -> CommandResult {
    let start = Instant::now();
    let timeout = command_timeout(args);
    let command_name = args.first().map_or("unknown", String::as_str);

    let mut command = Command::new(fsfs_binary());
    command.args(args.iter().map(String::as_str));
    command.arg("--config").arg(config_path);
    command.env("FRANKENSEARCH_OFFLINE", "1");
    command.env("FRANKENSEARCH_CHECK_UPDATES", "0");
    command.arg("--no-color");
    command.stdout(Stdio::piped());
    command.stderr(Stdio::piped());

    let mut child = command
        .spawn()
        .unwrap_or_else(|err| panic!("failed to spawn fsfs with args {args:?}: {err}"));

    let mut child_stdout = child.stdout.take().expect("missing piped stdout");
    let mut child_stderr = child.stderr.take().expect("missing piped stderr");

    let stdout_thread = std::thread::spawn(move || {
        let mut buf = String::new();
        child_stdout.read_to_string(&mut buf).ok();
        buf
    });
    let stderr_thread = std::thread::spawn(move || {
        let mut buf = String::new();
        child_stderr.read_to_string(&mut buf).ok();
        buf
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
            Err(err) => panic!("error waiting for fsfs process: {err}"),
        }
    };

    let stdout = stdout_thread.join().unwrap_or_default();
    let mut stderr = stderr_thread.join().unwrap_or_default();
    if timed_out {
        if !stderr.is_empty() {
            stderr.push('\n');
        }
        let _ = write!(
            stderr,
            "[harness-timeout] command '{command_name}' exceeded {}s and was terminated",
            timeout.as_secs()
        );
    }
    let elapsed_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);

    CommandResult {
        stdout,
        stderr,
        exit_code,
        elapsed_ms,
    }
}

fn command_timeout(args: &[String]) -> Duration {
    match args.first().map(String::as_str) {
        Some("index") => INDEX_PROCESS_TIMEOUT,
        Some("search") => SEARCH_PROCESS_TIMEOUT,
        _ => DEFAULT_PROCESS_TIMEOUT,
    }
}

fn parse_search_envelope(stdout: &str) -> SearchEnvelope {
    let json_start = stdout
        .find('{')
        .unwrap_or_else(|| panic!("no JSON payload found in stdout:\n{stdout}"));
    let json_text = &stdout[json_start..];

    serde_json::from_str::<SearchEnvelope>(json_text)
        .unwrap_or_else(|err| panic!("failed to parse search envelope: {err}\nraw:\n{json_text}"))
}

fn normalize_query(query: &str) -> String {
    query.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn load_query_slice_map() -> HashMap<String, String> {
    let fixtures = load_fixture::<Vec<QuerySliceFixture>>("queries.json");
    fixtures
        .into_iter()
        .map(|entry| {
            (
                normalize_query(&entry.query),
                normalize_fixture_slice(&entry.query_class),
            )
        })
        .collect()
}

fn normalize_fixture_slice(slice: &str) -> String {
    match slice {
        "Identifier" => "identifier".to_owned(),
        "ShortKeyword" => "short_keyword".to_owned(),
        "NaturalLanguage" => "natural_language".to_owned(),
        other => other.to_ascii_lowercase(),
    }
}

fn classify_query_slice(query: &str, fixture_slice_map: &HashMap<String, String>) -> String {
    let trimmed = query.trim();

    if looks_path_heavy(trimmed) {
        return "path_heavy".to_owned();
    }
    if looks_code_symbol(trimmed) {
        return "code_symbol".to_owned();
    }

    let normalized = normalize_query(trimmed);
    if let Some(slice) = fixture_slice_map.get(&normalized) {
        return slice.clone();
    }

    QueryClass::classify(trimmed).to_string()
}

fn looks_path_heavy(query: &str) -> bool {
    query.contains('/') || query.contains('\\') || query.contains(".rs") || query.contains(".toml")
}

fn looks_code_symbol(query: &str) -> bool {
    query.contains("::") || query.contains("fn ") || query.contains("struct ")
}

fn doc_id_from_hit_path(path: &str) -> Option<String> {
    let stem = Path::new(path).file_stem()?.to_str()?;
    Some(stem.to_owned())
}

fn report_path(temp_dir: &Path) -> PathBuf {
    temp_dir.join("quality_harness_report.json")
}

#[test]
#[allow(clippy::too_many_lines)]
fn quality_harness_reports_metrics_by_query_slice() {
    let corpus_fixture = load_fixture::<CorpusFixture>("corpus.json");
    let relevance_fixture = load_fixture::<RelevanceFixture>("relevance.json");
    let query_slice_map = load_query_slice_map();

    let temp = tempfile::tempdir().expect("create tempdir");
    let corpus_dir = temp.path().join("corpus");
    let index_dir = temp.path().join("index");
    let model_dir = temp.path().join("models");
    let db_dir = temp.path().join("db");
    let config_path = temp.path().join("fsfs-quality-harness.toml");
    let db_path = db_dir.join("quality_harness.db");

    fs::create_dir_all(&corpus_dir).expect("create corpus dir");
    fs::create_dir_all(&index_dir).expect("create index dir");
    fs::create_dir_all(&model_dir).expect("create model dir");
    fs::create_dir_all(&db_dir).expect("create db dir");

    write_corpus_fixture(&corpus_fixture, &corpus_dir);
    write_config(&config_path, &index_dir, &model_dir, &db_path);

    let index_args = vec![
        "index".to_owned(),
        corpus_dir.to_string_lossy().to_string(),
        "--format".to_owned(),
        "json".to_owned(),
    ];
    let index_result = run_fsfs(&index_args, &config_path);
    assert_eq!(
        index_result.exit_code, 0,
        "index command failed (exit={}):\nstdout:\n{}\nstderr:\n{}",
        index_result.exit_code, index_result.stdout, index_result.stderr
    );

    let query_count = relevance_fixture.queries.len();
    let mut overall = MetricAccumulator::default();
    let mut per_slice: BTreeMap<String, MetricAccumulator> = BTreeMap::new();
    for slice in ALL_SLICES {
        per_slice.insert(slice.to_owned(), MetricAccumulator::default());
    }

    let mut per_query = Vec::with_capacity(query_count);
    let top_k_string = TOP_K.to_string();

    for fixture_query in &relevance_fixture.queries {
        let search_args = vec![
            "search".to_owned(),
            fixture_query.query.clone(),
            "--format".to_owned(),
            "json".to_owned(),
            "--limit".to_owned(),
            top_k_string.clone(),
            "--index-dir".to_owned(),
            index_dir.to_string_lossy().to_string(),
        ];
        let search_result = run_fsfs(&search_args, &config_path);
        assert_eq!(
            search_result.exit_code,
            0,
            "search command failed for query '{}' (exit={}):\nstdout:\n{}\nstderr:\n{}",
            fixture_query.query,
            search_result.exit_code,
            search_result.stdout,
            search_result.stderr
        );

        let envelope = parse_search_envelope(&search_result.stdout);
        assert_eq!(envelope.v, 1, "unexpected search envelope version");
        assert!(
            envelope.ok,
            "search failed for query '{}': {:?}",
            fixture_query.query, envelope.error
        );
        let payload = envelope
            .data
            .unwrap_or_else(|| panic!("missing payload for query '{}'", fixture_query.query));
        assert_eq!(
            payload.query,
            normalize_query(&fixture_query.query),
            "query mismatch in search response"
        );

        let retrieved_doc_ids: Vec<String> = payload
            .hits
            .iter()
            .take(TOP_K)
            .filter_map(|hit| doc_id_from_hit_path(&hit.path))
            .collect();

        let expected_refs: Vec<&str> = fixture_query
            .expected_top_10
            .iter()
            .map(String::as_str)
            .collect();
        let retrieved_refs: Vec<&str> = retrieved_doc_ids.iter().map(String::as_str).collect();

        let ndcg = ndcg_at_k(&retrieved_refs, &expected_refs, TOP_K);
        let reciprocal_rank = mrr(&retrieved_refs, &expected_refs);
        let recall = recall_at_k(&retrieved_refs, &expected_refs, TOP_K);
        let slice = classify_query_slice(&fixture_query.query, &query_slice_map);

        overall.push(ndcg, reciprocal_rank, recall, search_result.elapsed_ms);
        per_slice.entry(slice.clone()).or_default().push(
            ndcg,
            reciprocal_rank,
            recall,
            search_result.elapsed_ms,
        );

        per_query.push(QueryMetricRecord {
            query: fixture_query.query.clone(),
            query_slice: slice,
            ndcg_at_10: ndcg,
            mrr: reciprocal_rank,
            recall_at_10: recall,
            latency_ms: search_result.elapsed_ms,
            retrieved_doc_ids,
            expected_doc_ids: fixture_query.expected_top_10.clone(),
        });
    }

    let overall_summary = overall.finalize();
    assert!(overall_summary.ndcg_at_10.is_finite());
    assert!(overall_summary.mrr.is_finite());
    assert!(overall_summary.recall_at_10.is_finite());
    assert!(
        overall_summary.ndcg_at_10 > 0.0,
        "ndcg@10 must be > 0, got {}",
        overall_summary.ndcg_at_10
    );
    assert!(
        overall_summary.mrr > 0.0,
        "mrr must be > 0, got {}",
        overall_summary.mrr
    );
    assert!(
        overall_summary.recall_at_10 > 0.0,
        "recall@10 must be > 0, got {}",
        overall_summary.recall_at_10
    );

    // Verify bootstrap confidence intervals on overall metrics.
    if let Some(ref ci) = overall_summary.ndcg_ci {
        assert!(
            ci.lower <= ci.mean && ci.mean <= ci.upper,
            "nDCG CI [{}, {}] should contain mean {}",
            ci.lower,
            ci.upper,
            ci.mean
        );
        assert!(ci.std_error >= 0.0, "std_error must be non-negative");
    }
    if let Some(ref ci) = overall_summary.mrr_ci {
        assert!(
            ci.lower <= ci.mean && ci.mean <= ci.upper,
            "MRR CI [{}, {}] should contain mean {}",
            ci.lower,
            ci.upper,
            ci.mean
        );
    }
    if let Some(ref ci) = overall_summary.recall_ci {
        assert!(
            ci.lower <= ci.mean && ci.mean <= ci.upper,
            "Recall CI [{}, {}] should contain mean {}",
            ci.lower,
            ci.upper,
            ci.mean
        );
    }

    // Profile variant comparison: evaluate at COMPARE_K from the same retrieved results
    // and compare against the full TOP_K metrics using quality_comparison().
    let mut k_short_ndcg = Vec::with_capacity(per_query.len());
    let mut k_short_mrr = Vec::with_capacity(per_query.len());
    let mut k_short_recall = Vec::with_capacity(per_query.len());
    let mut k_full_ndcg = Vec::with_capacity(per_query.len());
    let mut k_full_mrr = Vec::with_capacity(per_query.len());
    let mut k_full_recall = Vec::with_capacity(per_query.len());

    for record in &per_query {
        let truncated: Vec<&str> = record
            .retrieved_doc_ids
            .iter()
            .take(COMPARE_K)
            .map(String::as_str)
            .collect();
        let expected: Vec<&str> = record.expected_doc_ids.iter().map(String::as_str).collect();

        // Compare both profiles on the same metric definition (TOP_K).
        // The truncated profile simply has fewer retrieved candidates.
        k_short_ndcg.push(ndcg_at_k(&truncated, &expected, TOP_K));
        k_short_mrr.push(mrr(&truncated, &expected));
        k_short_recall.push(recall_at_k(&truncated, &expected, TOP_K));

        k_full_ndcg.push(record.ndcg_at_10);
        k_full_mrr.push(record.mrr);
        k_full_recall.push(record.recall_at_10);
    }

    let samples = [
        QualityMetricSamples {
            metric: QualityMetric::NdcgAtK(TOP_K),
            scores_a: &k_full_ndcg,
            scores_b: &k_short_ndcg,
        },
        QualityMetricSamples {
            metric: QualityMetric::Mrr,
            scores_a: &k_full_mrr,
            scores_b: &k_short_mrr,
        },
        QualityMetricSamples {
            metric: QualityMetric::RecallAtK(TOP_K),
            scores_a: &k_full_recall,
            scores_b: &k_short_recall,
        },
    ];

    let comparison = quality_comparison(
        &samples,
        BOOTSTRAP_CONFIDENCE,
        BOOTSTRAP_RESAMPLES,
        BOOTSTRAP_SEED,
    )
    .expect("quality comparison samples must be non-empty and aligned");
    let tsv_report = comparison.render_tsv_report();
    let profile_comparison = Some(ProfileComparisonSummary {
        profile_a: format!("full@{TOP_K}"),
        profile_b: format!("truncated@{COMPARE_K}"),
        query_count: comparison.query_count,
        confidence: comparison.confidence,
        n_resamples: comparison.n_resamples,
        metrics: comparison
            .metrics
            .iter()
            .map(|metric| ProfileMetricComparison {
                metric: metric.metric.to_string(),
                mean_a: metric.comparison.mean_a,
                mean_b: metric.comparison.mean_b,
                mean_diff: metric.comparison.mean_diff,
                ci_lower: metric.comparison.ci_lower,
                ci_upper: metric.comparison.ci_upper,
                p_value: metric.comparison.p_value,
                significant: metric.comparison.significant,
            })
            .collect(),
        tsv_report,
    });

    let per_slice_summary: BTreeMap<String, MetricSummary> = per_slice
        .into_iter()
        .map(|(slice, accumulator)| (slice, accumulator.finalize()))
        .collect();

    let populated_slice_count = per_slice_summary
        .values()
        .filter(|summary| summary.query_count > 0)
        .count();
    assert!(
        populated_slice_count >= 1,
        "expected at least one populated query slice, got {populated_slice_count}"
    );

    let report = QualityHarnessReport {
        schema_version: REPORT_SCHEMA_VERSION.to_owned(),
        matrix_version: REPORT_MATRIX_VERSION.to_owned(),
        top_k: TOP_K,
        corpus_fixture_version: corpus_fixture.version,
        relevance_fixture_version: relevance_fixture.version,
        total_queries: query_count,
        replay_command:
            "cargo test -p frankensearch-fsfs --test search_quality_harness -- --nocapture"
                .to_owned(),
        overall: overall_summary,
        per_slice: per_slice_summary,
        per_query,
        profile_comparison,
    };

    let output_path = report_path(temp.path());
    let json = serde_json::to_string_pretty(&report).expect("serialize harness report");
    fs::write(&output_path, json)
        .unwrap_or_else(|err| panic!("failed to write report {}: {err}", output_path.display()));

    let roundtrip_raw = fs::read_to_string(&output_path)
        .unwrap_or_else(|err| panic!("failed to read report {}: {err}", output_path.display()));
    let roundtrip = serde_json::from_str::<QualityHarnessReport>(&roundtrip_raw)
        .expect("deserialize harness report");

    assert_eq!(roundtrip.schema_version, REPORT_SCHEMA_VERSION);
    assert_eq!(roundtrip.matrix_version, REPORT_MATRIX_VERSION);
    assert_eq!(roundtrip.top_k, TOP_K);
    assert_eq!(roundtrip.total_queries, query_count);
    assert_eq!(roundtrip.per_query.len(), query_count);
    assert_eq!(
        roundtrip.replay_command,
        "cargo test -p frankensearch-fsfs --test search_quality_harness -- --nocapture"
    );

    // Verify bootstrap CI fields survive JSON roundtrip.
    assert_eq!(
        roundtrip.overall.ndcg_ci.is_some(),
        report.overall.ndcg_ci.is_some(),
        "ndcg_ci presence should survive roundtrip"
    );
    if let (Some(orig), Some(rt)) = (&report.overall.ndcg_ci, &roundtrip.overall.ndcg_ci) {
        assert!(
            (orig.mean - rt.mean).abs() < 1e-10,
            "ndcg_ci mean should roundtrip exactly"
        );
        assert!(
            (orig.lower - rt.lower).abs() < 1e-10,
            "ndcg_ci lower should roundtrip exactly"
        );
        assert!(
            (orig.upper - rt.upper).abs() < 1e-10,
            "ndcg_ci upper should roundtrip exactly"
        );
    }

    // Verify profile comparison survived roundtrip and is structurally valid.
    assert_eq!(
        roundtrip.profile_comparison.is_some(),
        report.profile_comparison.is_some(),
        "profile_comparison presence should survive roundtrip"
    );
    if let Some(ref cmp) = report.profile_comparison {
        assert_eq!(cmp.profile_a, format!("full@{TOP_K}"));
        assert_eq!(cmp.profile_b, format!("truncated@{COMPARE_K}"));
        assert_eq!(cmp.query_count, query_count);
        assert!(
            (cmp.confidence - BOOTSTRAP_CONFIDENCE).abs() < f64::EPSILON,
            "confidence should be {BOOTSTRAP_CONFIDENCE}, got {}",
            cmp.confidence
        );
        assert_eq!(cmp.n_resamples, BOOTSTRAP_RESAMPLES);
        assert_eq!(
            cmp.metrics.len(),
            3,
            "expected nDCG, MRR, Recall comparisons"
        );
        for metric_cmp in &cmp.metrics {
            assert!(
                metric_cmp.p_value > 0.0,
                "p-value must be > 0 (plus-one correction), got {} for {}",
                metric_cmp.p_value,
                metric_cmp.metric
            );
            assert!(
                metric_cmp.p_value <= 1.0,
                "p-value must be <= 1.0, got {} for {}",
                metric_cmp.p_value,
                metric_cmp.metric
            );
            assert!(
                metric_cmp.ci_lower <= metric_cmp.ci_upper,
                "CI lower {} must be <= upper {} for {}",
                metric_cmp.ci_lower,
                metric_cmp.ci_upper,
                metric_cmp.metric
            );
        }
        assert!(!cmp.tsv_report.is_empty(), "TSV report should not be empty");
        assert!(
            cmp.tsv_report.contains("metric\t"),
            "TSV report should contain header"
        );
    }
}
