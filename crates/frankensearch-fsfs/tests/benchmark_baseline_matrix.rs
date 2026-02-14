use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};
use tempfile::TempDir;

const GOLDEN_SCHEMA_VERSION: &str = "fsfs-benchmark-golden-v1";
const ARTIFACT_SCHEMA_VERSION: &str = "fsfs-benchmark-artifact-v1";
const MATRIX_VERSION: &str = "fsfs-benchmark-matrix-v1";
const GOLDEN_PROFILES: [&str; 3] = ["tiny", "small", "medium"];
const MAX_ALLOWED_REGRESSION_PCT: u16 = 20;
const REGRESSION_SCALE: u64 = 100;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum BenchmarkPath {
    Crawl,
    Index,
    Query,
    Tui,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ComparatorMetric {
    IndexingThroughputDocsPerSecond,
    SearchLatencyP50Ms,
    SearchLatencyP95Ms,
    SearchLatencyP99Ms,
    FastTierLatencyMs,
    QualityTierLatencyMs,
    IndexingPeakMemoryMb,
    SearchingPeakMemoryMb,
    IndexSizeBytesPerDocument,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct ComparatorDefinition {
    path: BenchmarkPath,
    metric: ComparatorMetric,
    baseline_key: String,
    max_regression_pct: u16,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct BenchmarkCase {
    path: BenchmarkPath,
    dataset_profile: String,
    warmup_iterations: u32,
    measured_iterations: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct BenchmarkMatrix {
    schema_version: String,
    matrix_version: String,
    cases: Vec<BenchmarkCase>,
    comparators: Vec<ComparatorDefinition>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct CorpusStats {
    files: u64,
    tokens: u64,
    lines: u64,
    bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct WorkloadStats {
    crawl_events: u64,
    index_documents: u64,
    query_cases: u64,
    tui_interactions: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct SearchLatencyPercentilesMs {
    p50: u64,
    p95: u64,
    p99: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct BaselineMetrics {
    indexing_throughput_docs_per_second: u64,
    search_latency_ms: SearchLatencyPercentilesMs,
    fast_tier_latency_ms: u64,
    quality_tier_latency_ms: u64,
    indexing_peak_memory_mb: u64,
    searching_peak_memory_mb: u64,
    index_size_bytes_per_document: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct GoldenDataset {
    schema_version: String,
    dataset_version: String,
    profile: String,
    corpus: CorpusStats,
    workload: WorkloadStats,
    baseline_metrics: BaselineMetrics,
    notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct BenchmarkArtifactManifest {
    schema_version: String,
    matrix_version: String,
    dataset_profile: String,
    dataset_version: String,
    dataset_sha256: String,
    comparator_count: usize,
    sample_count: usize,
    replay_command: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct BenchmarkObservation {
    metric: ComparatorMetric,
    measured_value: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RegressionViolation {
    metric: ComparatorMetric,
    baseline: u64,
    measured: u64,
    regression_pct_x100: u64,
    threshold_pct_x100: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(clippy::struct_field_names)]
struct ArtifactBundle {
    manifest_path: PathBuf,
    matrix_path: PathBuf,
    samples_path: PathBuf,
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/benchmark_golden")
}

fn build_baseline_matrix(dataset_profile: &str) -> BenchmarkMatrix {
    BenchmarkMatrix {
        schema_version: ARTIFACT_SCHEMA_VERSION.to_owned(),
        matrix_version: MATRIX_VERSION.to_owned(),
        cases: vec![
            BenchmarkCase {
                path: BenchmarkPath::Crawl,
                dataset_profile: dataset_profile.to_owned(),
                warmup_iterations: 3,
                measured_iterations: 30,
            },
            BenchmarkCase {
                path: BenchmarkPath::Index,
                dataset_profile: dataset_profile.to_owned(),
                warmup_iterations: 3,
                measured_iterations: 25,
            },
            BenchmarkCase {
                path: BenchmarkPath::Query,
                dataset_profile: dataset_profile.to_owned(),
                warmup_iterations: 5,
                measured_iterations: 50,
            },
            BenchmarkCase {
                path: BenchmarkPath::Tui,
                dataset_profile: dataset_profile.to_owned(),
                warmup_iterations: 5,
                measured_iterations: 60,
            },
        ],
        comparators: vec![
            ComparatorDefinition {
                path: BenchmarkPath::Index,
                metric: ComparatorMetric::IndexingThroughputDocsPerSecond,
                baseline_key: "baseline_metrics.indexing_throughput_docs_per_second".to_owned(),
                max_regression_pct: MAX_ALLOWED_REGRESSION_PCT,
            },
            ComparatorDefinition {
                path: BenchmarkPath::Query,
                metric: ComparatorMetric::SearchLatencyP50Ms,
                baseline_key: "baseline_metrics.search_latency_ms.p50".to_owned(),
                max_regression_pct: MAX_ALLOWED_REGRESSION_PCT,
            },
            ComparatorDefinition {
                path: BenchmarkPath::Query,
                metric: ComparatorMetric::SearchLatencyP95Ms,
                baseline_key: "baseline_metrics.search_latency_ms.p95".to_owned(),
                max_regression_pct: MAX_ALLOWED_REGRESSION_PCT,
            },
            ComparatorDefinition {
                path: BenchmarkPath::Query,
                metric: ComparatorMetric::SearchLatencyP99Ms,
                baseline_key: "baseline_metrics.search_latency_ms.p99".to_owned(),
                max_regression_pct: MAX_ALLOWED_REGRESSION_PCT,
            },
            ComparatorDefinition {
                path: BenchmarkPath::Query,
                metric: ComparatorMetric::FastTierLatencyMs,
                baseline_key: "baseline_metrics.fast_tier_latency_ms".to_owned(),
                max_regression_pct: MAX_ALLOWED_REGRESSION_PCT,
            },
            ComparatorDefinition {
                path: BenchmarkPath::Query,
                metric: ComparatorMetric::QualityTierLatencyMs,
                baseline_key: "baseline_metrics.quality_tier_latency_ms".to_owned(),
                max_regression_pct: MAX_ALLOWED_REGRESSION_PCT,
            },
            ComparatorDefinition {
                path: BenchmarkPath::Index,
                metric: ComparatorMetric::IndexingPeakMemoryMb,
                baseline_key: "baseline_metrics.indexing_peak_memory_mb".to_owned(),
                max_regression_pct: MAX_ALLOWED_REGRESSION_PCT,
            },
            ComparatorDefinition {
                path: BenchmarkPath::Query,
                metric: ComparatorMetric::SearchingPeakMemoryMb,
                baseline_key: "baseline_metrics.searching_peak_memory_mb".to_owned(),
                max_regression_pct: MAX_ALLOWED_REGRESSION_PCT,
            },
            ComparatorDefinition {
                path: BenchmarkPath::Index,
                metric: ComparatorMetric::IndexSizeBytesPerDocument,
                baseline_key: "baseline_metrics.index_size_bytes_per_document".to_owned(),
                max_regression_pct: MAX_ALLOWED_REGRESSION_PCT,
            },
        ],
    }
}

fn load_golden_dataset(profile: &str) -> GoldenDataset {
    let path = fixture_dir().join(format!("{profile}.json"));
    let raw = fs::read_to_string(&path).expect("read golden dataset fixture");
    serde_json::from_str::<GoldenDataset>(&raw).expect("parse golden dataset fixture")
}

fn baseline_value(dataset: &GoldenDataset, metric: ComparatorMetric) -> u64 {
    match metric {
        ComparatorMetric::IndexingThroughputDocsPerSecond => {
            dataset.baseline_metrics.indexing_throughput_docs_per_second
        }
        ComparatorMetric::SearchLatencyP50Ms => dataset.baseline_metrics.search_latency_ms.p50,
        ComparatorMetric::SearchLatencyP95Ms => dataset.baseline_metrics.search_latency_ms.p95,
        ComparatorMetric::SearchLatencyP99Ms => dataset.baseline_metrics.search_latency_ms.p99,
        ComparatorMetric::FastTierLatencyMs => dataset.baseline_metrics.fast_tier_latency_ms,
        ComparatorMetric::QualityTierLatencyMs => dataset.baseline_metrics.quality_tier_latency_ms,
        ComparatorMetric::IndexingPeakMemoryMb => dataset.baseline_metrics.indexing_peak_memory_mb,
        ComparatorMetric::SearchingPeakMemoryMb => dataset.baseline_metrics.searching_peak_memory_mb,
        ComparatorMetric::IndexSizeBytesPerDocument => {
            dataset.baseline_metrics.index_size_bytes_per_document
        }
    }
}

fn regression_pct_x100(metric: ComparatorMetric, baseline: u64, measured: u64) -> u64 {
    if baseline == 0 {
        return 0;
    }

    let regression_numerator = match metric {
        ComparatorMetric::IndexingThroughputDocsPerSecond => baseline.saturating_sub(measured),
        _ => measured.saturating_sub(baseline),
    };

    regression_numerator
        .saturating_mul(100)
        .saturating_mul(REGRESSION_SCALE)
        / baseline
}

fn evaluate_regressions(
    matrix: &BenchmarkMatrix,
    dataset: &GoldenDataset,
    observations: &[BenchmarkObservation],
) -> Vec<RegressionViolation> {
    matrix
        .comparators
        .iter()
        .filter_map(|comparator| {
            let observation = observations
                .iter()
                .find(|candidate| candidate.metric == comparator.metric)?;
            let baseline = baseline_value(dataset, comparator.metric);
            let regression_pct_x100 =
                regression_pct_x100(comparator.metric, baseline, observation.measured_value);
            let threshold_pct_x100 = u64::from(comparator.max_regression_pct) * REGRESSION_SCALE;

            (regression_pct_x100 > threshold_pct_x100).then_some(RegressionViolation {
                metric: comparator.metric,
                baseline,
                measured: observation.measured_value,
                regression_pct_x100,
                threshold_pct_x100,
            })
        })
        .collect()
}

fn sha256_hex_for_file(path: &Path) -> String {
    let output = Command::new("sha256sum")
        .arg(path)
        .output()
        .expect("invoke sha256sum");
    assert!(output.status.success(), "sha256sum command failed");
    let stdout = String::from_utf8(output.stdout).expect("decode sha256sum output");
    stdout
        .split_whitespace()
        .next()
        .expect("parse sha256sum digest")
        .to_owned()
}

fn sample_payload(case: &BenchmarkCase, dataset: &GoldenDataset) -> serde_json::Value {
    let baseline_snapshot = match case.path {
        BenchmarkPath::Crawl => serde_json::json!({
            "crawl_events": dataset.workload.crawl_events,
        }),
        BenchmarkPath::Index => serde_json::json!({
            "indexing_throughput_docs_per_second": dataset.baseline_metrics.indexing_throughput_docs_per_second,
            "indexing_peak_memory_mb": dataset.baseline_metrics.indexing_peak_memory_mb,
            "index_size_bytes_per_document": dataset.baseline_metrics.index_size_bytes_per_document,
        }),
        BenchmarkPath::Query => serde_json::json!({
            "search_latency_ms": dataset.baseline_metrics.search_latency_ms,
            "fast_tier_latency_ms": dataset.baseline_metrics.fast_tier_latency_ms,
            "quality_tier_latency_ms": dataset.baseline_metrics.quality_tier_latency_ms,
            "searching_peak_memory_mb": dataset.baseline_metrics.searching_peak_memory_mb,
        }),
        BenchmarkPath::Tui => serde_json::json!({
            "tui_interactions": dataset.workload.tui_interactions,
            "searching_peak_memory_mb": dataset.baseline_metrics.searching_peak_memory_mb,
        }),
    };

    serde_json::json!({
        "path": case.path,
        "dataset_profile": case.dataset_profile,
        "warmup_iterations": case.warmup_iterations,
        "measured_iterations": case.measured_iterations,
        "baseline_snapshot": baseline_snapshot,
    })
}

fn write_artifact_bundle(
    out_dir: &Path,
    matrix: &BenchmarkMatrix,
    dataset: &GoldenDataset,
) -> ArtifactBundle {
    fs::create_dir_all(out_dir).expect("create artifact output dir");

    let dataset_path = fixture_dir().join(format!("{}.json", dataset.profile));
    let dataset_sha256 = sha256_hex_for_file(&dataset_path);

    let manifest = BenchmarkArtifactManifest {
        schema_version: ARTIFACT_SCHEMA_VERSION.to_owned(),
        matrix_version: matrix.matrix_version.clone(),
        dataset_profile: dataset.profile.clone(),
        dataset_version: dataset.dataset_version.clone(),
        dataset_sha256,
        comparator_count: matrix.comparators.len(),
        sample_count: matrix.cases.len(),
        replay_command:
            "cargo test -p frankensearch-fsfs --test benchmark_baseline_matrix -- --nocapture"
                .to_string(),
    };

    let manifest_path = out_dir.join("benchmark_manifest.json");
    let matrix_path = out_dir.join("benchmark_matrix.json");
    let samples_path = out_dir.join("samples.jsonl");

    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).expect("serialize manifest"),
    )
    .expect("write manifest artifact");
    fs::write(
        &matrix_path,
        serde_json::to_vec_pretty(matrix).expect("serialize matrix"),
    )
    .expect("write matrix artifact");

    let mut samples = fs::File::create(&samples_path).expect("create samples artifact");
    for case in &matrix.cases {
        let payload = sample_payload(case, dataset);
        let encoded = serde_json::to_string(&payload).expect("encode sample line");
        writeln!(samples, "{encoded}").expect("write sample line");
    }

    ArtifactBundle {
        manifest_path,
        matrix_path,
        samples_path,
    }
}

#[test]
fn benchmark_matrix_covers_crawl_index_query_tui_paths() {
    let matrix = build_baseline_matrix("small");
    let covered: BTreeSet<_> = matrix.cases.iter().map(|case| case.path).collect();

    assert_eq!(matrix.schema_version, ARTIFACT_SCHEMA_VERSION);
    assert_eq!(matrix.matrix_version, MATRIX_VERSION);
    assert_eq!(matrix.cases.len(), 4);
    assert_eq!(matrix.comparators.len(), 9);

    let expected = BTreeSet::from([
        BenchmarkPath::Crawl,
        BenchmarkPath::Index,
        BenchmarkPath::Query,
        BenchmarkPath::Tui,
    ]);
    assert_eq!(covered, expected);
}

#[test]
fn benchmark_matrix_declares_required_regression_metrics() {
    let matrix = build_baseline_matrix("small");
    let actual_metrics: BTreeSet<_> = matrix
        .comparators
        .iter()
        .map(|comparator| comparator.metric)
        .collect();
    let expected_metrics = BTreeSet::from([
        ComparatorMetric::IndexingThroughputDocsPerSecond,
        ComparatorMetric::SearchLatencyP50Ms,
        ComparatorMetric::SearchLatencyP95Ms,
        ComparatorMetric::SearchLatencyP99Ms,
        ComparatorMetric::FastTierLatencyMs,
        ComparatorMetric::QualityTierLatencyMs,
        ComparatorMetric::IndexingPeakMemoryMb,
        ComparatorMetric::SearchingPeakMemoryMb,
        ComparatorMetric::IndexSizeBytesPerDocument,
    ]);

    assert_eq!(actual_metrics, expected_metrics);
    assert!(matrix
        .comparators
        .iter()
        .all(|comparator| comparator.max_regression_pct == MAX_ALLOWED_REGRESSION_PCT));
}

#[test]
fn regression_detector_enforces_twenty_percent_budget() {
    let matrix = build_baseline_matrix("small");
    let dataset = load_golden_dataset("small");
    let observations = vec![
        BenchmarkObservation {
            metric: ComparatorMetric::IndexingThroughputDocsPerSecond,
            measured_value: 280,
        },
        BenchmarkObservation {
            metric: ComparatorMetric::SearchLatencyP95Ms,
            measured_value: 31,
        },
        BenchmarkObservation {
            metric: ComparatorMetric::SearchingPeakMemoryMb,
            measured_value: 330,
        },
    ];

    let regressions = evaluate_regressions(&matrix, &dataset, &observations);
    let regressed_metrics: BTreeSet<_> = regressions
        .iter()
        .map(|violation| violation.metric)
        .collect();

    assert_eq!(regressions.len(), 2);
    assert_eq!(
        regressed_metrics,
        BTreeSet::from([
            ComparatorMetric::IndexingThroughputDocsPerSecond,
            ComparatorMetric::SearchingPeakMemoryMb,
        ])
    );

    let throughput = regressions
        .iter()
        .find(|violation| {
            violation.metric == ComparatorMetric::IndexingThroughputDocsPerSecond
        })
        .expect("throughput regression is present");
    assert!(throughput.regression_pct_x100 > throughput.threshold_pct_x100);
}

#[test]
fn golden_datasets_are_versioned_and_reproducible() {
    let expected_hashes = [
        (
            "tiny",
            "b66af1741fa5c4400c3aaabbcef2b74624f4348e0e02b9ca0d452a56b38ec267",
        ),
        (
            "small",
            "7456c946763118d90ad7749d615b50bb00f478dff443d389b25e993e4c2adc95",
        ),
        (
            "medium",
            "d5aa0568cd849a377a8980c4206bc5d6b63b0b3aa3b8af6d8b37f4278dea8d64",
        ),
    ];

    for (profile, expected_hash) in expected_hashes {
        let fixture_path = fixture_dir().join(format!("{profile}.json"));
        let raw = fs::read(&fixture_path).expect("read fixture bytes");
        let digest = sha256_hex_for_file(&fixture_path);

        let dataset: GoldenDataset =
            serde_json::from_slice(&raw).expect("parse golden dataset fixture");
        assert_eq!(dataset.schema_version, GOLDEN_SCHEMA_VERSION);
        assert_eq!(dataset.dataset_version, "2026-02-14");
        assert_eq!(dataset.profile, profile);
        assert_eq!(digest, expected_hash);

        assert!(dataset.baseline_metrics.indexing_throughput_docs_per_second > 0);
        assert!(dataset.baseline_metrics.fast_tier_latency_ms > 0);
        assert!(dataset.baseline_metrics.quality_tier_latency_ms > 0);
        assert!(dataset.baseline_metrics.indexing_peak_memory_mb > 0);
        assert!(dataset.baseline_metrics.searching_peak_memory_mb > 0);
        assert!(dataset.baseline_metrics.index_size_bytes_per_document > 0);
        assert!(
            dataset.baseline_metrics.search_latency_ms.p50
                <= dataset.baseline_metrics.search_latency_ms.p95
        );
        assert!(
            dataset.baseline_metrics.search_latency_ms.p95
                <= dataset.baseline_metrics.search_latency_ms.p99
        );
        assert!(
            dataset.baseline_metrics.fast_tier_latency_ms
                <= dataset.baseline_metrics.quality_tier_latency_ms
        );
    }

    assert_eq!(GOLDEN_PROFILES.len(), 3);
}

#[test]
fn artifact_capture_supports_later_statistical_comparison() {
    let matrix = build_baseline_matrix("small");
    let dataset = load_golden_dataset("small");
    let temp = TempDir::new().expect("create temp dir");

    let bundle = write_artifact_bundle(temp.path(), &matrix, &dataset);

    assert!(bundle.manifest_path.exists());
    assert!(bundle.matrix_path.exists());
    assert!(bundle.samples_path.exists());

    let manifest_raw = fs::read_to_string(&bundle.manifest_path).expect("read manifest");
    let manifest: BenchmarkArtifactManifest =
        serde_json::from_str(&manifest_raw).expect("parse manifest");

    assert_eq!(manifest.schema_version, ARTIFACT_SCHEMA_VERSION);
    assert_eq!(manifest.matrix_version, MATRIX_VERSION);
    assert_eq!(manifest.dataset_profile, "small");
    assert_eq!(manifest.dataset_version, "2026-02-14");
    assert_eq!(manifest.comparator_count, 9);
    assert_eq!(manifest.sample_count, 4);
    assert!(
        manifest
            .replay_command
            .contains("benchmark_baseline_matrix")
    );

    let sample_lines = fs::read_to_string(&bundle.samples_path).expect("read samples");
    let non_empty_count = sample_lines.lines().filter(|line| !line.is_empty()).count();
    assert_eq!(non_empty_count, 4);
}
