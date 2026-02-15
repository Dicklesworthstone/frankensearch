use std::collections::BTreeSet;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use frankensearch_core::metrics_eval::{
    BootstrapComparison, RunStabilityVerdict, bootstrap_compare, trim_outliers,
    verify_run_stability,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

const GOLDEN_SCHEMA_VERSION: &str = "fsfs-benchmark-golden-v1";
const ARTIFACT_SCHEMA_VERSION: &str = "fsfs-benchmark-artifact-v1";
const MATRIX_VERSION: &str = "fsfs-benchmark-matrix-v1";
const GOLDEN_PROFILES: [&str; 3] = ["tiny", "small", "medium"];
const MAX_ALLOWED_REGRESSION_PCT: u16 = 20;
const REGRESSION_SCALE: u64 = 100;

/// Default bootstrap parameters for statistical regression detection.
const BOOTSTRAP_CONFIDENCE: f64 = 0.95;
const BOOTSTRAP_RESAMPLES: usize = 2000;
const BOOTSTRAP_SEED: u64 = 0xBE0C_5EED;

/// Run-stability pre-gate parameters (bd-2vig).
/// Maximum coefficient of variation allowed before rejecting a benchmark run.
const STABILITY_MAX_CV: f64 = 0.15;
/// Minimum sample count required after outlier trimming.
const STABILITY_MIN_SAMPLES: usize = 5;
/// IQR factor for outlier detection/trimming before bootstrap comparison.
const OUTLIER_IQR_FACTOR: f64 = 1.5;

#[inline]
fn metric_u64_to_f64(value: u64) -> f64 {
    f64::from(u32::try_from(value).expect("benchmark fixture values must fit in u32"))
}

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
    matrix_sha256: String,
    samples_sha256: String,
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

/// Per-iteration sample set for a single metric, enabling bootstrap comparison.
///
/// When `iterations` contains multiple measurements from repeated benchmark runs,
/// `bootstrap_compare` can determine whether the difference from the baseline
/// distribution is statistically significant rather than relying on a fixed threshold.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct BenchmarkSampleSet {
    metric: ComparatorMetric,
    /// Per-iteration measured values (e.g., latency per query run).
    iterations: Vec<f64>,
}

/// Result of a statistical regression check using bootstrap paired comparison.
#[derive(Debug, Clone)]
struct StatisticalRegressionResult {
    metric: ComparatorMetric,
    comparison: BootstrapComparison,
    /// True when the difference is both statistically significant AND
    /// in the regression direction (worse performance).
    is_regression: bool,
    /// Run-stability verdict for the sample set (bd-2vig).
    /// `None` when the stability pre-gate was not applied.
    stability: Option<RunStabilityVerdict>,
    /// Number of outliers trimmed before bootstrap comparison (bd-2vig).
    outliers_trimmed: usize,
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

const fn baseline_value(dataset: &GoldenDataset, metric: ComparatorMetric) -> u64 {
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
        ComparatorMetric::SearchingPeakMemoryMb => {
            dataset.baseline_metrics.searching_peak_memory_mb
        }
        ComparatorMetric::IndexSizeBytesPerDocument => {
            dataset.baseline_metrics.index_size_bytes_per_document
        }
    }
}

const fn regression_pct_x100(metric: ComparatorMetric, baseline: u64, measured: u64) -> u64 {
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

/// Evaluate regressions using bootstrap paired comparison.
///
/// For each metric in `sample_sets`, constructs a synthetic baseline distribution
/// from the golden dataset value (repeated to match sample count) and compares
/// against the measured iterations using `bootstrap_compare`.
///
/// A regression is flagged when the difference is both statistically significant
/// (p < alpha) AND in the regression direction:
/// - For throughput metrics: measured < baseline (lower is worse)
/// - For latency/memory metrics: measured > baseline (higher is worse)
fn evaluate_regressions_statistical(
    dataset: &GoldenDataset,
    sample_sets: &[BenchmarkSampleSet],
) -> Vec<StatisticalRegressionResult> {
    evaluate_regressions_statistical_inner(dataset, sample_sets, false)
}

/// Evaluate regressions with run-stability pre-gate and outlier trimming (bd-2vig).
///
/// Before bootstrap comparison:
/// 1. **Outlier trimming**: removes IQR outliers from measurement iterations,
///    producing cleaner data for the bootstrap comparison.
/// 2. **Stability pre-gate**: verifies that trimmed measurements have acceptable
///    CV and sufficient sample count. Unstable runs are still reported but
///    `is_regression` is forced to `false` (unreliable data cannot confirm regression).
fn evaluate_regressions_statistical_gated(
    dataset: &GoldenDataset,
    sample_sets: &[BenchmarkSampleSet],
) -> Vec<StatisticalRegressionResult> {
    evaluate_regressions_statistical_inner(dataset, sample_sets, true)
}

fn evaluate_regressions_statistical_inner(
    dataset: &GoldenDataset,
    sample_sets: &[BenchmarkSampleSet],
    apply_stability_gate: bool,
) -> Vec<StatisticalRegressionResult> {
    sample_sets
        .iter()
        .filter_map(|sample_set| {
            if sample_set.iterations.is_empty() {
                return None;
            }

            // bd-2vig: optionally trim outliers and check stability before comparison.
            let (effective_iterations, stability, outliers_trimmed) = if apply_stability_gate {
                let trimmed = trim_outliers(&sample_set.iterations, OUTLIER_IQR_FACTOR);
                let outlier_count = sample_set.iterations.len().saturating_sub(trimmed.len());
                if trimmed.is_empty() {
                    // Preserve this metric in regression output instead of silently
                    // dropping it when trimming removes all samples.
                    let verdict = RunStabilityVerdict {
                        stable: false,
                        cv: None,
                        effective_sample_count: 0,
                        outlier_count,
                        reason: format!(
                            "all samples trimmed as outliers: 0 remaining from {}",
                            sample_set.iterations.len()
                        ),
                    };
                    (sample_set.iterations.clone(), Some(verdict), outlier_count)
                } else {
                    let verdict =
                        verify_run_stability(&trimmed, STABILITY_MAX_CV, STABILITY_MIN_SAMPLES);
                    (trimmed, Some(verdict), outlier_count)
                }
            } else {
                (sample_set.iterations.clone(), None, 0)
            };

            if effective_iterations.is_empty() {
                return None;
            }

            let baseline_scalar = baseline_value(dataset, sample_set.metric);
            let n = effective_iterations.len();
            let baseline_samples: Vec<f64> = vec![metric_u64_to_f64(baseline_scalar); n];

            let comparison = bootstrap_compare(
                &effective_iterations,
                &baseline_samples,
                BOOTSTRAP_CONFIDENCE,
                BOOTSTRAP_RESAMPLES,
                BOOTSTRAP_SEED,
            )?;

            // Determine regression direction:
            // - Throughput: regression when measured < baseline (mean_diff < 0)
            // - Latency/memory/size: regression when measured > baseline (mean_diff > 0)
            let is_regression_direction = match sample_set.metric {
                ComparatorMetric::IndexingThroughputDocsPerSecond => comparison.mean_diff < 0.0,
                _ => comparison.mean_diff > 0.0,
            };

            // bd-2vig: if stability gate is active and run is unstable,
            // suppress regression flag — noisy data cannot confirm a regression.
            let run_is_stable = stability.as_ref().is_none_or(|v| v.stable);
            let is_regression = comparison.significant && is_regression_direction && run_is_stable;

            Some(StatisticalRegressionResult {
                metric: sample_set.metric,
                comparison,
                is_regression,
                stability,
                outliers_trimmed,
            })
        })
        .collect()
}

fn sha256_hex_for_file(path: &Path) -> String {
    let mut file = fs::File::open(path).expect("open file for sha256");
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 8192];

    loop {
        let read = file.read(&mut buffer).expect("read file for sha256");
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }

    format!("{:x}", hasher.finalize())
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

    let manifest_path = out_dir.join("benchmark_manifest.json");
    let matrix_path = out_dir.join("benchmark_matrix.json");
    let samples_path = out_dir.join("samples.jsonl");

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

    let manifest = BenchmarkArtifactManifest {
        schema_version: ARTIFACT_SCHEMA_VERSION.to_owned(),
        matrix_version: matrix.matrix_version.clone(),
        dataset_profile: dataset.profile.clone(),
        dataset_version: dataset.dataset_version.clone(),
        dataset_sha256,
        matrix_sha256: sha256_hex_for_file(&matrix_path),
        samples_sha256: sha256_hex_for_file(&samples_path),
        comparator_count: matrix.comparators.len(),
        sample_count: matrix.cases.len(),
        replay_command:
            "cargo test -p frankensearch-fsfs --test benchmark_baseline_matrix -- --nocapture"
                .to_string(),
    };

    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).expect("serialize manifest"),
    )
    .expect("write manifest artifact");

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
    assert!(
        matrix
            .comparators
            .iter()
            .all(|comparator| comparator.max_regression_pct == MAX_ALLOWED_REGRESSION_PCT)
    );
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
        .find(|violation| violation.metric == ComparatorMetric::IndexingThroughputDocsPerSecond)
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
    assert_eq!(
        manifest.matrix_sha256,
        sha256_hex_for_file(&bundle.matrix_path),
        "manifest must carry deterministic matrix hash"
    );
    assert_eq!(
        manifest.samples_sha256,
        sha256_hex_for_file(&bundle.samples_path),
        "manifest must carry deterministic sample bundle hash"
    );
    assert!(
        manifest
            .replay_command
            .contains("benchmark_baseline_matrix")
    );

    let sample_lines = fs::read_to_string(&bundle.samples_path).expect("read samples");
    let non_empty_count = sample_lines.lines().filter(|line| !line.is_empty()).count();
    assert_eq!(non_empty_count, 4);
}

#[test]
fn artifact_hashes_are_stable_across_repeated_bundle_generation() {
    let matrix = build_baseline_matrix("small");
    let dataset = load_golden_dataset("small");
    let temp_a = TempDir::new().expect("create temp dir A");
    let temp_b = TempDir::new().expect("create temp dir B");

    let bundle_a = write_artifact_bundle(temp_a.path(), &matrix, &dataset);
    let bundle_b = write_artifact_bundle(temp_b.path(), &matrix, &dataset);

    let first_manifest_json = fs::read_to_string(&bundle_a.manifest_path).expect("read manifest A");
    let second_manifest_json =
        fs::read_to_string(&bundle_b.manifest_path).expect("read manifest B");
    let manifest_a: BenchmarkArtifactManifest =
        serde_json::from_str(&first_manifest_json).expect("parse manifest A");
    let manifest_b: BenchmarkArtifactManifest =
        serde_json::from_str(&second_manifest_json).expect("parse manifest B");

    assert_eq!(manifest_a.dataset_sha256, manifest_b.dataset_sha256);
    assert_eq!(manifest_a.matrix_sha256, manifest_b.matrix_sha256);
    assert_eq!(manifest_a.samples_sha256, manifest_b.samples_sha256);
    assert_eq!(
        manifest_a.matrix_sha256,
        sha256_hex_for_file(&bundle_a.matrix_path)
    );
    assert_eq!(
        manifest_a.samples_sha256,
        sha256_hex_for_file(&bundle_a.samples_path)
    );
    assert_eq!(
        manifest_b.matrix_sha256,
        sha256_hex_for_file(&bundle_b.matrix_path)
    );
    assert_eq!(
        manifest_b.samples_sha256,
        sha256_hex_for_file(&bundle_b.samples_path)
    );
}

// ── Bootstrap statistical regression detection (bd-2hz.9.8) ───────

#[test]
fn statistical_regression_detects_significant_latency_increase() {
    let dataset = load_golden_dataset("small");
    // Baseline p95 is 25ms. Simulate measured iterations significantly higher.
    let sample_sets = vec![BenchmarkSampleSet {
        metric: ComparatorMetric::SearchLatencyP95Ms,
        iterations: vec![35.0, 38.0, 32.0, 36.0, 34.0, 37.0, 33.0, 35.0, 39.0, 36.0],
    }];

    let results = evaluate_regressions_statistical(&dataset, &sample_sets);
    assert_eq!(results.len(), 1);
    let result = &results[0];
    assert!(
        result.is_regression,
        "significant latency increase should be flagged as regression"
    );
    assert!(
        result.comparison.significant,
        "difference should be statistically significant"
    );
    assert!(
        result.comparison.mean_diff > 0.0,
        "measured latency should be higher than baseline"
    );
}

#[test]
fn statistical_regression_not_flagged_for_stable_throughput() {
    let dataset = load_golden_dataset("small");
    // Baseline throughput is 350 docs/s. Simulate stable measurements.
    let base = metric_u64_to_f64(dataset.baseline_metrics.indexing_throughput_docs_per_second);
    let sample_sets = vec![BenchmarkSampleSet {
        metric: ComparatorMetric::IndexingThroughputDocsPerSecond,
        iterations: vec![
            base,
            base + 5.0,
            base - 3.0,
            base + 2.0,
            base - 1.0,
            base + 4.0,
            base - 2.0,
            base + 1.0,
        ],
    }];

    let results = evaluate_regressions_statistical(&dataset, &sample_sets);
    assert_eq!(results.len(), 1);
    assert!(
        !results[0].is_regression,
        "stable throughput near baseline should not be flagged as regression"
    );
}

#[test]
fn statistical_regression_detects_throughput_drop() {
    let dataset = load_golden_dataset("small");
    // Baseline throughput is 350 docs/s. Simulate a significant drop.
    let sample_sets = vec![BenchmarkSampleSet {
        metric: ComparatorMetric::IndexingThroughputDocsPerSecond,
        iterations: vec![200.0, 210.0, 195.0, 205.0, 198.0, 208.0, 202.0, 190.0],
    }];

    let results = evaluate_regressions_statistical(&dataset, &sample_sets);
    assert_eq!(results.len(), 1);
    let result = &results[0];
    assert!(
        result.is_regression,
        "significant throughput drop should be flagged as regression"
    );
    assert!(
        result.comparison.mean_diff < 0.0,
        "measured throughput should be lower than baseline (negative diff)"
    );
}

#[test]
fn statistical_regression_skips_empty_sample_sets() {
    let dataset = load_golden_dataset("small");
    let sample_sets = vec![BenchmarkSampleSet {
        metric: ComparatorMetric::SearchLatencyP50Ms,
        iterations: Vec::new(),
    }];

    let results = evaluate_regressions_statistical(&dataset, &sample_sets);
    assert!(
        results.is_empty(),
        "empty sample set should produce no results"
    );
}

#[test]
fn statistical_regression_handles_multiple_metrics_independently() {
    let dataset = load_golden_dataset("small");
    let mem_base = metric_u64_to_f64(dataset.baseline_metrics.searching_peak_memory_mb);
    let sample_sets = vec![
        // Latency regressed: measured >> baseline
        BenchmarkSampleSet {
            metric: ComparatorMetric::SearchLatencyP99Ms,
            iterations: vec![80.0, 85.0, 78.0, 82.0, 84.0, 79.0, 83.0, 81.0],
        },
        // Memory stable: measured ≈ baseline
        BenchmarkSampleSet {
            metric: ComparatorMetric::SearchingPeakMemoryMb,
            iterations: vec![
                mem_base,
                mem_base + 1.0,
                mem_base - 1.0,
                mem_base + 2.0,
                mem_base - 2.0,
                mem_base,
                mem_base + 1.0,
                mem_base - 1.0,
            ],
        },
    ];

    let results = evaluate_regressions_statistical(&dataset, &sample_sets);
    assert_eq!(results.len(), 2);

    let latency = results
        .iter()
        .find(|r| r.metric == ComparatorMetric::SearchLatencyP99Ms)
        .expect("p99 latency result should be present");
    assert!(latency.is_regression, "p99 latency should show regression");

    let memory = results
        .iter()
        .find(|r| r.metric == ComparatorMetric::SearchingPeakMemoryMb)
        .expect("memory result should be present");
    assert!(
        !memory.is_regression,
        "stable memory should not show regression"
    );
}

#[test]
fn statistical_regression_result_contains_bootstrap_ci_bounds() {
    let dataset = load_golden_dataset("small");
    let sample_sets = vec![BenchmarkSampleSet {
        metric: ComparatorMetric::FastTierLatencyMs,
        iterations: vec![2.0, 2.1, 1.9, 2.0, 2.2, 1.8, 2.1, 2.0],
    }];

    let results = evaluate_regressions_statistical(&dataset, &sample_sets);
    assert_eq!(results.len(), 1);
    let result = &results[0];

    // CI bounds should be well-formed
    assert!(
        result.comparison.ci_lower <= result.comparison.ci_upper,
        "CI lower should be <= CI upper"
    );
    assert!(
        (result.comparison.confidence - BOOTSTRAP_CONFIDENCE).abs() < f64::EPSILON,
        "confidence should match configured value"
    );
    assert_eq!(result.comparison.n_resamples, BOOTSTRAP_RESAMPLES);
    assert!(
        result.comparison.p_value > 0.0,
        "p-value should be positive (plus-one correction)"
    );
    assert!(
        result.comparison.p_value <= 1.0,
        "p-value should not exceed 1.0"
    );
}

#[test]
fn sample_set_serde_roundtrip() {
    let sample_set = BenchmarkSampleSet {
        metric: ComparatorMetric::SearchLatencyP95Ms,
        iterations: vec![25.0, 26.0, 24.5, 25.5, 27.0],
    };

    let json = serde_json::to_string(&sample_set).expect("serialize sample set");
    let back: BenchmarkSampleSet = serde_json::from_str(&json).expect("deserialize sample set");
    assert_eq!(sample_set, back);
}

// ── Run-stability pre-gate and outlier trimming (bd-2vig) ──────────

#[test]
fn gated_regression_trims_outliers_before_comparison() {
    let dataset = load_golden_dataset("small");
    // Baseline p95 is 25ms. Inject a single extreme outlier (500ms) among
    // otherwise-stable measurements near baseline. Without trimming, the
    // outlier inflates the mean and could cause a false regression signal.
    let sample_sets = vec![BenchmarkSampleSet {
        metric: ComparatorMetric::SearchLatencyP95Ms,
        iterations: vec![26.0, 25.0, 24.0, 25.5, 500.0, 26.5, 24.5, 25.0, 25.5, 26.0],
    }];

    let results = evaluate_regressions_statistical_gated(&dataset, &sample_sets);
    assert_eq!(results.len(), 1);
    let result = &results[0];

    assert!(
        result.outliers_trimmed > 0,
        "the 500ms outlier should be trimmed"
    );
    assert!(
        !result.is_regression,
        "stable measurements near baseline should not regress after outlier trimming"
    );
    assert!(
        result.stability.as_ref().is_some_and(|v| v.stable),
        "trimmed run should be stable"
    );
}

#[test]
fn gated_regression_suppresses_flag_when_run_is_unstable() {
    let dataset = load_golden_dataset("small");
    // Highly noisy measurements with CV >> 15%. Even though the mean is
    // higher than baseline (suggesting regression), the instability gate
    // should suppress the regression flag.
    let sample_sets = vec![BenchmarkSampleSet {
        metric: ComparatorMetric::SearchLatencyP95Ms,
        iterations: vec![10.0, 80.0, 15.0, 75.0, 12.0, 85.0, 20.0, 70.0],
    }];

    let results = evaluate_regressions_statistical_gated(&dataset, &sample_sets);
    assert_eq!(results.len(), 1);
    let result = &results[0];

    let stability = result
        .stability
        .as_ref()
        .expect("stability verdict should be present");
    assert!(
        !stability.stable,
        "highly variable measurements should fail stability check"
    );
    assert!(
        !result.is_regression,
        "unstable runs should not flag regressions (unreliable data)"
    );
}

#[test]
fn gated_regression_still_detects_genuine_regression_when_stable() {
    let dataset = load_golden_dataset("small");
    // Consistent measurements significantly above baseline (25ms → ~40ms).
    // Low noise, no outliers — should pass stability gate AND flag regression.
    let sample_sets = vec![BenchmarkSampleSet {
        metric: ComparatorMetric::SearchLatencyP95Ms,
        iterations: vec![39.0, 41.0, 40.0, 38.0, 42.0, 40.5, 39.5, 41.5],
    }];

    let results = evaluate_regressions_statistical_gated(&dataset, &sample_sets);
    assert_eq!(results.len(), 1);
    let result = &results[0];

    let stability = result
        .stability
        .as_ref()
        .expect("stability verdict should be present");
    assert!(stability.stable, "consistent measurements should be stable");
    assert!(
        result.is_regression,
        "genuine regression with stable data should still be flagged"
    );
    assert_eq!(
        result.outliers_trimmed, 0,
        "clean data should have no outliers trimmed"
    );
}

#[test]
fn gated_regression_ungated_path_has_no_stability_or_outlier_info() {
    let dataset = load_golden_dataset("small");
    let sample_sets = vec![BenchmarkSampleSet {
        metric: ComparatorMetric::SearchLatencyP95Ms,
        iterations: vec![35.0, 38.0, 32.0, 36.0, 34.0, 37.0, 33.0, 35.0],
    }];

    let results = evaluate_regressions_statistical(&dataset, &sample_sets);
    assert_eq!(results.len(), 1);
    let result = &results[0];

    assert!(
        result.stability.is_none(),
        "ungated path should not produce stability verdict"
    );
    assert_eq!(
        result.outliers_trimmed, 0,
        "ungated path should not trim outliers"
    );
}

#[test]
fn gated_regression_throughput_drop_with_outlier_trimming() {
    let dataset = load_golden_dataset("small");
    // Baseline throughput is 350 docs/s. Inject one high outlier (1000) that
    // would mask a real throughput drop if not trimmed.
    let sample_sets = vec![BenchmarkSampleSet {
        metric: ComparatorMetric::IndexingThroughputDocsPerSecond,
        iterations: vec![
            200.0, 210.0, 195.0, 205.0, 1000.0, 198.0, 208.0, 202.0, 190.0, 195.0,
        ],
    }];

    let results = evaluate_regressions_statistical_gated(&dataset, &sample_sets);
    assert_eq!(results.len(), 1);
    let result = &results[0];

    assert!(
        result.outliers_trimmed > 0,
        "1000 docs/s outlier should be trimmed"
    );
    assert!(
        result.is_regression,
        "throughput drop should be flagged after outlier removal"
    );
    assert!(
        result.comparison.mean_diff < 0.0,
        "measured throughput (after trimming) should be lower than baseline"
    );
}

#[test]
fn gated_regression_insufficient_samples_after_trimming() {
    let dataset = load_golden_dataset("small");
    // Only 4 samples, and after IQR trimming (which needs >=4 to apply),
    // if outliers are removed we might drop below STABILITY_MIN_SAMPLES.
    // Use wildly different values to force heavy outlier removal.
    let sample_sets = vec![BenchmarkSampleSet {
        metric: ComparatorMetric::SearchLatencyP50Ms,
        iterations: vec![10.0, 10.0, 500.0, 500.0],
    }];

    let results = evaluate_regressions_statistical_gated(&dataset, &sample_sets);
    assert_eq!(results.len(), 1, "gated path should report this metric");
    let result = &results[0];
    let stability = result
        .stability
        .as_ref()
        .expect("stability verdict should be present");
    assert!(
        !stability.stable,
        "insufficient stable samples should fail stability gate"
    );
    assert!(
        !result.is_regression,
        "unstable run should not flag regression"
    );
}
