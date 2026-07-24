//! Machine-readable performance-matrix contracts for Quill QG-1 through QG-10.
//!
//! The Criterion entry point owns engine execution. This module owns the
//! deterministic matrix, statistics, artifact schema, RSS probe, and human
//! rendering so the evidence format is unit-tested without running a benchmark.

use std::collections::BTreeSet;
use std::fmt::{self, Write as _};
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::GauntletError;

/// Version of the JSON emitted by the QG matrix harness.
pub const PERF_ARTIFACT_SCHEMA_VERSION: &str = "quill-perf-artifact-v2";
/// Minimum independent samples required by the standing statistical law.
pub const PERF_MIN_RUNS: usize = 10;
/// Maximum coefficient of variation admitted by an activated gate.
pub const PERF_MAX_CV_PCT: f64 = 5.0;
/// Oracle writer heap pinned for all same-binary comparisons (50 MiB).
pub const PERF_WRITER_HEAP_BYTES: usize = 50_000_000;
/// Tantivy's pinned minimum arena per writer thread. Multi-thread cells raise
/// both engines' equal total budget rather than silently reducing thread count.
pub const PERF_MIN_WRITER_HEAP_PER_THREAD_BYTES: usize = 15_000_000;

/// Equal total heap budget for one thread-count cell.
#[must_use]
pub const fn perf_writer_heap_bytes(threads: usize) -> usize {
    let per_thread = PERF_MIN_WRITER_HEAP_PER_THREAD_BYTES.saturating_mul(threads);
    if per_thread > PERF_WRITER_HEAP_BYTES {
        per_thread
    } else {
        PERF_WRITER_HEAP_BYTES
    }
}

/// One normative Quill performance gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PerfGate {
    Qg1,
    Qg2,
    Qg3,
    Qg4,
    Qg5,
    Qg6,
    Qg7,
    Qg8,
    Qg9,
    Qg10,
}

impl PerfGate {
    /// Gates in the normative manifest order.
    pub const ALL: [Self; 10] = [
        Self::Qg1,
        Self::Qg2,
        Self::Qg3,
        Self::Qg4,
        Self::Qg5,
        Self::Qg6,
        Self::Qg7,
        Self::Qg8,
        Self::Qg9,
        Self::Qg10,
    ];

    /// Stable manifest label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Qg1 => "QG-1",
            Self::Qg2 => "QG-2",
            Self::Qg3 => "QG-3",
            Self::Qg4 => "QG-4",
            Self::Qg5 => "QG-5",
            Self::Qg6 => "QG-6",
            Self::Qg7 => "QG-7",
            Self::Qg8 => "QG-8",
            Self::Qg9 => "QG-9",
            Self::Qg10 => "QG-10",
        }
    }
}

impl fmt::Display for PerfGate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.label())
    }
}

impl FromStr for PerfGate {
    type Err = GauntletError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let normalized = value.trim().to_ascii_uppercase().replace('_', "-");
        Self::ALL
            .into_iter()
            .find(|gate| gate.label() == normalized)
            .ok_or_else(|| GauntletError::InvalidCampaign {
                reason: format!("unknown Quill performance gate {value:?}"),
            })
    }
}

/// Pinned corpus sizes from the FSFS golden fixtures plus the E6 xlarge recipe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerfCorpus {
    Tiny,
    Small,
    Medium,
    Xlarge,
}

impl PerfCorpus {
    /// Number of documents in the committed profile.
    #[must_use]
    pub const fn document_count(self) -> u64 {
        match self {
            Self::Tiny => 500,
            Self::Small => 5_000,
            Self::Medium => 50_000,
            Self::Xlarge => 1_000_000,
        }
    }

    /// Stable fixture label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Tiny => "tiny",
            Self::Small => "small",
            Self::Medium => "medium",
            Self::Xlarge => "xlarge",
        }
    }
}

/// Whether text fields retain exact token positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PositionMode {
    On,
    Off,
}

impl PositionMode {
    #[must_use]
    pub const fn enabled(self) -> bool {
        matches!(self, Self::On)
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::On => "positions_on",
            Self::Off => "positions_off",
        }
    }
}

/// Visibility topology required by QG-3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerfTopology {
    InProcess,
    FreshProcess,
}

/// Query families pinned by QG-6.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerfQueryClass {
    Identifier,
    ShortKeyword,
    NaturalLanguage,
    Phrase,
    Boolean,
}

impl PerfQueryClass {
    pub const ALL: [Self; 5] = [
        Self::Identifier,
        Self::ShortKeyword,
        Self::NaturalLanguage,
        Self::Phrase,
        Self::Boolean,
    ];
}

/// One fully pinned matrix cell before it is measured.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerfCellSpec {
    pub gate: PerfGate,
    pub fixture: String,
    pub metric: String,
    pub corpus: Option<PerfCorpus>,
    pub document_count: Option<u64>,
    pub threads: Option<usize>,
    pub writer_heap_bytes: Option<usize>,
    pub positions: Option<PositionMode>,
    pub tombstone_density_pct: Option<u8>,
    pub query_class: Option<PerfQueryClass>,
    pub k: Option<usize>,
    pub topology: Option<PerfTopology>,
}

impl PerfCellSpec {
    fn new(gate: PerfGate, fixture: impl Into<String>, metric: impl Into<String>) -> Self {
        Self {
            gate,
            fixture: fixture.into(),
            metric: metric.into(),
            corpus: None,
            document_count: None,
            threads: None,
            writer_heap_bytes: None,
            positions: None,
            tombstone_density_pct: None,
            query_class: None,
            k: None,
            topology: None,
        }
    }
}

/// Complete, deterministic QG-1..QG-10 execution matrix.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerfMatrixSpec {
    pub manifest: String,
    pub cells: Vec<PerfCellSpec>,
}

impl PerfMatrixSpec {
    /// Build every normative cell. Runtime slice filters may select a gate or
    /// fixture, but they never redefine the matrix.
    #[must_use]
    pub fn complete() -> Self {
        let mut cells = Vec::new();
        let corpora = [
            PerfCorpus::Tiny,
            PerfCorpus::Small,
            PerfCorpus::Medium,
            PerfCorpus::Xlarge,
        ];
        for corpus in corpora {
            for threads in [1, 4, 8, 16] {
                for positions in [PositionMode::On, PositionMode::Off] {
                    let mut cell = PerfCellSpec::new(
                        PerfGate::Qg1,
                        format!("bulk/{}/{threads}/{}", corpus.label(), positions.label()),
                        "docs_per_second",
                    );
                    cell.corpus = Some(corpus);
                    cell.document_count = Some(corpus.document_count());
                    cell.threads = Some(threads);
                    cell.writer_heap_bytes = Some(perf_writer_heap_bytes(threads));
                    cell.positions = Some(positions);
                    cells.push(cell);
                }
            }
        }
        for corpus in [PerfCorpus::Medium, PerfCorpus::Xlarge] {
            let mut cell = PerfCellSpec::new(
                PerfGate::Qg1,
                format!("tokenize_only/{}", corpus.label()),
                "tokenize_docs_per_second",
            );
            cell.corpus = Some(corpus);
            cell.document_count = Some(corpus.document_count());
            cell.threads = Some(1);
            cell.writer_heap_bytes = Some(perf_writer_heap_bytes(1));
            cells.push(cell);
        }

        let mut single = PerfCellSpec::new(
            PerfGate::Qg2,
            "bulk/medium/1/positions_on",
            "docs_per_second",
        );
        single.corpus = Some(PerfCorpus::Medium);
        single.document_count = Some(PerfCorpus::Medium.document_count());
        single.threads = Some(1);
        single.writer_heap_bytes = Some(perf_writer_heap_bytes(1));
        single.positions = Some(PositionMode::On);
        cells.push(single);

        let mut initial =
            PerfCellSpec::new(PerfGate::Qg3, "watch/medium/initial", "docs_per_second");
        initial.corpus = Some(PerfCorpus::Medium);
        initial.document_count = Some(PerfCorpus::Medium.document_count());
        initial.threads = Some(1);
        initial.writer_heap_bytes = Some(perf_writer_heap_bytes(1));
        initial.positions = Some(PositionMode::On);
        cells.push(initial);

        for topology in [PerfTopology::InProcess, PerfTopology::FreshProcess] {
            for metric in ["updates_per_second", "update_to_searchable_ms"] {
                let mut cell = PerfCellSpec::new(
                    PerfGate::Qg3,
                    format!("watch/medium/5000/{topology:?}").to_ascii_lowercase(),
                    metric,
                );
                cell.corpus = Some(PerfCorpus::Medium);
                cell.document_count = Some(5_000);
                cell.threads = Some(1);
                cell.writer_heap_bytes = Some(perf_writer_heap_bytes(1));
                cell.positions = Some(PositionMode::On);
                cell.topology = Some(topology);
                cells.push(cell);
            }
        }

        let mut commit =
            PerfCellSpec::new(PerfGate::Qg4, "commit/100000/warm", "commit_latency_ms");
        commit.document_count = Some(100_000);
        commit.positions = Some(PositionMode::On);
        commit.threads = Some(1);
        commit.writer_heap_bytes = Some(perf_writer_heap_bytes(1));
        cells.push(commit);

        for density in [5, 20, 50] {
            let mut cell = PerfCellSpec::new(
                PerfGate::Qg5,
                format!("compaction/xlarge/{density}pct"),
                "wall_clock_ms",
            );
            cell.corpus = Some(PerfCorpus::Xlarge);
            cell.document_count = Some(PerfCorpus::Xlarge.document_count());
            cell.positions = Some(PositionMode::On);
            cell.threads = Some(1);
            cell.writer_heap_bytes = Some(perf_writer_heap_bytes(1));
            cell.tombstone_density_pct = Some(density);
            cells.push(cell);
        }

        for query_class in PerfQueryClass::ALL {
            for k in [10, 100] {
                for (label, document_count) in [("100k", 100_000), ("1m", 1_000_000)] {
                    let mut cell = PerfCellSpec::new(
                        PerfGate::Qg6,
                        format!("query/{query_class:?}/k{k}/{label}").to_ascii_lowercase(),
                        "latency_ms",
                    );
                    cell.document_count = Some(document_count);
                    cell.positions = Some(PositionMode::On);
                    cell.threads = Some(1);
                    cell.writer_heap_bytes = Some(perf_writer_heap_bytes(1));
                    cell.query_class = Some(query_class);
                    cell.k = Some(k);
                    cells.push(cell);
                }
            }
        }

        for corpus in [PerfCorpus::Medium, PerfCorpus::Xlarge] {
            for positions in [PositionMode::On, PositionMode::Off] {
                let mut rss = PerfCellSpec::new(
                    PerfGate::Qg7,
                    format!("memory/{}/{}", corpus.label(), positions.label()),
                    "peak_rss_bytes",
                );
                rss.corpus = Some(corpus);
                rss.document_count = Some(corpus.document_count());
                rss.threads = Some(8);
                rss.writer_heap_bytes = Some(perf_writer_heap_bytes(8));
                rss.positions = Some(positions);
                cells.push(rss);

                let mut bytes = PerfCellSpec::new(
                    PerfGate::Qg7,
                    format!("size/{}/{}", corpus.label(), positions.label()),
                    "index_bytes_per_document",
                );
                bytes.corpus = Some(corpus);
                bytes.document_count = Some(corpus.document_count());
                bytes.threads = Some(8);
                bytes.writer_heap_bytes = Some(perf_writer_heap_bytes(8));
                bytes.positions = Some(positions);
                cells.push(bytes);
            }
        }

        for threads in [1, 2, 4, 8, 16, 32] {
            let mut cell = PerfCellSpec::new(
                PerfGate::Qg8,
                format!("scaling/xlarge/{threads}/positions_on"),
                "docs_per_second",
            );
            cell.corpus = Some(PerfCorpus::Xlarge);
            cell.document_count = Some(PerfCorpus::Xlarge.document_count());
            cell.threads = Some(threads);
            cell.writer_heap_bytes = Some(perf_writer_heap_bytes(threads));
            cell.positions = Some(PositionMode::On);
            cells.push(cell);
        }

        let mut cold =
            PerfCellSpec::new(PerfGate::Qg9, "cold_open/xlarge/default", "open_latency_ms");
        cold.corpus = Some(PerfCorpus::Xlarge);
        cold.document_count = Some(PerfCorpus::Xlarge.document_count());
        cold.positions = Some(PositionMode::On);
        cold.threads = Some(1);
        cold.writer_heap_bytes = Some(perf_writer_heap_bytes(1));
        cells.push(cold);

        cells.push(PerfCellSpec::new(
            PerfGate::Qg10,
            "dependency_surface/default_lexical",
            "tantivy_nodes",
        ));

        Self {
            manifest: "docs/contracts/quill-perf-gates.toml".to_owned(),
            cells,
        }
    }

    /// Select an immutable matrix slice without changing its pins.
    #[must_use]
    pub fn for_gate(&self, gate: PerfGate) -> Vec<&PerfCellSpec> {
        self.cells.iter().filter(|cell| cell.gate == gate).collect()
    }
}

/// Distribution summary required for every timed cell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistributionSummary {
    pub value: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub mad: f64,
    pub cv_pct: f64,
    pub runs: usize,
}

impl DistributionSummary {
    /// Summarize finite non-negative samples. The result records fewer than ten
    /// runs but cannot be gate-activated until [`Self::stable_for_activation`]
    /// is true.
    ///
    /// # Errors
    ///
    /// Rejects an empty set, NaN/infinite values, and negative durations or
    /// counters.
    pub fn from_samples(samples: &[f64]) -> Result<Self, GauntletError> {
        if samples.is_empty()
            || samples
                .iter()
                .any(|sample| !sample.is_finite() || *sample < 0.0)
        {
            return Err(GauntletError::InvalidCampaign {
                reason: "performance samples must be finite, non-negative, and non-empty"
                    .to_owned(),
            });
        }
        let mut sorted = samples.to_vec();
        sorted.sort_unstable_by(f64::total_cmp);
        let p50 = percentile(&sorted, 0.50);
        let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
        let variance = sorted
            .iter()
            .map(|sample| {
                let delta = sample - mean;
                delta * delta
            })
            .sum::<f64>()
            / sorted.len() as f64;
        let cv_pct = if mean == 0.0 {
            0.0
        } else {
            variance.sqrt() / mean * 100.0
        };
        let mut deviations = sorted
            .iter()
            .map(|sample| (sample - p50).abs())
            .collect::<Vec<_>>();
        deviations.sort_unstable_by(f64::total_cmp);
        Ok(Self {
            value: p50,
            p50,
            p95: percentile(&sorted, 0.95),
            p99: percentile(&sorted, 0.99),
            mad: percentile(&deviations, 0.50),
            cv_pct,
            runs: sorted.len(),
        })
    }

    /// Whether this distribution can participate in an activated gate.
    #[must_use]
    pub fn stable_for_activation(&self) -> bool {
        self.runs >= PERF_MIN_RUNS && self.cv_pct < PERF_MAX_CV_PCT
    }
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    debug_assert!(!sorted.is_empty());
    let scaled = (sorted.len() - 1) as f64 * quantile;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let index = scaled.round() as usize;
    sorted[index]
}

/// One engine or comparison row in a gate artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerfCellResult {
    pub fixture: String,
    pub metric: String,
    pub engine: String,
    pub unit: String,
    #[serde(flatten)]
    pub distribution: DistributionSummary,
}

/// Per-gate JSON artifact matching the committed E0.6 schema contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerfGateArtifact {
    pub schema_version: String,
    pub gate: PerfGate,
    pub machine_fingerprint: String,
    pub git_rev: String,
    /// Shared identifier for the bounded candidate/rerun measurement window.
    pub run_window: String,
    /// Unique identifier for one pass inside the measurement window.
    pub run_id: String,
    pub corpus_manifest_hash: String,
    pub manifest_sha256: String,
    pub cells: Vec<PerfCellResult>,
    pub laws_attested: bool,
}

impl PerfGateArtifact {
    /// Encode canonical pretty JSON.
    ///
    /// # Errors
    ///
    /// Returns a serde error when a non-finite number slipped past validation.
    pub fn to_json_pretty(&self) -> Result<String, GauntletError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Render the compact operator table printed beside JSON.
    #[must_use]
    pub fn human_table(&self) -> String {
        let mut table = String::from(
            "fixture | engine | metric | p50 | p95 | p99 | cv_pct | runs | activation\n",
        );
        table.push_str("--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---\n");
        for cell in &self.cells {
            let activation = if cell.distribution.stable_for_activation() {
                "stable"
            } else {
                "noise"
            };
            let _ = writeln!(
                table,
                "{} | {} | {} ({}) | {:.6} | {:.6} | {:.6} | {:.3} | {} | {}",
                cell.fixture,
                cell.engine,
                cell.metric,
                cell.unit,
                cell.distribution.p50,
                cell.distribution.p95,
                cell.distribution.p99,
                cell.distribution.cv_pct,
                cell.distribution.runs,
                activation,
            );
        }
        table
    }

    /// Write JSON and Markdown artifacts for one gate.
    ///
    /// # Errors
    ///
    /// Returns typed serialization or filesystem errors.
    pub fn write_to(&self, output_dir: &Path) -> Result<(PathBuf, PathBuf), GauntletError> {
        fs::create_dir_all(output_dir)?;
        let stem = self.gate.label();
        let json_path = output_dir.join(format!("{stem}.json"));
        let table_path = output_dir.join(format!("{stem}.md"));
        fs::write(&json_path, self.to_json_pretty()?)?;
        fs::write(&table_path, self.human_table())?;
        Ok((json_path, table_path))
    }
}

/// Deterministic machine label that is specific enough to reject accidental
/// cross-machine ratchet comparisons.
#[must_use]
pub fn machine_fingerprint() -> String {
    let parallelism = std::thread::available_parallelism().map_or(1, usize::from);
    let cpu = fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|contents| {
            contents
                .lines()
                .find_map(|line| {
                    line.strip_prefix("model name\t:")
                        .or_else(|| line.strip_prefix("Hardware\t:"))
                        .map(str::trim)
                })
                .map(str::to_owned)
        })
        .unwrap_or_else(|| "unknown-cpu".to_owned());
    format!(
        "{}-{}-{parallelism}cpu-{}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        cpu.replace(['/', ' '], "_")
    )
}

/// Linux peak resident set size in bytes from `VmHWM`.
///
/// Other operating systems return `None`. The isolated macOS benchmark child
/// is wrapped in `/usr/bin/time -l` and parsed by
/// [`parse_macos_time_max_rss_bytes`] instead of fabricating an in-process
/// value.
#[must_use]
pub fn peak_rss_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let status = fs::read_to_string("/proc/self/status").ok()?;
        parse_linux_vmhwm_bytes(&status)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Parse the byte-valued peak RSS row emitted by macOS `/usr/bin/time -l`.
///
/// The parser requires the complete four-word label so unrelated counters such
/// as `peak memory footprint` cannot be mistaken for resident-set evidence.
#[must_use]
pub fn parse_macos_time_max_rss_bytes(report: &str) -> Option<u64> {
    report.lines().find_map(|line| {
        let mut fields = line.split_ascii_whitespace();
        let bytes = fields.next()?.parse::<u64>().ok()?;
        (fields.next() == Some("maximum")
            && fields.next() == Some("resident")
            && fields.next() == Some("set")
            && fields.next() == Some("size")
            && fields.next().is_none())
        .then_some(bytes)
    })
}

fn parse_linux_vmhwm_bytes(status: &str) -> Option<u64> {
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    let mut fields = line.split_ascii_whitespace();
    let _label = fields.next()?;
    let kib = fields.next()?.parse::<u64>().ok()?;
    match fields.next() {
        Some("kB") => kib.checked_mul(1024),
        _ => None,
    }
}

/// Assert that a matrix contains every gate and no dishonest zero-density QG-5
/// cell.
///
/// # Errors
///
/// Returns a typed campaign error when coverage is incomplete.
pub fn validate_matrix(matrix: &PerfMatrixSpec) -> Result<(), GauntletError> {
    let gates = matrix
        .cells
        .iter()
        .map(|cell| cell.gate)
        .collect::<BTreeSet<_>>();
    if gates != PerfGate::ALL.into_iter().collect() {
        return Err(GauntletError::InvalidCampaign {
            reason: "performance matrix does not cover QG-1 through QG-10".to_owned(),
        });
    }
    if matrix.cells.iter().any(|cell| {
        cell.gate == PerfGate::Qg5
            && cell
                .tombstone_density_pct
                .is_none_or(|density| density == 0)
    }) {
        return Err(GauntletError::InvalidCampaign {
            reason: "QG-5 requires a nonzero tombstone density".to_owned(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_matrix_covers_every_gate_and_required_cross_products() {
        let matrix = PerfMatrixSpec::complete();
        validate_matrix(&matrix).expect("complete matrix");
        assert_eq!(
            matrix
                .for_gate(PerfGate::Qg1)
                .into_iter()
                .filter(|cell| cell.metric == "docs_per_second")
                .count(),
            4 * 4 * 2
        );
        assert_eq!(matrix.for_gate(PerfGate::Qg1).len(), 4 * 4 * 2 + 2);
        assert_eq!(matrix.for_gate(PerfGate::Qg3).len(), 5);
        assert_eq!(matrix.for_gate(PerfGate::Qg5).len(), 3);
        assert_eq!(matrix.for_gate(PerfGate::Qg6).len(), 5 * 2 * 2);
        assert_eq!(matrix.for_gate(PerfGate::Qg8).len(), 6);
        assert_eq!(matrix.for_gate(PerfGate::Qg10).len(), 1);
    }

    #[test]
    fn distribution_reports_percentiles_mad_cv_and_stability() {
        let samples = [10.0, 10.1, 9.9, 10.0, 10.2, 9.8, 10.0, 10.1, 9.9, 10.0];
        let summary = DistributionSummary::from_samples(&samples).expect("summary");
        assert!((summary.p50 - 10.0).abs() < f64::EPSILON);
        assert_eq!(summary.runs, PERF_MIN_RUNS);
        assert!(summary.mad <= 0.1);
        assert!(summary.cv_pct < 2.0);
        assert!(summary.stable_for_activation());
        assert!(DistributionSummary::from_samples(&[]).is_err());
        assert!(DistributionSummary::from_samples(&[f64::NAN]).is_err());
        assert!(DistributionSummary::from_samples(&[-1.0]).is_err());
    }

    #[test]
    fn artifact_json_and_table_retain_required_e06_fields() {
        let distribution = DistributionSummary::from_samples(&[1.0; PERF_MIN_RUNS])
            .expect("constant distribution");
        let artifact = PerfGateArtifact {
            schema_version: PERF_ARTIFACT_SCHEMA_VERSION.to_owned(),
            gate: PerfGate::Qg1,
            machine_fingerprint: "linux-x86_64-test".to_owned(),
            git_rev: "0123456789abcdef".to_owned(),
            run_window: "test-window".to_owned(),
            run_id: "candidate".to_owned(),
            corpus_manifest_hash: "a".repeat(64),
            manifest_sha256: "b".repeat(64),
            cells: vec![PerfCellResult {
                fixture: "bulk/tiny/1/positions_on".to_owned(),
                metric: "docs_per_second".to_owned(),
                engine: "quill".to_owned(),
                unit: "docs/s".to_owned(),
                distribution,
            }],
            laws_attested: true,
        };
        let json = artifact.to_json_pretty().expect("artifact JSON");
        let value: serde_json::Value = serde_json::from_str(&json).expect("decode artifact");
        for key in [
            "schema_version",
            "gate",
            "machine_fingerprint",
            "git_rev",
            "run_window",
            "run_id",
            "corpus_manifest_hash",
            "manifest_sha256",
            "cells",
            "laws_attested",
        ] {
            assert!(value.get(key).is_some(), "missing required field {key}");
        }
        let table = artifact.human_table();
        assert!(table.contains("cv_pct"));
        assert!(table.contains("bulk/tiny/1/positions_on"));
        assert!(table.contains("stable"));
    }

    #[test]
    fn linux_vmhwm_parser_requires_the_documented_unit() {
        assert_eq!(
            parse_linux_vmhwm_bytes("Name:\tbench\nVmHWM:\t   1234 kB\n"),
            Some(1_263_616)
        );
        assert_eq!(parse_linux_vmhwm_bytes("VmHWM: 12 MB"), None);
        assert_eq!(parse_linux_vmhwm_bytes("VmRSS: 12 kB"), None);
    }

    #[test]
    fn macos_time_parser_requires_the_exact_peak_rss_label() {
        let report = "\
        0.12 real 0.10 user 0.02 sys
        1212416  maximum resident set size
        9122317  instructions retired
        917696  peak memory footprint
";
        assert_eq!(parse_macos_time_max_rss_bytes(report), Some(1_212_416));
        assert_eq!(
            parse_macos_time_max_rss_bytes("917696 peak memory footprint"),
            None
        );
        assert_eq!(
            parse_macos_time_max_rss_bytes("1212416 maximum resident set size bytes"),
            None
        );
    }

    #[test]
    fn gate_parser_accepts_only_normative_labels() {
        assert_eq!("qg-1".parse::<PerfGate>().expect("QG-1"), PerfGate::Qg1);
        assert_eq!("QG_10".parse::<PerfGate>().expect("QG-10"), PerfGate::Qg10);
        assert!("QG-0".parse::<PerfGate>().is_err());
    }
}
