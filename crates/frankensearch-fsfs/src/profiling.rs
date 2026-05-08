//! Profiling harness contracts for fsfs optimization cycles.
//!
//! This module defines:
//! - a deterministic profiling workflow (flamegraph/heap/syscall),
//! - a self-calibrating host/corpus profile recommendation artifact,
//! - an impact-confidence-effort opportunity matrix,
//! - a single-lever iteration validator for behavior-preserving optimization.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

/// Schema version for the profiling workflow contract.
pub const PROFILING_WORKFLOW_SCHEMA_VERSION: &str = "fsfs-profiling-workflow-v1";
/// Schema version for the opportunity-matrix contract.
pub const OPPORTUNITY_MATRIX_SCHEMA_VERSION: &str = "fsfs-opportunity-matrix-v1";
/// Schema version for crawl/ingest optimization track planning.
pub const CRAWL_INGEST_OPT_TRACK_SCHEMA_VERSION: &str = "fsfs-crawl-ingest-opt-track-v1";
/// Schema version for self-calibrating fsfs host/corpus profiles.
pub const SELF_CALIBRATING_PROFILE_SCHEMA_VERSION: &str = "fsfs-self-calibrating-profile-v1";
/// Schema version for fsfs model-cache diagnostics.
pub const MODEL_CACHE_DIAGNOSTICS_SCHEMA_VERSION: u32 = 1;
pub const MODEL_CACHE_DIAGNOSTICS_CONTRACT_KIND: &str = "fsfs_model_cache_diagnostics_contract";
pub const MODEL_CACHE_DIAGNOSTICS_REPORT_KIND: &str = "fsfs_model_cache_diagnostics_report";

/// Reason code emitted when an optimization iteration is accepted.
pub const ITERATION_REASON_ACCEPTED: &str = "opt.iteration.accepted";
/// Reason code emitted when no lever changed.
pub const ITERATION_REASON_NO_CHANGE: &str = "opt.iteration.invalid.no_change";
/// Reason code emitted when more than one lever changed.
pub const ITERATION_REASON_MULTI_CHANGE: &str = "opt.iteration.invalid.multiple_levers";

/// Profile lane required by the harness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfileKind {
    /// CPU hotspot profile (flamegraph).
    Flamegraph,
    /// Heap/allocation profile.
    Heap,
    /// Syscall profile.
    Syscall,
}

impl std::fmt::Display for ProfileKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Flamegraph => write!(f, "flamegraph"),
            Self::Heap => write!(f, "heap"),
            Self::Syscall => write!(f, "syscall"),
        }
    }
}

/// One deterministic profiling step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileStep {
    /// Profile lane this step captures.
    pub kind: ProfileKind,
    /// Human-readable label used in manifests and logs.
    pub label: String,
    /// Command template to run.
    pub command: String,
    /// Artifact path template.
    pub artifact_path: String,
}

/// Deterministic profile workflow for a dataset profile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileWorkflow {
    /// Contract schema version.
    pub schema_version: String,
    /// Dataset profile (tiny/small/medium/etc).
    pub dataset_profile: String,
    /// Ordered profile steps.
    pub steps: Vec<ProfileStep>,
}

impl ProfileWorkflow {
    /// Build the default profiling workflow for a dataset profile.
    #[must_use]
    pub fn for_dataset_profile(dataset_profile: &str) -> Self {
        let normalized = dataset_profile.trim();
        let normalized = if normalized.is_empty() {
            "small"
        } else {
            normalized
        };

        Self {
            schema_version: PROFILING_WORKFLOW_SCHEMA_VERSION.to_owned(),
            dataset_profile: normalized.to_owned(),
            steps: vec![
                ProfileStep {
                    kind: ProfileKind::Flamegraph,
                    label: "cpu.hotspots".to_owned(),
                    command: format!(
                        "cargo flamegraph -p frankensearch-fsfs --test benchmark_baseline_matrix -- --profile {normalized}"
                    ),
                    artifact_path: format!("profiles/{normalized}/flamegraph.svg"),
                },
                ProfileStep {
                    kind: ProfileKind::Heap,
                    label: "heap.allocations".to_owned(),
                    command: format!(
                        "heaptrack target/release/fsfs --mode benchmark --profile {normalized}"
                    ),
                    artifact_path: format!("profiles/{normalized}/heaptrack.out"),
                },
                ProfileStep {
                    kind: ProfileKind::Syscall,
                    label: "syscalls.io".to_owned(),
                    command: format!(
                        "strace -ff -ttT -o profiles/{normalized}/syscall target/release/fsfs --mode benchmark --profile {normalized}"
                    ),
                    artifact_path: format!("profiles/{normalized}/syscall.*"),
                },
            ],
        }
    }

    /// Materialize deterministic artifact entries for a run.
    #[must_use]
    pub fn artifact_manifest(&self, run_id: &str) -> Vec<ProfileArtifact> {
        self.steps
            .iter()
            .map(|step| ProfileArtifact {
                kind: step.kind,
                artifact_path: format!("{run_id}/{}", step.artifact_path),
                replay_command: format!(
                    "fsfs profile replay --run-id {run_id} --kind {}",
                    step.kind
                ),
            })
            .collect()
    }
}

/// Artifact descriptor emitted by the harness.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileArtifact {
    /// Profile lane.
    pub kind: ProfileKind,
    /// Path relative to benchmark artifact root.
    pub artifact_path: String,
    /// Replay command for deterministic triage.
    pub replay_command: String,
}

/// Model-cache state observed during a self-calibrating profile run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelCacheState {
    /// Quality model artifacts are present and warm in the OS/model cache.
    Warm,
    /// Model artifacts exist, but the run measured a cold-load path.
    Cold,
    /// Quality model artifacts are not available on this host.
    Missing,
    /// Cache availability was not measured.
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelCacheDownloadStatus {
    Ready,
    Offline,
    Disabled,
    Missing,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelCacheFallbackPath {
    QualityModel,
    FastEmbed,
    HashEmbedder,
    LexicalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelCacheAdviceSeverity {
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCacheDiagnosticRule {
    pub rule_id: String,
    pub state: ModelCacheState,
    pub reason_code: String,
    pub severity: ModelCacheAdviceSeverity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCacheDiagnosticsContractDefinition {
    pub kind: String,
    pub v: u32,
    pub redacted_path_required: bool,
    pub network_required_in_tests: bool,
    pub required_fields: Vec<String>,
    pub advice_env_vars: Vec<String>,
    pub advice_config_knobs: Vec<String>,
    pub state_rules: Vec<ModelCacheDiagnosticRule>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCacheArtifactIdentity {
    pub model_id: String,
    pub revision: String,
    pub digest_sha256: String,
    pub redacted_model_dir: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCacheWarmupDiagnostics {
    pub state: ModelCacheState,
    pub warmup_latency_ms: u32,
    pub cold_load_latency_ms: u32,
    pub memory_mib: u32,
    pub download_status: ModelCacheDownloadStatus,
    pub offline: bool,
    pub fallback_path: ModelCacheFallbackPath,
    pub artifact: ModelCacheArtifactIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCacheOperatorAdvice {
    pub reason_code: String,
    pub severity: ModelCacheAdviceSeverity,
    pub summary: String,
    pub env_var: String,
    pub config_knob: String,
    pub command: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCacheDiagnosticsReport {
    pub kind: String,
    pub v: u32,
    pub generated_at: String,
    pub surface: String,
    pub diagnostics: ModelCacheWarmupDiagnostics,
    pub advice: Vec<ModelCacheOperatorAdvice>,
    pub raw_paths_present: bool,
    pub network_required: bool,
}

#[must_use]
pub fn model_cache_diagnostics_contract_definition() -> ModelCacheDiagnosticsContractDefinition {
    ModelCacheDiagnosticsContractDefinition {
        kind: MODEL_CACHE_DIAGNOSTICS_CONTRACT_KIND.to_owned(),
        v: MODEL_CACHE_DIAGNOSTICS_SCHEMA_VERSION,
        redacted_path_required: true,
        network_required_in_tests: false,
        required_fields: vec![
            "state".to_owned(),
            "model_id".to_owned(),
            "revision".to_owned(),
            "digest_sha256".to_owned(),
            "warmup_latency_ms".to_owned(),
            "memory_mib".to_owned(),
            "download_status".to_owned(),
            "offline".to_owned(),
            "fallback_path".to_owned(),
        ],
        advice_env_vars: vec![
            "FRANKENSEARCH_MODEL_DIR".to_owned(),
            "FSFS_DOWNLOAD_MODE".to_owned(),
        ],
        advice_config_knobs: vec![
            "indexing.model_dir".to_owned(),
            "search.fast_only".to_owned(),
        ],
        state_rules: vec![
            ModelCacheDiagnosticRule {
                rule_id: "warm-ready".to_owned(),
                state: ModelCacheState::Warm,
                reason_code: "model_cache.ready".to_owned(),
                severity: ModelCacheAdviceSeverity::Info,
            },
            ModelCacheDiagnosticRule {
                rule_id: "cold-warmup".to_owned(),
                state: ModelCacheState::Cold,
                reason_code: "model_cache.cold.warmup".to_owned(),
                severity: ModelCacheAdviceSeverity::Warning,
            },
            ModelCacheDiagnosticRule {
                rule_id: "missing-offline".to_owned(),
                state: ModelCacheState::Missing,
                reason_code: "model_cache.missing.offline".to_owned(),
                severity: ModelCacheAdviceSeverity::Error,
            },
            ModelCacheDiagnosticRule {
                rule_id: "unknown-unmeasured".to_owned(),
                state: ModelCacheState::Unknown,
                reason_code: "model_cache.unknown.unmeasured".to_owned(),
                severity: ModelCacheAdviceSeverity::Warning,
            },
        ],
    }
}

#[must_use]
pub fn model_cache_warm_diagnostics_fixture() -> ModelCacheDiagnosticsReport {
    model_cache_report(
        ModelCacheState::Warm,
        18,
        430,
        512,
        ModelCacheDownloadStatus::Ready,
        false,
        ModelCacheFallbackPath::QualityModel,
        "minilm-l6-v2-fixture-rev",
        "aa",
        vec![advice(
            "model_cache.ready",
            ModelCacheAdviceSeverity::Info,
            "quality model cache is warm and ready",
            "FRANKENSEARCH_MODEL_DIR",
            "indexing.model_dir",
            "fsfs status --format json",
        )],
    )
}

#[must_use]
pub fn model_cache_cold_diagnostics_fixture() -> ModelCacheDiagnosticsReport {
    model_cache_report(
        ModelCacheState::Cold,
        420,
        430,
        512,
        ModelCacheDownloadStatus::Ready,
        false,
        ModelCacheFallbackPath::QualityModel,
        "minilm-l6-v2-fixture-rev",
        "bb",
        vec![
            advice(
                "model_cache.cold.warmup",
                ModelCacheAdviceSeverity::Warning,
                "quality model artifacts exist but cold warmup is over budget",
                "FRANKENSEARCH_MODEL_DIR",
                "indexing.model_dir",
                "fsfs doctor --check-models --format json",
            ),
            advice(
                "model_cache.cold.fast_only_escape",
                ModelCacheAdviceSeverity::Info,
                "set fast-only when cold model startup is unacceptable",
                "FSFS_DOWNLOAD_MODE",
                "search.fast_only",
                "fsfs search --fast-only --format json",
            ),
        ],
    )
}

#[must_use]
pub fn model_cache_missing_offline_diagnostics_fixture() -> ModelCacheDiagnosticsReport {
    model_cache_report(
        ModelCacheState::Missing,
        0,
        0,
        0,
        ModelCacheDownloadStatus::Offline,
        true,
        ModelCacheFallbackPath::HashEmbedder,
        "missing-offline-fixture",
        "cc",
        vec![advice(
            "model_cache.missing.offline",
            ModelCacheAdviceSeverity::Error,
            "quality model cache is missing and downloads are disabled",
            "FSFS_DOWNLOAD_MODE",
            "search.fast_only",
            "fsfs download-models --model minilm-l6-v2",
        )],
    )
}

impl ModelCacheDiagnosticsContractDefinition {
    /// Validates the model-cache diagnostics contract.
    ///
    /// # Errors
    ///
    /// Returns an error when required state rules or operator hints are missing.
    pub fn validate(&self) -> Result<(), String> {
        if self.kind != MODEL_CACHE_DIAGNOSTICS_CONTRACT_KIND {
            return Err("unexpected model-cache contract kind".to_owned());
        }
        if self.v != MODEL_CACHE_DIAGNOSTICS_SCHEMA_VERSION {
            return Err("unsupported model-cache contract version".to_owned());
        }
        if !self.redacted_path_required {
            return Err("model-cache diagnostics must require redacted paths".to_owned());
        }
        if self.network_required_in_tests {
            return Err("model-cache diagnostics tests must not require network".to_owned());
        }
        if !self
            .advice_env_vars
            .iter()
            .any(|env_var| env_var == "FRANKENSEARCH_MODEL_DIR")
        {
            return Err("operator advice must mention FRANKENSEARCH_MODEL_DIR".to_owned());
        }
        if !self
            .advice_config_knobs
            .iter()
            .any(|knob| knob == "indexing.model_dir")
        {
            return Err("operator advice must mention indexing.model_dir".to_owned());
        }

        let states: BTreeSet<ModelCacheState> =
            self.state_rules.iter().map(|rule| rule.state).collect();
        let covers_all_states = [
            ModelCacheState::Warm,
            ModelCacheState::Cold,
            ModelCacheState::Missing,
            ModelCacheState::Unknown,
        ]
        .into_iter()
        .all(|state| states.contains(&state));
        if !covers_all_states {
            return Err("model-cache contract missing state rule".to_owned());
        }
        Ok(())
    }
}

impl ModelCacheDiagnosticsReport {
    /// Validates a model-cache diagnostics report.
    ///
    /// # Errors
    ///
    /// Returns an error when the report leaks raw paths, requires network, or
    /// omits actionable operator advice.
    pub fn validate(&self) -> Result<(), String> {
        if self.kind != MODEL_CACHE_DIAGNOSTICS_REPORT_KIND {
            return Err("unexpected model-cache report kind".to_owned());
        }
        if self.v != MODEL_CACHE_DIAGNOSTICS_SCHEMA_VERSION {
            return Err("unsupported model-cache report version".to_owned());
        }
        if self.generated_at.trim().is_empty() || self.surface.trim().is_empty() {
            return Err("model-cache report missing stable identity fields".to_owned());
        }
        if self.raw_paths_present {
            return Err("model-cache report leaks raw paths".to_owned());
        }
        if self.network_required {
            return Err("model-cache diagnostics must not require network in tests".to_owned());
        }
        self.diagnostics.validate()?;
        if self.advice.is_empty() {
            return Err("model-cache report must include operator advice".to_owned());
        }
        for advice in &self.advice {
            advice.validate()?;
        }
        Ok(())
    }
}

impl ModelCacheWarmupDiagnostics {
    fn validate(&self) -> Result<(), String> {
        self.artifact.validate()?;
        match self.state {
            ModelCacheState::Warm => {
                if self.warmup_latency_ms == 0 || self.memory_mib == 0 {
                    return Err("warm cache diagnostics require latency and memory".to_owned());
                }
            }
            ModelCacheState::Cold => {
                if self.cold_load_latency_ms < self.warmup_latency_ms {
                    return Err("cold load latency must be at least warmup latency".to_owned());
                }
            }
            ModelCacheState::Missing => {
                if self.fallback_path == ModelCacheFallbackPath::QualityModel {
                    return Err("missing model cache must not use quality fallback".to_owned());
                }
            }
            ModelCacheState::Unknown => {}
        }
        Ok(())
    }
}

impl ModelCacheArtifactIdentity {
    fn validate(&self) -> Result<(), String> {
        if self.model_id.trim().is_empty() || self.revision.trim().is_empty() {
            return Err("model identity fields must not be empty".to_owned());
        }
        if !is_sha256_digest(&self.digest_sha256) {
            return Err("model digest must be sha256:<64 lowercase hex>".to_owned());
        }
        if !is_redacted_model_dir(&self.redacted_model_dir) {
            return Err("model directory must be redacted".to_owned());
        }
        Ok(())
    }
}

impl ModelCacheOperatorAdvice {
    fn validate(&self) -> Result<(), String> {
        if !self.reason_code.starts_with("model_cache.") {
            return Err("model-cache advice reason code must use model_cache namespace".to_owned());
        }
        if self.summary.trim().is_empty()
            || self.env_var.trim().is_empty()
            || self.config_knob.trim().is_empty()
            || self.command.trim().is_empty()
        {
            return Err("model-cache operator advice must be actionable".to_owned());
        }
        Ok(())
    }
}

fn model_cache_report(
    state: ModelCacheState,
    warmup_latency_ms: u32,
    cold_load_latency_ms: u32,
    memory_mib: u32,
    download_status: ModelCacheDownloadStatus,
    offline: bool,
    fallback_path: ModelCacheFallbackPath,
    revision: &str,
    digest_pair: &str,
    advice: Vec<ModelCacheOperatorAdvice>,
) -> ModelCacheDiagnosticsReport {
    ModelCacheDiagnosticsReport {
        kind: MODEL_CACHE_DIAGNOSTICS_REPORT_KIND.to_owned(),
        v: MODEL_CACHE_DIAGNOSTICS_SCHEMA_VERSION,
        generated_at: "2026-05-08T00:00:00Z".to_owned(),
        surface: "fsfs doctor --check-models --format json".to_owned(),
        diagnostics: ModelCacheWarmupDiagnostics {
            state,
            warmup_latency_ms,
            cold_load_latency_ms,
            memory_mib,
            download_status,
            offline,
            fallback_path,
            artifact: ModelCacheArtifactIdentity {
                model_id: "sentence-transformers/all-MiniLM-L6-v2".to_owned(),
                revision: revision.to_owned(),
                digest_sha256: fixed_sha256(digest_pair),
                redacted_model_dir: "<redacted:model-cache>/minilm-l6-v2".to_owned(),
            },
        },
        advice,
        raw_paths_present: false,
        network_required: false,
    }
}

fn advice(
    reason_code: &str,
    severity: ModelCacheAdviceSeverity,
    summary: &str,
    env_var: &str,
    config_knob: &str,
    command: &str,
) -> ModelCacheOperatorAdvice {
    ModelCacheOperatorAdvice {
        reason_code: reason_code.to_owned(),
        severity,
        summary: summary.to_owned(),
        env_var: env_var.to_owned(),
        config_knob: config_knob.to_owned(),
        command: command.to_owned(),
    }
}

fn fixed_sha256(pair: &str) -> String {
    format!("sha256:{}", pair.repeat(32))
}

fn is_sha256_digest(value: &str) -> bool {
    value
        .strip_prefix("sha256:")
        .is_some_and(|digest| digest.len() == 64 && digest.chars().all(is_lower_hex_digit))
}

fn is_lower_hex_digit(ch: char) -> bool {
    ch.is_ascii_digit() || ('a'..='f').contains(&ch)
}

fn is_redacted_model_dir(value: &str) -> bool {
    value.starts_with("<redacted:") && !value.starts_with('/')
}

/// Host capability snapshot used by the profile recommender.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostProfileSnapshot {
    /// Stable host label used in artifact paths.
    pub host_id: String,
    /// Logical CPU count available to the process.
    pub logical_cpus: u16,
    /// Total host memory in MiB.
    pub total_memory_mib: u32,
    /// Currently available memory in MiB at the start of the run.
    pub available_memory_mib: u32,
    /// SIMD lane width observed by the profiler.
    pub simd_lanes: u8,
    /// Model-cache state observed for quality-tier embeds.
    pub model_cache_state: ModelCacheState,
}

impl HostProfileSnapshot {
    /// Available-memory headroom as a bounded percentage.
    #[must_use]
    pub fn memory_headroom_pct(&self) -> u8 {
        if self.total_memory_mib == 0 {
            return 0;
        }

        let raw = (u64::from(self.available_memory_mib) * 100) / u64::from(self.total_memory_mib);
        u8::try_from(raw.min(100)).unwrap_or(100)
    }
}

/// Representative corpus summary for a self-calibrating profile run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusProfileSnapshot {
    /// Stable corpus identifier.
    pub corpus_id: String,
    /// Number of indexed documents sampled by the profile run.
    pub document_count: u32,
    /// Total corpus bytes sampled.
    pub total_bytes: u64,
    /// Number of known relevance/query clusters represented.
    pub cluster_count: u16,
    /// Number of representative queries used for calibration.
    pub representative_query_count: u16,
}

impl CorpusProfileSnapshot {
    /// Mean document size in bytes, rounded down.
    #[must_use]
    pub fn average_document_bytes(&self) -> u64 {
        if self.document_count == 0 {
            0
        } else {
            self.total_bytes / u64::from(self.document_count)
        }
    }
}

/// Search measurements collected by the self-calibrating profile lane.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchProfileMeasurements {
    /// Phase-1 p95 latency in microseconds.
    pub phase1_p95_us: u64,
    /// Phase-2 p95 latency in microseconds.
    pub phase2_p95_us: u64,
    /// Vector-search throughput measured as documents scanned per millisecond.
    pub vector_search_docs_per_ms: u32,
    /// Peak process memory observed during the run in MiB.
    pub peak_memory_mib: u32,
    /// Candidate multiplier used by the sampled run.
    pub observed_candidate_multiplier: u16,
    /// Warm-cache model initialization latency in milliseconds.
    pub model_cache_warm_ms: u32,
    /// Cold-cache model initialization latency in milliseconds.
    pub model_cache_cold_ms: u32,
    /// Quality-tier relevance uplift in basis points over the fast tier.
    pub quality_uplift_basis_points: u16,
}

/// Input envelope consumed by the deterministic profile recommender.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelfCalibratingProfileInput {
    /// Stable run id used in artifact paths and replay commands.
    pub run_id: String,
    /// Host capabilities and cache state.
    pub host: HostProfileSnapshot,
    /// Representative corpus profile.
    pub corpus: CorpusProfileSnapshot,
    /// Search-path measurements collected on this host/corpus pair.
    pub measurements: SearchProfileMeasurements,
    /// Requested user-facing result limit used for candidate-budget selection.
    pub requested_limit: u16,
}

/// Serializable recommendation matching the main `TwoTierConfig` knob names.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecommendedTwoTierConfig {
    /// Quality blend weight in per-mille units (`700` means `0.7`).
    pub quality_weight_per_mille: u16,
    /// RRF K constant.
    pub rrf_k: u16,
    /// Candidate multiplier to apply per source.
    pub candidate_multiplier: u16,
    /// Quality phase timeout in milliseconds.
    pub quality_timeout_ms: u32,
    /// Whether the host should skip quality refinement.
    pub fast_only: bool,
    /// Minimum corpus size before ANN/HNSW should be preferred.
    pub hnsw_threshold: u32,
    /// MRL search dimensions (`0` disables MRL).
    pub mrl_search_dims: u16,
    /// Number of candidates to rescore after an MRL scan.
    pub mrl_rescore_top_k: u16,
}

impl RecommendedTwoTierConfig {
    /// Safe default values matching the library's conservative config defaults.
    #[must_use]
    pub const fn safe_fallback_defaults() -> Self {
        Self {
            quality_weight_per_mille: 700,
            rrf_k: 60,
            candidate_multiplier: 3,
            quality_timeout_ms: 500,
            fast_only: false,
            hnsw_threshold: 50_000,
            mrl_search_dims: 0,
            mrl_rescore_top_k: 30,
        }
    }

    /// Deterministically recommend two-tier knobs for a profile input.
    #[must_use]
    pub fn recommend_for(input: &SelfCalibratingProfileInput) -> Self {
        let fast_only = should_use_fast_only(input);
        let candidate_multiplier = recommended_candidate_multiplier(input);
        let hnsw_threshold = recommended_hnsw_threshold(input);
        let rrf_k = recommended_rrf_k(input, candidate_multiplier);
        let quality_timeout_ms = recommended_quality_timeout_ms(input, fast_only);
        let quality_weight_per_mille = recommended_quality_weight_per_mille(input, fast_only);

        Self {
            quality_weight_per_mille,
            rrf_k,
            candidate_multiplier,
            quality_timeout_ms,
            fast_only,
            hnsw_threshold,
            mrl_search_dims: recommended_mrl_dims(input),
            mrl_rescore_top_k: recommended_mrl_rescore_top_k(input),
        }
    }
}

/// Artifact kind emitted by the self-calibrating profile lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelfCalibratingProfileArtifactKind {
    /// Structured JSONL event stream from the profiling script.
    StructuredJsonl,
    /// Recommendation JSON artifact.
    RecommendationJson,
    /// Replay manifest for deterministic reruns.
    ReplayManifest,
}

/// Artifact descriptor emitted with a self-calibrating report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelfCalibratingProfileArtifact {
    /// Artifact kind.
    pub kind: SelfCalibratingProfileArtifactKind,
    /// Path relative to the profile artifact root.
    pub artifact_path: String,
    /// MIME/content type.
    pub content_type: String,
    /// Deterministic command that replays this artifact.
    pub replay_command: String,
}

impl SelfCalibratingProfileArtifact {
    /// Build the canonical artifact manifest for a run/corpus pair.
    #[must_use]
    pub fn for_run(run_id: &str, corpus_id: &str) -> Vec<Self> {
        let root = format!("runs/{run_id}/self_calibrating/{corpus_id}");
        let replay_command =
            format!("scripts/check_fsfs_self_calibrating_profile.sh --mode e2e --run-id {run_id}");

        vec![
            Self {
                kind: SelfCalibratingProfileArtifactKind::StructuredJsonl,
                artifact_path: format!("{root}/profile-events.jsonl"),
                content_type: "application/jsonl".to_owned(),
                replay_command: replay_command.clone(),
            },
            Self {
                kind: SelfCalibratingProfileArtifactKind::RecommendationJson,
                artifact_path: format!("{root}/recommendation.json"),
                content_type: "application/json".to_owned(),
                replay_command: replay_command.clone(),
            },
            Self {
                kind: SelfCalibratingProfileArtifactKind::ReplayManifest,
                artifact_path: format!("{root}/replay-manifest.json"),
                content_type: "application/json".to_owned(),
                replay_command,
            },
        ]
    }
}

/// Deterministic self-calibrating profile recommendation report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelfCalibratingProfileReport {
    /// Contract schema version.
    pub schema_version: String,
    /// Input profile consumed by the recommender.
    pub profile: SelfCalibratingProfileInput,
    /// Recommended production knobs.
    pub recommended_config: RecommendedTwoTierConfig,
    /// Conservative fallback knobs used when confidence is too low.
    pub safe_fallback_defaults: RecommendedTwoTierConfig,
    /// Confidence in the recommendation from 0 to 100.
    pub confidence: u8,
    /// Machine-readable reason codes explaining the recommendation.
    pub reason_codes: Vec<String>,
    /// Structured artifacts produced by the e2e profiling lane.
    pub artifacts: Vec<SelfCalibratingProfileArtifact>,
}

impl SelfCalibratingProfileReport {
    /// Build a deterministic recommendation report from measured input.
    #[must_use]
    pub fn from_input(profile: SelfCalibratingProfileInput) -> Self {
        let recommended_config = RecommendedTwoTierConfig::recommend_for(&profile);
        let confidence = recommendation_confidence(&profile, &recommended_config);
        let reason_codes = recommendation_reason_codes(&profile, &recommended_config);
        let artifacts =
            SelfCalibratingProfileArtifact::for_run(&profile.run_id, &profile.corpus.corpus_id);

        Self {
            schema_version: SELF_CALIBRATING_PROFILE_SCHEMA_VERSION.to_owned(),
            profile,
            recommended_config,
            safe_fallback_defaults: RecommendedTwoTierConfig::safe_fallback_defaults(),
            confidence,
            reason_codes,
            artifacts,
        }
    }
}

fn should_use_fast_only(input: &SelfCalibratingProfileInput) -> bool {
    let measurements = &input.measurements;
    let host = &input.host;
    let unavailable_quality = matches!(host.model_cache_state, ModelCacheState::Missing);
    let low_quality_gain = measurements.quality_uplift_basis_points < 500;
    let phase2_unusable = measurements.phase2_p95_us > 220_000 && low_quality_gain;
    let memory_constrained = host.memory_headroom_pct() < 15
        || measurements.peak_memory_mib
            >= host
                .available_memory_mib
                .saturating_mul(9)
                .checked_div(10)
                .unwrap_or(0);

    unavailable_quality || phase2_unusable || memory_constrained
}

fn recommended_candidate_multiplier(input: &SelfCalibratingProfileInput) -> u16 {
    let measurements = &input.measurements;
    if measurements.phase1_p95_us > 15_000 || measurements.vector_search_docs_per_ms < 500 {
        1
    } else if measurements.phase1_p95_us > 10_000 || measurements.vector_search_docs_per_ms < 900 {
        2
    } else if input.corpus.document_count >= 100_000
        && measurements.quality_uplift_basis_points >= 1_200
    {
        4
    } else {
        measurements.observed_candidate_multiplier.clamp(2, 3)
    }
}

fn recommended_hnsw_threshold(input: &SelfCalibratingProfileInput) -> u32 {
    if input.corpus.document_count >= 100_000 || input.measurements.vector_search_docs_per_ms < 500
    {
        10_000
    } else if input.corpus.document_count >= 50_000 {
        25_000
    } else {
        RecommendedTwoTierConfig::safe_fallback_defaults().hnsw_threshold
    }
}

fn recommended_rrf_k(input: &SelfCalibratingProfileInput, candidate_multiplier: u16) -> u16 {
    if input.corpus.document_count < 1_000 {
        45
    } else if candidate_multiplier >= 4 {
        75
    } else {
        RecommendedTwoTierConfig::safe_fallback_defaults().rrf_k
    }
}

fn recommended_quality_timeout_ms(input: &SelfCalibratingProfileInput, fast_only: bool) -> u32 {
    if fast_only {
        return 10;
    }

    let measured_ms = input.measurements.phase2_p95_us.div_ceil(1_000);
    let with_margin = measured_ms.saturating_add(25).clamp(120, 500);
    u32::try_from(with_margin).unwrap_or(500)
}

fn recommended_quality_weight_per_mille(
    input: &SelfCalibratingProfileInput,
    fast_only: bool,
) -> u16 {
    if fast_only {
        0
    } else if input.measurements.quality_uplift_basis_points >= 1_200
        && input.measurements.phase2_p95_us <= 150_000
    {
        800
    } else if input.measurements.quality_uplift_basis_points < 500
        || input.measurements.phase2_p95_us > 180_000
    {
        550
    } else {
        RecommendedTwoTierConfig::safe_fallback_defaults().quality_weight_per_mille
    }
}

fn recommended_mrl_dims(input: &SelfCalibratingProfileInput) -> u16 {
    if input.corpus.document_count >= 100_000 && input.measurements.phase1_p95_us > 10_000 {
        128
    } else {
        RecommendedTwoTierConfig::safe_fallback_defaults().mrl_search_dims
    }
}

fn recommended_mrl_rescore_top_k(input: &SelfCalibratingProfileInput) -> u16 {
    let requested = input.requested_limit.max(1);
    requested.saturating_mul(3).clamp(30, 150)
}

fn recommendation_confidence(
    input: &SelfCalibratingProfileInput,
    config: &RecommendedTwoTierConfig,
) -> u8 {
    let mut confidence = 92_u8;

    if input.corpus.representative_query_count < 12 {
        confidence = confidence.saturating_sub(20);
    } else if input.corpus.representative_query_count < 25 {
        confidence = confidence.saturating_sub(8);
    }

    if input.corpus.cluster_count < 3 {
        confidence = confidence.saturating_sub(10);
    }

    if matches!(input.host.model_cache_state, ModelCacheState::Unknown) {
        confidence = confidence.saturating_sub(12);
    }

    if input.host.memory_headroom_pct() < 20 {
        confidence = confidence.saturating_sub(15);
    }

    if config.fast_only {
        confidence = confidence.saturating_sub(10);
    }

    confidence
}

fn recommendation_reason_codes(
    input: &SelfCalibratingProfileInput,
    config: &RecommendedTwoTierConfig,
) -> Vec<String> {
    let mut reasons = Vec::new();

    if config.fast_only {
        reasons.push("profile.self_calibrating.fast_only".to_owned());
    } else {
        reasons.push("profile.self_calibrating.quality_enabled".to_owned());
    }

    if input.measurements.phase1_p95_us > 15_000 {
        reasons.push("profile.self_calibrating.phase1_over_budget".to_owned());
    }
    if input.measurements.phase2_p95_us > 150_000 {
        reasons.push("profile.self_calibrating.phase2_over_budget".to_owned());
    }
    if input.host.memory_headroom_pct() < 20 {
        reasons.push("profile.self_calibrating.low_memory_headroom".to_owned());
    }
    if input.corpus.document_count >= 50_000 {
        reasons.push("profile.self_calibrating.large_corpus_ann_threshold".to_owned());
    }
    if matches!(input.host.model_cache_state, ModelCacheState::Cold) {
        reasons.push("profile.self_calibrating.model_cache_cold".to_owned());
    }
    if config.candidate_multiplier <= 2 {
        reasons.push("profile.self_calibrating.candidate_budget_constrained".to_owned());
    } else if config.candidate_multiplier >= 4 {
        reasons.push("profile.self_calibrating.candidate_budget_expanded".to_owned());
    }

    reasons
}

/// One candidate optimization opportunity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpportunityCandidate {
    /// Stable candidate id.
    pub id: String,
    /// Candidate summary.
    pub summary: String,
    /// Expected impact (0-100).
    pub impact: u16,
    /// Confidence in estimate (0-100).
    pub confidence: u16,
    /// Effort cost (1-100), lower is better.
    pub effort: u16,
}

impl OpportunityCandidate {
    /// Deterministic ICE score in per-mille units.
    ///
    /// Score = `(impact * confidence * 1000) / effort`.
    #[must_use]
    pub fn ice_score_per_mille(&self) -> u32 {
        let effort = if self.effort == 0 { 1 } else { self.effort };
        (u32::from(self.impact) * u32::from(self.confidence) * 1_000) / u32::from(effort)
    }
}

/// Opportunity scoring table for optimization planning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpportunityMatrix {
    /// Contract schema version.
    pub schema_version: String,
    /// Candidate table.
    pub candidates: Vec<OpportunityCandidate>,
}

impl OpportunityMatrix {
    /// Build a matrix from candidates.
    #[must_use]
    pub fn new(candidates: Vec<OpportunityCandidate>) -> Self {
        Self {
            schema_version: OPPORTUNITY_MATRIX_SCHEMA_VERSION.to_owned(),
            candidates,
        }
    }

    /// Return deterministically ranked candidates.
    ///
    /// Tie-break order:
    /// 1. ICE score descending
    /// 2. impact descending
    /// 3. confidence descending
    /// 4. effort ascending
    /// 5. id lexicographic ascending
    #[must_use]
    pub fn ranked(&self) -> Vec<RankedOpportunity> {
        let mut ranked: Vec<RankedOpportunity> = self
            .candidates
            .iter()
            .cloned()
            .map(|candidate| RankedOpportunity {
                rank: 0,
                ice_score_per_mille: candidate.ice_score_per_mille(),
                candidate,
            })
            .collect();

        ranked.sort_by(|left, right| {
            right
                .ice_score_per_mille
                .cmp(&left.ice_score_per_mille)
                .then_with(|| right.candidate.impact.cmp(&left.candidate.impact))
                .then_with(|| right.candidate.confidence.cmp(&left.candidate.confidence))
                .then_with(|| left.candidate.effort.cmp(&right.candidate.effort))
                .then_with(|| left.candidate.id.cmp(&right.candidate.id))
        });

        for (index, candidate) in ranked.iter_mut().enumerate() {
            candidate.rank = index + 1;
        }

        ranked
    }
}

/// Ranked matrix row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RankedOpportunity {
    /// 1-based rank.
    pub rank: usize,
    /// ICE score in per-mille units.
    pub ice_score_per_mille: u32,
    /// Original candidate row.
    pub candidate: OpportunityCandidate,
}

/// Canonical crawl/ingest stages used by the optimization track.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CrawlIngestStage {
    DiscoveryWalk,
    Classification,
    CatalogMutation,
    QueueAdmission,
    EmbeddingGate,
}

/// Ranked hotspot entry for the crawl/ingest path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrawlIngestHotspot {
    /// 1-based rank derived from ICE score ordering.
    pub rank: usize,
    /// Stable optimization lever id.
    pub lever_id: String,
    /// Crawl/ingest stage targeted by this lever.
    pub stage: CrawlIngestStage,
    /// Human-readable optimization summary.
    pub summary: String,
    /// Expected p50 latency improvement percentage.
    pub expected_p50_gain_pct: u8,
    /// Expected p95 latency improvement percentage.
    pub expected_p95_gain_pct: u8,
    /// Expected ingest throughput improvement percentage.
    pub expected_throughput_gain_pct: u8,
}

/// Isomorphism proof checklist item for one optimization lever.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IsomorphismProofChecklistItem {
    /// Optimization lever being validated.
    pub lever_id: String,
    /// Baseline behavior this lever must preserve.
    pub baseline_comparator: String,
    /// Explicit invariants that must remain true.
    pub required_invariants: Vec<String>,
    /// Deterministic replay command for triage/proof.
    pub replay_command: String,
}

/// Deterministic rollback guardrail for one optimization lever.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RollbackGuardrail {
    /// Optimization lever protected by this guardrail.
    pub lever_id: String,
    /// Rollback command to execute when abort conditions are met.
    pub rollback_command: String,
    /// Reason codes that force rollback.
    pub abort_reason_codes: Vec<String>,
    /// Reason code expected after rollback succeeds.
    pub recovery_reason_code: String,
}

/// Complete crawl/ingest optimization track contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrawlIngestOptimizationTrack {
    /// Contract schema version.
    pub schema_version: String,
    /// Prioritized hotspot list with expected gains.
    pub hotspots: Vec<CrawlIngestHotspot>,
    /// Behavior-preserving proof checklist for every lever.
    pub proof_checklist: Vec<IsomorphismProofChecklistItem>,
    /// Rollback guardrails per optimization class.
    pub rollback_guardrails: Vec<RollbackGuardrail>,
}

/// Build the canonical crawl/ingest optimization opportunity matrix.
#[must_use]
pub fn crawl_ingest_opportunity_matrix() -> OpportunityMatrix {
    OpportunityMatrix::new(vec![
        OpportunityCandidate {
            id: "ingest.catalog.batch_upsert".into(),
            summary: "Batch catalog/changelog writes to reduce transaction overhead".into(),
            impact: 90,
            confidence: 90,
            effort: 18,
        },
        OpportunityCandidate {
            id: "crawl.classification.policy_batching".into(),
            summary: "Classify discovery candidates in batched policy windows".into(),
            impact: 68,
            confidence: 78,
            effort: 14,
        },
        OpportunityCandidate {
            id: "ingest.queue.lane_budget_admission".into(),
            summary: "Use lane-budget aware queue admission to reduce saturation churn".into(),
            impact: 74,
            confidence: 80,
            effort: 16,
        },
        OpportunityCandidate {
            id: "crawl.discovery.path_metadata_cache".into(),
            summary: "Cache path metadata during crawl to cut repeated stat/syscall work".into(),
            impact: 82,
            confidence: 82,
            effort: 20,
        },
        OpportunityCandidate {
            id: "ingest.embed_gate.early_skip".into(),
            summary: "Apply early embedding skip gates for low-signal candidates".into(),
            impact: 76,
            confidence: 88,
            effort: 24,
        },
    ])
}

/// Build the canonical crawl/ingest optimization track with hotspots, proofs,
/// and rollback guardrails.
#[must_use]
pub fn crawl_ingest_optimization_track() -> CrawlIngestOptimizationTrack {
    let ranked = crawl_ingest_opportunity_matrix().ranked();
    let hotspots = ranked
        .iter()
        .map(|entry| {
            let (stage, p50_gain, p95_gain, throughput_gain) =
                hotspot_expectations_for(&entry.candidate.id);
            CrawlIngestHotspot {
                rank: entry.rank,
                lever_id: entry.candidate.id.clone(),
                stage,
                summary: entry.candidate.summary.clone(),
                expected_p50_gain_pct: p50_gain,
                expected_p95_gain_pct: p95_gain,
                expected_throughput_gain_pct: throughput_gain,
            }
        })
        .collect::<Vec<_>>();

    let proof_checklist = hotspots
        .iter()
        .map(|hotspot| IsomorphismProofChecklistItem {
            lever_id: hotspot.lever_id.clone(),
            baseline_comparator: baseline_comparator_for(hotspot.stage).to_owned(),
            required_invariants: invariants_for_stage(hotspot.stage)
                .iter()
                .map(ToString::to_string)
                .collect(),
            replay_command: format!(
                "fsfs profile replay --lane ingest --lever-id {} --compare baseline",
                hotspot.lever_id
            ),
        })
        .collect::<Vec<_>>();

    let rollback_guardrails = hotspots
        .iter()
        .map(|hotspot| RollbackGuardrail {
            lever_id: hotspot.lever_id.clone(),
            rollback_command: format!(
                "fsfs profile rollback --lever-id {} --restore baseline",
                hotspot.lever_id
            ),
            abort_reason_codes: rollback_abort_reason_codes(hotspot.stage)
                .iter()
                .map(ToString::to_string)
                .collect(),
            recovery_reason_code: "opt.rollback.completed".to_owned(),
        })
        .collect::<Vec<_>>();

    CrawlIngestOptimizationTrack {
        schema_version: CRAWL_INGEST_OPT_TRACK_SCHEMA_VERSION.to_owned(),
        hotspots,
        proof_checklist,
        rollback_guardrails,
    }
}

fn hotspot_expectations_for(lever_id: &str) -> (CrawlIngestStage, u8, u8, u8) {
    match lever_id {
        "ingest.catalog.batch_upsert" => (CrawlIngestStage::CatalogMutation, 16, 24, 20),
        "crawl.classification.policy_batching" => (CrawlIngestStage::Classification, 10, 16, 12),
        "ingest.queue.lane_budget_admission" => (CrawlIngestStage::QueueAdmission, 9, 14, 11),
        "crawl.discovery.path_metadata_cache" => (CrawlIngestStage::DiscoveryWalk, 8, 13, 10),
        "ingest.embed_gate.early_skip" => (CrawlIngestStage::EmbeddingGate, 7, 11, 9),
        _ => (CrawlIngestStage::DiscoveryWalk, 5, 8, 5),
    }
}

const fn baseline_comparator_for(stage: CrawlIngestStage) -> &'static str {
    match stage {
        CrawlIngestStage::DiscoveryWalk => "baseline.crawl.discovery.sequential_walk",
        CrawlIngestStage::Classification => "baseline.crawl.classification.single_item",
        CrawlIngestStage::CatalogMutation => "baseline.ingest.catalog.single_upsert",
        CrawlIngestStage::QueueAdmission => "baseline.ingest.queue.global_fifo",
        CrawlIngestStage::EmbeddingGate => "baseline.ingest.embed.defer_after_index",
    }
}

const fn invariants_for_stage(stage: CrawlIngestStage) -> &'static [&'static str] {
    match stage {
        CrawlIngestStage::DiscoveryWalk => &[
            "no path omission across mount boundaries",
            "path canonicalization remains deterministic",
            "discovery scope decisions are unchanged",
        ],
        CrawlIngestStage::Classification => &[
            "ingestion_class assignment remains deterministic",
            "skip/index decisions preserve expected-loss ordering",
            "utility-score tie-break semantics remain unchanged",
        ],
        CrawlIngestStage::CatalogMutation => &[
            "catalog revision monotonicity preserved",
            "changelog stream sequence monotonicity preserved",
            "idempotent upsert semantics preserved",
        ],
        CrawlIngestStage::QueueAdmission => &[
            "lane budgets remain within configured hard limit",
            "backpressure transitions preserve reason-code semantics",
            "replay ordering remains monotonic",
        ],
        CrawlIngestStage::EmbeddingGate => &[
            "semantic-vs-lexical gating follows discovery policy",
            "low-signal candidates remain explainable with reason codes",
            "degraded-mode transitions remain reversible",
        ],
    }
}

const fn rollback_abort_reason_codes(stage: CrawlIngestStage) -> &'static [&'static str] {
    match stage {
        CrawlIngestStage::DiscoveryWalk => &[
            "discovery.scope.regression",
            "discovery.path_omission_detected",
            "ingest.replay.sequence_gap",
        ],
        CrawlIngestStage::Classification => &[
            "ingest.classification.regression",
            "ingest.expected_loss_violation",
            "ingest.explainability.missing_reason_code",
        ],
        CrawlIngestStage::CatalogMutation => &[
            "ingest.catalog.revision_non_monotonic",
            "ingest.catalog.idempotency_violation",
            "ingest.catalog.changelog_gap",
        ],
        CrawlIngestStage::QueueAdmission => &[
            "ingest.queue.starvation_detected",
            "ingest.backpressure.unbounded_growth",
            "ingest.replay.reordering_detected",
        ],
        CrawlIngestStage::EmbeddingGate => &[
            "ingest.embed.skip_policy_regression",
            "ingest.degrade.transition_invalid",
            "ingest.embed.queue_loss_detected",
        ],
    }
}

/// Snapshot of tuning levers for a single optimization iteration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeverSnapshot {
    /// Lever name/value pairs (sorted for deterministic output).
    pub values: BTreeMap<String, String>,
}

impl LeverSnapshot {
    /// Build a snapshot from key/value pairs.
    #[must_use]
    pub fn from_pairs<'a, I>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (&'a str, &'a str)>,
    {
        let values = pairs
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value.to_owned()))
            .collect();
        Self { values }
    }

    fn changed_levers(&self, other: &Self) -> Vec<String> {
        let mut key_union = BTreeSet::new();
        key_union.extend(self.values.keys().cloned());
        key_union.extend(other.values.keys().cloned());

        key_union
            .into_iter()
            .filter(|key| self.values.get(key) != other.values.get(key))
            .collect()
    }
}

/// Validation result for a one-lever optimization transition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IterationValidation {
    /// Whether the transition is accepted by protocol.
    pub accepted: bool,
    /// Sorted list of changed levers.
    pub changed_levers: Vec<String>,
    /// Machine-readable reason code.
    pub reason_code: String,
}

/// Enforces one-lever-at-a-time optimization discipline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OneLeverIterationProtocol;

impl OneLeverIterationProtocol {
    /// Validate that exactly one lever changed from `baseline` to `candidate`.
    #[must_use]
    pub fn validate(baseline: &LeverSnapshot, candidate: &LeverSnapshot) -> IterationValidation {
        let changed_levers = baseline.changed_levers(candidate);
        let (accepted, reason_code) = match changed_levers.len() {
            1 => (true, ITERATION_REASON_ACCEPTED),
            0 => (false, ITERATION_REASON_NO_CHANGE),
            _ => (false, ITERATION_REASON_MULTI_CHANGE),
        };

        IterationValidation {
            accepted,
            changed_levers,
            reason_code: reason_code.to_owned(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CRAWL_INGEST_OPT_TRACK_SCHEMA_VERSION, CorpusProfileSnapshot, CrawlIngestStage,
        HostProfileSnapshot, ITERATION_REASON_ACCEPTED, ITERATION_REASON_MULTI_CHANGE,
        ITERATION_REASON_NO_CHANGE, LeverSnapshot, MODEL_CACHE_DIAGNOSTICS_SCHEMA_VERSION,
        ModelCacheFallbackPath, ModelCacheState, OneLeverIterationProtocol, OpportunityCandidate,
        OpportunityMatrix, PROFILING_WORKFLOW_SCHEMA_VERSION, ProfileKind, ProfileWorkflow,
        RecommendedTwoTierConfig, SELF_CALIBRATING_PROFILE_SCHEMA_VERSION,
        SearchProfileMeasurements, SelfCalibratingProfileArtifactKind, SelfCalibratingProfileInput,
        SelfCalibratingProfileReport, crawl_ingest_opportunity_matrix,
        crawl_ingest_optimization_track, model_cache_cold_diagnostics_fixture,
        model_cache_diagnostics_contract_definition,
        model_cache_missing_offline_diagnostics_fixture, model_cache_warm_diagnostics_fixture,
    };
    use std::collections::BTreeSet;

    fn healthy_profile_input() -> SelfCalibratingProfileInput {
        SelfCalibratingProfileInput {
            run_id: "run-self-cal-001".to_owned(),
            host: HostProfileSnapshot {
                host_id: "host-ci".to_owned(),
                logical_cpus: 16,
                total_memory_mib: 64_000,
                available_memory_mib: 48_000,
                simd_lanes: 8,
                model_cache_state: ModelCacheState::Warm,
            },
            corpus: CorpusProfileSnapshot {
                corpus_id: "golden_100".to_owned(),
                document_count: 2_500,
                total_bytes: 12_800_000,
                cluster_count: 5,
                representative_query_count: 32,
            },
            measurements: SearchProfileMeasurements {
                phase1_p95_us: 8_200,
                phase2_p95_us: 142_000,
                vector_search_docs_per_ms: 1_200,
                peak_memory_mib: 1_024,
                observed_candidate_multiplier: 3,
                model_cache_warm_ms: 18,
                model_cache_cold_ms: 430,
                quality_uplift_basis_points: 1_300,
            },
            requested_limit: 10,
        }
    }

    #[test]
    fn crawl_ingest_matrix_ranking_is_deterministic() {
        let ranked_first = crawl_ingest_opportunity_matrix().ranked();
        let ranked_second = crawl_ingest_opportunity_matrix().ranked();
        assert_eq!(ranked_first.len(), ranked_second.len());
        for (left, right) in ranked_first.iter().zip(ranked_second.iter()) {
            assert_eq!(left.rank, right.rank);
            assert_eq!(left.candidate.id, right.candidate.id);
            assert_eq!(left.ice_score_per_mille, right.ice_score_per_mille);
        }
    }

    #[test]
    fn crawl_ingest_track_covers_hotspots_proofs_and_rollbacks() {
        let track = crawl_ingest_optimization_track();
        assert_eq!(track.schema_version, CRAWL_INGEST_OPT_TRACK_SCHEMA_VERSION);
        assert_eq!(track.hotspots.len(), 5);
        assert_eq!(track.proof_checklist.len(), track.hotspots.len());
        assert_eq!(track.rollback_guardrails.len(), track.hotspots.len());

        let hotspot_ids: BTreeSet<&str> = track
            .hotspots
            .iter()
            .map(|hotspot| hotspot.lever_id.as_str())
            .collect();

        for hotspot in &track.hotspots {
            assert!(hotspot.expected_p50_gain_pct > 0);
            assert!(hotspot.expected_p95_gain_pct >= hotspot.expected_p50_gain_pct);
            assert!(hotspot.expected_throughput_gain_pct > 0);
        }

        for item in &track.proof_checklist {
            assert!(hotspot_ids.contains(item.lever_id.as_str()));
            assert!(!item.required_invariants.is_empty());
            assert!(item.replay_command.contains("--lane ingest"));
        }

        for guardrail in &track.rollback_guardrails {
            assert!(hotspot_ids.contains(guardrail.lever_id.as_str()));
            assert!(guardrail.rollback_command.contains("fsfs profile rollback"));
            assert!(!guardrail.abort_reason_codes.is_empty());
            assert_eq!(guardrail.recovery_reason_code, "opt.rollback.completed");
        }
    }

    #[test]
    fn crawl_ingest_track_includes_all_expected_stages() {
        let track = crawl_ingest_optimization_track();
        let stages: BTreeSet<CrawlIngestStage> =
            track.hotspots.iter().map(|hotspot| hotspot.stage).collect();
        assert_eq!(
            stages,
            BTreeSet::from([
                CrawlIngestStage::DiscoveryWalk,
                CrawlIngestStage::Classification,
                CrawlIngestStage::CatalogMutation,
                CrawlIngestStage::QueueAdmission,
                CrawlIngestStage::EmbeddingGate,
            ])
        );
    }

    #[test]
    fn profiling_workflow_contains_required_lanes() {
        let workflow = ProfileWorkflow::for_dataset_profile("small");
        let kinds: Vec<ProfileKind> = workflow.steps.iter().map(|step| step.kind).collect();

        assert_eq!(workflow.schema_version, PROFILING_WORKFLOW_SCHEMA_VERSION);
        assert_eq!(
            kinds,
            vec![
                ProfileKind::Flamegraph,
                ProfileKind::Heap,
                ProfileKind::Syscall
            ]
        );
    }

    #[test]
    fn opportunity_matrix_ranking_is_deterministic() {
        let matrix = OpportunityMatrix::new(vec![
            OpportunityCandidate {
                id: "query-fusion".into(),
                summary: "Reduce query fusion allocations".into(),
                impact: 85,
                confidence: 90,
                effort: 25,
            },
            OpportunityCandidate {
                id: "crawl-io".into(),
                summary: "Reduce crawl syscall count".into(),
                impact: 75,
                confidence: 80,
                effort: 30,
            },
            OpportunityCandidate {
                id: "tui-diff".into(),
                summary: "Skip unnecessary frame redraws".into(),
                impact: 70,
                confidence: 95,
                effort: 15,
            },
        ]);

        let ranked = matrix.ranked();

        assert_eq!(ranked.len(), 3);
        assert_eq!(ranked[0].candidate.id, "tui-diff");
        assert_eq!(ranked[1].candidate.id, "query-fusion");
        assert_eq!(ranked[2].candidate.id, "crawl-io");
        assert!(ranked[0].ice_score_per_mille >= ranked[1].ice_score_per_mille);
        assert!(ranked[1].ice_score_per_mille >= ranked[2].ice_score_per_mille);
    }

    // ── ICE score edge cases ─────────────────────────────────────────

    #[test]
    fn ice_score_effort_zero_treated_as_one() {
        let candidate = OpportunityCandidate {
            id: "test".into(),
            summary: "zero-effort candidate".into(),
            impact: 80,
            confidence: 90,
            effort: 0,
        };
        // effort=0 should be treated as 1 to avoid division by zero
        assert_eq!(
            candidate.ice_score_per_mille(),
            80 * 90 * 1_000 // = 7_200_000
        );
    }

    #[test]
    fn ice_score_zero_impact_yields_zero() {
        let candidate = OpportunityCandidate {
            id: "no-impact".into(),
            summary: "no impact".into(),
            impact: 0,
            confidence: 100,
            effort: 10,
        };
        assert_eq!(candidate.ice_score_per_mille(), 0);
    }

    #[test]
    fn ice_score_zero_confidence_yields_zero() {
        let candidate = OpportunityCandidate {
            id: "no-confidence".into(),
            summary: "no confidence".into(),
            impact: 100,
            confidence: 0,
            effort: 10,
        };
        assert_eq!(candidate.ice_score_per_mille(), 0);
    }

    // ── LeverSnapshot changed_levers edge cases ─────────────────────

    #[test]
    fn lever_added_in_candidate_is_detected_as_change() {
        let baseline = LeverSnapshot::from_pairs([("a", "1")]);
        let candidate = LeverSnapshot::from_pairs([("a", "1"), ("b", "2")]);
        let validation = OneLeverIterationProtocol::validate(&baseline, &candidate);
        assert!(validation.accepted);
        assert_eq!(validation.changed_levers, vec!["b"]);
        assert_eq!(validation.reason_code, ITERATION_REASON_ACCEPTED);
    }

    #[test]
    fn lever_removed_in_candidate_is_detected_as_change() {
        let baseline = LeverSnapshot::from_pairs([("a", "1"), ("b", "2")]);
        let candidate = LeverSnapshot::from_pairs([("a", "1")]);
        let validation = OneLeverIterationProtocol::validate(&baseline, &candidate);
        assert!(validation.accepted);
        assert_eq!(validation.changed_levers, vec!["b"]);
        assert_eq!(validation.reason_code, ITERATION_REASON_ACCEPTED);
    }

    #[test]
    fn both_empty_snapshots_are_no_change() {
        let empty_a = LeverSnapshot::from_pairs(std::iter::empty::<(&str, &str)>());
        let empty_b = LeverSnapshot::from_pairs(std::iter::empty::<(&str, &str)>());
        let validation = OneLeverIterationProtocol::validate(&empty_a, &empty_b);
        assert!(!validation.accepted);
        assert_eq!(validation.reason_code, ITERATION_REASON_NO_CHANGE);
    }

    // ── ProfileWorkflow edge cases ──────────────────────────────────

    #[test]
    fn profile_workflow_empty_dataset_defaults_to_small() {
        let workflow = ProfileWorkflow::for_dataset_profile("");
        assert_eq!(workflow.dataset_profile, "small");
    }

    #[test]
    fn profile_workflow_whitespace_dataset_defaults_to_small() {
        let workflow = ProfileWorkflow::for_dataset_profile("   ");
        assert_eq!(workflow.dataset_profile, "small");
    }

    #[test]
    fn artifact_manifest_prefixes_run_id() {
        let workflow = ProfileWorkflow::for_dataset_profile("tiny");
        let artifacts = workflow.artifact_manifest("run-42");
        assert_eq!(artifacts.len(), 3);
        for artifact in &artifacts {
            assert!(
                artifact.artifact_path.starts_with("run-42/"),
                "artifact path should start with run id: {}",
                artifact.artifact_path
            );
            assert!(
                artifact.replay_command.contains("--run-id run-42"),
                "replay command should reference run id: {}",
                artifact.replay_command
            );
        }
    }

    // ── ProfileKind Display ─────────────────────────────────────────

    #[test]
    fn profile_kind_display_format() {
        assert_eq!(ProfileKind::Flamegraph.to_string(), "flamegraph");
        assert_eq!(ProfileKind::Heap.to_string(), "heap");
        assert_eq!(ProfileKind::Syscall.to_string(), "syscall");
    }

    // ── OpportunityMatrix edge cases ────────────────────────────────

    #[test]
    fn empty_matrix_ranked_returns_empty() {
        let matrix = OpportunityMatrix::new(vec![]);
        assert!(matrix.ranked().is_empty());
    }

    #[test]
    fn ranking_tiebreak_by_id_when_scores_equal() {
        let matrix = OpportunityMatrix::new(vec![
            OpportunityCandidate {
                id: "z-last".into(),
                summary: "z".into(),
                impact: 50,
                confidence: 50,
                effort: 25,
            },
            OpportunityCandidate {
                id: "a-first".into(),
                summary: "a".into(),
                impact: 50,
                confidence: 50,
                effort: 25,
            },
        ]);
        let ranked = matrix.ranked();
        assert_eq!(ranked[0].candidate.id, "a-first");
        assert_eq!(ranked[1].candidate.id, "z-last");
        assert_eq!(ranked[0].rank, 1);
        assert_eq!(ranked[1].rank, 2);
    }

    // ── Original tests continue ─────────────────────────────────────

    #[test]
    fn one_lever_protocol_accepts_exactly_one_change() {
        let baseline = LeverSnapshot::from_pairs([
            ("query.semantic_fanout", "64"),
            ("crawl.batch_size", "200"),
        ]);

        let accepted_candidate = LeverSnapshot::from_pairs([
            ("query.semantic_fanout", "80"),
            ("crawl.batch_size", "200"),
        ]);
        let accepted = OneLeverIterationProtocol::validate(&baseline, &accepted_candidate);
        assert!(accepted.accepted);
        assert_eq!(accepted.changed_levers, vec!["query.semantic_fanout"]);
        assert_eq!(accepted.reason_code, ITERATION_REASON_ACCEPTED);

        let unchanged = OneLeverIterationProtocol::validate(&baseline, &baseline);
        assert!(!unchanged.accepted);
        assert_eq!(unchanged.changed_levers.len(), 0);
        assert_eq!(unchanged.reason_code, ITERATION_REASON_NO_CHANGE);

        let multi_change_candidate = LeverSnapshot::from_pairs([
            ("query.semantic_fanout", "80"),
            ("crawl.batch_size", "100"),
        ]);
        let multi_change = OneLeverIterationProtocol::validate(&baseline, &multi_change_candidate);
        assert!(!multi_change.accepted);
        assert_eq!(
            multi_change.changed_levers,
            vec!["crawl.batch_size", "query.semantic_fanout"]
        );
        assert_eq!(multi_change.reason_code, ITERATION_REASON_MULTI_CHANGE);
    }

    #[test]
    fn self_calibrating_profile_recommends_quality_when_host_is_healthy() {
        let report = SelfCalibratingProfileReport::from_input(healthy_profile_input());

        assert_eq!(
            report.schema_version,
            SELF_CALIBRATING_PROFILE_SCHEMA_VERSION
        );
        assert!(!report.recommended_config.fast_only);
        assert_eq!(report.recommended_config.quality_weight_per_mille, 800);
        assert_eq!(report.recommended_config.candidate_multiplier, 3);
        assert_eq!(report.recommended_config.quality_timeout_ms, 167);
        assert_eq!(
            report.safe_fallback_defaults,
            RecommendedTwoTierConfig::safe_fallback_defaults()
        );
        assert!(report.confidence >= 90);
        assert!(
            report
                .reason_codes
                .contains(&"profile.self_calibrating.quality_enabled".to_owned())
        );
    }

    #[test]
    fn self_calibrating_profile_constrains_degraded_host_to_fast_only() {
        let mut input = healthy_profile_input();
        input.host.available_memory_mib = 4_000;
        input.host.model_cache_state = ModelCacheState::Missing;
        input.measurements.phase2_p95_us = 260_000;
        input.measurements.peak_memory_mib = 3_900;
        input.measurements.quality_uplift_basis_points = 250;

        let report = SelfCalibratingProfileReport::from_input(input);

        assert!(report.recommended_config.fast_only);
        assert_eq!(report.recommended_config.quality_weight_per_mille, 0);
        assert_eq!(report.recommended_config.quality_timeout_ms, 10);
        assert!(report.confidence < 80);
        assert!(
            report
                .reason_codes
                .contains(&"profile.self_calibrating.fast_only".to_owned())
        );
        assert!(
            report
                .reason_codes
                .contains(&"profile.self_calibrating.low_memory_headroom".to_owned())
        );
    }

    #[test]
    fn self_calibrating_profile_adjusts_large_corpus_budget_and_ann_threshold() {
        let mut input = healthy_profile_input();
        input.corpus.corpus_id = "repo_large".to_owned();
        input.corpus.document_count = 125_000;
        input.measurements.vector_search_docs_per_ms = 1_100;
        input.measurements.phase1_p95_us = 9_500;

        let report = SelfCalibratingProfileReport::from_input(input);

        assert_eq!(report.recommended_config.candidate_multiplier, 4);
        assert_eq!(report.recommended_config.rrf_k, 75);
        assert_eq!(report.recommended_config.hnsw_threshold, 10_000);
        assert_eq!(report.recommended_config.mrl_search_dims, 0);
        assert!(
            report
                .reason_codes
                .contains(&"profile.self_calibrating.large_corpus_ann_threshold".to_owned())
        );
        assert!(
            report
                .reason_codes
                .contains(&"profile.self_calibrating.candidate_budget_expanded".to_owned())
        );
    }

    #[test]
    fn self_calibrating_profile_artifacts_include_jsonl_and_replay_command() {
        let report = SelfCalibratingProfileReport::from_input(healthy_profile_input());
        let artifact_kinds: BTreeSet<SelfCalibratingProfileArtifactKind> = report
            .artifacts
            .iter()
            .map(|artifact| artifact.kind)
            .collect();

        assert_eq!(
            artifact_kinds,
            BTreeSet::from([
                SelfCalibratingProfileArtifactKind::StructuredJsonl,
                SelfCalibratingProfileArtifactKind::RecommendationJson,
                SelfCalibratingProfileArtifactKind::ReplayManifest,
            ])
        );
        assert!(
            report
                .artifacts
                .iter()
                .any(|artifact| artifact.content_type == "application/jsonl")
        );
        for artifact in &report.artifacts {
            assert!(artifact.artifact_path.contains("run-self-cal-001"));
            assert!(artifact.artifact_path.contains("golden_100"));
            assert!(
                artifact
                    .replay_command
                    .contains("check_fsfs_self_calibrating_profile.sh --mode e2e")
            );
        }
    }

    #[test]
    fn self_calibrating_profile_report_serializes_stably() {
        let report = SelfCalibratingProfileReport::from_input(healthy_profile_input());
        let encoded = serde_json::to_string(&report).expect("serialize profile report");
        let decoded: SelfCalibratingProfileReport =
            serde_json::from_str(&encoded).expect("deserialize profile report");

        assert_eq!(decoded, report);
        assert_eq!(decoded.profile.corpus.average_document_bytes(), 5_120);
        assert_eq!(decoded.profile.host.memory_headroom_pct(), 75);
    }

    #[test]
    fn model_cache_diagnostics_contract_covers_states_and_operator_knobs() {
        let contract = model_cache_diagnostics_contract_definition();
        contract
            .validate()
            .expect("model-cache diagnostics contract should validate");

        let states: BTreeSet<ModelCacheState> =
            contract.state_rules.iter().map(|rule| rule.state).collect();
        assert_eq!(
            states,
            BTreeSet::from([
                ModelCacheState::Warm,
                ModelCacheState::Cold,
                ModelCacheState::Missing,
                ModelCacheState::Unknown,
            ])
        );
        assert_eq!(contract.v, MODEL_CACHE_DIAGNOSTICS_SCHEMA_VERSION);
        assert!(
            contract
                .advice_env_vars
                .contains(&"FRANKENSEARCH_MODEL_DIR".to_owned())
        );
        assert!(
            contract
                .advice_config_knobs
                .contains(&"indexing.model_dir".to_owned())
        );
        assert!(!contract.network_required_in_tests);
    }

    #[test]
    fn model_cache_diagnostics_validate_warm_cold_and_missing_offline() {
        let warm = model_cache_warm_diagnostics_fixture();
        warm.validate().expect("warm fixture should validate");
        assert_eq!(warm.diagnostics.state, ModelCacheState::Warm);
        assert_eq!(
            warm.diagnostics.fallback_path,
            ModelCacheFallbackPath::QualityModel
        );

        let cold = model_cache_cold_diagnostics_fixture();
        cold.validate().expect("cold fixture should validate");
        assert_eq!(cold.diagnostics.state, ModelCacheState::Cold);
        assert!(
            cold.advice
                .iter()
                .any(|advice| advice.config_knob == "search.fast_only")
        );

        let missing = model_cache_missing_offline_diagnostics_fixture();
        missing
            .validate()
            .expect("missing offline fixture should validate");
        assert_eq!(missing.diagnostics.state, ModelCacheState::Missing);
        assert!(missing.diagnostics.offline);
        assert_eq!(
            missing.diagnostics.fallback_path,
            ModelCacheFallbackPath::HashEmbedder
        );
    }

    #[test]
    fn model_cache_diagnostics_reject_raw_paths_and_missing_advice() {
        let mut raw_path = model_cache_warm_diagnostics_fixture();
        raw_path.diagnostics.artifact.redacted_model_dir =
            "/home/ubuntu/.cache/frankensearch/models".to_owned();
        let error = raw_path.validate().expect_err("raw path should fail");
        assert!(error.contains("redacted"));

        let mut missing_advice = model_cache_cold_diagnostics_fixture();
        missing_advice.advice.clear();
        let error = missing_advice
            .validate()
            .expect_err("missing advice should fail");
        assert!(error.contains("operator advice"));
    }
}
