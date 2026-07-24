//! Non-interfering lexical shadow-oracle support.
//!
//! [`ShadowLexical`] wraps the existing [`LexicalSearch`] seam. The serving
//! backend is always awaited and returned unchanged. Eligible shadow work is
//! detached into the caller's asupersync region behind a bounded admission
//! gate, so sampling, pressure shedding, a full gate, spawn failure, shadow
//! failure, or artifact failure cannot change the serving result.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as FmtWrite;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::{Instrument, info, info_span, warn};

use crate::{
    Cx, IndexableDocument, LexicalSearch, ScoredResult, SearchError, SearchFuture, SearchResult,
};

/// Version of one production divergence JSONL record.
pub const SHADOW_DIVERGENCE_SCHEMA_VERSION: u32 = 2;
/// Version of one typed shadow degradation JSONL record.
pub const SHADOW_DEGRADATION_SCHEMA_VERSION: u32 = 1;
/// Version of one completed shadow observation JSONL record.
pub const SHADOW_OBSERVATION_SCHEMA_VERSION: u32 = 1;
/// Fixed-point sampling denominator. One basis point is 1/10,000.
pub const SHADOW_SAMPLE_DENOMINATOR: u64 = 10_000;
/// Directory beneath an index root that owns shadow evidence.
pub const SHADOW_ARTIFACT_DIRECTORY: &str = ".quill-shadow";
/// Append-only classified divergence stream.
pub const SHADOW_DIVERGENCES_FILE: &str = "divergences.jsonl";
/// Append-only typed degradation stream.
pub const SHADOW_DEGRADATIONS_FILE: &str = "degradations.jsonl";
/// Append-only completed-comparison evidence stream.
pub const SHADOW_OBSERVATIONS_FILE: &str = "observations.jsonl";

const CORPUS_HASH_DOMAIN: &[u8] = b"frankensearch/shadow-corpus/v1\0";

/// Runtime policy for one shadow wrapper.
#[derive(Debug, Clone, PartialEq)]
pub struct ShadowLexicalConfig {
    /// Explicit opt-in. The default is off.
    pub enabled: bool,
    /// Deterministic sample rate in basis points.
    pub sample_rate_basis_points: u16,
    /// Maximum number of admitted background comparisons.
    pub max_in_flight: usize,
    /// Maximum accepted absolute score delta before score divergence.
    pub score_epsilon: f32,
    /// Evidence directory, normally `<index>/.quill-shadow`.
    pub artifact_directory: PathBuf,
    /// Initial committed manifest generation.
    pub initial_generation: u64,
}

impl ShadowLexicalConfig {
    /// Construct a default-off policy rooted beneath `index_root`.
    #[must_use]
    pub fn for_index_root(index_root: impl AsRef<Path>) -> Self {
        Self {
            artifact_directory: index_root.as_ref().join(SHADOW_ARTIFACT_DIRECTORY),
            ..Self::default()
        }
    }

    fn validate(&self) -> SearchResult<()> {
        if u64::from(self.sample_rate_basis_points) > SHADOW_SAMPLE_DENOMINATOR {
            return Err(SearchError::InvalidConfig {
                field: "search.shadow_sample_rate_basis_points".to_owned(),
                value: self.sample_rate_basis_points.to_string(),
                reason: format!("expected 0..={SHADOW_SAMPLE_DENOMINATOR}"),
            });
        }
        if self.max_in_flight == 0 {
            return Err(SearchError::InvalidConfig {
                field: "search.shadow_max_in_flight".to_owned(),
                value: self.max_in_flight.to_string(),
                reason: "expected at least one background task".to_owned(),
            });
        }
        if !self.score_epsilon.is_finite() || self.score_epsilon < 0.0 {
            return Err(SearchError::InvalidConfig {
                field: "search.shadow_score_epsilon".to_owned(),
                value: self.score_epsilon.to_string(),
                reason: "expected a finite non-negative value".to_owned(),
            });
        }
        if self.artifact_directory.as_os_str().is_empty() {
            return Err(SearchError::InvalidConfig {
                field: "search.shadow_artifact_directory".to_owned(),
                value: String::new(),
                reason: "expected a non-empty path".to_owned(),
            });
        }
        Ok(())
    }
}

impl Default for ShadowLexicalConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            sample_rate_basis_points: 10_000,
            max_in_flight: 2,
            score_epsilon: 1.0e-5,
            artifact_directory: PathBuf::from(SHADOW_ARTIFACT_DIRECTORY),
            initial_generation: 0,
        }
    }
}

/// Reason a sampled shadow comparison was shed before admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShadowShedReason {
    /// Host CPU pressure crossed the configured ceiling.
    CpuPressure,
    /// Host I/O pressure crossed the configured ceiling.
    IoPressure,
    /// An operator or lifecycle controller paused comparisons.
    ManualPause,
    /// The bounded background-task gate was full.
    Capacity,
    /// The shadow index is known to be incomplete or failed.
    ShadowDegraded,
    /// The caller did not provide a spawn-capable runtime context.
    RuntimeUnavailable,
    /// Index generation changed while the comparison ran.
    GenerationDrift,
}

impl ShadowShedReason {
    const fn label(self) -> &'static str {
        match self {
            Self::CpuPressure => "cpu_pressure",
            Self::IoPressure => "io_pressure",
            Self::ManualPause => "manual_pause",
            Self::Capacity => "capacity",
            Self::ShadowDegraded => "shadow_degraded",
            Self::RuntimeUnavailable => "runtime_unavailable",
            Self::GenerationDrift => "generation_drift",
        }
    }
}

/// Synchronous, allocation-free load-shed probe used before task admission.
pub trait ShadowLoadProbe: Send + Sync {
    /// Return a shed reason, or `None` when shadow work may be admitted.
    fn shed_reason(&self) -> Option<ShadowShedReason>;
}

/// Probe that admits every sampled comparison.
#[derive(Debug, Default)]
pub struct AlwaysAdmitShadowLoad;

impl ShadowLoadProbe for AlwaysAdmitShadowLoad {
    fn shed_reason(&self) -> Option<ShadowShedReason> {
        None
    }
}

/// Mutable probe for host pressure controllers and deterministic tests.
#[derive(Debug, Default)]
pub struct AtomicShadowLoadProbe {
    cpu_pressure: AtomicBool,
    io_pressure: AtomicBool,
    manually_paused: AtomicBool,
}

impl AtomicShadowLoadProbe {
    /// Update the CPU pressure gate.
    pub fn set_cpu_pressure(&self, pressured: bool) {
        self.cpu_pressure.store(pressured, Ordering::Release);
    }

    /// Update the I/O pressure gate.
    pub fn set_io_pressure(&self, pressured: bool) {
        self.io_pressure.store(pressured, Ordering::Release);
    }

    /// Pause or resume shadow admission without affecting serving.
    pub fn set_manually_paused(&self, paused: bool) {
        self.manually_paused.store(paused, Ordering::Release);
    }
}

impl ShadowLoadProbe for AtomicShadowLoadProbe {
    fn shed_reason(&self) -> Option<ShadowShedReason> {
        if self.manually_paused.load(Ordering::Acquire) {
            Some(ShadowShedReason::ManualPause)
        } else if self.cpu_pressure.load(Ordering::Acquire) {
            Some(ShadowShedReason::CpuPressure)
        } else if self.io_pressure.load(Ordering::Acquire) {
            Some(ShadowShedReason::IoPressure)
        } else {
            None
        }
    }
}

/// Engine-neutral document carried by an exact-replay shadow artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShadowDocument {
    /// Stable document identifier.
    pub id: String,
    /// Searchable content.
    pub content: String,
    /// Optional boosted title.
    pub title: Option<String>,
    /// Canonically ordered metadata.
    pub metadata: BTreeMap<String, String>,
}

impl From<&IndexableDocument> for ShadowDocument {
    fn from(document: &IndexableDocument) -> Self {
        Self {
            id: document.id.clone(),
            content: document.content.clone(),
            title: document.title.clone(),
            metadata: document
                .metadata
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        }
    }
}

impl From<ShadowDocument> for IndexableDocument {
    fn from(document: ShadowDocument) -> Self {
        Self {
            id: document.id,
            content: document.content,
            title: document.title,
            metadata: document.metadata.into_iter().collect(),
        }
    }
}

/// Query identity recorded for shrink and exact-generation replay.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShadowQuery {
    /// Raw query text supplied to both engines.
    pub text: String,
    /// Requested top-k limit.
    pub limit: usize,
    /// Whether the fusion-candidate path was exercised.
    pub fusion_candidates: bool,
}

/// One top-k result with exact floating-point witnesses.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShadowRankedHit {
    /// One-based rank.
    pub rank: usize,
    /// Stable document identifier.
    pub document_id: String,
    /// Exact result score bits.
    pub score_bits: u32,
    /// Exact lexical score bits, when supplied separately.
    pub lexical_score_bits: Option<u32>,
}

impl ShadowRankedHit {
    fn from_result(rank: usize, result: &ScoredResult) -> Self {
        Self {
            rank,
            document_id: result.doc_id.to_string(),
            score_bits: result.score.to_bits(),
            lexical_score_bits: result.lexical_score.map(f32::to_bits),
        }
    }

    fn score(&self) -> f32 {
        f32::from_bits(self.score_bits)
    }
}

/// Stable classification emitted by the production shadow comparator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShadowDivergenceClass {
    /// Identical document order and exact score bits.
    Exact,
    /// Identical order with score deltas inside the configured epsilon.
    ScoreEpsilon,
    /// The same document set was returned in a different order.
    TieOrder,
    /// Document identities or result cardinality differed.
    RankMismatch,
    /// Document order matched but a score exceeded the epsilon.
    ScoreMismatch,
}

impl ShadowDivergenceClass {
    const fn label(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::ScoreEpsilon => "score_epsilon",
            Self::TieOrder => "tie_order",
            Self::RankMismatch => "rank_mismatch",
            Self::ScoreMismatch => "score_mismatch",
        }
    }

    const fn is_artifact(self) -> bool {
        !matches!(self, Self::Exact)
    }
}

/// Versioned production record written to
/// `.quill-shadow/divergences.jsonl`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShadowDivergenceRecord {
    /// Record schema version.
    pub schema_version: u32,
    /// Committed manifest generation observed by both queries.
    pub manifest_generation: u64,
    /// SHA-256 identity of the replay corpus.
    pub corpus_hash: String,
    /// Full corpus required by the shrinker and exact replay.
    pub documents: Vec<ShadowDocument>,
    /// Query supplied to both engines.
    pub query: ShadowQuery,
    /// Untouched serving top-k list.
    pub serving_top_k: Vec<ShadowRankedHit>,
    /// Background shadow top-k list.
    pub shadow_top_k: Vec<ShadowRankedHit>,
    /// Classified comparison result.
    pub classification: ShadowDivergenceClass,
    /// Serving latency measured before shadow admission.
    pub serve_latency_micros: u64,
    /// Background shadow latency.
    pub shadow_latency_micros: u64,
}

impl ShadowDivergenceRecord {
    /// Validate the versioned artifact schema and replay invariants.
    ///
    /// # Errors
    ///
    /// Returns a stable reason for an unsupported version, malformed corpus
    /// hash, duplicate document identity, invalid rank sequence, non-finite
    /// score, or an `exact` record in the divergence-only stream.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != SHADOW_DIVERGENCE_SCHEMA_VERSION {
            return Err(format!("unsupported shadow schema {}", self.schema_version));
        }
        if self.corpus_hash.len() != 64
            || !self
                .corpus_hash
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err("corpus_hash must be 64 hexadecimal characters".to_owned());
        }
        if self.documents.is_empty() {
            return Err("documents must not be empty".to_owned());
        }
        let mut document_ids = BTreeSet::new();
        for document in &self.documents {
            if document.id.is_empty() {
                return Err("documents contain an empty id".to_owned());
            }
            if !document_ids.insert(document.id.as_str()) {
                return Err(format!("duplicate document id {:?}", document.id));
            }
        }
        if self
            .documents
            .windows(2)
            .any(|pair| pair[0].id >= pair[1].id)
        {
            return Err("documents must be ordered by ascending id".to_owned());
        }
        if self.corpus_hash != compute_shadow_corpus_hash(&self.documents) {
            return Err("corpus_hash does not match documents".to_owned());
        }
        validate_ranked_hits("serving_top_k", &self.serving_top_k)?;
        validate_ranked_hits("shadow_top_k", &self.shadow_top_k)?;
        if !self.classification.is_artifact() {
            return Err("divergence stream cannot contain exact comparisons".to_owned());
        }
        Ok(())
    }
}

fn validate_ranked_hits(label: &str, hits: &[ShadowRankedHit]) -> Result<(), String> {
    let mut identities = BTreeSet::new();
    for (index, hit) in hits.iter().enumerate() {
        if hit.rank != index.saturating_add(1) {
            return Err(format!("{label} ranks must be contiguous and one-based"));
        }
        if hit.document_id.is_empty() || !identities.insert(hit.document_id.as_str()) {
            return Err(format!(
                "{label} contains an empty or duplicate document id"
            ));
        }
        if !hit.score().is_finite()
            || hit
                .lexical_score_bits
                .is_some_and(|bits| !f32::from_bits(bits).is_finite())
        {
            return Err(format!("{label} contains a non-finite score"));
        }
    }
    Ok(())
}

/// Typed shadow failure written separately from classified divergences.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShadowDegradationKind {
    /// The configured shadow oracle could not be found or opened.
    OracleUnavailable,
    /// The oracle corpus could not be exported for exact replay.
    CorpusExport,
    /// Shadow document indexing failed.
    Index,
    /// Shadow commit failed after the serving commit succeeded.
    Commit,
    /// Background shadow search returned an error.
    Search,
    /// No spawn-capable runtime was attached to the request context.
    RuntimeUnavailable,
    /// A divergence artifact could not be persisted.
    ArtifactWrite,
    /// Generation or corpus revision changed during comparison.
    GenerationDrift,
    /// Host pressure sampling failed; cached admission state remains in use.
    PressureSample,
}

/// Append-only, doctor-readable shadow degradation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShadowDegradationRecord {
    /// Record schema version.
    pub schema_version: u32,
    /// Committed generation at the failure boundary.
    pub manifest_generation: u64,
    /// Typed failure class.
    pub kind: ShadowDegradationKind,
    /// Stable human-readable detail.
    pub detail: String,
}

/// One completed comparison, including clean observations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShadowObservationRecord {
    /// Record schema version.
    pub schema_version: u32,
    /// Committed generation shared by both engines.
    pub manifest_generation: u64,
    /// Raw query text.
    pub query: String,
    /// Classified result.
    pub classification: ShadowDivergenceClass,
    /// Serving latency measured before background admission.
    pub serve_latency_micros: u64,
    /// Background shadow latency.
    pub shadow_latency_micros: u64,
}

/// Cross-process evidence summary consumed by doctor/status surfaces.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ShadowArtifactSummary {
    /// Number of completed, schema-valid observations.
    pub observation_count: u64,
    /// Number of schema-valid classified divergence records.
    pub divergence_count: u64,
    /// Number of schema-valid typed degradation records.
    pub degradation_count: u64,
    /// Number of malformed JSONL lines across all three streams.
    pub malformed_line_count: u64,
    /// Highest generation found in either stream.
    pub latest_generation: Option<u64>,
}

impl ShadowArtifactSummary {
    /// Read evidence below `<index_root>/.quill-shadow`.
    ///
    /// Missing streams are treated as empty. Other I/O failures are surfaced.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when an existing stream cannot be read.
    pub fn read_index_root(index_root: impl AsRef<Path>) -> io::Result<Self> {
        Self::read_artifact_directory(index_root.as_ref().join(SHADOW_ARTIFACT_DIRECTORY))
    }

    /// Read evidence directly from an artifact directory.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when an existing stream cannot be read.
    pub fn read_artifact_directory(directory: impl AsRef<Path>) -> io::Result<Self> {
        let directory = directory.as_ref();
        let mut summary = Self::default();
        read_jsonl_if_present(&directory.join(SHADOW_OBSERVATIONS_FILE), |line| {
            match serde_json::from_str::<ShadowObservationRecord>(line) {
                Ok(record) if record.schema_version == SHADOW_OBSERVATION_SCHEMA_VERSION => {
                    summary.observation_count = summary.observation_count.saturating_add(1);
                    summary.latest_generation = Some(
                        summary
                            .latest_generation
                            .map_or(record.manifest_generation, |generation| {
                                generation.max(record.manifest_generation)
                            }),
                    );
                }
                _ => {
                    summary.malformed_line_count = summary.malformed_line_count.saturating_add(1);
                }
            }
        })?;
        read_jsonl_if_present(&directory.join(SHADOW_DIVERGENCES_FILE), |line| {
            match serde_json::from_str::<ShadowDivergenceRecord>(line) {
                Ok(record) if record.validate().is_ok() => {
                    summary.divergence_count = summary.divergence_count.saturating_add(1);
                    summary.latest_generation = Some(
                        summary
                            .latest_generation
                            .map_or(record.manifest_generation, |generation| {
                                generation.max(record.manifest_generation)
                            }),
                    );
                }
                _ => {
                    summary.malformed_line_count = summary.malformed_line_count.saturating_add(1);
                }
            }
        })?;
        read_jsonl_if_present(&directory.join(SHADOW_DEGRADATIONS_FILE), |line| {
            match serde_json::from_str::<ShadowDegradationRecord>(line) {
                Ok(record) if record.schema_version == SHADOW_DEGRADATION_SCHEMA_VERSION => {
                    summary.degradation_count = summary.degradation_count.saturating_add(1);
                    summary.latest_generation = Some(
                        summary
                            .latest_generation
                            .map_or(record.manifest_generation, |generation| {
                                generation.max(record.manifest_generation)
                            }),
                    );
                }
                _ => {
                    summary.malformed_line_count = summary.malformed_line_count.saturating_add(1);
                }
            }
        })?;
        Ok(summary)
    }
}

fn read_jsonl_if_present(path: &Path, mut visit: impl FnMut(&str)) -> io::Result<()> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    };
    for line in BufReader::new(file).lines() {
        let line = line?;
        if !line.trim().is_empty() {
            visit(&line);
        }
    }
    Ok(())
}

/// In-process counters for sampling, load shedding, and lifecycle proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShadowStatus {
    /// Current opt-in state.
    pub enabled: bool,
    /// Current committed generation.
    pub manifest_generation: u64,
    /// Current deterministic corpus hash.
    pub corpus_hash: String,
    /// Currently admitted background comparisons.
    pub in_flight: usize,
    /// Queries selected by the deterministic sampler.
    pub sampled: u64,
    /// Queries rejected by the deterministic sampler.
    pub sampled_out: u64,
    /// Sampled queries shed by pressure, capacity, or lifecycle state.
    pub shed: u64,
    /// Completed background comparisons.
    pub completed: u64,
    /// Classified divergence artifacts attempted.
    pub divergences: u64,
    /// Shadow errors or lifecycle degradations.
    pub degradations: u64,
}

#[derive(Debug, Clone, Copy)]
enum ShadowSearchPath {
    Search,
    FusionCandidates,
}

impl ShadowSearchPath {
    const fn fusion_candidates(self) -> bool {
        matches!(self, Self::FusionCandidates)
    }
}

struct ShadowArtifactSink {
    directory: PathBuf,
    append_lock: Mutex<()>,
}

impl ShadowArtifactSink {
    fn new(directory: PathBuf) -> Self {
        Self {
            directory,
            append_lock: Mutex::new(()),
        }
    }

    fn append_divergence(&self, record: &ShadowDivergenceRecord) -> io::Result<()> {
        record.validate().map_err(io::Error::other)?;
        self.append_json(SHADOW_DIVERGENCES_FILE, record)
    }

    fn append_degradation(&self, record: &ShadowDegradationRecord) -> io::Result<()> {
        self.append_json(SHADOW_DEGRADATIONS_FILE, record)
    }

    fn append_observation(&self, record: &ShadowObservationRecord) -> io::Result<()> {
        self.append_json(SHADOW_OBSERVATIONS_FILE, record)
    }

    fn append_json(&self, name: &str, value: &impl Serialize) -> io::Result<()> {
        let _guard = self
            .append_lock
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        fs::create_dir_all(&self.directory)?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.directory.join(name))?;
        serde_json::to_writer(&mut file, value).map_err(io::Error::other)?;
        file.write_all(b"\n")?;
        file.flush()?;
        file.sync_data()
    }
}

/// Append one typed preparation or lifecycle degradation.
///
/// This is used by serving adapters before a [`ShadowLexical`] or
/// [`ShadowLexicalObserver`] can be constructed. The directory is the
/// artifact directory itself, normally `<index>/.quill-shadow`.
///
/// # Errors
///
/// Returns an I/O error when the append-only JSONL record cannot be persisted.
pub fn append_shadow_degradation(
    artifact_directory: impl Into<PathBuf>,
    manifest_generation: u64,
    kind: ShadowDegradationKind,
    detail: impl Into<String>,
) -> io::Result<()> {
    ShadowArtifactSink::new(artifact_directory.into()).append_degradation(
        &ShadowDegradationRecord {
            schema_version: SHADOW_DEGRADATION_SCHEMA_VERSION,
            manifest_generation,
            kind,
            detail: detail.into(),
        },
    )
}

struct ShadowState {
    enabled: AtomicBool,
    sample_rate_basis_points: u16,
    max_in_flight: usize,
    score_epsilon: f32,
    sample_ordinal: AtomicU64,
    in_flight: AtomicUsize,
    sampled: AtomicU64,
    sampled_out: AtomicU64,
    shed: AtomicU64,
    completed: AtomicU64,
    divergences: AtomicU64,
    degradations: AtomicU64,
    manifest_generation: AtomicU64,
    corpus_revision: AtomicU64,
    shadow_ready: AtomicBool,
    corpus: Mutex<BTreeMap<String, ShadowDocument>>,
    sink: ShadowArtifactSink,
    load_probe: Arc<dyn ShadowLoadProbe>,
}

impl ShadowState {
    fn corpus_snapshot(&self) -> (String, Vec<ShadowDocument>) {
        let documents = self
            .corpus
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .values()
            .cloned()
            .collect::<Vec<_>>();
        (compute_shadow_corpus_hash(&documents), documents)
    }

    fn record_shed(&self, reason: ShadowShedReason, query: &str, generation: u64) {
        self.shed.fetch_add(1, Ordering::Relaxed);
        info!(
            event = "shadow_query_shed",
            query,
            manifest_generation = generation,
            shed_reason = reason.label(),
            "shadow lexical query was shed without affecting serving"
        );
    }

    fn record_degradation(
        &self,
        generation: u64,
        kind: ShadowDegradationKind,
        detail: impl Into<String>,
        persist: bool,
    ) {
        let detail = detail.into();
        self.degradations.fetch_add(1, Ordering::Relaxed);
        warn!(
            event = "shadow_degraded",
            manifest_generation = generation,
            degradation_kind = ?kind,
            detail,
            "shadow lexical failure was isolated from serving"
        );
        if persist {
            let record = ShadowDegradationRecord {
                schema_version: SHADOW_DEGRADATION_SCHEMA_VERSION,
                manifest_generation: generation,
                kind,
                detail,
            };
            if let Err(error) = self.sink.append_degradation(&record) {
                warn!(
                    event = "shadow_artifact_write_failed",
                    error = %error,
                    "shadow degradation record could not be persisted"
                );
            }
        }
    }

    fn try_admit(self: &Arc<Self>) -> Option<InFlightGuard> {
        let mut observed = self.in_flight.load(Ordering::Acquire);
        loop {
            if observed >= self.max_in_flight {
                return None;
            }
            match self.in_flight.compare_exchange_weak(
                observed,
                observed.saturating_add(1),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return Some(InFlightGuard {
                        state: Arc::clone(self),
                    });
                }
                Err(actual) => observed = actual,
            }
        }
    }
}

struct InFlightGuard {
    state: Arc<ShadowState>,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.state.in_flight.fetch_sub(1, Ordering::AcqRel);
    }
}

fn build_shadow_state(
    config: ShadowLexicalConfig,
    load_probe: Arc<dyn ShadowLoadProbe>,
) -> SearchResult<Arc<ShadowState>> {
    config.validate()?;
    Ok(Arc::new(ShadowState {
        enabled: AtomicBool::new(config.enabled),
        sample_rate_basis_points: config.sample_rate_basis_points,
        max_in_flight: config.max_in_flight,
        score_epsilon: config.score_epsilon,
        sample_ordinal: AtomicU64::new(0),
        in_flight: AtomicUsize::new(0),
        sampled: AtomicU64::new(0),
        sampled_out: AtomicU64::new(0),
        shed: AtomicU64::new(0),
        completed: AtomicU64::new(0),
        divergences: AtomicU64::new(0),
        degradations: AtomicU64::new(0),
        manifest_generation: AtomicU64::new(config.initial_generation),
        corpus_revision: AtomicU64::new(0),
        shadow_ready: AtomicBool::new(true),
        corpus: Mutex::new(BTreeMap::new()),
        sink: ShadowArtifactSink::new(config.artifact_directory),
        load_probe,
    }))
}

fn seed_shadow_corpus(
    state: &Arc<ShadowState>,
    documents: &[IndexableDocument],
    manifest_generation: u64,
) {
    let mut corpus = state.corpus.lock().unwrap_or_else(PoisonError::into_inner);
    corpus.clear();
    corpus.extend(
        documents
            .iter()
            .map(|document| (document.id.clone(), ShadowDocument::from(document))),
    );
    drop(corpus);
    state
        .manifest_generation
        .store(manifest_generation, Ordering::Release);
    state.corpus_revision.fetch_add(1, Ordering::AcqRel);
}

fn shadow_status(state: &Arc<ShadowState>) -> ShadowStatus {
    let (corpus_hash, _) = state.corpus_snapshot();
    ShadowStatus {
        enabled: state.enabled.load(Ordering::Acquire),
        manifest_generation: state.manifest_generation.load(Ordering::Acquire),
        corpus_hash,
        in_flight: state.in_flight.load(Ordering::Acquire),
        sampled: state.sampled.load(Ordering::Relaxed),
        sampled_out: state.sampled_out.load(Ordering::Relaxed),
        shed: state.shed.load(Ordering::Relaxed),
        completed: state.completed.load(Ordering::Relaxed),
        divergences: state.divergences.load(Ordering::Relaxed),
        degradations: state.degradations.load(Ordering::Relaxed),
    }
}

/// Query-only bridge for serving adapters whose richer direct result API
/// cannot itself implement [`LexicalSearch`].
///
/// The bridge accepts an already-produced serving result slice and schedules
/// the same bounded comparison used by [`ShadowLexical`]. It exists for fsfs's
/// snippet-preserving direct Quill path; the serving results are never changed.
#[derive(Clone)]
pub struct ShadowLexicalObserver {
    shadow: Arc<dyn LexicalSearch>,
    state: Arc<ShadowState>,
}

impl ShadowLexicalObserver {
    /// Build a query-only observer.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] for an invalid policy.
    pub fn new(shadow: Arc<dyn LexicalSearch>, config: ShadowLexicalConfig) -> SearchResult<Self> {
        Self::with_load_probe(shadow, config, Arc::new(AlwaysAdmitShadowLoad))
    }

    /// Build an observer with an injected pressure probe.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] for an invalid policy.
    pub fn with_load_probe(
        shadow: Arc<dyn LexicalSearch>,
        config: ShadowLexicalConfig,
        load_probe: Arc<dyn ShadowLoadProbe>,
    ) -> SearchResult<Self> {
        Ok(Self {
            shadow,
            state: build_shadow_state(config, load_probe)?,
        })
    }

    /// Seed the corpus required by exact-generation replay artifacts.
    pub fn seed_corpus(&self, documents: &[IndexableDocument], manifest_generation: u64) {
        seed_shadow_corpus(&self.state, documents, manifest_generation);
    }

    /// Mark the query-only oracle unhealthy at a successor serving generation.
    ///
    /// Existing tasks observe generation drift, and later queries are shed
    /// until the adapter constructs a matching oracle snapshot.
    pub fn mark_generation_degraded(&self, manifest_generation: u64, detail: impl Into<String>) {
        self.state
            .manifest_generation
            .store(manifest_generation, Ordering::Release);
        self.state.shadow_ready.store(false, Ordering::Release);
        self.state.record_degradation(
            manifest_generation,
            ShadowDegradationKind::GenerationDrift,
            detail,
            true,
        );
    }

    /// Persist a typed adapter-level failure without affecting serving.
    pub fn record_degradation(&self, kind: ShadowDegradationKind, detail: impl Into<String>) {
        self.state.record_degradation(
            self.state.manifest_generation.load(Ordering::Acquire),
            kind,
            detail,
            true,
        );
    }

    /// Submit already-produced serving results without awaiting shadow work.
    pub fn observe_serving_results(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        serving_results: &[ScoredResult],
        fusion_candidates: bool,
        serve_latency_micros: u64,
    ) {
        ShadowLexical::submit_shadow_parts(
            &self.shadow,
            &self.state,
            cx,
            query,
            limit,
            serving_results,
            if fusion_candidates {
                ShadowSearchPath::FusionCandidates
            } else {
                ShadowSearchPath::Search
            },
            serve_latency_micros,
        );
    }

    /// Snapshot in-process evidence counters.
    #[must_use]
    pub fn status(&self) -> ShadowStatus {
        shadow_status(&self.state)
    }
}

/// A serving-first lexical adapter with bounded, region-owned shadow queries.
pub struct ShadowLexical {
    serving: Arc<dyn LexicalSearch>,
    shadow: Arc<dyn LexicalSearch>,
    state: Arc<ShadowState>,
}

impl ShadowLexical {
    /// Build a default-off or explicitly enabled wrapper.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] for an invalid sampling, task,
    /// score, or artifact policy.
    pub fn new(
        serving: Arc<dyn LexicalSearch>,
        shadow: Arc<dyn LexicalSearch>,
        config: ShadowLexicalConfig,
    ) -> SearchResult<Self> {
        Self::with_load_probe(serving, shadow, config, Arc::new(AlwaysAdmitShadowLoad))
    }

    /// Build a wrapper with an injected host-pressure probe.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] for an invalid policy.
    pub fn with_load_probe(
        serving: Arc<dyn LexicalSearch>,
        shadow: Arc<dyn LexicalSearch>,
        config: ShadowLexicalConfig,
        load_probe: Arc<dyn ShadowLoadProbe>,
    ) -> SearchResult<Self> {
        Ok(Self {
            serving,
            shadow,
            state: build_shadow_state(config, load_probe)?,
        })
    }

    /// Seed exact-replay corpus state for indexes opened after a prior commit.
    pub fn seed_corpus(&self, documents: &[IndexableDocument], manifest_generation: u64) {
        seed_shadow_corpus(&self.state, documents, manifest_generation);
    }

    /// Enable or disable new shadow work. In-flight tasks remain region-owned
    /// and drain or cancel with their asupersync region.
    pub fn set_enabled(&self, enabled: bool) {
        self.state.enabled.store(enabled, Ordering::Release);
        info!(
            event = "shadow_lifecycle",
            enabled,
            manifest_generation = self.state.manifest_generation.load(Ordering::Acquire),
            "shadow lexical lifecycle state changed"
        );
    }

    /// Mark a rebuilt shadow index healthy or unhealthy.
    pub fn set_shadow_ready(&self, ready: bool) {
        self.state.shadow_ready.store(ready, Ordering::Release);
    }

    /// Snapshot deterministic in-process counters.
    #[must_use]
    pub fn status(&self) -> ShadowStatus {
        shadow_status(&self.state)
    }

    fn submit_shadow(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        serving_results: &[ScoredResult],
        path: ShadowSearchPath,
        serve_latency_micros: u64,
    ) {
        Self::submit_shadow_parts(
            &self.shadow,
            &self.state,
            cx,
            query,
            limit,
            serving_results,
            path,
            serve_latency_micros,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn submit_shadow_parts(
        shadow: &Arc<dyn LexicalSearch>,
        state: &Arc<ShadowState>,
        cx: &Cx,
        query: &str,
        limit: usize,
        serving_results: &[ScoredResult],
        path: ShadowSearchPath,
        serve_latency_micros: u64,
    ) {
        if !state.enabled.load(Ordering::Acquire) {
            return;
        }

        let generation = state.manifest_generation.load(Ordering::Acquire);
        let ordinal = state.sample_ordinal.fetch_add(1, Ordering::Relaxed);
        if ordinal % SHADOW_SAMPLE_DENOMINATOR >= u64::from(state.sample_rate_basis_points) {
            state.sampled_out.fetch_add(1, Ordering::Relaxed);
            info!(
                event = "shadow_query_shed",
                query,
                manifest_generation = generation,
                shed_reason = "sample_rate",
                sample_ordinal = ordinal,
                "shadow lexical query was sampled out without affecting serving"
            );
            return;
        }
        state.sampled.fetch_add(1, Ordering::Relaxed);

        if !state.shadow_ready.load(Ordering::Acquire) {
            state.record_shed(ShadowShedReason::ShadowDegraded, query, generation);
            return;
        }
        if let Some(reason) = state.load_probe.shed_reason() {
            state.record_shed(reason, query, generation);
            return;
        }
        let Some(in_flight) = state.try_admit() else {
            state.record_shed(ShadowShedReason::Capacity, query, generation);
            return;
        };

        let query = query.to_owned();
        let query_for_log = query.clone();
        let serving_top_k = ranked_hits(serving_results);
        let shadow = Arc::clone(shadow);
        let task_state = Arc::clone(state);
        let corpus_revision = task_state.corpus_revision.load(Ordering::Acquire);
        let span = info_span!(
            "shadow_lexical_query",
            query = %query,
            limit,
            manifest_generation = generation,
            serve_latency_micros,
            shadow_latency_micros = tracing::field::Empty,
            divergence_class = tracing::field::Empty,
            shed_reason = tracing::field::Empty,
        );
        let spawn_result = cx.spawn(move |task_cx| {
            async move {
                let _in_flight = in_flight;
                let shadow_started = Instant::now();
                let outcome = match path {
                    ShadowSearchPath::Search => shadow.search(&task_cx, &query, limit).await,
                    ShadowSearchPath::FusionCandidates => {
                        shadow
                            .search_fusion_candidates(&task_cx, &query, limit)
                            .await
                    }
                };
                let shadow_latency_micros = micros(shadow_started.elapsed());
                tracing::Span::current().record("shadow_latency_micros", shadow_latency_micros);

                if generation != task_state.manifest_generation.load(Ordering::Acquire)
                    || corpus_revision != task_state.corpus_revision.load(Ordering::Acquire)
                {
                    tracing::Span::current()
                        .record("shed_reason", ShadowShedReason::GenerationDrift.label());
                    task_state.record_shed(ShadowShedReason::GenerationDrift, &query, generation);
                    task_state.record_degradation(
                        generation,
                        ShadowDegradationKind::GenerationDrift,
                        "manifest generation or corpus revision changed during shadow query",
                        true,
                    );
                    return;
                }

                let shadow_results = match outcome {
                    Ok(results) => results,
                    Err(error) => {
                        task_state.record_degradation(
                            generation,
                            ShadowDegradationKind::Search,
                            error.to_string(),
                            true,
                        );
                        return;
                    }
                };
                let shadow_top_k = ranked_hits(&shadow_results);
                let classification =
                    classify_ranked_hits(&serving_top_k, &shadow_top_k, task_state.score_epsilon);
                tracing::Span::current().record("divergence_class", classification.label());
                task_state.completed.fetch_add(1, Ordering::Relaxed);
                let observation = ShadowObservationRecord {
                    schema_version: SHADOW_OBSERVATION_SCHEMA_VERSION,
                    manifest_generation: generation,
                    query: query.clone(),
                    classification,
                    serve_latency_micros,
                    shadow_latency_micros,
                };
                if let Err(error) = task_state.sink.append_observation(&observation) {
                    task_state.record_degradation(
                        generation,
                        ShadowDegradationKind::ArtifactWrite,
                        error.to_string(),
                        false,
                    );
                }
                if !classification.is_artifact() {
                    return;
                }

                task_state.divergences.fetch_add(1, Ordering::Relaxed);
                let (corpus_hash, documents) = task_state.corpus_snapshot();
                let record = ShadowDivergenceRecord {
                    schema_version: SHADOW_DIVERGENCE_SCHEMA_VERSION,
                    manifest_generation: generation,
                    corpus_hash,
                    documents,
                    query: ShadowQuery {
                        text: query,
                        limit,
                        fusion_candidates: path.fusion_candidates(),
                    },
                    serving_top_k,
                    shadow_top_k,
                    classification,
                    serve_latency_micros,
                    shadow_latency_micros,
                };
                if let Err(error) = task_state.sink.append_divergence(&record) {
                    task_state.record_degradation(
                        generation,
                        ShadowDegradationKind::ArtifactWrite,
                        error.to_string(),
                        false,
                    );
                }
            }
            .instrument(span)
        });
        if let Err(error) = spawn_result {
            state.record_shed(
                ShadowShedReason::RuntimeUnavailable,
                &query_for_log,
                generation,
            );
            state.record_degradation(
                generation,
                ShadowDegradationKind::RuntimeUnavailable,
                error.to_string(),
                false,
            );
        }
    }

    fn record_corpus_documents(&self, documents: &[IndexableDocument]) {
        let mut corpus = self
            .state
            .corpus
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        corpus.extend(
            documents
                .iter()
                .map(|document| (document.id.clone(), ShadowDocument::from(document))),
        );
        drop(corpus);
        self.state.corpus_revision.fetch_add(1, Ordering::AcqRel);
    }

    async fn mirror_documents(&self, cx: &Cx, documents: &[IndexableDocument]) {
        if !self.state.enabled.load(Ordering::Acquire)
            || !self.state.shadow_ready.load(Ordering::Acquire)
        {
            return;
        }
        if let Err(error) = self.shadow.index_documents(cx, documents).await {
            self.state.shadow_ready.store(false, Ordering::Release);
            self.state.record_degradation(
                self.state.manifest_generation.load(Ordering::Acquire),
                ShadowDegradationKind::Index,
                error.to_string(),
                true,
            );
        }
    }
}

impl LexicalSearch for ShadowLexical {
    fn search<'a>(
        &'a self,
        cx: &'a Cx,
        query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, Vec<ScoredResult>> {
        Box::pin(async move {
            let serve_started = Instant::now();
            let results = self.serving.search(cx, query, limit).await?;
            self.submit_shadow(
                cx,
                query,
                limit,
                &results,
                ShadowSearchPath::Search,
                micros(serve_started.elapsed()),
            );
            Ok(results)
        })
    }

    fn search_fusion_candidates<'a>(
        &'a self,
        cx: &'a Cx,
        query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, Vec<ScoredResult>> {
        Box::pin(async move {
            let serve_started = Instant::now();
            let results = self
                .serving
                .search_fusion_candidates(cx, query, limit)
                .await?;
            self.submit_shadow(
                cx,
                query,
                limit,
                &results,
                ShadowSearchPath::FusionCandidates,
                micros(serve_started.elapsed()),
            );
            Ok(results)
        })
    }

    fn fusion_metadata_is_deferred(&self) -> bool {
        self.serving.fusion_metadata_is_deferred()
    }

    fn hydrate_fusion_metadata<'a>(
        &'a self,
        cx: &'a Cx,
        results: &'a mut [ScoredResult],
    ) -> SearchFuture<'a, ()> {
        self.serving.hydrate_fusion_metadata(cx, results)
    }

    fn index_document<'a>(
        &'a self,
        cx: &'a Cx,
        document: &'a IndexableDocument,
    ) -> SearchFuture<'a, ()> {
        Box::pin(async move {
            self.serving.index_document(cx, document).await?;
            self.record_corpus_documents(std::slice::from_ref(document));
            self.mirror_documents(cx, std::slice::from_ref(document))
                .await;
            Ok(())
        })
    }

    fn index_documents<'a>(
        &'a self,
        cx: &'a Cx,
        documents: &'a [IndexableDocument],
    ) -> SearchFuture<'a, ()> {
        Box::pin(async move {
            self.serving.index_documents(cx, documents).await?;
            self.record_corpus_documents(documents);
            self.mirror_documents(cx, documents).await;
            Ok(())
        })
    }

    fn commit<'a>(&'a self, cx: &'a Cx) -> SearchFuture<'a, ()> {
        Box::pin(async move {
            self.serving.commit(cx).await?;
            let generation = self
                .state
                .manifest_generation
                .fetch_add(1, Ordering::AcqRel)
                .saturating_add(1);
            if self.state.enabled.load(Ordering::Acquire)
                && self.state.shadow_ready.load(Ordering::Acquire)
                && let Err(error) = self.shadow.commit(cx).await
            {
                self.state.shadow_ready.store(false, Ordering::Release);
                self.state.record_degradation(
                    generation,
                    ShadowDegradationKind::Commit,
                    error.to_string(),
                    true,
                );
            }
            Ok(())
        })
    }

    fn doc_count(&self) -> usize {
        self.serving.doc_count()
    }
}

fn ranked_hits(results: &[ScoredResult]) -> Vec<ShadowRankedHit> {
    results
        .iter()
        .enumerate()
        .map(|(index, result)| ShadowRankedHit::from_result(index.saturating_add(1), result))
        .collect()
}

fn classify_ranked_hits(
    serving: &[ShadowRankedHit],
    shadow: &[ShadowRankedHit],
    score_epsilon: f32,
) -> ShadowDivergenceClass {
    if serving.len() != shadow.len() {
        return ShadowDivergenceClass::RankMismatch;
    }
    let same_order = serving
        .iter()
        .zip(shadow)
        .all(|(left, right)| left.document_id == right.document_id);
    if same_order {
        let exact_scores = serving.iter().zip(shadow).all(|(left, right)| {
            left.score_bits == right.score_bits
                && left.lexical_score_bits == right.lexical_score_bits
        });
        if exact_scores {
            return ShadowDivergenceClass::Exact;
        }
        let within_epsilon = serving.iter().zip(shadow).all(|(left, right)| {
            score_delta_within(left.score(), right.score(), score_epsilon)
                && optional_score_delta_within(
                    left.lexical_score_bits,
                    right.lexical_score_bits,
                    score_epsilon,
                )
        });
        return if within_epsilon {
            ShadowDivergenceClass::ScoreEpsilon
        } else {
            ShadowDivergenceClass::ScoreMismatch
        };
    }

    let serving_ids = serving
        .iter()
        .map(|hit| hit.document_id.as_str())
        .collect::<BTreeSet<_>>();
    let shadow_ids = shadow
        .iter()
        .map(|hit| hit.document_id.as_str())
        .collect::<BTreeSet<_>>();
    if serving_ids == shadow_ids {
        ShadowDivergenceClass::TieOrder
    } else {
        ShadowDivergenceClass::RankMismatch
    }
}

fn score_delta_within(left: f32, right: f32, epsilon: f32) -> bool {
    left.is_finite() && right.is_finite() && (left - right).abs() <= epsilon
}

fn optional_score_delta_within(left: Option<u32>, right: Option<u32>, epsilon: f32) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => {
            score_delta_within(f32::from_bits(left), f32::from_bits(right), epsilon)
        }
        _ => false,
    }
}

/// Compute the canonical SHA-256 replay identity for an id-ordered corpus.
#[must_use]
pub fn compute_shadow_corpus_hash(documents: &[ShadowDocument]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(CORPUS_HASH_DOMAIN);
    for document in documents {
        hash_bytes(&mut hasher, document.id.as_bytes());
        hash_bytes(&mut hasher, document.content.as_bytes());
        match document.title.as_deref() {
            Some(title) => {
                hasher.update([1]);
                hash_bytes(&mut hasher, title.as_bytes());
            }
            None => hasher.update([0]),
        }
        hasher.update(
            u64::try_from(document.metadata.len())
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
        for (key, value) in &document.metadata {
            hash_bytes(&mut hasher, key.as_bytes());
            hash_bytes(&mut hasher, value.as_bytes());
        }
    }
    let digest = hasher.finalize();
    let mut encoded = String::with_capacity(digest.len().saturating_mul(2));
    for byte in digest {
        write!(&mut encoded, "{byte:02x}").expect("writing into String cannot fail");
    }
    encoded
}

fn hash_bytes(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update(u64::try_from(bytes.len()).unwrap_or(u64::MAX).to_le_bytes());
    hasher.update(bytes);
}

fn micros(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use asupersync::Budget;
    use asupersync::lab::{LabConfig, LabRuntime};
    use asupersync::runtime::RuntimeBuilder;
    use compact_str::CompactString;

    use super::*;
    use crate::types::ScoreSource;

    struct StaticLexical {
        results: Vec<ScoredResult>,
        fail_index: AtomicBool,
        search_count: AtomicUsize,
    }

    impl StaticLexical {
        fn new(results: Vec<ScoredResult>) -> Self {
            Self {
                results,
                fail_index: AtomicBool::new(false),
                search_count: AtomicUsize::new(0),
            }
        }
    }

    impl LexicalSearch for StaticLexical {
        fn search<'a>(
            &'a self,
            _cx: &'a Cx,
            _query: &'a str,
            limit: usize,
        ) -> SearchFuture<'a, Vec<ScoredResult>> {
            Box::pin(async move {
                self.search_count.fetch_add(1, Ordering::Relaxed);
                Ok(self.results.iter().take(limit).cloned().collect())
            })
        }

        fn index_document<'a>(
            &'a self,
            _cx: &'a Cx,
            _document: &'a IndexableDocument,
        ) -> SearchFuture<'a, ()> {
            Box::pin(async move {
                if self.fail_index.load(Ordering::Acquire) {
                    Err(SearchError::InvalidConfig {
                        field: "shadow.test".to_owned(),
                        value: "fail".to_owned(),
                        reason: "injected shadow index failure".to_owned(),
                    })
                } else {
                    Ok(())
                }
            })
        }

        fn commit<'a>(&'a self, _cx: &'a Cx) -> SearchFuture<'a, ()> {
            Box::pin(async { Ok(()) })
        }

        fn doc_count(&self) -> usize {
            self.results.len()
        }
    }

    struct SlowShadow {
        delay: std::time::Duration,
    }

    impl LexicalSearch for SlowShadow {
        fn search<'a>(
            &'a self,
            _cx: &'a Cx,
            _query: &'a str,
            _limit: usize,
        ) -> SearchFuture<'a, Vec<ScoredResult>> {
            Box::pin(async move {
                std::thread::sleep(self.delay);
                Ok(Vec::new())
            })
        }

        fn index_document<'a>(
            &'a self,
            _cx: &'a Cx,
            _document: &'a IndexableDocument,
        ) -> SearchFuture<'a, ()> {
            Box::pin(async { Ok(()) })
        }

        fn commit<'a>(&'a self, _cx: &'a Cx) -> SearchFuture<'a, ()> {
            Box::pin(async { Ok(()) })
        }

        fn doc_count(&self) -> usize {
            0
        }
    }

    fn result(id: &str, score: f32) -> ScoredResult {
        ScoredResult {
            doc_id: CompactString::from(id),
            score,
            source: ScoreSource::Lexical,
            index: None,
            fast_score: None,
            quality_score: None,
            lexical_score: Some(score),
            rerank_score: None,
            explanation: None,
            metadata: None,
        }
    }

    fn enabled_config(root: &Path) -> ShadowLexicalConfig {
        ShadowLexicalConfig {
            enabled: true,
            artifact_directory: root.join(SHADOW_ARTIFACT_DIRECTORY),
            ..ShadowLexicalConfig::default()
        }
    }

    #[test]
    fn classifier_covers_exact_epsilon_tie_rank_and_score_mismatch() {
        let exact = ranked_hits(&[result("a", 1.0), result("b", 0.5)]);
        assert_eq!(
            classify_ranked_hits(&exact, &exact, 0.01),
            ShadowDivergenceClass::Exact
        );
        let epsilon = ranked_hits(&[result("a", 1.005), result("b", 0.5)]);
        assert_eq!(
            classify_ranked_hits(&exact, &epsilon, 0.01),
            ShadowDivergenceClass::ScoreEpsilon
        );
        let tie = ranked_hits(&[result("b", 0.5), result("a", 1.0)]);
        assert_eq!(
            classify_ranked_hits(&exact, &tie, 0.01),
            ShadowDivergenceClass::TieOrder
        );
        let rank = ranked_hits(&[result("a", 1.0), result("c", 0.5)]);
        assert_eq!(
            classify_ranked_hits(&exact, &rank, 0.01),
            ShadowDivergenceClass::RankMismatch
        );
        let score = ranked_hits(&[result("a", 0.8), result("b", 0.5)]);
        assert_eq!(
            classify_ranked_hits(&exact, &score, 0.01),
            ShadowDivergenceClass::ScoreMismatch
        );
    }

    #[test]
    fn schema_roundtrip_and_doctor_summary_reject_malformed_lines() {
        let temp = tempfile::tempdir().expect("tempdir");
        let directory = temp.path().join(SHADOW_ARTIFACT_DIRECTORY);
        let sink = ShadowArtifactSink::new(directory.clone());
        let documents = vec![ShadowDocument {
            id: "doc".to_owned(),
            content: "alpha".to_owned(),
            title: None,
            metadata: BTreeMap::new(),
        }];
        let record = ShadowDivergenceRecord {
            schema_version: SHADOW_DIVERGENCE_SCHEMA_VERSION,
            manifest_generation: 7,
            corpus_hash: compute_shadow_corpus_hash(&documents),
            documents,
            query: ShadowQuery {
                text: "alpha".to_owned(),
                limit: 10,
                fusion_candidates: true,
            },
            serving_top_k: ranked_hits(&[result("doc", 1.0)]),
            shadow_top_k: ranked_hits(&[]),
            classification: ShadowDivergenceClass::RankMismatch,
            serve_latency_micros: 4,
            shadow_latency_micros: 9,
        };
        sink.append_divergence(&record).expect("append");
        let line =
            fs::read_to_string(directory.join(SHADOW_DIVERGENCES_FILE)).expect("read artifact");
        let decoded: ShadowDivergenceRecord = serde_json::from_str(line.trim()).expect("roundtrip");
        assert_eq!(decoded, record);
        decoded.validate().expect("schema validates");

        let mut malformed = OpenOptions::new()
            .append(true)
            .open(directory.join(SHADOW_DIVERGENCES_FILE))
            .expect("open append");
        writeln!(malformed, "{{not-json").expect("append malformed");
        let summary =
            ShadowArtifactSummary::read_artifact_directory(directory).expect("doctor summary");
        assert_eq!(summary.divergence_count, 1);
        assert_eq!(summary.malformed_line_count, 1);
        assert_eq!(summary.latest_generation, Some(7));
    }

    #[test]
    fn deterministic_sampling_and_pressure_shedding_never_query_shadow() {
        let temp = tempfile::tempdir().expect("tempdir");
        let serving = Arc::new(StaticLexical::new(vec![result("a", 1.0)]));
        let shadow = Arc::new(StaticLexical::new(vec![result("a", 1.0)]));
        let pressure = Arc::new(AtomicShadowLoadProbe::default());
        let wrapper = ShadowLexical::with_load_probe(
            serving,
            shadow.clone(),
            ShadowLexicalConfig {
                sample_rate_basis_points: 2_500,
                ..enabled_config(temp.path())
            },
            pressure.clone(),
        )
        .expect("wrapper");
        let cx = Cx::for_testing();
        let serving_results = vec![result("a", 1.0)];
        for _ in 0..10_000 {
            wrapper.submit_shadow(
                &cx,
                "alpha",
                1,
                &serving_results,
                ShadowSearchPath::Search,
                1,
            );
        }
        let status = wrapper.status();
        assert_eq!(status.sampled, 2_500);
        assert_eq!(status.sampled_out, 7_500);
        assert_eq!(status.shed, 2_500);
        assert_eq!(shadow.search_count.load(Ordering::Acquire), 0);

        pressure.set_cpu_pressure(true);
        wrapper.submit_shadow(&cx, "cpu", 1, &serving_results, ShadowSearchPath::Search, 1);
        assert_eq!(shadow.search_count.load(Ordering::Acquire), 0);
        pressure.set_cpu_pressure(false);
        pressure.set_io_pressure(true);
        wrapper.submit_shadow(&cx, "io", 1, &serving_results, ShadowSearchPath::Search, 1);
        assert_eq!(shadow.search_count.load(Ordering::Acquire), 0);
        assert_eq!(wrapper.status().shed, 2_502);
    }

    #[test]
    fn shadow_index_failure_is_typed_persisted_and_never_fails_serving() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let temp = tempfile::tempdir().expect("tempdir");
            let serving = Arc::new(StaticLexical::new(Vec::new()));
            let shadow = Arc::new(StaticLexical::new(Vec::new()));
            shadow.fail_index.store(true, Ordering::Release);
            let wrapper =
                ShadowLexical::new(serving, shadow, enabled_config(temp.path())).expect("wrapper");
            wrapper
                .index_document(&cx, &IndexableDocument::new("doc", "alpha"))
                .await
                .expect("serving indexing must remain healthy");
            assert_eq!(wrapper.status().degradations, 1);
            let summary =
                ShadowArtifactSummary::read_index_root(temp.path()).expect("doctor summary");
            assert_eq!(summary.degradation_count, 1);
        });
    }

    #[test]
    fn labruntime_shadow_result_is_non_interfering_and_region_reaches_quiescence() {
        let temp = tempfile::tempdir().expect("tempdir");
        let serving_result = vec![result("serving", 3.0), result("second", 2.0)];
        let serving = Arc::new(StaticLexical::new(serving_result.clone()));
        let shadow = Arc::new(StaticLexical::new(vec![result("shadow", 9.0)]));
        let wrapper = Arc::new(
            ShadowLexical::new(serving, shadow, enabled_config(temp.path())).expect("wrapper"),
        );
        wrapper.seed_corpus(&[IndexableDocument::new("serving", "alpha")], 0);

        let mut lab = LabRuntime::new(LabConfig::new(0x5ad0_0001).max_steps(100_000));
        let region = lab.state.create_root_region(Budget::INFINITE);
        let task_wrapper = wrapper.clone();
        let (task, handle) = lab
            .state
            .create_task(region, Budget::INFINITE, async move {
                let cx = Cx::current().expect("LabRuntime task context");
                let actual = task_wrapper
                    .search(&cx, "alpha", 10)
                    .await
                    .expect("serving search");
                assert_eq!(
                    ranked_hits(&actual),
                    ranked_hits(&serving_result),
                    "serving top-k must remain bit-identical"
                );
            })
            .expect("create task");
        lab.scheduler.lock().schedule(task, 0);
        let report = lab.run_until_quiescent_with_report();
        assert!(report.quiescent, "seed=0x5ad00001");
        assert!(report.oracle_report.all_passed(), "seed=0x5ad00001");
        assert!(report.invariant_violations.is_empty(), "seed=0x5ad00001");
        drop(handle);
        assert_eq!(wrapper.status().in_flight, 0);
        let summary = ShadowArtifactSummary::read_index_root(temp.path()).expect("summary");
        assert_eq!(summary.divergence_count, 1);
    }

    #[test]
    fn wall_clock_guard_never_awaits_slow_shadow_backend() {
        let temp = tempfile::tempdir().expect("tempdir");
        let wrapper = Arc::new(
            ShadowLexical::new(
                Arc::new(StaticLexical::new(vec![result("serving", 1.0)])),
                Arc::new(SlowShadow {
                    delay: std::time::Duration::from_millis(250),
                }),
                enabled_config(temp.path()),
            )
            .expect("wrapper"),
        );
        let runtime = RuntimeBuilder::current_thread()
            .build()
            .expect("build runtime");
        let task_wrapper = wrapper.clone();
        let started = Instant::now();
        let serving_task = runtime.handle().spawn(async move {
            let cx = Cx::current().expect("runtime installs a spawn-capable context");
            task_wrapper
                .search(&cx, "latency guard", 10)
                .await
                .expect("serving result")
        });
        let actual = runtime.block_on(serving_task);
        let elapsed = started.elapsed();
        assert_eq!(ranked_hits(&actual), ranked_hits(&[result("serving", 1.0)]));
        assert_eq!(
            wrapper.status().sampled,
            1,
            "wall-clock guard must exercise a real admitted shadow task"
        );
        assert!(
            elapsed < std::time::Duration::from_millis(100),
            "serving took {elapsed:?} and appears to have awaited the 250ms shadow"
        );
    }

    #[test]
    fn enable_disable_lifecycle_admits_only_enabled_queries() {
        let temp = tempfile::tempdir().expect("tempdir");
        let serving = Arc::new(StaticLexical::new(vec![result("a", 1.0)]));
        let shadow = Arc::new(StaticLexical::new(vec![result("a", 1.0)]));
        let wrapper =
            ShadowLexical::new(serving, shadow, enabled_config(temp.path())).expect("wrapper");
        wrapper.set_enabled(false);
        assert!(!wrapper.status().enabled);
        wrapper.set_enabled(true);
        assert!(wrapper.status().enabled);
        wrapper.set_shadow_ready(false);
        let cx = Cx::for_testing();
        wrapper.submit_shadow(
            &cx,
            "disabled-shadow",
            1,
            &[result("a", 1.0)],
            ShadowSearchPath::Search,
            1,
        );
        assert_eq!(wrapper.status().shed, 1);
        assert_eq!(wrapper.status().in_flight, 0);
    }

    #[test]
    fn corpus_hash_is_order_and_hashmap_iteration_independent() {
        let mut left_metadata = HashMap::new();
        left_metadata.insert("z".to_owned(), "last".to_owned());
        left_metadata.insert("a".to_owned(), "first".to_owned());
        let left = IndexableDocument {
            id: "b".to_owned(),
            content: "beta".to_owned(),
            title: None,
            metadata: left_metadata,
        };
        let right = IndexableDocument::new("a", "alpha");
        let ordered = BTreeMap::from([
            (left.id.clone(), ShadowDocument::from(&left)),
            (right.id.clone(), ShadowDocument::from(&right)),
        ]);
        let forward = ordered.values().cloned().collect::<Vec<_>>();
        let reverse_input = [right, left];
        let reverse_map = reverse_input
            .iter()
            .map(|document| (document.id.clone(), ShadowDocument::from(document)))
            .collect::<BTreeMap<_, _>>();
        let reverse = reverse_map.values().cloned().collect::<Vec<_>>();
        assert_eq!(
            compute_shadow_corpus_hash(&forward),
            compute_shadow_corpus_hash(&reverse)
        );
    }
}
