//! QG-1 H1: immutable corpus and continuous first-feed-to-terminal timing.
//!
//! The legacy bulk harness generated every batch *outside* its timer and
//! summed the per-call durations. Engines with background workers (Tantivy's
//! compressor/updater/merge threads) kept making progress during those
//! untimed generation gaps while synchronous engines received no such credit,
//! so every summed-call bulk ratio was structurally invalid. This module is
//! the typed contract that replaces it:
//!
//! 1. The corpus is materialized and sealed into a [`ContinuousCorpusManifest`]
//!    *before* the timed window; its identity, order, bytes, fields, positions,
//!    provenance, and xxh3 digest are bound up front and re-verified per run.
//! 2. The clock is one continuous monotonic interval from the first feed call
//!    through the terminal searchable commit *and* engine quiescence
//!    ([`EngineQuiescence`]); no inter-call gap is excluded because none
//!    exists.
//! 3. Accepted, processed, committed, and searchable document counts plus fed
//!    bytes must be equal ([`ContinuousWindowReceipt::validate`]); any
//!    mismatch, early stop, pending merge, or summed-call timing source is a
//!    typed rejection.
//! 4. Throughput is always a rate derived from equal work units over the
//!    continuous window ([`continuous_raw_sample`] emits
//!    [`PerfMetricSemantics::Throughput`] samples), never a stored gauge.
//!
//! QG-1 H2 (actual-work and lifecycle receipts) composes through the
//! [`LifecycleObserver`] seam without touching the timing loop.

use std::collections::BTreeMap;

use frankensearch_core::IndexableDocument;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use xxhash_rust::xxh3::Xxh3;

use crate::perf::{
    PerfMetricSemantics, PerfOperationScope, PerfRawSample, PerfSampleArm, PerfSampleOrder,
    PerfSamplePhase, PerfSampleProvenance,
};

/// Schema tag stamped on every continuous-timing evidence object.
pub const CONTINUOUS_TIMING_SCHEMA_VERSION: &str = "qg1-continuous-timing-v1";

/// The only corpus identity family admissible for QG-1 continuous evidence.
///
/// QG-1 must measure its own sealed native workload; a corpus identity
/// derived from another gate (for example a QG-6 query-latency fixture) is a
/// typed rejection, not a soft warning.
pub const QG1_NATIVE_CORPUS_IDENTITY_PREFIX: &str = "qg1-native/";

/// Timing semantics selected for one benchmark invocation.
///
/// `PerCall` preserves the legacy summed-call artifact byte-for-byte so
/// existing consumers keep the exact workload they always measured;
/// `Continuous` opts a run into the H1 contract. The fleet flips the default
/// only after the ratchet owner accepts the new stream identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimingMode {
    /// Legacy summed per-call timing (default; artifact-compatible).
    PerCall,
    /// One continuous monotonic window per sample (QG-1 H1 contract).
    Continuous,
}

impl TimingMode {
    /// Environment variable that selects the timing mode.
    pub const ENV_VAR: &'static str = "QUILL_PERF_TIMING_MODE";

    /// Parse an optional environment value.
    ///
    /// `None` (unset) selects [`Self::PerCall`] for artifact compatibility.
    ///
    /// # Errors
    ///
    /// Returns [`ContinuousTimingError::InvalidTimingMode`] for any value
    /// other than `continuous` or `per-call`.
    pub fn parse(value: Option<&str>) -> Result<Self, ContinuousTimingError> {
        match value {
            None => Ok(Self::PerCall),
            Some("per-call") => Ok(Self::PerCall),
            Some("continuous") => Ok(Self::Continuous),
            Some(other) => Err(ContinuousTimingError::InvalidTimingMode {
                value: other.to_owned(),
            }),
        }
    }

    /// Read the mode from [`Self::ENV_VAR`].
    ///
    /// # Errors
    ///
    /// Returns [`ContinuousTimingError::InvalidTimingMode`] for an
    /// unrecognized value.
    pub fn from_env() -> Result<Self, ContinuousTimingError> {
        let value = std::env::var(Self::ENV_VAR).ok();
        Self::parse(value.as_deref())
    }

    /// Stable artifact label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::PerCall => "per-call",
            Self::Continuous => "continuous",
        }
    }

    /// Whether this invocation runs the continuous H1 contract.
    #[must_use]
    pub const fn is_continuous(self) -> bool {
        matches!(self, Self::Continuous)
    }
}

/// Typed failure surface for the continuous timing contract.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ContinuousTimingError {
    #[error("QUILL_PERF_TIMING_MODE must be continuous or per-call, got {value:?}")]
    InvalidTimingMode { value: String },
    #[error(
        "corpus identity {identity:?} is not QG-1-native (must start with \
         {QG1_NATIVE_CORPUS_IDENTITY_PREFIX:?}); deriving the bulk corpus from another \
         gate's fixture is forbidden"
    )]
    NonNativeCorpusIdentity { identity: String },
    #[error("continuous corpus must contain at least one document")]
    EmptyCorpus,
    #[error(
        "corpus digest mismatch: sealed {expected} but recomputed {actual}; the corpus mutated"
    )]
    CorpusDigestMismatch { expected: String, actual: String },
    #[error("corpus {field} mismatch: sealed {expected} but recomputed {actual}")]
    CorpusShapeMismatch {
        field: &'static str,
        expected: u64,
        actual: u64,
    },
    #[error(
        "receipt timing source is summed per-call; continuous evidence requires one \
         monotonic first-feed-to-quiescence window"
    )]
    SummedPerCallTiming,
    #[error("work inequality at {stage}: expected {expected} but observed {actual}")]
    WorkInequality {
        stage: &'static str,
        expected: u64,
        actual: u64,
    },
    #[error("continuous window recorded zero feed calls")]
    NoFeedCalls,
    #[error("continuous phase order violation at {phase}")]
    PhaseOrderViolation { phase: &'static str },
    #[error(
        "clock stopped at {window_total_ns}ns but quiescence joined at \
         {quiescence_joined_ns}ns; the window must terminate exactly at quiescence"
    )]
    ClockStoppedBeforeQuiescence {
        window_total_ns: u64,
        quiescence_joined_ns: u64,
    },
    #[error(
        "engine {engine:?} paired with quiescence {quiescence:?}; each engine must attest \
         its own terminal quiescence mechanism"
    )]
    QuiescenceContractMismatch {
        engine: String,
        quiescence: &'static str,
    },
    #[error("terminal searchable probe missing or empty")]
    MissingTerminalProbe,
    #[error("continuous throughput samples require positive work units")]
    ZeroWorkUnits,
    #[error(
        "byte stage {stage} claims structural unobservability without naming the seam; \
         an unnamed gap is indistinguishable from a skipped observation"
    )]
    UnnamedUnobservableSeam { stage: &'static str },
    #[error(
        "continuous samples must carry Throughput semantics with a derived rate; \
         gauge semantics {semantics:?} rejected"
    )]
    GaugeSemanticsRejected { semantics: String },
}

fn canonical_document_digest(hasher: &mut Xxh3, document: &IndexableDocument) {
    hasher.update(document.id.as_bytes());
    hasher.update(&[0x1e]);
    match &document.title {
        Some(title) => {
            hasher.update(&[1]);
            hasher.update(title.as_bytes());
        }
        None => hasher.update(&[0]),
    }
    hasher.update(&[0x1e]);
    hasher.update(document.content.as_bytes());
    hasher.update(&[0x1e]);
    let metadata: BTreeMap<&str, &str> = document
        .metadata
        .iter()
        .map(|(key, value)| (key.as_str(), value.as_str()))
        .collect();
    for (key, value) in metadata {
        hasher.update(key.as_bytes());
        hasher.update(&[0x1f]);
        hasher.update(value.as_bytes());
        hasher.update(&[0x1f]);
    }
    hasher.update(&[0x1e]);
}

/// Sealed identity of one pre-materialized bulk corpus.
///
/// Sealed *before* the timed window: order, content, bytes, fields, positions,
/// provenance, and digest are bound up front, and [`Self::verify`] re-proves
/// them against the materialized documents before every window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContinuousCorpusManifest {
    pub schema_version: String,
    /// QG-1-native corpus identity (see [`QG1_NATIVE_CORPUS_IDENTITY_PREFIX`]).
    pub identity: String,
    /// Deterministic generator seed (provenance).
    pub generator_seed: u64,
    /// Position mode of the cell this corpus feeds (`positions_on`/`positions_off`).
    pub positions: String,
    /// Exact document count, bound to order via the digest.
    pub doc_count: u64,
    pub id_bytes: u64,
    pub title_bytes: u64,
    pub content_bytes: u64,
    pub metadata_bytes: u64,
    /// Sum of the four byte components; the equal-work byte denominator.
    pub total_bytes: u64,
    /// Documents carrying a title field.
    pub titled_docs: u64,
    /// Total metadata key/value entries across the corpus.
    pub metadata_entries: u64,
    /// Order-sensitive xxh3-64 digest over id/title/content/canonical metadata.
    pub corpus_xxh3: String,
}

impl ContinuousCorpusManifest {
    /// Seal a materialized corpus into an immutable manifest.
    ///
    /// # Errors
    ///
    /// Returns [`ContinuousTimingError::NonNativeCorpusIdentity`] for a
    /// non-QG-1-native identity and [`ContinuousTimingError::EmptyCorpus`]
    /// for an empty corpus.
    pub fn seal(
        identity: &str,
        generator_seed: u64,
        positions: &str,
        documents: &[IndexableDocument],
    ) -> Result<Self, ContinuousTimingError> {
        if !identity.starts_with(QG1_NATIVE_CORPUS_IDENTITY_PREFIX) {
            return Err(ContinuousTimingError::NonNativeCorpusIdentity {
                identity: identity.to_owned(),
            });
        }
        if documents.is_empty() {
            return Err(ContinuousTimingError::EmptyCorpus);
        }
        let mut hasher = Xxh3::new();
        let mut id_bytes = 0_u64;
        let mut title_bytes = 0_u64;
        let mut content_bytes = 0_u64;
        let mut metadata_bytes = 0_u64;
        let mut titled_docs = 0_u64;
        let mut metadata_entries = 0_u64;
        for document in documents {
            canonical_document_digest(&mut hasher, document);
            id_bytes += document.id.len() as u64;
            if let Some(title) = &document.title {
                title_bytes += title.len() as u64;
                titled_docs += 1;
            }
            content_bytes += document.content.len() as u64;
            for (key, value) in &document.metadata {
                metadata_bytes += key.len() as u64 + value.len() as u64;
                metadata_entries += 1;
            }
        }
        Ok(Self {
            schema_version: CONTINUOUS_TIMING_SCHEMA_VERSION.to_owned(),
            identity: identity.to_owned(),
            generator_seed,
            positions: positions.to_owned(),
            doc_count: documents.len() as u64,
            id_bytes,
            title_bytes,
            content_bytes,
            metadata_bytes,
            total_bytes: id_bytes + title_bytes + content_bytes + metadata_bytes,
            titled_docs,
            metadata_entries,
            corpus_xxh3: format!("{:016x}", hasher.digest()),
        })
    }

    /// Structural self-check of a sealed manifest.
    ///
    /// # Errors
    ///
    /// Returns typed identity, emptiness, or byte-total errors.
    pub fn validate(&self) -> Result<(), ContinuousTimingError> {
        if !self.identity.starts_with(QG1_NATIVE_CORPUS_IDENTITY_PREFIX) {
            return Err(ContinuousTimingError::NonNativeCorpusIdentity {
                identity: self.identity.clone(),
            });
        }
        if self.doc_count == 0 {
            return Err(ContinuousTimingError::EmptyCorpus);
        }
        let component_sum =
            self.id_bytes + self.title_bytes + self.content_bytes + self.metadata_bytes;
        if component_sum != self.total_bytes {
            return Err(ContinuousTimingError::CorpusShapeMismatch {
                field: "total_bytes",
                expected: self.total_bytes,
                actual: component_sum,
            });
        }
        Ok(())
    }

    /// Re-prove immutability: recompute every bound quantity from the
    /// materialized documents and compare against the sealed manifest.
    ///
    /// # Errors
    ///
    /// Returns [`ContinuousTimingError::CorpusDigestMismatch`] when content,
    /// order, or metadata drifted, and
    /// [`ContinuousTimingError::CorpusShapeMismatch`] when any bound count
    /// drifted.
    pub fn verify(&self, documents: &[IndexableDocument]) -> Result<(), ContinuousTimingError> {
        self.validate()?;
        let recomputed = Self::seal(
            &self.identity,
            self.generator_seed,
            &self.positions,
            documents,
        )?;
        for (field, expected, actual) in [
            ("doc_count", self.doc_count, recomputed.doc_count),
            ("id_bytes", self.id_bytes, recomputed.id_bytes),
            ("title_bytes", self.title_bytes, recomputed.title_bytes),
            (
                "content_bytes",
                self.content_bytes,
                recomputed.content_bytes,
            ),
            (
                "metadata_bytes",
                self.metadata_bytes,
                recomputed.metadata_bytes,
            ),
            ("total_bytes", self.total_bytes, recomputed.total_bytes),
            ("titled_docs", self.titled_docs, recomputed.titled_docs),
            (
                "metadata_entries",
                self.metadata_entries,
                recomputed.metadata_entries,
            ),
        ] {
            if expected != actual {
                return Err(ContinuousTimingError::CorpusShapeMismatch {
                    field,
                    expected,
                    actual,
                });
            }
        }
        if recomputed.corpus_xxh3 != self.corpus_xxh3 {
            return Err(ContinuousTimingError::CorpusDigestMismatch {
                expected: self.corpus_xxh3.clone(),
                actual: recomputed.corpus_xxh3,
            });
        }
        Ok(())
    }
}

/// How the elapsed time in a receipt was produced.
///
/// [`Self::SummedPerCall`] exists so the legacy shape is *representable and
/// rejected* rather than silently indistinguishable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimingSource {
    /// One monotonic interval covering feed, commit, verification, quiescence.
    ContinuousMonotonicWindow,
    /// Legacy sum of individually timed calls (always rejected for evidence).
    SummedPerCall,
}

/// Engine-specific terminal quiescence mechanism, attested per window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineQuiescence {
    /// Quill's `commit` seals shards, publishes the searchable snapshot, and
    /// returns synchronously; there are no background workers left running
    /// when it returns, so the committed snapshot *is* quiescence.
    QuillSealedSynchronousCommit,
    /// Tantivy's `IndexWriter::wait_merging_threads` was invoked inside the
    /// window: the operation channel closed, every indexing worker thread was
    /// joined, and the segment updater's merging thread completed all
    /// in-flight merges before the clock stopped.
    TantivyMergingThreadsJoined,
}

impl EngineQuiescence {
    /// Stable label for logs and error messages.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::QuillSealedSynchronousCommit => "quill_sealed_synchronous_commit",
            Self::TantivyMergingThreadsJoined => "tantivy_merging_threads_joined",
        }
    }
}

/// Window-relative phase boundaries, in nanoseconds since the first feed call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContinuousPhaseTimeline {
    /// Last feed call returned.
    pub feed_complete_ns: u64,
    /// Terminal commit returned.
    pub commit_complete_ns: u64,
    /// Searchable count and terminal probe verified.
    pub searchable_verified_ns: u64,
    /// Engine quiescence joined (always the final phase).
    pub quiescence_joined_ns: u64,
    /// Total window span; must equal `quiescence_joined_ns`.
    pub window_total_ns: u64,
}

/// Lifecycle boundary passed to the H2 receipts seam.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifecyclePhase {
    FirstFeed,
    FeedComplete,
    CommitComplete,
    SearchableVerified,
    QuiescenceJoined,
}

/// QG-1 H2 composition seam.
///
/// The continuous runner reports every phase boundary through this trait so
/// the actual-work/queue/worker-role receipts bead can attach collectors
/// without modifying the timing loop. H1 ships only [`NoopLifecycleObserver`].
pub trait LifecycleObserver {
    /// Called at each phase boundary with window-relative elapsed time.
    fn on_phase(&mut self, phase: LifecyclePhase, window_elapsed_ns: u64);
}

/// Default observer: records nothing, costs nothing.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopLifecycleObserver;

impl LifecycleObserver for NoopLifecycleObserver {
    fn on_phase(&mut self, _phase: LifecyclePhase, _window_elapsed_ns: u64) {}
}

/// Byte count for one lifecycle stage: observed from the actual documents at
/// the stage boundary, or typed as structurally unobservable at this seam.
///
/// The unobservable variant is an explicit, named gap — never a silently
/// copied denominator. H1 stays open behind every such gap until the H2
/// receipts bead makes the stage observable engine-side.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ByteStageObservation {
    /// Byte total observed from the actual documents at this stage.
    Observed(u64),
    /// The stage's byte total cannot be observed at this seam; `seam` names
    /// exactly which boundary lacks the observation.
    StructurallyUnobservable { seam: String },
}

/// Bytes exactly as the sealed manifest counts them for one document.
///
/// Covers id, optional title, content, and every metadata key/value. This is
/// the single shared definition used by both manifest sealing and
/// window-time observed byte accounting, so the two can never drift.
#[must_use]
pub fn document_bytes(document: &IndexableDocument) -> u64 {
    let mut bytes = document.id.len() as u64 + document.content.len() as u64;
    if let Some(title) = &document.title {
        bytes += title.len() as u64;
    }
    for (key, value) in &document.metadata {
        bytes += key.len() as u64 + value.len() as u64;
    }
    bytes
}

/// Receipt for one continuous first-feed-to-quiescence window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContinuousWindowReceipt {
    pub schema_version: String,
    /// Engine arm label (`quill` or `tantivy`).
    pub engine: String,
    pub timing_source: TimingSource,
    pub quiescence: EngineQuiescence,
    /// Number of feed calls issued inside the window.
    pub feed_calls: u64,
    /// Documents handed to the engine (tallied before each feed call).
    pub accepted_docs: u64,
    /// Documents the engine acknowledged (tallied after each successful call).
    pub processed_docs: u64,
    /// Engine-reported document count after the terminal commit.
    pub committed_docs: u64,
    /// Engine-reported document count at the terminal searchable probe.
    pub searchable_docs: u64,
    /// Bytes observed at hand-off: summed over the actual documents of each
    /// batch immediately before the feed call.
    pub accepted_bytes: u64,
    /// Bytes observed after each successful feed call returned.
    pub processed_bytes: u64,
    /// Byte total at the committed stage (observed or a typed gap).
    pub committed_bytes: ByteStageObservation,
    /// Byte total at the searchable stage (observed or a typed gap).
    pub searchable_bytes: ByteStageObservation,
    /// Visibility-parity commits observed inside the window, excluding the
    /// terminal commit: Quill's snapshot-generation delta across the feed
    /// phase only, or the Tantivy arm's counted cadence commits.
    pub periodic_commits: u64,
    /// Diagnostic side-by-side view: the sum of individually timed feed and
    /// commit calls *inside* the same window. The gap between this and
    /// `timeline.window_total_ns` is exactly the engine work the legacy
    /// summed-call harness never charged.
    pub percall_sum_ns: u64,
    pub timeline: ContinuousPhaseTimeline,
    /// Query text of the terminal searchable liveness probe.
    pub terminal_probe_query: String,
    /// Hits returned by the terminal probe (must be nonzero).
    pub terminal_probe_hits: u64,
}

impl ContinuousWindowReceipt {
    /// Validate the receipt against its sealed corpus manifest.
    ///
    /// Every acceptance clause of the H1 contract that can be checked
    /// structurally is checked here, fail-closed: summed-call timing, work
    /// inequality (accepted/processed/committed/searchable/bytes), early
    /// stop, pending merge (engine/quiescence mismatch), phase disorder, a
    /// clock that stopped before quiescence, and a missing terminal probe.
    ///
    /// # Errors
    ///
    /// Returns the first [`ContinuousTimingError`] violated.
    pub fn validate(
        &self,
        manifest: &ContinuousCorpusManifest,
    ) -> Result<(), ContinuousTimingError> {
        manifest.validate()?;
        if self.timing_source != TimingSource::ContinuousMonotonicWindow {
            return Err(ContinuousTimingError::SummedPerCallTiming);
        }
        if self.feed_calls == 0 {
            return Err(ContinuousTimingError::NoFeedCalls);
        }
        for (stage, actual) in [
            ("accepted_docs", self.accepted_docs),
            ("processed_docs", self.processed_docs),
            ("committed_docs", self.committed_docs),
            ("searchable_docs", self.searchable_docs),
        ] {
            if actual != manifest.doc_count {
                return Err(ContinuousTimingError::WorkInequality {
                    stage,
                    expected: manifest.doc_count,
                    actual,
                });
            }
        }
        for (stage, actual) in [
            ("accepted_bytes", self.accepted_bytes),
            ("processed_bytes", self.processed_bytes),
        ] {
            if actual != manifest.total_bytes {
                return Err(ContinuousTimingError::WorkInequality {
                    stage,
                    expected: manifest.total_bytes,
                    actual,
                });
            }
        }
        for (stage, observation) in [
            ("committed_bytes", &self.committed_bytes),
            ("searchable_bytes", &self.searchable_bytes),
        ] {
            match observation {
                ByteStageObservation::Observed(actual) => {
                    if *actual != manifest.total_bytes {
                        return Err(ContinuousTimingError::WorkInequality {
                            stage,
                            expected: manifest.total_bytes,
                            actual: *actual,
                        });
                    }
                }
                ByteStageObservation::StructurallyUnobservable { seam } => {
                    if seam.trim().is_empty() {
                        return Err(ContinuousTimingError::UnnamedUnobservableSeam { stage });
                    }
                }
            }
        }
        let timeline = &self.timeline;
        if timeline.feed_complete_ns == 0 {
            return Err(ContinuousTimingError::PhaseOrderViolation {
                phase: "feed_complete",
            });
        }
        if timeline.commit_complete_ns < timeline.feed_complete_ns {
            return Err(ContinuousTimingError::PhaseOrderViolation {
                phase: "commit_complete",
            });
        }
        if timeline.searchable_verified_ns < timeline.commit_complete_ns {
            return Err(ContinuousTimingError::PhaseOrderViolation {
                phase: "searchable_verified",
            });
        }
        if timeline.quiescence_joined_ns < timeline.searchable_verified_ns {
            return Err(ContinuousTimingError::PhaseOrderViolation {
                phase: "quiescence_joined",
            });
        }
        if timeline.window_total_ns != timeline.quiescence_joined_ns {
            return Err(ContinuousTimingError::ClockStoppedBeforeQuiescence {
                window_total_ns: timeline.window_total_ns,
                quiescence_joined_ns: timeline.quiescence_joined_ns,
            });
        }
        if self.percall_sum_ns > timeline.window_total_ns {
            return Err(ContinuousTimingError::PhaseOrderViolation {
                phase: "percall_sum_within_window",
            });
        }
        let quiescence_ok = match self.engine.as_str() {
            "quill" => self.quiescence == EngineQuiescence::QuillSealedSynchronousCommit,
            "tantivy" => self.quiescence == EngineQuiescence::TantivyMergingThreadsJoined,
            _ => false,
        };
        if !quiescence_ok {
            return Err(ContinuousTimingError::QuiescenceContractMismatch {
                engine: self.engine.clone(),
                quiescence: self.quiescence.label(),
            });
        }
        if self.terminal_probe_query.is_empty() || self.terminal_probe_hits == 0 {
            return Err(ContinuousTimingError::MissingTerminalProbe);
        }
        Ok(())
    }

    /// Derived throughput over the continuous window.
    ///
    /// Always computed from equal work units and the window span; never a
    /// stored gauge field.
    #[must_use]
    pub fn docs_per_second(&self) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let docs = self.accepted_docs as f64;
        #[allow(clippy::cast_precision_loss)]
        let elapsed_ns = (self.timeline.window_total_ns.max(1)) as f64;
        docs * 1_000_000_000.0 / elapsed_ns
    }
}

/// Continuous evidence for one matrix cell (bounded: the sealed manifest,
/// round counts, and the final measurement-round receipt per arm).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContinuousCellEvidence {
    pub schema_version: String,
    pub fixture: String,
    pub timing_mode: String,
    pub corpus: ContinuousCorpusManifest,
    pub rounds_quill: u64,
    pub rounds_tantivy: u64,
    pub last_quill_receipt: Option<ContinuousWindowReceipt>,
    pub last_tantivy_receipt: Option<ContinuousWindowReceipt>,
    /// True only when every window in the run validated against the manifest.
    pub all_windows_validated: bool,
}

/// Additive per-gate continuous-timing evidence block.
///
/// Absent (and never serialized) in per-call mode, so the legacy
/// `quill-perf-artifact-v3` byte shape is unchanged until a run opts in.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContinuousTimingEvidence {
    pub schema_version: String,
    pub timing_mode: String,
    pub cells: Vec<ContinuousCellEvidence>,
}

/// Identity fields for one continuous raw sample.
#[derive(Debug, Clone, Copy)]
pub struct ContinuousSampleIdentity {
    pub block_id: u64,
    pub sample_id: u64,
    pub arm: PerfSampleArm,
    pub order: PerfSampleOrder,
}

/// Origin-relative window and equal-work denominators for one sample.
#[derive(Debug, Clone, Copy)]
pub struct ContinuousSampleWindow {
    /// Monotonic ns (relative to the experiment origin) at first feed.
    pub started_ns: u64,
    /// Monotonic ns (relative to the experiment origin) at quiescence join.
    pub ended_ns: u64,
    /// Equal per-arm document denominator.
    pub work_units: u64,
    /// Equal per-arm byte denominator.
    pub byte_count: u64,
}

/// Build the versioned Throughput scope for a continuous operation.
///
/// The operation id is suffixed with `.continuous` and the scope version is
/// bumped so continuous and summed-call streams have independent identities
/// and can never be merged or substituted for one another.
#[must_use]
pub fn continuous_throughput_scope(base_operation_id: &str, unit: &str) -> PerfOperationScope {
    PerfOperationScope {
        operation_id: format!("{base_operation_id}.continuous"),
        version: 2,
        semantics: PerfMetricSemantics::Throughput,
        unit: unit.to_owned(),
    }
}

/// Build one continuous raw sample with derived-throughput semantics.
///
/// The sample's own monotonic interval *is* the continuous window, and the
/// estimator derives the rate from `work_units` over that interval. A gauge
/// `observed_value` is structurally impossible here.
///
/// # Errors
///
/// Rejects non-Throughput scopes ([`ContinuousTimingError::GaugeSemanticsRejected`]),
/// zero work or bytes ([`ContinuousTimingError::ZeroWorkUnits`]), and a
/// non-positive window ([`ContinuousTimingError::PhaseOrderViolation`]).
pub fn continuous_raw_sample(
    identity: ContinuousSampleIdentity,
    scope: &PerfOperationScope,
    provenance: &PerfSampleProvenance,
    window: &ContinuousSampleWindow,
) -> Result<PerfRawSample, ContinuousTimingError> {
    if scope.semantics != PerfMetricSemantics::Throughput {
        return Err(ContinuousTimingError::GaugeSemanticsRejected {
            semantics: format!("{:?}", scope.semantics),
        });
    }
    if window.work_units == 0 || window.byte_count == 0 {
        return Err(ContinuousTimingError::ZeroWorkUnits);
    }
    if window.ended_ns <= window.started_ns {
        return Err(ContinuousTimingError::PhaseOrderViolation {
            phase: "sample_window",
        });
    }
    Ok(PerfRawSample {
        block_id: identity.block_id,
        sample_id: identity.sample_id,
        arm: identity.arm,
        order: identity.order,
        phase: PerfSamplePhase::Measurement,
        scope: scope.clone(),
        provenance: provenance.clone(),
        started_ns: window.started_ns,
        ended_ns: window.ended_ns,
        work_units: Some(window.work_units),
        byte_count: Some(window.byte_count),
        observed_value: None,
        group_id: None,
        qg6_sample_binding: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perf::PerfGateArtifact;

    const IDENTITY: &str = "qg1-native/synthetic-zipf-s11-vocab8192-doc4096-v1";

    fn corpus() -> Vec<IndexableDocument> {
        (0..4)
            .map(|ordinal| {
                let mut document = IndexableDocument::new(
                    format!("doc-{ordinal:05}"),
                    format!("term{ordinal:05} generated record body {ordinal}"),
                );
                document.title = Some(format!("title {ordinal}"));
                document
                    .metadata
                    .insert("created_at_ms".to_owned(), ordinal.to_string());
                document
            })
            .collect()
    }

    fn sealed() -> (Vec<IndexableDocument>, ContinuousCorpusManifest) {
        let documents = corpus();
        let manifest = ContinuousCorpusManifest::seal(IDENTITY, 7, "positions_on", &documents)
            .expect("seal corpus");
        (documents, manifest)
    }

    fn good_receipt(manifest: &ContinuousCorpusManifest, engine: &str) -> ContinuousWindowReceipt {
        ContinuousWindowReceipt {
            schema_version: CONTINUOUS_TIMING_SCHEMA_VERSION.to_owned(),
            engine: engine.to_owned(),
            timing_source: TimingSource::ContinuousMonotonicWindow,
            quiescence: if engine == "quill" {
                EngineQuiescence::QuillSealedSynchronousCommit
            } else {
                EngineQuiescence::TantivyMergingThreadsJoined
            },
            feed_calls: 2,
            accepted_docs: manifest.doc_count,
            processed_docs: manifest.doc_count,
            committed_docs: manifest.doc_count,
            searchable_docs: manifest.doc_count,
            accepted_bytes: manifest.total_bytes,
            processed_bytes: manifest.total_bytes,
            committed_bytes: ByteStageObservation::Observed(manifest.total_bytes),
            searchable_bytes: ByteStageObservation::StructurallyUnobservable {
                seam: "test seam: engine exposes doc_count only".to_owned(),
            },
            periodic_commits: 0,
            percall_sum_ns: 4_000,
            timeline: ContinuousPhaseTimeline {
                feed_complete_ns: 3_000,
                commit_complete_ns: 5_000,
                searchable_verified_ns: 6_000,
                quiescence_joined_ns: 7_000,
                window_total_ns: 7_000,
            },
            terminal_probe_query: "term00001".to_owned(),
            terminal_probe_hits: 1,
        }
    }

    #[test]
    fn seal_binds_counts_bytes_fields_and_digest() {
        let (documents, manifest) = sealed();
        assert_eq!(manifest.doc_count, 4);
        assert_eq!(manifest.titled_docs, 4);
        assert_eq!(manifest.metadata_entries, 4);
        assert!(manifest.total_bytes > 0);
        assert_eq!(
            manifest.total_bytes,
            manifest.id_bytes
                + manifest.title_bytes
                + manifest.content_bytes
                + manifest.metadata_bytes
        );
        assert_eq!(manifest.corpus_xxh3.len(), 16);
        manifest.verify(&documents).expect("immutability holds");
    }

    #[test]
    fn corpus_mutation_is_rejected() {
        let (mut documents, manifest) = sealed();
        documents[2].content.push('x');
        assert!(matches!(
            manifest.verify(&documents),
            Err(ContinuousTimingError::CorpusShapeMismatch { field, .. }) if field == "content_bytes"
        ));
        let (mut documents, manifest) = sealed();
        // Same bytes, different order: only the digest can catch it.
        documents.swap(0, 3);
        assert!(matches!(
            manifest.verify(&documents),
            Err(ContinuousTimingError::CorpusDigestMismatch { .. })
        ));
        let (mut documents, manifest) = sealed();
        // Equal-length key rename: byte totals are unchanged, so only the
        // canonical metadata digest can catch it.
        let value = documents[1]
            .metadata
            .remove("created_at_ms")
            .expect("metadata fixture");
        documents[1]
            .metadata
            .insert("created_by_ms".to_owned(), value);
        assert!(matches!(
            manifest.verify(&documents),
            Err(ContinuousTimingError::CorpusDigestMismatch { .. })
        ));
    }

    #[test]
    fn non_native_corpus_identity_is_rejected() {
        let documents = corpus();
        assert!(matches!(
            ContinuousCorpusManifest::seal(
                "qg6-derived/query-latency-corpus",
                7,
                "positions_on",
                &documents
            ),
            Err(ContinuousTimingError::NonNativeCorpusIdentity { .. })
        ));
        assert!(matches!(
            ContinuousCorpusManifest::seal(IDENTITY, 7, "positions_on", &[]),
            Err(ContinuousTimingError::EmptyCorpus)
        ));
    }

    #[test]
    fn valid_receipts_pass_for_both_engines() {
        let (_, manifest) = sealed();
        good_receipt(&manifest, "quill")
            .validate(&manifest)
            .expect("quill receipt");
        good_receipt(&manifest, "tantivy")
            .validate(&manifest)
            .expect("tantivy receipt");
    }

    #[test]
    fn summed_per_call_timing_is_rejected() {
        let (_, manifest) = sealed();
        let mut receipt = good_receipt(&manifest, "quill");
        receipt.timing_source = TimingSource::SummedPerCall;
        assert_eq!(
            receipt.validate(&manifest),
            Err(ContinuousTimingError::SummedPerCallTiming)
        );
    }

    #[test]
    fn every_work_equality_stage_is_enforced() {
        let (_, manifest) = sealed();
        for stage in [
            "accepted_docs",
            "processed_docs",
            "committed_docs",
            "searchable_docs",
        ] {
            let mut receipt = good_receipt(&manifest, "quill");
            match stage {
                "accepted_docs" => receipt.accepted_docs -= 1,
                "processed_docs" => receipt.processed_docs -= 1,
                "committed_docs" => receipt.committed_docs += 1,
                // Early stop: fewer searchable documents than fed.
                _ => receipt.searchable_docs -= 1,
            }
            assert!(
                matches!(
                    receipt.validate(&manifest),
                    Err(ContinuousTimingError::WorkInequality { stage: got, .. }) if got == stage
                ),
                "stage {stage} must fail closed"
            );
        }
        let mut receipt = good_receipt(&manifest, "quill");
        receipt.accepted_bytes -= 1;
        assert!(matches!(
            receipt.validate(&manifest),
            Err(ContinuousTimingError::WorkInequality {
                stage: "accepted_bytes",
                ..
            })
        ));
        let mut receipt = good_receipt(&manifest, "quill");
        receipt.processed_bytes += 1;
        assert!(matches!(
            receipt.validate(&manifest),
            Err(ContinuousTimingError::WorkInequality {
                stage: "processed_bytes",
                ..
            })
        ));
        let mut receipt = good_receipt(&manifest, "quill");
        receipt.committed_bytes = ByteStageObservation::Observed(manifest.total_bytes - 1);
        assert!(matches!(
            receipt.validate(&manifest),
            Err(ContinuousTimingError::WorkInequality {
                stage: "committed_bytes",
                ..
            })
        ));
        let mut receipt = good_receipt(&manifest, "quill");
        receipt.searchable_bytes = ByteStageObservation::Observed(manifest.total_bytes + 1);
        assert!(matches!(
            receipt.validate(&manifest),
            Err(ContinuousTimingError::WorkInequality {
                stage: "searchable_bytes",
                ..
            })
        ));
    }

    #[test]
    fn unobservable_byte_stage_must_name_its_seam() {
        let (_, manifest) = sealed();
        let mut receipt = good_receipt(&manifest, "quill");
        receipt.searchable_bytes = ByteStageObservation::StructurallyUnobservable {
            seam: "  ".to_owned(),
        };
        assert!(matches!(
            receipt.validate(&manifest),
            Err(ContinuousTimingError::UnnamedUnobservableSeam {
                stage: "searchable_bytes"
            })
        ));
        // A named gap is admitted (H1 stays open behind it) and an observed,
        // equal byte total is admitted too.
        let mut receipt = good_receipt(&manifest, "quill");
        receipt.committed_bytes = ByteStageObservation::StructurallyUnobservable {
            seam: "LexicalRead exposes doc_count only".to_owned(),
        };
        receipt.validate(&manifest).expect("named seam is admitted");
    }

    #[test]
    fn document_bytes_matches_the_sealed_manifest_definition() {
        let (documents, manifest) = sealed();
        let summed: u64 = documents.iter().map(document_bytes).sum();
        assert_eq!(
            summed, manifest.total_bytes,
            "window-time byte observation and manifest sealing must share one definition"
        );
    }

    #[test]
    fn phase_disorder_and_early_clock_stop_are_rejected() {
        let (_, manifest) = sealed();
        let mut receipt = good_receipt(&manifest, "quill");
        receipt.timeline.commit_complete_ns = receipt.timeline.feed_complete_ns - 1;
        assert!(matches!(
            receipt.validate(&manifest),
            Err(ContinuousTimingError::PhaseOrderViolation {
                phase: "commit_complete"
            })
        ));
        // The clock must not stop before quiescence is joined...
        let mut receipt = good_receipt(&manifest, "quill");
        receipt.timeline.window_total_ns = receipt.timeline.quiescence_joined_ns - 1;
        assert!(matches!(
            receipt.validate(&manifest),
            Err(ContinuousTimingError::ClockStoppedBeforeQuiescence { .. })
        ));
        // ...nor keep running after it (an excluded tail is also a lie).
        let mut receipt = good_receipt(&manifest, "quill");
        receipt.timeline.window_total_ns = receipt.timeline.quiescence_joined_ns + 1;
        assert!(matches!(
            receipt.validate(&manifest),
            Err(ContinuousTimingError::ClockStoppedBeforeQuiescence { .. })
        ));
        let mut receipt = good_receipt(&manifest, "quill");
        receipt.percall_sum_ns = receipt.timeline.window_total_ns + 1;
        assert!(matches!(
            receipt.validate(&manifest),
            Err(ContinuousTimingError::PhaseOrderViolation {
                phase: "percall_sum_within_window"
            })
        ));
        let mut receipt = good_receipt(&manifest, "quill");
        receipt.feed_calls = 0;
        assert_eq!(
            receipt.validate(&manifest),
            Err(ContinuousTimingError::NoFeedCalls)
        );
    }

    #[test]
    fn pending_merge_shapes_are_rejected() {
        let (_, manifest) = sealed();
        // A Tantivy window that never joined its merging threads cannot claim
        // Quill's synchronous-commit quiescence.
        let mut receipt = good_receipt(&manifest, "tantivy");
        receipt.quiescence = EngineQuiescence::QuillSealedSynchronousCommit;
        assert!(matches!(
            receipt.validate(&manifest),
            Err(ContinuousTimingError::QuiescenceContractMismatch { .. })
        ));
        let mut receipt = good_receipt(&manifest, "quill");
        receipt.quiescence = EngineQuiescence::TantivyMergingThreadsJoined;
        assert!(matches!(
            receipt.validate(&manifest),
            Err(ContinuousTimingError::QuiescenceContractMismatch { .. })
        ));
        let mut receipt = good_receipt(&manifest, "quill");
        receipt.engine = "unknown-engine".to_owned();
        assert!(matches!(
            receipt.validate(&manifest),
            Err(ContinuousTimingError::QuiescenceContractMismatch { .. })
        ));
    }

    #[test]
    fn missing_terminal_probe_is_rejected() {
        let (_, manifest) = sealed();
        let mut receipt = good_receipt(&manifest, "quill");
        receipt.terminal_probe_hits = 0;
        assert_eq!(
            receipt.validate(&manifest),
            Err(ContinuousTimingError::MissingTerminalProbe)
        );
        let mut receipt = good_receipt(&manifest, "quill");
        receipt.terminal_probe_query.clear();
        assert_eq!(
            receipt.validate(&manifest),
            Err(ContinuousTimingError::MissingTerminalProbe)
        );
    }

    #[test]
    fn throughput_is_derived_never_a_stored_gauge() {
        let (_, manifest) = sealed();
        let receipt = good_receipt(&manifest, "quill");
        // 4 docs over 7000ns.
        let expected = 4.0 * 1_000_000_000.0 / 7_000.0;
        assert!((receipt.docs_per_second() - expected).abs() < 1e-9);

        let scope =
            continuous_throughput_scope("qg-1.bulk/tiny/1/positions_on.docs_per_second", "docs/s");
        assert_eq!(scope.semantics, PerfMetricSemantics::Throughput);
        assert_eq!(scope.version, 2);
        assert!(scope.operation_id.ends_with(".continuous"));

        let provenance = sample_provenance();
        let identity = ContinuousSampleIdentity {
            block_id: 0,
            sample_id: 0,
            arm: PerfSampleArm::Treatment,
            order: PerfSampleOrder::First,
        };
        let window = ContinuousSampleWindow {
            started_ns: 10,
            ended_ns: 7_010,
            work_units: 4,
            byte_count: manifest.total_bytes,
        };
        let sample = continuous_raw_sample(identity, &scope, &provenance, &window)
            .expect("continuous sample");
        assert_eq!(sample.observed_value, None);
        assert_eq!(sample.work_units, Some(4));
        assert_eq!(sample.byte_count, Some(manifest.total_bytes));

        // A gauge scope is structurally rejected.
        let mut gauge_scope = scope.clone();
        gauge_scope.semantics = PerfMetricSemantics::GaugeHigherIsBetter;
        assert!(matches!(
            continuous_raw_sample(identity, &gauge_scope, &provenance, &window),
            Err(ContinuousTimingError::GaugeSemanticsRejected { .. })
        ));
        let zero_work = ContinuousSampleWindow {
            work_units: 0,
            ..window
        };
        assert_eq!(
            continuous_raw_sample(identity, &scope, &provenance, &zero_work),
            Err(ContinuousTimingError::ZeroWorkUnits)
        );
        let inverted = ContinuousSampleWindow {
            started_ns: 7_010,
            ended_ns: 10,
            ..window
        };
        assert!(matches!(
            continuous_raw_sample(identity, &scope, &provenance, &inverted),
            Err(ContinuousTimingError::PhaseOrderViolation {
                phase: "sample_window"
            })
        ));
    }

    fn sample_provenance() -> PerfSampleProvenance {
        PerfSampleProvenance {
            run_id: "continuous-test".to_owned(),
            executable_sha256: "e".repeat(64),
            corpus_sha256: "c".repeat(64),
            input_identity: None,
            worker_id: "test-worker".to_owned(),
            build_profile: "dev".to_owned(),
        }
    }

    #[test]
    fn timing_mode_parsing_defaults_to_per_call() {
        assert_eq!(TimingMode::parse(None), Ok(TimingMode::PerCall));
        assert_eq!(TimingMode::parse(Some("per-call")), Ok(TimingMode::PerCall));
        assert_eq!(
            TimingMode::parse(Some("continuous")),
            Ok(TimingMode::Continuous)
        );
        assert!(matches!(
            TimingMode::parse(Some("percall")),
            Err(ContinuousTimingError::InvalidTimingMode { .. })
        ));
        assert!(TimingMode::Continuous.is_continuous());
        assert!(!TimingMode::PerCall.is_continuous());
    }

    #[test]
    fn artifact_field_is_schema_additive() {
        let (_, manifest) = sealed();
        let evidence = ContinuousTimingEvidence {
            schema_version: CONTINUOUS_TIMING_SCHEMA_VERSION.to_owned(),
            timing_mode: TimingMode::Continuous.label().to_owned(),
            cells: vec![ContinuousCellEvidence {
                schema_version: CONTINUOUS_TIMING_SCHEMA_VERSION.to_owned(),
                fixture: "bulk/tiny/1/positions_on".to_owned(),
                timing_mode: TimingMode::Continuous.label().to_owned(),
                corpus: manifest.clone(),
                rounds_quill: 10,
                rounds_tantivy: 30,
                last_quill_receipt: Some(good_receipt(&manifest, "quill")),
                last_tantivy_receipt: Some(good_receipt(&manifest, "tantivy")),
                all_windows_validated: true,
            }],
        };
        let encoded = serde_json::to_string(&evidence).expect("encode evidence");
        let decoded: ContinuousTimingEvidence =
            serde_json::from_str(&encoded).expect("decode evidence");
        assert_eq!(decoded, evidence);

        // A legacy v3 artifact without the field still deserializes, and a
        // per-call artifact serializes byte-identically to the legacy shape.
        let legacy_json = r#"{
            "schema_version": "quill-perf-artifact-v3",
            "gate": "Qg1",
            "bench_elf_sha256": "abc",
            "machine_fingerprint": "m",
            "git_rev": "r",
            "run_window": "w",
            "run_id": "i",
            "corpus_manifest_hash": "h",
            "manifest_sha256": "s",
            "cells": [],
            "laws_attested": false
        }"#;
        let artifact: PerfGateArtifact =
            serde_json::from_str(legacy_json).expect("legacy artifact decodes");
        assert!(artifact.continuous_timing.is_none());
        let reencoded = serde_json::to_string(&artifact).expect("re-encode");
        assert!(
            !reencoded.contains("continuous_timing"),
            "per-call artifacts must not grow the new field"
        );
        let mut artifact = artifact;
        artifact.continuous_timing = Some(evidence);
        let with_field = serde_json::to_string(&artifact).expect("encode with field");
        assert!(with_field.contains("continuous_timing"));
    }
}
