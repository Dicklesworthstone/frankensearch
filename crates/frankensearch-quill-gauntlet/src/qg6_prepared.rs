//! Prepared, parity-gated four-arm execution for the QG-6 query benchmark.
//!
//! The generic runner deliberately owns the lifecycle boundary: engines are
//! constructed and populated through [`Qg6PreparedExperiment::prepare_with`],
//! validated for exact or explicitly proven semantic result parity, warmed
//! equally, and only then exposed to the timed schedule. This keeps corpus
//! construction, commits, configuration, warmup, and parity checks outside
//! every timed interval.

use std::hint::black_box;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

const QG6_QUERY_MANIFEST_VERSION: &str = "frankensearch-qg6-query-manifest-v1";
const QG6_RESULT_DIGEST_VERSION: &str = "frankensearch-qg6-ordered-result-v1";
const MAX_QUERY_COUNT: usize = 4_096;
const MAX_QUERY_ID_BYTES: usize = 256;
const MAX_QUERY_TEXT_BYTES: usize = 16 * 1_024;
const MAX_DOC_ID_BYTES: usize = 4 * 1_024;
const MAX_K: usize = 100_000;

/// The four independent logical indexes in the QG-6 admission experiment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qg6ArmRole {
    /// Left side of the Tantivy/Tantivy null comparison.
    NullLeft,
    /// Right side of the Tantivy/Tantivy null comparison.
    NullRight,
    /// Tantivy side of the Quill/Tantivy effect comparison.
    EffectControl,
    /// Quill side of the Quill/Tantivy effect comparison.
    EffectTreatment,
}

impl Qg6ArmRole {
    /// Stable order used by preparation, preflight, and lifecycle receipts.
    pub const ALL: [Self; 4] = [
        Self::NullLeft,
        Self::NullRight,
        Self::EffectControl,
        Self::EffectTreatment,
    ];

    const fn index(self) -> usize {
        match self {
            Self::NullLeft => 0,
            Self::NullRight => 1,
            Self::EffectControl => 2,
            Self::EffectTreatment => 3,
        }
    }
}

/// Which independently measured pair a scheduled block belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qg6Comparison {
    /// Tantivy/Tantivy A/A null.
    Null,
    /// Tantivy/Quill A/B effect.
    Effect,
}

/// First or second position inside one paired timing block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qg6SampleOrder {
    /// First arm invoked in the block.
    First,
    /// Second arm invoked in the block.
    Second,
}

/// Lifecycle phase attached to a bounded failure diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qg6Phase {
    /// Engine construction, population, and commit.
    Prepare,
    /// Exact ordered-result validation before timing.
    Preflight,
    /// Equal untimed warmup.
    Warmup,
    /// Timed measurement.
    Measurement,
}

/// Whether the selected cells constitute the complete QG-6 gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qg6SelectionScope {
    /// Every normative QG-6 cell is present.
    CompleteGate,
    /// An explicit fixture filter selected only part of QG-6.
    FilteredPreAdmission,
}

/// Claim state contributed by the QG-6 selection boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qg6SelectionClaim {
    /// Selection completeness does not itself forbid gate validation.
    EligibleForGateValidation,
    /// Partial evidence is diagnostic and cannot support a publication claim.
    NoClaim,
}

impl Qg6SelectionScope {
    /// Classify a selected QG-6 slice against the normative cell count.
    ///
    /// # Errors
    ///
    /// Rejects an empty normative gate, an empty selection, or a selected
    /// count greater than the normative matrix.
    pub fn from_cell_counts(selected: usize, normative: usize) -> Result<Self, Qg6HarnessError> {
        if normative == 0 || selected == 0 || selected > normative {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: "QG-6 selected/normative cell counts are inconsistent".to_owned(),
            });
        }
        Ok(if selected == normative {
            Self::CompleteGate
        } else {
            Self::FilteredPreAdmission
        })
    }

    /// Convert an explicit selection boundary into its fail-closed claim state.
    #[must_use]
    pub const fn claim(self) -> Qg6SelectionClaim {
        match self {
            Self::CompleteGate => Qg6SelectionClaim::EligibleForGateValidation,
            Self::FilteredPreAdmission => Qg6SelectionClaim::NoClaim,
        }
    }
}

/// One frozen query in the prepared QG-6 workload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Qg6QuerySpec {
    id: String,
    text: String,
}

impl Qg6QuerySpec {
    /// Construct a bounded query. Diagnostics retain only `id`, never `text`.
    ///
    /// # Errors
    ///
    /// Rejects empty or oversized IDs and empty or oversized query text.
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Result<Self, Qg6HarnessError> {
        let id = id.into();
        let text = text.into();
        if id.is_empty() || id.len() > MAX_QUERY_ID_BYTES {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: "query ID must be non-empty and at most 256 bytes".to_owned(),
            });
        }
        if text.is_empty() || text.len() > MAX_QUERY_TEXT_BYTES {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: "query text must be non-empty and at most 16384 bytes".to_owned(),
            });
        }
        Ok(Self { id, text })
    }

    /// Stable, non-sensitive query identifier.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Query text supplied only to the engine adapter.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }
}

/// Immutable corpus, query, and semantic configuration identity shared by all arms.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Qg6ExperimentIdentity {
    /// SHA-256 of the exact ordered corpus.
    pub corpus_sha256: String,
    /// SHA-256 of the frozen ordered query manifest.
    pub query_manifest_sha256: String,
    /// SHA-256 of the cross-engine semantic configuration contract.
    pub config_contract_sha256: String,
    /// Documents supplied to each logical index.
    pub document_count: u64,
    /// Requested result count for every query.
    pub k: usize,
}

/// A search result whose optional native digest is checked outside the timer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qg6SearchResult {
    ordered_doc_ids: Vec<String>,
    claimed_sha256: Option<String>,
}

impl Qg6SearchResult {
    /// Wrap exact ordered document IDs without doing digest work in the adapter.
    #[must_use]
    pub fn from_ordered_doc_ids(ordered_doc_ids: Vec<String>) -> Self {
        Self {
            ordered_doc_ids,
            claimed_sha256: None,
        }
    }

    /// Wrap IDs plus a native digest. The runner independently recomputes it.
    pub fn with_claimed_sha256(
        ordered_doc_ids: Vec<String>,
        claimed_sha256: impl Into<String>,
    ) -> Self {
        Self {
            ordered_doc_ids,
            claimed_sha256: Some(claimed_sha256.into()),
        }
    }
}

impl From<Vec<String>> for Qg6SearchResult {
    fn from(value: Vec<String>) -> Self {
        Self::from_ordered_doc_ids(value)
    }
}

/// Stable result facts retained for parity and post-timing stability checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Qg6ResultReceipt {
    /// Exact number of returned hits.
    pub hit_count: usize,
    /// SHA-256 over length-delimited ordered document IDs.
    pub ordered_doc_ids_sha256: String,
}

/// Per-arm lifecycle counts proving preparation never re-enters measurement.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Qg6ArmLifecycle {
    /// Engine constructor invocations.
    pub build_calls: u64,
    /// Population batches supplied to the engine.
    pub populate_calls: u64,
    /// Total documents supplied across population batches.
    pub populated_documents: u64,
    /// Commit invocations.
    pub commit_calls: u64,
    /// Untimed exact-parity queries.
    pub preflight_search_calls: u64,
    /// Untimed warmup queries.
    pub warmup_search_calls: u64,
    /// Timed queries.
    pub timed_search_calls: u64,
    /// Setup operations observed after timing began. This must remain zero.
    pub timed_setup_calls: u64,
}

/// Complete lifecycle receipt in stable arm order.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Qg6LifecycleReceipt {
    /// One counter set for each [`Qg6ArmRole::ALL`] entry.
    pub arms: [Qg6ArmLifecycle; 4],
}

impl Qg6LifecycleReceipt {
    /// Counters for one logical arm.
    #[must_use]
    pub fn arm(&self, role: Qg6ArmRole) -> &Qg6ArmLifecycle {
        &self.arms[role.index()]
    }

    fn arm_mut(&mut self, role: Qg6ArmRole) -> &mut Qg6ArmLifecycle {
        &mut self.arms[role.index()]
    }
}

/// Mutable setup-only capability passed to the arm builder.
///
/// The type is not retained by the prepared experiment, so setup counters
/// cannot be incremented from warmup or timed search code.
pub struct Qg6SetupRecorder<'a> {
    counters: &'a mut Qg6ArmLifecycle,
}

impl Qg6SetupRecorder<'_> {
    /// Record one population call and its document count.
    pub fn record_population_batch(&mut self, document_count: u64) {
        self.counters.populate_calls = self.counters.populate_calls.saturating_add(1);
        self.counters.populated_documents = self
            .counters
            .populated_documents
            .saturating_add(document_count);
    }

    /// Record the commit that makes this arm searchable.
    pub fn record_commit(&mut self) {
        self.counters.commit_calls = self.counters.commit_calls.saturating_add(1);
    }
}

/// One paired block in the deterministic four-arm schedule.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Qg6PairBlock {
    /// Globally unique block identifier in this experiment.
    pub block_id: u64,
    /// Index into the frozen ordered query manifest.
    pub query_index: usize,
    /// Null or effect comparison.
    pub comparison: Qg6Comparison,
    /// Arm executed first.
    pub first: Qg6ArmRole,
    /// Arm executed second.
    pub second: Qg6ArmRole,
}

/// One directly observed timed sample.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Qg6TimedSample {
    /// Paired block shared with exactly one other arm.
    pub block_id: u64,
    /// Unique sample identifier.
    pub sample_id: u64,
    /// Stable, non-sensitive query ID.
    pub query_id: String,
    /// Index into the frozen ordered query manifest.
    pub query_index: usize,
    /// Null or effect stream.
    pub comparison: Qg6Comparison,
    /// Logical arm.
    pub arm: Qg6ArmRole,
    /// Execution order inside the block.
    pub order: Qg6SampleOrder,
    /// Monotonic start offset relative to the measurement origin.
    pub started_ns: u64,
    /// Monotonic end offset relative to the measurement origin.
    pub ended_ns: u64,
    /// Independently recomputed result digest, outside the timed interval.
    pub result_sha256: String,
}

/// Output of one prepared four-arm measurement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Qg6Measurement {
    /// Frozen identity shared by all four arms.
    pub identity: Qg6ExperimentIdentity,
    /// Seed that fully determines the timed schedule.
    pub schedule_seed: u64,
    /// Equal warmup count per arm and query.
    pub warmup_rounds: usize,
    /// Equal timed pair count per comparison and query.
    pub rounds_per_query: usize,
    /// Interleaved null/effect schedule.
    pub schedule: Vec<Qg6PairBlock>,
    /// Raw per-arm monotonic intervals.
    pub samples: Vec<Qg6TimedSample>,
    /// Lifecycle contamination proof.
    pub lifecycle: Qg6LifecycleReceipt,
}

/// Bounded, non-sensitive failure surface for prepared QG-6 execution.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum Qg6HarnessError {
    /// Invalid immutable input or run configuration.
    #[error("invalid QG-6 prepared-run specification: {reason}")]
    InvalidSpec {
        /// Bounded static reason; never includes corpus or query text.
        reason: String,
    },
    /// An engine adapter failed.
    #[error(
        "QG-6 {phase:?} adapter failure for {arm:?}, query {query_id:?}: \
         sha256={error_sha256}, bytes={error_bytes}"
    )]
    AdapterFailure {
        /// Lifecycle phase.
        phase: Qg6Phase,
        /// Arm being prepared or queried.
        arm: Qg6ArmRole,
        /// Stable query ID, or `"<setup>"`.
        query_id: String,
        /// SHA-256 of the adapter error string.
        error_sha256: String,
        /// Error string byte length.
        error_bytes: usize,
    },
    /// A prepared arm did not ingest the exact declared corpus once.
    #[error("invalid QG-6 setup lifecycle for {arm:?}: {reason}")]
    InvalidSetup {
        /// Logical arm.
        arm: Qg6ArmRole,
        /// Bounded reason.
        reason: String,
    },
    /// A native result digest disagreed with the independent digest.
    #[error(
        "QG-6 {phase:?} result digest mismatch for {arm:?}, query {query_id:?}: \
         claimed={claimed_sha256}, computed={computed_sha256}"
    )]
    ResultDigestMismatch {
        /// Lifecycle phase.
        phase: Qg6Phase,
        /// Logical arm.
        arm: Qg6ArmRole,
        /// Stable query ID.
        query_id: String,
        /// Digest supplied by the adapter.
        claimed_sha256: String,
        /// Independently recomputed digest.
        computed_sha256: String,
    },
    /// A result exceeded the declared `k` or a document ID exceeded its bound.
    #[error("invalid QG-6 {phase:?} result for {arm:?}, query {query_id:?}: {reason}")]
    InvalidResult {
        /// Lifecycle phase.
        phase: Qg6Phase,
        /// Logical arm.
        arm: Qg6ArmRole,
        /// Stable query ID.
        query_id: String,
        /// Bounded reason with no raw document ID.
        reason: String,
    },
    /// Exact hit counts differ across arms before timing.
    #[error(
        "QG-6 preflight hit-count mismatch for query {query_id:?}: \
         {expected_arm:?}={expected_count}, {observed_arm:?}={observed_count}"
    )]
    HitCountMismatch {
        /// Stable query ID.
        query_id: String,
        /// Baseline arm.
        expected_arm: Qg6ArmRole,
        /// Compared arm.
        observed_arm: Qg6ArmRole,
        /// Baseline hit count.
        expected_count: usize,
        /// Compared hit count.
        observed_count: usize,
    },
    /// Exact ordered document IDs differ across arms before timing.
    #[error(
        "QG-6 preflight order mismatch for query {query_id:?} at rank \
         {first_differing_rank}: {expected_arm:?} doc={expected_doc_sha256}, \
         {observed_arm:?} doc={observed_doc_sha256}"
    )]
    OrderedDocIdsMismatch {
        /// Stable query ID.
        query_id: String,
        /// Baseline arm.
        expected_arm: Qg6ArmRole,
        /// Compared arm.
        observed_arm: Qg6ArmRole,
        /// First zero-based differing rank.
        first_differing_rank: usize,
        /// SHA-256 of the baseline document ID.
        expected_doc_sha256: String,
        /// SHA-256 of the compared document ID.
        observed_doc_sha256: String,
    },
    /// An explicit semantic-parity proof rejected one preflight comparison.
    #[error(
        "QG-6 preflight semantic-parity failure for query {query_id:?}: \
         {expected_arm:?} vs {observed_arm:?}, sha256={error_sha256}, bytes={error_bytes}"
    )]
    SemanticParityFailure {
        /// Stable query ID.
        query_id: String,
        /// Baseline arm.
        expected_arm: Qg6ArmRole,
        /// Compared arm.
        observed_arm: Qg6ArmRole,
        /// SHA-256 of the bounded adapter diagnostic.
        error_sha256: String,
        /// Diagnostic byte length.
        error_bytes: usize,
    },
    /// A warmup or timed result drifted from its accepted preflight receipt.
    #[error(
        "QG-6 {phase:?} result drift for {arm:?}, query {query_id:?}: \
         expected count/digest={expected_count}/{expected_sha256}, \
         observed={observed_count}/{observed_sha256}"
    )]
    ResultDrift {
        /// Warmup or measurement.
        phase: Qg6Phase,
        /// Logical arm.
        arm: Qg6ArmRole,
        /// Stable query ID.
        query_id: String,
        /// Accepted preflight count.
        expected_count: usize,
        /// Observed count.
        observed_count: usize,
        /// Accepted preflight digest.
        expected_sha256: String,
        /// Observed digest.
        observed_sha256: String,
    },
    /// Final lifecycle counts do not match the declared experiment.
    #[error("invalid QG-6 lifecycle receipt: {reason}")]
    LifecycleViolation {
        /// Bounded reason.
        reason: String,
    },
}

struct Qg6FourArms<A> {
    null_left: A,
    null_right: A,
    effect_control: A,
    effect_treatment: A,
}

impl<A> Qg6FourArms<A> {
    fn get(&self, role: Qg6ArmRole) -> &A {
        match role {
            Qg6ArmRole::NullLeft => &self.null_left,
            Qg6ArmRole::NullRight => &self.null_right,
            Qg6ArmRole::EffectControl => &self.effect_control,
            Qg6ArmRole::EffectTreatment => &self.effect_treatment,
        }
    }
}

/// Four independently built arms before result parity has been established.
pub struct Qg6PreparedExperiment<A> {
    identity: Qg6ExperimentIdentity,
    queries: Vec<Qg6QuerySpec>,
    arms: Qg6FourArms<A>,
    lifecycle: Qg6LifecycleReceipt,
}

/// Prepared arms whose complete frozen query set passed result parity.
pub struct Qg6ValidatedExperiment<A> {
    prepared: Qg6PreparedExperiment<A>,
    expected_results: Vec<Qg6FourArms<Qg6ResultReceipt>>,
}

impl<A> Qg6PreparedExperiment<A> {
    /// Build, populate, and commit four independent arms exactly once.
    ///
    /// The builder must record every population batch and the searchable
    /// commit through [`Qg6SetupRecorder`]. All four builders receive the same
    /// immutable identity.
    ///
    /// # Errors
    ///
    /// Rejects malformed hashes, query manifests, result limits, adapter
    /// failures, population mismatches, and missing or duplicate commits.
    pub fn prepare_with<F>(
        corpus_sha256: impl Into<String>,
        config_contract_sha256: impl Into<String>,
        document_count: u64,
        k: usize,
        queries: Vec<Qg6QuerySpec>,
        mut build: F,
    ) -> Result<Self, Qg6HarnessError>
    where
        F: FnMut(
            Qg6ArmRole,
            &Qg6ExperimentIdentity,
            &mut Qg6SetupRecorder<'_>,
        ) -> Result<A, String>,
    {
        validate_experiment_inputs(document_count, k, &queries)?;
        let corpus_sha256 = corpus_sha256.into();
        let config_contract_sha256 = config_contract_sha256.into();
        if !is_lower_hex_sha256(&corpus_sha256) || !is_lower_hex_sha256(&config_contract_sha256) {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: "corpus and configuration identities must be lowercase SHA-256 values"
                    .to_owned(),
            });
        }
        let identity = Qg6ExperimentIdentity {
            corpus_sha256,
            query_manifest_sha256: query_manifest_sha256(&queries),
            config_contract_sha256,
            document_count,
            k,
        };
        let mut lifecycle = Qg6LifecycleReceipt::default();
        let null_left = build_one(Qg6ArmRole::NullLeft, &identity, &mut lifecycle, &mut build)?;
        let null_right = build_one(Qg6ArmRole::NullRight, &identity, &mut lifecycle, &mut build)?;
        let effect_control = build_one(
            Qg6ArmRole::EffectControl,
            &identity,
            &mut lifecycle,
            &mut build,
        )?;
        let effect_treatment = build_one(
            Qg6ArmRole::EffectTreatment,
            &identity,
            &mut lifecycle,
            &mut build,
        )?;
        Ok(Self {
            identity,
            queries,
            arms: Qg6FourArms {
                null_left,
                null_right,
                effect_control,
                effect_treatment,
            },
            lifecycle,
        })
    }

    /// Validate every frozen query against all four arms before timing.
    ///
    /// # Errors
    ///
    /// Stops at the first adapter, digest, hit-count, or exact-order failure.
    pub fn validate_exact_parity<F>(
        self,
        search: &mut F,
    ) -> Result<Qg6ValidatedExperiment<A>, Qg6HarnessError>
    where
        F: FnMut(&A, &Qg6QuerySpec, usize) -> Result<Qg6SearchResult, String>,
    {
        self.validate_exact_parity_with(search, &mut |result| result)
    }

    /// Validate exact parity while normalizing engine-native result types
    /// outside any future timing boundary.
    ///
    /// # Errors
    ///
    /// Stops at the first adapter, digest, hit-count, or exact-order failure.
    pub fn validate_exact_parity_with<R, F, N>(
        mut self,
        search: &mut F,
        normalize: &mut N,
    ) -> Result<Qg6ValidatedExperiment<A>, Qg6HarnessError>
    where
        F: FnMut(&A, &Qg6QuerySpec, usize) -> Result<R, String>,
        N: FnMut(R) -> Qg6SearchResult,
    {
        let mut expected_results = Vec::with_capacity(self.queries.len());
        for query in &self.queries {
            let null_left = invoke_search(
                &self.arms,
                query,
                self.identity.k,
                Qg6ArmRole::NullLeft,
                Qg6Phase::Preflight,
                search,
                normalize,
            )?;
            self.lifecycle
                .arm_mut(Qg6ArmRole::NullLeft)
                .preflight_search_calls += 1;
            let null_right = invoke_search(
                &self.arms,
                query,
                self.identity.k,
                Qg6ArmRole::NullRight,
                Qg6Phase::Preflight,
                search,
                normalize,
            )?;
            self.lifecycle
                .arm_mut(Qg6ArmRole::NullRight)
                .preflight_search_calls += 1;
            let effect_control = invoke_search(
                &self.arms,
                query,
                self.identity.k,
                Qg6ArmRole::EffectControl,
                Qg6Phase::Preflight,
                search,
                normalize,
            )?;
            self.lifecycle
                .arm_mut(Qg6ArmRole::EffectControl)
                .preflight_search_calls += 1;
            let effect_treatment = invoke_search(
                &self.arms,
                query,
                self.identity.k,
                Qg6ArmRole::EffectTreatment,
                Qg6Phase::Preflight,
                search,
                normalize,
            )?;
            self.lifecycle
                .arm_mut(Qg6ArmRole::EffectTreatment)
                .preflight_search_calls += 1;
            for (role, observed) in [
                (Qg6ArmRole::NullRight, &null_right),
                (Qg6ArmRole::EffectControl, &effect_control),
                (Qg6ArmRole::EffectTreatment, &effect_treatment),
            ] {
                compare_exact(query.id(), Qg6ArmRole::NullLeft, &null_left, role, observed)?;
            }
            expected_results.push(Qg6FourArms {
                null_left: null_left.receipt,
                null_right: null_right.receipt,
                effect_control: effect_control.receipt,
                effect_treatment: effect_treatment.receipt,
            });
        }
        Ok(Qg6ValidatedExperiment {
            prepared: self,
            expected_results,
        })
    }

    /// Validate parity with an explicit untimed semantic proof.
    ///
    /// The preflight adapter may return engine-native score and cutoff-tie
    /// evidence. `normalize` extracts the exact native top-k result whose
    /// per-arm receipt must remain stable during warmup and measurement, while
    /// `compare` proves that each compared result is semantically equivalent
    /// to the baseline. This permits reviewed native tie-order differences
    /// without requiring every engine to return the same external ID at a
    /// top-k cutoff.
    ///
    /// # Errors
    ///
    /// Stops at the first adapter, digest, result-shape, or semantic-parity
    /// failure.
    pub fn validate_semantic_parity_with<R, F, N, C>(
        mut self,
        search: &mut F,
        normalize: &mut N,
        compare: &mut C,
    ) -> Result<Qg6ValidatedExperiment<A>, Qg6HarnessError>
    where
        F: FnMut(&A, &Qg6QuerySpec, usize) -> Result<R, String>,
        N: FnMut(&R) -> Qg6SearchResult,
        C: FnMut(&Qg6QuerySpec, Qg6ArmRole, &R, Qg6ArmRole, &R) -> Result<(), String>,
    {
        let mut expected_results = Vec::with_capacity(self.queries.len());
        for query in &self.queries {
            let (null_left_native, null_left) = invoke_search_borrowed(
                &self.arms,
                query,
                self.identity.k,
                Qg6ArmRole::NullLeft,
                search,
                normalize,
            )?;
            self.lifecycle
                .arm_mut(Qg6ArmRole::NullLeft)
                .preflight_search_calls += 1;
            let (null_right_native, null_right) = invoke_search_borrowed(
                &self.arms,
                query,
                self.identity.k,
                Qg6ArmRole::NullRight,
                search,
                normalize,
            )?;
            self.lifecycle
                .arm_mut(Qg6ArmRole::NullRight)
                .preflight_search_calls += 1;
            let (effect_control_native, effect_control) = invoke_search_borrowed(
                &self.arms,
                query,
                self.identity.k,
                Qg6ArmRole::EffectControl,
                search,
                normalize,
            )?;
            self.lifecycle
                .arm_mut(Qg6ArmRole::EffectControl)
                .preflight_search_calls += 1;
            let (effect_treatment_native, effect_treatment) = invoke_search_borrowed(
                &self.arms,
                query,
                self.identity.k,
                Qg6ArmRole::EffectTreatment,
                search,
                normalize,
            )?;
            self.lifecycle
                .arm_mut(Qg6ArmRole::EffectTreatment)
                .preflight_search_calls += 1;
            for (role, observed) in [
                (Qg6ArmRole::NullRight, &null_right_native),
                (Qg6ArmRole::EffectControl, &effect_control_native),
                (Qg6ArmRole::EffectTreatment, &effect_treatment_native),
            ] {
                compare(
                    query,
                    Qg6ArmRole::NullLeft,
                    &null_left_native,
                    role,
                    observed,
                )
                .map_err(|error| {
                    semantic_parity_failure(query.id(), Qg6ArmRole::NullLeft, role, &error)
                })?;
            }
            expected_results.push(Qg6FourArms {
                null_left: null_left.receipt,
                null_right: null_right.receipt,
                effect_control: effect_control.receipt,
                effect_treatment: effect_treatment.receipt,
            });
        }
        Ok(Qg6ValidatedExperiment {
            prepared: self,
            expected_results,
        })
    }
}

impl<A> Qg6ValidatedExperiment<A> {
    /// Run equal warmups and an interleaved, balanced four-arm timing schedule.
    ///
    /// Result digest computation and stability checks occur after each timed
    /// interval. The engine adapter receives only an already prepared arm.
    ///
    /// # Errors
    ///
    /// Rejects invalid run counts, adapter failures, post-preflight result
    /// drift, or any lifecycle-count mismatch.
    pub fn measure<F>(
        self,
        warmup_rounds: usize,
        rounds_per_query: usize,
        schedule_seed: u64,
        search: &mut F,
    ) -> Result<Qg6Measurement, Qg6HarnessError>
    where
        F: FnMut(&A, &Qg6QuerySpec, usize) -> Result<Qg6SearchResult, String>,
    {
        self.measure_with_normalizer(
            warmup_rounds,
            rounds_per_query,
            schedule_seed,
            search,
            &mut |result| result,
        )
    }

    /// Run the prepared measurement while converting engine-native results
    /// only after each timed interval has ended.
    ///
    /// # Errors
    ///
    /// Rejects invalid run counts, adapter failures, post-preflight result
    /// drift, or any lifecycle-count mismatch.
    pub fn measure_with_normalizer<R, F, N>(
        mut self,
        warmup_rounds: usize,
        rounds_per_query: usize,
        schedule_seed: u64,
        search: &mut F,
        normalize: &mut N,
    ) -> Result<Qg6Measurement, Qg6HarnessError>
    where
        F: FnMut(&A, &Qg6QuerySpec, usize) -> Result<R, String>,
        N: FnMut(R) -> Qg6SearchResult,
    {
        if warmup_rounds == 0 {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: "QG-6 prepared measurement requires at least one warmup per arm and query"
                    .to_owned(),
            });
        }
        let schedule = seeded_interleaved_four_arm_schedule(
            self.prepared.queries.len(),
            rounds_per_query,
            schedule_seed,
        )?;
        self.run_warmups(warmup_rounds, schedule_seed, search, normalize)?;

        let origin = Instant::now();
        let mut samples = Vec::with_capacity(schedule.len() * 2);
        for block in &schedule {
            let query = &self.prepared.queries[block.query_index];
            for (order, role) in [
                (Qg6SampleOrder::First, block.first),
                (Qg6SampleOrder::Second, block.second),
            ] {
                let started_ns = monotonic_ns(origin);
                let result = search(
                    self.prepared.arms.get(role),
                    black_box(query),
                    black_box(self.prepared.identity.k),
                );
                let mut ended_ns = monotonic_ns(origin);
                if ended_ns <= started_ns {
                    ended_ns = started_ns.saturating_add(1);
                }
                self.prepared.lifecycle.arm_mut(role).timed_search_calls += 1;
                let result = result.map_err(|error| {
                    adapter_failure(Qg6Phase::Measurement, role, query.id(), &error)
                })?;
                let observed = observe_result(
                    normalize(result),
                    self.prepared.identity.k,
                    Qg6Phase::Measurement,
                    role,
                    query.id(),
                )?;
                ensure_stable(
                    Qg6Phase::Measurement,
                    role,
                    query.id(),
                    self.expected_results[block.query_index].get(role),
                    &observed.receipt,
                )?;
                let sample_id = block
                    .block_id
                    .checked_mul(2)
                    .and_then(|base| base.checked_add(u64::from(order == Qg6SampleOrder::Second)))
                    .ok_or_else(|| Qg6HarnessError::InvalidSpec {
                        reason: "timed sample ID overflow".to_owned(),
                    })?;
                samples.push(Qg6TimedSample {
                    block_id: block.block_id,
                    sample_id,
                    query_id: query.id().to_owned(),
                    query_index: block.query_index,
                    comparison: block.comparison,
                    arm: role,
                    order,
                    started_ns,
                    ended_ns,
                    result_sha256: observed.receipt.ordered_doc_ids_sha256,
                });
            }
        }
        verify_lifecycle(
            &self.prepared.lifecycle,
            self.prepared.identity.document_count,
            self.prepared.queries.len(),
            warmup_rounds,
            rounds_per_query,
        )?;
        Ok(Qg6Measurement {
            identity: self.prepared.identity,
            schedule_seed,
            warmup_rounds,
            rounds_per_query,
            schedule,
            samples,
            lifecycle: self.prepared.lifecycle,
        })
    }

    fn run_warmups<R, F, N>(
        &mut self,
        warmup_rounds: usize,
        schedule_seed: u64,
        search: &mut F,
        normalize: &mut N,
    ) -> Result<(), Qg6HarnessError>
    where
        F: FnMut(&A, &Qg6QuerySpec, usize) -> Result<R, String>,
        N: FnMut(R) -> Qg6SearchResult,
    {
        for round in 0..warmup_rounds {
            for (query_index, query) in self.prepared.queries.iter().enumerate() {
                let mut roles = Qg6ArmRole::ALL;
                let salt = usize_to_u64(round)?.wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    ^ usize_to_u64(query_index)?;
                shuffle(&mut roles, schedule_seed ^ salt);
                for role in roles {
                    let observed = invoke_search(
                        &self.prepared.arms,
                        query,
                        self.prepared.identity.k,
                        role,
                        Qg6Phase::Warmup,
                        search,
                        normalize,
                    )?;
                    self.prepared.lifecycle.arm_mut(role).warmup_search_calls += 1;
                    ensure_stable(
                        Qg6Phase::Warmup,
                        role,
                        query.id(),
                        self.expected_results[query_index].get(role),
                        &observed.receipt,
                    )?;
                    black_box(observed);
                }
            }
        }
        Ok(())
    }
}

/// Construct the deterministic schedule used by the prepared QG-6 runner.
///
/// Every query receives `rounds_per_query` null and effect blocks. Each
/// two-block unit contains one null and one effect comparison, while the
/// comparison order and both within-pair arm orders are independently balanced.
///
/// # Errors
///
/// Requires at least one query and two rounds per query, and rejects arithmetic
/// overflow.
pub fn seeded_interleaved_four_arm_schedule(
    query_count: usize,
    rounds_per_query: usize,
    seed: u64,
) -> Result<Vec<Qg6PairBlock>, Qg6HarnessError> {
    if query_count == 0 || query_count > MAX_QUERY_COUNT {
        return Err(Qg6HarnessError::InvalidSpec {
            reason: "QG-6 schedule query count is outside 1..=4096".to_owned(),
        });
    }
    if rounds_per_query < 2 {
        return Err(Qg6HarnessError::InvalidSpec {
            reason: "QG-6 schedule requires at least two rounds per query".to_owned(),
        });
    }
    let unit_count =
        query_count
            .checked_mul(rounds_per_query)
            .ok_or_else(|| Qg6HarnessError::InvalidSpec {
                reason: "QG-6 schedule unit count overflow".to_owned(),
            })?;
    let mut query_units = Vec::with_capacity(unit_count);
    for _ in 0..rounds_per_query {
        query_units.extend(0..query_count);
    }
    shuffle(&mut query_units, seed ^ 0x8d58_ac26_afe1_2e47);
    let comparison_order = balanced_bools(unit_count, seed ^ 0x243f_6a88_85a3_08d3);
    let null_left_first = balanced_bools(unit_count, seed ^ 0x1319_8a2e_0370_7344);
    let effect_control_first = balanced_bools(unit_count, seed ^ 0xa409_3822_299f_31d0);
    let block_capacity = unit_count
        .checked_mul(2)
        .ok_or_else(|| Qg6HarnessError::InvalidSpec {
            reason: "QG-6 block count overflow".to_owned(),
        })?;
    let mut schedule = Vec::with_capacity(block_capacity);
    for (unit_index, query_index) in query_units.into_iter().enumerate() {
        let null = pair_roles(Qg6Comparison::Null, null_left_first[unit_index]);
        let effect = pair_roles(Qg6Comparison::Effect, effect_control_first[unit_index]);
        let pairs = if comparison_order[unit_index] {
            [null, effect]
        } else {
            [effect, null]
        };
        for (comparison, first, second) in pairs {
            schedule.push(Qg6PairBlock {
                block_id: usize_to_u64(schedule.len())?,
                query_index,
                comparison,
                first,
                second,
            });
        }
    }
    Ok(schedule)
}

fn pair_roles(
    comparison: Qg6Comparison,
    control_first: bool,
) -> (Qg6Comparison, Qg6ArmRole, Qg6ArmRole) {
    let (control, treatment) = match comparison {
        Qg6Comparison::Null => (Qg6ArmRole::NullLeft, Qg6ArmRole::NullRight),
        Qg6Comparison::Effect => (Qg6ArmRole::EffectControl, Qg6ArmRole::EffectTreatment),
    };
    if control_first {
        (comparison, control, treatment)
    } else {
        (comparison, treatment, control)
    }
}

fn build_one<A, F>(
    role: Qg6ArmRole,
    identity: &Qg6ExperimentIdentity,
    lifecycle: &mut Qg6LifecycleReceipt,
    build: &mut F,
) -> Result<A, Qg6HarnessError>
where
    F: FnMut(Qg6ArmRole, &Qg6ExperimentIdentity, &mut Qg6SetupRecorder<'_>) -> Result<A, String>,
{
    let counters = lifecycle.arm_mut(role);
    counters.build_calls = counters.build_calls.saturating_add(1);
    let mut recorder = Qg6SetupRecorder { counters };
    let arm = build(role, identity, &mut recorder)
        .map_err(|error| adapter_failure(Qg6Phase::Prepare, role, "<setup>", &error))?;
    validate_setup(role, identity, recorder.counters)?;
    Ok(arm)
}

fn validate_setup(
    role: Qg6ArmRole,
    identity: &Qg6ExperimentIdentity,
    counters: &Qg6ArmLifecycle,
) -> Result<(), Qg6HarnessError> {
    if counters.build_calls != 1 {
        return Err(invalid_setup(
            role,
            "engine must be constructed exactly once",
        ));
    }
    if counters.populate_calls == 0 {
        return Err(invalid_setup(
            role,
            "engine must receive at least one population batch",
        ));
    }
    if counters.populated_documents != identity.document_count {
        return Err(invalid_setup(
            role,
            "populated document count must equal the frozen corpus count",
        ));
    }
    if counters.commit_calls != 1 {
        return Err(invalid_setup(
            role,
            "engine must commit exactly once before preflight",
        ));
    }
    Ok(())
}

fn invalid_setup(role: Qg6ArmRole, reason: &str) -> Qg6HarnessError {
    Qg6HarnessError::InvalidSetup {
        arm: role,
        reason: reason.to_owned(),
    }
}

struct ObservedResult {
    ordered_doc_ids: Vec<String>,
    receipt: Qg6ResultReceipt,
}

fn invoke_search<A, R, F, N>(
    arms: &Qg6FourArms<A>,
    query: &Qg6QuerySpec,
    k: usize,
    role: Qg6ArmRole,
    phase: Qg6Phase,
    search: &mut F,
    normalize: &mut N,
) -> Result<ObservedResult, Qg6HarnessError>
where
    F: FnMut(&A, &Qg6QuerySpec, usize) -> Result<R, String>,
    N: FnMut(R) -> Qg6SearchResult,
{
    let result = search(arms.get(role), black_box(query), black_box(k))
        .map_err(|error| adapter_failure(phase, role, query.id(), &error))?;
    observe_result(normalize(result), k, phase, role, query.id())
}

fn invoke_search_borrowed<A, R, F, N>(
    arms: &Qg6FourArms<A>,
    query: &Qg6QuerySpec,
    k: usize,
    role: Qg6ArmRole,
    search: &mut F,
    normalize: &mut N,
) -> Result<(R, ObservedResult), Qg6HarnessError>
where
    F: FnMut(&A, &Qg6QuerySpec, usize) -> Result<R, String>,
    N: FnMut(&R) -> Qg6SearchResult,
{
    let native = search(arms.get(role), black_box(query), black_box(k))
        .map_err(|error| adapter_failure(Qg6Phase::Preflight, role, query.id(), &error))?;
    let observed = observe_result(normalize(&native), k, Qg6Phase::Preflight, role, query.id())?;
    Ok((native, observed))
}

fn observe_result(
    result: Qg6SearchResult,
    k: usize,
    phase: Qg6Phase,
    role: Qg6ArmRole,
    query_id: &str,
) -> Result<ObservedResult, Qg6HarnessError> {
    if result.ordered_doc_ids.len() > k {
        return Err(Qg6HarnessError::InvalidResult {
            phase,
            arm: role,
            query_id: query_id.to_owned(),
            reason: "returned hit count exceeds the declared k".to_owned(),
        });
    }
    if result
        .ordered_doc_ids
        .iter()
        .any(|doc_id| doc_id.len() > MAX_DOC_ID_BYTES)
    {
        return Err(Qg6HarnessError::InvalidResult {
            phase,
            arm: role,
            query_id: query_id.to_owned(),
            reason: "returned document ID exceeds 4096 bytes".to_owned(),
        });
    }
    let computed_sha256 = ordered_doc_ids_sha256(&result.ordered_doc_ids);
    if let Some(claimed_sha256) = result.claimed_sha256 {
        if claimed_sha256 != computed_sha256 {
            return Err(Qg6HarnessError::ResultDigestMismatch {
                phase,
                arm: role,
                query_id: query_id.to_owned(),
                claimed_sha256,
                computed_sha256,
            });
        }
    }
    Ok(ObservedResult {
        receipt: Qg6ResultReceipt {
            hit_count: result.ordered_doc_ids.len(),
            ordered_doc_ids_sha256: computed_sha256,
        },
        ordered_doc_ids: result.ordered_doc_ids,
    })
}

fn compare_exact(
    query_id: &str,
    expected_arm: Qg6ArmRole,
    expected: &ObservedResult,
    observed_arm: Qg6ArmRole,
    observed: &ObservedResult,
) -> Result<(), Qg6HarnessError> {
    if expected.receipt.hit_count != observed.receipt.hit_count {
        return Err(Qg6HarnessError::HitCountMismatch {
            query_id: query_id.to_owned(),
            expected_arm,
            observed_arm,
            expected_count: expected.receipt.hit_count,
            observed_count: observed.receipt.hit_count,
        });
    }
    if expected.ordered_doc_ids != observed.ordered_doc_ids {
        let rank = expected
            .ordered_doc_ids
            .iter()
            .zip(&observed.ordered_doc_ids)
            .position(|(left, right)| left != right)
            .expect("equal counts and unequal vectors have a differing rank");
        return Err(Qg6HarnessError::OrderedDocIdsMismatch {
            query_id: query_id.to_owned(),
            expected_arm,
            observed_arm,
            first_differing_rank: rank,
            expected_doc_sha256: sha256_hex(expected.ordered_doc_ids[rank].as_bytes()),
            observed_doc_sha256: sha256_hex(observed.ordered_doc_ids[rank].as_bytes()),
        });
    }
    if expected.receipt.ordered_doc_ids_sha256 != observed.receipt.ordered_doc_ids_sha256 {
        return Err(Qg6HarnessError::ResultDigestMismatch {
            phase: Qg6Phase::Preflight,
            arm: observed_arm,
            query_id: query_id.to_owned(),
            claimed_sha256: observed.receipt.ordered_doc_ids_sha256.clone(),
            computed_sha256: expected.receipt.ordered_doc_ids_sha256.clone(),
        });
    }
    Ok(())
}

fn semantic_parity_failure(
    query_id: &str,
    expected_arm: Qg6ArmRole,
    observed_arm: Qg6ArmRole,
    error: &str,
) -> Qg6HarnessError {
    Qg6HarnessError::SemanticParityFailure {
        query_id: query_id.to_owned(),
        expected_arm,
        observed_arm,
        error_sha256: sha256_hex(error.as_bytes()),
        error_bytes: error.len(),
    }
}

fn ensure_stable(
    phase: Qg6Phase,
    role: Qg6ArmRole,
    query_id: &str,
    expected: &Qg6ResultReceipt,
    observed: &Qg6ResultReceipt,
) -> Result<(), Qg6HarnessError> {
    if expected != observed {
        return Err(Qg6HarnessError::ResultDrift {
            phase,
            arm: role,
            query_id: query_id.to_owned(),
            expected_count: expected.hit_count,
            observed_count: observed.hit_count,
            expected_sha256: expected.ordered_doc_ids_sha256.clone(),
            observed_sha256: observed.ordered_doc_ids_sha256.clone(),
        });
    }
    Ok(())
}

fn verify_lifecycle(
    lifecycle: &Qg6LifecycleReceipt,
    document_count: u64,
    query_count: usize,
    warmup_rounds: usize,
    rounds_per_query: usize,
) -> Result<(), Qg6HarnessError> {
    let expected_preflight = usize_to_u64(query_count)?;
    let expected_warmups =
        usize_to_u64(query_count.checked_mul(warmup_rounds).ok_or_else(|| {
            Qg6HarnessError::LifecycleViolation {
                reason: "warmup call count overflow".to_owned(),
            }
        })?)?;
    let expected_timed =
        usize_to_u64(query_count.checked_mul(rounds_per_query).ok_or_else(|| {
            Qg6HarnessError::LifecycleViolation {
                reason: "timed call count overflow".to_owned(),
            }
        })?)?;
    for role in Qg6ArmRole::ALL {
        let arm = lifecycle.arm(role);
        if arm.build_calls != 1
            || arm.populate_calls == 0
            || arm.populated_documents != document_count
            || arm.commit_calls != 1
            || arm.preflight_search_calls != expected_preflight
            || arm.warmup_search_calls != expected_warmups
            || arm.timed_search_calls != expected_timed
            || arm.timed_setup_calls != 0
        {
            return Err(Qg6HarnessError::LifecycleViolation {
                reason: format!(
                    "arm {role:?} counts differ from build=1, populated_documents={document_count}, \
                     commit=1, preflight={expected_preflight}, warmup={expected_warmups}, \
                     timed={expected_timed}, timed_setup=0"
                ),
            });
        }
    }
    Ok(())
}

fn validate_experiment_inputs(
    document_count: u64,
    k: usize,
    queries: &[Qg6QuerySpec],
) -> Result<(), Qg6HarnessError> {
    if document_count == 0 {
        return Err(Qg6HarnessError::InvalidSpec {
            reason: "QG-6 corpus must contain at least one document".to_owned(),
        });
    }
    if k == 0 || k > MAX_K {
        return Err(Qg6HarnessError::InvalidSpec {
            reason: "QG-6 result limit is outside 1..=100000".to_owned(),
        });
    }
    if queries.is_empty() || queries.len() > MAX_QUERY_COUNT {
        return Err(Qg6HarnessError::InvalidSpec {
            reason: "QG-6 query count is outside 1..=4096".to_owned(),
        });
    }
    let mut ids = queries
        .iter()
        .map(|query| query.id.as_str())
        .collect::<Vec<_>>();
    ids.sort_unstable();
    if ids.windows(2).any(|window| window[0] == window[1]) {
        return Err(Qg6HarnessError::InvalidSpec {
            reason: "QG-6 query IDs must be unique".to_owned(),
        });
    }
    Ok(())
}

fn adapter_failure(
    phase: Qg6Phase,
    arm: Qg6ArmRole,
    query_id: &str,
    error: &str,
) -> Qg6HarnessError {
    Qg6HarnessError::AdapterFailure {
        phase,
        arm,
        query_id: query_id.to_owned(),
        error_sha256: sha256_hex(error.as_bytes()),
        error_bytes: error.len(),
    }
}

fn query_manifest_sha256(queries: &[Qg6QuerySpec]) -> String {
    let mut hasher = Sha256::new();
    hash_len_prefixed(&mut hasher, QG6_QUERY_MANIFEST_VERSION.as_bytes());
    for query in queries {
        hash_len_prefixed(&mut hasher, query.id.as_bytes());
        hash_len_prefixed(&mut hasher, query.text.as_bytes());
    }
    lower_hex(hasher.finalize())
}

fn ordered_doc_ids_sha256(doc_ids: &[String]) -> String {
    let mut hasher = Sha256::new();
    hash_len_prefixed(&mut hasher, QG6_RESULT_DIGEST_VERSION.as_bytes());
    hasher.update(usize_to_u64_infallible(doc_ids.len()).to_le_bytes());
    for doc_id in doc_ids {
        hash_len_prefixed(&mut hasher, doc_id.as_bytes());
    }
    lower_hex(hasher.finalize())
}

fn hash_len_prefixed(hasher: &mut Sha256, value: &[u8]) {
    hasher.update(usize_to_u64_infallible(value.len()).to_le_bytes());
    hasher.update(value);
}

fn sha256_hex(bytes: &[u8]) -> String {
    lower_hex(Sha256::digest(bytes))
}

fn lower_hex(bytes: impl AsRef<[u8]>) -> String {
    let bytes = bytes.as_ref();
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn balanced_bools(count: usize, seed: u64) -> Vec<bool> {
    let mut values = (0..count)
        .map(|index| index < count / 2)
        .collect::<Vec<_>>();
    shuffle(&mut values, seed);
    values
}

fn shuffle<T>(values: &mut [T], mut seed: u64) {
    for index in (1..values.len()).rev() {
        seed = splitmix64(seed);
        let modulus = usize_to_u64_infallible(index + 1);
        let swap_index = usize::try_from(seed % modulus).expect("shuffle index fits usize");
        values.swap(index, swap_index);
    }
}

const fn splitmix64(mut state: u64) -> u64 {
    state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    state = (state ^ (state >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    state = (state ^ (state >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    state ^ (state >> 31)
}

fn monotonic_ns(origin: Instant) -> u64 {
    u64::try_from(origin.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn usize_to_u64(value: usize) -> Result<u64, Qg6HarnessError> {
    u64::try_from(value).map_err(|_| Qg6HarnessError::InvalidSpec {
        reason: "QG-6 count does not fit u64".to_owned(),
    })
}

fn usize_to_u64_infallible(value: usize) -> u64 {
    u64::try_from(value).expect("bounded QG-6 length fits u64")
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;

    #[derive(Debug)]
    struct FakeArm {
        role: Qg6ArmRole,
    }

    fn queries() -> Vec<Qg6QuerySpec> {
        vec![
            Qg6QuerySpec::new("identifier-0", "term00042").expect("query"),
            Qg6QuerySpec::new("identifier-1", "term00137").expect("query"),
            Qg6QuerySpec::new("identifier-2", "term00256").expect("query"),
            Qg6QuerySpec::new("identifier-3", "term00301").expect("query"),
        ]
    }

    fn prepare() -> Qg6PreparedExperiment<FakeArm> {
        Qg6PreparedExperiment::prepare_with(
            "a".repeat(64),
            "b".repeat(64),
            100_000,
            10,
            queries(),
            |role, identity, setup| {
                setup.record_population_batch(identity.document_count / 2);
                setup
                    .record_population_batch(identity.document_count - identity.document_count / 2);
                setup.record_commit();
                Ok(FakeArm { role })
            },
        )
        .expect("prepared experiment")
    }

    fn canonical_result(query: &Qg6QuerySpec) -> Vec<String> {
        vec![
            format!("{}-doc-0", query.id()),
            format!("{}-doc-1", query.id()),
        ]
    }

    #[test]
    fn schedule_is_deterministic_balanced_and_interleaves_comparisons() {
        let first = seeded_interleaved_four_arm_schedule(4, 11, 0x5155_494c).expect("schedule");
        let second = seeded_interleaved_four_arm_schedule(4, 11, 0x5155_494c).expect("schedule");
        assert_eq!(first, second);
        assert_eq!(first.len(), 4 * 11 * 2);

        let mut query_comparison_counts = BTreeMap::new();
        let mut first_counts = BTreeMap::new();
        for pair in first.as_chunks::<2>().0 {
            assert_ne!(pair[0].comparison, pair[1].comparison);
            for block in pair {
                *query_comparison_counts
                    .entry((block.query_index, block.comparison))
                    .or_insert(0_usize) += 1;
                *first_counts.entry(block.first).or_insert(0_usize) += 1;
            }
        }
        for query_index in 0..4 {
            assert_eq!(
                query_comparison_counts[&(query_index, Qg6Comparison::Null)],
                11
            );
            assert_eq!(
                query_comparison_counts[&(query_index, Qg6Comparison::Effect)],
                11
            );
        }
        assert!(
            first_counts
                .values()
                .max()
                .expect("first counts")
                .abs_diff(*first_counts.values().min().expect("first counts"))
                <= 1
        );
    }

    #[test]
    fn exact_parity_and_measurement_produce_complete_lifecycle_receipt() {
        let mut search = |arm: &FakeArm, query: &Qg6QuerySpec, _k: usize| {
            black_box(arm.role);
            Ok(Qg6SearchResult::from(canonical_result(query)))
        };
        let validated = prepare()
            .validate_exact_parity(&mut search)
            .expect("exact parity");
        let measurement = validated
            .measure(2, 10, 0x5eed, &mut search)
            .expect("measurement");

        assert_eq!(measurement.samples.len(), 4 * 10 * 4);
        assert_eq!(
            measurement
                .samples
                .iter()
                .map(|sample| sample.sample_id)
                .collect::<BTreeSet<_>>()
                .len(),
            measurement.samples.len()
        );
        for role in Qg6ArmRole::ALL {
            let lifecycle = measurement.lifecycle.arm(role);
            assert_eq!(lifecycle.build_calls, 1);
            assert_eq!(lifecycle.populate_calls, 2);
            assert_eq!(lifecycle.populated_documents, 100_000);
            assert_eq!(lifecycle.commit_calls, 1);
            assert_eq!(lifecycle.preflight_search_calls, 4);
            assert_eq!(lifecycle.warmup_search_calls, 8);
            assert_eq!(lifecycle.timed_search_calls, 40);
            assert_eq!(lifecycle.timed_setup_calls, 0);
        }
    }

    #[test]
    fn semantic_parity_retains_each_arms_native_result_receipt() {
        let native_result = |role: Qg6ArmRole, query: &Qg6QuerySpec| {
            let mut ids = canonical_result(query);
            if role == Qg6ArmRole::EffectTreatment {
                ids[1].push_str("-native-tie-choice");
            }
            (role, ids)
        };
        let mut preflight =
            |arm: &FakeArm, query: &Qg6QuerySpec, _k: usize| Ok(native_result(arm.role, query));
        let mut normalize =
            |result: &(Qg6ArmRole, Vec<String>)| Qg6SearchResult::from(result.1.clone());
        let mut compare = |_query: &Qg6QuerySpec,
                           _expected_role: Qg6ArmRole,
                           expected: &(Qg6ArmRole, Vec<String>),
                           observed_role: Qg6ArmRole,
                           observed: &(Qg6ArmRole, Vec<String>)| {
            if observed_role == Qg6ArmRole::EffectTreatment || expected.1 == observed.1 {
                Ok(())
            } else {
                Err("non-treatment native result changed".to_owned())
            }
        };
        let validated = prepare()
            .validate_semantic_parity_with(&mut preflight, &mut normalize, &mut compare)
            .expect("semantic tie-envelope parity");
        let mut timed_search = |arm: &FakeArm, query: &Qg6QuerySpec, _k: usize| {
            Ok(Qg6SearchResult::from(native_result(arm.role, query).1))
        };
        let measurement = validated
            .measure(1, 2, 0x5eed, &mut timed_search)
            .expect("per-arm native receipts remain stable");

        assert!(
            measurement
                .samples
                .iter()
                .any(|sample| sample.arm == Qg6ArmRole::EffectTreatment)
        );
    }

    #[test]
    fn semantic_parity_failure_hashes_adapter_diagnostic() {
        let canary = "SECRET-SEMANTIC-PARITY-DIAGNOSTIC";
        let mut preflight = |arm: &FakeArm, query: &Qg6QuerySpec, _k: usize| {
            Ok((arm.role, canonical_result(query)))
        };
        let mut normalize =
            |result: &(Qg6ArmRole, Vec<String>)| Qg6SearchResult::from(result.1.clone());
        let mut compare = |_query: &Qg6QuerySpec,
                           _expected_role: Qg6ArmRole,
                           _expected: &(Qg6ArmRole, Vec<String>),
                           observed_role: Qg6ArmRole,
                           _observed: &(Qg6ArmRole, Vec<String>)| {
            if observed_role == Qg6ArmRole::EffectTreatment {
                Err(canary.to_owned())
            } else {
                Ok(())
            }
        };
        let error = prepare()
            .validate_semantic_parity_with(&mut preflight, &mut normalize, &mut compare)
            .err()
            .expect("semantic parity rejection");
        assert!(matches!(
            error,
            Qg6HarnessError::SemanticParityFailure {
                observed_arm: Qg6ArmRole::EffectTreatment,
                ..
            }
        ));
        assert!(!error.to_string().contains(canary));
    }

    #[test]
    fn preflight_rejects_hit_count_mismatch_before_timing() {
        let mut search = |arm: &FakeArm, query: &Qg6QuerySpec, _k: usize| {
            let mut result = canonical_result(query);
            if arm.role == Qg6ArmRole::EffectTreatment {
                result.pop();
            }
            Ok(Qg6SearchResult::from(result))
        };
        let error = prepare()
            .validate_exact_parity(&mut search)
            .err()
            .expect("count mismatch");
        assert!(matches!(
            error,
            Qg6HarnessError::HitCountMismatch {
                observed_arm: Qg6ArmRole::EffectTreatment,
                ..
            }
        ));
    }

    #[test]
    fn preflight_rejects_order_mismatch_without_exposing_doc_ids() {
        let canary = "SECRET-CANARY-DOC-ID";
        let mut search = |arm: &FakeArm, query: &Qg6QuerySpec, _k: usize| {
            let mut result = vec![canary.to_owned(), format!("{}-other", query.id())];
            if arm.role == Qg6ArmRole::NullRight {
                result.swap(0, 1);
            }
            Ok(Qg6SearchResult::from(result))
        };
        let error = prepare()
            .validate_exact_parity(&mut search)
            .err()
            .expect("order mismatch");
        assert!(matches!(
            error,
            Qg6HarnessError::OrderedDocIdsMismatch {
                observed_arm: Qg6ArmRole::NullRight,
                first_differing_rank: 0,
                ..
            }
        ));
        assert!(!error.to_string().contains(canary));
    }

    #[test]
    fn preflight_rejects_claimed_digest_mismatch() {
        let mut search = |_arm: &FakeArm, query: &Qg6QuerySpec, _k: usize| {
            Ok(Qg6SearchResult::with_claimed_sha256(
                canonical_result(query),
                "f".repeat(64),
            ))
        };
        let error = prepare()
            .validate_exact_parity(&mut search)
            .err()
            .expect("digest mismatch");
        assert!(matches!(
            error,
            Qg6HarnessError::ResultDigestMismatch {
                phase: Qg6Phase::Preflight,
                arm: Qg6ArmRole::NullLeft,
                ..
            }
        ));
    }

    #[test]
    fn measurement_rejects_result_drift_after_preflight() {
        let calls = std::cell::Cell::new(0_usize);
        let preflight_calls = queries().len() * Qg6ArmRole::ALL.len();
        let mut search = |_arm: &FakeArm, query: &Qg6QuerySpec, _k: usize| {
            let call = calls.get();
            calls.set(call + 1);
            let mut result = canonical_result(query);
            if call >= preflight_calls {
                result[0].push_str("-drift");
            }
            Ok(Qg6SearchResult::from(result))
        };
        let validated = prepare()
            .validate_exact_parity(&mut search)
            .expect("preflight");
        let error = validated
            .measure(1, 2, 7, &mut search)
            .expect_err("warmup drift");
        assert!(matches!(
            error,
            Qg6HarnessError::ResultDrift {
                phase: Qg6Phase::Warmup,
                ..
            }
        ));
    }

    #[test]
    fn setup_requires_exact_population_and_one_commit() {
        let error = Qg6PreparedExperiment::prepare_with(
            "a".repeat(64),
            "b".repeat(64),
            100,
            10,
            queries(),
            |role, _identity, setup| {
                setup.record_population_batch(99);
                setup.record_commit();
                Ok(FakeArm { role })
            },
        )
        .err()
        .expect("population mismatch");
        assert!(matches!(
            error,
            Qg6HarnessError::InvalidSetup {
                arm: Qg6ArmRole::NullLeft,
                ..
            }
        ));
    }

    #[test]
    fn filtered_fixture_is_always_no_claim() {
        assert_eq!(
            Qg6SelectionScope::from_cell_counts(1, 20).expect("filtered"),
            Qg6SelectionScope::FilteredPreAdmission
        );
        assert_eq!(
            Qg6SelectionScope::from_cell_counts(20, 20).expect("complete"),
            Qg6SelectionScope::CompleteGate
        );
        assert_eq!(
            Qg6SelectionScope::FilteredPreAdmission.claim(),
            Qg6SelectionClaim::NoClaim
        );
        assert_eq!(
            Qg6SelectionScope::CompleteGate.claim(),
            Qg6SelectionClaim::EligibleForGateValidation
        );
    }

    #[test]
    fn error_diagnostics_hash_adapter_text() {
        let canary = "SECRET-ADAPTER-FAILURE";
        let mut search = |_arm: &FakeArm, _query: &Qg6QuerySpec, _k: usize| Err(canary.to_owned());
        let error = prepare()
            .validate_exact_parity(&mut search)
            .err()
            .expect("adapter error");
        assert!(matches!(error, Qg6HarnessError::AdapterFailure { .. }));
        assert!(!error.to_string().contains(canary));
    }
}
