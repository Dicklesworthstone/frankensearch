//! Index refresh worker (asupersync background task).
//!
//! [`RefreshWorker`] periodically drains the [`EmbeddingQueue`],
//! embeds documents in batches, and rebuilds the vector index. It runs as an
//! asupersync task within a structured concurrency region.
//!
//! # Single-writer guarantee
//!
//! The worker is the **only** component that writes to vector indices. All
//! reads go through the [`IndexCache`] which provides
//! atomic snapshot replacement.
//!
//! # Identity-bound refresh (bd-9xuj T2 C4-write)
//!
//! Every embedding harvested by this worker is identity-bound at the embedder
//! boundary ([`Embedder::embed_batch_bound`]) and carried as a
//! [`BoundQueryEmbedding`]; no raw vector enters a [`RefreshRecord`]. Two
//! lanes exist:
//!
//! - **Canonical bootstrap lane** ([`RefreshWorker::run_cycle`]): admissible
//!   only while the canonical generation retains nothing (missing artifacts
//!   or an empty legacy v1 seed). It publishes through the legacy two-tier
//!   writer with the producing identity declared process-locally
//!   (`TwoTierIndexBuilder::set_*_identity`) — DECLARED, never attested.
//! - **Staged identity-bound replacement**
//!   ([`RefreshWorker::stage_identity_bound_generation`]): the typed merge
//!   that replaces the former blanket refusal. It admits the existing
//!   generation's ATTESTED FSVI v2 identity through
//!   [`VectorIndex::open_admitted_v2`] (plain [`VectorIndex::open`] is
//!   strictly v1 and can never see a v2 identity), joins every bound
//!   embedding against it, and republishes a complete FSVI v2 replacement via
//!   [`VectorIndex::create_v2`] into a non-canonical staging directory.
//!
//! # Canonical publication is gated (composite generation authority)
//!
//! Installing a staged fast/quality pair over the canonical filenames is a
//! SPLIT publication — two renames with no atomic pair authority. Until the
//! composite generation-authority primitive lands (bd-xomn.1/.3), canonical
//! publication of a fully admitted v2 replacement is refused with the typed
//! `composite-generation-authority-unavailable` reason, *before any queue
//! drain*, so the permanent condition can never consume retry budget or
//! drop queued documents. [`RefreshWorker::publish_staged_canonical`] pins
//! the same refusal at the staged seam.
//!
//! # Pre-drain classification is read-only and performs real admission (r2)
//!
//! The r2 successor of the NO-GO'd C4-write slice (868c0801) repairs the
//! pre-drain seam in both directions:
//! - what it CLAIMS it now PERFORMS: attested v2 tiers are fully admitted at
//!   classification ([`VectorIndex::open_admitted_v2`]: exact binding,
//!   recomputed content/docset digests) with the sealed owners retained, so
//!   a header-valid/content-corrupt artifact fails with its own typed
//!   corruption error instead of the composite-authority refusal;
//! - what it MUST NOT do it no longer does: v1 tiers are classified through
//!   the read-only [`frankensearch_index::two_tier::observe_tier`] — the
//!   mutable [`VectorIndex::open`] (stale-WAL deletion, corrupt-trailer
//!   truncation) never runs during classification, so a mixed v2+v1
//!   generation can never mutate while being classified for refusal.
//!
//! # Lifecycle
//!
//! The worker loops until the parent `Cx` is cancelled. On cancellation it
//! finishes the current batch before exiting.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use asupersync::Cx;
use sha2::{Digest, Sha256};
use tracing::{debug, error, info, warn};

use frankensearch_core::config::TwoTierConfig;
use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::generation::{
    ArtifactGenerationIdentityV1, EmbeddingIdentityBundleV1, QuantizationFormat,
};
use frankensearch_core::traits::{Embedder, IdentityBoundEmbedding};
use frankensearch_core::{BoundQueryEmbedding, SpaceIdentityAdmission};
use frankensearch_index::two_tier::{FsviTierObservation, observe_tier};
use frankensearch_index::{
    FsviAdmissionError, FsviV2IdentityBinding, FsviV2IdentityMetadata, TwoTierIndex,
    TwoTierIndexPaths, VECTOR_INDEX_FALLBACK_FILENAME, VECTOR_INDEX_FAST_FILENAME,
    VECTOR_INDEX_QUALITY_FILENAME, ValidatedFsviBytes, VectorIndex, VectorMetadata,
};

use crate::cache::IndexCache;
use crate::queue::{EmbeddingJob, EmbeddingQueue, JobOutcome};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the index refresh worker.
#[derive(Debug, Clone)]
pub struct RefreshWorkerConfig {
    /// How often to poll the queue for new jobs. Default: 1000ms.
    pub poll_interval: Duration,
    /// Maximum documents to embed per refresh cycle. Default: 1000.
    pub max_docs_per_cycle: usize,
    /// Directory where vector indices are written. Required.
    pub index_dir: PathBuf,
    /// `TwoTierConfig` for newly built indices.
    pub index_config: TwoTierConfig,
}

impl RefreshWorkerConfig {
    /// Create a config with the given index directory and defaults.
    #[must_use]
    pub fn new(index_dir: impl Into<PathBuf>) -> Self {
        Self {
            poll_interval: Duration::from_secs(1),
            max_docs_per_cycle: 1000,
            index_dir: index_dir.into(),
            index_config: TwoTierConfig::default(),
        }
    }

    /// Override the poll interval.
    #[must_use]
    pub const fn with_poll_interval(mut self, interval: Duration) -> Self {
        self.poll_interval = interval;
        self
    }

    /// Override the max docs per cycle.
    #[must_use]
    pub const fn with_max_docs_per_cycle(mut self, max: usize) -> Self {
        self.max_docs_per_cycle = max;
        self
    }

    /// Override the index config.
    #[must_use]
    pub fn with_index_config(mut self, config: TwoTierConfig) -> Self {
        self.index_config = config;
        self
    }
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Lock-free counters for refresh worker telemetry.
#[derive(Debug, Default)]
pub struct RefreshMetrics {
    /// Total refresh cycles executed.
    pub cycles: AtomicU64,
    /// Total documents embedded.
    pub docs_embedded: AtomicU64,
    /// Total documents that failed embedding.
    pub docs_failed: AtomicU64,
    /// Total index rebuilds (successful).
    pub index_rebuilds: AtomicU64,
    /// Total index rebuild failures.
    pub rebuild_failures: AtomicU64,
    /// Total embedding time in microseconds.
    pub embed_time_us: AtomicU64,
    /// Total rebuild time in microseconds.
    pub rebuild_time_us: AtomicU64,
}

impl RefreshMetrics {
    /// Snapshot of the current metrics.
    #[must_use]
    pub fn snapshot(&self) -> RefreshMetricsSnapshot {
        RefreshMetricsSnapshot {
            cycles: self.cycles.load(Ordering::Relaxed),
            docs_embedded: self.docs_embedded.load(Ordering::Relaxed),
            docs_failed: self.docs_failed.load(Ordering::Relaxed),
            index_rebuilds: self.index_rebuilds.load(Ordering::Relaxed),
            rebuild_failures: self.rebuild_failures.load(Ordering::Relaxed),
            embed_time_us: self.embed_time_us.load(Ordering::Relaxed),
            rebuild_time_us: self.rebuild_time_us.load(Ordering::Relaxed),
        }
    }
}

/// Point-in-time snapshot of refresh metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefreshMetricsSnapshot {
    /// Total refresh cycles executed.
    pub cycles: u64,
    /// Total documents embedded.
    pub docs_embedded: u64,
    /// Total documents that failed embedding.
    pub docs_failed: u64,
    /// Total index rebuilds (successful).
    pub index_rebuilds: u64,
    /// Total index rebuild failures.
    pub rebuild_failures: u64,
    /// Total embedding time in microseconds.
    pub embed_time_us: u64,
    /// Total rebuild time in microseconds.
    pub rebuild_time_us: u64,
}

// ---------------------------------------------------------------------------
// Refresh record (identity-bound carrier)
// ---------------------------------------------------------------------------

/// A document with its identity-bound embeddings, ready for index insertion
/// (bd-9xuj T2 C4-write).
///
/// Both tiers carry [`BoundQueryEmbedding`] — a vector bound at the embedder
/// boundary to the complete identity bundle that produced it, validated at
/// bind time. The embeddings are produced ONLY via
/// [`Embedder::embed_batch_bound`]; no raw `Vec<f32>` enters this record, so
/// every downstream seam can verify space and producer identity instead of
/// trusting provenance-free floats.
#[derive(Debug)]
struct RefreshRecord {
    doc_id: String,
    fast_embedding: BoundQueryEmbedding,
    quality_embedding: Option<BoundQueryEmbedding>,
    content_hash: String,
}

/// Convert an embedder-bound output into the C1r2 verifier carrier.
///
/// [`BoundQueryEmbedding::new`] re-validates the bundle and precomputes the
/// identity and space fingerprints, so seam checks are string compares.
fn into_bound_query(bound: IdentityBoundEmbedding) -> SearchResult<BoundQueryEmbedding> {
    BoundQueryEmbedding::new(bound.values, bound.identity)
}

// ---------------------------------------------------------------------------
// Canonical generation inspection + typed refusals
// ---------------------------------------------------------------------------

/// Subdirectory of the index dir holding staged (non-canonical) identity-bound
/// v2 replacement generations. Never discovered by `TwoTierIndex::open`.
const STAGED_V2_DIR_NAME: &str = "v2-staged";

/// Typed state of one canonical tier artifact on disk.
enum TierState {
    /// No artifact at the tier's path.
    Missing,
    /// Recognized legacy FSVI v1 bytes, established by READ-ONLY observation
    /// only (r2 repair: classification never runs the mutable
    /// [`VectorIndex::open`], which deletes stale WAL sidecars and truncates
    /// corrupt WAL trailers). `retains_content` is
    /// [`frankensearch_index::two_tier::FsviV1Observation::retains_content`]:
    /// conservative — the main slab's `record_count` counts tombstoned rows
    /// too, so an all-tombstoned v1 tier fails CLOSED here instead of
    /// classifying as bootstrap-replaceable. Flag-level precision returns
    /// when a read-only record-table inspector lands in the index crate root
    /// (the observational-open train).
    LegacyV1 { retains_content: bool },
    /// Identity-complete FSVI v2 header (attested identity available from
    /// the artifact's own bytes). Header recognition only: content admission
    /// (digest recomputation, sealed-owner retention) is performed by
    /// [`RefreshWorker::admit_existing_generation`] through
    /// [`VectorIndex::open_admitted_v2`] before any admissibility claim is
    /// made.
    V2 { metadata: Box<VectorMetadata> },
}

/// Inspect one tier artifact without mutating anything — including its WAL
/// sidecar and its timestamps, to the extent the platform allows.
///
/// All recognition goes through the read-only
/// [`frankensearch_index::two_tier::observe_tier`]: v2 because plain
/// [`VectorIndex::open`] is strictly v1 (any refresh branch keyed on
/// `identity_v2()` from a plain open would be unreachable for on-disk v2),
/// and v1 because the mutable open's WAL side effects
/// (stale-sidecar deletion, corrupt-trailer truncation) must never fire
/// during classification (NO-GO item 2, 868c0801 refresh.rs:244-250).
fn inspect_tier(path: &Path) -> SearchResult<TierState> {
    if !path.exists() {
        return Ok(TierState::Missing);
    }
    match observe_tier(path)? {
        FsviTierObservation::V2IdentityComplete(metadata) => Ok(TierState::V2 { metadata }),
        FsviTierObservation::V1(observation) => Ok(TierState::LegacyV1 {
            retains_content: observation.retains_content(),
        }),
        FsviTierObservation::UpgradeRequired(upgrade) => Err(SearchError::InvalidConfig {
            field: "refresh.index_format".to_owned(),
            value: format!("fsvi-v{}", upgrade.found_version),
            reason: format!(
                "existing generation uses a newer FSVI schema than this reader supports \
                 (through v{}); a reader upgrade is required before any refresh",
                upgrade.supported_version
            ),
        }),
    }
}

/// The landed generation-containment refusal for identityless legacy tiers
/// (origin 5386b39e): equal display ids, revisions, and dimensions cannot
/// prove that two embeddings inhabit the same vector space, so a live v1
/// tier fails closed and requires a full identity-bound rebuild.
fn identityless_refusal(tier: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: format!("refresh.{tier}_index_identity"),
        value: "identityless-fsvi-v1".to_owned(),
        reason: "refusing incremental vector merge because the existing generation has no \
                 complete immutable embedding identity; a full identity-bound rebuild is required"
            .to_owned(),
    }
}

/// An attested quality tier retains content but this worker has no quality
/// embedder, so no identity-bound replacement of that tier can be produced.
fn quality_republication_unavailable() -> SearchError {
    SearchError::InvalidConfig {
        field: "refresh.index_publication".to_owned(),
        value: "identity-bound-republication-unavailable".to_owned(),
        reason: "the attested quality tier retains content but this worker has no quality \
                 embedder to produce its identity-bound replacement; configure a quality \
                 embedder attesting the same producer or run a full rebuild"
            .to_owned(),
    }
}

/// The bd-xomn composite-generation-authority gate (C4-write disposition,
/// r2 wording).
///
/// This refusal is only reachable AFTER the existing generation was FULLY
/// ADMITTED ([`RefreshWorker::admit_existing_generation`]): header identity
/// gates joined, content and docset digests recomputed byte-for-byte via
/// [`VectorIndex::open_admitted_v2`], and the sealed owners retained through
/// the check. What remains unavailable is only canonical INSTALLATION:
/// installing a fast/quality replacement over the canonical filenames is a
/// split two-rename publication with no atomic pair authority. This slice
/// deliberately does not invent one; canonical publication reopens when the
/// composite generation-authority primitive (bd-xomn.1/.3) lands. The
/// refusal fires BEFORE any queue drain so the (currently permanent)
/// condition never consumes retry budget or drops queued documents.
fn composite_authority_refusal(index_dir: &Path) -> SearchError {
    SearchError::InvalidConfig {
        field: "refresh.canonical_publication".to_owned(),
        value: "composite-generation-authority-unavailable".to_owned(),
        reason: format!(
            "the existing generation for {} was fully admitted (header identity gates plus \
             recomputed content/docset digests via exact v2 admission), and an identity-bound \
             replacement can be staged and proven via stage_identity_bound_generation, but \
             canonical installation of a split fast/quality generation pair is refused until \
             the composite generation-authority primitive lands (bd-xomn.1/.3); no per-tier \
             rename sequence can make the pair atomic",
            index_dir.display()
        ),
    }
}

/// Guard 7 at its strictest: only `SameProducer` is admitted in this slice.
///
/// A producer that differs from the expected/attested producer is refused
/// with this typed reason even when it carries a byte-identical golden-vector
/// certificate (i.e. even when the pairing would classify as
/// [`SpaceIdentityAdmission::ConformanceCompatibleProducer`]): certificate
/// equality attests what some implementation once produced, not what the
/// implementation executing THIS merge produces. The conformance-compatible
/// lane reopens when executing-producer attestation (attestation of the
/// running implementation, in code) lands; until then this is a deliberate
/// narrowing, and the copied-certificate attack shape (same space, foreign
/// producer, cloned certificate) is rejected by construction.
fn attestation_unavailable_refusal(
    tier: &str,
    query_producer_fingerprint: &str,
    expected_producer_fingerprint: &str,
) -> SearchError {
    SearchError::InvalidConfig {
        field: format!("refresh.{tier}_producer_conformance"),
        value: "executing-producer-attestation-unavailable".to_owned(),
        reason: format!(
            "producer fingerprint {query_producer_fingerprint} shares the {tier} tier's \
             embedding space but is not the attested producer \
             {expected_producer_fingerprint}; this slice admits SameProducer only — a \
             golden-vector certificate alone cannot attest the implementation executing \
             this merge, so the conformance-compatible lane stays closed until \
             executing-producer attestation lands"
        ),
    }
}

/// Cross-space refusal at the artifact level (pre-drain): the executing
/// embedder's space does not join the attested space of the existing tier.
fn space_identity_refusal(
    tier: &str,
    executing_space_fingerprint: &str,
    attested_space_fingerprint: &str,
) -> SearchError {
    SearchError::InvalidConfig {
        field: format!("refresh.{tier}_space_identity"),
        value: executing_space_fingerprint.to_owned(),
        reason: format!(
            "the executing {tier} embedder produces vectors in a different embedding space \
             than the attested {tier} generation (attested space fingerprint \
             {attested_space_fingerprint}); refusing the identity-bound merge — reindex \
             into a new generation instead"
        ),
    }
}

/// Map a typed [`FsviAdmissionError`] into the refresh error surface, naming
/// the tier. I/O and corruption pass through unchanged.
fn admission_error_to_refresh_error(
    error: FsviAdmissionError,
    tier: &str,
    path: &Path,
) -> SearchError {
    match error {
        FsviAdmissionError::Index(error) => error,
        other => SearchError::InvalidConfig {
            field: format!("refresh.{tier}_v2_admission"),
            value: path.display().to_string(),
            reason: other.to_string(),
        },
    }
}

/// Lowercase hex encoding of raw fingerprint bytes.
fn fingerprint_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut out, byte| {
            let _ = write!(out, "{byte:02x}");
            out
        })
}

/// The parsed v2 identity block of an inspected tier, or a typed error when
/// inspection and metadata disagree (never fabricated).
fn v2_identity_of<'metadata>(
    metadata: &'metadata VectorMetadata,
    tier: &str,
) -> SearchResult<&'metadata FsviV2IdentityMetadata> {
    metadata
        .identity_v2
        .as_ref()
        .ok_or_else(|| SearchError::InvalidConfig {
            field: format!("refresh.{tier}_index_identity"),
            value: "missing-v2-identity-metadata".to_owned(),
            reason: "inspection classified the artifact as FSVI v2 but its parsed metadata \
                     carries no identity block"
                .to_owned(),
        })
}

/// Derive the FSVI v2 artifact identity bundle for an embedder: the same
/// space, producer, and input contracts, with the storage component rewritten
/// to the canonical persisted contract (`fsvi-v2`, F16, little-endian) that
/// [`VectorIndex::create_v2`] requires. The result is re-validated: storage
/// is physical, so the space and producer fingerprints are unchanged.
fn artifact_identity_for(embedder: &dyn Embedder) -> SearchResult<EmbeddingIdentityBundleV1> {
    let mut bundle = embedder.identity()?.clone();
    "fsvi-v2".clone_into(&mut bundle.storage.format);
    bundle.storage.quantization = QuantizationFormat::F16;
    "little-endian".clone_into(&mut bundle.storage.endianness);
    bundle.validate()?;
    Ok(bundle)
}

/// Reconstruct the exact identity binding an existing attested tier was
/// published under, from the executing embedder's identity plus the
/// artifact's own header metadata.
///
/// The bundle canonical bytes in a v2 header are encode-only (no structured
/// decoder exists, by design), so exact admission needs the caller to hold
/// the structured bundle. The executing embedder holds the space/producer/
/// input components (already gated equal to the attested fingerprints); only
/// the physical storage quantization can legitimately vary, so both F16 and
/// F32 candidates are tried against the header's full-bundle fingerprint.
fn reconstruct_admission_binding(
    metadata: &VectorMetadata,
    embedder_identity: &EmbeddingIdentityBundleV1,
    tier: &str,
) -> SearchResult<FsviV2IdentityBinding> {
    let identity = v2_identity_of(metadata, tier)?;
    let attested_bundle_hex = fingerprint_hex(&identity.identity_bundle_fingerprint);
    for quantization in [QuantizationFormat::F16, QuantizationFormat::F32] {
        let mut candidate = embedder_identity.clone();
        "fsvi-v2".clone_into(&mut candidate.storage.format);
        candidate.storage.quantization = quantization;
        "little-endian".clone_into(&mut candidate.storage.endianness);
        if candidate.validate().is_err() {
            continue;
        }
        if candidate.fingerprint() == attested_bundle_hex {
            return FsviV2IdentityBinding::new(identity.generation, candidate.freeze()?);
        }
    }
    Err(SearchError::InvalidConfig {
        field: format!("refresh.{tier}_storage_identity"),
        value: fingerprint_hex(&identity.storage_fingerprint),
        reason: format!(
            "the attested {tier} generation's identity bundle ({attested_bundle_hex}) cannot \
             be reconstructed from the executing embedder's identity under any supported \
             storage contract; identity-bound republication is unavailable for this artifact"
        ),
    })
}

/// Unique non-zero nonce material for one staged generation build attempt.
///
/// The generation nonce is uniqueness material, never a credential (see
/// [`ArtifactGenerationIdentityV1`]): it exists so two build attempts of the
/// same sequence are distinguishable. Uniqueness comes from OS-seeded
/// [`std::collections::hash_map::RandomState`] draws (fresh keys per
/// process from OS randomness, distinct per instantiation), domain-mixed
/// with the deterministic build context.
fn generation_nonce(index_dir: &Path, tier: &str, sequence: u64) -> [u8; 16] {
    use std::hash::BuildHasher as _;

    let mut hasher = Sha256::new();
    hasher.update(index_dir.as_os_str().as_encoded_bytes());
    hasher.update(tier.as_bytes());
    hasher.update(sequence.to_le_bytes());
    for draw in 0_u64..4 {
        let os_seeded = std::collections::hash_map::RandomState::new();
        hasher.update(os_seeded.hash_one(draw).to_le_bytes());
    }
    let digest = hasher.finalize();
    let mut nonce = [0_u8; 16];
    nonce.copy_from_slice(&digest[..16]);
    nonce
}

/// Guard 7 seam check on one bound embedding: apply the complete C1r2
/// admission law against `expected` and admit ONLY `SameProducer`.
/// A `ConformanceCompatibleProducer` outcome is telemetry-logged and refused
/// (see [`attestation_unavailable_refusal`] for the deliberate narrowing and
/// its reopening condition).
fn require_same_producer(
    bound: &BoundQueryEmbedding,
    expected: &EmbeddingIdentityBundleV1,
    tier: &str,
) -> SearchResult<()> {
    match bound.verify_producer_conformance(expected, tier)? {
        SpaceIdentityAdmission::SameProducer => Ok(()),
        SpaceIdentityAdmission::ConformanceCompatibleProducer {
            query_producer_fingerprint,
            expected_producer_fingerprint,
        } => {
            warn!(
                target: "frankensearch.refresh",
                tier,
                %query_producer_fingerprint,
                %expected_producer_fingerprint,
                "conformance-compatible producer refused: executing-producer attestation \
                 unavailable (SameProducer-only slice)"
            );
            Err(attestation_unavailable_refusal(
                tier,
                &query_producer_fingerprint,
                &expected_producer_fingerprint,
            ))
        }
        // `SpaceIdentityAdmission` is #[non_exhaustive]: a future admission
        // basis this seam does not understand must fail CLOSED, never admit.
        other => {
            warn!(
                target: "frankensearch.refresh",
                tier,
                admission = other.code(),
                "unrecognized producer-conformance admission outcome refused (fail closed)"
            );
            Err(SearchError::InvalidConfig {
                field: format!("refresh.{tier}_producer_conformance"),
                value: other.code().to_owned(),
                reason: "unrecognized producer-conformance admission outcome; this \
                         SameProducer-only seam fails closed on admission bases it cannot \
                         verify"
                    .to_owned(),
            })
        }
    }
}

/// Gate one attested v2 tier against the executing embedder's artifact
/// bundle: space join first (typed cross-space refusal), then the
/// SameProducer-only producer law against the header's attested producer
/// fingerprint. Both expected values come from the artifact's OWN header —
/// never from a caller-supplied fingerprint.
fn admit_attested_tier(
    tier: &str,
    metadata: &VectorMetadata,
    executing_bundle: &EmbeddingIdentityBundleV1,
) -> SearchResult<()> {
    let identity = v2_identity_of(metadata, tier)?;
    let attested_space = fingerprint_hex(&identity.space_fingerprint);
    let executing_space = executing_bundle.space.fingerprint();
    if executing_space != attested_space {
        return Err(space_identity_refusal(
            tier,
            &executing_space,
            &attested_space,
        ));
    }
    let attested_producer = fingerprint_hex(&identity.producer_fingerprint);
    let executing_producer = executing_bundle.producer.fingerprint();
    if executing_producer != attested_producer {
        return Err(attestation_unavailable_refusal(
            tier,
            &executing_producer,
            &attested_producer,
        ));
    }
    Ok(())
}

/// One existing canonical tier admitted IN FULL at classification time:
/// binding reconstructed from the artifact's own header, content and docset
/// digests recomputed via [`VectorIndex::open_admitted_v2`], and the sealed
/// owner retained (never peeled).
struct AdmittedCanonicalTier {
    /// The retained sealed admission owner. Its `Arc`'d bytes — not the
    /// canonical pathname — are the authority for every subsequent read.
    owner: ValidatedFsviBytes,
    /// Attested space fingerprint (lowercase hex) read from the artifact's
    /// own validated header.
    attested_space_hex: String,
}

/// Admit one existing attested canonical tier exactly and retain the sealed
/// owner plus the attested space fingerprint (lowercase hex) read from the
/// artifact's own header.
fn admit_existing_tier(
    path: &Path,
    metadata: &VectorMetadata,
    embedder: &dyn Embedder,
    tier: &str,
) -> SearchResult<AdmittedCanonicalTier> {
    let attested_space_hex = fingerprint_hex(&v2_identity_of(metadata, tier)?.space_fingerprint);
    let binding = reconstruct_admission_binding(metadata, embedder.identity()?, tier)?;
    let owner = VectorIndex::open_admitted_v2(path, &binding)
        .map_err(|error| admission_error_to_refresh_error(error, tier, path))?;
    Ok(AdmittedCanonicalTier {
        owner,
        attested_space_hex,
    })
}

/// Next generation sequence for a staged replacement: the attested prior
/// generation's sequence plus one, or 1 for a fresh lineage.
fn next_generation_sequence(prior: Option<&ValidatedFsviBytes>) -> SearchResult<u64> {
    prior.map_or_else(
        || Ok(1),
        |owner| {
            owner
                .identity_v2()
                .generation
                .sequence
                .checked_add(1)
                .ok_or_else(|| SearchError::InvalidConfig {
                    field: "refresh.generation_sequence".to_owned(),
                    value: u64::MAX.to_string(),
                    reason: "generation sequence overflow".to_owned(),
                })
        },
    )
}

/// Classification of the existing canonical generation after all identity
/// gates AND full content admission have been applied
/// ([`RefreshWorker::admit_existing_generation`]).
// One value exists per classification and it lives for one cycle step; the
// variant-size asymmetry (retained sealed owners vs. no data) is the point
// of the r2 repair, not an allocation concern worth a Box indirection.
#[allow(clippy::large_enum_variant)]
enum ExistingGenerationClass {
    /// Nothing is retained on disk (missing artifacts or empty legacy v1
    /// seeds): a bootstrap replacement cannot mix vector spaces.
    BootstrapReplaceable,
    /// At least one tier is an attested FSVI v2 artifact, every gate (space
    /// join, `SameProducer`, quality republication capability) passed, and
    /// every attested tier was FULLY ADMITTED through
    /// [`VectorIndex::open_admitted_v2`] — content and docset digests
    /// recomputed — with the sealed owners RETAINED here. A header-valid but
    /// content-corrupt artifact can never reach this variant (it fails
    /// admission with its own typed corruption error instead).
    AttestedV2 {
        /// Retained admission owner of the canonical fast tier, when it is
        /// an attested v2 artifact.
        fast: Option<AdmittedCanonicalTier>,
        /// Retained admission owner of the canonical quality tier, when it
        /// is an attested v2 artifact.
        quality: Option<AdmittedCanonicalTier>,
    },
}

// ---------------------------------------------------------------------------
// Staged identity-bound generation
// ---------------------------------------------------------------------------

/// A proven, non-canonical identity-bound v2 replacement generation
/// (bd-9xuj T2 C4-write, owner retention per r2).
///
/// Produced by [`RefreshWorker::stage_identity_bound_generation`]: the tiers
/// live under the `v2-staged/` subdirectory of the index dir — never the
/// canonical filenames — and `index` is the staged pair re-admitted through
/// [`TwoTierIndex::open_admitted_v2_with_paths`], so its identity is
/// header-ATTESTED (`index.fast_identity_is_attested()`) and the sealed
/// [`ValidatedFsviBytes`] admission owners are RETAINED inside it
/// ([`Self::fast_admitted_owner`] / [`Self::quality_admitted_owner`]): the
/// `Arc`'d bytes, complete witness, and publication state stay the authority
/// for every read — replacing or renaming the staged files afterwards cannot
/// alter what this generation serves. Canonical installation is refused by
/// [`RefreshWorker::publish_staged_canonical`] until the composite
/// generation-authority primitive lands.
pub struct StagedIdentityBoundGeneration {
    /// The staged generation, opened through exact v2 admission with its
    /// sealed owners retained.
    pub index: TwoTierIndex,
    /// Staged fast-tier artifact path.
    pub fast_path: PathBuf,
    /// Staged quality-tier artifact path, when a quality tier was staged.
    pub quality_path: Option<PathBuf>,
    /// Exact binding the staged fast tier was written and admitted under.
    pub fast_binding: FsviV2IdentityBinding,
    /// Exact binding the staged quality tier was written and admitted under.
    pub quality_binding: Option<FsviV2IdentityBinding>,
}

impl StagedIdentityBoundGeneration {
    /// The retained sealed admission owner of the staged fast tier.
    ///
    /// Always `Some` for a value produced by
    /// [`RefreshWorker::stage_identity_bound_generation`] (the staged pair
    /// is opened exclusively through exact v2 admission); typed as `Option`
    /// only because `index` is a public field.
    #[must_use]
    pub fn fast_admitted_owner(&self) -> Option<&ValidatedFsviBytes> {
        self.index.fast_admitted_owner()
    }

    /// The retained sealed admission owner of the staged quality tier, when
    /// a quality tier was staged.
    #[must_use]
    pub fn quality_admitted_owner(&self) -> Option<&ValidatedFsviBytes> {
        self.index.quality_admitted_owner()
    }
}

impl std::fmt::Debug for StagedIdentityBoundGeneration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StagedIdentityBoundGeneration")
            .field("fast_path", &self.fast_path)
            .field("quality_path", &self.quality_path)
            .field("fast_binding", &self.fast_binding)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Refresh worker
// ---------------------------------------------------------------------------

/// Background worker that drains the embedding queue and rebuilds the index.
///
/// # Architecture
///
/// ```text
/// EmbeddingQueue ──drain──> RefreshWorker ──embed_batch_bound──> RefreshRecord
///                                                         │
///                                                    ┌────┘
///                                                    ▼
///                                              IndexCache.replace()
/// ```
///
/// The worker is the single writer for vector indices. It:
/// 1. Drains pending jobs from the [`EmbeddingQueue`]
/// 2. Batch-embeds via the fast-tier [`Embedder`] (and optionally
///    quality-tier), binding every output to its producing identity
/// 3. Rebuilds the `TwoTierIndex` (bootstrap lane) or refuses typed
/// 4. Atomically replaces the cached index via [`IndexCache::replace`]
///
/// # Identity admission boundary (r2)
///
/// A canonical generation that retains content is admissible for replacement
/// only when its identity is ATTESTED (FSVI v2 header), joins the executing
/// embedders' identity as the same space and the same producer, AND fully
/// admits through [`VectorIndex::open_admitted_v2`] — content and docset
/// digests recomputed, sealed owners retained. Pre-drain classification is
/// strictly READ-ONLY on the canonical artifacts: v1 tiers are observed
/// header-only (never the mutable [`VectorIndex::open`], which deletes stale
/// WAL sidecars and truncates corrupt trailers), and content-retaining v1
/// tiers keep the landed containment refusal (`identityless-fsvi-v1`),
/// applied conservatively (an all-tombstoned v1 slab fails closed). Fully
/// admitted v2 replacements can be staged and proven
/// ([`Self::stage_identity_bound_generation`]) but canonical installation is
/// refused pre-drain until composite generation authority lands (see the
/// module docs).
///
/// # Cancellation
///
/// The worker checks `cx.is_cancel_requested()` at each cycle boundary.
/// When cancelled, it finishes the current batch (no half-written index)
/// before returning.
pub struct RefreshWorker {
    config: RefreshWorkerConfig,
    queue: Arc<EmbeddingQueue>,
    fast_embedder: Arc<dyn Embedder>,
    quality_embedder: Option<Arc<dyn Embedder>>,
    cache: Arc<IndexCache>,
    metrics: Arc<RefreshMetrics>,
}

impl std::fmt::Debug for RefreshWorker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RefreshWorker")
            .field("config", &self.config)
            .field("fast_embedder", &self.fast_embedder.id())
            .finish_non_exhaustive()
    }
}

impl RefreshWorker {
    /// Create a new refresh worker.
    #[must_use]
    pub fn new(
        config: RefreshWorkerConfig,
        queue: Arc<EmbeddingQueue>,
        fast_embedder: Arc<dyn Embedder>,
        cache: Arc<IndexCache>,
    ) -> Self {
        Self {
            config,
            queue,
            fast_embedder,
            quality_embedder: None,
            cache,
            metrics: Arc::new(RefreshMetrics::default()),
        }
    }

    /// Set the quality-tier embedder for two-tier index building.
    #[must_use]
    pub fn with_quality_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.quality_embedder = Some(embedder);
        self
    }

    /// Shared reference to the metrics counters.
    #[must_use]
    pub const fn metrics(&self) -> &Arc<RefreshMetrics> {
        &self.metrics
    }

    /// Run the refresh loop.
    ///
    /// Polls the queue at `poll_interval`, embeds batches, and rebuilds the
    /// index. Returns `Ok(())` when the `Cx` is cancelled.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` on unrecoverable failures (e.g., index directory
    /// inaccessible). Transient embedding failures are logged and retried
    /// via the queue's retry mechanism.
    pub async fn run(&self, cx: &Cx) -> SearchResult<()> {
        info!(
            target: "frankensearch.refresh",
            poll_interval_ms = u64::try_from(self.config.poll_interval.as_millis()).unwrap_or(u64::MAX),
            max_docs = self.config.max_docs_per_cycle,
            index_dir = %self.config.index_dir.display(),
            "refresh worker started"
        );

        loop {
            // Cancel-aware sleep.
            asupersync::time::sleep(asupersync::time::wall_now(), self.config.poll_interval).await;

            if cx.is_cancel_requested() {
                info!(
                    target: "frankensearch.refresh",
                    "refresh worker shutting down (cancel requested)"
                );
                return Ok(());
            }

            // Run one refresh cycle. Transient errors are logged, not propagated.
            match self.run_cycle(cx).await {
                Ok(0) => {
                    // No work to do — continue polling.
                }
                Ok(n) => {
                    debug!(
                        target: "frankensearch.refresh",
                        docs = n,
                        "refresh cycle complete"
                    );
                }
                Err(e) => {
                    error!(
                        target: "frankensearch.refresh",
                        error = %e,
                        "refresh cycle failed"
                    );
                    // Continue polling — next cycle may succeed.
                }
            }

            self.metrics.cycles.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Run a single refresh cycle.
    ///
    /// Returns the number of documents successfully embedded, or an error
    /// if the cycle was refused or the index rebuild itself failed.
    ///
    /// # Errors
    ///
    /// Returns identity-admission and index creation/writing errors. Embedding
    /// failures for individual documents are handled via retry (requeue) and
    /// do not cause the cycle to fail.
    pub async fn run_cycle(&self, cx: &Cx) -> SearchResult<usize> {
        // Avoid opening the index on idle polls, but refuse any inadmissible
        // or currently-unpublishable replacement BEFORE draining work: a
        // permanent refusal (identityless v1, foreign space/producer, missing
        // composite generation authority) must not consume retry budget or
        // eventually drop the queued documents.
        if self.queue.is_empty() {
            return Ok(0);
        }
        self.ensure_canonical_cycle_admissible()?;

        // Drain at most `max_docs_per_cycle` jobs from the queue.
        let mut all_jobs = Vec::new();
        let batch_limit = self.config.max_docs_per_cycle;

        while all_jobs.len() < batch_limit {
            let remaining = batch_limit - all_jobs.len();
            let batch = self.queue.drain_batch_up_to(remaining);
            if batch.is_empty() {
                break;
            }
            all_jobs.extend(batch);
        }

        if all_jobs.is_empty() {
            return Ok(0);
        }

        let total_jobs = all_jobs.len();
        debug!(
            target: "frankensearch.refresh",
            jobs = total_jobs,
            "starting refresh cycle"
        );

        // Embed all documents, identity-bound at the embedder boundary.
        let embedded = self.embed_batch(cx, &all_jobs).await;

        if embedded.is_empty() {
            // All embeddings failed — nothing to index.
            warn!(
                target: "frankensearch.refresh",
                jobs = total_jobs,
                "all embeddings failed in cycle"
            );
            return Ok(0);
        }

        let embedded_count = embedded.len();

        // Rebuild the index.
        let rebuild_start = Instant::now();
        match self.rebuild_index(&embedded) {
            Ok(new_index) => {
                let rebuild_us =
                    u64::try_from(rebuild_start.elapsed().as_micros()).unwrap_or(u64::MAX);
                self.metrics
                    .rebuild_time_us
                    .fetch_add(rebuild_us, Ordering::Relaxed);
                self.metrics.index_rebuilds.fetch_add(1, Ordering::Relaxed);

                // Record all embedded hashes so the queue can skip unchanged docs.
                for record in &embedded {
                    self.queue
                        .record_embedded(&record.doc_id, &record.content_hash);
                }

                // Atomically swap the cached index.
                self.cache.replace(new_index);

                info!(
                    target: "frankensearch.refresh",
                    docs = embedded_count,
                    rebuild_ms = rebuild_us / 1000,
                    "index rebuilt and swapped"
                );

                Ok(embedded_count)
            }
            Err(e) => {
                self.metrics
                    .rebuild_failures
                    .fetch_add(1, Ordering::Relaxed);

                // Requeue all jobs so they aren't lost.
                let mut dropped_requeues = 0usize;
                for job in all_jobs {
                    if !self.requeue_job(job, "index_rebuild_failed") {
                        dropped_requeues = dropped_requeues.saturating_add(1);
                    }
                }
                if dropped_requeues > 0 {
                    self.metrics.docs_failed.fetch_add(
                        u64::try_from(dropped_requeues).unwrap_or(u64::MAX),
                        Ordering::Relaxed,
                    );
                }

                error!(
                    target: "frankensearch.refresh",
                    error = %e,
                    dropped_requeues,
                    "index rebuild failed, attempted to requeue jobs"
                );

                Err(e)
            }
        }
    }

    /// Embed a batch of jobs using the fast (and optionally quality)
    /// embedder, binding every output to its producing identity.
    ///
    /// Failed embeddings are requeued for retry. Returns only the
    /// successfully embedded, identity-bound records.
    async fn embed_batch(&self, cx: &Cx, jobs: &[EmbeddingJob]) -> Vec<RefreshRecord> {
        let embed_start = Instant::now();

        // Collect texts for batch embedding.
        let texts: Vec<&str> = jobs.iter().map(|j| j.canonical_text.as_str()).collect();

        // Fast-tier identity-bound embedding (required). An identity-less
        // embedder fails here typed (`embedder.identity`), which also means
        // it can no longer publish provenance-free vectors through this
        // worker — that is the C4-write bound-carrier contract.
        let fast_bound = match self.fast_embedder.embed_batch_bound(cx, &texts).await {
            Ok(embeddings) => embeddings,
            Err(e) => {
                warn!(
                    target: "frankensearch.refresh",
                    error = %e,
                    batch_size = jobs.len(),
                    "fast-tier bound batch embedding failed, requeueing all"
                );
                let mut dropped_requeues = 0usize;
                for job in jobs {
                    if !self.requeue_job(job.clone(), "fast_batch_embedding_failed") {
                        dropped_requeues = dropped_requeues.saturating_add(1);
                    }
                }
                if dropped_requeues > 0 {
                    warn!(
                        target: "frankensearch.refresh",
                        dropped_requeues,
                        "failed to requeue some jobs after fast-tier batch failure"
                    );
                }
                self.metrics.docs_failed.fetch_add(
                    u64::try_from(jobs.len()).unwrap_or(u64::MAX),
                    Ordering::Relaxed,
                );
                return Vec::new();
            }
        };
        let mut fast_bound: Vec<Option<IdentityBoundEmbedding>> =
            fast_bound.into_iter().map(Some).collect();

        // Quality-tier identity-bound embedding (optional).
        let mut quality_bound: Option<Vec<Option<IdentityBoundEmbedding>>> =
            if let Some(ref quality) = self.quality_embedder {
                match quality.embed_batch_bound(cx, &texts).await {
                    Ok(embeddings) => Some(embeddings.into_iter().map(Some).collect()),
                    Err(e) => {
                        warn!(
                            target: "frankensearch.refresh",
                            error = %e,
                            "quality-tier bound batch embedding failed, proceeding with fast only"
                        );
                        None
                    }
                }
            } else {
                None
            };

        let embed_us = u64::try_from(embed_start.elapsed().as_micros()).unwrap_or(u64::MAX);
        self.metrics
            .embed_time_us
            .fetch_add(embed_us, Ordering::Relaxed);

        // Assemble records.
        let mut records = Vec::with_capacity(jobs.len());
        for (i, job) in jobs.iter().enumerate() {
            let Some(fast_ib) = fast_bound.get_mut(i).and_then(Option::take) else {
                // Embedder returned fewer vectors than inputs — skip this doc
                // and requeue so it can be retried next cycle.
                warn!(
                    target: "frankensearch.refresh",
                    doc_id = %job.doc_id,
                    expected = jobs.len(),
                    "fast embedder returned fewer bound vectors than inputs, requeueing"
                );
                if !self.requeue_job(job.clone(), "fast_batch_missing_vector") {
                    warn!(
                        target: "frankensearch.refresh",
                        doc_id = %job.doc_id,
                        "failed to requeue job after fast embedder returned fewer vectors than expected"
                    );
                }
                self.metrics.docs_failed.fetch_add(1, Ordering::Relaxed);
                continue;
            };
            let fast_embedding = match into_bound_query(fast_ib) {
                Ok(bound) => bound,
                Err(e) => {
                    warn!(
                        target: "frankensearch.refresh",
                        doc_id = %job.doc_id,
                        error = %e,
                        "fast-tier bound embedding failed bind-time validation, requeueing"
                    );
                    if !self.requeue_job(job.clone(), "fast_bound_validation_failed") {
                        warn!(
                            target: "frankensearch.refresh",
                            doc_id = %job.doc_id,
                            "failed to requeue job after bind-time validation failure"
                        );
                    }
                    self.metrics.docs_failed.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
            };
            let quality_embedding = quality_bound
                .as_mut()
                .and_then(|bound| bound.get_mut(i))
                .and_then(Option::take)
                .and_then(|ib| match into_bound_query(ib) {
                    Ok(bound) => Some(bound),
                    Err(e) => {
                        warn!(
                            target: "frankensearch.refresh",
                            doc_id = %job.doc_id,
                            error = %e,
                            "quality-tier bound embedding failed bind-time validation, \
                             proceeding fast-only for this document"
                        );
                        None
                    }
                });

            records.push(RefreshRecord {
                doc_id: job.doc_id.clone(),
                fast_embedding,
                quality_embedding,
                content_hash: job.content_hash.clone(),
            });
        }

        self.metrics.docs_embedded.fetch_add(
            u64::try_from(records.len()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        self.queue
            .metrics()
            .total_embed_time_us
            .fetch_add(embed_us, Ordering::Relaxed);

        records
    }

    fn requeue_job(&self, job: EmbeddingJob, reason: &'static str) -> bool {
        let doc_id = job.doc_id.clone();
        match self.queue.requeue(job) {
            JobOutcome::Retryable => true,
            JobOutcome::Failed => {
                warn!(
                    target: "frankensearch.refresh",
                    doc_id = %doc_id,
                    reason,
                    "failed to requeue job"
                );
                false
            }
            outcome => {
                warn!(
                    target: "frankensearch.refresh",
                    doc_id = %doc_id,
                    reason,
                    ?outcome,
                    "unexpected requeue outcome"
                );
                false
            }
        }
    }

    /// Resolve the existing canonical fast-tier artifact path, if any.
    fn resolve_existing_fast_path(&self) -> Option<PathBuf> {
        let fast_path = self.config.index_dir.join(VECTOR_INDEX_FAST_FILENAME);
        if fast_path.exists() {
            return Some(fast_path);
        }
        let fallback_path = self.config.index_dir.join(VECTOR_INDEX_FALLBACK_FILENAME);
        if fallback_path.exists() {
            return Some(fallback_path);
        }
        None
    }

    /// Inspect both canonical tiers.
    fn canonical_tier_states(&self) -> SearchResult<(TierState, TierState)> {
        let fast_state = match self.resolve_existing_fast_path() {
            Some(path) => inspect_tier(&path)?,
            None => TierState::Missing,
        };
        let quality_path = self.config.index_dir.join(VECTOR_INDEX_QUALITY_FILENAME);
        let quality_state = inspect_tier(&quality_path)?;
        Ok((fast_state, quality_state))
    }

    /// Apply the full identity admission law to the existing canonical
    /// generation (guards 2, 7, 8) and PERFORM the admission it claims
    /// (r2 repair of NO-GO item 1: the r1 revision stopped at header
    /// inspection, so a header-valid/content-corrupt v2 artifact sailed past
    /// to the wrong refusal).
    ///
    /// - content-retaining legacy v1 tiers keep the landed containment
    ///   refusal, the fast tier named first (matching origin 5386b39e);
    ///   liveness is the conservative READ-ONLY observation (see
    ///   [`TierState::LegacyV1`]) — classification never mutably opens a v1
    ///   tier;
    /// - attested v2 tiers are gated per tier: space join against the
    ///   artifact's OWN header fingerprints, then SameProducer-only producer
    ///   conformance (cheap, typed refusals first); an attested quality tier
    ///   additionally requires a quality embedder capable of republishing
    ///   it; then each attested tier is FULLY ADMITTED via
    ///   [`VectorIndex::open_admitted_v2`] — exact binding reconstruction,
    ///   content/docset digest recomputation — and the sealed owner is
    ///   RETAINED in the returned classification;
    /// - only a content-free generation (missing artifacts or empty v1
    ///   seeds) classifies as bootstrap-replaceable.
    fn admit_existing_generation(
        &self,
        fast_state: &TierState,
        quality_state: &TierState,
    ) -> SearchResult<ExistingGenerationClass> {
        let fast_v1_retains = matches!(
            fast_state,
            TierState::LegacyV1 {
                retains_content: true
            }
        );
        let quality_v1_retains = matches!(
            quality_state,
            TierState::LegacyV1 {
                retains_content: true
            }
        );
        if fast_v1_retains || quality_v1_retains {
            // Name the fast tier whenever it is identityless, else quality —
            // the same attribution the landed containment refusal used.
            let tier = if matches!(fast_state, TierState::LegacyV1 { .. }) {
                "fast"
            } else {
                "quality"
            };
            return Err(identityless_refusal(tier));
        }

        let fast_is_v2 = matches!(fast_state, TierState::V2 { .. });
        let quality_is_v2 = matches!(quality_state, TierState::V2 { .. });
        if !fast_is_v2 && !quality_is_v2 {
            return Ok(ExistingGenerationClass::BootstrapReplaceable);
        }

        let fast = if let TierState::V2 { metadata } = fast_state {
            let bundle = artifact_identity_for(self.fast_embedder.as_ref())?;
            admit_attested_tier("fast", metadata, &bundle)?;
            let path =
                self.resolve_existing_fast_path()
                    .ok_or_else(|| SearchError::IndexNotFound {
                        path: self.config.index_dir.join(VECTOR_INDEX_FAST_FILENAME),
                    })?;
            Some(admit_existing_tier(
                &path,
                metadata,
                self.fast_embedder.as_ref(),
                "fast",
            )?)
        } else {
            None
        };
        let quality = if let TierState::V2 { metadata } = quality_state {
            let Some(quality_embedder) = &self.quality_embedder else {
                return Err(quality_republication_unavailable());
            };
            let bundle = artifact_identity_for(quality_embedder.as_ref())?;
            admit_attested_tier("quality", metadata, &bundle)?;
            let path = self.config.index_dir.join(VECTOR_INDEX_QUALITY_FILENAME);
            Some(admit_existing_tier(
                &path,
                metadata,
                quality_embedder.as_ref(),
                "quality",
            )?)
        } else {
            None
        };
        Ok(ExistingGenerationClass::AttestedV2 { fast, quality })
    }

    /// Pre-drain admission for the canonical cycle: only the bootstrap lane
    /// may proceed; a FULLY ADMITTED attested v2 generation (content digests
    /// recomputed, owners retained through the check) refuses with the
    /// composite generation-authority reason (see the module docs and
    /// [`composite_authority_refusal`]). The retained owners are dropped
    /// here because the refusal aborts the cycle before any consumer exists;
    /// [`Self::stage_identity_bound_generation`] is the path that carries
    /// them forward.
    fn ensure_canonical_cycle_admissible(&self) -> SearchResult<()> {
        let (fast_state, quality_state) = self.canonical_tier_states()?;
        match self.admit_existing_generation(&fast_state, &quality_state)? {
            ExistingGenerationClass::BootstrapReplaceable => Ok(()),
            ExistingGenerationClass::AttestedV2 { .. } => {
                Err(composite_authority_refusal(&self.config.index_dir))
            }
        }
    }

    /// Rebuild the `TwoTierIndex` from identity-bound records — the
    /// bootstrap lane (content-free canonical generation only).
    ///
    /// The producing identities are declared on the builder
    /// (`set_*_identity`, process-local, DECLARED — the persisted artifacts
    /// stay v1 and never read as attested), and every bound record is
    /// re-verified against the executing embedder's identity under the
    /// SameProducer-only law before its vector is written.
    fn rebuild_index(&self, records: &[RefreshRecord]) -> SearchResult<TwoTierIndex> {
        // Repeat the admission check after embedding to close the race where
        // another publisher installs a generation between the pre-drain check
        // and this rebuild. Only this race fallback consumes one retry.
        self.ensure_canonical_cycle_admissible()?;

        let mut builder =
            TwoTierIndex::create(&self.config.index_dir, self.config.index_config.clone())?;

        builder.set_fast_embedder_id(self.fast_embedder.id());
        if let Some(ref quality) = self.quality_embedder {
            builder.set_quality_embedder_id(quality.id());
        }

        // Declare the producing identities. Records exist only when bound
        // harvesting succeeded, so the fast identity is available; a quality
        // identity is required exactly when quality-bound records exist.
        let fast_expected = self.fast_embedder.identity()?.clone();
        builder.set_fast_identity(&fast_expected)?;
        let need_quality_identity = records
            .iter()
            .any(|record| record.quality_embedding.is_some());
        let quality_expected = if need_quality_identity {
            let Some(quality_embedder) = &self.quality_embedder else {
                return Err(SearchError::InvalidConfig {
                    field: "refresh.quality_identity".to_owned(),
                    value: "missing-quality-embedder".to_owned(),
                    reason: "quality-bound records exist without a quality embedder".to_owned(),
                });
            };
            let expected = quality_embedder.identity()?.clone();
            builder.set_quality_identity(&expected)?;
            Some(expected)
        } else {
            None
        };

        // Keep only the latest update per doc_id from this cycle.
        let mut latest_by_doc_id = HashMap::new();
        let mut consumed = vec![false; records.len()];
        for (idx, record) in records.iter().enumerate() {
            if let Some(previous) = latest_by_doc_id.insert(record.doc_id.as_str(), idx) {
                consumed[previous] = true;
            }
        }

        for (idx, record) in records.iter().enumerate() {
            if consumed[idx] {
                continue;
            }
            // Guard 7 at the write seam: every vector written must be bound
            // to the identity being declared for this generation. This is
            // the defense against an embedder whose bound outputs disagree
            // with its `identity()` (SameProducer only; certified-compatible
            // foreign producers are refused, see [`require_same_producer`]).
            require_same_producer(&record.fast_embedding, &fast_expected, "fast")?;
            if let Some(quality_embedding) = &record.quality_embedding {
                let Some(expected) = &quality_expected else {
                    return Err(SearchError::InvalidConfig {
                        field: "refresh.quality_identity".to_owned(),
                        value: "missing-quality-identity".to_owned(),
                        reason: "quality-bound record exists without a declared quality identity"
                            .to_owned(),
                    });
                };
                require_same_producer(quality_embedding, expected, "quality")?;
            }
            builder.add_record(
                &record.doc_id,
                record.fast_embedding.vector(),
                record
                    .quality_embedding
                    .as_ref()
                    .map(BoundQueryEmbedding::vector),
            )?;
        }

        builder.finish()
    }

    // -----------------------------------------------------------------------
    // Staged identity-bound replacement (the C4-write merge)
    // -----------------------------------------------------------------------

    /// Embed `jobs` identity-bound with NO queue interaction: staging must
    /// not consume retry budget or record content hashes, because the
    /// canonical generation is not being published. Any failure propagates.
    async fn embed_jobs_bound_strict(
        &self,
        cx: &Cx,
        jobs: &[EmbeddingJob],
    ) -> SearchResult<Vec<RefreshRecord>> {
        let embed_start = Instant::now();
        let texts: Vec<&str> = jobs.iter().map(|j| j.canonical_text.as_str()).collect();
        let fast_bound = self.fast_embedder.embed_batch_bound(cx, &texts).await?;
        if fast_bound.len() != jobs.len() {
            return Err(SearchError::InvalidConfig {
                field: "refresh.staged_embedding".to_owned(),
                value: fast_bound.len().to_string(),
                reason: format!(
                    "fast embedder returned {} bound vectors for {} inputs",
                    fast_bound.len(),
                    jobs.len()
                ),
            });
        }
        let quality_bound = match &self.quality_embedder {
            Some(quality) => {
                let bound = quality.embed_batch_bound(cx, &texts).await?;
                if bound.len() != jobs.len() {
                    return Err(SearchError::InvalidConfig {
                        field: "refresh.staged_embedding".to_owned(),
                        value: bound.len().to_string(),
                        reason: format!(
                            "quality embedder returned {} bound vectors for {} inputs",
                            bound.len(),
                            jobs.len()
                        ),
                    });
                }
                Some(bound)
            }
            None => None,
        };
        let embed_us = u64::try_from(embed_start.elapsed().as_micros()).unwrap_or(u64::MAX);
        self.metrics
            .embed_time_us
            .fetch_add(embed_us, Ordering::Relaxed);

        let mut quality_iter = quality_bound.map(Vec::into_iter);
        let mut records = Vec::with_capacity(jobs.len());
        for (job, fast_ib) in jobs.iter().zip(fast_bound) {
            let quality_embedding = match quality_iter.as_mut().and_then(Iterator::next) {
                Some(ib) => Some(into_bound_query(ib)?),
                None => None,
            };
            records.push(RefreshRecord {
                doc_id: job.doc_id.clone(),
                fast_embedding: into_bound_query(fast_ib)?,
                quality_embedding,
                content_hash: job.content_hash.clone(),
            });
        }
        self.metrics.docs_embedded.fetch_add(
            u64::try_from(records.len()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        Ok(records)
    }

    /// Stage the typed identity-bound replacement generation — the merge
    /// that replaces the former blanket refusal (bd-9xuj T2 C4-write).
    ///
    /// Reads the existing canonical generation through exact FSVI v2
    /// admission (attested identity only; live v1 keeps the typed
    /// `identityless-fsvi-v1` refusal), joins every bound embedding's space
    /// fingerprint against the tier's attested identity
    /// ([`BoundQueryEmbedding::verify_space_identity`]) and applies the
    /// SameProducer-only producer law
    /// ([`BoundQueryEmbedding::verify_producer_conformance`] via
    /// [`require_same_producer`]), merges the carried live rows with the new
    /// records (new wins per `doc_id`; tombstoned rows are never
    /// resurrected), and writes the replacement via
    /// [`VectorIndex::create_v2`] into the non-canonical `v2-staged/`
    /// directory. The staged pair is then re-admitted through
    /// [`TwoTierIndex::open_admitted_v2_with_paths`], so the returned
    /// generation's identity is proven from its own header bytes.
    ///
    /// The queue is untouched: `jobs` are caller-supplied, nothing is
    /// drained, no content hash is recorded, and the canonical generation's
    /// bytes are not modified. Canonical installation is a separate,
    /// currently-refused step ([`Self::publish_staged_canonical`]).
    ///
    /// # Errors
    ///
    /// Typed identity refusals (space, producer, legacy, republication),
    /// admission failures, and I/O errors from staging.
    #[allow(clippy::too_many_lines)]
    pub async fn stage_identity_bound_generation(
        &self,
        cx: &Cx,
        jobs: &[EmbeddingJob],
    ) -> SearchResult<StagedIdentityBoundGeneration> {
        // 1+3 (merged in r2). Gates over the existing canonical generation
        //    (typed refusals, fast-first, identical to the canonical lane)
        //    AND exact admission of the existing attested tiers in the same
        //    step: the reconstructed binding re-proves space/producer/input
        //    equality bit-for-bit against the artifact's own header, the
        //    content/docset digests are recomputed, and the sealed owners
        //    are RETAINED — the same owners the pre-drain check proves, not
        //    a second open.
        let (fast_state, quality_state) = self.canonical_tier_states()?;
        let (fast_admitted, quality_admitted) =
            match self.admit_existing_generation(&fast_state, &quality_state)? {
                ExistingGenerationClass::BootstrapReplaceable => (None, None),
                ExistingGenerationClass::AttestedV2 { fast, quality } => (fast, quality),
            };

        // 2. Harvest identity-bound records (strict; no queue interaction).
        let records = self.embed_jobs_bound_strict(cx, jobs).await?;

        // 4. Per-embedding seam verification (C1r2 verifiers): each bound
        //    embedding must be the same producer as the identity being
        //    republished, and must join the attested space of any tier it is
        //    merged into. (Producer equality against the ATTESTED producer
        //    follows transitively: record == executing identity ==
        //    reconstructed binding == header, all checked above.)
        let fast_expected = self.fast_embedder.identity()?.clone();
        let quality_expected = match &self.quality_embedder {
            Some(quality)
                if records
                    .iter()
                    .any(|record| record.quality_embedding.is_some()) =>
            {
                Some(quality.identity()?.clone())
            }
            _ => None,
        };
        for record in &records {
            require_same_producer(&record.fast_embedding, &fast_expected, "fast")?;
            if let Some(AdmittedCanonicalTier {
                attested_space_hex, ..
            }) = &fast_admitted
            {
                record
                    .fast_embedding
                    .verify_space_identity(attested_space_hex, "fast")?;
            }
            if let Some(quality_embedding) = &record.quality_embedding {
                let Some(expected) = &quality_expected else {
                    return Err(SearchError::InvalidConfig {
                        field: "refresh.quality_identity".to_owned(),
                        value: "missing-quality-identity".to_owned(),
                        reason: "quality-bound record exists without a quality embedder identity"
                            .to_owned(),
                    });
                };
                require_same_producer(quality_embedding, expected, "quality")?;
                if let Some(AdmittedCanonicalTier {
                    attested_space_hex, ..
                }) = &quality_admitted
                {
                    quality_embedding.verify_space_identity(attested_space_hex, "quality")?;
                }
            }
        }

        // 5. Merge: carried live rows first (tombstones stay dead), then the
        //    new records override per doc_id (last write in the batch wins).
        let mut merged: HashMap<String, (Vec<f32>, Option<Vec<f32>>)> = HashMap::new();
        if let Some(AdmittedCanonicalTier {
            owner: fast_owner, ..
        }) = &fast_admitted
        {
            let mut quality_lookup: HashMap<String, Vec<f32>> = HashMap::new();
            if let Some(AdmittedCanonicalTier {
                owner: quality_owner,
                ..
            }) = &quality_admitted
            {
                for i in 0..quality_owner.record_count() {
                    let row = quality_owner.row(i)?;
                    if !row.flags().is_live() {
                        continue;
                    }
                    let doc_id = row.doc_id().to_owned();
                    if let std::collections::hash_map::Entry::Vacant(entry) =
                        quality_lookup.entry(doc_id)
                    {
                        entry.insert(quality_owner.vector_at_f32(i)?);
                    }
                }
            }
            for i in 0..fast_owner.record_count() {
                let row = fast_owner.row(i)?;
                if !row.flags().is_live() {
                    continue;
                }
                let doc_id = row.doc_id().to_owned();
                let vector = fast_owner.vector_at_f32(i)?;
                let quality = quality_lookup.get(&doc_id).cloned();
                merged.insert(doc_id, (vector, quality));
            }
        }
        for record in &records {
            merged.insert(
                record.doc_id.clone(),
                (
                    record.fast_embedding.vector().to_vec(),
                    record
                        .quality_embedding
                        .as_ref()
                        .map(|bound| bound.vector().to_vec()),
                ),
            );
        }
        if merged.is_empty() {
            return Err(SearchError::InvalidConfig {
                field: "refresh.staged_generation".to_owned(),
                value: "empty".to_owned(),
                reason: "no rows to stage: the job batch produced no records and the canonical \
                         generation carries no live rows"
                    .to_owned(),
            });
        }

        // 6. Write the staged replacement via the production v2 writer.
        let staged_dir = self.config.index_dir.join(STAGED_V2_DIR_NAME);
        std::fs::create_dir_all(&staged_dir).map_err(SearchError::Io)?;
        let staged_fast = staged_dir.join(VECTOR_INDEX_FAST_FILENAME);
        let staged_quality = staged_dir.join(VECTOR_INDEX_QUALITY_FILENAME);
        let _ = std::fs::remove_file(&staged_fast);
        let _ = std::fs::remove_file(&staged_quality);

        let fast_sequence =
            next_generation_sequence(fast_admitted.as_ref().map(|tier| &tier.owner))?;
        let fast_artifact_bundle = artifact_identity_for(self.fast_embedder.as_ref())?;
        let fast_generation = ArtifactGenerationIdentityV1::new(
            fast_sequence,
            generation_nonce(&self.config.index_dir, "fast", fast_sequence),
        )?;
        let fast_binding =
            FsviV2IdentityBinding::new(fast_generation, fast_artifact_bundle.freeze()?)?;
        let mut fast_writer = VectorIndex::create_v2(&staged_fast, fast_binding.clone())?;
        for (doc_id, (vector, _)) in &merged {
            fast_writer.write_record(doc_id, vector)?;
        }
        fast_writer.finish()?;

        let quality_rows: Vec<(&String, &Vec<f32>)> = merged
            .iter()
            .filter_map(|(doc_id, (_, quality))| quality.as_ref().map(|q| (doc_id, q)))
            .collect();
        let quality_binding = if quality_rows.is_empty() {
            None
        } else {
            let Some(quality_embedder) = &self.quality_embedder else {
                return Err(SearchError::InvalidConfig {
                    field: "refresh.quality_identity".to_owned(),
                    value: "missing-quality-embedder".to_owned(),
                    reason: "quality rows were merged without a quality embedder".to_owned(),
                });
            };
            let quality_sequence =
                next_generation_sequence(quality_admitted.as_ref().map(|tier| &tier.owner))?;
            let quality_bundle = artifact_identity_for(quality_embedder.as_ref())?;
            let quality_generation = ArtifactGenerationIdentityV1::new(
                quality_sequence,
                generation_nonce(&self.config.index_dir, "quality", quality_sequence),
            )?;
            let binding = FsviV2IdentityBinding::new(quality_generation, quality_bundle.freeze()?)?;
            let mut quality_writer = VectorIndex::create_v2(&staged_quality, binding.clone())?;
            for (doc_id, vector) in quality_rows {
                quality_writer.write_record(doc_id, vector)?;
            }
            quality_writer.finish()?;
            Some(binding)
        };

        // 7. Prove the staged generation by re-admitting it through the only
        //    legitimate v2 open path; the returned index's identity is
        //    header-attested by construction.
        let mut staged_paths = TwoTierIndexPaths::new(&staged_fast);
        if quality_binding.is_some() {
            staged_paths = staged_paths.with_quality_index(&staged_quality);
        }
        let index = TwoTierIndex::open_admitted_v2_with_paths(
            &staged_paths,
            self.config.index_config.clone(),
            &fast_binding,
            quality_binding.as_ref(),
        )?;
        debug_assert!(index.fast_identity_is_attested());

        info!(
            target: "frankensearch.refresh",
            staged_fast = %staged_fast.display(),
            staged_quality = quality_binding.is_some(),
            rows = merged.len(),
            generation_sequence = fast_sequence,
            "identity-bound replacement generation staged and admitted (non-canonical)"
        );

        Ok(StagedIdentityBoundGeneration {
            index,
            fast_path: staged_fast,
            quality_path: quality_binding.as_ref().map(|_| staged_quality),
            fast_binding,
            quality_binding,
        })
    }

    /// Canonical installation of a staged generation — REFUSED in this slice.
    ///
    /// Installing the staged fast/quality pair over the canonical filenames
    /// would be a split two-rename publication with no atomic pair
    /// authority; a crash between the renames leaves mixed generations
    /// serving. This slice deliberately does not invent a pair-atomicity
    /// primitive. The refusal is typed (`refresh.canonical_publication` /
    /// `composite-generation-authority-unavailable`) and reopens when the
    /// composite generation-authority primitive lands (bd-xomn.1/.3).
    ///
    /// # Errors
    ///
    /// Always returns the typed composite-authority refusal in this slice.
    pub fn publish_staged_canonical(
        &self,
        staged: &StagedIdentityBoundGeneration,
    ) -> SearchResult<()> {
        warn!(
            target: "frankensearch.refresh",
            staged_fast = %staged.fast_path.display(),
            index_dir = %self.config.index_dir.display(),
            "canonical publication of a staged split generation refused: composite \
             generation authority unavailable (bd-xomn.1/.3)"
        );
        Err(composite_authority_refusal(&self.config.index_dir))
    }

    /// Reference to the index directory.
    #[must_use]
    pub fn index_dir(&self) -> &Path {
        &self.config.index_dir
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use std::sync::atomic::Ordering;
    use std::time::Instant;

    use frankensearch_core::canonicalize::DefaultCanonicalizer;
    use frankensearch_core::config::TwoTierConfig;
    use frankensearch_core::error::SearchError;
    use frankensearch_core::traits::{ModelCategory, SearchFuture};
    use frankensearch_index::{VECTOR_INDEX_FAST_FILENAME, VectorIndex};

    use super::*;
    use crate::cache::SentinelFileDetector;
    use crate::queue::{EmbeddingQueueConfig, EmbeddingRequest, JobOutcome};

    // -- Stub embedders for tests ----------------------------------------------

    struct StubEmbedder {
        id: &'static str,
        dimension: usize,
        space_offset: f32,
        identity: EmbeddingIdentityBundleV1,
    }

    impl StubEmbedder {
        fn new(id: &'static str, dimension: usize) -> Self {
            Self::in_space(id, dimension, 0.0)
        }

        /// Same display id, different `space_offset` = a genuinely different
        /// vector space AND a different typed space identity: the identity
        /// model key embeds the offset, mirroring how a real model change
        /// alters the immutable space identity even when the display id
        /// stays the same.
        fn in_space(id: &'static str, dimension: usize, space_offset: f32) -> Self {
            let identity = EmbeddingIdentityBundleV1::explicit_test_model(
                &format!("{id}#space-{space_offset}"),
                u32::try_from(dimension).expect("test dimension fits u32"),
            );
            Self {
                id,
                dimension,
                space_offset,
                identity,
            }
        }

        /// Stub with an explicit identity bundle (foreign-producer fixtures).
        fn with_identity(
            id: &'static str,
            dimension: usize,
            identity: EmbeddingIdentityBundleV1,
        ) -> Self {
            Self {
                id,
                dimension,
                space_offset: 0.0,
                identity,
            }
        }

        fn identity_bundle(&self) -> &EmbeddingIdentityBundleV1 {
            &self.identity
        }
    }

    impl Embedder for StubEmbedder {
        fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            let dim = self.dimension;
            let seed = text.len() as f32 + self.space_offset;
            Box::pin(async move { Ok((0..dim).map(|i| (seed + i as f32).sin()).collect()) })
        }

        fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
            Ok(&self.identity)
        }

        fn id(&self) -> &str {
            self.id
        }

        fn model_name(&self) -> &str {
            self.id
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn is_ready(&self) -> bool {
            true
        }

        fn is_semantic(&self) -> bool {
            false
        }

        fn category(&self) -> ModelCategory {
            ModelCategory::HashEmbedder
        }
    }

    struct FailingEmbedder;

    impl Embedder for FailingEmbedder {
        fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            Box::pin(async {
                Err(SearchError::EmbeddingFailed {
                    model: "failing-embedder".into(),
                    source: Box::new(std::io::Error::other("intentional failure")),
                })
            })
        }

        fn id(&self) -> &'static str {
            "failing-embedder"
        }

        fn model_name(&self) -> &'static str {
            "failing-embedder"
        }

        fn dimension(&self) -> usize {
            256
        }

        fn is_ready(&self) -> bool {
            true
        }

        fn is_semantic(&self) -> bool {
            false
        }

        fn category(&self) -> ModelCategory {
            ModelCategory::HashEmbedder
        }
    }

    /// An embedder whose bound outputs carry a DIFFERENT (but
    /// conformance-certified sibling) producer than its own `identity()` —
    /// the hostile shape the per-record seam check exists for.
    struct TwoFacedEmbedder {
        public_identity: EmbeddingIdentityBundleV1,
        bound_identity: EmbeddingIdentityBundleV1,
        dimension: usize,
    }

    impl Embedder for TwoFacedEmbedder {
        fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            let dim = self.dimension;
            let seed = text.len() as f32;
            Box::pin(async move { Ok((0..dim).map(|i| (seed + i as f32).sin()).collect()) })
        }

        fn embed_batch_bound<'a>(
            &'a self,
            cx: &'a Cx,
            texts: &'a [&'a str],
        ) -> SearchFuture<'a, Vec<IdentityBoundEmbedding>> {
            Box::pin(async move {
                let vectors = self.embed_batch(cx, texts).await?;
                vectors
                    .into_iter()
                    .map(|values| {
                        let bound = IdentityBoundEmbedding {
                            values,
                            identity: self.bound_identity.clone(),
                        };
                        bound.validate()?;
                        Ok(bound)
                    })
                    .collect()
            })
        }

        fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
            Ok(&self.public_identity)
        }

        fn id(&self) -> &'static str {
            "two-faced-embedder"
        }

        fn model_name(&self) -> &'static str {
            "two-faced-embedder"
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn is_semantic(&self) -> bool {
            false
        }

        fn category(&self) -> ModelCategory {
            ModelCategory::HashEmbedder
        }
    }

    // -- Test helpers ----------------------------------------------------------

    fn make_queue(capacity: usize) -> Arc<EmbeddingQueue> {
        Arc::new(EmbeddingQueue::new(
            EmbeddingQueueConfig {
                capacity,
                batch_size: 100,
                max_retries: 3,
            },
            Box::new(DefaultCanonicalizer::default()),
        ))
    }

    fn submit(queue: &EmbeddingQueue, doc_id: &str, text: &str) {
        queue
            .submit(EmbeddingRequest {
                doc_id: doc_id.into(),
                text: text.to_owned(),
                metadata: None,
                submitted_at: Instant::now(),
            })
            .unwrap();
    }

    /// Create a temporary directory with a unique name.
    fn temp_index_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "frankensearch-refresh-test-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Seed an initial index on disk (required for `IndexCache::open`).
    fn seed_index(dir: &Path, dimension: usize) -> TwoTierIndex {
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        let writer = VectorIndex::create(&fast_path, "stub-fast", dimension).unwrap();
        writer.finish().unwrap();
        TwoTierIndex::open(dir, TwoTierConfig::default()).unwrap()
    }

    fn make_cache(dir: &Path, dimension: usize) -> Arc<IndexCache> {
        seed_index(dir, dimension);
        let detector = Box::new(SentinelFileDetector::new());
        Arc::new(IndexCache::open(dir, TwoTierConfig::default(), detector).unwrap())
    }

    /// Seed the cache from an empty v1 FALLBACK artifact (`vector.idx`), so
    /// a v2 canonical fast artifact (`vector.fast.idx`) can be installed
    /// afterwards without breaking the (v1-only) cache open.
    fn make_cache_with_fallback_seed(dir: &Path, dimension: usize) -> Arc<IndexCache> {
        let seed_path = dir.join(VECTOR_INDEX_FALLBACK_FILENAME);
        VectorIndex::create(&seed_path, "stub-fast", dimension)
            .unwrap()
            .finish()
            .unwrap();
        let detector = Box::new(SentinelFileDetector::new());
        Arc::new(IndexCache::open(dir, TwoTierConfig::default(), detector).unwrap())
    }

    fn make_worker(
        queue: Arc<EmbeddingQueue>,
        dir: &Path,
        dimension: usize,
    ) -> (RefreshWorker, Arc<IndexCache>) {
        let cache = make_cache(dir, dimension);
        let config = RefreshWorkerConfig::new(dir).with_poll_interval(Duration::from_millis(10));
        let fast = Arc::new(StubEmbedder::new("stub-fast", dimension));
        let worker = RefreshWorker::new(config, queue, fast, cache.clone());
        (worker, cache)
    }

    fn normalized(dim: usize, seed: f32) -> Vec<f32> {
        let raw: Vec<f32> = (0..dim).map(|i| seed + 1.0 + i as f32).collect();
        let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        raw.iter().map(|x| x / norm).collect()
    }

    /// The fsvi-v2 artifact-storage variant of an in-memory identity bundle
    /// (what `artifact_identity_for` derives for an embedder).
    fn artifact_variant(bundle: &EmbeddingIdentityBundleV1) -> EmbeddingIdentityBundleV1 {
        let mut artifact = bundle.clone();
        "fsvi-v2".clone_into(&mut artifact.storage.format);
        artifact.storage.quantization = QuantizationFormat::F16;
        "little-endian".clone_into(&mut artifact.storage.endianness);
        artifact
    }

    fn v2_binding(
        bundle_in_memory: &EmbeddingIdentityBundleV1,
        sequence: u64,
    ) -> FsviV2IdentityBinding {
        let artifact = artifact_variant(bundle_in_memory);
        let generation =
            ArtifactGenerationIdentityV1::new(sequence, [0x3b; 16]).expect("test generation");
        FsviV2IdentityBinding::new(generation, artifact.freeze().expect("freeze artifact"))
            .expect("valid binding")
    }

    fn write_v2_tier(path: &Path, binding: &FsviV2IdentityBinding, rows: &[(&str, Vec<f32>)]) {
        let mut writer = VectorIndex::create_v2(path, binding.clone()).expect("create_v2 fixture");
        for (doc_id, vector) in rows {
            writer.write_record(doc_id, vector).expect("write v2 row");
        }
        writer.finish().expect("finish v2 fixture");
    }

    fn admitted_vectors_by_doc(owner: &ValidatedFsviBytes) -> HashMap<String, Vec<f32>> {
        let mut out = HashMap::new();
        for i in 0..owner.record_count() {
            let row = owner.row(i).expect("row");
            if !row.flags().is_live() {
                continue;
            }
            out.insert(
                row.doc_id().to_owned(),
                owner.vector_at_f32(i).expect("vector"),
            );
        }
        out
    }

    /// Assert a typed `InvalidConfig` refusal with the exact field and value.
    #[track_caller]
    fn assert_invalid_config(error: &SearchError, expected_field: &str, expected_value: &str) {
        assert!(
            matches!(
                error,
                SearchError::InvalidConfig { field, value, .. }
                    if field == expected_field && value == expected_value
            ),
            "expected InvalidConfig {{ field: {expected_field:?}, value: {expected_value:?} }}, \
             got {error:?}"
        );
    }

    /// Reason string of a typed `InvalidConfig` refusal (empty for other
    /// variants; call after [`assert_invalid_config`]).
    fn invalid_config_reason(error: &SearchError) -> &str {
        match error {
            SearchError::InvalidConfig { reason, .. } => reason.as_str(),
            _ => "",
        }
    }

    // -- Tests -----------------------------------------------------------------

    #[test]
    fn config_defaults() {
        let config = RefreshWorkerConfig::new("/tmp/test-idx");
        assert_eq!(config.poll_interval, Duration::from_secs(1));
        assert_eq!(config.max_docs_per_cycle, 1000);
    }

    #[test]
    fn config_builder_methods() {
        let config = RefreshWorkerConfig::new("/tmp/test-idx")
            .with_poll_interval(Duration::from_millis(500))
            .with_max_docs_per_cycle(50);
        assert_eq!(config.poll_interval, Duration::from_millis(500));
        assert_eq!(config.max_docs_per_cycle, 50);
    }

    #[test]
    fn metrics_snapshot() {
        let metrics = RefreshMetrics::default();
        metrics.cycles.fetch_add(5, Ordering::Relaxed);
        metrics.docs_embedded.fetch_add(100, Ordering::Relaxed);
        let snap = metrics.snapshot();
        assert_eq!(snap.cycles, 5);
        assert_eq!(snap.docs_embedded, 100);
        assert_eq!(snap.docs_failed, 0);
    }

    #[test]
    fn run_cycle_empty_queue_returns_zero() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("empty");
            let queue = make_queue(10);
            let (worker, _cache) = make_worker(queue, &dir, 256);

            let count = worker.run_cycle(&cx).await.unwrap();
            assert_eq!(count, 0);
        });
    }

    #[test]
    fn run_cycle_embeds_and_rebuilds_index() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("rebuild");
            let queue = make_queue(100);
            submit(&queue, "doc-1", "Hello world");
            submit(&queue, "doc-2", "Goodbye world");

            let (worker, cache) = make_worker(queue.clone(), &dir, 256);

            let count = worker.run_cycle(&cx).await.unwrap();
            assert_eq!(count, 2);

            // Verify index was rebuilt.
            assert_eq!(worker.metrics().index_rebuilds.load(Ordering::Relaxed), 1);
            assert_eq!(worker.metrics().docs_embedded.load(Ordering::Relaxed), 2);

            // Queue should be empty after processing.
            assert!(queue.is_empty());

            // Cache should have the new index (2 docs, not the seed).
            let current = cache.current();
            assert_eq!(current.doc_count(), 2);
        });
    }

    /// Red proof (c), production surface: the bootstrap lane declares the
    /// producing identity (retained, C2) but the published artifacts are v1
    /// — the identity is DECLARED, never attested, and the next cycle's
    /// admission does not accept it.
    #[test]
    fn bootstrap_publishes_declared_identity_never_attested() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("declared-not-attested");
            let queue = make_queue(100);
            submit(&queue, "doc-1", "Hello world");
            let (worker, cache) = make_worker(queue.clone(), &dir, 256);
            worker.run_cycle(&cx).await.unwrap();

            let current = cache.current();
            assert!(
                current.fast_declared_identity().is_some(),
                "the bootstrap lane must declare the producing identity (C2 retention)"
            );
            assert_eq!(
                current.fast_space_fingerprint_hex(),
                Some(
                    StubEmbedder::new("stub-fast", 256)
                        .identity_bundle()
                        .space
                        .fingerprint()
                        .as_str()
                ),
            );
            assert!(
                !current.fast_identity_is_attested(),
                "a v1 bootstrap publication is DECLARED, never header-attested"
            );

            // Declared-only retention never unlocks the merge: the on-disk
            // generation is live v1 and the next cycle keeps the typed
            // containment refusal.
            submit(&queue, "doc-2", "Second document");
            let error = worker
                .run_cycle(&cx)
                .await
                .expect_err("declared-only generation must keep the identityless refusal");
            assert_invalid_config(
                &error,
                "refresh.fast_index_identity",
                "identityless-fsvi-v1",
            );

            // Staging refuses the same way: declared-not-attested rejects.
            let jobs = queue.drain_batch();
            let error = worker
                .stage_identity_bound_generation(&cx, &jobs)
                .await
                .expect_err("staging must not admit a declared-only (v1) generation");
            assert_invalid_config(
                &error,
                "refresh.fast_index_identity",
                "identityless-fsvi-v1",
            );
        });
    }

    #[test]
    fn run_cycle_records_content_hashes() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("hashes");
            let queue = make_queue(100);
            submit(&queue, "doc-1", "First document text");

            let (worker, _cache) = make_worker(queue.clone(), &dir, 256);
            worker.run_cycle(&cx).await.unwrap();

            // Submitting the same text again should be deduped.
            let outcome = queue
                .submit(EmbeddingRequest {
                    doc_id: "doc-1".into(),
                    text: "First document text".to_owned(),
                    metadata: None,
                    submitted_at: Instant::now(),
                })
                .unwrap();
            assert_eq!(outcome, JobOutcome::SkippedUnchanged);
        });
    }

    #[test]
    fn run_cycle_with_quality_embedder() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("quality");
            let queue = make_queue(100);
            submit(&queue, "doc-1", "Test document");

            let cache = make_cache(&dir, 256);
            let config =
                RefreshWorkerConfig::new(&dir).with_poll_interval(Duration::from_millis(10));
            let fast = Arc::new(StubEmbedder::new("stub-fast", 256));
            let quality = Arc::new(StubEmbedder::new("stub-quality", 384));
            let worker = RefreshWorker::new(config, queue.clone(), fast, cache.clone())
                .with_quality_embedder(quality);

            let count = worker.run_cycle(&cx).await.unwrap();
            assert_eq!(count, 1);

            // Index should have been rebuilt with quality tier.
            let index = cache.current();
            assert!(index.has_quality_index());
        });
    }

    #[test]
    fn pre_drain_refusal_does_not_charge_embedding_or_rebuild_metrics() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("multi");
            let queue = make_queue(100);
            let (worker, _cache) = make_worker(queue.clone(), &dir, 256);

            // Cycle 1: 2 docs.
            submit(&queue, "doc-1", "First");
            submit(&queue, "doc-2", "Second");
            worker.run_cycle(&cx).await.unwrap();

            // Cycle 2 refuses before draining or embedding the pending job.
            submit(&queue, "doc-3", "Third");
            worker
                .run_cycle(&cx)
                .await
                .expect_err("legacy incremental merge must fail closed");

            let snap = worker.metrics().snapshot();
            assert_eq!(snap.docs_embedded, 2);
            assert_eq!(snap.index_rebuilds, 1);
            assert_eq!(snap.rebuild_failures, 0);
            assert_eq!(queue.pending_count(), 1);
        });
    }

    #[test]
    fn run_cycle_respects_max_docs_per_cycle() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("limit");
            let queue = make_queue(100);
            for i in 0..5 {
                submit(&queue, &format!("doc-{i}"), "Payload");
            }

            let cache = make_cache(&dir, 256);
            let config = RefreshWorkerConfig::new(&dir)
                .with_poll_interval(Duration::from_millis(10))
                .with_max_docs_per_cycle(3);
            let fast = Arc::new(StubEmbedder::new("stub-fast", 256));
            let worker = RefreshWorker::new(config, queue.clone(), fast, cache.clone());

            let count = worker.run_cycle(&cx).await.unwrap();
            assert_eq!(count, 3);
            assert_eq!(queue.pending_count(), 2);
        });
    }

    #[test]
    fn identityless_incremental_refusal_preserves_active_generation() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("preserve-existing");
            let queue = make_queue(100);
            let (worker, cache) = make_worker(queue.clone(), &dir, 256);

            submit(&queue, "doc-1", "First");
            submit(&queue, "doc-2", "Second");
            worker.run_cycle(&cx).await.expect("first cycle");

            let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
            let original_bytes = std::fs::read(&fast_path).expect("read active generation");

            submit(&queue, "doc-3", "Third");
            let error = worker
                .run_cycle(&cx)
                .await
                .expect_err("identityless incremental merge must fail closed");
            assert!(matches!(
                error,
                SearchError::InvalidConfig { ref field, ref value, .. }
                    if field == "refresh.fast_index_identity"
                        && value == "identityless-fsvi-v1"
            ));
            assert_eq!(queue.pending_count(), 1, "failed job must be requeued");
            assert_eq!(
                std::fs::read(&fast_path).expect("reread active generation"),
                original_bytes,
                "refusal must not mutate or replace the active generation"
            );

            let current = cache.current();
            assert_eq!(
                current.doc_count(),
                2,
                "cache must continue serving the prior active generation"
            );
            let doc_ids: Vec<String> = current.iter_doc_ids().filter_map(Result::ok).collect();
            assert!(doc_ids.iter().any(|id| id == "doc-1"));
            assert!(doc_ids.iter().any(|id| id == "doc-2"));
            assert!(!doc_ids.iter().any(|id| id == "doc-3"));
        });
    }

    #[test]
    fn same_id_same_dimension_changed_space_cannot_bypass_identityless_refusal() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("fast-id-change");
            let queue = make_queue(100);
            let cache = make_cache(&dir, 256);
            let config = RefreshWorkerConfig::new(&dir);
            let original_embedder = Arc::new(StubEmbedder::in_space("stub-fast", 256, 0.0));
            let changed_embedder = Arc::new(StubEmbedder::in_space("stub-fast", 256, 100.0));
            assert_eq!(original_embedder.id(), changed_embedder.id());
            assert_eq!(original_embedder.dimension(), changed_embedder.dimension());
            let original_probe = original_embedder
                .embed(&cx, "space probe")
                .await
                .expect("original probe");
            let changed_probe = changed_embedder
                .embed(&cx, "space probe")
                .await
                .expect("changed probe");
            assert!(
                original_probe
                    .iter()
                    .zip(&changed_probe)
                    .any(|(original, changed)| original.to_bits() != changed.to_bits()),
                "fixture must represent a real same-id, same-dimension space change"
            );
            let original = RefreshWorker::new(
                config.clone(),
                queue.clone(),
                original_embedder,
                cache.clone(),
            );

            submit(&queue, "doc-old", "old generation");
            original.run_cycle(&cx).await.expect("initial generation");

            let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
            let original_bytes = std::fs::read(&fast_path).expect("read original fast index");
            let changed = RefreshWorker::new(config, queue.clone(), changed_embedder, cache);

            submit(&queue, "doc-new", "new generation");
            let error = changed
                .run_cycle(&cx)
                .await
                .expect_err("identityless generation must fail closed before vector merge");

            assert!(matches!(
                error,
                SearchError::InvalidConfig { ref field, ref value, .. }
                    if field == "refresh.fast_index_identity"
                        && value == "identityless-fsvi-v1"
            ));
            assert_eq!(queue.pending_count(), 1, "failed job must be requeued");
            assert_eq!(
                std::fs::read(&fast_path).expect("reread fast index"),
                original_bytes,
                "failed refresh must not re-stamp the existing generation"
            );
            let reopened = VectorIndex::open(&fast_path).expect("reopen original fast index");
            assert_eq!(reopened.embedder_id(), "stub-fast");
            let live_ids = reopened.live_doc_ids().expect("live ids");
            assert_eq!(live_ids.len(), 1);
            assert!(live_ids.contains("doc-old"));
        });
    }

    #[test]
    fn unchanged_ids_cannot_exhaust_retry_budget_while_legacy_generation_is_blocked() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("unchanged-tier-ids");
            let queue = make_queue(100);
            let cache = make_cache(&dir, 256);
            let worker = RefreshWorker::new(
                RefreshWorkerConfig::new(&dir),
                queue.clone(),
                Arc::new(StubEmbedder::new("stub-fast", 256)),
                cache.clone(),
            )
            .with_quality_embedder(Arc::new(StubEmbedder::new("stub-quality", 384)));

            submit(&queue, "doc-old", "old generation");
            worker.run_cycle(&cx).await.expect("initial generation");

            let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
            let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);
            let original_fast_bytes = std::fs::read(&fast_path).expect("read original fast index");
            let original_quality_bytes =
                std::fs::read(&quality_path).expect("read original quality index");

            submit(&queue, "doc-new", "new generation");
            let refusal_cycles = queue.config().max_retries + 2;
            for _ in 0..refusal_cycles {
                let error = worker
                    .run_cycle(&cx)
                    .await
                    .expect_err("matching display ids must not admit a legacy merge");
                assert!(matches!(
                    error,
                    SearchError::InvalidConfig { ref field, ref value, .. }
                        if field == "refresh.fast_index_identity"
                            && value == "identityless-fsvi-v1"
                ));
            }
            assert_eq!(
                queue.pending_count(),
                1,
                "permanent refusal must leave the job queued beyond max_retries"
            );
            assert_eq!(
                std::fs::read(&fast_path).expect("reread fast index"),
                original_fast_bytes,
                "refusal must not replace the fast generation"
            );
            assert_eq!(
                std::fs::read(&quality_path).expect("reread quality index"),
                original_quality_bytes,
                "refusal must not replace the quality generation"
            );
            assert_eq!(cache.current().doc_count(), 1);
            let fast = VectorIndex::open(&fast_path).expect("open fast index");
            let quality = VectorIndex::open(&quality_path).expect("open quality index");
            assert_eq!(fast.embedder_id(), "stub-fast");
            assert_eq!(quality.embedder_id(), "stub-quality");
            assert_eq!(fast.live_doc_ids().expect("fast live ids").len(), 1);
            assert_eq!(quality.live_doc_ids().expect("quality live ids").len(), 1);

            let pending = queue.drain_batch();
            assert_eq!(pending.len(), 1);
            assert_eq!(pending[0].doc_id, "doc-new");
            assert_eq!(
                pending[0].retry_count, 0,
                "pre-drain refusal must not consume retry budget"
            );
        });
    }

    #[test]
    fn failed_embedding_requeues_jobs() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("fail");
            let queue = make_queue(100);
            submit(&queue, "doc-1", "Will fail");

            let cache = make_cache(&dir, 256);
            let config = RefreshWorkerConfig::new(&dir);
            let failing = Arc::new(FailingEmbedder);
            let worker = RefreshWorker::new(config, queue.clone(), failing, cache);

            let count = worker.run_cycle(&cx).await.unwrap();
            assert_eq!(count, 0);

            // Job should have been requeued.
            assert_eq!(queue.pending_count(), 1);
            assert_eq!(worker.metrics().docs_failed.load(Ordering::Relaxed), 1);
        });
    }

    #[test]
    fn index_dir_accessor() {
        let dir = temp_index_dir("accessor");
        let queue = make_queue(10);
        let (worker, _cache) = make_worker(queue, &dir, 256);
        assert_eq!(worker.index_dir(), dir.as_path());
    }

    #[test]
    fn debug_format() {
        let dir = temp_index_dir("debug");
        let queue = make_queue(10);
        let (worker, _cache) = make_worker(queue, &dir, 256);
        let debug = format!("{worker:?}");
        assert!(debug.contains("RefreshWorker"));
        assert!(debug.contains("stub-fast"));
    }

    // ─── bd-qkop tests begin ───

    #[test]
    fn config_with_index_config() {
        let custom = TwoTierConfig::default();
        let config = RefreshWorkerConfig::new("/tmp/idx").with_index_config(custom.clone());
        assert!((config.index_config.rrf_k - custom.rrf_k).abs() < f64::EPSILON);
    }

    #[test]
    fn config_clone() {
        let config = RefreshWorkerConfig::new("/tmp/idx")
            .with_poll_interval(Duration::from_millis(42))
            .with_max_docs_per_cycle(7);
        #[allow(clippy::redundant_clone)]
        let cloned = config.clone();
        assert_eq!(cloned.poll_interval, Duration::from_millis(42));
        assert_eq!(cloned.max_docs_per_cycle, 7);
        assert_eq!(cloned.index_dir, PathBuf::from("/tmp/idx"));
    }

    #[test]
    fn config_debug() {
        let config = RefreshWorkerConfig::new("/tmp/idx");
        let debug = format!("{config:?}");
        assert!(debug.contains("RefreshWorkerConfig"));
        assert!(debug.contains("poll_interval"));
    }

    #[test]
    fn metrics_default_all_zeros() {
        let m = RefreshMetrics::default();
        assert_eq!(m.cycles.load(Ordering::Relaxed), 0);
        assert_eq!(m.docs_embedded.load(Ordering::Relaxed), 0);
        assert_eq!(m.docs_failed.load(Ordering::Relaxed), 0);
        assert_eq!(m.index_rebuilds.load(Ordering::Relaxed), 0);
        assert_eq!(m.rebuild_failures.load(Ordering::Relaxed), 0);
        assert_eq!(m.embed_time_us.load(Ordering::Relaxed), 0);
        assert_eq!(m.rebuild_time_us.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn metrics_snapshot_equality() {
        let a = RefreshMetricsSnapshot {
            cycles: 1,
            docs_embedded: 2,
            docs_failed: 0,
            index_rebuilds: 1,
            rebuild_failures: 0,
            embed_time_us: 100,
            rebuild_time_us: 50,
        };
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn metrics_snapshot_inequality() {
        let a = RefreshMetricsSnapshot {
            cycles: 1,
            docs_embedded: 2,
            docs_failed: 0,
            index_rebuilds: 1,
            rebuild_failures: 0,
            embed_time_us: 100,
            rebuild_time_us: 50,
        };
        let b = RefreshMetricsSnapshot { cycles: 2, ..a };
        assert_ne!(a, b);
    }

    #[test]
    fn metrics_snapshot_clone_copy() {
        let a = RefreshMetricsSnapshot {
            cycles: 5,
            docs_embedded: 10,
            docs_failed: 1,
            index_rebuilds: 3,
            rebuild_failures: 0,
            embed_time_us: 200,
            rebuild_time_us: 100,
        };
        #[allow(clippy::clone_on_copy)]
        let cloned = a.clone();
        let copied: RefreshMetricsSnapshot = a; // Copy
        assert_eq!(cloned, a);
        assert_eq!(copied, a);
    }

    #[test]
    fn metrics_snapshot_debug() {
        let snap = RefreshMetricsSnapshot {
            cycles: 0,
            docs_embedded: 0,
            docs_failed: 0,
            index_rebuilds: 0,
            rebuild_failures: 0,
            embed_time_us: 0,
            rebuild_time_us: 0,
        };
        let debug = format!("{snap:?}");
        assert!(debug.contains("RefreshMetricsSnapshot"));
        assert!(debug.contains("cycles"));
    }

    #[test]
    fn metrics_individual_increments() {
        let m = RefreshMetrics::default();
        m.cycles.fetch_add(3, Ordering::Relaxed);
        m.docs_failed.fetch_add(2, Ordering::Relaxed);
        m.rebuild_failures.fetch_add(1, Ordering::Relaxed);
        m.embed_time_us.fetch_add(500, Ordering::Relaxed);
        m.rebuild_time_us.fetch_add(300, Ordering::Relaxed);
        let snap = m.snapshot();
        assert_eq!(snap.cycles, 3);
        assert_eq!(snap.docs_failed, 2);
        assert_eq!(snap.rebuild_failures, 1);
        assert_eq!(snap.embed_time_us, 500);
        assert_eq!(snap.rebuild_time_us, 300);
    }

    #[test]
    fn worker_metrics_accessor_returns_shared_arc() {
        let dir = temp_index_dir("metrics-arc");
        let queue = make_queue(10);
        let (worker, _cache) = make_worker(queue, &dir, 256);
        let metrics = worker.metrics();
        metrics.cycles.fetch_add(1, Ordering::Relaxed);
        // Same Arc should reflect the update.
        assert_eq!(worker.metrics().cycles.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn worker_with_quality_embedder_sets_embedder() {
        let dir = temp_index_dir("quality-set");
        let queue = make_queue(10);
        let cache = make_cache(&dir, 256);
        let config = RefreshWorkerConfig::new(&dir);
        let fast = Arc::new(StubEmbedder::new("stub-fast", 256));
        let quality = Arc::new(StubEmbedder::new("stub-quality", 384));
        let worker = RefreshWorker::new(config, queue, fast, cache).with_quality_embedder(quality);
        // Should be set (verified indirectly by debug containing only fast id)
        let debug = format!("{worker:?}");
        assert!(debug.contains("stub-fast"));
    }

    #[test]
    fn duplicate_doc_id_keeps_latest_in_cycle() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("dedup-cycle");
            let queue = make_queue(100);
            // Submit same doc_id with different texts.
            submit(&queue, "doc-dup", "First version");
            submit(&queue, "doc-dup", "Second version");

            let (worker, cache) = make_worker(queue.clone(), &dir, 256);
            let count = worker.run_cycle(&cx).await.unwrap();
            // Both submitted but doc count should be 1 (deduped).
            assert!(count >= 1);
            let current = cache.current();
            // Only one doc-dup in the index.
            let doc_ids: Vec<String> = current.iter_doc_ids().filter_map(Result::ok).collect();
            let dup_count = doc_ids.iter().filter(|id| *id == "doc-dup").count();
            assert_eq!(dup_count, 1);
        });
    }

    #[test]
    fn quality_embedder_failure_proceeds_with_fast_only() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("quality-fail");
            let queue = make_queue(100);
            submit(&queue, "doc-1", "Test document");

            let cache = make_cache(&dir, 256);
            let config =
                RefreshWorkerConfig::new(&dir).with_poll_interval(Duration::from_millis(10));
            let fast = Arc::new(StubEmbedder::new("stub-fast", 256));
            let failing_quality: Arc<dyn Embedder> = Arc::new(FailingEmbedder);
            let worker = RefreshWorker::new(config, queue.clone(), fast, cache.clone())
                .with_quality_embedder(failing_quality);

            let count = worker.run_cycle(&cx).await.unwrap();
            assert_eq!(count, 1);

            // Index should still have the doc (fast-only).
            let current = cache.current();
            assert_eq!(current.doc_count(), 1);
            // But no quality index since quality embedder failed.
            assert!(!current.has_quality_index());
        });
    }

    #[test]
    fn run_cycle_empty_after_prior_cycle_returns_zero() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("empty-after");
            let queue = make_queue(100);
            submit(&queue, "doc-1", "First");
            let (worker, _cache) = make_worker(queue.clone(), &dir, 256);

            worker.run_cycle(&cx).await.unwrap();
            // Second cycle with empty queue.
            let count = worker.run_cycle(&cx).await.unwrap();
            assert_eq!(count, 0);
        });
    }

    // ─── bd-qkop tests end ───

    // ─── bd-wt20 tests begin ──────────────────────────────────────────

    #[test]
    fn refresh_metrics_debug_format() {
        let m = RefreshMetrics::default();
        m.cycles.fetch_add(7, Ordering::Relaxed);
        m.docs_embedded.fetch_add(42, Ordering::Relaxed);
        let debug = format!("{m:?}");
        assert!(debug.contains("RefreshMetrics"));
        assert!(debug.contains("cycles"));
        assert!(debug.contains("docs_embedded"));
    }

    #[test]
    fn config_index_dir_from_new() {
        let config = RefreshWorkerConfig::new("/data/my-index");
        assert_eq!(config.index_dir, PathBuf::from("/data/my-index"));
    }

    #[test]
    fn config_index_config_override() {
        let custom = TwoTierConfig {
            rrf_k: 42.0,
            ..TwoTierConfig::default()
        };
        let config = RefreshWorkerConfig::new("/tmp/idx").with_index_config(custom);
        assert!((config.index_config.rrf_k - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn metrics_concurrent_increments() {
        let m = Arc::new(RefreshMetrics::default());
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let metrics = m.clone();
                std::thread::spawn(move || {
                    for _ in 0..100 {
                        metrics.docs_embedded.fetch_add(1, Ordering::Relaxed);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(m.docs_embedded.load(Ordering::Relaxed), 400);
    }

    #[test]
    fn identityless_incremental_refusal_does_not_overwrite_existing_doc() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("update-doc");
            let queue = make_queue(100);
            let (worker, cache) = make_worker(queue.clone(), &dir, 256);

            // Cycle 1: index doc-1 with first text.
            submit(&queue, "doc-1", "First version of the document");
            worker.run_cycle(&cx).await.unwrap();
            assert_eq!(cache.current().doc_count(), 1);

            // Cycle 2: submit doc-1 again with different text (hash changed).
            // Force a new submission by changing text.
            queue
                .submit(EmbeddingRequest {
                    doc_id: "doc-1".into(),
                    text: "Second version completely different".to_owned(),
                    metadata: None,
                    submitted_at: Instant::now(),
                })
                .unwrap();
            worker
                .run_cycle(&cx)
                .await
                .expect_err("legacy update must require an identity-bound rebuild");

            // The active generation remains available, and the update remains
            // queued for a later full identity-bound rebuild.
            let current = cache.current();
            assert_eq!(current.doc_count(), 1);
            let doc_ids: Vec<String> = current.iter_doc_ids().filter_map(Result::ok).collect();
            assert_eq!(doc_ids.len(), 1);
            assert_eq!(doc_ids[0], "doc-1");
            assert_eq!(queue.pending_count(), 1);
        });
    }

    /// Fleet-review regression pin: refusing an identityless generation must
    /// neither resurrect soft-deleted documents nor drop WAL-resident appends.
    #[test]
    fn identityless_rebuild_refusal_preserves_tombstones_and_wal_residents() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("tombstone-wal-rebuild");
            let queue = make_queue(100);
            let cache = make_cache(&dir, 256);
            let config =
                RefreshWorkerConfig::new(&dir).with_poll_interval(Duration::from_millis(10));
            let fast = Arc::new(StubEmbedder::new("stub-fast", 256));
            let worker = RefreshWorker::new(config, queue.clone(), fast.clone(), cache.clone());

            submit(&queue, "doc-keep", "kept document");
            submit(&queue, "doc-delete", "doomed document");
            worker.run_cycle(&cx).await.unwrap();
            assert_eq!(cache.current().doc_count(), 2);

            // Out-of-band mutations through the public index API, exactly
            // as an application deleting/appending between cycles does.
            let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
            {
                let mut index =
                    frankensearch_index::VectorIndex::open(&fast_path).expect("open fast tier");
                assert!(index.soft_delete("doc-delete").expect("soft delete"));
                let dim = index.dimension();
                index
                    .append("doc-wal", &vec![0.25_f32; dim])
                    .expect("wal append");
            }

            submit(&queue, "doc-new", "new document");
            worker
                .run_cycle(&cx)
                .await
                .expect_err("legacy WAL-bearing generation must not be republished");

            let preserved =
                frankensearch_index::VectorIndex::open(&fast_path).expect("reopen fast tier");
            let live = preserved.live_doc_ids().expect("live ids");
            assert!(
                !live.contains("doc-delete"),
                "refusal must not resurrect the soft-deleted document: {live:?}"
            );
            assert!(
                live.contains("doc-wal"),
                "refusal must preserve the WAL-resident append: {live:?}"
            );
            assert!(live.contains("doc-keep"), "{live:?}");
            assert!(!live.contains("doc-new"), "{live:?}");
            assert_eq!(queue.pending_count(), 1);
        });
    }

    #[test]
    fn identityless_refusal_preserves_prior_quality_tier() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("quality-preserve");
            let queue = make_queue(100);

            // First cycle with quality embedder.
            let cache = make_cache(&dir, 256);
            let config =
                RefreshWorkerConfig::new(&dir).with_poll_interval(Duration::from_millis(10));
            let fast = Arc::new(StubEmbedder::new("stub-fast", 256));
            let quality = Arc::new(StubEmbedder::new("stub-quality", 384));
            let worker_with_quality =
                RefreshWorker::new(config, queue.clone(), fast.clone(), cache.clone())
                    .with_quality_embedder(quality);

            submit(&queue, "doc-1", "Test document");
            worker_with_quality.run_cycle(&cx).await.unwrap();
            assert!(cache.current().has_quality_index());

            // Second cycle without quality embedder (fast-only worker).
            let config2 =
                RefreshWorkerConfig::new(&dir).with_poll_interval(Duration::from_millis(10));
            let worker_fast_only = RefreshWorker::new(config2, queue.clone(), fast, cache.clone());

            submit(&queue, "doc-2", "Another document");
            worker_fast_only
                .run_cycle(&cx)
                .await
                .expect_err("legacy two-tier generation must fail closed");

            // The prior quality generation remains active and the new work is
            // requeued for a full identity-bound rebuild.
            let current = cache.current();
            assert_eq!(current.doc_count(), 1);
            assert!(current.has_quality_index());
            assert_eq!(queue.pending_count(), 1);
        });
    }

    #[test]
    fn run_cycle_metrics_timing_nonzero() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("timing");
            let queue = make_queue(100);
            submit(&queue, "doc-1", "Timing test");

            let (worker, _cache) = make_worker(queue, &dir, 256);
            worker.run_cycle(&cx).await.unwrap();

            let snap = worker.metrics().snapshot();
            // Embed and rebuild should take some nonzero time.
            assert!(snap.embed_time_us > 0 || snap.rebuild_time_us > 0);
        });
    }

    #[test]
    fn worker_new_without_quality_debug() {
        let dir = temp_index_dir("no-quality");
        let queue = make_queue(10);
        let (worker, _cache) = make_worker(queue, &dir, 256);
        // Debug uses `finish_non_exhaustive` and includes fast embedder id.
        let debug = format!("{worker:?}");
        assert!(debug.contains("RefreshWorker"));
        assert!(debug.contains("stub-fast"));
    }

    // ─── bd-wt20 tests end ────────────────────────────────────────────

    // ─── bd-9xuj T2 C4-write red proofs ───────────────────────────────

    const V2_DIM: usize = 8;

    /// Fixture: cache seeded from an empty v1 FALLBACK artifact, then a live
    /// attested v2 canonical fast generation written under `bundle`.
    fn v2_canonical_fixture(
        label: &str,
        bundle: &EmbeddingIdentityBundleV1,
        rows: &[(&str, Vec<f32>)],
        sequence: u64,
    ) -> (PathBuf, Arc<IndexCache>, FsviV2IdentityBinding, PathBuf) {
        let dir = temp_index_dir(label);
        let cache = make_cache_with_fallback_seed(&dir, V2_DIM);
        let binding = v2_binding(bundle, sequence);
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_v2_tier(&fast_path, &binding, rows);
        (dir, cache, binding, fast_path)
    }

    /// Finding-1 regression + composite-authority guard: an attested,
    /// same-producer, space-matching v2 canonical generation is recognized
    /// through inspection/admission (never plain-opened), the identity gates
    /// PASS, and the cycle still refuses pre-drain with the typed
    /// composite-generation-authority reason — with ZERO side effects: no
    /// drain, no embed, no write, no staging.
    #[test]
    fn live_v2_same_producer_cycle_refuses_composite_authority_with_zero_side_effects() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let embedder = Arc::new(StubEmbedder::new("v2-stub", V2_DIM));
            let (dir, cache, _binding, fast_path) = v2_canonical_fixture(
                "v2-composite-guard",
                embedder.identity_bundle(),
                &[("doc-old", normalized(V2_DIM, 0.5))],
                7,
            );
            let canonical_bytes = std::fs::read(&fast_path).expect("read canonical");

            let queue = make_queue(100);
            submit(&queue, "doc-new", "new document");
            let worker = RefreshWorker::new(
                RefreshWorkerConfig::new(&dir),
                queue.clone(),
                embedder,
                cache,
            );

            let error = worker
                .run_cycle(&cx)
                .await
                .expect_err("canonical publication must be refused until composite authority");
            assert_invalid_config(
                &error,
                "refresh.canonical_publication",
                "composite-generation-authority-unavailable",
            );
            assert!(
                invalid_config_reason(&error).contains("bd-xomn"),
                "reason must name the dependency"
            );

            // Zero side effects: nothing drained, embedded, written, staged.
            assert_eq!(queue.pending_count(), 1, "refusal must fire before drain");
            let snap = worker.metrics().snapshot();
            assert_eq!(snap.docs_embedded, 0);
            assert_eq!(snap.index_rebuilds, 0);
            assert_eq!(snap.rebuild_failures, 0);
            assert_eq!(
                std::fs::read(&fast_path).expect("reread canonical"),
                canonical_bytes,
                "admission/inspection must not mutate the canonical artifact"
            );
            assert!(
                !dir.join(STAGED_V2_DIR_NAME).exists(),
                "run_cycle must not stage as a side effect"
            );
            let drained = queue.drain_batch();
            assert_eq!(drained[0].retry_count, 0, "no retry budget consumed");
        });
    }

    /// Red proof (a), positive re-enablement: the same-producer, attested,
    /// space-matching merge the containment refused is now staged
    /// successfully — carried rows plus updates plus new documents — and the
    /// replacement's identity is verified from the republished artifact's
    /// OWN header bytes through `open_admitted_v2` (reviewer caution: never
    /// only against the in-memory bundle).
    #[test]
    fn staging_same_producer_merges_and_republishes_attested_v2() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let embedder = Arc::new(StubEmbedder::new("v2-stub", V2_DIM));
            let (dir, cache, canonical_binding, fast_path) = v2_canonical_fixture(
                "v2-stage-positive",
                embedder.identity_bundle(),
                &[
                    ("old-1", normalized(V2_DIM, 0.25)),
                    ("old-2", normalized(V2_DIM, 0.75)),
                ],
                7,
            );
            let canonical_bytes = std::fs::read(&fast_path).expect("read canonical");

            let queue = make_queue(100);
            submit(&queue, "old-2", "updated content for old-2");
            submit(&queue, "doc-3", "a brand new document");
            let jobs = queue.drain_batch();
            assert_eq!(jobs.len(), 2);

            let worker = RefreshWorker::new(
                RefreshWorkerConfig::new(&dir),
                queue.clone(),
                embedder.clone(),
                cache,
            );
            let staged = worker
                .stage_identity_bound_generation(&cx, &jobs)
                .await
                .expect("same-producer attested merge must stage successfully");

            // The staged two-tier view is attested and carries all rows.
            assert_eq!(staged.index.doc_count(), 3);
            assert!(staged.index.fast_identity_is_attested());
            assert_eq!(
                staged.index.fast_space_fingerprint_hex(),
                Some(embedder.identity_bundle().space.fingerprint().as_str())
            );
            assert_eq!(
                staged.fast_binding.generation().sequence,
                8,
                "replacement generation must succeed the attested sequence 7"
            );

            // REVIEWER CAUTION: prove the identity from the artifact's own
            // header bytes via the only legitimate v2 open path.
            let staged_owner =
                VectorIndex::open_admitted_v2(&staged.fast_path, &staged.fast_binding)
                    .expect("staged artifact must admit exactly");
            assert_eq!(
                fingerprint_hex(&staged_owner.identity_v2().space_fingerprint),
                embedder.identity_bundle().space.fingerprint(),
                "the header of the republished artifact must carry the producing space \
                 fingerprint bit-for-bit"
            );
            let staged_vectors = admitted_vectors_by_doc(&staged_owner);
            assert_eq!(staged_vectors.len(), 3);

            // Carried row is byte-stable; updated row actually changed.
            let canonical_owner = VectorIndex::open_admitted_v2(&fast_path, &canonical_binding)
                .expect("canonical artifact still admits");
            let canonical_vectors = admitted_vectors_by_doc(&canonical_owner);
            assert_eq!(
                staged_vectors.get("old-1"),
                canonical_vectors.get("old-1"),
                "carried rows must be preserved exactly"
            );
            assert_ne!(
                staged_vectors.get("old-2"),
                canonical_vectors.get("old-2"),
                "updated rows must carry the NEW embedding"
            );
            assert!(staged_vectors.contains_key("doc-3"));

            // Staging is non-canonical and queue-neutral.
            assert!(staged.fast_path.starts_with(dir.join(STAGED_V2_DIR_NAME)));
            assert_eq!(
                std::fs::read(&fast_path).expect("reread canonical"),
                canonical_bytes,
                "staging must not touch the canonical generation"
            );
            assert_eq!(queue.pending_count(), 0);
        });
    }

    /// Red proof (b): a cross-space attempt is rejected with the NEW typed
    /// space-identity reason — not the generic republication refusal — both
    /// at the staging seam and pre-drain in the canonical cycle.
    #[test]
    fn cross_space_attempt_rejects_with_typed_space_identity() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            // Canonical generation from model alpha; executing embedder is
            // model beta at the SAME dimension (the case dimension checks
            // can never catch).
            let alpha = EmbeddingIdentityBundleV1::explicit_test_model(
                "canonical-model-alpha",
                u32::try_from(V2_DIM).expect("dim"),
            );
            let (dir, cache, _binding, fast_path) = v2_canonical_fixture(
                "v2-cross-space",
                &alpha,
                &[("doc-old", normalized(V2_DIM, 0.5))],
                3,
            );
            let canonical_bytes = std::fs::read(&fast_path).expect("read canonical");
            let beta = Arc::new(StubEmbedder::new("beta-stub", V2_DIM));
            let beta_space_fingerprint = beta.identity_bundle().space.fingerprint();
            assert_ne!(beta_space_fingerprint, alpha.space.fingerprint());

            let queue = make_queue(100);
            submit(&queue, "doc-new", "new document");
            let worker =
                RefreshWorker::new(RefreshWorkerConfig::new(&dir), queue.clone(), beta, cache);

            // Canonical cycle: typed cross-space refusal, pre-drain.
            let error = worker
                .run_cycle(&cx)
                .await
                .expect_err("cross-space merge must be refused");
            assert_invalid_config(
                &error,
                "refresh.fast_space_identity",
                &beta_space_fingerprint,
            );
            assert!(
                invalid_config_reason(&error).contains(&alpha.space.fingerprint()),
                "reason must carry the attested space fingerprint"
            );
            assert_eq!(queue.pending_count(), 1);

            // Staging seam: same typed refusal.
            let jobs = queue.drain_batch();
            let error = worker
                .stage_identity_bound_generation(&cx, &jobs)
                .await
                .expect_err("cross-space staging must be refused");
            assert_invalid_config(
                &error,
                "refresh.fast_space_identity",
                &beta_space_fingerprint,
            );
            assert_eq!(
                std::fs::read(&fast_path).expect("reread canonical"),
                canonical_bytes
            );
        });
    }

    /// Red proof (d): a producer that IS conformance-certified against the
    /// attested producer (same space, byte-identical golden certificate,
    /// different attested backend) — the exact
    /// `ConformanceCompatibleProducer` shape — is refused with the typed
    /// `executing-producer-attestation-unavailable` reason carrying both
    /// producer fingerprints (truthful telemetry).
    #[test]
    fn certified_sibling_producer_refused_attestation_unavailable() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let base = EmbeddingIdentityBundleV1::explicit_test_model(
                "shared-space-model",
                u32::try_from(V2_DIM).expect("dim"),
            );
            let mut sibling = base.clone();
            "alternate-conformant-backend".clone_into(&mut sibling.producer.backend);
            sibling.validate().expect("sibling must validate");
            assert!(
                sibling.is_conformance_compatible_with(&base)
                    && base.is_conformance_compatible_with(&sibling),
                "fixture must be the certified conformance-compatible shape"
            );
            assert_ne!(base.producer.fingerprint(), sibling.producer.fingerprint());
            assert_eq!(base.space.fingerprint(), sibling.space.fingerprint());

            let (dir, cache, _binding, fast_path) = v2_canonical_fixture(
                "v2-certified-sibling",
                &base,
                &[("doc-old", normalized(V2_DIM, 0.5))],
                3,
            );
            let canonical_bytes = std::fs::read(&fast_path).expect("read canonical");
            let executing = Arc::new(StubEmbedder::with_identity(
                "sibling-stub",
                V2_DIM,
                sibling.clone(),
            ));

            let queue = make_queue(100);
            submit(&queue, "doc-new", "new document");
            let worker = RefreshWorker::new(
                RefreshWorkerConfig::new(&dir),
                queue.clone(),
                executing,
                cache,
            );

            let error = worker
                .run_cycle(&cx)
                .await
                .expect_err("certified sibling must still be refused (SameProducer only)");
            assert_invalid_config(
                &error,
                "refresh.fast_producer_conformance",
                "executing-producer-attestation-unavailable",
            );
            let reason = invalid_config_reason(&error);
            assert!(
                reason.contains(&base.producer.fingerprint())
                    && reason.contains(&sibling.producer.fingerprint()),
                "truthful telemetry must carry both producer fingerprints: {reason}"
            );

            let jobs = queue.drain_batch();
            let error = worker
                .stage_identity_bound_generation(&cx, &jobs)
                .await
                .expect_err("staging must refuse the certified sibling too");
            assert_invalid_config(
                &error,
                "refresh.fast_producer_conformance",
                "executing-producer-attestation-unavailable",
            );
            assert_eq!(
                std::fs::read(&fast_path).expect("reread canonical"),
                canonical_bytes
            );
        });
    }

    /// Red proof (e), reviewer addition (carry-forward-#1 attack shape): a
    /// same-space, DIFFERENT-producer bundle whose golden-vector certificate
    /// is a byte-identical COPY of the attested producer's certificate must
    /// REJECT. Trivial under SameProducer-only — and exactly the regression
    /// guard that must stay red when the conformance-compatible lane opens:
    /// certificate bytes can be copied; executing-producer attestation
    /// cannot.
    #[test]
    fn copied_certificate_foreign_producer_rejects() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let victim = EmbeddingIdentityBundleV1::explicit_test_model(
                "victim-space-model",
                u32::try_from(V2_DIM).expect("dim"),
            );
            let mut attacker = victim.clone();
            "attacker-backend".clone_into(&mut attacker.producer.backend);
            "attacker-rev-1".clone_into(&mut attacker.producer.implementation_revision);
            attacker.validate().expect("attacker bundle validates");
            assert_eq!(
                attacker.producer.golden_vectors, victim.producer.golden_vectors,
                "the certificate is a byte-identical copy"
            );
            assert_ne!(
                attacker.producer.fingerprint(),
                victim.producer.fingerprint()
            );
            assert_eq!(attacker.space.fingerprint(), victim.space.fingerprint());

            let (dir, cache, _binding, fast_path) = v2_canonical_fixture(
                "v2-copied-cert",
                &victim,
                &[("doc-old", normalized(V2_DIM, 0.5))],
                3,
            );
            let canonical_bytes = std::fs::read(&fast_path).expect("read canonical");
            let executing = Arc::new(StubEmbedder::with_identity(
                "attacker-stub",
                V2_DIM,
                attacker,
            ));

            let queue = make_queue(100);
            submit(&queue, "doc-new", "new document");
            let worker = RefreshWorker::new(
                RefreshWorkerConfig::new(&dir),
                queue.clone(),
                executing,
                cache,
            );

            let error = worker
                .run_cycle(&cx)
                .await
                .expect_err("copied certificate must never admit a foreign producer");
            assert_invalid_config(
                &error,
                "refresh.fast_producer_conformance",
                "executing-producer-attestation-unavailable",
            );

            let jobs = queue.drain_batch();
            let error = worker
                .stage_identity_bound_generation(&cx, &jobs)
                .await
                .expect_err("staging must reject the copied certificate too");
            assert_invalid_config(
                &error,
                "refresh.fast_producer_conformance",
                "executing-producer-attestation-unavailable",
            );
            assert_eq!(
                std::fs::read(&fast_path).expect("reread canonical"),
                canonical_bytes
            );
        });
    }

    /// The per-record seam half of guard 7: an embedder whose BOUND outputs
    /// carry a conformance-certified sibling producer while its `identity()`
    /// attests another is caught at the bound-record seam
    /// (`verify_producer_conformance` per embedding), not just at the
    /// artifact gate.
    #[test]
    fn two_faced_embedder_refused_at_bound_record_seam() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let public = EmbeddingIdentityBundleV1::explicit_test_model(
                "two-faced-model",
                u32::try_from(V2_DIM).expect("dim"),
            );
            let mut bound_side = public.clone();
            "alternate-conformant-backend".clone_into(&mut bound_side.producer.backend);
            bound_side.validate().expect("bound-side validates");
            assert!(bound_side.is_conformance_compatible_with(&public));

            let dir = temp_index_dir("two-faced");
            let queue = make_queue(100);
            let cache = make_cache(&dir, V2_DIM);
            let embedder = Arc::new(TwoFacedEmbedder {
                public_identity: public,
                bound_identity: bound_side,
                dimension: V2_DIM,
            });
            let worker = RefreshWorker::new(
                RefreshWorkerConfig::new(&dir),
                queue.clone(),
                embedder,
                cache,
            );

            submit(&queue, "doc-1", "some document");
            let jobs = queue.drain_batch();
            let error = worker
                .stage_identity_bound_generation(&cx, &jobs)
                .await
                .expect_err("bound records from a different producer must be refused");
            assert_invalid_config(
                &error,
                "refresh.fast_producer_conformance",
                "executing-producer-attestation-unavailable",
            );
        });
    }

    /// Finding-2 pin: canonical installation of a staged (proven) generation
    /// is refused with the typed composite-generation-authority reason, and
    /// the canonical directory is untouched by both staging and the refusal.
    #[test]
    fn publish_staged_canonical_refuses_split_generation() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("staged-publish-guard");
            let queue = make_queue(100);
            let cache = make_cache(&dir, V2_DIM);
            let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
            let seed_bytes = std::fs::read(&fast_path).expect("read v1 seed");
            let embedder = Arc::new(StubEmbedder::new("v2-stub", V2_DIM));
            let worker = RefreshWorker::new(
                RefreshWorkerConfig::new(&dir),
                queue.clone(),
                embedder,
                cache,
            );

            submit(&queue, "doc-1", "some document");
            let jobs = queue.drain_batch();
            let staged = worker
                .stage_identity_bound_generation(&cx, &jobs)
                .await
                .expect("staging over an empty seed must succeed");
            assert!(staged.fast_path.exists());
            assert!(staged.index.fast_identity_is_attested());

            let error = worker
                .publish_staged_canonical(&staged)
                .expect_err("split canonical publication must be refused in this slice");
            assert_invalid_config(
                &error,
                "refresh.canonical_publication",
                "composite-generation-authority-unavailable",
            );
            assert!(invalid_config_reason(&error).contains("bd-xomn"));

            assert_eq!(
                std::fs::read(&fast_path).expect("reread canonical seed"),
                seed_bytes,
                "neither staging nor the publication refusal may touch the canonical files"
            );
        });
    }

    /// Tombstoned rows in an attested generation stay dead through the merge.
    #[test]
    fn staging_never_resurrects_tombstoned_rows() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let embedder = Arc::new(StubEmbedder::new("v2-stub", V2_DIM));
            let dir = temp_index_dir("v2-tombstones");
            let cache = make_cache_with_fallback_seed(&dir, V2_DIM);
            let binding = v2_binding(embedder.identity_bundle(), 4);
            let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
            {
                let mut writer =
                    VectorIndex::create_v2(&fast_path, binding.clone()).expect("create_v2");
                writer
                    .write_record("doc-keep", &normalized(V2_DIM, 0.5))
                    .expect("live row");
                writer
                    .write_tombstone_record("doc-dead", &normalized(V2_DIM, 0.7))
                    .expect("tombstone row");
                writer.finish().expect("finish");
            }

            let queue = make_queue(100);
            submit(&queue, "doc-b", "a new document");
            let jobs = queue.drain_batch();
            let worker = RefreshWorker::new(
                RefreshWorkerConfig::new(&dir),
                queue.clone(),
                embedder,
                cache,
            );
            let staged = worker
                .stage_identity_bound_generation(&cx, &jobs)
                .await
                .expect("staging must succeed");

            let owner = VectorIndex::open_admitted_v2(&staged.fast_path, &staged.fast_binding)
                .expect("admit staged");
            let vectors = admitted_vectors_by_doc(&owner);
            assert!(vectors.contains_key("doc-keep"));
            assert!(vectors.contains_key("doc-b"));
            assert!(
                !vectors.contains_key("doc-dead"),
                "a tombstoned document must never be resurrected by the merge"
            );
        });
    }

    /// Quality tier end-to-end: carried + new quality rows are staged under
    /// the quality embedder's identity and admit exactly.
    #[test]
    fn staging_quality_tier_carries_and_binds() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Arc::new(StubEmbedder::new("v2-fast", V2_DIM));
            let quality = Arc::new(StubEmbedder::new("v2-quality", 16));
            let (dir, cache, _fast_binding, _fast_path) = v2_canonical_fixture(
                "v2-quality-carry",
                fast.identity_bundle(),
                &[("old-1", normalized(V2_DIM, 0.25))],
                2,
            );
            let quality_binding = v2_binding(quality.identity_bundle(), 2);
            let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);
            write_v2_tier(
                &quality_path,
                &quality_binding,
                &[("old-1", normalized(16, 0.5))],
            );

            let queue = make_queue(100);
            submit(&queue, "doc-2", "a second document");
            let jobs = queue.drain_batch();
            let worker =
                RefreshWorker::new(RefreshWorkerConfig::new(&dir), queue.clone(), fast, cache)
                    .with_quality_embedder(quality.clone());

            let staged = worker
                .stage_identity_bound_generation(&cx, &jobs)
                .await
                .expect("two-tier staging must succeed");
            assert!(staged.quality_path.is_some());
            assert!(staged.index.quality_identity_is_attested());

            let quality_owner = VectorIndex::open_admitted_v2(
                staged.quality_path.as_ref().expect("quality path"),
                staged.quality_binding.as_ref().expect("quality binding"),
            )
            .expect("staged quality admits");
            assert_eq!(
                fingerprint_hex(&quality_owner.identity_v2().space_fingerprint),
                quality.identity_bundle().space.fingerprint(),
                "staged quality header must carry the quality producer's space"
            );
            let vectors = admitted_vectors_by_doc(&quality_owner);
            assert!(vectors.contains_key("old-1"), "carried quality row");
            assert!(vectors.contains_key("doc-2"), "new quality row");
        });
    }

    /// An attested quality tier with no quality embedder cannot be
    /// republished identity-bound: typed refusal, pre-drain.
    #[test]
    fn live_v2_quality_without_quality_embedder_refuses_republication() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Arc::new(StubEmbedder::new("v2-fast", V2_DIM));
            let quality_bundle =
                EmbeddingIdentityBundleV1::explicit_test_model("v2-quality-model", 16);
            let (dir, cache, _fast_binding, _fast_path) = v2_canonical_fixture(
                "v2-quality-missing-embedder",
                fast.identity_bundle(),
                &[("old-1", normalized(V2_DIM, 0.25))],
                2,
            );
            let quality_binding = v2_binding(&quality_bundle, 2);
            let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);
            write_v2_tier(
                &quality_path,
                &quality_binding,
                &[("old-1", normalized(16, 0.5))],
            );

            let queue = make_queue(100);
            submit(&queue, "doc-2", "a second document");
            let worker =
                RefreshWorker::new(RefreshWorkerConfig::new(&dir), queue.clone(), fast, cache);

            let error = worker
                .run_cycle(&cx)
                .await
                .expect_err("attested quality without a quality embedder must refuse");
            assert_invalid_config(
                &error,
                "refresh.index_publication",
                "identity-bound-republication-unavailable",
            );
            assert_eq!(queue.pending_count(), 1, "pre-drain refusal");
        });
    }

    // ─── C4-write r2: read-only pre-drain classification + retained owners ──

    /// Sorted (name, byte length, mtime) manifest of a directory's entries —
    /// the invariance witness for "classification touched nothing here".
    /// atime is deliberately NOT asserted (relatime makes it flaky and the
    /// no-atime open is best-effort by platform).
    fn dir_manifest(dir: &Path) -> Vec<(String, u64, std::time::SystemTime)> {
        let mut entries: Vec<(String, u64, std::time::SystemTime)> = std::fs::read_dir(dir)
            .expect("read dir")
            .map(|entry| {
                let entry = entry.expect("dir entry");
                let metadata = entry.metadata().expect("entry metadata");
                (
                    entry.file_name().to_string_lossy().into_owned(),
                    metadata.len(),
                    metadata.modified().expect("entry mtime"),
                )
            })
            .collect();
        entries.sort();
        entries
    }

    /// Plant a WAL sidecar beside `target_main` whose compaction generation
    /// cannot match the target's generation-0 main slab: the donor index is
    /// compacted once (main generation 0 -> 1) before the donor WAL is
    /// written, so the transplanted WAL carries generation next(1) = 2 while
    /// the target expects next(0) = 1.
    ///
    /// A control copy proves the staleness precondition the hazardous way:
    /// mutable `VectorIndex::open` on a copy of the pair DELETES the WAL.
    fn transplant_stale_wal(label: &str, target_main: &Path, dimension: usize) {
        use frankensearch_index::wal::wal_path_for;

        let donor_dir = temp_index_dir(&format!("{label}-wal-donor"));
        let donor_main = donor_dir.join("donor.idx");
        let mut writer =
            VectorIndex::create(&donor_main, "stub-fast", dimension).expect("create donor");
        writer
            .write_record("donor-seed", &normalized(dimension, 0.91))
            .expect("write donor seed");
        writer.finish().expect("finish donor");
        {
            let mut donor = VectorIndex::open(&donor_main).expect("open donor");
            donor
                .append("donor-wal-pre", &normalized(dimension, 0.81))
                .expect("append pre-compaction");
            donor.compact().expect("compact donor to bump generation");
            donor
                .append("donor-wal-resident", &normalized(dimension, 0.71))
                .expect("append post-compaction");
        }
        let donor_wal = wal_path_for(&donor_main);
        assert!(donor_wal.exists(), "donor WAL must exist");
        std::fs::copy(&donor_wal, wal_path_for(target_main)).expect("transplant WAL");

        // Control arm: prove the transplanted WAL is stale for a
        // generation-0 main slab by demonstrating the exact hazard on a
        // throwaway copy — the mutable v1 open deletes it.
        let control_dir = temp_index_dir(&format!("{label}-wal-control"));
        let control_main = control_dir.join("control.idx");
        std::fs::copy(target_main, &control_main).expect("copy control main");
        std::fs::copy(&donor_wal, wal_path_for(&control_main)).expect("copy control wal");
        let _ = VectorIndex::open(&control_main).expect("control open");
        assert!(
            !wal_path_for(&control_main).exists(),
            "control precondition: the transplanted WAL must be STALE for the target \
             (mutable VectorIndex::open deletes it)"
        );
    }

    /// Required test (i), stale-WAL v1 case — RED on 868c0801: the r1
    /// pre-drain classification routed v1 tiers through the mutable
    /// `VectorIndex::open` (refresh.rs:244-250 there), which DELETES a stale
    /// WAL sidecar during what claims to be classification. r2 classifies
    /// read-only: refusal fires with the canonical directory byte-identical.
    #[test]
    fn classification_never_deletes_a_stale_wal_v1() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("r2-stale-wal-invariance");
            let queue = make_queue(100);
            let (worker, _cache) = make_worker(queue.clone(), &dir, 256);

            submit(&queue, "doc-1", "First document");
            worker.run_cycle(&cx).await.expect("bootstrap cycle");

            let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
            transplant_stale_wal("r2-stale", &fast_path, 256);
            let wal_path = frankensearch_index::wal::wal_path_for(&fast_path);
            let fast_bytes = std::fs::read(&fast_path).expect("read fast");
            let wal_bytes = std::fs::read(&wal_path).expect("read wal");
            let manifest_before = dir_manifest(&dir);

            submit(&queue, "doc-2", "Second document");
            let error = worker
                .run_cycle(&cx)
                .await
                .expect_err("content-retaining v1 must refuse identityless");
            assert_invalid_config(
                &error,
                "refresh.fast_index_identity",
                "identityless-fsvi-v1",
            );

            assert!(
                wal_path.exists(),
                "read-only classification must NEVER delete a stale WAL sidecar"
            );
            assert_eq!(
                std::fs::read(&wal_path).expect("reread wal"),
                wal_bytes,
                "WAL bytes must be untouched by classification"
            );
            assert_eq!(
                std::fs::read(&fast_path).expect("reread fast"),
                fast_bytes,
                "main artifact bytes must be untouched by classification"
            );
            assert_eq!(
                dir_manifest(&dir),
                manifest_before,
                "no file in the canonical directory may change (names, sizes, mtimes)"
            );
            assert_eq!(queue.pending_count(), 1, "refusal fires before drain");
        });
    }

    /// Required test (i), corrupt-trailer v1 case — RED on 868c0801: the
    /// mutable open TRUNCATES a corrupt WAL trailer during classification.
    /// r2 leaves the trailer bytes exactly in place.
    #[test]
    fn classification_never_truncates_a_corrupt_wal_trailer_v1() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("r2-corrupt-trailer-invariance");
            let queue = make_queue(100);
            let (worker, _cache) = make_worker(queue.clone(), &dir, 256);

            submit(&queue, "doc-1", "First document");
            worker.run_cycle(&cx).await.expect("bootstrap cycle");

            let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
            {
                let mut index = VectorIndex::open(&fast_path).expect("fixture open");
                index
                    .append("doc-wal", &normalized(256, 0.55))
                    .expect("append fresh WAL resident");
            }
            let wal_path = frankensearch_index::wal::wal_path_for(&fast_path);
            {
                use std::io::Write as _;
                let mut wal_file = std::fs::OpenOptions::new()
                    .append(true)
                    .open(&wal_path)
                    .expect("open wal for corruption");
                wal_file
                    .write_all(&[0xAB; 32])
                    .expect("append corrupt trailer");
            }
            let wal_bytes = std::fs::read(&wal_path).expect("read wal");
            let manifest_before = dir_manifest(&dir);

            // Control arm: the mutable open truncates this trailer.
            {
                let control_dir = temp_index_dir("r2-corrupt-trailer-control");
                let control_main = control_dir.join("control.idx");
                std::fs::copy(&fast_path, &control_main).expect("copy control main");
                let control_wal = frankensearch_index::wal::wal_path_for(&control_main);
                std::fs::copy(&wal_path, &control_wal).expect("copy control wal");
                let _ = VectorIndex::open(&control_main).expect("control open");
                assert!(
                    std::fs::metadata(&control_wal)
                        .expect("stat control wal")
                        .len()
                        < wal_bytes.len() as u64,
                    "control precondition: mutable open truncates the corrupt trailer"
                );
            }

            submit(&queue, "doc-2", "Second document");
            let error = worker
                .run_cycle(&cx)
                .await
                .expect_err("content-retaining v1 must refuse identityless");
            assert_invalid_config(
                &error,
                "refresh.fast_index_identity",
                "identityless-fsvi-v1",
            );

            assert_eq!(
                std::fs::read(&wal_path).expect("reread wal"),
                wal_bytes,
                "read-only classification must NEVER truncate a corrupt WAL trailer"
            );
            assert_eq!(
                dir_manifest(&dir),
                manifest_before,
                "no file in the canonical directory may change"
            );
            assert_eq!(queue.pending_count(), 1, "refusal fires before drain");
        });
    }

    /// Required test (ii) — RED on 868c0801: a mixed v2-fast + v1-quality
    /// generation must classify (and refuse, on the quality tier) without
    /// mutating EITHER tier. Under r1, classifying the v1 quality tier ran
    /// the mutable open, which deleted its stale WAL before the refusal.
    #[test]
    fn mixed_v2_fast_v1_quality_classifies_without_mutating_either() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let embedder = Arc::new(StubEmbedder::new("v2-stub", V2_DIM));
            let (dir, cache, _binding, fast_path) = v2_canonical_fixture(
                "r2-mixed-generation",
                embedder.identity_bundle(),
                &[("doc-old", normalized(V2_DIM, 0.5))],
                4,
            );

            // Content-retaining v1 quality tier with a stale WAL beside it.
            let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);
            let mut writer =
                VectorIndex::create(&quality_path, "stub-quality", 256).expect("create quality");
            writer
                .write_record("q-doc", &normalized(256, 0.4))
                .expect("write quality row");
            writer.finish().expect("finish quality");
            transplant_stale_wal("r2-mixed", &quality_path, 256);
            let quality_wal = frankensearch_index::wal::wal_path_for(&quality_path);

            let fast_bytes = std::fs::read(&fast_path).expect("read v2 fast");
            let quality_bytes = std::fs::read(&quality_path).expect("read v1 quality");
            let wal_bytes = std::fs::read(&quality_wal).expect("read quality wal");
            let manifest_before = dir_manifest(&dir);

            let queue = make_queue(100);
            submit(&queue, "doc-new", "new document");
            let worker = RefreshWorker::new(
                RefreshWorkerConfig::new(&dir),
                queue.clone(),
                embedder,
                cache,
            );
            let error = worker
                .run_cycle(&cx)
                .await
                .expect_err("mixed generation with content-retaining v1 quality must refuse");
            assert_invalid_config(
                &error,
                "refresh.quality_index_identity",
                "identityless-fsvi-v1",
            );

            assert_eq!(
                std::fs::read(&fast_path).expect("reread v2 fast"),
                fast_bytes,
                "the v2 fast tier must be untouched"
            );
            assert_eq!(
                std::fs::read(&quality_path).expect("reread v1 quality"),
                quality_bytes,
                "the v1 quality tier must be untouched"
            );
            assert!(
                quality_wal.exists(),
                "classification must not delete the v1 quality tier's stale WAL"
            );
            assert_eq!(
                std::fs::read(&quality_wal).expect("reread quality wal"),
                wal_bytes,
                "the quality WAL must be byte-identical"
            );
            assert_eq!(dir_manifest(&dir), manifest_before);
            assert_eq!(queue.pending_count(), 1, "refusal fires before drain");
            let drained = queue.drain_batch();
            assert_eq!(drained[0].retry_count, 0, "no retry budget consumed");
        });
    }

    /// Required test (iii) — RED on 868c0801: a header-valid but
    /// content-corrupt v2 artifact must fail ADMISSION with its own typed
    /// corruption error, not sail past a header-only check to the
    /// composite-authority refusal (which would falsely certify the
    /// generation as fully admitted). Option-A choice: pre-drain performs
    /// full `open_admitted_v2` admission, so the digest recomputation
    /// catches the corruption.
    #[test]
    fn header_valid_content_corrupt_v2_refuses_admission_not_composite() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let embedder = Arc::new(StubEmbedder::new("v2-stub", V2_DIM));
            let (dir, cache, _binding, fast_path) = v2_canonical_fixture(
                "r2-content-corrupt-v2",
                embedder.identity_bundle(),
                &[("doc-old", normalized(V2_DIM, 0.5))],
                6,
            );

            // Flip one byte in the vector slab (the file tail): the header —
            // including its CRC — stays valid, so header-only inspection
            // still reports identity-complete v2.
            let mut bytes = std::fs::read(&fast_path).expect("read canonical");
            let last = bytes.len() - 1;
            bytes[last] ^= 0xFF;
            std::fs::write(&fast_path, &bytes).expect("plant content corruption");

            let queue = make_queue(100);
            submit(&queue, "doc-new", "new document");
            let worker = RefreshWorker::new(
                RefreshWorkerConfig::new(&dir),
                queue.clone(),
                embedder,
                cache,
            );
            let error = worker
                .run_cycle(&cx)
                .await
                .expect_err("content-corrupt v2 must fail admission");

            assert!(
                matches!(error, SearchError::IndexCorrupted { .. }),
                "the refusal must be the admission's own typed corruption error, got {error:?}"
            );
            assert!(
                error.to_string().contains("digest mismatch"),
                "the corruption must be caught by digest recomputation, got: {error}"
            );
            assert!(
                !error.to_string().contains("composite-generation-authority"),
                "a content-corrupt artifact must never reach the composite-authority refusal"
            );
            assert_eq!(queue.pending_count(), 1, "refusal fires before drain");
            let drained = queue.drain_batch();
            assert_eq!(drained[0].retry_count, 0, "no retry budget consumed");
        });
    }

    /// Required test (i), v2 case: the full pre-drain admission (option A)
    /// itself has zero side effects on the canonical directory — names,
    /// sizes, and mtimes are all invariant across the refused cycle, and no
    /// WAL sidecar or staging directory appears.
    #[test]
    fn pre_drain_full_admission_leaves_canonical_directory_invariant() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let embedder = Arc::new(StubEmbedder::new("v2-stub", V2_DIM));
            let (dir, cache, _binding, fast_path) = v2_canonical_fixture(
                "r2-v2-dir-invariance",
                embedder.identity_bundle(),
                &[("doc-old", normalized(V2_DIM, 0.5))],
                3,
            );
            let fast_bytes = std::fs::read(&fast_path).expect("read canonical");
            let manifest_before = dir_manifest(&dir);

            let queue = make_queue(100);
            submit(&queue, "doc-new", "new document");
            let worker = RefreshWorker::new(
                RefreshWorkerConfig::new(&dir),
                queue.clone(),
                embedder,
                cache,
            );
            let error = worker
                .run_cycle(&cx)
                .await
                .expect_err("fully admitted v2 still refuses canonical publication");
            assert_invalid_config(
                &error,
                "refresh.canonical_publication",
                "composite-generation-authority-unavailable",
            );

            assert_eq!(
                std::fs::read(&fast_path).expect("reread canonical"),
                fast_bytes
            );
            assert_eq!(
                dir_manifest(&dir),
                manifest_before,
                "full admission must not change names, sizes, or mtimes"
            );
            assert!(
                !frankensearch_index::wal::wal_path_for(&fast_path).exists(),
                "admission must not materialize a WAL sidecar"
            );
            assert!(!dir.join(STAGED_V2_DIR_NAME).exists());
        });
    }

    /// Required test (iv), integration form — RED on 868c0801 at compile
    /// time: `fast_admitted_owner` does not exist there because the r1
    /// two-tier open peeled `validated.index` and dropped the owner. After
    /// the r2 owner-retention rework, the staged generation retains its
    /// sealed admission owners, and replacing the staged file on disk does
    /// not affect what the retained owner serves.
    #[test]
    fn staged_generation_retains_admission_owners() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let embedder = Arc::new(StubEmbedder::new("v2-stub", V2_DIM));
            let (dir, cache, _binding, _fast_path) = v2_canonical_fixture(
                "r2-staged-owner-retention",
                embedder.identity_bundle(),
                &[("old-1", normalized(V2_DIM, 0.25))],
                7,
            );
            let queue = make_queue(100);
            submit(&queue, "doc-2", "a brand new document");
            let jobs = queue.drain_batch();
            let worker = RefreshWorker::new(
                RefreshWorkerConfig::new(&dir),
                queue.clone(),
                embedder,
                cache,
            );
            let staged = worker
                .stage_identity_bound_generation(&cx, &jobs)
                .await
                .expect("staging must succeed");

            let owner = staged
                .fast_admitted_owner()
                .expect("staged generation must retain its fast admission owner");
            assert!(
                owner.published_wal_absent(),
                "staged pathname admission proves WAL absence into the retained owner"
            );
            assert_eq!(owner.witness().record_count, 2, "old-1 + doc-2");
            let witness_before = owner.witness().clone();
            let hits_before = staged
                .index
                .search_fast(&normalized(V2_DIM, 0.25), 1)
                .expect("search staged");
            assert_eq!(hits_before.len(), 1);

            // Replace the staged file with garbage: the retained owner's
            // Arc'd bytes are the authority, not the pathname.
            std::fs::write(&staged.fast_path, b"garbage-not-an-index")
                .expect("clobber staged file");
            let owner = staged
                .fast_admitted_owner()
                .expect("owner remains retained");
            assert_eq!(
                owner.witness(),
                &witness_before,
                "the witness lives in the sealed owner, not the pathname"
            );
            let hits_after = staged
                .index
                .search_fast(&normalized(V2_DIM, 0.25), 1)
                .expect("search staged after clobber");
            assert_eq!(
                hits_after[0].doc_id, hits_before[0].doc_id,
                "reads must serve the admitted bytes, never the pathname"
            );
        });
    }

    /// Deliberate fail-closed narrowing (r2, documented on the card): an
    /// all-tombstoned v1 tier previously classified as bootstrap-replaceable
    /// because the mutable open could read record flags. Read-only
    /// classification counts record slots conservatively, so this now takes
    /// the identityless refusal instead of silently replacing the artifact.
    /// Flag-level precision returns with the read-only record-table
    /// inspector (observational-open train, index crate root).
    #[test]
    fn all_tombstoned_v1_fails_closed_as_retaining_content() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("r2-all-tombstoned-fail-closed");
            let queue = make_queue(100);
            let (worker, _cache) = make_worker(queue.clone(), &dir, 256);

            submit(&queue, "doc-1", "First document");
            worker.run_cycle(&cx).await.expect("bootstrap cycle");

            let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
            {
                let mut index = VectorIndex::open(&fast_path).expect("fixture open");
                assert!(index.soft_delete("doc-1").expect("tombstone the only row"));
                assert_eq!(index.live_count(), 0);
            }
            let fast_bytes = std::fs::read(&fast_path).expect("read tombstoned artifact");

            submit(&queue, "doc-2", "Second document");
            let error = worker
                .run_cycle(&cx)
                .await
                .expect_err("all-tombstoned v1 fails closed under read-only classification");
            assert_invalid_config(
                &error,
                "refresh.fast_index_identity",
                "identityless-fsvi-v1",
            );
            assert_eq!(
                std::fs::read(&fast_path).expect("reread artifact"),
                fast_bytes,
                "the refused artifact must not be replaced"
            );
            assert_eq!(queue.pending_count(), 1);
        });
    }

    /// Guard for the r2 flow change: classification no longer deletes a
    /// stale WAL beside an EMPTY v1 seed, so the bootstrap rebuild must not
    /// let that leftover sidecar resurrect foreign rows into the new
    /// generation (`TwoTierIndexBuilder::finish` removes sidecars of tiers it
    /// rewrites).
    #[test]
    fn bootstrap_over_empty_seed_with_stale_wal_does_not_resurrect_foreign_rows() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("r2-bootstrap-stale-wal");
            let queue = make_queue(100);
            let (worker, cache) = make_worker(queue.clone(), &dir, 256);

            // Empty v1 seed (from make_worker) + transplanted stale WAL.
            let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
            transplant_stale_wal("r2-bootstrap", &fast_path, 256);

            submit(&queue, "doc-1", "First document");
            let embedded = worker
                .run_cycle(&cx)
                .await
                .expect("empty seed with stale WAL stays bootstrap-replaceable");
            assert_eq!(embedded, 1);

            let current = cache.current();
            let ids: Vec<String> = current.iter_doc_ids().filter_map(Result::ok).collect();
            assert_eq!(ids, vec!["doc-1".to_owned()]);
            assert!(
                !ids.iter().any(|id| id.starts_with("donor-")),
                "no row from the transplanted WAL may leak into the bootstrap: {ids:?}"
            );
            assert!(
                !frankensearch_index::wal::wal_path_for(&fast_path).exists(),
                "the write path must have cleared the dead sidecar during rebuild"
            );
        });
    }
}
