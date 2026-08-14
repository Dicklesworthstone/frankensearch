use std::collections::{BTreeSet, HashMap};
use std::fmt::{self, Write as _};
use std::sync::Arc;
use std::time::Duration;

use compact_str::CompactString;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Document identifier type.
///
/// `CompactString` (SSO) stores ids ≤24 bytes inline, so the per-query
/// `limit_all` `doc_id` clones (RRF/blend/resolve materialization) are a stack
/// memcpy instead of a heap alloc — 29.8× cheaper for short ids
/// (`doc_id_clone_sso` bench). Drop-in for `String` at read sites (`Deref<str>`,
/// `as_str`, `From`, `PartialEq`, `Hash`, `Ord`, serde).
pub type DocId = CompactString;

use crate::SearchError;
use crate::error::SearchResult;
use crate::explanation::HitExplanation;
use crate::generation::{
    EmbeddingIdentityBundleV1, EmbeddingSpaceIdentityV1, EmbeddingSpaceKindV1,
};
use crate::query_class::QueryClass;
use crate::recovery_plan::COMPLETE_COVERAGE_PPM;
use crate::traits::IdentityBoundEmbedding;

// ---------------------------------------------------------------------------
// Document types
// ---------------------------------------------------------------------------

/// A document to be indexed for search.
///
/// This is the input type consumed by both vector indexing and lexical indexing.
/// It intentionally does NOT carry computed data (embeddings, BM25 scores) --
/// those are produced during indexing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexableDocument {
    /// Unique document identifier (caller-defined).
    pub id: String,
    /// Main searchable text content.
    pub content: String,
    /// Optional title (receives BM25 boost in lexical search).
    pub title: Option<String>,
    /// Extensible key-value metadata. Stored alongside results and available
    /// in `ScoredResult.metadata` after search.
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl IndexableDocument {
    /// Creates a new document with the required fields.
    #[must_use]
    pub fn new(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            title: None,
            metadata: HashMap::new(),
        }
    }

    /// Sets the optional title.
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Adds a metadata key-value pair.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Search result types
// ---------------------------------------------------------------------------

/// A raw hit from vector similarity search.
///
/// Produced by the vector index before fusion. Scores are raw cosine similarity
/// values (not normalized), typically in the range \[-1.0, 1.0\].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorHit {
    /// Positional index into the vector store (used for fast lookup).
    pub index: u32,
    /// Raw cosine similarity score.
    pub score: f32,
    /// Document identifier resolved from the index.
    pub doc_id: DocId,
}

impl VectorHit {
    /// Ordering by score descending with NaN-safe semantics.
    /// NaN sorts below all real values (treated as worst possible score).
    #[must_use]
    pub fn cmp_by_score(&self, other: &Self) -> std::cmp::Ordering {
        // Map NaN to NEG_INFINITY so it sorts last in descending order.
        let a = if self.score.is_nan() {
            f32::NEG_INFINITY
        } else {
            self.score
        };
        let b = if other.score.is_nan() {
            f32::NEG_INFINITY
        } else {
            other.score
        };
        // Descending: higher scores first.
        b.total_cmp(&a)
    }

    /// [`Self::cmp_by_score`] refined into a **total order** by breaking ties on `doc_id`.
    ///
    /// Phase-1 score-correction passes (hubness demotion, k-NN smoothing) rewrite scores and must
    /// re-sort, because the rank-based fusion operators take each hit's rank from its *position*
    /// in the slice. Sorting on score alone leaves tied hits in an order that depends on the
    /// upstream pool, so an identical query could fuse differently across runs; the `doc_id`
    /// tiebreak makes the corrected pool replayable.
    ///
    /// Inherits `cmp_by_score`'s NaN-last semantics. That is *not* what a bare
    /// `b.score.total_cmp(&a.score)` gives: IEEE 754 `totalOrder` ranks `+NaN` above `+inf`, so a
    /// descending `total_cmp` would sort a NaN score to rank 0 — exactly the doc a corrupted
    /// correction statistic would produce.
    #[must_use]
    pub fn cmp_rank(&self, other: &Self) -> std::cmp::Ordering {
        self.cmp_by_score(other)
            .then_with(|| self.doc_id.cmp(&other.doc_id))
    }
}

// ---------------------------------------------------------------------------
// Typed per-tier query embeddings (bd-9xuj)
// ---------------------------------------------------------------------------

/// A query embedding bound to the complete identity of the embedding space
/// that produced it.
///
/// Raw `&[f32]` crossing an API boundary is how one vector gets applied to
/// multiple spaces: same dimensions make the bug silently plausible, and
/// different dimensions merely error. A bound embedding carries its
/// [`EmbeddingIdentityBundleV1`], so every consumer can verify the vector
/// belongs to the space it is about to search — by fingerprint, not by
/// dimension coincidence.
#[derive(Clone)]
pub struct BoundQueryEmbedding {
    vector: Vec<f32>,
    identity: EmbeddingIdentityBundleV1,
    identity_fingerprint: String,
    space_fingerprint: String,
}

impl fmt::Debug for BoundQueryEmbedding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BoundQueryEmbedding")
            .field("dimension", &self.vector.len())
            .field("identity_fingerprint", &self.identity_fingerprint)
            .field("space_fingerprint", &self.space_fingerprint)
            .finish_non_exhaustive()
    }
}

impl BoundQueryEmbedding {
    /// Bind a vector to the identity of the space that produced it.
    ///
    /// The identity bundle is validated before anything is bound
    /// ([`EmbeddingIdentityBundleV1::validate`]): a bound embedding is a
    /// claim every downstream verifier trusts, so an incoherent bundle —
    /// component fingerprints that do not bind each other, dimension
    /// contradictions — must be rejected here, not discovered at a seam.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when the identity bundle is
    /// malformed or an in-process `Vec<f32>` is paired with a
    /// non-native/non-f32 storage identity. Returns
    /// [`SearchError::DimensionMismatch`] when the vector length disagrees
    /// with its validated mathematical space.
    pub fn new(vector: Vec<f32>, identity: EmbeddingIdentityBundleV1) -> SearchResult<Self> {
        identity.validate()?;
        let declared_dimension =
            usize::try_from(identity.space.dimension).map_err(|_| SearchError::InvalidConfig {
                field: "bound_query_embedding.dimension".to_owned(),
                value: identity.space.dimension.to_string(),
                reason: "dimension does not fit usize".to_owned(),
            })?;
        if vector.len() != declared_dimension {
            return Err(SearchError::DimensionMismatch {
                expected: declared_dimension,
                found: vector.len(),
            });
        }
        let validated = IdentityBoundEmbedding {
            values: vector,
            identity,
        };
        validated.validate()?;
        let IdentityBoundEmbedding {
            values: vector,
            identity,
        } = validated;
        let identity_fingerprint = identity.fingerprint();
        let space_fingerprint = identity.space.fingerprint();
        Ok(Self {
            vector,
            identity,
            identity_fingerprint,
            space_fingerprint,
        })
    }

    /// The query vector.
    #[must_use]
    pub fn vector(&self) -> &[f32] {
        &self.vector
    }

    /// The complete identity of the producing space.
    #[must_use]
    pub const fn identity(&self) -> &EmbeddingIdentityBundleV1 {
        &self.identity
    }

    /// Lowercase SHA-256 fingerprint of the full identity bundle, computed
    /// once at bind time.
    #[must_use]
    pub fn identity_fingerprint(&self) -> &str {
        &self.identity_fingerprint
    }

    /// Lowercase SHA-256 fingerprint of the *space* component only
    /// ([`EmbeddingIdentityBundleV1::space`]), computed once at bind time.
    ///
    /// This is the join key at every index seam: a query-side bundle binds
    /// in-memory `f32` storage while an index-side bundle binds its
    /// persisted storage format (for example `fsvi-v2`), so their
    /// full-bundle fingerprints legitimately differ even when both were
    /// produced by the same model in the same mathematical space.
    #[must_use]
    pub fn space_fingerprint(&self) -> &str {
        &self.space_fingerprint
    }

    /// Verify this embedding's complete identity bundle — space, producer,
    /// input contract, *and* storage — matches what a consumer expects.
    ///
    /// This is an embedder-to-embedder comparison: use it only when both
    /// sides bind the same storage identity. At an index seam the storage
    /// components legitimately differ (query-side `in-memory-*` versus the
    /// index's persisted format), so full-bundle fingerprints can never
    /// match there — join on [`Self::verify_space_identity`] and admit via
    /// [`Self::verify_producer_conformance`] instead.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] naming `tier` when the
    /// fingerprints differ — including the same-dimension wrong-space case
    /// that raw vector APIs silently accept.
    pub fn verify_space(&self, expected_fingerprint: &str, tier: &str) -> SearchResult<()> {
        if self.identity_fingerprint == expected_fingerprint {
            return Ok(());
        }
        Err(SearchError::InvalidConfig {
            field: format!("query_embedding.{tier}.identity"),
            value: self.identity_fingerprint.clone(),
            reason: format!(
                "query embedding was produced in a different embedding space than the \
                 {tier} index expects (expected identity fingerprint {expected_fingerprint})"
            ),
        })
    }

    /// Join this embedding against an index-side *space* fingerprint — the
    /// index-seam identity check (bd-9xuj T2-C1).
    ///
    /// At an index seam the two sides legitimately disagree on storage
    /// (see [`Self::verify_space`]), so the join key is the space
    /// fingerprint ([`EmbeddingSpaceIdentityV1::fingerprint`]), which binds
    /// model, revision, dimension, and the input contract via
    /// [`EmbeddingSpaceIdentityV1::input_contract_fingerprint`] — but not
    /// the physical storage.
    ///
    /// A matching space fingerprint is **necessary, not sufficient**, for
    /// admission: a bare fingerprint carries no producer attestation, so
    /// this check cannot certify that a *different producer's* vectors are
    /// interchangeable with the index's. Seams that retain the full
    /// expected identity bundle must call
    /// [`Self::verify_producer_conformance`], which applies the complete
    /// bd-9xuj admission law (same producer, or a conformance-certified
    /// foreign producer; anything else rejects).
    ///
    /// # The `LegacyUnidentified` boundary
    ///
    /// `expected_space_fingerprint` must come from an identity-bearing
    /// source: an FSVI v2 header written by `VectorIndex::create_v2`, or an
    /// explicitly supplied space identity. The only production writers that
    /// call `create_v2` today are the two staging call sites in the fusion
    /// crate's `RefreshWorker::stage_identity_bound_generation` (one per
    /// tier), whose output is staged and non-canonical until publication —
    /// `TwoTierIndexBuilder::finish` still routes through the legacy v1
    /// constructors with `identity_v2: None` — so every *canonical*
    /// production index on disk is v1 and has *no* space fingerprint to
    /// pass here. Those artifacts must never reach this verifier: the seam
    /// routes them as typed `LegacyUnidentified`
    /// (`FsviReindexReason::LegacyUnidentified` → `RecoveryPlan` reindex,
    /// then FSVI v2 activation) rather than fabricating a fingerprint,
    /// admitting on dimension equality, or warning and proceeding.
    ///
    /// [`EmbeddingSpaceIdentityV1::fingerprint`]: crate::generation::EmbeddingSpaceIdentityV1::fingerprint
    /// [`EmbeddingSpaceIdentityV1::input_contract_fingerprint`]: crate::generation::EmbeddingSpaceIdentityV1::input_contract_fingerprint
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] naming `tier` when the space
    /// fingerprints differ — including the same-dimension wrong-space case
    /// that raw vector APIs silently accept.
    pub fn verify_space_identity(
        &self,
        expected_space_fingerprint: &str,
        tier: &str,
    ) -> SearchResult<()> {
        if self.space_fingerprint == expected_space_fingerprint {
            return Ok(());
        }
        Err(SearchError::InvalidConfig {
            field: format!("query_embedding.{tier}.space_identity"),
            value: self.space_fingerprint.clone(),
            reason: format!(
                "query embedding was produced in a different embedding space than the \
                 {tier} index expects (expected space fingerprint {expected_space_fingerprint})"
            ),
        })
    }

    /// Apply the complete bd-9xuj admission law against a fully known
    /// expected identity bundle.
    ///
    /// Admission requires the space fingerprints to join (exactly as
    /// [`Self::verify_space_identity`]) **and** producer conformance:
    ///
    /// - the same attested producer →
    ///   [`SpaceIdentityAdmission::SameProducer`];
    /// - a different producer → admitted only when both producers carry the
    ///   byte-identical pinned golden-vector certificate
    ///   ([`crate::generation::GoldenVectorCertificateV1`]), returning
    ///   [`SpaceIdentityAdmission::ConformanceCompatibleProducer`] — typed
    ///   telemetry the caller is expected to record;
    /// - otherwise the pairing is rejected. A matching space fingerprint
    ///   alone never admits a foreign producer.
    ///
    /// # `ConformanceCompatibleProducer` is comparison-grade, NOT trust
    ///
    /// This is a *pairwise* check: it establishes only that the two bundles
    /// AGREE with each other. Mutual agreement does not bind either producer
    /// to any trusted fixture — two bundles carrying the same wrong (or
    /// fabricated) certificate bytes still "agree". Verification against a
    /// pinned trusted corpus is the trunk's certificate flow
    /// ([`crate::generation::GoldenVectorCertificateV1::verify_exact_f32`]
    /// and the witness flow in `generation.rs`), which this method deliberately
    /// does not perform. Callers must therefore treat
    /// `ConformanceCompatibleProducer` as comparison-grade telemetry and
    /// never as an admission basis — the one production caller
    /// (`require_same_producer` in the fusion crate's refresh path) logs the
    /// outcome and refuses it.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`]: propagated from
    /// [`EmbeddingIdentityBundleV1::validate`] when `expected` is not itself
    /// a coherent bundle; naming `query_embedding.<tier>.space_identity`
    /// when the space fingerprints differ; naming
    /// `query_embedding.<tier>.producer_conformance` when the spaces join
    /// but the foreign producer carries no matching golden-vector
    /// certificate.
    pub fn verify_producer_conformance(
        &self,
        expected: &EmbeddingIdentityBundleV1,
        tier: &str,
    ) -> SearchResult<SpaceIdentityAdmission> {
        expected.validate()?;
        self.verify_space_identity(&expected.space.fingerprint(), tier)?;
        let query_producer_fingerprint = self.identity.producer.fingerprint();
        let expected_producer_fingerprint = expected.producer.fingerprint();
        if query_producer_fingerprint == expected_producer_fingerprint {
            return Ok(SpaceIdentityAdmission::SameProducer);
        }
        // Pairwise certified-conformance shape (the bd-9xuj r3
        // `is_conformance_compatible_with` law, inlined against the trunk's
        // generation API, which superseded the pairwise method with the
        // witness/certificate flow): both bundles validate (self at bind
        // time, `expected` above), the spaces join (established above), and
        // both producers carry the byte-identical pinned golden-vector
        // certificate.
        if self.identity.producer.golden_vectors == expected.producer.golden_vectors {
            return Ok(SpaceIdentityAdmission::ConformanceCompatibleProducer {
                query_producer_fingerprint,
                expected_producer_fingerprint,
            });
        }
        Err(SearchError::InvalidConfig {
            field: format!("query_embedding.{tier}.producer_conformance"),
            value: query_producer_fingerprint,
            reason: format!(
                "query embedding's producer shares the {tier} index's embedding space but \
                 is not certified conformance-compatible with its producer (expected \
                 producer fingerprint {expected_producer_fingerprint}); a matching space \
                 fingerprint alone never admits a foreign producer — admission requires \
                 the identical pinned golden-vector certificate"
            ),
        })
    }
}

/// Typed outcome of the bd-9xuj admission law for a space-verified pairing.
///
/// Emitted by [`BoundQueryEmbedding::verify_producer_conformance`] so callers
/// can log and route the admission basis instead of collapsing it into a bare
/// `Ok`: a foreign producer admitted through its golden-vector certificate is
/// telemetry-visible by construction.
///
/// `#[non_exhaustive]` (review #8151): the different-corpus-digest case
/// today rejects only vacuously through the single reject arm — nothing
/// distinguishes "certificates pinned to different corpora, so conformance
/// is unattestable" from "same corpus, conformance failed". A future typed
/// `Unattestable` classification is expected as a third arm; marking the
/// enum non-exhaustive now means adding it will not be a semver break on
/// downstream matches.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "admission")]
#[non_exhaustive]
pub enum SpaceIdentityAdmission {
    /// Same space and the same attested producer.
    SameProducer,
    /// Same space, different producer, admitted because both producers carry
    /// the identical pinned golden-vector certificate
    /// ([`crate::generation::GoldenVectorCertificateV1`]).
    ConformanceCompatibleProducer {
        /// Producer fingerprint of the query-side bundle.
        query_producer_fingerprint: String,
        /// Producer fingerprint the consumer expected.
        expected_producer_fingerprint: String,
    },
}

impl SpaceIdentityAdmission {
    /// Stable `snake_case` code for logs and machine payloads.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::SameProducer => "same_producer",
            Self::ConformanceCompatibleProducer { .. } => "conformance_compatible_producer",
        }
    }
}

/// Per-tier query embeddings, each independently bound to its own space.
///
/// Replaces raw cross-tier `&[f32]` parameters (bd-9xuj): the fast and
/// quality tiers are different embedding spaces, so a search that consults
/// both must carry one bound embedding per tier. At least one tier is
/// always present — a tier-less value is unrepresentable.
#[derive(Clone)]
pub struct TieredQueryEmbeddings {
    fast: Option<BoundQueryEmbedding>,
    quality: Option<BoundQueryEmbedding>,
}

impl fmt::Debug for TieredQueryEmbeddings {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TieredQueryEmbeddings")
            .field("fast", &self.fast)
            .field("quality", &self.quality)
            .finish()
    }
}

impl TieredQueryEmbeddings {
    /// Embeddings for both tiers (full progressive search).
    #[must_use]
    pub const fn progressive(fast: BoundQueryEmbedding, quality: BoundQueryEmbedding) -> Self {
        Self {
            fast: Some(fast),
            quality: Some(quality),
        }
    }

    /// Fast-tier only (quality unavailable or deliberately skipped).
    #[must_use]
    pub const fn fast_only(fast: BoundQueryEmbedding) -> Self {
        Self {
            fast: Some(fast),
            quality: None,
        }
    }

    /// Quality-tier only: the quality index is the primary retrieval arm,
    /// never a rescoring pass over a fast-selected pool.
    #[must_use]
    pub const fn quality_only(quality: BoundQueryEmbedding) -> Self {
        Self {
            fast: None,
            quality: Some(quality),
        }
    }

    /// The fast-tier embedding, when present.
    #[must_use]
    pub const fn fast(&self) -> Option<&BoundQueryEmbedding> {
        self.fast.as_ref()
    }

    /// The quality-tier embedding, when present.
    #[must_use]
    pub const fn quality(&self) -> Option<&BoundQueryEmbedding> {
        self.quality.as_ref()
    }

    /// The topology these embeddings can support on their own (before
    /// index/coverage constraints narrow it).
    ///
    /// This is a *request-shape* helper derived from which tiers carry a
    /// bound embedding. It is deliberately NOT authority-bearing and must
    /// never be recorded as a realized topology: the only authority for that
    /// is [`TierCoveragePairV1::derive_topology`], which additionally
    /// validates bindings, per-tier coverage, and intent. Use this to answer
    /// "what could these embeddings ask for", never "what did the search do".
    ///
    /// # A hash-control space can never ask for a semantic topology
    ///
    /// If ANY bound tier was produced in an
    /// [`EmbeddingSpaceKindV1::HashControl`] space, the answer is
    /// [`RetrievalTopology::HashControl`] — never `FastOnly`, `QualityOnly`
    /// or `FullProgressive` (bd-ctzo C4). Those three names assert semantic
    /// retrieval; a deterministic control lane produces vectors whose
    /// neighbourhoods carry no learned meaning, so reporting one of them for
    /// a hash query is the "hash-as-semantic" claim this bead forbids. The
    /// r1 revision of this method looked only at which tiers were `Some`,
    /// which meant every hash-control fixture in the tree reported
    /// `FullProgressive` and looked like semantic availability.
    ///
    /// A MIXED pair — one semantic tier, one hash tier — also answers
    /// `HashControl`. It is the weaker of the two claims, and a topology is a
    /// statement about the whole retrieval, not about its best arm.
    ///
    /// [`EmbeddingSpaceKindV1::HashControl`]: crate::generation::EmbeddingSpaceKindV1::HashControl
    #[must_use]
    pub const fn supported_topology(&self) -> RetrievalTopology {
        if self.binds_hash_control() {
            return RetrievalTopology::HashControl;
        }
        match (&self.fast, &self.quality) {
            (Some(_), Some(_)) => RetrievalTopology::FullProgressive,
            (Some(_), None) => RetrievalTopology::FastOnly,
            (None, Some(_)) => RetrievalTopology::QualityOnly,
            // Unreachable by construction; lexical-only is the honest floor.
            (None, None) => RetrievalTopology::LexicalOnly,
        }
    }

    /// Whether any bound tier was produced in a non-semantic control space.
    #[must_use]
    pub const fn binds_hash_control(&self) -> bool {
        const fn is_hash(embedding: Option<&BoundQueryEmbedding>) -> bool {
            match embedding {
                Some(bound) => matches!(
                    bound.identity.space.kind,
                    crate::generation::EmbeddingSpaceKindV1::HashControl
                ),
                None => false,
            }
        }
        is_hash(self.fast.as_ref()) || is_hash(self.quality.as_ref())
    }
}

// ---------------------------------------------------------------------------
// Per-query coverage reconstructed from retained owner witnesses (bd-ctzo C4)
// ---------------------------------------------------------------------------

/// What one tier contributed to ONE query.
///
/// Every populated field is read from the retained owner's own witness or
/// counted off the candidates the search actually returned. There is no
/// constructor that accepts a coverage scalar, because a caller's claim about
/// how much of the corpus it searched is not evidence of anything.
///
/// # Unknown is not zero
///
/// The `Unknown` variant carries NO counts. That is the point: a legacy
/// artifact retains no witness, and reporting its coverage as `0` would be
/// indistinguishable from a live tier that genuinely matched nothing — the
/// exact conflation bd-ctzo C4 forbids. A consumer that wants a number must
/// first destructure `Witnessed`, which means it cannot accidentally average
/// an unknown into a total.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum TierQueryCoverageV1 {
    /// The query bound no embedding for this tier, so it was never consulted.
    NotRequested,
    /// The query bound this tier, but no owner-backed witness exists for it.
    Unknown {
        /// Closed reason no witness was available.
        reason: CoverageUnknownReasonV1,
    },
    /// The tier was served by a retained admitted owner.
    Witnessed {
        /// Generation sequence the owner witnesses. Not a caller's claim.
        generation_sequence: u64,
        /// Live document count the owner witnesses.
        live_count: u64,
        /// How many of the returned candidates this tier actually produced,
        /// counted against the results, not predicted from the request.
        contributed_candidates: u64,
    },
}

impl TierQueryCoverageV1 {
    /// Stable `snake_case` state code for logs and machine payloads.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::NotRequested => "not_requested",
            Self::Unknown { .. } => "unknown",
            Self::Witnessed { .. } => "witnessed",
        }
    }

    /// The witnessed live count, or `None` when this tier has no witness.
    ///
    /// Deliberately `Option<u64>` rather than `u64`: see the type-level note
    /// on why unknown must not collapse to zero.
    #[must_use]
    pub const fn witnessed_live_count(&self) -> Option<u64> {
        match self {
            Self::Witnessed { live_count, .. } => Some(*live_count),
            Self::NotRequested | Self::Unknown { .. } => None,
        }
    }
}

/// Coverage and contribution for one complete tiered query (bd-ctzo C4).
///
/// Built by the index layer from an activated search plus the candidates it
/// returned. Serialized form is stable and bounded: it carries counts,
/// generation sequences and closed enum codes only — never vectors, query
/// text, document ids, tokens, secrets or paths.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchCoverageV1 {
    /// Schema version of this receipt.
    pub schema_version: u16,
    /// Topology the bound embeddings could support — already hash-aware, so
    /// a control-lane query cannot be recorded here as semantic.
    pub topology: RetrievalTopology,
    /// Fast-tier coverage.
    pub fast: TierQueryCoverageV1,
    /// Quality-tier coverage.
    pub quality: TierQueryCoverageV1,
}

impl SearchCoverageV1 {
    /// Assemble a per-query coverage receipt.
    ///
    /// Intentionally has no `coverage_ratio`, `percent` or `total` parameter:
    /// everything here is either the topology the bindings support or a fact
    /// one of the two tier witnesses already established.
    #[must_use]
    pub const fn new(
        topology: RetrievalTopology,
        fast: TierQueryCoverageV1,
        quality: TierQueryCoverageV1,
    ) -> Self {
        Self {
            schema_version: TIER_COVERAGE_SCHEMA_VERSION_V1,
            topology,
            fast,
            quality,
        }
    }

    /// Whether this query ran on a non-semantic control lane.
    #[must_use]
    pub const fn is_hash_control(&self) -> bool {
        matches!(self.topology, RetrievalTopology::HashControl)
    }

    /// A bounded one-line summary safe to put in a log.
    ///
    /// Contains only closed state codes, generation sequences and counts. No
    /// vector, query, document id, token, secret or path can reach it: the
    /// type holds none of those to begin with, which is a stronger guarantee
    /// than remembering to redact at each call site.
    #[must_use]
    pub fn redacted_summary(&self) -> String {
        fn tier(label: &str, coverage: &TierQueryCoverageV1) -> String {
            match coverage {
                TierQueryCoverageV1::Witnessed {
                    generation_sequence,
                    live_count,
                    contributed_candidates,
                } => format!(
                    "{label}=witnessed(gen={generation_sequence},live={live_count},\
                     contributed={contributed_candidates})"
                ),
                TierQueryCoverageV1::Unknown { reason } => {
                    format!("{label}=unknown({reason:?})")
                }
                TierQueryCoverageV1::NotRequested => format!("{label}=not_requested"),
            }
        }
        format!(
            "topology={:?} {} {}",
            self.topology,
            tier("fast", &self.fast),
            tier("quality", &self.quality)
        )
    }
}

// ---------------------------------------------------------------------------
// Independent requested/realized per-tier coverage (bd-9xuj C1 AC3/AC4)
// ---------------------------------------------------------------------------

/// Schema version for the closed per-tier coverage and topology receipts.
pub const TIER_COVERAGE_SCHEMA_VERSION_V1: u16 = 1;
const MAX_COVERAGE_MEMBER_BYTES: usize = 4_096;

/// Semantic role of one independently witnessed retrieval tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoverageTierV1 {
    /// Latency-oriented embedding/index tier.
    Fast,
    /// Quality-oriented embedding/index tier.
    Quality,
}

impl CoverageTierV1 {
    const fn code(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Quality => "quality",
        }
    }
}

/// Canonical member representation used to witness one tier's coverage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoverageBasisV1 {
    /// An order-independent, duplicate-free canonical document-id set.
    CanonicalDocuments,
    /// A duplicate-free shard-id sequence whose order is identity material.
    OrderedShards,
}

impl CoverageBasisV1 {
    const fn code(self) -> &'static str {
        match self {
            Self::CanonicalDocuments => "canonical_documents",
            Self::OrderedShards => "ordered_shards",
        }
    }
}

/// Exact set relationship between realized indexed members and requested live members.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoverageRelationV1 {
    /// Both the requested live set and realized indexed set are empty.
    ZeroLive,
    /// Realized and requested sets are exactly equal and non-empty.
    Complete,
    /// The realized set is a proper subset of the requested live set.
    Subset,
    /// The realized set is a proper superset and therefore contains stale/foreign members.
    Superset,
    /// Both sides have unique members and a non-empty proper intersection.
    Overlap,
    /// Both sides are non-empty and have no member in common.
    Disjoint,
}

impl CoverageRelationV1 {
    const fn code(self) -> &'static str {
        match self {
            Self::ZeroLive => "zero_live",
            Self::Complete => "complete",
            Self::Subset => "subset",
            Self::Superset => "superset",
            Self::Overlap => "overlap",
            Self::Disjoint => "disjoint",
        }
    }
}

/// Closed reason why realized coverage is not known.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoverageUnknownReasonV1 {
    /// No immutable owner is available to make a coverage statement.
    OwnerUnavailable,
    /// A legacy artifact has no admissible identity or coverage witness.
    LegacyUnidentified,
    /// The active generation cannot be resolved atomically.
    GenerationUnresolved,
    /// The canonical live corpus cannot be resolved.
    CorpusUnresolved,
}

/// Closed reason why observed coverage facts are present but unverified.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoverageUnverifiedReasonV1 {
    /// No canonical member witness accompanies the observed count.
    WitnessMissing,
    /// A supplied witness digest did not verify against retained authority.
    WitnessDigestMismatch,
    /// Counts were observed through a source that cannot attest uniqueness.
    CountsUntrusted,
    /// Different generation/corpus scopes have no explicit comparison evidence.
    CrossScopeUnproven,
}

/// Immutable generation and corpus facts that scope one member witness.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct CoverageScopeV1 {
    generation_fingerprint: String,
    corpus_fingerprint: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CoverageScopeWireV1 {
    generation_fingerprint: String,
    corpus_fingerprint: String,
}

impl<'de> Deserialize<'de> for CoverageScopeV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = CoverageScopeWireV1::deserialize(deserializer)?;
        Self::new(wire.generation_fingerprint, wire.corpus_fingerprint)
            .map_err(serde::de::Error::custom)
    }
}

impl CoverageScopeV1 {
    /// Construct a scope from exact lowercase SHA-256 generation and corpus receipts.
    ///
    /// # Errors
    ///
    /// Rejects malformed or non-canonical fingerprints.
    pub fn new(
        generation_fingerprint: impl Into<String>,
        corpus_fingerprint: impl Into<String>,
    ) -> SearchResult<Self> {
        let scope = Self {
            generation_fingerprint: generation_fingerprint.into(),
            corpus_fingerprint: corpus_fingerprint.into(),
        };
        scope.validate()?;
        Ok(scope)
    }

    fn validate(&self) -> SearchResult<()> {
        validate_coverage_sha256(
            "coverage.scope.generation_fingerprint",
            &self.generation_fingerprint,
        )?;
        validate_coverage_sha256(
            "coverage.scope.corpus_fingerprint",
            &self.corpus_fingerprint,
        )
    }

    /// Exact generation receipt.
    #[must_use]
    pub fn generation_fingerprint(&self) -> &str {
        &self.generation_fingerprint
    }

    /// Exact corpus receipt.
    #[must_use]
    pub fn corpus_fingerprint(&self) -> &str {
        &self.corpus_fingerprint
    }

    /// Domain-separated stable fingerprint of both scope facts.
    #[must_use]
    pub fn fingerprint(&self) -> String {
        let mut encoder = CoverageEncoder::new(b"frankensearch.coverage-scope.v1");
        encoder.text(&self.generation_fingerprint);
        encoder.text(&self.corpus_fingerprint);
        encoder.fingerprint()
    }
}

/// Bounded, tier-tagged witness of a canonical document set or ordered shard sequence.
///
/// Only digests and counts are retained: query/document text and raw member identifiers never
/// enter logs or serialized diagnostics. The tier role participates in [`Self::fingerprint`], so
/// a fast witness cannot be borrowed as quality evidence even when both cover the same members.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct CoverageWitnessV1 {
    schema_version: u16,
    tier: CoverageTierV1,
    space_fingerprint: String,
    space_kind: EmbeddingSpaceKindV1,
    basis: CoverageBasisV1,
    scope: CoverageScopeV1,
    member_count: u64,
    membership_fingerprint: String,
    set_fingerprint: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CoverageWitnessWireV1 {
    schema_version: u16,
    tier: CoverageTierV1,
    space_fingerprint: String,
    space_kind: EmbeddingSpaceKindV1,
    basis: CoverageBasisV1,
    scope: CoverageScopeV1,
    member_count: u64,
    membership_fingerprint: String,
    set_fingerprint: String,
}

impl CoverageWitnessWireV1 {
    fn into_candidate(self) -> SearchResult<CoverageWitnessV1> {
        let witness = CoverageWitnessV1 {
            schema_version: self.schema_version,
            tier: self.tier,
            space_fingerprint: self.space_fingerprint,
            space_kind: self.space_kind,
            basis: self.basis,
            scope: self.scope,
            member_count: self.member_count,
            membership_fingerprint: self.membership_fingerprint,
            set_fingerprint: self.set_fingerprint,
        };
        witness.validate()?;
        Ok(witness)
    }
}

impl CoverageWitnessV1 {
    fn validate(&self) -> SearchResult<()> {
        validate_coverage_schema("coverage.witness.schema_version", self.schema_version)?;
        self.scope.validate()?;
        validate_coverage_sha256(
            "coverage.witness.space_fingerprint",
            &self.space_fingerprint,
        )?;
        validate_coverage_sha256(
            "coverage.witness.membership_fingerprint",
            &self.membership_fingerprint,
        )?;
        validate_coverage_sha256("coverage.witness.set_fingerprint", &self.set_fingerprint)
    }

    /// Tier role cryptographically bound by this witness.
    #[must_use]
    pub const fn tier(&self) -> CoverageTierV1 {
        self.tier
    }

    /// Validated mathematical-space fingerprint that produced this tier.
    #[must_use]
    pub fn space_fingerprint(&self) -> &str {
        &self.space_fingerprint
    }

    /// Explicit semantic versus hash-control classification of that space.
    #[must_use]
    pub const fn space_kind(&self) -> EmbeddingSpaceKindV1 {
        self.space_kind
    }

    /// Member representation bound by this witness.
    #[must_use]
    pub const fn basis(&self) -> CoverageBasisV1 {
        self.basis
    }

    /// Immutable generation/corpus scope.
    #[must_use]
    pub const fn scope(&self) -> &CoverageScopeV1 {
        &self.scope
    }

    /// Duplicate-free member count derived at construction.
    #[must_use]
    pub const fn member_count(&self) -> u64 {
        self.member_count
    }

    /// Order-sensitive membership digest (document ids are canonically sorted first).
    #[must_use]
    pub fn membership_fingerprint(&self) -> &str {
        &self.membership_fingerprint
    }

    /// Order-independent set digest used for checked relation evidence.
    #[must_use]
    pub fn set_fingerprint(&self) -> &str {
        &self.set_fingerprint
    }

    /// Domain-separated digest of every semantic witness field, including tier role.
    #[must_use]
    pub fn fingerprint(&self) -> String {
        let mut encoder = CoverageEncoder::new(b"frankensearch.coverage-witness.v1");
        encoder.u16(self.schema_version);
        encoder.text(self.tier.code());
        encoder.text(&self.space_fingerprint);
        encoder.text(embedding_space_kind_code(self.space_kind));
        encoder.text(self.basis.code());
        encoder.text(&self.scope.fingerprint());
        encoder.u64(self.member_count);
        encoder.text(&self.membership_fingerprint);
        encoder.text(&self.set_fingerprint);
        encoder.fingerprint()
    }
}

struct PreparedCoverageMembers {
    witness: CoverageWitnessV1,
    members: BTreeSet<String>,
}

fn prepare_coverage_members(
    tier: CoverageTierV1,
    space: &EmbeddingSpaceIdentityV1,
    basis: CoverageBasisV1,
    scope: CoverageScopeV1,
    members: Vec<String>,
) -> SearchResult<PreparedCoverageMembers> {
    space.validate()?;
    scope.validate()?;
    let member_count = u64::try_from(members.len()).map_err(|_| {
        coverage_error(
            "coverage.member_count",
            "redacted-overflow",
            "member count does not fit in u64",
        )
    })?;
    let mut membership_encoder = match basis {
        CoverageBasisV1::CanonicalDocuments => {
            CoverageEncoder::new(b"frankensearch.coverage-canonical-documents.v1")
        }
        CoverageBasisV1::OrderedShards => {
            CoverageEncoder::new(b"frankensearch.coverage-ordered-shards.v1")
        }
    };
    membership_encoder.u64(member_count);
    let mut unique = BTreeSet::new();
    for (index, member) in members.into_iter().enumerate() {
        if member.is_empty() || member.len() > MAX_COVERAGE_MEMBER_BYTES {
            return Err(coverage_error(
                "coverage.members",
                &format!("redacted-member-at-index-{index}"),
                "member identifiers must be non-empty and at most 4096 bytes",
            ));
        }
        if basis == CoverageBasisV1::OrderedShards {
            membership_encoder.text(&member);
        }
        if !unique.insert(member) {
            return Err(coverage_error(
                "coverage.members",
                &format!("redacted-duplicate-at-index-{index}"),
                "duplicate member identifiers are forbidden because they inflate coverage",
            ));
        }
    }

    let mut set_encoder = CoverageEncoder::new(b"frankensearch.coverage-member-set.v1");
    set_encoder.u64(member_count);
    for member in &unique {
        set_encoder.text(member);
    }
    let set_fingerprint = set_encoder.fingerprint();

    if basis == CoverageBasisV1::CanonicalDocuments {
        for member in &unique {
            membership_encoder.text(member);
        }
    }

    Ok(PreparedCoverageMembers {
        witness: CoverageWitnessV1 {
            schema_version: TIER_COVERAGE_SCHEMA_VERSION_V1,
            tier,
            space_fingerprint: space.fingerprint(),
            space_kind: space.kind,
            basis,
            scope,
            member_count,
            membership_fingerprint: membership_encoder.fingerprint(),
            set_fingerprint,
        },
        members: unique,
    })
}

/// Checked scalar facts derived from two duplicate-free member sets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct CoverageCountsV1 {
    live_count: u64,
    indexed_count: u64,
    intersection_count: u64,
    union_count: u64,
    coverage_ppm: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CoverageCountsWireV1 {
    live_count: u64,
    indexed_count: u64,
    intersection_count: u64,
    union_count: u64,
    coverage_ppm: u32,
}

impl<'de> Deserialize<'de> for CoverageCountsV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = CoverageCountsWireV1::deserialize(deserializer)?;
        Self::checked(
            wire.live_count,
            wire.indexed_count,
            wire.intersection_count,
            wire.union_count,
        )
        .and_then(|counts| {
            if counts.coverage_ppm == wire.coverage_ppm {
                Ok(counts)
            } else {
                Err(coverage_error(
                    "coverage.counts.coverage_ppm",
                    &wire.coverage_ppm.to_string(),
                    "must be derived from intersection_count/live_count",
                ))
            }
        })
        .map_err(serde::de::Error::custom)
    }
}

impl CoverageCountsV1 {
    fn checked(
        live_count: u64,
        indexed_count: u64,
        intersection_count: u64,
        union_count: u64,
    ) -> SearchResult<Self> {
        if intersection_count > live_count || intersection_count > indexed_count {
            return Err(coverage_error(
                "coverage.counts.intersection_count",
                &intersection_count.to_string(),
                "cannot exceed either live_count or indexed_count",
            ));
        }
        let expected_union = u128::from(live_count)
            .checked_add(u128::from(indexed_count))
            .and_then(|sum| sum.checked_sub(u128::from(intersection_count)))
            .ok_or_else(|| {
                coverage_error(
                    "coverage.counts.union_count",
                    "redacted-overflow",
                    "checked set-union arithmetic overflowed",
                )
            })?;
        let expected_union = u64::try_from(expected_union).map_err(|_| {
            coverage_error(
                "coverage.counts.union_count",
                "redacted-overflow",
                "set union cannot be represented by u64",
            )
        })?;
        if union_count != expected_union {
            return Err(coverage_error(
                "coverage.counts.union_count",
                &union_count.to_string(),
                "must equal live_count + indexed_count - intersection_count",
            ));
        }
        let coverage_ppm = if live_count == 0 {
            0
        } else {
            let scaled = u128::from(intersection_count) * u128::from(COMPLETE_COVERAGE_PPM);
            u32::try_from(scaled / u128::from(live_count)).map_err(|_| {
                coverage_error(
                    "coverage.counts.coverage_ppm",
                    "redacted-overflow",
                    "derived fixed-point coverage does not fit in u32",
                )
            })?
        };
        Ok(Self {
            live_count,
            indexed_count,
            intersection_count,
            union_count,
            coverage_ppm,
        })
    }

    fn from_member_sets(live: &BTreeSet<String>, indexed: &BTreeSet<String>) -> SearchResult<Self> {
        let live_count = u64::try_from(live.len()).map_err(|_| {
            coverage_error(
                "coverage.counts.live_count",
                "redacted-overflow",
                "live member count does not fit in u64",
            )
        })?;
        let indexed_count = u64::try_from(indexed.len()).map_err(|_| {
            coverage_error(
                "coverage.counts.indexed_count",
                "redacted-overflow",
                "indexed member count does not fit in u64",
            )
        })?;
        let intersection_count =
            u64::try_from(live.intersection(indexed).count()).map_err(|_| {
                coverage_error(
                    "coverage.counts.intersection_count",
                    "redacted-overflow",
                    "intersection member count does not fit in u64",
                )
            })?;
        let union_count = u64::try_from(live.union(indexed).count()).map_err(|_| {
            coverage_error(
                "coverage.counts.union_count",
                "redacted-overflow",
                "union member count does not fit in u64",
            )
        })?;
        Self::checked(live_count, indexed_count, intersection_count, union_count)
    }

    /// Canonical live-member count.
    #[must_use]
    pub const fn live_count(self) -> u64 {
        self.live_count
    }

    /// Canonical indexed-member count.
    #[must_use]
    pub const fn indexed_count(self) -> u64 {
        self.indexed_count
    }

    /// Checked intersection count.
    #[must_use]
    pub const fn intersection_count(self) -> u64 {
        self.intersection_count
    }

    /// Checked union count.
    #[must_use]
    pub const fn union_count(self) -> u64 {
        self.union_count
    }

    /// Floor of `intersection/live * 1_000_000`, derived with u128 intermediates.
    #[must_use]
    pub const fn coverage_ppm(self) -> u32 {
        self.coverage_ppm
    }
}

/// Explicit comparison evidence required when requested and realized scopes differ.
///
/// The evidence binds both scopes, both exact set digests, the computed intersection and union,
/// and a retained authority fingerprint. It is never inferred from counts or another tier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct CrossScopeCoverageEvidenceV1 {
    schema_version: u16,
    live_witness_fingerprint: String,
    indexed_witness_fingerprint: String,
    live_scope_fingerprint: String,
    indexed_scope_fingerprint: String,
    live_set_fingerprint: String,
    indexed_set_fingerprint: String,
    intersection_set_fingerprint: String,
    union_set_fingerprint: String,
    authority_fingerprint: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CrossScopeCoverageEvidenceWireV1 {
    schema_version: u16,
    live_witness_fingerprint: String,
    indexed_witness_fingerprint: String,
    live_scope_fingerprint: String,
    indexed_scope_fingerprint: String,
    live_set_fingerprint: String,
    indexed_set_fingerprint: String,
    intersection_set_fingerprint: String,
    union_set_fingerprint: String,
    authority_fingerprint: String,
}

impl CrossScopeCoverageEvidenceWireV1 {
    fn into_candidate(self) -> SearchResult<CrossScopeCoverageEvidenceV1> {
        let evidence = CrossScopeCoverageEvidenceV1 {
            schema_version: self.schema_version,
            live_witness_fingerprint: self.live_witness_fingerprint,
            indexed_witness_fingerprint: self.indexed_witness_fingerprint,
            live_scope_fingerprint: self.live_scope_fingerprint,
            indexed_scope_fingerprint: self.indexed_scope_fingerprint,
            live_set_fingerprint: self.live_set_fingerprint,
            indexed_set_fingerprint: self.indexed_set_fingerprint,
            intersection_set_fingerprint: self.intersection_set_fingerprint,
            union_set_fingerprint: self.union_set_fingerprint,
            authority_fingerprint: self.authority_fingerprint,
        };
        evidence.validate()?;
        Ok(evidence)
    }
}

impl CrossScopeCoverageEvidenceV1 {
    fn build(
        live: &PreparedCoverageMembers,
        indexed: &PreparedCoverageMembers,
        authority_fingerprint: String,
    ) -> SearchResult<Self> {
        validate_coverage_sha256(
            "coverage.cross_scope.authority_fingerprint",
            &authority_fingerprint,
        )?;
        let intersection = live
            .members
            .intersection(&indexed.members)
            .cloned()
            .collect::<BTreeSet<_>>();
        let union = live
            .members
            .union(&indexed.members)
            .cloned()
            .collect::<BTreeSet<_>>();
        let evidence = Self {
            schema_version: TIER_COVERAGE_SCHEMA_VERSION_V1,
            live_witness_fingerprint: live.witness.fingerprint(),
            indexed_witness_fingerprint: indexed.witness.fingerprint(),
            live_scope_fingerprint: live.witness.scope.fingerprint(),
            indexed_scope_fingerprint: indexed.witness.scope.fingerprint(),
            live_set_fingerprint: live.witness.set_fingerprint.clone(),
            indexed_set_fingerprint: indexed.witness.set_fingerprint.clone(),
            intersection_set_fingerprint: fingerprint_member_set(
                b"frankensearch.coverage-intersection-set.v1",
                &intersection,
            )?,
            union_set_fingerprint: fingerprint_member_set(
                b"frankensearch.coverage-union-set.v1",
                &union,
            )?,
            authority_fingerprint,
        };
        evidence.validate()?;
        Ok(evidence)
    }

    fn validate(&self) -> SearchResult<()> {
        validate_coverage_schema("coverage.cross_scope.schema_version", self.schema_version)?;
        for (field, value) in [
            ("live_witness_fingerprint", &self.live_witness_fingerprint),
            (
                "indexed_witness_fingerprint",
                &self.indexed_witness_fingerprint,
            ),
            ("live_scope_fingerprint", &self.live_scope_fingerprint),
            ("indexed_scope_fingerprint", &self.indexed_scope_fingerprint),
            ("live_set_fingerprint", &self.live_set_fingerprint),
            ("indexed_set_fingerprint", &self.indexed_set_fingerprint),
            (
                "intersection_set_fingerprint",
                &self.intersection_set_fingerprint,
            ),
            ("union_set_fingerprint", &self.union_set_fingerprint),
            ("authority_fingerprint", &self.authority_fingerprint),
        ] {
            validate_coverage_sha256(&format!("coverage.cross_scope.{field}"), value)?;
        }
        if self.live_scope_fingerprint == self.indexed_scope_fingerprint {
            return Err(coverage_error(
                "coverage.cross_scope",
                "redundant-same-scope-evidence",
                "cross-scope evidence is valid only when generation or corpus differs",
            ));
        }
        Ok(())
    }

    /// Domain-separated fingerprint of the complete cross-scope comparison receipt.
    #[must_use]
    pub fn fingerprint(&self) -> String {
        let mut encoder = CoverageEncoder::new(b"frankensearch.cross-scope-coverage.v1");
        encoder.u16(self.schema_version);
        encoder.text(&self.live_witness_fingerprint);
        encoder.text(&self.indexed_witness_fingerprint);
        encoder.text(&self.live_scope_fingerprint);
        encoder.text(&self.indexed_scope_fingerprint);
        encoder.text(&self.live_set_fingerprint);
        encoder.text(&self.indexed_set_fingerprint);
        encoder.text(&self.intersection_set_fingerprint);
        encoder.text(&self.union_set_fingerprint);
        encoder.text(&self.authority_fingerprint);
        encoder.fingerprint()
    }
}

/// Requested side of one tier's independent coverage contract.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum TierCoverageRequestV1 {
    /// This tier was not requested; it carries no borrowed witness.
    MissingTier,
    /// The exact live member set requested for this tier.
    Requested {
        /// Tier-tagged live member witness.
        live: CoverageWitnessV1,
    },
}

#[derive(Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
enum TierCoverageRequestWireV1 {
    MissingTier,
    Requested { live: CoverageWitnessWireV1 },
}

impl TierCoverageRequestWireV1 {
    fn into_candidate(self) -> SearchResult<TierCoverageRequestV1> {
        match self {
            Self::MissingTier => Ok(TierCoverageRequestV1::MissingTier),
            Self::Requested { live } => Ok(TierCoverageRequestV1::Requested {
                live: live.into_candidate()?,
            }),
        }
    }
}

/// Realized side of one tier's independent coverage contract.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum TierCoverageRealizationV1 {
    /// No realized tier/index exists.
    MissingTier,
    /// Coverage could not be observed from an immutable authority.
    Unknown {
        /// Closed unknown-state reason.
        reason: CoverageUnknownReasonV1,
    },
    /// A scalar observation exists, but no verified unique-member witness does.
    Unverified {
        /// Closed verification failure reason.
        reason: CoverageUnverifiedReasonV1,
        /// Optional observed count; never used to derive semantic availability.
        observed_indexed_count: Option<u64>,
    },
    /// Coverage was derived from two duplicate-free member sets.
    Verified {
        /// Tier-tagged indexed member witness.
        indexed: Box<CoverageWitnessV1>,
        /// Derived exact set relation.
        relation: CoverageRelationV1,
        /// Derived checked counts and fixed-point ratio.
        counts: CoverageCountsV1,
        /// Required and fully bound only when generation/corpus scopes differ.
        cross_scope_evidence: Option<Box<CrossScopeCoverageEvidenceV1>>,
    },
}

#[derive(Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
enum TierCoverageRealizationWireV1 {
    MissingTier,
    Unknown {
        reason: CoverageUnknownReasonV1,
    },
    Unverified {
        reason: CoverageUnverifiedReasonV1,
        observed_indexed_count: Option<u64>,
    },
    Verified {
        indexed: Box<CoverageWitnessWireV1>,
        relation: CoverageRelationV1,
        counts: CoverageCountsV1,
        cross_scope_evidence: Option<Box<CrossScopeCoverageEvidenceWireV1>>,
    },
}

impl TierCoverageRealizationWireV1 {
    fn into_candidate(self) -> SearchResult<TierCoverageRealizationV1> {
        match self {
            Self::MissingTier => Ok(TierCoverageRealizationV1::MissingTier),
            Self::Unknown { reason } => Ok(TierCoverageRealizationV1::Unknown { reason }),
            Self::Unverified {
                reason,
                observed_indexed_count,
            } => Ok(TierCoverageRealizationV1::Unverified {
                reason,
                observed_indexed_count,
            }),
            Self::Verified {
                indexed,
                relation,
                counts,
                cross_scope_evidence,
            } => Ok(TierCoverageRealizationV1::Verified {
                indexed: Box::new((*indexed).into_candidate()?),
                relation,
                counts,
                cross_scope_evidence: cross_scope_evidence
                    .map(|evidence| (*evidence).into_candidate().map(Box::new))
                    .transpose()?,
            }),
        }
    }
}

/// Requested and realized coverage for exactly one tier.
///
/// Fields are private and the trusted type is Serialize-only. Wire claims deserialize only as
/// [`UntrustedTierCoverageV1`] and acquire authority solely by exact comparison with a fresh
/// [`TrustedTierCoverageContextV1`] recomputation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct TierCoverageV1 {
    schema_version: u16,
    tier: CoverageTierV1,
    requested: TierCoverageRequestV1,
    realized: TierCoverageRealizationV1,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TierCoverageWireV1 {
    schema_version: u16,
    tier: CoverageTierV1,
    requested: TierCoverageRequestWireV1,
    realized: TierCoverageRealizationWireV1,
}

impl TierCoverageWireV1 {
    fn into_candidate(self) -> SearchResult<TierCoverageV1> {
        let coverage = TierCoverageV1 {
            schema_version: self.schema_version,
            tier: self.tier,
            requested: self.requested.into_candidate()?,
            realized: self.realized.into_candidate()?,
        };
        coverage.validate()?;
        Ok(coverage)
    }
}

impl TierCoverageV1 {
    /// Construct an explicitly unrequested/missing tier.
    #[must_use]
    pub const fn not_requested(tier: CoverageTierV1) -> Self {
        Self {
            schema_version: TIER_COVERAGE_SCHEMA_VERSION_V1,
            tier,
            requested: TierCoverageRequestV1::MissingTier,
            realized: TierCoverageRealizationV1::MissingTier,
        }
    }

    /// Construct a requested tier whose index is missing.
    ///
    /// # Errors
    ///
    /// Rejects malformed space/scope receipts, duplicate members, or unbounded member ids.
    pub fn requested_missing(
        tier: CoverageTierV1,
        space: &EmbeddingSpaceIdentityV1,
        basis: CoverageBasisV1,
        live_scope: CoverageScopeV1,
        live_members: Vec<String>,
    ) -> SearchResult<Self> {
        Self::requested_without_verified_realization(
            tier,
            space,
            basis,
            live_scope,
            live_members,
            TierCoverageRealizationV1::MissingTier,
        )
    }

    /// Construct a requested tier with explicitly unknown realized coverage.
    ///
    /// # Errors
    ///
    /// Rejects malformed space/scope receipts, duplicate members, or unbounded member ids.
    pub fn unknown(
        tier: CoverageTierV1,
        space: &EmbeddingSpaceIdentityV1,
        basis: CoverageBasisV1,
        live_scope: CoverageScopeV1,
        live_members: Vec<String>,
        reason: CoverageUnknownReasonV1,
    ) -> SearchResult<Self> {
        Self::requested_without_verified_realization(
            tier,
            space,
            basis,
            live_scope,
            live_members,
            TierCoverageRealizationV1::Unknown { reason },
        )
    }

    /// Construct a requested tier with an unverified scalar observation.
    ///
    /// `observed_indexed_count` remains diagnostic only and never contributes to topology.
    ///
    /// # Errors
    ///
    /// Rejects malformed space/scope receipts, duplicate members, or unbounded member ids.
    pub fn unverified(
        tier: CoverageTierV1,
        space: &EmbeddingSpaceIdentityV1,
        basis: CoverageBasisV1,
        live_scope: CoverageScopeV1,
        live_members: Vec<String>,
        reason: CoverageUnverifiedReasonV1,
        observed_indexed_count: Option<u64>,
    ) -> SearchResult<Self> {
        Self::requested_without_verified_realization(
            tier,
            space,
            basis,
            live_scope,
            live_members,
            TierCoverageRealizationV1::Unverified {
                reason,
                observed_indexed_count,
            },
        )
    }

    fn requested_without_verified_realization(
        tier: CoverageTierV1,
        space: &EmbeddingSpaceIdentityV1,
        basis: CoverageBasisV1,
        live_scope: CoverageScopeV1,
        live_members: Vec<String>,
        realized: TierCoverageRealizationV1,
    ) -> SearchResult<Self> {
        let live = prepare_coverage_members(tier, space, basis, live_scope, live_members)?.witness;
        let coverage = Self {
            schema_version: TIER_COVERAGE_SCHEMA_VERSION_V1,
            tier,
            requested: TierCoverageRequestV1::Requested { live },
            realized,
        };
        coverage.validate()?;
        Ok(coverage)
    }

    /// Derive a verified relation from requested live and realized indexed member collections.
    ///
    /// Canonical document membership is order-independent. Ordered shard membership preserves
    /// order in each witness while set comparison remains duplicate-free. When scope differs,
    /// `cross_scope_authority_fingerprint` is mandatory and is bound into exact set evidence.
    /// That fingerprint must come from retained independent owner authority: payload bytes and a
    /// digest computed by the claimant are not attestation and must never be passed here during
    /// wire promotion.
    /// A cross-scope exact-complete claim is rejected because generation/corpus equality cannot
    /// be inferred from equal member names.
    ///
    /// # Errors
    ///
    /// Rejects malformed receipts, duplicates, checked-count overflow, missing/redundant
    /// cross-scope authority, and impossible cross-scope complete claims.
    #[allow(clippy::too_many_arguments)]
    pub fn verified(
        tier: CoverageTierV1,
        space: &EmbeddingSpaceIdentityV1,
        basis: CoverageBasisV1,
        live_scope: CoverageScopeV1,
        live_members: Vec<String>,
        indexed_scope: CoverageScopeV1,
        indexed_members: Vec<String>,
        cross_scope_authority_fingerprint: Option<String>,
    ) -> SearchResult<Self> {
        let live = prepare_coverage_members(tier, space, basis, live_scope, live_members)?;
        let indexed = prepare_coverage_members(tier, space, basis, indexed_scope, indexed_members)?;
        let counts = CoverageCountsV1::from_member_sets(&live.members, &indexed.members)?;
        let relation = derive_coverage_relation(counts);
        let scopes_match = live.witness.scope == indexed.witness.scope;
        let cross_scope_evidence = match (scopes_match, cross_scope_authority_fingerprint) {
            (true, None) => None,
            (true, Some(_)) => {
                return Err(coverage_error(
                    "coverage.cross_scope_authority_fingerprint",
                    "redundant-same-scope-authority",
                    "must be absent when generation and corpus scopes are identical",
                ));
            }
            (false, None) => {
                return Err(coverage_error(
                    "coverage.cross_scope_authority_fingerprint",
                    "missing",
                    "different generation/corpus scopes require explicit comparison evidence",
                ));
            }
            (false, Some(authority)) => {
                if relation == CoverageRelationV1::Complete {
                    return Err(coverage_error(
                        "coverage.relation",
                        relation.code(),
                        "complete coverage requires identical generation and corpus scopes",
                    ));
                }
                Some(Box::new(CrossScopeCoverageEvidenceV1::build(
                    &live, &indexed, authority,
                )?))
            }
        };
        let coverage = Self {
            schema_version: TIER_COVERAGE_SCHEMA_VERSION_V1,
            tier,
            requested: TierCoverageRequestV1::Requested { live: live.witness },
            realized: TierCoverageRealizationV1::Verified {
                indexed: Box::new(indexed.witness),
                relation,
                counts,
                cross_scope_evidence,
            },
        };
        coverage.validate()?;
        Ok(coverage)
    }

    fn validate(&self) -> SearchResult<()> {
        validate_coverage_schema("coverage.schema_version", self.schema_version)?;
        match (&self.requested, &self.realized) {
            (TierCoverageRequestV1::MissingTier, TierCoverageRealizationV1::MissingTier) => {}
            (TierCoverageRequestV1::MissingTier, _) => {
                return Err(coverage_error(
                    "coverage.realized",
                    "present-for-unrequested-tier",
                    "realized coverage requires an independent requested-tier witness",
                ));
            }
            (TierCoverageRequestV1::Requested { live }, realized) => {
                live.validate()?;
                if live.tier != self.tier {
                    return Err(coverage_error(
                        "coverage.requested.live.tier",
                        live.tier.code(),
                        "a tier cannot borrow the other tier's live witness",
                    ));
                }
                if let TierCoverageRealizationV1::Verified {
                    indexed,
                    relation,
                    counts,
                    cross_scope_evidence,
                } = realized
                {
                    indexed.validate()?;
                    if indexed.tier != self.tier {
                        return Err(coverage_error(
                            "coverage.realized.indexed.tier",
                            indexed.tier.code(),
                            "a tier cannot borrow the other tier's indexed witness",
                        ));
                    }
                    if indexed.space_fingerprint != live.space_fingerprint
                        || indexed.space_kind != live.space_kind
                    {
                        return Err(coverage_error(
                            "coverage.realized.indexed.space",
                            "space-binding-mismatch",
                            "requested and realized witnesses must bind the same validated embedding space",
                        ));
                    }
                    if indexed.basis != live.basis {
                        return Err(coverage_error(
                            "coverage.realized.indexed.basis",
                            indexed.basis.code(),
                            "requested and realized witnesses must use the same member basis",
                        ));
                    }
                    if counts.live_count != live.member_count
                        || counts.indexed_count != indexed.member_count
                    {
                        return Err(coverage_error(
                            "coverage.counts",
                            "witness-count-mismatch",
                            "live/indexed counts must equal their independently derived witnesses",
                        ));
                    }
                    validate_relation_counts(*relation, *counts)?;
                    let scopes_match = live.scope == indexed.scope;
                    match (scopes_match, cross_scope_evidence) {
                        (true, None) => {}
                        (true, Some(_)) => {
                            return Err(coverage_error(
                                "coverage.cross_scope_evidence",
                                "present-for-same-scope",
                                "same-scope coverage must not carry cross-scope evidence",
                            ));
                        }
                        (false, None) => {
                            return Err(coverage_error(
                                "coverage.cross_scope_evidence",
                                "missing",
                                "different generation/corpus scopes are unverified without evidence",
                            ));
                        }
                        (false, Some(evidence)) => {
                            evidence.validate()?;
                            if *relation == CoverageRelationV1::Complete {
                                return Err(coverage_error(
                                    "coverage.relation",
                                    relation.code(),
                                    "complete coverage requires identical generation/corpus scopes",
                                ));
                            }
                            if evidence.live_scope_fingerprint != live.scope.fingerprint()
                                || evidence.indexed_scope_fingerprint != indexed.scope.fingerprint()
                                || evidence.live_set_fingerprint != live.set_fingerprint
                                || evidence.indexed_set_fingerprint != indexed.set_fingerprint
                                || evidence.live_witness_fingerprint != live.fingerprint()
                                || evidence.indexed_witness_fingerprint != indexed.fingerprint()
                            {
                                return Err(coverage_error(
                                    "coverage.cross_scope_evidence",
                                    "binding-mismatch",
                                    "evidence does not bind the exact live and indexed witnesses",
                                ));
                            }
                        }
                    }
                    if *relation == CoverageRelationV1::Complete
                        && (live.set_fingerprint != indexed.set_fingerprint
                            || live.membership_fingerprint != indexed.membership_fingerprint)
                    {
                        return Err(coverage_error(
                            "coverage.relation",
                            relation.code(),
                            "complete coverage requires exact same-scope membership witnesses",
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Tier role of this independent coverage value.
    #[must_use]
    pub const fn tier(&self) -> CoverageTierV1 {
        self.tier
    }

    /// Expected query/index mathematical-space fingerprint, absent only when unrequested.
    #[must_use]
    pub fn expected_space_fingerprint(&self) -> Option<&str> {
        self.live_witness()
            .map(CoverageWitnessV1::space_fingerprint)
    }

    /// Requested coverage state.
    #[must_use]
    pub const fn requested(&self) -> &TierCoverageRequestV1 {
        &self.requested
    }

    /// Realized coverage state.
    #[must_use]
    pub const fn realized(&self) -> &TierCoverageRealizationV1 {
        &self.realized
    }

    /// Derived relation/counts when and only when realized coverage is verified.
    #[must_use]
    pub const fn verified_facts(&self) -> Option<(CoverageRelationV1, CoverageCountsV1)> {
        match self.realized {
            TierCoverageRealizationV1::Verified {
                relation, counts, ..
            } => Some((relation, counts)),
            TierCoverageRealizationV1::MissingTier
            | TierCoverageRealizationV1::Unknown { .. }
            | TierCoverageRealizationV1::Unverified { .. } => None,
        }
    }

    /// Requested live witness, absent only when the tier was not requested.
    #[must_use]
    pub const fn live_witness(&self) -> Option<&CoverageWitnessV1> {
        match &self.requested {
            TierCoverageRequestV1::Requested { live } => Some(live),
            TierCoverageRequestV1::MissingTier => None,
        }
    }

    /// Realized indexed witness, present only for verified coverage.
    #[must_use]
    pub fn indexed_witness(&self) -> Option<&CoverageWitnessV1> {
        match &self.realized {
            TierCoverageRealizationV1::Verified { indexed, .. } => Some(indexed.as_ref()),
            TierCoverageRealizationV1::MissingTier
            | TierCoverageRealizationV1::Unknown { .. }
            | TierCoverageRealizationV1::Unverified { .. } => None,
        }
    }

    /// Explicit cross-scope comparison evidence, when generation/corpus differs.
    #[must_use]
    pub fn cross_scope_evidence(&self) -> Option<&CrossScopeCoverageEvidenceV1> {
        match &self.realized {
            TierCoverageRealizationV1::Verified {
                cross_scope_evidence,
                ..
            } => cross_scope_evidence.as_deref(),
            TierCoverageRealizationV1::MissingTier
            | TierCoverageRealizationV1::Unknown { .. }
            | TierCoverageRealizationV1::Unverified { .. } => None,
        }
    }

    /// Stable domain-separated digest of all requested and realized facts.
    #[must_use]
    pub fn fingerprint(&self) -> String {
        let mut encoder = CoverageEncoder::new(b"frankensearch.tier-coverage.v1");
        encoder.u16(self.schema_version);
        encoder.text(self.tier.code());
        encode_coverage_request(&mut encoder, &self.requested);
        encode_coverage_realization(&mut encoder, &self.realized);
        encoder.fingerprint()
    }

    const fn is_requested(&self) -> bool {
        matches!(self.requested, TierCoverageRequestV1::Requested { .. })
    }
}

#[derive(Clone)]
enum TrustedTierCoverageRealizationContextV1 {
    Missing,
    Unknown(CoverageUnknownReasonV1),
    Unverified {
        reason: CoverageUnverifiedReasonV1,
        observed_indexed_count: Option<u64>,
    },
    Verified {
        indexed_scope: CoverageScopeV1,
        indexed_members: Vec<String>,
        cross_scope_authority_fingerprint: Option<String>,
    },
}

/// Retained owner authority used to validate a serialized tier-coverage claim.
///
/// The context owns the validated mathematical space and the exact member
/// collections needed to recompute every digest, relation, count, and evidence
/// binding. It deliberately implements neither serialization nor deserialization;
/// its custom debug output never exposes member identifiers or producer identity.
#[derive(Clone)]
pub struct TrustedTierCoverageContextV1 {
    tier: CoverageTierV1,
    space: Option<EmbeddingSpaceIdentityV1>,
    basis: Option<CoverageBasisV1>,
    live_scope: Option<CoverageScopeV1>,
    live_members: Vec<String>,
    realization: TrustedTierCoverageRealizationContextV1,
}

impl fmt::Debug for TrustedTierCoverageContextV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TrustedTierCoverageContextV1")
            .field("tier", &self.tier)
            .field(
                "space_fingerprint",
                &self
                    .space
                    .as_ref()
                    .map(EmbeddingSpaceIdentityV1::fingerprint),
            )
            .field("basis", &self.basis)
            .field("live_member_count", &self.live_members.len())
            .field(
                "realization",
                &match self.realization {
                    TrustedTierCoverageRealizationContextV1::Missing => "missing",
                    TrustedTierCoverageRealizationContextV1::Unknown(_) => "unknown",
                    TrustedTierCoverageRealizationContextV1::Unverified { .. } => "unverified",
                    TrustedTierCoverageRealizationContextV1::Verified { .. } => "verified",
                },
            )
            .finish_non_exhaustive()
    }
}

impl TrustedTierCoverageContextV1 {
    /// Retain authority for an explicitly unrequested tier.
    #[must_use]
    pub const fn not_requested(tier: CoverageTierV1) -> Self {
        Self {
            tier,
            space: None,
            basis: None,
            live_scope: None,
            live_members: Vec::new(),
            realization: TrustedTierCoverageRealizationContextV1::Missing,
        }
    }

    /// Retain authority for a requested tier whose index is missing.
    ///
    /// # Errors
    ///
    /// Rejects malformed space or scope identity and invalid live membership.
    pub fn requested_missing(
        tier: CoverageTierV1,
        space: EmbeddingSpaceIdentityV1,
        basis: CoverageBasisV1,
        live_scope: CoverageScopeV1,
        live_members: Vec<String>,
    ) -> SearchResult<Self> {
        Self::requested(
            tier,
            space,
            basis,
            live_scope,
            live_members,
            TrustedTierCoverageRealizationContextV1::Missing,
        )
    }

    /// Retain authority for a requested tier with unknown realized coverage.
    ///
    /// # Errors
    ///
    /// Rejects malformed space or scope identity and invalid live membership.
    pub fn unknown(
        tier: CoverageTierV1,
        space: EmbeddingSpaceIdentityV1,
        basis: CoverageBasisV1,
        live_scope: CoverageScopeV1,
        live_members: Vec<String>,
        reason: CoverageUnknownReasonV1,
    ) -> SearchResult<Self> {
        Self::requested(
            tier,
            space,
            basis,
            live_scope,
            live_members,
            TrustedTierCoverageRealizationContextV1::Unknown(reason),
        )
    }

    /// Retain authority for a requested tier with an unverified observation.
    ///
    /// # Errors
    ///
    /// Rejects malformed space or scope identity, invalid live membership, and
    /// impossible observed counts.
    pub fn unverified(
        tier: CoverageTierV1,
        space: EmbeddingSpaceIdentityV1,
        basis: CoverageBasisV1,
        live_scope: CoverageScopeV1,
        live_members: Vec<String>,
        reason: CoverageUnverifiedReasonV1,
        observed_indexed_count: Option<u64>,
    ) -> SearchResult<Self> {
        Self::requested(
            tier,
            space,
            basis,
            live_scope,
            live_members,
            TrustedTierCoverageRealizationContextV1::Unverified {
                reason,
                observed_indexed_count,
            },
        )
    }

    /// Retain all authority needed to recompute verified coverage.
    ///
    /// A cross-scope fingerprint must be independently sourced from the owner
    /// that authorized comparison. Copying it from an untrusted receipt would
    /// make the caller self-attest and is not a valid promotion workflow.
    ///
    /// # Errors
    ///
    /// Rejects malformed identity, invalid membership, unauthorized cross-scope
    /// comparison, or impossible set algebra.
    #[allow(clippy::too_many_arguments)]
    pub fn verified(
        tier: CoverageTierV1,
        space: EmbeddingSpaceIdentityV1,
        basis: CoverageBasisV1,
        live_scope: CoverageScopeV1,
        live_members: Vec<String>,
        indexed_scope: CoverageScopeV1,
        indexed_members: Vec<String>,
        cross_scope_authority_fingerprint: Option<String>,
    ) -> SearchResult<Self> {
        Self::requested(
            tier,
            space,
            basis,
            live_scope,
            live_members,
            TrustedTierCoverageRealizationContextV1::Verified {
                indexed_scope,
                indexed_members,
                cross_scope_authority_fingerprint,
            },
        )
    }

    fn requested(
        tier: CoverageTierV1,
        space: EmbeddingSpaceIdentityV1,
        basis: CoverageBasisV1,
        live_scope: CoverageScopeV1,
        live_members: Vec<String>,
        realization: TrustedTierCoverageRealizationContextV1,
    ) -> SearchResult<Self> {
        let context = Self {
            tier,
            space: Some(space),
            basis: Some(basis),
            live_scope: Some(live_scope),
            live_members,
            realization,
        };
        context.recompute()?;
        Ok(context)
    }

    /// Recompute the trusted coverage value from retained authority.
    ///
    /// # Errors
    ///
    /// Rejects an internally inconsistent context or any identity, membership,
    /// evidence, or set-algebra invariant that fails fresh validation.
    pub fn recompute(&self) -> SearchResult<TierCoverageV1> {
        let Some(space) = self.space.as_ref() else {
            if self.basis.is_none()
                && self.live_scope.is_none()
                && self.live_members.is_empty()
                && matches!(
                    self.realization,
                    TrustedTierCoverageRealizationContextV1::Missing
                )
            {
                return Ok(TierCoverageV1::not_requested(self.tier));
            }
            return Err(coverage_error(
                "coverage.context",
                "incomplete-unrequested-context",
                "unrequested authority must not retain requested-tier facts",
            ));
        };
        let basis = self.basis.ok_or_else(|| {
            coverage_error(
                "coverage.context.basis",
                "missing",
                "requested authority requires a coverage basis",
            )
        })?;
        let live_scope = self.live_scope.clone().ok_or_else(|| {
            coverage_error(
                "coverage.context.live_scope",
                "missing",
                "requested authority requires a live scope",
            )
        })?;
        match &self.realization {
            TrustedTierCoverageRealizationContextV1::Missing => TierCoverageV1::requested_missing(
                self.tier,
                space,
                basis,
                live_scope,
                self.live_members.clone(),
            ),
            TrustedTierCoverageRealizationContextV1::Unknown(reason) => TierCoverageV1::unknown(
                self.tier,
                space,
                basis,
                live_scope,
                self.live_members.clone(),
                *reason,
            ),
            TrustedTierCoverageRealizationContextV1::Unverified {
                reason,
                observed_indexed_count,
            } => TierCoverageV1::unverified(
                self.tier,
                space,
                basis,
                live_scope,
                self.live_members.clone(),
                *reason,
                *observed_indexed_count,
            ),
            TrustedTierCoverageRealizationContextV1::Verified {
                indexed_scope,
                indexed_members,
                cross_scope_authority_fingerprint,
            } => TierCoverageV1::verified(
                self.tier,
                space,
                basis,
                live_scope,
                self.live_members.clone(),
                indexed_scope.clone(),
                indexed_members.clone(),
                cross_scope_authority_fingerprint.clone(),
            ),
        }
    }
}

/// Syntactically checked wire claim that has not yet acquired coverage authority.
#[derive(Deserialize)]
#[serde(transparent)]
pub struct UntrustedTierCoverageV1(TierCoverageWireV1);

impl UntrustedTierCoverageV1 {
    /// Promote only by exact comparison with a fresh owner-backed recomputation.
    ///
    /// # Errors
    ///
    /// Rejects malformed wire data and any claim that differs from the trusted
    /// owner context after exact recomputation.
    pub fn validate_against(
        self,
        context: &TrustedTierCoverageContextV1,
    ) -> SearchResult<TierCoverageV1> {
        let candidate = self.0.into_candidate()?;
        let recomputed = context.recompute()?;
        if candidate != recomputed {
            return Err(coverage_error(
                "coverage.untrusted",
                "authority-mismatch",
                "serialized coverage does not equal owner-backed recomputation",
            ));
        }
        Ok(recomputed)
    }
}

/// Typed proof that fast and quality requested the same exact member universe
/// under different generation/corpus scopes.
///
/// The proof cannot authorize differing membership: canonical document sets
/// must be equal, and ordered shard sequences must be byte-for-byte equal in
/// the same order. The independent authority fingerprint is supplied by the
/// retained owner context and is never trusted from a serialized payload.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct CrossTierUniverseEvidenceV1 {
    schema_version: u16,
    fast_live_witness_fingerprint: String,
    quality_live_witness_fingerprint: String,
    fast_scope_fingerprint: String,
    quality_scope_fingerprint: String,
    basis: CoverageBasisV1,
    member_count: u64,
    membership_fingerprint: String,
    set_fingerprint: String,
    authority_fingerprint: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CrossTierUniverseEvidenceWireV1 {
    schema_version: u16,
    fast_live_witness_fingerprint: String,
    quality_live_witness_fingerprint: String,
    fast_scope_fingerprint: String,
    quality_scope_fingerprint: String,
    basis: CoverageBasisV1,
    member_count: u64,
    membership_fingerprint: String,
    set_fingerprint: String,
    authority_fingerprint: String,
}

impl CrossTierUniverseEvidenceWireV1 {
    fn into_candidate(self) -> SearchResult<CrossTierUniverseEvidenceV1> {
        let evidence = CrossTierUniverseEvidenceV1 {
            schema_version: self.schema_version,
            fast_live_witness_fingerprint: self.fast_live_witness_fingerprint,
            quality_live_witness_fingerprint: self.quality_live_witness_fingerprint,
            fast_scope_fingerprint: self.fast_scope_fingerprint,
            quality_scope_fingerprint: self.quality_scope_fingerprint,
            basis: self.basis,
            member_count: self.member_count,
            membership_fingerprint: self.membership_fingerprint,
            set_fingerprint: self.set_fingerprint,
            authority_fingerprint: self.authority_fingerprint,
        };
        evidence.validate()?;
        Ok(evidence)
    }
}

impl CrossTierUniverseEvidenceV1 {
    fn build(
        fast: &CoverageWitnessV1,
        quality: &CoverageWitnessV1,
        authority_fingerprint: String,
    ) -> SearchResult<Self> {
        validate_coverage_sha256(
            "coverage.cross_tier.authority_fingerprint",
            &authority_fingerprint,
        )?;
        if fast.tier != CoverageTierV1::Fast || quality.tier != CoverageTierV1::Quality {
            return Err(coverage_error(
                "coverage.cross_tier",
                "swapped-tier-roles",
                "cross-tier evidence requires fast then quality live witnesses",
            ));
        }
        if fast.scope == quality.scope {
            return Err(coverage_error(
                "coverage.cross_tier",
                "redundant-same-scope-evidence",
                "cross-tier evidence is valid only when generation or corpus scope differs",
            ));
        }
        if fast.basis != quality.basis
            || fast.member_count != quality.member_count
            || fast.membership_fingerprint != quality.membership_fingerprint
            || fast.set_fingerprint != quality.set_fingerprint
        {
            return Err(coverage_error(
                "coverage.cross_tier",
                "requested-universe-mismatch",
                "cross-tier evidence cannot authorize differing sets, bases, or ordered sequences",
            ));
        }
        let evidence = Self {
            schema_version: TIER_COVERAGE_SCHEMA_VERSION_V1,
            fast_live_witness_fingerprint: fast.fingerprint(),
            quality_live_witness_fingerprint: quality.fingerprint(),
            fast_scope_fingerprint: fast.scope.fingerprint(),
            quality_scope_fingerprint: quality.scope.fingerprint(),
            basis: fast.basis,
            member_count: fast.member_count,
            membership_fingerprint: fast.membership_fingerprint.clone(),
            set_fingerprint: fast.set_fingerprint.clone(),
            authority_fingerprint,
        };
        evidence.validate_bound(fast, quality)?;
        Ok(evidence)
    }

    fn validate(&self) -> SearchResult<()> {
        validate_coverage_schema("coverage.cross_tier.schema_version", self.schema_version)?;
        for (field, value) in [
            (
                "fast_live_witness_fingerprint",
                &self.fast_live_witness_fingerprint,
            ),
            (
                "quality_live_witness_fingerprint",
                &self.quality_live_witness_fingerprint,
            ),
            ("fast_scope_fingerprint", &self.fast_scope_fingerprint),
            ("quality_scope_fingerprint", &self.quality_scope_fingerprint),
            ("membership_fingerprint", &self.membership_fingerprint),
            ("set_fingerprint", &self.set_fingerprint),
            ("authority_fingerprint", &self.authority_fingerprint),
        ] {
            validate_coverage_sha256(&format!("coverage.cross_tier.{field}"), value)?;
        }
        if self.fast_scope_fingerprint == self.quality_scope_fingerprint {
            return Err(coverage_error(
                "coverage.cross_tier",
                "redundant-same-scope-evidence",
                "cross-tier evidence requires different generation or corpus scope",
            ));
        }
        Ok(())
    }

    fn validate_bound(
        &self,
        fast: &CoverageWitnessV1,
        quality: &CoverageWitnessV1,
    ) -> SearchResult<()> {
        self.validate()?;
        if self.fast_live_witness_fingerprint != fast.fingerprint()
            || self.quality_live_witness_fingerprint != quality.fingerprint()
            || self.fast_scope_fingerprint != fast.scope.fingerprint()
            || self.quality_scope_fingerprint != quality.scope.fingerprint()
            || self.basis != fast.basis
            || self.basis != quality.basis
            || self.member_count != fast.member_count
            || self.member_count != quality.member_count
            || self.membership_fingerprint != fast.membership_fingerprint
            || self.membership_fingerprint != quality.membership_fingerprint
            || self.set_fingerprint != fast.set_fingerprint
            || self.set_fingerprint != quality.set_fingerprint
        {
            return Err(coverage_error(
                "coverage.cross_tier",
                "binding-mismatch",
                "cross-tier evidence does not bind both exact live witnesses",
            ));
        }
        Ok(())
    }

    /// Domain-separated fingerprint of the complete cross-tier proof.
    #[must_use]
    pub fn fingerprint(&self) -> String {
        let mut encoder = CoverageEncoder::new(b"frankensearch.cross-tier-universe.v1");
        encoder.u16(self.schema_version);
        encoder.text(&self.fast_live_witness_fingerprint);
        encoder.text(&self.quality_live_witness_fingerprint);
        encoder.text(&self.fast_scope_fingerprint);
        encoder.text(&self.quality_scope_fingerprint);
        encoder.text(self.basis.code());
        encoder.u64(self.member_count);
        encoder.text(&self.membership_fingerprint);
        encoder.text(&self.set_fingerprint);
        encoder.text(&self.authority_fingerprint);
        encoder.fingerprint()
    }
}

/// Fast and quality coverage values retained as two independently validated authorities.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct TierCoveragePairV1 {
    schema_version: u16,
    fast: TierCoverageV1,
    quality: TierCoverageV1,
    cross_tier_universe_evidence: Option<Box<CrossTierUniverseEvidenceV1>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TierCoveragePairWireV1 {
    schema_version: u16,
    fast: TierCoverageWireV1,
    quality: TierCoverageWireV1,
    cross_tier_universe_evidence: Option<Box<CrossTierUniverseEvidenceWireV1>>,
}

impl TierCoveragePairWireV1 {
    fn into_candidate(self) -> SearchResult<TierCoveragePairV1> {
        let pair = TierCoveragePairV1 {
            schema_version: self.schema_version,
            fast: self.fast.into_candidate()?,
            quality: self.quality.into_candidate()?,
            cross_tier_universe_evidence: self
                .cross_tier_universe_evidence
                .map(|evidence| (*evidence).into_candidate().map(Box::new))
                .transpose()?,
        };
        pair.validate()?;
        Ok(pair)
    }
}

/// Syntactically checked fast/quality wire claim without owner authority.
#[derive(Deserialize)]
#[serde(transparent)]
pub struct UntrustedTierCoveragePairV1(TierCoveragePairWireV1);

impl UntrustedTierCoveragePairV1 {
    /// Promote only after recomputing both tiers and any cross-tier proof from
    /// independent owner authority.
    ///
    /// # Errors
    ///
    /// Rejects malformed wire data, invalid trusted contexts, missing or
    /// malformed independent authority, and any non-identical recomputation.
    pub fn validate_against(
        self,
        fast_context: &TrustedTierCoverageContextV1,
        quality_context: &TrustedTierCoverageContextV1,
        cross_tier_authority_fingerprint: Option<&str>,
    ) -> SearchResult<TierCoveragePairV1> {
        let candidate = self.0.into_candidate()?;
        let fast = fast_context.recompute()?;
        let quality = quality_context.recompute()?;
        let recomputed = match cross_tier_authority_fingerprint {
            Some(authority) => TierCoveragePairV1::new_with_cross_tier_authority(
                fast,
                quality,
                authority.to_owned(),
            )?,
            None => TierCoveragePairV1::new(fast, quality)?,
        };
        if candidate != recomputed {
            return Err(coverage_error(
                "coverage.pair.untrusted",
                "authority-mismatch",
                "serialized coverage pair does not equal owner-backed recomputation",
            ));
        }
        Ok(recomputed)
    }
}

impl TierCoveragePairV1 {
    /// Pair exactly one fast and one quality coverage authority.
    ///
    /// # Errors
    ///
    /// Rejects swapped roles or any invalid child coverage object.
    pub fn new(fast: TierCoverageV1, quality: TierCoverageV1) -> SearchResult<Self> {
        Self::from_parts(fast, quality, None)
    }

    /// Pair equal requested membership under differing generation/corpus
    /// scopes using an independently supplied cross-tier authority.
    ///
    /// The authority fingerprint must be obtained from a retained owner outside
    /// the serialized claim. Calling this constructor with a self-sourced wire
    /// digest does not promote or attest untrusted data.
    ///
    /// # Errors
    ///
    /// Rejects same-scope evidence, mismatched bases or membership, reordered
    /// ordered shards, malformed authority fingerprints, and unrequested arms.
    pub fn new_with_cross_tier_authority(
        fast: TierCoverageV1,
        quality: TierCoverageV1,
        authority_fingerprint: impl Into<String>,
    ) -> SearchResult<Self> {
        let fast_live = fast.live_witness().ok_or_else(|| {
            coverage_error(
                "coverage.cross_tier.fast",
                "not-requested",
                "cross-tier evidence requires a requested fast universe",
            )
        })?;
        let quality_live = quality.live_witness().ok_or_else(|| {
            coverage_error(
                "coverage.cross_tier.quality",
                "not-requested",
                "cross-tier evidence requires a requested quality universe",
            )
        })?;
        let evidence = CrossTierUniverseEvidenceV1::build(
            fast_live,
            quality_live,
            authority_fingerprint.into(),
        )?;
        Self::from_parts(fast, quality, Some(Box::new(evidence)))
    }

    fn from_parts(
        fast: TierCoverageV1,
        quality: TierCoverageV1,
        cross_tier_universe_evidence: Option<Box<CrossTierUniverseEvidenceV1>>,
    ) -> SearchResult<Self> {
        let pair = Self {
            schema_version: TIER_COVERAGE_SCHEMA_VERSION_V1,
            fast,
            quality,
            cross_tier_universe_evidence,
        };
        pair.validate()?;
        Ok(pair)
    }

    fn validate(&self) -> SearchResult<()> {
        validate_coverage_schema("coverage.pair.schema_version", self.schema_version)?;
        self.fast.validate()?;
        self.quality.validate()?;
        if self.fast.tier != CoverageTierV1::Fast || self.quality.tier != CoverageTierV1::Quality {
            return Err(coverage_error(
                "coverage.pair",
                "swapped-tier-roles",
                "pair requires independent fast then quality coverage values",
            ));
        }
        match (self.fast.live_witness(), self.quality.live_witness()) {
            (None | Some(_), None) | (None, Some(_)) => {
                if self.cross_tier_universe_evidence.is_some() {
                    return Err(coverage_error(
                        "coverage.cross_tier",
                        "evidence-for-unrequested-tier",
                        "cross-tier evidence requires both requested universes",
                    ));
                }
            }
            (Some(fast), Some(quality)) => {
                if fast.basis != quality.basis
                    || fast.member_count != quality.member_count
                    || fast.membership_fingerprint != quality.membership_fingerprint
                    || fast.set_fingerprint != quality.set_fingerprint
                {
                    return Err(coverage_error(
                        "coverage.pair.requested_universe",
                        "mismatch",
                        "fast and quality must request the same canonical set or ordered shard sequence",
                    ));
                }
                match (
                    fast.scope == quality.scope,
                    self.cross_tier_universe_evidence.as_deref(),
                ) {
                    (true, None) => {}
                    (true, Some(_)) => {
                        return Err(coverage_error(
                            "coverage.cross_tier",
                            "redundant-same-scope-evidence",
                            "same-scope requested universes must not carry cross-tier evidence",
                        ));
                    }
                    (false, None) => {
                        return Err(coverage_error(
                            "coverage.cross_tier",
                            "missing",
                            "different requested generation/corpus scopes require typed cross-tier evidence",
                        ));
                    }
                    (false, Some(evidence)) => evidence.validate_bound(fast, quality)?,
                }
            }
        }
        Ok(())
    }

    /// Fast-tier coverage authority.
    #[must_use]
    pub const fn fast(&self) -> &TierCoverageV1 {
        &self.fast
    }

    /// Quality-tier coverage authority.
    #[must_use]
    pub const fn quality(&self) -> &TierCoverageV1 {
        &self.quality
    }

    /// Cross-tier universe proof, present only for equal exact membership
    /// under differing generation/corpus scopes.
    #[must_use]
    pub fn cross_tier_universe_evidence(&self) -> Option<&CrossTierUniverseEvidenceV1> {
        self.cross_tier_universe_evidence.as_deref()
    }

    /// Stable digest that preserves independent fast/quality witness roles.
    #[must_use]
    pub fn fingerprint(&self) -> String {
        let mut encoder = CoverageEncoder::new(b"frankensearch.tier-coverage-pair.v1");
        encoder.u16(self.schema_version);
        encoder.text(&self.fast.fingerprint());
        encoder.text(&self.quality.fingerprint());
        encoder.option_text(
            self.cross_tier_universe_evidence
                .as_ref()
                .map(|evidence| evidence.fingerprint())
                .as_deref(),
        );
        encoder.fingerprint()
    }

    /// Derive requested and realized topology from validated bindings plus both coverages.
    ///
    /// This is the authority-bearing constructor for realized topology. It validates complete
    /// identity bundles, joins each binding to its independently named tier/space, rejects a
    /// duplicate or swapped binding, and refuses semantic intent for hash-control spaces.
    /// Unknown/unverified coverage never becomes availability, and a realized superset fails
    /// closed because stale/foreign indexed members must not be searched.
    ///
    /// # Errors
    ///
    /// Rejects missing/extra/swapped/duplicate bindings, malformed identities, intent/kind
    /// mismatch, unsafe supersets, and any coverage-pair invariant failure.
    pub fn derive_topology(
        &self,
        intent: RetrievalIntentV1,
        embeddings: Option<&TieredQueryEmbeddings>,
    ) -> SearchResult<DerivedRetrievalTopologyV1> {
        self.validate()?;
        let fast_embedding = embeddings.and_then(TieredQueryEmbeddings::fast);
        let quality_embedding = embeddings.and_then(TieredQueryEmbeddings::quality);
        validate_tier_binding(&self.fast, fast_embedding)?;
        validate_tier_binding(&self.quality, quality_embedding)?;

        if let (Some(fast), Some(quality)) = (fast_embedding, quality_embedding)
            && fast.space_fingerprint() == quality.space_fingerprint()
        {
            return Err(coverage_error(
                "retrieval.bindings",
                "duplicate-space-binding",
                "fast and quality roles require independent space bindings",
            ));
        }

        let fast_kind = fast_embedding.map(|embedding| embedding.identity.space.kind);
        let quality_kind = quality_embedding.map(|embedding| embedding.identity.space.kind);
        let requested_from_arms = match (self.fast.is_requested(), self.quality.is_requested()) {
            (false, false) => RetrievalTopology::LexicalOnly,
            (true, false) => RetrievalTopology::FastOnly,
            (false, true) => RetrievalTopology::QualityOnly,
            (true, true) => RetrievalTopology::FullProgressive,
        };
        let fast_realized = RealizedTierCoverageV1::from_coverage(&self.fast)?;
        let quality_realized = RealizedTierCoverageV1::from_coverage(&self.quality)?;

        let (requested, coarse_realized) = match intent {
            RetrievalIntentV1::LexicalOnly => {
                if embeddings.is_some() || requested_from_arms != RetrievalTopology::LexicalOnly {
                    return Err(coverage_error(
                        "retrieval.intent",
                        "lexical_only",
                        "lexical-only intent cannot retain semantic/hash bindings or coverage",
                    ));
                }
                (
                    RetrievalTopology::LexicalOnly,
                    Some(RetrievalTopology::LexicalOnly),
                )
            }
            RetrievalIntentV1::HashControl => {
                if requested_from_arms == RetrievalTopology::LexicalOnly {
                    return Err(coverage_error(
                        "retrieval.intent",
                        "hash_control",
                        "hash-control intent requires an explicit bound control tier",
                    ));
                }
                validate_binding_kinds(
                    intent,
                    fast_kind,
                    quality_kind,
                    EmbeddingSpaceKindV1::HashControl,
                )?;
                let realized = if fast_realized.contributes() || quality_realized.contributes() {
                    Some(RetrievalTopology::HashControl)
                } else {
                    None
                };
                (RetrievalTopology::HashControl, realized)
            }
            RetrievalIntentV1::Semantic => {
                if requested_from_arms == RetrievalTopology::LexicalOnly {
                    return Err(coverage_error(
                        "retrieval.intent",
                        "semantic",
                        "semantic intent requires at least one independently bound tier",
                    ));
                }
                validate_binding_kinds(
                    intent,
                    fast_kind,
                    quality_kind,
                    EmbeddingSpaceKindV1::Semantic,
                )?;
                (
                    requested_from_arms,
                    derive_lossless_semantic_topology(&fast_realized, &quality_realized),
                )
            }
        };

        let semantic_available = intent == RetrievalIntentV1::Semantic
            && (fast_realized.contributes() || quality_realized.contributes());
        let decision = DerivedRetrievalTopologyV1 {
            schema_version: TIER_COVERAGE_SCHEMA_VERSION_V1,
            intent,
            requested,
            coarse_realized,
            fast_realized,
            quality_realized,
            fast_space_fingerprint: fast_embedding
                .map(|embedding| embedding.space_fingerprint().to_owned()),
            quality_space_fingerprint: quality_embedding
                .map(|embedding| embedding.space_fingerprint().to_owned()),
            fast_kind,
            quality_kind,
            coverage_pair_fingerprint: self.fingerprint(),
            semantic_available,
        };
        decision.validate()?;
        Ok(decision)
    }
}

/// Declared retrieval intent checked against the actual binding-space kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalIntentV1 {
    /// No vector arm may be present.
    LexicalOnly,
    /// Explicit deterministic non-semantic control/degradation lane.
    HashControl,
    /// Learned semantic vector retrieval.
    Semantic,
}

impl RetrievalIntentV1 {
    const fn code(self) -> &'static str {
        match self {
            Self::LexicalOnly => "lexical_only",
            Self::HashControl => "hash_control",
            Self::Semantic => "semantic",
        }
    }
}

/// Exact realized state of one independently admitted retrieval tier.
///
/// Unlike the legacy topology vocabulary, this preserves missing, unknown,
/// unverified, zero-live, complete, and partial facts per tier. Availability
/// is derived from exact nonzero intersection counts, never rounded PPM.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum RealizedTierCoverageV1 {
    /// The tier was not requested.
    NotRequested,
    /// The tier was requested but no realized index exists.
    Missing,
    /// No immutable owner could establish realized facts.
    Unknown {
        /// Closed reason for the unknown state.
        reason: CoverageUnknownReasonV1,
    },
    /// Observations exist but lack owner-backed member verification.
    Unverified {
        /// Closed reason verification failed.
        reason: CoverageUnverifiedReasonV1,
        /// Optional diagnostic count; never grants availability.
        observed_indexed_count: Option<u64>,
    },
    /// Requested and indexed member sets are both empty.
    ZeroLive {
        /// Exact checked zero-live counts.
        counts: CoverageCountsV1,
    },
    /// Requested and indexed member sets are exactly complete and non-empty.
    Complete {
        /// Exact checked complete counts.
        counts: CoverageCountsV1,
    },
    /// Incomplete same/cross-scope coverage, including exact disjoint facts.
    Partial {
        /// Exact derived relation; never `Complete`, `ZeroLive`, or `Superset`.
        relation: CoverageRelationV1,
        /// Exact checked counts and diagnostic PPM.
        counts: CoverageCountsV1,
    },
}

#[derive(Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
enum RealizedTierCoverageWireV1 {
    NotRequested,
    Missing,
    Unknown {
        reason: CoverageUnknownReasonV1,
    },
    Unverified {
        reason: CoverageUnverifiedReasonV1,
        observed_indexed_count: Option<u64>,
    },
    ZeroLive {
        counts: CoverageCountsV1,
    },
    Complete {
        counts: CoverageCountsV1,
    },
    Partial {
        relation: CoverageRelationV1,
        counts: CoverageCountsV1,
    },
}

impl RealizedTierCoverageWireV1 {
    fn into_candidate(self) -> SearchResult<RealizedTierCoverageV1> {
        let realized = match self {
            Self::NotRequested => RealizedTierCoverageV1::NotRequested,
            Self::Missing => RealizedTierCoverageV1::Missing,
            Self::Unknown { reason } => RealizedTierCoverageV1::Unknown { reason },
            Self::Unverified {
                reason,
                observed_indexed_count,
            } => RealizedTierCoverageV1::Unverified {
                reason,
                observed_indexed_count,
            },
            Self::ZeroLive { counts } => RealizedTierCoverageV1::ZeroLive { counts },
            Self::Complete { counts } => RealizedTierCoverageV1::Complete { counts },
            Self::Partial { relation, counts } => {
                RealizedTierCoverageV1::Partial { relation, counts }
            }
        };
        realized.validate()?;
        Ok(realized)
    }
}

impl RealizedTierCoverageV1 {
    fn from_coverage(coverage: &TierCoverageV1) -> SearchResult<Self> {
        let realized = match (&coverage.requested, &coverage.realized) {
            (TierCoverageRequestV1::MissingTier, TierCoverageRealizationV1::MissingTier) => {
                Self::NotRequested
            }
            (TierCoverageRequestV1::Requested { .. }, TierCoverageRealizationV1::MissingTier) => {
                Self::Missing
            }
            (
                TierCoverageRequestV1::Requested { .. },
                TierCoverageRealizationV1::Unknown { reason },
            ) => Self::Unknown { reason: *reason },
            (
                TierCoverageRequestV1::Requested { .. },
                TierCoverageRealizationV1::Unverified {
                    reason,
                    observed_indexed_count,
                },
            ) => Self::Unverified {
                reason: *reason,
                observed_indexed_count: *observed_indexed_count,
            },
            (
                TierCoverageRequestV1::Requested { .. },
                TierCoverageRealizationV1::Verified {
                    relation, counts, ..
                },
            ) => match relation {
                CoverageRelationV1::ZeroLive => Self::ZeroLive { counts: *counts },
                CoverageRelationV1::Complete => Self::Complete { counts: *counts },
                CoverageRelationV1::Subset
                | CoverageRelationV1::Overlap
                | CoverageRelationV1::Disjoint => Self::Partial {
                    relation: *relation,
                    counts: *counts,
                },
                CoverageRelationV1::Superset => {
                    return Err(coverage_error(
                        "retrieval.coverage",
                        "unsafe-superset",
                        "indexed coverage contains stale or foreign members",
                    ));
                }
            },
            (TierCoverageRequestV1::MissingTier, _) => {
                return Err(coverage_error(
                    "retrieval.coverage",
                    "realized-unrequested-tier",
                    "unrequested tier cannot carry realized coverage",
                ));
            }
        };
        realized.validate()?;
        Ok(realized)
    }

    fn validate(&self) -> SearchResult<()> {
        match self {
            Self::NotRequested | Self::Missing | Self::Unknown { .. } | Self::Unverified { .. } => {
                Ok(())
            }
            Self::ZeroLive { counts } => {
                validate_relation_counts(CoverageRelationV1::ZeroLive, *counts)
            }
            Self::Complete { counts } => {
                validate_relation_counts(CoverageRelationV1::Complete, *counts)
            }
            Self::Partial { relation, counts } => {
                if !matches!(
                    relation,
                    CoverageRelationV1::Subset
                        | CoverageRelationV1::Overlap
                        | CoverageRelationV1::Disjoint
                ) {
                    return Err(coverage_error(
                        "retrieval.realized_tier.relation",
                        relation.code(),
                        "partial realized state requires subset, overlap, or disjoint relation",
                    ));
                }
                validate_relation_counts(*relation, *counts)
            }
        }
    }

    /// True only when exact checked facts contain at least one live indexed member.
    #[must_use]
    pub const fn contributes(&self) -> bool {
        match self {
            Self::Complete { counts } | Self::Partial { counts, .. } => {
                counts.intersection_count > 0
            }
            Self::NotRequested
            | Self::Missing
            | Self::Unknown { .. }
            | Self::Unverified { .. }
            | Self::ZeroLive { .. } => false,
        }
    }

    /// Diagnostic PPM derived from exact counts, when coverage is verified.
    #[must_use]
    pub const fn coverage_ppm(&self) -> Option<u32> {
        match self {
            Self::ZeroLive { counts }
            | Self::Complete { counts }
            | Self::Partial { counts, .. } => Some(counts.coverage_ppm),
            Self::NotRequested | Self::Missing | Self::Unknown { .. } | Self::Unverified { .. } => {
                None
            }
        }
    }
}

/// Validated requested-versus-realized topology receipt.
///
/// This receipt contains bounded digests only. It records a successful derivation; it is not a
/// replacement for retaining the query embeddings and owner-backed coverage authorities used by
/// [`TierCoveragePairV1::derive_topology`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct DerivedRetrievalTopologyV1 {
    schema_version: u16,
    intent: RetrievalIntentV1,
    requested: RetrievalTopology,
    coarse_realized: Option<RetrievalTopology>,
    fast_realized: RealizedTierCoverageV1,
    quality_realized: RealizedTierCoverageV1,
    fast_space_fingerprint: Option<String>,
    quality_space_fingerprint: Option<String>,
    fast_kind: Option<EmbeddingSpaceKindV1>,
    quality_kind: Option<EmbeddingSpaceKindV1>,
    coverage_pair_fingerprint: String,
    semantic_available: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DerivedRetrievalTopologyWireV1 {
    schema_version: u16,
    intent: RetrievalIntentV1,
    requested: RetrievalTopology,
    coarse_realized: Option<RetrievalTopology>,
    fast_realized: RealizedTierCoverageWireV1,
    quality_realized: RealizedTierCoverageWireV1,
    fast_space_fingerprint: Option<String>,
    quality_space_fingerprint: Option<String>,
    fast_kind: Option<EmbeddingSpaceKindV1>,
    quality_kind: Option<EmbeddingSpaceKindV1>,
    coverage_pair_fingerprint: String,
    semantic_available: bool,
}

impl DerivedRetrievalTopologyWireV1 {
    fn into_candidate(self) -> SearchResult<DerivedRetrievalTopologyV1> {
        let decision = DerivedRetrievalTopologyV1 {
            schema_version: self.schema_version,
            intent: self.intent,
            requested: self.requested,
            coarse_realized: self.coarse_realized,
            fast_realized: self.fast_realized.into_candidate()?,
            quality_realized: self.quality_realized.into_candidate()?,
            fast_space_fingerprint: self.fast_space_fingerprint,
            quality_space_fingerprint: self.quality_space_fingerprint,
            fast_kind: self.fast_kind,
            quality_kind: self.quality_kind,
            coverage_pair_fingerprint: self.coverage_pair_fingerprint,
            semantic_available: self.semantic_available,
        };
        decision.validate()?;
        Ok(decision)
    }
}

/// Serialized derived-topology claim awaiting fresh coverage and binding authority.
#[derive(Deserialize)]
#[serde(transparent)]
pub struct UntrustedDerivedRetrievalTopologyV1(DerivedRetrievalTopologyWireV1);

impl UntrustedDerivedRetrievalTopologyV1 {
    /// Promote only by rerunning the complete topology derivation and comparing
    /// every receipt field with the wire claim.
    ///
    /// # Errors
    ///
    /// Rejects malformed wire data, invalid coverage or query bindings, and any
    /// receipt field that differs from fresh authoritative derivation.
    pub fn validate_against(
        self,
        pair: &TierCoveragePairV1,
        intent: RetrievalIntentV1,
        embeddings: Option<&TieredQueryEmbeddings>,
    ) -> SearchResult<DerivedRetrievalTopologyV1> {
        let candidate = self.0.into_candidate()?;
        let recomputed = pair.derive_topology(intent, embeddings)?;
        if candidate != recomputed {
            return Err(coverage_error(
                "retrieval.topology.untrusted",
                "authority-mismatch",
                "serialized topology does not equal a fresh authoritative derivation",
            ));
        }
        Ok(recomputed)
    }
}

impl DerivedRetrievalTopologyV1 {
    fn validate(&self) -> SearchResult<()> {
        validate_coverage_schema("retrieval.topology.schema_version", self.schema_version)?;
        self.fast_realized.validate()?;
        self.quality_realized.validate()?;
        validate_coverage_sha256(
            "retrieval.topology.coverage_pair_fingerprint",
            &self.coverage_pair_fingerprint,
        )?;
        for (field, fingerprint) in [
            (
                "retrieval.topology.fast_space_fingerprint",
                self.fast_space_fingerprint.as_deref(),
            ),
            (
                "retrieval.topology.quality_space_fingerprint",
                self.quality_space_fingerprint.as_deref(),
            ),
        ] {
            if let Some(fingerprint) = fingerprint {
                validate_coverage_sha256(field, fingerprint)?;
            }
        }
        if self.fast_space_fingerprint.is_some() != self.fast_kind.is_some()
            || self.quality_space_fingerprint.is_some() != self.quality_kind.is_some()
        {
            return Err(coverage_error(
                "retrieval.topology.bindings",
                "incomplete-kind-binding",
                "each retained space fingerprint must carry its validated space kind",
            ));
        }
        if self.fast_space_fingerprint.is_some()
            && self.fast_space_fingerprint == self.quality_space_fingerprint
        {
            return Err(coverage_error(
                "retrieval.topology.bindings",
                "duplicate-space-binding",
                "fast and quality roles require independent space bindings",
            ));
        }
        if self.fast_space_fingerprint.is_none()
            != matches!(self.fast_realized, RealizedTierCoverageV1::NotRequested)
            || self.quality_space_fingerprint.is_none()
                != matches!(self.quality_realized, RealizedTierCoverageV1::NotRequested)
        {
            return Err(coverage_error(
                "retrieval.topology.bindings",
                "request-state-mismatch",
                "each requested tier must have exactly one independently bound query embedding",
            ));
        }
        let binding_shape = match (
            self.fast_space_fingerprint.is_some(),
            self.quality_space_fingerprint.is_some(),
        ) {
            (false, false) => RetrievalTopology::LexicalOnly,
            (true, false) => RetrievalTopology::FastOnly,
            (false, true) => RetrievalTopology::QualityOnly,
            (true, true) => RetrievalTopology::FullProgressive,
        };
        match self.intent {
            RetrievalIntentV1::LexicalOnly => {
                if binding_shape != RetrievalTopology::LexicalOnly
                    || self.requested != RetrievalTopology::LexicalOnly
                    || self.coarse_realized != Some(RetrievalTopology::LexicalOnly)
                {
                    return Err(coverage_error(
                        "retrieval.topology",
                        "invalid-lexical-receipt",
                        "lexical-only receipt cannot carry vector bindings or topology",
                    ));
                }
            }
            RetrievalIntentV1::HashControl => {
                validate_binding_kinds(
                    self.intent,
                    self.fast_kind,
                    self.quality_kind,
                    EmbeddingSpaceKindV1::HashControl,
                )?;
                let expected =
                    if self.fast_realized.contributes() || self.quality_realized.contributes() {
                        Some(RetrievalTopology::HashControl)
                    } else {
                        None
                    };
                if self.requested != RetrievalTopology::HashControl
                    || self.coarse_realized != expected
                {
                    return Err(coverage_error(
                        "retrieval.topology",
                        "invalid-hash-control-receipt",
                        "hash control must remain explicit and non-semantic",
                    ));
                }
            }
            RetrievalIntentV1::Semantic => {
                validate_binding_kinds(
                    self.intent,
                    self.fast_kind,
                    self.quality_kind,
                    EmbeddingSpaceKindV1::Semantic,
                )?;
                if self.requested != binding_shape
                    || self.coarse_realized
                        != derive_lossless_semantic_topology(
                            &self.fast_realized,
                            &self.quality_realized,
                        )
                {
                    return Err(coverage_error(
                        "retrieval.topology",
                        "request-realization-mismatch",
                        "realized semantic topology exceeds or contradicts requested bindings",
                    ));
                }
            }
        }
        if let Some(RetrievalTopology::PartialQuality { coverage_ppm }) = self.coarse_realized
            && (coverage_ppm == 0 || coverage_ppm >= COMPLETE_COVERAGE_PPM)
        {
            return Err(coverage_error(
                "retrieval.topology.realized.coverage_ppm",
                &coverage_ppm.to_string(),
                "partial quality coverage must be strictly between zero and complete",
            ));
        }
        if let Some(realized) = self.coarse_realized
            && !retrieval_topology_fits_request(self.requested, realized)
        {
            return Err(coverage_error(
                "retrieval.topology",
                "request-realization-mismatch",
                "coarse realized topology exceeds or contradicts the requested topology",
            ));
        }
        let expected_semantic = self.intent == RetrievalIntentV1::Semantic
            && (self.fast_realized.contributes() || self.quality_realized.contributes());
        if self.semantic_available != expected_semantic {
            return Err(coverage_error(
                "retrieval.topology.semantic_available",
                &self.semantic_available.to_string(),
                "must be derived from semantic intent and realized semantic topology",
            ));
        }
        Ok(())
    }

    /// Validated request topology.
    #[must_use]
    pub const fn requested(&self) -> RetrievalTopology {
        self.requested
    }

    /// Lossless legacy coarse topology, absent when exact per-tier facts cannot
    /// be represented without lying.
    #[must_use]
    pub const fn coarse_realized(&self) -> Option<RetrievalTopology> {
        self.coarse_realized
    }

    /// Exact independently realized fast-tier state.
    #[must_use]
    pub const fn fast_realized(&self) -> &RealizedTierCoverageV1 {
        &self.fast_realized
    }

    /// Exact independently realized quality-tier state.
    #[must_use]
    pub const fn quality_realized(&self) -> &RealizedTierCoverageV1 {
        &self.quality_realized
    }

    /// Whether a learned semantic space is truthfully available.
    #[must_use]
    pub const fn semantic_available(&self) -> bool {
        self.semantic_available
    }

    /// Stable receipt digest; semantic mutations change it.
    #[must_use]
    pub fn fingerprint(&self) -> String {
        let mut encoder = CoverageEncoder::new(b"frankensearch.derived-retrieval-topology.v1");
        encoder.u16(self.schema_version);
        encoder.text(self.intent.code());
        encode_retrieval_topology(&mut encoder, self.requested);
        match self.coarse_realized {
            Some(realized) => {
                encoder.u8(1);
                encode_retrieval_topology(&mut encoder, realized);
            }
            None => encoder.u8(0),
        }
        encode_realized_tier_coverage(&mut encoder, &self.fast_realized);
        encode_realized_tier_coverage(&mut encoder, &self.quality_realized);
        encoder.option_text(self.fast_space_fingerprint.as_deref());
        encoder.option_text(self.quality_space_fingerprint.as_deref());
        encoder.option_text(self.fast_kind.map(embedding_space_kind_code));
        encoder.option_text(self.quality_kind.map(embedding_space_kind_code));
        encoder.text(&self.coverage_pair_fingerprint);
        encoder.bool(self.semantic_available);
        encoder.fingerprint()
    }
}

fn validate_tier_binding(
    coverage: &TierCoverageV1,
    embedding: Option<&BoundQueryEmbedding>,
) -> SearchResult<()> {
    match (coverage.is_requested(), embedding) {
        (false, None) => Ok(()),
        (false, Some(_)) => Err(coverage_error(
            "retrieval.bindings",
            coverage.tier.code(),
            "an unrequested tier cannot retain an extra query binding",
        )),
        (true, None) => Err(coverage_error(
            "retrieval.bindings",
            coverage.tier.code(),
            "a requested tier requires its own independently bound query embedding",
        )),
        (true, Some(embedding)) => {
            embedding.identity.validate()?;
            let expected = coverage.expected_space_fingerprint().ok_or_else(|| {
                coverage_error(
                    "retrieval.bindings",
                    coverage.tier.code(),
                    "requested coverage omitted its expected mathematical space",
                )
            })?;
            embedding.verify_space_identity(expected, coverage.tier.code())
        }
    }
}

fn validate_binding_kinds(
    intent: RetrievalIntentV1,
    fast_kind: Option<EmbeddingSpaceKindV1>,
    quality_kind: Option<EmbeddingSpaceKindV1>,
    expected: EmbeddingSpaceKindV1,
) -> SearchResult<()> {
    if fast_kind.is_none() && quality_kind.is_none() {
        return Err(coverage_error(
            "retrieval.bindings",
            intent.code(),
            "vector retrieval intent requires at least one validated binding",
        ));
    }
    for (tier, kind) in [
        (CoverageTierV1::Fast, fast_kind),
        (CoverageTierV1::Quality, quality_kind),
    ] {
        if let Some(kind) = kind
            && kind != expected
        {
            return Err(coverage_error(
                &format!("retrieval.bindings.{}.space_kind", tier.code()),
                embedding_space_kind_code(kind),
                match expected {
                    EmbeddingSpaceKindV1::Semantic => {
                        "hash control cannot satisfy semantic availability"
                    }
                    EmbeddingSpaceKindV1::HashControl => {
                        "hash-control intent cannot relabel a learned semantic space"
                    }
                },
            ));
        }
    }
    Ok(())
}

const fn embedding_space_kind_code(kind: EmbeddingSpaceKindV1) -> &'static str {
    match kind {
        EmbeddingSpaceKindV1::Semantic => "semantic",
        EmbeddingSpaceKindV1::HashControl => "hash_control",
    }
}

fn derive_lossless_semantic_topology(
    fast: &RealizedTierCoverageV1,
    quality: &RealizedTierCoverageV1,
) -> Option<RetrievalTopology> {
    match (fast, quality) {
        (RealizedTierCoverageV1::Complete { .. }, RealizedTierCoverageV1::Complete { .. }) => {
            Some(RetrievalTopology::FullProgressive)
        }
        (
            RealizedTierCoverageV1::Complete { .. },
            RealizedTierCoverageV1::Partial { counts, .. },
        ) if counts.intersection_count > 0
            && counts.coverage_ppm > 0
            && counts.coverage_ppm < COMPLETE_COVERAGE_PPM =>
        {
            Some(RetrievalTopology::PartialQuality {
                coverage_ppm: counts.coverage_ppm,
            })
        }
        (RealizedTierCoverageV1::Complete { .. }, quality) if !quality.contributes() => {
            Some(RetrievalTopology::FastOnly)
        }
        (fast, RealizedTierCoverageV1::Complete { .. }) if !fast.contributes() => {
            Some(RetrievalTopology::QualityOnly)
        }
        (fast, quality) if !fast.contributes() && !quality.contributes() => {
            Some(RetrievalTopology::LexicalOnly)
        }
        _ => None,
    }
}

/// Canonical compatibility law shared by coverage derivation and recovery contracts.
#[must_use]
pub const fn retrieval_topology_fits_request(
    requested: RetrievalTopology,
    realized: RetrievalTopology,
) -> bool {
    match requested {
        RetrievalTopology::LexicalOnly => matches!(realized, RetrievalTopology::LexicalOnly),
        RetrievalTopology::FastOnly => matches!(
            realized,
            RetrievalTopology::FastOnly | RetrievalTopology::LexicalOnly
        ),
        RetrievalTopology::QualityOnly => matches!(
            realized,
            RetrievalTopology::QualityOnly
                | RetrievalTopology::PartialQuality { .. }
                | RetrievalTopology::LexicalOnly
        ),
        RetrievalTopology::FullProgressive => matches!(
            realized,
            RetrievalTopology::FullProgressive
                | RetrievalTopology::PartialQuality { .. }
                | RetrievalTopology::FastOnly
                | RetrievalTopology::QualityOnly
                | RetrievalTopology::LexicalOnly
        ),
        RetrievalTopology::HashControl => matches!(realized, RetrievalTopology::HashControl),
        RetrievalTopology::PartialQuality { .. } => false,
    }
}

const fn derive_coverage_relation(counts: CoverageCountsV1) -> CoverageRelationV1 {
    if counts.live_count == 0 && counts.indexed_count == 0 {
        CoverageRelationV1::ZeroLive
    } else if counts.live_count == counts.indexed_count
        && counts.intersection_count == counts.live_count
    {
        CoverageRelationV1::Complete
    } else if counts.intersection_count == counts.indexed_count
        && counts.indexed_count < counts.live_count
    {
        CoverageRelationV1::Subset
    } else if counts.intersection_count == counts.live_count
        && counts.live_count < counts.indexed_count
    {
        CoverageRelationV1::Superset
    } else if counts.intersection_count == 0 {
        CoverageRelationV1::Disjoint
    } else {
        CoverageRelationV1::Overlap
    }
}

fn validate_relation_counts(
    relation: CoverageRelationV1,
    counts: CoverageCountsV1,
) -> SearchResult<()> {
    let derived = derive_coverage_relation(counts);
    if relation != derived {
        return Err(coverage_error(
            "coverage.relation",
            relation.code(),
            &format!("checked counts derive relation {}", derived.code()),
        ));
    }
    if counts.coverage_ppm > COMPLETE_COVERAGE_PPM {
        return Err(coverage_error(
            "coverage.counts.coverage_ppm",
            &counts.coverage_ppm.to_string(),
            "fixed-point coverage cannot exceed one million PPM",
        ));
    }
    Ok(())
}

fn fingerprint_member_set(domain: &[u8], members: &BTreeSet<String>) -> SearchResult<String> {
    let member_count = u64::try_from(members.len()).map_err(|_| {
        coverage_error(
            "coverage.member_count",
            "redacted-overflow",
            "member count does not fit in u64",
        )
    })?;
    let mut encoder = CoverageEncoder::new(domain);
    encoder.u64(member_count);
    for member in members {
        encoder.text(member);
    }
    Ok(encoder.fingerprint())
}

fn encode_coverage_request(encoder: &mut CoverageEncoder, request: &TierCoverageRequestV1) {
    match request {
        TierCoverageRequestV1::MissingTier => encoder.u8(0),
        TierCoverageRequestV1::Requested { live } => {
            encoder.u8(1);
            encoder.text(&live.fingerprint());
        }
    }
}

fn encode_coverage_realization(
    encoder: &mut CoverageEncoder,
    realization: &TierCoverageRealizationV1,
) {
    match realization {
        TierCoverageRealizationV1::MissingTier => encoder.u8(0),
        TierCoverageRealizationV1::Unknown { reason } => {
            encoder.u8(1);
            encoder.text(coverage_unknown_reason_code(*reason));
        }
        TierCoverageRealizationV1::Unverified {
            reason,
            observed_indexed_count,
        } => {
            encoder.u8(2);
            encoder.text(coverage_unverified_reason_code(*reason));
            encoder.option_u64(*observed_indexed_count);
        }
        TierCoverageRealizationV1::Verified {
            indexed,
            relation,
            counts,
            cross_scope_evidence,
        } => {
            encoder.u8(3);
            encoder.text(&indexed.fingerprint());
            encoder.text(relation.code());
            encoder.u64(counts.live_count);
            encoder.u64(counts.indexed_count);
            encoder.u64(counts.intersection_count);
            encoder.u64(counts.union_count);
            encoder.u32(counts.coverage_ppm);
            encoder.option_text(
                cross_scope_evidence
                    .as_ref()
                    .map(|evidence| evidence.fingerprint())
                    .as_deref(),
            );
        }
    }
}

fn encode_realized_tier_coverage(encoder: &mut CoverageEncoder, realized: &RealizedTierCoverageV1) {
    match realized {
        RealizedTierCoverageV1::NotRequested => encoder.u8(0),
        RealizedTierCoverageV1::Missing => encoder.u8(1),
        RealizedTierCoverageV1::Unknown { reason } => {
            encoder.u8(2);
            encoder.text(coverage_unknown_reason_code(*reason));
        }
        RealizedTierCoverageV1::Unverified {
            reason,
            observed_indexed_count,
        } => {
            encoder.u8(3);
            encoder.text(coverage_unverified_reason_code(*reason));
            encoder.option_u64(*observed_indexed_count);
        }
        RealizedTierCoverageV1::ZeroLive { counts } => {
            encoder.u8(4);
            encode_coverage_counts(encoder, *counts);
        }
        RealizedTierCoverageV1::Complete { counts } => {
            encoder.u8(5);
            encode_coverage_counts(encoder, *counts);
        }
        RealizedTierCoverageV1::Partial { relation, counts } => {
            encoder.u8(6);
            encoder.text(relation.code());
            encode_coverage_counts(encoder, *counts);
        }
    }
}

fn encode_coverage_counts(encoder: &mut CoverageEncoder, counts: CoverageCountsV1) {
    encoder.u64(counts.live_count);
    encoder.u64(counts.indexed_count);
    encoder.u64(counts.intersection_count);
    encoder.u64(counts.union_count);
    encoder.u32(counts.coverage_ppm);
}

const fn coverage_unknown_reason_code(reason: CoverageUnknownReasonV1) -> &'static str {
    match reason {
        CoverageUnknownReasonV1::OwnerUnavailable => "owner_unavailable",
        CoverageUnknownReasonV1::LegacyUnidentified => "legacy_unidentified",
        CoverageUnknownReasonV1::GenerationUnresolved => "generation_unresolved",
        CoverageUnknownReasonV1::CorpusUnresolved => "corpus_unresolved",
    }
}

const fn coverage_unverified_reason_code(reason: CoverageUnverifiedReasonV1) -> &'static str {
    match reason {
        CoverageUnverifiedReasonV1::WitnessMissing => "witness_missing",
        CoverageUnverifiedReasonV1::WitnessDigestMismatch => "witness_digest_mismatch",
        CoverageUnverifiedReasonV1::CountsUntrusted => "counts_untrusted",
        CoverageUnverifiedReasonV1::CrossScopeUnproven => "cross_scope_unproven",
    }
}

fn encode_retrieval_topology(encoder: &mut CoverageEncoder, topology: RetrievalTopology) {
    encoder.text(topology.code());
    if let RetrievalTopology::PartialQuality { coverage_ppm } = topology {
        encoder.u32(coverage_ppm);
    }
}

fn validate_coverage_schema(field: &str, schema_version: u16) -> SearchResult<()> {
    if schema_version == TIER_COVERAGE_SCHEMA_VERSION_V1 {
        return Ok(());
    }
    Err(coverage_error(
        field,
        &schema_version.to_string(),
        "unknown coverage schema version",
    ))
}

fn validate_coverage_sha256(field: &str, value: &str) -> SearchResult<()> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Ok(());
    }
    let redacted_value = format!("redacted-invalid-sha256-length-{}", value.len());
    Err(coverage_error(
        field,
        &redacted_value,
        "must be exactly 64 lowercase hexadecimal SHA-256 characters",
    ))
}

fn coverage_error(field: &str, value: &str, reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: field.to_owned(),
        value: value.to_owned(),
        reason: reason.to_owned(),
    }
}

#[derive(Debug)]
struct CoverageEncoder {
    bytes: Vec<u8>,
}

impl CoverageEncoder {
    fn new(domain: &[u8]) -> Self {
        let mut encoder = Self { bytes: Vec::new() };
        encoder.bytes(domain);
        encoder
    }

    fn bool(&mut self, value: bool) {
        self.u8(u8::from(value));
    }

    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn bytes(&mut self, value: &[u8]) {
        self.u64(u64::try_from(value.len()).unwrap_or(u64::MAX));
        self.bytes.extend_from_slice(value);
    }

    fn text(&mut self, value: &str) {
        self.bytes(value.as_bytes());
    }

    fn option_text(&mut self, value: Option<&str>) {
        match value {
            Some(value) => {
                self.u8(1);
                self.text(value);
            }
            None => self.u8(0),
        }
    }

    fn option_u64(&mut self, value: Option<u64>) {
        match value {
            Some(value) => {
                self.u8(1);
                self.u64(value);
            }
            None => self.u8(0),
        }
    }

    fn fingerprint(self) -> String {
        let digest = Sha256::digest(&self.bytes);
        let mut hex = String::with_capacity(digest.len() * 2);
        for byte in digest {
            let _ = write!(&mut hex, "{byte:02x}");
        }
        hex
    }
}

/// The retrieval shape a search actually ran with.
///
/// Requested versus realized topology is first-class telemetry (bd-9xuj):
/// a caller that asked for full progressive search and silently received a
/// hash-candidate pool rescored by the quality model was the defining
/// failure this vocabulary exists to make impossible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case", tag = "topology")]
pub enum RetrievalTopology {
    /// No semantic arm ran; lexical results only.
    LexicalOnly,
    /// Non-semantic hash vectors, as an explicit test/control lane or a
    /// declared lexical degradation — never semantic availability.
    HashControl,
    /// Fast-tier retrieval only.
    FastOnly,
    /// Quality index as the primary retrieval arm.
    QualityOnly,
    /// Fast retrieval for latency plus direct quality retrieval unioned
    /// before fusion; never a mere rescoring of the fast pool.
    FullProgressive,
    /// Progressive retrieval where only part of the doc set has
    /// quality-tier coverage.
    PartialQuality {
        /// Fraction of live documents with quality-tier embeddings, in
        /// parts per million for exact serialization.
        coverage_ppm: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum RetrievalTopologyTagWireV1 {
    LexicalOnly,
    HashControl,
    FastOnly,
    QualityOnly,
    FullProgressive,
    PartialQuality,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RetrievalTopologyWireV1 {
    topology: RetrievalTopologyTagWireV1,
    #[serde(default)]
    coverage_ppm: RetrievalTopologyFieldV1<u32>,
}

#[derive(Default)]
enum RetrievalTopologyFieldV1<T> {
    #[default]
    Missing,
    Present(T),
}

impl<'de, T> Deserialize<'de> for RetrievalTopologyFieldV1<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        T::deserialize(deserializer).map(Self::Present)
    }
}

impl<'de> Deserialize<'de> for RetrievalTopology {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = RetrievalTopologyWireV1::deserialize(deserializer)?;
        match (wire.topology, wire.coverage_ppm) {
            (RetrievalTopologyTagWireV1::LexicalOnly, RetrievalTopologyFieldV1::Missing) => {
                Ok(Self::LexicalOnly)
            }
            (RetrievalTopologyTagWireV1::HashControl, RetrievalTopologyFieldV1::Missing) => {
                Ok(Self::HashControl)
            }
            (RetrievalTopologyTagWireV1::FastOnly, RetrievalTopologyFieldV1::Missing) => {
                Ok(Self::FastOnly)
            }
            (RetrievalTopologyTagWireV1::QualityOnly, RetrievalTopologyFieldV1::Missing) => {
                Ok(Self::QualityOnly)
            }
            (RetrievalTopologyTagWireV1::FullProgressive, RetrievalTopologyFieldV1::Missing) => {
                Ok(Self::FullProgressive)
            }
            (
                RetrievalTopologyTagWireV1::PartialQuality,
                RetrievalTopologyFieldV1::Present(coverage_ppm),
            ) => Ok(Self::PartialQuality { coverage_ppm }),
            (RetrievalTopologyTagWireV1::PartialQuality, RetrievalTopologyFieldV1::Missing) => {
                Err(serde::de::Error::missing_field("coverage_ppm"))
            }
            (_, RetrievalTopologyFieldV1::Present(_)) => Err(serde::de::Error::custom(
                "coverage_ppm is valid only for partial_quality",
            )),
        }
    }
}

impl RetrievalTopology {
    /// Stable `snake_case` code for logs and machine payloads.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::LexicalOnly => "lexical_only",
            Self::HashControl => "hash_control",
            Self::FastOnly => "fast_only",
            Self::QualityOnly => "quality_only",
            Self::FullProgressive => "full_progressive",
            Self::PartialQuality { .. } => "partial_quality",
        }
    }

    /// True when semantic vectors contribute to results. `HashControl` is
    /// deliberately non-semantic.
    #[must_use]
    pub const fn is_semantic(&self) -> bool {
        matches!(
            self,
            Self::FastOnly
                | Self::QualityOnly
                | Self::FullProgressive
                | Self::PartialQuality { .. }
        )
    }
}

/// A hit from hybrid fusion (lexical + semantic combined via RRF).
///
/// RRF scores are computed in f64 for precision during accumulation of many
/// small `1/(K+rank+1)` values, then carried as f64 throughout fusion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedHit {
    /// Document identifier.
    pub doc_id: DocId,
    /// RRF-fused score (f64 for precision during fusion).
    pub rrf_score: f64,
    /// Rank in the lexical (BM25) source, if present.
    pub lexical_rank: Option<usize>,
    /// Rank in the semantic (vector) source, if present.
    pub semantic_rank: Option<usize>,
    /// Internal vector index, if present.
    pub semantic_index: Option<u32>,
    /// Raw BM25 score from lexical search, if applicable.
    pub lexical_score: Option<f32>,
    /// Raw cosine similarity from semantic search, if applicable.
    pub semantic_score: Option<f32>,
    /// True if this document appeared in both lexical and semantic results.
    pub in_both_sources: bool,
}

impl FusedHit {
    /// Four-level deterministic tie-breaking for RRF results:
    /// 1. Higher RRF score first
    /// 2. Documents in both sources preferred
    /// 3. Higher lexical score preferred
    /// 4. Lexicographic doc\_id (deterministic fallback)
    #[must_use]
    pub fn cmp_for_ranking(&self, other: &Self) -> std::cmp::Ordering {
        // 1. RRF score descending
        other
            .rrf_score
            .total_cmp(&self.rrf_score)
            // 2. in_both_sources preferred (true > false)
            .then(other.in_both_sources.cmp(&self.in_both_sources))
            // 3. Lexical score descending (treat None as -inf)
            .then_with(|| {
                let a = self.lexical_score.unwrap_or(f32::NEG_INFINITY);
                let b = other.lexical_score.unwrap_or(f32::NEG_INFINITY);
                b.total_cmp(&a)
            })
            // 4. doc_id ascending (deterministic)
            .then_with(|| self.doc_id.cmp(&other.doc_id))
    }
}

/// Which search backend produced a result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScoreSource {
    /// Lexical (BM25) search only.
    Lexical,
    /// Fast-tier semantic search only.
    SemanticFast,
    /// Quality-tier semantic search only.
    SemanticQuality,
    /// Hybrid fusion (lexical + vector via RRF).
    ///
    /// Vector here may be semantic or a hash-control generation. Hash-only
    /// vector hits use [`Self::HashControl`], not this variant.
    Hybrid,
    /// Result was reranked by cross-encoder.
    Reranked,
    /// Hash / FNV / JL control vector search. Not semantic.
    HashControl,
}

/// True when `embedder_id` is a hash/fnv/jl control identity, not a semantic model.
#[must_use]
pub fn is_hash_generation_id(embedder_id: &str) -> bool {
    let id = embedder_id.to_ascii_lowercase();
    id == "hash"
        || id.starts_with("hash-")
        || id.starts_with("hash/")
        || id.starts_with("fnv1a-")
        || id.starts_with("jl-")
}

/// The final scored search result delivered to consumers.
///
/// Intentionally does NOT carry document text. Text is expensive and most
/// consumers only need `doc_id` + scores. When text is needed (e.g., for
/// reranking or display), look it up from your document store via `doc_id`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredResult {
    /// Unique document identifier.
    pub doc_id: DocId,
    /// Primary relevance score (RRF or blended, truncated to f32).
    pub score: f32,
    /// Which search backend produced this result.
    pub source: ScoreSource,
    /// Internal vector index, if applicable.
    pub index: Option<u32>,
    /// Score from fast-tier semantic search, if applicable.
    pub fast_score: Option<f32>,
    /// Score from quality-tier semantic search, if applicable.
    pub quality_score: Option<f32>,
    /// BM25 score from lexical search, if applicable.
    pub lexical_score: Option<f32>,
    /// Cross-encoder score from reranking, if applicable.
    pub rerank_score: Option<f32>,
    /// Detailed explanation of scoring (if enabled). `Box`ed so the (usually
    /// `None`, `explain=false`) common case doesn't carry ~88 B of inline
    /// `HitExplanation` in every `ScoredResult` — halves the struct (168→88 B),
    /// speeding `limit_all` materialization + phase clones and halving result-set
    /// memory (`scoredresult_box_ab` bench). `Box<HitExplanation>` derefs to
    /// `HitExplanation`; the heap alloc happens only when `explain=true`.
    pub explanation: Option<Box<HitExplanation>>,
    /// Arbitrary document metadata (from index stored fields). `Arc`-wrapped so
    /// the per-winner metadata materialization at `limit_all` is a refcount bump,
    /// not a deep `Value` clone (map + strings + arrays re-allocated) — measured
    /// 200–278× cheaper for realistic metadata (`metadata_clone_ab` bench).
    /// `Arc<Value>` derefs to `Value`, so reads (`.get`, `.as_object`, filters via
    /// `.as_deref()`) are unchanged.
    pub metadata: Option<Arc<serde_json::Value>>,
}

// ---------------------------------------------------------------------------
// Search mode and phases
// ---------------------------------------------------------------------------

/// Search mode selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchMode {
    /// BM25 keyword matching only.
    Lexical,
    /// Embedding similarity only.
    Semantic,
    /// RRF fusion of lexical + semantic.
    Hybrid,
    /// Progressive: fast semantic -> quality refinement + lexical fusion.
    TwoTier,
}

/// Diagnostic metrics for a search phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseMetrics {
    /// Which embedder was used for this phase.
    pub embedder_id: String,
    /// Number of vectors searched.
    pub vectors_searched: usize,
    /// Number of lexical candidates retrieved.
    pub lexical_candidates: usize,
    /// Number of results after fusion.
    pub fused_count: usize,
    /// Why this phase skipped semantic work, when it did.
    ///
    /// Absent on a healthy semantic Initial. Present when results are
    /// lexical-only or otherwise degraded (hash embedder, missing quality
    /// index, circuit breaker, …). Callers that treat Initial as "semantic
    /// hits" must inspect this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skip_reason: Option<String>,
}

/// Structured telemetry for a completed search request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchMetrics {
    /// Which search mode ran.
    pub mode: SearchMode,
    /// Query class selected by the classifier, when available.
    pub query_class: Option<QueryClass>,
    /// End-to-end latency for the search request.
    pub total_latency_ms: f64,
    /// Latency for phase 1 (`Initial`) when available.
    pub phase1_latency_ms: Option<f64>,
    /// Latency for phase 2 (`Refined`) when available.
    pub phase2_latency_ms: Option<f64>,
    /// Number of results returned to the caller.
    pub result_count: usize,
    /// Number of lexical candidates retrieved.
    pub lexical_candidates: usize,
    /// Number of semantic candidates retrieved.
    pub semantic_candidates: usize,
    /// Whether quality refinement was applied.
    pub refined: bool,
}

/// Structured telemetry for an embedding operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingMetrics {
    /// Stable embedder identifier (for example, `potion-multilingual-128M`).
    pub embedder_id: String,
    /// Number of texts embedded in this operation.
    pub batch_size: usize,
    /// Embedding operation latency.
    pub duration_ms: f64,
    /// Embedding vector dimension.
    pub dimension: usize,
    /// Whether this embedder is semantically meaningful.
    pub is_semantic: bool,
}

/// Structured telemetry for index update operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetrics {
    /// Document count after the update.
    pub doc_count: usize,
    /// Total on-disk index size after the update.
    pub index_size_bytes: u64,
    /// Number of documents added or modified by this update.
    pub updated_docs: usize,
    /// Whether the index was marked stale during this update.
    pub staleness_detected: bool,
}

/// Tracks how rankings changed between initial and refined phases.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RankChanges {
    /// Documents that moved up in ranking after refinement.
    pub promoted: usize,
    /// Documents that moved down in ranking after refinement.
    pub demoted: usize,
    /// Documents whose rank did not change.
    pub stable: usize,
}

impl RankChanges {
    /// Total number of documents tracked.
    #[must_use]
    pub const fn total(&self) -> usize {
        self.promoted
            .saturating_add(self.demoted)
            .saturating_add(self.stable)
    }
}

/// Progressive search phases for three-tier display.
///
/// The iterator contract:
/// 1. Always yields `Initial` first (~15ms).
/// 2. Then yields either `Refined` or `RefinementFailed` (never both).
/// 3. Optionally yields `Reranked` if a cross-encoder is configured.
/// 4. Iterator is fused after yielding final phase (`next()` returns `None`).
///
/// Consumers can stop after `Initial` if latency-sensitive.
///
/// # Example
///
/// ```rust,ignore
/// for phase in searcher.search("distributed consensus", 10) {
///     match phase {
///         SearchPhase::Initial { results, .. } => display_immediately(&results),
///         SearchPhase::Refined { results, .. } => update_display(&results),
///         SearchPhase::Reranked { results, .. } => update_display_with_final_scores(&results),
///         SearchPhase::RefinementFailed { initial_results, error, .. } => {
///             // Keep showing initial results, log the error
///             log_warning(&error);
///         }
///     }
/// }
/// ```
#[derive(Debug)]
pub enum SearchPhase {
    /// Fast-tier results ready for immediate display.
    ///
    /// Contains RRF-fused results from fast embedding + BM25.
    /// Scores are RRF values (~0.01-0.03 range), sorted descending
    /// with deterministic tie-breaking.
    Initial {
        /// Fast-tier search results.
        results: Vec<ScoredResult>,
        /// Time elapsed for this phase.
        latency: Duration,
        /// Diagnostic metrics for this phase.
        metrics: PhaseMetrics,
    },

    /// Quality-refined results ready to replace initial display.
    ///
    /// Contains blended scores (0.7 quality + 0.3 fast by default).
    /// Results may have different ordering than `Initial`.
    Refined {
        /// Quality-refined search results.
        results: Vec<ScoredResult>,
        /// Time elapsed for this phase.
        latency: Duration,
        /// Diagnostic metrics for this phase.
        metrics: PhaseMetrics,
        /// How rankings changed compared to Initial.
        rank_changes: RankChanges,
    },

    /// Final reranked results from cross-encoder inference.
    ///
    /// Contains scores produced by a cross-encoder model processing
    /// (query, doc) pairs directly. Most accurate but slowest phase.
    Reranked {
        /// Reranked search results.
        results: Vec<ScoredResult>,
        /// Time elapsed for this phase (including earlier refinement).
        latency: Duration,
        /// Diagnostic metrics for this phase.
        metrics: PhaseMetrics,
    },

    /// Quality refinement failed; initial results remain valid.
    ///
    /// This is NOT an error state -- it is graceful degradation.
    /// The consumer should display `initial_results` and log the error.
    RefinementFailed {
        /// The original `Initial` results, carried forward unchanged.
        initial_results: Vec<ScoredResult>,
        /// Why refinement failed (timeout, model error, etc.).
        error: SearchError,
        /// How long we waited before failing.
        latency: Duration,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::EmbeddingArtifactIdentityV1;
    use proptest::prelude::*;

    fn vhit(index: u32, score: f32, doc_id: &str) -> VectorHit {
        VectorHit {
            index,
            score,
            doc_id: doc_id.into(),
        }
    }

    fn identity(name: &str, dim: u32) -> EmbeddingIdentityBundleV1 {
        EmbeddingIdentityBundleV1::explicit_test_model(name, dim)
    }

    fn test_digest(nibble: char) -> String {
        assert!(nibble.is_ascii_hexdigit() && !nibble.is_ascii_uppercase());
        nibble.to_string().repeat(64)
    }

    fn semantic_identity(name: &str, dim: u32) -> EmbeddingIdentityBundleV1 {
        let mut bundle = identity(name, dim);
        bundle.space.kind = EmbeddingSpaceKindV1::Semantic;
        bundle.space.hash_control = None;
        bundle.space.artifact_manifest_fingerprint = test_digest('a');
        bundle.space.artifacts = vec![EmbeddingArtifactIdentityV1 {
            role: "weights".to_owned(),
            sha256: test_digest('b'),
            size: 1,
        }];
        bundle.producer.space_fingerprint = bundle.space.fingerprint();
        bundle.validate().expect("semantic test identity validates");
        bundle
    }

    fn test_scope(generation: char, corpus: char) -> CoverageScopeV1 {
        CoverageScopeV1::new(test_digest(generation), test_digest(corpus))
            .expect("test scope validates")
    }

    fn members(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    fn verified_documents(
        tier: CoverageTierV1,
        bundle: &EmbeddingIdentityBundleV1,
        live: &[&str],
        indexed: &[&str],
    ) -> TierCoverageV1 {
        TierCoverageV1::verified(
            tier,
            &bundle.space,
            CoverageBasisV1::CanonicalDocuments,
            test_scope('1', '2'),
            members(live),
            test_scope('1', '2'),
            members(indexed),
            None,
        )
        .expect("same-scope document coverage validates")
    }

    fn verified_documents_context(
        tier: CoverageTierV1,
        bundle: &EmbeddingIdentityBundleV1,
        live: &[&str],
        indexed: &[&str],
    ) -> TrustedTierCoverageContextV1 {
        TrustedTierCoverageContextV1::verified(
            tier,
            bundle.space.clone(),
            CoverageBasisV1::CanonicalDocuments,
            test_scope('1', '2'),
            members(live),
            test_scope('1', '2'),
            members(indexed),
            None,
        )
        .expect("same-scope document context validates")
    }

    fn bound(bundle: EmbeddingIdentityBundleV1) -> BoundQueryEmbedding {
        BoundQueryEmbedding::new(vec![0.25; bundle.space.dimension as usize], bundle)
            .expect("test query binding validates")
    }

    #[test]
    fn bound_embedding_rejects_dimension_mismatch_at_bind_time() {
        let error = BoundQueryEmbedding::new(vec![0.5; 7], identity("fast-model", 8))
            .expect_err("7-dim vector cannot bind an 8-dim space");
        assert!(matches!(
            error,
            SearchError::DimensionMismatch {
                expected: 8,
                found: 7
            }
        ));
    }

    #[test]
    fn bound_embedding_matches_identity_bound_validation_and_redacts_debug() {
        let mut malformed = identity("malformed-bundle", 8);
        malformed.producer.space_fingerprint = test_digest('f');
        assert!(BoundQueryEmbedding::new(vec![0.0; 8], malformed).is_err());

        assert!(BoundQueryEmbedding::new(Vec::new(), identity("zero-dimension", 0)).is_err());

        let mut persisted = identity("persisted-storage", 8);
        persisted.storage.format = "fsvi-v2".to_owned();
        persisted.storage.endianness = "little-endian".to_owned();
        persisted
            .validate()
            .expect("persisted bundle is valid identity, but not an in-memory vector contract");
        assert!(BoundQueryEmbedding::new(vec![0.0; 8], persisted).is_err());

        let bound = BoundQueryEmbedding::new(
            vec![0.123_456, 0.654_321, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            identity("debug-private-model", 8),
        )
        .expect("valid in-memory binding");
        let diagnostic = format!("{bound:?}");
        assert!(!diagnostic.contains("0.123456"));
        assert!(!diagnostic.contains("0.654321"));
        assert!(!diagnostic.contains("debug-private-model"));
        assert!(diagnostic.contains("dimension: 8"));
    }

    #[test]
    fn same_dimension_wrong_space_is_rejected_by_fingerprint() {
        // The defining bd-9xuj case: identical dimensions, different models.
        // A raw &[f32] API cannot tell these apart; the bound embedding must.
        let fast =
            BoundQueryEmbedding::new(vec![0.5; 8], identity("fast-model", 8)).expect("bind fast");
        let quality_space = identity("quality-model", 8);
        assert_ne!(
            fast.identity_fingerprint(),
            quality_space.fingerprint(),
            "distinct models must have distinct fingerprints"
        );
        let error = fast
            .verify_space(&quality_space.fingerprint(), "quality")
            .expect_err("fast vector must not enter the quality space");
        assert!(
            matches!(&error, SearchError::InvalidConfig { .. }),
            "expected InvalidConfig, got {error:?}"
        );
        if let SearchError::InvalidConfig { field, reason, .. } = &error {
            assert_eq!(field, "query_embedding.quality.identity");
            assert!(reason.contains("different embedding space"));
        }
    }

    #[test]
    fn matching_space_verifies() {
        let bundle = identity("fast-model", 8);
        let expected = bundle.fingerprint();
        let bound = BoundQueryEmbedding::new(vec![0.5; 8], bundle).expect("bind");
        bound
            .verify_space(&expected, "fast")
            .expect("same space must verify");
        assert_eq!(bound.vector().len(), 8);
    }

    // ─── bd-9xuj T2-C1 (rebuild): bind-time validation + corrected admission law ───

    #[test]
    fn bind_time_validation_rejects_incoherent_identity_bundle() -> Result<(), String> {
        // The bundle must prove itself before it can prove anything to a
        // consumer: this producer's space_fingerprint no longer binds the
        // bundled space, so binding must surface the bundle's own typed
        // validation error instead of accepting the incoherent claim.
        let mut broken = identity("bind-validate-model", 8);
        broken.producer.space_fingerprint = broken.producer.golden_vectors.corpus_sha256.clone();
        let error = BoundQueryEmbedding::new(vec![0.5; 8], broken)
            .expect_err("an incoherent identity bundle must not bind");
        let rendered = format!("{error:?}");
        let SearchError::InvalidConfig { field, .. } = error else {
            return Err(format!("expected InvalidConfig, got {rendered}"));
        };
        assert_eq!(field, "embedding_identity.producer.space_fingerprint");
        Ok(())
    }

    /// Pins the T2-C1 impedance mismatch (readiness map §0.1): a query-side
    /// bundle binds `in-memory-*` storage while an index-side bundle binds
    /// `fsvi-v2`, so for the SAME mathematical space the full-bundle
    /// fingerprints can never match — `verify_space` fails closed on
    /// legitimate traffic at every index seam, while the space-scoped
    /// `verify_space_identity` joins on the space fingerprint and admits it.
    #[test]
    fn space_scoped_verify_joins_across_storage_formats() {
        // Query side: explicit test identity, in-memory f32 storage.
        let query_side = identity("shared-model", 8);
        // Index side: the SAME space, persisted as fsvi-v2 little-endian.
        let mut index_side = identity("shared-model", 8);
        index_side.storage.format = "fsvi-v2".to_owned();
        index_side.storage.endianness = "little-endian".to_owned();
        index_side
            .validate()
            .expect("index-side bundle must be a legitimate, validating bundle");

        assert_eq!(
            query_side.space.fingerprint(),
            index_side.space.fingerprint(),
            "same model + dimension is the same mathematical space"
        );
        assert_ne!(
            query_side.fingerprint(),
            index_side.fingerprint(),
            "storage difference must alter the full-bundle fingerprint"
        );

        let bound = BoundQueryEmbedding::new(vec![0.5; 8], query_side).expect("bind");
        assert_eq!(
            bound.space_fingerprint(),
            bound.identity().space.fingerprint(),
            "bind-time space fingerprint must match the bundled space"
        );

        // Full-bundle verify: fails closed on this legitimate pairing (§0.1).
        bound
            .verify_space(&index_side.fingerprint(), "quality")
            .expect_err("full-bundle fingerprints never match across storage formats");
        // Space-scoped verify: the correct join key at the index seam.
        bound
            .verify_space_identity(&index_side.space.fingerprint(), "quality")
            .expect("same space must verify space-scoped across storage formats");
    }

    /// Pins the T2-C1 impedance mismatch (readiness map §0.1) under the
    /// corrected admission law: same space, same producer, different
    /// storage. The full-bundle fingerprints can never match across an
    /// index seam, the space fingerprint joins, and admission is granted as
    /// typed [`SpaceIdentityAdmission::SameProducer`] — not because "same
    /// space is sufficient".
    #[test]
    fn same_producer_admits_across_storage_formats() {
        // Query side: explicit test identity, in-memory f32 storage.
        let query_side = identity("shared-model", 8);
        // Index side: the SAME space and producer, persisted as fsvi-v2.
        let mut index_side = identity("shared-model", 8);
        index_side.storage.format = "fsvi-v2".to_owned();
        index_side.storage.endianness = "little-endian".to_owned();
        index_side
            .validate()
            .expect("index-side bundle must be a legitimate, validating bundle");

        assert_eq!(
            query_side.space.fingerprint(),
            index_side.space.fingerprint(),
            "same model + dimension is the same mathematical space"
        );
        assert_ne!(
            query_side.fingerprint(),
            index_side.fingerprint(),
            "storage difference must alter the full-bundle fingerprint"
        );

        let bound = BoundQueryEmbedding::new(vec![0.5; 8], query_side).expect("bind");
        assert_eq!(
            bound.space_fingerprint(),
            bound.identity().space.fingerprint(),
            "bind-time space fingerprint must match the bundled space"
        );

        // Full-bundle verify: fails closed on this legitimate pairing (§0.1)...
        bound
            .verify_space(&index_side.fingerprint(), "quality")
            .expect_err("full-bundle fingerprints never match across storage formats");
        // ...the space fingerprint joins at the seam...
        bound
            .verify_space_identity(&index_side.space.fingerprint(), "quality")
            .expect("same space must join space-scoped across storage formats");
        // ...and the admission law grants the pairing as the SAME producer.
        let admission = bound
            .verify_producer_conformance(&index_side, "quality")
            .expect("same producer must be admitted");
        assert_eq!(admission, SpaceIdentityAdmission::SameProducer);
        assert_eq!(admission.code(), "same_producer");
    }

    #[test]
    fn same_dimension_wrong_space_is_rejected_by_space_fingerprint() -> Result<(), String> {
        // The defining bd-9xuj case through the seam verifier: identical
        // dimensions, different models. Dimension checks cannot tell these
        // apart; the space fingerprint must.
        let bound =
            BoundQueryEmbedding::new(vec![0.5; 8], identity("fast-model", 8)).expect("bind fast");
        let quality_space = identity("quality-model", 8);
        let error = bound
            .verify_space_identity(&quality_space.space.fingerprint(), "quality")
            .expect_err("fast vector must not enter the quality space");
        let rendered = format!("{error:?}");
        let SearchError::InvalidConfig {
            field,
            value,
            reason,
        } = error
        else {
            return Err(format!("expected InvalidConfig, got {rendered}"));
        };
        assert_eq!(field, "query_embedding.quality.space_identity");
        assert_eq!(value, bound.space_fingerprint());
        assert!(reason.contains("different embedding space"));
        assert!(reason.contains(&quality_space.space.fingerprint()));

        // The bundle-level admission law rejects the same pairing at the
        // space join, before producer conformance is even considered.
        let error = bound
            .verify_producer_conformance(&quality_space, "quality")
            .expect_err("wrong space must reject at the bundle level too");
        let rendered = format!("{error:?}");
        let SearchError::InvalidConfig { field, .. } = error else {
            return Err(format!("expected InvalidConfig, got {rendered}"));
        };
        assert_eq!(field, "query_embedding.quality.space_identity");
        Ok(())
    }

    /// The corrected admission law (bd-9xuj map, post-review supersession
    /// notice): a foreign producer in the same space is admitted ONLY when
    /// both producers carry the byte-identical pinned golden-vector
    /// certificate, with typed
    /// [`SpaceIdentityAdmission::ConformanceCompatibleProducer`] telemetry;
    /// otherwise it is rejected. Matching space fingerprints alone never
    /// admit a pairing.
    #[test]
    fn foreign_producer_admission_requires_conformance_certificate() -> Result<(), String> {
        let base = identity("conformance-model", 8);
        let bound = BoundQueryEmbedding::new(vec![0.5; 8], base.clone()).expect("bind");

        // Same space, different producer, SAME golden-vector certificate:
        // certified conformance-compatible → admitted, telemetry-typed.
        let mut certified = base.clone();
        certified.producer.backend = "alternate-conformant-backend".to_owned();
        certified
            .validate()
            .expect("certified sibling must validate");
        assert_ne!(
            certified.producer.fingerprint(),
            base.producer.fingerprint(),
            "a different backend attestation must alter the producer fingerprint"
        );
        let admission = bound
            .verify_producer_conformance(&certified, "fast")
            .expect("certified foreign producer must be admitted");
        assert_eq!(admission.code(), "conformance_compatible_producer");
        let rendered = format!("{admission:?}");
        let SpaceIdentityAdmission::ConformanceCompatibleProducer {
            query_producer_fingerprint,
            expected_producer_fingerprint,
        } = admission
        else {
            return Err(format!(
                "expected ConformanceCompatibleProducer, got {rendered}"
            ));
        };
        assert_eq!(query_producer_fingerprint, base.producer.fingerprint());
        assert_eq!(
            expected_producer_fingerprint,
            certified.producer.fingerprint()
        );

        // Same space, different producer, DIFFERENT golden-vector
        // certificate: same-space is not sufficient — typed rejection.
        let mut uncertified = base.clone();
        uncertified.producer.backend = "alternate-uncertified-backend".to_owned();
        uncertified.producer.golden_vectors.vectors_sha256 = base.space.fingerprint();
        uncertified
            .validate()
            .expect("uncertified sibling must still be a coherent bundle");
        let error = bound
            .verify_producer_conformance(&uncertified, "fast")
            .expect_err("uncertified foreign producer must be rejected");
        let rendered = format!("{error:?}");
        let SearchError::InvalidConfig {
            field,
            value,
            reason,
        } = error
        else {
            return Err(format!("expected InvalidConfig, got {rendered}"));
        };
        assert_eq!(field, "query_embedding.fast.producer_conformance");
        assert_eq!(value, base.producer.fingerprint());
        assert!(reason.contains("golden-vector certificate"));
        Ok(())
    }

    /// Pins the KNOWN GAP the review (#8151) named, so it is documented
    /// rather than hidden: two producers whose golden-vector certificates
    /// differ ONLY in the corpus digest were certified against different
    /// golden corpora, so their conformance is not comparable at all —
    /// yet today the admission law rejects them through the same single
    /// reject arm as a same-corpus conformance failure. The rejection is
    /// correct but VACUOUS: no typed `Unattestable` classification exists
    /// yet to distinguish "cannot be attested" from "attested and failed".
    /// When that third [`SpaceIdentityAdmission`] arm lands (the enum is
    /// `#[non_exhaustive]` for exactly that addition), this test pins the
    /// behavior it replaces.
    #[test]
    fn different_corpus_digest_certificates_reject_vacuously_today() -> Result<(), String> {
        let base = identity("corpus-digest-model", 8);
        let bound = BoundQueryEmbedding::new(vec![0.5; 8], base.clone()).expect("bind");

        // Expected side differs from the query side ONLY in the golden
        // corpus digest: a certificate genuinely pinned to a different
        // (equally valid) golden corpus.
        let mut foreign_corpus = base.clone();
        foreign_corpus.producer.golden_vectors.corpus_sha256 =
            crate::generation::GoldenVectorCertificateV1::corpus_fingerprint(&[
                "a different golden corpus",
            ])
            .expect("corpus fingerprint");
        foreign_corpus
            .validate()
            .expect("a certificate pinned to another corpus is still a coherent bundle");
        assert_ne!(
            foreign_corpus.producer.golden_vectors.corpus_sha256,
            base.producer.golden_vectors.corpus_sha256,
            "the corpus digests must differ"
        );
        assert_eq!(
            foreign_corpus.producer.golden_vectors.vectors_sha256,
            base.producer.golden_vectors.vectors_sha256,
            "everything but the corpus digest must be identical"
        );
        assert_eq!(
            foreign_corpus.producer.golden_vectors.vector_count,
            base.producer.golden_vectors.vector_count
        );
        assert_eq!(
            foreign_corpus.producer.golden_vectors.dimension,
            base.producer.golden_vectors.dimension
        );

        // Today: rejected through the single producer-conformance reject
        // arm — indistinguishable from a same-corpus conformance failure.
        let error = bound
            .verify_producer_conformance(&foreign_corpus, "fast")
            .expect_err("a different-corpus certificate must not be admitted");
        let rendered = format!("{error:?}");
        let SearchError::InvalidConfig { field, .. } = error else {
            return Err(format!("expected InvalidConfig, got {rendered}"));
        };
        assert_eq!(field, "query_embedding.fast.producer_conformance");
        Ok(())
    }

    #[test]
    fn distinct_models_at_equal_dimension_always_reject_space_scoped() {
        // Readiness-map §3.3(10) property at C1 scope: explicit_test_model(a, d)
        // vs (b, d) with a != b must always reject, for every tier label —
        // at the space join and under the full admission law alike.
        let dims = [8_u32, 384];
        let models = ["model-a", "model-b", "model-c"];
        for dim in dims {
            for (i, a) in models.iter().enumerate() {
                for b in &models[i + 1..] {
                    let bound =
                        BoundQueryEmbedding::new(vec![0.25; dim as usize], identity(a, dim))
                            .expect("bind");
                    let other = identity(b, dim);
                    for tier in ["fast", "quality"] {
                        bound
                            .verify_space_identity(&other.space.fingerprint(), tier)
                            .expect_err("distinct models must never share a space");
                        bound
                            .verify_producer_conformance(&other, tier)
                            .expect_err("distinct models must never be admitted");
                    }
                    // Reflexive: a model always verifies against its own space.
                    bound
                        .verify_space_identity(&identity(a, dim).space.fingerprint(), "fast")
                        .expect("own space must verify");
                }
            }
        }
    }

    #[test]
    fn tiered_constructors_retain_independent_bindings() {
        let fast = BoundQueryEmbedding::new(vec![0.1; 8], identity("fast-model", 8)).unwrap();
        let quality =
            BoundQueryEmbedding::new(vec![0.2; 16], identity("quality-model", 16)).unwrap();

        let progressive = TieredQueryEmbeddings::progressive(fast.clone(), quality.clone());
        assert!(progressive.fast().is_some() && progressive.quality().is_some());

        let fast_only = TieredQueryEmbeddings::fast_only(fast);
        assert!(fast_only.fast().is_some() && fast_only.quality().is_none());
        let quality_only = TieredQueryEmbeddings::quality_only(quality);
        assert!(quality_only.fast().is_none() && quality_only.quality().is_some());
    }

    #[test]
    fn space_identity_admission_codes_and_serde_are_stable() {
        let same = SpaceIdentityAdmission::SameProducer;
        assert_eq!(same.code(), "same_producer");
        let json = serde_json::to_string(&same).expect("serialize");
        assert!(json.contains("\"admission\":\"same_producer\""));

        let compat = SpaceIdentityAdmission::ConformanceCompatibleProducer {
            query_producer_fingerprint: "a".repeat(64),
            expected_producer_fingerprint: "b".repeat(64),
        };
        assert_eq!(compat.code(), "conformance_compatible_producer");
        let json = serde_json::to_string(&compat).expect("serialize");
        assert!(json.contains("\"admission\":\"conformance_compatible_producer\""));
        let back: SpaceIdentityAdmission = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, compat);
    }

    #[test]
    fn tiered_constructors_report_supported_topology() {
        let fast =
            BoundQueryEmbedding::new(vec![0.1; 8], semantic_identity("fast-model", 8)).unwrap();
        let quality =
            BoundQueryEmbedding::new(vec![0.2; 16], semantic_identity("quality-model", 16))
                .unwrap();

        let progressive = TieredQueryEmbeddings::progressive(fast.clone(), quality.clone());
        assert_eq!(
            progressive.supported_topology(),
            RetrievalTopology::FullProgressive
        );

        assert_eq!(
            TieredQueryEmbeddings::fast_only(fast).supported_topology(),
            RetrievalTopology::FastOnly
        );
        assert_eq!(
            TieredQueryEmbeddings::quality_only(quality).supported_topology(),
            RetrievalTopology::QualityOnly
        );
    }

    /// bd-ctzo C4: a hash-control space can never report a semantic topology.
    ///
    /// This test previously did not exist, and the one above asserted
    /// `FullProgressive` for two `explicit_test_model` bundles — which are
    /// `HashControl` spaces. Every fixture in the tree was therefore claiming
    /// semantic availability it did not have.
    #[test]
    fn a_hash_control_space_can_never_report_a_semantic_topology() {
        let hash_fast = BoundQueryEmbedding::new(vec![0.1; 8], identity("hash-fast", 8)).unwrap();
        let hash_quality =
            BoundQueryEmbedding::new(vec![0.2; 8], identity("hash-quality", 8)).unwrap();
        let semantic_fast =
            BoundQueryEmbedding::new(vec![0.1; 8], semantic_identity("real-fast", 8)).unwrap();
        let semantic_quality =
            BoundQueryEmbedding::new(vec![0.2; 8], semantic_identity("real-quality", 8)).unwrap();

        for (label, embeddings) in [
            (
                "both hash",
                TieredQueryEmbeddings::progressive(hash_fast.clone(), hash_quality.clone()),
            ),
            (
                "fast hash only",
                TieredQueryEmbeddings::fast_only(hash_fast.clone()),
            ),
            (
                "quality hash only",
                TieredQueryEmbeddings::quality_only(hash_quality.clone()),
            ),
            // A MIXED pair is still not semantic: a topology describes the
            // whole retrieval, not its strongest arm.
            (
                "semantic fast + hash quality",
                TieredQueryEmbeddings::progressive(semantic_fast.clone(), hash_quality),
            ),
            (
                "hash fast + semantic quality",
                TieredQueryEmbeddings::progressive(hash_fast, semantic_quality.clone()),
            ),
        ] {
            assert_eq!(
                embeddings.supported_topology(),
                RetrievalTopology::HashControl,
                "{label} must not claim a semantic topology"
            );
            assert!(embeddings.binds_hash_control(), "{label}");
        }

        // Control: two learned spaces DO report a semantic topology, so the
        // guard above is not simply answering HashControl for everything.
        let semantic = TieredQueryEmbeddings::progressive(semantic_fast, semantic_quality);
        assert_eq!(
            semantic.supported_topology(),
            RetrievalTopology::FullProgressive
        );
        assert!(!semantic.binds_hash_control());
    }

    /// bd-ctzo C4: unknown coverage has no number, so it cannot be reported
    /// as zero.
    #[test]
    fn unknown_tier_coverage_cannot_be_read_as_a_zero_count() {
        let unknown = TierQueryCoverageV1::Unknown {
            reason: CoverageUnknownReasonV1::LegacyUnidentified,
        };
        let empty_but_live = TierQueryCoverageV1::Witnessed {
            generation_sequence: 7,
            live_count: 0,
            contributed_candidates: 0,
        };
        assert_eq!(unknown.witnessed_live_count(), None);
        assert_eq!(empty_but_live.witnessed_live_count(), Some(0));
        assert_ne!(
            unknown, empty_but_live,
            "an unwitnessed tier and a genuinely empty one must not compare equal"
        );

        let coverage = SearchCoverageV1::new(
            RetrievalTopology::FastOnly,
            unknown,
            TierQueryCoverageV1::NotRequested,
        );
        let summary = coverage.redacted_summary();
        assert!(summary.contains("fast=unknown"), "{summary}");
        assert!(summary.contains("quality=not_requested"), "{summary}");
        // The summary is built from closed codes and integers only; there is
        // no field on the type through which a doc id or vector could reach it.
        assert!(!summary.contains("doc"), "{summary}");
    }

    #[test]
    fn topology_codes_are_stable_and_semantic_partition_is_correct() {
        let all = [
            RetrievalTopology::LexicalOnly,
            RetrievalTopology::HashControl,
            RetrievalTopology::FastOnly,
            RetrievalTopology::QualityOnly,
            RetrievalTopology::FullProgressive,
            RetrievalTopology::PartialQuality {
                coverage_ppm: 750_000,
            },
        ];
        let codes: Vec<&str> = all.iter().map(RetrievalTopology::code).collect();
        assert_eq!(
            codes,
            vec![
                "lexical_only",
                "hash_control",
                "fast_only",
                "quality_only",
                "full_progressive",
                "partial_quality",
            ]
        );
        for topology in all {
            let semantic = topology.is_semantic();
            match topology {
                RetrievalTopology::LexicalOnly | RetrievalTopology::HashControl => {
                    assert!(!semantic, "{topology:?} must not claim semantic");
                }
                _ => assert!(semantic, "{topology:?} is semantic"),
            }
        }
    }

    #[test]
    fn topology_serde_roundtrips_with_coverage() {
        let topology = RetrievalTopology::PartialQuality {
            coverage_ppm: 333_333,
        };
        let json = serde_json::to_string(&topology).expect("serialize");
        assert!(json.contains("partial_quality"));
        assert!(json.contains("333333"));
        let decoded: RetrievalTopology = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(decoded, topology);
    }

    #[test]
    fn coverage_relations_and_checked_counts_are_derived_from_unique_members() {
        let bundle = semantic_identity("coverage-relations", 8);
        let cases = [
            (
                members(&[]),
                members(&[]),
                CoverageRelationV1::ZeroLive,
                [0, 0, 0, 0, 0],
            ),
            (
                members(&["a", "b"]),
                members(&["b", "a"]),
                CoverageRelationV1::Complete,
                [2, 2, 2, 2, u64::from(COMPLETE_COVERAGE_PPM)],
            ),
            (
                members(&["a", "b", "c"]),
                members(&["a", "c"]),
                CoverageRelationV1::Subset,
                [3, 2, 2, 3, 666_666],
            ),
            (
                members(&["a", "b"]),
                members(&["a", "b", "c"]),
                CoverageRelationV1::Superset,
                [2, 3, 2, 3, u64::from(COMPLETE_COVERAGE_PPM)],
            ),
            (
                members(&["a", "b"]),
                members(&["b", "c"]),
                CoverageRelationV1::Overlap,
                [2, 2, 1, 3, 500_000],
            ),
            (
                members(&["a"]),
                members(&["b"]),
                CoverageRelationV1::Disjoint,
                [1, 1, 0, 2, 0],
            ),
            (
                members(&["a"]),
                members(&[]),
                CoverageRelationV1::Subset,
                [1, 0, 0, 1, 0],
            ),
        ];

        for (live, indexed, expected_relation, expected_counts) in cases {
            let coverage = TierCoverageV1::verified(
                CoverageTierV1::Fast,
                &bundle.space,
                CoverageBasisV1::CanonicalDocuments,
                test_scope('1', '2'),
                live,
                test_scope('1', '2'),
                indexed,
                None,
            )
            .expect("set relation derives");
            let facts = coverage.verified_facts();
            assert!(facts.is_some());
            if let Some((relation, counts)) = facts {
                assert_eq!(relation, expected_relation);
                assert_eq!(counts.live_count(), expected_counts[0]);
                assert_eq!(counts.indexed_count(), expected_counts[1]);
                assert_eq!(counts.intersection_count(), expected_counts[2]);
                assert_eq!(counts.union_count(), expected_counts[3]);
                assert_eq!(u64::from(counts.coverage_ppm()), expected_counts[4]);
            }
        }
    }

    #[test]
    fn canonical_documents_ignore_order_but_ordered_shards_bind_order() {
        let bundle = semantic_identity("witness-order", 8);
        let documents_a = verified_documents(
            CoverageTierV1::Fast,
            &bundle,
            &["doc-c", "doc-a", "doc-b"],
            &["doc-a"],
        );
        let documents_b = verified_documents(
            CoverageTierV1::Fast,
            &bundle,
            &["doc-b", "doc-c", "doc-a"],
            &["doc-a"],
        );
        assert_eq!(documents_a.fingerprint(), documents_b.fingerprint());

        let shards_a = TierCoverageV1::verified(
            CoverageTierV1::Fast,
            &bundle.space,
            CoverageBasisV1::OrderedShards,
            test_scope('1', '2'),
            members(&["shard-a", "shard-b"]),
            test_scope('1', '2'),
            members(&["shard-a"]),
            None,
        )
        .expect("ordered shard coverage validates");
        let shards_b = TierCoverageV1::verified(
            CoverageTierV1::Fast,
            &bundle.space,
            CoverageBasisV1::OrderedShards,
            test_scope('1', '2'),
            members(&["shard-b", "shard-a"]),
            test_scope('1', '2'),
            members(&["shard-a"]),
            None,
        )
        .expect("reordered shard coverage validates");
        assert_ne!(shards_a.fingerprint(), shards_b.fingerprint());
        assert_eq!(
            shards_a
                .live_witness()
                .map(CoverageWitnessV1::set_fingerprint),
            shards_b
                .live_witness()
                .map(CoverageWitnessV1::set_fingerprint),
            "set facts stay comparable while ordered membership identity changes"
        );
        assert_ne!(
            shards_a
                .live_witness()
                .map(CoverageWitnessV1::membership_fingerprint),
            shards_b
                .live_witness()
                .map(CoverageWitnessV1::membership_fingerprint)
        );
    }

    #[test]
    fn duplicate_members_never_inflate_document_or_shard_coverage() {
        let bundle = semantic_identity("duplicate-rejection", 8);
        for basis in [
            CoverageBasisV1::CanonicalDocuments,
            CoverageBasisV1::OrderedShards,
        ] {
            let error = TierCoverageV1::verified(
                CoverageTierV1::Fast,
                &bundle.space,
                basis,
                test_scope('1', '2'),
                members(&["same", "same"]),
                test_scope('1', '2'),
                members(&["same"]),
                None,
            )
            .expect_err("duplicate members fail closed");
            assert!(error.to_string().contains("duplicate member"));
        }
    }

    #[test]
    fn empty_and_oversized_member_diagnostics_are_bounded_and_redacted() {
        let bundle = semantic_identity("bounded-diagnostics", 8);
        for bad_member in [String::new(), "sensitive-query-document".repeat(300)] {
            let error = TierCoverageV1::verified(
                CoverageTierV1::Fast,
                &bundle.space,
                CoverageBasisV1::CanonicalDocuments,
                test_scope('1', '2'),
                vec![bad_member.clone()],
                test_scope('1', '2'),
                Vec::new(),
                None,
            )
            .expect_err("invalid member fails closed");
            let diagnostic = error.to_string();
            assert!(diagnostic.contains("redacted-member-at-index-0"));
            if !bad_member.is_empty() {
                assert!(!diagnostic.contains(&bad_member));
            }
            assert!(diagnostic.len() < 512);
        }
    }

    #[test]
    fn malformed_digest_diagnostics_never_echo_caller_material() {
        let caller_material = "caller-controlled-model-path-fragment".repeat(400);
        let error = CoverageScopeV1::new(caller_material.clone(), test_digest('2'))
            .expect_err("malformed generation digest fails closed");
        let diagnostic = error.to_string();
        assert!(diagnostic.contains("redacted-invalid-sha256-length"));
        assert!(!diagnostic.contains(&caller_material));
        assert!(diagnostic.len() < 512);
    }

    #[test]
    fn maximum_count_and_fixed_point_boundaries_are_overflow_safe() {
        let complete = CoverageCountsV1::checked(u64::MAX, u64::MAX, u64::MAX, u64::MAX)
            .expect("maximum complete counts fit");
        assert_eq!(complete.coverage_ppm(), COMPLETE_COVERAGE_PPM);

        let one_of_three = CoverageCountsV1::checked(3, 1, 1, 3)
            .expect("one-third counts are algebraically valid");
        assert_eq!(one_of_three.coverage_ppm(), 333_333);

        let floor_boundary = CoverageCountsV1::checked(u64::MAX, 1, 1, u64::MAX)
            .expect("large denominator uses a safe u128 intermediate");
        assert_eq!(floor_boundary.coverage_ppm(), 0);

        assert!(CoverageCountsV1::checked(u64::MAX, u64::MAX, 0, u64::MAX).is_err());
        assert!(CoverageCountsV1::checked(1, 1, 2, 0).is_err());
        assert!(CoverageCountsV1::checked(2, 2, 1, 2).is_err());
    }

    #[test]
    fn missing_unknown_unverified_and_zero_live_are_distinct_closed_states() {
        let bundle = semantic_identity("closed-states", 8);
        let missing = TierCoverageV1::not_requested(CoverageTierV1::Fast);
        let requested_missing = TierCoverageV1::requested_missing(
            CoverageTierV1::Fast,
            &bundle.space,
            CoverageBasisV1::CanonicalDocuments,
            test_scope('1', '2'),
            members(&["a"]),
        )
        .expect("requested missing tier validates");
        let unknown = TierCoverageV1::unknown(
            CoverageTierV1::Fast,
            &bundle.space,
            CoverageBasisV1::CanonicalDocuments,
            test_scope('1', '2'),
            members(&["a"]),
            CoverageUnknownReasonV1::OwnerUnavailable,
        )
        .expect("unknown tier validates");
        let unverified = TierCoverageV1::unverified(
            CoverageTierV1::Fast,
            &bundle.space,
            CoverageBasisV1::CanonicalDocuments,
            test_scope('1', '2'),
            members(&["a"]),
            CoverageUnverifiedReasonV1::CountsUntrusted,
            Some(9),
        )
        .expect("unverified tier validates");
        let zero_live = verified_documents(CoverageTierV1::Fast, &bundle, &[], &[]);

        assert!(matches!(
            missing.requested(),
            TierCoverageRequestV1::MissingTier
        ));
        assert!(matches!(
            requested_missing.realized(),
            TierCoverageRealizationV1::MissingTier
        ));
        assert!(matches!(
            unknown.realized(),
            TierCoverageRealizationV1::Unknown { .. }
        ));
        assert!(matches!(
            unverified.realized(),
            TierCoverageRealizationV1::Unverified { .. }
        ));
        assert_eq!(
            zero_live.verified_facts().map(|facts| facts.0),
            Some(CoverageRelationV1::ZeroLive)
        );

        let fingerprints = [
            missing.fingerprint(),
            requested_missing.fingerprint(),
            unknown.fingerprint(),
            unverified.fingerprint(),
            zero_live.fingerprint(),
        ];
        assert_eq!(fingerprints.iter().collect::<BTreeSet<_>>().len(), 5);
    }

    #[test]
    fn cross_scope_comparison_requires_exact_evidence_and_never_claims_complete() {
        let bundle = semantic_identity("cross-scope", 8);
        let no_evidence = TierCoverageV1::verified(
            CoverageTierV1::Quality,
            &bundle.space,
            CoverageBasisV1::CanonicalDocuments,
            test_scope('1', '2'),
            members(&["a", "b"]),
            test_scope('3', '2'),
            members(&["a"]),
            None,
        );
        assert!(no_evidence.is_err());

        let redundant_evidence = TierCoverageV1::verified(
            CoverageTierV1::Quality,
            &bundle.space,
            CoverageBasisV1::CanonicalDocuments,
            test_scope('1', '2'),
            members(&["a", "b"]),
            test_scope('1', '2'),
            members(&["a"]),
            Some(test_digest('9')),
        );
        assert!(redundant_evidence.is_err());

        let cross_scope_complete = TierCoverageV1::verified(
            CoverageTierV1::Quality,
            &bundle.space,
            CoverageBasisV1::CanonicalDocuments,
            test_scope('1', '2'),
            members(&["a", "b"]),
            test_scope('3', '2'),
            members(&["b", "a"]),
            Some(test_digest('9')),
        );
        assert!(cross_scope_complete.is_err());

        let partial = TierCoverageV1::verified(
            CoverageTierV1::Quality,
            &bundle.space,
            CoverageBasisV1::CanonicalDocuments,
            test_scope('1', '2'),
            members(&["a", "b"]),
            test_scope('3', '4'),
            members(&["a"]),
            Some(test_digest('9')),
        )
        .expect("explicit evidence admits a checked cross-scope partial relation");
        assert_eq!(
            partial.verified_facts().map(|facts| facts.0),
            Some(CoverageRelationV1::Subset)
        );
        assert!(partial.cross_scope_evidence().is_some());
    }

    #[test]
    fn cross_scope_evidence_mutation_and_witness_borrowing_fail_closed() {
        let bundle = semantic_identity("evidence-mutation", 8);
        let partial = TierCoverageV1::verified(
            CoverageTierV1::Quality,
            &bundle.space,
            CoverageBasisV1::CanonicalDocuments,
            test_scope('1', '2'),
            members(&["a", "b"]),
            test_scope('3', '2'),
            members(&["a"]),
            Some(test_digest('9')),
        )
        .expect("cross-scope partial validates");
        let mut mutated = serde_json::to_value(&partial).expect("serialize coverage");
        mutated["realized"]["cross_scope_evidence"]["live_set_fingerprint"] =
            serde_json::Value::String(test_digest('f'));
        let untrusted: UntrustedTierCoverageV1 =
            serde_json::from_value(mutated).expect("mutated claim remains syntactic");
        let context = TrustedTierCoverageContextV1::verified(
            CoverageTierV1::Quality,
            bundle.space.clone(),
            CoverageBasisV1::CanonicalDocuments,
            test_scope('1', '2'),
            members(&["a", "b"]),
            test_scope('3', '2'),
            members(&["a"]),
            Some(test_digest('9')),
        )
        .expect("trusted cross-scope context validates");
        assert!(untrusted.validate_against(&context).is_err());

        let fast = verified_documents(CoverageTierV1::Fast, &bundle, &["a", "b"], &["a"]);
        let mut borrowed = serde_json::to_value(&fast).expect("serialize fast coverage");
        borrowed["tier"] = serde_json::Value::String("quality".to_owned());
        let untrusted: UntrustedTierCoverageV1 =
            serde_json::from_value(borrowed).expect("borrowed claim remains syntactic");
        let fast_context =
            verified_documents_context(CoverageTierV1::Fast, &bundle, &["a", "b"], &["a"]);
        assert!(untrusted.validate_against(&fast_context).is_err());
    }

    #[test]
    fn pair_rejects_swapped_roles_and_tier_domains_make_equal_sets_distinct() {
        let bundle = semantic_identity("tier-domain", 8);
        let fast = verified_documents(CoverageTierV1::Fast, &bundle, &["a", "b"], &["a", "b"]);
        let quality =
            verified_documents(CoverageTierV1::Quality, &bundle, &["a", "b"], &["a", "b"]);
        assert_ne!(
            fast.live_witness().map(CoverageWitnessV1::fingerprint),
            quality.live_witness().map(CoverageWitnessV1::fingerprint),
            "tier role is identity material even for equal document sets"
        );
        assert!(TierCoveragePairV1::new(quality.clone(), fast.clone()).is_err());
        assert!(TierCoveragePairV1::new(fast, quality).is_ok());
    }

    #[test]
    fn complete_independent_semantic_tiers_derive_full_progressive_topology() {
        let fast_bundle = semantic_identity("topology-fast", 8);
        let quality_bundle = semantic_identity("topology-quality", 16);
        let coverage = TierCoveragePairV1::new(
            verified_documents(CoverageTierV1::Fast, &fast_bundle, &["a", "b"], &["a", "b"]),
            verified_documents(
                CoverageTierV1::Quality,
                &quality_bundle,
                &["a", "b"],
                &["b", "a"],
            ),
        )
        .expect("independent coverage pair validates");
        let embeddings =
            TieredQueryEmbeddings::progressive(bound(fast_bundle), bound(quality_bundle));
        let decision = coverage
            .derive_topology(RetrievalIntentV1::Semantic, Some(&embeddings))
            .expect("validated complete bindings derive topology");
        assert_eq!(decision.requested(), RetrievalTopology::FullProgressive);
        assert_eq!(
            decision.coarse_realized(),
            Some(RetrievalTopology::FullProgressive)
        );
        assert!(decision.semantic_available());
    }

    #[test]
    fn partial_quality_is_derived_from_quality_intersection_not_a_scalar_claim() {
        let fast_bundle = semantic_identity("partial-fast", 8);
        let quality_bundle = semantic_identity("partial-quality", 16);
        let coverage = TierCoveragePairV1::new(
            verified_documents(
                CoverageTierV1::Fast,
                &fast_bundle,
                &["a", "b", "c"],
                &["a", "b", "c"],
            ),
            verified_documents(
                CoverageTierV1::Quality,
                &quality_bundle,
                &["a", "b", "c"],
                &["a", "c"],
            ),
        )
        .expect("partial quality pair validates");
        let embeddings =
            TieredQueryEmbeddings::progressive(bound(fast_bundle), bound(quality_bundle));
        let decision = coverage
            .derive_topology(RetrievalIntentV1::Semantic, Some(&embeddings))
            .expect("checked quality intersection derives partial topology");
        assert_eq!(decision.requested(), RetrievalTopology::FullProgressive);
        assert_eq!(
            decision.coarse_realized(),
            Some(RetrievalTopology::PartialQuality {
                coverage_ppm: 666_666
            })
        );
        assert!(decision.semantic_available());
    }

    #[test]
    fn complete_quality_only_derives_direct_quality_retrieval() {
        let quality_bundle = semantic_identity("direct-quality", 16);
        let pair = TierCoveragePairV1::new(
            TierCoverageV1::not_requested(CoverageTierV1::Fast),
            verified_documents(
                CoverageTierV1::Quality,
                &quality_bundle,
                &["a", "b"],
                &["a", "b"],
            ),
        )
        .expect("quality-only pair validates");
        let embeddings = TieredQueryEmbeddings::quality_only(bound(quality_bundle));
        let decision = pair
            .derive_topology(RetrievalIntentV1::Semantic, Some(&embeddings))
            .expect("direct quality binding derives quality-only retrieval");
        assert_eq!(decision.requested(), RetrievalTopology::QualityOnly);
        assert_eq!(
            decision.coarse_realized(),
            Some(RetrievalTopology::QualityOnly)
        );
        assert!(decision.semantic_available());
    }

    #[test]
    fn exact_realization_preserves_partial_fast_multi_partial_and_quality_only_partial() {
        let fast_bundle = semantic_identity("exact-partial-fast", 8);
        let quality_bundle = semantic_identity("exact-partial-quality", 16);

        let fast_only_pair = TierCoveragePairV1::new(
            verified_documents(CoverageTierV1::Fast, &fast_bundle, &["a", "b", "c"], &["a"]),
            TierCoverageV1::not_requested(CoverageTierV1::Quality),
        )
        .expect("partial fast-only pair validates");
        let fast_only = fast_only_pair
            .derive_topology(
                RetrievalIntentV1::Semantic,
                Some(&TieredQueryEmbeddings::fast_only(bound(
                    fast_bundle.clone(),
                ))),
            )
            .expect("partial fast remains an exact available tier");
        assert!(fast_only.semantic_available());
        assert!(fast_only.coarse_realized().is_none());
        assert!(matches!(
            fast_only.fast_realized(),
            RealizedTierCoverageV1::Partial { counts, .. }
                if counts.intersection_count() == 1
        ));
        assert!(matches!(
            fast_only.quality_realized(),
            RealizedTierCoverageV1::NotRequested
        ));

        let multi_pair = TierCoveragePairV1::new(
            verified_documents(CoverageTierV1::Fast, &fast_bundle, &["a", "b", "c"], &["a"]),
            verified_documents(
                CoverageTierV1::Quality,
                &quality_bundle,
                &["a", "b", "c"],
                &["b"],
            ),
        )
        .expect("two independently partial tiers validate");
        let multi = multi_pair
            .derive_topology(
                RetrievalIntentV1::Semantic,
                Some(&TieredQueryEmbeddings::progressive(
                    bound(fast_bundle.clone()),
                    bound(quality_bundle.clone()),
                )),
            )
            .expect("multi-partial exact realization derives");
        assert!(multi.semantic_available());
        assert!(multi.coarse_realized().is_none());
        assert!(matches!(
            multi.fast_realized(),
            RealizedTierCoverageV1::Partial { .. }
        ));
        assert!(matches!(
            multi.quality_realized(),
            RealizedTierCoverageV1::Partial { .. }
        ));

        let quality_only_pair = TierCoveragePairV1::new(
            TierCoverageV1::not_requested(CoverageTierV1::Fast),
            verified_documents(
                CoverageTierV1::Quality,
                &quality_bundle,
                &["a", "b", "c"],
                &["c"],
            ),
        )
        .expect("partial quality-only pair validates");
        let quality_only = quality_only_pair
            .derive_topology(
                RetrievalIntentV1::Semantic,
                Some(&TieredQueryEmbeddings::quality_only(bound(quality_bundle))),
            )
            .expect("partial quality-only exact realization derives");
        assert!(quality_only.semantic_available());
        assert!(quality_only.coarse_realized().is_none());
        assert!(matches!(
            quality_only.quality_realized(),
            RealizedTierCoverageV1::Partial { .. }
        ));
    }

    #[test]
    fn positive_intersection_contributes_even_when_ppm_rounds_to_zero() {
        let counts = CoverageCountsV1::checked(u64::MAX, 1, 1, u64::MAX)
            .expect("extreme exact subset counts validate");
        assert_eq!(counts.coverage_ppm(), 0);
        let realized = RealizedTierCoverageV1::Partial {
            relation: CoverageRelationV1::Subset,
            counts,
        };
        realized.validate().expect("exact partial state validates");
        assert!(realized.contributes());
        assert!(
            derive_lossless_semantic_topology(&realized, &RealizedTierCoverageV1::NotRequested)
                .is_none()
        );
    }

    #[test]
    fn pair_requires_one_exact_requested_universe_and_typed_cross_scope_proof() {
        let fast_bundle = semantic_identity("universe-fast", 8);
        let quality_bundle = semantic_identity("universe-quality", 16);
        let fast = verified_documents(CoverageTierV1::Fast, &fast_bundle, &["a", "b"], &["a", "b"]);
        let different_members = verified_documents(
            CoverageTierV1::Quality,
            &quality_bundle,
            &["a", "c"],
            &["a", "c"],
        );
        assert!(TierCoveragePairV1::new(fast.clone(), different_members).is_err());

        let ordered_quality = TierCoverageV1::verified(
            CoverageTierV1::Quality,
            &quality_bundle.space,
            CoverageBasisV1::OrderedShards,
            test_scope('1', '2'),
            members(&["a", "b"]),
            test_scope('1', '2'),
            members(&["a", "b"]),
            None,
        )
        .expect("ordered quality coverage validates independently");
        assert!(TierCoveragePairV1::new(fast.clone(), ordered_quality).is_err());

        let fast_cross = TierCoverageV1::verified(
            CoverageTierV1::Fast,
            &fast_bundle.space,
            CoverageBasisV1::OrderedShards,
            test_scope('1', '2'),
            members(&["a", "b"]),
            test_scope('1', '2'),
            members(&["a", "b"]),
            None,
        )
        .expect("fast cross-tier scope validates");
        let quality_cross = TierCoverageV1::verified(
            CoverageTierV1::Quality,
            &quality_bundle.space,
            CoverageBasisV1::OrderedShards,
            test_scope('3', '2'),
            members(&["a", "b"]),
            test_scope('3', '2'),
            members(&["a", "b"]),
            None,
        )
        .expect("quality cross-tier scope validates");
        assert!(TierCoveragePairV1::new(fast_cross.clone(), quality_cross.clone()).is_err());
        assert!(
            TierCoveragePairV1::new_with_cross_tier_authority(
                fast_cross.clone(),
                quality_cross,
                test_digest('8'),
            )
            .is_ok()
        );

        let reordered_quality = TierCoverageV1::verified(
            CoverageTierV1::Quality,
            &quality_bundle.space,
            CoverageBasisV1::OrderedShards,
            test_scope('3', '2'),
            members(&["b", "a"]),
            test_scope('3', '2'),
            members(&["b", "a"]),
            None,
        )
        .expect("reordered quality is independently valid");
        assert!(
            TierCoveragePairV1::new_with_cross_tier_authority(
                fast_cross,
                reordered_quality,
                test_digest('8'),
            )
            .is_err()
        );
    }

    #[test]
    fn unknown_unverified_and_zero_live_never_become_semantic_availability() {
        let fast_bundle = semantic_identity("unavailable-fast", 8);
        let quality_bundle = semantic_identity("unavailable-quality", 16);
        let live = members(&["a"]);
        let variants = [
            TierCoverageV1::requested_missing(
                CoverageTierV1::Quality,
                &quality_bundle.space,
                CoverageBasisV1::CanonicalDocuments,
                test_scope('1', '2'),
                live.clone(),
            )
            .expect("missing quality validates"),
            TierCoverageV1::unknown(
                CoverageTierV1::Quality,
                &quality_bundle.space,
                CoverageBasisV1::CanonicalDocuments,
                test_scope('1', '2'),
                live.clone(),
                CoverageUnknownReasonV1::OwnerUnavailable,
            )
            .expect("unknown quality validates"),
            TierCoverageV1::unverified(
                CoverageTierV1::Quality,
                &quality_bundle.space,
                CoverageBasisV1::CanonicalDocuments,
                test_scope('1', '2'),
                live,
                CoverageUnverifiedReasonV1::CountsUntrusted,
                Some(1),
            )
            .expect("unverified quality validates"),
        ];
        for quality in variants {
            let pair = TierCoveragePairV1::new(
                verified_documents(CoverageTierV1::Fast, &fast_bundle, &["a"], &["a"]),
                quality,
            )
            .expect("coverage pair validates");
            let embeddings = TieredQueryEmbeddings::progressive(
                bound(fast_bundle.clone()),
                bound(quality_bundle.clone()),
            );
            let decision = pair
                .derive_topology(RetrievalIntentV1::Semantic, Some(&embeddings))
                .expect("unavailable quality conservatively drops to fast");
            assert_eq!(
                decision.coarse_realized(),
                Some(RetrievalTopology::FastOnly)
            );
        }

        let zero_pair = TierCoveragePairV1::new(
            verified_documents(CoverageTierV1::Fast, &fast_bundle, &[], &[]),
            verified_documents(CoverageTierV1::Quality, &quality_bundle, &[], &[]),
        )
        .expect("zero-live pair validates");
        let zero_embeddings =
            TieredQueryEmbeddings::progressive(bound(fast_bundle), bound(quality_bundle));
        let zero_decision = zero_pair
            .derive_topology(RetrievalIntentV1::Semantic, Some(&zero_embeddings))
            .expect("zero live degrades without inventing availability");
        assert_eq!(
            zero_decision.coarse_realized(),
            Some(RetrievalTopology::LexicalOnly)
        );
        assert!(!zero_decision.semantic_available());
    }

    #[test]
    fn hash_control_is_explicit_and_can_never_satisfy_semantic_intent() {
        let hash_bundle = identity("explicit-hash-control", 8);
        assert_eq!(hash_bundle.space.kind, EmbeddingSpaceKindV1::HashControl);
        let coverage = TierCoveragePairV1::new(
            verified_documents(CoverageTierV1::Fast, &hash_bundle, &["a"], &["a"]),
            TierCoverageV1::not_requested(CoverageTierV1::Quality),
        )
        .expect("hash coverage pair validates");
        let embeddings = TieredQueryEmbeddings::fast_only(bound(hash_bundle));
        let control = coverage
            .derive_topology(RetrievalIntentV1::HashControl, Some(&embeddings))
            .expect("explicit hash-control intent validates");
        assert_eq!(control.requested(), RetrievalTopology::HashControl);
        assert_eq!(
            control.coarse_realized(),
            Some(RetrievalTopology::HashControl)
        );
        assert!(!control.semantic_available());
        assert!(
            coverage
                .derive_topology(RetrievalIntentV1::Semantic, Some(&embeddings))
                .is_err()
        );
    }

    #[test]
    fn swapped_duplicate_missing_and_extra_bindings_are_rejected() {
        let fast_bundle = semantic_identity("binding-fast", 8);
        let quality_bundle = semantic_identity("binding-quality", 16);
        let pair = TierCoveragePairV1::new(
            verified_documents(CoverageTierV1::Fast, &fast_bundle, &["a"], &["a"]),
            verified_documents(CoverageTierV1::Quality, &quality_bundle, &["a"], &["a"]),
        )
        .expect("coverage pair validates");
        let swapped = TieredQueryEmbeddings::progressive(
            bound(quality_bundle.clone()),
            bound(fast_bundle.clone()),
        );
        assert!(
            pair.derive_topology(RetrievalIntentV1::Semantic, Some(&swapped))
                .is_err()
        );
        let only_fast = TieredQueryEmbeddings::fast_only(bound(fast_bundle));
        assert!(
            pair.derive_topology(RetrievalIntentV1::Semantic, Some(&only_fast))
                .is_err()
        );

        let unrequested_quality = TierCoveragePairV1::new(
            pair.fast().clone(),
            TierCoverageV1::not_requested(CoverageTierV1::Quality),
        )
        .expect("fast-only coverage validates");
        let extra_quality = TieredQueryEmbeddings::progressive(
            bound(semantic_identity("binding-fast", 8)),
            bound(quality_bundle),
        );
        assert!(
            unrequested_quality
                .derive_topology(RetrievalIntentV1::Semantic, Some(&extra_quality))
                .is_err()
        );

        let shared_bundle = semantic_identity("duplicate-binding", 8);
        let duplicate_pair = TierCoveragePairV1::new(
            verified_documents(CoverageTierV1::Fast, &shared_bundle, &["a"], &["a"]),
            verified_documents(CoverageTierV1::Quality, &shared_bundle, &["a"], &["a"]),
        )
        .expect("coverage roles remain independent before binding");
        let duplicate_embeddings =
            TieredQueryEmbeddings::progressive(bound(shared_bundle.clone()), bound(shared_bundle));
        assert!(
            duplicate_pair
                .derive_topology(RetrievalIntentV1::Semantic, Some(&duplicate_embeddings))
                .is_err()
        );
    }

    #[test]
    fn realized_superset_fails_topology_derivation_closed() {
        let bundle = semantic_identity("unsafe-superset", 8);
        let pair = TierCoveragePairV1::new(
            verified_documents(CoverageTierV1::Fast, &bundle, &["live"], &["live", "stale"]),
            TierCoverageV1::not_requested(CoverageTierV1::Quality),
        )
        .expect("superset remains representable as evidence");
        let embeddings = TieredQueryEmbeddings::fast_only(bound(bundle));
        assert!(
            pair.derive_topology(RetrievalIntentV1::Semantic, Some(&embeddings))
                .is_err()
        );
    }

    #[test]
    fn lexical_only_requires_no_requested_tiers_or_bindings() {
        let pair = TierCoveragePairV1::new(
            TierCoverageV1::not_requested(CoverageTierV1::Fast),
            TierCoverageV1::not_requested(CoverageTierV1::Quality),
        )
        .expect("missing pair validates");
        let decision = pair
            .derive_topology(RetrievalIntentV1::LexicalOnly, None)
            .expect("no vector authority derives lexical only");
        assert_eq!(decision.requested(), RetrievalTopology::LexicalOnly);
        assert_eq!(
            decision.coarse_realized(),
            Some(RetrievalTopology::LexicalOnly)
        );
        assert!(!decision.semantic_available());
    }

    #[test]
    fn coverage_pair_and_derived_topology_serde_roundtrip_canonically() {
        let fast_bundle = semantic_identity("serde-fast", 8);
        let quality_bundle = semantic_identity("serde-quality", 16);
        let fast_context = verified_documents_context(
            CoverageTierV1::Fast,
            &fast_bundle,
            &["a", "b", "c"],
            &["a", "b", "c"],
        );
        let quality_context = verified_documents_context(
            CoverageTierV1::Quality,
            &quality_bundle,
            &["a", "b", "c"],
            &["a", "c"],
        );
        let pair = TierCoveragePairV1::new(
            fast_context.recompute().expect("recompute fast"),
            quality_context.recompute().expect("recompute quality"),
        )
        .expect("serde pair validates");
        let embeddings = TieredQueryEmbeddings::progressive(
            bound(fast_bundle.clone()),
            bound(quality_bundle.clone()),
        );
        let decision = pair
            .derive_topology(RetrievalIntentV1::Semantic, Some(&embeddings))
            .expect("serde decision derives");

        let pair_json = serde_json::to_string(&pair).expect("serialize pair");
        let pair_claim: UntrustedTierCoveragePairV1 =
            serde_json::from_str(&pair_json).expect("deserialize pair");
        let pair_roundtrip = pair_claim
            .validate_against(&fast_context, &quality_context, None)
            .expect("owner-backed pair promotion succeeds");
        assert_eq!(pair_roundtrip, pair);
        assert_eq!(pair_roundtrip.fingerprint(), pair.fingerprint());

        let decision_json = serde_json::to_string(&decision).expect("serialize decision");
        let decision_claim: UntrustedDerivedRetrievalTopologyV1 =
            serde_json::from_str(&decision_json).expect("deserialize decision");
        let decision_roundtrip = decision_claim
            .validate_against(&pair, RetrievalIntentV1::Semantic, Some(&embeddings))
            .expect("fresh derivation promotes decision");
        assert_eq!(decision_roundtrip, decision);
        assert_eq!(decision_roundtrip.fingerprint(), decision.fingerprint());
        assert!(!pair_json.contains("doc-a"));
        assert!(!decision_json.contains("query"));
    }

    #[test]
    fn coherent_wire_claims_cannot_self_attest_coverage_or_cross_tier_authority() {
        let fast_bundle = semantic_identity("wire-authority-fast", 8);
        let quality_bundle = semantic_identity("wire-authority-quality", 16);
        let trusted_fast =
            verified_documents_context(CoverageTierV1::Fast, &fast_bundle, &["a", "b"], &["a"]);

        let coherent_other =
            verified_documents(CoverageTierV1::Fast, &fast_bundle, &["a", "c"], &["a"]);
        let claim: UntrustedTierCoverageV1 = serde_json::from_value(
            serde_json::to_value(coherent_other).expect("serialize coherent alternate claim"),
        )
        .expect("coherent alternate claim parses");
        assert!(claim.validate_against(&trusted_fast).is_err());

        let transplanted = verified_documents(
            CoverageTierV1::Fast,
            &semantic_identity("wire-authority-other-space", 8),
            &["a", "b"],
            &["a"],
        );
        let claim: UntrustedTierCoverageV1 = serde_json::from_value(
            serde_json::to_value(transplanted).expect("serialize space transplant"),
        )
        .expect("space transplant claim parses");
        assert!(claim.validate_against(&trusted_fast).is_err());

        let fast_context = TrustedTierCoverageContextV1::verified(
            CoverageTierV1::Fast,
            fast_bundle.space.clone(),
            CoverageBasisV1::CanonicalDocuments,
            test_scope('1', '2'),
            members(&["a", "b"]),
            test_scope('1', '2'),
            members(&["a", "b"]),
            None,
        )
        .expect("fast owner context validates");
        let quality_context = TrustedTierCoverageContextV1::verified(
            CoverageTierV1::Quality,
            quality_bundle.space.clone(),
            CoverageBasisV1::CanonicalDocuments,
            test_scope('3', '2'),
            members(&["a", "b"]),
            test_scope('3', '2'),
            members(&["a", "b"]),
            None,
        )
        .expect("quality owner context validates");
        let pair = TierCoveragePairV1::new_with_cross_tier_authority(
            fast_context.recompute().expect("recompute fast"),
            quality_context.recompute().expect("recompute quality"),
            test_digest('8'),
        )
        .expect("trusted cross-tier proof builds");
        let mut forged = serde_json::to_value(pair).expect("serialize pair");
        forged["cross_tier_universe_evidence"]["authority_fingerprint"] =
            serde_json::Value::String(test_digest('9'));
        let claim: UntrustedTierCoveragePairV1 =
            serde_json::from_value(forged).expect("self-consistent authority claim parses");
        let independent_authority = test_digest('8');
        assert!(
            claim
                .validate_against(
                    &fast_context,
                    &quality_context,
                    Some(independent_authority.as_str())
                )
                .is_err()
        );
    }

    #[test]
    fn derived_wire_claim_cannot_choose_pair_hash_or_swap_or_duplicate_spaces() {
        let fast_bundle = semantic_identity("derived-wire-fast", 8);
        let quality_bundle = semantic_identity("derived-wire-quality", 16);
        let pair = TierCoveragePairV1::new(
            verified_documents(CoverageTierV1::Fast, &fast_bundle, &["a"], &["a"]),
            verified_documents(CoverageTierV1::Quality, &quality_bundle, &["a"], &["a"]),
        )
        .expect("pair validates");
        let embeddings = TieredQueryEmbeddings::progressive(
            bound(fast_bundle.clone()),
            bound(quality_bundle.clone()),
        );
        let decision = pair
            .derive_topology(RetrievalIntentV1::Semantic, Some(&embeddings))
            .expect("decision derives");
        let base = serde_json::to_value(decision).expect("serialize decision");

        let mut arbitrary_pair = base.clone();
        arbitrary_pair["coverage_pair_fingerprint"] = serde_json::Value::String(test_digest('f'));
        let claim: UntrustedDerivedRetrievalTopologyV1 =
            serde_json::from_value(arbitrary_pair).expect("arbitrary pair hash parses");
        assert!(
            claim
                .validate_against(&pair, RetrievalIntentV1::Semantic, Some(&embeddings))
                .is_err()
        );

        let mut swapped = base.clone();
        swapped["fast_space_fingerprint"] =
            serde_json::Value::String(quality_bundle.space.fingerprint());
        swapped["quality_space_fingerprint"] =
            serde_json::Value::String(fast_bundle.space.fingerprint());
        let claim: UntrustedDerivedRetrievalTopologyV1 =
            serde_json::from_value(swapped).expect("swapped space claims parse");
        assert!(
            claim
                .validate_against(&pair, RetrievalIntentV1::Semantic, Some(&embeddings))
                .is_err()
        );

        let mut duplicate = base;
        duplicate["quality_space_fingerprint"] =
            serde_json::Value::String(fast_bundle.space.fingerprint());
        let claim: UntrustedDerivedRetrievalTopologyV1 =
            serde_json::from_value(duplicate).expect("duplicate space claim parses");
        assert!(
            claim
                .validate_against(&pair, RetrievalIntentV1::Semantic, Some(&embeddings))
                .is_err()
        );
    }

    #[test]
    fn coverage_serde_denies_unknown_duplicate_and_impossible_fields() {
        let bundle = semantic_identity("serde-deny", 8);
        let coverage = verified_documents(CoverageTierV1::Fast, &bundle, &["a", "b"], &["a"]);
        let mut unknown = serde_json::to_value(&coverage).expect("serialize coverage");
        unknown["borrowed_quality_coverage_ppm"] = serde_json::Value::from(500_000);
        assert!(serde_json::from_value::<UntrustedTierCoverageV1>(unknown).is_err());

        let generation = test_digest('1');
        let corpus = test_digest('2');
        let duplicate_scope = format!(
            "{{\"generation_fingerprint\":\"{generation}\",\"generation_fingerprint\":\"{generation}\",\"corpus_fingerprint\":\"{corpus}\"}}"
        );
        assert!(serde_json::from_str::<CoverageScopeV1>(&duplicate_scope).is_err());

        let mut impossible = serde_json::to_value(&coverage).expect("serialize coverage");
        impossible["realized"]["relation"] = serde_json::Value::String("complete".to_owned());
        let claim: UntrustedTierCoverageV1 =
            serde_json::from_value(impossible).expect("impossible relation remains syntactic");
        let context =
            verified_documents_context(CoverageTierV1::Fast, &bundle, &["a", "b"], &["a"]);
        assert!(claim.validate_against(&context).is_err());

        let topology_with_unknown = r#"{"topology":"fast_only","coverage_ppm":1000000}"#;
        assert!(serde_json::from_str::<RetrievalTopology>(topology_with_unknown).is_err());
        assert!(
            serde_json::from_str::<RetrievalTopology>(r#"{"topology":"partial_quality"}"#).is_err()
        );
        assert!(
            serde_json::from_str::<RetrievalTopology>(
                r#"{"topology":"partial_quality","coverage_ppm":null}"#
            )
            .is_err()
        );
        assert!(
            serde_json::from_str::<RetrievalTopology>(
                r#"{"topology":"fast_only","coverage_ppm":null}"#
            )
            .is_err()
        );
        assert!(
            serde_json::from_str::<RetrievalTopology>(
                r#"{"topology":"fast_only","extra":"field"}"#
            )
            .is_err()
        );
    }

    #[test]
    fn retrieval_topology_compatibility_law_is_exhaustive() {
        let variants = [
            RetrievalTopology::LexicalOnly,
            RetrievalTopology::HashControl,
            RetrievalTopology::FastOnly,
            RetrievalTopology::QualityOnly,
            RetrievalTopology::FullProgressive,
            RetrievalTopology::PartialQuality {
                coverage_ppm: 500_000,
            },
        ];
        for requested in variants {
            for realized in variants {
                let expected = match requested {
                    RetrievalTopology::LexicalOnly => {
                        matches!(realized, RetrievalTopology::LexicalOnly)
                    }
                    RetrievalTopology::HashControl => {
                        matches!(realized, RetrievalTopology::HashControl)
                    }
                    RetrievalTopology::FastOnly => matches!(
                        realized,
                        RetrievalTopology::FastOnly | RetrievalTopology::LexicalOnly
                    ),
                    RetrievalTopology::QualityOnly => matches!(
                        realized,
                        RetrievalTopology::QualityOnly
                            | RetrievalTopology::PartialQuality { .. }
                            | RetrievalTopology::LexicalOnly
                    ),
                    RetrievalTopology::FullProgressive => matches!(
                        realized,
                        RetrievalTopology::FullProgressive
                            | RetrievalTopology::PartialQuality { .. }
                            | RetrievalTopology::FastOnly
                            | RetrievalTopology::QualityOnly
                            | RetrievalTopology::LexicalOnly
                    ),
                    RetrievalTopology::PartialQuality { .. } => false,
                };
                assert_eq!(
                    retrieval_topology_fits_request(requested, realized),
                    expected,
                    "compatibility mismatch for {requested:?} -> {realized:?}"
                );
            }
        }
    }

    #[test]
    fn topology_receipt_mutations_fail_validation() {
        let fast_bundle = semantic_identity("receipt-fast", 8);
        let quality_bundle = semantic_identity("receipt-quality", 16);
        let pair = TierCoveragePairV1::new(
            verified_documents(CoverageTierV1::Fast, &fast_bundle, &["a", "b"], &["a", "b"]),
            verified_documents(
                CoverageTierV1::Quality,
                &quality_bundle,
                &["a", "b"],
                &["a"],
            ),
        )
        .expect("receipt pair validates");
        let embeddings =
            TieredQueryEmbeddings::progressive(bound(fast_bundle), bound(quality_bundle));
        let decision = pair
            .derive_topology(RetrievalIntentV1::Semantic, Some(&embeddings))
            .expect("receipt derives");

        let base = serde_json::to_value(&decision).expect("serialize decision");
        let mut semantic_lie = base.clone();
        semantic_lie["semantic_available"] = serde_json::Value::Bool(false);
        let claim: UntrustedDerivedRetrievalTopologyV1 =
            serde_json::from_value(semantic_lie).expect("semantic lie remains syntactic");
        assert!(
            claim
                .validate_against(&pair, RetrievalIntentV1::Semantic, Some(&embeddings))
                .is_err()
        );

        let mut complete_partial = base.clone();
        complete_partial["coarse_realized"]["coverage_ppm"] =
            serde_json::Value::from(COMPLETE_COVERAGE_PPM);
        let claim: UntrustedDerivedRetrievalTopologyV1 =
            serde_json::from_value(complete_partial).expect("boundary claim remains syntactic");
        assert!(
            claim
                .validate_against(&pair, RetrievalIntentV1::Semantic, Some(&embeddings))
                .is_err()
        );

        let mut hash_relabel = base;
        hash_relabel["fast_kind"] = serde_json::Value::String("hash_control".to_owned());
        let claim: UntrustedDerivedRetrievalTopologyV1 =
            serde_json::from_value(hash_relabel).expect("kind relabel remains syntactic");
        assert!(
            claim
                .validate_against(&pair, RetrievalIntentV1::Semantic, Some(&embeddings))
                .is_err()
        );
    }

    #[test]
    fn every_coverage_semantic_mutation_changes_the_stable_fingerprint() {
        let bundle = semantic_identity("fingerprint-base", 8);
        let other_bundle = semantic_identity("fingerprint-other", 8);
        let base = verified_documents(CoverageTierV1::Fast, &bundle, &["a", "b"], &["a"]);
        let reordered = verified_documents(CoverageTierV1::Fast, &bundle, &["b", "a"], &["a"]);
        assert_eq!(base.fingerprint(), reordered.fingerprint());

        let changed_member = verified_documents(CoverageTierV1::Fast, &bundle, &["a", "c"], &["a"]);
        let changed_space =
            verified_documents(CoverageTierV1::Fast, &other_bundle, &["a", "b"], &["a"]);
        let changed_scope = TierCoverageV1::verified(
            CoverageTierV1::Fast,
            &bundle.space,
            CoverageBasisV1::CanonicalDocuments,
            test_scope('3', '2'),
            members(&["a", "b"]),
            test_scope('3', '2'),
            members(&["a"]),
            None,
        )
        .expect("changed scope validates");
        let changed_basis = TierCoverageV1::verified(
            CoverageTierV1::Fast,
            &bundle.space,
            CoverageBasisV1::OrderedShards,
            test_scope('1', '2'),
            members(&["a", "b"]),
            test_scope('1', '2'),
            members(&["a"]),
            None,
        )
        .expect("changed basis validates");
        let changed_state = TierCoverageV1::unknown(
            CoverageTierV1::Fast,
            &bundle.space,
            CoverageBasisV1::CanonicalDocuments,
            test_scope('1', '2'),
            members(&["a", "b"]),
            CoverageUnknownReasonV1::OwnerUnavailable,
        )
        .expect("changed state validates");
        let fingerprints = [
            base.fingerprint(),
            changed_member.fingerprint(),
            changed_space.fingerprint(),
            changed_scope.fingerprint(),
            changed_basis.fingerprint(),
            changed_state.fingerprint(),
        ];
        assert_eq!(fingerprints.iter().collect::<BTreeSet<_>>().len(), 6);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn coverage_set_algebra_matches_u16_bitsets_and_canonical_order(
            live_mask in any::<u16>(),
            indexed_mask in any::<u16>(),
        ) {
            let bundle = semantic_identity("property-coverage", 8);
            let mut live = (0..16)
                .filter(|bit| live_mask & (1_u16 << bit) != 0)
                .map(|bit| format!("doc-{bit:02}"))
                .collect::<Vec<_>>();
            let indexed = (0..16)
                .filter(|bit| indexed_mask & (1_u16 << bit) != 0)
                .map(|bit| format!("doc-{bit:02}"))
                .collect::<Vec<_>>();
            let coverage_result = TierCoverageV1::verified(
                CoverageTierV1::Fast,
                &bundle.space,
                CoverageBasisV1::CanonicalDocuments,
                test_scope('1', '2'),
                live.clone(),
                test_scope('1', '2'),
                indexed.clone(),
                None,
            );
            prop_assert!(coverage_result.is_ok());
            if let Ok(coverage) = coverage_result {
                let intersection = live_mask & indexed_mask;
                let union = live_mask | indexed_mask;
                let expected_counts = CoverageCountsV1::checked(
                    u64::from(live_mask.count_ones()),
                    u64::from(indexed_mask.count_ones()),
                    u64::from(intersection.count_ones()),
                    u64::from(union.count_ones()),
                );
                prop_assert!(expected_counts.is_ok());
                if let Ok(expected_counts) = expected_counts {
                    prop_assert_eq!(coverage.verified_facts(), Some((derive_coverage_relation(expected_counts), expected_counts)));
                }
                live.reverse();
                let reordered = TierCoverageV1::verified(
                    CoverageTierV1::Fast,
                    &bundle.space,
                    CoverageBasisV1::CanonicalDocuments,
                    test_scope('1', '2'),
                    live,
                    test_scope('1', '2'),
                    indexed,
                    None,
                );
                prop_assert!(reordered.is_ok());
                if let Ok(reordered) = reordered {
                    prop_assert_eq!(coverage.fingerprint(), reordered.fingerprint());
                }
            }
        }
    }

    /// `cmp_rank` must be a total order: score descending, ties broken on `doc_id`.
    #[test]
    fn cmp_rank_breaks_score_ties_on_doc_id() {
        let mut hits = [
            vhit(0, 0.5, "b"),
            vhit(1, 0.9, "z"),
            vhit(2, 0.5, "a"),
            vhit(3, 0.5, "c"),
        ];
        hits.sort_unstable_by(VectorHit::cmp_rank);
        let ids: Vec<&str> = hits.iter().map(|h| h.doc_id.as_str()).collect();
        assert_eq!(ids, ["z", "a", "b", "c"]);
    }

    /// NaN sorts LAST, inheriting `cmp_by_score`. A bare descending `total_cmp` would do the
    /// opposite — IEEE 754 `totalOrder` ranks `+NaN` above `+inf`, putting a corrupted score at
    /// rank 0. Phase-1 score corrections re-sort with this comparator, so the distinction is load
    /// bearing.
    #[test]
    fn cmp_rank_sorts_nan_last_unlike_bare_total_cmp() {
        let mut hits = [
            vhit(0, f32::NAN, "nan"),
            vhit(1, 0.1, "lo"),
            vhit(2, 0.9, "hi"),
        ];
        hits.sort_unstable_by(VectorHit::cmp_rank);
        let ids: Vec<&str> = hits.iter().map(|h| h.doc_id.as_str()).collect();
        assert_eq!(ids, ["hi", "lo", "nan"], "NaN must sort last");

        // The trap this guards against.
        let mut bare = [vhit(0, f32::NAN, "nan"), vhit(1, 0.9, "hi")];
        bare.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
        assert_eq!(
            bare[0].doc_id.as_str(),
            "nan",
            "bare descending total_cmp puts +NaN at rank 0 — the reason cmp_rank exists"
        );
    }

    #[test]
    fn indexable_document_builder() {
        let doc = IndexableDocument::new("doc-1", "Hello world")
            .with_title("Greeting")
            .with_metadata("source", "test");

        assert_eq!(doc.id, "doc-1");
        assert_eq!(doc.content, "Hello world");
        assert_eq!(doc.title.as_deref(), Some("Greeting"));
        assert_eq!(doc.metadata.get("source").map(String::as_str), Some("test"));
    }

    #[test]
    fn indexable_document_minimal() {
        let doc = IndexableDocument::new("id", "text");
        assert!(doc.title.is_none());
        assert!(doc.metadata.is_empty());
    }

    #[test]
    fn vector_hit_nan_safe_ordering() {
        let hit_a = VectorHit {
            index: 0,
            score: 0.9,
            doc_id: "a".into(),
        };
        let hit_nan = VectorHit {
            index: 1,
            score: f32::NAN,
            doc_id: "b".into(),
        };
        // NaN should sort below real values (hit_a should come first).
        assert_eq!(
            hit_a.cmp_by_score(&hit_nan),
            std::cmp::Ordering::Less // a comes first (better score)
        );
    }

    #[test]
    fn fused_hit_tie_breaking() {
        let hit_both = FusedHit {
            doc_id: "a".into(),
            rrf_score: 0.02,
            lexical_rank: Some(1),
            semantic_rank: Some(3),
            semantic_index: Some(3),
            lexical_score: Some(5.0),
            semantic_score: Some(0.8),
            in_both_sources: true,
        };
        let hit_semantic_only = FusedHit {
            doc_id: "b".into(),
            rrf_score: 0.02, // Same RRF score
            lexical_rank: None,
            semantic_rank: Some(2),
            semantic_index: Some(2),
            lexical_score: None,
            semantic_score: Some(0.9),
            in_both_sources: false,
        };
        // Same RRF -> in_both_sources wins
        assert_eq!(
            hit_both.cmp_for_ranking(&hit_semantic_only),
            std::cmp::Ordering::Less // hit_both ranks first
        );
    }

    #[test]
    fn fused_hit_rrf_score_dominates() {
        let high = FusedHit {
            doc_id: "a".into(),
            rrf_score: 0.03,
            lexical_rank: None,
            semantic_rank: Some(1),
            semantic_index: Some(1),
            lexical_score: None,
            semantic_score: Some(0.9),
            in_both_sources: false,
        };
        let low = FusedHit {
            doc_id: "b".into(),
            rrf_score: 0.01,
            lexical_rank: Some(1),
            semantic_rank: Some(1),
            semantic_index: Some(1),
            lexical_score: Some(10.0),
            semantic_score: Some(0.99),
            in_both_sources: true,
        };
        // Higher RRF always wins regardless of other fields.
        assert_eq!(
            high.cmp_for_ranking(&low),
            std::cmp::Ordering::Less // high ranks first
        );
    }

    #[test]
    fn fused_hit_deterministic_doc_id_tiebreak() {
        let a = FusedHit {
            doc_id: "alpha".into(),
            rrf_score: 0.02,
            lexical_rank: None,
            semantic_rank: None,
            semantic_index: None,
            lexical_score: None,
            semantic_score: None,
            in_both_sources: false,
        };
        let b = FusedHit {
            doc_id: "beta".into(),
            rrf_score: 0.02,
            lexical_rank: None,
            semantic_rank: None,
            semantic_index: None,
            lexical_score: None,
            semantic_score: None,
            in_both_sources: false,
        };
        // All else equal -> lexicographic doc_id ascending.
        assert_eq!(a.cmp_for_ranking(&b), std::cmp::Ordering::Less);
    }

    #[test]
    fn scored_result_serde_roundtrip() {
        let result = ScoredResult {
            doc_id: "doc-42".into(),
            score: 0.85,
            source: ScoreSource::Hybrid,
            index: None,
            fast_score: Some(0.7),
            quality_score: Some(0.9),
            lexical_score: Some(12.5),
            rerank_score: None,
            explanation: None,
            metadata: Some(std::sync::Arc::new(
                serde_json::json!({"tags": ["rust", "search"]}),
            )),
        };
        let json = serde_json::to_string(&result).expect("serialize");
        let roundtripped: ScoredResult = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(roundtripped.doc_id, "doc-42");
        assert!((roundtripped.score - 0.85).abs() < f32::EPSILON);
        assert_eq!(roundtripped.source, ScoreSource::Hybrid);
        assert!(roundtripped.metadata.is_some());
    }

    #[test]
    fn rank_changes_total() {
        let changes = RankChanges {
            promoted: 3,
            demoted: 2,
            stable: 5,
        };
        assert_eq!(changes.total(), 10);
    }

    #[test]
    fn search_phase_initial_construction() {
        let phase = SearchPhase::Initial {
            results: vec![],
            latency: Duration::from_millis(12),
            metrics: PhaseMetrics {
                embedder_id: "potion-128M".into(),
                vectors_searched: 1000,
                lexical_candidates: 50,
                fused_count: 10,
                skip_reason: None,
            },
        };
        if let SearchPhase::Initial { latency, .. } = phase {
            assert_eq!(latency, Duration::from_millis(12));
        }
    }

    #[test]
    fn search_phase_refinement_failed_carries_results() {
        let initial = vec![ScoredResult {
            doc_id: "doc-1".into(),
            score: 0.5,
            source: ScoreSource::Hybrid,
            index: None,
            fast_score: None,
            quality_score: None,
            lexical_score: None,
            rerank_score: None,
            explanation: None,
            metadata: None,
        }];
        let phase = SearchPhase::RefinementFailed {
            initial_results: initial,
            error: SearchError::SearchTimeout {
                elapsed_ms: 500,
                budget_ms: 300,
            },
            latency: Duration::from_millis(500),
        };
        if let SearchPhase::RefinementFailed {
            initial_results, ..
        } = phase
        {
            assert_eq!(initial_results.len(), 1);
            assert_eq!(initial_results[0].doc_id, "doc-1");
        }
    }

    #[test]
    fn indexable_document_serde_roundtrip() {
        let doc = IndexableDocument::new("id-1", "Hello world")
            .with_title("Test")
            .with_metadata("lang", "en");
        let json = serde_json::to_string(&doc).expect("serialize");
        let rt: IndexableDocument = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(rt.id, "id-1");
        assert_eq!(rt.title.as_deref(), Some("Test"));
        assert_eq!(rt.metadata.get("lang").map(String::as_str), Some("en"));
    }

    #[test]
    fn search_metrics_serde_roundtrip() {
        let metrics = SearchMetrics {
            mode: SearchMode::Hybrid,
            query_class: Some(QueryClass::NaturalLanguage),
            total_latency_ms: 12.5,
            phase1_latency_ms: Some(5.4),
            phase2_latency_ms: Some(7.1),
            result_count: 10,
            lexical_candidates: 60,
            semantic_candidates: 45,
            refined: true,
        };

        let json = serde_json::to_string(&metrics).expect("serialize");
        let decoded: SearchMetrics = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(decoded.mode, SearchMode::Hybrid);
        assert_eq!(decoded.query_class, Some(QueryClass::NaturalLanguage));
        assert!((decoded.total_latency_ms - 12.5).abs() < f64::EPSILON);
        assert_eq!(decoded.result_count, 10);
        assert!(decoded.refined);
    }

    #[test]
    fn embedding_metrics_serde_roundtrip() {
        let metrics = EmbeddingMetrics {
            embedder_id: "potion-multilingual-128M".into(),
            batch_size: 32,
            duration_ms: 1.9,
            dimension: 256,
            is_semantic: true,
        };

        let json = serde_json::to_string(&metrics).expect("serialize");
        let decoded: EmbeddingMetrics = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(decoded.embedder_id, "potion-multilingual-128M");
        assert_eq!(decoded.batch_size, 32);
        assert_eq!(decoded.dimension, 256);
        assert!(decoded.is_semantic);
    }

    #[test]
    fn index_metrics_serde_roundtrip() {
        let metrics = IndexMetrics {
            doc_count: 1_000,
            index_size_bytes: 12_345_678,
            updated_docs: 11,
            staleness_detected: false,
        };

        let json = serde_json::to_string(&metrics).expect("serialize");
        let decoded: IndexMetrics = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(decoded.doc_count, 1_000);
        assert_eq!(decoded.index_size_bytes, 12_345_678);
        assert_eq!(decoded.updated_docs, 11);
        assert!(!decoded.staleness_detected);
    }

    // ─── bd-ta55 tests begin ───

    #[test]
    fn indexable_document_multiple_metadata() {
        let doc = IndexableDocument::new("d1", "content")
            .with_metadata("a", "1")
            .with_metadata("b", "2")
            .with_metadata("c", "3");
        assert_eq!(doc.metadata.len(), 3);
        assert_eq!(doc.metadata["a"], "1");
        assert_eq!(doc.metadata["b"], "2");
        assert_eq!(doc.metadata["c"], "3");
    }

    #[test]
    fn indexable_document_metadata_overwrite() {
        let doc = IndexableDocument::new("d1", "content")
            .with_metadata("key", "old")
            .with_metadata("key", "new");
        assert_eq!(doc.metadata.len(), 1);
        assert_eq!(doc.metadata["key"], "new");
    }

    #[test]
    fn indexable_document_clone_debug() {
        let doc = IndexableDocument::new("d1", "text").with_title("T");
        let cloned = doc.clone();
        assert_eq!(cloned.id, "d1");
        assert_eq!(cloned.title.as_deref(), Some("T"));
        let dbg = format!("{doc:?}");
        assert!(dbg.contains("IndexableDocument"));
        assert!(dbg.contains("d1"));
    }

    #[test]
    fn vector_hit_partial_eq() {
        let a = VectorHit {
            index: 0,
            score: 0.5,
            doc_id: "a".into(),
        };
        let b = VectorHit {
            index: 0,
            score: 0.5,
            doc_id: "a".into(),
        };
        assert_eq!(a, b);

        let c = VectorHit {
            index: 1,
            score: 0.5,
            doc_id: "a".into(),
        };
        assert_ne!(a, c);
    }

    #[test]
    fn vector_hit_serde_roundtrip() {
        let hit = VectorHit {
            index: 42,
            score: 0.95,
            doc_id: "doc-x".into(),
        };
        let json = serde_json::to_string(&hit).unwrap();
        let decoded: VectorHit = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, hit);
    }

    #[test]
    fn vector_hit_both_nan() {
        let a = VectorHit {
            index: 0,
            score: f32::NAN,
            doc_id: "a".into(),
        };
        let b = VectorHit {
            index: 1,
            score: f32::NAN,
            doc_id: "b".into(),
        };
        // Both NaN -> treated as equal (both map to NEG_INFINITY)
        assert_eq!(a.cmp_by_score(&b), std::cmp::Ordering::Equal);
    }

    #[test]
    fn vector_hit_equal_scores() {
        let a = VectorHit {
            index: 0,
            score: 0.75,
            doc_id: "a".into(),
        };
        let b = VectorHit {
            index: 1,
            score: 0.75,
            doc_id: "b".into(),
        };
        assert_eq!(a.cmp_by_score(&b), std::cmp::Ordering::Equal);
    }

    #[test]
    fn vector_hit_descending_order() {
        let high = VectorHit {
            index: 0,
            score: 0.9,
            doc_id: "h".into(),
        };
        let low = VectorHit {
            index: 1,
            score: 0.1,
            doc_id: "l".into(),
        };
        // Descending: high comes first (Less means "before")
        assert_eq!(high.cmp_by_score(&low), std::cmp::Ordering::Less);
        assert_eq!(low.cmp_by_score(&high), std::cmp::Ordering::Greater);
    }

    #[test]
    fn fused_hit_serde_roundtrip() {
        let hit = FusedHit {
            doc_id: "fused-1".into(),
            rrf_score: 0.025,
            lexical_rank: Some(3),
            semantic_rank: Some(7),
            semantic_index: Some(7),
            lexical_score: Some(8.5),
            semantic_score: Some(0.72),
            in_both_sources: true,
        };
        let json = serde_json::to_string(&hit).unwrap();
        let decoded: FusedHit = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.doc_id, "fused-1");
        assert!((decoded.rrf_score - 0.025).abs() < f64::EPSILON);
        assert!(decoded.in_both_sources);
        assert_eq!(decoded.lexical_rank, Some(3));
    }

    #[test]
    fn fused_hit_lexical_score_tiebreak() {
        // Same RRF, same in_both_sources -> lexical_score descending
        let high_lex = FusedHit {
            doc_id: "z".into(), // worse doc_id
            rrf_score: 0.02,
            lexical_rank: Some(1),
            semantic_rank: None,
            semantic_index: None,
            lexical_score: Some(15.0),
            semantic_score: None,
            in_both_sources: false,
        };
        let low_lex = FusedHit {
            doc_id: "a".into(), // better doc_id
            rrf_score: 0.02,
            lexical_rank: Some(5),
            semantic_rank: None,
            semantic_index: None,
            lexical_score: Some(3.0),
            semantic_score: None,
            in_both_sources: false,
        };
        // Higher lexical_score wins (level 3)
        assert_eq!(high_lex.cmp_for_ranking(&low_lex), std::cmp::Ordering::Less);
    }

    #[test]
    fn fused_hit_clone_debug() {
        let hit = FusedHit {
            doc_id: "test".into(),
            rrf_score: 0.01,
            lexical_rank: None,
            semantic_rank: Some(5),
            semantic_index: Some(5),
            lexical_score: None,
            semantic_score: Some(0.6),
            in_both_sources: false,
        };
        let cloned = hit.clone();
        assert_eq!(cloned.doc_id, "test");
        let dbg = format!("{hit:?}");
        assert!(dbg.contains("FusedHit"));
    }

    #[test]
    fn score_source_all_variants_serde() {
        let variants = [
            ScoreSource::Lexical,
            ScoreSource::SemanticFast,
            ScoreSource::SemanticQuality,
            ScoreSource::Hybrid,
            ScoreSource::Reranked,
            ScoreSource::HashControl,
        ];
        for variant in &variants {
            let json = serde_json::to_string(variant).unwrap();
            let decoded: ScoreSource = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, variant);
        }
    }

    #[test]
    fn is_hash_generation_id_matches_control_families() {
        assert!(super::is_hash_generation_id("hash"));
        assert!(super::is_hash_generation_id("HASH"));
        assert!(super::is_hash_generation_id("hash-fnv1a-256"));
        assert!(super::is_hash_generation_id("fnv1a-256"));
        assert!(super::is_hash_generation_id("jl-128"));
        assert!(super::is_hash_generation_id("hash/fnv1a"));
        assert!(super::is_hash_generation_id("HASH/FNV1A"));
        assert!(!super::is_hash_generation_id("minilm-l6-v2"));
        assert!(!super::is_hash_generation_id("potion-128m"));
        assert!(!super::is_hash_generation_id("stub-384"));
        assert!(!super::is_hash_generation_id("hashed-minilm"));
    }

    #[test]
    fn score_source_clone_copy_eq() {
        let a = ScoreSource::Hybrid;
        let b = a; // Copy
        let c = a; // Copy (ScoreSource is Copy)
        assert_eq!(a, b);
        assert_eq!(a, c);
        assert_ne!(ScoreSource::Lexical, ScoreSource::Hybrid);
    }

    #[test]
    fn scored_result_all_none_optionals() {
        let result = ScoredResult {
            doc_id: "min".into(),
            score: 0.1,
            source: ScoreSource::Lexical,
            index: None,
            fast_score: None,
            quality_score: None,
            lexical_score: None,
            rerank_score: None,
            explanation: None,
            metadata: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: ScoredResult = serde_json::from_str(&json).unwrap();
        assert!(decoded.fast_score.is_none());
        assert!(decoded.quality_score.is_none());
        assert!(decoded.rerank_score.is_none());
        assert!(decoded.metadata.is_none());
    }

    #[test]
    fn scored_result_all_some_optionals() {
        let result = ScoredResult {
            doc_id: "max".into(),
            score: 0.99,
            source: ScoreSource::Reranked,
            index: Some(3),
            fast_score: Some(0.7),
            quality_score: Some(0.9),
            lexical_score: Some(15.0),
            rerank_score: Some(0.95),
            explanation: None,
            metadata: Some(std::sync::Arc::new(serde_json::json!({"key": "value"}))),
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: ScoredResult = serde_json::from_str(&json).unwrap();
        assert!(decoded.fast_score.is_some());
        assert!(decoded.quality_score.is_some());
        assert!(decoded.lexical_score.is_some());
        assert!(decoded.rerank_score.is_some());
        assert!(decoded.metadata.is_some());
    }

    #[test]
    fn scored_result_clone_debug() {
        let result = ScoredResult {
            doc_id: "d".into(),
            score: 0.5,
            source: ScoreSource::SemanticFast,
            index: Some(0),
            fast_score: Some(0.5),
            quality_score: None,
            lexical_score: None,
            rerank_score: None,
            explanation: None,
            metadata: None,
        };
        let cloned = result.clone();
        assert_eq!(cloned.doc_id, "d");
        let dbg = format!("{result:?}");
        assert!(dbg.contains("ScoredResult"));
        assert!(dbg.contains("SemanticFast"));
    }

    #[test]
    fn search_mode_all_variants_serde() {
        let variants = [
            SearchMode::Lexical,
            SearchMode::Semantic,
            SearchMode::Hybrid,
            SearchMode::TwoTier,
        ];
        for variant in &variants {
            let json = serde_json::to_string(variant).unwrap();
            let decoded: SearchMode = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, variant);
        }
    }

    #[test]
    fn phase_metrics_serde_clone_debug() {
        let metrics = PhaseMetrics {
            embedder_id: "test-embed".into(),
            vectors_searched: 500,
            lexical_candidates: 30,
            fused_count: 10,
            skip_reason: None,
        };
        let json = serde_json::to_string(&metrics).unwrap();
        let decoded: PhaseMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.embedder_id, "test-embed");
        assert_eq!(decoded.vectors_searched, 500);

        let cloned = metrics.clone();
        assert_eq!(cloned.fused_count, 10);

        let dbg = format!("{metrics:?}");
        assert!(dbg.contains("PhaseMetrics"));
    }

    #[test]
    fn rank_changes_default_and_zero_total() {
        let changes = RankChanges::default();
        assert_eq!(changes.promoted, 0);
        assert_eq!(changes.demoted, 0);
        assert_eq!(changes.stable, 0);
        assert_eq!(changes.total(), 0);
    }

    #[test]
    fn rank_changes_serde_roundtrip() {
        let changes = RankChanges {
            promoted: 5,
            demoted: 3,
            stable: 12,
        };
        let json = serde_json::to_string(&changes).unwrap();
        let decoded: RankChanges = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.promoted, 5);
        assert_eq!(decoded.demoted, 3);
        assert_eq!(decoded.stable, 12);
        assert_eq!(decoded.total(), 20);
    }

    #[test]
    fn rank_changes_total_saturates_on_overflow() {
        let changes = RankChanges {
            promoted: usize::MAX,
            demoted: 1,
            stable: 1,
        };
        assert_eq!(changes.total(), usize::MAX);
    }

    #[test]
    fn search_phase_refined_construction() {
        let phase = SearchPhase::Refined {
            results: vec![ScoredResult {
                doc_id: "r1".into(),
                score: 0.9,
                source: ScoreSource::Hybrid,
                index: None,
                fast_score: Some(0.7),
                quality_score: Some(0.9),
                lexical_score: None,
                rerank_score: None,
                explanation: None,
                metadata: None,
            }],
            latency: Duration::from_millis(120),
            metrics: PhaseMetrics {
                embedder_id: "minilm".into(),
                vectors_searched: 2000,
                lexical_candidates: 100,
                fused_count: 20,
                skip_reason: None,
            },
            rank_changes: RankChanges {
                promoted: 4,
                demoted: 2,
                stable: 14,
            },
        };
        assert!(matches!(phase, SearchPhase::Refined { .. }));
        let SearchPhase::Refined {
            results,
            latency,
            rank_changes,
            ..
        } = phase
        else {
            return;
        };
        assert_eq!(results.len(), 1);
        assert_eq!(latency, Duration::from_millis(120));
        assert_eq!(rank_changes.total(), 20);
    }

    #[test]
    fn search_phase_reranked_construction() {
        let phase = SearchPhase::Reranked {
            results: vec![ScoredResult {
                doc_id: "r1".into(),
                score: 0.95,
                source: ScoreSource::Reranked,
                index: None,
                fast_score: Some(0.7),
                quality_score: Some(0.9),
                lexical_score: None,
                rerank_score: Some(0.95),
                explanation: None,
                metadata: None,
            }],
            latency: Duration::from_millis(450),
            metrics: PhaseMetrics {
                embedder_id: "flashrank".into(),
                vectors_searched: 2000,
                lexical_candidates: 100,
                fused_count: 20,
                skip_reason: None,
            },
        };
        assert!(matches!(phase, SearchPhase::Reranked { .. }));
        let SearchPhase::Reranked {
            results, latency, ..
        } = phase
        else {
            return;
        };
        assert_eq!(results.len(), 1);
        assert_eq!(latency, Duration::from_millis(450));
    }

    #[test]
    fn search_phase_debug() {
        let phase = SearchPhase::Initial {
            results: vec![],
            latency: Duration::from_millis(5),
            metrics: PhaseMetrics {
                embedder_id: "e".into(),
                vectors_searched: 0,
                lexical_candidates: 0,
                fused_count: 0,
                skip_reason: None,
            },
        };
        let dbg = format!("{phase:?}");
        assert!(dbg.contains("Initial"));
    }

    #[test]
    fn vector_hit_nan_sorts_below_real() {
        let real = VectorHit {
            index: 0,
            score: -100.0, // very low but real
            doc_id: "real".into(),
        };
        let nan = VectorHit {
            index: 1,
            score: f32::NAN,
            doc_id: "nan".into(),
        };
        // NaN maps to NEG_INFINITY, so even -100.0 beats it
        assert_eq!(real.cmp_by_score(&nan), std::cmp::Ordering::Less);
        assert_eq!(nan.cmp_by_score(&real), std::cmp::Ordering::Greater);
    }

    #[test]
    fn vector_hit_negative_scores_descending() {
        let a = VectorHit {
            index: 0,
            score: -0.1,
            doc_id: "a".into(),
        };
        let b = VectorHit {
            index: 1,
            score: -0.9,
            doc_id: "b".into(),
        };
        // -0.1 > -0.9, so a comes first (Less in descending order)
        assert_eq!(a.cmp_by_score(&b), std::cmp::Ordering::Less);
    }

    #[test]
    fn fused_hit_in_both_sources_tiebreak() {
        let both = FusedHit {
            doc_id: "z".into(), // worse doc_id
            rrf_score: 0.02,
            lexical_rank: Some(3),
            semantic_rank: Some(5),
            semantic_index: Some(5),
            lexical_score: None,
            semantic_score: None,
            in_both_sources: true,
        };
        let single = FusedHit {
            doc_id: "a".into(), // better doc_id
            rrf_score: 0.02,
            lexical_rank: Some(1),
            semantic_rank: None,
            semantic_index: None,
            lexical_score: None,
            semantic_score: None,
            in_both_sources: false,
        };
        // Same RRF -> in_both_sources=true wins (level 2)
        assert_eq!(both.cmp_for_ranking(&single), std::cmp::Ordering::Less);
    }

    // ─── bd-ta55 tests end ───
}
