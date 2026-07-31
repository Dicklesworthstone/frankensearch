use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use compact_str::CompactString;
use serde::{Deserialize, Serialize};

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
use crate::generation::EmbeddingIdentityBundleV1;
use crate::query_class::QueryClass;

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
#[derive(Debug, Clone)]
pub struct BoundQueryEmbedding {
    vector: Vec<f32>,
    identity: EmbeddingIdentityBundleV1,
    identity_fingerprint: String,
    space_fingerprint: String,
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
    /// Returns [`SearchError::InvalidConfig`] when the identity bundle fails
    /// its own validation, and [`SearchError::DimensionMismatch`] when the
    /// vector length does not match the validated identity's storage
    /// dimension — a bound embedding must be internally consistent before it
    /// can prove anything to a consumer.
    pub fn new(vector: Vec<f32>, identity: EmbeddingIdentityBundleV1) -> SearchResult<Self> {
        identity.validate()?;
        let expected = identity.storage.dimension as usize;
        if vector.len() != expected {
            return Err(SearchError::DimensionMismatch {
                expected,
                found: vector.len(),
            });
        }
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
    /// explicitly supplied space identity. Today **zero production writers
    /// call `create_v2`** — `TwoTierIndexBuilder::finish` routes through
    /// the legacy v1 constructors with `identity_v2: None` — so every
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
    /// - a different producer → admitted only when
    ///   [`EmbeddingIdentityBundleV1::is_conformance_compatible_with`]
    ///   verifies the pinned golden-vector certificate, returning
    ///   [`SpaceIdentityAdmission::ConformanceCompatibleProducer`] — typed
    ///   telemetry the caller is expected to record;
    /// - otherwise the pairing is rejected. A matching space fingerprint
    ///   alone never admits a foreign producer.
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
        if self.identity.is_conformance_compatible_with(expected) {
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
    /// ([`EmbeddingIdentityBundleV1::is_conformance_compatible_with`]).
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
#[derive(Debug, Clone)]
pub struct TieredQueryEmbeddings {
    fast: Option<BoundQueryEmbedding>,
    quality: Option<BoundQueryEmbedding>,
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
    #[must_use]
    pub const fn supported_topology(&self) -> RetrievalTopology {
        match (&self.fast, &self.quality) {
            (Some(_), Some(_)) => RetrievalTopology::FullProgressive,
            (Some(_), None) => RetrievalTopology::FastOnly,
            (None, Some(_)) => RetrievalTopology::QualityOnly,
            // Unreachable by construction; lexical-only is the honest floor.
            (None, None) => RetrievalTopology::LexicalOnly,
        }
    }
}

/// The retrieval shape a search actually ran with.
///
/// Requested versus realized topology is first-class telemetry (bd-9xuj):
/// a caller that asked for full progressive search and silently received a
/// hash-candidate pool rescored by the quality model was the defining
/// failure this vocabulary exists to make impossible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
    /// Hybrid fusion (lexical + semantic via RRF).
    Hybrid,
    /// Result was reranked by cross-encoder.
    Reranked,
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
        match error {
            SearchError::InvalidConfig { field, reason, .. } => {
                assert_eq!(field, "query_embedding.quality.identity");
                assert!(reason.contains("different embedding space"));
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
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
    /// [`EmbeddingIdentityBundleV1::is_conformance_compatible_with`]
    /// verifies the pinned golden-vector certificate, with typed
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
        let fast = BoundQueryEmbedding::new(vec![0.1; 8], identity("fast-model", 8)).unwrap();
        let quality =
            BoundQueryEmbedding::new(vec![0.2; 16], identity("quality-model", 16)).unwrap();

        let progressive = TieredQueryEmbeddings::progressive(fast.clone(), quality.clone());
        assert_eq!(
            progressive.supported_topology(),
            RetrievalTopology::FullProgressive
        );
        assert!(progressive.fast().is_some() && progressive.quality().is_some());

        assert_eq!(
            TieredQueryEmbeddings::fast_only(fast).supported_topology(),
            RetrievalTopology::FastOnly
        );
        let quality_only = TieredQueryEmbeddings::quality_only(quality);
        assert_eq!(
            quality_only.supported_topology(),
            RetrievalTopology::QualityOnly
        );
        assert!(quality_only.fast().is_none());
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
        ];
        for variant in &variants {
            let json = serde_json::to_string(variant).unwrap();
            let decoded: ScoreSource = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, variant);
        }
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
