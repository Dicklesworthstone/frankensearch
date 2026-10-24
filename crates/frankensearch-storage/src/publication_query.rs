//! Selection options and owned results for direct transactional publication queries.
//!
//! Unlike a reusable full-corpus snapshot, a direct query keeps only selected
//! publications. Query execution lives with the shared publication scanner and
//! exact scorer; these types do not introduce another admission or ranking path.

use std::collections::BTreeSet;

use frankensearch_core::VectorHit;

use crate::{PublishedEmbedding, PublishedEmbeddingScanStats, PublishedSearchDocument};

/// Exact result scope shared by vector-only and source-hydrated direct queries.
///
/// Membership is checked before top-k selection, not against an already
/// truncated result. IDs are compared in full, without hashing, normalization,
/// prefix matching or SQL interpolation. Unknown IDs simply do not match.
/// The set is borrowed and cannot change during the call or a transaction retry;
/// no user predicate executes while the database transaction is held.
///
/// This scopes results, not integrity admission: even an empty allow-set with
/// a positive limit scans and validates every current publication in the tier.
/// Read limits and scan statistics include excluded publications. Only final
/// winning sources are hydrated. A zero limit still admits the query but does
/// no database work. The caller-owned set's memory is outside the read budgets.
///
/// A set derived from earlier metadata is not re-authorized in the transaction.
/// This is not a live authorization or revocation mechanism. Use a captured
/// search snapshot for predicates over source-coherent metadata.
#[derive(Clone, Copy)]
pub struct PublishedQueryOptions<'a> {
    /// Maximum number of matching results, in best-first order.
    pub limit: usize,
    /// `None` admits all IDs; `Some(empty)` admits no result IDs.
    pub allowed_doc_ids: Option<&'a BTreeSet<String>>,
}

impl<'a> PublishedQueryOptions<'a> {
    /// Search the whole publication tier, retaining at most `limit` results.
    #[must_use]
    pub const fn new(limit: usize) -> Self {
        Self {
            limit,
            allowed_doc_ids: None,
        }
    }

    /// Restrict candidates to these exact IDs without copying the set.
    #[must_use]
    pub const fn with_doc_ids(mut self, ids: &'a BTreeSet<String>) -> Self {
        self.allowed_doc_ids = Some(ids);
        self
    }
}

impl std::fmt::Debug for PublishedQueryOptions<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PublishedQueryOptions")
            .field("limit", &self.limit)
            .field("allowed_doc_id_count", &self.allowed_doc_ids.map(BTreeSet::len))
            .finish()
    }
}

/// A scored winner retaining the exact publication that produced its score.
///
/// `hit().index` is its ordinal in the complete binary-ID-ordered scan, NOT an
/// index into the returned top-k list, a database rowid or a durable identifier.
/// The publication retains its full producer identity, source hash and attempt.
#[derive(Clone)]
pub struct PublishedEmbeddingQueryHit {
    pub(crate) hit: VectorHit,
    pub(crate) publication: PublishedEmbedding,
}

impl PublishedEmbeddingQueryHit {
    #[must_use]
    pub const fn hit(&self) -> &VectorHit {
        &self.hit
    }

    #[must_use]
    pub const fn publication(&self) -> &PublishedEmbedding {
        &self.publication
    }
}

impl std::fmt::Debug for PublishedEmbeddingQueryHit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PublishedEmbeddingQueryHit")
            .field("scan_index", &self.hit.index)
            .field("score", &self.hit.score)
            .finish_non_exhaustive()
    }
}

/// Exact best-first results owned independently of the database connection.
///
/// Only selected vectors are retained. This is not a reusable corpus snapshot
/// or evidence of unpublished/pending coverage; each new query reads a new
/// transaction. The optional scan accounting distinguishes a completed empty
/// scan from a zero-limit request that performed no database work.
#[derive(Clone)]
pub struct PublishedEmbeddingQueryResult {
    pub(crate) hits: Vec<PublishedEmbeddingQueryHit>,
    pub(crate) scan_stats: Option<PublishedEmbeddingScanStats>,
}

impl PublishedEmbeddingQueryResult {
    #[must_use]
    pub fn hits(&self) -> &[PublishedEmbeddingQueryHit] {
        &self.hits
    }

    /// Complete scan accounting, or `None` for a zero-limit no-op.
    #[must_use]
    pub const fn scan_stats(&self) -> Option<PublishedEmbeddingScanStats> {
        self.scan_stats
    }
}

impl std::fmt::Debug for PublishedEmbeddingQueryResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PublishedEmbeddingQueryResult")
            .field("hits", &self.hits.len())
            .field("scan_stats", &self.scan_stats)
            .finish_non_exhaustive()
    }
}

/// An owned scored publication and its verified canonical source.
///
/// The source, catalog metadata and vector were read in the SAME transaction.
/// They remain the scored revision after a later update or database close.
#[derive(Clone)]
pub struct PublishedSearchQueryHit {
    pub(crate) embedding: PublishedEmbeddingQueryHit,
    pub(crate) document: PublishedSearchDocument,
}

impl PublishedSearchQueryHit {
    #[must_use]
    pub const fn hit(&self) -> &VectorHit {
        self.embedding.hit()
    }

    #[must_use]
    pub const fn publication(&self) -> &PublishedEmbedding {
        self.embedding.publication()
    }

    #[must_use]
    pub const fn document(&self) -> &PublishedSearchDocument {
        &self.document
    }
}

impl std::fmt::Debug for PublishedSearchQueryHit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PublishedSearchQueryHit")
            .field("embedding", &self.embedding)
            .finish_non_exhaustive()
    }
}

/// Direct exact query with source hydration restricted to the final winners.
///
/// All current publications were validated, but ONLY winning sources were
/// loaded and checked. This is not a full-corpus source integrity audit or a
/// reusable captured corpus. It does not apply metadata authorization filters.
#[derive(Clone)]
pub struct PublishedSearchQueryResult {
    pub(crate) hits: Vec<PublishedSearchQueryHit>,
    pub(crate) scan_stats: Option<PublishedEmbeddingScanStats>,
    pub(crate) source_payload_bytes: usize,
}

impl PublishedSearchQueryResult {
    #[must_use]
    pub fn hits(&self) -> &[PublishedSearchQueryHit] {
        &self.hits
    }

    /// Complete vector scan accounting, or `None` for a zero-limit no-op.
    #[must_use]
    pub const fn scan_stats(&self) -> Option<PublishedEmbeddingScanStats> {
        self.scan_stats
    }

    /// Stored canonical text, preview, path and metadata bytes of winners only.
    #[must_use]
    pub const fn source_payload_bytes(&self) -> usize {
        self.source_payload_bytes
    }
}

impl std::fmt::Debug for PublishedSearchQueryResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PublishedSearchQueryResult")
            .field("hits", &self.hits.len())
            .field("scan_stats", &self.scan_stats)
            .field("source_payload_bytes", &self.source_payload_bytes)
            .finish_non_exhaustive()
    }
}
