//! Optional HNSW approximate nearest-neighbor index (`ann` feature).
//!
//! This module wraps `hnsw_rs` behind a frankensearch-native API.
//!
//! # Persistence
//!
//! The metadata sidecar (e.g. `vector.fast.hnsw`) stores `doc_ids`, config and
//! dimension as JSON. Since format v2 the native `hnsw_rs` graph is also
//! persisted in sidecars beside it, so `load()` deserializes the prebuilt graph
//! directly instead of rebuilding it from vectors. Format v4 fingerprints
//! every live source vector so a stale graph cannot survive an unsampled vector
//! change. Format v5 records the exact
//! generation directory and basename selected during atomic publication, so no
//! save truncates the pair named by installed metadata. A persistent advisory
//! save lock and durable in-generation READY receipt serialize writers, let
//! publication retries reuse complete generations, and reclaim generations
//! superseded by a successful metadata publication.
//! Format v6 attests the native graph's point and layer topology, invalidating
//! graphs produced by `hnsw_rs` versions that could misfile reverse edges.
//! Connectivity is attested as one weak base-layer component; directed
//! coverage from the search entry is measured and warned about but is not an
//! admission condition, because bounded neighbour pruning legitimately orphans
//! in-edges at scale (#32).
//! Legacy sidecars and any load failure fall back to the
//! rebuild-from-`VectorIndex` path.

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use frankensearch_core::config::ZeroSignalReason;
use frankensearch_core::generation::ArtifactGenerationIdentityV1;
use frankensearch_core::{SearchError, SearchResult, VectorHit};
use hnsw_rs::hnswio::ReloadOptions;
use hnsw_rs::prelude::{AnnT, DistDot, Hnsw, HnswIo, Neighbour, PointId};
use serde::{Deserialize, Serialize};

use crate::recall_certificate::{EfCalibration, calibrate_certified_ef};
use crate::{SHA256_BYTES, VectorIndex};

/// Default HNSW `M` (max connections per node).
pub const HNSW_DEFAULT_M: usize = 16;
/// Default HNSW `ef_construction` (build-time beam width).
pub const HNSW_DEFAULT_EF_CONSTRUCTION: usize = 200;
/// Default HNSW `ef_search` (query-time beam width).
pub const HNSW_DEFAULT_EF_SEARCH: usize = 100;
/// Default HNSW max layer depth.
pub const HNSW_DEFAULT_MAX_LAYER: usize = 16;

// `hnsw_rs` documents parallel insertion as efficient only for batches of
// roughly 1,000 points per Rayon worker. Small batches also have no useful
// parallel speedup, so keep their construction deterministic and serial.
const HNSW_PARALLEL_INSERT_MIN_POINTS_PER_THREAD: usize = 1_000;

/// ANN construction/runtime parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HnswConfig {
    /// HNSW `M` (max connections per node).
    pub m: usize,
    /// HNSW `ef_construction` (build-time beam width).
    pub ef_construction: usize,
    /// Default HNSW `ef_search` (query-time beam width).
    pub ef_search: usize,
    /// Maximum HNSW layer depth.
    pub max_layer: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: HNSW_DEFAULT_M,
            ef_construction: HNSW_DEFAULT_EF_CONSTRUCTION,
            ef_search: HNSW_DEFAULT_EF_SEARCH,
            max_layer: HNSW_DEFAULT_MAX_LAYER,
        }
    }
}

/// Current on-disk metadata format. v2 added the native graph sidecars
/// (`*.hnsw.graph` + `*.hnsw.data`) alongside the JSON metadata; v3 records
/// graphs built with the dimension-aware `DistDot` roundoff budget; v4 replaces
/// the sampled source fingerprint with a digest of every live vector; v5 records
/// the exact native sidecar generation and basename selected during publication;
/// v6 attests point/layer invariants after build and native load. Older native
/// graphs must be rebuilt under the current persistence contract.
pub(crate) const HNSW_META_FORMAT_CURRENT: u32 = 6;

const HNSW_GENERATION_RECEIPT_VERSION: u32 = 1;
const HNSW_GENERATION_RECEIPT_FILENAME: &str = ".frankensearch-hnsw-ready.json";
const HNSW_GENERATION_RECEIPT_MAX_BYTES: usize = 64 * 1024;
const HNSW_SAVE_LOCK_DIRECTORY: &str = ".frankensearch-hnsw-save-locks";

type HnswMetadataPublisher = fn(&Path, &Path, &[u8]) -> SearchResult<()>;

// Keep the classical gamma_k floating-point error model in its well-conditioned
// region. With u = f32::EPSILON / 2 and k = 8n + 32, this cap is exactly the
// largest dimension for which k*u <= 1/4.
const DIST_DOT_MAX_DIMENSION: usize = 524_284;

#[derive(Debug, Clone, Copy)]
struct DistDotBudget {
    radius_squared: f32,
    score_tolerance: f32,
}

/// On-disk metadata for the HNSW index.
#[derive(Debug, Serialize, Deserialize)]
struct HnswMeta {
    /// Sidecar format. Absent in legacy v1 metadata (deserializes to 0).
    #[serde(default)]
    format_version: u32,
    doc_ids: Vec<String>,
    config: HnswConfig,
    dimension: usize,
    /// Deterministic fingerprint of the vectors the persisted graph was built
    /// from (FNV-1a 64 over every live f32 vector; see [`fingerprint_vectors`]).
    /// Lets native-graph load detect "doc IDs match
    /// but the underlying vectors were silently swapped" and fall back to a
    /// rebuild rather than serve stale ANN hits.
    ///
    /// Absent in legacy metadata (deserializes to 0). Legacy formats rebuild
    /// before the native fast path; current-format sidecars compare 0 like any
    /// other fingerprint and therefore cannot use omission to bypass validation.
    #[serde(default)]
    vector_fingerprint: u64,
    /// Directory containing the native graph/data pair, relative to metadata.
    /// Every distinct graph state owns an atomically created generation so
    /// `file_dump` never truncates the pair referenced by installed metadata.
    #[serde(default)]
    sidecar_generation: Option<String>,
    /// Basename shared by the native `.hnsw.graph` and `.hnsw.data` files.
    ///
    /// A loaded `hnsw_rs` graph refuses to overwrite an occupied dump and
    /// returns a randomized basename instead. Persisting that returned value
    /// makes the metadata commit point authoritative. Missing location fields
    /// invalidate current-format native loading and force a rebuild.
    #[serde(default)]
    sidecar_basename: Option<String>,
    /// Exact identity of the FSVI generation this graph was built from
    /// (`bd-r65a`).
    ///
    /// Absent in every sidecar written before identity binding, and absent
    /// when the source index is a legacy v1 artifact that has no identity to
    /// bind. Absence is never treated as a match: see
    /// [`HnswSourceIdentityV1::admits`].
    #[serde(default)]
    source_identity: Option<HnswSourceIdentityV1>,
}

/// The source-generation identity a persisted ANN sidecar is bound to.
///
/// Content equality is NOT identity equality. `vector_fingerprint` and the
/// doc-id sequence prove the graph indexes the same vectors in the same order;
/// they cannot prove those vectors came from the same published generation or
/// the same embedding space. Two FSVI generations can hold byte-identical live
/// content and still be different artifacts — a re-publication carries a new
/// generation nonce, and a model revision bump changes the space fingerprint
/// without necessarily changing a single vector. Serving a sidecar across that
/// boundary is silent reuse of a stale graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct HnswSourceIdentityV1 {
    /// Full-width artifact generation of the source FSVI.
    generation: ArtifactGenerationIdentityV1,
    /// SHA-256 of the canonical full-width generation identity.
    generation_fingerprint: [u8; SHA256_BYTES],
    /// SHA-256 of the complete embedding identity bundle.
    identity_bundle_fingerprint: [u8; SHA256_BYTES],
    /// SHA-256 of the mathematical embedding space.
    space_fingerprint: [u8; SHA256_BYTES],
    /// SHA-256 of the physical vector storage identity (format, quantization,
    /// endianness).
    storage_fingerprint: [u8; SHA256_BYTES],
    /// SHA-256 of the ordered live document identifiers.
    ordered_live_docset_digest: [u8; SHA256_BYTES],
}

impl HnswSourceIdentityV1 {
    /// Capture the identity of `index`, or `None` for a legacy v1 source that
    /// carries no identity at all.
    fn capture(index: &VectorIndex) -> Option<Self> {
        let identity = index.metadata().identity_v2.as_ref()?;
        Some(Self {
            generation: identity.generation,
            generation_fingerprint: identity.generation_fingerprint,
            identity_bundle_fingerprint: identity.identity_bundle_fingerprint,
            space_fingerprint: identity.space_fingerprint,
            storage_fingerprint: identity.storage_fingerprint,
            ordered_live_docset_digest: identity.ordered_live_docset_digest,
        })
    }

    /// Whether a sidecar bound to `persisted` may be served against `live`.
    ///
    /// Both directions of absence fail closed. A sidecar with no recorded
    /// identity cannot be proven to belong to an identity-bearing generation,
    /// and a sidecar that names a generation cannot be served against a source
    /// that cannot name one. The only admissible cases are "both absent"
    /// (legacy v1 on both sides, where there is nothing to bind and the
    /// content checks stand alone) and "both present and equal".
    fn admits(persisted: Option<&Self>, live: Option<&Self>) -> bool {
        match (persisted, live) {
            (None, None) => true,
            (Some(persisted), Some(live)) => persisted == live,
            _ => false,
        }
    }
}

/// Durable proof that an immutable native generation finished writing before
/// metadata publication was attempted. A later save can validate and reuse the
/// generation after an interrupted or failed publication without deleting it
/// or dumping another complete copy.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct HnswGenerationReceipt {
    receipt_version: u32,
    metadata_file_name: String,
    format_version: u32,
    generation: String,
    sidecar_basename: String,
    doc_count: usize,
    doc_ids_fingerprint: u64,
    vector_fingerprint: u64,
    dimension: usize,
    config: HnswConfig,
    graph: HnswSidecarDigest,
    data: HnswSidecarDigest,
    /// Source FSVI generation identity this generation was dumped from
    /// (`bd-21zyj`). Absent in receipts written before identity binding, and
    /// absent for a legacy v1 source with no identity. Absence never matches
    /// a bound graph: see [`HnswSourceIdentityV1::admits`].
    #[serde(default)]
    source_identity: Option<HnswSourceIdentityV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct HnswSidecarDigest {
    byte_len: u64,
    fnv1a64: u64,
}

#[derive(Debug)]
struct ValidatedHnswGeneration {
    generation: String,
    basename: String,
    graph: PathBuf,
    /// Identity recorded in the generation's own receipt (`bd-21zyj`).
    source_identity: Option<HnswSourceIdentityV1>,
}

/// How an HNSW load obtained its in-memory graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HnswLoadDisposition {
    /// The current native graph/data pair was deserialized from disk.
    Native,
    /// Metadata was readable, but the graph had to be rebuilt from the source index.
    Rebuilt,
}

/// Diagnostics for one ANN query.
#[derive(Debug, Clone, PartialEq)]
pub struct AnnSearchStats {
    /// Number of vectors indexed.
    pub index_size: usize,
    /// Vector dimensionality.
    pub dimension: usize,
    /// Effective ef used for this query.
    pub ef_search: usize,
    /// Requested `k`.
    pub k_requested: usize,
    /// Returned result count.
    pub k_returned: usize,
    /// Query latency in microseconds.
    pub search_time_us: u64,
    /// Whether this path is approximate ANN.
    pub is_approximate: bool,
    /// Why the query fell back to an exact scan, if it did.
    pub fallback_reason: Option<AnnFallbackReason>,
    /// Estimated recall@k from the ef/k ratio (see [`estimate_recall`]).
    ///
    /// This is a heuristic point estimate with NO guarantee. For a certified,
    /// distribution-free recall bound (the automated replacement for a human
    /// recall-budget sign-off), use
    /// [`crate::recall_certificate::conformal_recall_lower_bound`] over a measured
    /// calibration sample instead.
    pub estimated_recall: f64,
    /// Typed classification when the query produced zero hits.
    ///
    /// `Some(reason)` if and only if the returned hit list is empty. Present
    /// so the ANN lane classifies zero-signal states with the same
    /// [`ZeroSignalReason`] vocabulary as the exact lane (bd-tqhc).
    pub zero_signal: Option<ZeroSignalReason>,
}

/// Reason an ANN query returned exact rather than approximate results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnnFallbackReason {
    /// The native HNSW search returned fewer unique neighbors than exist for
    /// the requested `k`, so an exact scan repaired the result set.
    Underfilled,
    /// The native HNSW search returned zero candidates although the graph
    /// indexes points. Stronger anomaly signal than a partial underfill:
    /// the graph produced no signal at all and an exact scan repaired it.
    EmptyDespiteIndexedPoints,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HnswHitResolution {
    CanonicalPublic,
    RawPhysical,
}

#[cfg(test)]
#[derive(Debug, Clone)]
struct HnswTopologyPoint {
    origin_id: usize,
    point_id: PointId,
    neighborhoods: Vec<Vec<Neighbour>>,
}

/// HNSW ANN index over vectors aligned to `VectorIndex` row order.
pub struct HnswIndex {
    hnsw: Hnsw<'static, f32, DistDot>,
    doc_ids: Vec<String>,
    /// Maps compact HNSW origin ids to canonical persisted `VectorIndex` rows.
    source_positions: Vec<u32>,
    /// Physical main-slab extent at construction/load time.
    ///
    /// Soft deletion leaves this unchanged and is supported between rebuilds.
    /// A different extent means the borrowed exact-repair source is no longer
    /// the immutable main slab this graph indexes.
    source_record_count: usize,
    dimension: usize,
    config: HnswConfig,
    /// Fingerprint of the vectors the graph was built from. See
    /// [`HnswMeta::vector_fingerprint`].
    vector_fingerprint: u64,
    /// Exact identity of the source FSVI generation, when it has one
    /// (`bd-r65a`). `None` for a legacy v1 source with no identity to bind.
    source_identity: Option<HnswSourceIdentityV1>,
    /// Whether this graph instance has already warned about an underfill.
    ///
    /// Graph instances are per-generation (rebuilt on reload), so gating the
    /// warning here yields the required once-per-generation bound; repeat
    /// underfills log at debug (bd-tqhc no-warn-storm policy).
    underfill_warned: AtomicBool,
}

impl std::fmt::Debug for HnswIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HnswIndex")
            .field("points", &self.hnsw.get_nb_point())
            .field("doc_ids", &self.doc_ids.len())
            .field("dimension", &self.dimension)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl HnswIndex {
    /// Build a new HNSW index from an opened `VectorIndex`.
    ///
    /// # Errors
    ///
    /// Returns:
    /// - `SearchError::InvalidConfig` for invalid HNSW params
    /// - `SearchError::IndexCorrupted` if `vectors/doc_ids` cannot be decoded
    pub fn build_from_vector_index(index: &VectorIndex, config: HnswConfig) -> SearchResult<Self> {
        let dimension = index.dimension();
        let mut doc_ids = Vec::with_capacity(index.record_count());
        let mut vectors = Vec::with_capacity(index.record_count());
        let mut live_positions = Vec::with_capacity(index.record_count());
        for i in 0..index.record_count() {
            if index.is_deleted(i) {
                continue;
            }
            doc_ids.push(index.doc_id_at(i)?.to_owned());
            vectors.push(index.vector_at_f32(i)?);
            live_positions.push(i);
        }
        let mut ann = Self::build_from_parts(doc_ids, vectors, dimension, config)?;
        // Bind the graph to the exact generation it was built from, so a later
        // load cannot serve it against a different one (bd-r65a).
        ann.source_identity = HnswSourceIdentityV1::capture(index);
        ann.source_record_count = index.record_count();
        ann.source_positions = live_positions
            .into_iter()
            .map(|position| {
                u32::try_from(position).map_err(|_| SearchError::InvalidConfig {
                    field: "source_position".to_owned(),
                    value: position.to_string(),
                    reason: "VectorIndex row exceeds u32".to_owned(),
                })
            })
            .collect::<SearchResult<Vec<_>>>()?;
        Ok(ann)
    }

    /// Load an ANN index from disk, rebuilding the graph from `source_index` when
    /// the native graph/data pair is legacy, stale, missing, or corrupt.
    ///
    /// The source index validates the persisted document sequence and vector
    /// fingerprint. A fallback rebuild reads every live row from that same
    /// source, preserving row-to-vector alignment even when document IDs are
    /// not unique.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::IndexCorrupted` if metadata is missing/malformed or
    /// if live rows cannot be decoded from `source_index`.
    pub fn load(path: &Path, source_index: &VectorIndex) -> SearchResult<Self> {
        Self::load_with_disposition(path, source_index).map(|(index, _)| index)
    }

    /// Try to load only the exact persisted native graph paired with
    /// `source_index`, without ever rebuilding it.
    ///
    /// `Ok(Some(index))` proves the current native graph/data generation,
    /// digest receipt, ordered document IDs, vector fingerprint, dimension,
    /// and topology all match `source_index`. `Ok(None)` means the metadata is
    /// readable but the selected graph is legacy, stale, incomplete, corrupt,
    /// or otherwise not admissible as that native artifact. This method never
    /// scans source rows to construct a replacement graph and never writes,
    /// replaces, renames, or changes permissions or mtimes.
    ///
    /// # Errors
    ///
    /// Returns an error when the metadata file cannot be read or parsed, its
    /// dimension disagrees with `source_index`, or its dimension exceeds the
    /// supported distance-kernel budget.
    pub fn try_load_native(path: &Path, source_index: &VectorIndex) -> SearchResult<Option<Self>> {
        let meta = Self::validated_load_metadata(path, source_index)?;
        if meta.format_version != HNSW_META_FORMAT_CURRENT {
            return Ok(None);
        }
        Ok(Self::try_load_native_graph(path, &meta, source_index))
    }

    /// Load an ANN index and report whether its graph came from native sidecars
    /// or was rebuilt from `source_index`.
    ///
    /// This is the fail-closed inspection surface for consumers that must
    /// distinguish the exact persisted ANN artifact from a compatible
    /// in-memory fallback. Neither outcome writes, replaces, renames, or
    /// changes permissions on `path`, its native sidecars, or `source_index`.
    /// Callers that require the selected on-disk graph must accept only
    /// [`HnswLoadDisposition::Native`].
    ///
    /// # Errors
    ///
    /// Returns `SearchError::IndexCorrupted` when metadata is unreadable,
    /// malformed, dimensionally incompatible, or the source rows cannot be
    /// decoded. A readable legacy/stale/corrupt native graph is instead
    /// reported as [`HnswLoadDisposition::Rebuilt`] after an in-memory rebuild.
    pub fn load_with_disposition(
        path: &Path,
        source_index: &VectorIndex,
    ) -> SearchResult<(Self, HnswLoadDisposition)> {
        let meta = Self::validated_load_metadata(path, source_index)?;

        // Current format: deserialize the prebuilt native graph directly, skipping the
        // O(n log n) rebuild. Any problem (missing/corrupt sidecars, point-count
        // mismatch, or stale vector fingerprint) returns None and we fall
        // through to the rebuild path, so a bad graph sidecar degrades to
        // "slow load" rather than a hard failure.
        if meta.format_version == HNSW_META_FORMAT_CURRENT
            && let Some(index) = Self::try_load_native_graph(path, &meta, source_index)
        {
            return Ok((index, HnswLoadDisposition::Native));
        } else if meta.format_version != HNSW_META_FORMAT_CURRENT {
            tracing::warn!(
                path = %path.display(),
                format_version = meta.format_version,
                current_format_version = HNSW_META_FORMAT_CURRENT,
                "rebuilding HNSW sidecar written with a different persistence contract; \
                 re-save to skip rebuild on the next cold load"
            );
        }

        // v1/legacy or fallback: rebuild directly from live source rows. Looking
        // vectors up by doc ID would collapse duplicate IDs onto their first
        // occurrence and silently attach the wrong vector to later rows.
        Self::build_from_vector_index(source_index, meta.config)
            .map(|index| (index, HnswLoadDisposition::Rebuilt))
    }

    fn validated_load_metadata(path: &Path, source_index: &VectorIndex) -> SearchResult<HnswMeta> {
        let metadata_bytes = std::fs::read(path).map_err(SearchError::Io)?;
        let meta: HnswMeta = serde_json::from_slice(&metadata_bytes)
            .map_err(|e| ann_corrupted(path, format!("failed to parse HNSW metadata: {e}")))?;

        if meta.dimension != source_index.dimension() {
            return Err(ann_corrupted(
                path,
                format!(
                    "dimension mismatch: hnsw={} source={}",
                    meta.dimension,
                    source_index.dimension()
                ),
            ));
        }

        // Validate before the native fast path. Otherwise a forged current-format sidecar
        // could bypass the dimension bound enforced by graph construction.
        dist_dot_budget(meta.dimension)?;
        Ok(meta)
    }

    /// Attempt to load the prebuilt native `hnsw_rs` graph for a current-format sidecar.
    ///
    /// Returns `None` if any of the following hold. Permissive callers may
    /// rebuild from the source index; strict callers can reject the artifact:
    /// - the `.hnsw.graph` / `.hnsw.data` sidecars are absent,
    /// - `hnsw_rs` fails to deserialize them,
    /// - the loaded point count disagrees with the metadata `doc_ids`,
    /// - the metadata `doc_ids` disagree with the live `VectorIndex`'s
    ///   doc-id sequence (live tombstones excluded),
    /// - the persisted vector fingerprint disagrees with the live
    ///   `VectorIndex`'s fingerprint (i.e. vectors were swapped behind the
    ///   same doc ids — the case the prompt explicitly calls out).
    fn try_load_native_graph(
        path: &Path,
        meta: &HnswMeta,
        source_index: &VectorIndex,
    ) -> Option<Self> {
        let (sidecar_parent, basename) = persisted_hnsw_sidecar_location(path, meta).ok()?;
        let graph = sidecar_parent.join(format!("{basename}.hnsw.graph"));
        let data = sidecar_parent.join(format!("{basename}.hnsw.data"));
        if !native_sidecar_pair_is_local(path, &sidecar_parent, &graph, &data) {
            return None;
        }
        let metadata_file_name = hnsw_metadata_file_name(path).ok()?;
        let validated_generation = match validate_hnsw_generation_receipt(
            path,
            &sidecar_parent,
            &metadata_file_name,
            &meta.doc_ids,
            meta.vector_fingerprint,
            meta.dimension,
            meta.config,
        ) {
            Ok(Some(validated)) => validated,
            Ok(None) => {
                tracing::warn!(
                    path = %path.display(),
                    "HNSW native sidecars lack a matching digest receipt; native load unavailable"
                );
                return None;
            }
            Err(error) => {
                tracing::warn!(
                    path = %path.display(),
                    ?error,
                    "HNSW native sidecar receipt validation failed; native load unavailable"
                );
                return None;
            }
        };
        if validated_generation.basename != basename || validated_generation.graph != graph {
            tracing::warn!(
                path = %path.display(),
                "HNSW metadata and digest receipt name different native sidecars; \
                 native load unavailable"
            );
            return None;
        }

        // Validate the SOURCE GENERATION IDENTITY before anything else about
        // the content (bd-r65a). Content equality is not identity equality:
        // two generations can hold byte-identical live vectors under the same
        // doc ids and still be different published artifacts — a
        // re-publication carries a new generation nonce, and a model revision
        // bump moves the space fingerprint without necessarily moving a single
        // vector. Every check below this one would pass in that case, so
        // ordering this first is what stops a stale graph being served.
        let live_identity = HnswSourceIdentityV1::capture(source_index);
        if !HnswSourceIdentityV1::admits(meta.source_identity.as_ref(), live_identity.as_ref()) {
            tracing::warn!(
                path = %path.display(),
                sidecar_bound = meta.source_identity.is_some(),
                source_bound = live_identity.is_some(),
                "HNSW sidecar is bound to a different FSVI generation identity than the live \
                 source; native load unavailable and the graph will be rebuilt"
            );
            return None;
        }

        // Validate doc-id sequence against the live VectorIndex *before*
        // touching the (potentially expensive) hnsw_rs load.
        if !meta_matches_live_doc_ids(meta, source_index).ok()? {
            tracing::warn!(
                path = %path.display(),
                "HNSW sidecar doc_ids disagree with live VectorIndex; native load unavailable"
            );
            return None;
        }

        // Validate the vector-content fingerprint against the live VectorIndex.
        // This is the critical stale-vectors guard: if a caller swaps the FSVI
        // contents while keeping the same doc IDs in the same order, the
        // persisted graph would otherwise silently serve hits against vectors
        // that no longer exist. `try_load_native_graph` is only called for the
        // current format, so a missing fingerprint cannot be treated as a
        // legacy exception: 0 is compared like any other digest value.
        let live_fp =
            fingerprint_live_vector_index(source_index, meta.doc_ids.len(), meta.dimension).ok()?;
        if live_fp != meta.vector_fingerprint {
            tracing::warn!(
                path = %path.display(),
                expected = meta.vector_fingerprint,
                actual = live_fp,
                "HNSW sidecar vector fingerprint disagrees with live VectorIndex \
                 (vectors swapped behind matching doc ids); native load unavailable"
            );
            return None;
        }

        // `HnswIo::load_hnsw` returns an `Hnsw` borrowed from the `HnswIo`
        // (`'a: 'b`), so to store it in the `'static` field we must keep the
        // `HnswIo` alive for the program. Load with `ReloadOptions::new(true)`
        // so the immutable `.hnsw.data` sidecar's vector payload stays
        // file-backed (read-only mmap) instead of being copied into a second
        // owned per-point heap: at multi-million-vector scale the owned copy
        // alone is gigabytes per cold process. Graph topology is still parsed
        // into owned state and attested below, and every admission check
        // (receipt, locality, doc-id/fingerprint binding) runs before the map
        // is created. Leaking the reloader (rather than a self-referential
        // struct or unsafe lifetime transmute) is the simplest sound way to
        // obtain a `'static` graph, keeps the mmap alive for exactly the graph
        // lifetime, and happens about once per process.
        let native_io: &'static mut HnswIo = Box::leak(Box::new(HnswIo::new_with_options(
            &sidecar_parent,
            &basename,
            ReloadOptions::new(true),
        )));
        let hnsw = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            // Moving the leaked mutable reference into a closure-local binding
            // makes this closure `FnOnce` and lets the parser's returned graph
            // carry the same `'static` lifetime as its deliberately leaked
            // owner.
            let native_io = native_io;
            native_io.load_hnsw::<'static, 'static, f32, DistDot>()
        })) {
            Ok(Ok(hnsw)) => hnsw,
            Ok(Err(error)) => {
                tracing::warn!(
                    path = %path.display(),
                    ?error,
                    "HNSW native sidecar parser rejected the installed generation; \
                     native load unavailable"
                );
                return None;
            }
            Err(_) => {
                tracing::warn!(
                    path = %path.display(),
                    "HNSW native sidecar parser panicked on the installed generation; \
                     native load unavailable"
                );
                return None;
            }
        };

        // Guard against a graph that doesn't match the metadata it shipped with
        // (e.g. truncated dump, mismatched sidecars). The caller additionally
        // validates doc_ids against the live VectorIndex.
        if hnsw.get_nb_point() != meta.doc_ids.len() {
            return None;
        }
        if let Err(detail) = validate_hnsw_topology(&hnsw, meta.doc_ids.len()) {
            tracing::warn!(
                path = %path.display(),
                %detail,
                "HNSW native sidecar failed topology attestation; native load unavailable"
            );
            return None;
        }

        Some(Self {
            hnsw,
            underfill_warned: AtomicBool::new(false),
            doc_ids: meta.doc_ids.clone(),
            source_positions: live_vector_positions(source_index)
                .into_iter()
                .map(u32::try_from)
                .collect::<Result<Vec<_>, _>>()
                .ok()?,
            source_record_count: source_index.record_count(),
            dimension: meta.dimension,
            config: meta.config,
            vector_fingerprint: meta.vector_fingerprint,
            source_identity: meta.source_identity.clone(),
        })
    }

    /// Persist the ANN index to disk.
    ///
    /// Writes the JSON metadata sidecar (`doc_ids`, config, dimension) at
    /// `path`, plus the native `hnsw_rs` graph and data pair next to it. The
    /// metadata records the exact generation directory and basename returned by
    /// `hnsw_rs`. A new graph state dumps into a fresh immutable generation, so
    /// neither a newly built nor mmap-backed loaded graph can truncate the
    /// currently installed pair. Equivalent retries validate and reuse a
    /// durable READY generation left by an uncertain metadata publication.
    /// Vectors are embedded in the native data sidecar; the `VectorIndex` is
    /// only consulted on a legacy or fallback rebuild.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::Io` on write/dump failure.
    pub fn save(&self, path: &Path) -> SearchResult<()> {
        self.save_with_metadata_publisher(path, publish_hnsw_metadata)
    }

    fn save_with_metadata_publisher(
        &self,
        path: &Path,
        publish_metadata: HnswMetadataPublisher,
    ) -> SearchResult<()> {
        let parent = path
            .parent()
            .filter(|dir| !dir.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        std::fs::create_dir_all(parent)?;

        let requested_basename = hnsw_sidecar_basename(path)?;
        let metadata_file_name = hnsw_metadata_file_name(path)?;
        // Serialize writers before any generation is staged. The lock file
        // lives in a reserved sibling namespace and remains persistent because
        // removing or replacing it creates an inode race in which two processes
        // can each hold a different lock.
        let _save_lock = acquire_hnsw_save_lock(path)?;

        if let Some(meta) = find_reusable_hnsw_generation(
            self,
            path,
            parent,
            &requested_basename,
            &metadata_file_name,
        )? {
            // A receipt can survive a crash before the generation's parent
            // directory entry was durable. Re-sync the parent before making
            // metadata point at it.
            sync_hnsw_directory(parent)?;
            let metadata_bytes = serialize_hnsw_metadata(&meta)?;
            publish_metadata(path, parent, &metadata_bytes)?;
            gc_superseded_hnsw_generations(parent, &requested_basename, &meta)?;
            return Ok(());
        }

        // Persist into a unique generation first. `hnsw_rs` truncates an
        // occupied basename for freshly built graphs and uses a racy random
        // suffix for loaded graphs; an atomically created directory avoids both
        // behaviors. Metadata remains the sole commit point.
        let generation_prefix =
            hnsw_generation_prefix(&requested_basename, self.vector_fingerprint);
        let mut generation_builder = tempfile::Builder::new();
        generation_builder.prefix(&generation_prefix);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            // Tempfile defaults to a private 0700 directory. Published ANN
            // generations must instead inherit the caller's umask, just like
            // the FSVI artifact in the same parent directory.
            generation_builder.permissions(std::fs::Permissions::from_mode(0o777));
        }
        let generation = generation_builder
            .tempdir_in(parent)
            .map_err(SearchError::Io)?;
        let dumped_basename = self
            .hnsw
            .file_dump(generation.path(), &requested_basename)
            .map_err(|error| {
                SearchError::Io(std::io::Error::other(format!(
                    "failed to dump HNSW graph: {error}"
                )))
            })?;
        let dumped_basename = validate_hnsw_sidecar_basename(path, &dumped_basename)?;
        sync_hnsw_sidecars(generation.path(), &dumped_basename)?;
        let generation_name = generation
            .path()
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| ann_corrupted(path, "HNSW generation has no UTF-8 directory name"))?;
        let generation_name = validate_hnsw_sidecar_basename(path, generation_name)?;
        let meta = self.metadata_for_generation(&generation_name, &dumped_basename);
        let graph_path = generation
            .path()
            .join(format!("{dumped_basename}.hnsw.graph"));
        let data_path = generation
            .path()
            .join(format!("{dumped_basename}.hnsw.data"));
        let receipt = HnswGenerationReceipt {
            receipt_version: HNSW_GENERATION_RECEIPT_VERSION,
            metadata_file_name,
            format_version: HNSW_META_FORMAT_CURRENT,
            generation: generation_name,
            sidecar_basename: dumped_basename,
            doc_count: self.doc_ids.len(),
            doc_ids_fingerprint: fingerprint_doc_ids(&self.doc_ids),
            vector_fingerprint: self.vector_fingerprint,
            dimension: self.dimension,
            config: self.config,
            source_identity: self.source_identity.clone(),
            graph: fingerprint_hnsw_sidecar(&graph_path)?,
            data: fingerprint_hnsw_sidecar(&data_path)?,
        };
        write_hnsw_generation_receipt(generation.path(), &receipt)?;

        // From this point onward every retained complete generation has a
        // durable receipt and is recoverable by a later save. Metadata remains
        // the atomic authority visible to readers.
        let _generation_path = generation.keep();
        sync_hnsw_directory(parent)?;

        let metadata_bytes = serialize_hnsw_metadata(&meta)?;
        publish_metadata(path, parent, &metadata_bytes)?;
        gc_superseded_hnsw_generations(parent, &requested_basename, &meta)?;

        Ok(())
    }

    fn metadata_for_generation(&self, generation: &str, basename: &str) -> HnswMeta {
        HnswMeta {
            format_version: HNSW_META_FORMAT_CURRENT,
            doc_ids: self.doc_ids.clone(),
            config: self.config,
            dimension: self.dimension,
            vector_fingerprint: self.vector_fingerprint,
            sidecar_generation: Some(generation.to_owned()),
            sidecar_basename: Some(basename.to_owned()),
            source_identity: self.source_identity.clone(),
        }
    }

    /// Run ANN query and return hits plus query diagnostics.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` if query dimension differs,
    /// and a subsystem error when a malformed/underfilled native graph cannot
    /// be repaired because no authoritative source was supplied.
    pub fn knn_search_with_stats(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> SearchResult<(Vec<VectorHit>, AnnSearchStats)> {
        self.knn_search_with_optional_source(
            None,
            query,
            k,
            ef_search,
            HnswHitResolution::CanonicalPublic,
        )
    }

    /// Run ANN query with an authoritative `VectorIndex` available for exact
    /// underfill repair.
    ///
    /// The source is borrowed only for this query. It verifies the physical
    /// main-slab extent and returned row identities, filters post-build
    /// tombstones, and supplies an exact scan when the native graph underfills.
    /// This avoids retaining a second mmap or depending on a path that may be
    /// renamed/replaced after the caller opened the canonical index.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` for incompatible vectors and
    /// rejects source-extent or returned-row identity divergence. The caller
    /// must supply the same immutable main slab used to build/load this graph;
    /// a full vector fingerprint is intentionally recomputed only before exact
    /// repair, avoiding an O(n) check on every successful ANN query.
    pub fn knn_search_with_stats_against(
        &self,
        source: &VectorIndex,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> SearchResult<(Vec<VectorHit>, AnnSearchStats)> {
        self.knn_search_with_optional_source(
            Some(source),
            query,
            k,
            ef_search,
            HnswHitResolution::CanonicalPublic,
        )
    }

    /// Return raw physical main-slab candidates for `TwoTierIndex`.
    ///
    /// Document-ID deduplication and WAL supersession are deliberately deferred
    /// until `TwoTierIndex` has ranked the main and resident-WAL physical
    /// candidates together. Exact underfill repair uses the same raw contract.
    pub(crate) fn knn_search_raw_with_stats_against(
        &self,
        source: &VectorIndex,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> SearchResult<(Vec<VectorHit>, AnnSearchStats)> {
        self.knn_search_with_optional_source(
            Some(source),
            query,
            k,
            ef_search,
            HnswHitResolution::RawPhysical,
        )
    }

    fn knn_search_with_optional_source(
        &self,
        source: Option<&VectorIndex>,
        query: &[f32],
        k: usize,
        ef_search: usize,
        resolution: HnswHitResolution,
    ) -> SearchResult<(Vec<VectorHit>, AnnSearchStats)> {
        if let Some(source) = source
            && source.dimension() != self.dimension
        {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension,
                found: source.dimension(),
            });
        }
        if let Some(source) = source
            && source.record_count() != self.source_record_count
        {
            return Err(ann_corrupted(
                &source.path,
                format!(
                    "canonical VectorIndex has {} physical rows, but HNSW was built against {}",
                    source.record_count(),
                    self.source_record_count
                ),
            ));
        }
        if query.len() != self.dimension {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension,
                found: query.len(),
            });
        }

        if query.iter().any(|value| !value.is_finite()) {
            return Err(SearchError::InvalidConfig {
                field: "query".to_owned(),
                value: "non-finite".to_owned(),
                reason: "all query vector values must be finite".to_owned(),
            });
        }

        if k == 0 || self.doc_ids.is_empty() {
            // k = 0 and empty-graph are distinct zero-signal states and must
            // not collapse into one indistinguishable empty result (bd-tqhc).
            // k = 0 takes precedence: it is request-scoped.
            let zero_signal = if k == 0 {
                ZeroSignalReason::CallerRequestedZeroK
            } else {
                ZeroSignalReason::NewlyCreatedEmpty
            };
            let stats = AnnSearchStats {
                index_size: self.len(),
                dimension: self.dimension,
                ef_search,
                k_requested: k,
                k_returned: 0,
                search_time_us: 0,
                is_approximate: true,
                fallback_reason: None,
                estimated_recall: 1.0,
                zero_signal: Some(zero_signal),
            };
            return Ok((Vec::new(), stats));
        }

        let effective_k = k.min(self.doc_ids.len());
        let effective_ef = ef_search.max(effective_k).max(1);
        let budget = dist_dot_budget(self.dimension)?;
        let normalized_query = normalize_for_dist_dot(query.to_vec(), budget);

        let start = Instant::now();
        let neighbors = self
            .hnsw
            .search(&normalized_query, effective_k, effective_ef);
        self.finish_search_with_neighbors_resolution(
            source,
            query,
            k,
            effective_k,
            effective_ef,
            neighbors,
            budget,
            start,
            resolution,
        )
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn finish_search_with_neighbors(
        &self,
        source: Option<&VectorIndex>,
        canonical_query: &[f32],
        k_requested: usize,
        effective_k: usize,
        effective_ef: usize,
        neighbors: Vec<Neighbour>,
        budget: DistDotBudget,
        start: Instant,
    ) -> SearchResult<(Vec<VectorHit>, AnnSearchStats)> {
        self.finish_search_with_neighbors_resolution(
            source,
            canonical_query,
            k_requested,
            effective_k,
            effective_ef,
            neighbors,
            budget,
            start,
            HnswHitResolution::CanonicalPublic,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_search_with_neighbors_resolution(
        &self,
        source: Option<&VectorIndex>,
        canonical_query: &[f32],
        k_requested: usize,
        effective_k: usize,
        effective_ef: usize,
        neighbors: Vec<Neighbour>,
        budget: DistDotBudget,
        start: Instant,
        resolution: HnswHitResolution,
    ) -> SearchResult<(Vec<VectorHit>, AnnSearchStats)> {
        let (hits, fallback_reason) = self.resolve_neighbors_with_fallback(
            source,
            canonical_query,
            effective_k,
            neighbors,
            budget,
            resolution,
        )?;
        let search_time_us = u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX);

        // An empty result here means the exact repair also found nothing, so
        // classify from the authoritative source census. When the census says
        // usable vectors exist yet even exact repair produced nothing, that is
        // the ANN-availability anomaly rather than a benign state.
        let zero_signal = if hits.is_empty() {
            source.map(|source_index| {
                let state = source_index.zero_signal_state();
                state.state_reason().unwrap_or_else(|| {
                    if state.is_wal_only() {
                        ZeroSignalReason::WalOnlyNoLiveRecords
                    } else {
                        ZeroSignalReason::AnnReturnedEmptyDespiteUsableVectors
                    }
                })
            })
        } else {
            None
        };

        let stats = AnnSearchStats {
            index_size: self.len(),
            dimension: self.dimension,
            ef_search: effective_ef,
            k_requested,
            k_returned: hits.len(),
            search_time_us,
            is_approximate: fallback_reason.is_none(),
            fallback_reason,
            estimated_recall: if fallback_reason.is_some() {
                1.0
            } else {
                estimate_recall(effective_ef, effective_k)
            },
            zero_signal,
        };
        Ok((hits, stats))
    }

    fn resolve_neighbors_with_fallback(
        &self,
        source: Option<&VectorIndex>,
        canonical_query: &[f32],
        effective_k: usize,
        neighbors: Vec<Neighbour>,
        budget: DistDotBudget,
        resolution: HnswHitResolution,
    ) -> SearchResult<(Vec<VectorHit>, Option<AnnFallbackReason>)> {
        let native_returned = neighbors.len();
        let mut seen_neighbors = HashSet::with_capacity(native_returned);
        let mut eligible_neighbors = Vec::with_capacity(native_returned);
        for neighbor in neighbors {
            if !seen_neighbors.insert(neighbor.d_id) {
                continue;
            }
            if let Some(source) = source {
                let physical_position = self
                    .source_positions
                    .get(neighbor.d_id)
                    .copied()
                    .and_then(|position| usize::try_from(position).ok())
                    .ok_or_else(|| {
                        ann_corrupted(
                            &source.path,
                            format!(
                                "native neighbor {} has no compact-to-physical source row",
                                neighbor.d_id
                            ),
                        )
                    })?;
                if physical_position >= source.record_count() {
                    return Err(ann_corrupted(
                        &source.path,
                        format!(
                            "native neighbor {} maps to source row {physical_position}, \
                             beyond {} rows",
                            neighbor.d_id,
                            source.record_count()
                        ),
                    ));
                }
                let expected_doc_id = self.doc_ids.get(neighbor.d_id).ok_or_else(|| {
                    ann_corrupted(
                        &source.path,
                        format!("native neighbor {} has no document identity", neighbor.d_id),
                    )
                })?;
                if source.doc_id_at(physical_position)? != expected_doc_id {
                    return Err(ann_corrupted(
                        &source.path,
                        format!(
                            "native neighbor {} maps to source row {physical_position} with a \
                             different document identity",
                            neighbor.d_id
                        ),
                    ));
                }
                if source.is_deleted(physical_position) {
                    continue;
                }
            }
            eligible_neighbors.push(neighbor);
        }
        if eligible_neighbors.len() < effective_k {
            // First underfill on this graph instance warns; repeats log at
            // debug. Instances are per-generation, giving the required
            // once-per-generation warning bound (bd-tqhc).
            if self.underfill_warned.swap(true, Ordering::Relaxed) {
                tracing::debug!(
                    index_size = self.len(),
                    effective_k,
                    native_returned,
                    native_unique_live = eligible_neighbors.len(),
                    "HNSW search underfilled; returning exact top-k"
                );
            } else {
                tracing::warn!(
                    index_size = self.len(),
                    effective_k,
                    native_returned,
                    native_unique_live = eligible_neighbors.len(),
                    "HNSW search underfilled; returning exact top-k \
                     (repeat underfills for this graph instance log at debug)"
                );
            }
            let source = source.ok_or_else(|| SearchError::SubsystemError {
                subsystem: "hnsw",
                source: Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "HNSW search underfilled; call knn_search_with_stats_against with the \
                     authoritative VectorIndex to permit canonical exact repair",
                )),
            })?;
            let hits = self.exact_scan(source, canonical_query, effective_k, resolution)?;
            // A graph that yields zero candidates despite indexing points is a
            // stronger anomaly than a partial underfill; record it distinctly.
            let reason = if native_returned == 0 && !self.doc_ids.is_empty() {
                AnnFallbackReason::EmptyDespiteIndexedPoints
            } else {
                AnnFallbackReason::Underfilled
            };
            return Ok((hits, Some(reason)));
        }

        self.hits_from_distances(
            eligible_neighbors
                .into_iter()
                .map(|neighbor| (neighbor.d_id, neighbor.distance)),
            budget,
            resolution,
        )
        .map(|hits| (hits, None))
    }

    fn exact_scan(
        &self,
        source: &VectorIndex,
        canonical_query: &[f32],
        effective_k: usize,
        resolution: HnswHitResolution,
    ) -> SearchResult<Vec<VectorHit>> {
        if source.dimension() != self.dimension {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension,
                found: source.dimension(),
            });
        }
        let source_fingerprint = fingerprint_vector_index_positions(
            source,
            &self.source_positions,
            &self.doc_ids,
            self.dimension,
        )?;
        if source_fingerprint != self.vector_fingerprint {
            return Err(ann_corrupted(
                &source.path,
                "canonical VectorIndex source rows changed after HNSW construction",
            ));
        }
        match resolution {
            HnswHitResolution::CanonicalPublic => {
                source.search_main_top_k(canonical_query, effective_k)
            }
            HnswHitResolution::RawPhysical => {
                source.search_main_top_k_raw(canonical_query, effective_k)
            }
        }
    }

    fn hits_from_distances(
        &self,
        distances: impl IntoIterator<Item = (usize, f32)>,
        budget: DistDotBudget,
        resolution: HnswHitResolution,
    ) -> SearchResult<Vec<VectorHit>> {
        let scores = distances
            .into_iter()
            .map(|(neighbor_id, distance)| {
                restore_dist_dot_score(distance, budget).map(|score| (neighbor_id, score))
            })
            .collect::<SearchResult<Vec<_>>>()?;
        self.hits_from_scores(scores, resolution)
    }

    fn hits_from_scores(
        &self,
        scores: impl IntoIterator<Item = (usize, f32)>,
        resolution: HnswHitResolution,
    ) -> SearchResult<Vec<VectorHit>> {
        let mut hits = Vec::new();
        for (neighbor_id, score) in scores {
            let doc_id =
                self.doc_ids
                    .get(neighbor_id)
                    .ok_or_else(|| SearchError::InvalidConfig {
                        field: "neighbor_id".to_owned(),
                        value: neighbor_id.to_string(),
                        reason: "neighbor id exceeds doc_id table".to_owned(),
                    })?;
            let index = self
                .source_positions
                .get(neighbor_id)
                .copied()
                .ok_or_else(|| SearchError::InvalidConfig {
                    field: "neighbor_id".to_owned(),
                    value: neighbor_id.to_string(),
                    reason: "neighbor id exceeds compact-to-physical row map".to_owned(),
                })?;
            hits.push(VectorHit {
                index,
                // Graph and query vectors use the same deterministic radius.
                // Undo that uniform scale so callers continue receiving cosine
                // similarity rather than a dimension-dependent proxy score.
                // Clamp only after proving the deviation is inside the derived
                // floating-point envelope; materially invalid distances fail.
                score,
                doc_id: doc_id.as_str().into(),
            });
        }
        hits.sort_by(|left, right| {
            left.cmp_by_score(right)
                .then_with(|| left.index.cmp(&right.index))
        });
        if resolution == HnswHitResolution::CanonicalPublic {
            let mut seen_doc_ids = HashSet::with_capacity(hits.len());
            hits.retain(|hit| seen_doc_ids.insert(hit.doc_id.clone()));
        }
        Ok(hits)
    }

    /// Run ANN query and return only the hits.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` if query dimension differs,
    /// and a subsystem error when a malformed/underfilled native graph cannot
    /// be repaired because no authoritative source was supplied.
    pub fn knn_search(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> SearchResult<Vec<VectorHit>> {
        self.knn_search_with_stats(query, k, ef_search)
            .map(|(hits, _)| hits)
    }

    /// Run ANN query with an authoritative source available for exact
    /// underfill repair, returning only hits.
    ///
    /// # Errors
    ///
    /// Propagates dimension, source-identity, native-search, or exact-repair
    /// errors from [`Self::knn_search_with_stats_against`].
    pub fn knn_search_against(
        &self,
        source: &VectorIndex,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> SearchResult<Vec<VectorHit>> {
        self.knn_search_with_stats_against(source, query, k, ef_search)
            .map(|(hits, _)| hits)
    }

    /// Certify the cheapest `ef_search` whose recall meets `target` — the automated
    /// replacement for the human "recall-budget sign-off" that gated ANN-in-BOLD.
    ///
    /// Measures this ANN index's recall@`k` against exact bruteforce
    /// ([`VectorIndex::search_top_k`]) over `calibration_queries`, and returns the
    /// smallest `ef` in `candidate_efs` whose split-conformal recall **lower bound**
    /// is `≥ target` at confidence `1 − alpha` (distribution-free, finite-sample
    /// valid; see [`crate::recall_certificate`]). If none qualifies, returns the
    /// best-certifiable `ef` with `meets_target = false`.
    ///
    /// The exact top-k for each query is independent of `ef`, so it is computed
    /// **once per query** (not per `ef`); only the ANN search re-runs per candidate,
    /// and the sweep short-circuits at the first certified `ef`, so no ANN search is
    /// run at an `ef` larger than the chosen one.
    ///
    /// # Errors
    ///
    /// Propagates any error from the exact [`VectorIndex::search_top_k`] pass. A
    /// failed ANN search or exact-underfill fallback for a single (query, ef)
    /// is treated as recall `0.0` for that query. This conservative direction
    /// prevents calibration from selecting an `ef` whose apparent recall came
    /// from repeatedly abandoning ANN for a full scan.
    pub fn certify_ef_search(
        &self,
        exact_index: &VectorIndex,
        calibration_queries: &[Vec<f32>],
        candidate_efs: &[usize],
        k: usize,
        target: f64,
        alpha: f64,
    ) -> SearchResult<Option<EfCalibration>> {
        // Exact top-k is ef-independent: compute it once per calibration query.
        let exact: Vec<Vec<VectorHit>> = calibration_queries
            .iter()
            .map(|q| exact_index.search_top_k(q, k, None))
            .collect::<SearchResult<_>>()?;

        Ok(calibrate_certified_ef(
            candidate_efs,
            |ef| {
                calibration_queries
                    .iter()
                    .zip(&exact)
                    .map(|(q, exact_hits)| {
                        certified_ann_recall_sample(
                            self.knn_search_with_stats_against(exact_index, q, k, ef),
                            exact_hits,
                        )
                    })
                    .collect()
            },
            target,
            alpha,
        ))
    }

    /// Returns true when this ANN index matches row order and shape of a `VectorIndex`.
    ///
    /// Since `HnswIndex` no longer stores vectors, this checks:
    /// 1. Dimension match.
    /// 2. Live physical-row and `doc_id` sequence match.
    /// 3. The fingerprint of every live vector.
    ///
    /// # Errors
    ///
    /// Propagates decoding errors from `VectorIndex::doc_id_at`.
    pub fn matches_vector_index(&self, index: &VectorIndex) -> SearchResult<bool> {
        if self.dimension != index.dimension() {
            return Ok(false);
        }
        // Same generation-identity gate as the native load path (bd-r65a):
        // matching dimension, doc ids, ordering, and vector fingerprint prove
        // the CONTENT is the same, never that the artifact is.
        if !HnswSourceIdentityV1::admits(
            self.source_identity.as_ref(),
            HnswSourceIdentityV1::capture(index).as_ref(),
        ) {
            return Ok(false);
        }
        let mut live_position = 0_usize;
        for i in 0..index.record_count() {
            if index.is_deleted(i) {
                continue;
            }
            let Some(expected_doc_id) = self.doc_ids.get(live_position) else {
                // HNSW has fewer docs than VectorIndex
                return Ok(false);
            };
            if expected_doc_id != index.doc_id_at(i)? {
                return Ok(false);
            }
            if self.source_positions.get(live_position).copied() != u32::try_from(i).ok() {
                return Ok(false);
            }
            live_position = live_position.saturating_add(1);
        }
        if live_position != self.doc_ids.len() {
            return Ok(false);
        }
        Ok(
            fingerprint_live_vector_index(index, self.doc_ids.len(), self.dimension)?
                == self.vector_fingerprint,
        )
    }

    /// Number of indexed vectors.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.doc_ids.len()
    }

    /// Whether ANN index has zero vectors.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.doc_ids.is_empty()
    }

    /// Vector dimensionality.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// ANN configuration used to build this index.
    #[must_use]
    pub const fn config(&self) -> HnswConfig {
        self.config
    }

    fn build_from_parts(
        doc_ids: Vec<String>,
        vectors: Vec<Vec<f32>>,
        dimension: usize,
        config: HnswConfig,
    ) -> SearchResult<Self> {
        validate_config(config)?;
        if dimension == 0 {
            return Err(SearchError::InvalidConfig {
                field: "dimension".to_owned(),
                value: "0".to_owned(),
                reason: "dimension must be greater than zero".to_owned(),
            });
        }
        let budget = dist_dot_budget(dimension)?;
        if doc_ids.len() != vectors.len() {
            return Err(SearchError::InvalidConfig {
                field: "vectors".to_owned(),
                value: vectors.len().to_string(),
                reason: format!("doc_id count {} must match vector count", doc_ids.len()),
            });
        }
        // Fingerprint the raw (un-normalized) input vectors. This is what
        // VectorIndex::vector_at_f32 returns at load time, so a fresh load from
        // the live VectorIndex will produce the same digest iff the underlying
        // bytes are unchanged. Used by the native-graph load path to detect
        // "doc IDs match but vectors were silently swapped" and trigger a
        // rebuild (see try_load_native_graph).
        let vector_fingerprint = fingerprint_vectors(&doc_ids, &vectors);

        let mut normalized_vectors = Vec::with_capacity(vectors.len());
        for (idx, vector) in vectors.into_iter().enumerate() {
            if vector.len() != dimension {
                return Err(SearchError::DimensionMismatch {
                    expected: dimension,
                    found: vector.len(),
                });
            }
            if vector.iter().any(|value| !value.is_finite()) {
                return Err(SearchError::InvalidConfig {
                    field: "vector".to_owned(),
                    value: idx.to_string(),
                    reason: "all vector values must be finite".to_owned(),
                });
            }
            normalized_vectors.push(normalize_for_dist_dot(vector, budget));
        }

        let use_parallel_insert = should_use_parallel_insert(normalized_vectors.len());
        let initial_hnsw = construct_hnsw_graph(&normalized_vectors, config, use_parallel_insert);
        let hnsw = attest_or_rebuild_serial(
            initial_hnsw,
            use_parallel_insert,
            doc_ids.len(),
            |graph| validate_hnsw_topology(graph, doc_ids.len()),
            || construct_hnsw_graph(&normalized_vectors, config, false),
        )
        .map_err(|detail| ann_topology_error(&detail))?;

        Ok(Self {
            hnsw,
            underfill_warned: AtomicBool::new(false),
            doc_ids,
            source_positions: (0..normalized_vectors.len())
                .map(|position| {
                    u32::try_from(position).map_err(|_| SearchError::InvalidConfig {
                        field: "source_position".to_owned(),
                        value: position.to_string(),
                        reason: "HNSW row exceeds u32".to_owned(),
                    })
                })
                .collect::<SearchResult<Vec<_>>>()?,
            source_record_count: normalized_vectors.len(),
            dimension,
            config,
            vector_fingerprint,
            source_identity: None,
        })
    }
}

fn should_use_parallel_insert(point_count: usize) -> bool {
    let remaining_points = point_count.saturating_sub(1);
    let minimum_parallel_batch =
        rayon::current_num_threads().saturating_mul(HNSW_PARALLEL_INSERT_MIN_POINTS_PER_THREAD);
    remaining_points >= minimum_parallel_batch
}

fn construct_hnsw_graph(
    vectors: &[Vec<f32>],
    config: HnswConfig,
    parallel: bool,
) -> Hnsw<'static, f32, DistDot> {
    let hnsw = Hnsw::new(
        config.m,
        vectors.len().max(1),
        config.max_layer,
        config.ef_construction,
        DistDot,
    );
    let Some((first, remaining)) = vectors.split_first() else {
        return hnsw;
    };

    // Seed the entry point serially. The upstream implementation admits a new
    // point into shared tables before reverse-edge updates are complete, so an
    // established entry point is required before any concurrent insertion.
    hnsw.insert((first, 0));
    if parallel {
        let vectors_with_ids = remaining
            .iter()
            .enumerate()
            .map(|(index, vector)| (vector, index.saturating_add(1)))
            .collect::<Vec<_>>();
        hnsw.parallel_insert(&vectors_with_ids);
    } else {
        for (index, vector) in remaining.iter().enumerate() {
            hnsw.insert((vector, index.saturating_add(1)));
        }
    }
    hnsw
}

fn attest_or_rebuild_serial<T>(
    initial_graph: T,
    attempted_parallel: bool,
    point_count: usize,
    mut validate: impl FnMut(&T) -> Result<(), String>,
    rebuild_serial: impl FnOnce() -> T,
) -> Result<T, String> {
    let Err(initial_detail) = validate(&initial_graph) else {
        return Ok(initial_graph);
    };
    if !attempted_parallel {
        return Err(initial_detail);
    }

    // Never publish a graph whose concurrent construction violated the
    // searchable-topology contract. Drop it before allocating the replacement:
    // parallel mode is reserved for large indexes, where retaining both full
    // graphs could turn a recoverable topology fault into an OOM.
    tracing::warn!(
        point_count,
        parallel_error = %initial_detail,
        "parallel HNSW construction failed topology attestation; rebuilding serially"
    );
    drop(initial_graph);

    let serial_graph = rebuild_serial();
    if let Err(serial_detail) = validate(&serial_graph) {
        return Err(format!(
            "parallel construction failed ({initial_detail}); \
             serial rebuild also failed ({serial_detail})"
        ));
    }
    Ok(serial_graph)
}

fn restore_dist_dot_score(distance: f32, budget: DistDotBudget) -> SearchResult<f32> {
    let restored_score = (1.0 - distance) / budget.radius_squared;
    let score_envelope = -1.0 - budget.score_tolerance..=1.0 + budget.score_tolerance;
    if !restored_score.is_finite() || !score_envelope.contains(&restored_score) {
        return Err(SearchError::SubsystemError {
            subsystem: "hnsw",
            source: Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "DistDot returned distance {distance} (restored score {restored_score}); \
                     expected a score within [{}, {}]",
                    score_envelope.start(),
                    score_envelope.end()
                ),
            )),
        });
    }
    Ok(restored_score.clamp(-1.0, 1.0))
}

fn validate_hnsw_topology(
    hnsw: &Hnsw<'_, f32, DistDot>,
    expected_points: usize,
) -> Result<(), String> {
    let observed_count = hnsw.get_nb_point();
    if observed_count != expected_points {
        return Err(format!(
            "point count mismatch: expected {expected_points}, observed {observed_count}"
        ));
    }
    if expected_points == 0 {
        return hnsw
            .get_entry_point_id()
            .is_none()
            .then_some(())
            .ok_or_else(|| "empty graph unexpectedly exposes an entry point".to_owned());
    }

    // Retain one Arc per point, indexed below by origin id. Neighborhoods are
    // still cloned only one point at a time, so the connectivity and coverage
    // walks below never materialize a second O(edges) adjacency graph.
    let points = hnsw.get_point_indexation().into_iter().collect::<Vec<_>>();
    let identities = points
        .iter()
        .map(|point| (point.get_origin_id(), point.get_point_id()))
        .collect::<Vec<_>>();
    let point_by_internal_id = validate_hnsw_identity_table(&identities, expected_points)?;
    let entry_origin = validate_hnsw_entry_point(
        hnsw.get_entry_point_id(),
        &identities,
        &point_by_internal_id,
    )?;
    let mut points_by_origin = std::iter::repeat_with(|| None)
        .take(expected_points)
        .collect::<Vec<_>>();
    for point in points {
        let origin_id = point.get_origin_id();
        points_by_origin[origin_id] = Some(point);
    }
    for point in points_by_origin.iter().flatten() {
        let neighborhoods = point.get_neighborhood_id();
        validate_hnsw_point_neighborhoods(
            point.get_origin_id(),
            point.get_point_id(),
            &neighborhoods,
            expected_points,
            &point_by_internal_id,
        )?;
    }
    let mut base_neighbors = |origin_id: usize| -> Result<Vec<usize>, String> {
        let point = points_by_origin[origin_id]
            .as_ref()
            .ok_or_else(|| format!("missing origin id {origin_id} during connectivity walk"))?;
        Ok(point
            .get_neighborhood_id()
            .first()
            .into_iter()
            .flatten()
            .map(|neighbor| neighbor.d_id)
            .collect())
    };
    validate_weak_base_connectivity(entry_origin, expected_points, &mut base_neighbors)?;
    report_directed_base_coverage(entry_origin, expected_points, &mut base_neighbors)
}

#[cfg(test)]
fn validate_hnsw_topology_observations(
    points: &[HnswTopologyPoint],
    expected_points: usize,
    entry_point: Option<(usize, PointId)>,
) -> Result<(), String> {
    if points.len() != expected_points {
        return Err(format!(
            "point iterator yielded {} entries for {expected_points} expected points",
            points.len()
        ));
    }

    let identities = points
        .iter()
        .map(|point| (point.origin_id, point.point_id))
        .collect::<Vec<_>>();
    let point_by_internal_id = validate_hnsw_identity_table(&identities, expected_points)?;
    if expected_points == 0 {
        return entry_point
            .is_none()
            .then_some(())
            .ok_or_else(|| "empty graph unexpectedly exposes an entry point".to_owned());
    }
    let entry_origin = validate_hnsw_entry_point(entry_point, &identities, &point_by_internal_id)?;
    let mut points_by_origin = vec![None; expected_points];
    for point in points {
        points_by_origin[point.origin_id] = Some(point);
        validate_hnsw_point_neighborhoods(
            point.origin_id,
            point.point_id,
            &point.neighborhoods,
            expected_points,
            &point_by_internal_id,
        )?;
    }
    let mut base_neighbors = |origin_id: usize| -> Result<Vec<usize>, String> {
        let point = points_by_origin[origin_id]
            .ok_or_else(|| format!("missing origin id {origin_id} during connectivity walk"))?;
        Ok(point
            .neighborhoods
            .first()
            .into_iter()
            .flatten()
            .map(|neighbor| neighbor.d_id)
            .collect())
    };
    validate_weak_base_connectivity(entry_origin, expected_points, &mut base_neighbors)?;
    report_directed_base_coverage(entry_origin, expected_points, &mut base_neighbors)
}

fn validate_hnsw_entry_point(
    entry_point: Option<(usize, PointId)>,
    identities: &[(usize, PointId)],
    point_by_internal_id: &HashMap<PointId, usize>,
) -> Result<usize, String> {
    let (entry_origin, entry_point_id) =
        entry_point.ok_or_else(|| "non-empty graph has no search entry point".to_owned())?;
    let mapped_origin = point_by_internal_id
        .get(&entry_point_id)
        .copied()
        .ok_or_else(|| format!("entry point {entry_point_id:?} is absent from the point table"))?;
    if mapped_origin != entry_origin {
        return Err(format!(
            "entry point {entry_point_id:?} maps to origin {mapped_origin}, \
             not advertised origin {entry_origin}"
        ));
    }
    let max_layer = identities
        .iter()
        .map(|(_, point_id)| point_id.0)
        .max()
        .ok_or_else(|| "non-empty graph has no point identities".to_owned())?;
    if entry_point_id.0 != max_layer {
        return Err(format!(
            "entry point {entry_point_id:?} is below maximum sampled layer {max_layer}"
        ));
    }
    Ok(entry_origin)
}

/// Attest that the base layer forms a single weak component containing the
/// search entry.
///
/// Universal *directed* reachability from the entry point is deliberately NOT
/// required here (#32). `hnsw_rs` mirrors every forward edge with a reverse
/// edge, but a neighbour list that exceeds its cap (`M`, or `2M` at the base
/// layer) is sorted by distance and its farthest edge popped, so later
/// insertions can legitimately prune away *all* of a point's in-edges. Such a
/// point becomes a directed "source": still weakly attached through its own
/// out-edges, but invisible to a directed walk from the entry. That outcome
/// is intrinsic to bounded-degree HNSW construction — serial builds exhibit
/// it as readily as parallel ones — so treating it as an integrity failure
/// rejects every sufficiently large or duplicate-heavy corpus.
///
/// A genuinely broken build (edgeless points, disjoint islands from misfiled
/// reverse edges) still fails this gate: those graphs split into multiple
/// weak components. Partial *directed* coverage is measured separately by
/// [`report_directed_base_coverage`] and surfaced as a warning, and observed
/// ANN quality has its own certificate in [`HnswIndex::certify_ef_search`].
fn validate_weak_base_connectivity(
    entry_origin: usize,
    expected_points: usize,
    mut base_neighbors: impl FnMut(usize) -> Result<Vec<usize>, String>,
) -> Result<(), String> {
    let mut components = WeakComponents::new(expected_points);
    let mut isolated_count = 0_usize;
    let mut first_isolated = None;
    for origin_id in 0..expected_points {
        let neighbors = base_neighbors(origin_id)?;
        if neighbors.is_empty() {
            // A point with zero base out-edges violates the HNSW insertion
            // invariant (every non-seed insert attaches to an existing point),
            // so record it: it separates construction faults from
            // pruning-induced splits when the gate fires.
            isolated_count += 1;
            if first_isolated.is_none() {
                first_isolated = Some(origin_id);
            }
        }
        for neighbor_id in neighbors {
            if neighbor_id >= expected_points {
                return Err(format!(
                    "origin id {origin_id} references out-of-range base neighbor {neighbor_id}"
                ));
            }
            components.merge(origin_id, neighbor_id);
        }
    }

    let entry_component = components.root(entry_origin);
    let mut attached_count = 0_usize;
    let mut first_detached = None;
    let mut component_count = 0_usize;
    for origin_id in 0..expected_points {
        if components.root(origin_id) == origin_id {
            component_count += 1;
        }
        if components.root(origin_id) == entry_component {
            attached_count += 1;
        } else if first_detached.is_none() {
            first_detached = Some(origin_id);
        }
    }
    if let Some(first_detached) = first_detached {
        let first_isolated =
            first_isolated.map_or_else(|| "none".to_owned(), |origin| format!("origin {origin}"));
        return Err(format!(
            "base layer splits into multiple weak components: entry origin {entry_origin}'s \
             component holds only {attached_count}/{expected_points} points ignoring edge \
             direction; first detached origin is {first_detached}; {component_count} weak \
             components total; {isolated_count} points have zero base out-edges (first: \
             {first_isolated})"
        ));
    }
    Ok(())
}

/// Disjoint-set forest over base-layer origin ids, used to attest weak
/// connectivity in O(points) memory without materializing an adjacency graph.
struct WeakComponents {
    parent: Vec<usize>,
    size: Vec<u32>,
}

impl WeakComponents {
    fn new(count: usize) -> Self {
        Self {
            parent: (0..count).collect(),
            size: vec![1; count],
        }
    }

    /// Iterative find with path halving; no recursion regardless of scale.
    fn root(&mut self, mut origin_id: usize) -> usize {
        while self.parent[origin_id] != origin_id {
            self.parent[origin_id] = self.parent[self.parent[origin_id]];
            origin_id = self.parent[origin_id];
        }
        origin_id
    }

    fn merge(&mut self, left_origin: usize, right_origin: usize) {
        let mut small = self.root(left_origin);
        let mut large = self.root(right_origin);
        if small == large {
            return;
        }
        if self.size[small] > self.size[large] {
            std::mem::swap(&mut small, &mut large);
        }
        self.parent[small] = large;
        self.size[large] = self.size[large].saturating_add(self.size[small]);
    }
}

/// Measure how much of the base layer a *directed* walk from the search entry
/// covers, and warn loudly when coverage is partial.
///
/// Partial coverage is admissible (see [`validate_weak_base_connectivity`])
/// but never silent: the points outside the covered set cannot be visited by
/// greedy base-layer expansion, so operators sizing a corpus should see the
/// exact count and follow up with [`HnswIndex::certify_ef_search`] to bound
/// the observed quality impact.
fn report_directed_base_coverage(
    entry_origin: usize,
    expected_points: usize,
    mut base_neighbors: impl FnMut(usize) -> Result<Vec<usize>, String>,
) -> Result<(), String> {
    let coverage =
        measure_directed_base_coverage(entry_origin, expected_points, &mut base_neighbors)?;
    if coverage.reached_count != expected_points {
        tracing::warn!(
            entry_origin,
            reached = coverage.reached_count,
            expected = expected_points,
            first_unreachable = ?coverage.first_unreachable,
            "HNSW base layer is weakly connected but a directed walk from the search entry \
             does not cover it; bounded neighbour pruning orphans in-edges at scale, so this \
             is not an integrity failure — verify quality via certify_ef_search"
        );
    }
    Ok(())
}

/// Outcome of the directed base-layer walk from the search entry.
struct DirectedBaseCoverage {
    reached_count: usize,
    first_unreachable: Option<usize>,
}

fn measure_directed_base_coverage(
    entry_origin: usize,
    expected_points: usize,
    mut base_neighbors: impl FnMut(usize) -> Result<Vec<usize>, String>,
) -> Result<DirectedBaseCoverage, String> {
    let mut reached = vec![false; expected_points];
    let mut queue = VecDeque::with_capacity(expected_points.min(4_096));
    reached[entry_origin] = true;
    queue.push_back(entry_origin);
    let mut reached_count = 1_usize;

    while let Some(origin_id) = queue.pop_front() {
        for neighbor_id in base_neighbors(origin_id)? {
            if neighbor_id >= expected_points {
                return Err(format!(
                    "origin id {origin_id} reaches out-of-range base neighbor {neighbor_id}"
                ));
            }
            if !reached[neighbor_id] {
                reached[neighbor_id] = true;
                reached_count += 1;
                queue.push_back(neighbor_id);
            }
        }
    }

    let first_unreachable = reached.iter().position(|is_reached| !is_reached);
    Ok(DirectedBaseCoverage {
        reached_count,
        first_unreachable,
    })
}

fn validate_hnsw_identity_table(
    identities: &[(usize, PointId)],
    expected_points: usize,
) -> Result<HashMap<PointId, usize>, String> {
    if identities.len() != expected_points {
        return Err(format!(
            "point iterator yielded {} entries for {expected_points} expected points",
            identities.len()
        ));
    }

    let mut point_by_internal_id = HashMap::with_capacity(identities.len());
    let mut origin_ids = HashSet::with_capacity(identities.len());
    for &(origin_id, point_id) in identities {
        if origin_id >= expected_points {
            return Err(format!(
                "origin id {} is outside 0..{expected_points}",
                origin_id
            ));
        }
        if !origin_ids.insert(origin_id) {
            return Err(format!("duplicate origin id {origin_id}"));
        }
        if point_id.1 < 0 {
            return Err(format!(
                "origin id {} has negative internal slot {:?}",
                origin_id, point_id
            ));
        }
        if point_by_internal_id.insert(point_id, origin_id).is_some() {
            return Err(format!("duplicate internal point id {point_id:?}"));
        }
    }

    for expected_origin_id in 0..expected_points {
        if !origin_ids.contains(&expected_origin_id) {
            return Err(format!("missing origin id {expected_origin_id}"));
        }
    }
    Ok(point_by_internal_id)
}

fn validate_hnsw_point_neighborhoods(
    origin_id: usize,
    point_id: PointId,
    neighborhoods: &[Vec<Neighbour>],
    expected_points: usize,
    point_by_internal_id: &HashMap<PointId, usize>,
) -> Result<(), String> {
    let source_max_layer = usize::from(point_id.0);
    if source_max_layer >= neighborhoods.len() {
        return Err(format!(
            "origin id {origin_id} sampled layer {} but exposes only {} neighborhoods",
            point_id.0,
            neighborhoods.len()
        ));
    }
    for (layer, neighbors) in neighborhoods.iter().enumerate() {
        if layer > source_max_layer && !neighbors.is_empty() {
            return Err(format!(
                "origin id {origin_id} has {} neighbors in layer {layer} above sampled layer \
                 {source_max_layer}",
                neighbors.len()
            ));
        }

        let mut seen_neighbors = HashSet::with_capacity(neighbors.len());
        for neighbor in neighbors {
            if !neighbor.distance.is_finite() {
                return Err(format!(
                    "origin id {origin_id} has a non-finite distance in layer {layer}"
                ));
            }
            if neighbor.d_id >= expected_points {
                return Err(format!(
                    "origin id {origin_id} references out-of-range neighbor {} in layer {layer}",
                    neighbor.d_id
                ));
            }
            if !seen_neighbors.insert(neighbor.d_id) {
                return Err(format!(
                    "origin id {origin_id} repeats neighbor {} in layer {layer}",
                    neighbor.d_id
                ));
            }
            if neighbor.d_id == origin_id {
                return Err(format!(
                    "origin id {origin_id} references itself in layer {layer}"
                ));
            }
            let Some(&internal_origin_id) = point_by_internal_id.get(&neighbor.p_id) else {
                return Err(format!(
                    "origin id {origin_id} references missing internal point {:?} in layer {layer}",
                    neighbor.p_id
                ));
            };
            if internal_origin_id != neighbor.d_id {
                return Err(format!(
                    "neighbor {:?} maps to origin {internal_origin_id}, not advertised origin {}",
                    neighbor.p_id, neighbor.d_id
                ));
            }
            if usize::from(neighbor.p_id.0) < layer {
                return Err(format!(
                    "origin id {origin_id} references neighbor {} at layer {layer}, above the \
                     neighbor's sampled layer {}",
                    neighbor.d_id, neighbor.p_id.0
                ));
            }
        }
    }
    Ok(())
}

fn ann_topology_error(detail: &str) -> SearchError {
    SearchError::SubsystemError {
        subsystem: "hnsw",
        source: Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("graph topology attestation failed: {detail}"),
        )),
    }
}

/// Derive the `hnsw_rs` sidecar basename from the metadata path.
///
/// `hnsw_rs` writes `{basename}.hnsw.graph` and `{basename}.hnsw.data`, so for
/// a metadata path of `dir/vector.fast.hnsw` we use the stem `vector.fast`,
/// yielding `dir/vector.fast.hnsw.graph` / `.data` — distinct from the
/// metadata file itself.
fn hnsw_sidecar_basename(path: &Path) -> SearchResult<String> {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| ann_corrupted(path, "ANN sidecar path has no usable file stem"))
}

fn hnsw_metadata_file_name(path: &Path) -> SearchResult<String> {
    path.file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| ann_corrupted(path, "ANN metadata path has no usable UTF-8 file name"))
}

fn hnsw_generation_prefix(requested_basename: &str, vector_fingerprint: u64) -> String {
    format!(".{requested_basename}.generation-{vector_fingerprint:016x}-")
}

fn hnsw_save_lock_path(path: &Path) -> SearchResult<PathBuf> {
    if path.components().any(|component| {
        component
            .as_os_str()
            .to_str()
            .is_some_and(|name| name.eq_ignore_ascii_case(HNSW_SAVE_LOCK_DIRECTORY))
    }) {
        return Err(ann_corrupted(
            path,
            format!(
                "ANN metadata paths cannot be inside the reserved '{HNSW_SAVE_LOCK_DIRECTORY}' directory"
            ),
        ));
    }
    let file_name = path
        .file_name()
        .filter(|name| !name.is_empty())
        .ok_or_else(|| ann_corrupted(path, "ANN metadata path has no usable file name"))?;
    let mut lock_name = file_name.to_os_string();
    lock_name.push(".lock");
    let parent = path
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    Ok(parent.join(HNSW_SAVE_LOCK_DIRECTORY).join(lock_name))
}

pub(crate) fn hnsw_save_lock_artifact_path(path: &Path) -> SearchResult<PathBuf> {
    hnsw_save_lock_path(path)
}

pub(crate) fn materialize_hnsw_save_lock_artifact(lock_path: &Path) -> SearchResult<PathBuf> {
    let lock_directory = lock_path
        .parent()
        .ok_or_else(|| ann_corrupted(lock_path, "HNSW save lock has no parent directory"))?;
    let artifact_parent = lock_directory
        .parent()
        .ok_or_else(|| ann_corrupted(lock_path, "HNSW save lock directory has no parent"))?;
    std::fs::create_dir_all(artifact_parent)?;
    match std::fs::create_dir(lock_directory) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => {
            return Err(SearchError::Io(std::io::Error::new(
                error.kind(),
                format!(
                    "failed to create persistent HNSW save-lock directory '{}': {error}",
                    lock_directory.display()
                ),
            )));
        }
    }
    let lock_directory_metadata = std::fs::symlink_metadata(lock_directory).map_err(|error| {
        SearchError::Io(std::io::Error::new(
            error.kind(),
            format!(
                "failed to inspect persistent HNSW save-lock directory '{}': {error}",
                lock_directory.display()
            ),
        ))
    })?;
    if lock_directory_metadata.file_type().is_symlink() || !lock_directory_metadata.is_dir() {
        return Err(SearchError::Io(std::io::Error::other(format!(
            "persistent HNSW save-lock directory '{}' is not a local directory",
            lock_directory.display()
        ))));
    }
    match std::fs::symlink_metadata(lock_path) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => {
            return Err(SearchError::Io(std::io::Error::other(format!(
                "persistent HNSW save lock '{}' is not a local regular file",
                lock_path.display()
            ))));
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(SearchError::Io(error)),
    }
    let lock = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(lock_path)
        .map_err(|error| {
            SearchError::Io(std::io::Error::new(
                error.kind(),
                format!(
                    "failed to open persistent HNSW save lock '{}': {error}",
                    lock_path.display()
                ),
            ))
        })?;
    lock.sync_all().map_err(SearchError::Io)?;
    sync_hnsw_directory(lock_directory)?;
    std::fs::canonicalize(lock_path).map_err(SearchError::Io)
}

fn acquire_hnsw_save_lock(path: &Path) -> SearchResult<std::fs::File> {
    let lock_path = hnsw_save_lock_path(path)?;
    let _identity = materialize_hnsw_save_lock_artifact(&lock_path)?;
    let lock = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .map_err(|error| {
            SearchError::Io(std::io::Error::new(
                error.kind(),
                format!(
                    "failed to open persistent HNSW save lock '{}': {error}",
                    lock_path.display()
                ),
            ))
        })?;
    lock.try_lock().map_err(|error| {
        let error: std::io::Error = error.into();
        SearchError::Io(std::io::Error::new(
            error.kind(),
            format!(
                "failed to acquire HNSW save lock '{}': {error}; another writer may be saving",
                lock_path.display()
            ),
        ))
    })?;
    Ok(lock)
}

fn serialize_hnsw_metadata(meta: &HnswMeta) -> SearchResult<Vec<u8>> {
    serde_json::to_vec(meta)
        .map_err(|error| SearchError::Io(std::io::Error::other(error.to_string())))
}

fn find_reusable_hnsw_generation(
    index: &HnswIndex,
    metadata_path: &Path,
    parent: &Path,
    requested_basename: &str,
    metadata_file_name: &str,
) -> SearchResult<Option<HnswMeta>> {
    let prefix = hnsw_generation_prefix(requested_basename, index.vector_fingerprint);
    let mut candidates = Vec::new();
    for entry in std::fs::read_dir(parent).map_err(SearchError::Io)? {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                tracing::debug!(
                    path = %metadata_path.display(),
                    ?error,
                    "ignoring unreadable HNSW generation directory entry"
                );
                continue;
            }
        };
        let file_name = entry.file_name();
        if file_name
            .to_str()
            .is_some_and(|name| name.starts_with(&prefix))
        {
            candidates.push(entry.path());
        }
    }
    candidates.sort_unstable();

    for generation in candidates {
        match reusable_hnsw_generation(index, metadata_path, &generation, metadata_file_name) {
            Ok(Some(meta)) => return Ok(Some(meta)),
            Ok(None) => {}
            Err(error) => {
                tracing::debug!(
                    path = %metadata_path.display(),
                    generation = %generation.display(),
                    ?error,
                    "ignoring invalid HNSW READY generation"
                );
            }
        }
    }
    Ok(None)
}

fn reusable_hnsw_generation(
    index: &HnswIndex,
    metadata_path: &Path,
    generation_path: &Path,
    metadata_file_name: &str,
) -> SearchResult<Option<HnswMeta>> {
    let Some(validated) = validate_hnsw_generation_receipt(
        metadata_path,
        generation_path,
        metadata_file_name,
        &index.doc_ids,
        index.vector_fingerprint,
        index.dimension,
        index.config,
    )?
    else {
        return Ok(None);
    };

    // PUBLICATION-TIME IDENTITY RE-VALIDATION (bd-21zyj). Everything the
    // receipt validator just checked is CONTENT — doc-id fingerprint, vector
    // fingerprint, dimension, params. A READY generation left on disk by a
    // DIFFERENT published FSVI generation can match every one of them, because
    // two generations may hold byte-identical live vectors under identical doc
    // ids (a re-publication carries a new nonce; a model revision bump moves
    // the space fingerprint without moving a vector). Adopting it here would
    // publish metadata that names THIS graph's identity while pointing at that
    // other generation's directory — an identity claim the bytes do not
    // support. Re-validate before reuse, failing closed in both directions
    // exactly as the load path does.
    if !HnswSourceIdentityV1::admits(
        validated.source_identity.as_ref(),
        index.source_identity.as_ref(),
    ) {
        tracing::debug!(
            path = %metadata_path.display(),
            generation = %generation_path.display(),
            receipt_bound = validated.source_identity.is_some(),
            graph_bound = index.source_identity.is_some(),
            "ignoring HNSW READY generation dumped from a different source generation"
        );
        return Ok(None);
    }

    // Byte identity alone is not enough: a complete pair can still be
    // semantically unreadable after a native-format change or a faulty dump.
    // Republishing such a receipt would make every fallback rebuild select the
    // same broken generation again. Prove that the current reader can load the
    // pair before treating it as reusable.
    if !hnsw_generation_is_loadable(
        generation_path,
        &validated.basename,
        &validated.graph,
        index.doc_ids.len(),
        index.dimension,
    ) {
        tracing::debug!(
            path = %metadata_path.display(),
            generation = %generation_path.display(),
            "ignoring digest-valid but unloadable HNSW READY generation"
        );
        return Ok(None);
    }

    Ok(Some(index.metadata_for_generation(
        &validated.generation,
        &validated.basename,
    )))
}

#[allow(clippy::too_many_arguments)]
fn validate_hnsw_generation_receipt(
    metadata_path: &Path,
    generation_path: &Path,
    metadata_file_name: &str,
    doc_ids: &[String],
    vector_fingerprint: u64,
    dimension: usize,
    config: HnswConfig,
) -> SearchResult<Option<ValidatedHnswGeneration>> {
    let Some(generation_name) = generation_path.file_name().and_then(|name| name.to_str()) else {
        return Ok(None);
    };
    let generation_name = validate_hnsw_sidecar_basename(metadata_path, generation_name)?;
    let receipt_path = generation_path.join(HNSW_GENERATION_RECEIPT_FILENAME);
    let Ok(generation_metadata) = std::fs::symlink_metadata(generation_path) else {
        return Ok(None);
    };
    let Ok(receipt_metadata) = std::fs::symlink_metadata(&receipt_path) else {
        return Ok(None);
    };
    if generation_metadata.file_type().is_symlink()
        || !generation_metadata.is_dir()
        || receipt_metadata.file_type().is_symlink()
        || !receipt_metadata.is_file()
    {
        return Ok(None);
    }
    let Ok(receipt_len) = usize::try_from(receipt_metadata.len()) else {
        return Ok(None);
    };
    if receipt_len > HNSW_GENERATION_RECEIPT_MAX_BYTES {
        return Ok(None);
    }

    let receipt_file = std::fs::File::open(&receipt_path).map_err(SearchError::Io)?;
    let opened_receipt_metadata = receipt_file.metadata().map_err(SearchError::Io)?;
    if !opened_receipt_metadata.is_file() || opened_receipt_metadata.len() != receipt_metadata.len()
    {
        return Ok(None);
    }
    let receipt_read_limit = u64::try_from(HNSW_GENERATION_RECEIPT_MAX_BYTES)
        .map_err(|_| SearchError::Io(std::io::Error::other("HNSW receipt limit exceeds u64")))?
        .saturating_add(1);
    let mut receipt_bytes = Vec::with_capacity(receipt_len);
    receipt_file
        .take(receipt_read_limit)
        .read_to_end(&mut receipt_bytes)
        .map_err(SearchError::Io)?;
    if receipt_bytes.len() > HNSW_GENERATION_RECEIPT_MAX_BYTES {
        return Ok(None);
    }
    let receipt: HnswGenerationReceipt =
        serde_json::from_slice(&receipt_bytes).map_err(|error| {
            ann_corrupted(
                metadata_path,
                format!("failed to parse HNSW generation receipt: {error}"),
            )
        })?;

    if receipt.receipt_version != HNSW_GENERATION_RECEIPT_VERSION
        || receipt.metadata_file_name != metadata_file_name
        || receipt.format_version != HNSW_META_FORMAT_CURRENT
        || receipt.generation.as_str().ne(generation_name.as_str())
        || receipt.doc_count != doc_ids.len()
        || receipt.doc_ids_fingerprint != fingerprint_doc_ids(doc_ids)
        || receipt.vector_fingerprint != vector_fingerprint
        || receipt.dimension != dimension
        || receipt.config != config
    {
        return Ok(None);
    }

    let basename = validate_hnsw_sidecar_basename(metadata_path, &receipt.sidecar_basename)?;
    let graph = generation_path.join(format!("{basename}.hnsw.graph"));
    let data = generation_path.join(format!("{basename}.hnsw.data"));
    if !native_sidecar_pair_is_local(metadata_path, generation_path, &graph, &data) {
        return Ok(None);
    }
    if fingerprint_hnsw_sidecar(&graph)? != receipt.graph
        || fingerprint_hnsw_sidecar(&data)? != receipt.data
    {
        return Ok(None);
    }

    Ok(Some(ValidatedHnswGeneration {
        generation: generation_name,
        basename,
        source_identity: receipt.source_identity.clone(),
        graph,
    }))
}

fn hnsw_generation_is_loadable(
    generation_path: &Path,
    basename: &str,
    graph_path: &Path,
    expected_points: usize,
    expected_dimension: usize,
) -> bool {
    let Ok(graph_file) = std::fs::File::open(graph_path) else {
        return false;
    };
    let mut graph_reader = std::io::BufReader::new(graph_file);
    let description = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        hnsw_rs::prelude::load_description(&mut graph_reader)
    }));
    let Ok(Ok(description)) = description else {
        return false;
    };
    if description.nb_point != expected_points
        || (expected_points != 0 && description.dimension != expected_dimension)
    {
        return false;
    }

    // `hnsw_rs` currently unwraps a few native-parser results internally. Keep
    // corrupt retained generations on the normal reject-and-redump path rather
    // than letting a validation-only reuse probe unwind through `save()`.
    let mut native_io = HnswIo::new(generation_path, basename);
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Ok(candidate) = native_io.load_hnsw::<f32, DistDot>() else {
            return false;
        };
        candidate.get_nb_point() == expected_points
            && validate_hnsw_topology(&candidate, expected_points).is_ok()
    }))
    .unwrap_or(false)
}

fn write_hnsw_generation_receipt(
    generation_path: &Path,
    receipt: &HnswGenerationReceipt,
) -> SearchResult<()> {
    let bytes = serde_json::to_vec(receipt)
        .map_err(|error| SearchError::Io(std::io::Error::other(error.to_string())))?;
    if bytes.len() > HNSW_GENERATION_RECEIPT_MAX_BYTES {
        return Err(SearchError::Io(std::io::Error::other(format!(
            "HNSW generation receipt exceeds {} bytes",
            HNSW_GENERATION_RECEIPT_MAX_BYTES
        ))));
    }

    let receipt_path = generation_path.join(HNSW_GENERATION_RECEIPT_FILENAME);
    let mut receipt_file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&receipt_path)
        .map_err(SearchError::Io)?;
    receipt_file.write_all(&bytes).map_err(SearchError::Io)?;
    receipt_file.sync_all().map_err(SearchError::Io)?;
    sync_hnsw_directory(generation_path)
}

fn persisted_hnsw_sidecar_location(
    path: &Path,
    meta: &HnswMeta,
) -> SearchResult<(PathBuf, String)> {
    let parent = path
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let generation = meta
        .sidecar_generation
        .as_deref()
        .ok_or_else(|| ann_corrupted(path, "current HNSW metadata has no sidecar generation"))?;
    let generation = validate_hnsw_sidecar_basename(path, generation)?;
    let basename = meta
        .sidecar_basename
        .as_deref()
        .ok_or_else(|| ann_corrupted(path, "current HNSW metadata has no sidecar basename"))?;
    let basename = validate_hnsw_sidecar_basename(path, basename)?;
    Ok((parent.join(generation), basename))
}

fn validate_hnsw_sidecar_basename(path: &Path, basename: &str) -> SearchResult<String> {
    let candidate = Path::new(basename);
    if candidate.file_name() != Some(candidate.as_os_str()) {
        return Err(ann_corrupted(
            path,
            "HNSW native sidecar basename must be one non-empty path component",
        ));
    }
    Ok(basename.to_owned())
}

fn native_sidecar_pair_is_local(
    metadata_path: &Path,
    generation: &Path,
    graph: &Path,
    data: &Path,
) -> bool {
    let metadata_parent = metadata_path
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let Ok(canonical_parent) = std::fs::canonicalize(metadata_parent) else {
        return false;
    };

    let Ok(generation_metadata) = std::fs::symlink_metadata(generation) else {
        return false;
    };
    if generation_metadata.file_type().is_symlink() || !generation_metadata.is_dir() {
        return false;
    }
    let Ok(canonical_generation) = std::fs::canonicalize(generation) else {
        return false;
    };
    if canonical_generation.parent() != Some(canonical_parent.as_path()) {
        return false;
    }

    [graph, data].into_iter().all(|sidecar| {
        let Ok(sidecar_metadata) = std::fs::symlink_metadata(sidecar) else {
            return false;
        };
        if sidecar_metadata.file_type().is_symlink() || !sidecar_metadata.is_file() {
            return false;
        }
        std::fs::canonicalize(sidecar).is_ok_and(|canonical_sidecar| {
            canonical_sidecar.parent() == Some(canonical_generation.as_path())
        })
    })
}

fn sync_hnsw_sidecars(parent: &Path, basename: &str) -> SearchResult<()> {
    for suffix in [".hnsw.graph", ".hnsw.data"] {
        let sidecar_path = parent.join(format!("{basename}{suffix}"));
        let sidecar = std::fs::File::open(&sidecar_path).map_err(|error| {
            SearchError::Io(std::io::Error::new(
                error.kind(),
                format!(
                    "failed to open dumped HNSW sidecar '{}': {error}",
                    sidecar_path.display()
                ),
            ))
        })?;
        sidecar.sync_all().map_err(|error| {
            SearchError::Io(std::io::Error::new(
                error.kind(),
                format!(
                    "failed to sync dumped HNSW sidecar '{}': {error}",
                    sidecar_path.display()
                ),
            ))
        })?;
    }
    sync_hnsw_directory(parent)?;
    Ok(())
}

fn sync_hnsw_directory(directory: &Path) -> SearchResult<()> {
    #[cfg(unix)]
    {
        let handle = std::fs::File::open(directory).map_err(|error| {
            SearchError::Io(std::io::Error::new(
                error.kind(),
                format!(
                    "failed to open HNSW parent directory '{}': {error}",
                    directory.display()
                ),
            ))
        })?;
        handle.sync_all().map_err(|error| {
            SearchError::Io(std::io::Error::new(
                error.kind(),
                format!(
                    "failed to sync HNSW parent directory '{}': {error}",
                    directory.display()
                ),
            ))
        })?;
    }
    #[cfg(not(unix))]
    {
        let _ = directory;
    }
    Ok(())
}

fn publish_hnsw_metadata(path: &Path, parent: &Path, bytes: &[u8]) -> SearchResult<()> {
    install_hnsw_metadata(path, parent, bytes)?;
    sync_hnsw_directory(parent)
}

/// Remove stale native HNSW generations after metadata has durably selected a
/// replacement.
///
/// The save lock is held by the caller for the complete publication and
/// cleanup sequence.  A failed or interrupted metadata publication never
/// reaches this function, preserving READY generations for a retry.  Only
/// sibling directories in this metadata role's private generation namespace
/// are eligible; symlinks and unrelated filesystem entries are left intact.
fn gc_superseded_hnsw_generations(
    parent: &Path,
    requested_basename: &str,
    published_metadata: &HnswMeta,
) -> SearchResult<()> {
    let retained_generation = published_metadata
        .sidecar_generation
        .as_deref()
        .ok_or_else(|| ann_corrupted(parent, "published HNSW metadata has no generation"))?;
    let generation_prefix = format!(".{requested_basename}.generation-");

    for entry in std::fs::read_dir(parent).map_err(SearchError::Io)? {
        let entry = entry.map_err(SearchError::Io)?;
        let file_name = entry.file_name();
        let Some(file_name) = file_name.to_str() else {
            continue;
        };
        if file_name == retained_generation || !file_name.starts_with(&generation_prefix) {
            continue;
        }

        let file_type = entry.file_type().map_err(SearchError::Io)?;
        if file_type.is_symlink() || !file_type.is_dir() {
            continue;
        }

        std::fs::remove_dir_all(entry.path()).map_err(|error| {
            SearchError::Io(std::io::Error::new(
                error.kind(),
                format!(
                    "failed to remove superseded HNSW generation '{}': {error}",
                    entry.path().display()
                ),
            ))
        })?;
    }

    sync_hnsw_directory(parent)
}

fn install_hnsw_metadata(path: &Path, parent: &Path, bytes: &[u8]) -> SearchResult<()> {
    let mut temporary_builder = tempfile::Builder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        // Keep the atomically persisted metadata consistent with ordinary
        // index files: 0666 constrained by the deployment's umask, rather
        // than tempfile's private 0600 default.
        temporary_builder.permissions(std::fs::Permissions::from_mode(0o666));
    }
    let mut temporary = temporary_builder
        .tempfile_in(parent)
        .map_err(SearchError::Io)?;
    temporary.write_all(bytes).map_err(SearchError::Io)?;
    temporary.as_file().sync_all().map_err(SearchError::Io)?;
    temporary.persist(path).map_err(|error| {
        SearchError::Io(std::io::Error::new(
            error.error.kind(),
            format!(
                "failed to atomically publish HNSW metadata '{}': {}",
                path.display(),
                error.error
            ),
        ))
    })?;
    Ok(())
}

/// FNV-1a 64-bit. Chosen because it is deterministic across processes (unlike
/// `ahash`) and stdlib (`DefaultHasher` is randomized + deprecated for
/// persistence), keeps the fingerprint dependency-free, and is plenty for an
/// integrity-style check — we only need collision resistance against a small
/// number of accidental byte-level edits, not adversarial inputs.
const FNV_OFFSET_BASIS_64: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME_64: u64 = 0x0000_0100_0000_01b3;

#[inline]
fn fnv1a_update(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME_64);
    }
    h
}

fn fingerprint_doc_ids(doc_ids: &[String]) -> u64 {
    let mut h = fnv1a_update(FNV_OFFSET_BASIS_64, &doc_ids.len().to_le_bytes());
    for (index, doc_id) in doc_ids.iter().enumerate() {
        h = fnv1a_update(h, &index.to_le_bytes());
        h = fnv1a_update(h, &doc_id.len().to_le_bytes());
        h = fnv1a_update(h, doc_id.as_bytes());
    }
    h
}

fn fingerprint_hnsw_sidecar(path: &Path) -> SearchResult<HnswSidecarDigest> {
    let mut file = std::fs::File::open(path).map_err(|error| {
        SearchError::Io(std::io::Error::new(
            error.kind(),
            format!(
                "failed to open HNSW sidecar '{}' for READY receipt: {error}",
                path.display()
            ),
        ))
    })?;
    let mut buffer = vec![0_u8; 64 * 1024].into_boxed_slice();
    let mut byte_len = 0_u64;
    let mut fingerprint = FNV_OFFSET_BASIS_64;
    loop {
        let read = file.read(&mut buffer).map_err(|error| {
            SearchError::Io(std::io::Error::new(
                error.kind(),
                format!(
                    "failed to read HNSW sidecar '{}' for READY receipt: {error}",
                    path.display()
                ),
            ))
        })?;
        if read == 0 {
            break;
        }
        let read_u64 = u64::try_from(read).map_err(|_| {
            SearchError::Io(std::io::Error::other(
                "HNSW sidecar read length exceeds u64",
            ))
        })?;
        byte_len = byte_len.checked_add(read_u64).ok_or_else(|| {
            SearchError::Io(std::io::Error::other("HNSW sidecar length exceeds u64"))
        })?;
        fingerprint = fnv1a_update(fingerprint, &buffer[..read]);
    }
    Ok(HnswSidecarDigest {
        byte_len,
        fnv1a64: fingerprint,
    })
}

/// Feed an `&[f32]` into the FNV-1a state in little-endian byte order without
/// any unsafe reinterpretation (the crate is `#![forbid(unsafe_code)]`).
/// One `to_le_bytes()` per element compiles to a tight loop.
#[inline]
fn fnv1a_update_f32(mut h: u64, vec: &[f32]) -> u64 {
    for &f in vec {
        h = fnv1a_update(h, &f.to_bits().to_le_bytes());
    }
    h
}

/// Compute the persistence fingerprint from the (raw, un-normalized) vectors
/// and doc IDs used to build the graph.
///
/// Every vector contributes in live-row order. Doc IDs are mixed in alongside
/// their vectors, so either a vector edit or a doc-id permutation changes the
/// digest. The output is stored in the native metadata sidecar and re-derived
/// at load time by [`fingerprint_live_vector_index`] against the live
/// `VectorIndex`; a mismatch means "doc IDs match but the underlying vector
/// bytes were silently swapped" → reject the persisted graph.
fn fingerprint_vectors(doc_ids: &[String], vectors: &[Vec<f32>]) -> u64 {
    // Mix length so a truncated-but-prefix-matching index doesn't collide.
    let mut h = fnv1a_update(FNV_OFFSET_BASIS_64, &(doc_ids.len() as u64).to_le_bytes());
    for (i, (doc_id, vector)) in doc_ids.iter().zip(vectors).enumerate() {
        h = fnv1a_update(h, &(i as u64).to_le_bytes());
        h = fnv1a_update(h, doc_id.as_bytes());
        h = fnv1a_update_f32(h, vector);
    }
    h
}

/// Compute the fingerprint from a live `VectorIndex`, matching the layout
/// [`fingerprint_vectors`] produced at build time.
///
/// `expected_len` and `expected_dim` come from the persisted metadata. If the
/// live index has fewer live records than the persisted graph, the digest
/// will not match and the caller falls back to a rebuild — which is the right
/// behavior.
fn fingerprint_live_vector_index(
    index: &VectorIndex,
    expected_len: usize,
    expected_dim: usize,
) -> SearchResult<u64> {
    let mut h = fnv1a_update(FNV_OFFSET_BASIS_64, &(expected_len as u64).to_le_bytes());

    // Walk live records (tombstones excluded), in row order — same iteration
    // order as `build_from_vector_index`.
    let mut live_idx = 0_usize;
    for raw in 0..index.record_count() {
        if index.is_deleted(raw) {
            continue;
        }
        if live_idx >= expected_len {
            // Live index has more records than the persisted graph.
            break;
        }
        let doc_id = index.doc_id_at(raw)?;
        h = fnv1a_update(h, &(live_idx as u64).to_le_bytes());
        h = fnv1a_update(h, doc_id.as_bytes());

        let vec = index.vector_at_f32(raw)?;
        if vec.len() != expected_dim {
            // Dimension drift — perturb the digest so the caller rejects and
            // rebuilds rather than asserting.
            return Ok(h.wrapping_add(1));
        }
        h = fnv1a_update_f32(h, &vec);
        live_idx += 1;
    }
    Ok(h)
}

/// Recompute a graph fingerprint from its original physical source rows.
///
/// Unlike [`fingerprint_live_vector_index`], this deliberately includes rows
/// that were tombstoned after the graph was built. Soft deletion is a valid
/// in-process state transition: native candidates for those rows are filtered,
/// while an exact underfill repair scans the source's current live set. A row
/// move, document-identity change, or vector mutation still changes the digest
/// and fails closed.
fn fingerprint_vector_index_positions(
    index: &VectorIndex,
    positions: &[u32],
    expected_doc_ids: &[String],
    expected_dim: usize,
) -> SearchResult<u64> {
    if positions.len() != expected_doc_ids.len() {
        return Err(ann_corrupted(
            &index.path,
            format!(
                "HNSW source map has {} rows for {} document identities",
                positions.len(),
                expected_doc_ids.len()
            ),
        ));
    }
    let mut h = fnv1a_update(FNV_OFFSET_BASIS_64, &(positions.len() as u64).to_le_bytes());
    for (logical_index, (&physical_position, expected_doc_id)) in
        positions.iter().zip(expected_doc_ids).enumerate()
    {
        let physical_position = usize::try_from(physical_position).map_err(|_| {
            ann_corrupted(
                &index.path,
                format!("HNSW source row {physical_position} does not fit usize"),
            )
        })?;
        if physical_position >= index.record_count() {
            return Err(ann_corrupted(
                &index.path,
                format!(
                    "HNSW source row {physical_position} exceeds {} physical records",
                    index.record_count()
                ),
            ));
        }
        if index.doc_id_at(physical_position)? != expected_doc_id {
            return Err(ann_corrupted(
                &index.path,
                format!("HNSW source row {physical_position} has a different document identity"),
            ));
        }
        let vector = index.vector_at_f32(physical_position)?;
        if vector.len() != expected_dim {
            return Err(SearchError::DimensionMismatch {
                expected: expected_dim,
                found: vector.len(),
            });
        }
        h = fnv1a_update(h, &(logical_index as u64).to_le_bytes());
        h = fnv1a_update(h, expected_doc_id.as_bytes());
        h = fnv1a_update_f32(h, &vector);
    }
    Ok(h)
}

fn live_vector_positions(index: &VectorIndex) -> Vec<usize> {
    (0..index.record_count())
        .filter(|&position| !index.is_deleted(position))
        .collect()
}

/// Verify the metadata `doc_ids` sequence matches the live `VectorIndex`'s
/// live (non-tombstoned) doc IDs in row order. Same semantics as the public
/// `matches_vector_index` but doesn't need a constructed `HnswIndex`, so we
/// can check it before paying for the native graph load.
fn meta_matches_live_doc_ids(meta: &HnswMeta, index: &VectorIndex) -> SearchResult<bool> {
    if meta.dimension != index.dimension() {
        return Ok(false);
    }
    let mut live_position = 0_usize;
    for i in 0..index.record_count() {
        if index.is_deleted(i) {
            continue;
        }
        let Some(expected_doc_id) = meta.doc_ids.get(live_position) else {
            return Ok(false);
        };
        if expected_doc_id != index.doc_id_at(i)? {
            return Ok(false);
        }
        live_position = live_position.saturating_add(1);
    }
    Ok(live_position == meta.doc_ids.len())
}

fn validate_config(config: HnswConfig) -> SearchResult<()> {
    if config.m == 0 {
        return Err(SearchError::InvalidConfig {
            field: "hnsw_m".to_owned(),
            value: "0".to_owned(),
            reason: "hnsw_m must be greater than zero".to_owned(),
        });
    }
    if config.m > usize::from(u8::MAX) {
        return Err(SearchError::InvalidConfig {
            field: "hnsw_m".to_owned(),
            value: config.m.to_string(),
            reason: format!("hnsw_m must be <= {}", u8::MAX),
        });
    }
    if config.ef_construction == 0 {
        return Err(SearchError::InvalidConfig {
            field: "hnsw_ef_construction".to_owned(),
            value: "0".to_owned(),
            reason: "hnsw_ef_construction must be greater than zero".to_owned(),
        });
    }
    if config.ef_search == 0 {
        return Err(SearchError::InvalidConfig {
            field: "hnsw_ef_search".to_owned(),
            value: "0".to_owned(),
            reason: "hnsw_ef_search must be greater than zero".to_owned(),
        });
    }
    if config.max_layer == 0 {
        return Err(SearchError::InvalidConfig {
            field: "hnsw_max_layer".to_owned(),
            value: "0".to_owned(),
            reason: "hnsw_max_layer must be greater than zero".to_owned(),
        });
    }
    if config.max_layer > HNSW_DEFAULT_MAX_LAYER {
        return Err(SearchError::InvalidConfig {
            field: "hnsw_max_layer".to_owned(),
            value: config.max_layer.to_string(),
            reason: format!("hnsw_max_layer must be <= {HNSW_DEFAULT_MAX_LAYER}"),
        });
    }
    Ok(())
}

fn ann_corrupted(path: &Path, detail: impl Into<String>) -> SearchError {
    SearchError::IndexCorrupted {
        path: path.to_path_buf(),
        detail: detail.into(),
    }
}

/// Derive the normalization and score-restoration budget for `DistDot`.
///
/// A length-`n` f32 dot product has a forward-error bound conventionally
/// expressed as `gamma_k = k*u/(1-k*u)`, where `u = eps/2`. We budget `8n+32`
/// rounded operations: component rescaling, multiplication, lane-local
/// accumulation, horizontal reduction, score restoration, and a 2x
/// architecture margin. Scaling exact unit vectors to
/// `1/sqrt(1+gamma_k)` makes the worst bounded computed self-dot no greater
/// than one. Dimensions for which `k*u > 1/4` are rejected rather than hiding
/// an ill-conditioned error bound behind an arbitrary shrink factor.
#[allow(clippy::cast_possible_truncation)]
fn dist_dot_budget(dimension: usize) -> SearchResult<DistDotBudget> {
    if dimension == 0 || dimension > DIST_DOT_MAX_DIMENSION {
        return Err(SearchError::InvalidConfig {
            field: "dimension".to_owned(),
            value: dimension.to_string(),
            reason: format!(
                "DistDot requires a dimension in 1..={DIST_DOT_MAX_DIMENSION} \
                 so its f32 roundoff bound remains finite and conservative"
            ),
        });
    }

    let dimension = u32::try_from(dimension).map_err(|_| SearchError::InvalidConfig {
        field: "dimension".to_owned(),
        value: dimension.to_string(),
        reason: "dimension exceeds the DistDot f32 error model".to_owned(),
    })?;
    let rounded_operations = f64::from(dimension).mul_add(8.0, 32.0);
    let unit_roundoff = f64::from(f32::EPSILON) / 2.0;
    let accumulated_roundoff = rounded_operations * unit_roundoff;
    debug_assert!(accumulated_roundoff <= 0.25);
    let gamma = accumulated_roundoff / (1.0 - accumulated_roundoff);
    let radius_squared = 1.0 / (1.0 + gamma);
    // Include a small fixed allowance for the final f32 subtraction and
    // division when converting DistDot's distance back to cosine score.
    let score_tolerance = gamma + 8.0 * f64::from(f32::EPSILON);
    Ok(DistDotBudget {
        radius_squared: radius_squared as f32,
        score_tolerance: score_tolerance as f32,
    })
}

#[allow(clippy::cast_possible_truncation)]
fn normalize_for_dist_dot(mut vector: Vec<f32>, budget: DistDotBudget) -> Vec<f32> {
    // f64 accumulation prevents the normalization pass itself from consuming
    // the f32 error budget intended for hnsw_rs/anndists' distance reduction.
    let norm_squared = vector
        .iter()
        .map(|&value| {
            let value = f64::from(value);
            value * value
        })
        .sum::<f64>();
    if norm_squared > 0.0 && norm_squared.is_finite() {
        let radius = f64::from(budget.radius_squared).sqrt();
        let scale = radius / norm_squared.sqrt();
        for value in &mut vector {
            *value = (f64::from(*value) * scale) as f32;
        }
    }
    vector
}

#[cfg(test)]
#[allow(dead_code)] // retained as utility; direct callers use vector_component_close
fn vectors_close(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(&l, &r)| vector_component_close(l, r))
}

#[cfg(test)]
fn vector_component_close(left: f32, right: f32) -> bool {
    if left.to_bits() == right.to_bits() {
        return true;
    }
    // Non-finite values (NaN, Inf) with different bit patterns are never close.
    if !left.is_finite() || !right.is_finite() {
        return false;
    }
    let diff = (left - right).abs();
    let scale = left.abs().max(right.abs()).max(1.0);
    diff <= (f32::EPSILON * 8.0 * scale)
}

fn estimate_recall(ef_search: usize, k: usize) -> f64 {
    if k == 0 {
        return 1.0;
    }
    let numerator = f64::from(u32::try_from(ef_search.max(1)).unwrap_or(u32::MAX));
    let denominator = f64::from(u32::try_from(k).unwrap_or(u32::MAX));
    let ratio = numerator / denominator;
    0.1_f64.mul_add(ratio.log2(), 0.9_f64).clamp(0.0, 1.0)
}

/// Recall@k of `approx` against exact `exact`: the fraction of exact `doc_id`s that
/// also appear in `approx`. Used by [`HnswIndex::certify_ef_search`] to build the
/// measured calibration sample fed to the conformal certificate. `k` is tiny (top-k),
/// so the nested membership scan is trivial.
fn recall_at_k_of(approx: &[VectorHit], exact: &[VectorHit]) -> f64 {
    if exact.is_empty() {
        return 1.0;
    }
    let overlap = exact
        .iter()
        .filter(|e| approx.iter().any(|a| a.doc_id == e.doc_id))
        .count();
    #[allow(clippy::cast_precision_loss)]
    let ratio = overlap as f64 / exact.len() as f64;
    ratio
}

fn certified_ann_recall_sample(
    result: SearchResult<(Vec<VectorHit>, AnnSearchStats)>,
    exact: &[VectorHit],
) -> f64 {
    match result {
        Ok((approx, stats)) if stats.fallback_reason.is_none() => recall_at_k_of(&approx, exact),
        Ok(_) | Err(_) => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::collections::HashSet;
    use std::path::PathBuf;
    use std::rc::Rc;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[cfg(feature = "hnsw-patch-ab")]
    use hnsw_rs_034::prelude::{
        AnnT as BaselineAnnT, DistDot as BaselineDistDot, Hnsw as BaselineHnsw,
    };

    use super::*;
    use crate::Quantization;

    #[derive(Debug)]
    struct GraphRepairProbe {
        label: &'static str,
        valid: bool,
        live_graphs: Rc<Cell<usize>>,
    }

    impl GraphRepairProbe {
        fn new(label: &'static str, valid: bool, live_graphs: &Rc<Cell<usize>>) -> Self {
            live_graphs.set(live_graphs.get().saturating_add(1));
            Self {
                label,
                valid,
                live_graphs: Rc::clone(live_graphs),
            }
        }
    }

    impl Drop for GraphRepairProbe {
        fn drop(&mut self) {
            self.live_graphs
                .set(self.live_graphs.get().saturating_sub(1));
        }
    }

    fn temp_path(label: &str, extension: &str) -> PathBuf {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "frankensearch-hnsw-{label}-{}-{now}.{extension}",
            std::process::id()
        ))
    }

    fn lcg_next(state: &mut u64) -> u32 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        u32::try_from((*state >> 32) & u64::from(u32::MAX)).unwrap_or(u32::MAX)
    }

    fn normalized_vector(seed: usize, dimension: usize) -> Vec<f32> {
        let mut state = u64::try_from(seed).unwrap_or(0).wrapping_add(1);
        let mut out = Vec::with_capacity(dimension);
        for _ in 0..dimension {
            let random = lcg_next(&mut state);
            let upper = u16::try_from((random >> 16) & u32::from(u16::MAX)).unwrap_or(u16::MAX);
            let raw = f32::from(upper) / f32::from(u16::MAX);
            out.push((raw * 2.0_f32) - 1.0_f32);
        }
        let norm = out.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut out {
                *value /= norm;
            }
        }
        out
    }

    fn write_index(path: &Path, vectors: &[Vec<f32>]) -> SearchResult<VectorIndex> {
        write_index_with_quantization(path, vectors, Quantization::F32)
    }

    fn write_index_with_quantization(
        path: &Path,
        vectors: &[Vec<f32>],
        quantization: Quantization,
    ) -> SearchResult<VectorIndex> {
        let dimension = vectors.first().map_or(8, Vec::len);
        let mut writer =
            VectorIndex::create_with_revision(path, "hash", "test", dimension, quantization)?;
        for (idx, vector) in vectors.iter().enumerate() {
            writer.write_record(&format!("doc-{idx:04}"), vector)?;
        }
        writer.finish()?;
        VectorIndex::open(path)
    }

    #[derive(Debug, PartialEq, Eq)]
    struct ImmutableTreeEntry {
        relative_path: PathBuf,
        kind: &'static str,
        len: u64,
        modified: Option<SystemTime>,
        readonly: bool,
        permission_mode: Option<u32>,
        bytes: Option<Vec<u8>>,
    }

    fn snapshot_immutable_tree(root: &Path) -> std::io::Result<Vec<ImmutableTreeEntry>> {
        fn visit(
            root: &Path,
            path: &Path,
            entries: &mut Vec<ImmutableTreeEntry>,
        ) -> std::io::Result<()> {
            let metadata = std::fs::symlink_metadata(path)?;
            let file_type = metadata.file_type();
            let kind = if file_type.is_file() {
                "file"
            } else if file_type.is_dir() {
                "directory"
            } else if file_type.is_symlink() {
                "symlink"
            } else {
                "other"
            };
            let relative_path = path.strip_prefix(root).unwrap_or(path).to_path_buf();
            let bytes = file_type
                .is_file()
                .then(|| std::fs::read(path))
                .transpose()?;
            #[cfg(unix)]
            let permission_mode = {
                use std::os::unix::fs::PermissionsExt;
                Some(metadata.permissions().mode())
            };
            #[cfg(not(unix))]
            let permission_mode = None;
            entries.push(ImmutableTreeEntry {
                relative_path,
                kind,
                len: metadata.len(),
                modified: metadata.modified().ok(),
                readonly: metadata.permissions().readonly(),
                permission_mode,
                bytes,
            });
            if file_type.is_dir() {
                let mut children = std::fs::read_dir(path)?
                    .map(|entry| entry.map(|entry| entry.path()))
                    .collect::<std::io::Result<Vec<_>>>()?;
                children.sort();
                for child in children {
                    visit(root, &child, entries)?;
                }
            }
            Ok(())
        }

        let mut entries = Vec::new();
        visit(root, root, &mut entries)?;
        entries.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
        Ok(entries)
    }

    fn reject_hnsw_metadata_publish(_: &Path, _: &Path, _: &[u8]) -> SearchResult<()> {
        Err(SearchError::Io(std::io::Error::other(
            "injected metadata publication failure",
        )))
    }

    fn install_then_report_parent_sync_failure(
        path: &Path,
        parent: &Path,
        bytes: &[u8],
    ) -> SearchResult<()> {
        install_hnsw_metadata(path, parent, bytes)?;
        Err(SearchError::Io(std::io::Error::other(
            "injected post-rename parent sync failure",
        )))
    }

    fn ready_generation_paths(metadata_path: &Path, vector_fingerprint: u64) -> Vec<PathBuf> {
        let parent = metadata_path.parent().unwrap_or_else(|| Path::new("."));
        let basename = hnsw_sidecar_basename(metadata_path).expect("metadata basename");
        let prefix = hnsw_generation_prefix(&basename, vector_fingerprint);
        let mut paths: Vec<PathBuf> = std::fs::read_dir(parent)
            .expect("read metadata parent")
            .filter_map(Result::ok)
            .filter(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .is_some_and(|name| name.starts_with(&prefix))
                    && entry
                        .path()
                        .join(HNSW_GENERATION_RECEIPT_FILENAME)
                        .is_file()
            })
            .map(|entry| entry.path())
            .collect();
        paths.sort_unstable();
        paths
    }

    fn install_digest_valid_unloadable_data(generation_path: &Path, basename: &str) {
        let data_path = generation_path.join(format!("{basename}.hnsw.data"));
        std::fs::write(&data_path, b"not loadable HNSW vector data")
            .expect("corrupt native data fixture");
        let receipt_path = generation_path.join(HNSW_GENERATION_RECEIPT_FILENAME);
        let mut receipt: HnswGenerationReceipt =
            serde_json::from_slice(&std::fs::read(&receipt_path).expect("read receipt"))
                .expect("parse receipt");
        receipt.data = fingerprint_hnsw_sidecar(&data_path).expect("fingerprint corrupt data");
        std::fs::write(
            receipt_path,
            serde_json::to_vec(&receipt).expect("serialize updated receipt"),
        )
        .expect("make corrupt pair digest-valid");
    }

    #[cfg(feature = "hnsw-patch-ab")]
    fn install_digest_valid_wrong_layer_generation(
        generation_path: &Path,
        installed_basename: &str,
        vectors: Vec<Vec<f32>>,
    ) {
        let installed_graph = generation_path.join(format!("{installed_basename}.hnsw.graph"));
        let installed_data = generation_path.join(format!("{installed_basename}.hnsw.data"));
        let malformed_directory = tempfile::tempdir().expect("malformed fixture directory");
        let budget = dist_dot_budget(vectors[0].len()).expect("distance budget");
        let normalized = vectors
            .into_iter()
            .map(|vector| normalize_for_dist_dot(vector, budget))
            .collect::<Vec<_>>();

        let mut malformed_basename = None;
        for attempt in 0..4_096 {
            let baseline = BaselineHnsw::new(16, normalized.len(), 16, 200, BaselineDistDot);
            for (origin_id, vector) in normalized.iter().enumerate() {
                baseline.insert((vector, origin_id));
            }
            let malformed = baseline.get_point_indexation().into_iter().any(|point| {
                let max_layer = usize::from(point.get_point_id().0);
                point
                    .get_neighborhood_id()
                    .iter()
                    .skip(max_layer.saturating_add(1))
                    .any(|neighbors| !neighbors.is_empty())
            });
            if malformed {
                malformed_basename = Some(
                    baseline
                        .file_dump(malformed_directory.path(), &format!("malformed-{attempt}"))
                        .expect("dump malformed baseline graph"),
                );
                break;
            }
        }
        let malformed_basename =
            malformed_basename.expect("published 0.3.4 must reproduce the wrong-layer topology");
        std::fs::copy(
            malformed_directory
                .path()
                .join(format!("{malformed_basename}.hnsw.graph")),
            &installed_graph,
        )
        .expect("install malformed graph");
        std::fs::copy(
            malformed_directory
                .path()
                .join(format!("{malformed_basename}.hnsw.data")),
            &installed_data,
        )
        .expect("install malformed data");

        let receipt_path = generation_path.join(HNSW_GENERATION_RECEIPT_FILENAME);
        let mut receipt: HnswGenerationReceipt =
            serde_json::from_slice(&std::fs::read(&receipt_path).expect("receipt"))
                .expect("parse receipt");
        receipt.graph = fingerprint_hnsw_sidecar(&installed_graph).expect("graph digest");
        receipt.data = fingerprint_hnsw_sidecar(&installed_data).expect("data digest");
        std::fs::write(
            receipt_path,
            serde_json::to_vec(&receipt).expect("serialize updated receipt"),
        )
        .expect("update receipt");
    }

    fn recall_at_k(approx: &[VectorHit], exact: &[VectorHit]) -> f64 {
        if exact.is_empty() {
            return 1.0;
        }
        let exact_ids: HashSet<&str> = exact.iter().map(|hit| hit.doc_id.as_str()).collect();
        let overlap = approx
            .iter()
            .filter(|hit| exact_ids.contains(hit.doc_id.as_str()))
            .count();
        f64::from(u32::try_from(overlap).unwrap_or(u32::MAX))
            / f64::from(u32::try_from(exact.len()).unwrap_or(u32::MAX))
    }

    #[test]
    fn empty_index_returns_no_hits() {
        let path = temp_path("empty", "fsvi");
        let writer = VectorIndex::create_with_revision(&path, "hash", "test", 8, Quantization::F16)
            .expect("create writer");
        writer.finish().expect("finish");

        let index = VectorIndex::open(&path).expect("open index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");
        let hits = ann
            .knn_search(&normalized_vector(7, 8), 10, HNSW_DEFAULT_EF_SEARCH)
            .expect("search");
        assert!(hits.is_empty());
    }

    #[test]
    fn single_vector_round_trip() {
        let path = temp_path("single", "fsvi");
        let index = write_index(&path, &[normalized_vector(1, 32)]).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");

        let hits = ann
            .knn_search(&normalized_vector(1, 32), 1, HNSW_DEFAULT_EF_SEARCH)
            .expect("search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "doc-0000");
    }

    #[test]
    fn small_serial_builds_preserve_topology_and_return_every_available_hit() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("four-thread test pool");
        pool.install(|| {
            for round in 0..32 {
                for count in [0_usize, 1, 2, 3, HNSW_DEFAULT_M, HNSW_DEFAULT_M + 1] {
                    let doc_ids = (0..count)
                        .map(|index| format!("round-{round}-doc-{index}"))
                        .collect::<Vec<_>>();
                    let vectors = (0..count)
                        .map(|index| normalized_vector(round * 100 + index, 16))
                        .collect::<Vec<_>>();
                    let ann = HnswIndex::build_from_parts(
                        doc_ids,
                        vectors.clone(),
                        16,
                        HnswConfig::default(),
                    )
                    .expect("small graph build");
                    validate_hnsw_topology(&ann.hnsw, count).expect("topology attestation");

                    for query in &vectors {
                        let (hits, stats) = ann
                            .knn_search_with_stats(query, count, HNSW_DEFAULT_EF_SEARCH)
                            .expect("full small-graph search");
                        assert_eq!(hits.len(), count, "round={round}, count={count}");
                        assert_eq!(stats.fallback_reason, None);
                    }
                }
            }
        });
    }

    #[test]
    fn parallel_insert_threshold_scales_with_the_active_rayon_pool() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("four-thread test pool");
        pool.install(|| {
            let threshold = 4 * HNSW_PARALLEL_INSERT_MIN_POINTS_PER_THREAD;
            // One point is seeded serially, so the decision is based on the
            // remaining insertion batch rather than total graph size.
            assert!(!should_use_parallel_insert(threshold));
            assert!(should_use_parallel_insert(threshold + 1));
        });
    }

    #[test]
    fn failed_parallel_attestation_drops_graph_then_repairs_once_serially() {
        let live_graphs = Rc::new(Cell::new(0_usize));
        let rebuilds = Cell::new(0_usize);
        let initial = GraphRepairProbe::new("parallel-invalid", false, &live_graphs);

        let repaired = attest_or_rebuild_serial(
            initial,
            true,
            50_000,
            |graph| {
                graph
                    .valid
                    .then_some(())
                    .ok_or_else(|| graph.label.to_owned())
            },
            || {
                assert_eq!(
                    live_graphs.get(),
                    0,
                    "invalid graph must be dropped before allocating its replacement"
                );
                rebuilds.set(rebuilds.get().saturating_add(1));
                GraphRepairProbe::new("serial-valid", true, &live_graphs)
            },
        )
        .expect("serial repair");

        assert_eq!(repaired.label, "serial-valid");
        assert_eq!(rebuilds.get(), 1);
        assert_eq!(live_graphs.get(), 1);
        drop(repaired);
        assert_eq!(live_graphs.get(), 0);
    }

    #[test]
    fn failed_serial_repair_reports_both_attestations_and_fails_closed() {
        let live_graphs = Rc::new(Cell::new(0_usize));
        let rebuilds = Cell::new(0_usize);
        let initial = GraphRepairProbe::new("parallel-invalid", false, &live_graphs);

        let error = attest_or_rebuild_serial(
            initial,
            true,
            50_000,
            |graph| {
                graph
                    .valid
                    .then_some(())
                    .ok_or_else(|| graph.label.to_owned())
            },
            || {
                assert_eq!(live_graphs.get(), 0);
                rebuilds.set(rebuilds.get().saturating_add(1));
                GraphRepairProbe::new("serial-invalid", false, &live_graphs)
            },
        )
        .expect_err("both invalid graphs must fail closed");

        assert!(error.contains("parallel-invalid"), "{error}");
        assert!(error.contains("serial-invalid"), "{error}");
        assert_eq!(rebuilds.get(), 1);
        assert_eq!(live_graphs.get(), 0);

        let serial_only = GraphRepairProbe::new("serial-only-invalid", false, &live_graphs);
        let error = attest_or_rebuild_serial(
            serial_only,
            false,
            3,
            |graph| Err(graph.label.to_owned()),
            || panic!("a serial build must never recursively rebuild"),
        )
        .expect_err("serial construction failure");
        assert_eq!(error, "serial-only-invalid");
        assert_eq!(rebuilds.get(), 1);
        assert_eq!(live_graphs.get(), 0);
    }

    #[test]
    fn two_vector_queries_return_both_directions_without_fallback() {
        let vectors = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]];
        for _ in 0..128 {
            let ann = HnswIndex::build_from_parts(
                vec!["east".to_owned(), "north".to_owned()],
                vectors.clone(),
                2,
                HnswConfig::default(),
            )
            .expect("two-vector graph");
            for (expected_first, query) in vectors.iter().enumerate() {
                let (hits, stats) = ann
                    .knn_search_with_stats(query, 2, HNSW_DEFAULT_EF_SEARCH)
                    .expect("two-vector search");
                assert_eq!(hits.len(), 2);
                assert_eq!(
                    usize::try_from(hits[0].index).expect("hit index"),
                    expected_first
                );
                assert_eq!(stats.fallback_reason, None);
                assert!(stats.is_approximate);
            }
        }
    }

    #[test]
    fn higher_ef_improves_or_matches_recall() {
        let fsvi_path = temp_path("ef", "fsvi");
        let vectors: Vec<Vec<f32>> = (0..256).map(|i| normalized_vector(i, 96)).collect();
        let index = write_index(&fsvi_path, &vectors).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");

        let mut low_total = 0.0_f64;
        let mut high_total = 0.0_f64;
        let mut count = 0_u32;
        for query_seed in (0..128).step_by(16) {
            let query = normalized_vector(query_seed, 96);
            let exact = index.search_top_k(&query, 10, None).expect("exact");
            let low = ann.knn_search(&query, 10, 10).expect("low ef");
            let high = ann.knn_search(&query, 10, 100).expect("high ef");
            low_total += recall_at_k(&low, &exact);
            high_total += recall_at_k(&high, &exact);
            count += 1;
        }

        let count_f = f64::from(count);
        assert!((high_total / count_f) >= (low_total / count_f));
    }

    #[test]
    fn recall_against_bruteforce_is_high() {
        let fsvi_path = temp_path("recall", "fsvi");
        let vectors: Vec<Vec<f32>> = (0..1_000).map(|i| normalized_vector(i, 384)).collect();
        let index = write_index(&fsvi_path, &vectors).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");

        let mut total_recall = 0.0_f64;
        let mut query_count = 0_u32;
        for query_seed in (0..1_000).step_by(40) {
            let query = normalized_vector(query_seed, 384);
            let exact = index.search_top_k(&query, 10, None).expect("exact");
            let approx = ann
                .knn_search(&query, 10, HNSW_DEFAULT_EF_SEARCH)
                .expect("approx");
            total_recall += recall_at_k(&approx, &exact);
            query_count += 1;
        }

        let avg_recall = total_recall / f64::from(query_count);
        assert!(
            avg_recall >= 0.95,
            "expected avg recall >= 0.95, got {avg_recall:.4}"
        );
    }

    #[test]
    fn certify_ef_search_wires_conformal_certificate_end_to_end() {
        // Real ANN index + real bruteforce feeding the conformal certificate.
        let fsvi_path = temp_path("certify", "fsvi");
        let vectors: Vec<Vec<f32>> = (0..800).map(|i| normalized_vector(i, 384)).collect();
        let index = write_index(&fsvi_path, &vectors).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");
        // Calibration queries disjoint from... they're near corpus points, fine for wiring.
        let calibration: Vec<Vec<f32>> = (5..1_600)
            .step_by(64)
            .map(|s| normalized_vector(s, 384))
            .collect();
        let candidate_efs = [10usize, 40, 100, 200];

        // target=0.0 is always certified => cheapest ef, and the sweep must
        // short-circuit immediately (only the smallest ef is ever ANN-searched).
        let trivial = ann
            .certify_ef_search(&index, &calibration, &candidate_efs, 10, 0.0, 0.1)
            .expect("certify")
            .expect("some");
        assert!(trivial.chosen.meets_target);
        assert_eq!(
            trivial.chosen.ef_search, 10,
            "cheapest ef for a trivial target"
        );
        assert_eq!(
            trivial.sweep.len(),
            1,
            "short-circuits at the first certified ef"
        );

        // An unreachable target => no ef meets it, fall back to the best-certifiable
        // (largest ef here, since recall is non-decreasing in ef), full sweep measured,
        // and the certified bound is a real recall in [0, 1].
        let strict = ann
            .certify_ef_search(&index, &calibration, &candidate_efs, 10, 2.0, 0.1)
            .expect("certify")
            .expect("some");
        assert!(!strict.chosen.meets_target);
        assert_eq!(
            strict.sweep.len(),
            candidate_efs.len(),
            "measures all when none certifies"
        );
        assert!((0.0..=1.0).contains(&strict.chosen.certified_recall));
        // The best-certifiable fallback should be a high ef with a strong bound on
        // this tight synthetic corpus (sanity: ANN recovers most exact neighbours).
        assert!(
            strict.chosen.certified_recall > 0.3,
            "expected a meaningful certified recall, got {}",
            strict.chosen.certified_recall
        );
    }

    // ── Config validation edge cases ──────────────────────────────────

    #[test]
    fn validate_config_rejects_m_zero() {
        let config = HnswConfig {
            m: 0,
            ..HnswConfig::default()
        };
        let error = validate_config(config).unwrap_err();
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "hnsw_m"),
            "expected InvalidConfig for hnsw_m, got {error:?}"
        );
    }

    #[test]
    fn validate_config_rejects_m_256_that_cannot_round_trip_through_u8() {
        let config = HnswConfig {
            m: 256,
            ..HnswConfig::default()
        };
        let error = validate_config(config).unwrap_err();
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "hnsw_m"),
            "expected InvalidConfig for hnsw_m, got {error:?}"
        );
    }

    #[test]
    fn validate_config_rejects_ef_construction_zero() {
        let config = HnswConfig {
            ef_construction: 0,
            ..HnswConfig::default()
        };
        let error = validate_config(config).unwrap_err();
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "hnsw_ef_construction"),
            "expected InvalidConfig for ef_construction, got {error:?}"
        );
    }

    #[test]
    fn validate_config_rejects_ef_search_zero() {
        let config = HnswConfig {
            ef_search: 0,
            ..HnswConfig::default()
        };
        let error = validate_config(config).unwrap_err();
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "hnsw_ef_search"),
            "expected InvalidConfig for ef_search, got {error:?}"
        );
    }

    #[test]
    fn validate_config_rejects_max_layer_zero() {
        let config = HnswConfig {
            max_layer: 0,
            ..HnswConfig::default()
        };
        let error = validate_config(config).unwrap_err();
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "hnsw_max_layer"),
            "expected InvalidConfig for max_layer, got {error:?}"
        );
    }

    #[test]
    fn validate_config_rejects_max_layer_above_native_limit() {
        let config = HnswConfig {
            max_layer: HNSW_DEFAULT_MAX_LAYER + 1,
            ..HnswConfig::default()
        };
        let error = validate_config(config).unwrap_err();
        assert!(
            matches!(
                error,
                SearchError::InvalidConfig { ref field, .. } if field == "hnsw_max_layer"
            ),
            "expected InvalidConfig for hnsw_max_layer, got {error:?}"
        );
    }

    #[test]
    fn native_storage_boundaries_validate_and_round_trip() {
        let source_path = temp_path("config-boundary-source", "fsvi");
        let source =
            write_index(&source_path, &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]]).expect("source");
        let config = HnswConfig {
            m: usize::from(u8::MAX),
            max_layer: HNSW_DEFAULT_MAX_LAYER,
            ..HnswConfig::default()
        };
        validate_config(config).expect("native storage boundaries must validate");

        let ann = HnswIndex::build_from_vector_index(&source, config)
            .expect("build at native storage boundaries");
        let metadata_path = temp_path("config-boundary-sidecar", "hnsw");
        ann.save(&metadata_path)
            .expect("persist native storage boundaries");
        let (loaded, disposition) = HnswIndex::load_with_disposition(&metadata_path, &source)
            .expect("load native storage boundaries");
        assert_eq!(disposition, HnswLoadDisposition::Native);
        assert_eq!(loaded.config(), config);
    }

    // ── build_from_parts error paths ────────────────────────────────────

    #[test]
    fn build_rejects_dimension_zero() {
        let error =
            HnswIndex::build_from_parts(vec![], vec![], 0, HnswConfig::default()).unwrap_err();
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "dimension"),
            "expected InvalidConfig for dimension, got {error:?}"
        );
    }

    #[test]
    fn build_rejects_doc_id_vector_count_mismatch() {
        let error = HnswIndex::build_from_parts(
            vec!["a".to_owned(), "b".to_owned()],
            vec![vec![1.0, 0.0]],
            2,
            HnswConfig::default(),
        )
        .unwrap_err();
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "vectors"),
            "expected InvalidConfig for vectors, got {error:?}"
        );
    }

    #[test]
    fn build_rejects_vector_dimension_mismatch() {
        let error = HnswIndex::build_from_parts(
            vec!["a".to_owned()],
            vec![vec![1.0, 0.0, 0.0]], // 3D but declared 2D
            2,
            HnswConfig::default(),
        )
        .unwrap_err();
        assert!(
            matches!(
                error,
                SearchError::DimensionMismatch {
                    expected: 2,
                    found: 3
                }
            ),
            "expected DimensionMismatch, got {error:?}"
        );
    }

    #[test]
    fn build_rejects_nan_in_vector() {
        let error = HnswIndex::build_from_parts(
            vec!["a".to_owned()],
            vec![vec![1.0, f32::NAN]],
            2,
            HnswConfig::default(),
        )
        .unwrap_err();
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, ref reason, .. }
                     if field == "vector" && reason.contains("finite")),
            "expected InvalidConfig for non-finite vector, got {error:?}"
        );
    }

    #[test]
    fn build_rejects_infinity_in_vector() {
        let error = HnswIndex::build_from_parts(
            vec!["a".to_owned()],
            vec![vec![f32::INFINITY, 0.0]],
            2,
            HnswConfig::default(),
        )
        .unwrap_err();
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "vector"),
            "expected InvalidConfig for non-finite vector, got {error:?}"
        );
    }

    // ── vector_component_close edge cases ────────────────────────────────

    #[test]
    fn vector_component_close_rejects_infinity_vs_finite() {
        // Regression: Inf <= (EPSILON * 8 * Inf) was true, causing false positive.
        assert!(!vector_component_close(f32::INFINITY, 100.0));
        assert!(!vector_component_close(100.0, f32::INFINITY));
        assert!(!vector_component_close(f32::NEG_INFINITY, 0.0));
        assert!(!vector_component_close(0.0, f32::NEG_INFINITY));
    }

    #[test]
    fn vector_component_close_accepts_identical_infinities() {
        assert!(vector_component_close(f32::INFINITY, f32::INFINITY));
        assert!(vector_component_close(f32::NEG_INFINITY, f32::NEG_INFINITY));
    }

    #[test]
    fn vector_component_close_rejects_opposite_infinities() {
        assert!(!vector_component_close(f32::INFINITY, f32::NEG_INFINITY));
    }

    #[test]
    fn vector_component_close_nan_vs_finite_is_rejected() {
        assert!(!vector_component_close(f32::NAN, 0.0));
        assert!(!vector_component_close(0.0, f32::NAN));
    }

    #[test]
    fn vector_component_close_identical_nan_bits_accepted() {
        // Same NaN bit pattern passes the to_bits() fast path. This is fine:
        // NaN vectors are rejected at construction time by build_from_parts().
        assert!(vector_component_close(f32::NAN, f32::NAN));
    }

    #[test]
    fn vector_component_close_accepts_equal_values() {
        assert!(vector_component_close(0.0, 0.0));
        assert!(vector_component_close(1.0, 1.0));
        assert!(vector_component_close(-42.5, -42.5));
    }

    // ── knn_search_with_stats boundary conditions ───────────────────────

    #[test]
    fn search_with_k_zero_returns_empty_with_stats() {
        let path = temp_path("k0", "fsvi");
        let index = write_index(&path, &[normalized_vector(1, 16)]).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");
        let (hits, stats) = ann
            .knn_search_with_stats(&normalized_vector(1, 16), 0, HNSW_DEFAULT_EF_SEARCH)
            .expect("search");
        assert!(hits.is_empty());
        assert_eq!(stats.k_requested, 0);
        assert_eq!(stats.k_returned, 0);
        assert!(stats.is_approximate);
        assert_eq!(stats.fallback_reason, None);
        assert!((stats.estimated_recall - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn search_dimension_mismatch_returns_error() {
        let path = temp_path("dimmis", "fsvi");
        let index = write_index(&path, &[normalized_vector(1, 16)]).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");
        let error = ann
            .knn_search_with_stats(&normalized_vector(1, 8), 5, HNSW_DEFAULT_EF_SEARCH)
            .unwrap_err();
        assert!(
            matches!(
                error,
                SearchError::DimensionMismatch {
                    expected: 16,
                    found: 8
                }
            ),
            "expected DimensionMismatch, got {error:?}"
        );
    }

    #[test]
    fn search_stats_fields_are_populated() {
        let path = temp_path("stats", "fsvi");
        let vectors: Vec<Vec<f32>> = (0..50).map(|i| normalized_vector(i, 32)).collect();
        let index = write_index(&path, &vectors).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");
        let (hits, stats) = ann
            .knn_search_with_stats(&normalized_vector(999, 32), 5, 64)
            .expect("search");
        assert_eq!(stats.index_size, 50);
        assert_eq!(stats.dimension, 32);
        assert_eq!(stats.ef_search, 64);
        assert_eq!(stats.k_requested, 5);
        assert_eq!(stats.k_returned, hits.len());
        assert!(stats.is_approximate);
        assert_eq!(stats.fallback_reason, None);
        assert!(stats.estimated_recall > 0.0);
        assert!(stats.estimated_recall <= 1.0);
    }

    #[test]
    fn underfilled_native_result_falls_back_to_exact_top_k() {
        let vectors = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0], vec![-1.0_f32, 0.0]];
        let source_path = temp_path("underfill-exact", "fsvi");
        let source = write_index(&source_path, &vectors).expect("source index");
        let ann =
            HnswIndex::build_from_vector_index(&source, HnswConfig::default()).expect("ANN index");
        let budget = dist_dot_budget(2).expect("distance budget");
        let canonical_query = vec![1.0_f32, 0.0];

        let (hits, stats) = ann
            .finish_search_with_neighbors(
                Some(&source),
                &canonical_query,
                2,
                2,
                HNSW_DEFAULT_EF_SEARCH,
                Vec::new(),
                budget,
                Instant::now(),
            )
            .expect("exact underfill fallback");

        // The injected native result is empty over a populated graph, which
        // classifies as the stronger empty-despite-points anomaly (bd-tqhc).
        assert_eq!(
            stats.fallback_reason,
            Some(AnnFallbackReason::EmptyDespiteIndexedPoints)
        );
        assert!(!stats.is_approximate);
        assert_eq!(stats.k_returned, 2);
        assert_eq!(stats.estimated_recall.to_bits(), 1.0_f64.to_bits());
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].doc_id, "doc-0000");
        assert_eq!(hits[1].doc_id, "doc-0001");
    }

    #[test]
    fn underfilled_native_result_without_source_fails_closed() {
        let source_path = temp_path("underfill-no-source", "fsvi");
        let source = write_index(&source_path, &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]])
            .expect("source index");
        let ann =
            HnswIndex::build_from_vector_index(&source, HnswConfig::default()).expect("ANN index");
        let budget = dist_dot_budget(2).expect("distance budget");
        let error = ann
            .finish_search_with_neighbors(
                None,
                &[1.0_f32, 0.0],
                2,
                2,
                HNSW_DEFAULT_EF_SEARCH,
                Vec::new(),
                budget,
                Instant::now(),
            )
            .expect_err("underfill without canonical source must not claim exact repair");
        assert!(
            error.to_string().contains("knn_search_with_stats_against"),
            "{error}"
        );
    }

    #[test]
    fn public_main_only_search_ignores_wal_consistently_before_and_after_underfill() {
        use crate::wal::WalEntry;

        let source_path = temp_path("underfill-public-main-only", "fsvi");
        let mut source = write_index(&source_path, &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]])
            .expect("source index");
        let ann =
            HnswIndex::build_from_vector_index(&source, HnswConfig::default()).expect("ANN index");
        source.wal_entries.push(WalEntry {
            doc_id: "doc-0000".into(),
            doc_id_hash: crate::fnv1a_hash(b"doc-0000"),
            embedding: vec![-1.0, 0.0],
        });
        let query = [1.0_f32, 0.0];

        let (normal, normal_stats) = ann
            .knn_search_with_stats_against(&source, &query, 1, HNSW_DEFAULT_EF_SEARCH)
            .expect("normal public main-only ANN");
        assert_eq!(normal_stats.fallback_reason, None);
        assert_eq!(normal.len(), 1);
        assert_eq!(normal[0].doc_id, "doc-0000");

        let budget = dist_dot_budget(2).expect("distance budget");
        let (fallback, fallback_stats) = ann
            .finish_search_with_neighbors(
                Some(&source),
                &query,
                1,
                1,
                HNSW_DEFAULT_EF_SEARCH,
                Vec::new(),
                budget,
                Instant::now(),
            )
            .expect("public main-only exact repair");
        assert_eq!(
            fallback_stats.fallback_reason,
            Some(AnnFallbackReason::EmptyDespiteIndexedPoints)
        );
        assert_eq!(fallback, normal);
    }

    #[test]
    fn bounded_underfill_scan_matches_flat_exact_top_k() {
        let vectors = (0..256)
            .map(|seed| normalized_vector(seed, 32))
            .collect::<Vec<_>>();
        let source_path = temp_path("underfill-flat-oracle", "fsvi");
        let source = write_index(&source_path, &vectors).expect("flat source");
        let ann =
            HnswIndex::build_from_vector_index(&source, HnswConfig::default()).expect("ANN index");
        let query = normalized_vector(10_000, 32);
        let exact = source.search_top_k(&query, 7, None).expect("flat exact");
        let budget = dist_dot_budget(32).expect("distance budget");
        let (fallback, stats) = ann
            .finish_search_with_neighbors(
                Some(&source),
                &query,
                7,
                7,
                HNSW_DEFAULT_EF_SEARCH,
                Vec::new(),
                budget,
                Instant::now(),
            )
            .expect("bounded exact fallback");

        assert_eq!(
            stats.fallback_reason,
            Some(AnnFallbackReason::EmptyDespiteIndexedPoints)
        );
        assert_eq!(fallback, exact);
    }

    #[test]
    fn f16_non_unit_underfill_fallback_is_bit_exact_with_canonical_scan() {
        let vectors = vec![
            vec![8.0_f32, 0.25, -0.5],
            vec![0.5_f32, 0.5, 0.5],
            vec![1.25_f32, 2.5, -0.75],
            vec![-3.0_f32, 0.125, 4.0],
        ];
        let source_path = temp_path("underfill-f16-non-unit", "fsvi");
        let source = write_index_with_quantization(&source_path, &vectors, Quantization::F16)
            .expect("F16 source");
        let ann =
            HnswIndex::build_from_vector_index(&source, HnswConfig::default()).expect("ANN index");
        let canonical_query = vec![2.0_f32, -0.5, 0.25];
        let exact = source
            .search_main_top_k(&canonical_query, 3)
            .expect("canonical persisted scan");
        let budget = dist_dot_budget(3).expect("distance budget");
        let (fallback, stats) = ann
            .finish_search_with_neighbors(
                Some(&source),
                &canonical_query,
                3,
                3,
                HNSW_DEFAULT_EF_SEARCH,
                Vec::new(),
                budget,
                Instant::now(),
            )
            .expect("F16 canonical fallback");

        assert_eq!(
            stats.fallback_reason,
            Some(AnnFallbackReason::EmptyDespiteIndexedPoints)
        );
        assert_eq!(fallback, exact);
        assert!(
            fallback
                .iter()
                .zip(&exact)
                .all(|(actual, expected)| actual.score.to_bits() == expected.score.to_bits())
        );

        let metadata_path = temp_path("underfill-f16-native", "hnsw");
        ann.save(&metadata_path).expect("save native graph");
        let (loaded, disposition) =
            HnswIndex::load_with_disposition(&metadata_path, &source).expect("load native graph");
        assert_eq!(disposition, HnswLoadDisposition::Native);
        let (native_fallback, native_stats) = loaded
            .finish_search_with_neighbors(
                Some(&source),
                &canonical_query,
                3,
                3,
                HNSW_DEFAULT_EF_SEARCH,
                Vec::new(),
                budget,
                Instant::now(),
            )
            .expect("native F16 canonical fallback");
        assert_eq!(
            native_stats.fallback_reason,
            Some(AnnFallbackReason::EmptyDespiteIndexedPoints)
        );
        assert_eq!(native_fallback, exact);
    }

    #[test]
    fn underfill_fallback_excludes_wal_for_single_merge_in_two_tier() {
        let vectors = vec![vec![0.5_f32, 0.0], vec![0.25_f32, 0.0], vec![-0.5_f32, 0.0]];
        let source_path = temp_path("underfill-wal", "fsvi");
        let mut source = write_index(&source_path, &vectors).expect("source");
        let ann =
            HnswIndex::build_from_vector_index(&source, HnswConfig::default()).expect("ANN index");
        source
            .append("wal-top", &[1.0_f32, 0.0])
            .expect("append resident WAL entry");
        let canonical_query = vec![1.0_f32, 0.0];
        let expected_main = source
            .search_main_top_k(&canonical_query, 2)
            .expect("persisted-only oracle");
        let full = source
            .search_top_k(&canonical_query, 2, None)
            .expect("main plus WAL oracle");
        assert_eq!(full[0].doc_id, "wal-top");
        let budget = dist_dot_budget(2).expect("distance budget");
        let (fallback, stats) = ann
            .finish_search_with_neighbors(
                Some(&source),
                &canonical_query,
                2,
                2,
                HNSW_DEFAULT_EF_SEARCH,
                Vec::new(),
                budget,
                Instant::now(),
            )
            .expect("main-only fallback");

        assert_eq!(
            stats.fallback_reason,
            Some(AnnFallbackReason::EmptyDespiteIndexedPoints)
        );
        assert_eq!(fallback, expected_main);
        assert!(
            fallback.iter().all(|hit| hit.doc_id != "wal-top"),
            "resident WAL must be merged exactly once by TwoTierIndex"
        );
    }

    #[test]
    fn duplicate_doc_ids_and_tombstone_gaps_keep_distinct_physical_rows() {
        let source_path = temp_path("duplicate-id-gap", "fsvi");
        let mut writer =
            VectorIndex::create_with_revision(&source_path, "hash", "test", 2, Quantization::F32)
                .expect("writer");
        writer
            .write_record("duplicate", &[1.0_f32, 0.0])
            .expect("first duplicate");
        writer.write_record("gap", &[0.0_f32, 1.0]).expect("gap");
        writer
            .write_record("duplicate", &[-1.0_f32, 0.0])
            .expect("second duplicate");
        writer.finish().expect("finish source");
        let mut source = VectorIndex::open(&source_path).expect("open source");
        assert!(source.soft_delete("gap").expect("tombstone gap"));
        let ann =
            HnswIndex::build_from_vector_index(&source, HnswConfig::default()).expect("ANN index");
        let expected_duplicate_positions = (0..source.record_count())
            .filter(|&position| {
                !source.is_deleted(position)
                    && source.doc_id_at(position).expect("source doc ID") == "duplicate"
            })
            .map(|position| u32::try_from(position).expect("test position fits u32"))
            .collect::<Vec<_>>();
        assert_eq!(ann.source_positions, expected_duplicate_positions);
        assert_eq!(ann.source_record_count, 3);
        let mut selected_positions = Vec::new();
        for query in [[1.0_f32, 0.0], [-1.0_f32, 0.0]] {
            let expected = source
                .search_main_top_k(&query, 1)
                .expect("canonical physical row");
            assert_eq!(expected.len(), 1);
            assert_eq!(expected[0].doc_id, "duplicate");

            let (ann_hits, ann_stats) = ann
                .knn_search_with_stats_against(&source, &query, 1, HNSW_DEFAULT_EF_SEARCH)
                .expect("normal ANN search");
            assert_eq!(ann_stats.fallback_reason, None);
            assert_eq!(
                ann_hits, expected,
                "ANN must preserve the winning duplicate's physical identity"
            );

            let budget = dist_dot_budget(2).expect("distance budget");
            let (fallback, fallback_stats) = ann
                .finish_search_with_neighbors(
                    Some(&source),
                    &query,
                    1,
                    1,
                    HNSW_DEFAULT_EF_SEARCH,
                    Vec::new(),
                    budget,
                    Instant::now(),
                )
                .expect("canonical fallback");
            assert_eq!(
                fallback_stats.fallback_reason,
                Some(AnnFallbackReason::EmptyDespiteIndexedPoints)
            );
            assert_eq!(fallback, expected);
            selected_positions.push(expected[0].index);
        }
        assert_ne!(
            selected_positions[0], selected_positions[1],
            "opposing queries must retain the two duplicate-ID physical rows instead of \
             remapping both through the first matching document ID"
        );

        let expected = source
            .search_main_top_k(&[1.0_f32, 0.0], 2)
            .expect("canonical duplicate-ID semantics");
        assert_eq!(
            expected.len(),
            1,
            "canonical search selects two physical winners, then deduplicates public doc IDs"
        );
        let (raw_hits, raw_stats) = ann
            .knn_search_raw_with_stats_against(&source, &[1.0_f32, 0.0], 2, HNSW_DEFAULT_EF_SEARCH)
            .expect("raw duplicate-ID ANN search");
        assert_eq!(raw_stats.fallback_reason, None);
        assert_eq!(raw_hits.len(), 2);
        assert!(
            raw_hits.iter().all(|hit| hit.doc_id == "duplicate"),
            "raw TwoTier candidates must retain both physical duplicate-ID winners"
        );
        assert_ne!(
            raw_hits[0].index, raw_hits[1].index,
            "raw candidates must preserve distinct physical rows"
        );
        let (ann_hits, ann_stats) = ann
            .knn_search_with_stats_against(&source, &[1.0_f32, 0.0], 2, HNSW_DEFAULT_EF_SEARCH)
            .expect("duplicate-ID ANN search");
        assert_eq!(ann_stats.fallback_reason, None);
        assert_eq!(
            ann_hits, expected,
            "ANN must preserve physical identity internally while matching canonical \
             duplicate-ID result semantics"
        );
    }

    #[test]
    fn post_build_tombstone_is_filtered_and_underfill_uses_current_exact_source() {
        let source_path = temp_path("post-build-tombstone", "fsvi");
        let mut source = write_index(
            &source_path,
            &[
                vec![1.0_f32, 0.0, 0.0],
                vec![0.0_f32, 1.0, 0.0],
                vec![0.0_f32, 0.0, 1.0],
            ],
        )
        .expect("source");
        let ann = HnswIndex::build_from_vector_index(&source, HnswConfig::default()).expect("ANN");
        assert!(source.soft_delete("doc-0000").expect("post-build delete"));
        let query = [1.0_f32, 0.0, 0.0];
        let expected = source
            .search_main_top_k(&query, 3)
            .expect("current exact source");
        assert_eq!(expected.len(), 2);
        assert!(expected.iter().all(|hit| hit.doc_id != "doc-0000"));

        let (hits, stats) = ann
            .knn_search_with_stats_against(&source, &query, 3, HNSW_DEFAULT_EF_SEARCH)
            .expect("tombstone-aware ANN search");
        assert_eq!(stats.fallback_reason, Some(AnnFallbackReason::Underfilled));
        assert_eq!(hits, expected);

        let budget = dist_dot_budget(3).expect("distance budget");
        let (forced, forced_stats) = ann
            .finish_search_with_neighbors(
                Some(&source),
                &query,
                3,
                3,
                HNSW_DEFAULT_EF_SEARCH,
                Vec::new(),
                budget,
                Instant::now(),
            )
            .expect("forced exact repair after tombstone");
        assert_eq!(
            forced_stats.fallback_reason,
            Some(AnnFallbackReason::EmptyDespiteIndexedPoints)
        );
        assert_eq!(forced, expected);
    }

    #[test]
    fn borrowed_source_fallback_survives_path_rename_after_open() {
        let source_path = temp_path("borrowed-source-before-rename", "fsvi");
        let renamed_path = temp_path("borrowed-source-after-rename", "fsvi");
        let source =
            write_index(&source_path, &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]]).expect("source");
        let ann =
            HnswIndex::build_from_vector_index(&source, HnswConfig::default()).expect("ANN index");
        std::fs::rename(&source_path, &renamed_path).expect("rename open source file");
        let expected = source
            .search_main_top_k(&[1.0_f32, 0.0], 2)
            .expect("open mmap remains authoritative after rename");
        let budget = dist_dot_budget(2).expect("distance budget");
        let (fallback, stats) = ann
            .finish_search_with_neighbors(
                Some(&source),
                &[1.0_f32, 0.0],
                2,
                2,
                HNSW_DEFAULT_EF_SEARCH,
                Vec::new(),
                budget,
                Instant::now(),
            )
            .expect("borrowed-source fallback after rename");
        assert_eq!(
            stats.fallback_reason,
            Some(AnnFallbackReason::EmptyDespiteIndexedPoints)
        );
        assert_eq!(fallback, expected);
    }

    #[test]
    fn topology_validator_accepts_valid_layers_and_rejects_captured_wrong_layer() {
        let point_zero_id = PointId(0, 0);
        let point_one_id = PointId(1, 0);
        let valid = vec![
            HnswTopologyPoint {
                origin_id: 0,
                point_id: point_zero_id,
                neighborhoods: vec![vec![Neighbour::new(1, 1.0, point_one_id)], Vec::new()],
            },
            HnswTopologyPoint {
                origin_id: 1,
                point_id: point_one_id,
                neighborhoods: vec![vec![Neighbour::new(0, 1.0, point_zero_id)], Vec::new()],
            },
        ];
        validate_hnsw_topology_observations(&valid, 2, Some((1, point_one_id)))
            .expect("valid topology");

        let mut wrong_layer = valid;
        wrong_layer[0].neighborhoods[1].push(Neighbour::new(1, 1.0, point_one_id));
        let detail = validate_hnsw_topology_observations(&wrong_layer, 2, Some((1, point_one_id)))
            .expect_err("neighbor above source sampled layer must fail");
        assert!(detail.contains("above sampled layer"), "{detail}");
    }

    #[test]
    fn topology_validator_rejects_duplicate_and_out_of_range_origin_ids() {
        let duplicate = vec![
            HnswTopologyPoint {
                origin_id: 0,
                point_id: PointId(0, 0),
                neighborhoods: vec![Vec::new()],
            },
            HnswTopologyPoint {
                origin_id: 0,
                point_id: PointId(0, 1),
                neighborhoods: vec![Vec::new()],
            },
        ];
        let detail = validate_hnsw_topology_observations(&duplicate, 2, Some((0, PointId(0, 0))))
            .expect_err("duplicate origin must fail");
        assert!(detail.contains("duplicate origin id"), "{detail}");

        let out_of_range = vec![HnswTopologyPoint {
            origin_id: 1,
            point_id: PointId(0, 0),
            neighborhoods: vec![Vec::new()],
        }];
        let detail =
            validate_hnsw_topology_observations(&out_of_range, 1, Some((1, PointId(0, 0))))
                .expect_err("out-of-range origin must fail");
        assert!(detail.contains("outside 0..1"), "{detail}");
    }

    #[test]
    fn topology_validator_rejects_edgeless_multi_point_graph() {
        let edgeless = vec![
            HnswTopologyPoint {
                origin_id: 0,
                point_id: PointId(0, 0),
                neighborhoods: vec![Vec::new()],
            },
            HnswTopologyPoint {
                origin_id: 1,
                point_id: PointId(0, 1),
                neighborhoods: vec![Vec::new()],
            },
        ];
        let detail = validate_hnsw_topology_observations(&edgeless, 2, Some((0, PointId(0, 0))))
            .expect_err("edgeless multi-point graph must fail");
        assert!(detail.contains("weak components"), "{detail}");
        assert!(detail.contains("only 1/2"), "{detail}");
        assert!(detail.contains("2 weak components total"), "{detail}");
        assert!(
            detail.contains("2 points have zero base out-edges (first: origin 0)"),
            "{detail}"
        );
    }

    /// #32: bounded neighbour pruning can strip every in-edge from a point,
    /// leaving a directed "source" that no walk from the entry can visit even
    /// though the graph is one weak component. Real serial and parallel builds
    /// of large corpora produce exactly this shape, so attestation must admit
    /// it (coverage is reported separately) instead of rejecting the build.
    #[test]
    fn topology_validator_admits_directed_source_points_in_one_weak_component() {
        let ids = [PointId(0, 0), PointId(0, 1), PointId(0, 2)];
        let points = vec![
            // Entry keeps no out-edges at the base layer; every other point
            // kept only its out-edge toward the entry (in-edges pruned away).
            HnswTopologyPoint {
                origin_id: 0,
                point_id: ids[0],
                neighborhoods: vec![Vec::new()],
            },
            HnswTopologyPoint {
                origin_id: 1,
                point_id: ids[1],
                neighborhoods: vec![vec![Neighbour::new(0, 1.0, ids[0])]],
            },
            HnswTopologyPoint {
                origin_id: 2,
                point_id: ids[2],
                neighborhoods: vec![vec![Neighbour::new(0, 1.0, ids[0])]],
            },
        ];

        validate_hnsw_topology_observations(&points, 3, Some((0, ids[0])))
            .expect("weakly connected graph with directed source points must be admitted");
    }

    #[test]
    fn directed_base_coverage_reports_unreached_points_without_failing() {
        // Same shape as the admission test above: entry 0 has no out-edges.
        let adjacency: Vec<Vec<usize>> = vec![Vec::new(), vec![0], vec![0]];
        let coverage =
            measure_directed_base_coverage(0, 3, |origin_id| Ok(adjacency[origin_id].clone()))
                .expect("coverage walk must succeed");
        assert_eq!(coverage.reached_count, 1);
        assert_eq!(coverage.first_unreachable, Some(1));

        // A directed cycle covers everything.
        let cycle: Vec<Vec<usize>> = vec![vec![1], vec![2], vec![0]];
        let coverage =
            measure_directed_base_coverage(0, 3, |origin_id| Ok(cycle[origin_id].clone()))
                .expect("coverage walk must succeed");
        assert_eq!(coverage.reached_count, 3);
        assert_eq!(coverage.first_unreachable, None);
    }

    #[test]
    fn weak_connectivity_rejects_out_of_range_neighbor() {
        let adjacency: Vec<Vec<usize>> = vec![vec![1], vec![7]];
        let detail =
            validate_weak_base_connectivity(0, 2, |origin_id| Ok(adjacency[origin_id].clone()))
                .expect_err("out-of-range neighbor must fail");
        assert!(detail.contains("out-of-range base neighbor 7"), "{detail}");
    }

    #[test]
    fn topology_validator_rejects_edge_above_target_sampled_layer() {
        let low_id = PointId(0, 0);
        let high_id = PointId(1, 0);
        let points = vec![
            HnswTopologyPoint {
                origin_id: 0,
                point_id: low_id,
                neighborhoods: vec![vec![Neighbour::new(1, 1.0, high_id)]],
            },
            HnswTopologyPoint {
                origin_id: 1,
                point_id: high_id,
                neighborhoods: vec![
                    vec![Neighbour::new(0, 1.0, low_id)],
                    vec![Neighbour::new(0, 1.0, low_id)],
                ],
            },
        ];
        let detail = validate_hnsw_topology_observations(&points, 2, Some((1, high_id)))
            .expect_err("edge above target sampled layer must fail");
        assert!(detail.contains("neighbor's sampled layer"), "{detail}");
    }

    #[test]
    fn topology_validator_rejects_disconnected_nonempty_components() {
        let ids = [PointId(0, 0), PointId(0, 1), PointId(0, 2), PointId(0, 3)];
        let points = vec![
            HnswTopologyPoint {
                origin_id: 0,
                point_id: ids[0],
                neighborhoods: vec![vec![Neighbour::new(1, 1.0, ids[1])]],
            },
            HnswTopologyPoint {
                origin_id: 1,
                point_id: ids[1],
                neighborhoods: vec![vec![Neighbour::new(0, 1.0, ids[0])]],
            },
            HnswTopologyPoint {
                origin_id: 2,
                point_id: ids[2],
                neighborhoods: vec![vec![Neighbour::new(3, 1.0, ids[3])]],
            },
            HnswTopologyPoint {
                origin_id: 3,
                point_id: ids[3],
                neighborhoods: vec![vec![Neighbour::new(2, 1.0, ids[2])]],
            },
        ];

        let detail = validate_hnsw_topology_observations(&points, 4, Some((0, ids[0])))
            .expect_err("two locally valid components must fail weak connectivity");
        assert!(detail.contains("only 2/4"), "{detail}");
        assert!(detail.contains("first detached origin is 2"), "{detail}");
        assert!(detail.contains("2 weak components total"), "{detail}");
        assert!(
            detail.contains("0 points have zero base out-edges (first: none)"),
            "{detail}"
        );
    }

    /// Pin the weak-component union-find at the exact cardinality of the
    /// largest real corpus that has exercised this gate (CASS's published
    /// quality tier). A forward-attached insertion chain — each non-seed point
    /// holding one base edge to its predecessor, the invariant `insert_slice`
    /// guarantees — must resolve to a single weak component with no overflow
    /// or path-compression fault at this scale.
    #[test]
    fn weak_connectivity_attests_single_chain_at_production_cardinality() {
        const PRODUCTION_POINT_COUNT: usize = 2_573_003;

        validate_weak_base_connectivity(0, PRODUCTION_POINT_COUNT, |origin_id| {
            Ok(if origin_id == 0 {
                Vec::new()
            } else {
                vec![origin_id - 1]
            })
        })
        .expect("a forward-attached insertion chain is one weak component");
    }

    #[test]
    fn topology_validator_rejects_mismatched_entry_identity() {
        let id = PointId(0, 0);
        let points = vec![HnswTopologyPoint {
            origin_id: 0,
            point_id: id,
            neighborhoods: vec![Vec::new()],
        }];

        let detail = validate_hnsw_topology_observations(&points, 1, Some((1, id)))
            .expect_err("entry origin must agree with its internal point identity");
        assert!(detail.contains("not advertised origin 1"), "{detail}");
    }

    #[test]
    fn calibration_rejects_fallback_tainted_exact_results() {
        let exact = vec![VectorHit {
            index: 0,
            score: 1.0,
            doc_id: "doc".into(),
        }];
        let stats = AnnSearchStats {
            index_size: 1,
            dimension: 2,
            ef_search: 1,
            k_requested: 1,
            k_returned: 1,
            search_time_us: 10,
            is_approximate: false,
            fallback_reason: Some(AnnFallbackReason::Underfilled),
            estimated_recall: 1.0,
            zero_signal: None,
        };
        assert_eq!(
            certified_ann_recall_sample(Ok((exact.clone(), stats)), &exact).to_bits(),
            0.0_f64.to_bits()
        );
    }

    #[test]
    fn search_k_larger_than_index_returns_all() {
        let path = temp_path("klarge", "fsvi");
        let vectors: Vec<Vec<f32>> = (0..5).map(|i| normalized_vector(i, 16)).collect();
        let index = write_index(&path, &vectors).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");
        let hits = ann
            .knn_search(&normalized_vector(999, 16), 100, HNSW_DEFAULT_EF_SEARCH)
            .expect("search");
        assert_eq!(hits.len(), 5);
    }

    // ── matches_vector_index edge cases ─────────────────────────────────

    #[test]
    fn matches_returns_false_for_dimension_mismatch() {
        let path_a = temp_path("match-a", "fsvi");
        let path_b = temp_path("match-b", "fsvi");
        let index_a = write_index(&path_a, &[normalized_vector(1, 16)]).expect("index_a");
        let index_b = write_index(&path_b, &[normalized_vector(1, 32)]).expect("index_b");
        let ann = HnswIndex::build_from_vector_index(&index_a, HnswConfig::default()).expect("ann");
        assert!(!ann.matches_vector_index(&index_b).expect("matches"));
    }

    #[test]
    fn matches_returns_false_for_record_count_mismatch() {
        let path_a = temp_path("match-rc-a", "fsvi");
        let path_b = temp_path("match-rc-b", "fsvi");
        let index_a = write_index(
            &path_a,
            &[normalized_vector(1, 16), normalized_vector(2, 16)],
        )
        .expect("index_a");
        let index_b = write_index(&path_b, &[normalized_vector(1, 16)]).expect("index_b");
        let ann = HnswIndex::build_from_vector_index(&index_a, HnswConfig::default()).expect("ann");
        assert!(!ann.matches_vector_index(&index_b).expect("matches"));
    }

    #[test]
    fn matches_returns_false_when_vectors_change_but_doc_ids_match() {
        // Matching document IDs and dimensions are insufficient: serving a
        // graph built from different vectors would silently return stale ANN
        // results. The full live-vector fingerprint must reject that source.
        let path_a = temp_path("match-vec-a", "fsvi");
        let path_b = temp_path("match-vec-b", "fsvi");
        let index_a = write_index(
            &path_a,
            &[normalized_vector(1, 16), normalized_vector(2, 16)],
        )
        .expect("index_a");
        let index_b = write_index(
            &path_b,
            &[normalized_vector(3, 16), normalized_vector(4, 16)],
        )
        .expect("index_b");
        let ann = HnswIndex::build_from_vector_index(&index_a, HnswConfig::default()).expect("ann");
        assert!(!ann.matches_vector_index(&index_b).expect("matches"));
    }

    #[test]
    fn build_from_vector_index_excludes_tombstoned_records() {
        let path = temp_path("tombstone-filter", "fsvi");
        let mut index = write_index(
            &path,
            &[
                normalized_vector(1, 16),
                normalized_vector(2, 16),
                normalized_vector(3, 16),
            ],
        )
        .expect("index");
        let deleted = index
            .soft_delete("doc-0001")
            .expect("soft_delete should succeed");
        assert!(deleted);

        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");
        assert_eq!(ann.len(), 2, "ANN should only index live vectors");

        let query = normalized_vector(2, 16);
        let hits = ann
            .knn_search(&query, 10, HNSW_DEFAULT_EF_SEARCH)
            .expect("search");
        assert!(
            !hits.iter().any(|hit| hit.doc_id == "doc-0001"),
            "ANN should never return tombstoned doc IDs"
        );
    }

    // ── normalize_for_dist_dot ──────────────────────────────────────────

    #[test]
    fn normalize_zero_vector_unchanged() {
        let zero = vec![0.0_f32; 8];
        let budget = dist_dot_budget(zero.len()).expect("budget");
        let result = normalize_for_dist_dot(zero.clone(), budget);
        assert_eq!(
            result, zero,
            "zero vector should remain zero after normalize"
        );
    }

    #[test]
    fn dist_dot_roundoff_budget_is_dimension_aware_and_ranking_neutral() {
        let radius_16 = dist_dot_budget(16).expect("budget").radius_squared;
        let radius_384 = dist_dot_budget(384).expect("budget").radius_squared;
        let radius_4096 = dist_dot_budget(4096).expect("budget").radius_squared;

        assert!((0.5..1.0).contains(&radius_16));
        assert!(radius_384 < radius_16);
        assert!(radius_4096 < radius_384);
        assert!(
            radius_384 > 0.999,
            "384-dim safety margin must stay small enough to preserve cosine resolution"
        );

        let error = dist_dot_budget(DIST_DOT_MAX_DIMENSION + 1)
            .expect_err("ill-conditioned f32 bound must fail closed");
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "dimension")
        );
    }

    #[test]
    fn dist_dot_normalization_stays_below_one_across_reduction_orders() {
        fn reassociated_dot(left: &[f32], right: &[f32], lanes: usize) -> f32 {
            let mut partials = vec![0.0_f32; lanes];
            for (index, (&left, &right)) in left.iter().zip(right).enumerate() {
                partials[index % lanes] += left * right;
            }
            partials.into_iter().fold(0.0_f32, |sum, value| sum + value)
        }

        for dimension in [16_usize, 384, 4_096] {
            let budget = dist_dot_budget(dimension).expect("budget");
            let vectors: Vec<Vec<f32>> = (0..16)
                .map(|seed| normalize_for_dist_dot(normalized_vector(seed, dimension), budget))
                .collect();
            for left in &vectors {
                for right in &vectors {
                    for lanes in [1_usize, 2, 4, 8, 16] {
                        let dot = reassociated_dot(left, right, lanes);
                        assert!(
                            dot <= 1.0,
                            "DistDot precondition violated: dim={dimension} lanes={lanes} dot={dot}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn safe_radius_scaling_preserves_valid_cosine_ranking() {
        fn dot(left: &[f32], right: &[f32]) -> f32 {
            left.iter().zip(right).map(|(a, b)| a * b).sum()
        }

        let query = vec![1.0_f32, 0.0, 0.0, 0.0];
        let candidates = [
            vec![1.0_f32, 0.0, 0.0, 0.0],
            vec![0.8_f32, 0.6, 0.0, 0.0],
            vec![0.6_f32, 0.8, 0.0, 0.0],
            vec![-0.2_f32, 0.0, 0.0, 0.98],
        ];
        let budget = dist_dot_budget(query.len()).expect("budget");
        let scaled_query = normalize_for_dist_dot(query.clone(), budget);

        let mut original: Vec<(usize, f32)> = candidates
            .iter()
            .enumerate()
            .map(|(index, candidate)| {
                let candidate_norm = dot(candidate, candidate).sqrt();
                (index, dot(&query, candidate) / candidate_norm)
            })
            .collect();
        let mut restored: Vec<(usize, f32)> = candidates
            .into_iter()
            .enumerate()
            .map(|(index, candidate)| {
                let scaled = normalize_for_dist_dot(candidate, budget);
                (index, dot(&scaled_query, &scaled) / budget.radius_squared)
            })
            .collect();
        original.sort_by(|left, right| right.1.total_cmp(&left.1));
        restored.sort_by(|left, right| right.1.total_cmp(&left.1));

        assert_eq!(
            original.iter().map(|entry| entry.0).collect::<Vec<_>>(),
            restored.iter().map(|entry| entry.0).collect::<Vec<_>>(),
            "uniform safety scaling must not change valid cosine ranking"
        );
        for ((_, expected), (_, actual)) in original.iter().zip(&restored) {
            assert!((expected - actual).abs() <= budget.score_tolerance);
        }
    }

    #[test]
    fn hnsw_restores_cosine_score_after_safe_radius_scaling() {
        let path = temp_path("safe-radius-score", "fsvi");
        let vector = normalized_vector(41, 384);
        let index = write_index(&path, std::slice::from_ref(&vector)).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");

        let hits = ann
            .knn_search(&vector, 1, HNSW_DEFAULT_EF_SEARCH)
            .expect("search");
        assert_eq!(hits.len(), 1);
        assert!(
            (hits[0].score - 1.0).abs() <= 1.0e-5,
            "uniform DistDot safety scaling must not leak into public cosine scores: {}",
            hits[0].score
        );
    }

    #[test]
    fn hnsw_rejects_non_finite_query_before_distance_evaluation() {
        let path = temp_path("non-finite-query", "fsvi");
        let vector = normalized_vector(42, 32);
        let index = write_index(&path, std::slice::from_ref(&vector)).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");
        let mut query = vector;
        query[7] = f32::NAN;

        let error = ann
            .knn_search(&query, 1, HNSW_DEFAULT_EF_SEARCH)
            .expect_err("non-finite queries must return an error rather than panic in DistDot");
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, ref reason, .. }
                if field == "query" && reason.contains("finite")),
            "expected query InvalidConfig, got {error:?}"
        );
    }

    // ── estimate_recall ─────────────────────────────────────────────────

    #[test]
    fn estimate_recall_k_zero_returns_one() {
        assert!((estimate_recall(100, 0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn estimate_recall_clamped_between_zero_and_one() {
        // Very low ef relative to k
        let low = estimate_recall(1, 1000);
        assert!((0.0..=1.0).contains(&low), "low recall: {low}");

        // Very high ef relative to k
        let high = estimate_recall(10_000, 1);
        assert!((0.0..=1.0).contains(&high), "high recall: {high}");
    }

    #[test]
    fn estimate_recall_increases_with_ef() {
        let r_low = estimate_recall(10, 10);
        let r_high = estimate_recall(100, 10);
        assert!(
            r_high >= r_low,
            "recall should increase with ef: {r_low} vs {r_high}"
        );
    }

    // ── len / is_empty / dimension / config accessors ───────────────────

    #[test]
    fn accessors_report_correct_values() {
        let path = temp_path("accessors", "fsvi");
        let vectors: Vec<Vec<f32>> = (0..10).map(|i| normalized_vector(i, 24)).collect();
        let index = write_index(&path, &vectors).expect("index");
        let config = HnswConfig {
            m: 8,
            ..HnswConfig::default()
        };
        let ann = HnswIndex::build_from_vector_index(&index, config).expect("ann");
        assert_eq!(ann.len(), 10);
        assert!(!ann.is_empty());
        assert_eq!(ann.dimension(), 24);
        assert_eq!(ann.config().m, 8);
    }

    // ── Debug impl ──────────────────────────────────────────────────────

    #[test]
    fn debug_impl_does_not_panic() {
        let path = temp_path("debug", "fsvi");
        let index = write_index(&path, &[normalized_vector(1, 8)]).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");
        let debug_str = format!("{ann:?}");
        assert!(debug_str.contains("HnswIndex"));
        assert!(debug_str.contains("dimension: 8"));
    }

    // ── Original tests ──────────────────────────────────────────────────

    #[test]
    fn scores_are_consistent_with_exact_top_hit() {
        let fsvi_path = temp_path("score", "fsvi");
        let vectors: Vec<Vec<f32>> = (0..128).map(|i| normalized_vector(i, 64)).collect();
        let index = write_index(&fsvi_path, &vectors).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");

        let query = normalized_vector(7, 64);
        let exact = index.search_top_k(&query, 1, None).expect("exact");
        let approx = ann
            .knn_search(&query, 1, HNSW_DEFAULT_EF_SEARCH)
            .expect("approx");

        assert_eq!(exact[0].doc_id, approx[0].doc_id);
        assert!((exact[0].score - approx[0].score).abs() < 1e-3);
    }

    #[test]
    fn persistence_round_trip() {
        let fsvi_path = temp_path("persist", "fsvi");
        let vectors: Vec<Vec<f32>> = (0..64).map(|i| normalized_vector(i, 32)).collect();
        let index = write_index(&fsvi_path, &vectors).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");

        let save_path = temp_path("persist", "hnsw");
        ann.save(&save_path).expect("save");

        let (loaded, disposition) =
            HnswIndex::load_with_disposition(&save_path, &index).expect("load");
        assert_eq!(disposition, HnswLoadDisposition::Native);
        assert_eq!(loaded.len(), 64);
        assert_eq!(loaded.dimension(), 32);

        let query = normalized_vector(10, 32);
        let hits = loaded
            .knn_search(&query, 5, HNSW_DEFAULT_EF_SEARCH)
            .expect("search");
        assert_eq!(hits[0].doc_id, "doc-0010");
        assert!((hits[0].score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn public_strict_load_reports_native_without_mutating_selected_artifacts() {
        let root = temp_path("public-strict-native", "dir");
        std::fs::create_dir_all(&root).expect("create strict native fixture");
        let fsvi_path = root.join("selected-source.fsvi");
        let vectors: Vec<Vec<f32>> = (0..32).map(|i| normalized_vector(i, 16)).collect();
        let source = write_index(&fsvi_path, &vectors).expect("write source FSVI");
        let ann =
            HnswIndex::build_from_vector_index(&source, HnswConfig::default()).expect("build ANN");
        let metadata_path = root.join("selected-ann.hnsw");
        ann.save(&metadata_path).expect("save selected native ANN");
        let before = snapshot_immutable_tree(&root).expect("snapshot selected artifacts");

        let native_only = HnswIndex::try_load_native(&metadata_path, &source)
            .expect("inspect native graph")
            .expect("selected graph must be native");
        let (loaded, disposition) =
            HnswIndex::load_with_disposition(&metadata_path, &source).expect("strict native load");

        assert_eq!(disposition, HnswLoadDisposition::Native);
        assert_eq!(native_only.len(), source.live_count());
        assert_eq!(loaded.len(), source.live_count());
        assert_eq!(
            snapshot_immutable_tree(&root).expect("snapshot after strict native load"),
            before,
            "native inspection must preserve bytes, mtimes, permissions, and directory inventory"
        );
    }

    #[test]
    fn public_strict_load_reports_rebuilt_for_stale_source_without_mutation() {
        let root = temp_path("public-strict-stale", "dir");
        std::fs::create_dir_all(&root).expect("create strict stale fixture");
        let original_path = root.join("original-source.fsvi");
        let original_vectors: Vec<Vec<f32>> = (0..24).map(|i| normalized_vector(i, 12)).collect();
        let original = write_index(&original_path, &original_vectors).expect("original FSVI");
        let ann = HnswIndex::build_from_vector_index(&original, HnswConfig::default())
            .expect("build original ANN");
        let metadata_path = root.join("selected-ann.hnsw");
        ann.save(&metadata_path).expect("save selected ANN");

        let replacement_path = root.join("replacement-source.fsvi");
        let replacement_vectors: Vec<Vec<f32>> =
            (10_000..10_024).map(|i| normalized_vector(i, 12)).collect();
        let replacement =
            write_index(&replacement_path, &replacement_vectors).expect("replacement FSVI");
        let before = snapshot_immutable_tree(&root).expect("snapshot stale fixture");

        assert!(
            HnswIndex::try_load_native(&metadata_path, &replacement)
                .expect("inspect stale sidecar")
                .is_none(),
            "native-only inspection must reject stale vectors without rebuilding"
        );
        let (loaded, disposition) = HnswIndex::load_with_disposition(&metadata_path, &replacement)
            .expect("stale sidecar should rebuild in memory");

        assert_eq!(disposition, HnswLoadDisposition::Rebuilt);
        assert_eq!(loaded.len(), replacement.live_count());
        assert!(
            loaded
                .matches_vector_index(&replacement)
                .expect("rebuilt graph matches replacement")
        );
        assert_eq!(
            snapshot_immutable_tree(&root).expect("snapshot after stale load"),
            before,
            "stale detection and in-memory rebuild must preserve every selected artifact"
        );
    }

    #[test]
    fn save_retry_reuses_ready_generation_after_pre_publish_error() {
        let source_path = temp_path("ready-pre-publish-source", "fsvi");
        let source_index = write_index(&source_path, &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]])
            .expect("source index");
        let ann = HnswIndex::build_from_vector_index(&source_index, HnswConfig::default())
            .expect("ANN index");
        let metadata_path = temp_path("ready-pre-publish", "hnsw");

        let error = ann
            .save_with_metadata_publisher(&metadata_path, reject_hnsw_metadata_publish)
            .expect_err("injected publication failure");
        assert!(matches!(&error, SearchError::Io(_)));
        assert!(
            !metadata_path.exists(),
            "pre-publication failure must leave metadata absent"
        );
        let before = ready_generation_paths(&metadata_path, ann.vector_fingerprint);
        assert_eq!(before.len(), 1, "one complete READY generation retained");
        let retained_generation = before[0]
            .file_name()
            .and_then(|name| name.to_str())
            .expect("retained generation name")
            .to_owned();

        ann.save(&metadata_path).expect("retry READY publication");
        let after = ready_generation_paths(&metadata_path, ann.vector_fingerprint);
        assert_eq!(
            after, before,
            "retry must reuse the complete generation instead of dumping another"
        );
        let metadata: HnswMeta =
            serde_json::from_slice(&std::fs::read(&metadata_path).expect("read metadata"))
                .expect("parse metadata");
        assert_eq!(
            metadata.sidecar_generation.as_deref(),
            Some(retained_generation.as_str())
        );
        let (_, disposition) = HnswIndex::load_with_disposition(&metadata_path, &source_index)
            .expect("native load after retry");
        assert_eq!(disposition, HnswLoadDisposition::Native);
    }

    #[test]
    fn save_retry_after_metadata_rename_sync_uncertainty_reuses_generation() {
        let source_path = temp_path("ready-post-rename-source", "fsvi");
        let source_index = write_index(&source_path, &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]])
            .expect("source index");
        let ann = HnswIndex::build_from_vector_index(&source_index, HnswConfig::default())
            .expect("ANN index");
        let metadata_path = temp_path("ready-post-rename", "hnsw");

        let error = ann
            .save_with_metadata_publisher(&metadata_path, install_then_report_parent_sync_failure)
            .expect_err("injected post-rename sync failure");
        assert!(matches!(error, SearchError::Io(_)));
        assert!(
            metadata_path.is_file(),
            "metadata rename may already be visible when parent sync fails"
        );
        let installed: HnswMeta =
            serde_json::from_slice(&std::fs::read(&metadata_path).expect("read metadata"))
                .expect("parse metadata");
        let installed_generation = installed
            .sidecar_generation
            .as_deref()
            .expect("installed generation")
            .to_owned();
        let before = ready_generation_paths(&metadata_path, ann.vector_fingerprint);
        assert_eq!(before.len(), 1);

        ann.save(&metadata_path)
            .expect("retry durability-uncertain publication");
        let after = ready_generation_paths(&metadata_path, ann.vector_fingerprint);
        assert_eq!(
            after, before,
            "retry must not strand the installed generation"
        );
        let repaired: HnswMeta =
            serde_json::from_slice(&std::fs::read(&metadata_path).expect("read repaired metadata"))
                .expect("parse repaired metadata");
        assert_eq!(
            repaired.sidecar_generation.as_deref(),
            Some(installed_generation.as_str())
        );
        let (_, disposition) = HnswIndex::load_with_disposition(&metadata_path, &source_index)
            .expect("native load after durability retry");
        assert_eq!(disposition, HnswLoadDisposition::Native);
    }

    #[test]
    fn save_lock_contention_fails_before_dump() {
        let source_path = temp_path("save-lock-source", "fsvi");
        let source_index = write_index(&source_path, &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]])
            .expect("source index");
        let ann = HnswIndex::build_from_vector_index(&source_index, HnswConfig::default())
            .expect("ANN index");
        let metadata_path = temp_path("save-lock", "hnsw");
        let lock = acquire_hnsw_save_lock(&metadata_path).expect("hold save lock");

        let error = ann
            .save(&metadata_path)
            .expect_err("contending save must fail before staging");
        assert!(matches!(&error, SearchError::Io(_)));
        assert!(
            error.to_string().contains("another writer may be saving"),
            "lock failure should be actionable: {error}"
        );
        assert!(!metadata_path.exists());
        assert!(
            ready_generation_paths(&metadata_path, ann.vector_fingerprint).is_empty(),
            "contending writer must not dump a generation"
        );

        drop(lock);
        ann.save(&metadata_path)
            .expect("persistent lock file remains reusable after holder exits");
        assert_eq!(
            ready_generation_paths(&metadata_path, ann.vector_fingerprint).len(),
            1
        );
    }

    #[test]
    fn save_ignores_mismatched_ready_receipt() {
        let source_path = temp_path("ready-mismatch-source", "fsvi");
        let source_index = write_index(&source_path, &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]])
            .expect("source index");
        let ann = HnswIndex::build_from_vector_index(&source_index, HnswConfig::default())
            .expect("ANN index");
        let metadata_path = temp_path("ready-mismatch", "hnsw");
        ann.save_with_metadata_publisher(&metadata_path, reject_hnsw_metadata_publish)
            .expect_err("retain unpublished READY generation");
        let before = ready_generation_paths(&metadata_path, ann.vector_fingerprint);
        assert_eq!(before.len(), 1);
        let rejected_generation = before[0]
            .file_name()
            .and_then(|name| name.to_str())
            .expect("generation name")
            .to_owned();
        let receipt_path = before[0].join(HNSW_GENERATION_RECEIPT_FILENAME);
        let mut receipt: HnswGenerationReceipt =
            serde_json::from_slice(&std::fs::read(&receipt_path).expect("read receipt"))
                .expect("parse receipt");
        receipt.config.m = receipt.config.m.saturating_add(1);
        std::fs::write(
            &receipt_path,
            serde_json::to_vec(&receipt).expect("serialize mismatched receipt"),
        )
        .expect("write mismatched receipt");

        ann.save(&metadata_path)
            .expect("save past mismatched retained receipt");
        let after = ready_generation_paths(&metadata_path, ann.vector_fingerprint);
        // bd-v03xp made publish reclaim every sibling generation the published
        // metadata does not reference, precisely so corrupt-but-receipt-valid
        // generations stop being re-probed with a full native load on every
        // save. The rejected generation is therefore reclaimed, not retained;
        // what this test pins is that save REFUSED TO REUSE it and published a
        // fresh one instead, which the assert_ne! below carries.
        assert_eq!(
            after.len(),
            1,
            "publish must reclaim the superseded invalid generation"
        );
        let metadata: HnswMeta =
            serde_json::from_slice(&std::fs::read(&metadata_path).expect("read metadata"))
                .expect("parse metadata");
        assert_ne!(
            metadata.sidecar_generation.as_deref(),
            Some(rejected_generation.as_str())
        );
        let (_, disposition) = HnswIndex::load_with_disposition(&metadata_path, &source_index)
            .expect("native load after rejecting receipt");
        assert_eq!(disposition, HnswLoadDisposition::Native);
    }

    #[test]
    fn save_ignores_digest_valid_but_unloadable_ready_generation() {
        let source_path = temp_path("ready-unloadable-source", "fsvi");
        let source_index = write_index(&source_path, &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]])
            .expect("source index");
        let ann = HnswIndex::build_from_vector_index(&source_index, HnswConfig::default())
            .expect("ANN index");
        let metadata_path = temp_path("ready-unloadable", "hnsw");
        ann.save_with_metadata_publisher(&metadata_path, reject_hnsw_metadata_publish)
            .expect_err("retain unpublished READY generation");
        let before = ready_generation_paths(&metadata_path, ann.vector_fingerprint);
        assert_eq!(before.len(), 1);
        let rejected_generation = before[0]
            .file_name()
            .and_then(|name| name.to_str())
            .expect("generation name")
            .to_owned();
        let receipt_path = before[0].join(HNSW_GENERATION_RECEIPT_FILENAME);
        let receipt: HnswGenerationReceipt =
            serde_json::from_slice(&std::fs::read(&receipt_path).expect("read receipt"))
                .expect("parse receipt");
        let corrupt_basename = &receipt.sidecar_basename;
        install_digest_valid_unloadable_data(&before[0], corrupt_basename);

        ann.save(&metadata_path)
            .expect("save past unloadable retained generation");
        let after = ready_generation_paths(&metadata_path, ann.vector_fingerprint);
        assert_eq!(
            after.len(),
            1,
            "an unloadable generation must not trap rebuild-save in a reuse loop, \
             and publish reclaims it once a fresh generation supersedes it (bd-v03xp)"
        );
        let metadata: HnswMeta =
            serde_json::from_slice(&std::fs::read(&metadata_path).expect("read metadata"))
                .expect("parse metadata");
        assert_ne!(
            metadata.sidecar_generation.as_deref(),
            Some(rejected_generation.as_str())
        );
        let (_, disposition) = HnswIndex::load_with_disposition(&metadata_path, &source_index)
            .expect("native load after rejecting unloadable generation");
        assert_eq!(disposition, HnswLoadDisposition::Native);
    }

    #[cfg(feature = "hnsw-patch-ab")]
    #[test]
    fn save_ignores_digest_valid_but_topologically_invalid_ready_generation() {
        let vectors = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]];
        let source_path = temp_path("ready-topology-source", "fsvi");
        let source_index = write_index(&source_path, &vectors).expect("source index");
        let ann = HnswIndex::build_from_vector_index(&source_index, HnswConfig::default())
            .expect("ANN index");
        let metadata_path = temp_path("ready-topology", "hnsw");
        ann.save_with_metadata_publisher(&metadata_path, reject_hnsw_metadata_publish)
            .expect_err("retain unpublished READY generation");
        let before = ready_generation_paths(&metadata_path, ann.vector_fingerprint);
        assert_eq!(before.len(), 1);
        let rejected_generation = before[0]
            .file_name()
            .and_then(|name| name.to_str())
            .expect("generation name")
            .to_owned();
        let receipt_path = before[0].join(HNSW_GENERATION_RECEIPT_FILENAME);
        let receipt: HnswGenerationReceipt =
            serde_json::from_slice(&std::fs::read(&receipt_path).expect("read receipt"))
                .expect("parse receipt");
        install_digest_valid_wrong_layer_generation(&before[0], &receipt.sidecar_basename, vectors);
        let graph_path = before[0].join(format!("{}.hnsw.graph", receipt.sidecar_basename));
        assert!(
            !hnsw_generation_is_loadable(
                &before[0],
                &receipt.sidecar_basename,
                &graph_path,
                ann.len(),
                ann.dimension(),
            ),
            "loadable but topologically invalid READY generation must not be reusable"
        );

        ann.save(&metadata_path)
            .expect("save past topologically invalid retained generation");
        let after = ready_generation_paths(&metadata_path, ann.vector_fingerprint);
        assert_eq!(
            after.len(),
            2,
            "invalid retained generation must remain untouched while a fresh one is published"
        );
        let metadata: HnswMeta =
            serde_json::from_slice(&std::fs::read(&metadata_path).expect("read metadata"))
                .expect("parse metadata");
        assert_ne!(
            metadata.sidecar_generation.as_deref(),
            Some(rejected_generation.as_str())
        );
        let (_, disposition) = HnswIndex::load_with_disposition(&metadata_path, &source_index)
            .expect("native load after rejecting invalid retained generation");
        assert_eq!(disposition, HnswLoadDisposition::Native);
    }

    #[test]
    fn save_ignores_oversized_ready_receipt() {
        let source_path = temp_path("ready-oversized-source", "fsvi");
        let source_index = write_index(&source_path, &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]])
            .expect("source index");
        let ann = HnswIndex::build_from_vector_index(&source_index, HnswConfig::default())
            .expect("ANN index");
        let metadata_path = temp_path("ready-oversized", "hnsw");
        ann.save_with_metadata_publisher(&metadata_path, reject_hnsw_metadata_publish)
            .expect_err("retain unpublished READY generation");
        let before = ready_generation_paths(&metadata_path, ann.vector_fingerprint);
        assert_eq!(before.len(), 1);
        let rejected_generation = before[0]
            .file_name()
            .and_then(|name| name.to_str())
            .expect("generation name")
            .to_owned();
        let receipt_path = before[0].join(HNSW_GENERATION_RECEIPT_FILENAME);
        std::fs::write(
            receipt_path,
            vec![b' '; HNSW_GENERATION_RECEIPT_MAX_BYTES + 1],
        )
        .expect("write oversized receipt fixture");

        ann.save(&metadata_path)
            .expect("save past oversized retained receipt");
        assert_eq!(
            ready_generation_paths(&metadata_path, ann.vector_fingerprint).len(),
            1,
            "publish must reclaim the superseded oversized-receipt generation"
        );
        // The load-bearing contract, which this test previously left implicit
        // in the directory count: the oversized receipt must not be REUSED.
        // Without this the count alone would pass even if save adopted it.
        let metadata: HnswMeta =
            serde_json::from_slice(&std::fs::read(&metadata_path).expect("read metadata"))
                .expect("parse metadata");
        assert_ne!(
            metadata.sidecar_generation.as_deref(),
            Some(rejected_generation.as_str()),
            "save must publish a fresh generation, never reuse the oversized-receipt one"
        );
        let (_, disposition) = HnswIndex::load_with_disposition(&metadata_path, &source_index)
            .expect("native load after rejecting oversized receipt");
        assert_eq!(disposition, HnswLoadDisposition::Native);
    }

    #[test]
    fn save_lock_namespace_prevents_metadata_name_collision() {
        let source_path = temp_path("save-lock-namespace-source", "fsvi");
        let source_index = write_index(&source_path, &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]])
            .expect("source index");
        let ann = HnswIndex::build_from_vector_index(&source_index, HnswConfig::default())
            .expect("ANN index");
        let metadata_path = temp_path("save-lock-namespace", "hnsw");
        let held_lock = acquire_hnsw_save_lock(&metadata_path).expect("hold save lock");
        let lock_path = hnsw_save_lock_path(&metadata_path).expect("lock path");

        let mut old_lock_name = metadata_path
            .file_name()
            .expect("metadata file name")
            .to_os_string();
        old_lock_name.push(".lock");
        let formerly_colliding_metadata_path = metadata_path.with_file_name(old_lock_name);
        ann.save(&formerly_colliding_metadata_path)
            .expect("adjacent .lock metadata name is isolated from the lock namespace");
        assert!(lock_path.is_file(), "held lock inode must remain installed");

        let contention = ann
            .save(&metadata_path)
            .expect_err("original metadata path must remain locked");
        assert!(
            contention
                .to_string()
                .contains("another writer may be saving")
        );

        let reserved_path_error = ann
            .save(&lock_path)
            .expect_err("metadata cannot overwrite the reserved lock namespace");
        assert!(
            reserved_path_error
                .to_string()
                .contains(HNSW_SAVE_LOCK_DIRECTORY)
        );

        drop(held_lock);
        ann.save(&metadata_path)
            .expect("original path saves after lock release");
    }

    #[test]
    fn persistence_writes_native_graph_sidecars() {
        let fsvi_path = temp_path("persist_native", "fsvi");
        let vectors: Vec<Vec<f32>> = (0..64).map(|i| normalized_vector(i, 32)).collect();
        let index = write_index(&fsvi_path, &vectors).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");

        let save_path = temp_path("persist_native", "hnsw");
        ann.save(&save_path).expect("save");

        let meta: HnswMeta =
            serde_json::from_slice(&std::fs::read(&save_path).expect("meta")).expect("parse meta");
        assert_eq!(meta.format_version, HNSW_META_FORMAT_CURRENT);

        // A fresh save gets the requested basename inside an owned generation
        // and records that exact pair in current-format metadata.
        let parent = save_path.parent().expect("parent");
        let basename = save_path.file_stem().unwrap().to_str().unwrap();
        let generation = meta
            .sidecar_generation
            .as_deref()
            .expect("current metadata generation");
        let sidecar_parent = parent.join(generation);
        assert_eq!(meta.sidecar_basename.as_deref(), Some(basename));
        assert!(
            sidecar_parent
                .join(format!("{basename}.hnsw.graph"))
                .is_file(),
            "native graph sidecar should exist"
        );
        assert!(
            sidecar_parent
                .join(format!("{basename}.hnsw.data"))
                .is_file(),
            "native data sidecar should exist"
        );

        // Load goes through the native graph path and still answers correctly.
        let (loaded, disposition) =
            HnswIndex::load_with_disposition(&save_path, &index).expect("native load");
        assert_eq!(disposition, HnswLoadDisposition::Native);
        assert_eq!(loaded.len(), 64);
        let query = normalized_vector(7, 32);
        let hits = loaded
            .knn_search(&query, 3, HNSW_DEFAULT_EF_SEARCH)
            .expect("search");
        assert_eq!(hits[0].doc_id, "doc-0007");
    }

    #[test]
    fn save_accepts_bare_relative_metadata_path() {
        const CHILD_ENV: &str = "FRANKENSEARCH_HNSW_RELATIVE_SAVE_CHILD";

        if std::env::var_os(CHILD_ENV).is_some() {
            let ann = HnswIndex::build_from_parts(
                vec!["relative-doc".to_owned()],
                vec![vec![1.0_f32, 0.0]],
                2,
                HnswConfig::default(),
            )
            .expect("build relative-path ANN");
            let path = Path::new("relative.hnsw");
            ann.save(path).expect("save to bare relative path");

            let meta: HnswMeta =
                serde_json::from_slice(&std::fs::read(path).expect("read relative-path metadata"))
                    .expect("parse relative-path metadata");
            let (sidecar_parent, basename) =
                persisted_hnsw_sidecar_location(path, &meta).expect("resolve relative sidecars");
            assert!(
                sidecar_parent
                    .join(format!("{basename}.hnsw.graph"))
                    .is_file()
            );
            assert!(
                sidecar_parent
                    .join(format!("{basename}.hnsw.data"))
                    .is_file()
            );
            return;
        }

        // The current working directory is process-global. Exercise the bare
        // path in a child test process so parallel tests cannot observe a cwd
        // change while still covering the end-to-end save contract.
        let directory = tempfile::tempdir().expect("relative-path test directory");
        let output =
            std::process::Command::new(std::env::current_exe().expect("current test executable"))
                .arg("hnsw::tests::save_accepts_bare_relative_metadata_path")
                .arg("--exact")
                .arg("--nocapture")
                .current_dir(directory.path())
                .env(CHILD_ENV, "1")
                .output()
                .expect("run relative-path child test");
        assert!(
            output.status.success(),
            "relative-path child failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn native_resave_atomically_publishes_new_generation() {
        let original_source_path = temp_path("resave_original_source", "fsvi");
        let original_source = write_index(
            &original_source_path,
            &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]],
        )
        .expect("original source");
        let original_ann =
            HnswIndex::build_from_vector_index(&original_source, HnswConfig::default())
                .expect("original ann");
        let destination = temp_path("resave_destination", "hnsw");
        original_ann
            .save(&destination)
            .expect("seed occupied destination");
        let original_meta: HnswMeta =
            serde_json::from_slice(&std::fs::read(&destination).expect("original metadata"))
                .expect("parse original metadata");
        let original_generation = original_meta
            .sidecar_generation
            .as_deref()
            .expect("original generation");
        let destination_parent = destination.parent().expect("destination parent");
        let original_sidecar_parent = destination_parent.join(original_generation);
        // Load a graph over the same IDs/count/dimension but different vectors.
        // hnsw_rs marks every loaded graph as mmap-backed. Saving this value
        // over occupied metadata must publish a fresh generation without
        // touching the previously authoritative pair.
        let changed_source_path = temp_path("resave_changed_source", "fsvi");
        let changed_source = write_index(
            &changed_source_path,
            &[vec![0.0_f32, 1.0], vec![1.0_f32, 0.0]],
        )
        .expect("changed source");
        let changed_seed =
            HnswIndex::build_from_vector_index(&changed_source, HnswConfig::default())
                .expect("changed ann");
        let changed_seed_path = temp_path("resave_changed_seed", "hnsw");
        changed_seed
            .save(&changed_seed_path)
            .expect("save changed native graph");
        let changed_meta: HnswMeta = serde_json::from_slice(
            &std::fs::read(&changed_seed_path).expect("changed native metadata"),
        )
        .expect("parse changed native metadata");
        let changed_loaded =
            HnswIndex::try_load_native_graph(&changed_seed_path, &changed_meta, &changed_source)
                .expect("force changed graph through the native mmap-backed load path");
        changed_loaded
            .save(&destination)
            .expect("resave loaded graph over occupied destination");

        let meta: HnswMeta =
            serde_json::from_slice(&std::fs::read(&destination).expect("metadata"))
                .expect("parse metadata");
        let requested = hnsw_sidecar_basename(&destination).expect("requested basename");
        let published_generation = meta
            .sidecar_generation
            .as_deref()
            .expect("current metadata must name its generation");
        let published = meta
            .sidecar_basename
            .as_deref()
            .expect("current metadata must name its native pair");
        assert_ne!(
            published_generation, original_generation,
            "every save must publish a collision-free generation"
        );
        assert_eq!(
            published, requested,
            "metadata must record file_dump's basename"
        );
        let published_parent = destination_parent.join(published_generation);
        assert!(
            published_parent
                .join(format!("{published}.hnsw.graph"))
                .is_file()
        );
        assert!(
            published_parent
                .join(format!("{published}.hnsw.data"))
                .is_file()
        );
        assert!(
            !original_sidecar_parent.exists(),
            "a successfully published replacement must reclaim its superseded generation"
        );
        assert_eq!(
            ready_generation_paths(&destination, changed_loaded.vector_fingerprint),
            vec![published_parent.clone()],
            "the published metadata generation must be the only retained generation for this vector state"
        );

        let native = HnswIndex::try_load_native_graph(&destination, &meta, &changed_source)
            .expect("published metadata must admit the returned native pair");
        let native_hits = native
            .knn_search(&[1.0, 0.0], 1, HNSW_DEFAULT_EF_SEARCH)
            .expect("search published native graph");
        assert_eq!(native_hits[0].doc_id, "doc-0001");

        let reloaded = HnswIndex::load(&destination, &changed_source)
            .expect("reload atomically published graph");
        let reloaded_hits = reloaded
            .knn_search(&[1.0, 0.0], 1, HNSW_DEFAULT_EF_SEARCH)
            .expect("search reloaded graph");
        assert_eq!(reloaded_hits[0].doc_id, "doc-0001");
    }

    #[cfg(unix)]
    #[test]
    fn generation_gc_never_follows_or_removes_a_generation_named_symlink() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().expect("temporary generation root");
        let parent = root.path().join("ann");
        std::fs::create_dir(&parent).expect("create ANN parent");
        let retained_generation = ".vector.fast.generation-0000000000000001-retained";
        std::fs::create_dir(parent.join(retained_generation)).expect("create retained generation");

        let external = root.path().join("external");
        std::fs::create_dir(&external).expect("create external directory");
        let external_sentinel = external.join("must-survive");
        std::fs::write(&external_sentinel, b"outside generation namespace")
            .expect("write external sentinel");
        let linked_generation = ".vector.fast.generation-0000000000000002-link";
        let linked_path = parent.join(linked_generation);
        symlink(&external, &linked_path)
            .expect("link external directory into generation namespace");

        let metadata = HnswMeta {
            format_version: HNSW_META_FORMAT_CURRENT,
            doc_ids: Vec::new(),
            config: HnswConfig::default(),
            dimension: 0,
            vector_fingerprint: 0,
            sidecar_generation: Some(retained_generation.to_owned()),
            sidecar_basename: Some("vector.fast".to_owned()),
            source_identity: None,
        };
        gc_superseded_hnsw_generations(&parent, "vector.fast", &metadata)
            .expect("GC must skip generation-named symlinks");

        assert!(parent.join(retained_generation).is_dir());
        assert!(
            std::fs::symlink_metadata(&linked_path)
                .expect("inspect generation-named symlink")
                .file_type()
                .is_symlink(),
            "GC must not remove a symlink merely because its name matches the generation prefix"
        );
        assert!(
            external_sentinel.is_file(),
            "GC must not traverse into the target of a generation-named symlink"
        );
    }

    #[cfg(unix)]
    #[test]
    fn native_publication_honors_umask_for_metadata_and_generation_artifacts() {
        use std::os::unix::fs::PermissionsExt;

        let root = tempfile::tempdir().expect("temporary ANN root");
        let source_path = root.path().join("source.fsvi");
        let source = write_index(&source_path, &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]])
            .expect("write source index");
        let ann = HnswIndex::build_from_vector_index(&source, HnswConfig::default())
            .expect("build native ANN");
        let metadata_path = root.path().join("vector.fast.hnsw");
        ann.save(&metadata_path).expect("save native ANN");

        let source_mode = std::fs::metadata(&source_path)
            .expect("inspect source index")
            .permissions()
            .mode()
            & 0o666;
        let metadata_mode = std::fs::metadata(&metadata_path)
            .expect("inspect HNSW metadata")
            .permissions()
            .mode()
            & 0o666;
        assert_eq!(
            metadata_mode, source_mode,
            "persisted HNSW metadata must retain the parent deployment's umask policy"
        );

        let metadata: HnswMeta =
            serde_json::from_slice(&std::fs::read(&metadata_path).expect("read HNSW metadata"))
                .expect("parse HNSW metadata");
        let generation = root.path().join(
            metadata
                .sidecar_generation
                .as_deref()
                .expect("metadata must name its generation"),
        );
        let generation_mode = std::fs::metadata(&generation)
            .expect("inspect HNSW generation")
            .permissions()
            .mode()
            & 0o666;
        assert_eq!(
            generation_mode, source_mode,
            "published generation data permissions must retain the parent deployment's umask policy"
        );
        let basename = metadata
            .sidecar_basename
            .as_deref()
            .expect("metadata must name its native sidecars");
        for path in [
            generation.join(format!("{basename}.hnsw.graph")),
            generation.join(format!("{basename}.hnsw.data")),
            generation.join(HNSW_GENERATION_RECEIPT_FILENAME),
        ] {
            assert_eq!(
                std::fs::metadata(&path)
                    .expect("inspect generation artifact")
                    .permissions()
                    .mode()
                    & 0o666,
                source_mode,
                "generation artifact '{}' must retain the parent deployment's umask policy",
                path.display()
            );
        }
    }

    #[test]
    fn load_rebuilds_from_legacy_v1_metadata() {
        let fsvi_path = temp_path("persist_v1", "fsvi");
        let vectors: Vec<Vec<f32>> = (0..64).map(|i| normalized_vector(i, 32)).collect();
        let index = write_index(&fsvi_path, &vectors).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");

        let save_path = temp_path("persist_v1", "hnsw");
        ann.save(&save_path).expect("save");

        // Fabricate a legacy v1 sidecar at a fresh path: identical metadata but
        // with `format_version` stripped (deserializes to 0) and no graph/data
        // sidecars beside it. load() must transparently rebuild from the
        // VectorIndex instead of failing.
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&save_path).expect("meta")).expect("parse");
        let object = value.as_object_mut().expect("object");
        object.remove("format_version");
        object.remove("sidecar_generation");
        object.remove("sidecar_basename");
        let v1_path = temp_path("persist_v1_legacy", "hnsw");
        std::fs::write(&v1_path, serde_json::to_vec(&value).expect("v1 json")).expect("write v1");

        let (loaded, disposition) =
            HnswIndex::load_with_disposition(&v1_path, &index).expect("v1 rebuild load");
        assert_eq!(disposition, HnswLoadDisposition::Rebuilt);
        assert_eq!(loaded.len(), 64);
        let query = normalized_vector(10, 32);
        let hits = loaded
            .knn_search(&query, 5, HNSW_DEFAULT_EF_SEARCH)
            .expect("search");
        assert_eq!(hits[0].doc_id, "doc-0010");
    }

    #[test]
    fn fallback_rebuild_preserves_duplicate_doc_id_vector_alignment() {
        let source_path = temp_path("duplicate-id-source", "fsvi");
        let mut writer =
            VectorIndex::create_with_revision(&source_path, "hash", "test", 2, Quantization::F32)
                .expect("create source index");
        writer
            .write_record("duplicate", &[1.0_f32, 0.0])
            .expect("write first duplicate");
        writer
            .write_record("duplicate", &[0.0_f32, 1.0])
            .expect("write second duplicate");
        writer.finish().expect("finish source index");
        let source_index = VectorIndex::open(&source_path).expect("open source index");

        let legacy_path = temp_path("duplicate-id-legacy", "hnsw");
        let legacy_meta = HnswMeta {
            format_version: 0,
            doc_ids: vec!["duplicate".to_owned(), "duplicate".to_owned()],
            config: HnswConfig::default(),
            dimension: 2,
            vector_fingerprint: 0,
            sidecar_generation: None,
            sidecar_basename: None,
            source_identity: None,
        };
        std::fs::write(
            &legacy_path,
            serde_json::to_vec(&legacy_meta).expect("serialize legacy metadata"),
        )
        .expect("write legacy metadata");

        let (rebuilt, disposition) = HnswIndex::load_with_disposition(&legacy_path, &source_index)
            .expect("fallback rebuild");
        assert_eq!(disposition, HnswLoadDisposition::Rebuilt);

        let expected_doc_ids = vec!["duplicate".to_owned(), "duplicate".to_owned()];
        let expected_vectors = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]];
        assert_eq!(
            rebuilt.vector_fingerprint,
            fingerprint_vectors(&expected_doc_ids, &expected_vectors),
            "fallback must fingerprint the vector at each source row, not the first row sharing its ID"
        );

        let first = rebuilt
            .knn_search(&[1.0, 0.0], 1, HNSW_DEFAULT_EF_SEARCH)
            .expect("search first duplicate vector");
        let second = rebuilt
            .knn_search(&[0.0, 1.0], 1, HNSW_DEFAULT_EF_SEARCH)
            .expect("search second duplicate vector");
        assert_eq!(first[0].index, 0);
        assert_eq!(second[0].index, 1);
    }

    #[test]
    fn load_never_treats_v5_native_graph_as_v6() {
        // The source index says doc-0000 is e0 and doc-0001 is e1.
        let source_path = temp_path("persist_v5_source", "fsvi");
        let source_index = write_index(&source_path, &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]])
            .expect("source index");

        // Build a valid native graph with the same document IDs but the two
        // vectors swapped, then mark its metadata as v5 while forging the live
        // source fingerprint. That makes the format boundary the sole guard: if
        // load accepts v5 as current, an e0 query returns doc-0001. Correct
        // v6-only loading rebuilds from source and returns doc-0000.
        let swapped_path = temp_path("persist_v5_swapped", "fsvi");
        let swapped_index = write_index(&swapped_path, &[vec![0.0_f32, 1.0], vec![1.0_f32, 0.0]])
            .expect("swapped index");
        let swapped_ann = HnswIndex::build_from_vector_index(&swapped_index, HnswConfig::default())
            .expect("swapped ann");
        let save_path = temp_path("persist_v5_native", "hnsw");
        swapped_ann.save(&save_path).expect("save native graph");

        let mut meta: HnswMeta =
            serde_json::from_slice(&std::fs::read(&save_path).expect("meta")).expect("parse");
        meta.format_version = 5;
        meta.vector_fingerprint =
            fingerprint_live_vector_index(&source_index, 2, 2).expect("live source fingerprint");
        std::fs::write(&save_path, serde_json::to_vec(&meta).expect("serialize v5"))
            .expect("write v5 metadata");

        let (loaded, disposition) =
            HnswIndex::load_with_disposition(&save_path, &source_index).expect("rebuild v5");
        assert_eq!(disposition, HnswLoadDisposition::Rebuilt);
        let hits = loaded
            .knn_search(&[1.0, 0.0], 1, HNSW_DEFAULT_EF_SEARCH)
            .expect("search rebuilt graph");
        assert_eq!(
            hits[0].doc_id, "doc-0000",
            "v5 graph without topology attestation must rebuild under v6"
        );
    }

    #[cfg(feature = "hnsw-patch-ab")]
    #[test]
    fn current_digest_valid_malformed_native_graph_rebuilds_instead_of_serving() {
        let vectors = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]];
        let source_path = temp_path("malformed-v6-source", "fsvi");
        let source_index = write_index(&source_path, &vectors).expect("source index");
        let ann = HnswIndex::build_from_vector_index(&source_index, HnswConfig::default())
            .expect("candidate graph");
        let metadata_path = temp_path("malformed-v6", "hnsw");
        ann.save(&metadata_path).expect("save candidate graph");

        let meta: HnswMeta =
            serde_json::from_slice(&std::fs::read(&metadata_path).expect("metadata"))
                .expect("parse metadata");
        let (generation_path, installed_basename) =
            persisted_hnsw_sidecar_location(&metadata_path, &meta).expect("sidecar location");
        // Keep the fixture cryptographically self-consistent. Receipt/digest
        // validation alone must pass so topology attestation is the reason the
        // current-format native pair is rejected.
        install_digest_valid_wrong_layer_generation(&generation_path, &installed_basename, vectors);

        let (loaded, disposition) = HnswIndex::load_with_disposition(&metadata_path, &source_index)
            .expect("malformed native graph must rebuild");
        assert_eq!(disposition, HnswLoadDisposition::Rebuilt);
        let (hits, stats) = loaded
            .knn_search_with_stats(&[1.0, 0.0], 2, HNSW_DEFAULT_EF_SEARCH)
            .expect("search rebuilt graph");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].doc_id, "doc-0000");
        assert_eq!(stats.fallback_reason, None);
    }

    #[test]
    fn current_digest_valid_parser_panic_rebuilds_in_fresh_process() {
        const CHILD_ROOT_ENV: &str = "FRANKENSEARCH_HNSW_PARSER_PANIC_CHILD_ROOT";
        const TEST_NAME: &str =
            "hnsw::tests::current_digest_valid_parser_panic_rebuilds_in_fresh_process";

        if let Some(root) = std::env::var_os(CHILD_ROOT_ENV) {
            let root = PathBuf::from(root);
            let source_path = root.join("source.fsvi");
            let source_index = write_index(&source_path, &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]])
                .expect("source index");
            let ann = HnswIndex::build_from_vector_index(&source_index, HnswConfig::default())
                .expect("ANN graph");
            let metadata_path = root.join("current.hnsw");
            ann.save(&metadata_path).expect("save native generation");
            let meta: HnswMeta =
                serde_json::from_slice(&std::fs::read(&metadata_path).expect("metadata"))
                    .expect("parse metadata");
            let (generation_path, basename) =
                persisted_hnsw_sidecar_location(&metadata_path, &meta)
                    .expect("installed generation");
            install_digest_valid_unloadable_data(&generation_path, &basename);

            let (loaded, disposition) =
                HnswIndex::load_with_disposition(&metadata_path, &source_index)
                    .expect("parser panic must degrade to a source rebuild");
            assert_eq!(disposition, HnswLoadDisposition::Rebuilt);
            let hits = loaded
                .knn_search(&[1.0, 0.0], 2, HNSW_DEFAULT_EF_SEARCH)
                .expect("search rebuilt graph");
            assert_eq!(hits.len(), 2);
            assert_eq!(hits[0].doc_id, "doc-0000");
            return;
        }

        let root = temp_path("parser-panic-child", "dir");
        std::fs::create_dir_all(&root).expect("create child fixture root");
        let status = std::process::Command::new(std::env::current_exe().expect("current test exe"))
            .arg("--exact")
            .arg(TEST_NAME)
            .arg("--nocapture")
            .env(CHILD_ROOT_ENV, &root)
            .status()
            .expect("run isolated malformed-parser child");
        assert!(
            status.success(),
            "digest-valid malformed native generation escaped the rebuild boundary: {status}"
        );
    }

    #[test]
    fn current_native_graph_with_stale_digest_receipt_rebuilds_before_load() {
        let vectors = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]];
        let source_path = temp_path("stale-receipt-source", "fsvi");
        let source_index = write_index(&source_path, &vectors).expect("source index");
        let ann = HnswIndex::build_from_vector_index(&source_index, HnswConfig::default())
            .expect("ANN graph");
        let metadata_path = temp_path("stale-receipt", "hnsw");
        ann.save(&metadata_path).expect("save ANN graph");
        let meta: HnswMeta =
            serde_json::from_slice(&std::fs::read(&metadata_path).expect("metadata"))
                .expect("parse metadata");
        let (generation_path, basename) =
            persisted_hnsw_sidecar_location(&metadata_path, &meta).expect("sidecar location");
        let graph_path = generation_path.join(format!("{basename}.hnsw.graph"));
        std::fs::OpenOptions::new()
            .append(true)
            .open(&graph_path)
            .expect("open graph for corruption")
            .write_all(&[0])
            .expect("append unreceipted byte");

        let (_, disposition) = HnswIndex::load_with_disposition(&metadata_path, &source_index)
            .expect("stale digest must rebuild");
        assert_eq!(disposition, HnswLoadDisposition::Rebuilt);
    }

    #[test]
    fn current_metadata_rejects_missing_or_nonlocal_sidecar_locations() {
        let source_path = temp_path("persist_location_validation_source", "fsvi");
        let source_index = write_index(&source_path, &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]])
            .expect("source index");
        let ann = HnswIndex::build_from_vector_index(&source_index, HnswConfig::default())
            .expect("build graph");
        let save_path = temp_path("persist_location_validation", "hnsw");
        ann.save(&save_path).expect("save native graph");
        let metadata_bytes = std::fs::read(&save_path).expect("metadata bytes");

        for (field, replacement) in [
            ("sidecar_generation", None),
            ("sidecar_basename", None),
            ("sidecar_generation", Some("../escape")),
            ("sidecar_basename", Some("nested/escape")),
        ] {
            let mut value: serde_json::Value =
                serde_json::from_slice(&metadata_bytes).expect("metadata value");
            let object = value.as_object_mut().expect("metadata object");
            if let Some(replacement) = replacement {
                object.insert(field.to_owned(), replacement.into());
            } else {
                object.remove(field);
            }
            let meta: HnswMeta = serde_json::from_value(value).expect("parse corrupted metadata");
            assert!(
                persisted_hnsw_sidecar_location(&save_path, &meta).is_err(),
                "invalid {field} must fail location validation"
            );
            assert!(
                HnswIndex::try_load_native_graph(&save_path, &meta, &source_index).is_none(),
                "invalid {field} must fail native loading closed"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn current_metadata_rejects_symlinked_native_sidecars() {
        use std::os::unix::fs::symlink;

        let source_path = temp_path("persist_symlink_source", "fsvi");
        let source_index = write_index(&source_path, &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]])
            .expect("source index");
        let ann = HnswIndex::build_from_vector_index(&source_index, HnswConfig::default())
            .expect("build graph");
        let save_path = temp_path("persist_symlink_validation", "hnsw");
        ann.save(&save_path).expect("save native graph");
        let metadata_bytes = std::fs::read(&save_path).expect("metadata bytes");
        let original: HnswMeta =
            serde_json::from_slice(&metadata_bytes).expect("parse native metadata");
        let parent = save_path.parent().expect("metadata parent");
        let generation = original
            .sidecar_generation
            .as_deref()
            .expect("native generation");
        let basename = original
            .sidecar_basename
            .as_deref()
            .expect("native basename");
        let original_generation = parent.join(generation);

        let generation_link_name = format!("{generation}-link");
        symlink(&original_generation, parent.join(&generation_link_name))
            .expect("create generation symlink");
        let mut generation_link_meta: HnswMeta =
            serde_json::from_slice(&metadata_bytes).expect("parse generation-link metadata");
        generation_link_meta.sidecar_generation = Some(generation_link_name);
        assert!(
            HnswIndex::try_load_native_graph(&save_path, &generation_link_meta, &source_index)
                .is_none(),
            "native loading must not follow a generation symlink"
        );

        let sidecar_link_generation_name = format!("{generation}-sidecar-links");
        let sidecar_link_generation = parent.join(&sidecar_link_generation_name);
        std::fs::create_dir(&sidecar_link_generation).expect("create sidecar-link generation");
        for suffix in [".hnsw.graph", ".hnsw.data"] {
            symlink(
                original_generation.join(format!("{basename}{suffix}")),
                sidecar_link_generation.join(format!("{basename}{suffix}")),
            )
            .expect("create native sidecar symlink");
        }
        let mut sidecar_link_meta: HnswMeta =
            serde_json::from_slice(&metadata_bytes).expect("parse sidecar-link metadata");
        sidecar_link_meta.sidecar_generation = Some(sidecar_link_generation_name);
        assert!(
            HnswIndex::try_load_native_graph(&save_path, &sidecar_link_meta, &source_index)
                .is_none(),
            "native loading must not follow graph or data symlinks"
        );
    }

    #[test]
    fn load_rejects_current_sidecar_with_missing_vector_fingerprint() {
        let source_path = temp_path("persist_missing_fp_source", "fsvi");
        let source_index = write_index(&source_path, &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]])
            .expect("source index");

        // Persist a current-format native graph whose vectors are swapped
        // behind the same document IDs, then remove only its fingerprint field.
        // Treating the serde default of 0 as an opt-out would admit this stale
        // graph and make an e0 query return doc-0001.
        let swapped_path = temp_path("persist_missing_fp_swapped", "fsvi");
        let swapped_index = write_index(&swapped_path, &[vec![0.0_f32, 1.0], vec![1.0_f32, 0.0]])
            .expect("swapped index");
        let swapped_ann = HnswIndex::build_from_vector_index(&swapped_index, HnswConfig::default())
            .expect("swapped ann");
        let save_path = temp_path("persist_missing_fp_native", "hnsw");
        swapped_ann.save(&save_path).expect("save native graph");

        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&save_path).expect("meta")).expect("parse");
        value
            .as_object_mut()
            .expect("metadata object")
            .remove("vector_fingerprint");
        std::fs::write(
            &save_path,
            serde_json::to_vec(&value).expect("serialize metadata without fingerprint"),
        )
        .expect("write metadata without fingerprint");

        let loaded = HnswIndex::load(&save_path, &source_index)
            .expect("missing fingerprint must rebuild from source");
        let hits = loaded
            .knn_search(&[1.0, 0.0], 1, HNSW_DEFAULT_EF_SEARCH)
            .expect("search rebuilt graph");
        assert_eq!(
            hits[0].doc_id, "doc-0000",
            "current-format metadata cannot omit its vector fingerprint to admit a stale graph"
        );
    }

    #[test]
    fn load_rejects_native_sidecar_when_previously_unsampled_vector_changes() {
        // With 300 rows, the v3 fingerprint's ceil(300 / 256) stride was 2.
        // Row 1 was therefore neither an even-stride sample nor the final row.
        const VECTOR_COUNT: usize = 300;
        let source_path = temp_path("persist_unsampled_source", "fsvi");
        let original_vectors: Vec<Vec<f32>> = (0..VECTOR_COUNT)
            .map(|i| normalized_vector(i, 32))
            .collect();
        let source_index = write_index(&source_path, &original_vectors).expect("source index");
        let ann = HnswIndex::build_from_vector_index(&source_index, HnswConfig::default())
            .expect("build original graph");
        let save_path = temp_path("persist_unsampled_native", "hnsw");
        ann.save(&save_path).expect("save native graph");

        // Keep the document IDs, count, and dimension identical while changing
        // only the old scheme's unsampled row. Negating the original unit
        // vector makes doc-0001 the worst possible match in the stale graph and
        // the exact match after a rebuild, keeping the result proof decisive.
        let changed_vector: Vec<f32> = original_vectors[1].iter().map(|value| -*value).collect();
        let mut changed_vectors = original_vectors;
        changed_vectors[1].clone_from(&changed_vector);
        let changed_path = temp_path("persist_unsampled_changed", "fsvi");
        let changed_index = write_index(&changed_path, &changed_vectors).expect("changed index");

        let loaded = HnswIndex::load(&save_path, &changed_index)
            .expect("fingerprint mismatch must rebuild from changed source");
        let hits = loaded
            .knn_search(&changed_vector, 1, VECTOR_COUNT)
            .expect("search rebuilt graph");
        assert_eq!(
            hits[0].doc_id, "doc-0001",
            "every live vector must affect persisted source identity; otherwise the stale v3 \
             graph survives a change to row 1"
        );
    }

    /// Stale-vectors validation: the native-graph load path must reject a
    /// sidecar whose persisted vector fingerprint disagrees with the live
    /// `VectorIndex` (i.e. someone swapped the FSVI contents behind matching
    /// doc IDs), and transparently fall back to rebuild rather than silently
    /// serving hits against vectors that no longer exist. This is the exact
    /// scenario the prompt for `frankensearch#25` calls out.
    #[test]
    fn load_rejects_native_sidecar_when_vectors_swapped_under_same_doc_ids() {
        // Build + save a native sidecar over an FSVI where doc-i ≈ basis_i.
        let fsvi_path = temp_path("persist_swap", "fsvi");
        let original_vectors: Vec<Vec<f32>> = (0..16).map(|i| normalized_vector(i, 32)).collect();
        let original_index = write_index(&fsvi_path, &original_vectors).expect("index");
        let ann = HnswIndex::build_from_vector_index(&original_index, HnswConfig::default())
            .expect("build");

        let save_path = temp_path("persist_swap", "hnsw");
        ann.save(&save_path).expect("save native graph");

        // Sanity: a load against the *original* FSVI takes the fast path and
        // returns doc-0007 for query≈doc-0007.
        let loaded_original =
            HnswIndex::load(&save_path, &original_index).expect("native load matches");
        let query = normalized_vector(7, 32);
        let original_hits = loaded_original
            .knn_search(&query, 1, HNSW_DEFAULT_EF_SEARCH)
            .expect("search original");
        assert_eq!(original_hits[0].doc_id, "doc-0007");
        // Sanity that the fingerprint actually got stamped.
        let mut meta: HnswMeta =
            serde_json::from_slice(&std::fs::read(&save_path).expect("meta")).expect("parse meta");
        assert_ne!(
            meta.vector_fingerprint, 0,
            "native save must stamp a fingerprint"
        );

        // Now swap vectors behind the same doc IDs (doc-0007 now points at
        // basis_99) while leaving the .hnsw.graph / .hnsw.data sidecars
        // unchanged. A naive native fast path would return doc-0007 against the
        // *old* graph; the fingerprint guard forces a rebuild instead.
        let mut swapped_vectors = original_vectors.clone();
        swapped_vectors[7] = normalized_vector(99, 32);
        let swapped_path = temp_path("persist_swap_after", "fsvi");
        let swapped_index = write_index(&swapped_path, &swapped_vectors).expect("swapped");

        // Copy the metadata + graph + data sidecars next to the swapped FSVI
        // so the load path's directory layout is plausible.
        let swap_save_path = temp_path("persist_swap_after", "hnsw");
        let src_parent = save_path.parent().expect("src parent");
        let dst_parent = swap_save_path.parent().expect("dst parent");
        let src_generation = meta
            .sidecar_generation
            .as_deref()
            .expect("source metadata generation");
        let src_stem = meta
            .sidecar_basename
            .as_deref()
            .expect("source metadata basename");
        let dst_stem = swap_save_path.file_stem().unwrap().to_str().unwrap();
        let dst_generation = format!(".{dst_stem}.relocated");
        let dst_sidecar_parent = dst_parent.join(&dst_generation);
        std::fs::create_dir(&dst_sidecar_parent).expect("create relocated generation");
        for ext in ["hnsw.graph", "hnsw.data"] {
            std::fs::copy(
                src_parent
                    .join(src_generation)
                    .join(format!("{src_stem}.{ext}")),
                dst_sidecar_parent.join(format!("{dst_stem}.{ext}")),
            )
            .expect("copy sidecar");
        }
        meta.sidecar_generation = Some(dst_generation);
        meta.sidecar_basename = Some(dst_stem.to_owned());
        std::fs::write(
            &swap_save_path,
            serde_json::to_vec(&meta).expect("serialize relocated metadata"),
        )
        .expect("write relocated metadata");

        // Load against the swapped FSVI. The fingerprint mismatch must trigger
        // the rebuild fallback, which sees doc-0007 ≈ basis_99 and therefore
        // returns doc-0007 *only* when querying ≈ basis_99 — not when querying
        // basis_7.
        let loaded_swapped =
            HnswIndex::load(&swap_save_path, &swapped_index).expect("load after swap");
        let stale_query = normalized_vector(7, 32);
        let stale_hits = loaded_swapped
            .knn_search(&stale_query, 1, HNSW_DEFAULT_EF_SEARCH)
            .expect("search stale");
        // The persisted graph would have returned doc-0007 (basis_7) here; the
        // rebuilt graph against the swapped FSVI returns whichever doc now
        // *actually* sits near basis_7, which is **not** doc-0007.
        assert_ne!(
            stale_hits[0].doc_id, "doc-0007",
            "fingerprint guard must reject the persisted graph and rebuild against \
             the swapped FSVI; otherwise we'd return stale ANN hits"
        );

        // And the rebuilt graph *can* find the swapped doc by its new vector.
        let new_query = normalized_vector(99, 32);
        let new_hits = loaded_swapped
            .knn_search(&new_query, 1, HNSW_DEFAULT_EF_SEARCH)
            .expect("search new");
        assert_eq!(
            new_hits[0].doc_id, "doc-0007",
            "rebuild path must have picked up the swapped vector for doc-0007"
        );
    }

    // ─── bd-r65a: ANN sidecars are bound to the source FSVI generation ───

    /// Two v2 generations holding byte-identical live content, differing only
    /// in their published generation identity.
    fn identity_bound_pair() -> (crate::ValidatedFsviBytes, crate::ValidatedFsviBytes) {
        use frankensearch_core::generation::{
            ArtifactGenerationIdentityV1, EmbeddingIdentityBundleV1, QuantizationFormat,
        };
        use std::sync::Arc;

        let make = |sequence: u64, nonce: u8, tag: &str| {
            let mut identity = EmbeddingIdentityBundleV1::explicit_test_model("r65a-ann-source", 8);
            identity.storage.format = "fsvi-v2".to_owned();
            identity.storage.quantization = QuantizationFormat::F32;
            identity.storage.endianness = "little-endian".to_owned();
            let binding = crate::FsviV2IdentityBinding::new(
                ArtifactGenerationIdentityV1::new(sequence, [nonce; 16])
                    .expect("valid test generation"),
                identity.freeze().expect("valid frozen identity"),
            )
            .expect("valid FSVI v2 binding");
            let path = temp_path(tag, "fsvi");
            let mut writer =
                VectorIndex::create_v2(&path, binding.clone()).expect("create v2 source");
            for row in 0..6_usize {
                writer
                    .write_record(&format!("doc-{row:04}"), &normalized_vector(row + 1, 8))
                    .expect("write v2 row");
            }
            writer.finish().expect("finish v2 source");
            let bytes = std::fs::read(&path).expect("read v2 source");
            let _ = std::fs::remove_file(&path);
            crate::ValidatedFsviBytes::from_arc(Arc::<[u8]>::from(bytes), &binding)
                .expect("admit v2 source")
        };

        (
            make(41, 0xa1, "r65a-generation-a"),
            make(42, 0xa2, "r65a-generation-b"),
        )
    }

    /// Planted negative: a stale sidecar whose CONTENT matches perfectly.
    ///
    /// Same path shape, same dimension, same doc ids in the same order, and —
    /// asserted below — the same vector fingerprint, so every content check
    /// the loader performs says "this graph belongs to this index". Only the
    /// published generation identity differs. An implementation that bound
    /// sidecars to content alone would serve the stale graph.
    #[test]
    fn a_sidecar_is_refused_against_a_content_identical_different_generation() {
        let (owner_a, owner_b) = identity_bound_pair();
        let index_a = &owner_a.index;
        let index_b = &owner_b.index;

        // The content checks cannot tell these apart.
        assert_eq!(index_a.dimension(), index_b.dimension());
        assert_eq!(index_a.record_count(), index_b.record_count());
        let doc_ids_a: Vec<String> = (0..index_a.record_count())
            .map(|row| index_a.doc_id_at(row).expect("doc id").to_owned())
            .collect();
        let doc_ids_b: Vec<String> = (0..index_b.record_count())
            .map(|row| index_b.doc_id_at(row).expect("doc id").to_owned())
            .collect();
        assert_eq!(doc_ids_a, doc_ids_b, "the fixture must share its doc set");
        let fingerprint_a = fingerprint_live_vector_index(index_a, doc_ids_a.len(), 8)
            .expect("fingerprint generation a");
        let fingerprint_b = fingerprint_live_vector_index(index_b, doc_ids_b.len(), 8)
            .expect("fingerprint generation b");
        assert_eq!(
            fingerprint_a, fingerprint_b,
            "the planted negative requires an identical vector fingerprint; \
             otherwise the content guard alone would catch it and prove nothing"
        );

        // Only the identity differs.
        assert_ne!(
            owner_a.identity_v2().generation,
            owner_b.identity_v2().generation
        );

        let ann = HnswIndex::build_from_vector_index(index_a, HnswConfig::default())
            .expect("build from generation a");
        assert!(
            ann.source_identity.is_some(),
            "a graph built from a v2 source must record its identity"
        );
        assert!(
            ann.matches_vector_index(index_a).expect("match against a"),
            "the graph must still match the generation it was built from"
        );
        assert!(
            !ann.matches_vector_index(index_b).expect("match against b"),
            "a graph built from generation a must NOT match generation b"
        );
    }

    /// The same refusal at the persisted-load boundary: the sidecar must be
    /// rebuilt rather than served natively.
    #[test]
    fn a_persisted_sidecar_rebuilds_against_a_different_generation() {
        let (owner_a, owner_b) = identity_bound_pair();
        let metadata_path = temp_path("r65a-sidecar", "hnsw");
        let ann = HnswIndex::build_from_vector_index(&owner_a.index, HnswConfig::default())
            .expect("build from generation a");
        ann.save(&metadata_path).expect("persist sidecar");

        let (_, native) = HnswIndex::load_with_disposition(&metadata_path, &owner_a.index)
            .expect("load against its own generation");
        assert_eq!(
            native,
            HnswLoadDisposition::Native,
            "the sidecar must load natively against the generation that built it"
        );

        let (rebuilt, disposition) =
            HnswIndex::load_with_disposition(&metadata_path, &owner_b.index)
                .expect("load against a different generation");
        assert_eq!(
            disposition,
            HnswLoadDisposition::Rebuilt,
            "a different generation must force a rebuild, never a native serve"
        );
        assert_eq!(rebuilt.len(), owner_b.index.record_count());

        // The strict read-only API must refuse it outright rather than
        // reporting a native load.
        assert!(
            HnswIndex::try_load_native(&metadata_path, &owner_b.index)
                .expect("strict load call")
                .is_none(),
            "the strict native API must not report a cross-generation sidecar as native"
        );
    }

    /// Absence is not a match, in either direction: a sidecar with no recorded
    /// identity cannot be proven to belong to an identity-bearing generation,
    /// and vice versa. Without this, every pre-binding sidecar would be served
    /// against any v2 source forever.
    #[test]
    fn unbound_and_bound_sidecars_never_admit_each_other() {
        let (owner_a, _) = identity_bound_pair();
        let bound = HnswSourceIdentityV1::capture(&owner_a.index).expect("v2 source is bound");

        assert!(HnswSourceIdentityV1::admits(None, None));
        assert!(HnswSourceIdentityV1::admits(Some(&bound), Some(&bound)));
        assert!(!HnswSourceIdentityV1::admits(None, Some(&bound)));
        assert!(!HnswSourceIdentityV1::admits(Some(&bound), None));
    }

    /// A legacy v1 source has no identity to bind, so the content checks stand
    /// alone and existing behavior is preserved.
    #[test]
    fn a_legacy_v1_source_still_loads_natively() {
        let source_path = temp_path("r65a-legacy-source", "fsvi");
        let source = write_index(
            &source_path,
            &[normalized_vector(1, 8), normalized_vector(2, 8)],
        )
        .expect("legacy source");
        assert!(
            HnswSourceIdentityV1::capture(&source).is_none(),
            "a v1 source carries no identity"
        );
        let ann = HnswIndex::build_from_vector_index(&source, HnswConfig::default())
            .expect("build from legacy source");
        let metadata_path = temp_path("r65a-legacy-sidecar", "hnsw");
        ann.save(&metadata_path).expect("persist legacy sidecar");
        let (_, disposition) =
            HnswIndex::load_with_disposition(&metadata_path, &source).expect("load legacy sidecar");
        assert_eq!(disposition, HnswLoadDisposition::Native);
    }

    /// Publication-time identity re-validation (bd-21zyj).
    ///
    /// Planted negative: a READY generation left on disk by a DIFFERENT
    /// published FSVI generation whose content is indistinguishable from this
    /// graph's. The test asserts the receipt agrees on doc-id fingerprint,
    /// vector fingerprint, dimension and params — every field the reuse
    /// validator checks — so the content guard provably cannot separate them.
    /// Only the source generation identity differs. Without the identity gate
    /// `save()` adopts that directory and publishes metadata naming THIS
    /// graph's identity while pointing at the other generation's bytes.
    #[test]
    fn save_refuses_to_reuse_a_ready_generation_from_another_source_generation() {
        let (owner_a, owner_b) = identity_bound_pair();

        // Generation B publishes first and is then abandoned mid-publication,
        // leaving a durable READY generation on disk.
        let metadata_path = temp_path("bd21zyj-sidecar", "hnsw");
        let ann_b = HnswIndex::build_from_vector_index(&owner_b.index, HnswConfig::default())
            .expect("build from generation b");
        ann_b
            .save_with_metadata_publisher(&metadata_path, reject_hnsw_metadata_publish)
            .expect_err("retain an unpublished READY generation from generation b");
        let leftover = ready_generation_paths(&metadata_path, ann_b.vector_fingerprint);
        assert_eq!(
            leftover.len(),
            1,
            "generation b must leave exactly one READY generation"
        );
        let leftover_name = leftover[0]
            .file_name()
            .and_then(|name| name.to_str())
            .expect("leftover generation name")
            .to_owned();

        // Generation A's graph. Every CONTENT field the reuse validator checks
        // is identical, which is what makes this a planted negative rather
        // than an ordinary mismatch.
        let ann_a = HnswIndex::build_from_vector_index(&owner_a.index, HnswConfig::default())
            .expect("build from generation a");
        assert_eq!(
            ann_a.vector_fingerprint, ann_b.vector_fingerprint,
            "the planted negative requires an identical vector fingerprint"
        );
        assert_eq!(
            ann_a.doc_ids, ann_b.doc_ids,
            "and an identical ordered doc set"
        );
        assert_eq!(ann_a.dimension, ann_b.dimension);
        assert_eq!(ann_a.config, ann_b.config);
        let receipt: HnswGenerationReceipt = serde_json::from_slice(
            &std::fs::read(leftover[0].join(HNSW_GENERATION_RECEIPT_FILENAME))
                .expect("read leftover receipt"),
        )
        .expect("parse leftover receipt");
        assert_eq!(
            receipt.doc_ids_fingerprint,
            fingerprint_doc_ids(&ann_a.doc_ids)
        );
        assert_eq!(receipt.vector_fingerprint, ann_a.vector_fingerprint);
        assert_eq!(receipt.dimension, ann_a.dimension);
        assert_eq!(receipt.config, ann_a.config);
        // ...and only the identity differs.
        assert_ne!(receipt.source_identity, ann_a.source_identity);
        assert!(receipt.source_identity.is_some() && ann_a.source_identity.is_some());

        ann_a.save(&metadata_path).expect("publish generation a");
        let published: HnswMeta = serde_json::from_slice(
            &std::fs::read(&metadata_path).expect("read published metadata"),
        )
        .expect("parse published metadata");
        assert_ne!(
            published.sidecar_generation.as_deref(),
            Some(leftover_name.as_str()),
            "publication must not adopt a READY generation dumped from another source generation"
        );
        assert_eq!(
            published.source_identity, ann_a.source_identity,
            "the published metadata must name the identity its bytes actually came from"
        );
    }

    /// Control: a leftover READY generation from the SAME source generation is
    /// still reused, so the gate refuses foreign generations rather than
    /// disabling reuse altogether.
    #[test]
    fn save_still_reuses_a_ready_generation_from_the_same_source_generation() {
        let (owner_a, _) = identity_bound_pair();
        let metadata_path = temp_path("bd21zyj-control", "hnsw");
        let ann = HnswIndex::build_from_vector_index(&owner_a.index, HnswConfig::default())
            .expect("build from generation a");
        ann.save_with_metadata_publisher(&metadata_path, reject_hnsw_metadata_publish)
            .expect_err("retain an unpublished READY generation");
        let leftover = ready_generation_paths(&metadata_path, ann.vector_fingerprint);
        assert_eq!(leftover.len(), 1);
        let leftover_name = leftover[0]
            .file_name()
            .and_then(|name| name.to_str())
            .expect("generation name")
            .to_owned();

        ann.save(&metadata_path)
            .expect("publish reusing the retained generation");
        let published: HnswMeta = serde_json::from_slice(
            &std::fs::read(&metadata_path).expect("read published metadata"),
        )
        .expect("parse published metadata");
        assert_eq!(
            published.sidecar_generation.as_deref(),
            Some(leftover_name.as_str()),
            "a same-generation READY directory must still be reused"
        );
    }
}
