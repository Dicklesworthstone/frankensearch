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
//! save lock and durable in-generation READY receipt serialize writers and let
//! publication retries reuse complete generations without deleting them.
//! Legacy sidecars and any load failure fall back to the
//! rebuild-from-`VectorIndex` path.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use frankensearch_core::{SearchError, SearchResult, VectorHit};
use hnsw_rs::prelude::{AnnT, DistDot, Hnsw, HnswIo};
use serde::{Deserialize, Serialize};

use crate::VectorIndex;
use crate::recall_certificate::{EfCalibration, calibrate_certified_ef};

/// Default HNSW `M` (max connections per node).
pub const HNSW_DEFAULT_M: usize = 16;
/// Default HNSW `ef_construction` (build-time beam width).
pub const HNSW_DEFAULT_EF_CONSTRUCTION: usize = 200;
/// Default HNSW `ef_search` (query-time beam width).
pub const HNSW_DEFAULT_EF_SEARCH: usize = 100;
/// Default HNSW max layer depth.
pub const HNSW_DEFAULT_MAX_LAYER: usize = 16;

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
/// the exact native sidecar generation and basename selected during publication.
/// Older native graphs must be rebuilt under the current persistence contract.
pub(crate) const HNSW_META_FORMAT_CURRENT: u32 = 5;

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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct HnswSidecarDigest {
    byte_len: u64,
    fnv1a64: u64,
}

/// How an HNSW load obtained its in-memory graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HnswLoadDisposition {
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
    /// Estimated recall@k from the ef/k ratio (see [`estimate_recall`]).
    ///
    /// This is a heuristic point estimate with NO guarantee. For a certified,
    /// distribution-free recall bound (the automated replacement for a human
    /// recall-budget sign-off), use
    /// [`crate::recall_certificate::conformal_recall_lower_bound`] over a measured
    /// calibration sample instead.
    pub estimated_recall: f64,
}

/// HNSW ANN index over vectors aligned to `VectorIndex` row order.
pub struct HnswIndex {
    hnsw: Hnsw<'static, f32, DistDot>,
    doc_ids: Vec<String>,
    dimension: usize,
    config: HnswConfig,
    /// Fingerprint of the vectors the graph was built from. See
    /// [`HnswMeta::vector_fingerprint`].
    vector_fingerprint: u64,
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
        for i in 0..index.record_count() {
            if index.is_deleted(i) {
                continue;
            }
            doc_ids.push(index.doc_id_at(i)?.to_owned());
            vectors.push(index.vector_at_f32(i)?);
        }
        Self::build_from_parts(doc_ids, vectors, dimension, config)
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

    /// Load an ANN index and report whether its graph came from native sidecars
    /// or was rebuilt from `source_index`.
    pub(crate) fn load_with_disposition(
        path: &Path,
        source_index: &VectorIndex,
    ) -> SearchResult<(Self, HnswLoadDisposition)> {
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

    /// Attempt to load the prebuilt native `hnsw_rs` graph for a current-format sidecar.
    ///
    /// Returns `None` (so the caller rebuilds) if any of the following hold —
    /// a degraded sidecar always degrades to "slow load", never a hard error:
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

        // Validate doc-id sequence against the live VectorIndex *before*
        // touching the (potentially expensive) hnsw_rs load.
        if !meta_matches_live_doc_ids(meta, source_index).ok()? {
            tracing::warn!(
                path = %path.display(),
                "HNSW sidecar doc_ids disagree with live VectorIndex; rebuilding"
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
                 (vectors swapped behind matching doc ids); rebuilding"
            );
            return None;
        }

        // `HnswIo::load_hnsw` returns an `Hnsw` borrowed from the `HnswIo`
        // (`'a: 'b`), so to store it in the `'static` field we must keep the
        // `HnswIo` alive for the program. With the default `ReloadOptions`
        // (`datamap: false`) the `HnswIo` holds no bulk data — the graph and
        // vectors are read into the returned `Hnsw`'s owned storage — so the
        // leaked shell is only a couple of paths plus an `Arc`. Leaking it
        // (rather than a self-referential struct or unsafe lifetime transmute)
        // is the simplest sound way to obtain a `'static` graph, and the cost
        // is negligible because a persisted load happens about once per
        // process.
        let hnsw = Box::leak(Box::new(HnswIo::new(&sidecar_parent, &basename)))
            .load_hnsw::<f32, DistDot>()
            .ok()?;

        // Guard against a graph that doesn't match the metadata it shipped with
        // (e.g. truncated dump, mismatched sidecars). The caller additionally
        // validates doc_ids against the live VectorIndex.
        if hnsw.get_nb_point() != meta.doc_ids.len() {
            return None;
        }

        Some(Self {
            hnsw,
            doc_ids: meta.doc_ids.clone(),
            dimension: meta.dimension,
            config: meta.config,
            vector_fingerprint: meta.vector_fingerprint,
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
            return Ok(());
        }

        // Persist into a unique generation first. `hnsw_rs` truncates an
        // occupied basename for freshly built graphs and uses a racy random
        // suffix for loaded graphs; an atomically created directory avoids both
        // behaviors. Metadata remains the sole commit point.
        let generation_prefix =
            hnsw_generation_prefix(&requested_basename, self.vector_fingerprint);
        let generation = tempfile::Builder::new()
            .prefix(&generation_prefix)
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
        }
    }

    /// Run ANN query and return hits plus query diagnostics.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` if query dimension differs.
    pub fn knn_search_with_stats(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> SearchResult<(Vec<VectorHit>, AnnSearchStats)> {
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
            let stats = AnnSearchStats {
                index_size: self.len(),
                dimension: self.dimension,
                ef_search,
                k_requested: k,
                k_returned: 0,
                search_time_us: 0,
                is_approximate: true,
                estimated_recall: 1.0,
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
        let elapsed = start.elapsed();
        let search_time_us = u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX);

        let mut hits = Vec::with_capacity(neighbors.len());
        for neighbor in neighbors {
            let doc_id =
                self.doc_ids
                    .get(neighbor.d_id)
                    .ok_or_else(|| SearchError::InvalidConfig {
                        field: "neighbor_id".to_owned(),
                        value: neighbor.d_id.to_string(),
                        reason: "neighbor id exceeds doc_id table".to_owned(),
                    })?;
            let index = u32::try_from(neighbor.d_id).map_err(|_| SearchError::InvalidConfig {
                field: "neighbor_id".to_owned(),
                value: neighbor.d_id.to_string(),
                reason: "neighbor id exceeds u32 range for VectorHit".to_owned(),
            })?;
            let restored_score = (1.0 - neighbor.distance) / budget.radius_squared;
            let score_envelope = -1.0 - budget.score_tolerance..=1.0 + budget.score_tolerance;
            if !restored_score.is_finite() || !score_envelope.contains(&restored_score) {
                return Err(SearchError::SubsystemError {
                    subsystem: "hnsw",
                    source: Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "DistDot returned distance {} (restored score {restored_score}); \
                             expected a score within [{}, {}]",
                            neighbor.distance,
                            score_envelope.start(),
                            score_envelope.end()
                        ),
                    )),
                });
            }
            hits.push(VectorHit {
                index,
                // Graph and query vectors use the same deterministic radius.
                // Undo that uniform scale so callers continue receiving cosine
                // similarity rather than a dimension-dependent proxy score.
                // Clamp only after proving the deviation is inside the derived
                // floating-point envelope; materially invalid distances fail.
                score: restored_score.clamp(-1.0, 1.0),
                doc_id: doc_id.as_str().into(),
            });
        }
        hits.sort_by(|left, right| {
            left.cmp_by_score(right)
                .then_with(|| left.index.cmp(&right.index))
        });

        let stats = AnnSearchStats {
            index_size: self.len(),
            dimension: self.dimension,
            ef_search: effective_ef,
            k_requested: k,
            k_returned: hits.len(),
            search_time_us,
            is_approximate: true,
            estimated_recall: estimate_recall(effective_ef, effective_k),
        };
        Ok((hits, stats))
    }

    /// Run ANN query and return only the hits.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` if query dimension differs.
    pub fn knn_search(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> SearchResult<Vec<VectorHit>> {
        self.knn_search_with_stats(query, k, ef_search)
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
    /// failed ANN search for a single (query, ef) is treated as recall `0.0` for that
    /// query — the conservative direction, which can only *lower* a certified bound
    /// (never over-certify).
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
                        let approx = self.knn_search(q, k, ef).unwrap_or_default();
                        recall_at_k_of(&approx, exact_hits)
                    })
                    .collect()
            },
            target,
            alpha,
        ))
    }

    /// Returns true when this ANN index matches row order and shape of a `VectorIndex`.
    ///
    /// Since `HnswIndex` no longer stores vectors, this only checks:
    /// 1. Dimension match.
    /// 2. `doc_id` sequence match (ignoring tombstones in `VectorIndex`).
    ///
    /// # Errors
    ///
    /// Propagates decoding errors from `VectorIndex::doc_id_at`.
    pub fn matches_vector_index(&self, index: &VectorIndex) -> SearchResult<bool> {
        if self.dimension != index.dimension() {
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
            // We implicitly assume vectors match if doc_ids match, as HNSW
            // is built *from* the VectorIndex on load.
            live_position = live_position.saturating_add(1);
        }
        // Check if HNSW has extra docs
        Ok(live_position == self.doc_ids.len())
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

        let hnsw = Hnsw::new(
            config.m,
            doc_ids.len().max(1),
            config.max_layer,
            config.ef_construction,
            DistDot,
        );
        if !normalized_vectors.is_empty() {
            let vectors_with_ids: Vec<(&Vec<f32>, usize)> = normalized_vectors
                .iter()
                .enumerate()
                .map(|(index, vector)| (vector, index))
                .collect();
            hnsw.parallel_insert(&vectors_with_ids);
        }

        Ok(Self {
            hnsw,
            doc_ids,
            // vectors: source_vectors, // Removed!
            dimension,
            config,
            vector_fingerprint,
        })
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
        || receipt.doc_count != index.doc_ids.len()
        || receipt.doc_ids_fingerprint != fingerprint_doc_ids(&index.doc_ids)
        || receipt.vector_fingerprint != index.vector_fingerprint
        || receipt.dimension != index.dimension
        || receipt.config != index.config
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

    // Byte identity alone is not enough: a complete pair can still be
    // semantically unreadable after a native-format change or a faulty dump.
    // Republishing such a receipt would make every fallback rebuild select the
    // same broken generation again. Prove that the current reader can load the
    // pair before treating it as reusable.
    if !hnsw_generation_is_loadable(
        generation_path,
        &basename,
        &graph,
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

    Ok(Some(
        index.metadata_for_generation(&generation_name, &basename),
    ))
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
    matches!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            native_io
                .load_hnsw::<f32, DistDot>()
                .map(|candidate| candidate.get_nb_point())
        })),
        Ok(Ok(points)) if points == expected_points
    )
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

fn install_hnsw_metadata(path: &Path, parent: &Path, bytes: &[u8]) -> SearchResult<()> {
    let mut temporary = tempfile::NamedTempFile::new_in(parent).map_err(SearchError::Io)?;
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
    if config.m > 256 {
        return Err(SearchError::InvalidConfig {
            field: "hnsw_m".to_owned(),
            value: config.m.to_string(),
            reason: "hnsw_m must be <= 256".to_owned(),
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

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use crate::Quantization;

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
        let dimension = vectors.first().map_or(8, Vec::len);
        let mut writer =
            VectorIndex::create_with_revision(path, "hash", "test", dimension, Quantization::F32)?;
        for (idx, vector) in vectors.iter().enumerate() {
            writer.write_record(&format!("doc-{idx:04}"), vector)?;
        }
        writer.finish()?;
        VectorIndex::open(path)
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
    fn validate_config_rejects_m_over_256() {
        let config = HnswConfig {
            m: 257,
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
    fn validate_config_accepts_m_256_boundary() {
        let config = HnswConfig {
            m: 256,
            ..HnswConfig::default()
        };
        assert!(validate_config(config).is_ok());
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
        assert!(stats.estimated_recall > 0.0);
        assert!(stats.estimated_recall <= 1.0);
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
    fn matches_returns_true_when_vectors_change_but_doc_ids_match() {
        // After the HNSW refactoring (vectors no longer stored in metadata),
        // matches_vector_index only checks doc_ids and dimension. When doc_ids
        // match, vectors are assumed correct because HNSW is rebuilt from the
        // VectorIndex's current vectors on load.
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
        // Same doc_ids (doc-0000, doc-0001) + same dimension → matches
        assert!(ann.matches_vector_index(&index_b).expect("matches"));
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
        assert_eq!(
            after.len(),
            2,
            "invalid retained receipt must remain untouched while a fresh generation is published"
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
        let mut receipt: HnswGenerationReceipt =
            serde_json::from_slice(&std::fs::read(&receipt_path).expect("read receipt"))
                .expect("parse receipt");
        let corrupt_basename = &receipt.sidecar_basename;
        let data_path = before[0].join(format!("{corrupt_basename}.hnsw.data"));
        std::fs::write(&data_path, b"not loadable HNSW vector data").expect("corrupt data fixture");
        receipt.data = fingerprint_hnsw_sidecar(&data_path).expect("fingerprint corrupt data");
        std::fs::write(
            &receipt_path,
            serde_json::to_vec(&receipt).expect("serialize updated receipt"),
        )
        .expect("make corrupt pair digest-valid");

        ann.save(&metadata_path)
            .expect("save past unloadable retained generation");
        let after = ready_generation_paths(&metadata_path, ann.vector_fingerprint);
        assert_eq!(
            after.len(),
            2,
            "an unloadable generation must not trap rebuild-save in a reuse loop"
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
            2,
            "oversized retained receipt must be ignored without removing it"
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
        let loaded = HnswIndex::load(&save_path, &index).expect("native load");
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
        let original_basename = original_meta
            .sidecar_basename
            .as_deref()
            .expect("original basename");
        let destination_parent = destination.parent().expect("destination parent");
        let original_sidecar_parent = destination_parent.join(original_generation);
        let original_graph_path =
            original_sidecar_parent.join(format!("{original_basename}.hnsw.graph"));
        let original_data_path =
            original_sidecar_parent.join(format!("{original_basename}.hnsw.data"));
        let original_graph = std::fs::read(&original_graph_path).expect("original graph bytes");
        let original_data = std::fs::read(&original_data_path).expect("original data bytes");

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
        assert_eq!(
            std::fs::read(&original_graph_path).expect("retained original graph"),
            original_graph,
            "resave must not truncate the graph named by old metadata"
        );
        assert_eq!(
            std::fs::read(&original_data_path).expect("retained original data"),
            original_data,
            "resave must not truncate the data named by old metadata"
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
    fn load_never_treats_v4_native_graph_as_v5() {
        // The source index says doc-0000 is e0 and doc-0001 is e1.
        let source_path = temp_path("persist_v4_source", "fsvi");
        let source_index = write_index(&source_path, &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]])
            .expect("source index");

        // Build a valid native graph with the same document IDs but the two
        // vectors swapped, then mark its metadata as v4 while forging the live
        // source fingerprint. That makes the format boundary the sole guard: if
        // load accepts v4 as current, an e0 query returns doc-0001. Correct
        // v5-only loading rebuilds from source and returns doc-0000.
        let swapped_path = temp_path("persist_v4_swapped", "fsvi");
        let swapped_index = write_index(&swapped_path, &[vec![0.0_f32, 1.0], vec![1.0_f32, 0.0]])
            .expect("swapped index");
        let swapped_ann = HnswIndex::build_from_vector_index(&swapped_index, HnswConfig::default())
            .expect("swapped ann");
        let save_path = temp_path("persist_v4_native", "hnsw");
        swapped_ann.save(&save_path).expect("save native graph");

        let mut meta: HnswMeta =
            serde_json::from_slice(&std::fs::read(&save_path).expect("meta")).expect("parse");
        meta.format_version = 4;
        meta.vector_fingerprint =
            fingerprint_live_vector_index(&source_index, 2, 2).expect("live source fingerprint");
        std::fs::write(&save_path, serde_json::to_vec(&meta).expect("serialize v4"))
            .expect("write v4 metadata");

        let loaded = HnswIndex::load(&save_path, &source_index).expect("rebuild v4");
        let hits = loaded
            .knn_search(&[1.0, 0.0], 1, HNSW_DEFAULT_EF_SEARCH)
            .expect("search rebuilt graph");
        assert_eq!(
            hits[0].doc_id, "doc-0000",
            "v4 metadata without the v5 generation publication contract must rebuild"
        );
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
}
