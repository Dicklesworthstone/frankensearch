//! In-memory vector index for zero-latency search.
//!
//! Unlike the file-backed [`crate::VectorIndex`] (memory-mapped FSVI), this
//! module stores all vectors in heap-allocated memory, guaranteeing no page
//! faults on access. Vectors are stored as f16 for 50% memory savings.
//!
//! # Usage
//!
//! ```rust,ignore
//! use frankensearch_index::in_memory::InMemoryVectorIndex;
//!
//! // From pre-computed f32 vectors
//! let index = InMemoryVectorIndex::from_vectors(
//!     doc_ids,
//!     vectors,
//!     256,
//! ).unwrap();
//!
//! let hits = index.search_top_k(&query, 10, None).unwrap();
//! ```

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
#[cfg(target_os = "linux")]
use std::ffi::OsString;
#[cfg(target_os = "linux")]
use std::fs::File;
#[cfg(target_os = "linux")]
use std::io::{Read, Write};
#[cfg(target_os = "linux")]
use std::os::fd::{AsRawFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::AtomicU64;

use ahash::AHashMap;
use frankensearch_core::config::{ZeroSignalReason, ZeroSignalState};
use frankensearch_core::filter::{BuildIdentityHasherU64, SearchFilter, fnv1a_hash};
use frankensearch_core::generation::EmbeddingSpaceIdentityV1;
use frankensearch_core::{SearchError, SearchResult, VectorHit};
use half::f16;
use rayon::prelude::*;
use sha2::{Digest, Sha256};

use crate::search::{ClassifiedHits, PARALLEL_CHUNK_SIZE, SearchParams};
use crate::simd::{dot_4bit_prepared, dot_i8_i8, dot_product_f16_f32, prepare_4bit_query};
use crate::{FsviV2Witness, ValidatedFsviBytes, VectorIndex};

const EXACT_RESIDUAL_SIDECAR_MAGIC: [u8; 8] = *b"FSRSIDX1";
const EXACT_RESIDUAL_SIDECAR_VERSION: u32 = 2;
const EXACT_RESIDUAL_BLOCK: usize = 32;
const EXACT_RESIDUAL_LANES: usize = 8;
const EXACT_RESIDUAL_SIDECAR_HEADER_BYTES: usize = 8 + 4 + 4 * 32 + 8 + 8 + 4 + 4;
const EXACT_RESIDUAL_SIDECAR_DIGEST_BYTES: usize = 32;
/// Sidecars are optional, so opening one must never permit unbounded allocation
/// or turn an untrusted artifact into a process-wide memory-pressure event.
const EXACT_RESIDUAL_SIDECAR_MAX_BYTES: usize = 512 * 1024 * 1024;
/// This is deliberately far above supported embedding dimensions while keeping
/// the permutation and its admission bitmap bounded independently of file size.
const EXACT_RESIDUAL_SIDECAR_MAX_DIMENSION: usize = 65_536;
/// A residual scan that finds no lane to prove irrelevant during this probe
/// budget stops paying transform-bound overhead and completes on the incumbent
/// flat scanner. This is a semantic-preserving adaptive fallback, not a speed
/// claim: every result still uses the exact f16 scorer.
const EXACT_RESIDUAL_ADAPTIVE_PROBE_GROUPS: usize = 32;
/// Bound generation-name collision retries while publishing an optional cache
/// artifact. Discovery has separate fixed entry and comparison budgets; if
/// either is exhausted, the optional cache simply remains unavailable.
const EXACT_RESIDUAL_CACHE_ATTEMPTS: usize = 64;
/// Bound every optional-cache directory walk, including unrelated entries.
/// Reaching this fixed limit rejects the cache as unavailable for this open;
/// publication is skipped rather than making an unbounded directory a source
/// of source-derived allocation or I/O.
const EXACT_RESIDUAL_CACHE_DIRECTORY_ENTRY_LIMIT: usize = 256;
/// Bound the number of descriptor-stream comparisons after a candidate passes
/// the fixed-header admission. This remains above the prior sixty-four-entry
/// regression while preventing a cache full of tiny corrupt siblings from
/// turning a product open into unbounded repeated work.
const EXACT_RESIDUAL_CACHE_COMPARISON_CANDIDATE_LIMIT: usize = 128;
/// Bound the total bytes that descriptor-stream comparison can read during one
/// product open. The cap admits one maximum-sized sidecar, but never 1024 of
/// them. Exhaustion is an optional-cache miss and skips publication.
const EXACT_RESIDUAL_CACHE_COMPARISON_BYTE_BUDGET: usize = EXACT_RESIDUAL_SIDECAR_MAX_BYTES;

static EXACT_RESIDUAL_CACHE_NONCE: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
std::thread_local! {
    /// Deterministic seam for the cache-only query-transform allocation path.
    /// It is thread-local so concurrent tests cannot make another search take
    /// an unintended fallback.
    static FAIL_NEXT_RESIDUAL_QUERY_TRANSFORM_ALLOCATION: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    /// Independent seam for allocation of the query suffix envelopes.
    static FAIL_NEXT_RESIDUAL_QUERY_SUFFIX_ALLOCATION: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

#[cfg(all(test, target_os = "linux"))]
std::thread_local! {
    /// Fail the cache publication-capability acquisition before any derived
    /// sidecar allocation. This witnesses the optional-cache fail-open rule.
    static FAIL_NEXT_EXACT_RESIDUAL_PUBLICATION_ACQUISITION: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    /// Counts real source-derived sidecar builds after an individual test has
    /// reset it. It is intentionally thread-local with the failure seam.
    static EXACT_RESIDUAL_SIDECAR_BUILD_COUNT: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn fail_next_residual_query_transform_allocation() {
    FAIL_NEXT_RESIDUAL_QUERY_TRANSFORM_ALLOCATION.with(|failure| failure.set(true));
}

#[cfg(test)]
fn take_residual_query_transform_allocation_failure() -> bool {
    FAIL_NEXT_RESIDUAL_QUERY_TRANSFORM_ALLOCATION.with(|failure| failure.replace(false))
}

#[cfg(test)]
fn residual_query_transform_allocation_failure_is_pending() -> bool {
    FAIL_NEXT_RESIDUAL_QUERY_TRANSFORM_ALLOCATION.with(std::cell::Cell::get)
}

#[cfg(test)]
fn fail_next_residual_query_suffix_allocation() {
    FAIL_NEXT_RESIDUAL_QUERY_SUFFIX_ALLOCATION.with(|failure| failure.set(true));
}

#[cfg(test)]
fn take_residual_query_suffix_allocation_failure() -> bool {
    FAIL_NEXT_RESIDUAL_QUERY_SUFFIX_ALLOCATION.with(|failure| failure.replace(false))
}

#[cfg(test)]
fn residual_query_suffix_allocation_failure_is_pending() -> bool {
    FAIL_NEXT_RESIDUAL_QUERY_SUFFIX_ALLOCATION.with(std::cell::Cell::get)
}

#[cfg(all(test, target_os = "linux"))]
fn fail_next_exact_residual_publication_acquisition() {
    FAIL_NEXT_EXACT_RESIDUAL_PUBLICATION_ACQUISITION.with(|failure| failure.set(true));
}

#[cfg(all(test, target_os = "linux"))]
fn take_exact_residual_publication_acquisition_failure() -> bool {
    FAIL_NEXT_EXACT_RESIDUAL_PUBLICATION_ACQUISITION.with(|failure| failure.replace(false))
}

#[cfg(all(test, target_os = "linux"))]
fn exact_residual_publication_acquisition_failure_is_pending() -> bool {
    FAIL_NEXT_EXACT_RESIDUAL_PUBLICATION_ACQUISITION.with(std::cell::Cell::get)
}

#[cfg(all(test, target_os = "linux"))]
fn reset_exact_residual_sidecar_build_count() {
    EXACT_RESIDUAL_SIDECAR_BUILD_COUNT.with(|count| count.set(0));
}

#[cfg(all(test, target_os = "linux"))]
fn exact_residual_sidecar_build_count() -> usize {
    EXACT_RESIDUAL_SIDECAR_BUILD_COUNT.with(std::cell::Cell::get)
}

#[derive(Clone, Copy, Debug)]
struct ExactResidualLayout {
    groups: usize,
    blocks: usize,
    permutation_len: usize,
    centroid_len: usize,
    residual_len: usize,
    suffix_len: usize,
    lane_len: usize,
    payload_bytes: usize,
    encoded_bytes: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ResidualPruningCensus {
    groups_scanned: usize,
    eligible_lanes: usize,
    lanes_pruned: usize,
    exact_sidecar_scores: usize,
    flat_fallback_rows: usize,
    adaptive_fallbacks: usize,
}

impl ResidualPruningCensus {
    fn merge(&mut self, other: Self) {
        self.groups_scanned = self.groups_scanned.saturating_add(other.groups_scanned);
        self.eligible_lanes = self.eligible_lanes.saturating_add(other.eligible_lanes);
        self.lanes_pruned = self.lanes_pruned.saturating_add(other.lanes_pruned);
        self.exact_sidecar_scores = self
            .exact_sidecar_scores
            .saturating_add(other.exact_sidecar_scores);
        self.flat_fallback_rows = self
            .flat_fallback_rows
            .saturating_add(other.flat_fallback_rows);
        self.adaptive_fallbacks = self
            .adaptive_fallbacks
            .saturating_add(other.adaptive_fallbacks);
    }
}

struct ResidualScanOutcome {
    heap: BinaryHeap<HeapEntry>,
    census: ResidualPruningCensus,
}

struct ResidualQueryTransform {
    transformed: Vec<f32>,
    norm: f64,
    suffix_norms: Vec<f64>,
    flat_f32_rounding_error: f64,
    f32_flat_envelope_is_finite: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExactResidualPublication {
    Published,
    DestinationExists,
}

/// An already-open unnamed inode plus the held no-follow parent descriptor.
/// Acquiring it proves that the optional cache can create an anonymous durable
/// candidate before source-derived sidecar allocation begins.
#[cfg(target_os = "linux")]
struct ExactResidualPublicationCapability {
    directory: OwnedFd,
    destination: OsString,
    file: File,
}

impl ExactResidualLayout {
    fn for_shape(count: usize, dimension: usize) -> SearchResult<Self> {
        if dimension == 0 || dimension > EXACT_RESIDUAL_SIDECAR_MAX_DIMENSION {
            return Err(residual_sidecar_error(
                "dimension",
                "is outside the exact residual sidecar resource bound",
            ));
        }
        let groups = count
            .checked_add(EXACT_RESIDUAL_LANES - 1)
            .ok_or_else(|| residual_sidecar_error("record_count", "group count overflow"))?
            / EXACT_RESIDUAL_LANES;
        let blocks = dimension
            .checked_add(EXACT_RESIDUAL_BLOCK - 1)
            .ok_or_else(|| residual_sidecar_error("dimension", "block count overflow"))?
            / EXACT_RESIDUAL_BLOCK;
        let permutation_len = dimension;
        let centroid_len = groups
            .checked_mul(dimension)
            .ok_or_else(|| residual_sidecar_error("centroids", "length overflow"))?;
        let lane_len = groups
            .checked_mul(EXACT_RESIDUAL_LANES)
            .ok_or_else(|| residual_sidecar_error("lanes", "length overflow"))?;
        let residual_len = lane_len
            .checked_mul(dimension)
            .ok_or_else(|| residual_sidecar_error("residuals", "length overflow"))?;
        let suffix_len = lane_len
            .checked_mul(
                blocks
                    .checked_add(1)
                    .ok_or_else(|| residual_sidecar_error("suffix_norms", "length overflow"))?,
            )
            .ok_or_else(|| residual_sidecar_error("suffix_norms", "length overflow"))?;
        let payload_bytes = permutation_len
            .checked_mul(std::mem::size_of::<u32>())
            .and_then(|bytes| {
                centroid_len
                    .checked_add(residual_len)?
                    .checked_add(suffix_len)?
                    .checked_add(lane_len)?
                    .checked_mul(std::mem::size_of::<f32>())?
                    .checked_add(bytes)
            })
            .ok_or_else(|| residual_sidecar_error("payload", "length overflow"))?;
        let encoded_bytes = EXACT_RESIDUAL_SIDECAR_HEADER_BYTES
            .checked_add(payload_bytes)
            .and_then(|bytes| bytes.checked_add(EXACT_RESIDUAL_SIDECAR_DIGEST_BYTES))
            .ok_or_else(|| residual_sidecar_error("payload", "length overflow"))?;
        if encoded_bytes > EXACT_RESIDUAL_SIDECAR_MAX_BYTES {
            return Err(residual_sidecar_error(
                "payload",
                "exceeds the exact residual sidecar resource bound",
            ));
        }
        Ok(Self {
            groups,
            blocks,
            permutation_len,
            centroid_len,
            residual_len,
            suffix_len,
            lane_len,
            payload_bytes,
            encoded_bytes,
        })
    }
}

/// Immutable identity fields that bind an exact-residual sidecar to one admitted
/// FSVI v2 source generation.  Every member is a cryptographic witness from the
/// source image; display names and dimensions alone are never enough to adopt a
/// transformed search layout.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ResidualSourceBinding {
    generation_fingerprint: [u8; 32],
    vector_content_digest: [u8; 32],
    ordered_live_docset_digest: [u8; 32],
    space_fingerprint: [u8; 32],
}

impl ResidualSourceBinding {
    const fn from_witness(witness: &FsviV2Witness) -> Self {
        Self {
            generation_fingerprint: witness.generation_fingerprint,
            vector_content_digest: witness.vector_content_digest,
            ordered_live_docset_digest: witness.ordered_live_docset_digest,
            space_fingerprint: witness.space_fingerprint,
        }
    }
}

/// Versioned exact early-abandon layout.  Candidate groups are stored in a
/// dimension-major, lane-minor residual slab, with one centroid per group and a
/// suffix norm for every lane/block boundary.  Final scores are always computed
/// through the incumbent f16 flat dot product; this layout can only remove a
/// candidate after its conservative Cauchy--Schwarz upper bound proves it cannot
/// beat the current exact cutoff.
#[derive(Clone, Debug)]
struct ExactResidualSidecar {
    source: ResidualSourceBinding,
    count: usize,
    dimension: usize,
    block: usize,
    lanes: usize,
    permutation: Vec<u32>,
    /// Group-major centroids in transformed dimension order.
    centroids: Vec<f32>,
    /// Group-major, dimension-major, lane-minor residuals.
    residuals: Vec<f32>,
    /// Group-major, block-boundary-major, lane-minor residual suffix norms.
    suffix_norms: Vec<f32>,
    /// Whole-vector reconstruction-error norm per group/lane.  It accounts for
    /// f32 centroid/residual rounding before a lane can be safely abandoned.
    correction_norms: Vec<f32>,
}

impl ExactResidualSidecar {
    fn group_count(&self) -> usize {
        self.count.div_ceil(self.lanes)
    }

    fn block_count(&self) -> usize {
        self.dimension.div_ceil(self.block)
    }

    #[cfg(test)]
    fn is_bound_to(&self, source: &ResidualSourceBinding, count: usize, dimension: usize) -> bool {
        self.source == *source
            && self.count == count
            && self.dimension == dimension
            && self.block == EXACT_RESIDUAL_BLOCK
            && self.lanes == EXACT_RESIDUAL_LANES
    }

    fn validated_layout(&self) -> SearchResult<ExactResidualLayout> {
        let layout = ExactResidualLayout::for_shape(self.count, self.dimension)?;
        if self.permutation.len() != layout.permutation_len
            || self.centroids.len() != layout.centroid_len
            || self.residuals.len() != layout.residual_len
            || self.suffix_norms.len() != layout.suffix_len
            || self.correction_norms.len() != layout.lane_len
        {
            return Err(residual_sidecar_error(
                "payload",
                "in-memory sidecar fields do not match its declared layout",
            ));
        }
        Ok(layout)
    }

    #[cfg(test)]
    fn encode(&self) -> SearchResult<Vec<u8>> {
        let layout = self.validated_layout()?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(layout.encoded_bytes)
            .map_err(|_| residual_sidecar_error("payload", "allocation failed"))?;
        bytes.extend_from_slice(&EXACT_RESIDUAL_SIDECAR_MAGIC);
        bytes.extend_from_slice(&EXACT_RESIDUAL_SIDECAR_VERSION.to_le_bytes());
        for digest in [
            self.source.generation_fingerprint,
            self.source.vector_content_digest,
            self.source.ordered_live_docset_digest,
            self.source.space_fingerprint,
        ] {
            bytes.extend_from_slice(&digest);
        }
        bytes.extend_from_slice(
            &u64::try_from(self.count)
                .map_err(|_| residual_sidecar_error("record_count", "does not fit u64"))?
                .to_le_bytes(),
        );
        bytes.extend_from_slice(
            &u64::try_from(self.dimension)
                .map_err(|_| residual_sidecar_error("dimension", "does not fit u64"))?
                .to_le_bytes(),
        );
        bytes.extend_from_slice(
            &u32::try_from(self.block)
                .map_err(|_| residual_sidecar_error("block", "does not fit u32"))?
                .to_le_bytes(),
        );
        bytes.extend_from_slice(
            &u32::try_from(self.lanes)
                .map_err(|_| residual_sidecar_error("lanes", "does not fit u32"))?
                .to_le_bytes(),
        );
        for value in &self.permutation {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for values in [
            &self.centroids,
            &self.residuals,
            &self.suffix_norms,
            &self.correction_norms,
        ] {
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        debug_assert_eq!(
            bytes.len(),
            EXACT_RESIDUAL_SIDECAR_HEADER_BYTES + layout.payload_bytes
        );
        let digest = Sha256::digest(&bytes);
        bytes.extend_from_slice(&digest);
        Ok(bytes)
    }

    #[cfg(test)]
    fn decode(bytes: &[u8]) -> SearchResult<Self> {
        if bytes.len() > EXACT_RESIDUAL_SIDECAR_MAX_BYTES {
            return Err(residual_sidecar_error(
                "payload",
                "exceeds the exact residual sidecar resource bound",
            ));
        }
        let mut cursor = SidecarCursor::new(bytes);
        if cursor.take_array::<8>("magic")? != EXACT_RESIDUAL_SIDECAR_MAGIC {
            return Err(residual_sidecar_error("magic", "invalid sidecar magic"));
        }
        if cursor.u32("version")? != EXACT_RESIDUAL_SIDECAR_VERSION {
            return Err(residual_sidecar_error(
                "version",
                "unsupported residual-sidecar schema version",
            ));
        }
        let source = ResidualSourceBinding {
            generation_fingerprint: cursor.take_array("generation_fingerprint")?,
            vector_content_digest: cursor.take_array("vector_content_digest")?,
            ordered_live_docset_digest: cursor.take_array("ordered_live_docset_digest")?,
            space_fingerprint: cursor.take_array("space_fingerprint")?,
        };
        let count = usize::try_from(cursor.u64("record_count")?).map_err(|_| {
            residual_sidecar_error("record_count", "does not fit the current platform")
        })?;
        let dimension = usize::try_from(cursor.u64("dimension")?).map_err(|_| {
            residual_sidecar_error("dimension", "does not fit the current platform")
        })?;
        let block = usize::try_from(cursor.u32("block")?)
            .map_err(|_| residual_sidecar_error("block", "does not fit the current platform"))?;
        let lanes = usize::try_from(cursor.u32("lanes")?)
            .map_err(|_| residual_sidecar_error("lanes", "does not fit the current platform"))?;
        if block != EXACT_RESIDUAL_BLOCK || lanes != EXACT_RESIDUAL_LANES {
            return Err(residual_sidecar_error(
                "layout",
                "sidecar layout is not the supported exact residual format",
            ));
        }
        let layout = ExactResidualLayout::for_shape(count, dimension)?;
        if bytes.len() != layout.encoded_bytes
            || cursor.remaining()
                != layout
                    .payload_bytes
                    .checked_add(EXACT_RESIDUAL_SIDECAR_DIGEST_BYTES)
                    .ok_or_else(|| residual_sidecar_error("payload", "length overflow"))?
        {
            return Err(residual_sidecar_error(
                "payload",
                "length does not match the versioned sidecar layout",
            ));
        }
        let integrity_start = bytes
            .len()
            .checked_sub(EXACT_RESIDUAL_SIDECAR_DIGEST_BYTES)
            .ok_or_else(|| residual_sidecar_error("integrity", "digest is truncated"))?;
        let supplied_digest: [u8; EXACT_RESIDUAL_SIDECAR_DIGEST_BYTES] = bytes[integrity_start..]
            .try_into()
            .map_err(|_| residual_sidecar_error("integrity", "digest is truncated"))?;
        let expected_digest: [u8; EXACT_RESIDUAL_SIDECAR_DIGEST_BYTES] =
            Sha256::digest(&bytes[..integrity_start]).into();
        if supplied_digest != expected_digest {
            return Err(residual_sidecar_error(
                "integrity",
                "whole-sidecar digest mismatch",
            ));
        }

        let mut permutation = Vec::new();
        permutation
            .try_reserve_exact(layout.permutation_len)
            .map_err(|_| residual_sidecar_error("permutation", "allocation failed"))?;
        let mut seen = Vec::new();
        seen.try_reserve_exact(dimension)
            .map_err(|_| residual_sidecar_error("permutation", "allocation failed"))?;
        seen.resize(dimension, false);
        for _ in 0..layout.permutation_len {
            let value = cursor.u32("permutation")?;
            let index = usize::try_from(value).map_err(|_| {
                residual_sidecar_error("permutation", "does not fit the current platform")
            })?;
            if index >= dimension || std::mem::replace(&mut seen[index], true) {
                return Err(residual_sidecar_error(
                    "permutation",
                    "must be a complete, duplicate-free dimension permutation",
                ));
            }
            permutation.push(value);
        }
        let centroids = cursor.f32_vec("centroids", layout.centroid_len, false)?;
        let residuals = cursor.f32_vec("residuals", layout.residual_len, false)?;
        let suffix_norms = cursor.f32_vec("suffix_norms", layout.suffix_len, true)?;
        let correction_norms = cursor.f32_vec("correction_norms", layout.lane_len, true)?;
        let digest = cursor.take_array::<EXACT_RESIDUAL_SIDECAR_DIGEST_BYTES>("integrity")?;
        debug_assert_eq!(digest, supplied_digest);
        if !cursor.is_exhausted() {
            return Err(residual_sidecar_error(
                "payload",
                "trailing bytes are forbidden",
            ));
        }
        Ok(Self {
            source,
            count,
            dimension,
            block,
            lanes,
            permutation,
            centroids,
            residuals,
            suffix_norms,
            correction_norms,
        })
    }

    /// Validate the fixed metadata that binds a sidecar header to its source.
    /// The caller separately validates the exact shape and whole-file digest.
    fn header_matches_source(
        header: &[u8],
        source: &ResidualSourceBinding,
        count: usize,
        dimension: usize,
    ) -> SearchResult<bool> {
        if header.len() != EXACT_RESIDUAL_SIDECAR_HEADER_BYTES {
            return Ok(false);
        }
        let mut cursor = SidecarCursor::new(header);
        if cursor.take_array::<8>("magic")? != EXACT_RESIDUAL_SIDECAR_MAGIC
            || cursor.u32("version")? != EXACT_RESIDUAL_SIDECAR_VERSION
        {
            return Ok(false);
        }
        let encoded_source = ResidualSourceBinding {
            generation_fingerprint: cursor.take_array("generation_fingerprint")?,
            vector_content_digest: cursor.take_array("vector_content_digest")?,
            ordered_live_docset_digest: cursor.take_array("ordered_live_docset_digest")?,
            space_fingerprint: cursor.take_array("space_fingerprint")?,
        };
        let encoded_count = usize::try_from(cursor.u64("record_count")?)
            .map_err(|_| residual_sidecar_error("record_count", "does not fit this platform"))?;
        let encoded_dimension = usize::try_from(cursor.u64("dimension")?)
            .map_err(|_| residual_sidecar_error("dimension", "does not fit this platform"))?;
        let block = usize::try_from(cursor.u32("block")?)
            .map_err(|_| residual_sidecar_error("block", "does not fit this platform"))?;
        let lanes = usize::try_from(cursor.u32("lanes")?)
            .map_err(|_| residual_sidecar_error("lanes", "does not fit this platform"))?;
        if encoded_source != *source
            || encoded_count != count
            || encoded_dimension != dimension
            || block != EXACT_RESIDUAL_BLOCK
            || lanes != EXACT_RESIDUAL_LANES
        {
            return Ok(false);
        }
        Ok(cursor.is_exhausted())
    }

    #[cfg(test)]
    fn exactly_matches_derived(&self, expected: &Self) -> bool {
        self.source == expected.source
            && self.count == expected.count
            && self.dimension == expected.dimension
            && self.block == expected.block
            && self.lanes == expected.lanes
            && self.permutation == expected.permutation
            && f32_bits_equal(&self.centroids, &expected.centroids)
            && f32_bits_equal(&self.residuals, &expected.residuals)
            && f32_bits_equal(&self.suffix_norms, &expected.suffix_norms)
            && f32_bits_equal(&self.correction_norms, &expected.correction_norms)
    }
}

#[cfg(target_os = "linux")]
fn write_sidecar_piece(file: &mut File, digest: &mut Sha256, bytes: &[u8]) -> SearchResult<()> {
    file.write_all(bytes)
        .map_err(|error| residual_sidecar_error("publish", &error.to_string()))?;
    digest.update(bytes);
    Ok(())
}

#[cfg(target_os = "linux")]
fn write_sidecar_u32_values(
    file: &mut File,
    digest: &mut Sha256,
    values: &[u32],
) -> SearchResult<()> {
    let mut buffer = [0_u8; 4096];
    for chunk in values.chunks(buffer.len() / std::mem::size_of::<u32>()) {
        for (slot, value) in buffer
            .chunks_exact_mut(std::mem::size_of::<u32>())
            .zip(chunk)
        {
            slot.copy_from_slice(&value.to_le_bytes());
        }
        write_sidecar_piece(
            file,
            digest,
            &buffer[..chunk.len() * std::mem::size_of::<u32>()],
        )?;
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn write_sidecar_f32_values(
    file: &mut File,
    digest: &mut Sha256,
    values: &[f32],
) -> SearchResult<()> {
    let mut buffer = [0_u8; 4096];
    for chunk in values.chunks(buffer.len() / std::mem::size_of::<f32>()) {
        for (slot, value) in buffer
            .chunks_exact_mut(std::mem::size_of::<f32>())
            .zip(chunk)
        {
            slot.copy_from_slice(&value.to_le_bytes());
        }
        write_sidecar_piece(
            file,
            digest,
            &buffer[..chunk.len() * std::mem::size_of::<f32>()],
        )?;
    }
    Ok(())
}

/// Serialize directly into an owned unnamed inode. Publication never needs a
/// second `Vec<u8>` alongside the source-derived transform; the exact layout
/// and whole-file digest match the test-only sidecar encoder's wire format.
#[cfg(target_os = "linux")]
fn write_exact_residual_sidecar_stream(
    file: &mut File,
    sidecar: &ExactResidualSidecar,
) -> SearchResult<()> {
    let layout = sidecar.validated_layout()?;
    let mut digest = Sha256::new();
    write_sidecar_piece(file, &mut digest, &EXACT_RESIDUAL_SIDECAR_MAGIC)?;
    write_sidecar_piece(
        file,
        &mut digest,
        &EXACT_RESIDUAL_SIDECAR_VERSION.to_le_bytes(),
    )?;
    for value in [
        sidecar.source.generation_fingerprint,
        sidecar.source.vector_content_digest,
        sidecar.source.ordered_live_docset_digest,
        sidecar.source.space_fingerprint,
    ] {
        write_sidecar_piece(file, &mut digest, &value)?;
    }
    for value in [
        u64::try_from(sidecar.count)
            .map_err(|_| residual_sidecar_error("record_count", "does not fit u64"))?,
        u64::try_from(sidecar.dimension)
            .map_err(|_| residual_sidecar_error("dimension", "does not fit u64"))?,
    ] {
        write_sidecar_piece(file, &mut digest, &value.to_le_bytes())?;
    }
    for value in [
        u32::try_from(sidecar.block)
            .map_err(|_| residual_sidecar_error("block", "does not fit u32"))?,
        u32::try_from(sidecar.lanes)
            .map_err(|_| residual_sidecar_error("lanes", "does not fit u32"))?,
    ] {
        write_sidecar_piece(file, &mut digest, &value.to_le_bytes())?;
    }
    write_sidecar_u32_values(file, &mut digest, &sidecar.permutation)?;
    for values in [
        &sidecar.centroids,
        &sidecar.residuals,
        &sidecar.suffix_norms,
        &sidecar.correction_norms,
    ] {
        write_sidecar_f32_values(file, &mut digest, values)?;
    }
    debug_assert!(layout.encoded_bytes >= EXACT_RESIDUAL_SIDECAR_DIGEST_BYTES);
    file.write_all(&digest.finalize())
        .map_err(|error| residual_sidecar_error("publish", &error.to_string()))
}

struct SidecarCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> SidecarCursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, field: &str, len: usize) -> SearchResult<&'a [u8]> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| residual_sidecar_error(field, "payload offset overflow"))?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| residual_sidecar_error(field, "payload is truncated"))?;
        self.offset = end;
        Ok(bytes)
    }

    fn take_array<const N: usize>(&mut self, field: &str) -> SearchResult<[u8; N]> {
        self.take(field, N)?
            .try_into()
            .map_err(|_| residual_sidecar_error(field, "fixed-width field is truncated"))
    }

    fn u32(&mut self, field: &str) -> SearchResult<u32> {
        Ok(u32::from_le_bytes(self.take_array(field)?))
    }

    fn u64(&mut self, field: &str) -> SearchResult<u64> {
        Ok(u64::from_le_bytes(self.take_array(field)?))
    }

    #[cfg(test)]
    fn f32_vec(&mut self, field: &str, len: usize, nonnegative: bool) -> SearchResult<Vec<f32>> {
        let byte_len = len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| residual_sidecar_error(field, "payload length overflow"))?;
        let bytes = self.take(field, byte_len)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(len)
            .map_err(|_| residual_sidecar_error(field, "allocation failed"))?;
        for chunk in bytes.chunks_exact(std::mem::size_of::<f32>()) {
            let value = f32::from_le_bytes(chunk.try_into().expect("chunks have f32 width"));
            if !value.is_finite() || (nonnegative && value < 0.0) {
                return Err(residual_sidecar_error(
                    field,
                    "must contain finite values in the declared range",
                ));
            }
            values.push(value);
        }
        Ok(values)
    }

    const fn is_exhausted(&self) -> bool {
        self.offset == self.bytes.len()
    }

    #[cfg(test)]
    const fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }
}

fn residual_sidecar_error(field: &str, reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: format!("exact_residual_sidecar.{field}"),
        value: "redacted".to_owned(),
        reason: reason.to_owned(),
    }
}

#[cfg(test)]
fn f32_bits_equal(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn try_filled_sidecar_vec<T: Clone>(len: usize, value: T, field: &str) -> SearchResult<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| residual_sidecar_error(field, "allocation failed"))?;
    values.resize(len, value);
    Ok(values)
}

/// Open a sidecar's parent once, without resolving a symlink in the route, so
/// all subsequent reads/writes remain bound to this held directory descriptor.
/// The optimization intentionally declines unsupported/unsafe path routes
/// instead of falling back to a path-reopen race.
#[cfg(target_os = "linux")]
fn open_exact_residual_sidecar_parent(path: &Path) -> SearchResult<(OwnedFd, OsString)> {
    use rustix::fs::{CWD, FileType, Mode, OFlags, ResolveFlags, fstat, openat2};

    if path
        .components()
        .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        return Err(residual_sidecar_error(
            "path",
            "parent-directory traversal is not admissible for a sidecar",
        ));
    }
    let name = path.file_name().ok_or_else(|| {
        residual_sidecar_error("path", "destination must name one regular sidecar file")
    })?;
    if name == std::ffi::OsStr::new(".") || name == std::ffi::OsStr::new("..") {
        return Err(residual_sidecar_error(
            "path",
            "destination must name one regular sidecar file",
        ));
    }
    let parent = path
        .parent()
        .filter(|candidate| !candidate.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let directory = openat2(
        CWD,
        parent,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC | OFlags::NONBLOCK | OFlags::NOFOLLOW,
        Mode::empty(),
        ResolveFlags::NO_SYMLINKS,
    )
    .map_err(|error| residual_sidecar_error("path", &error.to_string()))?;
    let stat =
        fstat(&directory).map_err(|error| residual_sidecar_error("path", &error.to_string()))?;
    if FileType::from_raw_mode(stat.st_mode) != FileType::Directory {
        return Err(residual_sidecar_error(
            "path",
            "sidecar parent must be a regular directory",
        ));
    }
    Ok((directory, name.to_os_string()))
}

#[cfg(target_os = "linux")]
fn open_exact_residual_sidecar_file(path: &Path) -> SearchResult<Option<(File, rustix::fs::Stat)>> {
    use rustix::fs::{FileType, Mode, OFlags, fstat, openat};
    use rustix::io::Errno;

    let (directory, name) = match open_exact_residual_sidecar_parent(path) {
        Ok(parts) => parts,
        Err(_) => return Ok(None),
    };
    let descriptor = match openat(
        &directory,
        &name,
        OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NONBLOCK | OFlags::NOFOLLOW,
        Mode::empty(),
    ) {
        Ok(descriptor) => descriptor,
        Err(Errno::NOENT | Errno::LOOP | Errno::NOTDIR) => return Ok(None),
        Err(error) => return Err(residual_sidecar_error("open", &error.to_string())),
    };
    let before =
        fstat(&descriptor).map_err(|error| residual_sidecar_error("open", &error.to_string()))?;
    if FileType::from_raw_mode(before.st_mode) != FileType::RegularFile || before.st_nlink != 1 {
        return Ok(None);
    }
    Ok(Some((File::from(descriptor), before)))
}

#[cfg(target_os = "linux")]
fn same_exact_residual_sidecar_file(
    file: &File,
    before: &rustix::fs::Stat,
    expected_len: usize,
) -> SearchResult<bool> {
    use rustix::fs::fstat;

    let after = fstat(file).map_err(|error| residual_sidecar_error("open", &error.to_string()))?;
    let actual_len = usize::try_from(after.st_size)
        .map_err(|_| residual_sidecar_error("open", "sidecar size does not fit this platform"))?;
    Ok(actual_len == expected_len
        && after.st_dev == before.st_dev
        && after.st_ino == before.st_ino
        && after.st_size == before.st_size
        && after.st_nlink == before.st_nlink)
}

#[cfg(target_os = "linux")]
fn read_sidecar_piece<R: Read>(
    reader: &mut R,
    digest: &mut Sha256,
    expected: &[u8],
) -> SearchResult<bool> {
    let mut actual = [0_u8; 4096];
    let mut offset = 0_usize;
    while offset < expected.len() {
        let take = (expected.len() - offset).min(actual.len());
        match reader.read_exact(&mut actual[..take]) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(false),
            Err(error) => return Err(residual_sidecar_error("open", &error.to_string())),
        }
        digest.update(&actual[..take]);
        if actual[..take] != expected[offset..offset + take] {
            return Ok(false);
        }
        offset += take;
    }
    Ok(true)
}

#[cfg(target_os = "linux")]
fn read_sidecar_u32_values<R: Read>(
    reader: &mut R,
    digest: &mut Sha256,
    values: &[u32],
) -> SearchResult<bool> {
    let mut expected = [0_u8; 4096];
    for values in values.chunks(expected.len() / std::mem::size_of::<u32>()) {
        for (slot, value) in expected
            .chunks_exact_mut(std::mem::size_of::<u32>())
            .zip(values)
        {
            slot.copy_from_slice(&value.to_le_bytes());
        }
        let byte_len = values.len() * std::mem::size_of::<u32>();
        if !read_sidecar_piece(reader, digest, &expected[..byte_len])? {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(target_os = "linux")]
fn read_sidecar_f32_values<R: Read>(
    reader: &mut R,
    digest: &mut Sha256,
    values: &[f32],
) -> SearchResult<bool> {
    let mut expected = [0_u8; 4096];
    for values in values.chunks(expected.len() / std::mem::size_of::<f32>()) {
        for (slot, value) in expected
            .chunks_exact_mut(std::mem::size_of::<f32>())
            .zip(values)
        {
            slot.copy_from_slice(&value.to_le_bytes());
        }
        let byte_len = values.len() * std::mem::size_of::<f32>();
        if !read_sidecar_piece(reader, digest, &expected[..byte_len])? {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(target_os = "linux")]
fn exact_residual_sidecar_header_matches_source(
    path: &Path,
    source: &ResidualSourceBinding,
    count: usize,
    dimension: usize,
) -> SearchResult<bool> {
    let layout = ExactResidualLayout::for_shape(count, dimension)?;
    let Some((mut file, before)) = open_exact_residual_sidecar_file(path)? else {
        return Ok(false);
    };
    let byte_len = usize::try_from(before.st_size)
        .map_err(|_| residual_sidecar_error("open", "sidecar size does not fit this platform"))?;
    if byte_len != layout.encoded_bytes {
        return Ok(false);
    }
    let mut header = [0_u8; EXACT_RESIDUAL_SIDECAR_HEADER_BYTES];
    match file.read_exact(&mut header) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(false),
        Err(error) => return Err(residual_sidecar_error("open", &error.to_string())),
    }
    Ok(
        ExactResidualSidecar::header_matches_source(&header, source, count, dimension)?
            && same_exact_residual_sidecar_file(&file, &before, layout.encoded_bytes)?,
    )
}

/// Compare a stream to one source-derived sidecar after the caller has fixed
/// its descriptor size with `fstat`. Every payload read is exact and the final
/// one-byte probe rejects growth without materializing a raw sidecar image.
/// The generic reader is an intentional deterministic truncation-after-fstat
/// test seam; production passes the held no-follow descriptor itself.
#[cfg(target_os = "linux")]
fn exact_residual_sidecar_stream_matches_reader<R: Read>(
    reader: &mut R,
    expected: &ExactResidualSidecar,
) -> SearchResult<bool> {
    let _ = expected.validated_layout()?;
    let mut header = [0_u8; EXACT_RESIDUAL_SIDECAR_HEADER_BYTES];
    match reader.read_exact(&mut header) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(false),
        Err(error) => return Err(residual_sidecar_error("open", &error.to_string())),
    }
    if !ExactResidualSidecar::header_matches_source(
        &header,
        &expected.source,
        expected.count,
        expected.dimension,
    )? {
        return Ok(false);
    }

    let mut digest = Sha256::new();
    digest.update(header);
    if !read_sidecar_u32_values(reader, &mut digest, &expected.permutation)? {
        return Ok(false);
    }
    for values in [
        &expected.centroids,
        &expected.residuals,
        &expected.suffix_norms,
        &expected.correction_norms,
    ] {
        if !read_sidecar_f32_values(reader, &mut digest, values)? {
            return Ok(false);
        }
    }
    let mut supplied_digest = [0_u8; EXACT_RESIDUAL_SIDECAR_DIGEST_BYTES];
    match reader.read_exact(&mut supplied_digest) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(false),
        Err(error) => return Err(residual_sidecar_error("open", &error.to_string())),
    }
    let mut trailing = [0_u8; 1];
    if reader
        .read(&mut trailing)
        .map_err(|error| residual_sidecar_error("open", &error.to_string()))?
        != 0
    {
        return Ok(false);
    }
    let expected_digest: [u8; EXACT_RESIDUAL_SIDECAR_DIGEST_BYTES] = digest.finalize().into();
    Ok(supplied_digest == expected_digest)
}

#[cfg(target_os = "linux")]
fn exact_residual_sidecar_stream_matches_derived(
    path: &Path,
    expected: &ExactResidualSidecar,
) -> SearchResult<bool> {
    let layout = expected.validated_layout()?;
    let Some((mut file, before)) = open_exact_residual_sidecar_file(path)? else {
        return Ok(false);
    };
    let byte_len = usize::try_from(before.st_size)
        .map_err(|_| residual_sidecar_error("open", "sidecar size does not fit this platform"))?;
    if byte_len != layout.encoded_bytes {
        return Ok(false);
    }
    Ok(
        exact_residual_sidecar_stream_matches_reader(&mut file, expected)?
            && same_exact_residual_sidecar_file(&file, &before, layout.encoded_bytes)?,
    )
}

/// Acquire an unnamed inode through a no-follow parent descriptor before any
/// large source-derived sidecar allocation. Failure proves the optional cache
/// cannot publish and callers must retain the exact flat route without deriving
/// a payload that would be discarded.
#[cfg(target_os = "linux")]
fn acquire_exact_residual_sidecar_publication(
    path: &Path,
) -> SearchResult<ExactResidualPublicationCapability> {
    use rustix::fs::{Mode, OFlags, openat};

    #[cfg(test)]
    if take_exact_residual_publication_acquisition_failure() {
        return Err(residual_sidecar_error(
            "publish",
            "injected publication-capability acquisition failure",
        ));
    }

    let (directory, destination) = open_exact_residual_sidecar_parent(path)?;
    let descriptor = openat(
        &directory,
        ".",
        OFlags::WRONLY | OFlags::TMPFILE | OFlags::CLOEXEC | OFlags::NONBLOCK,
        Mode::RUSR | Mode::WUSR,
    )
    .map_err(|error| residual_sidecar_error("publish", &error.to_string()))?;
    Ok(ExactResidualPublicationCapability {
        directory,
        destination,
        file: File::from(descriptor),
    })
}

/// Consume a previously acquired publication capability, then fsync the held
/// parent directory. `linkat` never replaces a destination, and the unnamed
/// inode is reclaimed automatically on failures before that link. Once linked,
/// a parent-directory `sync_all` failure is reported but the visible destination
/// is deliberately left intact: unlinking it would race the now-published
/// immutable artifact and could erase a durable cache.
#[cfg(target_os = "linux")]
fn publish_exact_residual_sidecar_with_capability(
    capability: ExactResidualPublicationCapability,
    sidecar: &ExactResidualSidecar,
) -> SearchResult<ExactResidualPublication> {
    use rustix::fs::{AtFlags, CWD, linkat};
    use rustix::io::Errno;

    let mut file = capability.file;
    write_exact_residual_sidecar_stream(&mut file, sidecar)?;
    file.sync_all()
        .map_err(|error| residual_sidecar_error("publish", &error.to_string()))?;

    // Link from `/proc/self/fd/<n>` while `file` remains open: this source is
    // the anonymous owned inode, not a temporary pathname an attacker could
    // replace. `linkat` fails with EEXIST rather than replacing `destination`.
    let descriptor_path = format!("/proc/self/fd/{}", file.as_raw_fd());
    match linkat(
        CWD,
        Path::new(&descriptor_path),
        &capability.directory,
        &capability.destination,
        AtFlags::SYMLINK_FOLLOW,
    ) {
        Ok(()) => {}
        Err(Errno::EXIST) => return Ok(ExactResidualPublication::DestinationExists),
        Err(error) => return Err(residual_sidecar_error("publish", &error.to_string())),
    }
    drop(file);
    File::from(capability.directory)
        .sync_all()
        .map_err(|error| residual_sidecar_error("publish", &error.to_string()))?;
    Ok(ExactResidualPublication::Published)
}

/// One-shot public writer wrapper. Product construction may acquire and hold
/// the capability before it derives the sidecar; callers of the explicit write
/// API simply acquire and consume it in this call.
#[cfg(target_os = "linux")]
fn publish_exact_residual_sidecar(
    path: &Path,
    sidecar: &ExactResidualSidecar,
) -> SearchResult<ExactResidualPublication> {
    let capability = acquire_exact_residual_sidecar_publication(path)?;
    publish_exact_residual_sidecar_with_capability(capability, sidecar)
}

#[cfg(not(target_os = "linux"))]
fn publish_exact_residual_sidecar(
    _path: &Path,
    _sidecar: &ExactResidualSidecar,
) -> SearchResult<ExactResidualPublication> {
    Err(residual_sidecar_error(
        "platform",
        "exact residual sidecar publication requires Linux descriptor APIs",
    ))
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn outward_f32_norm(value: f64, field: &str) -> SearchResult<f32> {
    if !value.is_finite() || value < 0.0 {
        return Err(residual_sidecar_error(
            field,
            "norm must be finite and non-negative",
        ));
    }
    let narrowed = value.sqrt() as f32;
    if !narrowed.is_finite() {
        return Err(residual_sidecar_error(field, "norm does not fit f32"));
    }
    // Persist an outward-rounded envelope, not a rounded-down estimate.  The
    // additional epsilon term covers the f64->f32 conversion and later f32
    // multiplication in a conservative Cauchy--Schwarz upper bound.
    let outward = narrowed.mul_add(1.0 + 4.0 * f32::EPSILON, 4.0 * f32::EPSILON);
    if !outward.is_finite() {
        return Err(residual_sidecar_error(
            field,
            "outward norm does not fit f32",
        ));
    }
    Ok(outward)
}

impl ResidualQueryTransform {
    fn from_query(query: &[f32], sidecar: &ExactResidualSidecar) -> SearchResult<Self> {
        #[cfg(test)]
        if take_residual_query_transform_allocation_failure() {
            return Err(residual_sidecar_error(
                "query",
                "injected transform allocation failure",
            ));
        }
        let mut transformed = Vec::new();
        transformed
            .try_reserve_exact(sidecar.dimension)
            .map_err(|_| residual_sidecar_error("query", "transform allocation failed"))?;
        for &source_dimension in &sidecar.permutation {
            let source_dimension = usize::try_from(source_dimension).map_err(|_| {
                residual_sidecar_error("query", "permutation does not fit this platform")
            })?;
            let value = *query.get(source_dimension).ok_or_else(|| {
                residual_sidecar_error("query", "permutation exceeds query dimension")
            })?;
            transformed.push(value);
        }
        let norm_sq = transformed
            .iter()
            .try_fold(0.0_f64, |sum, &value| {
                finite_f64_mul(f64::from(value), f64::from(value))
                    .and_then(|square| finite_f64_add(sum, square))
            })
            .unwrap_or(f64::INFINITY);
        // This norm multiplies the residual/correction envelopes in the
        // pruning inequality, so retain an outward f64 value rather than the
        // nearest-root value used only for presentation.
        let norm =
            outward_f64_norm_from_square_sum(norm_sq, sidecar.dimension).unwrap_or(f64::INFINITY);
        // The interval arithmetic itself is f64, but the authoritative scorer
        // is f32. `f32_flat_envelope` bounds the real absolute dot product of
        // this query against any finite f16 row. The standard Higham model for
        // at most `2d + 32` rounded f32 products/additions gives
        // `|fl(dot) - dot| <= gamma_(2d + 32) * envelope`, where
        // `gamma_n = n*u/(1-n*u)` and `u = f32::EPSILON / 2`. The scalar f16
        // fallback and the SIMD reduction both use no more than that operation
        // budget; the bound is intentionally looser than either implementation.
        // The relative model alone does not cover a subnormal f32 intermediate,
        // so the companion absolute term below budgets half a smallest
        // subnormal for every rounded operation as well.
        // If this envelope could approach f32 overflow, retain the incumbent
        // flat route rather than compare a finite interval to an `inf` tie.
        let f32_flat_sum = transformed.iter().try_fold(0.0_f64, |sum, value| {
            finite_f64_add(sum, f64::from(value.abs()))
        });
        let f32_flat_envelope = f32_flat_sum
            .and_then(|sum| finite_f64_mul(sum, f64::from(f16::MAX.to_f32())))
            .unwrap_or(f64::INFINITY);
        let flat_f32_rounding_error = f32_flat_rounding_error(sidecar.dimension, f32_flat_envelope)
            .filter(|error| error.is_finite())
            .unwrap_or(f64::INFINITY);
        #[cfg(test)]
        if take_residual_query_suffix_allocation_failure() {
            return Err(residual_sidecar_error(
                "query_suffix",
                "injected suffix allocation failure",
            ));
        }
        let mut suffix_norms =
            try_filled_sidecar_vec(sidecar.block_count() + 1, 0.0_f64, "query_suffix")?;
        let mut suffix_sum = 0.0_f64;
        for block_index in (0..sidecar.block_count()).rev() {
            let start = block_index * sidecar.block;
            let end = (start + sidecar.block).min(sidecar.dimension);
            for &value in &transformed[start..end] {
                suffix_sum += f64::from(value) * f64::from(value);
            }
            suffix_norms[block_index] =
                outward_f64_norm_from_square_sum(suffix_sum, sidecar.dimension)
                    .unwrap_or(f64::INFINITY);
        }
        Ok(Self {
            transformed,
            norm,
            suffix_norms,
            flat_f32_rounding_error,
            f32_flat_envelope_is_finite: finite_f64_add(f32_flat_envelope, flat_f32_rounding_error)
                .is_some_and(|total| total < f64::from(f32::MAX) * 0.5),
        })
    }
}

#[allow(clippy::cast_precision_loss)]
fn f32_rounding_gamma(dimension: usize) -> Option<f64> {
    let operation_count = f32_authoritative_dot_operation_count(dimension)?;
    let unit_roundoff = f64::from(f32::EPSILON) * 0.5;
    let rounded_operations = operation_count as f64;
    let numerator = finite_f64_mul(rounded_operations, unit_roundoff)?;
    let denominator = 1.0 - numerator;
    if !denominator.is_finite() || denominator <= 0.0 {
        return None;
    }
    let gamma = numerator / denominator;
    gamma.is_finite().then_some(gamma)
}

#[allow(clippy::cast_precision_loss)]
fn f32_flat_rounding_error(dimension: usize, envelope: f64) -> Option<f64> {
    let operation_count = f32_authoritative_dot_operation_count(dimension)?;
    let relative = finite_f64_mul(envelope, f32_rounding_gamma(dimension)?)?;
    // IEEE-754 gradual underflow gives each subnormal-result rounding an
    // absolute error no greater than half the least f32 subnormal. Adding one
    // such quantum per multiply/add covers the part the classic relative
    // gamma model deliberately excludes.
    let subnormal_quantum = f64::from(f32::from_bits(1)) * 0.5;
    let subnormal_error = finite_f64_mul(operation_count as f64, subnormal_quantum)?;
    finite_f64_add(relative, subnormal_error)
}

/// Perform a floating-point product only when the resulting interval endpoint
/// stays finite. `f64` intentionally has no `checked_mul`; callers use this
/// helper to fail closed to the flat scan rather than admitting an overflowed
/// pruning bound.
fn finite_f64_mul(left: f64, right: f64) -> Option<f64> {
    let product = left * right;
    product.is_finite().then_some(product)
}

/// As [`finite_f64_mul`], but for a conservative interval sum.
fn finite_f64_add(left: f64, right: f64) -> Option<f64> {
    let sum = left + right;
    sum.is_finite().then_some(sum)
}

/// Outward round a finite non-negative norm after its f64 square root. The
/// residual bound consumes this result as a multiplier, so a nearest-even root
/// may not be used directly.
fn outward_f64_sqrt(square: f64) -> Option<f64> {
    if !square.is_finite() || square < 0.0 {
        return None;
    }
    let root = square.sqrt();
    root.is_finite().then(|| root.next_up())
}

/// Conservative scalar-operation budget for the authoritative
/// [`dot_product_f16_f32`] scorer. Each dimension has a multiply and a SIMD
/// accumulator add; the four accumulator tree adds 3 × 8 lanes, `reduce_add`
/// adds 7 lanes, and one spare operation covers dispatch-equivalent tails and
/// reduction details. Thus this is at least `2d + 32` for every actual route.
fn f32_authoritative_dot_operation_count(dimension: usize) -> Option<usize> {
    dimension.checked_mul(2)?.checked_add(32)
}

/// Copy a source-owned document id without relying on infallible `Clone` or
/// `to_owned` allocation. The in-memory product must retain ownership after
/// its file-backed source is dropped, but an allocation failure remains a typed
/// opening error rather than an abort.
fn fallible_doc_id_copy(value: &str) -> SearchResult<String> {
    let mut copy = String::new();
    copy.try_reserve_exact(value.len())
        .map_err(|_| SearchError::InvalidConfig {
            field: "vectors.doc_id".to_owned(),
            value: value.len().to_string(),
            reason: "document-id allocation failed".to_owned(),
        })?;
    copy.push_str(value);
    Ok(copy)
}

#[allow(clippy::cast_precision_loss)]
fn f64_rounding_gamma(dimension: usize) -> Option<f64> {
    // Centroid and residual partials each have one multiply plus one add per
    // dimension, with room for the final combination terms. This controls the
    // f64 evaluation of the mathematical decomposition itself.
    let operation_count = dimension.checked_mul(4)?.checked_add(16)?;
    let unit_roundoff = f64::EPSILON * 0.5;
    let rounded_operations = operation_count as f64;
    let numerator = finite_f64_mul(rounded_operations, unit_roundoff)?;
    let denominator = 1.0 - numerator;
    if !denominator.is_finite() || denominator <= 0.0 {
        return None;
    }
    let gamma = numerator / denominator;
    gamma.is_finite().then_some(gamma)
}

/// Convert a rounded f64 sum-of-squares into an outward norm. The f64 gamma
/// first inflates the finite reduction error, then `outward_f64_sqrt` expands
/// the final root. This value is consumed by the decomposition magnitude, so
/// neither nearest-even step is allowed to point inward.
fn outward_f64_norm_from_square_sum(square_sum: f64, dimension: usize) -> Option<f64> {
    if !square_sum.is_finite() || square_sum < 0.0 {
        return None;
    }
    let inflation = finite_f64_mul(square_sum, f64_rounding_gamma(dimension)?)?;
    let outward_square = finite_f64_add(square_sum, inflation)?;
    outward_f64_sqrt(outward_square)
}

/// Conservative per-lane upper bound for the exact f16 score before the
/// current residual block has been accumulated.  It deliberately works in
/// f64 and inflates the result for the f32 products and reduction used by the
/// authoritative scorer.  A caller may prune only on a strict finite bound
/// below its current cutoff, preserving f32 score ties and the index tie-break.
#[allow(clippy::cast_precision_loss)]
fn residual_lane_upper_bound(
    sidecar: &ExactResidualSidecar,
    transformed: &ResidualQueryTransform,
    block_index: usize,
    centroid_dot: f64,
    centroid_norm: f64,
    residual_norm: f64,
    correction_norm: f64,
    partial: f64,
) -> f64 {
    let magnitude = transformed.norm * (centroid_norm + residual_norm + correction_norm)
        + centroid_dot.abs()
        + partial.abs();
    // The `gamma` term covers the f64 evaluation above; the transform's f32
    // term covers the authoritative f16×f32 score. Both are analytic forward
    // error bounds, not an empirical tolerance. `block_index` participates
    // through `suffix_norms[block_index]`, which is the residual tail being
    // bounded at this exact scan point.
    let decomposition_rounding_error = f64_rounding_gamma(sidecar.dimension)
        .map_or(f64::INFINITY, |gamma| gamma * magnitude.max(1.0));
    centroid_dot
        + partial
        + transformed.suffix_norms[block_index] * residual_norm
        + transformed.norm * correction_norm
        + decomposition_rounding_error
        + transformed.flat_f32_rounding_error
}

/// Fully-resident in-memory vector index with f16 quantization.
///
/// All vectors are stored in a contiguous `Vec<f16>` in row-major order,
/// eliminating memory-map page faults for deterministic sub-millisecond search.
#[derive(Debug, Clone)]
pub struct InMemoryVectorIndex {
    /// Document IDs, indexed by position.
    doc_ids: Vec<String>,
    /// Flat f16 vector slab: `doc_ids.len() * dimension` elements.
    vectors: Vec<f16>,
    /// Lazily-built flat int8 vector slab (same row-major layout) for the int8 ADC
    /// pass-1 of [`InMemoryVectorIndex::search_top_k_int8_two_pass`]. Quantized with
    /// a single corpus-wide max-abs scale, which preserves the dot-product ranking
    /// (the scale is a per-query constant). Built on first two-pass use so exact-only
    /// callers pay neither the quantization work nor its `N·d`-byte footprint.
    vectors_i8: OnceLock<Vec<i8>>,
    /// Lazily-built packed signed-4-bit quantization (2 dims/byte, `dim.div_ceil(2)`
    /// bytes/vector — half the int8 slab) for the optional 4-bit two-pass scan
    /// (`search_top_k_4bit_two_pass`). Built on first 4-bit-two-pass use.
    vectors_nibbles: OnceLock<Vec<u8>>,
    /// Lazily-built FNV-1a hashes of `doc_ids` (same `frankensearch_core::filter`
    /// hash that `BitsetFilter` uses). Lets the filtered scan call
    /// `SearchFilter::matches_doc_id_hash` with a precomputed hash instead of
    /// re-hashing each `doc_id` string per vector (the FSVI `search.rs` scan already
    /// does this). Built on first *filtered* search, so unfiltered callers pay
    /// neither the hashing nor the `8·N`-byte footprint.
    doc_id_hashes: OnceLock<Vec<u64>>,
    /// Lazily-built `doc_id → position` map for O(1) lookup, replacing the O(N)
    /// linear `doc_ids.iter().position(...)` scan in the per-hit quality-rerank
    /// path (`quality_scores_for_hits`), which was O(hits·N). Built on first
    /// doc-id lookup, so search-only callers pay nothing.
    doc_id_index: OnceLock<AHashMap<String, usize>>,
    /// Lazily-built `doc_id_hash → position` map (identity-hashed, same FNV-1a key
    /// space as `BitsetFilter`). Lets a *selective* filtered search gather the
    /// allow-set's positions directly — `O(|allow-set|)` exact dots instead of one
    /// filter probe per corpus document (`scan_gather` vs `scan_range`). Stored as
    /// `Option`: `None` means two `doc_ids` collide to the same hash, so the map is
    /// not a bijection and the gather fast-path is disabled (the per-document scan
    /// stays correct). Built on first selective-filter search; other callers pay
    /// neither the build nor its footprint.
    hash_to_pos: OnceLock<Option<HashMap<u64, usize, BuildIdentityHasherU64>>>,
    /// Lowercase hex SHA-256 fingerprint of the mathematical embedding space
    /// this index's vectors were produced in, when known (bd-9xuj T2-C3).
    ///
    /// `Some` only when the source declared it: an FSVI v2 file's validated
    /// identity header ([`Self::from_fsvi`]) or an explicit caller-supplied
    /// space ([`Self::from_vectors_with_identity`]). `None` is the typed
    /// legacy-unidentified state (v1 files, [`Self::from_vectors`]) — it is
    /// never fabricated from a default, and downstream seams route it as
    /// legacy rather than failing closed.
    space_fingerprint_hex: Option<String>,
    /// Embedder id recorded by the source that produced this index's vectors
    /// (bd-9xuj T2-C2): the FSVI header's `embedder_id` on the file-backed
    /// load paths, or the declared space's `logical_model_id` on
    /// [`Self::from_vectors_with_identity`]. Diagnostics only — compatibility
    /// decisions join on [`Self::space_fingerprint_hex`], never on this
    /// string. `None` means the source carried none ([`Self::from_vectors`]);
    /// absence stays typed, never defaulted.
    embedder_id: Option<String>,
    /// Embedder revision recorded by the source, captured under the same
    /// rules as [`Self::embedder_id`] (bd-9xuj T2-C2). Kept verbatim: a v1
    /// header's empty revision is preserved as `Some("")`, distinct from the
    /// `None` of a source with no header at all.
    embedder_revision: Option<String>,
    /// Whether [`Self::space_fingerprint_hex`] was derived from a validated
    /// FSVI v2 identity HEADER — i.e. read out of the artifact's own bytes
    /// through exact admission — rather than declared by a caller
    /// (bd-9xuj T2-C4-write, admission guards 2+8).
    ///
    /// `true` only on the [`Self::from_admitted_v2`] load path (the sole way
    /// a v2 header reaches this type: [`VectorIndex::open`] is strictly v1).
    /// A caller-supplied space on [`Self::from_vectors_with_identity`] is a
    /// DECLARED identity: retained for joins and diagnostics, but it is a
    /// claim by the constructing process, not an attestation persisted in an
    /// artifact header, so it stays `false`. Attested-only seams (the refresh
    /// identity-bound merge) admit against attested identity exclusively.
    space_identity_attested: bool,
    /// Vector dimensionality.
    dimension: usize,
    /// Present only for an index loaded from an exactly admitted FSVI v2 image.
    /// A residual sidecar requires this binding; legacy FSVI and caller-built
    /// indexes deliberately remain on the established flat route.
    residual_source_binding: Option<ResidualSourceBinding>,
    /// Lazily attached after the complete sidecar header, source witnesses,
    /// dimensions, permutation, and every numeric payload have been admitted.
    /// An absent or rejected sidecar is intentionally indistinguishable from no
    /// optimization request to the public search path.
    exact_residual_sidecar: OnceLock<ExactResidualSidecar>,
}

/// Quantize an f16 slab to int8 using one corpus-wide max-abs scale.
///
/// Symmetric int8: `q = round(x / max_abs * 127)`, clamped to `[-127, 127]`. A
/// single global scale keeps `Σ q_a·q_b` monotonic with the true dot for a fixed
/// query, so pass-1 ranking is preserved; the exact f16 rescore restores values.
#[allow(clippy::cast_possible_truncation)] // round()+clamp() bounds the f32->i8 cast
fn quantize_i8_slab(vectors_f16: &[f16]) -> Vec<i8> {
    // Runtime-dispatched (AVX2+F16C when available); see `simd` for the kernel.
    crate::simd::quantize_f16_slab_to_i8(vectors_f16)
}

/// Quantize an f32 query to int8 using its own max-abs scale (the scale is a
/// per-query constant and does not affect ranking).
#[allow(clippy::cast_possible_truncation)] // round()+clamp() bounds the f32->i8 cast
fn quantize_i8_query(query: &[f32]) -> Vec<i8> {
    let max_abs = query.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    if max_abs <= 0.0 {
        return vec![0_i8; query.len()];
    }
    let scale = 127.0 / max_abs;
    query
        .iter()
        .map(|&x| (x * scale).round().clamp(-127.0, 127.0) as i8)
        .collect()
}

/// Quantize one component to a signed 4-bit nibble (`[-7, 7]`, 4-bit two's
/// complement in the low 4 bits) given a scale.
#[allow(clippy::cast_possible_truncation)] // round()+clamp() bounds the cast
fn nibble_of(value: f32, scale: f32) -> u8 {
    let q = (value * scale).round().clamp(-7.0, 7.0) as i8;
    q.cast_unsigned() & 0x0F
}

/// Pack an f32 query into signed 4-bit nibbles, 2 dims/byte (low = even dim, high =
/// odd dim), using the query's own max-abs scale (a per-query constant that does not
/// change the dot ranking). Matches `pack_4bit_slab`'s layout for `dot_packed_4bit`.
fn pack_4bit_query(query: &[f32]) -> Vec<u8> {
    let max_abs = query.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    let scale = if max_abs > 1e-9 { 7.0 / max_abs } else { 0.0 };
    let mut packed = vec![0_u8; query.len().div_ceil(2)];
    for (d, &x) in query.iter().enumerate() {
        let nib = nibble_of(x, scale);
        if d % 2 == 0 {
            packed[d / 2] |= nib;
        } else {
            packed[d / 2] |= nib << 4;
        }
    }
    packed
}

/// Pack a contiguous f16 vector slab (`count·dim`) into signed 4-bit nibbles
/// (`dim.div_ceil(2)` bytes/vector) with one corpus-wide max-abs scale (a constant
/// factor, so the dot ranking is preserved).
fn pack_4bit_slab(vectors_f16: &[f16], dim: usize) -> Vec<u8> {
    // Runtime-dispatched (AVX2+F16C when available); see `simd` for the kernel.
    crate::simd::pack_f16_slab_to_4bit(vectors_f16, dim)
}

impl InMemoryVectorIndex {
    /// Build from pre-computed f32 vectors, quantizing to f16.
    ///
    /// The resulting index carries no embedding-space identity
    /// ([`Self::space_fingerprint_hex`] returns `None`): nothing about a bare
    /// `Vec<f32>` proves which space produced it, and fabricating an identity
    /// here would defeat the space verifier. Callers that know the producing
    /// space should use [`Self::from_vectors_with_identity`].
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` if any vector's length does not
    /// match `dimension`.
    pub fn from_vectors(
        doc_ids: Vec<String>,
        vectors: Vec<Vec<f32>>,
        dimension: usize,
    ) -> SearchResult<Self> {
        if doc_ids.len() != vectors.len() {
            return Err(SearchError::InvalidConfig {
                field: "vectors".to_owned(),
                value: format!("doc_ids={}, vectors={}", doc_ids.len(), vectors.len()),
                reason: "doc_ids and vectors must have the same length".to_owned(),
            });
        }
        let count = doc_ids.len();
        let flat_capacity =
            count
                .checked_mul(dimension)
                .ok_or_else(|| SearchError::InvalidConfig {
                    field: "vectors".to_owned(),
                    value: format!("count={count}, dimension={dimension}"),
                    reason: "vector slab length overflow".to_owned(),
                })?;
        let mut flat = Vec::new();
        flat.try_reserve_exact(flat_capacity)
            .map_err(|_| SearchError::InvalidConfig {
                field: "vectors".to_owned(),
                value: format!("count={count}, dimension={dimension}"),
                reason: "vector slab allocation failed".to_owned(),
            })?;
        for (i, vec) in vectors.into_iter().enumerate() {
            if vec.len() != dimension {
                return Err(SearchError::DimensionMismatch {
                    expected: dimension,
                    found: vec.len(),
                });
            }
            // Validate finite values
            for val in &vec {
                if !val.is_finite() {
                    return Err(SearchError::InvalidConfig {
                        field: "vectors".to_owned(),
                        value: format!("vector[{i}] contains non-finite value"),
                        reason: "all vector elements must be finite".to_owned(),
                    });
                }
            }
            crate::simd::encode_f32_to_f16_extend(&vec, &mut flat);
        }
        Ok(Self {
            doc_ids,
            vectors: flat,
            vectors_i8: OnceLock::new(),
            vectors_nibbles: OnceLock::new(),
            doc_id_hashes: OnceLock::new(),
            doc_id_index: OnceLock::new(),
            hash_to_pos: OnceLock::new(),
            space_fingerprint_hex: None,
            embedder_id: None,
            embedder_revision: None,
            space_identity_attested: false,
            dimension,
            residual_source_binding: None,
            exact_residual_sidecar: OnceLock::new(),
        })
    }

    /// Build from pre-computed f32 vectors that are known to have been
    /// produced in `space`, binding the index to that embedding space
    /// (bd-9xuj T2-C3).
    ///
    /// The space's fingerprint becomes this index's
    /// [`Self::space_fingerprint_hex`], the index-side join key for
    /// [`frankensearch_core::BoundQueryEmbedding::verify_space_identity`].
    /// The claim is checked before it is stored: `space` must itself
    /// validate, and its declared dimension must equal `dimension` — an
    /// index must never carry an identity that does not describe its
    /// vectors.
    ///
    /// # Errors
    ///
    /// Everything [`Self::from_vectors`] returns, plus
    /// `SearchError::InvalidConfig` when `space` fails validation or its
    /// dimension does not match `dimension`.
    pub fn from_vectors_with_identity(
        doc_ids: Vec<String>,
        vectors: Vec<Vec<f32>>,
        dimension: usize,
        space: &EmbeddingSpaceIdentityV1,
    ) -> SearchResult<Self> {
        space.validate()?;
        if usize::try_from(space.dimension).ok() != Some(dimension) {
            return Err(SearchError::InvalidConfig {
                field: "space_identity.dimension".to_owned(),
                value: space.dimension.to_string(),
                reason: format!(
                    "embedding-space dimension must equal the index dimension ({dimension}); \
                     refusing to bind an identity that does not describe this index's vectors"
                ),
            });
        }
        let mut index = Self::from_vectors(doc_ids, vectors, dimension)?;
        index.space_fingerprint_hex = Some(space.fingerprint());
        // The validated space also names its producing model; retain that as
        // the diagnostic embedder identity (bd-9xuj T2-C2). The join key
        // above stays the only compatibility authority.
        index.embedder_id = Some(space.logical_model_id.clone());
        index.embedder_revision = Some(space.immutable_revision.clone());
        // A caller-supplied space is a DECLARED identity, not one attested by
        // an artifact header (bd-9xuj T2-C4-write): the discriminator stays
        // false, and attested-only seams must not admit against it.
        index.space_identity_attested = false;
        Ok(index)
    }

    /// Load from an existing FSVI file, reading all data into memory.
    ///
    /// This reads the entire file-backed index into heap memory, eliminating
    /// page-fault latency on subsequent searches.
    ///
    /// # Errors
    ///
    /// Returns errors from [`VectorIndex::open_read_only`] or vector decoding failures.
    pub fn from_fsvi(path: &Path) -> SearchResult<Self> {
        let index = VectorIndex::open_read_only(path)?;
        Self::from_open_index(&index)
    }

    /// Build from a fully admitted immutable FSVI v2 artifact, preserving its
    /// validated embedding-space identity (bd-9xuj T2-C3).
    ///
    /// [`VectorIndex::open`] — and therefore [`Self::from_fsvi`] — reads
    /// legacy v1 files only; identity-complete v2 artifacts are opened
    /// exclusively through exact admission
    /// ([`VectorIndex::open_admitted_v2`]). This constructor is the in-memory
    /// load for that path: the resulting index always carries the artifact's
    /// validated space fingerprint in [`Self::space_fingerprint_hex`].
    ///
    /// # Errors
    ///
    /// Returns vector decoding failures from the admitted artifact.
    pub fn from_admitted_v2(source: &ValidatedFsviBytes) -> SearchResult<Self> {
        let mut index = Self::from_open_index(&source.index)?;
        index.residual_source_binding = Some(ResidualSourceBinding::from_witness(source.witness()));
        Ok(index)
    }

    /// Load an admitted FSVI v2 generation through the shipping residual-cache
    /// route. The cache directory may hold immutable sidecars for many source
    /// generations; only an exact generation-key match that re-derives bitwise
    /// from this admitted source is attached. Missing, stale, corrupt, or
    /// collision-held paths cause a new generation-keyed immutable candidate to
    /// be published without overwriting any prior artifact.
    ///
    /// This is the single-vector production construction seam. The two-tier
    /// shipping constructor below calls it for each admitted tier, so product
    /// users do not need test-only field mutation or disconnected I/O calls.
    ///
    /// # Errors
    ///
    /// Returns source-vector decoding errors. Optional sidecar failures are
    /// deliberately contained as flat-scan fallback.
    pub fn from_admitted_v2_with_residual_sidecar_cache(
        source: &ValidatedFsviBytes,
        cache_dir: &Path,
    ) -> SearchResult<Self> {
        let index = Self::from_admitted_v2(source)?;
        let candidates = match index.exact_residual_sidecar_cache_candidates(cache_dir) {
            Ok(Some(candidates)) => candidates,
            // A cache directory that cannot be safely enumerated, or exhausts
            // its fixed discovery/comparison budget, is unavailable. Do not
            // build a large sidecar merely to discard it after an inevitable
            // publication failure; flat exact search is already the semantic
            // default. In this case publication is deliberately skipped.
            Ok(None) | Err(_) => return Ok(index),
        };
        // Candidate verification is a read-only path: a valid immutable
        // sidecar must remain reusable from a read-only cache root. Its full
        // source-derived comparison is necessary before attachment, but must
        // not require a publication capability that reuse never consumes.
        let mut derived = None;
        for candidate in candidates {
            if derived.is_none() {
                let sidecar = match index.build_exact_residual_sidecar() {
                    Ok(sidecar) => sidecar,
                    Err(_) => return Ok(index),
                };
                derived = Some(sidecar);
            }
            #[cfg(target_os = "linux")]
            let exactly_matches = {
                let Some(expected) = derived.as_ref() else {
                    return Ok(index);
                };
                exact_residual_sidecar_stream_matches_derived(&candidate, expected).unwrap_or(false)
            };
            #[cfg(not(target_os = "linux"))]
            let exactly_matches = false;
            if exactly_matches {
                let Some(sidecar) = derived.take() else {
                    return Ok(index);
                };
                let _ = index.exact_residual_sidecar.set(sidecar);
                return Ok(index);
            }
        }
        // No reusable candidate was attached. Reserve the create-only
        // publication primitive before a no-candidate derivation so an
        // unavailable optional cache cannot cause a discarded large build.
        // A corrupt header-matching candidate may already have required one
        // derivation for read-only authentication above; that work is never
        // repeated here.
        #[cfg(target_os = "linux")]
        let reserved_publication = {
            let candidate = match index.next_exact_residual_sidecar_cache_path(cache_dir) {
                Ok(candidate) => candidate,
                Err(_) => return Ok(index),
            };
            match acquire_exact_residual_sidecar_publication(&candidate) {
                Ok(capability) => capability,
                Err(_) => return Ok(index),
            }
        };
        let sidecar = match derived {
            Some(sidecar) => sidecar,
            None => match index.build_exact_residual_sidecar() {
                Ok(sidecar) => sidecar,
                Err(_) => return Ok(index),
            },
        };
        #[cfg(target_os = "linux")]
        match publish_exact_residual_sidecar_with_capability(reserved_publication, &sidecar) {
            Ok(ExactResidualPublication::Published) => {
                let _ = index.exact_residual_sidecar.set(sidecar);
                return Ok(index);
            }
            // A concurrent owner may have claimed the deterministic
            // generation-keyed name after capability acquisition. The
            // already-derived sidecar is safe to retry under a new name.
            Ok(ExactResidualPublication::DestinationExists) => {}
            Err(_) => return Ok(index),
        }
        for _ in 0..EXACT_RESIDUAL_CACHE_ATTEMPTS {
            let candidate = match index.next_exact_residual_sidecar_cache_path(cache_dir) {
                Ok(candidate) => candidate,
                Err(_) => break,
            };
            match publish_exact_residual_sidecar(&candidate, &sidecar) {
                Ok(ExactResidualPublication::Published) => {
                    let _ = index.exact_residual_sidecar.set(sidecar);
                    break;
                }
                Ok(ExactResidualPublication::DestinationExists) => continue,
                Err(_) => break,
            }
        }
        Ok(index)
    }

    fn exact_residual_generation_cache_prefix(&self) -> SearchResult<String> {
        let source = self.residual_source_binding.as_ref().ok_or_else(|| {
            residual_sidecar_error(
                "source_generation",
                "generation-keyed caching requires an admitted FSVI v2 source",
            )
        })?;
        let mut digest = Sha256::new();
        digest.update(EXACT_RESIDUAL_SIDECAR_MAGIC);
        digest.update(EXACT_RESIDUAL_SIDECAR_VERSION.to_le_bytes());
        digest.update(source.generation_fingerprint);
        digest.update(source.vector_content_digest);
        digest.update(source.ordered_live_docset_digest);
        digest.update(source.space_fingerprint);
        digest.update(
            u64::try_from(self.record_count())
                .map_err(|_| residual_sidecar_error("record_count", "does not fit u64"))?
                .to_le_bytes(),
        );
        digest.update(
            u64::try_from(self.dimension)
                .map_err(|_| residual_sidecar_error("dimension", "does not fit u64"))?
                .to_le_bytes(),
        );
        let mut prefix = String::new();
        prefix
            .try_reserve_exact(9 + EXACT_RESIDUAL_SIDECAR_DIGEST_BYTES * 2)
            .map_err(|_| residual_sidecar_error("cache_path", "allocation failed"))?;
        prefix.push_str("fsrs-v2-");
        const HEX: &[u8; 16] = b"0123456789abcdef";
        for byte in digest.finalize() {
            prefix.push(char::from(HEX[usize::from(byte >> 4)]));
            prefix.push(char::from(HEX[usize::from(byte & 0x0f)]));
        }
        prefix.push('-');
        Ok(prefix)
    }

    fn next_exact_residual_sidecar_cache_path(&self, cache_dir: &Path) -> SearchResult<PathBuf> {
        let prefix = self.exact_residual_generation_cache_prefix()?;
        let nonce = EXACT_RESIDUAL_CACHE_NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(cache_dir.join(format!("{prefix}{}-{nonce}.fsrs", std::process::id())))
    }

    #[cfg(target_os = "linux")]
    fn exact_residual_sidecar_cache_candidates(
        &self,
        cache_dir: &Path,
    ) -> SearchResult<Option<Vec<PathBuf>>> {
        // Prove the optional cache root itself is a no-follow directory before
        // deriving a large payload. This catches a missing path, regular-file
        // obstacle, or symlinked cache root while the fallback is still free.
        if open_exact_residual_sidecar_parent(&cache_dir.join(".fsrs-cache-probe")).is_err() {
            return Ok(None);
        }
        let Some(source) = self.residual_source_binding.as_ref() else {
            return Ok(None);
        };
        let prefix = self.exact_residual_generation_cache_prefix()?;
        let layout = match ExactResidualLayout::for_shape(self.record_count(), self.dimension) {
            Ok(layout) => layout,
            Err(_) => return Ok(None),
        };
        let comparison_limit = EXACT_RESIDUAL_CACHE_COMPARISON_CANDIDATE_LIMIT
            .min(EXACT_RESIDUAL_CACHE_COMPARISON_BYTE_BUDGET / layout.encoded_bytes);
        if comparison_limit == 0 {
            return Ok(None);
        }
        let mut entries = match std::fs::read_dir(cache_dir) {
            Ok(entries) => entries,
            Err(_) => return Ok(None),
        };
        let mut paths = Vec::new();
        paths
            .try_reserve_exact(EXACT_RESIDUAL_CACHE_DIRECTORY_ENTRY_LIMIT)
            .map_err(|_| residual_sidecar_error("cache", "directory-entry allocation failed"))?;
        for _ in 0..=EXACT_RESIDUAL_CACHE_DIRECTORY_ENTRY_LIMIT {
            let entry = match entries.next() {
                Some(Ok(entry)) => entry,
                Some(Err(_)) => return Ok(None),
                None => break,
            };
            if paths.len() == EXACT_RESIDUAL_CACHE_DIRECTORY_ENTRY_LIMIT {
                return Ok(None);
            }
            paths.push(entry.path());
        }
        // `read_dir` order is filesystem-defined. Sorting the bounded set
        // makes both candidate selection and the work-budget verdict stable.
        paths.sort_unstable();
        let mut candidates = Vec::new();
        candidates
            .try_reserve_exact(comparison_limit)
            .map_err(|_| residual_sidecar_error("cache", "candidate allocation failed"))?;
        let mut comparison_bytes = 0_usize;
        for candidate in paths {
            let Some(name) = candidate.file_name() else {
                return Ok(None);
            };
            let Some(name) = name.to_str() else {
                continue;
            };
            if !name.starts_with(&prefix) || !name.ends_with(".fsrs") {
                continue;
            }
            let header_matches = exact_residual_sidecar_header_matches_source(
                &candidate,
                source,
                self.record_count(),
                self.dimension,
            )
            .unwrap_or(false);
            if header_matches {
                let Some(next_comparison_bytes) =
                    comparison_bytes.checked_add(layout.encoded_bytes)
                else {
                    return Ok(None);
                };
                if candidates.len() == comparison_limit
                    || next_comparison_bytes > EXACT_RESIDUAL_CACHE_COMPARISON_BYTE_BUDGET
                {
                    return Ok(None);
                }
                comparison_bytes = next_comparison_bytes;
                candidates.push(candidate);
            }
        }
        Ok(Some(candidates))
    }

    #[cfg(not(target_os = "linux"))]
    fn exact_residual_sidecar_cache_candidates(
        &self,
        _cache_dir: &Path,
    ) -> SearchResult<Option<Vec<PathBuf>>> {
        Ok(None)
    }

    /// Shared load path: read every live row (and any WAL tail) of an opened
    /// index into memory, preserving the embedding-space identity the source
    /// declares. An admitted v2 source carries a validated space fingerprint;
    /// a legacy v1 source carries none, and that absence is kept as the typed
    /// `None` state — never papered over with a fabricated identity.
    fn from_open_index(index: &VectorIndex) -> SearchResult<Self> {
        let count = index.record_count();
        let dimension = index.dimension();
        let capacity = count.checked_add(index.wal_entries.len()).ok_or_else(|| {
            SearchError::InvalidConfig {
                field: "vectors".to_owned(),
                value: format!("main={count}, wal={}", index.wal_entries.len()),
                reason: "document capacity overflow".to_owned(),
            }
        })?;
        let scalar_capacity =
            capacity
                .checked_mul(dimension)
                .ok_or_else(|| SearchError::InvalidConfig {
                    field: "vectors".to_owned(),
                    value: format!("count={capacity}, dimension={dimension}"),
                    reason: "vector slab length overflow".to_owned(),
                })?;
        let mut doc_ids = Vec::new();
        doc_ids
            .try_reserve_exact(capacity)
            .map_err(|_| SearchError::InvalidConfig {
                field: "vectors".to_owned(),
                value: capacity.to_string(),
                reason: "document-id allocation failed".to_owned(),
            })?;
        let mut flat = Vec::new();
        flat.try_reserve_exact(scalar_capacity)
            .map_err(|_| SearchError::InvalidConfig {
                field: "vectors".to_owned(),
                value: scalar_capacity.to_string(),
                reason: "vector slab allocation failed".to_owned(),
            })?;

        for i in 0..count {
            if index.is_deleted(i) {
                continue;
            }
            doc_ids.push(fallible_doc_id_copy(index.doc_id_at(i)?)?);
            index.extend_vector_at_f16(i, &mut flat)?;
        }

        for entry in &index.wal_entries {
            doc_ids.push(fallible_doc_id_copy(&entry.doc_id)?);
            flat.extend(entry.embedding.iter().copied().map(half::f16::from_f32));
        }

        // Capture the source's embedding-space identity before the backing
        // `VectorIndex` drops (bd-9xuj T2-C3). On the admitted-v2 path this
        // is the validated space fingerprint from the artifact's identity
        // header; on the legacy v1 `open` path `identity_v2()` is
        // structurally `None`, which is preserved as the typed absent state —
        // never substituted with a fabricated identity.
        let space_fingerprint_hex = index
            .identity_v2()
            .map(|identity| crate::fingerprint_hex(&identity.space_fingerprint));
        // The attested bit derives from WHERE the identity came from
        // (bd-9xuj T2-C4-write): `identity_v2()` is populated exclusively by
        // the v2 header parse inside exact admission (`VectorIndex::open` is
        // strictly v1), so its presence here means the fingerprint above was
        // read out of the artifact's own validated header bytes.
        let space_identity_attested = index.identity_v2().is_some();
        // Every FSVI header (v1 and v2) carries the embedder id/revision
        // strings; retain them verbatim instead of discarding them at load
        // (bd-9xuj T2-C2). A v1 header's empty revision stays `Some("")` —
        // that is what the header says, not a fabrication.
        let embedder_id = Some(index.embedder_id().to_owned());
        let embedder_revision = Some(index.embedder_revision().to_owned());

        Ok(Self {
            doc_ids,
            vectors: flat,
            vectors_i8: OnceLock::new(),
            vectors_nibbles: OnceLock::new(),
            doc_id_hashes: OnceLock::new(),
            doc_id_index: OnceLock::new(),
            hash_to_pos: OnceLock::new(),
            space_fingerprint_hex,
            embedder_id,
            embedder_revision,
            space_identity_attested,
            dimension,
            residual_source_binding: None,
            exact_residual_sidecar: OnceLock::new(),
        })
    }

    /// Number of vectors in the index.
    #[must_use]
    pub const fn record_count(&self) -> usize {
        self.doc_ids.len()
    }

    /// Vector dimensionality.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Lowercase hex SHA-256 fingerprint of the embedding space this index's
    /// vectors were produced in, when known (bd-9xuj T2-C3).
    ///
    /// This is the index-side join key for
    /// [`frankensearch_core::BoundQueryEmbedding::verify_space_identity`]:
    /// a bound query embedding is admissible against this index exactly when
    /// its space fingerprint equals this value. `None` means the source
    /// carried no identity (legacy v1 FSVI file, or the identity-less
    /// [`Self::from_vectors`] constructor); that absence is a legal, typed
    /// state which downstream seams route as legacy-unidentified — it must
    /// never be papered over with a fabricated fingerprint.
    #[must_use]
    pub fn space_fingerprint_hex(&self) -> Option<&str> {
        self.space_fingerprint_hex.as_deref()
    }

    /// Whether this index's space identity is FSVI-v2-HEADER-attested rather
    /// than builder/caller-declared (bd-9xuj T2-C4-write, guards 2+8).
    ///
    /// `true` exactly when the index was loaded through
    /// [`Self::from_admitted_v2`] from an artifact whose validated v2 header
    /// carried the identity — i.e. the fingerprint in
    /// [`Self::space_fingerprint_hex`] was read out of the artifact's own
    /// bytes. A `Some` fingerprint with `false` here is a DECLARED identity
    /// ([`Self::from_vectors_with_identity`]): usable for diagnostics and
    /// query-side joins, but never sufficient for an attested-only admission
    /// seam such as the refresh identity-bound merge.
    #[must_use]
    pub const fn space_identity_is_attested(&self) -> bool {
        self.space_identity_attested
    }

    /// Whether this index currently has a source-derived exact-residual
    /// sidecar attached. `true` means the cache passed descriptor-bound,
    /// generation-bound, and bitwise re-derivation checks; it says nothing
    /// about whether a particular query can prune, because non-finite or
    /// overflow-risk queries deliberately retain the exact flat fallback.
    #[must_use]
    pub fn has_exact_residual_sidecar(&self) -> bool {
        self.exact_residual_sidecar.get().is_some()
    }

    /// Embedder id recorded by this index's source, when the source carried
    /// one (bd-9xuj T2-C2): the FSVI header string on the file-backed load
    /// paths, or the declared space's `logical_model_id` on
    /// [`Self::from_vectors_with_identity`].
    ///
    /// Diagnostics only. Compatibility joins on
    /// [`Self::space_fingerprint_hex`]; an id string must never admit or
    /// reject a pairing. `None` means the source declared nothing
    /// ([`Self::from_vectors`]) — typed absence, never a default.
    #[must_use]
    pub fn embedder_id(&self) -> Option<&str> {
        self.embedder_id.as_deref()
    }

    /// Embedder revision recorded by this index's source, under the same
    /// retention rules as [`Self::embedder_id`] (bd-9xuj T2-C2). A v1
    /// header's empty revision is preserved as `Some("")`, distinct from the
    /// `None` of a source with no header at all.
    #[must_use]
    pub fn embedder_revision(&self) -> Option<&str> {
        self.embedder_revision.as_deref()
    }

    /// Build and atomically publish a versioned exact-residual sidecar for this
    /// admitted FSVI v2 generation.  Publication is descriptor-relative,
    /// no-symlink, `O_TMPFILE` + descriptor-bound `linkat` no-replace, and
    /// parent-directory durable; it never replaces an already published
    /// artifact. A failure before `linkat` has no visible temporary path. A
    /// failure syncing the parent after link is reported and intentionally does
    /// not unlink the visible artifact, because that path may already be
    /// durable. It never mutates the active FSVI image or a prior sidecar.
    ///
    /// Legacy and caller-built in-memory indexes have no source-generation
    /// witness, so they are deliberately rejected instead of fabricating one.
    pub fn write_exact_residual_sidecar(&self, path: &Path) -> SearchResult<()> {
        #[cfg(target_os = "linux")]
        {
            let capability = acquire_exact_residual_sidecar_publication(path)?;
            let sidecar = match self.build_exact_residual_sidecar() {
                Ok(sidecar) => sidecar,
                Err(error) => {
                    drop(capability);
                    return Err(error);
                }
            };
            match publish_exact_residual_sidecar_with_capability(capability, &sidecar)? {
                ExactResidualPublication::Published => Ok(()),
                ExactResidualPublication::DestinationExists => Err(residual_sidecar_error(
                    "publish",
                    "destination already exists; immutable sidecars are never overwritten",
                )),
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = path;
            Err(residual_sidecar_error(
                "platform",
                "exact residual sidecar publication requires Linux descriptor APIs",
            ))
        }
    }

    /// Attempt to admit an exact-residual sidecar.  A missing, malformed, stale,
    /// mismatched, or already-replaced sidecar returns `false` and leaves the
    /// established f16 flat scan selected.  The non-error result is intentional:
    /// sidecars are an optional optimization, never part of semantic admission.
    pub fn try_open_exact_residual_sidecar(&self, path: &Path) -> SearchResult<bool> {
        let Some(source) = self.residual_source_binding.as_ref() else {
            return Ok(false);
        };
        #[cfg(target_os = "linux")]
        let header_matches = exact_residual_sidecar_header_matches_source(
            path,
            source,
            self.record_count(),
            self.dimension,
        )
        .unwrap_or(false);
        #[cfg(not(target_os = "linux"))]
        let header_matches = false;
        if !header_matches {
            return Ok(false);
        }
        // The header preflight avoids derivation for absent/stale cache entries.
        // Once it matches, compare every serialized field and whole-file digest
        // directly from one descriptor with fixed stack buffers. This holds only
        // the trusted derived sidecar, never a second 512 MiB raw byte image.
        let expected = match self.build_exact_residual_sidecar() {
            Ok(expected) => expected,
            Err(error) => return Err(error),
        };
        #[cfg(target_os = "linux")]
        let exactly_matches =
            exact_residual_sidecar_stream_matches_derived(path, &expected).unwrap_or(false);
        #[cfg(not(target_os = "linux"))]
        let exactly_matches = false;
        if !exactly_matches {
            return Ok(false);
        }
        Ok(self.exact_residual_sidecar.set(expected).is_ok())
    }

    /// Sidecar content is treated as an untrusted derived cache, even after its
    /// file digest and source-generation header match.  Re-derive the complete
    /// transform from the admitted f16 source and require bitwise equality before
    /// it is allowed to prune a candidate.  This makes a finite, rehashed payload
    /// mutation fail closed rather than merely look structurally well-formed.
    #[cfg(test)]
    fn admit_exact_residual_sidecar(&self, sidecar: ExactResidualSidecar) -> SearchResult<bool> {
        let Some(source) = self.residual_source_binding.as_ref() else {
            return Ok(false);
        };
        if !sidecar.is_bound_to(source, self.record_count(), self.dimension) {
            return Ok(false);
        }
        let expected = self.build_exact_residual_sidecar()?;
        if !sidecar.exactly_matches_derived(&expected) {
            return Ok(false);
        }
        // The source-derived value, not caller-provided transformed data, is
        // what becomes live after a successful equality proof.
        Ok(self.exact_residual_sidecar.set(expected).is_ok())
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn build_exact_residual_sidecar(&self) -> SearchResult<ExactResidualSidecar> {
        #[cfg(all(test, target_os = "linux"))]
        EXACT_RESIDUAL_SIDECAR_BUILD_COUNT.with(|count| {
            count.set(count.get().saturating_add(1));
        });
        let source = self.residual_source_binding.clone().ok_or_else(|| {
            residual_sidecar_error(
                "source_generation",
                "exact residual sidecars require an admitted FSVI v2 source generation",
            )
        })?;
        let count = self.record_count();
        let dimension = self.dimension;
        let layout = ExactResidualLayout::for_shape(count, dimension)?;

        let mut energy = try_filled_sidecar_vec(dimension, 0.0_f64, "energy")?;
        for row in 0..count {
            for (dimension_index, value) in self.vector_slice(row).iter().enumerate() {
                let value = value.to_f32();
                if !value.is_finite() {
                    return Err(residual_sidecar_error(
                        "vectors",
                        "source contains a non-finite f16 value",
                    ));
                }
                energy[dimension_index] += f64::from(value) * f64::from(value);
            }
        }
        // Keep the final persisted permutation as the working sort buffer so
        // derivation never peaks at both a usize dimension list and a second
        // u32 payload list. The format cap is far below u32::MAX.
        let mut permutation = Vec::new();
        permutation
            .try_reserve_exact(dimension)
            .map_err(|_| residual_sidecar_error("permutation", "allocation failed"))?;
        for dimension_index in 0..dimension {
            permutation.push(u32::try_from(dimension_index).map_err(|_| {
                residual_sidecar_error("permutation", "dimension does not fit u32")
            })?);
        }
        permutation.sort_unstable_by(|left, right| {
            // `for_shape` caps the dimension at 65_536, so each stored
            // permutation value is at most 65_535 and fits every supported
            // `usize` without a fallible sorting closure.
            let left = *left as usize;
            let right = *right as usize;
            energy[right]
                .total_cmp(&energy[left])
                .then_with(|| left.cmp(&right))
        });
        drop(energy);

        let mut centroids = try_filled_sidecar_vec(layout.centroid_len, 0.0_f32, "centroids")?;
        let mut residuals = try_filled_sidecar_vec(layout.residual_len, 0.0_f32, "residuals")?;
        let mut suffix_norms = try_filled_sidecar_vec(layout.suffix_len, 0.0_f32, "suffix_norms")?;
        let mut correction_norms =
            try_filled_sidecar_vec(layout.lane_len, 0.0_f32, "correction_norms")?;
        for group in 0..layout.groups {
            let group_start = group * EXACT_RESIDUAL_LANES;
            let active_lanes = (count - group_start).min(EXACT_RESIDUAL_LANES);
            for (transformed_dimension, &source_dimension) in permutation.iter().enumerate() {
                let source_dimension = usize::try_from(source_dimension).map_err(|_| {
                    residual_sidecar_error("permutation", "dimension does not fit this platform")
                })?;
                let sum: f64 = (0..active_lanes)
                    .map(|lane| {
                        f64::from(self.vector_slice(group_start + lane)[source_dimension].to_f32())
                    })
                    .sum();
                centroids[group * dimension + transformed_dimension] =
                    (sum / active_lanes.max(1) as f64) as f32;
            }
            for lane in 0..active_lanes {
                let residual_base = group * dimension * EXACT_RESIDUAL_LANES + lane;
                let mut correction_sum = 0.0_f64;
                for (transformed_dimension, &source_dimension) in permutation.iter().enumerate() {
                    let source_dimension = usize::try_from(source_dimension).map_err(|_| {
                        residual_sidecar_error(
                            "permutation",
                            "dimension does not fit this platform",
                        )
                    })?;
                    let original = self.vector_slice(group_start + lane)[source_dimension].to_f32();
                    let centroid = centroids[group * dimension + transformed_dimension];
                    let residual = original - centroid;
                    residuals[residual_base + transformed_dimension * EXACT_RESIDUAL_LANES] =
                        residual;
                    // Keep the entire construction remainder in f64, rather
                    // than only the rounded f32 subtraction remainder. This
                    // contains both the residual-construction rounding and
                    // the later `(centroid + residual)` reconstruction
                    // rounding, so the Cauchy term bounds the actual f16 row.
                    let correction = finite_f64_add(
                        finite_f64_add(f64::from(original), -f64::from(centroid)).ok_or_else(
                            || {
                                residual_sidecar_error(
                                    "correction_norms",
                                    "construction subtraction overflowed",
                                )
                            },
                        )?,
                        -f64::from(residual),
                    )
                    .ok_or_else(|| {
                        residual_sidecar_error(
                            "correction_norms",
                            "construction subtraction overflowed",
                        )
                    })?;
                    let correction_squared =
                        finite_f64_mul(correction, correction).ok_or_else(|| {
                            residual_sidecar_error(
                                "correction_norms",
                                "construction square overflowed",
                            )
                        })?;
                    correction_sum = finite_f64_add(correction_sum, correction_squared)
                        .ok_or_else(|| {
                            residual_sidecar_error(
                                "correction_norms",
                                "construction norm overflowed",
                            )
                        })?;
                }
                correction_norms[group * EXACT_RESIDUAL_LANES + lane] =
                    outward_f32_norm(correction_sum, "correction_norms")?;
                let mut suffix_sum = 0.0_f64;
                for block_index in (0..layout.blocks).rev() {
                    let start = block_index * EXACT_RESIDUAL_BLOCK;
                    let end = (start + EXACT_RESIDUAL_BLOCK).min(dimension);
                    for transformed_dimension in start..end {
                        let residual =
                            residuals[residual_base + transformed_dimension * EXACT_RESIDUAL_LANES];
                        suffix_sum += f64::from(residual) * f64::from(residual);
                    }
                    let suffix_offset =
                        (group * (layout.blocks + 1) + block_index) * EXACT_RESIDUAL_LANES + lane;
                    suffix_norms[suffix_offset] = outward_f32_norm(suffix_sum, "suffix_norms")?;
                }
            }
        }
        Ok(ExactResidualSidecar {
            source,
            count,
            dimension,
            block: EXACT_RESIDUAL_BLOCK,
            lanes: EXACT_RESIDUAL_LANES,
            permutation,
            centroids,
            residuals,
            suffix_norms,
            correction_norms,
        })
    }

    /// Get the document ID at position `index`.
    ///
    /// # Errors
    ///
    /// Returns error if index is out of bounds.
    pub fn doc_id_at(&self, index: usize) -> SearchResult<&str> {
        self.doc_ids
            .get(index)
            .map(String::as_str)
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "index".to_owned(),
                value: index.to_string(),
                reason: format!(
                    "index {} out of bounds (record_count = {})",
                    index,
                    self.doc_ids.len()
                ),
            })
    }

    /// Get the f16 vector slice at position `index`.
    fn vector_slice(&self, index: usize) -> &[f16] {
        let start = index * self.dimension;
        &self.vectors[start..start + self.dimension]
    }

    /// Brute-force cosine-similarity top-k search.
    ///
    /// Query must be pre-normalized. Uses f16→f32 SIMD dot product.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` when `query.len() != dimension`.
    pub fn search_top_k(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<Vec<VectorHit>> {
        self.search_top_k_with_params(query, limit, filter, SearchParams::default())
    }

    /// Compute the zero-signal census for this in-memory index.
    ///
    /// The in-memory index has no tombstones and no WAL, so `live_count`
    /// equals `record_count` and both WAL-derived states are structurally
    /// unreachable. O(n·dim); intended for lazy classification of empty
    /// results, not the hot path.
    #[must_use]
    pub fn zero_signal_state(&self) -> ZeroSignalState {
        let record_count = self.record_count();
        let mut usable_vector_count = 0usize;
        for row in 0..record_count {
            let start = row * self.dimension;
            let row_slice = &self.vectors[start..start + self.dimension];
            let mut norm_sq = 0.0_f32;
            let mut finite = true;
            for &value in row_slice {
                let value = value.to_f32();
                if !value.is_finite() {
                    finite = false;
                    break;
                }
                norm_sq += value * value;
            }
            if finite && norm_sq > 0.0 && norm_sq.is_finite() {
                usable_vector_count += 1;
            }
        }
        ZeroSignalState {
            record_count,
            live_count: record_count,
            tombstone_count: 0,
            wal_count: 0,
            usable_vector_count,
        }
    }

    /// Brute-force top-k with typed zero-signal classification.
    ///
    /// Mirrors [`VectorIndex::search_top_k_classified`] exactly (bd-tqhc
    /// requires the in-memory and file-backed paths to classify equivalent
    /// states identically): non-finite queries are rejected fail-closed and
    /// an empty result always carries a typed [`ZeroSignalReason`].
    ///
    /// # Errors
    ///
    /// Everything [`Self::search_top_k`] returns, plus
    /// [`SearchError::InvalidConfig`] for non-finite query vectors.
    pub fn search_top_k_classified(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<ClassifiedHits> {
        if query.len() != self.dimension {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension,
                found: query.len(),
            });
        }
        if limit == 0 {
            return Ok(ClassifiedHits::empty(
                ZeroSignalReason::CallerRequestedZeroK,
            ));
        }
        if query.iter().any(|value| !value.is_finite()) {
            return Err(SearchError::InvalidConfig {
                field: "query".to_owned(),
                value: "<contains non-finite values>".to_owned(),
                reason: "query vector must be finite".to_owned(),
            });
        }
        if query.iter().all(|&value| value == 0.0) {
            return Ok(ClassifiedHits::empty(ZeroSignalReason::ZeroNormQuery));
        }
        let hits = self.search_top_k(query, limit, filter)?;
        let zero_signal = hits.is_empty().then(|| {
            self.zero_signal_state()
                .empty_result_reason(filter.is_some())
        });
        Ok(ClassifiedHits { hits, zero_signal })
    }

    /// Brute-force top-k search with configurable parallelism.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` when `query.len() != dimension`.
    pub fn search_top_k_with_params(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
        params: SearchParams,
    ) -> SearchResult<Vec<VectorHit>> {
        if query.len() != self.dimension {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension,
                found: query.len(),
            });
        }
        let count = self.record_count();
        if limit == 0 || count == 0 {
            return Ok(Vec::new());
        }

        // Selective hash-addressable filter → gather the allow-set and exact-scan
        // only those positions (work ∝ |allow-set|, not corpus N). Bit-identical to
        // the per-document scan below.
        if let Some(hits) = self.try_gather_filtered(query, limit, filter, count)? {
            return Ok(hits);
        }

        let use_parallel = params.parallel_enabled && count >= params.parallel_threshold;

        // A residual sidecar is optional and is reachable only after complete
        // source-bound admission. It has a matching parallel group route, so
        // the default 10K–100K parallel search uses the structural eliminator
        // while retaining the incumbent f16 scorer and merge order. `k >= N`
        // cannot prune anything, so it deliberately takes the unchanged flat
        // route and never allocates a `usize::MAX`-sized sidecar heap.
        if limit < count
            && let Some(sidecar) = self.exact_residual_sidecar.get()
        {
            let outcome = if use_parallel {
                self.scan_exact_residual_sidecar_parallel(
                    query,
                    limit,
                    filter,
                    sidecar,
                    params.parallel_chunk_size.max(1),
                )?
            } else {
                self.scan_exact_residual_sidecar(query, limit, filter, sidecar)?
            };
            return self.resolve_heap(outcome.heap);
        }

        let chunk_size = params.parallel_chunk_size.max(1);

        let heap = if use_parallel {
            self.scan_parallel(query, limit, filter, chunk_size)?
        } else {
            self.scan_sequential(query, limit, filter)?
        };

        self.resolve_heap(heap)
    }

    /// Approximate top-k via an **int8 ADC two-pass** (`bd-b5wl`): an int8 pass-1
    /// over all vectors keeps the top `limit * candidate_multiplier` candidates,
    /// then an exact f16 rescore with the same deterministic selection as
    /// [`Self::search_top_k`] produces the final ranking.
    ///
    /// Results are **bit-identical** to [`Self::search_top_k`] whenever pass-1
    /// retains the true top-k (recall = 1). Measured ~1.4–1.5× faster than the
    /// parallel exact path across 10k–100k at `candidate_multiplier = 5` (int8 is
    /// half the bytes + an integer `widening_mul` MAC — see `docs/PERF_LEDGER.md`), at
    /// the cost of *approximate* recall.
    ///
    /// **Tuning `candidate_multiplier`:** recall@10 = 1.0 held down to `mult = 2`
    /// for well-separated (random) vectors, so `mult = 5` is a good default; the
    /// candidate budget (`limit * mult`) is the selection-overhead knob — smaller is
    /// faster. Clustered real embeddings have closer neighbours, so re-measure recall
    /// on a representative corpus and raise `mult` if needed.
    ///
    /// Falls back to the exact path when the int8 slab is unavailable.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` when `query.len() != dimension`.
    pub fn search_top_k_int8_two_pass(
        &self,
        query: &[f32],
        limit: usize,
        candidate_multiplier: usize,
    ) -> SearchResult<Vec<VectorHit>> {
        self.search_top_k_int8_two_pass_filtered(query, limit, candidate_multiplier, None)
    }

    /// int8 ADC two-pass with an optional [`SearchFilter`]. Pass-1 pre-screens each
    /// vector by its precomputed `doc_id` hash (the same `matches_doc_id_hash` path
    /// the exact scan uses), so **filtered** large-N searches get the int8 speedup
    /// instead of falling back to the exact scan. The result matches the exact
    /// filtered top-k whenever pass-1 retains the true filtered top-k.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` when `query.len() != dimension`.
    pub fn search_top_k_int8_two_pass_filtered(
        &self,
        query: &[f32],
        limit: usize,
        candidate_multiplier: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<Vec<VectorHit>> {
        if query.len() != self.dimension {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension,
                found: query.len(),
            });
        }
        let count = self.record_count();
        if limit == 0 || count == 0 {
            return Ok(Vec::new());
        }
        let candidate_count = limit.saturating_mul(candidate_multiplier.max(1)).min(count);
        // Full-recall short-cut: when the candidate budget already covers every
        // vector (`limit·mult ≥ count`, e.g. `limit_all` or a large limit on a small
        // corpus), pass-1 prunes NOTHING — the int8 scan, the size-N candidate heap,
        // the query quantize, and the slab build are all pure overhead, and pass-2
        // would rescore all N regardless. Delegate to the exact f16 single pass.
        // **Bit-identical by construction:** the two-pass equals `search_top_k`
        // whenever pass-1 retains the true top-k (its own doc-contract), and here it
        // retains *every* candidate. `limit.min(count)` avoids a `usize::MAX`-sized
        // heap and returns the same `min(limit, count)` hits.
        if candidate_count >= count {
            return self.search_top_k(query, limit.min(count), filter);
        }
        // Selective hash-addressable filter → exact gather of the allow-set. The
        // gather scans only `|allow-set|` vectors (vs all N for the int8 pass-1) and
        // is *exact* f16, so it is strictly more accurate than this approximate
        // two-pass while doing far less work. Falls through when the filter is not a
        // selective allow-list.
        if let Some(hits) = self.try_gather_filtered(query, limit, filter, count)? {
            return Ok(hits);
        }
        let query_i8 = quantize_i8_query(query);
        // Build the int8 slab once, on first use — exact-only callers never pay the
        // O(N·d) quantization or its `N·d`-byte footprint at construction time.
        let vectors_i8 = self
            .vectors_i8
            .get_or_init(|| quantize_i8_slab(&self.vectors));

        // Pass 1: parallel **bounded-heap** int8 scan — each chunk keeps only its
        // top `candidate_count` (never materializing all N scores, unlike a
        // collect-all + select), then merge. This mirrors the exact path's
        // structure so the 3× int8 dot win is not eaten by selection overhead.
        // int8 dots peak well below 2^24 for realistic dims, so `i32 as f32` is
        // exact and preserves the candidate ranking + deterministic index tie-break.
        // Match the exact path's chunking so pass-1 uses all cores, not ~2.
        let chunk_size = PARALLEL_CHUNK_SIZE;
        let chunk_count = count.div_ceil(chunk_size);
        // When filtering, build the precomputed doc_id-hash slab once here (before
        // the parallel section) so pass-1 pre-screens by a hash lookup instead of
        // re-hashing each doc_id string per vector.
        let doc_id_hashes = filter.map(|_| self.doc_id_hashes());
        let partials: SearchResult<Vec<BinaryHeap<HeapEntry>>> = (0..chunk_count)
            .into_par_iter()
            .map(|chunk_index| {
                let start = chunk_index * chunk_size;
                let end = (start + chunk_size).min(count);
                let mut heap = bounded_heap(candidate_count.min(end - start), "int8_partial_heap")?;
                let mut cutoff = f32::NEG_INFINITY;
                for index in start..end {
                    if let Some(f) = filter {
                        let passed = doc_id_hashes
                            .and_then(|h| f.matches_doc_id_hash(h[index], None))
                            .unwrap_or_else(|| f.matches(&self.doc_ids[index], None));
                        if !passed {
                            continue;
                        }
                    }
                    let offset = index * self.dimension;
                    let stored = &vectors_i8[offset..offset + self.dimension];
                    let score = dot_i8_i8(stored, &query_i8) as f32;
                    // Skip the insert_candidate call for scores that cannot enter the
                    // full bounded heap — the same cutoff fast-path scan_range uses,
                    // which the int8 pass-1 previously lacked. Result is unchanged
                    // (a sub-cutoff score never makes the bounded heap anyway).
                    if heap.len() < candidate_count || score_key(score) >= cutoff {
                        insert_candidate(&mut heap, HeapEntry::new(index, score), candidate_count);
                        if heap.len() >= candidate_count
                            && let Some(&worst) = heap.peek()
                        {
                            cutoff = score_key(worst.score);
                        }
                    }
                }
                Ok(heap)
            })
            .collect();
        let candidate_heap = merge_partial_heaps(partials?, candidate_count)?;

        // Pass 2: exact f16 rescore of the candidates through the SAME bounded-heap
        // selection + tie-break as the exact path, so the final order matches
        // `search_top_k` exactly whenever pass-1 retained the true top-k.
        let mut heap = bounded_heap(limit.min(candidate_heap.len()), "int8_rescore_heap")?;
        for candidate in candidate_heap {
            let score = dot_product_f16_f32(self.vector_slice(candidate.index), query)?;
            insert_candidate(&mut heap, HeapEntry::new(candidate.index, score), limit);
        }
        self.resolve_heap(heap)
    }

    /// 4-bit (16-level) two-pass exact top-k — the in-memory twin of the FSVI
    /// `search_top_k_4bit_two_pass`. A parallel pass-1 over a packed signed-4-bit
    /// slab (`dim/2` bytes/vector — half the int8 slab) via the fused
    /// `dot_packed_4bit` kernel keeps the top `k·mult`, then an exact f16 rescore
    /// selects the final top-k. 16 levels are lossless at mult≈5 on realistic
    /// clustered data; the result matches the exact top-k whenever pass-1 retains it.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` when `query.len() != dimension`.
    pub fn search_top_k_4bit_two_pass(
        &self,
        query: &[f32],
        limit: usize,
        candidate_multiplier: usize,
    ) -> SearchResult<Vec<VectorHit>> {
        self.search_top_k_4bit_two_pass_filtered(query, limit, candidate_multiplier, None)
    }

    /// 4-bit two-pass with an optional [`SearchFilter`]. Pass-1 pre-screens each
    /// vector by its precomputed `doc_id` hash (the same path the int8 two-pass and
    /// exact scan use), so **filtered** searches keep the 4-bit speedup. Result
    /// matches the exact filtered top-k whenever pass-1 retains the true filtered top-k.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` when `query.len() != dimension`.
    pub fn search_top_k_4bit_two_pass_filtered(
        &self,
        query: &[f32],
        limit: usize,
        candidate_multiplier: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<Vec<VectorHit>> {
        if query.len() != self.dimension {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension,
                found: query.len(),
            });
        }
        let count = self.record_count();
        if limit == 0 || count == 0 {
            return Ok(Vec::new());
        }
        let candidate_count = limit.saturating_mul(candidate_multiplier.max(1)).min(count);
        // Full-recall short-cut (see the int8 two-pass for the full rationale): when
        // `limit·mult ≥ count`, pass-1 prunes nothing, so the 4-bit scan, size-N
        // heap, query nibble-prep, and slab build are pure overhead — delegate to the
        // exact f16 pass. Bit-identical by construction (pass-1 retains every
        // candidate ⇒ two-pass == `search_top_k`).
        if candidate_count >= count {
            return self.search_top_k(query, limit.min(count), filter);
        }
        // Selective hash-addressable filter → exact gather of the allow-set (see the
        // int8 two-pass for the rationale): scans only `|allow-set|` vectors and is
        // exact f16, so it is strictly more accurate than this approximate 4-bit
        // two-pass while doing far less work.
        if let Some(hits) = self.try_gather_filtered(query, limit, filter, count)? {
            return Ok(hits);
        }
        // Decode the (loop-invariant) query nibbles once, not per stored vector.
        let query_prepared = prepare_4bit_query(&pack_4bit_query(query));
        let bytes_per_vector = self.dimension.div_ceil(2);
        let nibbles = self
            .vectors_nibbles
            .get_or_init(|| pack_4bit_slab(&self.vectors, self.dimension));

        // Pass 1: parallel bounded-heap 4-bit scan (same chunking + cutoff fast-path
        // + filter pre-screen as the int8 two-pass).
        let chunk_size = PARALLEL_CHUNK_SIZE;
        let chunk_count = count.div_ceil(chunk_size);
        let doc_id_hashes = filter.map(|_| self.doc_id_hashes());
        let partials: SearchResult<Vec<BinaryHeap<HeapEntry>>> = (0..chunk_count)
            .into_par_iter()
            .map(|chunk_index| {
                let start = chunk_index * chunk_size;
                let end = (start + chunk_size).min(count);
                let mut heap = bounded_heap(candidate_count.min(end - start), "4bit_partial_heap")?;
                let mut cutoff = f32::NEG_INFINITY;
                for index in start..end {
                    if let Some(f) = filter {
                        let passed = doc_id_hashes
                            .and_then(|h| f.matches_doc_id_hash(h[index], None))
                            .unwrap_or_else(|| f.matches(&self.doc_ids[index], None));
                        if !passed {
                            continue;
                        }
                    }
                    let offset = index * bytes_per_vector;
                    let stored = &nibbles[offset..offset + bytes_per_vector];
                    let score = dot_4bit_prepared(stored, &query_prepared) as f32;
                    if heap.len() < candidate_count || score_key(score) >= cutoff {
                        insert_candidate(&mut heap, HeapEntry::new(index, score), candidate_count);
                        if heap.len() >= candidate_count
                            && let Some(&worst) = heap.peek()
                        {
                            cutoff = score_key(worst.score);
                        }
                    }
                }
                Ok(heap)
            })
            .collect();
        let candidate_heap = merge_partial_heaps(partials?, candidate_count)?;

        // Pass 2: exact f16 rescore of the candidates (same selection + tie-break).
        let mut heap = bounded_heap(limit.min(candidate_heap.len()), "4bit_rescore_heap")?;
        for candidate in candidate_heap {
            let score = dot_product_f16_f32(self.vector_slice(candidate.index), query)?;
            insert_candidate(&mut heap, HeapEntry::new(candidate.index, score), limit);
        }
        self.resolve_heap(heap)
    }

    #[allow(clippy::cast_precision_loss)]
    fn scan_exact_residual_sidecar(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
        sidecar: &ExactResidualSidecar,
    ) -> SearchResult<ResidualScanOutcome> {
        // The ordinary flat scan has established behaviour for non-finite
        // queries.  Do not reinterpret it through interval arithmetic: retain
        // that behaviour verbatim by declining the optional route.
        if query.iter().any(|value| !value.is_finite()) {
            return Ok(ResidualScanOutcome {
                heap: self.scan_sequential(query, limit, filter)?,
                census: ResidualPruningCensus::default(),
            });
        }
        let transformed = match ResidualQueryTransform::from_query(query, sidecar) {
            Ok(transformed) => transformed,
            // The transform and suffix norms belong exclusively to the
            // optional cache. Any preparation failure must be indistinguishable
            // from no cache: run the incumbent sequential exact scan.
            Err(_) => {
                return Ok(ResidualScanOutcome {
                    heap: self.scan_sequential(query, limit, filter)?,
                    census: ResidualPruningCensus::default(),
                });
            }
        };
        if !transformed.f32_flat_envelope_is_finite {
            return Ok(ResidualScanOutcome {
                heap: self.scan_sequential(query, limit, filter)?,
                census: ResidualPruningCensus::default(),
            });
        }
        self.scan_exact_residual_sidecar_groups(
            query,
            limit,
            filter,
            sidecar,
            &transformed,
            0,
            sidecar.group_count(),
            true,
        )
    }

    fn scan_exact_residual_sidecar_parallel(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
        sidecar: &ExactResidualSidecar,
        row_chunk_size: usize,
    ) -> SearchResult<ResidualScanOutcome> {
        if query.iter().any(|value| !value.is_finite()) {
            return Ok(ResidualScanOutcome {
                heap: self.scan_parallel(query, limit, filter, row_chunk_size)?,
                census: ResidualPruningCensus::default(),
            });
        }
        let transformed = match ResidualQueryTransform::from_query(query, sidecar) {
            Ok(transformed) => transformed,
            // See the sequential route: cache-only preparation can never
            // change the result or error surface of the parallel exact scan.
            Err(_) => {
                return Ok(ResidualScanOutcome {
                    heap: self.scan_parallel(query, limit, filter, row_chunk_size)?,
                    census: ResidualPruningCensus::default(),
                });
            }
        };
        if !transformed.f32_flat_envelope_is_finite {
            return Ok(ResidualScanOutcome {
                heap: self.scan_parallel(query, limit, filter, row_chunk_size)?,
                census: ResidualPruningCensus::default(),
            });
        }
        let groups = sidecar.group_count();
        if limit == 0 || groups == 0 {
            return Ok(ResidualScanOutcome {
                heap: BinaryHeap::new(),
                census: ResidualPruningCensus::default(),
            });
        }
        // Probe exactly once for this query before partitioning work across
        // Rayon. Per-chunk probes multiply the nonselective penalty by the
        // chunk count; this shared decision either enables pruning globally or
        // takes the incumbent exact parallel scan for the whole query.
        let probe_end = groups.min(EXACT_RESIDUAL_ADAPTIVE_PROBE_GROUPS);
        let probe = self.scan_exact_residual_sidecar_groups(
            query,
            limit,
            filter,
            sidecar,
            &transformed,
            0,
            probe_end,
            false,
        )?;
        if probe_end < groups && probe.census.lanes_pruned == 0 {
            let mut census = probe.census;
            census.flat_fallback_rows = census.flat_fallback_rows.saturating_add(sidecar.count);
            census.adaptive_fallbacks = census.adaptive_fallbacks.saturating_add(1);
            return Ok(ResidualScanOutcome {
                heap: self.scan_parallel(query, limit, filter, row_chunk_size)?,
                census,
            });
        }
        let groups_per_chunk = row_chunk_size.div_ceil(sidecar.lanes).max(1);
        let remaining_groups = groups.saturating_sub(probe_end);
        let chunk_count = remaining_groups.div_ceil(groups_per_chunk);
        let partials: SearchResult<Vec<ResidualScanOutcome>> = (0..chunk_count)
            .into_par_iter()
            .map(|chunk_index| {
                let first_group = probe_end + chunk_index * groups_per_chunk;
                let end_group = (first_group + groups_per_chunk).min(groups);
                self.scan_exact_residual_sidecar_groups(
                    query,
                    limit,
                    filter,
                    sidecar,
                    &transformed,
                    first_group,
                    end_group,
                    false,
                )
            })
            .collect();
        let mut census = probe.census;
        let mut heaps = Vec::new();
        heaps
            .try_reserve_exact(
                chunk_count
                    .checked_add(1)
                    .ok_or_else(|| residual_sidecar_error("parallel", "chunk count overflow"))?,
            )
            .map_err(|_| residual_sidecar_error("parallel", "partial heap allocation failed"))?;
        heaps.push(probe.heap);
        for partial in partials? {
            census.merge(partial.census);
            heaps.push(partial.heap);
        }
        Ok(ResidualScanOutcome {
            heap: merge_partial_heaps(heaps, limit)?,
            census,
        })
    }

    #[allow(clippy::cast_precision_loss)]
    fn scan_exact_residual_sidecar_groups(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
        sidecar: &ExactResidualSidecar,
        transformed: &ResidualQueryTransform,
        first_group: usize,
        end_group: usize,
        allow_adaptive_fallback: bool,
    ) -> SearchResult<ResidualScanOutcome> {
        let capped_end_group = end_group.min(sidecar.group_count());
        if limit == 0 || first_group >= capped_end_group {
            return Ok(ResidualScanOutcome {
                heap: BinaryHeap::new(),
                census: ResidualPruningCensus::default(),
            });
        }
        let range_start = first_group.saturating_mul(sidecar.lanes).min(sidecar.count);
        let range_end = capped_end_group
            .saturating_mul(sidecar.lanes)
            .min(sidecar.count);
        let rows_in_range = range_end.saturating_sub(range_start);
        let mut heap = bounded_heap(limit.min(rows_in_range), "residual_heap")?;
        let mut census = ResidualPruningCensus::default();
        let doc_id_hashes = filter.map(|_| self.doc_id_hashes());
        for group in first_group..capped_end_group {
            // When the first bounded probe cannot prune even one lane, retain
            // the exact heap accumulated so far and finish this work unit with
            // the incumbent scanner. This avoids paying residual arithmetic for
            // a query/cutoff geometry where this cache is not selective.
            if allow_adaptive_fallback
                && census.groups_scanned >= EXACT_RESIDUAL_ADAPTIVE_PROBE_GROUPS
                && census.lanes_pruned == 0
            {
                let fallback_start = group * sidecar.lanes;
                let fallback_end = (capped_end_group * sidecar.lanes).min(sidecar.count);
                let fallback =
                    self.scan_range(fallback_start, fallback_end, query, limit, filter)?;
                for candidate in fallback {
                    insert_candidate(&mut heap, candidate, limit);
                }
                census.flat_fallback_rows = census
                    .flat_fallback_rows
                    .saturating_add(fallback_end.saturating_sub(fallback_start));
                census.adaptive_fallbacks = census.adaptive_fallbacks.saturating_add(1);
                break;
            }

            census.groups_scanned = census.groups_scanned.saturating_add(1);
            let group_start = group * sidecar.lanes;
            let active_lanes = (sidecar.count - group_start).min(sidecar.lanes);
            let mut survives = [false; EXACT_RESIDUAL_LANES];
            for lane in 0..active_lanes {
                let index = group_start + lane;
                let passed = filter.map_or(true, |current_filter| {
                    doc_id_hashes
                        .and_then(|hashes| current_filter.matches_doc_id_hash(hashes[index], None))
                        .unwrap_or_else(|| current_filter.matches(&self.doc_ids[index], None))
                });
                survives[lane] = passed;
                if passed {
                    census.eligible_lanes = census.eligible_lanes.saturating_add(1);
                }
            }
            if !survives[..active_lanes].iter().any(|&live| live) {
                continue;
            }

            let centroid =
                &sidecar.centroids[group * sidecar.dimension..(group + 1) * sidecar.dimension];
            let mut centroid_dot = 0.0_f64;
            let mut centroid_norm_sq = 0.0_f64;
            for (&query_value, &centroid_value) in transformed.transformed.iter().zip(centroid) {
                centroid_dot += f64::from(query_value) * f64::from(centroid_value);
                centroid_norm_sq += f64::from(centroid_value) * f64::from(centroid_value);
            }
            // Like the transformed query norm, this enters a multiplicative
            // upper-bound term. Round away from zero before it reaches the
            // pruning predicate.
            let centroid_norm =
                outward_f64_norm_from_square_sum(centroid_norm_sq, sidecar.dimension)
                    .unwrap_or(f64::INFINITY);
            let mut partial = [0.0_f64; EXACT_RESIDUAL_LANES];
            for block_index in 0..sidecar.block_count() {
                if heap.len() >= limit {
                    let cutoff = f64::from(
                        heap.peek()
                            .expect("a full bounded heap has a worst candidate")
                            .score,
                    );
                    for lane in 0..active_lanes {
                        if !survives[lane] {
                            continue;
                        }
                        let suffix_offset = (group * (sidecar.block_count() + 1) + block_index)
                            * sidecar.lanes
                            + lane;
                        let residual_norm = f64::from(sidecar.suffix_norms[suffix_offset]);
                        let correction_norm =
                            f64::from(sidecar.correction_norms[group * sidecar.lanes + lane]);
                        let upper_bound = residual_lane_upper_bound(
                            sidecar,
                            transformed,
                            block_index,
                            centroid_dot,
                            centroid_norm,
                            residual_norm,
                            correction_norm,
                            partial[lane],
                        );
                        if upper_bound.is_finite() && upper_bound < cutoff {
                            survives[lane] = false;
                            census.lanes_pruned = census.lanes_pruned.saturating_add(1);
                        }
                    }
                    if !survives[..active_lanes].iter().any(|&live| live) {
                        break;
                    }
                }
                let start = block_index * sidecar.block;
                let end = (start + sidecar.block).min(sidecar.dimension);
                for transformed_dimension in start..end {
                    let query_value = f64::from(transformed.transformed[transformed_dimension]);
                    let base = group * sidecar.dimension * sidecar.lanes
                        + transformed_dimension * sidecar.lanes;
                    for lane in 0..active_lanes {
                        if survives[lane] {
                            partial[lane] +=
                                query_value * f64::from(sidecar.residuals[base + lane]);
                        }
                    }
                }
            }
            // The sidecar is an eliminator, never a scorer.  Survivors use the
            // exact incumbent operation, which makes the visible score and total
            // order identical to the flat path rather than merely numerically close.
            for lane in 0..active_lanes {
                if !survives[lane] {
                    continue;
                }
                let index = group_start + lane;
                let score = dot_product_f16_f32(self.vector_slice(index), query)?;
                census.exact_sidecar_scores = census.exact_sidecar_scores.saturating_add(1);
                let cutoff = heap
                    .peek()
                    .map_or(f32::NEG_INFINITY, |entry| score_key(entry.score));
                if heap.len() < limit || score_key(score) >= cutoff {
                    insert_candidate(&mut heap, HeapEntry::new(index, score), limit);
                }
            }
        }
        Ok(ResidualScanOutcome { heap, census })
    }

    fn scan_sequential(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        self.scan_range(0, self.record_count(), query, limit, filter)
    }

    fn scan_parallel(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
        chunk_size: usize,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        let count = self.record_count();
        let chunk_count = count.div_ceil(chunk_size);
        let partial_heaps: SearchResult<Vec<BinaryHeap<HeapEntry>>> = (0..chunk_count)
            .into_par_iter()
            .map(|chunk_index| {
                let start = chunk_index * chunk_size;
                let end = (start + chunk_size).min(count);
                self.scan_range(start, end, query, limit, filter)
            })
            .collect();

        merge_partial_heaps(partial_heaps?, limit)
    }

    /// Lazily-built FNV-1a hashes of every `doc_id` (matches the hash
    /// `BitsetFilter` computes), so the filtered scan can pre-screen by hash via
    /// `SearchFilter::matches_doc_id_hash` instead of re-hashing each `doc_id`
    /// string per vector. Built once on first filtered search.
    fn doc_id_hashes(&self) -> &[u64] {
        self.doc_id_hashes.get_or_init(|| {
            self.doc_ids
                .iter()
                .map(|id| fnv1a_hash(id.as_bytes()))
                .collect()
        })
    }

    /// Lazily-built `doc_id_hash → position` map for the selective-filter gather
    /// fast-path. Returns `None` when two `doc_ids` hash to the same value (the map
    /// would not be a bijection, so a gather could miss a colliding position the
    /// per-document scan would visit) — callers then fall back to the full scan,
    /// preserving exact results. Built once on first selective-filter search.
    fn hash_to_pos(&self) -> Option<&HashMap<u64, usize, BuildIdentityHasherU64>> {
        self.hash_to_pos
            .get_or_init(|| {
                let hashes = self.doc_id_hashes();
                let mut map =
                    HashMap::with_capacity_and_hasher(hashes.len(), BuildIdentityHasherU64);
                for (pos, &h) in hashes.iter().enumerate() {
                    if map.insert(h, pos).is_some() {
                        // Hash collision (or duplicate doc_id): the map can hold
                        // only one position per hash, so disable the fast path.
                        return None;
                    }
                }
                Some(map)
            })
            .as_ref()
    }

    /// Exact f16 top-k over an explicit list of candidate positions (the
    /// selective-filter gather fast-path). Parallel above `PARALLEL_CHUNK_SIZE`
    /// positions, serial below (tiny allow-sets don't amortize rayon overhead). The
    /// parallel path scans disjoint position chunks into per-chunk bounded heaps and
    /// merges by the `(score, index)` total order — order-independent, exactly like
    /// [`Self::scan_parallel`] — so it is bit-identical to the serial gather, to the
    /// per-document scan, and across thread counts.
    fn scan_gather(
        &self,
        positions: &[usize],
        query: &[f32],
        limit: usize,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        if positions.len() > PARALLEL_CHUNK_SIZE {
            let partials: SearchResult<Vec<BinaryHeap<HeapEntry>>> = positions
                .par_chunks(PARALLEL_CHUNK_SIZE)
                .map(|chunk| self.gather_range(chunk, query, limit))
                .collect();
            return merge_partial_heaps(partials?, limit);
        }
        self.gather_range(positions, query, limit)
    }

    /// Bounded-heap exact f16 scan over one slice of candidate positions. Mirrors
    /// [`Self::scan_range`]'s cutoff fast-path, minus the per-document filter probe
    /// (membership was already decided when the positions were gathered).
    fn gather_range(
        &self,
        positions: &[usize],
        query: &[f32],
        limit: usize,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        let mut heap = bounded_heap(limit.min(positions.len()), "gather_heap")?;
        let mut cutoff = f32::NEG_INFINITY;
        for &index in positions {
            let stored = self.vector_slice(index);
            let score = dot_product_f16_f32(stored, query)?;
            if heap.len() < limit || score_key(score) >= cutoff {
                insert_candidate(&mut heap, HeapEntry::new(index, score), limit);
                if heap.len() >= limit
                    && let Some(&worst) = heap.peek()
                {
                    cutoff = score_key(worst.score);
                }
            }
        }
        Ok(heap)
    }

    /// Try the selective-filter gather fast-path: when `filter` is a
    /// hash-addressable allow-list whose size is below
    /// `count / GATHER_SELECTIVITY_DIVISOR`, gather the allowed positions and exact
    /// f16-scan only those. Returns `Some(hits)` when the fast-path applied,
    /// `None` to fall through to the per-document scan. Bit-identical: the gathered
    /// passing set equals `{ pos : doc_id_hash[pos] ∈ allow-set }`, the same set the
    /// scan keeps, and both rank by the `(score, index)` total order.
    fn try_gather_filtered(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
        count: usize,
    ) -> SearchResult<Option<Vec<VectorHit>>> {
        let Some(f) = filter else {
            return Ok(None);
        };
        let Some(allowed) = f.candidate_hashes() else {
            return Ok(None);
        };
        // Selectivity gate: gather pays a map lookup + sort per allowed hash, so it
        // only wins when the allow-set is a small fraction of the corpus. Above the
        // crossover the per-document scan (which skips the gather's setup) is faster.
        if allowed.len().saturating_mul(GATHER_SELECTIVITY_DIVISOR) >= count {
            return Ok(None);
        }
        let Some(map) = self.hash_to_pos() else {
            return Ok(None);
        };
        let mut positions: Vec<usize> =
            allowed.iter().filter_map(|h| map.get(h).copied()).collect();
        // Ascending position order → sequential slab access (cache-friendly); not
        // required for correctness (the heap's total order is position-independent).
        positions.sort_unstable();
        let heap = self.scan_gather(&positions, query, limit)?;
        Ok(Some(self.resolve_heap(heap)?))
    }

    /// Bench-only: forced per-document filtered scan (the gather-fast-path baseline).
    /// Bypasses [`Self::try_gather_filtered`] so the A/B measures the old behavior.
    #[doc(hidden)]
    pub fn bench_scan_filtered(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<Vec<VectorHit>> {
        let count = self.record_count();
        if limit == 0 || count == 0 {
            return Ok(Vec::new());
        }
        let params = SearchParams::default();
        let use_parallel = params.parallel_enabled && count >= params.parallel_threshold;
        let heap = if use_parallel {
            self.scan_parallel(query, limit, filter, params.parallel_chunk_size.max(1))?
        } else {
            self.scan_sequential(query, limit, filter)?
        };
        self.resolve_heap(heap)
    }

    /// Bench-only: forced gather over a hash-addressable allow-set, ignoring the
    /// selectivity gate (so the crossover can be measured directly).
    #[doc(hidden)]
    pub fn bench_gather_filtered(
        &self,
        query: &[f32],
        limit: usize,
        filter: &dyn SearchFilter,
    ) -> SearchResult<Vec<VectorHit>> {
        let allowed = filter
            .candidate_hashes()
            .expect("bench_gather_filtered requires a hash-addressable allow-set");
        let map = self
            .hash_to_pos()
            .expect("bench_gather_filtered requires a bijective hash→pos map");
        let mut positions: Vec<usize> =
            allowed.iter().filter_map(|h| map.get(h).copied()).collect();
        positions.sort_unstable();
        let heap = self.scan_gather(&positions, query, limit)?;
        self.resolve_heap(heap)
    }

    /// O(1) `doc_id → position` lookup via a lazily-built map, replacing the O(N)
    /// `doc_ids.iter().position(...)` linear scan. First-insert-wins, matching
    /// `position`'s first-match semantics for any (non-canonical) duplicate ids.
    /// Built once on first lookup (the per-hit quality-rerank path); search-only
    /// callers never pay the `O(N)` build or its footprint. The map hashes with
    /// `ahash` (not SipHash): every quality query pays one hash of each hit's
    /// `doc_id` here, and `ahash` cuts the rescore loop ~1.27–1.48× at 128–300
    /// candidates (see `quality_rescore_hasher_ab`). Semantics are unchanged.
    fn index_of_doc_id(&self, doc_id: &str) -> Option<usize> {
        self.doc_id_index
            .get_or_init(|| {
                let mut map = AHashMap::with_capacity(self.doc_ids.len());
                for (i, id) in self.doc_ids.iter().enumerate() {
                    map.entry(id.clone()).or_insert(i);
                }
                map
            })
            .get(doc_id)
            .copied()
    }

    fn scan_range(
        &self,
        start: usize,
        end: usize,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        let max_elements = end.saturating_sub(start);
        let mut heap = bounded_heap(limit.min(max_elements), "scan_heap")?;
        let mut cutoff = f32::NEG_INFINITY;

        // When filtering, pre-screen by precomputed doc_id hash (one HashSet lookup)
        // instead of re-hashing the doc_id string per vector; fall back to the
        // string/metadata path only when the filter can't decide by hash.
        let doc_id_hashes = filter.map(|_| self.doc_id_hashes());

        for index in start..end {
            if let Some(f) = filter {
                let passed = doc_id_hashes
                    .and_then(|h| f.matches_doc_id_hash(h[index], None))
                    .unwrap_or_else(|| f.matches(&self.doc_ids[index], None));
                if !passed {
                    continue;
                }
            }
            let stored = self.vector_slice(index);
            let score = dot_product_f16_f32(stored, query)?;
            if heap.len() < limit || score_key(score) >= cutoff {
                insert_candidate(&mut heap, HeapEntry::new(index, score), limit);
                if heap.len() >= limit
                    && let Some(&worst) = heap.peek()
                {
                    cutoff = score_key(worst.score);
                }
            }
        }
        Ok(heap)
    }

    fn resolve_heap(&self, heap: BinaryHeap<HeapEntry>) -> SearchResult<Vec<VectorHit>> {
        if heap.is_empty() {
            return Ok(Vec::new());
        }
        let mut winners = heap.into_vec();
        // In-memory limit_all (`limit >= count`) builds a count-sized heap, so
        // `winners` can hold every record. Above a threshold the final sort
        // dominates and a parallel sort pays (~2.81× at 50k, `winners_sort` bench);
        // below it the rayon overhead is not worth it. Bit-identical — the same
        // gated lever as `search.rs:183` (`compare_best_first` is a strict total
        // order, so the parallel sort yields the same unique order).
        if winners.len() >= PAR_SORT_THRESHOLD {
            winners.par_sort_unstable_by(compare_best_first);
        } else {
            winners.sort_unstable_by(compare_best_first);
        }
        let mut hits = Vec::with_capacity(winners.len());
        for winner in winners {
            let index_u32 =
                u32::try_from(winner.index).map_err(|_| SearchError::InvalidConfig {
                    field: "index".to_owned(),
                    value: winner.index.to_string(),
                    reason: "index exceeds u32 range for VectorHit".to_owned(),
                })?;
            hits.push(VectorHit {
                index: index_u32,
                score: winner.score,
                doc_id: self.doc_ids[winner.index].as_str().into(),
            });
        }
        Ok(hits)
    }

    /// Iterate over all document IDs.
    pub fn iter_doc_ids(&self) -> impl Iterator<Item = &str> {
        self.doc_ids.iter().map(String::as_str)
    }

    /// Get the f32 vector at position `index`.
    ///
    /// # Errors
    ///
    /// Returns error if index is out of bounds.
    pub fn vector_at_f32(&self, index: usize) -> SearchResult<Vec<f32>> {
        if index >= self.record_count() {
            return Err(SearchError::InvalidConfig {
                field: "index".to_owned(),
                value: index.to_string(),
                reason: format!(
                    "index {} out of bounds (record_count = {})",
                    index,
                    self.record_count()
                ),
            });
        }
        let stored = self.vector_slice(index);
        // SIMD-widen 8 f16 per block (`widen8_f16_slice`, the same magic-factor widen
        // the f16 dot kernels use — bit-identical to the scalar `f16::to_f32`), then a
        // scalar tail for the last < 8. Mirrors the FSVI `VectorIndex::vector_at_f32`.
        let mut out = Vec::with_capacity(stored.len());
        let (blocks, remainder) = stored.as_chunks::<8>();
        for arr in blocks {
            out.extend_from_slice(&crate::simd::widen8_f16_slice(arr).to_array());
        }
        for v in remainder {
            out.push(v.to_f32());
        }
        Ok(out)
    }

    /// Compute dot products between a query and specific hit positions.
    ///
    /// Used for quality scoring when this index serves as the quality tier.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` when `query.len() != dimension`.
    pub fn scores_for_hits(&self, query: &[f32], hits: &[VectorHit]) -> SearchResult<Vec<f32>> {
        if query.len() != self.dimension {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension,
                found: query.len(),
            });
        }
        let mut scores = Vec::with_capacity(hits.len());
        for hit in hits {
            // Try to find by doc_id
            let score = self
                .index_of_doc_id(&hit.doc_id)
                .map(|idx| {
                    let stored = self.vector_slice(idx);
                    dot_product_f16_f32(stored, query)
                })
                .transpose()?
                .unwrap_or(0.0);
            scores.push(score);
        }
        Ok(scores)
    }
}

/// In-memory two-tier index wrapping fast and optional quality `InMemoryVectorIndex`.
///
/// Provides the same `search_fast()` / `quality_scores_for_hits()` API as
/// [`crate::TwoTierIndex`] but with fully-resident memory for deterministic latency.
#[derive(Debug, Clone)]
pub struct InMemoryTwoTierIndex {
    fast_index: InMemoryVectorIndex,
    quality_index: Option<InMemoryVectorIndex>,
}

impl InMemoryTwoTierIndex {
    /// Create from two pre-built in-memory indices.
    #[must_use]
    pub const fn new(
        fast_index: InMemoryVectorIndex,
        quality_index: Option<InMemoryVectorIndex>,
    ) -> Self {
        Self {
            fast_index,
            quality_index,
        }
    }

    /// Construct the shipping in-memory two-tier product from admitted v2
    /// owners and generation-keyed residual-cache directories. The fast tier
    /// and optional quality tier independently attach only their own exact
    /// sidecars; a cache failure leaves that tier's established flat search
    /// active without weakening FSVI admission or cross-tier identity.
    ///
    /// # Errors
    ///
    /// Returns errors while loading the admitted source vectors. Optional
    /// sidecar cache I/O is deliberately contained as per-tier flat fallback.
    pub fn from_admitted_v2_with_residual_sidecar_cache(
        fast_source: &ValidatedFsviBytes,
        fast_cache_dir: &Path,
        quality_source: Option<(&ValidatedFsviBytes, &Path)>,
    ) -> SearchResult<Self> {
        let fast_index = InMemoryVectorIndex::from_admitted_v2_with_residual_sidecar_cache(
            fast_source,
            fast_cache_dir,
        )?;
        let quality_index = quality_source
            .map(|(source, cache_dir)| {
                InMemoryVectorIndex::from_admitted_v2_with_residual_sidecar_cache(source, cache_dir)
            })
            .transpose()?;
        Ok(Self::new(fast_index, quality_index))
    }

    /// Load from an existing two-tier index directory, reading all data into memory.
    ///
    /// Looks for `vector.fast.idx` (required) and `vector.quality.idx` (optional).
    /// Falls back to `vector.idx` if the fast filename doesn't exist.
    ///
    /// # Errors
    ///
    /// Returns errors from FSVI parsing or vector loading.
    pub fn from_dir(dir: &Path) -> SearchResult<Self> {
        let fast_path = dir.join(crate::two_tier::VECTOR_INDEX_FAST_FILENAME);
        let fast_path = if fast_path.exists() {
            fast_path
        } else {
            let fallback = dir.join(crate::two_tier::VECTOR_INDEX_FALLBACK_FILENAME);
            if !fallback.exists() {
                return Err(SearchError::IndexNotFound { path: fast_path });
            }
            fallback
        };
        let fast_index = InMemoryVectorIndex::from_fsvi(&fast_path)?;

        let quality_path = dir.join(crate::two_tier::VECTOR_INDEX_QUALITY_FILENAME);
        let quality_index = if quality_path.exists() {
            Some(InMemoryVectorIndex::from_fsvi(&quality_path)?)
        } else {
            None
        };

        Ok(Self {
            fast_index,
            quality_index,
        })
    }

    /// Search the fast tier.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`InMemoryVectorIndex::search_top_k`].
    pub fn search_fast(&self, query_vec: &[f32], k: usize) -> SearchResult<Vec<VectorHit>> {
        self.fast_index.search_top_k(query_vec, k, None)
    }

    /// Search the fast tier with typed zero-signal classification.
    ///
    /// Mirrors [`crate::TwoTierIndex::search_fast_classified`] so in-memory
    /// and persistent two-tier paths classify equivalent states identically
    /// (bd-tqhc). The in-memory variant performs no transition logging: its
    /// indexes are ephemeral, and the once-per-generation logging bound is
    /// owned by the persistent path.
    ///
    /// # Errors
    ///
    /// Everything [`Self::search_fast`] returns, plus
    /// [`SearchError::InvalidConfig`] for non-finite query vectors.
    pub fn search_fast_classified(
        &self,
        query_vec: &[f32],
        k: usize,
    ) -> SearchResult<ClassifiedHits> {
        self.fast_index.search_top_k_classified(query_vec, k, None)
    }

    /// Search the fast tier with configurable parallelism.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`InMemoryVectorIndex::search_top_k_with_params`].
    pub fn search_fast_with_params(
        &self,
        query_vec: &[f32],
        k: usize,
        params: Option<SearchParams>,
    ) -> SearchResult<Vec<VectorHit>> {
        let params = params.unwrap_or_default();
        self.fast_index
            .search_top_k_with_params(query_vec, k, None, params)
    }

    /// Compute quality-tier scores for fast-index hits.
    ///
    /// Missing quality entries produce `0.0`.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` if `query_vec` doesn't match
    /// the quality index dimensionality.
    pub fn quality_scores_for_hits(
        &self,
        query_vec: &[f32],
        hits: &[VectorHit],
    ) -> SearchResult<Vec<Option<f32>>> {
        let Some(quality) = &self.quality_index else {
            return Ok(vec![None; hits.len()]);
        };
        if query_vec.len() != quality.dimension {
            return Err(SearchError::DimensionMismatch {
                expected: quality.dimension,
                found: query_vec.len(),
            });
        }
        let mut scores = Vec::with_capacity(hits.len());
        for hit in hits {
            let score = quality
                .index_of_doc_id(&hit.doc_id)
                .map(|idx| dot_product_f16_f32(quality.vector_slice(idx), query_vec))
                .transpose()?;
            scores.push(score);
        }
        Ok(scores)
    }

    /// Whether a quality index is loaded.
    #[must_use]
    pub const fn has_quality_index(&self) -> bool {
        self.quality_index.is_some()
    }

    /// Number of documents in the fast tier.
    #[must_use]
    pub fn doc_count(&self) -> usize {
        self.fast_index.record_count()
    }

    /// Iterate over all document IDs in fast-tier order.
    pub fn iter_doc_ids(&self) -> impl Iterator<Item = &str> {
        self.fast_index.iter_doc_ids()
    }

    /// Get a reference to the fast index.
    #[must_use]
    pub const fn fast_index(&self) -> &InMemoryVectorIndex {
        &self.fast_index
    }

    /// Get a reference to the quality index (if present).
    #[must_use]
    pub const fn quality_index(&self) -> Option<&InMemoryVectorIndex> {
        self.quality_index.as_ref()
    }

    /// Space fingerprint of the fast tier, when known
    /// (see [`InMemoryVectorIndex::space_fingerprint_hex`]).
    #[must_use]
    pub fn fast_space_fingerprint_hex(&self) -> Option<&str> {
        self.fast_index.space_fingerprint_hex()
    }

    /// Space fingerprint of the quality tier, when known.
    ///
    /// `None` both when no quality index is loaded and when the loaded one
    /// carries no identity; either way there is no quality-tier space to
    /// verify a query embedding against
    /// (see [`InMemoryVectorIndex::space_fingerprint_hex`]).
    #[must_use]
    pub fn quality_space_fingerprint_hex(&self) -> Option<&str> {
        self.quality_index
            .as_ref()
            .and_then(InMemoryVectorIndex::space_fingerprint_hex)
    }

    /// Whether the fast tier's space identity is FSVI-v2-HEADER-attested
    /// rather than declared
    /// (see [`InMemoryVectorIndex::space_identity_is_attested`];
    /// bd-9xuj T2-C4-write, guards 2+8).
    #[must_use]
    pub const fn fast_identity_is_attested(&self) -> bool {
        self.fast_index.space_identity_is_attested()
    }

    /// Whether the quality tier's space identity is FSVI-v2-HEADER-attested.
    ///
    /// `false` both when no quality index is loaded and when the loaded one's
    /// identity was declared rather than read from a validated v2 header
    /// (see [`InMemoryVectorIndex::space_identity_is_attested`];
    /// bd-9xuj T2-C4-write, guards 2+8).
    #[must_use]
    pub fn quality_identity_is_attested(&self) -> bool {
        self.quality_index
            .as_ref()
            .is_some_and(InMemoryVectorIndex::space_identity_is_attested)
    }

    /// Embedder id retained by the fast tier, when its source carried one
    /// (see [`InMemoryVectorIndex::embedder_id`]; bd-9xuj T2-C2).
    #[must_use]
    pub fn fast_embedder_id(&self) -> Option<&str> {
        self.fast_index.embedder_id()
    }

    /// Embedder revision retained by the fast tier, when its source carried
    /// one (see [`InMemoryVectorIndex::embedder_revision`]; bd-9xuj T2-C2).
    #[must_use]
    pub fn fast_embedder_revision(&self) -> Option<&str> {
        self.fast_index.embedder_revision()
    }

    /// Embedder id retained by the quality tier: `None` both when no quality
    /// index is loaded and when the loaded one's source carried no id
    /// (bd-9xuj T2-C2).
    #[must_use]
    pub fn quality_embedder_id(&self) -> Option<&str> {
        self.quality_index
            .as_ref()
            .and_then(InMemoryVectorIndex::embedder_id)
    }

    /// Embedder revision retained by the quality tier, under the same rules
    /// as [`Self::quality_embedder_id`] (bd-9xuj T2-C2).
    #[must_use]
    pub fn quality_embedder_revision(&self) -> Option<&str> {
        self.quality_index
            .as_ref()
            .and_then(InMemoryVectorIndex::embedder_revision)
    }
}

/// Selectivity threshold for the gather fast-path: take it only when the filter's
/// allow-set is smaller than `corpus / GATHER_SELECTIVITY_DIVISOR`. Below this the
/// gather (exact f16 dots over the allow-set, parallel above `PARALLEL_CHUNK_SIZE`)
/// beats the parallel per-document scan; above it the scan wins because the gather's
/// serial setup (allow-set collect + position sort) grows with the allow-set. The
/// `filtered_gather` selectivity-sweep bench (N=50k, dim 384) measured, with the
/// parallel gather: 14× at 0.5%, 8.6× at 1%, 2.1× at 5%, 1.3× at 10%, then a loss by
/// 25% (crossover ~13%). Gate at **N/10 (10%)** — inside the winning region
/// (≥1.3× at the boundary) with margin for a machine whose core count shifts the
/// crossover. (Tiny allow-sets stay serial and still hit 6.9–50× — see the ledger.)
const GATHER_SELECTIVITY_DIVISOR: usize = 10;

// ─── Heap helpers (mirrors search.rs internals) ─────────────────────────────

#[derive(Debug, Clone, Copy)]
struct HeapEntry {
    index: usize,
    score: f32,
}

impl HeapEntry {
    const fn new(index: usize, score: f32) -> Self {
        Self { index, score }
    }
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.score.to_bits() == other.score.to_bits()
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: "largest" == worst score, so peek() returns cutoff.
        match score_key(self.score).total_cmp(&score_key(other.score)) {
            Ordering::Less => Ordering::Greater,
            Ordering::Greater => Ordering::Less,
            Ordering::Equal => self.index.cmp(&other.index),
        }
    }
}

const fn score_key(score: f32) -> f32 {
    if score.is_nan() {
        f32::NEG_INFINITY
    } else {
        score
    }
}

/// Winners-count threshold above which the `limit_all` final sort uses a parallel
/// `par_sort_unstable_by` (mirrors `search::PAR_SORT_THRESHOLD`). Below it, rayon's
/// spawn/merge overhead is not amortized for the cheap `compare_best_first`.
const PAR_SORT_THRESHOLD: usize = 16_384;

fn compare_best_first(left: &HeapEntry, right: &HeapEntry) -> Ordering {
    match score_key(right.score).total_cmp(&score_key(left.score)) {
        Ordering::Equal => left.index.cmp(&right.index),
        other => other,
    }
}

fn insert_candidate(heap: &mut BinaryHeap<HeapEntry>, candidate: HeapEntry, limit: usize) {
    if limit == 0 {
        return;
    }
    if heap.len() < limit {
        heap.push(candidate);
        return;
    }
    if let Some(&worst) = heap.peek()
        && match score_key(candidate.score).total_cmp(&score_key(worst.score)) {
            Ordering::Greater => true,
            Ordering::Less => false,
            Ordering::Equal => candidate.index < worst.index,
        }
    {
        let _ = heap.pop();
        heap.push(candidate);
    }
}

fn bounded_heap(capacity: usize, field: &str) -> SearchResult<BinaryHeap<HeapEntry>> {
    let mut heap = BinaryHeap::new();
    heap.try_reserve_exact(capacity)
        .map_err(|_| residual_sidecar_error(field, "heap allocation failed"))?;
    Ok(heap)
}

fn merge_partial_heaps(
    partial_heaps: Vec<BinaryHeap<HeapEntry>>,
    limit: usize,
) -> SearchResult<BinaryHeap<HeapEntry>> {
    let mut total_elements = 0_usize;
    for heap in &partial_heaps {
        total_elements = total_elements.saturating_add(heap.len());
    }
    let capacity = limit.min(total_elements);
    let mut merged = bounded_heap(capacity, "merge_heap")?;
    for partial in partial_heaps {
        for entry in partial {
            insert_candidate(&mut merged, entry, limit);
        }
    }
    Ok(merged)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(
        clippy::cast_precision_loss,
        clippy::items_after_statements,
        clippy::redundant_clone,
        clippy::suboptimal_flops,
        clippy::unnecessary_literal_bound
    )]

    use super::*;
    use crate::{FsviV2IdentityBinding, Quantization};
    use frankensearch_core::BoundQueryEmbedding;
    use frankensearch_core::generation::{
        ArtifactGenerationIdentityV1, EmbeddingIdentityBundleV1, QuantizationFormat,
    };
    use proptest::prelude::*;
    use std::path::{Path, PathBuf};
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

    fn test_run_nonce() -> u128 {
        static RUN_NONCE: OnceLock<u128> = OnceLock::new();
        *RUN_NONCE.get_or_init(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos())
        })
    }

    fn temp_index_path(name: &str) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let nonce = COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
        let dir = std::env::temp_dir().join("frankensearch_in_memory_tests");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir.join(format!(
            "{name}-{}-{}-{nonce}.fsvi",
            std::process::id(),
            test_run_nonce()
        ))
    }

    fn owned_temp_dir(name: &str) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let nonce = COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
        let parent = std::env::temp_dir().join("frankensearch_in_memory_tests");
        std::fs::create_dir_all(&parent).expect("create temp parent");
        let dir = parent.join(format!(
            "{name}-{}-{}-{nonce}",
            std::process::id(),
            test_run_nonce()
        ));
        std::fs::create_dir(&dir).expect("create unique owned temp directory");
        dir
    }

    fn write_new_owned_file(path: &Path, bytes: &[u8]) {
        use std::io::Write as _;

        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .expect("create a unique owned test file without replacement");
        file.write_all(bytes)
            .expect("write a unique owned test file");
        file.sync_all().expect("sync a unique owned test file");
    }

    fn cleanup(path: &Path) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(path.with_extension("fsvi.wal"));
    }

    fn make_normalized_vec(dim: usize, seed: f32) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim).map(|i| (seed + i as f32 * 0.1).sin()).collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }

    fn bind_test_residual_source(index: &mut InMemoryVectorIndex) {
        index.residual_source_binding = Some(ResidualSourceBinding {
            generation_fingerprint: [0x19; 32],
            vector_content_digest: [0x2a; 32],
            ordered_live_docset_digest: [0x3b; 32],
            space_fingerprint: [0x4c; 32],
        });
    }

    fn finite_f16_bits(bits: u16) -> f32 {
        let value = f16::from_bits(bits);
        if value.is_finite() {
            value.to_f32()
        } else {
            0.0
        }
    }

    fn bounded_finite_f32_bits(bits: u32) -> f32 {
        // Preserve sign, every subnormal, and a broad finite exponent range
        // while clearing the top exponent bit. That keeps the test inside the
        // sidecar's analytically admitted finite-envelope route rather than
        // asking a pruning proof to cover the intentional flat fallback.
        f32::from_bits(bits & 0xbfff_ffff)
    }

    fn assert_residual_bounds_cover_exact_scores(
        index: &InMemoryVectorIndex,
        sidecar: &ExactResidualSidecar,
        query: &[f32],
    ) {
        let transformed =
            ResidualQueryTransform::from_query(query, sidecar).expect("finite test transform");
        assert!(
            transformed.f32_flat_envelope_is_finite,
            "the test vector range must use the interval route"
        );
        for group in 0..sidecar.group_count() {
            let group_start = group * sidecar.lanes;
            let active_lanes = (sidecar.count - group_start).min(sidecar.lanes);
            let centroid =
                &sidecar.centroids[group * sidecar.dimension..(group + 1) * sidecar.dimension];
            let mut centroid_dot = 0.0_f64;
            let mut centroid_norm_sq = 0.0_f64;
            for (&query_value, &centroid_value) in transformed.transformed.iter().zip(centroid) {
                centroid_dot += f64::from(query_value) * f64::from(centroid_value);
                centroid_norm_sq += f64::from(centroid_value) * f64::from(centroid_value);
            }
            let centroid_norm =
                outward_f64_norm_from_square_sum(centroid_norm_sq, sidecar.dimension)
                    .expect("finite centroid norm has an outward f64 root");
            let mut partial = [0.0_f64; EXACT_RESIDUAL_LANES];
            for block_index in 0..sidecar.block_count() {
                for lane in 0..active_lanes {
                    let suffix_offset =
                        (group * (sidecar.block_count() + 1) + block_index) * sidecar.lanes + lane;
                    let upper_bound = residual_lane_upper_bound(
                        sidecar,
                        &transformed,
                        block_index,
                        centroid_dot,
                        centroid_norm,
                        f64::from(sidecar.suffix_norms[suffix_offset]),
                        f64::from(sidecar.correction_norms[group * sidecar.lanes + lane]),
                        partial[lane],
                    );
                    let exact_score =
                        dot_product_f16_f32(index.vector_slice(group_start + lane), query)
                            .expect("finite f16 score");
                    assert!(
                        upper_bound.is_finite() && upper_bound >= f64::from(exact_score),
                        "upper bound underestimates group={group} lane={lane} block={block_index}: \
                         upper={upper_bound:?}, exact={exact_score:?}"
                    );
                }
                let start = block_index * sidecar.block;
                let end = (start + sidecar.block).min(sidecar.dimension);
                for transformed_dimension in start..end {
                    let query_value = f64::from(transformed.transformed[transformed_dimension]);
                    let base = group * sidecar.dimension * sidecar.lanes
                        + transformed_dimension * sidecar.lanes;
                    for lane in 0..active_lanes {
                        partial[lane] += query_value * f64::from(sidecar.residuals[base + lane]);
                    }
                }
            }
        }
    }

    #[test]
    fn from_vectors_basic() {
        let dim = 8;
        let doc_ids = vec!["a".into(), "b".into(), "c".into()];
        let vectors = vec![
            make_normalized_vec(dim, 1.0),
            make_normalized_vec(dim, 2.0),
            make_normalized_vec(dim, 3.0),
        ];
        let index = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();
        assert_eq!(index.record_count(), 3);
        assert_eq!(index.dimension(), 8);
        assert_eq!(index.doc_id_at(0).unwrap(), "a");
        assert_eq!(index.doc_id_at(2).unwrap(), "c");
    }

    #[test]
    fn from_vectors_dimension_mismatch() {
        let doc_ids = vec!["a".into()];
        let vectors = vec![vec![1.0, 2.0, 3.0]]; // dim 3 != expected 4
        let result = InMemoryVectorIndex::from_vectors(doc_ids, vectors, 4);
        assert!(result.is_err());
    }

    #[test]
    fn from_vectors_count_mismatch() {
        let doc_ids = vec!["a".into(), "b".into()];
        let vectors = vec![vec![1.0, 2.0]]; // 1 vector, 2 doc_ids
        let result = InMemoryVectorIndex::from_vectors(doc_ids, vectors, 2);
        assert!(result.is_err());
    }

    #[test]
    fn from_vectors_non_finite_rejected() {
        let doc_ids = vec!["a".into()];
        let vectors = vec![vec![1.0, f32::NAN]];
        let result = InMemoryVectorIndex::from_vectors(doc_ids, vectors, 2);
        assert!(result.is_err());
    }

    #[test]
    fn from_fsvi_matches_file_backed_search() {
        let path = temp_index_path("from_fsvi");
        cleanup(&path);

        let dim = 32;
        let docs = 64usize;
        let mut writer = crate::VectorIndex::create_with_revision(
            &path,
            "test-embedder",
            "rev-a",
            dim,
            Quantization::F16,
        )
        .unwrap();

        for i in 0..docs {
            let vector = make_normalized_vec(dim, i as f32 * 0.73);
            writer.write_record(&format!("doc-{i}"), &vector).unwrap();
        }
        writer.finish().unwrap();

        let file_index = crate::VectorIndex::open_read_only(&path).unwrap();
        let memory_index = InMemoryVectorIndex::from_fsvi(&path).unwrap();
        assert_eq!(memory_index.record_count(), docs);
        assert_eq!(memory_index.dimension(), dim);

        let query = make_normalized_vec(dim, 12.4);
        let file_hits = file_index.search_top_k(&query, 10, None).unwrap();
        let memory_hits = memory_index.search_top_k(&query, 10, None).unwrap();
        assert_eq!(file_hits.len(), memory_hits.len());

        for (file, memory) in file_hits.iter().zip(memory_hits.iter()) {
            assert_eq!(file.doc_id, memory.doc_id);
            assert!(
                (file.score - memory.score).abs() < 0.001,
                "score mismatch for {}: file={} memory={}",
                file.doc_id,
                file.score,
                memory.score
            );
        }

        // Verify vectors were loaded in quantized form and still round-trip.
        let recovered = memory_index.vector_at_f32(0).unwrap();
        assert_eq!(recovered.len(), dim);

        cleanup(&path);
    }

    // ─── Embedding-space identity (bd-9xuj T2-C3) ───────────────────────────

    /// Write a real FSVI v2 file at `path` whose identity binds
    /// `explicit_test_model(model_id, dimension)` with fsvi-v2 storage.
    /// Returns the exact identity binding (needed for admission) and the
    /// core-side space fingerprint the file's header preserves.
    fn write_fsvi_v2_fixture(
        path: &Path,
        model_id: &str,
        dimension: usize,
        generation_sequence: u64,
        rows: &[(String, Vec<f32>)],
    ) -> (FsviV2IdentityBinding, String) {
        let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(
            model_id,
            u32::try_from(dimension).expect("test dimension fits u32"),
        );
        identity.storage.format = "fsvi-v2".to_owned();
        identity.storage.quantization = QuantizationFormat::F16;
        identity.storage.endianness = "little-endian".to_owned();
        let space_fingerprint = identity.space.fingerprint();
        let generation = ArtifactGenerationIdentityV1::new(generation_sequence, [0x5c; 16])
            .expect("valid test generation");
        let binding =
            FsviV2IdentityBinding::new(generation, identity.freeze().expect("freeze identity"))
                .expect("valid FSVI v2 identity binding");
        let mut writer =
            crate::VectorIndex::create_v2(path, binding.clone()).expect("create v2 writer");
        for (doc_id, vector) in rows {
            writer.write_record(doc_id, vector).expect("write v2 row");
        }
        writer.finish().expect("finish v2 fixture");
        (binding, space_fingerprint)
    }

    fn identity_rows(dim: usize, count: usize) -> (Vec<String>, Vec<Vec<f32>>) {
        let doc_ids = (0..count).map(|i| format!("doc-{i}")).collect();
        let vectors = (0..count)
            .map(|i| make_normalized_vec(dim, (i + 1) as f32))
            .collect();
        (doc_ids, vectors)
    }

    #[test]
    fn from_vectors_with_identity_exposes_space_fingerprint() {
        let dim = 8;
        let bundle = EmbeddingIdentityBundleV1::explicit_test_model("mem-space-a", 8);
        let (doc_ids, vectors) = identity_rows(dim, 3);
        let index =
            InMemoryVectorIndex::from_vectors_with_identity(doc_ids, vectors, dim, &bundle.space)
                .expect("build identified index");
        assert_eq!(
            index.space_fingerprint_hex(),
            Some(bundle.space.fingerprint().as_str()),
            "the index must expose exactly the space fingerprint it was built with"
        );
        assert_eq!(index.record_count(), 3);
        assert!(
            !index.space_identity_is_attested(),
            "a caller-declared space is DECLARED, never header-attested (C4-write guards 2+8)"
        );

        // The identity-less constructor stays typed-absent: no fabrication.
        let (doc_ids, vectors) = identity_rows(dim, 3);
        let legacy = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).expect("build");
        assert_eq!(
            legacy.space_fingerprint_hex(),
            None,
            "an identity-less build must never fabricate a space fingerprint"
        );
        assert!(!legacy.space_identity_is_attested());
    }

    #[test]
    fn from_vectors_with_identity_rejects_dimension_mismatch() -> Result<(), String> {
        // A 16-dim space cannot describe an 8-dim index: the identity claim
        // is checked before it is stored, never trusted on assertion alone.
        let bundle = EmbeddingIdentityBundleV1::explicit_test_model("mem-space-mismatch", 16);
        let (doc_ids, vectors) = identity_rows(8, 2);
        let error =
            InMemoryVectorIndex::from_vectors_with_identity(doc_ids, vectors, 8, &bundle.space)
                .expect_err("16-dim space must not bind an 8-dim index");
        let rendered = format!("{error:?}");
        let SearchError::InvalidConfig { field, value, .. } = error else {
            return Err(format!("expected InvalidConfig, got {rendered}"));
        };
        assert_eq!(field, "space_identity.dimension");
        assert_eq!(value, "16");
        Ok(())
    }

    #[test]
    fn admitted_v2_load_preserves_space_identity() {
        // Readiness map §3.3(3), adapted to the only v2 open path that
        // exists: `VectorIndex::open` is v1-only by design, so an
        // identity-complete v2 artifact reaches memory through exact
        // admission. The in-memory index must preserve the admitted space
        // fingerprint, and its hex must equal the core-side
        // `EmbeddingSpaceIdentityV1::fingerprint()` that produced the file —
        // one join key across crates and codecs.
        //
        // The fixture lives in its OWN directory: since 58726e26 admission
        // snapshots the containing directory and fails closed with
        // `DirectoryChangedDuringRead` when sibling test files churn it, so
        // the shared test temp dir is not a legal admission site.
        let dir = owned_temp_dir("admitted_v2_space_identity");
        let path = dir.join("admitted_v2_space_identity.fsvi");
        let dim = 8;
        let (doc_ids, vectors) = identity_rows(dim, 4);
        let rows: Vec<(String, Vec<f32>)> = doc_ids.into_iter().zip(vectors).collect();
        let (binding, expected) = write_fsvi_v2_fixture(&path, "fsvi-space-model", dim, 11, &rows);

        let admitted = crate::VectorIndex::open_admitted_v2(&path, &binding)
            .expect("exact admission of the v2 fixture");
        let index = InMemoryVectorIndex::from_admitted_v2(&admitted)
            .expect("load admitted v2 artifact into memory");
        assert_eq!(index.record_count(), 4);
        assert_eq!(
            index.space_fingerprint_hex(),
            Some(expected.as_str()),
            "the in-memory index must preserve the admitted space identity"
        );
        assert!(
            index.space_identity_is_attested(),
            "an identity read out of a validated v2 header through exact admission is ATTESTED \
             (C4-write guards 2+8)"
        );
        let two_tier = InMemoryTwoTierIndex::new(index, None);
        assert!(two_tier.fast_identity_is_attested());
        assert!(
            !two_tier.quality_identity_is_attested(),
            "no quality tier means no attested quality identity"
        );

        // Sanity pin: the legacy pathname loader cannot read v2 at all, so
        // the admitted path above is not optional for identified artifacts.
        let error = InMemoryVectorIndex::from_fsvi(&path)
            .expect_err("VectorIndex::open must reject v2 bytes");
        assert!(matches!(error, SearchError::IndexVersionMismatch { .. }));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn admitted_v2_two_tier_shipping_cache_skips_corrupt_entries_without_overwrite() {
        // Use a private directory because v2 admission snapshots the parent;
        // sidecar publication and discovery must share that same held-directory
        // model without test-suite sibling churn or cross-process path reuse.
        let dir = owned_temp_dir("admitted_v2_residual_product");
        let path = dir.join("index.fsvi");
        let cache_dir = dir.join("residual-cache");
        std::fs::create_dir(&cache_dir).expect("create owned cache directory");
        let dimension = 35;
        let (doc_ids, vectors) = identity_rows(dimension, 17);
        let rows: Vec<(String, Vec<f32>)> = doc_ids.into_iter().zip(vectors).collect();
        let (binding, _) = write_fsvi_v2_fixture(&path, "residual-product", dimension, 29, &rows);
        let admitted =
            crate::VectorIndex::open_admitted_v2(&path, &binding).expect("admit private v2 source");
        let query = make_normalized_vec(dimension, 3.25);
        let flat = InMemoryVectorIndex::from_admitted_v2(&admitted).expect("flat admitted index");
        let cache_prefix = flat
            .exact_residual_generation_cache_prefix()
            .expect("derive admitted generation cache key");
        let mut header_valid_corrupt = flat
            .build_exact_residual_sidecar()
            .expect("derive a cache-shaped sidecar")
            .encode()
            .expect("encode a cache-shaped sidecar");
        header_valid_corrupt[EXACT_RESIDUAL_SIDECAR_HEADER_BYTES] ^= 0x01;
        let mut corrupt_paths = Vec::new();
        corrupt_paths
            .try_reserve_exact(EXACT_RESIDUAL_CACHE_ATTEMPTS + 1)
            .expect("reserve bounded corrupt-cache fixture paths");
        for sequence in 0..=EXACT_RESIDUAL_CACHE_ATTEMPTS {
            let corrupt_path = cache_dir.join(format!("{cache_prefix}corrupt-{sequence}.fsrs"));
            write_new_owned_file(&corrupt_path, &header_valid_corrupt);
            corrupt_paths.push(corrupt_path);
        }
        let indexed = InMemoryTwoTierIndex::from_admitted_v2_with_residual_sidecar_cache(
            &admitted, &cache_dir, None,
        )
        .expect("shipping two-tier constructor keeps the admitted index usable");
        assert!(
            indexed.fast_index.exact_residual_sidecar.get().is_some(),
            "the shipping route publishes and attaches the source-derived sidecar"
        );
        assert_eq!(
            indexed
                .search_fast(&query, 5)
                .expect("sidecar product search"),
            flat.search_top_k(&query, 5, None)
                .expect("flat product search")
        );

        let entry_count_after_publish = std::fs::read_dir(&cache_dir)
            .expect("read owned cache after publication")
            .flatten()
            .count();
        assert_eq!(
            entry_count_after_publish,
            corrupt_paths.len() + 1,
            "publication retains every corrupt immutable entry and adds one generation-matched artifact"
        );
        for corrupt_path in &corrupt_paths {
            assert_eq!(
                std::fs::read(corrupt_path).expect("read planted corrupt cache entry"),
                header_valid_corrupt,
                "the shipping cache route never overwrites a stale/corrupt artifact"
            );
        }

        let reopened = InMemoryTwoTierIndex::from_admitted_v2_with_residual_sidecar_cache(
            &admitted, &cache_dir, None,
        )
        .expect("cache reader scans past corrupt entries to the valid generation artifact");
        assert!(reopened.fast_index.exact_residual_sidecar.get().is_some());
        assert_eq!(
            reopened
                .search_fast(&query, 5)
                .expect("reopened product search"),
            flat.search_top_k(&query, 5, None)
                .expect("baseline flat search")
        );
        let entry_count_after_reopen = std::fs::read_dir(&cache_dir)
            .expect("read owned cache after reopening")
            .flatten()
            .count();
        assert_eq!(
            entry_count_after_reopen, entry_count_after_publish,
            "reopening selects the existing valid sidecar instead of writing another artifact"
        );

        let unavailable_cache = dir.join("unavailable-cache");
        let fallback = InMemoryTwoTierIndex::from_admitted_v2_with_residual_sidecar_cache(
            &admitted,
            &unavailable_cache,
            None,
        )
        .expect("unavailable optional cache retains the admitted flat tier");
        assert!(fallback.fast_index.exact_residual_sidecar.get().is_none());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn exact_residual_capability_failure_precedes_public_writer_derivation() {
        let dimension = 35;
        let mut index = InMemoryVectorIndex::from_vectors(
            (0..17)
                .map(|row| format!("writer-capability-{row}"))
                .collect(),
            (0..17)
                .map(|row| make_normalized_vec(dimension, row as f32 + 0.25))
                .collect(),
            dimension,
        )
        .expect("finite public-writer source");
        bind_test_residual_source(&mut index);
        let directory = owned_temp_dir("exact_residual_writer_capability");
        let destination = directory.join("sidecar.fsrs");

        reset_exact_residual_sidecar_build_count();
        fail_next_exact_residual_publication_acquisition();
        assert!(
            index.write_exact_residual_sidecar(&destination).is_err(),
            "a failed create-only capability must leave public writing unavailable"
        );
        assert!(
            !exact_residual_publication_acquisition_failure_is_pending(),
            "the public writer must consume the acquisition-failure seam"
        );
        assert_eq!(
            exact_residual_sidecar_build_count(),
            0,
            "public writer must acquire publication capability before deriving a sidecar"
        );
        assert_eq!(
            std::fs::read_dir(&directory)
                .expect("inspect private writer directory")
                .count(),
            0,
            "failed capability acquisition must leave no visible temporary artifact"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn valid_exact_residual_candidate_attaches_and_prunes_without_publication_capability() {
        let directory = owned_temp_dir("exact_residual_read_only_candidate");
        let source_path = directory.join("index.fsvi");
        let cache_dir = directory.join("residual-cache");
        std::fs::create_dir(&cache_dir).expect("create private cache directory");
        let dimension = 35;
        let count = 17;
        let mut vectors = vec![vec![0.0_f32; dimension]; count];
        for vector in vectors.iter_mut().take(EXACT_RESIDUAL_LANES) {
            vector[0] = 1.0;
        }
        for vector in vectors.iter_mut().skip(EXACT_RESIDUAL_LANES) {
            vector[0] = -1.0;
        }
        let doc_ids = (0..count)
            .map(|row| format!("read-only-candidate-{row}"))
            .collect::<Vec<_>>();
        let rows: Vec<(String, Vec<f32>)> = doc_ids.into_iter().zip(vectors).collect();
        let (binding, _) =
            write_fsvi_v2_fixture(&source_path, "residual-read-only", dimension, 57, &rows);
        let admitted = crate::VectorIndex::open_admitted_v2(&source_path, &binding)
            .expect("admit private v2 source");
        let flat =
            InMemoryVectorIndex::from_admitted_v2(&admitted).expect("load flat admitted source");
        let cache_prefix = flat
            .exact_residual_generation_cache_prefix()
            .expect("derive generation cache prefix");
        let valid = flat
            .build_exact_residual_sidecar()
            .expect("derive cache-shaped valid fixture")
            .encode()
            .expect("encode cache-shaped valid fixture");
        let valid_path = cache_dir.join(format!("{cache_prefix}valid.fsrs"));
        write_new_owned_file(&valid_path, &valid);

        reset_exact_residual_sidecar_build_count();
        fail_next_exact_residual_publication_acquisition();
        let cached = InMemoryVectorIndex::from_admitted_v2_with_residual_sidecar_cache(
            &admitted, &cache_dir,
        )
        .expect("read-only reusable cache candidate must retain the exact source");
        assert!(
            exact_residual_publication_acquisition_failure_is_pending(),
            "a reusable immutable candidate must attach without acquiring publication capability"
        );
        assert_eq!(
            exact_residual_sidecar_build_count(),
            1,
            "the sidecar must be derived exactly once to authenticate the read-only candidate"
        );
        assert!(
            cached.has_exact_residual_sidecar(),
            "a valid immutable candidate must remain attached when publication is unavailable"
        );
        let mut query = vec![0.0_f32; dimension];
        query[0] = 1.0;
        let outcome = cached
            .scan_exact_residual_sidecar(
                &query,
                1,
                None,
                cached
                    .exact_residual_sidecar
                    .get()
                    .expect("read-only candidate is attached"),
            )
            .expect("scan attached read-only candidate");
        assert!(
            outcome.census.lanes_pruned > 0,
            "the reused candidate must retain real exact-residual pruning"
        );
        assert_eq!(
            cached
                .resolve_heap(outcome.heap)
                .expect("resolve read-only candidate result"),
            flat.search_top_k(&query, 1, None)
                .expect("flat exact baseline"),
            "read-only candidate pruning preserves the exact result"
        );
        assert!(
            take_exact_residual_publication_acquisition_failure(),
            "clear the planted unavailable-publication seam after proving reuse bypassed it"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn exact_residual_capability_failure_precedes_empty_cache_derivation() {
        let directory = owned_temp_dir("exact_residual_empty_capability");
        let source_path = directory.join("index.fsvi");
        let cache_dir = directory.join("residual-cache");
        std::fs::create_dir(&cache_dir).expect("create empty private cache directory");
        let dimension = 35;
        let (doc_ids, vectors) = identity_rows(dimension, 17);
        let rows: Vec<(String, Vec<f32>)> = doc_ids.into_iter().zip(vectors).collect();
        let (binding, _) =
            write_fsvi_v2_fixture(&source_path, "residual-capability", dimension, 57, &rows);
        let admitted = crate::VectorIndex::open_admitted_v2(&source_path, &binding)
            .expect("admit private v2 source");

        reset_exact_residual_sidecar_build_count();
        fail_next_exact_residual_publication_acquisition();
        let fallback = InMemoryVectorIndex::from_admitted_v2_with_residual_sidecar_cache(
            &admitted, &cache_dir,
        )
        .expect("optional cache failure must retain an exact flat source");
        assert!(
            !exact_residual_publication_acquisition_failure_is_pending(),
            "an empty cache must consume the publication-capability failure seam"
        );
        assert_eq!(
            exact_residual_sidecar_build_count(),
            0,
            "an unavailable empty cache must acquire capability before deriving a sidecar"
        );
        assert!(
            !fallback.has_exact_residual_sidecar(),
            "an unavailable optional empty cache must stay on the exact flat route"
        );
        assert_eq!(
            std::fs::read_dir(&cache_dir)
                .expect("inspect private empty cache directory")
                .count(),
            0,
            "failed capability acquisition must leave no visible temporary artifact"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn admitted_v2_cache_discovery_budget_skips_unrelated_and_corrupt_work() {
        // Each hostile directory is private and create-only. When either
        // deterministic budget is reached, cache use and publication are both
        // skipped; the exact flat product remains available.
        let dir = owned_temp_dir("admitted_v2_residual_cache_budget");
        let source_path = dir.join("index.fsvi");
        let dimension = 35;
        let (doc_ids, vectors) = identity_rows(dimension, 17);
        let rows: Vec<(String, Vec<f32>)> = doc_ids.into_iter().zip(vectors).collect();
        let (binding, _) =
            write_fsvi_v2_fixture(&source_path, "residual-cache-budget", dimension, 41, &rows);
        let admitted = crate::VectorIndex::open_admitted_v2(&source_path, &binding)
            .expect("admit private v2 source");
        let flat = InMemoryVectorIndex::from_admitted_v2(&admitted).expect("open exact baseline");
        let query = make_normalized_vec(dimension, 6.5);
        let expected = flat
            .search_top_k(&query, 5, None)
            .expect("flat exact baseline");

        let unrelated_dir = dir.join("many-unrelated");
        std::fs::create_dir(&unrelated_dir).expect("create private unrelated cache directory");
        for entry in 0..=EXACT_RESIDUAL_CACHE_DIRECTORY_ENTRY_LIMIT {
            write_new_owned_file(
                &unrelated_dir.join(format!("unrelated-{entry:04}")),
                b"not a sidecar",
            );
        }
        let unrelated_entries_before = std::fs::read_dir(&unrelated_dir)
            .expect("read private unrelated cache directory")
            .count();
        let unrelated = InMemoryVectorIndex::from_admitted_v2_with_residual_sidecar_cache(
            &admitted,
            &unrelated_dir,
        )
        .expect("entry-budget exhaustion remains an exact open");
        assert!(
            !unrelated.has_exact_residual_sidecar(),
            "entry-budget exhaustion must skip optional cache attachment"
        );
        assert_eq!(
            unrelated
                .search_top_k(&query, 5, None)
                .expect("unrelated-cache exact search"),
            expected,
            "entry-budget exhaustion must retain the exact flat result"
        );
        assert_eq!(
            std::fs::read_dir(&unrelated_dir)
                .expect("re-read private unrelated cache directory")
                .count(),
            unrelated_entries_before,
            "entry-budget exhaustion skips cache publication"
        );

        let corrupt_dir = dir.join("many-corrupt");
        std::fs::create_dir(&corrupt_dir).expect("create private corrupt cache directory");
        let cache_prefix = flat
            .exact_residual_generation_cache_prefix()
            .expect("derive generation-keyed cache prefix");
        let mut header_valid_corrupt = flat
            .build_exact_residual_sidecar()
            .expect("derive cache-shaped sidecar")
            .encode()
            .expect("encode cache-shaped sidecar");
        header_valid_corrupt[EXACT_RESIDUAL_SIDECAR_HEADER_BYTES] ^= 0x01;
        for entry in 0..=EXACT_RESIDUAL_CACHE_COMPARISON_CANDIDATE_LIMIT {
            write_new_owned_file(
                &corrupt_dir.join(format!("{cache_prefix}corrupt-{entry:04}.fsrs")),
                &header_valid_corrupt,
            );
        }
        let corrupt_entries_before = std::fs::read_dir(&corrupt_dir)
            .expect("read private corrupt cache directory")
            .count();
        let corrupt = InMemoryVectorIndex::from_admitted_v2_with_residual_sidecar_cache(
            &admitted,
            &corrupt_dir,
        )
        .expect("comparison-budget exhaustion remains an exact open");
        assert!(
            !corrupt.has_exact_residual_sidecar(),
            "comparison-budget exhaustion must skip optional cache attachment"
        );
        assert_eq!(
            corrupt
                .search_top_k(&query, 5, None)
                .expect("corrupt-cache exact search"),
            expected,
            "comparison-budget exhaustion must retain the exact flat result"
        );
        assert_eq!(
            std::fs::read_dir(&corrupt_dir)
                .expect("re-read private corrupt cache directory")
                .count(),
            corrupt_entries_before,
            "comparison-budget exhaustion skips cache publication"
        );
    }

    #[test]
    fn from_fsvi_legacy_v1_stays_unidentified() {
        // Legacy v1 files carry no identity bundle. Absence must survive the
        // load as the typed None state (the C4 seams route it as
        // LegacyUnidentified) — never a fabricated fingerprint.
        let path = temp_index_path("from_fsvi_legacy_v1");
        cleanup(&path);
        let dim = 8;
        let mut writer = crate::VectorIndex::create(&path, "legacy-embedder", dim)
            .expect("create legacy v1 writer");
        writer
            .write_record("doc-0", &make_normalized_vec(dim, 1.0))
            .expect("write v1 row");
        writer.finish().expect("finish v1 file");

        let index = InMemoryVectorIndex::from_fsvi(&path).expect("load v1 file into memory");
        assert_eq!(
            index.space_fingerprint_hex(),
            None,
            "legacy v1 absence must stay typed, never fabricated"
        );
        assert!(
            !index.space_identity_is_attested(),
            "a v1 header attests nothing (C4-write guards 2+8)"
        );
        cleanup(&path);
    }

    #[test]
    fn from_fsvi_preserves_embedder_identity_strings() {
        // bd-9xuj T2-C2: from_fsvi used to keep only the dimension and drop
        // the header's embedder id/revision on the floor. Both must survive
        // the load verbatim — including a v1 header's EMPTY revision, which
        // stays `Some("")` (what the header says), distinct from the `None`
        // of a bare-vector build (no header at all).
        let path = temp_index_path("from_fsvi_embedder_identity");
        cleanup(&path);
        let dim = 8;
        let mut writer = crate::VectorIndex::create_with_revision(
            &path,
            "kept-embedder",
            "kept-revision-v7",
            dim,
            Quantization::F16,
        )
        .expect("create v1 writer with revision");
        writer
            .write_record("doc-0", &make_normalized_vec(dim, 1.0))
            .expect("write v1 row");
        writer.finish().expect("finish v1 file");

        let index = InMemoryVectorIndex::from_fsvi(&path).expect("load v1 file into memory");
        assert_eq!(index.embedder_id(), Some("kept-embedder"));
        assert_eq!(index.embedder_revision(), Some("kept-revision-v7"));
        assert_eq!(
            index.space_fingerprint_hex(),
            None,
            "id strings never synthesize a space identity"
        );
        cleanup(&path);

        // Empty-revision v1 header: kept verbatim as Some("").
        let path = temp_index_path("from_fsvi_empty_revision");
        cleanup(&path);
        let mut writer = crate::VectorIndex::create(&path, "legacy-embedder", dim)
            .expect("create legacy v1 writer");
        writer
            .write_record("doc-0", &make_normalized_vec(dim, 1.0))
            .expect("write v1 row");
        writer.finish().expect("finish v1 file");
        let index = InMemoryVectorIndex::from_fsvi(&path).expect("load v1 file into memory");
        assert_eq!(index.embedder_id(), Some("legacy-embedder"));
        assert_eq!(
            index.embedder_revision(),
            Some(""),
            "an empty header revision is the header's content, not absence"
        );
        cleanup(&path);
    }

    #[test]
    fn constructor_embedder_identity_rules() {
        // Bare vectors: no source header, no declared space — both stay the
        // typed None (nothing to retain, nothing fabricated).
        let dim = 8;
        let (doc_ids, vectors) = identity_rows(dim, 2);
        let bare = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).expect("build");
        assert_eq!(bare.embedder_id(), None);
        assert_eq!(bare.embedder_revision(), None);

        // Declared space: the same id/revision rule the production v2 writer
        // uses (`create_v2` writes logical_model_id / immutable_revision).
        let bundle = EmbeddingIdentityBundleV1::explicit_test_model("declared-model", 8);
        let (doc_ids, vectors) = identity_rows(dim, 2);
        let declared =
            InMemoryVectorIndex::from_vectors_with_identity(doc_ids, vectors, dim, &bundle.space)
                .expect("build identified index");
        assert_eq!(declared.embedder_id(), Some("declared-model"));
        assert_eq!(
            declared.embedder_revision(),
            Some(bundle.space.immutable_revision.as_str())
        );
    }

    #[test]
    fn admitted_v2_load_preserves_embedder_identity_strings() {
        // The admitted-v2 in-memory load must retain the artifact header's
        // id/revision strings alongside the space fingerprint C3 already
        // preserves. Own directory: admission fails closed with
        // DirectoryChangedDuringRead when sibling test files churn the dir.
        let dir = std::env::temp_dir()
            .join("frankensearch_in_memory_tests")
            .join("admitted_v2_embedder_identity_dir");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create isolated admission dir");
        let path = dir.join("admitted_v2_embedder_identity.fsvi");
        let dim = 8;
        let (doc_ids, vectors) = identity_rows(dim, 3);
        let rows: Vec<(String, Vec<f32>)> = doc_ids.into_iter().zip(vectors).collect();
        let (binding, expected_space) =
            write_fsvi_v2_fixture(&path, "v2-embedder-model", dim, 13, &rows);

        let admitted = crate::VectorIndex::open_admitted_v2(&path, &binding)
            .expect("exact admission of the v2 fixture");
        let index = InMemoryVectorIndex::from_admitted_v2(&admitted)
            .expect("load admitted v2 artifact into memory");
        // create_v2 writes logical_model_id / immutable_revision as the
        // header strings; explicit_test_model pins the revision value.
        assert_eq!(index.embedder_id(), Some("v2-embedder-model"));
        assert_eq!(index.embedder_revision(), Some("explicit-test-v1"));
        assert_eq!(index.space_fingerprint_hex(), Some(expected_space.as_str()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn in_memory_two_tier_exposes_per_tier_embedder_identity() {
        let dim = 8;
        let fast_bundle = EmbeddingIdentityBundleV1::explicit_test_model("tier-fast-model", 8);
        let (doc_ids, vectors) = identity_rows(dim, 2);
        let fast = InMemoryVectorIndex::from_vectors_with_identity(
            doc_ids,
            vectors,
            dim,
            &fast_bundle.space,
        )
        .expect("build fast tier");
        let (doc_ids, vectors) = identity_rows(dim, 2);
        let quality = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).expect("build");

        // Identified fast + identity-less quality: each tier reports its own
        // state; the identified tier never bleeds into the absent one.
        let two_tier = InMemoryTwoTierIndex::new(fast, Some(quality));
        assert_eq!(two_tier.fast_embedder_id(), Some("tier-fast-model"));
        assert_eq!(
            two_tier.fast_embedder_revision(),
            Some(fast_bundle.space.immutable_revision.as_str())
        );
        assert_eq!(two_tier.quality_embedder_id(), None);
        assert_eq!(two_tier.quality_embedder_revision(), None);

        // No quality tier at all: same typed absence.
        let (doc_ids, vectors) = identity_rows(dim, 2);
        let fast_only = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).expect("build");
        let composed = InMemoryTwoTierIndex::new(fast_only, None);
        assert_eq!(composed.fast_embedder_id(), None);
        assert_eq!(composed.quality_embedder_id(), None);
        assert_eq!(composed.quality_embedder_revision(), None);
    }

    /// First production-shaped consumer of T2-C1's `verify_space_identity`:
    /// an in-memory index built WITH identity supplies the index-side join
    /// key, a query embedding bound through `BoundQueryEmbedding` supplies
    /// the query side, and the verifier decides admission before the real
    /// search path runs.
    #[test]
    fn bound_query_joins_in_memory_space_identity_through_verifier() -> Result<(), String> {
        let dim = 8;
        let fast_bundle = EmbeddingIdentityBundleV1::explicit_test_model("consumer-fast-model", 8);
        let other_bundle =
            EmbeddingIdentityBundleV1::explicit_test_model("consumer-quality-model", 8);

        let (doc_ids, vectors) = identity_rows(dim, 3);
        let index = InMemoryVectorIndex::from_vectors_with_identity(
            doc_ids,
            vectors,
            dim,
            &fast_bundle.space,
        )
        .expect("build fast-space index");
        let (doc_ids, vectors) = identity_rows(dim, 3);
        let wrong_space_index = InMemoryVectorIndex::from_vectors_with_identity(
            doc_ids,
            vectors,
            dim,
            &other_bundle.space,
        )
        .expect("build quality-space index");

        // Query side: bind the vector to the space that produced it (C1).
        let query = make_normalized_vec(dim, 1.0);
        let bound = BoundQueryEmbedding::new(query, fast_bundle).expect("bind query embedding");

        // Matching space: the verifier admits, then the REAL search path runs.
        let fingerprint = index
            .space_fingerprint_hex()
            .expect("identified index exposes its space");
        bound
            .verify_space_identity(fingerprint, "fast")
            .expect("same space must verify");
        let hits = index
            .search_top_k(bound.vector(), 1, None)
            .expect("search admitted query");
        assert_eq!(hits[0].doc_id, "doc-0", "query equals doc-0's vector");

        // The trap this slice exists for: the raw path CANNOT detect a
        // same-dimension wrong-space vector — it searches happily...
        let raw_hits = wrong_space_index
            .search_top_k(bound.vector(), 1, None)
            .expect("raw path accepts same-dimension wrong-space silently");
        assert!(!raw_hits.is_empty());

        // ...and the space-fingerprint join is what closes exactly that hole.
        let wrong_fingerprint = wrong_space_index
            .space_fingerprint_hex()
            .expect("identified index exposes its space");
        let error = bound
            .verify_space_identity(wrong_fingerprint, "quality")
            .expect_err("wrong space at equal dimension must reject");
        let rendered = format!("{error:?}");
        let SearchError::InvalidConfig { field, .. } = error else {
            return Err(format!("expected InvalidConfig, got {rendered}"));
        };
        assert_eq!(field, "query_embedding.quality.space_identity");
        Ok(())
    }

    #[test]
    fn in_memory_two_tier_exposes_per_tier_space_identity() {
        let dim = 8;

        // Composed: each tier built with its own space identity exposes its
        // own, distinct fingerprint through the two-tier wrapper.
        let fast_bundle = EmbeddingIdentityBundleV1::explicit_test_model("two-tier-fast-model", 8);
        let quality_bundle =
            EmbeddingIdentityBundleV1::explicit_test_model("two-tier-quality-model", 8);
        let (doc_ids, vectors) = identity_rows(dim, 3);
        let fast = InMemoryVectorIndex::from_vectors_with_identity(
            doc_ids,
            vectors,
            dim,
            &fast_bundle.space,
        )
        .expect("build fast tier");
        let (doc_ids, vectors) = identity_rows(dim, 3);
        let quality = InMemoryVectorIndex::from_vectors_with_identity(
            doc_ids,
            vectors,
            dim,
            &quality_bundle.space,
        )
        .expect("build quality tier");
        let two_tier = InMemoryTwoTierIndex::new(fast, Some(quality));
        assert_eq!(
            two_tier.fast_space_fingerprint_hex(),
            Some(fast_bundle.space.fingerprint().as_str())
        );
        assert_eq!(
            two_tier.quality_space_fingerprint_hex(),
            Some(quality_bundle.space.fingerprint().as_str())
        );
        assert_ne!(
            two_tier.fast_space_fingerprint_hex(),
            two_tier.quality_space_fingerprint_hex(),
            "distinct models must expose distinct per-tier spaces"
        );

        // from_dir loads through the v1-only pathname loader: absence stays
        // typed through the directory path — no tier identity is fabricated.
        static DIR_NONCE: AtomicU64 = AtomicU64::new(0);
        let nonce = DIR_NONCE.fetch_add(1, AtomicOrdering::Relaxed);
        let dir = std::env::temp_dir()
            .join("frankensearch_in_memory_tests")
            .join(format!("two_tier_space_identity-{nonce}"));
        // Scrub any stale state a previously interrupted run left behind.
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create two-tier dir");
        let fast_path = dir.join(crate::two_tier::VECTOR_INDEX_FAST_FILENAME);
        let mut writer = crate::VectorIndex::create(&fast_path, "legacy-fast", dim)
            .expect("create legacy v1 fast tier");
        writer
            .write_record("doc-0", &make_normalized_vec(dim, 1.0))
            .expect("write v1 row");
        writer.finish().expect("finish v1 fast tier");
        let loaded = InMemoryTwoTierIndex::from_dir(&dir).expect("load v1 two-tier dir");
        assert_eq!(loaded.fast_space_fingerprint_hex(), None);
        assert_eq!(loaded.quality_space_fingerprint_hex(), None);
        let _ = std::fs::remove_dir_all(&dir);

        // new(): a composed two-tier without a quality index has no
        // quality-tier space to verify against — typed absence again.
        let (doc_ids, vectors) = identity_rows(dim, 2);
        let fast_only = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).expect("build");
        let composed = InMemoryTwoTierIndex::new(fast_only, None);
        assert_eq!(composed.fast_space_fingerprint_hex(), None);
        assert_eq!(composed.quality_space_fingerprint_hex(), None);
    }

    #[test]
    fn search_top_k_correctness() {
        let dim = 16;
        let n = 50;
        let doc_ids: Vec<String> = (0..n).map(|i| format!("doc-{i}")).collect();
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|i| make_normalized_vec(dim, i as f32 * 0.7))
            .collect();
        let query = make_normalized_vec(dim, 0.7); // should match doc-1 best

        let index = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();
        let hits = index.search_top_k(&query, 5, None).unwrap();

        assert_eq!(hits.len(), 5);
        // Scores should be descending
        for w in hits.windows(2) {
            assert!(w[0].score >= w[1].score, "scores not descending");
        }
        // Top hit should be doc-1 (same seed as query)
        assert_eq!(hits[0].doc_id, "doc-1");
    }

    #[test]
    fn int8_two_pass_matches_exact_topk() {
        let dim = 32;
        let n = 200;
        let doc_ids: Vec<String> = (0..n).map(|i| format!("doc-{i}")).collect();
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|i| make_normalized_vec(dim, i as f32 * 0.31))
            .collect();
        let index = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();

        for qseed in [0.31_f32, 3.0, 17.5, 99.9] {
            let query = make_normalized_vec(dim, qseed);
            let exact = index.search_top_k(&query, 10, None).unwrap();
            // mult=10 -> 100 candidates of 200; pass-1 recall is 1 here, so the
            // two-pass result must be bit-identical to the exact top-k.
            let two_pass = index.search_top_k_int8_two_pass(&query, 10, 10).unwrap();

            assert_eq!(two_pass.len(), exact.len(), "qseed={qseed}");
            for w in two_pass.windows(2) {
                assert!(w[0].score >= w[1].score, "two-pass not descending");
            }
            let exact_ids: Vec<&str> = exact.iter().map(|h| h.doc_id.as_str()).collect();
            let tp_ids: Vec<&str> = two_pass.iter().map(|h| h.doc_id.as_str()).collect();
            assert_eq!(
                tp_ids, exact_ids,
                "int8 two-pass should match exact top-k at mult=10 (qseed={qseed})"
            );
            for (a, b) in two_pass.iter().zip(exact.iter()) {
                assert!((a.score - b.score).abs() < 1e-6, "scores differ");
            }
        }
    }

    #[test]
    fn four_bit_two_pass_keep_all_matches_exact() {
        // With a multiplier large enough to retain every record, the exact f16
        // rescore must reproduce `search_top_k` bit-for-bit — verifying the nibble
        // pack/unpack, offsets, rescore, and resolve.
        let dim = 34; // odd-ish, > 32, exercises a partial last packed byte
        let n = 200;
        let doc_ids: Vec<String> = (0..n).map(|i| format!("doc-{i}")).collect();
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|i| make_normalized_vec(dim, i as f32 * 0.31))
            .collect();
        let index = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();

        for qseed in [0.31_f32, 3.0, 17.5, 99.9] {
            let query = make_normalized_vec(dim, qseed);
            let exact = index.search_top_k(&query, 10, None).unwrap();
            // mult=20 → candidate_count clamps to n → pass-1 retains all → identical.
            let two_pass = index.search_top_k_4bit_two_pass(&query, 10, 20).unwrap();
            let exact_ids: Vec<&str> = exact.iter().map(|h| h.doc_id.as_str()).collect();
            let tp_ids: Vec<&str> = two_pass.iter().map(|h| h.doc_id.as_str()).collect();
            assert_eq!(
                tp_ids, exact_ids,
                "4bit two-pass (keep-all) should match exact top-k (qseed={qseed})"
            );
        }
    }

    #[test]
    fn int8_two_pass_dimension_mismatch() {
        let dim = 8;
        let doc_ids: Vec<String> = (0..4).map(|i| format!("doc-{i}")).collect();
        let vectors: Vec<Vec<f32>> = (0..4).map(|i| make_normalized_vec(dim, i as f32)).collect();
        let index = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();
        let err = index
            .search_top_k_int8_two_pass(&[1.0; 7], 3, 4)
            .expect_err("dimension mismatch");
        assert!(matches!(err, SearchError::DimensionMismatch { .. }));
    }

    #[test]
    fn search_top_k_with_filter() {
        let dim = 8;
        let doc_ids: Vec<String> = (0..10).map(|i| format!("doc-{i}")).collect();
        let vectors: Vec<Vec<f32>> = (0..10)
            .map(|i| make_normalized_vec(dim, i as f32))
            .collect();
        let query = make_normalized_vec(dim, 0.0); // matches doc-0

        struct OddFilter;
        impl SearchFilter for OddFilter {
            fn matches(&self, doc_id: &str, _metadata: Option<&serde_json::Value>) -> bool {
                // Only allow odd-numbered docs
                doc_id
                    .strip_prefix("doc-")
                    .and_then(|n| n.parse::<usize>().ok())
                    .is_some_and(|n| n % 2 == 1)
            }
            fn name(&self) -> &str {
                "odd"
            }
        }

        let index = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();
        let hits = index.search_top_k(&query, 5, Some(&OddFilter)).unwrap();
        assert_eq!(hits.len(), 5);
        for hit in &hits {
            let num: usize = hit.doc_id.strip_prefix("doc-").unwrap().parse().unwrap();
            assert!(num % 2 == 1, "filter should exclude even docs");
        }
    }

    #[test]
    fn search_with_bitset_filter_uses_precomputed_hash_path() {
        // BitsetFilter resolves via matches_doc_id_hash (the precomputed-hash
        // prescreen). Result must equal the allowed doc-id set's top-k — i.e. the
        // precomputed hashes match BitsetFilter's own hashing.
        use frankensearch_core::filter::BitsetFilter;
        let dim = 8;
        let doc_ids: Vec<String> = (0..20).map(|i| format!("doc-{i}")).collect();
        let vectors: Vec<Vec<f32>> = (0..20)
            .map(|i| make_normalized_vec(dim, i as f32))
            .collect();
        let allowed: Vec<String> = doc_ids.iter().step_by(3).cloned().collect(); // doc-0,3,6,...
        let index = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();
        let filter = BitsetFilter::from_doc_ids(allowed.iter().cloned());

        let query = make_normalized_vec(dim, 6.0);
        let hits = index.search_top_k(&query, 20, Some(&filter)).unwrap();

        assert!(!hits.is_empty());
        for hit in &hits {
            assert!(
                allowed.iter().any(|a| a.as_str() == hit.doc_id.as_str()),
                "bitset filter must only return allowed doc-ids; got {}",
                hit.doc_id
            );
        }
        // Every allowed doc should be returned (limit 20 ≥ allowed count).
        assert_eq!(hits.len(), allowed.len());
    }

    #[test]
    fn int8_two_pass_filtered_matches_exact_filtered() {
        // The filtered int8 two-pass must return the same top-k as the exact
        // filtered scan (pass-1 pre-screens by the same doc_id hash; lossless when
        // pass-1 retains the true filtered top-k at a generous multiplier).
        use frankensearch_core::filter::BitsetFilter;
        let dim = 16;
        let doc_ids: Vec<String> = (0..200).map(|i| format!("doc-{i:04}")).collect();
        let vectors: Vec<Vec<f32>> = (0..200)
            .map(|i| make_normalized_vec(dim, i as f32))
            .collect();
        let allowed: Vec<String> = doc_ids.iter().step_by(2).cloned().collect();
        let filter = BitsetFilter::from_doc_ids(allowed.iter().cloned());
        let index = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();

        for qseed in [3.0_f32, 17.0, 88.0] {
            let query = make_normalized_vec(dim, qseed);
            let exact: Vec<String> = index
                .search_top_k(&query, 10, Some(&filter))
                .unwrap()
                .into_iter()
                .map(|h| h.doc_id.to_string())
                .collect();
            let two_pass: Vec<String> = index
                .search_top_k_int8_two_pass_filtered(&query, 10, 10, Some(&filter))
                .unwrap()
                .into_iter()
                .map(|h| h.doc_id.to_string())
                .collect();
            // Only allowed docs, and identical to the exact filtered top-k.
            for id in &two_pass {
                assert!(allowed.contains(id), "two-pass returned filtered-out {id}");
            }
            assert_eq!(
                two_pass, exact,
                "filtered two-pass != exact (qseed={qseed})"
            );
        }
    }

    #[test]
    fn selective_filter_gather_matches_scan() {
        // A selective hash-addressable filter takes the gather fast-path through
        // `search_top_k`/the two-pass filtered fns; it must be bit-identical to the
        // forced per-document filtered scan (same passing set, `(score,index)` order).
        use frankensearch_core::filter::BitsetFilter;
        let dim = 16;
        let doc_ids: Vec<String> = (0..500).map(|i| format!("doc-{i:04}")).collect();
        let vectors: Vec<Vec<f32>> = (0..500)
            .map(|i| make_normalized_vec(dim, i as f32))
            .collect();
        // ~5% allow-set (well under the selectivity gate) → gather path is taken.
        let allowed: Vec<String> = doc_ids.iter().step_by(20).cloned().collect();
        let filter = BitsetFilter::from_doc_ids(allowed.iter().cloned());
        let index = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();

        let ids = |hits: Vec<VectorHit>| -> Vec<String> {
            hits.into_iter().map(|h| h.doc_id.to_string()).collect()
        };
        for qseed in [1.0_f32, 42.0, 313.0] {
            let query = make_normalized_vec(dim, qseed);
            let scan = ids(index
                .bench_scan_filtered(&query, 10, Some(&filter))
                .unwrap());
            let gather = ids(index.bench_gather_filtered(&query, 10, &filter).unwrap());
            let public = ids(index.search_top_k(&query, 10, Some(&filter)).unwrap());
            let int8 = ids(index
                .search_top_k_int8_two_pass_filtered(&query, 10, 3, Some(&filter))
                .unwrap());
            let fourbit = ids(index
                .search_top_k_4bit_two_pass_filtered(&query, 10, 3, Some(&filter))
                .unwrap());
            for id in &gather {
                assert!(allowed.contains(id), "gather returned filtered-out {id}");
            }
            assert_eq!(gather, scan, "gather != scan (qseed={qseed})");
            assert_eq!(public, scan, "search_top_k gather != scan (qseed={qseed})");
            assert_eq!(
                int8, scan,
                "int8 two-pass gather != exact scan (qseed={qseed})"
            );
            assert_eq!(
                fourbit, scan,
                "4bit two-pass gather != exact scan (qseed={qseed})"
            );
        }
    }

    #[test]
    fn parallel_gather_matches_scan() {
        // Allow-set larger than PARALLEL_CHUNK_SIZE → the gather runs its parallel
        // per-chunk-heap + merge path, which must stay bit-identical to the scan.
        use frankensearch_core::filter::BitsetFilter;
        let dim = 16;
        let n = 3000; // > PARALLEL_CHUNK_SIZE allow-set below forces the parallel path
        let doc_ids: Vec<String> = (0..n).map(|i| format!("doc-{i:05}")).collect();
        let vectors: Vec<Vec<f32>> = (0..n).map(|i| make_normalized_vec(dim, i as f32)).collect();
        // ~half the corpus (1500 > 1024 chunk size) → parallel gather.
        let allowed: Vec<String> = doc_ids.iter().step_by(2).cloned().collect();
        assert!(
            allowed.len() > PARALLEL_CHUNK_SIZE,
            "must exceed chunk size"
        );
        let filter = BitsetFilter::from_doc_ids(allowed.iter().cloned());
        let index = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();
        let ids = |hits: Vec<VectorHit>| -> Vec<String> {
            hits.into_iter().map(|h| h.doc_id.to_string()).collect()
        };
        for qseed in [2.0_f32, 99.0, 1234.0] {
            let query = make_normalized_vec(dim, qseed);
            let scan = ids(index
                .bench_scan_filtered(&query, 25, Some(&filter))
                .unwrap());
            let gather = ids(index.bench_gather_filtered(&query, 25, &filter).unwrap());
            assert_eq!(gather, scan, "parallel gather != scan (qseed={qseed})");
        }
    }

    #[test]
    fn search_empty_index() {
        let index = InMemoryVectorIndex::from_vectors(Vec::new(), Vec::new(), 4).unwrap();
        let hits = index.search_top_k(&[0.0, 0.0, 0.0, 0.0], 10, None).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn search_dimension_mismatch() {
        let index = InMemoryVectorIndex::from_vectors(
            vec!["a".into()],
            vec![make_normalized_vec(4, 1.0)],
            4,
        )
        .unwrap();
        let result = index.search_top_k(&[1.0, 0.0], 10, None); // dim 2 != 4
        assert!(result.is_err());
    }

    #[test]
    fn f16_precision_tolerance() {
        let dim = 256;
        let v = make_normalized_vec(dim, 42.0);
        let index =
            InMemoryVectorIndex::from_vectors(vec!["test".into()], vec![v.clone()], dim).unwrap();

        // Self-similarity should be ~1.0 (within f16 precision)
        let hits = index.search_top_k(&v, 1, None).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(
            (hits[0].score - 1.0).abs() < 0.001,
            "f16 self-similarity should be within 0.001 of 1.0, got {}",
            hits[0].score
        );
    }

    #[test]
    fn vector_at_f32_roundtrip() {
        let dim = 8;
        let original = make_normalized_vec(dim, 5.0);
        let index =
            InMemoryVectorIndex::from_vectors(vec!["a".into()], vec![original.clone()], dim)
                .unwrap();
        let recovered = index.vector_at_f32(0).unwrap();
        assert_eq!(recovered.len(), dim);
        for (orig, rec) in original.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 0.002, "f16 round-trip error too large");
        }
    }

    #[test]
    fn two_tier_search_fast() {
        let dim = 8;
        let n = 20;
        let doc_ids: Vec<String> = (0..n).map(|i| format!("doc-{i}")).collect();
        let vectors: Vec<Vec<f32>> = (0..n).map(|i| make_normalized_vec(dim, i as f32)).collect();
        let query = make_normalized_vec(dim, 5.0);

        let fast = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();
        let two_tier = InMemoryTwoTierIndex::new(fast, None);

        assert!(!two_tier.has_quality_index());
        assert_eq!(two_tier.doc_count(), 20);

        let hits = two_tier.search_fast(&query, 5).unwrap();
        assert_eq!(hits.len(), 5);
        assert_eq!(hits[0].doc_id, "doc-5");
    }

    #[test]
    fn two_tier_quality_scores() {
        let dim_fast = 8;
        let dim_quality = 16;
        let n = 10;

        let doc_ids: Vec<String> = (0..n).map(|i| format!("doc-{i}")).collect();
        let fast_vecs: Vec<Vec<f32>> = (0..n)
            .map(|i| make_normalized_vec(dim_fast, i as f32))
            .collect();
        let quality_vecs: Vec<Vec<f32>> = (0..n)
            .map(|i| make_normalized_vec(dim_quality, i as f32 * 0.5))
            .collect();

        let fast = InMemoryVectorIndex::from_vectors(doc_ids.clone(), fast_vecs, dim_fast).unwrap();
        let quality =
            InMemoryVectorIndex::from_vectors(doc_ids, quality_vecs, dim_quality).unwrap();

        let two_tier = InMemoryTwoTierIndex::new(fast, Some(quality));
        assert!(two_tier.has_quality_index());

        let fast_query = make_normalized_vec(dim_fast, 3.0);
        let hits = two_tier.search_fast(&fast_query, 5).unwrap();

        let quality_query = make_normalized_vec(dim_quality, 1.5);
        let scores = two_tier
            .quality_scores_for_hits(&quality_query, &hits)
            .unwrap();
        assert_eq!(scores.len(), 5);
        // All scores should be Some and finite
        for s in &scores {
            assert!(
                s.is_some_and(|v| v.is_finite()),
                "quality score should be Some and finite"
            );
        }
    }

    #[test]
    fn two_tier_no_quality_returns_nones() {
        let dim = 4;
        let fast = InMemoryVectorIndex::from_vectors(
            vec!["a".into()],
            vec![make_normalized_vec(dim, 1.0)],
            dim,
        )
        .unwrap();
        let two_tier = InMemoryTwoTierIndex::new(fast, None);

        let hits = two_tier
            .search_fast(&make_normalized_vec(dim, 1.0), 1)
            .unwrap();
        let scores = two_tier
            .quality_scores_for_hits(&make_normalized_vec(dim, 1.0), &hits)
            .unwrap();
        assert_eq!(scores, vec![None]);
    }

    #[test]
    fn parallel_search_matches_sequential() {
        let dim = 16;
        let n = 200;
        let doc_ids: Vec<String> = (0..n).map(|i| format!("doc-{i}")).collect();
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|i| make_normalized_vec(dim, i as f32 * 0.3))
            .collect();
        let query = make_normalized_vec(dim, 7.0);

        let index = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();

        let seq_params = SearchParams {
            parallel_enabled: false,
            parallel_threshold: 1,
            parallel_chunk_size: 32,
        };
        let par_params = SearchParams {
            parallel_enabled: true,
            parallel_threshold: 1, // force parallel even for small index
            parallel_chunk_size: 32,
        };

        let seq_hits = index
            .search_top_k_with_params(&query, 10, None, seq_params)
            .unwrap();
        let par_hits = index
            .search_top_k_with_params(&query, 10, None, par_params)
            .unwrap();

        assert_eq!(seq_hits.len(), par_hits.len());
        for (s, p) in seq_hits.iter().zip(par_hits.iter()) {
            assert_eq!(s.doc_id, p.doc_id);
            assert!(
                (s.score - p.score).abs() < 1e-6,
                "parallel vs sequential score mismatch"
            );
        }
    }

    #[test]
    fn exact_residual_sidecar_is_exact_on_odd_tail_ties_filters_and_repeated_queries() {
        use frankensearch_core::PredicateFilter;

        let dimension = 35; // exercises the final partial 32-dimension block.
        let count = 17; // exercises the final partial eight-candidate group.
        let doc_ids: Vec<String> = (0..count).map(|index| format!("doc-{index:02}")).collect();
        let mut vectors: Vec<Vec<f32>> = (0..count)
            .map(|index| make_normalized_vec(dimension, index as f32 * 0.37))
            .collect();
        // Exact score ties must retain the incumbent index tie-break; a sidecar
        // may only abandon on a strict proven upper-bound inequality.
        vectors[1] = vectors[0].clone();
        let mut index = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dimension)
            .expect("finite source vectors");
        assert!(
            index.build_exact_residual_sidecar().is_err(),
            "caller-built vectors have no FSVI v2 generation witness"
        );
        index.residual_source_binding = Some(ResidualSourceBinding {
            generation_fingerprint: [0x11; 32],
            vector_content_digest: [0x22; 32],
            ordered_live_docset_digest: [0x33; 32],
            space_fingerprint: [0x44; 32],
        });
        let sidecar = index
            .build_exact_residual_sidecar()
            .expect("test-only witnessed source builds sidecar");
        let encoded = sidecar.encode().expect("encode sidecar");
        let decoded = ExactResidualSidecar::decode(&encoded).expect("sidecar round trip");
        assert!(
            decoded.is_bound_to(
                index
                    .residual_source_binding
                    .as_ref()
                    .expect("test witness"),
                count,
                dimension,
            )
        );
        let mut corrupt = encoded.clone();
        corrupt[8] ^= 0x01; // corrupt the fixed-width schema version.
        assert!(
            ExactResidualSidecar::decode(&corrupt).is_err(),
            "a corrupted sidecar is rejected before it can affect search"
        );

        let flat = index.clone();
        assert!(
            index
                .admit_exact_residual_sidecar(decoded)
                .expect("admit source-derived sidecar")
        );
        let filter = PredicateFilter::new("even ids", |id| {
            id.strip_prefix("doc-")
                .and_then(|suffix| suffix.parse::<usize>().ok())
                .is_some_and(|value| value % 2 == 0)
        });
        let queries = [
            make_normalized_vec(dimension, 0.0),
            make_normalized_vec(dimension, 3.1),
            make_normalized_vec(dimension, 7.9),
        ];
        for query in &queries {
            for &limit in &[0, 1, count, count + 7] {
                for filter in [None, Some(&filter as &dyn SearchFilter)] {
                    let expected = flat
                        .search_top_k(query, limit, filter)
                        .expect("flat search");
                    let actual = index
                        .search_top_k(query, limit, filter)
                        .expect("sidecar search");
                    assert_eq!(actual.len(), expected.len());
                    for (actual_hit, expected_hit) in actual.iter().zip(&expected) {
                        assert_eq!(actual_hit.doc_id, expected_hit.doc_id);
                        assert_eq!(actual_hit.index, expected_hit.index);
                        assert_eq!(actual_hit.score.to_bits(), expected_hit.score.to_bits());
                    }
                }
            }
            let first = index.search_top_k(query, 5, None).expect("first search");
            let second = index.search_top_k(query, 5, None).expect("repeat search");
            assert_eq!(first, second, "sidecar search is repeatable");
        }
        let parallel = SearchParams {
            parallel_enabled: true,
            parallel_threshold: 1,
            parallel_chunk_size: 4,
        };
        let query = make_normalized_vec(dimension, 5.5);
        let expected = flat
            .search_top_k_with_params(&query, 5, None, parallel)
            .expect("parallel flat search");
        let actual = index
            .search_top_k_with_params(&query, 5, None, parallel)
            .expect("parallel sidecar fallback");
        assert_eq!(actual.len(), expected.len());
        for (actual_hit, expected_hit) in actual.iter().zip(&expected) {
            assert_eq!(actual_hit.doc_id, expected_hit.doc_id);
            assert_eq!(actual_hit.index, expected_hit.index);
            assert_eq!(actual_hit.score.to_bits(), expected_hit.score.to_bits());
        }
    }

    #[test]
    fn exact_residual_preparation_failures_keep_public_search_flat_exact() {
        let dimension = 35;
        let count = 17;
        let mut cached = InMemoryVectorIndex::from_vectors(
            (0..count).map(|row| format!("fallback-{row}")).collect(),
            (0..count)
                .map(|row| make_normalized_vec(dimension, row as f32 + 0.75))
                .collect(),
            dimension,
        )
        .expect("create finite source");
        bind_test_residual_source(&mut cached);
        let uncached = cached.clone();
        let sidecar = cached
            .build_exact_residual_sidecar()
            .expect("build witnessed sidecar");
        assert!(
            cached
                .admit_exact_residual_sidecar(sidecar)
                .expect("admit verified sidecar")
        );
        let query = make_normalized_vec(dimension, 4.25);
        let modes = [
            SearchParams {
                parallel_enabled: false,
                parallel_threshold: 1,
                parallel_chunk_size: 8,
            },
            SearchParams {
                parallel_enabled: true,
                parallel_threshold: 1,
                parallel_chunk_size: 8,
            },
        ];

        let injections: [(&str, fn(), fn() -> bool); 2] = [
            (
                "transform",
                fail_next_residual_query_transform_allocation,
                residual_query_transform_allocation_failure_is_pending,
            ),
            (
                "suffix",
                fail_next_residual_query_suffix_allocation,
                residual_query_suffix_allocation_failure_is_pending,
            ),
        ];
        for params in modes {
            let expected = uncached
                .search_top_k_with_params(&query, 3, None, params)
                .expect("uncached public exact search");
            for (site, inject, is_pending) in injections {
                inject();
                let actual = cached
                    .search_top_k_with_params(&query, 3, None, params)
                    .expect("cache-only preparation failure must not escape public search");
                assert!(
                    !is_pending(),
                    "{site} injection must be consumed by the public cached route"
                );
                assert_eq!(actual.len(), expected.len());
                for (actual, expected) in actual.iter().zip(&expected) {
                    assert_eq!(actual.doc_id, expected.doc_id, "site={site}");
                    assert_eq!(actual.index, expected.index, "site={site}");
                    assert_eq!(
                        actual.score.to_bits(),
                        expected.score.to_bits(),
                        "site={site}"
                    );
                }
            }
        }
    }

    #[test]
    fn exact_residual_sidecar_rejects_finite_transformed_payload_mutations() {
        // The first group establishes a low exact cutoff.  The second group
        // contains the true winner, so a forged zero residual envelope would
        // incorrectly abandon it if transformed data were admitted on header
        // identity alone.
        let dimension = 2;
        let mut vectors = vec![vec![0.1, 0.0]; 16];
        vectors[8] = vec![1.0, 0.0];
        let mut index = InMemoryVectorIndex::from_vectors(
            (0..16).map(|row| format!("doc-{row:02}")).collect(),
            vectors,
            dimension,
        )
        .expect("finite source vectors");
        index.residual_source_binding = Some(ResidualSourceBinding {
            generation_fingerprint: [0x71; 32],
            vector_content_digest: [0x72; 32],
            ordered_live_docset_digest: [0x73; 32],
            space_fingerprint: [0x74; 32],
        });
        let expected_sidecar = index
            .build_exact_residual_sidecar()
            .expect("build source-derived sidecar");
        let layout = ExactResidualLayout::for_shape(index.record_count(), dimension)
            .expect("bounded test layout");
        let encoded = expected_sidecar.encode().expect("encode sidecar");

        // A finite corruption in every transformed payload family must fail the
        // whole-sidecar SHA-256 check before decode allocates its vectors.
        let permutation_start = EXACT_RESIDUAL_SIDECAR_HEADER_BYTES;
        let centroids_start = permutation_start + layout.permutation_len * 4;
        let residuals_start = centroids_start + layout.centroid_len * 4;
        let suffixes_start = residuals_start + layout.residual_len * 4;
        let second_group_residual = residuals_start + dimension * EXACT_RESIDUAL_LANES * 4;
        let second_group_suffix = suffixes_start + EXACT_RESIDUAL_LANES * 2 * 4;
        for offset in [
            permutation_start,
            second_group_residual,
            second_group_suffix,
        ] {
            let mut corrupt = encoded.clone();
            if offset == permutation_start {
                corrupt.swap(offset, offset + 4);
            } else {
                corrupt[offset..offset + 4].copy_from_slice(&0.0_f32.to_le_bytes());
            }
            assert!(
                ExactResidualSidecar::decode(&corrupt).is_err(),
                "digest must reject finite transformed corruption at byte {offset}"
            );
        }
        let mut oversized_header = encoded[..EXACT_RESIDUAL_SIDECAR_HEADER_BYTES].to_vec();
        let dimension_offset = 8 + 4 + 4 * 32 + 8;
        oversized_header[dimension_offset..dimension_offset + 8].copy_from_slice(
            &u64::try_from(EXACT_RESIDUAL_SIDECAR_MAX_DIMENSION + 1)
                .expect("test dimension fits u64")
                .to_le_bytes(),
        );
        assert!(
            ExactResidualSidecar::decode(&oversized_header).is_err(),
            "an oversized declared transform is rejected before decode allocation"
        );

        // A writer who recomputes the digest still cannot attach a finite
        // forged transform: admission deterministically re-derives it from the
        // admitted f16 source and compares every float bit and permutation.
        let mut forged = expected_sidecar.clone();
        forged.permutation.swap(0, 1);
        for value in &mut forged.centroids[dimension..] {
            *value = 0.0;
        }
        for value in &mut forged.residuals[dimension * EXACT_RESIDUAL_LANES..] {
            *value = 0.0;
        }
        for value in &mut forged.suffix_norms[EXACT_RESIDUAL_LANES * 2..] {
            *value = 0.0;
        }
        for value in &mut forged.correction_norms[EXACT_RESIDUAL_LANES..] {
            *value = 0.0;
        }
        let forged = ExactResidualSidecar::decode(
            &forged
                .encode()
                .expect("re-encode a finite, self-consistent forgery"),
        )
        .expect("a rehashed forgery is structurally decodable");
        assert!(
            !forged.exactly_matches_derived(&expected_sidecar),
            "finite transformed contents differ from the admitted derivation"
        );
        let query = [1.0, 0.0];
        let expected = index.search_top_k(&query, 1, None).expect("flat result");
        let forged_outcome = index
            .scan_exact_residual_sidecar(&query, 1, None, &forged)
            .expect("demonstrate the otherwise-dangerous forged scan");
        let forged_hits = index
            .resolve_heap(forged_outcome.heap)
            .expect("resolve forged scan");
        assert_ne!(
            forged_hits, expected,
            "the planted finite forgery would corrupt the exact winner without re-derivation"
        );
        assert!(
            !index
                .admit_exact_residual_sidecar(forged)
                .expect("reject rehashed transformed forgery")
        );
        assert!(index.exact_residual_sidecar.get().is_none());
    }

    #[test]
    fn exact_residual_sidecar_keeps_empty_and_nonfinite_query_contracts() {
        let dimension = 3;
        let mut empty = InMemoryVectorIndex::from_vectors(Vec::new(), Vec::new(), dimension)
            .expect("empty index is valid");
        empty.residual_source_binding = Some(ResidualSourceBinding {
            generation_fingerprint: [0x81; 32],
            vector_content_digest: [0x82; 32],
            ordered_live_docset_digest: [0x83; 32],
            space_fingerprint: [0x84; 32],
        });
        let sidecar = empty
            .build_exact_residual_sidecar()
            .expect("empty exact sidecar is structurally valid");
        assert!(
            empty
                .admit_exact_residual_sidecar(sidecar)
                .expect("admit empty source-derived sidecar")
        );
        assert!(
            empty
                .search_top_k(&[1.0, 0.0, 0.0], 1, None)
                .expect("empty sidecar search")
                .is_empty()
        );
        let nonfinite = [f32::NAN, 0.0, 0.0];
        assert!(
            empty.search_top_k_classified(&nonfinite, 1, None).is_err(),
            "the classified entry point rejects non-finite queries before any sidecar route"
        );
    }

    #[test]
    fn exact_residual_sidecar_corruption_and_source_mismatch_leave_flat_fallback_selected() {
        let dimension = 9;
        let mut index = InMemoryVectorIndex::from_vectors(
            vec!["a".into(), "b".into(), "c".into()],
            vec![
                make_normalized_vec(dimension, 1.0),
                make_normalized_vec(dimension, 2.0),
                make_normalized_vec(dimension, 3.0),
            ],
            dimension,
        )
        .expect("finite vectors");
        let query = make_normalized_vec(dimension, 1.5);
        let expected = index.search_top_k(&query, 2, None).expect("flat fallback");
        index.residual_source_binding = Some(ResidualSourceBinding {
            generation_fingerprint: [0x51; 32],
            vector_content_digest: [0x52; 32],
            ordered_live_docset_digest: [0x53; 32],
            space_fingerprint: [0x54; 32],
        });
        let mut foreign = index
            .build_exact_residual_sidecar()
            .expect("build sidecar")
            .encode()
            .expect("encode sidecar");
        // The first source digest starts immediately after magic/version and the
        // four source digests are admission-critical, not advisory metadata.
        foreign[12 + 32] ^= 0x80;
        let digest_start = foreign.len() - EXACT_RESIDUAL_SIDECAR_DIGEST_BYTES;
        let digest = Sha256::digest(&foreign[..digest_start]);
        foreign[digest_start..].copy_from_slice(&digest);
        let foreign = ExactResidualSidecar::decode(&foreign).expect("well-formed foreign sidecar");
        assert!(
            !foreign.is_bound_to(
                index
                    .residual_source_binding
                    .as_ref()
                    .expect("source binding"),
                index.record_count(),
                dimension,
            ),
            "a sidecar from another generation cannot be attached"
        );
        assert!(
            !index
                .admit_exact_residual_sidecar(foreign)
                .expect("foreign sidecar is rejected after re-derivation")
        );
        assert!(index.exact_residual_sidecar.get().is_none());
        let actual = index
            .search_top_k(&query, 2, None)
            .expect("flat remains selected");
        assert_eq!(actual, expected);
    }

    #[test]
    fn exact_residual_upper_bound_covers_f16_extremes_partial_tails_and_signed_zero() {
        // Two partial shapes (35 dimensions and 17 rows) force both a tail
        // residual block and a tail lane group.  The bit patterns pin the
        // f16-specific boundary families that are easy to lose in a generic
        // floating-point differential: max finite values, signed zeros, and
        // both signs of the smallest subnormal.
        let dimension = 35;
        let count = 17;
        let special = [
            f16::MAX.to_bits(),
            (-f16::MAX).to_bits(),
            0x0000,
            0x8000,
            0x0001,
            0x8001,
            f16::from_f32(1.0).to_bits(),
            f16::from_f32(-1.0).to_bits(),
        ];
        let vectors: Vec<Vec<f32>> = (0..count)
            .map(|row| {
                (0..dimension)
                    .map(|column| finite_f16_bits(special[(row * 3 + column) % special.len()]))
                    .collect()
            })
            .collect();
        let query: Vec<f32> = (0..dimension)
            .map(|column| finite_f16_bits(special[(column * 5 + 1) % special.len()]))
            .collect();
        let mut index = InMemoryVectorIndex::from_vectors(
            (0..count).map(|row| format!("edge-{row}")).collect(),
            vectors,
            dimension,
        )
        .expect("finite f16 edge source");
        assert_eq!(index.vector_slice(0)[3].to_bits(), 0x8000);
        assert_eq!(index.vector_slice(0)[4].to_bits(), 0x0001);
        assert_eq!(index.vector_slice(0)[5].to_bits(), 0x8001);
        bind_test_residual_source(&mut index);
        let sidecar = index
            .build_exact_residual_sidecar()
            .expect("bounded source-derived sidecar");
        assert_residual_bounds_cover_exact_scores(&index, &sidecar, &query);
    }

    #[test]
    fn exact_residual_upper_bound_covers_large_finite_queries_and_tail_construction_error() {
        // A five-lane tail forces a non-power-of-two centroid division. Its
        // f32 residuals therefore exercise the f64 construction remainder
        // persisted in `correction_norms`, while the query magnitude stays
        // just inside the analytically admitted f32-flat envelope.
        let dimension = 35;
        let tail = [
            f16::MAX.to_f32(),
            1.0,
            -1.0,
            0.0,
            f16::from_bits(1).to_f32(),
        ];
        let vectors: Vec<Vec<f32>> = (0..13)
            .map(|row| {
                (0..dimension)
                    .map(|column| tail[(row + column * 3) % tail.len()])
                    .collect()
            })
            .collect();
        let large_component =
            f32::MAX / (f16::MAX.to_f32() * dimension as f32 * EXACT_RESIDUAL_LANES as f32);
        let query: Vec<f32> = (0..dimension)
            .map(|column| {
                if column % 2 == 0 {
                    large_component
                } else {
                    -large_component * 0.5
                }
            })
            .collect();
        let mut index = InMemoryVectorIndex::from_vectors(
            (0..vectors.len())
                .map(|row| format!("construction-{row}"))
                .collect(),
            vectors,
            dimension,
        )
        .expect("finite f16 construction source");
        bind_test_residual_source(&mut index);
        let sidecar = index
            .build_exact_residual_sidecar()
            .expect("construct a tail-aware sidecar");
        let transformed =
            ResidualQueryTransform::from_query(&query, &sidecar).expect("finite large transform");
        assert!(
            transformed.f32_flat_envelope_is_finite,
            "the large finite query must exercise the bound rather than its flat fallback"
        );
        assert_residual_bounds_cover_exact_scores(&index, &sidecar, &query);
    }

    #[test]
    fn exact_residual_near_cutoff_simd_differential_keeps_the_true_winner() {
        // The first group establishes a cutoff one f16 ULP below the winner.
        // The query is scaled so the authoritative f16×f32 SIMD score is near
        // the top finite f32 range while remaining inside the sidecar's flat
        // envelope. This makes the distinction only a few hundred output ULPs
        // and exercises the real SIMD accumulation/reduction route, not a
        // scalar reference sum.
        let dimension = 35;
        let high = f16::MAX.to_f32();
        let one_f16_ulp_lower = f16::from_bits(f16::MAX.to_bits() - 1).to_f32();
        let query_component = f32::MAX / (high * dimension as f32 * 8.0);
        let query = vec![query_component; dimension];
        let mut near_cutoff = vec![high; dimension];
        near_cutoff[0] = one_f16_ulp_lower;
        let mut vectors = vec![near_cutoff.clone(); EXACT_RESIDUAL_LANES];
        vectors.push(vec![high; dimension]); // row 8, the true winner.
        vectors.extend(vec![near_cutoff; EXACT_RESIDUAL_LANES - 1]);
        // A uniform negative group gives the eliminator a real safe prune while
        // the previous group tests that the near-cutoff winner survives.
        vectors.extend(vec![vec![-high; dimension]; EXACT_RESIDUAL_LANES]);
        let mut index = InMemoryVectorIndex::from_vectors(
            (0..vectors.len())
                .map(|row| format!("simd-near-cutoff-{row}"))
                .collect(),
            vectors,
            dimension,
        )
        .expect("finite f16 SIMD differential source");
        bind_test_residual_source(&mut index);
        let flat = index.clone();
        let sidecar = index
            .build_exact_residual_sidecar()
            .expect("build witnessed SIMD sidecar");
        assert_residual_bounds_cover_exact_scores(&index, &sidecar, &query);
        assert!(
            index
                .admit_exact_residual_sidecar(sidecar)
                .expect("admit exact SIMD sidecar")
        );

        let incumbent = dot_product_f16_f32(flat.vector_slice(0), &query)
            .expect("authoritative SIMD incumbent score");
        let winner = dot_product_f16_f32(flat.vector_slice(EXACT_RESIDUAL_LANES), &query)
            .expect("authoritative SIMD winner score");
        assert!(winner.is_finite() && winner > f32::MAX * 0.1);
        assert!(winner > incumbent, "one f16 ULP must make row 8 win");
        assert!(
            winner - incumbent <= (winner.next_up() - winner) * 512.0,
            "fixture must remain near the authoritative SIMD cutoff"
        );

        let expected = flat
            .search_top_k(&query, 1, None)
            .expect("flat SIMD exact result");
        let outcome = index
            .scan_exact_residual_sidecar(
                &query,
                1,
                None,
                index
                    .exact_residual_sidecar
                    .get()
                    .expect("admitted SIMD sidecar"),
            )
            .expect("sidecar SIMD differential");
        assert!(
            outcome.census.lanes_pruned > 0,
            "the fixture must exercise a real sidecar prune as well as the near cutoff"
        );
        let actual = index
            .resolve_heap(outcome.heap)
            .expect("resolve sidecar SIMD result");
        assert_eq!(actual, expected);
        assert_eq!(
            actual[0].index,
            u32::try_from(EXACT_RESIDUAL_LANES).expect("residual lane count fits u32")
        );
    }

    proptest! {
        #[test]
        fn exact_residual_upper_bound_property_never_underestimates_across_shapes(
            dimension in 1_usize..129,
            row_bits in proptest::collection::vec(any::<u16>(), 1..65),
            query_bits in proptest::collection::vec(any::<u32>(), 128),
        ) {
            let vectors: Vec<Vec<f32>> = row_bits
                .iter()
                .enumerate()
                .map(|(row, &seed)| {
                    (0..dimension)
                        .map(|column| {
                            finite_f16_bits(
                                seed.rotate_left(((row + column) % 16) as u32)
                                    .wrapping_add((column * 211) as u16),
                            )
                        })
                        .collect()
                })
                .collect();
            let query: Vec<f32> = query_bits
                .iter()
                .take(dimension)
                .enumerate()
                .map(|(column, &bits)| {
                    bounded_finite_f32_bits(bits.rotate_right((column % 32) as u32))
                })
                .collect();
            let mut index = InMemoryVectorIndex::from_vectors(
                (0..row_bits.len()).map(|row| format!("property-{row}")).collect(),
                vectors,
                dimension,
            )
            .expect("finite f16 property source");
            bind_test_residual_source(&mut index);
            let sidecar = index
                .build_exact_residual_sidecar()
                .expect("bounded property sidecar");
            assert_residual_bounds_cover_exact_scores(&index, &sidecar, &query);
        }
    }

    #[test]
    fn exact_residual_sidecar_pruning_census_proves_honest_lane_elimination() {
        let dimension = 35;
        let count = 17;
        let mut vectors = vec![vec![0.0_f32; dimension]; count];
        for vector in vectors.iter_mut().take(EXACT_RESIDUAL_LANES) {
            vector[0] = 1.0;
        }
        for vector in vectors.iter_mut().skip(EXACT_RESIDUAL_LANES) {
            vector[0] = -1.0;
        }
        let mut index = InMemoryVectorIndex::from_vectors(
            (0..count).map(|row| format!("prune-{row}")).collect(),
            vectors,
            dimension,
        )
        .expect("finite pruning source");
        bind_test_residual_source(&mut index);
        let flat = index.clone();
        let sidecar = index
            .build_exact_residual_sidecar()
            .expect("build sidecar from admitted test source");
        assert!(
            index
                .admit_exact_residual_sidecar(sidecar)
                .expect("admit exact sidecar")
        );
        let query = {
            let mut query = vec![0.0_f32; dimension];
            query[0] = 1.0;
            query
        };
        let outcome = index
            .scan_exact_residual_sidecar(
                &query,
                1,
                None,
                index
                    .exact_residual_sidecar
                    .get()
                    .expect("admitted sidecar"),
            )
            .expect("exact residual scan");
        assert!(
            outcome.census.lanes_pruned > 0,
            "the honest cache prunes lanes"
        );
        assert!(
            outcome.census.exact_sidecar_scores < outcome.census.eligible_lanes,
            "pruned lanes must avoid exact f16 rescoring"
        );
        let actual = index
            .resolve_heap(outcome.heap)
            .expect("resolve pruned result");
        let expected = flat.search_top_k(&query, 1, None).expect("flat result");
        assert_eq!(actual, expected, "pruning preserves the exact result");
    }

    #[test]
    fn exact_residual_sidecar_adaptive_fallback_is_censused_and_exact() {
        let dimension = 35;
        let count = (EXACT_RESIDUAL_ADAPTIVE_PROBE_GROUPS + 2) * EXACT_RESIDUAL_LANES;
        let mut vectors = vec![vec![0.0_f32; dimension]; count];
        for vector in &mut vectors {
            vector[0] = 1.0;
        }
        let mut index = InMemoryVectorIndex::from_vectors(
            (0..count).map(|row| format!("adaptive-{row}")).collect(),
            vectors,
            dimension,
        )
        .expect("finite non-selective source");
        bind_test_residual_source(&mut index);
        let flat = index.clone();
        let sidecar = index.build_exact_residual_sidecar().expect("build sidecar");
        assert!(
            index
                .admit_exact_residual_sidecar(sidecar)
                .expect("admit sidecar")
        );
        let query = {
            let mut query = vec![0.0_f32; dimension];
            query[0] = 1.0;
            query
        };
        let outcome = index
            .scan_exact_residual_sidecar(
                &query,
                1,
                None,
                index
                    .exact_residual_sidecar
                    .get()
                    .expect("admitted sidecar"),
            )
            .expect("adaptive exact scan");
        assert_eq!(outcome.census.lanes_pruned, 0, "tied rows cannot prune");
        assert_eq!(outcome.census.adaptive_fallbacks, 1);
        assert!(outcome.census.flat_fallback_rows > 0);
        assert_eq!(
            outcome.census.groups_scanned,
            EXACT_RESIDUAL_ADAPTIVE_PROBE_GROUPS
        );
        let actual = index
            .resolve_heap(outcome.heap)
            .expect("resolve adaptive fallback");
        let expected = flat.search_top_k(&query, 1, None).expect("flat result");
        assert_eq!(actual, expected);
    }

    #[test]
    fn exact_residual_sidecar_caps_extreme_k_at_the_available_rows() {
        let dimension = 35;
        let count = 17;
        let mut index = InMemoryVectorIndex::from_vectors(
            (0..count).map(|row| format!("k-{row}")).collect(),
            (0..count)
                .map(|row| make_normalized_vec(dimension, row as f32 + 0.25))
                .collect(),
            dimension,
        )
        .expect("finite source");
        bind_test_residual_source(&mut index);
        let flat = index.clone();
        let sidecar = index.build_exact_residual_sidecar().expect("build sidecar");
        assert!(
            index
                .admit_exact_residual_sidecar(sidecar)
                .expect("admit sidecar")
        );
        let query = make_normalized_vec(dimension, 4.0);
        let outcome = index
            .scan_exact_residual_sidecar(
                &query,
                usize::MAX,
                None,
                index
                    .exact_residual_sidecar
                    .get()
                    .expect("admitted sidecar"),
            )
            .expect("extreme k uses a count-bounded heap");
        let actual = index
            .resolve_heap(outcome.heap)
            .expect("resolve sidecar heap");
        let expected = flat
            .search_top_k(&query, usize::MAX, None)
            .expect("count-bounded flat heap");
        assert_eq!(actual, expected);
        assert_eq!(actual.len(), count);
    }

    #[test]
    fn exact_residual_sidecar_nonselective_parallel_probe_is_global_at_100k() {
        const COUNT: usize = 100_000;
        let dimension = 3;
        let mut index = InMemoryVectorIndex::from_vectors(
            (0..COUNT)
                .map(|row| format!("nonselective-{row}"))
                .collect(),
            vec![vec![1.0, 0.0, 0.0]; COUNT],
            dimension,
        )
        .expect("create 100k nonselective source");
        bind_test_residual_source(&mut index);
        let flat = index.clone();
        let sidecar = index
            .build_exact_residual_sidecar()
            .expect("build nonselective sidecar");
        assert!(
            index
                .admit_exact_residual_sidecar(sidecar)
                .expect("admit nonselective sidecar")
        );
        let params = SearchParams {
            parallel_enabled: true,
            ..SearchParams::default()
        };
        let query = [1.0, 0.0, 0.0];
        let outcome = index
            .scan_exact_residual_sidecar_parallel(
                &query,
                1,
                None,
                index
                    .exact_residual_sidecar
                    .get()
                    .expect("admitted sidecar"),
                params.parallel_chunk_size,
            )
            .expect("run globally adaptive parallel scan");
        assert_eq!(
            outcome.census.groups_scanned, EXACT_RESIDUAL_ADAPTIVE_PROBE_GROUPS,
            "one query-level probe replaces per-chunk residual work"
        );
        assert_eq!(outcome.census.lanes_pruned, 0);
        assert_eq!(outcome.census.adaptive_fallbacks, 1);
        assert_eq!(outcome.census.flat_fallback_rows, COUNT);
        let actual = index
            .resolve_heap(outcome.heap)
            .expect("resolve globally adaptive result");
        let expected = flat
            .search_top_k_with_params(&query, 1, None, params)
            .expect("flat exact parallel result");
        assert_eq!(actual, expected);
    }

    #[test]
    fn exact_residual_sidecar_f32_max_queries_fail_closed_to_exact_sequential_and_parallel() {
        let dimension = 35;
        let count = 17;
        let mut index = InMemoryVectorIndex::from_vectors(
            (0..count).map(|row| format!("max-query-{row}")).collect(),
            vec![vec![f16::MAX.to_f32(); dimension]; count],
            dimension,
        )
        .expect("finite f16 source for max-query fallback");
        bind_test_residual_source(&mut index);
        let flat = index.clone();
        let sidecar = index.build_exact_residual_sidecar().expect("build sidecar");
        assert!(
            index
                .admit_exact_residual_sidecar(sidecar)
                .expect("admit sidecar")
        );
        let query = vec![f32::MAX; dimension];
        let sequential = index
            .scan_exact_residual_sidecar(
                &query,
                3,
                None,
                index
                    .exact_residual_sidecar
                    .get()
                    .expect("admitted sidecar"),
            )
            .expect("sequential max-query fallback");
        let parallel = index
            .scan_exact_residual_sidecar_parallel(
                &query,
                3,
                None,
                index
                    .exact_residual_sidecar
                    .get()
                    .expect("admitted sidecar"),
                8,
            )
            .expect("parallel max-query fallback");
        assert_eq!(sequential.census, ResidualPruningCensus::default());
        assert_eq!(parallel.census, ResidualPruningCensus::default());
        let sequential = index
            .resolve_heap(sequential.heap)
            .expect("resolve sequential fallback");
        let parallel = index
            .resolve_heap(parallel.heap)
            .expect("resolve parallel fallback");
        let expected_sequential = flat
            .search_top_k_with_params(
                &query,
                3,
                None,
                SearchParams {
                    parallel_enabled: false,
                    ..SearchParams::default()
                },
            )
            .expect("flat sequential max-query result");
        let expected_parallel = flat
            .search_top_k_with_params(
                &query,
                3,
                None,
                SearchParams {
                    parallel_enabled: true,
                    parallel_threshold: 1,
                    parallel_chunk_size: 8,
                },
            )
            .expect("flat parallel max-query result");
        assert_eq!(sequential, expected_sequential);
        assert_eq!(parallel, expected_parallel);
        assert_eq!(sequential, parallel);
    }

    #[test]
    fn exact_residual_sidecar_parallel_uses_the_default_10k_to_100k_target_scales() {
        let dimension = 3;
        let params = SearchParams {
            parallel_enabled: true,
            ..SearchParams::default()
        };
        assert_eq!(params.parallel_threshold, crate::search::PARALLEL_THRESHOLD);
        let query = [1.0, 0.0, 0.0];
        for count in [crate::search::PARALLEL_THRESHOLD + 1, 100_000] {
            let mut index = InMemoryVectorIndex::from_vectors(
                (0..count).map(|row| format!("parallel-{row}")).collect(),
                (0..count)
                    .map(|row| {
                        vec![
                            if row % EXACT_RESIDUAL_LANES == 0 {
                                1.0
                            } else {
                                -1.0
                            },
                            0.0,
                            0.0,
                        ]
                    })
                    .collect(),
                dimension,
            )
            .expect("target-scale finite source");
            bind_test_residual_source(&mut index);
            let flat = index.clone();
            let sidecar = index
                .build_exact_residual_sidecar()
                .expect("build target-scale sidecar");
            assert!(
                index
                    .admit_exact_residual_sidecar(sidecar)
                    .expect("admit sidecar")
            );
            let census_outcome = index
                .scan_exact_residual_sidecar_parallel(
                    &query,
                    10,
                    None,
                    index
                        .exact_residual_sidecar
                        .get()
                        .expect("admitted sidecar"),
                    params.parallel_chunk_size,
                )
                .expect("parallel sidecar route");
            assert!(census_outcome.census.groups_scanned > 0);
            assert!(
                census_outcome.census.lanes_pruned > 0,
                "{count}-row target must exercise per-lane pruning"
            );
            assert!(
                census_outcome.census.exact_sidecar_scores < census_outcome.census.eligible_lanes,
                "{count}-row target must avoid exact scores for proven lanes"
            );
            let expected = flat
                .search_top_k_with_params(&query, 10, None, params)
                .expect("flat parallel result");
            assert_eq!(
                index
                    .resolve_heap(census_outcome.heap)
                    .expect("resolve censused parallel result"),
                expected,
                "{count}-row censused route preserves exact ordering"
            );
            let actual = index
                .search_top_k_with_params(&query, 10, None, params)
                .expect("sidecar parallel result");
            assert_eq!(
                actual, expected,
                "{count}-row sidecar path preserves exact ordering"
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn exact_residual_sidecar_public_io_rejects_snapshot_changes_symlinks_and_overwrite_races() {
        use std::os::unix::fs::symlink;

        let dimension = 35;
        let mut index = InMemoryVectorIndex::from_vectors(
            (0..17).map(|row| format!("io-{row}")).collect(),
            (0..17)
                .map(|row| make_normalized_vec(dimension, row as f32 * 0.75))
                .collect(),
            dimension,
        )
        .expect("finite sidecar source");
        bind_test_residual_source(&mut index);
        let dir = owned_temp_dir("exact_residual_public_io");
        let sidecar_path = dir.join("sidecar.fsrs");
        let occupied_path = dir.join("occupied.fsrs");
        let raced_path = dir.join("raced.fsrs");
        let symlink_path = dir.join("symlink.fsrs");
        let target_path = dir.join("target.fsrs");

        index
            .write_exact_residual_sidecar(&sidecar_path)
            .expect("public write publishes an owned sidecar");
        let reader = index.clone();
        assert!(
            reader
                .try_open_exact_residual_sidecar(&sidecar_path)
                .expect("public no-follow open")
        );
        let expected = index
            .build_exact_residual_sidecar()
            .expect("derive test sidecar without touching the published file");
        let encoded = expected.encode().expect("encode test sidecar in memory");
        let mut truncated_after_stat = std::io::Cursor::new(encoded[..encoded.len() - 1].to_vec());
        assert!(
            !exact_residual_sidecar_stream_matches_reader(&mut truncated_after_stat, &expected)
                .expect("bounded reader comparison"),
            "the production reader seam rejects a stream truncated after fstat"
        );
        let mut grown_after_stat = encoded.clone();
        grown_after_stat.push(0x5a);
        let mut grown_after_stat = std::io::Cursor::new(grown_after_stat);
        assert!(
            !exact_residual_sidecar_stream_matches_reader(&mut grown_after_stat, &expected)
                .expect("bounded reader comparison"),
            "the production one-byte probe rejects a stream that grows after fstat"
        );

        write_new_owned_file(&occupied_path, b"incumbent destination");
        for _ in 0..3 {
            assert!(
                index.write_exact_residual_sidecar(&occupied_path).is_err(),
                "linkat publication refuses an occupied destination"
            );
        }
        assert_eq!(
            std::fs::read(&occupied_path).expect("read incumbent"),
            b"incumbent destination"
        );
        assert_eq!(
            std::fs::read_dir(&dir)
                .expect("inspect owned public-I/O directory")
                .count(),
            2,
            "failed no-replace publications must leave no visible temporary artifact"
        );

        let start = std::sync::Arc::new(std::sync::Barrier::new(3));
        let left_start = std::sync::Arc::clone(&start);
        let left_index = index.clone();
        let left_path = raced_path.clone();
        let left = std::thread::spawn(move || {
            left_start.wait();
            left_index.write_exact_residual_sidecar(&left_path).is_ok()
        });
        let right_start = std::sync::Arc::clone(&start);
        let right_index = index.clone();
        let right_path = raced_path.clone();
        let right = std::thread::spawn(move || {
            right_start.wait();
            right_index
                .write_exact_residual_sidecar(&right_path)
                .is_ok()
        });
        start.wait();
        let published = [
            left.join().expect("left writer did not panic"),
            right.join().expect("right writer did not panic"),
        ]
        .into_iter()
        .filter(|published| *published)
        .count();
        assert_eq!(
            published, 1,
            "concurrent public writers publish exactly one immutable destination"
        );
        let race_reader = index.clone();
        assert!(
            race_reader
                .try_open_exact_residual_sidecar(&raced_path)
                .expect("winner remains descriptor-admissible")
        );
        assert!(
            std::fs::read_dir(&dir)
                .expect("read test parent")
                .flatten()
                .count()
                == 3,
            "anonymous O_TMPFILE failures create no visible temporary paths"
        );

        write_new_owned_file(&target_path, b"symlink target");
        symlink(&target_path, &symlink_path).expect("create final-component symlink");
        assert!(
            !index
                .try_open_exact_residual_sidecar(&symlink_path)
                .expect("no-follow symlink open is an optional miss")
        );
        assert!(
            index.write_exact_residual_sidecar(&symlink_path).is_err(),
            "atomic no-replace publication refuses a symlink destination"
        );
        assert_eq!(
            std::fs::read(&target_path).expect("read symlink target"),
            b"symlink target"
        );
    }
}
