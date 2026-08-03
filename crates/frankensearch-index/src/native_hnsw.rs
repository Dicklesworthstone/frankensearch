//! Native HNSW graph index (bd-kcek).
//!
//! frankensearch's own approximate-nearest-neighbour engine, owned in-tree
//! rather than adapted from a third-party crate. Two things motivate it:
//! we cannot publish to crates.io while the ANN engine is a git-only
//! dependency, and we should never be blocked on an external maintainer's
//! release timing for an algorithm this size.
//!
//! # Correctness is inherited, not re-derived
//!
//! bd-u3wt reverse-engineered the layer invariants by finding four coupled
//! defects in the previous engine. Those findings are the specification
//! this module implements, and each is enforced at the point it can be
//! violated:
//!
//! 1. **Descent must not attach above the sampled level.** The greedy
//!    descent that locates an entry point runs strictly above the new
//!    point's own level and never links there.
//! 2. **Connection must start at `min(new_level, previous_max_level)`.**
//!    Starting higher leaves a new point's upper layers empty; starting
//!    from the new level unconditionally links at layers no other point
//!    occupies yet.
//! 3. **Reverse edges belong to the layer being processed.** A reciprocal
//!    edge discovered while linking at layer `l` is stored at layer `l` —
//!    not at the new point's top layer. Storing it at the top layer both
//!    omits required lower-layer edges and can install an edge in a layer
//!    the endpoints do not occupy, which is what produced asymmetric
//!    underfill.
//! 4. **Layer participation is `level >= layer`, not `level == layer`.** A
//!    point participates in every layer at or below its sampled level, so
//!    an empty exact-level bucket does not mean the logical layer is empty.
//!    (Exposed by a `high, high, zero` sampled-level sequence.)
//!
//! # Safety
//!
//! This module contains no `unsafe`, and needs none: the graph stores ids
//! and distances, never reinterpreted bytes. So the class of defect the
//! previous engine's reload path carried — a misaligned pointer cast whose
//! element count came from a file header rather than the bytes actually
//! read — cannot occur here.
//!
//! That is a property of this module, not a blanket guarantee. The
//! workspace lints `unsafe_code` at `deny` rather than `forbid`
//! specifically so crates can opt in, and this crate does exactly that for
//! memory-mapping (`mapped_file.rs` and the `MmapMut` path in `lib.rs`).
//! Any future persistence layer for this graph inherits that risk the
//! moment it maps a file, so it should parse into owned values rather than
//! casting mapped bytes.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};
use std::fmt;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use crate::ValidatedFsviBytes;
use frankensearch_core::{
    error::{SearchError, SearchResult},
    generation::ArtifactGenerationIdentityV1,
    sha256_checksum,
};
use serde::{Deserialize, Serialize};

const NATIVE_HNSW_MAGIC: [u8; 8] = *b"FSHNSW\0\0";
/// Current owned native-HNSW graph format.
pub const NATIVE_HNSW_FORMAT_VERSION: u32 = 1;
const NATIVE_HNSW_HEADER_LEN: usize = 96;
const NATIVE_HNSW_HEADER_LEN_U64: u64 = 96;
const NATIVE_HNSW_HEADER_CRC_OFFSET: usize = NATIVE_HNSW_HEADER_LEN - 4;
const NATIVE_HNSW_NO_ENTRY: u64 = u64::MAX;
const NATIVE_HNSW_RECEIPT_MAGIC: [u8; 8] = *b"FSHNRC\0\0";
/// Current schema for native-HNSW-to-FSVI generation receipts.
pub const NATIVE_HNSW_GENERATION_RECEIPT_SCHEMA_V2: u16 = 2;
/// Canonical sidecar suffix appended to the complete `.fshnsw` basename.
pub const NATIVE_HNSW_GENERATION_RECEIPT_SUFFIX: &str = ".receipt";
const NATIVE_HNSW_MAX_BASENAME_BYTES: usize = 255 - NATIVE_HNSW_GENERATION_RECEIPT_SUFFIX.len();
const SHA256_HEX_LEN: usize = 64;
const SHA256_BYTES: usize = 32;

/// A violation of the graph's structural invariants.
///
/// Structured rather than stringly so an attestation gate can branch on the
/// defect class, and so the regression fixtures below assert on the variant
/// instead of matching message text. The graph itself indexes rows of a
/// caller-owned store and owns no file, so this is deliberately not a
/// [`SearchError`]: the wiring layer, which knows the sidecar path, maps a
/// defect into `SearchError::IndexCorrupted`.
#[derive(Debug, Clone, PartialEq, Eq)]
enum GraphDefect {
    /// The graph and its caller-owned vector store describe different row
    /// counts, so graph ids cannot be interpreted against that store.
    StoreCardinalityMismatch {
        /// Rows indexed by the graph.
        graph_points: usize,
        /// Rows exposed by the vector store.
        store_rows: usize,
    },
    /// A graph with no points nonetheless names an entry point.
    EntryPointInEmptyGraph {
        /// The spurious entry point id.
        entry: u32,
    },
    /// A graph with no points nonetheless claims an occupied upper layer.
    MaxLevelInEmptyGraph {
        /// The spurious maximum level.
        max_level: usize,
    },
    /// A graph with points has no entry point, so nothing is searchable.
    MissingEntryPoint,
    /// The entry point is not a known row.
    EntryPointUnknown {
        /// The dangling entry point id.
        entry: u32,
    },
    /// The entry point does not sit at the graph's maximum level, so upper
    /// layers are unreachable.
    EntryPointBelowMaxLevel {
        /// The entry point id.
        entry: u32,
        /// The level it actually occupies.
        level: usize,
        /// The graph's maximum level.
        max_level: usize,
    },
    /// A point claims a level above the graph maximum.
    LevelAboveMax {
        /// The offending point.
        id: u32,
        /// Its claimed level.
        level: usize,
        /// The graph's maximum level.
        max_level: usize,
    },
    /// A neighbourhood exceeds its layer's degree budget.
    DegreeExceeded {
        /// The offending point.
        id: u32,
        /// The layer.
        layer: usize,
        /// Neighbours actually held.
        held: usize,
        /// The budget for this layer.
        budget: usize,
    },
    /// A neighbourhood lists the same neighbour twice.
    DuplicateNeighbour {
        /// The offending point.
        id: u32,
        /// The layer.
        layer: usize,
    },
    /// A point lists itself as its own neighbour.
    SelfEdge {
        /// The offending point.
        id: u32,
        /// The layer.
        layer: usize,
    },
    /// An edge names a neighbour that is not a known row.
    NeighbourUnknown {
        /// The offending point.
        id: u32,
        /// The layer.
        layer: usize,
        /// The dangling neighbour id.
        neighbour: u32,
    },
    /// An edge exists in a layer one endpoint does not occupy.
    ///
    /// This is the bd-u3wt asymmetric-underfill defect: reverse edges
    /// stored at the new point's top layer rather than the layer being
    /// processed installed edges in layers the target never reached.
    EdgeAboveNeighbourLevel {
        /// The point holding the edge.
        id: u32,
        /// The layer the edge sits in.
        layer: usize,
        /// The neighbour that cannot participate there.
        neighbour: u32,
        /// The level the neighbour actually reaches.
        neighbour_level: usize,
    },
    /// One endpoint names an edge whose peer does not name the reverse edge
    /// at the same logical layer.
    MissingReciprocalEdge {
        /// The point holding the one-way edge.
        id: u32,
        /// The layer containing the one-way edge.
        layer: usize,
        /// The peer missing the reverse edge.
        neighbour: u32,
    },
    /// A point's persisted layer count disagrees with the level sampled from
    /// the graph's seed and parameters.
    SampledLevelMismatch {
        /// The point whose level is not reproducible.
        id: u32,
        /// Level deterministically sampled for this point.
        expected: usize,
        /// Level stored in the graph.
        actual: usize,
    },
    /// Some points cannot be reached from the entry point at layer 0, so no
    /// query can find them however wide the beam.
    Unreachable {
        /// How many points were reached.
        reached: usize,
        /// How many points exist.
        total: usize,
        /// The lowest-numbered unreachable point.
        first_unreachable: u32,
    },
    /// A full-width search over every physical row did not produce the
    /// live-result cardinality promised by the admitted owner's census.
    ///
    /// This can only happen if graph reachability or the owner census has
    /// become inconsistent after admission. Returning a short result would
    /// make that corruption look like ordinary low recall, so search fails
    /// with this typed defect instead.
    LiveResultUnderfill {
        /// Result count requested by the caller.
        requested_k: usize,
        /// Live rows the owner census says must be returned.
        expected_live_hits: usize,
        /// Live rows actually found after searching every physical row.
        returned_live_hits: usize,
        /// Physical rows used as the hard search-width bound.
        physical_rows: usize,
    },
}

impl fmt::Display for GraphDefect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StoreCardinalityMismatch {
                graph_points,
                store_rows,
            } => write!(
                f,
                "graph indexes {graph_points} points but the vector store exposes {store_rows} rows"
            ),
            Self::EntryPointInEmptyGraph { entry } => {
                write!(f, "empty graph names entry point {entry}")
            }
            Self::MaxLevelInEmptyGraph { max_level } => {
                write!(f, "empty graph claims maximum level {max_level}")
            }
            Self::MissingEntryPoint => write!(f, "non-empty graph has no entry point"),
            Self::EntryPointUnknown { entry } => {
                write!(f, "entry point {entry} is not a known row")
            }
            Self::EntryPointBelowMaxLevel {
                entry,
                level,
                max_level,
            } => write!(
                f,
                "entry point {entry} is at level {level} but the graph maximum is {max_level}"
            ),
            Self::LevelAboveMax {
                id,
                level,
                max_level,
            } => write!(
                f,
                "point {id} claims level {level} above the graph maximum {max_level}"
            ),
            Self::DegreeExceeded {
                id,
                layer,
                held,
                budget,
            } => write!(
                f,
                "point {id} holds {held} neighbours at layer {layer}, above the {budget} budget"
            ),
            Self::DuplicateNeighbour { id, layer } => {
                write!(f, "point {id} has duplicate neighbours at layer {layer}")
            }
            Self::SelfEdge { id, layer } => {
                write!(f, "point {id} is its own neighbour at layer {layer}")
            }
            Self::NeighbourUnknown {
                id,
                layer,
                neighbour,
            } => write!(
                f,
                "point {id} names neighbour {neighbour} at layer {layer}, which is not a known row"
            ),
            Self::EdgeAboveNeighbourLevel {
                id,
                layer,
                neighbour,
                neighbour_level,
            } => write!(
                f,
                "point {id} has an edge to {neighbour} at layer {layer}, but {neighbour} only \
                 reaches level {neighbour_level}"
            ),
            Self::MissingReciprocalEdge {
                id,
                layer,
                neighbour,
            } => write!(
                f,
                "point {id} names neighbour {neighbour} at layer {layer}, but the reverse edge is \
                 missing"
            ),
            Self::SampledLevelMismatch {
                id,
                expected,
                actual,
            } => write!(
                f,
                "point {id} stores level {actual}, but seed and parameters sample level {expected}"
            ),
            Self::Unreachable {
                reached,
                total,
                first_unreachable,
            } => write!(
                f,
                "only {reached} of {total} points are reachable at layer 0 (first unreachable: \
                 {first_unreachable})"
            ),
            Self::LiveResultUnderfill {
                requested_k,
                expected_live_hits,
                returned_live_hits,
                physical_rows,
            } => write!(
                f,
                "full-width native HNSW search over {physical_rows} physical rows returned \
                 {returned_live_hits} live hits, but request k={requested_k} and the admitted \
                 owner census require {expected_live_hits}"
            ),
        }
    }
}

impl std::error::Error for GraphDefect {}

/// Double a search window without ever crossing its physical-row bound.
///
/// The `current + 1` floor guarantees progress even if a future caller starts
/// at zero; the early return makes that addition safe at `usize::MAX`.
fn widen_search_width(current: usize, physical_rows: usize) -> usize {
    if current >= physical_rows {
        return physical_rows;
    }
    current
        .saturating_mul(2)
        .max(current + 1)
        .min(physical_rows)
}

/// Maximum number of graph layers.
///
/// Level assignment is geometric, so the probability of sampling a level at
/// all near this bound is negligible for any corpus that fits in memory;
/// the cap exists so layer tables are fixed-size and a corrupt or hostile
/// level can never drive an unbounded allocation.
pub const MAX_LEVEL: usize = 16;

/// Tuning parameters for graph construction and search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HnswParams {
    /// Neighbours kept per point per layer above layer 0.
    pub m: usize,
    /// Neighbours kept per point at layer 0.
    ///
    /// Layer 0 holds every point and carries the final hop of every query,
    /// so it is conventionally given twice the degree of upper layers.
    pub m0: usize,
    /// Beam width during construction.
    pub ef_construction: usize,
    /// Default beam width during search. Callers may override per query.
    pub ef_search: usize,
}

/// Integrity metadata read from or written with one owned graph artifact.
///
/// The CRC fields detect accidental corruption inside the graph format. They
/// are not a cryptographic generation identity; the production wiring layer
/// binds the complete artifact to its FSVI generation with a SHA-256 receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NativeHnswFileMetadata {
    format_version: u32,
    byte_len: u64,
    point_count: u64,
    payload_crc32: u32,
    header_crc32: u32,
}

impl NativeHnswFileMetadata {
    /// Binary format version.
    #[must_use]
    const fn format_version(self) -> u32 {
        self.format_version
    }

    /// Complete file length.
    #[must_use]
    const fn byte_len(self) -> u64 {
        self.byte_len
    }

    /// Number of graph points encoded in the artifact.
    #[must_use]
    const fn point_count(self) -> u64 {
        self.point_count
    }

    /// CRC-32 of the canonical adjacency payload.
    #[must_use]
    const fn payload_crc32(self) -> u32 {
        self.payload_crc32
    }

    /// CRC-32 of the fixed header before its checksum field.
    #[must_use]
    const fn header_crc32(self) -> u32 {
        self.header_crc32
    }
}

/// Canonical persisted identity of native-HNSW construction parameters.
///
/// The receipt uses full-width integers so its encoding is independent of the
/// reader's pointer width. Conversion back to [`HnswParams`] remains checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeHnswParamsIdentityV1 {
    /// Neighbours retained per upper-layer point.
    pub m: u64,
    /// Neighbours retained per layer-zero point.
    pub m0: u64,
    /// Construction beam width.
    pub ef_construction: u64,
    /// Default search beam width.
    pub ef_search: u64,
}

impl NativeHnswParamsIdentityV1 {
    fn from_params(params: HnswParams) -> SearchResult<Self> {
        Ok(Self {
            m: usize_to_u64(params.m, "receipt.params.m")?,
            m0: usize_to_u64(params.m0, "receipt.params.m0")?,
            ef_construction: usize_to_u64(
                params.ef_construction,
                "receipt.params.ef_construction",
            )?,
            ef_search: usize_to_u64(params.ef_search, "receipt.params.ef_search")?,
        })
    }

    fn to_params(self) -> SearchResult<HnswParams> {
        let params = HnswParams {
            m: receipt_usize(self.m, "params.m")?,
            m0: receipt_usize(self.m0, "params.m0")?,
            ef_construction: receipt_usize(self.ef_construction, "params.ef_construction")?,
            ef_search: receipt_usize(self.ef_search, "params.ef_search")?,
        };
        params.validate()?;
        Ok(params)
    }
}

/// Cryptographic binding between one owned FSHNSW artifact and one FSVI v2
/// generation.
///
/// The public serde representation rejects unknown fields. The on-disk
/// sidecar is a separate canonical binary encoding with a SHA-256 body seal;
/// readers reject alternate encodings and trailing bytes rather than silently
/// accepting a second representation of the same receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeHnswGenerationReceiptV2 {
    /// [`NATIVE_HNSW_GENERATION_RECEIPT_SCHEMA_V2`].
    pub schema_version: u16,
    /// Exact immutable FSVI artifact generation.
    pub artifact_generation: ArtifactGenerationIdentityV1,
    /// SHA-256 of the generation's canonical identity bytes.
    pub artifact_generation_fingerprint: String,
    /// SHA-256 of the complete frozen embedding identity bundle.
    pub embedding_identity_fingerprint: String,
    /// SHA-256 of the mathematical embedding-space identity.
    pub embedding_space_fingerprint: String,
    /// SHA-256 of the embedding-producer attestation.
    pub embedding_producer_fingerprint: String,
    /// SHA-256 of the outer embedding-input contract.
    pub embedding_input_fingerprint: String,
    /// SHA-256 of the physical FSVI v2 storage identity.
    pub vector_storage_fingerprint: String,
    /// SHA-256 of the exact persisted FSVI vector slab and shape.
    pub vector_content_digest: String,
    /// SHA-256 of the ordered searchable FSVI document identifiers.
    pub ordered_live_docset_digest: String,
    /// SHA-256 of every byte in the admitted immutable FSVI image.
    pub fsvi_whole_image_sha256: String,
    /// Exact number of physical FSVI rows, including tombstones.
    pub fsvi_physical_row_count: u64,
    /// Canonical UTF-8 basename of the adjacent FSHNSW artifact.
    pub graph_basename: String,
    /// Complete FSHNSW file length.
    pub graph_byte_len: u64,
    /// SHA-256 of the complete FSHNSW file bytes.
    pub graph_sha256: String,
    /// Owned FSHNSW binary format version.
    pub native_format_version: u32,
    /// Construction and default-search parameters.
    pub params: NativeHnswParamsIdentityV1,
    /// Deterministic level-sampling seed.
    pub seed: u64,
    /// Number of graph rows, which must equal the vector-store cardinality.
    pub point_count: u64,
    /// Search entry point, absent only for an empty graph.
    pub entry_point: Option<u32>,
    /// Highest occupied graph layer.
    pub max_level: u64,
    /// CRC-32 of the canonical adjacency payload.
    pub payload_crc32: u32,
    /// CRC-32 of the fixed FSHNSW header.
    pub header_crc32: u32,
    /// SHA-256 of the semantic topology, independent of file checksums.
    pub topology_sha256: String,
    /// SHA-256 of every preceding field in canonical binary order.
    pub receipt_sha256: String,
}

/// Internal exact-FSVI trust material for an owner-bound native HNSW sidecar.
///
/// This type is deliberately private. Public callers can create, persist,
/// reopen, and search a receipted graph only through [`ValidatedNativeHnsw`],
/// whose retained owner and private fields keep the graph paired with the
/// admitted bytes that supplied this trust material.
#[derive(Debug, Clone, PartialEq, Eq)]
struct NativeHnswGenerationBindingV2 {
    artifact_generation: ArtifactGenerationIdentityV1,
    artifact_generation_fingerprint: String,
    embedding_identity_fingerprint: String,
    embedding_space_fingerprint: String,
    embedding_producer_fingerprint: String,
    embedding_input_fingerprint: String,
    vector_storage_fingerprint: String,
    vector_content_digest: String,
    ordered_live_docset_digest: String,
    fsvi_whole_image_sha256: String,
    fsvi_physical_row_count: u64,
}

impl NativeHnswGenerationBindingV2 {
    fn from_validated_fsvi(owner: &ValidatedFsviBytes) -> SearchResult<Self> {
        let identity = owner.identity_v2();
        let fsvi_physical_row_count =
            usize_to_u64(owner.record_count(), "binding.fsvi_physical_row_count")?;

        let binding = Self {
            artifact_generation: identity.generation,
            artifact_generation_fingerprint: encode_lower_hex(identity.generation_fingerprint),
            embedding_identity_fingerprint: encode_lower_hex(identity.identity_bundle_fingerprint),
            embedding_space_fingerprint: encode_lower_hex(identity.space_fingerprint),
            embedding_producer_fingerprint: encode_lower_hex(identity.producer_fingerprint),
            embedding_input_fingerprint: encode_lower_hex(identity.input_fingerprint),
            vector_storage_fingerprint: encode_lower_hex(identity.storage_fingerprint),
            vector_content_digest: encode_lower_hex(identity.vector_content_digest),
            ordered_live_docset_digest: encode_lower_hex(identity.ordered_live_docset_digest),
            fsvi_whole_image_sha256: encode_lower_hex(owner.witness().whole_image_sha256),
            fsvi_physical_row_count,
        };
        binding.validate_fingerprints()?;
        Ok(binding)
    }

    /// Save `graph` atomically, then publish its canonical adjacent receipt.
    ///
    /// Both destination paths are preflighted before the graph is touched.
    /// A crash between the two atomic renames can leave a new graph with a
    /// missing or stale receipt, but that state fails closed on every load.
    ///
    /// # Errors
    ///
    /// Returns the same graph errors as [`NativeHnsw::save`], plus
    /// [`SearchError::InvalidConfig`] for a noncanonical graph path and
    /// [`SearchError::Io`] for receipt persistence failures.
    fn save_bound_graph(
        &self,
        graph: &NativeHnsw,
        graph_path: &Path,
    ) -> SearchResult<NativeHnswGenerationReceiptV2> {
        let graph_point_count = usize_to_u64(graph.len(), "graph.point_count")?;
        if graph_point_count != self.fsvi_physical_row_count {
            return Err(native_hnsw_receipt_config_error(
                "graph.point_count",
                &graph_point_count.to_string(),
                "must equal the admitted FSVI physical row count",
            ));
        }
        let receipt_path = native_hnsw_generation_receipt_path(graph_path)?;
        reject_symlink_ancestors(graph_path)?;
        reject_non_regular_destination(graph_path)?;
        reject_non_regular_receipt_destination(&receipt_path)?;

        let metadata = graph.save(graph_path)?;
        let graph_bytes = read_regular_file_bytes(graph_path, "native HNSW artifact")?;
        if u64::try_from(graph_bytes.len()).ok() != Some(metadata.byte_len()) {
            return Err(native_hnsw_corrupted(
                graph_path,
                "native HNSW file length changed after atomic publication",
            ));
        }
        let mut receipt = NativeHnswGenerationReceiptV2 {
            schema_version: NATIVE_HNSW_GENERATION_RECEIPT_SCHEMA_V2,
            artifact_generation: self.artifact_generation,
            artifact_generation_fingerprint: self.artifact_generation_fingerprint.clone(),
            embedding_identity_fingerprint: self.embedding_identity_fingerprint.clone(),
            embedding_space_fingerprint: self.embedding_space_fingerprint.clone(),
            embedding_producer_fingerprint: self.embedding_producer_fingerprint.clone(),
            embedding_input_fingerprint: self.embedding_input_fingerprint.clone(),
            vector_storage_fingerprint: self.vector_storage_fingerprint.clone(),
            vector_content_digest: self.vector_content_digest.clone(),
            ordered_live_docset_digest: self.ordered_live_docset_digest.clone(),
            fsvi_whole_image_sha256: self.fsvi_whole_image_sha256.clone(),
            fsvi_physical_row_count: self.fsvi_physical_row_count,
            graph_basename: canonical_graph_basename(graph_path)?,
            graph_byte_len: metadata.byte_len(),
            graph_sha256: sha256_hex(&graph_bytes),
            native_format_version: metadata.format_version(),
            params: NativeHnswParamsIdentityV1::from_params(graph.params())?,
            seed: graph.seed(),
            point_count: metadata.point_count(),
            entry_point: graph.entry_point(),
            max_level: usize_to_u64(graph.max_level(), "receipt.max_level")?,
            payload_crc32: metadata.payload_crc32(),
            header_crc32: metadata.header_crc32(),
            topology_sha256: graph.topology_sha256()?,
            receipt_sha256: String::new(),
        };
        receipt.seal()?;
        let encoded = receipt.to_bytes()?;
        persist_native_hnsw_receipt(&receipt_path, &encoded)?;
        Ok(receipt)
    }

    /// Load a graph only after its adjacent receipt proves the exact expected
    /// FSVI generation, embedding identity, file bytes, and topology.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::IndexNotFound`] when either artifact is missing,
    /// [`SearchError::IndexCorrupted`] for malformed, stale, replaced, or
    /// mismatched receipt/graph material, and [`SearchError::Io`] for other
    /// filesystem failures.
    fn load_bound_graph(
        &self,
        graph_path: &Path,
        owner: &ValidatedFsviBytes,
    ) -> SearchResult<(NativeHnsw, NativeHnswGenerationReceiptV2)> {
        self.load_bound_graph_with_after_first_observation(graph_path, owner, || Ok(()))
    }

    fn load_bound_graph_with_after_first_observation(
        &self,
        graph_path: &Path,
        owner: &ValidatedFsviBytes,
        after_first_observation: impl FnOnce() -> SearchResult<()>,
    ) -> SearchResult<(NativeHnsw, NativeHnswGenerationReceiptV2)> {
        let observed_binding = Self::from_validated_fsvi(owner)?;
        if observed_binding != *self {
            return Err(native_hnsw_receipt_config_error(
                "owner",
                "redacted-witness-mismatch",
                "must be the exact admitted FSVI owner that created this binding",
            ));
        }
        let receipt_path = native_hnsw_generation_receipt_path(graph_path)?;
        reject_symlink_ancestors(graph_path)?;
        let receipt_bytes =
            read_regular_file_bytes(&receipt_path, "native HNSW generation receipt")?;
        let receipt = NativeHnswGenerationReceiptV2::from_bytes(&receipt_bytes, &receipt_path)?;
        self.validate_receipt_identity(&receipt, graph_path, &receipt_path)?;

        let graph_bytes_before = read_regular_file_bytes(graph_path, "native HNSW artifact")?;
        Self::validate_graph_bytes(&receipt, graph_path, &graph_bytes_before)?;
        after_first_observation()?;
        let (graph, metadata) = NativeHnsw::load_with_metadata(graph_path, owner)?;
        Self::validate_graph_identity(&receipt, graph_path, &graph, metadata)?;

        // A second exact byte observation closes the replacement window
        // between the cryptographic check and the owned graph parser.
        let graph_bytes_after = read_regular_file_bytes(graph_path, "native HNSW artifact")?;
        // ubs:ignore — public artifact bytes are integrity material, not a secret.
        if graph_bytes_after != graph_bytes_before {
            return Err(native_hnsw_corrupted(
                graph_path,
                "native HNSW artifact changed during receipt verification",
            ));
        }
        Ok((graph, receipt))
    }

    fn validate_fingerprints(&self) -> SearchResult<()> {
        for (field, fingerprint) in [
            (
                "artifact_generation_fingerprint",
                &self.artifact_generation_fingerprint,
            ),
            (
                "embedding_identity_fingerprint",
                &self.embedding_identity_fingerprint,
            ),
            (
                "embedding_space_fingerprint",
                &self.embedding_space_fingerprint,
            ),
            (
                "embedding_producer_fingerprint",
                &self.embedding_producer_fingerprint,
            ),
            (
                "embedding_input_fingerprint",
                &self.embedding_input_fingerprint,
            ),
            (
                "vector_storage_fingerprint",
                &self.vector_storage_fingerprint,
            ),
            ("vector_content_digest", &self.vector_content_digest),
            (
                "ordered_live_docset_digest",
                &self.ordered_live_docset_digest,
            ),
            ("fsvi_whole_image_sha256", &self.fsvi_whole_image_sha256),
        ] {
            validate_sha256_hex(field, fingerprint)?;
        }
        // ubs:ignore — generation fingerprints are public artifact identities.
        if self.artifact_generation_fingerprint != self.artifact_generation.fingerprint() {
            return Err(native_hnsw_receipt_config_error(
                "artifact_generation_fingerprint",
                "redacted-digest-mismatch",
                "must match the admitted FSVI generation",
            ));
        }
        Ok(())
    }

    fn validate_receipt_identity(
        &self,
        receipt: &NativeHnswGenerationReceiptV2,
        graph_path: &Path,
        receipt_path: &Path,
    ) -> SearchResult<()> {
        let mismatch = |field: &str| {
            native_hnsw_receipt_corrupted(
                receipt_path,
                format!("native HNSW receipt {field} mismatch"),
            )
        };
        // ubs:ignore — generation identity is public artifact metadata, not a credential.
        if receipt.artifact_generation != self.artifact_generation {
            return Err(mismatch("artifact generation"));
        }
        // ubs:ignore — this public SHA-256 is an integrity fingerprint, not a MAC or secret.
        if receipt.artifact_generation_fingerprint != self.artifact_generation_fingerprint {
            return Err(mismatch("artifact generation fingerprint"));
        }
        for (field, actual, expected) in [
            (
                "embedding identity fingerprint",
                &receipt.embedding_identity_fingerprint,
                &self.embedding_identity_fingerprint,
            ),
            (
                "embedding space fingerprint",
                &receipt.embedding_space_fingerprint,
                &self.embedding_space_fingerprint,
            ),
            (
                "embedding producer fingerprint",
                &receipt.embedding_producer_fingerprint,
                &self.embedding_producer_fingerprint,
            ),
            (
                "embedding input fingerprint",
                &receipt.embedding_input_fingerprint,
                &self.embedding_input_fingerprint,
            ),
            (
                "vector storage fingerprint",
                &receipt.vector_storage_fingerprint,
                &self.vector_storage_fingerprint,
            ),
            (
                "vector content digest",
                &receipt.vector_content_digest,
                &self.vector_content_digest,
            ),
            (
                "ordered live-docset digest",
                &receipt.ordered_live_docset_digest,
                &self.ordered_live_docset_digest,
            ),
            (
                "FSVI whole-image SHA-256",
                &receipt.fsvi_whole_image_sha256,
                &self.fsvi_whole_image_sha256,
            ),
        ] {
            // ubs:ignore — frozen identity fingerprints are public compatibility metadata.
            if actual != expected {
                return Err(mismatch(field));
            }
        }
        if receipt.fsvi_physical_row_count != self.fsvi_physical_row_count {
            return Err(mismatch("FSVI physical row count"));
        }
        if receipt.graph_basename != canonical_graph_basename(graph_path)? {
            return Err(mismatch("graph basename"));
        }
        Ok(())
    }

    fn validate_graph_bytes(
        receipt: &NativeHnswGenerationReceiptV2,
        graph_path: &Path,
        graph_bytes: &[u8],
    ) -> SearchResult<()> {
        let actual_len = u64::try_from(graph_bytes.len()).map_err(|_| {
            native_hnsw_corrupted(graph_path, "native HNSW file length does not fit u64")
        })?;
        if actual_len != receipt.graph_byte_len {
            return Err(native_hnsw_corrupted(
                graph_path,
                "native HNSW file length disagrees with generation receipt",
            ));
        }
        // ubs:ignore — the graph SHA-256 is public integrity metadata, not authentication.
        if sha256_hex(graph_bytes) != receipt.graph_sha256 {
            return Err(native_hnsw_corrupted(
                graph_path,
                "native HNSW SHA-256 disagrees with generation receipt",
            ));
        }
        Ok(())
    }

    fn validate_graph_identity(
        receipt: &NativeHnswGenerationReceiptV2,
        graph_path: &Path,
        graph: &NativeHnsw,
        metadata: NativeHnswFileMetadata,
    ) -> SearchResult<()> {
        let mismatch = |field: &str| {
            native_hnsw_corrupted(
                graph_path,
                format!("native HNSW {field} disagrees with generation receipt"),
            )
        };
        if metadata.format_version() != receipt.native_format_version {
            return Err(mismatch("format version"));
        }
        if metadata.byte_len() != receipt.graph_byte_len {
            return Err(mismatch("byte length"));
        }
        // ubs:ignore — point counts are public graph-cardinality evidence, not secrets.
        if metadata.point_count() != receipt.point_count {
            return Err(mismatch("point count"));
        }
        if metadata.payload_crc32() != receipt.payload_crc32 {
            return Err(mismatch("payload CRC"));
        }
        if metadata.header_crc32() != receipt.header_crc32 {
            return Err(mismatch("header CRC"));
        }
        if graph.params() != receipt.params.to_params()? {
            return Err(mismatch("parameter identity"));
        }
        // ubs:ignore — this deterministic topology seed is public receipt metadata.
        if graph.seed() != receipt.seed {
            return Err(mismatch("level-sampling seed"));
        }
        if graph.entry_point() != receipt.entry_point {
            return Err(mismatch("entry point"));
        }
        if usize_to_u64(graph.max_level(), "verified.max_level")? != receipt.max_level {
            return Err(mismatch("maximum level"));
        }
        // ubs:ignore — the topology SHA-256 is public integrity metadata.
        if graph.topology_sha256()? != receipt.topology_sha256 {
            return Err(mismatch("topology SHA-256"));
        }
        Ok(())
    }
}

/// A receipted native HNSW graph that retains the exact admitted FSVI byte
/// owner used for construction or load.
///
/// The graph, receipt binding, and retained owner are private. Public callers
/// cannot extract a receipted graph and reattach it to another same-cardinality
/// vector store:
///
/// ```compile_fail
/// use frankensearch_index::native_hnsw::NativeHnsw;
///
/// fn bypass_receipts() {
///     let _ = std::mem::size_of::<NativeHnsw>();
/// }
/// ```
///
/// Even type inference cannot extract the private graph field:
///
/// ```compile_fail
/// use frankensearch_index::native_hnsw::ValidatedNativeHnsw;
///
/// fn detach(bound: ValidatedNativeHnsw) {
///     let _ = bound.graph;
/// }
/// ```
///
/// Search likewise accepts no caller-provided store:
///
/// ```compile_fail
/// use frankensearch_index::ValidatedFsviBytes;
/// use frankensearch_index::native_hnsw::ValidatedNativeHnsw;
///
/// fn substitute_store(
///     bound: &ValidatedNativeHnsw,
///     other: &ValidatedFsviBytes,
///     query: &[f32],
/// ) {
///     let _ = bound.search(query, 10, None, other);
/// }
/// ```
///
/// A hit borrows the retained owner through the graph handle and therefore
/// cannot escape after the handle is dropped:
///
/// ```compile_fail
/// use frankensearch_index::native_hnsw::ValidatedNativeHnsw;
///
/// fn detach_hit(bound: ValidatedNativeHnsw, query: &[f32]) {
///     let hit = bound.search(query, 1, None).unwrap().remove(0);
///     drop(bound);
///     let _ = hit.doc_id();
/// }
/// ```
#[derive(Debug)]
pub struct ValidatedNativeHnsw {
    owner: Arc<ValidatedFsviBytes>,
    binding: NativeHnswGenerationBindingV2,
    graph: NativeHnsw,
}

/// One native-HNSW result resolved through the same admitted FSVI owner used
/// for graph construction or load.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ValidatedNativeHnswHit<'owner> {
    physical_row: u32,
    distance: f32,
    doc_id: &'owner str,
    flags: crate::FsviRecordFlags,
}

impl ValidatedNativeHnswHit<'_> {
    /// Exact physical FSVI row indexed by the graph.
    #[must_use]
    pub const fn physical_row(&self) -> u32 {
        self.physical_row
    }

    /// Native HNSW distance; smaller values rank nearer.
    #[must_use]
    pub const fn distance(&self) -> f32 {
        self.distance
    }

    /// Document identifier resolved from the bound admitted owner.
    #[must_use]
    pub const fn doc_id(&self) -> &str {
        self.doc_id
    }

    /// Validated LIVE/TOMBSTONE state from the same physical owner row.
    #[must_use]
    pub const fn flags(&self) -> crate::FsviRecordFlags {
        self.flags
    }
}

impl ValidatedNativeHnsw {
    fn binding_for_owner(
        owner: &ValidatedFsviBytes,
    ) -> SearchResult<NativeHnswGenerationBindingV2> {
        NativeHnswGenerationBindingV2::from_validated_fsvi(owner)
    }

    fn from_verified_graph(
        owner: Arc<ValidatedFsviBytes>,
        binding: NativeHnswGenerationBindingV2,
        graph: NativeHnsw,
    ) -> SearchResult<Self> {
        let owner_binding = Self::binding_for_owner(owner.as_ref())?;
        // ubs:ignore -- these are public artifact identity digests, not credentials.
        if binding != owner_binding {
            return Err(native_hnsw_receipt_config_error(
                "generation_binding",
                "redacted-mismatch",
                "must be derived from the exact retained FSVI owner",
            ));
        }
        graph
            .verify_for_store(owner.as_ref())
            .map_err(|source| SearchError::SubsystemError {
                subsystem: "native-hnsw",
                source: Box::new(source),
            })?;
        Ok(Self {
            owner,
            binding,
            graph,
        })
    }

    /// Build a graph from every physical row of one admitted FSVI owner.
    ///
    /// The supplied [`Arc`] is retained by the returned handle. Callers may
    /// drop every other owner and the original source pathname immediately
    /// after this method returns without detaching the graph from its bytes.
    ///
    /// Tombstoned rows remain in the graph as routing nodes. They retain their
    /// exact physical-row identity in topology and receipt material, but
    /// [`Self::search`] never exposes them as hits.
    ///
    /// # Errors
    ///
    /// Returns identity/cardinality errors from receipt binding construction
    /// or distance-computation errors from the admitted owner, and a
    /// `native-hnsw` [`SearchError::SubsystemError`] if post-build structural
    /// attestation rejects the graph.
    pub fn build(
        owner: Arc<ValidatedFsviBytes>,
        params: HnswParams,
        seed: u64,
    ) -> SearchResult<Self> {
        let binding = Self::binding_for_owner(owner.as_ref())?;
        let graph = NativeHnsw::build(params, seed, owner.as_ref())?;
        Self::from_verified_graph(owner, binding, graph)
    }

    /// Load a graph only after its receipt matches the exact admitted owner.
    ///
    /// The supplied [`Arc`] is retained by the returned handle. Receipt
    /// verification, structural attestation, and every later search therefore
    /// resolve rows from the same allocation even after the caller drops its
    /// own reference.
    ///
    /// Receipt identity is checked before graph bytes are opened, and graph
    /// parsing and structural verification use the same retained owner.
    /// The exact whole-image witness binds the LIVE/TOMBSTONE layout, while
    /// every physical row remains present in the loaded routing topology.
    ///
    /// # Errors
    ///
    /// Returns receipt, graph, filesystem, or owner-distance errors without
    /// falling back to a caller-supplied vector store.
    pub fn load(
        owner: Arc<ValidatedFsviBytes>,
        graph_path: &Path,
    ) -> SearchResult<(Self, NativeHnswGenerationReceiptV2)> {
        let binding = Self::binding_for_owner(owner.as_ref())?;
        let (graph, receipt) = binding.load_bound_graph(graph_path, owner.as_ref())?;
        Ok((
            Self {
                owner,
                binding,
                graph,
            },
            receipt,
        ))
    }

    #[cfg(test)]
    fn load_with_after_first_observation(
        owner: Arc<ValidatedFsviBytes>,
        graph_path: &Path,
        after_first_observation: impl FnOnce() -> SearchResult<()>,
    ) -> SearchResult<(Self, NativeHnswGenerationReceiptV2)> {
        let binding = Self::binding_for_owner(owner.as_ref())?;
        let (graph, receipt) = binding.load_bound_graph_with_after_first_observation(
            graph_path,
            owner.as_ref(),
            after_first_observation,
        )?;
        Ok((
            Self {
                owner,
                binding,
                graph,
            },
            receipt,
        ))
    }

    /// Durably publish this owner-built graph, then its adjacent receipt.
    ///
    /// The two renames are not a single transaction: a crash or concurrent
    /// writer between them can leave a mismatched pair, which every load
    /// rejects. Production publication must therefore use a serialized,
    /// generation-specific staging path and select the generation only after
    /// both artifacts are sealed.
    ///
    /// # Errors
    ///
    /// Returns graph, receipt, path-hardening, or filesystem errors. No API
    /// accepts a replacement graph at this boundary.
    pub fn save(&self, graph_path: &Path) -> SearchResult<NativeHnswGenerationReceiptV2> {
        self.binding.save_bound_graph(&self.graph, graph_path)
    }

    /// Search only against the owner used to build or load this graph.
    ///
    /// Every physical row remains eligible for graph traversal, including the
    /// entry point and intermediate tombstones. Returned rows are filtered
    /// through the same owner and are always LIVE. If the initial ANN window
    /// contains too many tombstones, both candidate count and beam width widen
    /// deterministically, never beyond the physical row count, until exactly
    /// `min(k, owner.live_count())` hits are available. A full-width underfill
    /// is a typed structural defect rather than a silent short result.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::DimensionMismatch`] when the query does not
    /// match the admitted owner's dimension, [`SearchError::InvalidConfig`]
    /// for non-finite query values, or graph-distance and owner row-resolution
    /// errors.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: Option<usize>,
    ) -> SearchResult<Vec<ValidatedNativeHnswHit<'_>>> {
        if query.len() != self.owner.dimension() {
            return Err(SearchError::DimensionMismatch {
                expected: self.owner.dimension(),
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
        let owner = self.owner.as_ref();
        let expected_live_hits = k.min(owner.live_count());
        // ubs:ignore -- cardinalities are public index metadata, not secrets.
        if expected_live_hits == 0 {
            return Ok(Vec::new());
        }

        let physical_rows = owner.record_count();
        let mut candidate_count = expected_live_hits.min(physical_rows);
        let mut search_ef = ef
            .unwrap_or_else(|| self.graph.params().ef_search)
            .max(candidate_count)
            .min(physical_rows);

        loop {
            let candidates = self
                .graph
                .search(query, candidate_count, Some(search_ef), owner)?;
            let mut live_hits = Vec::with_capacity(expected_live_hits);
            for (physical_row, distance) in candidates {
                let physical_index =
                    usize::try_from(physical_row).map_err(|_| SearchError::InvalidConfig {
                        field: "native_hnsw.physical_row".to_owned(),
                        value: physical_row.to_string(),
                        reason: "physical row does not fit usize".to_owned(),
                    })?;
                let flags = owner.row(physical_index)?.flags();
                if flags.is_tombstone() {
                    continue;
                }
                let doc_id = owner.doc_id_at(physical_index)?;
                live_hits.push(ValidatedNativeHnswHit {
                    physical_row,
                    distance,
                    doc_id,
                    flags,
                });
                // ubs:ignore -- result cardinalities are not secret material.
                if live_hits.len() == expected_live_hits {
                    return Ok(live_hits);
                }
            }

            // ubs:ignore -- search widths are public index cardinalities.
            if candidate_count == physical_rows {
                return Err(SearchError::SubsystemError {
                    subsystem: "native-hnsw",
                    source: Box::new(GraphDefect::LiveResultUnderfill {
                        requested_k: k,
                        expected_live_hits,
                        returned_live_hits: live_hits.len(),
                        physical_rows,
                    }),
                });
            }

            candidate_count = widen_search_width(candidate_count, physical_rows);
            search_ef = widen_search_width(search_ef, physical_rows).max(candidate_count);
        }
    }

    /// Number of physical owner rows indexed by this graph.
    #[must_use]
    pub fn len(&self) -> usize {
        self.graph.len()
    }

    /// Whether the bound graph and owner are empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.graph.is_empty()
    }

    /// Exact admitted FSVI witness paired with this graph.
    #[must_use]
    pub fn owner_witness(&self) -> &crate::FsviV2Witness {
        self.owner.witness()
    }
}

impl NativeHnswGenerationReceiptV2 {
    /// Validate every internal schema, digest, cardinality, and topology field.
    ///
    /// This does not establish compatibility with caller-held expectations;
    /// use [`ValidatedNativeHnsw::load`] for admission.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] for malformed or internally
    /// inconsistent receipt material.
    pub fn validate(&self) -> SearchResult<()> {
        if self.schema_version != NATIVE_HNSW_GENERATION_RECEIPT_SCHEMA_V2 {
            return Err(native_hnsw_receipt_config_error(
                "schema_version",
                &self.schema_version.to_string(),
                "unsupported native HNSW receipt schema",
            ));
        }
        self.artifact_generation.validate()?;
        validate_sha256_hex(
            "artifact_generation_fingerprint",
            &self.artifact_generation_fingerprint,
        )?;
        // ubs:ignore — the generation SHA-256 is public integrity metadata.
        if self.artifact_generation_fingerprint != self.artifact_generation.fingerprint() {
            return Err(native_hnsw_receipt_config_error(
                "artifact_generation_fingerprint",
                "redacted-digest-mismatch",
                "does not bind the canonical artifact generation",
            ));
        }
        for (field, fingerprint) in [
            (
                "embedding_identity_fingerprint",
                &self.embedding_identity_fingerprint,
            ),
            (
                "embedding_space_fingerprint",
                &self.embedding_space_fingerprint,
            ),
            (
                "embedding_producer_fingerprint",
                &self.embedding_producer_fingerprint,
            ),
            (
                "embedding_input_fingerprint",
                &self.embedding_input_fingerprint,
            ),
            (
                "vector_storage_fingerprint",
                &self.vector_storage_fingerprint,
            ),
            ("vector_content_digest", &self.vector_content_digest),
            (
                "ordered_live_docset_digest",
                &self.ordered_live_docset_digest,
            ),
            ("fsvi_whole_image_sha256", &self.fsvi_whole_image_sha256),
            ("graph_sha256", &self.graph_sha256),
            ("topology_sha256", &self.topology_sha256),
            ("receipt_sha256", &self.receipt_sha256),
        ] {
            validate_sha256_hex(field, fingerprint)?;
        }
        validate_graph_basename(&self.graph_basename)?;
        if self.graph_byte_len < NATIVE_HNSW_HEADER_LEN_U64 {
            return Err(native_hnsw_receipt_config_error(
                "graph_byte_len",
                &self.graph_byte_len.to_string(),
                "must include the complete fixed FSHNSW header",
            ));
        }
        if self.native_format_version != NATIVE_HNSW_FORMAT_VERSION {
            return Err(native_hnsw_receipt_config_error(
                "native_format_version",
                &self.native_format_version.to_string(),
                "unsupported owned FSHNSW format version",
            ));
        }
        let _ = self.params.to_params()?;
        if self.point_count > u64::from(u32::MAX) {
            return Err(native_hnsw_receipt_config_error(
                "point_count",
                &self.point_count.to_string(),
                "must fit the native u32 point-id space",
            ));
        }
        // ubs:ignore — these are public physical graph-row cardinalities, not credentials.
        if self.fsvi_physical_row_count != self.point_count {
            return Err(native_hnsw_receipt_config_error(
                "fsvi_physical_row_count",
                &self.fsvi_physical_row_count.to_string(),
                "must equal the native HNSW point count",
            ));
        }
        if self.max_level >= usize_to_u64(MAX_LEVEL, "max_level_limit")? {
            return Err(native_hnsw_receipt_config_error(
                "max_level",
                &self.max_level.to_string(),
                "must be below the native HNSW layer bound",
            ));
        }
        match (self.point_count, self.entry_point, self.max_level) {
            (0, None, 0) => {}
            (0, Some(_), _) => {
                return Err(native_hnsw_receipt_config_error(
                    "entry_point",
                    "redacted",
                    "must be absent for an empty graph",
                ));
            }
            (0, None, _) => {
                return Err(native_hnsw_receipt_config_error(
                    "max_level",
                    &self.max_level.to_string(),
                    "must be zero for an empty graph",
                ));
            }
            (_, None, _) => {
                return Err(native_hnsw_receipt_config_error(
                    "entry_point",
                    "redacted",
                    "must be present for a non-empty graph",
                ));
            }
            (count, Some(entry), _) if u64::from(entry) >= count => {
                return Err(native_hnsw_receipt_config_error(
                    "entry_point",
                    &entry.to_string(),
                    "must name a row within point_count",
                ));
            }
            _ => {}
        }
        let expected_receipt_sha256 = sha256_hex(&self.canonical_body_bytes()?);
        // ubs:ignore — the receipt SHA-256 is an unkeyed public integrity seal.
        if self.receipt_sha256 != expected_receipt_sha256 {
            return Err(native_hnsw_receipt_config_error(
                "receipt_sha256",
                "redacted-digest-mismatch",
                "does not match the canonical receipt body",
            ));
        }
        Ok(())
    }

    /// Encode this validated receipt to its one canonical binary form.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when the receipt is invalid.
    pub fn to_bytes(&self) -> SearchResult<Vec<u8>> {
        self.validate()?;
        self.encode_unchecked()
    }

    /// Decode and validate one canonical binary receipt.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::IndexCorrupted`] for truncation, an unknown
    /// schema, invalid fields, a body-seal mismatch, or noncanonical/trailing
    /// bytes.
    pub fn from_bytes(bytes: &[u8], receipt_path: &Path) -> SearchResult<Self> {
        let receipt = Self::decode_unchecked(bytes, receipt_path)?;
        receipt.validate().map_err(|error| {
            native_hnsw_receipt_corrupted(
                receipt_path,
                format!("native HNSW generation receipt is invalid: {error}"),
            )
        })?;
        let canonical = receipt.encode_unchecked().map_err(|error| {
            native_hnsw_receipt_corrupted(
                receipt_path,
                format!("native HNSW generation receipt cannot be canonicalized: {error}"),
            )
        })?;
        if canonical != bytes {
            return Err(native_hnsw_receipt_corrupted(
                receipt_path,
                "native HNSW generation receipt is not canonically encoded",
            ));
        }
        Ok(receipt)
    }

    fn seal(&mut self) -> SearchResult<()> {
        self.receipt_sha256 = sha256_hex(&self.canonical_body_bytes()?);
        Ok(())
    }

    fn canonical_body_bytes(&self) -> SearchResult<Vec<u8>> {
        let basename = self.graph_basename.as_bytes();
        let basename_len = u16::try_from(basename.len()).map_err(|_| {
            native_hnsw_receipt_config_error(
                "graph_basename",
                "redacted-oversized",
                "basename length must fit the canonical u16 field",
            )
        })?;
        let mut bytes = Vec::with_capacity(488 + basename.len());
        bytes.extend_from_slice(&NATIVE_HNSW_RECEIPT_MAGIC);
        bytes.extend_from_slice(&self.schema_version.to_be_bytes());
        bytes.extend_from_slice(&self.artifact_generation.schema_version.to_be_bytes());
        bytes.extend_from_slice(&self.artifact_generation.sequence.to_be_bytes());
        bytes.extend_from_slice(&self.artifact_generation.nonce);
        for fingerprint in [
            &self.artifact_generation_fingerprint,
            &self.embedding_identity_fingerprint,
            &self.embedding_space_fingerprint,
            &self.embedding_producer_fingerprint,
            &self.embedding_input_fingerprint,
            &self.vector_storage_fingerprint,
        ] {
            bytes.extend_from_slice(&decode_sha256_hex("receipt fingerprint", fingerprint)?);
        }
        bytes.extend_from_slice(&decode_sha256_hex(
            "vector_content_digest",
            &self.vector_content_digest,
        )?);
        bytes.extend_from_slice(&decode_sha256_hex(
            "ordered_live_docset_digest",
            &self.ordered_live_docset_digest,
        )?);
        bytes.extend_from_slice(&decode_sha256_hex(
            "fsvi_whole_image_sha256",
            &self.fsvi_whole_image_sha256,
        )?);
        bytes.extend_from_slice(&self.fsvi_physical_row_count.to_be_bytes());
        bytes.extend_from_slice(&basename_len.to_be_bytes());
        bytes.extend_from_slice(basename);
        bytes.extend_from_slice(&self.graph_byte_len.to_be_bytes());
        bytes.extend_from_slice(&decode_sha256_hex("graph_sha256", &self.graph_sha256)?);
        bytes.extend_from_slice(&self.native_format_version.to_be_bytes());
        for value in [
            self.params.m,
            self.params.m0,
            self.params.ef_construction,
            self.params.ef_search,
            self.seed,
            self.point_count,
            self.entry_point.map_or(NATIVE_HNSW_NO_ENTRY, u64::from),
            self.max_level,
        ] {
            bytes.extend_from_slice(&value.to_be_bytes());
        }
        bytes.extend_from_slice(&self.payload_crc32.to_be_bytes());
        bytes.extend_from_slice(&self.header_crc32.to_be_bytes());
        bytes.extend_from_slice(&decode_sha256_hex(
            "topology_sha256",
            &self.topology_sha256,
        )?);
        Ok(bytes)
    }

    fn encode_unchecked(&self) -> SearchResult<Vec<u8>> {
        let mut bytes = self.canonical_body_bytes()?;
        bytes.extend_from_slice(&decode_sha256_hex("receipt_sha256", &self.receipt_sha256)?);
        Ok(bytes)
    }

    fn decode_unchecked(bytes: &[u8], receipt_path: &Path) -> SearchResult<Self> {
        let mut reader = NativeHnswReceiptReader::new(bytes, receipt_path);
        let magic = reader.take_array::<8>("receipt magic")?;
        if magic != NATIVE_HNSW_RECEIPT_MAGIC {
            return Err(native_hnsw_receipt_corrupted(
                receipt_path,
                "bad native HNSW generation receipt magic",
            ));
        }
        let schema_version = reader.read_u16("schema version")?;
        if schema_version != NATIVE_HNSW_GENERATION_RECEIPT_SCHEMA_V2 {
            return Err(native_hnsw_receipt_corrupted(
                receipt_path,
                format!(
                    "unsupported native HNSW generation receipt schema {schema_version}; expected \
                     {NATIVE_HNSW_GENERATION_RECEIPT_SCHEMA_V2}"
                ),
            ));
        }
        let artifact_generation = ArtifactGenerationIdentityV1 {
            schema_version: reader.read_u16("artifact generation schema")?,
            sequence: reader.read_u64("artifact generation sequence")?,
            nonce: reader.take_array::<16>("artifact generation nonce")?,
        };
        let artifact_generation_fingerprint =
            encode_lower_hex(reader.take_array::<SHA256_BYTES>("artifact generation fingerprint")?);
        let embedding_identity_fingerprint =
            encode_lower_hex(reader.take_array::<SHA256_BYTES>("embedding identity fingerprint")?);
        let embedding_space_fingerprint =
            encode_lower_hex(reader.take_array::<SHA256_BYTES>("embedding space fingerprint")?);
        let embedding_producer_fingerprint =
            encode_lower_hex(reader.take_array::<SHA256_BYTES>("embedding producer fingerprint")?);
        let embedding_input_fingerprint =
            encode_lower_hex(reader.take_array::<SHA256_BYTES>("embedding input fingerprint")?);
        let vector_storage_fingerprint =
            encode_lower_hex(reader.take_array::<SHA256_BYTES>("vector storage fingerprint")?);
        let vector_content_digest =
            encode_lower_hex(reader.take_array::<SHA256_BYTES>("vector content digest")?);
        let ordered_live_docset_digest =
            encode_lower_hex(reader.take_array::<SHA256_BYTES>("ordered live-docset digest")?);
        let fsvi_whole_image_sha256 =
            encode_lower_hex(reader.take_array::<SHA256_BYTES>("FSVI whole-image SHA-256")?);
        let fsvi_physical_row_count = reader.read_u64("FSVI physical row count")?;
        let basename_len = usize::from(reader.read_u16("graph basename length")?);
        if basename_len == 0 || basename_len > NATIVE_HNSW_MAX_BASENAME_BYTES {
            return Err(native_hnsw_receipt_corrupted(
                receipt_path,
                "native HNSW receipt graph basename length is invalid",
            ));
        }
        let graph_basename = std::str::from_utf8(reader.take(basename_len, "graph basename")?)
            .map_err(|_| {
                native_hnsw_receipt_corrupted(
                    receipt_path,
                    "native HNSW receipt graph basename is not UTF-8",
                )
            })?
            .to_owned();
        let graph_byte_len = reader.read_u64("graph byte length")?;
        let graph_sha256 = encode_lower_hex(reader.take_array::<SHA256_BYTES>("graph SHA-256")?);
        let native_format_version = reader.read_u32("native format version")?;
        let params = NativeHnswParamsIdentityV1 {
            m: reader.read_u64("m")?,
            m0: reader.read_u64("m0")?,
            ef_construction: reader.read_u64("ef_construction")?,
            ef_search: reader.read_u64("ef_search")?,
        };
        let seed = reader.read_u64("seed")?;
        let point_count = reader.read_u64("point count")?;
        let entry_wire = reader.read_u64("entry point")?;
        let entry_point = if entry_wire == NATIVE_HNSW_NO_ENTRY {
            None
        } else {
            Some(u32::try_from(entry_wire).map_err(|_| {
                native_hnsw_receipt_corrupted(
                    receipt_path,
                    "native HNSW receipt entry point exceeds u32",
                )
            })?)
        };
        let max_level = reader.read_u64("maximum level")?;
        let payload_crc32 = reader.read_u32("payload CRC")?;
        let header_crc32 = reader.read_u32("header CRC")?;
        let topology_sha256 =
            encode_lower_hex(reader.take_array::<SHA256_BYTES>("topology SHA-256")?);
        let receipt_sha256 =
            encode_lower_hex(reader.take_array::<SHA256_BYTES>("receipt SHA-256")?);
        reader.finish()?;

        Ok(Self {
            schema_version,
            artifact_generation,
            artifact_generation_fingerprint,
            embedding_identity_fingerprint,
            embedding_space_fingerprint,
            embedding_producer_fingerprint,
            embedding_input_fingerprint,
            vector_storage_fingerprint,
            vector_content_digest,
            ordered_live_docset_digest,
            fsvi_whole_image_sha256,
            fsvi_physical_row_count,
            graph_basename,
            graph_byte_len,
            graph_sha256,
            native_format_version,
            params,
            seed,
            point_count,
            entry_point,
            max_level,
            payload_crc32,
            header_crc32,
            topology_sha256,
            receipt_sha256,
        })
    }
}

/// Canonical adjacent receipt path for `graph_path`.
///
/// # Errors
///
/// Returns [`SearchError::InvalidConfig`] for a non-UTF-8, non-`.fshnsw`,
/// control-bearing, dot-relative, or otherwise noncanonical graph path.
pub fn native_hnsw_generation_receipt_path(graph_path: &Path) -> SearchResult<PathBuf> {
    validate_native_hnsw_graph_path(graph_path)?;
    let basename = canonical_graph_basename(graph_path)?;
    let receipt_basename = format!("{basename}{NATIVE_HNSW_GENERATION_RECEIPT_SUFFIX}");
    Ok(graph_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(receipt_basename))
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            m: 16,
            m0: 32,
            ef_construction: 200,
            ef_search: 100,
        }
    }
}

impl HnswParams {
    /// Validate the parameters.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when a parameter would make
    /// the graph degenerate: a zero degree disconnects the graph, and a
    /// beam narrower than one candidate cannot make progress.
    pub fn validate(&self) -> SearchResult<()> {
        let invalid = |field: &str, value: usize, reason: &str| SearchError::InvalidConfig {
            field: field.to_owned(),
            value: value.to_string(),
            reason: reason.to_owned(),
        };
        if self.m == 0 {
            return Err(invalid("m", self.m, "must be >= 1"));
        }
        if self.m0 == 0 {
            return Err(invalid("m0", self.m0, "must be >= 1"));
        }
        if self.ef_construction == 0 {
            return Err(invalid(
                "ef_construction",
                self.ef_construction,
                "must be >= 1",
            ));
        }
        if self.ef_search == 0 {
            return Err(invalid("ef_search", self.ef_search, "must be >= 1"));
        }
        Ok(())
    }

    /// Neighbour budget for `layer`.
    #[must_use]
    pub const fn degree_at(&self, layer: usize) -> usize {
        if layer == 0 { self.m0 } else { self.m }
    }
}

/// A candidate under consideration, ordered by distance.
///
/// Distances are compared with [`f32::total_cmp`] so a NaN — which a
/// corrupt vector could produce — yields a deterministic order instead of
/// undefined heap behaviour.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Candidate {
    distance: f32,
    id: u32,
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Min-heap ordering: `BinaryHeap` is a max-heap, so invert.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Nearest(Candidate);

impl Eq for Nearest {}

impl Ord for Nearest {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.cmp(&self.0)
    }
}

impl PartialOrd for Nearest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Reusable "have I seen this point?" set for a beam search.
///
/// A fresh `vec![false; len]` per beam search would make construction
/// quadratic in the corpus size purely in zeroing cost — every insert runs
/// one beam search per layer, so an n-point build writes O(n² · layers)
/// bytes before doing any useful work. Instead the buffer is allocated once
/// and reused: each search bumps an epoch, and a point counts as visited
/// only if its mark equals the current epoch, so clearing is O(1).
#[derive(Debug, Clone, Default)]
struct VisitedSet {
    marks: Vec<u32>,
    epoch: u32,
}

impl VisitedSet {
    /// Begin a new search over `len` points, discarding previous marks.
    fn begin(&mut self, len: usize) {
        if self.marks.len() < len {
            self.marks.resize(len, 0);
        }
        // `0` means "never visited", so the epoch must never be 0. On
        // wraparound, clear the marks once rather than let a stale mark
        // from 2^32 searches ago read as visited.
        match self.epoch.checked_add(1) {
            Some(next) => self.epoch = next,
            None => {
                self.marks.fill(0);
                self.epoch = 1;
            }
        }
    }

    /// Mark `id` visited, returning whether it was already seen.
    ///
    /// Ids outside the buffer report as already-visited so callers skip
    /// them; a point that does not exist cannot be explored.
    fn visit(&mut self, id: u32) -> bool {
        let Some(mark) = self.marks.get_mut(id as usize) else {
            return true;
        };
        if *mark == self.epoch {
            return true;
        }
        *mark = self.epoch;
        false
    }
}

/// Deterministic level sampler.
///
/// HNSW assigns each point a level from a geometric distribution. Using a
/// seeded counter-based hash rather than a thread RNG makes graph topology
/// a pure function of insertion order and seed, which is what lets the
/// regression fixtures below pin exact structural outcomes.
#[derive(Debug, Clone, Copy)]
struct LevelSampler {
    seed: u64,
    level_scale: f64,
}

impl LevelSampler {
    /// Create a sampler for the given degree and seed.
    #[must_use]
    fn new(m: usize, seed: u64) -> Self {
        // The standard 1/ln(M) scale; guard M == 1, whose ln is zero.
        let level_scale = if m <= 1 { 1.0 } else { 1.0 / (m as f64).ln() };
        Self { seed, level_scale }
    }

    /// Sample the level for `id`, in `0..MAX_LEVEL`.
    #[must_use]
    fn level_for(&self, id: u32) -> usize {
        // SplitMix64 finalizer over (seed, id): deterministic, well
        // distributed, and independent of insertion order.
        let mut z = self
            .seed
            .wrapping_add(u64::from(id).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        // Map to (0, 1] and take the geometric level.
        let unit = ((z >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
        let level = (-unit.ln() * self.level_scale).floor();
        if !level.is_finite() || level <= 0.0 {
            return 0;
        }
        // Walk up to the sampled level rather than casting the float.
        // `MAX_LEVEL` is small, so this is at most a handful of comparisons,
        // and it makes the ceiling structural: there is no `as usize` whose
        // out-of-range behaviour would silently saturate a corrupt or
        // extreme value into a plausible-looking level.
        let mut chosen = 0usize;
        while chosen + 1 < MAX_LEVEL && (chosen + 1) as f64 <= level {
            chosen += 1;
        }
        chosen
    }
}

/// Distance function over stored vectors.
///
/// The graph stores no vectors of its own: it indexes rows of a caller-owned
/// store and asks for distances by row id. That keeps a single copy of the
/// vector data (the FSVI slab) and lets the graph reuse whichever SIMD
/// kernel the store already selected.
trait VectorDistance {
    /// Distance between stored row `id` and `query`. Smaller is nearer.
    ///
    /// # Errors
    ///
    /// Returns whatever the underlying store returns for an unreadable row.
    fn distance_to_query(&self, id: u32, query: &[f32]) -> SearchResult<f32>;

    /// Distance between two stored rows.
    ///
    /// # Errors
    ///
    /// Returns whatever the underlying store returns for an unreadable row.
    fn distance_between(&self, a: u32, b: u32) -> SearchResult<f32>;

    /// Number of rows available.
    fn len(&self) -> usize;
}

impl VectorDistance for ValidatedFsviBytes {
    fn distance_to_query(&self, id: u32, query: &[f32]) -> SearchResult<f32> {
        let index = usize::try_from(id).map_err(|_| SearchError::InvalidConfig {
            field: "native_hnsw.physical_row".to_owned(),
            value: id.to_string(),
            reason: "physical row does not fit usize".to_owned(),
        })?;
        let vector = self.vector_at_f32(index)?;
        Ok(1.0 - crate::dot_product_f32_f32(&vector, query)?)
    }

    fn distance_between(&self, a: u32, b: u32) -> SearchResult<f32> {
        let index = usize::try_from(a).map_err(|_| SearchError::InvalidConfig {
            field: "native_hnsw.physical_row".to_owned(),
            value: a.to_string(),
            reason: "physical row does not fit usize".to_owned(),
        })?;
        let left = self.vector_at_f32(index)?;
        self.distance_to_query(b, &left)
    }

    fn len(&self) -> usize {
        self.record_count()
    }
}

/// Per-point adjacency: one neighbour list per layer the point occupies.
#[derive(Debug, Clone, Default)]
struct Adjacency {
    /// `layers[l]` holds the neighbours at layer `l`. Length is
    /// `level + 1`, so indexing beyond the point's level is impossible by
    /// construction rather than by check.
    layers: Vec<Vec<u32>>,
}

impl Adjacency {
    fn with_level(level: usize) -> Self {
        Self {
            layers: vec![Vec::new(); level + 1],
        }
    }

    /// The point's sampled level.
    fn level(&self) -> usize {
        self.layers.len().saturating_sub(1)
    }

    /// Whether this point participates in `layer`.
    ///
    /// bd-u3wt defect (4): participation is `level >= layer`, never
    /// `level == layer`.
    fn participates_in(&self, layer: usize) -> bool {
        layer < self.layers.len()
    }

    fn neighbours(&self, layer: usize) -> &[u32] {
        self.layers.get(layer).map_or(&[], Vec::as_slice)
    }
}

/// Original neighbour lists touched by one insertion.
///
/// Insertion mutates only a bounded neighbourhood, so journaling those lists
/// avoids cloning the complete graph while still making every returned error
/// failure-atomic. Lists belonging to the newly appended point are omitted:
/// rollback removes that point wholesale.
#[derive(Debug)]
struct MutationJournal {
    original_point_count: usize,
    original_lists: BTreeMap<(usize, usize), Vec<u32>>,
}

impl MutationJournal {
    fn new(original_point_count: usize) -> Self {
        Self {
            original_point_count,
            original_lists: BTreeMap::new(),
        }
    }

    fn record(&mut self, adjacency: &[Adjacency], id: u32, layer: usize) {
        let point_index = id as usize;
        if point_index >= self.original_point_count
            || self.original_lists.contains_key(&(point_index, layer))
        {
            return;
        }
        if let Some(original) = adjacency
            .get(point_index)
            .and_then(|point| point.layers.get(layer))
        {
            self.original_lists
                .insert((point_index, layer), original.clone());
        }
    }

    fn rollback(self, adjacency: &mut [Adjacency]) {
        for ((point_index, layer), original) in self.original_lists {
            if let Some(neighbours) = adjacency
                .get_mut(point_index)
                .and_then(|point| point.layers.get_mut(layer))
            {
                *neighbours = original;
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct NativeHnswHeader {
    params: HnswParams,
    seed: u64,
    point_count: usize,
    entry: Option<u32>,
    max_level: usize,
    payload_len: u64,
    payload_crc32: u32,
    header_crc32: u32,
}

impl NativeHnswHeader {
    fn encode(self) -> SearchResult<[u8; NATIVE_HNSW_HEADER_LEN]> {
        let mut bytes = [0_u8; NATIVE_HNSW_HEADER_LEN];
        bytes[..NATIVE_HNSW_MAGIC.len()].copy_from_slice(&NATIVE_HNSW_MAGIC);
        put_u32(&mut bytes, 8, NATIVE_HNSW_FORMAT_VERSION);
        put_u32(
            &mut bytes,
            12,
            u32::try_from(NATIVE_HNSW_HEADER_LEN).map_err(|_| SearchError::InvalidConfig {
                field: "native_hnsw_header_len".to_owned(),
                value: NATIVE_HNSW_HEADER_LEN.to_string(),
                reason: "header length must fit in u32".to_owned(),
            })?,
        );
        put_u64(&mut bytes, 16, usize_to_u64(self.params.m, "m")?);
        put_u64(&mut bytes, 24, usize_to_u64(self.params.m0, "m0")?);
        put_u64(
            &mut bytes,
            32,
            usize_to_u64(self.params.ef_construction, "ef_construction")?,
        );
        put_u64(
            &mut bytes,
            40,
            usize_to_u64(self.params.ef_search, "ef_search")?,
        );
        put_u64(&mut bytes, 48, self.seed);
        put_u64(
            &mut bytes,
            56,
            usize_to_u64(self.point_count, "point_count")?,
        );
        put_u64(
            &mut bytes,
            64,
            self.entry.map_or(NATIVE_HNSW_NO_ENTRY, u64::from),
        );
        put_u64(&mut bytes, 72, usize_to_u64(self.max_level, "max_level")?);
        put_u64(&mut bytes, 80, self.payload_len);
        put_u32(&mut bytes, 88, self.payload_crc32);
        let header_crc32 = crc32fast::hash(&bytes[..NATIVE_HNSW_HEADER_CRC_OFFSET]);
        put_u32(&mut bytes, NATIVE_HNSW_HEADER_CRC_OFFSET, header_crc32);
        Ok(bytes)
    }

    // ubs:ignore — this parses the owned graph header, not a JWT or authentication token.
    fn decode(
        path: &Path,
        bytes: &[u8; NATIVE_HNSW_HEADER_LEN],
        file_len: u64,
    ) -> SearchResult<Self> {
        if bytes[..NATIVE_HNSW_MAGIC.len()] != NATIVE_HNSW_MAGIC {
            return Err(native_hnsw_corrupted(path, "bad native HNSW magic"));
        }
        let version = get_u32(bytes, 8);
        if version != NATIVE_HNSW_FORMAT_VERSION {
            return Err(native_hnsw_corrupted(
                path,
                format!(
                    "unsupported native HNSW format version {version}; expected \
                     {NATIVE_HNSW_FORMAT_VERSION}"
                ),
            ));
        }
        let header_len = usize::try_from(get_u32(bytes, 12)).map_err(|_| {
            native_hnsw_corrupted(path, "native HNSW header length does not fit usize")
        })?;
        if header_len != NATIVE_HNSW_HEADER_LEN {
            return Err(native_hnsw_corrupted(
                path,
                format!(
                    "native HNSW header length {header_len} does not equal \
                     {NATIVE_HNSW_HEADER_LEN}"
                ),
            ));
        }
        let header_crc32 = get_u32(bytes, NATIVE_HNSW_HEADER_CRC_OFFSET);
        let expected_header_crc32 = crc32fast::hash(&bytes[..NATIVE_HNSW_HEADER_CRC_OFFSET]);
        if header_crc32 != expected_header_crc32 {
            return Err(native_hnsw_corrupted(
                path,
                "native HNSW header CRC mismatch",
            ));
        }

        let params = HnswParams {
            m: persisted_usize(path, get_u64(bytes, 16), "m")?,
            m0: persisted_usize(path, get_u64(bytes, 24), "m0")?,
            ef_construction: persisted_usize(path, get_u64(bytes, 32), "ef_construction")?,
            ef_search: persisted_usize(path, get_u64(bytes, 40), "ef_search")?,
        };
        params.validate().map_err(|error| {
            native_hnsw_corrupted(path, format!("invalid persisted HNSW parameters: {error}"))
        })?;

        let point_count_wire = get_u64(bytes, 56);
        let u32_id_space = u64::from(u32::MAX) + 1;
        if point_count_wire > u32_id_space {
            return Err(native_hnsw_corrupted(
                path,
                "native HNSW point count exceeds u32 id space",
            ));
        }
        let point_count = persisted_usize(path, point_count_wire, "point_count")?;
        let entry_wire = get_u64(bytes, 64);
        let entry =
            if entry_wire == NATIVE_HNSW_NO_ENTRY {
                None
            } else {
                Some(u32::try_from(entry_wire).map_err(|_| {
                    native_hnsw_corrupted(path, "native HNSW entry point exceeds u32")
                })?)
            };
        let max_level = persisted_usize(path, get_u64(bytes, 72), "max_level")?;
        if max_level >= MAX_LEVEL {
            return Err(native_hnsw_corrupted(
                path,
                format!("native HNSW maximum level {max_level} exceeds format cap"),
            ));
        }
        let payload_len = get_u64(bytes, 80);
        let expected_file_len = NATIVE_HNSW_HEADER_LEN_U64
            .checked_add(payload_len)
            .ok_or_else(|| native_hnsw_corrupted(path, "native HNSW file length overflow"))?;
        if file_len != expected_file_len {
            return Err(native_hnsw_corrupted(
                path,
                format!(
                    "native HNSW file length {file_len} does not match header-declared \
                     length {expected_file_len}"
                ),
            ));
        }
        let minimum_payload = usize_to_u64(point_count, "point_count")?
            .checked_mul(8)
            .ok_or_else(|| native_hnsw_corrupted(path, "minimum payload length overflow"))?;
        if payload_len < minimum_payload {
            return Err(native_hnsw_corrupted(
                path,
                "native HNSW payload is too short for its point count",
            ));
        }

        Ok(Self {
            params,
            seed: get_u64(bytes, 48),
            point_count,
            entry,
            max_level,
            payload_len,
            payload_crc32: get_u32(bytes, 88),
            header_crc32,
        })
    }

    fn metadata(self) -> SearchResult<NativeHnswFileMetadata> {
        Ok(NativeHnswFileMetadata {
            format_version: NATIVE_HNSW_FORMAT_VERSION,
            byte_len: NATIVE_HNSW_HEADER_LEN_U64
                .checked_add(self.payload_len)
                .ok_or_else(|| SearchError::InvalidConfig {
                    field: "native_hnsw_file_len".to_owned(),
                    value: self.payload_len.to_string(),
                    reason: "file length overflow".to_owned(),
                })?,
            point_count: usize_to_u64(self.point_count, "point_count")?,
            payload_crc32: self.payload_crc32,
            header_crc32: self.header_crc32,
        })
    }
}

struct NativeHnswPayloadReader<'a> {
    file: &'a mut File,
    remaining: u64,
    hasher: crc32fast::Hasher,
}

impl<'a> NativeHnswPayloadReader<'a> {
    fn new(file: &'a mut File, payload_len: u64) -> Self {
        Self {
            file,
            remaining: payload_len,
            hasher: crc32fast::Hasher::new(),
        }
    }

    fn read_u32(&mut self, path: &Path, field: &str) -> SearchResult<u32> {
        let field_len = 4_u64;
        if self.remaining < field_len {
            return Err(native_hnsw_corrupted(
                path,
                format!("native HNSW payload ended while reading {field}"),
            ));
        }
        let mut bytes = [0_u8; 4];
        self.file.read_exact(&mut bytes).map_err(|error| {
            native_hnsw_corrupted(path, format!("could not read native HNSW {field}: {error}"))
        })?;
        self.hasher.update(&bytes);
        self.remaining -= field_len;
        Ok(u32::from_le_bytes(bytes))
    }

    fn finish(self, path: &Path, expected_crc32: u32) -> SearchResult<()> {
        if self.remaining != 0 {
            return Err(native_hnsw_corrupted(
                path,
                format!(
                    "native HNSW payload has {} unparsed trailing bytes",
                    self.remaining
                ),
            ));
        }
        let actual_crc32 = self.hasher.finalize();
        if actual_crc32 != expected_crc32 {
            return Err(native_hnsw_corrupted(
                path,
                "native HNSW payload CRC mismatch",
            ));
        }
        Ok(())
    }
}

/// A navigable small-world graph over rows of a vector store.
#[derive(Debug, Clone)]
struct NativeHnsw {
    params: HnswParams,
    sampler: LevelSampler,
    adjacency: Vec<Adjacency>,
    /// Entry point: the id of a point at the current maximum level.
    entry: Option<u32>,
    max_level: usize,
    /// Visited-set buffer reused across every insert, so construction does
    /// not re-zero an n-sized array per layer per point.
    scratch: VisitedSet,
}

impl NativeHnsw {
    /// Create an empty graph.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] for degenerate parameters.
    fn new(params: HnswParams, seed: u64) -> SearchResult<Self> {
        params.validate()?;
        Ok(Self {
            sampler: LevelSampler::new(params.m, seed),
            params,
            adjacency: Vec::new(),
            entry: None,
            max_level: 0,
            scratch: VisitedSet::default(),
        })
    }

    /// Number of indexed points.
    #[must_use]
    fn len(&self) -> usize {
        self.adjacency.len()
    }

    /// Whether the graph indexes no points.
    #[must_use]
    fn is_empty(&self) -> bool {
        self.adjacency.is_empty()
    }

    /// The entry point id, if any point has been inserted.
    #[must_use]
    const fn entry_point(&self) -> Option<u32> {
        self.entry
    }

    /// The current maximum level.
    #[must_use]
    const fn max_level(&self) -> usize {
        self.max_level
    }

    /// Construction and search parameters persisted with this graph.
    #[must_use]
    const fn params(&self) -> HnswParams {
        self.params
    }

    /// Deterministic level-sampling seed persisted with this graph.
    #[must_use]
    const fn seed(&self) -> u64 {
        self.sampler.seed
    }

    /// SHA-256 of the graph's semantic topology and construction identity.
    ///
    /// File checksums and byte offsets are intentionally excluded. Point and
    /// neighbour order remain included because both are search-visible state
    /// and the owned builder is deterministic for a seed.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::IndexCorrupted`] when the graph is structurally
    /// invalid and [`SearchError::InvalidConfig`] when a count cannot be
    /// represented canonically.
    fn topology_sha256(&self) -> SearchResult<String> {
        self.verify()
            .map_err(|defect| native_hnsw_corrupted(Path::new("<memory>"), defect.to_string()))?;
        let mut bytes = Vec::new();
        append_canonical_bytes(&mut bytes, b"frankensearch.native-hnsw.topology.v1")?;
        bytes.extend_from_slice(&NATIVE_HNSW_FORMAT_VERSION.to_be_bytes());
        for value in [
            usize_to_u64(self.params.m, "topology.params.m")?,
            usize_to_u64(self.params.m0, "topology.params.m0")?,
            usize_to_u64(
                self.params.ef_construction,
                "topology.params.ef_construction",
            )?,
            usize_to_u64(self.params.ef_search, "topology.params.ef_search")?,
            self.sampler.seed,
            usize_to_u64(self.adjacency.len(), "topology.point_count")?,
            self.entry.map_or(NATIVE_HNSW_NO_ENTRY, u64::from),
            usize_to_u64(self.max_level, "topology.max_level")?,
        ] {
            bytes.extend_from_slice(&value.to_be_bytes());
        }
        for point in &self.adjacency {
            bytes.extend_from_slice(
                &u32::try_from(point.layers.len())
                    .map_err(|_| {
                        native_hnsw_receipt_config_error(
                            "topology.layer_count",
                            &point.layers.len().to_string(),
                            "must fit the canonical u32 field",
                        )
                    })?
                    .to_be_bytes(),
            );
            for neighbours in &point.layers {
                bytes.extend_from_slice(
                    &u32::try_from(neighbours.len())
                        .map_err(|_| {
                            native_hnsw_receipt_config_error(
                                "topology.neighbour_count",
                                &neighbours.len().to_string(),
                                "must fit the canonical u32 field",
                            )
                        })?
                        .to_be_bytes(),
                );
                for neighbour in neighbours {
                    bytes.extend_from_slice(&neighbour.to_be_bytes());
                }
            }
        }
        Ok(sha256_hex(&bytes))
    }

    /// Persist this graph as an owned, versioned adjacency artifact.
    ///
    /// The graph is structurally attested before the destination is touched.
    /// Publication writes and syncs a temporary file, atomically renames it
    /// over `path`, then syncs the parent directory. Existing symlink and
    /// special-file targets are rejected rather than followed.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::IndexCorrupted`] when the in-memory graph fails
    /// structural attestation, [`SearchError::InvalidConfig`] when a value
    /// cannot be represented by the format, and [`SearchError::Io`] for
    /// filesystem failures.
    fn save(&self, path: &Path) -> SearchResult<NativeHnswFileMetadata> {
        self.verify()
            .map_err(|defect| native_hnsw_corrupted(path, defect.to_string()))?;

        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        std::fs::create_dir_all(parent).map_err(SearchError::Io)?;
        reject_non_regular_destination(path)?;

        let mut temporary = tempfile::NamedTempFile::new_in(parent).map_err(SearchError::Io)?;
        temporary
            .as_file_mut()
            .write_all(&[0_u8; NATIVE_HNSW_HEADER_LEN])
            .map_err(SearchError::Io)?;

        let mut payload_hasher = crc32fast::Hasher::new();
        let mut payload_len = 0_u64;
        for point in &self.adjacency {
            write_payload_u32(
                temporary.as_file_mut(),
                &mut payload_hasher,
                &mut payload_len,
                u32::try_from(point.layers.len()).map_err(|_| SearchError::InvalidConfig {
                    field: "native_hnsw_layer_count".to_owned(),
                    value: point.layers.len().to_string(),
                    reason: "layer count must fit in u32".to_owned(),
                })?,
            )?;
            for neighbours in &point.layers {
                write_payload_u32(
                    temporary.as_file_mut(),
                    &mut payload_hasher,
                    &mut payload_len,
                    u32::try_from(neighbours.len()).map_err(|_| SearchError::InvalidConfig {
                        field: "native_hnsw_neighbour_count".to_owned(),
                        value: neighbours.len().to_string(),
                        reason: "neighbour count must fit in u32".to_owned(),
                    })?,
                )?;
                for &neighbour in neighbours {
                    write_payload_u32(
                        temporary.as_file_mut(),
                        &mut payload_hasher,
                        &mut payload_len,
                        neighbour,
                    )?;
                }
            }
        }

        let payload_crc32 = payload_hasher.finalize();
        let header = NativeHnswHeader {
            params: self.params,
            seed: self.sampler.seed,
            point_count: self.adjacency.len(),
            entry: self.entry,
            max_level: self.max_level,
            payload_len,
            payload_crc32,
            header_crc32: 0,
        };
        let encoded_header = header.encode()?;
        temporary
            .as_file_mut()
            .seek(SeekFrom::Start(0))
            .map_err(SearchError::Io)?;
        temporary
            .as_file_mut()
            .write_all(&encoded_header)
            .map_err(SearchError::Io)?;
        temporary.as_file().sync_all().map_err(SearchError::Io)?;

        let metadata = NativeHnswHeader {
            header_crc32: get_u32(&encoded_header, NATIVE_HNSW_HEADER_CRC_OFFSET),
            ..header
        }
        .metadata()?;
        temporary.persist(path).map_err(|error| {
            SearchError::Io(std::io::Error::new(
                error.error.kind(),
                format!(
                    "failed to atomically publish native HNSW graph '{}': {}",
                    path.display(),
                    error.error
                ),
            ))
        })?;
        sync_parent_directory(path)?;
        Ok(metadata)
    }

    /// Load and attest an owned graph artifact against its exact vector store.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::IndexNotFound`] when `path` is absent,
    /// [`SearchError::IndexCorrupted`] for malformed bytes or a graph/store
    /// mismatch, and [`SearchError::Io`] for other filesystem failures.
    #[cfg(test)]
    fn load<D: VectorDistance>(path: &Path, store: &D) -> SearchResult<Self> {
        Self::load_with_metadata(path, store).map(|(graph, _)| graph)
    }

    /// Load a graph together with its checked format metadata.
    ///
    /// The parser reads bounded fields directly into owned adjacency lists;
    /// it never exposes persisted bytes as graph memory. The opened inode is
    /// compared with the path metadata to reject symlink and replacement
    /// races at the open boundary.
    ///
    /// # Errors
    ///
    /// Returns not-found, corruption, owner-cardinality, and filesystem
    /// errors from bounded parsing and structural attestation.
    fn load_with_metadata<D: VectorDistance>(
        path: &Path,
        store: &D,
    ) -> SearchResult<(Self, NativeHnswFileMetadata)> {
        let mut file = open_regular_file(path)?;
        let file_len = file.metadata().map_err(SearchError::Io)?.len();
        if file_len < NATIVE_HNSW_HEADER_LEN_U64 {
            return Err(native_hnsw_corrupted(
                path,
                "native HNSW file is shorter than its fixed header",
            ));
        }

        let mut header_bytes = [0_u8; NATIVE_HNSW_HEADER_LEN];
        file.read_exact(&mut header_bytes).map_err(|error| {
            native_hnsw_corrupted(path, format!("could not read native HNSW header: {error}"))
        })?;
        // ubs:ignore — this decodes the owned graph header, not a JWT or authentication token.
        let header = NativeHnswHeader::decode(path, &header_bytes, file_len)?;
        let metadata = header.metadata()?;
        if header.point_count != store.len() {
            return Err(native_hnsw_corrupted(
                path,
                GraphDefect::StoreCardinalityMismatch {
                    graph_points: header.point_count,
                    store_rows: store.len(),
                }
                .to_string(),
            ));
        }

        let mut adjacency = Vec::new();
        adjacency
            .try_reserve_exact(header.point_count)
            .map_err(|error| {
                native_hnsw_corrupted(
                    path,
                    format!(
                        "could not allocate native HNSW point table for {} points: {error}",
                        header.point_count
                    ),
                )
            })?;
        let sampler = LevelSampler::new(header.params.m, header.seed);
        let mut payload = NativeHnswPayloadReader::new(&mut file, header.payload_len);
        for point_index in 0..header.point_count {
            let point_id = u32::try_from(point_index)
                .map_err(|_| native_hnsw_corrupted(path, "native HNSW point id exceeds u32"))?;
            let layer_count = persisted_usize(
                path,
                u64::from(payload.read_u32(path, "point layer count")?),
                "layer_count",
            )?;
            if !(1..=MAX_LEVEL).contains(&layer_count) {
                return Err(native_hnsw_corrupted(
                    path,
                    format!("point {point_id} has invalid native HNSW layer count {layer_count}"),
                ));
            }
            let expected_level = sampler.level_for(point_id);
            let actual_level = layer_count - 1;
            if actual_level != expected_level {
                return Err(native_hnsw_corrupted(
                    path,
                    GraphDefect::SampledLevelMismatch {
                        id: point_id,
                        expected: expected_level,
                        actual: actual_level,
                    }
                    .to_string(),
                ));
            }

            let mut layers = Vec::new();
            layers.try_reserve_exact(layer_count).map_err(|error| {
                native_hnsw_corrupted(
                    path,
                    format!(
                        "could not allocate {layer_count} native HNSW layers for point \
                         {point_id}: {error}"
                    ),
                )
            })?;
            for layer in 0..layer_count {
                let neighbour_count = persisted_usize(
                    path,
                    u64::from(payload.read_u32(path, "layer neighbour count")?),
                    "neighbour_count",
                )?;
                let budget = header.params.degree_at(layer);
                if neighbour_count > budget {
                    return Err(native_hnsw_corrupted(
                        path,
                        format!(
                            "point {point_id} layer {layer} holds {neighbour_count} neighbours, \
                             exceeding degree budget {budget}"
                        ),
                    ));
                }
                let neighbour_bytes = usize_to_u64(neighbour_count, "neighbour_count")?
                    .checked_mul(4)
                    .ok_or_else(|| {
                        native_hnsw_corrupted(path, "native HNSW neighbour-byte count overflow")
                    })?;
                if neighbour_bytes > payload.remaining {
                    return Err(native_hnsw_corrupted(
                        path,
                        format!(
                            "point {point_id} layer {layer} declares {neighbour_count} \
                             neighbours beyond the remaining payload"
                        ),
                    ));
                }
                let mut neighbours = Vec::new();
                neighbours
                    .try_reserve_exact(neighbour_count)
                    .map_err(|error| {
                        native_hnsw_corrupted(
                            path,
                            format!(
                                "could not allocate {neighbour_count} native HNSW neighbours for \
                                 point {point_id} layer {layer}: {error}"
                            ),
                        )
                    })?;
                for _ in 0..neighbour_count {
                    neighbours.push(payload.read_u32(path, "neighbour id")?);
                }
                layers.push(neighbours);
            }
            adjacency.push(Adjacency { layers });
        }
        payload.finish(path, header.payload_crc32)?;

        let graph = Self {
            params: header.params,
            sampler,
            adjacency,
            entry: header.entry,
            max_level: header.max_level,
            scratch: VisitedSet::default(),
        };
        graph
            .verify_for_store(store)
            .map_err(|defect| native_hnsw_corrupted(path, defect.to_string()))?;
        Ok((graph, metadata))
    }

    /// Build a graph over every row of `store`, in row order.
    ///
    /// Serial by construction: the previous engine's parallel insert
    /// carried an entry-point race, and correctness comes before
    /// parallelism. A parallel builder can be added behind the same
    /// structural attestation once this path is proven.
    ///
    /// # Errors
    ///
    /// Propagates distance-computation failures from `store`.
    fn build<D: VectorDistance>(params: HnswParams, seed: u64, store: &D) -> SearchResult<Self> {
        let mut graph = Self::new(params, seed)?;
        let count = u32::try_from(store.len()).map_err(|_| SearchError::InvalidConfig {
            field: "store.len".to_owned(),
            value: store.len().to_string(),
            reason: "row count must fit in u32".to_owned(),
        })?;
        for id in 0..count {
            graph.insert(id, store)?;
        }
        Ok(graph)
    }

    /// Insert row `id`, which must be the next unindexed row.
    ///
    /// Failure-atomic: every existing neighbour list touched by linking or
    /// pruning is journaled before mutation. If the vector store returns an
    /// error at any distance-computation boundary, those lists, the entry
    /// point, the maximum level, and the point count are restored exactly.
    /// The same row can then be retried without rebuilding the graph.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] if `id` is not the next row or
    /// the graph is full, and propagates distance-computation failures.
    fn insert<D: VectorDistance>(&mut self, id: u32, store: &D) -> SearchResult<()> {
        // Borrow the reusable visited buffer out of `self` so the insert
        // body can hold `&mut self` and the buffer at the same time; it is
        // restored on every path, including the error path.
        let mut visited = std::mem::take(&mut self.scratch);
        let outcome = self.insert_with(id, store, &mut visited);
        self.scratch = visited;
        outcome
    }

    fn insert_with<D: VectorDistance>(
        &mut self,
        id: u32,
        store: &D,
        visited: &mut VisitedSet,
    ) -> SearchResult<()> {
        // A saturating conversion here would let the (4-billionth) insert
        // into a full graph look like a valid in-order one, so report the
        // capacity explicitly instead.
        let expected =
            u32::try_from(self.adjacency.len()).map_err(|_| SearchError::InvalidConfig {
                field: "points".to_owned(),
                value: self.adjacency.len().to_string(),
                reason: "graph is full: point ids must fit in u32".to_owned(),
            })?;
        if id != expected {
            return Err(SearchError::InvalidConfig {
                field: "id".to_owned(),
                value: id.to_string(),
                reason: format!("points must be inserted in row order; expected {expected}"),
            });
        }
        if id as usize >= store.len() {
            return Err(SearchError::InvalidConfig {
                field: "store.len".to_owned(),
                value: store.len().to_string(),
                reason: format!("store does not contain insertion row {id}"),
            });
        }

        let original_point_count = self.adjacency.len();
        let original_entry = self.entry;
        let original_max_level = self.max_level;
        let mut journal = MutationJournal::new(original_point_count);
        let level = self.sampler.level_for(id);
        self.adjacency.push(Adjacency::with_level(level));

        let outcome = (|| {
            // First point: it becomes the entry and has nothing to link to.
            let Some(mut current) = self.entry else {
                self.entry = Some(id);
                self.max_level = level;
                return Ok(());
            };

            let previous_max = self.max_level;

            // Phase 1 — greedy descent through the layers ABOVE this point's
            // own level, purely to find a good entry point.
            //
            // bd-u3wt defect (1): this descent must never link the new point.
            // It runs strictly above `level`, so no edge can be created in a
            // layer the new point does not occupy.
            let mut layer = previous_max;
            while layer > level {
                current = self.greedy_descend(current, id, layer, store)?;
                layer -= 1;
            }

            // Phase 2 — connect, starting at min(new level, previous max).
            //
            // bd-u3wt defect (2): starting above `previous_max` would link at
            // layers no other point occupies; starting below the new point's
            // level would leave its upper layers empty. The floor is exactly
            // this minimum.
            let mut entry_points = vec![current];
            for layer in (0..=level.min(previous_max)).rev() {
                let candidates = self.search_layer(
                    &entry_points,
                    id,
                    layer,
                    self.params.ef_construction,
                    store,
                    visited,
                )?;
                let selected =
                    Self::select_neighbours(id, &candidates, self.params.degree_at(layer), store)?;

                self.link(id, &selected, layer, store, &mut journal)?;

                entry_points = candidates.iter().map(|candidate| candidate.id).collect();
                if entry_points.is_empty() {
                    entry_points.push(current);
                }
            }

            // A point sampled above the previous maximum becomes the new
            // entry only after every fallible operation has succeeded.
            if level > previous_max {
                self.entry = Some(id);
                self.max_level = level;
            }
            Ok(())
        })();

        if let Err(error) = outcome {
            journal.rollback(&mut self.adjacency);
            self.adjacency.truncate(original_point_count);
            self.entry = original_entry;
            self.max_level = original_max_level;
            return Err(error);
        }
        Ok(())
    }

    /// Walk greedily to the nearest point to `target` at `layer`.
    fn greedy_descend<D: VectorDistance>(
        &self,
        start: u32,
        target: u32,
        layer: usize,
        store: &D,
    ) -> SearchResult<u32> {
        let mut current = start;
        let mut current_distance = store.distance_between(current, target)?;
        loop {
            let mut improved = false;
            for &neighbour in self.neighbours_at(current, layer) {
                let distance = store.distance_between(neighbour, target)?;
                if distance < current_distance {
                    current_distance = distance;
                    current = neighbour;
                    improved = true;
                }
            }
            if !improved {
                return Ok(current);
            }
        }
    }

    /// Beam search at `layer`, returning up to `ef` nearest candidates.
    ///
    /// `target_id` names a stored row; distances are row-to-row, which is
    /// what construction needs. [`Self::search_layer_query`] is the
    /// query-vector twin used at search time.
    fn search_layer<D: VectorDistance>(
        &self,
        entry_points: &[u32],
        target_id: u32,
        layer: usize,
        ef: usize,
        store: &D,
        visited: &mut VisitedSet,
    ) -> SearchResult<Vec<Candidate>> {
        self.beam_search(entry_points, layer, ef, store, visited, &mut |id, store| {
            store.distance_between(id, target_id)
        })
    }

    /// Beam search at `layer` against a raw query vector.
    fn search_layer_query<D: VectorDistance>(
        &self,
        entry_points: &[u32],
        query: &[f32],
        layer: usize,
        ef: usize,
        store: &D,
        visited: &mut VisitedSet,
    ) -> SearchResult<Vec<Candidate>> {
        self.beam_search(entry_points, layer, ef, store, visited, &mut |id, store| {
            store.distance_to_query(id, query)
        })
    }

    /// The shared beam search over a layer.
    ///
    /// Maintains a visited set, a min-heap frontier and a max-heap result
    /// set of size `ef`, expanding the nearest unexplored candidate until
    /// nothing closer than the current worst result remains.
    fn beam_search<D: VectorDistance>(
        &self,
        entry_points: &[u32],
        layer: usize,
        ef: usize,
        store: &D,
        visited: &mut VisitedSet,
        distance: &mut dyn FnMut(u32, &D) -> SearchResult<f32>,
    ) -> SearchResult<Vec<Candidate>> {
        let ef = ef.max(1);
        visited.begin(self.adjacency.len());
        let mut frontier: BinaryHeap<Nearest> = BinaryHeap::new();
        let mut results: BinaryHeap<Candidate> = BinaryHeap::new();

        for &entry in entry_points {
            if visited.visit(entry) {
                continue;
            }
            let candidate = Candidate {
                distance: distance(entry, store)?,
                id: entry,
            };
            frontier.push(Nearest(candidate));
            results.push(candidate);
        }
        while results.len() > ef {
            results.pop();
        }

        while let Some(Nearest(current)) = frontier.pop() {
            // Stop once the frontier's nearest is worse than the worst kept
            // result and the result set is already full.
            if let Some(worst) = results.peek()
                && results.len() >= ef
                && current.distance > worst.distance
            {
                break;
            }
            for &neighbour in self.neighbours_at(current.id, layer) {
                if visited.visit(neighbour) {
                    continue;
                }
                let candidate = Candidate {
                    distance: distance(neighbour, store)?,
                    id: neighbour,
                };
                let keep = results.len() < ef
                    || results
                        .peek()
                        .is_some_and(|worst| candidate.distance < worst.distance);
                if keep {
                    frontier.push(Nearest(candidate));
                    results.push(candidate);
                    while results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut ordered = results.into_vec();
        ordered.sort_unstable();
        Ok(ordered)
    }

    /// Choose which candidates to keep as neighbours.
    ///
    /// Uses the relative-neighbourhood heuristic: a candidate is kept only
    /// if it is closer to the new point than to any already-kept
    /// neighbour. This preserves long-range links that plain
    /// nearest-`m` selection prunes away, which is what keeps the graph
    /// navigable rather than clustered.
    fn select_neighbours<D: VectorDistance>(
        new_id: u32,
        candidates: &[Candidate],
        limit: usize,
        store: &D,
    ) -> SearchResult<Vec<u32>> {
        let mut selected: Vec<u32> = Vec::with_capacity(limit.min(candidates.len()));
        for candidate in candidates {
            if selected.len() >= limit {
                break;
            }
            if candidate.id == new_id {
                continue;
            }
            let mut dominated = false;
            for &kept in &selected {
                if store.distance_between(candidate.id, kept)? < candidate.distance {
                    dominated = true;
                    break;
                }
            }
            if !dominated {
                selected.push(candidate.id);
            }
        }
        // Top up to the budget with the nearest remaining candidates.
        //
        // Measured, not assumed: leaving the heuristic to under-fill dropped
        // recall@10 from 0.66 to 0.54 and average layer-0 degree from 32 to
        // 11.6 on the same corpus. The heuristic decides *preference* among
        // candidates; starving the neighbourhood costs more recall than the
        // clustering it avoids.
        if selected.len() < limit {
            for candidate in candidates {
                if selected.len() >= limit {
                    break;
                }
                if candidate.id != new_id && !selected.contains(&candidate.id) {
                    selected.push(candidate.id);
                }
            }
        }
        Ok(selected)
    }

    /// Install edges between `id` and `selected` at `layer`.
    ///
    /// bd-u3wt defect (3): the reciprocal edge is stored at `layer` — the
    /// layer being processed — never at the new point's top level. Both
    /// endpoints are checked for participation first, so an edge can never
    /// land in a layer either endpoint does not occupy.
    fn link<D: VectorDistance>(
        &mut self,
        id: u32,
        selected: &[u32],
        layer: usize,
        store: &D,
        journal: &mut MutationJournal,
    ) -> SearchResult<()> {
        for &neighbour in selected {
            if neighbour == id {
                continue;
            }
            let both_participate = self
                .adjacency
                .get(id as usize)
                .is_some_and(|point| point.participates_in(layer))
                && self
                    .adjacency
                    .get(neighbour as usize)
                    .is_some_and(|point| point.participates_in(layer));
            if !both_participate {
                continue;
            }

            journal.record(&self.adjacency, id, layer);
            if let Some(point) = self.adjacency.get_mut(id as usize)
                && let Some(list) = point.layers.get_mut(layer)
                && !list.contains(&neighbour)
            {
                list.push(neighbour);
            }
            journal.record(&self.adjacency, neighbour, layer);
            if let Some(point) = self.adjacency.get_mut(neighbour as usize)
                && let Some(list) = point.layers.get_mut(layer)
                && !list.contains(&id)
            {
                list.push(id);
            }

            // The reverse edge may have pushed the neighbour over budget.
            // `id` is protected: pruning the edge that was just created
            // would orphan the new point, since nothing else links to it
            // yet. Measured before this guard: 33 of 5000 points reachable.
            self.prune(neighbour, layer, Some(id), store, journal)?;
        }
        Ok(())
    }

    /// Trim a neighbourhood back to the layer's degree budget.
    fn prune<D: VectorDistance>(
        &mut self,
        id: u32,
        layer: usize,
        protected: Option<u32>,
        store: &D,
        journal: &mut MutationJournal,
    ) -> SearchResult<()> {
        let budget = self.params.degree_at(layer);
        let current = self.neighbours_at(id, layer).to_vec();
        if current.len() <= budget {
            return Ok(());
        }
        let mut scored: Vec<Candidate> = Vec::with_capacity(current.len());
        for neighbour in current {
            scored.push(Candidate {
                distance: store.distance_between(id, neighbour)?,
                id: neighbour,
            });
        }
        scored.sort_unstable();
        let mut kept = Self::select_neighbours(id, &scored, budget, store)?;
        // Reinstate the protected edge if the heuristic dropped it, evicting
        // the farthest kept neighbour to stay inside the budget.
        //
        // The eviction must find the genuinely farthest entry rather than
        // popping the last one: `select_neighbours` appends its top-up fill
        // after the heuristic's picks, so the list is not in distance order
        // and popping would usually evict a NEAR neighbour. `scored` is
        // sorted ascending, so the last kept entry it mentions is the
        // farthest.
        if let Some(protected) = protected
            && !kept.contains(&protected)
        {
            if let Some(farthest) = scored
                .iter()
                .rev()
                .find(|candidate| kept.contains(&candidate.id))
                .map(|candidate| candidate.id)
            {
                kept.retain(|&neighbour| neighbour != farthest);
            }
            kept.push(protected);
        }

        // Dropping an edge must drop BOTH directions. Trimming only this
        // point's list leaves the counterpart pointing back at a node that
        // no longer points to it, so traversal can enter a region it cannot
        // leave — the graph silently becomes directed and fragments. This is
        // the same asymmetry class bd-u3wt found in the previous engine
        // (measured here as 33 of 5000 points reachable before the fix).
        let dropped: Vec<u32> = scored
            .iter()
            .map(|candidate| candidate.id)
            .filter(|candidate| !kept.contains(candidate))
            .collect();
        journal.record(&self.adjacency, id, layer);
        if let Some(point) = self.adjacency.get_mut(id as usize)
            && let Some(list) = point.layers.get_mut(layer)
        {
            *list = kept;
        }
        for neighbour in dropped {
            journal.record(&self.adjacency, neighbour, layer);
            if let Some(point) = self.adjacency.get_mut(neighbour as usize)
                && let Some(list) = point.layers.get_mut(layer)
            {
                list.retain(|&back| back != id);
            }
        }
        Ok(())
    }

    /// Neighbours of `id` at `layer`, empty if it does not participate.
    fn neighbours_at(&self, id: u32, layer: usize) -> &[u32] {
        self.adjacency
            .get(id as usize)
            .map_or(&[], |point| point.neighbours(layer))
    }

    /// Search for the `k` nearest rows to `query`.
    ///
    /// Returns `(row id, distance)` pairs, nearest first. `ef` overrides the
    /// configured beam width; it is raised to at least `k` so the beam can
    /// hold the answer.
    ///
    /// # Errors
    ///
    /// Propagates distance-computation failures from `store`.
    fn search<D: VectorDistance>(
        &self,
        query: &[f32],
        k: usize,
        ef: Option<usize>,
        store: &D,
    ) -> SearchResult<Vec<(u32, f32)>> {
        self.verify_store_cardinality(store)
            .map_err(|source| SearchError::SubsystemError {
                subsystem: "native-hnsw",
                source: Box::new(source),
            })?;
        if k == 0 || self.adjacency.is_empty() {
            return Ok(Vec::new());
        }
        let Some(entry) = self.entry else {
            return Ok(Vec::new());
        };
        let ef = ef.unwrap_or(self.params.ef_search).max(k);
        // One buffer for the whole query rather than one per layer.
        let mut visited = VisitedSet::default();

        // Descend the upper layers greedily, then run one wide beam search
        // at layer 0 where every point participates.
        let mut current = entry;
        let mut layer = self.max_level;
        while layer > 0 {
            let found =
                self.search_layer_query(&[current], query, layer, 1, store, &mut visited)?;
            if let Some(best) = found.first() {
                current = best.id;
            }
            layer -= 1;
        }

        let mut candidates =
            self.search_layer_query(&[current], query, 0, ef, store, &mut visited)?;
        candidates.truncate(k);
        Ok(candidates
            .into_iter()
            .map(|candidate| (candidate.id, candidate.distance))
            .collect())
    }

    /// Verify the graph's structural invariants.
    ///
    /// This is the executable form of the bd-u3wt findings, suitable for
    /// use as a post-build gate: every defect that campaign found would be
    /// caught here.
    ///
    /// # Errors
    ///
    /// Returns the first [`GraphDefect`] found.
    fn verify(&self) -> Result<(), GraphDefect> {
        let count = self.adjacency.len();

        if count == 0 {
            if let Some(entry) = self.entry {
                return Err(GraphDefect::EntryPointInEmptyGraph { entry });
            }
            if self.max_level != 0 {
                return Err(GraphDefect::MaxLevelInEmptyGraph {
                    max_level: self.max_level,
                });
            }
            return Ok(());
        }

        let entry = self.entry.ok_or(GraphDefect::MissingEntryPoint)?;
        let entry_level = self
            .adjacency
            .get(entry as usize)
            .ok_or(GraphDefect::EntryPointUnknown { entry })?
            .level();
        if entry_level != self.max_level {
            return Err(GraphDefect::EntryPointBelowMaxLevel {
                entry,
                level: entry_level,
                max_level: self.max_level,
            });
        }

        for (id, point) in self.adjacency.iter().enumerate() {
            let id = u32::try_from(id).unwrap_or(u32::MAX);
            let expected_level = self.sampler.level_for(id);
            if point.level() != expected_level {
                return Err(GraphDefect::SampledLevelMismatch {
                    id,
                    expected: expected_level,
                    actual: point.level(),
                });
            }
            if point.level() > self.max_level {
                return Err(GraphDefect::LevelAboveMax {
                    id,
                    level: point.level(),
                    max_level: self.max_level,
                });
            }
            for (layer, neighbours) in point.layers.iter().enumerate() {
                let budget = self.params.degree_at(layer);
                if neighbours.len() > budget {
                    return Err(GraphDefect::DegreeExceeded {
                        id,
                        layer,
                        held: neighbours.len(),
                        budget,
                    });
                }
                let mut seen = neighbours.clone();
                seen.sort_unstable();
                let before = seen.len();
                seen.dedup();
                if seen.len() != before {
                    return Err(GraphDefect::DuplicateNeighbour { id, layer });
                }
                for &neighbour in neighbours {
                    if neighbour == id {
                        return Err(GraphDefect::SelfEdge { id, layer });
                    }
                    let target = self.adjacency.get(neighbour as usize).ok_or(
                        GraphDefect::NeighbourUnknown {
                            id,
                            layer,
                            neighbour,
                        },
                    )?;
                    // The invariant the previous engine violated: an edge
                    // may only exist in a layer BOTH endpoints occupy.
                    if !target.participates_in(layer) {
                        return Err(GraphDefect::EdgeAboveNeighbourLevel {
                            id,
                            layer,
                            neighbour,
                            neighbour_level: target.level(),
                        });
                    }
                    if !target.neighbours(layer).contains(&id) {
                        return Err(GraphDefect::MissingReciprocalEdge {
                            id,
                            layer,
                            neighbour,
                        });
                    }
                }
            }
        }

        // Every point must be reachable from the entry at layer 0, or a
        // query can never find it however wide the beam.
        let mut reached = vec![false; count];
        let mut stack = vec![entry];
        if let Some(slot) = reached.get_mut(entry as usize) {
            *slot = true;
        }
        let mut reached_count = 1usize;
        while let Some(current) = stack.pop() {
            for &neighbour in self.neighbours_at(current, 0) {
                let Some(slot) = reached.get_mut(neighbour as usize) else {
                    continue;
                };
                if !*slot {
                    *slot = true;
                    reached_count += 1;
                    stack.push(neighbour);
                }
            }
        }
        if reached_count != count {
            let first = reached.iter().position(|seen| !seen).unwrap_or(0);
            return Err(GraphDefect::Unreachable {
                reached: reached_count,
                total: count,
                first_unreachable: u32::try_from(first).unwrap_or(u32::MAX),
            });
        }
        Ok(())
    }

    /// Verify graph structure against the exact caller-owned vector store.
    ///
    /// # Errors
    ///
    /// Returns [`GraphDefect::StoreCardinalityMismatch`] before structural
    /// verification when graph ids and store rows do not describe the same
    /// universe; otherwise returns the first defect from [`Self::verify`].
    fn verify_for_store<D: VectorDistance>(&self, store: &D) -> Result<(), GraphDefect> {
        self.verify_store_cardinality(store)?;
        self.verify()
    }

    fn verify_store_cardinality<D: VectorDistance>(&self, store: &D) -> Result<(), GraphDefect> {
        let graph_points = self.adjacency.len();
        let store_rows = store.len();
        if graph_points != store_rows {
            return Err(GraphDefect::StoreCardinalityMismatch {
                graph_points,
                store_rows,
            });
        }
        Ok(())
    }
}

struct NativeHnswReceiptReader<'a> {
    bytes: &'a [u8],
    offset: usize,
    path: &'a Path,
}

impl<'a> NativeHnswReceiptReader<'a> {
    const fn new(bytes: &'a [u8], path: &'a Path) -> Self {
        Self {
            bytes,
            offset: 0,
            path,
        }
    }

    fn take(&mut self, len: usize, field: &str) -> SearchResult<&'a [u8]> {
        let end = self.offset.checked_add(len).ok_or_else(|| {
            native_hnsw_receipt_corrupted(
                self.path,
                format!("native HNSW receipt offset overflow while reading {field}"),
            )
        })?;
        let value = self.bytes.get(self.offset..end).ok_or_else(|| {
            native_hnsw_receipt_corrupted(
                self.path,
                format!("native HNSW receipt ended while reading {field}"),
            )
        })?;
        self.offset = end;
        Ok(value)
    }

    fn take_array<const N: usize>(&mut self, field: &str) -> SearchResult<[u8; N]> {
        let mut value = [0_u8; N];
        value.copy_from_slice(self.take(N, field)?);
        Ok(value)
    }

    fn read_u16(&mut self, field: &str) -> SearchResult<u16> {
        Ok(u16::from_be_bytes(self.take_array(field)?))
    }

    fn read_u32(&mut self, field: &str) -> SearchResult<u32> {
        Ok(u32::from_be_bytes(self.take_array(field)?))
    }

    fn read_u64(&mut self, field: &str) -> SearchResult<u64> {
        Ok(u64::from_be_bytes(self.take_array(field)?))
    }

    fn finish(self) -> SearchResult<()> {
        if self.offset != self.bytes.len() {
            return Err(native_hnsw_receipt_corrupted(
                self.path,
                format!(
                    "native HNSW generation receipt has {} trailing bytes",
                    self.bytes.len() - self.offset
                ),
            ));
        }
        Ok(())
    }
}

fn canonical_graph_basename(graph_path: &Path) -> SearchResult<String> {
    let basename = graph_path
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .ok_or_else(|| {
            native_hnsw_receipt_config_error(
                "graph_path",
                "redacted-invalid-basename",
                "must end in one canonical UTF-8 basename",
            )
        })?
        .to_owned();
    validate_graph_basename(&basename)?;
    Ok(basename)
}

fn validate_graph_basename(basename: &str) -> SearchResult<()> {
    if basename.is_empty() || basename.len() > NATIVE_HNSW_MAX_BASENAME_BYTES {
        return Err(native_hnsw_receipt_config_error(
            "graph_basename",
            "redacted-invalid-length",
            "must be non-empty and leave room for the adjacent receipt suffix",
        ));
    }
    if basename.chars().any(char::is_control) {
        return Err(native_hnsw_receipt_config_error(
            "graph_basename",
            "redacted-control-character",
            "must not contain control characters",
        ));
    }
    let mut components = Path::new(basename).components();
    if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
        return Err(native_hnsw_receipt_config_error(
            "graph_basename",
            "redacted-noncanonical",
            "must contain exactly one normal path component",
        ));
    }
    if Path::new(basename)
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        != Some("fshnsw")
    {
        return Err(native_hnsw_receipt_config_error(
            "graph_basename",
            basename,
            "must use the canonical .fshnsw extension",
        ));
    }
    Ok(())
}

fn validate_native_hnsw_graph_path(graph_path: &Path) -> SearchResult<()> {
    if graph_path.as_os_str().is_empty() {
        return Err(native_hnsw_receipt_config_error(
            "graph_path",
            "redacted-empty",
            "must not be empty",
        ));
    }
    if graph_path
        .components()
        .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
    {
        return Err(native_hnsw_receipt_config_error(
            "graph_path",
            "redacted-dot-relative",
            "must not contain '.' or '..' components",
        ));
    }
    let _ = canonical_graph_basename(graph_path)?;
    Ok(())
}

fn reject_symlink_ancestors(path: &Path) -> SearchResult<()> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    let mut current = PathBuf::new();
    for component in parent.components() {
        current.push(component.as_os_str());
        if matches!(component, Component::Prefix(_) | Component::RootDir) {
            continue;
        }
        match std::fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(SearchError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "native HNSW path ancestor '{}' must not be a symbolic link",
                        current.display()
                    ),
                )));
            }
            Ok(metadata) if metadata.file_type().is_dir() => {}
            Ok(_) => {
                return Err(SearchError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "native HNSW path ancestor '{}' must be a directory",
                        current.display()
                    ),
                )));
            }
            // ubs:ignore — this compares a public I/O error enum, not secret material.
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(SearchError::Io(error)),
        }
    }
    Ok(())
}

fn read_regular_file_bytes(path: &Path, kind: &str) -> SearchResult<Vec<u8>> {
    let path_metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        // ubs:ignore — this compares a public I/O error enum, not secret material.
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(SearchError::IndexNotFound {
                path: path.to_path_buf(),
            });
        }
        Err(error) => return Err(SearchError::Io(error)),
    };
    if !path_metadata.file_type().is_file() {
        return Err(SearchError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "{kind} '{}' must be a regular file, not a symlink or special file",
                path.display()
            ),
        )));
    }
    let mut file = File::open(path).map_err(SearchError::Io)?;
    let opened_metadata = file.metadata().map_err(SearchError::Io)?;
    ensure_same_open_file(&path_metadata, &opened_metadata).map_err(SearchError::Io)?;
    let expected_len = opened_metadata.len();
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(usize::try_from(expected_len).map_err(|_| {
            SearchError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{kind} '{}' is too large for this platform", path.display()),
            ))
        })?)
        .map_err(|error| {
            SearchError::Io(std::io::Error::other(format!(
                "could not allocate {expected_len} bytes for {kind} '{}': {error}",
                path.display()
            )))
        })?;
    file.read_to_end(&mut bytes).map_err(SearchError::Io)?;
    if u64::try_from(bytes.len()).ok() != Some(expected_len) {
        return Err(SearchError::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!(
                "{kind} '{}' changed length while being read",
                path.display()
            ),
        )));
    }
    let final_path_metadata = std::fs::symlink_metadata(path).map_err(SearchError::Io)?;
    ensure_same_open_file(&final_path_metadata, &opened_metadata).map_err(SearchError::Io)?;
    Ok(bytes)
}

fn persist_native_hnsw_receipt(path: &Path, bytes: &[u8]) -> SearchResult<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent).map_err(SearchError::Io)?;
    reject_non_regular_receipt_destination(path)?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent).map_err(SearchError::Io)?;
    temporary.write_all(bytes).map_err(SearchError::Io)?;
    temporary.as_file().sync_all().map_err(SearchError::Io)?;
    temporary.persist(path).map_err(|error| {
        SearchError::Io(std::io::Error::new(
            error.error.kind(),
            format!(
                "failed to atomically publish native HNSW generation receipt '{}': {}",
                path.display(),
                error.error
            ),
        ))
    })?;
    sync_parent_directory(path)
}

fn reject_non_regular_receipt_destination(path: &Path) -> SearchResult<()> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_file() => Ok(()),
        Ok(_) => Err(SearchError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "native HNSW generation receipt destination '{}' must be a regular file, not a \
                 symlink or special file",
                path.display()
            ),
        ))),
        // ubs:ignore — this compares a public I/O error enum, not secret material.
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(SearchError::Io(error)),
    }
}

fn append_canonical_bytes(destination: &mut Vec<u8>, value: &[u8]) -> SearchResult<()> {
    destination
        .extend_from_slice(&usize_to_u64(value.len(), "topology.domain_length")?.to_be_bytes());
    destination.extend_from_slice(value);
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let checksum = sha256_checksum(bytes);
    checksum
        .strip_prefix("sha256:")
        .unwrap_or(checksum.as_str())
        .to_owned()
}

fn validate_sha256_hex(field: &str, value: &str) -> SearchResult<()> {
    if value.len() == SHA256_HEX_LEN
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Ok(());
    }
    Err(native_hnsw_receipt_config_error(
        field,
        "redacted-invalid-sha256",
        "must be a lowercase 64-character SHA-256 digest",
    ))
}

fn decode_sha256_hex(field: &str, value: &str) -> SearchResult<[u8; SHA256_BYTES]> {
    validate_sha256_hex(field, value)?;
    let mut decoded = [0_u8; SHA256_BYTES];
    let (pairs, remainder) = value.as_bytes().as_chunks::<2>();
    if !remainder.is_empty() {
        return Err(native_hnsw_receipt_config_error(
            field,
            "redacted-invalid-sha256",
            "must be a lowercase 64-character SHA-256 digest",
        ));
    }
    for (index, pair) in pairs.iter().enumerate() {
        let (Some(high), Some(low)) = (
            decode_lower_hex_nibble(pair[0]),
            decode_lower_hex_nibble(pair[1]),
        ) else {
            return Err(native_hnsw_receipt_config_error(
                field,
                "redacted-invalid-sha256",
                "must be a lowercase 64-character SHA-256 digest",
            ));
        };
        decoded[index] = (high << 4) | low;
    }
    Ok(decoded)
}

const fn decode_lower_hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

fn encode_lower_hex(bytes: impl AsRef<[u8]>) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let bytes = bytes.as_ref();
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

fn native_hnsw_receipt_config_error(field: &str, value: &str, reason: &str) -> SearchError {
    let bounded_value = if value.len() <= 128 && !value.chars().any(char::is_control) {
        value.to_owned()
    } else {
        "redacted".to_owned()
    };
    SearchError::InvalidConfig {
        field: format!("native_hnsw_receipt.{field}"),
        value: bounded_value,
        reason: reason.to_owned(),
    }
}

fn native_hnsw_receipt_corrupted(path: &Path, detail: impl Into<String>) -> SearchError {
    SearchError::IndexCorrupted {
        path: path.to_path_buf(),
        detail: detail.into(),
    }
}

fn receipt_usize(value: u64, field: &str) -> SearchResult<usize> {
    usize::try_from(value).map_err(|_| {
        native_hnsw_receipt_config_error(
            field,
            &value.to_string(),
            "does not fit this platform's usize",
        )
    })
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn get_u32(bytes: &[u8], offset: usize) -> u32 {
    let mut value = [0_u8; 4];
    value.copy_from_slice(&bytes[offset..offset + 4]);
    u32::from_le_bytes(value)
}

fn get_u64(bytes: &[u8], offset: usize) -> u64 {
    let mut value = [0_u8; 8];
    value.copy_from_slice(&bytes[offset..offset + 8]);
    u64::from_le_bytes(value)
}

fn usize_to_u64(value: usize, field: &str) -> SearchResult<u64> {
    u64::try_from(value).map_err(|_| SearchError::InvalidConfig {
        field: format!("native_hnsw_{field}"),
        value: value.to_string(),
        reason: "value must fit in the persisted u64 field".to_owned(),
    })
}

fn persisted_usize(path: &Path, value: u64, field: &str) -> SearchResult<usize> {
    usize::try_from(value).map_err(|_| {
        native_hnsw_corrupted(
            path,
            format!("native HNSW {field} value {value} does not fit this platform"),
        )
    })
}

fn native_hnsw_corrupted(path: &Path, detail: impl Into<String>) -> SearchError {
    SearchError::IndexCorrupted {
        path: path.to_path_buf(),
        detail: detail.into(),
    }
}

fn write_payload_u32(
    file: &mut File,
    hasher: &mut crc32fast::Hasher,
    payload_len: &mut u64,
    value: u32,
) -> SearchResult<()> {
    let next_len = payload_len
        .checked_add(4)
        .ok_or_else(|| SearchError::InvalidConfig {
            field: "native_hnsw_payload_len".to_owned(),
            value: payload_len.to_string(),
            reason: "payload length overflow".to_owned(),
        })?;
    let bytes = value.to_le_bytes();
    file.write_all(&bytes).map_err(SearchError::Io)?;
    hasher.update(&bytes);
    *payload_len = next_len;
    Ok(())
}

fn reject_non_regular_destination(path: &Path) -> SearchResult<()> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_file() => Ok(()),
        Ok(_) => Err(SearchError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "native HNSW destination '{}' must be a regular file, not a symlink or special \
                 file",
                path.display()
            ),
        ))),
        // ubs:ignore — this compares a public I/O error enum, not secret material.
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(SearchError::Io(error)),
    }
}

fn open_regular_file(path: &Path) -> SearchResult<File> {
    let path_metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        // ubs:ignore — this compares a public I/O error enum, not secret material.
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(SearchError::IndexNotFound {
                path: path.to_path_buf(),
            });
        }
        Err(error) => return Err(SearchError::Io(error)),
    };
    if !path_metadata.file_type().is_file() {
        return Err(SearchError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "native HNSW artifact '{}' must be a regular file, not a symlink or special file",
                path.display()
            ),
        )));
    }
    let file = File::open(path).map_err(SearchError::Io)?;
    ensure_same_open_file(&path_metadata, &file.metadata().map_err(SearchError::Io)?)
        .map_err(SearchError::Io)?;
    Ok(file)
}

#[cfg(unix)]
fn ensure_same_open_file(
    path_metadata: &std::fs::Metadata,
    opened_metadata: &std::fs::Metadata,
) -> std::io::Result<()> {
    use std::os::unix::fs::MetadataExt;

    if path_metadata.dev() != opened_metadata.dev() || path_metadata.ino() != opened_metadata.ino()
    {
        return Err(std::io::Error::other(
            "native HNSW artifact changed while it was being opened",
        ));
    }
    Ok(())
}

#[cfg(not(unix))]
fn ensure_same_open_file(
    _: &std::fs::Metadata,
    opened_metadata: &std::fs::Metadata,
) -> std::io::Result<()> {
    if !opened_metadata.is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "native HNSW artifact must be a regular file",
        ));
    }
    Ok(())
}

fn sync_parent_directory(path: &Path) -> SearchResult<()> {
    #[cfg(unix)]
    {
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(SearchError::Io)?;
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FsviRecordFlags, FsviV2IdentityBinding, VectorIndex, fnv1a_hash};
    use ::tempfile as raw_tempfile;
    use frankensearch_core::generation::{EmbeddingIdentityBundleV1, QuantizationFormat};
    use std::cell::Cell;

    /// Test-only facade that preserves the production no-symlink-ancestor
    /// contract on hosts whose default temporary path contains an alias.
    ///
    /// macOS commonly returns `/var/folders/...` even though `/var` is a
    /// symlink to `/private/var`. Canonicalising the fixture root keeps every
    /// test artifact under the same directory while ensuring failures exercise
    /// the artifact under test rather than that platform-provided alias.
    mod tempfile {
        use std::io;
        use std::path::{Path, PathBuf};

        use super::raw_tempfile;

        pub(super) struct CanonicalTempDir {
            _directory: raw_tempfile::TempDir,
            canonical_path: PathBuf,
        }

        impl CanonicalTempDir {
            pub(super) fn path(&self) -> &Path {
                &self.canonical_path
            }
        }

        pub(super) fn tempdir() -> io::Result<CanonicalTempDir> {
            let directory = raw_tempfile::tempdir()?;
            let canonical_path = std::fs::canonicalize(directory.path())?;
            Ok(CanonicalTempDir {
                _directory: directory,
                canonical_path,
            })
        }
    }

    /// An in-memory store of unit-normalised vectors using cosine distance
    /// (`1 - dot`), which is the metric the two-tier index uses.
    struct TestStore {
        vectors: Vec<Vec<f32>>,
    }

    impl TestStore {
        fn new(vectors: Vec<Vec<f32>>) -> Self {
            let vectors = vectors
                .into_iter()
                .map(|vector| {
                    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
                    if norm <= f32::EPSILON {
                        vector
                    } else {
                        vector.into_iter().map(|v| v / norm).collect()
                    }
                })
                .collect();
            Self { vectors }
        }

        /// Deterministic pseudo-random corpus: seeded LCG, so a failure
        /// reproduces from the test name alone while the data still behaves
        /// like real embeddings.
        ///
        /// ★ Do NOT replace this with a closed-form lattice such as
        /// `(i * 31 + d * 17) % 101`. That looks like a reasonable
        /// deterministic corpus and is a trap: the modular structure makes
        /// vast numbers of points equidistant, which both depresses measured
        /// recall (ties make "the" exact top-k arbitrary) and genuinely
        /// fragments the graph — it measured 0.64 recall and 33-of-5000
        /// reachability on data that is pathological rather than
        /// representative. The same build scores recall 1.0 here.
        fn synthetic(count: usize, dim: usize) -> Self {
            let vectors = (0..count)
                .map(|i| {
                    let mut state = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0xdead_beef;
                    (0..dim)
                        .map(|_| {
                            state = state
                                .wrapping_mul(6_364_136_223_846_793_005)
                                .wrapping_add(1_442_695_040_888_963_407);
                            ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
                        })
                        .collect()
                })
                .collect();
            Self::new(vectors)
        }

        /// A deterministic pseudo-random query vector.
        fn query(seed: usize, dim: usize) -> Vec<f32> {
            let mut state = (seed as u64).wrapping_mul(0x2545_f491_4f6c_dd1d) ^ 0x00ab_cdef;
            (0..dim)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
                })
                .collect()
        }

        fn dot(a: &[f32], b: &[f32]) -> f32 {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        }

        /// Ground truth: exact nearest neighbours by full scan.
        fn exact_top_k(&self, query: &[f32], k: usize) -> Vec<u32> {
            let mut scored: Vec<(f32, u32)> = self
                .vectors
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let id = u32::try_from(i).expect("test corpus length fits u32");
                    (1.0 - Self::dot(v, query), id)
                })
                .collect();
            scored.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            scored.into_iter().take(k).map(|(_, id)| id).collect()
        }
    }

    impl VectorDistance for TestStore {
        fn distance_to_query(&self, id: u32, query: &[f32]) -> SearchResult<f32> {
            let vector =
                self.vectors
                    .get(id as usize)
                    .ok_or_else(|| SearchError::InvalidConfig {
                        field: "row".to_owned(),
                        value: id.to_string(),
                        reason: "row index out of range".to_owned(),
                    })?;
            Ok(1.0 - Self::dot(vector, query))
        }

        fn distance_between(&self, a: u32, b: u32) -> SearchResult<f32> {
            let left = self
                .vectors
                .get(a as usize)
                .ok_or_else(|| SearchError::InvalidConfig {
                    field: "row".to_owned(),
                    value: a.to_string(),
                    reason: "row index out of range".to_owned(),
                })?;
            self.distance_to_query(b, left)
        }

        fn len(&self) -> usize {
            self.vectors.len()
        }
    }

    struct FailingStore<'a> {
        inner: &'a TestStore,
        fail_between_at: Option<usize>,
        between_calls: Cell<usize>,
    }

    impl<'a> FailingStore<'a> {
        fn new(inner: &'a TestStore, fail_between_at: Option<usize>) -> Self {
            Self {
                inner,
                fail_between_at,
                between_calls: Cell::new(0),
            }
        }

        fn between_calls(&self) -> usize {
            self.between_calls.get()
        }
    }

    impl VectorDistance for FailingStore<'_> {
        fn distance_to_query(&self, id: u32, query: &[f32]) -> SearchResult<f32> {
            self.inner.distance_to_query(id, query)
        }

        fn distance_between(&self, a: u32, b: u32) -> SearchResult<f32> {
            let ordinal = self.between_calls.get() + 1;
            self.between_calls.set(ordinal);
            if self.fail_between_at == Some(ordinal) {
                return Err(SearchError::InvalidConfig {
                    field: "fault_injection".to_owned(),
                    value: ordinal.to_string(),
                    reason: "injected distance_between failure".to_owned(),
                });
            }
            self.inner.distance_between(a, b)
        }

        fn len(&self) -> usize {
            self.inner.len()
        }
    }

    fn topology_snapshot(graph: &NativeHnsw) -> (Vec<Vec<Vec<u32>>>, Option<u32>, usize, usize) {
        (
            graph
                .adjacency
                .iter()
                .map(|point| point.layers.clone())
                .collect(),
            graph.entry,
            graph.max_level,
            graph.len(),
        )
    }

    fn reseal_persisted_checksums(bytes: &mut [u8]) {
        let payload_crc32 = crc32fast::hash(&bytes[NATIVE_HNSW_HEADER_LEN..]);
        put_u32(bytes, 88, payload_crc32);
        let header_crc32 = crc32fast::hash(&bytes[..NATIVE_HNSW_HEADER_CRC_OFFSET]);
        put_u32(bytes, NATIVE_HNSW_HEADER_CRC_OFFSET, header_crc32);
    }

    fn first_persisted_neighbour(bytes: &[u8]) -> Option<(usize, u32)> {
        let point_count = usize::try_from(get_u64(bytes, 56)).expect("test point count");
        let mut offset = NATIVE_HNSW_HEADER_LEN;
        for point in 0..point_count {
            let layer_count =
                usize::try_from(get_u32(bytes, offset)).expect("test layer count fits usize");
            offset += 4;
            for _ in 0..layer_count {
                let neighbour_count = usize::try_from(get_u32(bytes, offset))
                    .expect("test neighbour count fits usize");
                offset += 4;
                if neighbour_count > 0 {
                    return Some((
                        offset,
                        u32::try_from(point).expect("test point id fits u32"),
                    ));
                }
                offset += neighbour_count * 4;
            }
        }
        None
    }

    fn params() -> HnswParams {
        HnswParams {
            m: 8,
            m0: 16,
            ef_construction: 64,
            ef_search: 64,
        }
    }

    fn fsvi_v2_binding(
        sequence: u64,
        nonce_byte: u8,
        model_id: &str,
        dimension: u32,
    ) -> FsviV2IdentityBinding {
        let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(model_id, dimension);
        identity.storage.format = "fsvi-v2".to_owned();
        identity.storage.quantization = QuantizationFormat::F16;
        identity.storage.endianness = "little-endian".to_owned();
        let generation = ArtifactGenerationIdentityV1::new(sequence, [nonce_byte; 16])
            .expect("valid test generation");
        FsviV2IdentityBinding::new(
            generation,
            identity.freeze().expect("valid frozen FSVI v2 identity"),
        )
        .expect("valid FSVI v2 identity binding")
    }

    fn fsvi_rows(count: usize, dimension: usize) -> Vec<(String, Vec<f32>)> {
        (0..count)
            .map(|row| {
                let mut vector = vec![0.0; dimension];
                vector[row % dimension] = 1.0;
                (format!("doc-{row:04}"), vector)
            })
            .collect()
    }

    fn admit_owned_fsvi_fixture(
        path: &Path,
        binding: &FsviV2IdentityBinding,
    ) -> Arc<ValidatedFsviBytes> {
        let bytes = std::fs::read(path).expect("read completed FSVI v2 owner fixture");
        Arc::new(
            ValidatedFsviBytes::from_arc(Arc::<[u8]>::from(bytes), binding)
                .expect("admit owned FSVI v2 fixture bytes"),
        )
    }

    fn admitted_fsvi_owner(
        directory: &Path,
        basename: &str,
        binding: &FsviV2IdentityBinding,
        rows: &[(String, Vec<f32>)],
    ) -> Arc<ValidatedFsviBytes> {
        let path = directory.join(basename);
        let mut writer =
            VectorIndex::create_v2(&path, binding.clone()).expect("create FSVI v2 owner fixture");
        for (doc_id, vector) in rows {
            writer
                .write_record(doc_id, vector)
                .expect("write FSVI v2 owner row");
        }
        writer.finish().expect("finish FSVI v2 owner fixture");
        admit_owned_fsvi_fixture(&path, binding)
    }

    fn admitted_fsvi_owner_with_flags(
        directory: &Path,
        basename: &str,
        binding: &FsviV2IdentityBinding,
        rows: &[(String, Vec<f32>, FsviRecordFlags)],
    ) -> Arc<ValidatedFsviBytes> {
        let path = directory.join(basename);
        let mut writer =
            VectorIndex::create_v2(&path, binding.clone()).expect("create FSVI v2 owner fixture");
        for (doc_id, vector, flags) in rows {
            if *flags == FsviRecordFlags::TOMBSTONE {
                writer
                    .write_tombstone_record(doc_id, vector)
                    .expect("write FSVI v2 tombstone row");
            } else {
                assert_eq!(*flags, FsviRecordFlags::LIVE);
                writer
                    .write_record(doc_id, vector)
                    .expect("write FSVI v2 live row");
            }
        }
        writer.finish().expect("finish FSVI v2 owner fixture");
        admit_owned_fsvi_fixture(&path, binding)
    }

    fn generation_owner(
        sequence: u64,
        nonce_byte: u8,
        model_id: &str,
        dimension: u32,
        physical_row_count: usize,
    ) -> Arc<ValidatedFsviBytes> {
        let directory = tempfile::tempdir().expect("temporary FSVI owner directory");
        let binding = fsvi_v2_binding(sequence, nonce_byte, model_id, dimension);
        let rows = fsvi_rows(
            physical_row_count,
            usize::try_from(dimension).expect("test dimension fits usize"),
        );
        admitted_fsvi_owner(directory.path(), "owner.fsvi", &binding, &rows)
    }

    fn write_resealed_receipt(path: &Path, receipt: &mut NativeHnswGenerationReceiptV2) {
        receipt.seal().expect("seal mutated test receipt");
        std::fs::write(
            path,
            receipt
                .encode_unchecked()
                .expect("encode mutated test receipt"),
        )
        .expect("write mutated test receipt");
    }

    fn sampled_entry_for_count(count: usize, seed: u64) -> Option<u32> {
        // ubs:ignore -- the public fixture cardinality is not secret material.
        if count == 0 {
            return None;
        }
        let sampler = LevelSampler::new(params().m, seed);
        let mut entry = 0_u32;
        let mut maximum = sampler.level_for(entry);
        for physical_index in 1..count {
            let id = u32::try_from(physical_index).expect("test owner count fits u32");
            let level = sampler.level_for(id);
            if level > maximum {
                entry = id;
                maximum = level;
            }
        }
        Some(entry)
    }

    fn seed_with_tombstoned_entry(owner: &ValidatedFsviBytes) -> u64 {
        (0..10_000_u64)
            .find(|&seed| {
                let entry = sampled_entry_for_count(owner.record_count(), seed)
                    .expect("nonempty test owner has an entry");
                owner
                    .row(usize::try_from(entry).expect("entry fits usize"))
                    .expect("entry row")
                    .flags()
                    .is_tombstone()
            })
            .expect("find deterministic seed whose entry point is tombstoned")
    }

    #[test]
    fn validated_handle_retains_build_owner_after_caller_and_path_scope_end() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ValidatedNativeHnsw>();

        // `generation_owner` returns only owned admitted bytes; its temporary
        // source directory has already left scope before this call returns.
        let owner = generation_owner(80, 0x80, "retained-build-owner", 4, 16);
        let weak_owner = Arc::downgrade(&owner);
        let bound = ValidatedNativeHnsw::build(Arc::clone(&owner), params(), 80)
            .expect("build retains the admitted owner");
        assert_eq!(Arc::strong_count(&owner), 2);

        drop(owner);
        assert_eq!(weak_owner.strong_count(), 1);
        let hits = bound
            .search(&[1.0, 0.0, 0.0, 0.0], 4, None)
            .expect("retained build owner remains searchable");
        assert_eq!(hits.len(), 4);
        assert!(hits.iter().all(|hit| hit.flags().is_live()));
        assert!(hits.iter().all(|hit| hit.doc_id().starts_with("doc-")));

        drop(hits);
        drop(bound);
        assert!(
            weak_owner.upgrade().is_none(),
            "the handle must be the final strong owner after the caller drops its Arc"
        );
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn owner_bound_ann_remains_on_admitted_bytes_after_pathname_replacement() {
        let directory = tempfile::tempdir().expect("temporary FSVI owner directory");
        let current_path = directory.path().join("current.fsvi");
        let replacement_path = directory.path().join("replacement.fsvi");
        let retained_path = directory.path().join("retained-original.fsvi");
        let binding = fsvi_v2_binding(82, 0x82, "ann-path-replacement", 4);

        let mut original =
            VectorIndex::create_v2(&current_path, binding.clone()).expect("create original FSVI");
        original
            .write_record("doc-alpha", &[1.0, 0.0, 0.0, 0.0])
            .expect("write original alpha");
        original
            .write_record("doc-beta", &[0.0, 1.0, 0.0, 0.0])
            .expect("write original beta");
        original.finish().expect("finish original FSVI");

        let owner = Arc::new(
            ValidatedFsviBytes::open_published(&current_path, &binding)
                .expect("admit original published FSVI"),
        );
        let bound = ValidatedNativeHnsw::build(Arc::clone(&owner), params(), 82)
            .expect("build graph from admitted owner");
        assert_eq!(
            bound
                .search(&[1.0, 0.0, 0.0, 0.0], 1, Some(owner.record_count()))
                .expect("search original owner through ANN")[0]
                .doc_id(),
            "doc-alpha"
        );

        let mut replacement = VectorIndex::create_v2(&replacement_path, binding.clone())
            .expect("create replacement FSVI");
        replacement
            .write_record("doc-alpha", &[0.0, 1.0, 0.0, 0.0])
            .expect("write replacement alpha");
        replacement
            .write_record("doc-beta", &[1.0, 0.0, 0.0, 0.0])
            .expect("write replacement beta");
        replacement.finish().expect("finish replacement FSVI");
        std::fs::rename(&current_path, &retained_path).expect("retain original pathname target");
        std::fs::rename(&replacement_path, &current_path)
            .expect("publish semantically different replacement");

        let fresh_owner = ValidatedFsviBytes::open_published(&current_path, &binding)
            .expect("fresh admission observes replacement bytes");
        assert_eq!(
            fresh_owner
                .search_top_k(&[1.0, 0.0, 0.0, 0.0], 1, None)
                .expect("exact search replacement owner")[0]
                .doc_id,
            "doc-beta",
            "the substituted pathname must expose replacement semantics to fresh admission"
        );
        assert_eq!(
            bound
                .search(&[1.0, 0.0, 0.0, 0.0], 1, Some(owner.record_count()))
                .expect("search retained owner-bound ANN")[0]
                .doc_id(),
            "doc-alpha",
            "the owner-built graph must resolve through its retained admitted bytes"
        );
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn owner_bound_ann_survives_same_size_in_place_path_rewrite() {
        use std::os::unix::fs::MetadataExt;

        let directory = tempfile::tempdir().expect("temporary FSVI owner directory");
        let current_path = directory.path().join("current.fsvi");
        let replacement_path = directory.path().join("replacement.fsvi");
        let binding = fsvi_v2_binding(83, 0x83, "ann-in-place-rewrite", 4);

        let mut original =
            VectorIndex::create_v2(&current_path, binding.clone()).expect("create original FSVI");
        original
            .write_record("doc-alpha", &[1.0, 0.0, 0.0, 0.0])
            .expect("write original alpha");
        original
            .write_record("doc-beta", &[0.0, 1.0, 0.0, 0.0])
            .expect("write original beta");
        original.finish().expect("finish original FSVI");

        let owner = Arc::new(
            ValidatedFsviBytes::open_published(&current_path, &binding)
                .expect("admit original published FSVI"),
        );
        let bound = ValidatedNativeHnsw::build(Arc::clone(&owner), params(), 83)
            .expect("build graph from admitted owner");
        assert_eq!(
            bound
                .search(&[1.0, 0.0, 0.0, 0.0], 1, Some(owner.record_count()))
                .expect("search original owner through ANN")[0]
                .doc_id(),
            "doc-alpha"
        );

        let mut replacement = VectorIndex::create_v2(&replacement_path, binding.clone())
            .expect("create replacement FSVI");
        replacement
            .write_record("doc-alpha", &[0.0, 1.0, 0.0, 0.0])
            .expect("write replacement alpha");
        replacement
            .write_record("doc-beta", &[1.0, 0.0, 0.0, 0.0])
            .expect("write replacement beta");
        replacement.finish().expect("finish replacement FSVI");
        let replacement_bytes = std::fs::read(&replacement_path).expect("read replacement FSVI");
        let original_metadata =
            std::fs::symlink_metadata(&current_path).expect("read original metadata");
        assert_eq!(
            usize::try_from(original_metadata.len()).expect("original length fits usize"),
            replacement_bytes.len(),
            "fixture must preserve pathname length for the in-place rewrite"
        );
        std::fs::write(&current_path, replacement_bytes)
            .expect("rewrite admitted pathname in place");
        assert_eq!(
            std::fs::symlink_metadata(&current_path)
                .expect("read rewritten metadata")
                .ino(),
            original_metadata.ino(),
            "fixture must retain the original inode rather than rename a replacement"
        );

        let fresh_owner = ValidatedFsviBytes::open_published(&current_path, &binding)
            .expect("fresh admission observes rewritten bytes");
        assert_eq!(
            fresh_owner
                .search_top_k(&[1.0, 0.0, 0.0, 0.0], 1, None)
                .expect("exact search rewritten owner")[0]
                .doc_id,
            "doc-beta",
            "fresh admission must observe the rewritten pathname semantics"
        );
        assert_eq!(
            bound
                .search(&[1.0, 0.0, 0.0, 0.0], 1, Some(owner.record_count()))
                .expect("search retained owner-bound ANN")[0]
                .doc_id(),
            "doc-alpha",
            "the owner-built graph must continue to resolve through admitted bytes"
        );
    }

    #[test]
    fn owner_bound_ann_full_beam_matches_exact_owner_ranking() {
        let directory = tempfile::tempdir().expect("temporary FSVI owner directory");
        let binding = fsvi_v2_binding(84, 0x84, "ann-exact-owner-parity", 4);
        let rows = [
            (
                "dead-best".to_owned(),
                vec![1.0, 0.0, 0.0, 0.0],
                FsviRecordFlags::TOMBSTONE,
            ),
            (
                "live-alpha".to_owned(),
                vec![0.9, 0.1, 0.0, 0.0],
                FsviRecordFlags::LIVE,
            ),
            (
                "live-beta".to_owned(),
                vec![0.6, 0.4, 0.0, 0.0],
                FsviRecordFlags::LIVE,
            ),
            (
                "live-gamma".to_owned(),
                vec![0.1, 0.9, 0.0, 0.0],
                FsviRecordFlags::LIVE,
            ),
        ];
        let owner =
            admitted_fsvi_owner_with_flags(directory.path(), "owner-parity.fsvi", &binding, &rows);
        let query = [1.0, 0.0, 0.0, 0.0];
        let exact = owner
            .search_top_k(&query, owner.live_count(), None)
            .expect("exact same-owner search");
        let bound = ValidatedNativeHnsw::build(Arc::clone(&owner), params(), 84)
            .expect("build graph from exact-search owner");
        let ann = bound
            .search(&query, owner.live_count(), Some(owner.record_count()))
            .expect("full-beam owner-bound ANN search");

        assert_eq!(
            ann.iter()
                .map(ValidatedNativeHnswHit::doc_id)
                .collect::<Vec<_>>(),
            exact
                .iter()
                .map(|hit| hit.doc_id.as_str())
                .collect::<Vec<_>>(),
            "full-beam ANN must preserve the exact ranking from its retained owner"
        );
        assert_eq!(
            ann.iter()
                .map(ValidatedNativeHnswHit::physical_row)
                .collect::<Vec<_>>(),
            exact.iter().map(|hit| hit.index).collect::<Vec<_>>(),
            "ANN hits must resolve the exact physical rows owned by the same image"
        );
        assert!(ann.iter().all(|hit| hit.flags().is_live()));
        assert!(ann.iter().all(|hit| hit.doc_id() != "dead-best"));
    }

    #[test]
    fn validated_handle_retains_loaded_owner_after_caller_scope_end() {
        let owner = generation_owner(81, 0x81, "retained-load-owner", 4, 16);
        let weak_owner = Arc::downgrade(&owner);
        let loaded = {
            let directory = tempfile::tempdir().expect("temporary graph directory");
            let graph_path = directory.path().join("retained-owner.fshnsw");
            let built = ValidatedNativeHnsw::build(Arc::clone(&owner), params(), 81)
                .expect("build owner-bound graph");
            built.save(&graph_path).expect("save owner-bound graph");
            drop(built);
            assert_eq!(Arc::strong_count(&owner), 1);

            let (loaded, _) = ValidatedNativeHnsw::load(Arc::clone(&owner), &graph_path)
                .expect("load retains the exact admitted owner");
            loaded
        };
        // Both the original FSVI source and the persisted graph path are gone;
        // only the loaded graph and its retained owner allocation remain.
        assert_eq!(Arc::strong_count(&owner), 2);

        drop(owner);
        assert_eq!(weak_owner.strong_count(), 1);
        let hits = loaded
            .search(&[0.0, 1.0, 0.0, 0.0], 4, None)
            .expect("retained loaded owner remains searchable");
        assert_eq!(hits.len(), 4);
        assert!(hits.iter().all(|hit| hit.flags().is_live()));

        drop(hits);
        drop(loaded);
        assert!(weak_owner.upgrade().is_none());
    }

    // ─── FSVI generation receipts ──────────────────────────────────────

    #[test]
    fn generation_receipt_round_trip_binds_every_identity_layer_and_is_deterministic() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let graph_path = directory.path().join("graph.fshnsw");
        let receipt_path = directory.path().join("graph.fshnsw.receipt");
        let owner = generation_owner(u64::MAX, 0xa5, "receipt-round-trip", 8, 48);
        let bound = ValidatedNativeHnsw::build(Arc::clone(&owner), params(), 0x51de_cafe)
            .expect("owner-bound build");
        let binding = &bound.binding;

        assert_eq!(
            native_hnsw_generation_receipt_path(&graph_path).expect("canonical receipt path"),
            receipt_path
        );
        let receipt = bound.save(&graph_path).expect("save owner-bound graph");
        receipt.validate().expect("receipt validates");
        let persisted = std::fs::read(&receipt_path).expect("receipt bytes");
        let decoded =
            NativeHnswGenerationReceiptV2::from_bytes(&persisted, &receipt_path).expect("decode");
        assert_eq!(decoded, receipt);
        assert_eq!(decoded.artifact_generation, binding.artifact_generation);
        assert_eq!(
            decoded.embedding_identity_fingerprint,
            binding.embedding_identity_fingerprint
        );
        assert_eq!(
            decoded.graph_byte_len,
            std::fs::metadata(&graph_path)
                .expect("graph metadata")
                .len()
        );
        assert_eq!(
            decoded.graph_sha256,
            sha256_hex(&std::fs::read(&graph_path).unwrap())
        );
        assert_eq!(decoded.native_format_version, NATIVE_HNSW_FORMAT_VERSION);
        assert_eq!(
            decoded.params,
            NativeHnswParamsIdentityV1::from_params(params()).unwrap()
        );
        assert_eq!(decoded.point_count, 48);
        assert_eq!(decoded.fsvi_physical_row_count, 48);
        assert_eq!(decoded.vector_content_digest, binding.vector_content_digest);
        assert_eq!(
            decoded.ordered_live_docset_digest,
            binding.ordered_live_docset_digest
        );
        assert_eq!(
            decoded.fsvi_whole_image_sha256,
            binding.fsvi_whole_image_sha256
        );
        assert_eq!(
            decoded.fsvi_whole_image_sha256,
            encode_lower_hex(owner.witness().whole_image_sha256)
        );
        assert_eq!(
            decoded.topology_sha256,
            bound.graph.topology_sha256().unwrap()
        );

        let (loaded, observed) =
            ValidatedNativeHnsw::load(Arc::clone(&owner), &graph_path).expect("owner-bound load");
        assert_eq!(observed, receipt);
        assert_eq!(
            topology_snapshot(&loaded.graph),
            topology_snapshot(&bound.graph)
        );
        let hits = loaded
            .search(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 8, None)
            .expect("owner-bound search");
        assert!(!hits.is_empty());
        for hit in hits {
            let physical_index =
                usize::try_from(hit.physical_row()).expect("physical row fits usize");
            let owner_row = owner.row(physical_index).expect("same-owner row");
            assert_eq!(hit.doc_id(), owner_row.doc_id());
            assert_eq!(hit.flags(), owner_row.flags());
        }

        bound.save(&graph_path).expect("repeat owner-bound save");
        assert_eq!(
            std::fs::read(&receipt_path).expect("repeat receipt"),
            persisted,
            "same graph/generation/identity must have one stable receipt encoding"
        );

        let mut serde_value = serde_json::to_value(&receipt).expect("serialize receipt");
        serde_value["future_field"] = serde_json::json!(true);
        assert!(
            serde_json::from_value::<NativeHnswGenerationReceiptV2>(serde_value).is_err(),
            "the public receipt schema must reject unknown fields"
        );
    }

    #[test]
    fn generation_binding_is_derived_from_the_admitted_fsvi_owner() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let fsvi_binding = fsvi_v2_binding(7, 7, "admitted-owner", 8);
        let rows = fsvi_rows(9, 8);
        let owner = admitted_fsvi_owner(directory.path(), "owner.fsvi", &fsvi_binding, &rows);
        let binding = NativeHnswGenerationBindingV2::from_validated_fsvi(&owner)
            .expect("derive binding from admitted owner");
        let identity = owner.identity_v2();

        assert_eq!(binding.artifact_generation, identity.generation);
        assert_eq!(
            binding.embedding_identity_fingerprint,
            encode_lower_hex(identity.identity_bundle_fingerprint)
        );
        assert_eq!(
            binding.vector_content_digest,
            encode_lower_hex(identity.vector_content_digest)
        );
        assert_eq!(
            binding.ordered_live_docset_digest,
            encode_lower_hex(identity.ordered_live_docset_digest)
        );
        assert_eq!(
            binding.fsvi_whole_image_sha256,
            encode_lower_hex(owner.witness().whole_image_sha256)
        );
        assert_eq!(binding.fsvi_physical_row_count, 9);
    }

    #[test]
    fn bound_load_rejects_same_identity_and_count_with_different_vectors_before_graph_read() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let fsvi_binding = fsvi_v2_binding(31, 0x31, "vector-content-binding", 4);
        let rows_a = vec![
            ("doc-alpha".to_owned(), vec![1.0, 0.0, 0.0, 0.0]),
            ("doc-beta".to_owned(), vec![0.0, 1.0, 0.0, 0.0]),
            ("doc-gamma".to_owned(), vec![0.0, 0.0, 1.0, 0.0]),
        ];
        let rows_b = vec![
            ("doc-alpha".to_owned(), vec![0.0, 1.0, 0.0, 0.0]),
            ("doc-beta".to_owned(), vec![1.0, 0.0, 0.0, 0.0]),
            ("doc-gamma".to_owned(), vec![0.0, 0.0, 0.0, 1.0]),
        ];
        let owner_a =
            admitted_fsvi_owner(directory.path(), "vectors-a.fsvi", &fsvi_binding, &rows_a);
        let owner_b =
            admitted_fsvi_owner(directory.path(), "vectors-b.fsvi", &fsvi_binding, &rows_b);
        let bound_a = ValidatedNativeHnsw::build(Arc::clone(&owner_a), params(), 31)
            .expect("owner-bound graph");
        let binding_a = &bound_a.binding;
        let binding_b =
            NativeHnswGenerationBindingV2::from_validated_fsvi(&owner_b).expect("binding b");

        assert_eq!(binding_a.artifact_generation, binding_b.artifact_generation);
        assert_eq!(
            binding_a.embedding_identity_fingerprint,
            binding_b.embedding_identity_fingerprint
        );
        assert_eq!(
            binding_a.ordered_live_docset_digest,
            binding_b.ordered_live_docset_digest
        );
        assert_eq!(
            binding_a.fsvi_physical_row_count,
            binding_b.fsvi_physical_row_count
        );
        assert_ne!(
            binding_a.vector_content_digest,
            binding_b.vector_content_digest
        );

        let graph_path = directory.path().join("vectors.fshnsw");
        bound_a.save(&graph_path).expect("owner-bound save");
        std::fs::write(&graph_path, b"graph bytes must not be read")
            .expect("poison graph after receipt publication");

        let error = ValidatedNativeHnsw::load(Arc::clone(&owner_b), &graph_path)
            .expect_err("different vector content must fail");
        assert!(
            matches!(error, SearchError::IndexCorrupted { ref detail, .. }
                if detail.contains("receipt vector content digest mismatch")),
            "vector mismatch did not fail at receipt identity: {error:?}"
        );
    }

    #[test]
    fn bound_load_rejects_same_vectors_and_identity_with_different_ordered_docset_before_graph_read()
     {
        let directory = tempfile::tempdir().expect("temporary directory");
        let fsvi_binding = fsvi_v2_binding(32, 0x32, "ordered-docset-binding", 4);
        let shared_vector = vec![1.0, 0.0, 0.0, 0.0];
        let rows_a = vec![
            ("doc-alpha".to_owned(), shared_vector.clone()),
            ("doc-beta".to_owned(), shared_vector.clone()),
            ("doc-gamma".to_owned(), shared_vector.clone()),
        ];
        let rows_b = vec![
            ("doc-alpha".to_owned(), shared_vector.clone()),
            ("doc-delta".to_owned(), shared_vector.clone()),
            ("doc-gamma".to_owned(), shared_vector),
        ];
        let owner_a =
            admitted_fsvi_owner(directory.path(), "docset-a.fsvi", &fsvi_binding, &rows_a);
        let owner_b =
            admitted_fsvi_owner(directory.path(), "docset-b.fsvi", &fsvi_binding, &rows_b);
        let bound_a = ValidatedNativeHnsw::build(Arc::clone(&owner_a), params(), 32)
            .expect("owner-bound graph");
        let binding_a = &bound_a.binding;
        let binding_b =
            NativeHnswGenerationBindingV2::from_validated_fsvi(&owner_b).expect("binding b");

        assert_eq!(binding_a.artifact_generation, binding_b.artifact_generation);
        assert_eq!(
            binding_a.embedding_identity_fingerprint,
            binding_b.embedding_identity_fingerprint
        );
        assert_eq!(
            binding_a.vector_content_digest,
            binding_b.vector_content_digest
        );
        assert_eq!(
            binding_a.fsvi_physical_row_count,
            binding_b.fsvi_physical_row_count
        );
        assert_ne!(
            binding_a.ordered_live_docset_digest,
            binding_b.ordered_live_docset_digest
        );

        let graph_path = directory.path().join("docset.fshnsw");
        bound_a.save(&graph_path).expect("owner-bound save");
        std::fs::write(&graph_path, b"graph bytes must not be read")
            .expect("poison graph after receipt publication");

        let error = ValidatedNativeHnsw::load(Arc::clone(&owner_b), &graph_path)
            .expect_err("different ordered docset must fail");
        assert!(
            matches!(error, SearchError::IndexCorrupted { ref detail, .. }
                if detail.contains("receipt ordered live-docset digest mismatch")),
            "docset mismatch did not fail at receipt identity: {error:?}"
        );
    }

    #[test]
    fn bound_load_rejects_same_vectors_and_live_docset_with_different_tombstone_layout() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let fsvi_binding = fsvi_v2_binding(35, 0x35, "tombstone-layout-binding", 4);
        let live_ids = ["live-alpha", "live-beta"];
        let smallest_live_hash = live_ids
            .iter()
            .map(|doc_id| fnv1a_hash(doc_id.as_bytes()))
            .min()
            .expect("live id");
        let largest_live_hash = live_ids
            .iter()
            .map(|doc_id| fnv1a_hash(doc_id.as_bytes()))
            .max()
            .expect("live id");
        let tombstone_before = (0..100_000_u32)
            .map(|candidate| format!("dead-before-{candidate}"))
            .find(|doc_id| fnv1a_hash(doc_id.as_bytes()) < smallest_live_hash)
            .expect("find tombstone id sorting before the live rows");
        let tombstone_after = (0..100_000_u32)
            .map(|candidate| format!("dead-after-{candidate}"))
            .find(|doc_id| fnv1a_hash(doc_id.as_bytes()) > largest_live_hash)
            .expect("find tombstone id sorting after the live rows");
        let shared_vector = vec![1.0, 0.0, 0.0, 0.0];
        let rows_a = vec![
            (
                live_ids[0].to_owned(),
                shared_vector.clone(),
                FsviRecordFlags::LIVE,
            ),
            (
                live_ids[1].to_owned(),
                shared_vector.clone(),
                FsviRecordFlags::LIVE,
            ),
            (
                tombstone_before,
                shared_vector.clone(),
                FsviRecordFlags::TOMBSTONE,
            ),
        ];
        let rows_b = vec![
            (
                live_ids[0].to_owned(),
                shared_vector.clone(),
                FsviRecordFlags::LIVE,
            ),
            (
                live_ids[1].to_owned(),
                shared_vector.clone(),
                FsviRecordFlags::LIVE,
            ),
            (tombstone_after, shared_vector, FsviRecordFlags::TOMBSTONE),
        ];
        let owner_a = admitted_fsvi_owner_with_flags(
            directory.path(),
            "layout-a.fsvi",
            &fsvi_binding,
            &rows_a,
        );
        let owner_b = admitted_fsvi_owner_with_flags(
            directory.path(),
            "layout-b.fsvi",
            &fsvi_binding,
            &rows_b,
        );
        let binding_a =
            NativeHnswGenerationBindingV2::from_validated_fsvi(&owner_a).expect("binding a");
        let binding_b =
            NativeHnswGenerationBindingV2::from_validated_fsvi(&owner_b).expect("binding b");

        assert_eq!(binding_a.artifact_generation, binding_b.artifact_generation);
        assert_eq!(
            binding_a.embedding_identity_fingerprint,
            binding_b.embedding_identity_fingerprint
        );
        assert_eq!(
            binding_a.vector_content_digest,
            binding_b.vector_content_digest
        );
        assert_eq!(
            binding_a.ordered_live_docset_digest,
            binding_b.ordered_live_docset_digest
        );
        assert_eq!(
            binding_a.fsvi_physical_row_count,
            binding_b.fsvi_physical_row_count
        );
        assert_ne!(
            binding_a.fsvi_whole_image_sha256,
            binding_b.fsvi_whole_image_sha256
        );
        let layout_a: Vec<(String, FsviRecordFlags)> = (0..owner_a.record_count())
            .map(|index| {
                let row = owner_a.row(index).expect("owner-a row");
                (row.doc_id().to_owned(), row.flags())
            })
            .collect();
        let layout_b: Vec<(String, FsviRecordFlags)> = (0..owner_b.record_count())
            .map(|index| {
                let row = owner_b.row(index).expect("owner-b row");
                (row.doc_id().to_owned(), row.flags())
            })
            .collect();
        assert_eq!(layout_a[0].1, FsviRecordFlags::TOMBSTONE);
        assert_eq!(layout_b[layout_b.len() - 1].1, FsviRecordFlags::TOMBSTONE);
        assert_ne!(layout_a, layout_b);

        let bound_a = ValidatedNativeHnsw::build(Arc::clone(&owner_a), params(), 35)
            .expect("public tombstone-aware graph fixture");
        let graph_path = directory.path().join("layout.fshnsw");
        bound_a
            .save(&graph_path)
            .expect("public receipted graph fixture");
        std::fs::write(&graph_path, b"graph bytes must not be read")
            .expect("poison graph after receipt publication");

        let error = ValidatedNativeHnsw::load(Arc::clone(&owner_b), &graph_path)
            .expect_err("different physical tombstone layout must fail");
        assert!(
            matches!(error, SearchError::IndexCorrupted { ref detail, .. }
                if detail.contains("receipt FSVI whole-image SHA-256 mismatch")),
            "tombstone-layout mismatch did not fail at receipt identity: {error:?}"
        );
    }

    #[test]
    fn validated_owner_bound_api_routes_through_tombstones_and_round_trips() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let fsvi_binding = fsvi_v2_binding(36, 0x36, "tombstone-admission", 4);
        let rows = vec![
            (
                "live".to_owned(),
                vec![1.0, 0.0, 0.0, 0.0],
                FsviRecordFlags::LIVE,
            ),
            (
                "dead".to_owned(),
                vec![0.0, 1.0, 0.0, 0.0],
                FsviRecordFlags::TOMBSTONE,
            ),
        ];
        let owner = admitted_fsvi_owner_with_flags(
            directory.path(),
            "tombstoned.fsvi",
            &fsvi_binding,
            &rows,
        );

        let bound = ValidatedNativeHnsw::build(Arc::clone(&owner), params(), 36)
            .expect("tombstoned owner builds a public ANN handle");
        assert_eq!(bound.len(), 2, "both physical rows remain routing nodes");
        assert_eq!(owner.live_count(), 1);
        assert_eq!(owner.tombstone_count(), 1);

        let tombstone_index = (0..owner.record_count())
            .find(|&index| {
                owner
                    .row(index)
                    .expect("admitted owner row")
                    .flags()
                    .is_tombstone()
            })
            .expect("fixture tombstone");
        // ubs:ignore -- a public test query must not taint unrelated comparisons.
        let query = owner
            .vector_at_f32(tombstone_index)
            .expect("query equal to the nearest tombstoned vector");
        let hits = bound
            .search(&query, 8, Some(1))
            .expect("bounded search widens past its tombstoned nearest candidate");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id(), "live");
        assert!(hits[0].flags().is_live());
        assert_ne!(
            usize::try_from(hits[0].physical_row()).expect("physical row fits usize"),
            tombstone_index
        );

        let graph_path = directory.path().join("tombstoned.fshnsw");
        // ubs:ignore -- a public integrity receipt is not authentication material.
        let receipt = bound
            .save(&graph_path)
            .expect("save tombstone-aware receipt");
        assert_eq!(receipt.fsvi_physical_row_count, 2);
        assert_eq!(
            receipt.fsvi_whole_image_sha256,
            encode_lower_hex(owner.witness().whole_image_sha256)
        );

        let (loaded, loaded_receipt) = ValidatedNativeHnsw::load(Arc::clone(&owner), &graph_path)
            .expect("load tombstone-aware graph");
        assert_eq!(loaded_receipt, receipt);
        assert_eq!(
            topology_snapshot(&loaded.graph),
            topology_snapshot(&bound.graph)
        );
        assert_eq!(
            loaded
                .search(&query, 8, Some(1))
                .expect("loaded bounded widening"),
            hits
        );
    }

    #[test]
    fn ninety_percent_tombstones_widen_to_exact_live_cardinality_in_stable_order() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let fsvi_binding = fsvi_v2_binding(40, 0x40, "ninety-percent-tombstones", 8);
        let corpus = TestStore::synthetic(20, 8);
        let rows: Vec<(String, Vec<f32>, FsviRecordFlags)> = corpus
            .vectors
            .into_iter()
            .enumerate()
            .map(|(row, vector)| {
                (
                    format!("ninety-percent-{row:02}"),
                    vector,
                    if row < 2 {
                        FsviRecordFlags::LIVE
                    } else {
                        FsviRecordFlags::TOMBSTONE
                    },
                )
            })
            .collect();
        let owner = admitted_fsvi_owner_with_flags(
            directory.path(),
            "ninety-percent.fsvi",
            &fsvi_binding,
            &rows,
        );
        assert_eq!(owner.record_count(), 20);
        assert_eq!(owner.live_count(), 2);
        assert_eq!(owner.tombstone_count(), 18);

        let (seed, bound) = (0..10_000_u64)
            .find_map(|seed| {
                let entry = sampled_entry_for_count(owner.record_count(), seed)?;
                if owner
                    .row(usize::try_from(entry).ok()?)
                    .ok()?
                    .flags()
                    .is_live()
                {
                    return None;
                }
                ValidatedNativeHnsw::build(Arc::clone(&owner), params(), seed)
                    .ok()
                    .map(|bound| (seed, bound))
            })
            .expect("find a sound graph with a tombstoned entry point");
        let entry = bound.graph.entry_point().expect("nonempty graph entry");
        assert!(
            owner
                .row(usize::try_from(entry).expect("entry fits usize"))
                .expect("entry row")
                .flags()
                .is_tombstone(),
            "the graph entry itself must exercise tombstone routing"
        );

        // ubs:ignore -- a public test query must not taint unrelated comparisons.
        let query = owner
            .vector_at_f32(usize::try_from(entry).expect("entry fits usize"))
            .expect("query equal to the tombstoned entry vector");
        let first = bound
            .search(&query, usize::MAX, Some(1))
            .expect("bounded widening reaches every required live row");
        let second = bound
            .search(&query, usize::MAX, Some(1))
            .expect("repeat bounded widening is deterministic");
        assert_eq!(first, second);
        assert_eq!(first.len(), owner.live_count());
        assert!(first.iter().all(|hit| hit.flags().is_live()));
        assert!(
            first
                .iter()
                .all(|hit| hit.doc_id().starts_with("ninety-percent-0")),
            "only the two live document ids may cross the boundary: {first:?}"
        );

        let expected_live_rows: Vec<u32> = (0..owner.record_count())
            .filter(|&index| owner.row(index).expect("owner row").flags().is_live())
            .map(|index| u32::try_from(index).expect("physical row fits u32"))
            .collect();
        let mut returned_rows: Vec<u32> = first
            .iter()
            .map(ValidatedNativeHnswHit::physical_row)
            .collect();
        returned_rows.sort_unstable();
        assert_eq!(
            returned_rows, expected_live_rows,
            "every live physical row must be returned exactly once"
        );
        assert_eq!(bound.graph.seed(), seed);

        let mut widths = vec![owner.live_count()];
        while *widths.last().expect("initial width") < owner.record_count() {
            widths.push(widen_search_width(
                *widths.last().expect("prior width"),
                owner.record_count(),
            ));
        }
        assert_eq!(widths, vec![2, 4, 8, 16, 20]);
        assert_eq!(
            widen_search_width(owner.record_count(), owner.record_count()),
            owner.record_count(),
            "the physical-row bound is a fixed point"
        );
    }

    #[test]
    fn all_tombstoned_owner_and_zero_k_return_truthful_empty_results() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let fsvi_binding = fsvi_v2_binding(41, 0x41, "all-tombstoned", 4);
        let rows: Vec<(String, Vec<f32>, FsviRecordFlags)> = (0..8)
            .map(|row| {
                let mut vector = vec![0.0; 4];
                vector[row % 4] = 1.0;
                (
                    format!("all-dead-{row}"),
                    vector,
                    FsviRecordFlags::TOMBSTONE,
                )
            })
            .collect();
        let owner = admitted_fsvi_owner_with_flags(
            directory.path(),
            "all-tombstoned.fsvi",
            &fsvi_binding,
            &rows,
        );
        let bound = ValidatedNativeHnsw::build(Arc::clone(&owner), params(), 41)
            .expect("all physical tombstones remain valid routing nodes");

        assert_eq!(bound.len(), 8);
        assert_eq!(owner.live_count(), 0);
        assert_eq!(owner.tombstone_count(), 8);
        assert!(
            bound
                .search(&[1.0, 0.0, 0.0, 0.0], 8, Some(1))
                .expect("all-tombstoned search")
                .is_empty()
        );
        assert!(
            bound
                .search(&[1.0, 0.0, 0.0, 0.0], 0, Some(1))
                .expect("zero-k search")
                .is_empty()
        );
    }

    #[test]
    fn physical_row_selection_never_deduplicates_equal_document_labels() {
        let store = TestStore::new(vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ]);
        let graph = NativeHnsw::build(params(), 42, &store).expect("build raw graph");
        let tombstoned = [false, true, false];
        let duplicate_labels = ["duplicate", "duplicate", "duplicate"];
        let hits = graph
            .search(&[0.0, 1.0, 0.0], 3, Some(3), &store)
            .expect("full-width physical-row search");
        let live_resolved: Vec<(u32, &str)> = hits
            .into_iter()
            .filter(|(physical_row, _)| {
                let physical_index =
                    usize::try_from(*physical_row).expect("physical row fits usize");
                !tombstoned[physical_index]
            })
            .map(|(physical_row, _)| {
                let physical_index =
                    usize::try_from(physical_row).expect("physical row fits usize");
                (physical_row, duplicate_labels[physical_index])
            })
            .collect();

        assert_eq!(live_resolved.len(), 2);
        assert_ne!(live_resolved[0].0, live_resolved[1].0);
        assert!(
            live_resolved
                .iter()
                // ubs:ignore -- this is a public document-label regression fixture.
                .all(|(_, doc_id)| *doc_id == "duplicate"),
            "equal document labels must not collapse distinct physical rows"
        );

        // Identity-complete FSVI v2 currently requires unique document ids,
        // so an owner-bound public duplicate-id state cannot be constructed.
        // Pin that upstream admission boundary explicitly instead of
        // weakening it merely to manufacture an HNSW fixture.
        let directory = tempfile::tempdir().expect("temporary directory");
        let fsvi_binding = fsvi_v2_binding(42, 0x42, "duplicate-doc-ids", 3);
        let fsvi_path = directory.path().join("duplicate-doc-ids.fsvi");
        let mut writer =
            VectorIndex::create_v2(&fsvi_path, fsvi_binding).expect("create duplicate-id fixture");
        writer
            .write_record("duplicate", &[1.0, 0.0, 0.0])
            .expect("first duplicate row");
        writer
            .write_record("duplicate", &[0.0, 1.0, 0.0])
            .expect("second duplicate row is buffered before canonical admission");
        let error = writer
            .finish()
            .expect_err("FSVI v2 must reject duplicate document ids before HNSW admission");
        assert!(
            matches!(
                error,
                SearchError::InvalidConfig {
                    ref field,
                    ref value,
                    ref reason,
                // ubs:ignore -- public schema field name, not secret material.
                } if field == "doc_id"
                    // ubs:ignore -- public duplicate marker, not secret material.
                    && value == "<duplicate>"
                    && reason.contains("unique physical row per document id")
            ),
            "unexpected duplicate-id owner rejection: {error:?}"
        );
    }

    #[test]
    fn full_width_live_underfill_is_a_typed_graph_defect() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let fsvi_binding = fsvi_v2_binding(43, 0x43, "typed-live-underfill", 4);
        let rows: Vec<(String, Vec<f32>, FsviRecordFlags)> = (0..8)
            .map(|row| {
                // ubs:ignore -- deterministic fixture row index, not secret material.
                let live = row == 0;
                let mut vector = vec![0.0; 4];
                vector[row % 4] = 1.0;
                (
                    format!("typed-underfill-{row}"),
                    vector,
                    if live {
                        FsviRecordFlags::LIVE
                    } else {
                        FsviRecordFlags::TOMBSTONE
                    },
                )
            })
            .collect();
        let owner = admitted_fsvi_owner_with_flags(
            directory.path(),
            "typed-live-underfill.fsvi",
            &fsvi_binding,
            &rows,
        );
        let seed = seed_with_tombstoned_entry(&owner);
        let mut bound = ValidatedNativeHnsw::build(Arc::clone(&owner), params(), seed)
            .expect("build structurally valid owner-bound graph");
        let live_row = (0..owner.record_count())
            .find(|&index| owner.row(index).expect("owner row").flags().is_live())
            .expect("one live row");
        let live_id = u32::try_from(live_row).expect("live row fits u32");
        assert_ne!(bound.graph.entry_point(), Some(live_id));

        // Forge an impossible post-admission reachability defect. Public
        // build/load cannot create this state because both structurally
        // attest first; the mutation proves search reports a typed defect if
        // the invariant is ever violated rather than returning zero hits.
        for point in &mut bound.graph.adjacency {
            for neighbours in &mut point.layers {
                // ubs:ignore -- public graph row id, not secret material.
                neighbours.retain(|&candidate| candidate != live_id);
            }
        }
        for neighbours in &mut bound.graph.adjacency[live_row].layers {
            neighbours.clear();
        }

        // ubs:ignore -- a public test query must not taint unrelated comparisons.
        let query = owner.vector_at_f32(live_row).expect("live query vector");
        let error = bound
            .search(&query, 1, Some(1))
            .expect_err("full-width underfill must fail closed");
        assert!(
            matches!(&error, SearchError::SubsystemError { .. }),
            "expected typed native-hnsw subsystem defect, got {error:?}"
        );
        let SearchError::SubsystemError { subsystem, source } = error else {
            return;
        };
        assert_eq!(subsystem, "native-hnsw");
        assert_eq!(
            source.downcast_ref::<GraphDefect>(),
            Some(&GraphDefect::LiveResultUnderfill {
                requested_k: 1,
                expected_live_hits: 1,
                returned_live_hits: 0,
                physical_rows: 8,
            })
        );
    }

    #[test]
    fn validated_owner_bound_search_rejects_invalid_queries_before_zero_result_paths() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let nonempty_binding = fsvi_v2_binding(37, 0x37, "query-validation-nonempty", 4);
        let nonempty_rows = fsvi_rows(8, 4);
        let nonempty_owner = admitted_fsvi_owner(
            directory.path(),
            "query-validation-nonempty.fsvi",
            &nonempty_binding,
            &nonempty_rows,
        );
        let nonempty = ValidatedNativeHnsw::build(Arc::clone(&nonempty_owner), params(), 37)
            .expect("build nonempty owner-bound graph");

        let wrong_dimension = nonempty
            .search(&[1.0, 0.0, 0.0], 0, None)
            .expect_err("zero-k must not bypass query-dimension validation");
        assert!(matches!(
            wrong_dimension,
            SearchError::DimensionMismatch {
                expected: 4,
                found: 3
            }
        ));
        for non_finite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let error = nonempty
                .search(&[non_finite, 0.0, 0.0, 0.0], 0, None)
                .expect_err("zero-k must not bypass non-finite query validation");
            assert!(
                matches!(
                    error,
                    SearchError::InvalidConfig {
                        ref field,
                        ref value,
                        ref reason,
                    } if field == "query"
                        && value == "non-finite"
                        && reason == "all query vector values must be finite"
                ),
                "unexpected non-finite query rejection: {error:?}"
            );
        }
        assert!(
            nonempty
                .search(&[1.0, 0.0, 0.0, 0.0], 0, None)
                .expect("valid zero-k query")
                .is_empty()
        );

        let empty_binding = fsvi_v2_binding(38, 0x38, "query-validation-empty", 4);
        let empty_owner = admitted_fsvi_owner(
            directory.path(),
            "query-validation-empty.fsvi",
            &empty_binding,
            &[],
        );
        let empty = ValidatedNativeHnsw::build(Arc::clone(&empty_owner), params(), 38)
            .expect("build empty owner-bound graph");

        let wrong_dimension = empty
            .search(&[1.0, 0.0, 0.0], 1, None)
            .expect_err("empty graph must not bypass query-dimension validation");
        assert!(matches!(
            wrong_dimension,
            SearchError::DimensionMismatch {
                expected: 4,
                found: 3
            }
        ));
        for non_finite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let error = empty
                .search(&[non_finite, 0.0, 0.0, 0.0], 1, None)
                .expect_err("empty graph must not bypass non-finite query validation");
            assert!(
                matches!(
                    error,
                    SearchError::InvalidConfig {
                        ref field,
                        ref value,
                        ref reason,
                    } if field == "query"
                        && value == "non-finite"
                        && reason == "all query vector values must be finite"
                ),
                "unexpected non-finite empty-graph rejection: {error:?}"
            );
        }
        assert!(
            empty
                .search(&[1.0, 0.0, 0.0, 0.0], 1, None)
                .expect("valid empty-graph query")
                .is_empty()
        );
    }

    #[test]
    fn validated_owner_bound_constructor_attests_graph_before_handle_escape() {
        fn assert_graph_defect(error: SearchError, expected: &GraphDefect) {
            assert!(
                matches!(&error, SearchError::SubsystemError { .. }),
                "expected native-hnsw subsystem error, got {error:?}"
            );
            if let SearchError::SubsystemError { subsystem, source } = error {
                assert_eq!(subsystem, "native-hnsw");
                assert_eq!(
                    source.downcast_ref::<GraphDefect>(),
                    Some(expected),
                    "verified constructor must preserve the exact graph defect"
                );
            }
        }

        let owner = generation_owner(39, 0x39, "post-build-attestation", 4, 32);
        let binding =
            ValidatedNativeHnsw::binding_for_owner(&owner).expect("derive exact owner binding");
        let other_directory = tempfile::tempdir().expect("temporary other-owner directory");
        let shared_identity = fsvi_v2_binding(39, 0x39, "post-build-attestation", 4);
        let mut other_rows = fsvi_rows(32, 4);
        other_rows[0].1 = vec![0.5, 0.5, 0.0, 0.0];
        let other_owner = admitted_fsvi_owner(
            other_directory.path(),
            "other-owner.fsvi",
            &shared_identity,
            &other_rows,
        );
        let other_binding = ValidatedNativeHnsw::binding_for_owner(&other_owner)
            .expect("derive same-cardinality mismatched binding");
        assert_eq!(
            binding.artifact_generation,
            other_binding.artifact_generation
        );
        assert_eq!(
            binding.embedding_identity_fingerprint,
            other_binding.embedding_identity_fingerprint
        );
        assert_eq!(
            binding.embedding_space_fingerprint,
            other_binding.embedding_space_fingerprint
        );
        assert_eq!(
            binding.ordered_live_docset_digest,
            other_binding.ordered_live_docset_digest
        );
        assert_eq!(
            binding.fsvi_physical_row_count,
            other_binding.fsvi_physical_row_count
        );
        assert_ne!(
            binding.vector_content_digest,
            other_binding.vector_content_digest
        );

        let mut binding_mismatch_graph =
            NativeHnsw::build(params(), 39, owner.as_ref()).expect("build binding canary graph");
        binding_mismatch_graph
            .adjacency
            .pop()
            .expect("binding canary graph has a final row");
        let binding_error = ValidatedNativeHnsw::from_verified_graph(
            Arc::clone(&owner),
            other_binding,
            binding_mismatch_graph,
        )
        .expect_err("another owner's binding must not escape with this owner and graph");
        assert!(
            matches!(
                binding_error,
                SearchError::InvalidConfig {
                    ref field,
                    ref value,
                    reason: _,
                } if field == "native_hnsw_receipt.generation_binding"
                    && value == "redacted-mismatch"
            ),
            "unexpected mismatched-binding rejection: {binding_error:?}"
        );

        let sound = NativeHnsw::build(params(), 39, owner.as_ref()).expect("build sound graph");
        ValidatedNativeHnsw::from_verified_graph(Arc::clone(&owner), binding.clone(), sound)
            .expect("sound graph passes public-handle attestation");

        let mut one_way =
            NativeHnsw::build(params(), 39, owner.as_ref()).expect("build graph to corrupt");
        let (id, layer, neighbour) = one_way
            .adjacency
            .iter()
            .enumerate()
            .find_map(|(id, point)| {
                point
                    .layers
                    .iter()
                    .enumerate()
                    .find_map(|(layer, neighbours)| {
                        neighbours
                            .first()
                            .copied()
                            .map(|neighbour| (id, layer, neighbour))
                    })
            })
            .expect("nontrivial graph has an edge");
        let id = u32::try_from(id).expect("test graph length fits u32");
        // ubs:ignore — graph point IDs are public topology data, not authenticators.
        one_way.adjacency[neighbour as usize].layers[layer].retain(|&candidate| candidate != id);
        let error =
            ValidatedNativeHnsw::from_verified_graph(Arc::clone(&owner), binding.clone(), one_way)
                .expect_err("one-way graph must not escape as a public handle");
        assert_graph_defect(
            error,
            &GraphDefect::MissingReciprocalEdge {
                id,
                layer,
                neighbour,
            },
        );

        let mut truncated =
            NativeHnsw::build(params(), 39, owner.as_ref()).expect("build graph to truncate");
        truncated.adjacency.pop().expect("graph has a final row");
        let error =
            ValidatedNativeHnsw::from_verified_graph(Arc::clone(&owner), binding, truncated)
                .expect_err("cardinality-mismatched graph must not escape as a public handle");
        assert_graph_defect(
            error,
            &GraphDefect::StoreCardinalityMismatch {
                graph_points: 31,
                store_rows: 32,
            },
        );
    }

    #[test]
    fn legacy_or_missing_fsvi_identity_is_rejected_before_graph_bytes_are_read() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let graph_path = directory.path().join("legacy.fshnsw");
        let receipt_path = native_hnsw_generation_receipt_path(&graph_path).expect("receipt path");
        let owner = generation_owner(33, 0x33, "legacy-rejection", 4, 4);
        let bound = ValidatedNativeHnsw::build(Arc::clone(&owner), params(), 33)
            .expect("owner-bound graph");
        bound.save(&graph_path).expect("owner-bound save");
        let canonical = std::fs::read(&receipt_path).expect("canonical receipt");
        std::fs::write(&graph_path, b"graph bytes must not be read")
            .expect("poison graph after receipt publication");

        let mut legacy = canonical.clone();
        legacy[8..10].copy_from_slice(&1u16.to_be_bytes());
        std::fs::write(&receipt_path, legacy).expect("write schema-v1 receipt");
        let legacy_error = ValidatedNativeHnsw::load(Arc::clone(&owner), &graph_path)
            .expect_err("schema-v1 receipt must fail");
        assert!(
            matches!(legacy_error, SearchError::IndexCorrupted { ref path, ref detail }
                if path == &receipt_path && detail.contains("schema 1")),
            "legacy receipt did not fail before graph access: {legacy_error:?}"
        );

        let identity_prefix_len = 8 + 2 + 2 + 8 + 16 + (6 * SHA256_BYTES);
        std::fs::write(&receipt_path, &canonical[..identity_prefix_len])
            .expect("write receipt missing FSVI content identity");
        let missing_error = ValidatedNativeHnsw::load(Arc::clone(&owner), &graph_path)
            .expect_err("missing FSVI content identity must fail");
        assert!(
            matches!(missing_error, SearchError::IndexCorrupted { ref path, ref detail }
                if path == &receipt_path && detail.contains("vector content digest")),
            "missing identity did not fail before graph access: {missing_error:?}"
        );

        let whole_image_prefix_len = identity_prefix_len + (2 * SHA256_BYTES);
        std::fs::write(&receipt_path, &canonical[..whole_image_prefix_len])
            .expect("write receipt missing whole-image witness");
        let missing_whole_image = ValidatedNativeHnsw::load(Arc::clone(&owner), &graph_path)
            .expect_err("missing whole-image witness must fail");
        assert!(
            matches!(missing_whole_image, SearchError::IndexCorrupted { ref path, ref detail }
                if path == &receipt_path && detail.contains("whole-image")),
            "missing whole-image witness did not fail before graph access: {missing_whole_image:?}"
        );
    }

    #[test]
    fn save_rejects_physical_row_count_mismatch_before_graph_publication() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let graph_path = directory.path().join("count-mismatch.fshnsw");
        let receipt_path = native_hnsw_generation_receipt_path(&graph_path).expect("receipt path");
        let owner = generation_owner(34, 0x34, "count-mismatch", 4, 5);
        let mut bound = ValidatedNativeHnsw::build(Arc::clone(&owner), params(), 34)
            .expect("owner-bound graph");
        bound.graph.adjacency.pop().expect("test graph row");

        let error = bound
            .save(&graph_path)
            .expect_err("physical row count mismatch must fail");
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, .. }
                // ubs:ignore — this public error-field schema label is not secret material.
                if field == "native_hnsw_receipt.graph.point_count"),
            "unexpected count mismatch error: {error:?}"
        );
        assert!(!graph_path.exists());
        assert!(!receipt_path.exists());
    }

    #[test]
    fn bound_load_rejects_missing_truncated_tampered_and_noncanonical_receipts() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let graph_path = directory.path().join("graph.fshnsw");
        let receipt_path = native_hnsw_generation_receipt_path(&graph_path).expect("receipt path");
        let owner = generation_owner(9, 9, "receipt-corruption", 6, 32);
        let bound = ValidatedNativeHnsw::build(Arc::clone(&owner), params(), 0x99)
            .expect("owner-bound graph");

        bound.graph.save(&graph_path).expect("unbound graph save");
        let missing = ValidatedNativeHnsw::load(Arc::clone(&owner), &graph_path)
            .expect_err("missing receipt must fail");
        assert!(
            // ubs:ignore — receipt paths are public filesystem identities, not credentials.
            matches!(missing, SearchError::IndexNotFound { ref path } if path == &receipt_path)
        );

        bound.save(&graph_path).expect("owner-bound save");
        let canonical = std::fs::read(&receipt_path).expect("canonical receipt");

        std::fs::write(&receipt_path, &canonical[..canonical.len() - 1]).expect("truncate receipt");
        assert!(matches!(
            ValidatedNativeHnsw::load(Arc::clone(&owner), &graph_path)
                .expect_err("truncated receipt must fail"),
            SearchError::IndexCorrupted { .. }
        ));

        let mut body_tamper = canonical.clone();
        body_tamper[20] ^= 1;
        std::fs::write(&receipt_path, body_tamper).expect("tamper receipt body");
        assert!(
            ValidatedNativeHnsw::load(Arc::clone(&owner), &graph_path).is_err(),
            "body tamper without a new SHA-256 seal must fail"
        );

        let mut seal_tamper = canonical.clone();
        *seal_tamper.last_mut().expect("receipt seal byte") ^= 1;
        std::fs::write(&receipt_path, seal_tamper).expect("tamper receipt seal");
        assert!(
            ValidatedNativeHnsw::load(Arc::clone(&owner), &graph_path).is_err(),
            "receipt SHA-256 tamper must fail"
        );

        let mut trailing = canonical;
        trailing.push(0);
        std::fs::write(&receipt_path, trailing).expect("append noncanonical byte");
        let error = ValidatedNativeHnsw::load(Arc::clone(&owner), &graph_path)
            .expect_err("trailing receipt bytes must fail");
        assert!(
            matches!(error, SearchError::IndexCorrupted { ref detail, .. }
                if detail.contains("trailing bytes") || detail.contains("canonically encoded")),
            "unexpected noncanonical receipt error: {error:?}"
        );
    }

    #[test]
    fn bound_load_rejects_swapped_stale_resealed_and_replaced_graphs() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let first_path = directory.path().join("first.fshnsw");
        let second_path = directory.path().join("second.fshnsw");
        let first_receipt =
            native_hnsw_generation_receipt_path(&first_path).expect("first receipt");
        let second_receipt =
            native_hnsw_generation_receipt_path(&second_path).expect("second receipt");
        let owner = generation_owner(11, 0x11, "swap-stale", 6, 40);
        let first = ValidatedNativeHnsw::build(Arc::clone(&owner), params(), 11)
            .expect("first owner-bound graph");
        let second = ValidatedNativeHnsw::build(Arc::clone(&owner), params(), 29)
            .expect("second owner-bound graph");

        first.save(&first_path).expect("save first");
        second.save(&second_path).expect("save second");
        let first_receipt_bytes = std::fs::read(&first_receipt).expect("first receipt bytes");
        let second_receipt_bytes = std::fs::read(&second_receipt).expect("second receipt bytes");
        std::fs::write(&first_receipt, &second_receipt_bytes).expect("swap second into first");
        std::fs::write(&second_receipt, &first_receipt_bytes).expect("swap first into second");
        assert!(
            ValidatedNativeHnsw::load(Arc::clone(&owner), &first_path).is_err(),
            "receipt from a different basename/artifact must fail"
        );
        assert!(
            ValidatedNativeHnsw::load(Arc::clone(&owner), &second_path).is_err(),
            "the reciprocal receipt swap must also fail"
        );

        std::fs::write(&first_receipt, &first_receipt_bytes).expect("restore first receipt");
        second
            .graph
            .save(&first_path)
            .expect("replace graph without receipt");
        let stale = ValidatedNativeHnsw::load(Arc::clone(&owner), &first_path)
            .expect_err("stale receipt must fail");
        assert!(
            matches!(stale, SearchError::IndexCorrupted { ref detail, .. }
                if detail.contains("SHA-256") || detail.contains("receipt")),
            "unexpected stale-receipt error: {stale:?}"
        );

        first.save(&first_path).expect("restore bound first");
        let mut forged_graph = std::fs::read(&first_path).expect("graph bytes");
        let (neighbour_offset, edge_owner) =
            first_persisted_neighbour(&forged_graph).expect("graph edge");
        put_u32(&mut forged_graph, neighbour_offset, edge_owner);
        reseal_persisted_checksums(&mut forged_graph);
        std::fs::write(&first_path, forged_graph).expect("write resealed graph forgery");
        let resealed = ValidatedNativeHnsw::load(Arc::clone(&owner), &first_path)
            .expect_err("resealed graph replacement must fail receipt SHA-256");
        assert!(
            matches!(resealed, SearchError::IndexCorrupted { ref detail, .. }
                if detail.contains("SHA-256")),
            "unexpected resealed graph error: {resealed:?}"
        );
    }

    #[test]
    fn bound_load_rejects_replacement_between_hash_and_parse() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let graph_path = directory.path().join("race.fshnsw");
        let owner = generation_owner(3, 3, "replacement-race", 6, 48);
        let original = ValidatedNativeHnsw::build(Arc::clone(&owner), params(), 3)
            .expect("original owner-bound graph");
        let replacement = ValidatedNativeHnsw::build(Arc::clone(&owner), params(), 5)
            .expect("replacement owner-bound graph");
        original.save(&graph_path).expect("save original binding");

        let error = ValidatedNativeHnsw::load_with_after_first_observation(
            Arc::clone(&owner),
            &graph_path,
            || replacement.graph.save(&graph_path).map(|_| ()),
        )
        .expect_err("replacement during verification must fail");
        assert!(
            matches!(error, SearchError::IndexCorrupted { .. }),
            "replacement race was not rejected: {error:?}"
        );
    }

    #[test]
    fn every_resealed_receipt_field_is_checked_against_trust_or_graph_state() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let graph_path = directory.path().join("all-fields.fshnsw");
        let receipt_path = native_hnsw_generation_receipt_path(&graph_path).expect("receipt path");
        let owner = generation_owner(77, 0x77, "every-receipt-field", 8, 48);
        let bound = ValidatedNativeHnsw::build(Arc::clone(&owner), params(), 0xfeed)
            .expect("owner-bound graph");
        let original = bound.save(&graph_path).expect("owner-bound save");

        macro_rules! rejects_resealed {
            ($label:literal, $change:expr) => {{
                let mut changed = original.clone();
                ($change)(&mut changed);
                write_resealed_receipt(&receipt_path, &mut changed);
                let error = ValidatedNativeHnsw::load(Arc::clone(&owner), &graph_path)
                    .expect_err(concat!($label, " drift must fail"));
                assert!(
                    matches!(
                        error,
                        SearchError::IndexCorrupted { .. } | SearchError::InvalidConfig { .. }
                    ),
                    "{} drift returned an unexpected error: {error:?}",
                    $label
                );
            }};
        }

        rejects_resealed!("receipt schema", |r: &mut NativeHnswGenerationReceiptV2| {
            r.schema_version += 1;
        });
        rejects_resealed!(
            "generation schema",
            |r: &mut NativeHnswGenerationReceiptV2| {
                r.artifact_generation.schema_version += 1;
                r.artifact_generation_fingerprint = r.artifact_generation.fingerprint();
            }
        );
        rejects_resealed!(
            "generation sequence",
            |r: &mut NativeHnswGenerationReceiptV2| {
                r.artifact_generation.sequence += 1;
                r.artifact_generation_fingerprint = r.artifact_generation.fingerprint();
            }
        );
        rejects_resealed!(
            "generation nonce",
            |r: &mut NativeHnswGenerationReceiptV2| {
                r.artifact_generation.nonce[0] ^= 1;
                r.artifact_generation_fingerprint = r.artifact_generation.fingerprint();
            }
        );
        rejects_resealed!(
            "generation fingerprint",
            |r: &mut NativeHnswGenerationReceiptV2| {
                r.artifact_generation_fingerprint = "0".repeat(64);
            }
        );
        rejects_resealed!(
            "embedding identity fingerprint",
            |r: &mut NativeHnswGenerationReceiptV2| {
                r.embedding_identity_fingerprint = "1".repeat(64);
            }
        );
        rejects_resealed!(
            "embedding space fingerprint",
            |r: &mut NativeHnswGenerationReceiptV2| {
                r.embedding_space_fingerprint = "2".repeat(64);
            }
        );
        rejects_resealed!(
            "embedding producer fingerprint",
            |r: &mut NativeHnswGenerationReceiptV2| {
                r.embedding_producer_fingerprint = "3".repeat(64);
            }
        );
        rejects_resealed!(
            "embedding input fingerprint",
            |r: &mut NativeHnswGenerationReceiptV2| {
                r.embedding_input_fingerprint = "4".repeat(64);
            }
        );
        rejects_resealed!(
            "vector storage fingerprint",
            |r: &mut NativeHnswGenerationReceiptV2| {
                r.vector_storage_fingerprint = "5".repeat(64);
            }
        );
        rejects_resealed!(
            "vector content digest",
            |r: &mut NativeHnswGenerationReceiptV2| {
                r.vector_content_digest = "8".repeat(64);
            }
        );
        rejects_resealed!(
            "ordered live-docset digest",
            |r: &mut NativeHnswGenerationReceiptV2| {
                r.ordered_live_docset_digest = "9".repeat(64);
            }
        );
        rejects_resealed!(
            "FSVI whole-image SHA-256",
            |r: &mut NativeHnswGenerationReceiptV2| {
                r.fsvi_whole_image_sha256 = "a".repeat(64);
            }
        );
        rejects_resealed!(
            "FSVI physical row count",
            |r: &mut NativeHnswGenerationReceiptV2| {
                r.fsvi_physical_row_count += 1;
            }
        );
        rejects_resealed!("graph basename", |r: &mut NativeHnswGenerationReceiptV2| {
            r.graph_basename = "other.fshnsw".to_owned();
        });
        rejects_resealed!(
            "graph byte length",
            |r: &mut NativeHnswGenerationReceiptV2| {
                r.graph_byte_len += 1;
            }
        );
        rejects_resealed!("graph SHA-256", |r: &mut NativeHnswGenerationReceiptV2| {
            r.graph_sha256 = "6".repeat(64);
        });
        rejects_resealed!(
            "native format version",
            |r: &mut NativeHnswGenerationReceiptV2| {
                r.native_format_version += 1;
            }
        );
        rejects_resealed!("parameter m", |r: &mut NativeHnswGenerationReceiptV2| {
            r.params.m += 1;
        });
        rejects_resealed!("parameter m0", |r: &mut NativeHnswGenerationReceiptV2| {
            r.params.m0 += 1;
        });
        rejects_resealed!(
            "parameter ef_construction",
            |r: &mut NativeHnswGenerationReceiptV2| {
                r.params.ef_construction += 1;
            }
        );
        rejects_resealed!(
            "parameter ef_search",
            |r: &mut NativeHnswGenerationReceiptV2| {
                r.params.ef_search += 1;
            }
        );
        rejects_resealed!("seed", |r: &mut NativeHnswGenerationReceiptV2| {
            r.seed ^= 1;
        });
        rejects_resealed!("point count", |r: &mut NativeHnswGenerationReceiptV2| {
            r.point_count += 1;
        });
        rejects_resealed!("entry point", |r: &mut NativeHnswGenerationReceiptV2| {
            let count = u32::try_from(r.point_count).expect("test count");
            r.entry_point = Some((r.entry_point.expect("non-empty graph") + 1) % count);
        });
        rejects_resealed!("maximum level", |r: &mut NativeHnswGenerationReceiptV2| {
            r.max_level = u64::from(r.max_level == 0);
        });
        rejects_resealed!("payload CRC", |r: &mut NativeHnswGenerationReceiptV2| {
            r.payload_crc32 ^= 1;
        });
        rejects_resealed!("header CRC", |r: &mut NativeHnswGenerationReceiptV2| {
            r.header_crc32 ^= 1;
        });
        rejects_resealed!(
            "topology SHA-256",
            |r: &mut NativeHnswGenerationReceiptV2| {
                r.topology_sha256 = "7".repeat(64);
            }
        );

        let mut receipt_seal_tamper = original.to_bytes().expect("canonical receipt");
        *receipt_seal_tamper.last_mut().expect("receipt seal byte") ^= 1;
        std::fs::write(&receipt_path, receipt_seal_tamper).expect("receipt seal tamper");
        assert!(
            ValidatedNativeHnsw::load(Arc::clone(&owner), &graph_path).is_err(),
            "receipt SHA-256 is itself a bound field"
        );

        std::fs::write(
            &receipt_path,
            original.to_bytes().expect("restore original receipt"),
        )
        .expect("restore receipt");
        let wrong_generation = generation_owner(78, 0x78, "every-receipt-field", 8, 48);
        assert!(
            ValidatedNativeHnsw::load(Arc::clone(&wrong_generation), &graph_path).is_err(),
            "caller-held generation drift must fail"
        );
        let wrong_identity = generation_owner(77, 0x77, "different-identity", 8, 48);
        assert!(
            ValidatedNativeHnsw::load(Arc::clone(&wrong_identity), &graph_path).is_err(),
            "caller-held frozen identity drift must fail"
        );
    }

    #[test]
    fn receipt_paths_fail_closed_on_malformed_or_nonlocal_names() {
        for malformed in [
            Path::new(""),
            Path::new("./graph.fshnsw"),
            Path::new("nested/../graph.fshnsw"),
            Path::new("graph.bin"),
            Path::new("graph\nforged.fshnsw"),
        ] {
            assert!(
                native_hnsw_generation_receipt_path(malformed).is_err(),
                "malformed graph path was admitted: {malformed:?}"
            );
        }

        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt;

            let invalid = PathBuf::from(std::ffi::OsString::from_vec(
                b"invalid-\xff.fshnsw".to_vec(),
            ));
            assert!(
                native_hnsw_generation_receipt_path(&invalid).is_err(),
                "non-UTF-8 basename must not enter the receipt"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn receipt_save_and_load_reject_symlinks_special_files_and_symlinked_parents() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().expect("temporary directory");
        let owner = generation_owner(19, 0x19, "path-hardening", 4, 16);
        let bound = ValidatedNativeHnsw::build(Arc::clone(&owner), params(), 19)
            .expect("owner-bound graph");

        let preflight_graph = directory.path().join("preflight.fshnsw");
        let preflight_receipt =
            native_hnsw_generation_receipt_path(&preflight_graph).expect("receipt path");
        let receipt_target = directory.path().join("receipt-target");
        std::fs::write(&receipt_target, b"sentinel").expect("receipt target");
        symlink(&receipt_target, &preflight_receipt).expect("receipt symlink");
        assert!(
            matches!(
                bound
                    .save(&preflight_graph)
                    .expect_err("receipt symlink must fail before graph save"),
                SearchError::Io(_)
            ),
            "receipt symlink returned wrong error"
        );
        assert!(
            !preflight_graph.exists(),
            "receipt preflight failure must not publish the graph"
        );

        let source_graph = directory.path().join("source.fshnsw");
        bound.save(&source_graph).expect("source binding");
        let source_receipt =
            native_hnsw_generation_receipt_path(&source_graph).expect("source receipt");
        let alias_graph = directory.path().join("alias.fshnsw");
        bound.graph.save(&alias_graph).expect("alias graph bytes");
        let alias_receipt =
            native_hnsw_generation_receipt_path(&alias_graph).expect("alias receipt");
        symlink(&source_receipt, &alias_receipt).expect("load receipt symlink");
        assert!(matches!(
            ValidatedNativeHnsw::load(Arc::clone(&owner), &alias_graph)
                .expect_err("load must reject receipt symlink"),
            SearchError::Io(_)
        ));

        let special_graph = directory.path().join("special.fshnsw");
        let special_receipt =
            native_hnsw_generation_receipt_path(&special_graph).expect("special receipt");
        std::fs::create_dir(&special_receipt).expect("special receipt directory");
        assert!(matches!(
            bound
                .save(&special_graph)
                .expect_err("special receipt must fail before graph save"),
            SearchError::Io(_)
        ));
        assert!(!special_graph.exists());

        let real_parent = directory.path().join("real-parent");
        std::fs::create_dir(&real_parent).expect("real parent");
        let alias_parent = directory.path().join("alias-parent");
        symlink(&real_parent, &alias_parent).expect("symlink parent");
        assert!(matches!(
            bound
                .save(&alias_parent.join("graph.fshnsw"))
                .expect_err("symlinked parent must fail"),
            SearchError::Io(_)
        ));
    }

    // ─── Owned persistence ──────────────────────────────────────────────

    #[test]
    fn owned_empty_graph_round_trip_preserves_empty_identity() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("empty.fshnsw");
        let store = TestStore::new(Vec::new());
        let graph = NativeHnsw::build(params(), 0x5eed_cafe, &store).expect("build empty");

        let saved = graph.save(&path).expect("save empty");
        let loaded = NativeHnsw::load(&path, &store).expect("load empty");

        assert_eq!(saved.point_count(), 0);
        assert_eq!(
            saved.byte_len(),
            u64::try_from(NATIVE_HNSW_HEADER_LEN).expect("header length fits u64")
        );
        assert_eq!(loaded.params(), graph.params());
        assert_eq!(loaded.seed(), graph.seed());
        assert_eq!(loaded.len(), 0);
        assert!(loaded.entry_point().is_none());
        assert_eq!(loaded.max_level(), 0);
        assert!(
            loaded
                .search(&[1.0], 10, None, &store)
                .expect("search")
                .is_empty()
        );
    }

    #[test]
    fn owned_empty_graph_rejects_resealed_nonzero_max_level() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("empty.fshnsw");
        let store = TestStore::new(Vec::new());
        NativeHnsw::build(params(), 0x5eed_cafe, &store)
            .expect("build empty")
            .save(&path)
            .expect("save empty");

        let mut bytes = std::fs::read(&path).expect("empty graph bytes");
        put_u64(&mut bytes, 72, 1);
        reseal_persisted_checksums(&mut bytes);
        std::fs::write(&path, bytes).expect("write resealed empty-level forgery");

        let error =
            NativeHnsw::load(&path, &store).expect_err("empty graph maximum must remain zero");
        assert!(
            matches!(error, SearchError::IndexCorrupted { ref detail, .. }
                if detail.contains("empty graph claims maximum level 1")),
            "unexpected empty-graph maximum-level error: {error:?}"
        );
    }

    #[test]
    fn owned_graph_round_trip_preserves_topology_identity_and_search() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("graph.fshnsw");
        let store = TestStore::synthetic(96, 8);
        let graph = NativeHnsw::build(params(), 0x5eed_cafe, &store).expect("build");

        let saved = graph.save(&path).expect("save");
        let (loaded, observed) =
            NativeHnsw::load_with_metadata(&path, &store).expect("load and attest");

        assert_eq!(saved, observed);
        assert_eq!(saved.format_version(), NATIVE_HNSW_FORMAT_VERSION);
        assert_eq!(saved.point_count(), 96);
        assert_eq!(
            saved.byte_len(),
            std::fs::metadata(&path).expect("metadata").len()
        );
        assert_ne!(saved.payload_crc32(), 0);
        assert_ne!(saved.header_crc32(), 0);
        assert_eq!(loaded.params(), graph.params());
        assert_eq!(loaded.seed(), graph.seed());
        assert_eq!(topology_snapshot(&loaded), topology_snapshot(&graph));

        for query_id in 0..8 {
            let query = TestStore::query(query_id, 8);
            assert_eq!(
                loaded
                    .search(&query, 10, None, &store)
                    .expect("loaded search"),
                graph
                    .search(&query, 10, None, &store)
                    .expect("original search")
            );
        }
    }

    #[test]
    fn owned_graph_serialization_is_byte_deterministic() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let first_path = directory.path().join("first.fshnsw");
        let second_path = directory.path().join("second.fshnsw");
        let store = TestStore::synthetic(64, 6);
        let graph = NativeHnsw::build(params(), 0x1234_5678, &store).expect("build");

        graph.save(&first_path).expect("first save");
        graph.save(&second_path).expect("second save");

        assert_eq!(
            std::fs::read(first_path).expect("first bytes"),
            std::fs::read(second_path).expect("second bytes")
        );
    }

    #[test]
    fn owned_graph_load_rejects_truncation_and_payload_tampering() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("graph.fshnsw");
        let store = TestStore::synthetic(48, 6);
        let graph = NativeHnsw::build(params(), 91, &store).expect("build");
        graph.save(&path).expect("save");
        let original = std::fs::read(&path).expect("graph bytes");

        std::fs::write(&path, &original[..original.len() - 1]).expect("truncate");
        let error = NativeHnsw::load(&path, &store).expect_err("truncation must fail");
        assert!(
            matches!(error, SearchError::IndexCorrupted { .. }),
            "unexpected truncation error: {error:?}"
        );

        let mut tampered = original;
        let last = tampered.last_mut().expect("non-empty graph file");
        *last ^= 0x80;
        std::fs::write(&path, tampered).expect("tamper payload");
        let error = NativeHnsw::load(&path, &store).expect_err("payload tamper must fail");
        assert!(
            matches!(error, SearchError::IndexCorrupted { ref detail, .. }
                if detail.contains("payload CRC mismatch")),
            "payload tamper did not reach the CRC gate: {error:?}"
        );
    }

    #[test]
    fn owned_graph_load_rejects_header_version_and_checksum_tampering() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("graph.fshnsw");
        let store = TestStore::synthetic(24, 4);
        NativeHnsw::build(params(), 17, &store)
            .expect("build")
            .save(&path)
            .expect("save");
        let original = std::fs::read(&path).expect("graph bytes");

        let mut wrong_version = original.clone();
        put_u32(
            &mut wrong_version,
            8,
            NATIVE_HNSW_FORMAT_VERSION.saturating_add(1),
        );
        reseal_persisted_checksums(&mut wrong_version);
        std::fs::write(&path, wrong_version).expect("write version tamper");
        let error = NativeHnsw::load(&path, &store).expect_err("version tamper must fail");
        assert!(
            matches!(error, SearchError::IndexCorrupted { ref detail, .. }
                if detail.contains("unsupported native HNSW format version")),
            "unexpected version error: {error:?}"
        );

        let mut bad_header_crc = original;
        bad_header_crc[NATIVE_HNSW_HEADER_CRC_OFFSET] ^= 0x01;
        std::fs::write(&path, bad_header_crc).expect("write CRC tamper");
        let error = NativeHnsw::load(&path, &store).expect_err("header CRC tamper must fail");
        assert!(
            matches!(error, SearchError::IndexCorrupted { ref detail, .. }
                if detail.contains("header CRC mismatch")),
            "unexpected header CRC error: {error:?}"
        );
    }

    #[test]
    fn owned_graph_structural_attestation_rejects_resealed_forgery() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("graph.fshnsw");
        let store = TestStore::synthetic(40, 6);
        NativeHnsw::build(params(), 73, &store)
            .expect("build")
            .save(&path)
            .expect("save");

        let mut bytes = std::fs::read(&path).expect("graph bytes");
        let (neighbour_offset, owner) =
            first_persisted_neighbour(&bytes).expect("non-trivial graph contains an edge");
        put_u32(&mut bytes, neighbour_offset, owner);
        reseal_persisted_checksums(&mut bytes);
        std::fs::write(&path, bytes).expect("write resealed forgery");

        let error =
            NativeHnsw::load(&path, &store).expect_err("resealed structural forgery must fail");
        assert!(
            matches!(error, SearchError::IndexCorrupted { ref detail, .. }
                if detail.contains("is its own neighbour")),
            "unexpected structural error: {error:?}"
        );
    }

    #[test]
    fn owned_graph_seed_is_attested_against_every_persisted_level() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("graph.fshnsw");
        let store = TestStore::synthetic(64, 6);
        let graph = NativeHnsw::build(params(), 11, &store).expect("build");
        graph.save(&path).expect("save");

        let mut bytes = std::fs::read(&path).expect("graph bytes");
        let forged_seed = (12_u64..=u64::from(u16::MAX))
            .find(|&candidate| {
                let sampler = LevelSampler::new(params().m, candidate);
                graph.adjacency.iter().enumerate().any(|(id, point)| {
                    sampler.level_for(u32::try_from(id).expect("test id")) != point.level()
                })
            })
            .expect("some alternate seed changes a sampled level");
        put_u64(&mut bytes, 48, forged_seed);
        reseal_persisted_checksums(&mut bytes);
        std::fs::write(&path, bytes).expect("write resealed seed forgery");

        let error = NativeHnsw::load(&path, &store).expect_err("seed forgery must fail");
        assert!(
            matches!(error, SearchError::IndexCorrupted { ref detail, .. }
                if detail.contains("seed and parameters sample level")),
            "unexpected seed-attestation error: {error:?}"
        );
    }

    #[test]
    fn owned_graph_load_rejects_store_cardinality_mismatch() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("graph.fshnsw");
        let store = TestStore::synthetic(16, 4);
        NativeHnsw::build(params(), 41, &store)
            .expect("build")
            .save(&path)
            .expect("save");
        let wrong_store = TestStore::synthetic(15, 4);

        let error =
            NativeHnsw::load(&path, &wrong_store).expect_err("store cardinality must bind load");
        assert!(
            matches!(error, SearchError::IndexCorrupted { ref detail, .. }
                if detail.contains("vector store exposes 15 rows")),
            "unexpected cardinality error: {error:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn owned_graph_load_and_save_reject_symbolic_links() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().expect("temporary directory");
        let target = directory.path().join("target.fshnsw");
        let alias = directory.path().join("alias.fshnsw");
        let store = TestStore::synthetic(8, 4);
        let graph = NativeHnsw::build(params(), 29, &store).expect("build");
        graph.save(&target).expect("save target");
        symlink(&target, &alias).expect("create symlink");

        let load_error = NativeHnsw::load(&alias, &store).expect_err("load must reject symlink");
        assert!(matches!(load_error, SearchError::Io(_)));
        let save_error = graph.save(&alias).expect_err("save must reject symlink");
        assert!(matches!(save_error, SearchError::Io(_)));
        assert!(
            NativeHnsw::load(&target, &store).is_ok(),
            "rejecting the alias must not alter its target"
        );
    }

    #[test]
    fn malformed_graph_refuses_save_before_touching_existing_artifact() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("graph.fshnsw");
        let sentinel = b"existing artifact must survive";
        std::fs::write(&path, sentinel).expect("write sentinel");
        let store = TestStore::synthetic(32, 4);
        let mut graph = NativeHnsw::build(params(), 17, &store).expect("build");
        let (id, layer, neighbour) = graph
            .adjacency
            .iter()
            .enumerate()
            .find_map(|(id, point)| {
                point
                    .layers
                    .iter()
                    .enumerate()
                    .find_map(|(layer, neighbours)| {
                        neighbours
                            .first()
                            .copied()
                            .map(|neighbour| (id, layer, neighbour))
                    })
            })
            .expect("non-trivial graph edge");
        let id = u32::try_from(id).expect("test id");
        graph.adjacency[neighbour as usize].layers[layer].retain(|&candidate| candidate != id);

        let error = graph
            .save(&path)
            .expect_err("malformed graph must not publish");
        assert!(
            matches!(error, SearchError::IndexCorrupted { .. }),
            "unexpected save error: {error:?}"
        );
        assert_eq!(std::fs::read(path).expect("surviving bytes"), sentinel);
    }

    #[test]
    fn owned_graph_save_atomically_replaces_complete_prior_generation() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("graph.fshnsw");
        let first_store = TestStore::synthetic(12, 4);
        let first = NativeHnsw::build(params(), 3, &first_store).expect("first build");
        let first_metadata = first.save(&path).expect("first save");

        let second_store = TestStore::synthetic(48, 4);
        let second = NativeHnsw::build(params(), 5, &second_store).expect("second build");
        let second_metadata = second.save(&path).expect("replacement save");
        let loaded = NativeHnsw::load(&path, &second_store).expect("load replacement");

        assert_ne!(first_metadata.point_count(), second_metadata.point_count());
        assert_eq!(second_metadata.point_count(), 48);
        assert_eq!(topology_snapshot(&loaded), topology_snapshot(&second));
    }

    // ─── bd-u3wt regression fixtures ────────────────────────────────────
    //
    // Each of these reproduces a defect the previous engine shipped.

    /// bd-u3wt: two orthogonal vectors, k=2 — the original reproduction.
    /// One direction returned only one result because the reverse edge was
    /// stored at the wrong layer, leaving the other point unreachable.
    #[test]
    fn two_orthogonal_points_are_mutually_reachable() {
        let store = TestStore::new(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        let graph = NativeHnsw::build(params(), 0x5eed, &store).expect("build");
        graph.verify().expect("structure");

        for query in [&[1.0_f32, 0.0], &[0.0_f32, 1.0]] {
            let hits = graph.search(query, 2, None, &store).expect("search");
            assert_eq!(
                hits.len(),
                2,
                "both points must be reachable from query {query:?}, got {hits:?}"
            );
        }
    }

    /// bd-u3wt defect (4): a `high, high, zero` level sequence left the
    /// logical layer looking empty because participation was tested as
    /// `level == layer` rather than `level >= layer`.
    #[test]
    fn high_high_zero_level_sequence_stays_searchable() {
        // Force the exact shape by driving the sampler's output through a
        // constructed graph rather than hoping the seed produces it.
        let store = TestStore::new(vec![vec![1.0, 0.0], vec![0.9, 0.1], vec![0.0, 1.0]]);
        // Seed 3 deterministically samples levels [1, 1, 0] for ids
        // [0, 1, 2] with this parameter set.
        let mut graph = NativeHnsw::new(params(), 3).expect("new");
        let first_level = graph.sampler.level_for(0);
        assert_eq!(first_level, 1);
        graph.adjacency.push(Adjacency::with_level(first_level));
        graph.entry = Some(0);
        graph.max_level = first_level;
        // Points 1 and 2 arrive at level 1 and level 0 respectively.
        for id in [1_u32, 2] {
            let level = graph.sampler.level_for(id);
            graph.adjacency.push(Adjacency::with_level(level));
            let selected = (0..id).collect::<Vec<_>>();
            for layer in 0..=level.min(graph.max_level) {
                let mut journal = MutationJournal::new(id as usize);
                graph
                    .link(id, &selected, layer, &store, &mut journal)
                    .expect("link");
            }
        }
        graph
            .verify()
            .expect("structure must hold for high,high,zero");

        let hits = graph.search(&[0.0, 1.0], 3, None, &store).expect("search");
        assert_eq!(hits.len(), 3, "every point must remain findable: {hits:?}");
    }

    /// bd-u3wt defect (3): an edge must never exist in a layer either
    /// endpoint does not occupy. `verify` is the executable statement of
    /// that invariant, so prove it actually rejects a violation rather
    /// than passing vacuously.
    #[test]
    fn verify_rejects_an_edge_above_a_neighbours_level() {
        let store = TestStore::synthetic(32, 4);
        let mut graph = NativeHnsw::build(params(), 7, &store).expect("build");
        graph.verify().expect("a freshly built graph is sound");

        // Find a point that reaches a layer above some neighbour, then
        // forge exactly the edge the old engine used to create.
        let low = graph
            .adjacency
            .iter()
            .position(|point| point.level() == 0)
            .expect("some point sits at level 0");
        let high = graph
            .adjacency
            .iter()
            .position(|point| point.level() >= 1)
            .expect("some point reaches level 1");
        let low_id = u32::try_from(low).expect("test graph length fits u32");
        graph.adjacency[high].layers[1].push(low_id);

        let defect = graph
            .verify()
            .expect_err("an edge above the neighbour's level must be rejected");
        assert!(
            matches!(
                defect,
                GraphDefect::EdgeAboveNeighbourLevel { neighbour, .. }
                    if neighbour == low_id
            ),
            "expected the layer-violation defect naming point {low}, got: {defect:?}"
        );
    }

    #[test]
    fn verify_rejects_a_missing_reciprocal_edge() {
        let store = TestStore::synthetic(32, 4);
        let mut graph = NativeHnsw::build(params(), 17, &store).expect("build");
        graph.verify().expect("a freshly built graph is sound");

        let (id, layer, neighbour) = graph
            .adjacency
            .iter()
            .enumerate()
            .find_map(|(id, point)| {
                point
                    .layers
                    .iter()
                    .enumerate()
                    .find_map(|(layer, neighbours)| {
                        neighbours
                            .first()
                            .copied()
                            .map(|neighbour| (id, layer, neighbour))
                    })
            })
            .expect("a non-trivial graph has an edge");
        let id = u32::try_from(id).expect("test graph length fits u32");
        graph.adjacency[neighbour as usize].layers[layer].retain(|&candidate| candidate != id);

        assert_eq!(
            graph
                .verify()
                .expect_err("a one-way edge must fail structural attestation"),
            GraphDefect::MissingReciprocalEdge {
                id,
                layer,
                neighbour,
            }
        );
    }

    #[test]
    fn store_cardinality_is_part_of_attestation_and_search_admission() {
        let store = TestStore::synthetic(12, 4);
        let graph = NativeHnsw::build(params(), 23, &store).expect("build");
        graph.verify_for_store(&store).expect("matching store");

        for mismatched_count in [11usize, 13] {
            let mismatched = TestStore::synthetic(mismatched_count, 4);
            assert_eq!(
                graph
                    .verify_for_store(&mismatched)
                    .expect_err("mismatched store must fail attestation"),
                GraphDefect::StoreCardinalityMismatch {
                    graph_points: 12,
                    store_rows: mismatched_count,
                }
            );
            let error = graph
                .search(&[1.0, 0.0, 0.0, 0.0], 0, None, &mismatched)
                .expect_err("cardinality admission must precede the zero-k fast path");
            assert!(
                error.to_string().contains("vector store exposes"),
                "typed cardinality defect was not preserved: {error}"
            );
        }

        let empty = NativeHnsw::new(params(), 23).expect("empty graph");
        let error = empty
            .search(&[1.0, 0.0, 0.0, 0.0], 0, None, &store)
            .expect_err("cardinality admission must precede the empty-graph fast path");
        assert!(
            error.to_string().contains("graph indexes 0 points"),
            "empty mismatch was silently accepted: {error}"
        );
    }

    #[test]
    fn insert_is_failure_atomic_at_every_distance_boundary() {
        let store = TestStore::synthetic(48, 8);
        let mut before = NativeHnsw::new(params(), 0x00a7_0b1c).expect("new");
        for id in 0..40u32 {
            before.insert(id, &store).expect("build prefix");
        }
        before.verify().expect("prefix graph");
        let before_topology = topology_snapshot(&before);

        let counting = FailingStore::new(&store, None);
        let mut expected = before.clone();
        expected.insert(40, &counting).expect("reference insertion");
        let distance_boundaries = counting.between_calls();
        assert!(
            distance_boundaries > params().m0,
            "fixture must exercise search, selection, and post-link pruning boundaries"
        );
        let expected_topology = topology_snapshot(&expected);

        for fail_at in 1..=distance_boundaries {
            let failing = FailingStore::new(&store, Some(fail_at));
            let mut trial = before.clone();
            let error = trial
                .insert(40, &failing)
                .expect_err("the selected distance boundary must fail");
            assert!(
                error
                    .to_string()
                    .contains("injected distance_between failure"),
                "unexpected error at distance boundary {fail_at}: {error}"
            );
            assert_eq!(
                topology_snapshot(&trial),
                before_topology,
                "distance failure {fail_at}/{distance_boundaries} changed graph topology, entry, \
                 maximum level, or point count"
            );
            trial.verify().expect("rollback must leave a sound graph");

            trial
                .insert(40, &store)
                .expect("retry after a rolled-back failure must succeed");
            assert_eq!(
                topology_snapshot(&trial),
                expected_topology,
                "retry after distance failure {fail_at} did not reproduce the clean insertion"
            );
        }
    }

    // ─── Boundaries ─────────────────────────────────────────────────────

    #[test]
    fn point_count_boundaries_build_and_verify() {
        // n = 0,1,2,3 and around the degree budget m and m+1.
        for count in [0usize, 1, 2, 3, 8, 9, 17] {
            let store = TestStore::synthetic(count, 6);
            let graph =
                NativeHnsw::build(params(), 42, &store).expect("boundary-sized graph must build");
            graph.verify().expect("boundary-sized graph must verify");
            assert_eq!(graph.len(), count);

            let hits = graph
                .search(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 4, None, &store)
                .expect("search");
            assert_eq!(
                hits.len(),
                count.min(4),
                "k is capped by the corpus size, not by graph structure"
            );
        }
    }

    #[test]
    fn empty_graph_and_zero_k_are_benign() {
        let store = TestStore::synthetic(0, 4);
        let graph = NativeHnsw::build(params(), 1, &store).expect("build empty");
        assert!(graph.is_empty());
        assert_eq!(graph.entry_point(), None);
        assert!(
            graph
                .search(&[1.0, 0.0, 0.0, 0.0], 5, None, &store)
                .expect("search")
                .is_empty()
        );

        let store = TestStore::synthetic(10, 4);
        let graph = NativeHnsw::build(params(), 1, &store).expect("build");
        assert!(
            graph
                .search(&[1.0, 0.0, 0.0, 0.0], 0, None, &store)
                .expect("search")
                .is_empty()
        );
    }

    // ─── Differential recall against exact search ───────────────────────

    /// The property that actually matters: the graph must find what an
    /// exhaustive scan finds. Exact search is the oracle.
    #[test]
    fn recall_against_exact_search_is_high() {
        let store = TestStore::synthetic(500, 12);
        let graph = NativeHnsw::build(params(), 0xa11ce, &store).expect("build");
        graph.verify().expect("structure");

        let mut total = 0usize;
        let mut found = 0usize;
        for q in 0..25usize {
            let query = TestStore::query(q, 12);
            let expected = store.exact_top_k(&query, 10);
            let actual: Vec<u32> = graph
                .search(&query, 10, Some(128), &store)
                .expect("search")
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            total += expected.len();
            found += expected.iter().filter(|id| actual.contains(id)).count();
        }
        let recall = found as f64 / total as f64;
        // Measured at 1.0 on this corpus; the bound leaves headroom for
        // tuning without letting a real regression through.
        assert!(
            recall >= 0.98,
            "recall@10 against exact search was {recall:.4} ({found}/{total}); the graph must \
             find what a full scan finds"
        );
    }

    /// A wide beam must reach exact agreement on a small corpus: with ef
    /// at or above the corpus size there is nowhere for a true neighbour
    /// to hide, so anything missing is a structural defect.
    #[test]
    fn exhaustive_beam_matches_exact_search_exactly() {
        let store = TestStore::synthetic(64, 8);
        let graph = NativeHnsw::build(params(), 0xb0b, &store).expect("build");
        graph.verify().expect("structure");

        for q in 0..10usize {
            let query = TestStore::query(q, 8);
            let expected = store.exact_top_k(&query, 5);
            let actual: Vec<u32> = graph
                .search(&query, 5, Some(64), &store)
                .expect("search")
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            assert_eq!(
                actual, expected,
                "with a full-width beam the graph must agree with exact search on query {q}"
            );
        }
    }

    // ─── Determinism and parameters ─────────────────────────────────────

    #[test]
    fn construction_is_deterministic_for_a_seed() {
        let store = TestStore::synthetic(120, 6);
        let first = NativeHnsw::build(params(), 99, &store).expect("build");
        let second = NativeHnsw::build(params(), 99, &store).expect("build");
        assert_eq!(first.max_level(), second.max_level());
        assert_eq!(first.entry_point(), second.entry_point());
        let point_count = u32::try_from(first.len()).expect("test graph length fits u32");
        for id in 0..point_count {
            for layer in 0..=first.max_level() {
                assert_eq!(
                    first.neighbours_at(id, layer),
                    second.neighbours_at(id, layer),
                    "point {id} layer {layer} must be identical across builds"
                );
            }
        }
    }

    #[test]
    fn insert_out_of_order_is_rejected() {
        let store = TestStore::synthetic(4, 4);
        let mut graph = NativeHnsw::new(params(), 1).expect("new");
        graph.insert(0, &store).expect("first row");
        let error = graph
            .insert(2, &store)
            .expect_err("skipping a row must be rejected");
        assert!(error.to_string().contains("row order"), "got: {error}");
    }

    #[test]
    fn degenerate_parameters_are_rejected() {
        for (m, m0, efc, efs) in [
            (0, 16, 64, 64),
            (8, 0, 64, 64),
            (8, 16, 0, 64),
            (8, 16, 64, 0),
        ] {
            let params = HnswParams {
                m,
                m0,
                ef_construction: efc,
                ef_search: efs,
            };
            assert!(
                NativeHnsw::new(params, 1).is_err(),
                "degenerate params {params:?} must be rejected"
            );
        }
    }

    #[test]
    fn sampled_levels_are_bounded_and_deterministic() {
        let sampler = LevelSampler::new(16, 12345);
        for id in 0..10_000u32 {
            let level = sampler.level_for(id);
            assert!(level < MAX_LEVEL, "level {level} for {id} exceeds the cap");
            assert_eq!(level, sampler.level_for(id), "sampling must be pure");
        }
    }
}
