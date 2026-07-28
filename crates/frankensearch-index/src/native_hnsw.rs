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
use std::path::Path;

use frankensearch_core::error::{SearchError, SearchResult};

const NATIVE_HNSW_MAGIC: [u8; 8] = *b"FSHNSW\0\0";
/// Current owned native-HNSW graph format.
pub const NATIVE_HNSW_FORMAT_VERSION: u32 = 1;
const NATIVE_HNSW_HEADER_LEN: usize = 96;
const NATIVE_HNSW_HEADER_LEN_U64: u64 = 96;
const NATIVE_HNSW_HEADER_CRC_OFFSET: usize = NATIVE_HNSW_HEADER_LEN - 4;
const NATIVE_HNSW_NO_ENTRY: u64 = u64::MAX;

/// A violation of the graph's structural invariants.
///
/// Structured rather than stringly so an attestation gate can branch on the
/// defect class, and so the regression fixtures below assert on the variant
/// instead of matching message text. The graph itself indexes rows of a
/// caller-owned store and owns no file, so this is deliberately not a
/// [`SearchError`]: the wiring layer, which knows the sidecar path, maps a
/// defect into `SearchError::IndexCorrupted`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphDefect {
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
        }
    }
}

impl std::error::Error for GraphDefect {}

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
pub struct NativeHnswFileMetadata {
    format_version: u32,
    byte_len: u64,
    point_count: u64,
    payload_crc32: u32,
    header_crc32: u32,
}

impl NativeHnswFileMetadata {
    /// Binary format version.
    #[must_use]
    pub const fn format_version(self) -> u32 {
        self.format_version
    }

    /// Complete file length.
    #[must_use]
    pub const fn byte_len(self) -> u64 {
        self.byte_len
    }

    /// Number of graph points encoded in the artifact.
    #[must_use]
    pub const fn point_count(self) -> u64 {
        self.point_count
    }

    /// CRC-32 of the canonical adjacency payload.
    #[must_use]
    pub const fn payload_crc32(self) -> u32 {
        self.payload_crc32
    }

    /// CRC-32 of the fixed header before its checksum field.
    #[must_use]
    pub const fn header_crc32(self) -> u32 {
        self.header_crc32
    }
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
pub struct LevelSampler {
    seed: u64,
    level_scale: f64,
}

impl LevelSampler {
    /// Create a sampler for the given degree and seed.
    #[must_use]
    pub fn new(m: usize, seed: u64) -> Self {
        // The standard 1/ln(M) scale; guard M == 1, whose ln is zero.
        let level_scale = if m <= 1 { 1.0 } else { 1.0 / (m as f64).ln() };
        Self { seed, level_scale }
    }

    /// Sample the level for `id`, in `0..MAX_LEVEL`.
    #[must_use]
    pub fn level_for(&self, id: u32) -> usize {
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
pub trait VectorDistance {
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

    /// Whether the store holds no rows.
    fn is_empty(&self) -> bool {
        self.len() == 0
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
pub struct NativeHnsw {
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
    pub fn new(params: HnswParams, seed: u64) -> SearchResult<Self> {
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
    pub fn len(&self) -> usize {
        self.adjacency.len()
    }

    /// Whether the graph indexes no points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.adjacency.is_empty()
    }

    /// The entry point id, if any point has been inserted.
    #[must_use]
    pub const fn entry_point(&self) -> Option<u32> {
        self.entry
    }

    /// The current maximum level.
    #[must_use]
    pub const fn max_level(&self) -> usize {
        self.max_level
    }

    /// Construction and search parameters persisted with this graph.
    #[must_use]
    pub const fn params(&self) -> HnswParams {
        self.params
    }

    /// Deterministic level-sampling seed persisted with this graph.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.sampler.seed
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
    pub fn save(&self, path: &Path) -> SearchResult<NativeHnswFileMetadata> {
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
    pub fn load<D: VectorDistance>(path: &Path, store: &D) -> SearchResult<Self> {
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
    /// Returns the same errors as [`Self::load`].
    pub fn load_with_metadata<D: VectorDistance>(
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
    pub fn build<D: VectorDistance>(
        params: HnswParams,
        seed: u64,
        store: &D,
    ) -> SearchResult<Self> {
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
    pub fn insert<D: VectorDistance>(&mut self, id: u32, store: &D) -> SearchResult<()> {
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

    /// How many neighbours `id` holds at `layer`.
    ///
    /// Exposed for diagnostics and structural attestation: degree
    /// distribution is the cheapest signal that a graph has degenerated
    /// (all-to-all, or starved) without running a recall measurement.
    #[must_use]
    pub fn neighbour_count(&self, id: u32, layer: usize) -> usize {
        self.neighbours_at(id, layer).len()
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
    pub fn search<D: VectorDistance>(
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
    pub fn verify(&self) -> Result<(), GraphDefect> {
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
    pub fn verify_for_store<D: VectorDistance>(&self, store: &D) -> Result<(), GraphDefect> {
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
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(SearchError::Io(error)),
    }
}

fn open_regular_file(path: &Path) -> SearchResult<File> {
    let path_metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
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
    use std::cell::Cell;

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
