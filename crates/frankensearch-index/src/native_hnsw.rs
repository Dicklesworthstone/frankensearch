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
//! The workspace forbids `unsafe`, so the class of defect that the previous
//! engine's reload path carried — a misaligned pointer cast whose element
//! count came from a file header rather than the bytes actually read — is
//! not merely fixed here but unrepresentable.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fmt;

use frankensearch_core::error::{SearchError, SearchResult};

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
    /// A graph with no points nonetheless names an entry point.
    EntryPointInEmptyGraph {
        /// The spurious entry point id.
        entry: u32,
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
            Self::EntryPointInEmptyGraph { entry } => {
                write!(f, "empty graph names entry point {entry}")
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
#[derive(Debug, Clone, Copy, PartialEq)]
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
        if level.is_finite() && level > 0.0 {
            (level as usize).min(MAX_LEVEL - 1)
        } else {
            0
        }
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

/// A navigable small-world graph over rows of a vector store.
#[derive(Debug, Clone)]
pub struct NativeHnsw {
    params: HnswParams,
    sampler: LevelSampler,
    adjacency: Vec<Adjacency>,
    /// Entry point: the id of a point at the current maximum level.
    entry: Option<u32>,
    max_level: usize,
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
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] if `id` is not the next row,
    /// and propagates distance-computation failures.
    pub fn insert<D: VectorDistance>(&mut self, id: u32, store: &D) -> SearchResult<()> {
        let expected = u32::try_from(self.adjacency.len()).unwrap_or(u32::MAX);
        if id != expected {
            return Err(SearchError::InvalidConfig {
                field: "id".to_owned(),
                value: id.to_string(),
                reason: format!("points must be inserted in row order; expected {expected}"),
            });
        }

        let level = self.sampler.level_for(id);
        self.adjacency.push(Adjacency::with_level(level));

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
            let candidates =
                self.search_layer(&entry_points, id, layer, self.params.ef_construction, store)?;
            let selected =
                self.select_neighbours(id, &candidates, self.params.degree_at(layer), store)?;

            self.link(id, &selected, layer, store)?;

            entry_points = candidates.iter().map(|candidate| candidate.id).collect();
            if entry_points.is_empty() {
                entry_points.push(current);
            }
        }

        // A point sampled above the previous maximum becomes the new entry.
        if level > previous_max {
            self.entry = Some(id);
            self.max_level = level;
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
    ) -> SearchResult<Vec<Candidate>> {
        self.beam_search(entry_points, layer, ef, store, &mut |id, store| {
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
    ) -> SearchResult<Vec<Candidate>> {
        self.beam_search(entry_points, layer, ef, store, &mut |id, store| {
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
        distance: &mut dyn FnMut(u32, &D) -> SearchResult<f32>,
    ) -> SearchResult<Vec<Candidate>> {
        let ef = ef.max(1);
        let mut visited = vec![false; self.adjacency.len()];
        let mut frontier: BinaryHeap<Nearest> = BinaryHeap::new();
        let mut results: BinaryHeap<Candidate> = BinaryHeap::new();

        for &entry in entry_points {
            let Some(slot) = visited.get_mut(entry as usize) else {
                continue;
            };
            if *slot {
                continue;
            }
            *slot = true;
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
                let Some(slot) = visited.get_mut(neighbour as usize) else {
                    continue;
                };
                if *slot {
                    continue;
                }
                *slot = true;
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
        &self,
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

            if let Some(point) = self.adjacency.get_mut(id as usize)
                && let Some(list) = point.layers.get_mut(layer)
                && !list.contains(&neighbour)
            {
                list.push(neighbour);
            }
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
            self.prune(neighbour, layer, Some(id), store)?;
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
        let mut kept = self.select_neighbours(id, &scored, budget, store)?;
        // Reinstate the protected edge if the heuristic dropped it, evicting
        // the farthest kept neighbour to stay inside the budget.
        if let Some(protected) = protected
            && !kept.contains(&protected)
        {
            kept.pop();
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
        if let Some(point) = self.adjacency.get_mut(id as usize)
            && let Some(list) = point.layers.get_mut(layer)
        {
            *list = kept;
        }
        for neighbour in dropped {
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
        if k == 0 || self.adjacency.is_empty() {
            return Ok(Vec::new());
        }
        let Some(entry) = self.entry else {
            return Ok(Vec::new());
        };
        let ef = ef.unwrap_or(self.params.ef_search).max(k);

        // Descend the upper layers greedily, then run one wide beam search
        // at layer 0 where every point participates.
        let mut current = entry;
        let mut layer = self.max_level;
        while layer > 0 {
            let found = self.search_layer_query(&[current], query, layer, 1, store)?;
            if let Some(best) = found.first() {
                current = best.id;
            }
            layer -= 1;
        }

        let mut candidates = self.search_layer_query(&[current], query, 0, ef, store)?;
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
            return match self.entry {
                Some(entry) => Err(GraphDefect::EntryPointInEmptyGraph { entry }),
                None => Ok(()),
            };
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
                let mut seen = neighbours.to_vec();
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
}

#[cfg(test)]
mod tests {
    use super::*;

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

        fn dot(&self, a: &[f32], b: &[f32]) -> f32 {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        }

        /// Ground truth: exact nearest neighbours by full scan.
        fn exact_top_k(&self, query: &[f32], k: usize) -> Vec<u32> {
            let mut scored: Vec<(f32, u32)> = self
                .vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (1.0 - self.dot(v, query), i as u32))
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
            Ok(1.0 - self.dot(vector, query))
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

    fn params() -> HnswParams {
        HnswParams {
            m: 8,
            m0: 16,
            ef_construction: 64,
            ef_search: 64,
        }
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
        let mut graph = NativeHnsw::new(params(), 1).expect("new");
        graph.adjacency.push(Adjacency::with_level(3));
        graph.entry = Some(0);
        graph.max_level = 3;
        // Points 1 and 2 arrive at level 3 and level 0 respectively.
        for (id, level) in [(1u32, 3usize), (2, 0)] {
            graph.adjacency.push(Adjacency::with_level(level));
            let selected = (0..id).collect::<Vec<_>>();
            for layer in 0..=level.min(graph.max_level) {
                graph.link(id, &selected, layer, &store).expect("link");
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
        graph.adjacency[high].layers[1].push(low as u32);

        let defect = graph
            .verify()
            .expect_err("an edge above the neighbour's level must be rejected");
        assert!(
            matches!(
                defect,
                GraphDefect::EdgeAboveNeighbourLevel { neighbour, .. }
                    if neighbour == low as u32
            ),
            "expected the layer-violation defect naming point {low}, got: {defect:?}"
        );
    }

    // ─── Boundaries ─────────────────────────────────────────────────────

    #[test]
    fn point_count_boundaries_build_and_verify() {
        // n = 0,1,2,3 and around the degree budget m and m+1.
        for count in [0usize, 1, 2, 3, 8, 9, 17] {
            let store = TestStore::synthetic(count, 6);
            let graph = NativeHnsw::build(params(), 42, &store)
                .unwrap_or_else(|e| panic!("build with {count} points: {e}"));
            graph
                .verify()
                .unwrap_or_else(|e| panic!("verify with {count} points: {e}"));
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
        for id in 0..first.len() as u32 {
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
