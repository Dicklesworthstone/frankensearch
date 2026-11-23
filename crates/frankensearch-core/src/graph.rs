//! Optional document-graph model for graph-aware ranking extensions.
//!
//! This module intentionally provides only zero-dependency core types.
//! Fusion/ranking algorithms (e.g. query-biased `PageRank`) are implemented
//! in higher-level crates when graph features are enabled.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

/// Stable identifier type for graph nodes (documents).
pub type GraphDocId = String;

/// Relationship taxonomy between documents.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeType {
    Reference,
    CoLocation,
    Import,
    ThreadReply,
    Similar,
    Custom(String),
}

/// One directed graph edge from a source node to a neighbor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphEdge {
    pub neighbor_doc_id: GraphDocId,
    pub edge_type: EdgeType,
    pub weight: f32,
}

impl GraphEdge {
    #[must_use]
    pub const fn new(neighbor_doc_id: GraphDocId, edge_type: EdgeType, weight: f32) -> Self {
        Self {
            neighbor_doc_id,
            edge_type,
            weight,
        }
    }
}

/// Optional adjacency model supplied by consumers at index or query time.
///
/// `DocumentGraph` is intentionally lightweight and does not enforce
/// graph-theory invariants beyond stable node counting and deterministic
/// upsert behavior for identical `(neighbor, edge_type)` edges.
/// Deserialization restores every referenced endpoint and recomputes the
/// cached node count, while preserving the serialized adjacency lists.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct DocumentGraph {
    adjacency: HashMap<GraphDocId, Vec<GraphEdge>>,
    node_count: usize,
}

impl<'de> Deserialize<'de> for DocumentGraph {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Keep both the named and sequential serde representations unchanged.
        // The stored count is a cache, not authority for the imported node set.
        #[derive(Deserialize)]
        #[serde(rename = "DocumentGraph")]
        struct Fields {
            adjacency: HashMap<GraphDocId, Vec<GraphEdge>>,
            #[serde(rename = "node_count")]
            _node_count: usize,
        }

        let Fields { mut adjacency, .. } = Fields::deserialize(deserializer)?;
        // Rankers dense-index adjacency keys, so sinks referenced only by an
        // edge must become explicit keys just as they do through add_edge().
        // Deduplicate borrowed IDs before cloning: many edges may share a sink.
        let missing_nodes: HashSet<&str> = adjacency
            .values()
            .flatten()
            .map(|edge| edge.neighbor_doc_id.as_str())
            .filter(|id| !adjacency.contains_key(*id))
            .collect();
        let missing_nodes: Vec<GraphDocId> = missing_nodes.into_iter().map(str::to_owned).collect();
        for doc_id in missing_nodes {
            adjacency.entry(doc_id).or_default();
        }

        Ok(Self {
            node_count: adjacency.len(),
            adjacency,
        })
    }
}

impl DocumentGraph {
    /// Create an empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of unique nodes currently known to the graph.
    #[must_use]
    pub const fn node_count(&self) -> usize {
        self.node_count
    }

    /// Number of directed edges currently stored.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.adjacency
            .values()
            .map(std::vec::Vec::len)
            .sum::<usize>()
    }

    /// Whether the graph has zero nodes.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.node_count == 0
    }

    /// Whether a document node is present.
    #[must_use]
    pub fn contains_node(&self, doc_id: &str) -> bool {
        self.adjacency.contains_key(doc_id)
    }

    /// Insert a node if absent.
    pub fn add_node(&mut self, doc_id: impl Into<GraphDocId>) {
        let doc_id = doc_id.into();
        if let std::collections::hash_map::Entry::Vacant(entry) = self.adjacency.entry(doc_id) {
            entry.insert(Vec::new());
            self.node_count += 1;
        }
    }

    /// Insert or update a directed edge.
    ///
    /// If an edge with identical `(neighbor_doc_id, edge_type)` already
    /// exists for the same source node, only the `weight` is updated.
    pub fn add_edge(
        &mut self,
        from_doc_id: impl Into<GraphDocId>,
        to_doc_id: impl Into<GraphDocId>,
        edge_type: EdgeType,
        weight: f32,
    ) {
        let from_doc_id = from_doc_id.into();
        let to_doc_id = to_doc_id.into();

        self.add_node(to_doc_id.clone());

        let edges = match self.adjacency.entry(from_doc_id) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                self.node_count += 1;
                entry.insert(Vec::new())
            }
        };
        if let Some(existing) = edges
            .iter_mut()
            .find(|edge| edge.neighbor_doc_id == to_doc_id && edge.edge_type == edge_type)
        {
            existing.weight = weight;
            return;
        }
        edges.push(GraphEdge::new(to_doc_id, edge_type, weight));
    }

    /// Return neighbors for `doc_id` (empty when node is missing).
    #[must_use]
    pub fn neighbors(&self, doc_id: &str) -> &[GraphEdge] {
        self.adjacency
            .get(doc_id)
            .map_or(&[], std::vec::Vec::as_slice)
    }

    /// Borrow the underlying adjacency map.
    #[must_use]
    pub const fn adjacency(&self) -> &HashMap<GraphDocId, Vec<GraphEdge>> {
        &self.adjacency
    }
}

#[cfg(test)]
mod tests {
    use super::{DocumentGraph, EdgeType};

    #[test]
    fn add_edge_inserts_source_target_and_edge() {
        let mut graph = DocumentGraph::new();
        graph.add_edge("doc-a", "doc-b", EdgeType::Reference, 1.0);

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
        assert!(graph.contains_node("doc-a"));
        assert!(graph.contains_node("doc-b"));
        let neighbors = graph.neighbors("doc-a");
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].neighbor_doc_id, "doc-b");
        assert_eq!(neighbors[0].edge_type, EdgeType::Reference);
        assert!((neighbors[0].weight - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn add_edge_upserts_weight_for_same_neighbor_and_type() {
        let mut graph = DocumentGraph::new();
        graph.add_edge("doc-a", "doc-b", EdgeType::Reference, 1.0);
        graph.add_edge("doc-a", "doc-b", EdgeType::Reference, 0.25);

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
        let neighbors = graph.neighbors("doc-a");
        assert_eq!(neighbors.len(), 1);
        assert!((neighbors[0].weight - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn add_edge_keeps_distinct_edge_types() {
        let mut graph = DocumentGraph::new();
        graph.add_edge("doc-a", "doc-b", EdgeType::Reference, 1.0);
        graph.add_edge("doc-a", "doc-b", EdgeType::Import, 0.5);

        let neighbors = graph.neighbors("doc-a");
        assert_eq!(neighbors.len(), 2);
        assert!(
            neighbors
                .iter()
                .any(|edge| edge.edge_type == EdgeType::Reference)
        );
        assert!(
            neighbors
                .iter()
                .any(|edge| edge.edge_type == EdgeType::Import)
        );
    }

    #[test]
    fn unknown_node_has_empty_neighbors() {
        let graph = DocumentGraph::new();
        assert!(graph.neighbors("missing").is_empty());
    }

    #[test]
    fn custom_edge_type_round_trip_json() {
        let mut graph = DocumentGraph::new();
        graph.add_edge(
            "doc-a",
            "doc-b",
            EdgeType::Custom("semantic_link".to_string()),
            0.9,
        );

        let encoded = serde_json::to_string(&graph).expect("serialize graph");
        let decoded: DocumentGraph = serde_json::from_str(&encoded).expect("deserialize graph");
        assert_eq!(decoded, graph);
        let neighbors = decoded.neighbors("doc-a");
        assert_eq!(neighbors.len(), 1);
        assert_eq!(
            neighbors[0].edge_type,
            EdgeType::Custom("semantic_link".to_string())
        );
    }

    #[test]
    fn deserialize_recomputes_stale_node_counts() {
        for stored_count in [0, 1, 3, usize::MAX] {
            let encoded = serde_json::json!({
                "adjacency": {
                    "doc-a": [{
                        "neighbor_doc_id": "doc-b",
                        "edge_type": "reference",
                        "weight": 1.0
                    }],
                    "doc-b": []
                },
                "node_count": stored_count
            });
            let graph: DocumentGraph = serde_json::from_value(encoded).expect("import graph");

            assert_eq!(graph.node_count(), 2);
            assert!(!graph.is_empty());
            assert_eq!(graph.edge_count(), 1);
            assert!(graph.contains_node("doc-a"));
            assert!(graph.contains_node("doc-b"));
            assert_eq!(
                serde_json::to_value(&graph).expect("serialize repaired graph")["node_count"],
                2,
            );
        }
    }

    #[test]
    fn deserialize_restores_shared_endpoints_and_preserves_mutation_invariants() {
        let encoded = serde_json::json!({
            "adjacency": {
                "doc-a": [
                    {"neighbor_doc_id": "sink", "edge_type": "reference", "weight": 1.0},
                    {"neighbor_doc_id": "sink", "edge_type": "similar", "weight": 0.5},
                    {"neighbor_doc_id": "doc-a", "edge_type": "thread_reply", "weight": 0.25}
                ],
                "doc-b": [
                    {"neighbor_doc_id": "sink", "edge_type": "import", "weight": 0.75}
                ],
                "isolated": []
            },
            "node_count": 3
        });
        let mut graph: DocumentGraph = serde_json::from_value(encoded).expect("import graph");
        let mut expected = DocumentGraph::new();
        expected.add_edge("doc-a", "sink", EdgeType::Reference, 1.0);
        expected.add_edge("doc-a", "sink", EdgeType::Similar, 0.5);
        expected.add_edge("doc-a", "doc-a", EdgeType::ThreadReply, 0.25);
        expected.add_edge("doc-b", "sink", EdgeType::Import, 0.75);
        expected.add_node("isolated");

        assert_eq!(graph, expected);
        assert_eq!(graph.node_count(), 4);
        assert_eq!(graph.edge_count(), 4);
        assert!(graph.contains_node("sink"));
        assert!(graph.neighbors("sink").is_empty());
        let repaired = serde_json::to_value(&graph).expect("serialize repaired graph");
        assert_eq!(repaired["node_count"], 4);
        assert_eq!(repaired["adjacency"]["sink"], serde_json::json!([]));
        assert_eq!(
            serde_json::from_value::<DocumentGraph>(repaired).expect("reopen repaired graph"),
            graph,
        );

        graph.add_node("sink");
        assert_eq!(graph.node_count(), 4);
        graph.add_edge("sink", "new-doc", EdgeType::Reference, 1.0);
        assert_eq!(graph.node_count(), 5);
        graph.add_edge("sink", "new-doc", EdgeType::Reference, 0.5);
        assert_eq!(graph.node_count(), 5);
        assert_eq!(graph.edge_count(), 5);
        assert!((graph.neighbors("sink")[0].weight - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn deserialize_empty_graph_discards_stale_nonzero_count() {
        let graph: DocumentGraph = serde_json::from_value(serde_json::json!({
            "adjacency": {},
            "node_count": 42
        }))
        .expect("import empty graph");

        assert!(graph.is_empty());
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
        assert_eq!(graph, DocumentGraph::new());
    }

    #[test]
    fn deserialize_preserves_sequence_wire_representation() {
        let graph: DocumentGraph = serde_json::from_value(serde_json::json!([
            {"doc-a": []},
            1
        ]))
        .expect("deserialize the original two-field sequence");

        assert_eq!(graph.node_count(), 1);
        assert!(graph.contains_node("doc-a"));
        assert_eq!(
            serde_json::to_value(&graph).expect("serialize named fields"),
            serde_json::json!({"adjacency": {"doc-a": []}, "node_count": 1}),
        );
    }

    #[test]
    fn deserialize_preserves_required_field_and_edge_validation() {
        for encoded in [
            serde_json::json!({"adjacency": {}}),
            serde_json::json!({"node_count": 0}),
            serde_json::json!({"adjacency": {}, "node_count": -1}),
            serde_json::json!({"adjacency": {"doc-a": {}}, "node_count": 1}),
            serde_json::json!({
                "adjacency": {"doc-a": [{"edge_type": "reference", "weight": 1.0}]},
                "node_count": 1
            }),
        ] {
            assert!(serde_json::from_value::<DocumentGraph>(encoded).is_err());
        }
    }
}
