//! Optional document-graph model for graph-aware ranking extensions.
//!
//! This module intentionally provides only zero-dependency core types.
//! Fusion/ranking algorithms (e.g. query-biased `PageRank`) are implemented
//! in higher-level crates when graph features are enabled.

use std::collections::HashMap;
use std::collections::hash_map::Entry;

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
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DocumentGraph {
    adjacency: HashMap<GraphDocId, Vec<GraphEdge>>,
    node_count: usize,
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
        let neighbors = decoded.neighbors("doc-a");
        assert_eq!(neighbors.len(), 1);
        assert_eq!(
            neighbors[0].edge_type,
            EdgeType::Custom("semantic_link".to_string())
        );
    }
}
