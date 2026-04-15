use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SnippetPolicy {
    pub max_snippets_per_result: u32,
    pub context_lines_before: u32,
    pub context_lines_after: u32,
    pub max_chars_per_snippet: u32,
    pub truncation_strategy: String,
    pub binary_fallback: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HighlightPolicy {
    pub offset_unit: String,
    pub enforce_grapheme_boundaries: bool,
    pub merge_overlapping_ranges: bool,
    pub max_highlights_per_snippet: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProvenancePolicy {
    pub required_fields: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DiagnosticsPolicy {
    pub required_fields: Vec<String>,
    pub reason_code_prefixes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PerformanceTargets {
    pub max_extraction_ms_per_result: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SnippetHighlightProvenanceContractDefinition {
    pub kind: String, // "fsfs_snippet_highlight_provenance_contract_definition"
    pub v: u32,       // 1
    pub snippet_policy: SnippetPolicy,
    pub highlight_policy: HighlightPolicy,
    pub provenance_policy: ProvenancePolicy,
    pub diagnostics_policy: DiagnosticsPolicy,
    pub performance_targets: PerformanceTargets,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HighlightRange {
    pub start_byte: u64,
    pub end_byte: u64,
    pub highlight_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query_term: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SnippetSegment {
    pub segment_id: String,
    pub text: String,
    pub line_range: [u32; 2],
    pub byte_range: [u64; 2],
    pub truncated: bool,
    pub highlights: Vec<HighlightRange>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScoreContributor {
    pub name: String,
    pub weight: f64,
    pub raw_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Provenance {
    pub path: String,
    pub segment_id: String,
    pub index_revision: u64,
    pub content_hash: String,
    pub score_contributors: Vec<ScoreContributor>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Diagnostics {
    pub event: String,
    pub reason_code: String,
    pub unicode_safe: bool,
    pub offsets_verified: bool,
    pub emitted_fields: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DegradedInfo {
    pub applied: bool,
    pub mode: String,
    pub fallback_action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SnippetHighlightProvenanceDecision {
    pub kind: String, // "fsfs_snippet_highlight_provenance_decision"
    pub v: u32,       // 1
    pub trace_id: String,
    pub doc_id: String,
    pub path: String,
    pub render_mode: String,
    pub snippet_status: String,
    pub snippet_segments: Vec<SnippetSegment>,
    pub provenance: Provenance,
    pub diagnostics: Diagnostics,
    pub degraded: DegradedInfo,
}
