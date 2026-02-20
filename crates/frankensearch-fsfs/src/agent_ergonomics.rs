//! Ultra-agent ergonomics for token-efficient CLI interactions.
//!
//! This module provides three ergonomic layers for AI agent workflows:
//!
//! 1. **Compact payload profile** — Reduced-verbosity output mode that strips
//!    optional metadata, abbreviates field names, and omits null/empty fields.
//!    Optimized for LLM token budgets (~30-50% reduction vs full JSON).
//!
//! 2. **Stable result IDs** — Deterministic, short identifiers for search
//!    results that persist across the lifetime of a stream or session. Agents
//!    can reference results by ID in follow-up commands (e.g., `explain R3`).
//!
//! 3. **Query templates** — Parameterized query patterns for common agent
//!    workflows (search-then-explain, incremental refinement, batch queries).

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::output_schema::{OutputEnvelope, OutputWarning};

// ─── Compact Mode ───────────────────────────────────────────────────────────

/// Compact mode verbosity level for agent-optimized output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactLevel {
    /// Full output with all fields (default).
    #[default]
    Full,
    /// Abbreviated field names, omit null/empty optional fields.
    Compact,
    /// Minimal: only essential fields (ok, data/error, id).
    Minimal,
}

impl fmt::Display for CompactLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Full => write!(f, "full"),
            Self::Compact => write!(f, "compact"),
            Self::Minimal => write!(f, "minimal"),
        }
    }
}

impl std::str::FromStr for CompactLevel {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "full" => Ok(Self::Full),
            "compact" => Ok(Self::Compact),
            "minimal" | "min" => Ok(Self::Minimal),
            _ => Err(()),
        }
    }
}

/// A compact search result for token-efficient agent output.
///
/// Field names are abbreviated: `id` (not `doc_id`), `s` (not `score`),
/// `r` (not `rank`), `snip` (not `snippet`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactHit {
    /// Stable result ID for follow-up commands (e.g., "R0", "R1").
    pub id: String,
    /// Document identifier.
    pub doc: String,
    /// Combined score.
    pub s: f64,
    /// Rank position (0-based).
    pub r: usize,
    /// Optional snippet (omitted if absent).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snip: Option<String>,
}

/// A compact search response wrapping results with minimal metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactSearchResponse {
    /// Result count.
    pub n: usize,
    /// Compact hit list.
    pub hits: Vec<CompactHit>,
    /// Duration in milliseconds (omitted in minimal mode).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ms: Option<u64>,
    /// Phase indicator: "fast" or "full".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
}

/// A compact error response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactError {
    /// Error code.
    pub code: String,
    /// Error message.
    pub msg: String,
    /// Exit code.
    pub exit: i32,
    /// Whether retryable.
    pub retry: bool,
}

/// A compact envelope wrapping either results or an error.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactEnvelope {
    /// Success indicator.
    pub ok: bool,
    /// Compact results (when ok == true).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<CompactSearchResponse>,
    /// Compact error (when ok == false).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub err: Option<CompactError>,
    /// Warnings (omitted when empty).
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub w: Vec<String>,
}

// ─── Stable Result IDs ──────────────────────────────────────────────────────

/// Prefix for stable result IDs.
pub const RESULT_ID_PREFIX: &str = "R";

/// Generate a stable result ID from a rank index.
///
/// Format: `R{rank}` where rank is 0-based.
/// Example: `R0`, `R1`, `R42`.
#[must_use]
pub fn result_id(rank: usize) -> String {
    format!("{RESULT_ID_PREFIX}{rank}")
}

/// Parse a stable result ID back to a rank index.
///
/// Returns `None` if the ID doesn't match the `R{n}` pattern.
#[must_use]
pub fn parse_result_id(id: &str) -> Option<usize> {
    id.strip_prefix(RESULT_ID_PREFIX)
        .and_then(|n| n.parse::<usize>().ok())
}

/// A registry that maps stable result IDs to document identifiers within
/// a session or stream.
#[derive(Debug, Clone, Default)]
pub struct ResultIdRegistry {
    entries: Vec<ResultIdEntry>,
}

/// A single entry in the result ID registry.
#[derive(Debug, Clone, PartialEq)]
pub struct ResultIdEntry {
    /// Stable result ID (e.g., "R0").
    pub result_id: String,
    /// Document identifier.
    pub doc_id: String,
    /// Combined score at time of registration.
    pub score: f64,
}

impl ResultIdRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a batch of results, assigning stable IDs.
    ///
    /// IDs are assigned sequentially starting from the current registry size.
    pub fn register_batch(&mut self, docs: &[(String, f64)]) -> Vec<String> {
        let start = self.entries.len();
        docs.iter()
            .enumerate()
            .map(|(i, (doc_id, score))| {
                let id = result_id(start + i);
                self.entries.push(ResultIdEntry {
                    result_id: id.clone(),
                    doc_id: doc_id.clone(),
                    score: *score,
                });
                id
            })
            .collect()
    }

    /// Look up a document ID by stable result ID.
    #[must_use]
    pub fn resolve(&self, id: &str) -> Option<&ResultIdEntry> {
        let rank = parse_result_id(id)?;
        self.entries.get(rank)
    }

    /// Number of registered results.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the registry is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// All registered entries.
    #[must_use]
    pub fn entries(&self) -> &[ResultIdEntry] {
        &self.entries
    }
}

// ─── Query Templates ────────────────────────────────────────────────────────

/// Schema version for query templates.
pub const QUERY_TEMPLATE_VERSION: &str = "fsfs.template.v1";

/// A parameterized query template for common agent workflows.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryTemplate {
    /// Template name (e.g., `search_then_explain`).
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Ordered steps in the template.
    pub steps: Vec<TemplateStep>,
}

/// A single step in a query template.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemplateStep {
    /// Step command (e.g., "search", "explain").
    pub command: String,
    /// Parameter placeholders with default values.
    pub params: BTreeMap<String, TemplateParam>,
    /// Whether this step depends on the previous step's output.
    pub depends_on_previous: bool,
}

/// A template parameter with type information and defaults.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemplateParam {
    /// Parameter description.
    pub description: String,
    /// Default value (if any).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<String>,
    /// Whether this parameter is required.
    pub required: bool,
}

/// Built-in query templates for common agent workflows.
#[must_use]
pub fn builtin_templates() -> Vec<QueryTemplate> {
    vec![
        search_then_explain_template(),
        incremental_refinement_template(),
        batch_search_template(),
    ]
}

fn search_then_explain_template() -> QueryTemplate {
    QueryTemplate {
        name: "search_then_explain".into(),
        description: "Search for documents, then explain the top result's ranking.".into(),
        steps: vec![
            TemplateStep {
                command: "search".into(),
                params: BTreeMap::from([
                    (
                        "query".into(),
                        TemplateParam {
                            description: "Search query text".into(),
                            default: None,
                            required: true,
                        },
                    ),
                    (
                        "limit".into(),
                        TemplateParam {
                            description: "Maximum results to return".into(),
                            default: Some("10".into()),
                            required: false,
                        },
                    ),
                ]),
                depends_on_previous: false,
            },
            TemplateStep {
                command: "explain".into(),
                params: BTreeMap::from([(
                    "result_id".into(),
                    TemplateParam {
                        description: "Stable result ID from search (e.g., R0)".into(),
                        default: Some("R0".into()),
                        required: true,
                    },
                )]),
                depends_on_previous: true,
            },
        ],
    }
}

fn incremental_refinement_template() -> QueryTemplate {
    QueryTemplate {
        name: "incremental_refinement".into(),
        description: "Search with fast phase, optionally wait for refined results.".into(),
        steps: vec![
            TemplateStep {
                command: "search".into(),
                params: BTreeMap::from([
                    (
                        "query".into(),
                        TemplateParam {
                            description: "Search query text".into(),
                            default: None,
                            required: true,
                        },
                    ),
                    (
                        "stream".into(),
                        TemplateParam {
                            description: "Enable streaming mode for progressive results".into(),
                            default: Some("true".into()),
                            required: false,
                        },
                    ),
                ]),
                depends_on_previous: false,
            },
            TemplateStep {
                command: "search".into(),
                params: BTreeMap::from([
                    (
                        "query".into(),
                        TemplateParam {
                            description: "Same query (refined phase)".into(),
                            default: None,
                            required: true,
                        },
                    ),
                    (
                        "fast_only".into(),
                        TemplateParam {
                            description: "Set to false to get quality-refined results".into(),
                            default: Some("false".into()),
                            required: false,
                        },
                    ),
                ]),
                depends_on_previous: true,
            },
        ],
    }
}

fn batch_search_template() -> QueryTemplate {
    QueryTemplate {
        name: "batch_search".into(),
        description: "Execute multiple independent queries in sequence.".into(),
        steps: vec![TemplateStep {
            command: "search".into(),
            params: BTreeMap::from([
                (
                    "queries".into(),
                    TemplateParam {
                        description: "Comma-separated list of queries to execute".into(),
                        default: None,
                        required: true,
                    },
                ),
                (
                    "limit".into(),
                    TemplateParam {
                        description: "Maximum results per query".into(),
                        default: Some("5".into()),
                        required: false,
                    },
                ),
            ]),
            depends_on_previous: false,
        }],
    }
}

// ─── Compact Conversion ─────────────────────────────────────────────────────

/// Convert a full output envelope with search results into a compact envelope.
///
/// The `hits_extractor` maps the typed data payload to a list of
/// `(doc_id, score, rank, snippet)` tuples.
#[must_use]
pub fn compactify<T>(
    envelope: &OutputEnvelope<T>,
    compact_level: CompactLevel,
    registry: &mut ResultIdRegistry,
    hits_extractor: impl Fn(&T) -> Vec<(String, f64, usize, Option<String>)>,
) -> CompactEnvelope {
    if !envelope.ok {
        let err = envelope.error.as_ref().map(|e| CompactError {
            code: e.code.clone(),
            msg: e.message.clone(),
            exit: e.exit_code,
            retry: is_retryable_code(&e.code),
        });
        return CompactEnvelope {
            ok: false,
            data: None,
            err,
            w: compact_warnings(&envelope.warnings),
        };
    }

    let raw_hits: Vec<(String, f64, usize, Option<String>)> = envelope
        .data
        .as_ref()
        .map(hits_extractor)
        .unwrap_or_default();

    // Register hits and assign stable IDs
    let ids = registry.register_batch(
        &raw_hits
            .iter()
            .map(|(doc, score, _, _)| (doc.clone(), *score))
            .collect::<Vec<_>>(),
    );

    let hits: Vec<CompactHit> = raw_hits
        .into_iter()
        .zip(ids)
        .map(|((doc, score, rank, snippet), id)| {
            let snip = match compact_level {
                CompactLevel::Minimal => None,
                _ => snippet,
            };
            CompactHit {
                id,
                doc,
                s: score,
                r: rank,
                snip,
            }
        })
        .collect();

    let ms = match compact_level {
        CompactLevel::Minimal => None,
        _ => envelope.meta.duration_ms,
    };

    CompactEnvelope {
        ok: true,
        data: Some(CompactSearchResponse {
            n: hits.len(),
            hits,
            ms,
            phase: None,
        }),
        err: None,
        w: compact_warnings(&envelope.warnings),
    }
}

fn compact_warnings(warnings: &[OutputWarning]) -> Vec<String> {
    warnings.iter().map(|w| w.code.clone()).collect()
}

fn is_retryable_code(code: &str) -> bool {
    matches!(
        code,
        "embedding_failed"
            | "search_timeout"
            | "queue_full"
            | "reranker_unavailable"
            | "rerank_failed"
            | "io_error"
            | "subsystem_error"
    )
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output_schema::{
        OutputEnvelope, OutputError, OutputErrorCode, OutputMeta, OutputWarningCode,
    };

    fn sample_ts() -> &'static str {
        "2026-02-14T12:00:00Z"
    }

    // ─── Compact Level ────────────────────────────────────────────────

    #[test]
    fn compact_level_display_and_parse() {
        for (expected, text) in &[
            (CompactLevel::Full, "full"),
            (CompactLevel::Compact, "compact"),
            (CompactLevel::Minimal, "minimal"),
        ] {
            assert_eq!(expected.to_string(), *text);
            assert_eq!(text.parse::<CompactLevel>().unwrap(), *expected);
        }
        assert_eq!(
            "min".parse::<CompactLevel>().unwrap(),
            CompactLevel::Minimal
        );
        assert!("unknown".parse::<CompactLevel>().is_err());
    }

    // ─── Stable Result IDs ────────────────────────────────────────────

    #[test]
    fn result_id_generation() {
        assert_eq!(result_id(0), "R0");
        assert_eq!(result_id(42), "R42");
        assert_eq!(result_id(999), "R999");
    }

    #[test]
    fn result_id_parsing() {
        assert_eq!(parse_result_id("R0"), Some(0));
        assert_eq!(parse_result_id("R42"), Some(42));
        assert_eq!(parse_result_id("R999"), Some(999));
        assert_eq!(parse_result_id("X0"), None);
        assert_eq!(parse_result_id("R"), None);
        assert_eq!(parse_result_id("Rabc"), None);
        assert_eq!(parse_result_id(""), None);
    }

    #[test]
    fn result_id_registry_basics() {
        let mut registry = ResultIdRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);

        let ids = registry.register_batch(&[
            ("doc-a".into(), 0.95),
            ("doc-b".into(), 0.87),
            ("doc-c".into(), 0.72),
        ]);

        assert_eq!(ids, vec!["R0", "R1", "R2"]);
        assert_eq!(registry.len(), 3);
        assert!(!registry.is_empty());
    }

    #[test]
    fn result_id_registry_resolve() {
        let mut registry = ResultIdRegistry::new();
        registry.register_batch(&[("doc-a".into(), 0.95), ("doc-b".into(), 0.87)]);

        let entry = registry.resolve("R0").unwrap();
        assert_eq!(entry.doc_id, "doc-a");
        assert!((entry.score - 0.95).abs() < f64::EPSILON);

        let entry = registry.resolve("R1").unwrap();
        assert_eq!(entry.doc_id, "doc-b");

        assert!(registry.resolve("R2").is_none());
        assert!(registry.resolve("X0").is_none());
    }

    #[test]
    fn result_id_registry_sequential_batches() {
        let mut registry = ResultIdRegistry::new();

        let ids1 = registry.register_batch(&[("doc-a".into(), 0.95)]);
        assert_eq!(ids1, vec!["R0"]);

        let ids2 = registry.register_batch(&[("doc-b".into(), 0.80), ("doc-c".into(), 0.70)]);
        assert_eq!(ids2, vec!["R1", "R2"]);

        assert_eq!(registry.len(), 3);
        assert_eq!(registry.resolve("R2").unwrap().doc_id, "doc-c");
    }

    // ─── Query Templates ──────────────────────────────────────────────

    #[test]
    fn builtin_templates_are_well_formed() {
        let templates = builtin_templates();
        assert_eq!(templates.len(), 3);

        for template in &templates {
            assert!(!template.name.is_empty());
            assert!(!template.description.is_empty());
            assert!(!template.steps.is_empty());
            for step in &template.steps {
                assert!(!step.command.is_empty());
            }
        }
    }

    #[test]
    fn search_then_explain_template_structure() {
        let templates = builtin_templates();
        let t = templates
            .iter()
            .find(|t| t.name == "search_then_explain")
            .unwrap();
        assert_eq!(t.steps.len(), 2);
        assert_eq!(t.steps[0].command, "search");
        assert!(!t.steps[0].depends_on_previous);
        assert_eq!(t.steps[1].command, "explain");
        assert!(t.steps[1].depends_on_previous);
        assert!(t.steps[1].params.contains_key("result_id"));
    }

    #[test]
    fn templates_serialize_roundtrip() {
        let templates = builtin_templates();
        let json = serde_json::to_string(&templates).unwrap();
        let decoded: Vec<QueryTemplate> = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, templates);
    }

    // ─── Compact Conversion ───────────────────────────────────────────

    #[test]
    fn compactify_success_envelope() {
        let meta = OutputMeta::new("search", "json").with_duration_ms(42);
        let env = OutputEnvelope::success(
            vec![("doc-a".to_string(), 0.95), ("doc-b".to_string(), 0.80)],
            meta,
            sample_ts(),
        );

        let mut registry = ResultIdRegistry::new();
        let compact = compactify(&env, CompactLevel::Compact, &mut registry, |data| {
            data.iter()
                .enumerate()
                .map(|(i, (doc, score))| (doc.clone(), *score, i, None))
                .collect()
        });

        assert!(compact.ok);
        assert!(compact.err.is_none());
        let data = compact.data.unwrap();
        assert_eq!(data.n, 2);
        assert_eq!(data.hits[0].id, "R0");
        assert_eq!(data.hits[0].doc, "doc-a");
        assert!((data.hits[0].s - 0.95).abs() < f64::EPSILON);
        assert_eq!(data.hits[1].id, "R1");
        assert_eq!(data.ms, Some(42));
    }

    #[test]
    fn compactify_minimal_strips_optional_fields() {
        let meta = OutputMeta::new("search", "json").with_duration_ms(42);
        let env = OutputEnvelope::success(vec![("doc-a".to_string(), 0.95)], meta, sample_ts());

        let mut registry = ResultIdRegistry::new();
        let compact = compactify(&env, CompactLevel::Minimal, &mut registry, |data| {
            data.iter()
                .enumerate()
                .map(|(i, (doc, score))| (doc.clone(), *score, i, Some("snippet".into())))
                .collect()
        });

        let data = compact.data.unwrap();
        // Minimal mode strips snippets and duration
        assert!(data.hits[0].snip.is_none());
        assert!(data.ms.is_none());
    }

    #[test]
    fn compactify_error_envelope() {
        let err = OutputError::new(OutputErrorCode::SEARCH_TIMEOUT, "timeout after 50ms", 1);
        let env: OutputEnvelope<Vec<(String, f64)>> =
            OutputEnvelope::error(err, OutputMeta::new("search", "json"), sample_ts());

        let mut registry = ResultIdRegistry::new();
        let compact = compactify(&env, CompactLevel::Compact, &mut registry, |_| Vec::new());

        assert!(!compact.ok);
        assert!(compact.data.is_none());
        let err = compact.err.unwrap();
        assert_eq!(err.code, "search_timeout");
        assert!(err.retry); // search_timeout is retryable
        assert_eq!(err.exit, 1);
    }

    #[test]
    fn compactify_with_warnings() {
        let env = OutputEnvelope::success(
            Vec::<(String, f64)>::new(),
            OutputMeta::new("search", "json"),
            sample_ts(),
        )
        .with_warnings(vec![
            OutputWarning::new(OutputWarningCode::DEGRADED_MODE, "quality skipped"),
            OutputWarning::new(OutputWarningCode::FAST_ONLY_RESULTS, "fast only"),
        ]);

        let mut registry = ResultIdRegistry::new();
        let compact = compactify(&env, CompactLevel::Compact, &mut registry, |_| Vec::new());

        assert_eq!(compact.w, vec!["degraded_mode", "fast_only_results"]);
    }

    #[test]
    fn compactify_preserves_registry_state() {
        let meta = OutputMeta::new("search", "json");
        let env1 =
            OutputEnvelope::success(vec![("doc-a".to_string(), 0.95)], meta.clone(), sample_ts());
        let env2 = OutputEnvelope::success(vec![("doc-b".to_string(), 0.80)], meta, sample_ts());

        let mut registry = ResultIdRegistry::new();
        let extractor = |data: &Vec<(String, f64)>| {
            data.iter()
                .enumerate()
                .map(|(i, (doc, score))| (doc.clone(), *score, i, None))
                .collect()
        };

        let _ = compactify(&env1, CompactLevel::Compact, &mut registry, extractor);
        let _ = compactify(&env2, CompactLevel::Compact, &mut registry, extractor);

        // Registry accumulated IDs across both calls
        assert_eq!(registry.len(), 2);
        assert_eq!(registry.resolve("R0").unwrap().doc_id, "doc-a");
        assert_eq!(registry.resolve("R1").unwrap().doc_id, "doc-b");
    }

    // ─── Retryability ─────────────────────────────────────────────────

    #[test]
    fn retryable_codes() {
        assert!(is_retryable_code("embedding_failed"));
        assert!(is_retryable_code("search_timeout"));
        assert!(is_retryable_code("queue_full"));
        assert!(is_retryable_code("io_error"));
        assert!(!is_retryable_code("invalid_config"));
        assert!(!is_retryable_code("index_corrupted"));
        assert!(!is_retryable_code("unknown"));
    }

    // ─── CompactHit Serialization ─────────────────────────────────────

    #[test]
    fn compact_hit_omits_null_snippet() {
        let hit = CompactHit {
            id: "R0".into(),
            doc: "doc-a".into(),
            s: 0.95,
            r: 0,
            snip: None,
        };
        let json = serde_json::to_string(&hit).unwrap();
        assert!(!json.contains("snip"));

        let hit_with_snip = CompactHit {
            id: "R1".into(),
            doc: "doc-b".into(),
            s: 0.87,
            r: 1,
            snip: Some("matched text".into()),
        };
        let json = serde_json::to_string(&hit_with_snip).unwrap();
        assert!(json.contains("\"snip\":\"matched text\""));
    }

    #[test]
    fn compact_envelope_serialization() {
        let envelope = CompactEnvelope {
            ok: true,
            data: Some(CompactSearchResponse {
                n: 1,
                hits: vec![CompactHit {
                    id: "R0".into(),
                    doc: "doc-a".into(),
                    s: 0.95,
                    r: 0,
                    snip: None,
                }],
                ms: Some(15),
                phase: Some("fast".into()),
            }),
            err: None,
            w: Vec::new(),
        };

        let json = serde_json::to_string(&envelope).unwrap();
        // Error and warnings should be omitted when absent/empty
        assert!(!json.contains("\"err\""));
        assert!(!json.contains("\"w\""));
        assert!(json.contains("\"ok\":true"));
        assert!(json.contains("\"id\":\"R0\""));
    }
}
