//! Engine-neutral query trees and native default/CASS lenient parsers.
//!
//! Parsing is deliberately separated from scorer construction. The tree uses
//! stable Quill field identifiers, never Tantivy field handles, and malformed
//! user input is recovered or dropped with diagnostics rather than promoted to
//! a search error.

use std::fmt;
use std::ops::Bound;

use thiserror::Error;

use crate::schema::{Analyzer as AnalyzerKind, FieldDescriptor, FieldKind, SchemaDescriptor};
use crate::scribe::{FrankensearchTokenizer, analyze_admitted, is_cass_cjk};

/// Maximum number of Unicode scalar values admitted to either native parser.
pub const MAX_QUERY_LENGTH: usize = 10_000;

/// Maximum recursive group depth accepted by the lenient parser.
///
/// The input-length limit already bounds work; this separate cap prevents a
/// hostile parenthesis run from turning into unbounded native-stack growth.
pub const MAX_QUERY_DEPTH: usize = 64;

const CONTENT_FIELD_NAME: &str = "content";
const TITLE_FIELD_NAME: &str = "title";
const TITLE_BOOST: f32 = 2.0;

/// One queryable field and its parser-time score multiplier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QueryField {
    /// Stable schema-local field identifier.
    pub field_id: u16,
    /// Multiplicative score boost applied when this branch is lowered.
    pub boost: f32,
}

impl QueryField {
    /// Construct a field target.
    #[must_use]
    pub const fn new(field_id: u16, boost: f32) -> Self {
        Self { field_id, boost }
    }
}

/// One normalized phrase term with its analyzer position.
///
/// Positions are retained even when an oversized predecessor was rejected, so
/// lowerers cannot accidentally turn a gapped phrase into an adjacent phrase.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PositionedTerm {
    /// Logical analyzer position.
    pub position: u32,
    /// Normalized term text.
    pub text: String,
}

/// One schema-typed scalar used by numeric ranges and sets.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum QueryValue {
    /// Signed 64-bit value.
    I64(i64),
    /// Unsigned 64-bit value.
    U64(u64),
    /// Exact field-typed text value after analyzer normalization.
    Str(String),
}

impl PositionedTerm {
    /// Construct a positioned phrase term.
    #[must_use]
    pub fn new(position: u32, text: impl Into<String>) -> Self {
        Self {
            position,
            text: text.into(),
        }
    }
}

/// Clause occurrence shared by query planning and scorer lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Occur {
    /// The child must match.
    Must,
    /// The child is optional unless the Boolean node has no required child.
    Should,
    /// The child must not match and contributes no score.
    MustNot,
}

/// Explicit source operator retained for fixture explanations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BooleanOperator {
    /// Explicit conjunction.
    And,
    /// Explicit disjunction.
    Or,
}

/// One owned child in a Boolean query.
#[derive(Debug, Clone, PartialEq)]
pub struct BooleanClause {
    /// Match occurrence.
    pub occur: Occur,
    /// Child query.
    pub query: Query,
}

impl BooleanClause {
    /// Construct a Boolean child.
    #[must_use]
    pub fn new(occur: Occur, query: Query) -> Self {
        Self { occur, query }
    }
}

/// Engine-neutral lexical query tree.
///
/// Text and phrase leaves may target several fields when analysis produced the
/// same token sequence for each field. This preserves the default parser's
/// ordered `[content, title]` expansion without smuggling backend handles into
/// the durable query boundary.
#[derive(Debug, Clone, PartialEq)]
pub enum Query {
    /// Match nothing.
    Empty,
    /// Match every live document.
    All,
    /// One analyzed term over one or more fields.
    Term {
        /// Ordered field expansion.
        fields: Vec<QueryField>,
        /// Normalized term text.
        text: String,
    },
    /// Exact-position phrase over one or more fields.
    Phrase {
        /// Ordered field expansion.
        fields: Vec<QueryField>,
        /// Normalized terms in phrase order.
        terms: Vec<PositionedTerm>,
        /// Permitted positional slop.
        slop: u32,
        /// Whether the last term is a prefix.
        prefix: bool,
    },
    /// Boolean combination.
    Boolean {
        /// Stable construction-order children.
        clauses: Vec<BooleanClause>,
        /// Explicit source operator, when one was present.
        operator: Option<BooleanOperator>,
    },
    /// Inclusive/exclusive signed numeric range.
    Range {
        /// Stable numeric field identifier.
        field_id: u16,
        /// Lower bound.
        lower: Bound<QueryValue>,
        /// Upper bound.
        upper: Bound<QueryValue>,
    },
    /// Exact set membership over one typed field.
    Set {
        /// Stable field identifier.
        field_id: u16,
        /// Values in stable source order after exact duplicate removal.
        values: Vec<QueryValue>,
    },
    /// Glob pattern over an ordered set of fields.
    Glob {
        /// Stable field identifiers.
        field_ids: Vec<u16>,
        /// Lowercased glob pattern in source syntax.
        pattern: String,
    },
    /// Explicit score multiplier.
    Boost {
        /// Wrapped query.
        query: Box<Self>,
        /// Finite, non-negative multiplier.
        factor: f32,
    },
}

impl Query {
    /// Whether this tree is the match-none sentinel.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }
}

/// Outcome of one [`canonicalize_query`] pass (bd-quill-duel-ast-dedup-7hsu).
///
/// Every count is score-neutral by construction: scoring clauses
/// (`Term`/`Phrase`/scoring `Must`/`Should`) are never removed, so oracle
/// conformance on duplicate-term queries is preserved exactly.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct QueryCanonicalizationReport {
    /// Exact-duplicate `MustNot` clauses removed (exclusion is idempotent).
    pub must_not_duplicates_removed: usize,
    /// Reserved for score-neutral positive-filter cleanup. Positive branches
    /// are currently all retained because ancestor boosts affect their score.
    pub filter_duplicates_removed: usize,
    /// Reserved for score-neutral field expansion cleanup. Duplicate scoring
    /// field branches are intentionally retained, so this is currently zero.
    pub duplicate_fields_removed: usize,
    /// Boolean nodes whose occurrence groups were put into the deterministic
    /// `Must`, `Should`, `MustNot` order while preserving construction order
    /// within every group.
    pub boolean_levels_sorted: usize,
    /// Glob nodes whose field list was sorted while retaining duplicates.
    pub glob_fields_canonicalized: usize,
}

/// Rank used to group canonical Boolean children: required, optional,
/// excluded. Within one group, construction order is preserved.
const fn occur_rank(occur: Occur) -> u8 {
    match occur {
        Occur::Must => 0,
        Occur::Should => 1,
        Occur::MustNot => 2,
    }
}

fn push_framed_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&bytes.len().to_be_bytes());
    out.extend_from_slice(bytes);
}

fn push_query_field_key(out: &mut Vec<u8>, field: QueryField) {
    out.extend_from_slice(&field.field_id.to_be_bytes());
    out.extend_from_slice(&field.boost.to_bits().to_be_bytes());
}

fn push_query_value_key(out: &mut Vec<u8>, value: &QueryValue) {
    match value {
        QueryValue::I64(v) => {
            out.push(0);
            out.extend_from_slice(&v.to_be_bytes());
        }
        QueryValue::U64(v) => {
            out.push(1);
            out.extend_from_slice(&v.to_be_bytes());
        }
        QueryValue::Str(text) => {
            out.push(2);
            push_framed_bytes(out, text.as_bytes());
        }
    }
}

fn push_bound_key(out: &mut Vec<u8>, bound: &Bound<QueryValue>) {
    match bound {
        Bound::Included(value) => {
            out.push(0);
            push_query_value_key(out, value);
        }
        Bound::Excluded(value) => {
            out.push(1);
            push_query_value_key(out, value);
        }
        Bound::Unbounded => out.push(2),
    }
}

/// Total deterministic structural key for one query subtree. Two subtrees
/// share a key iff they are structurally identical (field ids, boost bits,
/// terms, positions, bounds, values, patterns, and child order-sensitive
/// structure), so the key is safe both for deterministic ordering and for
/// exact-duplicate detection.
fn canonical_sort_key(query: &Query) -> Vec<u8> {
    let mut out = Vec::new();
    match query {
        Query::Empty => out.push(0),
        Query::All => out.push(1),
        Query::Term { fields, text } => {
            out.push(2);
            out.extend_from_slice(&fields.len().to_be_bytes());
            for field in fields {
                push_query_field_key(&mut out, *field);
            }
            push_framed_bytes(&mut out, text.as_bytes());
        }
        Query::Phrase {
            fields,
            terms,
            slop,
            prefix,
        } => {
            out.push(3);
            out.extend_from_slice(&fields.len().to_be_bytes());
            for field in fields {
                push_query_field_key(&mut out, *field);
            }
            out.extend_from_slice(&terms.len().to_be_bytes());
            for term in terms {
                out.extend_from_slice(&term.position.to_be_bytes());
                push_framed_bytes(&mut out, term.text.as_bytes());
            }
            out.extend_from_slice(&slop.to_be_bytes());
            out.push(u8::from(*prefix));
        }
        Query::Boolean { clauses, operator } => {
            out.push(4);
            out.push(match operator {
                None => 0,
                Some(BooleanOperator::And) => 1,
                Some(BooleanOperator::Or) => 2,
            });
            out.extend_from_slice(&clauses.len().to_be_bytes());
            for clause in clauses {
                out.push(occur_rank(clause.occur));
                let child = canonical_sort_key(&clause.query);
                push_framed_bytes(&mut out, &child);
            }
        }
        Query::Range {
            field_id,
            lower,
            upper,
        } => {
            out.push(5);
            out.extend_from_slice(&field_id.to_be_bytes());
            push_bound_key(&mut out, lower);
            push_bound_key(&mut out, upper);
        }
        Query::Set { field_id, values } => {
            out.push(6);
            out.extend_from_slice(&field_id.to_be_bytes());
            out.extend_from_slice(&values.len().to_be_bytes());
            for value in values {
                push_query_value_key(&mut out, value);
            }
        }
        Query::Glob { field_ids, pattern } => {
            out.push(7);
            out.extend_from_slice(&field_ids.len().to_be_bytes());
            for field_id in field_ids {
                out.extend_from_slice(&field_id.to_be_bytes());
            }
            push_framed_bytes(&mut out, pattern.as_bytes());
        }
        Query::Boost { query, factor } => {
            out.push(8);
            out.extend_from_slice(&factor.to_bits().to_be_bytes());
            let child = canonical_sort_key(query);
            push_framed_bytes(&mut out, &child);
        }
    }
    out
}

/// Canonicalize a parsed query tree for deterministic planning and
/// score-neutral deduplication (bd-quill-duel-ast-dedup-7hsu).
///
/// The pass is recursive (bottom-up) and applies three score-neutral rules:
///
/// 1. **Deterministic occurrence groups**: Boolean children are stably grouped
///    as `Must`, `Should`, `MustNot`. Construction order within each group is
///    retained because `f32` score accumulation is not associative and the
///    oracle exposes that order in exact score bits.
/// 2. **`MustNot` dedup**: exact-duplicate exclusion clauses collapse to
///    one; exclusion is idempotent.
/// 3. **Field-list ordering**: `Glob` field ids are sorted without removing
///    duplicates. `Term`/`Phrase` expansions retain exact duplicate branches
///    because each branch contributes to the score.
///
/// Every positive branch is retained, including `All`: an ancestor boost can
/// make apparently unit-score children contribute independently. The scorer
/// owns the narrower effective-unit-`All` collapse because it sees cumulative
/// boosts during lowering.
///
/// Clause-level repetition caps are unnecessary here: `MAX_QUERY_LENGTH`
/// already bounds the source, and this pass only rewrites structure.
/// Idempotent: a second pass is a no-op and returns a zero report.
#[must_use]
pub fn canonicalize_query(query: &mut Query) -> QueryCanonicalizationReport {
    let mut report = QueryCanonicalizationReport::default();
    canonicalize_query_inner(query, &mut report);
    report
}

fn canonicalize_query_inner(query: &mut Query, report: &mut QueryCanonicalizationReport) {
    match query {
        Query::Boolean { clauses, .. } => {
            for clause in clauses.iter_mut() {
                canonicalize_query_inner(&mut clause.query, report);
            }
            // MustNot dedup: exclusion is idempotent.
            let mut seen_must_not = Vec::new();
            clauses.retain(|clause| {
                if clause.occur != Occur::MustNot {
                    return true;
                }
                let key = canonical_sort_key(&clause.query);
                if seen_must_not.contains(&key) {
                    report.must_not_duplicates_removed += 1;
                    return false;
                }
                seen_must_not.push(key);
                true
            });
            // Group occurrences stably. Argus constructs separate required
            // and optional aggregates in this order, but the relative order of
            // score-bearing clauses inside either aggregate is observable via
            // non-associative f32 addition and must not be canonicalized away.
            let before = clauses
                .iter()
                .map(|clause| clause.occur)
                .collect::<Vec<_>>();
            clauses.sort_by_key(|clause| occur_rank(clause.occur));
            let after = clauses
                .iter()
                .map(|clause| clause.occur)
                .collect::<Vec<_>>();
            if before != after {
                report.boolean_levels_sorted += 1;
            }
        }
        Query::Term { .. } | Query::Phrase { .. } => {}
        Query::Glob { field_ids, .. } => {
            let original = field_ids.clone();
            field_ids.sort_unstable();
            if *field_ids != original {
                report.glob_fields_canonicalized += 1;
            }
        }
        Query::Boost { query, .. } => canonicalize_query_inner(query, report),
        Query::Empty | Query::All | Query::Range { .. } | Query::Set { .. } => {}
    }
}

/// Stable user-facing classification preserved from the incumbent adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum QueryExplanation {
    /// Empty or whitespace-only input.
    Empty,
    /// One whitespace-delimited fragment.
    Simple,
    /// Input wrapped in matching single or double quotes.
    Phrase,
    /// More than one whitespace-delimited fragment.
    Boolean,
}

impl fmt::Display for QueryExplanation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("empty"),
            Self::Simple => formatter.write_str("simple"),
            Self::Phrase => formatter.write_str("phrase"),
            Self::Boolean => formatter.write_str("boolean"),
        }
    }
}

/// Classify a raw query exactly as the shipping explanation surface does.
#[must_use]
pub fn classify_query(query: &str) -> QueryExplanation {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return QueryExplanation::Empty;
    }
    if (trimmed.starts_with('"') && trimmed.ends_with('"'))
        || (trimmed.starts_with('\'') && trimmed.ends_with('\''))
    {
        return QueryExplanation::Phrase;
    }
    if trimmed.split_whitespace().count() <= 1 {
        QueryExplanation::Simple
    } else {
        QueryExplanation::Boolean
    }
}

/// Parser recovery/normalization diagnostic category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryDiagnosticKind {
    /// The query exceeded [`MAX_QUERY_LENGTH`].
    Truncated,
    /// Invalid syntax was repaired or skipped.
    SyntaxRecovery,
    /// A field prefix was not present in the schema.
    UnknownField,
    /// A known field cannot be queried by this parser.
    UnsupportedField,
    /// A fragment emitted no usable query branch.
    DroppedFragment,
    /// A typed range endpoint or set member did not match its schema field.
    InvalidTypedValue,
    /// A boost was malformed or outside the finite non-negative domain.
    InvalidBoost,
    /// Match-all was added to give an all-negative user query complement semantics.
    AllNegativeRepair,
    /// The nesting cap was reached.
    DepthLimit,
}

/// One best-effort parse diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryDiagnostic {
    /// Stable diagnostic category.
    pub kind: QueryDiagnosticKind,
    /// Human-readable contract explanation.
    pub message: String,
    /// Source byte offset, when tied to a fragment.
    pub byte_offset: Option<usize>,
    /// Source fragment, when useful and bounded by the query cap.
    pub fragment: Option<String>,
}

/// Complete output of a lenient parse.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedQuery {
    /// Recovered engine-neutral tree.
    pub query: Query,
    /// Incumbent-compatible explanation classification.
    pub explanation: QueryExplanation,
    /// Ordered recovery/truncation diagnostics.
    pub diagnostics: Vec<QueryDiagnostic>,
    /// Whether the input was truncated before lexing.
    pub was_truncated: bool,
}

/// Maximum raw-query bytes admitted by the dev-only CASS observation boundary.
///
/// Shipping parsing still admits [`MAX_QUERY_LENGTH`] Unicode scalar values.
/// This separate byte cap prevents a conformance caller from retaining an
/// arbitrarily large pre-truncation witness.
#[cfg(any(test, feature = "conformance-internals"))]
pub const MAX_CASS_OBSERVATION_RAW_QUERY_BYTES: usize = 1_048_576;

/// Maximum number of exact filter values admitted by one CASS observation.
#[cfg(any(test, feature = "conformance-internals"))]
pub const MAX_CASS_OBSERVATION_FILTER_VALUES: usize = 4_096;

/// Maximum bytes admitted for any individual exact filter value.
#[cfg(any(test, feature = "conformance-internals"))]
pub const MAX_CASS_OBSERVATION_FILTER_VALUE_BYTES: usize = 4_096;

/// Maximum aggregate bytes admitted across all exact filter values.
#[cfg(any(test, feature = "conformance-internals"))]
pub const MAX_CASS_OBSERVATION_FILTER_BYTES: usize = 1_048_576;

/// One token consumed by the native CASS grammar, including its source offset.
///
/// This is an evidence boundary rather than a second lexer: values are projected
/// from the private token stream that [`CassQueryParser`] actually lowers. Text
/// is intentionally retained here only in memory; durable gauntlet artifacts
/// must redact it before persistence.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg(any(test, feature = "conformance-internals"))]
pub enum CassQueryToken {
    /// A term operand and its starting UTF-8 byte offset.
    Term { text: String, byte_offset: usize },
    /// A quoted phrase operand and its opening-quote byte offset.
    Phrase { text: String, byte_offset: usize },
    /// An explicit conjunction.
    And { byte_offset: usize },
    /// An explicit disjunction.
    Or { byte_offset: usize },
    /// An explicit or prefix negation.
    Not { byte_offset: usize },
}

/// Complete in-memory witness emitted by the real native CASS parser.
///
/// `parsed` is exactly the value returned by [`CassQueryParser::parse`]. The
/// additional fields expose the parser's actual token stream and the named
/// CASS sanitization transform so a conformance adapter does not reconstruct
/// either value from the source corpus.
#[derive(Debug, Clone, PartialEq)]
#[cfg(any(test, feature = "conformance-internals"))]
pub struct CassParsedQueryObservation {
    /// Lowered query tree, classification, diagnostics, and truncation state.
    pub parsed: ParsedQuery,
    /// Shipping CASS sanitization applied to the admitted query prefix.
    pub sanitized_query: String,
    /// Exact query prefix admitted to the shipping parser.
    pub admitted_query: String,
    /// Ordered tokens consumed by the native grammar.
    pub tokens: Vec<CassQueryToken>,
}

/// A bounded CASS observation request was rejected before witness allocation.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
#[cfg(any(test, feature = "conformance-internals"))]
pub enum CassQueryObservationError {
    /// One request dimension exceeded its fixed conformance-only cap.
    #[error("CASS observation field {field} has size {actual}, exceeding limit {limit}")]
    BoundExceeded {
        /// Stable request dimension.
        field: &'static str,
        /// Observed size, saturated to `usize::MAX` on arithmetic overflow.
        actual: usize,
        /// Maximum admitted size.
        limit: usize,
    },
}

/// Invalid schema configuration for the shipping default parser.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum QueryParserConfigError {
    /// The schema violates the dense-ID descriptor contract.
    #[error("query schema {schema} is invalid: {detail}")]
    InvalidSchema {
        /// Schema diagnostic name.
        schema: &'static str,
        /// Descriptor validation detail.
        detail: String,
    },
    /// A mandatory default field is absent.
    #[error("default query field {field} is absent from schema {schema}")]
    MissingDefaultField {
        /// Schema diagnostic name.
        schema: &'static str,
        /// Missing field name.
        field: &'static str,
    },
    /// A mandatory default field is not default-analyzed text.
    #[error("default query field {field} in schema {schema} is not default-analyzed text")]
    InvalidDefaultField {
        /// Schema diagnostic name.
        schema: &'static str,
        /// Invalid field name.
        field: &'static str,
    },
}

/// A posting-list capability an operator needs from every field it names.
///
/// Centralizing this (`bd-vmpb`) keeps capability knowledge in one table rather
/// than scattering `positions: true` special cases through lowering, so a new
/// operator cannot silently bypass the check: it must declare what it reads.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndexCapability {
    /// The operator reads a position list, so the field must be indexed with
    /// `positions: true`.
    Positions,
}

impl fmt::Display for IndexCapability {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Positions => formatter.write_str("positions"),
        }
    }
}

/// A query names an operator the index cannot serve on that field.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum QueryCapabilityError {
    /// A position-dependent operator named a field indexed without positions.
    #[error(
        "operator {operator} requires {capability} on field {field} of schema {schema}, \
         which is indexed without them"
    )]
    PositionsRequired {
        /// Schema diagnostic name.
        schema: &'static str,
        /// Field the operator named, by schema name where resolvable.
        field: String,
        /// Operator that requires the capability.
        operator: QueryExplanation,
        /// Capability the operator requires.
        capability: IndexCapability,
    },
}

/// Capabilities `query` requires that `schema` does not provide.
///
/// Position-independent operators — `Term`, `Range`, `Set`, `Glob`, `Boolean`,
/// `Boost`, `All`, `Empty` — never read a position list and are therefore valid
/// against a positionless index, matching and ranking identically to a
/// positioned one over the same documents. Only `Phrase` consumes positions.
///
/// A single-term phrase is exempt on purpose: it degenerates to a term match
/// and reads no positions, which is the same rule lowering already applies.
///
/// # Errors
///
/// Returns [`QueryCapabilityError::PositionsRequired`] naming the schema, the
/// field, and the operator — never a bare "unsupported query".
pub fn validate_index_capabilities(
    query: &Query,
    schema: SchemaDescriptor,
) -> Result<(), QueryCapabilityError> {
    match query {
        // Position-independent: nothing to check.
        Query::Empty
        | Query::All
        | Query::Term { .. }
        | Query::Range { .. }
        | Query::Set { .. }
        | Query::Glob { .. } => Ok(()),
        Query::Phrase { fields, terms, .. } => {
            if terms.len() < 2 {
                return Ok(());
            }
            for field in fields {
                let descriptor = schema
                    .fields
                    .iter()
                    .find(|candidate| candidate.id == field.field_id);
                let positioned = matches!(
                    descriptor.map(|descriptor| descriptor.kind),
                    Some(FieldKind::Text {
                        positions: true,
                        ..
                    })
                );
                if !positioned {
                    return Err(QueryCapabilityError::PositionsRequired {
                        schema: schema.name,
                        field: descriptor.map_or_else(
                            || format!("#{}", field.field_id),
                            |descriptor| descriptor.name.to_owned(),
                        ),
                        operator: QueryExplanation::Phrase,
                        capability: IndexCapability::Positions,
                    });
                }
            }
            Ok(())
        }
        Query::Boolean { clauses, .. } => {
            for clause in clauses {
                validate_index_capabilities(&clause.query, schema)?;
            }
            Ok(())
        }
        Query::Boost { query, .. } => validate_index_capabilities(query, schema),
    }
}

/// Shipping lenient parser for the default `[content, title^2]` expansion.
#[derive(Debug, Clone, Copy)]
pub struct DefaultQueryParser {
    schema: SchemaDescriptor,
    default_fields: [QueryField; 2],
}

impl DefaultQueryParser {
    /// Bind the parser to a schema containing default-analyzed `content` and
    /// `title` fields.
    ///
    /// # Errors
    ///
    /// Returns a configuration error when either mandatory field is missing or
    /// uses a non-default analyzer.
    pub fn new(schema: SchemaDescriptor) -> Result<Self, QueryParserConfigError> {
        schema
            .validate()
            .map_err(|error| QueryParserConfigError::InvalidSchema {
                schema: schema.name,
                detail: error.to_string(),
            })?;
        let content = required_default_field(schema, CONTENT_FIELD_NAME)?;
        let title = required_default_field(schema, TITLE_FIELD_NAME)?;
        Ok(Self {
            schema,
            default_fields: [
                QueryField::new(content.id, 1.0),
                QueryField::new(title.id, TITLE_BOOST),
            ],
        })
    }

    /// Parse arbitrary user input without returning a syntax error.
    #[must_use]
    pub fn parse(&self, query: &str) -> ParsedQuery {
        self.parse_lenient(query)
    }

    /// Explicitly named alias emphasizing that syntax never becomes a search
    /// failure.
    #[must_use]
    pub fn parse_lenient(&self, query: &str) -> ParsedQuery {
        let (query, was_truncated) = truncated_prefix(query);
        let mut diagnostics = Vec::new();
        if was_truncated {
            emit_diagnostic(
                &mut diagnostics,
                QueryDiagnostic {
                    kind: QueryDiagnosticKind::Truncated,
                    message: format!("query truncated to {MAX_QUERY_LENGTH} Unicode scalar values"),
                    byte_offset: Some(query.len()),
                    fragment: None,
                },
            );
        }

        let explanation = classify_query(query);
        let tokens = lex(query, &mut diagnostics);
        let mut grammar = Grammar {
            parser: *self,
            tokens,
            cursor: 0,
            diagnostics,
            lowered_atoms: Vec::new(),
            field_scopes: Vec::new(),
        };
        grammar.recover_leading_binary_operators();
        let mut parsed = grammar.parse_expression(0);
        grammar.recover_trailing_tokens();
        if let Some(node) = parsed.as_mut() {
            rewrite_parser_syntax(node);
            trim_parser_dropped(&mut node.query, &node.dedup_key);
            flatten_should_of_should(&mut node.query);
        }
        repair_root_all_negative(&mut parsed, &mut grammar);

        ParsedQuery {
            query: parsed.map_or(Query::Empty, |node| node.query),
            explanation,
            diagnostics: grammar.diagnostics,
            was_truncated,
        }
    }
}

fn required_default_field(
    schema: SchemaDescriptor,
    name: &'static str,
) -> Result<FieldDescriptor, QueryParserConfigError> {
    let Some(field) = schema.fields.iter().find(|field| field.name == name) else {
        return Err(QueryParserConfigError::MissingDefaultField {
            schema: schema.name,
            field: name,
        });
    };
    // Positions are deliberately NOT required here (`bd-vmpb`). A position-free
    // text index is a valid, smaller, faster configuration: every
    // position-independent operator — bare term, prefix, boolean, range, set,
    // glob — matches and ranks identically with or without them, because none
    // of them reads a position list. Requiring `positions: true` to *construct*
    // the parser made a positionless schema unqueryable end to end, so
    // `DefaultQueryParser::new` failed before any query could be parsed and the
    // QG positions_off fixture aborted the whole matrix run.
    //
    // What a default field genuinely requires is the intended analyzer, so the
    // term text a query produces matches the term text indexing produced.
    // Position-dependent operators are rejected later and precisely, by
    // [`validate_index_capabilities`], which names the offending field and
    // operator instead of disabling the schema wholesale.
    if !matches!(
        field.kind,
        FieldKind::Text {
            analyzer: AnalyzerKind::FrankensearchDefault,
            ..
        }
    ) {
        return Err(QueryParserConfigError::InvalidDefaultField {
            schema: schema.name,
            field: name,
        });
    }
    Ok(*field)
}

/// Return the character-safe prefix accepted by the parser and log truncation.
#[must_use]
pub fn truncate_query(query: &str) -> &str {
    let (prefix, was_truncated) = truncated_prefix(query);
    if was_truncated {
        tracing::warn!(
            target: crate::tracing_conventions::TARGET,
            phase = crate::tracing_conventions::ARGUS_QUERY,
            query_len = query.len(),
            "query truncated to MAX_QUERY_LENGTH"
        );
    }
    prefix
}

fn truncated_prefix(query: &str) -> (&str, bool) {
    if query.len() <= MAX_QUERY_LENGTH {
        return (query, false);
    }
    query
        .char_indices()
        .nth(MAX_QUERY_LENGTH)
        .map_or((query, false), |(end, _)| (&query[..end], true))
}

fn emit_diagnostic(diagnostics: &mut Vec<QueryDiagnostic>, diagnostic: QueryDiagnostic) {
    tracing::warn!(
        target: crate::tracing_conventions::TARGET,
        phase = crate::tracing_conventions::ARGUS_QUERY,
        "lenient query parse diagnostic"
    );
    diagnostics.push(diagnostic);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AtomDelimiter {
    None,
    SingleQuotes,
    DoubleQuotes,
}

impl AtomDelimiter {
    const fn from_quote(delimiter: char) -> Self {
        match delimiter {
            '\'' => Self::SingleQuotes,
            '"' => Self::DoubleQuotes,
            _ => Self::None,
        }
    }

    const fn is_quoted(self) -> bool {
        !matches!(self, Self::None)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AtomToken {
    raw: String,
    field: Option<String>,
    delimiter: AtomDelimiter,
    slop: u32,
    prefix: bool,
    boost: Option<String>,
    byte_offset: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RangeToken {
    field: Option<String>,
    lower: String,
    upper: String,
    lower_inclusive: bool,
    upper_inclusive: bool,
    boost: Option<String>,
    byte_offset: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SetToken {
    field: Option<String>,
    values: Vec<String>,
    boost: Option<String>,
    byte_offset: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SyntaxKey {
    Atom {
        field: Option<String>,
        raw: String,
        delimiter: AtomDelimiter,
        slop: u32,
        prefix: bool,
    },
    Range {
        field: Option<String>,
        lower: String,
        upper: String,
        lower_inclusive: bool,
        upper_inclusive: bool,
    },
    Set {
        field: Option<String>,
        values: Vec<String>,
    },
    Regex {
        field: Option<String>,
        pattern: String,
    },
    LoweredAway(Box<Self>),
    Boolean(Vec<(Option<Occur>, Self)>),
    Boost {
        query: Box<Self>,
        factor_bits: u64,
    },
}

fn take_syntax_key(slot: &mut SyntaxKey) -> SyntaxKey {
    std::mem::replace(slot, SyntaxKey::Boolean(Vec::new()))
}

fn boost_syntax_key(query: SyntaxKey, raw: Option<&str>) -> SyntaxKey {
    let Some(raw) = raw else {
        return query;
    };
    let Some(factor) = raw.parse::<f64>().ok().filter(|factor| *factor >= 0.0) else {
        return query;
    };
    if parse_boost_factor(raw).is_none() {
        return query;
    }
    if (factor - 1.0).abs() <= f64::EPSILON {
        query
    } else {
        SyntaxKey::Boost {
            query: Box::new(query),
            factor_bits: factor.to_bits(),
        }
    }
}

#[allow(clippy::cast_possible_truncation)]
fn empty_query_for_syntax(syntax: &SyntaxKey) -> Query {
    if let SyntaxKey::LoweredAway(inner) = syntax {
        return empty_query_for_syntax(inner);
    }
    let SyntaxKey::Boost { query, factor_bits } = syntax else {
        return Query::Empty;
    };
    let inner = empty_query_for_syntax(query);
    // Tantivy parses boost syntax as f64, then lowers it to the f32 scoring
    // domain. Preserve that intentional narrowing, including overflow to
    // infinity, for differential parity.
    let factor = f64::from_bits(*factor_bits) as f32;
    if factor.to_bits() == 1.0_f32.to_bits() {
        inner
    } else {
        Query::Boost {
            query: Box::new(inner),
            factor,
        }
    }
}

fn mark_syntax_lowered_away(syntax: SyntaxKey) -> SyntaxKey {
    match syntax {
        SyntaxKey::Boost { query, factor_bits } => SyntaxKey::Boost {
            query: Box::new(mark_syntax_lowered_away(*query)),
            factor_bits,
        },
        syntax => SyntaxKey::LoweredAway(Box::new(syntax)),
    }
}

fn range_dedup_key(token: &RangeToken) -> SyntaxKey {
    boost_syntax_key(
        SyntaxKey::Range {
            field: token.field.clone(),
            lower: token.lower.clone(),
            upper: token.upper.clone(),
            lower_inclusive: token.lower != "*" && token.lower_inclusive,
            upper_inclusive: token.upper != "*" && token.upper_inclusive,
        },
        token.boost.as_deref(),
    )
}

fn set_dedup_key(token: &SetToken) -> SyntaxKey {
    boost_syntax_key(
        SyntaxKey::Set {
            field: token.field.clone(),
            values: token.values.clone(),
        },
        token.boost.as_deref(),
    )
}

fn atom_dedup_key(atom: &AtomToken) -> SyntaxKey {
    boost_syntax_key(
        SyntaxKey::Atom {
            field: atom.field.clone(),
            raw: atom.raw.clone(),
            delimiter: atom.delimiter,
            slop: atom.slop,
            prefix: atom.prefix,
        },
        atom.boost.as_deref(),
    )
}

fn atom_cache_key(atom: &AtomToken) -> Option<String> {
    if atom
        .boost
        .as_deref()
        .is_some_and(|raw| parse_boost_factor(raw).is_none())
    {
        None
    } else {
        Some(format!("{atom:?}"))
    }
}

fn parse_boost_factor(raw: &str) -> Option<f32> {
    let (integer, fraction) = raw
        .split_once('.')
        .map_or((raw, None), |(integer, fraction)| (integer, Some(fraction)));
    if integer.is_empty()
        || !integer.bytes().all(|byte| byte.is_ascii_digit())
        || fraction.is_some_and(|fraction| {
            fraction.is_empty() || !fraction.bytes().all(|byte| byte.is_ascii_digit())
        })
    {
        return None;
    }
    raw.parse::<f32>()
        .ok()
        .filter(|factor| factor.is_finite() && *factor >= 0.0)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum LexToken {
    Atom(AtomToken),
    Range(RangeToken),
    Set(SetToken),
    UnsupportedRegex {
        byte_offset: usize,
        field: Option<String>,
        pattern: String,
        boost: Option<String>,
    },
    Dropped(usize),
    And(usize),
    Or(usize),
    Not(usize),
    Plus(usize),
    Minus(usize),
    LeftParen {
        byte_offset: usize,
        field: Option<String>,
    },
    RightParen {
        byte_offset: usize,
        boost: Option<String>,
    },
}

impl LexToken {
    const fn byte_offset(&self) -> usize {
        match self {
            Self::Atom(atom) => atom.byte_offset,
            Self::Range(token) => token.byte_offset,
            Self::Set(token) => token.byte_offset,
            Self::UnsupportedRegex {
                byte_offset: offset,
                ..
            }
            | Self::Dropped(offset)
            | Self::And(offset)
            | Self::Or(offset)
            | Self::Not(offset)
            | Self::Plus(offset)
            | Self::Minus(offset) => *offset,
            Self::LeftParen { byte_offset, .. } | Self::RightParen { byte_offset, .. } => {
                *byte_offset
            }
        }
    }

    const fn can_start_operand(&self) -> bool {
        matches!(
            self,
            Self::Atom(_)
                | Self::Range(_)
                | Self::Set(_)
                | Self::UnsupportedRegex { .. }
                | Self::Dropped(_)
                | Self::Not(_)
                | Self::Plus(_)
                | Self::Minus(_)
                | Self::LeftParen { .. }
        )
    }
}

fn lex(query: &str, diagnostics: &mut Vec<QueryDiagnostic>) -> Vec<LexToken> {
    let mut tokens = Vec::new();
    let mut cursor = 0;
    let mut open_groups = 0_usize;

    while cursor < query.len() {
        let Some(ch) = query[cursor..].chars().next() else {
            break;
        };
        if ch.is_whitespace() {
            cursor += ch.len_utf8();
            continue;
        }

        if let Some((field, value_start)) = field_prefix(query, cursor) {
            let value_start = skip_whitespace(query, value_start);
            if query[value_start..].starts_with('(') {
                tokens.push(LexToken::LeftParen {
                    byte_offset: value_start,
                    field: Some(unescape_field_name(field)),
                });
                open_groups += 1;
                cursor = value_start + 1;
                continue;
            }
            if let Some(delimiter) = query[value_start..]
                .chars()
                .next()
                .filter(|candidate| matches!(candidate, '"' | '\''))
            {
                let (atom, next) = lex_quoted(
                    query,
                    value_start,
                    delimiter,
                    Some(unescape_field_name(field)),
                    diagnostics,
                );
                tokens.push(LexToken::Atom(atom));
                cursor = next;
                continue;
            }
            if matches!(query[value_start..].chars().next(), Some('[' | '{')) {
                let (token, next) =
                    lex_typed_range(query, cursor, Some(field), value_start, diagnostics);
                if let Some(token) = token {
                    tokens.push(token);
                } else {
                    tokens.push(LexToken::Dropped(cursor));
                }
                cursor = next;
                continue;
            }
            if let Some(open_offset) = set_open_offset(query, value_start) {
                let (token, next) =
                    lex_typed_set(query, cursor, Some(field), open_offset, diagnostics);
                if let Some(token) = token {
                    tokens.push(token);
                } else {
                    tokens.push(LexToken::Dropped(cursor));
                }
                cursor = next;
                continue;
            }
            if comparison_operator(query, value_start).is_some() {
                let (token, next) =
                    lex_comparison_range(query, cursor, Some(field), value_start, diagnostics);
                if let Some(token) = token {
                    tokens.push(token);
                } else {
                    tokens.push(LexToken::Dropped(cursor));
                }
                cursor = next;
                continue;
            }

            let end = scan_fragment_end(query, value_start);
            let fragment = &query[value_start..end];
            let (raw, boost, trailing_boost) = split_boost(fragment);
            if let Some(relative) = trailing_boost {
                emit_trailing_boost_diagnostic(diagnostics, value_start + relative);
            }
            let raw = unescape_fragment(raw).0;
            if is_regex_fragment(&raw) {
                emit_regex_diagnostic(diagnostics, cursor);
                tokens.push(unsupported_regex_token(
                    cursor,
                    Some(unescape_field_name(field)),
                    &raw,
                    boost,
                ));
            } else {
                tokens.push(LexToken::Atom(AtomToken {
                    raw,
                    field: Some(unescape_field_name(field)),
                    delimiter: AtomDelimiter::None,
                    slop: 0,
                    prefix: false,
                    boost: boost.map(str::to_owned),
                    byte_offset: cursor,
                }));
            }
            cursor = end;
            continue;
        }

        if matches!(ch, '[' | '{') {
            let (token, next) = lex_typed_range(query, cursor, None, cursor, diagnostics);
            if let Some(token) = token {
                tokens.push(token);
            } else {
                tokens.push(LexToken::Dropped(cursor));
            }
            cursor = next;
            continue;
        }
        if let Some(open_offset) = set_open_offset(query, cursor) {
            let (token, next) = lex_typed_set(query, cursor, None, open_offset, diagnostics);
            if let Some(token) = token {
                tokens.push(token);
            } else {
                tokens.push(LexToken::Dropped(cursor));
            }
            cursor = next;
            continue;
        }
        if comparison_operator(query, cursor).is_some() {
            let (token, next) = lex_comparison_range(query, cursor, None, cursor, diagnostics);
            if let Some(token) = token {
                tokens.push(token);
            } else {
                tokens.push(LexToken::Dropped(cursor));
            }
            cursor = next;
            continue;
        }

        match ch {
            '(' => {
                tokens.push(LexToken::LeftParen {
                    byte_offset: cursor,
                    field: None,
                });
                open_groups += 1;
                cursor += 1;
                continue;
            }
            ')' => {
                let (boost, next) = lex_boost_suffix(query, cursor + 1, diagnostics);
                if open_groups == 0 {
                    emit_diagnostic(
                        diagnostics,
                        QueryDiagnostic {
                            kind: QueryDiagnosticKind::SyntaxRecovery,
                            message: "syntax recovery: unmatched closing parenthesis dropped"
                                .to_owned(),
                            byte_offset: Some(cursor),
                            fragment: None,
                        },
                    );
                    tokens.push(LexToken::Dropped(cursor));
                    break;
                }
                tokens.push(LexToken::RightParen {
                    byte_offset: cursor,
                    boost,
                });
                open_groups -= 1;
                cursor = next;
                continue;
            }
            '+' => {
                tokens.push(LexToken::Plus(cursor));
                cursor += 1;
                continue;
            }
            '-' => {
                tokens.push(LexToken::Minus(cursor));
                cursor += 1;
                continue;
            }
            '"' | '\'' => {
                let (atom, next) = lex_quoted(query, cursor, ch, None, diagnostics);
                tokens.push(LexToken::Atom(atom));
                cursor = next;
                continue;
            }
            _ => {}
        }

        let end = scan_fragment_end(query, cursor);
        let fragment = &query[cursor..end];
        let field: Option<String> = None;
        let value = fragment;
        let value_offset = cursor;
        let (raw, boost, trailing_boost) = split_boost(value);
        if let Some(relative) = trailing_boost {
            emit_trailing_boost_diagnostic(diagnostics, value_offset + relative);
        }
        let (raw, raw_was_escaped) = unescape_fragment(raw);
        if is_regex_fragment(&raw) {
            emit_regex_diagnostic(diagnostics, cursor);
            tokens.push(unsupported_regex_token(cursor, None, &raw, boost));
            cursor = end;
            continue;
        }
        let operator_has_ascii_space = query.as_bytes().get(end) == Some(&b' ');
        let token = if field.is_none() && boost.is_none() && !raw_was_escaped {
            match raw.as_str() {
                "AND" if operator_has_ascii_space => LexToken::And(cursor),
                "OR" if operator_has_ascii_space => LexToken::Or(cursor),
                "NOT" if operator_has_ascii_space => LexToken::Not(cursor),
                _ => LexToken::Atom(AtomToken {
                    raw,
                    field,
                    delimiter: AtomDelimiter::None,
                    slop: 0,
                    prefix: false,
                    boost: None,
                    byte_offset: cursor,
                }),
            }
        } else {
            LexToken::Atom(AtomToken {
                raw,
                field,
                delimiter: AtomDelimiter::None,
                slop: 0,
                prefix: false,
                boost: boost.map(str::to_owned),
                byte_offset: cursor,
            })
        };
        tokens.push(token);
        cursor = end;
    }

    if tokens.len() == 1 {
        let replacement = match &tokens[0] {
            LexToken::And(offset) => Some(operator_atom("AND", *offset)),
            LexToken::Or(offset) => Some(operator_atom("OR", *offset)),
            LexToken::Not(offset) => {
                emit_diagnostic(
                    diagnostics,
                    QueryDiagnostic {
                        kind: QueryDiagnosticKind::SyntaxRecovery,
                        message: "syntax recovery: standalone NOT retained as a literal".to_owned(),
                        byte_offset: Some(*offset),
                        fragment: None,
                    },
                );
                Some(operator_atom("NOT", *offset))
            }
            _ => None,
        };
        if let Some(replacement) = replacement {
            tokens[0] = replacement;
        }
    }

    tokens
}

fn field_prefix(query: &str, start: usize) -> Option<(&str, usize)> {
    let first = query[start..].chars().next()?;
    if first == '-' || (first != '\\' && is_field_special(first)) {
        return None;
    }
    let mut escaped = false;
    let mut field_end = None;
    for (relative, ch) in query[start..].char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' {
            escaped = true;
            continue;
        }
        if ch == ':' {
            let colon = start + relative;
            let end = field_end.map_or(colon, |end| start + end);
            return (end > start).then_some((&query[start..end], colon + 1));
        }
        if is_field_special(ch) && ch != ' ' {
            return None;
        }
        if ch.is_whitespace() {
            field_end.get_or_insert(relative);
            continue;
        }
        if field_end.is_some() {
            break;
        }
    }
    None
}

fn is_field_special(ch: char) -> bool {
    matches!(
        ch,
        '+' | '^'
            | '`'
            | ':'
            | '{'
            | '}'
            | '"'
            | '\''
            | '['
            | ']'
            | '('
            | ')'
            | '!'
            | '\\'
            | '*'
            | ' '
    )
}

fn unescape_field_name(field: &str) -> String {
    let mut value = String::with_capacity(field.len());
    let mut escaped = false;
    for ch in field.chars() {
        if escaped {
            if !is_field_special(ch) {
                value.push('\\');
            }
            value.push(ch);
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else {
            value.push(ch);
        }
    }
    if escaped {
        value.push('\\');
    }
    value
}

fn skip_whitespace(query: &str, mut cursor: usize) -> usize {
    while let Some(ch) = query[cursor..].chars().next() {
        if !ch.is_whitespace() {
            break;
        }
        cursor += ch.len_utf8();
    }
    cursor
}

fn set_open_offset(query: &str, start: usize) -> Option<usize> {
    let rest = query.get(start..)?.strip_prefix("IN")?;
    let after_in = start + 2;
    if !rest.chars().next().is_some_and(char::is_whitespace) {
        return None;
    }
    let open = skip_whitespace(query, after_in);
    query[open..].starts_with('[').then_some(open)
}

fn lex_typed_range(
    query: &str,
    byte_offset: usize,
    field: Option<&str>,
    open_offset: usize,
    diagnostics: &mut Vec<QueryDiagnostic>,
) -> (Option<LexToken>, usize) {
    let lower_inclusive = query.as_bytes().get(open_offset) == Some(&b'[');
    let (close_offset, upper_inclusive, missing_close) = typed_close(query, open_offset).map_or(
        (query.len(), lower_inclusive, true),
        |(offset, inclusive)| (offset, inclusive, false),
    );
    if missing_close {
        emit_typed_syntax_diagnostic(
            diagnostics,
            byte_offset,
            "syntax recovery: unterminated typed range retained to end of query",
        );
    }
    let interior = query[open_offset + 1..close_offset].trim();
    let Some(parts) = split_set_values(interior) else {
        emit_typed_syntax_diagnostic(
            diagnostics,
            byte_offset,
            "syntax recovery: malformed typed range dropped",
        );
        return (
            None,
            close_offset.saturating_add(usize::from(!missing_close)),
        );
    };
    let (lower, upper) = match parts.as_slice() {
        [lower, separator, upper] if separator == "TO" => (lower.clone(), upper.clone()),
        [lower, separator] if separator == "TO" => {
            emit_typed_syntax_diagnostic(
                diagnostics,
                byte_offset,
                "syntax recovery: typed range missing upper bound",
            );
            (lower.clone(), "*".to_owned())
        }
        [lower, upper] => {
            emit_typed_syntax_diagnostic(
                diagnostics,
                byte_offset,
                "syntax recovery: typed range missing keyword `TO`",
            );
            (lower.clone(), upper.clone())
        }
        [lower] => {
            emit_typed_syntax_diagnostic(
                diagnostics,
                byte_offset,
                "syntax recovery: typed range missing keyword `TO` and upper bound",
            );
            (lower.clone(), "*".to_owned())
        }
        _ => {
            emit_typed_syntax_diagnostic(
                diagnostics,
                byte_offset,
                "syntax recovery: typed range requires `lower TO upper`",
            );
            return (
                None,
                close_offset.saturating_add(usize::from(!missing_close)),
            );
        }
    };
    let after_close = close_offset.saturating_add(usize::from(!missing_close));
    let (boost, next) = lex_boost_suffix(query, after_close, diagnostics);
    (
        Some(LexToken::Range(RangeToken {
            field: field.map(unescape_field_name),
            lower,
            upper,
            lower_inclusive,
            upper_inclusive,
            boost,
            byte_offset,
        })),
        next,
    )
}

fn lex_typed_set(
    query: &str,
    byte_offset: usize,
    field: Option<&str>,
    open_offset: usize,
    diagnostics: &mut Vec<QueryDiagnostic>,
) -> (Option<LexToken>, usize) {
    let (close_offset, closed_with_bracket, missing_close) = typed_close(query, open_offset)
        .map_or((query.len(), true, true), |(offset, bracket)| {
            (offset, bracket, false)
        });
    if missing_close {
        emit_typed_syntax_diagnostic(
            diagnostics,
            byte_offset,
            "syntax recovery: unterminated typed set retained to end of query",
        );
    }
    if !missing_close && !closed_with_bracket {
        emit_typed_syntax_diagnostic(
            diagnostics,
            byte_offset,
            "syntax recovery: typed set requires square brackets",
        );
        return (None, close_offset + 1);
    }
    let after_close = close_offset.saturating_add(usize::from(!missing_close));
    let Some(values) = split_set_values(&query[open_offset + 1..close_offset]) else {
        emit_typed_syntax_diagnostic(
            diagnostics,
            byte_offset,
            "syntax recovery: malformed typed set dropped",
        );
        return (None, after_close);
    };
    let (boost, next) = lex_boost_suffix(query, after_close, diagnostics);
    (
        Some(LexToken::Set(SetToken {
            field: field.map(unescape_field_name),
            values,
            boost,
            byte_offset,
        })),
        next,
    )
}

fn split_set_values(interior: &str) -> Option<Vec<String>> {
    let mut values = Vec::new();
    let mut current = String::new();
    let mut delimiter = None;
    let mut escaped = false;
    let mut token_started = false;
    for ch in interior.chars() {
        if escaped {
            if delimiter.is_none() && !is_query_escape(ch) {
                current.push('\\');
            }
            current.push(ch);
            escaped = false;
            token_started = true;
        } else if ch == '\\' {
            escaped = true;
            token_started = true;
        } else if let Some(quote) = delimiter {
            if ch == quote {
                delimiter = None;
            } else {
                current.push(ch);
            }
        } else if matches!(ch, '"' | '\'') {
            delimiter = Some(ch);
            token_started = true;
        } else if ch == ',' {
            current.push(ch);
            token_started = true;
        } else if ch.is_whitespace() {
            if token_started {
                values.push(std::mem::take(&mut current));
                token_started = false;
            }
        } else {
            current.push(ch);
            token_started = true;
        }
    }
    if escaped || delimiter.is_some() {
        return None;
    }
    if token_started {
        values.push(current);
    }
    Some(values)
}

fn comparison_operator(query: &str, start: usize) -> Option<(&'static str, usize)> {
    for operator in [">=", "<=", ">", "<"] {
        if query[start..].starts_with(operator) {
            return Some((operator, operator.len()));
        }
    }
    None
}

fn lex_comparison_range(
    query: &str,
    byte_offset: usize,
    field: Option<&str>,
    operator_offset: usize,
    diagnostics: &mut Vec<QueryDiagnostic>,
) -> (Option<LexToken>, usize) {
    let Some((operator, operator_len)) = comparison_operator(query, operator_offset) else {
        return (None, operator_offset);
    };
    let value_start = skip_whitespace(query, operator_offset + operator_len);
    let end = scan_fragment_end(query, value_start);
    let (value, boost, trailing_boost) = split_boost(&query[value_start..end]);
    if let Some(relative) = trailing_boost {
        emit_trailing_boost_diagnostic(diagnostics, value_start + relative);
    }
    if value.is_empty() {
        emit_typed_syntax_diagnostic(
            diagnostics,
            byte_offset,
            "syntax recovery: comparison range requires a bound",
        );
        return (None, end);
    }
    let value = unescape_fragment(value).0;
    let (lower, upper, lower_inclusive, upper_inclusive) = match operator {
        ">" => (value, "*".to_owned(), false, false),
        ">=" => (value, "*".to_owned(), true, false),
        "<" => ("*".to_owned(), value, false, false),
        "<=" => ("*".to_owned(), value, false, true),
        _ => return (None, end),
    };
    (
        Some(LexToken::Range(RangeToken {
            field: field.map(unescape_field_name),
            lower,
            upper,
            lower_inclusive,
            upper_inclusive,
            boost: boost.map(str::to_owned),
            byte_offset,
        })),
        end,
    )
}

fn typed_close(query: &str, open_offset: usize) -> Option<(usize, bool)> {
    let mut delimiter = None;
    let mut escaped = false;
    for (relative, ch) in query[open_offset + 1..].char_indices() {
        if escaped {
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if let Some(quote) = delimiter {
            if ch == quote {
                delimiter = None;
            }
        } else if matches!(ch, '"' | '\'') {
            delimiter = Some(ch);
        } else if matches!(ch, ']' | '}') {
            return Some((open_offset + 1 + relative, ch == ']'));
        }
    }
    None
}

fn emit_typed_syntax_diagnostic(
    diagnostics: &mut Vec<QueryDiagnostic>,
    byte_offset: usize,
    message: &str,
) {
    emit_diagnostic(
        diagnostics,
        QueryDiagnostic {
            kind: QueryDiagnosticKind::SyntaxRecovery,
            message: message.to_owned(),
            byte_offset: Some(byte_offset),
            fragment: None,
        },
    );
}

fn lex_boost_suffix(
    query: &str,
    start: usize,
    diagnostics: &mut Vec<QueryDiagnostic>,
) -> (Option<String>, usize) {
    if query.as_bytes().get(start) != Some(&b'^') {
        return (None, start);
    }
    let end = scan_fragment_end(query, start);
    let suffix = &query[start + 1..end];
    let (boost, trailing) = split_boost_component(suffix);
    if let Some(relative) = trailing {
        emit_trailing_boost_diagnostic(diagnostics, start + 1 + relative);
    }
    (Some(boost.to_owned()), end)
}

fn emit_trailing_boost_diagnostic(diagnostics: &mut Vec<QueryDiagnostic>, byte_offset: usize) {
    emit_diagnostic(
        diagnostics,
        QueryDiagnostic {
            kind: QueryDiagnosticKind::SyntaxRecovery,
            message: "syntax recovery: trailing boost suffix dropped".to_owned(),
            byte_offset: Some(byte_offset),
            fragment: None,
        },
    );
}

fn is_regex_fragment(raw: &str) -> bool {
    raw.starts_with('/')
}

fn unsupported_regex_token(
    byte_offset: usize,
    field: Option<String>,
    raw: &str,
    boost: Option<&str>,
) -> LexToken {
    let pattern = raw.strip_prefix('/').unwrap_or(raw);
    let pattern = pattern.strip_suffix('/').unwrap_or(pattern);
    LexToken::UnsupportedRegex {
        byte_offset,
        field,
        pattern: pattern.replace("\\/", "/"),
        boost: boost.map(str::to_owned),
    }
}

fn emit_regex_diagnostic(diagnostics: &mut Vec<QueryDiagnostic>, byte_offset: usize) {
    emit_diagnostic(
        diagnostics,
        QueryDiagnostic {
            kind: QueryDiagnosticKind::DroppedFragment,
            message: "unsupported regular-expression query dropped".to_owned(),
            byte_offset: Some(byte_offset),
            fragment: None,
        },
    );
}

fn operator_atom(raw: &str, byte_offset: usize) -> LexToken {
    LexToken::Atom(AtomToken {
        raw: raw.to_owned(),
        field: None,
        delimiter: AtomDelimiter::None,
        slop: 0,
        prefix: false,
        boost: None,
        byte_offset,
    })
}

fn scan_fragment_end(query: &str, start: usize) -> usize {
    let mut escaped = false;
    for (relative, ch) in query[start..].char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' {
            escaped = true;
            continue;
        }
        if ch.is_whitespace() || matches!(ch, '(' | ')') {
            return start + relative;
        }
    }
    query.len()
}

fn unescape_fragment(fragment: &str) -> (String, bool) {
    let mut value = String::with_capacity(fragment.len());
    let mut escaped = false;
    let mut had_escape = false;
    for ch in fragment.chars() {
        if escaped {
            if !is_query_escape(ch) {
                value.push('\\');
            }
            value.push(ch);
            escaped = false;
            had_escape = true;
        } else if ch == '\\' {
            escaped = true;
        } else {
            value.push(ch);
        }
    }
    had_escape |= escaped;
    (value, had_escape)
}

fn is_query_escape(ch: char) -> bool {
    ch.is_whitespace()
        || matches!(
            ch,
            '^' | '`' | ':' | '[' | ']' | '{' | '}' | '"' | '\'' | '(' | ')' | '\\' | '-'
        )
}

fn lex_quoted(
    query: &str,
    quote_offset: usize,
    delimiter: char,
    field: Option<String>,
    diagnostics: &mut Vec<QueryDiagnostic>,
) -> (AtomToken, usize) {
    let content_start = quote_offset + delimiter.len_utf8();
    let mut raw = String::new();
    let mut cursor = content_start;
    let mut escaped = false;
    let mut terminated = false;
    while let Some(ch) = query[cursor..].chars().next() {
        cursor += ch.len_utf8();
        if escaped {
            raw.push(ch);
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == delimiter {
            terminated = true;
            break;
        } else {
            raw.push(ch);
        }
    }
    if escaped {
        raw.push('\\');
    }
    if !terminated {
        emit_diagnostic(
            diagnostics,
            QueryDiagnostic {
                kind: QueryDiagnosticKind::SyntaxRecovery,
                message: "syntax recovery: unterminated quoted fragment".to_owned(),
                byte_offset: Some(quote_offset),
                fragment: Some(raw.clone()),
            },
        );
    }

    let suffix_start = cursor;
    let suffix_end = scan_fragment_end(query, suffix_start);
    let suffix = &query[suffix_start..suffix_end];
    let parsed_suffix = parse_phrase_suffix(suffix);
    if let Some(relative) = parsed_suffix.recovery_offset {
        let byte_offset = suffix_start + relative;
        if parsed_suffix.trailing_boost {
            emit_trailing_boost_diagnostic(diagnostics, byte_offset);
        } else {
            emit_typed_syntax_diagnostic(
                diagnostics,
                byte_offset,
                if parsed_suffix.relex_suffix {
                    "syntax recovery: unsupported phrase suffix retained as a separate fragment"
                } else {
                    "syntax recovery: unsupported phrase suffix dropped"
                },
            );
        }
    }
    let next = if parsed_suffix.relex_suffix {
        suffix_start
    } else {
        suffix_end
    };
    (
        AtomToken {
            raw,
            field,
            delimiter: AtomDelimiter::from_quote(delimiter),
            slop: parsed_suffix.slop,
            prefix: parsed_suffix.prefix,
            boost: parsed_suffix.boost.map(str::to_owned),
            byte_offset: quote_offset,
        },
        next,
    )
}

struct ParsedPhraseSuffix<'a> {
    slop: u32,
    prefix: bool,
    boost: Option<&'a str>,
    recovery_offset: Option<usize>,
    trailing_boost: bool,
    relex_suffix: bool,
}

fn parse_phrase_suffix(suffix: &str) -> ParsedPhraseSuffix<'_> {
    let mut slop = 0;
    let mut prefix = false;
    let mut consumed = 0;
    if suffix.starts_with('*') {
        prefix = true;
        consumed = 1;
    } else if let Some(rest) = suffix.strip_prefix('~') {
        let digit_count = rest.bytes().take_while(u8::is_ascii_digit).count();
        if digit_count != 0 {
            let Ok(parsed_slop) = rest[..digit_count].parse() else {
                return ParsedPhraseSuffix {
                    slop,
                    prefix,
                    boost: None,
                    recovery_offset: Some(0),
                    trailing_boost: false,
                    relex_suffix: true,
                };
            };
            slop = parsed_slop;
            consumed = 1 + digit_count;
        } else {
            return ParsedPhraseSuffix {
                slop,
                prefix,
                boost: None,
                recovery_offset: Some(0),
                trailing_boost: false,
                relex_suffix: true,
            };
        }
    }
    let rest = &suffix[consumed..];
    if let Some(boost_suffix) = rest.strip_prefix('^') {
        let (boost, trailing) = split_boost_component(boost_suffix);
        if let Some(relative) = trailing {
            return ParsedPhraseSuffix {
                slop,
                prefix,
                boost: Some(boost),
                recovery_offset: Some(consumed + 1 + relative),
                trailing_boost: true,
                relex_suffix: false,
            };
        }
        return ParsedPhraseSuffix {
            slop,
            prefix,
            boost: Some(boost_suffix),
            recovery_offset: None,
            trailing_boost: false,
            relex_suffix: false,
        };
    }
    ParsedPhraseSuffix {
        slop,
        prefix,
        boost: None,
        recovery_offset: (!rest.is_empty()).then_some(consumed),
        trailing_boost: false,
        relex_suffix: false,
    }
}

fn find_unescaped(fragment: &str, needle: char) -> Option<usize> {
    let mut escaped = false;
    for (offset, ch) in fragment.char_indices() {
        if escaped {
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == needle {
            return Some(offset);
        }
    }
    None
}

fn split_boost(fragment: &str) -> (&str, Option<&str>, Option<usize>) {
    let Some(first) = find_unescaped(fragment, '^').filter(|offset| *offset != 0) else {
        return (fragment, None, None);
    };
    let suffix = &fragment[first + 1..];
    let (boost, trailing) = split_boost_component(suffix);
    (
        &fragment[..first],
        Some(boost),
        trailing.map(|offset| first + 1 + offset),
    )
}

fn split_boost_component(suffix: &str) -> (&str, Option<usize>) {
    let syntax_end = find_unescaped(suffix, '^').unwrap_or(suffix.len());
    let candidate = &suffix[..syntax_end];
    let integer_digits = candidate.bytes().take_while(u8::is_ascii_digit).count();
    if integer_digits == 0 {
        return (
            candidate,
            (syntax_end != suffix.len()).then_some(syntax_end),
        );
    }

    let mut valid_end = integer_digits;
    if candidate.as_bytes().get(integer_digits) == Some(&b'.') {
        let fraction_digits = candidate[integer_digits + 1..]
            .bytes()
            .take_while(u8::is_ascii_digit)
            .count();
        if fraction_digits != 0 {
            valid_end = integer_digits + 1 + fraction_digits;
        }
    }
    let trailing = if valid_end != candidate.len() {
        Some(valid_end)
    } else {
        (syntax_end != suffix.len()).then_some(syntax_end)
    };
    (&candidate[..valid_end], trailing)
}

fn trim_typed_literal(raw: &str) -> &str {
    if raw.len() >= 2 {
        let bytes = raw.as_bytes();
        if matches!(
            (bytes.first(), bytes.last()),
            (Some(&b'"'), Some(&b'"')) | (Some(&b'\''), Some(&b'\''))
        ) {
            return &raw[1..raw.len() - 1];
        }
    }
    raw
}

#[derive(Debug)]
struct ParsedNode {
    query: Query,
    occur: Option<Occur>,
    from_not: bool,
    dedup_key: SyntaxKey,
}

struct Grammar {
    parser: DefaultQueryParser,
    tokens: Vec<LexToken>,
    cursor: usize,
    diagnostics: Vec<QueryDiagnostic>,
    lowered_atoms: Vec<(String, Option<Query>)>,
    field_scopes: Vec<String>,
}

impl Grammar {
    fn recover_leading_binary_operators(&mut self) {
        while matches!(self.peek(), Some(LexToken::And(_) | LexToken::Or(_))) {
            let offset = self.peek().map_or(0, LexToken::byte_offset);
            self.cursor += 1;
            self.syntax_diagnostic_at("syntax recovery: leading Boolean operator dropped", offset);
        }
    }

    fn recover_operator_as_operand(&mut self) -> bool {
        let Some((raw, offset)) = self.peek().and_then(|token| match token {
            LexToken::And(offset) => Some(("AND", *offset)),
            LexToken::Or(offset) => Some(("OR", *offset)),
            _ => None,
        }) else {
            return false;
        };
        self.tokens[self.cursor] = operator_atom(raw, offset);
        self.syntax_diagnostic_at(
            "syntax recovery: repeated Boolean operator retained as a literal",
            offset,
        );
        true
    }

    fn parse_expression(&mut self, depth: usize) -> Option<ParsedNode> {
        self.parse_or(depth)
    }

    fn parse_or(&mut self, depth: usize) -> Option<ParsedNode> {
        let mut top_nodes = Vec::new();
        let mut top_operands = 0_usize;
        let mut run_nodes = Vec::new();
        let mut run_operands = 0_usize;
        let mut run_explicit = false;
        let mut joined_from_explicit_or = false;

        loop {
            if !self.peek().is_some_and(LexToken::can_start_operand) {
                break;
            }
            run_operands += 1;
            let mut node = self.parse_and(depth);
            let joined_to_explicit_or = matches!(self.peek(), Some(LexToken::Or(_)));
            if let Some(node) = node.as_mut()
                && node.occur == Some(Occur::MustNot)
                && (joined_from_explicit_or || joined_to_explicit_or)
            {
                wrap_direct_negative_or_operand(node);
            }
            if let Some(node) = node {
                run_nodes.push(node);
            }
            if let Some(LexToken::Or(offset)) = self.peek() {
                let offset = *offset;
                self.cursor += 1;
                if !self.peek().is_some_and(LexToken::can_start_operand)
                    && !self.recover_operator_as_operand()
                {
                    self.syntax_diagnostic_at("syntax recovery: OR has no right operand", offset);
                    break;
                }
                run_explicit = true;
                joined_from_explicit_or = true;
            } else if !self.peek().is_some_and(LexToken::can_start_operand) {
                break;
            } else {
                joined_from_explicit_or = false;
                Self::flush_or_run(
                    &mut top_nodes,
                    &mut top_operands,
                    &mut run_nodes,
                    &mut run_operands,
                    &mut run_explicit,
                );
            }
        }

        if top_operands == 0 {
            // The whole chain is one run — a pure explicit-OR chain, a pure
            // implicit chain, or a single operand. Keep the historical flat
            // combination, including its same-level duplicate-retention rule.
            return combine_or(run_nodes, run_operands, run_explicit);
        }
        Self::flush_or_run(
            &mut top_nodes,
            &mut top_operands,
            &mut run_nodes,
            &mut run_operands,
            &mut run_explicit,
        );
        combine_or(top_nodes, top_operands, false)
    }

    /// Close one explicit-`OR` run at an implicit-adjacency boundary.
    ///
    /// The pinned grammar nests an explicit `OR` expression as its own node
    /// beneath the implicit disjunction, so its operands live one level below
    /// operands joined by plain adjacency — and the duplicate-retention
    /// rewrite, which only compares children of the SAME boolean level, never
    /// dedups across that boundary. Flattening the whole mixed chain instead
    /// made Quill drop a trailing operand that duplicates an OR-run member,
    /// scoring exactly one clause short of the oracle (bd-htcun, found by the
    /// bd-bsjw campaign as a clean 2.0x score deficit on
    /// `slices OR title:Nested AND reused slices`).
    fn flush_or_run(
        top_nodes: &mut Vec<ParsedNode>,
        top_operands: &mut usize,
        run_nodes: &mut Vec<ParsedNode>,
        run_operands: &mut usize,
        run_explicit: &mut bool,
    ) {
        if *run_operands == 0 && run_nodes.is_empty() {
            return;
        }
        *top_operands += 1;
        let nodes = std::mem::take(run_nodes);
        let operands = std::mem::replace(run_operands, 0);
        let explicit = std::mem::replace(run_explicit, false);
        if explicit {
            if let Some(node) = combine_or(nodes, operands, true) {
                top_nodes.push(node);
            }
        } else {
            // An adjacency boundary closes a run immediately, so a run
            // without an explicit OR is a single implicit operand.
            top_nodes.extend(nodes);
        }
    }

    fn parse_and(&mut self, depth: usize) -> Option<ParsedNode> {
        let mut nodes = Vec::new();
        let mut syntactic_operands = 1;
        if let Some(mut node) = self.parse_unary(depth) {
            if node.from_not && matches!(self.peek(), Some(LexToken::And(_))) {
                wrap_not_for_and(&mut node);
            }
            nodes.push(node);
        }
        let mut explicit_and = false;
        while let Some(LexToken::And(offset)) = self.peek() {
            let offset = *offset;
            self.cursor += 1;
            if !self.peek().is_some_and(LexToken::can_start_operand)
                && !self.recover_operator_as_operand()
            {
                self.syntax_diagnostic_at("syntax recovery: AND has no right operand", offset);
                break;
            }
            explicit_and = true;
            syntactic_operands += 1;
            if let Some(mut node) = self.parse_unary(depth) {
                if node.from_not {
                    wrap_not_for_and(&mut node);
                }
                nodes.push(node);
            }
        }
        if explicit_and {
            combine_and(nodes, syntactic_operands)
        } else {
            nodes.pop()
        }
    }

    fn parse_unary(&mut self, depth: usize) -> Option<ParsedNode> {
        let mut occur = None;
        let mut unary_offset = None;
        let mut not_count = 0_usize;
        while let Some(token) = self.peek() {
            match token {
                LexToken::Plus(offset) => {
                    unary_offset = Some(*offset);
                    occur = Some(Occur::Must);
                    self.cursor += 1;
                }
                LexToken::Minus(offset) => {
                    unary_offset = Some(*offset);
                    occur = Some(Occur::MustNot);
                    self.cursor += 1;
                }
                LexToken::Not(offset) => {
                    unary_offset = Some(*offset);
                    not_count += 1;
                    self.cursor += 1;
                }
                _ => break,
            }
        }

        let Some(mut node) = self.parse_primary(depth) else {
            if let Some(offset) = unary_offset {
                self.drop_with_diagnostic(
                    QueryDiagnosticKind::DroppedFragment,
                    "unparseable unary fragment dropped",
                    offset,
                    None,
                );
            }
            return None;
        };
        for _ in 1..not_count {
            let dedup_key = take_syntax_key(&mut node.dedup_key);
            node.query = Query::Boolean {
                clauses: vec![BooleanClause::new(Occur::MustNot, node.query)],
                operator: None,
            };
            node.dedup_key = negative_boolean_dedup_key(dedup_key);
        }
        if not_count != 0 {
            // A `NOT` stacked with an explicit `+`/`-` prefix (`NOT -x`) sits
            // outside the contract grammar's single-prefix fragment rule; the
            // pinned oracle collapses the stack to ONE exclusion rather than
            // composing a double negation (bd-251nt, found by the bd-bsjw
            // structure-aware campaign; oracle behavior pinned by
            // stacked_negation_prefixes_pin_oracle_semantics in the lexical
            // crate). `NOT +x` is unprobed on the oracle and takes the same
            // collapse; revisit with a probe before relying on it.
            node.occur = Some(Occur::MustNot);
            node.from_not = true;
        } else if let Some(occur) = occur {
            node.occur = Some(occur);
        }
        Some(node)
    }

    fn parse_primary(&mut self, depth: usize) -> Option<ParsedNode> {
        let token = self.tokens.get(self.cursor)?.clone();
        let token_offset = token.byte_offset();
        match token {
            LexToken::Atom(mut atom) => {
                self.cursor += 1;
                if atom.field.is_none() {
                    atom.field = self.field_scopes.last().cloned();
                }
                let mut dedup_key = atom_dedup_key(&atom);
                let cache_key = atom_cache_key(&atom);
                if let Some(cache_key) = cache_key.as_ref() {
                    if let Some(index) = self
                        .lowered_atoms
                        .iter()
                        .position(|(candidate, _)| candidate == cache_key)
                    {
                        let cached = self.lowered_atoms[index].1.clone();
                        if cached.is_none() {
                            dedup_key = mark_syntax_lowered_away(dedup_key);
                        }
                        let query = cached.unwrap_or_else(|| empty_query_for_syntax(&dedup_key));
                        return Some(ParsedNode {
                            query,
                            occur: None,
                            from_not: false,
                            dedup_key,
                        });
                    }
                }
                let query = self.lower_atom(&atom);
                if let Some(cache_key) = cache_key {
                    self.lowered_atoms.push((cache_key, query.clone()));
                }
                if query.is_none() {
                    dedup_key = mark_syntax_lowered_away(dedup_key);
                }
                let query = query.unwrap_or_else(|| empty_query_for_syntax(&dedup_key));
                Some(ParsedNode {
                    query,
                    occur: None,
                    from_not: false,
                    dedup_key,
                })
            }
            LexToken::Range(mut token) => {
                self.cursor += 1;
                if token.field.is_none() {
                    token.field = self.field_scopes.last().cloned();
                }
                let mut dedup_key = range_dedup_key(&token);
                let lowered = self.lower_range(&token);
                if lowered.is_none() {
                    dedup_key = mark_syntax_lowered_away(dedup_key);
                }
                let query = lowered.unwrap_or_else(|| empty_query_for_syntax(&dedup_key));
                Some(ParsedNode {
                    query,
                    occur: None,
                    from_not: false,
                    dedup_key,
                })
            }
            LexToken::Set(mut token) => {
                self.cursor += 1;
                if token.field.is_none() {
                    token.field = self.field_scopes.last().cloned();
                }
                let mut dedup_key = set_dedup_key(&token);
                let lowered = self.lower_set(&token);
                if lowered.is_none() {
                    dedup_key = mark_syntax_lowered_away(dedup_key);
                }
                let query = lowered.unwrap_or_else(|| empty_query_for_syntax(&dedup_key));
                Some(ParsedNode {
                    query,
                    occur: None,
                    from_not: false,
                    dedup_key,
                })
            }
            LexToken::UnsupportedRegex {
                mut field,
                pattern,
                boost,
                ..
            } => {
                self.cursor += 1;
                if field.is_none() {
                    field = self.field_scopes.last().cloned();
                }
                let dedup_key = mark_syntax_lowered_away(boost_syntax_key(
                    SyntaxKey::Regex { field, pattern },
                    boost.as_deref(),
                ));
                Some(ParsedNode {
                    query: empty_query_for_syntax(&dedup_key),
                    occur: None,
                    from_not: false,
                    dedup_key,
                })
            }
            LexToken::Dropped(_) => {
                self.cursor += 1;
                None
            }
            LexToken::LeftParen { byte_offset, field } => {
                self.cursor += 1;
                if depth >= MAX_QUERY_DEPTH {
                    self.skip_group();
                    self.drop_with_diagnostic(
                        QueryDiagnosticKind::DepthLimit,
                        "query group nesting limit reached; fragment dropped",
                        byte_offset,
                        None,
                    );
                    return None;
                }
                let scoped = field.is_some();
                if let Some(field_name) = field {
                    self.field_scopes.push(field_name);
                }
                self.recover_leading_binary_operators();
                let parsed = self.parse_expression(depth + 1);
                if scoped {
                    let _ = self.field_scopes.pop();
                }
                let group_boost = if let Some(LexToken::RightParen {
                    byte_offset: close_offset,
                    boost,
                }) = self.peek().cloned()
                {
                    self.cursor += 1;
                    Some((boost, close_offset))
                } else {
                    self.syntax_diagnostic_at(
                        "syntax recovery: missing closing parenthesis",
                        byte_offset,
                    );
                    None
                };
                parsed.map(|mut node| {
                    if let Some((boost, close_offset)) = group_boost {
                        node.query =
                            self.apply_raw_boost(node.query, boost.as_deref(), close_offset, None);
                        node.dedup_key = boost_syntax_key(node.dedup_key, boost.as_deref());
                    }
                    node
                })
            }
            LexToken::RightParen { .. } | LexToken::And(_) | LexToken::Or(_) => None,
            LexToken::Not(_) | LexToken::Plus(_) | LexToken::Minus(_) => {
                self.syntax_diagnostic_at(
                    "syntax recovery: unexpected unary token dropped",
                    token_offset,
                );
                None
            }
        }
    }

    fn lower_atom(&mut self, atom: &AtomToken) -> Option<Query> {
        if !atom.delimiter.is_quoted() && atom.raw == "*" {
            if atom.field.is_none() {
                return Some(self.apply_boost(Query::All, atom));
            }
            self.drop_with_diagnostic(
                QueryDiagnosticKind::UnsupportedField,
                "field-scoped match-all requires an unsupported existence query",
                atom.byte_offset,
                None,
            );
            return None;
        }

        let fields = if let Some(field_name) = atom.field.as_deref() {
            let Some(descriptor) = self
                .parser
                .schema
                .fields
                .iter()
                .find(|field| field.name == field_name)
                .copied()
            else {
                self.drop_with_diagnostic(
                    QueryDiagnosticKind::UnknownField,
                    &format!("unknown field {field_name}; unsupported field fragment dropped"),
                    atom.byte_offset,
                    Some(atom.raw.clone()),
                );
                return None;
            };
            vec![QueryField::new(
                descriptor.id,
                if descriptor.name == TITLE_FIELD_NAME {
                    TITLE_BOOST
                } else {
                    1.0
                },
            )]
        } else {
            self.parser.default_fields.to_vec()
        };

        let mut field_queries = Vec::with_capacity(fields.len());
        for field in fields {
            let Some(descriptor) = self.field_by_id(field.field_id) else {
                continue;
            };
            let Some(query) = self.analyze_field(descriptor, field, atom) else {
                continue;
            };
            field_queries.push(query);
        }
        let query = merge_field_queries(field_queries)?;
        Some(self.apply_boost(query, atom))
    }

    fn lower_range(&mut self, token: &RangeToken) -> Option<Query> {
        let descriptor = self.resolve_typed_field(token.field.as_deref(), token.byte_offset)?;
        let (lower, lower_invalid) = self.lower_bound(
            descriptor,
            &token.lower,
            token.lower_inclusive,
            token.byte_offset,
        );
        let (upper, upper_invalid) = self.lower_bound(
            descriptor,
            &token.upper,
            token.upper_inclusive,
            token.byte_offset,
        );
        if (lower_invalid && upper_invalid)
            || matches!((&lower, &upper), (Bound::Unbounded, Bound::Unbounded))
        {
            self.push_diagnostic(QueryDiagnostic {
                kind: QueryDiagnosticKind::DroppedFragment,
                message: "typed range without a usable bound dropped".to_owned(),
                byte_offset: Some(token.byte_offset),
                fragment: None,
            });
            return None;
        }
        let query = Query::Range {
            field_id: descriptor.id,
            lower,
            upper,
        };
        Some(self.apply_raw_boost(query, token.boost.as_deref(), token.byte_offset, None))
    }

    fn lower_set(&mut self, token: &SetToken) -> Option<Query> {
        let descriptor = self.resolve_typed_field(token.field.as_deref(), token.byte_offset)?;
        let mut values = Vec::with_capacity(token.values.len());
        for raw in &token.values {
            match Self::lower_typed_value(descriptor, raw) {
                Some(value) if !values.contains(&value) => values.push(value),
                Some(_) => {}
                None => self.push_diagnostic(QueryDiagnostic {
                    kind: QueryDiagnosticKind::InvalidTypedValue,
                    message: format!(
                        "invalid typed set member for field {}; member dropped",
                        descriptor.name
                    ),
                    byte_offset: Some(token.byte_offset),
                    fragment: None,
                }),
            }
        }
        let query = Query::Set {
            field_id: descriptor.id,
            values,
        };
        Some(self.apply_raw_boost(query, token.boost.as_deref(), token.byte_offset, None))
    }

    fn resolve_typed_field(
        &mut self,
        field_name: Option<&str>,
        byte_offset: usize,
    ) -> Option<FieldDescriptor> {
        let Some(field_name) = field_name else {
            self.drop_with_diagnostic(
                QueryDiagnosticKind::DroppedFragment,
                "typed range or set requires a field scope",
                byte_offset,
                None,
            );
            return None;
        };
        let Some(descriptor) = self
            .parser
            .schema
            .fields
            .iter()
            .find(|field| field.name == field_name)
            .copied()
        else {
            self.drop_with_diagnostic(
                QueryDiagnosticKind::UnknownField,
                &format!("unknown field {field_name}; typed fragment dropped"),
                byte_offset,
                None,
            );
            return None;
        };
        let queryable = match descriptor.kind {
            FieldKind::Keyword => true,
            FieldKind::Text {
                analyzer: AnalyzerKind::FrankensearchDefault,
                ..
            } => true,
            FieldKind::I64 { indexed, fast } | FieldKind::U64 { indexed, fast } => indexed || fast,
            FieldKind::StoredOnly | FieldKind::Text { .. } => false,
        };
        if !queryable {
            self.drop_with_diagnostic(
                QueryDiagnosticKind::UnsupportedField,
                &format!("field {} does not support typed queries", descriptor.name),
                byte_offset,
                None,
            );
            return None;
        }
        Some(descriptor)
    }

    fn lower_bound(
        &mut self,
        descriptor: FieldDescriptor,
        raw: &str,
        inclusive: bool,
        byte_offset: usize,
    ) -> (Bound<QueryValue>, bool) {
        if raw == "*" {
            return (Bound::Unbounded, false);
        }
        let Some(value) = Self::lower_typed_value(descriptor, raw) else {
            self.push_diagnostic(QueryDiagnostic {
                kind: QueryDiagnosticKind::InvalidTypedValue,
                message: format!(
                    "invalid typed range bound for field {}; bound made unbounded",
                    descriptor.name
                ),
                byte_offset: Some(byte_offset),
                fragment: None,
            });
            return (Bound::Unbounded, true);
        };
        if inclusive {
            (Bound::Included(value), false)
        } else {
            (Bound::Excluded(value), false)
        }
    }

    fn lower_typed_value(descriptor: FieldDescriptor, raw: &str) -> Option<QueryValue> {
        let raw = trim_typed_literal(raw);
        match descriptor.kind {
            FieldKind::I64 { indexed, fast } if indexed || fast => {
                raw.parse().ok().map(QueryValue::I64)
            }
            FieldKind::U64 { indexed, fast } if indexed || fast => {
                raw.parse().ok().map(QueryValue::U64)
            }
            FieldKind::Keyword => Some(QueryValue::Str(raw.to_owned())),
            FieldKind::Text {
                analyzer: AnalyzerKind::FrankensearchDefault,
                ..
            } => {
                let mut analyzer = FrankensearchTokenizer::default();
                let mut terms = Vec::new();
                let report = analyze_admitted(
                    &mut analyzer,
                    AnalyzerKind::FrankensearchDefault,
                    raw,
                    &mut |token| terms.push(token.text.clone()),
                )
                .ok()?;
                if report.oversized_tokens == 0 && terms.len() == 1 {
                    terms.pop().map(QueryValue::Str)
                } else {
                    None
                }
            }
            FieldKind::StoredOnly
            | FieldKind::Text { .. }
            | FieldKind::I64 { .. }
            | FieldKind::U64 { .. } => None,
        }
    }

    fn field_by_id(&self, field_id: u16) -> Option<FieldDescriptor> {
        self.parser
            .schema
            .fields
            .get(usize::from(field_id))
            .copied()
            .filter(|field| field.id == field_id)
    }

    fn analyze_field(
        &mut self,
        descriptor: FieldDescriptor,
        field: QueryField,
        atom: &AtomToken,
    ) -> Option<Query> {
        match descriptor.kind {
            FieldKind::Text {
                analyzer: AnalyzerKind::FrankensearchDefault,
                ..
            } => {
                let mut analyzer = FrankensearchTokenizer::default();
                let mut terms = Vec::new();
                let Ok(report) = analyze_admitted(
                    &mut analyzer,
                    AnalyzerKind::FrankensearchDefault,
                    &atom.raw,
                    &mut |token| {
                        terms.push(PositionedTerm::new(token.position, token.text.clone()));
                    },
                ) else {
                    self.drop_with_diagnostic(
                        QueryDiagnosticKind::UnsupportedField,
                        "default analyzer implementation is unavailable",
                        atom.byte_offset,
                        None,
                    );
                    return None;
                };
                if report.oversized_tokens != 0 {
                    return Some(Query::Empty);
                }
                match terms.len() {
                    0 => None,
                    1 if atom.prefix => {
                        self.push_diagnostic(QueryDiagnostic {
                            kind: QueryDiagnosticKind::DroppedFragment,
                            message: "phrase prefix requires at least two analyzed terms; field branch dropped"
                                .to_owned(),
                            byte_offset: Some(atom.byte_offset),
                            fragment: None,
                        });
                        None
                    }
                    1 => terms.pop().map(|term| Query::Term {
                        fields: vec![field],
                        text: term.text,
                    }),
                    _ => Some(Query::Phrase {
                        fields: vec![field],
                        terms,
                        slop: atom.slop,
                        prefix: atom.prefix,
                    }),
                }
            }
            FieldKind::Keyword if atom.prefix => {
                self.push_diagnostic(QueryDiagnostic {
                    kind: QueryDiagnosticKind::DroppedFragment,
                    message:
                        "phrase prefix requires at least two analyzed terms; field branch dropped"
                            .to_owned(),
                    byte_offset: Some(atom.byte_offset),
                    fragment: None,
                });
                None
            }
            FieldKind::Keyword => (!atom.raw.is_empty()).then(|| Query::Term {
                fields: vec![field],
                text: atom.raw.clone(),
            }),
            FieldKind::Text { .. } => {
                self.drop_with_diagnostic(
                    QueryDiagnosticKind::UnsupportedField,
                    &format!(
                        "field {} uses an analyzer unsupported by the default parser",
                        descriptor.name
                    ),
                    atom.byte_offset,
                    Some(atom.raw.clone()),
                );
                None
            }
            FieldKind::StoredOnly
            | FieldKind::I64 { indexed: false, .. }
            | FieldKind::U64 { indexed: false, .. }
            | FieldKind::I64 { indexed: true, .. }
            | FieldKind::U64 { indexed: true, .. } => {
                self.drop_with_diagnostic(
                    QueryDiagnosticKind::UnsupportedField,
                    &format!(
                        "field {} is not a text field supported by this parser",
                        descriptor.name
                    ),
                    atom.byte_offset,
                    Some(atom.raw.clone()),
                );
                None
            }
        }
    }

    fn apply_boost(&mut self, query: Query, atom: &AtomToken) -> Query {
        self.apply_raw_boost(
            query,
            atom.boost.as_deref(),
            atom.byte_offset,
            Some(atom.raw.clone()),
        )
    }

    fn apply_raw_boost(
        &mut self,
        query: Query,
        raw_boost: Option<&str>,
        byte_offset: usize,
        fragment: Option<String>,
    ) -> Query {
        let Some(raw_boost) = raw_boost else {
            return query;
        };
        match parse_boost_factor(raw_boost) {
            Some(factor) if factor.to_bits() == 1.0_f32.to_bits() => query,
            Some(factor) => Query::Boost {
                query: Box::new(query),
                factor,
            },
            _ => {
                self.push_diagnostic(QueryDiagnostic {
                    kind: QueryDiagnosticKind::InvalidBoost,
                    message: format!("invalid boost {raw_boost:?}; recovered branch without boost"),
                    byte_offset: Some(byte_offset),
                    fragment,
                });
                query
            }
        }
    }

    fn skip_group(&mut self) {
        let mut depth = 1_usize;
        while let Some(token) = self.tokens.get(self.cursor) {
            self.cursor += 1;
            match token {
                LexToken::LeftParen { .. } => depth += 1,
                LexToken::RightParen { .. } => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                _ => {}
            }
        }
    }

    fn peek(&self) -> Option<&LexToken> {
        self.tokens.get(self.cursor)
    }

    fn push_diagnostic(&mut self, diagnostic: QueryDiagnostic) {
        emit_diagnostic(&mut self.diagnostics, diagnostic);
    }

    fn drop_with_diagnostic(
        &mut self,
        kind: QueryDiagnosticKind,
        message: &str,
        byte_offset: usize,
        fragment: Option<String>,
    ) {
        self.push_diagnostic(QueryDiagnostic {
            kind,
            message: message.to_owned(),
            byte_offset: Some(byte_offset),
            fragment,
        });
    }

    fn syntax_diagnostic_at(&mut self, message: &str, byte_offset: usize) {
        self.push_diagnostic(QueryDiagnostic {
            kind: QueryDiagnosticKind::SyntaxRecovery,
            message: message.to_owned(),
            byte_offset: Some(byte_offset),
            fragment: None,
        });
    }

    fn recover_trailing_tokens(&mut self) {
        while let Some(token) = self.tokens.get(self.cursor) {
            let offset = token.byte_offset();
            self.cursor += 1;
            self.drop_with_diagnostic(
                QueryDiagnosticKind::DroppedFragment,
                "unparseable trailing fragment dropped",
                offset,
                None,
            );
        }
    }
}

fn wrap_direct_negative_or_operand(node: &mut ParsedNode) {
    let dedup_key = take_syntax_key(&mut node.dedup_key);
    let query = std::mem::replace(&mut node.query, Query::Empty);
    node.query = Query::Boolean {
        clauses: vec![BooleanClause::new(Occur::MustNot, query)],
        operator: None,
    };
    node.occur = None;
    node.from_not = false;
    node.dedup_key = negative_boolean_dedup_key(dedup_key);
}

fn wrap_not_for_and(node: &mut ParsedNode) {
    let dedup_key = take_syntax_key(&mut node.dedup_key);
    let query = std::mem::replace(&mut node.query, Query::Empty);
    node.query = Query::Boolean {
        clauses: vec![BooleanClause::new(Occur::MustNot, query)],
        operator: None,
    };
    node.occur = None;
    node.from_not = false;
    node.dedup_key = negative_boolean_dedup_key(dedup_key);
}

fn negative_boolean_dedup_key(child: SyntaxKey) -> SyntaxKey {
    SyntaxKey::Boolean(vec![(Some(Occur::MustNot), child)])
}

/// Mirror the pinned grammar's lowering: an optional (`Should`) clause whose
/// child is an all-optional boolean splices its children into the parent.
///
/// Duplicate retention already ran on the NESTED syntax (where the pinned
/// grammar keeps explicit-`OR` runs and parenthesised groups one level below
/// the implicit disjunction — bd-htcun), but the oracle then FLATTENS
/// pure-disjunctive nesting for execution, so score accumulation associates
/// left-to-right over one flat clause list. Executing the nested shape
/// instead diverges by ULPs on boosted mixes (campaign finding: 2-ULP
/// `RankMismatch` on `((small^3.0 OR when AND this reused) OR tracing NOT
/// generic)`). Flattening after dedup reproduces the oracle's association
/// exactly. A non-`Should` occurrence or a mixed-occurrence child is a
/// boundary, as is any non-boolean node (boosts included).
fn flatten_should_of_should(query: &mut Query) {
    let Query::Boolean { clauses, .. } = query else {
        return;
    };
    for clause in clauses.iter_mut() {
        flatten_should_of_should(&mut clause.query);
    }
    let mut flattened = Vec::with_capacity(clauses.len());
    for clause in clauses.drain(..) {
        let splice = clause.occur == Occur::Should
            && matches!(
                &clause.query,
                Query::Boolean { clauses: children, .. }
                    if children
                        .iter()
                        .all(|child| child.occur == Occur::Should)
            );
        if splice {
            let Query::Boolean {
                clauses: children, ..
            } = clause.query
            else {
                unreachable!("splice guard matched a boolean");
            };
            flattened.extend(children);
        } else {
            flattened.push(clause);
        }
    }
    *clauses = flattened;
}

fn combine_and(nodes: Vec<ParsedNode>, syntactic_operands: usize) -> Option<ParsedNode> {
    combine_boolean(
        nodes,
        Occur::Must,
        Some(BooleanOperator::And),
        syntactic_operands,
    )
}

fn combine_or(
    nodes: Vec<ParsedNode>,
    syntactic_operands: usize,
    explicit_or: bool,
) -> Option<ParsedNode> {
    combine_boolean(
        nodes,
        Occur::Should,
        explicit_or.then_some(BooleanOperator::Or),
        syntactic_operands,
    )
}

fn combine_boolean(
    nodes: Vec<ParsedNode>,
    default_occur: Occur,
    operator: Option<BooleanOperator>,
    syntactic_operands: usize,
) -> Option<ParsedNode> {
    let mut clauses = Vec::with_capacity(nodes.len() + 1);
    let mut keys = Vec::with_capacity(nodes.len());
    for node in nodes {
        let syntax_occur = node.occur.or(match operator {
            Some(BooleanOperator::And) => Some(Occur::Must),
            Some(BooleanOperator::Or) => Some(Occur::Should),
            None => None,
        });
        let occur = node.occur.unwrap_or(default_occur);
        keys.push((syntax_occur, node.dedup_key));
        clauses.push(BooleanClause::new(occur, node.query));
    }
    if clauses.is_empty() {
        return None;
    }

    if clauses.len() == 1 && syntactic_operands == 1 && clauses[0].occur != Occur::MustNot {
        let clause = clauses.pop()?;
        let (_, dedup_key) = keys.pop()?;
        return Some(ParsedNode {
            query: clause.query,
            occur: None,
            from_not: false,
            dedup_key,
        });
    }

    let dedup_key = SyntaxKey::Boolean(keys);
    Some(ParsedNode {
        query: Query::Boolean { clauses, operator },
        occur: None,
        from_not: false,
        dedup_key,
    })
}

/// Mirror `tantivy-query-grammar`'s parser-AST rewrite while the raw syntax
/// identity is still paired with the lowered query tree.
///
/// The pinned grammar recurses only through Boolean clauses, so duplicates
/// beneath a non-identity boost deliberately survive. At each visited Boolean
/// it retains the first exact `(Option<Occur>, child)` pair and then flattens a
/// singleton child whose outer occurrence is implicit, transferring the
/// child's explicit occurrence to the parent clause.
fn rewrite_parser_syntax(node: &mut ParsedNode) {
    rewrite_parser_syntax_inner(&mut node.query, &mut node.dedup_key);
}

fn rewrite_parser_syntax_inner(query: &mut Query, syntax: &mut SyntaxKey) {
    let SyntaxKey::Boolean(keys) = syntax else {
        return;
    };
    let Query::Boolean { clauses, .. } = query else {
        debug_assert!(false, "Boolean syntax key must pair with a Boolean query");
        return;
    };
    if clauses.len() != keys.len() {
        debug_assert_eq!(clauses.len(), keys.len());
        return;
    }

    for (clause, (_, child_key)) in clauses.iter_mut().zip(keys.iter_mut()) {
        rewrite_parser_syntax_inner(&mut clause.query, child_key);
    }

    let mut index = 0;
    while index < keys.len() {
        if keys[..index].contains(&keys[index]) {
            keys.remove(index);
            clauses.remove(index);
        } else {
            index += 1;
        }
    }

    for (key, outer_clause) in keys.iter_mut().zip(clauses.iter_mut()) {
        let (outer_occur, child_key) = key;
        if outer_occur.is_none()
            && let SyntaxKey::Boolean(inner_keys) = child_key
            && let Query::Boolean {
                clauses: inner_clauses,
                ..
            } = &mut outer_clause.query
            && inner_keys.len() == 1
            && inner_clauses.len() == 1
        {
            let (inner_occur, inner_key) = inner_keys.remove(0);
            let inner_clause = inner_clauses.remove(0);
            *outer_occur = inner_occur;
            *child_key = inner_key;
            *outer_clause = inner_clause;
        }
    }
}

/// Mirror Tantivy's post-lowering `trim_ast` for parser leaves that failed
/// semantic lowering. Intentional match-none leaves remain executable; only
/// syntax explicitly tagged `LoweredAway` disappears. A boost is a
/// boundary even when its child failed lowering.
fn trim_parser_dropped(query: &mut Query, syntax: &SyntaxKey) -> bool {
    if matches!(syntax, SyntaxKey::LoweredAway(_)) {
        return true;
    }
    if let SyntaxKey::Boost {
        query: child_key, ..
    } = syntax
    {
        if let Query::Boost {
            query: child_query, ..
        } = query
        {
            let _ = trim_parser_dropped(child_query, child_key);
            return false;
        }
        return trim_parser_dropped(query, child_key);
    }
    let SyntaxKey::Boolean(keys) = syntax else {
        return false;
    };
    let Query::Boolean { clauses, .. } = query else {
        debug_assert!(false, "Boolean syntax key must pair with a Boolean query");
        return false;
    };
    if clauses.len() != keys.len() {
        debug_assert_eq!(clauses.len(), keys.len());
        return false;
    }

    let original = std::mem::take(clauses);
    clauses.reserve(original.len());
    for (mut clause, (_, child_key)) in original.into_iter().zip(keys) {
        if !trim_parser_dropped(&mut clause.query, child_key) {
            clauses.push(clause);
        }
    }
    if clauses.is_empty() {
        *query = Query::Empty;
        true
    } else {
        false
    }
}

fn repair_root_all_negative(node: &mut Option<ParsedNode>, grammar: &mut Grammar) {
    let Some(parsed) = node else {
        return;
    };
    if !query_is_all_negative(&parsed.query) {
        return;
    }
    if !add_all_to_negative_root(&mut parsed.query) {
        return;
    }
    grammar.push_diagnostic(QueryDiagnostic {
        kind: QueryDiagnosticKind::AllNegativeRepair,
        message: "all-negative query repaired with All".to_owned(),
        byte_offset: None,
        fragment: None,
    });
}

fn query_is_all_negative(query: &Query) -> bool {
    match query {
        Query::Boolean { clauses, .. } => {
            !clauses.is_empty()
                && clauses.iter().all(|clause| {
                    clause.occur == Occur::MustNot || query_is_all_negative(&clause.query)
                })
        }
        Query::Boost { query, .. } => query_is_all_negative(query),
        Query::Empty
        | Query::All
        | Query::Term { .. }
        | Query::Phrase { .. }
        | Query::Range { .. }
        | Query::Set { .. }
        | Query::Glob { .. } => false,
    }
}

fn add_all_to_negative_root(query: &mut Query) -> bool {
    match query {
        Query::Boolean { clauses, .. } => {
            clauses.push(BooleanClause::new(Occur::Should, Query::All));
            true
        }
        Query::Boost { query, .. } => add_all_to_negative_root(query),
        Query::Empty
        | Query::All
        | Query::Term { .. }
        | Query::Phrase { .. }
        | Query::Range { .. }
        | Query::Set { .. }
        | Query::Glob { .. } => false,
    }
}

fn merge_field_queries(queries: Vec<Query>) -> Option<Query> {
    let first = queries.first()?.clone();
    match first {
        Query::Term { text, .. }
            if queries.iter().all(
                |query| matches!(query, Query::Term { text: candidate, .. } if candidate == &text),
            ) =>
        {
            let mut fields = Vec::new();
            for query in queries {
                let Query::Term {
                    fields: query_fields,
                    ..
                } = query
                else {
                    return None;
                };
                fields.extend(query_fields);
            }
            Some(Query::Term { fields, text })
        }
        Query::Phrase {
            terms,
            slop,
            prefix,
            ..
        } if queries.iter().all(|query| {
            matches!(
                query,
                Query::Phrase {
                    terms: candidate_terms,
                    slop: candidate_slop,
                    prefix: candidate_prefix,
                    ..
                } if candidate_terms == &terms
                    && *candidate_slop == slop
                    && *candidate_prefix == prefix
            )
        }) =>
        {
            let mut fields = Vec::new();
            for query in queries {
                let Query::Phrase {
                    fields: query_fields,
                    ..
                } = query
                else {
                    return None;
                };
                fields.extend(query_fields);
            }
            Some(Query::Phrase {
                fields,
                terms,
                slop,
                prefix,
            })
        }
        Query::Empty if queries.iter().all(Query::is_empty) => Some(Query::Empty),
        _ => Some(Query::Boolean {
            clauses: queries
                .into_iter()
                .map(|query| BooleanClause::new(Occur::Should, query))
                .collect(),
            operator: None,
        }),
    }
}

/// Source restriction used by the native CASS query parser.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum CassSourceFilter {
    /// Do not restrict the durable source identity.
    #[default]
    All,
    /// Match documents indexed from the local machine.
    Local,
    /// Match documents indexed through SSH.
    Remote,
    /// Match one exact source identifier.
    SourceId(String),
}

/// Structured filters appended to one native CASS query.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CassQueryFilters {
    /// Exact agent names, combined with OR and required by the root query.
    pub agents: Vec<String>,
    /// Exact workspace names, combined with OR and required by the root query.
    pub workspaces: Vec<String>,
    /// Inclusive lower timestamp bound.
    pub created_from: Option<i64>,
    /// Inclusive upper timestamp bound.
    pub created_to: Option<i64>,
    /// Durable source restriction.
    pub source_filter: CassSourceFilter,
}

/// Shipping wildcard classes for one CASS term.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CassWildcardPattern {
    /// No wildcard, lowered through exact term queries.
    Exact(String),
    /// One trailing wildcard, accelerated through the prefix fields.
    Prefix(String),
    /// One leading wildcard, lowered through regex expansion.
    Suffix(String),
    /// Leading and trailing wildcards, lowered through regex expansion.
    Substring(String),
    /// One or more interior wildcards, lowered through regex expansion.
    Complex(String),
}

impl CassWildcardPattern {
    /// Classify and lowercase one sanitized CASS query term.
    #[must_use]
    pub fn parse(term: &str) -> Self {
        let starts_with_star = term.starts_with('*');
        let ends_with_star = term.ends_with('*');
        let core = term.trim_matches('*');
        if core.is_empty() {
            return Self::Exact(String::new());
        }
        if core.contains('*') {
            return Self::Complex(term.to_lowercase());
        }
        let core = core.to_lowercase();
        match (starts_with_star, ends_with_star) {
            (true, true) => Self::Substring(core),
            (true, false) => Self::Suffix(core),
            (false, true) => Self::Prefix(core),
            (false, false) => Self::Exact(core),
        }
    }

    /// Return the anchor-free Tantivy FST regex for regex-backed classes.
    ///
    /// The FST engine matches complete terms and rejects `^` / `$` assertions.
    #[must_use]
    pub fn to_regex(&self) -> Option<String> {
        match self {
            Self::Suffix(core) => Some(format!(".*{}", cass_escape_regex(core))),
            Self::Substring(core) => Some(format!(".*{}.*", cass_escape_regex(core))),
            Self::Complex(pattern) => Some(cass_complex_regex(pattern)),
            Self::Exact(_) | Self::Prefix(_) => None,
        }
    }
}

/// Sanitize a CASS query using the shipping hyphen-normalize boundary.
///
/// Alphanumeric scalars, wildcards, quotes, and hyphens survive. Every other
/// scalar becomes a space so later splitting matches the indexed tokenizer.
#[must_use]
pub fn cass_sanitize_query(raw: &str) -> String {
    raw.chars()
        .map(|ch| {
            if ch.is_alphanumeric() || matches!(ch, '*' | '"' | '-') {
                ch
            } else {
                ' '
            }
        })
        .collect()
}

fn cass_escape_regex(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len().saturating_mul(2));
    for ch in value.chars() {
        if matches!(
            ch,
            '\\' | '.' | '+' | '*' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '^' | '$'
        ) {
            escaped.push('\\');
        }
        escaped.push(ch);
    }
    escaped
}

fn cass_complex_regex(pattern: &str) -> String {
    let mut regex = String::with_capacity(pattern.len().saturating_mul(2).saturating_add(2));
    if pattern.starts_with('*') {
        regex.push_str(".*");
    }
    let core = pattern.trim_start_matches('*').trim_end_matches('*');
    for ch in core.chars() {
        if ch == '*' {
            regex.push_str(".*");
        } else {
            if matches!(
                ch,
                '\\' | '.' | '+' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '^' | '$'
            ) {
                regex.push('\\');
            }
            regex.push(ch);
        }
    }
    if pattern.ends_with('*') {
        regex.push_str(".*");
    }
    regex
}

/// Invalid schema configuration for [`CassQueryParser`].
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum CassQueryParserConfigError {
    /// The schema violates the dense descriptor contract.
    #[error("CASS query schema {schema} is invalid: {detail}")]
    InvalidSchema {
        /// Schema diagnostic name.
        schema: &'static str,
        /// Descriptor validation detail.
        detail: String,
    },
    /// A mandatory CASS field is absent.
    #[error("CASS query field {field} is absent from schema {schema}")]
    MissingField {
        /// Schema diagnostic name.
        schema: &'static str,
        /// Missing field name.
        field: &'static str,
    },
    /// A mandatory field has the wrong durable shape.
    #[error("CASS query field {field} has an incompatible shape in schema {schema}")]
    InvalidField {
        /// Schema diagnostic name.
        schema: &'static str,
        /// Invalid field name.
        field: &'static str,
    },
}

#[derive(Debug, Clone, Copy)]
struct CassParserFields {
    agent: u16,
    workspace: u16,
    created_at: u16,
    title: u16,
    content: u16,
    title_prefix: u16,
    content_prefix: u16,
    source_id: u16,
    origin_kind: u16,
}

impl CassParserFields {
    fn searchable(self) -> Vec<QueryField> {
        [
            self.title,
            self.content,
            self.title_prefix,
            self.content_prefix,
        ]
        .into_iter()
        .map(|field_id| QueryField::new(field_id, 1.0))
        .collect()
    }

    fn regex_fields(self) -> Vec<u16> {
        vec![self.content, self.title]
    }
}

/// Native parser for the intentionally non-standard CASS Boolean grammar.
///
/// OR binds tighter than AND. Negation is idempotent rather than parity-based,
/// and a negative used as an OR operand or as the complete root is wrapped in
/// `All + MustNot` so it denotes a complement.
#[derive(Debug, Clone, Copy)]
pub struct CassQueryParser {
    fields: CassParserFields,
}

impl CassQueryParser {
    /// Bind the parser to a CASS-compatible schema.
    ///
    /// # Errors
    ///
    /// Returns a typed configuration error for a missing or incompatible
    /// field, or for an invalid schema descriptor.
    pub fn new(schema: SchemaDescriptor) -> Result<Self, CassQueryParserConfigError> {
        schema
            .validate()
            .map_err(|error| CassQueryParserConfigError::InvalidSchema {
                schema: schema.name,
                detail: error.to_string(),
            })?;
        let keyword = |kind| kind == FieldKind::Keyword;
        let cass_text = |kind| {
            matches!(
                kind,
                FieldKind::Text {
                    analyzer: AnalyzerKind::CassHyphenNormalize,
                    positions: true,
                }
            )
        };
        let cass_prefix = |kind| {
            matches!(
                kind,
                FieldKind::Text {
                    analyzer: AnalyzerKind::CassPrefixNormalize,
                    positions: false,
                }
            )
        };
        let created_at = |kind| {
            matches!(
                kind,
                FieldKind::I64 {
                    indexed: true,
                    fast: true,
                }
            )
        };
        Ok(Self {
            fields: CassParserFields {
                agent: required_cass_field(schema, "agent", keyword)?.id,
                workspace: required_cass_field(schema, "workspace", keyword)?.id,
                created_at: required_cass_field(schema, "created_at", created_at)?.id,
                title: required_cass_field(schema, "title", cass_text)?.id,
                content: required_cass_field(schema, "content", cass_text)?.id,
                title_prefix: required_cass_field(schema, "title_prefix", cass_prefix)?.id,
                content_prefix: required_cass_field(schema, "content_prefix", cass_prefix)?.id,
                source_id: required_cass_field(schema, "source_id", keyword)?.id,
                origin_kind: required_cass_field(schema, "origin_kind", keyword)?.id,
            },
        })
    }

    /// Parse one CASS query and append its structured filters.
    #[must_use]
    pub fn parse(&self, raw_query: &str, filters: &CassQueryFilters) -> ParsedQuery {
        let (admitted_query, was_truncated) = truncated_prefix(raw_query);
        self.parse_admitted(admitted_query, was_truncated, filters, |_| {})
    }

    /// Parse one CASS query and retain the exact lexer/lowering witness.
    ///
    /// This calls the same implementation as [`Self::parse`]. It exists for
    /// dev-only conformance adapters that must observe the real parser boundary
    /// without echoing or independently reparsing the input corpus.
    /// # Errors
    ///
    /// Returns a typed bound error before parsing or cloning evidence when the
    /// raw query, filter count, individual filter value, or aggregate filter
    /// bytes exceed the diagnostic contract.
    #[cfg(any(test, feature = "conformance-internals"))]
    pub fn parse_observed(
        &self,
        raw_query: &str,
        filters: &CassQueryFilters,
    ) -> Result<CassParsedQueryObservation, CassQueryObservationError> {
        cass_observation_preflight(raw_query, filters)?;
        let (admitted_query, was_truncated) = truncated_prefix(raw_query);
        let mut observed_tokens = Vec::new();
        let parsed = self.parse_admitted(admitted_query, was_truncated, filters, |tokens| {
            observed_tokens.extend(tokens.iter().map(CassLexToken::observed));
        });
        Ok(CassParsedQueryObservation {
            parsed,
            sanitized_query: cass_sanitize_query(admitted_query),
            admitted_query: admitted_query.to_owned(),
            tokens: observed_tokens,
        })
    }

    fn parse_admitted<Observe>(
        &self,
        admitted_query: &str,
        was_truncated: bool,
        filters: &CassQueryFilters,
        mut observe_tokens: Observe,
    ) -> ParsedQuery
    where
        Observe: FnMut(&[CassLexToken]),
    {
        let mut diagnostics = Vec::new();
        if was_truncated {
            emit_diagnostic(
                &mut diagnostics,
                QueryDiagnostic {
                    kind: QueryDiagnosticKind::Truncated,
                    message: format!(
                        "CASS query truncated to {MAX_QUERY_LENGTH} Unicode scalar values"
                    ),
                    byte_offset: Some(admitted_query.len()),
                    fragment: None,
                },
            );
        }
        let tokens = cass_lex(admitted_query, &mut diagnostics);
        observe_tokens(&tokens);
        let mut grammar = CassGrammar {
            parser: *self,
            tokens,
            diagnostics,
        };
        let parsed = grammar.parse();
        let root = parsed.map_or(Query::All, |node| {
            if node.negative {
                cass_complement(node.query)
            } else {
                node.query
            }
        });
        let query = self.apply_filters(root, filters);
        ParsedQuery {
            query,
            explanation: classify_query(admitted_query),
            diagnostics: grammar.diagnostics,
            was_truncated,
        }
    }

    fn apply_filters(&self, root: Query, filters: &CassQueryFilters) -> Query {
        if filters.agents.is_empty()
            && filters.workspaces.is_empty()
            && filters.created_from.is_none()
            && filters.created_to.is_none()
            && filters.source_filter == CassSourceFilter::All
        {
            return root;
        }
        let mut clauses = Vec::with_capacity(5);
        clauses.push(BooleanClause::new(Occur::Must, root));
        if let Some(filter) = cass_string_filter(self.fields.agent, &filters.agents) {
            clauses.push(BooleanClause::new(Occur::Must, filter));
        }
        if let Some(filter) = cass_string_filter(self.fields.workspace, &filters.workspaces) {
            clauses.push(BooleanClause::new(Occur::Must, filter));
        }
        if filters.created_from.is_some() || filters.created_to.is_some() {
            clauses.push(BooleanClause::new(
                Occur::Must,
                Query::Range {
                    field_id: self.fields.created_at,
                    lower: filters.created_from.map_or(Bound::Unbounded, |value| {
                        Bound::Included(QueryValue::I64(value))
                    }),
                    upper: filters.created_to.map_or(Bound::Unbounded, |value| {
                        Bound::Included(QueryValue::I64(value))
                    }),
                },
            ));
        }
        let source = match &filters.source_filter {
            CassSourceFilter::All => None,
            CassSourceFilter::Local => Some((self.fields.origin_kind, "local")),
            CassSourceFilter::Remote => Some((self.fields.origin_kind, "ssh")),
            CassSourceFilter::SourceId(source_id) => {
                Some((self.fields.source_id, source_id.as_str()))
            }
        };
        if let Some((field_id, value)) = source {
            clauses.push(BooleanClause::new(
                Occur::Must,
                cass_exact_term(field_id, value.to_owned()),
            ));
        }
        if clauses.len() == 1 {
            clauses.pop().map_or(Query::All, |clause| clause.query)
        } else {
            Query::Boolean {
                clauses,
                operator: None,
            }
        }
    }

    fn lower_term(&self, raw: &str) -> Query {
        let parts = cass_sanitize_query(raw)
            .split_whitespace()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        self.lower_compound(&parts)
    }

    fn lower_compound(&self, parts: &[String]) -> Query {
        let queries = parts
            .iter()
            .filter_map(|part| self.lower_term_part(part))
            .collect::<Vec<_>>();
        cass_required_query(queries)
    }

    fn lower_term_part(&self, raw: &str) -> Option<Query> {
        let pattern = CassWildcardPattern::parse(raw);
        match pattern {
            CassWildcardPattern::Exact(term) | CassWildcardPattern::Prefix(term) => {
                if term.is_empty() {
                    return None;
                }
                if term.chars().any(is_cass_cjk) {
                    let terms = cass_cjk_terms(&term);
                    return Some(cass_required_query(
                        terms
                            .into_iter()
                            .map(|text| Query::Term {
                                fields: self.fields.searchable(),
                                text,
                            })
                            .collect(),
                    ));
                }
                Some(Query::Term {
                    fields: self.fields.searchable(),
                    text: term,
                })
            }
            CassWildcardPattern::Suffix(_)
            | CassWildcardPattern::Substring(_)
            | CassWildcardPattern::Complex(_) => Some(Query::Glob {
                field_ids: self.fields.regex_fields(),
                pattern: raw.to_lowercase(),
            }),
        }
    }

    fn lower_phrase(&self, raw: &str) -> Query {
        let terms = cass_sanitize_query(raw)
            .split_whitespace()
            .map(|term| term.trim_matches('*').to_lowercase())
            .filter(|term| !term.is_empty())
            .collect::<Vec<_>>();
        if terms.len() <= 1 || terms.iter().any(|term| term.chars().any(is_cass_cjk)) {
            return self.lower_compound(&terms);
        }
        Query::Phrase {
            fields: vec![
                QueryField::new(self.fields.title, 1.0),
                QueryField::new(self.fields.content, 1.0),
            ],
            terms: terms
                .into_iter()
                .zip(0_u32..)
                .map(|(text, position)| PositionedTerm::new(position, text))
                .collect(),
            slop: 0,
            prefix: false,
        }
    }
}

fn required_cass_field<F>(
    schema: SchemaDescriptor,
    name: &'static str,
    accepts: F,
) -> Result<FieldDescriptor, CassQueryParserConfigError>
where
    F: FnOnce(FieldKind) -> bool,
{
    let Some(field) = schema.fields.iter().find(|field| field.name == name) else {
        return Err(CassQueryParserConfigError::MissingField {
            schema: schema.name,
            field: name,
        });
    };
    if !accepts(field.kind) {
        return Err(CassQueryParserConfigError::InvalidField {
            schema: schema.name,
            field: name,
        });
    }
    Ok(*field)
}

fn cass_string_filter(field_id: u16, values: &[String]) -> Option<Query> {
    let clauses = values
        .iter()
        .map(|value| BooleanClause::new(Occur::Should, cass_exact_term(field_id, value.to_owned())))
        .collect::<Vec<_>>();
    (!clauses.is_empty()).then_some(Query::Boolean {
        clauses,
        operator: None,
    })
}

fn cass_exact_term(field_id: u16, text: String) -> Query {
    Query::Term {
        fields: vec![QueryField::new(field_id, 1.0)],
        text,
    }
}

fn cass_required_query(mut queries: Vec<Query>) -> Query {
    queries.retain(|query| !query.is_empty());
    match queries.len() {
        0 => Query::Empty,
        1 => queries.pop().unwrap_or(Query::Empty),
        _ => Query::Boolean {
            clauses: queries
                .into_iter()
                .map(|query| BooleanClause::new(Occur::Must, query))
                .collect(),
            operator: Some(BooleanOperator::And),
        },
    }
}

fn cass_cjk_terms(term: &str) -> Vec<String> {
    let chars = term
        .chars()
        .filter(|ch| is_cass_cjk(*ch))
        .collect::<Vec<_>>();
    if chars.len() <= 1 {
        return chars.into_iter().map(|ch| ch.to_string()).collect();
    }
    chars
        .windows(2)
        .map(|pair| pair.iter().collect::<String>())
        .collect()
}

fn cass_complement(query: Query) -> Query {
    Query::Boolean {
        clauses: vec![
            BooleanClause::new(Occur::Must, Query::All),
            BooleanClause::new(Occur::MustNot, query),
        ],
        operator: None,
    }
}

#[cfg(any(test, feature = "conformance-internals"))]
fn cass_observation_preflight(
    raw_query: &str,
    filters: &CassQueryFilters,
) -> Result<(), CassQueryObservationError> {
    cass_observation_bound(
        "raw_query_bytes",
        raw_query.len(),
        MAX_CASS_OBSERVATION_RAW_QUERY_BYTES,
    )?;

    let source_value = match &filters.source_filter {
        CassSourceFilter::SourceId(value) => Some(value.as_str()),
        CassSourceFilter::All | CassSourceFilter::Local | CassSourceFilter::Remote => None,
    };
    let filter_count = filters
        .agents
        .len()
        .checked_add(filters.workspaces.len())
        .and_then(|count| count.checked_add(usize::from(source_value.is_some())))
        .unwrap_or(usize::MAX);
    cass_observation_bound(
        "filter_value_count",
        filter_count,
        MAX_CASS_OBSERVATION_FILTER_VALUES,
    )?;

    let mut aggregate_bytes = 0_usize;
    for value in filters
        .agents
        .iter()
        .chain(&filters.workspaces)
        .map(String::as_str)
        .chain(source_value)
    {
        cass_observation_bound(
            "filter_value_bytes",
            value.len(),
            MAX_CASS_OBSERVATION_FILTER_VALUE_BYTES,
        )?;
        aggregate_bytes = aggregate_bytes.saturating_add(value.len());
        cass_observation_bound(
            "aggregate_filter_bytes",
            aggregate_bytes,
            MAX_CASS_OBSERVATION_FILTER_BYTES,
        )?;
    }
    Ok(())
}

#[cfg(any(test, feature = "conformance-internals"))]
fn cass_observation_bound(
    field: &'static str,
    actual: usize,
    limit: usize,
) -> Result<(), CassQueryObservationError> {
    if actual > limit {
        return Err(CassQueryObservationError::BoundExceeded {
            field,
            actual,
            limit,
        });
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CassLexToken {
    Term { text: String, offset: usize },
    Phrase { text: String, offset: usize },
    And { offset: usize },
    Or { offset: usize },
    Not { offset: usize },
}

impl CassLexToken {
    #[cfg(any(test, feature = "conformance-internals"))]
    fn observed(&self) -> CassQueryToken {
        match self {
            Self::Term { text, offset } => CassQueryToken::Term {
                text: text.clone(),
                byte_offset: *offset,
            },
            Self::Phrase { text, offset } => CassQueryToken::Phrase {
                text: text.clone(),
                byte_offset: *offset,
            },
            Self::And { offset } => CassQueryToken::And {
                byte_offset: *offset,
            },
            Self::Or { offset } => CassQueryToken::Or {
                byte_offset: *offset,
            },
            Self::Not { offset } => CassQueryToken::Not {
                byte_offset: *offset,
            },
        }
    }
}

fn cass_lex(query: &str, diagnostics: &mut Vec<QueryDiagnostic>) -> Vec<CassLexToken> {
    let mut tokens = Vec::new();
    let mut chars = query.char_indices().peekable();
    let mut word = String::new();
    let mut word_offset = 0_usize;
    while let Some((offset, ch)) = chars.next() {
        match ch {
            '"' => {
                cass_flush_word(&mut tokens, &mut word, word_offset);
                let mut phrase = String::new();
                let mut closed = false;
                for (_, next) in chars.by_ref() {
                    if next == '"' {
                        closed = true;
                        break;
                    }
                    phrase.push(next);
                }
                if !phrase.is_empty() {
                    tokens.push(CassLexToken::Phrase {
                        text: phrase,
                        offset,
                    });
                }
                if !closed {
                    emit_diagnostic(
                        diagnostics,
                        QueryDiagnostic {
                            kind: QueryDiagnosticKind::SyntaxRecovery,
                            message: "CASS syntax recovery: unterminated phrase".to_owned(),
                            byte_offset: Some(offset),
                            fragment: None,
                        },
                    );
                }
            }
            '&' if chars.peek().is_some_and(|(_, next)| *next == '&') => {
                chars.next();
                cass_flush_word(&mut tokens, &mut word, word_offset);
                tokens.push(CassLexToken::And { offset });
            }
            '|' if chars.peek().is_some_and(|(_, next)| *next == '|') => {
                chars.next();
                cass_flush_word(&mut tokens, &mut word, word_offset);
                tokens.push(CassLexToken::Or { offset });
            }
            '-' if word.is_empty() => tokens.push(CassLexToken::Not { offset }),
            ' ' | '\t' | '\n' => {
                cass_flush_word(&mut tokens, &mut word, word_offset);
            }
            _ => {
                if word.is_empty() {
                    word_offset = offset;
                }
                word.push(ch);
            }
        }
    }
    cass_flush_word(&mut tokens, &mut word, word_offset);
    tokens
}

fn cass_flush_word(tokens: &mut Vec<CassLexToken>, word: &mut String, offset: usize) {
    if word.is_empty() {
        return;
    }
    let text = std::mem::take(word);
    let token = if text.eq_ignore_ascii_case("AND") {
        CassLexToken::And { offset }
    } else if text.eq_ignore_ascii_case("OR") {
        CassLexToken::Or { offset }
    } else if text.eq_ignore_ascii_case("NOT") {
        CassLexToken::Not { offset }
    } else {
        CassLexToken::Term { text, offset }
    };
    tokens.push(token);
}

#[derive(Debug)]
struct CassNode {
    query: Query,
    negative: bool,
}

struct CassGrammar {
    parser: CassQueryParser,
    tokens: Vec<CassLexToken>,
    diagnostics: Vec<QueryDiagnostic>,
}

impl CassGrammar {
    fn parse(&mut self) -> Option<CassNode> {
        let mut clauses = Vec::new();
        let mut pending_or_group = Vec::new();
        let mut next_occur = Occur::Must;
        let mut in_or_sequence = false;
        let mut just_saw_or = false;
        let mut saw_operand = false;
        let mut last_binary_offset = None;
        let mut dangling_not_offset = None;

        for token in std::mem::take(&mut self.tokens) {
            match token {
                CassLexToken::And { offset } => {
                    if !saw_operand || last_binary_offset.is_some() {
                        self.syntax_diagnostic(
                            "AND without an adjacent operand was recovered",
                            offset,
                        );
                    }
                    if let Some(not_offset) = dangling_not_offset.take() {
                        self.syntax_diagnostic("NOT has no operand before AND", not_offset);
                    }
                    cass_flush_native_or_group(&mut pending_or_group, &mut clauses);
                    in_or_sequence = false;
                    just_saw_or = false;
                    next_occur = Occur::Must;
                    last_binary_offset = Some(offset);
                }
                CassLexToken::Or { offset } => {
                    if !saw_operand || last_binary_offset.is_some() {
                        self.syntax_diagnostic(
                            "OR without an adjacent operand was recovered",
                            offset,
                        );
                    }
                    in_or_sequence = true;
                    just_saw_or = true;
                    last_binary_offset = Some(offset);
                }
                CassLexToken::Not { offset } => {
                    if !just_saw_or {
                        cass_flush_native_or_group(&mut pending_or_group, &mut clauses);
                        in_or_sequence = false;
                        just_saw_or = false;
                    }
                    next_occur = Occur::MustNot;
                    dangling_not_offset.get_or_insert(offset);
                    last_binary_offset = None;
                }
                CassLexToken::Term { text, offset } => {
                    let query = self.parser.lower_term(&text);
                    if query.is_empty() {
                        self.syntax_diagnostic("empty term operand was skipped", offset);
                        continue;
                    }
                    cass_apply_native_query(
                        query,
                        next_occur,
                        &mut in_or_sequence,
                        &mut just_saw_or,
                        &mut pending_or_group,
                        &mut clauses,
                    );
                    next_occur = Occur::Must;
                    saw_operand = true;
                    last_binary_offset = None;
                    dangling_not_offset = None;
                }
                CassLexToken::Phrase { text, offset } => {
                    let query = self.parser.lower_phrase(&text);
                    if query.is_empty() {
                        self.syntax_diagnostic("empty phrase operand was skipped", offset);
                        continue;
                    }
                    cass_apply_native_query(
                        query,
                        next_occur,
                        &mut in_or_sequence,
                        &mut just_saw_or,
                        &mut pending_or_group,
                        &mut clauses,
                    );
                    next_occur = Occur::Must;
                    saw_operand = true;
                    last_binary_offset = None;
                    dangling_not_offset = None;
                }
            }
        }

        cass_flush_native_or_group(&mut pending_or_group, &mut clauses);
        if let Some(offset) = dangling_not_offset {
            self.syntax_diagnostic("dangling NOT has no operand", offset);
        }
        if let Some(offset) = last_binary_offset {
            self.syntax_diagnostic("dangling binary operator has no operand", offset);
        }
        cass_finish_native_clauses(clauses)
    }

    fn syntax_diagnostic(&mut self, message: &str, offset: usize) {
        emit_diagnostic(
            &mut self.diagnostics,
            QueryDiagnostic {
                kind: QueryDiagnosticKind::SyntaxRecovery,
                message: format!("CASS syntax recovery: {message}"),
                byte_offset: Some(offset),
                fragment: None,
            },
        );
    }
}

fn cass_flush_native_or_group(pending_or_group: &mut Vec<Query>, clauses: &mut Vec<BooleanClause>) {
    if pending_or_group.is_empty() {
        return;
    }
    let query = Query::Boolean {
        clauses: std::mem::take(pending_or_group)
            .into_iter()
            .map(|query| BooleanClause::new(Occur::Should, query))
            .collect(),
        operator: Some(BooleanOperator::Or),
    };
    clauses.push(BooleanClause::new(Occur::Must, query));
}

fn cass_apply_native_query(
    query: Query,
    next_occur: Occur,
    in_or_sequence: &mut bool,
    just_saw_or: &mut bool,
    pending_or_group: &mut Vec<Query>,
    clauses: &mut Vec<BooleanClause>,
) {
    if *in_or_sequence && *just_saw_or {
        if pending_or_group.is_empty()
            && clauses
                .last()
                .is_some_and(|clause| matches!(clause.occur, Occur::Must | Occur::MustNot))
            && let Some(clause) = clauses.pop()
        {
            pending_or_group.push(if clause.occur == Occur::MustNot {
                cass_complement(clause.query)
            } else {
                clause.query
            });
        }
        pending_or_group.push(if next_occur == Occur::MustNot {
            cass_complement(query)
        } else {
            query
        });
    } else {
        cass_flush_native_or_group(pending_or_group, clauses);
        *in_or_sequence = false;
        clauses.push(BooleanClause::new(next_occur, query));
    }
    *just_saw_or = false;
}

fn cass_finish_native_clauses(mut clauses: Vec<BooleanClause>) -> Option<CassNode> {
    if clauses.len() == 1 {
        let clause = clauses.pop()?;
        return Some(CassNode {
            query: clause.query,
            negative: clause.occur == Occur::MustNot,
        });
    }
    if clauses.is_empty() {
        return None;
    }
    if clauses.iter().all(|clause| clause.occur == Occur::MustNot) {
        clauses.insert(0, BooleanClause::new(Occur::Must, Query::All));
    }
    Some(CassNode {
        query: Query::Boolean {
            clauses,
            operator: Some(BooleanOperator::And),
        },
        negative: false,
    })
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "tantivy-oracle")]
    use std::collections::BTreeSet;
    use std::io::{self, Write};
    use std::sync::{Arc, Mutex};

    #[cfg(feature = "tantivy-oracle")]
    use frankensearch_core::traits::LexicalWrite;
    #[cfg(feature = "tantivy-oracle")]
    use frankensearch_core::types::IndexableDocument;
    #[cfg(feature = "tantivy-oracle")]
    use frankensearch_lexical::TantivyIndex;
    use serde_json::{Value, json};

    use super::*;
    use crate::schema::{
        Analyzer, CASS_SEMANTIC_SCHEMA, DEFAULT_SCHEMA, FSFS_CHUNK_SCHEMA, FieldDescriptor,
        FieldKind, SchemaDescriptor,
    };
    #[cfg(feature = "tantivy-oracle")]
    use crate::scribe::{CassAnalyzer, cass_generate_edge_ngrams};

    const TYPED_FIELDS: [FieldDescriptor; 7] = [
        FieldDescriptor {
            id: 0,
            name: "id",
            kind: FieldKind::Keyword,
            stored: true,
        },
        FieldDescriptor {
            id: 1,
            name: "content",
            kind: FieldKind::Text {
                analyzer: Analyzer::FrankensearchDefault,
                positions: true,
            },
            stored: true,
        },
        FieldDescriptor {
            id: 2,
            name: "title",
            kind: FieldKind::Text {
                analyzer: Analyzer::FrankensearchDefault,
                positions: true,
            },
            stored: true,
        },
        FieldDescriptor {
            id: 3,
            name: "signed",
            kind: FieldKind::I64 {
                indexed: true,
                fast: false,
            },
            stored: false,
        },
        FieldDescriptor {
            id: 4,
            name: "unsigned",
            kind: FieldKind::U64 {
                indexed: false,
                fast: true,
            },
            stored: false,
        },
        FieldDescriptor {
            id: 5,
            name: "stored",
            kind: FieldKind::StoredOnly,
            stored: true,
        },
        FieldDescriptor {
            id: 6,
            name: "summary",
            kind: FieldKind::Text {
                analyzer: Analyzer::FrankensearchDefault,
                positions: false,
            },
            stored: false,
        },
    ];

    const TYPED_SCHEMA: SchemaDescriptor = SchemaDescriptor {
        name: "query-parser-typed-test",
        fields: &TYPED_FIELDS,
    };

    fn parser() -> DefaultQueryParser {
        DefaultQueryParser::new(DEFAULT_SCHEMA).expect("default schema supports its parser")
    }

    #[derive(Debug, Clone, Copy)]
    struct EvalDoc {
        id: &'static str,
        content: &'static str,
        title: &'static str,
    }

    #[cfg(feature = "tantivy-oracle")]
    #[derive(Debug, Clone, Copy)]
    struct CassEvalDoc {
        msg_idx: u64,
        agent: &'static str,
        workspace: Option<&'static str>,
        created_at: Option<i64>,
        title: &'static str,
        content: &'static str,
        source_id: &'static str,
        origin_kind: &'static str,
    }

    fn analyzed_terms(text: &str) -> Vec<String> {
        let mut analyzer = FrankensearchTokenizer::default();
        let mut terms = Vec::new();
        let result = analyze_admitted(
            &mut analyzer,
            AnalyzerKind::FrankensearchDefault,
            text,
            &mut |token| terms.push(token.text.clone()),
        );
        assert!(result.is_ok(), "default analyzer is registered");
        terms
    }

    #[cfg(feature = "tantivy-oracle")]
    fn cass_analyzed_terms(text: &str, analyzer_kind: AnalyzerKind) -> Vec<PositionedTerm> {
        let mut analyzer = CassAnalyzer::default();
        let mut terms = Vec::new();
        let result = analyze_admitted(&mut analyzer, analyzer_kind, text, &mut |token| {
            terms.push(PositionedTerm::new(token.position, token.text.clone()));
        });
        assert!(result.is_ok(), "CASS analyzer is registered");
        terms
    }

    #[cfg(feature = "tantivy-oracle")]
    fn cass_field_text(doc: CassEvalDoc, field_id: u16) -> Option<&'static str> {
        match CASS_SEMANTIC_SCHEMA.fields.get(usize::from(field_id))?.name {
            "agent" => Some(doc.agent),
            "workspace" => doc.workspace,
            "title" => Some(doc.title),
            "content" => Some(doc.content),
            "source_id" => Some(doc.source_id),
            "origin_kind" => Some(doc.origin_kind),
            _ => None,
        }
    }

    #[cfg(feature = "tantivy-oracle")]
    fn cass_field_terms(doc: CassEvalDoc, field_id: u16) -> Vec<PositionedTerm> {
        let Some(field) = CASS_SEMANTIC_SCHEMA.fields.get(usize::from(field_id)) else {
            return Vec::new();
        };
        match field.name {
            "title" => cass_analyzed_terms(doc.title, AnalyzerKind::CassHyphenNormalize),
            "content" => cass_analyzed_terms(doc.content, AnalyzerKind::CassHyphenNormalize),
            "title_prefix" => cass_analyzed_terms(
                &cass_generate_edge_ngrams(doc.title),
                AnalyzerKind::CassPrefixNormalize,
            ),
            "content_prefix" => cass_analyzed_terms(
                &cass_generate_edge_ngrams(doc.content),
                AnalyzerKind::CassPrefixNormalize,
            ),
            _ => cass_field_text(doc, field_id)
                .map_or_else(Vec::new, |text| vec![PositionedTerm::new(0, text)]),
        }
    }

    #[cfg(feature = "tantivy-oracle")]
    fn cass_term_matches(doc: CassEvalDoc, field: QueryField, needle: &str) -> bool {
        cass_field_terms(doc, field.field_id)
            .iter()
            .any(|term| term.text == needle)
    }

    #[cfg(feature = "tantivy-oracle")]
    fn cass_phrase_matches(doc: CassEvalDoc, field: QueryField, phrase: &[PositionedTerm]) -> bool {
        let haystack = cass_field_terms(doc, field.field_id);
        let Some(first) = phrase.first() else {
            return false;
        };
        haystack.iter().any(|candidate| {
            candidate.text == first.text
                && phrase.iter().all(|needle| {
                    let position = candidate
                        .position
                        .saturating_add(needle.position.saturating_sub(first.position));
                    haystack
                        .iter()
                        .any(|term| term.position == position && term.text == needle.text)
                })
        })
    }

    #[cfg(feature = "tantivy-oracle")]
    fn cass_bound_matches(
        value: i64,
        lower: &Bound<QueryValue>,
        upper: &Bound<QueryValue>,
    ) -> bool {
        let lower_matches = match lower {
            Bound::Included(QueryValue::I64(lower)) => value >= *lower,
            Bound::Excluded(QueryValue::I64(lower)) => value > *lower,
            Bound::Unbounded => true,
            Bound::Included(_) | Bound::Excluded(_) => false,
        };
        let upper_matches = match upper {
            Bound::Included(QueryValue::I64(upper)) => value <= *upper,
            Bound::Excluded(QueryValue::I64(upper)) => value < *upper,
            Bound::Unbounded => true,
            Bound::Included(_) | Bound::Excluded(_) => false,
        };
        lower_matches && upper_matches
    }

    #[cfg(feature = "tantivy-oracle")]
    fn star_glob_matches(pattern: &str, text: &str) -> bool {
        let pattern = pattern.chars().collect::<Vec<_>>();
        let text = text.chars().collect::<Vec<_>>();
        let mut pattern_index = 0_usize;
        let mut text_index = 0_usize;
        let mut star_index = None;
        let mut star_match = 0_usize;
        while text_index < text.len() {
            if pattern.get(pattern_index) == text.get(text_index) {
                pattern_index += 1;
                text_index += 1;
            } else if pattern.get(pattern_index) == Some(&'*') {
                star_index = Some(pattern_index);
                pattern_index += 1;
                star_match = text_index;
            } else if let Some(star) = star_index {
                star_match += 1;
                text_index = star_match;
                pattern_index = star + 1;
            } else {
                return false;
            }
        }
        pattern[pattern_index..].iter().all(|ch| *ch == '*')
    }

    #[cfg(feature = "tantivy-oracle")]
    fn cass_ast_matches(query: &Query, doc: CassEvalDoc) -> bool {
        match query {
            Query::Empty => false,
            Query::All => true,
            Query::Term { fields, text } => fields
                .iter()
                .copied()
                .any(|field| cass_term_matches(doc, field, text)),
            Query::Phrase { fields, terms, .. } => fields
                .iter()
                .copied()
                .any(|field| cass_phrase_matches(doc, field, terms)),
            Query::Boolean { clauses, .. } => {
                let mut has_must = false;
                let mut should_count = 0_usize;
                let mut should_match = false;
                for clause in clauses {
                    let child_matches = cass_ast_matches(&clause.query, doc);
                    match clause.occur {
                        Occur::Must => {
                            has_must = true;
                            if !child_matches {
                                return false;
                            }
                        }
                        Occur::Should => {
                            should_count += 1;
                            should_match |= child_matches;
                        }
                        Occur::MustNot if child_matches => return false,
                        Occur::MustNot => {}
                    }
                }
                has_must || (should_count != 0 && should_match)
            }
            Query::Range {
                field_id,
                lower,
                upper,
            } if CASS_SEMANTIC_SCHEMA.fields[usize::from(*field_id)].name == "created_at" => doc
                .created_at
                .is_some_and(|value| cass_bound_matches(value, lower, upper)),
            Query::Range { .. } => false,
            Query::Set { field_id, values } => {
                cass_field_text(doc, *field_id).is_some_and(|actual| {
                    values
                        .iter()
                        .any(|value| value == &QueryValue::Str(actual.to_owned()))
                })
            }
            Query::Glob { field_ids, pattern } => field_ids.iter().copied().any(|field_id| {
                cass_field_terms(doc, field_id)
                    .iter()
                    .any(|term| star_glob_matches(pattern, &term.text))
            }),
            Query::Boost { query, .. } => cass_ast_matches(query, doc),
        }
    }

    fn field_text(doc: EvalDoc, field_id: u16) -> Option<&'static str> {
        match DEFAULT_SCHEMA.fields.get(usize::from(field_id))?.name {
            "id" => Some(doc.id),
            "content" => Some(doc.content),
            "title" => Some(doc.title),
            _ => None,
        }
    }

    fn field_term_matches(doc: EvalDoc, field: QueryField, needle: &str) -> bool {
        let Some(text) = field_text(doc, field.field_id) else {
            return false;
        };
        if DEFAULT_SCHEMA.fields[usize::from(field.field_id)].kind == FieldKind::Keyword {
            text == needle
        } else {
            analyzed_terms(text).iter().any(|term| term == needle)
        }
    }

    fn field_phrase_matches(
        doc: EvalDoc,
        field: QueryField,
        phrase: &[PositionedTerm],
        prefix: bool,
    ) -> bool {
        let Some(text) = field_text(doc, field.field_id) else {
            return false;
        };
        let haystack = analyzed_terms(text);
        let Some(first_position) = phrase.first().map(|term| term.position) else {
            return false;
        };
        haystack.iter().enumerate().any(|(start, _)| {
            phrase.iter().enumerate().all(|(index, term)| {
                let relative = term.position.saturating_sub(first_position);
                let Ok(relative) = usize::try_from(relative) else {
                    return false;
                };
                let Some(candidate) = haystack.get(start.saturating_add(relative)) else {
                    return false;
                };
                if prefix && index + 1 == phrase.len() {
                    candidate.starts_with(&term.text)
                } else {
                    candidate == &term.text
                }
            })
        })
    }

    fn string_bound_matches(
        value: &str,
        lower: &Bound<QueryValue>,
        upper: &Bound<QueryValue>,
    ) -> bool {
        let lower_matches = match lower {
            Bound::Included(QueryValue::Str(lower)) => value >= lower.as_str(),
            Bound::Excluded(QueryValue::Str(lower)) => value > lower.as_str(),
            Bound::Unbounded => true,
            Bound::Included(_) | Bound::Excluded(_) => false,
        };
        let upper_matches = match upper {
            Bound::Included(QueryValue::Str(upper)) => value <= upper.as_str(),
            Bound::Excluded(QueryValue::Str(upper)) => value < upper.as_str(),
            Bound::Unbounded => true,
            Bound::Included(_) | Bound::Excluded(_) => false,
        };
        lower_matches && upper_matches
    }

    fn ast_matches(query: &Query, doc: EvalDoc) -> bool {
        match query {
            Query::Empty => false,
            Query::All => true,
            Query::Term { fields, text } => fields
                .iter()
                .copied()
                .any(|field| field_term_matches(doc, field, text)),
            Query::Phrase {
                fields,
                terms,
                prefix,
                ..
            } => fields
                .iter()
                .copied()
                .any(|field| field_phrase_matches(doc, field, terms, *prefix)),
            Query::Boolean { clauses, .. } => {
                let mut has_must = false;
                let mut should_count = 0_usize;
                let mut should_match = false;
                for clause in clauses {
                    let child_matches = ast_matches(&clause.query, doc);
                    match clause.occur {
                        Occur::Must => {
                            has_must = true;
                            if !child_matches {
                                return false;
                            }
                        }
                        Occur::Should => {
                            should_count += 1;
                            should_match |= child_matches;
                        }
                        Occur::MustNot if child_matches => return false,
                        Occur::MustNot => {}
                    }
                }
                has_must || (should_count != 0 && should_match)
            }
            Query::Range {
                field_id,
                lower,
                upper,
            } => field_text(doc, *field_id)
                .is_some_and(|value| string_bound_matches(value, lower, upper)),
            Query::Set { field_id, values } => field_text(doc, *field_id).is_some_and(|value| {
                values
                    .iter()
                    .any(|candidate| candidate == &QueryValue::Str(value.to_owned()))
            }),
            Query::Glob { field_ids, pattern } => field_ids
                .iter()
                .copied()
                .any(|field_id| field_text(doc, field_id).is_some_and(|value| value == pattern)),
            Query::Boost { query, .. } => ast_matches(query, doc),
        }
    }

    fn field_name(schema: SchemaDescriptor, field_id: u16) -> &'static str {
        schema.fields[usize::from(field_id)].name
    }

    fn value_json(value: &QueryValue) -> Value {
        match value {
            QueryValue::I64(value) => json!(value),
            QueryValue::U64(value) => json!(value),
            QueryValue::Str(value) => json!(value),
        }
    }

    fn value_tag(value: &QueryValue) -> &'static str {
        match value {
            QueryValue::I64(_) => "I64",
            QueryValue::U64(_) => "U64",
            QueryValue::Str(_) => "Str",
        }
    }

    fn bound_json(bound: &Bound<QueryValue>) -> Value {
        match bound {
            Bound::Included(value) => {
                json!({ "bound": "Included", "value": value_json(value) })
            }
            Bound::Excluded(value) => {
                json!({ "bound": "Excluded", "value": value_json(value) })
            }
            Bound::Unbounded => json!({ "bound": "Unbounded" }),
        }
    }

    fn typed_tag<'a>(prefix: &'a str, values: impl IntoIterator<Item = &'a QueryValue>) -> String {
        let suffix = values.into_iter().next().map_or("Str", value_tag);
        format!("{prefix}{suffix}")
    }

    fn fixture_json(query: &Query, occur: Option<Occur>) -> Value {
        fixture_json_for_schema(DEFAULT_SCHEMA, query, occur)
    }

    fn fixture_json_for_schema(
        schema: SchemaDescriptor,
        query: &Query,
        occur: Option<Occur>,
    ) -> Value {
        match query {
            Query::Empty => json!({ "type": "Empty" }),
            Query::All => json!({ "type": "All" }),
            Query::Term { fields, text } => {
                let fields = fields
                    .iter()
                    .map(|field| {
                        json!({
                            "name": field_name(schema, field.field_id),
                            "boost": field.boost,
                        })
                    })
                    .collect::<Vec<_>>();
                let mut value = json!({ "type": "Term", "text": text, "fields": fields });
                if occur == Some(Occur::MustNot) {
                    value["score"] = json!(0.0);
                }
                value
            }
            Query::Phrase {
                fields,
                terms,
                slop,
                prefix,
            } => {
                let fields = fields
                    .iter()
                    .map(|field| {
                        json!({
                            "name": field_name(schema, field.field_id),
                            "boost": field.boost,
                        })
                    })
                    .collect::<Vec<_>>();
                let terms = terms
                    .iter()
                    .map(|term| term.text.as_str())
                    .collect::<Vec<_>>();
                let mut value = json!({
                    "type": if *prefix { "PhrasePrefix" } else { "Phrase" },
                    "terms": terms,
                    "slop": slop,
                    "fields": fields,
                });
                if occur == Some(Occur::MustNot) {
                    value["score"] = json!(0.0);
                }
                value
            }
            Query::Boolean { clauses, operator } => {
                let children = clauses
                    .iter()
                    .map(|clause| {
                        json!({
                            "occur": match clause.occur {
                                Occur::Must => "Must",
                                Occur::Should => "Should",
                                Occur::MustNot => "MustNot",
                            },
                            "query": fixture_json_for_schema(
                                schema,
                                &clause.query,
                                Some(clause.occur),
                            ),
                        })
                    })
                    .collect::<Vec<_>>();
                let mut value = json!({ "type": "Boolean", "children": children });
                if let Some(operator) = operator {
                    value["operator"] = json!(match operator {
                        BooleanOperator::And => "AND",
                        BooleanOperator::Or => "OR",
                    });
                }
                value
            }
            Query::Range {
                field_id,
                lower,
                upper,
            } => {
                let values = [lower, upper].into_iter().filter_map(|bound| match bound {
                    Bound::Included(value) | Bound::Excluded(value) => Some(value),
                    Bound::Unbounded => None,
                });
                let mut value = json!({
                    "type": typed_tag("Range", values),
                    "field": field_name(schema, *field_id),
                    "lower": bound_json(lower),
                    "upper": bound_json(upper),
                });
                if schema == CASS_SEMANTIC_SCHEMA {
                    value["matched_score"] = json!(1.0);
                }
                value
            }
            Query::Set { field_id, values } => json!({
                "type": typed_tag("Set", values),
                "field": field_name(schema, *field_id),
                "values": values.iter().map(value_json).collect::<Vec<_>>(),
            }),
            Query::Glob { field_ids, pattern } => json!({
                "type": "Glob",
                "pattern": pattern,
                "fields": field_ids
                    .iter()
                    .map(|field_id| field_name(schema, *field_id))
                    .collect::<Vec<_>>(),
            }),
            Query::Boost { query, factor } => json!({
                "type": "Boost",
                "factor": factor,
                "query": fixture_json_for_schema(schema, query, occur),
            }),
        }
    }

    #[test]
    fn default_parser_matches_pinned_tantivy_fixture_trees() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/quill_language_contract.json"
        ))
        .expect("language fixture parses");
        let cases = fixture["parse_tree_cases"]
            .as_array()
            .expect("parse tree case array");
        let expected = cases
            .iter()
            .filter(|case| case["parser"] == "default_lenient")
            .count();
        let mut executed = 0;
        for case in cases {
            if case["parser"] != "default_lenient" {
                continue;
            }
            let id = case["id"].as_str().expect("fixture id");
            let input = case["input"].as_str().expect("fixture input");
            let parsed = parser().parse(input);
            assert_eq!(
                fixture_json(&parsed.query, None),
                case["expected_ast"],
                "fixture {id}"
            );
            if let Some(expected) = case.get("expected_diagnostic").and_then(Value::as_str) {
                assert!(
                    parsed
                        .diagnostics
                        .iter()
                        .any(|diagnostic| diagnostic.message.contains(expected)),
                    "fixture {id} expected diagnostic {expected:?}, got {:?}",
                    parsed.diagnostics
                );
            } else {
                assert!(
                    parsed.diagnostics.is_empty(),
                    "fixture {id} unexpectedly diagnosed {:?}",
                    parsed.diagnostics
                );
            }
            executed += 1;
        }
        assert_eq!(executed, expected, "every pinned default parser case ran");
    }

    fn cass_parser() -> CassQueryParser {
        CassQueryParser::new(CASS_SEMANTIC_SCHEMA).expect("CASS schema supports its parser")
    }

    #[test]
    fn cass_parser_observation_uses_the_real_lexer_and_admitted_prefix() {
        let parser = cass_parser();
        let raw = "c++ \"exact phrase\" OR -deprecated";
        let filters = CassQueryFilters::default();
        let observed = parser
            .parse_observed(raw, &filters)
            .expect("bounded observation");

        assert_eq!(observed.parsed, parser.parse(raw, &filters));
        assert_eq!(observed.admitted_query, raw);
        assert_eq!(
            observed.sanitized_query,
            "c   \"exact phrase\" OR -deprecated"
        );
        assert_eq!(
            observed.tokens,
            vec![
                CassQueryToken::Term {
                    text: "c++".to_owned(),
                    byte_offset: 0,
                },
                CassQueryToken::Phrase {
                    text: "exact phrase".to_owned(),
                    byte_offset: 4,
                },
                CassQueryToken::Or { byte_offset: 19 },
                CassQueryToken::Not { byte_offset: 22 },
                CassQueryToken::Term {
                    text: "deprecated".to_owned(),
                    byte_offset: 23,
                },
            ]
        );

        let mut oversized = "auth".to_owned();
        oversized.push_str(&" ".repeat(MAX_QUERY_LENGTH - oversized.chars().count()));
        oversized.push_str("private-tail");
        let truncated = parser
            .parse_observed(&oversized, &filters)
            .expect("bounded truncated observation");
        assert!(truncated.parsed.was_truncated);
        assert_eq!(truncated.admitted_query.chars().count(), MAX_QUERY_LENGTH);
        assert!(!truncated.sanitized_query.contains("private-tail"));
        assert!(truncated.tokens.iter().all(|token| {
            !matches!(
                token,
                CassQueryToken::Term { text, .. } if text.contains("private-tail")
            )
        }));
    }

    #[test]
    fn cass_parser_observation_uses_utf8_byte_offsets_within_the_admitted_prefix() {
        let parser = cass_parser();
        let filters = CassQueryFilters::default();
        let raw = "搜索 \"éclair\" OR 尾巴";
        let observed = parser
            .parse_observed(raw, &filters)
            .expect("multibyte observation is bounded");

        assert_eq!(observed.admitted_query, raw);
        assert_eq!(
            observed.tokens,
            vec![
                CassQueryToken::Term {
                    text: "搜索".to_owned(),
                    byte_offset: 0,
                },
                CassQueryToken::Phrase {
                    text: "éclair".to_owned(),
                    byte_offset: "搜索 ".len(),
                },
                CassQueryToken::Or {
                    byte_offset: "搜索 \"éclair\" ".len(),
                },
                CassQueryToken::Term {
                    text: "尾巴".to_owned(),
                    byte_offset: "搜索 \"éclair\" OR ".len(),
                },
            ]
        );
        assert!(observed.tokens.iter().all(|token| {
            let offset = match token {
                CassQueryToken::Term { byte_offset, .. }
                | CassQueryToken::Phrase { byte_offset, .. }
                | CassQueryToken::And { byte_offset }
                | CassQueryToken::Or { byte_offset }
                | CassQueryToken::Not { byte_offset } => *byte_offset,
            };
            offset < observed.admitted_query.len()
                && observed.admitted_query.is_char_boundary(offset)
        }));
    }

    #[test]
    fn cass_parser_observation_rejects_hostile_bounds_before_witness_allocation() {
        let parser = cass_parser();
        let oversized_raw = "x".repeat(MAX_CASS_OBSERVATION_RAW_QUERY_BYTES + 1);
        assert_eq!(
            parser.parse_observed(&oversized_raw, &CassQueryFilters::default()),
            Err(CassQueryObservationError::BoundExceeded {
                field: "raw_query_bytes",
                actual: MAX_CASS_OBSERVATION_RAW_QUERY_BYTES + 1,
                limit: MAX_CASS_OBSERVATION_RAW_QUERY_BYTES,
            })
        );

        let oversized_filter = "x".repeat(MAX_CASS_OBSERVATION_FILTER_VALUE_BYTES + 1);
        let filters = CassQueryFilters {
            agents: vec![oversized_filter],
            ..CassQueryFilters::default()
        };
        assert_eq!(
            parser.parse_observed("bounded", &filters),
            Err(CassQueryObservationError::BoundExceeded {
                field: "filter_value_bytes",
                actual: MAX_CASS_OBSERVATION_FILTER_VALUE_BYTES + 1,
                limit: MAX_CASS_OBSERVATION_FILTER_VALUE_BYTES,
            })
        );

        let aggregate_filters = CassQueryFilters {
            agents: vec![
                "x".repeat(MAX_CASS_OBSERVATION_FILTER_VALUE_BYTES);
                MAX_CASS_OBSERVATION_FILTER_BYTES / MAX_CASS_OBSERVATION_FILTER_VALUE_BYTES
                    + 1
            ],
            ..CassQueryFilters::default()
        };
        assert!(matches!(
            parser.parse_observed("bounded", &aggregate_filters),
            Err(CassQueryObservationError::BoundExceeded {
                field: "aggregate_filter_bytes",
                ..
            })
        ));

        let filters = CassQueryFilters {
            agents: vec!["x".to_owned(); MAX_CASS_OBSERVATION_FILTER_VALUES + 1],
            ..CassQueryFilters::default()
        };
        assert_eq!(
            parser.parse_observed("bounded", &filters),
            Err(CassQueryObservationError::BoundExceeded {
                field: "filter_value_count",
                actual: MAX_CASS_OBSERVATION_FILTER_VALUES + 1,
                limit: MAX_CASS_OBSERVATION_FILTER_VALUES,
            })
        );
    }

    fn cass_filters_from_fixture(case: &Value) -> CassQueryFilters {
        let Some(filters) = case.get("filters") else {
            return CassQueryFilters::default();
        };
        let strings = |name: &str| {
            filters[name]
                .as_array()
                .map(|values| {
                    values
                        .iter()
                        .map(|value| value.as_str().expect("string filter").to_owned())
                        .collect()
                })
                .unwrap_or_default()
        };
        let source_filter = match filters["source_filter"].as_str() {
            Some("local") => CassSourceFilter::Local,
            Some("remote") => CassSourceFilter::Remote,
            Some("source_id") => CassSourceFilter::SourceId(
                filters["source_id"]
                    .as_str()
                    .expect("source_id filter value")
                    .to_owned(),
            ),
            Some(unexpected) => panic!("unexpected CASS source fixture {unexpected:?}"),
            None => CassSourceFilter::All,
        };
        CassQueryFilters {
            agents: strings("agents"),
            workspaces: strings("workspaces"),
            created_from: filters["created_from"].as_i64(),
            created_to: filters["created_to"].as_i64(),
            source_filter,
        }
    }

    fn cass_wildcard_fixture(input: &str) -> Value {
        let pattern = CassWildcardPattern::parse(input);
        let searchable = ["title", "content", "title_prefix", "content_prefix"];
        let regex_fields = ["content", "title"];
        match pattern {
            CassWildcardPattern::Exact(_) => json!({
                "type": "Glob",
                "pattern": input,
                "class": "Exact",
                "strategy": "TermQuery",
                "fields": searchable,
            }),
            CassWildcardPattern::Prefix(term) => json!({
                "type": "Glob",
                "pattern": input,
                "normalized_term": term,
                "class": "Prefix",
                "strategy": "TermQuery",
                "fields": searchable,
            }),
            CassWildcardPattern::Suffix(term) => json!({
                "type": "Glob",
                "pattern": input,
                "class": "Suffix",
                "strategy": "RegexQuery",
                "regex": CassWildcardPattern::Suffix(term).to_regex(),
                "fields": regex_fields,
            }),
            CassWildcardPattern::Substring(term) => json!({
                "type": "Glob",
                "pattern": input,
                "class": "Substring",
                "strategy": "RegexQuery",
                "regex": CassWildcardPattern::Substring(term).to_regex(),
                "fields": regex_fields,
            }),
            CassWildcardPattern::Complex(term) => json!({
                "type": "Glob",
                "pattern": input,
                "class": "Complex",
                "strategy": "RegexQuery",
                "regex": CassWildcardPattern::Complex(term).to_regex(),
                "fields": regex_fields,
                "question_mark_operator": false,
            }),
        }
    }

    #[test]
    fn cass_parser_matches_pinned_tantivy_fixture_trees() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/quill_language_contract.json"
        ))
        .expect("language fixture parses");
        let cases = fixture["parse_tree_cases"]
            .as_array()
            .expect("parse tree case array");
        let expected = cases.iter().filter(|case| case["parser"] == "cass").count();
        let mut executed = 0_usize;
        for case in cases {
            if case["parser"] != "cass" {
                continue;
            }
            let id = case["id"].as_str().expect("fixture id");
            let input = case["input"].as_str().expect("fixture input");
            let filters = cass_filters_from_fixture(case);
            let parsed = cass_parser().parse(input, &filters);
            let expected_ast =
                case["expected_ast"]["ref"].as_str().map_or_else(
                    || &case["expected_ast"],
                    |reference| {
                        &cases
                            .iter()
                            .find(|candidate| candidate["id"] == reference)
                            .unwrap_or_else(|| {
                                panic!("fixture {id} references missing {reference}")
                            })["expected_ast"]
                    },
                );
            if case["query_class"] == "glob" {
                assert_eq!(&cass_wildcard_fixture(input), expected_ast, "fixture {id}");
                match (CassWildcardPattern::parse(input), &parsed.query) {
                    (
                        CassWildcardPattern::Exact(expected)
                        | CassWildcardPattern::Prefix(expected),
                        Query::Term { fields, text },
                    ) => {
                        assert_eq!(text, &expected, "native parser glob text for {id}");
                        assert_eq!(
                            fields,
                            &cass_parser().fields.searchable(),
                            "native parser glob fields for {id}"
                        );
                    }
                    (
                        CassWildcardPattern::Suffix(_)
                        | CassWildcardPattern::Substring(_)
                        | CassWildcardPattern::Complex(_),
                        Query::Glob { field_ids, pattern },
                    ) => {
                        assert_eq!(pattern, &input.to_lowercase(), "native glob for {id}");
                        assert_eq!(
                            field_ids,
                            &cass_parser().fields.regex_fields(),
                            "native parser regex fields for {id}"
                        );
                    }
                    (wildcard, query) => {
                        panic!("native parser glob mismatch for {id}: {wildcard:?} -> {query:?}")
                    }
                }
            } else {
                assert_eq!(
                    &fixture_json_for_schema(CASS_SEMANTIC_SCHEMA, &parsed.query, None),
                    expected_ast,
                    "fixture {id}"
                );
            }
            if let Some(expected_diagnostic) = case["expected_diagnostic"].as_str() {
                assert!(
                    parsed
                        .diagnostics
                        .iter()
                        .any(|diagnostic| diagnostic.message.contains(expected_diagnostic)),
                    "fixture {id} lacked diagnostic {expected_diagnostic:?}: {:?}",
                    parsed.diagnostics,
                );
            } else {
                assert!(
                    parsed.diagnostics.is_empty(),
                    "fixture {id} unexpectedly diagnosed {:?}",
                    parsed.diagnostics
                );
            }
            executed += 1;
        }
        assert_eq!(executed, expected, "every pinned CASS parser case ran");
        assert!(executed >= 39, "the full harvested CASS contract slice ran");
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn cass_helpers_match_the_shipping_adapter() {
        use frankensearch_lexical::{
            CassQueryToken as OracleToken, CassWildcardPattern as OracleWildcard,
            cass_parse_boolean_query as oracle_parse, cass_sanitize_query as oracle_sanitize,
        };

        for input in [
            "",
            "auth token",
            "auth AND token",
            "auth&&token OR cache",
            "NOT -deprecated",
            "\"exact phrase\"",
            "c++ hello_world bd-q3fy",
            "検索 abc搜索def",
            "auth\rOR\rcache",
            "NOT AND cache",
            "auth OR NOT AND deprecated",
        ] {
            assert_eq!(
                cass_sanitize_query(input),
                oracle_sanitize(input),
                "{input:?}"
            );
            let native = cass_lex(input, &mut Vec::new());
            let oracle = oracle_parse(input);
            let normalized_native = native
                .iter()
                .map(|token| match token {
                    CassLexToken::Term { text, .. } => ("term", text.as_str()),
                    CassLexToken::Phrase { text, .. } => ("phrase", text.as_str()),
                    CassLexToken::And { .. } => ("and", ""),
                    CassLexToken::Or { .. } => ("or", ""),
                    CassLexToken::Not { .. } => ("not", ""),
                })
                .collect::<Vec<_>>();
            let normalized_oracle = oracle
                .iter()
                .map(|token| match token {
                    OracleToken::Term(text) => ("term", text.as_str()),
                    OracleToken::Phrase(text) => ("phrase", text.as_str()),
                    OracleToken::And => ("and", ""),
                    OracleToken::Or => ("or", ""),
                    OracleToken::Not => ("not", ""),
                })
                .collect::<Vec<_>>();
            assert_eq!(normalized_native, normalized_oracle, "{input:?}");
        }

        for input in ["exact", "prefix*", "*suffix", "*middle*", "a*b*c", "f.o*"] {
            let native = CassWildcardPattern::parse(input);
            let oracle = OracleWildcard::parse(input);
            assert_eq!(format!("{native:?}"), format!("{oracle:?}"), "{input:?}");
            assert_eq!(native.to_regex(), oracle.to_regex(), "{input:?}");
        }
    }

    #[test]
    fn cass_parser_validates_configuration_and_recovers_hostile_input() {
        assert!(matches!(
            CassQueryParser::new(DEFAULT_SCHEMA),
            Err(CassQueryParserConfigError::MissingField { field: "agent", .. })
        ));

        let parser = cass_parser();
        let mut oversized = "auth".to_owned();
        oversized.push_str(&" ".repeat(MAX_QUERY_LENGTH - oversized.chars().count()));
        oversized.push_str("cache");
        let truncated = parser.parse(&oversized, &CassQueryFilters::default());
        assert!(truncated.was_truncated);
        assert!(truncated.diagnostics.iter().any(|diagnostic| {
            diagnostic.kind == QueryDiagnosticKind::Truncated
                && diagnostic.message.contains("10000 Unicode scalar values")
        }));
        assert!(matches!(
            truncated.query,
            Query::Term { ref text, .. } if text == "auth"
        ));

        for raw in [
            "AND auth",
            "auth OR",
            "NOT",
            "NOT NOT",
            "auth NOT",
            "* auth",
            "auth AND OR cache",
            "auth OR NOT AND deprecated",
            "\"***\" auth",
            "\0\0\0",
        ] {
            let parsed = parser.parse(raw, &CassQueryFilters::default());
            assert!(
                !parsed.diagnostics.is_empty(),
                "hostile input {raw:?} must explain its recovery"
            );
        }

        let recovered = parser.parse("NOT AND cache", &CassQueryFilters::default());
        assert!(matches!(
            recovered.query,
            Query::Term { ref text, .. } if text == "cache"
        ));
        let recovered = parser.parse("auth OR NOT AND deprecated", &CassQueryFilters::default());
        assert!(matches!(
            recovered.query,
            Query::Boolean {
                operator: Some(BooleanOperator::And),
                ..
            }
        ));
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn cass_parser_result_sets_match_the_shipping_tantivy_builder() {
        use frankensearch_lexical::tantivy_crate::collector::DocSetCollector;
        use frankensearch_lexical::{
            CassDocumentRef, CassQueryFilters as OracleFilters, CassSourceFilter as OracleSource,
            CassTantivyIndex, TantivyDocument, Value as TantivyValue, cass_build_tantivy_query,
        };

        const DOCS: [CassEvalDoc; 8] = [
            CassEvalDoc {
                msg_idx: 0,
                agent: "claude",
                workspace: Some("/alpha"),
                created_at: Some(100),
                title: "Authentication Cache",
                content: "auth token cache active",
                source_id: "local-a",
                origin_kind: "local",
            },
            CassEvalDoc {
                msg_idx: 1,
                agent: "codex",
                workspace: Some("/alpha"),
                created_at: Some(200),
                title: "Legacy Authentication",
                content: "auth deprecated token",
                source_id: "remote-a",
                origin_kind: "ssh",
            },
            CassEvalDoc {
                msg_idx: 2,
                agent: "claude",
                workspace: Some("/beta"),
                created_at: Some(300),
                title: "Cache Search",
                content: "cache search engine",
                source_id: "archive-7",
                origin_kind: "local",
            },
            CassEvalDoc {
                msg_idx: 3,
                agent: "gemini",
                workspace: None,
                created_at: Some(400),
                title: "搜索引擎",
                content: "这是一个搜索引擎测试",
                source_id: "remote-b",
                origin_kind: "ssh",
            },
            CassEvalDoc {
                msg_idx: 4,
                agent: "codex",
                workspace: Some("/beta"),
                created_at: Some(500),
                title: "Error Handling",
                content: "error handling guide auth cache",
                source_id: "archive-7",
                origin_kind: "local",
            },
            CassEvalDoc {
                msg_idx: 5,
                agent: "claude",
                workspace: Some("/alpha"),
                created_at: Some(600),
                title: "Foobar Service",
                content: "foobarbaz middleware",
                source_id: "remote-c",
                origin_kind: "ssh",
            },
            CassEvalDoc {
                msg_idx: 6,
                agent: "codex",
                workspace: Some("/gamma"),
                created_at: None,
                title: "Unrelated",
                content: "nothing applicable",
                source_id: "local-b",
                origin_kind: "local",
            },
            CassEvalDoc {
                msg_idx: 7,
                agent: "claude",
                workspace: Some("/alpha"),
                created_at: Some(700),
                title: "Token Store",
                content: "token cache deprecated",
                source_id: "remote-d",
                origin_kind: "ssh",
            },
        ];

        let directory = tempfile::tempdir().expect("temporary CASS index directory");
        let mut index = CassTantivyIndex::open_or_create(directory.path()).expect("CASS index");
        let oracle_documents = DOCS
            .iter()
            .map(|doc| CassDocumentRef {
                agent: doc.agent,
                workspace: doc.workspace,
                workspace_original: doc.workspace,
                source_path: "/fixture/session.jsonl",
                msg_idx: doc.msg_idx,
                created_at: doc.created_at,
                title: Some(doc.title),
                content: doc.content,
                source_id: doc.source_id,
                origin_kind: doc.origin_kind,
                origin_host: None,
                conversation_id: None,
            })
            .collect::<Vec<_>>();
        index
            .add_cass_document_refs(&oracle_documents)
            .expect("index CASS oracle documents");
        index.commit().expect("commit CASS oracle documents");
        let reader = index.reader().expect("open CASS oracle reader");
        reader.reload().expect("reload committed CASS oracle");
        let searcher = reader.searcher();
        let fields = index.fields();

        let cases = vec![
            ("auth", CassQueryFilters::default()),
            ("auth token", CassQueryFilters::default()),
            ("auth OR token AND cache", CassQueryFilters::default()),
            ("auth && cache", CassQueryFilters::default()),
            ("auth || search", CassQueryFilters::default()),
            ("\"error handling\"", CassQueryFilters::default()),
            ("foo*", CassQueryFilters::default()),
            ("*bar", CassQueryFilters::default()),
            ("*baz", CassQueryFilters::default()),
            ("*bar*", CassQueryFilters::default()),
            ("f*bar", CassQueryFilters::default()),
            ("*foo*baz", CassQueryFilters::default()),
            ("*bar *baz", CassQueryFilters::default()),
            ("f*o", CassQueryFilters::default()),
            ("搜索", CassQueryFilters::default()),
            ("auth AND NOT deprecated", CassQueryFilters::default()),
            ("auth OR NOT deprecated", CassQueryFilters::default()),
            ("NOT deprecated", CassQueryFilters::default()),
            ("-deprecated", CassQueryFilters::default()),
            (
                "cache",
                CassQueryFilters {
                    agents: vec!["claude".to_owned(), "codex".to_owned()],
                    workspaces: vec!["/alpha".to_owned(), "/beta".to_owned()],
                    ..CassQueryFilters::default()
                },
            ),
            (
                "*cache",
                CassQueryFilters {
                    agents: vec!["claude".to_owned()],
                    workspaces: vec!["/alpha".to_owned()],
                    source_filter: CassSourceFilter::Remote,
                    ..CassQueryFilters::default()
                },
            ),
            (
                "",
                CassQueryFilters {
                    created_from: Some(200),
                    created_to: Some(500),
                    ..CassQueryFilters::default()
                },
            ),
            (
                "",
                CassQueryFilters {
                    source_filter: CassSourceFilter::Local,
                    ..CassQueryFilters::default()
                },
            ),
            (
                "",
                CassQueryFilters {
                    source_filter: CassSourceFilter::Remote,
                    ..CassQueryFilters::default()
                },
            ),
            (
                "",
                CassQueryFilters {
                    source_filter: CassSourceFilter::SourceId("archive-7".to_owned()),
                    ..CassQueryFilters::default()
                },
            ),
        ];

        for (raw, filters) in cases {
            let oracle_filters = OracleFilters {
                agents: filters.agents.clone(),
                workspaces: filters.workspaces.clone(),
                created_from: filters.created_from,
                created_to: filters.created_to,
                source_filter: match &filters.source_filter {
                    CassSourceFilter::All => OracleSource::All,
                    CassSourceFilter::Local => OracleSource::Local,
                    CassSourceFilter::Remote => OracleSource::Remote,
                    CassSourceFilter::SourceId(source_id) => {
                        OracleSource::SourceId(source_id.clone())
                    }
                },
            };
            let oracle_query = cass_build_tantivy_query(raw, &oracle_filters, &fields);
            let oracle = searcher
                .search(&*oracle_query, &DocSetCollector)
                .expect("search shipping CASS builder")
                .into_iter()
                .map(|address| {
                    let document: TantivyDocument =
                        searcher.doc(address).expect("load CASS oracle document");
                    document
                        .get_first(fields.msg_idx)
                        .and_then(|value| value.as_u64())
                        .expect("stored CASS msg_idx")
                })
                .collect::<BTreeSet<_>>();
            let parsed = cass_parser().parse(raw, &filters);
            let native = DOCS
                .iter()
                .copied()
                .filter(|doc| cass_ast_matches(&parsed.query, *doc))
                .map(|doc| doc.msg_idx)
                .collect::<BTreeSet<_>>();
            assert_eq!(native, oracle, "result-set differential for {raw:?}");
        }
    }

    #[test]
    fn harvested_query_corpus_never_fails_or_panics() {
        let fixture: Value =
            serde_json::from_str(include_str!("../../../tests/fixtures/queries.json"))
                .expect("query corpus parses");
        for row in fixture.as_array().expect("query rows") {
            let query = row["query"].as_str().expect("query string");
            let _ = parser().parse(query);
        }
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn parser_result_sets_match_the_pinned_tantivy_adapter() {
        const DOCS: [EvalDoc; 10] = [
            EvalDoc {
                id: "doc-a",
                content: "rust ownership error handling",
                title: "Rust",
            },
            EvalDoc {
                id: "doc-b",
                content: "rust deprecated",
                title: "Legacy Rust",
            },
            EvalDoc {
                id: "doc-c",
                content: "deprecated ownership",
                title: "Old",
            },
            EvalDoc {
                id: "doc-d",
                content: "and",
                title: "Conjunction",
            },
            EvalDoc {
                id: "doc-e",
                content: "or",
                title: "Disjunction",
            },
            EvalDoc {
                id: "doc-f",
                content: "transformers text embeddings",
                title: "Semantic Search",
            },
            EvalDoc {
                id: "doc-g",
                content: "search index new",
                title: "SearchIndex",
            },
            EvalDoc {
                id: "doc-z",
                content: "unrelated",
                title: "Nothing",
            },
            EvalDoc {
                id: "foo bar",
                content: "space sentinel",
                title: "Escaped",
            },
            EvalDoc {
                id: "foo:bar",
                content: "colon sentinel",
                title: "Escaped",
            },
        ];

        let fixture: Value =
            serde_json::from_str(include_str!("../../../tests/fixtures/queries.json"))
                .expect("query corpus parses");
        let mut queries = fixture
            .as_array()
            .expect("query rows")
            .iter()
            .map(|row| row["query"].as_str().expect("query string").to_owned())
            .collect::<Vec<_>>();
        queries.extend(
            [
                "rust OR -deprecated",
                "-rust OR deprecated",
                "-deprecated rust OR ownership",
                "rust -deprecated OR ownership",
                "rust OR -deprecated AND ownership",
                "rust OR (-deprecated)",
                "rust AND NOT deprecated",
                "rust AND -deprecated",
                "rust NOT deprecated",
                "rust NOT",
                "NOT\tdeprecated",
                "rust NOT\tdeprecated",
                "NOT NOT rust",
                "NOT NOT",
                "rust AND",
                "rust OR",
                "rust AND\townership",
                "rust\nAND ownership",
                "rust OR OR ownership",
                "rust AND AND ownership",
                "(OR rust)",
                "rust ) ownership",
                "(rust OR ownership)^2",
                "rust^2^3",
                "id:doc-a^2^3",
                "\"rust\"^2^3",
                "\"rust ownership\"~deprecated",
                "\"rust ownership\"~4294967296",
                "title:(rust OR \"error handling\")",
                "title : ownership",
                "title: ownership",
                "+title:Rust",
                "-title:Rust",
                "foo*:bar",
                r"id:foo\ bar",
                r"id:foo\:bar",
                r"id:foo\q",
                r"SearchIndex\:\:new",
                "/rust/",
                "rust OR /ownership/",
                "-/rust/ rust",
                "rust AND /ownership/ deprecated",
                "id:[doc-a TO doc-z}",
                "id:[doc-a doc-c]",
                "id:[doc-a TO",
                "id:[doc-a",
                "id: IN [doc-a doc-c]",
                "id:([doc-a TO doc-c} OR IN [doc-z])",
            ]
            .into_iter()
            .map(str::to_owned),
        );

        asupersync::test_utils::run_test_with_cx(move |cx| async move {
            let index = TantivyIndex::in_memory().expect("create Tantivy oracle");
            let documents = DOCS
                .iter()
                .map(|doc| {
                    IndexableDocument::new(doc.id, doc.content).with_title(doc.title.to_owned())
                })
                .collect::<Vec<_>>();
            index
                .index_documents(&cx, &documents)
                .await
                .expect("index oracle documents");
            index.commit(&cx).await.expect("commit oracle documents");

            let mut mismatches = Vec::new();
            for raw in queries {
                let mut oracle = index
                    .search_doc_ids(&cx, &raw, DOCS.len())
                    .expect("execute Tantivy oracle")
                    .into_iter()
                    .map(|hit| hit.doc_id.to_string())
                    .collect::<Vec<_>>();
                oracle.sort_unstable();
                let parsed = parser().parse(&raw);
                let mut quill = DOCS
                    .iter()
                    .copied()
                    .filter(|doc| ast_matches(&parsed.query, *doc))
                    .map(|doc| doc.id.to_owned())
                    .collect::<Vec<_>>();
                quill.sort_unstable();
                if quill != oracle {
                    mismatches.push(format!(
                        "{raw:?}: quill={quill:?}, oracle={oracle:?}, ast={:?}",
                        parsed.query
                    ));
                }
            }
            assert!(
                mismatches.is_empty(),
                "result-set differential mismatches:\n{}",
                mismatches.join("\n")
            );
        });
    }

    #[test]
    fn classification_preserves_incumbent_surface() {
        assert_eq!(classify_query(""), QueryExplanation::Empty);
        assert_eq!(classify_query("   "), QueryExplanation::Empty);
        assert_eq!(classify_query("rust"), QueryExplanation::Simple);
        assert_eq!(
            classify_query("  authentication  "),
            QueryExplanation::Simple
        );
        assert_eq!(
            classify_query("\"error handling\""),
            QueryExplanation::Phrase
        );
        assert_eq!(classify_query("'single quotes'"), QueryExplanation::Phrase);
        assert_eq!(classify_query("rust async"), QueryExplanation::Boolean);
        assert_eq!(QueryExplanation::Empty.to_string(), "empty");
        assert_eq!(QueryExplanation::Simple.to_string(), "simple");
        assert_eq!(QueryExplanation::Phrase.to_string(), "phrase");
        assert_eq!(QueryExplanation::Boolean.to_string(), "boolean");

        let json =
            serde_json::to_string(&QueryExplanation::Phrase).expect("query explanation serializes");
        let decoded: QueryExplanation =
            serde_json::from_str(&json).expect("query explanation deserializes");
        assert_eq!(decoded, QueryExplanation::Phrase);
    }

    #[test]
    fn empty_whitespace_and_lone_signs_lower_to_match_none() {
        for query in ["", "   ", "-", "+"] {
            assert!(
                parser().parse(query).query.is_empty(),
                "{query:?} must not become a match-all query"
            );
        }
    }

    #[test]
    fn truncation_is_character_counted_and_boundary_safe() {
        let short = "hello world";
        assert_eq!(truncate_query(short), short);

        let at_limit = "a".repeat(MAX_QUERY_LENGTH);
        assert_eq!(truncate_query(&at_limit), at_limit);

        let over = "a".repeat(MAX_QUERY_LENGTH + 3);
        assert_eq!(truncate_query(&over).len(), MAX_QUERY_LENGTH);

        let multibyte = "é".repeat(MAX_QUERY_LENGTH + 3);
        let truncated = truncate_query(&multibyte);
        assert!(truncated.is_char_boundary(truncated.len()));
        assert_eq!(truncated.chars().count(), MAX_QUERY_LENGTH);
        assert_eq!(truncated.len(), MAX_QUERY_LENGTH * 'é'.len_utf8());

        let byte_long_but_char_short = "é".repeat(MAX_QUERY_LENGTH / 2 + 1);
        assert_eq!(
            truncate_query(&byte_long_but_char_short),
            byte_long_but_char_short
        );
    }

    #[test]
    fn parse_reports_truncation_once_and_uses_the_prefix() {
        let query = format!("needle {}", "padding ".repeat(2_000));
        let parsed = parser().parse(&query);
        assert!(parsed.was_truncated);
        assert_eq!(
            parsed
                .diagnostics
                .iter()
                .filter(|diagnostic| diagnostic.kind == QueryDiagnosticKind::Truncated)
                .count(),
            1
        );
    }

    #[test]
    fn lenient_corners_drop_or_recover_without_error() {
        let cases = [
            "-", "+", "NOT", "AND", "OR", "rust AND", "OR rust", "@user", "#hashtag", "foo:bar",
            "a+b", "hello!", "((rust)", "rust ))",
        ];
        for case in cases {
            let _ = parser().parse(case);
        }
    }

    #[test]
    fn standalone_and_leading_operator_recovery_is_pinned() {
        for (raw, normalized) in [("AND", "and"), ("OR", "or"), ("NOT", "not")] {
            let query = parser().parse(raw).query;
            assert!(
                matches!(&query, Query::Term { .. }),
                "standalone {raw} must be retained as a literal"
            );
            let Query::Term { text, .. } = query else {
                continue;
            };
            assert_eq!(text, normalized);
        }
        for raw in ["AND rust", "OR rust"] {
            let parsed = parser().parse(raw);
            assert!(matches!(parsed.query, Query::Term { ref text, .. } if text == "rust"));
            assert!(
                parsed
                    .diagnostics
                    .iter()
                    .any(|diagnostic| diagnostic.kind == QueryDiagnosticKind::SyntaxRecovery)
            );
        }
    }

    #[test]
    fn explicit_grouping_and_phrase_suffixes_are_retained() {
        let parsed = parser().parse("(rust OR ownership) AND \"error handling\"~2");
        assert!(matches!(
            &parsed.query,
            Query::Boolean {
                operator: Some(BooleanOperator::And),
                ..
            }
        ));
        let Query::Boolean {
            clauses,
            operator: Some(BooleanOperator::And),
        } = parsed.query
        else {
            return;
        };
        assert_eq!(clauses.len(), 2);
        assert!(matches!(
            clauses[1].query,
            Query::Phrase {
                slop: 2,
                prefix: false,
                ..
            }
        ));

        let recovered = parser().parse("\"error handling\"~2*");
        assert!(matches!(
            recovered.query,
            Query::Phrase {
                slop: 2,
                prefix: false,
                ..
            }
        ));
        assert!(recovered.diagnostics.iter().any(|diagnostic| {
            diagnostic.kind == QueryDiagnosticKind::SyntaxRecovery
                && diagnostic.message.contains("phrase suffix")
        }));
    }

    #[test]
    fn analyzed_phrase_terms_retain_positions() {
        let query = parser().parse("src/main.rs").query;
        assert!(matches!(&query, Query::Phrase { .. }));
        let Query::Phrase { terms, .. } = query else {
            return;
        };
        assert_eq!(
            terms,
            [
                PositionedTerm::new(0, "src"),
                PositionedTerm::new(1, "main"),
                PositionedTerm::new(2, "rs"),
            ]
        );
    }

    /// A trailing operand that duplicates an explicit-OR-run member must
    /// SURVIVE: the pinned grammar nests the OR run one level below the
    /// implicit disjunction, so duplicate retention never crosses that
    /// boundary (bd-htcun) — and the nesting then FLATTENS for execution so
    /// score accumulation associates exactly like the oracle's. A pure
    /// implicit chain still dedups.
    #[test]
    fn mixed_chain_duplicate_survives_or_run_nesting() {
        let mixed = parser().parse("alpha OR beta AND gamma alpha");
        let Query::Boolean { clauses, .. } = &mixed.query else {
            panic!("mixed chain must lower to a boolean: {:?}", mixed.query);
        };
        assert_eq!(
            clauses.len(),
            3,
            "flattened execution shape: alpha, the AND group, and the \
             SURVIVING trailing duplicate: {clauses:?}"
        );
        assert!(
            matches!(&clauses[0].query, Query::Term { text, .. } if text == "alpha"),
            "leading operand first: {clauses:?}"
        );
        assert!(
            matches!(&clauses[2].query, Query::Term { text, .. } if text == "alpha"),
            "the trailing duplicate survives after flattening: {clauses:?}"
        );

        let pure = parser().parse("alpha alpha");
        let Query::Boolean {
            clauses: pure_clauses,
            ..
        } = &pure.query
        else {
            panic!(
                "two-operand implicit chain lowers to a boolean: {:?}",
                pure.query
            );
        };
        assert_eq!(
            pure_clauses.len(),
            1,
            "a pure implicit chain still retains only the first duplicate: {pure_clauses:?}"
        );
    }

    #[test]
    fn field_ids_not_backend_handles_are_stored_in_leaves() {
        let parsed = parser().parse("title:Rust");
        assert_eq!(
            parsed.query,
            Query::Term {
                fields: vec![QueryField::new(2, TITLE_BOOST)],
                text: "rust".to_owned(),
            }
        );

        let id = parser().parse("id:Case-Sensitive");
        assert_eq!(
            id.query,
            Query::Term {
                fields: vec![QueryField::new(0, 1.0)],
                text: "Case-Sensitive".to_owned(),
            }
        );
    }

    #[test]
    fn boosts_groups_and_scopes_canonicalize_before_dedup() {
        // The pinned Tantivy grammar removes identical syntactic children
        // before field expansion. Parentheses disappear at leaf parsing,
        // identity boosts disappear before the rewrite, and boost equality is
        // based on parsed f64 values.
        for raw in [
            "rust rust",
            "(rust) rust",
            "rust rust^1",
            "rust^2 rust^2.0",
            "rust OR rust",
        ] {
            let Query::Boolean { clauses, .. } = parser().parse(raw).query else {
                panic!("{raw:?} must retain the rewritten root clause");
            };
            assert_eq!(clauses.len(), 1, "{raw:?} must score one clause");
        }

        // Syntax identity precedes analysis: case and quote delimiter remain
        // distinct even when the analyzer produces the same lowercase term.
        for raw in ["Rust rust", "\"rust\" rust", "'rust' \"rust\""] {
            let Query::Boolean { clauses, .. } = &parser().parse(raw).query else {
                panic!("{raw:?} must retain both scoring clauses");
            };
            assert_eq!(clauses.len(), 2, "{raw:?}");
            assert_eq!(clauses[0].query, clauses[1].query, "{raw:?}");
        }
        // Occurrence is part of the syntax key, and duplicate exclusions merge
        // exactly as duplicate positive clauses do.
        let merged = parser().parse("-rust -rust").query;
        let count_must_not = |query: &Query| match query {
            Query::Boolean { clauses, .. } => clauses
                .iter()
                .filter(|clause| clause.occur == Occur::MustNot)
                .count(),
            _ => 0,
        };
        assert_eq!(count_must_not(&merged), 1, "duplicate exclusions merge");

        for raw in ["rust^2^3", "(rust)^2^3", "\"rust\"^2^3"] {
            let recovered = parser().parse(raw);
            assert!(matches!(
                recovered.query,
                Query::Boost { factor, .. } if factor.to_bits() == 2.0_f32.to_bits()
            ));
            assert!(recovered.diagnostics.iter().any(|diagnostic| {
                diagnostic.kind == QueryDiagnosticKind::SyntaxRecovery
                    && diagnostic.message.contains("trailing boost")
            }));
        }
        let leading_decimal = parser().parse("rust^.5");
        assert!(matches!(leading_decimal.query, Query::Term { .. }));
        assert!(
            leading_decimal
                .diagnostics
                .iter()
                .any(|diagnostic| { diagnostic.kind == QueryDiagnosticKind::InvalidBoost })
        );
        let trailing_decimal = parser().parse("rust^2.");
        assert!(matches!(
            trailing_decimal.query,
            Query::Boost { factor, .. } if factor.to_bits() == 2.0_f32.to_bits()
        ));
        assert!(trailing_decimal.diagnostics.iter().any(|diagnostic| {
            diagnostic.kind == QueryDiagnosticKind::SyntaxRecovery
                && diagnostic.message.contains("trailing boost")
        }));

        let invalid = parser().parse("rust rust^scientific1e2");
        // The invalid boost is rejected and erased from the syntax tree, so
        // the surviving atom deduplicates with its bare sibling.
        assert!(matches!(
            invalid.query,
            Query::Boolean { ref clauses, .. } if clauses.len() == 1
        ));
        assert!(
            invalid
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.kind == QueryDiagnosticKind::InvalidBoost)
        );

        let scoped = parser().parse("title:(rust) content:(rust)").query;
        let Query::Boolean { clauses, .. } = scoped else {
            return;
        };
        assert_eq!(clauses.len(), 2);
        assert_ne!(clauses[0].query, clauses[1].query);

        let recovered = parser().parse("unknown:(bad title:good)");
        assert!(
            ast_matches(
                &recovered.query,
                EvalDoc {
                    id: "explicit-child",
                    content: "bad",
                    title: "good",
                },
            ),
            "the valid explicit inner field was not preserved: {:?}",
            recovered.query
        );
    }

    #[test]
    fn parser_syntax_rewrite_stops_at_non_identity_boosts() {
        let Query::Boost { query, factor } = parser().parse("(rust rust)^2").query else {
            panic!("the group boost must survive");
        };
        assert_eq!(factor.to_bits(), 2.0_f32.to_bits());
        let Query::Boolean { clauses, .. } = *query else {
            panic!("the boosted group must retain its Boolean syntax boundary");
        };
        assert_eq!(clauses.len(), 2, "rewrite_ast does not enter Boost");
        assert_eq!(clauses[0].query, clauses[1].query);

        let Query::Boolean { clauses, .. } = parser().parse("(rust rust)^2 rust^2").query else {
            panic!("the distinct boosted siblings must both survive");
        };
        assert_eq!(clauses.len(), 2);
        assert_ne!(clauses[0].query, clauses[1].query);
    }

    #[test]
    fn parser_syntax_rewrite_flattens_singletons_and_transfers_occurrence() {
        assert!(matches!(parser().parse("+a").query, Query::Term { .. }));
        assert!(matches!(parser().parse("(+a)").query, Query::Term { .. }));
        let Query::Boolean { clauses, .. } = parser().parse("(+a) a").query else {
            panic!("the rewritten duplicate keeps its root Boolean");
        };
        assert_eq!(clauses.len(), 1, "singleton unary Must is erased first");

        let Query::Boolean { clauses, .. } = parser().parse("(+unknown:x^2) a").query else {
            panic!("the optional boosted-empty branch and term remain Boolean");
        };
        assert_eq!(clauses.len(), 2);
        assert!(clauses.iter().all(|clause| clause.occur == Occur::Should));
        assert!(matches!(clauses[0].query, Query::Boost { .. }));

        let Query::Boolean { clauses, .. } = parser().parse("(-a) b").query else {
            panic!("negative group plus optional term must remain Boolean");
        };
        assert_eq!(clauses.len(), 2);
        assert_eq!(clauses[0].occur, Occur::MustNot);
        assert_eq!(clauses[1].occur, Occur::Should);

        let Query::Boolean { clauses, .. } = parser().parse("(a AND a) b").query else {
            panic!("required group plus optional term must remain Boolean");
        };
        assert_eq!(clauses.len(), 2);
        assert_eq!(clauses[0].occur, Occur::Must);
        assert_eq!(clauses[1].occur, Occur::Should);

        let Query::Boolean { clauses, .. } = parser().parse("((a AND a) b) (+a b)").query else {
            panic!("the rewritten root must remain Boolean");
        };
        assert_eq!(clauses.len(), 1, "recursively equal groups deduplicate");
        let Query::Boolean { clauses: inner, .. } = &clauses[0].query else {
            panic!("the two-clause child must retain its grouping");
        };
        assert_eq!(inner.len(), 2);
        assert_eq!(inner[0].occur, Occur::Must);
        assert_eq!(inner[1].occur, Occur::Should);

        let Query::Boolean { clauses, .. } = parser().parse("(a AND a) +a").query else {
            panic!("dedup-before-flatten must leave the root Boolean");
        };
        assert_eq!(clauses.len(), 2, "the rewrite does not dedup a second time");
        assert!(clauses.iter().all(|clause| clause.occur == Occur::Must));
        assert_eq!(clauses[0].query, clauses[1].query);
    }

    #[test]
    fn parser_syntax_rewrite_retains_lowered_away_raw_identity_until_trim() {
        let Query::Boolean { clauses, .. } = parser().parse("(a /x/) (a /y/)").query else {
            panic!("raw-distinct groups must both survive");
        };
        assert_eq!(clauses.len(), 2);
        // Raw-identity dedup ran on the NESTED syntax (distinct /x/ vs /y/
        // prevent it), then the post-trim singleton groups splice into the
        // parent for oracle-associated execution (bd-htcun flatten).
        assert!(
            clauses
                .iter()
                .all(|clause| { matches!(&clause.query, Query::Term { text, .. } if text == "a") })
        );

        let Query::Boolean { clauses, .. } = parser().parse("(a /x/) (a /x/)").query else {
            panic!("the rewritten duplicate keeps its root Boolean");
        };
        assert_eq!(clauses.len(), 1, "raw-identical groups deduplicate");

        let Query::Boolean { clauses, .. } = parser().parse("(a unknown:x) (a unknown:y)").query
        else {
            panic!("distinct unknown-field syntax must prevent parent dedup");
        };
        assert_eq!(clauses.len(), 2);

        let Query::Boost { query, .. } = parser().parse("(a AND unknown:x)^2").query else {
            panic!("a valid boost remains a lowering boundary");
        };
        let Query::Boolean { clauses, .. } = *query else {
            panic!("the boosted conjunction remains Boolean");
        };
        assert_eq!(clauses.len(), 1, "semantic drops trim inside Boost");
        assert_eq!(clauses[0].occur, Occur::Must);

        let Query::Boolean { clauses, .. } = parser().parse("a unknown:x^1.00000001").query else {
            panic!("the valid sibling keeps the root Boolean");
        };
        assert_eq!(
            clauses.len(),
            1,
            "an f64-only boost boundary erases after its f32 identity cast"
        );
    }

    #[test]
    fn overflowing_boost_recovers_without_nonfinite_score() {
        for overflowing in [format!("4{}", "0".repeat(38)), "9".repeat(400)] {
            let parsed = parser().parse(&format!("rust rust^{overflowing}"));
            assert!(
                parsed
                    .diagnostics
                    .iter()
                    .any(|diagnostic| { diagnostic.kind == QueryDiagnosticKind::InvalidBoost })
            );
            let Query::Boolean { clauses, .. } = parsed.query else {
                panic!("the recovered duplicate keeps the rewritten root clause");
            };
            assert_eq!(clauses.len(), 1, "the invalid boost is absent from its key");

            let Query::Boolean { clauses, .. } =
                parser().parse(&format!("(rust rust)^{overflowing}")).query
            else {
                panic!("the recovered group keeps its rewritten root Boolean");
            };
            assert_eq!(clauses.len(), 1, "invalid boost cannot block child rewrite");
        }
    }

    #[test]
    fn typed_ranges_sets_comparisons_and_partial_recovery_are_pinned() {
        let typed_parser =
            DefaultQueryParser::new(TYPED_SCHEMA).expect("typed test schema is valid");
        assert_eq!(
            typed_parser.parse("signed:[-5 TO 10}").query,
            Query::Range {
                field_id: 3,
                lower: Bound::Included(QueryValue::I64(-5)),
                upper: Bound::Excluded(QueryValue::I64(10)),
            }
        );
        assert_eq!(
            typed_parser
                .parse("unsigned: >= 18446744073709551615")
                .query,
            Query::Range {
                field_id: 4,
                lower: Bound::Included(QueryValue::U64(u64::MAX)),
                upper: Bound::Unbounded,
            }
        );

        for raw in [
            "signed:[* TO 5] signed:{* TO 5]",
            "signed:[-5 TO *] signed:[-5 TO *}",
            "signed:<5 signed:[* TO 5}",
            "signed:>=5 signed:[5 TO *]",
        ] {
            let Query::Boolean { clauses, .. } = typed_parser.parse(raw).query else {
                panic!("equivalent unbounded ranges must retain one root clause: {raw}");
            };
            assert_eq!(
                clauses.len(),
                1,
                "unbounded delimiters normalize before parser dedup: {raw}"
            );
        }

        let partial = typed_parser.parse("unsigned:[bad TO 3]");
        assert_eq!(
            partial.query,
            Query::Range {
                field_id: 4,
                lower: Bound::Unbounded,
                upper: Bound::Included(QueryValue::U64(3)),
            }
        );
        assert!(
            partial
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.kind == QueryDiagnosticKind::InvalidTypedValue)
        );

        let set = typed_parser.parse("unsigned: IN [1 bad 2 1]");
        assert_eq!(
            set.query,
            Query::Set {
                field_id: 4,
                values: vec![QueryValue::U64(1), QueryValue::U64(2)],
            }
        );
        assert_eq!(
            typed_parser.parse("unsigned: IN []").query,
            Query::Set {
                field_id: 4,
                values: Vec::new(),
            }
        );
        assert_eq!(
            parser().parse(r#"id: IN [foo,bar ""]"#).query,
            Query::Set {
                field_id: 0,
                values: vec![
                    QueryValue::Str("foo,bar".to_owned()),
                    QueryValue::Str(String::new()),
                ],
            }
        );
        assert_eq!(
            parser().parse(r#"id: IN ["foo]bar"]"#).query,
            Query::Set {
                field_id: 0,
                values: vec![QueryValue::Str("foo]bar".to_owned())],
            }
        );
        assert_eq!(
            parser().parse(r#"id: IN ["foo\q"]"#).query,
            Query::Set {
                field_id: 0,
                values: vec![QueryValue::Str("fooq".to_owned())],
            }
        );
        let unterminated_set = parser().parse("id: IN [doc-a doc-b");
        assert_eq!(
            unterminated_set.query,
            Query::Set {
                field_id: 0,
                values: vec![
                    QueryValue::Str("doc-a".to_owned()),
                    QueryValue::Str("doc-b".to_owned()),
                ],
            }
        );
        assert!(unterminated_set.diagnostics.iter().any(|diagnostic| {
            diagnostic.kind == QueryDiagnosticKind::SyntaxRecovery
                && diagnostic.message.contains("unterminated typed set")
        }));
        assert_eq!(
            parser().parse(r"id:[foo\ bar TO z]").query,
            Query::Range {
                field_id: 0,
                lower: Bound::Included(QueryValue::Str("foo bar".to_owned())),
                upper: Bound::Included(QueryValue::Str("z".to_owned())),
            }
        );
        assert_eq!(
            parser().parse(r"id:[foo\\:bar TO z]").query,
            Query::Range {
                field_id: 0,
                lower: Bound::Included(QueryValue::Str(r"foo\:bar".to_owned())),
                upper: Bound::Included(QueryValue::Str("z".to_owned())),
            }
        );
        let unterminated = parser().parse("id:[a TO z");
        assert_eq!(
            unterminated.query,
            Query::Range {
                field_id: 0,
                lower: Bound::Included(QueryValue::Str("a".to_owned())),
                upper: Bound::Included(QueryValue::Str("z".to_owned())),
            }
        );
        assert!(unterminated.diagnostics.iter().any(|diagnostic| {
            diagnostic.kind == QueryDiagnosticKind::SyntaxRecovery
                && diagnostic.message.contains("unterminated typed range")
        }));
        let missing_to = parser().parse("id:[a z]");
        assert_eq!(
            missing_to.query,
            Query::Range {
                field_id: 0,
                lower: Bound::Included(QueryValue::Str("a".to_owned())),
                upper: Bound::Included(QueryValue::Str("z".to_owned())),
            }
        );
        assert!(missing_to.diagnostics.iter().any(|diagnostic| {
            diagnostic.kind == QueryDiagnosticKind::SyntaxRecovery
                && diagnostic.message.contains("missing keyword `TO`")
        }));
        for raw in ["id:[a TO", "id:[a"] {
            assert_eq!(
                parser().parse(raw).query,
                Query::Range {
                    field_id: 0,
                    lower: Bound::Included(QueryValue::Str("a".to_owned())),
                    upper: Bound::Unbounded,
                },
                "{raw}"
            );
        }
        assert!(typed_parser.parse("stored:[a TO b]").query.is_empty());
        let positionless_phrase = typed_parser.parse("summary:\"two words\"");
        assert!(matches!(
            positionless_phrase.query,
            Query::Phrase { ref terms, .. } if terms.len() == 2
        ));
        assert!(matches!(
            validate_index_capabilities(&positionless_phrase.query, TYPED_SCHEMA),
            Err(QueryCapabilityError::PositionsRequired {
                schema: "query-parser-typed-test",
                ref field,
                operator: QueryExplanation::Phrase,
                capability: IndexCapability::Positions,
            }) if field == "summary"
        ));
        assert!(typed_parser.parse("signed:[* TO *]").query.is_empty());

        assert_eq!(
            parser().parse("ord:[0 TO 1]").query,
            Query::Range {
                field_id: 4,
                lower: Bound::Included(QueryValue::U64(0)),
                upper: Bound::Included(QueryValue::U64(1)),
            }
        );
        assert_eq!(
            parser().parse("ord: IN [0 1]").query,
            Query::Set {
                field_id: 4,
                values: vec![QueryValue::U64(0), QueryValue::U64(1)],
            }
        );
    }

    #[test]
    fn recovery_retains_valid_siblings_and_repairs_recursive_negatives() {
        for raw in ["(OR rust)", "(AND rust)"] {
            let parsed = parser().parse(raw);
            assert!(matches!(parsed.query, Query::Term { ref text, .. } if text == "rust"));
            assert!(parsed.diagnostics.iter().any(|diagnostic| {
                diagnostic.kind == QueryDiagnosticKind::SyntaxRecovery
                    && diagnostic.byte_offset == Some(1)
            }));
        }

        let unmatched = parser().parse("rust ) ownership");
        assert!(matches!(
            unmatched.query,
            Query::Boolean { ref clauses, .. }
                if clauses.len() == 1
                    && matches!(&clauses[0].query, Query::Term { text, .. } if text == "rust")
        ));
        assert!(unmatched.diagnostics.iter().any(|diagnostic| {
            diagnostic.kind == QueryDiagnosticKind::SyntaxRecovery
                && diagnostic.byte_offset == Some(5)
        }));
        assert_eq!(
            parser()
                .parse("rust ))")
                .diagnostics
                .iter()
                .filter(|diagnostic| diagnostic.kind == QueryDiagnosticKind::SyntaxRecovery)
                .count(),
            1
        );

        let all_negative = parser().parse("(-alpha) OR (-beta)");
        let Query::Boolean { clauses, .. } = all_negative.query else {
            return;
        };
        assert!(matches!(
            clauses.last(),
            Some(BooleanClause {
                occur: Occur::Should,
                query: Query::All
            })
        ));
        assert!(
            all_negative
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.kind == QueryDiagnosticKind::AllNegativeRepair)
        );

        let boosted = parser().parse("(-alpha)^2");
        assert!(matches!(
            boosted.query,
            Query::Boost { ref query, .. }
                if matches!(query.as_ref(), Query::Boolean { clauses, .. }
                    if matches!(clauses.last(), Some(BooleanClause {
                        occur: Occur::Should,
                        query: Query::All,
                    })))
        ));
    }

    #[test]
    fn quoted_escapes_single_quotes_and_phrase_prefix_errors_are_pinned() {
        assert!(matches!(
            parser().parse("title:'error handling'").query,
            Query::Phrase { ref terms, .. } if terms.len() == 2
        ));
        assert!(matches!(
            parser().parse(r#"title:"say \"hello\"""#).query,
            Query::Phrase { ref terms, .. }
                if terms.iter().map(|term| term.text.as_str()).eq(["say", "hello"])
        ));
        let prefix = parser().parse("\"word\"*");
        assert!(prefix.query.is_empty());
        assert!(
            prefix
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.kind == QueryDiagnosticKind::DroppedFragment)
        );

        let malformed_slop = parser().parse("\"rust ownership\"~deprecated");
        assert!(matches!(
            malformed_slop.query,
            Query::Boolean { ref clauses, .. }
                if clauses.len() == 2
                    && matches!(&clauses[0].query, Query::Phrase { slop: 0, .. })
                    && matches!(&clauses[1].query, Query::Term { text, .. } if text == "deprecated")
        ));
        assert!(malformed_slop.diagnostics.iter().any(|diagnostic| {
            diagnostic.kind == QueryDiagnosticKind::SyntaxRecovery
                && diagnostic
                    .message
                    .contains("retained as a separate fragment")
        }));

        let overflowing_slop = parser().parse("\"rust ownership\"~4294967296");
        assert!(matches!(
            overflowing_slop.query,
            Query::Boolean { ref clauses, .. }
                if clauses.len() == 2
                    && matches!(&clauses[0].query, Query::Phrase { slop: 0, .. })
                    && matches!(&clauses[1].query, Query::Term { text, .. } if text == "4294967296")
        ));
    }

    #[test]
    fn unquoted_escapes_and_unsupported_regex_recover_without_retargeting() {
        assert_eq!(
            parser().parse(r"id:foo\ bar").query,
            Query::Term {
                fields: vec![QueryField::new(0, 1.0)],
                text: "foo bar".to_owned(),
            }
        );
        assert_eq!(
            parser().parse(r"id:foo\:bar").query,
            Query::Term {
                fields: vec![QueryField::new(0, 1.0)],
                text: "foo:bar".to_owned(),
            }
        );
        assert_eq!(
            parser().parse(r"id:foo\q").query,
            Query::Term {
                fields: vec![QueryField::new(0, 1.0)],
                text: r"foo\q".to_owned(),
            }
        );

        let regex = parser().parse("/rust/");
        assert!(regex.query.is_empty());
        assert!(regex.diagnostics.iter().any(|diagnostic| {
            diagnostic.kind == QueryDiagnosticKind::DroppedFragment
                && diagnostic.message.contains("regular-expression")
        }));
        assert!(matches!(
            parser().parse("-/rust/ rust").query,
            Query::Boolean { ref clauses, .. }
                if clauses.len() == 1
                    && matches!(&clauses[0].query, Query::Term { text, .. } if text == "rust")
        ));
    }

    #[test]
    fn parser_dedup_is_raw_and_preserves_distinct_normalized_fragments() {
        let distinct = parser().parse("Rust rust").query;
        assert!(matches!(&distinct, Query::Boolean { .. }));
        let Query::Boolean { clauses, .. } = distinct else {
            return;
        };
        assert_eq!(clauses.len(), 2);
        assert_eq!(clauses[0].query, clauses[1].query);

        // Idempotent MustNot clauses DO merge, even nested.
        let recursive = parser().parse("a OR -b OR (-b)").query;
        assert!(matches!(
            recursive,
            Query::Boolean { ref clauses, .. } if clauses.len() == 2
        ));
    }

    #[test]
    fn all_negative_repair_is_root_only() {
        let query = parser().parse("rust OR (-deprecated)").query;
        assert!(matches!(&query, Query::Boolean { .. }));
        let Query::Boolean { clauses, .. } = query else {
            return;
        };
        assert!(matches!(&clauses[1].query, Query::Boolean { .. }));
        let Query::Boolean {
            clauses: negative, ..
        } = &clauses[1].query
        else {
            return;
        };
        assert_eq!(negative.len(), 1);
        assert_eq!(negative[0].occur, Occur::MustNot);
    }

    /// A position-free text schema is a valid, smaller index configuration.
    /// Before `bd-vmpb` it was unqueryable end to end: `DefaultQueryParser::new`
    /// refused to construct, so the QG `positions_off` fixture panicked and
    /// aborted every QG-1 matrix run before any evidence could be written.
    const POSITIONLESS_FIELDS: [FieldDescriptor; 2] = [
        FieldDescriptor {
            id: 0,
            name: "content",
            kind: FieldKind::Text {
                analyzer: Analyzer::FrankensearchDefault,
                positions: false,
            },
            stored: true,
        },
        FieldDescriptor {
            id: 1,
            name: "title",
            kind: FieldKind::Text {
                analyzer: Analyzer::FrankensearchDefault,
                positions: false,
            },
            stored: true,
        },
    ];
    const POSITIONLESS_SCHEMA: SchemaDescriptor = SchemaDescriptor {
        name: "query-parser-positionless-test",
        fields: &POSITIONLESS_FIELDS,
    };

    #[test]
    fn positionless_default_fields_are_accepted_and_parse_position_independent_queries() {
        let parser = DefaultQueryParser::new(POSITIONLESS_SCHEMA)
            .expect("a positionless default-analyzed schema must bind a parser (bd-vmpb)");

        // Every position-independent operator must parse AND clear the
        // capability gate on a positionless index.
        for query in [
            "alpha",
            "alpha beta",
            "alpha OR beta",
            "alpha AND beta",
            "-alpha beta",
            "content:alpha",
            "alph*",
        ] {
            let parsed = parser.parse(query);
            validate_index_capabilities(&parsed.query, POSITIONLESS_SCHEMA).unwrap_or_else(
                |error| panic!("{query:?} is position-independent and must be servable: {error}"),
            );
        }
    }

    /// The capability gate rejects a phrase regardless of whether it came
    /// from the string parser or an already-built AST.
    #[test]
    fn capability_gate_rejects_a_constructed_phrase_on_positionless_fields() {
        let phrase = Query::Phrase {
            fields: vec![QueryField::new(0, 1.0)],
            terms: vec![
                PositionedTerm {
                    position: 0,
                    text: "alpha".to_owned(),
                },
                PositionedTerm {
                    position: 1,
                    text: "beta".to_owned(),
                },
            ],
            slop: 0,
            prefix: false,
        };

        let error = validate_index_capabilities(&phrase, POSITIONLESS_SCHEMA)
            .expect_err("a multi-term phrase reads positions and cannot be served");
        let QueryCapabilityError::PositionsRequired {
            schema,
            field,
            operator,
            capability,
        } = error;
        assert_eq!(schema, "query-parser-positionless-test");
        // The error names the offending FIELD, not a bare "unsupported query".
        assert_eq!(field, "content");
        assert_eq!(operator, QueryExplanation::Phrase);
        assert_eq!(capability, IndexCapability::Positions);

        // Nesting must not launder it.
        let nested = Query::Boolean {
            clauses: vec![BooleanClause {
                occur: Occur::Should,
                query: phrase,
            }],
            operator: None,
        };
        assert!(validate_index_capabilities(&nested, POSITIONLESS_SCHEMA).is_err());
    }

    #[test]
    fn positionless_parser_preserves_phrases_for_typed_capability_rejection() {
        let parser =
            DefaultQueryParser::new(POSITIONLESS_SCHEMA).expect("bind positionless parser");
        let parsed = parser.parse("\"alpha beta\"");

        assert_ne!(
            parsed.query,
            Query::Empty,
            "a capability mismatch must not silently erase the parsed branch"
        );
        let error = validate_index_capabilities(&parsed.query, POSITIONLESS_SCHEMA)
            .expect_err("the preserved phrase must fail at the typed capability boundary");
        assert!(matches!(
            error,
            QueryCapabilityError::PositionsRequired {
                schema: "query-parser-positionless-test",
                ref field,
                operator: QueryExplanation::Phrase,
                capability: IndexCapability::Positions,
            } if field == "content"
        ));
    }

    #[test]
    fn single_term_phrase_needs_no_positions() {
        let parser =
            DefaultQueryParser::new(POSITIONLESS_SCHEMA).expect("bind positionless parser");
        let parsed = parser.parse("\"alpha\"");
        // A one-term phrase degenerates to a term match and reads no position
        // list, which is the same exemption lowering already applies.
        validate_index_capabilities(&parsed.query, POSITIONLESS_SCHEMA)
            .expect("a single-term phrase reads no positions");
    }

    #[test]
    fn positioned_schema_still_serves_phrases() {
        let parser = DefaultQueryParser::new(DEFAULT_SCHEMA).expect("bind shipping parser");
        let parsed = parser.parse("\"alpha beta\"");
        validate_index_capabilities(&parsed.query, DEFAULT_SCHEMA)
            .expect("positionful behaviour must be unchanged");
    }

    #[test]
    fn schemas_without_both_default_fields_are_rejected() {
        assert!(matches!(
            DefaultQueryParser::new(FSFS_CHUNK_SCHEMA),
            Err(QueryParserConfigError::MissingDefaultField { field: "title", .. })
        ));

        // A default field with the WRONG ANALYZER is still rejected: query-time
        // term text would not match the text indexing produced, so the schema
        // genuinely cannot be served. Positions are a separate axis — see
        // `positionless_default_fields_are_accepted` (`bd-vmpb`).
        const WRONG_ANALYZER_FIELDS: [FieldDescriptor; 2] = [
            FieldDescriptor {
                id: 0,
                name: "content",
                kind: FieldKind::Text {
                    analyzer: Analyzer::CassHyphenNormalize,
                    positions: true,
                },
                stored: true,
            },
            FieldDescriptor {
                id: 1,
                name: "title",
                kind: FieldKind::Text {
                    analyzer: Analyzer::FrankensearchDefault,
                    positions: true,
                },
                stored: true,
            },
        ];
        const WRONG_ANALYZER_SCHEMA: SchemaDescriptor = SchemaDescriptor {
            name: "query-parser-wrong-analyzer-test",
            fields: &WRONG_ANALYZER_FIELDS,
        };
        assert!(matches!(
            DefaultQueryParser::new(WRONG_ANALYZER_SCHEMA),
            Err(QueryParserConfigError::InvalidDefaultField {
                field: "content",
                ..
            })
        ));

        const GAPPED_FIELDS: [FieldDescriptor; 2] = [
            FieldDescriptor {
                id: 0,
                name: "content",
                kind: FieldKind::Text {
                    analyzer: Analyzer::FrankensearchDefault,
                    positions: true,
                },
                stored: true,
            },
            FieldDescriptor {
                id: 2,
                name: "title",
                kind: FieldKind::Text {
                    analyzer: Analyzer::FrankensearchDefault,
                    positions: true,
                },
                stored: true,
            },
        ];
        const GAPPED_SCHEMA: SchemaDescriptor = SchemaDescriptor {
            name: "query-parser-gapped-test",
            fields: &GAPPED_FIELDS,
        };
        assert!(matches!(
            DefaultQueryParser::new(GAPPED_SCHEMA),
            Err(QueryParserConfigError::InvalidSchema { .. })
        ));
    }

    #[test]
    fn range_and_glob_ast_variants_use_stable_field_ids() {
        let range = Query::Range {
            field_id: 5,
            lower: Bound::Included(QueryValue::I64(10)),
            upper: Bound::Excluded(QueryValue::I64(20)),
        };
        let set = Query::Set {
            field_id: 5,
            values: vec![QueryValue::I64(10), QueryValue::I64(20)],
        };
        let glob = Query::Glob {
            field_ids: vec![6, 7],
            pattern: "cache*".to_owned(),
        };
        assert!(matches!(range, Query::Range { field_id: 5, .. }));
        assert!(matches!(set, Query::Set { field_id: 5, .. }));
        assert!(matches!(glob, Query::Glob { field_ids, .. } if field_ids == [6, 7]));
    }

    #[test]
    fn recursive_depth_is_bounded() {
        let query = format!(
            "{}needle{}",
            "(".repeat(MAX_QUERY_DEPTH + 20),
            ")".repeat(MAX_QUERY_DEPTH + 20)
        );
        let parsed = parser().parse(&query);
        assert!(
            parsed
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.kind == QueryDiagnosticKind::DepthLimit)
        );
    }

    #[test]
    fn deterministic_hostile_strings_do_not_panic() {
        let alphabet = b"abc XYZ0123\"'():^+-*/._@#\t\n";
        let alphabet_len = u64::try_from(alphabet.len()).expect("alphabet length fits u64");
        let mut state = 0x6a09_e667_f3bc_c909_u64;
        for length in 0..512 {
            let mut input = String::with_capacity(length);
            for _ in 0..length {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let index =
                    usize::try_from(state % alphabet_len).expect("alphabet index fits usize");
                input.push(char::from(alphabet[index]));
            }
            let _ = parser().parse(&input);
        }
    }

    #[derive(Clone, Debug)]
    struct TestLogWriter {
        buffer: Arc<Mutex<Vec<u8>>>,
    }

    impl Write for TestLogWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.buffer
                .lock()
                .expect("test log buffer lock is not poisoned")
                .extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn every_lenient_diagnostic_is_also_logged_at_warn() {
        let buffer = Arc::new(Mutex::new(Vec::<u8>::new()));
        let writer_buffer = Arc::clone(&buffer);
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .without_time()
            .with_max_level(tracing::Level::WARN)
            .with_writer(move || TestLogWriter {
                buffer: Arc::clone(&writer_buffer),
            })
            .finish();
        let capture_buffer = Arc::clone(&buffer);
        let parsed = tracing::subscriber::with_default(subscriber, || {
            let mut parsed = parser().parse("unknown_field:value \"unterminated");
            for _ in 0..256 {
                let captured_bytes = capture_buffer
                    .lock()
                    .expect("test log buffer lock is not poisoned");
                let captured = String::from_utf8_lossy(&captured_bytes).lines().count();
                if captured >= parsed.diagnostics.len() {
                    break;
                }
                drop(captured_bytes);
                tracing::callsite::rebuild_interest_cache();
                std::thread::sleep(std::time::Duration::from_millis(1));
                parsed = parser().parse("unknown_field:value \"unterminated");
            }
            parsed
        });
        assert!(!parsed.diagnostics.is_empty());
        let logs = String::from_utf8(
            buffer
                .lock()
                .expect("test log buffer lock is not poisoned")
                .clone(),
        )
        .expect("captured tracing output is UTF-8");
        let diagnostic_lines = logs
            .lines()
            .filter(|line| line.contains("lenient query parse diagnostic"))
            .collect::<Vec<_>>();
        assert!(
            diagnostic_lines.len() >= parsed.diagnostics.len(),
            "captured {} warning events for {} diagnostics: {logs:?}",
            diagnostic_lines.len(),
            parsed.diagnostics.len()
        );
        assert!(diagnostic_lines.iter().all(|line| {
            line.trim_start().starts_with("WARN ")
                && line.contains(crate::tracing_conventions::TARGET)
        }));
        assert!(!logs.contains("unknown_field"));
        assert!(!logs.contains("unterminated"));
        assert!(!logs.contains("value"));
    }

    /// Parse through the real Grammar while bypassing the public 10,000-byte
    /// truncation, so the oversized-token admission branch is exercisable.
    fn parse_untruncated(parser: &DefaultQueryParser, query: &str) -> ParsedQuery {
        let mut diagnostics = Vec::new();
        let tokens = lex(query, &mut diagnostics);
        let mut grammar = Grammar {
            parser: *parser,
            tokens,
            cursor: 0,
            diagnostics,
            lowered_atoms: Vec::new(),
            field_scopes: Vec::new(),
        };
        grammar.recover_leading_binary_operators();
        let mut parsed = grammar.parse_expression(0);
        grammar.recover_trailing_tokens();
        if let Some(node) = parsed.as_mut() {
            rewrite_parser_syntax(node);
            trim_parser_dropped(&mut node.query, &node.dedup_key);
            flatten_should_of_should(&mut node.query);
        }
        repair_root_all_negative(&mut parsed, &mut grammar);
        ParsedQuery {
            query: parsed.map_or(Query::Empty, |node| node.query),
            explanation: classify_query(query),
            diagnostics: grammar.diagnostics,
            was_truncated: false,
        }
    }

    /// One ASCII token of exactly `bytes` length plus its quoted variant.
    fn oversized_atom() -> String {
        "x".repeat(crate::grimoire::MAX_TERM_BYTES + 1)
    }

    #[test]
    fn public_query_strings_cannot_carry_oversized_tokens() {
        let parser = parser();
        // A 70,000-byte single token exceeds the 65,530-byte admission bound,
        // but the public string cap (10,000 bytes) truncates long before the
        // admission rule can ever see it.
        let query = "x".repeat(70_000);
        let parsed = parser.parse(&query);
        assert!(parsed.was_truncated);
        assert!(
            parsed
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.kind == QueryDiagnosticKind::Truncated)
        );
        fn max_term_bytes(query: &Query) -> usize {
            match query {
                Query::Term { text, .. } => text.len(),
                Query::Phrase { terms, .. } => {
                    terms.iter().map(|term| term.text.len()).max().unwrap_or(0)
                }
                Query::Boolean { clauses, .. } => clauses
                    .iter()
                    .map(|clause| max_term_bytes(&clause.query))
                    .max()
                    .unwrap_or(0),
                Query::Boost { query, .. } => max_term_bytes(query),
                _ => 0,
            }
        }
        assert!(
            max_term_bytes(&parsed.query) <= MAX_QUERY_LENGTH,
            "public parser must never surface a token beyond the 10,000-byte cap"
        );
    }

    #[test]
    fn oversized_standalone_and_phrase_atoms_lower_to_match_none() {
        let parser = parser();
        // Standalone oversized token: MatchNone.
        let standalone = parse_untruncated(&parser, &oversized_atom());
        assert!(
            standalone.query.is_empty(),
            "oversized standalone atom must lower to Query::Empty, got {:?}",
            standalone.query
        );
        // Oversized member of a quoted phrase: the whole phrase is MatchNone
        // (positions of surviving terms are retained; the atom still cannot
        // match because one required position is unsatisfiable).
        let phrase = parse_untruncated(&parser, &format!("\"cache {}\"", oversized_atom()));
        assert!(
            phrase.query.is_empty(),
            "phrase containing an oversized term must lower to Query::Empty, got {:?}",
            phrase.query
        );
    }

    #[test]
    fn oversized_clauses_keep_boolean_sibling_semantics() {
        let parser = parser();
        // Required conjunction: the parser retains the Empty clause so the
        // scorer shorts the whole conjunction to MatchNone (argus proof).
        let conjunction = parse_untruncated(&parser, &format!("cache AND {}", oversized_atom()));
        let Query::Boolean { clauses, .. } = &conjunction.query else {
            panic!(
                "conjunction must remain a Boolean, got {:?}",
                conjunction.query
            );
        };
        assert!(
            clauses
                .iter()
                .any(|clause| clause.occur == Occur::Must && clause.query.is_empty()),
            "conjunction must retain Must(Empty) for the oversized operand: {clauses:?}"
        );
        assert!(
            clauses
                .iter()
                .any(|clause| clause.occur == Occur::Must && !clause.query.is_empty()),
            "conjunction must retain the matchable sibling: {clauses:?}"
        );

        // Optional disjunction: the oversized operand lowers to an Empty
        // Should clause, which the scorer drops while the matchable sibling
        // determines results (argus proof).
        let disjunction = parse_untruncated(&parser, &format!("cache OR {}", oversized_atom()));
        let Query::Boolean { clauses, .. } = &disjunction.query else {
            panic!(
                "disjunction must remain a Boolean, got {:?}",
                disjunction.query
            );
        };
        assert!(
            clauses
                .iter()
                .any(|clause| clause.occur == Occur::Should && clause.query.is_empty()),
            "disjunction must retain Should(Empty) for the oversized operand: {clauses:?}"
        );

        // Negated oversized operand: the exclusion is vacuous (an oversized
        // token matches nothing), and the parser's all-negative repair
        // inserts the All sibling that preserves complement semantics.
        let negation = parse_untruncated(&parser, &format!("-{}", oversized_atom()));
        let Query::Boolean { clauses, .. } = &negation.query else {
            panic!(
                "repaired negation must be a Boolean, got {:?}",
                negation.query
            );
        };
        assert!(
            clauses
                .iter()
                .any(|clause| matches!(clause.query, Query::All)),
            "negated oversized query must gain the All sibling: {clauses:?}"
        );
        assert!(
            clauses
                .iter()
                .any(|clause| clause.occur == Occur::MustNot && clause.query.is_empty()),
            "negated oversized operand lowers to MustNot(Empty): {clauses:?}"
        );
    }

    // ==== AST canonicalization (bd-quill-duel-ast-dedup-7hsu) ====

    fn term_clause(occur: Occur, field_id: u16, text: &str) -> BooleanClause {
        BooleanClause::new(
            occur,
            Query::Term {
                fields: vec![QueryField::new(field_id, 1.0)],
                text: text.to_owned(),
            },
        )
    }

    fn range_clause(occur: Occur, field_id: u16, low: i64, high: i64) -> BooleanClause {
        BooleanClause::new(
            occur,
            Query::Range {
                field_id,
                lower: Bound::Included(QueryValue::I64(low)),
                upper: Bound::Included(QueryValue::I64(high)),
            },
        )
    }

    #[test]
    fn canonicalize_dedups_must_not_and_preserves_distinct_exclusions() {
        let mut query = Query::Boolean {
            clauses: vec![
                term_clause(Occur::MustNot, 1, "alpha"),
                term_clause(Occur::MustNot, 1, "beta"),
                term_clause(Occur::MustNot, 1, "alpha"),
                term_clause(Occur::Must, 1, "gamma"),
            ],
            operator: None,
        };
        let report = canonicalize_query(&mut query);
        assert_eq!(report.must_not_duplicates_removed, 1);
        let Query::Boolean { clauses, .. } = &query else {
            panic!("stays Boolean")
        };
        let must_not_terms: Vec<_> = clauses
            .iter()
            .filter(|clause| clause.occur == Occur::MustNot)
            .filter_map(|clause| match &clause.query {
                Query::Term { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(must_not_terms, ["alpha", "beta"]);
    }

    #[test]
    fn canonicalize_never_dedups_scoring_clauses_for_oracle_conformance() {
        // THE CAVEAT: (A OR A) scores A twice in the oracle's sum-union, so
        // scoring duplicates MUST survive canonicalization.
        let mut should_query = Query::Boolean {
            clauses: vec![
                term_clause(Occur::Should, 1, "alpha"),
                term_clause(Occur::Should, 1, "alpha"),
                term_clause(Occur::Should, 1, "alpha"),
            ],
            operator: Some(BooleanOperator::Or),
        };
        let report = canonicalize_query(&mut should_query);
        assert_eq!(report.must_not_duplicates_removed, 0);
        assert_eq!(report.filter_duplicates_removed, 0);
        let Query::Boolean { clauses, .. } = &should_query else {
            panic!("stays Boolean")
        };
        assert_eq!(clauses.len(), 3, "every scoring duplicate is retained");

        // Scoring Must (Term) duplicates are retained for the same reason.
        let mut must_query = Query::Boolean {
            clauses: vec![
                term_clause(Occur::Must, 1, "alpha"),
                term_clause(Occur::Must, 1, "alpha"),
            ],
            operator: Some(BooleanOperator::And),
        };
        let _ = canonicalize_query(&mut must_query);
        let Query::Boolean { clauses, .. } = &must_query else {
            panic!("stays Boolean")
        };
        assert_eq!(clauses.len(), 2, "scoring Must duplicates are retained");
    }

    #[test]
    fn canonicalize_retains_all_positive_scorers() {
        let mut query = Query::Boost {
            query: Box::new(Query::Boolean {
                clauses: vec![
                    range_clause(Occur::Must, 7, 10, 99),
                    term_clause(Occur::Must, 1, "alpha"),
                    range_clause(Occur::Must, 7, 10, 99),
                    BooleanClause::new(Occur::Must, Query::All),
                    BooleanClause::new(Occur::Must, Query::All),
                ],
                operator: None,
            }),
            factor: 2.0,
        };
        let report = canonicalize_query(&mut query);
        assert_eq!(report.filter_duplicates_removed, 0);
        let Query::Boost { query, factor } = &query else {
            panic!("stays boosted")
        };
        assert_eq!(factor.to_bits(), 2.0_f32.to_bits());
        let Query::Boolean { clauses, .. } = query.as_ref() else {
            panic!("boosted child stays Boolean")
        };
        assert_eq!(clauses.len(), 5);
        assert_eq!(
            clauses
                .iter()
                .filter(|clause| matches!(clause.query, Query::Range { .. }))
                .count(),
            2,
            "duplicate constant scorers retain both score contributions",
        );
        assert!(
            clauses
                .iter()
                .any(|clause| matches!(&clause.query, Query::Term { text, .. } if text == "alpha"))
        );
        assert_eq!(
            clauses
                .iter()
                .filter(|clause| matches!(clause.query, Query::All))
                .count(),
            2,
            "ancestor boosts can make duplicate All children score independently",
        );
    }

    #[test]
    fn canonicalize_must_not_key_frames_distinct_string_bounds() {
        let first = Query::Range {
            field_id: 7,
            lower: Bound::Included(QueryValue::Str("L".to_owned())),
            upper: Bound::Included(QueryValue::Str("p\0\u{2}q".to_owned())),
        };
        let second = Query::Range {
            field_id: 7,
            lower: Bound::Included(QueryValue::Str("L\0\u{2}p".to_owned())),
            upper: Bound::Included(QueryValue::Str("q".to_owned())),
        };
        assert_ne!(canonical_sort_key(&first), canonical_sort_key(&second));

        let mut query = Query::Boolean {
            clauses: vec![
                BooleanClause::new(Occur::Must, Query::All),
                BooleanClause::new(Occur::MustNot, first),
                BooleanClause::new(Occur::MustNot, second),
            ],
            operator: None,
        };
        let report = canonicalize_query(&mut query);
        assert_eq!(report.must_not_duplicates_removed, 0);
        let Query::Boolean { clauses, .. } = query else {
            panic!("stays Boolean")
        };
        assert_eq!(clauses.len(), 3);
    }

    #[test]
    fn canonicalize_groups_occurrences_without_reordering_scoring_clauses() {
        let mut query = Query::Boolean {
            clauses: vec![
                term_clause(Occur::MustNot, 1, "zzz"),
                term_clause(Occur::Must, 9, "beta"),
                BooleanClause::new(
                    Occur::Should,
                    Query::Boost {
                        query: Box::new(term_clause(Occur::Should, 3, "alpha").query),
                        factor: 16_777_216.0,
                    },
                ),
                term_clause(Occur::Must, 1, "alpha"),
                BooleanClause::new(
                    Occur::Should,
                    Query::Boost {
                        query: Box::new(term_clause(Occur::Should, 3, "gamma").query),
                        factor: 1.0,
                    },
                ),
                BooleanClause::new(
                    Occur::Should,
                    Query::Boost {
                        query: Box::new(term_clause(Occur::Should, 3, "beta").query),
                        factor: 0.5,
                    },
                ),
            ],
            operator: None,
        };
        let report = canonicalize_query(&mut query);
        assert_eq!(report.boolean_levels_sorted, 1);
        let Query::Boolean { clauses, .. } = &query else {
            panic!("stays Boolean")
        };
        let occurs: Vec<_> = clauses.iter().map(|clause| clause.occur).collect();
        assert_eq!(
            occurs,
            [
                Occur::Must,
                Occur::Must,
                Occur::Should,
                Occur::Should,
                Occur::Should,
                Occur::MustNot,
            ]
        );
        let should_factors = clauses
            .iter()
            .filter(|clause| clause.occur == Occur::Should)
            .filter_map(|clause| match &clause.query {
                Query::Boost { factor, .. } => Some(*factor),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            should_factors,
            [16_777_216.0, 1.0, 0.5],
            "score-bearing Should construction order is bit-observable",
        );
    }

    #[test]
    fn canonicalize_recurses_and_is_idempotent() {
        let mut query = Query::Boolean {
            clauses: vec![
                BooleanClause::new(
                    Occur::Must,
                    Query::Boolean {
                        clauses: vec![
                            term_clause(Occur::MustNot, 2, "dup"),
                            term_clause(Occur::MustNot, 2, "dup"),
                            term_clause(Occur::Must, 2, "b"),
                            term_clause(Occur::Must, 2, "a"),
                        ],
                        operator: None,
                    },
                ),
                term_clause(Occur::Must, 1, "outer"),
            ],
            operator: None,
        };
        let first = canonicalize_query(&mut query);
        assert_eq!(first.must_not_duplicates_removed, 1);
        assert!(first.boolean_levels_sorted >= 1);
        let after_first = query.clone();
        let second = canonicalize_query(&mut query);
        assert_eq!(query, after_first, "idempotent");
        assert_eq!(second.must_not_duplicates_removed, 0);
        assert_eq!(second.filter_duplicates_removed, 0);
        assert_eq!(second.boolean_levels_sorted, 0);
    }

    #[test]
    fn canonicalize_sorts_glob_fields_and_preserves_scoring_expansions() {
        let mut glob = Query::Glob {
            field_ids: vec![3, 1, 3, 2, 1],
            pattern: "alp*".to_owned(),
        };
        let report = canonicalize_query(&mut glob);
        assert_eq!(report.glob_fields_canonicalized, 1);
        let Query::Glob { field_ids, .. } = &glob else {
            panic!("stays Glob")
        };
        assert_eq!(field_ids, &[1, 1, 2, 3, 3]);

        let mut term = Query::Term {
            fields: vec![
                QueryField::new(1, 1.0),
                QueryField::new(2, 2.0),
                QueryField::new(1, 1.0),
                QueryField::new(1, 2.0), // same id, different boost: kept
            ],
            text: "alpha".to_owned(),
        };
        let report = canonicalize_query(&mut term);
        assert_eq!(report.duplicate_fields_removed, 0);
        let Query::Term { fields, .. } = &term else {
            panic!("stays Term")
        };
        assert_eq!(fields.len(), 4);
        assert_eq!(fields[0], QueryField::new(1, 1.0));
        assert_eq!(fields[1], QueryField::new(2, 2.0));
        assert_eq!(fields[2], QueryField::new(1, 1.0));
        assert_eq!(fields[3], QueryField::new(1, 2.0));
    }

    #[test]
    fn canonicalize_leaves_parse_fixtures_exact_when_already_canonical() {
        // A query that is already canonical is byte-stable (no report, no
        // reorder), so differential-pinned parser fixtures never drift.
        let mut query = Query::Boolean {
            clauses: vec![
                term_clause(Occur::Must, 1, "alpha"),
                term_clause(Occur::Should, 3, "beta"),
            ],
            operator: Some(BooleanOperator::And),
        };
        let original = query.clone();
        let report = canonicalize_query(&mut query);
        assert_eq!(query, original);
        assert_eq!(report.boolean_levels_sorted, 0);
    }
}
