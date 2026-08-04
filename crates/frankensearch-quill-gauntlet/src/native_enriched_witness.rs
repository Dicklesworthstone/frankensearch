//! Native enriched Quill/Tantivy witness with independent semantic oracles
//! (bd-8nqz.4.1).
//!
//! # What makes this a witness rather than a comparison
//!
//! The oracle here is a COMMITTED FIXTURE EXPECTATION, hand-derived from the
//! corpus text, and never either engine's output. Cross-engine agreement is
//! computed too, but it is a second and strictly weaker check: two engines can
//! be identically wrong, and when they are, agreement reports success. The
//! discriminating proof in this module is a COMMON-MODE mutation — one applied
//! to both observations at once — which agreement passes and the fixture
//! oracle still catches.
//!
//! # Why the expectations are engine-neutral
//!
//! Quill and Tantivy both implement BM25, but not bit-identically: their score
//! floats differ, so a receipt that expected equal scores would be asserting an
//! implementation detail rather than a contract. The independent expectations
//! are therefore restricted to facts any correct BM25 implementation must
//! agree on:
//!
//! - the exact SET of matching documents for a query;
//! - the exact TOTAL match count, independent of `limit` and `offset`;
//! - the exact live document count;
//! - the top-ranked document WHERE AND ONLY WHERE term-frequency separation
//!   makes it unambiguous (a document containing a term three times outranks
//!   one containing it once, in the same corpus, at comparable length).
//!
//! Score bits ARE recorded per engine, because a one-bit score change must
//! fail the receipt — but they are compared against that engine's own
//! observation, never across engines. Conflating "both engines scored this
//! 1.4142" with "this is the contract" is how a witness becomes a tautology.
//!
//! # This receipt cannot authorize the flip
//!
//! [`NativeEnrichedReceiptV1`] carries no authorization field, and
//! [`NativeEnrichedReceiptV1::authorizes_replacement`] is a `const fn`
//! returning `false`. Only the terminal release-gate aggregator may authorize
//! a replacement; this is one partial receipt among several.

use std::fmt::Write as _;

use asupersync::Cx;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::GauntletError;
use crate::artifact::GauntletProducerBuildIdentity;

/// Schema version of the native enriched receipt. Separately versioned from
/// Core Lexical V3, which this layers over and never edits.
pub const NATIVE_ENRICHED_RECEIPT_SCHEMA_VERSION: u32 = 1;

/// Domain separator for the receipt hash.
///
/// Deliberately NOT the enriched-body hash domain and not the Core
/// `CampaignReport` domain: a receipt that hashed under a shared domain could
/// be replayed as a different receipt kind.
const RECEIPT_HASH_DOMAIN: &str = "frankensearch.gauntlet.native-enriched-receipt.v1";

/// Which real engine produced an observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeEngineV1 {
    /// The native Quill engine (`frankensearch_quill::QuillIndex`).
    Quill,
    /// The pinned Tantivy incumbent (`frankensearch_lexical::TantivyIndex`).
    Tantivy,
}

impl NativeEngineV1 {
    /// Stable code for hashing and diagnostics.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::Quill => "quill",
            Self::Tantivy => "tantivy",
        }
    }
}

// ---------------------------------------------------------------------------
// The committed fixture corpus and its independent expectations
// ---------------------------------------------------------------------------

/// The committed corpus. Ordinary prose, chosen so that every expectation
/// below can be derived BY HAND from this text without running any engine.
pub const FIXTURE_CORPUS: &[(&str, &str)] = &[
    // "quill" x3 — the decisive term-frequency separation used by the
    // top-rank expectations.
    ("doc-alpha", "quill quill quill indexes text"),
    // "quill" x1, "lexical" x2.
    ("doc-beta", "lexical lexical quill backend"),
    // "lexical" x1, no "quill" at all.
    ("doc-gamma", "lexical retrieval"),
    // Matches neither probe term: the corpus must contain a document that a
    // correct engine excludes, or "total" would be indistinguishable from
    // "doc_count".
    ("doc-delta", "unrelated prose about weather"),
    // Carries markup in its BODY. Deliberately shares no term with the
    // "quill"/"lexical" probes so the pagination expectations above are
    // untouched; its own probe term is "escaping".
    (MARKUP_DOC_ID, MARKUP_DOC_BODY),
    // Multi-byte content for the UTF-8 window boundary dimension. Its probe
    // term is "unicode"; the surrounding text is deliberately CJK and
    // combining-accent heavy so a byte-indexed snippet window would split a
    // character here.
    (UTF8_DOC_ID, UTF8_DOC_BODY),
];

/// Exact live document count of [`FIXTURE_CORPUS`], asserted rather than
/// derived from `.len()` so a fixture edit has to be deliberate.
pub const FIXTURE_DOC_COUNT: usize = 6;

/// Document id whose body exercises UTF-8 window boundaries.
pub const UTF8_DOC_ID: &str = "doc-utf8";
/// Multi-byte body. Every non-ASCII run is longer than one byte, so a snippet
/// window that counted BYTES without respecting character boundaries would
/// cut inside a scalar value and surface `U+FFFD`.
pub const UTF8_DOC_BODY: &str =
    "unicode 日本語テキスト ünïcödé combining é and 🦀 emoji tail padding padding padding";

/// The query-length cap both engines document.
///
/// Tantivy keeps its `MAX_QUERY_LENGTH` private and truncates silently;
/// Quill exports it and additionally emits a `QueryDiagnosticKind::Truncated`
/// diagnostic. The VALUE agrees at 10,000, and the value is all the
/// independent expectation needs — the divergence in whether truncation is
/// REPORTED is recorded by the witness rather than asserted away.
pub const SHARED_MAX_QUERY_LENGTH: usize = 10_000;

/// One hand-derived expectation. Every field is a claim about what any correct
/// BM25 engine must produce for [`FIXTURE_CORPUS`]; none of it is copied from
/// an engine run.
#[derive(Debug, Clone, Copy)]
pub struct EnrichedExpectationV1 {
    /// Query string handed to both engines verbatim.
    pub query: &'static str,
    /// Requested page size.
    pub limit: usize,
    /// Requested page offset.
    pub offset: usize,
    /// The exact set of documents that must match, in canonical (sorted)
    /// order. Set membership is engine-neutral; page ORDER is not, except
    /// where `unambiguous_top` says so.
    pub matching_docs: &'static [&'static str],
    /// Exact total matches, independent of `limit` and `offset`.
    pub total: usize,
    /// The document that must rank first, when term-frequency separation
    /// makes that unambiguous for any correct BM25. `None` where the corpus
    /// does not separate the candidates strongly enough to claim an order
    /// without asserting one engine's length-normalization constants.
    pub unambiguous_top: Option<&'static str>,
}

/// The committed expectation table.
pub const FIXTURE_EXPECTATIONS: &[EnrichedExpectationV1] = &[
    // "quill": doc-alpha has it three times, doc-beta once, in comparable
    // document lengths. Every BM25 puts alpha first.
    EnrichedExpectationV1 {
        query: "quill",
        limit: 10,
        offset: 0,
        matching_docs: &["doc-alpha", "doc-beta"],
        total: 2,
        unambiguous_top: Some("doc-alpha"),
    },
    // "lexical": beta has it twice, gamma once, alpha never. beta first.
    EnrichedExpectationV1 {
        query: "lexical",
        limit: 10,
        offset: 0,
        matching_docs: &["doc-beta", "doc-gamma"],
        total: 2,
        unambiguous_top: Some("doc-beta"),
    },
    // Pagination: the SAME query with limit 1 must report the same TOTAL.
    // A total inferred from the page length would report 1 here, which is
    // exactly the defect this row exists to catch.
    EnrichedExpectationV1 {
        query: "quill",
        limit: 1,
        offset: 0,
        matching_docs: &["doc-alpha", "doc-beta"],
        total: 2,
        unambiguous_top: Some("doc-alpha"),
    },
    // Offset past the end: an empty page, still with the true total.
    EnrichedExpectationV1 {
        query: "quill",
        limit: 10,
        offset: 5,
        matching_docs: &["doc-alpha", "doc-beta"],
        total: 2,
        unambiguous_top: None,
    },
    // No-match: a term present in no document. Empty page, zero total, and a
    // live doc_count that is still the full corpus.
    EnrichedExpectationV1 {
        query: "absent",
        limit: 10,
        offset: 0,
        matching_docs: &[],
        total: 0,
        unambiguous_top: None,
    },
];

// ---------------------------------------------------------------------------
// Enrichment: snippets, query classification, metadata (slice 2)
// ---------------------------------------------------------------------------

/// Semantic state of a hit's stored metadata.
///
/// The bead requires `None`, empty and value to stay distinguishable. They are
/// three variants here rather than an `Option<Value>` because the two engines
/// legitimately REPRESENT "no metadata" differently — one may answer `None`,
/// the other `Some({})` — and collapsing that into a single `Option` would
/// either force a false expectation or hide a real divergence. The oracle
/// adjudicates the SEMANTIC state; the raw representation is recorded beside
/// it for the divergence census.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum MetadataStateV1 {
    /// The engine returned no metadata object at all.
    Absent,
    /// The engine returned a metadata object with no entries.
    EmptyObject,
    /// The engine returned metadata, rendered as canonical sorted `k=v` pairs
    /// so two engines' map orderings cannot masquerade as a difference.
    Entries {
        /// Sorted `key=value` pairs.
        pairs: Vec<String>,
    },
}

impl MetadataStateV1 {
    /// Classify a raw metadata payload into its semantic state.
    #[must_use]
    pub fn classify(raw: Option<&serde_json::Value>) -> Self {
        match raw {
            None | Some(serde_json::Value::Null) => Self::Absent,
            Some(serde_json::Value::Object(map)) if map.is_empty() => Self::EmptyObject,
            Some(serde_json::Value::Object(map)) => {
                let mut pairs: Vec<String> = map
                    .iter()
                    .map(|(key, value)| match value {
                        serde_json::Value::String(text) => format!("{key}={text}"),
                        other => format!("{key}={other}"),
                    })
                    .collect();
                pairs.sort_unstable();
                Self::Entries { pairs }
            }
            Some(other) => Self::Entries {
                pairs: vec![format!("<non-object>={other}")],
            },
        }
    }

    /// Whether any metadata content is present.
    #[must_use]
    pub const fn is_present(&self) -> bool {
        matches!(self, Self::Entries { .. })
    }
}

/// One enriched hit, normalized across the two engines.
///
/// `query_type_code` is a STRING, deliberately. `QueryExplanation` is defined
/// independently in `frankensearch-lexical` and `frankensearch-quill`: two
/// distinct types that happen to share a name and, today, a variant set. They
/// cannot be compared by type, and comparing the two engines' codes to EACH
/// OTHER would be the cross-engine-agreement oracle this bead rejects — if
/// both crates drift the same way, agreement passes. Each engine's code is
/// therefore normalized here and adjudicated against a hand-derived expected
/// classification instead.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeEnrichedHitV1 {
    /// Stable external document identifier.
    pub doc_id: String,
    /// Zero-based rank in the returned page.
    pub rank: usize,
    /// Rendered snippet. `None` stays distinct from `Some("")`.
    pub snippet: Option<String>,
    /// Engine's own query classification, normalized to its `Display` code.
    pub query_type_code: String,
    /// Semantic metadata state.
    pub metadata: MetadataStateV1,
}

/// One engine's enriched answer to one enrichment expectation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeEnrichedObservationV1 {
    /// Engine that produced this observation.
    pub engine: NativeEngineV1,
    /// Query, echoed so a serialized receipt is self-describing.
    pub query: String,
    /// Highlight markup the caller configured, echoed so a receipt records
    /// which tag contract was exercised.
    pub highlight_prefix: String,
    /// Closing highlight markup.
    pub highlight_postfix: String,
    /// Page-ordered enriched hits.
    pub hits: Vec<NativeEnrichedHitV1>,
}

/// A hand-derived expectation about the ENRICHED surface.
#[derive(Debug, Clone, Copy)]
pub struct EnrichmentExpectationV1 {
    /// Query handed to both engines verbatim.
    pub query: &'static str,
    /// Page size.
    pub limit: usize,
    /// Highlight markup to configure. Trusted caller-supplied tags.
    pub highlight_prefix: &'static str,
    /// Closing highlight markup.
    pub highlight_postfix: &'static str,
    /// Classification any reader derives from the query TEXT — a quoted
    /// string is a phrase, one bare word is simple, several are boolean.
    /// Derived from the query, never copied from an engine.
    pub expected_query_type_code: &'static str,
    /// Document whose enriched hit the remaining fields describe.
    pub subject_doc: &'static str,
    /// Substring the snippet must contain, wrapped in the configured tags,
    /// when the engine renders a snippet at all.
    pub highlighted_term: &'static str,
    /// Metadata content the subject document was indexed with, as sorted
    /// `k=v` pairs. Empty means the document carries no metadata.
    pub expected_metadata_pairs: &'static [&'static str],
}

/// Documents carrying metadata, keyed by document id. Absent ids are indexed
/// with no metadata at all, which is how the `Absent`/`EmptyObject` states are
/// reached without a second corpus.
pub const FIXTURE_METADATA: &[(&str, &[(&str, &str)])] =
    &[("doc-alpha", &[("kind", "primary"), ("lang", "en")])];

/// Content deliberately containing markup, so escaping can be adjudicated.
///
/// A snippet engine that echoed this verbatim would emit live markup from
/// UNTRUSTED document text, which is a different thing entirely from the
/// TRUSTED highlight tags the caller configured.
pub const MARKUP_DOC_ID: &str = "doc-markup";
/// The markup-bearing document body. Its probe term is "escaping", chosen so
/// this document does not disturb the "quill"/"lexical" pagination rows.
pub const MARKUP_DOC_BODY: &str = "escaping <script>alert(1)</script> payload";

/// The committed enrichment expectation table.
pub const FIXTURE_ENRICHMENT_EXPECTATIONS: &[EnrichmentExpectationV1] = &[
    // Default tags, single bare term -> "simple".
    EnrichmentExpectationV1 {
        query: "quill",
        limit: 10,
        highlight_prefix: "<b>",
        highlight_postfix: "</b>",
        expected_query_type_code: "simple",
        subject_doc: "doc-alpha",
        highlighted_term: "quill",
        expected_metadata_pairs: &["kind=primary", "lang=en"],
    },
    // CUSTOM tags: the receipt must prove the configured markup is what gets
    // rendered, not a hard-coded <b>.
    EnrichmentExpectationV1 {
        query: "quill",
        limit: 10,
        highlight_prefix: "[[",
        highlight_postfix: "]]",
        expected_query_type_code: "simple",
        subject_doc: "doc-alpha",
        highlighted_term: "quill",
        expected_metadata_pairs: &["kind=primary", "lang=en"],
    },
    // Two bare terms -> "boolean". doc-beta carries NO metadata, which is how
    // the absent/empty state gets exercised.
    EnrichmentExpectationV1 {
        query: "lexical backend",
        limit: 10,
        highlight_prefix: "<b>",
        highlight_postfix: "</b>",
        expected_query_type_code: "boolean",
        subject_doc: "doc-beta",
        highlighted_term: "lexical",
        expected_metadata_pairs: &[],
    },
    // ESCAPING: the subject document's BODY contains markup. The configured
    // tags are trusted and must render; the document's own <script> must not
    // survive as live markup. Both halves are adjudicated, because a check
    // for the tags alone would pass an engine that echoed the body verbatim.
    EnrichmentExpectationV1 {
        query: "escaping",
        limit: 10,
        highlight_prefix: "<b>",
        highlight_postfix: "</b>",
        expected_query_type_code: "simple",
        subject_doc: MARKUP_DOC_ID,
        highlighted_term: "escaping",
        expected_metadata_pairs: &[],
    },
    // UTF-8 window boundaries. The snippet for this document is additionally
    // adjudicated by `adjudicate_utf8_window`.
    EnrichmentExpectationV1 {
        query: "unicode",
        limit: 10,
        highlight_prefix: "<b>",
        highlight_postfix: "</b>",
        expected_query_type_code: "simple",
        subject_doc: UTF8_DOC_ID,
        highlighted_term: "unicode",
        expected_metadata_pairs: &[],
    },
];

/// The multi-byte token that must survive a snippet window intact.
pub const UTF8_INTACT_TOKEN: &str = "日本語テキスト";

/// Adjudicate one ENRICHED observation against its committed expectation.
///
/// Like [`adjudicate`], this never consults the other engine.
#[must_use]
pub fn adjudicate_enrichment(
    expectation: &EnrichmentExpectationV1,
    observation: &NativeEnrichedObservationV1,
) -> NativeVerdictV1 {
    let mut failures = Vec::new();

    let subject = observation
        .hits
        .iter()
        .find(|hit| hit.doc_id == expectation.subject_doc);
    let Some(subject) = subject else {
        failures.push(format!(
            "subject document {} absent from the enriched page",
            expectation.subject_doc
        ));
        return NativeVerdictV1 {
            engine: observation.engine,
            query: observation.query.clone(),
            offset: 0,
            oracle_failures: failures,
        };
    };

    // Query classification, against the hand-derived expectation.
    if subject.query_type_code != expectation.expected_query_type_code {
        failures.push(format!(
            "query_type {} != expected {}",
            subject.query_type_code, expectation.expected_query_type_code
        ));
    }

    // Metadata semantics.
    let expected_metadata_present = !expectation.expected_metadata_pairs.is_empty();
    match (&subject.metadata, expected_metadata_present) {
        (MetadataStateV1::Entries { pairs }, true) => {
            let expected: Vec<String> = expectation
                .expected_metadata_pairs
                .iter()
                .map(|pair| (*pair).to_owned())
                .collect();
            if *pairs != expected {
                failures.push(format!("metadata {pairs:?} != expected {expected:?}"));
            }
        }
        (MetadataStateV1::Entries { pairs }, false) => {
            failures.push(format!(
                "metadata {pairs:?} present for a document indexed without any"
            ));
        }
        // Absent and EmptyObject are both legal representations of "indexed
        // without metadata"; the receipt records WHICH, and the census reads
        // it. Claiming one is correct would assert an engine's internal
        // choice as a contract.
        (MetadataStateV1::Absent | MetadataStateV1::EmptyObject, false) => {}
        (state, true) => {
            failures.push(format!(
                "metadata {state:?} for a document indexed with {:?}",
                expectation.expected_metadata_pairs
            ));
        }
    }

    // Snippet contract. `None` is a legal answer (a schema may not store
    // source text) and is recorded; when a snippet IS rendered, it must use
    // the configured markup and must not leak document markup.
    if let Some(snippet) = subject.snippet.as_deref() {
        let highlighted = format!(
            "{}{}{}",
            observation.highlight_prefix,
            expectation.highlighted_term,
            observation.highlight_postfix
        );
        if !snippet.contains(&highlighted) {
            failures.push(format!(
                "snippet does not render {highlighted:?} with the configured tags"
            ));
        }
        // The document's own markup must not survive as live markup. This is
        // asserted separately from the tag check above precisely because a
        // single "contains <b>" assertion would pass an engine that echoed
        // untrusted document markup verbatim.
        // The assertion is SEMANTIC — untrusted markup must not survive as
        // live markup — rather than a claim about how that is achieved.
        // Escaping and stripping are both safe, and pinning one would assert
        // an engine's rendering choice as a contract. Observed today: Quill
        // escapes, emitting `&lt;script&gt;`, beside live `<b>` tags.
        if snippet.contains("<script>") {
            failures.push(
                "snippet echoed untrusted document markup (<script>) as live markup".to_owned(),
            );
        }
    }

    NativeVerdictV1 {
        engine: observation.engine,
        query: observation.query.clone(),
        offset: 0,
        oracle_failures: failures,
    }
}

/// Adjudicate the UTF-8 window-boundary contract for one enriched snippet.
///
/// The property is engine-neutral and independently derivable: a snippet is a
/// window over the document's text, so it must consist of WHOLE Unicode
/// scalar values. A byte-indexed window that cut mid-character would surface
/// `U+FFFD`, and one that silently dropped the partial character would lose
/// text the document contains.
///
/// Rust's `String` type guarantees the bytes are valid UTF-8, so validity
/// alone proves nothing here — the assertion is about REPLACEMENT characters
/// and about the multi-byte token surviving intact.
#[must_use]
pub fn adjudicate_utf8_window(
    observation: &NativeEnrichedObservationV1,
    subject_doc: &str,
    intact_multibyte_token: &str,
) -> NativeVerdictV1 {
    let mut failures = Vec::new();
    match observation
        .hits
        .iter()
        .find(|hit| hit.doc_id == subject_doc)
        .and_then(|hit| hit.snippet.as_deref())
    {
        Some(snippet) => {
            if snippet.contains('\u{FFFD}') {
                failures.push(
                    "snippet contains U+FFFD, so the window cut inside a scalar value".to_owned(),
                );
            }
            // A window that ends mid-token would drop part of the multi-byte
            // run; requiring the whole token proves the boundary was respected
            // rather than merely that the result re-encoded cleanly.
            if !snippet.contains(intact_multibyte_token) {
                failures.push(format!(
                    "snippet does not carry {intact_multibyte_token:?} intact: {snippet:?}"
                ));
            }
        }
        None => failures.push(format!("no snippet rendered for {subject_doc}")),
    }
    NativeVerdictV1 {
        engine: observation.engine,
        query: observation.query.clone(),
        offset: 0,
        oracle_failures: failures,
    }
}

/// Adjudicate deterministic long-query truncation.
///
/// The independent property, derivable from the documented cap alone: a query
/// longer than [`SHARED_MAX_QUERY_LENGTH`] must behave EXACTLY like its
/// first-`SHARED_MAX_QUERY_LENGTH`-character prefix. That is checkable without
/// knowing either engine's internals, and it is strictly stronger than "does
/// not error" — an engine that truncated at a byte offset, or truncated
/// nondeterministically, or rejected the query outright, all fail it.
///
/// `over_length` and `prefix` must be observations of the same corpus through
/// the same engine.
#[must_use]
pub fn adjudicate_truncation_determinism(
    over_length: &NativeObservationV1,
    prefix: &NativeObservationV1,
) -> NativeVerdictV1 {
    let mut failures = Vec::new();
    if over_length.engine != prefix.engine {
        failures.push("truncation determinism compares one engine against itself".to_owned());
    }
    if over_length.page_doc_ids != prefix.page_doc_ids {
        failures.push(format!(
            "over-length page {:?} != truncated-prefix page {:?}",
            over_length.page_doc_ids, prefix.page_doc_ids
        ));
    }
    if over_length.total != prefix.total {
        failures.push(format!(
            "over-length total {} != truncated-prefix total {}",
            over_length.total, prefix.total
        ));
    }
    if over_length.page_score_bits != prefix.page_score_bits {
        failures.push(
            "over-length scores differ from the truncated prefix bit-for-bit; truncation is not \
             deterministic"
                .to_owned(),
        );
    }
    NativeVerdictV1 {
        engine: over_length.engine,
        query: "<over-length>".to_owned(),
        offset: 0,
        oracle_failures: failures,
    }
}

/// Build the over-length probe query, the prefix it must behave like, and the
/// document that must be EXCLUDED because its term falls beyond the cut.
///
/// # Why the terms are placed rather than appended
///
/// A probe made only of non-matching padding proves determinism but not the
/// CUT POINT: an engine that truncated at 5,000 characters would pass it,
/// because the one matching term sat at the head and survived any prefix.
/// This probe therefore straddles the boundary:
///
/// - `"lexical"` sits JUST INSIDE the first [`SHARED_MAX_QUERY_LENGTH`]
///   characters, so an engine cutting early would drop it and change the page;
/// - `"escaping"` sits JUST BEYOND it, so an engine cutting late (or not at
///   all) would pick up `doc-markup` and change the page the other way.
///
/// Both failure directions are therefore observable, which is what makes this
/// a boundary test rather than a determinism test.
#[must_use]
pub fn truncation_probe_queries() -> (String, String, &'static str) {
    let padding = "zzz ";
    let inside_term = "lexical ";
    let beyond_term = "escaping ";

    // Fill to just short of the cap, leaving room for the inside term.
    let mut long = String::new();
    while long.chars().count() + inside_term.chars().count() < SHARED_MAX_QUERY_LENGTH {
        long.push_str(padding);
    }
    long.push_str(inside_term);
    debug_assert!(long.chars().count() <= SHARED_MAX_QUERY_LENGTH);

    // Pad past the cap, then place the term that must NOT survive.
    while long.chars().count() < SHARED_MAX_QUERY_LENGTH + 8 {
        long.push_str(padding);
    }
    long.push_str(beyond_term);

    let prefix: String = long.chars().take(SHARED_MAX_QUERY_LENGTH).collect();
    (long, prefix, MARKUP_DOC_ID)
}

/// Whether two engines agree on the enriched facts.
///
/// Telemetry only, exactly like [`engines_agree`]: snippets and metadata
/// representations legitimately differ, so this compares only the fields both
/// engines are contractually obliged to share.
#[must_use]
pub fn enriched_engines_agree(
    left: &NativeEnrichedObservationV1,
    right: &NativeEnrichedObservationV1,
) -> bool {
    let key = |observation: &NativeEnrichedObservationV1| {
        observation
            .hits
            .iter()
            .map(|hit| (hit.doc_id.clone(), hit.query_type_code.clone()))
            .collect::<Vec<_>>()
    };
    key(left) == key(right)
}

// ---------------------------------------------------------------------------
// Typed query-capability refusal and its positionful control (slice 4)
// ---------------------------------------------------------------------------

/// Diagnostic name of the witness's own positionless schema.
///
/// Deliberately distinct from the QG positionless schema in `engine.rs`: the
/// receipt names the schema it actually exercised, and two harnesses sharing
/// one schema name would let a refusal observed under one be replayed as
/// evidence for the other.
pub const WITNESS_POSITIONLESS_SCHEMA_NAME: &str = "frankensearch-witness-no-positions-v1";

/// The witness's positionless schema: field-for-field the shipping default,
/// with `positions` turned OFF on both text fields and nothing else changed.
///
/// Holding every other axis fixed is what makes the refusal attributable. A
/// schema that also dropped a field or changed an analyzer would produce a
/// refusal that could be explained by something other than the missing
/// capability.
const WITNESS_POSITIONLESS_FIELDS: [frankensearch_quill::FieldDescriptor; 5] = [
    frankensearch_quill::FieldDescriptor {
        id: 0,
        name: "id",
        kind: frankensearch_quill::FieldKind::Keyword,
        stored: true,
    },
    frankensearch_quill::FieldDescriptor {
        id: 1,
        name: "content",
        kind: frankensearch_quill::FieldKind::Text {
            analyzer: frankensearch_quill::Analyzer::FrankensearchDefault,
            positions: false,
        },
        stored: true,
    },
    frankensearch_quill::FieldDescriptor {
        id: 2,
        name: "title",
        kind: frankensearch_quill::FieldKind::Text {
            analyzer: frankensearch_quill::Analyzer::FrankensearchDefault,
            positions: false,
        },
        stored: true,
    },
    frankensearch_quill::FieldDescriptor {
        id: 3,
        name: "metadata_json",
        kind: frankensearch_quill::FieldKind::StoredOnly,
        stored: true,
    },
    frankensearch_quill::FieldDescriptor {
        id: 4,
        name: "ord",
        kind: frankensearch_quill::FieldKind::U64 {
            indexed: false,
            fast: true,
        },
        stored: true,
    },
];

/// The witness's positionless schema descriptor.
pub const WITNESS_POSITIONLESS_SCHEMA: frankensearch_quill::SchemaDescriptor =
    frankensearch_quill::SchemaDescriptor {
        name: WITNESS_POSITIONLESS_SCHEMA_NAME,
        fields: &WITNESS_POSITIONLESS_FIELDS,
    };

/// Longest engine diagnostic the witness will carry.
///
/// An unmapped engine error is recorded, not swallowed, but a receipt must not
/// become an unbounded channel for engine-internal text.
const MAX_DIAGNOSTIC_CHARS: usize = 200;

/// The witness's OWN stable error classification.
///
/// This is deliberately a separate namespace from the engine's error rendering.
/// The engine's `Display` string is an implementation detail that may be
/// reworded; a receipt that pinned it would either break on a typo fix or, far
/// worse, silently accept a reworded error as a different one. The class and
/// code below are the witness's contract, mapped FROM the engine's typed error,
/// and they change only by deliberate edit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnrichedErrorClassV1 {
    /// A position-dependent operator named a field indexed without positions.
    QueryCapabilityPositionsRequired,
}

impl EnrichedErrorClassV1 {
    /// Stable dotted class name.
    #[must_use]
    pub const fn class_code(self) -> &'static str {
        match self {
            Self::QueryCapabilityPositionsRequired => {
                "enriched.query_capability.positions_required"
            }
        }
    }

    /// Stable short code, pinned by test so a rename must be deliberate.
    #[must_use]
    pub const fn stable_code(self) -> &'static str {
        match self {
            Self::QueryCapabilityPositionsRequired => "FSNE-CAP-0001",
        }
    }
}

/// A typed capability refusal, normalized into the witness's own vocabulary.
///
/// Every field is carried separately rather than as one rendered message,
/// because the acceptance names each of them: a refusal that got the schema
/// right and the field wrong is a different defect from one that got the
/// operator wrong, and a single string cannot distinguish them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnrichedCapabilityRefusalV1 {
    /// The witness's classification of the refusal.
    pub class: EnrichedErrorClassV1,
    /// The class's stable short code, serialized so a stored receipt is
    /// readable without this crate.
    pub code: String,
    /// Schema the engine named.
    pub schema: String,
    /// Field the operator named.
    pub field: String,
    /// Operator that required the capability, as its `Display` code.
    pub operator: String,
    /// Capability the operator required, as its `Display` code.
    pub capability: String,
}

/// What one capability probe actually did.
///
/// `UnmappedError` exists so a refusal the witness does NOT understand can
/// never be laundered into the capability class. A bare
/// `QuillIndexError::UnsupportedQuery` is exactly the shape the bead rejects,
/// and it lands here rather than in [`CapabilityProbeOutcomeV1::Refused`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case", deny_unknown_fields)]
pub enum CapabilityProbeOutcomeV1 {
    /// The engine served the query.
    Served {
        /// Matching document ids in canonical (sorted) order.
        ///
        /// Sorted on purpose: this table's claim is MEMBERSHIP. Page order is
        /// adjudicated by [`adjudicate`], and only where term-frequency
        /// separation makes it unambiguous.
        doc_ids: Vec<String>,
        /// Exact total. `None` means the engine returned no usable exact
        /// count despite being asked for one, which is itself a failure.
        total: Option<usize>,
    },
    /// The engine refused with a typed capability error the witness maps.
    Refused(EnrichedCapabilityRefusalV1),
    /// The engine failed in a way the witness does not classify. Recorded with
    /// a bounded rendering so the receipt stays diagnosable without becoming a
    /// conduit for arbitrary engine text.
    UnmappedError {
        /// Bounded, truncation-marked rendering of the engine error.
        rendered: String,
    },
}

/// Bound an engine diagnostic to [`MAX_DIAGNOSTIC_CHARS`] characters.
fn bounded_diagnostic(text: &str) -> String {
    if text.chars().count() <= MAX_DIAGNOSTIC_CHARS {
        return text.to_owned();
    }
    let mut bounded: String = text.chars().take(MAX_DIAGNOSTIC_CHARS).collect();
    bounded.push_str("…[truncated]");
    bounded
}

/// Map a typed Quill failure into the witness's own error vocabulary.
///
/// Only [`frankensearch_quill::index::QuillIndexError::QueryCapability`] maps
/// to a class; everything else is recorded as unmapped. That asymmetry is the
/// point: the acceptance requires the exact capability class, and an engine
/// that degraded the typed error into a generic failure must not still pass.
#[must_use]
pub fn classify_quill_refusal(
    error: &frankensearch_quill::QuillIndexError,
) -> CapabilityProbeOutcomeV1 {
    use frankensearch_quill::query::QueryCapabilityError;

    match error {
        frankensearch_quill::QuillIndexError::QueryCapability(
            QueryCapabilityError::PositionsRequired {
                schema,
                field,
                operator,
                capability,
            },
        ) => {
            let class = EnrichedErrorClassV1::QueryCapabilityPositionsRequired;
            CapabilityProbeOutcomeV1::Refused(EnrichedCapabilityRefusalV1 {
                class,
                code: class.stable_code().to_owned(),
                schema: (*schema).to_owned(),
                field: field.clone(),
                operator: operator.to_string(),
                capability: capability.to_string(),
            })
        }
        other => CapabilityProbeOutcomeV1::UnmappedError {
            rendered: bounded_diagnostic(&other.to_string()),
        },
    }
}

/// Which schema arm a capability row runs against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilitySchemaArmV1 {
    /// The witness's positionless schema.
    Positionless,
    /// The shipping default schema, which indexes positions.
    Positioned,
}

impl CapabilitySchemaArmV1 {
    /// Schema descriptor this arm builds its index from.
    #[must_use]
    pub const fn schema(self) -> frankensearch_quill::SchemaDescriptor {
        match self {
            Self::Positionless => WITNESS_POSITIONLESS_SCHEMA,
            Self::Positioned => frankensearch_quill::DEFAULT_SCHEMA,
        }
    }

    /// Stable code for hashing and diagnostics.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::Positionless => "positionless",
            Self::Positioned => "positioned",
        }
    }
}

/// What a capability row expects, hand-derived from the schema and the corpus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilityOutcomeExpectationV1 {
    /// The engine must refuse, naming exactly these facts.
    Refused {
        /// Schema the refusal must name.
        schema: &'static str,
        /// Field the refusal must name.
        field: &'static str,
        /// Operator `Display` code the refusal must name.
        operator_code: &'static str,
        /// Capability `Display` code the refusal must name.
        capability_code: &'static str,
        /// The witness class the refusal must map to.
        class: EnrichedErrorClassV1,
    },
    /// The engine must serve, returning exactly this match set and total.
    Served {
        /// Matching documents in sorted order.
        docs: &'static [&'static str],
        /// Exact total.
        total: usize,
    },
}

/// One committed capability expectation.
#[derive(Debug, Clone, Copy)]
pub struct CapabilityProbeExpectationV1 {
    /// Stable row label, used in diagnostics and in the row digest.
    pub label: &'static str,
    /// Query handed to the engine verbatim.
    pub query: &'static str,
    /// Schema arm the row runs against.
    pub arm: CapabilitySchemaArmV1,
    /// Hand-derived expected outcome.
    pub outcome: CapabilityOutcomeExpectationV1,
}

/// The committed capability table.
///
/// # Why the served rows are not optional
///
/// A table containing only the refusal row would be passed by an engine that
/// refused EVERY query, or that refused everything on a positionless schema —
/// which is not the contract. Three independent controls close that:
///
/// - a position-INDEPENDENT term query on the SAME positionless index must be
///   served, so the index itself is demonstrably usable;
/// - a single-term quote on the positionless index must be served, because it
///   degenerates to a term match and reads no positions — so the refusal is
///   scoped to genuine position consumption rather than to quoting;
/// - the identical multi-term phrase on the POSITIONED schema must be served
///   and return the right document, so the refusal is attributable to the
///   missing capability and not to a broken query.
///
/// A fourth row makes the positioned control non-vacuous in the other
/// direction: the REVERSED phrase must match nothing. An engine that ignored
/// positions and degenerated the phrase into a conjunction would return
/// `doc-alpha` for it and pass every other row.
pub const FIXTURE_CAPABILITY_EXPECTATIONS: &[CapabilityProbeExpectationV1] = &[
    // THE ACCEPTANCE ROW: a real positionless stored-schema phrase.
    CapabilityProbeExpectationV1 {
        label: "positionless-multi-term-phrase",
        query: "\"indexes text\"",
        arm: CapabilitySchemaArmV1::Positionless,
        outcome: CapabilityOutcomeExpectationV1::Refused {
            schema: WITNESS_POSITIONLESS_SCHEMA_NAME,
            // The default parser expands over `[content, title^2]`, and
            // `content` is the first field lacking positions, so it is the
            // field the refusal must name.
            field: "content",
            operator_code: "phrase",
            capability_code: "positions",
            class: EnrichedErrorClassV1::QueryCapabilityPositionsRequired,
        },
    },
    // Control: a single-term quote reads no positions and stays servable.
    // "indexes" occurs only in doc-alpha.
    CapabilityProbeExpectationV1 {
        label: "positionless-single-term-quote",
        query: "\"indexes\"",
        arm: CapabilitySchemaArmV1::Positionless,
        outcome: CapabilityOutcomeExpectationV1::Served {
            docs: &["doc-alpha"],
            total: 1,
        },
    },
    // Control: the positionless index serves position-independent queries.
    // "quill" occurs in doc-alpha and doc-beta.
    CapabilityProbeExpectationV1 {
        label: "positionless-bare-term",
        query: "quill",
        arm: CapabilitySchemaArmV1::Positionless,
        outcome: CapabilityOutcomeExpectationV1::Served {
            docs: &["doc-alpha", "doc-beta"],
            total: 2,
        },
    },
    // THE POSITIONFUL CONTROL: the same phrase, served, on the schema that
    // provides the capability. doc-alpha reads "quill quill quill indexes
    // text", so "indexes text" is adjacent there and nowhere else.
    CapabilityProbeExpectationV1 {
        label: "positionful-multi-term-phrase",
        query: "\"indexes text\"",
        arm: CapabilitySchemaArmV1::Positioned,
        outcome: CapabilityOutcomeExpectationV1::Served {
            docs: &["doc-alpha"],
            total: 1,
        },
    },
    // The control's own control: reversed, the phrase matches nothing, which
    // is only true if positions are genuinely consumed.
    CapabilityProbeExpectationV1 {
        label: "positionful-reversed-phrase",
        query: "\"text indexes\"",
        arm: CapabilitySchemaArmV1::Positioned,
        outcome: CapabilityOutcomeExpectationV1::Served {
            docs: &[],
            total: 0,
        },
    },
];

/// Digest of the committed capability table.
#[must_use]
pub fn capability_manifest_sha256() -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"frankensearch.gauntlet.native-enriched-capability.v1\0");
    for expectation in FIXTURE_CAPABILITY_EXPECTATIONS {
        let mut row = String::new();
        let _ = write!(
            row,
            "{}|{}|{}|",
            expectation.label,
            expectation.query,
            expectation.arm.code()
        );
        match expectation.outcome {
            CapabilityOutcomeExpectationV1::Refused {
                schema,
                field,
                operator_code,
                capability_code,
                class,
            } => {
                let _ = write!(
                    row,
                    "refused|{schema}|{field}|{operator_code}|{capability_code}|{}|{}",
                    class.class_code(),
                    class.stable_code()
                );
            }
            CapabilityOutcomeExpectationV1::Served { docs, total } => {
                let _ = write!(row, "served|{}|{total}", docs.join(","));
            }
        }
        hasher.update(row.as_bytes());
        hasher.update([0u8]);
    }
    hex_lower(&hasher.finalize())
}

/// Run one capability row against a REAL Quill index.
///
/// Takes an already-built index rather than building one, so the witness's
/// production code needs no bench-only constructor; the two schema arms are
/// constructed by the live suite, which is where the schema-bound index is
/// legitimately available.
///
/// Never returns `Result`: a refusal IS the observable for the acceptance row,
/// so turning it into an error would force the caller to decide what a refusal
/// means, which is the oracle's job.
#[must_use]
pub fn probe_quill_capability(
    cx: &Cx,
    index: &frankensearch_quill::QuillIndex,
    query: &str,
    limit: usize,
) -> CapabilityProbeOutcomeV1 {
    match index.search_paginated(cx, query, limit, 0, true) {
        Ok(result) => {
            let mut doc_ids: Vec<String> = result
                .hits
                .iter()
                .map(|hit| hit.document_id.clone())
                .collect();
            doc_ids.sort();
            CapabilityProbeOutcomeV1::Served {
                doc_ids,
                total: result
                    .total_count
                    .and_then(|total| usize::try_from(total).ok()),
            }
        }
        Err(error) => classify_quill_refusal(&error),
    }
}

/// Adjudicate one capability probe against its committed expectation.
///
/// Independent, like every other oracle here: it consults only the committed
/// row. There is no cross-engine arm to consult even in principle — Tantivy has
/// no typed capability surface, and that difference is a divergence the witness
/// records rather than an equality it could assert.
#[must_use]
pub fn adjudicate_capability_probe(
    expectation: &CapabilityProbeExpectationV1,
    outcome: &CapabilityProbeOutcomeV1,
) -> NativeVerdictV1 {
    let mut failures = Vec::new();

    match (expectation.outcome, outcome) {
        (
            CapabilityOutcomeExpectationV1::Refused {
                schema,
                field,
                operator_code,
                capability_code,
                class,
            },
            CapabilityProbeOutcomeV1::Refused(refusal),
        ) => {
            if refusal.class != class {
                failures.push(format!(
                    "refusal class {:?} != expected {class:?}",
                    refusal.class
                ));
            }
            if refusal.code != class.stable_code() {
                failures.push(format!(
                    "refusal code {} != expected {}",
                    refusal.code,
                    class.stable_code()
                ));
            }
            if refusal.schema != schema {
                failures.push(format!(
                    "refusal names schema {} != expected {schema}",
                    refusal.schema
                ));
            }
            if refusal.field != field {
                failures.push(format!(
                    "refusal names field {} != expected {field}",
                    refusal.field
                ));
            }
            if refusal.operator != operator_code {
                failures.push(format!(
                    "refusal names operator {} != expected {operator_code}",
                    refusal.operator
                ));
            }
            if refusal.capability != capability_code {
                failures.push(format!(
                    "refusal names capability {} != expected {capability_code}",
                    refusal.capability
                ));
            }
        }
        (
            CapabilityOutcomeExpectationV1::Refused { .. },
            CapabilityProbeOutcomeV1::Served { .. },
        ) => {
            failures.push(
                "the engine served a query that requires a capability its schema lacks".to_owned(),
            );
        }
        (
            CapabilityOutcomeExpectationV1::Refused { .. },
            CapabilityProbeOutcomeV1::UnmappedError { rendered },
        ) => {
            // The bead's words: never a bare "unsupported query". A refusal
            // the witness cannot classify does NOT satisfy a typed-refusal
            // expectation, however plausible its text.
            failures.push(format!(
                "the engine refused with an unclassified error instead of the typed capability \
                 class: {rendered}"
            ));
        }
        (
            CapabilityOutcomeExpectationV1::Served { docs, total },
            CapabilityProbeOutcomeV1::Served {
                doc_ids,
                total: observed_total,
            },
        ) => {
            let expected: Vec<String> = docs.iter().map(|doc| (*doc).to_owned()).collect();
            if *doc_ids != expected {
                failures.push(format!("served {doc_ids:?} != expected {expected:?}"));
            }
            match observed_total {
                Some(observed) if *observed == total => {}
                Some(observed) => {
                    failures.push(format!("served total {observed} != expected {total}"));
                }
                None => failures.push(
                    "no exact total was returned although an exact count was requested".to_owned(),
                ),
            }
        }
        (
            CapabilityOutcomeExpectationV1::Served { .. },
            CapabilityProbeOutcomeV1::Refused(refusal),
        ) => {
            // The row that stops a blanket-refusal engine from passing the
            // acceptance row vacuously.
            failures.push(format!(
                "the engine refused a servable query as {} ({})",
                refusal.class.class_code(),
                refusal.code
            ));
        }
        (
            CapabilityOutcomeExpectationV1::Served { .. },
            CapabilityProbeOutcomeV1::UnmappedError { rendered },
        ) => {
            failures.push(format!("the engine failed a servable query: {rendered}"));
        }
    }

    NativeVerdictV1 {
        engine: NativeEngineV1::Quill,
        query: expectation.query.to_owned(),
        offset: 0,
        oracle_failures: failures,
    }
}

// ---------------------------------------------------------------------------
// Engine identity and accepted-candidate binding (slice 5)
// ---------------------------------------------------------------------------

/// How one engine expresses the identity of the schema it indexed under.
///
/// An enum rather than two nullable strings, because the two engines genuinely
/// differ here and the witness records the difference instead of inventing
/// symmetry: Quill compiles a schema DESCRIPTOR with a canonical 64-bit
/// identity, and the Tantivy adapter exposes no equivalent surface. Giving
/// Tantivy a fabricated schema name would put a hand-written string where an
/// independently derived identity is claimed to be.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "surface", rename_all = "snake_case", deny_unknown_fields)]
pub enum NativeSchemaIdentityV1 {
    /// Quill's compile-time schema descriptor: its diagnostic name and the
    /// canonical schema id computed over its encoding.
    QuillDescriptor {
        /// Descriptor name.
        name: String,
        /// Canonical schema id, lowercase hex.
        schema_id_hex: String,
    },
    /// The engine exposes no schema-descriptor identity. Recorded as an
    /// explicit absence so a reader cannot mistake it for one that was checked.
    NoDescriptorSurface,
}

/// Independently derived identity of one engine arm.
///
/// Every field is read from the ENGINE crate's own exports — never copied into
/// this module as a literal — so an engine that changed its backend code,
/// crate version, or schema encoding changes this identity without anyone
/// editing the witness.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeEngineIdentityV1 {
    /// Which arm this identity describes.
    pub engine: NativeEngineV1,
    /// Backend code the engine crate publishes for itself.
    pub backend_code: String,
    /// Exact crate version compiled into this build.
    pub crate_version: String,
    /// Schema identity, or the explicit absence of one.
    pub schema: NativeSchemaIdentityV1,
}

/// Derive the shipping-default Quill arm identity from the Quill crate itself.
///
/// # Errors
///
/// Returns [`GauntletError::InvalidContract`] when the shipping schema cannot
/// be canonically encoded, which would mean the schema identity is
/// unobtainable rather than merely different.
pub fn derive_quill_engine_identity() -> Result<NativeEngineIdentityV1, GauntletError> {
    let schema_id = frankensearch_quill::DEFAULT_SCHEMA
        .schema_id()
        .map_err(|error| GauntletError::InvalidContract {
            reason: format!("shipping Quill schema has no canonical identity: {error}"),
        })?;
    Ok(NativeEngineIdentityV1 {
        engine: NativeEngineV1::Quill,
        backend_code: frankensearch_quill::QUILL_LEXICAL_BACKEND.to_owned(),
        crate_version: frankensearch_quill::FRANKENSEARCH_QUILL_CRATE_VERSION.to_owned(),
        schema: NativeSchemaIdentityV1::QuillDescriptor {
            name: frankensearch_quill::DEFAULT_SCHEMA.name.to_owned(),
            schema_id_hex: format!("{schema_id:016x}"),
        },
    })
}

/// Derive the Tantivy arm identity from the lexical crate itself.
#[cfg(feature = "tantivy-oracle")]
#[must_use]
pub fn derive_tantivy_engine_identity() -> NativeEngineIdentityV1 {
    NativeEngineIdentityV1 {
        engine: NativeEngineV1::Tantivy,
        // Not the Quill backend code, and deliberately not a fabricated
        // schema name: see `NoDescriptorSurface`.
        backend_code: "tantivy".to_owned(),
        crate_version: frankensearch_lexical::FRANKENSEARCH_LEXICAL_CRATE_VERSION.to_owned(),
        schema: NativeSchemaIdentityV1::NoDescriptorSurface,
    }
}

impl NativeEngineIdentityV1 {
    /// Reject a CASS engine identity.
    ///
    /// The semantic contract is checked alongside the identity because it is
    /// the one CASS marker BOTH arms can answer: Quill's CASS profile is
    /// visible in its schema descriptor, while the Tantivy CASS index exposes
    /// its own schema hash rather than a descriptor. Checking the contract
    /// covers both arms uniformly; the Quill-specific checks below are the
    /// stronger, engine-local evidence layered on top.
    ///
    /// The schema NAME and the schema ID are checked separately on purpose. A
    /// renamed CASS schema would slip a name check while keeping its encoding,
    /// and content identity is what actually decides whether the CASS profile
    /// was exercised.
    ///
    /// # Errors
    ///
    /// Returns [`GauntletError::InvalidContract`] naming the marker that
    /// matched.
    pub fn reject_cass_identity(
        &self,
        contract: &crate::runner::SemanticContract,
    ) -> Result<(), GauntletError> {
        if *contract == crate::runner::SemanticContract::cass() {
            return Err(GauntletError::InvalidContract {
                reason: "engine identity carries the CASS semantic contract".to_owned(),
            });
        }
        for (label, value) in [
            ("backend code", self.backend_code.as_str()),
            ("crate version", self.crate_version.as_str()),
        ] {
            if value.to_ascii_lowercase().contains("cass") {
                return Err(GauntletError::InvalidContract {
                    reason: format!("engine identity {label} names a CASS profile: {value}"),
                });
            }
        }
        if let NativeSchemaIdentityV1::QuillDescriptor {
            name,
            schema_id_hex,
        } = &self.schema
        {
            if name == frankensearch_quill::CASS_SEMANTIC_SCHEMA.name {
                return Err(GauntletError::InvalidContract {
                    reason: format!("engine identity names the CASS schema {name}"),
                });
            }
            let cass_id = frankensearch_quill::CASS_SEMANTIC_SCHEMA
                .schema_id()
                .map_err(|error| GauntletError::InvalidContract {
                    reason: format!("CASS schema has no canonical identity: {error}"),
                })?;
            if *schema_id_hex == format!("{cass_id:016x}") {
                return Err(GauntletError::InvalidContract {
                    reason: "engine identity carries the CASS schema identity under another name"
                        .to_owned(),
                });
            }
        }
        Ok(())
    }
}

/// Require a shipping semantic contract.
///
/// The accept-set is the one the campaign runner already enforces for typed
/// scalar evidence (`runner.rs`: shipping-default maps to `ScalarShipping`,
/// scalar-G1a to `ScalarG1a`, anything else is refused). Binding to that
/// existing rule rather than inventing a witness-local policy keeps one
/// definition of "shipping" in the harness.
///
/// # Errors
///
/// Returns [`GauntletError::InvalidContract`] for the CASS contract and for
/// any contract outside the accept-set.
pub fn require_shipping_semantic_contract(
    contract: &crate::runner::SemanticContract,
) -> Result<(), GauntletError> {
    if *contract == crate::runner::SemanticContract::shipping_default()
        || *contract == crate::runner::SemanticContract::scalar_g1a()
    {
        return Ok(());
    }
    let reason = if *contract == crate::runner::SemanticContract::cass() {
        "the witness requires a shipping semantic contract, but the CASS contract was supplied"
    } else {
        "the witness requires the shipping-default or scalar-G1a semantic contract"
    };
    Err(GauntletError::InvalidContract {
        reason: reason.to_owned(),
    })
}

/// Whether a value is a canonical lowercase 40-hex Git revision.
fn is_canonical_git_revision(value: &str) -> bool {
    value.len() == 40
        && value.bytes().all(|byte| byte.is_ascii_hexdigit())
        && !value.bytes().any(|byte| byte.is_ascii_uppercase())
}

/// The exact accepted candidate this receipt is evidence about.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcceptedCandidateBindingV1 {
    /// Canonical 40-hex source revision of the accepted candidate.
    pub candidate_source_revision: String,
    /// Contract surface the candidate's own campaign evidence exercised.
    pub contract_mode: crate::campaign_contract::CampaignContractModeV1,
}

impl AcceptedCandidateBindingV1 {
    /// Derive a candidate binding from a campaign report.
    ///
    /// # Why `passed` is necessary and nowhere near sufficient
    ///
    /// `CampaignReport::passed` says a campaign met its own bar. It says
    /// nothing about WHICH contract surface was exercised, and the repository
    /// contains a committed counter-example: the pinned V7 report is
    /// `passed: true` while covering only the rank envelope, carrying no
    /// provenance, and having been produced by a dirty tree. A witness that
    /// accepted a passed report would accept exactly that object as evidence
    /// for a replacement it cannot support.
    ///
    /// Every gate below is therefore independent, and each is proved
    /// load-bearing by repairing the real fixture one axis at a time.
    ///
    /// # Errors
    ///
    /// Returns [`GauntletError::InvalidContract`] naming the first gate the
    /// report fails.
    pub fn from_campaign_report(
        report: &crate::runner::CampaignReport,
    ) -> Result<Self, GauntletError> {
        let reject = |reason: &str| {
            Err(GauntletError::InvalidContract {
                reason: reason.to_owned(),
            })
        };

        if !report.passed {
            return reject("candidate campaign report did not pass");
        }
        match &report.lexical_coverage {
            crate::runner::CampaignLexicalCoverageSummary::CoreLexicalV3 {
                admissible: true,
                ..
            } => {}
            crate::runner::CampaignLexicalCoverageSummary::CoreLexicalV3 {
                admissible: false,
                ..
            } => {
                return reject(
                    "candidate campaign report claims core-v3 coverage that is not admissible",
                );
            }
            crate::runner::CampaignLexicalCoverageSummary::RankEnvelopeOnly => {
                return reject(
                    "candidate campaign report exercised only the rank-envelope contract, which \
                     cannot bind an enriched candidate",
                );
            }
            crate::runner::CampaignLexicalCoverageSummary::LegacyMissing => {
                return reject(
                    "candidate campaign report declares no lexical coverage scope, so passing \
                     says nothing about the enriched contract",
                );
            }
        }
        require_shipping_semantic_contract(&report.semantic_contract)?;
        if report.provenance.is_none() {
            return reject("candidate campaign report carries no immutable provenance");
        }
        if report.producer_build_identity.source_git_dirty {
            return reject("candidate campaign report was produced by a dirty tree");
        }
        if !is_canonical_git_revision(&report.producer_build_identity.source_git_revision) {
            return reject("candidate campaign report names no canonical 40-hex source revision");
        }

        Ok(Self {
            candidate_source_revision: report.producer_build_identity.source_git_revision.clone(),
            contract_mode: crate::campaign_contract::CampaignContractModeV1::CoreLexicalV3,
        })
    }

    /// Require that this receipt's producer IS the bound candidate.
    ///
    /// This is the stale-candidate gate. A receipt naming one candidate while
    /// having been produced by a different binary is describing an engine it
    /// did not run, which is precisely the failure a frozen-candidate receipt
    /// exists to prevent.
    ///
    /// # Errors
    ///
    /// Returns [`GauntletError::ManifestMismatch`] when the producer revision
    /// differs from the bound candidate, and
    /// [`GauntletError::InvalidContract`] when the producer is dirty or the
    /// bound mode is not the full core-v3 surface.
    pub fn require_produced_by_the_candidate(
        &self,
        producer: &GauntletProducerBuildIdentity,
    ) -> Result<(), GauntletError> {
        if self.contract_mode != crate::campaign_contract::CampaignContractModeV1::CoreLexicalV3 {
            return Err(GauntletError::InvalidContract {
                reason: format!(
                    "candidate binding exercises {:?}, not the full core-v3 surface",
                    self.contract_mode
                ),
            });
        }
        if producer.source_git_dirty {
            return Err(GauntletError::InvalidContract {
                reason: "receipt producer provenance is dirty".to_owned(),
            });
        }
        if self.candidate_source_revision != producer.source_git_revision {
            return Err(GauntletError::ManifestMismatch {
                reason: format!(
                    "receipt binds candidate {} but was produced by {}",
                    self.candidate_source_revision, producer.source_git_revision
                ),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Observations
// ---------------------------------------------------------------------------

/// One engine's answer to one expectation, normalized to the facts the
/// independent oracle can adjudicate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeObservationV1 {
    /// Engine that produced this observation.
    pub engine: NativeEngineV1,
    /// Query, echoed so a serialized receipt is self-describing.
    pub query: String,
    /// Requested page size.
    pub limit: usize,
    /// Requested page offset.
    pub offset: usize,
    /// Page-ordered document ids exactly as the engine returned them.
    pub page_doc_ids: Vec<String>,
    /// Exact total the engine reported, from its own counting collector.
    pub total: usize,
    /// Live document count the engine reported.
    pub doc_count: usize,
    /// Per-hit score bits, recorded for one-bit-change detection. Compared
    /// against this engine's own prior observation, NEVER across engines.
    pub page_score_bits: Vec<u32>,
}

/// Verdict of one expectation against one engine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeVerdictV1 {
    /// Engine adjudicated.
    pub engine: NativeEngineV1,
    /// Query adjudicated.
    pub query: String,
    /// Requested page offset, which distinguishes rows sharing a query.
    pub offset: usize,
    /// Independent-oracle failures. Empty means the engine matched the
    /// committed expectation.
    pub oracle_failures: Vec<String>,
}

impl NativeVerdictV1 {
    /// Whether this engine satisfied the independent expectation.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.oracle_failures.is_empty()
    }
}

/// Adjudicate ONE observation against the committed expectation.
///
/// This is the independent oracle. It never consults another engine, which is
/// what lets it catch a common-mode defect that cross-engine agreement cannot.
#[must_use]
pub fn adjudicate(
    expectation: &EnrichedExpectationV1,
    observation: &NativeObservationV1,
) -> NativeVerdictV1 {
    let mut failures = Vec::new();

    if observation.total != expectation.total {
        failures.push(format!(
            "total {} != expected {}",
            observation.total, expectation.total
        ));
    }
    if observation.doc_count != FIXTURE_DOC_COUNT {
        failures.push(format!(
            "doc_count {} != expected {FIXTURE_DOC_COUNT}",
            observation.doc_count
        ));
    }

    // The page must be a subset of the expected match set, in the requested
    // window. A document outside the match set is a false positive no matter
    // how it ranked.
    for doc_id in &observation.page_doc_ids {
        if !expectation.matching_docs.contains(&doc_id.as_str()) {
            failures.push(format!("page contains non-matching document {doc_id}"));
        }
    }

    // Page size: the window the caller asked for, bounded by what is left
    // after the offset. Derived from the EXPECTED total, not from the
    // engine's own reported total, so a wrong total cannot excuse a wrong
    // page length.
    let remaining = expectation.total.saturating_sub(expectation.offset);
    let expected_page_len = remaining.min(expectation.limit);
    if observation.page_doc_ids.len() != expected_page_len {
        failures.push(format!(
            "page length {} != expected {expected_page_len}",
            observation.page_doc_ids.len()
        ));
    }

    if let Some(top) = expectation.unambiguous_top
        && expectation.offset == 0
        && let Some(actual_top) = observation.page_doc_ids.first()
        && actual_top != top
    {
        failures.push(format!("top-ranked {actual_top} != expected {top}"));
    }

    // Score bits must accompany every returned hit: a page that reports hits
    // with no scores cannot be checked for a one-bit change later.
    if observation.page_score_bits.len() != observation.page_doc_ids.len() {
        failures.push(format!(
            "score-bit count {} does not cover {} hits",
            observation.page_score_bits.len(),
            observation.page_doc_ids.len()
        ));
    }

    NativeVerdictV1 {
        engine: observation.engine,
        query: observation.query.clone(),
        offset: observation.offset,
        oracle_failures: failures,
    }
}

/// Whether two engines agree on the engine-neutral facts.
///
/// Recorded on the receipt as telemetry. It is NOT an oracle: see the module
/// header on common-mode failure.
#[must_use]
pub fn engines_agree(left: &NativeObservationV1, right: &NativeObservationV1) -> bool {
    left.page_doc_ids == right.page_doc_ids
        && left.total == right.total
        && left.doc_count == right.doc_count
}

// ---------------------------------------------------------------------------
// The receipt
// ---------------------------------------------------------------------------

/// One partial native enriched receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeEnrichedReceiptV1 {
    /// Schema version of this receipt kind.
    pub schema_version: u32,
    /// Build-sealed provenance of the binary that produced it: git revision
    /// and dirty state, Cargo.lock digest, rustc, target, profile, feature
    /// selection and executable digest. Reused from the existing gauntlet
    /// producer identity rather than re-derived, so a receipt cannot claim a
    /// provenance the rest of the harness would reject.
    pub producer: GauntletProducerBuildIdentity,
    /// Digest of the committed corpus, so a fixture edit invalidates the
    /// receipt instead of silently changing what was witnessed.
    pub corpus_manifest_sha256: String,
    /// Digest of the committed expectation table.
    pub query_manifest_sha256: String,
    /// Digest of the committed capability table.
    pub capability_manifest_sha256: String,
    /// Schema/analyzer axis, bound as the harness's existing semantic
    /// contract rather than as a witness-local restatement of it.
    pub semantic_contract: crate::runner::SemanticContract,
    /// Independently derived identity of every engine arm observed.
    pub engine_identities: Vec<NativeEngineIdentityV1>,
    /// The exact accepted candidate this receipt is evidence about.
    pub candidate: AcceptedCandidateBindingV1,
    /// Every paginated observation, in table order.
    pub observations: Vec<NativeObservationV1>,
    /// Every enriched observation, in table order.
    pub enriched_observations: Vec<NativeEnrichedObservationV1>,
    /// Every capability probe, in table order.
    pub capability_outcomes: Vec<CapabilityProbeRecordV1>,
    /// Every verdict, recomputed at assembly and again at load. A verdict is
    /// a DERIVED field: it is never taken from a caller and never trusted
    /// from a stored body.
    pub verdicts: Vec<NativeVerdictV1>,
    /// Whether both engines were observed. A single-engine receipt is legal
    /// and honest — the Tantivy arm needs the `tantivy-oracle` feature — but
    /// it must say so rather than imply cross-engine coverage.
    pub both_engines_observed: bool,
}

/// One capability probe, labelled by its committed row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapabilityProbeRecordV1 {
    /// Committed row label.
    pub label: String,
    /// What the engine actually did.
    pub outcome: CapabilityProbeOutcomeV1,
}

/// Everything one witness run observed, before it becomes a receipt.
///
/// Deliberately carries NO verdicts: the adjudication is the receipt's job, so
/// a caller cannot hand in observations together with the conclusion it would
/// like drawn from them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeEnrichedRunV1 {
    /// Paginated observations.
    pub observations: Vec<NativeObservationV1>,
    /// Enriched observations.
    pub enriched_observations: Vec<NativeEnrichedObservationV1>,
    /// Capability probes.
    pub capability_outcomes: Vec<CapabilityProbeRecordV1>,
    /// Whether both engines were driven.
    pub both_engines_observed: bool,
}

/// Independently derive the identity of every engine arm this build can drive.
///
/// # Errors
///
/// Propagates a Quill schema-identity failure.
pub fn derive_engine_identities() -> Result<Vec<NativeEngineIdentityV1>, GauntletError> {
    #[allow(unused_mut)]
    let mut identities = vec![derive_quill_engine_identity()?];
    #[cfg(feature = "tantivy-oracle")]
    identities.push(derive_tantivy_engine_identity());
    Ok(identities)
}

/// Recompute every verdict from the committed tables.
///
/// This is the function that makes a stored verdict worthless as a claim: a
/// receipt's verdicts are compared against what THIS binary derives from the
/// same observations, so an edited verdict is a mismatch rather than a fact.
///
/// Coverage is required, not optional. Every committed row must be observed
/// once by every engine the receipt claims to have driven, because a receipt
/// that could omit rows could hide exactly the rows that failed.
///
/// # Errors
///
/// Returns [`GauntletError::InvalidContract`] when an observation names a
/// query outside the committed tables, when a row is observed twice, or when
/// a claimed engine did not observe every row.
pub fn recompute_verdicts(
    run: &NativeEnrichedRunV1,
) -> Result<Vec<NativeVerdictV1>, GauntletError> {
    let invalid = |reason: String| GauntletError::InvalidContract { reason };
    let engines: Vec<NativeEngineV1> = if run.both_engines_observed {
        vec![NativeEngineV1::Quill, NativeEngineV1::Tantivy]
    } else {
        vec![NativeEngineV1::Quill]
    };

    let mut verdicts = Vec::new();

    // Paginated rows.
    for observation in &run.observations {
        let expectation = FIXTURE_EXPECTATIONS
            .iter()
            .find(|row| {
                row.query == observation.query
                    && row.limit == observation.limit
                    && row.offset == observation.offset
            })
            .ok_or_else(|| {
                invalid(format!(
                    "observation of {:?} (limit {}, offset {}) has no committed expectation",
                    observation.query, observation.limit, observation.offset
                ))
            })?;
        verdicts.push(adjudicate(expectation, observation));
    }
    let expected_pages = FIXTURE_EXPECTATIONS.len() * engines.len();
    if run.observations.len() != expected_pages {
        return Err(invalid(format!(
            "receipt carries {} paginated observations, but {} engines x {} committed rows \
             require {expected_pages}",
            run.observations.len(),
            engines.len(),
            FIXTURE_EXPECTATIONS.len()
        )));
    }

    // Enriched rows.
    for observation in &run.enriched_observations {
        let expectation = FIXTURE_ENRICHMENT_EXPECTATIONS
            .iter()
            .find(|row| {
                row.query == observation.query
                    && row.highlight_prefix == observation.highlight_prefix
            })
            .ok_or_else(|| {
                invalid(format!(
                    "enriched observation of {:?} with prefix {:?} has no committed expectation",
                    observation.query, observation.highlight_prefix
                ))
            })?;
        verdicts.push(adjudicate_enrichment(expectation, observation));
    }
    let expected_enriched = FIXTURE_ENRICHMENT_EXPECTATIONS.len() * engines.len();
    if run.enriched_observations.len() != expected_enriched {
        return Err(invalid(format!(
            "receipt carries {} enriched observations but {expected_enriched} are required",
            run.enriched_observations.len()
        )));
    }

    // Capability rows. Quill-only: Tantivy has no typed capability surface,
    // which the witness records as a divergence rather than a gap.
    for record in &run.capability_outcomes {
        let expectation = FIXTURE_CAPABILITY_EXPECTATIONS
            .iter()
            .find(|row| row.label == record.label)
            .ok_or_else(|| {
                invalid(format!(
                    "capability probe {:?} has no committed expectation",
                    record.label
                ))
            })?;
        verdicts.push(adjudicate_capability_probe(expectation, &record.outcome));
    }
    if run.capability_outcomes.len() != FIXTURE_CAPABILITY_EXPECTATIONS.len() {
        return Err(invalid(format!(
            "receipt carries {} capability probes but {} committed rows require all of them",
            run.capability_outcomes.len(),
            FIXTURE_CAPABILITY_EXPECTATIONS.len()
        )));
    }

    Ok(verdicts)
}

impl NativeEnrichedReceiptV1 {
    /// This receipt can NEVER authorize a replacement.
    ///
    /// A `const fn` returning `false` rather than a field, so there is no
    /// serialized value an attacker or a careless edit could flip, and no
    /// deserialized payload that could arrive claiming authorization.
    #[must_use]
    pub const fn authorizes_replacement(&self) -> bool {
        false
    }

    /// Whether every adjudicated engine satisfied the independent oracle.
    #[must_use]
    pub fn all_verdicts_passed(&self) -> bool {
        self.verdicts.iter().all(NativeVerdictV1::passed)
    }

    /// Assemble a receipt from one run, sealing it to THIS binary.
    ///
    /// The verdicts are computed here from the committed tables; the producer
    /// identity, manifests and engine identities are derived, never supplied.
    ///
    /// # Errors
    ///
    /// Propagates producer-identity, coverage and adjudication failures, and
    /// rejects a candidate binding that this binary did not produce.
    pub fn assemble(
        run: &NativeEnrichedRunV1,
        candidate: AcceptedCandidateBindingV1,
    ) -> Result<Self, GauntletError> {
        Self::assemble_with_producer(run, candidate, GauntletProducerBuildIdentity::compiled()?)
    }

    /// Assemble against an explicitly supplied producer identity.
    ///
    /// Crate-visible on purpose. An injected producer is exactly the synthetic
    /// construction the bead rejects as release evidence, and
    /// [`NativeEnrichedReceiptV1::load_canonical`] re-derives the producer from
    /// the running binary, so a receipt built this way cannot survive a load
    /// anywhere but the machine that would have produced it anyway.
    pub(crate) fn assemble_with_producer(
        run: &NativeEnrichedRunV1,
        candidate: AcceptedCandidateBindingV1,
        producer: GauntletProducerBuildIdentity,
    ) -> Result<Self, GauntletError> {
        let identities = derive_engine_identities()?;
        if run.both_engines_observed && identities.len() < 2 {
            return Err(GauntletError::InvalidContract {
                reason: "receipt claims both engines but this build can drive only one".to_owned(),
            });
        }
        let semantic_contract = crate::runner::SemanticContract::shipping_default();
        require_shipping_semantic_contract(&semantic_contract)?;
        for identity in &identities {
            identity.reject_cass_identity(&semantic_contract)?;
        }
        if candidate.candidate_source_revision != producer.source_git_revision {
            return Err(GauntletError::ManifestMismatch {
                reason: format!(
                    "receipt binds candidate {} but was produced by {}",
                    candidate.candidate_source_revision, producer.source_git_revision
                ),
            });
        }
        Ok(Self {
            schema_version: NATIVE_ENRICHED_RECEIPT_SCHEMA_VERSION,
            producer,
            corpus_manifest_sha256: corpus_manifest_sha256(),
            query_manifest_sha256: query_manifest_sha256(),
            capability_manifest_sha256: capability_manifest_sha256(),
            semantic_contract,
            engine_identities: identities,
            candidate,
            verdicts: recompute_verdicts(run)?,
            observations: run.observations.clone(),
            enriched_observations: run.enriched_observations.clone(),
            capability_outcomes: run.capability_outcomes.clone(),
            both_engines_observed: run.both_engines_observed,
        })
    }

    /// The run this receipt records, for re-adjudication.
    fn run(&self) -> NativeEnrichedRunV1 {
        NativeEnrichedRunV1 {
            observations: self.observations.clone(),
            enriched_observations: self.enriched_observations.clone(),
            capability_outcomes: self.capability_outcomes.clone(),
            both_engines_observed: self.both_engines_observed,
        }
    }

    /// Load a stored receipt, recomputing everything it claims.
    ///
    /// # Why the address alone is not verification
    ///
    /// The address IS the hash of the body, so any edit changes it — but an
    /// editor can simply recompute the address, and then the object is
    /// internally consistent and still false. What an editor CANNOT do is make
    /// the stored manifests agree with this build's committed fixtures, the
    /// stored producer agree with this binary's sealed identity, the stored
    /// engine identities agree with the engine crates' own exports, or the
    /// stored verdicts agree with a fresh adjudication of the stored
    /// observations. Those recomputations, not the address, are what makes a
    /// stored body evidence.
    ///
    /// A [`VerifiedNativeEnrichedReceiptV1`] can be produced ONLY here. There
    /// is no constructor that takes a body, which is what makes synthetic
    /// from-body construction unusable as release evidence.
    ///
    /// # Errors
    ///
    /// Returns [`GauntletError::Json`] for a body that is not canonical (an
    /// unknown field is a parse failure), [`GauntletError::ManifestMismatch`]
    /// for an address or provenance that does not replay, and
    /// [`GauntletError::InvalidContract`] for a contract violation.
    pub fn load_canonical(
        bytes: &[u8],
        expected_address: &str,
    ) -> Result<VerifiedNativeEnrichedReceiptV1, GauntletError> {
        let receipt: Self = serde_json::from_slice(bytes)?;
        let mismatch = |reason: String| GauntletError::ManifestMismatch { reason };

        if receipt.schema_version != NATIVE_ENRICHED_RECEIPT_SCHEMA_VERSION {
            return Err(GauntletError::InvalidContract {
                reason: format!(
                    "receipt schema version {} is not {NATIVE_ENRICHED_RECEIPT_SCHEMA_VERSION}",
                    receipt.schema_version
                ),
            });
        }

        // 1. Object address.
        let address = receipt.receipt_hash()?;
        if address != expected_address {
            return Err(mismatch(format!(
                "receipt address {address} does not match the expected {expected_address}"
            )));
        }

        // 2. Fixture manifests, recomputed from the committed tables.
        for (label, stored, recomputed) in [
            (
                "corpus manifest",
                &receipt.corpus_manifest_sha256,
                corpus_manifest_sha256(),
            ),
            (
                "query manifest",
                &receipt.query_manifest_sha256,
                query_manifest_sha256(),
            ),
            (
                "capability manifest",
                &receipt.capability_manifest_sha256,
                capability_manifest_sha256(),
            ),
        ] {
            if *stored != recomputed {
                return Err(mismatch(format!(
                    "{label} {stored} does not match this build's {recomputed}"
                )));
            }
        }

        // 3. Schema/analyzer axis, and the CASS refusal.
        require_shipping_semantic_contract(&receipt.semantic_contract)?;
        for identity in &receipt.engine_identities {
            identity.reject_cass_identity(&receipt.semantic_contract)?;
        }

        // 4. Engine identities, re-derived from the engine crates.
        let derived = derive_engine_identities()?;
        if receipt.engine_identities != derived {
            return Err(mismatch(
                "receipt engine identities do not match the engines this build links".to_owned(),
            ));
        }

        // 5. Producer provenance, axis by axis so a failure names what moved.
        let compiled = GauntletProducerBuildIdentity::compiled()?;
        for (label, stored, current) in [
            (
                "source revision",
                &receipt.producer.source_git_revision,
                &compiled.source_git_revision,
            ),
            (
                "Cargo.lock digest",
                &receipt.producer.cargo_lock_sha256,
                &compiled.cargo_lock_sha256,
            ),
            (
                "toolchain",
                &receipt.producer.rustc_version_verbose_hex,
                &compiled.rustc_version_verbose_hex,
            ),
            (
                "target triple",
                &receipt.producer.target_triple,
                &compiled.target_triple,
            ),
            (
                "cargo profile",
                &receipt.producer.cargo_profile,
                &compiled.cargo_profile,
            ),
            (
                "feature selection",
                &receipt.producer.enabled_features_sha256,
                &compiled.enabled_features_sha256,
            ),
            (
                "executable digest",
                &receipt.producer.executable_sha256,
                &compiled.executable_sha256,
            ),
        ] {
            if stored != current {
                return Err(mismatch(format!(
                    "receipt {label} {stored} was not produced by this binary ({current})"
                )));
            }
        }
        if receipt.producer != compiled {
            return Err(mismatch(
                "receipt producer identity differs from this binary's sealed identity".to_owned(),
            ));
        }

        // 6. The candidate this receipt claims to be evidence about.
        if receipt.candidate.contract_mode
            != crate::campaign_contract::CampaignContractModeV1::CoreLexicalV3
        {
            return Err(GauntletError::InvalidContract {
                reason: format!(
                    "receipt candidate exercises {:?}, not the full core-v3 surface",
                    receipt.candidate.contract_mode
                ),
            });
        }
        if receipt.candidate.candidate_source_revision != receipt.producer.source_git_revision {
            return Err(mismatch(format!(
                "receipt binds candidate {} but was produced by {}",
                receipt.candidate.candidate_source_revision, receipt.producer.source_git_revision
            )));
        }

        // 7. The derived fields: verdicts are recomputed, never read.
        let recomputed = recompute_verdicts(&receipt.run())?;
        if receipt.verdicts != recomputed {
            return Err(mismatch(
                "receipt verdicts do not match a fresh adjudication of its own observations"
                    .to_owned(),
            ));
        }

        Ok(VerifiedNativeEnrichedReceiptV1 { receipt, address })
    }

    /// Content address over the domain-separated canonical body.
    ///
    /// # Errors
    ///
    /// Returns [`GauntletError::InvalidContract`] when the body cannot be
    /// canonically serialized.
    pub fn receipt_hash(&self) -> Result<String, GauntletError> {
        let body = serde_json::to_vec(self).map_err(|error| GauntletError::InvalidContract {
            reason: format!("native enriched receipt is not canonically serializable: {error}"),
        })?;
        let mut hasher = Sha256::new();
        hasher.update(RECEIPT_HASH_DOMAIN.as_bytes());
        hasher.update([0u8]);
        hasher.update(&body);
        Ok(hex_lower(&hasher.finalize()))
    }
}

/// A receipt whose every claim was recomputed at load.
///
/// The inner receipt is PRIVATE and there is no public constructor: the only
/// way to hold one of these is to have passed
/// [`NativeEnrichedReceiptV1::load_canonical`]. A body deserialized by hand is
/// a `NativeEnrichedReceiptV1` and can never become this type, which is what
/// "rejects synthetic from-body-only construction as release evidence" means
/// in the type system rather than in a comment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedNativeEnrichedReceiptV1 {
    receipt: NativeEnrichedReceiptV1,
    address: String,
}

impl VerifiedNativeEnrichedReceiptV1 {
    /// The verified receipt body.
    #[must_use]
    pub const fn receipt(&self) -> &NativeEnrichedReceiptV1 {
        &self.receipt
    }

    /// The verified content address.
    #[must_use]
    pub fn address(&self) -> &str {
        &self.address
    }

    /// Even a VERIFIED receipt cannot authorize a replacement.
    ///
    /// Verification establishes that the receipt describes what it claims. It
    /// says nothing about whether the enriched surface is complete enough to
    /// replace an engine, which only the terminal release-gate aggregator may
    /// decide.
    #[must_use]
    pub const fn authorizes_replacement(&self) -> bool {
        false
    }

    /// Whether this receipt is admissible as release evidence.
    ///
    /// Separate from loading on purpose. Loading answers "does this object
    /// describe what it claims"; admissibility answers "may it be cited". A
    /// dirty-tree receipt is perfectly loadable and perfectly inadmissible,
    /// and collapsing the two would make the harness unable to record an
    /// honest development run at all.
    ///
    /// # Errors
    ///
    /// Returns [`GauntletError::InvalidContract`] naming the first
    /// admissibility requirement the receipt fails.
    pub fn require_release_admissible(&self) -> Result<(), GauntletError> {
        self.receipt
            .candidate
            .require_produced_by_the_candidate(&self.receipt.producer)?;
        if !self.receipt.both_engines_observed {
            return Err(GauntletError::InvalidContract {
                reason: "a single-engine receipt is not release evidence".to_owned(),
            });
        }
        if !self.receipt.all_verdicts_passed() {
            return Err(GauntletError::InvalidContract {
                reason: "receipt carries failing verdicts".to_owned(),
            });
        }
        Ok(())
    }
}

/// Digest of the committed corpus text.
#[must_use]
pub fn corpus_manifest_sha256() -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"frankensearch.gauntlet.native-enriched-corpus.v1\0");
    for (doc_id, body) in FIXTURE_CORPUS {
        hasher.update(doc_id.as_bytes());
        hasher.update([0u8]);
        hasher.update(body.as_bytes());
        hasher.update([0u8]);
    }
    hex_lower(&hasher.finalize())
}

/// Digest of the committed expectation table.
#[must_use]
pub fn query_manifest_sha256() -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"frankensearch.gauntlet.native-enriched-queries.v1\0");
    for expectation in FIXTURE_EXPECTATIONS {
        let mut row = String::new();
        let _ = write!(
            row,
            "{}|{}|{}|{}|{}|{}",
            expectation.query,
            expectation.limit,
            expectation.offset,
            expectation.matching_docs.join(","),
            expectation.total,
            expectation.unambiguous_top.unwrap_or("-"),
        );
        hasher.update(row.as_bytes());
        hasher.update([0u8]);
    }
    hex_lower(&hasher.finalize())
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

// ---------------------------------------------------------------------------
// Engine drivers — real native APIs, no facade alias
// ---------------------------------------------------------------------------

/// Observe one expectation through the REAL native Quill paginated API.
///
/// Deliberately calls `frankensearch_quill::QuillIndex` directly rather than
/// the facade's `lexical` alias: after bd-8nqz.4.2 flipped `lexical` to Quill
/// (d117ce1f), driving both arms through the facade would observe Quill twice
/// and report it as cross-engine agreement.
///
/// # Errors
///
/// Propagates typed Quill query-execution failures.
pub fn observe_quill(
    cx: &Cx,
    index: &frankensearch_quill::QuillIndex,
    expectation: &EnrichedExpectationV1,
) -> Result<NativeObservationV1, GauntletError> {
    let result = index
        .search_paginated(
            cx,
            expectation.query,
            expectation.limit,
            expectation.offset,
            true,
        )
        .map_err(|error| GauntletError::InvalidContract {
            reason: format!("native Quill paginated search failed: {error}"),
        })?;
    let total =
        usize::try_from(
            result
                .total_count
                .ok_or_else(|| GauntletError::InvalidContract {
                    reason: "native Quill exact-count was requested but not returned".to_owned(),
                })?,
        )
        .map_err(|_| GauntletError::InvalidContract {
            reason: "native Quill total does not fit usize".to_owned(),
        })?;
    let doc_count =
        usize::try_from(result.doc_count).map_err(|_| GauntletError::InvalidContract {
            reason: "native Quill doc_count does not fit usize".to_owned(),
        })?;
    Ok(NativeObservationV1 {
        engine: NativeEngineV1::Quill,
        query: expectation.query.to_owned(),
        limit: expectation.limit,
        offset: expectation.offset,
        page_doc_ids: result
            .hits
            .iter()
            .map(|hit| hit.document_id.clone())
            .collect(),
        total,
        doc_count,
        page_score_bits: result.hits.iter().map(|hit| hit.score.to_bits()).collect(),
    })
}

/// Observe one ENRICHED expectation through the REAL native Quill
/// `search_with_snippets` API.
///
/// # Errors
///
/// Propagates typed Quill snippet/query failures.
pub fn observe_quill_enrichment(
    cx: &Cx,
    index: &frankensearch_quill::QuillIndex,
    expectation: &EnrichmentExpectationV1,
) -> Result<NativeEnrichedObservationV1, GauntletError> {
    let config = frankensearch_quill::SnippetConfig {
        max_chars: frankensearch_quill::DEFAULT_SNIPPET_MAX_CHARS,
        highlight_prefix: expectation.highlight_prefix.to_owned(),
        highlight_postfix: expectation.highlight_postfix.to_owned(),
    };
    let hits = index
        .search_with_snippets(cx, expectation.query, expectation.limit, &config)
        .map_err(|error| GauntletError::InvalidContract {
            reason: format!("native Quill enriched search failed: {error}"),
        })?;
    Ok(NativeEnrichedObservationV1 {
        engine: NativeEngineV1::Quill,
        query: expectation.query.to_owned(),
        highlight_prefix: expectation.highlight_prefix.to_owned(),
        highlight_postfix: expectation.highlight_postfix.to_owned(),
        hits: hits
            .into_iter()
            .map(|hit| NativeEnrichedHitV1 {
                doc_id: hit.document_id,
                rank: hit.rank,
                snippet: hit.snippet,
                // Normalized through Display: the two crates' enums are
                // DIFFERENT types and cannot be compared any other way.
                query_type_code: hit.query_type.to_string(),
                metadata: MetadataStateV1::classify(hit.metadata.as_deref()),
            })
            .collect(),
    })
}

/// Observe one ENRICHED expectation through the REAL native Tantivy
/// `search_with_snippets` API.
///
/// # Errors
///
/// Propagates typed Tantivy snippet/query failures.
#[cfg(feature = "tantivy-oracle")]
pub fn observe_tantivy_enrichment(
    cx: &Cx,
    index: &frankensearch_lexical::TantivyIndex,
    expectation: &EnrichmentExpectationV1,
) -> Result<NativeEnrichedObservationV1, GauntletError> {
    // Tantivy keeps its default window private, so the shared default is taken
    // from `SnippetConfig::default()` and only the tag fields are overridden.
    // Both crates default to the same 200-byte window, which is why the two
    // arms remain comparable.
    let config = frankensearch_lexical::SnippetConfig {
        highlight_prefix: expectation.highlight_prefix.to_owned(),
        highlight_postfix: expectation.highlight_postfix.to_owned(),
        ..frankensearch_lexical::SnippetConfig::default()
    };
    let hits = index
        .search_with_snippets(cx, expectation.query, expectation.limit, &config)
        .map_err(|error| GauntletError::InvalidContract {
            reason: format!("native Tantivy enriched search failed: {error}"),
        })?;
    Ok(NativeEnrichedObservationV1 {
        engine: NativeEngineV1::Tantivy,
        query: expectation.query.to_owned(),
        highlight_prefix: expectation.highlight_prefix.to_owned(),
        highlight_postfix: expectation.highlight_postfix.to_owned(),
        hits: hits
            .into_iter()
            .map(|hit| NativeEnrichedHitV1 {
                doc_id: hit.doc_id,
                rank: hit.rank,
                snippet: hit.snippet,
                query_type_code: hit.query_type.to_string(),
                metadata: MetadataStateV1::classify(hit.metadata.as_ref()),
            })
            .collect(),
    })
}

/// Observe one expectation through the REAL native Tantivy paginated API.
///
/// # Errors
///
/// Propagates typed Tantivy query-execution failures.
#[cfg(feature = "tantivy-oracle")]
pub fn observe_tantivy(
    cx: &Cx,
    index: &frankensearch_lexical::TantivyIndex,
    expectation: &EnrichedExpectationV1,
) -> Result<NativeObservationV1, GauntletError> {
    let page = index
        .oracle_observe_page(cx, expectation.query, expectation.limit, expectation.offset)
        .map_err(|error| GauntletError::InvalidContract {
            reason: format!("native Tantivy paginated search failed: {error}"),
        })?;
    Ok(NativeObservationV1 {
        engine: NativeEngineV1::Tantivy,
        query: expectation.query.to_owned(),
        limit: expectation.limit,
        offset: expectation.offset,
        page_doc_ids: page.hits.iter().map(|hit| hit.doc_id.clone()).collect(),
        total: page.total_count,
        doc_count: page.doc_count,
        page_score_bits: page.hits.iter().map(|hit| hit.score_bits).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic observation, used to exercise the ORACLE itself without a
    /// live engine. Engine-driven tests live in the integration suite.
    fn observation(
        engine: NativeEngineV1,
        expectation: &EnrichedExpectationV1,
        page: &[&str],
        total: usize,
    ) -> NativeObservationV1 {
        NativeObservationV1 {
            engine,
            query: expectation.query.to_owned(),
            limit: expectation.limit,
            offset: expectation.offset,
            page_doc_ids: page.iter().map(|id| (*id).to_owned()).collect(),
            total,
            doc_count: FIXTURE_DOC_COUNT,
            page_score_bits: vec![0x3f80_0000; page.len()],
        }
    }

    fn expectation(query: &str, offset: usize) -> &'static EnrichedExpectationV1 {
        FIXTURE_EXPECTATIONS
            .iter()
            .find(|row| row.query == query && row.offset == offset)
            .expect("fixture row")
    }

    #[test]
    fn a_correct_observation_passes_the_independent_oracle() {
        let row = expectation("quill", 0);
        let correct = observation(NativeEngineV1::Quill, row, &["doc-alpha", "doc-beta"], 2);
        let verdict = adjudicate(row, &correct);
        assert!(verdict.passed(), "{:?}", verdict.oracle_failures);
    }

    /// THE CENTRAL PROOF OF THIS MODULE: a COMMON-MODE mutation makes both
    /// engines identically wrong. Cross-engine agreement reports success. The
    /// independent fixture oracle still fails it.
    ///
    /// If this module's oracle were "the two engines agree", the mutation
    /// below would pass and the receipt would be worthless in precisely the
    /// case a witness exists for: a shared defect in a shared assumption.
    #[test]
    fn a_common_mode_mutation_passes_agreement_and_still_fails_the_oracle() {
        let row = expectation("quill", 0);
        // Both engines drop the same true match and report the same wrong
        // total — a shared analyzer or stop-word defect looks exactly like
        // this.
        let quill = observation(NativeEngineV1::Quill, row, &["doc-alpha"], 1);
        let tantivy = observation(NativeEngineV1::Tantivy, row, &["doc-alpha"], 1);

        assert!(
            engines_agree(&quill, &tantivy),
            "the mutation must be COMMON-MODE, or this proves nothing"
        );

        for observed in [&quill, &tantivy] {
            let verdict = adjudicate(row, observed);
            assert!(
                !verdict.passed(),
                "the independent oracle must catch what agreement cannot: {observed:?}"
            );
            assert!(
                verdict
                    .oracle_failures
                    .iter()
                    .any(|failure| failure.contains("total 1 != expected 2")),
                "got {:?}",
                verdict.oracle_failures
            );
        }
    }

    /// A total inferred from the page length instead of a counting collector
    /// is the defect the limit=1 row exists to catch.
    #[test]
    fn a_total_inferred_from_page_length_fails() {
        let row = expectation("quill", 0);
        let paged = FIXTURE_EXPECTATIONS
            .iter()
            .find(|candidate| candidate.query == "quill" && candidate.limit == 1)
            .expect("limit-1 row");
        assert_eq!(paged.total, row.total, "the fixture pins total == 2 at k=1");

        let inferred = observation(NativeEngineV1::Quill, paged, &["doc-alpha"], 1);
        let verdict = adjudicate(paged, &inferred);
        assert!(!verdict.passed());
        assert!(
            verdict
                .oracle_failures
                .iter()
                .any(|failure| failure.contains("total 1 != expected 2")),
            "got {:?}",
            verdict.oracle_failures
        );
    }

    #[test]
    fn a_reordered_page_fails_where_term_frequency_makes_order_unambiguous() {
        let row = expectation("quill", 0);
        let swapped = observation(NativeEngineV1::Quill, row, &["doc-beta", "doc-alpha"], 2);
        let verdict = adjudicate(row, &swapped);
        assert!(!verdict.passed());
        assert!(
            verdict
                .oracle_failures
                .iter()
                .any(|failure| failure.contains("top-ranked doc-beta")),
            "got {:?}",
            verdict.oracle_failures
        );
    }

    #[test]
    fn a_false_positive_document_fails() {
        let row = expectation("quill", 0);
        let extra = observation(
            NativeEngineV1::Quill,
            row,
            &["doc-alpha", "doc-beta", "doc-delta"],
            2,
        );
        let verdict = adjudicate(row, &extra);
        assert!(!verdict.passed());
        assert!(
            verdict
                .oracle_failures
                .iter()
                .any(|failure| failure.contains("non-matching document doc-delta")),
            "got {:?}",
            verdict.oracle_failures
        );
    }

    #[test]
    fn an_offset_past_the_end_still_reports_the_true_total() {
        let row = expectation("quill", 5);
        let empty_page = observation(NativeEngineV1::Quill, row, &[], 2);
        assert!(adjudicate(row, &empty_page).passed());

        let wrong_total = observation(NativeEngineV1::Quill, row, &[], 0);
        assert!(
            !adjudicate(row, &wrong_total).passed(),
            "an offset past the end must not zero the total"
        );
    }

    #[test]
    fn a_no_match_query_keeps_the_full_live_doc_count() {
        let row = expectation("absent", 0);
        let empty = observation(NativeEngineV1::Quill, row, &[], 0);
        assert!(adjudicate(row, &empty).passed());

        let mut shrunk = empty.clone();
        shrunk.doc_count = 0;
        assert!(
            !adjudicate(row, &shrunk).passed(),
            "zero matches must not be reported as zero live documents"
        );
    }

    #[test]
    fn score_bits_must_cover_every_returned_hit() {
        let row = expectation("quill", 0);
        let mut stripped = observation(NativeEngineV1::Quill, row, &["doc-alpha", "doc-beta"], 2);
        stripped.page_score_bits.clear();
        let verdict = adjudicate(row, &stripped);
        assert!(!verdict.passed());
        assert!(
            verdict
                .oracle_failures
                .iter()
                .any(|failure| failure.contains("score-bit count")),
            "got {:?}",
            verdict.oracle_failures
        );
    }

    fn enrichment_row(query: &str, prefix: &str) -> &'static EnrichmentExpectationV1 {
        FIXTURE_ENRICHMENT_EXPECTATIONS
            .iter()
            .find(|row| row.query == query && row.highlight_prefix == prefix)
            .expect("enrichment fixture row")
    }

    fn enriched(
        engine: NativeEngineV1,
        expectation: &EnrichmentExpectationV1,
        snippet: Option<&str>,
        query_type_code: &str,
        metadata: MetadataStateV1,
    ) -> NativeEnrichedObservationV1 {
        NativeEnrichedObservationV1 {
            engine,
            query: expectation.query.to_owned(),
            highlight_prefix: expectation.highlight_prefix.to_owned(),
            highlight_postfix: expectation.highlight_postfix.to_owned(),
            hits: vec![NativeEnrichedHitV1 {
                doc_id: expectation.subject_doc.to_owned(),
                rank: 0,
                snippet: snippet.map(str::to_owned),
                query_type_code: query_type_code.to_owned(),
                metadata,
            }],
        }
    }

    /// THE ENRICHMENT COMMON-MODE PROOF. Both engines misclassify the same
    /// query the same way. `enriched_engines_agree` reports success; the
    /// independent oracle still fails both.
    ///
    /// This is the case the two-`QueryExplanation`-types finding makes real:
    /// the enums are defined independently in each crate, so they can drift
    /// together — a shared misreading of "two bare words" — and a receipt
    /// whose oracle was cross-engine code equality would bless it.
    #[test]
    fn a_common_mode_query_type_drift_passes_agreement_and_still_fails_the_oracle() {
        let row = enrichment_row("lexical backend", "<b>");
        let snippet = "<b>lexical</b> lexical quill backend";
        let quill = enriched(
            NativeEngineV1::Quill,
            row,
            Some(snippet),
            "simple",
            MetadataStateV1::Absent,
        );
        let tantivy = enriched(
            NativeEngineV1::Tantivy,
            row,
            Some(snippet),
            "simple",
            MetadataStateV1::Absent,
        );

        assert!(
            enriched_engines_agree(&quill, &tantivy),
            "the drift must be COMMON-MODE, or this proves nothing"
        );
        for observed in [&quill, &tantivy] {
            let verdict = adjudicate_enrichment(row, observed);
            assert!(!verdict.passed(), "{observed:?}");
            assert!(
                verdict
                    .oracle_failures
                    .iter()
                    .any(|failure| failure.contains("query_type simple != expected boolean")),
                "got {:?}",
                verdict.oracle_failures
            );
        }
    }

    /// Untrusted document markup surviving as live markup fails, even though
    /// the configured tags are present — which is why the two halves are
    /// asserted separately.
    #[test]
    fn echoed_document_markup_fails_even_when_the_configured_tags_render() {
        let row = enrichment_row("escaping", "<b>");
        let leaked = enriched(
            NativeEngineV1::Quill,
            row,
            Some("<b>escaping</b> <script>alert(1)</script> payload"),
            "simple",
            MetadataStateV1::Absent,
        );
        let verdict = adjudicate_enrichment(row, &leaked);
        assert!(!verdict.passed());
        assert!(
            verdict
                .oracle_failures
                .iter()
                .any(|failure| failure.contains("untrusted document markup")),
            "got {:?}",
            verdict.oracle_failures
        );

        // The safe rendering passes.
        let escaped = enriched(
            NativeEngineV1::Quill,
            row,
            Some("<b>escaping</b> &lt;script&gt;alert(1)&lt;/script&gt; payload"),
            "simple",
            MetadataStateV1::Absent,
        );
        assert!(adjudicate_enrichment(row, &escaped).passed());
    }

    /// A hard-coded `<b>` that ignores the caller's configured tags fails.
    #[test]
    fn ignoring_the_configured_highlight_tags_fails() {
        let row = enrichment_row("quill", "[[");
        let hard_coded = enriched(
            NativeEngineV1::Quill,
            row,
            Some("<b>quill</b> indexes text"),
            "simple",
            MetadataStateV1::Entries {
                pairs: vec!["kind=primary".to_owned(), "lang=en".to_owned()],
            },
        );
        let verdict = adjudicate_enrichment(row, &hard_coded);
        assert!(!verdict.passed());
        assert!(
            verdict
                .oracle_failures
                .iter()
                .any(|failure| failure.contains("[[quill]]")),
            "got {:?}",
            verdict.oracle_failures
        );
    }

    /// Absent and empty-object are both legal for a document indexed without
    /// metadata; fabricated entries are not.
    #[test]
    fn metadata_states_stay_distinguishable() {
        let row = enrichment_row("lexical backend", "<b>");
        let snippet = Some("<b>lexical</b> <b>backend</b>");
        for state in [MetadataStateV1::Absent, MetadataStateV1::EmptyObject] {
            let observed = enriched(NativeEngineV1::Quill, row, snippet, "boolean", state);
            assert!(
                adjudicate_enrichment(row, &observed).passed(),
                "both representations of 'no metadata' are legal"
            );
        }
        let fabricated = enriched(
            NativeEngineV1::Quill,
            row,
            snippet,
            "boolean",
            MetadataStateV1::Entries {
                pairs: vec!["kind=invented".to_owned()],
            },
        );
        let verdict = adjudicate_enrichment(row, &fabricated);
        assert!(!verdict.passed());
        assert!(
            verdict
                .oracle_failures
                .iter()
                .any(|failure| failure.contains("present for a document indexed without any")),
            "got {:?}",
            verdict.oracle_failures
        );
        assert!(!MetadataStateV1::Absent.is_present());
        assert!(!MetadataStateV1::EmptyObject.is_present());
    }

    // -----------------------------------------------------------------
    // Typed capability refusal (slice 4)
    // -----------------------------------------------------------------

    fn capability_row(label: &str) -> &'static CapabilityProbeExpectationV1 {
        FIXTURE_CAPABILITY_EXPECTATIONS
            .iter()
            .find(|row| row.label == label)
            .expect("capability fixture row")
    }

    fn refusal() -> EnrichedCapabilityRefusalV1 {
        let class = EnrichedErrorClassV1::QueryCapabilityPositionsRequired;
        EnrichedCapabilityRefusalV1 {
            class,
            code: class.stable_code().to_owned(),
            schema: WITNESS_POSITIONLESS_SCHEMA_NAME.to_owned(),
            field: "content".to_owned(),
            operator: "phrase".to_owned(),
            capability: "positions".to_owned(),
        }
    }

    #[test]
    fn the_expected_typed_refusal_passes() {
        let row = capability_row("positionless-multi-term-phrase");
        let verdict =
            adjudicate_capability_probe(row, &CapabilityProbeOutcomeV1::Refused(refusal()));
        assert!(verdict.passed(), "{:?}", verdict.oracle_failures);
    }

    /// One named mutation of a refusal: its label, the edit, and the substring
    /// the resulting failure must contain.
    type RefusalMutation = (
        &'static str,
        fn(&mut EnrichedCapabilityRefusalV1),
        &'static str,
    );

    /// Each named field of the refusal is adjudicated SEPARATELY, so a refusal
    /// that is right about the schema and wrong about the operator is caught.
    /// A single rendered-message comparison could not distinguish these.
    #[test]
    fn every_named_field_of_the_refusal_is_pinned() {
        let row = capability_row("positionless-multi-term-phrase");
        let mutations: [RefusalMutation; 5] = [
            (
                "schema",
                |refusal| refusal.schema = "frankensearch-default-v1".to_owned(),
                "names schema",
            ),
            (
                "field",
                |refusal| refusal.field = "title".to_owned(),
                "names field",
            ),
            (
                "operator",
                |refusal| refusal.operator = "boolean".to_owned(),
                "names operator",
            ),
            (
                "capability",
                |refusal| refusal.capability = "fast".to_owned(),
                "names capability",
            ),
            (
                "code",
                |refusal| refusal.code = "FSNE-CAP-9999".to_owned(),
                "refusal code",
            ),
        ];
        for (name, mutate, expected_failure) in mutations {
            let mut mutated = refusal();
            mutate(&mut mutated);
            let verdict =
                adjudicate_capability_probe(row, &CapabilityProbeOutcomeV1::Refused(mutated));
            assert!(!verdict.passed(), "the {name} mutation must fail");
            assert!(
                verdict
                    .oracle_failures
                    .iter()
                    .any(|failure| failure.contains(expected_failure)),
                "the {name} mutation must fail for the right reason: {:?}",
                verdict.oracle_failures
            );
        }
    }

    /// A bare `UnsupportedQuery` is exactly the degraded error the bead
    /// rejects. It maps to `UnmappedError`, never into the capability class,
    /// and it does not satisfy a typed-refusal expectation.
    #[test]
    fn a_bare_unsupported_query_is_never_laundered_into_the_capability_class() {
        let generic = frankensearch_quill::QuillIndexError::UnsupportedQuery {
            detail: "phrase".to_owned(),
        };
        let outcome = classify_quill_refusal(&generic);
        assert!(
            matches!(outcome, CapabilityProbeOutcomeV1::UnmappedError { .. }),
            "got {outcome:?}"
        );

        let row = capability_row("positionless-multi-term-phrase");
        let verdict = adjudicate_capability_probe(row, &outcome);
        assert!(!verdict.passed());
        assert!(
            verdict
                .oracle_failures
                .iter()
                .any(|failure| failure.contains("unclassified error")),
            "got {:?}",
            verdict.oracle_failures
        );
    }

    /// The real typed error maps to the real refusal, so the mapper is proved
    /// against the engine's own error type rather than a hand-built stand-in.
    #[test]
    fn the_real_typed_capability_error_maps_to_the_witness_class() {
        use frankensearch_quill::query::{IndexCapability, QueryCapabilityError, QueryExplanation};

        let typed = frankensearch_quill::QuillIndexError::QueryCapability(
            QueryCapabilityError::PositionsRequired {
                schema: WITNESS_POSITIONLESS_SCHEMA_NAME,
                field: "content".to_owned(),
                operator: QueryExplanation::Phrase,
                capability: IndexCapability::Positions,
            },
        );
        assert_eq!(
            classify_quill_refusal(&typed),
            CapabilityProbeOutcomeV1::Refused(refusal())
        );
    }

    /// An engine that refused EVERYTHING would pass the acceptance row on its
    /// own. The served control rows are what stop that, so this proves they
    /// actually fail on a blanket refusal.
    #[test]
    fn a_blanket_refusal_fails_the_served_control_rows() {
        for label in [
            "positionless-single-term-quote",
            "positionless-bare-term",
            "positionful-multi-term-phrase",
        ] {
            let row = capability_row(label);
            let verdict =
                adjudicate_capability_probe(row, &CapabilityProbeOutcomeV1::Refused(refusal()));
            assert!(!verdict.passed(), "{label} must reject a blanket refusal");
            assert!(
                verdict
                    .oracle_failures
                    .iter()
                    .any(|failure| failure.contains("refused a servable query")),
                "got {:?}",
                verdict.oracle_failures
            );
        }
    }

    /// The reversed-phrase row is what makes the positionful control
    /// non-vacuous: an engine that degenerated a phrase into a conjunction
    /// would return doc-alpha for it.
    #[test]
    fn a_phrase_degenerated_into_a_conjunction_fails_the_reversed_row() {
        let row = capability_row("positionful-reversed-phrase");
        let degenerated = CapabilityProbeOutcomeV1::Served {
            doc_ids: vec!["doc-alpha".to_owned()],
            total: Some(1),
        };
        let verdict = adjudicate_capability_probe(row, &degenerated);
        assert!(!verdict.passed());
        assert!(
            verdict
                .oracle_failures
                .iter()
                .any(|failure| failure.contains("served [\"doc-alpha\"] != expected []")),
            "got {:?}",
            verdict.oracle_failures
        );
    }

    /// A served page with no exact count fails: `exact_count` was requested.
    #[test]
    fn a_served_row_without_an_exact_count_fails() {
        let row = capability_row("positionful-multi-term-phrase");
        let countless = CapabilityProbeOutcomeV1::Served {
            doc_ids: vec!["doc-alpha".to_owned()],
            total: None,
        };
        let verdict = adjudicate_capability_probe(row, &countless);
        assert!(!verdict.passed());
        assert!(
            verdict
                .oracle_failures
                .iter()
                .any(|failure| failure.contains("no exact total")),
            "got {:?}",
            verdict.oracle_failures
        );
    }

    #[test]
    fn the_stable_error_code_is_pinned() {
        let class = EnrichedErrorClassV1::QueryCapabilityPositionsRequired;
        assert_eq!(class.stable_code(), "FSNE-CAP-0001");
        assert_eq!(
            class.class_code(),
            "enriched.query_capability.positions_required"
        );
    }

    #[test]
    fn engine_diagnostics_are_bounded() {
        let long = "x".repeat(MAX_DIAGNOSTIC_CHARS * 3);
        let bounded = bounded_diagnostic(&long);
        assert!(bounded.starts_with(&"x".repeat(MAX_DIAGNOSTIC_CHARS)));
        assert!(bounded.ends_with("…[truncated]"));
        assert_eq!(bounded.chars().count(), MAX_DIAGNOSTIC_CHARS + 12);
        // A multi-byte diagnostic must be cut on a character boundary, not a
        // byte offset, or the bound itself would corrupt the record.
        let multibyte = "é".repeat(MAX_DIAGNOSTIC_CHARS * 2);
        assert!(!bounded_diagnostic(&multibyte).contains('\u{FFFD}'));
    }

    /// The positionless schema must differ from the shipping default in
    /// EXACTLY one axis. If it drifted in another, a refusal would no longer
    /// be attributable to the missing capability.
    #[test]
    fn the_positionless_schema_differs_only_in_positions() {
        use frankensearch_quill::{DEFAULT_SCHEMA, FieldKind};

        WITNESS_POSITIONLESS_SCHEMA
            .validate()
            .expect("the witness positionless schema must be valid");
        assert_ne!(
            WITNESS_POSITIONLESS_SCHEMA.name, DEFAULT_SCHEMA.name,
            "the arms must be distinguishable by name"
        );
        assert_eq!(
            WITNESS_POSITIONLESS_SCHEMA.fields.len(),
            DEFAULT_SCHEMA.fields.len()
        );
        let mut positionless_text_fields = 0;
        for (witness, shipping) in WITNESS_POSITIONLESS_SCHEMA
            .fields
            .iter()
            .zip(DEFAULT_SCHEMA.fields)
        {
            assert_eq!(witness.id, shipping.id);
            assert_eq!(witness.name, shipping.name);
            assert_eq!(witness.stored, shipping.stored);
            match (witness.kind, shipping.kind) {
                (
                    FieldKind::Text {
                        analyzer: witness_analyzer,
                        positions: witness_positions,
                    },
                    FieldKind::Text {
                        analyzer: shipping_analyzer,
                        positions: shipping_positions,
                    },
                ) => {
                    assert_eq!(
                        witness_analyzer, shipping_analyzer,
                        "the analyzer must not drift with the capability"
                    );
                    assert!(
                        shipping_positions,
                        "the shipping control must be positioned"
                    );
                    assert!(!witness_positions);
                    positionless_text_fields += 1;
                }
                (witness_kind, shipping_kind) => {
                    assert_eq!(witness_kind, shipping_kind, "non-text fields must match");
                }
            }
        }
        assert_eq!(
            positionless_text_fields, 2,
            "both text fields must lose positions"
        );
    }

    /// The served rows are written sorted, because the probe canonicalizes its
    /// observation by sorting. An unsorted row would fail for a reason that
    /// has nothing to do with the engine.
    #[test]
    fn the_committed_served_rows_are_canonically_sorted() {
        for row in FIXTURE_CAPABILITY_EXPECTATIONS {
            if let CapabilityOutcomeExpectationV1::Served { docs, .. } = row.outcome {
                let mut sorted = docs.to_vec();
                sorted.sort_unstable();
                assert_eq!(docs, sorted.as_slice(), "row {} is not sorted", row.label);
            }
        }
    }

    // -----------------------------------------------------------------
    // Engine identity and accepted-candidate binding (slice 5)
    // -----------------------------------------------------------------

    #[test]
    fn the_derived_quill_identity_is_the_shipping_default() {
        let identity = derive_quill_engine_identity().expect("derive the Quill arm identity");
        assert_eq!(identity.engine, NativeEngineV1::Quill);
        assert_eq!(
            identity.backend_code,
            frankensearch_quill::QUILL_LEXICAL_BACKEND
        );
        // The `assert!` first, so the `if let` below cannot skip silently: a
        // conditional block whose branch never fires is a green test that
        // asserts nothing.
        assert!(
            matches!(
                identity.schema,
                NativeSchemaIdentityV1::QuillDescriptor { .. }
            ),
            "Quill publishes a schema descriptor"
        );
        if let NativeSchemaIdentityV1::QuillDescriptor {
            name,
            schema_id_hex,
        } = &identity.schema
        {
            assert_eq!(name, frankensearch_quill::DEFAULT_SCHEMA.name);
            assert_eq!(schema_id_hex.len(), 16, "a 64-bit id renders as 16 hex");
            // Derived, not transcribed: the id must track the schema's own
            // canonical encoding.
            let expected = frankensearch_quill::DEFAULT_SCHEMA
                .schema_id()
                .expect("shipping schema id");
            assert_eq!(*schema_id_hex, format!("{expected:016x}"));
        }
        identity
            .reject_cass_identity(&crate::runner::SemanticContract::shipping_default())
            .expect("the shipping default is not a CASS identity");
    }

    /// Every CASS marker is load-bearing on its own. In particular the schema
    /// ID is checked separately from the schema NAME, because a renamed CASS
    /// schema keeps its encoding — content identity, not the label, decides
    /// whether the CASS profile was exercised.
    #[test]
    fn each_cass_marker_is_independently_rejected() {
        let shipping = crate::runner::SemanticContract::shipping_default();
        let clean = derive_quill_engine_identity().expect("clean identity");

        // 1. The contract alone, with an otherwise clean identity.
        let error = clean
            .reject_cass_identity(&crate::runner::SemanticContract::cass())
            .expect_err("the CASS semantic contract must be rejected");
        assert!(error.to_string().contains("CASS semantic contract"));

        // 2. The backend code.
        let mut cass_backend = clean.clone();
        cass_backend.backend_code = "quill-cass".to_owned();
        let error = cass_backend
            .reject_cass_identity(&shipping)
            .expect_err("a CASS backend code must be rejected");
        assert!(error.to_string().contains("backend code"));

        // 3. The schema NAME.
        let mut cass_named = clean.clone();
        cass_named.schema = NativeSchemaIdentityV1::QuillDescriptor {
            name: frankensearch_quill::CASS_SEMANTIC_SCHEMA.name.to_owned(),
            schema_id_hex: "0000000000000000".to_owned(),
        };
        let error = cass_named
            .reject_cass_identity(&shipping)
            .expect_err("the CASS schema name must be rejected");
        assert!(error.to_string().contains("names the CASS schema"));

        // 4. The schema ID under an innocuous NAME — the renamed-CASS case a
        //    name check alone would pass.
        let cass_id = frankensearch_quill::CASS_SEMANTIC_SCHEMA
            .schema_id()
            .expect("CASS schema id");
        let mut renamed = clean.clone();
        renamed.schema = NativeSchemaIdentityV1::QuillDescriptor {
            name: "frankensearch-totally-not-cass-v1".to_owned(),
            schema_id_hex: format!("{cass_id:016x}"),
        };
        let error = renamed
            .reject_cass_identity(&shipping)
            .expect_err("the CASS schema identity must be rejected under any name");
        assert!(
            error.to_string().contains("under another name"),
            "got {error}"
        );

        // The control: the real shipping identity survives all four checks.
        assert_ne!(
            frankensearch_quill::DEFAULT_SCHEMA.name,
            frankensearch_quill::CASS_SEMANTIC_SCHEMA.name
        );
        clean
            .reject_cass_identity(&shipping)
            .expect("the shipping identity must survive");
    }

    #[test]
    fn the_semantic_contract_accept_set_admits_shipping_and_refuses_cass() {
        for accepted in [
            crate::runner::SemanticContract::shipping_default(),
            crate::runner::SemanticContract::scalar_g1a(),
        ] {
            require_shipping_semantic_contract(&accepted).expect("a shipping contract");
        }
        let error = require_shipping_semantic_contract(&crate::runner::SemanticContract::cass())
            .expect_err("CASS is not a shipping contract");
        assert!(error.to_string().contains("CASS contract"), "got {error}");
    }

    /// Minimal provenance for the repaired baseline.
    ///
    /// The witness gate checks PRESENCE; the campaign layer owns provenance's
    /// internal consistency and this module deliberately does not re-implement
    /// it, which would create a second definition that could drift.
    fn baseline_provenance() -> crate::runner::CampaignProvenance {
        crate::runner::CampaignProvenance {
            producer_build_identity_sha256: "0".repeat(64),
            cargo_lock_sha256: "1".repeat(64),
            rustc_version_verbose: "rustc 0.0.0 (witness baseline)".to_owned(),
            rust_toolchain_channel: "nightly-0000-00-00".to_owned(),
            unicode_version: "0.0.0".to_owned(),
            unicode_normalization_version: "0.0.0".to_owned(),
            unicode_normalization_table_version: "0.0.0".to_owned(),
            query_generator_id: "witness-baseline".to_owned(),
            query_generator_schema_version: 1,
            query_seed: 0,
            query_source_identity_sha256: "2".repeat(64),
            query_profile_sha256: "3".repeat(64),
            analyzer_contract_hash: "4".repeat(64),
            schema_contract_hash: "5".repeat(64),
            corpus_manifest_hash: "6".repeat(64),
            query_manifest_hash: "7".repeat(64),
            corpus_seed: None,
        }
    }

    /// The pinned fixture repaired along every axis the witness gates.
    fn accepted_candidate_report() -> crate::runner::CampaignReport {
        let mut report = crate::runner::load_pinned_campaign_report_v7()
            .expect("the pinned V7 campaign report fixture");
        report.lexical_coverage = crate::runner::CampaignLexicalCoverageSummary::CoreLexicalV3 {
            subject: Box::new(crate::runner::LexicalSideCoverageCounts::default()),
            oracle: Box::new(crate::runner::LexicalSideCoverageCounts::default()),
            admissible: true,
        };
        report.semantic_contract = crate::runner::SemanticContract::shipping_default();
        report.provenance = Some(baseline_provenance());
        report.producer_build_identity.source_git_dirty = false;
        report
    }

    /// THE REAL ARTIFACT. The only committed `CampaignReport` in this
    /// repository passed its campaign, and the witness still refuses it. This
    /// is the acceptance's "generic passed `CampaignReport` /
    /// `RankEnvelopeOnly` report is rejected" proved against a real object
    /// rather than a hand-built straw man.
    #[test]
    fn the_real_pinned_campaign_report_is_rejected_although_it_passed() {
        let report = crate::runner::load_pinned_campaign_report_v7()
            .expect("the pinned V7 campaign report fixture");
        assert!(
            report.passed,
            "the fixture must really be a PASSED report, or this test proves nothing"
        );
        assert!(
            matches!(
                report.lexical_coverage,
                crate::runner::CampaignLexicalCoverageSummary::RankEnvelopeOnly
            ),
            "the fixture must really be rank-envelope-only"
        );
        let error = AcceptedCandidateBindingV1::from_campaign_report(&report)
            .expect_err("a passed rank-envelope report must not bind a candidate");
        assert!(error.to_string().contains("rank-envelope"), "got {error}");
    }

    /// One named mutation of a candidate report: its label, the edit, and the
    /// substring the resulting rejection must contain.
    type ReportMutation = (
        &'static str,
        fn(&mut crate::runner::CampaignReport),
        &'static str,
    );

    /// Each gate is proved load-bearing by mutating ONE axis of an otherwise
    /// admissible report. Order-independent, unlike walking the ladder.
    #[test]
    fn every_candidate_gate_is_independently_load_bearing() {
        let baseline = accepted_candidate_report();
        let binding = AcceptedCandidateBindingV1::from_campaign_report(&baseline)
            .expect("the repaired report must bind, or the mutations prove nothing");
        assert_eq!(
            binding.contract_mode,
            crate::campaign_contract::CampaignContractModeV1::CoreLexicalV3
        );
        assert_eq!(
            binding.candidate_source_revision,
            baseline.producer_build_identity.source_git_revision
        );

        let mutations: [ReportMutation; 6] = [
            ("not passed", |report| report.passed = false, "did not pass"),
            (
                "rank envelope only",
                |report| {
                    report.lexical_coverage =
                        crate::runner::CampaignLexicalCoverageSummary::RankEnvelopeOnly;
                },
                "rank-envelope",
            ),
            (
                "coverage scope absent",
                |report| {
                    report.lexical_coverage =
                        crate::runner::CampaignLexicalCoverageSummary::LegacyMissing;
                },
                "no lexical coverage scope",
            ),
            (
                "CASS semantics",
                |report| report.semantic_contract = crate::runner::SemanticContract::cass(),
                "CASS contract",
            ),
            (
                "provenance missing",
                |report| report.provenance = None,
                "no immutable provenance",
            ),
            (
                "dirty producer",
                |report| report.producer_build_identity.source_git_dirty = true,
                "dirty tree",
            ),
        ];
        for (name, mutate, expected) in mutations {
            let mut mutated = baseline.clone();
            mutate(&mut mutated);
            let error = AcceptedCandidateBindingV1::from_campaign_report(&mutated).expect_err(name);
            assert!(
                error.to_string().contains(expected),
                "the {name} mutation must be rejected for its own reason, got {error}"
            );
        }

        // Admissible-but-not-admitted core-v3 coverage is its own rejection:
        // claiming the scope is not the same as clearing it.
        let mut inadmissible = baseline;
        inadmissible.lexical_coverage =
            crate::runner::CampaignLexicalCoverageSummary::CoreLexicalV3 {
                subject: Box::new(crate::runner::LexicalSideCoverageCounts::default()),
                oracle: Box::new(crate::runner::LexicalSideCoverageCounts::default()),
                admissible: false,
            };
        let error = AcceptedCandidateBindingV1::from_campaign_report(&inadmissible)
            .expect_err("inadmissible core-v3 coverage");
        assert!(error.to_string().contains("not admissible"), "got {error}");
    }

    /// A receipt naming one candidate while produced by another binary is
    /// describing an engine it did not run.
    #[test]
    fn a_stale_candidate_binding_is_rejected() {
        let baseline = accepted_candidate_report();
        let binding =
            AcceptedCandidateBindingV1::from_campaign_report(&baseline).expect("baseline binds");
        let mut producer = baseline.producer_build_identity.clone();

        binding
            .require_produced_by_the_candidate(&producer)
            .expect("the producer IS the bound candidate");

        producer.source_git_revision = "b".repeat(40);
        let error = binding
            .require_produced_by_the_candidate(&producer)
            .expect_err("a different producer revision is a stale candidate");
        assert!(
            error.to_string().contains("but was produced by"),
            "got {error}"
        );

        let mut dirty = baseline.producer_build_identity.clone();
        dirty.source_git_dirty = true;
        let error = binding
            .require_produced_by_the_candidate(&dirty)
            .expect_err("dirty producer provenance is rejected");
        assert!(error.to_string().contains("dirty"), "got {error}");

        let rank_only = AcceptedCandidateBindingV1 {
            candidate_source_revision: binding.candidate_source_revision.clone(),
            contract_mode: crate::campaign_contract::CampaignContractModeV1::RankEnvelopeOnly,
        };
        let error = rank_only
            .require_produced_by_the_candidate(&baseline.producer_build_identity)
            .expect_err("a rank-envelope binding is not the core-v3 surface");
        assert!(error.to_string().contains("core-v3"), "got {error}");
    }

    #[test]
    fn canonical_git_revisions_are_recognized_exactly() {
        assert!(is_canonical_git_revision(&"a".repeat(40)));
        assert!(is_canonical_git_revision(
            "2c6bbf1a49c3ad663b968a08dcf260cb1e53e61b"
        ));
        assert!(!is_canonical_git_revision(&"a".repeat(39)));
        assert!(!is_canonical_git_revision(&"A".repeat(40)));
        assert!(!is_canonical_git_revision("unavailable"));
        assert!(!is_canonical_git_revision(&"g".repeat(40)));
    }

    // -----------------------------------------------------------------
    // Canonical load/replay and the provenance mutation battery (slice 6)
    // -----------------------------------------------------------------

    /// A run that satisfies every committed table, built synthetically.
    ///
    /// Synthetic on purpose: these tests exercise the LOADER, whose job is to
    /// recompute, not to re-run engines. The engines are exercised against
    /// these same tables by the live suite, so the tables are not merely
    /// self-consistent.
    fn passing_run() -> NativeEnrichedRunV1 {
        let observations = FIXTURE_EXPECTATIONS
            .iter()
            .map(|row| {
                let page: Vec<&str> = row
                    .matching_docs
                    .iter()
                    .skip(row.offset)
                    .take(row.limit)
                    .copied()
                    .collect();
                observation(NativeEngineV1::Quill, row, &page, row.total)
            })
            .collect();
        let enriched_observations = FIXTURE_ENRICHMENT_EXPECTATIONS
            .iter()
            .map(|row| {
                let metadata = if row.expected_metadata_pairs.is_empty() {
                    MetadataStateV1::Absent
                } else {
                    MetadataStateV1::Entries {
                        pairs: row
                            .expected_metadata_pairs
                            .iter()
                            .map(|pair| (*pair).to_owned())
                            .collect(),
                    }
                };
                let snippet = format!(
                    "{}{}{} {UTF8_INTACT_TOKEN}",
                    row.highlight_prefix, row.highlighted_term, row.highlight_postfix
                );
                enriched(
                    NativeEngineV1::Quill,
                    row,
                    Some(&snippet),
                    row.expected_query_type_code,
                    metadata,
                )
            })
            .collect();
        let capability_outcomes = FIXTURE_CAPABILITY_EXPECTATIONS
            .iter()
            .map(|row| CapabilityProbeRecordV1 {
                label: row.label.to_owned(),
                outcome: match row.outcome {
                    CapabilityOutcomeExpectationV1::Refused { .. } => {
                        CapabilityProbeOutcomeV1::Refused(refusal())
                    }
                    CapabilityOutcomeExpectationV1::Served { docs, total } => {
                        CapabilityProbeOutcomeV1::Served {
                            doc_ids: docs.iter().map(|doc| (*doc).to_owned()).collect(),
                            total: Some(total),
                        }
                    }
                },
            })
            .collect();
        NativeEnrichedRunV1 {
            observations,
            enriched_observations,
            capability_outcomes,
            both_engines_observed: false,
        }
    }

    /// A receipt sealed to THIS binary, so every loader gate except the one
    /// under test is satisfied.
    fn loadable_receipt() -> NativeEnrichedReceiptV1 {
        let producer =
            GauntletProducerBuildIdentity::compiled().expect("this binary has a sealed identity");
        let candidate = AcceptedCandidateBindingV1 {
            candidate_source_revision: producer.source_git_revision.clone(),
            contract_mode: crate::campaign_contract::CampaignContractModeV1::CoreLexicalV3,
        };
        NativeEnrichedReceiptV1::assemble_with_producer(&passing_run(), candidate, producer)
            .expect("assemble a receipt for this binary")
    }

    fn load(
        receipt: &NativeEnrichedReceiptV1,
    ) -> Result<VerifiedNativeEnrichedReceiptV1, GauntletError> {
        let address = receipt.receipt_hash().expect("address");
        let bytes = serde_json::to_vec(receipt).expect("canonical body");
        NativeEnrichedReceiptV1::load_canonical(&bytes, &address)
    }

    #[test]
    fn a_receipt_sealed_to_this_binary_loads_and_replays() {
        let receipt = loadable_receipt();
        let verified = load(&receipt).expect("the sealed receipt must load");
        assert_eq!(verified.receipt(), &receipt);
        assert_eq!(verified.address(), receipt.receipt_hash().expect("address"));
        // The non-authorization property survives verification itself.
        assert!(!verified.authorizes_replacement());
        assert!(!receipt.authorizes_replacement());
    }

    /// The address is checked, but the address is the WEAK half: an editor can
    /// recompute it. Both halves are proved here — a stale address is caught,
    /// and so is an edit whose address was recomputed.
    #[test]
    fn a_recomputed_address_does_not_launder_an_edited_body() {
        let receipt = loadable_receipt();
        let honest_address = receipt.receipt_hash().expect("address");

        // Stale address: body edited, address left alone.
        let mut edited = receipt.clone();
        edited.corpus_manifest_sha256 = "0".repeat(64);
        let bytes = serde_json::to_vec(&edited).expect("body");
        let error = NativeEnrichedReceiptV1::load_canonical(&bytes, &honest_address)
            .expect_err("an edited body must not match the old address");
        assert!(error.to_string().contains("does not match the expected"));

        // Recomputed address: internally consistent, still false.
        let error = load(&edited).expect_err("a recomputed address must not launder the edit");
        assert!(
            error.to_string().contains("corpus manifest"),
            "the recomputation, not the address, must catch it: {error}"
        );
    }

    /// One named mutation of a stored receipt: its label, the edit, and the
    /// substring the resulting rejection must contain.
    type ReceiptMutation = (&'static str, fn(&mut NativeEnrichedReceiptV1), &'static str);

    /// THE PROVENANCE MUTATION BATTERY. Each axis the acceptance names is
    /// mutated alone, with the address recomputed every time — the hostile
    /// case, not the careless one.
    #[test]
    fn every_provenance_axis_is_independently_replayed() {
        let baseline = loadable_receipt();
        load(&baseline).expect("the baseline must load, or the mutations prove nothing");

        let mutations: [ReceiptMutation; 12] = [
            (
                "corpus manifest",
                |receipt| receipt.corpus_manifest_sha256 = "0".repeat(64),
                "corpus manifest",
            ),
            (
                "query manifest",
                |receipt| receipt.query_manifest_sha256 = "0".repeat(64),
                "query manifest",
            ),
            (
                "capability manifest",
                |receipt| receipt.capability_manifest_sha256 = "0".repeat(64),
                "capability manifest",
            ),
            (
                "candidate SHA",
                |receipt| receipt.producer.source_git_revision = "c".repeat(40),
                "source revision",
            ),
            (
                "lockfile",
                |receipt| receipt.producer.cargo_lock_sha256 = "0".repeat(64),
                "Cargo.lock digest",
            ),
            (
                "toolchain",
                |receipt| receipt.producer.rustc_version_verbose_hex = "00".to_owned(),
                "toolchain",
            ),
            (
                "target triple",
                |receipt| receipt.producer.target_triple = "wasm32-unknown-unknown".to_owned(),
                "target triple",
            ),
            (
                "profile",
                |receipt| receipt.producer.cargo_profile = "release".to_owned(),
                "cargo profile",
            ),
            (
                "features",
                |receipt| receipt.producer.enabled_features_sha256 = "0".repeat(64),
                "feature selection",
            ),
            (
                "engine identity",
                |receipt| {
                    if let Some(identity) = receipt.engine_identities.first_mut() {
                        identity.backend_code = "not-quill".to_owned();
                    }
                },
                "engine identities do not match",
            ),
            (
                "stale candidate",
                |receipt| receipt.candidate.candidate_source_revision = "d".repeat(40),
                "binds candidate",
            ),
            (
                "schema version",
                |receipt| receipt.schema_version = 99,
                "schema version",
            ),
        ];
        for (name, mutate, expected) in mutations {
            let mut mutated = baseline.clone();
            mutate(&mut mutated);
            let error = load(&mutated).expect_err(name).to_string();
            assert!(
                error.contains(expected),
                "the {name} mutation must be rejected for its own reason, got {error}"
            );
        }
    }

    /// A DERIVED field is not a claim. Editing a verdict to say "passed", or
    /// editing an observation without editing its verdict, are both caught by
    /// the same fresh adjudication.
    #[test]
    fn edited_derived_fields_do_not_survive_replay() {
        let baseline = loadable_receipt();

        // Claim a pass that the observations do not support.
        let mut broken_observation = baseline.clone();
        broken_observation.observations[0].total += 1;
        let error = load(&broken_observation)
            .expect_err("an edited observation must not keep its old verdict");
        assert!(
            error.to_string().contains("fresh adjudication"),
            "got {error}"
        );

        // Silence a failing verdict without touching the observation.
        let mut silenced = broken_observation.clone();
        silenced.verdicts = recompute_verdicts(&silenced.run()).expect("recompute");
        assert!(
            !silenced.all_verdicts_passed(),
            "the edited observation really must fail its oracle"
        );
        for verdict in &mut silenced.verdicts {
            verdict.oracle_failures.clear();
        }
        let error =
            load(&silenced).expect_err("a hand-cleared verdict must not survive recomputation");
        assert!(
            error.to_string().contains("fresh adjudication"),
            "got {error}"
        );
    }

    /// ONE-BIT SCORE AND SNIPPET STATE: the classes no independent oracle can
    /// adjudicate, and therefore the classes the ADDRESS exists for.
    ///
    /// A BM25 score float is an engine implementation detail — the committed
    /// expectations deliberately do not predict it (see the module header), so
    /// no recomputation can catch a changed score bit. The same is true of
    /// whether an engine rendered a snippet at all: `None` is a legal answer.
    /// What binds them is the content address, which is why the receipt
    /// records score bits and snippet state at all and why the aggregator must
    /// hold the address out of band. This test is the proof that the binding
    /// actually works for both.
    #[test]
    fn a_one_bit_score_change_and_a_dropped_snippet_fail_against_the_bound_address() {
        let receipt = loadable_receipt();
        let bound_address = receipt.receipt_hash().expect("address");

        let mut one_bit = receipt.clone();
        let original = one_bit.observations[0].page_score_bits[0];
        one_bit.observations[0].page_score_bits[0] = original ^ 1;
        assert_eq!(
            one_bit.observations[0].page_score_bits[0].count_ones() as i32
                - original.count_ones() as i32,
            1,
            "the mutation must really be a single bit"
        );
        // The independent oracle cannot see it — proving the address is what
        // does the work here, not a recomputation.
        assert_eq!(
            recompute_verdicts(&one_bit.run()).expect("recompute"),
            receipt.verdicts,
            "a score bit is invisible to the fixture oracle by design"
        );
        let bytes = serde_json::to_vec(&one_bit).expect("body");
        let error = NativeEnrichedReceiptV1::load_canonical(&bytes, &bound_address)
            .expect_err("a one-bit score change must fail against the bound address");
        assert!(error.to_string().contains("does not match the expected"));

        // Snippet state: Some -> None is a legal ENGINE answer, so only the
        // address can bind which one was actually observed.
        let mut dropped = receipt;
        dropped.enriched_observations[0].hits[0].snippet = None;
        let bytes = serde_json::to_vec(&dropped).expect("body");
        let error = NativeEnrichedReceiptV1::load_canonical(&bytes, &bound_address)
            .expect_err("a dropped snippet must fail against the bound address");
        assert!(error.to_string().contains("does not match the expected"));
    }

    /// A receipt cannot quietly drop the rows that failed.
    #[test]
    fn a_receipt_that_omits_committed_rows_is_rejected() {
        let mut truncated = loadable_receipt();
        truncated.observations.pop();
        truncated.verdicts = Vec::new();
        let error = load(&truncated).expect_err("dropping a committed row must be rejected");
        assert!(
            error.to_string().contains("paginated observations"),
            "got {error}"
        );

        let mut padded = loadable_receipt();
        let mut extra = padded.observations[0].clone();
        extra.query = "a query nobody committed".to_owned();
        padded.observations.push(extra);
        let error = load(&padded).expect_err("an unaudited row must be rejected");
        assert!(
            error.to_string().contains("no committed expectation"),
            "got {error}"
        );
    }

    /// An unknown field is a parse failure, not a silently ignored extension.
    #[test]
    fn an_unknown_field_is_not_silently_accepted() {
        let receipt = loadable_receipt();
        let mut body: serde_json::Value =
            serde_json::to_value(&receipt).expect("receipt serializes");
        body.as_object_mut()
            .expect("receipt is a JSON object")
            .insert("authorizes_replacement".to_owned(), serde_json::json!(true));
        let bytes = serde_json::to_vec(&body).expect("body");
        let error = NativeEnrichedReceiptV1::load_canonical(&bytes, "unused")
            .expect_err("an unknown field must fail the parse");
        assert!(matches!(error, GauntletError::Json(_)), "got {error}");
    }

    /// Loading proves the object describes what it claims; admissibility
    /// decides whether it may be cited. This tree is dirty while peers work in
    /// it, so the dirty arm is the one exercised here — and it is the arm that
    /// matters, because a dirty receipt that loaded AND was admissible would
    /// be the failure.
    #[test]
    fn loading_is_not_admissibility() {
        let receipt = loadable_receipt();
        let verified = load(&receipt).expect("integrity holds regardless of admissibility");
        let admissible = verified.require_release_admissible();
        if receipt.producer.source_git_dirty {
            let error = admissible.expect_err("a dirty-tree receipt is not release evidence");
            assert!(error.to_string().contains("dirty"), "got {error}");
        } else {
            // A clean tree still fails on single-engine coverage, which this
            // synthetic run declares honestly.
            let error = admissible.expect_err("a single-engine receipt is not release evidence");
            assert!(error.to_string().contains("single-engine"), "got {error}");
        }
    }

    #[test]
    fn the_manifests_change_when_the_fixtures_change() {
        let corpus = corpus_manifest_sha256();
        let queries = query_manifest_sha256();
        let capabilities = capability_manifest_sha256();
        assert_eq!(corpus.len(), 64);
        assert_eq!(queries.len(), 64);
        assert_eq!(capabilities.len(), 64);
        for (left, right) in [
            (&corpus, &queries),
            (&corpus, &capabilities),
            (&queries, &capabilities),
        ] {
            assert_ne!(
                left, right,
                "distinct domains must not collide into one digest"
            );
        }
        assert_eq!(
            FIXTURE_CORPUS.len(),
            FIXTURE_DOC_COUNT,
            "the pinned doc count must track the committed corpus"
        );
    }
}
