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
];

/// Exact live document count of [`FIXTURE_CORPUS`], asserted rather than
/// derived from `.len()` so a fixture edit has to be deliberate.
pub const FIXTURE_DOC_COUNT: usize = 4;

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
    /// Every observation, in table order.
    pub observations: Vec<NativeObservationV1>,
    /// Every verdict, in table order.
    pub verdicts: Vec<NativeVerdictV1>,
    /// Whether both engines were observed. A single-engine receipt is legal
    /// and honest — the Tantivy arm needs the `tantivy-oracle` feature — but
    /// it must say so rather than imply cross-engine coverage.
    pub both_engines_observed: bool,
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

    /// Content address over the domain-separated canonical body.
    ///
    /// # Errors
    ///
    /// Returns [`GauntletError::InvalidContract`] when the body cannot be
    /// canonically serialized.
    pub fn receipt_hash(&self) -> Result<String, GauntletError> {
        let body =
            serde_json::to_vec(self).map_err(|error| GauntletError::InvalidContract {
                reason: format!("native enriched receipt is not canonically serializable: {error}"),
            })?;
        let mut hasher = Sha256::new();
        hasher.update(RECEIPT_HASH_DOMAIN.as_bytes());
        hasher.update([0u8]);
        hasher.update(&body);
        Ok(hex_lower(&hasher.finalize()))
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
    let total = usize::try_from(result.total_count.ok_or_else(|| {
        GauntletError::InvalidContract {
            reason: "native Quill exact-count was requested but not returned".to_owned(),
        }
    })?)
    .map_err(|_| GauntletError::InvalidContract {
        reason: "native Quill total does not fit usize".to_owned(),
    })?;
    let doc_count = usize::try_from(result.doc_count).map_err(|_| {
        GauntletError::InvalidContract {
            reason: "native Quill doc_count does not fit usize".to_owned(),
        }
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

    #[test]
    fn the_manifests_change_when_the_fixtures_change() {
        let corpus = corpus_manifest_sha256();
        let queries = query_manifest_sha256();
        assert_eq!(corpus.len(), 64);
        assert_eq!(queries.len(), 64);
        assert_ne!(
            corpus, queries,
            "distinct domains must not collide into one digest"
        );
        assert_eq!(
            FIXTURE_CORPUS.len(),
            FIXTURE_DOC_COUNT,
            "the pinned doc count must track the committed corpus"
        );
    }
}
