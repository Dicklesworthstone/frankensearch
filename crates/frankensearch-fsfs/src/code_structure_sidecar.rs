//! Lightweight code-structure sidecar for symbol-aware search priors.
//!
//! The sidecar intentionally uses deterministic line scanners instead of
//! parser dependencies. If richer language parsing is added later, it should be
//! feature-gated and preserve this module's disabled/no-sidecar behavior.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::query_execution::{FusedCandidate, RankingPriorSignals};

/// Stable schema identifier for serialized code-structure sidecar artifacts.
pub const CODE_STRUCTURE_SIDECAR_SCHEMA_VERSION: &str = "fsfs-code-structure-sidecar-v1";

/// Deterministic rank contract for sidecar-adjusted candidate lists.
pub const CODE_STRUCTURE_TIE_BREAK_CONTRACT: &str =
    "sort(adjusted_score desc, base_score desc, doc_id asc)";

/// Default additive score weight used by [`CodeStructureSidecarConfig`].
pub const DEFAULT_CODE_STRUCTURE_WEIGHT: f64 = 0.012;

/// Default maximum additive sidecar boost.
pub const DEFAULT_CODE_STRUCTURE_MAX_BOOST: f64 = 0.02;

/// Supported lightweight scanner languages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodeStructureLanguage {
    Rust,
    Python,
    Markdown,
    Unknown,
}

/// Kinds of structure signals captured by the sidecar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodeStructureSignalKind {
    FilePath,
    ModuleName,
    Function,
    Class,
    Type,
    Import,
    MarkdownHeading,
}

/// One normalized structure signal for a document.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CodeStructureSignal {
    pub kind: CodeStructureSignalKind,
    pub value: String,
}

/// Sidecar row for a single indexed document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodeStructureDocument {
    pub doc_id: String,
    pub path: String,
    pub language: CodeStructureLanguage,
    pub signals: Vec<CodeStructureSignal>,
}

/// Deterministic optional sidecar index keyed by document id.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodeStructureSidecar {
    pub schema_version: String,
    pub documents: BTreeMap<String, CodeStructureDocument>,
}

/// Sidecar ranking controls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CodeStructureSidecarConfig {
    pub enabled: bool,
    pub weight: f64,
    pub max_boost: f64,
}

impl Default for CodeStructureSidecarConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            weight: DEFAULT_CODE_STRUCTURE_WEIGHT,
            max_boost: DEFAULT_CODE_STRUCTURE_MAX_BOOST,
        }
    }
}

impl CodeStructureSidecarConfig {
    #[must_use]
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            weight: 0.0,
            max_boost: DEFAULT_CODE_STRUCTURE_MAX_BOOST,
        }
    }

    #[must_use]
    pub fn normalized(self) -> Self {
        let weight = if self.weight.is_finite() && self.weight >= 0.0 {
            self.weight
        } else {
            0.0
        };
        let max_boost = if self.max_boost.is_finite() && self.max_boost > 0.0 {
            self.max_boost
        } else {
            DEFAULT_CODE_STRUCTURE_MAX_BOOST
        };
        Self {
            enabled: self.enabled,
            weight,
            max_boost,
        }
    }

    #[must_use]
    fn boost_for(self, signal_score: f64) -> f64 {
        let config = self.normalized();
        if !config.enabled {
            return 0.0;
        }
        (sanitize_signal_score(signal_score) * config.weight).clamp(0.0, config.max_boost)
    }
}

/// Matched signal evidence attached to sidecar ranking decisions.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CodeStructureMatchedSignal {
    pub kind: CodeStructureSignalKind,
    pub value: String,
    pub matched_token: String,
}

/// Query/document sidecar score with auditable reason code.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodeStructureMatchEvidence {
    pub doc_id: String,
    pub score: f64,
    pub reason_code: String,
    pub matched_signals: Vec<CodeStructureMatchedSignal>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PreparedCodeStructureQuery {
    tokens: BTreeSet<String>,
    normalized: String,
}

/// Candidate row after optional sidecar adjustment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodeStructureRankedCandidate {
    pub doc_id: String,
    pub base_score: f64,
    pub adjusted_score: f64,
    pub sidecar_boost: f64,
    pub evidence: CodeStructureMatchEvidence,
}

impl CodeStructureSidecar {
    #[must_use]
    pub fn new() -> Self {
        Self {
            schema_version: CODE_STRUCTURE_SIDECAR_SCHEMA_VERSION.to_owned(),
            documents: BTreeMap::new(),
        }
    }

    #[must_use]
    pub fn from_documents<'a>(docs: impl IntoIterator<Item = (&'a str, &'a str, &'a str)>) -> Self {
        let mut sidecar = Self::new();
        for (doc_id, path, content) in docs {
            sidecar.insert_document(doc_id, path, content);
        }
        sidecar
    }

    pub fn insert_document(
        &mut self,
        doc_id: impl Into<String>,
        path: impl AsRef<str>,
        content: impl AsRef<str>,
    ) {
        let doc_id = doc_id.into();
        let path = path.as_ref();
        let language = detect_language(path);
        let signals = extract_code_structure_signals(path, content.as_ref());
        let document = CodeStructureDocument {
            doc_id: doc_id.clone(),
            path: normalize_signal_value(path),
            language,
            signals,
        };
        self.documents.insert(doc_id, document);
    }

    #[must_use]
    pub fn score_query(&self, query: &str, doc_id: &str) -> CodeStructureMatchEvidence {
        let Some(document) = self.documents.get(doc_id) else {
            return CodeStructureMatchEvidence {
                doc_id: doc_id.to_owned(),
                score: 0.0,
                reason_code: "code_structure.no_document".to_owned(),
                matched_signals: Vec::new(),
            };
        };
        score_document(query, document)
    }

    #[must_use]
    pub fn prior_signals_for_candidates(
        &self,
        query: &str,
        candidates: &[FusedCandidate],
    ) -> HashMap<String, RankingPriorSignals> {
        let prepared_query = prepare_query(query);
        let mut signals = HashMap::with_capacity(candidates.len());
        for candidate in candidates {
            let Some(document) = self.documents.get(&candidate.doc_id) else {
                continue;
            };
            let evidence = score_document_prepared(&prepared_query, document);
            if evidence.score > 0.0 {
                signals.insert(
                    candidate.doc_id.clone(),
                    RankingPriorSignals::default().with_code_structure(Some(evidence.score)),
                );
            }
        }
        signals
    }

    #[must_use]
    pub fn rank_candidates(
        &self,
        query: &str,
        candidates: &[FusedCandidate],
        config: CodeStructureSidecarConfig,
    ) -> Vec<CodeStructureRankedCandidate> {
        let config = config.normalized();
        let prepared_query = config.enabled.then(|| prepare_query(query));
        let mut ranked: Vec<CodeStructureRankedCandidate> = candidates
            .iter()
            .map(|candidate| {
                let evidence = prepared_query.as_ref().map_or_else(
                    || CodeStructureMatchEvidence {
                        doc_id: candidate.doc_id.clone(),
                        score: 0.0,
                        reason_code: "code_structure.disabled".to_owned(),
                        matched_signals: Vec::new(),
                    },
                    |prepared_query| {
                        self.documents.get(&candidate.doc_id).map_or_else(
                            || CodeStructureMatchEvidence {
                                doc_id: candidate.doc_id.clone(),
                                score: 0.0,
                                reason_code: "code_structure.no_document".to_owned(),
                                matched_signals: Vec::new(),
                            },
                            |document| score_document_prepared(prepared_query, document),
                        )
                    },
                );
                let sidecar_boost = config.boost_for(evidence.score);
                CodeStructureRankedCandidate {
                    doc_id: candidate.doc_id.clone(),
                    base_score: candidate.fused_score,
                    adjusted_score: candidate.fused_score + sidecar_boost,
                    sidecar_boost,
                    evidence,
                }
            })
            .collect();

        if config.enabled {
            ranked.sort_by(code_structure_candidate_cmp);
        }
        ranked
    }
}

impl Default for CodeStructureSidecar {
    fn default() -> Self {
        Self::new()
    }
}

#[must_use]
pub fn detect_language(path: &str) -> CodeStructureLanguage {
    match Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("rs") => CodeStructureLanguage::Rust,
        Some("py") => CodeStructureLanguage::Python,
        Some("md" | "markdown") => CodeStructureLanguage::Markdown,
        _ => CodeStructureLanguage::Unknown,
    }
}

#[must_use]
pub fn extract_code_structure_signals(path: &str, content: &str) -> Vec<CodeStructureSignal> {
    let language = detect_language(path);
    let mut signals = BTreeSet::new();
    push_signal(&mut signals, CodeStructureSignalKind::FilePath, path);
    if let Some(stem) = Path::new(path).file_stem().and_then(|stem| stem.to_str()) {
        push_signal(&mut signals, CodeStructureSignalKind::ModuleName, stem);
    }

    match language {
        CodeStructureLanguage::Rust => extract_rust_signals(content, &mut signals),
        CodeStructureLanguage::Python => extract_python_signals(content, &mut signals),
        CodeStructureLanguage::Markdown => extract_markdown_signals(content, &mut signals),
        CodeStructureLanguage::Unknown => {}
    }

    signals.into_iter().collect()
}

fn extract_rust_signals(content: &str, signals: &mut BTreeSet<CodeStructureSignal>) {
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("//") {
            continue;
        }
        if let Some(import) = rust_import_value(trimmed) {
            push_signal(signals, CodeStructureSignalKind::Import, import);
        }
        if let Some(name) = rust_name_after_keyword(trimmed, "mod") {
            push_signal(signals, CodeStructureSignalKind::ModuleName, &name);
        }
        if let Some(name) = rust_name_after_keyword(trimmed, "fn") {
            push_signal(signals, CodeStructureSignalKind::Function, &name);
        }
        for keyword in ["struct", "enum", "trait"] {
            if let Some(name) = rust_name_after_keyword(trimmed, keyword) {
                push_signal(signals, CodeStructureSignalKind::Type, &name);
            }
        }
    }
}

fn extract_python_signals(content: &str, signals: &mut BTreeSet<CodeStructureSignal>) {
    for line in content.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with('#') {
            continue;
        }
        if let Some(name) = python_decl_name(trimmed, "class ") {
            push_signal(signals, CodeStructureSignalKind::Class, &name);
        }
        if let Some(name) =
            python_decl_name(trimmed, "def ").or_else(|| python_decl_name(trimmed, "async def "))
        {
            push_signal(signals, CodeStructureSignalKind::Function, &name);
        }
        if trimmed.starts_with("import ") || trimmed.starts_with("from ") {
            let import = trimmed.trim_end_matches(':').trim();
            push_signal(signals, CodeStructureSignalKind::Import, import);
        }
    }
}

fn extract_markdown_signals(content: &str, signals: &mut BTreeSet<CodeStructureSignal>) {
    for line in content.lines() {
        let trimmed = line.trim_start();
        let heading_level = trimmed.chars().take_while(|&ch| ch == '#').count();
        if (1..=6).contains(&heading_level)
            && trimmed.as_bytes().get(heading_level) == Some(&b' ')
            && let Some(heading) = trimmed.get(heading_level..)
        {
            push_signal(
                signals,
                CodeStructureSignalKind::MarkdownHeading,
                heading.trim(),
            );
        }
    }
}

fn push_signal(
    signals: &mut BTreeSet<CodeStructureSignal>,
    kind: CodeStructureSignalKind,
    value: &str,
) {
    let value = normalize_signal_value(value);
    if !value.is_empty() {
        signals.insert(CodeStructureSignal { kind, value });
    }
}

fn rust_import_value(line: &str) -> Option<&str> {
    line.strip_prefix("use ")
        .or_else(|| line.strip_prefix("pub use "))
        .map(|rest| rest.trim_end_matches(';').trim())
        .filter(|rest| !rest.is_empty())
}

fn rust_name_after_keyword(line: &str, keyword: &str) -> Option<String> {
    let mut parts = line
        .split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
        .filter(|part| !part.is_empty());
    while let Some(part) = parts.next() {
        if part == keyword {
            return parts.next().map(ToOwned::to_owned);
        }
    }
    None
}

fn python_decl_name(line: &str, prefix: &str) -> Option<String> {
    line.strip_prefix(prefix)
        .and_then(|rest| {
            rest.split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
                .next()
        })
        .filter(|name| !name.is_empty())
        .map(ToOwned::to_owned)
}

fn prepare_query(query: &str) -> PreparedCodeStructureQuery {
    PreparedCodeStructureQuery {
        tokens: tokenize(query),
        normalized: normalize_signal_value(query),
    }
}

fn score_document(query: &str, document: &CodeStructureDocument) -> CodeStructureMatchEvidence {
    let prepared_query = prepare_query(query);
    score_document_prepared(&prepared_query, document)
}

fn score_document_prepared(
    prepared_query: &PreparedCodeStructureQuery,
    document: &CodeStructureDocument,
) -> CodeStructureMatchEvidence {
    if prepared_query.tokens.is_empty() {
        return CodeStructureMatchEvidence {
            doc_id: document.doc_id.clone(),
            score: 0.0,
            reason_code: "code_structure.empty_query".to_owned(),
            matched_signals: Vec::new(),
        };
    }

    let mut raw_score = 0.0;
    let mut matched = BTreeSet::new();
    // Reused across signals: the per-token lowercase scratch is allocated once for
    // the whole document rather than once per signal token.
    let mut token_scratch = String::new();

    for signal in &document.signals {
        if let Some(token) =
            smallest_matching_token(&prepared_query.tokens, &signal.value, &mut token_scratch)
        {
            raw_score += signal_weight(signal.kind);
            matched.insert(CodeStructureMatchedSignal {
                kind: signal.kind,
                value: signal.value.clone(),
                matched_token: token,
            });
        }
        if prepared_query.normalized.len() >= 3 && signal.value.contains(&prepared_query.normalized)
        {
            raw_score += 0.2;
        }
    }

    let matched_signals: Vec<CodeStructureMatchedSignal> = matched.into_iter().collect();
    let score = sanitize_signal_score(raw_score);
    let reason_code = if matched_signals.is_empty() {
        "code_structure.no_match"
    } else {
        "code_structure.match"
    };

    CodeStructureMatchEvidence {
        doc_id: document.doc_id.clone(),
        score,
        reason_code: reason_code.to_owned(),
        matched_signals,
    }
}

fn code_structure_candidate_cmp(
    left: &CodeStructureRankedCandidate,
    right: &CodeStructureRankedCandidate,
) -> Ordering {
    right
        .adjusted_score
        .total_cmp(&left.adjusted_score)
        .then_with(|| right.base_score.total_cmp(&left.base_score))
        .then_with(|| left.doc_id.cmp(&right.doc_id))
}

#[must_use]
fn signal_weight(kind: CodeStructureSignalKind) -> f64 {
    match kind {
        CodeStructureSignalKind::FilePath => 0.20,
        CodeStructureSignalKind::ModuleName => 0.35,
        CodeStructureSignalKind::Function => 0.60,
        CodeStructureSignalKind::Class => 0.60,
        CodeStructureSignalKind::Type => 0.50,
        CodeStructureSignalKind::Import => 0.35,
        CodeStructureSignalKind::MarkdownHeading => 0.45,
    }
}

fn normalize_signal_value(value: &str) -> String {
    // Single pass into one `String`, byte-identical to the former
    // `split_whitespace().collect::<Vec<_>>().join(" ").trim().to_ascii_lowercase()`
    // (three allocations): collapse `char::is_whitespace` runs to a single ASCII
    // space, strip leading/trailing whitespace, ASCII-lowercase each char.
    // `split_whitespace` and `char::is_whitespace` share the Unicode White_Space
    // definition, so token boundaries match exactly. Output bytes <= input bytes,
    // so the pre-reserved capacity never reallocates. (~1.12×, `normalize_signal_ab`.)
    let mut out = String::with_capacity(value.len());
    let mut pending_space = false;
    for ch in value.chars() {
        if ch.is_whitespace() {
            pending_space = true;
        } else {
            if pending_space && !out.is_empty() {
                out.push(' ');
            }
            pending_space = false;
            out.push(ch.to_ascii_lowercase());
        }
    }
    out
}

fn tokenize(value: &str) -> BTreeSet<String> {
    let mut tokens = BTreeSet::new();
    let mut current = String::new();
    // ASCII fast path (the common case for code symbols/queries): a byte loop with
    // `to_ascii_lowercase` avoids the per-char `chars().flat_map(char::to_lowercase)`
    // overhead (`char::to_lowercase` builds a `ToLowercase` iterator for every char,
    // even though only `is_ascii_alphanumeric` chars survive). Bit-identical for
    // ASCII input: for an ASCII char, `char::to_lowercase` yields exactly its
    // ASCII lowercase. Non-ASCII falls back to the Unicode path (which still
    // catches the rare char that *lowercases* to ASCII, e.g. the Kelvin sign).
    if value.is_ascii() {
        for &b in value.as_bytes() {
            let lowered = b.to_ascii_lowercase();
            if lowered.is_ascii_alphanumeric() {
                current.push(lowered as char);
            } else if !current.is_empty() {
                tokens.insert(std::mem::take(&mut current));
            }
        }
        if !current.is_empty() {
            tokens.insert(current);
        }
        return tokens;
    }
    for ch in value.chars().flat_map(char::to_lowercase) {
        if ch.is_ascii_alphanumeric() {
            current.push(ch);
        } else if !current.is_empty() {
            tokens.insert(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.insert(current);
    }
    tokens
}

/// The lexicographically smallest token of `value` that also occurs in
/// `query_tokens`, or `None`. This is exactly `tokenize(value)`-then-
/// `query_tokens.intersection(..).next()`, but it streams `value`'s tokens through
/// the reused `scratch` buffer and probes `query_tokens` directly, so it never
/// materialises the intermediate `BTreeSet<String>` nor its per-token heap
/// allocations (a token is cloned only when it becomes the new smallest match).
/// Bit-identical to that pair: the minimum of the intersection is unaffected by
/// the set's deduplication, so streaming (with possible duplicates) and taking the
/// min over query-hits yields the same token; non-ASCII input delegates to the
/// exact reference computation. Measured ~1.8× on the per-document signal loop
/// (`code_signal_probe_ab` bench).
fn smallest_matching_token(
    query_tokens: &BTreeSet<String>,
    value: &str,
    scratch: &mut String,
) -> Option<String> {
    if !value.is_ascii() {
        let signal_tokens = tokenize(value);
        return query_tokens.intersection(&signal_tokens).next().cloned();
    }
    let mut best: Option<String> = None;
    scratch.clear();
    for &b in value.as_bytes() {
        let lowered = b.to_ascii_lowercase();
        if lowered.is_ascii_alphanumeric() {
            scratch.push(lowered as char);
        } else if !scratch.is_empty() {
            consider_token(&mut best, scratch.as_str(), query_tokens);
            scratch.clear();
        }
    }
    if !scratch.is_empty() {
        consider_token(&mut best, scratch.as_str(), query_tokens);
    }
    best
}

#[inline]
fn consider_token(best: &mut Option<String>, current: &str, query_tokens: &BTreeSet<String>) {
    if query_tokens.contains(current) && best.as_deref().is_none_or(|best| current < best) {
        *best = Some(current.to_owned());
    }
}

fn sanitize_signal_score(value: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PressureProfile;
    use crate::query_execution::{
        LexicalCandidate, QueryExecutionOrchestrator, RankingPriorTuning, SemanticCandidate,
    };

    /// Reference: the pre-fast-path pure Unicode `tokenize`.
    fn tokenize_reference(value: &str) -> std::collections::BTreeSet<String> {
        let mut tokens = std::collections::BTreeSet::new();
        let mut current = String::new();
        for ch in value.chars().flat_map(char::to_lowercase) {
            if ch.is_ascii_alphanumeric() {
                current.push(ch);
            } else if !current.is_empty() {
                tokens.insert(std::mem::take(&mut current));
            }
        }
        if !current.is_empty() {
            tokens.insert(current);
        }
        tokens
    }

    #[test]
    fn tokenize_ascii_fastpath_matches_unicode_reference() {
        let inputs = [
            "",
            "rank_symbols",
            "pub fn run_fast(x: i32)",
            "FooBar BAZ-123 qux.quux",
            "src/main.rs -> async def rank()",
            "café Σigma naïve",    // non-ASCII letters: fallback
            "日本語 token test42", // CJK + ASCII: fallback
            "\u{212A}elvin sign",  // KELVIN SIGN lowercases to ASCII 'k' (fallback must keep it)
        ];
        for input in inputs {
            assert_eq!(
                super::tokenize(input),
                tokenize_reference(input),
                "tokenize fast path diverged for {input:?}"
            );
        }
    }

    #[test]
    fn smallest_matching_token_matches_intersection_reference() {
        // The streaming probe must return exactly what
        // `tokenize(value).intersection(query).next()` (the smallest common token)
        // would, for every query/signal combination — including non-ASCII fallback,
        // no-match, single-token, and duplicate-token signals.
        let query_sets = [
            vec!["auth", "user", "token", "session", "login"],
            vec!["rank", "symbols"],
            vec![], // empty query
            vec!["zzz_never_matches"],
            vec!["k"], // exercises the Kelvin-sign fallback
        ];
        let values = [
            "",
            "authenticate_user_session",
            "src/auth/login_handler.rs",
            "NoMatchHereJustFiller",
            "rank_symbols rank_symbols", // duplicate token
            "café Σigma auth naïve",     // non-ASCII: fallback path
            "\u{212A}elvin token",       // KELVIN SIGN → ascii 'k' (fallback)
            "user",                      // single token
        ];
        for qs in &query_sets {
            let query: BTreeSet<String> = qs.iter().map(|s| (*s).to_owned()).collect();
            let mut scratch = String::new();
            for value in values {
                let signal_tokens = tokenize(value);
                let expected = query.intersection(&signal_tokens).next().cloned();
                let got = smallest_matching_token(&query, value, &mut scratch);
                assert_eq!(expected, got, "diverged for query={qs:?} value={value:?}");
            }
        }
    }

    #[test]
    fn normalize_signal_value_matches_three_alloc_reference() {
        // Byte-identical to the former three-allocation
        // `split_whitespace().collect().join(" ").trim().to_ascii_lowercase()`.
        fn reference(value: &str) -> String {
            value
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
                .trim()
                .to_ascii_lowercase()
        }
        let cases = [
            "",
            "   ",
            "src/auth/login_handler.rs",
            "use   std::collections :: HashMap",
            "  ##   Authentication  Guide   ",
            "\ttrait   TokenStore\t",
            "MixedCase_Ident",
            "café  Σigma  NAÏVE", // non-ASCII: only ASCII is lowercased
            "line\u{000B}with\u{000B}vtab", // vertical tab is Unicode whitespace
            "trailing and leading  \n\t ",
        ];
        for value in cases {
            assert_eq!(
                normalize_signal_value(value),
                reference(value),
                "normalize diverged for {value:?}"
            );
        }
    }

    fn candidate(doc_id: &str, score: f64) -> FusedCandidate {
        FusedCandidate {
            doc_id: doc_id.to_owned(),
            fused_score: score,
            prior_boost: 0.0,
            lexical_rank: None,
            semantic_rank: None,
            hash_rank: None,
            lexical_score: None,
            semantic_score: None,
            hash_score: None,
            in_both_sources: false,
        }
    }

    #[test]
    fn extracts_rust_python_and_markdown_structure_fixtures() {
        let sidecar = CodeStructureSidecar::from_documents([
            (
                "src/lib.rs",
                "src/lib.rs",
                r"
                    use std::collections::HashMap;
                    mod ranking_priors;
                    pub struct SearchIndex {}
                    pub fn execute_search() {}
                ",
            ),
            (
                "tools/search.py",
                "tools/search.py",
                r"
                    from pathlib import Path
                    class SearchService:
                        async def rank_symbols(self): pass
                ",
            ),
            (
                "docs/guide.md",
                "docs/guide.md",
                "# Operator Search Guide\n\n## Ranking Controls\n",
            ),
        ]);

        let rust = sidecar.documents.get("src/lib.rs").unwrap();
        assert_eq!(rust.language, CodeStructureLanguage::Rust);
        assert!(
            rust.signals
                .iter()
                .any(|signal| signal.kind == CodeStructureSignalKind::Function
                    && signal.value == "execute_search")
        );
        assert!(
            rust.signals
                .iter()
                .any(|signal| signal.kind == CodeStructureSignalKind::Import
                    && signal.value == "std::collections::hashmap")
        );

        let python = sidecar.documents.get("tools/search.py").unwrap();
        assert_eq!(python.language, CodeStructureLanguage::Python);
        assert!(
            python
                .signals
                .iter()
                .any(|signal| signal.kind == CodeStructureSignalKind::Class
                    && signal.value == "searchservice")
        );
        assert!(
            python
                .signals
                .iter()
                .any(|signal| signal.kind == CodeStructureSignalKind::Function
                    && signal.value == "rank_symbols")
        );

        let markdown = sidecar.documents.get("docs/guide.md").unwrap();
        assert_eq!(markdown.language, CodeStructureLanguage::Markdown);
        assert!(markdown.signals.iter().any(|signal| signal.kind
            == CodeStructureSignalKind::MarkdownHeading
            && signal.value == "ranking controls"));
    }

    #[test]
    fn disabled_sidecar_preserves_candidate_order_and_scores() {
        let sidecar = CodeStructureSidecar::from_documents([(
            "src/rank.rs",
            "src/rank.rs",
            "pub fn rank_symbols() {}",
        )]);
        let candidates = vec![candidate("doc-a", 0.2), candidate("src/rank.rs", 0.1)];

        let ranked = sidecar.rank_candidates(
            "rank symbols",
            &candidates,
            CodeStructureSidecarConfig::disabled(),
        );

        let ids: Vec<&str> = ranked.iter().map(|row| row.doc_id.as_str()).collect();
        assert_eq!(ids, vec!["doc-a", "src/rank.rs"]);
        assert!(ranked.iter().all(|row| row.sidecar_boost == 0.0));
        assert!((ranked[1].adjusted_score - candidates[1].fused_score).abs() < f64::EPSILON);
    }

    #[test]
    fn enabled_sidecar_promotes_symbol_match_with_evidence() {
        let sidecar = CodeStructureSidecar::from_documents([(
            "src/rank.rs",
            "src/rank.rs",
            "pub fn rank_symbols() {}",
        )]);
        let candidates = vec![candidate("doc-a", 0.2), candidate("src/rank.rs", 0.19)];
        let config = CodeStructureSidecarConfig {
            enabled: true,
            weight: 0.05,
            max_boost: 0.05,
        };

        let ranked = sidecar.rank_candidates("rank symbols", &candidates, config);

        assert_eq!(ranked[0].doc_id, "src/rank.rs");
        assert!(ranked[0].sidecar_boost > 0.0);
        assert_eq!(ranked[0].evidence.reason_code, "code_structure.match");
        assert!(
            ranked[0]
                .evidence
                .matched_signals
                .iter()
                .any(|signal| signal.kind == CodeStructureSignalKind::Function
                    && signal.value == "rank_symbols")
        );
    }

    #[test]
    fn sidecar_prior_signals_feed_existing_fusion_priors() {
        let sidecar = CodeStructureSidecar::from_documents([(
            "doc-b",
            "src/rank.rs",
            "pub fn rank_symbols() {}",
        )]);
        let orchestrator = QueryExecutionOrchestrator::default();
        let lexical = vec![
            LexicalCandidate::new("doc-a", 1.0),
            LexicalCandidate::new("doc-b", 1.0),
        ];
        let semantic: Vec<SemanticCandidate> = Vec::new();
        let base = orchestrator.fuse_rankings(&lexical, &semantic, 10, 0);
        assert_eq!(base[0].doc_id, "doc-a");

        let signals = sidecar.prior_signals_for_candidates("rank symbols", &base);
        let tuning = RankingPriorTuning {
            max_total_boost: 0.01,
            ..RankingPriorTuning::for_profile(PressureProfile::Performance)
        };
        let boosted =
            orchestrator.fuse_rankings_with_priors(&lexical, &semantic, 10, 0, &signals, tuning);

        assert_eq!(boosted[0].doc_id, "doc-b");
        assert!(boosted[0].prior_boost > 0.0);
    }

    #[test]
    fn prior_signals_match_legacy_per_candidate_scoring() {
        let sidecar = CodeStructureSidecar::from_documents([
            ("doc-a", "src/rank.rs", "pub fn rank_symbols() {}"),
            ("doc-b", "docs/rank.md", "# Ranking Symbols\n"),
        ]);
        let candidates = vec![
            candidate("doc-a", 0.4),
            candidate("missing", 0.3),
            candidate("doc-b", 0.2),
        ];
        let query = "rank symbols";
        let mut legacy = HashMap::new();
        for candidate in &candidates {
            let evidence = sidecar.score_query(query, &candidate.doc_id);
            if evidence.score > 0.0 {
                legacy.insert(
                    candidate.doc_id.clone(),
                    RankingPriorSignals::default().with_code_structure(Some(evidence.score)),
                );
            }
        }

        assert_eq!(
            sidecar.prior_signals_for_candidates(query, &candidates),
            legacy
        );
    }

    #[test]
    fn enabled_sidecar_tie_breaks_equal_adjusted_scores_by_doc_id() {
        let sidecar = CodeStructureSidecar::new();
        let candidates = vec![candidate("doc-z", 0.3), candidate("doc-a", 0.3)];

        let ranked = sidecar.rank_candidates(
            "missing",
            &candidates,
            CodeStructureSidecarConfig::default(),
        );

        let ids: Vec<&str> = ranked.iter().map(|row| row.doc_id.as_str()).collect();
        assert_eq!(ids, vec!["doc-a", "doc-z"]);
        assert!(ranked.iter().all(|row| row.evidence.score == 0.0));
    }
}
