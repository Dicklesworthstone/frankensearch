//! Query classification for adaptive retrieval strategy.
//!
//! frankensearch adapts its retrieval strategy based on query type:
//!
//! | Query Class       | Example                              | Strategy          |
//! |--------------------|--------------------------------------|-------------------|
//! | `Empty`            | `""`                                 | Return empty      |
//! | `Identifier`       | `"bd-123"`, `"src/main.rs"`          | Lean lexical      |
//! | `ShortKeyword`     | `"error handling"`                   | Balanced          |
//! | `NaturalLanguage`  | `"how does the search work?"`        | Lean semantic     |
//!
//! Each class gets adaptive candidate budgets — identifiers fetch more lexical
//! candidates, natural language queries fetch more semantic candidates.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Classification of a search query by type.
///
/// Determines the retrieval budget allocation between lexical and semantic
/// search backends, and influences RRF fusion behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QueryClass {
    /// Empty or whitespace-only query. Returns empty results immediately.
    Empty,
    /// Looks like an identifier: file path, issue ID, function name, symbol.
    /// Lexical search is prioritized for exact-match capability.
    Identifier,
    /// Short keyword query (1-3 words, no question structure).
    /// Balanced between lexical and semantic retrieval.
    ShortKeyword,
    /// Natural language query (question or multi-word descriptive phrase).
    /// Semantic search is prioritized for meaning comprehension.
    NaturalLanguage,
}

impl QueryClass {
    /// Classify a query string into a `QueryClass`.
    ///
    /// Classification is based on heuristics (no ML model required):
    /// - Empty/whitespace → `Empty`
    /// - Contains path separators, `::`, dots-without-spaces, or ID patterns → `Identifier`
    /// - 1-3 words → `ShortKeyword`
    /// - 4+ words → `NaturalLanguage`
    #[must_use]
    pub fn classify(query: &str) -> Self {
        let trimmed = query.trim();
        if trimmed.is_empty() {
            return Self::Empty;
        }

        if Self::looks_like_identifier(trimmed) {
            return Self::Identifier;
        }

        let word_count = trimmed.split_whitespace().count();
        if word_count <= 3 {
            Self::ShortKeyword
        } else {
            Self::NaturalLanguage
        }
    }

    /// Heuristic check for identifier-like queries.
    fn looks_like_identifier(s: &str) -> bool {
        // Contains path separators
        if s.contains('/') || s.contains('\\') {
            return true;
        }

        // No spaces and contains dots or Rust path separators
        if !s.contains(' ') && (s.contains('.') || s.contains("::")) {
            return true;
        }

        // Issue/ticket ID pattern: prefix-digits (e.g., bd-123, JIRA-456)
        if !s.contains(' ') && s.contains('-') {
            let parts: Vec<&str> = s.splitn(2, '-').collect();
            if parts.len() == 2
                && parts[0].chars().all(|c| c.is_ascii_alphanumeric())
                && parts[1].chars().all(|c| c.is_ascii_digit())
                && !parts[0].is_empty()
                && !parts[1].is_empty()
            {
                return true;
            }
        }

        // Starts with common code prefixes
        if s.starts_with("fn ") || s.starts_with("struct ") || s.starts_with("impl ") {
            return true;
        }

        false
    }

    /// Suggested candidate multiplier for lexical search.
    ///
    /// Applied to `TwoTierConfig::candidate_multiplier` to produce the
    /// per-source candidate budget.
    #[must_use]
    pub const fn lexical_budget_multiplier(self) -> f32 {
        match self {
            Self::Empty => 0.0,
            Self::Identifier => 2.0,      // Lean heavily lexical
            Self::ShortKeyword => 1.0,    // Balanced
            Self::NaturalLanguage => 0.5, // Lean semantic
        }
    }

    /// Suggested candidate multiplier for semantic search.
    #[must_use]
    pub const fn semantic_budget_multiplier(self) -> f32 {
        match self {
            Self::Empty => 0.0,
            Self::Identifier => 0.5,      // Lean lexical
            Self::ShortKeyword => 1.0,    // Balanced
            Self::NaturalLanguage => 2.0, // Lean heavily semantic
        }
    }
}

impl fmt::Display for QueryClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "empty"),
            Self::Identifier => write!(f, "identifier"),
            Self::ShortKeyword => write!(f, "short_keyword"),
            Self::NaturalLanguage => write!(f, "natural_language"),
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    // ── Empty ───────────────────────────────────────────────────────────

    #[test]
    fn classify_empty_string() {
        assert_eq!(QueryClass::classify(""), QueryClass::Empty);
    }

    #[test]
    fn classify_whitespace_only() {
        assert_eq!(QueryClass::classify("   "), QueryClass::Empty);
        assert_eq!(QueryClass::classify("\t\n"), QueryClass::Empty);
    }

    // ── Identifier ──────────────────────────────────────────────────────

    #[test]
    fn classify_file_path() {
        assert_eq!(QueryClass::classify("src/main.rs"), QueryClass::Identifier);
        assert_eq!(
            QueryClass::classify("path/to/file.txt"),
            QueryClass::Identifier
        );
    }

    #[test]
    fn classify_issue_id() {
        assert_eq!(QueryClass::classify("bd-123"), QueryClass::Identifier);
        assert_eq!(QueryClass::classify("JIRA-456"), QueryClass::Identifier);
    }

    #[test]
    fn classify_hyphenated_keywords_as_short_keyword() {
        assert_eq!(
            QueryClass::classify("error-handling"),
            QueryClass::ShortKeyword
        );
        assert_eq!(
            QueryClass::classify("load-balancer"),
            QueryClass::ShortKeyword
        );
        assert_eq!(QueryClass::classify("bd-ab"), QueryClass::ShortKeyword);
    }

    #[test]
    fn classify_rust_path() {
        assert_eq!(
            QueryClass::classify("std::collections::HashMap"),
            QueryClass::Identifier
        );
    }

    #[test]
    fn classify_dotted_name() {
        assert_eq!(QueryClass::classify("config.toml"), QueryClass::Identifier);
    }

    #[test]
    fn classify_code_prefix() {
        assert_eq!(
            QueryClass::classify("fn search_query"),
            QueryClass::Identifier
        );
        assert_eq!(
            QueryClass::classify("struct TwoTierConfig"),
            QueryClass::Identifier
        );
    }

    // ── Short Keyword ───────────────────────────────────────────────────

    #[test]
    fn classify_single_word() {
        assert_eq!(QueryClass::classify("search"), QueryClass::ShortKeyword);
    }

    #[test]
    fn classify_two_words() {
        assert_eq!(
            QueryClass::classify("error handling"),
            QueryClass::ShortKeyword
        );
    }

    #[test]
    fn classify_three_words() {
        assert_eq!(
            QueryClass::classify("vector index search"),
            QueryClass::ShortKeyword
        );
    }

    // ── Natural Language ────────────────────────────────────────────────

    #[test]
    fn classify_question() {
        assert_eq!(
            QueryClass::classify("how does the search pipeline work?"),
            QueryClass::NaturalLanguage
        );
    }

    #[test]
    fn classify_long_phrase() {
        assert_eq!(
            QueryClass::classify("find all documents about distributed consensus"),
            QueryClass::NaturalLanguage
        );
    }

    // ── Budget Multipliers ──────────────────────────────────────────────

    #[test]
    fn identifier_leans_lexical() {
        assert!(
            QueryClass::Identifier.lexical_budget_multiplier()
                > QueryClass::Identifier.semantic_budget_multiplier()
        );
    }

    #[test]
    fn natural_language_leans_semantic() {
        assert!(
            QueryClass::NaturalLanguage.semantic_budget_multiplier()
                > QueryClass::NaturalLanguage.lexical_budget_multiplier()
        );
    }

    #[test]
    fn short_keyword_is_balanced() {
        assert!(
            (QueryClass::ShortKeyword.lexical_budget_multiplier()
                - QueryClass::ShortKeyword.semantic_budget_multiplier())
            .abs()
                < f32::EPSILON
        );
    }

    #[test]
    fn empty_has_zero_budgets() {
        assert!(QueryClass::Empty.lexical_budget_multiplier().abs() < f32::EPSILON);
        assert!(QueryClass::Empty.semantic_budget_multiplier().abs() < f32::EPSILON);
    }

    // ── Display ─────────────────────────────────────────────────────────

    #[test]
    fn display_all_variants() {
        assert_eq!(QueryClass::Empty.to_string(), "empty");
        assert_eq!(QueryClass::Identifier.to_string(), "identifier");
        assert_eq!(QueryClass::ShortKeyword.to_string(), "short_keyword");
        assert_eq!(QueryClass::NaturalLanguage.to_string(), "natural_language");
    }

    // ── Serialization ───────────────────────────────────────────────────

    #[test]
    fn serialization_roundtrip() {
        for variant in [
            QueryClass::Empty,
            QueryClass::Identifier,
            QueryClass::ShortKeyword,
            QueryClass::NaturalLanguage,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: QueryClass = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    // ── Property Invariants ───────────────────────────────────────────

    proptest! {
        #[test]
        fn classify_is_trim_invariant(query in ".{0,128}") {
            prop_assert_eq!(
                QueryClass::classify(&query),
                QueryClass::classify(query.trim()),
            );
        }

        #[test]
        fn budget_multipliers_are_consistent(query in ".{0,128}") {
            let class = QueryClass::classify(&query);
            let lexical = class.lexical_budget_multiplier();
            let semantic = class.semantic_budget_multiplier();

            prop_assert!(lexical.is_finite());
            prop_assert!(semantic.is_finite());
            prop_assert!(lexical >= 0.0);
            prop_assert!(semantic >= 0.0);

            if class == QueryClass::Empty {
                prop_assert!(query.trim().is_empty());
                prop_assert!(lexical.abs() < f32::EPSILON);
                prop_assert!(semantic.abs() < f32::EPSILON);
            } else {
                prop_assert!(!query.trim().is_empty());
                prop_assert!(lexical > 0.0 || semantic > 0.0);
            }
        }
    }
}
