//! Negative/exclusion query syntax parser.
//!
//! Supports two negation syntaxes:
//!
//! - **Dash prefix:** `-term` excludes a single word
//! - **NOT keyword:** `NOT "exact phrase"` excludes a quoted phrase
//!
//! # Examples
//!
//! ```
//! use frankensearch_core::parsed_query::ParsedQuery;
//!
//! let q = ParsedQuery::parse("rust async -tokio");
//! assert_eq!(q.positive, "rust async");
//! assert_eq!(q.negative_terms, vec!["tokio"]);
//!
//! let q = ParsedQuery::parse(r#"machine learning NOT "deep learning""#);
//! assert_eq!(q.positive, "machine learning");
//! assert_eq!(q.negative_phrases, vec!["deep learning"]);
//! ```

use serde::{Deserialize, Serialize};

/// A parsed query with positive content and optional negations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParsedQuery {
    /// The main query text (what to search for).
    pub positive: String,
    /// Individual terms to exclude (from `-term` syntax).
    pub negative_terms: Vec<String>,
    /// Exact phrases to exclude (from `NOT "phrase"` syntax).
    pub negative_phrases: Vec<String>,
}

impl ParsedQuery {
    /// Parse a raw query string into positive and negative components.
    ///
    /// # Syntax
    ///
    /// - `-term` : exclude documents containing "term"
    /// - `NOT "exact phrase"` : exclude documents containing exact phrase
    /// - `\-term` : literal dash (escaped, not negation)
    /// - Multiple exclusions are supported: `query -foo -bar`
    #[must_use]
    pub fn parse(raw: &str) -> Self {
        let mut positive_parts = Vec::new();
        let mut negative_terms = Vec::new();
        let mut negative_phrases = Vec::new();

        let chars: Vec<char> = raw.chars().collect();
        let len = chars.len();
        let mut i = 0;

        while i < len {
            // Skip whitespace.
            if chars[i].is_whitespace() {
                i += 1;
                continue;
            }

            // Check for NOT keyword followed by quoted phrase.
            if i + 3 < len && matches_not_keyword(&chars, i) {
                // Advance past "NOT".
                let after_not = i + 3;
                // Skip whitespace between NOT and the quote.
                let mut j = after_not;
                while j < len && chars[j].is_whitespace() {
                    j += 1;
                }
                if j < len && chars[j] == '"' {
                    // Parse quoted phrase.
                    j += 1; // skip opening quote
                    let start = j;
                    while j < len && chars[j] != '"' {
                        j += 1;
                    }
                    let phrase: String = chars[start..j].iter().collect();
                    if !phrase.is_empty() {
                        negative_phrases.push(phrase);
                    }
                    if j < len {
                        j += 1; // skip closing quote
                    }
                    i = j;
                    continue;
                }
                // NOT without a following quote — treat as regular word.
            }

            // Check for escaped dash: \-
            if chars[i] == '\\' && i + 1 < len && chars[i + 1] == '-' {
                // Collect the rest of the token as a literal word starting with dash.
                i += 1; // skip backslash
                let start = i;
                while i < len && !chars[i].is_whitespace() {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                positive_parts.push(word);
                continue;
            }

            // Check for negation dash: -term
            // Must be preceded by whitespace or be at start of string.
            if chars[i] == '-'
                && (i == 0 || chars[i - 1].is_whitespace())
                && i + 1 < len
                && chars[i + 1] != '-'
                && !chars[i + 1].is_whitespace()
            {
                i += 1; // skip dash
                let start = i;
                while i < len && !chars[i].is_whitespace() {
                    i += 1;
                }
                let term: String = chars[start..i].iter().collect();
                if !term.is_empty() {
                    negative_terms.push(term);
                }
                continue;
            }

            // Regular word: collect until whitespace.
            let start = i;
            while i < len && !chars[i].is_whitespace() {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            positive_parts.push(word);
        }

        let positive = positive_parts.join(" ");

        Self {
            positive,
            negative_terms,
            negative_phrases,
        }
    }

    /// Whether this query contains any negations.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn has_negations(&self) -> bool {
        !self.negative_terms.is_empty() || !self.negative_phrases.is_empty()
    }

    /// Whether the positive portion is empty (only negations).
    #[must_use]
    #[allow(clippy::missing_const_for_fn)] // str::trim() is not const
    pub fn is_positive_empty(&self) -> bool {
        self.positive.trim().is_empty()
    }

    /// Total number of negative components (terms + phrases).
    #[must_use]
    pub const fn negation_count(&self) -> usize {
        self.negative_terms.len() + self.negative_phrases.len()
    }
}

/// Check if the characters at position `i` match "NOT" (case-insensitive)
/// followed by whitespace or end of input.
fn matches_not_keyword(chars: &[char], i: usize) -> bool {
    let len = chars.len();
    if i + 3 > len {
        return false;
    }
    let word: String = chars[i..i + 3].iter().collect();
    if !word.eq_ignore_ascii_case("NOT") {
        return false;
    }
    // Must be at the start or preceded by whitespace.
    if i > 0 && !chars[i - 1].is_whitespace() {
        return false;
    }
    // Must be followed by whitespace or end.
    i + 3 >= len || chars[i + 3].is_whitespace()
}

impl std::fmt::Display for ParsedQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.positive)?;
        for term in &self.negative_terms {
            write!(f, " -{term}")?;
        }
        for phrase in &self.negative_phrases {
            write!(f, " NOT \"{phrase}\"")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Basic parsing ──────────────────────────────────────────────────

    #[test]
    fn parse_no_negations() {
        let q = ParsedQuery::parse("hello world");
        assert_eq!(q.positive, "hello world");
        assert!(q.negative_terms.is_empty());
        assert!(q.negative_phrases.is_empty());
        assert!(!q.has_negations());
    }

    #[test]
    fn parse_single_negative_term() {
        let q = ParsedQuery::parse("rust async -tokio");
        assert_eq!(q.positive, "rust async");
        assert_eq!(q.negative_terms, vec!["tokio"]);
        assert!(q.has_negations());
    }

    #[test]
    fn parse_multiple_negative_terms() {
        let q = ParsedQuery::parse("search engine -spam -ads -bot");
        assert_eq!(q.positive, "search engine");
        assert_eq!(q.negative_terms, vec!["spam", "ads", "bot"]);
    }

    #[test]
    fn parse_not_phrase() {
        let q = ParsedQuery::parse(r#"machine learning NOT "deep learning""#);
        assert_eq!(q.positive, "machine learning");
        assert!(q.negative_terms.is_empty());
        assert_eq!(q.negative_phrases, vec!["deep learning"]);
    }

    #[test]
    fn parse_mixed_negations() {
        let q = ParsedQuery::parse(r#"rust web framework -actix NOT "rocket framework""#);
        assert_eq!(q.positive, "rust web framework");
        assert_eq!(q.negative_terms, vec!["actix"]);
        assert_eq!(q.negative_phrases, vec!["rocket framework"]);
    }

    // ── Edge cases ─────────────────────────────────────────────────────

    #[test]
    fn parse_empty_query() {
        let q = ParsedQuery::parse("");
        assert!(q.positive.is_empty());
        assert!(!q.has_negations());
        assert!(q.is_positive_empty());
    }

    #[test]
    fn parse_only_whitespace() {
        let q = ParsedQuery::parse("   ");
        assert!(q.positive.is_empty());
        assert!(q.is_positive_empty());
    }

    #[test]
    fn parse_only_negations() {
        let q = ParsedQuery::parse("-spam -ads");
        assert!(q.positive.is_empty());
        assert!(q.is_positive_empty());
        assert_eq!(q.negative_terms, vec!["spam", "ads"]);
        assert!(q.has_negations());
    }

    #[test]
    fn parse_escaped_dash() {
        let q = ParsedQuery::parse(r"\-literal term");
        assert_eq!(q.positive, "-literal term");
        assert!(q.negative_terms.is_empty());
    }

    #[test]
    fn parse_dash_at_start() {
        let q = ParsedQuery::parse("-excluded query");
        assert_eq!(q.positive, "query");
        assert_eq!(q.negative_terms, vec!["excluded"]);
    }

    #[test]
    fn parse_dash_inside_word_is_literal() {
        let q = ParsedQuery::parse("foo-bar baz");
        assert_eq!(q.positive, "foo-bar baz");
        assert!(q.negative_terms.is_empty());
        assert!(!q.has_negations());
    }

    #[test]
    fn parse_not_without_quote() {
        // "NOT" without a following quote is treated as a regular word.
        let q = ParsedQuery::parse("why NOT this");
        assert_eq!(q.positive, "why NOT this");
        assert!(!q.has_negations());
    }

    #[test]
    fn parse_not_case_insensitive() {
        let q = ParsedQuery::parse(r#"query not "excluded""#);
        assert_eq!(q.positive, "query");
        assert_eq!(q.negative_phrases, vec!["excluded"]);
    }

    #[test]
    fn parse_consecutive_spaces() {
        let q = ParsedQuery::parse("  hello   world   -bad  ");
        assert_eq!(q.positive, "hello world");
        assert_eq!(q.negative_terms, vec!["bad"]);
    }

    #[test]
    fn parse_unclosed_not_quote() {
        // Unclosed quote — phrase extends to end of string.
        let q = ParsedQuery::parse(r#"query NOT "unclosed"#);
        assert_eq!(q.positive, "query");
        assert_eq!(q.negative_phrases, vec!["unclosed"]);
    }

    #[test]
    fn parse_empty_not_phrase() {
        // NOT "" — empty phrase is ignored.
        let q = ParsedQuery::parse(r#"query NOT """#);
        assert_eq!(q.positive, "query");
        assert!(q.negative_phrases.is_empty());
    }

    #[test]
    fn parse_standalone_dash() {
        // A lone dash with nothing after it.
        let q = ParsedQuery::parse("query -");
        assert_eq!(q.positive, "query -");
        assert!(q.negative_terms.is_empty());
        assert!(!q.has_negations());
    }

    #[test]
    fn parse_not_embedded_in_word() {
        // "cannot" contains "not" but shouldn't trigger NOT keyword.
        let q = ParsedQuery::parse(r#"cannot "be parsed""#);
        assert_eq!(q.positive, r#"cannot "be parsed""#);
        assert!(!q.has_negations());
    }

    // ── Helpers ────────────────────────────────────────────────────────

    #[test]
    fn negation_count_works() {
        let q = ParsedQuery::parse(r#"-a -b NOT "c d""#);
        assert_eq!(q.negation_count(), 3);
    }

    // ── Display ────────────────────────────────────────────────────────

    #[test]
    fn display_roundtrip_simple() {
        let q = ParsedQuery::parse("query -bad");
        assert_eq!(q.to_string(), "query -bad");
    }

    #[test]
    fn display_with_phrase() {
        let q = ParsedQuery::parse(r#"query NOT "bad phrase""#);
        assert_eq!(q.to_string(), r#"query NOT "bad phrase""#);
    }

    // ── Serialization ──────────────────────────────────────────────────

    #[test]
    fn serde_roundtrip() {
        let q = ParsedQuery::parse(r#"rust -tokio NOT "deep learning""#);
        let json = serde_json::to_string(&q).unwrap();
        let decoded: ParsedQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, q);
    }

    // ── bd-3un.49: Additional coverage ───────────────────────────────

    #[test]
    fn parse_negation_in_middle_of_query() {
        let q = ParsedQuery::parse("foo -bar baz");
        assert_eq!(q.positive, "foo baz");
        assert_eq!(q.negative_terms, vec!["bar"]);
    }

    #[test]
    fn parse_double_dash_is_treated_as_literal() {
        // `--flag` is treated as a literal token (CLI-like), not exclusion syntax.
        let q = ParsedQuery::parse("query --flag");
        assert_eq!(q.positive, "query --flag");
        assert!(q.negative_terms.is_empty());
        assert!(!q.has_negations());
    }

    #[test]
    fn parse_special_chars_in_negated_term() {
        let q = ParsedQuery::parse("query -foo.bar");
        assert_eq!(q.positive, "query");
        assert_eq!(q.negative_terms, vec!["foo.bar"]);
    }

    #[test]
    fn parse_negated_path() {
        let q = ParsedQuery::parse("search -src/main.rs");
        assert_eq!(q.positive, "search");
        assert_eq!(q.negative_terms, vec!["src/main.rs"]);
    }

    #[test]
    fn parse_negated_term_preserves_case() {
        let q = ParsedQuery::parse("search -Rust -ASYNC");
        assert_eq!(q.positive, "search");
        assert_eq!(q.negative_terms, vec!["Rust", "ASYNC"]);
    }

    #[test]
    fn parse_multiple_not_phrases() {
        let q = ParsedQuery::parse(r#"language NOT "deep learning" NOT "neural network""#);
        assert_eq!(q.positive, "language");
        assert_eq!(q.negative_phrases, vec!["deep learning", "neural network"]);
    }

    #[test]
    fn parse_unicode_in_negated_term() {
        let q = ParsedQuery::parse("search -café");
        assert_eq!(q.positive, "search");
        assert_eq!(q.negative_terms, vec!["café"]);
    }

    #[test]
    fn parse_unicode_in_not_phrase() {
        let q = ParsedQuery::parse(r#"query NOT "über cool""#);
        assert_eq!(q.positive, "query");
        assert_eq!(q.negative_phrases, vec!["über cool"]);
    }

    #[test]
    fn has_negations_false_for_plain_query() {
        let q = ParsedQuery::parse("just a plain query");
        assert!(!q.has_negations());
        assert_eq!(q.negation_count(), 0);
    }

    #[test]
    fn has_negations_true_for_term_only() {
        let q = ParsedQuery::parse("-excluded");
        assert!(q.has_negations());
        assert_eq!(q.negation_count(), 1);
        assert!(q.is_positive_empty());
    }

    #[test]
    fn has_negations_true_for_phrase_only() {
        let q = ParsedQuery::parse(r#"NOT "excluded phrase""#);
        assert!(q.has_negations());
        assert_eq!(q.negation_count(), 1);
        assert!(q.is_positive_empty());
    }

    #[test]
    fn display_all_components() {
        let q = ParsedQuery::parse(r#"find stuff -bad NOT "worse thing""#);
        let displayed = q.to_string();
        assert_eq!(displayed, r#"find stuff -bad NOT "worse thing""#);
    }

    #[test]
    fn display_only_negations() {
        let q = ParsedQuery::parse(r#"-a NOT "b c""#);
        let displayed = q.to_string();
        assert_eq!(displayed, r#" -a NOT "b c""#);
    }

    #[test]
    fn serde_roundtrip_complex() {
        let q = ParsedQuery::parse(r#"complex -query NOT "phrase one" -again NOT "phrase two""#);
        let json = serde_json::to_string(&q).unwrap();
        let decoded: ParsedQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, q);
        assert_eq!(decoded.negative_terms, vec!["query", "again"]);
        assert_eq!(decoded.negative_phrases, vec!["phrase one", "phrase two"]);
    }

    #[test]
    fn parse_tab_and_newline_as_whitespace() {
        let q = ParsedQuery::parse("hello\t-bad\nworld");
        assert_eq!(q.positive, "hello world");
        assert_eq!(q.negative_terms, vec!["bad"]);
    }

    #[test]
    fn parse_many_negations() {
        let q = ParsedQuery::parse("query -a -b -c -d -e -f -g -h -i -j");
        assert_eq!(q.positive, "query");
        assert_eq!(q.negative_terms.len(), 10);
        assert_eq!(q.negation_count(), 10);
    }

    #[test]
    fn parse_not_at_end_without_content() {
        // "NOT" at end of string without a quote.
        let q = ParsedQuery::parse("query NOT");
        assert_eq!(q.positive, "query NOT");
        assert!(!q.has_negations());
    }

    #[test]
    fn parse_numeric_negated_term() {
        let q = ParsedQuery::parse("error -404 -500");
        assert_eq!(q.positive, "error");
        assert_eq!(q.negative_terms, vec!["404", "500"]);
    }

    #[test]
    fn equality_and_clone() {
        let q1 = ParsedQuery::parse("query -bad");
        let q2 = q1.clone();
        assert_eq!(q1, q2);
    }
}
