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

use std::borrow::Cow;

use serde::{Deserialize, Serialize};

/// A parsed query with positive content and optional negations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParsedQuery {
    /// The main query text (what to search for).
    pub positive: String,
    /// Terms or field-qualified operands to exclude (from `-term` syntax).
    pub negative_terms: Vec<String>,
    /// Exact phrases to exclude (from `NOT "phrase"` syntax).
    pub negative_phrases: Vec<String>,
}

enum LexicalNegationEdit {
    NotKeyword(usize),
    BooleanWord { start: usize, end: usize },
}

impl ParsedQuery {
    /// Parse a raw query string into positive and negative components.
    ///
    /// # Syntax
    ///
    /// - `-term` : exclude documents containing "term"
    /// - `-title:"exact phrase"` : preserve a field-qualified lexical exclusion
    /// - `NOT "exact phrase"` or `-"exact phrase"` : exclude a quoted phrase
    /// - Single quotes also delimit phrases; a backslash escapes the next character
    /// - `\-term` : literal dash (escaped, not negation)
    /// - Multiple exclusions are supported: `query -foo -bar`
    ///
    /// Positive quoted fragments retain their delimiters, escapes, and internal
    /// whitespace for the lexical parser. Exclusion operators inside them are
    /// literal text. Negative phrases contain decoded text; use
    /// [`quote_query_phrase`] when rendering them into another query.
    #[must_use]
    pub fn parse(raw: &str) -> Self {
        Self::parse_with_lexical_edits(raw, None)
    }

    fn parse_with_lexical_edits(
        raw: &str,
        mut edits: Option<&mut Vec<LexicalNegationEdit>>,
    ) -> Self {
        // Without syntax characters, only whitespace normalization is needed.
        if !raw
            .bytes()
            .any(|b| matches!(b, b'-' | b'"' | b'\'' | b'\\'))
        {
            let mut positive = String::with_capacity(raw.len());
            for word in raw.split_whitespace() {
                if !positive.is_empty() {
                    positive.push(' ');
                }
                positive.push_str(word);
            }
            return Self {
                positive,
                negative_terms: Vec::new(),
                negative_phrases: Vec::new(),
            };
        }

        let mut positive_parts = Vec::new();
        let mut negative_terms = Vec::new();
        let mut negative_phrases = Vec::new();

        let chars: Vec<char> = raw.chars().collect();
        let len = chars.len();
        let mut i = 0;

        while i < len {
            if chars[i].is_whitespace() {
                i += 1;
                continue;
            }

            if i + 3 < len && matches_not_keyword(&chars, i) {
                let mut quote = i + 3;
                while quote < len && chars[quote].is_whitespace() {
                    quote += 1;
                }
                if quote < len && matches!(chars[quote], '"' | '\'') {
                    let (phrase, next) = parse_quoted_phrase(&chars, quote);
                    if !phrase.trim().is_empty() {
                        if (chars[i] != 'N' || chars[i + 1] != 'O' || chars[i + 2] != 'T')
                            && let Some(edits) = edits.as_deref_mut()
                        {
                            edits.push(LexicalNegationEdit::NotKeyword(i));
                        }
                        negative_phrases.push(phrase);
                    }
                    i = next;
                    continue;
                }
                // NOT without a following quote remains a regular word.
            }

            if chars[i] == '\\' && i + 1 < len && chars[i + 1] == '-' {
                // Retain the established literal-dash representation.
                let start = i + 1;
                i = query_fragment_end(&chars, i);
                positive_parts.push(chars[start..i].iter().collect::<String>());
                continue;
            }

            if chars[i] == '-'
                && (i == 0 || chars[i - 1].is_whitespace())
                && i + 1 < len
                && matches!(chars[i + 1], '"' | '\'')
            {
                let (phrase, next) = parse_quoted_phrase(&chars, i + 1);
                if !phrase.trim().is_empty() {
                    negative_phrases.push(phrase);
                }
                i = next;
                continue;
            }
            if chars[i] == '-'
                && (i == 0 || chars[i - 1].is_whitespace())
                && i + 1 < len
                && chars[i + 1] != '-'
                && !chars[i + 1].is_whitespace()
            {
                i += 1;
                let start = i;
                i = query_fragment_end(&chars, start);
                let term: String = chars[start..i].iter().collect();
                if !term.is_empty() {
                    if let Some(edits) = edits.as_deref_mut()
                        && matches!(term.as_str(), "AND" | "OR" | "NOT")
                    {
                        edits.push(LexicalNegationEdit::BooleanWord { start, end: i });
                    }
                    negative_terms.push(term);
                }
                continue;
            }

            // Scan the complete positive fragment, including any quoted field
            // value or operand. Whitespace inside a phrase cannot start a new
            // exclusion, and escaped quotes cannot end that phrase.
            let start = i;
            i = query_fragment_end(&chars, start);
            positive_parts.push(chars[start..i].iter().collect::<String>());
        }

        Self {
            positive: positive_parts.join(" "),
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

/// Normalize recognized exclusions for a lexical parser.
///
/// Case-insensitive `NOT` keywords before quoted phrases are uppercased, and
/// bare `-AND`, `-OR`, and `-NOT` operands are quoted so they remain literal
/// words rather than Boolean operators. The exclusion scanner identifies every
/// edit; positive text, quoted literals, escaped dashes, fields, and whitespace
/// are preserved.
#[must_use]
pub fn normalize_query_negations(raw: &str) -> Cow<'_, str> {
    if !raw.contains('-')
        && !raw
            .as_bytes()
            .windows(3)
            .any(|word| word != b"NOT" && word.eq_ignore_ascii_case(b"NOT"))
    {
        return Cow::Borrowed(raw);
    }
    let mut edits = Vec::new();
    let _ = ParsedQuery::parse_with_lexical_edits(raw, Some(&mut edits));
    if edits.is_empty() {
        return Cow::Borrowed(raw);
    }

    let chars: Vec<char> = raw.chars().collect();
    let mut normalized = String::with_capacity(raw.len());
    let mut cursor = 0;
    for edit in edits {
        let (start, end) = match edit {
            LexicalNegationEdit::NotKeyword(start) => (start, start + 3),
            LexicalNegationEdit::BooleanWord { start, end } => (start, end),
        };
        normalized.extend(chars[cursor..start].iter().copied());
        match edit {
            LexicalNegationEdit::NotKeyword(_) => normalized.push_str("NOT"),
            LexicalNegationEdit::BooleanWord { .. } => {
                normalized.push('"');
                normalized.extend(chars[start..end].iter().copied());
                normalized.push('"');
            }
        }
        cursor = end;
    }
    normalized.extend(chars[cursor..].iter().copied());
    Cow::Owned(normalized)
}

/// Render decoded text as one double-quoted query phrase.
///
/// Escapes both backslashes and quote delimiters so literal phrase content
/// cannot become query syntax. This is the inverse of quoted-phrase decoding,
/// including for Unicode text and a trailing backslash.
#[must_use]
pub fn quote_query_phrase(phrase: &str) -> String {
    let mut quoted = String::with_capacity(phrase.len().saturating_add(2));
    quoted.push('"');
    for ch in phrase.chars() {
        if matches!(ch, '"' | '\\') {
            quoted.push('\\');
        }
        quoted.push(ch);
    }
    quoted.push('"');
    quoted
}

/// Decode a phrase using the lexical parser's backslash rules.
fn parse_quoted_phrase(chars: &[char], quote: usize) -> (String, usize) {
    let delimiter = chars[quote];
    let mut phrase = String::new();
    let mut cursor = quote + 1;
    while cursor < chars.len() {
        let ch = chars[cursor];
        cursor += 1;
        if ch == delimiter {
            break;
        }
        if ch == '\\' && cursor < chars.len() {
            phrase.push(chars[cursor]);
            cursor += 1;
        } else {
            // An unterminated phrase includes the remaining text, including
            // a backslash at end of input.
            phrase.push(ch);
        }
    }
    (phrase, cursor)
}

/// Preserve the source spelling of query operands without interpreting
/// apostrophes or embedded quotes in ordinary words as phrase delimiters.
fn query_fragment_end(chars: &[char], start: usize) -> usize {
    let mut cursor = start;
    let mut quote = None;
    let mut typed_literal = false;
    let mut operand_start = true;
    let mut field_start = Some(start);
    while cursor < chars.len() {
        let ch = chars[cursor];
        if ch == '\\' {
            cursor += 1;
            if cursor < chars.len() {
                cursor += 1;
            }
            operand_start = false;
            continue;
        }
        if let Some(delimiter) = quote {
            if ch == delimiter {
                quote = None;
            }
        } else if typed_literal {
            if matches!(ch, '"' | '\'') {
                quote = Some(ch);
            } else if matches!(ch, ']' | '}') {
                typed_literal = false;
                operand_start = true;
                field_start = Some(cursor + 1);
            }
        } else {
            if operand_start && let Some(open) = query_set_open(chars, cursor) {
                // IN and its bracketed members form one operand, including
                // when that operand is an exclusion after a field prefix.
                cursor = open;
                continue;
            }
            if ch.is_whitespace() {
                // A field separator may follow whitespace after its name.
                // Preserve that complete operand before inspecting exclusions.
                let mut next = cursor;
                while next < chars.len() && chars[next].is_whitespace() {
                    next += 1;
                }
                if next < chars.len() && operand_start && field_start.is_none() {
                    // A field value may also begin after whitespace following
                    // its colon. Keep it inside the same exclusion operand.
                    cursor = next;
                    continue;
                }
                if next < chars.len()
                    && chars[next] == ':'
                    && field_start.is_some_and(|start| cursor > start)
                {
                    cursor = next;
                    continue;
                }
                break;
            }
            match ch {
                '[' | '{' if operand_start => {
                    typed_literal = true;
                    operand_start = false;
                    field_start = None;
                }
                '"' | '\'' if operand_start => {
                    quote = Some(ch);
                    operand_start = false;
                    field_start = None;
                }
                '(' | ')' => {
                    operand_start = true;
                    field_start = Some(cursor + 1);
                }
                '+' | '-' if operand_start && field_start.is_some() => {
                    field_start = Some(cursor + 1);
                }
                ':' if field_start.is_some_and(|start| cursor > start) => {
                    operand_start = true;
                    // A colon in the resulting bare value is ordinary text;
                    // it cannot start a second field-qualified phrase.
                    field_start = None;
                }
                _ => {
                    operand_start = false;
                    if matches!(
                        ch,
                        '+' | '^' | '`' | '{' | '}' | '"' | '\'' | '[' | ']' | '!' | '*' | ':'
                    ) {
                        field_start = None;
                    }
                }
            }
        }
        cursor += 1;
    }
    cursor
}

fn query_set_open(chars: &[char], start: usize) -> Option<usize> {
    let after_in = start.checked_add(2)?;
    if chars.get(start..after_in)? != ['I', 'N'] {
        return None;
    }
    let mut open = after_in;
    while chars.get(open).is_some_and(|ch| ch.is_whitespace()) {
        open += 1;
    }
    (open > after_in && chars.get(open) == Some(&'[')).then_some(open)
}

/// Check if the characters at position `i` match "NOT" (case-insensitive)
/// followed by whitespace or end of input.
fn matches_not_keyword(chars: &[char], i: usize) -> bool {
    let len = chars.len();
    if i + 3 > len {
        return false;
    }
    if !matches!(chars[i], 'N' | 'n')
        || !matches!(chars[i + 1], 'O' | 'o')
        || !matches!(chars[i + 2], 'T' | 't')
    {
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
        let mut wrote_any = !self.positive.is_empty();
        if wrote_any {
            write!(f, "{}", self.positive)?;
        }
        for term in &self.negative_terms {
            if wrote_any {
                write!(f, " ")?;
            }
            write!(f, "-{term}")?;
            wrote_any = true;
        }
        for phrase in &self.negative_phrases {
            if wrote_any {
                write!(f, " ")?;
            }
            write!(f, "NOT {}", quote_query_phrase(phrase))?;
            wrote_any = true;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Basic parsing ──────────────────────────────────────────────────

    #[test]
    fn normalize_phrase_negations_preserves_positive_source_and_unicode_offsets() {
        for (raw, expected) in [
            (
                r#"needle nOt "internal secret""#,
                r#"needle NOT "internal secret""#,
            ),
            (r#"\-literal not "secret""#, r#"\-literal NOT "secret""#),
            (
                r#"title:"not \"secret\"" Not 'outside secret'"#,
                r#"title:"not \"secret\"" NOT 'outside secret'"#,
            ),
            (
                r#"雪猫 not "éclair" αβγ nOt 'another phrase'"#,
                r#"雪猫 NOT "éclair" αβγ NOT 'another phrase'"#,
            ),
            (
                "needle\tNoT\t\"secret\"\u{2003}not 'other'",
                "needle\tNOT\t\"secret\"\u{2003}NOT 'other'",
            ),
            (
                r#"needle not "quote \"inside\" and path\\" NOT "other""#,
                r#"needle NOT "quote \"inside\" and path\\" NOT "other""#,
            ),
        ] {
            let normalized = normalize_query_negations(raw);
            assert_eq!(normalized, expected, "{raw}");
            assert_eq!(ParsedQuery::parse(&normalized), ParsedQuery::parse(raw));
        }
    }

    #[test]
    fn normalize_phrase_negations_borrows_unchanged_source() {
        for raw in [
            "plain query",
            r#"needle NOT "secret""#,
            "this is not excluded",
            r#""say not \"secret\"""#,
            "'say not \"secret\"'",
            r#"title : "not \"secret\"""#,
            r#"\not "secret""#,
            r#"cannot "be excluded""#,
            r#"needle -"say not \"secret\"""#,
            r#"(not "secret")"#,
            r#""-AND" title:"-OR" \-NOT --AND dash-OR -ORganic"#,
            "needle -and -or -not",
        ] {
            assert!(
                matches!(
                    normalize_query_negations(raw),
                    Cow::Borrowed(value) if value == raw
                ),
                "{raw}",
            );
        }
    }

    #[test]
    fn normalize_boolean_exclusions_preserves_positive_source_and_edit_offsets() {
        for (raw, expected) in [
            ("-AND", "-\"AND\""),
            ("-OR ", "-\"OR\" "),
            ("needle -NOT -absent", "needle -\"NOT\" -absent"),
            (
                "\t-AND\u{2003}-OR \n-NOT ",
                "\t-\"AND\"\u{2003}-\"OR\" \n-\"NOT\" ",
            ),
            (
                r#"title:"say -AND" \-OR -AND OR needle"#,
                r#"title:"say -AND" \-OR -"AND" OR needle"#,
            ),
            (
                r#"雪猫 not "éclair" -AND αβγ nOt 'another phrase' -OR -NOT"#,
                r#"雪猫 NOT "éclair" -"AND" αβγ NOT 'another phrase' -"OR" -"NOT""#,
            ),
        ] {
            let normalized = normalize_query_negations(raw);
            assert_eq!(normalized, expected, "{raw}");
            assert_eq!(
                ParsedQuery::parse(&normalized).positive,
                ParsedQuery::parse(raw).positive,
                "positive query source: {raw}"
            );
            assert_eq!(normalize_query_negations(&normalized), normalized);
        }
    }

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
    fn parse_dash_phrase() {
        let q = ParsedQuery::parse(r#"rrf -"rank fusion" scoring"#);
        assert_eq!(q.positive, "rrf scoring");
        assert!(q.negative_terms.is_empty());
        assert_eq!(q.negative_phrases, vec!["rank fusion"]);
        // Unclosed and empty forms stay bounded and add nothing positive.
        let q = ParsedQuery::parse(r#"rrf -"rank fusion"#);
        assert_eq!(q.positive, "rrf");
        assert_eq!(q.negative_phrases, vec!["rank fusion"]);
        let q = ParsedQuery::parse(r#"rrf -"" x"#);
        assert_eq!(q.positive, "rrf x");
        assert!(!q.has_negations());
    }

    #[test]
    fn parse_mixed_negations() {
        let q = ParsedQuery::parse(r#"rust web framework -actix NOT "rocket framework""#);
        assert_eq!(q.positive, "rust web framework");
        assert_eq!(q.negative_terms, vec!["actix"]);
        assert_eq!(q.negative_phrases, vec!["rocket framework"]);
    }

    #[test]
    fn positive_quoted_fragments_do_not_create_exclusions() {
        for (raw, positive) in [
            (r#""alpha -banned" -outside"#, r#""alpha -banned""#),
            (
                r#"title:"alpha -banned" -outside"#,
                r#"title:"alpha -banned""#,
            ),
            ("'alpha -banned' -outside", "'alpha -banned'"),
            ("title:'alpha -banned' -outside", "title:'alpha -banned'"),
            (
                r#"title : "alpha -banned" -outside"#,
                r#"title : "alpha -banned""#,
            ),
            (r#"+("alpha -banned") -outside"#, r#"+("alpha -banned")"#),
            (
                r#""alpha NOT \"secret\" -banned" -outside"#,
                r#""alpha NOT \"secret\" -banned""#,
            ),
            (r#""alpha   -banned" -outside"#, r#""alpha   -banned""#),
        ] {
            let parsed = ParsedQuery::parse(raw);
            assert_eq!(parsed.positive, positive, "{raw}");
            assert_eq!(parsed.negative_terms, ["outside"], "{raw}");
            assert!(parsed.negative_phrases.is_empty(), "{raw}");
        }
    }

    #[test]
    fn positive_unterminated_phrases_protect_the_remaining_text() {
        for raw in [
            r#""alpha -banned NOT \"secret\""#,
            r#"title:"alpha -banned"#,
            "'alpha -banned",
        ] {
            let parsed = ParsedQuery::parse(raw);
            assert_eq!(parsed.positive, raw);
            assert!(!parsed.has_negations(), "{raw}");
        }
    }

    #[test]
    fn negative_field_operands_preserve_phrases_and_source_spelling() {
        for operand in [
            r#"title:"internal secret""#,
            "title:'internal secret'",
            r#"title : "internal secret""#,
            r#"title: "internal secret""#,
            r#"title:"internal   secret""#,
            r#"content:"café 日本語 -literal NOT \"quoted\" \\ path""#,
            r"title:'it\'s -literal'",
            r#"title:"internal secret"~2"#,
            r"title: internal",
            r#"title:"internal secret"#,
            r#"id:['a " b' TO b]"#,
            r#"id: IN ['a " b']"#,
            "ord:[0 TO -1]",
            r#"id:IN ['a -b' 'NOT "phrase"']"#,
        ] {
            let raw = format!("needle -{operand}");
            let parsed = ParsedQuery::parse(&raw);
            assert_eq!(parsed.positive, "needle", "{raw}");
            assert_eq!(parsed.negative_terms, [operand], "{raw}");
            assert!(parsed.negative_phrases.is_empty(), "{raw}");
            assert_eq!(normalize_query_negations(&raw), raw, "{raw}");
        }
        let raw = r#"needle OR other -title : "internal -secret" -outside"#;
        let parsed = ParsedQuery::parse(raw);
        assert_eq!(parsed.positive, "needle OR other");
        assert_eq!(
            parsed.negative_terms,
            [r#"title : "internal -secret""#, "outside"]
        );
        assert!(parsed.negative_phrases.is_empty());
    }

    #[test]
    fn typed_literal_operands_do_not_create_exclusions() {
        for operand in [
            r"id:['a -b' TO z]",
            r#"id:{'a not "b"' TO z]"#,
            r#"id:IN ['not "phrase"' -AND]"#,
            r"id: IN ['a -b']",
            "ord:[0 TO -1]",
            r"id:['a\' ] -b' TO z]",
            "id:['café -猫' TO z]",
            "id:IN [a -b]",
        ] {
            let raw = format!("{operand} -outside");
            let parsed = ParsedQuery::parse(&raw);
            assert_eq!(parsed.positive, operand, "{raw}");
            assert_eq!(parsed.negative_terms, ["outside"], "{raw}");
            assert!(parsed.negative_phrases.is_empty(), "{raw}");
            assert_eq!(normalize_query_negations(&raw), raw, "{raw}");
        }
        for raw in [
            r#"id:['unterminated -inside NOT "phrase""#,
            "id:IN [a -inside",
        ] {
            let parsed = ParsedQuery::parse(raw);
            assert_eq!(parsed.positive, raw, "{raw}");
            assert!(!parsed.has_negations(), "{raw}");
        }
    }

    #[test]
    fn negative_bare_words_do_not_swallow_following_operands() {
        for operand in [
            "can't",
            r#"say"hello"#,
            r#"title\:"literal"#,
            r#"id:part:"literal"#,
            r#"foo+bar:"literal"#,
        ] {
            let raw = format!("needle -{operand} tail -outside");
            let parsed = ParsedQuery::parse(&raw);
            assert_eq!(parsed.positive, "needle tail", "{raw}");
            assert_eq!(parsed.negative_terms, [operand, "outside"], "{raw}");
            assert!(parsed.negative_phrases.is_empty(), "{raw}");
        }
        let parsed = ParsedQuery::parse(r#"needle -title:"alpha \\\" -inside" -outside"#);
        assert_eq!(parsed.positive, "needle");
        assert_eq!(
            parsed.negative_terms,
            [r#"title:"alpha \\\" -inside""#, "outside"]
        );
        let parsed = ParsedQuery::parse(r#"needle -title:"alpha \\" -outside"#);
        assert_eq!(parsed.positive, "needle");
        assert_eq!(parsed.negative_terms, [r#"title:"alpha \\""#, "outside"]);
    }

    #[test]
    fn quotes_inside_bare_words_do_not_hide_real_exclusions() {
        for (raw, positive) in [
            ("don't -outside", "don't"),
            (r#"say"hello -outside"#, r#"say"hello"#),
            (r#"say\"hello -outside"#, r#"say\"hello"#),
            (r"\-literal -outside", "-literal"),
            (r#"\-"literal -outside"#, r#"-"literal"#),
            (r#"id:part:"literal -outside"#, r#"id:part:"literal"#),
            (r#"foo+bar:"literal -outside"#, r#"foo+bar:"literal"#),
            (r#"id:+"literal -outside"#, r#"id:+"literal"#),
            (r#":"literal -outside"#, r#":"literal"#),
        ] {
            let parsed = ParsedQuery::parse(raw);
            assert_eq!(parsed.positive, positive, "{raw}");
            assert_eq!(parsed.negative_terms, ["outside"], "{raw}");
            assert!(parsed.negative_phrases.is_empty(), "{raw}");
        }
    }

    #[test]
    fn negative_phrases_decode_quotes_backslashes_and_single_delimiters() {
        for raw in [
            r#"query NOT "alpha \"beta\" \\ path" -outside"#,
            r#"query -"alpha \"beta\" \\ path" -outside"#,
            r#"query not 'alpha "beta" \\ path' -outside"#,
            r#"query -'alpha "beta" \\ path' -outside"#,
        ] {
            let parsed = ParsedQuery::parse(raw);
            assert_eq!(parsed.positive, "query", "{raw}");
            assert_eq!(parsed.negative_terms, ["outside"], "{raw}");
            assert_eq!(parsed.negative_phrases, [r#"alpha "beta" \ path"#], "{raw}");
        }
        let parsed = ParsedQuery::parse(r"query NOT 'it\'s -literal'");
        assert_eq!(parsed.negative_phrases, ["it's -literal"]);
        assert!(parsed.negative_terms.is_empty());
    }

    #[test]
    fn quoted_phrase_boundaries_distinguish_odd_and_even_backslashes() {
        let even = ParsedQuery::parse(r#"query NOT "alpha \\" -outside"#);
        assert_eq!(even.negative_phrases, ["alpha \\"]);
        assert_eq!(even.negative_terms, ["outside"]);
        assert_eq!(even.positive, "query");

        let odd = ParsedQuery::parse(r#"query NOT "alpha \\\" -inside" -outside"#);
        assert_eq!(odd.negative_phrases, [r#"alpha \" -inside"#]);
        assert_eq!(odd.negative_terms, ["outside"]);
        assert_eq!(odd.positive, "query");

        let positive = ParsedQuery::parse(r#""alpha \\\" -inside" -outside"#);
        assert_eq!(positive.positive, r#""alpha \\\" -inside""#);
        assert_eq!(positive.negative_terms, ["outside"]);
    }

    #[test]
    fn unterminated_negative_phrase_preserves_trailing_backslash() {
        for raw in [r#"query NOT "path\"#, r#"query -"path\"#] {
            let parsed = ParsedQuery::parse(raw);
            assert_eq!(parsed.positive, "query");
            assert_eq!(parsed.negative_phrases, ["path\\"]);
            assert!(parsed.negative_terms.is_empty());
        }
    }

    #[test]
    fn phrase_rendering_preserves_decoded_content_on_display_roundtrip() {
        assert_eq!(quote_query_phrase("a\"b\\c"), r#""a\"b\\c""#);
        assert_eq!(quote_query_phrase(""), r#""""#);
        let parsed = ParsedQuery {
            positive: r#""positive -literal""#.to_owned(),
            negative_terms: vec!["outside".to_owned()],
            negative_phrases: vec![
                r#"alpha "quoted" beta"#.to_owned(),
                r"C:\work\cache".to_owned(),
                "café 日本語".to_owned(),
                "trailing\\".to_owned(),
                "line one\nline two".to_owned(),
            ],
        };
        assert_eq!(ParsedQuery::parse(&parsed.to_string()), parsed);
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
        assert_eq!(displayed, r#"-a NOT "b c""#);
    }

    #[test]
    fn display_only_phrase_negation_has_no_leading_space() {
        let q = ParsedQuery::parse(r#"NOT "b c""#);
        let displayed = q.to_string();
        assert_eq!(displayed, r#"NOT "b c""#);
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
