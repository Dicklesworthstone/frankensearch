//! Text canonicalization pipeline for frankensearch.
//!
//! All text is preprocessed before embedding to maximize search quality.
//! The default pipeline applies:
//! 1. NFC Unicode normalization (hash stability across representations)
//! 2. Markdown stripping (bold, italic, headers, links, blockquotes, list markers, inline code)
//! 3. Code block collapsing (first 20 + last 10 lines of fenced blocks)
//! 4. Whitespace normalization (collapse runs to single space)
//! 5. Low-signal filtering (short ack phrases like "OK", "Done.", "Thanks")
//! 6. Length truncation (default 2000 characters)
//!
//! Query canonicalization is simpler (NFC + trim only) since queries are
//! typically short natural language.

use std::borrow::Cow;

use unicode_normalization::UnicodeNormalization;

/// Low-signal content to filter out (exact matches, case-insensitive).
///
/// When the entire canonicalized text matches one of these patterns,
/// the result is an empty string (the message carries no semantic value).
const LOW_SIGNAL_CONTENT: &[&str] = &[
    "ok",
    "done",
    "done.",
    "got it",
    "got it.",
    "understood",
    "understood.",
    "sure",
    "sure.",
    "yes",
    "no",
    "thanks",
    "thanks.",
    "thank you",
    "thank you.",
];

/// Trait for text preprocessing before embedding.
///
/// Custom implementations can add domain-specific preprocessing
/// (e.g., abbreviation expansion, jargon normalization).
pub trait Canonicalizer: Send + Sync {
    /// Preprocess document text for embedding.
    fn canonicalize(&self, text: &str) -> String;

    /// Preprocess a search query.
    ///
    /// Typically simpler than document canonicalization since queries
    /// are short and don't contain markdown or code blocks.
    fn canonicalize_query(&self, query: &str) -> String;
}

/// Default canonicalization pipeline.
///
/// Applies NFC normalization, markdown stripping, code block collapsing,
/// whitespace normalization, low-signal filtering, and length truncation.
pub struct DefaultCanonicalizer {
    /// Maximum characters for canonicalized text. Default: 2000.
    pub max_length: usize,
    /// Maximum lines to keep from the start of a fenced code block. Default: 20.
    pub code_head_lines: usize,
    /// Maximum lines to keep from the end of a fenced code block. Default: 10.
    pub code_tail_lines: usize,
}

impl Default for DefaultCanonicalizer {
    fn default() -> Self {
        Self {
            max_length: 2000,
            code_head_lines: 20,
            code_tail_lines: 10,
        }
    }
}

/// NFC-normalize with a fast path for ASCII text.
///
/// ASCII is always already in NFC (no codepoint has a decomposition/composition),
/// so for ASCII input — the common case for code and English prose — this skips
/// the unicode-normalization state machine and copies the bytes directly. This
/// mirrors the ASCII fast paths in Lucene/Tantivy's analysis chains. Output is
/// byte-identical to `text.nfc().collect()` for every input.
#[inline]
fn nfc_normalize(text: &str) -> String {
    if text.is_ascii() {
        text.to_owned()
    } else {
        text.nfc().collect()
    }
}

impl Canonicalizer for DefaultCanonicalizer {
    fn canonicalize(&self, text: &str) -> String {
        // 1. NFC Unicode normalization (critical for hash stability)
        let normalized: String = nfc_normalize(text);
        // 2. Strip markdown and collapse code blocks
        let stripped = self.strip_markdown_and_code(&normalized);
        // 3. Normalize whitespace
        let ws_normalized = normalize_whitespace(&stripped);
        // 4. Filter low-signal content — drop the whole doc if it's just an ack.
        if is_low_signal(&ws_normalized) {
            return String::new();
        }
        // 5. Truncate to max length — pass the owned buffer straight through,
        //    avoiding the old `filter_low_signal` whole-document copy.
        truncate_to_chars(&ws_normalized, self.max_length)
    }

    fn canonicalize_query(&self, query: &str) -> String {
        // Queries are short — just NFC normalize and trim
        let normalized: String = nfc_normalize(query);
        let trimmed = normalized.trim();
        truncate_to_chars(trimmed, self.max_length)
    }
}

impl DefaultCanonicalizer {
    /// Strip markdown formatting and collapse code blocks.
    fn strip_markdown_and_code(&self, text: &str) -> String {
        let mut result = String::with_capacity(text.len());
        let mut in_code_block = false;
        let mut code_block_lang = String::new();
        let mut code_lines: Vec<&str> = Vec::new();

        for line in text.lines() {
            if line.starts_with("```") {
                if in_code_block {
                    // End of code block — collapse it
                    result.push_str(&collapse_code_block(
                        &code_block_lang,
                        &code_lines,
                        self.code_head_lines,
                        self.code_tail_lines,
                    ));
                    result.push('\n');
                    code_lines.clear();
                    code_block_lang.clear();
                    in_code_block = false;
                } else {
                    // Start of code block
                    in_code_block = true;
                    code_block_lang = line.trim_start_matches('`').trim().to_string();
                }
            } else if in_code_block {
                code_lines.push(line);
            } else {
                // Strip markdown from regular text
                let stripped = strip_markdown_line(line);
                if !stripped.is_empty() {
                    result.push_str(&stripped);
                    result.push('\n');
                }
            }
        }

        // Handle unclosed code block
        if in_code_block && !code_lines.is_empty() {
            result.push_str(&collapse_code_block(
                &code_block_lang,
                &code_lines,
                self.code_head_lines,
                self.code_tail_lines,
            ));
            result.push('\n');
        }

        result
    }
}

/// Collapse a code block to first N + last M lines.
fn collapse_code_block(lang: &str, lines: &[&str], head: usize, tail: usize) -> String {
    let lang_label = if lang.is_empty() {
        "code".to_string()
    } else {
        format!("code: {lang}")
    };

    if lines.len() <= head + tail {
        // Short enough to keep in full
        format!("[{lang_label}]\n{}", lines.join("\n"))
    } else {
        // Collapse middle
        let head_part: Vec<_> = lines.iter().take(head).copied().collect();
        let tail_part: Vec<_> = lines.iter().skip(lines.len() - tail).copied().collect();
        let omitted = lines.len() - head - tail;
        format!(
            "[{lang_label}]\n{}\n[... {omitted} lines omitted ...]\n{}",
            head_part.join("\n"),
            tail_part.join("\n")
        )
    }
}

/// Strip markdown formatting from a single line.
///
/// Returns a borrowed slice for the common case (plain line, no inline markdown),
/// so a plain document flows through `canonicalize` with no per-line allocation;
/// the inline-markdown slow path returns an owned `String`.
fn strip_markdown_line(line: &str) -> Cow<'_, str> {
    // Fast path: lines containing no inline-markdown characters skip the entire
    // replace/link/italic chain (each pass allocates + scans the line but is a
    // content no-op when its trigger char — `*` `_` `` ` `` `[` — is absent).
    // Byte-identical to the full path; the header/blockquote/list stripping below
    // still applies. Plain prose and most code-comment text hit this path.
    if line.bytes().any(|b| matches!(b, b'*' | b'_' | b'`' | b'[')) {
        // Remove bold/italic markers
        let mut r = line.replace("**", "");
        r = r.replace("__", "");
        r = r.replace('*', "");
        r = strip_italic_underscores(&r);
        // Remove inline code backticks
        r = r.replace('`', "");
        // Convert links [text](url) to just text
        let r = strip_markdown_links(&r);
        // `r` is a local owned buffer, so the borrowed result can't escape — take
        // ownership for the return (the slow path allocates regardless).
        Cow::Owned(strip_prefixes_and_list_marker(&r).into_owned())
    } else {
        // Fast path: no inline markdown, so operate directly on the borrowed
        // `line`. The prefix/blockquote trims and list-marker strip all return
        // borrowed `&str` slices, so a plain line flows through with **zero**
        // allocations — only the caller's single `push_str` copies the bytes.
        // Byte-identical to the prior owned-String path.
        strip_prefixes_and_list_marker(line)
    }
}

/// Strip leading header (`#`) / blockquote (`>`) prefixes plus their leading
/// whitespace as a single `&str` chain, then remove any list marker. Returns a
/// borrowed slice whenever nothing is stripped or only a prefix slice remains —
/// no allocation. Byte-identical to the prior owned-String trim chain.
fn strip_prefixes_and_list_marker(s: &str) -> Cow<'_, str> {
    let prefix_stripped = s
        .trim_start_matches('#')
        .trim_start()
        .trim_start_matches('>')
        .trim_start();
    strip_list_marker(prefix_stripped)
}

/// Strip italic underscore markers (`_word_`) while preserving underscores inside
/// identifiers (`snake_case`). An underscore is treated as an italic marker only
/// when it lies on a word boundary: no adjacent alphanumeric or underscore on
/// the side facing away from the emphasized span.
fn strip_italic_underscores(text: &str) -> String {
    let is_word = |c: char| c.is_alphanumeric() || c == '_';

    // Single pass building the output directly: `prev` tracks the previous source
    // char (kept or not) and `chars.peek()` supplies the next, so the same
    // boundary test as before is applied without materializing a `Vec<char>` + a
    // `Vec<bool>` + a final `collect` (three allocations → one). Byte-identical.
    let mut result = String::with_capacity(text.len());
    let mut prev: Option<char> = None;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        let drop_marker = if c == '_' {
            let prev_is_word = prev.is_some_and(|p| is_word(p) && p != '_');
            let next_is_word = chars.peek().is_some_and(|&n| is_word(n) && n != '_');
            // Opening marker: preceded by non-word (or BOL), followed by word.
            // Closing marker: preceded by word, followed by non-word (or EOL).
            (!prev_is_word && next_is_word) || (prev_is_word && !next_is_word)
        } else {
            false
        };
        if !drop_marker {
            result.push(c);
        }
        prev = Some(c);
    }
    result
}

/// Strip markdown links: `[text](url)` → `text`.
fn strip_markdown_links(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '[' {
            // Potential link start
            let mut link_text = String::new();
            let mut found_close = false;
            let mut bracket_depth = 1;

            for inner in chars.by_ref() {
                if inner == '[' {
                    bracket_depth += 1;
                } else if inner == ']' {
                    bracket_depth -= 1;
                    if bracket_depth == 0 {
                        found_close = true;
                        break;
                    }
                }
                link_text.push(inner);
            }

            if found_close && chars.peek() == Some(&'(') {
                // Potential URL start
                chars.next(); // consume '('
                let mut url_part = String::from("(");
                let mut depth = 1;
                let mut valid_link = false;

                for inner in chars.by_ref() {
                    url_part.push(inner);
                    match inner {
                        '(' => depth += 1,
                        ')' => {
                            depth -= 1;
                            if depth == 0 {
                                valid_link = true;
                                break;
                            }
                        }
                        _ => {}
                    }
                }

                if valid_link {
                    // Valid link: [text](url) -> text
                    result.push_str(&link_text);
                } else {
                    // Unbalanced parens or EOF: restore everything
                    result.push('[');
                    result.push_str(&link_text);
                    result.push(']');
                    result.push_str(&url_part);
                }
            } else {
                // Not a proper link (no '(' after ']'), keep original
                result.push('[');
                result.push_str(&link_text);
                if found_close {
                    result.push(']');
                }
            }
        } else {
            result.push(c);
        }
    }

    result
}

/// Strip markdown list markers from the start of a line.
///
/// Strips unordered (`- `, `+ `) and ordered (`1. `, `10. `) markers.
/// Does NOT strip arbitrary numbers (`3.14159` stays intact).
fn strip_list_marker(line: &str) -> Cow<'_, str> {
    let trimmed = line.trim_start();

    // Check for unordered list markers: "- " or "+ "
    if let Some(rest) = trimmed.strip_prefix("- ") {
        return Cow::Borrowed(rest);
    }
    if let Some(rest) = trimmed.strip_prefix("+ ") {
        return Cow::Borrowed(rest);
    }

    // Check for ordered list markers: digits followed by ". "
    let mut chars = trimmed.chars().peekable();
    let mut digit_count = 0;

    while let Some(&c) = chars.peek() {
        if c.is_ascii_digit() {
            digit_count += 1;
            chars.next();
        } else {
            break;
        }
    }

    // Must have at least one digit, followed by ". " (dot then space)
    if digit_count > 0 && chars.next() == Some('.') && chars.peek() == Some(&' ') {
        // All consumed chars (digits, '.', ' ') are single-byte ASCII, so the
        // remainder starts at byte offset `digit_count + 2` — a borrowed slice,
        // byte-identical to the prior `chars.collect()`.
        return Cow::Borrowed(&trimmed[digit_count + 2..]);
    }

    // Not a list marker, return original (borrowed).
    Cow::Borrowed(line)
}

/// Normalize whitespace: collapse runs to single space, trim.
fn normalize_whitespace(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut prev_whitespace = true; // Start as true to trim leading

    for c in text.chars() {
        if c.is_whitespace() {
            if !prev_whitespace {
                result.push(' ');
                prev_whitespace = true;
            }
        } else {
            result.push(c);
            prev_whitespace = false;
        }
    }

    // Drop any trailing space the loop pushed, in place — avoids a second full
    // allocation via `trim_end().to_string()`.
    let trimmed_len = result.trim_end().len();
    result.truncate(trimmed_len);
    result
}

/// Filter out low-signal content.
///
/// If the entire text (after trimming and lowercasing) matches a known
/// low-signal pattern, returns empty string.
/// Whether `text` is a low-signal ack phrase (case-insensitive, trimmed).
///
/// Returning a `bool` (vs the old `String`) lets `canonicalize` early-return the
/// empty string and pass its already-owned buffer straight to truncation — saving
/// a whole-document copy on the common (non-filtered) path. `LOW_SIGNAL_CONTENT` is
/// all ASCII, so `eq_ignore_ascii_case` is byte-identical to lowercasing while
/// short-circuiting on length and never allocating.
fn is_low_signal(text: &str) -> bool {
    let trimmed = text.trim();
    LOW_SIGNAL_CONTENT
        .iter()
        .any(|pattern| trimmed.eq_ignore_ascii_case(pattern))
}

/// Truncate string to at most N characters, respecting char boundaries.
fn truncate_to_chars(text: &str, max_chars: usize) -> String {
    // Fast path: each char is >= 1 byte, so if the whole text fits in `max_chars`
    // *bytes* it certainly fits in `max_chars` *chars* — no scan needed (the common
    // case for short docs/queries).
    if text.len() <= max_chars {
        return text.to_owned();
    }
    for (count, (idx, _)) in text.char_indices().enumerate() {
        if count == max_chars {
            return text[..idx].to_owned();
        }
    }
    text.to_owned()
}

#[cfg(test)]
mod tests {
    use std::fmt::Write;

    use super::*;

    #[test]
    fn nfc_normalization() {
        let canon = DefaultCanonicalizer::default();
        // e + combining acute accent → precomposed é
        let input = "caf\u{0065}\u{0301}";
        let result = canon.canonicalize(input);
        assert!(result.contains("caf\u{00e9}"));
    }

    #[test]
    fn nfc_normalize_ascii_fast_path_matches_reference() {
        use unicode_normalization::UnicodeNormalization;
        // The ASCII fast path must be byte-identical to the full nfc() pipeline,
        // and the non-ASCII path must still go through it.
        let cases = [
            "",
            "plain ascii text 123 _-./",
            "fn main() { let x = 0; }",
            "café\u{0301}",                // non-ASCII (combining mark)
            "caf\u{0065}\u{0301}\u{00e9}", // mixed decomposed/precomposed
            "日本語テキスト",              // non-ASCII
            "naïve façade",
        ];
        for c in cases {
            let reference: String = c.nfc().collect();
            assert_eq!(nfc_normalize(c), reference, "input={c:?}");
            // ASCII inputs must hit the fast path but still equal the reference.
            if c.is_ascii() {
                assert_eq!(nfc_normalize(c), c.to_owned(), "ascii fast path {c:?}");
            }
        }
    }

    #[test]
    fn strip_markdown_headings() {
        let canon = DefaultCanonicalizer::default();
        let input = "## Heading\nText";
        let result = canon.canonicalize(input);
        assert!(result.contains("Heading"));
        assert!(!result.contains("##"));
    }

    #[test]
    fn strip_markdown_preserves_inline_hash_tokens() {
        let canon = DefaultCanonicalizer::default();
        // Note: strip_markdown_line uses trim_start_matches('#'), which strips
        // leading '#' chars. "C# and #hashtag" starts with "C", so # is preserved.
        // But "## Heading" starts with ##, so those are stripped.
        let input = "C# and #hashtag\n## Heading";
        let result = canon.canonicalize(input);
        assert!(result.contains("C#"));
        assert!(result.contains("#hashtag"));
        assert!(result.contains("Heading"));
        assert!(!result.contains("## "));
    }

    #[test]
    fn strip_markdown_bold_italic() {
        let canon = DefaultCanonicalizer::default();
        let input = "**bold** and *italic* and __underline__";
        let result = canon.canonicalize(input);
        assert!(result.contains("bold"));
        assert!(result.contains("italic"));
        assert!(!result.contains("**"));
        assert!(!result.contains("__"));
    }

    #[test]
    fn strip_markdown_links() {
        let canon = DefaultCanonicalizer::default();
        let input = "See [the docs](https://example.com/path) for details";
        let result = canon.canonicalize(input);
        assert!(result.contains("the docs"));
        assert!(!result.contains("https://example.com"));
    }

    #[test]
    fn strip_inline_code_backticks() {
        let canon = DefaultCanonicalizer::default();
        let input = "Use `fn main()` to start.";
        let result = canon.canonicalize(input);
        assert!(result.contains("fn main()"));
        assert!(!result.contains('`'));
    }

    #[test]
    fn strip_blockquotes() {
        let canon = DefaultCanonicalizer::default();
        let input = "> This is a quote\n> spanning multiple lines";
        let result = canon.canonicalize(input);
        assert!(result.contains("This is a quote"));
        // After whitespace normalization, blockquote text is collapsed
        assert!(!result.starts_with('>'));
    }

    #[test]
    fn strip_list_markers_ordered() {
        let canon = DefaultCanonicalizer::default();
        let input = "1. First item\n2. Second item\n10. Tenth item";
        let result = canon.canonicalize(input);
        assert!(result.contains("First item"));
        assert!(result.contains("Second item"));
        assert!(result.contains("Tenth item"));
    }

    #[test]
    fn strip_list_markers_unordered() {
        let canon = DefaultCanonicalizer::default();
        let input = "- First\n+ Second";
        let result = canon.canonicalize(input);
        assert!(result.contains("First"));
        assert!(result.contains("Second"));
    }

    #[test]
    fn numbers_not_list_markers_preserved() {
        let canon = DefaultCanonicalizer::default();
        let input = "3.14159 is pi";
        let result = canon.canonicalize(input);
        assert!(result.contains("3.14159"));
    }

    #[test]
    fn collapse_short_code_block() {
        let canon = DefaultCanonicalizer::default();
        let input = "text\n```\nline1\nline2\nline3\n```\nmore text";
        let result = canon.canonicalize(input);
        assert!(result.contains("line1"));
        assert!(result.contains("line3"));
        assert!(result.contains("[code]"));
        assert!(!result.contains("omitted"));
    }

    #[test]
    fn collapse_long_code_block() {
        let mut input = String::from("before\n```\n");
        for i in 0..50 {
            let _ = writeln!(input, "code line {i}");
        }
        input.push_str("```\nafter");

        let canon = DefaultCanonicalizer::default();
        let result = canon.canonicalize(&input);

        // Should keep first 20 lines
        assert!(result.contains("code line 0"));
        assert!(result.contains("code line 19"));
        // Should have omission marker
        assert!(result.contains("lines omitted"));
        // Should keep last 10 lines
        assert!(result.contains("code line 40"));
        assert!(result.contains("code line 49"));
        // Should NOT have middle lines
        assert!(!result.contains("code line 25"));
    }

    #[test]
    fn whitespace_normalization() {
        let canon = DefaultCanonicalizer::default();
        let input = "hello    world\n\n\nwith   multiple   spaces";
        let result = canon.canonicalize(input);
        // Multiple spaces should be collapsed
        assert!(!result.contains("  "));
        assert!(result.contains("hello"));
        assert!(result.contains("world"));
    }

    #[test]
    fn strip_italic_underscores_matches_reference() {
        // Reference: the prior Vec<char> + Vec<bool> + collect implementation.
        // The single-pass version must be byte-identical for every input,
        // especially snake_case (underscores kept) vs `_italic_` (markers dropped).
        fn reference(text: &str) -> String {
            let chars: Vec<char> = text.chars().collect();
            let n = chars.len();
            let mut keep = vec![true; n];
            let is_word = |c: char| c.is_alphanumeric() || c == '_';
            for i in 0..n {
                if chars[i] != '_' {
                    continue;
                }
                let prev_is_word = i > 0 && is_word(chars[i - 1]) && chars[i - 1] != '_';
                let next_is_word = i + 1 < n && is_word(chars[i + 1]) && chars[i + 1] != '_';
                if (!prev_is_word && next_is_word) || (prev_is_word && !next_is_word) {
                    keep[i] = false;
                }
            }
            chars
                .into_iter()
                .zip(keep)
                .filter_map(|(c, k)| if k { Some(c) } else { None })
                .collect()
        }
        let cases = [
            "",
            "_",
            "__",
            "snake_case_variable",
            "_italic_",
            "_emphasized text_",
            "a _b_ c",
            "leading_ and _trailing",
            "mixed snake_case and _italic_ together",
            "fn compute_value(a_b, c_d) -> retry_count",
            "naïve_façade_test", // non-ASCII word chars around underscores
            "x_1_2_y",
            "_a_b_c_",
            "trailing_",
            "_leading",
        ];
        for c in cases {
            assert_eq!(strip_italic_underscores(c), reference(c), "input={c:?}");
        }
    }

    #[test]
    fn low_signal_filtered() {
        let canon = DefaultCanonicalizer::default();
        assert_eq!(canon.canonicalize("OK"), "");
        assert_eq!(canon.canonicalize("Done."), "");
        assert_eq!(canon.canonicalize("Got it."), "");
        assert_eq!(canon.canonicalize("Thanks!"), "Thanks!"); // Not exact match
    }

    #[test]
    fn truncate_long_text() {
        let canon = DefaultCanonicalizer {
            max_length: 50,
            ..Default::default()
        };
        let input = "a".repeat(100);
        let result = canon.canonicalize(&input);
        assert_eq!(result.chars().count(), 50);
    }

    #[test]
    fn truncate_at_char_boundary() {
        let canon = DefaultCanonicalizer {
            max_length: 4,
            ..Default::default()
        };
        // "café" is 5 bytes but 4 chars; truncating at 4 chars should produce "café"
        let input = "café!extra";
        let result = canon.canonicalize(input);
        assert!(result.chars().count() <= 4);
    }

    #[test]
    fn query_canonicalization_trims() {
        let canon = DefaultCanonicalizer::default();
        let result = canon.canonicalize_query("  hello world  ");
        assert_eq!(result, "hello world");
    }

    #[test]
    fn query_canonicalization_nfc() {
        let canon = DefaultCanonicalizer::default();
        let input = "caf\u{0065}\u{0301}";
        let result = canon.canonicalize_query(input);
        assert!(result.contains("caf\u{00e9}"));
    }

    #[test]
    fn empty_input() {
        let canon = DefaultCanonicalizer::default();
        let result = canon.canonicalize("");
        assert_eq!(result, "");
    }

    #[test]
    fn unclosed_code_block() {
        let canon = DefaultCanonicalizer::default();
        let input = "text\n```\ncode line 1\ncode line 2";
        let result = canon.canonicalize(input);
        assert!(result.contains("code line 1"));
        assert!(result.contains("code line 2"));
    }

    #[test]
    fn default_config_exact_values() {
        let canon = DefaultCanonicalizer::default();
        assert_eq!(canon.max_length, 2000);
        assert_eq!(canon.code_head_lines, 20);
        assert_eq!(canon.code_tail_lines, 10);
    }

    #[test]
    fn multiple_code_blocks_independently_collapsed() {
        let mut input = String::from("intro\n```\n");
        for i in 0..5 {
            let _ = writeln!(input, "block1 line {i}");
        }
        input.push_str("```\nmiddle text\n```\n");
        for i in 0..5 {
            let _ = writeln!(input, "block2 line {i}");
        }
        input.push_str("```\nend");

        let canon = DefaultCanonicalizer::default();
        let result = canon.canonicalize(&input);
        assert!(result.contains("block1 line 0"));
        assert!(result.contains("block2 line 0"));
        assert!(result.contains("middle text"));
    }

    #[test]
    fn nested_markdown_bold_inside_link() {
        let canon = DefaultCanonicalizer::default();
        let input = "See [**important** docs](https://example.com) here";
        let result = canon.canonicalize(input);
        assert!(result.contains("important"));
        assert!(result.contains("docs"));
        assert!(!result.contains("https://"));
    }

    #[test]
    fn all_heading_levels_stripped() {
        let canon = DefaultCanonicalizer::default();
        let input = "# H1\n## H2\n### H3\n#### H4\n##### H5\n###### H6";
        let result = canon.canonicalize(input);
        assert!(result.contains("H1"));
        assert!(result.contains("H6"));
    }

    #[test]
    fn language_tagged_code_block() {
        let canon = DefaultCanonicalizer::default();
        let input = "text\n```rust\nfn main() {}\n```\nmore";
        let result = canon.canonicalize(input);
        assert!(result.contains("fn main()"));
        assert!(result.contains("more"));
    }

    #[test]
    fn blank_lines_collapsed_via_whitespace_normalization() {
        let canon = DefaultCanonicalizer::default();
        let input = "paragraph one\n\nparagraph two";
        let result = canon.canonicalize(input);
        // Whitespace normalization collapses newlines to single space
        assert!(result.contains("paragraph one"));
        assert!(result.contains("paragraph two"));
    }

    #[test]
    fn query_truncation_respects_max_length() {
        let canon = DefaultCanonicalizer {
            max_length: 10,
            ..Default::default()
        };
        let result = canon.canonicalize_query("a very long query that should be truncated");
        assert!(result.chars().count() <= 10);
    }

    #[test]
    fn canonicalizer_trait_is_object_safe() {
        let canon: Box<dyn Canonicalizer> = Box::new(DefaultCanonicalizer::default());
        let result = canon.canonicalize("## Hello **world**");
        assert!(result.contains("Hello"));
        assert!(result.contains("world"));
        assert!(!result.contains("##"));
    }

    #[test]
    fn large_document_pipeline_completes() {
        let canon = DefaultCanonicalizer::default();
        let mut input = String::new();
        for i in 0..500 {
            let _ = writeln!(input, "Line {i} with some content for testing");
        }
        let result = canon.canonicalize(&input);
        assert!(result.chars().count() <= canon.max_length);
        assert!(!result.is_empty());
    }

    #[test]
    fn emoji_preserved() {
        let canon = DefaultCanonicalizer::default();
        let input = "Hello 👋 World 🌍";
        let result = canon.canonicalize(input);
        assert!(result.contains('👋'));
        assert!(result.contains('🌍'));
    }

    #[test]
    fn nested_markdown_links_with_parens() {
        let canon = DefaultCanonicalizer::default();
        let input = "See [link with (parens)](http://example.com/path(1))";
        let result = canon.canonicalize(input);
        assert!(result.contains("link with (parens)"));
        assert!(!result.contains("http"));
    }

    #[test]
    fn unbalanced_link_preserves_content() {
        let canon = DefaultCanonicalizer::default();
        let input = "Check [link](url( unbalanced. Next sentence.";
        let result = canon.canonicalize(input);
        assert!(
            result.contains("Next sentence"),
            "Should not swallow content"
        );
        assert!(result.contains("unbalanced"), "Should not swallow content");
    }
}
