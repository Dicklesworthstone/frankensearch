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

impl Canonicalizer for DefaultCanonicalizer {
    fn canonicalize(&self, text: &str) -> String {
        // 1. NFC Unicode normalization (critical for hash stability)
        let normalized: String = text.nfc().collect();
        // 2. Strip markdown and collapse code blocks
        let stripped = self.strip_markdown_and_code(&normalized);
        // 3. Normalize whitespace
        let ws_normalized = normalize_whitespace(&stripped);
        // 4. Filter low-signal content
        let filtered = filter_low_signal(&ws_normalized);
        // 5. Truncate to max length
        truncate_to_chars(&filtered, self.max_length)
    }

    fn canonicalize_query(&self, query: &str) -> String {
        // Queries are short — just NFC normalize and trim
        let normalized: String = query.nfc().collect();
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
fn strip_markdown_line(line: &str) -> String {
    let mut result = line.to_string();

    // Remove bold/italic markers
    result = result.replace("**", "");
    result = result.replace("__", "");
    result = result.replace('*', "");
    result = result.replace('_', " "); // Underscore often used in identifiers

    // Remove inline code backticks
    result = result.replace('`', "");

    // Convert links [text](url) to just text
    result = strip_markdown_links(&result);

    // Remove headers (# prefix)
    result = result.trim_start_matches('#').trim_start().to_string();

    // Remove blockquote prefix
    result = result.trim_start_matches('>').trim_start().to_string();

    // Remove list markers
    result = strip_list_marker(&result);

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

            for inner in chars.by_ref() {
                if inner == ']' {
                    found_close = true;
                    break;
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
fn strip_list_marker(line: &str) -> String {
    let trimmed = line.trim_start();

    // Check for unordered list markers: "- " or "+ "
    if let Some(rest) = trimmed.strip_prefix("- ") {
        return rest.to_string();
    }
    if let Some(rest) = trimmed.strip_prefix("+ ") {
        return rest.to_string();
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
        chars.next(); // consume the space
        return chars.collect();
    }

    // Not a list marker, return original
    line.to_string()
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

    // Trim trailing whitespace
    result.trim_end().to_string()
}

/// Filter out low-signal content.
///
/// If the entire text (after trimming and lowercasing) matches a known
/// low-signal pattern, returns empty string.
fn filter_low_signal(text: &str) -> String {
    let trimmed = text.trim();
    let lower = trimmed.to_lowercase();

    for pattern in LOW_SIGNAL_CONTENT {
        if lower == *pattern {
            return String::new();
        }
    }

    text.to_string()
}

/// Truncate string to at most N characters, respecting char boundaries.
fn truncate_to_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        text.to_string()
    } else {
        text.chars().take(max_chars).collect()
    }
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
