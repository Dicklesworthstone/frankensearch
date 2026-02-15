//! Text canonicalization pipeline for frankensearch.
//!
//! All text is preprocessed before embedding to maximize search quality.
//! The default pipeline applies:
//! 1. NFC Unicode normalization (hash stability across representations)
//! 2. Markdown stripping (`#`, `**`, `*`, `_`, `[text](url)` → `text`)
//! 3. Code block collapsing (first 20 + last 10 lines of fenced blocks)
//! 4. Low-signal line filtering (pure URL lines, empty sections)
//! 5. Length truncation (default 2000 characters)
//!
//! Query canonicalization is simpler (NFC + trim only) since queries are
//! typically short natural language.

use unicode_normalization::UnicodeNormalization;

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
/// low-signal filtering, and length truncation in sequence.
pub struct DefaultCanonicalizer {
    /// Maximum character length for canonicalized text. Default: 2000.
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

impl DefaultCanonicalizer {
    /// Strip common markdown formatting characters while preserving text content.
    fn strip_markdown(text: &str) -> String {
        let mut result = String::with_capacity(text.len());
        let mut chars = text.chars().peekable();
        let mut at_line_start = true;

        while let Some(c) = chars.next() {
            match c {
                '\n' => {
                    result.push('\n');
                    at_line_start = true;
                }
                '#' if at_line_start => {
                    // Skip heading markers only at start of line.
                    while chars.peek() == Some(&'#') {
                        chars.next();
                    }
                    if chars.peek() == Some(&' ') {
                        chars.next();
                    }
                }
                '*' | '_' => {
                    // Skip bold/italic markers (**, *, __, _)
                    if chars.peek() == Some(&c) {
                        chars.next();
                    }
                    at_line_start = false;
                }
                '[' => {
                    // Convert [text](url) → text
                    let mut link_text = String::new();
                    for lc in chars.by_ref() {
                        if lc == ']' {
                            break;
                        }
                        link_text.push(lc);
                    }
                    if chars.peek() == Some(&'(') {
                        chars.next();
                        let mut depth = 1;
                        for lc in chars.by_ref() {
                            if lc == '(' {
                                depth += 1;
                            } else if lc == ')' {
                                depth -= 1;
                                if depth == 0 {
                                    break;
                                }
                            }
                        }
                    }
                    result.push_str(&link_text);
                    if !link_text.is_empty() {
                        at_line_start = false;
                    }
                }
                ' ' | '\t' if at_line_start => {
                    result.push(c);
                }
                _ => {
                    result.push(c);
                    at_line_start = false;
                }
            }
        }

        result
    }

    /// Collapse fenced code blocks, keeping first N + last M lines.
    fn collapse_code_blocks(&self, text: &str) -> String {
        let max_keep = self.code_head_lines + self.code_tail_lines;
        let mut result = String::with_capacity(text.len());
        let mut in_code_block = false;
        let mut code_lines: Vec<&str> = Vec::with_capacity(max_keep);

        for line in text.lines() {
            if line.trim_start().starts_with("```") {
                if in_code_block {
                    // End of code block — emit collapsed version
                    let total = code_lines.len();
                    if total <= max_keep {
                        for cl in &code_lines {
                            result.push_str(cl);
                            result.push('\n');
                        }
                    } else {
                        for cl in &code_lines[..self.code_head_lines] {
                            result.push_str(cl);
                            result.push('\n');
                        }
                        result.push_str("/* ... collapsed ... */\n");
                        for cl in &code_lines[total - self.code_tail_lines..] {
                            result.push_str(cl);
                            result.push('\n');
                        }
                    }
                    code_lines.clear();
                    in_code_block = false;
                } else {
                    in_code_block = true;
                }
                continue;
            }

            if in_code_block {
                code_lines.push(line);
            } else {
                result.push_str(line);
                result.push('\n');
            }
        }

        // Handle unclosed code block — emit what we have
        for cl in &code_lines {
            result.push_str(cl);
            result.push('\n');
        }

        result
    }

    /// Filter out low-signal lines.
    fn filter_low_signal(text: &str) -> String {
        text.lines()
            .filter(|line| {
                let trimmed = line.trim();
                // Keep blank lines for paragraph structure
                if trimmed.is_empty() {
                    return true;
                }
                // Filter pure URL lines (no surrounding text)
                if (trimmed.starts_with("http://") || trimmed.starts_with("https://"))
                    && !trimmed.contains(' ')
                {
                    return false;
                }
                true
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Truncate to `max_length` at a char boundary.
    fn truncate(text: &str, max_length: usize) -> String {
        if text.len() <= max_length {
            return text.to_string();
        }
        // Find a valid char boundary at or before max_length
        let mut end = max_length;
        while !text.is_char_boundary(end) && end > 0 {
            end -= 1;
        }
        text[..end].to_string()
    }
}

impl Canonicalizer for DefaultCanonicalizer {
    fn canonicalize(&self, text: &str) -> String {
        // 1. NFC Unicode normalization
        let normalized: String = text.nfc().collect();
        // 2. Markdown stripping
        let stripped = Self::strip_markdown(&normalized);
        // 3. Code block collapsing
        let collapsed = self.collapse_code_blocks(&stripped);
        // 4. Low-signal filtering
        let filtered = Self::filter_low_signal(&collapsed);
        // 5. Length truncation
        Self::truncate(&filtered, self.max_length)
    }

    fn canonicalize_query(&self, query: &str) -> String {
        // Queries are short — just NFC normalize and trim
        let normalized: String = query.nfc().collect();
        let trimmed = normalized.trim();
        Self::truncate(trimmed, self.max_length)
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
        let input = "C# and #hashtag\n## Heading";
        let result = canon.canonicalize(input);
        assert!(result.contains("C#"));
        assert!(result.contains("#hashtag"));
        assert!(result.contains("Heading"));
        assert!(!result.contains("##"));
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
    fn collapse_short_code_block() {
        let canon = DefaultCanonicalizer::default();
        let input = "text\n```\nline1\nline2\nline3\n```\nmore text";
        let result = canon.canonicalize(input);
        assert!(result.contains("line1"));
        assert!(result.contains("line3"));
        assert!(!result.contains("collapsed"));
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
        // Should have collapse marker
        assert!(result.contains("collapsed"));
        // Should keep last 10 lines
        assert!(result.contains("code line 40"));
        assert!(result.contains("code line 49"));
        // Should NOT have middle lines
        assert!(!result.contains("code line 25"));
    }

    #[test]
    fn filter_pure_url_lines() {
        let canon = DefaultCanonicalizer::default();
        let input = "text\nhttps://example.com\nmore text";
        let result = canon.canonicalize(input);
        assert!(result.contains("text"));
        assert!(!result.contains("https://example.com"));
        assert!(result.contains("more text"));
    }

    #[test]
    fn keep_urls_with_text() {
        let canon = DefaultCanonicalizer::default();
        let input = "Visit https://example.com for details";
        let result = canon.canonicalize(input);
        // URL with surrounding text should be kept
        assert!(result.contains("https://example.com"));
    }

    #[test]
    fn truncate_long_text() {
        let canon = DefaultCanonicalizer {
            max_length: 50,
            ..Default::default()
        };
        let input = "a".repeat(100);
        let result = canon.canonicalize(&input);
        assert_eq!(result.len(), 50);
    }

    #[test]
    fn truncate_at_char_boundary() {
        let canon = DefaultCanonicalizer {
            max_length: 5,
            ..Default::default()
        };
        // "café" is 5 bytes (é is 2 bytes), truncating at 5 should work
        let input = "café!extra";
        let result = canon.canonicalize(input);
        assert!(result.len() <= 5);
        assert!(result.is_char_boundary(result.len()));
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
        assert_eq!(result.trim(), "");
    }

    #[test]
    fn unclosed_code_block() {
        let canon = DefaultCanonicalizer::default();
        let input = "text\n```\ncode line 1\ncode line 2";
        let result = canon.canonicalize(input);
        // Unclosed code block content should still appear
        assert!(result.contains("code line 1"));
        assert!(result.contains("code line 2"));
    }

    // ─── bd-1p4b tests begin ───

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
        // Link text is collected raw (bold markers preserved inside links)
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
        assert!(!result.starts_with('#'));
        // No heading markers should remain at line starts
        for line in result.lines() {
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                assert!(!trimmed.starts_with("# "));
            }
        }
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
    fn http_url_lines_filtered() {
        let canon = DefaultCanonicalizer::default();
        let input = "text\nhttp://example.com\nhttps://other.com\nmore text";
        let result = canon.canonicalize(input);
        assert!(result.contains("text"));
        assert!(result.contains("more text"));
        assert!(!result.contains("http://example.com"));
        assert!(!result.contains("https://other.com"));
    }

    #[test]
    fn blank_lines_preserved_for_paragraph_structure() {
        let canon = DefaultCanonicalizer::default();
        let input = "paragraph one\n\nparagraph two";
        let result = canon.canonicalize(input);
        assert!(result.contains("paragraph one\n\nparagraph two"));
    }

    #[test]
    fn query_truncation_respects_max_length() {
        let canon = DefaultCanonicalizer {
            max_length: 10,
            ..Default::default()
        };
        let result = canon.canonicalize_query("a very long query that should be truncated");
        assert!(result.len() <= 10);
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
        assert!(result.len() <= canon.max_length);
        assert!(!result.is_empty());
    }

    // ─── bd-1p4b tests end ───
}
