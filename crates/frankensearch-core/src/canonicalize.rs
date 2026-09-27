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

/// NFC-normalize, borrowing the input whenever it is already NFC.
///
/// ASCII is always already in NFC (no codepoint has a decomposition/composition),
/// so for ASCII input — the common case for code and English prose — this skips
/// the unicode-normalization state machine entirely. This mirrors the ASCII fast
/// paths in Lucene/Tantivy's analysis chains. Non-ASCII text runs the UAX #15
/// quick check and is only decomposed and recomposed when that check cannot
/// prove it normalized. Output is byte-identical to `text.nfc().collect()` for
/// every input.
#[inline]
fn nfc_normalize(text: &str) -> Cow<'_, str> {
    if text.is_ascii() {
        // ASCII is already NFC, so borrow the input — no whole-document copy. The
        // next pipeline stage (`strip_markdown_and_code`) only needs a `&str` and
        // allocates its own buffer, so the prior `to_owned()` here was pure waste
        // on the common (ASCII) path.
        Cow::Borrowed(text)
    } else if unicode_normalization::is_nfc_quick(text.chars())
        == unicode_normalization::IsNormalized::Yes
    {
        // A UAX #15 quick-check "Yes" means the text is already NFC, which holds
        // for most non-ASCII prose and code (one arrow in a comment used to send
        // the whole file through decomposition and recomposition).
        Cow::Borrowed(text)
    } else {
        Cow::Owned(text.nfc().collect())
    }
}

impl Canonicalizer for DefaultCanonicalizer {
    fn canonicalize(&self, text: &str) -> String {
        // 1. NFC Unicode normalization (critical for hash stability). ASCII text
        //    borrows the input (no copy); non-ASCII allocates the normalized form.
        let normalized = nfc_normalize(text);
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
        // Queries are short — just NFC normalize and trim (ASCII borrows the input).
        let normalized = nfc_normalize(query);
        let trimmed = normalized.trim();
        truncate_to_chars(trimmed, self.max_length)
    }
}

impl DefaultCanonicalizer {
    /// Strip markdown formatting and collapse code blocks.
    fn strip_markdown_and_code(&self, text: &str) -> String {
        let mut result = String::with_capacity(text.len());
        let mut in_code_block = false;
        let mut code_block_lang = "";
        let mut code_lines: Vec<&str> = Vec::new();

        for line in text.lines() {
            if line.starts_with("```") {
                if in_code_block {
                    // End of code block — collapse it
                    push_collapsed_code_block(
                        &mut result,
                        code_block_lang,
                        &code_lines,
                        self.code_head_lines,
                        self.code_tail_lines,
                    );
                    result.push('\n');
                    code_lines.clear();
                    code_block_lang = "";
                    in_code_block = false;
                } else {
                    // Start of code block
                    in_code_block = true;
                    code_block_lang = code_block_language(line);
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
            push_collapsed_code_block(
                &mut result,
                code_block_lang,
                &code_lines,
                self.code_head_lines,
                self.code_tail_lines,
            );
            result.push('\n');
        }

        result
    }
}

/// Return the language suffix of a fenced-code opener.
///
/// The result is already a slice of the canonicalizer's input, so retaining it
/// for the duration of the block avoids a throwaway `String` allocation. The
/// caller has already checked that `line` starts with a fence.
#[inline]
fn code_block_language(line: &str) -> &str {
    line.trim_start_matches('`').trim()
}

/// Pre-borrow owned language extraction retained for same-binary benchmarks.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
#[must_use]
pub fn code_block_language_owned_bench(line: &str) -> String {
    code_block_language(line).to_owned()
}

/// Shipping borrowed language extraction exposed for same-binary benchmarks.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
#[must_use]
pub fn code_block_language_borrowed_bench(line: &str) -> &str {
    code_block_language(line)
}

/// Append `lines` joined by `'\n'` (no trailing newline) directly into `out` —
/// the `[&str]::join("\n")` semantics without the intermediate `String`.
fn push_joined<'a>(out: &mut String, mut lines: impl Iterator<Item = &'a str>) {
    if let Some(first) = lines.next() {
        out.push_str(first);
        for line in lines {
            out.push('\n');
            out.push_str(line);
        }
    }
}

/// Append a code block collapsed to first N + last M lines.
///
/// Writes into the canonicalizer's existing document buffer so the collapsed
/// bytes are never allocated in a temporary `String` and copied again by the
/// caller. The caller owns the trailing newline; this function appends only the
/// exact former `collapse_code_block` bytes.
fn push_collapsed_code_block(
    out: &mut String,
    lang: &str,
    lines: &[&str],
    head: usize,
    tail: usize,
) {
    use std::fmt::Write as _;

    let collapse = lines.len() > head + tail;
    out.push('[');
    if lang.is_empty() {
        out.push_str("code");
    } else {
        out.push_str("code: ");
        out.push_str(lang);
    }
    out.push_str("]\n");

    if collapse {
        push_joined(out, lines.iter().take(head).copied());
        let omitted = lines.len() - head - tail;
        let _ = write!(out, "\n[... {omitted} lines omitted ...]\n");
        push_joined(out, lines.iter().skip(lines.len() - tail).copied());
    } else {
        push_joined(out, lines.iter().copied());
    }
}

/// Pre-direct-append form retained as the exact same-binary comparator.
#[cfg(any(test, feature = "bench-internals"))]
fn collapse_code_block(lang: &str, lines: &[&str], head: usize, tail: usize) -> String {
    let collapse = lines.len() > head + tail;
    let label_len = if lang.is_empty() { 4 } else { 6 + lang.len() };
    let kept_bytes: usize = if collapse {
        lines
            .iter()
            .take(head)
            .chain(lines.iter().skip(lines.len() - tail))
            .map(|line| line.len() + 1)
            .sum()
    } else {
        lines.iter().map(|line| line.len() + 1).sum()
    };
    let mut out = String::with_capacity(label_len + 3 + kept_bytes + if collapse { 32 } else { 0 });
    push_collapsed_code_block(&mut out, lang, lines, head, tail);
    out
}

/// Pre-fusion `format!`/`join` form of [`collapse_code_block`], retained as the
/// exact-parity bench oracle (it double-copies the kept bytes: `join` then
/// `format!`).
#[cfg(any(test, feature = "bench-internals"))]
#[doc(hidden)]
#[must_use]
pub fn collapse_code_block_slow(lang: &str, lines: &[&str], head: usize, tail: usize) -> String {
    let lang_label = if lang.is_empty() {
        "code".to_string()
    } else {
        format!("code: {lang}")
    };

    if lines.len() <= head + tail {
        format!("[{lang_label}]\n{}", lines.join("\n"))
    } else {
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

/// Doc-hidden bench wrapper for the shipped one-pass [`collapse_code_block`].
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
#[must_use]
pub fn collapse_code_block_fast_bench(
    lang: &str,
    lines: &[&str],
    head: usize,
    tail: usize,
) -> String {
    collapse_code_block(lang, lines, head, tail)
}

/// Doc-hidden bench wrapper for direct append into the caller's output buffer.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub fn push_collapsed_code_block_fast_bench(
    out: &mut String,
    lang: &str,
    lines: &[&str],
    head: usize,
    tail: usize,
) {
    push_collapsed_code_block(out, lang, lines, head, tail);
}

/// Strip markdown formatting from a single line.
///
/// Returns a borrowed slice for the common case (plain line, no inline markdown),
/// so a plain document flows through `canonicalize` with no per-line allocation;
/// the inline-markdown slow path returns an owned `String`.
fn strip_markdown_line(line: &str) -> Cow<'_, str> {
    // One scan classifies which inline-markdown trigger chars are present. Each
    // transform below was previously run unconditionally and allocated a whole-line
    // copy even when its trigger char was absent (a no-op). Guarding each by its
    // trigger skips those no-op passes — e.g. a `snake_case` line (only `_`) no
    // longer pays the `**`/`*`/`` ` ``/link passes. Same order, so byte-identical.
    let mut has_star = false;
    let mut has_underscore = false;
    let mut has_backtick = false;
    let mut has_bracket = false;
    for b in line.bytes() {
        match b {
            b'*' => has_star = true,
            b'_' => has_underscore = true,
            b'`' => has_backtick = true,
            b'[' => has_bracket = true,
            _ => {}
        }
    }

    if !(has_star || has_underscore || has_backtick || has_bracket) {
        // Fast path: no inline markdown, so operate directly on the borrowed
        // `line`. The prefix/blockquote trims and list-marker strip all return
        // borrowed `&str` slices, so a plain line flows through with **zero**
        // allocations — only the caller's single `push_str` copies the bytes.
        return strip_prefixes_and_list_marker(line);
    }

    // Apply transforms in the original order, each guarded by its trigger char.
    let mut r: Cow<'_, str> = Cow::Borrowed(line);
    if has_star {
        r = Cow::Owned(r.replace("**", "")); // bold
    }
    if has_underscore {
        r = Cow::Owned(r.replace("__", "")); // bold via underscores
    }
    if has_star {
        r = Cow::Owned(r.replace('*', "")); // remaining italic stars
    }
    if has_underscore && let Some(stripped) = strip_italic_underscores(&r) {
        r = Cow::Owned(stripped); // italic underscores
    }
    if has_backtick {
        r = Cow::Owned(r.replace('`', "")); // inline code
    }
    if has_bracket {
        r = Cow::Owned(strip_markdown_links(&r)); // [text](url) → text
    }
    Cow::Owned(strip_prefixes_and_list_marker(&r).into_owned())
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
/// Strip italic underscore markers (`_word_`), borrowing when nothing is dropped.
///
/// Returns `None` when no `_` sits on an italic word boundary — the common case
/// for `snake_case` / `variable_name` lines, which the former always-`String`
/// version copied in full (identical to the input) despite dropping nothing. This
/// is the last strip in [`strip_markdown_line`]'s chain to gain the borrow-when-
/// unchanged shape its siblings already have. `Some(stripped)` is returned only
/// when at least one marker is removed; the buffer is lazily seeded from the
/// borrowed prefix at the first drop, so the built string is byte-identical to the
/// prior full copy. Same boundary test and order → byte-identical materialization.
fn strip_italic_underscores(text: &str) -> Option<String> {
    let is_word = |c: char| c.is_alphanumeric() || c == '_';

    let mut result: Option<String> = None;
    let mut prev: Option<char> = None;
    let mut chars = text.char_indices().peekable();
    while let Some((idx, c)) = chars.next() {
        let drop_marker = c == '_' && {
            let prev_is_word = prev.is_some_and(|p| is_word(p) && p != '_');
            let next_is_word = chars.peek().is_some_and(|&(_, n)| is_word(n) && n != '_');
            // Opening marker: preceded by non-word (or BOL), followed by word.
            // Closing marker: preceded by word, followed by non-word (or EOL).
            (!prev_is_word && next_is_word) || (prev_is_word && !next_is_word)
        };
        if drop_marker {
            // First drop: seed the buffer with the untouched borrowed prefix.
            result.get_or_insert_with(|| {
                let mut buf = String::with_capacity(text.len());
                buf.push_str(&text[..idx]);
                buf
            });
        } else if let Some(buf) = result.as_mut() {
            buf.push(c);
        }
        prev = Some(c);
    }
    result
}

/// Strip markdown links: `[text](url)` → `text`.
fn strip_markdown_links(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut result = String::with_capacity(text.len());
    let mut cursor = 0;

    // ASCII delimiter bytes cannot occur inside a UTF-8 multibyte code point,
    // so every index used to slice `text` below is a valid character boundary.
    while let Some(relative_open) = text[cursor..].find('[') {
        let open = cursor + relative_open;
        result.push_str(&text[cursor..open]);

        let mut bracket_depth = 1_usize;
        let mut scan = open + 1;
        let mut label_close = None;
        while scan < bytes.len() {
            match bytes[scan] {
                b'[' => bracket_depth += 1,
                b']' => {
                    bracket_depth -= 1;
                    if bracket_depth == 0 {
                        label_close = Some(scan);
                        break;
                    }
                }
                _ => {}
            }
            scan += 1;
        }

        let Some(close) = label_close else {
            result.push_str(&text[open..]);
            return result;
        };

        if bytes.get(close + 1) == Some(&b'(') {
            let mut paren_depth = 1_usize;
            scan = close + 2;
            let mut url_close = None;
            while scan < bytes.len() {
                match bytes[scan] {
                    b'(' => paren_depth += 1,
                    b')' => {
                        paren_depth -= 1;
                        if paren_depth == 0 {
                            url_close = Some(scan);
                            break;
                        }
                    }
                    _ => {}
                }
                scan += 1;
            }

            let Some(end) = url_close else {
                result.push_str(&text[open..]);
                return result;
            };
            result.push_str(&text[open + 1..close]);
            cursor = end + 1;
        } else {
            result.push_str(&text[open..=close]);
            cursor = close + 1;
        }
    }

    result.push_str(&text[cursor..]);
    result
}

/// Pre-source-slice markdown-link parser retained for same-binary parity and timing.
#[cfg(any(test, feature = "bench-internals"))]
fn strip_markdown_links_reused_buffers(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut link_text = String::new();
    let mut url_part = String::new();
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '[' {
            // Potential link start
            link_text.clear();
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
                url_part.clear();
                url_part.push('(');
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

/// Pre-scratch-reuse markdown-link parser retained for same-binary parity and timing.
#[cfg(any(test, feature = "bench-internals"))]
fn strip_markdown_links_fresh_buffers(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '[' {
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
                chars.next();
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
                    result.push_str(&link_text);
                } else {
                    result.push('[');
                    result.push_str(&link_text);
                    result.push(']');
                    result.push_str(&url_part);
                }
            } else {
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

/// Former per-link allocation path exposed for the retained A/B.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
#[must_use]
pub fn strip_markdown_links_fresh_buffers_bench(text: &str) -> String {
    strip_markdown_links_fresh_buffers(text)
}

/// Former scratch-reuse path exposed for the retained A/B.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
#[must_use]
pub fn strip_markdown_links_reused_buffers_bench(text: &str) -> String {
    strip_markdown_links_reused_buffers(text)
}

/// Shipping source-slice path exposed for the retained A/B.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
#[must_use]
pub fn strip_markdown_links_source_slices_bench(text: &str) -> String {
    strip_markdown_links(text)
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
    // Byte-level scan with an ASCII fast-path. The original `text.chars()` decoded EVERY char and ran
    // the Unicode `is_whitespace()` per char, then re-encoded each kept char via `push(c)`. Here an
    // ASCII byte (the common case) is classified by a cheap byte test and copied without a decode,
    // and only a non-ASCII lead byte decodes a char. For ASCII, `char::is_whitespace()` equals
    // `is_ascii_whitespace() || b == 0x0B` — U+000B (vertical tab) is Unicode `White_Space` but NOT
    // `u8::is_ascii_whitespace`, so it must be added back. Byte-for-byte identical output
    // (`normalize_whitespace_matches_slow`).
    let mut result = String::with_capacity(text.len());
    let mut prev_whitespace = true; // Start as true to trim leading
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b < 0x80 {
            if b == 0x0B || b.is_ascii_whitespace() {
                if !prev_whitespace {
                    result.push(' ');
                    prev_whitespace = true;
                }
            } else {
                result.push(char::from(b));
                prev_whitespace = false;
            }
            i += 1;
        } else {
            let ch = text[i..]
                .chars()
                .next()
                .unwrap_or(char::REPLACEMENT_CHARACTER);
            let len = ch.len_utf8();
            if ch.is_whitespace() {
                if !prev_whitespace {
                    result.push(' ');
                    prev_whitespace = true;
                }
            } else {
                result.push(ch);
                prev_whitespace = false;
            }
            i += len;
        }
    }

    // Drop any trailing space the loop pushed, in place — avoids a second full
    // allocation via `trim_end().to_string()`.
    let trimmed_len = result.trim_end().len();
    result.truncate(trimmed_len);
    result
}

/// Pre-byte-fast-path [`normalize_whitespace`], retained for the same-binary A/B + parity test.
#[cfg(any(test, feature = "bench-internals"))]
#[doc(hidden)]
#[must_use]
pub fn normalize_whitespace_slow(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut prev_whitespace = true;
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
    let trimmed_len = result.trim_end().len();
    result.truncate(trimmed_len);
    result
}

/// Doc-hidden bench wrapper for the shipped (byte-fast) `normalize_whitespace` (it is private).
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
#[must_use]
pub fn normalize_whitespace_fast_bench(text: &str) -> String {
    normalize_whitespace(text)
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
    // ASCII fast-path: if the first `max_chars` bytes are all ASCII, they are exactly `max_chars`
    // single-byte chars and byte `max_chars` is a char boundary — cut there directly, skipping the
    // O(max_chars) `char_indices` decode. `is_ascii()` is a SIMD byte scan; `max_chars < text.len()`
    // here, so the slice is valid. Byte-identical to the decode path (`truncate_to_chars_matches_slow`).
    if text.as_bytes()[..max_chars].is_ascii() {
        return text[..max_chars].to_owned();
    }
    for (count, (idx, _)) in text.char_indices().enumerate() {
        if count == max_chars {
            return text[..idx].to_owned();
        }
    }
    text.to_owned()
}

/// Pre-ASCII-fast-path [`truncate_to_chars`], retained for the same-binary A/B + parity test.
#[cfg(any(test, feature = "bench-internals"))]
#[doc(hidden)]
#[must_use]
pub fn truncate_to_chars_slow(text: &str, max_chars: usize) -> String {
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

/// Doc-hidden bench wrapper for the shipped (ASCII-fast) `truncate_to_chars` (it is private).
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
#[must_use]
pub fn truncate_to_chars_fast_bench(text: &str, max_chars: usize) -> String {
    truncate_to_chars(text, max_chars)
}

#[cfg(test)]
mod tests {
    use std::fmt::Write;

    use super::*;

    /// PARITY GATE: the byte-fast `normalize_whitespace` must equal the char-by-char original for
    /// ASCII, every ASCII whitespace byte (incl. the tricky U+000B vertical tab), Unicode whitespace
    /// (NBSP U+00A0, NEL U+0085, ideographic space U+3000, en/em spaces), runs, leading/trailing, and
    /// mixed ASCII/Unicode text.
    /// PARITY GATE: the ASCII-fast `truncate_to_chars` must equal the `char_indices` original — for
    /// content shorter/equal/longer than the cap, ASCII and multibyte, and cuts landing on an ASCII
    /// boundary, before a multibyte char, and mid-multibyte (must round down to the char boundary).
    #[test]
    fn truncate_to_chars_matches_slow() {
        let ascii = "abcdefghij".repeat(300); // 3000 bytes, all ASCII
        let uni = "aéb日cé".repeat(300); // multibyte at varied positions
        let lead = format!("{}{}", "x".repeat(50), "é".repeat(50)); // ascii then multibyte
        for text in [
            "",
            "hi",
            "hello world",
            ascii.as_str(),
            uni.as_str(),
            lead.as_str(),
        ] {
            for max_chars in [0usize, 1, 3, 49, 50, 51, 100, 500, 2000, 100_000] {
                assert_eq!(
                    truncate_to_chars(text, max_chars),
                    truncate_to_chars_slow(text, max_chars),
                    "text.len()={} max_chars={max_chars}",
                    text.len()
                );
            }
        }
    }

    #[test]
    fn collapse_code_block_matches_slow() {
        // Byte-identity proof: the one-pass builder equals the format!/join form
        // across empty/short/exactly-at-threshold/collapsed shapes and edge
        // head/tail values, with and without a language label.
        let body: Vec<&str> = (0..50)
            .map(|i| ["fn main() {", "    let x = 1;", "}", ""][i % 4])
            .collect();
        let langs = ["", "rust", "python-with-a-long-name"];
        let shapes: &[(&[&str], usize, usize)] = &[
            (&[], 20, 10),
            (&["only one line"], 20, 10),
            (&body[..5], 20, 10),  // short (< head+tail): full keep
            (&body[..30], 20, 10), // exactly head+tail: full keep
            (&body[..31], 20, 10), // one over: collapse
            (&body[..], 20, 10),   // large collapse
            (&body[..], 0, 5),     // head=0
            (&body[..], 5, 0),     // tail=0
            (&body[..], 0, 0),     // both zero -> collapse, no kept lines
        ];
        for lang in langs {
            for &(lines, head, tail) in shapes {
                assert_eq!(
                    collapse_code_block(lang, lines, head, tail),
                    collapse_code_block_slow(lang, lines, head, tail),
                    "lang={lang:?} len={} head={head} tail={tail}",
                    lines.len()
                );

                let prefix = "before\n";
                let expected = format!("{prefix}{}", collapse_code_block(lang, lines, head, tail));
                let mut appended = prefix.to_owned();
                push_collapsed_code_block(&mut appended, lang, lines, head, tail);
                assert_eq!(
                    appended,
                    expected,
                    "append parity: lang={lang:?} len={} head={head} tail={tail}",
                    lines.len()
                );
            }
        }
    }

    #[test]
    fn normalize_whitespace_matches_slow() {
        let cases = [
            "",
            "   ",
            "hello world",
            "  leading and   collapsed\ttabs\n\n and trailing  ",
            "a\u{0B}b\u{0C}c\u{0D}d\u{09}e", // vertical tab, form feed, CR, tab between letters
            "no\u{00A0}break\u{00A0}space",  // NBSP is Unicode whitespace
            "next\u{0085}line and \u{3000}ideographic",
            "café  déjà\tvu  \u{2003}em space",
            "日本語　テスト", // ideographic space U+3000 between CJK
            "mix\u{0B}\u{00A0}\u{09}collapse",
            "trailing unicode ws\u{00A0}",
        ];
        for text in cases {
            assert_eq!(
                normalize_whitespace(text),
                normalize_whitespace_slow(text),
                "mismatch for {text:?}"
            );
        }
    }

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
    fn nfc_normalize_borrows_already_normalized_text_and_fixes_the_rest() {
        use unicode_normalization::UnicodeNormalization;
        // Already NFC: borrowed, and equal to the reference.
        for text in [
            "// maps a → b (naïve café, 日本語, Ω, 한국어)",
            "emoji 🚀 ✅ ok",
            "Straße",
        ] {
            assert!(matches!(nfc_normalize(text), Cow::Borrowed(_)), "{text:?}");
            assert_eq!(nfc_normalize(text), text.nfc().collect::<String>());
        }
        // Not NFC: decomposed accents, combining marks out of canonical order,
        // Hangul jamo and a singleton (U+212B ANGSTROM SIGN -> U+00C5).
        for text in [
            "cafe\u{301} → x",
            "a\u{323}\u{302}b → ",
            "a\u{302}\u{323}b → ",
            "\u{1100}\u{1161}\u{11A8} 한",
            "\u{212B} unit",
        ] {
            let normalized = nfc_normalize(text);
            assert_eq!(normalized, text.nfc().collect::<String>(), "{text:?}");
            assert!(matches!(normalized, Cow::Owned(_)), "{text:?}");
        }
        // A deterministic sweep over mixed scripts and combining marks.
        let alphabet: Vec<char> = "aeoAÅé→ñ日本한\u{301}\u{302}\u{323}\u{308}\u{1161}\u{212B} \n"
            .chars()
            .collect();
        let mut state = 0x9E37_79B9_u32;
        for _ in 0..2_000 {
            let text: String = (0..12)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    alphabet[state as usize % alphabet.len()]
                })
                .collect();
            assert_eq!(
                nfc_normalize(&text),
                text.nfc().collect::<String>(),
                "{text:?}"
            );
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
    fn strip_markdown_link_source_slices_match_former_paths() {
        for input in [
            "",
            "plain text",
            "See [the docs](https://example.com/path) for details",
            "[one](a) [two [nested]](b(c)) [three](d)",
            "] before [empty]() after",
            "prefix [closed only] suffix",
            "prefix [unclosed suffix",
            "Check [link](url( unbalanced. Next sentence.",
            "Unicode [café 日](https://example.test/é) tail",
        ] {
            assert_eq!(
                super::strip_markdown_links(input),
                strip_markdown_links_reused_buffers(input),
                "source slices vs scratch reuse, input={input:?}",
            );
            assert_eq!(
                strip_markdown_links_reused_buffers(input),
                strip_markdown_links_fresh_buffers(input),
                "input={input:?}",
            );
        }
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
            let reference_out = reference(c);
            // `None` means "unchanged", equivalent to materializing the input.
            let got = strip_italic_underscores(c).unwrap_or_else(|| c.to_owned());
            assert_eq!(got, reference_out, "input={c:?}");
            // The borrow elision must fire exactly when nothing is stripped.
            assert_eq!(
                strip_italic_underscores(c).is_none(),
                reference_out == c,
                "borrow-elision parity for input={c:?}"
            );
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
        assert!(result.contains("[code: rust]"));
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
