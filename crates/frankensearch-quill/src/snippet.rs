//! Native snippet generation over already-analyzed query terms.
//!
//! The kernel deliberately mirrors the pinned Tantivy 0.26.1 behavior used by
//! the incumbent lexical engine: token offsets are byte offsets, fragment
//! scores sum one document-frequency weight per matching token occurrence, and
//! equal-score fragments prefer the earliest (then shortest) window. Source
//! text is escaped before trusted caller-configured highlight tags are added.
//!
//! Retrieving the source text is intentionally outside this module. Some Quill
//! schemas retain content in `STOREDMETA`; others require their host to supply
//! canonical source bytes. The E7 integration layer owns that distinction.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::ops::Range;

use crate::grimoire::MAX_TERM_BYTES;
use crate::schema::Analyzer;
use crate::scribe::{AnalyzedToken, CassAnalyzer, FrankensearchTokenizer, TokenAnalyzer};

/// Incumbent `search_with_snippets` window default.
pub const DEFAULT_SNIPPET_MAX_CHARS: usize = 200;

/// Rendering controls for native snippet generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnippetConfig {
    /// Maximum token-fragment byte span.
    ///
    /// The name preserves the incumbent API, but the pinned behavior compares
    /// tokenizer byte offsets. A single token may exceed this value because
    /// fragment boundaries never split a token.
    pub max_chars: usize,
    /// Trusted markup inserted immediately before a highlighted range.
    pub highlight_prefix: String,
    /// Trusted markup inserted immediately after a highlighted range.
    pub highlight_postfix: String,
}

impl Default for SnippetConfig {
    fn default() -> Self {
        Self {
            max_chars: DEFAULT_SNIPPET_MAX_CHARS,
            highlight_prefix: "<b>".to_owned(),
            highlight_postfix: "</b>".to_owned(),
        }
    }
}

/// One already-analyzed query term used to score and highlight snippets.
///
/// Callers supply terms after query analysis and expansion. The document
/// frequency is the snapshot-wide frequency for the same field; it produces
/// the incumbent weight `1 / (1 + document_frequency)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnippetTerm {
    /// Normalized term text emitted by the field analyzer.
    pub text: String,
    /// Snapshot-wide document frequency for this field and term.
    pub document_frequency: u64,
}

impl SnippetTerm {
    /// Construct one post-analysis snippet term.
    pub fn new(text: impl Into<String>, document_frequency: u64) -> Self {
        Self {
            text: text.into(),
            document_frequency,
        }
    }
}

#[derive(Debug, Clone)]
enum BuiltInAnalyzer {
    FrankensearchDefault(FrankensearchTokenizer),
    CassHyphenNormalize(CassAnalyzer),
    CassPrefixNormalize(CassAnalyzer),
}

impl BuiltInAnalyzer {
    fn new(kind: Analyzer) -> Self {
        match kind {
            Analyzer::FrankensearchDefault => {
                Self::FrankensearchDefault(FrankensearchTokenizer::default())
            }
            Analyzer::CassHyphenNormalize => Self::CassHyphenNormalize(CassAnalyzer::default()),
            Analyzer::CassPrefixNormalize => Self::CassPrefixNormalize(CassAnalyzer::default()),
        }
    }

    fn analyze(&mut self, text: &str, sink: &mut dyn FnMut(&AnalyzedToken)) {
        match self {
            Self::FrankensearchDefault(analyzer) => {
                analyzer.analyze(Analyzer::FrankensearchDefault, text, sink);
            }
            Self::CassHyphenNormalize(analyzer) => {
                analyzer.analyze(Analyzer::CassHyphenNormalize, text, sink);
            }
            Self::CassPrefixNormalize(analyzer) => {
                analyzer.analyze(Analyzer::CassPrefixNormalize, text, sink);
            }
        }
    }
}

/// Reusable analyzer-driven snippet generator for one compiled query.
///
/// Construction owns and deduplicates the post-analysis terms once. The same
/// generator can then render every winning document without rebuilding the
/// term map or leaking analyzer state across documents.
#[derive(Debug, Clone)]
pub struct SnippetGenerator {
    analyzer: BuiltInAnalyzer,
    term_weights: BTreeMap<String, f32>,
    config: SnippetConfig,
}

impl SnippetGenerator {
    /// Compile a generator from post-analysis terms and their document counts.
    ///
    /// Empty terms and terms absent from the snapshot (`document_frequency ==
    /// 0`) are ignored, matching the incumbent query-term extraction path.
    /// Duplicate term text is retained once with its greatest valid weight.
    #[must_use]
    pub fn new(
        analyzer: Analyzer,
        terms: impl IntoIterator<Item = SnippetTerm>,
        config: SnippetConfig,
    ) -> Self {
        let mut term_weights = BTreeMap::<String, f32>::new();
        for term in terms {
            if term.text.is_empty() || term.document_frequency == 0 {
                continue;
            }
            let weight = 1.0 / (1.0 + term.document_frequency as f32);
            term_weights
                .entry(term.text)
                .and_modify(|current| *current = current.max(weight))
                .or_insert(weight);
        }
        Self {
            analyzer: BuiltInAnalyzer::new(analyzer),
            term_weights,
            config,
        }
    }

    /// Generate an oracle-compatible highlighted snippet.
    ///
    /// Returns `None` when the source is empty or none of the supplied terms
    /// occurs in the source field. This is the behavior used by the incumbent
    /// `search_with_snippets` wrapper.
    pub fn snippet(&mut self, source: &str) -> Option<String> {
        self.generate(source, MissingMatch::None)
    }

    /// Generate a highlighted snippet, falling back to an escaped prefix.
    ///
    /// This explicit variant serves preview callers that want useful text when
    /// a document matched another field. Keeping the fallback out of
    /// [`Self::snippet`] preserves exact incumbent search behavior.
    pub fn snippet_or_prefix(&mut self, source: &str) -> Option<String> {
        self.generate(source, MissingMatch::Prefix)
    }

    fn generate(&mut self, source: &str, missing_match: MissingMatch) -> Option<String> {
        let source = source.trim();
        if source.is_empty() {
            return None;
        }

        let mut current = FragmentCandidate::new(0);
        let mut best = None;
        let mut fallback_stop = 0;
        let max_chars = self.config.max_chars;
        let term_weights = &self.term_weights;

        self.analyzer.analyze(source, &mut |token| {
            debug_assert!(token.offset_from <= token.offset_to);
            debug_assert!(source.is_char_boundary(token.offset_from));
            debug_assert!(source.is_char_boundary(token.offset_to));

            if token.offset_to <= max_chars {
                fallback_stop = token.offset_to;
            }
            if token.offset_to.saturating_sub(current.start_offset) > max_chars {
                let completed =
                    std::mem::replace(&mut current, FragmentCandidate::new(token.offset_from));
                retain_better_candidate(&mut best, completed);
            }
            current.add_token(token, term_weights, token.text.len() <= MAX_TERM_BYTES);
        });
        retain_better_candidate(&mut best, current);

        if let Some(fragment) = best {
            return Some(render_fragment(source, &fragment, &self.config));
        }
        if missing_match == MissingMatch::None {
            return None;
        }

        let stop = if source.len() <= max_chars {
            source.len()
        } else if fallback_stop == 0 {
            prefix_char_boundary(source, max_chars)
        } else {
            fallback_stop
        };
        if stop == 0 {
            return None;
        }
        let mut escaped = String::new();
        push_escaped_html(&mut escaped, &source[..stop]);
        Some(escaped)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MissingMatch {
    None,
    Prefix,
}

#[derive(Debug)]
struct FragmentCandidate {
    score: f32,
    start_offset: usize,
    stop_offset: usize,
    highlighted: Vec<Range<usize>>,
}

impl FragmentCandidate {
    fn new(start_offset: usize) -> Self {
        Self {
            score: 0.0,
            start_offset,
            stop_offset: start_offset,
            highlighted: Vec::new(),
        }
    }

    fn add_token(
        &mut self,
        token: &AnalyzedToken,
        terms: &BTreeMap<String, f32>,
        score_eligible: bool,
    ) {
        self.stop_offset = token.offset_to;
        if score_eligible && let Some(score) = terms.get(token.text.as_str()) {
            self.score += score;
            self.highlighted.push(token.offset_from..token.offset_to);
        }
    }
}

fn retain_better_candidate(best: &mut Option<FragmentCandidate>, candidate: FragmentCandidate) {
    if candidate.score <= 0.0 {
        return;
    }
    if best
        .as_ref()
        .is_none_or(|current| compare_candidates(&candidate, current) == Ordering::Greater)
    {
        *best = Some(candidate);
    }
}

fn compare_candidates(left: &FragmentCandidate, right: &FragmentCandidate) -> Ordering {
    left.score
        .total_cmp(&right.score)
        .then_with(|| right.start_offset.cmp(&left.start_offset))
        .then_with(|| right.stop_offset.cmp(&left.stop_offset))
}

fn render_fragment(source: &str, fragment: &FragmentCandidate, config: &SnippetConfig) -> String {
    let text = &source[fragment.start_offset..fragment.stop_offset];
    let relative = fragment
        .highlighted
        .iter()
        .map(|range| range.start - fragment.start_offset..range.end - fragment.start_offset);
    let highlighted = collapse_overlapping_ranges(relative);
    let mut output = String::new();
    let mut cursor = 0;
    for range in highlighted {
        let (start, end) = (range.start, range.end);
        push_escaped_html(&mut output, &text[cursor..start]);
        output.push_str(&config.highlight_prefix);
        push_escaped_html(&mut output, &text[start..end]);
        output.push_str(&config.highlight_postfix);
        cursor = end;
    }
    push_escaped_html(&mut output, &text[cursor..]);
    output
}

fn collapse_overlapping_ranges(
    ranges: impl IntoIterator<Item = Range<usize>>,
) -> Vec<Range<usize>> {
    let mut ranges = ranges.into_iter().collect::<Vec<_>>();
    ranges.sort_unstable_by_key(|range| (range.start, range.end));
    ranges.dedup();

    let mut collapsed = Vec::<Range<usize>>::with_capacity(ranges.len());
    for range in ranges {
        if let Some(last) = collapsed.last_mut()
            && last.end > range.start
        {
            last.end = last.end.max(range.end);
        } else {
            collapsed.push(range);
        }
    }
    collapsed
}

fn prefix_char_boundary(text: &str, max_bytes: usize) -> usize {
    let mut boundary = max_bytes.min(text.len());
    while !text.is_char_boundary(boundary) {
        boundary -= 1;
    }
    boundary
}

fn push_escaped_html(output: &mut String, text: &str) {
    for character in text.chars() {
        match character {
            '"' => output.push_str("&quot;"),
            '&' => output.push_str("&amp;"),
            '\'' => output.push_str("&#x27;"),
            '<' => output.push_str("&lt;"),
            '>' => output.push_str("&gt;"),
            _ => output.push(character),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_SNIPPET_MAX_CHARS, FragmentCandidate, SnippetConfig, SnippetGenerator, SnippetTerm,
        collapse_overlapping_ranges, compare_candidates,
    };
    use crate::schema::Analyzer;
    #[cfg(feature = "tantivy-oracle")]
    use frankensearch_core::{IndexableDocument, LexicalSearch};
    #[cfg(feature = "tantivy-oracle")]
    use frankensearch_lexical::{SnippetConfig as OracleSnippetConfig, TantivyIndex};
    #[cfg(feature = "tantivy-oracle")]
    use std::collections::BTreeMap;

    fn terms(entries: &[(&str, u64)]) -> Vec<SnippetTerm> {
        entries
            .iter()
            .map(|(text, frequency)| SnippetTerm::new(*text, *frequency))
            .collect()
    }

    fn generator(
        analyzer: Analyzer,
        entries: &[(&str, u64)],
        max_chars: usize,
    ) -> SnippetGenerator {
        SnippetGenerator::new(
            analyzer,
            terms(entries),
            SnippetConfig {
                max_chars,
                ..SnippetConfig::default()
            },
        )
    }

    #[test]
    fn config_defaults_match_incumbent_api() {
        let config = SnippetConfig::default();
        assert_eq!(config.max_chars, DEFAULT_SNIPPET_MAX_CHARS);
        assert_eq!(config.highlight_prefix, "<b>");
        assert_eq!(config.highlight_postfix, "</b>");
    }

    #[test]
    fn unicode_window_uses_valid_token_byte_offsets() {
        let mut generator = generator(Analyzer::FrankensearchDefault, &[("éé", 1)], 6);
        assert_eq!(generator.snippet("éé alpha").as_deref(), Some("<b>éé</b>"));
    }

    #[test]
    fn document_frequency_weights_choose_the_rare_term_window() {
        let mut generator = generator(
            Analyzer::FrankensearchDefault,
            &[("common", 2), ("rust", 1)],
            12,
        );
        assert_eq!(
            generator.snippet("common alpha beta gamma rust").as_deref(),
            Some("<b>rust</b>")
        );
    }

    #[test]
    fn repeated_occurrences_each_contribute_to_fragment_score() {
        let mut generator = generator(
            Analyzer::FrankensearchDefault,
            &[("rare", 1), ("common", 3)],
            20,
        );
        assert_eq!(
            generator
                .snippet("rare xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx common common common")
                .as_deref(),
            Some("<b>common</b> <b>common</b> <b>common</b>")
        );
    }

    #[test]
    fn candidate_ties_prefer_earlier_then_shorter_windows() {
        let earlier = FragmentCandidate {
            score: 1.0,
            start_offset: 2,
            stop_offset: 12,
            highlighted: Vec::new(),
        };
        let later = FragmentCandidate {
            score: 1.0,
            start_offset: 3,
            stop_offset: 8,
            highlighted: Vec::new(),
        };
        assert!(compare_candidates(&earlier, &later).is_gt());

        let shorter = FragmentCandidate {
            score: 1.0,
            start_offset: 2,
            stop_offset: 9,
            highlighted: Vec::new(),
        };
        assert!(compare_candidates(&shorter, &earlier).is_gt());
    }

    #[test]
    fn overlap_collapses_but_adjacency_remains_separate() {
        assert_eq!(
            collapse_overlapping_ranges([0..4, 0..4, 2..6, 6..8]),
            vec![0..6, 6..8]
        );
    }

    #[test]
    fn cass_same_span_alternatives_render_one_highlight() {
        let mut generator = generator(
            Analyzer::CassHyphenNormalize,
            &[("error-handling", 1), ("error", 1), ("handling", 1)],
            200,
        );
        assert_eq!(
            generator.snippet("error-handling").as_deref(),
            Some("<b>error-handling</b>")
        );
    }

    #[test]
    fn source_is_minimally_escaped_and_custom_tags_are_verbatim() {
        let config = SnippetConfig {
            max_chars: 200,
            highlight_prefix: "<em>".to_owned(),
            highlight_postfix: "</em>".to_owned(),
        };
        let mut generator = SnippetGenerator::new(
            Analyzer::FrankensearchDefault,
            terms(&[
                ("rust", 1),
                ("café", 1),
                ("quote", 1),
                ("single", 1),
                ("end", 1),
            ]),
            config,
        );
        assert_eq!(
            generator
                .snippet("Rust & <tag> café \"quote\" 'single' end")
                .as_deref(),
            Some(
                "<em>Rust</em> &amp; &lt;tag&gt; <em>café</em> &quot;<em>quote</em>&quot; &#x27;<em>single</em>&#x27; <em>end</em>"
            )
        );
    }

    #[test]
    fn exact_mode_and_explicit_prefix_fallback_remain_distinct() {
        let mut missing_generator = generator(Analyzer::FrankensearchDefault, &[("missing", 1)], 6);
        assert_eq!(missing_generator.snippet("alpha beta gamma"), None);
        assert_eq!(
            missing_generator
                .snippet_or_prefix("alpha beta gamma")
                .as_deref(),
            Some("alpha")
        );

        assert_eq!(missing_generator.snippet(" <&> "), None);
        assert_eq!(
            missing_generator.snippet_or_prefix(" <&> ").as_deref(),
            Some("&lt;&amp;&gt;")
        );
        let mut whole_prefix = generator(Analyzer::FrankensearchDefault, &[("missing", 1)], 7);
        assert_eq!(
            whole_prefix.snippet_or_prefix("(alpha)").as_deref(),
            Some("(alpha)")
        );
        assert_eq!(missing_generator.snippet_or_prefix("   "), None);
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn oversized_nonmatch_still_advances_oracle_fragment_geometry() {
        use frankensearch_lexical::tantivy_crate::{
            schema::Field, snippet::SnippetGenerator as TantivySnippetGenerator,
        };

        let source = format!("hit {}", "x".repeat(crate::grimoire::MAX_TERM_BYTES + 1));
        let max_chars = source.len();
        let oracle = TantivySnippetGenerator::new(
            BTreeMap::from([("hit".to_owned(), 0.5)]),
            frankensearch_lexical::default_tokenizer_for_bench(),
            Field::from_field_id(0),
            max_chars,
        );
        let oracle_html = oracle.snippet(&source).to_html();

        let mut native = generator(Analyzer::FrankensearchDefault, &[("hit", 1)], max_chars);
        assert_eq!(
            native.snippet(&source).as_deref(),
            Some(oracle_html.as_str())
        );
    }

    #[test]
    fn token_boundaries_win_over_zero_or_short_budgets() {
        let mut matched = generator(Analyzer::FrankensearchDefault, &[("alphabet", 1)], 0);
        assert_eq!(
            matched.snippet("alphabet").as_deref(),
            Some("<b>alphabet</b>")
        );

        let mut missing = generator(Analyzer::FrankensearchDefault, &[("missing", 1)], 3);
        assert_eq!(missing.snippet_or_prefix("éé alpha").as_deref(), Some("é"));
        let mut zero = generator(Analyzer::FrankensearchDefault, &[("missing", 1)], 0);
        assert_eq!(zero.snippet_or_prefix("alphabet"), None);
    }

    #[test]
    fn generator_reuse_does_not_leak_fragment_state() {
        let mut generator = generator(Analyzer::FrankensearchDefault, &[("rust", 1)], 8);
        assert_eq!(
            generator.snippet("rust alpha").as_deref(),
            Some("<b>rust</b>")
        );
        assert_eq!(generator.snippet("alpha beta"), None);
        assert_eq!(
            generator.snippet("gamma rust").as_deref(),
            Some("<b>rust</b>")
        );
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn native_output_matches_pinned_tantivy_oracle() {
        const TARGET: &str = "common alpha beta gamma rust";
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = TantivyIndex::in_memory().expect("create Tantivy oracle");
            index
                .index_documents(
                    &cx,
                    &[
                        IndexableDocument::new("target", TARGET),
                        IndexableDocument::new("common-only", "common elsewhere"),
                    ],
                )
                .await
                .expect("index oracle documents");
            index.commit(&cx).await.expect("commit oracle documents");

            let oracle = index
                .search_with_snippets(
                    &cx,
                    "rust common",
                    10,
                    &OracleSnippetConfig {
                        max_chars: 12,
                        highlight_prefix: "<b>".to_owned(),
                        highlight_postfix: "</b>".to_owned(),
                    },
                )
                .expect("run oracle snippet search")
                .into_iter()
                .find(|hit| hit.doc_id == "target")
                .and_then(|hit| hit.snippet)
                .expect("target oracle snippet");

            let mut native = generator(
                Analyzer::FrankensearchDefault,
                &[("common", 2), ("rust", 1)],
                12,
            );
            assert_eq!(native.snippet(TARGET).as_deref(), Some(oracle.as_str()));
        });
    }
}
