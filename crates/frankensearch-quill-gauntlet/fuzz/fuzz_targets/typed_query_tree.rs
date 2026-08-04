#![no_main]

use asupersync::test_utils::run_test_with_cx;
use frankensearch_core::IndexableDocument;
use frankensearch_quill_gauntlet::{
    ComparatorConfig, ComparisonMode, ComparisonStatus, DifferentialCase, DifferentialHarness,
    ScoreEpsilonReason, scalar_g1a_fuzz_pair,
};
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 64;
const VOCABULARY: [&str; 6] = ["rust", "search", "parser", "index", "query", "engine"];

#[derive(Clone, Copy)]
enum QueryTree {
    Empty,
    Term(usize),
    Phrase(usize, usize),
    Negated(usize),
    Group(usize, usize),
    Slop(usize, usize),
    PhrasePrefix(usize, usize),
}

impl QueryTree {
    fn from_input(input: &[u8]) -> Self {
        let kind = input.first().copied().unwrap_or(0) % 7;
        let first = usize::from(input.get(1).copied().unwrap_or(0)) % VOCABULARY.len();
        let second = usize::from(input.get(2).copied().unwrap_or(1)) % VOCABULARY.len();
        match kind {
            0 => Self::Empty,
            1 => Self::Term(first),
            2 => Self::Phrase(first, second),
            3 => Self::Negated(first),
            4 => Self::Group(first, second),
            5 => Self::Slop(first, second),
            _ => Self::PhrasePrefix(first, second),
        }
    }

    fn render(self) -> String {
        match self {
            Self::Empty => String::new(),
            Self::Term(term) => VOCABULARY[term].to_owned(),
            Self::Phrase(first, second) => {
                format!("\"{} {}\"", VOCABULARY[first], VOCABULARY[second])
            }
            Self::Negated(term) => format!("-{}", VOCABULARY[term]),
            Self::Group(first, second) => {
                format!("({} OR {})", VOCABULARY[first], VOCABULARY[second])
            }
            Self::Slop(first, second) => {
                format!("\"{} {}\"~1", VOCABULARY[first], VOCABULARY[second])
            }
            Self::PhrasePrefix(first, second) => {
                format!("\"{} {}\"*", VOCABULARY[first], VOCABULARY[second])
            }
        }
    }

    const fn is_typed_refusal(self) -> bool {
        matches!(self, Self::Slop(..) | Self::PhrasePrefix(..))
    }
}

fuzz_target!(|input: &[u8]| {
    if input.len() > MAX_INPUT_BYTES {
        return;
    }
    let tree = QueryTree::from_input(input);
    let query = tree.render();
    run_test_with_cx(|cx| async move {
        let documents = [
            IndexableDocument::new("typed-fuzz-0", "rust search parser"),
            IndexableDocument::new("typed-fuzz-1", "query engine index"),
            IndexableDocument::new("typed-fuzz-2", "parser index search"),
        ];
        let (subject, oracle) = scalar_g1a_fuzz_pair(&cx, &documents)
            .await
            .expect("fresh live fuzz campaign pair");
        let case = DifferentialCase::new("typed-query-tree", query, 20);
        let harness = DifferentialHarness::new(
            ComparisonMode::CrossEngine,
            ComparatorConfig::default()
                .with_score_epsilon_reason(ScoreEpsilonReason::SummationAssociation),
        );
        let result = harness.run(&cx, &subject, &oracle, &case).await;
        if tree.is_typed_refusal() {
            assert!(
                result.is_err(),
                "unsupported typed AST must not fabricate a comparison"
            );
        } else {
            let run = result.expect("supported typed AST must execute on both live engines");
            assert_ne!(run.comparison.status, ComparisonStatus::Failed);
        }
    });
});
