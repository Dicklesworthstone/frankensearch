#![no_main]

use asupersync::{Cx, test_utils::run_test_with_cx};
use frankensearch_quill::QuillIndexError;
use frankensearch_quill_gauntlet::{
    ComparatorConfig, ComparisonMode, ComparisonReport, ComparisonStatus, DifferentialCase,
    DifferentialCaseMetadata, DifferentialHarness, Divergence, DivergenceClass, GauntletEngine,
    GauntletError, GeneratedDocument, RankClass, ScoreEpsilonReason, SyntheticCorpus,
    SyntheticCorpusSpec, ZipfExponent, scalar_g1a_fuzz_pair,
};
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 64;
const GENERATOR_ID: &str = "typed-query-tree-fuzz-v2";
const SHRINK_FUEL: usize = 64;
const FUZZ_DOCUMENT_COUNT: u64 = 16;
const FUZZ_VOCABULARY_SIZE: u32 = 32;
const FUZZ_DOCUMENT_BYTES: u32 = 256;
const OVERSIZED_TOKEN_BYTES: usize = 65_531;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum QueryTree {
    Empty,
    Term(u8),
    Phrase(u8, u8),
    NegatedTerm(u8),
    Boolean(u8, u8, u8),
    NestedBoolean(u8, u8, u8),
    Fielded(u8),
    BoostedTerm(u8),
    Slop(u8, u8),
    PhrasePrefix(u8, u8),
    UnterminatedPhrase(u8, u8),
    EscapedTerm(u8, u8),
    TrailingBoolean(u8),
    NonFiniteBoost(u8),
    OversizedToken(u8),
}

impl QueryTree {
    fn from_input(input: &[u8]) -> Self {
        let byte = |index| input.get(index).copied().unwrap_or(0);
        let kind = byte(0) % 15;
        let first = byte(1);
        let second = byte(2);
        match kind {
            0 => Self::Empty,
            1 => Self::Term(first),
            2 => Self::Phrase(first, second),
            3 => Self::NegatedTerm(first),
            4 => Self::Boolean(first, second, byte(3)),
            5 => Self::NestedBoolean(first, second, byte(3)),
            6 => Self::Fielded(first),
            7 => Self::BoostedTerm(first),
            8 => Self::Slop(first, second),
            9 => Self::PhrasePrefix(first, second),
            10 => Self::UnterminatedPhrase(first, second),
            11 => Self::EscapedTerm(first, second),
            12 => Self::TrailingBoolean(first),
            13 => Self::NonFiniteBoost(first),
            _ => Self::OversizedToken(first),
        }
    }

    fn render(self, vocabulary: &[String]) -> String {
        let word = |index: u8| vocabulary[usize::from(index) % vocabulary.len()].as_str();
        match self {
            Self::Empty => String::new(),
            Self::Term(term) => word(term).to_owned(),
            Self::Phrase(first, second) => format!("\"{} {}\"", word(first), word(second)),
            Self::NegatedTerm(term) => format!("-{}", word(term)),
            Self::Boolean(first, second, connective) => {
                let operator = if connective.is_multiple_of(2) {
                    "AND"
                } else {
                    "OR"
                };
                format!("{} {operator} {}", word(first), word(second))
            }
            Self::NestedBoolean(first, second, third) => {
                format!("({} OR {}) AND {}", word(first), word(second), word(third))
            }
            Self::Fielded(term) => format!("content:{}", word(term)),
            Self::BoostedTerm(term) => format!("{}^2", word(term)),
            Self::Slop(first, second) => format!("\"{} {}\"~1", word(first), word(second)),
            Self::PhrasePrefix(first, second) => {
                format!("\"{} {}\"*", word(first), word(second))
            }
            Self::UnterminatedPhrase(first, second) => format!("\"{} {}", word(first), word(second)),
            Self::EscapedTerm(first, second) => format!(r"{}\:{}", word(first), word(second)),
            Self::TrailingBoolean(term) => format!("{} OR", word(term)),
            Self::NonFiniteBoost(term) => format!("{} {}^{}", word(term), word(term), "9".repeat(400)),
            Self::OversizedToken(term) => {
                let suffix_len = OVERSIZED_TOKEN_BYTES.saturating_sub(word(term).len());
                format!("{}{}", word(term), "x".repeat(suffix_len))
            }
        }
    }

    const fn exact_refusal_detail(self) -> Option<&'static str> {
        match self {
            Self::Slop(..) => Some("phrase slop=1 prefix=false"),
            Self::PhrasePrefix(..) => Some("phrase slop=0 prefix=true"),
            Self::Empty
            | Self::Term(..)
            | Self::Phrase(..)
            | Self::NegatedTerm(..)
            | Self::Boolean(..)
            | Self::NestedBoolean(..)
            | Self::Fielded(..)
            | Self::BoostedTerm(..)
            | Self::UnterminatedPhrase(..)
            | Self::EscapedTerm(..)
            | Self::TrailingBoolean(..)
            | Self::NonFiniteBoost(..)
            | Self::OversizedToken(..) => None,
        }
    }

    const fn is_nonfinite_boost(self) -> bool {
        matches!(self, Self::NonFiniteBoost(..))
    }

    const fn is_reviewed_oversized_lowering(self) -> bool {
        matches!(self, Self::OversizedToken(..))
    }

    fn shrink_candidates(self) -> Vec<Self> {
        let mut candidates = match self {
            Self::Empty => Vec::new(),
            Self::Term(term) => vec![Self::Term(0).filter_not_equal(self, term != 0)],
            Self::Phrase(first, second) => vec![
                Some(Self::Term(first)),
                Some(Self::Term(second)),
                (first != 0 || second != 0).then_some(Self::Phrase(0, 0)),
            ],
            Self::NegatedTerm(term) => vec![
                Some(Self::Term(term)),
                (term != 0).then_some(Self::NegatedTerm(0)),
            ],
            Self::Boolean(first, second, _) => vec![
                Some(Self::Term(first)),
                Some(Self::Term(second)),
                (first != 0 || second != 0).then_some(Self::Boolean(0, 0, 0)),
            ],
            Self::NestedBoolean(first, second, third) => vec![
                Some(Self::Boolean(first, second, 1)),
                Some(Self::Boolean(second, third, 0)),
                Some(Self::Term(third)),
                (first != 0 || second != 0 || third != 0).then_some(Self::NestedBoolean(0, 0, 0)),
            ],
            Self::Fielded(term) => vec![
                Some(Self::Term(term)),
                (term != 0).then_some(Self::Fielded(0)),
            ],
            Self::BoostedTerm(term) => vec![
                Some(Self::Term(term)),
                (term != 0).then_some(Self::BoostedTerm(0)),
            ],
            Self::Slop(first, second) => vec![
                (first != 0 || second != 0).then_some(Self::Slop(0, 0)),
            ],
            Self::PhrasePrefix(first, second) => vec![
                (first != 0 || second != 0).then_some(Self::PhrasePrefix(0, 0)),
            ],
            Self::UnterminatedPhrase(first, second) => vec![
                Some(Self::UnterminatedPhrase(0, 0)).filter_not_equal(self, first != 0 || second != 0),
            ],
            Self::EscapedTerm(first, second) => vec![
                Some(Self::EscapedTerm(0, 0)).filter_not_equal(self, first != 0 || second != 0),
            ],
            Self::TrailingBoolean(term) => vec![
                (term != 0).then_some(Self::TrailingBoolean(0)),
            ],
            Self::NonFiniteBoost(term) => vec![
                (term != 0).then_some(Self::NonFiniteBoost(0)),
            ],
            Self::OversizedToken(term) => vec![
                (term != 0).then_some(Self::OversizedToken(0)),
            ],
        }
        .into_iter()
        .flatten()
        .filter(|candidate| *candidate != self)
        .collect::<Vec<_>>();
        candidates.sort_unstable_by_key(|candidate| format!("{candidate:?}"));
        candidates.dedup();
        candidates
    }
}

trait CandidateOption {
    fn filter_not_equal(self, original: QueryTree, include: bool) -> Option<QueryTree>;
}

impl CandidateOption for QueryTree {
    fn filter_not_equal(self, original: QueryTree, include: bool) -> Option<QueryTree> {
        include.then_some(self).filter(|candidate| *candidate != original)
    }
}

impl CandidateOption for Option<QueryTree> {
    fn filter_not_equal(self, original: QueryTree, include: bool) -> Option<QueryTree> {
        include.then_some(self?).filter(|candidate| *candidate != original)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct FailureFingerprint {
    status: ComparisonStatus,
    rank_class: RankClass,
    first_divergence: Option<String>,
    divergences: Vec<Divergence>,
}

impl FailureFingerprint {
    fn from_report(report: &ComparisonReport) -> Self {
        Self {
            status: report.status,
            rank_class: report.rank_class,
            first_divergence: report.first_divergence.clone(),
            divergences: report.divergences.clone(),
        }
    }
}

fn seed_from(input: &[u8]) -> u64 {
    input.iter().fold(0x6273_6a77_0002_f29b_6a5d_u64, |state, byte| {
        state
            .rotate_left(5)
            .wrapping_mul(0x1000_0000_01b3)
            ^ u64::from(*byte)
    })
}

fn input_hex(input: &[u8]) -> String {
    let mut encoded = String::with_capacity(input.len().saturating_mul(2));
    for byte in input {
        use std::fmt::Write as _;

        write!(&mut encoded, "{byte:02x}").expect("writing into a String is infallible");
    }
    encoded
}

fn seeded_vocabulary(documents: &[GeneratedDocument]) -> Vec<String> {
    let mut vocabulary = documents
        .iter()
        .filter(|document| document.pathology.is_none())
        .flat_map(|document| document.content.split_whitespace())
        .filter(|word| {
            word.strip_prefix("term")
                .is_some_and(|suffix| !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit()))
        })
        .map(str::to_owned)
        .collect::<Vec<_>>();
    vocabulary.sort_unstable();
    vocabulary.dedup();
    assert!(
        !vocabulary.is_empty(),
        "the seeded regular corpus must provide searchable terms for the generated AST"
    );
    vocabulary
}

fn provenance(input_hex: &str, seed: u64, tree: QueryTree, corpus_hash: &str) -> String {
    format!(
        "generator={GENERATOR_ID};input={input_hex};corpus_seed={seed:016x};docs={FUZZ_DOCUMENT_COUNT};vocab={FUZZ_VOCABULARY_SIZE};zipf=s11;bytes={FUZZ_DOCUMENT_BYTES};ast={tree:?};manifest={corpus_hash}"
    )
}

fn differential_case(
    input_hex: &str,
    seed: u64,
    tree: QueryTree,
    query: String,
    corpus_hash: &str,
) -> DifferentialCase {
    DifferentialCase {
        fixture_id: format!("typed-query-tree-v2-{input_hex}-{tree:?}"),
        query,
        limit: 20,
        offset: 0,
        tie_expansion_limit: 256,
        count_requested: true,
        snippet_max_chars: None,
        metadata: DifferentialCaseMetadata {
            generator_id: Some(provenance(input_hex, seed, tree, corpus_hash)),
            generator_seed: Some(seed),
            corpus_hash: Some(corpus_hash.to_owned()),
        },
    }
}

async fn run_supported_case(
    cx: &Cx,
    documents: &[frankensearch_core::IndexableDocument],
    case: &DifferentialCase,
    comparator_config: ComparatorConfig,
) -> Result<ComparisonReport, GauntletError> {
    let (subject, oracle) = scalar_g1a_fuzz_pair(cx, documents).await?;
    DifferentialHarness::new(ComparisonMode::CrossEngine, comparator_config)
        .run(cx, &subject, &oracle, case)
        .await
        .map(|run| run.comparison)
}

fn assert_exact_refusal(error: GauntletError, expected_detail: &str, context: &str) {
    match error {
        GauntletError::Quill(QuillIndexError::UnsupportedQuery { detail })
            if detail == expected_detail => {}
        other => panic!(
            "{context} must be the exact typed UnsupportedQuery({expected_detail:?}) refusal, got {other:?}"
        ),
    }
}

fn assert_nonfinite_oracle_refusal(error: GauntletError, documents: &[GeneratedDocument]) {
    let GauntletError::InvalidObservation { reason } = error else {
        panic!("non-finite boost must fail as an oracle observation refusal");
    };
    let prefix = "oracle.hits has a non-finite score for ";
    let Some(doc_id) = reason.strip_prefix(prefix) else {
        panic!("non-finite boost must retain the exact oracle hits refusal, got {reason:?}");
    };
    assert!(
        documents.iter().any(|document| document.id == doc_id),
        "non-finite oracle refusal must name an indexed document, got {doc_id:?}"
    );
}

fn assert_reviewed_oversized_lowering(report: &ComparisonReport) {
    assert_eq!(report.status, ComparisonStatus::Classified);
    assert_eq!(report.rank_class, RankClass::RankExact);
    assert_eq!(report.divergences.len(), 1);
    assert_eq!(
        report.divergences[0].class,
        DivergenceClass::OversizedQueryToken
    );
    assert_eq!(
        report.divergences[0].pointer,
        "/comparison/subject/ast_differences/0"
    );
}

async fn shrink_preserving_fingerprint(
    cx: &Cx,
    documents: &[frankensearch_core::IndexableDocument],
    input_hex: &str,
    seed: u64,
    corpus_hash: &str,
    vocabulary: &[String],
    tree: QueryTree,
    case: DifferentialCase,
    report: ComparisonReport,
    comparator_config: ComparatorConfig,
) -> (QueryTree, DifferentialCase, ComparisonReport, usize) {
    let original_fingerprint = FailureFingerprint::from_report(&report);
    let mut current_tree = tree;
    let mut current_case = case;
    let mut current_report = report;
    let mut candidates_evaluated = 0_usize;
    let mut reduction_steps = 0_usize;

    while candidates_evaluated < SHRINK_FUEL {
        let mut reduced = false;
        for candidate_tree in current_tree.shrink_candidates() {
            if candidates_evaluated == SHRINK_FUEL {
                break;
            }
            candidates_evaluated += 1;
            let candidate_case = differential_case(
                input_hex,
                seed,
                candidate_tree,
                candidate_tree.render(vocabulary),
                corpus_hash,
            );
            let candidate_report = run_supported_case(
                cx,
                documents,
                &candidate_case,
                comparator_config,
            )
            .await
            .unwrap_or_else(|error| {
                panic!(
                    "AST shrink candidate must execute before comparison: provenance={} candidate={candidate_tree:?} error={error}",
                    provenance(input_hex, seed, candidate_tree, corpus_hash)
                )
            });
            if candidate_report.status == ComparisonStatus::Failed
                && FailureFingerprint::from_report(&candidate_report) == original_fingerprint
            {
                current_tree = candidate_tree;
                current_case = candidate_case;
                current_report = candidate_report;
                reduction_steps += 1;
                reduced = true;
                break;
            }
        }
        if !reduced {
            break;
        }
    }

    assert_eq!(
        FailureFingerprint::from_report(&current_report),
        original_fingerprint,
        "AST shrink must retain the original failure class, direction, and signature"
    );
    (current_tree, current_case, current_report, reduction_steps)
}

fuzz_target!(|input: &[u8]| {
    if input.len() > MAX_INPUT_BYTES {
        return;
    }
    let input_hex = input_hex(input);
    let tree = QueryTree::from_input(input);
    let seed = seed_from(input);
    run_test_with_cx(|cx| async move {
        let corpus = SyntheticCorpus::new(SyntheticCorpusSpec {
            seed,
            document_count: FUZZ_DOCUMENT_COUNT,
            vocabulary_size: FUZZ_VOCABULARY_SIZE,
            zipf_exponent: ZipfExponent::S11,
            max_document_bytes: FUZZ_DOCUMENT_BYTES,
        })
        .expect("bounded deterministic fuzz corpus");
        let manifest = corpus.manifest().expect("fuzz corpus manifest");
        corpus
            .verify_manifest(&manifest)
            .expect("seeded fuzz corpus must replay its recorded manifest");
        let corpus_hash = manifest.manifest_hash().expect("fuzz corpus manifest hash");
        let documents = corpus.iter().collect::<Vec<_>>();
        let vocabulary = seeded_vocabulary(&documents);
        let query = tree.render(&vocabulary);
        let case = differential_case(&input_hex, seed, tree, query, &corpus_hash);
        let indexable = documents
            .iter()
            .cloned()
            .map(frankensearch_core::IndexableDocument::from)
            .collect::<Vec<_>>();
        let comparator_config = ComparatorConfig::default()
            .with_score_epsilon_reason(ScoreEpsilonReason::SummationAssociation);

        if let Some(expected_detail) = tree.exact_refusal_detail() {
            let (subject, oracle) = scalar_g1a_fuzz_pair(&cx, &indexable)
                .await
                .expect("fresh committed pair for the typed-refusal lane");
            let subject_error = GauntletEngine::observe(&subject, &cx, &case)
                .await
                .expect_err("unsupported AST must refuse in Quill before comparison");
            assert_exact_refusal(subject_error, expected_detail, "Quill observation");
            GauntletEngine::observe(&oracle, &cx, &case)
                .await
                .unwrap_or_else(|error| {
                    panic!(
                        "pinned Tantivy oracle must execute the declared unsupported AST: provenance={} error={error}",
                        provenance(&input_hex, seed, tree, &corpus_hash)
                    )
                });
            let harness_error = DifferentialHarness::new(ComparisonMode::CrossEngine, comparator_config)
                .run(&cx, &subject, &oracle, &case)
                .await
                .expect_err("typed Quill refusal must not create a one-sided comparison");
            assert_exact_refusal(harness_error, expected_detail, "harness observation");
            return;
        }

        let result = run_supported_case(&cx, &indexable, &case, comparator_config).await;
        if tree.is_nonfinite_boost() {
            assert_nonfinite_oracle_refusal(
                result.expect_err("non-finite oracle scores must fail closed before comparison"),
                &documents,
            );
            return;
        }
        let report = result.unwrap_or_else(|error| {
            panic!(
                "supported or malformed AST failed before comparison: provenance={} query={:?} error={error}",
                provenance(&input_hex, seed, tree, &corpus_hash),
                case.query
            )
        });

        if tree.is_reviewed_oversized_lowering() {
            assert_reviewed_oversized_lowering(&report);
            return;
        }

        match report.status {
            ComparisonStatus::Exact => {}
            ComparisonStatus::Classified => assert!(
                !report.divergences.is_empty()
                    && report.divergences.iter().all(|divergence| {
                        matches!(
                            divergence.class,
                            DivergenceClass::ScoreEpsilon | DivergenceClass::TieOrder
                        )
                    }),
                "only reviewed automatic comparison classes may pass externally: provenance={} query={:?} divergences={:?}",
                provenance(&input_hex, seed, tree, &corpus_hash),
                case.query,
                report.divergences
            ),
            ComparisonStatus::Failed => {
                let original_fingerprint = FailureFingerprint::from_report(&report);
                let (minimized_tree, minimized_case, minimized_report, reduction_steps) =
                    shrink_preserving_fingerprint(
                        &cx,
                        &indexable,
                        &input_hex,
                        seed,
                        &corpus_hash,
                        &vocabulary,
                        tree,
                        case,
                        report,
                        comparator_config,
                    )
                    .await;
                panic!(
                    "unclassified Quill/Tantivy divergence: provenance={} original_fingerprint={original_fingerprint:?} minimized_ast={minimized_tree:?} minimized_query={:?} minimized_fingerprint={:?} shrink_steps={reduction_steps}",
                    provenance(&input_hex, seed, minimized_tree, &corpus_hash),
                    minimized_case.query,
                    FailureFingerprint::from_report(&minimized_report),
                );
            }
        }
    });
});
