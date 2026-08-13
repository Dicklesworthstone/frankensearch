//! Quill scoring-contract constants vendored from the tantivy oracle.
//!
//! Quill (the ground-up lexical engine that replaces tantivy) must reproduce
//! tantivy's BM25 rank order exactly for equal-stats configurations (the
//! `RankExact` conformance class). BM25's `|d|` term comes from tantivy's
//! 1-byte log-scale fieldnorm (document length) quantization, so rank parity
//! requires reproducing that table and its encode/decode functions bit-for-bit.
//!
//! Provenance — vendored from tantivy 0.26.1 (pinned oracle, quill-e0.7):
//! - [`FIELD_NORMS_TABLE`], [`id_to_fieldnorm`], [`fieldnorm_to_id`]:
//!   `src/fieldnorm/code.rs` (table at line 13).
//! - [`BM25_K1`], [`BM25_B`], [`idf`], [`compute_tf_cache`]:
//!   `src/query/bm25.rs` (constants at lines 8-9, `idf` at line 52,
//!   tf cache at lines 58-66).
//!
//! Contract notes:
//! - Per-document `|d|` is decoded from the one-byte fieldnorm via
//!   [`id_to_fieldnorm`]. In contrast, `avgdl` is tantivy's raw
//!   `total_num_tokens / total_num_docs`; averaging decoded fieldnorm buckets
//!   would shift scores and is not conformant. The Language Contract
//!   (quill-e0.1) pins that distinction.
//! - The per-snapshot tf cache ([`compute_tf_cache`]) is tantivy's O(1)
//!   per-doc `tf_part` trick: one 256-entry table per (field, snapshot).
//!   Quill's evaluator (quill-e4.3) mirrors it for both parity and cost.
//! - Staging home: this module lives in `frankensearch-lexical` (the
//!   tantivy-facing crate, so pin tests can diff against the real oracle).
//!   Moving it to `frankensearch_quill::contract` is part of quill-e1.0's
//!   acceptance criteria.

/// BM25 `k1` parameter, pinned to tantivy 0.26.1 `src/query/bm25.rs:8`.
pub const BM25_K1: f32 = 1.2;

/// BM25 `b` parameter, pinned to tantivy 0.26.1 `src/query/bm25.rs:9`.
pub const BM25_B: f32 = 0.75;

/// Fieldnorm id → decoded fieldnorm (document length) table.
///
/// Vendored verbatim from tantivy 0.26.1 `src/fieldnorm/code.rs:13`
/// (`FIELD_NORMS_TABLE`), which uses the exact same log-scale scheme as
/// Lucene: exact lengths 0..=39, then geometrically widening buckets up to
/// `2_013_265_944`. Strictly increasing, so [`fieldnorm_to_id`] can bucket
/// via binary search.
pub const FIELD_NORMS_TABLE: [u32; 256] = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    42,
    44,
    46,
    48,
    50,
    52,
    54,
    56,
    60,
    64,
    68,
    72,
    76,
    80,
    84,
    88,
    96,
    104,
    112,
    120,
    128,
    136,
    144,
    152,
    168,
    184,
    200,
    216,
    232,
    248,
    264,
    280,
    312,
    344,
    376,
    408,
    440,
    472,
    504,
    536,
    600,
    664,
    728,
    792,
    856,
    920,
    984,
    1_048,
    1_176,
    1_304,
    1_432,
    1_560,
    1_688,
    1_816,
    1_944,
    2_072,
    2_328,
    2_584,
    2_840,
    3_096,
    3_352,
    3_608,
    3_864,
    4_120,
    4_632,
    5_144,
    5_656,
    6_168,
    6_680,
    7_192,
    7_704,
    8_216,
    9_240,
    10_264,
    11_288,
    12_312,
    13_336,
    14_360,
    15_384,
    16_408,
    18_456,
    20_504,
    22_552,
    24_600,
    26_648,
    28_696,
    30_744,
    32_792,
    36_888,
    40_984,
    45_080,
    49_176,
    53_272,
    57_368,
    61_464,
    65_560,
    73_752,
    81_944,
    90_136,
    98_328,
    106_520,
    114_712,
    122_904,
    131_096,
    147_480,
    163_864,
    180_248,
    196_632,
    213_016,
    229_400,
    245_784,
    262_168,
    294_936,
    327_704,
    360_472,
    393_240,
    426_008,
    458_776,
    491_544,
    524_312,
    589_848,
    655_384,
    720_920,
    786_456,
    851_992,
    917_528,
    983_064,
    1_048_600,
    1_179_672,
    1_310_744,
    1_441_816,
    1_572_888,
    1_703_960,
    1_835_032,
    1_966_104,
    2_097_176,
    2_359_320,
    2_621_464,
    2_883_608,
    3_145_752,
    3_407_896,
    3_670_040,
    3_932_184,
    4_194_328,
    4_718_616,
    5_242_904,
    5_767_192,
    6_291_480,
    6_815_768,
    7_340_056,
    7_864_344,
    8_388_632,
    9_437_208,
    10_485_784,
    11_534_360,
    12_582_936,
    13_631_512,
    14_680_088,
    15_728_664,
    16_777_240,
    18_874_392,
    20_971_544,
    23_068_696,
    25_165_848,
    27_263_000,
    29_360_152,
    31_457_304,
    33_554_456,
    37_748_760,
    41_943_064,
    46_137_368,
    50_331_672,
    54_525_976,
    58_720_280,
    62_914_584,
    67_108_888,
    75_497_496,
    83_886_104,
    92_274_712,
    100_663_320,
    109_051_928,
    117_440_536,
    125_829_144,
    134_217_752,
    150_994_968,
    167_772_184,
    184_549_400,
    201_326_616,
    218_103_832,
    234_881_048,
    251_658_264,
    268_435_480,
    301_989_912,
    335_544_344,
    369_098_776,
    402_653_208,
    436_207_640,
    469_762_072,
    503_316_504,
    536_870_936,
    603_979_800,
    671_088_664,
    738_197_528,
    805_306_392,
    872_415_256,
    939_524_120,
    1_006_632_984,
    1_073_741_848,
    1_207_959_576,
    1_342_177_304,
    1_476_395_032,
    1_610_612_760,
    1_744_830_488,
    1_879_048_216,
    2_013_265_944,
];

/// Decodes a 1-byte fieldnorm id to its fieldnorm (document length) value.
///
/// Mirrors tantivy 0.26.1 `src/fieldnorm/code.rs::id_to_fieldnorm`.
#[inline]
#[must_use]
pub fn id_to_fieldnorm(id: u8) -> u32 {
    FIELD_NORMS_TABLE[usize::from(id)]
}

/// Encodes a fieldnorm (document length) to its 1-byte id, rounding down to
/// the containing bucket.
///
/// Mirrors tantivy 0.26.1 `src/fieldnorm/code.rs::fieldnorm_to_id`:
/// exact matches return their index; misses land in the preceding bucket.
// The binary-search index over a 256-entry table always fits in u8: an exact
// hit is <= 255, and a miss's insertion point is >= 1 (table starts at 0), so
// `idx - 1` is <= 255 and never underflows.
#[allow(clippy::cast_possible_truncation)]
#[inline]
#[must_use]
pub fn fieldnorm_to_id(fieldnorm: u32) -> u8 {
    FIELD_NORMS_TABLE
        .binary_search(&fieldnorm)
        .unwrap_or_else(|idx| idx - 1) as u8
}

/// BM25 inverse document frequency.
///
/// Mirrors tantivy 0.26.1 `src/query/bm25.rs::idf` including the f32
/// (`tantivy::Score`) intermediate arithmetic and the `doc_count >= doc_freq`
/// assertion.
///
/// # Panics
///
/// Panics if `doc_freq > doc_count`, matching the oracle.
// `x.ln_1p()` would be more accurate (and is what clippy suggests), but the
// oracle computes `(1.0 + x).ln()` in f32; matching its exact op sequence —
// including the small-x cancellation — is the contract.
#[allow(clippy::imprecise_flops)]
#[must_use]
pub fn idf(doc_freq: u64, doc_count: u64) -> f32 {
    assert!(doc_count >= doc_freq, "{doc_count} >= {doc_freq}");
    let x = ((doc_count - doc_freq) as f32 + 0.5) / (doc_freq as f32 + 0.5);
    (1.0 + x).ln()
}

/// One entry of the BM25 tf cache: `k1 * (1 - b + b * |d| / avgdl)`.
///
/// Mirrors tantivy 0.26.1 `src/query/bm25.rs::cached_tf_component`. The
/// operation order is part of the contract — reassociating or fusing
/// (`mul_add`) changes f32 results and breaks bit parity with the oracle.
/// `fieldnorm` is a decoded per-document value; `average_fieldnorm` is the
/// raw `total_num_tokens / total_num_docs` average passed by tantivy despite
/// that historical parameter name.
#[inline]
#[must_use]
pub fn cached_tf_component(fieldnorm: u32, average_fieldnorm: f32) -> f32 {
    BM25_K1 * (1.0 - BM25_B + BM25_B * fieldnorm as f32 / average_fieldnorm)
}

/// Precomputes the 256-entry BM25 tf cache for one (field, snapshot) pair.
///
/// Mirrors tantivy 0.26.1 `src/query/bm25.rs::compute_tf_cache`: entry `i` is
/// [`cached_tf_component`] of the *decoded* fieldnorm for id `i`. Scoring a
/// doc then costs one table lookup per (term, doc) instead of a division. The
/// `average_fieldnorm` argument is the raw average field length, not an
/// average of values decoded from [`FIELD_NORMS_TABLE`].
#[must_use]
pub fn compute_tf_cache(average_fieldnorm: f32) -> [f32; 256] {
    let mut cache = [0.0_f32; 256];
    for (fieldnorm_id, slot) in cache.iter_mut().enumerate() {
        // Enumerating a [_; 256] array: index is always <= 255.
        #[allow(clippy::cast_possible_truncation)]
        let fieldnorm = id_to_fieldnorm(fieldnorm_id as u8);
        *slot = cached_tf_component(fieldnorm, average_fieldnorm);
    }
    cache
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::path::{Component, Path};

    use serde_json::Value;
    use tantivy::Index;
    use tantivy::fieldnorm::FieldNormReader;
    use tantivy::schema::{FieldType, IndexRecordOption, Schema, TEXT};
    use tantivy::tokenizer::{TextAnalyzer, TokenStream};

    use super::*;

    const LANGUAGE_CONTRACT_FIXTURE: &str =
        include_str!("../../../tests/fixtures/quill_language_contract.json");
    const SHARED_CORPUS_FIXTURE: &str = include_str!("../../../tests/fixtures/corpus.json");

    fn contract_array<'a>(root: &'a Value, key: &str) -> &'a [Value] {
        root.get(key)
            .and_then(Value::as_array)
            .expect("language contract section must be an array")
            .as_slice()
    }

    fn assert_repo_relative_source_path(repo_root: &Path, source_path: &str) {
        let path = Path::new(source_path);
        assert!(
            !path.is_absolute()
                && path
                    .components()
                    .all(|component| matches!(component, Component::Normal(_))),
            "contract source path `{source_path}` must stay within the repository"
        );
        assert!(
            repo_root.join(path).is_file(),
            "contract source path `{source_path}` does not exist"
        );
    }

    fn fixture_input(case: &Value) -> String {
        if let Some(input) = case.get("input").and_then(Value::as_str) {
            return input.to_owned();
        }
        let generated = case
            .get("generated_input")
            .and_then(Value::as_object)
            .expect("fixture case must contain input or generated_input");
        let repeat = generated
            .get("repeat")
            .and_then(Value::as_str)
            .expect("generated input must name its repeat text");
        let count = ["count", "count_bytes", "count_chars"]
            .into_iter()
            .find_map(|key| generated.get(key).and_then(Value::as_u64))
            .expect("generated input must contain a repeat count");
        repeat.repeat(usize::try_from(count).expect("fixture repeat count must fit usize"))
    }

    fn token_stream_json(analyzer: &mut TextAnalyzer, input: &str) -> Value {
        let mut tokens = Vec::new();
        let mut stream = analyzer.token_stream(input);
        while stream.advance() {
            let token = stream.token();
            tokens.push(serde_json::json!({
                "text": token.text,
                "position": token.position,
                "offset_from": token.offset_from,
                "offset_to": token.offset_to,
                "position_length": token.position_length,
            }));
        }
        Value::Array(tokens)
    }

    fn cass_analyzer(name: &str) -> TextAnalyzer {
        let mut schema_builder = Schema::builder();
        schema_builder.add_text_field("body", TEXT);
        let mut index = Index::create_in_ram(schema_builder.build());
        crate::cass_compat::cass_ensure_tokenizer(&mut index);
        index
            .tokenizers()
            .get(name)
            .expect("shipping CASS analyzer must be registered")
    }

    #[test]
    fn language_contract_fixture_is_complete_and_self_consistent() {
        let root: Value = serde_json::from_str(LANGUAGE_CONTRACT_FIXTURE)
            .expect("quill language contract fixture must be valid JSON");
        assert_eq!(root["schema_version"], 1);
        assert_eq!(root["contract_version"], "1.0.3");
        assert_eq!(root["oracle"]["engine"], "tantivy");
        assert_eq!(root["oracle"]["version"], "0.27.0");
        assert_eq!(
            root["fixture_execution"]["lexical_loader_executes"],
            serde_json::json!(["schema", "analyzers", "helpers", "bm25_operation_order"])
        );

        let mut all_record_ids = BTreeSet::new();
        let mut fixture_ids = BTreeSet::new();
        for section in [
            "source_corpora",
            "analyzer_cases",
            "helper_cases",
            "parse_tree_cases",
            "scoring_cases",
            "behavior_cases",
            "harvested_queries",
        ] {
            for case in contract_array(&root, section) {
                let id = case["id"]
                    .as_str()
                    .expect("language contract case must contain a string id");
                assert!(
                    all_record_ids.insert(id),
                    "duplicate language contract fixture id `{id}`"
                );
                if !matches!(section, "source_corpora" | "harvested_queries") {
                    fixture_ids.insert(id);
                }
            }
        }

        const REQUIRED_SURFACES: [&str; 11] = [
            "schema",
            "analyzers",
            "queries",
            "collectors",
            "scoring",
            "writer",
            "reader",
            "snippets",
            "segments_durability",
            "errors",
            "concurrency",
        ];
        let mut covered_surfaces = BTreeSet::new();
        for coverage in contract_array(&root, "surface_coverage") {
            let surface = coverage["surface"]
                .as_str()
                .expect("surface coverage row must name its surface");
            assert!(
                covered_surfaces.insert(surface),
                "duplicate surface coverage row `{surface}`"
            );
            let ids = coverage["fixture_ids"]
                .as_array()
                .expect("surface coverage row must contain fixture_ids");
            assert!(
                !ids.is_empty(),
                "surface `{surface}` has no fixture coverage"
            );
            for id in ids {
                let id = id
                    .as_str()
                    .expect("surface fixture reference must be a string");
                assert!(
                    fixture_ids.contains(id),
                    "surface `{surface}` has dangling fixture reference `{id}`"
                );
            }
        }
        assert_eq!(covered_surfaces.len(), REQUIRED_SURFACES.len());
        for required in REQUIRED_SURFACES {
            assert!(
                covered_surfaces.contains(required),
                "missing plan section 3.1 surface `{required}`"
            );
        }

        let analyzer_names: BTreeSet<_> = contract_array(&root, "analyzer_cases")
            .iter()
            .filter_map(|case| case["analyzer"].as_str())
            .collect();
        for required in [
            "frankensearch_default",
            "hyphen_normalize",
            "prefix_normalize",
        ] {
            assert!(
                analyzer_names.contains(required),
                "missing analyzer fixture for `{required}`"
            );
        }

        let query_classes: BTreeSet<_> = contract_array(&root, "harvested_queries")
            .iter()
            .filter_map(|case| case["query_class"].as_str())
            .collect();
        for required in [
            "identifier",
            "short_keyword",
            "natural_language",
            "phrase",
            "boolean",
            "glob",
            "range",
        ] {
            assert!(
                query_classes.contains(required),
                "missing harvested `{required}` query class"
            );
        }
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut harvested_count = 0usize;
        for case in contract_array(&root, "harvested_queries") {
            let source_kind = case["source_kind"]
                .as_str()
                .expect("query-class corpus row must name source_kind");
            assert!(
                matches!(source_kind, "harvested" | "constructed"),
                "unknown query source_kind `{source_kind}`"
            );
            match source_kind {
                "harvested" => {
                    harvested_count += 1;
                    let source_path = case["source"]
                        .as_str()
                        .expect("harvested query must name its source path");
                    assert_repo_relative_source_path(&repo_root, source_path);
                }
                "constructed" => assert!(
                    case["source_fact"].as_str().is_some(),
                    "constructed query `{}` must name its source fact",
                    case["id"].as_str().unwrap_or("<unknown>")
                ),
                _ => {}
            }
        }
        assert_eq!(harvested_count, 3);

        let mut source_paths = BTreeSet::new();
        for source in contract_array(&root, "source_corpora") {
            let source_path = source["path"]
                .as_str()
                .expect("source-corpus row must name a string path");
            assert!(
                source_paths.insert(source_path),
                "duplicate source-corpus path `{source_path}`"
            );
            assert_repo_relative_source_path(&repo_root, source_path);
        }
        for required in ["tests/fixtures/corpus.json", "tests/fixtures/queries.json"] {
            assert!(
                source_paths.contains(required),
                "missing required source corpus `{required}`"
            );
        }

        let corpus_source = contract_array(&root, "source_corpora")
            .iter()
            .find(|source| source["id"] == "source-corpus-120")
            .expect("shared corpus source row must be present");
        let expected_document_count = corpus_source["expected_document_count"]
            .as_u64()
            .and_then(|count| usize::try_from(count).ok())
            .expect("shared corpus source row must contain a valid document count");
        let shared_corpus: Value = serde_json::from_str(SHARED_CORPUS_FIXTURE)
            .expect("shared corpus fixture must be valid JSON");
        assert_eq!(
            shared_corpus["documents"]
                .as_array()
                .expect("shared corpus fixture must contain documents")
                .len(),
            expected_document_count,
            "language-contract corpus count drifted from the shared fixture"
        );

        for case in contract_array(&root, "parse_tree_cases") {
            assert!(
                case["expected_ast"].is_object(),
                "query fixture `{}` lacks a canonical expected_ast",
                case["id"].as_str().unwrap_or("<unknown>")
            );
        }

        let oversized = contract_array(&root, "analyzer_cases")
            .iter()
            .find(|case| case["id"] == "analyzer-tantivy-index-limit-65531-dropped")
            .expect("missing oversized Tantivy term fixture");
        assert_eq!(oversized["generated_input"]["count_bytes"], 65_531);
        assert_eq!(oversized["token_admission"], "dropped");
        assert_eq!(
            oversized["applies_to"],
            serde_json::json!(["document_indexing"])
        );
        let quill_query_limit = contract_array(&root, "analyzer_cases")
            .iter()
            .find(|case| case["id"] == "analyzer-quill-query-limit-65531-dropped")
            .expect("missing symmetric Quill query-side term limit fixture");
        assert_eq!(quill_query_limit["token_admission"], "dropped");
        assert_eq!(
            quill_query_limit["applies_to"],
            serde_json::json!(["quill_query_analysis"])
        );
        assert_eq!(quill_query_limit["ordinary_string_query_reachable"], false);

        let avgdl = contract_array(&root, "scoring_cases")
            .iter()
            .find(|case| case["id"] == "score-avgdl-uses-raw-statistics")
            .expect("missing raw avgdl scoring fixture");
        assert_eq!(avgdl["expected_total_num_docs"], 3);
        assert_eq!(avgdl["expected_total_num_tokens"], 21);
        assert_eq!(avgdl["expected_avgdl"], 7.0);
        assert!(
            avgdl["forbidden_calculation"]
                .as_str()
                .is_some_and(|text| text.contains("decoded"))
        );

        let total_order = contract_array(&root, "scoring_cases")
            .iter()
            .find(|case| case["id"] == "score-deterministic-total-order")
            .expect("missing deterministic total-order fixture");
        assert_eq!(
            total_order["oracle_order"],
            serde_json::json!([
                "score descending via f32 total_cmp",
                "DocAddress ascending by segment_ord then segment-local doc_id"
            ])
        );
        assert_eq!(
            total_order["quill_native_tie_key"],
            "global u32 docid ascending"
        );
    }

    #[test]
    fn language_contract_schema_matches_shipping_schema() {
        let root: Value = serde_json::from_str(LANGUAGE_CONTRACT_FIXTURE)
            .expect("quill language contract fixture must be valid JSON");
        let fixture = contract_array(&root, "behavior_cases")
            .iter()
            .find(|case| case["id"] == "behavior-schema-default-fields")
            .expect("missing shipping schema fixture");
        let expected_fields = fixture["fields"]
            .as_array()
            .expect("schema fixture fields must be an array");
        let (schema, _) = crate::build_schema();
        assert_eq!(schema.num_fields(), expected_fields.len());

        for expected in expected_fields {
            let name = expected["name"]
                .as_str()
                .expect("schema fixture field must have a name");
            let field = schema
                .get_field(name)
                .expect("shipping schema must contain every contracted field");
            let entry = schema.get_field_entry(field);
            let (actual_type, actual_tokenizer) = match entry.field_type() {
                FieldType::Str(options) => (
                    "text",
                    options
                        .get_indexing_options()
                        .map(|indexing| indexing.tokenizer()),
                ),
                FieldType::U64(_) => ("u64", None),
                other => {
                    assert!(
                        matches!(other, FieldType::Str(_) | FieldType::U64(_)),
                        "unexpected shipping schema type for `{name}`: {other:?}"
                    );
                    continue;
                }
            };
            assert_eq!(expected["type"], actual_type);
            assert_eq!(expected["indexed"], entry.is_indexed());
            assert_eq!(expected["stored"], entry.is_stored());
            assert_eq!(expected["fast"], entry.is_fast());
            assert_eq!(expected["tokenizer"].as_str(), actual_tokenizer);

            let expected_record_option = match expected.get("record_option") {
                None | Some(Value::Null) => None,
                Some(Value::String(option)) if option == "Basic" => Some(IndexRecordOption::Basic),
                Some(Value::String(option)) if option == "WithFreqsAndPositions" => {
                    Some(IndexRecordOption::WithFreqsAndPositions)
                }
                option => {
                    assert!(
                        option
                            .as_ref()
                            .and_then(|value| value.as_str())
                            .is_some_and(
                                |value| value == "Basic" || value == "WithFreqsAndPositions"
                            ),
                        "unsupported record option for `{name}`: {option:?}"
                    );
                    continue;
                }
            };
            assert_eq!(
                entry.field_type().index_record_option(),
                expected_record_option,
                "record option diverged for `{name}`"
            );
        }
    }

    #[test]
    fn language_contract_analyzer_goldens_execute_against_shipping_code() {
        let root: Value = serde_json::from_str(LANGUAGE_CONTRACT_FIXTURE)
            .expect("quill language contract fixture must be valid JSON");
        for case in contract_array(&root, "analyzer_cases") {
            if case.get("expected_tokens").is_none() && case.get("expected_token_count").is_none() {
                continue;
            }
            let name = case["analyzer"]
                .as_str()
                .expect("analyzer fixture must name an analyzer");
            assert!(
                matches!(
                    name,
                    "frankensearch_default" | "hyphen_normalize" | "prefix_normalize"
                ),
                "unknown analyzer fixture `{name}`"
            );
            let mut analyzer = match name {
                "frankensearch_default" => crate::build_tokenizer(),
                "hyphen_normalize" | "prefix_normalize" => cass_analyzer(name),
                _ => continue,
            };
            let input = fixture_input(case);
            let actual = token_stream_json(&mut analyzer, &input);
            if let Some(expected) = case.get("expected_tokens") {
                assert_eq!(
                    &actual,
                    expected,
                    "analyzer golden `{}` diverged",
                    case["id"].as_str().unwrap_or("<unknown>")
                );
            }
            if let Some(expected_count) = case.get("expected_token_count").and_then(Value::as_u64) {
                let actual_tokens = actual
                    .as_array()
                    .expect("token-stream result must be an array");
                assert_eq!(
                    actual_tokens.len(),
                    usize::try_from(expected_count).expect("token count must fit usize")
                );
                let expected_bytes = case["expected_token_bytes"]
                    .as_u64()
                    .expect("generated token case must name expected bytes");
                let token_text = actual_tokens[0]["text"]
                    .as_str()
                    .expect("generated token must have text");
                assert_eq!(
                    token_text.len(),
                    usize::try_from(expected_bytes).expect("token byte count must fit usize")
                );
                let repeat = case["expected_token_repeat"]
                    .as_str()
                    .expect("generated token case must name expected repeat text");
                assert_eq!(token_text, repeat.repeat(token_text.len()));
            }
        }
    }

    #[test]
    fn language_contract_helper_goldens_execute_against_shipping_code() {
        let root: Value = serde_json::from_str(LANGUAGE_CONTRACT_FIXTURE)
            .expect("quill language contract fixture must be valid JSON");
        for case in contract_array(&root, "helper_cases") {
            let input = fixture_input(case);
            let helper = case["helper"]
                .as_str()
                .expect("helper fixture must name a helper");
            assert!(
                matches!(
                    helper,
                    "cass_generate_edge_ngrams" | "cass_build_preview" | "truncate_query"
                ),
                "unknown helper fixture `{helper}`"
            );
            match helper {
                "cass_generate_edge_ngrams" => {
                    let actual = crate::cass_compat::cass_generate_edge_ngrams(&input);
                    if let Some(expected) = case.get("expected").and_then(Value::as_str) {
                        assert_eq!(actual, expected);
                    }
                    if let Some(expected_count) =
                        case.get("expected_prefix_count").and_then(Value::as_u64)
                    {
                        let prefixes: Vec<_> = actual.split_whitespace().collect();
                        assert_eq!(
                            prefixes.len(),
                            usize::try_from(expected_count).expect("prefix count must fit usize")
                        );
                        assert_eq!(
                            prefixes.last().copied(),
                            case.get("last_expected_prefix").and_then(Value::as_str)
                        );
                    }
                }
                "cass_build_preview" => {
                    let max_chars = case["max_chars"]
                        .as_u64()
                        .and_then(|count| usize::try_from(count).ok())
                        .expect("preview max_chars must fit usize");
                    assert_eq!(
                        crate::cass_compat::cass_build_preview(&input, max_chars),
                        case["expected"]
                            .as_str()
                            .expect("preview fixture must contain expected output")
                    );
                }
                "truncate_query" => {
                    let actual = crate::TantivyIndex::truncate_query(&input);
                    assert_eq!(
                        actual.chars().count(),
                        case["expected_chars"]
                            .as_u64()
                            .and_then(|count| usize::try_from(count).ok())
                            .expect("truncate expected_chars must fit usize")
                    );
                    if let Some(expected_bytes) = case.get("expected_bytes").and_then(Value::as_u64)
                    {
                        assert_eq!(
                            actual.len(),
                            usize::try_from(expected_bytes)
                                .expect("truncate expected bytes must fit usize")
                        );
                    }
                }
                _ => {}
            }
        }
    }

    #[test]
    fn language_contract_bm25_operation_order_differs_from_reassociation() {
        let root: Value = serde_json::from_str(LANGUAGE_CONTRACT_FIXTURE)
            .expect("quill language contract fixture must be valid JSON");
        let case = contract_array(&root, "scoring_cases")
            .iter()
            .find(|case| case["id"] == "score-bm25-operation-order-bits")
            .expect("missing BM25 operation-order fixture");
        assert_eq!(
            case["input"],
            serde_json::json!({
                "doc_freq": 1,
                "doc_count": 1,
                "term_freq": 1,
                "fieldnorm_id": 0,
                "avgdl": 1.0,
            })
        );

        let norm = cached_tf_component(id_to_fieldnorm(0), 1.0);
        let weight = idf(1, 1) * (1.0 + BM25_K1);
        let oracle_score = weight * (1.0 / (1.0 + norm));
        let reassociated = idf(1, 1) * (1.0 * (1.0 + BM25_K1) / (1.0 + norm));
        assert_eq!(case["universal_libm_bit_pattern"], false);
        assert_eq!(
            oracle_score.to_bits() != reassociated.to_bits(),
            case["expect_exact_and_reassociated_bits_to_differ"]
                .as_bool()
                .expect("BM25 fixture must state whether operation orders differ")
        );
    }

    #[test]
    fn language_contract_epsilon_components_are_deterministic() {
        let root: Value = serde_json::from_str(LANGUAGE_CONTRACT_FIXTURE)
            .expect("quill language contract fixture must be valid JSON");
        let case = contract_array(&root, "scoring_cases")
            .iter()
            .find(|case| case["id"] == "score-epsilon-component-boundary")
            .expect("missing epsilon-component boundary fixture");
        let ordered = case["oracle_order"]
            .as_array()
            .expect("epsilon fixture oracle_order must be an array");
        let mut components: Vec<Vec<String>> = Vec::new();
        for (index, item) in ordered.iter().enumerate() {
            let id = item["id"]
                .as_str()
                .expect("epsilon fixture result must have an id")
                .to_owned();
            if index == 0 {
                components.push(vec![id]);
                continue;
            }
            let previous = ordered[index - 1]["score"]
                .as_f64()
                .expect("epsilon fixture score must be numeric");
            let score = item["score"]
                .as_f64()
                .expect("epsilon fixture score must be numeric");
            let relative_delta =
                (previous - score).abs() / previous.abs().max(score.abs()).max(1e-12);
            if relative_delta <= 1e-4 {
                components.last_mut().expect("component exists").push(id);
            } else {
                components.push(vec![id]);
            }
        }
        assert_eq!(
            serde_json::to_value(components).expect("components serialize"),
            case["expected_components"]
        );
    }

    #[test]
    fn decode_matches_tantivy_for_all_256_ids() {
        for id in 0..=u8::MAX {
            assert_eq!(
                id_to_fieldnorm(id),
                FieldNormReader::id_to_fieldnorm(id),
                "decode diverges from oracle at id {id}"
            );
        }
    }

    /// Landmarks + checksum keep the pin meaningful after the tantivy
    /// dev-dependency is eventually dropped (a botched re-vendor cannot
    /// satisfy both the oracle diff and these constants).
    #[test]
    fn table_landmarks_pinned_independent_of_oracle() {
        assert_eq!(FIELD_NORMS_TABLE[0], 0);
        assert_eq!(FIELD_NORMS_TABLE[39], 39);
        assert_eq!(FIELD_NORMS_TABLE[40], 40);
        assert_eq!(FIELD_NORMS_TABLE[41], 42);
        assert_eq!(FIELD_NORMS_TABLE[64], 152);
        assert_eq!(FIELD_NORMS_TABLE[128], 32_792);
        assert_eq!(FIELD_NORMS_TABLE[200], 16_777_240);
        assert_eq!(FIELD_NORMS_TABLE[255], 2_013_265_944);
        let sum: u64 = FIELD_NORMS_TABLE.iter().map(|&v| u64::from(v)).sum();
        assert_eq!(sum, 24_696_067_732, "table checksum drifted");
    }

    #[test]
    fn table_is_strictly_increasing() {
        for window in FIELD_NORMS_TABLE.windows(2) {
            assert!(
                window[0] < window[1],
                "table not strictly increasing at {window:?}"
            );
        }
    }

    #[test]
    fn encode_decode_roundtrip_all_ids() {
        for id in 0..=u8::MAX {
            assert_eq!(fieldnorm_to_id(id_to_fieldnorm(id)), id);
        }
    }

    #[test]
    fn encode_matches_tantivy_for_lengths_0_to_10k() {
        for len in 0..10_000_u32 {
            let id = fieldnorm_to_id(len);
            assert_eq!(
                id,
                FieldNormReader::fieldnorm_to_id(len),
                "encode diverges from oracle at length {len}"
            );
            let decoded = id_to_fieldnorm(id);
            assert!(decoded <= len, "bucket floor exceeds length {len}");
            if id < u8::MAX {
                assert!(
                    id_to_fieldnorm(id + 1) > len,
                    "length {len} not in tightest bucket"
                );
            }
        }
    }

    #[test]
    fn encode_saturates_at_max_bucket() {
        assert_eq!(fieldnorm_to_id(u32::MAX), u8::MAX);
        assert_eq!(fieldnorm_to_id(2_013_265_944), u8::MAX);
        assert_eq!(fieldnorm_to_id(2_013_265_943), 254);
    }

    #[test]
    fn tf_cache_is_bitwise_identical_to_oracle_formula() {
        for &avg in &[1.0_f32, 10.5, 487.3, 65_530.0] {
            let cache = compute_tf_cache(avg);
            for id in 0..=u8::MAX {
                let fieldnorm = FieldNormReader::id_to_fieldnorm(id) as f32;
                let expected = 1.2_f32 * (1.0 - 0.75 + 0.75 * fieldnorm / avg);
                assert_eq!(
                    cache[usize::from(id)].to_bits(),
                    expected.to_bits(),
                    "tf cache diverges at id {id}, avgdl {avg}"
                );
            }
        }
    }

    #[test]
    fn idf_matches_f64_reference_within_f32_tolerance() {
        for &(doc_freq, doc_count) in &[
            (1_u64, 1_u64),
            (1, 2),
            (1, 1_000),
            (37, 1_000),
            (500, 1_000),
            (999, 1_000),
        ] {
            let reference = {
                let x = ((doc_count - doc_freq) as f64 + 0.5) / (doc_freq as f64 + 0.5);
                x.ln_1p()
            };
            let got = f64::from(idf(doc_freq, doc_count));
            // The oracle's f32 `(1.0 + x).ln()` cancels low bits of x for
            // small x, bounding ABSOLUTE error near f32 epsilon (~1.2e-7)
            // regardless of how small the result is; the relative term covers
            // the large-idf regime. Both get ~10x headroom.
            assert!(
                (got - reference).abs() <= 1e-6 + reference.abs() * 1e-5,
                "idf({doc_freq}, {doc_count}) = {got}, f64 reference {reference}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "2 >= 3")]
    fn idf_rejects_doc_freq_above_doc_count() {
        let _ = idf(3, 2);
    }
}
