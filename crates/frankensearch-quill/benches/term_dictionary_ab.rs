#![recursion_limit = "256"]
//! Same-binary TERMDICT-versus-sorted-`Vec<String>` honesty benchmark.
//!
//! Deterministic field-major rows, section spans, exact probes, and prefix
//! probes are built before timing. Exact results and every ordered scan are
//! asserted equal before Criterion starts. Timed arms reuse exact-lookup
//! scratch, consume every scan result, and compare the opened durable
//! dictionary with a binary-searched sorted vector rather than a hash table.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 RCH_WORKER=<pinned-worker> \
//!   rch exec -- env CARGO_TARGET_DIR=/tmp/frankensearch-quill-termdict \
//!   cargo bench -p frankensearch-quill --features bench-internals \
//!   --bench term_dictionary_ab
//! ```
//!
//! The million-term glob distribution is deliberately opt-in because fixture
//! construction is substantially heavier than the default A/B lanes:
//!
//! ```bash
//! FRANKENSEARCH_QUILL_GLOB_SCALE_1M=1 \
//!   cargo bench -p frankensearch-quill --features bench-internals \
//!   --bench term_dictionary_ab
//! ```
//!
//! Its emitted percentiles and raw samples are provisional QG-6 evidence only;
//! the gate is inactive and this benchmark asserts no latency threshold.

use std::cmp::Ordering;
use std::hint::black_box;
use std::time::Instant;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use frankensearch_quill::DEFAULT_SCHEMA;
use frankensearch_quill::grimoire::{
    ByteSpan, EncodedTermDictionary, OwnedTerm, TermCursor, TermDictionary, TermDictionaryError,
    TermInput, TermMatch, TermMetadata, TermScratch, TermSectionLengths,
};

const FIELD_ORDS: [u16; 3] = [0, 1, 2];
const MODULES_PER_FIELD: usize = 96;
const SYMBOLS_PER_MODULE: usize = 64;
const EXACT_QUERY_COUNT: usize = 4_096;
const PREFIXES_PER_FIELD: usize = 16;
const DIGEST_SEED: u64 = 0xcbf2_9ce4_8422_2325;
const DIGEST_PRIME: u64 = 0x0000_0100_0000_01b3;
const GLOB_SCALE_ENV: &str = "FRANKENSEARCH_QUILL_GLOB_SCALE_1M";
const GLOB_SCALE_TERM_COUNT: usize = 1_000_000;
const GLOB_SCALE_FIELD_ORD: u16 = 1;
const GLOB_SCALE_BUCKET_COUNT: usize = 1_024;
const GLOB_SCALE_MATCH_LIMIT: usize = 64;
const GLOB_SCALE_WARMUP_COUNT: usize = 5;
const GLOB_SCALE_SAMPLE_COUNT: usize = 100;

#[derive(Debug)]
struct BaselineTerm {
    term_ord: u32,
    field_ord: u16,
    term: String,
    metadata: TermMetadata,
}

#[derive(Debug)]
struct Query {
    field_ord: u16,
    term: String,
}

#[derive(Debug)]
struct Fixture {
    encoded: EncodedTermDictionary,
    sections: TermSectionLengths,
    baseline: Vec<BaselineTerm>,
    exact_queries: Vec<Query>,
    prefix_queries: Vec<Query>,
}

#[derive(Clone, Copy, Debug)]
struct GlobProbe {
    label: &'static str,
    pattern: &'static [u8],
}

const GLOB_SCALE_PROBES: [GlobProbe; 3] = [
    GlobProbe {
        label: "suffix",
        pattern: b"*::suffix-probe",
    },
    GlobProbe {
        label: "substring",
        pattern: b"*substring-probe*",
    },
    GlobProbe {
        label: "complex",
        pattern: b"term-*::bucket-*::complex-alpha*complex-omega*done",
    },
];

#[derive(Clone, Copy, Debug)]
struct PreparedGlobProbe {
    probe: GlobProbe,
    expected_count: usize,
}

#[derive(Debug)]
struct GlobScaleFixture {
    encoded: EncodedTermDictionary,
    sections: TermSectionLengths,
    terms: Vec<String>,
}

#[derive(Debug, Eq, PartialEq)]
struct ObservedTerm {
    term_ord: u32,
    field_ord: u16,
    term: Vec<u8>,
    metadata: TermMetadata,
}

fn fail(stage: &str, error: &TermDictionaryError) -> ! {
    eprintln!("term-dictionary-ab {stage} failed: {error}");
    std::process::exit(2);
}

fn fail_glob_scale(stage: &str, detail: std::fmt::Arguments<'_>) -> ! {
    eprintln!("term-dictionary-ab glob-scale {stage} failed: {detail}");
    std::process::exit(2);
}

fn require<T>(stage: &str, result: Result<T, TermDictionaryError>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => fail(stage, &error),
    }
}

fn as_u32(stage: &str, value: usize) -> u32 {
    match u32::try_from(value) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("term-dictionary-ab {stage} cannot represent {value} as u32: {error}");
            std::process::exit(2);
        }
    }
}

fn as_u64(stage: &str, value: usize) -> u64 {
    match u64::try_from(value) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("term-dictionary-ab {stage} cannot represent {value} as u64: {error}");
            std::process::exit(2);
        }
    }
}

fn glob_scale_enabled() -> bool {
    std::env::var(GLOB_SCALE_ENV).is_ok_and(|value| value == "1")
}

fn checked_advance(stage: &str, offset: &mut u64, len: u64) -> ByteSpan {
    let span = ByteSpan::new(*offset, len);
    *offset = offset.checked_add(len).unwrap_or_else(|| {
        eprintln!("term-dictionary-ab {stage} offset overflow");
        std::process::exit(2);
    });
    span
}

fn build_fixture() -> Fixture {
    let capacity = FIELD_ORDS
        .len()
        .saturating_mul(MODULES_PER_FIELD)
        .saturating_mul(SYMBOLS_PER_MODULE);
    let mut baseline = Vec::with_capacity(capacity);
    let mut postings_offset = 0_u64;
    let mut positions_offset = 0_u64;
    let mut blockmax_offset = 0_u64;

    for field_ord in FIELD_ORDS {
        for module in 0..MODULES_PER_FIELD {
            for symbol in 0..SYMBOLS_PER_MODULE {
                let ordinal = baseline.len();
                let ordinal_u64 = as_u64("term ordinal", ordinal);
                let postings =
                    checked_advance("POSTINGS", &mut postings_offset, 5 + ordinal_u64 % 11);
                let blockmax =
                    checked_advance("BLOCKMAX", &mut blockmax_offset, 3 + ordinal_u64 % 5);
                let positions = if field_ord == 0 {
                    None
                } else {
                    Some(checked_advance(
                        "POSITIONS",
                        &mut positions_offset,
                        4 + ordinal_u64 % 7,
                    ))
                };
                let doc_freq = as_u32("doc frequency", ordinal % 4_096 + 1);
                let metadata = positions.map_or_else(
                    || TermMetadata::without_positions(doc_freq, postings, blockmax),
                    |positions| {
                        TermMetadata::with_positions(doc_freq, postings, positions, blockmax)
                    },
                );
                baseline.push(BaselineTerm {
                    term_ord: as_u32("term ordinal", ordinal),
                    field_ord,
                    term: format!(
                        "crate::{field_ord:02}::module::{module:04}::symbol::{symbol:04}"
                    ),
                    metadata,
                });
            }
        }
    }

    let sections = TermSectionLengths {
        postings: postings_offset,
        positions: Some(positions_offset),
        blockmax: blockmax_offset,
    };
    let inputs: Vec<_> = baseline
        .iter()
        .map(|row| TermInput::new(row.field_ord, row.term.as_bytes(), row.metadata))
        .collect();
    let encoded = require(
        "encode",
        EncodedTermDictionary::encode_sorted(DEFAULT_SCHEMA, sections, &inputs),
    );

    let mut exact_queries = Vec::with_capacity(EXACT_QUERY_COUNT);
    for index in 0..EXACT_QUERY_COUNT / 2 {
        let field_ord = FIELD_ORDS[index % FIELD_ORDS.len()];
        let module = index.wrapping_mul(37) % MODULES_PER_FIELD;
        let symbol = index.wrapping_mul(17) % SYMBOLS_PER_MODULE;
        exact_queries.push(Query {
            field_ord,
            term: format!("crate::{field_ord:02}::module::{module:04}::symbol::{symbol:04}"),
        });
        exact_queries.push(Query {
            field_ord,
            term: format!("crate::{field_ord:02}::module::{module:04}::symbol::9999"),
        });
    }

    let mut prefix_queries = Vec::with_capacity(FIELD_ORDS.len() * PREFIXES_PER_FIELD);
    for field_ord in FIELD_ORDS {
        for prefix_index in 0..PREFIXES_PER_FIELD {
            let module = prefix_index.wrapping_mul(5) % MODULES_PER_FIELD;
            prefix_queries.push(Query {
                field_ord,
                term: format!("crate::{field_ord:02}::module::{module:04}::"),
            });
        }
    }

    Fixture {
        encoded,
        sections,
        baseline,
        exact_queries,
        prefix_queries,
    }
}

fn glob_scale_metadata(index: usize) -> TermMetadata {
    let offset = as_u64("glob scale metadata offset", index);
    TermMetadata::with_positions(
        1,
        ByteSpan::new(offset, 1),
        ByteSpan::new(offset, 1),
        ByteSpan::new(offset, 1),
    )
}

fn glob_scale_term(index: usize) -> String {
    let bucket = index % GLOB_SCALE_BUCKET_COUNT;
    let tail = match index % 100_000 {
        17 => "suffix-probe",
        29 => "left::substring-probe::right",
        41 => "complex-alpha::bridge::complex-omega::done",
        _ => "ordinary",
    };
    format!("term-{index:06}::bucket-{bucket:04}::{tail}")
}

fn build_glob_scale_fixture() -> GlobScaleFixture {
    let mut terms = Vec::with_capacity(GLOB_SCALE_TERM_COUNT);
    for index in 0..GLOB_SCALE_TERM_COUNT {
        terms.push(glob_scale_term(index));
    }

    let sections = TermSectionLengths {
        postings: as_u64("glob scale POSTINGS length", GLOB_SCALE_TERM_COUNT),
        // CASS suffix/substring/complex globs target positional text fields,
        // so the scale fixture includes the same per-term POSITIONS metadata.
        positions: Some(as_u64("glob scale POSITIONS length", GLOB_SCALE_TERM_COUNT)),
        blockmax: as_u64("glob scale BLOCKMAX length", GLOB_SCALE_TERM_COUNT),
    };
    let inputs = terms
        .iter()
        .enumerate()
        .map(|(index, term)| {
            TermInput::new(
                GLOB_SCALE_FIELD_ORD,
                term.as_bytes(),
                glob_scale_metadata(index),
            )
        })
        .collect::<Vec<_>>();
    let encoded = require(
        "glob scale encode",
        EncodedTermDictionary::encode_sorted(DEFAULT_SCHEMA, sections, &inputs),
    );
    GlobScaleFixture {
        encoded,
        sections,
        terms,
    }
}

fn compare_row(row: &BaselineTerm, field_ord: u16, term: &[u8]) -> Ordering {
    row.field_ord
        .cmp(&field_ord)
        .then_with(|| row.term.as_bytes().cmp(term))
}

fn baseline_lookup(baseline: &[BaselineTerm], field_ord: u16, term: &[u8]) -> Option<TermMatch> {
    baseline
        .binary_search_by(|row| compare_row(row, field_ord, term))
        .ok()
        .map(|index| TermMatch {
            term_ord: baseline[index].term_ord,
            metadata: baseline[index].metadata,
        })
}

fn dictionary_rows(mut cursor: TermCursor<'_, '_>) -> Vec<ObservedTerm> {
    let mut rows = Vec::new();
    while let Some(current) = cursor.current() {
        rows.push(ObservedTerm {
            term_ord: current.term_ord,
            field_ord: current.field_ord,
            term: current.term.to_vec(),
            metadata: current.metadata,
        });
        require("cursor advance during parity", cursor.next());
    }
    rows
}

fn baseline_rows(baseline: &[BaselineTerm]) -> Vec<ObservedTerm> {
    baseline
        .iter()
        .map(|row| ObservedTerm {
            term_ord: row.term_ord,
            field_ord: row.field_ord,
            term: row.term.as_bytes().to_vec(),
            metadata: row.metadata,
        })
        .collect()
}

fn baseline_prefix_rows(baseline: &[BaselineTerm], query: &Query) -> Vec<ObservedTerm> {
    let prefix = query.term.as_bytes();
    let start =
        baseline.partition_point(|row| compare_row(row, query.field_ord, prefix) == Ordering::Less);
    baseline[start..]
        .iter()
        .take_while(|row| {
            row.field_ord == query.field_ord && row.term.as_bytes().starts_with(prefix)
        })
        .map(|row| ObservedTerm {
            term_ord: row.term_ord,
            field_ord: row.field_ord,
            term: row.term.as_bytes().to_vec(),
            metadata: row.metadata,
        })
        .collect()
}

fn brute_force_star_matches(pattern: &[u8], term: &[u8]) -> bool {
    let Some((&head, tail)) = pattern.split_first() else {
        return term.is_empty();
    };
    if head == b'*' {
        let mut remaining_pattern = tail;
        while remaining_pattern.first() == Some(&b'*') {
            remaining_pattern = &remaining_pattern[1..];
        }
        return (0..=term.len()).any(|matched_bytes| {
            brute_force_star_matches(remaining_pattern, &term[matched_bytes..])
        });
    }
    let Some((&term_head, term_tail)) = term.split_first() else {
        return false;
    };
    head == term_head && brute_force_star_matches(tail, term_tail)
}

fn prepare_glob_scale_probes(
    dictionary: &TermDictionary<'_>,
    terms: &[String],
) -> Vec<PreparedGlobProbe> {
    let mut prepared = Vec::with_capacity(GLOB_SCALE_PROBES.len());
    for probe in GLOB_SCALE_PROBES {
        let expected = terms
            .iter()
            .enumerate()
            .filter(|(_, term)| brute_force_star_matches(probe.pattern, term.as_bytes()))
            .collect::<Vec<_>>();
        if expected.is_empty() {
            fail_glob_scale(
                "parity",
                format_args!("probe {} matched no baseline terms", probe.label),
            );
        }
        if expected.len() > GLOB_SCALE_MATCH_LIMIT {
            fail_glob_scale(
                "parity",
                format_args!(
                    "probe {} matched {} baseline terms, exceeding benchmark limit {}",
                    probe.label,
                    expected.len(),
                    GLOB_SCALE_MATCH_LIMIT
                ),
            );
        }

        let actual = require(
            "glob scale parity expansion",
            dictionary.expand_glob(GLOB_SCALE_FIELD_ORD, probe.pattern, GLOB_SCALE_MATCH_LIMIT),
        );
        if actual.len() != expected.len() {
            fail_glob_scale(
                "parity",
                format_args!(
                    "probe {} returned {} terms; brute force returned {}",
                    probe.label,
                    actual.len(),
                    expected.len()
                ),
            );
        }
        for (actual_row, (expected_index, expected_term)) in actual.iter().zip(&expected) {
            let expected_term_ord = as_u32("glob scale expected term ordinal", *expected_index);
            let expected_metadata = glob_scale_metadata(*expected_index);
            if actual_row.term_ord != expected_term_ord
                || actual_row.field_ord != GLOB_SCALE_FIELD_ORD
                || actual_row.term.as_slice() != expected_term.as_bytes()
                || actual_row.metadata != expected_metadata
            {
                fail_glob_scale(
                    "parity",
                    format_args!(
                        "probe {} diverged at expected term ordinal {}",
                        probe.label, expected_term_ord
                    ),
                );
            }
        }
        prepared.push(PreparedGlobProbe {
            probe,
            expected_count: expected.len(),
        });
    }
    prepared
}

fn assert_parity(
    dictionary: &TermDictionary<'_>,
    baseline: &[BaselineTerm],
    exact_queries: &[Query],
    prefix_queries: &[Query],
) {
    let mut scratch = TermScratch::new();
    for query in exact_queries {
        let encoded = require(
            "exact lookup parity",
            dictionary.lookup_with_scratch(query.field_ord, query.term.as_bytes(), &mut scratch),
        );
        let sorted = baseline_lookup(baseline, query.field_ord, query.term.as_bytes());
        assert_eq!(encoded, sorted, "exact lookup parity for {query:?}");
    }

    let encoded_full = dictionary_rows(require("full cursor parity", dictionary.cursor()));
    let sorted_full = baseline_rows(baseline);
    assert_eq!(encoded_full, sorted_full, "full ordered scan parity");

    for query in prefix_queries {
        let encoded = dictionary_rows(require(
            "prefix cursor parity",
            dictionary.prefix_cursor(query.field_ord, query.term.as_bytes()),
        ));
        let sorted = baseline_prefix_rows(baseline, query);
        assert_eq!(encoded, sorted, "prefix scan parity for {query:?}");
    }
}

fn mix(digest: u64, value: u64) -> u64 {
    digest.wrapping_mul(DIGEST_PRIME) ^ value
}

fn digest_metadata(mut digest: u64, metadata: TermMetadata) -> u64 {
    digest = mix(digest, u64::from(metadata.doc_freq));
    digest = mix(digest, metadata.postings.offset);
    digest = mix(digest, metadata.postings.len);
    if let Some(positions) = metadata.positions {
        digest = mix(digest, positions.offset);
        digest = mix(digest, positions.len);
    }
    digest = mix(digest, metadata.blockmax.offset);
    mix(digest, metadata.blockmax.len)
}

fn digest_term(
    mut digest: u64,
    term_ord: u32,
    field_ord: u16,
    term: &[u8],
    metadata: TermMetadata,
) -> u64 {
    digest = mix(digest, u64::from(term_ord));
    digest = mix(digest, u64::from(field_ord));
    for byte in term {
        digest = mix(digest, u64::from(*byte));
    }
    digest_metadata(digest, metadata)
}

fn digest_match(digest: u64, found: Option<TermMatch>) -> u64 {
    found.map_or_else(
        || mix(digest, u64::MAX),
        |found| digest_metadata(mix(digest, u64::from(found.term_ord)), found.metadata),
    )
}

fn owned_terms_digest(rows: &[OwnedTerm]) -> u64 {
    rows.iter().fold(DIGEST_SEED, |digest, row| {
        digest_term(digest, row.term_ord, row.field_ord, &row.term, row.metadata)
    })
}

fn dictionary_scan_digest(mut cursor: TermCursor<'_, '_>) -> (usize, u64) {
    let mut count = 0_usize;
    let mut digest = DIGEST_SEED;
    while let Some(current) = cursor.current() {
        digest = digest_term(
            digest,
            current.term_ord,
            current.field_ord,
            current.term,
            current.metadata,
        );
        count = count.saturating_add(1);
        require("timed cursor advance", cursor.next());
    }
    (count, digest)
}

fn baseline_scan_digest<'a>(rows: impl Iterator<Item = &'a BaselineTerm>) -> (usize, u64) {
    let mut count = 0_usize;
    let mut digest = DIGEST_SEED;
    for row in rows {
        digest = digest_term(
            digest,
            row.term_ord,
            row.field_ord,
            row.term.as_bytes(),
            row.metadata,
        );
        count = count.saturating_add(1);
    }
    (count, digest)
}

fn baseline_prefix_digest(baseline: &[BaselineTerm], query: &Query) -> (usize, u64) {
    let prefix = query.term.as_bytes();
    let start =
        baseline.partition_point(|row| compare_row(row, query.field_ord, prefix) == Ordering::Less);
    baseline_scan_digest(baseline[start..].iter().take_while(|row| {
        row.field_ord == query.field_ord && row.term.as_bytes().starts_with(prefix)
    }))
}

fn latency_percentile(samples: &[u128], percentile: usize) -> Option<u128> {
    if samples.is_empty() || !(1..=100).contains(&percentile) {
        return None;
    }
    let rank = samples.len().saturating_mul(percentile).saturating_add(99) / 100;
    samples.get(rank.saturating_sub(1)).copied()
}

fn record_glob_scale_distribution(dictionary: &TermDictionary<'_>, probes: &[PreparedGlobProbe]) {
    for prepared in probes {
        for _ in 0..GLOB_SCALE_WARMUP_COUNT {
            let rows = require(
                "glob scale warmup",
                dictionary.expand_glob(
                    GLOB_SCALE_FIELD_ORD,
                    prepared.probe.pattern,
                    GLOB_SCALE_MATCH_LIMIT,
                ),
            );
            black_box((rows.len(), owned_terms_digest(&rows)));
        }

        let mut samples_ns = Vec::with_capacity(GLOB_SCALE_SAMPLE_COUNT);
        for _ in 0..GLOB_SCALE_SAMPLE_COUNT {
            let started = Instant::now();
            let rows = require(
                "timed glob scale expansion",
                dictionary.expand_glob(
                    GLOB_SCALE_FIELD_ORD,
                    prepared.probe.pattern,
                    GLOB_SCALE_MATCH_LIMIT,
                ),
            );
            let elapsed_ns = started.elapsed().as_nanos();
            if rows.len() != prepared.expected_count {
                fail_glob_scale(
                    "timed result",
                    format_args!(
                        "probe {} returned {} terms; expected {}",
                        prepared.probe.label,
                        rows.len(),
                        prepared.expected_count
                    ),
                );
            }
            black_box(owned_terms_digest(&rows));
            samples_ns.push(elapsed_ns);
        }
        samples_ns.sort_unstable();

        let Some(min_ns) = samples_ns.first().copied() else {
            fail_glob_scale(
                "distribution",
                format_args!("probe {} recorded no samples", prepared.probe.label),
            );
        };
        let Some(max_ns) = samples_ns.last().copied() else {
            fail_glob_scale(
                "distribution",
                format_args!("probe {} recorded no maximum", prepared.probe.label),
            );
        };
        let Some(p50_ns) = latency_percentile(&samples_ns, 50) else {
            fail_glob_scale(
                "distribution",
                format_args!("probe {} has no p50", prepared.probe.label),
            );
        };
        let Some(p95_ns) = latency_percentile(&samples_ns, 95) else {
            fail_glob_scale(
                "distribution",
                format_args!("probe {} has no p95", prepared.probe.label),
            );
        };
        let Some(p99_ns) = latency_percentile(&samples_ns, 99) else {
            fail_glob_scale(
                "distribution",
                format_args!("probe {} has no p99", prepared.probe.label),
            );
        };
        let total_ns = samples_ns
            .iter()
            .copied()
            .fold(0_u128, u128::saturating_add);
        let mean_ns = total_ns / u128::from(as_u64("glob scale sample count", samples_ns.len()));
        eprintln!(
            "grimoire_glob_scale_distribution gate=QG-6 status=inactive evidence=provisional terms={} field={} probe={} pattern={:?} matches={} samples={} min_ns={} mean_ns={} p50_ns={} p95_ns={} p99_ns={} max_ns={} samples_ns={:?}",
            GLOB_SCALE_TERM_COUNT,
            GLOB_SCALE_FIELD_ORD,
            prepared.probe.label,
            prepared.probe.pattern,
            prepared.expected_count,
            samples_ns.len(),
            min_ns,
            mean_ns,
            p50_ns,
            p95_ns,
            p99_ns,
            max_ns,
            samples_ns
        );
    }
}

fn run_glob_scale_distribution() {
    let GlobScaleFixture {
        encoded,
        sections,
        terms,
    } = build_glob_scale_fixture();
    let dictionary = require(
        "glob scale open",
        encoded.dictionary(DEFAULT_SCHEMA, sections),
    );
    let probes = prepare_glob_scale_probes(&dictionary, &terms);
    drop(terms);
    record_glob_scale_distribution(&dictionary, &probes);
}

// Criterion groups intentionally own their reporting state across all registrations.
#[allow(clippy::significant_drop_tightening)]
fn bench(c: &mut Criterion) {
    let fixture = build_fixture();
    let dictionary = require(
        "open",
        fixture.encoded.dictionary(DEFAULT_SCHEMA, fixture.sections),
    );
    assert_parity(
        &dictionary,
        &fixture.baseline,
        &fixture.exact_queries,
        &fixture.prefix_queries,
    );

    let mut exact = c.benchmark_group("grimoire_exact_lookup");
    exact.throughput(Throughput::Elements(as_u64(
        "exact query throughput",
        fixture.exact_queries.len(),
    )));
    let mut scratch = TermScratch::new();
    exact.bench_with_input(
        BenchmarkId::new("termdict", fixture.baseline.len()),
        &fixture.exact_queries,
        |bencher, queries| {
            bencher.iter(|| {
                let mut digest = DIGEST_SEED;
                for query in black_box(queries.as_slice()) {
                    let found = require(
                        "timed exact lookup",
                        dictionary.lookup_with_scratch(
                            query.field_ord,
                            black_box(query.term.as_bytes()),
                            &mut scratch,
                        ),
                    );
                    digest = digest_match(digest, found);
                }
                black_box(digest)
            });
        },
    );
    exact.bench_with_input(
        BenchmarkId::new("sorted_vec", fixture.baseline.len()),
        &fixture.exact_queries,
        |bencher, queries| {
            bencher.iter(|| {
                let mut digest = DIGEST_SEED;
                for query in black_box(queries.as_slice()) {
                    let found = baseline_lookup(
                        &fixture.baseline,
                        query.field_ord,
                        black_box(query.term.as_bytes()),
                    );
                    digest = digest_match(digest, found);
                }
                black_box(digest)
            });
        },
    );
    exact.finish();

    let prefix_elements = fixture
        .prefix_queries
        .iter()
        .map(|query| baseline_prefix_digest(&fixture.baseline, query).0)
        .sum::<usize>();
    let mut prefix = c.benchmark_group("grimoire_prefix_scan");
    prefix.throughput(Throughput::Elements(as_u64(
        "prefix scan throughput",
        prefix_elements,
    )));
    prefix.bench_with_input(
        BenchmarkId::new("termdict", fixture.baseline.len()),
        &fixture.prefix_queries,
        |bencher, queries| {
            bencher.iter(|| {
                let mut digest = DIGEST_SEED;
                for query in black_box(queries.as_slice()) {
                    let cursor = require(
                        "timed prefix cursor",
                        dictionary.prefix_cursor(query.field_ord, black_box(query.term.as_bytes())),
                    );
                    let (count, scan_digest) = dictionary_scan_digest(cursor);
                    digest = mix(digest, as_u64("prefix result count", count));
                    digest = mix(digest, scan_digest);
                }
                black_box(digest)
            });
        },
    );
    prefix.bench_with_input(
        BenchmarkId::new("sorted_vec", fixture.baseline.len()),
        &fixture.prefix_queries,
        |bencher, queries| {
            bencher.iter(|| {
                let mut digest = DIGEST_SEED;
                for query in black_box(queries.as_slice()) {
                    let (count, scan_digest) =
                        baseline_prefix_digest(&fixture.baseline, black_box(query));
                    digest = mix(digest, as_u64("prefix result count", count));
                    digest = mix(digest, scan_digest);
                }
                black_box(digest)
            });
        },
    );
    prefix.finish();

    let mut full = c.benchmark_group("grimoire_full_scan");
    full.throughput(Throughput::Elements(as_u64(
        "full scan throughput",
        fixture.baseline.len(),
    )));
    full.bench_function(
        BenchmarkId::new("termdict", fixture.baseline.len()),
        |bencher| {
            bencher.iter(|| {
                let cursor = require("timed full cursor", dictionary.cursor());
                black_box(dictionary_scan_digest(cursor))
            });
        },
    );
    full.bench_function(
        BenchmarkId::new("sorted_vec", fixture.baseline.len()),
        |bencher| {
            bencher.iter(|| {
                black_box(baseline_scan_digest(
                    black_box(fixture.baseline.as_slice()).iter(),
                ))
            });
        },
    );
    full.finish();

    if glob_scale_enabled() {
        run_glob_scale_distribution();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
