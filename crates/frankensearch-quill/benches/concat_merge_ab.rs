#![recursion_limit = "256"]
//! E3.5 whole-segment concat-merge scaling benchmark.
//!
//! Every case contains the same 1,024 logical documents, 16 tombstones, and
//! caller-supplied logical payload bytes. The documents are divided across 2,
//! 4, 8, or 16 deterministic in-memory leaf FSLX segments. Each source starts
//! at its own `DOC_ORDS_PER_LEASE`-aligned base and uses lease-local ordinals,
//! so concat merge must materialize the production-shaped burned tails between
//! sources. Numeric values count as their canonical eight-byte inputs; a value
//! that Quill also stores counts once rather than once per physical section.
//! Criterion throughput is the checked sum of exact source FSLX bytes plus the
//! exact merged-output FSLX bytes, so physical gap materialization is included.
//! Leaf construction, publication, verification, tombstone/live-identity
//! checks, output sizing, and reopen checks all happen before timed iterations.
//!
//! This benchmark deliberately does not claim query, numeric-column, or stored
//! field semantic parity. Those behavioral proofs belong to Quill's unit and
//! integration tests and the E3.5 `frankensearch-quill-gauntlet` contract.
//!
//! The growing-span sweep below preserves the original production-shaped
//! burned-tail diagnostic. A second fixed-span control distributes every
//! fan-in across the same document-id span, isolating source-count growth from
//! output-span growth. Before fixed-span results are observed, the declared
//! CPU-efficiency acceptance rule is that its spread across the four cases must
//! be at most 1.35x:
//! `max(median ns / exact physical I/O byte) / min(...) <= 1.35`, where exact
//! physical I/O bytes are `sum(source FSLX bytes) + merged output FSLX bytes`.
//! Criterion's median elapsed time is the CPU-cost proxy; this benchmark
//! declares the tolerance and prints it before registering any timed case, but
//! result comparison remains an analysis step over Criterion's estimates.
//! Timed outputs are retained until the sample stopwatch stops so allocator
//! reuse versus mmap/munmap thresholds cannot bias one fan-in. Flat sampling
//! keeps retained resident bytes approximately constant across each sample.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 RCH_WORKER=<pinned-worker> \
//!   rch exec -- cargo bench --profile release -p frankensearch-quill \
//!   --bench concat_merge_ab
//! ```

use std::fmt::Display;
use std::hint::black_box;
use std::mem::size_of;
use std::time::{Duration, Instant};

use criterion::{
    BenchmarkId, Criterion, SamplingMode, Throughput, criterion_group, criterion_main,
};
use frankensearch_quill::quiver::{StatsSection, aggregate_field_stats};
use frankensearch_quill::scribe::{
    ColumnarAccumulator, DOC_ORDS_PER_LEASE, FlushDocumentInput, FlushMode, FlushSegmentInput,
    IndexedFieldValue, IndexedNumericValue, StoredFieldValue, flush_accumulator_with_mode,
};
use frankensearch_quill::{
    Analyzer, CURRENT_ENGINE_VERSION, EncodedSegment, FieldDescriptor, FieldKind, KeeperSnapshot,
    ManifestFieldStats, ManifestSegment, SchemaDescriptor, SectionKind, SegmentReader,
    TombstoneSet,
};

const SOURCE_SEGMENT_COUNTS: [usize; 4] = [2, 4, 8, 16];
const FIXED_SPAN_LAST_LEASE_INDEX: usize = 15;
const TOTAL_LOGICAL_DOCS: usize = 1_024;
const TOTAL_TOMBSTONES: usize = 16;
const TOTAL_LIVE_DOCS: usize = TOTAL_LOGICAL_DOCS - TOTAL_TOMBSTONES;
const TOMBSTONE_LOGICAL_STRIDE: usize = TOTAL_LOGICAL_DOCS / TOTAL_TOMBSTONES;
const TOMBSTONE_LOCAL_OFFSET: usize = TOMBSTONE_LOGICAL_STRIDE / 2;
const DOCUMENT_ID_BYTES: usize = 8;
const BODY: &str = "shared alpha beta gamma delta epsilon deterministic concat payload";
const STORED_METADATA: &[u8] = b"fixture-metadata-v1";
const LOGICAL_BYTES_PER_DOCUMENT: usize =
    DOCUMENT_ID_BYTES + BODY.len() + size_of::<i64>() + size_of::<u64>() + STORED_METADATA.len();
const TOTAL_LOGICAL_BYTES: usize = TOTAL_LOGICAL_DOCS * LOGICAL_BYTES_PER_DOCUMENT;
const LEAF_SEGMENT_ID_BASE: u64 = 0xe351_0000_0000_0000;
const OUTPUT_SEGMENT_ID_BASE: u64 = 0xe350_0000_0000_0000;
const CREATED_UNIX_S: i64 = 1_700_000_035;
const PUBLISHED_UNIX_S: i64 = 1_700_000_036;

/// Predeclared bound for `max(median ns/physical I/O byte) / min(...)`.
const MAX_CPU_NS_PER_PHYSICAL_IO_BYTE_SPREAD_RATIO: f64 = 1.35;

const BENCH_FIELDS: [FieldDescriptor; 5] = [
    FieldDescriptor {
        id: 0,
        name: "id",
        kind: FieldKind::Keyword,
        stored: true,
    },
    FieldDescriptor {
        id: 1,
        name: "body",
        kind: FieldKind::Text {
            analyzer: Analyzer::FrankensearchDefault,
            positions: true,
        },
        stored: true,
    },
    FieldDescriptor {
        id: 2,
        name: "signed",
        kind: FieldKind::I64 {
            indexed: true,
            fast: true,
        },
        stored: true,
    },
    FieldDescriptor {
        id: 3,
        name: "unsigned",
        kind: FieldKind::U64 {
            indexed: true,
            fast: true,
        },
        stored: true,
    },
    FieldDescriptor {
        id: 4,
        name: "metadata",
        kind: FieldKind::StoredOnly,
        stored: true,
    },
];

const BENCH_SCHEMA: SchemaDescriptor = SchemaDescriptor {
    name: "concat-merge-ab",
    fields: &BENCH_FIELDS,
};
const STATS_FIELD_ORDS: [u16; 2] = [0, 1];

struct Fixture {
    span_mode: SpanMode,
    segment_count: usize,
    snapshot: KeeperSnapshot,
    source_segment_ids: Vec<u64>,
    lease_docid_bases: Vec<u64>,
    tombstoned_docids: Vec<u32>,
    source_fslx_bytes: u64,
    merged_output_fslx_bytes: u64,
    physical_io_bytes: u64,
    output_segment_id: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SpanMode {
    Growing,
    Fixed,
}

impl SpanMode {
    const fn label(self) -> &'static str {
        match self {
            Self::Growing => "growing-span",
            Self::Fixed => "fixed-span-control",
        }
    }
}

fn fail(stage: &str, error: &impl Display) -> ! {
    eprintln!("concat-merge-ab {stage} failed: {error}");
    std::process::exit(2);
}

fn fail_missing(stage: &str) -> ! {
    eprintln!("concat-merge-ab {stage} failed: required value is absent");
    std::process::exit(2);
}

fn fail_overflow(stage: &str) -> ! {
    eprintln!("concat-merge-ab {stage} failed: arithmetic overflow");
    std::process::exit(2);
}

fn require<T, E: Display>(stage: &str, result: Result<T, E>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => fail(stage, &error),
    }
}

fn require_some<T>(stage: &str, value: Option<T>) -> T {
    value.unwrap_or_else(|| fail_missing(stage))
}

fn as_u32(stage: &str, value: usize) -> u32 {
    match u32::try_from(value) {
        Ok(value) => value,
        Err(error) => fail(stage, &error),
    }
}

fn u64_as_u32(stage: &str, value: u64) -> u32 {
    match u32::try_from(value) {
        Ok(value) => value,
        Err(error) => fail(stage, &error),
    }
}

fn as_u64(stage: &str, value: usize) -> u64 {
    match u64::try_from(value) {
        Ok(value) => value,
        Err(error) => fail(stage, &error),
    }
}

fn checked_add(stage: &str, left: u64, right: u64) -> u64 {
    left.checked_add(right)
        .unwrap_or_else(|| fail_overflow(stage))
}

fn lease_base(segment_index: usize) -> u64 {
    as_u64("lease index", segment_index)
        .checked_mul(u64::from(DOC_ORDS_PER_LEASE))
        .unwrap_or_else(|| fail_overflow("lease base"))
}

fn fixture_lease_index(span_mode: SpanMode, segment_index: usize, segment_count: usize) -> usize {
    match span_mode {
        SpanMode::Growing => segment_index,
        SpanMode::Fixed => segment_index
            .checked_mul(FIXED_SPAN_LAST_LEASE_INDEX)
            .and_then(|numerator| numerator.checked_div(segment_count - 1))
            .unwrap_or_else(|| fail_overflow("fixed-span lease index")),
    }
}

fn document_id(logical_index: usize) -> String {
    let id = format!("d{logical_index:07}");
    assert_eq!(
        id.len(),
        DOCUMENT_ID_BYTES,
        "fixture identifier width drifted"
    );
    id
}

fn build_document_ids() -> Vec<String> {
    (0..TOTAL_LOGICAL_DOCS).map(document_id).collect()
}

fn build_leaf_segment(
    document_ids: &[String],
    first_document: usize,
    document_end: usize,
    segment_id: u64,
    lease_docid_base: u64,
) -> EncodedSegment {
    let mut accumulator = require(
        "construct leaf accumulator",
        ColumnarAccumulator::new(BENCH_SCHEMA),
    );
    let mut identity_rows = Vec::with_capacity(document_end - first_document);
    for (local_index, document_index) in (first_document..document_end).enumerate() {
        let document_ord = as_u32("lease-local document ordinal", local_index);
        let logical_ord = as_u32("logical document ordinal", document_index);
        let document_id = &document_ids[document_index];
        let indexed = [
            IndexedFieldValue::new(0, document_id),
            IndexedFieldValue::new(1, BODY),
        ];
        let numeric = [
            IndexedNumericValue::i64(2, i64::from(logical_ord % 31) - 15),
            IndexedNumericValue::u64(3, u64::from(logical_ord % 17)),
        ];
        let stored = [StoredFieldValue::new(4, STORED_METADATA)];
        require(
            "accumulate leaf document",
            accumulator.add_document_with_values(document_ord, &indexed, &numeric, &stored),
        );
        identity_rows.push(FlushDocumentInput::from_canonical_content(
            document_ord,
            document_id,
            BODY.as_bytes(),
        ));
    }

    require(
        "flush leaf segment",
        flush_accumulator_with_mode(
            &accumulator,
            FlushSegmentInput {
                segment_id,
                lease_docid_base,
                created_unix_s: CREATED_UNIX_S,
                engine_version: CURRENT_ENGINE_VERSION,
                documents: &identity_rows,
            },
            FlushMode::Scalar,
        ),
    )
}

fn leaf_stats(encoded: &EncodedSegment) -> StatsSection {
    let reader = require(
        "reopen leaf segment",
        SegmentReader::from_bytes(encoded.as_bytes(), BENCH_SCHEMA),
    );
    require("verify leaf segment", reader.verify());
    let bytes = require_some(
        "read leaf STATS",
        require("validate leaf STATS", reader.section(SectionKind::STATS)),
    );
    require(
        "parse leaf STATS",
        StatsSection::parse(bytes, &STATS_FIELD_ORDS, encoded.header().doc_count),
    )
}

fn build_source_tombstones(
    lease_docid_base: u64,
    documents_per_segment: usize,
    tombstones_per_segment: usize,
    tombstone_union: &mut Vec<u32>,
) -> TombstoneSet {
    let mut tombstones = TombstoneSet::new();
    for tombstone_index in 0..tombstones_per_segment {
        let local_ord = TOMBSTONE_LOCAL_OFFSET + tombstone_index * TOMBSTONE_LOGICAL_STRIDE;
        assert!(
            local_ord < documents_per_segment,
            "tombstone ordinal must name a physical source row"
        );
        let global_docid = u64_as_u32(
            "tombstoned global docid",
            checked_add(
                "tombstoned global docid",
                lease_docid_base,
                as_u64("tombstoned local ordinal", local_ord),
            ),
        );
        assert!(
            require("insert source tombstone", tombstones.insert(global_docid)),
            "fixture tombstone must be new"
        );
        tombstone_union.push(global_docid);
    }
    assert_eq!(
        tombstones.cardinality(),
        as_u64("source tombstone count", tombstones_per_segment)
    );
    tombstones
}

fn exact_source_fslx_bytes(snapshot: &KeeperSnapshot) -> u64 {
    let mut total = 0_u64;
    for segment in snapshot.segments() {
        let byte_count = as_u64("source FSLX byte length", segment.source_bytes().len());
        assert_eq!(
            byte_count,
            segment.manifest().file_len,
            "published FSLX length differs from its MANIFEST witness"
        );
        total = checked_add("sum exact source FSLX bytes", total, byte_count);
    }
    total
}

fn build_fixture(span_mode: SpanMode, segment_count: usize) -> Fixture {
    assert!(
        SOURCE_SEGMENT_COUNTS.contains(&segment_count),
        "unsupported source-segment count"
    );
    assert_eq!(
        TOTAL_LOGICAL_DOCS % segment_count,
        0,
        "documents must divide evenly across source segments"
    );
    assert_eq!(
        TOTAL_TOMBSTONES % segment_count,
        0,
        "tombstones must divide evenly across source segments"
    );
    let documents_per_segment = TOTAL_LOGICAL_DOCS / segment_count;
    let tombstones_per_segment = TOTAL_TOMBSTONES / segment_count;
    assert!(
        tombstones_per_segment > 0,
        "every source must carry at least one tombstone"
    );
    assert_eq!(documents_per_segment % TOMBSTONE_LOGICAL_STRIDE, 0);
    assert_eq!(
        documents_per_segment / TOMBSTONE_LOGICAL_STRIDE,
        tombstones_per_segment,
        "each source must receive its share of the fixed logical tombstone pattern"
    );
    let document_ids = build_document_ids();
    let mut encoded_segments = Vec::with_capacity(segment_count);
    let mut manifest_segments = Vec::with_capacity(segment_count);
    let mut stats_sections = Vec::with_capacity(segment_count);
    let mut tombstoned_docids = Vec::with_capacity(TOTAL_TOMBSTONES);
    let mut lease_docid_bases = Vec::with_capacity(segment_count);

    for segment_index in 0..segment_count {
        let first_document = segment_index * documents_per_segment;
        let document_end = first_document + documents_per_segment;
        let lease_docid_base =
            lease_base(fixture_lease_index(span_mode, segment_index, segment_count));
        lease_docid_bases.push(lease_docid_base);
        let segment_id = LEAF_SEGMENT_ID_BASE
            + (as_u64("source-segment fan-in", segment_count) << 16)
            + as_u64("source-segment index", segment_index);
        let encoded = build_leaf_segment(
            &document_ids,
            first_document,
            document_end,
            segment_id,
            lease_docid_base,
        );
        let header = encoded.header();
        assert_eq!(header.docid_lo, lease_docid_base);
        assert_eq!(
            header.docid_hi,
            checked_add(
                "leaf document upper bound",
                lease_docid_base,
                as_u64("documents per source", documents_per_segment),
            )
        );
        stats_sections.push(leaf_stats(&encoded));
        manifest_segments.push(ManifestSegment {
            segment_id,
            seal_seq: as_u64("source seal sequence", segment_index) + 1,
            file_len: encoded.file_len(),
            file_xxh3: encoded.file_xxh3(),
            docid_lo: header.docid_lo,
            docid_hi: header.docid_hi,
            doc_count: header.doc_count,
            tombstones: build_source_tombstones(
                lease_docid_base,
                documents_per_segment,
                tombstones_per_segment,
                &mut tombstoned_docids,
            ),
        });
        encoded_segments.push(encoded);
    }
    assert_eq!(tombstoned_docids.len(), TOTAL_TOMBSTONES);
    assert!(
        tombstoned_docids.windows(2).all(|pair| pair[0] < pair[1]),
        "fixture tombstone union must be strictly ordered"
    );

    let field_stats = require(
        "aggregate leaf STATS",
        aggregate_field_stats(stats_sections.iter()),
    )
    .into_iter()
    .map(|row| ManifestFieldStats {
        field_ord: row.field_ord,
        total_tokens: row.total_tokens,
        doc_count: u64_as_u32("aggregate field document count", row.doc_count),
    })
    .collect::<Vec<_>>();
    let genesis = require(
        "construct in-memory Keeper",
        KeeperSnapshot::in_memory(BENCH_SCHEMA),
    );
    let mut manifest = require("advance fixture MANIFEST", genesis.next_manifest());
    manifest.docid_high_watermark = checked_add(
        "fixture document-id high watermark",
        *lease_docid_bases
            .last()
            .unwrap_or_else(|| fail_missing("last fixture lease base")),
        u64::from(DOC_ORDS_PER_LEASE),
    );
    manifest.last_publish_unix_s = PUBLISHED_UNIX_S;
    manifest.segments = manifest_segments;
    manifest.field_stats = field_stats;
    require("validate fixture MANIFEST", manifest.validate());
    let snapshot = require(
        "publish in-memory leaf segments",
        genesis.publish_owned_segments(&manifest, encoded_segments),
    );
    assert_eq!(snapshot.segments().len(), segment_count);
    assert_eq!(
        snapshot.doc_count(),
        as_u64("total live document count", TOTAL_LIVE_DOCS)
    );
    assert_eq!(
        snapshot.at_seal_doc_count(),
        as_u64("total at-seal document count", TOTAL_LOGICAL_DOCS)
    );
    assert_eq!(
        snapshot.tombstone_count(),
        as_u64("total tombstone count", TOTAL_TOMBSTONES)
    );
    let source_segment_ids = snapshot
        .segments()
        .iter()
        .map(|segment| segment.manifest().segment_id)
        .collect::<Vec<_>>();
    let source_fslx_bytes = exact_source_fslx_bytes(&snapshot);
    let output_segment_id = OUTPUT_SEGMENT_ID_BASE + as_u64("output fan-in", segment_count);

    let mut fixture = Fixture {
        span_mode,
        segment_count,
        snapshot,
        source_segment_ids,
        lease_docid_bases,
        tombstoned_docids,
        source_fslx_bytes,
        merged_output_fslx_bytes: 0,
        physical_io_bytes: 0,
        output_segment_id: output_segment_id
            + match span_mode {
                SpanMode::Growing => 0,
                SpanMode::Fixed => 1 << 32,
            },
    };
    fixture.merged_output_fslx_bytes = assert_untimed_union_identity_reopen(&fixture);
    fixture.physical_io_bytes = checked_add(
        "exact physical I/O bytes",
        fixture.source_fslx_bytes,
        fixture.merged_output_fslx_bytes,
    );
    fixture
}

fn assert_untimed_union_identity_reopen(fixture: &Fixture) -> u64 {
    let source_manifest = &fixture.snapshot.loaded_manifest().manifest;
    let merged = require(
        "untimed concat-merge structural check",
        fixture.snapshot.concat_merge_owned(
            &fixture.source_segment_ids,
            fixture.output_segment_id,
            CREATED_UNIX_S,
        ),
    );
    assert_eq!(merged.segments().len(), 1, "merge must emit one segment");
    assert_eq!(
        merged.loaded_manifest().manifest.generation,
        source_manifest.generation + 1,
        "merge must publish exactly one successor generation"
    );
    assert_eq!(
        merged.loaded_manifest().manifest.docid_high_watermark,
        source_manifest.docid_high_watermark,
        "merge changed the allocation high-water mark"
    );
    assert_eq!(
        merged.doc_count(),
        as_u64("merged live document count", TOTAL_LIVE_DOCS),
        "merge changed live document count"
    );
    assert_eq!(
        merged.at_seal_doc_count(),
        as_u64("merged at-seal document count", TOTAL_LOGICAL_DOCS),
        "merge changed at-seal document count"
    );
    assert_eq!(
        merged.tombstone_count(),
        as_u64("merged tombstone count", TOTAL_TOMBSTONES),
        "merge changed tombstone cardinality"
    );
    assert_eq!(
        merged.loaded_manifest().manifest.field_stats,
        source_manifest.field_stats,
        "merge changed aggregate field statistics"
    );

    let output = &merged.segments()[0];
    assert_eq!(output.manifest().segment_id, fixture.output_segment_id);
    assert_eq!(
        output.manifest().docid_lo,
        fixture.snapshot.segments()[0].manifest().docid_lo
    );
    assert_eq!(
        output.manifest().docid_hi,
        fixture.snapshot.segments()[fixture.segment_count - 1]
            .manifest()
            .docid_hi
    );
    assert_eq!(
        output.manifest().tombstones.cardinality(),
        as_u64("merged tombstone union size", TOTAL_TOMBSTONES)
    );
    assert_eq!(
        output.doc_count(),
        as_u32("merged segment live document count", TOTAL_LIVE_DOCS)
    );
    assert_eq!(
        output.at_seal_doc_count(),
        as_u32("merged segment at-seal document count", TOTAL_LOGICAL_DOCS)
    );
    for &global_docid in &fixture.tombstoned_docids {
        assert!(
            output.manifest().tombstones.contains(global_docid),
            "merged tombstone union omitted global docid {global_docid}"
        );
        assert!(!fixture.snapshot.is_live(global_docid));
        assert!(!merged.is_live(global_docid));
        assert_eq!(fixture.snapshot.materialize_document_id(global_docid), None);
        assert_eq!(merged.materialize_document_id(global_docid), None);
    }

    let documents_per_segment = TOTAL_LOGICAL_DOCS / fixture.segment_count;
    for segment_index in 0..fixture.segment_count {
        let base = fixture.lease_docid_bases[segment_index];
        for local_index in 0..documents_per_segment {
            let global_docid = u64_as_u32(
                "identity-check global docid",
                checked_add(
                    "identity-check global docid",
                    base,
                    as_u64("identity-check local ordinal", local_index),
                ),
            );
            let logical_index = segment_index * documents_per_segment + local_index;
            let tombstoned = fixture
                .tombstoned_docids
                .binary_search(&global_docid)
                .is_ok();
            assert_eq!(
                tombstoned,
                logical_index % TOMBSTONE_LOGICAL_STRIDE == TOMBSTONE_LOCAL_OFFSET,
                "physical tombstone mapping drifted from the fixed logical pattern"
            );
            if tombstoned {
                assert!(!fixture.snapshot.is_live(global_docid));
                assert!(!merged.is_live(global_docid));
                continue;
            }
            assert!(fixture.snapshot.is_live(global_docid));
            assert!(merged.is_live(global_docid));
            let source_id = require_some(
                "materialize source live identity",
                fixture.snapshot.materialize_document_id(global_docid),
            );
            let merged_id = require_some(
                "materialize merged live identity",
                merged.materialize_document_id(global_docid),
            );
            let expected_id = document_id(logical_index);
            assert_eq!(source_id.as_str(), expected_id.as_str());
            assert_eq!(merged_id, source_id);
        }

        if let Some(next) = fixture.snapshot.segments().get(segment_index + 1) {
            let gap_docid = u64_as_u32(
                "representative burned-tail docid",
                fixture.snapshot.segments()[segment_index]
                    .manifest()
                    .docid_hi,
            );
            assert!(
                u64::from(gap_docid) < next.manifest().docid_lo,
                "adjacent source leases must retain a burned-tail gap"
            );
            assert!(!fixture.snapshot.is_live(gap_docid));
            assert!(!merged.is_live(gap_docid));
            assert_eq!(fixture.snapshot.materialize_document_id(gap_docid), None);
            assert_eq!(merged.materialize_document_id(gap_docid), None);
        }
    }

    require("verify merged segment", output.verify());
    let merged_output_fslx_bytes = as_u64(
        "merged output FSLX byte length",
        output.source_bytes().len(),
    );
    assert_eq!(merged_output_fslx_bytes, output.manifest().file_len);
    let reopened = require(
        "reopen merged FSLX bytes",
        SegmentReader::from_bytes(output.source_bytes(), BENCH_SCHEMA),
    );
    require("verify reopened merged FSLX", reopened.verify());
    assert_eq!(reopened.header(), output.header());
    assert_eq!(reopened.section_entries(), output.section_entries());
    assert_eq!(reopened.file_len(), output.manifest().file_len);
    assert_eq!(reopened.file_xxh3(), output.manifest().file_xxh3);
    merged_output_fslx_bytes
}

fn run_fixture_group(c: &mut Criterion, group_name: &str, fixtures: &[Fixture]) {
    let mut group = c.benchmark_group(group_name);
    group.sample_size(20);
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(5));
    for fixture in fixtures {
        group.throughput(Throughput::Bytes(fixture.physical_io_bytes));
        group.bench_with_input(
            BenchmarkId::new("source_segments", fixture.segment_count),
            fixture,
            |b, fixture| {
                b.iter_custom(|iterations| {
                    let capacity = usize::try_from(iterations)
                        .expect("Criterion iteration count must fit usize");
                    let mut retained = Vec::new();
                    retained
                        .try_reserve_exact(capacity)
                        .expect("timed output retention allocation must succeed");
                    let started = Instant::now();
                    for _ in 0..iterations {
                        retained.push(require(
                            "timed concat merge",
                            black_box(&fixture.snapshot).concat_merge_owned(
                                black_box(fixture.source_segment_ids.as_slice()),
                                black_box(fixture.output_segment_id),
                                black_box(CREATED_UNIX_S),
                            ),
                        ));
                    }
                    let elapsed = started.elapsed();
                    black_box(&retained);
                    drop(retained);
                    elapsed
                });
            },
        );
    }
    group.finish();
}

fn bench_concat_merge(c: &mut Criterion) {
    eprintln!(
        "concat-merge-ab predeclared fixed-span CPU/physical-I/O-byte spread: \
         max(median ns / exact (source + merged-output) FSLX byte) / min(...) <= \
         {MAX_CPU_NS_PER_PHYSICAL_IO_BYTE_SPREAD_RATIO:.2}x"
    );
    let growing_span_fixtures = SOURCE_SEGMENT_COUNTS
        .into_iter()
        .map(|segment_count| build_fixture(SpanMode::Growing, segment_count))
        .collect::<Vec<_>>();
    let fixed_span_fixtures = SOURCE_SEGMENT_COUNTS
        .into_iter()
        .map(|segment_count| build_fixture(SpanMode::Fixed, segment_count))
        .collect::<Vec<_>>();
    for fixture in growing_span_fixtures.iter().chain(&fixed_span_fixtures) {
        eprintln!(
            "concat-merge-ab fixture: span_mode={} source_segments={} logical_docs={} live_docs={} \
             tombstones={} logical_bytes={} exact_source_fslx_bytes={} \
             exact_merged_output_fslx_bytes={} exact_physical_io_bytes={}",
            fixture.span_mode.label(),
            fixture.segment_count,
            TOTAL_LOGICAL_DOCS,
            TOTAL_LIVE_DOCS,
            TOTAL_TOMBSTONES,
            TOTAL_LOGICAL_BYTES,
            fixture.source_fslx_bytes,
            fixture.merged_output_fslx_bytes,
            fixture.physical_io_bytes,
        );
    }

    run_fixture_group(
        c,
        "concat_merge_owned_fixed_logical_work",
        &growing_span_fixtures,
    );
    run_fixture_group(
        c,
        "concat_merge_owned_fixed_span_control",
        &fixed_span_fixtures,
    );
}

criterion_group!(benches, bench_concat_merge);
criterion_main!(benches);
