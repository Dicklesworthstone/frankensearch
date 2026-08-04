//! E3.6 tombstone-folding compaction versus canonical live-document rebuild.
//!
//! Every timed cell starts from one immutable 20k-document FSLX segment with
//! exactly 5%, 20%, or 50% scattered tombstones. Zero-density input is
//! deliberately absent: it cannot execute the machinery under test. Before
//! timing, shipped compaction and a fresh Scribe rebuild of the same live rows
//! must emit byte-identical FSLX files with the same sparse global docids.
//!
//! The paired A/A rebuild null must bracket 1.0 before the corresponding A/B is
//! admissible. A throughput KEEP additionally requires a decidable
//! `compaction/rebuild <= 0.90` at every density; a slower result is an explicit
//! REJECT, not an implementation failure or a hidden performance claim. The
//! benchmark is an E3.6 measurement gate; the later QG-5 matrix remains
//! responsible for the 1M-document Tantivy comparison.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 rch exec -- env QUILL_E36_SCALE=full \
//!   cargo bench -p frankensearch-quill --profile release \
//!   --bench compaction_ab
//! ```

use std::hint::black_box;

use frankensearch_core::bench_support::{PairedRatio, paired_median_ratio};
use frankensearch_quill::quiver::StatsSection;
use frankensearch_quill::scribe::{
    ColumnarAccumulator, FlushDocumentInput, FlushMode, FlushSegmentInput, IndexedFieldValue,
    StoredFieldValue, flush_accumulator_with_mode,
};
use frankensearch_quill::{
    CURRENT_ENGINE_VERSION, CompactionPolicy, DEFAULT_SCHEMA, EncodedSegment, KeeperSnapshot,
    ManifestFieldStats, ManifestSegment, SectionKind, SegmentReader, TombstoneSet,
};

const FULL_DOCUMENTS: usize = 20_000;
const SMOKE_DOCUMENTS: usize = 2_000;
const DENSITY_PERCENTAGES: [usize; 3] = [5, 20, 50];
const FULL_ROUNDS: usize = 9;
const SMOKE_ROUNDS: usize = 5;
const BODY: &str = "shared alpha beta gamma delta epsilon compact parity";
const TITLE: &str = "compact fixture";
const METADATA: &[u8] = b"{\"fixture\":\"e3.6\"}";
const SOURCE_SEGMENT_ID: u64 = 0xe360_0000_0000_0001;
const CREATED_UNIX_S: i64 = 1_700_000_036;
const TERM_FIELDS: [u16; 3] = [0, 1, 2];
const MAX_COMPACTION_TO_REBUILD_RATIO: f64 = 0.90;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Scale {
    Full,
    Smoke,
}

impl Scale {
    fn from_env() -> Self {
        match std::env::var("QUILL_E36_SCALE").as_deref() {
            Ok("smoke") => Self::Smoke,
            Ok("full") | Err(_) => Self::Full,
            Ok(other) => panic!("QUILL_E36_SCALE must be full or smoke, got {other}"),
        }
    }

    const fn document_count(self) -> usize {
        match self {
            Self::Full => FULL_DOCUMENTS,
            Self::Smoke => SMOKE_DOCUMENTS,
        }
    }

    const fn rounds(self) -> usize {
        match self {
            Self::Full => FULL_ROUNDS,
            Self::Smoke => SMOKE_ROUNDS,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Smoke => "smoke",
        }
    }
}

struct Corpus {
    ids: Vec<String>,
}

impl Corpus {
    fn new(document_count: usize) -> Self {
        let ids = (0..document_count)
            .map(|ordinal| format!("compact-{ordinal:08}"))
            .collect();
        Self { ids }
    }
}

struct Case<'a> {
    corpus: &'a Corpus,
    tombstoned: KeeperSnapshot,
    deleted: Vec<bool>,
    policy: CompactionPolicy,
    output_segment_id: u64,
    source_bytes: u64,
    output_bytes: u64,
    density_percentage: usize,
}

fn add_document(accumulator: &mut ColumnarAccumulator, ordinal: u32, document_id: &str) {
    let ordinal_bytes = u64::from(ordinal).to_le_bytes();
    accumulator
        .add_document_with_values(
            ordinal,
            &[
                IndexedFieldValue::new(0, document_id),
                IndexedFieldValue::new(1, BODY),
                IndexedFieldValue::new(2, TITLE),
            ],
            &[],
            &[
                StoredFieldValue::new(3, METADATA),
                StoredFieldValue::new(4, &ordinal_bytes),
            ],
        )
        .expect("E3.6 fixture document must accumulate");
}

fn flush_corpus(corpus: &Corpus, deleted: &[bool], segment_id: u64) -> EncodedSegment {
    let mut accumulator =
        ColumnarAccumulator::new(DEFAULT_SCHEMA).expect("default schema must build");
    let mut identities = Vec::with_capacity(corpus.ids.len());
    for (ordinal, document_id) in corpus.ids.iter().enumerate() {
        if deleted[ordinal] {
            continue;
        }
        let ordinal = u32::try_from(ordinal).expect("E3.6 fixture fits one lease");
        add_document(&mut accumulator, ordinal, document_id);
        identities.push(FlushDocumentInput::from_canonical_content(
            ordinal,
            document_id,
            BODY.as_bytes(),
        ));
    }
    flush_accumulator_with_mode(
        &accumulator,
        FlushSegmentInput {
            segment_id,
            lease_docid_base: 0,
            created_unix_s: CREATED_UNIX_S,
            engine_version: CURRENT_ENGINE_VERSION,
            documents: &identities,
        },
        FlushMode::Scalar,
    )
    .expect("E3.6 canonical rebuild must flush")
}

fn source_snapshot(corpus: &Corpus) -> KeeperSnapshot {
    let deleted = vec![false; corpus.ids.len()];
    let encoded = flush_corpus(corpus, &deleted, SOURCE_SEGMENT_ID);
    let reader = SegmentReader::from_bytes(encoded.as_bytes(), DEFAULT_SCHEMA)
        .expect("reopen E3.6 source segment");
    reader.verify().expect("verify E3.6 source segment");
    let stats = StatsSection::parse(
        reader
            .section(SectionKind::STATS)
            .expect("validate E3.6 source STATS")
            .expect("E3.6 source STATS are present"),
        &TERM_FIELDS,
        encoded.header().doc_count,
    )
    .expect("parse E3.6 source STATS");

    let genesis = KeeperSnapshot::in_memory(DEFAULT_SCHEMA).expect("E3.6 in-memory genesis");
    let mut manifest = genesis.next_manifest().expect("E3.6 source MANIFEST");
    manifest.docid_high_watermark =
        u64::try_from(corpus.ids.len()).expect("fixture count fits u64");
    manifest.segments.push(ManifestSegment {
        segment_id: encoded.header().segment_id,
        seal_seq: 1,
        file_len: encoded.file_len(),
        file_xxh3: encoded.file_xxh3(),
        docid_lo: encoded.header().docid_lo,
        docid_hi: encoded.header().docid_hi,
        doc_count: encoded.header().doc_count,
        tombstones: TombstoneSet::new(),
    });
    manifest.field_stats = stats
        .rows()
        .iter()
        .map(|row| ManifestFieldStats {
            field_ord: row.field_ord,
            total_tokens: row.total_tokens,
            doc_count: row.doc_count,
        })
        .collect();
    genesis
        .publish_owned_segments(&manifest, vec![encoded])
        .expect("publish E3.6 source segment")
}

fn deleted_rows(document_count: usize, density_percentage: usize) -> Vec<bool> {
    let deleted_count = document_count * density_percentage / 100;
    let stride = document_count / deleted_count;
    let mut deleted = vec![false; document_count];
    for index in 0..deleted_count {
        let ordinal = index * stride + usize::from(index + 1 != deleted_count);
        assert!(
            ordinal < document_count - 1,
            "keep the final covering row live"
        );
        assert!(!deleted[ordinal], "scattered tombstones must be unique");
        deleted[ordinal] = true;
    }
    assert_eq!(
        deleted.iter().filter(|&&value| value).count(),
        deleted_count
    );
    deleted
}

fn build_case<'a>(
    corpus: &'a Corpus,
    source: &KeeperSnapshot,
    density_percentage: usize,
) -> Case<'a> {
    let deleted = deleted_rows(corpus.ids.len(), density_percentage);
    let mut manifest = source.next_manifest().expect("E3.6 tombstone MANIFEST");
    for (ordinal, document_id) in corpus.ids.iter().enumerate() {
        if deleted[ordinal] {
            assert!(
                source
                    .delete_document(&mut manifest, document_id)
                    .expect("stage E3.6 tombstone"),
            );
        }
    }
    let tombstoned = source
        .publish_owned_segments(&manifest, Vec::new())
        .expect("publish E3.6 tombstone snapshot");
    let density = density_percentage as f64 / 100.0;
    let policy = CompactionPolicy::new(density - 0.001);
    let (compacted, report) = tombstoned
        .compact_owned(policy, CREATED_UNIX_S)
        .expect("preflight E3.6 compaction");
    assert!(
        report.changed(),
        "every nonzero benchmark density must compact"
    );
    assert_eq!(report.compacted_segments, 1);
    assert_eq!(
        usize::try_from(report.dropped_documents).expect("fixture tombstones fit usize"),
        deleted.iter().filter(|&&v| v).count()
    );
    let replacement = &compacted.segments()[0];
    let rebuilt = flush_corpus(corpus, &deleted, replacement.header().segment_id);
    let replacement_bytes = replacement.source_bytes();
    let rebuilt_bytes = rebuilt.as_bytes();
    let first_difference = replacement_bytes
        .iter()
        .zip(rebuilt_bytes)
        .position(|(left, right)| left != right);
    assert!(
        replacement_bytes == rebuilt_bytes,
        "compaction/rebuild byte mismatch: replacement_len={} rebuild_len={} replacement_xxh3={:#018x} rebuild_xxh3={:#018x} first_difference={first_difference:?}",
        replacement_bytes.len(),
        rebuilt_bytes.len(),
        xxhash_rust::xxh3::xxh3_64(replacement_bytes),
        xxhash_rust::xxh3::xxh3_64(rebuilt_bytes),
    );
    Case {
        corpus,
        tombstoned,
        deleted,
        policy,
        output_segment_id: replacement.header().segment_id,
        source_bytes: report.input_bytes,
        output_bytes: report.output_bytes,
        density_percentage,
    }
}

fn compact(case: &Case<'_>) -> u64 {
    let (snapshot, report) = case
        .tombstoned
        .compact_owned(case.policy, CREATED_UNIX_S)
        .expect("timed E3.6 compaction");
    assert_eq!(report.compacted_segments, 1);
    black_box(snapshot.segments()[0].manifest().file_len)
}

fn rebuild(case: &Case<'_>) -> u64 {
    black_box(flush_corpus(case.corpus, &case.deleted, case.output_segment_id).file_len())
}

fn print_ratio(kind: &str, case: &Case<'_>, ratio: PairedRatio) {
    eprintln!(
        "[{kind}] density={}pct compaction/rebuild median={:.4} p5={:.4} p95={:.4} rounds={}",
        case.density_percentage, ratio.median, ratio.p5, ratio.p95, ratio.rounds,
    );
}

fn main() {
    let scale = Scale::from_env();
    let document_count = scale.document_count();
    let rounds = scale.rounds();
    eprintln!(
        "[gate] scale={} documents={} densities={:?} zero_density=FORBIDDEN max_compaction_to_rebuild_ratio={MAX_COMPACTION_TO_REBUILD_RATIO:.2} null_must_be_unbiased=true decidable=true",
        scale.label(),
        document_count,
        DENSITY_PERCENTAGES,
    );
    let corpus = Corpus::new(document_count);
    let source = source_snapshot(&corpus);
    let mut evidence_admissible = true;
    let mut all_throughput_keep = true;
    for density_percentage in DENSITY_PERCENTAGES {
        let case = build_case(&corpus, &source, density_percentage);
        let null = paired_median_ratio(
            rounds,
            1,
            || {
                black_box(rebuild(&case));
            },
            || {
                black_box(rebuild(&case));
            },
        );
        let lever = paired_median_ratio(
            rounds,
            1,
            || {
                black_box(rebuild(&case));
            },
            || {
                black_box(compact(&case));
            },
        );
        print_ratio("null", &case, null);
        print_ratio("lever", &case, lever);
        // `decidable_against` already requires an admissible control; this is
        // reported separately so a HOLD names which half failed. It used to be
        // an independent `p5 <= 1.0 <= p95` straddle test, which vetoed a null
        // for being tight rather than for being wrong (`bd-pjh09`).
        let null_admissible = null.is_admissible_null();
        let decidable = lever.decidable_against(&null);
        let throughput_keep = decidable && lever.median <= MAX_COMPACTION_TO_REBUILD_RATIO;
        evidence_admissible &= null_admissible;
        all_throughput_keep &= throughput_keep;
        let decision = if throughput_keep {
            "KEEP"
        } else if decidable {
            "REJECT"
        } else {
            "HOLD"
        };
        eprintln!(
            "[cell] density={}pct threshold={:.3} source_bytes={} output_bytes={} live_docs={} byte_parity=PASS decision={}",
            density_percentage,
            case.policy.tombstone_density,
            case.source_bytes,
            case.output_bytes,
            case.deleted.iter().filter(|&&value| !value).count(),
            decision,
        );
    }
    eprintln!(
        "[gate-summary] evidence={} throughput_claim={}",
        if evidence_admissible { "PASS" } else { "HOLD" },
        if all_throughput_keep {
            "KEEP"
        } else {
            "REJECT_OR_HOLD"
        },
    );
    assert!(
        evidence_admissible,
        "E3.6 compaction A/B null control did not admit a performance conclusion"
    );
}
