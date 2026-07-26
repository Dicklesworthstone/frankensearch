//! Hash/arena-chain Delta seal vs columnar/radix Scribe seal
//! (`bd-quill-e1-scribe-bejd.7`).
//!
//! Both arms receive byte-identical documents under [`DEFAULT_SCHEMA`]. Setup
//! builds the two immutable source representations once and reports their
//! accumulation time and retained bytes separately. The timed window contains
//! the complete canonical FSLX seal in both arms: Delta sorts its term-keyed
//! arena chains, while Scribe collects columnar rows and stable-radix
//! partitions them. Every cell proves exact output-byte parity before timing.
//!
//! The full scale pins the fsfs golden small (5,000 docs / 1.45M tokens) and
//! medium (50,000 / 17.5M) shapes. The xlarge cell is one maximum-sized Quill
//! lease sampled from the pending 1M-doc synthetic lane, because one FSLX
//! mini-segment cannot contain more than 65,536 local ordinals. `smoke` is only
//! for harness development and must never be entered in the performance
//! ledger.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR \
//!   rch exec -- env QUILL_E17_SCALE=full QUILL_E17_PROFILE=all \
//!     QUILL_E17_THREADS=all QUILL_E17_ROUNDS=9 \
//!     cargo bench -p frankensearch-quill --features bench-internals \
//!       --profile release --bench scribe_flush_ab
//!
//! # Fast parity/harness check (not performance evidence):
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR \
//!   rch exec -- env QUILL_E17_SCALE=smoke QUILL_E17_ROUNDS=3 \
//!     QUILL_E17_THREADS=1 \
//!     cargo bench -p frankensearch-quill --features bench-internals \
//!       --profile release --bench scribe_flush_ab
//! ```

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::hint::black_box;
use std::time::{Duration, Instant};

use frankensearch_core::DocId;
use frankensearch_core::bench_support::{PairedRatio, paired_median_ratio, print_bench_elf_sha256};
use frankensearch_quill::contract::fieldnorm_to_id;
use frankensearch_quill::delta::{
    DeltaFieldNorm, DeltaSegment, DeltaSnapshot, DeltaStoredValue, DeltaTermPosting,
};
use frankensearch_quill::scribe::{
    ColumnarAccumulator, DeltaFlushInput, FlushDocumentInput, FlushMode, FlushSegmentInput,
    IndexedFieldValue, StoredFieldValue, flush_accumulator_with_mode, flush_delta_snapshot,
};
use frankensearch_quill::{CURRENT_ENGINE_VERSION, DEFAULT_SCHEMA, EncodedSegment};
use rayon::{ThreadPool, ThreadPoolBuilder};
use xxhash_rust::xxh3::xxh3_64;

const ID_FIELD: u16 = 0;
const CONTENT_FIELD: u16 = 1;
const TITLE_FIELD: u16 = 2;
const METADATA_FIELD: u16 = 3;
const ORD_FIELD: u16 = 4;
const METADATA: &[u8] = b"{}";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scale {
    Full,
    Smoke,
}

impl Scale {
    fn from_env() -> Self {
        match std::env::var("QUILL_E17_SCALE").as_deref() {
            Ok("smoke") => Self::Smoke,
            Ok("full") | Err(_) => Self::Full,
            Ok(other) => panic!("QUILL_E17_SCALE must be full or smoke, got {other}"),
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Smoke => "smoke",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Profile {
    name: &'static str,
    document_count: usize,
    tokens_per_document: usize,
    vocabulary_size: usize,
}

impl Profile {
    const fn token_count(self) -> usize {
        self.document_count * self.tokens_per_document
    }
}

const FULL_PROFILES: [Profile; 3] = [
    Profile {
        name: "golden-small",
        document_count: 5_000,
        tokens_per_document: 290,
        vocabulary_size: 16_384,
    },
    Profile {
        name: "golden-medium",
        document_count: 50_000,
        tokens_per_document: 350,
        vocabulary_size: 131_072,
    },
    Profile {
        name: "synthetic-xlarge-max-lease",
        document_count: 65_536,
        tokens_per_document: 350,
        vocabulary_size: 262_144,
    },
];

const SMOKE_PROFILES: [Profile; 3] = [
    Profile {
        name: "golden-small-smoke",
        document_count: 128,
        tokens_per_document: 48,
        vocabulary_size: 512,
    },
    Profile {
        name: "golden-medium-smoke",
        document_count: 1_024,
        tokens_per_document: 64,
        vocabulary_size: 4_096,
    },
    Profile {
        name: "synthetic-xlarge-smoke",
        document_count: 4_096,
        tokens_per_document: 96,
        vocabulary_size: 16_384,
    },
];

#[derive(Debug)]
struct FixtureDocument {
    id: String,
    content: String,
    content_hash: u64,
}

#[derive(Debug)]
struct PreparedCase {
    columnar: ColumnarAccumulator,
    delta: DeltaSnapshot,
    columnar_accumulation: Duration,
    delta_accumulation: Duration,
}

fn selected_profiles(scale: Scale) -> Vec<Profile> {
    let profiles = match scale {
        Scale::Full => FULL_PROFILES.as_slice(),
        Scale::Smoke => SMOKE_PROFILES.as_slice(),
    };
    let selected = std::env::var("QUILL_E17_PROFILE").unwrap_or_else(|_| "all".to_owned());
    profiles
        .iter()
        .copied()
        .filter(|profile| selected == "all" || selected == profile.name)
        .collect()
}

fn selected_threads() -> Vec<usize> {
    match std::env::var("QUILL_E17_THREADS").as_deref() {
        Ok("1") => vec![1],
        Ok("8") => vec![8],
        Ok("all") | Err(_) => vec![1, 8],
        Ok(other) => panic!("QUILL_E17_THREADS must be 1, 8, or all, got {other}"),
    }
}

fn rounds(scale: Scale) -> usize {
    std::env::var("QUILL_E17_ROUNDS").map_or_else(
        |_| match scale {
            Scale::Full => 9,
            Scale::Smoke => 15,
        },
        |value| {
            value
                .parse::<usize>()
                .expect("QUILL_E17_ROUNDS must be a positive integer")
        },
    )
}

fn build_corpus(profile: Profile) -> Vec<FixtureDocument> {
    let mut corpus = Vec::with_capacity(profile.document_count);
    for document_index in 0..profile.document_count {
        let id = format!("doc-{document_index:08}");
        let mut content = String::with_capacity(profile.tokens_per_document.saturating_mul(12));
        let mut state = (u64::try_from(document_index).expect("document index fits u64") + 1)
            .wrapping_mul(0x9e37_79b9_7f4a_7c15);
        for token_index in 0..profile.tokens_per_document {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let hot_rank = usize::try_from(state >> 32).expect("upper u32 fits usize");
            let term_index = (hot_rank
                .wrapping_add(token_index.wrapping_mul(17))
                .wrapping_add(document_index.wrapping_mul(31)))
                % profile.vocabulary_size;
            if !content.is_empty() {
                content.push(' ');
            }
            write!(&mut content, "term{term_index:06}").expect("writing to String is infallible");
        }
        let content_hash = xxh3_64(id.as_bytes()) ^ xxh3_64(content.as_bytes()).rotate_left(1);
        corpus.push(FixtureDocument {
            id,
            content,
            content_hash,
        });
    }
    corpus
}

fn build_columnar(corpus: &[FixtureDocument]) -> ColumnarAccumulator {
    let mut accumulator =
        ColumnarAccumulator::new(DEFAULT_SCHEMA).expect("default schema must be valid");
    for (document_index, document) in corpus.iter().enumerate() {
        let document_ord = u32::try_from(document_index).expect("profile fits one Quill lease");
        let ordinal = u64::from(document_ord).to_le_bytes();
        accumulator
            .add_document_with_values(
                document_ord,
                &[
                    IndexedFieldValue::new(ID_FIELD, &document.id),
                    IndexedFieldValue::new(CONTENT_FIELD, &document.content),
                    IndexedFieldValue::new(TITLE_FIELD, ""),
                ],
                &[],
                &[
                    StoredFieldValue::new(METADATA_FIELD, METADATA),
                    StoredFieldValue::new(ORD_FIELD, &ordinal),
                ],
            )
            .expect("columnar fixture document must accumulate");
    }
    accumulator
}

fn build_delta(corpus: &[FixtureDocument]) -> DeltaSnapshot {
    let mut delta =
        DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("default Delta lease must build");
    for (document_index, document) in corpus.iter().enumerate() {
        let document_ord = u32::try_from(document_index).expect("profile fits one Quill lease");
        let mut term_positions = BTreeMap::<&str, Vec<u32>>::new();
        for (position, term) in document.content.split_ascii_whitespace().enumerate() {
            term_positions
                .entry(term)
                .or_default()
                .push(u32::try_from(position).expect("fixture position fits u32"));
        }
        let token_count = u32::try_from(document.content.split_ascii_whitespace().count())
            .expect("fixture token count fits u32");
        let fieldnorms = [
            DeltaFieldNorm {
                field_ord: ID_FIELD,
                raw_length: 1,
                fieldnorm_id: fieldnorm_to_id(1),
            },
            DeltaFieldNorm {
                field_ord: CONTENT_FIELD,
                raw_length: token_count,
                fieldnorm_id: fieldnorm_to_id(token_count),
            },
            DeltaFieldNorm {
                field_ord: TITLE_FIELD,
                raw_length: 0,
                fieldnorm_id: fieldnorm_to_id(0),
            },
        ];
        let mut postings = Vec::with_capacity(term_positions.len() + 1);
        postings.push(DeltaTermPosting {
            field_ord: ID_FIELD,
            term: document.id.as_bytes(),
            frequency: 1,
            positions: None,
        });
        for (term, positions) in &term_positions {
            postings.push(DeltaTermPosting {
                field_ord: CONTENT_FIELD,
                term: term.as_bytes(),
                frequency: u32::try_from(positions.len()).expect("term frequency fits u32"),
                positions: Some(positions),
            });
        }
        let ordinal = u64::from(document_ord).to_le_bytes();
        let stored = [
            DeltaStoredValue::new(ID_FIELD, document.id.as_bytes()),
            DeltaStoredValue::new(CONTENT_FIELD, document.content.as_bytes()),
            DeltaStoredValue::new(TITLE_FIELD, b""),
            DeltaStoredValue::new(METADATA_FIELD, METADATA),
            DeltaStoredValue::new(ORD_FIELD, &ordinal),
        ];
        delta
            .apply_document_with_values(
                document_ord,
                DocId::from(document.id.as_str()),
                document.content_hash,
                &fieldnorms,
                &postings,
                &[],
                &stored,
            )
            .expect("Delta fixture document must accumulate");
    }
    delta.freeze(1)
}

fn prepare_case(corpus: &[FixtureDocument]) -> PreparedCase {
    let columnar_started = Instant::now();
    let columnar = build_columnar(corpus);
    let columnar_accumulation = columnar_started.elapsed();
    let delta_started = Instant::now();
    let delta = build_delta(corpus);
    let delta_accumulation = delta_started.elapsed();
    PreparedCase {
        columnar,
        delta,
        columnar_accumulation,
        delta_accumulation,
    }
}

const fn metadata() -> DeltaFlushInput {
    DeltaFlushInput {
        segment_id: 0xe1_7000,
        created_unix_s: 1_700_000_000,
        engine_version: CURRENT_ENGINE_VERSION,
    }
}

fn flush_hash(prepared: &PreparedCase) -> EncodedSegment {
    flush_delta_snapshot(&prepared.delta, metadata())
        .expect("Delta seal must succeed")
        .expect("the fixture is nonempty")
}

fn flush_radix(
    prepared: &PreparedCase,
    documents: &[FlushDocumentInput<'_>],
    pool: &ThreadPool,
) -> EncodedSegment {
    let fixed = metadata();
    pool.install(|| {
        flush_accumulator_with_mode(
            &prepared.columnar,
            FlushSegmentInput {
                segment_id: fixed.segment_id,
                lease_docid_base: 0,
                created_unix_s: fixed.created_unix_s,
                engine_version: fixed.engine_version,
                documents,
            },
            FlushMode::Automatic,
        )
        .expect("columnar/radix seal must succeed")
    })
}

fn print_ratio(kind: &str, profile: Profile, threads: usize, ratio: PairedRatio) {
    eprintln!(
        "[{kind}] {}/{}t: new/old median {:.4} median_ci95 [{:.4}, {:.4}] \
         p5 {:.4} p95 {:.4} ({} rounds)",
        profile.name,
        threads,
        ratio.median,
        ratio.median_ci95_low,
        ratio.median_ci95_high,
        ratio.p5,
        ratio.p95,
        ratio.rounds
    );
}

fn run_cell(
    profile: Profile,
    threads: usize,
    rounds: usize,
    prepared: &PreparedCase,
    documents: &[FlushDocumentInput<'_>],
) {
    let pool = ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(move |index| format!("quill-e17-{threads}t-{index}"))
        .build()
        .expect("fixed Rayon pool must build");
    let old = flush_hash(prepared);
    let new = flush_radix(prepared, documents, &pool);
    assert_eq!(
        old.as_bytes(),
        new.as_bytes(),
        "hash-chain and columnar/radix seals must be byte-identical"
    );
    let output_bytes = old.file_len();
    drop((old, new));

    let null = paired_median_ratio(
        rounds,
        1,
        || {
            black_box(flush_hash(black_box(prepared)).file_len());
        },
        || {
            black_box(flush_hash(black_box(prepared)).file_len());
        },
    );
    let lever = paired_median_ratio(
        rounds,
        1,
        || {
            black_box(flush_hash(black_box(prepared)).file_len());
        },
        || {
            black_box(flush_radix(black_box(prepared), black_box(documents), &pool).file_len());
        },
    );
    print_ratio("null", profile, threads, null);
    print_ratio("lever", profile, threads, lever);
    let decision = if !lever.decidable_against(&null) || (0.97..=1.03).contains(&lever.median) {
        "NOISE"
    } else if lever.median < 1.0 {
        "COLUMNAR_RADIX_WINS"
    } else {
        "HASH_CHAIN_WINS"
    };
    eprintln!(
        "[decision] {}/{}t: {decision}; docs={} tokens={} terms={} output_bytes={} \
         columnar_accumulate_ms={:.3} delta_accumulate_ms={:.3} \
         columnar_bytes={} delta_bytes={} byte_parity=PASS",
        profile.name,
        threads,
        profile.document_count,
        profile.token_count(),
        prepared.columnar.terms().len(),
        output_bytes,
        prepared.columnar_accumulation.as_secs_f64() * 1_000.0,
        prepared.delta_accumulation.as_secs_f64() * 1_000.0,
        prepared.columnar.bytes_used(),
        prepared.delta.segment().memory_stats().bytes_used,
    );
}

fn main() {
    let _identity = print_bench_elf_sha256().expect("hash executing Scribe benchmark");
    let scale = Scale::from_env();
    let profiles = selected_profiles(scale);
    assert!(
        !profiles.is_empty(),
        "QUILL_E17_PROFILE did not select a profile at scale {}",
        scale.label()
    );
    let threads = selected_threads();
    let rounds = rounds(scale);
    assert!(rounds > 0, "QUILL_E17_ROUNDS must be nonzero");
    eprintln!(
        "[config] scale={} profiles={} threads={threads:?} rounds={rounds}",
        scale.label(),
        profiles.len()
    );
    for profile in profiles {
        let corpus_started = Instant::now();
        let corpus = build_corpus(profile);
        let corpus_elapsed = corpus_started.elapsed();
        let documents = corpus
            .iter()
            .enumerate()
            .map(|(document_index, document)| {
                FlushDocumentInput::new(
                    u32::try_from(document_index).expect("profile fits one Quill lease"),
                    &document.id,
                    document.content_hash,
                )
            })
            .collect::<Vec<_>>();
        let prepared = prepare_case(&corpus);
        eprintln!(
            "[setup] {}: corpus_ms={:.3} columnar_ms={:.3} delta_ms={:.3}",
            profile.name,
            corpus_elapsed.as_secs_f64() * 1_000.0,
            prepared.columnar_accumulation.as_secs_f64() * 1_000.0,
            prepared.delta_accumulation.as_secs_f64() * 1_000.0,
        );
        for &thread_count in &threads {
            run_cell(profile, thread_count, rounds, &prepared, &documents);
        }
    }
}
