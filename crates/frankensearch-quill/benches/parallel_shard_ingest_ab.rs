//! Shared-nothing parallel segment build vs today's serial shard fill.
//!
//! Quill already partitions ingest into `W` shard accumulators
//! ([`frankensearch_quill::scribe::ShardRouter`] routes whole batches
//! round-robin), but the shards are a *memory* partition only: every
//! accumulate and every seal runs inline on the single task that holds the
//! exclusive writer lock. This bench measures the ceiling of turning that
//! partition into a *compute* partition.
//!
//! Three configurations are scored against one A/A null, all producing
//! byte-identical sealed segments from byte-identical shard assignment:
//!
//! - **baseline** — serial shard walk, `FlushMode::Scalar`. This is what
//!   `index.rs` does today: all four production seal sites force the
//!   single-threaded radix.
//! - **seal-automatic** (lever 1a) — serial shard walk, `FlushMode::Automatic`.
//!   The parallel radix seal already exists, is the crate default, and is
//!   parity-tested against Scalar in `scribe.rs`; production simply never asks
//!   for it.
//! - **shard-fanout** (lever 1b) — shared-nothing shard fan-out, seal mode
//!   unchanged, so the win is attributable purely to parallel accumulation.
//!
//! Parity is asserted per shard before any timing.
//!
//! ```bash
//! QUILL_PSI_DOCS=50000 QUILL_PSI_SHARDS=all QUILL_PSI_ROUNDS=9 \
//!   cargo bench -p frankensearch-quill --features bench-internals \
//!     --profile release --bench parallel_shard_ingest_ab
//! ```

use std::fmt::Write as _;
use std::hint::black_box;

use frankensearch_core::bench_support::{PairedRatio, paired_median_ratio, print_bench_elf_sha256};
use frankensearch_quill::scribe::{
    ColumnarAccumulator, FlushDocumentInput, FlushMode, FlushSegmentInput, IndexedFieldValue,
    IndexedNumericValue, StoredFieldValue, flush_accumulator_with_mode,
};
use frankensearch_quill::{CURRENT_ENGINE_VERSION, DEFAULT_SCHEMA, EncodedSegment};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use xxhash_rust::xxh3::xxh3_64;

const ID_FIELD: u16 = 0;
const CONTENT_FIELD: u16 = 1;
const TITLE_FIELD: u16 = 2;
const METADATA_FIELD: u16 = 3;
const ORD_FIELD: u16 = 4;
const METADATA: &[u8] = b"{}";

/// Vocabulary is deliberately Heaps'-law-free here: this bench scores thread
/// scaling, not dictionary construction. See the corpus-hash memo — the QG
/// fixture pins 8,192 terms at every N, so a fixed vocabulary keeps this
/// comparable to the blessed shapes.
const VOCABULARY_SIZE: usize = 8_192;
const TOKENS_PER_DOCUMENT: usize = 350;

struct FixtureDocument {
    id: String,
    content: String,
    content_hash: u64,
}

/// One shard's arrival-ordered document set, exactly as `ShardRouter` would
/// have routed it.
struct ShardWork {
    segment_id: u64,
    /// Indices into the corpus, in the order they reach this shard.
    documents: Vec<usize>,
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).map_or(default, |value| {
        value
            .parse::<usize>()
            .unwrap_or_else(|_| panic!("{key} must be a positive integer, got {value}"))
    })
}

fn selected_shards() -> Vec<usize> {
    match std::env::var("QUILL_PSI_SHARDS").as_deref() {
        Ok("all") | Err(_) => {
            let cores = std::thread::available_parallelism().map_or(8, std::num::NonZero::get);
            vec![2, 4, 8, 16, cores]
        }
        Ok(other) => other
            .split(',')
            .map(|part| {
                part.trim()
                    .parse::<usize>()
                    .expect("QUILL_PSI_SHARDS must be `all` or a comma-separated integer list")
            })
            .collect(),
    }
}

fn build_corpus(document_count: usize) -> Vec<FixtureDocument> {
    let mut corpus = Vec::with_capacity(document_count);
    for document_index in 0..document_count {
        let id = format!("doc-{document_index:08}");
        let mut content = String::with_capacity(TOKENS_PER_DOCUMENT.saturating_mul(12));
        let mut state = (u64::try_from(document_index).expect("document index fits u64") + 1)
            .wrapping_mul(0x9e37_79b9_7f4a_7c15);
        for token_index in 0..TOKENS_PER_DOCUMENT {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let hot_rank = usize::try_from(state >> 32).expect("upper u32 fits usize");
            let term_index = (hot_rank
                .wrapping_add(token_index.wrapping_mul(17))
                .wrapping_add(document_index.wrapping_mul(31)))
                % VOCABULARY_SIZE;
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

/// Reproduce `ShardRouter::route_batch`: whole batches fan round-robin, so a
/// shard's documents stay contiguous per batch and its ordinals stay dense.
fn route(document_count: usize, shard_count: usize, batch_documents: usize) -> Vec<ShardWork> {
    let mut shards: Vec<ShardWork> = (0..shard_count)
        .map(|shard| ShardWork {
            segment_id: 0x0051_0000 | u64::try_from(shard).expect("shard index fits u64"),
            documents: Vec::new(),
        })
        .collect();
    let mut next_shard = 0_usize;
    let mut document_index = 0_usize;
    while document_index < document_count {
        let end = (document_index + batch_documents).min(document_count);
        shards[next_shard].documents.extend(document_index..end);
        next_shard = (next_shard + 1) % shard_count;
        document_index = end;
    }
    shards
}

/// Accumulate and seal one shard. This is the whole shared-nothing unit: it
/// touches only its own arena and its own output buffer.
fn build_shard(corpus: &[FixtureDocument], shard: &ShardWork, mode: FlushMode) -> EncodedSegment {
    let mut accumulator =
        ColumnarAccumulator::new(DEFAULT_SCHEMA).expect("default schema must be valid");
    let mut rows = Vec::with_capacity(shard.documents.len());
    for (position, &document_index) in shard.documents.iter().enumerate() {
        let document = &corpus[document_index];
        let doc_ord = u32::try_from(position).expect("shard fits one Quill lease");
        accumulator
            .add_document_with_values(
                doc_ord,
                &[
                    IndexedFieldValue::new(ID_FIELD, &document.id),
                    IndexedFieldValue::new(CONTENT_FIELD, &document.content),
                    IndexedFieldValue::new(TITLE_FIELD, ""),
                ],
                &[IndexedNumericValue::u64(ORD_FIELD, u64::from(doc_ord))],
                &[StoredFieldValue::new(METADATA_FIELD, METADATA)],
            )
            .expect("fixture document must accumulate");
        rows.push(FlushDocumentInput::new(
            doc_ord,
            &document.id,
            document.content_hash,
        ));
    }
    flush_accumulator_with_mode(
        &accumulator,
        FlushSegmentInput {
            segment_id: shard.segment_id,
            lease_docid_base: 0,
            created_unix_s: 1_700_000_000,
            engine_version: CURRENT_ENGINE_VERSION,
            documents: &rows,
        },
        mode,
    )
    .expect("shard seal must succeed")
}

/// Baseline — today's engine: every shard built inline on the caller's thread,
/// sealed with the single-threaded radix that `index.rs` actually passes.
fn build_serial(
    corpus: &[FixtureDocument],
    shards: &[ShardWork],
    mode: FlushMode,
) -> Vec<EncodedSegment> {
    shards
        .iter()
        .map(|shard| build_shard(corpus, shard, mode))
        .collect()
}

/// Shard-level fan-out: shared-nothing, one shard per core.
fn build_parallel(
    corpus: &[FixtureDocument],
    shards: &[ShardWork],
    pool: &ThreadPool,
    mode: FlushMode,
) -> Vec<EncodedSegment> {
    pool.install(|| {
        shards
            .par_iter()
            .map(|shard| build_shard(corpus, shard, mode))
            .collect()
    })
}

fn total_len(segments: &[EncodedSegment]) -> u64 {
    segments.iter().map(EncodedSegment::file_len).sum()
}

/// Process-wide CPU time (user+system) in seconds, from `/proc/self/stat`.
///
/// A parallelism lever must publish CPU/wall per arm: on a shared host the
/// wall ratio alone cannot distinguish "we parallelised" from "a peer stopped
/// competing for cores". CPU/wall is the achieved parallelism and is stable
/// under neighbour load even when wall time is not.
fn cpu_seconds() -> f64 {
    let stat = std::fs::read_to_string("/proc/self/stat").expect("/proc/self/stat must be readable");
    // Fields after the parenthesised comm: utime is 14th, stime 15th (1-based
    // over the whole line), so index 11 and 12 after the closing paren.
    let tail = &stat[stat.rfind(')').expect("stat always has a comm field") + 2..];
    let fields = tail.split_ascii_whitespace().collect::<Vec<_>>();
    let utime = fields[11].parse::<u64>().expect("utime is an integer");
    let stime = fields[12].parse::<u64>().expect("stime is an integer");
    let ticks_per_second = 100.0_f64; // USER_HZ is 100 on every supported target
    (utime + stime) as f64 / ticks_per_second
}

/// Wall and CPU seconds for `iterations` back-to-back runs of one arm.
fn measure_arm(iterations: usize, mut run: impl FnMut()) -> (f64, f64) {
    let cpu_start = cpu_seconds();
    let wall_start = std::time::Instant::now();
    for _ in 0..iterations {
        run();
    }
    let wall = wall_start.elapsed().as_secs_f64();
    (wall, cpu_seconds() - cpu_start)
}

fn print_ratio(kind: &str, documents: usize, shards: usize, ratio: PairedRatio) {
    eprintln!(
        "[{kind}] docs={documents} shards={shards}: parallel/serial median {:.4} \
         median_ci95 [{:.4}, {:.4}] p5 {:.4} p95 {:.4} ({} rounds)",
        ratio.median,
        ratio.median_ci95_low,
        ratio.median_ci95_high,
        ratio.p5,
        ratio.p95,
        ratio.rounds
    );
}

fn run_cell(
    corpus: &[FixtureDocument],
    document_count: usize,
    shard_count: usize,
    batch_documents: usize,
    rounds: usize,
) {
    let shards = route(document_count, shard_count, batch_documents);
    let pool = ThreadPoolBuilder::new()
        .num_threads(shard_count)
        .thread_name(move |index| format!("quill-psi-{shard_count}s-{index}"))
        .build()
        .expect("fixed Rayon pool must build");

    // Parity gate: neither lever may change a single output byte. This also
    // re-proves in-bench that the parallel radix seal is byte-identical to the
    // single-threaded one production currently forces.
    let baseline = build_serial(corpus, &shards, FlushMode::Scalar);
    let seal_parallel = build_serial(corpus, &shards, FlushMode::Automatic);
    let shard_parallel = build_parallel(corpus, &shards, &pool, FlushMode::Scalar);
    for (index, other) in seal_parallel.iter().enumerate() {
        assert_eq!(
            baseline[index].as_bytes(),
            other.as_bytes(),
            "shard {index}: Automatic seal must match the Scalar seal byte-for-byte"
        );
    }
    for (index, other) in shard_parallel.iter().enumerate() {
        assert_eq!(
            baseline[index].as_bytes(),
            other.as_bytes(),
            "shard {index}: shard fan-out must not change sealed bytes"
        );
    }
    let sealed_bytes = total_len(&baseline);
    drop((baseline, seal_parallel, shard_parallel));

    let base = || {
        black_box(total_len(&build_serial(
            black_box(corpus),
            black_box(&shards),
            FlushMode::Scalar,
        )));
    };
    let null = paired_median_ratio(rounds, 1, base, base);
    // Lever 1a: keep the serial shard walk, but stop forcing the single-threaded
    // seal that `index.rs` passes today.
    let seal_lever = paired_median_ratio(rounds, 1, base, || {
        black_box(total_len(&build_serial(
            black_box(corpus),
            black_box(&shards),
            FlushMode::Automatic,
        )));
    });
    // Lever 1b: shared-nothing shard fan-out, seal mode unchanged.
    let shard_lever = paired_median_ratio(rounds, 1, base, || {
        black_box(total_len(&build_parallel(
            black_box(corpus),
            black_box(&shards),
            &pool,
            FlushMode::Scalar,
        )));
    });
    print_ratio("null", document_count, shard_count, null);
    print_ratio("seal-automatic", document_count, shard_count, seal_lever);
    print_ratio("shard-fanout", document_count, shard_count, shard_lever);
    let lever = shard_lever;

    // CPU/wall per arm. Serial must sit at ~1.0x; the lever's achieved
    // parallelism is its CPU/wall, and its CPU total should be close to
    // serial's if the split is not doing redundant work.
    let (serial_wall, serial_cpu) = measure_arm(2, || {
        black_box(total_len(&build_serial(
            black_box(corpus),
            black_box(&shards),
            FlushMode::Scalar,
        )));
    });
    let (parallel_wall, parallel_cpu) = measure_arm(2, || {
        black_box(total_len(&build_parallel(
            black_box(corpus),
            black_box(&shards),
            &pool,
            FlushMode::Scalar,
        )));
    });
    eprintln!(
        "[cpuwall] docs={document_count} shards={shard_count} \
         serial wall {serial_wall:.3}s cpu {serial_cpu:.3}s ({:.2}x) | \
         parallel wall {parallel_wall:.3}s cpu {parallel_cpu:.3}s ({:.2}x) | \
         cpu_overhead {:.3}x",
        serial_cpu / serial_wall.max(f64::MIN_POSITIVE),
        parallel_cpu / parallel_wall.max(f64::MIN_POSITIVE),
        parallel_cpu / serial_cpu.max(f64::MIN_POSITIVE),
    );

    let decision = if !lever.decidable_against(&null) || (0.97..=1.03).contains(&lever.median) {
        "NOISE"
    } else if lever.median < 1.0 {
        "PARALLEL_WINS"
    } else {
        "SERIAL_WINS"
    };
    let speedup = 1.0 / lever.median;
    eprintln!(
        "[decision] docs={document_count} shards={shard_count} \
         sealed_bytes={sealed_bytes} speedup={speedup:.2}x {decision}"
    );
}

fn main() {
    let _identity =
        print_bench_elf_sha256().expect("hash the executing parallel-shard-ingest benchmark");
    let document_count = env_usize("QUILL_PSI_DOCS", 50_000);
    let batch_documents = env_usize("QUILL_PSI_BATCH", 250);
    let rounds = env_usize("QUILL_PSI_ROUNDS", 7);
    let shard_counts = selected_shards();
    eprintln!(
        "[config] docs={document_count} batch={batch_documents} rounds={rounds} \
         shards={shard_counts:?} tokens_per_doc={TOKENS_PER_DOCUMENT} \
         vocabulary={VOCABULARY_SIZE}"
    );
    // A parallelism ratio measured on a busy host understates the lever, so
    // the load average is part of the record, not a footnote.
    if let Ok(loadavg) = std::fs::read_to_string("/proc/loadavg") {
        eprintln!(
            "[host] cores={} loadavg={}",
            std::thread::available_parallelism().map_or(0, std::num::NonZero::get),
            loadavg.trim()
        );
    }
    let corpus = build_corpus(document_count);
    for &shard_count in &shard_counts {
        run_cell(
            &corpus,
            document_count,
            shard_count,
            batch_documents,
            rounds,
        );
    }
}
