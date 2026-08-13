//! Shared-nothing parallel segment build and radix execution controls.
//!
//! Quill partitions ingest into `W` shard accumulators and builds independent
//! budget-crossing shards through Rayon. This benchmark retains the former
//! scalar paths as explicit controls so the two production levers remain
//! measurable against byte-identical work.
//!
//! Three configurations are scored against one A/A null, all producing
//! byte-identical sealed segments from byte-identical shard assignment:
//!
//! - **baseline** — the former serial shard walk and scalar radix reference.
//! - **seal-automatic** — the same serial shard walk with the automatic stable
//!   radix used by production parallel shard builds. This isolates the radix
//!   lever from shard fan-out.
//! - **shard-fanout** — shared-nothing shard fan-out with scalar radix, which
//!   isolates the outer fan-out from the radix lever.
//!
//! Parity is asserted per shard before any timing.
//!
//! `QUILL_PSI_LEVER=stored-meta` instead compares production's borrowed
//! STOREDMETA emission and direct final assembly with the former full
//! intermediate section image plus canonical section copier. Both arms retain
//! automatic radix and the same outer shard fan-out.
//!
//! `QUILL_PSI_ROUNDS` must be **at least 10**: `PairedRatio::is_admissible_null`
//! requires `rounds >= 10`, and `decidable_against` returns false for an
//! inadmissible null, so every cell prints `NOISE` at 9 rounds no matter how
//! large the effect is.
//!
//! ```bash
//! QUILL_PSI_DOCS=50000 QUILL_PSI_SHARDS=all QUILL_PSI_ROUNDS=11 \
//!   cargo bench -p frankensearch-quill --features bench-internals \
//!     --profile release --bench parallel_shard_ingest_ab
//! ```

use std::fmt::Write as _;
use std::hint::black_box;

use frankensearch_core::bench_support::{PairedRatio, paired_median_ratio, print_bench_elf_sha256};
use frankensearch_quill::scribe::{
    ColumnarAccumulator, FlushDocumentInput, FlushMode, FlushSegmentInput, IndexedFieldValue,
    IndexedNumericValue, StoredFieldValue, flush_accumulator_with_mode,
    flush_accumulator_with_staged_stored_meta,
};
use frankensearch_quill::{CURRENT_ENGINE_VERSION, DEFAULT_SCHEMA, EncodedSegment, SectionKind};
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

#[derive(Clone, Copy)]
enum StoredMetaMode {
    Borrowed,
    Staged,
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
fn build_shard(
    corpus: &[FixtureDocument],
    shard: &ShardWork,
    mode: FlushMode,
    stored_meta_mode: StoredMetaMode,
) -> EncodedSegment {
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
    let input = FlushSegmentInput {
        segment_id: shard.segment_id,
        lease_docid_base: 0,
        created_unix_s: 1_700_000_000,
        engine_version: CURRENT_ENGINE_VERSION,
        documents: &rows,
    };
    match stored_meta_mode {
        StoredMetaMode::Borrowed => flush_accumulator_with_mode(&accumulator, input, mode),
        StoredMetaMode::Staged => {
            flush_accumulator_with_staged_stored_meta(&accumulator, input, mode)
        }
    }
    .expect("shard seal must succeed")
}

/// Former shipping baseline: build every shard inline with scalar radix.
fn build_serial(
    corpus: &[FixtureDocument],
    shards: &[ShardWork],
    mode: FlushMode,
    stored_meta_mode: StoredMetaMode,
) -> Vec<EncodedSegment> {
    shards
        .iter()
        .map(|shard| build_shard(corpus, shard, mode, stored_meta_mode))
        .collect()
}

/// Shard-level fan-out: shared-nothing, one shard per core.
fn build_parallel(
    corpus: &[FixtureDocument],
    shards: &[ShardWork],
    pool: &ThreadPool,
    mode: FlushMode,
    stored_meta_mode: StoredMetaMode,
) -> Vec<EncodedSegment> {
    pool.install(|| {
        shards
            .par_iter()
            .map(|shard| build_shard(corpus, shard, mode, stored_meta_mode))
            .collect()
    })
}

fn total_len(segments: &[EncodedSegment]) -> u64 {
    segments.iter().map(EncodedSegment::file_len).sum()
}

fn stored_meta_len(segments: &[EncodedSegment]) -> u64 {
    segments
        .iter()
        .flat_map(EncodedSegment::section_entries)
        .filter(|entry| entry.kind == SectionKind::STOREDMETA)
        .map(|entry| entry.len)
        .sum()
}

/// Process-wide CPU time (user+system) in seconds, from `/proc/self/stat`.
///
/// A parallelism lever must publish CPU/wall per arm: on a shared host the
/// wall ratio alone cannot distinguish "we parallelised" from "a peer stopped
/// competing for cores". CPU/wall is the achieved parallelism and is stable
/// under neighbour load even when wall time is not.
fn cpu_seconds() -> f64 {
    let stat =
        std::fs::read_to_string("/proc/self/stat").expect("/proc/self/stat must be readable");
    // Fields after the parenthesised comm: utime is 14th, stime 15th (1-based
    // over the whole line), so index 11 and 12 after the closing paren.
    let tail = &stat[stat.rfind(')').expect("stat always has a comm field") + 2..];
    let fields = tail.split_ascii_whitespace().collect::<Vec<_>>();
    let utime = fields[11].parse::<u64>().expect("utime is an integer");
    let stime = fields[12].parse::<u64>().expect("stime is an integer");
    let ticks_per_second = 100.0_f64; // USER_HZ is 100 on every supported target
    (utime + stime) as f64 / ticks_per_second
}

fn peak_rss_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let kib = status.lines().find_map(|line| {
        let value = line.strip_prefix("VmHWM:")?.trim();
        value.split_ascii_whitespace().next()?.parse::<u64>().ok()
    })?;
    kib.checked_mul(1024)
}

fn measure_peak_rss_child(
    arm: &str,
    document_count: usize,
    shard_count: usize,
    batch_documents: usize,
) -> u64 {
    let executable = std::env::current_exe().expect("resolve current benchmark executable");
    let output = std::process::Command::new(executable)
        .env("QUILL_PSI_CHILD_ARM", arm)
        .env("QUILL_PSI_DOCS", document_count.to_string())
        .env("QUILL_PSI_SHARDS", shard_count.to_string())
        .env("QUILL_PSI_BATCH", batch_documents.to_string())
        .output()
        .expect("launch fresh peak-RSS child");
    assert!(
        output.status.success(),
        "fresh {arm} peak-RSS child failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("peak-RSS child writes UTF-8");
    stdout
        .lines()
        .find_map(|line| {
            line.strip_prefix("peak_rss_bytes=")?
                .split_ascii_whitespace()
                .next()?
                .parse::<u64>()
                .ok()
        })
        .expect("peak-RSS child must publish a numeric VmHWM")
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

fn print_stored_meta_ratio(kind: &str, documents: usize, shards: usize, ratio: PairedRatio) {
    eprintln!(
        "[{kind}] docs={documents} shards={shards}: borrowed/staged median {:.4} \
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
    // re-proves in-bench that production's automatic radix seal is
    // byte-identical to the former scalar reference.
    let baseline = build_serial(corpus, &shards, FlushMode::Scalar, StoredMetaMode::Borrowed);
    let seal_parallel = build_serial(
        corpus,
        &shards,
        FlushMode::Automatic,
        StoredMetaMode::Borrowed,
    );
    let shard_parallel = build_parallel(
        corpus,
        &shards,
        &pool,
        FlushMode::Scalar,
        StoredMetaMode::Borrowed,
    );
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
            StoredMetaMode::Borrowed,
        )));
    };
    let null = paired_median_ratio(rounds, 1, base, base);
    // Lever 1a: keep the serial shard walk and isolate automatic radix from the
    // production outer shard fan-out.
    let seal_lever = paired_median_ratio(rounds, 1, base, || {
        black_box(total_len(&build_serial(
            black_box(corpus),
            black_box(&shards),
            FlushMode::Automatic,
            StoredMetaMode::Borrowed,
        )));
    });
    // Lever 1b: shared-nothing shard fan-out, seal mode unchanged.
    let shard_lever = paired_median_ratio(rounds, 1, base, || {
        black_box(total_len(&build_parallel(
            black_box(corpus),
            black_box(&shards),
            &pool,
            FlushMode::Scalar,
            StoredMetaMode::Borrowed,
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
            StoredMetaMode::Borrowed,
        )));
    });
    let (parallel_wall, parallel_cpu) = measure_arm(2, || {
        black_box(total_len(&build_parallel(
            black_box(corpus),
            black_box(&shards),
            &pool,
            FlushMode::Scalar,
            StoredMetaMode::Borrowed,
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

fn run_stored_meta_cell(
    corpus: &[FixtureDocument],
    document_count: usize,
    shard_count: usize,
    batch_documents: usize,
    rounds: usize,
) {
    let shards = route(document_count, shard_count, batch_documents);
    let pool = ThreadPoolBuilder::new()
        .num_threads(shard_count)
        .thread_name(move |index| format!("quill-stored-{shard_count}s-{index}"))
        .build()
        .expect("fixed Rayon pool must build");

    let staged = build_parallel(
        corpus,
        &shards,
        &pool,
        FlushMode::Automatic,
        StoredMetaMode::Staged,
    );
    let borrowed = build_parallel(
        corpus,
        &shards,
        &pool,
        FlushMode::Automatic,
        StoredMetaMode::Borrowed,
    );
    for (index, (staged_segment, borrowed_segment)) in staged.iter().zip(&borrowed).enumerate() {
        assert_eq!(
            borrowed_segment.as_bytes(),
            staged_segment.as_bytes(),
            "shard {index}: borrowed STOREDMETA must match staged output byte-for-byte"
        );
        eprintln!(
            "[parity] shard={index} bytes={} xxh3={:016x}",
            borrowed_segment.file_len(),
            xxh3_64(borrowed_segment.as_bytes())
        );
    }
    let sealed_bytes = total_len(&borrowed);
    let stored_meta_bytes = stored_meta_len(&borrowed);
    drop((staged, borrowed));

    let staged_arm = || {
        black_box(total_len(&build_parallel(
            black_box(corpus),
            black_box(&shards),
            &pool,
            FlushMode::Automatic,
            StoredMetaMode::Staged,
        )));
    };
    let borrowed_arm = || {
        black_box(total_len(&build_parallel(
            black_box(corpus),
            black_box(&shards),
            &pool,
            FlushMode::Automatic,
            StoredMetaMode::Borrowed,
        )));
    };
    let null = paired_median_ratio(rounds, 1, staged_arm, staged_arm);
    let effect = paired_median_ratio(rounds, 1, staged_arm, borrowed_arm);
    print_stored_meta_ratio("staged-null", document_count, shard_count, null);
    print_stored_meta_ratio("borrowed-stored-meta", document_count, shard_count, effect);

    // Reverse the second CPU ordering so a thermal/host trend cannot
    // systematically favour one arm. Resource evidence is deliberately a
    // veto as well as telemetry: a wall-only win is not accepted when total
    // CPU fails to improve by a clear 3%.
    let (staged_wall_first, staged_cpu_first) = measure_arm(1, staged_arm);
    let (borrowed_wall_first, borrowed_cpu_first) = measure_arm(1, borrowed_arm);
    let (borrowed_wall_second, borrowed_cpu_second) = measure_arm(1, borrowed_arm);
    let (staged_wall_second, staged_cpu_second) = measure_arm(1, staged_arm);
    let staged_wall = staged_wall_first + staged_wall_second;
    let staged_cpu = staged_cpu_first + staged_cpu_second;
    let borrowed_wall = borrowed_wall_first + borrowed_wall_second;
    let borrowed_cpu = borrowed_cpu_first + borrowed_cpu_second;

    // Fresh-process VmHWM is lifetime-max state, so each observation gets its
    // own child. Reverse the second pair and require complete, separated
    // ranges; launch/parse failure is fatal in `measure_peak_rss_child`.
    let staged_rss_first =
        measure_peak_rss_child("staged", document_count, shard_count, batch_documents);
    let borrowed_rss_first =
        measure_peak_rss_child("borrowed", document_count, shard_count, batch_documents);
    let borrowed_rss_second =
        measure_peak_rss_child("borrowed", document_count, shard_count, batch_documents);
    let staged_rss_second =
        measure_peak_rss_child("staged", document_count, shard_count, batch_documents);
    let staged_rss_min = staged_rss_first.min(staged_rss_second);
    let staged_rss_max = staged_rss_first.max(staged_rss_second);
    let borrowed_rss_min = borrowed_rss_first.min(borrowed_rss_second);
    let borrowed_rss_max = borrowed_rss_first.max(borrowed_rss_second);
    eprintln!(
        "[stored-meta-cpuwall] docs={document_count} shards={shard_count} \
         staged wall {staged_wall:.3}s cpu {staged_cpu:.3}s ({:.2}x) | \
         borrowed wall {borrowed_wall:.3}s cpu {borrowed_cpu:.3}s ({:.2}x) | \
         borrowed_cpu/staged_cpu {:.3}x",
        staged_cpu / staged_wall.max(f64::MIN_POSITIVE),
        borrowed_cpu / borrowed_wall.max(f64::MIN_POSITIVE),
        borrowed_cpu / staged_cpu.max(f64::MIN_POSITIVE),
    );
    eprintln!(
        "[stored-meta-rss] docs={document_count} shards={shard_count} \
         staged_bytes=[{staged_rss_min},{staged_rss_max}] \
         borrowed_bytes=[{borrowed_rss_min},{borrowed_rss_max}]",
    );

    let resources_improve = borrowed_cpu < staged_cpu * 0.97 && borrowed_rss_max < staged_rss_min;
    let resource_regression = borrowed_cpu > staged_cpu * 1.03 || borrowed_rss_min > staged_rss_max;
    let decision = if !null.is_admissible_null() {
        "NULL_INVALID"
    } else if !effect.decidable_against(&null) || (0.97..=1.03).contains(&effect.median) {
        "NO_EFFECT"
    } else if effect.median < 0.97 && resource_regression {
        "SPEED_ONLY_RESOURCE_REGRESSION"
    } else if effect.median < 0.97 && resources_improve {
        "BORROWED_WALL_AND_RESOURCE_WIN"
    } else if effect.median < 0.97 {
        "WALL_WIN_RESOURCE_UNPROVEN"
    } else {
        "BORROWED_WALL_LOSS"
    };
    eprintln!(
        "[stored-meta-decision] docs={document_count} shards={shard_count} \
         sealed_bytes={sealed_bytes} stored_meta_bytes={stored_meta_bytes} \
         borrowed_speedup={:.3}x {decision}",
        1.0 / effect.median
    );
}

fn run_peak_rss_child(
    arm: &str,
    document_count: usize,
    shard_count: usize,
    batch_documents: usize,
) {
    let corpus = build_corpus(document_count);
    let shards = route(document_count, shard_count, batch_documents);
    let pool = ThreadPoolBuilder::new()
        .num_threads(shard_count)
        .build()
        .expect("fixed Rayon pool must build");
    let mode = match arm {
        "staged" => StoredMetaMode::Staged,
        "borrowed" => StoredMetaMode::Borrowed,
        _ => panic!("QUILL_PSI_CHILD_ARM must be `staged` or `borrowed`, got {arm}"),
    };
    let sealed_bytes = black_box(total_len(&build_parallel(
        &corpus,
        &shards,
        &pool,
        FlushMode::Automatic,
        mode,
    )));
    let peak_rss = peak_rss_bytes().expect("Linux /proc must publish nonzero VmHWM");
    assert!(peak_rss > 0, "Linux VmHWM must be nonzero");
    println!("peak_rss_bytes={peak_rss} sealed_bytes={sealed_bytes}");
}

fn main() {
    let _identity =
        print_bench_elf_sha256().expect("hash the executing parallel-shard-ingest benchmark");
    let document_count = env_usize("QUILL_PSI_DOCS", 50_000);
    let batch_documents = env_usize("QUILL_PSI_BATCH", 250);
    // Default must clear `is_admissible_null`'s `rounds >= 10` floor, or every
    // cell reports NOISE regardless of effect size.
    let rounds = env_usize("QUILL_PSI_ROUNDS", 11);
    if let Ok(arm) = std::env::var("QUILL_PSI_CHILD_ARM") {
        let shard_counts = selected_shards();
        assert_eq!(
            shard_counts.len(),
            1,
            "peak-RSS child requires exactly one shard count"
        );
        run_peak_rss_child(&arm, document_count, shard_counts[0], batch_documents);
        return;
    }
    let lever = std::env::var("QUILL_PSI_LEVER").unwrap_or_else(|_| String::from("parallel"));
    assert!(
        matches!(lever.as_str(), "parallel" | "stored-meta"),
        "QUILL_PSI_LEVER must be `parallel` or `stored-meta`, got {lever}"
    );
    let shard_counts = selected_shards();
    // Fail loudly rather than emitting an unfalsifiable NOISE verdict.
    assert!(
        rounds >= 10,
        "QUILL_PSI_ROUNDS={rounds} is below the rounds>=10 floor that \
         PairedRatio::is_admissible_null enforces; every cell would print NOISE \
         regardless of effect size"
    );
    eprintln!(
        "[config] lever={lever} docs={document_count} batch={batch_documents} rounds={rounds} \
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
        if lever == "stored-meta" {
            run_stored_meta_cell(
                &corpus,
                document_count,
                shard_count,
                batch_documents,
                rounds,
            );
        } else {
            run_cell(
                &corpus,
                document_count,
                shard_count,
                batch_documents,
                rounds,
            );
        }
    }
}
