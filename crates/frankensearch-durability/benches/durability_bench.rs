//! Durability benchmarks: encode/decode throughput, repair latency, xxh3 verify.
//!
//! Run with:
//!   cargo bench -p frankensearch-durability

use std::hint::black_box;
use std::io::{BufRead, BufReader, Write};
use std::sync::Arc;
use std::time::{Duration, Instant};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_durability::config::DurabilityConfig;
use frankensearch_durability::file_protector::FileProtector;
use frankensearch_durability::metrics::DurabilityMetrics;
use fsqlite_core::raptorq_integration::{CodecDecodeResult, CodecEncodeResult, SymbolCodec};
use fsqlite_types::cx::Cx;
use xxhash_rust::xxh3::xxh3_64;

/// Mock codec that mimics `RaptorQ` behavior for benchmarking the pipeline
/// overhead (serialization, CRC, file I/O) without the actual erasure coding.
#[derive(Debug)]
struct BenchCodec;

impl SymbolCodec for BenchCodec {
    fn encode(
        &self,
        _cx: &Cx,
        source_data: &[u8],
        symbol_size: u32,
        _repair_overhead: f64,
    ) -> fsqlite_error::Result<CodecEncodeResult> {
        let symbol_size_usize = usize::try_from(symbol_size).unwrap_or(1);
        let mut source_symbols = Vec::new();
        let mut repair_symbols = Vec::new();

        let mut esi: u32 = 0;
        for chunk in source_data.chunks(symbol_size_usize) {
            let mut data = chunk.to_vec();
            if data.len() < symbol_size_usize {
                data.resize(symbol_size_usize, 0);
            }
            source_symbols.push((esi, data.clone()));
            repair_symbols.push((esi + 1_000_000, data));
            esi = esi.saturating_add(1);
        }

        Ok(CodecEncodeResult {
            source_symbols,
            repair_symbols,
            k_source: esi,
        })
    }

    fn decode(
        &self,
        _cx: &Cx,
        symbols: &[(u32, Vec<u8>)],
        k_source: u32,
        _symbol_size: u32,
    ) -> fsqlite_error::Result<CodecDecodeResult> {
        let mut reconstructed = Vec::new();
        for source_esi in 0..k_source {
            let primary = symbols
                .iter()
                .find(|(esi, _)| *esi == source_esi)
                .map(|(_, data)| data.clone());
            let fallback = symbols
                .iter()
                .find(|(esi, _)| *esi == source_esi + 1_000_000)
                .map(|(_, data)| data.clone());

            match primary.or(fallback) {
                Some(data) => reconstructed.extend_from_slice(&data),
                None => {
                    return Ok(CodecDecodeResult::Failure {
                        reason:
                            fsqlite_core::raptorq_integration::DecodeFailureReason::InsufficientSymbols,
                        symbols_received: u32::try_from(symbols.len()).unwrap_or(u32::MAX),
                        k_required: k_source,
                    });
                }
            }
        }

        Ok(CodecDecodeResult::Success {
            data: reconstructed,
            symbols_used: k_source,
            peeled_count: k_source,
            inactivated_count: 0,
        })
    }
}

fn test_config() -> DurabilityConfig {
    DurabilityConfig {
        symbol_size: 4096,
        repair_overhead: 1.20,
        ..DurabilityConfig::default()
    }
}

fn make_protector() -> FileProtector {
    let metrics = Arc::new(DurabilityMetrics::default());
    FileProtector::new_with_metrics(Arc::new(BenchCodec), test_config(), metrics)
        .expect("protector")
}

// --- Repair-log rotation allocation A/B ---

const REPAIR_LOG_LINE: &[u8] = br#"{"timestamp":"2026-07-14T00:00:00Z","path":"/data/indexes/tenant-0042/shard-0007.fsvi","corrupted":true,"repair_succeeded":true,"bytes_written":1048576,"source_crc32_expected":305419896,"source_crc32_after":305419896,"repair_time_ms":17}
"#;
const ROTATION_PAIRED_ROUNDS: usize = 21;
const ROTATION_INNER_APPENDS: usize = 16;

#[derive(Clone, Copy)]
enum RotationArm {
    ReadToString,
    ReusedLineBuffer,
}

#[derive(Clone, Copy)]
struct RatioSummary {
    median: f64,
    p5: f64,
    p95: f64,
}

impl RatioSummary {
    /// Largest `|median - 1|` an A/A control may show and still be admissible.
    /// Mirrors `frankensearch_core::bench_support::NULL_MEDIAN_TOLERANCE`.
    const NULL_MEDIAN_TOLERANCE: f64 = 0.02;

    /// A/A control admissibility: gate the null's accuracy, never its precision.
    ///
    /// Requiring `p5 <= 1.0 <= p95` vetoed a null for being *tight*: the
    /// narrower its spread, the likelier it excluded 1.0 — however small the
    /// bias and however large the effect (`bd-pjh09`). The spread still sets
    /// the effect floor below; only the median says whether the sampler itself
    /// is trustworthy.
    fn null_is_unbiased(self) -> bool {
        (self.median - 1.0).abs() <= Self::NULL_MEDIAN_TOLERANCE
    }
}

fn should_rotate_read_to_string(
    log_path: &std::path::Path,
    max_entries: usize,
) -> std::io::Result<bool> {
    if !log_path.exists() {
        return Ok(false);
    }
    let contents = std::fs::read_to_string(log_path)?;
    Ok(contents.lines().count() >= max_entries)
}

fn should_rotate_reused_line_buffer(
    log_path: &std::path::Path,
    max_entries: usize,
) -> std::io::Result<bool> {
    if !log_path.exists() {
        return Ok(false);
    }
    let file = std::fs::File::open(log_path)?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    let mut line_count = 0_usize;
    while reader.read_line(&mut line)? != 0 {
        line_count += 1;
        line.clear();
    }
    Ok(line_count >= max_entries)
}

fn rotation_decision(
    arm: RotationArm,
    log_path: &std::path::Path,
    max_entries: usize,
) -> std::io::Result<bool> {
    match arm {
        RotationArm::ReadToString => should_rotate_read_to_string(log_path, max_entries),
        RotationArm::ReusedLineBuffer => should_rotate_reused_line_buffer(log_path, max_entries),
    }
}

fn seed_repair_log(path: &std::path::Path, entries: usize) -> u64 {
    let mut file = std::fs::File::create(path).expect("create seeded repair log");
    for _ in 0..entries {
        file.write_all(REPAIR_LOG_LINE)
            .expect("write seeded repair event");
    }
    file.metadata().expect("seeded log metadata").len()
}

fn append_repair_event(
    arm: RotationArm,
    log_path: &std::path::Path,
    rotated_path: &std::path::Path,
    max_entries: usize,
) {
    if rotation_decision(arm, log_path, max_entries).expect("rotation decision") {
        let _ = std::fs::rename(log_path, rotated_path);
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .expect("open repair log for append");
    file.write_all(REPAIR_LOG_LINE)
        .expect("append repair event");
}

fn prove_repair_log_parity() {
    for &(entries, max_entries) in &[(999, 1_000), (1_000, 1_000)] {
        let reference_dir = tempfile_dir("repair-log-parity-reference");
        let candidate_dir = tempfile_dir("repair-log-parity-candidate");
        let reference_log = reference_dir.join("repair-events.jsonl");
        let candidate_log = candidate_dir.join("repair-events.jsonl");
        let reference_rotated = reference_dir.join("repair-events.1.jsonl");
        let candidate_rotated = candidate_dir.join("repair-events.1.jsonl");
        seed_repair_log(&reference_log, entries);
        seed_repair_log(&candidate_log, entries);

        append_repair_event(
            RotationArm::ReadToString,
            &reference_log,
            &reference_rotated,
            max_entries,
        );
        append_repair_event(
            RotationArm::ReusedLineBuffer,
            &candidate_log,
            &candidate_rotated,
            max_entries,
        );

        assert_eq!(
            std::fs::read(&candidate_log).expect("read candidate active log"),
            std::fs::read(&reference_log).expect("read reference active log"),
            "active log differs at entries={entries} max_entries={max_entries}"
        );
        assert_eq!(
            candidate_rotated.exists(),
            reference_rotated.exists(),
            "rotation decision differs at entries={entries} max_entries={max_entries}"
        );
        if reference_rotated.exists() {
            assert_eq!(
                std::fs::read(&candidate_rotated).expect("read candidate rotated log"),
                std::fs::read(&reference_rotated).expect("read reference rotated log"),
                "rotated log differs at entries={entries} max_entries={max_entries}"
            );
        }
    }
}

fn timed_append_batch(arm: RotationArm, log_path: &std::path::Path, base_len: u64) -> Duration {
    std::fs::OpenOptions::new()
        .write(true)
        .open(log_path)
        .expect("open repair log for reset")
        .set_len(base_len)
        .expect("reset repair log before timing");

    let start = Instant::now();
    for _ in 0..ROTATION_INNER_APPENDS {
        let rotate = rotation_decision(arm, log_path, 2_048).expect("timed rotation decision");
        assert!(!rotate, "timed fixture must remain below rotation limit");
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(log_path)
            .expect("open timed repair log");
        file.write_all(REPAIR_LOG_LINE)
            .expect("append timed repair event");
    }
    let elapsed = start.elapsed();

    std::fs::OpenOptions::new()
        .write(true)
        .open(log_path)
        .expect("open repair log after timing")
        .set_len(base_len)
        .expect("reset repair log after timing");
    elapsed
}

fn paired_ratio(
    arm_a: RotationArm,
    arm_b: RotationArm,
    path_a: &std::path::Path,
    path_b: &std::path::Path,
    base_len: u64,
) -> RatioSummary {
    let mut ratios = Vec::with_capacity(ROTATION_PAIRED_ROUNDS);
    for _ in 0..ROTATION_PAIRED_ROUNDS {
        let a_ab = timed_append_batch(arm_a, path_a, base_len).as_secs_f64();
        let b_ab = timed_append_batch(arm_b, path_b, base_len).as_secs_f64();
        let b_ba = timed_append_batch(arm_b, path_b, base_len).as_secs_f64();
        let a_ba = timed_append_batch(arm_a, path_a, base_len).as_secs_f64();
        ratios.push(((b_ab / a_ab) * (b_ba / a_ba)).sqrt());
    }
    ratios.sort_unstable_by(f64::total_cmp);
    let percentile = |pct: usize| ratios[((ratios.len() - 1) * pct + 50) / 100];
    RatioSummary {
        median: percentile(50),
        p5: percentile(5),
        p95: percentile(95),
    }
}

fn bench_repair_log_rotation_ab(c: &mut Criterion) {
    prove_repair_log_parity();

    let dir = tempfile_dir("repair-log-rotation-ab");
    let null_a = dir.join("null-a.jsonl");
    let null_b = dir.join("null-b.jsonl");
    let reference = dir.join("reference.jsonl");
    let candidate = dir.join("candidate.jsonl");
    let base_len = seed_repair_log(&null_a, 999);
    seed_repair_log(&null_b, 999);
    seed_repair_log(&reference, 999);
    seed_repair_log(&candidate, 999);

    for _ in 0..3 {
        black_box(timed_append_batch(
            RotationArm::ReadToString,
            &reference,
            base_len,
        ));
        black_box(timed_append_batch(
            RotationArm::ReusedLineBuffer,
            &candidate,
            base_len,
        ));
    }

    let null = paired_ratio(
        RotationArm::ReadToString,
        RotationArm::ReadToString,
        &null_a,
        &null_b,
        base_len,
    );
    let candidate_ratio = paired_ratio(
        RotationArm::ReadToString,
        RotationArm::ReusedLineBuffer,
        &reference,
        &candidate,
        base_len,
    );
    let gate_pass = null.null_is_unbiased() && candidate_ratio.median < null.p5;
    eprintln!(
        "[parity] repair_log_append active_and_rotated_bytes=IDENTICAL shapes=999/1000,1000/1000"
    );
    eprintln!(
        "[paired] comparison=null_read_to_string median={:.6} p5={:.6} p95={:.6} round_pairs={ROTATION_PAIRED_ROUNDS}",
        null.median, null.p5, null.p95
    );
    eprintln!(
        "[paired] comparison=reused_line_buffer_vs_read_to_string median={:.6} p5={:.6} p95={:.6} round_pairs={ROTATION_PAIRED_ROUNDS}",
        candidate_ratio.median, candidate_ratio.p5, candidate_ratio.p95
    );
    eprintln!(
        "[gate] decision={} speedup={:.6}x null_is_unbiased={} candidate_median_below_null_p5={}",
        if gate_pass { "KEEP" } else { "HOLD" },
        1.0 / candidate_ratio.median,
        null.null_is_unbiased(),
        candidate_ratio.median < null.p5
    );

    let mut group = c.benchmark_group("repair_log_rotation_check");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.bench_function("read_to_string/999", |b| {
        b.iter(|| {
            black_box(
                should_rotate_read_to_string(black_box(&reference), black_box(1_000))
                    .expect("reference rotation check"),
            );
        });
    });
    group.bench_function("reused_line_buffer/999", |b| {
        b.iter(|| {
            black_box(
                should_rotate_reused_line_buffer(black_box(&candidate), black_box(1_000))
                    .expect("candidate rotation check"),
            );
        });
    });
    group.finish();
}

// --- Encode throughput ---

#[allow(clippy::cast_possible_truncation)]
fn bench_encode_throughput(c: &mut Criterion) {
    let protector = make_protector();
    let mut group = c.benchmark_group("encode_throughput");
    group.measurement_time(Duration::from_secs(5));

    for &size_mb in &[1u32, 10] {
        let data_size = size_mb * 1024 * 1024;
        let dir = tempfile_dir("bench-encode");
        let file = dir.join(format!("source-{size_mb}mb.bin"));

        let payload: Vec<u8> = (0..data_size).map(|i| (i % 256) as u8).collect();
        std::fs::write(&file, &payload).expect("write source");

        group.throughput(criterion::Throughput::Bytes(u64::from(data_size)));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{size_mb}MB")),
            &file,
            |b, file| {
                b.iter(|| {
                    let sidecar = FileProtector::sidecar_path(file);
                    let _ = std::fs::remove_file(&sidecar);
                    protector.protect_file(black_box(file)).expect("protect");
                });
            },
        );
    }

    group.finish();
}

// --- Verify fast path (xxh3) ---

#[allow(clippy::cast_possible_truncation)]
fn bench_verify_fast_path(c: &mut Criterion) {
    let mut group = c.benchmark_group("verify_xxh3_fast_path");

    for &size_mb in &[1u32, 10, 50, 100] {
        let data_size = size_mb * 1024 * 1024;
        let data: Vec<u8> = (0..data_size).map(|i| (i % 256) as u8).collect();
        let expected = xxh3_64(&data);

        group.throughput(criterion::Throughput::Bytes(u64::from(data_size)));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{size_mb}MB")),
            &data,
            |b, data| {
                b.iter(|| {
                    let hash = xxh3_64(black_box(data));
                    assert_eq!(hash, expected);
                });
            },
        );
    }

    group.finish();
}

// --- File protect + verify roundtrip ---

#[allow(clippy::cast_possible_truncation)]
fn bench_protect_verify_roundtrip(c: &mut Criterion) {
    let protector = make_protector();
    let mut group = c.benchmark_group("protect_verify_roundtrip");
    group.measurement_time(Duration::from_secs(5));

    for &size_mb in &[1u32, 5] {
        let data_size = size_mb * 1024 * 1024;
        let dir = tempfile_dir("bench-roundtrip");
        let file = dir.join(format!("source-{size_mb}mb.bin"));
        let payload: Vec<u8> = (0..data_size).map(|i| (i % 256) as u8).collect();
        std::fs::write(&file, &payload).expect("write source");
        let result = protector.protect_file(&file).expect("protect");

        group.throughput(criterion::Throughput::Bytes(u64::from(data_size)));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{size_mb}MB")),
            &(file.clone(), result.sidecar_path.clone()),
            |b, (file, sidecar)| {
                b.iter(|| {
                    let verify = protector
                        .verify_file(black_box(file), black_box(sidecar))
                        .expect("verify");
                    assert!(verify.healthy);
                });
            },
        );
    }

    group.finish();
}

// --- Repair latency ---

#[allow(clippy::cast_possible_truncation)]
fn bench_repair_latency(c: &mut Criterion) {
    let config = DurabilityConfig {
        symbol_size: 4096,
        repair_overhead: 2.0,
        ..DurabilityConfig::default()
    };
    let metrics = Arc::new(DurabilityMetrics::default());
    let protector =
        FileProtector::new_with_metrics(Arc::new(BenchCodec), config, metrics).expect("protector");

    let mut group = c.benchmark_group("repair_latency");
    group.measurement_time(Duration::from_secs(5));

    let data_size = 1024u32 * 1024; // 1MB
    let dir = tempfile_dir("bench-repair");
    let file = dir.join("repair-target.bin");
    let payload: Vec<u8> = (0..data_size).map(|i| (i % 256) as u8).collect();
    std::fs::write(&file, &payload).expect("write source");
    let result = protector.protect_file(&file).expect("protect");

    group.bench_function("repair_1mb_single_block_corruption", |b| {
        b.iter(|| {
            let mut corrupted = payload.clone();
            corrupted[0..4096].fill(0xFF);
            std::fs::write(&file, &corrupted).expect("corrupt");

            let outcome = protector
                .repair_file(black_box(&file), black_box(&result.sidecar_path))
                .expect("repair");
            assert!(matches!(
                outcome,
                frankensearch_durability::file_protector::FileRepairOutcome::Repaired { .. }
            ));
        });
    });

    group.finish();
}

// --- Overhead measurement ---

#[allow(clippy::cast_possible_truncation)]
fn bench_overhead_ratio(c: &mut Criterion) {
    let mut group = c.benchmark_group("overhead_ratio");
    group.measurement_time(Duration::from_secs(3));

    let data_size = 1024u32 * 1024; // 1MB
    let payload: Vec<u8> = (0..data_size).map(|i| (i % 256) as u8).collect();

    for &overhead in &[1.05, 1.10, 1.20, 1.30, 1.50] {
        let config = DurabilityConfig {
            symbol_size: 4096,
            repair_overhead: overhead,
            ..DurabilityConfig::default()
        };
        let metrics = Arc::new(DurabilityMetrics::default());
        let protector = FileProtector::new_with_metrics(Arc::new(BenchCodec), config, metrics)
            .expect("protector");

        let dir = tempfile_dir("bench-overhead");
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let overhead_label = (overhead * 100.0) as u32;
        let file = dir.join(format!("overhead-{overhead_label}.bin"));
        std::fs::write(&file, &payload).expect("write");

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let pct = ((overhead - 1.0) * 100.0) as u32;
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{pct}%")),
            &file,
            |b, file| {
                b.iter(|| {
                    let sidecar = FileProtector::sidecar_path(file);
                    let _ = std::fs::remove_file(&sidecar);
                    let result = protector.protect_file(black_box(file)).expect("protect");
                    let fec_size = std::fs::metadata(&result.sidecar_path).map_or(0, |m| m.len());
                    black_box(fec_size);
                });
            },
        );
    }

    group.finish();
}

fn tempfile_dir(prefix: &str) -> std::path::PathBuf {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "frankensearch-bench-{prefix}-{}-{nanos}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

// --- DefaultSymbolCodec source-symbol build: zero-init elision A/B ---
//
// The built-in `DefaultSymbolCodec::encode` materializes each source symbol. The prior form allocated
// `vec![0; symbol_size]` and then `copy_from_slice`d the whole window over it — a wasted `memset` per
// FULL symbol (only the final short symbol needs a zero-padded tail). The new form copies full symbols
// directly (`to_vec`, no zero-init) and pads only the tail. Output is byte-identical (asserted).

/// Prior symbol build: zero-init every buffer, then overwrite.
fn build_symbols_old(source: &[u8], symbol_size: usize) -> Vec<Vec<u8>> {
    let k = source.len().div_ceil(symbol_size).max(1);
    let mut out = Vec::with_capacity(k);
    for i in 0..k {
        let start = i.saturating_mul(symbol_size);
        let end = start.saturating_add(symbol_size).min(source.len());
        let mut symbol = vec![0_u8; symbol_size];
        if start < end {
            symbol[..end - start].copy_from_slice(&source[start..end]);
        }
        out.push(symbol);
    }
    out
}

/// New symbol build: full symbols copy directly (no zero-init); only the tail pads.
fn build_symbols_new(source: &[u8], symbol_size: usize) -> Vec<Vec<u8>> {
    let k = source.len().div_ceil(symbol_size).max(1);
    let mut out = Vec::with_capacity(k);
    for i in 0..k {
        let start = i.saturating_mul(symbol_size);
        let end = start.saturating_add(symbol_size).min(source.len());
        let symbol = if end - start == symbol_size {
            source[start..end].to_vec()
        } else {
            let mut symbol = vec![0_u8; symbol_size];
            if start < end {
                symbol[..end - start].copy_from_slice(&source[start..end]);
            }
            symbol
        };
        out.push(symbol);
    }
    out
}

#[allow(clippy::cast_possible_truncation)]
fn bench_symbol_build(c: &mut Criterion) {
    let symbol_size = 4096_usize;
    let mut group = c.benchmark_group("symbol_build");
    for &size_mb in &[1u32, 10] {
        let data_size = (size_mb * 1024 * 1024) as usize;
        let source: Vec<u8> = (0..data_size).map(|i| (i % 251) as u8).collect();
        // Byte-identity gate: the elision must not change any symbol.
        assert_eq!(
            build_symbols_old(&source, symbol_size),
            build_symbols_new(&source, symbol_size),
            "symbol build diverged at {size_mb}MB"
        );

        group.throughput(criterion::Throughput::Bytes(
            u64::from(size_mb) * 1024 * 1024,
        ));
        group.bench_with_input(BenchmarkId::new("zero_init_old", size_mb), &(), |b, ()| {
            b.iter(|| black_box(build_symbols_old(black_box(&source), symbol_size)));
        });
        group.bench_with_input(
            BenchmarkId::new("direct_copy_new", size_mb),
            &(),
            |b, ()| {
                b.iter(|| black_box(build_symbols_new(black_box(&source), symbol_size)));
            },
        );
    }
    group.finish();
}

// --- verify_and_repair_file: reuse the corruption-detection decode vs re-verify (bench-internals) ---
//
// On the corrupt path `verify_and_repair_file` ran `verify_file` (mmap + source CRC32 + trailer
// deserialize) and THEN `repair_file_internal` re-ran its own verify (mmap + CRC32 + deserialize) —
// the residual fe866683's standalone-repair reuse left. The `reuse` arm (shipping) hands the verified
// decode into repair so it skips that second pass; `no_reuse` is the prior double-work. Byte-identical
// repaired output (asserted). Re-corrupts each iteration (repair is destructive), so the timed region
// includes the same corrupt-write cost in both arms — the delta is the eliminated second verify.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::needless_pass_by_ref_mut)] // `c` is mutated only under the `bench-internals` gate
fn bench_verify_repair_reuse(c: &mut Criterion) {
    #[cfg(feature = "bench-internals")]
    {
        let config = DurabilityConfig {
            symbol_size: 4096,
            repair_overhead: 2.0,
            ..DurabilityConfig::default()
        };
        let metrics = Arc::new(DurabilityMetrics::default());
        let protector = FileProtector::new_with_metrics(Arc::new(BenchCodec), config, metrics)
            .expect("protector");
        let data_size = 1024u32 * 1024; // 1MB
        let dir = tempfile_dir("bench-vr-reuse");
        let file = dir.join("vr-target.bin");
        let payload: Vec<u8> = (0..data_size).map(|i| (i % 256) as u8).collect();
        std::fs::write(&file, &payload).expect("write source");
        let _protect = protector.protect_file(&file).expect("protect");

        // Parity: both paths repair to the identical original bytes.
        let corrupt_and_repair = |reuse: bool| -> Vec<u8> {
            let mut corrupted = payload.clone();
            corrupted[0..4096].fill(0xFF);
            std::fs::write(&file, &corrupted).expect("corrupt");
            if reuse {
                protector.verify_and_repair_file(&file).expect("vr");
            } else {
                protector
                    .verify_and_repair_file_no_reuse(&file)
                    .expect("vr");
            }
            std::fs::read(&file).expect("read repaired")
        };
        assert_eq!(
            corrupt_and_repair(false),
            payload,
            "no_reuse must restore original"
        );
        assert_eq!(
            corrupt_and_repair(true),
            payload,
            "reuse must restore original"
        );

        let mut group = c.benchmark_group("verify_repair_reuse");
        group.measurement_time(Duration::from_secs(5));
        group.bench_function("no_reuse", |b| {
            b.iter(|| {
                let mut corrupted = payload.clone();
                corrupted[0..4096].fill(0xFF);
                std::fs::write(&file, &corrupted).expect("corrupt");
                black_box(
                    protector
                        .verify_and_repair_file_no_reuse(black_box(&file))
                        .expect("vr"),
                );
            });
        });
        group.bench_function("reuse", |b| {
            b.iter(|| {
                let mut corrupted = payload.clone();
                corrupted[0..4096].fill(0xFF);
                std::fs::write(&file, &corrupted).expect("corrupt");
                black_box(
                    protector
                        .verify_and_repair_file(black_box(&file))
                        .expect("vr"),
                );
            });
        });
        group.finish();
    }
    #[cfg(not(feature = "bench-internals"))]
    {
        let _ = c;
    }
}

criterion_group!(
    benches,
    bench_encode_throughput,
    bench_verify_fast_path,
    bench_protect_verify_roundtrip,
    bench_repair_latency,
    bench_overhead_ratio,
    bench_repair_log_rotation_ab,
    bench_symbol_build,
    bench_verify_repair_reuse,
);

criterion_main!(benches);
