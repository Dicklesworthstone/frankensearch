//! Durability benchmarks: encode/decode throughput, repair latency, xxh3 verify.
//!
//! Run with:
//!   cargo bench -p frankensearch-durability

use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_durability::config::DurabilityConfig;
use frankensearch_durability::file_protector::FileProtector;
use frankensearch_durability::metrics::DurabilityMetrics;
use fsqlite_core::raptorq_integration::{CodecDecodeResult, CodecEncodeResult, SymbolCodec};
use xxhash_rust::xxh3::xxh3_64;

/// Mock codec that mimics `RaptorQ` behavior for benchmarking the pipeline
/// overhead (serialization, CRC, file I/O) without the actual erasure coding.
#[derive(Debug)]
struct BenchCodec;

impl SymbolCodec for BenchCodec {
    fn encode(
        &self,
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

criterion_group!(
    benches,
    bench_encode_throughput,
    bench_verify_fast_path,
    bench_protect_verify_roundtrip,
    bench_repair_latency,
    bench_overhead_ratio,
);

criterion_main!(benches);
