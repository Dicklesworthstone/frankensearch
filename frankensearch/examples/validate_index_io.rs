//! E2E validation: vector index (FSVI) read/write integrity (bd-3un.40).
//!
//! Tests index creation, round-trip vector accuracy (f16 quantization loss),
//! header CRC validation, and corruption detection.
//!
//! Run with: `cargo run --example validate_index_io`

use std::path::Path;
use std::time::Instant;

use frankensearch_index::{Quantization, VectorIndex};

#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
fn main() {
    let start = Instant::now();
    let mut pass = 0u32;
    let mut fail = 0u32;

    println!("\n\x1b[1;36m=== frankensearch E2E: Vector Index I/O Validation ===\x1b[0m\n");

    let dir =
        std::env::temp_dir().join(format!("frankensearch-e2e-indexio-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");

    // ── Step 1: Write FSVI index with known vectors ───────────────────────
    log_info(
        "WRITE",
        "Creating FSVI index with 100 vectors (dim=128, f16)...",
    );
    let dim = 128;
    let n = 100;
    let idx_path = dir.join("test.idx");

    let original_vectors: Vec<(String, Vec<f32>)> = (0..n)
        .map(|i| {
            let doc_id = format!("doc-{i:04}");
            // Generate and L2-normalize (dot product search needs unit vectors)
            let mut vec: Vec<f32> = (0..dim)
                .map(|d| {
                    let seed = (i as u64)
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(d as u64);
                    (seed as f32 * 1e-10).sin()
                })
                .collect();
            let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut vec {
                    *x /= norm;
                }
            }
            (doc_id, vec)
        })
        .collect();

    let mut writer = VectorIndex::create(&idx_path, "test-embedder", dim).expect("create writer");
    for (doc_id, vec) in &original_vectors {
        writer.write_record(doc_id, vec).expect("write record");
    }
    writer.finish().expect("finish");

    let file_size = std::fs::metadata(&idx_path).expect("metadata").len();
    log_info(
        "WRITE",
        &format!("Written: {n} records, dim={dim}, file size={file_size} bytes"),
    );
    check(&mut pass, &mut fail, "Index write completes", true);

    // ── Step 2: Reopen and verify metadata ────────────────────────────────
    log_info("READ", "Reopening index and verifying metadata...");
    let index = VectorIndex::open(&idx_path).expect("open index");

    check(
        &mut pass,
        &mut fail,
        "Record count matches",
        index.record_count() == n,
    );
    check(
        &mut pass,
        &mut fail,
        "Dimension matches",
        index.dimension() == dim,
    );
    check(
        &mut pass,
        &mut fail,
        "Embedder ID matches",
        index.embedder_id() == "test-embedder",
    );
    check(
        &mut pass,
        &mut fail,
        "Quantization is F16",
        index.quantization() == Quantization::F16,
    );

    log_info(
        "READ",
        &format!(
            "Metadata OK: records={}, dim={}, embedder={}, quant={:?}",
            index.record_count(),
            index.dimension(),
            index.embedder_id(),
            index.quantization(),
        ),
    );

    // ── Step 3: Read back vectors and measure f16 quantization error ──────
    log_info(
        "ROUND-TRIP",
        "Reading back vectors and measuring f16 quantization loss...",
    );
    let mut max_abs_error: f32 = 0.0;
    let mut total_abs_error: f64 = 0.0;
    let mut error_count: usize = 0;
    let mut readback_ok = true;

    for (doc_id, original) in &original_vectors {
        // Find this doc in the index (records are sorted by hash, not insertion order)
        let doc_hash = fnv1a_hash(doc_id.as_bytes());
        let Some(idx) = index.find_index_by_doc_hash(doc_hash) else {
            log_fail("ROUND-TRIP", &format!("doc_id {doc_id} not found in index"));
            readback_ok = false;
            continue;
        };

        // Verify doc_id matches
        let read_id = index.doc_id_at(idx).expect("doc_id_at");
        if read_id != doc_id {
            log_fail(
                "ROUND-TRIP",
                &format!("doc_id mismatch at idx {idx}: expected {doc_id}, got {read_id}"),
            );
            readback_ok = false;
            continue;
        }

        // Read back as f32 (includes f16→f32 dequantization)
        let readback = index.vector_at_f32(idx).expect("vector_at_f32");
        if readback.len() != dim {
            log_fail(
                "ROUND-TRIP",
                &format!(
                    "dim mismatch for {doc_id}: expected {dim}, got {}",
                    readback.len()
                ),
            );
            readback_ok = false;
            continue;
        }

        for (d, (orig, read)) in original.iter().zip(readback.iter()).enumerate() {
            let err = (orig - read).abs();
            if err > max_abs_error {
                max_abs_error = err;
            }
            total_abs_error += f64::from(err);
            error_count += 1;

            // f16 has ~3 decimal digits of precision; max error should be < 0.01
            // for values in [-1, 1] range
            if err > 0.01 {
                log_fail(
                    "ROUND-TRIP",
                    &format!(
                        "{doc_id}[{d}]: orig={orig:.6}, read={read:.6}, err={err:.6} (exceeds 0.01)"
                    ),
                );
                readback_ok = false;
            }
        }
    }

    let avg_error = if error_count > 0 {
        total_abs_error / error_count as f64
    } else {
        0.0
    };
    log_info(
        "ROUND-TRIP",
        &format!(
            "Quantization error: max={max_abs_error:.6}, avg={avg_error:.6} ({error_count} elements)"
        ),
    );
    check(
        &mut pass,
        &mut fail,
        "All vectors read back within f16 tolerance",
        readback_ok,
    );

    // ── Step 4: F32 quantization round-trip (exact) ───────────────────────
    log_info("F32", "Testing F32 quantization (exact round-trip)...");
    let f32_path = dir.join("test_f32.idx");
    let mut writer_f32 = VectorIndex::create_with_revision(
        &f32_path,
        "test-embedder",
        "v1.0",
        dim,
        Quantization::F32,
    )
    .expect("create f32 writer");

    for (doc_id, vec) in &original_vectors {
        writer_f32.write_record(doc_id, vec).expect("write f32");
    }
    writer_f32.finish().expect("finish f32");

    let idx_f32 = VectorIndex::open(&f32_path).expect("open f32");
    let mut f32_exact = true;
    for (doc_id, original) in &original_vectors {
        let doc_hash = fnv1a_hash(doc_id.as_bytes());
        if let Some(idx) = idx_f32.find_index_by_doc_hash(doc_hash) {
            let readback = idx_f32.vector_at_f32(idx).expect("vector_at_f32");
            if readback != *original {
                log_fail("F32", &format!("{doc_id}: f32 round-trip mismatch"));
                f32_exact = false;
            }
        } else {
            log_fail("F32", &format!("{doc_id}: not found in f32 index"));
            f32_exact = false;
        }
    }
    check(&mut pass, &mut fail, "F32 exact round-trip", f32_exact);

    // ── Step 5: Search top-k accuracy ─────────────────────────────────────
    log_info("SEARCH", "Testing brute-force top-k search on F16 index...");
    let query = &original_vectors[0].1;
    let results = index.search_top_k(query, 10, None).expect("search");
    check(
        &mut pass,
        &mut fail,
        "Top-k returns results",
        !results.is_empty(),
    );
    check(
        &mut pass,
        &mut fail,
        "Top-k returns <= k results",
        results.len() <= 10,
    );

    // Verify scores are descending
    let scores_desc = results.windows(2).all(|w| w[0].score >= w[1].score);
    check(
        &mut pass,
        &mut fail,
        "Top-k scores are descending",
        scores_desc,
    );

    // Use the F32 index for self-search (exact round-trip, no quantization noise)
    log_info(
        "SEARCH",
        "Testing self-search on F32 index (exact vectors)...",
    );
    let results_f32 = idx_f32.search_top_k(query, 10, None).expect("search f32");
    if !results_f32.is_empty() {
        let top_ids: Vec<&str> = results_f32.iter().map(|r| r.doc_id.as_str()).collect();
        log_info(
            "SEARCH",
            &format!(
                "F32 top-3: {:?}, scores: [{:.4}, {:.4}, {:.4}]",
                &top_ids[..3.min(top_ids.len())],
                results_f32[0].score,
                results_f32.get(1).map_or(0.0, |r| r.score),
                results_f32.get(2).map_or(0.0, |r| r.score),
            ),
        );
        check(
            &mut pass,
            &mut fail,
            "F32 self-search: doc-0000 is top result",
            results_f32[0].doc_id == "doc-0000",
        );
    }

    // ── Step 6: Corruption detection ──────────────────────────────────────
    log_info("CORRUPT", "Testing corruption detection...");

    // 6a: Flip a byte in the magic header
    let corrupt_path = dir.join("corrupt_magic.idx");
    let mut data = std::fs::read(&idx_path).expect("read original");
    data[0] ^= 0xFF; // corrupt first byte of magic
    std::fs::write(&corrupt_path, &data).expect("write corrupted");
    let magic_result = VectorIndex::open(&corrupt_path);
    check(
        &mut pass,
        &mut fail,
        "Detects corrupted magic bytes",
        magic_result.is_err(),
    );
    if let Err(e) = &magic_result {
        log_info("CORRUPT", &format!("Magic corruption: {e}"));
    }

    // 6b: Corrupt the header CRC
    let corrupt_crc_path = dir.join("corrupt_crc.idx");
    let mut data2 = std::fs::read(&idx_path).expect("read original");
    // Find the CRC field — it's right before the record table.
    // The CRC is at a variable offset, so just corrupt a byte in the middle of the header.
    if data2.len() > 20 {
        data2[15] ^= 0xFF; // corrupt a byte within the header (embedder_id area)
        std::fs::write(&corrupt_crc_path, &data2).expect("write crc corrupted");
        let crc_result = VectorIndex::open(&corrupt_crc_path);
        check(
            &mut pass,
            &mut fail,
            "Detects header CRC mismatch",
            crc_result.is_err(),
        );
        if let Err(e) = &crc_result {
            log_info("CORRUPT", &format!("CRC corruption: {e}"));
        }
    }

    // 6c: Truncated file
    let truncated_path = dir.join("truncated.idx");
    let data3 = std::fs::read(&idx_path).expect("read original");
    std::fs::write(&truncated_path, &data3[..6]).expect("write truncated");
    let trunc_result = VectorIndex::open(&truncated_path);
    check(
        &mut pass,
        &mut fail,
        "Detects truncated file",
        trunc_result.is_err(),
    );
    if let Err(e) = &trunc_result {
        log_info("CORRUPT", &format!("Truncation: {e}"));
    }

    // 6d: Empty file
    let empty_path = dir.join("empty.idx");
    std::fs::write(&empty_path, []).expect("write empty");
    let empty_result = VectorIndex::open(&empty_path);
    check(
        &mut pass,
        &mut fail,
        "Detects empty file",
        empty_result.is_err(),
    );

    // 6e: Non-existent file
    let missing_result = VectorIndex::open(Path::new("/tmp/nonexistent_frankensearch.idx"));
    check(
        &mut pass,
        &mut fail,
        "Detects missing file",
        missing_result.is_err(),
    );

    // ── Cleanup and summary ───────────────────────────────────────────────
    let _ = std::fs::remove_dir_all(&dir);

    println!();
    println!("\x1b[1;36m=== Summary ===\x1b[0m");
    println!("  \x1b[32mPassed: {pass}\x1b[0m  \x1b[31mFailed: {fail}\x1b[0m");
    println!(
        "  Total time: {:.1}ms",
        start.elapsed().as_secs_f64() * 1000.0
    );
    println!();

    if fail > 0 {
        std::process::exit(1);
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn fnv1a_hash(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn log_info(step: &str, msg: &str) {
    println!("\x1b[36m[INFO] [{step}]\x1b[0m {msg}");
}

fn log_fail(step: &str, msg: &str) {
    println!("\x1b[31m[FAIL] [{step}]\x1b[0m {msg}");
}

fn log_pass(step: &str, msg: &str) {
    println!("\x1b[32m[PASS] [{step}]\x1b[0m {msg}");
}

fn check(pass: &mut u32, fail: &mut u32, name: &str, ok: bool) {
    if ok {
        log_pass("CHECK", name);
        *pass += 1;
    } else {
        log_fail("CHECK", name);
        *fail += 1;
    }
}
