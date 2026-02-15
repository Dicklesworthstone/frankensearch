//! Integration tests for FSVI binary format: write/read roundtrip, CRC validation,
//! WAL lifecycle, tombstone/vacuum, compaction, and search correctness.

use std::path::Path;

use frankensearch_index::{Quantization, VectorIndex};

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn temp_index_path(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("frankensearch_test");
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir.join(format!("{name}.fsvi"))
}

fn cleanup(path: &Path) {
    let _ = std::fs::remove_file(path);
    // Also clean WAL sidecar if present.
    let wal_path = path.with_extension("fsvi.wal");
    let _ = std::fs::remove_file(&wal_path);
}

/// Normalize an f32 vector to unit length.
fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < f32::EPSILON {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

// ─── FSVI Write/Read Roundtrip ────────────────────────────────────────────────

#[test]
fn write_and_read_f16_single_record() {
    let path = temp_index_path("single_f16");
    cleanup(&path);

    let dim = 8;
    let embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

    // Write.
    let mut writer = VectorIndex::create(&path, "test-embedder", dim).unwrap();
    writer.write_record("doc-1", &embedding).unwrap();
    writer.finish().unwrap();

    // Read back.
    let index = VectorIndex::open(&path).unwrap();
    assert_eq!(index.record_count(), 1);
    assert_eq!(index.dimension(), dim);
    assert_eq!(index.embedder_id(), "test-embedder");
    assert_eq!(index.quantization(), Quantization::F16);

    // Verify doc_id roundtrip.
    assert_eq!(index.doc_id_at(0).unwrap(), "doc-1");

    // Verify vector roundtrip (f16 quantization introduces small error).
    let recovered = index.vector_at_f32(0).unwrap();
    assert_eq!(recovered.len(), dim);
    for (orig, recovered) in embedding.iter().zip(&recovered) {
        assert!(
            (orig - recovered).abs() < 0.01,
            "f16 roundtrip error too large: orig={orig}, recovered={recovered}"
        );
    }

    cleanup(&path);
}

#[test]
fn write_and_read_f32_roundtrip() {
    let path = temp_index_path("f32_roundtrip");
    cleanup(&path);

    let dim = 4;
    let embedding = vec![1.5, -2.3, 0.001, 99.99];

    let mut writer =
        VectorIndex::create_with_revision(&path, "test-f32", "rev-1", dim, Quantization::F32)
            .unwrap();
    writer.write_record("exact-doc", &embedding).unwrap();
    writer.finish().unwrap();

    let index = VectorIndex::open(&path).unwrap();
    assert_eq!(index.quantization(), Quantization::F32);
    assert_eq!(index.embedder_revision(), "rev-1");

    // F32 should round-trip exactly.
    let recovered = index.vector_at_f32(0).unwrap();
    assert_eq!(recovered, embedding);

    cleanup(&path);
}

#[test]
fn write_multiple_records_preserves_all() {
    let path = temp_index_path("multi_records");
    cleanup(&path);

    let dim = 4;
    let records: Vec<(&str, Vec<f32>)> = vec![
        ("alpha", vec![1.0, 0.0, 0.0, 0.0]),
        ("beta", vec![0.0, 1.0, 0.0, 0.0]),
        ("gamma", vec![0.0, 0.0, 1.0, 0.0]),
        ("delta", vec![0.0, 0.0, 0.0, 1.0]),
    ];

    let mut writer =
        VectorIndex::create_with_revision(&path, "multi", "", dim, Quantization::F32).unwrap();
    for (doc_id, emb) in &records {
        writer.write_record(doc_id, emb).unwrap();
    }
    writer.finish().unwrap();

    let index = VectorIndex::open(&path).unwrap();
    assert_eq!(index.record_count(), 4);

    // Records are sorted by doc_id_hash in the file, so we verify by
    // doc_id_at for each index rather than assuming insertion order.
    let mut found_ids: Vec<String> = (0..4)
        .map(|i| index.doc_id_at(i).unwrap().to_owned())
        .collect();
    found_ids.sort();
    let mut expected_ids: Vec<String> = records.iter().map(|(id, _)| (*id).to_owned()).collect();
    expected_ids.sort();
    assert_eq!(found_ids, expected_ids);

    cleanup(&path);
}

// ─── Metadata ─────────────────────────────────────────────────────────────────

#[test]
fn metadata_fields_are_preserved() {
    let path = temp_index_path("metadata");
    cleanup(&path);

    let mut writer = VectorIndex::create_with_revision(
        &path,
        "my-embedder-v2",
        "abc123def456",
        128,
        Quantization::F16,
    )
    .unwrap();
    writer.write_record("m1", &vec![0.5; 128]).unwrap();
    writer.finish().unwrap();

    let index = VectorIndex::open(&path).unwrap();
    let meta = index.metadata();
    assert_eq!(meta.embedder_id, "my-embedder-v2");
    assert_eq!(meta.embedder_revision, "abc123def456");
    assert_eq!(meta.dimension, 128);
    assert_eq!(meta.quantization, Quantization::F16);
    assert_eq!(meta.record_count, 1);

    cleanup(&path);
}

// ─── Error Handling ───────────────────────────────────────────────────────────

#[test]
fn open_nonexistent_file_returns_not_found() {
    let path = temp_index_path("nonexistent_12345");
    cleanup(&path);

    let result = VectorIndex::open(&path);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        format!("{err:?}").contains("IndexNotFound"),
        "expected IndexNotFound, got: {err:?}"
    );
}

#[test]
fn zero_dimension_returns_invalid_config() {
    let path = temp_index_path("zero_dim");
    let result = VectorIndex::create(&path, "test", 0);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        format!("{err:?}").contains("InvalidConfig"),
        "expected InvalidConfig for zero dimension, got: {err:?}"
    );
}

#[test]
fn dimension_mismatch_on_write_is_rejected() {
    let path = temp_index_path("dim_mismatch");
    cleanup(&path);

    let mut writer = VectorIndex::create(&path, "test", 4).unwrap();
    // Correct dimension works.
    writer.write_record("ok", &[1.0, 2.0, 3.0, 4.0]).unwrap();
    // Wrong dimension fails.
    let err = writer.write_record("bad", &[1.0, 2.0]).unwrap_err();
    assert!(
        format!("{err:?}").contains("DimensionMismatch"),
        "expected DimensionMismatch, got: {err:?}"
    );

    cleanup(&path);
}

#[test]
fn non_finite_embedding_is_rejected() {
    let path = temp_index_path("non_finite");
    cleanup(&path);

    let mut writer = VectorIndex::create(&path, "test", 4).unwrap();
    let err = writer
        .write_record("nan", &[1.0, f32::NAN, 3.0, 4.0])
        .unwrap_err();
    assert!(
        format!("{err:?}").contains("non-finite"),
        "expected non-finite rejection, got: {err:?}"
    );

    let err = writer
        .write_record("inf", &[1.0, 2.0, f32::INFINITY, 4.0])
        .unwrap_err();
    assert!(
        format!("{err:?}").contains("non-finite"),
        "expected non-finite rejection, got: {err:?}"
    );

    cleanup(&path);
}

// ─── Empty Index ──────────────────────────────────────────────────────────────

#[test]
fn empty_index_roundtrip() {
    let path = temp_index_path("empty");
    cleanup(&path);

    let writer = VectorIndex::create(&path, "empty-emb", 16).unwrap();
    writer.finish().unwrap();

    let index = VectorIndex::open(&path).unwrap();
    assert_eq!(index.record_count(), 0);
    assert_eq!(index.dimension(), 16);
    assert_eq!(index.tombstone_count(), 0);
    assert!(!index.needs_vacuum());
    assert!(!index.needs_compaction());

    cleanup(&path);
}

// ─── WAL Append + Compact Lifecycle ───────────────────────────────────────────

#[test]
fn wal_append_and_compact() {
    let path = temp_index_path("wal_lifecycle");
    cleanup(&path);

    // Create an initial index with 2 records.
    let dim = 4;
    let mut writer =
        VectorIndex::create_with_revision(&path, "wal-test", "", dim, Quantization::F32).unwrap();
    writer
        .write_record("main-1", &[1.0, 0.0, 0.0, 0.0])
        .unwrap();
    writer
        .write_record("main-2", &[0.0, 1.0, 0.0, 0.0])
        .unwrap();
    writer.finish().unwrap();

    // Open and append via WAL.
    let mut index = VectorIndex::open(&path).unwrap();
    assert_eq!(index.record_count(), 2);
    assert_eq!(index.wal_record_count(), 0);

    index.append("wal-1", &[0.0, 0.0, 1.0, 0.0]).unwrap();
    index.append("wal-2", &[0.0, 0.0, 0.0, 1.0]).unwrap();
    assert_eq!(index.wal_record_count(), 2);

    // Compact: merge WAL into main index.
    let stats = index.compact().unwrap();
    assert_eq!(stats.main_records_before, 2);
    assert_eq!(stats.wal_records, 2);
    assert_eq!(stats.total_records_after, 4);
    assert_eq!(index.wal_record_count(), 0);

    // Re-open and verify all 4 records are in the main index.
    let reopened = VectorIndex::open(&path).unwrap();
    assert_eq!(reopened.record_count(), 4);
    assert_eq!(reopened.wal_record_count(), 0);

    cleanup(&path);
}

#[test]
fn wal_append_dimension_mismatch() {
    let path = temp_index_path("wal_dim_mismatch");
    cleanup(&path);

    let mut writer = VectorIndex::create(&path, "test", 4).unwrap();
    writer.write_record("x", &[1.0, 2.0, 3.0, 4.0]).unwrap();
    writer.finish().unwrap();

    let mut index = VectorIndex::open(&path).unwrap();
    let err = index.append("bad", &[1.0, 2.0]).unwrap_err();
    assert!(
        format!("{err:?}").contains("DimensionMismatch"),
        "expected DimensionMismatch on WAL append, got: {err:?}"
    );

    cleanup(&path);
}

#[test]
fn compact_empty_wal_is_noop() {
    let path = temp_index_path("compact_noop");
    cleanup(&path);

    let mut writer = VectorIndex::create(&path, "test", 4).unwrap();
    writer.write_record("x", &[1.0, 2.0, 3.0, 4.0]).unwrap();
    writer.finish().unwrap();

    let mut index = VectorIndex::open(&path).unwrap();
    let stats = index.compact().unwrap();
    assert_eq!(stats.wal_records, 0);
    assert_eq!(stats.total_records_after, 1);

    cleanup(&path);
}

// ─── Tombstone + Vacuum ──────────────────────────────────────────────────────

#[test]
fn soft_delete_and_vacuum() {
    let path = temp_index_path("tombstone_vacuum");
    cleanup(&path);

    let dim = 4;
    let mut writer =
        VectorIndex::create_with_revision(&path, "vac", "", dim, Quantization::F32).unwrap();
    writer.write_record("keep", &[1.0, 0.0, 0.0, 0.0]).unwrap();
    writer
        .write_record("delete-me", &[0.0, 1.0, 0.0, 0.0])
        .unwrap();
    writer
        .write_record("also-keep", &[0.0, 0.0, 1.0, 0.0])
        .unwrap();
    writer.finish().unwrap();

    let mut index = VectorIndex::open(&path).unwrap();
    assert_eq!(index.record_count(), 3);
    assert_eq!(index.tombstone_count(), 0);

    // Delete one record.
    let deleted = index.soft_delete("delete-me").unwrap();
    assert!(deleted);
    assert_eq!(index.tombstone_count(), 1);

    // Double-delete returns false.
    let deleted_again = index.soft_delete("delete-me").unwrap();
    assert!(!deleted_again);

    // Delete non-existent returns false.
    let not_found = index.soft_delete("no-such-doc").unwrap();
    assert!(!not_found);

    // Vacuum.
    let stats = index.vacuum().unwrap();
    assert_eq!(stats.records_before, 3);
    assert_eq!(stats.records_after, 2);
    assert_eq!(stats.tombstones_removed, 1);
    assert_eq!(index.tombstone_count(), 0);
    assert_eq!(index.record_count(), 2);

    cleanup(&path);
}

#[test]
fn vacuum_empty_index_is_noop() {
    let path = temp_index_path("vacuum_empty");
    cleanup(&path);

    let writer = VectorIndex::create(&path, "test", 4).unwrap();
    writer.finish().unwrap();

    let mut index = VectorIndex::open(&path).unwrap();
    let stats = index.vacuum().unwrap();
    assert_eq!(stats.records_before, 0);
    assert_eq!(stats.records_after, 0);
    assert_eq!(stats.tombstones_removed, 0);

    cleanup(&path);
}

// ─── Search Correctness ──────────────────────────────────────────────────────

#[test]
fn search_returns_closest_vector() {
    let path = temp_index_path("search_closest");
    cleanup(&path);

    let dim = 4;
    // Use unit vectors as "documents" so cosine similarity = dot product.
    let docs = vec![
        ("north", normalize(&[1.0, 0.0, 0.0, 0.0])),
        ("east", normalize(&[0.0, 1.0, 0.0, 0.0])),
        ("northeast", normalize(&[1.0, 1.0, 0.0, 0.0])),
    ];

    let mut writer =
        VectorIndex::create_with_revision(&path, "search", "", dim, Quantization::F32).unwrap();
    for (id, emb) in &docs {
        writer.write_record(id, emb).unwrap();
    }
    writer.finish().unwrap();

    let index = VectorIndex::open(&path).unwrap();

    // Query close to "north" — north should rank highest.
    let query = normalize(&[0.9, 0.1, 0.0, 0.0]);
    let results = index.search_top_k(&query, 3, None).unwrap();
    assert!(!results.is_empty());
    assert_eq!(
        results[0].doc_id, "north",
        "north should be closest to [0.9, 0.1, 0, 0]"
    );

    // Query close to "east" — east should rank highest.
    let query = normalize(&[0.1, 0.9, 0.0, 0.0]);
    let results = index.search_top_k(&query, 3, None).unwrap();
    assert_eq!(
        results[0].doc_id, "east",
        "east should be closest to [0.1, 0.9, 0, 0]"
    );

    cleanup(&path);
}

#[test]
fn search_respects_limit() {
    let path = temp_index_path("search_limit");
    cleanup(&path);

    let dim = 4;
    let mut writer =
        VectorIndex::create_with_revision(&path, "limit", "", dim, Quantization::F32).unwrap();
    for i in 0..10 {
        let mut emb = vec![0.0; dim];
        emb[i % dim] = 1.0;
        writer.write_record(&format!("doc-{i}"), &emb).unwrap();
    }
    writer.finish().unwrap();

    let index = VectorIndex::open(&path).unwrap();
    let query = vec![1.0, 0.0, 0.0, 0.0];

    let results = index.search_top_k(&query, 3, None).unwrap();
    assert!(
        results.len() <= 3,
        "should respect limit=3, got {}",
        results.len()
    );

    let results = index.search_top_k(&query, 100, None).unwrap();
    assert_eq!(results.len(), 10, "should return all 10 when limit > count");

    cleanup(&path);
}

#[test]
fn search_on_empty_index_returns_empty() {
    let path = temp_index_path("search_empty");
    cleanup(&path);

    let writer = VectorIndex::create(&path, "test", 4).unwrap();
    writer.finish().unwrap();

    let index = VectorIndex::open(&path).unwrap();
    let results = index.search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, None).unwrap();
    assert!(results.is_empty());

    cleanup(&path);
}

#[test]
fn search_dimension_mismatch_is_error() {
    let path = temp_index_path("search_dim_err");
    cleanup(&path);

    let mut writer = VectorIndex::create(&path, "test", 4).unwrap();
    writer.write_record("x", &[1.0, 0.0, 0.0, 0.0]).unwrap();
    writer.finish().unwrap();

    let index = VectorIndex::open(&path).unwrap();
    let err = index.search_top_k(&[1.0, 0.0], 1, None).unwrap_err();
    assert!(
        format!("{err:?}").contains("DimensionMismatch"),
        "expected DimensionMismatch, got: {err:?}"
    );

    cleanup(&path);
}

// ─── WAL entries are searchable before compaction ─────────────────────────────

#[test]
fn wal_entries_are_searchable_before_compaction() {
    let path = temp_index_path("wal_search");
    cleanup(&path);

    let dim = 4;
    let mut writer =
        VectorIndex::create_with_revision(&path, "wal-s", "", dim, Quantization::F32).unwrap();
    writer
        .write_record("main-doc", &normalize(&[1.0, 0.0, 0.0, 0.0]))
        .unwrap();
    writer.finish().unwrap();

    let mut index = VectorIndex::open(&path).unwrap();
    // Append a WAL entry that should be a better match for a specific query.
    index
        .append("wal-doc", &normalize(&[0.0, 1.0, 0.0, 0.0]))
        .unwrap();

    // Search for something close to the WAL entry.
    let query = normalize(&[0.1, 0.9, 0.0, 0.0]);
    let results = index.search_top_k(&query, 2, None).unwrap();
    assert!(!results.is_empty());
    assert_eq!(
        results[0].doc_id, "wal-doc",
        "WAL entry should be searchable and rank highest"
    );

    cleanup(&path);
}

// ─── CRC Validation ──────────────────────────────────────────────────────────

#[test]
fn corrupted_header_is_detected() {
    let path = temp_index_path("corrupt_header");
    cleanup(&path);

    // Write a valid index.
    let mut writer = VectorIndex::create(&path, "crc", 4).unwrap();
    writer.write_record("x", &[1.0, 2.0, 3.0, 4.0]).unwrap();
    writer.finish().unwrap();

    // Corrupt a byte in the header (after magic, before CRC).
    let mut data = std::fs::read(&path).unwrap();
    assert!(data.len() > 10);
    data[6] ^= 0xFF; // Flip a byte in the header region.
    std::fs::write(&path, &data).unwrap();

    // Attempt to open — should fail with corruption error.
    let result = VectorIndex::open(&path);
    assert!(result.is_err(), "corrupted header should be detected");
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("Corrupted")
            || err_msg.contains("corrupted")
            || err_msg.contains("CRC")
            || err_msg.contains("crc"),
        "expected corruption/CRC error, got: {err_msg}"
    );

    cleanup(&path);
}

#[test]
fn truncated_file_is_detected() {
    let path = temp_index_path("truncated");
    cleanup(&path);

    let mut writer = VectorIndex::create(&path, "trunc", 4).unwrap();
    writer.write_record("x", &[1.0, 2.0, 3.0, 4.0]).unwrap();
    writer.finish().unwrap();

    // Truncate file to just the magic bytes.
    let data = std::fs::read(&path).unwrap();
    std::fs::write(&path, &data[..8]).unwrap();

    let result = VectorIndex::open(&path);
    assert!(result.is_err(), "truncated file should fail to open");

    cleanup(&path);
}

// ─── Batch Operations ────────────────────────────────────────────────────────

#[test]
fn soft_delete_batch() {
    let path = temp_index_path("batch_delete");
    cleanup(&path);

    let dim = 4;
    let mut writer =
        VectorIndex::create_with_revision(&path, "batch", "", dim, Quantization::F32).unwrap();
    for i in 0..5 {
        let mut emb = vec![0.0; dim];
        emb[i % dim] = 1.0;
        writer.write_record(&format!("doc-{i}"), &emb).unwrap();
    }
    writer.finish().unwrap();

    let mut index = VectorIndex::open(&path).unwrap();
    let deleted = index
        .soft_delete_batch(&["doc-1", "doc-3", "no-such-doc"])
        .unwrap();
    assert_eq!(deleted, 2, "should delete 2 existing docs");
    assert_eq!(index.tombstone_count(), 2);

    cleanup(&path);
}

#[test]
fn append_batch_via_wal() {
    let path = temp_index_path("wal_batch");
    cleanup(&path);

    let dim = 4;
    let writer =
        VectorIndex::create_with_revision(&path, "batch", "", dim, Quantization::F32).unwrap();
    writer.finish().unwrap();

    let mut index = VectorIndex::open(&path).unwrap();
    let entries = vec![
        ("b1".to_owned(), vec![1.0, 0.0, 0.0, 0.0]),
        ("b2".to_owned(), vec![0.0, 1.0, 0.0, 0.0]),
        ("b3".to_owned(), vec![0.0, 0.0, 1.0, 0.0]),
    ];
    index.append_batch(&entries).unwrap();
    assert_eq!(index.wal_record_count(), 3);

    cleanup(&path);
}

// ─── Needs Compaction Heuristics ─────────────────────────────────────────────

#[test]
fn needs_compaction_threshold() {
    let path = temp_index_path("compact_heuristic");
    cleanup(&path);

    let dim = 4;
    let mut writer =
        VectorIndex::create_with_revision(&path, "heur", "", dim, Quantization::F32).unwrap();
    writer.write_record("x", &[1.0, 0.0, 0.0, 0.0]).unwrap();
    writer.finish().unwrap();

    let mut index = VectorIndex::open(&path).unwrap();
    assert!(!index.needs_compaction(), "no WAL entries → no compaction");

    // The default compaction_ratio is 0.10, so 1 WAL entry on 1 main record
    // (ratio = 1.0) should trigger compaction.
    index.append("wal", &[0.0, 1.0, 0.0, 0.0]).unwrap();
    assert!(
        index.needs_compaction(),
        "WAL ratio 1.0 > 0.10 → should need compaction"
    );

    cleanup(&path);
}

// ─── f16 vs f32 Quantization Search Consistency ──────────────────────────────

#[test]
fn f16_and_f32_search_produce_same_ranking() {
    let dim = 8;
    let docs: Vec<(&str, Vec<f32>)> = vec![
        ("a", normalize(&[1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        ("b", normalize(&[0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])),
        ("c", normalize(&[0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0])),
    ];
    let query = normalize(&[0.9, 0.6, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]);

    // Write f16 index.
    let path_f16 = temp_index_path("quant_f16");
    cleanup(&path_f16);
    let mut w16 =
        VectorIndex::create_with_revision(&path_f16, "q", "", dim, Quantization::F16).unwrap();
    for (id, emb) in &docs {
        w16.write_record(id, emb).unwrap();
    }
    w16.finish().unwrap();

    // Write f32 index.
    let path_f32 = temp_index_path("quant_f32");
    cleanup(&path_f32);
    let mut w32 =
        VectorIndex::create_with_revision(&path_f32, "q", "", dim, Quantization::F32).unwrap();
    for (id, emb) in &docs {
        w32.write_record(id, emb).unwrap();
    }
    w32.finish().unwrap();

    let idx16 = VectorIndex::open(&path_f16).unwrap();
    let idx32 = VectorIndex::open(&path_f32).unwrap();

    let r16 = idx16.search_top_k(&query, 3, None).unwrap();
    let r32 = idx32.search_top_k(&query, 3, None).unwrap();

    // Rankings should match (doc ids in same order).
    let ranking_f16: Vec<&str> = r16.iter().map(|h| h.doc_id.as_str()).collect();
    let ranking_f32: Vec<&str> = r32.iter().map(|h| h.doc_id.as_str()).collect();
    assert_eq!(
        ranking_f16, ranking_f32,
        "f16 and f32 should produce the same ranking order"
    );

    cleanup(&path_f16);
    cleanup(&path_f32);
}
