//! Write-ahead log for incremental FSVI index updates.
//!
//! New vectors are appended to a `.fsvi.wal` sidecar file in batches.
//! Each batch is CRC32-protected so partial writes from crashes are
//! detected and discarded on reload.
//!
//! # File Layout
//!
//! ```text
//! ┌─────────────────────────────────────┐
//! │ WAL Header (20 bytes)               │
//! │   magic: b"FWAL" (4)               │
//! │   version: u16 LE                  │
//! │   dimension: u32 LE                │
//! │   quantization: u8                 │
//! │   reserved: [u8; 5]               │
//! │   header_crc32: u32 LE             │
//! ├─────────────────────────────────────┤
//! │ Batch 0                             │
//! │   batch_magic: b"FWB1" (4)         │
//! │   entry_count: u32 LE              │
//! │   entries...                        │
//! │   batch_crc32: u32 LE              │
//! ├─────────────────────────────────────┤
//! │ Batch 1 ...                         │
//! └─────────────────────────────────────┘
//! ```

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

use crc32fast::Hasher as Crc32Hasher;
use frankensearch_core::{SearchError, SearchResult};
use half::f16;
use tracing::{debug, warn};

use crate::Quantization;

// ─── Constants ──────────────────────────────────────────────────────────────

const WAL_MAGIC: [u8; 4] = *b"FWAL";
const WAL_VERSION: u16 = 1;
const BATCH_MAGIC: [u8; 4] = *b"FWB1";
/// Minimum WAL file size (header only).
const WAL_HEADER_SIZE: usize = 20;

// ─── Configuration ──────────────────────────────────────────────────────────

/// Configuration for WAL-based incremental updates.
#[derive(Debug, Clone)]
pub struct WalConfig {
    /// Maximum WAL entries before compaction is recommended.
    pub compaction_threshold: usize,
    /// Compaction threshold as fraction of main index size.
    pub compaction_ratio: f64,
    /// Whether to fsync after each batch write.
    pub fsync_on_write: bool,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            compaction_threshold: 1000,
            compaction_ratio: 0.10,
            fsync_on_write: true,
        }
    }
}

// ─── Types ──────────────────────────────────────────────────────────────────

/// A single WAL entry (in-memory representation).
#[derive(Debug, Clone)]
pub(crate) struct WalEntry {
    pub doc_id: String,
    /// Stored for future dedup/lookup optimizations during compaction.
    #[allow(dead_code)]
    pub doc_id_hash: u64,
    pub embedding: Vec<f32>,
}

/// Statistics from a compaction operation.
#[derive(Debug, Clone)]
pub struct CompactionStats {
    /// Records in the main index before compaction.
    pub main_records_before: usize,
    /// Records from the WAL merged in.
    pub wal_records: usize,
    /// Total records in the compacted index.
    pub total_records_after: usize,
    /// Time taken in milliseconds.
    pub elapsed_ms: f64,
}

// ─── Index tagging ──────────────────────────────────────────────────────────

/// Sentinel bit marking WAL-sourced entries in the search heap.
/// Uses the MSB of usize so main index positions (always < 2^63) are unaffected.
pub(crate) const WAL_INDEX_BIT: usize = 1_usize << (usize::BITS - 1);

pub(crate) const fn is_wal_index(index: usize) -> bool {
    index & WAL_INDEX_BIT != 0
}

pub(crate) const fn to_wal_index(wal_pos: usize) -> usize {
    wal_pos | WAL_INDEX_BIT
}

pub(crate) const fn from_wal_index(tagged: usize) -> usize {
    tagged & !WAL_INDEX_BIT
}

// ─── Path helpers ───────────────────────────────────────────────────────────

/// Derive the WAL sidecar path from the main FSVI index path.
#[must_use]
pub fn wal_path_for(fsvi_path: &Path) -> PathBuf {
    let mut p = fsvi_path.as_os_str().to_os_string();
    p.push(".wal");
    PathBuf::from(p)
}

// ─── WAL reading ────────────────────────────────────────────────────────────

/// Load all valid WAL entries from disk.
///
/// Partially-written batches (crash recovery) are silently discarded.
/// Returns an empty vec if the WAL file does not exist.
pub(crate) fn read_wal(
    path: &Path,
    expected_dimension: usize,
    quantization: Quantization,
) -> SearchResult<Vec<WalEntry>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let data = std::fs::read(path)?;
    if data.len() < WAL_HEADER_SIZE {
        warn!(path = %path.display(), len = data.len(), "WAL file too small, ignoring");
        return Ok(Vec::new());
    }
    parse_wal_bytes(&data, expected_dimension, quantization, path)
}

fn parse_wal_bytes(
    data: &[u8],
    expected_dimension: usize,
    quantization: Quantization,
    path: &Path,
) -> SearchResult<Vec<WalEntry>> {
    // Header validation.
    if data[..4] != WAL_MAGIC {
        return Err(wal_corrupted(path, "bad magic bytes"));
    }
    let version = u16::from_le_bytes([data[4], data[5]]);
    if version != WAL_VERSION {
        return Err(wal_corrupted(
            path,
            format!("version mismatch: expected {WAL_VERSION}, got {version}"),
        ));
    }
    let dimension = u32::from_le_bytes([data[6], data[7], data[8], data[9]]) as usize;
    if dimension != expected_dimension {
        return Err(wal_corrupted(
            path,
            format!("dimension mismatch: expected {expected_dimension}, got {dimension}"),
        ));
    }
    let quant_byte = data[10];
    let wal_quant = Quantization::from_wire(quant_byte, path)?;
    if wal_quant != quantization {
        return Err(wal_corrupted(path, "quantization mismatch"));
    }
    // reserved: data[11..16]
    let header_crc_stored = u32::from_le_bytes([data[16], data[17], data[18], data[19]]);
    let header_crc_computed = crc32_of(&data[..16]);
    if header_crc_stored != header_crc_computed {
        return Err(wal_corrupted(path, "header CRC mismatch"));
    }

    // Parse batches.
    let mut entries = Vec::new();
    let mut cursor = WAL_HEADER_SIZE;
    let vector_bytes = dimension * quantization.bytes_per_element();

    while cursor + 8 <= data.len() {
        if let Ok((batch_entries, batch_len)) =
            parse_batch(&data[cursor..], dimension, vector_bytes, quantization)
        {
            entries.extend(batch_entries);
            cursor += batch_len;
        } else {
            warn!(
                path = %path.display(),
                offset = cursor,
                entries_recovered = entries.len(),
                "WAL batch corrupt or truncated, discarding remainder"
            );
            break;
        }
    }

    debug!(path = %path.display(), entries = entries.len(), "loaded WAL entries");
    Ok(entries)
}

fn parse_batch(
    data: &[u8],
    dimension: usize,
    vector_bytes: usize,
    quantization: Quantization,
) -> Result<(Vec<WalEntry>, usize), ()> {
    if data.len() < 8 {
        return Err(());
    }
    if data[..4] != BATCH_MAGIC {
        return Err(());
    }
    let entry_count = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;

    // Bound capacity to what the data can actually hold (minimum 2 bytes header
    // + vector_bytes per entry) to prevent OOM from malicious entry_count values.
    let min_entry_bytes = 2_usize.saturating_add(vector_bytes).max(1);
    let max_possible = data.len().saturating_sub(8) / min_entry_bytes;
    let mut cursor: usize = 8;
    let mut entries = Vec::with_capacity(entry_count.min(max_possible));

    for _ in 0..entry_count {
        if cursor.checked_add(2).is_none_or(|end| end > data.len()) {
            return Err(());
        }
        let doc_id_len = u16::from_le_bytes([data[cursor], data[cursor + 1]]) as usize;
        cursor += 2;

        let needed = doc_id_len.checked_add(vector_bytes).ok_or(())?;
        if cursor
            .checked_add(needed)
            .is_none_or(|end| end > data.len())
        {
            return Err(());
        }
        let doc_id = std::str::from_utf8(&data[cursor..cursor + doc_id_len])
            .map_err(|_| ())?
            .to_owned();
        cursor += doc_id_len;

        let embedding = decode_vector(
            &data[cursor..cursor + vector_bytes],
            dimension,
            quantization,
        );
        cursor += vector_bytes;

        entries.push(WalEntry {
            doc_id_hash: crate::fnv1a_hash(doc_id.as_bytes()),
            doc_id,
            embedding,
        });
    }

    // Verify batch CRC.
    if cursor + 4 > data.len() {
        return Err(());
    }
    let stored_crc = u32::from_le_bytes([
        data[cursor],
        data[cursor + 1],
        data[cursor + 2],
        data[cursor + 3],
    ]);
    let computed_crc = crc32_of(&data[..cursor]);
    if stored_crc != computed_crc {
        return Err(());
    }
    cursor += 4;

    Ok((entries, cursor))
}

fn decode_vector(bytes: &[u8], dimension: usize, quantization: Quantization) -> Vec<f32> {
    match quantization {
        Quantization::F16 => bytes
            .chunks_exact(2)
            .take(dimension)
            .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
            .collect(),
        Quantization::F32 => bytes
            .chunks_exact(4)
            .take(dimension)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
    }
}

// ─── WAL writing ────────────────────────────────────────────────────────────

/// Append a batch of entries to the WAL file.
///
/// If the WAL file does not exist, a header is written first.
/// The batch is CRC32-protected so partial writes are detectable.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn append_wal_batch(
    wal_path: &Path,
    entries: &[WalEntry],
    dimension: usize,
    quantization: Quantization,
    fsync: bool,
) -> SearchResult<()> {
    let file_exists = wal_path.exists();

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(wal_path)?;

    if !file_exists {
        write_wal_header(&mut file, dimension, quantization)?;
    }

    // Build batch bytes.
    let mut batch = Vec::new();
    batch.extend_from_slice(&BATCH_MAGIC);
    batch.extend_from_slice(&(entries.len() as u32).to_le_bytes());

    for entry in entries {
        let doc_bytes = entry.doc_id.as_bytes();
        batch.extend_from_slice(&(doc_bytes.len() as u16).to_le_bytes());
        batch.extend_from_slice(doc_bytes);
        encode_vector(&mut batch, &entry.embedding, quantization);
    }

    let crc = crc32_of(&batch);
    batch.extend_from_slice(&crc.to_le_bytes());

    file.write_all(&batch)?;

    if fsync {
        file.sync_all()?;
    }

    debug!(
        path = %wal_path.display(),
        batch_entries = entries.len(),
        batch_bytes = batch.len(),
        "appended WAL batch"
    );
    Ok(())
}

#[allow(clippy::cast_possible_truncation)]
fn write_wal_header(
    writer: &mut impl Write,
    dimension: usize,
    quantization: Quantization,
) -> SearchResult<()> {
    let mut header = Vec::with_capacity(WAL_HEADER_SIZE);
    header.extend_from_slice(&WAL_MAGIC);
    header.extend_from_slice(&WAL_VERSION.to_le_bytes());
    header.extend_from_slice(&(dimension as u32).to_le_bytes());
    header.push(quantization as u8);
    header.extend_from_slice(&[0u8; 5]); // reserved
    let crc = crc32_of(&header);
    header.extend_from_slice(&crc.to_le_bytes());
    writer.write_all(&header)?;
    Ok(())
}

fn encode_vector(buf: &mut Vec<u8>, embedding: &[f32], quantization: Quantization) {
    match quantization {
        Quantization::F16 => {
            for &val in embedding {
                buf.extend_from_slice(&f16::from_f32(val).to_le_bytes());
            }
        }
        Quantization::F32 => {
            for &val in embedding {
                buf.extend_from_slice(&val.to_le_bytes());
            }
        }
    }
}

/// Remove the WAL sidecar file.
pub(crate) fn remove_wal(path: &Path) -> SearchResult<()> {
    if path.exists() {
        std::fs::remove_file(path)?;
    }
    Ok(())
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn crc32_of(data: &[u8]) -> u32 {
    let mut hasher = Crc32Hasher::new();
    hasher.update(data);
    hasher.finalize()
}

fn wal_corrupted(path: &Path, detail: impl Into<String>) -> SearchError {
    SearchError::IndexCorrupted {
        path: path.to_path_buf(),
        detail: detail.into(),
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_wal_path(name: &str) -> PathBuf {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "frankensearch-wal-{name}-{}-{now}.fsvi.wal",
            std::process::id()
        ))
    }

    fn make_entry(doc_id: &str, base: f32, dim: usize) -> WalEntry {
        WalEntry {
            doc_id: doc_id.to_owned(),
            doc_id_hash: crate::fnv1a_hash(doc_id.as_bytes()),
            embedding: vec![base; dim],
        }
    }

    #[test]
    fn roundtrip_single_batch() {
        let path = temp_wal_path("roundtrip-single");
        let dim = 4;
        let entries = vec![make_entry("doc-0", 1.0, dim), make_entry("doc-1", 2.0, dim)];

        append_wal_batch(&path, &entries, dim, Quantization::F16, false).unwrap();

        let loaded = read_wal(&path, dim, Quantization::F16).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].doc_id, "doc-0");
        assert_eq!(loaded[1].doc_id, "doc-1");
        assert!((loaded[0].embedding[0] - 1.0).abs() < 0.01);
        assert!((loaded[1].embedding[0] - 2.0).abs() < 0.01);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn roundtrip_multiple_batches() {
        let path = temp_wal_path("roundtrip-multi");
        let dim = 4;

        append_wal_batch(
            &path,
            &[make_entry("doc-0", 1.0, dim)],
            dim,
            Quantization::F16,
            false,
        )
        .unwrap();
        append_wal_batch(
            &path,
            &[make_entry("doc-1", 2.0, dim), make_entry("doc-2", 3.0, dim)],
            dim,
            Quantization::F16,
            false,
        )
        .unwrap();

        let loaded = read_wal(&path, dim, Quantization::F16).unwrap();
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded[0].doc_id, "doc-0");
        assert_eq!(loaded[1].doc_id, "doc-1");
        assert_eq!(loaded[2].doc_id, "doc-2");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn truncated_batch_is_discarded() {
        let path = temp_wal_path("truncated");
        let dim = 4;

        append_wal_batch(
            &path,
            &[make_entry("doc-good", 1.0, dim)],
            dim,
            Quantization::F16,
            false,
        )
        .unwrap();

        // Append garbage to simulate a partial batch write.
        let mut data = std::fs::read(&path).unwrap();
        data.extend_from_slice(&BATCH_MAGIC);
        data.extend_from_slice(&1_u32.to_le_bytes()); // claims 1 entry
        data.extend_from_slice(&[0xFF; 3]); // truncated entry
        std::fs::write(&path, &data).unwrap();

        let loaded = read_wal(&path, dim, Quantization::F16).unwrap();
        assert_eq!(loaded.len(), 1, "only the good batch should survive");
        assert_eq!(loaded[0].doc_id, "doc-good");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn crc_corrupted_batch_is_discarded() {
        let path = temp_wal_path("crc-corrupt");
        let dim = 4;

        append_wal_batch(
            &path,
            &[make_entry("doc-good", 1.0, dim)],
            dim,
            Quantization::F16,
            false,
        )
        .unwrap();
        append_wal_batch(
            &path,
            &[make_entry("doc-bad", 2.0, dim)],
            dim,
            Quantization::F16,
            false,
        )
        .unwrap();

        // Corrupt the CRC of the second batch.
        let mut data = std::fs::read(&path).unwrap();
        let last_byte = data.len() - 1;
        data[last_byte] ^= 0xFF;
        std::fs::write(&path, &data).unwrap();

        let loaded = read_wal(&path, dim, Quantization::F16).unwrap();
        assert_eq!(
            loaded.len(),
            1,
            "corrupted second batch should be discarded"
        );
        assert_eq!(loaded[0].doc_id, "doc-good");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn nonexistent_wal_returns_empty() {
        let path = temp_wal_path("nonexistent");
        let loaded = read_wal(&path, 4, Quantization::F16).unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn dimension_mismatch_is_error() {
        let path = temp_wal_path("dim-mismatch");
        append_wal_batch(
            &path,
            &[make_entry("doc-0", 1.0, 4)],
            4,
            Quantization::F16,
            false,
        )
        .unwrap();

        let result = read_wal(&path, 8, Quantization::F16);
        assert!(result.is_err());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn f32_quantization_roundtrip() {
        let path = temp_wal_path("f32-quant");
        let dim = 4;
        let entries = vec![make_entry("doc-0", 0.123_456, dim)];

        append_wal_batch(&path, &entries, dim, Quantization::F32, false).unwrap();

        let loaded = read_wal(&path, dim, Quantization::F32).unwrap();
        assert_eq!(loaded.len(), 1);
        assert!(
            (loaded[0].embedding[0] - 0.123_456).abs() < f32::EPSILON,
            "f32 should round-trip exactly"
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn wal_path_derivation() {
        let fsvi = Path::new("/data/index.fsvi");
        let wal = wal_path_for(fsvi);
        assert_eq!(wal.to_str().unwrap(), "/data/index.fsvi.wal");
    }

    #[test]
    fn remove_wal_is_idempotent() {
        let path = temp_wal_path("remove-idem");
        remove_wal(&path).unwrap(); // doesn't exist — ok
        std::fs::write(&path, b"dummy").unwrap();
        remove_wal(&path).unwrap(); // exists — removed
        assert!(!path.exists());
        remove_wal(&path).unwrap(); // gone again — ok
    }

    #[test]
    fn index_tagging_roundtrip() {
        let pos = 42;
        let tagged = to_wal_index(pos);
        assert!(is_wal_index(tagged));
        assert!(!is_wal_index(pos));
        assert_eq!(from_wal_index(tagged), pos);
    }

    // ─── bd-vbzm tests begin ───

    #[test]
    fn wal_config_default_values() {
        let config = WalConfig::default();
        assert_eq!(config.compaction_threshold, 1000);
        assert!((config.compaction_ratio - 0.10).abs() < f64::EPSILON);
        assert!(config.fsync_on_write);
    }

    #[test]
    fn wal_config_debug_clone() {
        let config = WalConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.compaction_threshold, config.compaction_threshold);
        let debug = format!("{config:?}");
        assert!(debug.contains("WalConfig"));
    }

    #[test]
    fn compaction_stats_debug_clone() {
        let stats = CompactionStats {
            main_records_before: 100,
            wal_records: 10,
            total_records_after: 110,
            elapsed_ms: 42.5,
        };
        let cloned = stats.clone();
        assert_eq!(cloned.main_records_before, 100);
        assert_eq!(cloned.wal_records, 10);
        assert_eq!(cloned.total_records_after, 110);
        assert!((cloned.elapsed_ms - 42.5).abs() < f64::EPSILON);
        let debug = format!("{stats:?}");
        assert!(debug.contains("CompactionStats"));
    }

    #[test]
    fn wal_too_small_returns_empty() {
        let path = temp_wal_path("too-small");
        std::fs::write(path.as_path(), &[0u8; 10]).unwrap();
        let loaded = read_wal(&path, 4, Quantization::F16).unwrap();
        assert!(loaded.is_empty());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn bad_magic_bytes_is_error() {
        let path = temp_wal_path("bad-magic");
        let mut header = vec![0xDE, 0xAD, 0xBE, 0xEF]; // wrong magic
        header.extend_from_slice(&WAL_VERSION.to_le_bytes());
        header.extend_from_slice(&4_u32.to_le_bytes());
        header.push(Quantization::F16 as u8);
        header.extend_from_slice(&[0u8; 5]);
        let crc = crc32_of(&header);
        header.extend_from_slice(&crc.to_le_bytes());
        std::fs::write(&path, &header).unwrap();

        let result = read_wal(&path, 4, Quantization::F16);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("magic"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn version_mismatch_is_error() {
        let path = temp_wal_path("version-mismatch");
        let mut header = Vec::new();
        header.extend_from_slice(&WAL_MAGIC);
        header.extend_from_slice(&99_u16.to_le_bytes()); // wrong version
        header.extend_from_slice(&4_u32.to_le_bytes());
        header.push(Quantization::F16 as u8);
        header.extend_from_slice(&[0u8; 5]);
        let crc = crc32_of(&header);
        header.extend_from_slice(&crc.to_le_bytes());
        std::fs::write(&path, &header).unwrap();

        let result = read_wal(&path, 4, Quantization::F16);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("version"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn quantization_mismatch_is_error() {
        let path = temp_wal_path("quant-mismatch");
        append_wal_batch(
            &path,
            &[make_entry("doc-0", 1.0, 4)],
            4,
            Quantization::F16,
            false,
        )
        .unwrap();

        let result = read_wal(&path, 4, Quantization::F32);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("quantization"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn header_crc_mismatch_is_error() {
        let path = temp_wal_path("header-crc");
        append_wal_batch(
            &path,
            &[make_entry("doc-0", 1.0, 4)],
            4,
            Quantization::F16,
            false,
        )
        .unwrap();

        let mut data = std::fs::read(&path).unwrap();
        // Corrupt header CRC (bytes 16-19)
        data[16] ^= 0xFF;
        std::fs::write(&path, &data).unwrap();

        let result = read_wal(&path, 4, Quantization::F16);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("CRC"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn empty_batch_roundtrip() {
        let path = temp_wal_path("empty-batch");
        let dim = 4;
        append_wal_batch(&path, &[], dim, Quantization::F16, false).unwrap();
        let loaded = read_wal(&path, dim, Quantization::F16).unwrap();
        assert!(loaded.is_empty());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn wal_entry_doc_id_hash_matches_fnv1a() {
        let entry = make_entry("hello", 1.0, 4);
        assert_eq!(entry.doc_id_hash, crate::fnv1a_hash(b"hello"));
    }

    #[test]
    fn wal_entry_clone() {
        let entry = make_entry("test-doc", 0.5, 8);
        #[allow(clippy::redundant_clone)]
        let cloned = entry.clone();
        assert_eq!(cloned.doc_id, "test-doc");
        assert_eq!(cloned.embedding.len(), 8);
        assert!((cloned.embedding[0] - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn index_tagging_zero() {
        let tagged = to_wal_index(0);
        assert!(is_wal_index(tagged));
        assert_eq!(from_wal_index(tagged), 0);
    }

    #[test]
    fn index_tagging_large_value() {
        let pos = (1_usize << (usize::BITS - 2)) - 1;
        let tagged = to_wal_index(pos);
        assert!(is_wal_index(tagged));
        assert_eq!(from_wal_index(tagged), pos);
    }

    #[test]
    fn is_wal_index_false_for_normal_values() {
        assert!(!is_wal_index(0));
        assert!(!is_wal_index(1));
        assert!(!is_wal_index(usize::MAX >> 1));
    }

    #[test]
    fn crc32_of_empty() {
        let crc = crc32_of(&[]);
        assert_eq!(crc, 0); // CRC32 of empty is 0
    }

    #[test]
    fn crc32_of_known_data() {
        let crc = crc32_of(b"hello");
        assert_ne!(crc, 0);
        // Same input should produce same CRC
        assert_eq!(crc, crc32_of(b"hello"));
    }

    #[test]
    fn wal_corrupted_produces_index_corrupted_error() {
        let err = wal_corrupted(Path::new("/test/path.fsvi.wal"), "test detail");
        match err {
            SearchError::IndexCorrupted { path, detail } => {
                assert_eq!(path, Path::new("/test/path.fsvi.wal"));
                assert_eq!(detail, "test detail");
            }
            other => panic!("expected IndexCorrupted, got {other:?}"),
        }
    }

    #[test]
    fn wal_path_for_various_inputs() {
        let wal = wal_path_for(Path::new("index.fsvi"));
        assert_eq!(wal, PathBuf::from("index.fsvi.wal"));

        let wal = wal_path_for(Path::new("/abs/path/data.fsvi"));
        assert_eq!(wal, PathBuf::from("/abs/path/data.fsvi.wal"));
    }

    #[test]
    fn append_with_fsync_succeeds() {
        let path = temp_wal_path("fsync-test");
        let dim = 4;
        append_wal_batch(
            &path,
            &[make_entry("doc-sync", 1.0, dim)],
            dim,
            Quantization::F16,
            true,
        )
        .unwrap();

        let loaded = read_wal(&path, dim, Quantization::F16).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].doc_id, "doc-sync");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn append_preserves_existing_header() {
        let path = temp_wal_path("preserve-header");
        let dim = 4;
        append_wal_batch(
            &path,
            &[make_entry("doc-0", 1.0, dim)],
            dim,
            Quantization::F16,
            false,
        )
        .unwrap();

        let data_after_first = std::fs::read(&path).unwrap();
        let header_bytes = &data_after_first[..WAL_HEADER_SIZE];

        append_wal_batch(
            &path,
            &[make_entry("doc-1", 2.0, dim)],
            dim,
            Quantization::F16,
            false,
        )
        .unwrap();

        let data_after_second = std::fs::read(&path).unwrap();
        // Header should be unchanged
        assert_eq!(&data_after_second[..WAL_HEADER_SIZE], header_bytes);
        // File should be larger
        assert!(data_after_second.len() > data_after_first.len());

        let loaded = read_wal(&path, dim, Quantization::F16).unwrap();
        assert_eq!(loaded.len(), 2);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn bad_batch_magic_discards_batch() {
        let path = temp_wal_path("bad-batch-magic");
        let dim = 4;
        append_wal_batch(
            &path,
            &[make_entry("doc-good", 1.0, dim)],
            dim,
            Quantization::F16,
            false,
        )
        .unwrap();

        let mut data = std::fs::read(&path).unwrap();
        // Append a batch with wrong magic
        data.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF]); // bad batch magic
        data.extend_from_slice(&1_u32.to_le_bytes());
        std::fs::write(&path, &data).unwrap();

        let loaded = read_wal(&path, dim, Quantization::F16).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].doc_id, "doc-good");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn f16_quantization_has_limited_precision() {
        let path = temp_wal_path("f16-precision");
        let dim = 2;
        let entries = vec![WalEntry {
            doc_id: "precise".to_owned(),
            doc_id_hash: crate::fnv1a_hash(b"precise"),
            embedding: vec![0.123_456, -0.987_654],
        }];

        append_wal_batch(&path, &entries, dim, Quantization::F16, false).unwrap();
        let loaded = read_wal(&path, dim, Quantization::F16).unwrap();
        assert_eq!(loaded.len(), 1);
        // F16 has ~3 decimal digits of precision
        assert!((loaded[0].embedding[0] - 0.123_456).abs() < 0.001);
        assert!((loaded[0].embedding[1] + 0.987_654).abs() < 0.001);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn f32_quantization_exact_roundtrip() {
        let path = temp_wal_path("f32-exact");
        let dim = 3;
        let val = std::f32::consts::PI;
        let entries = vec![WalEntry {
            doc_id: "pi-doc".to_owned(),
            doc_id_hash: crate::fnv1a_hash(b"pi-doc"),
            embedding: vec![val, -val, 0.0],
        }];

        append_wal_batch(&path, &entries, dim, Quantization::F32, false).unwrap();
        let loaded = read_wal(&path, dim, Quantization::F32).unwrap();
        assert!((loaded[0].embedding[0] - val).abs() < f32::EPSILON);
        assert!((loaded[0].embedding[1] + val).abs() < f32::EPSILON);
        assert!(loaded[0].embedding[2].abs() < f32::EPSILON);
        std::fs::remove_file(&path).ok();
    }

    // ─── bd-vbzm tests end ───
}
