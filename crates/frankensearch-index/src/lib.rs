//! Vector index storage and loading for frankensearch.
//!
//! This crate implements the FSVI binary format reader/writer plus exact
//! brute-force top-k vector search, with optional HNSW ANN acceleration.
//!
//! # FSVI File Layout
//!
//! All multi-byte integers are little-endian. The vector slab is 64-byte
//! aligned for cache-line / SIMD friendliness.
//!
//! ```text
//! ┌───────────────────────────────────────────┐
//! │ Header (variable length)                  │
//! │   magic: b"FSVI"              (4 bytes)   │
//! │   version: u16                (2 bytes)   │
//! │   embedder_id_len: u16        (2 bytes)   │
//! │   embedder_id: [u8]           (variable)  │
//! │   embedder_revision_len: u16  (2 bytes)   │
//! │   embedder_revision: [u8]     (variable)  │
//! │   dimension: u32              (4 bytes)   │
//! │   quantization: u8            (1 byte)    │
//! │   reserved: [u8; 3]           (3 bytes)   │
//! │   record_count: u64           (8 bytes)   │
//! │   vectors_offset: u64         (8 bytes)   │
//! │   header_crc32: u32           (4 bytes)   │
//! ├───────────────────────────────────────────┤
//! │ Record Table                              │
//! │   record_count × 16 bytes each:           │
//! │     doc_id_hash: u64          (8 bytes)   │
//! │     doc_id_offset: u32        (4 bytes)   │
//! │     doc_id_len: u16           (2 bytes)   │
//! │     flags: u16                (2 bytes)   │
//! ├───────────────────────────────────────────┤
//! │ String Table                              │
//! │   Concatenated UTF-8 doc_id strings       │
//! ├───────────────────────────────────────────┤
//! │ Padding (to 64-byte alignment)            │
//! ├───────────────────────────────────────────┤
//! │ Vector Slab                               │
//! │   record_count × dimension × elem_size    │
//! │   (2 bytes/elem for f16, 4 for f32)       │
//! └───────────────────────────────────────────┘
//! ```

#[cfg(feature = "ann")]
pub mod hnsw;
pub mod mrl;
pub mod quantization;
mod repro_soft_delete_rollback;
mod repro_wal_truncation;
pub mod search;
pub mod simd;
pub mod two_tier;
pub mod wal;
pub mod warmup;

use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crc32fast::Hasher as Crc32;
use frankensearch_core::{SearchError, SearchResult};
use half::f16;
use memmap2::MmapMut;
use tracing::debug;

#[cfg(feature = "ann")]
pub use hnsw::{
    AnnSearchStats, HNSW_DEFAULT_EF_CONSTRUCTION, HNSW_DEFAULT_EF_SEARCH, HNSW_DEFAULT_M,
    HNSW_DEFAULT_MAX_LAYER, HnswConfig, HnswIndex,
};
pub use mrl::{MrlConfig, MrlSearchStats};
pub use quantization::ScalarQuantizer;
pub use search::{PARALLEL_CHUNK_SIZE, PARALLEL_THRESHOLD, SearchParams};
pub use simd::{
    cosine_similarity_f16, dot_product_f16_bytes_f32, dot_product_f16_f32,
    dot_product_f32_bytes_f32, dot_product_f32_f32,
};
pub use two_tier::{
    TwoTierIndex, TwoTierIndexBuilder, VECTOR_INDEX_FALLBACK_FILENAME, VECTOR_INDEX_FAST_FILENAME,
    VECTOR_INDEX_QUALITY_FILENAME,
};
#[cfg(feature = "ann")]
pub use two_tier::{VECTOR_ANN_FAST_FILENAME, VECTOR_ANN_QUALITY_FILENAME};
pub use wal::{CompactionStats, WalConfig, wal_path_for};
pub use warmup::{AdaptiveConfig, HeatMap, WarmUpConfig, WarmUpResult, WarmUpStrategy};

/// Magic bytes at the start of every FSVI file.
pub const FSVI_MAGIC: [u8; 4] = *b"FSVI";

/// Supported FSVI format version.
pub const FSVI_VERSION: u16 = 1;

const RECORD_SIZE_BYTES: usize = 16;
const VECTOR_ALIGN_BYTES: u64 = 64;
const RECORD_FLAG_TOMBSTONE: u16 = 0x0001;
const TOMBSTONE_VACUUM_THRESHOLD: f64 = 0.20;

/// Vector element quantization stored in the FSVI slab.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Quantization {
    /// Full-precision float32.
    F32 = 0,
    /// Half-precision float16.
    F16 = 1,
}

impl Quantization {
    pub(crate) fn from_wire(value: u8, path: &Path) -> SearchResult<Self> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            _ => Err(index_corrupted(
                path,
                format!("unsupported quantization byte: {value}"),
            )),
        }
    }

    const fn bytes_per_element(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
        }
    }
}

/// Parsed metadata from an FSVI file header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorMetadata {
    /// Stable embedder id used to build the index.
    pub embedder_id: String,
    /// Model revision identifier (e.g. pinned commit hash).
    pub embedder_revision: String,
    /// Vector dimensionality.
    pub dimension: usize,
    /// Stored quantization.
    pub quantization: Quantization,
    /// Compaction generation counter (0-255) used for stale WAL detection.
    pub compaction_gen: u8,
    /// Number of records in the index.
    pub record_count: usize,
    /// Byte offset to the aligned vector slab.
    pub vectors_offset: u64,
}

/// Statistics returned by [`VectorIndex::vacuum`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VacuumStats {
    /// Records in the main index before vacuum.
    pub records_before: usize,
    /// Records in the main index after vacuum.
    pub records_after: usize,
    /// Tombstoned records removed by vacuum.
    pub tombstones_removed: usize,
    /// Approximate number of bytes reclaimed in the main index file.
    pub bytes_reclaimed: usize,
    /// Time taken by the vacuum operation.
    pub duration: Duration,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RecordEntry {
    pub(crate) doc_id_hash: u64,
    pub(crate) doc_id_offset: u32,
    pub(crate) doc_id_len: u16,
    pub(crate) flags: u16,
}

#[derive(Debug)]
pub struct VectorIndex {
    pub(crate) path: PathBuf,
    pub(crate) data: MmapMut,
    pub(crate) metadata: VectorMetadata,
    pub(crate) records_offset: usize,
    pub(crate) strings_offset: usize,
    pub(crate) vectors_offset: usize,
    /// WAL entries for incremental updates (empty if no WAL exists).
    pub(crate) wal_entries: Vec<wal::WalEntry>,
    /// WAL configuration.
    wal_config: WalConfig,
}

impl VectorIndex {
    /// Open an existing FSVI index from disk.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::IndexNotFound` if the file does not exist and
    /// `SearchError::IndexCorrupted` when header/layout validation fails.
    #[allow(unsafe_code, clippy::too_many_lines)] // MmapMut::map_mut requires unsafe for memory-mapped I/O.
    pub fn open(path: &Path) -> SearchResult<Self> {
        if !path.exists() {
            return Err(SearchError::IndexNotFound {
                path: path.to_path_buf(),
            });
        }

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(SearchError::Io)?;
        let data = unsafe { MmapMut::map_mut(&file).map_err(SearchError::Io)? };
        let (metadata, header_len) = parse_header(path, &data)?;

        let records_bytes = metadata
            .record_count
            .checked_mul(RECORD_SIZE_BYTES)
            .ok_or_else(|| index_corrupted(path, "record table size overflow"))?;
        let records_offset = header_len;
        let strings_offset = records_offset
            .checked_add(records_bytes)
            .ok_or_else(|| index_corrupted(path, "record table offset overflow"))?;
        let vectors_offset = usize::try_from(metadata.vectors_offset)
            .map_err(|_| index_corrupted(path, "vectors_offset does not fit in usize"))?;
        if vectors_offset < strings_offset {
            return Err(index_corrupted(
                path,
                "vectors_offset points inside the record table/string table region",
            ));
        }

        let vector_bytes = metadata
            .record_count
            .checked_mul(metadata.dimension)
            .and_then(|v| v.checked_mul(metadata.quantization.bytes_per_element()))
            .ok_or_else(|| index_corrupted(path, "vector slab size overflow"))?;
        let required_len = vectors_offset
            .checked_add(vector_bytes)
            .ok_or_else(|| index_corrupted(path, "vector slab end overflow"))?;
        if data.len() < required_len {
            return Err(index_corrupted(
                path,
                format!(
                    "truncated file: have {} bytes, need at least {} bytes",
                    data.len(),
                    required_len
                ),
            ));
        }

        let warm_up_config = WarmUpConfig::from_env();
        if !matches!(warm_up_config.strategy, WarmUpStrategy::None) {
            let warm_up = warmup::warm_up_bytes(&data, header_len, &warm_up_config, None);
            debug!(
                target: "frankensearch.warmup",
                path = %path.display(),
                strategy = %warm_up.strategy_name,
                pages_touched = warm_up.pages_touched,
                bytes_touched = warm_up.bytes_touched,
                budget_exhausted = warm_up.budget_exhausted,
                "index warm-up complete"
            );
        }

        // Load WAL entries if a sidecar file exists.
        let wal_path = wal::wal_path_for(path);
        let (wal_entries_raw, wal_compaction_gen, valid_len) =
            wal::read_wal(&wal_path, metadata.dimension, metadata.quantization)?;

        let mut deduped_wal = Vec::with_capacity(wal_entries_raw.len());
        let mut seen_ids = std::collections::HashSet::new();
        for entry in wal_entries_raw.into_iter().rev() {
            if seen_ids.insert(entry.doc_id.clone()) {
                deduped_wal.push(entry);
            }
        }
        deduped_wal.reverse();
        let mut wal_entries = deduped_wal;

        let is_stale = if valid_len > 0 {
            if wal_compaction_gen == 0 {
                metadata.compaction_gen > 0
            } else {
                let expected = next_generation(metadata.compaction_gen);
                wal_compaction_gen != expected
            }
        } else {
            false
        };

        if is_stale {
            tracing::warn!(
                path = %path.display(),
                main_gen = metadata.compaction_gen,
                wal_gen = wal_compaction_gen,
                "discarding stale/mismatched WAL entries and removing file"
            );
            wal_entries.clear();
            if wal_path.exists() {
                let _ = std::fs::remove_file(&wal_path);
            }
        } else if wal_path.exists() {
            let actual_len = std::fs::metadata(&wal_path).map_err(SearchError::Io)?.len();
            if actual_len > valid_len {
                tracing::warn!(
                    path = %wal_path.display(),
                    actual_len,
                    valid_len,
                    "truncating corrupted WAL trailer"
                );
                let file = OpenOptions::new()
                    .write(true)
                    .open(&wal_path)
                    .map_err(SearchError::Io)?;
                file.set_len(valid_len).map_err(SearchError::Io)?;
                file.sync_all().map_err(SearchError::Io)?;
            }
        }

        Ok(Self {
            path: path.to_path_buf(),
            data,
            metadata,
            records_offset,
            strings_offset,
            vectors_offset,
            wal_entries,
            wal_config: WalConfig::default(),
        })
    }

    /// Create a writer that stores vectors as f16 with an empty revision string.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` when arguments are invalid
    /// (for example, zero dimension or oversized header fields).
    pub fn create(
        path: &Path,
        embedder_id: &str,
        dimension: usize,
    ) -> SearchResult<VectorIndexWriter> {
        Self::create_with_revision(path, embedder_id, "", dimension, Quantization::F16)
    }

    /// Create a writer with explicit embedder revision and quantization.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` when arguments are invalid
    /// (for example, zero dimension or oversized header fields).
    pub fn create_with_revision(
        path: &Path,
        embedder_id: &str,
        embedder_revision: &str,
        dimension: usize,
        quantization: Quantization,
    ) -> SearchResult<VectorIndexWriter> {
        if dimension == 0 {
            return Err(SearchError::InvalidConfig {
                field: "dimension".to_owned(),
                value: "0".to_owned(),
                reason: "dimension must be greater than zero".to_owned(),
            });
        }
        validate_header_string(embedder_id, "embedder_id")?;
        validate_header_string(embedder_revision, "embedder_revision")?;
        let _ = u32::try_from(dimension).map_err(|_| SearchError::InvalidConfig {
            field: "dimension".to_owned(),
            value: dimension.to_string(),
            reason: "dimension must fit in u32 for FSVI header encoding".to_owned(),
        })?;

        Ok(VectorIndexWriter {
            path: path.to_path_buf(),
            embedder_id: embedder_id.to_owned(),
            embedder_revision: embedder_revision.to_owned(),
            dimension,
            quantization,
            compaction_gen: 1,
            records: Vec::new(),
        })
    }

    /// Number of vectors in this index.
    #[must_use]
    pub const fn record_count(&self) -> usize {
        self.metadata.record_count
    }

    /// Embedding dimensionality.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.metadata.dimension
    }

    /// Embedder id stored in the index header.
    #[must_use]
    pub fn embedder_id(&self) -> &str {
        &self.metadata.embedder_id
    }

    /// Embedder revision stored in the index header.
    #[must_use]
    pub fn embedder_revision(&self) -> &str {
        &self.metadata.embedder_revision
    }

    /// Stored quantization.
    #[must_use]
    pub const fn quantization(&self) -> Quantization {
        self.metadata.quantization
    }

    /// Full parsed metadata.
    #[must_use]
    pub const fn metadata(&self) -> &VectorMetadata {
        &self.metadata
    }

    // ─── WAL / Incremental Update API ───────────────────────────────────

    /// Set the WAL configuration for incremental updates.
    pub const fn set_wal_config(&mut self, config: WalConfig) {
        self.wal_config = config;
    }

    /// Number of entries in the write-ahead log (pending compaction).
    #[must_use]
    pub const fn wal_record_count(&self) -> usize {
        self.wal_entries.len()
    }

    /// Whether the WAL is large enough that compaction is recommended.
    ///
    /// Returns `true` when the WAL exceeds either the absolute threshold
    /// or the ratio threshold relative to the main index size.
    #[must_use]
    pub fn needs_compaction(&self) -> bool {
        if self.wal_entries.is_empty() {
            return false;
        }
        if self.wal_entries.len() >= self.wal_config.compaction_threshold {
            return true;
        }
        if self.record_count() > 0 {
            #[allow(clippy::cast_precision_loss)]
            let ratio = self.wal_entries.len() as f64 / self.record_count() as f64;
            // NaN compaction_ratio makes >= always false, silently disabling
            // ratio-based compaction. Fall back to the default.
            let threshold = if self.wal_config.compaction_ratio.is_finite() {
                self.wal_config.compaction_ratio
            } else {
                0.10
            };
            if ratio >= threshold {
                return true;
            }
        }
        false
    }

    /// Tombstone (soft-delete) a document by `doc_id`.
    ///
    /// Returns `Ok(true)` when a live record was marked deleted, and `Ok(false)`
    /// when the document does not exist or is already tombstoned.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::Io` for filesystem write/sync failures and
    /// `SearchError::IndexCorrupted` if the on-disk record table is malformed.
    pub fn soft_delete(&mut self, doc_id: &str) -> SearchResult<bool> {
        self.soft_delete_batch(&[doc_id]).map(|count| count > 0)
    }

    /// Tombstone a batch of document ids.
    ///
    /// Returns the number of records that transitioned from live -> deleted.
    ///
    /// # Errors
    ///
    /// Returns the first IO/corruption error encountered while updating flags.
    pub fn soft_delete_batch(&mut self, doc_ids: &[&str]) -> SearchResult<usize> {
        let mut deleted = 0usize;
        let mut wal_changed = false;
        
        // Track modified main index entries for potential rollback
        let mut modified_main_entries = Vec::new();

        // Use a fast lookup for WAL entries to delete
        let mut to_delete_set = std::collections::HashSet::with_capacity(doc_ids.len());
        for &id in doc_ids {
            to_delete_set.insert(id);
        }

        // 1. Mark all matching records in the main index as tombstoned.
        for &doc_id in doc_ids {
            let doc_id_hash = fnv1a_hash(doc_id.as_bytes());
            if let Some(mut index) = self.find_first_hash_match(doc_id_hash)? {
                while index > 0 {
                    let prev = self.record_at(index - 1)?;
                    if prev.doc_id_hash != doc_id_hash {
                        break;
                    }
                    index -= 1;
                }

                for candidate in index..self.record_count() {
                    let entry = self.record_at(candidate)?;
                    if entry.doc_id_hash != doc_id_hash {
                        break;
                    }
                    if !is_tombstoned_flags(entry.flags) {
                        let candidate_doc_id = self.doc_id_at(candidate)?;
                        if candidate_doc_id == doc_id {
                            let flags = entry.flags | RECORD_FLAG_TOMBSTONE;
                            self.set_record_flags(candidate, flags)?;
                            modified_main_entries.push((candidate, entry.flags));
                            deleted += 1;
                        }
                    }
                }
            }
        }

        // 2. Remove all matching records from WAL entries.
        let original_wal_len = self.wal_entries.len();
        let filtered: Vec<wal::WalEntry> = self
            .wal_entries
            .iter()
            .filter(|entry| !to_delete_set.contains(entry.doc_id.as_str()))
            .cloned()
            .collect();
            
        let mut prev_wal = Vec::new();
        if filtered.len() < original_wal_len {
            deleted += original_wal_len - filtered.len();
            wal_changed = true;
            prev_wal = std::mem::replace(&mut self.wal_entries, filtered);
        }

        // 3. Rewrite WAL sidecar once if anything was removed.
        if wal_changed {
            if let Err(err) = self.rewrite_wal_sidecar() {
                self.wal_entries = prev_wal;
                // Rollback main index modifications
                for (candidate, original_flags) in modified_main_entries {
                    if let Err(rollback_err) = self.set_record_flags(candidate, original_flags) {
                        tracing::error!(
                            error = %rollback_err,
                            candidate,
                            "failed to rollback main index flag during soft_delete_batch failure"
                        );
                    }
                }
                tracing::error!(
                    error = %err,
                    "failed to rewrite WAL sidecar during batch delete"
                );
                return Err(err);
            }
        }

        Ok(deleted)
    }

    /// Whether the record at `record_index` is tombstoned.
    #[must_use]
    pub fn is_deleted(&self, record_index: usize) -> bool {
        matches!(
            self.record_at(record_index),
            Ok(entry) if is_tombstoned_flags(entry.flags)
        )
    }

    /// Number of tombstoned records in the main index.
    #[must_use]
    pub fn tombstone_count(&self) -> usize {
        (0..self.record_count())
            .filter(|&index| self.is_deleted(index))
            .count()
    }

    /// Fraction of records that are tombstoned (`tombstones / record_count`).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn tombstone_ratio(&self) -> f64 {
        if self.record_count() == 0 {
            return 0.0;
        }
        self.tombstone_count() as f64 / self.record_count() as f64
    }

    /// Whether the tombstone ratio exceeds the default vacuum threshold.
    #[must_use]
    pub fn needs_vacuum(&self) -> bool {
        self.tombstone_ratio() > TOMBSTONE_VACUUM_THRESHOLD
    }

    /// Rewrite the main index file without tombstoned records.
    ///
    /// WAL entries are preserved and reloaded after the rewrite.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::Io` for filesystem failures and
    /// `SearchError::IndexCorrupted` for malformed data.
    pub fn vacuum(&mut self) -> SearchResult<VacuumStats> {
        let start = Instant::now();
        let records_before = self.record_count();
        let bytes_before = self.data.len();
        let tombstones_before = self.tombstone_count();

        if records_before == 0 || tombstones_before == 0 {
            return Ok(VacuumStats {
                records_before,
                records_after: records_before,
                tombstones_removed: 0,
                bytes_reclaimed: 0,
                duration: start.elapsed(),
            });
        }

        // Collect live entries from main index.
        let mut sources = Vec::with_capacity(records_before - tombstones_before);
        for index in 0..records_before {
            if !self.is_deleted(index) {
                sources.push(MergeSource::Main(index));
            }
        }

        self.rewrite_index(&sources, self.metadata.compaction_gen)?;

        let records_after = self.record_count();
        let bytes_reclaimed = bytes_before.saturating_sub(self.data.len());
        Ok(VacuumStats {
            records_before,
            records_after,
            tombstones_removed: records_before.saturating_sub(records_after),
            bytes_reclaimed,
            duration: start.elapsed(),
        })
    }

    /// Append a single vector to the index via the WAL.
    ///
    /// The vector is immediately searchable. It is written to the WAL
    /// sidecar file for crash safety.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` for wrong embedding lengths
    /// and `SearchError::Io` for filesystem failures.
    pub fn append(&mut self, doc_id: &str, vector: &[f32]) -> SearchResult<()> {
        self.append_batch(&[(doc_id.to_owned(), vector.to_vec())])
    }

    /// Append a batch of vectors to the index via the WAL.
    ///
    /// All vectors in the batch are written atomically to a single WAL
    /// batch (one CRC covers the whole batch).
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` for wrong embedding lengths,
    /// `SearchError::InvalidConfig` for invalid values, and
    /// `SearchError::Io` for filesystem failures.
    pub fn append_batch(&mut self, entries: &[(String, Vec<f32>)]) -> SearchResult<()> {
        if entries.is_empty() {
            return Ok(());
        }

        // Validate all entries before writing anything.
        for (doc_id, vector) in entries {
            if vector.len() != self.dimension() {
                return Err(SearchError::DimensionMismatch {
                    expected: self.dimension(),
                    found: vector.len(),
                });
            }
            if vector.iter().any(|v| !v.is_finite()) {
                return Err(SearchError::InvalidConfig {
                    field: "embedding".to_owned(),
                    value: "<contains non-finite values>".to_owned(),
                    reason: "all embedding values must be finite".to_owned(),
                });
            }
            let _ = u16::try_from(doc_id.len()).map_err(|_| SearchError::InvalidConfig {
                field: "doc_id".to_owned(),
                value: doc_id.clone(),
                reason: "doc_id byte length must fit in u16".to_owned(),
            })?;
        }

        let wal_entries: Vec<wal::WalEntry> = entries
            .iter()
            .map(|(doc_id, embedding)| wal::WalEntry {
                doc_id: doc_id.clone(),
                doc_id_hash: fnv1a_hash(doc_id.as_bytes()),
                embedding: embedding.clone(),
            })
            .collect();

        // Write to WAL file.
        let wal_path = wal::wal_path_for(&self.path);
        wal::append_wal_batch(
            &wal_path,
            &wal_entries,
            self.dimension(),
            self.quantization(),
            next_generation(self.metadata.compaction_gen),
            self.wal_config.fsync_on_write,
        )?;

        // Deduplicate existing WAL entries by doc_id before extending.
        for new_entry in &wal_entries {
            self.wal_entries.retain(|existing| existing.doc_id != new_entry.doc_id);
        }
        // Add to in-memory entries (immediately searchable).
        self.wal_entries.extend(wal_entries);

        debug!(
            target: "frankensearch.index",
            path = %self.path.display(),
            batch_size = entries.len(),
            wal_total = self.wal_entries.len(),
            "appended to WAL"
        );
        Ok(())
    }

    /// Compact the WAL into the main index.
    ///
    /// Rewrites the main index file with all main + WAL records merged,
    /// then removes the WAL sidecar. The index is atomically swapped
    /// (write to tmp, rename over original).
    ///
    /// # Errors
    ///
    /// Returns `SearchError::Io` for filesystem failures and
    /// `SearchError::InvalidConfig` for encoding issues.
    #[allow(clippy::cast_precision_loss)]
    pub fn compact(&mut self) -> SearchResult<CompactionStats> {
        let start = Instant::now();
        let main_before = self.record_count();
        let wal_count = self.wal_entries.len();

        if wal_count == 0 {
            return Ok(CompactionStats {
                main_records_before: main_before,
                wal_records: 0,
                total_records_after: main_before,
                elapsed_ms: 0.0,
            });
        }

        // Collect all sources.
        let mut sources = Vec::with_capacity(main_before + wal_count);
        for i in 0..main_before {
            if !self.is_deleted(i) {
                sources.push(MergeSource::Main(i));
            }
        }
        for (idx, _) in self.wal_entries.iter().enumerate() {
            sources.push(MergeSource::Wal(idx));
        }

        // Sort to ensure binary search property.
        // We can't use sort_by_key easily because we need `self` for Main lookups.
        sources.sort_by(|a, b| {
            let (hash_a, id_a) = self.resolve_sort_key(a);
            let (hash_b, id_b) = self.resolve_sort_key(b);
            hash_a.cmp(&hash_b).then(id_a.cmp(id_b))
        });

        // Perform the rewrite.
        self.rewrite_index(&sources, next_generation(self.metadata.compaction_gen))?;

        // Remove WAL sidecar.
        let wal_path = wal::wal_path_for(&self.path);
        wal::remove_wal(&wal_path)?;

        // Clear WAL entries in memory (they are now in Main).
        self.wal_entries.clear();

        let elapsed = start.elapsed();
        let stats = CompactionStats {
            main_records_before: main_before,
            wal_records: wal_count,
            total_records_after: self.record_count(),
            elapsed_ms: elapsed.as_secs_f64() * 1000.0,
        };

        debug!(
            target: "frankensearch.index",
            path = %self.path.display(),
            main_before,
            wal_count,
            total_after = stats.total_records_after,
            elapsed_ms = format_args!("{:.1}", stats.elapsed_ms),
            "compaction complete"
        );
        Ok(stats)
    }

    fn resolve_sort_key<'a>(&'a self, source: &MergeSource) -> (u64, &'a str) {
        match source {
            MergeSource::Main(idx) => {
                // These unwraps are safe because we only create Main(idx) for valid indices
                // and the index is immutable during compaction.
                let entry = self
                    .record_at(*idx)
                    .expect("index corrupted during compaction");
                let id = self
                    .doc_id_at(*idx)
                    .expect("index corrupted during compaction");
                (entry.doc_id_hash, id)
            }
            MergeSource::Wal(idx) => {
                let entry = &self.wal_entries[*idx];
                (entry.doc_id_hash, &entry.doc_id)
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    fn rewrite_index(&mut self, sources: &[MergeSource], new_gen: u8) -> SearchResult<()> {
        let record_count = sources.len();
        let records_bytes = record_count.checked_mul(RECORD_SIZE_BYTES).ok_or_else(|| {
            SearchError::InvalidConfig {
                field: "record_count".to_owned(),
                value: record_count.to_string(),
                reason: "record table size overflow".to_owned(),
            }
        })?;
        let records_bytes_u64 = u64::try_from(records_bytes).unwrap(); // fits if usize fits

        // Pass 1: Build Record Table and calculate layout.
        // We buffer the Record Table in memory (16 bytes * N).
        // 10M records = 160MB, which is acceptable.
        let mut record_table = Vec::with_capacity(records_bytes);
        let mut current_string_offset = 0u32;
        let mut string_table_len = 0u64;

        for source in sources {
            let (doc_id_hash, doc_id) = self.resolve_sort_key(source);
            let doc_id_len = doc_id.len();

            // Validation
            let len_u16 = u16::try_from(doc_id_len).map_err(|_| SearchError::InvalidConfig {
                field: "doc_id_len".to_owned(),
                value: doc_id_len.to_string(),
                reason: "doc_id length exceeds u16".to_owned(),
            })?;
            let len_u32 = u32::from(len_u16);
            let len_u64 = u64::from(len_u16);
            if current_string_offset.checked_add(len_u32).is_none() {
                return Err(SearchError::InvalidConfig {
                    field: "doc_id_offset".to_owned(),
                    value: "overflow".to_owned(),
                    reason: "string table offset exceeds u32".to_owned(),
                });
            }

            // Append to record table
            record_table.extend_from_slice(&doc_id_hash.to_le_bytes());
            record_table.extend_from_slice(&current_string_offset.to_le_bytes());
            record_table.extend_from_slice(&len_u16.to_le_bytes());
            record_table.extend_from_slice(&0u16.to_le_bytes()); // Flags cleared (tombstones gone)

            current_string_offset += len_u32;
            string_table_len += len_u64;
        }

        // Calculate layout
        let provisional_header = build_header_prefix(
            &self.metadata.embedder_id,
            &self.metadata.embedder_revision,
            self.dimension(),
            self.quantization(),
            new_gen,
            record_count,
            0,
        )?;
        let header_len = provisional_header.len() + 4; // + CRC
        let header_len_u64 = u64::try_from(header_len).unwrap();

        let pre_vector = header_len_u64
            .checked_add(records_bytes_u64)
            .and_then(|v| v.checked_add(string_table_len))
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "layout".to_owned(),
                value: "overflow".to_owned(),
                reason: "layout offset overflow".to_owned(),
            })?;

        let vectors_offset = align_up(pre_vector, VECTOR_ALIGN_BYTES)?;
        let padding_len = usize::try_from(vectors_offset - pre_vector).map_err(|_| {
            SearchError::InvalidConfig {
                field: "padding_len".to_owned(),
                value: (vectors_offset - pre_vector).to_string(),
                reason: "padding length exceeds usize".to_owned(),
            }
        })?;

        // Open temp file
        let tmp_path = temporary_output_path(&self.path);
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&tmp_path)?;
        {
            let mut writer = BufWriter::with_capacity(256 * 1024, &mut file);

            // Pass 2: Write Header and Record Table
            let mut header_prefix = build_header_prefix(
                &self.metadata.embedder_id,
                &self.metadata.embedder_revision,
                self.dimension(),
                self.quantization(),
                new_gen,
                record_count,
                vectors_offset,
            )?;
            let header_crc = crc32(&header_prefix);
            header_prefix.extend_from_slice(&header_crc.to_le_bytes());

            writer.write_all(&header_prefix)?;
            writer.write_all(&record_table)?;

            // Pass 3: Write String Table
            for source in sources {
                let (_, doc_id) = self.resolve_sort_key(source);
                writer.write_all(doc_id.as_bytes())?;
            }

            // Padding
            if padding_len > 0 {
                writer.write_all(&vec![0u8; padding_len])?;
            }

            // Pass 4: Write Vectors
            match self.quantization() {
                Quantization::F16 => {
                    for source in sources {
                        match source {
                            MergeSource::Main(idx) => {
                                // Fast path: copy raw bytes
                                let start = self.vector_start(*idx)?;
                                let len = self.dimension() * 2;
                                let bytes = &self.data[start..start + len];
                                writer.write_all(bytes)?;
                            }
                            MergeSource::Wal(idx) => {
                                // Slow path: encode
                                let entry = &self.wal_entries[*idx];
                                for &val in &entry.embedding {
                                    writer.write_all(&f16::from_f32(val).to_le_bytes())?;
                                }
                            }
                        }
                    }
                }
                Quantization::F32 => {
                    for source in sources {
                        match source {
                            MergeSource::Main(idx) => {
                                // Fast path: copy raw bytes
                                let start = self.vector_start(*idx)?;
                                let len = self.dimension() * 4;
                                let bytes = &self.data[start..start + len];
                                writer.write_all(bytes)?;
                            }
                            MergeSource::Wal(idx) => {
                                // Slow path: encode
                                let entry = &self.wal_entries[*idx];
                                for &val in &entry.embedding {
                                    writer.write_all(&val.to_le_bytes())?;
                                }
                            }
                        }
                    }
                }
            }
            writer.flush()?;
        }

        file.sync_all()?;
        fs::rename(&tmp_path, &self.path)?;
        sync_parent_directory(&self.path)?;

        // Reload
        let config = self.wal_config.clone();
        let reloaded = Self::open(&self.path)?;
        self.data = reloaded.data;
        self.metadata = reloaded.metadata;
        self.records_offset = reloaded.records_offset;
        self.strings_offset = reloaded.strings_offset;
        self.vectors_offset = reloaded.vectors_offset;
        // WAL entries are cleared by caller if compacting, or preserved if vacuuming
        // But vacuum preserves WAL on disk, so open() loads them.
        // Vacuum caller ignores the reloaded WAL entries? No, vacuum preserves them.
        // self.vacuum() impl:
        //   writer.finish()
        //   Self::open() -> loads WAL entries
        //   self.wal_entries = reloaded.wal_entries
        // So we need to update self.wal_entries from reloaded.
        self.wal_entries = reloaded.wal_entries;
        self.wal_config = config;

        Ok(())
    }

    /// Resolve the document id at `index`.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` for out-of-range indices and
    /// `SearchError::IndexCorrupted` for malformed record/string tables.
    pub fn doc_id_at(&self, index: usize) -> SearchResult<&str> {
        self.ensure_index(index)?;
        let entry = self.record_at(index)?;
        let doc_id_offset = usize::try_from(entry.doc_id_offset).map_err(|_| {
            index_corrupted(
                &self.path,
                format!("doc_id_offset overflow for record at index {index}"),
            )
        })?;
        let doc_id_len = usize::from(entry.doc_id_len);
        let start = self
            .strings_offset
            .checked_add(doc_id_offset)
            .ok_or_else(|| index_corrupted(&self.path, "doc_id start offset overflow"))?;
        let end = start
            .checked_add(doc_id_len)
            .ok_or_else(|| index_corrupted(&self.path, "doc_id end offset overflow"))?;
        if end > self.vectors_offset {
            return Err(index_corrupted(
                &self.path,
                format!(
                    "doc_id range [{start}, {end}) exceeds string table end {}",
                    self.vectors_offset
                ),
            ));
        }
        std::str::from_utf8(&self.data[start..end]).map_err(|error| {
            index_corrupted(
                &self.path,
                format!("invalid UTF-8 in doc_id at index {index}: {error}"),
            )
        })
    }

    /// Decode a vector as f32 values.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` for out-of-range indices and
    /// `SearchError::IndexCorrupted` for malformed vector slab data.
    pub fn vector_at_f32(&self, index: usize) -> SearchResult<Vec<f32>> {
        self.ensure_index(index)?;
        let start = self.vector_start(index)?;
        let dim = self.dimension();
        match self.quantization() {
            Quantization::F32 => {
                let byte_len = dim.checked_mul(4).ok_or_else(|| {
                    index_corrupted(&self.path, "f32 vector byte length overflow")
                })?;
                let end = start
                    .checked_add(byte_len)
                    .ok_or_else(|| index_corrupted(&self.path, "f32 vector end overflow"))?;
                if end > self.data.len() {
                    return Err(index_corrupted(
                        &self.path,
                        "f32 vector extends past file end",
                    ));
                }
                let mut out = Vec::with_capacity(dim);
                for chunk in self.data[start..end].chunks_exact(4) {
                    out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                Ok(out)
            }
            Quantization::F16 => {
                let byte_len = dim.checked_mul(2).ok_or_else(|| {
                    index_corrupted(&self.path, "f16 vector byte length overflow")
                })?;
                let end = start
                    .checked_add(byte_len)
                    .ok_or_else(|| index_corrupted(&self.path, "f16 vector end overflow"))?;
                if end > self.data.len() {
                    return Err(index_corrupted(
                        &self.path,
                        "f16 vector extends past file end",
                    ));
                }
                let mut out = Vec::with_capacity(dim);
                for chunk in self.data[start..end].chunks_exact(2) {
                    out.push(f16::from_le_bytes([chunk[0], chunk[1]]).to_f32());
                }
                Ok(out)
            }
        }
    }

    /// Decode a vector as f16 values.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` for out-of-range indices and
    /// `SearchError::IndexCorrupted` for malformed vector slab data.
    pub fn vector_at_f16(&self, index: usize) -> SearchResult<Vec<f16>> {
        self.ensure_index(index)?;
        let start = self.vector_start(index)?;
        let dim = self.dimension();
        match self.quantization() {
            Quantization::F16 => {
                let byte_len = dim.checked_mul(2).ok_or_else(|| {
                    index_corrupted(&self.path, "f16 vector byte length overflow")
                })?;
                let end = start
                    .checked_add(byte_len)
                    .ok_or_else(|| index_corrupted(&self.path, "f16 vector end overflow"))?;
                if end > self.data.len() {
                    return Err(index_corrupted(
                        &self.path,
                        "f16 vector extends past file end",
                    ));
                }
                let mut out = Vec::with_capacity(dim);
                for chunk in self.data[start..end].chunks_exact(2) {
                    out.push(f16::from_le_bytes([chunk[0], chunk[1]]));
                }
                Ok(out)
            }
            Quantization::F32 => {
                let byte_len = dim.checked_mul(4).ok_or_else(|| {
                    index_corrupted(&self.path, "f32 vector byte length overflow")
                })?;
                let end = start
                    .checked_add(byte_len)
                    .ok_or_else(|| index_corrupted(&self.path, "f32 vector end overflow"))?;
                if end > self.data.len() {
                    return Err(index_corrupted(
                        &self.path,
                        "f32 vector extends past file end",
                    ));
                }
                let mut out = Vec::with_capacity(dim);
                for chunk in self.data[start..end].chunks_exact(4) {
                    out.push(f16::from_f32(f32::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3],
                    ])));
                }
                Ok(out)
            }
        }
    }

    /// Binary-search the sorted record table by document hash.
    #[must_use]
    pub fn find_index_by_doc_hash(&self, doc_id_hash: u64) -> Option<usize> {
        let mut low = 0usize;
        let mut high = self.record_count();
        while low < high {
            let mid = low + (high - low) / 2;
            let entry = self.record_at(mid).ok()?;
            match entry.doc_id_hash.cmp(&doc_id_hash) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Greater => high = mid,
                std::cmp::Ordering::Equal => {
                    let mut first = mid;
                    while first > 0 {
                        let prev = self.record_at(first - 1).ok()?;
                        if prev.doc_id_hash != doc_id_hash {
                            break;
                        }
                        first -= 1;
                    }
                    for index in first..self.record_count() {
                        let entry = self.record_at(index).ok()?;
                        if entry.doc_id_hash != doc_id_hash {
                            break;
                        }
                        if !is_tombstoned_flags(entry.flags) {
                            return Some(index);
                        }
                    }
                    return None;
                }
            }
        }
        None
    }

    /// Fetch embeddings for hashed doc ids (f16 values).
    ///
    /// Missing hashes return `None` entries at the same position.
    #[must_use]
    pub fn get_embeddings(&self, doc_id_hashes: &[u64]) -> Vec<Option<Vec<f16>>> {
        doc_id_hashes
            .iter()
            .map(|&hash| {
                if let Some(index) = self.find_index_by_doc_hash(hash) {
                    if let Ok(vec) = self.vector_at_f16(index) {
                        return Some(vec);
                    }
                }
                for entry in &self.wal_entries {
                    if entry.doc_id_hash == hash {
                        // WAL embeddings are f32, we need to convert them to f16
                        return Some(entry.embedding.iter().map(|&v| half::f16::from_f32(v)).collect());
                    }
                }
                None
            })
            .collect()
    }

    fn ensure_index(&self, index: usize) -> SearchResult<()> {
        if index >= self.record_count() {
            return Err(SearchError::InvalidConfig {
                field: "index".to_owned(),
                value: index.to_string(),
                reason: format!(
                    "index out of range for record_count={}",
                    self.record_count()
                ),
            });
        }
        Ok(())
    }

    pub(crate) fn find_index_by_doc_id(&self, doc_id: &str) -> SearchResult<Option<usize>> {
        let doc_id_hash = fnv1a_hash(doc_id.as_bytes());
        let Some(mut index) = self.find_first_hash_match(doc_id_hash)? else {
            return Ok(None);
        };
        while index > 0 {
            let prev = self.record_at(index - 1)?;
            if prev.doc_id_hash != doc_id_hash {
                break;
            }
            index -= 1;
        }

        for candidate in index..self.record_count() {
            let entry = self.record_at(candidate)?;
            if entry.doc_id_hash != doc_id_hash {
                break;
            }
            if !is_tombstoned_flags(entry.flags) {
                let candidate_doc_id = self.doc_id_at(candidate)?;
                if candidate_doc_id == doc_id {
                    return Ok(Some(candidate));
                }
            }
        }
        Ok(None)
    }

    fn find_first_hash_match(&self, doc_id_hash: u64) -> SearchResult<Option<usize>> {
        let mut low = 0usize;
        let mut high = self.record_count();
        while low < high {
            let mid = low + (high - low) / 2;
            let entry = self.record_at(mid)?;
            match entry.doc_id_hash.cmp(&doc_id_hash) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Greater => high = mid,
                std::cmp::Ordering::Equal => return Ok(Some(mid)),
            }
        }
        Ok(None)
    }

    fn record_flags_offset(&self, index: usize) -> SearchResult<usize> {
        self.ensure_index(index)?;
        let record_offset = self
            .records_offset
            .checked_add(index.checked_mul(RECORD_SIZE_BYTES).ok_or_else(|| {
                index_corrupted(&self.path, "record offset multiplication overflow")
            })?)
            .ok_or_else(|| index_corrupted(&self.path, "record offset overflow"))?;
        record_offset
            .checked_add(14)
            .ok_or_else(|| index_corrupted(&self.path, "flags offset overflow"))
    }

    fn set_record_flags(&mut self, index: usize, flags: u16) -> SearchResult<()> {
        let flags_offset = self.record_flags_offset(index)?;
        let end = flags_offset
            .checked_add(2)
            .ok_or_else(|| index_corrupted(&self.path, "flags end overflow"))?;
        if end > self.data.len() {
            return Err(index_corrupted(
                &self.path,
                "flags offset points beyond mapped data",
            ));
        }

        let flag_bytes = flags.to_le_bytes();
        self.data[flags_offset..end].copy_from_slice(&flag_bytes);
        self.data
            .flush_range(flags_offset, 2)
            .map_err(SearchError::Io)?;
        Ok(())
    }

    fn rewrite_wal_sidecar(&self) -> SearchResult<()> {
        let wal_path = wal::wal_path_for(&self.path);
        if self.wal_entries.is_empty() {
            wal::remove_wal(&wal_path)?;
            return Ok(());
        }

        let mut tmp = wal_path.as_os_str().to_os_string();
        tmp.push(".tmp");
        let tmp_path = PathBuf::from(tmp);
        let _ = wal::remove_wal(&tmp_path);

        wal::append_wal_batch(
            &tmp_path,
            &self.wal_entries,
            self.dimension(),
            self.quantization(),
            next_generation(self.metadata.compaction_gen),
            self.wal_config.fsync_on_write,
        )?;

        match fs::rename(&tmp_path, &wal_path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                wal::remove_wal(&wal_path)?;
                fs::rename(&tmp_path, &wal_path)?;
                Ok(())
            }
            Err(error) => {
                let _ = wal::remove_wal(&tmp_path);
                Err(error.into())
            }
        }
    }

    pub(crate) fn record_at(&self, index: usize) -> SearchResult<RecordEntry> {
        self.ensure_index(index)?;
        let offset = self
            .records_offset
            .checked_add(index.checked_mul(RECORD_SIZE_BYTES).ok_or_else(|| {
                index_corrupted(&self.path, "record offset multiplication overflow")
            })?)
            .ok_or_else(|| index_corrupted(&self.path, "record offset overflow"))?;
        let end = offset
            .checked_add(RECORD_SIZE_BYTES)
            .ok_or_else(|| index_corrupted(&self.path, "record end overflow"))?;
        if end > self.data.len() {
            return Err(index_corrupted(
                &self.path,
                "record table extends beyond file size",
            ));
        }
        let chunk = &self.data[offset..end];
        Ok(RecordEntry {
            doc_id_hash: u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]),
            doc_id_offset: u32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]),
            doc_id_len: u16::from_le_bytes([chunk[12], chunk[13]]),
            flags: u16::from_le_bytes([chunk[14], chunk[15]]),
        })
    }

    fn vector_start(&self, index: usize) -> SearchResult<usize> {
        let stride = self
            .dimension()
            .checked_mul(self.quantization().bytes_per_element())
            .ok_or_else(|| index_corrupted(&self.path, "vector stride overflow"))?;
        self.vectors_offset
            .checked_add(
                index
                    .checked_mul(stride)
                    .ok_or_else(|| index_corrupted(&self.path, "vector index overflow"))?,
            )
            .ok_or_else(|| index_corrupted(&self.path, "vector offset overflow"))
    }
}

#[derive(Debug, Clone)]
struct PendingRecord {
    doc_id: String,
    doc_id_hash: u64,
    flags: u16,
    embedding: Vec<f32>,
}

#[derive(Debug, Clone, Copy)]
enum MergeSource {
    Main(usize),
    Wal(usize),
}

#[derive(Debug)]
pub struct VectorIndexWriter {
    path: PathBuf,
    embedder_id: String,
    embedder_revision: String,
    dimension: usize,
    quantization: Quantization,
    compaction_gen: u8,
    records: Vec<PendingRecord>,
}

impl VectorIndexWriter {
    /// Append a single `(doc_id, embedding)` record.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` for wrong embedding lengths
    /// and `SearchError::InvalidConfig` for invalid values.
    pub fn write_record(&mut self, doc_id: &str, embedding: &[f32]) -> SearchResult<()> {
        if embedding.len() != self.dimension {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension,
                found: embedding.len(),
            });
        }
        if embedding.iter().any(|value| !value.is_finite()) {
            return Err(SearchError::InvalidConfig {
                field: "embedding".to_owned(),
                value: "<contains non-finite values>".to_owned(),
                reason: "all embedding values must be finite".to_owned(),
            });
        }
        let _ = u16::try_from(doc_id.len()).map_err(|_| SearchError::InvalidConfig {
            field: "doc_id".to_owned(),
            value: doc_id.to_owned(),
            reason: "doc_id byte length must fit in u16".to_owned(),
        })?;
        self.records.push(PendingRecord {
            doc_id: doc_id.to_owned(),
            doc_id_hash: fnv1a_hash(doc_id.as_bytes()),
            flags: 0,
            embedding: embedding.to_vec(),
        });
        Ok(())
    }

    #[allow(dead_code)]
    pub(crate) const fn with_generation(mut self, generation: u8) -> Self {
        self.compaction_gen = generation;
        self
    }

    /// Persist the index to disk, including fsync of file and parent directory.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` for layout/encoding failures and
    /// `SearchError::Io` for filesystem write/sync failures.
    #[allow(clippy::too_many_lines)]
    pub fn finish(mut self) -> SearchResult<()> {
        self.records.sort_by(|left, right| {
            left.doc_id_hash
                .cmp(&right.doc_id_hash)
                .then(left.doc_id.cmp(&right.doc_id))
        });

        let record_count = self.records.len();
        let records_bytes = record_count.checked_mul(RECORD_SIZE_BYTES).ok_or_else(|| {
            SearchError::InvalidConfig {
                field: "record_count".to_owned(),
                value: record_count.to_string(),
                reason: "record table size overflow".to_owned(),
            }
        })?;
        let records_bytes_u64 =
            u64::try_from(records_bytes).map_err(|_| SearchError::InvalidConfig {
                field: "record_count".to_owned(),
                value: record_count.to_string(),
                reason: "record table size does not fit in u64".to_owned(),
            })?;

        let mut string_table = Vec::<u8>::new();
        let mut record_entries = Vec::<RecordEntry>::with_capacity(record_count);
        for record in &self.records {
            let offset_u32 =
                u32::try_from(string_table.len()).map_err(|_| SearchError::InvalidConfig {
                    field: "doc_id_offset".to_owned(),
                    value: string_table.len().to_string(),
                    reason: "string table offset exceeds u32".to_owned(),
                })?;
            let doc_id_bytes = record.doc_id.as_bytes();
            let len_u16 =
                u16::try_from(doc_id_bytes.len()).map_err(|_| SearchError::InvalidConfig {
                    field: "doc_id_len".to_owned(),
                    value: doc_id_bytes.len().to_string(),
                    reason: "doc_id length exceeds u16".to_owned(),
                })?;
            string_table.extend_from_slice(doc_id_bytes);
            record_entries.push(RecordEntry {
                doc_id_hash: record.doc_id_hash,
                doc_id_offset: offset_u32,
                doc_id_len: len_u16,
                flags: record.flags,
            });
        }

        let string_table_len_u64 =
            u64::try_from(string_table.len()).map_err(|_| SearchError::InvalidConfig {
                field: "string_table".to_owned(),
                value: string_table.len().to_string(),
                reason: "string table length does not fit in u64".to_owned(),
            })?;

        let provisional_header = build_header_prefix(
            &self.embedder_id,
            &self.embedder_revision,
            self.dimension,
            self.quantization,
            self.compaction_gen,
            record_count,
            0,
        )?;
        let header_len =
            provisional_header
                .len()
                .checked_add(4)
                .ok_or_else(|| SearchError::InvalidConfig {
                    field: "header".to_owned(),
                    value: provisional_header.len().to_string(),
                    reason: "header length overflow".to_owned(),
                })?;
        let header_len_u64 = u64::try_from(header_len).map_err(|_| SearchError::InvalidConfig {
            field: "header".to_owned(),
            value: header_len.to_string(),
            reason: "header length does not fit in u64".to_owned(),
        })?;
        let pre_vector = header_len_u64
            .checked_add(records_bytes_u64)
            .and_then(|value| value.checked_add(string_table_len_u64))
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "layout".to_owned(),
                value: format!("{header_len_u64}+{records_bytes_u64}+{string_table_len_u64}"),
                reason: "layout offset overflow".to_owned(),
            })?;
        let vectors_offset = align_up(pre_vector, VECTOR_ALIGN_BYTES)?;
        let padding_len_u64 =
            vectors_offset
                .checked_sub(pre_vector)
                .ok_or_else(|| SearchError::InvalidConfig {
                    field: "layout".to_owned(),
                    value: format!("{vectors_offset}-{pre_vector}"),
                    reason: "negative padding detected".to_owned(),
                })?;
        let padding_len =
            usize::try_from(padding_len_u64).map_err(|_| SearchError::InvalidConfig {
                field: "padding".to_owned(),
                value: padding_len_u64.to_string(),
                reason: "padding length does not fit in usize".to_owned(),
            })?;

        let mut header_prefix = build_header_prefix(
            &self.embedder_id,
            &self.embedder_revision,
            self.dimension,
            self.quantization,
            self.compaction_gen,
            record_count,
            vectors_offset,
        )?;
        let header_crc = crc32(&header_prefix);
        header_prefix.extend_from_slice(&header_crc.to_le_bytes());

        let tmp_path = temporary_output_path(&self.path);
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&tmp_path)?;
        {
            let mut writer = BufWriter::with_capacity(256 * 1024, &mut file);

            writer.write_all(&header_prefix)?;
            for entry in &record_entries {
                writer.write_all(&entry.doc_id_hash.to_le_bytes())?;
                writer.write_all(&entry.doc_id_offset.to_le_bytes())?;
                writer.write_all(&entry.doc_id_len.to_le_bytes())?;
                writer.write_all(&entry.flags.to_le_bytes())?;
            }
            writer.write_all(&string_table)?;
            if padding_len > 0 {
                writer.write_all(&vec![0_u8; padding_len])?;
            }
            write_vector_slab(&mut writer, &self.records, self.quantization)?;
            writer.flush()?;
        }

        file.sync_all()?;
        fs::rename(&tmp_path, &self.path)?;
        sync_parent_directory(&self.path)?;

        debug!(
            target: "frankensearch.index",
            path = %self.path.display(),
            record_count,
            dimension = self.dimension,
            quantization = self.quantization as u8,
            vectors_offset,
            "wrote fsvi index"
        );
        Ok(())
    }
}

fn parse_header(path: &Path, data: &[u8]) -> SearchResult<(VectorMetadata, usize)> {
    let mut cursor = 0usize;
    let magic = read_array::<4>(path, data, &mut cursor, "magic")?;
    if magic != FSVI_MAGIC {
        return Err(index_corrupted(
            path,
            format!("bad magic bytes: expected {FSVI_MAGIC:?}, found {magic:?}"),
        ));
    }

    let version = u16::from_le_bytes(read_array::<2>(path, data, &mut cursor, "version")?);
    if version != FSVI_VERSION {
        return Err(SearchError::IndexVersionMismatch {
            expected: FSVI_VERSION,
            found: version,
        });
    }

    let embedder_id_len = usize::from(u16::from_le_bytes(read_array::<2>(
        path,
        data,
        &mut cursor,
        "embedder_id_len",
    )?));
    let embedder_id_bytes = read_slice(path, data, &mut cursor, embedder_id_len, "embedder_id")?;
    let embedder_id = std::str::from_utf8(embedder_id_bytes)
        .map_err(|error| index_corrupted(path, format!("invalid UTF-8 in embedder_id: {error}")))?
        .to_owned();

    let embedder_revision_len = usize::from(u16::from_le_bytes(read_array::<2>(
        path,
        data,
        &mut cursor,
        "embedder_revision_len",
    )?));
    let embedder_revision_bytes = read_slice(
        path,
        data,
        &mut cursor,
        embedder_revision_len,
        "embedder_revision",
    )?;
    let embedder_revision = std::str::from_utf8(embedder_revision_bytes)
        .map_err(|error| {
            index_corrupted(path, format!("invalid UTF-8 in embedder_revision: {error}"))
        })?
        .to_owned();

    let dimension_u32 = u32::from_le_bytes(read_array::<4>(path, data, &mut cursor, "dimension")?);
    let dimension = usize::try_from(dimension_u32)
        .map_err(|_| index_corrupted(path, "dimension does not fit in usize"))?;
    if dimension == 0 {
        return Err(index_corrupted(path, "dimension must be greater than zero"));
    }

    let quantization_byte = read_array::<1>(path, data, &mut cursor, "quantization")?[0];
    let quantization = Quantization::from_wire(quantization_byte, path)?;

    // Use first reserved byte for compaction generation
    let reserved = read_array::<3>(path, data, &mut cursor, "reserved")?;
    let compaction_gen = reserved[0];
    // reserved[1..2] remain unused

    let record_count_u64 =
        u64::from_le_bytes(read_array::<8>(path, data, &mut cursor, "record_count")?);
    let record_count = usize::try_from(record_count_u64)
        .map_err(|_| index_corrupted(path, "record_count does not fit in usize"))?;
    let vectors_offset =
        u64::from_le_bytes(read_array::<8>(path, data, &mut cursor, "vectors_offset")?);
    let expected_crc =
        u32::from_le_bytes(read_array::<4>(path, data, &mut cursor, "header_crc32")?);
    let actual_crc = crc32(&data[..cursor - 4]);
    if actual_crc != expected_crc {
        return Err(index_corrupted(
            path,
            format!("header CRC mismatch: expected {expected_crc:#010x}, got {actual_crc:#010x}"),
        ));
    }

    Ok((
        VectorMetadata {
            embedder_id,
            embedder_revision,
            dimension,
            quantization,
            compaction_gen,
            record_count,
            vectors_offset,
        },
        cursor,
    ))
}

fn read_array<const N: usize>(
    path: &Path,
    data: &[u8],
    cursor: &mut usize,
    field: &str,
) -> SearchResult<[u8; N]> {
    let slice = read_slice(path, data, cursor, N, field)?;
    let mut out = [0_u8; N];
    out.copy_from_slice(slice);
    Ok(out)
}

fn read_slice<'a>(
    path: &Path,
    data: &'a [u8],
    cursor: &mut usize,
    len: usize,
    field: &str,
) -> SearchResult<&'a [u8]> {
    let end = cursor
        .checked_add(len)
        .ok_or_else(|| index_corrupted(path, format!("{field} offset overflow")))?;
    if end > data.len() {
        return Err(index_corrupted(
            path,
            format!("{field} is truncated (wanted {len} bytes)"),
        ));
    }
    let out = &data[*cursor..end];
    *cursor = end;
    Ok(out)
}

fn build_header_prefix(
    embedder_id: &str,
    embedder_revision: &str,
    dimension: usize,
    quantization: Quantization,
    compaction_gen: u8,
    record_count: usize,
    vectors_offset: u64,
) -> SearchResult<Vec<u8>> {
    validate_header_string(embedder_id, "embedder_id")?;
    validate_header_string(embedder_revision, "embedder_revision")?;
    let dimension_u32 = u32::try_from(dimension).map_err(|_| SearchError::InvalidConfig {
        field: "dimension".to_owned(),
        value: dimension.to_string(),
        reason: "dimension must fit in u32".to_owned(),
    })?;
    let record_count_u64 = u64::try_from(record_count).map_err(|_| SearchError::InvalidConfig {
        field: "record_count".to_owned(),
        value: record_count.to_string(),
        reason: "record_count must fit in u64".to_owned(),
    })?;
    let mut out = Vec::with_capacity(
        4 + 2 + 2 + embedder_id.len() + 2 + embedder_revision.len() + 4 + 1 + 3 + 8 + 8,
    );
    out.extend_from_slice(&FSVI_MAGIC);
    out.extend_from_slice(&FSVI_VERSION.to_le_bytes());
    out.extend_from_slice(
        &u16::try_from(embedder_id.len())
            .map_err(|_| SearchError::InvalidConfig {
                field: "embedder_id".to_owned(),
                value: embedder_id.to_owned(),
                reason: "embedder_id byte length must fit in u16".to_owned(),
            })?
            .to_le_bytes(),
    );
    out.extend_from_slice(embedder_id.as_bytes());
    out.extend_from_slice(
        &u16::try_from(embedder_revision.len())
            .map_err(|_| SearchError::InvalidConfig {
                field: "embedder_revision".to_owned(),
                value: embedder_revision.to_owned(),
                reason: "embedder_revision byte length must fit in u16".to_owned(),
            })?
            .to_le_bytes(),
    );
    out.extend_from_slice(embedder_revision.as_bytes());
    out.extend_from_slice(&dimension_u32.to_le_bytes());
    out.push(quantization as u8);
    out.push(compaction_gen);
    out.extend_from_slice(&[0_u8; 2]);
    out.extend_from_slice(&record_count_u64.to_le_bytes());
    out.extend_from_slice(&vectors_offset.to_le_bytes());
    Ok(out)
}

fn validate_header_string(value: &str, field: &str) -> SearchResult<()> {
    if value.is_empty() && field == "embedder_id" {
        return Err(SearchError::InvalidConfig {
            field: field.to_owned(),
            value: value.to_owned(),
            reason: "embedder_id cannot be empty".to_owned(),
        });
    }
    let _ = u16::try_from(value.len()).map_err(|_| SearchError::InvalidConfig {
        field: field.to_owned(),
        value: value.to_owned(),
        reason: "value length must fit in u16".to_owned(),
    })?;
    Ok(())
}

fn write_vector_slab<W: Write>(
    writer: &mut W,
    records: &[PendingRecord],
    quantization: Quantization,
) -> SearchResult<()> {
    match quantization {
        Quantization::F16 => {
            for record in records {
                for value in &record.embedding {
                    writer.write_all(&f16::from_f32(*value).to_le_bytes())?;
                }
            }
        }
        Quantization::F32 => {
            for record in records {
                for value in &record.embedding {
                    writer.write_all(&value.to_le_bytes())?;
                }
            }
        }
    }
    Ok(())
}

fn align_up(value: u64, alignment: u64) -> SearchResult<u64> {
    if alignment == 0 {
        return Ok(value);
    }
    let add = alignment
        .checked_sub(1)
        .ok_or_else(|| SearchError::InvalidConfig {
            field: "alignment".to_owned(),
            value: alignment.to_string(),
            reason: "alignment underflow".to_owned(),
        })?;
    let padded = value
        .checked_add(add)
        .ok_or_else(|| SearchError::InvalidConfig {
            field: "alignment".to_owned(),
            value: format!("{value}+{add}"),
            reason: "alignment overflow".to_owned(),
        })?;
    Ok((padded / alignment) * alignment)
}

fn temporary_output_path(path: &Path) -> PathBuf {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let pid = std::process::id();
    let mut os = path.as_os_str().to_os_string();
    os.push(format!(".tmp.{pid}.{now}"));
    PathBuf::from(os)
}

fn sync_parent_directory(path: &Path) -> SearchResult<()> {
    #[cfg(unix)]
    {
        if let Some(parent) = path.parent() {
            let dir = File::open(parent)?;
            dir.sync_all()?;
        }
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}

fn index_corrupted(path: &Path, detail: impl Into<String>) -> SearchError {
    SearchError::IndexCorrupted {
        path: path.to_path_buf(),
        detail: detail.into(),
    }
}

fn crc32(data: &[u8]) -> u32 {
    let mut hasher = Crc32::new();
    hasher.update(data);
    hasher.finalize()
}

pub(crate) fn fnv1a_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3_u64);
    }
    hash
}

const fn is_tombstoned_flags(flags: u16) -> bool {
    flags & RECORD_FLAG_TOMBSTONE != 0
}

const fn next_generation(current: u8) -> u8 {
    if current == 255 { 1 } else { current + 1 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_index_path(name: &str) -> PathBuf {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "frankensearch-index-{name}-{}-{now}.fsvi",
            std::process::id()
        ))
    }

    fn sample_vector(base: f32, dim: usize) -> Vec<f32> {
        vec![base; dim]
    }

    #[test]
    fn round_trip_f16_with_revision_and_lookup() {
        let path = temp_index_path("round-trip");
        let mut writer =
            VectorIndex::create_with_revision(&path, "fnv1a-384", "rev-123", 8, Quantization::F16)
                .expect("writer");
        writer
            .write_record("doc-b", &sample_vector(1.0, 8))
            .expect("write doc-b");
        writer
            .write_record("doc-a", &sample_vector(2.0, 8))
            .expect("write doc-a");
        writer.finish().expect("finish");

        let index = VectorIndex::open(&path).expect("open index");
        assert_eq!(index.record_count(), 2);
        assert_eq!(index.dimension(), 8);
        assert_eq!(index.embedder_id(), "fnv1a-384");
        assert_eq!(index.embedder_revision(), "rev-123");
        assert_eq!(index.quantization(), Quantization::F16);
        assert_eq!(index.metadata().vectors_offset % VECTOR_ALIGN_BYTES, 0);

        let hash_a = fnv1a_hash(b"doc-a");
        let pos_a = index
            .find_index_by_doc_hash(hash_a)
            .expect("hash lookup should find doc-a");
        let doc_id = index.doc_id_at(pos_a).expect("doc id");
        assert_eq!(doc_id, "doc-a");
        let vec_a = index.vector_at_f32(pos_a).expect("vector");
        assert_eq!(vec_a.len(), 8);
        assert!((vec_a[0] - 2.0).abs() < 0.002);
    }

    #[test]
    fn detects_header_crc_corruption() {
        let path = temp_index_path("crc");
        let mut writer = VectorIndex::create(&path, "fnv1a-384", 4).expect("writer");
        writer
            .write_record("doc-1", &sample_vector(0.5, 4))
            .expect("write");
        writer.finish().expect("finish");

        let mut bytes = fs::read(&path).expect("read index");
        // Flip a byte in the header payload before crc.
        bytes[6] ^= 0xAA;
        fs::write(&path, bytes).expect("rewrite corrupt index");

        let error = VectorIndex::open(&path).expect_err("corruption should be detected");
        assert!(matches!(error, SearchError::IndexCorrupted { .. }));
    }

    #[test]
    fn write_record_dimension_mismatch_is_error() {
        let path = temp_index_path("dim-mismatch");
        let mut writer = VectorIndex::create(&path, "fnv1a-384", 3).expect("writer");
        let error = writer
            .write_record("doc-1", &[1.0, 2.0])
            .expect_err("must reject wrong dimension");
        assert!(matches!(
            error,
            SearchError::DimensionMismatch {
                expected: 3,
                found: 2
            }
        ));
    }

    #[test]
    fn empty_index_round_trip() {
        let path = temp_index_path("empty");
        let writer = VectorIndex::create(&path, "fnv1a-384", 16).expect("writer");
        writer.finish().expect("finish");

        let index = VectorIndex::open(&path).expect("open");
        assert_eq!(index.record_count(), 0);
        assert_eq!(index.dimension(), 16);
    }

    #[test]
    fn get_embeddings_returns_none_for_missing_hashes() {
        let path = temp_index_path("get-embeddings");
        let mut writer = VectorIndex::create(&path, "fnv1a-384", 4).expect("writer");
        writer
            .write_record("doc-1", &[0.1, 0.2, 0.3, 0.4])
            .expect("write");
        writer.finish().expect("finish");

        let index = VectorIndex::open(&path).expect("open");
        let existing = fnv1a_hash(b"doc-1");
        let missing = fnv1a_hash(b"missing");
        let embeddings = index.get_embeddings(&[existing, missing]);
        assert!(embeddings[0].is_some());
        assert!(embeddings[1].is_none());
        assert_eq!(embeddings[0].as_ref().expect("existing").len(), 4);
    }

    #[test]
    fn soft_delete_marks_record_and_hides_hash_lookup() {
        let path = temp_index_path("soft-delete-main");
        let mut writer = VectorIndex::create(&path, "fnv1a-384", 4).expect("writer");
        writer
            .write_record("doc-a", &[1.0, 0.0, 0.0, 0.0])
            .expect("write doc-a");
        writer
            .write_record("doc-b", &[0.0, 1.0, 0.0, 0.0])
            .expect("write doc-b");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        assert!(index.soft_delete("doc-a").expect("soft delete"));
        assert!(!index.soft_delete("doc-a").expect("idempotent soft delete"));

        let hash_a = fnv1a_hash(b"doc-a");
        let hash_b = fnv1a_hash(b"doc-b");
        assert_eq!(index.find_index_by_doc_hash(hash_a), None);
        assert!(index.find_index_by_doc_hash(hash_b).is_some());
        assert_eq!(index.tombstone_count(), 1);

        let hits = index
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, None)
            .expect("search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "doc-b");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn soft_delete_missing_returns_false() {
        let path = temp_index_path("soft-delete-missing");
        let mut writer = VectorIndex::create(&path, "fnv1a-384", 4).expect("writer");
        writer
            .write_record("doc-a", &[1.0, 0.0, 0.0, 0.0])
            .expect("write");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        assert!(
            !index
                .soft_delete("missing-doc")
                .expect("missing soft delete")
        );
        assert_eq!(index.tombstone_count(), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn soft_delete_batch_counts_only_new_tombstones() {
        let path = temp_index_path("soft-delete-batch");
        let mut writer = VectorIndex::create(&path, "fnv1a-384", 4).expect("writer");
        writer
            .write_record("doc-a", &[1.0, 0.0, 0.0, 0.0])
            .expect("write a");
        writer
            .write_record("doc-b", &[0.0, 1.0, 0.0, 0.0])
            .expect("write b");
        writer
            .write_record("doc-c", &[0.0, 0.0, 1.0, 0.0])
            .expect("write c");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        let deleted = index
            .soft_delete_batch(&["doc-a", "doc-b", "missing", "doc-a"])
            .expect("batch delete");
        assert_eq!(deleted, 2);
        assert_eq!(index.tombstone_count(), 2);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn tombstone_ratio_and_needs_vacuum_threshold() {
        let path = temp_index_path("soft-delete-ratio");
        let mut writer = VectorIndex::create(&path, "fnv1a-384", 4).expect("writer");
        for i in 0..10 {
            writer
                .write_record(&format!("doc-{i}"), &sample_vector(0.1, 4))
                .expect("write");
        }
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        assert!(index.tombstone_ratio().abs() < f64::EPSILON);
        assert!(!index.needs_vacuum());

        index.soft_delete("doc-0").expect("delete 0");
        index.soft_delete("doc-1").expect("delete 1");
        assert_eq!(index.tombstone_count(), 2);
        assert!((index.tombstone_ratio() - 0.2).abs() < f64::EPSILON);
        assert!(!index.needs_vacuum(), "threshold is strict greater-than");

        index.soft_delete("doc-2").expect("delete 2");
        assert_eq!(index.tombstone_count(), 3);
        assert!(index.needs_vacuum());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn vacuum_removes_tombstones_and_preserves_live_results() {
        let path = temp_index_path("soft-delete-vacuum");
        let mut writer = VectorIndex::create(&path, "fnv1a-384", 4).expect("writer");
        writer
            .write_record("doc-a", &[1.0, 0.0, 0.0, 0.0])
            .expect("write a");
        writer
            .write_record("doc-b", &[0.0, 1.0, 0.0, 0.0])
            .expect("write b");
        writer
            .write_record("doc-c", &[0.0, 0.0, 1.0, 0.0])
            .expect("write c");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        index.soft_delete("doc-b").expect("delete b");

        let pre_hits = index
            .search_top_k(&[0.0, 1.0, 0.0, 0.0], 10, None)
            .expect("pre-vacuum search");
        assert_eq!(pre_hits.len(), 2);
        assert!(pre_hits.iter().all(|hit| hit.doc_id != "doc-b"));

        let stats = index.vacuum().expect("vacuum");
        assert_eq!(stats.records_before, 3);
        assert_eq!(stats.records_after, 2);
        assert_eq!(stats.tombstones_removed, 1);
        assert!(stats.bytes_reclaimed > 0);
        assert!(stats.duration >= Duration::ZERO);

        assert_eq!(index.record_count(), 2);
        assert_eq!(index.tombstone_count(), 0);
        assert_eq!(index.find_index_by_doc_hash(fnv1a_hash(b"doc-b")), None);

        let post_hits = index
            .search_top_k(&[0.0, 1.0, 0.0, 0.0], 10, None)
            .expect("post-vacuum search");
        assert_eq!(post_hits.len(), 2);
        assert!(post_hits.iter().all(|hit| hit.doc_id != "doc-b"));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn soft_delete_and_search_interleaving_has_no_corruption() {
        use std::collections::HashSet;
        use std::sync::{Arc, Mutex};

        let path = temp_index_path("soft-delete-concurrent");
        let dim = 4;

        let mut writer = VectorIndex::create(&path, "fnv1a-384", dim).expect("writer");
        for i in 0..128 {
            writer
                .write_record(&format!("doc-{i:03}"), &[1.0, 0.0, 0.0, 0.0])
                .expect("write");
        }
        writer.finish().expect("finish");

        let shared = Arc::new(Mutex::new(VectorIndex::open(&path).expect("open")));
        let deleter = {
            let index = Arc::clone(&shared);
            std::thread::spawn(move || {
                for i in 0..32 {
                    let mut guard = index.lock().expect("lock for delete");
                    let doc_id = format!("doc-{i:03}");
                    let _ = guard.soft_delete(&doc_id).expect("soft delete");
                }
            })
        };

        let query = [1.0, 0.0, 0.0, 0.0];
        let searchers: Vec<_> = (0..4)
            .map(|_| {
                let index = Arc::clone(&shared);
                std::thread::spawn(move || {
                    for _ in 0..32 {
                        let hits = index
                            .lock()
                            .expect("lock for search")
                            .search_top_k(&query, 10, None)
                            .expect("search");
                        assert!(!hits.is_empty());
                    }
                })
            })
            .collect();

        deleter.join().expect("join deleter");
        for handle in searchers {
            handle.join().expect("join searcher");
        }

        let hits = shared
            .lock()
            .expect("lock final")
            .search_top_k(&query, 64, None)
            .expect("final search");
        let deleted_ids: HashSet<String> = (0..32).map(|i| format!("doc-{i:03}")).collect();
        assert!(hits.iter().all(|hit| !deleted_ids.contains(&hit.doc_id)));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn soft_delete_preserves_existing_non_tombstone_flags() {
        let path = temp_index_path("soft-delete-flags");
        let mut writer = VectorIndex::create(&path, "fnv1a-384", 4).expect("writer");
        writer
            .write_record("doc-a", &[1.0, 0.0, 0.0, 0.0])
            .expect("write doc-a");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        let hash_a = fnv1a_hash(b"doc-a");
        let record_index = index
            .find_index_by_doc_hash(hash_a)
            .expect("record index for doc-a");

        let custom_flag: u16 = 0x0004;
        index
            .set_record_flags(record_index, custom_flag)
            .expect("seed custom flag");
        assert_eq!(
            index.record_at(record_index).expect("read flags").flags,
            custom_flag
        );

        assert!(index.soft_delete("doc-a").expect("soft delete doc-a"));
        let flags_after = index.record_at(record_index).expect("read flags").flags;
        assert_eq!(
            flags_after & RECORD_FLAG_TOMBSTONE,
            RECORD_FLAG_TOMBSTONE,
            "tombstone bit must be set",
        );
        assert_eq!(
            flags_after & custom_flag,
            custom_flag,
            "non-tombstone bits must remain untouched",
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn tombstone_flag_persists_after_reopen() {
        let path = temp_index_path("soft-delete-persist");
        let mut writer = VectorIndex::create(&path, "fnv1a-384", 4).expect("writer");
        writer
            .write_record("doc-a", &[1.0, 0.0, 0.0, 0.0])
            .expect("write a");
        writer
            .write_record("doc-b", &[0.0, 1.0, 0.0, 0.0])
            .expect("write b");
        writer.finish().expect("finish");

        {
            let mut index = VectorIndex::open(&path).expect("open for delete");
            assert!(index.soft_delete("doc-a").expect("delete doc-a"));
            assert_eq!(index.tombstone_count(), 1);
        }

        let reopened = VectorIndex::open(&path).expect("reopen");
        assert_eq!(reopened.tombstone_count(), 1);
        assert_eq!(reopened.find_index_by_doc_hash(fnv1a_hash(b"doc-a")), None);
        let hits = reopened
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, None)
            .expect("search after reopen");
        assert!(hits.iter().all(|hit| hit.doc_id != "doc-a"));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn delete_vacuum_append_cycle_keeps_expected_live_set() {
        use std::collections::HashSet;

        let path = temp_index_path("soft-delete-reindex-cycle");
        let dim = 4;

        let mut writer = VectorIndex::create(&path, "fnv1a-384", dim).expect("writer");
        for i in 0..100 {
            writer
                .write_record(&format!("doc-{i:03}"), &[1.0, 0.0, 0.0, 0.0])
                .expect("write initial doc");
        }
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        let delete_ids: Vec<String> = (0..50).map(|i| format!("doc-{i:03}")).collect();
        let delete_refs: Vec<&str> = delete_ids.iter().map(String::as_str).collect();
        let deleted = index.soft_delete_batch(&delete_refs).expect("batch delete");
        assert_eq!(deleted, 50);
        assert_eq!(index.tombstone_count(), 50);

        let vacuum_stats = index.vacuum().expect("vacuum");
        assert_eq!(vacuum_stats.records_before, 100);
        assert_eq!(vacuum_stats.records_after, 50);
        assert_eq!(index.tombstone_count(), 0);
        assert_eq!(index.record_count(), 50);

        let append_entries: Vec<(String, Vec<f32>)> = (100..150)
            .map(|i| (format!("doc-{i:03}"), vec![1.0, 0.0, 0.0, 0.0]))
            .collect();
        index.append_batch(&append_entries).expect("append batch");
        assert_eq!(index.wal_record_count(), 50);

        let compact_stats = index.compact().expect("compact");
        assert_eq!(compact_stats.total_records_after, 100);
        assert_eq!(index.record_count(), 100);
        assert_eq!(index.wal_record_count(), 0);

        let hits = index
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 150, None)
            .expect("search");
        assert_eq!(hits.len(), 100);
        let ids: HashSet<String> = hits.iter().map(|hit| hit.doc_id.clone()).collect();

        for i in 0..50 {
            assert!(
                !ids.contains(&format!("doc-{i:03}")),
                "deleted id must not be present",
            );
        }
        for i in 50..150 {
            assert!(
                ids.contains(&format!("doc-{i:03}")),
                "live id must be present",
            );
        }

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(wal::wal_path_for(&path)).ok();
    }

    #[test]
    fn tombstones_remain_excluded_with_wal_and_after_compaction() {
        let path = temp_index_path("soft-delete-wal-integration");
        let dim = 4;

        let mut writer = VectorIndex::create(&path, "fnv1a-384", dim).expect("writer");
        writer
            .write_record("doc-a", &[1.0, 0.0, 0.0, 0.0])
            .expect("write a");
        writer
            .write_record("doc-b", &[1.0, 0.0, 0.0, 0.0])
            .expect("write b");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        assert!(index.soft_delete("doc-a").expect("delete a"));
        index
            .append("doc-c", &[1.0, 0.0, 0.0, 0.0])
            .expect("append c");
        assert_eq!(index.wal_record_count(), 1);

        let pre_compact = index
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, None)
            .expect("pre-compact search");
        assert_eq!(pre_compact.len(), 2);
        assert!(pre_compact.iter().all(|hit| hit.doc_id != "doc-a"));
        assert!(pre_compact.iter().any(|hit| hit.doc_id == "doc-b"));
        assert!(pre_compact.iter().any(|hit| hit.doc_id == "doc-c"));

        index.compact().expect("compact");
        let post_compact = index
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, None)
            .expect("post-compact search");
        assert_eq!(post_compact.len(), 2);
        assert!(post_compact.iter().all(|hit| hit.doc_id != "doc-a"));
        assert!(post_compact.iter().any(|hit| hit.doc_id == "doc-b"));
        assert!(post_compact.iter().any(|hit| hit.doc_id == "doc-c"));

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(wal::wal_path_for(&path)).ok();
    }

    #[test]
    fn vacuum_noop_when_no_tombstones() {
        let path = temp_index_path("soft-delete-vacuum-noop");
        let mut writer = VectorIndex::create(&path, "fnv1a-384", 4).expect("writer");
        writer
            .write_record("doc-a", &[1.0, 0.0, 0.0, 0.0])
            .expect("write a");
        writer
            .write_record("doc-b", &[0.0, 1.0, 0.0, 0.0])
            .expect("write b");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        assert_eq!(index.tombstone_count(), 0);

        let stats = index.vacuum().expect("vacuum with no tombstones");
        assert_eq!(stats.records_before, 2);
        assert_eq!(stats.records_after, 2);
        assert_eq!(stats.tombstones_removed, 0);
        assert_eq!(index.record_count(), 2);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn soft_delete_all_records_yields_empty_search() {
        let path = temp_index_path("soft-delete-all");
        let mut writer = VectorIndex::create(&path, "fnv1a-384", 4).expect("writer");
        for i in 0..5 {
            writer
                .write_record(&format!("doc-{i}"), &sample_vector(0.1, 4))
                .expect("write");
        }
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        for i in 0..5 {
            assert!(index.soft_delete(&format!("doc-{i}")).expect("delete"));
        }
        assert_eq!(index.tombstone_count(), 5);
        assert!((index.tombstone_ratio() - 1.0).abs() < f64::EPSILON);
        assert!(index.needs_vacuum());

        let hits = index
            .search_top_k(&sample_vector(0.1, 4), 10, None)
            .expect("search");
        assert!(
            hits.is_empty(),
            "search with all deleted should return nothing"
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn vacuum_after_deleting_all_records_yields_empty_index() {
        let path = temp_index_path("soft-delete-vacuum-all");
        let mut writer = VectorIndex::create(&path, "fnv1a-384", 4).expect("writer");
        for i in 0..3 {
            writer
                .write_record(&format!("doc-{i}"), &[1.0, 0.0, 0.0, 0.0])
                .expect("write");
        }
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        for i in 0..3 {
            index.soft_delete(&format!("doc-{i}")).expect("delete");
        }

        let stats = index.vacuum().expect("vacuum all deleted");
        assert_eq!(stats.records_before, 3);
        assert_eq!(stats.records_after, 0);
        assert_eq!(stats.tombstones_removed, 3);
        assert_eq!(index.record_count(), 0);
        assert_eq!(index.tombstone_count(), 0);
        assert!(index.tombstone_ratio().abs() < f64::EPSILON);
        assert!(!index.needs_vacuum());

        let hits = index
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, None)
            .expect("search");
        assert!(hits.is_empty());

        std::fs::remove_file(&path).ok();
    }

    // ─── WAL integration tests ─────────────────────────────────────────

    #[test]
    fn append_single_vector_is_searchable() {
        let path = temp_index_path("wal-append-single");
        let dim = 4;

        // Build initial index.
        let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
        writer
            .write_record("main-0", &[1.0, 0.0, 0.0, 0.0])
            .expect("write");
        writer.finish().expect("finish");

        // Append via WAL.
        let mut index = VectorIndex::open(&path).expect("open");
        assert_eq!(index.wal_record_count(), 0);
        index
            .append("wal-0", &[0.0, 1.0, 0.0, 0.0])
            .expect("append");
        assert_eq!(index.wal_record_count(), 1);

        // Search should find both main and WAL entries.
        let hits = index
            .search_top_k(&[0.0, 1.0, 0.0, 0.0], 10, None)
            .expect("search");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].doc_id, "wal-0", "WAL entry should rank first");

        // Cleanup.
        std::fs::remove_file(&path).ok();
        std::fs::remove_file(wal::wal_path_for(&path)).ok();
    }

    #[test]
    fn append_batch_all_searchable() {
        let path = temp_index_path("wal-append-batch");
        let dim = 4;

        let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
        writer
            .write_record("main-0", &[1.0, 0.0, 0.0, 0.0])
            .expect("write");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        index
            .append_batch(&[
                ("wal-0".to_owned(), vec![0.0, 1.0, 0.0, 0.0]),
                ("wal-1".to_owned(), vec![0.0, 0.0, 1.0, 0.0]),
                ("wal-2".to_owned(), vec![0.0, 0.0, 0.0, 1.0]),
            ])
            .expect("append batch");
        assert_eq!(index.wal_record_count(), 3);

        let hits = index
            .search_top_k(&[1.0, 1.0, 1.0, 1.0], 10, None)
            .expect("search");
        assert_eq!(hits.len(), 4, "all 4 vectors should be returned");

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(wal::wal_path_for(&path)).ok();
    }

    #[test]
    fn compaction_merges_wal_into_main() {
        let path = temp_index_path("wal-compact");
        let dim = 4;

        let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
        writer
            .write_record("main-0", &[1.0, 0.0, 0.0, 0.0])
            .expect("write");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        index
            .append("wal-0", &[0.0, 1.0, 0.0, 0.0])
            .expect("append");
        index
            .append("wal-1", &[0.0, 0.0, 1.0, 0.0])
            .expect("append");

        assert_eq!(index.record_count(), 1);
        assert_eq!(index.wal_record_count(), 2);

        let stats = index.compact().expect("compact");
        assert_eq!(stats.main_records_before, 1);
        assert_eq!(stats.wal_records, 2);
        assert_eq!(stats.total_records_after, 3);
        assert_eq!(index.record_count(), 3);
        assert_eq!(index.wal_record_count(), 0);
        assert!(!wal::wal_path_for(&path).exists(), "WAL should be deleted");

        // All records should still be searchable from main index.
        let hits = index
            .search_top_k(&[1.0, 1.0, 1.0, 1.0], 10, None)
            .expect("search");
        assert_eq!(hits.len(), 3);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn needs_compaction_threshold() {
        let path = temp_index_path("wal-threshold");
        let dim = 4;

        let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
        for i in 0..10 {
            writer
                .write_record(&format!("main-{i}"), &sample_vector(0.1, dim))
                .expect("write");
        }
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        index.set_wal_config(WalConfig {
            compaction_threshold: 5,
            compaction_ratio: 0.10,
            fsync_on_write: false,
        });

        assert!(!index.needs_compaction());

        // Add 1 entry: ratio = 1/10 = 0.10, hits the ratio threshold.
        index
            .append("wal-0", &sample_vector(0.2, dim))
            .expect("append");
        assert!(index.needs_compaction());

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(wal::wal_path_for(&path)).ok();
    }

    #[test]
    fn wal_survives_reopen() {
        let path = temp_index_path("wal-reopen");
        let dim = 4;

        let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
        writer
            .write_record("main-0", &[1.0, 0.0, 0.0, 0.0])
            .expect("write");
        writer.finish().expect("finish");

        // Append and drop.
        {
            let mut index = VectorIndex::open(&path).expect("open");
            index
                .append("wal-0", &[0.0, 1.0, 0.0, 0.0])
                .expect("append");
        }

        // Reopen — WAL should be loaded automatically.
        let index = VectorIndex::open(&path).expect("reopen");
        assert_eq!(index.wal_record_count(), 1);

        let hits = index
            .search_top_k(&[0.0, 1.0, 0.0, 0.0], 10, None)
            .expect("search");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].doc_id, "wal-0");

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(wal::wal_path_for(&path)).ok();
    }

    #[test]
    fn append_dimension_mismatch_rejected() {
        let path = temp_index_path("wal-dim-mismatch");
        let dim = 4;

        let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
        writer
            .write_record("main-0", &sample_vector(1.0, dim))
            .expect("write");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        let err = index
            .append("bad", &[1.0, 2.0])
            .expect_err("should reject wrong dimension");
        assert!(matches!(err, SearchError::DimensionMismatch { .. }));
        assert_eq!(
            index.wal_record_count(),
            0,
            "failed append should not persist"
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn compact_empty_wal_is_noop() {
        let path = temp_index_path("wal-compact-empty");
        let dim = 4;

        let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
        writer
            .write_record("main-0", &sample_vector(1.0, dim))
            .expect("write");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        let stats = index.compact().expect("compact empty WAL");
        assert_eq!(stats.wal_records, 0);
        assert_eq!(stats.total_records_after, 1);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn wal_entries_rank_correctly_against_main() {
        let path = temp_index_path("wal-ranking");
        let dim = 4;

        // Main index has a mediocre match.
        let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
        writer
            .write_record("main-mediocre", &[0.5, 0.5, 0.0, 0.0])
            .expect("write");
        writer.finish().expect("finish");

        // WAL has a perfect match.
        let mut index = VectorIndex::open(&path).expect("open");
        index
            .append("wal-perfect", &[1.0, 0.0, 0.0, 0.0])
            .expect("append");

        let hits = index
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 2, None)
            .expect("search");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].doc_id, "wal-perfect");
        assert!(hits[0].score > hits[1].score);

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(wal::wal_path_for(&path)).ok();
    }

    #[test]
    fn append_duplicate_doc_id_both_searchable() {
        let path = temp_index_path("wal-dup-docid");
        let dim = 4;

        let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
        writer
            .write_record("doc-a", &[1.0, 0.0, 0.0, 0.0])
            .expect("write");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        // Append a second entry with the same doc_id but different vector.
        index
            .append("doc-a", &[0.0, 0.0, 0.0, 1.0])
            .expect("append duplicate");
        assert_eq!(index.wal_record_count(), 1);

        // Both entries should appear (WAL doesn't deduplicate).
        let hits = index
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, None)
            .expect("search");
        assert_eq!(hits.len(), 2);
        let doc_ids: Vec<&str> = hits.iter().map(|h| h.doc_id.as_str()).collect();
        assert!(doc_ids.iter().all(|id| *id == "doc-a"));

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(wal::wal_path_for(&path)).ok();
    }

    #[test]
    fn append_large_batch_100_vectors() {
        let path = temp_index_path("wal-large-batch");
        let dim = 8;

        let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
        writer
            .write_record("main-0", &sample_vector(1.0, dim))
            .expect("write");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        let batch: Vec<(String, Vec<f32>)> = (0..100)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let base = (i as f32) * 0.01;
                (format!("wal-{i:03}"), sample_vector(base, dim))
            })
            .collect();
        index.append_batch(&batch).expect("large batch");
        assert_eq!(index.wal_record_count(), 100);

        let hits = index
            .search_top_k(&sample_vector(1.0, dim), 5, None)
            .expect("search");
        assert_eq!(hits.len(), 5);
        // The main-0 (base=1.0) should rank near the top with query [1.0, ...].
        assert!(hits.iter().any(|h| h.doc_id == "main-0"));

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(wal::wal_path_for(&path)).ok();
    }

    #[test]
    fn concurrent_append_and_search() {
        use std::sync::Arc;

        let path = temp_index_path("wal-concurrent");
        let dim = 4;

        let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
        for i in 0..10 {
            writer
                .write_record(&format!("main-{i}"), &sample_vector(0.1, dim))
                .expect("write");
        }
        writer.finish().expect("finish");

        // Append sequentially (VectorIndex is not Send+Sync for shared mutation),
        // then search from multiple threads using a snapshot.
        let mut index = VectorIndex::open(&path).expect("open");
        for i in 0..20 {
            index
                .append(&format!("wal-{i}"), &sample_vector(0.5, dim))
                .expect("append");
        }

        let index = Arc::new(index);
        let query = sample_vector(1.0, dim);

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let idx = Arc::clone(&index);
                let q = query.clone();
                std::thread::spawn(move || idx.search_top_k(&q, 10, None).expect("search"))
            })
            .collect();

        for handle in handles {
            let hits = handle.join().expect("thread join");
            assert_eq!(hits.len(), 10);
            // All scores should be positive (dot product of positive vectors).
            assert!(hits.iter().all(|h| h.score > 0.0));
        }

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(wal::wal_path_for(&path)).ok();
    }

    #[test]
    fn wal_record_count_across_append_compact_cycles() {
        let path = temp_index_path("wal-count-cycle");
        let dim = 4;

        let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
        writer
            .write_record("main-0", &sample_vector(1.0, dim))
            .expect("write");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        assert_eq!(index.wal_record_count(), 0);
        assert_eq!(index.record_count(), 1);

        // Append 3 entries.
        index.append("w1", &sample_vector(0.1, dim)).expect("a1");
        index.append("w2", &sample_vector(0.2, dim)).expect("a2");
        index.append("w3", &sample_vector(0.3, dim)).expect("a3");
        assert_eq!(index.wal_record_count(), 3);
        assert_eq!(index.record_count(), 1);

        // Compact.
        index.compact().expect("compact");
        assert_eq!(index.wal_record_count(), 0);
        assert_eq!(index.record_count(), 4);

        // Append 2 more.
        index.append("w4", &sample_vector(0.4, dim)).expect("a4");
        index.append("w5", &sample_vector(0.5, dim)).expect("a5");
        assert_eq!(index.wal_record_count(), 2);
        assert_eq!(index.record_count(), 4);

        // Total searchable = 4 + 2 = 6.
        let hits = index
            .search_top_k(&sample_vector(1.0, dim), 100, None)
            .expect("search");
        assert_eq!(hits.len(), 6);

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(wal::wal_path_for(&path)).ok();
    }

    #[test]
    fn soft_delete_removes_wal_only_record_and_persists() {
        let path = temp_index_path("wal-soft-delete-only");
        let dim = 4;

        let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
        writer
            .write_record("main-0", &sample_vector(1.0, dim))
            .expect("write");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        index
            .append("wal-only", &[0.0, 1.0, 0.0, 0.0])
            .expect("append wal-only");
        assert_eq!(index.wal_record_count(), 1);

        assert!(index.soft_delete("wal-only").expect("soft delete wal-only"));
        assert_eq!(index.wal_record_count(), 0);
        let hits = index
            .search_top_k(&[0.0, 1.0, 0.0, 0.0], 10, None)
            .expect("search");
        assert!(hits.iter().all(|hit| hit.doc_id != "wal-only"));

        drop(index);
        let reopened = VectorIndex::open(&path).expect("reopen");
        assert_eq!(reopened.wal_record_count(), 0);
        let reopened_hits = reopened
            .search_top_k(&[0.0, 1.0, 0.0, 0.0], 10, None)
            .expect("search after reopen");
        assert!(reopened_hits.iter().all(|hit| hit.doc_id != "wal-only"));

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(wal::wal_path_for(&path)).ok();
    }

    #[test]
    fn soft_delete_clears_pending_wal_updates_for_same_doc_id() {
        let path = temp_index_path("wal-soft-delete-main-and-wal");
        let dim = 4;

        let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
        writer
            .write_record("doc-a", &[1.0, 0.0, 0.0, 0.0])
            .expect("write doc-a");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        index
            .append("doc-a", &[0.0, 1.0, 0.0, 0.0])
            .expect("append doc-a update");
        index
            .append("doc-b", &[0.0, 0.0, 1.0, 0.0])
            .expect("append doc-b");
        assert_eq!(index.wal_record_count(), 2);

        assert!(index.soft_delete("doc-a").expect("soft delete doc-a"));
        assert_eq!(
            index.wal_record_count(),
            1,
            "doc-a WAL entries should be purged"
        );

        let hits = index
            .search_top_k(&[0.0, 1.0, 0.0, 0.0], 10, None)
            .expect("search");
        assert!(
            hits.iter().all(|hit| hit.doc_id != "doc-a"),
            "doc-a should not be searchable from main or WAL"
        );
        assert!(hits.iter().any(|hit| hit.doc_id == "doc-b"));

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(wal::wal_path_for(&path)).ok();
    }

    #[test]
    fn empty_index_append_only() {
        let path = temp_index_path("wal-empty-append");
        let dim = 4;

        // Create an empty main index.
        let writer = VectorIndex::create(&path, "test", dim).expect("writer");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        assert_eq!(index.record_count(), 0);

        // Append to empty index via WAL.
        index
            .append("first", &[1.0, 0.0, 0.0, 0.0])
            .expect("append");
        assert_eq!(index.wal_record_count(), 1);

        // Should still be searchable.
        let hits = index
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, None)
            .expect("search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "first");

        // Compact from empty main + WAL.
        let stats = index.compact().expect("compact");
        assert_eq!(stats.main_records_before, 0);
        assert_eq!(stats.wal_records, 1);
        assert_eq!(stats.total_records_after, 1);
        assert_eq!(index.record_count(), 1);

        std::fs::remove_file(&path).ok();
    }

    // ─── Quantization edge cases ────────────────────────────────────────

    #[test]
    fn quantization_bytes_per_element() {
        assert_eq!(Quantization::F32.bytes_per_element(), 4);
        assert_eq!(Quantization::F16.bytes_per_element(), 2);
    }

    #[test]
    fn quantization_from_wire_valid() {
        let path = Path::new("test.fsvi");
        assert_eq!(Quantization::from_wire(0, path).unwrap(), Quantization::F32);
        assert_eq!(Quantization::from_wire(1, path).unwrap(), Quantization::F16);
    }

    #[test]
    fn quantization_from_wire_invalid() {
        let path = Path::new("test.fsvi");
        assert!(Quantization::from_wire(2, path).is_err());
        assert!(Quantization::from_wire(255, path).is_err());
    }

    // ─── align_up edge cases ────────────────────────────────────────────

    #[test]
    fn align_up_zero_alignment() {
        assert_eq!(align_up(42, 0).unwrap(), 42);
    }

    #[test]
    fn align_up_already_aligned() {
        assert_eq!(align_up(128, 64).unwrap(), 128);
    }

    #[test]
    fn align_up_zero_value() {
        assert_eq!(align_up(0, 64).unwrap(), 0);
    }

    #[test]
    fn align_up_one_over() {
        assert_eq!(align_up(65, 64).unwrap(), 128);
    }

    // ─── fnv1a_hash edge cases ──────────────────────────────────────────

    #[test]
    fn fnv1a_hash_empty_input() {
        let hash = fnv1a_hash(b"");
        assert_eq!(hash, 0xcbf2_9ce4_8422_2325);
    }

    #[test]
    fn fnv1a_hash_deterministic() {
        let h1 = fnv1a_hash(b"hello");
        let h2 = fnv1a_hash(b"hello");
        assert_eq!(h1, h2);
    }

    #[test]
    fn fnv1a_hash_different_inputs_differ() {
        let h1 = fnv1a_hash(b"doc-a");
        let h2 = fnv1a_hash(b"doc-b");
        assert_ne!(h1, h2);
    }

    // ─── is_tombstoned_flags ────────────────────────────────────────────

    #[test]
    fn tombstone_flag_logic() {
        assert!(!is_tombstoned_flags(0x0000));
        assert!(is_tombstoned_flags(RECORD_FLAG_TOMBSTONE));
        assert!(is_tombstoned_flags(0x0003)); // tombstone + custom
        assert!(!is_tombstoned_flags(0x0002)); // only custom
    }

    // ─── validate_header_string ─────────────────────────────────────────

    #[test]
    fn validate_header_string_empty_embedder_id_rejected() {
        let result = validate_header_string("", "embedder_id");
        assert!(result.is_err());
    }

    #[test]
    fn validate_header_string_empty_embedder_revision_ok() {
        let result = validate_header_string("", "embedder_revision");
        assert!(result.is_ok());
    }

    #[test]
    fn validate_header_string_normal_ok() {
        let result = validate_header_string("potion-128M", "embedder_id");
        assert!(result.is_ok());
    }

    // ─── VectorMetadata clone/eq ────────────────────────────────────────

    #[test]
    fn vector_metadata_clone_eq() {
        let meta = VectorMetadata {
            embedder_id: "test".to_owned(),
            embedder_revision: "v1".to_owned(),
            dimension: 256,
            quantization: Quantization::F16,
            compaction_gen: 0,
            record_count: 100,
            vectors_offset: 1024,
        };
        let cloned = meta.clone();
        assert_eq!(meta, cloned);
    }

    // ─── VectorIndex::create validation ─────────────────────────────────

    #[test]
    fn create_zero_dimension_rejected() {
        let path = temp_index_path("zero-dim");
        let result = VectorIndex::create(&path, "test", 0);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SearchError::InvalidConfig { .. }
        ));
    }

    #[test]
    fn create_empty_embedder_id_rejected() {
        let path = temp_index_path("empty-embedder");
        let result = VectorIndex::create(&path, "", 4);
        assert!(result.is_err());
    }

    #[test]
    fn create_with_revision_empty_revision_ok() {
        let path = temp_index_path("empty-rev");
        let writer =
            VectorIndex::create_with_revision(&path, "test", "", 4, Quantization::F16).unwrap();
        writer.finish().unwrap();
        let index = VectorIndex::open(&path).unwrap();
        assert_eq!(index.embedder_revision(), "");
        std::fs::remove_file(&path).ok();
    }

    // ─── VectorIndexWriter rejection cases ──────────────────────────────

    #[test]
    fn write_record_nan_embedding_rejected() {
        let path = temp_index_path("nan-embed");
        let mut writer = VectorIndex::create(&path, "test", 3).unwrap();
        let result = writer.write_record("doc", &[1.0, f32::NAN, 0.0]);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("non-finite"),
            "expected non-finite error, got: {err}"
        );
    }

    #[test]
    fn write_record_inf_embedding_rejected() {
        let path = temp_index_path("inf-embed");
        let mut writer = VectorIndex::create(&path, "test", 3).unwrap();
        let result = writer.write_record("doc", &[1.0, f32::INFINITY, 0.0]);
        assert!(result.is_err());
    }

    // ─── VectorIndex::open edge cases ───────────────────────────────────

    #[test]
    fn open_nonexistent_file_returns_index_not_found() {
        let path = temp_index_path("nonexistent-open");
        let result = VectorIndex::open(&path);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SearchError::IndexNotFound { .. }
        ));
    }

    #[test]
    fn open_truncated_file_detected() {
        let path = temp_index_path("truncated-open");
        let mut writer = VectorIndex::create(&path, "test", 4).unwrap();
        writer.write_record("doc-0", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        writer.finish().unwrap();

        let data = std::fs::read(&path).unwrap();
        std::fs::write(&path, &data[..data.len() - 4]).unwrap();

        let result = VectorIndex::open(&path);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("truncated") || err.contains("too small") || err.contains("extends"),
            "expected truncation error, got: {err}"
        );

        std::fs::remove_file(&path).ok();
    }

    // ─── FSVI constants ─────────────────────────────────────────────────

    #[test]
    fn fsvi_magic_is_four_bytes() {
        assert_eq!(FSVI_MAGIC.len(), 4);
        assert_eq!(&FSVI_MAGIC, b"FSVI");
    }

    #[test]
    fn fsvi_version_is_one() {
        assert_eq!(FSVI_VERSION, 1);
    }

    #[test]
    fn record_size_is_sixteen() {
        assert_eq!(RECORD_SIZE_BYTES, 16);
    }

    // ─── vector_at_f16 on f16 index ─────────────────────────────────────

    #[test]
    fn vector_at_f16_roundtrip() {
        let path = temp_index_path("f16-at-roundtrip");
        let mut writer =
            VectorIndex::create_with_revision(&path, "test", "r1", 3, Quantization::F16).unwrap();
        writer.write_record("doc", &[0.5, -0.5, 1.0]).unwrap();
        writer.finish().unwrap();

        let index = VectorIndex::open(&path).unwrap();
        let f16_vec = index.vector_at_f16(0).unwrap();
        assert_eq!(f16_vec.len(), 3);
        assert!((f16_vec[0].to_f32() - 0.5).abs() < 0.01);
        assert!((f16_vec[1].to_f32() - (-0.5)).abs() < 0.01);
        assert!((f16_vec[2].to_f32() - 1.0).abs() < 0.01);

        std::fs::remove_file(&path).ok();
    }

    // ─── vector_at_f16 on f32 index (converts) ─────────────────────────

    #[test]
    fn vector_at_f16_from_f32_index() {
        let path = temp_index_path("f16-from-f32");
        let mut writer =
            VectorIndex::create_with_revision(&path, "test", "r1", 3, Quantization::F32).unwrap();
        writer.write_record("doc", &[0.25, -0.75, 1.0]).unwrap();
        writer.finish().unwrap();

        let index = VectorIndex::open(&path).unwrap();
        let f16_vec = index.vector_at_f16(0).unwrap();
        assert_eq!(f16_vec.len(), 3);
        assert!((f16_vec[0].to_f32() - 0.25).abs() < 0.01);

        std::fs::remove_file(&path).ok();
    }

    // ─── metadata accessor ──────────────────────────────────────────────

    #[test]
    fn metadata_accessor_returns_consistent_data() {
        let path = temp_index_path("metadata-accessor");
        let mut writer =
            VectorIndex::create_with_revision(&path, "emb-1", "rev-9", 16, Quantization::F32)
                .unwrap();
        writer.write_record("d", &[0.0; 16]).unwrap();
        writer.finish().unwrap();

        let index = VectorIndex::open(&path).unwrap();
        let meta = index.metadata();
        assert_eq!(meta.embedder_id, "emb-1");
        assert_eq!(meta.embedder_revision, "rev-9");
        assert_eq!(meta.dimension, 16);
        assert_eq!(meta.quantization, Quantization::F32);
        assert_eq!(meta.record_count, 1);
        assert_eq!(meta.vectors_offset % 64, 0);

        std::fs::remove_file(&path).ok();
    }

    // ─── is_deleted accessor ────────────────────────────────────────────

    #[test]
    fn is_deleted_false_for_live_record() {
        let path = temp_index_path("is-deleted-live");
        let mut writer = VectorIndex::create(&path, "test", 4).unwrap();
        writer.write_record("doc", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        writer.finish().unwrap();

        let index = VectorIndex::open(&path).unwrap();
        assert!(!index.is_deleted(0));

        std::fs::remove_file(&path).ok();
    }

    // ─── tombstone_ratio empty index ────────────────────────────────────

    #[test]
    fn tombstone_ratio_empty_index_is_zero() {
        let path = temp_index_path("tomb-ratio-empty");
        let writer = VectorIndex::create(&path, "test", 4).unwrap();
        writer.finish().unwrap();

        let index = VectorIndex::open(&path).unwrap();
        assert!(index.tombstone_ratio().abs() < f64::EPSILON);
        assert!(!index.needs_vacuum());

        std::fs::remove_file(&path).ok();
    }

    // ─── WalConfig default ──────────────────────────────────────────────

    #[test]
    fn wal_config_default_values() {
        let cfg = WalConfig::default();
        assert!(cfg.compaction_threshold > 0);
        assert!(cfg.compaction_ratio > 0.0);
    }

    // ─── F32 roundtrip with explicit revision ───────────────────────────

    #[test]
    fn f32_roundtrip_with_revision() {
        let path = temp_index_path("f32-rev-roundtrip");
        let original = vec![std::f32::consts::PI, std::f32::consts::E, 0.0, -1.0];
        let mut writer =
            VectorIndex::create_with_revision(&path, "f32-emb", "rev-42", 4, Quantization::F32)
                .unwrap();
        writer.write_record("doc", &original).unwrap();
        writer.finish().unwrap();

        let index = VectorIndex::open(&path).unwrap();
        let recovered = index.vector_at_f32(0).unwrap();
        assert_eq!(recovered, original, "f32 must roundtrip exactly");
        assert_eq!(index.embedder_revision(), "rev-42");

        std::fs::remove_file(&path).ok();
    }

    // ─── Header CRC corruption by flipping data byte ────────────────────

    #[test]
    fn header_crc_detects_embedder_id_corruption() {
        let path = temp_index_path("crc-embedder-corrupt");
        let mut writer = VectorIndex::create(&path, "test-embedder-long", 4).unwrap();
        writer.write_record("doc", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        writer.finish().unwrap();

        let mut data = std::fs::read(&path).unwrap();
        // Flip a byte in the embedder_id region (after magic+version+id_len = 8 bytes)
        data[10] ^= 0xFF;
        std::fs::write(&path, &data).unwrap();

        let result = VectorIndex::open(&path);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("CRC") || err.contains("crc"),
            "expected CRC error, got: {err}"
        );

        std::fs::remove_file(&path).ok();
    }

    // ─── bd-1fh4 tests begin ──────────────────────────────────────────

    #[test]
    fn vacuum_stats_debug_clone_partial_eq() {
        let stats = VacuumStats {
            records_before: 10,
            records_after: 8,
            tombstones_removed: 2,
            bytes_reclaimed: 1024,
            duration: Duration::from_millis(5),
        };
        let debug = format!("{stats:?}");
        assert!(debug.contains("VacuumStats"));
        assert!(debug.contains("records_before: 10"));

        let cloned = stats.clone();
        assert_eq!(stats, cloned);
    }

    #[test]
    fn quantization_debug_clone_copy_eq() {
        let f16 = Quantization::F16;
        let f32q = Quantization::F32;

        let debug_f16 = format!("{f16:?}");
        assert!(debug_f16.contains("F16"));
        let debug_f32 = format!("{f32q:?}");
        assert!(debug_f32.contains("F32"));

        let f16_copy = f16;
        assert_eq!(f16, f16_copy);
        let f32_copy = f32q;
        assert_eq!(f32q, f32_copy);
        assert_ne!(f16, f32q);
    }

    #[test]
    fn vector_index_debug_includes_path() {
        let path = temp_index_path("debug-fmt");
        let writer = VectorIndex::create(&path, "test", 4).unwrap();
        writer.finish().unwrap();

        let index = VectorIndex::open(&path).unwrap();
        let debug = format!("{index:?}");
        assert!(debug.contains("VectorIndex"));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn set_wal_config_overrides_defaults() {
        let path = temp_index_path("wal-cfg-override");
        let dim = 4;
        let mut writer = VectorIndex::create(&path, "test", dim).unwrap();
        for i in 0..100 {
            writer
                .write_record(&format!("d{i}"), &sample_vector(0.1, dim))
                .unwrap();
        }
        writer.finish().unwrap();

        let mut index = VectorIndex::open(&path).unwrap();
        // With 100 main records and default config, 1 WAL entry should not trigger.
        index.append("wal-1", &sample_vector(0.5, dim)).unwrap();
        assert!(!index.needs_compaction());

        // Set a low threshold to trigger compaction.
        index.set_wal_config(WalConfig {
            compaction_threshold: 1,
            compaction_ratio: 0.001,
            fsync_on_write: false,
        });
        assert!(index.needs_compaction());

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(wal::wal_path_for(&path)).ok();
    }

    #[test]
    fn find_index_by_doc_hash_empty_index_none() {
        let path = temp_index_path("hash-empty");
        let writer = VectorIndex::create(&path, "test", 4).unwrap();
        writer.finish().unwrap();

        let index = VectorIndex::open(&path).unwrap();
        assert!(index.find_index_by_doc_hash(0xDEAD_BEEF).is_none());
        assert!(index.find_index_by_doc_hash(0).is_none());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn get_embeddings_mixed_hit_miss() {
        let path = temp_index_path("emb-mixed");
        let mut writer =
            VectorIndex::create_with_revision(&path, "test", "r1", 3, Quantization::F16).unwrap();
        writer.write_record("alpha", &[1.0, 0.0, 0.0]).unwrap();
        writer.write_record("beta", &[0.0, 1.0, 0.0]).unwrap();
        writer.finish().unwrap();

        let index = VectorIndex::open(&path).unwrap();
        let alpha_hash = fnv1a_hash(b"alpha");
        let beta_hash = fnv1a_hash(b"beta");
        let missing_hash = fnv1a_hash(b"gamma");

        let results = index.get_embeddings(&[alpha_hash, missing_hash, beta_hash]);
        assert_eq!(results.len(), 3);
        assert!(results[0].is_some(), "alpha should be found");
        assert!(results[1].is_none(), "gamma should be missing");
        assert!(results[2].is_some(), "beta should be found");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn append_batch_empty_is_noop() {
        let path = temp_index_path("append-empty-batch");
        let writer = VectorIndex::create(&path, "test", 4).unwrap();
        writer.finish().unwrap();

        let mut index = VectorIndex::open(&path).unwrap();
        index.append_batch(&[]).unwrap();
        assert_eq!(index.wal_record_count(), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn append_nan_embedding_rejected() {
        let path = temp_index_path("append-nan");
        let writer = VectorIndex::create(&path, "test", 4).unwrap();
        writer.finish().unwrap();

        let mut index = VectorIndex::open(&path).unwrap();
        let result = index.append("doc", &[1.0, f32::NAN, 0.0, 0.0]);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("finite"), "expected finite error, got: {err}");
    }

    #[test]
    fn append_inf_embedding_rejected() {
        let path = temp_index_path("append-inf");
        let writer = VectorIndex::create(&path, "test", 4).unwrap();
        writer.finish().unwrap();

        let mut index = VectorIndex::open(&path).unwrap();
        let result = index.append("doc", &[1.0, 0.0, f32::INFINITY, 0.0]);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("finite"), "expected finite error, got: {err}");
    }

    #[test]
    fn soft_delete_already_deleted_returns_false() {
        let path = temp_index_path("double-delete");
        let mut writer = VectorIndex::create(&path, "test", 4).unwrap();
        writer.write_record("doc", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        writer.finish().unwrap();

        let mut index = VectorIndex::open(&path).unwrap();
        assert!(index.soft_delete("doc").unwrap(), "first delete");
        assert!(!index.soft_delete("doc").unwrap(), "second delete");
        assert!(!index.soft_delete("doc").unwrap(), "third delete");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn compact_preserves_wal_config() {
        let path = temp_index_path("compact-cfg");
        let dim = 4;
        let mut writer = VectorIndex::create(&path, "test", dim).unwrap();
        for i in 0..20 {
            writer
                .write_record(&format!("d{i}"), &sample_vector(0.1, dim))
                .unwrap();
        }
        writer.finish().unwrap();

        let mut index = VectorIndex::open(&path).unwrap();
        let custom = WalConfig {
            compaction_threshold: 99,
            compaction_ratio: 0.90,
            fsync_on_write: false,
        };
        index.set_wal_config(custom);
        index.append("wal-1", &sample_vector(0.5, dim)).unwrap();
        index.compact().unwrap();

        // After compaction, the custom config should be preserved.
        assert_eq!(index.wal_record_count(), 0);
        // Verify config persists: threshold=99 and ratio=0.90,
        // with 21 main records, 1 WAL entry → ratio ~0.048 < 0.90.
        index.append("wal-2", &sample_vector(0.3, dim)).unwrap();
        assert!(!index.needs_compaction());

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(wal::wal_path_for(&path)).ok();
    }

    #[test]
    fn soft_delete_wal_restores_state_on_rewrite_failure() {
        let path = temp_index_path("wal-delete-restore");
        let dim = 4;

        let mut writer = VectorIndex::create(&path, "test", dim).unwrap();
        writer
            .write_record("main-0", &sample_vector(1.0, dim))
            .unwrap();
        writer.finish().unwrap();

        let mut index = VectorIndex::open(&path).unwrap();
        index.append("wal-a", &[0.0, 1.0, 0.0, 0.0]).unwrap();
        index.append("wal-b", &[0.0, 0.0, 1.0, 0.0]).unwrap();
        assert_eq!(index.wal_record_count(), 2);

        // Make the WAL parent directory read-only to force a rewrite failure.
        let wal_file = wal::wal_path_for(&path);
        let wal_dir = wal_file.parent().unwrap();
        let original_perms = fs::metadata(wal_dir).unwrap().permissions();
        let mut readonly = original_perms.clone();
        readonly.set_readonly(true);
        if fs::set_permissions(wal_dir, readonly).is_err() {
            // Sandboxed environments may not allow permission changes; skip.
            std::fs::remove_file(&path).ok();
            std::fs::remove_file(wal::wal_path_for(&path)).ok();
            return;
        }

        let result = index.soft_delete("wal-a");

        // Restore directory permissions before any assertions so cleanup works.
        fs::set_permissions(wal_dir, original_perms).unwrap();

        // The delete should have failed.
        assert!(result.is_err(), "expected error from read-only directory");

        // In-memory WAL entries must be fully restored.
        assert_eq!(
            index.wal_record_count(),
            2,
            "WAL entries should be restored after rewrite failure"
        );

        // Both entries should still be searchable.
        let hits = index.search_top_k(&[0.0, 1.0, 0.0, 0.0], 10, None).unwrap();
        assert!(hits.iter().any(|h| h.doc_id == "wal-a"));
        assert!(hits.iter().any(|h| h.doc_id == "wal-b"));

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(wal::wal_path_for(&path)).ok();
    }

    // ─── Regression: Duplicate entries on compaction crash ──────────────

    #[test]
    fn repro_duplicate_entries_on_compaction_crash() {
        let path = temp_index_path("compaction-crash");
        let dim = 4;

        // 1. Create initial index with 1 document
        let mut writer =
            VectorIndex::create_with_revision(&path, "test", "v1", dim, Quantization::F16).unwrap();
        writer.write_record("doc-A", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        writer.finish().unwrap();

        let mut index = VectorIndex::open(&path).unwrap();

        // 2. Append a document to WAL
        index.append("doc-B", &[0.0, 1.0, 0.0, 0.0]).unwrap();

        // Check state before "compaction"
        let hits = index.search_top_k(&[1.0, 1.0, 0.0, 0.0], 10, None).unwrap();
        assert_eq!(hits.len(), 2);

        // 3. Simulate compaction crash:
        // We want to create a state where "doc-B" is in Main Index AND in WAL.
        // We can do this by running `compact` but preventing the WAL deletion.
        // Since we can't easily interrupt `compact`, we'll simulate the filesystem state.

        // Close index to flush everything
        drop(index);

        // Manually create the "post-compaction" main index that includes both A and B.
        let mut compact_writer =
            VectorIndex::create_with_revision(&path, "test", "v1", dim, Quantization::F16)
                .unwrap()
                .with_generation(2); // Simulate correct compaction increment
        compact_writer
            .write_record("doc-A", &[1.0, 0.0, 0.0, 0.0])
            .unwrap();
        compact_writer
            .write_record("doc-B", &[0.0, 1.0, 0.0, 0.0])
            .unwrap();
        compact_writer.finish().unwrap(); // Overwrites `path` with new index containing A and B.

        // Restore the WAL file (because `finish` doesn't touch it, but we need to ensure it exists and has doc-B)
        // Actually, `finish` overwrites `path`. The WAL file is at `path.wal`.
        // We didn't delete `path.wal`. So `path.wal` still contains "doc-B".

        // 4. Re-open index. It should load Main (A, B) and WAL (B).
        let index_reopened = VectorIndex::open(&path).unwrap();

        // 5. Search. If bug exists, we'll see "doc-B" twice.
        let hits = index_reopened
            .search_top_k(&[1.0, 1.0, 0.0, 0.0], 10, None)
            .unwrap();

        // Debug output
        for hit in &hits {
            println!("Hit: {} score={}", hit.doc_id, hit.score);
        }

        // Clean up
        let _ = fs::remove_file(&path);
        let _ = wal::remove_wal(&wal::wal_path_for(&path));

        // Assert failure
        let hit_count = hits.len();
        assert_eq!(
            hit_count, 2,
            "Should have exactly 2 hits (A and B), found {hit_count}"
        );
        let b_count = hits.iter().filter(|h| h.doc_id == "doc-B").count();
        assert_eq!(b_count, 1, "Should have exactly 1 'doc-B', found {b_count}");
    }

    // ─── bd-1fh4 tests end ────────────────────────────────────────────
}
