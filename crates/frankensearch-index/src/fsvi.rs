//! FSVI (FrankenSearch Vector Index) binary format.
//!
//! A universal vector index format for storing `(doc_id, embedding)` pairs
//! with f16 quantization and memory-mapped access.
//!
//! # File Layout
//!
//! ```text
//! ┌─────────────────────────────────────┐
//! │ Header (variable length)            │
//! │   magic: b"FSVI" (4 bytes)          │
//! │   version: u16                      │
//! │   dimension: u16                    │
//! │   quantization: u8                  │
//! │   embedder_id_len + embedder_id     │
//! │   embedder_rev_len + embedder_rev   │
//! │   reserved: [u8; 3]                 │
//! │   record_count: u64                 │
//! │   header_crc32: u32                 │
//! ├─────────────────────────────────────┤
//! │ Record Table                        │
//! │   record_count × 6 bytes each       │
//! │   (doc_id_offset: u32, len: u16)    │
//! ├─────────────────────────────────────┤
//! │ String Table                        │
//! │   Concatenated UTF-8 doc_id strings │
//! ├─────────────────────────────────────┤
//! │ Padding (to 64-byte alignment)      │
//! ├─────────────────────────────────────┤
//! │ Vector Slab                         │
//! │   record_count × dimension × elem   │
//! │   (2 bytes/elem for f16)            │
//! └─────────────────────────────────────┘
//! ```

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use half::f16;
use memmap2::Mmap;

use frankensearch_core::{SearchError, SearchResult};

// ─── Constants ──────────────────────────────────────────────────────────────

/// Magic bytes identifying an FSVI file.
const MAGIC: [u8; 4] = *b"FSVI";

/// Current format version.
const FORMAT_VERSION: u16 = 1;

/// Alignment boundary for the vector slab (cache-line aligned for SIMD).
const VECTOR_ALIGN_BYTES: usize = 64;

/// Size of each record table entry (doc_id_offset: u32 + doc_id_len: u16).
const RECORD_ENTRY_SIZE: usize = 6;

/// Buffer size for the writer (256 KB for reduced syscall overhead).
const WRITE_BUFFER_SIZE: usize = 256 * 1024;

// ─── Quantization ───────────────────────────────────────────────────────────

/// Vector element quantization format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Quantization {
    /// 32-bit float (4 bytes per element).
    F32 = 0,
    /// 16-bit float (2 bytes per element). Default.
    F16 = 1,
}

impl Quantization {
    /// Bytes per vector element.
    #[must_use]
    pub const fn element_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
        }
    }

    fn from_u8(v: u8) -> SearchResult<Self> {
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            other => Err(SearchError::IndexCorrupted {
                path: String::new(),
                detail: format!("unknown quantization byte: {other}"),
            }),
        }
    }
}

// ─── Header ─────────────────────────────────────────────────────────────────

/// Parsed FSVI header metadata.
#[derive(Debug, Clone)]
pub struct FsviHeader {
    /// Format version.
    pub version: u16,
    /// Embedding dimensionality.
    pub dimension: u16,
    /// Quantization format.
    pub quantization: Quantization,
    /// Stable embedder identifier.
    pub embedder_id: String,
    /// Embedder revision hash (model version tracking).
    pub embedder_revision: String,
    /// Number of vectors stored.
    pub record_count: u64,
    /// Total header size in bytes (including CRC).
    pub header_size: usize,
}

/// Write a little-endian u16.
fn write_u16(buf: &mut Vec<u8>, val: u16) {
    buf.extend_from_slice(&val.to_le_bytes());
}

/// Write a little-endian u32.
fn write_u32(buf: &mut Vec<u8>, val: u32) {
    buf.extend_from_slice(&val.to_le_bytes());
}

/// Write a little-endian u64.
fn write_u64(buf: &mut Vec<u8>, val: u64) {
    buf.extend_from_slice(&val.to_le_bytes());
}

/// Read a little-endian u16 from a byte slice at offset.
fn read_u16(data: &[u8], offset: usize) -> SearchResult<u16> {
    let bytes: [u8; 2] = data
        .get(offset..offset + 2)
        .and_then(|s| s.try_into().ok())
        .ok_or_else(|| SearchError::IndexCorrupted {
            path: String::new(),
            detail: format!("truncated header at offset {offset}"),
        })?;
    Ok(u16::from_le_bytes(bytes))
}

/// Read a little-endian u32 from a byte slice at offset.
fn read_u32(data: &[u8], offset: usize) -> SearchResult<u32> {
    let bytes: [u8; 4] = data
        .get(offset..offset + 4)
        .and_then(|s| s.try_into().ok())
        .ok_or_else(|| SearchError::IndexCorrupted {
            path: String::new(),
            detail: format!("truncated header at offset {offset}"),
        })?;
    Ok(u32::from_le_bytes(bytes))
}

/// Read a little-endian u64 from a byte slice at offset.
fn read_u64(data: &[u8], offset: usize) -> SearchResult<u64> {
    let bytes: [u8; 8] = data
        .get(offset..offset + 8)
        .and_then(|s| s.try_into().ok())
        .ok_or_else(|| SearchError::IndexCorrupted {
            path: String::new(),
            detail: format!("truncated header at offset {offset}"),
        })?;
    Ok(u64::from_le_bytes(bytes))
}

impl FsviHeader {
    /// Serialize header bytes (without CRC — CRC is appended separately).
    fn to_bytes_without_crc(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(64);

        // Magic
        buf.extend_from_slice(&MAGIC);
        // Version
        write_u16(&mut buf, self.version);
        // Dimension
        write_u16(&mut buf, self.dimension);
        // Quantization
        buf.push(self.quantization as u8);
        // Embedder ID
        let id_bytes = self.embedder_id.as_bytes();
        #[allow(clippy::cast_possible_truncation)]
        buf.push(id_bytes.len() as u8);
        buf.extend_from_slice(id_bytes);
        // Embedder revision
        let rev_bytes = self.embedder_revision.as_bytes();
        #[allow(clippy::cast_possible_truncation)]
        buf.push(rev_bytes.len() as u8);
        buf.extend_from_slice(rev_bytes);
        // Reserved (3 bytes padding to 4-byte boundary)
        buf.extend_from_slice(&[0u8; 3]);
        // Record count
        write_u64(&mut buf, self.record_count);

        buf
    }

    /// Serialize full header including CRC32.
    fn to_bytes(&self) -> Vec<u8> {
        let mut buf = self.to_bytes_without_crc();
        let crc = crc32fast::hash(&buf);
        write_u32(&mut buf, crc);
        buf
    }

    /// Parse header from file data.
    fn from_bytes(data: &[u8], path: &str) -> SearchResult<Self> {
        let mk_err = |detail: String| SearchError::IndexCorrupted {
            path: path.to_owned(),
            detail,
        };

        // Minimum header: magic(4) + version(2) + dim(2) + quant(1) + id_len(1)
        //                 + rev_len(1) + reserved(3) + count(8) + crc(4) = 26 bytes
        if data.len() < 26 {
            return Err(mk_err("file too small for FSVI header".into()));
        }

        // Magic
        if data[0..4] != MAGIC {
            return Err(mk_err(format!(
                "bad magic: expected FSVI, got {:?}",
                &data[0..4]
            )));
        }

        let version = read_u16(data, 4)?;
        if version != FORMAT_VERSION {
            return Err(SearchError::IndexVersionMismatch {
                expected: u32::from(FORMAT_VERSION),
                found: u32::from(version),
                path: path.to_owned(),
            });
        }

        let dimension = read_u16(data, 6)?;
        let quantization = Quantization::from_u8(data[8])?;

        // Embedder ID
        let id_len = data[9] as usize;
        let id_start = 10;
        let id_end = id_start + id_len;
        if data.len() < id_end + 1 {
            return Err(mk_err("truncated embedder_id".into()));
        }
        let embedder_id = std::str::from_utf8(&data[id_start..id_end])
            .map_err(|e| mk_err(format!("invalid UTF-8 in embedder_id: {e}")))?
            .to_owned();

        // Embedder revision
        let rev_len = data[id_end] as usize;
        let rev_start = id_end + 1;
        let rev_end = rev_start + rev_len;
        if data.len() < rev_end + 3 {
            return Err(mk_err("truncated embedder_revision".into()));
        }
        let embedder_revision = std::str::from_utf8(&data[rev_start..rev_end])
            .map_err(|e| mk_err(format!("invalid UTF-8 in embedder_revision: {e}")))?
            .to_owned();

        // Skip reserved (3 bytes)
        let count_offset = rev_end + 3;
        let record_count = read_u64(data, count_offset)?;

        let crc_offset = count_offset + 8;
        let stored_crc = read_u32(data, crc_offset)?;
        let header_size = crc_offset + 4;

        // Verify CRC
        let computed_crc = crc32fast::hash(&data[..crc_offset]);
        if stored_crc != computed_crc {
            return Err(mk_err(format!(
                "header CRC mismatch: stored={stored_crc:#010x}, computed={computed_crc:#010x}"
            )));
        }

        Ok(Self {
            version,
            dimension,
            quantization,
            embedder_id,
            embedder_revision,
            record_count,
            header_size,
        })
    }
}

// ─── VectorIndex (Reader) ───────────────────────────────────────────────────

/// Memory-mapped FSVI vector index for zero-copy access.
///
/// The index file is memory-mapped, so vectors are read directly from the OS
/// page cache without copying into userspace buffers.
pub struct VectorIndex {
    mmap: Mmap,
    header: FsviHeader,
    /// Byte offset where the record table starts.
    record_table_offset: usize,
    /// Byte offset where the string table starts.
    string_table_offset: usize,
    /// Byte offset where the vector slab starts (64-byte aligned).
    vector_slab_offset: usize,
}

impl VectorIndex {
    /// Open an existing FSVI index file.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if the file cannot be opened, is corrupted,
    /// or has a version mismatch.
    pub fn open(path: &Path) -> SearchResult<Self> {
        let file = File::open(path).map_err(SearchError::Io)?;
        let path_str = path.display().to_string();

        // SAFETY: memmap2::Mmap is safe for read-only access.
        // The file must not be concurrently truncated. We rely on
        // the caller holding appropriate file locks.
        let mmap = memmap2::MmapOptions::new()
            .map(&file)
            .map_err(SearchError::Io)?;

        let header = FsviHeader::from_bytes(&mmap, &path_str)?;

        let record_table_offset = header.header_size;
        let record_count = header.record_count as usize;
        let string_table_offset = record_table_offset
            .checked_add(record_count.checked_mul(RECORD_ENTRY_SIZE).ok_or_else(|| {
                SearchError::IndexCorrupted {
                    path: path_str.clone(),
                    detail: "record_count overflow in record table size".into(),
                }
            })?)
            .ok_or_else(|| SearchError::IndexCorrupted {
                path: path_str.clone(),
                detail: "record table offset overflow".into(),
            })?;

        // Compute string table size by scanning record entries
        let string_table_size = Self::compute_string_table_size(&mmap, &header, &path_str)?;

        let raw_slab_start = string_table_offset
            .checked_add(string_table_size)
            .ok_or_else(|| SearchError::IndexCorrupted {
                path: path_str.clone(),
                detail: "string table offset overflow".into(),
            })?;
        let vector_slab_offset = align_up(raw_slab_start, VECTOR_ALIGN_BYTES);

        // Validate file size
        let dim = header.dimension as usize;
        let elem = header.quantization.element_size();
        let expected_slab_size = record_count
            .checked_mul(dim)
            .and_then(|v| v.checked_mul(elem))
            .ok_or_else(|| SearchError::IndexCorrupted {
                path: path_str.clone(),
                detail: "vector slab size overflow".into(),
            })?;
        let expected_file_size = vector_slab_offset
            .checked_add(expected_slab_size)
            .ok_or_else(|| SearchError::IndexCorrupted {
                path: path_str.clone(),
                detail: "expected file size overflow".into(),
            })?;
        if mmap.len() < expected_file_size {
            return Err(SearchError::IndexCorrupted {
                path: path_str,
                detail: format!(
                    "file too small: expected at least {expected_file_size} bytes, got {}",
                    mmap.len()
                ),
            });
        }

        Ok(Self {
            mmap,
            header,
            record_table_offset,
            string_table_offset,
            vector_slab_offset,
        })
    }

    /// Compute total string table size by finding the max (offset + len) across records.
    fn compute_string_table_size(
        data: &[u8],
        header: &FsviHeader,
        path: &str,
    ) -> SearchResult<usize> {
        let record_table_offset = header.header_size;
        let record_count = header.record_count as usize;
        let mut max_end: usize = 0;

        for i in 0..record_count {
            let entry_offset = record_table_offset
                .checked_add(i.checked_mul(RECORD_ENTRY_SIZE).ok_or_else(|| {
                    SearchError::IndexCorrupted {
                        path: path.to_owned(),
                        detail: "record entry offset overflow".into(),
                    }
                })?)
                .ok_or_else(|| SearchError::IndexCorrupted {
                    path: path.to_owned(),
                    detail: "record entry offset overflow".into(),
                })?;
            let doc_offset = read_u32(data, entry_offset)? as usize;
            let doc_len = read_u16(data, entry_offset + 4)? as usize;
            let end =
                doc_offset
                    .checked_add(doc_len)
                    .ok_or_else(|| SearchError::IndexCorrupted {
                        path: path.to_owned(),
                        detail: "doc_id string range overflow".into(),
                    })?;
            if end > max_end {
                max_end = end;
            }
        }

        if record_count == 0 {
            return Ok(0);
        }

        // Verify the string table is within bounds
        let string_table_offset = record_table_offset
            .checked_add(record_count.checked_mul(RECORD_ENTRY_SIZE).ok_or_else(|| {
                SearchError::IndexCorrupted {
                    path: path.to_owned(),
                    detail: "record table size overflow".into(),
                }
            })?)
            .ok_or_else(|| SearchError::IndexCorrupted {
                path: path.to_owned(),
                detail: "string table offset overflow".into(),
            })?;
        let bounds_end = string_table_offset.checked_add(max_end).ok_or_else(|| {
            SearchError::IndexCorrupted {
                path: path.to_owned(),
                detail: "string table bounds overflow".into(),
            }
        })?;
        if bounds_end > data.len() {
            return Err(SearchError::IndexCorrupted {
                path: path.to_owned(),
                detail: "string table extends beyond file".into(),
            });
        }

        Ok(max_end)
    }

    /// Number of vectors stored.
    #[must_use]
    pub fn record_count(&self) -> usize {
        self.header.record_count as usize
    }

    /// Embedding dimensionality.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.header.dimension as usize
    }

    /// Embedder identifier.
    #[must_use]
    pub fn embedder_id(&self) -> &str {
        &self.header.embedder_id
    }

    /// Embedder revision hash.
    #[must_use]
    pub fn embedder_revision(&self) -> &str {
        &self.header.embedder_revision
    }

    /// Quantization format.
    #[must_use]
    pub fn quantization(&self) -> Quantization {
        self.header.quantization
    }

    /// Get the doc_id for record at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= record_count()`.
    #[must_use]
    pub fn doc_id_at(&self, index: usize) -> &str {
        assert!(index < self.record_count(), "index out of bounds");
        let entry_offset = self.record_table_offset + index * RECORD_ENTRY_SIZE;
        let doc_offset = u32::from_le_bytes(
            self.mmap[entry_offset..entry_offset + 4]
                .try_into()
                .expect("4 bytes"),
        ) as usize;
        let doc_len = u16::from_le_bytes(
            self.mmap[entry_offset + 4..entry_offset + 6]
                .try_into()
                .expect("2 bytes"),
        ) as usize;
        let start = self.string_table_offset + doc_offset;
        std::str::from_utf8(&self.mmap[start..start + doc_len]).expect("valid UTF-8 doc_id")
    }

    /// Get the vector at `index` as a `Vec<f16>`.
    ///
    /// Converts from the raw byte representation element-by-element
    /// (safe, no pointer casts needed).
    ///
    /// # Panics
    ///
    /// Panics if `index >= record_count()` or quantization is not F16.
    #[must_use]
    pub fn vector_f16_at(&self, index: usize) -> Vec<f16> {
        assert!(index < self.record_count(), "index out of bounds");
        assert_eq!(
            self.header.quantization,
            Quantization::F16,
            "vector_f16_at requires F16 quantization"
        );
        let dim = self.header.dimension as usize;
        let byte_offset = self.vector_slab_offset + index * dim * 2;
        let byte_end = byte_offset + dim * 2;
        let bytes = &self.mmap[byte_offset..byte_end];

        bytes
            .chunks_exact(2)
            .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]))
            .collect()
    }

    /// Get the vector at `index` converted to f32.
    ///
    /// For F16 indices, this converts each element from f16 to f32.
    /// For F32 indices, this copies the raw f32 values.
    ///
    /// # Panics
    ///
    /// Panics if `index >= record_count()`.
    #[must_use]
    pub fn vector_f32_at(&self, index: usize) -> Vec<f32> {
        assert!(index < self.record_count(), "index out of bounds");
        let dim = self.header.dimension as usize;
        let elem_size = self.header.quantization.element_size();
        let byte_offset = self.vector_slab_offset + index * dim * elem_size;

        match self.header.quantization {
            Quantization::F16 => {
                let byte_end = byte_offset + dim * 2;
                let bytes = &self.mmap[byte_offset..byte_end];
                bytes
                    .chunks_exact(2)
                    .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
                    .collect()
            }
            Quantization::F32 => {
                let byte_end = byte_offset + dim * 4;
                let bytes = &self.mmap[byte_offset..byte_end];
                bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("4 bytes")))
                    .collect()
            }
        }
    }
}

// ─── VectorIndexWriter ──────────────────────────────────────────────────────

/// Builder for creating new FSVI index files.
///
/// Records are appended one at a time. The file is finalized (with header
/// fixup) when [`finish()`](Self::finish) is called.
///
/// # Usage
///
/// ```no_run
/// use frankensearch_index::fsvi::{VectorIndexWriter, Quantization};
/// use std::path::Path;
///
/// let mut writer = VectorIndexWriter::create(
///     Path::new("index.fsvi"),
///     "potion-128M",
///     "abc123",
///     256,
///     Quantization::F16,
/// ).unwrap();
///
/// writer.write_record("doc-1", &[0.1, 0.2, 0.3]).unwrap();
/// writer.finish().unwrap();
/// ```
pub struct VectorIndexWriter {
    /// Buffered file writer.
    writer: BufWriter<File>,
    /// File path for error messages and fsync.
    path: String,
    /// Embedder identifier.
    embedder_id: String,
    /// Embedder revision.
    embedder_revision: String,
    /// Embedding dimensionality.
    dimension: u16,
    /// Quantization format.
    quantization: Quantization,
    /// Accumulated doc_id strings.
    doc_ids: Vec<String>,
    /// Accumulated vectors (f32, to be quantized on finish).
    vectors: Vec<Vec<f32>>,
}

impl VectorIndexWriter {
    /// Create a new FSVI index writer.
    ///
    /// The file is NOT written until [`finish()`](Self::finish) is called.
    /// All records are buffered in memory during construction.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::Io` if the file cannot be created.
    pub fn create(
        path: &Path,
        embedder_id: &str,
        embedder_revision: &str,
        dimension: u16,
        quantization: Quantization,
    ) -> SearchResult<Self> {
        let file = File::create(path).map_err(SearchError::Io)?;
        let writer = BufWriter::with_capacity(WRITE_BUFFER_SIZE, file);

        Ok(Self {
            writer,
            path: path.display().to_string(),
            embedder_id: embedder_id.to_owned(),
            embedder_revision: embedder_revision.to_owned(),
            dimension,
            quantization,
            doc_ids: Vec::new(),
            vectors: Vec::new(),
        })
    }

    /// Append a record (doc_id + embedding vector).
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` if the vector length
    /// doesn't match the configured dimension.
    pub fn write_record(&mut self, doc_id: &str, embedding: &[f32]) -> SearchResult<()> {
        if embedding.len() != self.dimension as usize {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension as usize,
                actual: embedding.len(),
            });
        }

        self.doc_ids.push(doc_id.to_owned());
        self.vectors.push(embedding.to_vec());
        Ok(())
    }

    /// Number of records written so far.
    #[must_use]
    pub fn count(&self) -> usize {
        self.doc_ids.len()
    }

    /// Finalize the index: write header, record table, string table,
    /// padding, and vector slab. Flushes and fsyncs the file.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::Io` if writing or fsyncing fails.
    pub fn finish(mut self) -> SearchResult<()> {
        let record_count = self.doc_ids.len() as u64;

        // Build header
        let header = FsviHeader {
            version: FORMAT_VERSION,
            dimension: self.dimension,
            quantization: self.quantization,
            embedder_id: self.embedder_id.clone(),
            embedder_revision: self.embedder_revision.clone(),
            record_count,
            header_size: 0, // Computed during serialization
        };
        let header_bytes = header.to_bytes();

        // Build string table and record entries
        let mut string_table = Vec::new();
        let mut record_entries = Vec::with_capacity(self.doc_ids.len() * RECORD_ENTRY_SIZE);

        for doc_id in &self.doc_ids {
            let offset = string_table.len();
            let id_bytes = doc_id.as_bytes();
            #[allow(clippy::cast_possible_truncation)]
            {
                write_u32(&mut record_entries, offset as u32);
                write_u16(&mut record_entries, id_bytes.len() as u16);
            }
            string_table.extend_from_slice(id_bytes);
        }

        // Compute vector slab offset with 64-byte alignment
        let raw_slab_start = header_bytes.len() + record_entries.len() + string_table.len();
        let vector_slab_offset = align_up(raw_slab_start, VECTOR_ALIGN_BYTES);
        let padding_size = vector_slab_offset - raw_slab_start;

        // Write everything
        self.writer
            .write_all(&header_bytes)
            .map_err(SearchError::Io)?;
        self.writer
            .write_all(&record_entries)
            .map_err(SearchError::Io)?;
        self.writer
            .write_all(&string_table)
            .map_err(SearchError::Io)?;

        // Padding
        if padding_size > 0 {
            let padding = vec![0u8; padding_size];
            self.writer.write_all(&padding).map_err(SearchError::Io)?;
        }

        // Vector slab
        match self.quantization {
            Quantization::F16 => {
                for vec in &self.vectors {
                    for &val in vec {
                        let f16_val = f16::from_f32(val);
                        self.writer
                            .write_all(&f16_val.to_le_bytes())
                            .map_err(SearchError::Io)?;
                    }
                }
            }
            Quantization::F32 => {
                for vec in &self.vectors {
                    for &val in vec {
                        self.writer
                            .write_all(&val.to_le_bytes())
                            .map_err(SearchError::Io)?;
                    }
                }
            }
        }

        // Flush and fsync
        self.writer.flush().map_err(SearchError::Io)?;
        let file = self
            .writer
            .into_inner()
            .map_err(|e| SearchError::Io(e.into_error()))?;
        file.sync_all().map_err(SearchError::Io)?;

        // fsync parent directory for durability
        if let Some(parent) = Path::new(&self.path).parent() {
            if let Ok(dir) = fs::File::open(parent) {
                if let Err(sync_err) = dir.sync_all() {
                    tracing::warn!(
                        dir = %parent.display(),
                        error = %sync_err,
                        "directory fsync failed after FSVI index write"
                    );
                }
            }
        }

        Ok(())
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Round `val` up to the next multiple of `align`.
#[must_use]
const fn align_up(val: usize, align: usize) -> usize {
    (val + align - 1) & !(align - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("frankensearch_test");
        fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    // ─── Round-trip tests ───────────────────────────────────────────────

    #[test]
    fn roundtrip_f16_basic() {
        let path = temp_path("roundtrip_f16.fsvi");
        let dim = 4;
        let vecs: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];

        // Write
        let mut writer =
            VectorIndexWriter::create(&path, "test-embedder", "rev1", dim, Quantization::F16)
                .unwrap();
        for (i, v) in vecs.iter().enumerate() {
            writer.write_record(&format!("doc-{i}"), v).unwrap();
        }
        writer.finish().unwrap();

        // Read
        let index = VectorIndex::open(&path).unwrap();
        assert_eq!(index.record_count(), 3);
        assert_eq!(index.dimension(), dim as usize);
        assert_eq!(index.embedder_id(), "test-embedder");
        assert_eq!(index.embedder_revision(), "rev1");
        assert_eq!(index.quantization(), Quantization::F16);

        for (i, original) in vecs.iter().enumerate() {
            assert_eq!(index.doc_id_at(i), format!("doc-{i}"));
            let recovered = index.vector_f32_at(i);
            assert_eq!(recovered.len(), dim as usize);
            for (a, b) in original.iter().zip(recovered.iter()) {
                assert!((a - b).abs() < 0.001, "f16 roundtrip error: {a} vs {b}");
            }
        }

        fs::remove_file(&path).ok();
    }

    #[test]
    fn roundtrip_f32_basic() {
        let path = temp_path("roundtrip_f32.fsvi");
        let dim = 3;
        let vecs = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];

        let mut writer =
            VectorIndexWriter::create(&path, "f32-embedder", "", dim, Quantization::F32).unwrap();
        for (i, v) in vecs.iter().enumerate() {
            writer.write_record(&format!("f32-{i}"), v).unwrap();
        }
        writer.finish().unwrap();

        let index = VectorIndex::open(&path).unwrap();
        assert_eq!(index.record_count(), 2);
        assert_eq!(index.quantization(), Quantization::F32);

        for (i, original) in vecs.iter().enumerate() {
            let recovered = index.vector_f32_at(i);
            for (a, b) in original.iter().zip(recovered.iter()) {
                assert!(
                    (a - b).abs() < f32::EPSILON,
                    "f32 roundtrip error: {a} vs {b}"
                );
            }
        }

        fs::remove_file(&path).ok();
    }

    // ─── Empty index ────────────────────────────────────────────────────

    #[test]
    fn empty_index_roundtrip() {
        let path = temp_path("empty.fsvi");
        let writer =
            VectorIndexWriter::create(&path, "empty-embedder", "rev0", 384, Quantization::F16)
                .unwrap();
        assert_eq!(writer.count(), 0);
        writer.finish().unwrap();

        let index = VectorIndex::open(&path).unwrap();
        assert_eq!(index.record_count(), 0);
        assert_eq!(index.dimension(), 384);
        assert_eq!(index.embedder_id(), "empty-embedder");

        fs::remove_file(&path).ok();
    }

    // ─── Dimension mismatch ─────────────────────────────────────────────

    #[test]
    fn dimension_mismatch_rejected() {
        let path = temp_path("dim_mismatch.fsvi");
        let mut writer =
            VectorIndexWriter::create(&path, "test", "r1", 3, Quantization::F16).unwrap();

        let result = writer.write_record("doc", &[1.0, 2.0]); // 2 != 3
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(
                err,
                SearchError::DimensionMismatch {
                    expected: 3,
                    actual: 2
                }
            ),
            "expected DimensionMismatch, got: {err}"
        );

        fs::remove_file(&path).ok();
    }

    // ─── Header CRC corruption ──────────────────────────────────────────

    #[test]
    fn crc_corruption_detected() {
        let path = temp_path("crc_corrupt.fsvi");

        // Write valid index
        let mut writer =
            VectorIndexWriter::create(&path, "test", "r1", 2, Quantization::F16).unwrap();
        writer.write_record("doc-0", &[1.0, 0.0]).unwrap();
        writer.finish().unwrap();

        // Corrupt a byte in the header
        let mut data = fs::read(&path).unwrap();
        data[5] ^= 0xFF; // Flip bits in version field
        fs::write(&path, &data).unwrap();

        let result = VectorIndex::open(&path);
        assert!(result.is_err());
        let err_str = format!("{}", result.unwrap_err());
        assert!(
            err_str.contains("CRC") || err_str.contains("crc") || err_str.contains("mismatch"),
            "expected CRC error, got: {err_str}"
        );

        fs::remove_file(&path).ok();
    }

    // ─── 64-byte alignment ──────────────────────────────────────────────

    #[test]
    fn vector_slab_is_64_byte_aligned() {
        let path = temp_path("alignment.fsvi");

        let mut writer =
            VectorIndexWriter::create(&path, "align-test", "r1", 4, Quantization::F16).unwrap();
        writer.write_record("doc-0", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        writer.finish().unwrap();

        let index = VectorIndex::open(&path).unwrap();
        assert_eq!(
            index.vector_slab_offset % VECTOR_ALIGN_BYTES,
            0,
            "vector slab at offset {} is not 64-byte aligned",
            index.vector_slab_offset
        );

        fs::remove_file(&path).ok();
    }

    // ─── f16 quantization fidelity ──────────────────────────────────────

    #[test]
    fn f16_quantization_fidelity() {
        let path = temp_path("f16_fidelity.fsvi");
        let dim = 8;
        let vec = vec![0.123_456, -0.789_012, 0.5, -0.5, 1.0, -1.0, 0.001, 0.999];

        let mut writer =
            VectorIndexWriter::create(&path, "fidelity", "r1", dim, Quantization::F16).unwrap();
        writer.write_record("doc-0", &vec).unwrap();
        writer.finish().unwrap();

        let index = VectorIndex::open(&path).unwrap();
        let recovered = index.vector_f32_at(0);

        for (a, b) in vec.iter().zip(recovered.iter()) {
            assert!(
                (a - b).abs() < 0.001,
                "f16 fidelity: {a} -> {b} (delta={})",
                (a - b).abs()
            );
        }

        fs::remove_file(&path).ok();
    }

    // ─── Large index ────────────────────────────────────────────────────

    #[test]
    fn large_index_roundtrip() {
        let path = temp_path("large.fsvi");
        let dim = 16;
        let n = 1000;

        let mut writer =
            VectorIndexWriter::create(&path, "large-test", "r1", dim, Quantization::F16).unwrap();

        for i in 0..n {
            let vec: Vec<f32> = (0..dim)
                .map(|d| ((i * dim as usize + d) as f32) * 0.001)
                .collect();
            writer.write_record(&format!("doc-{i:04}"), &vec).unwrap();
        }
        writer.finish().unwrap();

        let index = VectorIndex::open(&path).unwrap();
        assert_eq!(index.record_count(), n);

        // Spot-check first, middle, and last
        assert_eq!(index.doc_id_at(0), "doc-0000");
        assert_eq!(index.doc_id_at(500), "doc-0500");
        assert_eq!(index.doc_id_at(999), "doc-0999");

        // Verify vector data for first record
        let v0 = index.vector_f32_at(0);
        assert_eq!(v0.len(), dim as usize);
        assert!((v0[0] - 0.0).abs() < 0.001);
        assert!((v0[1] - 0.001).abs() < 0.001);

        fs::remove_file(&path).ok();
    }

    // ─── Align helper ───────────────────────────────────────────────────

    #[test]
    fn align_up_correct() {
        assert_eq!(align_up(0, 64), 0);
        assert_eq!(align_up(1, 64), 64);
        assert_eq!(align_up(63, 64), 64);
        assert_eq!(align_up(64, 64), 64);
        assert_eq!(align_up(65, 64), 128);
    }

    // ─── Doc ID retrieval ───────────────────────────────────────────────

    #[test]
    fn variable_length_doc_ids() {
        let path = temp_path("var_ids.fsvi");
        let dim = 2;

        let mut writer =
            VectorIndexWriter::create(&path, "test", "r1", dim, Quantization::F16).unwrap();
        writer.write_record("a", &[1.0, 0.0]).unwrap();
        writer
            .write_record("a-longer-document-id", &[0.0, 1.0])
            .unwrap();
        writer.write_record("unicode-café", &[0.5, 0.5]).unwrap();
        writer.finish().unwrap();

        let index = VectorIndex::open(&path).unwrap();
        assert_eq!(index.doc_id_at(0), "a");
        assert_eq!(index.doc_id_at(1), "a-longer-document-id");
        assert_eq!(index.doc_id_at(2), "unicode-café");

        fs::remove_file(&path).ok();
    }
}
