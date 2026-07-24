#![allow(clippy::module_name_repetitions)]
//! FSLX segment container encoding and validation.
//!
//! This module deliberately treats section payloads as opaque bytes. Grimoire,
//! Quiver, and later codecs own their payload grammars; the segment container
//! owns only framing, alignment, schema-derived presence, and checksums.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use frankensearch_index::mapped_file::ReadOnlyMappedFile;
use xxhash_rust::xxh3::{Xxh3, xxh3_64};

use crate::error::QuillError;
use crate::schema::{FieldKind, SchemaDescriptor};

/// Magic bytes at the start of every FSLX segment.
pub const FSLX_SEGMENT_MAGIC: [u8; 8] = *b"FSLXSEG\0";
/// The only FSLX segment format version understood by this reader.
pub const FSLX_FORMAT_VERSION: u32 = 1;
/// Required byte alignment of every section start.
pub const FSLX_SECTION_ALIGNMENT: usize = 64;
/// Unknown section kinds are skippable only when this flag is the sole flag.
pub const OPTIONAL_SKIPPABLE: u16 = 1;

const FILE_PREFIX_LEN: usize = 16;
const FIXED_HEADER_LEN: usize = 56;
const SECTION_ENTRY_LEN: usize = 28;
const HEADER_CRC_LEN: usize = 4;
const TRAILER_LEN: usize = 12;
const DEFAULT_MAX_FILE_BYTES: u64 = u64::MAX;
const DEFAULT_MAX_HEADER_BYTES: usize = 2 * 1024 * 1024;
const DEFAULT_MAX_SECTIONS: usize = 65_535;
const MEMORY_PATH: &str = "<in-memory FSLX segment>";
const MAX_DOCID_EXCLUSIVE: u64 = 1_u64 << 32;

/// Stable numeric identity of one FSLX section kind.
///
/// A newtype, rather than a closed enum, preserves forward-compatible unknown
/// optional kinds.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SectionKind(u16);

impl SectionKind {
    /// Prefix-block term dictionary.
    pub const TERMDICT: Self = Self(1);
    /// Posting-list streams.
    pub const POSTINGS: Self = Self(2);
    /// Optional token-position streams.
    pub const POSITIONS: Self = Self(3);
    /// Block-max and skip metadata.
    pub const BLOCKMAX: Self = Self(4);
    /// Per-field fieldnorm columns.
    pub const DOCLEN: Self = Self(5);
    /// Global docid to external identifier mapping.
    pub const IDMAP: Self = Self(6);
    /// External identifier to global docid hash table.
    pub const IDHASH: Self = Self(7);
    /// Optional indexed or fast numeric columns.
    pub const NUMERIC: Self = Self(8);
    /// Optional stored-field bytes.
    pub const STOREDMETA: Self = Self(9);
    /// Per-field at-seal statistics.
    pub const STATS: Self = Self(10);

    /// Preserve a raw durable section kind, including future kinds.
    #[must_use]
    pub const fn from_raw(raw: u16) -> Self {
        Self(raw)
    }

    /// Return the durable `u16` section kind.
    #[must_use]
    pub const fn raw(self) -> u16 {
        self.0
    }

    const fn is_known(self) -> bool {
        self.0 >= Self::TERMDICT.0 && self.0 <= Self::STATS.0
    }
}

/// One opaque payload supplied to the canonical segment writer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SectionInput<'a> {
    /// Stable section kind.
    pub kind: SectionKind,
    /// Versioned section flags.
    pub flags: u16,
    /// Exact payload bytes, excluding alignment padding.
    pub bytes: &'a [u8],
}

impl<'a> SectionInput<'a> {
    /// Construct an ordinary known section with no flags.
    #[must_use]
    pub const fn new(kind: SectionKind, bytes: &'a [u8]) -> Self {
        Self {
            kind,
            flags: 0,
            bytes,
        }
    }
}

/// Caller-supplied immutable header fields for a fresh segment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SegmentHeaderInput {
    /// Random collision-checked segment identifier.
    pub segment_id: u64,
    /// Compile-time schema governing conditional sections.
    pub schema: SchemaDescriptor,
    /// Inclusive global docid lower bound.
    pub docid_lo: u64,
    /// Exclusive global docid upper bound.
    pub docid_hi: u64,
    /// Documents live when the segment was sealed.
    pub doc_count: u32,
    /// Informational creation timestamp.
    pub created_unix_s: i64,
    /// Informational packed engine version.
    pub engine_version: u32,
}

/// Parsed immutable fields from an FSLX segment header.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SegmentHeader {
    /// Random collision-checked segment identifier.
    pub segment_id: u64,
    /// Stable schema descriptor hash.
    pub schema_id: u64,
    /// Inclusive global docid lower bound.
    pub docid_lo: u64,
    /// Exclusive global docid upper bound.
    pub docid_hi: u64,
    /// Documents live when the segment was sealed.
    pub doc_count: u32,
    /// Informational creation timestamp.
    pub created_unix_s: i64,
    /// Informational packed engine version.
    pub engine_version: u32,
    /// Number of entries in the section table.
    pub section_count: u16,
}

/// Validated section-table metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SectionEntry {
    /// Stable or forward-compatible section kind.
    pub kind: SectionKind,
    /// Versioned section flags.
    pub flags: u16,
    /// Absolute, 64-byte-aligned file offset.
    pub offset: u64,
    /// Exact payload length, excluding padding.
    pub len: u64,
    /// xxh3-64 over the exact payload bytes.
    pub xxh3: u64,
}

/// Explicit resource ceilings for hostile segment input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SegmentLimits {
    /// Maximum complete segment file size.
    pub max_file_bytes: u64,
    /// Maximum header block size.
    pub max_header_bytes: usize,
    /// Maximum retained section metadata entries.
    pub max_sections: usize,
}

impl Default for SegmentLimits {
    fn default() -> Self {
        Self {
            max_file_bytes: DEFAULT_MAX_FILE_BYTES,
            max_header_bytes: DEFAULT_MAX_HEADER_BYTES,
            max_sections: DEFAULT_MAX_SECTIONS,
        }
    }
}

/// Internal fault-injection points in the canonical segment temp writer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SegmentWriteCheckpoint {
    /// All bytes reached the file handle but have not been fsynced.
    TempWritten,
    /// The complete temp file has been fsynced.
    TempSynced,
}

/// Owned canonical FSLX bytes ready for Keeper publication.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedSegment {
    bytes: Vec<u8>,
    header: SegmentHeader,
    sections: Vec<SectionEntry>,
    file_xxh3: u64,
    source_xxh3: u64,
}

/// Exact layout declaration for one section emitted directly into a segment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PlannedSection {
    kind: SectionKind,
    flags: u16,
    len: usize,
}

impl PlannedSection {
    /// Declare one ordinary known section with no flags.
    pub(crate) const fn new(kind: SectionKind, len: usize) -> Self {
        Self {
            kind,
            flags: 0,
            len,
        }
    }
}

/// Pre-sized canonical FSLX container for direct section emission.
///
/// The assembler owns the final allocation from the outset. Callers write each
/// declared payload into its exact final slice, then [`Self::finish`] patches
/// the section hashes, header CRC, and trailer checksums.
pub(crate) struct SegmentAssembler {
    bytes: Vec<u8>,
    header: SegmentHeader,
    sections: Vec<SectionEntry>,
    next_section: usize,
    header_len: usize,
}

impl SegmentAssembler {
    /// Validate a complete section plan and allocate its exact canonical file.
    pub(crate) fn new(
        header: SegmentHeaderInput,
        planned: &[PlannedSection],
    ) -> Result<Self, QuillError> {
        let limits = SegmentLimits::default();
        header.schema.validate()?;
        validate_header_range(header.docid_lo, header.docid_hi, header.doc_count)
            .map_err(invalid_segment)?;
        validate_section_sequence(
            header.schema,
            planned.len(),
            |index| {
                let section = planned[index];
                (section.kind, section.flags)
            },
            false,
        )
        .map_err(invalid_segment)?;

        if planned.len() > limits.max_sections {
            return Err(resource(
                "section table",
                format!(
                    "section count {} exceeds limit {}",
                    planned.len(),
                    limits.max_sections
                ),
            ));
        }
        let section_count = u16::try_from(planned.len())
            .map_err(|_| invalid_segment("section count does not fit u16"))?;
        let header_len = header_len(planned.len()).map_err(invalid_segment)?;
        if header_len > limits.max_header_bytes {
            return Err(resource(
                "segment header",
                format!(
                    "header length {header_len} exceeds limit {}",
                    limits.max_header_bytes
                ),
            ));
        }
        let mut cursor = FILE_PREFIX_LEN
            .checked_add(header_len)
            .and_then(|value| value.checked_add(HEADER_CRC_LEN))
            .ok_or_else(|| invalid_segment("header end overflow"))?;
        let mut sections = Vec::new();
        sections
            .try_reserve_exact(planned.len())
            .map_err(|_| resource("section table", "allocation failed"))?;
        for section in planned {
            cursor = align_up(cursor, FSLX_SECTION_ALIGNMENT)
                .ok_or_else(|| invalid_segment("section alignment overflow"))?;
            let offset = u64::try_from(cursor)
                .map_err(|_| invalid_segment("section offset does not fit u64"))?;
            let len = u64::try_from(section.len)
                .map_err(|_| invalid_segment("section length does not fit u64"))?;
            sections.push(SectionEntry {
                kind: section.kind,
                flags: section.flags,
                offset,
                len,
                xxh3: 0,
            });
            cursor = cursor
                .checked_add(section.len)
                .ok_or_else(|| invalid_segment("section end overflow"))?;
        }
        let file_len = cursor
            .checked_add(TRAILER_LEN)
            .ok_or_else(|| invalid_segment("file length overflow"))?;
        let file_len_u64 =
            u64::try_from(file_len).map_err(|_| invalid_segment("file length does not fit u64"))?;
        if file_len_u64 > limits.max_file_bytes {
            return Err(resource(
                "segment file",
                format!(
                    "file length {file_len_u64} exceeds limit {}",
                    limits.max_file_bytes
                ),
            ));
        }

        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(file_len)
            .map_err(|_| resource("segment file", "allocation failed"))?;
        let header_end = FILE_PREFIX_LEN
            .checked_add(header_len)
            .and_then(|value| value.checked_add(HEADER_CRC_LEN))
            .ok_or_else(|| invalid_segment("header end overflow"))?;
        bytes.resize(header_end, 0);
        Ok(Self {
            bytes,
            header: SegmentHeader {
                segment_id: header.segment_id,
                schema_id: header.schema.schema_id()?,
                docid_lo: header.docid_lo,
                docid_hi: header.docid_hi,
                doc_count: header.doc_count,
                created_unix_s: header.created_unix_s,
                engine_version: header.engine_version,
                section_count,
            },
            sections,
            next_section: 0,
            header_len,
        })
    }

    /// Append one payload at its exact planned offset without reallocating.
    pub(crate) fn write_section(
        &mut self,
        kind: SectionKind,
        write: impl FnOnce(&mut Vec<u8>),
    ) -> Result<(), QuillError> {
        let entry = self
            .sections
            .get(self.next_section)
            .copied()
            .ok_or_else(|| invalid_segment(format!("unplanned section kind {}", kind.raw())))?;
        if entry.kind != kind {
            return Err(invalid_segment(format!(
                "section writer supplied kind {}, expected {}",
                kind.raw(),
                entry.kind.raw()
            )));
        }
        let offset = usize::try_from(entry.offset)
            .map_err(|_| invalid_segment("section offset does not fit usize"))?;
        let len = usize::try_from(entry.len)
            .map_err(|_| invalid_segment("section length does not fit usize"))?;
        let expected_end = offset
            .checked_add(len)
            .ok_or_else(|| invalid_segment("section end overflow"))?;
        self.bytes.resize(offset, 0);
        write(&mut self.bytes);
        if self.bytes.len() != expected_end {
            return Err(invalid_segment(format!(
                "section kind {} writer ended at {}, expected {}",
                kind.raw(),
                self.bytes.len(),
                expected_end
            )));
        }
        self.next_section += 1;
        Ok(())
    }

    /// Copy one small prebuilt payload into its exact final section slice.
    pub(crate) fn copy_section(
        &mut self,
        kind: SectionKind,
        bytes: &[u8],
    ) -> Result<(), QuillError> {
        self.write_section(kind, |output| output.extend_from_slice(bytes))
    }

    /// Borrow one fully emitted section before the remaining plan is written.
    pub(crate) fn written_section(&self, kind: SectionKind) -> Result<&[u8], QuillError> {
        let index = self
            .sections
            .iter()
            .position(|entry| entry.kind == kind)
            .ok_or_else(|| invalid_segment(format!("unplanned section kind {}", kind.raw())))?;
        if index >= self.next_section {
            return Err(invalid_segment(format!(
                "section kind {} has not been written",
                kind.raw()
            )));
        }
        let entry = self.sections[index];
        let start = usize::try_from(entry.offset)
            .map_err(|_| invalid_segment("section offset does not fit usize"))?;
        let len = usize::try_from(entry.len)
            .map_err(|_| invalid_segment("section length does not fit usize"))?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| invalid_segment("section end overflow"))?;
        self.bytes
            .get(start..end)
            .ok_or_else(|| invalid_segment("written section exceeds segment bytes"))
    }

    /// Finalize hashes and framing after every planned section was emitted.
    pub(crate) fn finish(mut self) -> Result<EncodedSegment, QuillError> {
        if self.next_section != self.sections.len() {
            return Err(invalid_segment(format!(
                "planned section kind {} was not written",
                self.sections[self.next_section].kind.raw()
            )));
        }
        for entry in &mut self.sections {
            let start = usize::try_from(entry.offset)
                .map_err(|_| invalid_segment("section offset does not fit usize"))?;
            let len = usize::try_from(entry.len)
                .map_err(|_| invalid_segment("section length does not fit usize"))?;
            let end = start
                .checked_add(len)
                .ok_or_else(|| invalid_segment("section end overflow"))?;
            let section = self
                .bytes
                .get(start..end)
                .ok_or_else(|| invalid_segment("planned section exceeds segment bytes"))?;
            entry.xxh3 = xxh3_64(section);
        }

        let header_len_u32 = u32::try_from(self.header_len)
            .map_err(|_| invalid_segment("header length does not fit u32"))?;
        let mut header_bytes = Vec::new();
        header_bytes
            .try_reserve_exact(self.header_len)
            .map_err(|_| resource("segment header", "allocation failed"))?;
        write_header_fixed(&mut header_bytes, self.header);
        for entry in &self.sections {
            write_section_entry(&mut header_bytes, *entry);
        }
        debug_assert_eq!(header_bytes.len(), self.header_len);
        self.bytes[..FSLX_SEGMENT_MAGIC.len()].copy_from_slice(&FSLX_SEGMENT_MAGIC);
        self.bytes[8..12].copy_from_slice(&FSLX_FORMAT_VERSION.to_le_bytes());
        self.bytes[12..FILE_PREFIX_LEN].copy_from_slice(&header_len_u32.to_le_bytes());
        let header_end = FILE_PREFIX_LEN
            .checked_add(self.header_len)
            .ok_or_else(|| invalid_segment("header end overflow"))?;
        self.bytes[FILE_PREFIX_LEN..header_end].copy_from_slice(&header_bytes);
        let crc_end = header_end
            .checked_add(HEADER_CRC_LEN)
            .ok_or_else(|| invalid_segment("header CRC end overflow"))?;
        self.bytes[header_end..crc_end].copy_from_slice(&crc32(&header_bytes).to_le_bytes());

        let mut source_hasher = Xxh3::new();
        source_hasher.update(&self.bytes);
        let file_xxh3 = source_hasher.digest();
        let hash_bytes = file_xxh3.to_le_bytes();
        let trailer_crc_bytes = crc32(&hash_bytes).to_le_bytes();
        self.bytes.extend_from_slice(&hash_bytes);
        self.bytes.extend_from_slice(&trailer_crc_bytes);
        source_hasher.update(&hash_bytes);
        source_hasher.update(&trailer_crc_bytes);
        let source_xxh3 = source_hasher.digest();

        Ok(EncodedSegment {
            bytes: self.bytes,
            header: self.header,
            sections: self.sections,
            file_xxh3,
            source_xxh3,
        })
    }
}

impl EncodedSegment {
    /// Encode one canonical FSLX segment with default resource ceilings.
    ///
    /// # Errors
    ///
    /// Returns a typed invariant error for invalid metadata, section presence,
    /// flags, ordering, layout overflow, or resource exhaustion.
    pub fn encode(
        header: SegmentHeaderInput,
        sections: &[SectionInput<'_>],
    ) -> Result<Self, QuillError> {
        Self::encode_with_limits(header, sections, SegmentLimits::default())
    }

    /// Encode one canonical FSLX segment with explicit resource ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::encode`].
    pub fn encode_with_limits(
        header: SegmentHeaderInput,
        sections: &[SectionInput<'_>],
        limits: SegmentLimits,
    ) -> Result<Self, QuillError> {
        Self::encode_with_limits_impl(header, sections, limits, false)
    }

    fn encode_with_limits_impl(
        header: SegmentHeaderInput,
        sections: &[SectionInput<'_>],
        limits: SegmentLimits,
        allow_unregistered_extensions: bool,
    ) -> Result<Self, QuillError> {
        header.schema.validate()?;
        validate_header_range(header.docid_lo, header.docid_hi, header.doc_count)
            .map_err(invalid_segment)?;
        validate_section_sequence(
            header.schema,
            sections.len(),
            |index| {
                let section = sections[index];
                (section.kind, section.flags)
            },
            allow_unregistered_extensions,
        )
        .map_err(invalid_segment)?;

        if sections.len() > limits.max_sections {
            return Err(resource(
                "section table",
                format!(
                    "section count {} exceeds limit {}",
                    sections.len(),
                    limits.max_sections
                ),
            ));
        }
        let section_count = u16::try_from(sections.len())
            .map_err(|_| invalid_segment("section count does not fit u16"))?;
        let header_len = header_len(sections.len()).map_err(invalid_segment)?;
        if header_len > limits.max_header_bytes {
            return Err(resource(
                "segment header",
                format!(
                    "header length {header_len} exceeds limit {}",
                    limits.max_header_bytes
                ),
            ));
        }
        let header_len_u32 = u32::try_from(header_len)
            .map_err(|_| invalid_segment("header length does not fit u32"))?;
        let mut cursor = FILE_PREFIX_LEN
            .checked_add(header_len)
            .and_then(|value| value.checked_add(HEADER_CRC_LEN))
            .ok_or_else(|| invalid_segment("header end overflow"))?;

        let mut table = Vec::new();
        table
            .try_reserve_exact(sections.len())
            .map_err(|_| resource("section table", "allocation failed"))?;
        for section in sections {
            cursor = align_up(cursor, FSLX_SECTION_ALIGNMENT)
                .ok_or_else(|| invalid_segment("section alignment overflow"))?;
            let offset = u64::try_from(cursor)
                .map_err(|_| invalid_segment("section offset does not fit u64"))?;
            let len = u64::try_from(section.bytes.len())
                .map_err(|_| invalid_segment("section length does not fit u64"))?;
            table.push(SectionEntry {
                kind: section.kind,
                flags: section.flags,
                offset,
                len,
                xxh3: xxh3_64(section.bytes),
            });
            cursor = cursor
                .checked_add(section.bytes.len())
                .ok_or_else(|| invalid_segment("section end overflow"))?;
        }
        let file_len = cursor
            .checked_add(TRAILER_LEN)
            .ok_or_else(|| invalid_segment("file length overflow"))?;
        let file_len_u64 =
            u64::try_from(file_len).map_err(|_| invalid_segment("file length does not fit u64"))?;
        if file_len_u64 > limits.max_file_bytes {
            return Err(resource(
                "segment file",
                format!(
                    "file length {file_len_u64} exceeds limit {}",
                    limits.max_file_bytes
                ),
            ));
        }

        let schema_id = header.schema.schema_id()?;
        let parsed_header = SegmentHeader {
            segment_id: header.segment_id,
            schema_id,
            docid_lo: header.docid_lo,
            docid_hi: header.docid_hi,
            doc_count: header.doc_count,
            created_unix_s: header.created_unix_s,
            engine_version: header.engine_version,
            section_count,
        };
        let mut header_bytes = Vec::new();
        header_bytes
            .try_reserve_exact(header_len)
            .map_err(|_| resource("segment header", "allocation failed"))?;
        write_header_fixed(&mut header_bytes, parsed_header);
        for entry in &table {
            write_section_entry(&mut header_bytes, *entry);
        }
        debug_assert_eq!(header_bytes.len(), header_len);

        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(file_len)
            .map_err(|_| resource("segment file", "allocation failed"))?;
        bytes.extend_from_slice(&FSLX_SEGMENT_MAGIC);
        bytes.extend_from_slice(&FSLX_FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&header_len_u32.to_le_bytes());
        bytes.extend_from_slice(&header_bytes);
        bytes.extend_from_slice(&crc32(&header_bytes).to_le_bytes());
        for (section, entry) in sections.iter().zip(&table) {
            let offset = usize::try_from(entry.offset)
                .map_err(|_| invalid_segment("section offset does not fit usize"))?;
            bytes.resize(offset, 0);
            bytes.extend_from_slice(section.bytes);
        }
        debug_assert_eq!(bytes.len(), file_len - TRAILER_LEN);
        let mut source_hasher = Xxh3::new();
        source_hasher.update(&bytes);
        let file_xxh3 = source_hasher.digest();
        let hash_bytes = file_xxh3.to_le_bytes();
        let trailer_crc_bytes = crc32(&hash_bytes).to_le_bytes();
        bytes.extend_from_slice(&hash_bytes);
        bytes.extend_from_slice(&trailer_crc_bytes);
        source_hasher.update(&hash_bytes);
        source_hasher.update(&trailer_crc_bytes);
        let source_xxh3 = source_hasher.digest();
        debug_assert_eq!(bytes.len(), file_len);

        Ok(Self {
            bytes,
            header: parsed_header,
            sections: table,
            file_xxh3,
            source_xxh3,
        })
    }

    /// Exact canonical segment bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consume the wrapper and return its canonical bytes.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Parsed header represented by these bytes.
    #[must_use]
    pub const fn header(&self) -> SegmentHeader {
        self.header
    }

    /// Canonical section-table entries.
    #[must_use]
    pub fn section_entries(&self) -> &[SectionEntry] {
        &self.sections
    }

    /// Exact complete file length.
    #[must_use]
    pub fn file_len(&self) -> u64 {
        u64::try_from(self.bytes.len()).unwrap_or(u64::MAX)
    }

    /// xxh3-64 over the file prefix before the trailer, stored in the trailer.
    #[must_use]
    pub const fn file_xxh3(&self) -> u64 {
        self.file_xxh3
    }

    /// xxh3-64 over every canonical FSLX byte, including the trailer.
    ///
    /// This derived witness is not persisted in FSLX. It binds a durability
    /// sidecar to the complete source file without hashing the encoded bytes a
    /// second time.
    #[must_use]
    pub const fn source_xxh3(&self) -> u64 {
        self.source_xxh3
    }

    /// Create and durably sync the canonical Keeper temp file.
    ///
    /// This method never renames or publishes the file. It uses `create_new`,
    /// so an existing temp is preserved and reported as an I/O error.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the directory is unavailable, the temp already
    /// exists, a write fails, or `sync_all` fails.
    pub fn write_temp(&self, directory: &Path) -> Result<PendingSegmentFile, QuillError> {
        self.write_temp_with_observer(directory, |_, _| Ok(()))
    }

    /// Write the canonical temp, or adopt an exact synced temp from a prior attempt.
    ///
    /// A differing same-name artifact is preserved and rejected. This method is
    /// crate-private because only a retained `KeeperWriter` may establish that
    /// retrying the temp is serialized with every other directory mutation.
    pub(crate) fn write_temp_retryable(
        &self,
        directory: &Path,
    ) -> Result<PendingSegmentFile, QuillError> {
        let path = directory.join(format!(".tmp-segment-{:016x}", self.header.segment_id));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(mut file) => {
                file.write_all(&self.bytes)?;
                file.flush()?;
                file.sync_all()?;
            }
            Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists => {
                let metadata = fs::symlink_metadata(&path)?;
                if !metadata.file_type().is_file() || metadata.len() != self.file_len() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "existing segment temp is not the exact expected regular file",
                    )
                    .into());
                }
                let existing = fs::read(&path)?;
                if existing.as_slice() != self.bytes.as_slice() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::AlreadyExists,
                        "existing segment temp differs from the staged FSLX bytes",
                    )
                    .into());
                }
                OpenOptions::new().read(true).open(&path)?.sync_all()?;
            }
            Err(source) => return Err(source.into()),
        }
        Ok(self.pending_file(path))
    }

    fn pending_file(&self, path: PathBuf) -> PendingSegmentFile {
        PendingSegmentFile {
            path,
            segment_id: self.header.segment_id,
            file_len: self.file_len(),
            file_xxh3: self.file_xxh3,
            source_xxh3: self.source_xxh3,
        }
    }

    pub(crate) fn write_temp_with_observer<O>(
        &self,
        directory: &Path,
        mut observe: O,
    ) -> Result<PendingSegmentFile, QuillError>
    where
        O: FnMut(SegmentWriteCheckpoint, &Path) -> Result<(), QuillError>,
    {
        let path = directory.join(format!(".tmp-segment-{:016x}", self.header.segment_id));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        file.write_all(&self.bytes)?;
        file.flush()?;
        observe(SegmentWriteCheckpoint::TempWritten, &path)?;
        file.sync_all()?;
        observe(SegmentWriteCheckpoint::TempSynced, &path)?;
        Ok(self.pending_file(path))
    }
}

/// A synced but unpublished `.tmp-segment-*` artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PendingSegmentFile {
    path: PathBuf,
    segment_id: u64,
    file_len: u64,
    file_xxh3: u64,
    source_xxh3: u64,
}

impl PendingSegmentFile {
    /// Temp path awaiting Keeper rename and manifest publication.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Random segment identifier.
    #[must_use]
    pub const fn segment_id(&self) -> u64 {
        self.segment_id
    }

    /// Exact file length.
    #[must_use]
    pub const fn file_len(&self) -> u64 {
        self.file_len
    }

    /// xxh3-64 over the file prefix before the trailer.
    #[must_use]
    pub const fn file_xxh3(&self) -> u64 {
        self.file_xxh3
    }

    /// xxh3-64 over every canonical FSLX byte, including the trailer.
    ///
    /// This derived witness is not persisted in FSLX.
    #[must_use]
    pub const fn source_xxh3(&self) -> u64 {
        self.source_xxh3
    }
}

/// Validating FSLX reader over an owned, borrowed, or memory-mapped byte source.
pub struct SegmentReader<B: AsRef<[u8]>> {
    source: B,
    path: PathBuf,
    schema: SchemaDescriptor,
    limits: SegmentLimits,
    header: SegmentHeader,
    sections: Vec<SectionEntry>,
    section_checks: Vec<OnceLock<Result<(), String>>>,
    file_xxh3: u64,
}

impl SegmentReader<Vec<u8>> {
    /// Parse an owned in-memory segment with default resource ceilings.
    ///
    /// # Errors
    ///
    /// Returns a typed corruption, schema, or resource error.
    pub fn from_owned(bytes: Vec<u8>, schema: SchemaDescriptor) -> Result<Self, QuillError> {
        Self::from_owned_with_limits(bytes, schema, SegmentLimits::default())
    }

    /// Parse an owned in-memory segment with explicit resource ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::from_owned`].
    pub fn from_owned_with_limits(
        bytes: Vec<u8>,
        schema: SchemaDescriptor,
        limits: SegmentLimits,
    ) -> Result<Self, QuillError> {
        Self::parse_source(bytes, PathBuf::from(MEMORY_PATH), schema, limits)
    }
}

impl<'a> SegmentReader<&'a [u8]> {
    /// Parse borrowed segment bytes with default resource ceilings.
    ///
    /// # Errors
    ///
    /// Returns a typed corruption, schema, or resource error.
    pub fn from_bytes(bytes: &'a [u8], schema: SchemaDescriptor) -> Result<Self, QuillError> {
        Self::parse_source(
            bytes,
            PathBuf::from(MEMORY_PATH),
            schema,
            SegmentLimits::default(),
        )
    }
}

impl SegmentReader<ReadOnlyMappedFile> {
    /// Open and validate an immutable, published FSLX file through the shared mmap facade.
    ///
    /// Only canonical `seg-<hex16>.fslx` generation files are accepted. Keeper
    /// must finish the temp-file sync and atomic rename before calling this
    /// method, and published generations must never be mutated in place.
    ///
    /// # Errors
    ///
    /// Returns a typed I/O, corruption, schema, or resource error.
    pub fn open_published(path: &Path, schema: SchemaDescriptor) -> Result<Self, QuillError> {
        Self::open_published_with_limits(path, schema, SegmentLimits::default())
    }

    /// Open with explicit resource ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::open_published`].
    pub fn open_published_with_limits(
        path: &Path,
        schema: SchemaDescriptor,
        limits: SegmentLimits,
    ) -> Result<Self, QuillError> {
        let expected_segment_id =
            published_segment_id(path).map_err(|detail| corrupted(path, detail))?;
        let metadata = fs::symlink_metadata(path)?;
        if !metadata.file_type().is_file() {
            return Err(corrupted(path, "published segment is not a regular file"));
        }
        let file_len = metadata.len();
        if file_len > limits.max_file_bytes {
            return Err(resource(
                "segment file",
                format!(
                    "{} has length {file_len}, exceeding limit {}",
                    path.display(),
                    limits.max_file_bytes
                ),
            ));
        }
        if file_len
            < u64::try_from(FILE_PREFIX_LEN + FIXED_HEADER_LEN + HEADER_CRC_LEN + TRAILER_LEN)
                .unwrap_or(u64::MAX)
        {
            return Err(corrupted(path, format!("truncated file length {file_len}")));
        }
        let mapped = ReadOnlyMappedFile::open_published(path)?;
        let reader = Self::parse_source(mapped, path.to_path_buf(), schema, limits)?;
        if reader.header.segment_id != expected_segment_id {
            return Err(corrupted(
                path,
                format!(
                    "published filename identifies segment {expected_segment_id:#018x}, but the header identifies {:#018x}",
                    reader.header.segment_id
                ),
            ));
        }
        Ok(reader)
    }
}

impl<B: AsRef<[u8]>> SegmentReader<B> {
    fn parse_source(
        source: B,
        path: PathBuf,
        schema: SchemaDescriptor,
        limits: SegmentLimits,
    ) -> Result<Self, QuillError> {
        schema.validate()?;
        let parsed = parse_container(source.as_ref(), &path, schema, limits)?;
        let mut section_checks = Vec::new();
        section_checks
            .try_reserve_exact(parsed.sections.len())
            .map_err(|_| {
                resource(
                    "lazy section checks",
                    format!("allocation failed for {}", path.display()),
                )
            })?;
        for _ in &parsed.sections {
            section_checks.push(OnceLock::new());
        }
        Ok(Self {
            source,
            path,
            schema,
            limits,
            header: parsed.header,
            sections: parsed.sections,
            section_checks,
            file_xxh3: parsed.file_xxh3,
        })
    }

    /// Parsed immutable segment header.
    #[must_use]
    pub const fn header(&self) -> SegmentHeader {
        self.header
    }

    /// Validated section-table metadata in durable order.
    #[must_use]
    pub fn section_entries(&self) -> &[SectionEntry] {
        &self.sections
    }

    /// Exact complete file length.
    #[must_use]
    pub fn file_len(&self) -> u64 {
        u64::try_from(self.source.as_ref().len()).unwrap_or(u64::MAX)
    }

    /// Borrow the complete canonical FSLX image backing this reader.
    #[must_use]
    pub fn source_bytes(&self) -> &[u8] {
        self.source.as_ref()
    }

    /// xxh3-64 over the file prefix before the trailer, stored in the trailer.
    #[must_use]
    pub const fn file_xxh3(&self) -> u64 {
        self.file_xxh3
    }

    /// Verify one section once and borrow its exact payload bytes.
    ///
    /// Unknown optional kinds remain visible in [`Self::section_entries`] and
    /// are hashed by [`Self::verify`], but normal reads skip their payloads.
    /// `Ok(None)` means the known kind is absent or the requested kind is
    /// unknown to this build.
    ///
    /// # Errors
    ///
    /// Returns a typed corruption error on a checksum mismatch.
    pub fn section(&self, kind: SectionKind) -> Result<Option<&[u8]>, QuillError> {
        if !kind.is_known() {
            return Ok(None);
        }
        let Ok(index) = self
            .sections
            .binary_search_by_key(&kind, |entry| entry.kind)
        else {
            return Ok(None);
        };
        let entry = self.sections[index];
        let bytes = section_bytes(self.source.as_ref(), entry)
            .ok_or_else(|| corrupted(&self.path, "validated section range became invalid"))?;
        let result = self.section_checks[index].get_or_init(|| {
            let actual = xxh3_64(bytes);
            if actual == entry.xxh3 {
                Ok(())
            } else {
                Err(format!(
                    "section {} checksum mismatch: expected {:#018x}, got {actual:#018x}",
                    entry.kind.raw(),
                    entry.xxh3
                ))
            }
        });
        match result {
            Ok(()) => Ok(Some(bytes)),
            Err(detail) => Err(corrupted(&self.path, detail.clone())),
        }
    }

    /// Eagerly revalidate the structure, every section, and file-prefix witness.
    ///
    /// Unlike [`Self::section`], this always recomputes hashes so doctor flows
    /// do not trust an earlier lazy result.
    ///
    /// # Errors
    ///
    /// Returns a typed corruption error for any checksum mismatch.
    pub fn verify(&self) -> Result<(), QuillError> {
        let bytes = self.source.as_ref();
        let parsed = parse_container(bytes, &self.path, self.schema, self.limits)?;
        for entry in &parsed.sections {
            let payload = section_bytes(bytes, *entry)
                .ok_or_else(|| corrupted(&self.path, "validated section range became invalid"))?;
            let actual = xxh3_64(payload);
            if actual != entry.xxh3 {
                return Err(corrupted(
                    &self.path,
                    format!(
                        "section {} checksum mismatch: expected {:#018x}, got {actual:#018x}",
                        entry.kind.raw(),
                        entry.xxh3
                    ),
                ));
            }
        }
        let actual = xxh3_64(&bytes[..parsed.trailer_start]);
        if actual != parsed.file_xxh3 {
            return Err(corrupted(
                &self.path,
                format!(
                    "file checksum mismatch: expected {:#018x}, got {actual:#018x}",
                    parsed.file_xxh3
                ),
            ));
        }
        Ok(())
    }
}

struct ParsedContainer {
    header: SegmentHeader,
    sections: Vec<SectionEntry>,
    trailer_start: usize,
    file_xxh3: u64,
}

fn parse_container(
    bytes: &[u8],
    path: &Path,
    schema: SchemaDescriptor,
    limits: SegmentLimits,
) -> Result<ParsedContainer, QuillError> {
    let file_len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    if file_len > limits.max_file_bytes {
        return Err(resource(
            "segment file",
            format!(
                "{} has length {file_len}, exceeding limit {}",
                path.display(),
                limits.max_file_bytes
            ),
        ));
    }
    let trailer_start = bytes
        .len()
        .checked_sub(TRAILER_LEN)
        .ok_or_else(|| corrupted(path, "file is shorter than its trailer"))?;
    if bytes.get(..FSLX_SEGMENT_MAGIC.len()) != Some(FSLX_SEGMENT_MAGIC.as_slice()) {
        return Err(corrupted(path, "invalid FSLX segment magic"));
    }
    let version =
        read_u32_at(bytes, 8).ok_or_else(|| corrupted(path, "truncated format version"))?;
    if version != FSLX_FORMAT_VERSION {
        return Err(corrupted(
            path,
            format!("unsupported FSLX format version {version}"),
        ));
    }
    let declared_header_len = usize::try_from(
        read_u32_at(bytes, 12).ok_or_else(|| corrupted(path, "truncated header length"))?,
    )
    .map_err(|_| corrupted(path, "header length does not fit usize"))?;
    if declared_header_len < FIXED_HEADER_LEN {
        return Err(corrupted(
            path,
            format!("header length {declared_header_len} is shorter than {FIXED_HEADER_LEN}"),
        ));
    }
    if declared_header_len > limits.max_header_bytes {
        return Err(resource(
            "segment header",
            format!(
                "{} declares length {declared_header_len}, exceeding limit {}",
                path.display(),
                limits.max_header_bytes
            ),
        ));
    }
    let header_end = FILE_PREFIX_LEN
        .checked_add(declared_header_len)
        .ok_or_else(|| corrupted(path, "header end overflow"))?;
    let header_crc_end = header_end
        .checked_add(HEADER_CRC_LEN)
        .ok_or_else(|| corrupted(path, "header CRC end overflow"))?;
    if header_crc_end > trailer_start {
        return Err(corrupted(path, "header overlaps the file trailer"));
    }
    let header_bytes = bytes
        .get(FILE_PREFIX_LEN..header_end)
        .ok_or_else(|| corrupted(path, "truncated header block"))?;
    let expected_header_crc =
        read_u32_at(bytes, header_end).ok_or_else(|| corrupted(path, "truncated header CRC"))?;
    let actual_header_crc = crc32(header_bytes);
    if actual_header_crc != expected_header_crc {
        return Err(corrupted(
            path,
            format!(
                "header CRC mismatch: expected {expected_header_crc:#010x}, got {actual_header_crc:#010x}"
            ),
        ));
    }

    let mut reader = ByteReader::new(header_bytes);
    let segment_id = reader
        .read_u64()
        .map_err(|detail| corrupted(path, detail))?;
    let schema_id = reader
        .read_u64()
        .map_err(|detail| corrupted(path, detail))?;
    let docid_lo = reader
        .read_u64()
        .map_err(|detail| corrupted(path, detail))?;
    let docid_hi = reader
        .read_u64()
        .map_err(|detail| corrupted(path, detail))?;
    let doc_count = reader
        .read_u32()
        .map_err(|detail| corrupted(path, detail))?;
    let reserved = reader
        .read_u32()
        .map_err(|detail| corrupted(path, detail))?;
    if reserved != 0 {
        return Err(corrupted(
            path,
            format!("reserved header word is {reserved}"),
        ));
    }
    let created_unix_s = reader
        .read_i64()
        .map_err(|detail| corrupted(path, detail))?;
    let engine_version = reader
        .read_u32()
        .map_err(|detail| corrupted(path, detail))?;
    let section_count = reader
        .read_u16()
        .map_err(|detail| corrupted(path, detail))?;
    let reserved2 = reader
        .read_u16()
        .map_err(|detail| corrupted(path, detail))?;
    if reserved2 != 0 {
        return Err(corrupted(
            path,
            format!("reserved section word is {reserved2}"),
        ));
    }
    let section_count_usize = usize::from(section_count);
    if section_count_usize > limits.max_sections {
        return Err(resource(
            "section table",
            format!(
                "{} declares {section_count_usize} sections, exceeding limit {}",
                path.display(),
                limits.max_sections
            ),
        ));
    }
    let expected_header_len =
        header_len(section_count_usize).map_err(|detail| corrupted(path, detail))?;
    if declared_header_len != expected_header_len {
        return Err(corrupted(
            path,
            format!(
                "non-canonical header length {declared_header_len}; expected {expected_header_len}"
            ),
        ));
    }
    let expected_schema_id = schema.schema_id()?;
    if schema_id != expected_schema_id {
        return Err(QuillError::UnknownSchema { schema_id });
    }
    validate_header_range(docid_lo, docid_hi, doc_count)
        .map_err(|detail| corrupted(path, detail))?;

    let mut sections = Vec::new();
    sections
        .try_reserve_exact(section_count_usize)
        .map_err(|_| {
            resource(
                "section table",
                format!("allocation failed for {}", path.display()),
            )
        })?;
    for _ in 0..section_count_usize {
        sections.push(SectionEntry {
            kind: SectionKind::from_raw(
                reader
                    .read_u16()
                    .map_err(|detail| corrupted(path, detail))?,
            ),
            flags: reader
                .read_u16()
                .map_err(|detail| corrupted(path, detail))?,
            offset: reader
                .read_u64()
                .map_err(|detail| corrupted(path, detail))?,
            len: reader
                .read_u64()
                .map_err(|detail| corrupted(path, detail))?,
            xxh3: reader
                .read_u64()
                .map_err(|detail| corrupted(path, detail))?,
        });
    }
    if !reader.is_empty() {
        return Err(corrupted(path, "trailing bytes in header block"));
    }
    validate_section_sequence(
        schema,
        sections.len(),
        |index| {
            let entry = sections[index];
            (entry.kind, entry.flags)
        },
        true,
    )
    .map_err(|detail| corrupted(path, detail))?;
    validate_layout(bytes, header_crc_end, trailer_start, &sections)
        .map_err(|detail| corrupted(path, detail))?;

    let file_xxh3 = read_u64_at(bytes, trailer_start)
        .ok_or_else(|| corrupted(path, "truncated file checksum"))?;
    let trailer_crc = read_u32_at(bytes, trailer_start + 8)
        .ok_or_else(|| corrupted(path, "truncated trailer CRC"))?;
    let hash_bytes = file_xxh3.to_le_bytes();
    let actual_trailer_crc = crc32(&hash_bytes);
    if trailer_crc != actual_trailer_crc {
        return Err(corrupted(
            path,
            format!(
                "trailer CRC mismatch: expected {trailer_crc:#010x}, got {actual_trailer_crc:#010x}"
            ),
        ));
    }

    Ok(ParsedContainer {
        header: SegmentHeader {
            segment_id,
            schema_id,
            docid_lo,
            docid_hi,
            doc_count,
            created_unix_s,
            engine_version,
            section_count,
        },
        sections,
        trailer_start,
        file_xxh3,
    })
}

fn validate_header_range(docid_lo: u64, docid_hi: u64, doc_count: u32) -> Result<(), String> {
    if docid_lo >= docid_hi {
        return Err(format!(
            "invalid empty or reversed docid range [{docid_lo}, {docid_hi})"
        ));
    }
    if docid_hi > MAX_DOCID_EXCLUSIVE {
        return Err(format!(
            "docid upper bound {docid_hi} exceeds the u32 payload domain {MAX_DOCID_EXCLUSIVE}"
        ));
    }
    let span = docid_hi
        .checked_sub(docid_lo)
        .ok_or_else(|| format!("invalid docid range [{docid_lo}, {docid_hi})"))?;
    if u64::from(doc_count) > span {
        return Err(format!("doc_count {doc_count} exceeds docid span {span}"));
    }
    Ok(())
}

fn validate_section_sequence<F>(
    schema: SchemaDescriptor,
    count: usize,
    mut get: F,
    allow_unregistered_extensions: bool,
) -> Result<(), String>
where
    F: FnMut(usize) -> (SectionKind, u16),
{
    let mut seen = [false; 11];
    let mut previous = None;
    for index in 0..count {
        let (kind, flags) = get(index);
        if let Some(previous) = previous {
            if kind <= previous {
                return Err(format!(
                    "section kinds are not strictly ascending at index {index}: {} after {}",
                    kind.raw(),
                    previous.raw()
                ));
            }
        }
        previous = Some(kind);
        if kind.is_known() {
            if flags != 0 {
                return Err(format!(
                    "known section {} has unsupported flags {flags:#06x}",
                    kind.raw()
                ));
            }
            seen[usize::from(kind.raw())] = true;
        } else {
            if kind.raw() <= SectionKind::STATS.raw() || flags != OPTIONAL_SKIPPABLE {
                return Err(format!(
                    "unknown section {} is not canonically optional-skippable",
                    kind.raw()
                ));
            }
            if !allow_unregistered_extensions {
                return Err(format!(
                    "writer cannot emit unregistered optional section {}",
                    kind.raw()
                ));
            }
        }
    }

    let mut expected = [false; 11];
    for kind in [
        SectionKind::TERMDICT,
        SectionKind::POSTINGS,
        SectionKind::BLOCKMAX,
        SectionKind::DOCLEN,
        SectionKind::IDMAP,
        SectionKind::IDHASH,
        SectionKind::STATS,
    ] {
        expected[usize::from(kind.raw())] = true;
    }
    expected[usize::from(SectionKind::POSITIONS.raw())] = schema_has_positions(schema);
    expected[usize::from(SectionKind::NUMERIC.raw())] = schema_has_numeric_columns(schema);
    expected[usize::from(SectionKind::STOREDMETA.raw())] = schema_has_stored_fields(schema);
    for raw in usize::from(SectionKind::TERMDICT.raw())..=usize::from(SectionKind::STATS.raw()) {
        if seen[raw] != expected[raw] {
            let requirement = if expected[raw] {
                "missing"
            } else {
                "unexpected"
            };
            return Err(format!("{requirement} known section kind {raw}"));
        }
    }
    Ok(())
}

fn published_segment_id(path: &Path) -> Result<u64, String> {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "published segment name is not valid UTF-8".to_owned())?;
    let Some(hex) = name
        .strip_prefix("seg-")
        .and_then(|rest| rest.strip_suffix(".fslx"))
    else {
        return Err("published segment name must match seg-<hex16>.fslx".to_owned());
    };
    if hex.len() != 16
        || !hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err("published segment name must match seg-<hex16>.fslx".to_owned());
    }
    u64::from_str_radix(hex, 16).map_err(|_| "published segment id is not valid hex".to_owned())
}

fn schema_has_positions(schema: SchemaDescriptor) -> bool {
    schema.fields.iter().any(|field| {
        matches!(
            field.kind,
            FieldKind::Text {
                positions: true,
                ..
            }
        )
    })
}

fn schema_has_numeric_columns(schema: SchemaDescriptor) -> bool {
    schema
        .fields
        .iter()
        .any(|field| field.kind.has_numeric_column())
}

fn schema_has_stored_fields(schema: SchemaDescriptor) -> bool {
    schema.fields.iter().any(|field| field.stored)
}

fn validate_layout(
    bytes: &[u8],
    header_crc_end: usize,
    trailer_start: usize,
    sections: &[SectionEntry],
) -> Result<(), String> {
    let mut cursor = header_crc_end;
    for entry in sections {
        let expected_offset = align_up(cursor, FSLX_SECTION_ALIGNMENT)
            .ok_or_else(|| "section alignment overflow".to_owned())?;
        let offset = usize::try_from(entry.offset)
            .map_err(|_| "section offset does not fit usize".to_owned())?;
        if offset != expected_offset {
            return Err(format!(
                "section {} has non-canonical offset {offset}; expected {expected_offset}",
                entry.kind.raw()
            ));
        }
        if offset % FSLX_SECTION_ALIGNMENT != 0 {
            return Err(format!(
                "section {} is not 64-byte aligned",
                entry.kind.raw()
            ));
        }
        let padding = bytes
            .get(cursor..offset)
            .ok_or_else(|| "alignment padding lies outside the file".to_owned())?;
        if let Some(relative) = padding.iter().position(|byte| *byte != 0) {
            return Err(format!(
                "non-zero alignment padding at byte {}",
                cursor + relative
            ));
        }
        let len = usize::try_from(entry.len)
            .map_err(|_| "section length does not fit usize".to_owned())?;
        let end = offset
            .checked_add(len)
            .ok_or_else(|| "section end overflow".to_owned())?;
        if end > trailer_start {
            return Err(format!(
                "section {} ends at {end}, beyond trailer start {trailer_start}",
                entry.kind.raw()
            ));
        }
        cursor = end;
    }
    if cursor != trailer_start {
        return Err(format!(
            "trailing bytes before file trailer: sections end at {cursor}, trailer starts at {trailer_start}"
        ));
    }
    Ok(())
}

fn header_len(section_count: usize) -> Result<usize, String> {
    section_count
        .checked_mul(SECTION_ENTRY_LEN)
        .and_then(|table| FIXED_HEADER_LEN.checked_add(table))
        .ok_or_else(|| "header length overflow".to_owned())
}

fn align_up(value: usize, alignment: usize) -> Option<usize> {
    value
        .checked_add(alignment - 1)
        .map(|sum| sum & !(alignment - 1))
}

fn write_header_fixed(output: &mut Vec<u8>, header: SegmentHeader) {
    output.extend_from_slice(&header.segment_id.to_le_bytes());
    output.extend_from_slice(&header.schema_id.to_le_bytes());
    output.extend_from_slice(&header.docid_lo.to_le_bytes());
    output.extend_from_slice(&header.docid_hi.to_le_bytes());
    output.extend_from_slice(&header.doc_count.to_le_bytes());
    output.extend_from_slice(&0_u32.to_le_bytes());
    output.extend_from_slice(&header.created_unix_s.to_le_bytes());
    output.extend_from_slice(&header.engine_version.to_le_bytes());
    output.extend_from_slice(&header.section_count.to_le_bytes());
    output.extend_from_slice(&0_u16.to_le_bytes());
}

fn write_section_entry(output: &mut Vec<u8>, entry: SectionEntry) {
    output.extend_from_slice(&entry.kind.raw().to_le_bytes());
    output.extend_from_slice(&entry.flags.to_le_bytes());
    output.extend_from_slice(&entry.offset.to_le_bytes());
    output.extend_from_slice(&entry.len.to_le_bytes());
    output.extend_from_slice(&entry.xxh3.to_le_bytes());
}

fn section_bytes(bytes: &[u8], entry: SectionEntry) -> Option<&[u8]> {
    let start = usize::try_from(entry.offset).ok()?;
    let len = usize::try_from(entry.len).ok()?;
    bytes.get(start..start.checked_add(len)?)
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(bytes);
    hasher.finalize()
}

fn invalid_segment(detail: impl Into<String>) -> QuillError {
    QuillError::Invariant {
        detail: format!("invalid FSLX segment input: {}", detail.into()),
    }
}

fn resource(resource: &'static str, detail: impl Into<String>) -> QuillError {
    QuillError::Resource {
        resource,
        detail: detail.into(),
    }
}

fn corrupted(path: impl AsRef<Path>, detail: impl Into<String>) -> QuillError {
    QuillError::IndexCorrupted {
        path: path.as_ref().to_path_buf(),
        detail: detail.into(),
    }
}

fn read_u32_at(bytes: &[u8], offset: usize) -> Option<u32> {
    let slice = bytes.get(offset..offset.checked_add(4)?)?;
    Some(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn read_u64_at(bytes: &[u8], offset: usize) -> Option<u64> {
    let slice = bytes.get(offset..offset.checked_add(8)?)?;
    Some(u64::from_le_bytes([
        slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
    ]))
}

struct ByteReader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> ByteReader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], String> {
        let end = self
            .position
            .checked_add(len)
            .ok_or_else(|| "header cursor overflow".to_owned())?;
        let result = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| format!("truncated header at byte {}", self.position))?;
        self.position = end;
        Ok(result)
    }

    fn read_u16(&mut self) -> Result<u16, String> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_u32(&mut self) -> Result<u32, String> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_u64(&mut self) -> Result<u64, String> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_i64(&mut self) -> Result<i64, String> {
        let bytes = self.take(8)?;
        Ok(i64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn is_empty(&self) -> bool {
        self.position == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use frankensearch_core::SearchError;

    use crate::schema::{Analyzer, DEFAULT_SCHEMA, FieldDescriptor};

    use super::*;

    type TestResult = Result<(), Box<dyn Error>>;

    fn init_replay_tracing() {
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .without_time()
            .with_test_writer()
            .with_max_level(tracing::Level::INFO)
            .finish();
        let _ = tracing::subscriber::set_global_default(subscriber);
    }

    const MINIMAL_FIELDS: [FieldDescriptor; 1] = [FieldDescriptor {
        id: 0,
        name: "term",
        kind: FieldKind::Keyword,
        stored: false,
    }];
    const MINIMAL_SCHEMA: SchemaDescriptor = SchemaDescriptor {
        name: "segment-test-minimal",
        fields: &MINIMAL_FIELDS,
    };
    const CONDITIONAL_FIELDS: [FieldDescriptor; 2] = [
        FieldDescriptor {
            id: 0,
            name: "body",
            kind: FieldKind::Text {
                analyzer: Analyzer::FrankensearchDefault,
                positions: true,
            },
            stored: true,
        },
        FieldDescriptor {
            id: 1,
            name: "rank",
            kind: FieldKind::I64 {
                indexed: true,
                fast: false,
            },
            stored: false,
        },
    ];
    const CONDITIONAL_SCHEMA: SchemaDescriptor = SchemaDescriptor {
        name: "segment-test-all-conditionals",
        fields: &CONDITIONAL_FIELDS,
    };
    const U64_FIELDS: [FieldDescriptor; 1] = [FieldDescriptor {
        id: 0,
        name: "unsigned",
        kind: FieldKind::U64 {
            indexed: true,
            fast: false,
        },
        stored: false,
    }];
    const U64_SCHEMA: SchemaDescriptor = SchemaDescriptor {
        name: "segment-test-indexed-u64",
        fields: &U64_FIELDS,
    };
    const PINNED_DEFAULT_ENTRIES: [SectionEntry; 10] = [
        SectionEntry {
            kind: SectionKind::TERMDICT,
            flags: 0,
            offset: 384,
            len: 4,
            xxh3: 0xb311_f463_fe75_7daa,
        },
        SectionEntry {
            kind: SectionKind::POSTINGS,
            flags: 0,
            offset: 448,
            len: 5,
            xxh3: 0x226e_c10f_2d51_8a15,
        },
        SectionEntry {
            kind: SectionKind::POSITIONS,
            flags: 0,
            offset: 512,
            len: 6,
            xxh3: 0x8464_7628_6cec_0ef4,
        },
        SectionEntry {
            kind: SectionKind::BLOCKMAX,
            flags: 0,
            offset: 576,
            len: 7,
            xxh3: 0x5ae0_ae5f_0af5_abf3,
        },
        SectionEntry {
            kind: SectionKind::DOCLEN,
            flags: 0,
            offset: 640,
            len: 8,
            xxh3: 0x4bd4_2ac3_c7ea_e3b1,
        },
        SectionEntry {
            kind: SectionKind::IDMAP,
            flags: 0,
            offset: 704,
            len: 9,
            xxh3: 0x38cc_9c70_1a48_1084,
        },
        SectionEntry {
            kind: SectionKind::IDHASH,
            flags: 0,
            offset: 768,
            len: 10,
            xxh3: 0x2518_0e76_4f5a_2769,
        },
        SectionEntry {
            kind: SectionKind::NUMERIC,
            flags: 0,
            offset: 832,
            len: 11,
            xxh3: 0xf32b_8ca9_4e2a_52bc,
        },
        SectionEntry {
            kind: SectionKind::STOREDMETA,
            flags: 0,
            offset: 896,
            len: 12,
            xxh3: 0xffff_1c57_4139_b461,
        },
        SectionEntry {
            kind: SectionKind::STATS,
            flags: 0,
            offset: 960,
            len: 13,
            xxh3: 0xd325_c744_07be_efd7,
        },
    ];
    const PINNED_DEFAULT_PAYLOADS: [&[u8]; 10] = [
        &[0x01, 0x12, 0x23, 0x34],
        &[0x02, 0x13, 0x24, 0x35, 0x46],
        &[0x03, 0x14, 0x25, 0x36, 0x47, 0x58],
        &[0x04, 0x15, 0x26, 0x37, 0x48, 0x59, 0x6a],
        &[0x05, 0x16, 0x27, 0x38, 0x49, 0x5a, 0x6b, 0x7c],
        &[0x06, 0x17, 0x28, 0x39, 0x4a, 0x5b, 0x6c, 0x7d, 0x8e],
        &[0x07, 0x18, 0x29, 0x3a, 0x4b, 0x5c, 0x6d, 0x7e, 0x8f, 0xa0],
        &[
            0x08, 0x19, 0x2a, 0x3b, 0x4c, 0x5d, 0x6e, 0x7f, 0x90, 0xa1, 0xb2,
        ],
        &[
            0x09, 0x1a, 0x2b, 0x3c, 0x4d, 0x5e, 0x6f, 0x80, 0x91, 0xa2, 0xb3, 0xc4,
        ],
        &[
            0x0a, 0x1b, 0x2c, 0x3d, 0x4e, 0x5f, 0x70, 0x81, 0x92, 0xa3, 0xb4, 0xc5, 0xd6,
        ],
    ];

    #[derive(Clone, Debug)]
    struct OwnedSection {
        kind: SectionKind,
        flags: u16,
        bytes: Vec<u8>,
    }

    fn fixture_header(schema: SchemaDescriptor) -> SegmentHeaderInput {
        SegmentHeaderInput {
            segment_id: 0x0123_4567_89ab_cdef,
            schema,
            docid_lo: 100,
            docid_hi: 107,
            doc_count: 5,
            created_unix_s: 1_726_000_123,
            engine_version: 0x0002_0001,
        }
    }

    fn fixture_sections(schema: SchemaDescriptor, large: bool) -> Vec<OwnedSection> {
        let mut result = Vec::new();
        for raw in SectionKind::TERMDICT.raw()..=SectionKind::STATS.raw() {
            let kind = SectionKind::from_raw(raw);
            let present = match kind {
                SectionKind::POSITIONS => schema_has_positions(schema),
                SectionKind::NUMERIC => schema_has_numeric_columns(schema),
                SectionKind::STOREDMETA => schema_has_stored_fields(schema),
                _ => true,
            };
            if present {
                let len = if large && kind == SectionKind::POSTINGS {
                    5_137
                } else {
                    usize::from(raw) + 3
                };
                let raw_byte = u8::try_from(raw).unwrap_or(0);
                let bytes = (0..len)
                    .map(|index| {
                        let low = u8::try_from(index & 0xff).unwrap_or(0);
                        low.wrapping_mul(17).wrapping_add(raw_byte)
                    })
                    .collect();
                result.push(OwnedSection {
                    kind,
                    flags: 0,
                    bytes,
                });
            }
        }
        result
    }

    fn encode_owned(
        header: SegmentHeaderInput,
        sections: &[OwnedSection],
    ) -> Result<EncodedSegment, QuillError> {
        let borrowed: Vec<_> = sections
            .iter()
            .map(|section| SectionInput {
                kind: section.kind,
                flags: section.flags,
                bytes: &section.bytes,
            })
            .collect();
        EncodedSegment::encode(header, &borrowed)
    }

    fn assemble_owned(
        header: SegmentHeaderInput,
        sections: &[OwnedSection],
    ) -> Result<EncodedSegment, QuillError> {
        let planned = sections
            .iter()
            .map(|section| PlannedSection::new(section.kind, section.bytes.len()))
            .collect::<Vec<_>>();
        let mut assembler = SegmentAssembler::new(header, &planned)?;
        for section in sections {
            assembler.copy_section(section.kind, &section.bytes)?;
        }
        assembler.finish()
    }

    fn encode_owned_with_unregistered_extensions(
        header: SegmentHeaderInput,
        sections: &[OwnedSection],
    ) -> Result<EncodedSegment, QuillError> {
        let borrowed: Vec<_> = sections
            .iter()
            .map(|section| SectionInput {
                kind: section.kind,
                flags: section.flags,
                bytes: &section.bytes,
            })
            .collect();
        EncodedSegment::encode_with_limits_impl(header, &borrowed, SegmentLimits::default(), true)
    }

    #[test]
    fn planned_assembler_is_byte_identical_to_canonical_writer() -> TestResult {
        for &(schema, large) in &[
            (MINIMAL_SCHEMA, false),
            (MINIMAL_SCHEMA, true),
            (DEFAULT_SCHEMA, false),
            (DEFAULT_SCHEMA, true),
        ] {
            let sections = fixture_sections(schema, large);
            let encoded = encode_owned(fixture_header(schema), &sections)?;
            let assembled = assemble_owned(fixture_header(schema), &sections)?;
            assert_eq!(assembled, encoded);
            SegmentReader::from_bytes(assembled.as_bytes(), schema)?.verify()?;
        }
        Ok(())
    }

    #[test]
    fn planned_assembler_rejects_out_of_order_and_inexact_writes() -> TestResult {
        let sections = fixture_sections(MINIMAL_SCHEMA, false);
        let planned = sections
            .iter()
            .map(|section| PlannedSection::new(section.kind, section.bytes.len()))
            .collect::<Vec<_>>();
        let first = &sections[0];
        assert!(!first.bytes.is_empty());

        let mut wrong_kind = SegmentAssembler::new(fixture_header(MINIMAL_SCHEMA), &planned)?;
        let error = wrong_kind
            .copy_section(sections[1].kind, &sections[1].bytes)
            .expect_err("out-of-order section must fail");
        assert!(error.to_string().contains("expected"));

        let mut short = SegmentAssembler::new(fixture_header(MINIMAL_SCHEMA), &planned)?;
        let error = short
            .copy_section(first.kind, &first.bytes[..first.bytes.len() - 1])
            .expect_err("short section must fail");
        assert!(error.to_string().contains("writer ended"));

        let mut long = SegmentAssembler::new(fixture_header(MINIMAL_SCHEMA), &planned)?;
        let error = long
            .write_section(first.kind, |bytes| {
                bytes.extend_from_slice(&first.bytes);
                bytes.push(0);
            })
            .expect_err("long section must fail");
        assert!(error.to_string().contains("writer ended"));

        let incomplete = SegmentAssembler::new(fixture_header(MINIMAL_SCHEMA), &planned)?;
        let error = incomplete
            .finish()
            .expect_err("missing planned sections must fail");
        assert!(error.to_string().contains("was not written"));
        Ok(())
    }

    /// Load the committed v1 fixture without invoking the production writer or
    /// any checksum helper. The textual hex form keeps the durable bytes easy
    /// to review while still freezing every byte across future format bumps.
    fn pinned_v1_fixture() -> Result<Vec<u8>, Box<dyn Error>> {
        let hex: String = include_str!("../tests/fixtures/fslx-v1-golden.hex")
            .chars()
            .filter(|character| !character.is_ascii_whitespace())
            .collect();
        if !hex.len().is_multiple_of(2) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "pinned FSLX v1 fixture has an odd number of hex digits",
            )
            .into());
        }

        let mut bytes = Vec::with_capacity(hex.len() / 2);
        for pair in hex.as_bytes().as_chunks::<2>().0 {
            bytes.push(u8::from_str_radix(std::str::from_utf8(pair)?, 16)?);
        }
        Ok(bytes)
    }

    #[test]
    fn reader_accepts_committed_pinned_v1_fixture() -> TestResult {
        let bytes = pinned_v1_fixture()?;
        let reader = SegmentReader::from_bytes(&bytes, DEFAULT_SCHEMA)?;

        assert_eq!(
            reader.header(),
            SegmentHeader {
                segment_id: 0x0123_4567_89ab_cdef,
                schema_id: 0xa312_ebf6_d136_07a5,
                docid_lo: 100,
                docid_hi: 107,
                doc_count: 5,
                created_unix_s: 1_726_000_123,
                engine_version: 0x0002_0001,
                section_count: 10,
            }
        );
        assert_eq!(reader.section_entries(), &PINNED_DEFAULT_ENTRIES);
        assert_eq!(reader.file_len(), 985);
        assert_eq!(reader.file_xxh3(), 0x3d1c_8570_73b5_390f);
        for (entry, payload) in PINNED_DEFAULT_ENTRIES.iter().zip(PINNED_DEFAULT_PAYLOADS) {
            assert_eq!(
                reader.section(entry.kind)?.expect("pinned fixture section"),
                payload
            );
        }
        reader.verify()?;
        Ok(())
    }

    #[test]
    fn pinned_wire_oracle_and_roundtrip_preserve_header_table_and_payloads() -> TestResult {
        let owned = fixture_sections(DEFAULT_SCHEMA, false);
        let first = encode_owned(fixture_header(DEFAULT_SCHEMA), &owned)?;
        let second = encode_owned(fixture_header(DEFAULT_SCHEMA), &owned)?;
        let header_len = usize::try_from(read_u32_at(first.as_bytes(), 12).expect("header len"))?;
        let header_crc =
            read_u32_at(first.as_bytes(), FILE_PREFIX_LEN + header_len).expect("header crc");
        let trailer_start = first.as_bytes().len() - TRAILER_LEN;
        let trailer_crc = read_u32_at(first.as_bytes(), trailer_start + 8).expect("trailer crc");
        assert_eq!(
            &first.as_bytes()[..FILE_PREFIX_LEN],
            b"FSLXSEG\0\x01\0\0\0\x50\x01\0\0"
        );
        assert_eq!(first.file_len(), 985);
        assert_eq!(header_len, 336);
        assert_eq!(first.header().schema_id, 0xa312_ebf6_d136_07a5);
        assert_eq!(header_crc, 0x3e9c_358f);
        assert_eq!(first.file_xxh3(), 0x3d1c_8570_73b5_390f);
        assert_eq!(trailer_crc, 0x4752_5d26);
        assert_eq!(first.section_entries(), &PINNED_DEFAULT_ENTRIES);
        assert_eq!(first.as_bytes(), pinned_v1_fixture()?);
        assert_eq!(first.as_bytes(), second.as_bytes());
        assert_eq!(first.file_xxh3(), second.file_xxh3());
        assert_eq!(first.source_xxh3(), xxh3_64(first.as_bytes()));
        assert_eq!(first.source_xxh3(), second.source_xxh3());

        let reader = SegmentReader::from_owned(first.as_bytes().to_vec(), DEFAULT_SCHEMA)?;
        assert_eq!(reader.header(), first.header());
        assert_eq!(reader.section_entries(), first.section_entries());
        assert_eq!(reader.file_len(), first.file_len());
        assert_eq!(reader.file_xxh3(), first.file_xxh3());
        for expected in &owned {
            assert_eq!(
                reader.section(expected.kind)?.expect("fixture section"),
                expected.bytes
            );
        }
        reader.verify()?;

        let borrowed = SegmentReader::from_bytes(first.as_bytes(), DEFAULT_SCHEMA)?;
        assert_eq!(borrowed.header(), reader.header());
        borrowed.verify()?;
        Ok(())
    }

    #[test]
    fn every_segment_byte_class_flip_is_public_typed_and_never_panics() -> TestResult {
        const SEED: u64 = 0x6d5a_56da_1b2c_3e4f;

        init_replay_tracing();
        let owned = fixture_sections(CONDITIONAL_SCHEMA, false);
        let encoded = encode_owned(fixture_header(CONDITIONAL_SCHEMA), &owned)?;
        let entries = encoded.section_entries().to_vec();
        assert_eq!(entries.len(), 10, "fixture must cover every v1 section");
        let golden = encoded.into_bytes();
        let header_len =
            usize::try_from(read_u32_at(&golden, 12).ok_or("corruption-sweep header length")?)?;
        let header_crc_offset = FILE_PREFIX_LEN + header_len;
        let trailer_start = golden.len() - TRAILER_LEN;
        let mut cases = vec![
            ("magic", 0),
            ("format version", 8),
            ("header length", 12),
            ("fixed header", FILE_PREFIX_LEN + 8),
            ("section table", table_entry_offset(0)),
            ("header CRC", header_crc_offset),
            ("alignment padding", header_crc_offset + HEADER_CRC_LEN),
            ("file checksum", trailer_start),
            ("trailer CRC", trailer_start + 8),
        ];
        for entry in entries {
            cases.push(("section payload", usize::try_from(entry.offset)?));
        }

        for (region, offset) in cases.iter().copied() {
            tracing::info!(
                codec = "FSLX_CONTAINER",
                seed = SEED,
                case = offset,
                region,
                replay = "every_segment_byte_class_flip_is_public_typed_and_never_panics",
                "starting byte-class corruption case"
            );
            let offset_u64 = u64::try_from(offset)?;
            let bit = u32::try_from((SEED ^ offset_u64) & 7)?;
            let mut mutated = golden.clone();
            mutated[offset] ^= 1_u8 << bit;

            let result = catch_unwind(AssertUnwindSafe(|| {
                SegmentReader::from_owned(mutated, CONDITIONAL_SCHEMA)
                    .and_then(|reader| reader.verify())
            }));
            assert!(
                result.is_ok(),
                "seed={SEED:#x} offset={offset} region={region}: parser panicked"
            );
            let Err(codec_error) = result.expect("byte-class parser result") else {
                panic!(
                    "seed={SEED:#x} offset={offset} region={region}: byte-class mutation was accepted"
                );
            };
            let public_error: SearchError = codec_error.into();
            assert!(
                matches!(public_error, SearchError::IndexCorrupted { .. }),
                "seed={SEED:#x} offset={offset} region={region}: mutation was not public typed corruption"
            );
        }

        tracing::info!(
            seed = SEED,
            mutations = cases.len(),
            "FSLX v1 byte-class corruption sweep passed"
        );
        Ok(())
    }

    #[test]
    fn schema_conditionals_require_exact_known_section_sets() -> TestResult {
        for schema in [
            MINIMAL_SCHEMA,
            DEFAULT_SCHEMA,
            CONDITIONAL_SCHEMA,
            U64_SCHEMA,
        ] {
            let owned = fixture_sections(schema, false);
            let encoded = encode_owned(fixture_header(schema), &owned)?;
            let reader = SegmentReader::from_owned(encoded.into_bytes(), schema)?;
            assert_eq!(
                reader.section(SectionKind::POSITIONS)?.is_some(),
                schema_has_positions(schema)
            );
            assert_eq!(
                reader.section(SectionKind::NUMERIC)?.is_some(),
                schema_has_numeric_columns(schema)
            );
            assert_eq!(
                reader.section(SectionKind::STOREDMETA)?.is_some(),
                schema_has_stored_fields(schema)
            );
            reader.verify()?;
        }

        assert!(schema_has_numeric_columns(U64_SCHEMA));
        let mut missing_u64_numeric = fixture_sections(U64_SCHEMA, false);
        assert!(
            missing_u64_numeric
                .iter()
                .any(|section| section.kind == SectionKind::NUMERIC)
        );
        missing_u64_numeric.retain(|section| section.kind != SectionKind::NUMERIC);
        assert!(matches!(
            encode_owned(fixture_header(U64_SCHEMA), &missing_u64_numeric),
            Err(QuillError::Invariant { .. })
        ));

        for conditional in [
            SectionKind::POSITIONS,
            SectionKind::NUMERIC,
            SectionKind::STOREDMETA,
        ] {
            let mut missing = fixture_sections(CONDITIONAL_SCHEMA, false);
            missing.retain(|section| section.kind != conditional);
            assert!(matches!(
                encode_owned(fixture_header(CONDITIONAL_SCHEMA), &missing),
                Err(QuillError::Invariant { .. })
            ));

            let mut unexpected = fixture_sections(MINIMAL_SCHEMA, false);
            unexpected.push(OwnedSection {
                kind: conditional,
                flags: 0,
                bytes: vec![1],
            });
            unexpected.sort_unstable_by_key(|section| section.kind);
            assert!(matches!(
                encode_owned(fixture_header(MINIMAL_SCHEMA), &unexpected),
                Err(QuillError::Invariant { .. })
            ));
        }

        let encoded = encode_owned(
            fixture_header(MINIMAL_SCHEMA),
            &fixture_sections(MINIMAL_SCHEMA, false),
        )?;
        let stats_index = encoded
            .section_entries()
            .iter()
            .position(|entry| entry.kind == SectionKind::STATS)
            .expect("STATS entry");
        let mut unexpected_numeric = encoded.into_bytes();
        let kind_offset = table_entry_offset(stats_index);
        unexpected_numeric[kind_offset..kind_offset + 2]
            .copy_from_slice(&SectionKind::NUMERIC.raw().to_le_bytes());
        rewrite_header_crc(&mut unexpected_numeric);
        assert!(matches!(
            SegmentReader::from_owned(unexpected_numeric, MINIMAL_SCHEMA),
            Err(QuillError::IndexCorrupted { .. })
        ));
        Ok(())
    }

    #[test]
    fn section_layout_is_canonical_aligned_and_zero_padded() -> TestResult {
        let owned = fixture_sections(CONDITIONAL_SCHEMA, false);
        let encoded = encode_owned(fixture_header(CONDITIONAL_SCHEMA), &owned)?;
        let bytes = encoded.as_bytes();
        let declared_header_len = usize::try_from(read_u32_at(bytes, 12).expect("header len"))?;
        let mut cursor = FILE_PREFIX_LEN + declared_header_len + HEADER_CRC_LEN;
        for entry in encoded.section_entries() {
            let expected = align_up(cursor, FSLX_SECTION_ALIGNMENT).expect("alignment");
            let offset = usize::try_from(entry.offset)?;
            assert_eq!(offset, expected);
            assert_eq!(offset % FSLX_SECTION_ALIGNMENT, 0);
            assert!(bytes[cursor..offset].iter().all(|byte| *byte == 0));
            cursor = offset + usize::try_from(entry.len)?;
        }
        assert_eq!(cursor, bytes.len() - TRAILER_LEN);
        Ok(())
    }

    fn rewrite_header_crc(bytes: &mut [u8]) {
        let declared = usize::try_from(read_u32_at(bytes, 12).expect("header length"))
            .expect("usize header length");
        let header_end = FILE_PREFIX_LEN + declared;
        let checksum = crc32(&bytes[FILE_PREFIX_LEN..header_end]);
        bytes[header_end..header_end + HEADER_CRC_LEN].copy_from_slice(&checksum.to_le_bytes());
    }

    fn table_entry_offset(index: usize) -> usize {
        FILE_PREFIX_LEN + FIXED_HEADER_LEN + index * SECTION_ENTRY_LEN
    }

    #[test]
    fn reader_skips_unknown_optional_sections_but_writer_requires_registry_entry() -> TestResult {
        let mut owned = fixture_sections(MINIMAL_SCHEMA, false);
        owned.push(OwnedSection {
            kind: SectionKind::from_raw(11),
            flags: OPTIONAL_SKIPPABLE,
            bytes: b"future optional payload".to_vec(),
        });
        assert!(matches!(
            encode_owned(fixture_header(MINIMAL_SCHEMA), &owned),
            Err(QuillError::Invariant { .. })
        ));

        let encoded =
            encode_owned_with_unregistered_extensions(fixture_header(MINIMAL_SCHEMA), &owned)?;
        let reader = SegmentReader::from_owned(encoded.as_bytes().to_vec(), MINIMAL_SCHEMA)?;
        assert!(reader.section(SectionKind::from_raw(11))?.is_none());
        assert!(
            reader
                .section_entries()
                .iter()
                .any(|entry| entry.kind == SectionKind::from_raw(11))
        );
        reader.verify()?;

        let unknown = encoded
            .section_entries()
            .last()
            .copied()
            .expect("unknown section entry");
        let mut corrupt_unknown = encoded.as_bytes().to_vec();
        corrupt_unknown[usize::try_from(unknown.offset)?] ^= 0x20;
        let reader = SegmentReader::from_owned(corrupt_unknown, MINIMAL_SCHEMA)?;
        assert!(reader.section(SectionKind::from_raw(11))?.is_none());
        assert!(matches!(
            reader.verify(),
            Err(QuillError::IndexCorrupted { .. })
        ));

        let mut caller_required = owned.clone();
        caller_required.last_mut().expect("unknown section").flags = 0;
        assert!(matches!(
            encode_owned(fixture_header(MINIMAL_SCHEMA), &caller_required),
            Err(QuillError::Invariant { .. })
        ));

        let mut durable_required = encoded.into_bytes();
        let last_index = owned.len() - 1;
        let flags_offset = table_entry_offset(last_index) + 2;
        durable_required[flags_offset..flags_offset + 2].copy_from_slice(&0_u16.to_le_bytes());
        rewrite_header_crc(&mut durable_required);
        assert!(matches!(
            SegmentReader::from_owned(durable_required, MINIMAL_SCHEMA),
            Err(QuillError::IndexCorrupted { .. })
        ));
        Ok(())
    }

    #[test]
    fn writer_rejects_bad_flags_order_ranges_and_budgets() -> TestResult {
        let mut bad_flags = fixture_sections(MINIMAL_SCHEMA, false);
        bad_flags[0].flags = OPTIONAL_SKIPPABLE;
        assert!(matches!(
            encode_owned(fixture_header(MINIMAL_SCHEMA), &bad_flags),
            Err(QuillError::Invariant { .. })
        ));

        let mut descending = fixture_sections(MINIMAL_SCHEMA, false);
        descending.swap(0, 1);
        assert!(matches!(
            encode_owned(fixture_header(MINIMAL_SCHEMA), &descending),
            Err(QuillError::Invariant { .. })
        ));

        let mut reversed = fixture_header(MINIMAL_SCHEMA);
        reversed.docid_lo = 108;
        assert!(matches!(
            encode_owned(reversed, &fixture_sections(MINIMAL_SCHEMA, false)),
            Err(QuillError::Invariant { .. })
        ));
        let mut empty = fixture_header(MINIMAL_SCHEMA);
        empty.docid_hi = empty.docid_lo;
        assert!(matches!(
            encode_owned(empty, &fixture_sections(MINIMAL_SCHEMA, false)),
            Err(QuillError::Invariant { .. })
        ));
        let mut wider_than_u32 = fixture_header(MINIMAL_SCHEMA);
        wider_than_u32.docid_lo = MAX_DOCID_EXCLUSIVE - 1;
        wider_than_u32.docid_hi = MAX_DOCID_EXCLUSIVE + 1;
        wider_than_u32.doc_count = 1;
        assert!(matches!(
            encode_owned(wider_than_u32, &fixture_sections(MINIMAL_SCHEMA, false)),
            Err(QuillError::Invariant { .. })
        ));
        let mut too_many_docs = fixture_header(MINIMAL_SCHEMA);
        too_many_docs.doc_count = 8;
        assert!(matches!(
            encode_owned(too_many_docs, &fixture_sections(MINIMAL_SCHEMA, false)),
            Err(QuillError::Invariant { .. })
        ));
        let mut u32_boundary = fixture_header(MINIMAL_SCHEMA);
        u32_boundary.docid_lo = MAX_DOCID_EXCLUSIVE - 1;
        u32_boundary.docid_hi = MAX_DOCID_EXCLUSIVE;
        u32_boundary.doc_count = 1;
        encode_owned(u32_boundary, &fixture_sections(MINIMAL_SCHEMA, false))?;

        let owned = fixture_sections(MINIMAL_SCHEMA, false);
        let borrowed: Vec<_> = owned
            .iter()
            .map(|section| SectionInput::new(section.kind, &section.bytes))
            .collect();
        assert!(matches!(
            EncodedSegment::encode_with_limits(
                fixture_header(MINIMAL_SCHEMA),
                &borrowed,
                SegmentLimits {
                    max_sections: owned.len() - 1,
                    ..SegmentLimits::default()
                }
            ),
            Err(QuillError::Resource { .. })
        ));
        assert!(matches!(
            EncodedSegment::encode_with_limits(
                fixture_header(MINIMAL_SCHEMA),
                &borrowed,
                SegmentLimits {
                    max_file_bytes: 1,
                    ..SegmentLimits::default()
                }
            ),
            Err(QuillError::Resource { .. })
        ));

        let encoded = EncodedSegment::encode(fixture_header(MINIMAL_SCHEMA), &borrowed)?;
        assert!(matches!(
            SegmentReader::from_owned_with_limits(
                encoded.into_bytes(),
                MINIMAL_SCHEMA,
                SegmentLimits {
                    max_sections: owned.len() - 1,
                    ..SegmentLimits::default()
                }
            ),
            Err(QuillError::Resource { .. })
        ));
        Ok(())
    }

    #[test]
    fn section_checksums_are_lazy_and_cached_per_section() -> TestResult {
        let owned = fixture_sections(MINIMAL_SCHEMA, false);
        let encoded = encode_owned(fixture_header(MINIMAL_SCHEMA), &owned)?;
        let postings = encoded
            .section_entries()
            .iter()
            .find(|entry| entry.kind == SectionKind::POSTINGS)
            .copied()
            .expect("POSTINGS entry");
        let mut corrupt_bytes = encoded.into_bytes();
        let postings_offset = usize::try_from(postings.offset)?;
        corrupt_bytes[postings_offset] ^= 0x80;

        let reader = SegmentReader::from_owned(corrupt_bytes, MINIMAL_SCHEMA)?;
        assert!(reader.section(SectionKind::TERMDICT)?.is_some());
        for _ in 0..2 {
            assert!(matches!(
                reader.section(SectionKind::POSTINGS),
                Err(QuillError::IndexCorrupted { .. })
            ));
        }
        assert!(matches!(
            reader.verify(),
            Err(QuillError::IndexCorrupted { .. })
        ));
        Ok(())
    }

    #[test]
    fn position_free_paths_do_not_touch_a_corrupt_positions_section() -> TestResult {
        assert!(matches!(DEFAULT_SCHEMA.fields[0].kind, FieldKind::Keyword));
        assert!(schema_has_positions(DEFAULT_SCHEMA));
        let owned = fixture_sections(DEFAULT_SCHEMA, false);
        let encoded = encode_owned(fixture_header(DEFAULT_SCHEMA), &owned)?;
        let positions = encoded
            .section_entries()
            .iter()
            .find(|entry| entry.kind == SectionKind::POSITIONS)
            .copied()
            .expect("default-schema POSITIONS entry");
        let mut corrupt_bytes = encoded.into_bytes();
        let positions_offset = usize::try_from(positions.offset)?;
        corrupt_bytes[positions_offset] ^= 0x80;

        // Opening and the sections needed by an ordinary keyword term path
        // remain independent of POSITIONS first-touch validation.
        let reader = SegmentReader::from_owned(corrupt_bytes, DEFAULT_SCHEMA)?;
        for kind in [
            SectionKind::TERMDICT,
            SectionKind::POSTINGS,
            SectionKind::BLOCKMAX,
            SectionKind::DOCLEN,
        ] {
            assert!(reader.section(kind)?.is_some(), "kind={}", kind.raw());
        }
        assert!(matches!(
            reader.section(SectionKind::POSITIONS),
            Err(QuillError::IndexCorrupted { .. })
        ));
        Ok(())
    }

    #[test]
    fn metadata_free_paths_do_not_touch_a_corrupt_stored_meta_section() -> TestResult {
        assert!(schema_has_stored_fields(DEFAULT_SCHEMA));
        let owned = fixture_sections(DEFAULT_SCHEMA, false);
        let encoded = encode_owned(fixture_header(DEFAULT_SCHEMA), &owned)?;
        let stored_meta = encoded
            .section_entries()
            .iter()
            .find(|entry| entry.kind == SectionKind::STOREDMETA)
            .copied()
            .expect("default-schema STOREDMETA entry");
        let mut corrupt_bytes = encoded.into_bytes();
        let stored_meta_offset = usize::try_from(stored_meta.offset)?;
        corrupt_bytes[stored_meta_offset] ^= 0x20;

        // Structural open and score-only sections stay independent of lazy
        // stored-field checksum validation.
        let reader = SegmentReader::from_owned(corrupt_bytes, DEFAULT_SCHEMA)?;
        for kind in [
            SectionKind::TERMDICT,
            SectionKind::POSTINGS,
            SectionKind::BLOCKMAX,
            SectionKind::DOCLEN,
            SectionKind::STATS,
        ] {
            assert!(reader.section(kind)?.is_some(), "kind={}", kind.raw());
        }
        assert!(matches!(
            reader.section(SectionKind::STOREDMETA),
            Err(QuillError::IndexCorrupted { .. })
        ));
        Ok(())
    }

    #[test]
    fn verify_freshly_checks_whole_file_after_section_hashes_pass() -> TestResult {
        let owned = fixture_sections(MINIMAL_SCHEMA, false);
        let encoded = encode_owned(fixture_header(MINIMAL_SCHEMA), &owned)?;
        let postings_index = encoded
            .section_entries()
            .iter()
            .position(|entry| entry.kind == SectionKind::POSTINGS)
            .expect("POSTINGS index");
        let postings = encoded.section_entries()[postings_index];
        let mut bytes = encoded.into_bytes();
        let postings_offset = usize::try_from(postings.offset)?;
        bytes[postings_offset] ^= 0x40;
        let postings_len = usize::try_from(postings.len)?;
        let replacement_hash = xxh3_64(&bytes[postings_offset..postings_offset + postings_len]);
        let hash_offset = table_entry_offset(postings_index) + 20;
        bytes[hash_offset..hash_offset + 8].copy_from_slice(&replacement_hash.to_le_bytes());
        rewrite_header_crc(&mut bytes);

        let reader = SegmentReader::from_owned(bytes, MINIMAL_SCHEMA)?;
        assert!(reader.section(SectionKind::POSTINGS)?.is_some());
        assert!(matches!(
            reader.verify(),
            Err(QuillError::IndexCorrupted { .. })
        ));
        Ok(())
    }

    #[test]
    fn retryable_temp_adopts_exact_existing_bytes() -> TestResult {
        let directory = tempfile::tempdir()?.keep();
        let owned = fixture_sections(DEFAULT_SCHEMA, true);
        let encoded = encode_owned(fixture_header(DEFAULT_SCHEMA), &owned)?;
        let first = encoded.write_temp(&directory)?;
        let temp_path = first.path().to_path_buf();
        let expected = fs::read(&temp_path)?;

        let retry = encoded.write_temp_retryable(&directory)?;

        assert_eq!(retry.path(), temp_path);
        assert_eq!(retry.segment_id(), first.segment_id());
        assert_eq!(retry.file_len(), first.file_len());
        assert_eq!(retry.file_xxh3(), first.file_xxh3());
        assert_eq!(retry.source_xxh3(), first.source_xxh3());
        assert_eq!(fs::read(&temp_path)?, expected);
        Ok(())
    }

    #[test]
    fn retryable_temp_rejects_and_preserves_differing_bytes() -> TestResult {
        let directory = tempfile::tempdir()?.keep();
        let owned = fixture_sections(DEFAULT_SCHEMA, true);
        let encoded = encode_owned(fixture_header(DEFAULT_SCHEMA), &owned)?;
        let pending = encoded.write_temp(&directory)?;
        let temp_path = pending.path().to_path_buf();
        let mut differing = fs::read(&temp_path)?;
        let changed = differing
            .last_mut()
            .ok_or("encoded segment fixture must not be empty")?;
        *changed ^= 0x80;
        fs::write(&temp_path, &differing)?;

        assert!(matches!(
            encoded.write_temp_retryable(&directory),
            Err(QuillError::Io(error))
                if error.kind() == std::io::ErrorKind::AlreadyExists
        ));
        assert_eq!(fs::read(&temp_path)?, differing);
        Ok(())
    }

    #[test]
    fn mmap_matches_owned_reader_and_temp_creation_never_overwrites() -> TestResult {
        let owned = fixture_sections(DEFAULT_SCHEMA, true);
        let encoded = encode_owned(fixture_header(DEFAULT_SCHEMA), &owned)?;
        let directory = tempfile::tempdir()?;
        let pending = encoded.write_temp(directory.path())?;
        assert_eq!(
            pending.path().file_name().and_then(|name| name.to_str()),
            Some(".tmp-segment-0123456789abcdef")
        );
        assert_eq!(pending.segment_id(), encoded.header().segment_id);
        assert_eq!(pending.file_len(), encoded.file_len());
        assert_eq!(pending.file_xxh3(), encoded.file_xxh3());
        assert_eq!(pending.source_xxh3(), encoded.source_xxh3());
        let pending_bytes = fs::read(pending.path())?;
        assert_eq!(pending_bytes, encoded.as_bytes());
        assert_eq!(pending.source_xxh3(), xxh3_64(&pending_bytes));

        assert!(matches!(
            SegmentReader::open_published(pending.path(), DEFAULT_SCHEMA),
            Err(QuillError::IndexCorrupted { .. })
        ));
        let second_create = encoded.write_temp(directory.path());
        assert!(matches!(
            second_create,
            Err(QuillError::Io(error)) if error.kind() == std::io::ErrorKind::AlreadyExists
        ));
        assert_eq!(fs::read(pending.path())?, encoded.as_bytes());

        let published = directory.path().join("seg-0123456789abcdef.fslx");
        fs::rename(pending.path(), &published)?;
        let mapped = SegmentReader::open_published(&published, DEFAULT_SCHEMA)?;
        let memory = SegmentReader::from_owned(encoded.as_bytes().to_vec(), DEFAULT_SCHEMA)?;
        assert_eq!(mapped.header(), memory.header());
        assert_eq!(mapped.section_entries(), memory.section_entries());
        for expected in &owned {
            assert_eq!(
                mapped.section(expected.kind)?,
                memory.section(expected.kind)?
            );
        }
        mapped.verify()?;

        let wrong_name = directory.path().join("seg-fedcba9876543210.fslx");
        fs::write(&wrong_name, encoded.as_bytes())?;
        assert!(matches!(
            SegmentReader::open_published(&wrong_name, DEFAULT_SCHEMA),
            Err(QuillError::IndexCorrupted { .. })
        ));
        Ok(())
    }

    #[test]
    fn published_segment_names_are_canonical_lowercase_hex() {
        assert_eq!(
            published_segment_id(Path::new("seg-0123456789abcdef.fslx")),
            Ok(0x0123_4567_89ab_cdef)
        );
        for invalid in [
            ".tmp-segment-0123456789abcdef",
            "seg-0123456789ABCDEF.fslx",
            "seg-1234.fslx",
            "seg-0123456789abcdef.fslx.quarantine",
        ] {
            assert!(
                published_segment_id(Path::new(invalid)).is_err(),
                "accepted {invalid}"
            );
        }
    }

    #[test]
    fn framing_layout_and_schema_corruption_are_typed() -> TestResult {
        let owned = fixture_sections(MINIMAL_SCHEMA, false);
        let encoded = encode_owned(fixture_header(MINIMAL_SCHEMA), &owned)?;
        let original = encoded.as_bytes();

        let mut bad_magic = original.to_vec();
        bad_magic[0] ^= 1;
        assert!(matches!(
            SegmentReader::from_owned(bad_magic, MINIMAL_SCHEMA),
            Err(QuillError::IndexCorrupted { .. })
        ));

        let mut bad_version = original.to_vec();
        bad_version[8..12].copy_from_slice(&2_u32.to_le_bytes());
        assert!(matches!(
            SegmentReader::from_owned(bad_version, MINIMAL_SCHEMA),
            Err(QuillError::IndexCorrupted { .. })
        ));

        let mut bad_reserved = original.to_vec();
        let reserved_offset = FILE_PREFIX_LEN + 36;
        bad_reserved[reserved_offset..reserved_offset + 4].copy_from_slice(&1_u32.to_le_bytes());
        rewrite_header_crc(&mut bad_reserved);
        assert!(matches!(
            SegmentReader::from_owned(bad_reserved, MINIMAL_SCHEMA),
            Err(QuillError::IndexCorrupted { .. })
        ));

        let mut bad_offset = original.to_vec();
        let first_offset_field = table_entry_offset(0) + 4;
        let first_offset = read_u64_at(&bad_offset, first_offset_field).expect("first offset");
        bad_offset[first_offset_field..first_offset_field + 8]
            .copy_from_slice(&(first_offset + 1).to_le_bytes());
        rewrite_header_crc(&mut bad_offset);
        assert!(matches!(
            SegmentReader::from_owned(bad_offset, MINIMAL_SCHEMA),
            Err(QuillError::IndexCorrupted { .. })
        ));

        let declared = usize::try_from(read_u32_at(original, 12).expect("header length"))?;
        let header_crc_end = FILE_PREFIX_LEN + declared + HEADER_CRC_LEN;
        let first_section = usize::try_from(encoded.section_entries()[0].offset)?;
        assert!(header_crc_end < first_section);
        let mut bad_padding = original.to_vec();
        bad_padding[header_crc_end] = 1;
        assert!(matches!(
            SegmentReader::from_owned(bad_padding, MINIMAL_SCHEMA),
            Err(QuillError::IndexCorrupted { .. })
        ));

        let mut bad_trailer = original.to_vec();
        let last = bad_trailer.len() - 1;
        bad_trailer[last] ^= 1;
        assert!(matches!(
            SegmentReader::from_owned(bad_trailer, MINIMAL_SCHEMA),
            Err(QuillError::IndexCorrupted { .. })
        ));

        assert!(matches!(
            SegmentReader::from_owned(original.to_vec(), DEFAULT_SCHEMA),
            Err(QuillError::UnknownSchema { .. })
        ));
        Ok(())
    }

    #[test]
    fn every_kib_torn_boundary_and_hostile_prefix_fails_without_panicking() -> TestResult {
        const HOSTILE_SEED: u64 = 0x9e37_79b9_7f4a_7c15;

        init_replay_tracing();
        let owned = fixture_sections(DEFAULT_SCHEMA, true);
        let encoded = encode_owned(fixture_header(DEFAULT_SCHEMA), &owned)?;
        let bytes = encoded.as_bytes();
        assert!(bytes.len() > 4 * 1024);
        for cut in (0..bytes.len()).step_by(1024) {
            let result = catch_unwind(AssertUnwindSafe(|| {
                SegmentReader::from_owned(bytes[..cut].to_vec(), DEFAULT_SCHEMA)
            }));
            assert!(result.is_ok(), "parser panicked at torn boundary {cut}");
            assert!(matches!(
                result.expect("caught result"),
                Err(QuillError::IndexCorrupted { .. })
            ));
        }
        let final_cut = bytes.len() - 1;
        let result = catch_unwind(AssertUnwindSafe(|| {
            SegmentReader::from_owned(bytes[..final_cut].to_vec(), DEFAULT_SCHEMA)
        }));
        assert!(result.is_ok());
        assert!(matches!(
            result.expect("caught final-cut result"),
            Err(QuillError::IndexCorrupted { .. })
        ));

        let mut hostile_header = bytes.to_vec();
        hostile_header[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        let result = catch_unwind(AssertUnwindSafe(|| {
            SegmentReader::from_owned(hostile_header, DEFAULT_SCHEMA)
        }));
        assert!(result.is_ok());
        assert!(matches!(
            result.expect("caught hostile-header result"),
            Err(QuillError::Resource { .. })
        ));

        let mut state = HOSTILE_SEED;
        for case in 0..256 {
            tracing::info!(
                codec = "FSLX_CONTAINER",
                seed = HOSTILE_SEED,
                case,
                replay = "every_kib_torn_boundary_and_hostile_prefix_fails_without_panicking",
                "starting hostile-prefix case"
            );
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let len = usize::try_from(state % 513).unwrap_or(0);
            let mut input = Vec::new();
            input.try_reserve_exact(len)?;
            for index in 0..len {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let byte =
                    u8::try_from((state ^ u64::try_from(index).unwrap_or(0)) & 0xff).unwrap_or(0);
                input.push(byte);
            }
            let result = catch_unwind(AssertUnwindSafe(|| {
                SegmentReader::from_owned(input, DEFAULT_SCHEMA)
            }));
            assert!(
                result.is_ok(),
                "seed={HOSTILE_SEED:#x} hostile case {case} panicked"
            );
            assert!(matches!(
                result.expect("caught hostile result"),
                Err(QuillError::IndexCorrupted { .. })
            ));
        }
        Ok(())
    }
}
