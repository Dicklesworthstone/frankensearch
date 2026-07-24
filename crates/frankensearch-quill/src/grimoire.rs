//! Grimoire term dictionaries.
//!
//! FSLX TERMDICT sections use field-namespaced, prefix-compressed blocks. The
//! durable bytes contain a block-first-key index and full restart keys every
//! [`TERM_RESTART_INTERVAL`] entries. Opening a dictionary validates the whole
//! section and builds a bounded in-memory restart directory; exact lookup then
//! decodes at most one restart group while ordered scans reuse one key buffer.

use std::cmp::Ordering;
use std::ops::{Bound, Range};

use thiserror::Error;

use crate::schema::{FieldKind, SchemaDescriptor};

/// Number of entries between full-key restart points within a block.
pub const TERM_RESTART_INTERVAL: usize = 16;
/// Canonical target size for an ordinary TERMDICT block, including its header.
pub const TERM_BLOCK_TARGET_BYTES: usize = 4_096;
/// Language-contract maximum for the term portion of a composite key.
pub const MAX_TERM_BYTES: usize = 65_530;
/// Maximum composite key size: big-endian field ordinal plus term bytes.
pub const MAX_COMPOSITE_KEY_BYTES: usize = MAX_TERM_BYTES + 2;
/// Default cap on a borrowed TERMDICT section.
pub const DEFAULT_MAX_TERM_DICTIONARY_BYTES: usize = 1 << 30;
/// Default cap on block metadata retained by a reader.
pub const DEFAULT_MAX_TERM_BLOCKS: usize = 1 << 20;
/// Default cap on terms eagerly validated by a reader.
pub const DEFAULT_MAX_TERMS: usize = 1 << 24;
/// Default cap on restart metadata retained by a reader.
pub const DEFAULT_MAX_TERM_RESTARTS: usize = 1 << 20;

const BLOCK_COUNT_BYTES: usize = 4;
const BLOCK_ENTRY_COUNT_BYTES: usize = 2;
const MIN_WIRE_BYTES_PER_BLOCK: usize = 14;
const MIN_WIRE_BYTES_PER_ENTRY: usize = 7;

/// One byte range inside another FSLX section.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ByteSpan {
    /// Offset from the referenced section's first byte.
    pub offset: u64,
    /// Number of referenced bytes.
    pub len: u64,
}

impl ByteSpan {
    /// Construct a section-relative byte span.
    #[must_use]
    pub const fn new(offset: u64, len: u64) -> Self {
        Self { offset, len }
    }

    fn end(self, section: &'static str) -> Result<u64, TermDictionaryError> {
        self.offset
            .checked_add(self.len)
            .ok_or(TermDictionaryError::ReferenceOverflow {
                section,
                offset: self.offset,
                len: self.len,
            })
    }
}

/// Durable metadata associated with one composite term key.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TermMetadata {
    /// Documents containing the term in this segment, including tombstoned docs.
    pub doc_freq: u32,
    /// Exact term stream inside POSTINGS.
    pub postings: ByteSpan,
    /// Exact term stream inside POSITIONS for position-indexed text fields.
    pub positions: Option<ByteSpan>,
    /// Exact term stream inside BLOCKMAX.
    pub blockmax: ByteSpan,
}

impl TermMetadata {
    /// Construct metadata for a field that does not store positions.
    #[must_use]
    pub const fn without_positions(doc_freq: u32, postings: ByteSpan, blockmax: ByteSpan) -> Self {
        Self {
            doc_freq,
            postings,
            positions: None,
            blockmax,
        }
    }

    /// Construct metadata for a field that stores positions.
    #[must_use]
    pub const fn with_positions(
        doc_freq: u32,
        postings: ByteSpan,
        positions: ByteSpan,
        blockmax: ByteSpan,
    ) -> Self {
        Self {
            doc_freq,
            postings,
            positions: Some(positions),
            blockmax,
        }
    }
}

/// Borrowed, sorted encoder input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TermInput<'a> {
    /// Stable schema field ordinal.
    pub field_ord: u16,
    /// Opaque term bytes, with a maximum length of [`MAX_TERM_BYTES`].
    pub term: &'a [u8],
    /// Section references and document frequency.
    pub metadata: TermMetadata,
}

impl<'a> TermInput<'a> {
    /// Construct one encoder input row.
    #[must_use]
    pub const fn new(field_ord: u16, term: &'a [u8], metadata: TermMetadata) -> Self {
        Self {
            field_ord,
            term,
            metadata,
        }
    }
}

/// Section lengths required to validate every TERMDICT reference.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TermSectionLengths {
    /// POSTINGS section length.
    pub postings: u64,
    /// POSITIONS section length; presence must match the schema.
    pub positions: Option<u64>,
    /// BLOCKMAX section length.
    pub blockmax: u64,
}

/// Explicit work and allocation budgets for opening a TERMDICT section.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TermDictionaryLimits {
    /// Maximum accepted section bytes.
    pub max_bytes: usize,
    /// Maximum retained block metadata entries.
    pub max_blocks: usize,
    /// Maximum validated terms.
    pub max_terms: usize,
    /// Maximum retained restart metadata entries.
    pub max_restarts: usize,
}

impl Default for TermDictionaryLimits {
    fn default() -> Self {
        Self {
            max_bytes: DEFAULT_MAX_TERM_DICTIONARY_BYTES,
            max_blocks: DEFAULT_MAX_TERM_BLOCKS,
            max_terms: DEFAULT_MAX_TERMS,
            max_restarts: DEFAULT_MAX_TERM_RESTARTS,
        }
    }
}

/// Result of an exact term lookup.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TermMatch {
    /// Dense segment-local ordinal in composite-key order.
    pub term_ord: u32,
    /// Validated section references and document frequency.
    pub metadata: TermMetadata,
}

/// Borrowed view of the cursor's current reconstructed key.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TermRef<'a> {
    /// Dense segment-local ordinal in composite-key order.
    pub term_ord: u32,
    /// Stable schema field ordinal.
    pub field_ord: u16,
    /// Opaque term bytes borrowed from cursor scratch storage.
    pub term: &'a [u8],
    /// Validated section references and document frequency.
    pub metadata: TermMetadata,
}

/// Owned row returned by explicitly bounded materialization.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OwnedTerm {
    /// Dense segment-local ordinal in composite-key order.
    pub term_ord: u32,
    /// Stable schema field ordinal.
    pub field_ord: u16,
    /// Opaque term bytes.
    pub term: Vec<u8>,
    /// Validated section references and document frequency.
    pub metadata: TermMetadata,
}

impl TermRef<'_> {
    fn to_owned(self) -> Result<OwnedTerm, TermDictionaryError> {
        let mut term = Vec::new();
        term.try_reserve_exact(self.term.len())
            .map_err(|_| TermDictionaryError::Allocation {
                context: "owned term bytes",
                count: self.term.len(),
            })?;
        term.extend_from_slice(self.term);
        Ok(OwnedTerm {
            term_ord: self.term_ord,
            field_ord: self.field_ord,
            term,
            metadata: self.metadata,
        })
    }
}

/// Typed failures from encoding or validating an FSLX TERMDICT section.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum TermDictionaryError {
    /// The supplied compile-time schema violates its own invariants.
    #[error("invalid schema for TERMDICT: {detail}")]
    InvalidSchema {
        /// Schema validation detail.
        detail: String,
    },
    /// The section exceeds a caller-selected byte budget.
    #[error("TERMDICT length is at least {actual} bytes, exceeding byte budget {limit}")]
    ByteBudgetExceeded {
        /// Configured byte cap.
        limit: usize,
        /// Supplied section length.
        actual: usize,
    },
    /// The declared block count exceeds a caller-selected cap.
    #[error("TERMDICT block count {actual} exceeds block budget {limit}")]
    BlockBudgetExceeded {
        /// Configured block cap.
        limit: usize,
        /// Declared block count.
        actual: usize,
    },
    /// The validated term count exceeds a caller-selected cap.
    #[error("TERMDICT term count {actual} exceeds term budget {limit}")]
    TermBudgetExceeded {
        /// Configured term cap.
        limit: usize,
        /// Declared or accumulated term count.
        actual: usize,
    },
    /// The restart count exceeds a caller-selected cap.
    #[error("TERMDICT restart count {actual} exceeds restart budget {limit}")]
    RestartBudgetExceeded {
        /// Configured restart cap.
        limit: usize,
        /// Declared or accumulated restart count.
        actual: usize,
    },
    /// A tiny byte slice declares more nonempty blocks than can fit.
    #[error("TERMDICT declares {block_count} blocks in only {byte_len} bytes")]
    ImplausibleBlockCount {
        /// Declared block count.
        block_count: usize,
        /// Total supplied bytes.
        byte_len: usize,
    },
    /// Durable bytes ended before a declared field did.
    #[error("truncated TERMDICT at offset {offset}: need {needed} bytes, only {remaining} remain")]
    Truncated {
        /// Absolute offset of the failed read.
        offset: usize,
        /// Required byte count.
        needed: usize,
        /// Available byte count inside the current bounded region.
        remaining: usize,
    },
    /// An unsigned LEB128 value used more bytes than its shortest form.
    #[error("non-canonical vint at TERMDICT offset {offset}")]
    NonCanonicalVint {
        /// Vint start offset.
        offset: usize,
    },
    /// An unsigned LEB128 value exceeds u64 or ten bytes.
    #[error("vint overflow at TERMDICT offset {offset}")]
    VintOverflow {
        /// Vint start offset.
        offset: usize,
    },
    /// A decoded scalar does not fit its semantic domain.
    #[error("TERMDICT {field} value {value} is out of range at offset {offset}")]
    ValueOutOfRange {
        /// Stable field name.
        field: &'static str,
        /// Rejected value.
        value: u64,
        /// Field or entry offset.
        offset: usize,
    },
    /// A term exceeds the language contract.
    #[error("term at ordinal {term_ordinal} has {length} bytes; maximum is {MAX_TERM_BYTES}")]
    TermTooLong {
        /// Dense term ordinal or encoder input index.
        term_ordinal: usize,
        /// Rejected term length.
        length: usize,
    },
    /// A composite key references no schema field.
    #[error("TERMDICT key references unknown field ordinal {field_ord}")]
    UnknownField {
        /// Rejected field ordinal.
        field_ord: u16,
    },
    /// Numeric and stored-only fields cannot appear in TERMDICT.
    #[error("schema field ordinal {field_ord} is not a term-indexed field")]
    NonTermField {
        /// Rejected field ordinal.
        field_ord: u16,
    },
    /// Encoder inputs are not strictly sorted and unique.
    #[error("encoder keys are not strictly ascending at input {index}")]
    NonAscendingInput {
        /// Rejected input index.
        index: usize,
    },
    /// Decoded composite keys are not globally strict.
    #[error("decoded TERMDICT keys are not strictly ascending at term ordinal {term_ordinal}")]
    NonAscendingKey {
        /// Rejected dense term ordinal.
        term_ordinal: usize,
    },
    /// A term entry has no posting-bearing documents.
    #[error("TERMDICT term ordinal {term_ordinal} has zero doc_freq")]
    ZeroDocFrequency {
        /// Rejected dense term ordinal.
        term_ordinal: usize,
    },
    /// A required per-term section range is empty.
    #[error("TERMDICT term ordinal {term_ordinal} has an empty {section} span")]
    EmptyReference {
        /// Referenced section.
        section: &'static str,
        /// Rejected dense term ordinal.
        term_ordinal: usize,
    },
    /// Adding a range offset and length overflowed u64.
    #[error("{section} range overflows: offset {offset}, length {len}")]
    ReferenceOverflow {
        /// Referenced section.
        section: &'static str,
        /// Section-relative offset.
        offset: u64,
        /// Range length.
        len: u64,
    },
    /// A per-term range exceeds its referenced section.
    #[error("{section} range ends at {end}, beyond section length {limit}")]
    ReferenceOutOfBounds {
        /// Referenced section.
        section: &'static str,
        /// Checked range end.
        end: u64,
        /// Declared section length.
        limit: u64,
    },
    /// Per-term section ranges are not contiguous in composite-key order.
    #[error("non-contiguous {section} range: expected offset {expected}, got {actual}")]
    NonContiguousReference {
        /// Referenced section.
        section: &'static str,
        /// Required next offset.
        expected: u64,
        /// Encoded next offset.
        actual: u64,
    },
    /// Final per-term ranges do not consume their declared section.
    #[error("{section} ranges consume {actual} bytes, expected {expected}")]
    SectionLengthMismatch {
        /// Referenced section.
        section: &'static str,
        /// Declared section length.
        expected: u64,
        /// Accumulated range end.
        actual: u64,
    },
    /// Entry payload width disagrees with its schema field.
    #[error("field {field_ord} positions presence mismatch: expected {expected}, got {actual}")]
    PositionsPresenceMismatch {
        /// Schema field ordinal.
        field_ord: u16,
        /// Schema-derived positions presence.
        expected: bool,
        /// Entry payload positions presence.
        actual: bool,
    },
    /// POSITIONS section presence disagrees with the schema as a whole.
    #[error("POSITIONS section presence mismatch: expected {expected}, got {actual}")]
    PositionsSectionMismatch {
        /// Whether the schema requires the section.
        expected: bool,
        /// Whether a section length was supplied.
        actual: bool,
    },
    /// A block index key is not strictly ascending.
    #[error("TERMDICT block index keys are not strictly ascending at block {block_index}")]
    NonAscendingIndexKey {
        /// Rejected block index.
        block_index: usize,
    },
    /// A relative block offset violates the contiguous block-area layout.
    #[error("invalid relative block offset {offset} for block {block_index}: {detail}")]
    InvalidBlockOffset {
        /// Rejected block index.
        block_index: usize,
        /// Encoded relative offset.
        offset: u64,
        /// Stable invariant detail.
        detail: &'static str,
    },
    /// Blocks must contain at least one entry.
    #[error("TERMDICT block {block_index} is empty")]
    EmptyBlock {
        /// Rejected block index.
        block_index: usize,
    },
    /// A declared entry count cannot fit in its block bytes.
    #[error("TERMDICT block {block_index} declares {entry_count} entries in {byte_len} bytes")]
    ImplausibleEntryCount {
        /// Rejected block index.
        block_index: usize,
        /// Declared entry count.
        entry_count: usize,
        /// Delimited block length.
        byte_len: usize,
    },
    /// Only an oversized singleton term may exceed the ordinary block target.
    #[error("TERMDICT block {block_index} has {entry_count} entries in {byte_len} bytes")]
    OversizedBlock {
        /// Rejected block index.
        block_index: usize,
        /// Declared entry count.
        entry_count: usize,
        /// Delimited block length.
        byte_len: usize,
    },
    /// A writer split a block even though the next entry still fit.
    #[error("TERMDICT block {block_index} was split before the 4096-byte target")]
    PrematureBlockSplit {
        /// Earlier block index.
        block_index: usize,
    },
    /// A compressed entry references bytes outside its previous key.
    #[error(
        "invalid shared prefix {shared} for previous key length {previous_len} at offset {offset}"
    )]
    InvalidPrefix {
        /// Encoded shared-prefix length.
        shared: usize,
        /// Previous key length.
        previous_len: usize,
        /// Entry start offset.
        offset: usize,
    },
    /// A compressed entry did not use the longest possible shared prefix.
    #[error("non-canonical shared prefix at TERMDICT offset {offset}")]
    NonCanonicalPrefix {
        /// Entry start offset.
        offset: usize,
    },
    /// The block index first key disagrees with its decoded block.
    #[error("TERMDICT block {block_index} first key disagrees with its index key")]
    IndexKeyMismatch {
        /// Rejected block index.
        block_index: usize,
    },
    /// Declared entries did not consume their exact delimited block.
    #[error("TERMDICT block {block_index} has {remaining} trailing bytes")]
    TrailingBlockBytes {
        /// Rejected block index.
        block_index: usize,
        /// Unconsumed bytes.
        remaining: usize,
    },
    /// A zero-block dictionary contains bytes after its count.
    #[error("empty TERMDICT has {remaining} trailing bytes")]
    TrailingEmptyBytes {
        /// Bytes after the zero block count.
        remaining: usize,
    },
    /// A requested term range has reversed endpoints.
    #[error("TERMDICT range start sorts after its end")]
    InvalidRange,
    /// Explicitly bounded result materialization exceeded its cap.
    #[error("term materialization limit {limit} exceeded")]
    MaterializationLimitExceeded {
        /// Caller-selected result cap.
        limit: usize,
    },
    /// A glob matched one more dictionary term than the caller allowed.
    #[error(
        "glob expansion for field {field_ord} observed at least {actual} matches, exceeding limit {limit}"
    )]
    GlobExpansionLimitExceeded {
        /// Field whose dictionary terms were scanned.
        field_ord: u16,
        /// Caller-selected expansion cap.
        limit: usize,
        /// Number of matches observed before the scan stopped, always `limit + 1`.
        actual: usize,
    },
    /// A fallible allocation could not reserve the required space.
    #[error("unable to reserve {count} items for {context}")]
    Allocation {
        /// Stable allocation context.
        context: &'static str,
        /// Requested item or byte count.
        count: usize,
    },
    /// Checked host-size arithmetic overflowed.
    #[error("TERMDICT size arithmetic overflow while computing {field}")]
    SizeOverflow {
        /// Stable size field.
        field: &'static str,
    },
}

/// Owned bytes produced by the canonical TERMDICT writer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedTermDictionary {
    bytes: Vec<u8>,
    term_count: u32,
    block_count: u32,
    restart_count: usize,
}

impl EncodedTermDictionary {
    /// Encode strictly composite-key-sorted inputs with default resource caps.
    ///
    /// # Errors
    ///
    /// Returns a typed error for invalid schema/reference metadata, duplicate or
    /// descending keys, overlong terms, resource exhaustion, or arithmetic
    /// overflow.
    pub fn encode_sorted(
        schema: SchemaDescriptor,
        sections: TermSectionLengths,
        terms: &[TermInput<'_>],
    ) -> Result<Self, TermDictionaryError> {
        Self::encode_sorted_with_limits(schema, sections, terms, TermDictionaryLimits::default())
    }

    /// Encode strictly sorted inputs with explicit resource caps.
    ///
    /// # Errors
    ///
    /// Returns a typed error when input, metadata, or the resulting canonical
    /// bytes violate the FSLX contract or supplied budgets.
    pub fn encode_sorted_with_limits(
        schema: SchemaDescriptor,
        sections: TermSectionLengths,
        terms: &[TermInput<'_>],
        limits: TermDictionaryLimits,
    ) -> Result<Self, TermDictionaryError> {
        validate_schema(schema)?;
        validate_positions_section(schema, sections)?;
        if BLOCK_COUNT_BYTES > limits.max_bytes {
            return Err(TermDictionaryError::ByteBudgetExceeded {
                limit: limits.max_bytes,
                actual: BLOCK_COUNT_BYTES,
            });
        }
        if terms.len() > limits.max_terms {
            return Err(TermDictionaryError::TermBudgetExceeded {
                limit: limits.max_terms,
                actual: terms.len(),
            });
        }
        if !terms.is_empty() && limits.max_blocks == 0 {
            return Err(TermDictionaryError::BlockBudgetExceeded {
                limit: 0,
                actual: 1,
            });
        }
        if !terms.is_empty() && limits.max_restarts == 0 {
            return Err(TermDictionaryError::RestartBudgetExceeded {
                limit: 0,
                actual: 1,
            });
        }
        let term_count =
            u32::try_from(terms.len()).map_err(|_| TermDictionaryError::ValueOutOfRange {
                field: "term_count",
                value: u64::try_from(terms.len()).unwrap_or(u64::MAX),
                offset: 0,
            })?;

        let mut blocks = Vec::new();
        let mut references = ReferenceValidator::new(schema, sections)?;
        let mut previous_global = Vec::new();
        let mut previous_block = Vec::new();
        let mut current = new_block_buffer(limits.max_bytes)?;
        let mut current_first = None;
        let mut current_count = 0_usize;
        let mut completed_block_bytes = 0_usize;
        let mut restart_count = 0_usize;

        for (index, input) in terms.iter().enumerate() {
            let key = composite_key(input.field_ord, input.term, index)?;
            let has_positions = validate_composite_key(schema, &key, index)?;
            if !previous_global.is_empty() && key.as_slice() <= previous_global.as_slice() {
                return Err(TermDictionaryError::NonAscendingInput { index });
            }
            validate_metadata_basic(
                input.field_ord,
                has_positions,
                input.metadata,
                sections,
                index,
            )?;
            references.push(input.metadata)?;

            let mut encoded_entry = encode_entry(
                &key,
                &previous_block,
                current_count,
                input.metadata,
                has_positions,
            )?;
            if current_count != 0
                && current.len().checked_add(encoded_entry.len()).ok_or(
                    TermDictionaryError::SizeOverflow {
                        field: "block length",
                    },
                )? > TERM_BLOCK_TARGET_BYTES
            {
                finish_encoded_block(
                    &mut blocks,
                    &mut current,
                    &mut current_first,
                    current_count,
                    limits,
                    &mut completed_block_bytes,
                    &mut restart_count,
                )?;
                current = new_block_buffer(limits.max_bytes)?;
                current_count = 0;
                previous_block.clear();
                encoded_entry = encode_entry(
                    &key,
                    &previous_block,
                    current_count,
                    input.metadata,
                    has_positions,
                )?;
            }

            let minimum_output_len = checked_add_size(
                BLOCK_COUNT_BYTES,
                checked_add_size(
                    completed_block_bytes,
                    checked_add_size(current.len(), encoded_entry.len(), "pending block bytes")?,
                    "encoded blocks bytes",
                )?,
                "minimum TERMDICT length",
            )?;
            if minimum_output_len > limits.max_bytes {
                return Err(TermDictionaryError::ByteBudgetExceeded {
                    limit: limits.max_bytes,
                    actual: minimum_output_len,
                });
            }

            if current_first.is_none() {
                current_first = Some(clone_bytes(&key, "block first key")?);
            }
            append_bytes(&mut current, &encoded_entry, "encoded block bytes")?;
            current_count =
                current_count
                    .checked_add(1)
                    .ok_or(TermDictionaryError::SizeOverflow {
                        field: "block entry count",
                    })?;
            copy_bytes(&mut previous_block, &key, "previous block key")?;
            copy_bytes(&mut previous_global, &key, "previous global key")?;
        }
        if current_count != 0 {
            finish_encoded_block(
                &mut blocks,
                &mut current,
                &mut current_first,
                current_count,
                limits,
                &mut completed_block_bytes,
                &mut restart_count,
            )?;
        }
        references.finish()?;

        let block_count =
            u32::try_from(blocks.len()).map_err(|_| TermDictionaryError::ValueOutOfRange {
                field: "block_count",
                value: u64::try_from(blocks.len()).unwrap_or(u64::MAX),
                offset: 0,
            })?;
        if blocks.len() > limits.max_blocks {
            return Err(TermDictionaryError::BlockBudgetExceeded {
                limit: limits.max_blocks,
                actual: blocks.len(),
            });
        }
        debug_assert_eq!(
            restart_count,
            blocks
                .iter()
                .map(|block| block.entry_count.div_ceil(TERM_RESTART_INTERVAL))
                .sum::<usize>()
        );
        debug_assert_eq!(
            completed_block_bytes,
            blocks.iter().map(|block| block.bytes.len()).sum::<usize>()
        );

        let mut relative_offset = 0_u64;
        let mut output_len = BLOCK_COUNT_BYTES;
        for block in &blocks {
            output_len = checked_add_size(
                output_len,
                vint_len(u64::try_from(block.first_key.len()).map_err(|_| {
                    TermDictionaryError::SizeOverflow {
                        field: "index first-key length",
                    }
                })?),
                "block index key length vint",
            )?;
            output_len =
                checked_add_size(output_len, block.first_key.len(), "block index key bytes")?;
            output_len = checked_add_size(
                output_len,
                vint_len(relative_offset),
                "block index offset vint",
            )?;
            relative_offset = relative_offset
                .checked_add(u64::try_from(block.bytes.len()).map_err(|_| {
                    TermDictionaryError::SizeOverflow {
                        field: "relative block offset",
                    }
                })?)
                .ok_or(TermDictionaryError::SizeOverflow {
                    field: "relative block offset",
                })?;
        }
        output_len = checked_add_size(
            output_len,
            usize::try_from(relative_offset).map_err(|_| TermDictionaryError::SizeOverflow {
                field: "blocks area length",
            })?,
            "TERMDICT length",
        )?;
        if output_len > limits.max_bytes {
            return Err(TermDictionaryError::ByteBudgetExceeded {
                limit: limits.max_bytes,
                actual: output_len,
            });
        }

        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(output_len)
            .map_err(|_| TermDictionaryError::Allocation {
                context: "encoded TERMDICT",
                count: output_len,
            })?;
        bytes.extend_from_slice(&block_count.to_le_bytes());
        relative_offset = 0;
        for block in &blocks {
            write_vint(
                u64::try_from(block.first_key.len()).map_err(|_| {
                    TermDictionaryError::SizeOverflow {
                        field: "index key length",
                    }
                })?,
                &mut bytes,
            );
            bytes.extend_from_slice(&block.first_key);
            write_vint(relative_offset, &mut bytes);
            relative_offset = relative_offset
                .checked_add(u64::try_from(block.bytes.len()).map_err(|_| {
                    TermDictionaryError::SizeOverflow {
                        field: "relative block offset",
                    }
                })?)
                .ok_or(TermDictionaryError::SizeOverflow {
                    field: "relative block offset",
                })?;
        }
        for block in &blocks {
            bytes.extend_from_slice(&block.bytes);
        }
        debug_assert_eq!(bytes.len(), output_len);

        let owned_limits = TermDictionaryLimits {
            max_bytes: bytes.len(),
            max_blocks: blocks.len(),
            max_terms: terms.len(),
            max_restarts: restart_count,
        };
        let parsed = TermDictionary::parse_with_limits(&bytes, schema, sections, owned_limits)?;
        debug_assert_eq!(parsed.restart_count(), restart_count);
        debug_assert_eq!(parsed.term_count(), term_count);
        Ok(Self {
            bytes,
            term_count,
            block_count,
            restart_count,
        })
    }

    /// Exact canonical durable bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consume the wrapper and return its canonical bytes.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Number of encoded terms.
    #[must_use]
    pub const fn term_count(&self) -> u32 {
        self.term_count
    }

    /// Number of encoded blocks.
    #[must_use]
    pub const fn block_count(&self) -> u32 {
        self.block_count
    }

    /// Re-open the owned bytes through the same validating reader.
    ///
    /// # Errors
    ///
    /// Returns a typed error if an internal encoder invariant was violated.
    pub fn dictionary(
        &self,
        schema: SchemaDescriptor,
        sections: TermSectionLengths,
    ) -> Result<TermDictionary<'_>, TermDictionaryError> {
        TermDictionary::parse_with_limits(
            &self.bytes,
            schema,
            sections,
            TermDictionaryLimits {
                max_bytes: self.bytes.len(),
                max_blocks: usize::try_from(self.block_count).unwrap_or(usize::MAX),
                max_terms: usize::try_from(self.term_count).unwrap_or(usize::MAX),
                max_restarts: self.restart_count,
            },
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct EncodedBlock {
    first_key: Vec<u8>,
    bytes: Vec<u8>,
    entry_count: usize,
}

#[derive(Clone, Debug)]
struct BlockMeta {
    first_key_range: Range<usize>,
    byte_range: Range<usize>,
    entry_count: usize,
    base_term_ord: u32,
    restart_range: Range<usize>,
}

#[derive(Clone, Debug)]
struct RestartMeta {
    entry_ordinal: usize,
    term_ordinal: u32,
    entry_offset: usize,
    key_range: Range<usize>,
}

#[derive(Clone, Copy, Debug)]
struct BlockTail {
    byte_len: usize,
    entry_count: usize,
}

#[derive(Clone, Debug)]
struct IndexRecord {
    first_key_range: Range<usize>,
    relative_offset: u64,
}

/// Borrowed, eagerly validated TERMDICT view.
#[derive(Clone, Debug)]
pub struct TermDictionary<'a> {
    bytes: &'a [u8],
    schema: SchemaDescriptor,
    sections: TermSectionLengths,
    blocks: Vec<BlockMeta>,
    restarts: Vec<RestartMeta>,
    term_count: u32,
}

impl<'a> TermDictionary<'a> {
    /// Validate a complete TERMDICT section with default resource caps.
    ///
    /// # Errors
    ///
    /// Returns a typed error for malformed, non-canonical, out-of-budget, or
    /// cross-section-inconsistent bytes.
    pub fn parse(
        bytes: &'a [u8],
        schema: SchemaDescriptor,
        sections: TermSectionLengths,
    ) -> Result<Self, TermDictionaryError> {
        Self::parse_with_limits(bytes, schema, sections, TermDictionaryLimits::default())
    }

    /// Validate a complete TERMDICT section with explicit resource caps.
    ///
    /// # Errors
    ///
    /// Returns a typed error before exceeding the supplied byte, block, term,
    /// or restart budgets.
    pub fn parse_with_limits(
        bytes: &'a [u8],
        schema: SchemaDescriptor,
        sections: TermSectionLengths,
        limits: TermDictionaryLimits,
    ) -> Result<Self, TermDictionaryError> {
        validate_schema(schema)?;
        validate_positions_section(schema, sections)?;
        if bytes.len() > limits.max_bytes {
            return Err(TermDictionaryError::ByteBudgetExceeded {
                limit: limits.max_bytes,
                actual: bytes.len(),
            });
        }

        let mut reader = SliceReader::new(bytes);
        let block_count = usize::try_from(reader.read_u32()?).map_err(|_| {
            TermDictionaryError::ValueOutOfRange {
                field: "block_count",
                value: u64::MAX,
                offset: 0,
            }
        })?;
        if block_count > limits.max_blocks {
            return Err(TermDictionaryError::BlockBudgetExceeded {
                limit: limits.max_blocks,
                actual: block_count,
            });
        }
        if block_count > limits.max_terms {
            return Err(TermDictionaryError::TermBudgetExceeded {
                limit: limits.max_terms,
                actual: block_count,
            });
        }
        if block_count > limits.max_restarts {
            return Err(TermDictionaryError::RestartBudgetExceeded {
                limit: limits.max_restarts,
                actual: block_count,
            });
        }

        if block_count == 0 {
            let remaining = bytes.len().saturating_sub(reader.position());
            if remaining != 0 {
                return Err(TermDictionaryError::TrailingEmptyBytes { remaining });
            }
            ReferenceValidator::new(schema, sections)?.finish()?;
            return Ok(Self {
                bytes,
                schema,
                sections,
                blocks: Vec::new(),
                restarts: Vec::new(),
                term_count: 0,
            });
        }

        let feasible = bytes.len().saturating_sub(BLOCK_COUNT_BYTES) / MIN_WIRE_BYTES_PER_BLOCK;
        if block_count > feasible {
            return Err(TermDictionaryError::ImplausibleBlockCount {
                block_count,
                byte_len: bytes.len(),
            });
        }

        let mut index = Vec::new();
        index
            .try_reserve_exact(block_count)
            .map_err(|_| TermDictionaryError::Allocation {
                context: "block index metadata",
                count: block_count,
            })?;
        let mut previous_index_key: Option<Range<usize>> = None;
        for block_index in 0..block_count {
            let key_len_offset = reader.position();
            let key_len = usize_from_vint(reader.read_vint()?, "first_key_len", key_len_offset)?;
            if !(2..=MAX_COMPOSITE_KEY_BYTES).contains(&key_len) {
                return Err(TermDictionaryError::TermTooLong {
                    term_ordinal: block_index,
                    length: key_len.saturating_sub(2),
                });
            }
            let key_range = reader.take_range(key_len)?;
            validate_composite_key(schema, &bytes[key_range.clone()], block_index)?;
            if let Some(previous) = previous_index_key.as_ref() {
                if bytes[key_range.clone()] <= bytes[previous.clone()] {
                    return Err(TermDictionaryError::NonAscendingIndexKey { block_index });
                }
            }
            previous_index_key = Some(key_range.clone());
            let relative_offset = reader.read_vint()?;
            index.push(IndexRecord {
                first_key_range: key_range,
                relative_offset,
            });
        }

        let blocks_start = reader.position();
        let blocks_len = bytes.len().saturating_sub(blocks_start);
        let mut absolute_offsets = Vec::new();
        absolute_offsets
            .try_reserve_exact(block_count)
            .map_err(|_| TermDictionaryError::Allocation {
                context: "absolute block offsets",
                count: block_count,
            })?;
        let mut previous_relative = None;
        for (block_index, record) in index.iter().enumerate() {
            if block_index == 0 && record.relative_offset != 0 {
                return Err(TermDictionaryError::InvalidBlockOffset {
                    block_index,
                    offset: record.relative_offset,
                    detail: "first offset must be zero",
                });
            }
            if previous_relative.is_some_and(|previous| record.relative_offset <= previous) {
                return Err(TermDictionaryError::InvalidBlockOffset {
                    block_index,
                    offset: record.relative_offset,
                    detail: "offsets must be strictly increasing",
                });
            }
            let relative = usize::try_from(record.relative_offset).map_err(|_| {
                TermDictionaryError::InvalidBlockOffset {
                    block_index,
                    offset: record.relative_offset,
                    detail: "offset does not fit host usize",
                }
            })?;
            if relative >= blocks_len {
                return Err(TermDictionaryError::InvalidBlockOffset {
                    block_index,
                    offset: record.relative_offset,
                    detail: "offset lies outside the blocks area",
                });
            }
            absolute_offsets.push(blocks_start.checked_add(relative).ok_or(
                TermDictionaryError::SizeOverflow {
                    field: "absolute block offset",
                },
            )?);
            previous_relative = Some(record.relative_offset);
        }

        let mut blocks = Vec::new();
        let mut restarts = Vec::new();
        let mut references = ReferenceValidator::new(schema, sections)?;
        let mut previous_key = Vec::new();
        let mut decode_key = Vec::new();
        let mut previous_tail = None;
        let mut term_count = 0_usize;

        for block_index in 0..block_count {
            let start = absolute_offsets[block_index];
            let end = absolute_offsets
                .get(block_index + 1)
                .copied()
                .unwrap_or(bytes.len());
            let base_term_ord =
                u32::try_from(term_count).map_err(|_| TermDictionaryError::ValueOutOfRange {
                    field: "term ordinal",
                    value: u64::try_from(term_count).unwrap_or(u64::MAX),
                    offset: start,
                })?;
            let (meta, tail) = validate_block(
                bytes,
                schema,
                sections,
                limits,
                block_index,
                start..end,
                index[block_index].first_key_range.clone(),
                base_term_ord,
                &mut term_count,
                &mut restarts,
                &mut references,
                &mut decode_key,
                &mut previous_key,
                previous_tail.as_ref(),
            )?;
            blocks
                .try_reserve(1)
                .map_err(|_| TermDictionaryError::Allocation {
                    context: "block metadata",
                    count: blocks.len().saturating_add(1),
                })?;
            blocks.push(meta);
            previous_tail = Some(tail);
        }
        references.finish()?;
        let term_count_u32 =
            u32::try_from(term_count).map_err(|_| TermDictionaryError::ValueOutOfRange {
                field: "term_count",
                value: u64::try_from(term_count).unwrap_or(u64::MAX),
                offset: bytes.len(),
            })?;
        Ok(Self {
            bytes,
            schema,
            sections,
            blocks,
            restarts,
            term_count: term_count_u32,
        })
    }

    /// Exact validated durable bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Number of validated terms.
    #[must_use]
    pub const fn term_count(&self) -> u32 {
        self.term_count
    }

    /// Number of validated prefix-compressed blocks.
    #[must_use]
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Number of retained restart metadata entries.
    #[must_use]
    pub fn restart_count(&self) -> usize {
        self.restarts.len()
    }

    /// Exact lookup with temporary scratch allocated by this call.
    ///
    /// # Errors
    ///
    /// Returns a typed error for an unknown/non-term field, an overlong query
    /// term, allocation failure, or an internal validated-decode failure.
    pub fn lookup(
        &self,
        field_ord: u16,
        term: &[u8],
    ) -> Result<Option<TermMatch>, TermDictionaryError> {
        self.lookup_with_scratch(field_ord, term, &mut TermScratch::default())
    }

    /// Exact lookup reusing caller-owned key buffers.
    ///
    /// The block and restart indexes are binary-searched, then no more than 16
    /// entries are decoded.
    ///
    /// # Errors
    ///
    /// Returns a typed error for invalid query shape, scratch allocation
    /// failure, or an internal validated-decode failure.
    pub fn lookup_with_scratch(
        &self,
        field_ord: u16,
        term: &[u8],
        scratch: &mut TermScratch,
    ) -> Result<Option<TermMatch>, TermDictionaryError> {
        validate_query_term(self.schema, field_ord, term)?;
        build_composite_in(&mut scratch.target, field_ord, term, "lookup target")?;
        let target = scratch.target.as_slice();
        let Some(block_index) = self.block_for_key(target) else {
            return Ok(None);
        };
        let block = &self.blocks[block_index];
        let restart_slice = &self.restarts[block.restart_range.clone()];
        let restart_position = restart_slice
            .partition_point(|restart| self.bytes[restart.key_range.clone()] <= *target)
            .saturating_sub(1);
        let restart = &restart_slice[restart_position];
        scratch.decoded.clear();
        let mut reader =
            SliceReader::with_range(self.bytes, restart.entry_offset, block.byte_range.end)?;
        let group_end = restart
            .entry_ordinal
            .saturating_add(TERM_RESTART_INTERVAL)
            .min(block.entry_count);
        for entry_ordinal in restart.entry_ordinal..group_end {
            let restart_delta = entry_ordinal.saturating_sub(restart.entry_ordinal);
            let term_ordinal = restart
                .term_ordinal
                .checked_add(u32::try_from(restart_delta).map_err(|_| {
                    TermDictionaryError::ValueOutOfRange {
                        field: "term ordinal",
                        value: u64::try_from(entry_ordinal).unwrap_or(u64::MAX),
                        offset: reader.position(),
                    }
                })?)
                .ok_or(TermDictionaryError::SizeOverflow {
                    field: "term ordinal",
                })?;
            let decoded = decode_entry(
                &mut reader,
                entry_ordinal,
                &mut scratch.decoded,
                self.schema,
                self.sections,
                usize::try_from(term_ordinal).unwrap_or(usize::MAX),
            )?;
            match scratch.decoded.as_slice().cmp(target) {
                Ordering::Less => {}
                Ordering::Equal => {
                    return Ok(Some(TermMatch {
                        term_ord: term_ordinal,
                        metadata: decoded.metadata,
                    }));
                }
                Ordering::Greater => return Ok(None),
            }
        }
        Ok(None)
    }

    /// Cursor over every term in global field-major order.
    ///
    /// # Errors
    ///
    /// Returns a typed error if scratch allocation or validated decoding fails.
    pub fn cursor(&self) -> Result<TermCursor<'_, 'a>, TermDictionaryError> {
        TermCursor::new(self, None, None, None, None)
    }

    /// Cursor over one schema field in term-byte order.
    ///
    /// # Errors
    ///
    /// Returns a typed error for an unknown/non-term field or cursor setup
    /// failure.
    pub fn field_cursor(&self, field_ord: u16) -> Result<TermCursor<'_, 'a>, TermDictionaryError> {
        validate_query_term(self.schema, field_ord, &[])?;
        let lower = CursorBound::included(composite_key(field_ord, &[], 0)?);
        TermCursor::new(self, Some(lower), None, Some(field_ord), None)
    }

    /// Cursor over terms in one field that start with `prefix`.
    ///
    /// # Errors
    ///
    /// Returns a typed error for invalid field/prefix shape or cursor setup
    /// failure.
    pub fn prefix_cursor(
        &self,
        field_ord: u16,
        prefix: &[u8],
    ) -> Result<TermCursor<'_, 'a>, TermDictionaryError> {
        validate_query_term(self.schema, field_ord, prefix)?;
        let lower_key = composite_key(field_ord, prefix, 0)?;
        let prefix_key = clone_bytes(&lower_key, "prefix cursor key")?;
        TermCursor::new(
            self,
            Some(CursorBound::included(lower_key)),
            None,
            Some(field_ord),
            Some(prefix_key),
        )
    }

    /// Cursor over a field-scoped range with standard bound semantics.
    ///
    /// The query contract's canonical half-open form is represented by an
    /// included `start` and excluded `end`; general bounds are accepted for
    /// callers that need an unbounded or inclusive endpoint.
    ///
    /// # Errors
    ///
    /// Returns a typed error for invalid fields, overlong endpoints, reversed
    /// bounds, or cursor setup failure.
    pub fn range_cursor(
        &self,
        field_ord: u16,
        start: Bound<&[u8]>,
        end: Bound<&[u8]>,
    ) -> Result<TermCursor<'_, 'a>, TermDictionaryError> {
        validate_query_term(self.schema, field_ord, &[])?;
        validate_bound_term(self.schema, field_ord, &start)?;
        validate_bound_term(self.schema, field_ord, &end)?;
        if range_is_reversed(&start, &end) {
            return Err(TermDictionaryError::InvalidRange);
        }
        let lower = match start {
            Bound::Included(term) => {
                Some(CursorBound::included(composite_key(field_ord, term, 0)?))
            }
            Bound::Excluded(term) => {
                Some(CursorBound::excluded(composite_key(field_ord, term, 0)?))
            }
            Bound::Unbounded => Some(CursorBound::included(composite_key(field_ord, &[], 0)?)),
        };
        let upper = match end {
            Bound::Included(term) => {
                Some(CursorBound::included(composite_key(field_ord, term, 0)?))
            }
            Bound::Excluded(term) => {
                Some(CursorBound::excluded(composite_key(field_ord, term, 0)?))
            }
            Bound::Unbounded => None,
        };
        TermCursor::new(self, lower, upper, Some(field_ord), None)
    }

    /// Expand a field-scoped byte glob under an explicit result cap.
    ///
    /// `*` is the only wildcard and matches any byte sequence, including the
    /// empty sequence. Repeated stars are equivalent to one star. Exact terms
    /// use indexed lookup, a sole trailing star uses the prefix cursor, and all
    /// other shapes scan only the requested field. Results retain dictionary
    /// order.
    ///
    /// # Errors
    ///
    /// Returns [`TermDictionaryError::GlobExpansionLimitExceeded`] as soon as
    /// one more matching term than `limit` is observed. It also propagates
    /// invalid field, allocation, and validated-decode errors.
    pub fn expand_glob(
        &self,
        field_ord: u16,
        pattern: &[u8],
        limit: usize,
    ) -> Result<Vec<OwnedTerm>, TermDictionaryError> {
        validate_query_term(self.schema, field_ord, &[])?;

        if !pattern.contains(&b'*') {
            return self.expand_exact_glob(field_ord, pattern, limit);
        }

        if let Some(prefix) = trailing_star_prefix(pattern) {
            return collect_glob_matches(
                self.prefix_cursor(field_ord, prefix)?,
                field_ord,
                limit,
                |_| true,
            );
        }

        let longest_literal = longest_glob_literal(pattern);
        collect_glob_matches(self.field_cursor(field_ord)?, field_ord, limit, |term| {
            contains_subslice(term, longest_literal) && star_glob_matches(pattern, term)
        })
    }

    fn expand_exact_glob(
        &self,
        field_ord: u16,
        term: &[u8],
        limit: usize,
    ) -> Result<Vec<OwnedTerm>, TermDictionaryError> {
        let Some(term_match) = self.lookup(field_ord, term)? else {
            return Ok(Vec::new());
        };
        if limit == 0 {
            return Err(glob_limit_error(field_ord, limit, 1));
        }

        let mut output = Vec::new();
        output
            .try_reserve_exact(1)
            .map_err(|_| TermDictionaryError::Allocation {
                context: "glob expansion rows",
                count: 1,
            })?;
        output.push(OwnedTerm {
            term_ord: term_match.term_ord,
            field_ord,
            term: clone_bytes(term, "exact glob term")?,
            metadata: term_match.metadata,
        });
        Ok(output)
    }

    fn block_for_key(&self, key: &[u8]) -> Option<usize> {
        let insertion = self
            .blocks
            .partition_point(|block| self.bytes[block.first_key_range.clone()] <= *key);
        insertion.checked_sub(1)
    }
}

fn collect_glob_matches<F>(
    mut cursor: TermCursor<'_, '_>,
    field_ord: u16,
    limit: usize,
    mut matches: F,
) -> Result<Vec<OwnedTerm>, TermDictionaryError>
where
    F: FnMut(&[u8]) -> bool,
{
    let mut output = Vec::new();
    while let Some(current) = cursor.current() {
        if matches(current.term) {
            if output.len() >= limit {
                return Err(glob_limit_error(
                    field_ord,
                    limit,
                    output.len().saturating_add(1),
                ));
            }
            output
                .try_reserve(1)
                .map_err(|_| TermDictionaryError::Allocation {
                    context: "glob expansion rows",
                    count: output.len().saturating_add(1),
                })?;
            output.push(current.to_owned()?);
        }
        cursor.next()?;
    }
    Ok(output)
}

const fn glob_limit_error(field_ord: u16, limit: usize, actual: usize) -> TermDictionaryError {
    TermDictionaryError::GlobExpansionLimitExceeded {
        field_ord,
        limit,
        actual,
    }
}

pub(crate) fn trailing_star_prefix(pattern: &[u8]) -> Option<&[u8]> {
    let first_star = pattern.iter().position(|byte| *byte == b'*')?;
    pattern[first_star..]
        .iter()
        .all(|byte| *byte == b'*')
        .then_some(&pattern[..first_star])
}

fn longest_glob_literal(pattern: &[u8]) -> &[u8] {
    pattern
        .split(|byte| *byte == b'*')
        .max_by_key(|literal| literal.len())
        .unwrap_or_default()
}

fn contains_subslice(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() {
        return true;
    }
    if needle.len() > haystack.len() {
        return false;
    }

    // Reject eight impossible starts per word before confirming the complete
    // literal. This is the safe SWAR analogue of memmem's first-byte filter;
    // the final `starts_with` remains the semantic authority.
    const BYTE_ONES: u64 = 0x0101_0101_0101_0101;
    const BYTE_HIGHS: u64 = 0x8080_8080_8080_8080;

    let final_start = haystack.len() - needle.len();
    let candidates = &haystack[..=final_start];
    let repeated_first = u64::from(needle[0]).wrapping_mul(BYTE_ONES);
    let (chunks, remainder) = candidates.as_chunks::<8>();
    let mut base = 0_usize;
    for chunk in chunks {
        let mut bytes = [0_u8; 8];
        bytes.copy_from_slice(chunk);
        let differences = u64::from_ne_bytes(bytes) ^ repeated_first;
        let has_first = differences.wrapping_sub(BYTE_ONES) & !differences & BYTE_HIGHS != 0;
        if has_first
            && chunk.iter().enumerate().any(|(offset, &candidate)| {
                candidate == needle[0] && haystack[base + offset..].starts_with(needle)
            })
        {
            return true;
        }
        base += chunk.len();
    }
    remainder.iter().enumerate().any(|(offset, &candidate)| {
        candidate == needle[0] && haystack[base + offset..].starts_with(needle)
    })
}

pub(crate) fn star_glob_matches(pattern: &[u8], term: &[u8]) -> bool {
    let mut pattern_pos = 0;
    let mut term_pos = 0;
    let mut resume_after_star = None;
    let mut star_match_end = 0;

    while term_pos < term.len() {
        match pattern.get(pattern_pos) {
            Some(pattern_byte) if *pattern_byte == term[term_pos] && *pattern_byte != b'*' => {
                pattern_pos = pattern_pos.saturating_add(1);
                term_pos = term_pos.saturating_add(1);
            }
            Some(&b'*') => {
                while pattern.get(pattern_pos) == Some(&b'*') {
                    pattern_pos = pattern_pos.saturating_add(1);
                }
                resume_after_star = Some(pattern_pos);
                star_match_end = term_pos;
            }
            _ => {
                let Some(resume_pos) = resume_after_star else {
                    return false;
                };
                if star_match_end >= term.len() {
                    return false;
                }
                star_match_end = star_match_end.saturating_add(1);
                term_pos = star_match_end;
                pattern_pos = resume_pos;
            }
        }
    }

    pattern[pattern_pos..].iter().all(|byte| *byte == b'*')
}

/// Reusable exact-lookup buffers.
#[derive(Debug, Default)]
pub struct TermScratch {
    target: Vec<u8>,
    decoded: Vec<u8>,
}

impl TermScratch {
    /// Construct empty reusable lookup scratch.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            target: Vec::new(),
            decoded: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
struct CursorBound {
    key: Vec<u8>,
    inclusive: bool,
}

impl CursorBound {
    const fn included(key: Vec<u8>) -> Self {
        Self {
            key,
            inclusive: true,
        }
    }

    const fn excluded(key: Vec<u8>) -> Self {
        Self {
            key,
            inclusive: false,
        }
    }
}

/// Allocation-reusing ordered TERMDICT cursor.
#[derive(Debug)]
pub struct TermCursor<'dict, 'bytes> {
    dictionary: &'dict TermDictionary<'bytes>,
    block_index: usize,
    next_entry_ordinal: usize,
    next_offset: usize,
    key: Vec<u8>,
    current: Option<TermMatch>,
    upper: Option<CursorBound>,
    field_filter: Option<u16>,
    prefix_filter: Option<Vec<u8>>,
    finished: bool,
}

impl<'dict, 'bytes> TermCursor<'dict, 'bytes> {
    fn new(
        dictionary: &'dict TermDictionary<'bytes>,
        lower: Option<CursorBound>,
        upper: Option<CursorBound>,
        field_filter: Option<u16>,
        prefix_filter: Option<Vec<u8>>,
    ) -> Result<Self, TermDictionaryError> {
        let mut cursor = Self {
            dictionary,
            block_index: 0,
            next_entry_ordinal: 0,
            next_offset: 0,
            key: Vec::new(),
            current: None,
            upper,
            field_filter,
            prefix_filter,
            finished: dictionary.blocks.is_empty(),
        };
        if cursor.finished {
            return Ok(cursor);
        }

        if let Some(lower) = lower {
            let block_index = dictionary.block_for_key(&lower.key).unwrap_or(0);
            let block = &dictionary.blocks[block_index];
            let restart_slice = &dictionary.restarts[block.restart_range.clone()];
            let restart_index = restart_slice
                .partition_point(|restart| {
                    dictionary.bytes[restart.key_range.clone()] <= *lower.key.as_slice()
                })
                .saturating_sub(1);
            let restart = &restart_slice[restart_index];
            cursor.block_index = block_index;
            cursor.next_entry_ordinal = restart.entry_ordinal;
            cursor.next_offset = restart.entry_offset;
            cursor.advance_raw()?;
            while let Some(current) = cursor.current() {
                let ordering = composite_cmp(current.field_ord, current.term, &lower.key);
                if ordering == Ordering::Greater || (ordering == Ordering::Equal && lower.inclusive)
                {
                    break;
                }
                cursor.advance_raw()?;
            }
        } else {
            cursor.start_block(0)?;
            cursor.advance_raw()?;
        }
        cursor.enforce_filters()?;
        Ok(cursor)
    }

    /// Current term, or `None` after the cursor is fused.
    #[must_use]
    pub fn current(&self) -> Option<TermRef<'_>> {
        let current = self.current?;
        let key = self.key.as_slice();
        let field_bytes = key.get(..2)?;
        let field_ord = u16::from_be_bytes([field_bytes[0], field_bytes[1]]);
        Some(TermRef {
            term_ord: current.term_ord,
            field_ord,
            term: key.get(2..).unwrap_or_default(),
            metadata: current.metadata,
        })
    }

    /// Zero-based prefix-compressed block containing the current term.
    ///
    /// Returns `None` after exhaustion. Query execution uses this only for
    /// coarse cancellation and deterministic fuel checkpoints.
    #[must_use]
    pub fn current_block_index(&self) -> Option<usize> {
        self.current.map(|_| self.block_index)
    }

    /// Advance once in composite-key order.
    ///
    /// # Errors
    ///
    /// Returns a typed error if validated bytes cannot be decoded or scratch
    /// growth fails.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<Option<TermRef<'_>>, TermDictionaryError> {
        if self.current.is_none() {
            return Ok(None);
        }
        self.advance_raw()?;
        self.enforce_filters()?;
        Ok(self.current())
    }

    /// Collect the remaining range under an explicit result cap.
    ///
    /// # Errors
    ///
    /// Returns a typed error on limit exhaustion, allocation failure, or
    /// validated-decode failure.
    pub fn collect_bounded(mut self, limit: usize) -> Result<Vec<OwnedTerm>, TermDictionaryError> {
        let mut output = Vec::new();
        while let Some(current) = self.current() {
            if output.len() >= limit {
                return Err(TermDictionaryError::MaterializationLimitExceeded { limit });
            }
            output
                .try_reserve(1)
                .map_err(|_| TermDictionaryError::Allocation {
                    context: "materialized term rows",
                    count: output.len().saturating_add(1),
                })?;
            output.push(current.to_owned()?);
            self.next()?;
        }
        Ok(output)
    }

    fn start_block(&mut self, block_index: usize) -> Result<(), TermDictionaryError> {
        let Some(block) = self.dictionary.blocks.get(block_index) else {
            self.current = None;
            self.finished = true;
            return Ok(());
        };
        self.block_index = block_index;
        self.next_entry_ordinal = 0;
        self.next_offset = block
            .byte_range
            .start
            .checked_add(BLOCK_ENTRY_COUNT_BYTES)
            .ok_or(TermDictionaryError::SizeOverflow {
                field: "block entry start",
            })?;
        self.key.clear();
        Ok(())
    }

    fn advance_raw(&mut self) -> Result<(), TermDictionaryError> {
        if self.finished {
            self.current = None;
            return Ok(());
        }
        loop {
            let block = &self.dictionary.blocks[self.block_index];
            if self.next_entry_ordinal >= block.entry_count {
                self.start_block(self.block_index.saturating_add(1))?;
                if self.finished {
                    return Ok(());
                }
                continue;
            }
            let entry_ordinal = self.next_entry_ordinal;
            let mut reader = SliceReader::with_range(
                self.dictionary.bytes,
                self.next_offset,
                block.byte_range.end,
            )?;
            let term_ordinal = block
                .base_term_ord
                .checked_add(u32::try_from(entry_ordinal).map_err(|_| {
                    TermDictionaryError::ValueOutOfRange {
                        field: "term ordinal",
                        value: u64::try_from(entry_ordinal).unwrap_or(u64::MAX),
                        offset: self.next_offset,
                    }
                })?)
                .ok_or(TermDictionaryError::SizeOverflow {
                    field: "cursor term ordinal",
                })?;
            let decoded = decode_entry(
                &mut reader,
                entry_ordinal,
                &mut self.key,
                self.dictionary.schema,
                self.dictionary.sections,
                usize::try_from(term_ordinal).unwrap_or(usize::MAX),
            )?;
            self.next_offset = reader.position();
            self.next_entry_ordinal = self.next_entry_ordinal.saturating_add(1);
            self.current = Some(TermMatch {
                term_ord: term_ordinal,
                metadata: decoded.metadata,
            });
            return Ok(());
        }
    }

    fn enforce_filters(&mut self) -> Result<(), TermDictionaryError> {
        loop {
            let Some(current) = self.current() else {
                return Ok(());
            };
            if let Some(field_ord) = self.field_filter {
                match current.field_ord.cmp(&field_ord) {
                    Ordering::Less => {
                        self.advance_raw()?;
                        continue;
                    }
                    Ordering::Greater => {
                        self.current = None;
                        self.finished = true;
                        return Ok(());
                    }
                    Ordering::Equal => {}
                }
            }
            if let Some(prefix) = self.prefix_filter.as_ref() {
                if !self.key.starts_with(prefix) {
                    self.current = None;
                    self.finished = true;
                    return Ok(());
                }
            }
            if let Some(upper) = self.upper.as_ref() {
                let ordering = self.key.as_slice().cmp(&upper.key);
                if ordering == Ordering::Greater
                    || (ordering == Ordering::Equal && !upper.inclusive)
                {
                    self.current = None;
                    self.finished = true;
                }
            }
            return Ok(());
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct DecodedEntry {
    field_ord: u16,
    metadata: TermMetadata,
    full_key_range: Option<RangeMarker>,
}

#[derive(Clone, Copy, Debug)]
struct RangeMarker {
    start: usize,
    end: usize,
}

impl RangeMarker {
    const fn into_range(self) -> Range<usize> {
        self.start..self.end
    }
}

fn validate_block(
    bytes: &[u8],
    schema: SchemaDescriptor,
    sections: TermSectionLengths,
    limits: TermDictionaryLimits,
    block_index: usize,
    byte_range: Range<usize>,
    expected_first_key: Range<usize>,
    base_term_ord: u32,
    term_count: &mut usize,
    restarts: &mut Vec<RestartMeta>,
    references: &mut ReferenceValidator,
    block_key: &mut Vec<u8>,
    previous_key: &mut Vec<u8>,
    previous_tail: Option<&BlockTail>,
) -> Result<(BlockMeta, BlockTail), TermDictionaryError> {
    let block_len = byte_range.end.saturating_sub(byte_range.start);
    let mut reader = SliceReader::with_range(bytes, byte_range.start, byte_range.end)?;
    let entry_count = usize::from(reader.read_u16()?);
    if entry_count == 0 {
        return Err(TermDictionaryError::EmptyBlock { block_index });
    }
    if entry_count
        > block_len
            .saturating_sub(BLOCK_ENTRY_COUNT_BYTES)
            .saturating_div(MIN_WIRE_BYTES_PER_ENTRY)
    {
        return Err(TermDictionaryError::ImplausibleEntryCount {
            block_index,
            entry_count,
            byte_len: block_len,
        });
    }
    if entry_count > 1 && block_len > TERM_BLOCK_TARGET_BYTES {
        return Err(TermDictionaryError::OversizedBlock {
            block_index,
            entry_count,
            byte_len: block_len,
        });
    }
    let next_term_count =
        term_count
            .checked_add(entry_count)
            .ok_or(TermDictionaryError::SizeOverflow {
                field: "validated term count",
            })?;
    if next_term_count > limits.max_terms {
        return Err(TermDictionaryError::TermBudgetExceeded {
            limit: limits.max_terms,
            actual: next_term_count,
        });
    }
    let block_restarts = entry_count.div_ceil(TERM_RESTART_INTERVAL);
    let next_restart_count =
        restarts
            .len()
            .checked_add(block_restarts)
            .ok_or(TermDictionaryError::SizeOverflow {
                field: "restart count",
            })?;
    if next_restart_count > limits.max_restarts {
        return Err(TermDictionaryError::RestartBudgetExceeded {
            limit: limits.max_restarts,
            actual: next_restart_count,
        });
    }
    restarts
        .try_reserve(block_restarts)
        .map_err(|_| TermDictionaryError::Allocation {
            context: "restart metadata",
            count: next_restart_count,
        })?;

    let restart_start = restarts.len();
    block_key.clear();
    for entry_ordinal in 0..entry_count {
        let entry_offset = reader.position();
        let term_ordinal = usize::try_from(base_term_ord)
            .unwrap_or(usize::MAX)
            .saturating_add(entry_ordinal);
        let decoded = decode_entry(
            &mut reader,
            entry_ordinal,
            block_key,
            schema,
            sections,
            term_ordinal,
        )?;
        if entry_ordinal == 0 {
            if block_key.as_slice() != &bytes[expected_first_key.clone()] {
                return Err(TermDictionaryError::IndexKeyMismatch { block_index });
            }
            if let Some(tail) = previous_tail {
                let appended = encoded_entry_len(
                    block_key,
                    previous_key,
                    tail.entry_count,
                    decoded.metadata,
                    field_has_positions(schema, decoded.field_ord)?,
                )?;
                if tail.byte_len <= TERM_BLOCK_TARGET_BYTES
                    && tail.byte_len.checked_add(appended).ok_or(
                        TermDictionaryError::SizeOverflow {
                            field: "canonical block split",
                        },
                    )? <= TERM_BLOCK_TARGET_BYTES
                {
                    return Err(TermDictionaryError::PrematureBlockSplit {
                        block_index: block_index.saturating_sub(1),
                    });
                }
            }
        }
        if !previous_key.is_empty() && block_key.as_slice() <= previous_key.as_slice() {
            return Err(TermDictionaryError::NonAscendingKey { term_ordinal });
        }
        if let Some(marker) = decoded.full_key_range {
            restarts.push(RestartMeta {
                entry_ordinal,
                term_ordinal: u32::try_from(term_ordinal).map_err(|_| {
                    TermDictionaryError::ValueOutOfRange {
                        field: "term ordinal",
                        value: u64::try_from(term_ordinal).unwrap_or(u64::MAX),
                        offset: entry_offset,
                    }
                })?,
                entry_offset,
                key_range: marker.into_range(),
            });
        }
        references.push(decoded.metadata)?;
        copy_bytes(previous_key, block_key, "validated previous key")?;
    }
    let remaining = byte_range.end.saturating_sub(reader.position());
    if remaining != 0 {
        return Err(TermDictionaryError::TrailingBlockBytes {
            block_index,
            remaining,
        });
    }
    *term_count = next_term_count;
    Ok((
        BlockMeta {
            first_key_range: expected_first_key,
            byte_range,
            entry_count,
            base_term_ord,
            restart_range: restart_start..restarts.len(),
        },
        BlockTail {
            byte_len: block_len,
            entry_count,
        },
    ))
}

fn decode_entry(
    reader: &mut SliceReader<'_>,
    entry_ordinal: usize,
    key: &mut Vec<u8>,
    schema: SchemaDescriptor,
    sections: TermSectionLengths,
    term_ordinal: usize,
) -> Result<DecodedEntry, TermDictionaryError> {
    let entry_offset = reader.position();
    let full_key_range = if entry_ordinal.is_multiple_of(TERM_RESTART_INTERVAL) {
        let key_len_offset = reader.position();
        let key_len = usize_from_vint(reader.read_vint()?, "key_len", key_len_offset)?;
        if !(2..=MAX_COMPOSITE_KEY_BYTES).contains(&key_len) {
            return Err(TermDictionaryError::TermTooLong {
                term_ordinal,
                length: key_len.saturating_sub(2),
            });
        }
        let range = reader.take_range(key_len)?;
        copy_bytes(key, &reader.bytes[range.clone()], "restart key")?;
        Some(RangeMarker {
            start: range.start,
            end: range.end,
        })
    } else {
        let shared_offset = reader.position();
        let shared = usize_from_vint(reader.read_vint()?, "shared_prefix_len", shared_offset)?;
        let suffix_offset = reader.position();
        let suffix_len = usize_from_vint(reader.read_vint()?, "suffix_len", suffix_offset)?;
        if shared > key.len() {
            return Err(TermDictionaryError::InvalidPrefix {
                shared,
                previous_len: key.len(),
                offset: entry_offset,
            });
        }
        let suffix = reader.take(suffix_len)?;
        if shared < key.len() && suffix.first().is_some_and(|byte| *byte == key[shared]) {
            return Err(TermDictionaryError::NonCanonicalPrefix {
                offset: entry_offset,
            });
        }
        let new_len = shared
            .checked_add(suffix_len)
            .ok_or(TermDictionaryError::SizeOverflow {
                field: "reconstructed key length",
            })?;
        if !(2..=MAX_COMPOSITE_KEY_BYTES).contains(&new_len) {
            return Err(TermDictionaryError::TermTooLong {
                term_ordinal,
                length: new_len.saturating_sub(2),
            });
        }
        if new_len > key.capacity() {
            key.try_reserve(new_len.saturating_sub(key.len()))
                .map_err(|_| TermDictionaryError::Allocation {
                    context: "reconstructed key",
                    count: new_len,
                })?;
        }
        key.truncate(shared);
        key.extend_from_slice(suffix);
        None
    };

    let has_positions = validate_composite_key(schema, key, term_ordinal)?;
    let field_ord = u16::from_be_bytes([key[0], key[1]]);
    let doc_freq_offset = reader.position();
    let doc_freq_raw = reader.read_vint()?;
    let doc_freq =
        u32::try_from(doc_freq_raw).map_err(|_| TermDictionaryError::ValueOutOfRange {
            field: "doc_freq",
            value: doc_freq_raw,
            offset: doc_freq_offset,
        })?;
    let postings = ByteSpan::new(reader.read_vint()?, reader.read_vint()?);
    let positions = if has_positions {
        Some(ByteSpan::new(reader.read_vint()?, reader.read_vint()?))
    } else {
        None
    };
    let blockmax = ByteSpan::new(reader.read_vint()?, reader.read_vint()?);
    let metadata = TermMetadata {
        doc_freq,
        postings,
        positions,
        blockmax,
    };
    validate_metadata_basic(field_ord, has_positions, metadata, sections, term_ordinal)?;
    Ok(DecodedEntry {
        field_ord,
        metadata,
        full_key_range,
    })
}

#[derive(Debug)]
struct ReferenceValidator {
    sections: TermSectionLengths,
    postings_end: u64,
    positions_end: u64,
    blockmax_end: u64,
}

impl ReferenceValidator {
    fn new(
        schema: SchemaDescriptor,
        sections: TermSectionLengths,
    ) -> Result<Self, TermDictionaryError> {
        validate_positions_section(schema, sections)?;
        Ok(Self {
            sections,
            postings_end: 0,
            positions_end: 0,
            blockmax_end: 0,
        })
    }

    fn push(&mut self, metadata: TermMetadata) -> Result<(), TermDictionaryError> {
        if metadata.postings.offset != self.postings_end {
            return Err(TermDictionaryError::NonContiguousReference {
                section: "POSTINGS",
                expected: self.postings_end,
                actual: metadata.postings.offset,
            });
        }
        self.postings_end = metadata.postings.end("POSTINGS")?;
        if let Some(positions) = metadata.positions {
            if positions.offset != self.positions_end {
                return Err(TermDictionaryError::NonContiguousReference {
                    section: "POSITIONS",
                    expected: self.positions_end,
                    actual: positions.offset,
                });
            }
            self.positions_end = positions.end("POSITIONS")?;
        }
        if metadata.blockmax.offset != self.blockmax_end {
            return Err(TermDictionaryError::NonContiguousReference {
                section: "BLOCKMAX",
                expected: self.blockmax_end,
                actual: metadata.blockmax.offset,
            });
        }
        self.blockmax_end = metadata.blockmax.end("BLOCKMAX")?;
        Ok(())
    }

    fn finish(self) -> Result<(), TermDictionaryError> {
        validate_final_length("POSTINGS", self.sections.postings, self.postings_end)?;
        validate_final_length("BLOCKMAX", self.sections.blockmax, self.blockmax_end)?;
        if let Some(expected) = self.sections.positions {
            validate_final_length("POSITIONS", expected, self.positions_end)?;
        }
        Ok(())
    }
}

fn validate_final_length(
    section: &'static str,
    expected: u64,
    actual: u64,
) -> Result<(), TermDictionaryError> {
    if expected == actual {
        Ok(())
    } else {
        Err(TermDictionaryError::SectionLengthMismatch {
            section,
            expected,
            actual,
        })
    }
}

fn validate_metadata_basic(
    field_ord: u16,
    has_positions: bool,
    metadata: TermMetadata,
    sections: TermSectionLengths,
    term_ordinal: usize,
) -> Result<(), TermDictionaryError> {
    if metadata.doc_freq == 0 {
        return Err(TermDictionaryError::ZeroDocFrequency { term_ordinal });
    }
    validate_span(
        "POSTINGS",
        metadata.postings,
        sections.postings,
        term_ordinal,
    )?;
    if metadata.positions.is_some() != has_positions {
        return Err(TermDictionaryError::PositionsPresenceMismatch {
            field_ord,
            expected: has_positions,
            actual: metadata.positions.is_some(),
        });
    }
    if let Some(positions) = metadata.positions {
        let limit = sections
            .positions
            .ok_or(TermDictionaryError::PositionsSectionMismatch {
                expected: true,
                actual: false,
            })?;
        validate_span("POSITIONS", positions, limit, term_ordinal)?;
    }
    validate_span(
        "BLOCKMAX",
        metadata.blockmax,
        sections.blockmax,
        term_ordinal,
    )
}

fn validate_span(
    section: &'static str,
    span: ByteSpan,
    limit: u64,
    term_ordinal: usize,
) -> Result<(), TermDictionaryError> {
    if span.len == 0 {
        return Err(TermDictionaryError::EmptyReference {
            section,
            term_ordinal,
        });
    }
    let end = span.end(section)?;
    if end > limit {
        return Err(TermDictionaryError::ReferenceOutOfBounds {
            section,
            end,
            limit,
        });
    }
    Ok(())
}

fn validate_schema(schema: SchemaDescriptor) -> Result<(), TermDictionaryError> {
    schema
        .validate()
        .map_err(|error| TermDictionaryError::InvalidSchema {
            detail: error.to_string(),
        })
}

fn validate_positions_section(
    schema: SchemaDescriptor,
    sections: TermSectionLengths,
) -> Result<(), TermDictionaryError> {
    let expected = schema.fields.iter().any(|field| {
        matches!(
            field.kind,
            FieldKind::Text {
                positions: true,
                ..
            }
        )
    });
    let actual = sections.positions.is_some();
    if expected == actual {
        Ok(())
    } else {
        Err(TermDictionaryError::PositionsSectionMismatch { expected, actual })
    }
}

fn field_has_positions(
    schema: SchemaDescriptor,
    field_ord: u16,
) -> Result<bool, TermDictionaryError> {
    let field = schema
        .fields
        .get(usize::from(field_ord))
        .ok_or(TermDictionaryError::UnknownField { field_ord })?;
    if field.id != field_ord {
        return Err(TermDictionaryError::UnknownField { field_ord });
    }
    match field.kind {
        FieldKind::Keyword => Ok(false),
        FieldKind::Text { positions, .. } => Ok(positions),
        FieldKind::StoredOnly | FieldKind::I64 { .. } | FieldKind::U64 { .. } => {
            Err(TermDictionaryError::NonTermField { field_ord })
        }
    }
}

fn validate_composite_key(
    schema: SchemaDescriptor,
    key: &[u8],
    term_ordinal: usize,
) -> Result<bool, TermDictionaryError> {
    if key.len() < 2 {
        return Err(TermDictionaryError::Truncated {
            offset: 0,
            needed: 2,
            remaining: key.len(),
        });
    }
    let term_len = key.len().saturating_sub(2);
    if term_len > MAX_TERM_BYTES {
        return Err(TermDictionaryError::TermTooLong {
            term_ordinal,
            length: term_len,
        });
    }
    field_has_positions(schema, u16::from_be_bytes([key[0], key[1]]))
}

pub(crate) fn validate_query_term(
    schema: SchemaDescriptor,
    field_ord: u16,
    term: &[u8],
) -> Result<(), TermDictionaryError> {
    field_has_positions(schema, field_ord)?;
    if term.len() > MAX_TERM_BYTES {
        return Err(TermDictionaryError::TermTooLong {
            term_ordinal: 0,
            length: term.len(),
        });
    }
    Ok(())
}

pub(crate) fn validate_bound_term(
    schema: SchemaDescriptor,
    field_ord: u16,
    bound: &Bound<&[u8]>,
) -> Result<(), TermDictionaryError> {
    match bound {
        Bound::Included(term) | Bound::Excluded(term) => {
            validate_query_term(schema, field_ord, term)
        }
        Bound::Unbounded => Ok(()),
    }
}

fn range_is_reversed(start: &Bound<&[u8]>, end: &Bound<&[u8]>) -> bool {
    match (start, end) {
        (
            Bound::Included(start) | Bound::Excluded(start),
            Bound::Included(end) | Bound::Excluded(end),
        ) => start > end,
        _ => false,
    }
}

fn composite_key(
    field_ord: u16,
    term: &[u8],
    term_ordinal: usize,
) -> Result<Vec<u8>, TermDictionaryError> {
    if term.len() > MAX_TERM_BYTES {
        return Err(TermDictionaryError::TermTooLong {
            term_ordinal,
            length: term.len(),
        });
    }
    let mut key = Vec::new();
    build_composite_in(&mut key, field_ord, term, "composite key")?;
    Ok(key)
}

fn build_composite_in(
    output: &mut Vec<u8>,
    field_ord: u16,
    term: &[u8],
    context: &'static str,
) -> Result<(), TermDictionaryError> {
    let length = term
        .len()
        .checked_add(2)
        .ok_or(TermDictionaryError::SizeOverflow {
            field: "composite key length",
        })?;
    output.clear();
    output
        .try_reserve(length)
        .map_err(|_| TermDictionaryError::Allocation {
            context,
            count: length,
        })?;
    output.extend_from_slice(&field_ord.to_be_bytes());
    output.extend_from_slice(term);
    Ok(())
}

fn composite_cmp(field_ord: u16, term: &[u8], composite: &[u8]) -> Ordering {
    let field = field_ord.to_be_bytes();
    match field.as_slice().cmp(composite.get(..2).unwrap_or_default()) {
        Ordering::Equal => term.cmp(composite.get(2..).unwrap_or_default()),
        ordering => ordering,
    }
}

fn new_block_buffer(max_bytes: usize) -> Result<Vec<u8>, TermDictionaryError> {
    let mut bytes = Vec::new();
    let reserve = TERM_BLOCK_TARGET_BYTES.min(max_bytes.saturating_sub(BLOCK_COUNT_BYTES));
    bytes
        .try_reserve(reserve)
        .map_err(|_| TermDictionaryError::Allocation {
            context: "TERMDICT block",
            count: reserve,
        })?;
    bytes.extend_from_slice(&0_u16.to_le_bytes());
    Ok(bytes)
}

fn finish_encoded_block(
    blocks: &mut Vec<EncodedBlock>,
    current: &mut Vec<u8>,
    current_first: &mut Option<Vec<u8>>,
    entry_count: usize,
    limits: TermDictionaryLimits,
    completed_block_bytes: &mut usize,
    restart_count: &mut usize,
) -> Result<(), TermDictionaryError> {
    if blocks.len() >= limits.max_blocks {
        return Err(TermDictionaryError::BlockBudgetExceeded {
            limit: limits.max_blocks,
            actual: blocks.len().saturating_add(1),
        });
    }
    let next_restart_count = restart_count
        .checked_add(entry_count.div_ceil(TERM_RESTART_INTERVAL))
        .ok_or(TermDictionaryError::SizeOverflow {
            field: "restart count",
        })?;
    if next_restart_count > limits.max_restarts {
        return Err(TermDictionaryError::RestartBudgetExceeded {
            limit: limits.max_restarts,
            actual: next_restart_count,
        });
    }
    let next_completed_bytes = completed_block_bytes.checked_add(current.len()).ok_or(
        TermDictionaryError::SizeOverflow {
            field: "encoded blocks bytes",
        },
    )?;
    let minimum_output_len = checked_add_size(
        BLOCK_COUNT_BYTES,
        next_completed_bytes,
        "minimum TERMDICT length",
    )?;
    if minimum_output_len > limits.max_bytes {
        return Err(TermDictionaryError::ByteBudgetExceeded {
            limit: limits.max_bytes,
            actual: minimum_output_len,
        });
    }
    let count = u16::try_from(entry_count).map_err(|_| TermDictionaryError::ValueOutOfRange {
        field: "block entry_count",
        value: u64::try_from(entry_count).unwrap_or(u64::MAX),
        offset: 0,
    })?;
    current[..2].copy_from_slice(&count.to_le_bytes());
    blocks
        .try_reserve(1)
        .map_err(|_| TermDictionaryError::Allocation {
            context: "encoded block list",
            count: blocks.len().saturating_add(1),
        })?;
    blocks.push(EncodedBlock {
        first_key: current_first
            .take()
            .ok_or(TermDictionaryError::EmptyBlock {
                block_index: blocks.len(),
            })?,
        bytes: std::mem::take(current),
        entry_count,
    });
    *completed_block_bytes = next_completed_bytes;
    *restart_count = next_restart_count;
    Ok(())
}

fn encode_entry(
    key: &[u8],
    previous: &[u8],
    entry_ordinal: usize,
    metadata: TermMetadata,
    has_positions: bool,
) -> Result<Vec<u8>, TermDictionaryError> {
    let length = encoded_entry_len(key, previous, entry_ordinal, metadata, has_positions)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(length)
        .map_err(|_| TermDictionaryError::Allocation {
            context: "encoded term entry",
            count: length,
        })?;
    if entry_ordinal.is_multiple_of(TERM_RESTART_INTERVAL) {
        write_vint(
            u64::try_from(key.len()).map_err(|_| TermDictionaryError::SizeOverflow {
                field: "restart key length",
            })?,
            &mut output,
        );
        output.extend_from_slice(key);
    } else {
        let shared = longest_shared_prefix(previous, key);
        let suffix = &key[shared..];
        write_vint(
            u64::try_from(shared).map_err(|_| TermDictionaryError::SizeOverflow {
                field: "shared prefix length",
            })?,
            &mut output,
        );
        write_vint(
            u64::try_from(suffix.len()).map_err(|_| TermDictionaryError::SizeOverflow {
                field: "suffix length",
            })?,
            &mut output,
        );
        output.extend_from_slice(suffix);
    }
    write_vint(u64::from(metadata.doc_freq), &mut output);
    write_vint(metadata.postings.offset, &mut output);
    write_vint(metadata.postings.len, &mut output);
    if has_positions {
        let positions =
            metadata
                .positions
                .ok_or_else(|| TermDictionaryError::PositionsPresenceMismatch {
                    field_ord: u16::from_be_bytes([key[0], key[1]]),
                    expected: true,
                    actual: false,
                })?;
        write_vint(positions.offset, &mut output);
        write_vint(positions.len, &mut output);
    }
    write_vint(metadata.blockmax.offset, &mut output);
    write_vint(metadata.blockmax.len, &mut output);
    debug_assert_eq!(output.len(), length);
    Ok(output)
}

fn encoded_entry_len(
    key: &[u8],
    previous: &[u8],
    entry_ordinal: usize,
    metadata: TermMetadata,
    has_positions: bool,
) -> Result<usize, TermDictionaryError> {
    let key_len = if entry_ordinal.is_multiple_of(TERM_RESTART_INTERVAL) {
        checked_add_size(
            vint_len(
                u64::try_from(key.len()).map_err(|_| TermDictionaryError::SizeOverflow {
                    field: "restart key length",
                })?,
            ),
            key.len(),
            "full key encoding",
        )?
    } else {
        let shared = longest_shared_prefix(previous, key);
        let suffix_len = key.len().saturating_sub(shared);
        let prefix_len =
            vint_len(
                u64::try_from(shared).map_err(|_| TermDictionaryError::SizeOverflow {
                    field: "shared prefix length",
                })?,
            );
        checked_add_size(
            checked_add_size(
                prefix_len,
                vint_len(u64::try_from(suffix_len).map_err(|_| {
                    TermDictionaryError::SizeOverflow {
                        field: "suffix length",
                    }
                })?),
                "compressed key lengths",
            )?,
            suffix_len,
            "compressed key bytes",
        )?
    };
    let mut length = key_len;
    for value in [
        u64::from(metadata.doc_freq),
        metadata.postings.offset,
        metadata.postings.len,
    ] {
        length = checked_add_size(length, vint_len(value), "term payload")?;
    }
    if has_positions {
        let positions =
            metadata
                .positions
                .ok_or_else(|| TermDictionaryError::PositionsPresenceMismatch {
                    field_ord: u16::from_be_bytes([key[0], key[1]]),
                    expected: true,
                    actual: false,
                })?;
        length = checked_add_size(length, vint_len(positions.offset), "positions offset")?;
        length = checked_add_size(length, vint_len(positions.len), "positions length")?;
    }
    length = checked_add_size(
        length,
        vint_len(metadata.blockmax.offset),
        "blockmax offset",
    )?;
    checked_add_size(length, vint_len(metadata.blockmax.len), "blockmax length")
}

fn longest_shared_prefix(left: &[u8], right: &[u8]) -> usize {
    left.iter()
        .zip(right)
        .position(|(left, right)| left != right)
        .unwrap_or_else(|| left.len().min(right.len()))
}

fn append_bytes(
    output: &mut Vec<u8>,
    bytes: &[u8],
    context: &'static str,
) -> Result<(), TermDictionaryError> {
    output
        .try_reserve(bytes.len())
        .map_err(|_| TermDictionaryError::Allocation {
            context,
            count: bytes.len(),
        })?;
    output.extend_from_slice(bytes);
    Ok(())
}

fn clone_bytes(bytes: &[u8], context: &'static str) -> Result<Vec<u8>, TermDictionaryError> {
    let mut output = Vec::new();
    copy_bytes(&mut output, bytes, context)?;
    Ok(output)
}

fn copy_bytes(
    output: &mut Vec<u8>,
    bytes: &[u8],
    context: &'static str,
) -> Result<(), TermDictionaryError> {
    output.clear();
    output
        .try_reserve(bytes.len())
        .map_err(|_| TermDictionaryError::Allocation {
            context,
            count: bytes.len(),
        })?;
    output.extend_from_slice(bytes);
    Ok(())
}

fn checked_add_size(
    left: usize,
    right: usize,
    field: &'static str,
) -> Result<usize, TermDictionaryError> {
    left.checked_add(right)
        .ok_or(TermDictionaryError::SizeOverflow { field })
}

fn usize_from_vint(
    value: u64,
    field: &'static str,
    offset: usize,
) -> Result<usize, TermDictionaryError> {
    usize::try_from(value).map_err(|_| TermDictionaryError::ValueOutOfRange {
        field,
        value,
        offset,
    })
}

fn vint_len(mut value: u64) -> usize {
    let mut length = 1;
    while value >= 0x80 {
        value >>= 7;
        length += 1;
    }
    length
}

#[allow(clippy::cast_possible_truncation)]
fn write_vint(mut value: u64, output: &mut Vec<u8>) {
    while value >= 0x80 {
        output.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    output.push(value as u8);
}

struct SliceReader<'a> {
    bytes: &'a [u8],
    position: usize,
    end: usize,
}

impl<'a> SliceReader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            position: 0,
            end: bytes.len(),
        }
    }

    fn with_range(bytes: &'a [u8], start: usize, end: usize) -> Result<Self, TermDictionaryError> {
        if start > end || end > bytes.len() {
            return Err(TermDictionaryError::Truncated {
                offset: start.min(bytes.len()),
                needed: end.saturating_sub(start),
                remaining: bytes.len().saturating_sub(start.min(bytes.len())),
            });
        }
        Ok(Self {
            bytes,
            position: start,
            end,
        })
    }

    const fn position(&self) -> usize {
        self.position
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], TermDictionaryError> {
        let range = self.take_range(length)?;
        Ok(&self.bytes[range])
    }

    fn take_range(&mut self, length: usize) -> Result<Range<usize>, TermDictionaryError> {
        let remaining = self.end.saturating_sub(self.position);
        let end = self
            .position
            .checked_add(length)
            .ok_or(TermDictionaryError::SizeOverflow {
                field: "reader cursor",
            })?;
        if end > self.end {
            return Err(TermDictionaryError::Truncated {
                offset: self.position,
                needed: length,
                remaining,
            });
        }
        let range = self.position..end;
        self.position = end;
        Ok(range)
    }

    fn read_u8(&mut self) -> Result<u8, TermDictionaryError> {
        let offset = self.position;
        self.take(1)?
            .first()
            .copied()
            .ok_or(TermDictionaryError::Truncated {
                offset,
                needed: 1,
                remaining: 0,
            })
    }

    fn read_u16(&mut self) -> Result<u16, TermDictionaryError> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_u32(&mut self) -> Result<u32, TermDictionaryError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    // Profile-directed (bd-b6tc): `read_vint` was 16% of query self-time and
    // its hot instructions were the *function prologue* — it was never inlined,
    // so every call in the tight `decode_entry` loops paid a full 6-register
    // spill. The 1-byte case (value < 128) dominates gap-encoded
    // lengths/offsets and is always canonical, so it is inlined into callers
    // here while the rare multi-byte loop is kept out-of-line and `#[cold]` to
    // avoid code-size bloat. Byte-identical to the former single function.
    #[inline]
    fn read_vint(&mut self) -> Result<u64, TermDictionaryError> {
        if self.position < self.end
            && let Some(&byte) = self.bytes.get(self.position)
            && byte & 0x80 == 0
        {
            self.position += 1;
            return Ok(u64::from(byte));
        }
        self.read_vint_multibyte()
    }

    #[cold]
    #[inline(never)]
    fn read_vint_multibyte(&mut self) -> Result<u64, TermDictionaryError> {
        let start = self.position;
        let mut value = 0_u64;
        for byte_index in 0..10 {
            let byte = self.read_u8()?;
            if byte_index == 9 && byte & 0xfe != 0 {
                return Err(TermDictionaryError::VintOverflow { offset: start });
            }
            value |= u64::from(byte & 0x7f) << (byte_index * 7);
            if byte & 0x80 == 0 {
                if vint_len(value) != byte_index + 1 {
                    return Err(TermDictionaryError::NonCanonicalVint { offset: start });
                }
                return Ok(value);
            }
        }
        Err(TermDictionaryError::VintOverflow { offset: start })
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::ops::Bound;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use crate::schema::{Analyzer, FieldDescriptor};

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

    const KEYWORD_FIELDS: [FieldDescriptor; 1] = [FieldDescriptor {
        id: 0,
        name: "keyword",
        kind: FieldKind::Keyword,
        stored: false,
    }];
    const KEYWORD_SCHEMA: SchemaDescriptor = SchemaDescriptor {
        name: "grimoire-keyword-tests",
        fields: &KEYWORD_FIELDS,
    };

    const MIXED_FIELDS: [FieldDescriptor; 3] = [
        FieldDescriptor {
            id: 0,
            name: "keyword",
            kind: FieldKind::Keyword,
            stored: false,
        },
        FieldDescriptor {
            id: 1,
            name: "text_without_positions",
            kind: FieldKind::Text {
                analyzer: Analyzer::FrankensearchDefault,
                positions: false,
            },
            stored: false,
        },
        FieldDescriptor {
            id: 2,
            name: "text_with_positions",
            kind: FieldKind::Text {
                analyzer: Analyzer::FrankensearchDefault,
                positions: true,
            },
            stored: false,
        },
    ];
    const MIXED_SCHEMA: SchemaDescriptor = SchemaDescriptor {
        name: "grimoire-mixed-tests",
        fields: &MIXED_FIELDS,
    };

    fn fixture_inputs(
        schema: SchemaDescriptor,
        keys: &[(u16, Vec<u8>)],
    ) -> (Vec<TermInput<'_>>, TermSectionLengths) {
        let positions_present = schema.fields.iter().any(|field| {
            matches!(
                field.kind,
                FieldKind::Text {
                    positions: true,
                    ..
                }
            )
        });
        let mut postings_offset = 0_u64;
        let mut positions_offset = 0_u64;
        let mut blockmax_offset = 0_u64;
        let mut inputs = Vec::with_capacity(keys.len());
        for (index, (field_ord, term)) in keys.iter().enumerate() {
            let postings = ByteSpan::new(postings_offset, 1);
            postings_offset += 1;
            let blockmax = ByteSpan::new(blockmax_offset, 1);
            blockmax_offset += 1;
            let doc_freq = u32::try_from(index % 7 + 1).expect("bounded fixture doc_freq");
            let metadata = if field_has_positions(schema, *field_ord)
                .expect("fixture uses a term-indexed field")
            {
                let positions = ByteSpan::new(positions_offset, 1);
                positions_offset += 1;
                TermMetadata::with_positions(doc_freq, postings, positions, blockmax)
            } else {
                TermMetadata::without_positions(doc_freq, postings, blockmax)
            };
            inputs.push(TermInput::new(*field_ord, term, metadata));
        }
        (
            inputs,
            TermSectionLengths {
                postings: postings_offset,
                positions: positions_present.then_some(positions_offset),
                blockmax: blockmax_offset,
            },
        )
    }

    fn encode_fixture(
        schema: SchemaDescriptor,
        keys: &[(u16, Vec<u8>)],
    ) -> Result<
        (
            EncodedTermDictionary,
            Vec<TermInput<'_>>,
            TermSectionLengths,
        ),
        TermDictionaryError,
    > {
        let (inputs, sections) = fixture_inputs(schema, keys);
        let encoded = EncodedTermDictionary::encode_sorted(schema, sections, &inputs)?;
        Ok((encoded, inputs, sections))
    }

    fn terms_from_cursor(
        cursor: TermCursor<'_, '_>,
    ) -> Result<Vec<OwnedTerm>, TermDictionaryError> {
        cursor.collect_bounded(usize::MAX)
    }

    fn sorted_numbered_keys(count: usize) -> Vec<(u16, Vec<u8>)> {
        (0..count)
            .map(|index| (0, format!("term-{index:05}").into_bytes()))
            .collect()
    }

    fn expanded_glob_terms(
        dictionary: &TermDictionary<'_>,
        field_ord: u16,
        pattern: &[u8],
        limit: usize,
    ) -> Result<Vec<Vec<u8>>, TermDictionaryError> {
        Ok(dictionary
            .expand_glob(field_ord, pattern, limit)?
            .into_iter()
            .map(|row| row.term)
            .collect())
    }

    fn brute_force_glob_matches(pattern: &[u8], term: &[u8]) -> bool {
        let Some((&head, tail)) = pattern.split_first() else {
            return term.is_empty();
        };
        if head == b'*' {
            let mut remaining_pattern = tail;
            while remaining_pattern.first() == Some(&b'*') {
                remaining_pattern = &remaining_pattern[1..];
            }
            return (0..=term.len()).any(|matched_bytes| {
                brute_force_glob_matches(remaining_pattern, &term[matched_bytes..])
            });
        }
        let Some((&term_head, term_tail)) = term.split_first() else {
            return false;
        };
        head == term_head && brute_force_glob_matches(tail, term_tail)
    }

    fn next_glob_random(state: &mut u64) -> u64 {
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        state.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    #[test]
    fn empty_and_single_entry_have_golden_wire_bytes() -> TestResult {
        let empty_sections = TermSectionLengths {
            postings: 0,
            positions: None,
            blockmax: 0,
        };
        let empty = EncodedTermDictionary::encode_sorted(KEYWORD_SCHEMA, empty_sections, &[])?;
        assert_eq!(empty.as_bytes(), &[0, 0, 0, 0]);
        assert_eq!(empty.term_count(), 0);
        assert_eq!(empty.block_count(), 0);
        assert!(
            empty
                .dictionary(KEYWORD_SCHEMA, empty_sections)?
                .cursor()?
                .current()
                .is_none()
        );

        let metadata = TermMetadata::without_positions(1, ByteSpan::new(0, 1), ByteSpan::new(0, 1));
        let input = [TermInput::new(0, b"a", metadata)];
        let sections = TermSectionLengths {
            postings: 1,
            positions: None,
            blockmax: 1,
        };
        let single = EncodedTermDictionary::encode_sorted(KEYWORD_SCHEMA, sections, &input)?;
        assert_eq!(
            single.as_bytes(),
            &[
                1, 0, 0, 0, // block_count
                3, 0, 0, b'a', 0, // index key and blocks-area-relative offset
                1, 0, // entry_count
                3, 0, 0, b'a', // restart key
                1, 0, 1, 0, 1, // doc_freq and section spans
            ]
        );
        assert_eq!(
            single
                .dictionary(KEYWORD_SCHEMA, sections)?
                .lookup(0, b"a")?,
            Some(TermMatch {
                term_ord: 0,
                metadata,
            })
        );

        let mixed_inputs = [
            TermInput::new(
                1,
                b"a",
                TermMetadata::without_positions(1, ByteSpan::new(0, 1), ByteSpan::new(0, 1)),
            ),
            TermInput::new(
                2,
                b"a",
                TermMetadata::with_positions(
                    1,
                    ByteSpan::new(1, 1),
                    ByteSpan::new(0, 1),
                    ByteSpan::new(1, 1),
                ),
            ),
        ];
        let mixed_sections = TermSectionLengths {
            postings: 2,
            positions: Some(1),
            blockmax: 2,
        };
        let mixed =
            EncodedTermDictionary::encode_sorted(MIXED_SCHEMA, mixed_sections, &mixed_inputs)?;
        assert_eq!(
            mixed.as_bytes(),
            &[
                1, 0, 0, 0, // block_count
                3, 0, 1, b'a', 0, // index
                2, 0, // entry_count
                3, 0, 1, b'a', 1, 0, 1, 0, 1, // non-positional restart entry
                1, 2, 2, b'a', 1, 1, 1, 0, 1, 1, 1, // positional compressed entry
            ]
        );
        Ok(())
    }

    #[test]
    fn round_trip_preserves_raw_terms_fields_ordinals_and_metadata() -> TestResult {
        let keys = vec![
            (0, Vec::new()),
            (0, vec![0]),
            (0, b"alpha".to_vec()),
            (0, vec![0xfe, 0xff]),
            (1, b"alpha".to_vec()),
            (1, b"omega".to_vec()),
            (2, b"alpha".to_vec()),
            (2, vec![0xff]),
        ];
        let (encoded, inputs, sections) = encode_fixture(MIXED_SCHEMA, &keys)?;
        let dictionary = encoded.dictionary(MIXED_SCHEMA, sections)?;
        let rows = terms_from_cursor(dictionary.cursor()?)?;
        assert_eq!(rows.len(), keys.len());
        for (ordinal, row) in rows.iter().enumerate() {
            assert_eq!(row.term_ord, u32::try_from(ordinal)?);
            assert_eq!(row.field_ord, keys[ordinal].0);
            assert_eq!(row.term, keys[ordinal].1);
            assert_eq!(row.metadata, inputs[ordinal].metadata);
        }
        assert_eq!(
            encoded.as_bytes(),
            EncodedTermDictionary::encode_sorted(MIXED_SCHEMA, sections, &inputs,)?.as_bytes()
        );
        Ok(())
    }

    #[test]
    fn restart_boundaries_support_hits_and_neighbor_misses() -> TestResult {
        for (count, expected_restarts) in [(15, 1), (16, 1), (17, 2)] {
            let keys = sorted_numbered_keys(count);
            let (encoded, inputs, sections) = encode_fixture(KEYWORD_SCHEMA, &keys)?;
            let dictionary = encoded.dictionary(KEYWORD_SCHEMA, sections)?;
            assert_eq!(dictionary.block_count(), 1);
            assert_eq!(dictionary.restart_count(), expected_restarts);
            let mut scratch = TermScratch::new();
            for (ordinal, (_, term)) in keys.iter().enumerate() {
                assert_eq!(
                    dictionary.lookup_with_scratch(0, term, &mut scratch)?,
                    Some(TermMatch {
                        term_ord: u32::try_from(ordinal)?,
                        metadata: inputs[ordinal].metadata,
                    })
                );
            }
            assert_eq!(dictionary.lookup(0, b"term-00000-")?, None);
            assert_eq!(dictionary.lookup(0, b"term-99999")?, None);
        }
        Ok(())
    }

    #[test]
    fn greedy_blocks_and_oversized_singletons_reopen_canonically() -> TestResult {
        let keys = sorted_numbered_keys(900);
        let (encoded, _, sections) = encode_fixture(KEYWORD_SCHEMA, &keys)?;
        let dictionary = encoded.dictionary(KEYWORD_SCHEMA, sections)?;
        assert!(dictionary.block_count() > 1);
        for block in &dictionary.blocks {
            let byte_len = block.byte_range.end - block.byte_range.start;
            assert!(block.entry_count == 1 || byte_len <= TERM_BLOCK_TARGET_BYTES);
        }
        assert_eq!(terms_from_cursor(dictionary.cursor()?)?.len(), keys.len());

        let maximum_term = vec![b'x'; MAX_TERM_BYTES];
        let maximum_key = vec![(0, maximum_term.clone())];
        let (oversized, _, oversized_sections) = encode_fixture(KEYWORD_SCHEMA, &maximum_key)?;
        let oversized_dictionary = oversized.dictionary(KEYWORD_SCHEMA, oversized_sections)?;
        assert_eq!(oversized_dictionary.block_count(), 1);
        assert!(
            oversized_dictionary.blocks[0].byte_range.end
                - oversized_dictionary.blocks[0].byte_range.start
                > TERM_BLOCK_TARGET_BYTES
        );
        assert!(oversized_dictionary.lookup(0, &maximum_term)?.is_some());

        let too_long = vec![b'x'; MAX_TERM_BYTES + 1];
        let too_long_key = vec![(0, too_long)];
        let (inputs, sections) = fixture_inputs(KEYWORD_SCHEMA, &too_long_key);
        assert!(matches!(
            EncodedTermDictionary::encode_sorted(KEYWORD_SCHEMA, sections, &inputs),
            Err(TermDictionaryError::TermTooLong { length, .. })
                if length == MAX_TERM_BYTES + 1
        ));
        Ok(())
    }

    #[test]
    fn field_prefix_and_range_cursors_match_naive_filtering() -> TestResult {
        let keys = vec![
            (0, Vec::new()),
            (0, b"a".to_vec()),
            (0, b"aa".to_vec()),
            (0, b"ab".to_vec()),
            (0, b"b".to_vec()),
            (0, vec![0xff]),
            (0, vec![0xff, 0]),
            (1, b"a".to_vec()),
            (1, b"z".to_vec()),
            (2, b"a".to_vec()),
            (2, b"b".to_vec()),
        ];
        let (encoded, _, sections) = encode_fixture(MIXED_SCHEMA, &keys)?;
        let dictionary = encoded.dictionary(MIXED_SCHEMA, sections)?;

        let field_zero = terms_from_cursor(dictionary.field_cursor(0)?)?;
        let expected_field_zero: Vec<&[u8]> = keys
            .iter()
            .filter(|(field, _)| *field == 0)
            .map(|(_, term)| term.as_slice())
            .collect();
        assert_eq!(
            field_zero
                .iter()
                .map(|row| row.term.as_slice())
                .collect::<Vec<_>>(),
            expected_field_zero
        );

        let a_prefix = terms_from_cursor(dictionary.prefix_cursor(0, b"a")?)?;
        assert_eq!(
            a_prefix
                .iter()
                .map(|row| row.term.as_slice())
                .collect::<Vec<_>>(),
            vec![b"a".as_slice(), b"aa".as_slice(), b"ab".as_slice()]
        );
        let high_prefix = terms_from_cursor(dictionary.prefix_cursor(0, &[0xff])?)?;
        assert_eq!(
            high_prefix
                .iter()
                .map(|row| row.term.as_slice())
                .collect::<Vec<_>>(),
            vec![&[0xff][..], &[0xff, 0][..]]
        );
        assert_eq!(
            terms_from_cursor(dictionary.prefix_cursor(0, b"")?)?.len(),
            expected_field_zero.len()
        );

        let range = terms_from_cursor(dictionary.range_cursor(
            0,
            Bound::Included(b"aa"),
            Bound::Excluded(&[0xff]),
        )?)?;
        assert_eq!(
            range
                .iter()
                .map(|row| row.term.as_slice())
                .collect::<Vec<_>>(),
            vec![b"aa".as_slice(), b"ab".as_slice(), b"b".as_slice()]
        );
        assert!(
            terms_from_cursor(dictionary.range_cursor(
                0,
                Bound::Included(b"a"),
                Bound::Excluded(b"a"),
            )?)?
            .is_empty()
        );
        assert!(matches!(
            dictionary.range_cursor(0, Bound::Included(b"z"), Bound::Included(b"a")),
            Err(TermDictionaryError::InvalidRange)
        ));
        Ok(())
    }

    #[test]
    fn glob_expansion_covers_every_pattern_class_and_field_scope() -> TestResult {
        let mut keys = vec![
            (0, Vec::new()),
            (0, b"alpha".to_vec()),
            (0, b"alphabet".to_vec()),
            (0, b"beta".to_vec()),
            (0, b"delta".to_vec()),
            (0, b"omega".to_vec()),
            (0, b"pre-middle-suf".to_vec()),
            (0, b"prefix".to_vec()),
            (0, b"quest?mark".to_vec()),
            (0, "résumé".as_bytes().to_vec()),
            (0, "世界-end".as_bytes().to_vec()),
            (0, vec![0xff, 0]),
            (1, b"alpha".to_vec()),
            (1, b"omega".to_vec()),
            (1, b"prefix".to_vec()),
        ];
        keys.sort();
        let (encoded, _, sections) = encode_fixture(MIXED_SCHEMA, &keys)?;
        let dictionary = encoded.dictionary(MIXED_SCHEMA, sections)?;
        let generous_limit = keys.len();

        assert_eq!(
            expanded_glob_terms(&dictionary, 0, b"", generous_limit)?,
            vec![Vec::<u8>::new()]
        );
        assert_eq!(
            expanded_glob_terms(&dictionary, 0, b"alpha", generous_limit)?,
            vec![b"alpha".to_vec()]
        );
        assert_eq!(
            expanded_glob_terms(&dictionary, 0, b"alp*", generous_limit)?,
            vec![b"alpha".to_vec(), b"alphabet".to_vec()]
        );
        assert_eq!(
            expanded_glob_terms(&dictionary, 0, b"alp***", generous_limit)?,
            vec![b"alpha".to_vec(), b"alphabet".to_vec()]
        );
        assert_eq!(
            expanded_glob_terms(&dictionary, 0, b"*ta", generous_limit)?,
            vec![b"beta".to_vec(), b"delta".to_vec()]
        );
        assert_eq!(
            expanded_glob_terms(&dictionary, 0, b"*middle*", generous_limit)?,
            vec![b"pre-middle-suf".to_vec()]
        );
        assert_eq!(
            expanded_glob_terms(&dictionary, 0, b"p*e*suf", generous_limit)?,
            vec![b"pre-middle-suf".to_vec()]
        );
        assert_eq!(
            expanded_glob_terms(&dictionary, 0, b"p**e***suf", generous_limit)?,
            vec![b"pre-middle-suf".to_vec()]
        );
        assert_eq!(
            expanded_glob_terms(&dictionary, 0, b"quest?mark", generous_limit)?,
            vec![b"quest?mark".to_vec()]
        );
        assert_eq!(
            expanded_glob_terms(&dictionary, 0, b"quest?*", generous_limit)?,
            vec![b"quest?mark".to_vec()]
        );
        assert_eq!(
            expanded_glob_terms(&dictionary, 0, "ré*".as_bytes(), generous_limit)?,
            vec!["résumé".as_bytes().to_vec()]
        );
        assert_eq!(
            expanded_glob_terms(&dictionary, 0, "*界*".as_bytes(), generous_limit)?,
            vec!["世界-end".as_bytes().to_vec()]
        );
        assert_eq!(
            expanded_glob_terms(&dictionary, 0, &[b'*', 0xff, b'*'], generous_limit)?,
            vec![vec![0xff, 0]]
        );

        let expected_field_zero = keys
            .iter()
            .filter(|(field_ord, _)| *field_ord == 0)
            .map(|(_, term)| term.clone())
            .collect::<Vec<_>>();
        let all_once = expanded_glob_terms(&dictionary, 0, b"*", generous_limit)?;
        let all_twice = expanded_glob_terms(&dictionary, 0, b"**", generous_limit)?;
        assert_eq!(all_once, expected_field_zero);
        assert_eq!(all_twice, expected_field_zero);
        assert_eq!(
            expanded_glob_terms(&dictionary, 1, b"*", generous_limit)?,
            vec![b"alpha".to_vec(), b"omega".to_vec(), b"prefix".to_vec()]
        );
        assert!(expanded_glob_terms(&dictionary, 2, b"*", generous_limit)?.is_empty());
        assert!(contains_subslice(b"1234567needle", b"needle"));
        assert!(contains_subslice(b"12345678needle", b"needle"));
        assert!(!contains_subslice(b"12345678needlf", b"needle"));
        Ok(())
    }

    #[test]
    fn glob_expansion_limit_fails_on_exactly_one_additional_match() -> TestResult {
        let keys = vec![
            (0, b"a".to_vec()),
            (0, b"aa".to_vec()),
            (0, b"ab".to_vec()),
            (0, b"b".to_vec()),
        ];
        let (encoded, _, sections) = encode_fixture(KEYWORD_SCHEMA, &keys)?;
        let dictionary = encoded.dictionary(KEYWORD_SCHEMA, sections)?;

        assert_eq!(
            expanded_glob_terms(&dictionary, 0, b"a*", 3)?,
            vec![b"a".to_vec(), b"aa".to_vec(), b"ab".to_vec()]
        );
        assert!(matches!(
            dictionary.expand_glob(0, b"a*", 2),
            Err(TermDictionaryError::GlobExpansionLimitExceeded {
                field_ord: 0,
                limit: 2,
                actual: 3,
            })
        ));
        assert!(matches!(
            dictionary.expand_glob(0, b"*a*", 2),
            Err(TermDictionaryError::GlobExpansionLimitExceeded {
                field_ord: 0,
                limit: 2,
                actual: 3,
            })
        ));
        assert!(matches!(
            dictionary.expand_glob(0, b"a", 0),
            Err(TermDictionaryError::GlobExpansionLimitExceeded {
                field_ord: 0,
                limit: 0,
                actual: 1,
            })
        ));
        assert_eq!(
            expanded_glob_terms(&dictionary, 0, b"a", 1)?,
            vec![b"a".to_vec()]
        );
        assert!(expanded_glob_terms(&dictionary, 0, b"missing", 0)?.is_empty());
        Ok(())
    }

    #[test]
    fn seeded_glob_expansion_matches_independent_brute_force_filter() -> TestResult {
        const BASE_SEED: u64 = 0x5eed_fade_cafe_babe;
        const TERM_ALPHABET: [u8; 8] = [b'a', b'b', b'c', b'-', b'?', 0, 0x80, 0xff];
        const PATTERN_ALPHABET: [u8; 9] = [b'a', b'b', b'c', b'-', b'?', 0, 0x80, 0xff, b'*'];

        init_replay_tracing();
        for seed in 0..16_u64 {
            let case_seed = BASE_SEED ^ seed.wrapping_mul(0x9e37_79b9_7f4a_7c15);
            let mut state = case_seed;
            let mut keys = Vec::new();
            for field_ord in 0..=1_u16 {
                for _ in 0..96 {
                    let term_len = usize::try_from(next_glob_random(&mut state) % 13)?;
                    let mut term = Vec::with_capacity(term_len);
                    for _ in 0..term_len {
                        let alphabet_len = u64::try_from(TERM_ALPHABET.len())?;
                        let index = usize::try_from(next_glob_random(&mut state) % alphabet_len)?;
                        term.push(TERM_ALPHABET[index]);
                    }
                    keys.push((field_ord, term));
                }
            }
            keys.sort();
            keys.dedup();

            let (encoded, _, sections) = encode_fixture(MIXED_SCHEMA, &keys)?;
            let dictionary = encoded.dictionary(MIXED_SCHEMA, sections)?;
            for pattern_case in 0..48 {
                let pattern_len = usize::try_from(next_glob_random(&mut state) % 9)?;
                let mut pattern = Vec::with_capacity(pattern_len.saturating_add(2));
                for _ in 0..pattern_len {
                    let alphabet_len = u64::try_from(PATTERN_ALPHABET.len())?;
                    let index = usize::try_from(next_glob_random(&mut state) % alphabet_len)?;
                    pattern.push(PATTERN_ALPHABET[index]);
                }
                if pattern_case % 7 == 0 {
                    pattern.extend_from_slice(b"**");
                }

                for field_ord in 0..=1_u16 {
                    let expected = keys
                        .iter()
                        .filter(|(candidate_field, term)| {
                            *candidate_field == field_ord
                                && brute_force_glob_matches(&pattern, term)
                        })
                        .map(|(_, term)| term.clone())
                        .collect::<Vec<_>>();
                    let actual = expanded_glob_terms(&dictionary, field_ord, &pattern, keys.len())?;
                    assert_eq!(
                        actual, expected,
                        "seed={case_seed:#x} pattern_case={pattern_case} field={field_ord} pattern={pattern:?}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn fixed_seed_term_sets_match_naive_lookup_iteration_and_prefixes() -> TestResult {
        init_replay_tracing();
        for seed in 0..24_u64 {
            let case_seed = seed.wrapping_add(1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            tracing::info!(
                codec = "TERMDICT",
                seed = case_seed,
                case = seed,
                replay = "fixed_seed_term_sets_match_naive_lookup_iteration_and_prefixes",
                "starting randomized dictionary case"
            );
            let mut state = case_seed;
            let mut keys = Vec::new();
            keys.push((0, Vec::new()));
            keys.push((0, vec![u8::try_from(seed)?; 256]));
            for index in 0..192_usize {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                let random = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
                let field_ord = u16::try_from(index % MIXED_FIELDS.len())?;
                let mut term = format!("crate::module::{:02}::symbol::", index % 12).into_bytes();
                term.extend_from_slice(&random.to_be_bytes());
                keys.push((field_ord, term));
            }
            keys.sort_by(|(left_field, left_term), (right_field, right_term)| {
                left_field
                    .cmp(right_field)
                    .then_with(|| left_term.cmp(right_term))
            });
            keys.dedup();

            let (encoded, inputs, sections) = encode_fixture(MIXED_SCHEMA, &keys)?;
            let dictionary = encoded.dictionary(MIXED_SCHEMA, sections)?;
            let rows = terms_from_cursor(dictionary.cursor()?)?;
            assert_eq!(rows.len(), keys.len());
            for (ordinal, row) in rows.iter().enumerate() {
                assert_eq!(row.field_ord, keys[ordinal].0);
                assert_eq!(row.term, keys[ordinal].1);
                assert_eq!(row.metadata, inputs[ordinal].metadata);
                if ordinal.is_multiple_of(11) {
                    assert_eq!(
                        dictionary.lookup(row.field_ord, &row.term)?,
                        Some(TermMatch {
                            term_ord: u32::try_from(ordinal)?,
                            metadata: row.metadata,
                        })
                    );
                }
            }

            for field_ord in 0..u16::try_from(MIXED_FIELDS.len())? {
                let sample = keys
                    .iter()
                    .find(|(field, term)| *field == field_ord && term.len() >= 12)
                    .expect("generated field has a long term");
                let prefix = &sample.1[..12];
                let expected: Vec<&[u8]> = keys
                    .iter()
                    .filter(|(field, term)| *field == field_ord && term.starts_with(prefix))
                    .map(|(_, term)| term.as_slice())
                    .collect();
                let actual = terms_from_cursor(dictionary.prefix_cursor(field_ord, prefix)?)?;
                assert_eq!(
                    actual
                        .iter()
                        .map(|row| row.term.as_slice())
                        .collect::<Vec<_>>(),
                    expected
                );
            }
        }
        Ok(())
    }

    #[test]
    fn encoder_rejects_order_schema_and_reference_violations() {
        let sections = TermSectionLengths {
            postings: 2,
            positions: None,
            blockmax: 2,
        };
        let first = TermMetadata::without_positions(1, ByteSpan::new(0, 1), ByteSpan::new(0, 1));
        let second = TermMetadata::without_positions(1, ByteSpan::new(1, 1), ByteSpan::new(1, 1));
        let descending = [
            TermInput::new(0, b"b", first),
            TermInput::new(0, b"a", second),
        ];
        assert!(matches!(
            EncodedTermDictionary::encode_sorted(KEYWORD_SCHEMA, sections, &descending),
            Err(TermDictionaryError::NonAscendingInput { index: 1 })
        ));
        let duplicate = [
            TermInput::new(0, b"a", first),
            TermInput::new(0, b"a", second),
        ];
        assert!(matches!(
            EncodedTermDictionary::encode_sorted(KEYWORD_SCHEMA, sections, &duplicate),
            Err(TermDictionaryError::NonAscendingInput { index: 1 })
        ));

        let zero = [TermInput::new(
            0,
            b"a",
            TermMetadata::without_positions(0, ByteSpan::new(0, 1), ByteSpan::new(0, 1)),
        )];
        assert!(matches!(
            EncodedTermDictionary::encode_sorted(
                KEYWORD_SCHEMA,
                TermSectionLengths {
                    postings: 1,
                    positions: None,
                    blockmax: 1,
                },
                &zero,
            ),
            Err(TermDictionaryError::ZeroDocFrequency { .. })
        ));

        let unknown = [TermInput::new(9, b"a", first)];
        assert!(matches!(
            EncodedTermDictionary::encode_sorted(
                KEYWORD_SCHEMA,
                TermSectionLengths {
                    postings: 1,
                    positions: None,
                    blockmax: 1,
                },
                &unknown,
            ),
            Err(TermDictionaryError::UnknownField { field_ord: 9 })
        ));

        let gap = [TermInput::new(
            0,
            b"a",
            TermMetadata::without_positions(1, ByteSpan::new(1, 1), ByteSpan::new(0, 1)),
        )];
        assert!(matches!(
            EncodedTermDictionary::encode_sorted(KEYWORD_SCHEMA, sections, &gap),
            Err(TermDictionaryError::NonContiguousReference {
                section: "POSTINGS",
                ..
            })
        ));

        let positions_on_keyword = [TermInput::new(
            0,
            b"a",
            TermMetadata::with_positions(
                1,
                ByteSpan::new(0, 1),
                ByteSpan::new(0, 1),
                ByteSpan::new(0, 1),
            ),
        )];
        assert!(matches!(
            EncodedTermDictionary::encode_sorted(
                KEYWORD_SCHEMA,
                TermSectionLengths {
                    postings: 1,
                    positions: None,
                    blockmax: 1,
                },
                &positions_on_keyword,
            ),
            Err(TermDictionaryError::PositionsPresenceMismatch { .. })
        ));
    }

    #[test]
    fn section_presence_and_exact_consumption_are_enforced() {
        let empty = [0, 0, 0, 0];
        assert!(
            TermDictionary::parse(
                &empty,
                MIXED_SCHEMA,
                TermSectionLengths {
                    postings: 0,
                    positions: Some(0),
                    blockmax: 0,
                },
            )
            .is_ok()
        );
        assert!(matches!(
            TermDictionary::parse(
                &empty,
                MIXED_SCHEMA,
                TermSectionLengths {
                    postings: 0,
                    positions: None,
                    blockmax: 0,
                },
            ),
            Err(TermDictionaryError::PositionsSectionMismatch { .. })
        ));
        assert!(matches!(
            TermDictionary::parse(
                &empty,
                KEYWORD_SCHEMA,
                TermSectionLengths {
                    postings: 1,
                    positions: None,
                    blockmax: 0,
                },
            ),
            Err(TermDictionaryError::SectionLengthMismatch {
                section: "POSTINGS",
                ..
            })
        ));
    }

    #[test]
    fn parser_rejects_structural_mutations_and_every_truncation() -> TestResult {
        let keys = vec![(0, b"a".to_vec())];
        let (encoded, _, sections) = encode_fixture(KEYWORD_SCHEMA, &keys)?;
        let bytes = encoded.as_bytes();
        for end in 0..bytes.len() {
            assert!(TermDictionary::parse(&bytes[..end], KEYWORD_SCHEMA, sections).is_err());
        }

        let mut index_mismatch = bytes.to_vec();
        index_mismatch[7] = b'b';
        assert!(matches!(
            TermDictionary::parse(&index_mismatch, KEYWORD_SCHEMA, sections),
            Err(TermDictionaryError::IndexKeyMismatch { block_index: 0 })
        ));

        let mut bad_offset = bytes.to_vec();
        bad_offset[8] = 1;
        assert!(matches!(
            TermDictionary::parse(&bad_offset, KEYWORD_SCHEMA, sections),
            Err(TermDictionaryError::InvalidBlockOffset { block_index: 0, .. })
        ));

        let mut empty_block = bytes.to_vec();
        empty_block[9..11].copy_from_slice(&0_u16.to_le_bytes());
        assert!(matches!(
            TermDictionary::parse(&empty_block, KEYWORD_SCHEMA, sections),
            Err(TermDictionaryError::EmptyBlock { block_index: 0 })
        ));

        let mut trailing = bytes.to_vec();
        trailing.push(0);
        assert!(matches!(
            TermDictionary::parse(&trailing, KEYWORD_SCHEMA, sections),
            Err(TermDictionaryError::TrailingBlockBytes { block_index: 0, .. })
        ));

        let mut bad_key = vec![0, 0, b'a'];
        let mut invalid_prefix_reader = SliceReader::new(&[4, 1, b'b']);
        assert!(matches!(
            decode_entry(
                &mut invalid_prefix_reader,
                1,
                &mut bad_key,
                KEYWORD_SCHEMA,
                sections,
                1,
            ),
            Err(TermDictionaryError::InvalidPrefix { .. })
        ));
        let mut noncanonical_key = vec![0, 0, b'a'];
        let mut noncanonical_reader = SliceReader::new(&[1, 2, 0, b'b']);
        assert!(matches!(
            decode_entry(
                &mut noncanonical_reader,
                1,
                &mut noncanonical_key,
                KEYWORD_SCHEMA,
                sections,
                1,
            ),
            Err(TermDictionaryError::NonCanonicalPrefix { .. })
        ));
        Ok(())
    }

    #[test]
    fn parser_rejects_split_order_size_vint_and_reference_corruption() -> TestResult {
        fn singleton_block(term: u8, postings_offset: u64, blockmax_offset: u64) -> Vec<u8> {
            let mut block = 1_u16.to_le_bytes().to_vec();
            write_vint(3, &mut block);
            block.extend_from_slice(&[0, 0, term]);
            for value in [1, postings_offset, 1, blockmax_offset, 1] {
                write_vint(value, &mut block);
            }
            block
        }

        fn indexed_dictionary(first_keys: &[(u8, u64)], blocks: &[Vec<u8>]) -> Vec<u8> {
            let mut bytes = u32::try_from(first_keys.len())
                .expect("bounded fixture block count")
                .to_le_bytes()
                .to_vec();
            for (term, relative_offset) in first_keys {
                write_vint(3, &mut bytes);
                bytes.extend_from_slice(&[0, 0, *term]);
                write_vint(*relative_offset, &mut bytes);
            }
            for block in blocks {
                bytes.extend_from_slice(block);
            }
            bytes
        }

        let first = singleton_block(b'a', 0, 0);
        let second = singleton_block(b'b', 1, 1);
        let premature = indexed_dictionary(
            &[(b'a', 0), (b'b', u64::try_from(first.len())?)],
            &[first.clone(), second],
        );
        let two_sections = TermSectionLengths {
            postings: 2,
            positions: None,
            blockmax: 2,
        };
        assert!(matches!(
            TermDictionary::parse(&premature, KEYWORD_SCHEMA, two_sections),
            Err(TermDictionaryError::PrematureBlockSplit { block_index: 0 })
        ));

        let mut descending_block = 2_u16.to_le_bytes().to_vec();
        write_vint(3, &mut descending_block);
        descending_block.extend_from_slice(&[0, 0, b'a']);
        for value in [1, 0, 1, 0, 1] {
            write_vint(value, &mut descending_block);
        }
        write_vint(2, &mut descending_block);
        write_vint(0, &mut descending_block);
        for value in [1, 1, 1, 1, 1] {
            write_vint(value, &mut descending_block);
        }
        let descending = indexed_dictionary(&[(b'a', 0)], &[descending_block]);
        assert!(matches!(
            TermDictionary::parse(&descending, KEYWORD_SCHEMA, two_sections),
            Err(TermDictionaryError::NonAscendingKey { term_ordinal: 1 })
        ));

        let mut oversized_block = 2_u16.to_le_bytes().to_vec();
        oversized_block.resize(TERM_BLOCK_TARGET_BYTES + 1, 0);
        let oversized = indexed_dictionary(&[(b'a', 0)], &[oversized_block]);
        assert!(matches!(
            TermDictionary::parse(&oversized, KEYWORD_SCHEMA, two_sections),
            Err(TermDictionaryError::OversizedBlock {
                block_index: 0,
                entry_count: 2,
                ..
            })
        ));

        let valid = indexed_dictionary(&[(b'a', 0)], &[first]);
        let one_section = TermSectionLengths {
            postings: 1,
            positions: None,
            blockmax: 1,
        };
        let mut out_of_bounds = valid.clone();
        out_of_bounds[16] = 2;
        assert!(matches!(
            TermDictionary::parse(&out_of_bounds, KEYWORD_SCHEMA, one_section),
            Err(TermDictionaryError::ReferenceOutOfBounds {
                section: "POSTINGS",
                ..
            })
        ));
        assert!(matches!(
            TermDictionary::parse(
                &valid,
                KEYWORD_SCHEMA,
                TermSectionLengths {
                    postings: 2,
                    ..one_section
                },
            ),
            Err(TermDictionaryError::SectionLengthMismatch {
                section: "POSTINGS",
                ..
            })
        ));

        let mut overflow_block = 1_u16.to_le_bytes().to_vec();
        write_vint(3, &mut overflow_block);
        overflow_block.extend_from_slice(&[0, 0, b'a']);
        for value in [1, u64::MAX, 1, 0, 1] {
            write_vint(value, &mut overflow_block);
        }
        let overflow = indexed_dictionary(&[(b'a', 0)], &[overflow_block]);
        assert!(matches!(
            TermDictionary::parse(
                &overflow,
                KEYWORD_SCHEMA,
                TermSectionLengths {
                    postings: u64::MAX,
                    positions: None,
                    blockmax: 1,
                },
            ),
            Err(TermDictionaryError::ReferenceOverflow {
                section: "POSTINGS",
                ..
            })
        ));

        let mut noncanonical_vint = valid;
        noncanonical_vint.splice(8..9, [0x80, 0]);
        assert!(matches!(
            TermDictionary::parse(&noncanonical_vint, KEYWORD_SCHEMA, one_section),
            Err(TermDictionaryError::NonCanonicalVint { offset: 8 })
        ));

        let hostile_header = 2_u32.to_le_bytes();
        assert!(matches!(
            TermDictionary::parse_with_limits(
                &hostile_header,
                KEYWORD_SCHEMA,
                TermSectionLengths {
                    postings: 0,
                    positions: None,
                    blockmax: 0,
                },
                TermDictionaryLimits {
                    max_bytes: hostile_header.len(),
                    max_blocks: 2,
                    max_terms: 1,
                    max_restarts: 2,
                },
            ),
            Err(TermDictionaryError::TermBudgetExceeded {
                limit: 1,
                actual: 2,
            })
        ));
        Ok(())
    }

    #[test]
    fn vint_boundaries_are_canonical_and_overflow_safe() -> TestResult {
        for value in [0, 127, 128, 16_383, 16_384, u64::from(u32::MAX), u64::MAX] {
            let mut bytes = Vec::new();
            write_vint(value, &mut bytes);
            assert_eq!(bytes.len(), vint_len(value));
            let mut reader = SliceReader::new(&bytes);
            assert_eq!(reader.read_vint()?, value);
            assert_eq!(reader.position(), bytes.len());
        }
        assert!(matches!(
            SliceReader::new(&[0x80, 0]).read_vint(),
            Err(TermDictionaryError::NonCanonicalVint { offset: 0 })
        ));
        assert!(matches!(
            SliceReader::new(&[0xff; 10]).read_vint(),
            Err(TermDictionaryError::VintOverflow { offset: 0 })
        ));
        assert!(matches!(
            SliceReader::new(&[0x80]).read_vint(),
            Err(TermDictionaryError::Truncated { .. })
        ));
        Ok(())
    }

    #[test]
    fn explicit_reader_and_writer_budgets_fail_closed() -> TestResult {
        let keys = sorted_numbered_keys(17);
        let (encoded, inputs, sections) = encode_fixture(KEYWORD_SCHEMA, &keys)?;
        let bytes = encoded.as_bytes();
        let base = TermDictionaryLimits {
            max_bytes: bytes.len(),
            max_blocks: usize::try_from(encoded.block_count())?,
            max_terms: keys.len(),
            max_restarts: 2,
        };
        assert!(TermDictionary::parse_with_limits(bytes, KEYWORD_SCHEMA, sections, base).is_ok());
        assert!(matches!(
            TermDictionary::parse_with_limits(
                bytes,
                KEYWORD_SCHEMA,
                sections,
                TermDictionaryLimits {
                    max_bytes: bytes.len() - 1,
                    ..base
                },
            ),
            Err(TermDictionaryError::ByteBudgetExceeded { .. })
        ));
        assert!(matches!(
            TermDictionary::parse_with_limits(
                bytes,
                KEYWORD_SCHEMA,
                sections,
                TermDictionaryLimits {
                    max_blocks: 0,
                    ..base
                },
            ),
            Err(TermDictionaryError::BlockBudgetExceeded { .. })
        ));
        assert!(matches!(
            TermDictionary::parse_with_limits(
                bytes,
                KEYWORD_SCHEMA,
                sections,
                TermDictionaryLimits {
                    max_terms: 16,
                    ..base
                },
            ),
            Err(TermDictionaryError::TermBudgetExceeded { .. })
        ));
        assert!(matches!(
            TermDictionary::parse_with_limits(
                bytes,
                KEYWORD_SCHEMA,
                sections,
                TermDictionaryLimits {
                    max_restarts: 1,
                    ..base
                },
            ),
            Err(TermDictionaryError::RestartBudgetExceeded { .. })
        ));
        assert!(matches!(
            EncodedTermDictionary::encode_sorted_with_limits(
                KEYWORD_SCHEMA,
                sections,
                &inputs,
                TermDictionaryLimits {
                    max_restarts: 1,
                    ..TermDictionaryLimits::default()
                },
            ),
            Err(TermDictionaryError::RestartBudgetExceeded {
                limit: 1,
                actual: 2,
            })
        ));
        assert!(matches!(
            EncodedTermDictionary::encode_sorted_with_limits(
                KEYWORD_SCHEMA,
                sections,
                &inputs,
                TermDictionaryLimits {
                    max_bytes: bytes.len() - 1,
                    ..TermDictionaryLimits::default()
                },
            ),
            Err(TermDictionaryError::ByteBudgetExceeded { .. })
        ));
        assert!(matches!(
            EncodedTermDictionary::encode_sorted_with_limits(
                KEYWORD_SCHEMA,
                sections,
                &inputs,
                TermDictionaryLimits {
                    max_blocks: 0,
                    ..TermDictionaryLimits::default()
                },
            ),
            Err(TermDictionaryError::BlockBudgetExceeded {
                limit: 0,
                actual: 1,
            })
        ));
        assert!(matches!(
            EncodedTermDictionary::encode_sorted_with_limits(
                KEYWORD_SCHEMA,
                sections,
                &inputs,
                TermDictionaryLimits {
                    max_terms: 16,
                    ..TermDictionaryLimits::default()
                },
            ),
            Err(TermDictionaryError::TermBudgetExceeded {
                limit: 16,
                actual: 17,
            })
        ));
        Ok(())
    }

    #[test]
    fn bounded_materialization_and_cursor_fusion_are_explicit() -> TestResult {
        let keys = sorted_numbered_keys(3);
        let (encoded, _, sections) = encode_fixture(KEYWORD_SCHEMA, &keys)?;
        let dictionary = encoded.dictionary(KEYWORD_SCHEMA, sections)?;
        assert!(matches!(
            dictionary.cursor()?.collect_bounded(2),
            Err(TermDictionaryError::MaterializationLimitExceeded { limit: 2 })
        ));
        let mut cursor = dictionary.cursor()?;
        assert!(cursor.current().is_some());
        assert!(cursor.next()?.is_some());
        assert!(cursor.next()?.is_some());
        assert!(cursor.next()?.is_none());
        assert!(cursor.next()?.is_none());
        Ok(())
    }

    #[test]
    fn arbitrary_and_unaligned_bytes_never_panic() -> TestResult {
        const BASE_SEED: u64 = 0x4d59_5df4_d0f3_3173;

        init_replay_tracing();
        let valid_keys = sorted_numbered_keys(17);
        let (encoded, _, sections) = encode_fixture(KEYWORD_SCHEMA, &valid_keys)?;
        for shift in 0..32 {
            let mut padded = vec![0xa5; shift];
            padded.extend_from_slice(encoded.as_bytes());
            let dictionary = TermDictionary::parse(&padded[shift..], KEYWORD_SCHEMA, sections)?;
            assert_eq!(dictionary.term_count(), 17);
        }

        let mut state = BASE_SEED;
        for length in 0..=512 {
            tracing::info!(
                codec = "TERMDICT",
                seed = BASE_SEED,
                case = length,
                replay = "arbitrary_and_unaligned_bytes_never_panic",
                "starting hostile-byte case"
            );
            let mut bytes = Vec::with_capacity(length);
            for _ in 0..length {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                bytes.push(state.wrapping_mul(0x2545_f491_4f6c_dd1d).to_le_bytes()[0]);
            }
            let result = catch_unwind(AssertUnwindSafe(|| {
                let _ = TermDictionary::parse(
                    &bytes,
                    MIXED_SCHEMA,
                    TermSectionLengths {
                        postings: 64,
                        positions: Some(64),
                        blockmax: 64,
                    },
                );
            }));
            assert!(
                result.is_ok(),
                "seed={BASE_SEED:#x} parser panicked for {length} hostile bytes"
            );
        }
        Ok(())
    }
}
