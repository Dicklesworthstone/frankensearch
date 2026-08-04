//! FSVI v2 structural-corruption matrix reachable from the public API
//! (`bd-fsvi-readonly-semantic-inspection-qxo6`).
//!
//! The in-crate matrices cover the identity/fingerprint half of the header and
//! the content half (record flags bit 0, document-id bytes, alignment padding,
//! vector slab, trailing bytes, truncation). This file covers the cells they
//! do not reach:
//!
//! - the LAYOUT header fields — `header_size`, `record_count`, `vectors_offset`
//!   — mutated directly, including the `checked_mul`/`checked_add` overflow
//!   boundaries in the layout validator;
//! - the RECORD TABLE fields — `doc_id_hash`, `doc_id_offset`, `doc_id_len`,
//!   and an unsupported flag BIT (the in-crate matrix flips bit 0, which is
//!   the tombstone flag, so the unsupported-flag branch stays unexercised).
//!
//! Every mutation is presented to the parser as a SELF-CONSISTENT artifact:
//! header mutations are followed by a recomputed header CRC, so the parser is
//! forced to reject on the field's meaning rather than on a checksum. A test
//! that corrupted the CRC as a side effect would prove only that CRC checking
//! works.
//!
//! No production source is edited by this file. The fixture is written by the
//! real `VectorIndex::create_v2` writer and admitted through the real
//! `ValidatedFsviBytes::from_arc` owner, so the matrix exercises the shipping
//! parse/validate path rather than a test-local reimplementation of it.

use std::path::PathBuf;
use std::sync::Arc;

use frankensearch_core::SearchError;
use frankensearch_core::generation::{
    ArtifactGenerationIdentityV1, EmbeddingIdentityBundleV1, QuantizationFormat,
};
use frankensearch_index::{
    FsviAdmissionError, FsviV2IdentityBinding, ValidatedFsviBytes, VectorIndex,
};

// ─── Format facts (documented in the frankensearch-index module header) ──────

/// `header_size: u32` immediately after magic and version.
const HEADER_SIZE_OFFSET: usize = 6;
/// `record_count: u64`.
const RECORD_COUNT_OFFSET: usize = 20;
/// `vectors_offset: u64`.
const VECTORS_OFFSET_OFFSET: usize = 28;
/// One record: `doc_id_hash: u64`, `doc_id_offset: u32`, `doc_id_len: u16`,
/// `flags: u16`.
const RECORD_SIZE_BYTES: usize = 16;
const RECORD_DOC_ID_HASH: usize = 0;
const RECORD_DOC_ID_OFFSET: usize = 8;
const RECORD_DOC_ID_LEN: usize = 12;
const RECORD_FLAGS: usize = 14;
/// Bit 0 is the tombstone flag; every other bit is unsupported.
const RECORD_FLAG_TOMBSTONE: u16 = 0x0001;
/// The vector slab is 64-byte aligned.
const VECTOR_ALIGN_BYTES: u64 = 64;

// ─── Expected rejection reason per cell ─────────────────────────────────────
//
// Each cell pins the branch it is aimed at. These strings were read off the
// real parser, not guessed: a cell that starts rejecting somewhere earlier
// fails loudly instead of silently proving nothing.

/// A self-consistent `header_size` drift is caught by the header's own
/// internal-consistency check: the canonical identity lengths no longer end
/// where the declared CRC position says they must.
const HEADER_SIZE_DETAIL: &str = "canonical identity lengths end at byte";
const LENGTH_DETAIL: &str = "v2 file length must exactly match the bound layout";
const INSIDE_TABLE_DETAIL: &str = "vectors_offset points inside the record table";
const TABLE_SIZE_OVERFLOW_DETAIL: &str = "record table size overflow";
const TABLE_OFFSET_OVERFLOW_DETAIL: &str = "record table offset overflow";
const SLAB_END_OVERFLOW_DETAIL: &str = "vector slab end overflow";
const HASH_DETAIL: &str = "v2 document hash mismatch";
const CONTIGUOUS_DETAIL: &str = "v2 document strings must be contiguous in record order";
const EMPTY_ID_DETAIL: &str = "v2 document ids must not be empty";
const ID_BOUNDS_DETAIL: &str = "v2 document id extends beyond the bound string table";
const UNSUPPORTED_FLAGS_DETAIL: &str = "uses unsupported flags";

fn temp_dir() -> PathBuf {
    let dir = std::env::temp_dir().join("frankensearch_fsvi_v2_corruption_matrix");
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn temp_index_path(name: &str) -> PathBuf {
    temp_dir().join(format!("{name}-{}.fsvi", std::process::id()))
}

fn binding(model_id: &str, dimension: u32, sequence: u64, nonce: u8) -> FsviV2IdentityBinding {
    let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(model_id, dimension);
    identity.storage.format.clear();
    identity.storage.format.push_str("fsvi-v2");
    identity.storage.quantization = QuantizationFormat::F16;
    identity.storage.endianness.clear();
    identity.storage.endianness.push_str("little-endian");
    FsviV2IdentityBinding::new(
        ArtifactGenerationIdentityV1::new(sequence, [nonce; 16]).expect("valid test generation"),
        identity.freeze().expect("valid frozen identity"),
    )
    .expect("valid FSVI v2 binding")
}

/// Write a real v2 fixture through the production writer and return its bytes.
fn fixture(name: &str) -> (FsviV2IdentityBinding, Vec<u8>) {
    fixture_with_records(name, 2)
}

/// Some cells need a specific record count: the vector slab is
/// `record_count * dimension * bytes_per_element` bytes, and the slab-end
/// addition can only overflow when that product exceeds the 64-byte alignment
/// granularity, because the largest aligned offset is `u64::MAX - 63`.
fn fixture_with_records(name: &str, records: usize) -> (FsviV2IdentityBinding, Vec<u8>) {
    let path = temp_index_path(name);
    let identity = binding("v2-corruption-matrix", 4, 21, 0xc1);
    let mut writer =
        VectorIndex::create_v2(&path, identity.clone()).expect("create v2 fixture writer");
    for record in 0..records {
        let mut vector = [0.0_f32; 4];
        vector[record % 4] = 1.0;
        writer
            .write_record(&format!("doc-{record:04}"), &vector)
            .expect("write fixture record");
    }
    writer.finish().expect("finish v2 fixture");
    let bytes = std::fs::read(&path).expect("read v2 fixture");
    let _ = std::fs::remove_file(&path);
    (identity, bytes)
}

/// Bytes the vector slab occupies for a fixture of `records` rows
/// (dimension 4, F16 = 2 bytes per element).
fn slab_bytes(records: u64) -> u64 {
    records * 4 * 2
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("u32 field is in range"),
    )
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("u64 field is in range"),
    )
}

fn header_size(bytes: &[u8]) -> usize {
    usize::try_from(read_u32(bytes, HEADER_SIZE_OFFSET)).expect("header size fits usize")
}

fn crc32(data: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(data);
    hasher.finalize()
}

/// Recompute the stored header CRC so a mutated field is judged on its
/// meaning, not on a checksum the mutation happened to break.
fn refresh_header_crc(bytes: &mut [u8]) {
    let size = header_size(bytes);
    let crc_offset = size - 4;
    let crc = crc32(&bytes[..crc_offset]);
    bytes[crc_offset..size].copy_from_slice(&crc.to_le_bytes());
}

fn header_crc_is_valid(bytes: &[u8]) -> bool {
    let size = header_size(bytes);
    let crc_offset = size - 4;
    read_u32(bytes, crc_offset) == crc32(&bytes[..crc_offset])
}

fn record_offset(bytes: &[u8], index: usize) -> usize {
    header_size(bytes) + index * RECORD_SIZE_BYTES
}

/// Reject, AND reject for this cell's own reason.
///
/// Matching only on `IndexCorrupted` would let a cell pass while the parser
/// rejected it for an unrelated earlier check — the mutation would then prove
/// nothing about the field it targets. Each cell pins the detail substring the
/// branch it is aimed at produces.
fn assert_rejected(
    bytes: Vec<u8>,
    expected: &FsviV2IdentityBinding,
    cell: &str,
    expected_detail: &str,
) {
    let observed = ValidatedFsviBytes::from_arc(Arc::<[u8]>::from(bytes), expected);
    match &observed {
        Err(FsviAdmissionError::Index(SearchError::IndexCorrupted { detail, .. })) => {
            assert!(
                detail.contains(expected_detail),
                "{cell}: rejected, but not for its own reason. \
                 expected a detail containing {expected_detail:?}, observed {detail:?}"
            );
        }
        other => panic!("{cell}: admission must reject this artifact, observed {other:?}"),
    }
}

fn assert_admitted(bytes: Vec<u8>, expected: &FsviV2IdentityBinding, cell: &str) {
    let observed = ValidatedFsviBytes::from_arc(Arc::<[u8]>::from(bytes), expected);
    assert!(
        observed.is_ok(),
        "{cell}: the control artifact must admit, observed {observed:?}"
    );
}

#[test]
fn the_unmutated_fixture_admits() {
    let (expected, bytes) = fixture("control");
    assert_admitted(bytes, &expected, "control");
}

/// `header_size`, `record_count`, and `vectors_offset` are the layout fields
/// the in-crate header matrix does not mutate. Each is corrupted with a
/// recomputed CRC, so the parser sees an internally consistent header.
#[test]
fn layout_header_fields_fail_closed_with_a_valid_crc() {
    let (expected, source) = fixture("layout-fields");

    // header_size: shifts the record table base, so the bound layout no longer
    // matches the file length.
    let mut grown = source.clone();
    let original_size = header_size(&grown);
    grown[HEADER_SIZE_OFFSET..HEADER_SIZE_OFFSET + 4].copy_from_slice(
        &u32::try_from(original_size + 1)
            .expect("header size fits u32")
            .to_le_bytes(),
    );
    refresh_header_crc(&mut grown);
    assert!(header_crc_is_valid(&grown));
    assert_rejected(grown, &expected, "header_size+1", HEADER_SIZE_DETAIL);

    // record_count: one more record than the file can hold.
    let mut counted = source.clone();
    let original_count = read_u64(&counted, RECORD_COUNT_OFFSET);
    assert_eq!(original_count, 2, "fixture writes two records");
    counted[RECORD_COUNT_OFFSET..RECORD_COUNT_OFFSET + 8]
        .copy_from_slice(&(original_count + 1).to_le_bytes());
    refresh_header_crc(&mut counted);
    assert!(header_crc_is_valid(&counted));
    assert_rejected(counted, &expected, "record_count+1", LENGTH_DETAIL);

    // vectors_offset pointing back inside the record/string table region. Zero
    // is 64-byte aligned, so the alignment check cannot be what rejects it.
    let mut underflowed = source.clone();
    underflowed[VECTORS_OFFSET_OFFSET..VECTORS_OFFSET_OFFSET + 8]
        .copy_from_slice(&0_u64.to_le_bytes());
    refresh_header_crc(&mut underflowed);
    assert_eq!(0 % VECTOR_ALIGN_BYTES, 0);
    assert_rejected(
        underflowed,
        &expected,
        "vectors_offset=0",
        INSIDE_TABLE_DETAIL,
    );
}

/// The `checked_mul`/`checked_add` guards in the layout validator. These are
/// the arithmetic boundaries: a validator that computed the record table or
/// slab extent with wrapping arithmetic would land back inside the file and
/// read whatever happened to be there.
#[test]
fn layout_arithmetic_overflow_boundaries_fail_closed() {
    let (expected, source) = fixture("layout-overflow");

    // record_count * RECORD_SIZE_BYTES overflows.
    let mut table_overflow = source.clone();
    table_overflow[RECORD_COUNT_OFFSET..RECORD_COUNT_OFFSET + 8]
        .copy_from_slice(&u64::MAX.to_le_bytes());
    refresh_header_crc(&mut table_overflow);
    assert_rejected(
        table_overflow,
        &expected,
        "record_count=u64::MAX",
        TABLE_SIZE_OVERFLOW_DETAIL,
    );

    // The product fits, but header_size + record_bytes overflows.
    let mut offset_overflow = source.clone();
    offset_overflow[RECORD_COUNT_OFFSET..RECORD_COUNT_OFFSET + 8]
        .copy_from_slice(&(u64::MAX / RECORD_SIZE_BYTES as u64).to_le_bytes());
    refresh_header_crc(&mut offset_overflow);
    assert_rejected(
        offset_overflow,
        &expected,
        "record table offset overflow",
        TABLE_OFFSET_OVERFLOW_DETAIL,
    );

    // vectors_offset is aligned, fits usize, and sits above the string table,
    // so only the slab-end addition can catch it. This needs a slab WIDER than
    // the alignment granularity: with the two-record fixture the slab is 16
    // bytes, the largest aligned offset is u64::MAX - 63, and their sum does
    // not overflow at all — the artifact is then rejected by the length check
    // and this branch is never reached. The per-cell detail assertion is what
    // exposed that; a bare "is it corrupted?" check would have called the
    // unreached branch covered.
    let (wide_expected, wide_source) = fixture_with_records("layout-overflow-wide", 9);
    let mut slab_overflow = wide_source;
    let aligned_max = u64::MAX - (u64::MAX % VECTOR_ALIGN_BYTES);
    assert_eq!(aligned_max % VECTOR_ALIGN_BYTES, 0);
    assert!(
        slab_bytes(9) > VECTOR_ALIGN_BYTES,
        "the slab must exceed the alignment granularity or the sum cannot overflow"
    );
    assert!(
        aligned_max.checked_add(slab_bytes(9)).is_none(),
        "the fixture must actually make the slab-end addition overflow"
    );
    slab_overflow[VECTORS_OFFSET_OFFSET..VECTORS_OFFSET_OFFSET + 8]
        .copy_from_slice(&aligned_max.to_le_bytes());
    refresh_header_crc(&mut slab_overflow);
    assert_rejected(
        slab_overflow,
        &wide_expected,
        "vector slab end overflow",
        SLAB_END_OVERFLOW_DETAIL,
    );
}

/// Record-table fields. The in-crate content matrix mutates record flags bit 0
/// (the tombstone flag) and the document-id bytes; the hash, offset, length,
/// and unsupported flag bits are untouched by it.
#[test]
fn record_table_fields_fail_closed() {
    let (expected, source) = fixture("record-fields");
    let first = record_offset(&source, 0);

    // Stored document-id hash no longer matches the string it points at. A
    // reader that trusted the stored hash for lookup would resolve the wrong
    // row rather than reject.
    let mut hashed = source.clone();
    hashed[first + RECORD_DOC_ID_HASH] ^= 0x01;
    assert_rejected(hashed, &expected, "doc_id_hash", HASH_DETAIL);

    // Document strings must be contiguous in record order; the first record's
    // offset must be zero.
    let mut offset = source.clone();
    offset[first + RECORD_DOC_ID_OFFSET] ^= 0x01;
    assert_rejected(offset, &expected, "doc_id_offset", CONTIGUOUS_DETAIL);

    // A zero-length document id.
    let mut empty = source.clone();
    empty[first + RECORD_DOC_ID_LEN..first + RECORD_DOC_ID_LEN + 2]
        .copy_from_slice(&0_u16.to_le_bytes());
    assert_rejected(empty, &expected, "doc_id_len=0", EMPTY_ID_DETAIL);

    // A document id that runs past the string table into the vector slab.
    let mut overrun = source.clone();
    overrun[first + RECORD_DOC_ID_LEN..first + RECORD_DOC_ID_LEN + 2]
        .copy_from_slice(&u16::MAX.to_le_bytes());
    assert_rejected(overrun, &expected, "doc_id_len=u16::MAX", ID_BOUNDS_DETAIL);

    // An unsupported flag BIT. Bit 0 is the tombstone flag and is legal, so
    // this must be a different bit or the cell proves nothing.
    let mut flagged = source.clone();
    let unsupported = RECORD_FLAG_TOMBSTONE << 1;
    assert_ne!(unsupported & RECORD_FLAG_TOMBSTONE, unsupported);
    let stored = u16::from_le_bytes(
        flagged[first + RECORD_FLAGS..first + RECORD_FLAGS + 2]
            .try_into()
            .expect("flags field"),
    );
    flagged[first + RECORD_FLAGS..first + RECORD_FLAGS + 2]
        .copy_from_slice(&(stored | unsupported).to_le_bytes());
    assert_rejected(
        flagged,
        &expected,
        "unsupported flag bit",
        UNSUPPORTED_FLAGS_DETAIL,
    );
}

/// Planted negative: a forgery that satisfies every structural check a naive
/// validator performs.
///
/// The last record's document id is shortened by one byte and the freed byte
/// is zeroed. The result keeps a valid header CRC, the exact declared file
/// length, in-bounds record and string offsets, contiguous strings in record
/// order, and all-zero alignment padding. Only recomputing the document-id
/// hash — and the ordered live-docset digest built from it — separates this
/// artifact from a healthy one, and a reader that accepted it would serve a
/// document id the sealed generation never contained.
#[test]
fn a_truncated_document_id_that_passes_every_structural_check_is_still_rejected() {
    let (expected, source) = fixture("silent-forgery");
    let size = header_size(&source);
    let record_count = usize::try_from(read_u64(&source, RECORD_COUNT_OFFSET)).expect("count");
    let vectors_offset =
        usize::try_from(read_u64(&source, VECTORS_OFFSET_OFFSET)).expect("vectors offset");
    let strings_offset = size + record_count * RECORD_SIZE_BYTES;

    let last = record_offset(&source, record_count - 1);
    let last_id_offset = usize::try_from(read_u32(&source, last + RECORD_DOC_ID_OFFSET))
        .expect("document id offset");
    let last_id_len = usize::from(u16::from_le_bytes(
        source[last + RECORD_DOC_ID_LEN..last + RECORD_DOC_ID_LEN + 2]
            .try_into()
            .expect("document id length"),
    ));
    assert!(last_id_len > 1, "the fixture id must be shortenable");

    let mut forged = source.clone();
    // Shorten the declared length and zero the byte that leaves the table, so
    // the padding region stays all-zero.
    forged[last + RECORD_DOC_ID_LEN..last + RECORD_DOC_ID_LEN + 2].copy_from_slice(
        &u16::try_from(last_id_len - 1)
            .expect("shortened length")
            .to_le_bytes(),
    );
    let freed = strings_offset + last_id_offset + last_id_len - 1;
    forged[freed] = 0;

    // Everything a naive validator would check still holds.
    assert_eq!(
        forged.len(),
        source.len(),
        "forgery must preserve the exact file length"
    );
    assert!(
        header_crc_is_valid(&forged),
        "forgery must preserve a valid header CRC"
    );
    let mut expected_string_offset = 0_usize;
    for index in 0..record_count {
        let record = record_offset(&forged, index);
        let declared_offset =
            usize::try_from(read_u32(&forged, record + RECORD_DOC_ID_OFFSET)).expect("offset");
        let declared_len = usize::from(u16::from_le_bytes(
            forged[record + RECORD_DOC_ID_LEN..record + RECORD_DOC_ID_LEN + 2]
                .try_into()
                .expect("length"),
        ));
        assert_eq!(
            declared_offset, expected_string_offset,
            "forgery must keep the string table contiguous in record order"
        );
        assert!(
            strings_offset + declared_offset + declared_len <= vectors_offset,
            "forgery must keep every document id inside the string table"
        );
        expected_string_offset += declared_len;
    }
    let string_end = strings_offset + expected_string_offset;
    assert!(
        forged[string_end..vectors_offset]
            .iter()
            .all(|byte| *byte == 0),
        "forgery must keep the alignment padding all-zero"
    );

    assert_rejected(
        forged,
        &expected,
        "truncated document id forgery",
        HASH_DETAIL,
    );

    // Control: the same fixture without the forgery admits, so the rejection
    // above is the forgery and not the fixture.
    assert_admitted(source, &expected, "silent-forgery control");
}
