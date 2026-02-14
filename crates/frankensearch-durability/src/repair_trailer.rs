use std::path::PathBuf;

use frankensearch_core::{SearchError, SearchResult};

/// Magic prefix for durability sidecar trailers.
pub const REPAIR_TRAILER_MAGIC: [u8; 4] = *b"FSDR";
/// Binary trailer version.
pub const REPAIR_TRAILER_VERSION: u16 = 1;

const FIXED_HEADER_BYTES: usize = 4 + 2 + 4 + 4 + 8 + 4 + 4;
const TRAILER_CRC_BYTES: usize = 4;
const MIN_TRAILER_BYTES: usize = FIXED_HEADER_BYTES + TRAILER_CRC_BYTES;
const LENGTH_PREFIX_BYTES: usize = 8;

/// Header metadata stored before serialized repair symbols.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RepairTrailerHeader {
    pub symbol_size: u32,
    pub k_source: u32,
    pub source_len: u64,
    pub source_crc32: u32,
    pub repair_symbol_count: u32,
}

/// One repair symbol entry in the trailer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepairSymbol {
    pub esi: u32,
    pub data: Vec<u8>,
}

/// Serialize trailer header + repair symbols + trailer CRC32.
pub fn serialize_repair_trailer(
    header: &RepairTrailerHeader,
    symbols: &[RepairSymbol],
) -> SearchResult<Vec<u8>> {
    let expected =
        usize::try_from(header.repair_symbol_count).map_err(|_| SearchError::InvalidConfig {
            field: "repair_symbol_count".to_owned(),
            value: header.repair_symbol_count.to_string(),
            reason: "cannot convert to usize".to_owned(),
        })?;
    if symbols.len() != expected {
        return Err(SearchError::InvalidConfig {
            field: "repair_symbol_count".to_owned(),
            value: header.repair_symbol_count.to_string(),
            reason: format!("does not match symbol payload count {}", symbols.len()),
        });
    }

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&REPAIR_TRAILER_MAGIC);
    bytes.extend_from_slice(&REPAIR_TRAILER_VERSION.to_le_bytes());
    bytes.extend_from_slice(&header.symbol_size.to_le_bytes());
    bytes.extend_from_slice(&header.k_source.to_le_bytes());
    bytes.extend_from_slice(&header.source_len.to_le_bytes());
    bytes.extend_from_slice(&header.source_crc32.to_le_bytes());
    bytes.extend_from_slice(&header.repair_symbol_count.to_le_bytes());

    for symbol in symbols {
        bytes.extend_from_slice(&symbol.esi.to_le_bytes());
        let symbol_len =
            u32::try_from(symbol.data.len()).map_err(|_| SearchError::InvalidConfig {
                field: "repair_symbol".to_owned(),
                value: symbol.data.len().to_string(),
                reason: "symbol byte length exceeds u32".to_owned(),
            })?;
        bytes.extend_from_slice(&symbol_len.to_le_bytes());
        bytes.extend_from_slice(&symbol.data);
    }

    let checksum = crc32fast::hash(&bytes);
    bytes.extend_from_slice(&checksum.to_le_bytes());
    Ok(bytes)
}

/// Deserialize and validate a repair trailer.
pub fn deserialize_repair_trailer(
    bytes: &[u8],
) -> SearchResult<(RepairTrailerHeader, Vec<RepairSymbol>)> {
    if bytes.len() < MIN_TRAILER_BYTES {
        return Err(trailer_corruption("repair trailer too short"));
    }

    let payload_len = bytes
        .len()
        .checked_sub(TRAILER_CRC_BYTES)
        .ok_or_else(|| trailer_corruption("missing trailer crc"))?;
    let payload = &bytes[..payload_len];

    let expected_crc = read_u32_le(bytes, payload_len)?;
    let actual_crc = crc32fast::hash(payload);
    if actual_crc != expected_crc {
        return Err(trailer_corruption("repair trailer crc mismatch"));
    }

    if read_magic(payload, 0)? != REPAIR_TRAILER_MAGIC {
        return Err(trailer_corruption("invalid repair trailer magic"));
    }

    let version = read_u16_le(payload, 4)?;
    if version != REPAIR_TRAILER_VERSION {
        return Err(trailer_corruption("unsupported repair trailer version"));
    }

    let header = RepairTrailerHeader {
        symbol_size: read_u32_le(payload, 6)?,
        k_source: read_u32_le(payload, 10)?,
        source_len: read_u64_le(payload, 14)?,
        source_crc32: read_u32_le(payload, 22)?,
        repair_symbol_count: read_u32_le(payload, 26)?,
    };
    if header.symbol_size == 0 {
        return Err(trailer_corruption(
            "repair trailer symbol_size must be greater than zero",
        ));
    }
    if header.k_source == 0 && header.source_len > 0 {
        return Err(trailer_corruption(
            "repair trailer k_source must be greater than zero for non-empty payloads",
        ));
    }

    let mut cursor = FIXED_HEADER_BYTES;
    let mut symbols = Vec::new();
    while cursor < payload.len() {
        if payload.len() - cursor < LENGTH_PREFIX_BYTES {
            return Err(trailer_corruption(
                "truncated repair symbol length prefix in trailer",
            ));
        }

        let esi = read_u32_le(payload, cursor)?;
        cursor = cursor
            .checked_add(4)
            .ok_or_else(|| trailer_corruption("repair symbol cursor overflow"))?;

        let symbol_len = usize::try_from(read_u32_le(payload, cursor)?)
            .map_err(|_| trailer_corruption("invalid repair symbol length"))?;
        cursor = cursor
            .checked_add(4)
            .ok_or_else(|| trailer_corruption("repair symbol cursor overflow"))?;

        let end = cursor
            .checked_add(symbol_len)
            .ok_or_else(|| trailer_corruption("repair symbol slice overflow"))?;
        if end > payload.len() {
            return Err(trailer_corruption(
                "repair symbol extends past trailer payload",
            ));
        }

        symbols.push(RepairSymbol {
            esi,
            data: payload[cursor..end].to_vec(),
        });
        cursor = end;
    }

    let expected = usize::try_from(header.repair_symbol_count)
        .map_err(|_| trailer_corruption("invalid repair_symbol_count"))?;
    if symbols.len() != expected {
        return Err(trailer_corruption(
            "repair_symbol_count does not match payload",
        ));
    }

    Ok((header, symbols))
}

fn read_magic(bytes: &[u8], offset: usize) -> SearchResult<[u8; 4]> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| trailer_corruption("repair trailer offset overflow"))?;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| trailer_corruption("repair trailer truncated"))?;
    let mut out = [0_u8; 4];
    out.copy_from_slice(slice);
    Ok(out)
}

fn read_u16_le(bytes: &[u8], offset: usize) -> SearchResult<u16> {
    let end = offset
        .checked_add(2)
        .ok_or_else(|| trailer_corruption("repair trailer offset overflow"))?;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| trailer_corruption("repair trailer truncated"))?;
    let mut buf = [0_u8; 2];
    buf.copy_from_slice(slice);
    Ok(u16::from_le_bytes(buf))
}

fn read_u32_le(bytes: &[u8], offset: usize) -> SearchResult<u32> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| trailer_corruption("repair trailer offset overflow"))?;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| trailer_corruption("repair trailer truncated"))?;
    let mut buf = [0_u8; 4];
    buf.copy_from_slice(slice);
    Ok(u32::from_le_bytes(buf))
}

fn read_u64_le(bytes: &[u8], offset: usize) -> SearchResult<u64> {
    let end = offset
        .checked_add(8)
        .ok_or_else(|| trailer_corruption("repair trailer offset overflow"))?;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| trailer_corruption("repair trailer truncated"))?;
    let mut buf = [0_u8; 8];
    buf.copy_from_slice(slice);
    Ok(u64::from_le_bytes(buf))
}

fn trailer_corruption(detail: &str) -> SearchError {
    SearchError::IndexCorrupted {
        path: PathBuf::from("<repair-trailer>"),
        detail: detail.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        RepairSymbol, RepairTrailerHeader, deserialize_repair_trailer, serialize_repair_trailer,
    };

    #[test]
    fn trailer_roundtrip() {
        let header = RepairTrailerHeader {
            symbol_size: 4096,
            k_source: 10,
            source_len: 123_456,
            source_crc32: 0xABCD_EF01,
            repair_symbol_count: 2,
        };
        let symbols = vec![
            RepairSymbol {
                esi: 11,
                data: vec![1, 2, 3],
            },
            RepairSymbol {
                esi: 12,
                data: vec![4, 5, 6, 7],
            },
        ];

        let encoded = serialize_repair_trailer(&header, &symbols).expect("serialize");
        let (decoded_header, decoded_symbols) =
            deserialize_repair_trailer(&encoded).expect("deserialize");

        assert_eq!(decoded_header, header);
        assert_eq!(decoded_symbols, symbols);
    }

    #[test]
    fn crc_mismatch_is_rejected() {
        let header = RepairTrailerHeader {
            symbol_size: 4096,
            k_source: 1,
            source_len: 16,
            source_crc32: 7,
            repair_symbol_count: 1,
        };
        let symbols = vec![RepairSymbol {
            esi: 1,
            data: vec![9, 9, 9],
        }];

        let mut encoded = serialize_repair_trailer(&header, &symbols).expect("serialize");
        encoded[10] ^= 0xFF;
        assert!(deserialize_repair_trailer(&encoded).is_err());
    }

    #[test]
    fn corrupt_magic_is_rejected() {
        let header = RepairTrailerHeader {
            symbol_size: 256,
            k_source: 1,
            source_len: 16,
            source_crc32: 7,
            repair_symbol_count: 1,
        };
        let symbols = vec![RepairSymbol {
            esi: 1,
            data: vec![0; 256],
        }];
        let mut encoded = serialize_repair_trailer(&header, &symbols).expect("serialize");
        // Overwrite magic bytes and fix CRC.
        encoded[0] = b'X';
        encoded[1] = b'Y';
        encoded[2] = b'Z';
        encoded[3] = b'W';
        // Fix the CRC at the end.
        let payload_len = encoded.len() - 4;
        let new_crc = crc32fast::hash(&encoded[..payload_len]);
        encoded[payload_len..].copy_from_slice(&new_crc.to_le_bytes());
        let err = deserialize_repair_trailer(&encoded).unwrap_err();
        assert!(err.to_string().contains("magic"), "error: {err}");
    }

    #[test]
    fn future_version_is_rejected() {
        let header = RepairTrailerHeader {
            symbol_size: 256,
            k_source: 1,
            source_len: 16,
            source_crc32: 7,
            repair_symbol_count: 1,
        };
        let symbols = vec![RepairSymbol {
            esi: 1,
            data: vec![0; 256],
        }];
        let mut encoded = serialize_repair_trailer(&header, &symbols).expect("serialize");
        // Overwrite version (bytes 4..6) to version 99.
        encoded[4..6].copy_from_slice(&99_u16.to_le_bytes());
        // Fix CRC.
        let payload_len = encoded.len() - 4;
        let new_crc = crc32fast::hash(&encoded[..payload_len]);
        encoded[payload_len..].copy_from_slice(&new_crc.to_le_bytes());
        let err = deserialize_repair_trailer(&encoded).unwrap_err();
        assert!(
            err.to_string().contains("version"),
            "error should mention version: {err}"
        );
    }

    #[test]
    fn truncated_trailer_is_rejected() {
        // A trailer that's too short to contain even the fixed header + CRC.
        let too_short = vec![0_u8; 10];
        assert!(deserialize_repair_trailer(&too_short).is_err());
    }

    #[test]
    fn symbol_count_mismatch_is_rejected() {
        let header = RepairTrailerHeader {
            symbol_size: 256,
            k_source: 1,
            source_len: 16,
            source_crc32: 7,
            repair_symbol_count: 2, // claims 2 symbols
        };
        // Only provide 1 symbol — serialize should fail.
        let symbols = vec![RepairSymbol {
            esi: 1,
            data: vec![0; 256],
        }];
        let err = serialize_repair_trailer(&header, &symbols).unwrap_err();
        assert!(
            err.to_string().contains("repair_symbol_count"),
            "error: {err}"
        );
    }

    #[test]
    fn zero_symbol_size_is_rejected() {
        let header = RepairTrailerHeader {
            symbol_size: 0,
            k_source: 1,
            source_len: 16,
            source_crc32: 7,
            repair_symbol_count: 1,
        };
        let symbols = vec![RepairSymbol {
            esi: 1,
            data: vec![9, 9, 9],
        }];
        let encoded = serialize_repair_trailer(&header, &symbols).expect("serialize");
        let err = deserialize_repair_trailer(&encoded).expect_err("must fail");
        assert!(
            err.to_string().contains("symbol_size"),
            "error should mention symbol_size: {err}"
        );
    }

    #[test]
    fn empty_symbols_roundtrip() {
        let header = RepairTrailerHeader {
            symbol_size: 256,
            k_source: 1,
            source_len: 100,
            source_crc32: 42,
            repair_symbol_count: 0,
        };
        let symbols: Vec<RepairSymbol> = Vec::new();
        let encoded = serialize_repair_trailer(&header, &symbols).expect("serialize empty");
        let (decoded_header, decoded_symbols) =
            deserialize_repair_trailer(&encoded).expect("deserialize empty");
        assert_eq!(decoded_header, header);
        assert!(decoded_symbols.is_empty());
    }

    #[test]
    fn zero_k_source_with_nonzero_source_len_is_rejected() {
        let header = RepairTrailerHeader {
            symbol_size: 256,
            k_source: 0,
            source_len: 16,
            source_crc32: 7,
            repair_symbol_count: 1,
        };
        let symbols = vec![RepairSymbol {
            esi: 1,
            data: vec![0; 256],
        }];
        let encoded = serialize_repair_trailer(&header, &symbols).expect("serialize");
        let err = deserialize_repair_trailer(&encoded).expect_err("must fail");
        assert!(
            err.to_string().contains("k_source"),
            "error should mention k_source: {err}"
        );
    }

    #[test]
    fn zero_k_source_with_zero_source_len_is_accepted() {
        let header = RepairTrailerHeader {
            symbol_size: 256,
            k_source: 0,
            source_len: 0,
            source_crc32: crc32fast::hash(&[]),
            repair_symbol_count: 0,
        };
        let symbols: Vec<RepairSymbol> = Vec::new();
        let encoded = serialize_repair_trailer(&header, &symbols).expect("serialize");
        let (decoded_header, decoded_symbols) =
            deserialize_repair_trailer(&encoded).expect("deserialize");
        assert_eq!(decoded_header, header);
        assert!(decoded_symbols.is_empty());
    }

    #[test]
    fn large_symbol_data_roundtrips() {
        let header = RepairTrailerHeader {
            symbol_size: 4096,
            k_source: 1,
            source_len: 4096,
            source_crc32: 0xDEAD_BEEF,
            repair_symbol_count: 1,
        };
        let large_data = vec![0xAB_u8; 4096];
        let symbols = vec![RepairSymbol {
            esi: 1_000_000,
            data: large_data.clone(),
        }];
        let encoded = serialize_repair_trailer(&header, &symbols).expect("serialize");
        let (decoded_header, decoded_symbols) =
            deserialize_repair_trailer(&encoded).expect("deserialize");
        assert_eq!(decoded_header, header);
        assert_eq!(decoded_symbols.len(), 1);
        assert_eq!(decoded_symbols[0].esi, 1_000_000);
        assert_eq!(decoded_symbols[0].data, large_data);
    }

    #[test]
    fn sidecar_path_convention() {
        use crate::file_protector::FileProtector;
        use std::path::PathBuf;

        let fsvi = std::path::Path::new("/data/search/vector.fast.fsvi");
        assert_eq!(
            FileProtector::sidecar_path(fsvi),
            PathBuf::from("/data/search/vector.fast.fsvi.fec")
        );

        let idx = std::path::Path::new("/data/tantivy/segment.idx");
        assert_eq!(
            FileProtector::sidecar_path(idx),
            PathBuf::from("/data/tantivy/segment.idx.fec")
        );
    }
}
