//! Compile-time schema descriptors and stable FSLX schema identifiers.
//!
//! Canonical bytes are hand-written and versioned. They never depend on Rust
//! enum discriminants, debug output, native endianness, serde, or process-seeded
//! hashers.

use std::collections::HashSet;

use xxhash_rust::xxh3::xxh3_64;

use crate::error::QuillError;

const CANONICAL_MAGIC: &[u8] = b"FSLXSCHEMA\0";
const CANONICAL_VERSION: u16 = 1;

/// Analyzer pipelines compiled into Quill.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Analyzer {
    /// Shipping `SimpleTokenizer + LowerCaser`-compatible analyzer.
    FrankensearchDefault,
    /// CASS hyphen/CJK normalization analyzer.
    CassHyphenNormalize,
    /// CASS pre-expanded prefix-field analyzer.
    CassPrefixNormalize,
}

impl Analyzer {
    const fn canonical_tag(self) -> u8 {
        match self {
            Self::FrankensearchDefault => 0,
            Self::CassHyphenNormalize => 1,
            Self::CassPrefixNormalize => 2,
        }
    }
}

/// Monomorphic field shapes understood by Quill's ingest and query paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldKind {
    /// Exact-match text with no tokenization.
    Keyword,
    /// Analyzed text; `positions` controls phrase-query support.
    Text {
        /// Compile-time analyzer pipeline.
        analyzer: Analyzer,
        /// Whether token positions are persisted.
        positions: bool,
    },
    /// Opaque stored bytes with no index entries.
    StoredOnly,
    /// Signed numeric field.
    I64 {
        /// Whether range/query lookup is supported.
        indexed: bool,
        /// Whether a columnar fast field is emitted.
        fast: bool,
    },
    /// Unsigned numeric field.
    ///
    /// `indexed` is explicit because the shipped default `ord` field is fast
    /// but not indexed, while CASS `msg_idx` is indexed but not fast.
    U64 {
        /// Whether range/query lookup is supported.
        indexed: bool,
        /// Whether a columnar fast field is emitted.
        fast: bool,
    },
}

impl FieldKind {
    const fn canonical_tag(self) -> u8 {
        match self {
            Self::Keyword => 0,
            Self::Text { .. } => 1,
            Self::StoredOnly => 2,
            Self::I64 { .. } => 3,
            Self::U64 { .. } => 4,
        }
    }

    /// Whether this numeric field owns a persisted NUMERIC column.
    pub(crate) const fn has_numeric_column(self) -> bool {
        match self {
            Self::I64 { indexed, fast } | Self::U64 { indexed, fast } => indexed || fast,
            Self::Keyword | Self::Text { .. } | Self::StoredOnly => false,
        }
    }
}

/// One field in a compile-time schema table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldDescriptor {
    /// Dense stable field identifier, starting at zero.
    pub id: u16,
    /// Stable query/materialization name.
    pub name: &'static str,
    /// Monomorphic storage/index shape.
    pub kind: FieldKind,
    /// Whether original field bytes are stored in FSLX STOREDMETA.
    pub stored: bool,
}

/// A complete schema compiled into this Quill build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchemaDescriptor {
    /// Diagnostic name. It is deliberately excluded from the durable hash.
    pub name: &'static str,
    /// Strictly ID-ordered field table.
    pub fields: &'static [FieldDescriptor],
}

impl SchemaDescriptor {
    /// Validate the invariants required by the canonical encoder.
    ///
    /// # Errors
    ///
    /// Returns [`QuillError::InvalidDescriptor`] for empty names/tables,
    /// non-contiguous IDs, duplicate field names, or lengths wider than u16.
    pub fn validate(self) -> Result<(), QuillError> {
        if self.name.is_empty() {
            return Err(invalid_descriptor("descriptor name must not be empty"));
        }
        if self.fields.is_empty() {
            return Err(invalid_descriptor(
                "descriptor must contain at least one field",
            ));
        }
        if self.fields.len() > usize::from(u16::MAX) {
            return Err(invalid_descriptor(
                "descriptor has more than u16::MAX fields",
            ));
        }

        let mut names = HashSet::with_capacity(self.fields.len());
        for (index, field) in self.fields.iter().enumerate() {
            let expected_id = u16::try_from(index)
                .map_err(|_| invalid_descriptor("field index does not fit in u16"))?;
            if field.id != expected_id {
                return Err(invalid_descriptor(format!(
                    "field {} has id {}, expected {expected_id}",
                    field.name, field.id
                )));
            }
            if field.name.is_empty() {
                return Err(invalid_descriptor(format!(
                    "field id {} has an empty name",
                    field.id
                )));
            }
            if field.name.len() > usize::from(u16::MAX) {
                return Err(invalid_descriptor(format!(
                    "field {} name is wider than u16",
                    field.name
                )));
            }
            if !names.insert(field.name) {
                return Err(invalid_descriptor(format!(
                    "duplicate field name {}",
                    field.name
                )));
            }
            match field.kind {
                FieldKind::StoredOnly if !field.stored => {
                    return Err(invalid_descriptor(format!(
                        "stored-only field {} must set stored=true",
                        field.name
                    )));
                }
                FieldKind::I64 {
                    indexed: false,
                    fast: false,
                }
                | FieldKind::U64 {
                    indexed: false,
                    fast: false,
                } if !field.stored => {
                    return Err(invalid_descriptor(format!(
                        "numeric field {} must be indexed, fast, or stored",
                        field.name
                    )));
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Emit the versioned canonical little-endian encoding hashed by FSLX.
    ///
    /// The diagnostic descriptor name is not encoded. Field IDs, names, kinds,
    /// kind payloads, and storage bits all affect compatibility.
    ///
    /// # Errors
    ///
    /// Returns an invalid-descriptor error when [`Self::validate`] fails.
    pub fn canonical_encoding(self) -> Result<Vec<u8>, QuillError> {
        self.validate()?;
        let mut bytes = Vec::with_capacity(CANONICAL_MAGIC.len() + 4 + self.fields.len() * 12);
        bytes.extend_from_slice(CANONICAL_MAGIC);
        bytes.extend_from_slice(&CANONICAL_VERSION.to_le_bytes());
        let field_count = u16::try_from(self.fields.len())
            .map_err(|_| invalid_descriptor("field count does not fit in u16"))?;
        bytes.extend_from_slice(&field_count.to_le_bytes());

        for field in self.fields {
            bytes.extend_from_slice(&field.id.to_le_bytes());
            let name_len = u16::try_from(field.name.len())
                .map_err(|_| invalid_descriptor("field name length does not fit in u16"))?;
            bytes.extend_from_slice(&name_len.to_le_bytes());
            bytes.extend_from_slice(field.name.as_bytes());
            bytes.push(field.kind.canonical_tag());
            match field.kind {
                FieldKind::Keyword | FieldKind::StoredOnly => {}
                FieldKind::Text {
                    analyzer,
                    positions,
                } => {
                    bytes.push(analyzer.canonical_tag());
                    bytes.push(u8::from(positions));
                }
                FieldKind::I64 { indexed, fast } | FieldKind::U64 { indexed, fast } => {
                    bytes.push(u8::from(indexed));
                    bytes.push(u8::from(fast));
                }
            }
            bytes.push(u8::from(field.stored));
        }
        Ok(bytes)
    }

    /// Compute the stable xxh3-64 identifier persisted in FSLX headers.
    ///
    /// # Errors
    ///
    /// Returns an invalid-descriptor error when canonical encoding fails.
    pub fn schema_id(self) -> Result<u64, QuillError> {
        self.canonical_encoding().map(|bytes| xxh3_64(&bytes))
    }
}

fn invalid_descriptor(detail: impl Into<String>) -> QuillError {
    QuillError::InvalidDescriptor {
        detail: detail.into(),
    }
}

const DEFAULT_FIELDS: [FieldDescriptor; 5] = [
    FieldDescriptor {
        id: 0,
        name: "id",
        kind: FieldKind::Keyword,
        stored: true,
    },
    FieldDescriptor {
        id: 1,
        name: "content",
        kind: FieldKind::Text {
            analyzer: Analyzer::FrankensearchDefault,
            positions: true,
        },
        stored: true,
    },
    FieldDescriptor {
        id: 2,
        name: "title",
        kind: FieldKind::Text {
            analyzer: Analyzer::FrankensearchDefault,
            positions: true,
        },
        stored: true,
    },
    FieldDescriptor {
        id: 3,
        name: "metadata_json",
        kind: FieldKind::StoredOnly,
        stored: true,
    },
    FieldDescriptor {
        id: 4,
        name: "ord",
        kind: FieldKind::U64 {
            indexed: false,
            fast: true,
        },
        stored: true,
    },
];

/// Shipping five-field schema mirroring `frankensearch-lexical::build_schema`.
pub const DEFAULT_SCHEMA: SchemaDescriptor = SchemaDescriptor {
    name: "frankensearch-default-v1",
    fields: &DEFAULT_FIELDS,
};

const FSFS_CHUNK_FIELDS: [FieldDescriptor; 8] = [
    FieldDescriptor {
        id: 0,
        name: "id",
        kind: FieldKind::Keyword,
        stored: true,
    },
    FieldDescriptor {
        id: 1,
        name: "parent_id",
        kind: FieldKind::Keyword,
        stored: true,
    },
    FieldDescriptor {
        id: 2,
        name: "revision",
        kind: FieldKind::U64 {
            indexed: false,
            fast: true,
        },
        stored: true,
    },
    FieldDescriptor {
        id: 3,
        name: "chunk_ordinal",
        kind: FieldKind::U64 {
            indexed: false,
            fast: true,
        },
        stored: true,
    },
    FieldDescriptor {
        id: 4,
        name: "byte_start",
        kind: FieldKind::U64 {
            indexed: false,
            fast: true,
        },
        stored: true,
    },
    FieldDescriptor {
        id: 5,
        name: "byte_end",
        kind: FieldKind::U64 {
            indexed: false,
            fast: true,
        },
        stored: true,
    },
    FieldDescriptor {
        id: 6,
        name: "content",
        kind: FieldKind::Text {
            analyzer: Analyzer::FrankensearchDefault,
            positions: true,
        },
        stored: false,
    },
    FieldDescriptor {
        id: 7,
        name: "token_count",
        kind: FieldKind::U64 {
            indexed: false,
            fast: true,
        },
        stored: true,
    },
];

/// FSFS chunk-policy schema.
///
/// `id` is a deterministic unique chunk ID; `parent_id` and `revision` retain
/// parent-level replace/delete semantics. Content is hydrated from canonical
/// storage rather than duplicated in STOREDMETA.
pub const FSFS_CHUNK_SCHEMA: SchemaDescriptor = SchemaDescriptor {
    name: "frankensearch-fsfs-chunk-v1",
    fields: &FSFS_CHUNK_FIELDS,
};

const CASS_SEMANTIC_FIELDS: [FieldDescriptor; 15] = [
    FieldDescriptor {
        id: 0,
        name: "agent",
        kind: FieldKind::Keyword,
        stored: true,
    },
    FieldDescriptor {
        id: 1,
        name: "workspace",
        kind: FieldKind::Keyword,
        stored: true,
    },
    FieldDescriptor {
        id: 2,
        name: "workspace_original",
        kind: FieldKind::StoredOnly,
        stored: true,
    },
    FieldDescriptor {
        id: 3,
        name: "source_path",
        kind: FieldKind::StoredOnly,
        stored: true,
    },
    FieldDescriptor {
        id: 4,
        name: "msg_idx",
        kind: FieldKind::U64 {
            indexed: true,
            fast: false,
        },
        stored: true,
    },
    FieldDescriptor {
        id: 5,
        name: "created_at",
        kind: FieldKind::I64 {
            indexed: true,
            fast: true,
        },
        stored: true,
    },
    FieldDescriptor {
        id: 6,
        name: "title",
        kind: FieldKind::Text {
            analyzer: Analyzer::CassHyphenNormalize,
            positions: true,
        },
        stored: true,
    },
    FieldDescriptor {
        id: 7,
        name: "content",
        kind: FieldKind::Text {
            analyzer: Analyzer::CassHyphenNormalize,
            positions: true,
        },
        stored: false,
    },
    FieldDescriptor {
        id: 8,
        name: "title_prefix",
        kind: FieldKind::Text {
            analyzer: Analyzer::CassPrefixNormalize,
            positions: false,
        },
        stored: false,
    },
    FieldDescriptor {
        id: 9,
        name: "content_prefix",
        kind: FieldKind::Text {
            analyzer: Analyzer::CassPrefixNormalize,
            positions: false,
        },
        stored: false,
    },
    FieldDescriptor {
        id: 10,
        name: "preview",
        kind: FieldKind::StoredOnly,
        stored: true,
    },
    FieldDescriptor {
        id: 11,
        name: "source_id",
        kind: FieldKind::Keyword,
        stored: true,
    },
    FieldDescriptor {
        id: 12,
        name: "origin_kind",
        kind: FieldKind::Keyword,
        stored: true,
    },
    FieldDescriptor {
        id: 13,
        name: "origin_host",
        kind: FieldKind::Keyword,
        stored: true,
    },
    FieldDescriptor {
        id: 14,
        name: "conversation_id",
        kind: FieldKind::I64 {
            indexed: false,
            fast: false,
        },
        stored: true,
    },
];

/// CASS semantic field set, independent of CASS's legacy Tantivy disk format.
pub const CASS_SEMANTIC_SCHEMA: SchemaDescriptor = SchemaDescriptor {
    name: "frankensearch-cass-semantic-v1",
    fields: &CASS_SEMANTIC_FIELDS,
};

#[cfg(test)]
mod tests {
    use super::*;

    const PINNED_DEFAULT_SCHEMA_ID: u64 = 0xa312_ebf6_d136_07a5;
    const PINNED_FSFS_CHUNK_SCHEMA_ID: u64 = 0xe1c8_4ac7_e5e0_c4a1;
    const PINNED_CASS_SEMANTIC_SCHEMA_ID: u64 = 0xc2a2_d236_2aa9_e14f;

    #[test]
    fn canonical_encoding_fixture_is_pinned() {
        const FIELDS: [FieldDescriptor; 1] = [FieldDescriptor {
            id: 0,
            name: "id",
            kind: FieldKind::Keyword,
            stored: true,
        }];
        let descriptor = SchemaDescriptor {
            name: "diagnostic-only",
            fields: &FIELDS,
        };
        assert_eq!(
            descriptor.canonical_encoding().unwrap(),
            [
                b'F', b'S', b'L', b'X', b'S', b'C', b'H', b'E', b'M', b'A', 0, 1, 0, 1, 0, 0, 0, 2,
                0, b'i', b'd', 0, 1,
            ]
        );
    }

    #[test]
    fn shipped_schema_ids_are_pinned_and_unique() {
        let ids = [
            DEFAULT_SCHEMA.schema_id().unwrap(),
            FSFS_CHUNK_SCHEMA.schema_id().unwrap(),
            CASS_SEMANTIC_SCHEMA.schema_id().unwrap(),
        ];
        assert_eq!(
            ids,
            [
                PINNED_DEFAULT_SCHEMA_ID,
                PINNED_FSFS_CHUNK_SCHEMA_ID,
                PINNED_CASS_SEMANTIC_SCHEMA_ID,
            ]
        );
        assert_ne!(ids[0], ids[1]);
        assert_ne!(ids[0], ids[2]);
        assert_ne!(ids[1], ids[2]);
    }

    #[test]
    fn shipped_tables_validate_and_have_expected_widths() {
        for descriptor in [DEFAULT_SCHEMA, FSFS_CHUNK_SCHEMA, CASS_SEMANTIC_SCHEMA] {
            descriptor.validate().unwrap();
        }
        assert_eq!(DEFAULT_SCHEMA.fields.len(), 5);
        assert_eq!(FSFS_CHUNK_SCHEMA.fields.len(), 8);
        assert_eq!(CASS_SEMANTIC_SCHEMA.fields.len(), 15);
        assert_eq!(
            DEFAULT_SCHEMA
                .fields
                .iter()
                .map(|field| field.name)
                .collect::<Vec<_>>(),
            ["id", "content", "title", "metadata_json", "ord"]
        );
    }

    #[test]
    fn u64_flags_distinguish_default_ord_from_cass_msg_idx() {
        assert_eq!(
            DEFAULT_SCHEMA.fields[4].kind,
            FieldKind::U64 {
                indexed: false,
                fast: true,
            }
        );
        assert_eq!(
            CASS_SEMANTIC_SCHEMA.fields[4].kind,
            FieldKind::U64 {
                indexed: true,
                fast: false,
            }
        );
    }

    #[test]
    fn diagnostic_descriptor_name_does_not_change_schema_id() {
        let renamed = SchemaDescriptor {
            name: "alias",
            fields: DEFAULT_SCHEMA.fields,
        };
        assert_eq!(
            renamed.schema_id().unwrap(),
            DEFAULT_SCHEMA.schema_id().unwrap()
        );
    }

    #[test]
    fn validation_rejects_fields_that_retain_no_data() {
        const STORED_ONLY_FIELDS: [FieldDescriptor; 1] = [FieldDescriptor {
            id: 0,
            name: "lost",
            kind: FieldKind::StoredOnly,
            stored: false,
        }];
        const NUMERIC_FIELDS: [FieldDescriptor; 1] = [FieldDescriptor {
            id: 0,
            name: "lost",
            kind: FieldKind::U64 {
                indexed: false,
                fast: false,
            },
            stored: false,
        }];
        for fields in [&STORED_ONLY_FIELDS[..], &NUMERIC_FIELDS[..]] {
            let descriptor = SchemaDescriptor {
                name: "invalid",
                fields,
            };
            assert!(matches!(
                descriptor.validate(),
                Err(QuillError::InvalidDescriptor { .. })
            ));
        }
    }
}
