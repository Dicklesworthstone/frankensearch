use std::collections::BTreeSet;

use asupersync::Cx;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::GauntletError;

const ORACLE_VERSION_CONTRACT_JSON: &str = include_str!("../oracle-version-contract.json");
const Q1_FIXTURE_CATALOG_JSON: &str = include_str!("../fixtures/q1-obligations.json");
const ORACLE_V2_TANTIVY_VERSION: &str = "0.26.1";
const ORACLE_V2_TANTIVY_SOURCE: &str = "registry+https://github.com/rust-lang/crates.io-index";
const ORACLE_V2_TANTIVY_CHECKSUM_SHA256: &str =
    "edde6a10743fff00a4e1a8c9ef020bf5f3cbad301b7d2d39f2b07f123c4eac07";
const ORACLE_V2_LEXICAL_PACKAGE: &str = "frankensearch-lexical";
const ORACLE_V2_LEXICAL_PACKAGE_VERSION: &str = "0.2.1";
const ORACLE_V2_LEXICAL_CONTRACT_AUDIT_REVISION: &str = "062a5e5b2d41653b1c8b07888eda1a765e421f49";
const ORACLE_V3_TANTIVY_VERSION: &str = "0.27.0";
const ORACLE_V3_TANTIVY_SOURCE: &str = "git+https://github.com/quickwit-oss/tantivy?rev=1f32c1a8af0eb9d68bd3f5576caf20941364b657#1f32c1a8af0eb9d68bd3f5576caf20941364b657";
const ORACLE_V3_TANTIVY_CHECKSUM_SHA256: &str =
    "db62510712cc8d0a776d35af8e5bd18565b0d3bc466299202d775ae75aa11aa1";
const ORACLE_V3_LEXICAL_PACKAGE: &str = "frankensearch-lexical";
const ORACLE_V3_LEXICAL_PACKAGE_VERSION: &str = "0.2.1";
const ORACLE_V3_LEXICAL_CONTRACT_AUDIT_REVISION: &str = "062a5e5b2d41653b1c8b07888eda1a765e421f49";
// v4 (gh#416): crates.io publication forbids git dependencies, so the oracle
// moved from the v3 git 0.27-dev pin back to the same registry 0.26.1 artifact
// the frozen v2 record described (identical checksum), now wrapped by
// frankensearch-lexical 0.2.2. The v3 line is retained decode-only, exactly
// like v2: historical artifacts stay verifiable, but nothing new admits under
// the git dependency.
pub const CURRENT_ORACLE_TANTIVY_VERSION: &str = "0.26.1";
const ORACLE_V4_TANTIVY_SOURCE: &str = "registry+https://github.com/rust-lang/crates.io-index";
const ORACLE_V4_TANTIVY_CHECKSUM_SHA256: &str =
    "edde6a10743fff00a4e1a8c9ef020bf5f3cbad301b7d2d39f2b07f123c4eac07";
const ORACLE_V4_LEXICAL_PACKAGE: &str = "frankensearch-lexical";
const ORACLE_V4_LEXICAL_PACKAGE_VERSION: &str = "0.2.2";
const ORACLE_V4_LEXICAL_CONTRACT_AUDIT_REVISION: &str = "062a5e5b2d41653b1c8b07888eda1a765e421f49";
const LOCKED_TANTIVY_VERSION: &str = env!("QUILL_ORACLE_TANTIVY_VERSION");
const LOCKED_TANTIVY_SOURCE: &str = env!("QUILL_ORACLE_TANTIVY_SOURCE");
const LOCKED_TANTIVY_CHECKSUM_SHA256: &str = env!("QUILL_ORACLE_TANTIVY_CHECKSUM_SHA256");
const ORACLE_V2_DEPENDENCY_CONTRACT_HASH_DOMAIN: &[u8] =
    b"frankensearch/quill/oracle-dependency-contract/v2\0";
const ORACLE_V3_DEPENDENCY_CONTRACT_HASH_DOMAIN: &[u8] =
    b"frankensearch/quill/oracle-dependency-contract/v3\0";
const ORACLE_V4_DEPENDENCY_CONTRACT_HASH_DOMAIN: &[u8] =
    b"frankensearch/quill/oracle-dependency-contract/v4\0";
/// Exact `frankensearch-lexical` crate version resolved by this build.
pub const FRANKENSEARCH_LEXICAL_CRATE_VERSION: &str = env!("FRANKENSEARCH_LEXICAL_CRATE_VERSION");
const QUIVER_DIFFERENTIAL_FIXTURE_ID: &str = "quiver-postings-bitpack-scalar-wide-v1";
const Q1_FIXTURE_CATALOG_SHA256: [u8; 32] = [
    0x16, 0xd4, 0x4d, 0xf2, 0x13, 0x28, 0x25, 0xa5, 0x06, 0x7e, 0x49, 0x5c, 0xb5, 0x3d, 0x03, 0x38,
    0xe5, 0xca, 0x5a, 0xf1, 0xc8, 0xe7, 0xaa, 0x07, 0x4b, 0x80, 0x94, 0x8c, 0xf3, 0x80, 0x00, 0x9c,
];

/// Committed provenance contract for the shipping Tantivy oracle adapter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OracleVersionContract {
    pub schema_version: u32,
    pub tantivy_version: String,
    pub tantivy_source: String,
    pub tantivy_checksum_sha256: String,
    pub lexical_package: String,
    pub lexical_package_version: String,
    /// Historical revision where the wrapper contract was independently
    /// audited. This is not the source identity of the executing producer.
    pub lexical_contract_audit_revision: String,
}

impl OracleVersionContract {
    /// Validate the self-contained v3 dependency record without consulting the
    /// current checkout, executable, manifest, or lockfile.
    ///
    /// # Errors
    ///
    /// Returns an error when any v3 package, source, checksum, or historical
    /// audit pin is malformed or changed.
    pub fn validate_stored_structure(&self) -> Result<(), GauntletError> {
        self.validate_structure(false)
    }

    /// Validate an exact historical dependency record while inspecting a
    /// committed, decode-only witness. This admits only the frozen v2 and v3
    /// records in addition to the current v4 record; it never authorizes
    /// creation or admission under an old dependency.
    pub(crate) fn validate_retained_structure(&self) -> Result<(), GauntletError> {
        self.validate_structure(true)
    }

    fn validate_structure(&self, permit_retained_historical: bool) -> Result<(), GauntletError> {
        let matches_v2 = self.schema_version == 2
            && self.tantivy_version == ORACLE_V2_TANTIVY_VERSION
            && self.tantivy_source == ORACLE_V2_TANTIVY_SOURCE
            && self.tantivy_checksum_sha256 == ORACLE_V2_TANTIVY_CHECKSUM_SHA256
            && self.lexical_package == ORACLE_V2_LEXICAL_PACKAGE
            && self.lexical_package_version == ORACLE_V2_LEXICAL_PACKAGE_VERSION
            && self.lexical_contract_audit_revision == ORACLE_V2_LEXICAL_CONTRACT_AUDIT_REVISION;
        let matches_v3 = self.schema_version == 3
            && self.tantivy_version == ORACLE_V3_TANTIVY_VERSION
            && self.tantivy_source == ORACLE_V3_TANTIVY_SOURCE
            && self.tantivy_checksum_sha256 == ORACLE_V3_TANTIVY_CHECKSUM_SHA256
            && self.lexical_package == ORACLE_V3_LEXICAL_PACKAGE
            && self.lexical_package_version == ORACLE_V3_LEXICAL_PACKAGE_VERSION
            && self.lexical_contract_audit_revision == ORACLE_V3_LEXICAL_CONTRACT_AUDIT_REVISION;
        let matches_v4 = self.schema_version == 4
            && self.tantivy_version == CURRENT_ORACLE_TANTIVY_VERSION
            && self.tantivy_source == ORACLE_V4_TANTIVY_SOURCE
            && self.tantivy_checksum_sha256 == ORACLE_V4_TANTIVY_CHECKSUM_SHA256
            && self.lexical_package == ORACLE_V4_LEXICAL_PACKAGE
            && self.lexical_package_version == ORACLE_V4_LEXICAL_PACKAGE_VERSION
            && self.lexical_contract_audit_revision == ORACLE_V4_LEXICAL_CONTRACT_AUDIT_REVISION;
        if !(matches_v4 || permit_retained_historical && (matches_v2 || matches_v3))
            || !is_lower_hex(&self.tantivy_checksum_sha256, 64)
            || !is_lower_hex(&self.lexical_contract_audit_revision, 40)
        {
            return Err(GauntletError::InvalidContract {
                reason: "stored oracle dependency contract is malformed or internally inconsistent"
                    .to_owned(),
            });
        }
        Ok(())
    }

    /// Validate that this v4 record describes the exact dependency resolved by
    /// the current producer build.
    pub(crate) fn validate_current_dependency(&self) -> Result<(), GauntletError> {
        self.validate_stored_structure()?;
        if self.tantivy_version != LOCKED_TANTIVY_VERSION
            || self.tantivy_source != LOCKED_TANTIVY_SOURCE
            || self.tantivy_checksum_sha256 != LOCKED_TANTIVY_CHECKSUM_SHA256
            || self.lexical_package_version != FRANKENSEARCH_LEXICAL_CRATE_VERSION
        {
            return Err(GauntletError::InvalidContract {
                reason: "embedded oracle dependency contract does not match the exact Cargo.lock resolution"
                    .to_owned(),
            });
        }
        #[cfg(feature = "tantivy-oracle")]
        if self.lexical_package_version
            != frankensearch_lexical::FRANKENSEARCH_LEXICAL_CRATE_VERSION
        {
            return Err(GauntletError::InvalidContract {
                reason: "embedded oracle wrapper version does not match the linked frankensearch-lexical crate"
                    .to_owned(),
            });
        }
        Ok(())
    }

    /// Domain-separated identity of the self-contained dependency record.
    pub(crate) fn identity_sha256(&self) -> Result<String, GauntletError> {
        self.validate_retained_structure()?;
        let hash_domain = match self.schema_version {
            2 => ORACLE_V2_DEPENDENCY_CONTRACT_HASH_DOMAIN,
            3 => ORACLE_V3_DEPENDENCY_CONTRACT_HASH_DOMAIN,
            4 => ORACLE_V4_DEPENDENCY_CONTRACT_HASH_DOMAIN,
            _ => unreachable!("validated oracle dependency contract has an unknown schema"),
        };
        let mut hasher = Sha256::new();
        hasher.update(hash_domain);
        hasher.update(serde_json::to_vec(self)?);
        Ok(lower_hex(&hasher.finalize()))
    }
}

/// One Q1 obligation with its executable enforcement surface.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Q1Fixture {
    pub id: String,
    pub status: String,
    pub enforcement: String,
    pub assertion: String,
}

/// Committed catalog of executable Q1 obligations and live internal differentials.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Q1FixtureCatalog {
    pub schema_version: u32,
    pub source_contract: String,
    pub fixtures: Vec<Q1Fixture>,
    pub internal_differentials: Vec<InternalDifferentialFixture>,
}

/// One registered same-engine kernel differential.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InternalDifferentialFixture {
    pub id: String,
    pub status: String,
    pub source_contract: String,
    pub enforcement: String,
    pub assertion: String,
}

/// Parse and validate the embedded oracle version contract.
///
/// # Errors
///
/// Returns an error when a pin is absent, malformed, or no longer exact.
pub fn oracle_version_contract() -> Result<OracleVersionContract, GauntletError> {
    let contract: OracleVersionContract = serde_json::from_str(ORACLE_VERSION_CONTRACT_JSON)?;
    contract.validate_current_dependency()?;
    Ok(contract)
}

/// Parse and validate the complete Q1 ownership catalog and live differentials.
///
/// # Errors
///
/// Returns an error when IDs are missing, duplicated, reordered, assigned an
/// unknown status, or when any normative Q1 obligation is not executable.
pub fn q1_fixture_catalog() -> Result<Q1FixtureCatalog, GauntletError> {
    let catalog: Q1FixtureCatalog = serde_json::from_str(Q1_FIXTURE_CATALOG_JSON)?;
    let catalog_hash = Sha256::digest(Q1_FIXTURE_CATALOG_JSON.as_bytes());
    const EXPECTED: [&str; 8] = [
        "Q1-OB1", "Q1-OB2a", "Q1-OB2b", "Q1-OB2c", "Q1-OB3", "Q1-OB4", "Q1-OB5", "Q1-OB6",
    ];
    const EXPECTED_STATUSES: [&str; 8] = [
        "executable",
        "executable",
        "executable",
        "executable",
        "executable",
        "executable",
        "executable",
        "executable",
    ];
    let ids = catalog
        .fixtures
        .iter()
        .map(|fixture| fixture.id.as_str())
        .collect::<Vec<_>>();
    let statuses = catalog
        .fixtures
        .iter()
        .map(|fixture| fixture.status.as_str())
        .collect::<Vec<_>>();
    let unique = ids.iter().copied().collect::<BTreeSet<_>>();
    let internal = catalog.internal_differentials.as_slice();
    // ubs:ignore — this SHA-256 pins public fixture bytes, not a secret or authenticator.
    if catalog_hash.as_slice() != Q1_FIXTURE_CATALOG_SHA256
        || catalog.schema_version != 1
        || catalog.source_contract
            != "docs/contracts/quill-q1-docid-discipline.md#6-obligations-each-becomes-a-gauntlet-fixture"
        || ids != EXPECTED
        || statuses != EXPECTED_STATUSES
        || unique.len() != EXPECTED.len()
        || catalog.fixtures.iter().any(|fixture| {
            !matches!(fixture.status.as_str(), "executable" | "deferred")
                || fixture.enforcement.is_empty()
                || fixture.assertion.is_empty()
        })
        || internal.len() != 1
        || internal[0].id != QUIVER_DIFFERENTIAL_FIXTURE_ID
        || internal[0].status != "executable"
        || internal[0].source_contract != "docs/contracts/fslx-format-registry.md#52-postings"
        || internal[0].enforcement.is_empty()
        || internal[0].assertion.is_empty()
    {
        return Err(GauntletError::InvalidContract {
            reason: "fixture catalog must contain eight ordered executable Q1 rows plus the executable Quiver differential"
                .to_owned(),
        });
    }
    Ok(catalog)
}

/// Execute the bounded Q1 sub-fixtures owned by G1a and E3.5.
///
/// This exercises a real three-segment concat path, rejects a manifest-order
/// violation, binds and verifies the merged successor, and compares live
/// identity and query behavior before and after publication. The exhaustive
/// raw-codec shapes named by the catalog remain inline Quiver fixtures so this
/// runner does not duplicate the 1-through-127 partial-block sweep.
///
/// # Errors
///
/// Returns a contract error if interval validation, merge/reopen equivalence,
/// identity rebuilding, query invariance, lease disjointness, boundary
/// cutting, burn accounting, or watermark monotonicity regresses.
pub async fn run_q1_live_fixtures(cx: &Cx) -> Result<Vec<String>, GauntletError> {
    use frankensearch_quill::scribe::{DOCID_LEASE_BLOCK, DocIdAllocator};
    use frankensearch_quill::{
        CURRENT_ENGINE_VERSION, DEFAULT_SCHEMA, Manifest, ManifestSegment, TombstoneSet,
    };

    let schema_id = DEFAULT_SCHEMA
        .schema_id()
        .map_err(|error| GauntletError::InvalidContract {
            reason: format!("default schema is invalid while running Q1 fixtures: {error}"),
        })?;
    let segment = |segment_id, seal_seq, docid_lo, docid_hi| ManifestSegment {
        segment_id,
        seal_seq,
        file_len: 1,
        file_xxh3: segment_id,
        docid_lo,
        docid_hi,
        doc_count: 1,
        tombstones: TombstoneSet::new(),
    };
    let mut manifest = Manifest {
        generation: 2,
        docid_high_watermark: 2 * DOCID_LEASE_BLOCK,
        schema_id,
        engine_version: CURRENT_ENGINE_VERSION,
        flags: 0,
        last_publish_unix_s: 0,
        segments: vec![
            segment(1, 1, 0, 1),
            segment(2, 2, DOCID_LEASE_BLOCK, DOCID_LEASE_BLOCK + 1),
        ],
        field_stats: Vec::new(),
    };
    manifest
        .validate()
        .map_err(|error| GauntletError::InvalidContract {
            reason: format!("Q1-OB1 valid disjoint intervals failed: {error}"),
        })?;
    manifest.segments[1].docid_lo = 0;
    if manifest.validate().is_ok() {
        return Err(GauntletError::InvalidContract {
            reason: "Q1-OB1 admitted overlapping segment intervals".to_owned(),
        });
    }

    let mut allocator = DocIdAllocator::open(0, 2).map_err(q1_allocator_error)?;
    let first = allocator.alloc_batch(0, 10).map_err(q1_allocator_error)?;
    let second = allocator.alloc_batch(1, 10).map_err(q1_allocator_error)?;
    if first.spans()[0].global_end() > second.spans()[0].global_first() {
        return Err(GauntletError::InvalidContract {
            reason: "Q1-OB5 granted overlapping concurrent leases".to_owned(),
        });
    }
    let lease_batch =
        u32::try_from(DOCID_LEASE_BLOCK).map_err(|_| GauntletError::InvalidContract {
            reason: "Q1 lease block does not fit the batch-allocation contract".to_owned(),
        })?;
    let boundary = allocator
        .alloc_batch(0, lease_batch)
        .map_err(q1_allocator_error)?;
    if !boundary.crossed_lease() {
        return Err(GauntletError::InvalidContract {
            reason: "Q1-OB5 did not cut an allocation at lease exhaustion".to_owned(),
        });
    }
    let burn = allocator.end_session();
    if burn.total_burned == 0 || burn.final_watermark < 2 * DOCID_LEASE_BLOCK {
        return Err(GauntletError::InvalidContract {
            reason: "Q1-OB5 failed burn accounting or watermark monotonicity".to_owned(),
        });
    }
    let mut next = DocIdAllocator::open(burn.final_watermark, 1).map_err(q1_allocator_error)?;
    let next_batch = next.alloc_batch(0, 1).map_err(q1_allocator_error)?;
    if next_batch.spans()[0].global_first() < burn.final_watermark {
        return Err(GauntletError::InvalidContract {
            reason: "Q1-OB5 reused a burned document-id tail".to_owned(),
        });
    }

    let mut completed = vec!["Q1-OB1/manifest-overlap".to_owned()];
    completed.extend(run_q1_concat_merge_fixture(cx).await?);
    completed.push("Q1-OB5/allocator-boundary".to_owned());
    Ok(completed)
}

async fn run_q1_concat_merge_fixture(cx: &Cx) -> Result<Vec<String>, GauntletError> {
    use frankensearch_core::IndexableDocument;
    use frankensearch_quill::{
        ConcatMergeError, KeeperError, QuillConfig, QuillIndex, QuillIndexError, SectionKind,
    };

    let index = QuillIndex::in_memory(QuillConfig {
        deterministic_ingest: true,
        ..QuillConfig::default()
    })
    .map_err(|error| q1_merge_error("create deterministic index", error))?;
    let batches = [
        vec![
            IndexableDocument::new("merge-a1", "shared alpha exact phrase")
                .with_title("Alpha segment")
                .with_metadata("batch", "a"),
            IndexableDocument::new("merge-a2", "shared common anchor")
                .with_title("Common alpha")
                .with_metadata("batch", "a"),
        ],
        vec![
            IndexableDocument::new("merge-b1", "shared beta exact phrase")
                .with_title("Beta segment")
                .with_metadata("batch", "b"),
            IndexableDocument::new("merge-b2", "common beta anchor")
                .with_title("Common beta")
                .with_metadata("batch", "b"),
        ],
        vec![
            IndexableDocument::new("merge-c1", "shared gamma exact phrase")
                .with_title("Gamma segment")
                .with_metadata("batch", "c"),
            IndexableDocument::new("merge-c2", "common gamma anchor")
                .with_title("Common gamma")
                .with_metadata("batch", "c"),
        ],
        vec![
            IndexableDocument::new("merge-d1", "shared delta exact phrase")
                .with_title("Delta segment")
                .with_metadata("batch", "d"),
            IndexableDocument::new("merge-d2", "common delta anchor")
                .with_title("Common delta")
                .with_metadata("batch", "d"),
        ],
    ];
    for (batch_index, documents) in batches.iter().enumerate() {
        index
            .index_documents(cx, documents)
            .await
            .map_err(|error| q1_merge_error("index source batch", error))?;
        index
            .commit(cx)
            .await
            .map_err(|error| q1_merge_error("commit source batch", error))?;
        if batch_index == 1 {
            let first_stage_source_ids = index
                .snapshot()?
                .segments()
                .iter()
                .map(|segment| segment.manifest().segment_id)
                .collect::<Vec<_>>();
            if first_stage_source_ids.len() != 2 {
                return Err(GauntletError::InvalidContract {
                    reason: format!(
                        "Q1 E3.5 expected two first-stage source segments, got {}",
                        first_stage_source_ids.len()
                    ),
                });
            }
            let first_stage_output_segment_id = [
                0xe34f_0000_0000_0001,
                0xe34f_0000_0000_0002,
                0xe34f_0000_0000_0003,
                0xe34f_0000_0000_0004,
            ]
            .into_iter()
            .find(|candidate| !first_stage_source_ids.contains(candidate))
            .ok_or_else(|| GauntletError::InvalidContract {
                reason: "Q1 E3.5 could not choose a first-stage fixture segment id".to_owned(),
            })?;
            // The explicit first-stage merge retires the active ingest lease.
            // Appending the remaining batches from the successor lease then
            // creates a real burned-tail gap inside the final merge hull.
            index
                .concat_merge(
                    cx,
                    &first_stage_source_ids,
                    first_stage_output_segment_id,
                    1_700_000_034,
                )
                .await
                .map_err(|error| q1_merge_error("publish first-stage concat merge", error))?;
        }
    }

    let source_ids = index
        .snapshot()?
        .segments()
        .iter()
        .map(|segment| segment.manifest().segment_id)
        .collect::<Vec<_>>();
    if source_ids.len() != 3 {
        return Err(GauntletError::InvalidContract {
            reason: format!(
                "Q1 E3.5 expected three committed source segments, got {}",
                source_ids.len()
            ),
        });
    }
    for segment in index.snapshot()?.segments() {
        segment
            .verify()
            .map_err(|error| q1_merge_error("verify source segment", error))?;
    }

    let before_snapshot = index.snapshot()?;
    let before_manifest = &before_snapshot.loaded_manifest().manifest;
    let before_generation = before_manifest.generation;
    let before_watermark = before_manifest.docid_high_watermark;
    let before_field_stats = before_manifest.field_stats.clone();
    let before_at_seal_doc_count = index.snapshot()?.at_seal_doc_count();
    let before_live_doc_count = index.snapshot()?.doc_count();
    let first_hole = index
        .snapshot()?
        .segments()
        .windows(2)
        .find_map(|pair| {
            let left_hi = pair[0].manifest().docid_hi;
            let right_lo = pair[1].manifest().docid_lo;
            (left_hi < right_lo).then_some(left_hi)
        })
        .ok_or_else(|| GauntletError::InvalidContract {
            reason: "Q1 E3.5 did not construct an interior burned lease tail".to_owned(),
        })?;
    let merged_docid_lo = index.snapshot()?.segments()[0].manifest().docid_lo;
    let merged_docid_hi = index.snapshot()?.segments()[2].manifest().docid_hi;

    const DOCUMENT_IDS: [&str; 8] = [
        "merge-a1", "merge-a2", "merge-b1", "merge-b2", "merge-c1", "merge-c2", "merge-d1",
        "merge-d2",
    ];
    let mut identity_rows = Vec::with_capacity(DOCUMENT_IDS.len());
    for document_id in DOCUMENT_IDS {
        let resolved = index
            .snapshot()?
            .resolve_document_id(document_id)
            .map_err(|error| q1_merge_error("resolve source identity", error))?
            .ok_or_else(|| GauntletError::InvalidContract {
                reason: format!("Q1 E3.5 source identity {document_id:?} is missing"),
            })?;
        identity_rows.push((document_id, resolved.global_docid));
    }

    const QUERIES: [&str; 4] = [
        "shared",
        "\"shared alpha\"",
        "shared AND beta",
        "alpha OR gamma",
    ];
    let mut before_queries = Vec::with_capacity(QUERIES.len());
    for query in QUERIES {
        let ranked = index
            .search_paginated(cx, query, 32, 0, true)
            .map_err(|error| q1_merge_error("run source query", error))?;
        let docids = index
            .collect_docids(cx, query)
            .map_err(|error| q1_merge_error("collect source docids", error))?;
        before_queries.push((query, ranked, docids));
    }

    let output_segment_id = [
        0xe350_0000_0000_0001,
        0xe350_0000_0000_0002,
        0xe350_0000_0000_0003,
        0xe350_0000_0000_0004,
    ]
    .into_iter()
    .find(|candidate| !source_ids.contains(candidate))
    .ok_or_else(|| GauntletError::InvalidContract {
        reason: "Q1 E3.5 could not choose a collision-free fixture segment id".to_owned(),
    })?;
    let rejected_order = [source_ids[0], source_ids[2]];
    let Err(rejection) = index
        .concat_merge(cx, &rejected_order, output_segment_id, 1_700_000_035)
        .await
    else {
        return Err(GauntletError::InvalidContract {
            reason: "Q1-OB1 admitted a nonconsecutive manifest source list".to_owned(),
        });
    };
    match &rejection {
        QuillIndexError::Keeper(KeeperError::ConcatMerge {
            source:
                ConcatMergeError::NonConsecutiveSources {
                    position,
                    expected,
                    actual,
                },
        }) if *position == 1 && *expected == source_ids[1] && *actual == source_ids[2] => {}
        _ => {
            return Err(GauntletError::InvalidContract {
                reason: format!(
                    "Q1-OB1 manifest-order violation returned the wrong diagnosis: {rejection}"
                ),
            });
        }
    }
    if index.snapshot()?.loaded_manifest().manifest.generation != before_generation
        || index
            .snapshot()?
            .segments()
            .iter()
            .map(|segment| segment.manifest().segment_id)
            .ne(source_ids.iter().copied())
    {
        return Err(GauntletError::InvalidContract {
            reason: "Q1-OB1 rejected merge mutated the committed snapshot".to_owned(),
        });
    }

    let merged = index
        .concat_merge(cx, &source_ids, output_segment_id, 1_700_000_035)
        .await
        .map_err(|error| q1_merge_error("publish valid concat merge", error))?;
    let expected_generation =
        before_generation
            .checked_add(1)
            .ok_or_else(|| GauntletError::InvalidContract {
                reason: "Q1 E3.5 source generation cannot advance".to_owned(),
            })?;
    let merged_manifest = &merged.loaded_manifest().manifest;
    let Some(merged_segment) = merged.segments().first() else {
        return Err(GauntletError::InvalidContract {
            reason: "Q1 E3.5 merged snapshot contains no segment".to_owned(),
        });
    };
    if merged.segments().len() != 1
        || merged_segment.manifest().segment_id != output_segment_id
        || merged_segment.manifest().docid_lo != merged_docid_lo
        || merged_segment.manifest().docid_hi != merged_docid_hi
        || merged_manifest.generation != expected_generation
        || merged_manifest.docid_high_watermark != before_watermark
        || merged_manifest.field_stats != before_field_stats
        || merged.at_seal_doc_count() != before_at_seal_doc_count
        || merged.doc_count() != before_live_doc_count
    {
        return Err(GauntletError::InvalidContract {
            reason:
                "Q1-OB2a merged successor changed range, generation, statistics, or document counts"
                    .to_owned(),
        });
    }
    merged_segment
        .verify()
        .map_err(|error| q1_merge_error("verify rebound merged segment", error))?;
    for section in [
        SectionKind::POSTINGS,
        SectionKind::POSITIONS,
        SectionKind::BLOCKMAX,
        SectionKind::IDMAP,
        SectionKind::IDHASH,
        SectionKind::STOREDMETA,
        SectionKind::STATS,
    ] {
        if merged_segment
            .section(section)
            .map_err(|error| q1_merge_error("open merged section", error))?
            .is_none()
        {
            return Err(GauntletError::InvalidContract {
                reason: format!("Q1 E3.5 merged segment omitted {section:?}"),
            });
        }
    }

    for (document_id, global_docid) in identity_rows {
        let resolved = index
            .snapshot()?
            .resolve_document_id(document_id)
            .map_err(|error| q1_merge_error("resolve merged identity", error))?
            .ok_or_else(|| GauntletError::InvalidContract {
                reason: format!("Q1-OB2c merged identity {document_id:?} is missing"),
            })?;
        let materialized = index
            .snapshot()?
            .materialize_document_id(global_docid)
            .map(|value| value.to_string());
        if resolved.global_docid != global_docid
            || resolved.segment_id != output_segment_id
            || materialized.as_deref() != Some(document_id)
        {
            return Err(GauntletError::InvalidContract {
                reason: format!(
                    "Q1-OB2c identity {document_id:?} changed representative across merge"
                ),
            });
        }
    }
    let first_hole = u32::try_from(first_hole).map_err(|_| GauntletError::InvalidContract {
        reason: "Q1 E3.5 source hole does not fit the public docid domain".to_owned(),
    })?;
    if index
        .snapshot()?
        .materialize_document_id(first_hole)
        .is_some()
    {
        return Err(GauntletError::InvalidContract {
            reason: "Q1-OB2c merged IDMAP filled a burned lease tail".to_owned(),
        });
    }

    for (query, before_ranked, before_docids) in before_queries {
        let after_ranked = index
            .search_paginated(cx, query, 32, 0, true)
            .map_err(|error| q1_merge_error("run merged query", error))?;
        let after_docids = index
            .collect_docids(cx, query)
            .map_err(|error| q1_merge_error("collect merged docids", error))?;
        if after_ranked != before_ranked || after_docids != before_docids {
            return Err(GauntletError::InvalidContract {
                reason: format!("Q1-OB3 query {query:?} changed across concat merge"),
            });
        }
    }

    Ok(vec![
        "Q1-OB1/merge-bound-consecutive".to_owned(),
        "Q1-OB2a/merge-reopen-equivalence".to_owned(),
        "Q1-OB2b/bounded-live-concat".to_owned(),
        "Q1-OB2c/identity-rebuild-reopen".to_owned(),
        "Q1-OB3/query-invariance".to_owned(),
    ])
}

fn q1_allocator_error(error: impl std::fmt::Display) -> GauntletError {
    GauntletError::InvalidContract {
        reason: format!("Q1 live allocator fixture failed: {error}"),
    }
}

fn q1_merge_error(context: &str, error: impl std::fmt::Display) -> GauntletError {
    GauntletError::InvalidContract {
        reason: format!("Q1 E3.5 {context} failed: {error}"),
    }
}

fn is_lower_hex(value: &str, width: usize) -> bool {
    value.len() == width
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn lower_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    let mut output = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        let _ = write!(output, "{byte:02x}");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use frankensearch_quill::quiver::differential::{
        BitpackError, FIXTURE_ID, SPEC_SECTION, pack_values, unpack_scalar_into, unpack_wide_into,
    };

    fn bitpack_fixture_values(width: u8, count: usize) -> Vec<u32> {
        let mask = match width {
            0 => 0,
            32 => u32::MAX,
            _ => (1_u32 << width) - 1,
        };
        let mut state = 0x9e37_79b9_u32
            ^ u32::from(width)
            ^ u32::try_from(count).expect("fixture count fits in u32");
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            values.push(state & mask);
        }
        if let Some(first) = values.first_mut() {
            *first = 0;
        }
        if let Some(last) = values.last_mut() {
            *last = mask;
        }
        values
    }

    fn matching_decode_error(input: &[u8], width: u8, count: usize) -> BitpackError {
        let untouched = vec![0xa5a5_a5a5; count];
        let mut scalar = untouched.clone();
        let mut wide = untouched.clone();
        let scalar_result = unpack_scalar_into(input, width, &mut scalar);
        let wide_result = unpack_wide_into(input, width, &mut wide);
        assert_eq!(scalar_result, wide_result, "typed error mismatch");
        assert_eq!(scalar, untouched, "scalar mutated output before rejecting");
        assert_eq!(wide, untouched, "wide mutated output before rejecting");
        scalar_result.expect_err("malformed fixture must be rejected")
    }

    #[test]
    fn embedded_oracle_contract_matches_exact_lock_resolution() {
        let contract = oracle_version_contract().expect("valid oracle contract");
        assert_eq!(contract.tantivy_version, LOCKED_TANTIVY_VERSION);
        assert_eq!(contract.tantivy_source, LOCKED_TANTIVY_SOURCE);
        assert_eq!(
            contract.tantivy_checksum_sha256,
            LOCKED_TANTIVY_CHECKSUM_SHA256
        );
        assert_eq!(
            contract.lexical_package_version,
            FRANKENSEARCH_LEXICAL_CRATE_VERSION
        );
        assert_eq!(contract.lexical_contract_audit_revision.len(), 40);
        contract
            .validate_stored_structure()
            .expect("stored v3 contract is self-contained");
        contract
            .validate_current_dependency()
            .expect("stored contract matches current lock resolution");
    }

    #[test]
    fn stored_oracle_contract_rejects_every_mutated_field() {
        let contract = oracle_version_contract().expect("valid oracle contract");
        let mut mutations = Vec::new();

        let mut mutated = contract.clone();
        mutated.schema_version = 2;
        mutations.push(mutated);
        let mut mutated = contract.clone();
        // 0.26.1 is the CURRENT v4 version, so a version mutation must move
        // elsewhere — the retired v3 git line is the realistic substitution.
        mutated.tantivy_version = "0.27.0".to_owned();
        mutations.push(mutated);
        let mut mutated = contract.clone();
        mutated.tantivy_source = "git+https://example.invalid/tantivy".to_owned();
        mutations.push(mutated);
        let mut mutated = contract.clone();
        mutated.tantivy_checksum_sha256 = "0".repeat(64);
        mutations.push(mutated);
        let mut mutated = contract.clone();
        mutated.lexical_package = "lookalike-lexical".to_owned();
        mutations.push(mutated);
        let mut mutated = contract.clone();
        mutated.lexical_package_version = "0.2.0".to_owned();
        mutations.push(mutated);
        let mut mutated = contract;
        mutated.lexical_contract_audit_revision = "f".repeat(40);
        mutations.push(mutated);

        for mutation in mutations {
            assert!(
                mutation.validate_stored_structure().is_err(),
                "mutated stored dependency contract was accepted: {mutation:?}"
            );
        }
    }

    #[test]
    fn retained_v2_oracle_contract_is_exact_but_never_current() {
        let retained = OracleVersionContract {
            schema_version: 2,
            tantivy_version: ORACLE_V2_TANTIVY_VERSION.to_owned(),
            tantivy_source: ORACLE_V2_TANTIVY_SOURCE.to_owned(),
            tantivy_checksum_sha256: ORACLE_V2_TANTIVY_CHECKSUM_SHA256.to_owned(),
            lexical_package: ORACLE_V2_LEXICAL_PACKAGE.to_owned(),
            lexical_package_version: ORACLE_V2_LEXICAL_PACKAGE_VERSION.to_owned(),
            lexical_contract_audit_revision: ORACLE_V2_LEXICAL_CONTRACT_AUDIT_REVISION.to_owned(),
        };

        retained
            .validate_retained_structure()
            .expect("the committed v2 dependency record remains decode-only evidence");
        assert!(retained.validate_stored_structure().is_err());
        assert!(retained.validate_current_dependency().is_err());
        assert_eq!(
            retained.identity_sha256().expect("retained v2 identity"),
            "58988c117389e3222fc6c2202d311fc70158ef0e65e069100181be5fd98d3ba3",
            "retained evidence must keep the v2 domain used by committed ledger records",
        );

        let mut substituted = retained;
        substituted.tantivy_checksum_sha256 = "0".repeat(64);
        assert!(substituted.validate_retained_structure().is_err());
        assert!(substituted.identity_sha256().is_err());
    }

    #[test]
    fn retained_v3_oracle_contract_is_exact_but_never_current() {
        let retained = OracleVersionContract {
            schema_version: 3,
            tantivy_version: ORACLE_V3_TANTIVY_VERSION.to_owned(),
            tantivy_source: ORACLE_V3_TANTIVY_SOURCE.to_owned(),
            tantivy_checksum_sha256: ORACLE_V3_TANTIVY_CHECKSUM_SHA256.to_owned(),
            lexical_package: ORACLE_V3_LEXICAL_PACKAGE.to_owned(),
            lexical_package_version: ORACLE_V3_LEXICAL_PACKAGE_VERSION.to_owned(),
            lexical_contract_audit_revision: ORACLE_V3_LEXICAL_CONTRACT_AUDIT_REVISION.to_owned(),
        };

        retained
            .validate_retained_structure()
            .expect("the committed v3 dependency record remains decode-only evidence");
        assert!(retained.validate_stored_structure().is_err());
        assert!(retained.validate_current_dependency().is_err());
        assert_eq!(
            retained.identity_sha256().expect("retained v3 identity"),
            "f85cb172b2e16d98bb80b067add5478294ef2618b6fa36c4a9de5dc7316f6f1f",
            "retained evidence must keep the v3 domain used by committed ledger records",
        );

        let mut substituted = retained;
        substituted.tantivy_source = ORACLE_V4_TANTIVY_SOURCE.to_owned();
        assert!(substituted.validate_retained_structure().is_err());
        assert!(substituted.identity_sha256().is_err());
    }

    #[test]
    fn q1_catalog_marks_every_obligation_executable_and_runs_live_merge_subfixtures() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let catalog = q1_fixture_catalog().expect("valid Q1 catalog");
            assert_eq!(
                Sha256::digest(Q1_FIXTURE_CATALOG_JSON.as_bytes()).as_slice(),
                Q1_FIXTURE_CATALOG_SHA256
            );
            assert_eq!(catalog.fixtures.len(), 8);
            assert_eq!(
                catalog
                    .fixtures
                    .iter()
                    .filter(|fixture| fixture.status == "executable")
                    .map(|fixture| fixture.id.as_str())
                    .collect::<Vec<_>>(),
                [
                    "Q1-OB1", "Q1-OB2a", "Q1-OB2b", "Q1-OB2c", "Q1-OB3", "Q1-OB4", "Q1-OB5",
                    "Q1-OB6",
                ]
            );
            assert_eq!(
                run_q1_live_fixtures(&cx).await.expect("live Q1 fixtures"),
                [
                    "Q1-OB1/manifest-overlap",
                    "Q1-OB1/merge-bound-consecutive",
                    "Q1-OB2a/merge-reopen-equivalence",
                    "Q1-OB2b/bounded-live-concat",
                    "Q1-OB2c/identity-rebuild-reopen",
                    "Q1-OB3/query-invariance",
                    "Q1-OB5/allocator-boundary",
                ]
            );
            assert_eq!(catalog.internal_differentials.len(), 1);
            assert_eq!(catalog.internal_differentials[0].id, FIXTURE_ID);
            assert_eq!(catalog.internal_differentials[0].status, "executable");
        });
    }

    #[test]
    fn quiver_bitpack_scalar_and_wide_fixture_is_executable() -> Result<(), BitpackError> {
        let catalog = q1_fixture_catalog().expect("valid fixture catalog");
        let fixture = &catalog.internal_differentials[0];
        assert_eq!(fixture.id, FIXTURE_ID);
        assert_eq!(fixture.status, "executable");
        assert_eq!(
            fixture.source_contract,
            "docs/contracts/fslx-format-registry.md#52-postings"
        );
        assert_eq!(FIXTURE_ID, "quiver-postings-bitpack-scalar-wide-v1");
        assert_eq!(
            SPEC_SECTION,
            "FSLX v1 section 5.2 LSB-first bitpacked payloads"
        );

        let known_answers: [(&[u8], u8, &[u32]); 3] = [
            (&[0x51, 0x01], 3, &[1, 2, 5]),
            (&[0x39], 2, &[1, 2, 3]),
            (
                &[0x12, 0x34, 0x56, 0x78, 0xef, 0xbe, 0xad, 0xde],
                32,
                &[0x7856_3412, 0xdead_beef],
            ),
        ];
        for (packed, width, expected) in known_answers {
            assert_eq!(pack_values(expected, width)?, packed);
            let mut scalar = vec![u32::MAX; expected.len()];
            let mut wide = scalar.clone();
            unpack_scalar_into(packed, width, &mut scalar)?;
            unpack_wide_into(packed, width, &mut wide)?;
            assert_eq!(scalar, expected, "known scalar answer width={width}");
            assert_eq!(wide, expected, "known wide answer width={width}");
        }

        for (shape, count) in [("doc-delta", 127), ("frequency", 128)] {
            for width in 0..=32 {
                let expected = bitpack_fixture_values(width, count);
                let packed = pack_values(&expected, width)?;
                for offset in 0..32 {
                    let mut storage = vec![0x5a; offset];
                    let payload_start = storage.len();
                    storage.extend_from_slice(&packed);
                    let payload_end = storage.len();
                    storage.extend_from_slice(&[0xa5; 32]);
                    let input = &storage[payload_start..payload_end];
                    let mut scalar = vec![u32::MAX; count];
                    let mut wide = scalar.clone();

                    let scalar_result = unpack_scalar_into(input, width, &mut scalar);
                    let wide_result = unpack_wide_into(input, width, &mut wide);
                    assert_eq!(
                        scalar_result, wide_result,
                        "result mismatch for {shape} width={width} offset={offset}"
                    );
                    scalar_result?;
                    assert_eq!(
                        scalar, expected,
                        "scalar mismatch for {shape} width={width} offset={offset}"
                    );
                    assert_eq!(
                        wide, expected,
                        "wide mismatch for {shape} width={width} offset={offset}"
                    );
                }
            }
        }

        let expected = bitpack_fixture_values(7, 127);
        let packed = pack_values(&expected, 7)?;
        let truncated = &packed[..packed.len() - 1];
        assert_eq!(
            matching_decode_error(truncated, 7, expected.len()),
            BitpackError::LengthMismatch {
                expected: packed.len(),
                actual: truncated.len(),
            }
        );

        let mut overlong = packed.clone();
        overlong.push(0);
        assert_eq!(
            matching_decode_error(&overlong, 7, expected.len()),
            BitpackError::LengthMismatch {
                expected: packed.len(),
                actual: overlong.len(),
            }
        );

        let mut noncanonical = pack_values(&bitpack_fixture_values(3, 127), 3)?;
        let final_byte = noncanonical
            .last_mut()
            .expect("127 three-bit values have a final byte");
        *final_byte |= 0x80;
        let final_byte = *final_byte;
        assert_eq!(
            matching_decode_error(&noncanonical, 3, 127),
            BitpackError::NonCanonicalPadding {
                byte: final_byte,
                used_bits: 5,
            }
        );
        assert_eq!(
            matching_decode_error(&[], 33, 128),
            BitpackError::InvalidWidth { width: 33 }
        );
        assert_eq!(
            matching_decode_error(&[0], 0, 128),
            BitpackError::LengthMismatch {
                expected: 0,
                actual: 1,
            }
        );
        assert_eq!(
            pack_values(&[1], 0),
            Err(BitpackError::ValueOutOfRange {
                index: 0,
                value: 1,
                width: 0,
            })
        );
        assert_eq!(
            pack_values(&[0, 8], 3),
            Err(BitpackError::ValueOutOfRange {
                index: 1,
                value: 8,
                width: 3,
            })
        );

        Ok(())
    }
}
