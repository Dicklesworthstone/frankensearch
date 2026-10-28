//! Fingerprint indexing policy without making query tuning invalidate its work.
//!
//! This is an explicit exclusion list, not a projection onto today's known
//! indexing fields. New configuration fields remain part of the fingerprint
//! until their effect on indexing has been reviewed. Model identity, source
//! hashes, executable identity and artifact admission are checked by the owner.

use frankensearch_core::{DefaultCanonicalizer, SearchError, SearchResult};
use serde::{Deserialize, Serialize};

use crate::config::{FsfsConfig, SearchConfig};

use super::content_sha256_hex;

pub(super) const INDEXING_CONTRACT_VERSION_V1: &str = "fsfs.indexing_contract.v1";

/// Explicit, versioned indexing-pipeline contract (`bd-ky0j2`).
///
/// Binds every compile-time and configuration-derived contract that shapes
/// persisted lexical segments, vector rows, window plans, and catalog records:
/// - checkpoint schema version (`INDEXING_CHECKPOINT_SCHEMA_VERSION`)
/// - FSVI binary format version (`frankensearch_index::FSVI_V2_VERSION`)
/// - Quill lexical engine/segment/manifest format versions and canonicalizer limits
/// - embedding canonicalizer limits (`DefaultCanonicalizer::default()`)
/// - fast semantic windowing contract (`WINDOW_CHARS`, `MAX_WINDOWS_PER_FILE`)
/// - indexing configuration digest (`digest(config)`)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct IndexingContractDigestV1 {
    pub contract_version: String,
    pub checkpoint_schema_version: u16,
    pub fsvi_format_version: u16,
    pub quill_engine_version: u32,
    pub quill_fslx_format_version: u32,
    pub quill_manifest_format_version: u32,
    pub lexical_canonicalizer_max_length: usize,
    pub lexical_canonicalizer_code_head_lines: usize,
    pub lexical_canonicalizer_code_tail_lines: usize,
    pub embedding_canonicalizer_max_length: usize,
    pub embedding_canonicalizer_code_head_lines: usize,
    pub embedding_canonicalizer_code_tail_lines: usize,
    pub fast_window_chars: usize,
    pub fast_window_max_per_file_cap: usize,
    pub configuration_sha256: String,
}

impl IndexingContractDigestV1 {
    pub(super) fn from_config(config: &FsfsConfig) -> SearchResult<Self> {
        let embedding_canonicalizer = DefaultCanonicalizer::default();
        let lexical_canonicalizer = crate::runtime::LEXICAL_CANONICALIZER;
        Ok(Self {
            contract_version: INDEXING_CONTRACT_VERSION_V1.to_owned(),
            checkpoint_schema_version: super::INDEXING_CHECKPOINT_SCHEMA_VERSION,
            fsvi_format_version: frankensearch_index::FSVI_V2_VERSION,
            quill_engine_version: frankensearch_quill::CURRENT_ENGINE_VERSION,
            quill_fslx_format_version: frankensearch_quill::FSLX_FORMAT_VERSION,
            quill_manifest_format_version: frankensearch_quill::MANIFEST_FORMAT_VERSION,
            lexical_canonicalizer_max_length: lexical_canonicalizer.max_length,
            lexical_canonicalizer_code_head_lines: lexical_canonicalizer.code_head_lines,
            lexical_canonicalizer_code_tail_lines: lexical_canonicalizer.code_tail_lines,
            embedding_canonicalizer_max_length: embedding_canonicalizer.max_length,
            embedding_canonicalizer_code_head_lines: embedding_canonicalizer.code_head_lines,
            embedding_canonicalizer_code_tail_lines: embedding_canonicalizer.code_tail_lines,
            fast_window_chars: crate::runtime::semantic_windows::WINDOW_CHARS,
            fast_window_max_per_file_cap: crate::runtime::semantic_windows::MAX_WINDOWS_PER_FILE,
            configuration_sha256: digest(config)?,
        })
    }

    pub(super) fn sha256_hex(&self) -> SearchResult<String> {
        let bytes = serde_json::to_vec(self).map_err(|source| SearchError::SubsystemError {
            subsystem: "fsfs.complete_generation.indexing_contract",
            source: Box::new(source),
        })?;
        Ok(content_sha256_hex(&bytes))
    }
}

pub(super) fn contract_digest(config: &FsfsConfig) -> SearchResult<String> {
    IndexingContractDigestV1::from_config(config)?.sha256_hex()
}

pub(super) fn digest(config: &FsfsConfig) -> SearchResult<String> {
    let mut config = config.clone();
    // Keeping the command alive after indexing does not change indexed bytes.
    config.indexing.watch_mode = false;

    // These controls apply after query-time Initial publication, or to query
    // ranking. They do not select documents, producers or indexing coverage.
    // In particular, do NOT replace SearchConfig wholesale: fast_only affects
    // which tiers are indexed, and unknown future fields must stay conservative.
    let defaults = SearchConfig::default();
    config.search.quality_weight = defaults.quality_weight;
    config.search.rrf_k = defaults.rrf_k;
    config.search.quality_timeout_ms = defaults.quality_timeout_ms;
    config.search.rerank_timeout_ms = defaults.rerank_timeout_ms;

    let mut value =
        serde_json::to_value(&config).map_err(|source| SearchError::SubsystemError {
            subsystem: "fsfs.complete_generation.reuse_config",
            source: Box::new(source),
        })?;
    // Admission policy does not shape any indexed byte. Omit the new key
    // entirely so receipts minted before this switch retain their exact digest;
    // serializing a default `false` here would invalidate those receipts too.
    value
        .get_mut("indexing")
        .and_then(serde_json::Value::as_object_mut)
        .ok_or_else(|| SearchError::InvalidConfig {
            field: "indexing".to_owned(),
            value: "serialized configuration".to_owned(),
            reason: "the serialized indexing configuration must be an object".to_owned(),
        })?
        .shift_remove("allow_contract_upgrade_reuse");
    // Query-time ranking policy also leaves persisted source/vector/lexical bytes unchanged.
    value
        .get_mut("search")
        .and_then(serde_json::Value::as_object_mut)
        .ok_or_else(|| SearchError::InvalidConfig {
            field: "search".to_owned(),
            value: "serialized configuration".to_owned(),
            reason: "the serialized search configuration must be an object".to_owned(),
        })?
        .shift_remove("ranking_priors");
    let bytes = serde_json::to_vec(&value).map_err(|source| SearchError::SubsystemError {
        subsystem: "fsfs.complete_generation.reuse_config",
        source: Box::new(source),
    })?;
    Ok(content_sha256_hex(&bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DegradationOverrideMode, TextSelectionMode};

    type Change = (&'static str, fn(&mut FsfsConfig));

    /// The previous production algorithm, retained only as a negative control.
    fn previous_digest(config: &FsfsConfig) -> String {
        let mut config = config.clone();
        config.indexing.watch_mode = false;
        // The old configuration type had neither key. Retain its original
        // struct field order independently of the production removal method;
        // swap-removing a key would silently change historical receipt bytes.
        let mut value = serde_json::to_value(&config).unwrap();
        value["indexing"]
            .as_object_mut()
            .unwrap()
            .retain(|key, _| key != "allow_contract_upgrade_reuse");
        value["search"]
            .as_object_mut()
            .unwrap()
            .retain(|key, _| key != "ranking_priors");
        content_sha256_hex(&serde_json::to_vec(&value).unwrap())
    }

    #[test]
    fn upgrade_admission_switch_does_not_change_persisted_index_contract() {
        let config = FsfsConfig::default();
        let mut opted_in = config.clone();
        opted_in.indexing.allow_contract_upgrade_reuse = true;
        opted_in.search.ranking_priors = true;
        assert_ne!(
            serde_json::to_value(&config).unwrap(),
            serde_json::to_value(&opted_in).unwrap()
        );
        assert_eq!(digest(&config).unwrap(), digest(&opted_in).unwrap());
        assert_eq!(
            contract_digest(&config).unwrap(),
            contract_digest(&opted_in).unwrap()
        );
        opted_in.indexing.fast_window_max_per_file += 1;
        assert_ne!(
            contract_digest(&config).unwrap(),
            contract_digest(&opted_in).unwrap()
        );
    }

    #[test]
    fn query_ranking_and_deadline_changes_reuse_the_same_indexing_contract() {
        let config = FsfsConfig::default();
        let expected = digest(&config).unwrap();
        let expected_contract = contract_digest(&config).unwrap();
        let changes: &[Change] = &[
            ("quality_weight", |c| c.search.quality_weight = 0.25),
            ("rrf_k", |c| c.search.rrf_k = 30.0),
            ("quality_timeout_ms", |c| {
                c.search.quality_timeout_ms = 4_000;
            }),
            ("rerank_timeout_ms", |c| {
                c.search.rerank_timeout_ms = 5_000;
            }),
        ];
        for (name, change) in changes {
            let mut tuned = config.clone();
            change(&mut tuned);
            assert_ne!(previous_digest(&tuned), previous_digest(&config), "{name}");
            assert_eq!(digest(&tuned).unwrap(), expected, "{name}");
            assert_eq!(
                contract_digest(&tuned).unwrap(),
                expected_contract,
                "{name}"
            );
            tuned.indexing.watch_mode = true;
            assert_eq!(digest(&tuned).unwrap(), expected, "watch: {name}");
            assert_eq!(
                contract_digest(&tuned).unwrap(),
                expected_contract,
                "watch: {name}"
            );
        }
    }

    #[test]
    fn indexing_contract_digest_changes_when_schema_or_window_or_canonicalizer_contract_drifts() {
        let config = FsfsConfig::default();
        let baseline = IndexingContractDigestV1::from_config(&config).unwrap();
        let expected = baseline.sha256_hex().unwrap();
        assert_eq!(expected.len(), 64);

        let mut changed_version = baseline.clone();
        changed_version.contract_version = "fsfs.indexing_contract.v2".to_owned();
        assert_ne!(changed_version.sha256_hex().unwrap(), expected);

        let mut changed_fsvi = baseline.clone();
        changed_fsvi.fsvi_format_version += 1;
        assert_ne!(changed_fsvi.sha256_hex().unwrap(), expected);

        let mut changed_quill = baseline.clone();
        changed_quill.quill_fslx_format_version += 1;
        assert_ne!(changed_quill.sha256_hex().unwrap(), expected);

        let mut changed_canonicalizer = baseline.clone();
        changed_canonicalizer.embedding_canonicalizer_max_length += 1;
        assert_ne!(changed_canonicalizer.sha256_hex().unwrap(), expected);

        let mut changed_window = baseline;
        changed_window.fast_window_chars += 100;
        assert_ne!(changed_window.sha256_hex().unwrap(), expected);
    }

    #[test]
    fn default_query_controls_preserve_the_existing_fingerprint() {
        let mut config = FsfsConfig::default();
        config.indexing.fast_model = "owner-selected-fast-model".to_owned();
        config.discovery.roots = vec!["project-a".to_owned(), "project-b".to_owned()];
        for watch in [false, true] {
            config.indexing.watch_mode = watch;
            assert_eq!(digest(&config).unwrap(), previous_digest(&config));
        }
    }

    #[test]
    fn fingerprinting_does_not_change_the_callers_query_policy() {
        let mut config = FsfsConfig::default();
        config.indexing.watch_mode = true;
        config.search.quality_weight = 0.25;
        config.search.rrf_k = 30.0;
        config.search.quality_timeout_ms = 4_000;
        config.search.rerank_timeout_ms = 5_000;
        let before = config.clone();
        let _ = digest(&config).unwrap();
        assert_eq!(config, before);
    }

    #[test]
    fn producer_topology_and_indexing_policy_changes_still_invalidate_reuse() {
        let config = FsfsConfig::default();
        let expected = digest(&config).unwrap();
        let changes: &[Change] = &[
            ("fast_model", |c| {
                c.indexing.fast_model.push_str("-changed");
            }),
            ("quality_model", |c| {
                c.indexing.quality_model.push_str("-changed");
            }),
            ("quality_disabled", |c| c.indexing.quality_model.clear()),
            ("model_dir", |c| c.indexing.model_dir.push_str("/changed")),
            ("offline", |c| c.indexing.offline = !c.indexing.offline),
            ("batch_size", |c| c.indexing.embedding_batch_size += 1),
            ("window_coverage", |c| {
                c.indexing.fast_window_max_per_file = 1;
            }),
            ("reindex_on_change", |c| {
                c.indexing.reindex_on_change = false;
            }),
            ("fast_only", |c| c.search.fast_only = true),
        ];
        for (name, change) in changes {
            let mut mutated = config.clone();
            change(&mut mutated);
            assert_ne!(digest(&mutated).unwrap(), expected, "{name}");
        }
    }

    #[test]
    fn discovery_storage_pressure_and_privacy_remain_in_the_contract() {
        let config = FsfsConfig::default();
        let expected = digest(&config).unwrap();
        let changes: &[Change] = &[
            ("roots", |c| {
                c.discovery.roots.push("other-source".to_owned());
            }),
            ("exclusions", |c| {
                c.discovery.exclude_patterns.push("private".to_owned());
            }),
            ("selection_mode", |c| {
                c.discovery.text_selection_mode = TextSelectionMode::Allowlist;
            }),
            ("binary_policy", |c| {
                c.discovery
                    .binary_blocklist_extensions
                    .push("new-format".to_owned());
            }),
            ("file_size", |c| c.discovery.max_file_size_mb += 1),
            ("symlinks", |c| c.discovery.follow_symlinks = true),
            ("network_mounts", |c| c.discovery.skip_network_mounts = true),
            ("index_dir", |c| c.storage.index_dir.push_str("-other")),
            ("catalog", |c| c.storage.db_path.push_str("-other")),
            ("disk_budget", |c| c.storage.disk_budget_bytes = Some(1_024)),
            ("degradation", |c| {
                c.pressure.degradation_override = DegradationOverrideMode::ForceLexicalOnly;
            }),
            ("quality_circuit", |c| {
                c.pressure.quality_circuit_open = true;
            }),
            ("pause", |c| c.pressure.hard_pause_requested = true),
            ("memory_ceiling", |c| c.pressure.memory_ceiling_mb += 1),
            ("content_privacy", |c| {
                c.privacy.redact_file_contents_in_logs = false;
            }),
            ("path_privacy", |c| {
                c.privacy.redact_paths_in_telemetry = false;
            }),
        ];
        for (name, change) in changes {
            let mut mutated = config.clone();
            change(&mut mutated);
            assert_ne!(digest(&mutated).unwrap(), expected, "{name}");
        }
    }

    #[test]
    fn unreviewed_search_options_are_not_accidentally_erased() {
        let config = FsfsConfig::default();
        let expected = digest(&config).unwrap();
        let changes: &[Change] = &[
            ("default_limit", |c| c.search.default_limit += 1),
            ("explain", |c| c.search.explain = true),
            ("rerank", |c| c.search.rerank = true),
            ("shadow", |c| c.search.shadow_mode = true),
            ("shadow_rate", |c| {
                c.search.shadow_sample_rate_basis_points += 1;
            }),
            ("shadow_concurrency", |c| {
                c.search.shadow_max_in_flight += 1;
            }),
            ("shadow_epsilon", |c| c.search.shadow_score_epsilon = 0.01),
        ];
        for (name, change) in changes {
            let mut mutated = config.clone();
            change(&mut mutated);
            assert_ne!(digest(&mutated).unwrap(), expected, "{name}");
        }
    }
}
