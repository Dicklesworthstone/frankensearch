//! Fingerprint indexing policy without making query tuning invalidate its work.
//!
//! This is an explicit exclusion list, not a projection onto today's known
//! indexing fields. New configuration fields remain part of the fingerprint
//! until their effect on indexing has been reviewed. Model identity, source
//! hashes, executable identity and artifact admission are checked by the owner.

use frankensearch_core::{SearchError, SearchResult};

use crate::config::{FsfsConfig, SearchConfig};

use super::content_sha256_hex;

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

    let bytes = serde_json::to_vec(&config).map_err(|source| SearchError::SubsystemError {
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
        content_sha256_hex(&serde_json::to_vec(&config).unwrap())
    }

    #[test]
    fn query_ranking_and_deadline_changes_reuse_the_same_indexing_contract() {
        let config = FsfsConfig::default();
        let expected = digest(&config).unwrap();
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
            tuned.indexing.watch_mode = true;
            assert_eq!(digest(&tuned).unwrap(), expected, "watch: {name}");
        }
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
            ("window_coverage", |c| c.indexing.fast_window_max_per_file = 1),
            ("reindex_on_change", |c| c.indexing.reindex_on_change = false),
            ("fast_only", |c| c.search.fast_only = true),
        ];
        for (name, change) in changes {
            let mut changed = config.clone();
            change(&mut changed);
            assert_ne!(digest(&changed).unwrap(), expected, "{name}");
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
            ("quality_circuit", |c| c.pressure.quality_circuit_open = true),
            ("pause", |c| c.pressure.hard_pause_requested = true),
            ("memory_ceiling", |c| c.pressure.memory_ceiling_mb += 1),
            ("content_privacy", |c| c.privacy.redact_file_contents_in_logs = false),
            ("path_privacy", |c| c.privacy.redact_paths_in_telemetry = false),
        ];
        for (name, change) in changes {
            let mut changed = config.clone();
            change(&mut changed);
            assert_ne!(digest(&changed).unwrap(), expected, "{name}");
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
            let mut changed = config.clone();
            change(&mut changed);
            assert_ne!(digest(&changed).unwrap(), expected, "{name}");
        }
    }
}
