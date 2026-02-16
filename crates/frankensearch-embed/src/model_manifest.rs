//! Model manifest definitions and verification helpers.
//!
//! This module is intentionally synchronous and runtime-agnostic:
//! it performs filesystem and hashing work only, and leaves transport/network
//! to higher-level download orchestration.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::{OnceLock, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use frankensearch_core::error::{SearchError, SearchResult};

/// Environment variable for explicit model-download consent.
pub const DOWNLOAD_CONSENT_ENV: &str = "FRANKENSEARCH_ALLOW_DOWNLOAD";

/// Placeholder checksum used until a model file is downloaded and verified.
pub const PLACEHOLDER_VERIFY_AFTER_DOWNLOAD: &str = "PLACEHOLDER_VERIFY_AFTER_DOWNLOAD";

/// Placeholder revision used by built-in manifests until pinned revisions are filled in.
pub const PLACEHOLDER_PINNED_REVISION: &str = "UNPINNED_VERIFY_AFTER_DOWNLOAD";

/// Schema version for the manifest catalog format.
///
/// Bump this when the manifest structure changes in a backwards-incompatible way.
/// Consumers compare the embedded schema version against the cached manifest to
/// detect model upgrades that require re-download.
pub const MANIFEST_SCHEMA_VERSION: u32 = 2;

/// Which search tier a model serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelTier {
    /// Fast tier (~0.57ms per query), lower dimension.
    Fast,
    /// Quality tier (~128ms per query), higher dimension.
    Quality,
    /// Cross-encoder reranker, applied to top-K results.
    Reranker,
}

const HASH_BUFFER_SIZE: usize = 8 * 1024;

/// One file required by a model manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelFile {
    /// Relative path inside the model directory.
    pub name: String,
    /// Expected lowercase SHA256 hex digest.
    pub sha256: String,
    /// Expected size in bytes.
    pub size: u64,
    /// Explicit download URL. When `None`, the URL is derived from the parent
    /// manifest's `repo` + `revision` using the `HuggingFace` `/resolve/` path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

impl ModelFile {
    /// Returns true when the file still uses the placeholder checksum.
    #[must_use]
    pub fn uses_placeholder_checksum(&self) -> bool {
        self.sha256 == PLACEHOLDER_VERIFY_AFTER_DOWNLOAD
    }

    /// Returns true when checksum is usable for production verification.
    #[must_use]
    pub fn has_verified_checksum(&self) -> bool {
        is_valid_sha256_hex(&self.sha256) && !self.uses_placeholder_checksum()
    }

    /// Return the download URL for this file, preferring the explicit `url`
    /// field and falling back to the standard `HuggingFace` `/resolve/` path.
    #[must_use]
    pub fn download_url(&self, repo: &str, revision: &str) -> String {
        self.url.as_ref().map_or_else(
            || {
                format!(
                    "https://huggingface.co/{repo}/resolve/{revision}/{}",
                    self.name
                )
            },
            Clone::clone,
        )
    }
}

/// Manifest for one downloadable model bundle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Stable model identifier.
    pub id: String,
    /// Human-readable version tag for manifest-managed model assets.
    #[serde(default)]
    pub version: String,
    /// Human-readable display name (e.g., "Potion Base 128M (fast tier)").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    /// Optional longer description for CLI/help output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// `HuggingFace` repository slug.
    pub repo: String,
    /// Pinned revision (commit SHA).
    pub revision: String,
    /// Required files for this model.
    pub files: Vec<ModelFile>,
    /// SPDX-style license identifier.
    pub license: String,
    /// Output embedding dimension (e.g., 256 for potion, 384 for `MiniLM`).
    /// `None` for models that don't produce fixed-dim embeddings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimension: Option<u32>,
    /// Which search tier this model serves.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier: Option<ModelTier>,
    /// Optional precomputed aggregate download size in bytes.
    #[serde(
        default,
        rename = "total_size_bytes",
        skip_serializing_if = "is_zero_u64"
    )]
    pub download_size_bytes: u64,
}

#[allow(clippy::trivially_copy_pass_by_ref)] // serde requires &T signature
const fn is_zero_u64(value: &u64) -> bool {
    *value == 0
}

impl ModelManifest {
    /// Built-in manifest for MiniLM-L6-v2 (quality tier).
    #[must_use]
    pub fn minilm_v2() -> Self {
        const REVISION: &str = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf";
        const REPO: &str = "sentence-transformers/all-MiniLM-L6-v2";
        Self {
            id: "all-minilm-l6-v2".to_owned(),
            version: "v1".to_owned(),
            display_name: Some("All MiniLM L6 v2 (quality tier)".to_owned()),
            description: Some(
                "MiniLM-L6-v2 ONNX sentence embedding model for quality-tier semantic search"
                    .to_owned(),
            ),
            repo: REPO.to_owned(),
            revision: REVISION.to_owned(),
            files: vec![
                ModelFile {
                    name: "onnx/model.onnx".to_owned(),
                    sha256: "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452"
                        .to_owned(),
                    size: 90_405_214,
                    url: Some(
                        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/onnx/model.onnx"
                            .to_owned(),
                    ),
                },
                ModelFile {
                    name: "tokenizer.json".to_owned(),
                    sha256: "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"
                        .to_owned(),
                    size: 466_247,
                    url: Some(
                        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/tokenizer.json"
                            .to_owned(),
                    ),
                },
                ModelFile {
                    name: "config.json".to_owned(),
                    sha256: "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41"
                        .to_owned(),
                    size: 612,
                    url: Some(
                        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/config.json"
                            .to_owned(),
                    ),
                },
                ModelFile {
                    name: "special_tokens_map.json".to_owned(),
                    sha256: "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3"
                        .to_owned(),
                    size: 112,
                    url: Some(
                        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/special_tokens_map.json"
                            .to_owned(),
                    ),
                },
                ModelFile {
                    name: "tokenizer_config.json".to_owned(),
                    sha256: "acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b"
                        .to_owned(),
                    size: 350,
                    url: Some(
                        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/tokenizer_config.json"
                            .to_owned(),
                    ),
                },
            ],
            license: "Apache-2.0".to_owned(),
            dimension: Some(384),
            tier: Some(ModelTier::Quality),
            download_size_bytes: 90_872_535,
        }
    }

    /// Built-in manifest for potion-128M style `Model2Vec` assets (fast tier).
    #[must_use]
    pub fn potion_128m() -> Self {
        const REVISION: &str = "a28f4eebecd4dc585034f605e52d414878a0417c";
        const REPO: &str = "minishlab/potion-multilingual-128M";
        Self {
            id: "potion-multilingual-128m".to_owned(),
            version: "v1".to_owned(),
            display_name: Some("Potion Multilingual 128M (fast tier)".to_owned()),
            description: Some(
                "Model2Vec static embedding model for fast-tier multilingual retrieval".to_owned(),
            ),
            repo: REPO.to_owned(),
            revision: REVISION.to_owned(),
            files: vec![
                ModelFile {
                    name: "tokenizer.json".to_owned(),
                    sha256: "19f1909063da3cfe3bd83a782381f040dccea475f4816de11116444a73e1b6a1"
                        .to_owned(),
                    size: 18_616_131,
                    url: Some(
                        "https://huggingface.co/minishlab/potion-multilingual-128M/resolve/a28f4eebecd4dc585034f605e52d414878a0417c/tokenizer.json"
                            .to_owned(),
                    ),
                },
                ModelFile {
                    name: "model.safetensors".to_owned(),
                    sha256: "14b5eb39cb4ce5666da8ad1f3dc6be4346e9b2d601c073302fa0a31bf7943397"
                        .to_owned(),
                    size: 512_361_560,
                    url: Some(
                        "https://huggingface.co/minishlab/potion-multilingual-128M/resolve/a28f4eebecd4dc585034f605e52d414878a0417c/model.safetensors"
                            .to_owned(),
                    ),
                },
            ],
            license: "Apache-2.0".to_owned(),
            dimension: Some(256),
            tier: Some(ModelTier::Fast),
            download_size_bytes: 530_977_691,
        }
    }

    /// Built-in manifest for MS MARCO `MiniLM` reranker (cross-encoder).
    #[must_use]
    pub fn ms_marco_reranker() -> Self {
        const REVISION: &str = "c5ee24cb16019beea0893ab7796b1df96625c6b8";
        const REPO: &str = "cross-encoder/ms-marco-MiniLM-L-6-v2";
        Self {
            id: "ms-marco-minilm-l-6-v2".to_owned(),
            version: "v1".to_owned(),
            display_name: Some("MS MARCO MiniLM L-6 v2 (reranker)".to_owned()),
            description: Some(
                "MS MARCO cross-encoder reranker model for final relevance scoring".to_owned(),
            ),
            repo: REPO.to_owned(),
            revision: REVISION.to_owned(),
            files: vec![
                ModelFile {
                    name: "onnx/model.onnx".to_owned(),
                    sha256: "5d3e70fd0c9ff14b9b5169a51e957b7a9c74897afd0a35ce4bd318150c1d4d4a"
                        .to_owned(),
                    size: 91_011_230,
                    url: Some(
                        "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/c5ee24cb16019beea0893ab7796b1df96625c6b8/onnx/model.onnx"
                            .to_owned(),
                    ),
                },
                ModelFile {
                    name: "tokenizer.json".to_owned(),
                    sha256: "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"
                        .to_owned(),
                    size: 711_396,
                    url: Some(
                        "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/c5ee24cb16019beea0893ab7796b1df96625c6b8/tokenizer.json"
                            .to_owned(),
                    ),
                },
                ModelFile {
                    name: "config.json".to_owned(),
                    sha256: "380e02c93f431831be65d99a4e7e5f67c133985bf2e77d9d4eba46847190bacc"
                        .to_owned(),
                    size: 794,
                    url: Some(
                        "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/c5ee24cb16019beea0893ab7796b1df96625c6b8/config.json"
                            .to_owned(),
                    ),
                },
                ModelFile {
                    name: "special_tokens_map.json".to_owned(),
                    sha256: "3c3507f36dff57bce437223db3b3081d1e2b52ec3e56ee55438193ecb2c94dd6"
                        .to_owned(),
                    size: 132,
                    url: Some(
                        "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/c5ee24cb16019beea0893ab7796b1df96625c6b8/special_tokens_map.json"
                            .to_owned(),
                    ),
                },
                ModelFile {
                    name: "tokenizer_config.json".to_owned(),
                    sha256: "a5c2e5a7b1a29a0702cd28c08a399b5ecc110c263009d17f7e3b415f25905fd8"
                        .to_owned(),
                    size: 1_330,
                    url: Some(
                        "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/c5ee24cb16019beea0893ab7796b1df96625c6b8/tokenizer_config.json"
                            .to_owned(),
                    ),
                },
            ],
            license: "Apache-2.0".to_owned(),
            dimension: None, // Cross-encoder produces scores, not embeddings
            tier: Some(ModelTier::Reranker),
            download_size_bytes: 91_724_882,
        }
    }

    /// Return the compiled-in catalog of all built-in model manifests.
    ///
    /// This is the single source of truth for what models frankensearch needs.
    /// The binary always knows what models it requires without network access.
    #[must_use]
    pub fn builtin_catalog() -> ModelManifestCatalog {
        ModelManifestCatalog {
            schema_version: MANIFEST_SCHEMA_VERSION,
            models: vec![
                Self::potion_128m(),
                Self::minilm_v2(),
                Self::ms_marco_reranker(),
            ],
        }
    }

    /// Parse a manifest from JSON and validate basic structure.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` if JSON parsing or validation fails.
    pub fn from_json_str(raw: &str) -> SearchResult<Self> {
        let manifest =
            serde_json::from_str::<Self>(raw).map_err(|source| SearchError::InvalidConfig {
                field: "manifest_json".to_owned(),
                value: truncate_for_error(raw),
                reason: format!("failed to parse manifest JSON: {source}"),
            })?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Serialize this manifest to pretty JSON.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` if serialization fails.
    pub fn to_pretty_json(&self) -> SearchResult<String> {
        serde_json::to_string_pretty(self).map_err(|source| SearchError::InvalidConfig {
            field: "manifest_json".to_owned(),
            value: self.id.clone(),
            reason: format!("failed to serialize manifest: {source}"),
        })
    }

    /// Returns true when all files have non-placeholder concrete checksums.
    #[must_use]
    pub fn has_verified_checksums(&self) -> bool {
        !self.files.is_empty() && self.files.iter().all(ModelFile::has_verified_checksum)
    }

    /// Returns true when revision appears pinned (not empty and not floating aliases).
    #[must_use]
    pub fn has_pinned_revision(&self) -> bool {
        let revision = self.revision.trim();
        !(revision.is_empty()
            || revision.eq_ignore_ascii_case("main")
            || revision.eq_ignore_ascii_case("master")
            || revision.eq_ignore_ascii_case("latest")
            || revision.eq_ignore_ascii_case("head")
            || revision == PLACEHOLDER_PINNED_REVISION)
    }

    /// Returns true when this manifest is ready for production-grade verification.
    #[must_use]
    pub fn is_production_ready(&self) -> bool {
        self.has_verified_checksums() && self.has_pinned_revision()
    }

    /// Sum of expected bytes for all files.
    #[must_use]
    pub fn total_size_bytes(&self) -> u64 {
        if self.download_size_bytes > 0 {
            return self.download_size_bytes;
        }
        self.files.iter().map(|file| file.size).sum()
    }

    /// Validate manifest fields for shape and checksum format.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` for malformed fields.
    pub fn validate(&self) -> SearchResult<()> {
        if self.id.trim().is_empty() {
            return Err(invalid_manifest_field("id", &self.id, "must not be empty"));
        }
        if self.repo.trim().is_empty() {
            return Err(invalid_manifest_field(
                "repo",
                &self.repo,
                "must not be empty",
            ));
        }
        if self.revision.trim().is_empty() {
            return Err(invalid_manifest_field(
                "revision",
                &self.revision,
                "must not be empty",
            ));
        }
        if self.license.trim().is_empty() {
            return Err(invalid_manifest_field(
                "license",
                &self.license,
                "must not be empty",
            ));
        }

        for file in &self.files {
            if file.name.trim().is_empty() {
                return Err(invalid_manifest_field(
                    "files[].name",
                    &file.name,
                    "must not be empty",
                ));
            }
            // Reject path traversal and absolute paths to prevent writes
            // outside the staging/model directory.
            for component in std::path::Path::new(&file.name).components() {
                match component {
                    std::path::Component::ParentDir => {
                        return Err(invalid_manifest_field(
                            "files[].name",
                            &file.name,
                            "must not contain '..' path traversal",
                        ));
                    }
                    std::path::Component::RootDir | std::path::Component::Prefix(_) => {
                        return Err(invalid_manifest_field(
                            "files[].name",
                            &file.name,
                            "must be a relative path without root",
                        ));
                    }
                    _ => {}
                }
            }
            if file.uses_placeholder_checksum() {
                continue;
            }
            if !is_valid_sha256_hex(&file.sha256) {
                return Err(invalid_manifest_field(
                    "files[].sha256",
                    &file.sha256,
                    "must be lowercase 64-char SHA256 hex or placeholder",
                ));
            }
        }

        if self.download_size_bytes > 0 {
            let computed_size: u64 = self.files.iter().map(|file| file.size).sum();
            if computed_size != self.download_size_bytes {
                return Err(invalid_manifest_field(
                    "total_size_bytes",
                    &self.download_size_bytes.to_string(),
                    "must match the sum of files[].size",
                ));
            }
        }

        Ok(())
    }

    /// Enforce checksum policy; placeholder checksums are rejected in release mode.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` if a release policy violation is detected.
    pub fn validate_checksum_policy(&self) -> SearchResult<()> {
        self.validate_checksum_policy_for(cfg!(not(debug_assertions)))
    }

    /// Enforce checksum policy with explicit release-mode toggle (useful for tests).
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` if release-mode requires concrete checksums.
    pub fn validate_checksum_policy_for(&self, release_mode: bool) -> SearchResult<()> {
        if release_mode && self.files.iter().any(ModelFile::uses_placeholder_checksum) {
            return Err(invalid_manifest_field(
                "files[].sha256",
                PLACEHOLDER_VERIFY_AFTER_DOWNLOAD,
                "placeholder checksums are forbidden in release mode",
            ));
        }
        Ok(())
    }

    /// Verify all manifest files in `model_dir` using streaming SHA256 checks.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` when any file is missing or hash/size verification fails.
    pub fn verify_dir(&self, model_dir: &Path) -> SearchResult<()> {
        for file in &self.files {
            let path = model_dir.join(&file.name);
            verify_file_sha256(&path, &file.sha256, file.size)?;
        }
        Ok(())
    }

    /// Promote a staged model directory to final destination atomically after verification.
    ///
    /// Returns the backup path when an existing install was moved out of the way.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if verification or filesystem rename operations fail.
    pub fn promote_verified_installation(
        &self,
        staged_dir: &Path,
        destination_dir: &Path,
    ) -> SearchResult<Option<PathBuf>> {
        self.verify_dir(staged_dir)?;
        promote_atomically(staged_dir, destination_dir)
    }

    /// Return `UpdateAvailable` when installed revision differs from pinned revision.
    #[must_use]
    pub fn detect_update_state(&self, installed_revision: &str) -> Option<ModelState> {
        if !self.has_pinned_revision() {
            return None;
        }
        let current = installed_revision.trim();
        if current == self.revision {
            return None;
        }
        Some(ModelState::UpdateAvailable {
            current_revision: if current.is_empty() {
                "unknown".to_owned()
            } else {
                current.to_owned()
            },
            latest_revision: self.revision.clone(),
        })
    }

    /// Register this manifest in the in-process registry.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if validation fails or registry lock is poisoned.
    pub fn register(self) -> SearchResult<()> {
        self.validate()?;
        manifest_registry()
            .write()
            .map_err(|_| manifest_registry_lock_error("write"))?
            .insert(self.id.clone(), self);
        Ok(())
    }

    /// Look up a registered manifest by id.
    #[must_use]
    pub fn lookup(id: &str) -> Option<Self> {
        let guard = manifest_registry().read().unwrap_or_else(|poisoned| {
            tracing::warn!(
                "model manifest registry lock poisoned on read during lookup; using recovered state"
            );
            poisoned.into_inner()
        });
        guard.get(id).cloned()
    }

    /// Return all registered manifests in deterministic id order.
    #[must_use]
    pub fn registered() -> Vec<Self> {
        let guard = manifest_registry().read().unwrap_or_else(|poisoned| {
            tracing::warn!(
                "model manifest registry lock poisoned on read during listing; using recovered state"
            );
            poisoned.into_inner()
        });
        guard.values().cloned().collect()
    }
}

/// Model manifest catalog for bulk load/validation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelManifestCatalog {
    /// Schema version for forward compatibility.
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    /// Manifests contained in this catalog.
    #[serde(default)]
    pub models: Vec<ModelManifest>,
}

const fn default_schema_version() -> u32 {
    MANIFEST_SCHEMA_VERSION
}

impl Default for ModelManifestCatalog {
    fn default() -> Self {
        Self {
            schema_version: MANIFEST_SCHEMA_VERSION,
            models: Vec::new(),
        }
    }
}

impl ModelManifestCatalog {
    /// Parse a catalog from JSON.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` if parsing fails.
    pub fn from_json_str(raw: &str) -> SearchResult<Self> {
        serde_json::from_str::<Self>(raw).map_err(|source| SearchError::InvalidConfig {
            field: "manifest_catalog_json".to_owned(),
            value: truncate_for_error(raw),
            reason: format!("failed to parse manifest catalog JSON: {source}"),
        })
    }

    /// Validate every manifest in the catalog.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if any contained manifest is invalid.
    pub fn validate(&self) -> SearchResult<()> {
        for model in &self.models {
            model.validate()?;
        }
        Ok(())
    }
}

/// Runtime state of model availability and lifecycle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelState {
    NotInstalled,
    NeedsConsent,
    Downloading {
        progress_pct: u8,
        bytes_downloaded: u64,
        total_bytes: u64,
    },
    Verifying,
    Ready,
    Disabled {
        reason: String,
    },
    VerificationFailed {
        reason: String,
    },
    UpdateAvailable {
        current_revision: String,
        latest_revision: String,
    },
    Cancelled,
}

/// Where a consent decision came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsentSource {
    Programmatic,
    Environment,
    Interactive,
    ConfigFile,
}

/// Resolved consent decision for model downloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DownloadConsent {
    /// Whether downloads are allowed.
    pub granted: bool,
    /// Origin of the consent signal.
    pub source: Option<ConsentSource>,
}

impl DownloadConsent {
    /// Explicitly granted consent.
    #[must_use]
    pub const fn granted(source: ConsentSource) -> Self {
        Self {
            granted: true,
            source: Some(source),
        }
    }

    /// Explicitly denied consent.
    #[must_use]
    pub const fn denied(source: Option<ConsentSource>) -> Self {
        Self {
            granted: false,
            source,
        }
    }
}

/// Resolve download consent using priority:
/// programmatic > environment > interactive > config.
#[must_use]
pub fn resolve_download_consent(
    programmatic: Option<bool>,
    interactive: Option<bool>,
    config_file: Option<bool>,
) -> DownloadConsent {
    let env_value = std::env::var(DOWNLOAD_CONSENT_ENV).ok();
    resolve_download_consent_with_env(programmatic, env_value.as_deref(), interactive, config_file)
}

fn resolve_download_consent_with_env(
    programmatic: Option<bool>,
    env_value: Option<&str>,
    interactive: Option<bool>,
    config_file: Option<bool>,
) -> DownloadConsent {
    if let Some(granted) = programmatic {
        return DownloadConsent {
            granted,
            source: Some(ConsentSource::Programmatic),
        };
    }

    if let Some(raw) = env_value
        && let Some(granted) = parse_bool_flag(raw)
    {
        return DownloadConsent {
            granted,
            source: Some(ConsentSource::Environment),
        };
    }

    if let Some(granted) = interactive {
        return DownloadConsent {
            granted,
            source: Some(ConsentSource::Interactive),
        };
    }

    if let Some(granted) = config_file {
        return DownloadConsent {
            granted,
            source: Some(ConsentSource::ConfigFile),
        };
    }

    DownloadConsent::denied(None)
}

/// Stateful lifecycle helper for model installation progress.
#[derive(Debug, Clone)]
pub struct ModelLifecycle {
    manifest: ModelManifest,
    state: ModelState,
    consent: DownloadConsent,
}

impl ModelLifecycle {
    /// Create lifecycle state for a manifest.
    #[must_use]
    pub const fn new(manifest: ModelManifest, consent: DownloadConsent) -> Self {
        let state = if consent.granted {
            ModelState::NotInstalled
        } else {
            ModelState::NeedsConsent
        };
        Self {
            manifest,
            state,
            consent,
        }
    }

    /// Current lifecycle state.
    #[must_use]
    pub const fn state(&self) -> &ModelState {
        &self.state
    }

    /// Underlying manifest for this lifecycle.
    #[must_use]
    pub const fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }

    /// Mark consent as granted (e.g., after explicit user approval).
    pub fn approve_consent(&mut self, source: ConsentSource) {
        self.consent = DownloadConsent::granted(source);
        if matches!(self.state, ModelState::NeedsConsent) {
            self.state = ModelState::NotInstalled;
        }
    }

    /// Start the download state.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` on invalid transition or zero total bytes.
    pub fn begin_download(&mut self, total_bytes: u64) -> SearchResult<()> {
        if !self.consent.granted {
            self.state = ModelState::NeedsConsent;
            return Err(SearchError::EmbedderUnavailable {
                model: self.manifest.id.clone(),
                reason: "download consent required".to_owned(),
            });
        }
        if total_bytes == 0 {
            return Err(SearchError::InvalidConfig {
                field: "total_bytes".to_owned(),
                value: "0".to_owned(),
                reason: "must be greater than zero".to_owned(),
            });
        }

        match self.state {
            ModelState::NotInstalled
            | ModelState::Cancelled
            | ModelState::VerificationFailed { .. } => {
                self.state = ModelState::Downloading {
                    progress_pct: 0,
                    bytes_downloaded: 0,
                    total_bytes,
                };
                Ok(())
            }
            _ => Err(invalid_state_transition(
                &self.state,
                "begin_download",
                "expected NotInstalled/Cancelled/VerificationFailed",
            )),
        }
    }

    /// Update bytes downloaded and recompute bounded percent.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` if not currently downloading.
    pub fn update_download_progress(&mut self, bytes_downloaded: u64) -> SearchResult<()> {
        let (progress_pct, total_bytes, bounded_bytes) = match self.state {
            ModelState::Downloading { total_bytes, .. } => {
                let bounded = bytes_downloaded.min(total_bytes);
                let pct_u64 = bounded.saturating_mul(100) / total_bytes;
                #[allow(clippy::cast_possible_truncation)]
                let pct = pct_u64 as u8;
                (pct.min(100), total_bytes, bounded)
            }
            _ => {
                return Err(invalid_state_transition(
                    &self.state,
                    "update_download_progress",
                    "expected Downloading",
                ));
            }
        };

        self.state = ModelState::Downloading {
            progress_pct,
            bytes_downloaded: bounded_bytes,
            total_bytes,
        };
        Ok(())
    }

    /// Move from downloading to verifying.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` if not currently downloading.
    pub fn begin_verification(&mut self) -> SearchResult<()> {
        if matches!(self.state, ModelState::Downloading { .. }) {
            self.state = ModelState::Verifying;
            return Ok(());
        }
        Err(invalid_state_transition(
            &self.state,
            "begin_verification",
            "expected Downloading",
        ))
    }

    /// Mark install ready.
    pub fn mark_ready(&mut self) {
        self.state = ModelState::Ready;
    }

    /// Mark install verification failed.
    pub fn fail_verification(&mut self, reason: impl Into<String>) {
        self.state = ModelState::VerificationFailed {
            reason: reason.into(),
        };
    }

    /// Mark model disabled.
    pub fn disable(&mut self, reason: impl Into<String>) {
        self.state = ModelState::Disabled {
            reason: reason.into(),
        };
    }

    /// Mark update available.
    pub fn mark_update_available(
        &mut self,
        current_revision: impl Into<String>,
        latest_revision: impl Into<String>,
    ) {
        self.state = ModelState::UpdateAvailable {
            current_revision: current_revision.into(),
            latest_revision: latest_revision.into(),
        };
    }

    /// Cancel current operation.
    pub fn cancel(&mut self) {
        self.state = ModelState::Cancelled;
    }

    /// Recover from cancelled state so a new download can start.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` if current state is not `Cancelled`.
    pub fn recover_after_cancel(&mut self) -> SearchResult<()> {
        if !matches!(self.state, ModelState::Cancelled) {
            return Err(invalid_state_transition(
                &self.state,
                "recover_after_cancel",
                "expected Cancelled",
            ));
        }
        self.state = if self.consent.granted {
            ModelState::NotInstalled
        } else {
            ModelState::NeedsConsent
        };
        Ok(())
    }
}

/// Verify file size + SHA256 using streaming read.
///
/// # Errors
///
/// Returns `SearchError` when file is missing, unreadable, or hash/size mismatch occurs.
pub fn verify_file_sha256(
    path: &Path,
    expected_sha256: &str,
    expected_size: u64,
) -> SearchResult<()> {
    if expected_sha256 == PLACEHOLDER_VERIFY_AFTER_DOWNLOAD {
        return Err(SearchError::InvalidConfig {
            field: "sha256".to_owned(),
            value: expected_sha256.to_owned(),
            reason: "placeholder checksum cannot be verified".to_owned(),
        });
    }
    if !is_valid_sha256_hex(expected_sha256) {
        return Err(SearchError::InvalidConfig {
            field: "sha256".to_owned(),
            value: expected_sha256.to_owned(),
            reason: "expected lowercase 64-char SHA256 hex".to_owned(),
        });
    }
    if !path.exists() {
        return Err(SearchError::ModelNotFound {
            name: format!("missing model file: {}", path.display()),
        });
    }

    let metadata = fs::metadata(path).map_err(|source| SearchError::ModelLoadFailed {
        path: path.to_path_buf(),
        source: Box::new(source),
    })?;
    if !metadata.is_file() {
        return Err(SearchError::ModelLoadFailed {
            path: path.to_path_buf(),
            source: "expected a regular file".into(),
        });
    }

    let file = File::open(path).map_err(|source| SearchError::ModelLoadFailed {
        path: path.to_path_buf(),
        source: Box::new(source),
    })?;
    let mut reader = BufReader::new(file);
    let mut buffer = [0_u8; HASH_BUFFER_SIZE];
    let mut hasher = Sha256::new();
    let mut bytes_read = 0_u64;

    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|source| SearchError::ModelLoadFailed {
                path: path.to_path_buf(),
                source: Box::new(source),
            })?;
        if read == 0 {
            break;
        }
        let read_u64 = u64::try_from(read).map_err(|_| SearchError::InvalidConfig {
            field: "read_size".to_owned(),
            value: read.to_string(),
            reason: "read size does not fit u64".to_owned(),
        })?;
        bytes_read = bytes_read.saturating_add(read_u64);
        hasher.update(&buffer[..read]);
    }

    let actual_sha256 = to_hex_lowercase(&hasher.finalize());
    let expected_lower = expected_sha256.to_ascii_lowercase();
    if bytes_read != expected_size || actual_sha256 != expected_lower {
        return Err(SearchError::HashMismatch {
            path: path.to_path_buf(),
            expected: format!("sha256={expected_lower},size={expected_size}"),
            actual: format!("sha256={actual_sha256},size={bytes_read}"),
        });
    }

    Ok(())
}

// ─── Verification Cache ────────────────────────────────────────────────────

/// Name of the verification marker file within a model directory.
const VERIFIED_MARKER_FILE: &str = ".verified";

/// Lightweight filesystem fingerprint for one verified model file.
///
/// This is intentionally cheap to read and compare:
/// - file size (bytes)
/// - last-modified timestamp (unix nanos)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileVerificationState {
    /// File size in bytes.
    pub size_bytes: u64,
    /// Last-modified timestamp since unix epoch (nanoseconds).
    pub modified_unix_nanos: u64,
}

fn capture_file_verification_state(path: &Path) -> Option<FileVerificationState> {
    let metadata = fs::metadata(path).ok()?;
    if !metadata.is_file() {
        return None;
    }
    let modified = metadata
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_nanos();
    let modified_unix_nanos = u64::try_from(modified).ok()?;
    Some(FileVerificationState {
        size_bytes: metadata.len(),
        modified_unix_nanos,
    })
}

/// Cached verification result stored as a small JSON file alongside model files.
///
/// When a model directory passes SHA-256 verification, a `.verified` marker is written
/// containing the manifest ID, schema version, and lightweight file fingerprints
/// captured at verification time. Subsequent loads check whether the marker is still
/// valid (all fingerprints unchanged) and skip re-hashing when it is.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationMarker {
    /// Manifest identifier that was verified against.
    pub manifest_id: String,
    /// Manifest schema version at time of verification.
    pub schema_version: u32,
    /// Unix timestamp (seconds) when verification was performed.
    pub verified_at: u64,
    /// Per-file lightweight fingerprint at verification time, keyed by file name.
    pub file_states: BTreeMap<String, FileVerificationState>,
}

impl VerificationMarker {
    /// Create a new marker for a successfully verified model directory.
    fn new_for(manifest: &ModelManifest, model_dir: &Path) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());

        let mut file_states = BTreeMap::new();
        for file in &manifest.files {
            let path = model_dir.join(&file.name);
            if let Some(state) = capture_file_verification_state(&path) {
                file_states.insert(file.name.clone(), state);
            }
        }

        Self {
            manifest_id: manifest.id.clone(),
            schema_version: MANIFEST_SCHEMA_VERSION,
            verified_at: now,
            file_states,
        }
    }

    /// Check whether this cached marker is still valid for the given manifest and directory.
    ///
    /// Returns `true` when:
    /// 1. The manifest ID matches.
    /// 2. The schema version matches.
    /// 3. No model file metadata fingerprint has changed since verification.
    fn is_valid_for(&self, manifest: &ModelManifest, model_dir: &Path) -> bool {
        if self.manifest_id != manifest.id || self.schema_version != MANIFEST_SCHEMA_VERSION {
            return false;
        }

        for file in &manifest.files {
            let Some(expected_state) = self.file_states.get(&file.name) else {
                return false;
            };
            let path = model_dir.join(&file.name);
            let Some(current_state) = capture_file_verification_state(&path) else {
                return false;
            };
            if current_state != *expected_state {
                return false;
            }
        }

        true
    }
}

/// Write a `.verified` marker after successful verification of a model directory.
///
/// The marker is best-effort: failures to write are logged but do not propagate errors.
pub fn write_verification_marker(manifest: &ModelManifest, model_dir: &Path) {
    let marker = VerificationMarker::new_for(manifest, model_dir);
    let Ok(json) = serde_json::to_string_pretty(&marker) else {
        return;
    };
    let _ = (|| -> std::io::Result<()> {
        let mut file = File::create(model_dir.join(VERIFIED_MARKER_FILE))?;
        std::io::Write::write_all(&mut file, json.as_bytes())?;
        file.sync_all()?;
        Ok(())
    })();
}

/// Check whether a valid verification marker exists for the given manifest and directory.
///
/// Returns `true` when a `.verified` file exists, parses correctly, and all file mtimes
/// match. In that case, the caller can skip full SHA-256 re-verification.
#[must_use]
pub fn is_verification_cached(manifest: &ModelManifest, model_dir: &Path) -> bool {
    let path = model_dir.join(VERIFIED_MARKER_FILE);
    let Ok(raw) = fs::read_to_string(&path) else {
        return false;
    };
    let Ok(marker) = serde_json::from_str::<VerificationMarker>(&raw) else {
        return false;
    };
    marker.is_valid_for(manifest, model_dir)
}

/// Verify a model directory, using cached results when available.
///
/// If a valid `.verified` marker exists (matching manifest ID, schema version, and
/// file mtimes), verification succeeds immediately without re-hashing. Otherwise,
/// full SHA-256 verification is performed via [`ModelManifest::verify_dir`], and
/// on success a new marker is written for future loads.
///
/// # Errors
///
/// Returns `SearchError` when the manifest has verified checksums and full
/// verification fails (hash mismatch, missing files, etc.).
pub fn verify_dir_cached(manifest: &ModelManifest, model_dir: &Path) -> SearchResult<()> {
    if !manifest.has_verified_checksums() {
        return Ok(());
    }

    if is_verification_cached(manifest, model_dir) {
        return Ok(());
    }

    manifest.verify_dir(model_dir)?;
    write_verification_marker(manifest, model_dir);
    Ok(())
}

fn promote_atomically(staged_dir: &Path, destination_dir: &Path) -> SearchResult<Option<PathBuf>> {
    let destination_parent =
        destination_dir
            .parent()
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "destination_dir".to_owned(),
                value: destination_dir.display().to_string(),
                reason: "destination must have a parent directory".to_owned(),
            })?;
    fs::create_dir_all(destination_parent).map_err(SearchError::from)?;

    let stage_name = destination_dir.file_name().map_or_else(
        || "model".to_owned(),
        |part| part.to_string_lossy().into_owned(),
    );
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    let pid = std::process::id();
    let stage_target =
        destination_parent.join(format!(".{stage_name}.installing.{timestamp}.{pid}"));
    fs::rename(staged_dir, &stage_target).map_err(SearchError::from)?;

    let backup_path = if destination_dir.exists() {
        let backup = destination_parent.join(format!("{stage_name}.backup.{timestamp}.{pid}"));
        fs::rename(destination_dir, &backup).map_err(SearchError::from)?;
        Some(backup)
    } else {
        None
    };

    fs::rename(&stage_target, destination_dir).map_err(SearchError::from)?;
    Ok(backup_path)
}

fn manifest_registry() -> &'static RwLock<BTreeMap<String, ModelManifest>> {
    static REGISTRY: OnceLock<RwLock<BTreeMap<String, ModelManifest>>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let catalog = ModelManifest::builtin_catalog();
        let mut data = BTreeMap::new();
        for manifest in catalog.models {
            data.insert(manifest.id.clone(), manifest);
        }
        RwLock::new(data)
    })
}

fn manifest_registry_lock_error(action: &str) -> SearchError {
    SearchError::SubsystemError {
        subsystem: "model_manifest",
        source: std::io::Error::other(format!("manifest registry {action} lock poisoned")).into(),
    }
}

fn invalid_manifest_field(field: &str, value: &str, reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: field.to_owned(),
        value: value.to_owned(),
        reason: reason.to_owned(),
    }
}

fn invalid_state_transition(state: &ModelState, operation: &str, reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: "model_state".to_owned(),
        value: format!("{state:?}"),
        reason: format!("invalid transition for {operation}: {reason}"),
    }
}

fn truncate_for_error(value: &str) -> String {
    const MAX: usize = 120;
    let mut chars = value.chars();
    let truncated: String = chars.by_ref().take(MAX).collect();
    if chars.next().is_none() {
        return truncated;
    }
    let mut out = truncated;
    out.push_str("...");
    out
}

fn parse_bool_flag(raw: &str) -> Option<bool> {
    let value = raw.trim();
    if value == "1"
        || value.eq_ignore_ascii_case("true")
        || value.eq_ignore_ascii_case("yes")
        || value.eq_ignore_ascii_case("on")
    {
        return Some(true);
    }
    if value == "0"
        || value.eq_ignore_ascii_case("false")
        || value.eq_ignore_ascii_case("no")
        || value.eq_ignore_ascii_case("off")
    {
        return Some(false);
    }
    None
}

fn is_valid_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn to_hex_lowercase(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp_file(path: &Path, bytes: &[u8]) {
        let mut file = File::create(path).unwrap();
        file.write_all(bytes).unwrap();
        file.flush().unwrap();
    }

    #[test]
    fn invalid_manifest_json_returns_clear_error() {
        let err = ModelManifest::from_json_str("{not-valid-json]").unwrap_err();
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
        assert!(err.to_string().contains("manifest JSON"));
    }

    #[test]
    fn valid_manifest_json_round_trips_expected_fields() {
        let manifest = ModelManifest::from_json_str(
            r#"{
                "id":"test-model",
                "repo":"acme/test-model",
                "revision":"0123456789abcdef0123456789abcdef01234567",
                "files":[
                    {
                        "name":"model.bin",
                        "sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                        "size":42
                    }
                ],
                "license":"MIT"
            }"#,
        )
        .unwrap();

        assert_eq!(manifest.id, "test-model");
        assert_eq!(manifest.repo, "acme/test-model");
        assert_eq!(manifest.total_size_bytes(), 42);
        assert!(manifest.has_verified_checksums());
        assert!(manifest.has_pinned_revision());
        assert!(manifest.is_production_ready());
    }

    #[test]
    fn missing_required_manifest_field_surfaces_field_name() {
        let err = ModelManifest::from_json_str(
            r#"{
                "id":"test-model",
                "repo":"acme/test-model",
                "revision":"0123456789abcdef0123456789abcdef01234567",
                "files":[]
            }"#,
        )
        .unwrap_err();

        assert!(matches!(err, SearchError::InvalidConfig { .. }));
        assert!(err.to_string().contains("license"));
    }

    #[test]
    fn verify_file_sha256_success_wrong_hash_and_truncated() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("model.bin");
        let bytes = b"model-bytes";
        write_temp_file(&path, bytes);

        let expected_hash = to_hex_lowercase(&Sha256::digest(bytes));
        let expected_size = u64::try_from(bytes.len()).unwrap();
        verify_file_sha256(&path, &expected_hash, expected_size).unwrap();

        let wrong_hash = "0000000000000000000000000000000000000000000000000000000000000000";
        let err = verify_file_sha256(&path, wrong_hash, expected_size).unwrap_err();
        assert!(matches!(err, SearchError::HashMismatch { .. }));

        let err = verify_file_sha256(&path, &expected_hash, expected_size + 1).unwrap_err();
        assert!(matches!(err, SearchError::HashMismatch { .. }));
    }

    #[test]
    fn verify_file_sha256_rejects_placeholder_invalid_hash_and_missing_file() {
        let temp = tempfile::tempdir().unwrap();
        let missing_path = temp.path().join("missing.bin");

        let err =
            verify_file_sha256(&missing_path, PLACEHOLDER_VERIFY_AFTER_DOWNLOAD, 1).unwrap_err();
        assert!(matches!(err, SearchError::InvalidConfig { .. }));

        let err = verify_file_sha256(&missing_path, "NOT-A-HASH", 1).unwrap_err();
        assert!(matches!(err, SearchError::InvalidConfig { .. }));

        let valid_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let err = verify_file_sha256(&missing_path, valid_hash, 1).unwrap_err();
        assert!(matches!(err, SearchError::ModelNotFound { .. }));
    }

    #[test]
    fn catalog_validate_reports_invalid_nested_manifest() {
        let catalog = ModelManifestCatalog::from_json_str(
            r#"{
                "models":[
                    {
                        "id":"bad-model",
                        "repo":"acme/bad-model",
                        "revision":"0123456789abcdef0123456789abcdef01234567",
                        "files":[
                            {"name":"model.bin","sha256":"bad-hash","size":10}
                        ],
                        "license":"MIT"
                    }
                ]
            }"#,
        )
        .unwrap();

        let err = catalog.validate().unwrap_err();
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[test]
    fn lifecycle_state_machine_success_path() {
        let manifest = ModelManifest::potion_128m();
        let mut lifecycle = ModelLifecycle::new(
            manifest,
            DownloadConsent::granted(ConsentSource::Programmatic),
        );

        assert_eq!(lifecycle.state(), &ModelState::NotInstalled);

        lifecycle.begin_download(100).unwrap();
        lifecycle.update_download_progress(40).unwrap();
        lifecycle.begin_verification().unwrap();
        lifecycle.mark_ready();

        assert_eq!(lifecycle.state(), &ModelState::Ready);
    }

    #[test]
    fn lifecycle_state_machine_failure_path() {
        let manifest = ModelManifest::potion_128m();
        let mut lifecycle = ModelLifecycle::new(
            manifest,
            DownloadConsent::granted(ConsentSource::Programmatic),
        );

        lifecycle.begin_download(100).unwrap();
        lifecycle.fail_verification("checksum mismatch");
        assert!(matches!(
            lifecycle.state(),
            ModelState::VerificationFailed { .. }
        ));
    }

    #[test]
    fn download_progress_percent_is_bounded_to_100() {
        let manifest = ModelManifest::minilm_v2();
        let mut lifecycle = ModelLifecycle::new(
            manifest,
            DownloadConsent::granted(ConsentSource::Programmatic),
        );
        lifecycle.begin_download(10).unwrap();
        lifecycle.update_download_progress(10_000).unwrap();

        let progress_pct = match lifecycle.state() {
            ModelState::Downloading { progress_pct, .. } => *progress_pct,
            _ => 0,
        };
        assert!(progress_pct <= 100);
        assert_eq!(progress_pct, 100);
    }

    #[test]
    fn placeholder_checksums_are_rejected_in_release_policy_mode() {
        let mut manifest = ModelManifest::minilm_v2();
        manifest.files[0].sha256 = PLACEHOLDER_VERIFY_AFTER_DOWNLOAD.to_owned();
        manifest.files[0].size = 0;
        manifest.files[0].url = None;
        let err = manifest.validate_checksum_policy_for(true).unwrap_err();
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[test]
    fn cancelled_state_can_recover() {
        let manifest = ModelManifest::potion_128m();
        let mut lifecycle = ModelLifecycle::new(
            manifest,
            DownloadConsent::granted(ConsentSource::Programmatic),
        );
        lifecycle.begin_download(10).unwrap();
        lifecycle.cancel();
        lifecycle.recover_after_cancel().unwrap();
        assert_eq!(lifecycle.state(), &ModelState::NotInstalled);
    }

    #[test]
    fn empty_manifest_catalog_is_valid() {
        let catalog = ModelManifestCatalog::from_json_str(r#"{"models":[]}"#).unwrap();
        assert!(catalog.models.is_empty());
        catalog.validate().unwrap();
    }

    #[test]
    fn unreadable_model_file_returns_clear_error() {
        let temp = tempfile::tempdir().unwrap();
        let model_root = temp.path();
        let bogus_path = model_root.join("tokenizer.json");
        fs::create_dir_all(&bogus_path).unwrap();

        let manifest = ModelManifest {
            id: "test".to_owned(),
            version: "test-v1".to_owned(),
            display_name: None,
            description: None,
            repo: "owner/repo".to_owned(),
            revision: "abcdef1".to_owned(),
            files: vec![ModelFile {
                name: "tokenizer.json".to_owned(),
                sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    .to_owned(),
                size: 1,
                url: None,
            }],
            license: "MIT".to_owned(),
            dimension: None,
            tier: None,
            download_size_bytes: 0,
        };

        let err = manifest.verify_dir(model_root).unwrap_err();
        assert!(matches!(err, SearchError::ModelLoadFailed { .. }));
        assert!(err.to_string().contains("regular file"));
    }

    #[test]
    fn can_register_and_lookup_custom_manifest() {
        let unique_id = format!(
            "custom-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let manifest = ModelManifest {
            id: unique_id.clone(),
            version: "test-v1".to_owned(),
            display_name: None,
            description: None,
            repo: "acme/custom".to_owned(),
            revision: "0123456789abcdef0123456789abcdef01234567".to_owned(),
            files: vec![ModelFile {
                name: "weights.bin".to_owned(),
                sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    .to_owned(),
                size: 42,
                url: None,
            }],
            license: "MIT".to_owned(),
            dimension: None,
            tier: None,
            download_size_bytes: 0,
        };

        manifest.clone().register().unwrap();
        let loaded = ModelManifest::lookup(&unique_id).unwrap();
        assert_eq!(loaded, manifest);
    }

    #[test]
    fn resolve_download_consent_priority_order() {
        let consent =
            resolve_download_consent_with_env(Some(false), Some("1"), Some(true), Some(true));
        assert_eq!(consent.source, Some(ConsentSource::Programmatic));
        assert!(!consent.granted);

        let consent = resolve_download_consent_with_env(None, Some("1"), Some(false), Some(true));
        assert_eq!(consent.source, Some(ConsentSource::Environment));
        assert!(consent.granted);

        let consent = resolve_download_consent_with_env(None, None, Some(false), Some(true));
        assert_eq!(consent.source, Some(ConsentSource::Interactive));
        assert!(!consent.granted);
    }

    // ── bd-3un.51: Additional coverage ───────────────────────────────

    #[test]
    fn valid_manifest_parses_all_fields() {
        let json = r#"{
            "id": "test-model",
            "repo": "owner/test-model",
            "revision": "abc123def456",
            "files": [
                {"name": "model.onnx", "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "size": 1024},
                {"name": "tokenizer.json", "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "size": 512}
            ],
            "license": "Apache-2.0"
        }"#;
        let manifest = ModelManifest::from_json_str(json).unwrap();
        assert_eq!(manifest.id, "test-model");
        assert_eq!(manifest.repo, "owner/test-model");
        assert_eq!(manifest.revision, "abc123def456");
        assert_eq!(manifest.files.len(), 2);
        assert_eq!(manifest.files[0].name, "model.onnx");
        assert_eq!(manifest.files[1].size, 512);
        assert_eq!(manifest.license, "Apache-2.0");
    }

    #[test]
    fn missing_id_field_returns_clear_error() {
        let json = r#"{"id": "", "repo": "r", "revision": "v", "files": [], "license": "MIT"}"#;
        let err = ModelManifest::from_json_str(json).unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn missing_repo_field_returns_clear_error() {
        let json = r#"{"id": "m", "repo": " ", "revision": "v", "files": [], "license": "MIT"}"#;
        let err = ModelManifest::from_json_str(json).unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn missing_revision_field_returns_clear_error() {
        let json = r#"{"id": "m", "repo": "r", "revision": "", "files": [], "license": "MIT"}"#;
        let err = ModelManifest::from_json_str(json).unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn missing_license_field_returns_clear_error() {
        let json = r#"{"id": "m", "repo": "r", "revision": "v", "files": [], "license": ""}"#;
        let err = ModelManifest::from_json_str(json).unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn invalid_sha256_format_rejected() {
        let json = r#"{
            "id": "m", "repo": "r", "revision": "v", "license": "MIT",
            "files": [{"name": "f.bin", "sha256": "not-a-valid-hash", "size": 1}]
        }"#;
        let err = ModelManifest::from_json_str(json).unwrap_err();
        assert!(err.to_string().contains("SHA256 hex"));
    }

    #[test]
    fn file_with_zero_size_and_non_placeholder_hash_rejected() {
        let json = r#"{
            "id": "m", "repo": "r", "revision": "v", "license": "MIT",
            "files": [{"name": "f.bin", "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "size": 0}]
        }"#;
        let err = ModelManifest::from_json_str(json).unwrap_err();
        assert!(err.to_string().contains("must be > 0"));
    }

    #[test]
    fn empty_file_name_rejected() {
        let json = r#"{
            "id": "m", "repo": "r", "revision": "v", "license": "MIT",
            "files": [{"name": "", "sha256": "PLACEHOLDER_VERIFY_AFTER_DOWNLOAD", "size": 0}]
        }"#;
        let err = ModelManifest::from_json_str(json).unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn verify_missing_file_returns_model_not_found() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("does_not_exist.bin");
        let hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let err = verify_file_sha256(&path, hash, 100).unwrap_err();
        assert!(matches!(err, SearchError::ModelNotFound { .. }));
    }

    #[test]
    fn verify_placeholder_checksum_rejected() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("file.bin");
        write_temp_file(&path, b"data");
        let err = verify_file_sha256(&path, PLACEHOLDER_VERIFY_AFTER_DOWNLOAD, 4).unwrap_err();
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
        assert!(err.to_string().contains("placeholder"));
    }

    #[test]
    fn verify_zero_expected_size_accepts_empty_file() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("empty.bin");
        write_temp_file(&path, b"");
        let hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        verify_file_sha256(&path, hash, 0).unwrap();
    }

    #[test]
    fn verify_zero_expected_size_still_rejects_non_empty_file() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("file.bin");
        write_temp_file(&path, b"data");
        let hash = to_hex_lowercase(&Sha256::digest(b"data"));
        let err = verify_file_sha256(&path, &hash, 0).unwrap_err();
        assert!(matches!(err, SearchError::HashMismatch { .. }));
    }

    #[test]
    fn to_pretty_json_roundtrip() {
        let manifest = ModelManifest::potion_128m();
        let json = manifest.to_pretty_json().unwrap();
        let restored = ModelManifest::from_json_str(&json).unwrap();
        assert_eq!(restored.id, manifest.id);
        assert_eq!(restored.files.len(), manifest.files.len());
    }

    #[test]
    fn builtin_manifests_validate() {
        ModelManifest::minilm_v2().validate().unwrap();
        ModelManifest::potion_128m().validate().unwrap();
    }

    #[test]
    fn builtin_manifests_are_production_ready() {
        assert!(ModelManifest::minilm_v2().is_production_ready());
        assert!(ModelManifest::potion_128m().is_production_ready());
        assert!(ModelManifest::ms_marco_reranker().is_production_ready());
    }

    #[test]
    fn has_pinned_revision_rejects_floating_aliases() {
        for alias in &[
            "main",
            "master",
            "latest",
            "HEAD",
            PLACEHOLDER_PINNED_REVISION,
        ] {
            let m = ModelManifest {
                revision: alias.to_string(),
                ..ModelManifest::potion_128m()
            };
            assert!(
                !m.has_pinned_revision(),
                "'{alias}' should not be considered pinned"
            );
        }
    }

    #[test]
    fn has_pinned_revision_accepts_commit_sha() {
        let m = ModelManifest {
            revision: "0123456789abcdef0123456789abcdef01234567".to_owned(),
            ..ModelManifest::potion_128m()
        };
        assert!(m.has_pinned_revision());
    }

    #[test]
    fn total_size_bytes_sums_all_files() {
        let m = ModelManifest {
            files: vec![
                ModelFile {
                    name: "a".to_owned(),
                    sha256: PLACEHOLDER_VERIFY_AFTER_DOWNLOAD.to_owned(),
                    size: 100,
                    url: None,
                },
                ModelFile {
                    name: "b".to_owned(),
                    sha256: PLACEHOLDER_VERIFY_AFTER_DOWNLOAD.to_owned(),
                    size: 200,
                    url: None,
                },
            ],
            download_size_bytes: 0,
            ..ModelManifest::potion_128m()
        };
        assert_eq!(m.total_size_bytes(), 300);
    }

    #[test]
    fn model_state_serde_roundtrip() {
        let states = vec![
            ModelState::NotInstalled,
            ModelState::NeedsConsent,
            ModelState::Downloading {
                progress_pct: 50,
                bytes_downloaded: 1000,
                total_bytes: 2000,
            },
            ModelState::Verifying,
            ModelState::Ready,
            ModelState::Disabled {
                reason: "out of disk".to_owned(),
            },
            ModelState::VerificationFailed {
                reason: "hash mismatch".to_owned(),
            },
            ModelState::UpdateAvailable {
                current_revision: "old".to_owned(),
                latest_revision: "new".to_owned(),
            },
            ModelState::Cancelled,
        ];
        for state in &states {
            let json = serde_json::to_string(state).unwrap();
            let decoded: ModelState = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, state);
        }
    }

    #[test]
    fn consent_source_serde_roundtrip() {
        for source in &[
            ConsentSource::Programmatic,
            ConsentSource::Environment,
            ConsentSource::Interactive,
            ConsentSource::ConfigFile,
        ] {
            let json = serde_json::to_string(source).unwrap();
            let decoded: ConsentSource = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, source);
        }
    }

    #[test]
    fn lifecycle_needs_consent_when_not_granted() {
        let manifest = ModelManifest::potion_128m();
        let lifecycle = ModelLifecycle::new(manifest, DownloadConsent::denied(None));
        assert_eq!(lifecycle.state(), &ModelState::NeedsConsent);
    }

    #[test]
    fn lifecycle_begin_download_without_consent_fails() {
        let manifest = ModelManifest::potion_128m();
        let mut lifecycle = ModelLifecycle::new(manifest, DownloadConsent::denied(None));
        let err = lifecycle.begin_download(100).unwrap_err();
        assert!(matches!(err, SearchError::EmbedderUnavailable { .. }));
    }

    #[test]
    fn lifecycle_begin_download_zero_bytes_fails() {
        let manifest = ModelManifest::potion_128m();
        let mut lifecycle = ModelLifecycle::new(
            manifest,
            DownloadConsent::granted(ConsentSource::Programmatic),
        );
        let err = lifecycle.begin_download(0).unwrap_err();
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[test]
    fn lifecycle_approve_consent_transitions() {
        let manifest = ModelManifest::potion_128m();
        let mut lifecycle = ModelLifecycle::new(manifest, DownloadConsent::denied(None));
        assert_eq!(lifecycle.state(), &ModelState::NeedsConsent);

        lifecycle.approve_consent(ConsentSource::Interactive);
        assert_eq!(lifecycle.state(), &ModelState::NotInstalled);
    }

    #[test]
    fn lifecycle_disable_and_update() {
        let manifest = ModelManifest::potion_128m();
        let mut lifecycle = ModelLifecycle::new(
            manifest,
            DownloadConsent::granted(ConsentSource::Programmatic),
        );

        lifecycle.disable("maintenance");
        assert!(matches!(lifecycle.state(), ModelState::Disabled { .. }));

        lifecycle.mark_update_available("v1", "v2");
        assert!(matches!(
            lifecycle.state(),
            ModelState::UpdateAvailable { .. }
        ));
    }

    #[test]
    fn lifecycle_recovery_from_non_cancelled_fails() {
        let manifest = ModelManifest::potion_128m();
        let mut lifecycle = ModelLifecycle::new(
            manifest,
            DownloadConsent::granted(ConsentSource::Programmatic),
        );
        let err = lifecycle.recover_after_cancel().unwrap_err();
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[test]
    fn lifecycle_begin_verification_from_not_downloading_fails() {
        let manifest = ModelManifest::potion_128m();
        let mut lifecycle = ModelLifecycle::new(
            manifest,
            DownloadConsent::granted(ConsentSource::Programmatic),
        );
        let err = lifecycle.begin_verification().unwrap_err();
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[test]
    fn lifecycle_update_progress_from_not_downloading_fails() {
        let manifest = ModelManifest::potion_128m();
        let mut lifecycle = ModelLifecycle::new(
            manifest,
            DownloadConsent::granted(ConsentSource::Programmatic),
        );
        let err = lifecycle.update_download_progress(50).unwrap_err();
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[test]
    fn detect_update_state_same_revision_returns_none() {
        let m = ModelManifest {
            revision: "abc123".to_owned(),
            ..ModelManifest::potion_128m()
        };
        assert!(m.detect_update_state("abc123").is_none());
    }

    #[test]
    fn detect_update_state_different_revision_returns_update() {
        let m = ModelManifest {
            revision: "new_rev".to_owned(),
            ..ModelManifest::potion_128m()
        };
        let state = m.detect_update_state("old_rev").unwrap();
        assert!(matches!(state, ModelState::UpdateAvailable { .. }));
    }

    #[test]
    fn detect_update_state_unpinned_returns_none() {
        let manifest = ModelManifest {
            revision: PLACEHOLDER_PINNED_REVISION.to_owned(),
            ..ModelManifest::potion_128m()
        };
        assert!(manifest.detect_update_state("anything").is_none());
    }

    #[test]
    fn resolve_consent_config_file_path() {
        let consent = resolve_download_consent_with_env(None, None, None, Some(true));
        assert_eq!(consent.source, Some(ConsentSource::ConfigFile));
        assert!(consent.granted);
    }

    #[test]
    fn resolve_consent_no_source_denies() {
        let consent = resolve_download_consent_with_env(None, None, None, None);
        assert!(!consent.granted);
        assert!(consent.source.is_none());
    }

    #[test]
    fn resolve_consent_env_values() {
        for (val, expected) in &[
            ("1", true),
            ("true", true),
            ("yes", true),
            ("on", true),
            ("0", false),
            ("false", false),
            ("no", false),
            ("off", false),
        ] {
            let consent = resolve_download_consent_with_env(None, Some(val), None, None);
            assert_eq!(consent.granted, *expected, "env={val}");
        }
    }

    #[test]
    fn resolve_consent_invalid_env_skipped() {
        let consent = resolve_download_consent_with_env(None, Some("maybe"), Some(true), None);
        assert_eq!(consent.source, Some(ConsentSource::Interactive));
        assert!(consent.granted);
    }

    #[test]
    fn model_file_placeholder_detection() {
        let file = ModelFile {
            name: "f.bin".to_owned(),
            sha256: PLACEHOLDER_VERIFY_AFTER_DOWNLOAD.to_owned(),
            size: 0,
            url: None,
        };
        assert!(file.uses_placeholder_checksum());
        assert!(!file.has_verified_checksum());
    }

    #[test]
    fn model_file_verified_checksum_detection() {
        let file = ModelFile {
            name: "f.bin".to_owned(),
            sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
            size: 42,
            url: None,
        };
        assert!(!file.uses_placeholder_checksum());
        assert!(file.has_verified_checksum());
    }

    #[test]
    fn model_file_zero_byte_verified_checksum_detection() {
        let file = ModelFile {
            name: "empty.bin".to_owned(),
            sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".to_owned(),
            size: 0,
            url: None,
        };
        assert!(!file.uses_placeholder_checksum());
        assert!(file.has_verified_checksum());
    }

    #[test]
    fn promote_verified_installation_success() {
        let temp = tempfile::tempdir().unwrap();
        let staged = temp.path().join("staged");
        let dest = temp.path().join("final");
        fs::create_dir_all(&staged).unwrap();

        let data = b"model data";
        write_temp_file(&staged.join("model.bin"), data);
        let hash = to_hex_lowercase(&Sha256::digest(data));
        let size = u64::try_from(data.len()).unwrap();

        let manifest = ModelManifest {
            id: "test".to_owned(),
            version: "test-v1".to_owned(),
            display_name: None,
            description: None,
            repo: "owner/repo".to_owned(),
            revision: "abc".to_owned(),
            files: vec![ModelFile {
                name: "model.bin".to_owned(),
                sha256: hash,
                size,
                url: None,
            }],
            license: "MIT".to_owned(),
            dimension: None,
            tier: None,
            download_size_bytes: 0,
        };

        let backup = manifest
            .promote_verified_installation(&staged, &dest)
            .unwrap();
        assert!(backup.is_none());
        assert!(dest.join("model.bin").exists());
    }

    #[test]
    fn promote_verified_creates_backup_of_existing() {
        let temp = tempfile::tempdir().unwrap();
        let staged = temp.path().join("staged");
        let dest = temp.path().join("final");
        fs::create_dir_all(&staged).unwrap();
        fs::create_dir_all(&dest).unwrap();
        write_temp_file(&dest.join("old.bin"), b"old");

        let data = b"new model";
        write_temp_file(&staged.join("model.bin"), data);
        let hash = to_hex_lowercase(&Sha256::digest(data));
        let size = u64::try_from(data.len()).unwrap();

        let manifest = ModelManifest {
            id: "test".to_owned(),
            version: "test-v1".to_owned(),
            display_name: None,
            description: None,
            repo: "owner/repo".to_owned(),
            revision: "abc".to_owned(),
            files: vec![ModelFile {
                name: "model.bin".to_owned(),
                sha256: hash,
                size,
                url: None,
            }],
            license: "MIT".to_owned(),
            dimension: None,
            tier: None,
            download_size_bytes: 0,
        };

        let backup = manifest
            .promote_verified_installation(&staged, &dest)
            .unwrap();
        assert!(backup.is_some());
        assert!(dest.join("model.bin").exists());
    }

    #[test]
    fn manifest_catalog_with_multiple_models() {
        let json = r#"{"models": [
            {"id": "m1", "repo": "r1", "revision": "v1", "files": [], "license": "MIT"},
            {"id": "m2", "repo": "r2", "revision": "v2", "files": [], "license": "Apache-2.0"}
        ]}"#;
        let catalog = ModelManifestCatalog::from_json_str(json).unwrap();
        assert_eq!(catalog.models.len(), 2);
        catalog.validate().unwrap();
    }

    #[test]
    fn manifest_catalog_invalid_model_fails_validation() {
        let json = r#"{"models": [
            {"id": "", "repo": "r", "revision": "v", "files": [], "license": "MIT"}
        ]}"#;
        let catalog = ModelManifestCatalog::from_json_str(json).unwrap();
        let err = catalog.validate().unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn is_valid_sha256_hex_checks() {
        assert!(is_valid_sha256_hex(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        ));
        assert!(!is_valid_sha256_hex("short"));
        // Uppercase rejected.
        assert!(!is_valid_sha256_hex(
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        ));
        // Invalid hex chars rejected.
        assert!(!is_valid_sha256_hex(
            "gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg"
        ));
    }

    #[test]
    fn download_consent_constructors() {
        let granted = DownloadConsent::granted(ConsentSource::Programmatic);
        assert!(granted.granted);
        assert_eq!(granted.source, Some(ConsentSource::Programmatic));

        let denied = DownloadConsent::denied(Some(ConsentSource::Environment));
        assert!(!denied.granted);
        assert_eq!(denied.source, Some(ConsentSource::Environment));

        let denied_none = DownloadConsent::denied(None);
        assert!(!denied_none.granted);
        assert!(denied_none.source.is_none());
    }

    #[test]
    fn lifecycle_can_restart_after_verification_failure() {
        let manifest = ModelManifest::potion_128m();
        let mut lifecycle = ModelLifecycle::new(
            manifest,
            DownloadConsent::granted(ConsentSource::Programmatic),
        );

        lifecycle.begin_download(100).unwrap();
        lifecycle.fail_verification("bad hash");
        assert!(matches!(
            lifecycle.state(),
            ModelState::VerificationFailed { .. }
        ));

        lifecycle.begin_download(100).unwrap();
        assert!(matches!(lifecycle.state(), ModelState::Downloading { .. }));
    }

    #[test]
    fn lifecycle_double_begin_download_from_ready_fails() {
        let manifest = ModelManifest::potion_128m();
        let mut lifecycle = ModelLifecycle::new(
            manifest,
            DownloadConsent::granted(ConsentSource::Programmatic),
        );

        lifecycle.begin_download(100).unwrap();
        lifecycle.begin_verification().unwrap();
        lifecycle.mark_ready();

        let err = lifecycle.begin_download(200).unwrap_err();
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[test]
    fn truncate_for_error_short_passthrough() {
        let short = "hello world";
        assert_eq!(truncate_for_error(short), "hello world");
    }

    #[test]
    fn truncate_for_error_long_truncated() {
        let long = "x".repeat(200);
        let result = truncate_for_error(&long);
        assert!(result.ends_with("..."));
        assert!(result.len() < 200);
    }

    // ── bd-2w7x.5: Model manifest enrichment tests ─────────────────────

    #[test]
    fn manifest_schema_version_is_two() {
        assert_eq!(MANIFEST_SCHEMA_VERSION, 2);
    }

    #[test]
    fn model_tier_serde_roundtrip() {
        for tier in &[ModelTier::Fast, ModelTier::Quality, ModelTier::Reranker] {
            let json = serde_json::to_string(tier).unwrap();
            let decoded: ModelTier = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, tier);
        }
    }

    #[test]
    fn model_tier_serde_uses_snake_case() {
        assert_eq!(serde_json::to_string(&ModelTier::Fast).unwrap(), "\"fast\"");
        assert_eq!(
            serde_json::to_string(&ModelTier::Quality).unwrap(),
            "\"quality\""
        );
        assert_eq!(
            serde_json::to_string(&ModelTier::Reranker).unwrap(),
            "\"reranker\""
        );
    }

    #[test]
    fn builtin_potion_has_correct_metadata() {
        let m = ModelManifest::potion_128m();
        assert_eq!(m.dimension, Some(256));
        assert_eq!(m.tier, Some(ModelTier::Fast));
        assert!(m.display_name.is_some());
        assert!(m.display_name.as_deref().unwrap().contains("fast"));
    }

    #[test]
    fn builtin_minilm_has_correct_metadata() {
        let m = ModelManifest::minilm_v2();
        assert_eq!(m.dimension, Some(384));
        assert_eq!(m.tier, Some(ModelTier::Quality));
        assert!(m.display_name.is_some());
        assert!(m.display_name.as_deref().unwrap().contains("quality"));
    }

    #[test]
    fn builtin_reranker_has_correct_metadata() {
        let m = ModelManifest::ms_marco_reranker();
        assert_eq!(m.id, "ms-marco-minilm-l-6-v2");
        assert_eq!(m.dimension, None); // Cross-encoder, no embedding dim
        assert_eq!(m.tier, Some(ModelTier::Reranker));
        assert!(m.display_name.is_some());
        assert!(m.display_name.as_deref().unwrap().contains("reranker"));
        m.validate().unwrap();
    }

    #[test]
    fn builtin_catalog_contains_all_three_models() {
        let catalog = ModelManifest::builtin_catalog();
        assert_eq!(catalog.schema_version, MANIFEST_SCHEMA_VERSION);
        assert_eq!(catalog.models.len(), 3);

        let ids: Vec<&str> = catalog.models.iter().map(|m| m.id.as_str()).collect();
        assert!(ids.contains(&"potion-multilingual-128m"));
        assert!(ids.contains(&"all-minilm-l6-v2"));
        assert!(ids.contains(&"ms-marco-minilm-l-6-v2"));

        catalog.validate().unwrap();
    }

    #[test]
    fn builtin_catalog_covers_all_tiers() {
        let catalog = ModelManifest::builtin_catalog();
        let tiers: Vec<Option<ModelTier>> = catalog.models.iter().map(|m| m.tier).collect();
        assert!(tiers.contains(&Some(ModelTier::Fast)));
        assert!(tiers.contains(&Some(ModelTier::Quality)));
        assert!(tiers.contains(&Some(ModelTier::Reranker)));
    }

    #[test]
    fn builtin_manifests_include_version_description_and_size_metadata() {
        let manifests = [
            ModelManifest::potion_128m(),
            ModelManifest::minilm_v2(),
            ModelManifest::ms_marco_reranker(),
        ];

        for manifest in manifests {
            assert!(!manifest.version.is_empty());
            assert!(manifest.description.is_some());
            assert!(manifest.download_size_bytes > 0);
            let summed_size: u64 = manifest.files.iter().map(|file| file.size).sum();
            assert_eq!(manifest.download_size_bytes, summed_size);
        }
    }

    #[test]
    fn model_file_download_url_uses_explicit_when_present() {
        let file = ModelFile {
            name: "model.onnx".to_owned(),
            sha256: PLACEHOLDER_VERIFY_AFTER_DOWNLOAD.to_owned(),
            size: 0,
            url: Some("https://mirror.example.com/model.onnx".to_owned()),
        };
        let url = file.download_url("owner/repo", "abc123");
        assert_eq!(url, "https://mirror.example.com/model.onnx");
    }

    #[test]
    fn model_file_download_url_derives_from_repo_when_none() {
        let file = ModelFile {
            name: "onnx/model.onnx".to_owned(),
            sha256: PLACEHOLDER_VERIFY_AFTER_DOWNLOAD.to_owned(),
            size: 0,
            url: None,
        };
        let url = file.download_url("sentence-transformers/all-MiniLM-L6-v2", "abc123");
        assert_eq!(
            url,
            "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/abc123/onnx/model.onnx"
        );
    }

    #[test]
    fn model_file_url_field_is_optional_in_json() {
        // URL absent: should deserialize with url=None
        let json = r#"{"name":"f.bin","sha256":"PLACEHOLDER_VERIFY_AFTER_DOWNLOAD","size":0}"#;
        let file: ModelFile = serde_json::from_str(json).unwrap();
        assert!(file.url.is_none());

        // URL present: should deserialize
        let json = r#"{"name":"f.bin","sha256":"PLACEHOLDER_VERIFY_AFTER_DOWNLOAD","size":0,"url":"https://example.com/f.bin"}"#;
        let file: ModelFile = serde_json::from_str(json).unwrap();
        assert_eq!(file.url.as_deref(), Some("https://example.com/f.bin"));
    }

    #[test]
    fn model_file_url_skipped_in_serialization_when_none() {
        let file = ModelFile {
            name: "f.bin".to_owned(),
            sha256: PLACEHOLDER_VERIFY_AFTER_DOWNLOAD.to_owned(),
            size: 0,
            url: None,
        };
        let json = serde_json::to_string(&file).unwrap();
        assert!(!json.contains("url"));
    }

    #[test]
    fn manifest_display_name_optional_in_json() {
        let json = r#"{
            "id": "test", "repo": "r", "revision": "v",
            "files": [], "license": "MIT"
        }"#;
        let m: ModelManifest = serde_json::from_str(json).unwrap();
        assert!(m.display_name.is_none());
        assert!(m.dimension.is_none());
        assert!(m.tier.is_none());
    }

    #[test]
    fn manifest_with_all_new_fields_roundtrips() {
        let m = ModelManifest {
            id: "test".to_owned(),
            version: "test-v1".to_owned(),
            display_name: Some("Test Model".to_owned()),
            description: Some("test manifest".to_owned()),
            repo: "owner/repo".to_owned(),
            revision: "abc123".to_owned(),
            files: vec![ModelFile {
                name: "model.onnx".to_owned(),
                sha256: PLACEHOLDER_VERIFY_AFTER_DOWNLOAD.to_owned(),
                size: 0,
                url: Some("https://example.com/model.onnx".to_owned()),
            }],
            license: "MIT".to_owned(),
            dimension: Some(384),
            tier: Some(ModelTier::Quality),
            download_size_bytes: 0,
        };
        let json = m.to_pretty_json().unwrap();
        let restored = ModelManifest::from_json_str(&json).unwrap();
        assert_eq!(restored.display_name, m.display_name);
        assert_eq!(restored.dimension, m.dimension);
        assert_eq!(restored.tier, m.tier);
        assert_eq!(restored.files[0].url, m.files[0].url);
    }

    #[test]
    fn catalog_schema_version_defaults_on_missing() {
        let json = r#"{"models":[]}"#;
        let catalog = ModelManifestCatalog::from_json_str(json).unwrap();
        assert_eq!(catalog.schema_version, MANIFEST_SCHEMA_VERSION);
    }

    #[test]
    fn catalog_schema_version_preserved_from_json() {
        let json = r#"{"schema_version": 42, "models":[]}"#;
        let catalog = ModelManifestCatalog::from_json_str(json).unwrap();
        assert_eq!(catalog.schema_version, 42);
    }

    #[test]
    fn builtin_catalog_json_roundtrip() {
        let catalog = ModelManifest::builtin_catalog();
        let json = serde_json::to_string_pretty(&catalog).unwrap();
        let restored = ModelManifestCatalog::from_json_str(&json).unwrap();
        assert_eq!(restored.schema_version, catalog.schema_version);
        assert_eq!(restored.models.len(), catalog.models.len());
        for (orig, rest) in catalog.models.iter().zip(restored.models.iter()) {
            assert_eq!(orig.id, rest.id);
            assert_eq!(orig.dimension, rest.dimension);
            assert_eq!(orig.tier, rest.tier);
        }
    }

    #[test]
    fn registry_includes_reranker() {
        let all = ModelManifest::registered();
        let ids: Vec<&str> = all.iter().map(|m| m.id.as_str()).collect();
        assert!(
            ids.contains(&"ms-marco-minilm-l-6-v2"),
            "registry should contain ms-marco reranker, got: {ids:?}"
        );
    }

    // ─── Verification Cache Tests ──────────────────────────────────────

    fn make_test_manifest(file_name: &str, content: &[u8]) -> ModelManifest {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(content);
        let sha = to_hex_lowercase(&hasher.finalize());
        ModelManifest {
            id: "test-model".to_owned(),
            repo: "test/repo".to_owned(),
            revision: "abc".to_owned(),
            files: vec![ModelFile {
                name: file_name.to_owned(),
                sha256: sha,
                size: u64::try_from(content.len()).unwrap(),
                url: None,
            }],
            license: "MIT".to_owned(),
            tier: None,
            dimension: None,
            display_name: None,
            version: String::new(),
            description: None,
            download_size_bytes: u64::try_from(content.len()).unwrap(),
        }
    }

    #[test]
    fn verification_marker_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let content = b"hello model";
        let manifest = make_test_manifest("model.bin", content);
        write_temp_file(&tmp.path().join("model.bin"), content);

        let marker = VerificationMarker::new_for(&manifest, tmp.path());
        let json = serde_json::to_string_pretty(&marker).unwrap();
        let restored: VerificationMarker = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.manifest_id, "test-model");
        assert_eq!(restored.schema_version, MANIFEST_SCHEMA_VERSION);
        let state = restored.file_states.get("model.bin").unwrap();
        assert_eq!(state.size_bytes, u64::try_from(content.len()).unwrap());
        assert!(state.modified_unix_nanos > 0);
    }

    #[test]
    fn verification_cache_hit_when_files_unchanged() {
        let tmp = tempfile::tempdir().unwrap();
        let content = b"model data";
        let manifest = make_test_manifest("model.bin", content);
        write_temp_file(&tmp.path().join("model.bin"), content);

        assert!(!is_verification_cached(&manifest, tmp.path()));
        write_verification_marker(&manifest, tmp.path());
        assert!(is_verification_cached(&manifest, tmp.path()));
    }

    #[test]
    fn verification_cache_miss_when_manifest_id_changes() {
        let tmp = tempfile::tempdir().unwrap();
        let content = b"model data";
        let manifest = make_test_manifest("model.bin", content);
        write_temp_file(&tmp.path().join("model.bin"), content);

        write_verification_marker(&manifest, tmp.path());
        let mut changed = manifest;
        changed.id = "different-model".to_owned();
        assert!(!is_verification_cached(&changed, tmp.path()));
    }

    #[test]
    fn verification_cache_miss_when_file_state_differs() {
        let tmp = tempfile::tempdir().unwrap();
        let content = b"model data";
        let manifest = make_test_manifest("model.bin", content);
        write_temp_file(&tmp.path().join("model.bin"), content);

        // Write marker, then tamper with the recorded file state.
        write_verification_marker(&manifest, tmp.path());
        assert!(is_verification_cached(&manifest, tmp.path()));

        let marker_path = tmp.path().join(VERIFIED_MARKER_FILE);
        let raw = std::fs::read_to_string(&marker_path).unwrap();
        let mut marker: VerificationMarker = serde_json::from_str(&raw).unwrap();
        // Change recorded metadata so it no longer matches the actual file.
        marker.file_states.insert(
            "model.bin".to_owned(),
            FileVerificationState {
                size_bytes: 1,
                modified_unix_nanos: 1,
            },
        );
        let tampered = serde_json::to_string_pretty(&marker).unwrap();
        std::fs::write(&marker_path, tampered).unwrap();

        assert!(!is_verification_cached(&manifest, tmp.path()));
    }

    #[test]
    fn verify_dir_cached_writes_marker_on_success() {
        let tmp = tempfile::tempdir().unwrap();
        let content = b"model data for verify";
        let manifest = make_test_manifest("model.bin", content);
        write_temp_file(&tmp.path().join("model.bin"), content);

        assert!(!tmp.path().join(VERIFIED_MARKER_FILE).exists());
        verify_dir_cached(&manifest, tmp.path()).unwrap();
        assert!(tmp.path().join(VERIFIED_MARKER_FILE).exists());
        assert!(is_verification_cached(&manifest, tmp.path()));
    }

    #[test]
    fn verify_dir_cached_skips_rehash_on_cached_hit() {
        let tmp = tempfile::tempdir().unwrap();
        let content = b"model data cached";
        let manifest = make_test_manifest("model.bin", content);
        write_temp_file(&tmp.path().join("model.bin"), content);

        // First call: full verification + writes marker
        verify_dir_cached(&manifest, tmp.path()).unwrap();

        // Second call: should succeed from cache (no rehash)
        verify_dir_cached(&manifest, tmp.path()).unwrap();
    }

    #[test]
    fn verify_dir_cached_skips_when_no_verified_checksums() {
        let tmp = tempfile::tempdir().unwrap();
        let manifest = ModelManifest {
            id: "test".to_owned(),
            repo: "r".to_owned(),
            revision: "v".to_owned(),
            files: vec![ModelFile {
                name: "f.bin".to_owned(),
                sha256: PLACEHOLDER_VERIFY_AFTER_DOWNLOAD.to_owned(),
                size: 0,
                url: None,
            }],
            license: "MIT".to_owned(),
            tier: None,
            dimension: None,
            display_name: None,
            version: String::new(),
            description: None,
            download_size_bytes: 0,
        };
        // Should not error even though file doesn't exist — skips verification
        verify_dir_cached(&manifest, tmp.path()).unwrap();
    }
}
