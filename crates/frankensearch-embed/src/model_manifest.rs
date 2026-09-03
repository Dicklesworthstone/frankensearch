//! Model manifest definitions and verification helpers.
//!
//! This module is intentionally synchronous and runtime-agnostic:
//! it performs filesystem and hashing work only, and leaves transport/network
//! to higher-level download orchestration.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{OnceLock, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::generation::{
    EMBEDDING_INPUT_CONTRACT_SCHEMA_V1, EMBEDDING_PRODUCER_ATTESTATION_SCHEMA_V1,
    EMBEDDING_SPACE_IDENTITY_SCHEMA_V1, EmbeddingArtifactIdentityV1, EmbeddingIdentityBundleV1,
    EmbeddingInputContractV1, EmbeddingProducerAttestationV1, EmbeddingSpaceIdentityV1,
    EmbeddingSpaceKindV1, GoldenVectorCertificateV1, QuantizationFormat,
    VECTOR_STORAGE_IDENTITY_SCHEMA_V1, VectorStorageIdentityV1,
};

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

/// Schema for the frozen artifact-plus-execution manifest.
pub const MODEL_ARTIFACT_MANIFEST_SCHEMA_V1: u16 = 1;

const MAX_FROZEN_MANIFEST_FIELD_BYTES: usize = 4_096;

/// Ordered synthetic corpus used by every registered embedding producer's
/// bit-exact conformance certificate.
///
/// These bounded strings contain no user data. Ordering and bytes are part of
/// the certificate contract.
pub const MODEL_CONFORMANCE_TEXTS_V1: [&str; 4] = [
    "hello world",
    "semantic search finds related ideas",
    "identifier fsvi_v2",
    "naive cafe Tokyo",
];

/// Pinned adapter-level token budget passed to `FastEmbed` 5.17.2.
#[cfg(feature = "fastembed")]
pub(crate) const FASTEMBED_MAX_LENGTH_V1: usize = 512;
/// Exact truncation and padding policy imposed by the pinned `FastEmbed` adapter.
pub(crate) const FASTEMBED_SEQUENCE_POLICY_V1: &str =
    "max-length=512;longest-first;batch-longest-padding";
/// Exact `FastEmbed` plus adapter-level output normalization pipeline.
pub(crate) const FASTEMBED_OUTPUT_NORMALIZATION_V1: &str =
    "fastembed-l2-eps-1e-12-then-l2-f32-zero-on-degenerate-v1";
/// Exact native `Model2Vec` input preparation and empty/OOV behavior.
pub(crate) const MODEL2VEC_PREPROCESSING_V1: &str =
    "encode-special-tokens=false;discard-oov=true;empty-or-all-oov=zero-vector";
/// Exact sequence behavior of the frozen Potion tokenizer.
pub(crate) const MODEL2VEC_SEQUENCE_POLICY_V1: &str = "tokenizer-configured;no-padding";
/// Exact native `Model2Vec` pooling rule.
pub(crate) const MODEL2VEC_POOLING_V1: &str = "mean-in-vocabulary-token-rows-v1";
/// Exact native `Model2Vec` output normalization rule.
pub(crate) const MODEL2VEC_OUTPUT_NORMALIZATION_V1: &str = "l2-f32-zero-on-degenerate-v1";

/// Semantic role of one artifact in a frozen model bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelArtifactRoleV1 {
    /// Learned weights or static embedding table.
    Weights,
    /// Tokenizer implementation/configuration.
    Tokenizer,
    /// Standalone vocabulary.
    Vocabulary,
    /// Model architecture/configuration.
    ModelConfig,
    /// Special-token mapping.
    SpecialTokens,
    /// Tokenizer runtime configuration.
    TokenizerConfig,
    /// Projection/MRL matrix or descriptor.
    Projection,
}

impl ModelArtifactRoleV1 {
    const fn tag(self) -> &'static str {
        match self {
            Self::Weights => "weights",
            Self::Tokenizer => "tokenizer",
            Self::Vocabulary => "vocabulary",
            Self::ModelConfig => "model_config",
            Self::SpecialTokens => "special_tokens",
            Self::TokenizerConfig => "tokenizer_config",
            Self::Projection => "projection",
        }
    }
}

/// One role-tagged immutable artifact in a frozen model manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelArtifactFileV1 {
    /// Semantic role. Roles are unique within a manifest.
    pub role: ModelArtifactRoleV1,
    /// Safe relative path inside the model directory.
    pub relative_path: String,
    /// Immutable upstream URL. Never emitted in structured runtime logs.
    pub upstream_url: String,
    /// Exact byte size.
    pub size: u64,
    /// Exact lowercase SHA-256.
    pub sha256: String,
}

/// Execution semantics that turn verified artifacts into one mathematical space.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelExecutionContractV1 {
    /// Native/provider backend family.
    pub backend: String,
    /// Immutable implementation revision.
    pub implementation_revision: String,
    /// Inference or wire protocol revision.
    pub protocol_revision: String,
    /// Numeric execution profile.
    pub numeric_profile: String,
    /// Weights/container format.
    pub weights_format: String,
    /// Tokenizer implementation identity.
    pub tokenizer_family: String,
    /// Model-internal preprocessing.
    pub model_preprocessing: String,
    /// Sequence length, truncation, and padding.
    pub sequence_policy: String,
    /// Pooling rule.
    pub pooling: String,
    /// Output normalization.
    pub output_normalization: String,
    /// Query instruction.
    pub query_instruction: String,
    /// Document instruction.
    pub document_instruction: String,
    /// Outer content-selection/canonicalization contract.
    pub input_contract: EmbeddingInputContractV1,
    /// Pinned implementation conformance certificate.
    pub golden_vectors: GoldenVectorCertificateV1,
}

/// Complete frozen model artifact manifest shared by local/native backends.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelArtifactManifestV1 {
    /// Schema version; unknown versions fail closed.
    pub schema_version: u16,
    /// Stable artifact provider.
    pub provider: String,
    /// Semantic model identifier.
    pub logical_model_id: String,
    /// Immutable upstream model revision.
    pub upstream_revision: String,
    /// Immutable upstream repository identity.
    pub upstream_repository: String,
    /// Unique role-tagged artifacts.
    pub artifacts: Vec<ModelArtifactFileV1>,
    /// SPDX license identifier.
    pub license_spdx: String,
    /// Digest of the canonical pinned license metadata assertion.
    pub license_metadata_sha256: String,
    /// Output dimension.
    pub dimension: u32,
    /// Complete execution semantics.
    pub execution: ModelExecutionContractV1,
}

/// Canonical bytes and digest proven to correspond to a validated manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenModelArtifactManifestV1 {
    /// Validated structured manifest.
    pub manifest: ModelArtifactManifestV1,
    /// Domain-separated canonical bytes.
    pub canonical_bytes: Vec<u8>,
    /// SHA-256 of `canonical_bytes`.
    pub fingerprint: String,
}

/// Proof that every file in a frozen manifest was verified in one directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedModelArtifactsV1 {
    frozen: FrozenModelArtifactManifestV1,
}

impl VerifiedModelArtifactsV1 {
    /// Validated frozen manifest.
    #[must_use]
    pub const fn frozen(&self) -> &FrozenModelArtifactManifestV1 {
        &self.frozen
    }

    /// Derive the complete runtime identity from verified artifacts.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` if any cross-contract fingerprint is inconsistent.
    pub fn identity_bundle(
        &self,
        quantization: QuantizationFormat,
        storage_format: &str,
    ) -> SearchResult<EmbeddingIdentityBundleV1> {
        self.frozen
            .manifest
            .identity_bundle(quantization, storage_format)
    }
}

impl ModelArtifactManifestV1 {
    /// Frozen native `Model2Vec` contract for the built-in potion fast tier.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` if the built-in pinned download manifest drifts
    /// from the registered execution contract.
    pub fn potion_128m_native() -> SearchResult<Self> {
        let execution = ModelExecutionContractV1 {
            backend: "model2vec-native".to_owned(),
            implementation_revision: format!(
                "frankensearch-embed-{}+model2vec-native-v1",
                env!("CARGO_PKG_VERSION")
            ),
            protocol_revision: "tokenizers-0.23.1+safetensors-0.7.0-static-table-v1".to_owned(),
            numeric_profile: "f32-row-gather-mean-l2-v1".to_owned(),
            weights_format: "safetensors-f32-matrix-v1".to_owned(),
            tokenizer_family: "huggingface-tokenizers-json-v1".to_owned(),
            model_preprocessing: MODEL2VEC_PREPROCESSING_V1.to_owned(),
            sequence_policy: MODEL2VEC_SEQUENCE_POLICY_V1.to_owned(),
            pooling: MODEL2VEC_POOLING_V1.to_owned(),
            output_normalization: MODEL2VEC_OUTPUT_NORMALIZATION_V1.to_owned(),
            query_instruction: String::new(),
            document_instruction: String::new(),
            input_contract: default_plain_text_input_contract(),
            golden_vectors: GoldenVectorCertificateV1 {
                corpus_sha256: conformance_corpus_fingerprint()?,
                vectors_sha256: "f7dabe71dbb62abf9271f9568d799accb558b982521d3a48337c6a760d7e6c74"
                    .to_owned(),
                vector_count: 4,
                dimension: 256,
            },
        };
        Self::from_download_manifest(
            &ModelManifest::potion_128m(),
            "minishlab-huggingface",
            execution,
        )
    }

    /// Frozen FastEmbed/ONNX contract for the built-in `MiniLM` quality tier.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` if the built-in pinned download manifest drifts
    /// from the registered execution contract.
    pub fn minilm_fastembed() -> SearchResult<Self> {
        let execution = ModelExecutionContractV1 {
            backend: "fastembed-onnx".to_owned(),
            implementation_revision: format!(
                "frankensearch-embed-{}+fastembed-5.17.2",
                env!("CARGO_PKG_VERSION")
            ),
            protocol_revision: "fastembed-5.17.2+ort-2.0.0-rc.12-user-defined-onnx-v1".to_owned(),
            numeric_profile: "onnxruntime-2.0.0-rc.12-cpu-f32-host-default-intra-threads-v1"
                .to_owned(),
            weights_format: "onnx-opset-pinned-v1".to_owned(),
            tokenizer_family: "huggingface-tokenizers-json-v1".to_owned(),
            model_preprocessing: "bert-tokenizer-special-tokens=true;empty-input=zero-vector"
                .to_owned(),
            sequence_policy: FASTEMBED_SEQUENCE_POLICY_V1.to_owned(),
            pooling: "attention-mask-mean-pool-v1".to_owned(),
            output_normalization: FASTEMBED_OUTPUT_NORMALIZATION_V1.to_owned(),
            query_instruction: String::new(),
            document_instruction: String::new(),
            input_contract: default_plain_text_input_contract(),
            golden_vectors: GoldenVectorCertificateV1 {
                corpus_sha256: conformance_corpus_fingerprint()?,
                vectors_sha256: "11620592994a30c5df2ec108983c8a5ce304760f78666c42f56db285a7f3d948"
                    .to_owned(),
                vector_count: 4,
                dimension: 384,
            },
        };
        Self::from_download_manifest(
            &ModelManifest::minilm_v2(),
            "sentence-transformers-huggingface",
            execution,
        )
    }

    /// Frozen pure-Rust frankentorch/safetensors contract for `MiniLM`.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` if the immutable upstream artifact metadata or
    /// native execution contract is incomplete.
    pub fn minilm_native_frankentorch() -> SearchResult<Self> {
        let mut download_manifest = ModelManifest::minilm_v2();
        let weights = download_manifest
            .files
            .iter_mut()
            .find(|file| file.name == "onnx/model.onnx")
            .ok_or_else(|| {
                invalid_manifest_field(
                    "artifacts[].role",
                    "weights",
                    "MiniLM download manifest is missing its registered weights artifact",
                )
            })?;
        "model.safetensors".clone_into(&mut weights.name);
        weights.url = Some(
            "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/model.safetensors"
                .to_owned(),
        );
        "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
            .clone_into(&mut weights.sha256);
        weights.size = 90_868_376;
        download_manifest.download_size_bytes =
            download_manifest.files.iter().map(|file| file.size).sum();
        let execution = ModelExecutionContractV1 {
            backend: "frankentorch-native-minilm".to_owned(),
            implementation_revision:
                "frankensearch-rerank-native-embedder-v1+frankentorch-c305306b251753099620ad5fe02e78c07c167cf6"
                    .to_owned(),
            protocol_revision:
                "tokenizers-0.23.1+frankentorch-bert-encoder-v1".to_owned(),
            numeric_profile: "f32-weights-int8-linear-f32-accumulate-v2".to_owned(),
            weights_format: "safetensors-f32-runtime-int8-linear-v1".to_owned(),
            tokenizer_family: "huggingface-tokenizers-json-v1".to_owned(),
            model_preprocessing: "bert-special-tokens=true;token-type-ids=zero".to_owned(),
            sequence_policy: "max-length=512;longest-first;no-padding".to_owned(),
            pooling: "mean-all-returned-tokens-including-specials-no-padding-v1".to_owned(),
            output_normalization: "l2-f32-if-norm-gt-zero-else-unchanged-v1".to_owned(),
            query_instruction: String::new(),
            document_instruction: String::new(),
            input_contract: default_plain_text_input_contract(),
            golden_vectors: GoldenVectorCertificateV1 {
                corpus_sha256: conformance_corpus_fingerprint()?,
                vectors_sha256: "bed15455ed5910d6ebcf39b28a22e321d83a904f99c53e79724995b662e67c26"
                    .to_owned(),
                vector_count: 4,
                dimension: 384,
            },
        };
        Self::from_download_manifest(
            &download_manifest,
            "sentence-transformers-huggingface",
            execution,
        )
    }

    /// Frozen pure-Rust Frankentorch contract for the opt-in multilingual
    /// `paraphrase-multilingual-MiniLM-L12-v2` embedding space.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` if the immutable upstream artifact metadata or
    /// native execution contract is incomplete.
    pub fn multilingual_minilm_native_frankentorch() -> SearchResult<Self> {
        let execution = ModelExecutionContractV1 {
            backend: "frankentorch-native-multilingual-minilm".to_owned(),
            implementation_revision: "frankensearch-rerank-native-embedder-v2+frankentorch-0.1.0"
                .to_owned(),
            protocol_revision: "tokenizers-0.23.1-xlmr-unigram+frankentorch-bert-dynamic-layers-v2"
                .to_owned(),
            numeric_profile: "f32-weights-int8-linear-f32-accumulate-v2".to_owned(),
            weights_format: "safetensors-f32-runtime-int8-linear-v1".to_owned(),
            tokenizer_family: "huggingface-tokenizers-json-xlmr-unigram-v1".to_owned(),
            model_preprocessing: "xlmr-special-tokens=true;token-type-ids=zero".to_owned(),
            sequence_policy: "max-length=512;longest-first;no-padding".to_owned(),
            pooling: "mean-all-returned-tokens-including-specials-no-padding-v1".to_owned(),
            output_normalization: "l2-f32-if-norm-gt-zero-else-unchanged-v1".to_owned(),
            query_instruction: String::new(),
            document_instruction: String::new(),
            input_contract: default_plain_text_input_contract(),
            golden_vectors: GoldenVectorCertificateV1 {
                corpus_sha256: conformance_corpus_fingerprint()?,
                vectors_sha256: "c7dcf38c4aff04846e5457e658da704dd3d8ac177182ec2104411c064be7ec7d"
                    .to_owned(),
                vector_count: 4,
                dimension: 384,
            },
        };
        Self::from_download_manifest(
            &ModelManifest::multilingual_minilm_l12_v2(),
            "sentence-transformers-huggingface",
            execution,
        )
    }

    /// Frozen `FastEmbed` contract for Snowflake Arctic Embed S.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` if the pinned manifest or contract is incomplete.
    pub fn snowflake_fastembed() -> SearchResult<Self> {
        let execution = fastembed_execution_contract(
            384,
            "snowflake-arctic-embed-s",
            "fb999e00707c8f3709844de704529c29c1f87b540311c05ee211aa93d0dad3a6",
        )?;
        Self::from_download_manifest(
            &ModelManifest::snowflake_arctic_s(),
            "snowflake-huggingface",
            execution,
        )
    }

    /// Frozen `FastEmbed` contract for Nomic Embed Text v1.5.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` if the pinned manifest or contract is incomplete.
    pub fn nomic_fastembed() -> SearchResult<Self> {
        let execution = fastembed_execution_contract(
            768,
            "nomic-embed-text-v1.5",
            "dbb7e33fdb5ccb4864faf9ff425b35a83a2d9dcd4f8d736033d7f819e0c1e851",
        )?;
        Self::from_download_manifest(
            &ModelManifest::nomic_embed(),
            "nomic-ai-huggingface",
            execution,
        )
    }

    /// Convert the existing pinned download manifest plus explicit execution
    /// semantics into the frozen identity contract.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` for unknown/duplicate roles or incomplete metadata.
    pub fn from_download_manifest(
        manifest: &ModelManifest,
        provider: &str,
        execution: ModelExecutionContractV1,
    ) -> SearchResult<Self> {
        manifest.validate()?;
        if !manifest.is_production_ready() {
            return Err(invalid_manifest_field(
                "production_ready",
                &manifest.id,
                "frozen manifests require pinned revision, size, and SHA-256 for every artifact",
            ));
        }
        let dimension = manifest.dimension.ok_or_else(|| {
            invalid_manifest_field(
                "dimension",
                &manifest.id,
                "embedding manifests require a fixed output dimension",
            )
        })?;
        let artifacts = manifest
            .files
            .iter()
            .map(|file| {
                Ok(ModelArtifactFileV1 {
                    role: artifact_role_for_path(&file.name)?,
                    relative_path: file.name.clone(),
                    upstream_url: manifest.download_url(file),
                    size: file.size,
                    sha256: file.sha256.clone(),
                })
            })
            .collect::<SearchResult<Vec<_>>>()?;
        let license_metadata_sha256 = license_metadata_fingerprint(
            &manifest.license,
            provider,
            &manifest.repo,
            &manifest.revision,
        );
        let frozen = Self {
            schema_version: MODEL_ARTIFACT_MANIFEST_SCHEMA_V1,
            provider: provider.to_owned(),
            logical_model_id: manifest.id.clone(),
            upstream_revision: manifest.revision.clone(),
            upstream_repository: manifest.repo.clone(),
            artifacts,
            license_spdx: manifest.license.clone(),
            license_metadata_sha256,
            dimension,
            execution,
        };
        frozen.validate()?;
        Ok(frozen)
    }

    /// Validate schema, role uniqueness, hashes, sizes, license, and execution semantics.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` for any incomplete or ambiguous contract.
    pub fn validate(&self) -> SearchResult<()> {
        if self.schema_version != MODEL_ARTIFACT_MANIFEST_SCHEMA_V1 {
            return Err(invalid_manifest_field(
                "artifact_manifest.schema_version",
                &self.schema_version.to_string(),
                "unsupported frozen artifact manifest schema",
            ));
        }
        for (field, value) in [
            ("provider", self.provider.as_str()),
            ("logical_model_id", self.logical_model_id.as_str()),
            ("upstream_repository", self.upstream_repository.as_str()),
            ("license_spdx", self.license_spdx.as_str()),
            ("execution.backend", self.execution.backend.as_str()),
            (
                "execution.implementation_revision",
                self.execution.implementation_revision.as_str(),
            ),
            (
                "execution.protocol_revision",
                self.execution.protocol_revision.as_str(),
            ),
            (
                "execution.numeric_profile",
                self.execution.numeric_profile.as_str(),
            ),
            (
                "execution.weights_format",
                self.execution.weights_format.as_str(),
            ),
            (
                "execution.tokenizer_family",
                self.execution.tokenizer_family.as_str(),
            ),
            (
                "execution.model_preprocessing",
                self.execution.model_preprocessing.as_str(),
            ),
            (
                "execution.sequence_policy",
                self.execution.sequence_policy.as_str(),
            ),
            ("execution.pooling", self.execution.pooling.as_str()),
            (
                "execution.output_normalization",
                self.execution.output_normalization.as_str(),
            ),
        ] {
            validate_frozen_manifest_text(field, value, false)?;
        }
        for (field, value) in [
            (
                "execution.query_instruction",
                self.execution.query_instruction.as_str(),
            ),
            (
                "execution.document_instruction",
                self.execution.document_instruction.as_str(),
            ),
        ] {
            validate_frozen_manifest_text(field, value, true)?;
        }
        if !matches!(self.upstream_revision.len(), 40 | 64)
            || !self
                .upstream_revision
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(invalid_manifest_field(
                "upstream_revision",
                &self.upstream_revision,
                "must be a complete lowercase 40- or 64-character immutable commit digest",
            ));
        }
        if self.dimension == 0 {
            return Err(invalid_manifest_field(
                "dimension",
                "0",
                "must be greater than zero",
            ));
        }
        validate_frozen_sha256("license_metadata_sha256", &self.license_metadata_sha256)?;
        let expected_license_metadata = license_metadata_fingerprint(
            &self.license_spdx,
            &self.provider,
            &self.upstream_repository,
            &self.upstream_revision,
        );
        if self.license_metadata_sha256 != expected_license_metadata {
            return Err(invalid_manifest_field(
                "license_metadata_sha256",
                &self.license_metadata_sha256,
                "must bind the canonical SPDX, provider, repository, and immutable revision assertion",
            ));
        }
        self.execution.input_contract.validate()?;
        self.execution.golden_vectors.validate().map_err(|error| {
            invalid_manifest_field(
                "execution.golden_vectors",
                &self.logical_model_id,
                &error.to_string(),
            )
        })?;
        if self.execution.golden_vectors.dimension != self.dimension {
            return Err(invalid_manifest_field(
                "execution.golden_vectors.dimension",
                &self.execution.golden_vectors.dimension.to_string(),
                "must equal manifest dimension",
            ));
        }

        let mut roles = std::collections::BTreeSet::new();
        let mut paths = std::collections::BTreeSet::new();
        for artifact in &self.artifacts {
            validate_frozen_manifest_text(
                "artifacts[].relative_path",
                &artifact.relative_path,
                false,
            )?;
            validate_model_file_name(&artifact.relative_path)?;
            validate_frozen_sha256("artifacts[].sha256", &artifact.sha256)?;
            if artifact.size == 0 {
                return Err(invalid_manifest_field(
                    "artifacts[].size",
                    "0",
                    "must be greater than zero",
                ));
            }
            validate_frozen_manifest_text(
                "artifacts[].upstream_url",
                &artifact.upstream_url,
                false,
            )?;
            if !artifact.upstream_url.starts_with("https://")
                || artifact
                    .upstream_url
                    .bytes()
                    .any(|byte| byte.is_ascii_whitespace())
                || artifact.upstream_url.contains('@')
                || artifact.upstream_url.contains('?')
                || artifact.upstream_url.contains('#')
            {
                return Err(invalid_manifest_field(
                    "artifacts[].upstream_url",
                    "redacted",
                    "must be credential-free HTTPS without userinfo, query, or fragment",
                ));
            }
            if !roles.insert(artifact.role) {
                return Err(invalid_manifest_field(
                    "artifacts[].role",
                    artifact.role.tag(),
                    "duplicate semantic artifact role",
                ));
            }
            if !paths.insert(artifact.relative_path.as_str()) {
                return Err(invalid_manifest_field(
                    "artifacts[].relative_path",
                    &artifact.relative_path,
                    "duplicate artifact path",
                ));
            }
        }
        if !roles.contains(&ModelArtifactRoleV1::Weights)
            || !roles.contains(&ModelArtifactRoleV1::Tokenizer)
        {
            return Err(invalid_manifest_field(
                "artifacts",
                &self.logical_model_id,
                "embedding manifests require unique weights and tokenizer roles",
            ));
        }
        Ok(())
    }

    /// Domain-separated canonical bytes. Artifact order is canonicalized by the
    /// frozen serialized role tag rather than enum declaration order.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        append_frozen_bytes(&mut bytes, b"frankensearch.model-artifact-manifest.v1");
        bytes.extend_from_slice(&self.schema_version.to_be_bytes());
        for value in [
            &self.provider,
            &self.logical_model_id,
            &self.upstream_revision,
            &self.upstream_repository,
            &self.license_spdx,
            &self.license_metadata_sha256,
        ] {
            append_frozen_bytes(&mut bytes, value.as_bytes());
        }
        bytes.extend_from_slice(&self.dimension.to_be_bytes());
        let mut artifacts = self.artifacts.iter().collect::<Vec<_>>();
        artifacts.sort_by_key(|artifact| artifact.role.tag());
        append_frozen_len(&mut bytes, artifacts.len());
        for artifact in artifacts {
            append_frozen_bytes(&mut bytes, artifact.role.tag().as_bytes());
            append_frozen_bytes(&mut bytes, artifact.relative_path.as_bytes());
            append_frozen_bytes(&mut bytes, artifact.upstream_url.as_bytes());
            bytes.extend_from_slice(&artifact.size.to_be_bytes());
            append_frozen_bytes(&mut bytes, artifact.sha256.as_bytes());
        }
        for value in [
            &self.execution.backend,
            &self.execution.implementation_revision,
            &self.execution.protocol_revision,
            &self.execution.numeric_profile,
            &self.execution.weights_format,
            &self.execution.tokenizer_family,
            &self.execution.model_preprocessing,
            &self.execution.sequence_policy,
            &self.execution.query_instruction,
            &self.execution.document_instruction,
            &self.execution.pooling,
            &self.execution.output_normalization,
        ] {
            append_frozen_bytes(&mut bytes, value.as_bytes());
        }
        append_frozen_bytes(&mut bytes, &self.execution.input_contract.canonical_bytes());
        append_frozen_bytes(
            &mut bytes,
            self.execution.golden_vectors.corpus_sha256.as_bytes(),
        );
        append_frozen_bytes(
            &mut bytes,
            self.execution.golden_vectors.vectors_sha256.as_bytes(),
        );
        bytes.extend_from_slice(&self.execution.golden_vectors.vector_count.to_be_bytes());
        bytes.extend_from_slice(&self.execution.golden_vectors.dimension.to_be_bytes());
        bytes
    }

    /// Fingerprint only the fields that define the mathematical embedding
    /// space, excluding producer/backend and distribution metadata.
    ///
    /// This deliberately omits provider, repository URL, license, backend,
    /// implementation/protocol revision, numeric execution profile, and golden
    /// certificate. Those remain covered by the full frozen manifest and the
    /// separately bound producer attestation. Consequently, two conformant
    /// implementations over the same artifacts and model semantics may share a
    /// space identity without pretending to be the same producer.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` if the complete manifest is invalid.
    pub fn space_contract_fingerprint(&self) -> SearchResult<String> {
        self.validate()?;
        Ok(sha256_hex_bytes(&self.space_contract_canonical_bytes()))
    }

    fn space_contract_canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        append_frozen_bytes(
            &mut bytes,
            b"frankensearch.model-artifact-space-contract.v1",
        );
        bytes.extend_from_slice(&self.schema_version.to_be_bytes());
        append_frozen_bytes(&mut bytes, self.logical_model_id.as_bytes());
        append_frozen_bytes(&mut bytes, self.upstream_revision.as_bytes());
        bytes.extend_from_slice(&self.dimension.to_be_bytes());

        let mut artifacts = self.artifacts.iter().collect::<Vec<_>>();
        artifacts.sort_by_key(|artifact| artifact.role.tag());
        append_frozen_len(&mut bytes, artifacts.len());
        for artifact in artifacts {
            append_frozen_bytes(&mut bytes, artifact.role.tag().as_bytes());
            bytes.extend_from_slice(&artifact.size.to_be_bytes());
            append_frozen_bytes(&mut bytes, artifact.sha256.as_bytes());
        }

        for value in [
            &self.execution.weights_format,
            &self.execution.tokenizer_family,
            &self.execution.model_preprocessing,
            &self.execution.sequence_policy,
            &self.execution.query_instruction,
            &self.execution.document_instruction,
            &self.execution.pooling,
            &self.execution.output_normalization,
        ] {
            append_frozen_bytes(&mut bytes, value.as_bytes());
        }
        append_frozen_bytes(&mut bytes, &self.execution.input_contract.canonical_bytes());
        bytes
    }

    /// Validate and freeze exact canonical bytes plus their fingerprint.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` if the manifest is incomplete.
    pub fn freeze(&self) -> SearchResult<FrozenModelArtifactManifestV1> {
        self.validate()?;
        let canonical_bytes = self.canonical_bytes();
        Ok(FrozenModelArtifactManifestV1 {
            manifest: self.clone(),
            fingerprint: sha256_hex_bytes(&canonical_bytes),
            canonical_bytes,
        })
    }

    /// Verify all registered files and reject partial layouts or unregistered
    /// files that collide with a registered semantic artifact role. Unrelated
    /// cache metadata such as notices may coexist with the artifact set.
    ///
    /// # Errors
    ///
    /// Returns a typed search error on missing, unexpected, size-drifted, or
    /// hash-drifted artifacts.
    pub fn verify_dir(&self, model_dir: &Path) -> SearchResult<VerifiedModelArtifactsV1> {
        let frozen = self.freeze()?;
        let mut failures = Vec::new();
        for artifact in &self.artifacts {
            let path = resolve_model_file_path(model_dir, &artifact.relative_path)?;
            match verify_file_sha256(&path, &artifact.sha256, artifact.size) {
                Ok(()) => {}
                Err(SearchError::ModelNotFound { .. }) => {
                    failures.push((artifact, "missing"));
                }
                Err(SearchError::HashMismatch { .. }) => {
                    failures.push((artifact, "sha256-or-size-mismatch"));
                }
                Err(SearchError::ModelLoadFailed { .. } | SearchError::Io(_)) => {
                    failures.push((artifact, "unreadable-or-not-regular-file"));
                }
                Err(error) => return Err(error),
            }
        }
        if !failures.is_empty() {
            let mut detail = String::new();
            for (artifact, reason) in &failures {
                if !detail.is_empty() {
                    detail.push_str("; ");
                }
                let _ = write!(
                    detail,
                    "{}:{}:{reason}:fetch={}",
                    artifact.role.tag(),
                    artifact.relative_path,
                    artifact.upstream_url
                );
            }
            // ubs:ignore — reason is a public diagnostic tag, not a secret.
            if failures.iter().all(|(_, reason)| *reason == "missing") {
                return Err(SearchError::ModelNotFound {
                    name: format!(
                        "{} (incomplete frozen artifact set: {detail})",
                        self.logical_model_id
                    ),
                });
            }
            return Err(SearchError::ModelLoadFailed {
                path: PathBuf::from("<redacted-model-dir>"),
                source: format!("frozen artifact set rejected: {detail}").into(),
            });
        }
        verify_no_extra_artifacts(model_dir, &self.artifacts)?;
        Ok(VerifiedModelArtifactsV1 { frozen })
    }

    /// Verify a model directory against this frozen manifest, reusing the
    /// download manifest's verification receipt when it is still valid.
    ///
    /// A native manifest is derived from its download manifest by
    /// [`Self::from_download_manifest`], so both describe one artifact set.
    /// When `download_manifest` carries a valid `.verified` receipt for
    /// `model_dir` (minted only by [`verify_dir_and_record`] after a full
    /// SHA-256 pass, and invalidated by any size, mtime, or identity change)
    /// and every artifact here matches that manifest's file entry on relative
    /// path, size, and SHA-256, the full hash pass is skipped. This is what
    /// keeps a 512 MB model from being re-hashed on every process start. Any
    /// mismatch, missing receipt, or stale receipt falls back to
    /// [`Self::verify_dir`]; nothing is minted here.
    ///
    /// # Errors
    ///
    /// Same as [`Self::verify_dir`].
    pub fn verify_dir_cached(
        &self,
        download_manifest: &ModelManifest,
        model_dir: &Path,
    ) -> SearchResult<VerifiedModelArtifactsV1> {
        if self.artifacts_match_download_manifest(download_manifest)
            && is_verification_cached(download_manifest, model_dir)
        {
            let frozen = self.freeze()?;
            verify_no_extra_artifacts(model_dir, &self.artifacts)?;
            return Ok(VerifiedModelArtifactsV1 { frozen });
        }
        self.verify_dir(model_dir)
    }

    /// True when every artifact here is byte-for-byte the same file entry
    /// (relative path, size, SHA-256) as in `download_manifest`, and the two
    /// describe the same logical model with the same number of files.
    fn artifacts_match_download_manifest(&self, download_manifest: &ModelManifest) -> bool {
        if self.logical_model_id != download_manifest.id
            || self.artifacts.len() != download_manifest.files.len()
        {
            return false;
        }
        self.artifacts.iter().all(|artifact| {
            download_manifest.files.iter().any(|file| {
                // A download-manifest file is keyed by `name`, a frozen
                // artifact by `relative_path`; clippy's operator-grouping
                // heuristic misreads the pair, so bind it first.
                let same_path = file.name == artifact.relative_path;
                same_path
                    && file.size == artifact.size
                    && file.sha256.eq_ignore_ascii_case(&artifact.sha256)
            })
        })
    }

    /// Promote a fully verified frozen artifact set into its final directory.
    ///
    /// Every registered file is synced before the sibling-directory rename.
    /// An existing generation is moved to a unique backup and is never deleted.
    ///
    /// # Errors
    ///
    /// Returns a typed search error when verification, syncing, or publication
    /// fails. If the final rename fails after moving the previous generation,
    /// the implementation attempts to restore that generation immediately.
    pub fn promote_verified_installation(
        &self,
        staged_dir: &Path,
        destination_dir: &Path,
    ) -> SearchResult<Option<PathBuf>> {
        self.verify_dir(staged_dir)?;
        sync_registered_artifacts(
            staged_dir,
            self.artifacts
                .iter()
                .map(|artifact| artifact.relative_path.as_str()),
        )?;
        promote_atomically(staged_dir, destination_dir)
    }

    /// Derive the identity declared by this frozen manifest without claiming
    /// that a local artifact directory has already been verified.
    ///
    /// This is intended for lazy embedders that must expose their immutable
    /// expected identity before first use. They still must verify the artifact
    /// directory and compare the loaded identity byte-for-byte before returning
    /// any vector.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` if the manifest or cross-contract bindings are
    /// incomplete.
    pub fn declared_identity_bundle(
        &self,
        quantization: QuantizationFormat,
        storage_format: &str,
    ) -> SearchResult<EmbeddingIdentityBundleV1> {
        self.identity_bundle(quantization, storage_format)
    }

    fn identity_bundle(
        &self,
        quantization: QuantizationFormat,
        storage_format: &str,
    ) -> SearchResult<EmbeddingIdentityBundleV1> {
        let space_contract_fingerprint = self.space_contract_fingerprint()?;
        let provenance_manifest_fingerprint = self.freeze()?.fingerprint;
        let input = self.execution.input_contract.clone();
        let tokenizer_fingerprint = role_fingerprint(self, ModelArtifactRoleV1::Tokenizer)?;
        let vocabulary_fingerprint = role_fingerprint_optional(
            self,
            ModelArtifactRoleV1::Vocabulary,
            &tokenizer_fingerprint,
        );
        let model_config_fingerprint = role_fingerprint_optional(
            self,
            ModelArtifactRoleV1::ModelConfig,
            &space_contract_fingerprint,
        );
        let space = EmbeddingSpaceIdentityV1 {
            schema_version: EMBEDDING_SPACE_IDENTITY_SCHEMA_V1,
            logical_model_id: self.logical_model_id.clone(),
            immutable_revision: self.upstream_revision.clone(),
            kind: EmbeddingSpaceKindV1::Semantic,
            artifact_manifest_fingerprint: space_contract_fingerprint,
            artifacts: self
                .artifacts
                .iter()
                .map(|artifact| EmbeddingArtifactIdentityV1 {
                    role: artifact.role.tag().to_owned(),
                    sha256: artifact.sha256.clone(),
                    size: artifact.size,
                })
                .collect(),
            tokenizer_fingerprint,
            vocabulary_fingerprint,
            model_config_fingerprint,
            model_preprocessing: self.execution.model_preprocessing.clone(),
            sequence_policy: self.execution.sequence_policy.clone(),
            query_instruction: self.execution.query_instruction.clone(),
            document_instruction: self.execution.document_instruction.clone(),
            pooling: self.execution.pooling.clone(),
            output_normalization: self.execution.output_normalization.clone(),
            dimension: self.dimension,
            input_contract_fingerprint: input.fingerprint(),
            hash_control: None,
            projection: None,
        };
        let producer = EmbeddingProducerAttestationV1 {
            schema_version: EMBEDDING_PRODUCER_ATTESTATION_SCHEMA_V1,
            backend: self.execution.backend.clone(),
            implementation_revision: self.execution.implementation_revision.clone(),
            protocol_revision: self.execution.protocol_revision.clone(),
            numeric_profile: self.execution.numeric_profile.clone(),
            provenance_manifest_fingerprint,
            space_fingerprint: space.fingerprint(),
            golden_vectors: self.execution.golden_vectors.clone(),
        };
        let bundle = EmbeddingIdentityBundleV1 {
            space,
            producer,
            input,
            storage: VectorStorageIdentityV1 {
                schema_version: VECTOR_STORAGE_IDENTITY_SCHEMA_V1,
                format: storage_format.to_owned(),
                quantization,
                endianness: storage_endianness(storage_format).to_owned(),
                vector_normalization: self.execution.output_normalization.clone(),
                dimension: self.dimension,
            },
        };
        bundle.validate()?;
        Ok(bundle)
    }
}

impl FrozenModelArtifactManifestV1 {
    /// Recompute canonical bytes and fingerprint, rejecting any disagreement.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` if structured fields, bytes, or digest disagree.
    pub fn validate(&self) -> SearchResult<()> {
        self.manifest.validate()?;
        validate_frozen_sha256("fingerprint", &self.fingerprint)?;
        let canonical = self.manifest.canonical_bytes();
        if canonical != self.canonical_bytes {
            return Err(invalid_manifest_field(
                "canonical_bytes",
                &self.manifest.logical_model_id,
                "stored canonical bytes disagree with structured manifest",
            ));
        }
        let fingerprint = sha256_hex_bytes(&canonical);
        if fingerprint != self.fingerprint {
            return Err(invalid_manifest_field(
                "fingerprint",
                &self.fingerprint,
                "stored fingerprint disagrees with canonical bytes",
            ));
        }
        Ok(())
    }
}

fn default_plain_text_input_contract() -> EmbeddingInputContractV1 {
    EmbeddingInputContractV1 {
        schema_version: EMBEDDING_INPUT_CONTRACT_SCHEMA_V1,
        canonicalization: "caller-utf8-as-is-v1".to_owned(),
        content_selection: "single-caller-supplied-text-v1".to_owned(),
        chunking: "none-at-embedder-boundary-v1".to_owned(),
        query_instruction: String::new(),
        document_instruction: String::new(),
        doc_id_semantics: "vector-independent-of-document-id-v1".to_owned(),
    }
}

fn storage_endianness(storage_format: &str) -> &'static str {
    if storage_format.starts_with("in-memory-") {
        "native-f32-values"
    } else {
        "little-endian"
    }
}

fn fastembed_execution_contract(
    dimension: u32,
    model_id: &str,
    vectors_sha256: &str,
) -> SearchResult<ModelExecutionContractV1> {
    Ok(ModelExecutionContractV1 {
        backend: "fastembed-onnx".to_owned(),
        implementation_revision: format!(
            "frankensearch-embed-{}+fastembed-5.17.2:{model_id}",
            env!("CARGO_PKG_VERSION")
        ),
        protocol_revision: "fastembed-5.17.2+ort-2.0.0-rc.12-user-defined-onnx-v1".to_owned(),
        numeric_profile: "onnxruntime-2.0.0-rc.12-cpu-f32-host-default-intra-threads-v1".to_owned(),
        weights_format: "onnx-opset-pinned-v1".to_owned(),
        tokenizer_family: "huggingface-tokenizers-json-v1".to_owned(),
        model_preprocessing: "model-tokenizer-special-tokens=true;empty-input=zero-vector"
            .to_owned(),
        sequence_policy: FASTEMBED_SEQUENCE_POLICY_V1.to_owned(),
        pooling: "attention-mask-mean-pool-v1".to_owned(),
        output_normalization: FASTEMBED_OUTPUT_NORMALIZATION_V1.to_owned(),
        query_instruction: String::new(),
        document_instruction: String::new(),
        input_contract: default_plain_text_input_contract(),
        golden_vectors: GoldenVectorCertificateV1 {
            corpus_sha256: conformance_corpus_fingerprint()?,
            vectors_sha256: vectors_sha256.to_owned(),
            vector_count: 4,
            dimension,
        },
    })
}

fn conformance_corpus_fingerprint() -> SearchResult<String> {
    GoldenVectorCertificateV1::corpus_fingerprint(&MODEL_CONFORMANCE_TEXTS_V1)
}

fn artifact_role_for_path(path: &str) -> SearchResult<ModelArtifactRoleV1> {
    let file_name = path.rsplit('/').next().unwrap_or(path);
    match file_name {
        "model.safetensors" | "model_f32.safetensors" | "model.onnx" => {
            Ok(ModelArtifactRoleV1::Weights)
        }
        "tokenizer.json" => Ok(ModelArtifactRoleV1::Tokenizer),
        "vocab.json" | "vocab.txt" => Ok(ModelArtifactRoleV1::Vocabulary),
        "config.json" => Ok(ModelArtifactRoleV1::ModelConfig),
        "special_tokens_map.json" => Ok(ModelArtifactRoleV1::SpecialTokens),
        "tokenizer_config.json" => Ok(ModelArtifactRoleV1::TokenizerConfig),
        "projection.safetensors" | "projection.bin" => Ok(ModelArtifactRoleV1::Projection),
        _ => Err(invalid_manifest_field(
            "artifacts[].role",
            path,
            "artifact path has no registered semantic role",
        )),
    }
}

fn validate_frozen_sha256(field: &str, value: &str) -> SearchResult<()> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Ok(());
    }
    Err(invalid_manifest_field(
        field,
        "redacted-invalid-sha256",
        "must be lowercase 64-character SHA-256",
    ))
}

fn validate_frozen_manifest_text(field: &str, value: &str, allow_empty: bool) -> SearchResult<()> {
    if value.len() > MAX_FROZEN_MANIFEST_FIELD_BYTES {
        return Err(invalid_manifest_field(
            field,
            "redacted-oversized",
            "field exceeds the bounded frozen-manifest size",
        ));
    }
    if value.chars().any(char::is_control) {
        return Err(invalid_manifest_field(
            field,
            "redacted-control-character",
            "field must not contain control characters",
        ));
    }
    if !allow_empty && value.trim().is_empty() {
        return Err(invalid_manifest_field(field, value, "must not be empty"));
    }
    Ok(())
}

fn append_frozen_len(bytes: &mut Vec<u8>, value: usize) {
    let value = u64::try_from(value).unwrap_or(u64::MAX);
    bytes.extend_from_slice(&value.to_be_bytes());
}

fn append_frozen_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    append_frozen_len(bytes, value.len());
    bytes.extend_from_slice(value);
}

fn sha256_hex_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    to_hex_lowercase(&digest)
}

fn license_metadata_fingerprint(
    license_spdx: &str,
    provider: &str,
    repository: &str,
    revision: &str,
) -> String {
    let mut bytes = Vec::new();
    append_frozen_bytes(&mut bytes, b"frankensearch.model-license-metadata.v1");
    for value in [license_spdx, provider, repository, revision] {
        append_frozen_bytes(&mut bytes, value.as_bytes());
    }
    sha256_hex_bytes(&bytes)
}

fn role_fingerprint(
    manifest: &ModelArtifactManifestV1,
    role: ModelArtifactRoleV1,
) -> SearchResult<String> {
    manifest
        .artifacts
        .iter()
        .find(|artifact| artifact.role == role)
        .map(|artifact| artifact.sha256.clone())
        .ok_or_else(|| {
            invalid_manifest_field("artifacts[].role", role.tag(), "required role is absent")
        })
}

fn role_fingerprint_optional(
    manifest: &ModelArtifactManifestV1,
    role: ModelArtifactRoleV1,
    fallback: &str,
) -> String {
    manifest
        .artifacts
        .iter()
        .find(|artifact| artifact.role == role)
        .map_or_else(|| fallback.to_owned(), |artifact| artifact.sha256.clone())
}

fn verify_no_extra_artifacts(
    model_dir: &Path,
    expected: &[ModelArtifactFileV1],
) -> SearchResult<()> {
    let expected_paths = expected
        .iter()
        .map(|artifact| artifact.relative_path.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    let mut pending = vec![model_dir.to_path_buf()];
    while let Some(directory) = pending.pop() {
        let entries = fs::read_dir(&directory).map_err(SearchError::Io)?;
        for entry in entries {
            let entry = entry.map_err(SearchError::Io)?;
            let path = entry.path();
            let file_type = entry.file_type().map_err(SearchError::Io)?;
            let relative = path.strip_prefix(model_dir).map_err(|_| {
                invalid_manifest_field(
                    "model_dir",
                    "redacted",
                    "artifact escaped the verified model directory",
                )
            })?;
            let relative = relative.to_string_lossy().replace('\\', "/");
            if relative == ".verified" || expected_paths.contains(relative.as_str()) {
                continue;
            }
            if let Ok(role) = artifact_role_for_path(&relative) {
                return Err(invalid_manifest_field(
                    "artifacts[].role",
                    role.tag(),
                    "unregistered file collides with an expected semantic role",
                ));
            }
            if file_type.is_dir() {
                pending.push(path);
            }
        }
    }
    Ok(())
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

    /// Get the local filename (basename) for saving.
    ///
    /// For paths like `"onnx/model.onnx"`, returns `"model.onnx"`.
    /// This handles `HuggingFace` repos that restructure files into subdirectories.
    #[must_use]
    pub fn local_name(&self) -> &str {
        self.name.rsplit('/').next().unwrap_or(&self.name)
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

/// Canonical identity used to bind a verification receipt to one production manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
struct FrozenVerificationManifest {
    fingerprint: String,
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

    /// Opt-in manifest for multilingual `MiniLM` L12 sentence embeddings.
    ///
    /// This model is deliberately absent from [`Self::builtin_catalog`]: its
    /// 384-dimensional output is a distinct vector space from `all-MiniLM-L6-v2`,
    /// and its larger artifact must never be acquired or selected implicitly.
    #[must_use]
    pub fn multilingual_minilm_l12_v2() -> Self {
        const REVISION: &str = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42";
        const REPO: &str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2";
        Self {
            id: "paraphrase-multilingual-minilm-l12-v2".to_owned(),
            version: "v1".to_owned(),
            display_name: Some("Paraphrase Multilingual MiniLM L12 v2 (opt-in)".to_owned()),
            description: Some(
                "Opt-in 384-dimensional multilingual sentence embedder for CJK and mixed-language retrieval"
                    .to_owned(),
            ),
            repo: REPO.to_owned(),
            revision: REVISION.to_owned(),
            files: vec![
                ModelFile {
                    name: "model.safetensors".to_owned(),
                    sha256: "eaa086f0ffee582aeb45b36e34cdd1fe2d6de2bef61f8a559a1bbc9bd955917b"
                        .to_owned(),
                    size: 470_641_600,
                    url: None,
                },
                ModelFile {
                    name: "tokenizer.json".to_owned(),
                    sha256: "2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8"
                        .to_owned(),
                    size: 9_081_518,
                    url: None,
                },
                ModelFile {
                    name: "config.json".to_owned(),
                    sha256: "6300193cb75e01cf80c96decef7187dfb33094d97cc1490b7ead6ff134476e4e"
                        .to_owned(),
                    size: 645,
                    url: None,
                },
                ModelFile {
                    name: "special_tokens_map.json".to_owned(),
                    sha256: "378eb3bf733eb16e65792d7e3fda5b8a4631387ca04d2015199c4d4f22ae554d"
                        .to_owned(),
                    size: 239,
                    url: None,
                },
                ModelFile {
                    name: "tokenizer_config.json".to_owned(),
                    sha256: "5036ea374ffedd706e3bef33e2e0d6953cb868ef8a490e76e32ba0faa37a6b9b"
                        .to_owned(),
                    size: 526,
                    url: None,
                },
            ],
            license: "Apache-2.0".to_owned(),
            dimension: Some(384),
            tier: Some(ModelTier::Quality),
            download_size_bytes: 479_724_528,
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
            license: "MIT".to_owned(),
            dimension: Some(256),
            tier: Some(ModelTier::Fast),
            download_size_bytes: 530_977_691,
        }
    }

    /// Built-in manifest for flashrank-nano (cross-encoder reranker).
    #[must_use]
    pub fn flashrank_nano() -> Self {
        const REVISION: &str = PLACEHOLDER_PINNED_REVISION;
        const REPO: &str = "prithivida/flashrank-nano";
        Self {
            id: "flashrank-nano".to_owned(),
            version: "v1".to_owned(),
            display_name: Some("FlashRank Nano (Reranker)".to_owned()),
            description: Some("FlashRank compact ONNX cross-encoder reranker model".to_owned()),
            repo: REPO.to_owned(),
            revision: REVISION.to_owned(),
            files: vec![
                ModelFile {
                    name: "onnx/model.onnx".to_owned(),
                    sha256: PLACEHOLDER_VERIFY_AFTER_DOWNLOAD.to_owned(),
                    size: 0,
                    url: None,
                },
                ModelFile {
                    name: "tokenizer.json".to_owned(),
                    sha256: PLACEHOLDER_VERIFY_AFTER_DOWNLOAD.to_owned(),
                    size: 0,
                    url: None,
                },
            ],
            license: "Apache-2.0".to_owned(),
            dimension: None,
            tier: Some(ModelTier::Reranker),
            download_size_bytes: 0,
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
                // The f32 safetensors export feeds the pure-Rust frankentorch
                // cross-encoder (`frankensearch-rerank` `native`), which is
                // the reranker fsfs ships; the ONNX export below stays for the
                // optional `fastembed-reranker` backend.
                ModelFile {
                    name: "model.safetensors".to_owned(),
                    sha256: "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"
                        .to_owned(),
                    size: 90_870_598,
                    url: Some(
                        "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/c5ee24cb16019beea0893ab7796b1df96625c6b8/model.safetensors"
                            .to_owned(),
                    ),
                },
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
            download_size_bytes: 182_595_480,
        }
    }

    // ==================== Bake-off Eligible Models ====================

    /// Snowflake Arctic Embed S manifest.
    ///
    /// Dimension: 384. Small, fast model with the same width as `MiniLM` but a
    /// distinct mathematical vector space.
    /// Verified checksums from `HuggingFace`.
    #[must_use]
    pub fn snowflake_arctic_s() -> Self {
        const REVISION: &str = "e596f507467533e48a2e17c007f0e1dacc837b33";
        const REPO: &str = "Snowflake/snowflake-arctic-embed-s";
        Self {
            id: "snowflake-arctic-embed-s".to_owned(),
            version: "v1".to_owned(),
            display_name: Some("Snowflake Arctic Embed S".to_owned()),
            description: Some(
                "Small, fast embedding model with MiniLM-compatible 384 dimensions".to_owned(),
            ),
            repo: REPO.to_owned(),
            revision: REVISION.to_owned(),
            files: vec![
                ModelFile {
                    name: "onnx/model.onnx".to_owned(),
                    sha256: "579c1f1778a0993eb0d2a1403340ffb491c769247fb46acc4f5cf8ac5b89c1e1"
                        .to_owned(),
                    size: 133_093_492,
                    url: None,
                },
                ModelFile {
                    name: "tokenizer.json".to_owned(),
                    sha256: "91f1def9b9391fdabe028cd3f3fcc4efd34e5d1f08c3bf2de513ebb5911a1854"
                        .to_owned(),
                    size: 711_649,
                    url: None,
                },
                ModelFile {
                    name: "config.json".to_owned(),
                    sha256: "4e519aa92ec40943356032afe458c8829d70c5766b109e4a57490b82f72dcfb7"
                        .to_owned(),
                    size: 703,
                    url: None,
                },
                ModelFile {
                    name: "special_tokens_map.json".to_owned(),
                    sha256: "5d5b662e421ea9fac075174bb0688ee0d9431699900b90662acd44b2a350503a"
                        .to_owned(),
                    size: 695,
                    url: None,
                },
                ModelFile {
                    name: "tokenizer_config.json".to_owned(),
                    sha256: "9ca59277519f6e3692c8685e26b94d4afca2d5438deff66483db495e48735810"
                        .to_owned(),
                    size: 1_433,
                    url: None,
                },
            ],
            license: "Apache-2.0".to_owned(),
            dimension: Some(384),
            tier: Some(ModelTier::Quality),
            download_size_bytes: 133_807_972,
        }
    }

    /// Nomic Embed Text v1.5 manifest.
    ///
    /// Dimension: 768. Long context support with Matryoshka embedding capability.
    /// Verified checksums from `HuggingFace`.
    #[must_use]
    pub fn nomic_embed() -> Self {
        const REVISION: &str = "e5cf08aadaa33385f5990def41f7a23405aec398";
        const REPO: &str = "nomic-ai/nomic-embed-text-v1.5";
        Self {
            id: "nomic-embed-text-v1.5".to_owned(),
            version: "v1".to_owned(),
            display_name: Some("Nomic Embed Text v1.5".to_owned()),
            description: Some(
                "Long context embedding model with Matryoshka capability (768 dims)".to_owned(),
            ),
            repo: REPO.to_owned(),
            revision: REVISION.to_owned(),
            files: vec![
                ModelFile {
                    name: "onnx/model.onnx".to_owned(),
                    sha256: "147d5aa88c2101237358e17796cf3a227cead1ec304ec34b465bb08e9d952965"
                        .to_owned(),
                    size: 547_310_275,
                    url: None,
                },
                ModelFile {
                    name: "tokenizer.json".to_owned(),
                    sha256: "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"
                        .to_owned(),
                    size: 711_396,
                    url: None,
                },
                ModelFile {
                    name: "config.json".to_owned(),
                    sha256: "0168e0883705b0bf8f2b381e10f45a9f3e1ef4b13869b43c160e4c8a70ddf442"
                        .to_owned(),
                    size: 2_331,
                    url: None,
                },
                ModelFile {
                    name: "special_tokens_map.json".to_owned(),
                    sha256: "5d5b662e421ea9fac075174bb0688ee0d9431699900b90662acd44b2a350503a"
                        .to_owned(),
                    size: 695,
                    url: None,
                },
                ModelFile {
                    name: "tokenizer_config.json".to_owned(),
                    sha256: "d7e0000bcc80134debd2222220427e6bf5fa20a669f40a0d0d1409cc18e0a9bc"
                        .to_owned(),
                    size: 1_191,
                    url: None,
                },
            ],
            license: "Apache-2.0".to_owned(),
            dimension: Some(768),
            tier: Some(ModelTier::Quality),
            download_size_bytes: 548_025_888,
        }
    }

    /// Jina Reranker v1 Turbo EN manifest.
    ///
    /// Fast, optimized for English. Verified checksums from `HuggingFace`.
    #[must_use]
    pub fn jina_reranker_turbo() -> Self {
        const REVISION: &str = "b8c14f4e723d9e0aab4732a7b7b93741eeeb77c2";
        const REPO: &str = "jinaai/jina-reranker-v1-turbo-en";
        Self {
            id: "jina-reranker-v1-turbo-en".to_owned(),
            version: "v1".to_owned(),
            display_name: Some("Jina Reranker v1 Turbo EN".to_owned()),
            description: Some("Fast cross-encoder reranker optimized for English".to_owned()),
            repo: REPO.to_owned(),
            revision: REVISION.to_owned(),
            files: vec![
                ModelFile {
                    name: "onnx/model.onnx".to_owned(),
                    sha256: "c1296c66c119de645fa9cdee536d8637740efe85224cfa270281e50f213aa565"
                        .to_owned(),
                    size: 151_296_975,
                    url: None,
                },
                ModelFile {
                    name: "tokenizer.json".to_owned(),
                    sha256: "0046da43cc8c424b317f56b092b0512aaaa65c4f925d2f16af9d9eeb4d0ef902"
                        .to_owned(),
                    size: 2_030_772,
                    url: None,
                },
                ModelFile {
                    name: "config.json".to_owned(),
                    sha256: "e050ff6a15ae9295e84882fa0e98051bd8754856cd5201395ebf00ce9f2d609b"
                        .to_owned(),
                    size: 1_206,
                    url: None,
                },
                ModelFile {
                    name: "special_tokens_map.json".to_owned(),
                    sha256: "06e405a36dfe4b9604f484f6a1e619af1a7f7d09e34a8555eb0b77b66318067f"
                        .to_owned(),
                    size: 280,
                    url: None,
                },
                ModelFile {
                    name: "tokenizer_config.json".to_owned(),
                    sha256: "d291c6652d96d56ffdbcf1ea19d9bae5ed79003f7648c627e725a619227ce8fa"
                        .to_owned(),
                    size: 1_215,
                    url: None,
                },
            ],
            license: "Apache-2.0".to_owned(),
            dimension: None, // Cross-encoder produces scores, not embeddings
            tier: Some(ModelTier::Reranker),
            download_size_bytes: 153_330_448,
        }
    }

    // ==================== Lookup & Listing Functions ====================

    /// Get manifest by embedder name.
    #[must_use]
    pub fn for_embedder(name: &str) -> Option<Self> {
        match name {
            "minilm" => Some(Self::minilm_v2()),
            "multilingual-minilm" | "paraphrase-multilingual-minilm-l12-v2" => {
                Some(Self::multilingual_minilm_l12_v2())
            }
            "snowflake-arctic-s" => Some(Self::snowflake_arctic_s()),
            "nomic-embed" => Some(Self::nomic_embed()),
            "potion-128m" => Some(Self::potion_128m()),
            _ => None,
        }
    }

    /// Get manifest by reranker name.
    #[must_use]
    pub fn for_reranker(name: &str) -> Option<Self> {
        match name {
            "ms-marco" => Some(Self::ms_marco_reranker()),
            "jina-reranker-turbo" => Some(Self::jina_reranker_turbo()),
            _ => None,
        }
    }

    /// Get all bake-off eligible embedder manifests.
    #[must_use]
    pub fn bakeoff_embedder_candidates() -> Vec<Self> {
        vec![Self::snowflake_arctic_s(), Self::nomic_embed()]
    }

    /// Get all bake-off eligible reranker manifests.
    #[must_use]
    pub fn bakeoff_reranker_candidates() -> Vec<Self> {
        vec![Self::jina_reranker_turbo()]
    }

    /// Get all bake-off eligible model manifests (embedders + rerankers).
    #[must_use]
    pub fn bakeoff_candidates() -> Vec<Self> {
        let mut candidates = Self::bakeoff_embedder_candidates();
        candidates.extend(Self::bakeoff_reranker_candidates());
        candidates
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
                Self::snowflake_arctic_s(),
                Self::nomic_embed(),
                Self::jina_reranker_turbo(),
                Self::flashrank_nano(),
            ],
        }
    }

    /// Models that are discoverable and downloadable only after an explicit
    /// model selection. They are excluded from default bulk acquisition.
    #[must_use]
    pub fn opt_in_catalog() -> ModelManifestCatalog {
        ModelManifestCatalog {
            schema_version: MANIFEST_SCHEMA_VERSION,
            models: vec![Self::multilingual_minilm_l12_v2()],
        }
    }

    /// Parse a manifest from JSON and validate basic structure.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` if JSON parsing or validation fails.
    pub fn from_json_str(raw: &str) -> SearchResult<Self> {
        let manifest =
            serde_json::from_str::<Self>(raw).map_err(|_source| SearchError::InvalidConfig {
                field: "manifest_json".to_owned(),
                value: "redacted-manifest-json".to_owned(),
                reason: "failed to parse manifest JSON; input was malformed or unsupported"
                    .to_owned(),
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

    /// Returns true only for a complete lowercase commit digest.
    #[must_use]
    pub fn has_pinned_revision(&self) -> bool {
        let revision = self.revision.trim();
        matches!(revision.len(), 40 | 64)
            && revision
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    }

    /// Returns true when this manifest is ready for production-grade verification.
    #[must_use]
    pub fn is_production_ready(&self) -> bool {
        self.has_verified_checksums() && self.has_pinned_revision()
    }

    /// Freeze the complete validated download manifest into one stable identity digest.
    ///
    /// The canonical payload includes every serialized manifest field, including the
    /// pinned revision and every file name, size, URL, and SHA-256. Any manifest evolution
    /// therefore invalidates an earlier verification receipt even when the logical ID is
    /// unchanged.
    fn freeze_verification_manifest(&self) -> SearchResult<FrozenVerificationManifest> {
        self.validate()?;
        if !self.is_production_ready() {
            return Err(invalid_manifest_field(
                "production_ready",
                &self.id,
                "cached verification requires a pinned revision and SHA-256 for every file",
            ));
        }
        let serialized = serde_json::to_vec(self).map_err(|_| {
            invalid_manifest_field(
                "manifest_fingerprint",
                &self.id,
                "failed to serialize the validated manifest for canonical verification",
            )
        })?;
        let mut canonical = Vec::with_capacity(serialized.len().saturating_add(64));
        append_frozen_bytes(
            &mut canonical,
            b"frankensearch.model-download-verification-manifest.v1",
        );
        canonical.extend_from_slice(&MANIFEST_SCHEMA_VERSION.to_be_bytes());
        append_frozen_bytes(&mut canonical, &serialized);
        Ok(FrozenVerificationManifest {
            fingerprint: sha256_hex_bytes(&canonical),
        })
    }

    /// Sum of expected bytes for all files.
    #[must_use]
    pub fn total_size_bytes(&self) -> u64 {
        if self.download_size_bytes > 0 {
            return self.download_size_bytes;
        }
        self.files.iter().map(|file| file.size).sum()
    }

    /// Alias for [`total_size_bytes`](Self::total_size_bytes).
    #[must_use]
    pub fn total_size(&self) -> u64 {
        self.total_size_bytes()
    }

    /// `HuggingFace` download URL for a specific file in this manifest.
    #[must_use]
    pub fn download_url(&self, file: &ModelFile) -> String {
        file.download_url(&self.repo, &self.revision)
    }

    /// Validate manifest fields for shape and checksum format.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` for malformed fields.
    pub fn validate(&self) -> SearchResult<()> {
        for (field, value) in [
            ("id", self.id.as_str()),
            ("repo", self.repo.as_str()),
            ("revision", self.revision.as_str()),
            ("license", self.license.as_str()),
        ] {
            validate_frozen_manifest_text(field, value, false)?;
        }
        for (field, value) in [
            ("repo", self.repo.as_str()),
            ("revision", self.revision.as_str()),
        ] {
            if value.bytes().any(|byte| byte.is_ascii_whitespace())
                || value.contains('@')
                || value.contains('?')
                || value.contains('#')
            {
                return Err(invalid_manifest_field(
                    field,
                    "redacted",
                    "download coordinates must not contain whitespace, userinfo, query, or fragment",
                ));
            }
        }
        validate_frozen_manifest_text("version", &self.version, true)?;
        for (field, value) in [
            ("display_name", self.display_name.as_deref()),
            ("description", self.description.as_deref()),
        ] {
            if let Some(value) = value {
                validate_frozen_manifest_text(field, value, true)?;
            }
        }

        for file in &self.files {
            validate_model_file_name(&file.name)?;
            if file.uses_placeholder_checksum() {
            } else if !is_valid_sha256_hex(&file.sha256) {
                return Err(invalid_manifest_field(
                    "files[].sha256",
                    "redacted-invalid-sha256",
                    "must be lowercase 64-char SHA256 hex or placeholder",
                ));
            }
            if let Some(url) = &file.url {
                validate_frozen_manifest_text("files[].url", url, false)?;
                if !url.starts_with("https://")
                    || url.bytes().any(|byte| byte.is_ascii_whitespace())
                    || url.contains('@')
                    || url.contains('?')
                    || url.contains('#')
                {
                    return Err(invalid_manifest_field(
                        "files[].url",
                        "redacted",
                        "must be credential-free HTTPS without userinfo, query, or fragment",
                    ));
                }
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
            let path = resolve_model_file_path(model_dir, &file.name)?;
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
        verify_dir_and_record(self, staged_dir)?;
        sync_registered_artifacts(staged_dir, self.files.iter().map(|file| file.name.as_str()))?;
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
        serde_json::from_str::<Self>(raw).map_err(|_source| SearchError::InvalidConfig {
            field: "manifest_catalog_json".to_owned(),
            value: "redacted-manifest-catalog-json".to_owned(),
            reason: "failed to parse manifest catalog JSON; input was malformed or unsupported"
                .to_owned(),
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
    /// Every staged byte is verified, but the generation is not published.
    StagedVerified,
    /// Artifacts are installed and load-tested, but no compatible index is selected yet.
    AcquiredNeedsReindex,
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

impl ModelState {
    /// Whether the model is ready for use.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Ready)
    }

    /// Whether a download is in progress.
    #[must_use]
    pub fn is_downloading(&self) -> bool {
        matches!(self, Self::Downloading { .. })
    }

    /// Whether user consent is needed.
    #[must_use]
    pub fn needs_consent(&self) -> bool {
        matches!(self, Self::NeedsConsent)
    }

    /// Human-readable summary of the state.
    #[must_use]
    pub fn summary(&self) -> String {
        match self {
            Self::NotInstalled => "not installed".into(),
            Self::NeedsConsent => "needs consent".into(),
            Self::Downloading { progress_pct, .. } => {
                format!("downloading ({progress_pct}%)")
            }
            Self::Verifying => "verifying".into(),
            Self::StagedVerified => "staged and verified".into(),
            Self::AcquiredNeedsReindex => "acquired; compatible index required".into(),
            Self::Ready => "ready".into(),
            Self::Disabled { reason } => format!("disabled: {reason}"),
            Self::VerificationFailed { reason } => format!("verification failed: {reason}"),
            Self::UpdateAvailable {
                current_revision,
                latest_revision,
            } => {
                format!("update available: {current_revision} -> {latest_revision}")
            }
            Self::Cancelled => "cancelled".into(),
        }
    }
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

    /// Mark the unpublished staging directory verified.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` unless verification is the active phase.
    pub fn mark_staged_verified(&mut self) -> SearchResult<()> {
        if matches!(self.state, ModelState::Verifying) {
            self.state = ModelState::StagedVerified;
            return Ok(());
        }
        Err(invalid_state_transition(
            &self.state,
            "mark_staged_verified",
            "expected Verifying",
        ))
    }

    /// Record a published, load-tested generation that still needs a compatible index.
    ///
    /// A verified warm cache may bypass consent and transport states. A newly
    /// acquired generation must arrive from `StagedVerified`.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` while transport, verification, readiness, or a
    /// disabled/update state is active.
    pub fn mark_acquired_needs_reindex(&mut self) -> SearchResult<()> {
        if matches!(
            self.state,
            ModelState::NotInstalled
                | ModelState::NeedsConsent
                | ModelState::StagedVerified
                | ModelState::VerificationFailed { .. }
                | ModelState::Cancelled
        ) {
            self.state = ModelState::AcquiredNeedsReindex;
            return Ok(());
        }
        Err(invalid_state_transition(
            &self.state,
            "mark_acquired_needs_reindex",
            "expected a verified warm cache or StagedVerified generation",
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
            value: "redacted-invalid-sha256".to_owned(),
            reason: "expected lowercase 64-char SHA256 hex".to_owned(),
        });
    }
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
            return Err(SearchError::ModelNotFound {
                name: format!("missing model file: {}", path.display()),
            });
        }
        Err(source) => {
            return Err(SearchError::ModelLoadFailed {
                path: path.to_path_buf(),
                source: Box::new(source),
            });
        }
    };
    if !metadata.file_type().is_file() {
        return Err(SearchError::ModelLoadFailed {
            path: path.to_path_buf(),
            source: "expected a regular file, not a symlink or special node".into(),
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
    // ubs:ignore — artifact sizes and SHA-256 digests are public integrity metadata.
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

/// Schema version for the verification receipt itself.
///
/// This is intentionally independent from [`MANIFEST_SCHEMA_VERSION`]. Older binaries
/// compare their marker field to the manifest schema, so using a distinct value also
/// makes a newly written fingerprint-bound receipt fail closed after a binary downgrade.
pub const VERIFICATION_MARKER_SCHEMA_VERSION: u32 = 1;

/// Lightweight filesystem fingerprint for one verified model file.
///
/// This is intentionally cheap to read and compare:
/// - file size (bytes)
/// - last-modified timestamp (unix nanos)
/// - creation timestamp when the platform exposes it
/// - platform file identity and change timestamp on Unix
///
/// These fields detect ordinary replacement and mutation without re-hashing large
/// model artifacts on every process start. They are not an authenticated defense
/// against an attacker who can rewrite both the model bytes and this user-owned
/// receipt; that threat requires an OS-backed trust root such as fs-verity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileVerificationState {
    /// File size in bytes.
    pub size_bytes: u64,
    /// Last-modified timestamp since unix epoch (nanoseconds).
    pub modified_unix_nanos: u64,
    /// Creation timestamp since unix epoch (nanoseconds), when available.
    pub created_unix_nanos: Option<u64>,
    /// Stable platform file identity, when available.
    pub platform_file_id: Option<String>,
    /// Platform change timestamp, when available.
    pub platform_change_stamp: Option<String>,
}

fn capture_file_verification_state(path: &Path) -> Option<FileVerificationState> {
    let metadata = fs::symlink_metadata(path).ok()?;
    if !metadata.file_type().is_file() {
        return None;
    }
    let modified = metadata
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_nanos();
    let modified_unix_nanos = u64::try_from(modified).ok()?;
    let created_unix_nanos = metadata
        .created()
        .ok()
        .and_then(|created| created.duration_since(UNIX_EPOCH).ok())
        .and_then(|created| u64::try_from(created.as_nanos()).ok());
    #[cfg(unix)]
    let (platform_file_id, platform_change_stamp) = {
        use std::os::unix::fs::MetadataExt as _;
        (
            Some(format!(
                "unix-dev={};ino={}",
                metadata.dev(),
                metadata.ino()
            )),
            Some(format!(
                "unix-ctime={};nsec={}",
                metadata.ctime(),
                metadata.ctime_nsec()
            )),
        )
    };
    #[cfg(not(unix))]
    let (platform_file_id, platform_change_stamp) = (None, None);
    Some(FileVerificationState {
        size_bytes: metadata.len(),
        modified_unix_nanos,
        created_unix_nanos,
        platform_file_id,
        platform_change_stamp,
    })
}

fn resolve_model_file_path(model_dir: &Path, file_name: &str) -> SearchResult<PathBuf> {
    validate_model_file_name(file_name)?;
    Ok(model_dir.join(file_name))
}

fn validate_model_file_name(file_name: &str) -> SearchResult<()> {
    if file_name.len() > MAX_FROZEN_MANIFEST_FIELD_BYTES {
        return Err(invalid_manifest_field(
            "files[].name",
            "redacted-oversized",
            "must fit the bounded manifest field size",
        ));
    }
    if file_name.chars().any(char::is_control) {
        return Err(invalid_manifest_field(
            "files[].name",
            "redacted-control-character",
            "must not contain control characters",
        ));
    }
    if file_name.trim().is_empty() {
        return Err(invalid_manifest_field(
            "files[].name",
            file_name,
            "must not be empty",
        ));
    }
    for component in Path::new(file_name).components() {
        match component {
            std::path::Component::ParentDir => {
                return Err(invalid_manifest_field(
                    "files[].name",
                    "redacted-path-traversal",
                    "must not contain '..' path traversal",
                ));
            }
            std::path::Component::RootDir | std::path::Component::Prefix(_) => {
                return Err(invalid_manifest_field(
                    "files[].name",
                    "redacted-absolute-path",
                    "must be a relative path without root",
                ));
            }
            _ => {}
        }
    }
    if file_name.contains('\\')
        || file_name
            .bytes()
            .any(|byte| matches!(byte, b':' | b'<' | b'>' | b'"' | b'|' | b'?' | b'*'))
        || file_name.split('/').any(|component| {
            component.is_empty()
                || component == "."
                || component.trim() != component
                || component.ends_with('.')
                || is_windows_reserved_path_component(component)
        })
    {
        return Err(invalid_manifest_field(
            "files[].name",
            "redacted-nonportable-path",
            "must use a canonical portable relative path",
        ));
    }
    Ok(())
}

fn is_windows_reserved_path_component(component: &str) -> bool {
    let stem = component
        .split_once('.')
        .map_or(component, |(stem, _suffix)| stem);
    let upper = stem.to_ascii_uppercase();
    matches!(
        upper.as_str(),
        "CON"
            | "PRN"
            | "AUX"
            | "NUL"
            | "CLOCK$"
            | "CONIN$"
            | "CONOUT$"
            | "COM¹"
            | "COM²"
            | "COM³"
            | "LPT¹"
            | "LPT²"
            | "LPT³"
    ) || upper
        .strip_prefix("COM")
        .or_else(|| upper.strip_prefix("LPT"))
        .is_some_and(|suffix| matches!(suffix.as_bytes(), &[b'1'..=b'9']))
}

/// Cached verification receipt stored as a small JSON file alongside model files.
///
/// When a model directory passes SHA-256 verification, a `.verified` marker is written
/// containing the exact frozen production-manifest fingerprint and lightweight file
/// fingerprints captured at verification time. Subsequent loads may skip re-hashing
/// only when both the manifest identity and every file state remain unchanged.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerificationMarker {
    /// Manifest identifier that was verified against.
    pub manifest_id: String,
    /// Manifest schema version at time of verification.
    pub schema_version: u32,
    /// SHA-256 of the complete canonical production manifest.
    pub manifest_fingerprint: String,
    /// Unix timestamp (seconds) when verification was performed.
    pub verified_at: u64,
    /// Per-file lightweight fingerprint at verification time, keyed by file name.
    pub file_states: BTreeMap<String, FileVerificationState>,
}

impl VerificationMarker {
    /// Create a receipt from file states captured around a successful full hash pass.
    fn from_verified_states(
        manifest: &ModelManifest,
        manifest_fingerprint: String,
        file_states: BTreeMap<String, FileVerificationState>,
    ) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        Self {
            manifest_id: manifest.id.clone(),
            schema_version: VERIFICATION_MARKER_SCHEMA_VERSION,
            manifest_fingerprint,
            verified_at: now,
            file_states,
        }
    }

    /// Check whether this cached marker is still valid for the given manifest and directory.
    ///
    /// Returns `true` when:
    /// 1. The manifest is production-ready and its frozen fingerprint matches.
    /// 2. The manifest ID and schema version match.
    /// 3. No model file metadata fingerprint has changed since verification.
    fn is_valid_for(&self, manifest: &ModelManifest, model_dir: &Path) -> bool {
        let Ok(frozen) = manifest.freeze_verification_manifest() else {
            return false;
        };
        if self.manifest_id != manifest.id
            || self.schema_version != VERIFICATION_MARKER_SCHEMA_VERSION
            || self.manifest_fingerprint != frozen.fingerprint
            || self.file_states.len() != manifest.files.len()
        {
            return false;
        }

        for file in &manifest.files {
            let Some(expected_state) = self.file_states.get(&file.name) else {
                return false;
            };
            let Ok(path) = resolve_model_file_path(model_dir, &file.name) else {
                return false;
            };
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

fn capture_manifest_file_states(
    manifest: &ModelManifest,
    model_dir: &Path,
) -> SearchResult<BTreeMap<String, FileVerificationState>> {
    let mut file_states = BTreeMap::new();
    for file in &manifest.files {
        let path = resolve_model_file_path(model_dir, &file.name)?;
        let state =
            capture_file_verification_state(&path).ok_or_else(|| SearchError::ModelNotFound {
                name: format!("{}:{}", manifest.id, file.name),
            })?;
        file_states.insert(file.name.clone(), state);
    }
    Ok(file_states)
}

#[cfg(test)]
thread_local! {
    static TEST_AFTER_FULL_HASH_HOOK: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
}

fn after_full_hash_boundary() {
    #[cfg(test)]
    TEST_AFTER_FULL_HASH_HOOK.with(|slot| {
        if let Some(hook) = slot.borrow_mut().take() {
            hook();
        }
    });
}

fn write_verification_marker_atomic(
    marker: &VerificationMarker,
    model_dir: &Path,
) -> SearchResult<()> {
    let encoded =
        serde_json::to_vec_pretty(marker).map_err(|source| SearchError::SubsystemError {
            subsystem: "model_verification_receipt",
            source: Box::new(source),
        })?;
    let mut temporary = tempfile::Builder::new()
        .prefix(".verified.")
        .suffix(".tmp")
        .tempfile_in(model_dir)
        .map_err(SearchError::from)?;
    temporary.write_all(&encoded).map_err(SearchError::from)?;
    temporary.as_file().sync_all().map_err(SearchError::from)?;

    let current_states = capture_manifest_file_states_from_names(
        marker.file_states.keys().map(String::as_str),
        model_dir,
    )?;
    if current_states != marker.file_states {
        return Err(model_changed_during_verification(model_dir));
    }

    temporary
        .persist(model_dir.join(VERIFIED_MARKER_FILE))
        .map_err(|error| SearchError::from(error.error))?;
    sync_directory(model_dir)
}

fn capture_manifest_file_states_from_names<'a>(
    file_names: impl IntoIterator<Item = &'a str>,
    model_dir: &Path,
) -> SearchResult<BTreeMap<String, FileVerificationState>> {
    let mut file_states = BTreeMap::new();
    for file_name in file_names {
        let path = resolve_model_file_path(model_dir, file_name)?;
        let state =
            capture_file_verification_state(&path).ok_or_else(|| SearchError::ModelNotFound {
                name: file_name.to_owned(),
            })?;
        file_states.insert(file_name.to_owned(), state);
    }
    Ok(file_states)
}

fn model_changed_during_verification(model_dir: &Path) -> SearchError {
    SearchError::HashMismatch {
        path: model_dir.to_path_buf(),
        expected: "stable file identity and metadata across full SHA-256 verification".to_owned(),
        actual: "model file state changed while verification was in progress".to_owned(),
    }
}

/// Check whether a valid verification marker exists for the given manifest and directory.
///
/// Returns `true` when a `.verified` file exists, parses as the current complete schema,
/// matches the exact frozen production manifest, and all recorded file states match.
///
/// The marker is a same-user performance receipt, not an authenticated artifact. Explicit
/// verification and every bundled write/promotion still perform full SHA-256 verification.
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

/// Verify a model directory, using a valid cached receipt when available.
///
/// If a valid `.verified` receipt exists (matching the exact frozen production manifest
/// and all file states), verification succeeds immediately without re-hashing. Otherwise,
/// full SHA-256 verification is performed via [`ModelManifest::verify_dir`] for this
/// invocation without mutating the cache. Non-production manifests are never admitted to
/// the cached path.
///
/// # Errors
///
/// Returns `SearchError` when the manifest is not production-ready or full verification
/// fails (hash mismatch, missing files, etc.).
pub fn verify_dir_cached(manifest: &ModelManifest, model_dir: &Path) -> SearchResult<()> {
    manifest.freeze_verification_manifest()?;

    if is_verification_cached(manifest, model_dir) {
        return Ok(());
    }

    manifest.verify_dir(model_dir)
}

/// Perform full SHA-256 verification and atomically record a cache receipt.
///
/// This is the only receipt-minting API. It captures every registered file's state before
/// hashing, performs full size-and-SHA verification, then refuses publication if any file
/// identity or metadata changed during verification. The receipt file is synced and
/// atomically renamed before its directory is synced.
///
/// External callers cannot bypass the hash pass with a raw marker writer:
///
/// ```compile_fail
/// use frankensearch_embed::write_verification_marker;
/// ```
///
/// # Errors
///
/// Returns `SearchError` when the manifest is not production-ready, any registered file
/// fails full verification, a file changes during verification, or durable receipt
/// publication fails.
pub fn verify_dir_and_record(manifest: &ModelManifest, model_dir: &Path) -> SearchResult<()> {
    let frozen = manifest.freeze_verification_manifest()?;
    let before = capture_manifest_file_states(manifest, model_dir)?;
    manifest.verify_dir(model_dir)?;
    after_full_hash_boundary();
    let after = capture_manifest_file_states(manifest, model_dir)?;
    if before != after {
        return Err(model_changed_during_verification(model_dir));
    }

    let marker = VerificationMarker::from_verified_states(manifest, frozen.fingerprint, after);
    write_verification_marker_atomic(&marker, model_dir)
}

fn sync_registered_artifacts<'a>(
    staged_dir: &Path,
    relative_paths: impl IntoIterator<Item = &'a str>,
) -> SearchResult<()> {
    let mut parent_dirs = BTreeSet::<PathBuf>::new();
    for relative_path in relative_paths {
        let artifact_path = resolve_model_file_path(staged_dir, relative_path)?;
        File::open(&artifact_path)
            .and_then(|file| file.sync_all())
            .map_err(SearchError::from)?;
        if let Some(parent) = artifact_path.parent() {
            parent_dirs.insert(parent.to_path_buf());
        }
    }
    for parent in &parent_dirs {
        sync_directory(parent)?;
    }
    sync_directory(staged_dir)
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> SearchResult<()> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(SearchError::from)
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> SearchResult<()> {
    // Windows does not expose a portable directory fsync through std. Every
    // registered file is still synced before the atomic rename.
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PublicationBoundary {
    InstallingParentSync,
    BackupParentSync,
    PublishRename,
    PublishedParentSync,
}

#[cfg(test)]
thread_local! {
    static TEST_PUBLICATION_FAULT: std::cell::Cell<Option<PublicationBoundary>> =
        const { std::cell::Cell::new(None) };
}

#[allow(clippy::unnecessary_wraps)] // Test builds inject typed durability failures here.
fn publication_boundary(boundary: PublicationBoundary) -> SearchResult<()> {
    #[cfg(test)]
    if TEST_PUBLICATION_FAULT.with(|fault| fault.get() == Some(boundary)) {
        return Err(SearchError::from(std::io::Error::other(format!(
            "injected publication failure at {boundary:?}"
        ))));
    }
    #[cfg(not(test))]
    let _ = boundary;
    Ok(())
}

#[cfg(test)]
struct PublicationFaultGuard;

#[cfg(test)]
impl PublicationFaultGuard {
    fn install(boundary: PublicationBoundary) -> Self {
        TEST_PUBLICATION_FAULT.with(|fault| {
            assert!(
                fault.replace(Some(boundary)).is_none(),
                "publication test fault already installed"
            );
        });
        Self
    }
}

#[cfg(test)]
impl Drop for PublicationFaultGuard {
    fn drop(&mut self) {
        TEST_PUBLICATION_FAULT.with(|fault| fault.set(None));
    }
}

fn promote_atomically(staged_dir: &Path, destination_dir: &Path) -> SearchResult<Option<PathBuf>> {
    let destination_parent =
        destination_dir
            .parent()
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "destination_dir".to_owned(),
                value: "redacted".to_owned(),
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
    publication_boundary(PublicationBoundary::InstallingParentSync)?;
    sync_directory(destination_parent)?;

    let backup_path = if destination_dir.exists() {
        let backup = destination_parent.join(format!("{stage_name}.backup.{timestamp}.{pid}"));
        fs::rename(destination_dir, &backup).map_err(SearchError::from)?;
        let backup_sync = publication_boundary(PublicationBoundary::BackupParentSync)
            .and_then(|()| sync_directory(destination_parent));
        if let Err(sync_error) = backup_sync {
            if let Err(rollback_error) = fs::rename(&backup, destination_dir) {
                tracing::error!(
                    rollback_error_kind = ?rollback_error.kind(),
                    "model backup sync failed and the prior generation remains in its backup"
                );
            } else {
                let _ = sync_directory(destination_parent);
            }
            return Err(sync_error);
        }
        Some(backup)
    } else {
        None
    };

    let publish = publication_boundary(PublicationBoundary::PublishRename)
        .and_then(|()| fs::rename(&stage_target, destination_dir).map_err(SearchError::from));
    if let Err(publish_error) = publish {
        if let Some(backup) = &backup_path
            && let Err(rollback_error) = fs::rename(backup, destination_dir)
        {
            tracing::error!(
                rollback_error_kind = ?rollback_error.kind(),
                "model publication failed and prior generation remains in its backup"
            );
        }
        let _ = sync_directory(destination_parent);
        return Err(publish_error);
    }
    publication_boundary(PublicationBoundary::PublishedParentSync)?;
    sync_directory(destination_parent)?;
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
        value: truncate_for_error(value),
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
    use frankensearch_core::generation::{
        ForeignProducerConformanceCertificateV1, ProducerCompatibilityErrorV1,
        ProducerCompatibilityKindV1, TrustedProducerConformanceContextV1,
        VerifiedGoldenConformanceManifestV1,
    };
    use std::io::Write;

    fn write_temp_file(path: &Path, bytes: &[u8]) {
        let mut file = File::create(path).unwrap();
        file.write_all(bytes).unwrap();
        file.flush().unwrap();
    }

    fn sample_artifact_manifest() -> ModelArtifactManifestV1 {
        let input_contract = default_plain_text_input_contract();
        ModelArtifactManifestV1 {
            schema_version: MODEL_ARTIFACT_MANIFEST_SCHEMA_V1,
            provider: "fixture-provider".to_owned(),
            logical_model_id: "fixture-semantic-model".to_owned(),
            upstream_revision: "0123456789abcdef0123456789abcdef01234567".to_owned(),
            upstream_repository: "fixture/model".to_owned(),
            artifacts: vec![
                ModelArtifactFileV1 {
                    role: ModelArtifactRoleV1::Weights,
                    relative_path: "model.safetensors".to_owned(),
                    upstream_url: "https://models.example.invalid/immutable/model.safetensors"
                        .to_owned(),
                    size: 7,
                    sha256: to_hex_lowercase(&Sha256::digest(b"weights")),
                },
                ModelArtifactFileV1 {
                    role: ModelArtifactRoleV1::Tokenizer,
                    relative_path: "tokenizer.json".to_owned(),
                    upstream_url: "https://models.example.invalid/immutable/tokenizer.json"
                        .to_owned(),
                    size: 9,
                    sha256: to_hex_lowercase(&Sha256::digest(b"tokenizer")),
                },
            ],
            license_spdx: "Apache-2.0".to_owned(),
            license_metadata_sha256: license_metadata_fingerprint(
                "Apache-2.0",
                "fixture-provider",
                "fixture/model",
                "0123456789abcdef0123456789abcdef01234567",
            ),
            dimension: 3,
            execution: ModelExecutionContractV1 {
                backend: "fixture-native".to_owned(),
                implementation_revision: "fixture-implementation-v1".to_owned(),
                protocol_revision: "fixture-protocol-v1".to_owned(),
                numeric_profile: "f32-fixture-v1".to_owned(),
                weights_format: "safetensors-f32-v1".to_owned(),
                tokenizer_family: "fixture-tokenizer-v1".to_owned(),
                model_preprocessing: "fixture-preprocessing-v1".to_owned(),
                sequence_policy: "max-length=8;truncate-right;no-padding".to_owned(),
                pooling: "mean-v1".to_owned(),
                output_normalization: "l2-f32-v1".to_owned(),
                query_instruction: "query: ".to_owned(),
                document_instruction: "document: ".to_owned(),
                input_contract,
                golden_vectors: GoldenVectorCertificateV1 {
                    corpus_sha256: "1".repeat(64),
                    vectors_sha256: "2".repeat(64),
                    vector_count: 2,
                    dimension: 3,
                },
            },
        }
    }

    #[test]
    fn frozen_artifact_manifest_is_role_order_canonical_and_exact() {
        let left = sample_artifact_manifest();
        left.validate().unwrap();
        let left_frozen = left.freeze().unwrap();
        left_frozen.validate().unwrap();

        let mut right = left;
        right.artifacts.reverse();
        right.validate().unwrap();
        let right_frozen = right.freeze().unwrap();
        assert_eq!(left_frozen.canonical_bytes, right_frozen.canonical_bytes);
        assert_eq!(left_frozen.fingerprint, right_frozen.fingerprint);
    }

    #[test]
    fn foreign_producers_require_explicit_trusted_conformance_certificates() {
        let conformance_texts = ["query: alpha", "document: beta"];
        let conformance_vectors = vec![vec![0.0, -0.0, 1.0], vec![0.5, -0.5, 0.25]];
        let fixture = VerifiedGoldenConformanceManifestV1::from_exact_pair_f32(
            &conformance_texts,
            &conformance_vectors,
            &conformance_vectors,
        )
        .unwrap();
        let policy_fingerprint = "9".repeat(64);

        let mut left = sample_artifact_manifest();
        left.execution.golden_vectors = fixture.certificate().clone();
        let left_manifest_fingerprint = left.freeze().unwrap().fingerprint;
        let left_identity = left
            .identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
            .unwrap();

        let mut right = left.clone();
        right.execution.backend = "fixture-alternate-backend".to_owned();
        right.execution.implementation_revision = "fixture-implementation-v2".to_owned();
        right.execution.protocol_revision = "fixture-protocol-v2".to_owned();
        right.execution.numeric_profile = "f32-fixture-alternate-kernel-v1".to_owned();
        let right_manifest_fingerprint = right.freeze().unwrap().fingerprint;
        let right_identity = right
            .identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
            .unwrap();

        assert_ne!(left_manifest_fingerprint, right_manifest_fingerprint);
        assert_eq!(
            left_identity.producer.provenance_manifest_fingerprint,
            left_manifest_fingerprint
        );
        assert_eq!(
            right_identity.producer.provenance_manifest_fingerprint,
            right_manifest_fingerprint
        );
        assert_eq!(
            left_identity.space.fingerprint(),
            right_identity.space.fingerprint()
        );
        assert_eq!(
            left_identity.verify_exact_producer_with(&right_identity),
            Err(ProducerCompatibilityErrorV1::CertificateRequired)
        );
        let right_certificate =
            ForeignProducerConformanceCertificateV1::new_untrusted_receipt_from_verified_pair(
                &left_identity,
                &right_identity,
                &fixture,
                &policy_fingerprint,
                1,
                100,
                200,
            )
            .unwrap();
        let right_certificate_fingerprint = right_certificate.fingerprint();
        let trusted = TrustedProducerConformanceContextV1::from_independent_policy(
            &policy_fingerprint,
            &right_certificate_fingerprint,
            &fixture,
            150,
            1,
            1,
        )
        .unwrap();
        assert_eq!(
            left_identity
                .verify_certified_foreign_producer_with(
                    &right_identity,
                    &right_certificate,
                    trusted,
                )
                .unwrap()
                .kind(),
            ProducerCompatibilityKindV1::Certified
        );
        assert_ne!(
            left_identity.producer.fingerprint(),
            right_identity.producer.fingerprint()
        );
        assert_ne!(left_identity.fingerprint(), right_identity.fingerprint());

        let mut redistributed = left.clone();
        redistributed.provider = "fixture-mirror-provider".to_owned();
        redistributed.upstream_repository = "fixture/model-mirror".to_owned();
        redistributed.license_spdx = "MIT".to_owned();
        redistributed.license_metadata_sha256 = license_metadata_fingerprint(
            &redistributed.license_spdx,
            &redistributed.provider,
            &redistributed.upstream_repository,
            &redistributed.upstream_revision,
        );
        for artifact in &mut redistributed.artifacts {
            artifact.upstream_url = format!(
                "https://mirror.example.invalid/immutable/{}",
                artifact.relative_path
            );
        }
        let redistributed_identity = redistributed
            .identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
            .unwrap();
        assert_eq!(
            left_identity.space.fingerprint(),
            redistributed_identity.space.fingerprint()
        );
        assert_ne!(
            left_identity.producer.fingerprint(),
            redistributed_identity.producer.fingerprint()
        );
        assert_eq!(
            left_identity.verify_exact_producer_with(&redistributed_identity),
            Err(ProducerCompatibilityErrorV1::CertificateRequired)
        );
        let redistributed_certificate =
            ForeignProducerConformanceCertificateV1::new_untrusted_receipt_from_verified_pair(
                &left_identity,
                &redistributed_identity,
                &fixture,
                &policy_fingerprint,
                2,
                100,
                200,
            )
            .unwrap();
        let redistributed_certificate_fingerprint = redistributed_certificate.fingerprint();
        let trusted = TrustedProducerConformanceContextV1::from_independent_policy(
            &policy_fingerprint,
            &redistributed_certificate_fingerprint,
            &fixture,
            150,
            2,
            2,
        )
        .unwrap();
        assert_eq!(
            left_identity
                .verify_certified_foreign_producer_with(
                    &redistributed_identity,
                    &redistributed_certificate,
                    trusted,
                )
                .unwrap()
                .kind(),
            ProducerCompatibilityKindV1::Certified,
            "distribution metadata may differ only with explicit trusted evidence"
        );

        let mut nonconformant = right.clone();
        nonconformant.execution.golden_vectors.vectors_sha256 = "8".repeat(64);
        let nonconformant_identity = nonconformant
            .identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
            .unwrap();
        assert_eq!(
            left_identity.space.fingerprint(),
            nonconformant_identity.space.fingerprint()
        );
        assert_eq!(
            ForeignProducerConformanceCertificateV1::new_untrusted_receipt_from_verified_pair(
                &left_identity,
                &nonconformant_identity,
                &fixture,
                &policy_fingerprint,
                3,
                100,
                200,
            ),
            Err(ProducerCompatibilityErrorV1::GoldenVectorMismatch)
        );

        let mut changed_semantics = right;
        changed_semantics.execution.pooling.push_str("-drift");
        assert_ne!(
            left_identity.space.fingerprint(),
            changed_semantics
                .identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
                .unwrap()
                .space
                .fingerprint()
        );
    }

    #[test]
    fn every_artifact_manifest_field_participates_in_the_fingerprint() {
        let base = sample_artifact_manifest();
        let base_fingerprint = sha256_hex_bytes(&base.canonical_bytes());
        macro_rules! changed {
            ($label:literal, $mutate:expr) => {{
                let mut candidate = base.clone();
                $mutate(&mut candidate);
                assert_ne!(
                    base_fingerprint,
                    sha256_hex_bytes(&candidate.canonical_bytes()),
                    $label
                );
            }};
        }

        changed!("schema", |m: &mut ModelArtifactManifestV1| m
            .schema_version +=
            1);
        changed!("provider", |m: &mut ModelArtifactManifestV1| m
            .provider
            .push('x'));
        changed!("model", |m: &mut ModelArtifactManifestV1| m
            .logical_model_id
            .push('x'));
        changed!("revision", |m: &mut ModelArtifactManifestV1| m
            .upstream_revision
            .push('x'));
        changed!("repository", |m: &mut ModelArtifactManifestV1| m
            .upstream_repository
            .push('x'));
        changed!("role", |m: &mut ModelArtifactManifestV1| m.artifacts[0]
            .role =
            ModelArtifactRoleV1::Projection);
        changed!("path", |m: &mut ModelArtifactManifestV1| m.artifacts[0]
            .relative_path
            .push('x'));
        changed!("url", |m: &mut ModelArtifactManifestV1| m.artifacts[0]
            .upstream_url
            .push('x'));
        changed!("size", |m: &mut ModelArtifactManifestV1| m.artifacts[0]
            .size += 1);
        changed!("artifact digest", |m: &mut ModelArtifactManifestV1| m
            .artifacts[0]
            .sha256 =
            "3".repeat(64));
        changed!("license", |m: &mut ModelArtifactManifestV1| m
            .license_spdx
            .push('x'));
        changed!("license digest", |m: &mut ModelArtifactManifestV1| m
            .license_metadata_sha256 =
            "4".repeat(64));
        changed!("dimension", |m: &mut ModelArtifactManifestV1| m
            .dimension +=
            1);
        changed!("backend", |m: &mut ModelArtifactManifestV1| m
            .execution
            .backend
            .push('x'));
        changed!("implementation", |m: &mut ModelArtifactManifestV1| m
            .execution
            .implementation_revision
            .push('x'));
        changed!("protocol", |m: &mut ModelArtifactManifestV1| m
            .execution
            .protocol_revision
            .push('x'));
        changed!("numeric", |m: &mut ModelArtifactManifestV1| m
            .execution
            .numeric_profile
            .push('x'));
        changed!("weights format", |m: &mut ModelArtifactManifestV1| m
            .execution
            .weights_format
            .push('x'));
        changed!("tokenizer family", |m: &mut ModelArtifactManifestV1| m
            .execution
            .tokenizer_family
            .push('x'));
        changed!("preprocessing", |m: &mut ModelArtifactManifestV1| m
            .execution
            .model_preprocessing
            .push('x'));
        changed!("sequence", |m: &mut ModelArtifactManifestV1| m
            .execution
            .sequence_policy
            .push('x'));
        changed!("pooling", |m: &mut ModelArtifactManifestV1| m
            .execution
            .pooling
            .push('x'));
        changed!("normalization", |m: &mut ModelArtifactManifestV1| m
            .execution
            .output_normalization
            .push('x'));
        changed!("query instruction", |m: &mut ModelArtifactManifestV1| m
            .execution
            .query_instruction
            .push('x'));
        changed!("document instruction", |m: &mut ModelArtifactManifestV1| m
            .execution
            .document_instruction
            .push('x'));
        changed!("input contract", |m: &mut ModelArtifactManifestV1| m
            .execution
            .input_contract
            .canonicalization
            .push('x'));
        changed!("golden corpus", |m: &mut ModelArtifactManifestV1| m
            .execution
            .golden_vectors
            .corpus_sha256 =
            "5".repeat(64));
        changed!("golden vectors", |m: &mut ModelArtifactManifestV1| m
            .execution
            .golden_vectors
            .vectors_sha256 =
            "6".repeat(64));
        changed!("golden count", |m: &mut ModelArtifactManifestV1| m
            .execution
            .golden_vectors
            .vector_count +=
            1);
        changed!("golden dimension", |m: &mut ModelArtifactManifestV1| m
            .execution
            .golden_vectors
            .dimension +=
            1);
    }

    #[test]
    fn frozen_manifest_rejects_schema_roles_bytes_and_digest_disagreement() {
        let manifest = sample_artifact_manifest();
        let frozen = manifest.freeze().unwrap();

        let mut unknown_schema = manifest.clone();
        unknown_schema.schema_version += 1;
        assert!(unknown_schema.validate().is_err());

        let mut floating_revision = manifest.clone();
        floating_revision.upstream_revision = "main".to_owned();
        assert!(floating_revision.validate().is_err());

        let mut duplicate_role = manifest.clone();
        duplicate_role.artifacts[1].role = ModelArtifactRoleV1::Weights;
        assert!(duplicate_role.validate().is_err());

        let mut stale_license_assertion = manifest.clone();
        stale_license_assertion.license_spdx = "MIT".to_owned();
        assert!(stale_license_assertion.validate().is_err());

        let mut oversized_instruction = manifest.clone();
        oversized_instruction.execution.query_instruction =
            "x".repeat(MAX_FROZEN_MANIFEST_FIELD_BYTES + 1);
        assert!(oversized_instruction.validate().is_err());

        let mut missing_role = manifest;
        missing_role
            .artifacts
            .retain(|artifact| artifact.role != ModelArtifactRoleV1::Tokenizer);
        assert!(missing_role.validate().is_err());

        let mut bad_bytes = frozen.clone();
        bad_bytes.canonical_bytes.push(0);
        assert!(bad_bytes.validate().is_err());

        let mut bad_digest = frozen.clone();
        bad_digest.fingerprint = "0".repeat(64);
        assert!(bad_digest.validate().is_err());

        let mut injected_digest = frozen;
        injected_digest.fingerprint = "digest\nforged-log-line".to_owned();
        let error = injected_digest.validate().unwrap_err();
        assert!(error.to_string().contains("redacted-invalid-sha256"));
        assert!(!error.to_string().contains("forged-log-line"));

        let mut unknown_field = serde_json::to_value(sample_artifact_manifest()).unwrap();
        unknown_field["execution"]["future_unregistered_field"] = serde_json::json!(true);
        assert!(
            serde_json::from_value::<ModelArtifactManifestV1>(unknown_field).is_err(),
            "versioned manifests must reject unknown fields instead of silently dropping them"
        );

        let mut credentialed_url = sample_artifact_manifest();
        credentialed_url.artifacts[0].upstream_url =
            "https://models.example.invalid/model.safetensors?token=secret".to_owned();
        let error = credentialed_url.freeze().unwrap_err();
        assert!(error.to_string().contains("credential-free HTTPS"));
        assert!(!error.to_string().contains("secret"));

        let mut log_injection = sample_artifact_manifest();
        log_injection.logical_model_id = "fixture\nforged-log-line".to_owned();
        let error = log_injection.freeze().unwrap_err();
        assert!(error.to_string().contains("control characters"));
        assert!(!error.to_string().contains("forged-log-line"));

        let mut digest_injection = sample_artifact_manifest();
        digest_injection.artifacts[0].sha256 = "digest\nforged-log-line".to_owned();
        let error = digest_injection.freeze().unwrap_err();
        assert!(error.to_string().contains("redacted-invalid-sha256"));
        assert!(!error.to_string().contains("forged-log-line"));
    }

    #[test]
    fn streaming_artifact_verification_rejects_size_hash_missing_and_role_collision() {
        let manifest = sample_artifact_manifest();
        let exact = tempfile::tempdir().unwrap();
        write_temp_file(&exact.path().join("model.safetensors"), b"weights");
        write_temp_file(&exact.path().join("tokenizer.json"), b"tokenizer");
        write_temp_file(&exact.path().join("NOTICE"), b"fixture notice");
        let verified = manifest.verify_dir(exact.path()).unwrap();
        verified
            .identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
            .unwrap()
            .validate()
            .unwrap();

        let wrong_size = tempfile::tempdir().unwrap();
        write_temp_file(&wrong_size.path().join("model.safetensors"), b"weight");
        write_temp_file(&wrong_size.path().join("tokenizer.json"), b"tokenizer");
        assert!(manifest.verify_dir(wrong_size.path()).is_err());

        let wrong_hash = tempfile::tempdir().unwrap();
        write_temp_file(&wrong_hash.path().join("model.safetensors"), b"WeightS");
        write_temp_file(&wrong_hash.path().join("tokenizer.json"), b"tokenizer");
        assert!(manifest.verify_dir(wrong_hash.path()).is_err());

        let missing = tempfile::tempdir().unwrap();
        write_temp_file(&missing.path().join("model.safetensors"), b"weights");
        assert!(manifest.verify_dir(missing.path()).is_err());

        let all_missing = tempfile::tempdir().unwrap();
        let error = manifest.verify_dir(all_missing.path()).unwrap_err();
        assert!(matches!(&error, SearchError::ModelNotFound { .. }));
        let diagnostic = error.to_string();
        assert!(diagnostic.contains("model.safetensors:missing"));
        assert!(diagnostic.contains("tokenizer.json:missing"));
        assert!(diagnostic.contains("https://models.example.invalid/immutable/model.safetensors"));
        assert!(diagnostic.contains("https://models.example.invalid/immutable/tokenizer.json"));
        assert!(
            !diagnostic.contains(&all_missing.path().display().to_string()),
            "artifact diagnostics must not expose the raw model-directory path"
        );

        let mixed_failure = tempfile::tempdir().unwrap();
        write_temp_file(&mixed_failure.path().join("model.safetensors"), b"WeightS");
        let error = manifest.verify_dir(mixed_failure.path()).unwrap_err();
        assert!(matches!(&error, SearchError::ModelLoadFailed { .. }));
        let diagnostic = error.to_string();
        assert!(diagnostic.contains("model.safetensors:sha256-or-size-mismatch"));
        assert!(diagnostic.contains("tokenizer.json:missing"));
        assert!(diagnostic.contains("<redacted-model-dir>"));
        assert!(
            !diagnostic.contains(&mixed_failure.path().display().to_string()),
            "mixed artifact diagnostics must not expose the raw model-directory path"
        );

        let collision = tempfile::tempdir().unwrap();
        write_temp_file(&collision.path().join("model.safetensors"), b"weights");
        write_temp_file(&collision.path().join("tokenizer.json"), b"tokenizer");
        fs::create_dir_all(collision.path().join("unregistered")).unwrap();
        write_temp_file(
            &collision.path().join("unregistered/model.safetensors"),
            b"collision",
        );
        assert!(manifest.verify_dir(collision.path()).is_err());

        let unexpected_role = tempfile::tempdir().unwrap();
        write_temp_file(
            &unexpected_role.path().join("model.safetensors"),
            b"weights",
        );
        write_temp_file(&unexpected_role.path().join("tokenizer.json"), b"tokenizer");
        write_temp_file(&unexpected_role.path().join("config.json"), b"{}");
        assert!(manifest.verify_dir(unexpected_role.path()).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn artifact_verification_rejects_symlinked_extra_role_collision() {
        let manifest = sample_artifact_manifest();
        let collision = tempfile::tempdir().unwrap();
        write_temp_file(&collision.path().join("model.safetensors"), b"weights");
        write_temp_file(&collision.path().join("tokenizer.json"), b"tokenizer");
        fs::create_dir_all(collision.path().join("unregistered")).unwrap();
        let target = collision.path().join("unrelated.bin");
        write_temp_file(&target, b"unrelated");
        std::os::unix::fs::symlink(&target, collision.path().join("unregistered/model.onnx"))
            .unwrap();

        let error = manifest.verify_dir(collision.path()).unwrap_err();
        assert!(error.to_string().contains("semantic role"));
    }

    #[test]
    fn every_registered_local_backend_has_a_frozen_manifest() {
        for manifest in [
            ModelArtifactManifestV1::potion_128m_native().unwrap(),
            ModelArtifactManifestV1::minilm_fastembed().unwrap(),
            ModelArtifactManifestV1::minilm_native_frankentorch().unwrap(),
            ModelArtifactManifestV1::multilingual_minilm_native_frankentorch().unwrap(),
            ModelArtifactManifestV1::snowflake_fastembed().unwrap(),
            ModelArtifactManifestV1::nomic_fastembed().unwrap(),
        ] {
            let frozen = manifest.freeze().unwrap();
            frozen.validate().unwrap();
            assert_eq!(
                frozen.manifest.dimension,
                frozen.manifest.execution.golden_vectors.dimension
            );
            assert_eq!(
                frozen.manifest.execution.golden_vectors.corpus_sha256,
                GoldenVectorCertificateV1::corpus_fingerprint(&MODEL_CONFORMANCE_TEXTS_V1).unwrap()
            );
            let identity = frozen
                .manifest
                .declared_identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
                .unwrap();
            assert_eq!(identity.storage.endianness, "native-f32-values");
        }
    }

    #[test]
    fn persisted_storage_identity_remains_little_endian() {
        let identity = ModelArtifactManifestV1::potion_128m_native()
            .unwrap()
            .declared_identity_bundle(QuantizationFormat::F16, "fsvi-v2")
            .unwrap();
        assert_eq!(identity.storage.endianness, "little-endian");
    }

    #[test]
    fn registered_manifest_fingerprints_are_exact_fixtures() {
        let observed = [
            ModelArtifactManifestV1::potion_128m_native().unwrap(),
            ModelArtifactManifestV1::minilm_fastembed().unwrap(),
            ModelArtifactManifestV1::minilm_native_frankentorch().unwrap(),
            ModelArtifactManifestV1::multilingual_minilm_native_frankentorch().unwrap(),
            ModelArtifactManifestV1::snowflake_fastembed().unwrap(),
            ModelArtifactManifestV1::nomic_fastembed().unwrap(),
        ]
        .map(|manifest| {
            let frozen = manifest.freeze().unwrap();
            (frozen.manifest.execution.backend, frozen.fingerprint)
        });
        let expected = [
            (
                "model2vec-native".to_owned(),
                "860061ab2a8de3ad3a36a235ebf856eec6bb3d952840be655b0670595882d3cb".to_owned(),
            ),
            (
                "fastembed-onnx".to_owned(),
                "bad3145f51094f257ed565fa02a6c4c2de9e9294b2e154d6c6086693ac42cfd2".to_owned(),
            ),
            (
                "frankentorch-native-minilm".to_owned(),
                "726d6dde1d25946d1c7287a26f92518c158ee6d009b8a76f52748275f063e72c".to_owned(),
            ),
            (
                "frankentorch-native-multilingual-minilm".to_owned(),
                "59160d9e43d396d05b4139c99f9feb7922da14868587fca7e33d379821a41405".to_owned(),
            ),
            (
                "fastembed-onnx".to_owned(),
                "ea886582d1908a21790748d3ef9cabfa4992a30889b1736f21dae7b79b3d0463".to_owned(),
            ),
            (
                "fastembed-onnx".to_owned(),
                "b08612d52b1dea243aa4c9abcd65e156fc411813220a2280d1bf4ac45ef7027e".to_owned(),
            ),
        ];
        assert_eq!(observed, expected);
    }

    #[test]
    fn invalid_manifest_json_returns_clear_error() {
        let err = ModelManifest::from_json_str("{\"credential\":\"secret-token\",not-valid-json}")
            .unwrap_err();
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
        assert!(err.to_string().contains("manifest JSON"));
        assert!(!err.to_string().contains("secret-token"));
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
    fn downloadable_manifest_rejects_log_control_characters_redacted() {
        let mut manifest = ModelManifest::potion_128m();
        manifest.id = "model\nforged-log-line".to_owned();
        let error = manifest.validate().unwrap_err();
        assert!(error.to_string().contains("control characters"));
        assert!(!error.to_string().contains("forged-log-line"));

        let mut manifest = ModelManifest::potion_128m();
        manifest.files[0].name = "tokenizer.json\nforged-log-line".to_owned();
        let error = manifest.validate().unwrap_err();
        assert!(error.to_string().contains("control characters"));
        assert!(!error.to_string().contains("forged-log-line"));

        let mut manifest = ModelManifest::potion_128m();
        manifest.files[0].sha256 = "digest\nforged-log-line".to_owned();
        let error = manifest.validate().unwrap_err();
        assert!(error.to_string().contains("redacted-invalid-sha256"));
        assert!(!error.to_string().contains("forged-log-line"));

        let mut manifest = ModelManifest::potion_128m();
        manifest.files[0].url =
            Some("https://models.example.invalid/tokenizer.json?token=secret".to_owned());
        let error = manifest.validate().unwrap_err();
        assert!(error.to_string().contains("credential-free HTTPS"));
        assert!(!error.to_string().contains("secret"));
    }

    #[test]
    fn missing_required_manifest_field_is_rejected_with_redacted_input() {
        let err = ModelManifest::from_json_str(
            r#"{
                "id":"test-model",
                "repo":"acme/test-model",
                "revision":"0123456789abcdef0123456789abcdef01234567",
                "files":[]
            }"#,
        )
        .unwrap_err();

        assert!(matches!(
            err,
            SearchError::InvalidConfig {
                ref field,
                ref value,
                ..
            // ubs:ignore — this is a public fixed redaction sentinel, not a secret.
            } if field == "manifest_json" && value == "redacted-manifest-json"
        ));
        assert!(err.to_string().contains("malformed or unsupported"));
        assert!(!err.to_string().contains("acme/test-model"));
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

        let err = verify_file_sha256(&missing_path, "digest\nforged-log-line", 1).unwrap_err();
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
        assert!(err.to_string().contains("redacted-invalid-sha256"));
        assert!(!err.to_string().contains("forged-log-line"));

        let valid_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let err = verify_file_sha256(&missing_path, valid_hash, 1).unwrap_err();
        assert!(matches!(err, SearchError::ModelNotFound { .. }));
    }

    #[cfg(unix)]
    #[test]
    fn verify_file_sha256_rejects_symlinked_artifacts() {
        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("target.bin");
        let link = temp.path().join("model.bin");
        write_temp_file(&target, b"model-bytes");
        std::os::unix::fs::symlink(&target, &link).unwrap();

        let expected_hash = to_hex_lowercase(&Sha256::digest(b"model-bytes"));
        let error = verify_file_sha256(&link, &expected_hash, 11).unwrap_err();
        assert!(matches!(&error, SearchError::ModelLoadFailed { .. }));
        assert!(error.to_string().contains("symlink"));
        assert!(
            capture_file_verification_state(&link).is_none(),
            "verification cache metadata must not follow symlinks"
        );
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
        lifecycle.mark_staged_verified().unwrap();
        assert_eq!(lifecycle.state(), &ModelState::StagedVerified);
        lifecycle.mark_acquired_needs_reindex().unwrap();

        assert_eq!(lifecycle.state(), &ModelState::AcquiredNeedsReindex);
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
    fn verify_dir_rejects_traversal_file_names_without_needing_validate_call() {
        let temp = tempfile::tempdir().expect("tempdir");
        let manifest = ModelManifest {
            id: "test".to_owned(),
            version: "test-v1".to_owned(),
            display_name: None,
            description: None,
            repo: "owner/repo".to_owned(),
            revision: "abcdef1".to_owned(),
            files: vec![ModelFile {
                name: "../escape.bin".to_owned(),
                sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    .to_owned(),
                size: 0,
                url: None,
            }],
            license: "MIT".to_owned(),
            dimension: None,
            tier: None,
            download_size_bytes: 0,
        };

        let err = manifest
            .verify_dir(temp.path())
            .expect_err("must reject traversal");
        assert!(matches!(
            err,
            SearchError::InvalidConfig { ref field, .. } if field == "files[].name"
        ));
        assert!(err.to_string().contains("path traversal"));
    }

    #[test]
    fn model_file_names_reject_noncanonical_portable_encodings() {
        for name in [
            "./model.safetensors",
            "C:/model.safetensors",
            "model.safetensors:alternate-stream",
            "onnx//model.onnx",
            "onnx\\model.onnx",
            "tokenizer.json/",
            "onnx/model?.onnx",
            "weights*/model.onnx",
            "NUL",
            "com1.bin",
            "COM¹.txt",
            "LPT9.config",
            "clock$",
            "CONOUT$/model.bin",
            "model.onnx.",
            " model.onnx",
            "model.onnx ",
        ] {
            let error = validate_model_file_name(name).unwrap_err();
            assert!(error.to_string().contains("canonical portable"));
            assert!(!error.to_string().contains(name));
        }
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
    fn file_with_zero_size_and_valid_hash_accepted() {
        let json = r#"{
            "id": "m", "repo": "r", "revision": "v", "license": "MIT",
            "files": [{"name": "f.bin", "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "size": 0}]
        }"#;
        let manifest = ModelManifest::from_json_str(json).unwrap();
        assert_eq!(manifest.files[0].size, 0);
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
    fn has_pinned_revision_rejects_every_non_digest_alias() {
        for alias in &[
            "main",
            "latest",
            "HEAD",
            "legacy-default-branch",
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
            ModelState::StagedVerified,
            ModelState::AcquiredNeedsReindex,
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
        let latest_revision = "a".repeat(40);
        let installed_revision = "b".repeat(40);
        let m = ModelManifest {
            revision: latest_revision,
            ..ModelManifest::potion_128m()
        };
        let state = m.detect_update_state(&installed_revision).unwrap();
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
            revision: "a".repeat(40),
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
            revision: "a".repeat(40),
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

    fn publication_failure_fixture() -> (tempfile::TempDir, ModelManifest, PathBuf, PathBuf) {
        let temp = tempfile::tempdir().unwrap();
        let staged = temp.path().join("staged");
        let destination = temp.path().join("final");
        fs::create_dir_all(&staged).unwrap();
        fs::create_dir_all(&destination).unwrap();
        write_temp_file(&staged.join("model.bin"), b"new model");
        write_temp_file(&destination.join("prior.bin"), b"prior model");
        let manifest = ModelManifest {
            id: "publication-fault-fixture".to_owned(),
            version: "test-v1".to_owned(),
            display_name: None,
            description: None,
            repo: "owner/repo".to_owned(),
            revision: "a".repeat(40),
            files: vec![ModelFile {
                name: "model.bin".to_owned(),
                sha256: to_hex_lowercase(&Sha256::digest(b"new model")),
                size: 9,
                url: None,
            }],
            license: "MIT".to_owned(),
            dimension: None,
            tier: None,
            download_size_bytes: 9,
        };
        (temp, manifest, staged, destination)
    }

    fn sibling_names_with_prefix(parent: &Path, prefix: &str) -> Vec<String> {
        fs::read_dir(parent)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name.starts_with(prefix))
            .collect()
    }

    #[test]
    fn interrupted_installing_parent_fsync_preserves_prior_generation() {
        let (temp, manifest, staged, destination) = publication_failure_fixture();
        let _fault = PublicationFaultGuard::install(PublicationBoundary::InstallingParentSync);

        let error = manifest
            .promote_verified_installation(&staged, &destination)
            .unwrap_err();

        assert!(matches!(error, SearchError::Io(_)));
        assert_eq!(
            fs::read(destination.join("prior.bin")).unwrap(),
            b"prior model"
        );
        assert_eq!(
            sibling_names_with_prefix(temp.path(), ".final.installing.").len(),
            1
        );
    }

    #[test]
    fn interrupted_backup_parent_fsync_rolls_prior_generation_back() {
        let (temp, manifest, staged, destination) = publication_failure_fixture();
        let _fault = PublicationFaultGuard::install(PublicationBoundary::BackupParentSync);

        let error = manifest
            .promote_verified_installation(&staged, &destination)
            .unwrap_err();

        assert!(matches!(error, SearchError::Io(_)));
        assert_eq!(
            fs::read(destination.join("prior.bin")).unwrap(),
            b"prior model"
        );
        assert_eq!(
            sibling_names_with_prefix(temp.path(), ".final.installing.").len(),
            1
        );
    }

    #[test]
    fn interrupted_publish_rename_rolls_prior_generation_back() {
        let (temp, manifest, staged, destination) = publication_failure_fixture();
        let _fault = PublicationFaultGuard::install(PublicationBoundary::PublishRename);

        let error = manifest
            .promote_verified_installation(&staged, &destination)
            .unwrap_err();

        assert!(matches!(error, SearchError::Io(_)));
        assert_eq!(
            fs::read(destination.join("prior.bin")).unwrap(),
            b"prior model"
        );
        assert_eq!(
            sibling_names_with_prefix(temp.path(), ".final.installing.").len(),
            1
        );
    }

    #[test]
    fn interrupted_published_parent_fsync_preserves_prior_backup() {
        let (temp, manifest, staged, destination) = publication_failure_fixture();
        let _fault = PublicationFaultGuard::install(PublicationBoundary::PublishedParentSync);

        let error = manifest
            .promote_verified_installation(&staged, &destination)
            .unwrap_err();

        assert!(matches!(error, SearchError::Io(_)));
        assert_eq!(
            fs::read(destination.join("model.bin")).unwrap(),
            b"new model"
        );
        let backups = sibling_names_with_prefix(temp.path(), "final.backup.");
        assert_eq!(backups.len(), 1);
        assert_eq!(
            fs::read(temp.path().join(&backups[0]).join("prior.bin")).unwrap(),
            b"prior model"
        );
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
        assert_eq!(m.license, "MIT");
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
    fn multilingual_minilm_is_production_ready_but_not_in_default_catalog() {
        let manifest = ModelManifest::multilingual_minilm_l12_v2();
        manifest.validate().unwrap();
        assert!(manifest.is_production_ready());
        assert_eq!(manifest.dimension, Some(384));
        assert_eq!(manifest.tier, Some(ModelTier::Quality));
        assert_eq!(manifest.total_size_bytes(), 479_724_528);
        assert!(
            !ModelManifest::builtin_catalog()
                .models
                .iter()
                .any(|candidate| candidate.id == manifest.id)
        );
        let opt_in = ModelManifest::opt_in_catalog();
        assert_eq!(opt_in.models, vec![manifest]);
        opt_in.validate().unwrap();
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
    fn builtin_catalog_contains_all_models() {
        let catalog = ModelManifest::builtin_catalog();
        assert_eq!(catalog.schema_version, MANIFEST_SCHEMA_VERSION);
        assert_eq!(catalog.models.len(), 7);

        let ids: Vec<&str> = catalog.models.iter().map(|m| m.id.as_str()).collect();
        assert!(ids.contains(&"potion-multilingual-128m"));
        assert!(ids.contains(&"all-minilm-l6-v2"));
        assert!(ids.contains(&"ms-marco-minilm-l-6-v2"));
        assert!(ids.contains(&"snowflake-arctic-embed-s"));
        assert!(ids.contains(&"nomic-embed-text-v1.5"));
        assert!(ids.contains(&"jina-reranker-v1-turbo-en"));
        assert!(ids.contains(&"flashrank-nano"));

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
            ModelManifest::snowflake_arctic_s(),
            ModelManifest::nomic_embed(),
            ModelManifest::jina_reranker_turbo(),
            ModelManifest::flashrank_nano(),
        ];

        for manifest in manifests {
            assert!(!manifest.version.is_empty());
            assert!(manifest.description.is_some());
            // Some built-in manifests intentionally use placeholder metadata
            // (sha256=PLACEHOLDER_VERIFY_AFTER_DOWNLOAD, size=0) so that file
            // sizes are confirmed at runtime during the first download rather
            // than baked into source code. For those, we only assert the
            // size-sum invariant, not a non-zero total.
            let all_placeholder = manifest
                .files
                .iter()
                .all(|file| file.sha256 == PLACEHOLDER_VERIFY_AFTER_DOWNLOAD);
            if !all_placeholder {
                assert!(manifest.download_size_bytes > 0);
            }
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
            revision: "a".repeat(40),
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

        verify_dir_and_record(&manifest, tmp.path()).unwrap();
        let json = std::fs::read_to_string(tmp.path().join(VERIFIED_MARKER_FILE)).unwrap();
        let restored: VerificationMarker = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.manifest_id, "test-model");
        assert_eq!(restored.schema_version, VERIFICATION_MARKER_SCHEMA_VERSION);
        assert_eq!(restored.manifest_fingerprint.len(), 64);
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
        verify_dir_and_record(&manifest, tmp.path()).unwrap();
        assert!(is_verification_cached(&manifest, tmp.path()));
    }

    #[test]
    fn verification_cache_miss_when_manifest_id_changes() {
        let tmp = tempfile::tempdir().unwrap();
        let content = b"model data";
        let manifest = make_test_manifest("model.bin", content);
        write_temp_file(&tmp.path().join("model.bin"), content);

        verify_dir_and_record(&manifest, tmp.path()).unwrap();
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
        verify_dir_and_record(&manifest, tmp.path()).unwrap();
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
                created_unix_nanos: None,
                platform_file_id: None,
                platform_change_stamp: None,
            },
        );
        let tampered = serde_json::to_string_pretty(&marker).unwrap();
        std::fs::write(&marker_path, tampered).unwrap();

        assert!(!is_verification_cached(&manifest, tmp.path()));
    }

    #[test]
    fn verify_dir_cached_is_observational_on_uncached_success() {
        let tmp = tempfile::tempdir().unwrap();
        let content = b"model data for verify";
        let manifest = make_test_manifest("model.bin", content);
        write_temp_file(&tmp.path().join("model.bin"), content);

        assert!(!tmp.path().join(VERIFIED_MARKER_FILE).exists());
        verify_dir_cached(&manifest, tmp.path()).unwrap();
        assert!(
            !tmp.path().join(VERIFIED_MARKER_FILE).exists(),
            "observational verification must not mint a receipt"
        );
    }

    #[test]
    fn verify_dir_cached_skips_rehash_on_cached_hit() {
        let tmp = tempfile::tempdir().unwrap();
        let content = b"model data cached";
        let manifest = make_test_manifest("model.bin", content);
        write_temp_file(&tmp.path().join("model.bin"), content);

        // The authority-bearing operation performs the full hash and mints the receipt.
        verify_dir_and_record(&manifest, tmp.path()).unwrap();

        // The consumer operation may then admit the unchanged receipt.
        verify_dir_cached(&manifest, tmp.path()).unwrap();
    }

    /// Build a two-file download manifest plus the native manifest derived
    /// from it, exactly the way `potion_128m_native` derives from
    /// `ModelManifest::potion_128m`, but over small fixture bytes.
    fn native_fixture(
        tokenizer: &[u8],
        weights: &[u8],
    ) -> (ModelManifest, ModelArtifactManifestV1) {
        use sha2::{Digest, Sha256};
        let sha = |bytes: &[u8]| {
            let mut hasher = Sha256::new();
            hasher.update(bytes);
            to_hex_lowercase(&hasher.finalize())
        };
        let mut manifest = make_test_manifest("tokenizer.json", tokenizer);
        manifest.files.push(ModelFile {
            name: "model.safetensors".to_owned(),
            sha256: sha(weights),
            size: u64::try_from(weights.len()).unwrap(),
            url: None,
        });
        manifest.dimension = Some(4);
        manifest.download_size_bytes = u64::try_from(tokenizer.len() + weights.len()).unwrap();
        let execution = ModelExecutionContractV1 {
            backend: "test-native".to_owned(),
            implementation_revision: "test-impl-v1".to_owned(),
            protocol_revision: "test-proto-v1".to_owned(),
            numeric_profile: "f32-test-v1".to_owned(),
            weights_format: "safetensors-f32-matrix-v1".to_owned(),
            tokenizer_family: "huggingface-tokenizers-json-v1".to_owned(),
            model_preprocessing: MODEL2VEC_PREPROCESSING_V1.to_owned(),
            sequence_policy: MODEL2VEC_SEQUENCE_POLICY_V1.to_owned(),
            pooling: MODEL2VEC_POOLING_V1.to_owned(),
            output_normalization: MODEL2VEC_OUTPUT_NORMALIZATION_V1.to_owned(),
            query_instruction: String::new(),
            document_instruction: String::new(),
            input_contract: default_plain_text_input_contract(),
            golden_vectors: GoldenVectorCertificateV1 {
                corpus_sha256: conformance_corpus_fingerprint().unwrap(),
                vectors_sha256: "f".repeat(64),
                vector_count: 4,
                dimension: 4,
            },
        };
        let native =
            ModelArtifactManifestV1::from_download_manifest(&manifest, "test-provider", execution)
                .expect("fixture native manifest must validate");
        (manifest, native)
    }

    #[test]
    fn native_verify_dir_cached_falls_back_to_full_hash_without_a_receipt() {
        let tmp = tempfile::tempdir().unwrap();
        let (manifest, native) = native_fixture(b"{\"tokenizer\":1}", b"weights-for-test");
        write_temp_file(&tmp.path().join("tokenizer.json"), b"{\"tokenizer\":1}");
        write_temp_file(&tmp.path().join("model.safetensors"), b"weights-for-test");

        // No receipt yet: the cached entry point must behave exactly like the
        // full pass, succeed on matching bytes, and mint nothing.
        native.verify_dir_cached(&manifest, tmp.path()).unwrap();
        assert!(!tmp.path().join(VERIFIED_MARKER_FILE).exists());

        // Drifted bytes with no receipt are still caught by the full pass.
        write_temp_file(&tmp.path().join("model.safetensors"), b"weights-for-tesT");
        assert!(
            native.verify_dir_cached(&manifest, tmp.path()).is_err(),
            "without a receipt the cached path must still hash and reject drift"
        );
    }

    #[cfg(unix)]
    #[test]
    fn native_verify_dir_cached_skips_the_hash_pass_on_a_valid_receipt() {
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::tempdir().unwrap();
        let (manifest, native) = native_fixture(b"{\"tokenizer\":2}", b"weights-cached");
        write_temp_file(&tmp.path().join("tokenizer.json"), b"{\"tokenizer\":2}");
        let weights_path = tmp.path().join("model.safetensors");
        write_temp_file(&weights_path, b"weights-cached");

        // Only the download-manifest authority mints the receipt.
        verify_dir_and_record(&manifest, tmp.path()).unwrap();

        // Make the weights unreadable without changing size, mtime, or
        // identity. A full hash pass must now fail; the receipt path only
        // stats the file, so a cached hit is the one way this can succeed.
        let original = std::fs::metadata(&weights_path).unwrap().permissions();
        std::fs::set_permissions(&weights_path, std::fs::Permissions::from_mode(0o000)).unwrap();
        let cached = native.verify_dir_cached(&manifest, tmp.path());
        let full = native.verify_dir(tmp.path());
        std::fs::set_permissions(&weights_path, original).unwrap();

        assert!(
            full.is_err(),
            "control: the full pass must not be able to read the weights"
        );
        cached.expect("a valid receipt must let the native verify skip the hash pass");

        // A native manifest that no longer matches the download manifest must
        // not borrow its receipt: with the file unreadable that means Err.
        let mut drifted = manifest.clone();
        drifted.files[1].sha256 = "0".repeat(64);
        std::fs::set_permissions(&weights_path, std::fs::Permissions::from_mode(0o000)).unwrap();
        let mismatched = native.verify_dir_cached(&drifted, tmp.path());
        std::fs::set_permissions(
            &weights_path,
            std::fs::metadata(tmp.path().join("tokenizer.json"))
                .unwrap()
                .permissions(),
        )
        .unwrap();
        assert!(
            mismatched.is_err(),
            "a receipt for a different artifact set must never satisfy the native manifest"
        );
    }

    #[test]
    fn verification_cache_miss_when_same_id_revision_changes() {
        let tmp = tempfile::tempdir().unwrap();
        let content = b"revision-bound model data";
        let manifest = make_test_manifest("model.bin", content);
        write_temp_file(&tmp.path().join("model.bin"), content);

        verify_dir_and_record(&manifest, tmp.path()).unwrap();
        assert!(is_verification_cached(&manifest, tmp.path()));

        let mut changed = manifest;
        changed.revision = "b".repeat(40);
        assert!(
            !is_verification_cached(&changed, tmp.path()),
            "same-ID revision evolution must invalidate the old receipt"
        );
    }

    #[test]
    fn verification_cache_miss_when_same_id_sha_changes() {
        let tmp = tempfile::tempdir().unwrap();
        let content = b"checksum-bound model data";
        let manifest = make_test_manifest("model.bin", content);
        write_temp_file(&tmp.path().join("model.bin"), content);

        verify_dir_and_record(&manifest, tmp.path()).unwrap();
        assert!(is_verification_cached(&manifest, tmp.path()));

        let mut changed = manifest;
        changed.files[0].sha256 = "b".repeat(64);
        assert!(
            !is_verification_cached(&changed, tmp.path()),
            "same-ID checksum evolution must invalidate the old receipt"
        );
    }

    #[test]
    fn verification_cache_rejects_legacy_marker_without_manifest_fingerprint() {
        let tmp = tempfile::tempdir().unwrap();
        let content = b"legacy marker model data";
        let manifest = make_test_manifest("model.bin", content);
        let path = tmp.path().join("model.bin");
        write_temp_file(&path, content);
        let state = capture_file_verification_state(&path).unwrap();
        let legacy = serde_json::json!({
            "manifest_id": manifest.id,
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "verified_at": 1,
            "file_states": {
                "model.bin": {
                    "size_bytes": state.size_bytes,
                    "modified_unix_nanos": state.modified_unix_nanos
                }
            }
        });
        std::fs::write(
            tmp.path().join(VERIFIED_MARKER_FILE),
            serde_json::to_vec_pretty(&legacy).unwrap(),
        )
        .unwrap();

        assert!(
            !is_verification_cached(&manifest, tmp.path()),
            "pre-fingerprint markers must fail closed"
        );
    }

    #[test]
    fn verification_cache_rejects_partial_current_marker() {
        let tmp = tempfile::tempdir().unwrap();
        let content = b"partial marker model data";
        let manifest = make_test_manifest("model.bin", content);
        write_temp_file(&tmp.path().join("model.bin"), content);

        verify_dir_and_record(&manifest, tmp.path()).unwrap();
        let marker_path = tmp.path().join(VERIFIED_MARKER_FILE);
        let raw = std::fs::read_to_string(&marker_path).unwrap();
        let mut marker: VerificationMarker = serde_json::from_str(&raw).unwrap();
        marker.file_states.clear();
        std::fs::write(&marker_path, serde_json::to_vec_pretty(&marker).unwrap()).unwrap();

        assert!(
            !is_verification_cached(&manifest, tmp.path()),
            "a current-schema receipt missing any manifest file state must fail closed"
        );
    }

    #[test]
    fn verification_cache_rejects_unknown_marker_fields() {
        let tmp = tempfile::tempdir().unwrap();
        let content = b"unknown marker field model data";
        let manifest = make_test_manifest("model.bin", content);
        write_temp_file(&tmp.path().join("model.bin"), content);

        verify_dir_and_record(&manifest, tmp.path()).unwrap();
        let marker_path = tmp.path().join(VERIFIED_MARKER_FILE);
        let raw = std::fs::read_to_string(&marker_path).unwrap();
        let mut marker: serde_json::Value = serde_json::from_str(&raw).unwrap();
        marker.as_object_mut().unwrap().insert(
            "unrecognized_trust_claim".to_owned(),
            serde_json::json!(true),
        );
        std::fs::write(&marker_path, serde_json::to_vec_pretty(&marker).unwrap()).unwrap();

        assert!(
            !is_verification_cached(&manifest, tmp.path()),
            "receipts with unknown trust claims must fail closed"
        );
    }

    #[test]
    fn full_verify_and_record_rejects_corrupt_bytes_without_minting() {
        let tmp = tempfile::tempdir().unwrap();
        let manifest = make_test_manifest("model.bin", b"registered bytes");
        write_temp_file(&tmp.path().join("model.bin"), b"corrupted bytes!");

        let error = verify_dir_and_record(&manifest, tmp.path()).unwrap_err();
        assert!(matches!(error, SearchError::HashMismatch { .. }));
        assert!(
            !tmp.path().join(VERIFIED_MARKER_FILE).exists(),
            "failed full verification must never mint a receipt"
        );
    }

    #[test]
    fn full_verify_and_record_rejects_mutation_after_hash_before_receipt() {
        let tmp = tempfile::tempdir().unwrap();
        let content = b"stable registered bytes";
        let manifest = make_test_manifest("model.bin", content);
        let model_path = tmp.path().join("model.bin");
        write_temp_file(&model_path, content);

        TEST_AFTER_FULL_HASH_HOOK.with(|slot| {
            assert!(slot.borrow().is_none());
            slot.replace(Some(Box::new(move || {
                std::fs::write(model_path, b"changed after the full hash completed").unwrap();
            })));
        });

        let error = verify_dir_and_record(&manifest, tmp.path()).unwrap_err();
        assert!(matches!(error, SearchError::HashMismatch { .. }));
        assert!(
            !tmp.path().join(VERIFIED_MARKER_FILE).exists(),
            "a hash-vs-metadata race must fail before receipt publication"
        );
    }

    #[test]
    fn verified_promotion_publishes_model_and_receipt_together() {
        let parent = tempfile::tempdir().unwrap();
        let staged = tempfile::tempdir_in(parent.path()).unwrap();
        let content = b"atomically promoted model bytes";
        let manifest = make_test_manifest("model.bin", content);
        write_temp_file(&staged.path().join("model.bin"), content);
        let destination = parent.path().join("published-model");

        manifest
            .promote_verified_installation(staged.path(), &destination)
            .unwrap();

        assert_eq!(
            std::fs::read(destination.join("model.bin")).unwrap(),
            content
        );
        assert!(destination.join(VERIFIED_MARKER_FILE).is_file());
        assert!(is_verification_cached(&manifest, &destination));
    }

    #[test]
    fn partial_receipt_never_prevents_full_rehash() {
        let tmp = tempfile::tempdir().unwrap();
        let content = b"partial receipt bytes";
        let manifest = make_test_manifest("model.bin", content);
        let model_path = tmp.path().join("model.bin");
        write_temp_file(&model_path, content);
        verify_dir_and_record(&manifest, tmp.path()).unwrap();

        let marker_path = tmp.path().join(VERIFIED_MARKER_FILE);
        let raw = std::fs::read_to_string(&marker_path).unwrap();
        let mut marker: VerificationMarker = serde_json::from_str(&raw).unwrap();
        marker.file_states.clear();
        std::fs::write(&marker_path, serde_json::to_vec_pretty(&marker).unwrap()).unwrap();
        std::fs::write(model_path, b"corrupt partial bytes").unwrap();

        assert!(
            matches!(
                verify_dir_cached(&manifest, tmp.path()),
                Err(SearchError::HashMismatch { .. })
            ),
            "a partial receipt must fall through to full SHA verification"
        );
    }

    #[test]
    fn built_in_verification_manifest_fingerprints_are_golden() {
        let potion = ModelManifest::potion_128m()
            .freeze_verification_manifest()
            .unwrap()
            .fingerprint;
        let minilm = ModelManifest::minilm_v2()
            .freeze_verification_manifest()
            .unwrap()
            .fingerprint;

        assert_eq!(
            potion,
            "32b58266fc633cf0e95c05d80bcb9c8f943786bb9e83aed6676eba8311b779e9"
        );
        assert_eq!(
            minilm,
            "5a563a081f9d3febe93302339766bb9ef314e170c8579a33c500f18fdcbe3f8e"
        );
    }

    #[test]
    fn verify_dir_cached_rejects_non_production_manifest() {
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
        let error = verify_dir_cached(&manifest, tmp.path()).unwrap_err();
        assert!(
            matches!(error, SearchError::InvalidConfig { .. }),
            "placeholder manifests must never receive cached admission: {error}"
        );
        assert!(!tmp.path().join(VERIFIED_MARKER_FILE).exists());
    }
}
