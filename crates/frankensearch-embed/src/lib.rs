//! Embedder implementations for the frankensearch hybrid search library.
//!
//! Provides three tiers of text embedding:
//! - **Hash** (`hash` feature, default): FNV-1a hash embedder, zero dependencies, always available.
//! - **`Model2Vec`** (`model2vec` feature): potion-128M static embedder, fast tier (~0.57ms).
//! - **`FastEmbed`** (`fastembed` feature): MiniLM-L6-v2 ONNX embedder, quality tier (~128ms).
//!
//! The `EmbedderStack` auto-detection probes for available models and configures
//! the best fast+quality pair automatically.

pub mod auto_detect;
pub mod batch_coalescer;
#[cfg(feature = "bundled-default-models")]
pub mod bundled_default_models;
pub mod cached_embedder;
pub mod model_cache;
pub mod model_manifest;
pub mod model_registry;
pub use auto_detect::{
    DimReduceEmbedder, EmbedderStack, ModelAvailabilityDiagnostic, ModelStatus, TwoTierAvailability,
};
pub use batch_coalescer::{
    BatchCoalescer, CoalescedBatch, CoalescerConfig, CoalescerMetrics, Priority,
};
#[cfg(feature = "bundled-default-models")]
pub use bundled_default_models::{EmbeddedModelInstallSummary, ensure_default_semantic_models};
pub use cached_embedder::{CacheStats, CachedEmbedder};
pub use model_cache::{
    ENV_DATA_DIR, ENV_MODEL_DIR, KnownModel, MODEL_CACHE_LAYOUT_VERSION, ModelCacheLayout,
    ModelDirEntry, ensure_cache_layout, ensure_default_cache, is_model_installed, known_models,
    model_file_path, resolve_cache_root,
};
pub use model_manifest::{
    ConsentSource, DOWNLOAD_CONSENT_ENV, DownloadConsent, MANIFEST_SCHEMA_VERSION, ModelFile,
    ModelLifecycle, ModelManifest, ModelManifestCatalog, ModelState, ModelTier,
    PLACEHOLDER_VERIFY_AFTER_DOWNLOAD, VerificationMarker, is_verification_cached,
    resolve_download_consent, verify_dir_cached, verify_file_sha256, write_verification_marker,
};
pub use model_registry::{
    BAKEOFF_CUTOFF_DATE, EmbedderRegistry, RegisteredEmbedder, RegisteredReranker,
    registered_embedders, registered_rerankers,
};

#[cfg(feature = "hash")]
pub mod hash_embedder;

#[cfg(feature = "hash")]
pub use hash_embedder::{HashAlgorithm, HashEmbedder};

#[cfg(feature = "model2vec")]
pub mod model2vec_embedder;

#[cfg(feature = "model2vec")]
pub use model2vec_embedder::{Model2VecEmbedder, find_model_dir};

#[cfg(feature = "fastembed")]
pub mod fastembed_embedder;

#[cfg(feature = "fastembed")]
pub use fastembed_embedder::FastEmbedEmbedder;

#[cfg(feature = "download")]
pub mod model_download;

#[cfg(feature = "download")]
pub use model_download::{DownloadConfig, DownloadProgress, ModelDownloader};
