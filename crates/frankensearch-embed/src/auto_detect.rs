//! Embedder auto-detection and fallback stack assembly.

#[cfg(any(feature = "model2vec", feature = "fastembed"))]
use std::collections::BTreeSet;
use std::fmt::{self, Write as _};
#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
use std::io::{self, IsTerminal, Write};
#[cfg(not(any(feature = "model2vec", feature = "fastembed")))]
use std::path::Path;
#[cfg(any(feature = "model2vec", feature = "fastembed"))]
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
use std::sync::atomic::{AtomicU8, Ordering};
#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
use std::time::Instant;

use asupersync::Cx;
#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
use asupersync::sync::OnceCell;
#[cfg(not(any(feature = "model2vec", feature = "fastembed")))]
use tracing::info;
#[cfg(any(feature = "model2vec", feature = "fastembed"))]
use tracing::{info, warn};

use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::traits::{Embedder, SearchFuture};
#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
use frankensearch_core::traits::{ModelCategory, ModelTier};

#[cfg(feature = "fastembed")]
use crate::fastembed_embedder::{
    DEFAULT_DIMENSION as MINILM_DIMENSION, DEFAULT_HF_ID as MINILM_HF_ID,
    DEFAULT_MODEL_NAME as MINILM_MODEL_NAME, FastEmbedEmbedder,
    find_model_dir_with_hf_id as find_fastembed_model_dir,
};
#[cfg(feature = "hash")]
use crate::hash_embedder::HashEmbedder;
#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
use crate::model_download::{DownloadProgress, ModelDownloader};
#[cfg(any(feature = "model2vec", feature = "fastembed"))]
use crate::model_manifest::ModelManifest;
#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
use crate::model_manifest::{
    ConsentSource, DownloadConsent, ModelLifecycle, resolve_download_consent,
};
#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
use crate::model_registry::ensure_model_storage_layout;
#[cfg(feature = "model2vec")]
use crate::model2vec_embedder::{
    Model2VecEmbedder, find_model_dir_with_hf_id as find_model2vec_model_dir,
};

#[cfg(feature = "model2vec")]
const POTION_MODEL_NAME: &str = "potion-multilingual-128M";
#[cfg(feature = "model2vec")]
const POTION_HF_ID: &str = "minishlab/potion-multilingual-128M";
#[cfg(feature = "model2vec")]
const POTION_DIMENSION: usize = 256;
#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
const OFFLINE_ENV: &str = "FRANKENSEARCH_OFFLINE";
#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
const PROGRESS_BAR_WIDTH: usize = 30;

/// Availability classification for two-tier search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TwoTierAvailability {
    /// Fast + quality embedders available.
    Full,
    /// Semantic fast embedder available, no quality tier.
    FastOnly,
    /// Hash-only fallback path.
    HashOnly,
}

impl TwoTierAvailability {
    /// Whether this availability level represents a degraded state.
    #[must_use]
    pub const fn is_degraded(self) -> bool {
        matches!(self, Self::FastOnly | Self::HashOnly)
    }

    /// Human-readable summary of what is missing at this availability level.
    #[must_use]
    pub const fn degradation_summary(self) -> Option<&'static str> {
        match self {
            Self::Full => None,
            Self::FastOnly => Some(
                "Quality model unavailable: search will return fast-tier results only (no refinement phase).",
            ),
            Self::HashOnly => Some(
                "No semantic models available: search uses hash-based embedding only (reduced relevance).",
            ),
        }
    }
}

impl fmt::Display for TwoTierAvailability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Full => write!(f, "full (fast + quality)"),
            Self::FastOnly => write!(f, "degraded (fast-only, no quality refinement)"),
            Self::HashOnly => write!(f, "minimal (hash-only, no semantic search)"),
        }
    }
}

/// Diagnostic information about model availability for user-facing messages.
///
/// Provides actionable guidance when models are missing, including cache paths,
/// download URLs, and environment variable hints.
#[derive(Debug, Clone)]
pub struct ModelAvailabilityDiagnostic {
    /// Current availability classification.
    pub availability: TwoTierAvailability,
    /// Resolved model cache directory path.
    pub cache_dir: std::path::PathBuf,
    /// Whether the system is in offline mode.
    pub offline: bool,
    /// Fast-tier model status.
    pub fast_status: ModelStatus,
    /// Quality-tier model status.
    pub quality_status: ModelStatus,
    /// Suggestions for the user to resolve degraded state.
    pub suggestions: Vec<String>,
}

/// Status of an individual model tier.
#[derive(Debug, Clone)]
pub enum ModelStatus {
    /// Model is loaded and ready.
    Ready {
        /// Model identifier.
        id: String,
    },
    /// Model files not found locally.
    NotFound {
        /// Model name that was searched for.
        model_name: String,
        /// `HuggingFace` repository URL for manual download.
        hf_repo_url: String,
        /// Paths that were searched.
        searched_paths: Vec<std::path::PathBuf>,
    },
    /// Download was blocked by policy (offline mode or consent denied).
    DownloadBlocked {
        /// Model name.
        model_name: String,
        /// Reason download was blocked.
        reason: String,
    },
    /// Feature not compiled in.
    FeatureDisabled {
        /// The feature flag that would enable this tier.
        feature_flag: String,
    },
    /// Model uses hash fallback (always available).
    HashFallback,
}

impl fmt::Display for ModelAvailabilityDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Model availability: {}", self.availability)?;
        writeln!(f, "Cache directory: {}", self.cache_dir.display())?;
        if self.offline {
            writeln!(f, "Mode: OFFLINE (FRANKENSEARCH_OFFLINE=1)")?;
        }
        writeln!(f)?;
        writeln!(f, "Fast tier:    {}", self.fast_status)?;
        writeln!(f, "Quality tier: {}", self.quality_status)?;
        if !self.suggestions.is_empty() {
            writeln!(f)?;
            writeln!(f, "To resolve:")?;
            for suggestion in &self.suggestions {
                writeln!(f, "  - {suggestion}")?;
            }
        }
        Ok(())
    }
}

impl fmt::Display for ModelStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ready { id } => write!(f, "ready ({id})"),
            Self::NotFound {
                model_name,
                hf_repo_url,
                ..
            } => {
                write!(f, "NOT FOUND ({model_name}) — download from {hf_repo_url}")
            }
            Self::DownloadBlocked { model_name, reason } => {
                write!(f, "BLOCKED ({model_name}): {reason}")
            }
            Self::FeatureDisabled { feature_flag } => {
                write!(f, "DISABLED (compile with --features {feature_flag})")
            }
            Self::HashFallback => write!(f, "hash fallback (no semantic model)"),
        }
    }
}

/// Resolved fast/quality embedder stack for progressive search.
#[derive(Clone)]
pub struct EmbedderStack {
    fast: Arc<dyn Embedder>,
    quality: Option<Arc<dyn Embedder>>,
    availability: TwoTierAvailability,
}

impl fmt::Debug for EmbedderStack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EmbedderStack")
            .field("availability", &self.availability)
            .field("fast_id", &self.fast.id())
            .field("fast_dim", &self.fast.dimension())
            .field(
                "quality_id",
                &self.quality.as_ref().map(|embedder| embedder.id()),
            )
            .finish()
    }
}

impl EmbedderStack {
    /// Build from explicit parts.
    #[must_use]
    pub fn from_parts(fast: Arc<dyn Embedder>, quality: Option<Arc<dyn Embedder>>) -> Self {
        let availability = if quality.is_some() {
            TwoTierAvailability::Full
        } else if fast.is_semantic() {
            TwoTierAvailability::FastOnly
        } else {
            TwoTierAvailability::HashOnly
        };
        Self {
            fast,
            quality,
            availability,
        }
    }

    /// Auto-detect best available embedders from default search paths.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::EmbedderUnavailable` when no usable fast embedder is available.
    pub fn auto_detect() -> SearchResult<Self> {
        Self::auto_detect_with(None)
    }

    /// Auto-detect embedders with an optional explicit model root override.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::EmbedderUnavailable` when no usable fast embedder is available.
    pub fn auto_detect_with(model_root: Option<&Path>) -> SearchResult<Self> {
        #[cfg(all(
            feature = "download",
            any(feature = "model2vec", feature = "fastembed")
        ))]
        {
            Self::auto_detect_with_policy(model_root, download_policy_from_environment())
        }
        #[cfg(not(all(
            feature = "download",
            any(feature = "model2vec", feature = "fastembed")
        )))]
        {
            Self::auto_detect_with_policy(model_root)
        }
    }

    #[cfg(all(
        feature = "download",
        any(feature = "model2vec", feature = "fastembed")
    ))]
    fn auto_detect_with_policy(
        model_root: Option<&Path>,
        policy: DownloadPolicy,
    ) -> SearchResult<Self> {
        let quality = detect_quality_embedder(model_root)
            .or_else(|| maybe_lazy_quality_embedder(model_root, policy));
        let fast = detect_fast_embedder(model_root)
            .or_else(|| maybe_lazy_fast_embedder(model_root, policy))
            .or_else(hash_fallback_embedder)
            .ok_or_else(|| SearchError::EmbedderUnavailable {
                model: "fast-tier".to_owned(),
                reason: "no model2vec/hash embedder available in this build".to_owned(),
            })?;

        let stack = Self::from_parts(fast, quality);
        info!(
            availability = ?stack.availability,
            fast = stack.fast.id(),
            quality = stack.quality.as_ref().map(|embedder| embedder.id()),
            "embedder stack ready"
        );
        Ok(stack)
    }

    #[cfg(not(all(
        feature = "download",
        any(feature = "model2vec", feature = "fastembed")
    )))]
    fn auto_detect_with_policy(model_root: Option<&Path>) -> SearchResult<Self> {
        let quality = detect_quality_embedder(model_root);
        let fast = detect_fast_embedder(model_root)
            .or_else(hash_fallback_embedder)
            .ok_or_else(|| SearchError::EmbedderUnavailable {
                model: "fast-tier".to_owned(),
                reason: "no model2vec/hash embedder available in this build".to_owned(),
            })?;

        let stack = Self::from_parts(fast, quality);
        info!(
            availability = ?stack.availability,
            fast = stack.fast.id(),
            quality = stack.quality.as_ref().map(|embedder| embedder.id()),
            "embedder stack ready"
        );
        Ok(stack)
    }

    /// Apply MRL-style dimensionality reduction where supported.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` when `target_dim` is zero.
    pub fn with_mrl_target_dim(mut self, target_dim: usize) -> SearchResult<Self> {
        if target_dim == 0 {
            return Err(SearchError::InvalidConfig {
                field: "target_dim".to_owned(),
                value: "0".to_owned(),
                reason: "target dimension must be at least 1".to_owned(),
            });
        }

        self.fast = maybe_wrap_mrl(self.fast.clone(), target_dim)?;
        self.quality = self
            .quality
            .clone()
            .map(|embedder| maybe_wrap_mrl(embedder, target_dim))
            .transpose()?;
        self.availability = if self.quality.is_some() {
            TwoTierAvailability::Full
        } else if self.fast.is_semantic() {
            TwoTierAvailability::FastOnly
        } else {
            TwoTierAvailability::HashOnly
        };
        Ok(self)
    }

    /// Fast embedder reference.
    #[must_use]
    pub fn fast(&self) -> &dyn Embedder {
        self.fast.as_ref()
    }

    /// Fast embedder alias for API compatibility.
    #[must_use]
    pub fn fast_embedder(&self) -> &dyn Embedder {
        self.fast()
    }

    /// Cloned fast embedder handle.
    #[must_use]
    pub fn fast_arc(&self) -> Arc<dyn Embedder> {
        self.fast.clone()
    }

    /// Optional quality embedder reference.
    #[must_use]
    pub fn quality(&self) -> Option<&dyn Embedder> {
        self.quality.as_deref()
    }

    /// Optional quality embedder alias for API compatibility.
    #[must_use]
    pub fn quality_embedder(&self) -> Option<&dyn Embedder> {
        self.quality()
    }

    /// Cloned quality embedder handle.
    #[must_use]
    pub fn quality_arc(&self) -> Option<Arc<dyn Embedder>> {
        self.quality.clone()
    }

    /// Availability state.
    #[must_use]
    pub const fn availability(&self) -> TwoTierAvailability {
        self.availability
    }

    /// Generate a user-facing diagnostic report about model availability.
    ///
    /// Includes actionable suggestions for resolving degraded states:
    /// cache directory paths, manual download URLs, environment variable hints.
    #[must_use]
    pub fn diagnose(&self) -> ModelAvailabilityDiagnostic {
        let cache_dir = crate::model_cache::resolve_cache_root();
        let offline = std::env::var("FRANKENSEARCH_OFFLINE")
            .ok()
            .as_deref()
            .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));

        let fast_status = if self.fast.is_semantic() {
            ModelStatus::Ready {
                id: self.fast.id().to_owned(),
            }
        } else {
            ModelStatus::HashFallback
        };

        #[allow(clippy::option_if_let_else)] // cfg blocks in else arm prevent map_or_else
        let quality_status = if let Some(ref quality) = self.quality {
            ModelStatus::Ready {
                id: quality.id().to_owned(),
            }
        } else {
            #[cfg(feature = "fastembed")]
            {
                if offline {
                    ModelStatus::DownloadBlocked {
                        model_name: "all-MiniLM-L6-v2".to_owned(),
                        reason: "FRANKENSEARCH_OFFLINE=1 disables auto-download".to_owned(),
                    }
                } else {
                    ModelStatus::NotFound {
                        model_name: "all-MiniLM-L6-v2".to_owned(),
                        hf_repo_url:
                            "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"
                                .to_owned(),
                        searched_paths: vec![cache_dir.join("all-MiniLM-L6-v2")],
                    }
                }
            }
            #[cfg(not(feature = "fastembed"))]
            {
                ModelStatus::FeatureDisabled {
                    feature_flag: "fastembed".to_owned(),
                }
            }
        };

        let mut suggestions = Vec::new();
        if self.availability.is_degraded() {
            if offline {
                suggestions.push(
                    "Unset FRANKENSEARCH_OFFLINE to allow automatic model downloads.".to_owned(),
                );
            }
            suggestions.push(format!(
                "Set FRANKENSEARCH_MODEL_DIR to point to a pre-populated model cache (current: {}).",
                cache_dir.display()
            ));

            if matches!(self.availability, TwoTierAvailability::HashOnly) {
                #[cfg(feature = "model2vec")]
                suggestions.push(
                    "Download potion-multilingual-128M from https://huggingface.co/minishlab/potion-multilingual-128M and place in cache dir."
                        .to_owned(),
                );
                #[cfg(not(feature = "model2vec"))]
                suggestions.push(
                    "Compile with --features model2vec to enable the fast semantic tier."
                        .to_owned(),
                );
            }
            if self.quality.is_none() {
                #[cfg(feature = "fastembed")]
                suggestions.push(
                    "Download all-MiniLM-L6-v2 from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 and place in cache dir."
                        .to_owned(),
                );
                #[cfg(not(feature = "fastembed"))]
                suggestions.push(
                    "Compile with --features fastembed to enable the quality semantic tier."
                        .to_owned(),
                );
            }
            suggestions.push(
                "For air-gapped environments: run `fsfs download-models --output ./models/` on a networked machine, then copy to target."
                    .to_owned(),
            );
        }

        ModelAvailabilityDiagnostic {
            availability: self.availability,
            cache_dir,
            offline,
            fast_status,
            quality_status,
            suggestions,
        }
    }

    /// Returns a user-facing message if operating in a degraded mode, or `None` if fully available.
    #[must_use]
    pub fn degradation_message(&self) -> Option<String> {
        if !self.availability.is_degraded() {
            return None;
        }

        let diag = self.diagnose();
        let mut msg = String::new();
        if let Some(summary) = self.availability.degradation_summary() {
            msg.push_str(summary);
            msg.push('\n');
        }
        let _ = writeln!(msg, "Model cache: {}", diag.cache_dir.display());
        if diag.offline {
            msg.push_str("Offline mode: enabled (FRANKENSEARCH_OFFLINE=1)\n");
        }
        if !diag.suggestions.is_empty() {
            msg.push_str("\nTo improve search quality:\n");
            for suggestion in &diag.suggestions {
                let _ = writeln!(msg, "  - {suggestion}");
            }
        }
        Some(msg)
    }
}

/// MRL dimension reduction wrapper.
pub struct DimReduceEmbedder {
    inner: Arc<dyn Embedder>,
    target_dim: usize,
    id: String,
    model_name: String,
}

impl fmt::Debug for DimReduceEmbedder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DimReduceEmbedder")
            .field("inner", &self.inner.id())
            .field("target_dim", &self.target_dim)
            .finish_non_exhaustive()
    }
}

impl DimReduceEmbedder {
    /// Create a dimension-reduced view of an embedder.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` when requested dimension is invalid.
    pub fn new(inner: Arc<dyn Embedder>, target_dim: usize) -> SearchResult<Self> {
        if target_dim == 0 {
            return Err(SearchError::InvalidConfig {
                field: "target_dim".to_owned(),
                value: "0".to_owned(),
                reason: "target dimension must be at least 1".to_owned(),
            });
        }
        if target_dim > inner.dimension() {
            return Err(SearchError::InvalidConfig {
                field: "target_dim".to_owned(),
                value: target_dim.to_string(),
                reason: format!(
                    "target dimension cannot exceed embedder dimension {}",
                    inner.dimension()
                ),
            });
        }
        if !inner.supports_mrl() {
            return Err(SearchError::InvalidConfig {
                field: "embedder.supports_mrl".to_owned(),
                value: inner.id().to_owned(),
                reason: "embedder does not support MRL truncation".to_owned(),
            });
        }

        Ok(Self {
            id: format!("{}-mrl-{target_dim}", inner.id()),
            model_name: format!("{} (MRL {target_dim})", inner.model_name()),
            inner,
            target_dim,
        })
    }
}

impl Embedder for DimReduceEmbedder {
    fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            let full = self.inner.embed(cx, text).await?;
            self.inner.truncate_embedding(&full, self.target_dim)
        })
    }

    fn embed_batch<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<Vec<f32>>> {
        Box::pin(async move {
            let full_batch = self.inner.embed_batch(cx, texts).await?;
            full_batch
                .iter()
                .map(|embedding| self.inner.truncate_embedding(embedding, self.target_dim))
                .collect()
        })
    }

    fn dimension(&self) -> usize {
        self.target_dim
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn is_ready(&self) -> bool {
        self.inner.is_ready()
    }

    fn is_semantic(&self) -> bool {
        self.inner.is_semantic()
    }

    fn category(&self) -> frankensearch_core::traits::ModelCategory {
        self.inner.category()
    }

    fn tier(&self) -> frankensearch_core::traits::ModelTier {
        self.inner.tier()
    }

    fn supports_mrl(&self) -> bool {
        true
    }
}

fn maybe_wrap_mrl(
    embedder: Arc<dyn Embedder>,
    target_dim: usize,
) -> SearchResult<Arc<dyn Embedder>> {
    if target_dim >= embedder.dimension() || !embedder.supports_mrl() {
        return Ok(embedder);
    }
    Ok(Arc::new(DimReduceEmbedder::new(embedder, target_dim)?))
}

#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
#[derive(Debug, Clone, Copy)]
struct DownloadPolicy {
    consent: DownloadConsent,
    offline: bool,
    stderr_is_tty: bool,
}

#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
impl DownloadPolicy {
    const fn can_download(self) -> bool {
        self.consent.granted && !self.offline
    }

    fn blocked_reason(self) -> String {
        if self.offline {
            return format!("{OFFLINE_ENV}=1 disables model auto-download");
        }
        if !self.consent.granted {
            let source = self
                .consent
                .source
                .map_or_else(|| "unset".to_owned(), |s| format!("{s:?}"));
            return format!("download consent denied (source={source})");
        }
        "download policy blocked".to_owned()
    }

    #[cfg(test)]
    const fn for_tests(consent: DownloadConsent, offline: bool, stderr_is_tty: bool) -> Self {
        Self {
            consent,
            offline,
            stderr_is_tty,
        }
    }
}

#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
fn download_policy_from_environment() -> DownloadPolicy {
    let offline = std::env::var(OFFLINE_ENV)
        .ok()
        .as_deref()
        .and_then(parse_bool_flag)
        .unwrap_or(false);
    let consent = if offline {
        DownloadConsent::denied(Some(ConsentSource::Environment))
    } else {
        // Default to "allowed" unless explicitly denied via FRANKENSEARCH_ALLOW_DOWNLOAD.
        resolve_download_consent(None, None, Some(true))
    };
    DownloadPolicy {
        consent,
        offline,
        stderr_is_tty: io::stderr().is_terminal(),
    }
}

#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
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

#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed"),
    feature = "model2vec"
))]
fn maybe_lazy_fast_embedder(
    model_root: Option<&Path>,
    policy: DownloadPolicy,
) -> Option<Arc<dyn Embedder>> {
    if !policy.can_download() {
        warn!(
            model = POTION_MODEL_NAME,
            tier = "fast",
            reason = %policy.blocked_reason(),
            "auto-download disabled; fast tier falling back"
        );
        return None;
    }

    info!(
        model = POTION_MODEL_NAME,
        tier = "fast",
        "model not found locally; deferring download to first embed call"
    );
    Some(Arc::new(LazyModel2VecEmbedder::new(
        model_root.map(Path::to_path_buf),
        policy,
    )))
}

#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed"),
    not(feature = "model2vec")
))]
fn maybe_lazy_fast_embedder(
    _model_root: Option<&Path>,
    _policy: DownloadPolicy,
) -> Option<Arc<dyn Embedder>> {
    None
}

#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed"),
    feature = "fastembed"
))]
fn maybe_lazy_quality_embedder(
    model_root: Option<&Path>,
    policy: DownloadPolicy,
) -> Option<Arc<dyn Embedder>> {
    if !policy.can_download() {
        warn!(
            model = MINILM_MODEL_NAME,
            tier = "quality",
            reason = %policy.blocked_reason(),
            "auto-download disabled; quality tier unavailable"
        );
        return None;
    }

    info!(
        model = MINILM_MODEL_NAME,
        tier = "quality",
        "model not found locally; deferring download to first embed call"
    );
    Some(Arc::new(LazyFastEmbedEmbedder::new(
        model_root.map(Path::to_path_buf),
        policy,
    )))
}

#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed"),
    not(feature = "fastembed")
))]
fn maybe_lazy_quality_embedder(
    _model_root: Option<&Path>,
    _policy: DownloadPolicy,
) -> Option<Arc<dyn Embedder>> {
    None
}

#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
fn install_destination_dir(model_root: Option<&Path>, model_name: &str) -> PathBuf {
    if let Some(root) = model_root {
        if root.ends_with(model_name) {
            return root.to_path_buf();
        }
        return root.join(model_name);
    }
    ensure_model_storage_layout().join(model_name)
}

#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
async fn download_and_install_manifest(
    _cx: &Cx,
    manifest: &ModelManifest,
    destination_dir: &Path,
    policy: DownloadPolicy,
) -> SearchResult<()> {
    let start = Instant::now();
    let downloader = ModelDownloader::with_defaults();
    let mut lifecycle = ModelLifecycle::new(manifest.clone(), policy.consent);
    let staging_root = destination_dir
        .parent()
        .map_or_else(|| destination_dir.to_path_buf(), Path::to_path_buf);
    std::fs::create_dir_all(&staging_root)?;

    let reporter = Arc::new(DownloadProgressReporter::new(
        manifest.id.clone(),
        policy.stderr_is_tty,
    ));
    let reporter_for_cb = Arc::clone(&reporter);

    info!(
        model = %manifest.id,
        destination = %destination_dir.display(),
        bytes = manifest.total_size_bytes(),
        "starting automatic model download"
    );

    let staged = match downloader
        .download_model(manifest, &staging_root, &mut lifecycle, move |progress| {
            reporter_for_cb.report(progress);
        })
        .await
    {
        Ok(staged) => staged,
        Err(error) => {
            reporter.finish_failed(start.elapsed(), &error);
            warn!(
                model = %manifest.id,
                duration_ms = start.elapsed().as_millis(),
                error = %error,
                "automatic model download failed"
            );
            return Err(error);
        }
    };

    match manifest.promote_verified_installation(&staged, destination_dir) {
        Ok(backup) => {
            let backup_path = backup
                .as_ref()
                .map_or_else(|| "none".to_owned(), |path| path.display().to_string());
            reporter.finish_ok(start.elapsed(), manifest.total_size_bytes());
            info!(
                model = %manifest.id,
                destination = %destination_dir.display(),
                backup = %backup_path,
                duration_ms = start.elapsed().as_millis(),
                bytes = manifest.total_size_bytes(),
                "automatic model download completed"
            );
            Ok(())
        }
        Err(error) => {
            reporter.finish_failed(start.elapsed(), &error);
            warn!(
                model = %manifest.id,
                destination = %destination_dir.display(),
                duration_ms = start.elapsed().as_millis(),
                error = %error,
                "automatic model promotion failed"
            );
            Err(error)
        }
    }
}

#[cfg(all(feature = "download", feature = "model2vec"))]
struct LazyModel2VecEmbedder {
    model_root: Option<PathBuf>,
    policy: DownloadPolicy,
    inner: OnceCell<Arc<dyn Embedder>>,
}

#[cfg(all(feature = "download", feature = "model2vec"))]
impl LazyModel2VecEmbedder {
    const fn new(model_root: Option<PathBuf>, policy: DownloadPolicy) -> Self {
        Self {
            model_root,
            policy,
            inner: OnceCell::new(),
        }
    }

    async fn ensure_loaded(&self, cx: &Cx) -> SearchResult<Arc<dyn Embedder>> {
        let embedder = self
            .inner
            .get_or_try_init(|| async { self.initialize(cx).await })
            .await?;
        Ok(Arc::clone(embedder))
    }

    async fn initialize(&self, cx: &Cx) -> SearchResult<Arc<dyn Embedder>> {
        if let Some(existing) = detect_fast_embedder(self.model_root.as_deref()) {
            return Ok(existing);
        }
        if !self.policy.can_download() {
            return Err(SearchError::EmbedderUnavailable {
                model: POTION_MODEL_NAME.to_owned(),
                reason: self.policy.blocked_reason(),
            });
        }

        let manifest = ModelManifest::potion_128m();
        let destination = install_destination_dir(self.model_root.as_deref(), POTION_MODEL_NAME);
        download_and_install_manifest(cx, &manifest, &destination, self.policy).await?;
        Model2VecEmbedder::load_with_name(&destination, POTION_MODEL_NAME)
            .map(|embedder| Arc::new(embedder) as Arc<dyn Embedder>)
    }
}

#[cfg(all(feature = "download", feature = "model2vec"))]
impl Embedder for LazyModel2VecEmbedder {
    fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            let embedder = self.ensure_loaded(cx).await?;
            embedder.embed(cx, text).await
        })
    }

    fn embed_batch<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<Vec<f32>>> {
        Box::pin(async move {
            let embedder = self.ensure_loaded(cx).await?;
            embedder.embed_batch(cx, texts).await
        })
    }

    fn dimension(&self) -> usize {
        POTION_DIMENSION
    }

    fn id(&self) -> &str {
        POTION_MODEL_NAME
    }

    fn model_name(&self) -> &str {
        POTION_MODEL_NAME
    }

    fn is_ready(&self) -> bool {
        self.inner.get().is_some_and(|embedder| embedder.is_ready())
    }

    fn is_semantic(&self) -> bool {
        true
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::StaticEmbedder
    }

    fn tier(&self) -> ModelTier {
        ModelTier::Fast
    }

    fn supports_mrl(&self) -> bool {
        true
    }
}

#[cfg(all(feature = "download", feature = "fastembed"))]
struct LazyFastEmbedEmbedder {
    model_root: Option<PathBuf>,
    policy: DownloadPolicy,
    inner: OnceCell<Arc<dyn Embedder>>,
}

#[cfg(all(feature = "download", feature = "fastembed"))]
impl LazyFastEmbedEmbedder {
    const fn new(model_root: Option<PathBuf>, policy: DownloadPolicy) -> Self {
        Self {
            model_root,
            policy,
            inner: OnceCell::new(),
        }
    }

    async fn ensure_loaded(&self, cx: &Cx) -> SearchResult<Arc<dyn Embedder>> {
        let embedder = self
            .inner
            .get_or_try_init(|| async { self.initialize(cx).await })
            .await?;
        Ok(Arc::clone(embedder))
    }

    async fn initialize(&self, cx: &Cx) -> SearchResult<Arc<dyn Embedder>> {
        if let Some(existing) = detect_quality_embedder(self.model_root.as_deref()) {
            return Ok(existing);
        }
        if !self.policy.can_download() {
            return Err(SearchError::EmbedderUnavailable {
                model: MINILM_MODEL_NAME.to_owned(),
                reason: self.policy.blocked_reason(),
            });
        }

        let manifest = ModelManifest::minilm_v2();
        let destination = install_destination_dir(self.model_root.as_deref(), MINILM_MODEL_NAME);
        download_and_install_manifest(cx, &manifest, &destination, self.policy).await?;
        FastEmbedEmbedder::load_with_name(&destination, MINILM_MODEL_NAME)
            .map(|embedder| Arc::new(embedder) as Arc<dyn Embedder>)
    }
}

#[cfg(all(feature = "download", feature = "fastembed"))]
impl Embedder for LazyFastEmbedEmbedder {
    fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            let embedder = self.ensure_loaded(cx).await?;
            embedder.embed(cx, text).await
        })
    }

    fn embed_batch<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<Vec<f32>>> {
        Box::pin(async move {
            let embedder = self.ensure_loaded(cx).await?;
            embedder.embed_batch(cx, texts).await
        })
    }

    fn dimension(&self) -> usize {
        MINILM_DIMENSION
    }

    fn id(&self) -> &str {
        MINILM_MODEL_NAME
    }

    fn model_name(&self) -> &str {
        MINILM_MODEL_NAME
    }

    fn is_ready(&self) -> bool {
        self.inner.get().is_some_and(|embedder| embedder.is_ready())
    }

    fn is_semantic(&self) -> bool {
        true
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::TransformerEmbedder
    }

    fn tier(&self) -> ModelTier {
        ModelTier::Quality
    }
}

#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
#[derive(Debug)]
struct DownloadProgressReporter {
    model_id: String,
    stderr_is_tty: bool,
    last_bucket: AtomicU8,
}

#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
impl DownloadProgressReporter {
    const fn new(model_id: String, stderr_is_tty: bool) -> Self {
        Self {
            model_id,
            stderr_is_tty,
            last_bucket: AtomicU8::new(0),
        }
    }

    fn report(&self, progress: &DownloadProgress) {
        let progress_x100 = progress_percent_x100(progress);
        if self.stderr_is_tty {
            self.report_tty(progress, progress_x100);
        } else {
            self.report_non_tty(progress, progress_x100);
        }
    }

    fn finish_ok(&self, elapsed: std::time::Duration, total_bytes: u64) {
        if self.stderr_is_tty {
            eprintln!(
                "\rDownloaded {} in {:.1}s ({})",
                self.model_id,
                elapsed.as_secs_f64(),
                format_bytes(total_bytes),
            );
        } else {
            eprintln!(
                "Downloaded {} in {:.1}s ({})",
                self.model_id,
                elapsed.as_secs_f64(),
                format_bytes(total_bytes),
            );
        }
    }

    fn finish_failed(&self, elapsed: std::time::Duration, error: &SearchError) {
        if self.stderr_is_tty {
            eprintln!(
                "\rDownload failed for {} after {:.1}s: {}",
                self.model_id,
                elapsed.as_secs_f64(),
                error
            );
        } else {
            eprintln!(
                "Download failed for {} after {:.1}s: {}",
                self.model_id,
                elapsed.as_secs_f64(),
                error
            );
        }
    }

    fn report_tty(&self, progress: &DownloadProgress, progress_x100: u64) {
        let pct_whole = progress_x100 / 100;
        let pct_frac = progress_x100 % 100;
        let bar = render_progress_bar(progress_x100);
        let total = progress
            .total_bytes
            .map_or_else(|| "?".to_owned(), format_bytes);
        eprint!(
            "\rDownloading {} [{}] {:>3}.{pct_frac:02}% {}/{} {} ETA {} ({}/{})",
            self.model_id,
            bar,
            pct_whole,
            format_bytes(progress.bytes_downloaded),
            total,
            format_speed(progress.speed_bytes_per_sec),
            format_eta(progress.eta_seconds),
            progress.files_completed + 1,
            progress.files_total.max(1),
        );
        let _ = io::stderr().flush();
    }

    fn report_non_tty(&self, progress: &DownloadProgress, progress_x100: u64) {
        let bucket = u8::try_from((progress_x100 / 1000).min(10)).unwrap_or(10);
        let previous = self.last_bucket.load(Ordering::Relaxed);
        if bucket <= previous {
            return;
        }
        if self
            .last_bucket
            .compare_exchange(previous, bucket, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            eprintln!(
                "Downloading {}... {}% ({}/{} {} ETA {})",
                self.model_id,
                bucket.saturating_mul(10),
                format_bytes(progress.bytes_downloaded),
                progress
                    .total_bytes
                    .map_or_else(|| "?".to_owned(), format_bytes),
                format_speed(progress.speed_bytes_per_sec),
                format_eta(progress.eta_seconds),
            );
        }
    }
}

#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
fn progress_percent_x100(progress: &DownloadProgress) -> u64 {
    let files_total = u64::try_from(progress.files_total).unwrap_or(1).max(1);
    let files_completed = u64::try_from(progress.files_completed)
        .unwrap_or(files_total)
        .min(files_total);
    let current_file_percent_x100 = progress
        .total_bytes
        .filter(|&total| total > 0)
        .map_or(0, |total| {
            progress.bytes_downloaded.min(total).saturating_mul(10_000) / total
        });
    files_completed
        .saturating_mul(10_000)
        .saturating_add(current_file_percent_x100)
        / files_total
}

#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
fn render_progress_bar(progress_x100: u64) -> String {
    let width = u64::try_from(PROGRESS_BAR_WIDTH).unwrap_or(30);
    let filled = usize::try_from(progress_x100.saturating_mul(width) / 10_000)
        .unwrap_or(PROGRESS_BAR_WIDTH)
        .min(PROGRESS_BAR_WIDTH);
    let mut bar = String::with_capacity(PROGRESS_BAR_WIDTH);
    bar.push_str(&"=".repeat(filled));
    bar.push_str(&" ".repeat(PROGRESS_BAR_WIDTH.saturating_sub(filled)));
    bar
}

#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
fn format_eta(seconds: Option<f64>) -> String {
    match seconds {
        Some(value) if value.is_finite() && value >= 0.0 => format!("{value:.1}s"),
        _ => "?".to_owned(),
    }
}

#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
fn format_speed(bytes_per_sec: f64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;
    if !bytes_per_sec.is_finite() || bytes_per_sec <= 0.0 {
        return "0 B/s".to_owned();
    }
    if bytes_per_sec >= GB {
        format!("{:.1} GB/s", bytes_per_sec / GB)
    } else if bytes_per_sec >= MB {
        format!("{:.1} MB/s", bytes_per_sec / MB)
    } else if bytes_per_sec >= KB {
        format!("{:.1} KB/s", bytes_per_sec / KB)
    } else {
        format!("{bytes_per_sec:.0} B/s")
    }
}

#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * 1024;
    const GB: u64 = 1024 * 1024 * 1024;

    if bytes >= GB {
        let whole = bytes / GB;
        let frac = bytes % GB * 10 / GB;
        format!("{whole}.{frac} GB")
    } else if bytes >= MB {
        let whole = bytes / MB;
        let frac = bytes % MB * 10 / MB;
        format!("{whole}.{frac} MB")
    } else if bytes >= KB {
        let whole = bytes / KB;
        let frac = bytes % KB * 10 / KB;
        format!("{whole}.{frac} KB")
    } else {
        format!("{bytes} B")
    }
}

#[cfg(feature = "model2vec")]
fn detect_fast_embedder(model_root: Option<&Path>) -> Option<Arc<dyn Embedder>> {
    let manifest = ModelManifest::potion_128m();
    let discovered = find_model2vec_model_dir(POTION_MODEL_NAME, POTION_HF_ID);
    let candidates = candidate_directories(model_root, POTION_MODEL_NAME, discovered.as_deref());
    let checked_paths: Vec<String> = candidates
        .iter()
        .map(|path| path.display().to_string())
        .collect();

    for candidate in candidates {
        if !manifest_files_exist(&manifest, &candidate) {
            continue;
        }
        if let Err(error) = crate::model_manifest::verify_dir_cached(&manifest, &candidate) {
            warn!(
                model = POTION_MODEL_NAME,
                path = %candidate.display(),
                error = %error,
                "model2vec manifest verification failed, skipping candidate"
            );
            continue;
        }

        match Model2VecEmbedder::load_with_name(&candidate, POTION_MODEL_NAME) {
            Ok(embedder) => {
                info!(
                    model = POTION_MODEL_NAME,
                    tier = "fast",
                    path = %candidate.display(),
                    dimension = embedder.dimension(),
                    "embedder detected"
                );
                return Some(Arc::new(embedder));
            }
            Err(error) => {
                warn!(
                    model = POTION_MODEL_NAME,
                    tier = "fast",
                    path = %candidate.display(),
                    error = %error,
                    "embedder unavailable"
                );
            }
        }
    }

    warn!(
        model = POTION_MODEL_NAME,
        tier = "fast",
        checked_paths = ?checked_paths,
        "embedder unavailable"
    );
    None
}

#[cfg(not(feature = "model2vec"))]
fn detect_fast_embedder(_model_root: Option<&Path>) -> Option<Arc<dyn Embedder>> {
    None
}

#[cfg(feature = "fastembed")]
fn detect_quality_embedder(model_root: Option<&Path>) -> Option<Arc<dyn Embedder>> {
    let manifest = ModelManifest::minilm_v2();
    let discovered = find_fastembed_model_dir(MINILM_MODEL_NAME, MINILM_HF_ID);
    let candidates = candidate_directories(model_root, MINILM_MODEL_NAME, discovered.as_deref());
    let checked_paths: Vec<String> = candidates
        .iter()
        .map(|path| path.display().to_string())
        .collect();

    for candidate in candidates {
        if !manifest_files_exist(&manifest, &candidate) {
            continue;
        }
        if let Err(error) = crate::model_manifest::verify_dir_cached(&manifest, &candidate) {
            warn!(
                model = MINILM_MODEL_NAME,
                path = %candidate.display(),
                error = %error,
                "quality manifest verification failed, skipping candidate"
            );
            continue;
        }

        match FastEmbedEmbedder::load_with_name(&candidate, MINILM_MODEL_NAME) {
            Ok(embedder) => {
                info!(
                    model = MINILM_MODEL_NAME,
                    tier = "quality",
                    path = %candidate.display(),
                    dimension = embedder.dimension(),
                    "embedder detected"
                );
                return Some(Arc::new(embedder));
            }
            Err(error) => {
                warn!(
                    model = MINILM_MODEL_NAME,
                    tier = "quality",
                    path = %candidate.display(),
                    error = %error,
                    "embedder unavailable"
                );
            }
        }
    }

    warn!(
        model = MINILM_MODEL_NAME,
        tier = "quality",
        checked_paths = ?checked_paths,
        "embedder unavailable"
    );
    None
}

#[cfg(not(feature = "fastembed"))]
fn detect_quality_embedder(_model_root: Option<&Path>) -> Option<Arc<dyn Embedder>> {
    None
}

#[cfg(feature = "hash")]
#[allow(clippy::unnecessary_wraps)]
fn hash_fallback_embedder() -> Option<Arc<dyn Embedder>> {
    Some(Arc::new(HashEmbedder::default_256()))
}

#[cfg(not(feature = "hash"))]
fn hash_fallback_embedder() -> Option<Arc<dyn Embedder>> {
    None
}

#[cfg(any(feature = "model2vec", feature = "fastembed"))]
fn manifest_files_exist(manifest: &ModelManifest, model_dir: &Path) -> bool {
    manifest
        .files
        .iter()
        .all(|file| model_dir.join(&file.name).is_file())
}

#[cfg(any(feature = "model2vec", feature = "fastembed"))]
fn candidate_directories(
    model_root: Option<&Path>,
    model_name: &str,
    discovered: Option<&Path>,
) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(root) = model_root {
        paths.push(root.join(model_name));
        paths.push(root.to_path_buf());
    }
    if let Some(path) = discovered {
        paths.push(path.to_path_buf());
    }

    let mut seen = BTreeSet::new();
    paths
        .into_iter()
        .filter(|path| seen.insert(path.display().to_string()))
        .collect()
}

#[cfg(test)]
mod tests {
    #[cfg(all(feature = "model2vec", feature = "hash"))]
    use std::fs;

    #[cfg(all(feature = "download", feature = "model2vec"))]
    use asupersync::test_utils::run_test_with_cx;

    use super::*;
    use frankensearch_core::traits::ModelCategory;

    #[cfg(feature = "hash")]
    #[test]
    fn auto_detect_hash_only_when_no_models_present() {
        let temp = tempfile::tempdir().unwrap();
        #[cfg(all(
            feature = "download",
            any(feature = "model2vec", feature = "fastembed")
        ))]
        let stack = EmbedderStack::auto_detect_with_policy(
            Some(temp.path()),
            DownloadPolicy::for_tests(
                DownloadConsent::denied(Some(ConsentSource::Programmatic)),
                false,
                false,
            ),
        )
        .unwrap();
        #[cfg(not(all(
            feature = "download",
            any(feature = "model2vec", feature = "fastembed")
        )))]
        let stack = EmbedderStack::auto_detect_with(Some(temp.path())).unwrap();
        assert_eq!(stack.availability(), TwoTierAvailability::HashOnly);
        assert_eq!(stack.fast().category(), ModelCategory::HashEmbedder);
        assert!(stack.quality().is_none());
    }

    #[cfg(all(feature = "model2vec", feature = "hash"))]
    #[test]
    fn auto_detect_fast_only_when_model2vec_is_available() {
        let temp = tempfile::tempdir().unwrap();
        let model_dir = temp.path().join(POTION_MODEL_NAME);
        fs::create_dir_all(&model_dir).unwrap();
        create_test_model2vec_layout(&model_dir, 16, 8);

        #[cfg(all(
            feature = "download",
            any(feature = "model2vec", feature = "fastembed")
        ))]
        let stack = EmbedderStack::auto_detect_with_policy(
            Some(temp.path()),
            DownloadPolicy::for_tests(
                DownloadConsent::denied(Some(ConsentSource::Programmatic)),
                false,
                false,
            ),
        )
        .unwrap();
        #[cfg(not(all(
            feature = "download",
            any(feature = "model2vec", feature = "fastembed")
        )))]
        let stack = EmbedderStack::auto_detect_with(Some(temp.path())).unwrap();
        assert_eq!(stack.availability(), TwoTierAvailability::FastOnly);
        assert_eq!(stack.fast().id(), POTION_MODEL_NAME);
    }

    #[cfg(all(feature = "model2vec", feature = "hash"))]
    #[test]
    fn corrupted_model2vec_falls_back_to_hash() {
        let temp = tempfile::tempdir().unwrap();
        let model_dir = temp.path().join(POTION_MODEL_NAME);
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("tokenizer.json"), "{}").unwrap();
        fs::write(model_dir.join("model.safetensors"), b"not-safetensors").unwrap();

        #[cfg(all(
            feature = "download",
            any(feature = "model2vec", feature = "fastembed")
        ))]
        let stack = EmbedderStack::auto_detect_with_policy(
            Some(temp.path()),
            DownloadPolicy::for_tests(
                DownloadConsent::denied(Some(ConsentSource::Programmatic)),
                false,
                false,
            ),
        )
        .unwrap();
        #[cfg(not(all(
            feature = "download",
            any(feature = "model2vec", feature = "fastembed")
        )))]
        let stack = EmbedderStack::auto_detect_with(Some(temp.path())).unwrap();
        assert_eq!(stack.availability(), TwoTierAvailability::HashOnly);
        assert_eq!(stack.fast().category(), ModelCategory::HashEmbedder);
    }

    #[cfg(all(feature = "download", feature = "model2vec", feature = "hash"))]
    #[test]
    fn auto_detect_prefers_lazy_fast_embedder_when_download_enabled() {
        let temp = tempfile::tempdir().unwrap();
        let stack = EmbedderStack::auto_detect_with_policy(
            Some(temp.path()),
            DownloadPolicy::for_tests(
                DownloadConsent::granted(ConsentSource::Programmatic),
                false,
                false,
            ),
        )
        .unwrap();
        assert_eq!(stack.fast().id(), POTION_MODEL_NAME);
        assert_eq!(stack.fast().category(), ModelCategory::StaticEmbedder);
    }

    #[cfg(all(feature = "download", feature = "model2vec"))]
    #[test]
    fn lazy_model2vec_returns_unavailable_when_download_is_denied() {
        let temp = tempfile::tempdir().unwrap();
        let lazy = LazyModel2VecEmbedder::new(
            Some(temp.path().to_path_buf()),
            DownloadPolicy::for_tests(
                DownloadConsent::denied(Some(ConsentSource::Programmatic)),
                false,
                false,
            ),
        );

        run_test_with_cx(|cx| async move {
            let err = lazy
                .embed(&cx, "hello world")
                .await
                .expect_err("download-denied lazy model should error");
            assert!(matches!(err, SearchError::EmbedderUnavailable { .. }));
        });
    }

    #[cfg(all(
        feature = "download",
        any(feature = "model2vec", feature = "fastembed")
    ))]
    #[test]
    fn parse_bool_flag_supports_common_values() {
        assert_eq!(parse_bool_flag("1"), Some(true));
        assert_eq!(parse_bool_flag("true"), Some(true));
        assert_eq!(parse_bool_flag("YES"), Some(true));
        assert_eq!(parse_bool_flag("0"), Some(false));
        assert_eq!(parse_bool_flag("false"), Some(false));
        assert_eq!(parse_bool_flag("off"), Some(false));
        assert_eq!(parse_bool_flag("invalid"), None);
    }

    #[cfg(all(
        feature = "download",
        any(feature = "model2vec", feature = "fastembed")
    ))]
    #[test]
    fn progress_percent_accounts_for_current_file_fraction() {
        let progress = DownloadProgress {
            file_name: "model.onnx".to_owned(),
            bytes_downloaded: 50,
            total_bytes: Some(100),
            files_completed: 1,
            files_total: 4,
            speed_bytes_per_sec: 1.0,
            eta_seconds: Some(2.0),
        };
        // 1 full file + 50% of one file over 4 total files = 37.5%
        assert_eq!(progress_percent_x100(&progress), 3_750);
    }

    #[cfg(all(feature = "model2vec", feature = "hash"))]
    fn create_test_model2vec_layout(dir: &Path, vocab_size: usize, dimensions: usize) {
        let tokenizer_json = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [{
                "id": 0,
                "content": "[UNK]",
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true
            }],
            "normalizer": { "type": "Lowercase" },
            "pre_tokenizer": { "type": "Whitespace" },
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": create_test_vocab(vocab_size),
                "unk_token": "[UNK]"
            }
        });
        fs::write(
            dir.join("tokenizer.json"),
            serde_json::to_string_pretty(&tokenizer_json).unwrap(),
        )
        .unwrap();
        create_test_safetensors(dir, vocab_size, dimensions);
    }

    #[cfg(all(feature = "model2vec", feature = "hash"))]
    fn create_test_vocab(vocab_size: usize) -> serde_json::Value {
        let mut vocab = serde_json::Map::new();
        vocab.insert("[UNK]".to_owned(), serde_json::Value::from(0));
        for idx in 1..vocab_size {
            vocab.insert(format!("token{idx}"), serde_json::Value::from(idx));
        }
        serde_json::Value::Object(vocab)
    }

    #[cfg(feature = "hash")]
    #[test]
    fn from_parts_hash_only_availability() {
        let hash = Arc::new(crate::hash_embedder::HashEmbedder::default_256());
        let stack = EmbedderStack::from_parts(hash, None);
        assert_eq!(stack.availability(), TwoTierAvailability::HashOnly);
        assert!(stack.quality().is_none());
        assert!(stack.quality_arc().is_none());
        assert_eq!(stack.fast().category(), ModelCategory::HashEmbedder);
        assert_eq!(stack.fast_embedder().id(), stack.fast().id());
    }

    #[cfg(feature = "hash")]
    #[test]
    fn from_parts_with_quality_is_full() {
        let fast: Arc<dyn Embedder> = Arc::new(crate::hash_embedder::HashEmbedder::default_256());
        let quality: Arc<dyn Embedder> =
            Arc::new(crate::hash_embedder::HashEmbedder::default_384());
        let stack = EmbedderStack::from_parts(fast, Some(quality));
        assert_eq!(stack.availability(), TwoTierAvailability::Full);
        assert!(stack.quality().is_some());
        assert!(stack.quality_arc().is_some());
        assert_eq!(
            stack.quality_embedder().unwrap().id(),
            stack.quality().unwrap().id()
        );
    }

    #[cfg(feature = "hash")]
    #[test]
    fn dim_reduce_rejects_zero_target_dim() {
        let inner: Arc<dyn Embedder> = Arc::new(crate::hash_embedder::HashEmbedder::default_256());
        let err = DimReduceEmbedder::new(inner, 0).expect_err("should reject target_dim=0");
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[cfg(feature = "hash")]
    #[test]
    fn dim_reduce_rejects_target_exceeding_inner_dim() {
        let inner: Arc<dyn Embedder> = Arc::new(crate::hash_embedder::HashEmbedder::default_256());
        let err =
            DimReduceEmbedder::new(inner, 512).expect_err("should reject target_dim > inner dim");
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[cfg(feature = "hash")]
    #[test]
    fn dim_reduce_rejects_non_mrl_embedder() {
        let inner: Arc<dyn Embedder> = Arc::new(crate::hash_embedder::HashEmbedder::default_256());
        assert!(!inner.supports_mrl());
        let err = DimReduceEmbedder::new(inner, 64).expect_err("should reject non-MRL embedder");
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[cfg(feature = "hash")]
    #[test]
    fn with_mrl_target_dim_zero_is_rejected() {
        let fast: Arc<dyn Embedder> = Arc::new(crate::hash_embedder::HashEmbedder::default_256());
        let stack = EmbedderStack::from_parts(fast, None);
        let err = stack
            .with_mrl_target_dim(0)
            .expect_err("should reject target_dim=0");
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[cfg(feature = "hash")]
    #[test]
    fn with_mrl_passthrough_when_non_mrl() {
        let fast: Arc<dyn Embedder> = Arc::new(crate::hash_embedder::HashEmbedder::default_256());
        let stack = EmbedderStack::from_parts(fast, None);
        // hash embedder doesn't support MRL, so with_mrl_target_dim should
        // pass through without wrapping (target_dim < dimension but !supports_mrl)
        let stack = stack.with_mrl_target_dim(64).unwrap();
        assert_eq!(stack.availability(), TwoTierAvailability::HashOnly);
        // dimension should be unchanged since wrapping was skipped
        assert_eq!(stack.fast().dimension(), 256);
    }

    #[cfg(feature = "hash")]
    #[test]
    fn embedder_stack_debug_format() {
        let fast: Arc<dyn Embedder> = Arc::new(crate::hash_embedder::HashEmbedder::default_256());
        let stack = EmbedderStack::from_parts(fast, None);
        let debug = format!("{stack:?}");
        assert!(debug.contains("EmbedderStack"));
        assert!(debug.contains("HashOnly"));
    }

    // ─── Offline Fallback Diagnostic Tests ───────────────────────────────────

    #[cfg(feature = "hash")]
    #[test]
    fn availability_is_degraded_for_hash_only() {
        assert!(TwoTierAvailability::HashOnly.is_degraded());
        assert!(TwoTierAvailability::FastOnly.is_degraded());
        assert!(!TwoTierAvailability::Full.is_degraded());
    }

    #[cfg(feature = "hash")]
    #[test]
    fn availability_display_format() {
        let full = format!("{}", TwoTierAvailability::Full);
        assert!(full.contains("full"));
        let fast = format!("{}", TwoTierAvailability::FastOnly);
        assert!(fast.contains("degraded"));
        let hash = format!("{}", TwoTierAvailability::HashOnly);
        assert!(hash.contains("minimal"));
    }

    #[cfg(feature = "hash")]
    #[test]
    fn degradation_summary_none_for_full() {
        assert!(TwoTierAvailability::Full.degradation_summary().is_none());
    }

    #[cfg(feature = "hash")]
    #[test]
    fn degradation_summary_present_for_degraded() {
        assert!(
            TwoTierAvailability::FastOnly
                .degradation_summary()
                .unwrap()
                .contains("Quality model")
        );
        assert!(
            TwoTierAvailability::HashOnly
                .degradation_summary()
                .unwrap()
                .contains("semantic")
        );
    }

    #[cfg(feature = "hash")]
    #[test]
    fn diagnose_hash_only_provides_suggestions() {
        let fast: Arc<dyn Embedder> = Arc::new(crate::hash_embedder::HashEmbedder::default_256());
        let stack = EmbedderStack::from_parts(fast, None);
        let diag = stack.diagnose();
        assert_eq!(diag.availability, TwoTierAvailability::HashOnly);
        assert!(matches!(diag.fast_status, ModelStatus::HashFallback));
        assert!(!diag.suggestions.is_empty());
        assert!(
            diag.suggestions
                .iter()
                .any(|s| s.contains("FRANKENSEARCH_MODEL_DIR"))
        );
        assert!(diag.suggestions.iter().any(|s| s.contains("air-gapped")));
    }

    #[cfg(feature = "hash")]
    #[test]
    fn diagnose_full_has_no_suggestions() {
        let fast: Arc<dyn Embedder> = Arc::new(crate::hash_embedder::HashEmbedder::default_256());
        let quality: Arc<dyn Embedder> =
            Arc::new(crate::hash_embedder::HashEmbedder::default_384());
        let stack = EmbedderStack::from_parts(fast, Some(quality));
        let diag = stack.diagnose();
        assert_eq!(diag.availability, TwoTierAvailability::Full);
        assert!(diag.suggestions.is_empty());
    }

    #[cfg(feature = "hash")]
    #[test]
    fn degradation_message_none_when_full() {
        let fast: Arc<dyn Embedder> = Arc::new(crate::hash_embedder::HashEmbedder::default_256());
        let quality: Arc<dyn Embedder> =
            Arc::new(crate::hash_embedder::HashEmbedder::default_384());
        let stack = EmbedderStack::from_parts(fast, Some(quality));
        assert!(stack.degradation_message().is_none());
    }

    #[cfg(feature = "hash")]
    #[test]
    fn degradation_message_present_when_degraded() {
        let fast: Arc<dyn Embedder> = Arc::new(crate::hash_embedder::HashEmbedder::default_256());
        let stack = EmbedderStack::from_parts(fast, None);
        let msg = stack
            .degradation_message()
            .expect("should be present for hash-only");
        assert!(msg.contains("Model cache:"));
        assert!(msg.contains("FRANKENSEARCH_MODEL_DIR"));
    }

    #[cfg(feature = "hash")]
    #[test]
    fn model_status_display_ready() {
        let status = ModelStatus::Ready {
            id: "test-model".to_owned(),
        };
        let display = format!("{status}");
        assert!(display.contains("ready"));
        assert!(display.contains("test-model"));
    }

    #[cfg(feature = "hash")]
    #[test]
    fn model_status_display_not_found() {
        let status = ModelStatus::NotFound {
            model_name: "test-model".to_owned(),
            hf_repo_url: "https://huggingface.co/test/model".to_owned(),
            searched_paths: vec![],
        };
        let display = format!("{status}");
        assert!(display.contains("NOT FOUND"));
        assert!(display.contains("https://huggingface.co/test/model"));
    }

    #[cfg(feature = "hash")]
    #[test]
    fn model_status_display_download_blocked() {
        let status = ModelStatus::DownloadBlocked {
            model_name: "test-model".to_owned(),
            reason: "offline mode".to_owned(),
        };
        let display = format!("{status}");
        assert!(display.contains("BLOCKED"));
        assert!(display.contains("offline mode"));
    }

    #[cfg(feature = "hash")]
    #[test]
    fn model_status_display_feature_disabled() {
        let status = ModelStatus::FeatureDisabled {
            feature_flag: "fastembed".to_owned(),
        };
        let display = format!("{status}");
        assert!(display.contains("DISABLED"));
        assert!(display.contains("fastembed"));
    }

    #[cfg(feature = "hash")]
    #[test]
    fn diagnostic_display_includes_all_sections() {
        let diag = ModelAvailabilityDiagnostic {
            availability: TwoTierAvailability::HashOnly,
            cache_dir: std::path::PathBuf::from("/tmp/test-cache"),
            offline: false,
            fast_status: ModelStatus::HashFallback,
            quality_status: ModelStatus::FeatureDisabled {
                feature_flag: "fastembed".to_owned(),
            },
            suggestions: vec!["Fix something".to_owned()],
        };
        let display = format!("{diag}");
        assert!(display.contains("minimal"));
        assert!(display.contains("/tmp/test-cache"));
        assert!(display.contains("hash fallback"));
        assert!(display.contains("Fix something"));
    }

    #[cfg(feature = "hash")]
    #[test]
    fn diagnostic_display_offline_mode_indicator() {
        let diag = ModelAvailabilityDiagnostic {
            availability: TwoTierAvailability::HashOnly,
            cache_dir: std::path::PathBuf::from("/tmp/test-cache"),
            offline: true,
            fast_status: ModelStatus::HashFallback,
            quality_status: ModelStatus::DownloadBlocked {
                model_name: "test".to_owned(),
                reason: "offline".to_owned(),
            },
            suggestions: vec![],
        };
        let display = format!("{diag}");
        assert!(display.contains("OFFLINE"));
    }

    #[cfg(all(feature = "model2vec", feature = "hash"))]
    fn create_test_safetensors(dir: &Path, vocab_size: usize, dimensions: usize) {
        use std::collections::HashMap;

        let mut data = Vec::with_capacity(vocab_size * dimensions * 4);
        for row in 0..vocab_size {
            for col in 0..dimensions {
                #[allow(clippy::cast_precision_loss)]
                let value = (row as f32).mul_add(0.01, (col as f32) * 0.001);
                data.extend_from_slice(&value.to_le_bytes());
            }
        }

        let mut tensors = HashMap::new();
        tensors.insert(
            "embeddings".to_owned(),
            safetensors::tensor::TensorView::new(
                safetensors::Dtype::F32,
                vec![vocab_size, dimensions],
                &data,
            )
            .unwrap(),
        );
        let encoded = safetensors::tensor::serialize(&tensors, &None).unwrap();
        fs::write(dir.join("model.safetensors"), encoded).unwrap();
    }
}
